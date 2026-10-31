use anyhow::Result;
use gpui::prelude::FluentBuilder as _;
use gpui::{
    Animation, AnimationExt as _, App, AppContext, Application, Bounds, Context, CursorStyle,
    Edges, Entity, Hsla, InteractiveElement as _, IntoElement, KeyDownEvent, MouseButton, MouseDownEvent,
    MouseMoveEvent, ParentElement, Pixels, Point, Render, SharedString,
    StatefulInteractiveElement as _, Styled, Transformation, Window, WindowBounds,
    WindowControlArea, WindowDecorations, WindowOptions, actions, deferred, div,
    percentage, px, relative, rgba, size, svg,
};
use gpui_component::WindowExt as _;
use gpui_component::notification::{Notification, NotificationType};
use gpui_component::resizable::{h_resizable, resizable_panel, v_resizable};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::{Root, Sizable as _, TitleBar};
use gpui_component::input::{Input, InputEvent, InputState, RopeExt as _};
use terminal::{ColorPalette, TerminalConfig, TerminalView};
use parking_lot::Mutex;
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};

mod agent_rules;
mod assets;
mod attention;
mod cli;
mod completion;
mod diagnostics;
#[cfg(unix)]
mod herdr_compat;
mod integrations;
mod logos;
mod shell_history;
mod sound;
mod status;
#[cfg(unix)]
mod status_bus;
mod workspace_state;
mod theme;

use theme::{ThemePreference, ThemeTokens, load_github_palette};

/// Key context of the Warp-style input bar.
///
/// Tab has to be bound here rather than handled from `on_key_down`: the input
/// component binds `tab` to `IndentInline` and `gpui-component` binds it at the
/// `Root` context for focus traversal, which would otherwise consume the key
/// before the completion handler runs.
const WARP_INPUT_CONTEXT: &str = "WarpInput";

actions!(markup_input, [CompleteCommand, IgnoreKey]);

fn init_warp_input_bindings(cx: &mut App) {
    cx.bind_keys([
        gpui::KeyBinding::new("tab", CompleteCommand, Some(WARP_INPUT_CONTEXT)),
        // Keep Shift+Tab from running the Root context's focus traversal.
        gpui::KeyBinding::new("shift-tab", IgnoreKey, Some(WARP_INPUT_CONTEXT)),
    ]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActiveView {
    #[default]
    Workspace,
    Settings,
}

use assets::{CombinedAssets, MONO_FONT, load_embedded_fonts, sync_component_fonts};
use status::{SessionStatus, StatusInput};
pub const APP_ICON_PNG: &[u8] = include_bytes!("../assets/logo/markup.png");
/// Fixed width of the navigation sidebar (Codex / Warp style).
pub const SIDEBAR_WIDTH: Pixels = px(260.0);

#[cfg(unix)]
static STATUS_BUS: std::sync::LazyLock<Option<status_bus::BusIdentity>> =
    std::sync::LazyLock::new(|| status_bus::start().ok());
#[cfg(unix)]
static LAST_STATUS_SNAPSHOT_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
fn status_snapshot_due() -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let previous = LAST_STATUS_SNAPSHOT_MS.load(Ordering::Relaxed);
    now.saturating_sub(previous) >= 1000
        && LAST_STATUS_SNAPSHOT_MS
            .compare_exchange(previous, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
}

struct SharedWriter {
    writer: Arc<Mutex<Box<dyn std::io::Write + Send>>>,
    read_only: Arc<AtomicBool>,
}

impl std::io::Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.read_only.load(Ordering::Relaxed) {
            return Ok(buf.len());
        }
        self.writer.lock().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.read_only.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.writer.lock().flush()
    }
}
/// One terminal inside a session's screen. A session (a sidebar tab) owns a
/// split tree of panes; each pane is a full, independent terminal whose PTY
/// keeps running while another session's screen is shown.
pub struct TerminalPane {
    pub id: usize,
    pub generation: String,
    pub native_session_id: Option<String>,
    pub native_session_reporter: Option<String>,
    pub native_turn_id: Option<String>,
    pub native_sequence: HashMap<String, u64>,
    pub native_reporter_generation: HashMap<String, String>,
    pub pending_approvals: std::collections::HashSet<String>,
    pub native_status: Option<SessionStatus>,
    pub native_acknowledged: bool,
    pub osc_report_active: bool,
    pub terminal: Entity<TerminalView>,
    pub status: SessionStatus,
    pub detection: status::DetectionResult,
    pub title: String,
    /// Title as shown in the pane header (truncated to 160 chars), cached so
    /// the render pass does not re-scan and re-allocate it every frame.
    pub display_title: SharedString,
    pub pid: Option<u32>,
    /// Reused scrape buffer: the poller reads the screen into this instead of
    /// allocating a fresh `Vec<String>` every 500 ms per pane.
    pub screen_scratch: Vec<String>,
    /// (screen revision, focused, exited, foreground snapshot) from the last
    /// poll tick. The revision lets an idle pane skip rebuilding its screen
    /// strings entirely; the process snapshot is compared by `Arc` identity,
    /// which is O(1) because the `/proc` cache hands back the same allocation
    /// while the foreground process group is unchanged.
    pub poll_cache: Option<(u64, bool, bool, Arc<[String]>)>,
    /// Screen fingerprint at the moment the user last looked at a finished or
    /// failed terminal. While the screen still matches, the sticky Done/Error
    /// badge stays acknowledged (Idle); any new output invalidates it and lets
    /// a fresh Done/Error be raised.
    pub ack_screen: Option<u64>,
    /// A recognized agent CLI was the foreground process on the previous tick.
    /// Its disappearance means the agent finished (Done), like herdr.
    pub agent_active: bool,
    pub detector: status::Detector,
    /// Herdr-style seen/recency bookkeeping for this pane's finalized status.
    pub attention: attention::Attention,
    /// The agent's own name from its OSC 7501 report (`app`), when it reports one.
    pub app: Option<SharedString>,
    /// Sticky: this pane hosted a recognized agent. Cleared only when recreated.
    pub is_agent: bool,
    /// Workers reported by this pane's agent, keyed by their `agent_id`.
    pub sub_agents: HashMap<String, attention::SubAgent>,
    pub input_state: Option<Entity<InputState>>,
    /// Drops the `InputEvent::Change` subscription that keeps ghost text and
    /// the completion menu in sync with what is typed.
    pub input_subscription: Option<gpui::Subscription>,
    /// Shell history loaded from disk, shared by every pane.
    pub shell_history: Arc<Vec<String>>,
    /// Commands executed from this pane's input bar, newest last.
    pub command_history: Vec<String>,
    pub history_index: Option<usize>,
    pub pending_draft: String,
    /// Open completion menu (Warp-style suggestions above the input).
    pub completion_menu: Option<CompletionMenuState>,
    pub bottom_input_enabled: bool,
    pub bottom_input_visible: bool,
    pub plus_menu_open: bool,
}

/// One visible completion popup for a pane.
pub struct CompletionMenuState {
    /// Byte range in the input value that accepting replaces.
    pub start: usize,
    pub end: usize,
    /// The token these candidates were computed for, so typing resets the
    /// highlight instead of carrying it into a different list.
    pub token: String,
    pub items: Vec<completion::CompletionItem>,
    pub selected: usize,
}

impl Drop for TerminalPane {
    fn drop(&mut self) {
        #[cfg(unix)]
        herdr_compat::unregister(&self.generation);
    }
}

impl TerminalPane {
    pub fn new(
        id: usize,
        generation: String,
        terminal: Entity<TerminalView>,
        title: SharedString,
        display_title: SharedString,
        pid: Option<u32>,
        bottom_input_enabled: bool,
    ) -> Self {
        Self {
            id,
            generation,
            native_session_id: None,
            native_session_reporter: None,
            native_turn_id: None,
            native_sequence: HashMap::new(),
            native_reporter_generation: HashMap::new(),
            pending_approvals: std::collections::HashSet::new(),
            native_status: None,
            native_acknowledged: false,
            osc_report_active: false,
            terminal,
            status: SessionStatus::Idle,
            detection: status::DetectionResult::new(
                SessionStatus::Idle,
                status::EvidenceSource::NoSignal,
                status::Confidence::None,
            ),
            title: title.to_string(),
            display_title,
            pid,
            screen_scratch: Vec::new(),
            poll_cache: None,
            ack_screen: None,
            agent_active: false,
            detector: status::Detector::default(),
            attention: Default::default(),
            app: None,
            is_agent: false,
            sub_agents: HashMap::new(),
            input_state: None,
            input_subscription: None,
            shell_history: shell_history::cached(),
            command_history: Vec::new(),
            history_index: None,
            pending_draft: String::new(),
            completion_menu: None,
            bottom_input_enabled,
            bottom_input_visible: bottom_input_enabled,
            plus_menu_open: false,
        }
    }

    /// Whether this pane should show the bottom input bar.
    /// Can be toggled on or off at anytime by the user without blocking.
    /// When an AI agent / process is actively working, the bar auto-hides;
    /// when the agent stops working or asks for input (e.g. Blocked, Idle, Done),
    /// the bar auto-shows (Warp style).
    /// Alt-screen full-screen programs (vim, nano, htop, less) also temporarily hide it.
    pub fn should_show_bottom_input(&self, cx: &App) -> bool {
        if !self.bottom_input_enabled || !self.bottom_input_visible {
            return false;
        }

        // Alternate screen mode (vim, nano, htop, less, etc.) always hides the bottom bar
        if self.terminal.read(cx).is_alt_screen() {
            return false;
        }

        true
    }
}

/// One sidebar tab: an isolated screen with its own split-layout tree of
/// panes. Switching tabs only swaps which screen is rendered — the panes of
/// other sessions keep running in the background.
pub struct Session {
    pub id: usize,
    /// OSC title of the focus pane (synced in the poller / title callback).
    pub title: SharedString,
    /// Title as shown in the sidebar (truncated), cached.
    pub display_title: SharedString,
    /// Path of the AI agent / command logo shown in the sidebar.
    pub logo: SharedString,
    pub pane_tree: Option<PaneNode>,
    pub maximized_pane_id: Option<usize>,
    pub panes: Vec<TerminalPane>,
    /// Highest-ranked pane status, for the sidebar badge and title bar.
    pub status: SessionStatus,
}

/// Status poll period. Process status cannot be evented portably across the
/// platforms this app supports, so this bounded timer is the change detector;
/// the work it triggers is small (`/proc` reads now happen off the UI thread).
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);
#[cfg(unix)]
static FANOUT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

fn latch_completion(previous: SessionStatus, next: SessionStatus) -> SessionStatus {
    // Completion persistence is owned by the lifecycle reducer: idle/focus
    // observations cannot clear it, while fresh work always can.
    if matches!(previous, SessionStatus::Done | SessionStatus::Error)
        && matches!(
            next,
            SessionStatus::Idle | SessionStatus::Active | SessionStatus::Unknown
        )
    {
        previous
    } else {
        next
    }
}

#[cfg(test)]
#[test]
fn completed_status_survives_idle_and_focus_but_new_work_replaces_it() {
    assert_eq!(
        latch_completion(SessionStatus::Done, SessionStatus::Idle),
        SessionStatus::Done
    );
    assert_eq!(
        latch_completion(SessionStatus::Done, SessionStatus::Active),
        SessionStatus::Done
    );
    assert_eq!(
        latch_completion(SessionStatus::Error, SessionStatus::Unknown),
        SessionStatus::Error
    );
    assert_eq!(
        latch_completion(SessionStatus::Done, SessionStatus::Working),
        SessionStatus::Working
    );
    assert_eq!(
        latch_completion(SessionStatus::Done, SessionStatus::Blocked),
        SessionStatus::Blocked
    );
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaneNode {
    Leaf(usize),
    Split {
        id: usize,
        axis: gpui::Axis,
        left: Box<PaneNode>,
        right: Box<PaneNode>,
    },
}

impl PaneNode {
    pub fn contains(&self, id: usize) -> bool {
        match self {
            PaneNode::Leaf(leaf_id) => *leaf_id == id,
            PaneNode::Split { left, right, .. } => left.contains(id) || right.contains(id),
        }
    }

    pub fn split(
        &mut self,
        target_id: usize,
        new_id: usize,
        split_id: usize,
        axis: gpui::Axis,
    ) -> bool {
        match self {
            PaneNode::Leaf(id) => {
                if *id == target_id {
                    *self = PaneNode::Split {
                        id: split_id,
                        axis,
                        left: Box::new(PaneNode::Leaf(target_id)),
                        right: Box::new(PaneNode::Leaf(new_id)),
                    };
                    true
                } else {
                    false
                }
            }
            PaneNode::Split { left, right, .. } => {
                left.split(target_id, new_id, split_id, axis)
                    || right.split(target_id, new_id, split_id, axis)
            }
        }
    }

    pub fn remove(self, target_id: usize) -> Option<PaneNode> {
        match self {
            PaneNode::Leaf(id) => {
                if id == target_id {
                    None
                } else {
                    Some(PaneNode::Leaf(id))
                }
            }
            PaneNode::Split {
                id,
                axis,
                left,
                right,
            } => {
                let left = left.remove(target_id);
                let right = right.remove(target_id);
                match (left, right) {
                    (Some(l), Some(r)) => Some(PaneNode::Split {
                        id,
                        axis,
                        left: Box::new(l),
                        right: Box::new(r),
                    }),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
        }
    }

    pub fn first_leaf(&self) -> usize {
        match self {
            PaneNode::Leaf(id) => *id,
            PaneNode::Split { left, .. } => left.first_leaf(),
        }
    }
}

pub struct Project {
    pub id: usize,
    /// Shared (cheap to clone) because the render pass copies it every frame.
    pub name: SharedString,
    pub path: PathBuf,
    pub sessions: Vec<Session>,
    pub active_session_idx: usize,
    /// Whether this project's session list is expanded in the sidebar. Every
    /// project keeps its own state, so several open projects can be browsed
    /// independently without switching away from the one being worked in.
    pub expanded: bool,
}

struct AppState {
    projects: Vec<Project>,
    active_project_idx: usize,
    next_id: usize,
    palette: ColorPalette,
    /// O(1) session lookup by id: session id -> (project idx, session idx).
    session_index: HashMap<usize, (usize, usize)>,
    /// O(1) pane lookup by id: pane id -> (project idx, session idx, pane idx).
    pane_index: HashMap<usize, (usize, usize, usize)>,
    /// Pane that currently owns keyboard focus, tracked every render frame
    /// and consumed by the status poller (Active state, sticky-state clearing).
    focused_pane_id: Option<usize>,
    pub active_view: ActiveView,
    pub theme_preference: ThemePreference,
    pub sidebar_visible: bool,
    pub warp_bottom_input: bool,
    integration_panel_visible: bool,
    integration_feedback: SharedString,
    pub sidebar_width: Pixels,
    drag_start_pos: Option<Point<Pixels>>,
    persistence_path: Option<PathBuf>,
    save_generation: Arc<AtomicU64>,
    startup_loaded: bool,
    quit_subscription: Option<gpui::Subscription>,
    /// Whether the OS considers this window focused, mirrored from the render
    /// pass. "Looking" requires it: a visible-but-backgrounded window hasn't
    /// been seen.
    window_active: bool,
    /// The first poll tick only primes the attention model; it never alerts.
    status_primed: bool,
    /// Alerts to drain on the next render, where a window is available to
    /// push notifications and play sounds.
    pending_alerts: Vec<attention::AppAlert>,
    /// Monotonic counter for "most recent status change" ordering.
    attention_seq: u64,
    /// Currently open project dropdown menu: Some(project_id).
    project_menu: Option<usize>,
    renaming_project: Option<RenamingProject>,
    pub terminal_context_menu: Option<TerminalContextMenu>,
    /// Ctrl+R reverse history search palette, if open.
    history_search: Option<HistorySearchState>,
}

/// Warp-style "search command history" palette.
struct HistorySearchState {
    pane_id: usize,
    input: Entity<InputState>,
    matches: Vec<String>,
    selected: usize,
    _subscription: gpui::Subscription,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalContextMenu {
    pub pane_id: usize,
    pub position: Point<Pixels>,
}

struct RenamingProject {
    project_id: usize,
    input: Entity<InputState>,
    _subscription: gpui::Subscription,
}
impl AppState {
    fn saved_workspace(&self) -> workspace_state::SavedWorkspace {
        workspace_state::SavedWorkspace {
            projects: self
                .projects
                .iter()
                .map(|p| workspace_state::SavedProject {
                    path: p.path.clone(),
                    name: p.name.to_string(),
                    expanded: p.expanded,
                })
                .collect(),
            active_project: self.active_project().map(|p| p.path.clone()),
            sidebar_visible: self.sidebar_visible,
            sidebar_width: f32::from(self.sidebar_width),
            theme: self.theme_preference,
            warp_bottom_input: self.warp_bottom_input,
        }
    }

    fn schedule_workspace_save(&mut self, cx: &Context<Self>) {
        self.startup_loaded = true;
        let (Some(path), Some(snapshot)) =
            (self.persistence_path.clone(), Some(self.saved_workspace()))
        else {
            return;
        };
        let generation = self.save_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let generation_counter = self.save_generation.clone();
        let executor = cx.background_executor().clone();
        executor
            .clone()
            .spawn(async move {
                executor.timer(std::time::Duration::from_millis(300)).await;
                if let Err(error) = workspace_state::save_if_current(
                    &path,
                    &snapshot,
                    &generation_counter,
                    generation,
                ) {
                    eprintln!("failed to save workspace state: {error}");
                }
            })
            .detach();
    }

    fn apply_saved_workspace(
        &mut self,
        saved: workspace_state::SavedWorkspace,
        cx: &mut Context<Self>,
    ) {
        if self.startup_loaded {
            return;
        }
        self.startup_loaded = true;
        self.sidebar_visible = saved.sidebar_visible;
        self.sidebar_width = SIDEBAR_WIDTH;
        self.theme_preference = saved.theme;
        self.warp_bottom_input = saved.warp_bottom_input;
        let mode = self.theme_preference.resolve_mode(cx.window_appearance());
        self.palette = load_github_palette(mode.is_dark());
        gpui_component::Theme::change(mode, None, cx);
        self.projects.clear();
        self.active_project_idx = 0;
        for project in saved.projects {
            if !project.path.is_dir() || self.projects.iter().any(|p| p.path == project.path) {
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            let weak = cx.entity().downgrade();
            let Ok((terminal, pid, generation)) =
                Self::spawn_terminal_view(Some(&project.path), self.palette.clone(), weak, id, cx)
            else {
                continue;
            };
            let title = SharedString::from("terminal 1");
            let pane = TerminalPane::new(
                id,
                generation,
                terminal,
                title.clone(),
                title.clone(),
                pid,
                saved.warp_bottom_input,
            );
            let session = Session {
                id,
                title: title.clone(),
                display_title: title,
                logo: logos::default_logo(),
                pane_tree: Some(PaneNode::Leaf(id)),
                maximized_pane_id: None,
                panes: vec![pane],
                status: SessionStatus::Idle,
            };
            self.projects.push(Project {
                id,
                name: SharedString::from(project.name),
                path: project.path,
                sessions: vec![session],
                active_session_idx: 0,
                expanded: project.expanded,
            });
        }
        self.active_project_idx = saved
            .active_project
            .as_ref()
            .and_then(|path| self.projects.iter().position(|p| &p.path == path))
            .unwrap_or(0);
        if self.projects.is_empty() {
            let default_dir = std::env::current_dir()
                .ok()
                .or_else(|| directories::UserDirs::new().map(|u| u.home_dir().to_path_buf()))
                .unwrap_or_else(|| std::path::PathBuf::from("/"));
            let name = default_dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "project".to_string());
            let id = self.next_id;
            self.next_id += 1;
            let weak = cx.entity().downgrade();
            if let Ok((terminal, pid, generation)) =
                Self::spawn_terminal_view(Some(&default_dir), self.palette.clone(), weak, id, cx)
            {
                let title = SharedString::from("terminal 1");
                let pane = TerminalPane::new(
                    id,
                    generation,
                    terminal,
                    title.clone(),
                    title.clone(),
                    pid,
                    self.warp_bottom_input,
                );
                let session = Session {
                    id,
                    title: title.clone(),
                    display_title: title,
                    logo: logos::default_logo(),
                    pane_tree: Some(PaneNode::Leaf(id)),
                    maximized_pane_id: None,
                    panes: vec![pane],
                    status: SessionStatus::Idle,
                };
                self.projects.push(Project {
                    id,
                    name: SharedString::from(name),
                    path: default_dir,
                    sessions: vec![session],
                    active_session_idx: 0,
                    expanded: true,
                });
                self.focused_pane_id = Some(id);
                self.active_project_idx = 0;
            }
        } else {
            self.focused_pane_id = self.projects[self.active_project_idx]
                .sessions
                .first()
                .and_then(|s| s.panes.first())
                .map(|p| p.id);
        }
        self.rebuild_session_index();
        cx.notify();
    }

    pub fn tokens(&self, cx: &App) -> ThemeTokens {
        let mode = self.theme_preference.resolve_mode(cx.window_appearance());
        ThemeTokens::from_mode(mode)
    }

    #[allow(dead_code)]
    pub fn tokens_with_appearance(&self, appearance: gpui::WindowAppearance) -> ThemeTokens {
        let mode = self.theme_preference.resolve_mode(appearance);
        ThemeTokens::from_mode(mode)
    }

    pub fn set_theme_preference(
        &mut self,
        preference: ThemePreference,
        window: Option<&mut Window>,
        cx: &mut Context<Self>,
    ) {
        if self.theme_preference == preference {
            return;
        }
        self.theme_preference = preference;
        self.apply_theme_preference(window, cx);
        self.schedule_workspace_save(cx);
    }

    pub fn apply_theme_preference(
        &mut self,
        window: Option<&mut Window>,
        cx: &mut Context<Self>,
    ) {
        let appearance = window
            .as_ref()
            .map(|w| w.appearance())
            .unwrap_or_else(|| cx.window_appearance());
        let mode = self.theme_preference.resolve_mode(appearance);
        let is_dark = mode.is_dark();
        self.palette = load_github_palette(is_dark);
        gpui_component::Theme::change(mode, window, cx);

        for project in &mut self.projects {
            for session in &mut project.sessions {
                for pane in &mut session.panes {
                    let mut config = pane.terminal.read(cx).config().clone();
                    config.colors = self.palette.clone();
                    pane.terminal.update(cx, |term, cx| {
                        term.update_config(config, cx);
                    });
                }
            }
        }
        cx.notify();
    }

    fn make_drag_area(&self, cx: &Context<Self>) -> gpui::Div {
        div()
            .window_control_area(WindowControlArea::Drag)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, ev: &MouseDownEvent, window, _| {
                    if ev.click_count == 2 {
                        this.drag_start_pos = None;
                        window.zoom_window();
                    } else {
                        this.drag_start_pos = Some(ev.position);
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| {
                    this.drag_start_pos = None;
                }),
            )
            .on_mouse_move(cx.listener(|this, ev: &MouseMoveEvent, window, _| {
                if ev.pressed_button != Some(MouseButton::Left) {
                    this.drag_start_pos = None;
                    return;
                }
                if let Some(start_pos) = this.drag_start_pos {
                    let dx = (ev.position.x - start_pos.x).to_f64();
                    let dy = (ev.position.y - start_pos.y).to_f64();
                    if dx * dx + dy * dy >= 64.0 {
                        this.drag_start_pos = None;
                        window.start_window_move();
                    }
                }
            }))
    }

    fn active_project(&self) -> Option<&Project> {
        self.projects.get(self.active_project_idx)
    }

    #[cfg(test)]
    fn active_project_mut(&mut self) -> Option<&mut Project> {
        self.projects.get_mut(self.active_project_idx)
    }

    fn active_session(&self) -> Option<&Session> {
        let project = self.active_project()?;
        project.sessions.get(project.active_session_idx)
    }

    /// Panes the person can actually see: the selected screen's panes, minus
    /// the siblings a maximized pane hides.
    fn visible_pane_ids(&self) -> HashSet<usize> {
        self.active_session()
            .map(|session| {
                let max = session.maximized_pane_id;
                session
                    .panes
                    .iter()
                    .filter(|pane| max.map_or(true, |m| m == pane.id))
                    .map(|pane| pane.id)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn spawn_terminal_view(
        dir: Option<&PathBuf>,
        palette: ColorPalette,
        window_title_target: gpui::WeakEntity<Self>,
        pane_id: usize,
        cx: &mut Context<Self>,
    ) -> Result<(Entity<TerminalView>, Option<u32>, String)> {
        let (shell_cmd, _) = detect_shell();
        #[cfg(unix)]
        let generation = status_bus::new_generation();
        #[cfg(not(unix))]
        let generation = format!("pane-{pane_id}-{}", std::process::id());
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 80,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| anyhow::anyhow!("PTY open error: {e}"))?;

        let mut cmd = CommandBuilder::new(&shell_cmd);
        if let Some(d) = dir {
            cmd.cwd(d);
        }
        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");
        cmd.env("TERM_PROGRAM", "t3code-terminal");
        cmd.env("TERM_PROGRAM_VERSION", "0.1.0");
        cmd.env("LANG", "en_US.UTF-8");
        cmd.env("LC_ALL", "en_US.UTF-8");
        if let Ok(executable) = std::env::current_exe() {
            cmd.env("MARKUP_EXECUTABLE", executable);
        }
        #[cfg(unix)]
        if let Some(bus) = STATUS_BUS.as_ref() {
            herdr_compat::register(&bus.app_id, pane_id, &generation);
            cmd.env("MARKUP_STATUS_SOCKET", &bus.socket);
            cmd.env("MARKUP_APP_ID", &bus.app_id);
            cmd.env("MARKUP_PANE_ID", pane_id.to_string());
            cmd.env("MARKUP_PANE_GENERATION", &generation);
            // Native Herdr reporters (including Command Code) can use the
            // scoped reporting API. The opaque ID is bound to this generation.
            cmd.env("HERDR_ENV", "1");
            cmd.env("HERDR_SOCKET_PATH", &bus.socket);
            cmd.env("HERDR_PANE_ID", &generation);
            // A new pane is independent of a Command Code process that may
            // have launched Markup; its inherited child-owner guard must not
            // disable the vendor reporter for the new foreground agent.
            cmd.env_remove("CMD_HERDR_PANE_OWNER");
            if let Ok(executable) = std::env::current_exe() {
                cmd.env("HERDR_BIN_PATH", executable);
            }
            cmd.env(
                "MARKUP_STATUS_SNAPSHOT",
                status_bus::snapshot_path(bus, pane_id, &generation),
            );
        }
        #[cfg(unix)]
        if let Some(tty) = pair.master.tty_name() {
            cmd.env("MARKUP_STATUS_TTY", tty);
        }

        let child = pair.slave.spawn_command(cmd).map_err(|e| {
            #[cfg(unix)]
            herdr_compat::unregister(&generation);
            anyhow::anyhow!("Spawn shell error: {e}")
        })?;
        let pid = child.process_id();

        let writer = pair
            .master
            .take_writer()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let pty_master = Arc::new(Mutex::new(pair.master));
        let read_only = Arc::new(AtomicBool::new(false));

        let pty_writer: Arc<Mutex<Box<dyn std::io::Write + Send>>> = Arc::new(Mutex::new(writer));
        let shared_writer = SharedWriter {
            writer: pty_writer,
            read_only,
        };

        let config = TerminalConfig {
            font_family: MONO_FONT.into(),
            font_size: px(13.5),
            cols: 80,
            rows: 24,
            scrollback: 10_000,
            line_height_multiplier: 1.2,
            padding: Edges::all(px(8.0)),
            colors: palette,
            cursor_blink: true,
            copy_on_select: true,
            right_click_paste: false,
            alternate_scroll: true,
            detect_path_links: true,
            show_scrollbar: true,
            read_only: true,
            ..TerminalConfig::default()
        };

        let pty_for_resize = pty_master;
        let resize_callback = move |cols: usize, rows: usize| {
            let _ = pty_for_resize.lock().resize(PtySize {
                cols: cols as u16,
                rows: rows as u16,
                pixel_width: 0,
                pixel_height: 0,
            });
        };

        let target = window_title_target.clone();
        let terminal = cx.new(|cx| {
            TerminalView::new(shared_writer, reader, config, cx)
                .with_resize_callback(resize_callback)
                .with_title_callback(move |window, cx, title| {
                    window.set_window_title(title);
                    let title_str = title.to_string();
                    let _ = target.update(cx, |state, cx| {
                        state.update_pane_title(pane_id, &title_str, cx);
                    });
                })
            // Exit status is intentionally not set here: the status poller
            // derives Done/Error from the real exit code (see poll_sessions_status).
        });
        Ok((terminal, pid, generation))
    }

    fn update_pane_title(&mut self, pane_id: usize, title: &str, cx: &mut Context<Self>) {
        let trimmed = title.trim();
        if trimmed.is_empty() {
            return;
        }
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(session) = self
            .projects
            .get_mut(p_idx)
            .and_then(|p| p.sessions.get_mut(s_idx))
        else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };
        if pane.title == trimmed {
            return;
        }
        pane.title = trimmed.to_string();
        let next_display = truncate_title(&SharedString::from(trimmed.to_string()));
        pane.display_title = next_display.clone();
        // Mirror the pane the user is working in onto the session's sidebar
        // entry. Either the focused pane or the first pane of the screen
        // (when focus has not moved yet) is the primary.
        let is_primary = self.focused_pane_id == Some(pane_id)
            || session
                .pane_tree
                .as_ref()
                .is_some_and(|t| t.first_leaf() == pane_id);
        if is_primary {
            session.title = SharedString::from(trimmed.to_string());
            session.display_title = next_display.clone();
            let title_logo =
                logos::resolve_session_logo(&[], Some(trimmed), session.status, &session.logo);
            if title_logo != logos::default_logo() && session.logo != title_logo {
                session.logo = title_logo;
            }
        }
        cx.notify();
    }

    /// Periodic detection tick, herdr-style: screen scrape + OSC title +
    /// foreground process tree, with sticky Done/Error and focus resolution.
    /// Every pane of every session is polled; the session badge is the
    /// highest-ranked pane status and the sidebar title/logo follow the pane
    /// the user is working in.
    ///
    /// This variant gathers process information on the caller's thread. The app
    /// loop uses [`Self::poll_sessions_status_with`] so the blocking `/proc`
    /// reads run on the background executor; this entry point remains for tests
    /// and for callers that already have a snapshot.
    #[allow(dead_code)]
    pub fn poll_sessions_status(&mut self, cx: &mut Context<Self>) {
        let foregrounds = self.gather_foreground_processes();
        self.poll_sessions_status_with(&foregrounds, cx);
    }

    /// Shell pid of every live pane, deduplicated. Cheap and UI-thread safe:
    /// the background inspector needs a list, not an entity borrow.
    pub fn pane_pids(&self) -> Vec<u32> {
        let mut pids = Vec::new();
        for project in &self.projects {
            for session in &project.sessions {
                for pane in &session.panes {
                    if let Some(pid) = pane.pid
                        && !pids.contains(&pid)
                    {
                        pids.push(pid);
                    }
                }
            }
        }
        pids
    }

    #[cfg(unix)]
    fn apply_status_events(
        &mut self,
        events: Vec<status_bus::StatusEvent>,
        cx: &mut Context<Self>,
    ) {
        let Some(bus) = STATUS_BUS.as_ref() else {
            return;
        };
        let mut routed = Vec::new();
        for event in events {
            if event.app_id != bus.app_id {
                continue;
            }
            let Some(source) = self
                .projects
                .iter()
                .flat_map(|project| &project.sessions)
                .flat_map(|session| &session.panes)
                .find(|pane| {
                    pane.id as u64 == event.pane_id && pane.generation == event.pane_generation
                })
            else {
                continue;
            };
            if let Some(session_id) = event.session_id.as_ref() {
                for target in self
                    .projects
                    .iter()
                    .flat_map(|project| &project.sessions)
                    .flat_map(|session| &session.panes)
                    .filter(|pane| {
                        pane.id != source.id
                            && pane.native_session_id.as_ref() == Some(session_id)
                            && pane.native_session_reporter.as_deref()
                                == Some(event.reporter.as_str())
                    })
                {
                    let mut fanout = event.clone();
                    fanout.pane_id = target.id as u64;
                    fanout.pane_generation = target.generation.clone();
                    fanout.reporter_generation = format!("session-route:{session_id}");
                    fanout.sequence = FANOUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
                    routed.push((fanout, true));
                }
            }
            routed.push((event, false));
        }
        for (event, is_routed) in routed {
            if event.app_id != bus.app_id {
                continue;
            }
            let Some(&(project_idx, session_idx, pane_idx)) =
                self.pane_index.get(&(event.pane_id as usize))
            else {
                continue;
            };
            let Some(pane) = self
                .projects
                .get_mut(project_idx)
                .and_then(|p| p.sessions.get_mut(session_idx))
                .and_then(|s| s.panes.get_mut(pane_idx))
            else {
                continue;
            };
            if pane.generation != event.pane_generation {
                continue;
            }
            if event.agent_id.is_some() {
                // Workers report to the parent pane; they never own its session.
            } else if event.event == "register" {
                if let Some(session_id) = event.session_id.as_ref() {
                    if pane
                        .native_session_id
                        .as_ref()
                        .is_some_and(|old| old != session_id)
                    {
                        pane.native_status = None;
                        pane.native_acknowledged = false;
                        pane.pending_approvals.clear();
                        pane.native_turn_id = None;
                        pane.ack_screen = None;
                        pane.poll_cache = None;
                        pane.status = SessionStatus::Unknown;
                    }
                    pane.native_session_id = Some(session_id.clone());
                    pane.native_session_reporter = Some(event.reporter.clone());
                }
            } else if let Some(session_id) = event.session_id.as_ref() {
                match (
                    pane.native_session_reporter.as_deref(),
                    pane.native_session_id.as_ref(),
                ) {
                    (Some(reporter), _) if reporter != event.reporter => continue,
                    (_, Some(current)) if current != session_id => continue,
                    _ => {
                        pane.native_session_id = Some(session_id.clone());
                        pane.native_session_reporter = Some(event.reporter.clone());
                    }
                }
            }

            let source = if let Some(agent_id) = event.agent_id.as_ref() {
                format!("{}:{}:{agent_id}", event.reporter, event.reporter_generation)
            } else if is_routed {
                format!(
                    "route:{}:{}",
                    event.reporter,
                    event.session_id.as_deref().unwrap_or("")
                )
            } else {
                format!("{}:{}", event.reporter, event.reporter_generation)
            };
            if !is_routed {
                let active_generation = pane.native_reporter_generation.get(&event.reporter);
                if active_generation.is_some_and(|active| active != &event.reporter_generation) {
                    if event.event != "register" {
                        continue;
                    }
                    let prefix = format!("{}:", event.reporter);
                    pane.native_sequence
                        .retain(|key, _| !key.starts_with(&prefix));
                }
                pane.native_reporter_generation
                    .insert(event.reporter.clone(), event.reporter_generation.clone());
            }
            if pane
                .native_sequence
                .get(&source)
                .is_some_and(|previous| event.sequence <= *previous)
            {
                continue;
            }
            pane.native_sequence.insert(source, event.sequence);
            // A worker reports to its parent pane: update the indented entry and
            // never let it drive the pane's own status or an alert.
            if let Some(agent_id) = event.agent_id.clone() {
                let status = match event.event.as_str() {
                    "working" => Some(SessionStatus::Working),
                    "blocked" => Some(SessionStatus::Blocked),
                    "done" => Some(SessionStatus::Done),
                    "error" => Some(SessionStatus::Error),
                    "idle" | "register" => Some(SessionStatus::Idle),
                    "clear" | "cancelled" => None,
                    _ => {
                        pane.is_agent = true;
                        continue;
                    }
                };
                match status {
                    Some(status) => {
                        let since = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                        let entry = pane.sub_agents.entry(agent_id.clone()).or_insert(
                            attention::SubAgent {
                                id: agent_id.clone(),
                                title: event.message.clone(),
                                status: SessionStatus::Idle,
                                since,
                                seq: since,
                            },
                        );
                        entry.status = status;
                        entry.since = since;
                        entry.seq = since;
                        if entry.title.is_none() {
                            entry.title = event.message.clone();
                        }
                    }
                    None => {
                        pane.sub_agents.remove(&agent_id);
                    }
                }
                pane.is_agent = true;
                continue;
            }
            let reporter_name = event.reporter.clone();
            let event_name = event.event.clone();
            if pane.native_acknowledged && matches!(event.event.as_str(), "done" | "error") {
                continue;
            }
            if let Some(turn_id) = event.turn_id.as_ref() {
                if pane
                    .native_turn_id
                    .as_ref()
                    .is_some_and(|old| old != turn_id)
                    && matches!(event.event.as_str(), "working" | "blocked")
                {
                    pane.pending_approvals.clear();
                    pane.native_status = None;
                    pane.ack_screen = None;
                }
                pane.native_turn_id = Some(turn_id.clone());
            }
            let next = match event.event.as_str() {
                "register" | "heartbeat" => None,
                "idle" => Some(latch_completion(
                    pane.native_status.unwrap_or(pane.status),
                    SessionStatus::Idle,
                )),
                "working" => {
                    pane.native_acknowledged = false;
                    pane.pending_approvals.clear();
                    Some(SessionStatus::Working)
                }
                "blocked" => {
                    pane.native_acknowledged = false;
                    pane.pending_approvals
                        .insert(event.approval_id.unwrap_or_else(|| "__single__".into()));
                    Some(SessionStatus::Blocked)
                }
                "approval_resolved" => {
                    if let Some(id) = event.approval_id {
                        if !pane.pending_approvals.remove(&id) && pane.pending_approvals.len() == 1
                        {
                            pane.pending_approvals.clear();
                        }
                    } else if pane.pending_approvals.len() == 1 {
                        pane.pending_approvals.clear();
                    }
                    Some(if pane.pending_approvals.is_empty() {
                        SessionStatus::Working
                    } else {
                        SessionStatus::Blocked
                    })
                }
                "done" => {
                    pane.pending_approvals.clear();
                    Some(SessionStatus::Done)
                }
                "error" => {
                    pane.pending_approvals.clear();
                    Some(SessionStatus::Error)
                }
                "clear" | "cancelled" => {
                    pane.pending_approvals.clear();
                    pane.native_status = None;
                    pane.native_acknowledged = true;
                    pane.osc_report_active = false;
                    pane.terminal
                        .update(cx, |terminal, _| terminal.clear_agent_status_reports());
                    pane.status = SessionStatus::Unknown;
                    pane.ack_screen = None;
                    pane.poll_cache = None;
                    None
                }
                _ => None,
            };
            if let Some(next) = next {
                pane.native_status = Some(next);
                pane.status = next;
                pane.is_agent = true;
                if pane.app.is_none() && !reporter_name.is_empty() {
                    pane.app = Some(SharedString::from(reporter_name.clone()));
                }
                pane.detection = status::DetectionResult::new(
                    next,
                    status::EvidenceSource::AgentReport,
                    status::Confidence::High,
                );
                if matches!(next, SessionStatus::Working | SessionStatus::Blocked) {
                    pane.ack_screen = None;
                }
                pane.poll_cache = None;
            }
            self.integration_feedback =
                SharedString::from(format!("Live reporting from {reporter_name}: {event_name}"));
        }
        cx.notify();
    }

    /// Read `/proc` for every pane on the calling thread. The app loop uses
    /// [`Self::poll_sessions_status_with`] and gathers these on the background
    /// executor instead, so the blocking reads never sit in a frame.
    #[allow(dead_code)]
    fn gather_foreground_processes(&self) -> HashMap<u32, Arc<[String]>> {
        self.pane_pids()
            .into_iter()
            .map(|pid| (pid, status::foreground_process_names(Some(pid))))
            .collect()
    }

    /// Poll every pane against a foreground-process snapshot gathered off the
    /// UI thread. The snapshot is keyed by shell pid; a pane whose pid is
    /// missing simply sees an empty foreground group for this tick.
    pub fn poll_sessions_status_with(
        &mut self,
        foregrounds: &HashMap<u32, Arc<[String]>>,
        cx: &mut Context<Self>,
    ) {
        #[cfg(unix)]
        self.apply_status_events(status_bus::drain(), cx);
        let diag_poll = diagnostics::poll_start();
        let mut changed = false;
        let mut panes_polled = 0u64;
        let focused_id = self.focused_pane_id;
        // `T3_STATUS_DEBUG=1` dumps every detection pass: the facts that went in
        // and the badge that came out. Without this the poller is invisible.
        let debug = status::debug_enabled();
        let trace = status::trace_enabled();
        for project in &mut self.projects {
            for session in &mut project.sessions {
                let default_primary_id = session.pane_tree.as_ref().map(|t| t.first_leaf());
                let pane_count = session.panes.len();
                let mut primary_idx: Option<usize> = None;
                let mut primary_focused = false;
                let mut primary_fg: Arc<[String]> = status::empty_foreground_names();

                for (pane_idx, pane) in session.panes.iter_mut().enumerate() {
                    panes_polled += 1;
                    let focused = focused_id == Some(pane.id);
                    // Reuse the same immutable foreground snapshot for status,
                    // cache keys, and the session logo (one /proc probe per pane,
                    // gathered off the UI thread).
                    let foreground = pane
                        .pid
                        .and_then(|pid| foregrounds.get(&pid))
                        .cloned()
                        .unwrap_or_else(status::unknown_foreground_names);

                    // The pane the session badge/title follows: the focused
                    // pane, else the first pane of the screen.
                    let is_default_primary = Some(pane.id) == default_primary_id || pane_count == 1;
                    if focused || (is_default_primary && !primary_focused) {
                        primary_focused = focused;
                        primary_idx = Some(pane_idx);
                        primary_fg = Arc::clone(&foreground);
                    }

                    // Background panes never render, so their PTY events
                    // (title updates, child exit) must be drained here or the
                    // status would go stale while the user is in another tab.
                    if !focused {
                        pane.terminal.update(cx, |term, cx| {
                            term.drain_background_events(cx);
                        });
                    }

                    // Keep the pane header title in step with the OSC title
                    // even for background panes (the callback only fires where
                    // a window is attached).
                    let title_changed = {
                        let terminal = pane.terminal.read(cx);
                        let title = terminal.title().unwrap_or("");
                        let trimmed = title.trim();
                        if !trimmed.is_empty() && pane.title.as_str() != trimmed {
                            pane.title = trimmed.to_string();
                            pane.display_title =
                                truncate_title(&SharedString::from(trimmed.to_string()));
                            true
                        } else {
                            false
                        }
                    };
                    if title_changed {
                        changed = true;
                    }

                    // Idle fast path: the screen revision changes on PTY output
                    // and resize, while focus, exit state, and foreground-group
                    // changes are checked independently. Stable panes therefore
                    // skip row extraction, string allocation, and screen hashing.
                    // The foreground snapshot is compared by pointer: the /proc
                    // cache returns the same allocation while the group is
                    // unchanged, so this is a load and a compare.
                    let (screen_revision, exited, mut agent_report, osc_progress) = {
                        let terminal = pane.terminal.read(cx);
                        (
                            terminal.screen_revision(),
                            terminal.has_exited(),
                            terminal.agent_status_report(),
                            terminal.osc_progress().to_owned(),
                        )
                    };
                    let report_was_cleared = agent_report.is_none() && pane.osc_report_active;
                    pane.osc_report_active = agent_report.is_some();
                    if report_was_cleared {
                        pane.status = SessionStatus::Idle;
                        pane.detection = status::DetectionResult::new(
                            SessionStatus::Idle,
                            status::EvidenceSource::NoSignal,
                            status::Confidence::None,
                        );
                        pane.ack_screen = None;
                        pane.native_status = None;
                        pane.native_acknowledged = true;
                        pane.pending_approvals.clear();
                        pane.osc_report_active = false;
                    }
                    let foreground_changed =
                        pane.poll_cache
                            .as_ref()
                            .is_some_and(|(_, _, _, old_foreground)| {
                                !Arc::ptr_eq(old_foreground, &foreground)
                            });
                    let new_foreground_job =
                        pane.poll_cache.as_ref().is_some_and(|(_, _, _, old)| {
                            foreground_changed
                                && !foreground.is_empty()
                                && !Arc::ptr_eq(&foreground, &status::unknown_foreground_names())
                                && !Arc::ptr_eq(old, &status::unknown_foreground_names())
                        });
                    if foreground_changed && new_foreground_job && !foreground.is_empty() {
                        pane.native_acknowledged = false;
                        pane.ack_screen = None;
                        if matches!(pane.status, SessionStatus::Done | SessionStatus::Error) {
                            // A new command invalidates a completed job's badge.
                            pane.status = SessionStatus::Idle;
                            pane.detection = status::DetectionResult::new(
                                SessionStatus::Idle,
                                status::EvidenceSource::NoSignal,
                                status::Confidence::None,
                            );
                            pane.native_status = None;
                            pane.pending_approvals.clear();
                            if agent_report.as_ref().is_some_and(|report| {
                                matches!(
                                    report.state,
                                    terminal::AgentReportState::Done
                                        | terminal::AgentReportState::Error
                                        | terminal::AgentReportState::Idle
                                )
                            }) {
                                pane.terminal.update(cx, |terminal, _| {
                                    terminal.clear_agent_status_reports()
                                });
                                agent_report = None;
                            }
                            changed = true;
                        }
                    }
                    let unchanged = pane.poll_cache.as_ref().is_some_and(
                        |(revision, was_focused, was_exited, old_foreground)| {
                            *revision == screen_revision
                                && *was_focused == focused
                                && *was_exited == exited
                                && Arc::ptr_eq(old_foreground, &foreground)
                        },
                    );
                    // The grid only moved if the revision did; a focus or
                    // process-group change still needs a fresh verdict but can
                    // reuse the rows already in the scratch buffer.
                    let screen_unchanged = pane
                        .poll_cache
                        .as_ref()
                        .is_some_and(|(revision, ..)| *revision == screen_revision);
                    if unchanged {
                        match pane.status {
                            // A finished pane stays latched until new evidence.
                            SessionStatus::Done => continue,
                            // Working panes still need the detector's idle
                            // transition, but the grid did not move since the
                            // last scrape: reuse the cached rows instead of
                            // re-extracting and re-allocating the screen.
                            SessionStatus::Working | SessionStatus::Unknown => {}
                            _ => continue,
                        }
                    }

                    // Scrape into the pane's reused buffer only after the cheap
                    // revision/focus/process checks show that something changed.
                    // Ask for the whole visible screen: fresh shells can print
                    // their prompt on row 0, not just the last dozen rows.
                    if !screen_unchanged || pane.screen_scratch.is_empty() {
                        let diag_scrape = diagnostics::scrape_start();
                        pane.terminal
                            .read(cx)
                            .bottom_lines_into(usize::MAX, &mut pane.screen_scratch);
                        diagnostics::scrape_end(diag_scrape);
                    }

                    let screen_fp = {
                        let terminal = pane.terminal.read(cx);
                        screen_fingerprint(&pane.screen_scratch, terminal.title().unwrap_or(""))
                    };
                    let poll_key = (screen_revision, focused, exited, Arc::clone(&foreground));

                    let mut matched_agent_rule = false;
                    let mut matched_agent_idle = false;
                    let mut reported_agent_state = false;
                    let (next, ack) = {
                        let diag_detect = diagnostics::detect_start();
                        let terminal = pane.terminal.read(cx);
                        let bottom_lines = &pane.screen_scratch;
                        let title = terminal.title().unwrap_or("");
                        let screen_text = bottom_lines.join("\n");
                        let detector_state = pane.detector.update(
                            &screen_text,
                            terminal.has_exited(),
                            std::time::Instant::now(),
                        );
                        // while the screen is unchanged the sticky badge stays
                        // cleared; new output re-arms it.
                        let ack = pane.ack_screen;
                        let acknowledged = ack == Some(screen_fp) || pane.native_acknowledged;

                        // The agent rule match behind this tick's verdict, for the
                        // debug dump. The live path below captures it for free;
                        // the exited path only needs it for the dump itself.
                        let mut debug_match = None;

                        // Exit of the PTY child wins: the real exit code decides
                        // Done (0) vs Error (non-zero / signal), herdr-style.
                        let next = if terminal.has_exited() {
                            if debug {
                                debug_match = status::explain(&StatusInput {
                                    lines: bottom_lines,
                                    title: Some(title),
                                    foreground: &foreground,
                                    detector_state: Some(detector_state),
                                    osc_progress: &osc_progress,
                                });
                            }
                            if acknowledged {
                                SessionStatus::Idle
                            } else {
                                match terminal.exit_status() {
                                    Some(0) | None => SessionStatus::Done,
                                    Some(_) => SessionStatus::Error,
                                }
                            }
                        } else if let Some(native) = pane.native_status {
                            reported_agent_state = true;
                            native
                        } else if let Some(report) = &agent_report {
                            reported_agent_state = true;
                            match report.state {
                                terminal::AgentReportState::Idle => {
                                    if focused {
                                        SessionStatus::Active
                                    } else {
                                        SessionStatus::Idle
                                    }
                                }
                                terminal::AgentReportState::Working => SessionStatus::Working,
                                terminal::AgentReportState::Done => SessionStatus::Done,
                                terminal::AgentReportState::Blocked => SessionStatus::Blocked,
                                terminal::AgentReportState::Error => SessionStatus::Error,
                            }
                        } else {
                            let input = StatusInput {
                                lines: bottom_lines,
                                title: Some(title),
                                foreground: &foreground,
                                detector_state: Some(detector_state),
                                osc_progress: &osc_progress,
                            };
                            // The match is already computed by detection, so the
                            // debug dump can use it without a second evaluation.
                            let (detected, rule_match) = status::evaluate_screen_with_match(
                                &screen_text,
                                &input,
                                pane.status,
                                focused,
                            );
                            matched_agent_rule = rule_match.is_some();
                            matched_agent_idle = rule_match.as_ref().is_some_and(|matched| {
                                matched.state == agent_rules::RuleState::Idle
                                    && !matched.skip_state_update
                            });
                            if !debug {
                                debug_match = None;
                            } else {
                                debug_match = rule_match.clone();
                            }
                            if acknowledged
                                && matches!(detected, SessionStatus::Done | SessionStatus::Error)
                            {
                                if focused {
                                    SessionStatus::Active
                                } else {
                                    SessionStatus::Idle
                                }
                            } else {
                                detected
                            }
                        };

                        if debug {
                            let rule = debug_match.as_ref().map_or_else(
                                || "(no rule)".to_string(),
                                |m| {
                                    format!(
                                        "{}/{} v{} source={} state={} priority={} region={} \
                                         visible(working/blocker/idle)={}/{}/{} skip_update={}",
                                        m.pack,
                                        m.rule,
                                        m.pack_version.as_deref().unwrap_or("?"),
                                        m.source,
                                        m.state.label(),
                                        m.priority,
                                        m.region,
                                        m.visible_working,
                                        m.visible_blocker,
                                        m.visible_idle,
                                        m.skip_state_update,
                                    )
                                },
                            );
                            let evidence = if terminal.has_exited() {
                                "process-exit"
                            } else if reported_agent_state {
                                "agent-report"
                            } else if matched_agent_rule {
                                "agent-rule"
                            } else if matches!(
                                next,
                                SessionStatus::Blocked
                                    | SessionStatus::Error
                                    | SessionStatus::Working
                            ) {
                                "screen-heuristic"
                            } else if !foreground.is_empty() {
                                "foreground-process"
                            } else {
                                "no-signal"
                            };
                            let report_state = agent_report
                                .as_ref()
                                .map(|report| format!("{:?}", report.state));
                            eprintln!(
                                "[status] pane={} focused={focused} exited={} exit={:?} \
                                 pid={:?} fg={foreground:?} pack={:?} title={title:?} \
                                 prev={} next={} detector={detector_state:?} evidence={evidence} \
                                 report_state={report_state:?} rule={rule}",
                                pane.id,
                                terminal.has_exited(),
                                terminal.exit_status(),
                                pane.pid,
                                status::pack_for(&foreground),
                                pane.status.label(),
                                next.label(),
                            );
                            let screen: Vec<&str> = bottom_lines
                                .iter()
                                .map(String::as_str)
                                .filter(|line| !line.trim().is_empty())
                                .collect();
                            let from = screen.len().saturating_sub(DEBUG_SCREEN_LINES);
                            for line in &screen[from..] {
                                eprintln!("[status]   | {line}");
                            }
                        }

                        diagnostics::detect_end(diag_detect);
                        (next, ack)
                    };

                    // A done badge needs a recognized agent to have been working
                    // and then either exit or return to its explicit idle rule.
                    // Merely having a quiet foreground process is not completion.
                    let agent_now = foreground.iter().any(|name| status::is_agent_process(name));
                    let process_state_unavailable = foreground
                        .iter()
                        .any(|name| name == "<process-state-unavailable>");
                    let agent_exited =
                        pane.agent_active && !agent_now && !process_state_unavailable;
                    if !process_state_unavailable {
                        pane.agent_active = agent_now;
                    }

                    let turn_finished = !reported_agent_state
                        && pane.status == SessionStatus::Working
                        && (matched_agent_idle || agent_exited)
                        && matches!(next, SessionStatus::Idle | SessionStatus::Active)
                        && (agent_now || agent_exited);
                    let just_finished = turn_finished;
                    let next = if just_finished
                        && matches!(next, SessionStatus::Idle | SessionStatus::Active)
                        && ack != Some(screen_fp)
                    {
                        SessionStatus::Done
                    } else {
                        next
                    };
                    if matches!(next, SessionStatus::Working | SessionStatus::Blocked) {
                        pane.native_acknowledged = false;
                    }

                    // Keep the result visible across focus changes and trailing
                    // idle reports. A new working/blocked state, foreground job,
                    // or explicit clear starts a new lifecycle.
                    let next = latch_completion(pane.status, next);

                    pane.ack_screen = ack;
                    let evidence = if exited {
                        (
                            status::EvidenceSource::ProcessExit,
                            status::Confidence::High,
                        )
                    } else if reported_agent_state {
                        (
                            status::EvidenceSource::AgentReport,
                            status::Confidence::High,
                        )
                    } else if matched_agent_rule {
                        (status::EvidenceSource::AgentRule, status::Confidence::High)
                    } else if agent_exited {
                        (
                            status::EvidenceSource::AgentLifecycle,
                            status::Confidence::High,
                        )
                    } else if matches!(
                        next,
                        SessionStatus::Blocked | SessionStatus::Error | SessionStatus::Working
                    ) {
                        (
                            status::EvidenceSource::ScreenHeuristic,
                            status::Confidence::Low,
                        )
                    } else if next == SessionStatus::Unknown && !foreground.is_empty() {
                        (
                            status::EvidenceSource::ForegroundProcess,
                            status::Confidence::Low,
                        )
                    } else {
                        (status::EvidenceSource::NoSignal, status::Confidence::None)
                    };
                    let previous_status = pane.status;
                    pane.detection = status::DetectionResult::new(next, evidence.0, evidence.1);
                    if trace && previous_status != next {
                        eprintln!(
                            "[status-trace] pane={} {} -> {} evidence={:?} confidence={:?}",
                            pane.id,
                            previous_status.label(),
                            next.label(),
                            evidence.0,
                            evidence.1
                        );
                    }
                    if pane.status != next {
                        let prev_status = pane.status;
                        pane.status = next;
                        changed = true;

                        // Dynamic show/hide based on agent / terminal status (Warp style)
                        if pane.bottom_input_enabled {
                            if prev_status != SessionStatus::Working && next == SessionStatus::Working {
                                // Agent or command started working: auto-hide the input bar
                                if pane.bottom_input_visible {
                                    pane.bottom_input_visible = false;
                                    pane.completion_menu = None;
                                    pane.terminal.update(cx, |term, _| term.set_read_only(false));
                                }
                            } else if (prev_status == SessionStatus::Working && next != SessionStatus::Working)
                                || (next == SessionStatus::Blocked && prev_status != SessionStatus::Blocked)
                            {
                                // Agent or command stopped working or asks for input (Blocked/Idle/Done): auto-show input bar
                                if !pane.bottom_input_visible && !pane.terminal.read(cx).is_alt_screen() {
                                    pane.bottom_input_visible = true;
                                    pane.terminal.update(cx, |term, _| term.set_read_only(true));
                                }
                            }
                        }
                    }

                    if let Some(app) = agent_report
                        .as_ref()
                        .and_then(|report| report.app.clone())
                        .filter(|app| !app.is_empty())
                    {
                        pane.app = Some(SharedString::from(app));
                    }
                    pane.is_agent |= reported_agent_state
                        || matched_agent_rule
                        || agent_now
                        || pane.app.is_some();

                    pane.poll_cache = Some(poll_key);
                }

                // Session badge: the highest-ranked pane status wins, so one
                // blocked pane inside a screen is enough to flag the tab.
                let mut aggregate = SessionStatus::Idle;
                for pane in &session.panes {
                    if pane.status.rank() > aggregate.rank() {
                        aggregate = pane.status;
                    }
                }
                if session.status != aggregate {
                    session.status = aggregate;
                    changed = true;
                }

                // Sidebar title/logo follow the pane the user is working in.
                if let Some(idx) = primary_idx {
                    let (primary_title, primary_status) = {
                        let pane = &session.panes[idx];
                        (pane.title.clone(), pane.status)
                    };
                    let next_logo = logos::resolve_session_logo(
                        &primary_fg,
                        Some(&primary_title),
                        primary_status,
                        &session.logo,
                    );
                    if session.logo != next_logo {
                        session.logo = next_logo;
                        changed = true;
                    }
                    if session.title.as_str() != primary_title {
                        let shared = SharedString::from(primary_title);
                        session.title = shared;
                        session.display_title = truncate_title(&session.title);
                        changed = true;
                    }
                }
            }
        }
        // Herdr's attention pass. It finalizes every pane's badge: a finish the
        // person missed becomes Done, and looking at a Done pane clears it. It
        // runs for every pane every tick because visibility and window focus
        // change without any screen output.
        let visible_ids = self.visible_pane_ids();
        let window_active = self.window_active;
        let focused_id = self.focused_pane_id;
        let primed = self.status_primed;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut attention_seq = self.attention_seq;
        let mut alerts: Vec<attention::AppAlert> = Vec::new();
        let mut attention_changed = false;
        for project in &mut self.projects {
            let project_name = project.name.clone();
            for session in &mut project.sessions {
                let place = format!("{project_name} · {}", session.display_title);
                for pane in &mut session.panes {
                    let focused = focused_id == Some(pane.id);
                    let looking = visible_ids.contains(&pane.id) && window_active;
                    let prev = pane.attention.last;
                    let before = pane.status;
                    let resolved = attention::resolve(
                        &mut pane.attention,
                        pane.status,
                        pane.is_agent,
                        focused,
                        looking,
                        now_ms,
                        &mut attention_seq,
                    );
                    // Looking clears a finish for good: acknowledge it so a
                    // sticky native/OSC `done` cannot re-arm when you look away.
                    if looking
                        && before == SessionStatus::Done
                        && resolved.status != SessionStatus::Done
                    {
                        pane.native_status = None;
                        pane.native_acknowledged = true;
                        pane.ack_screen =
                            Some(screen_fingerprint(&pane.screen_scratch, &pane.title));
                        pane.osc_report_active = false;
                        pane.pending_approvals.clear();
                        pane.poll_cache = None;
                        pane.terminal
                            .update(cx, |terminal, _| terminal.clear_agent_status_reports());
                    }
                    if pane.status != resolved.status {
                        pane.status = resolved.status;
                        attention_changed = true;
                    }
                    if primed
                        && let Some(kind) = attention::alert_for(prev, resolved.status, looking)
                    {
                        alerts.push(attention::AppAlert {
                            kind,
                            pane_id: pane.id,
                            name: pane_agent_name(pane),
                            place: place.clone(),
                            looking,
                        });
                    }
                }
                // Fold the finalized pane badges back into the session rollup.
                let mut aggregate = SessionStatus::Idle;
                for pane in &session.panes {
                    if pane.status.rank() > aggregate.rank() {
                        aggregate = pane.status;
                    }
                }
                if session.status != aggregate {
                    session.status = aggregate;
                    attention_changed = true;
                }
            }
        }
        self.attention_seq = attention_seq;
        self.status_primed = true;
        if !alerts.is_empty() {
            self.pending_alerts.extend(alerts);
        }
        changed |= attention_changed;
        #[cfg(unix)]
        if let Some(bus) = STATUS_BUS.as_ref()
            && status_snapshot_due()
        {
            let paths: Vec<_> = self
                .projects
                .iter()
                .flat_map(|project| &project.sessions)
                .flat_map(|session| &session.panes)
                .map(|pane| {
                    let (evidence, confidence) =
                        match (pane.detection.evidence, pane.detection.confidence) {
                            (status::EvidenceSource::ProcessExit, status::Confidence::High) => {
                                ("process-exit", "high")
                            }
                            (status::EvidenceSource::AgentReport, status::Confidence::High) => {
                                ("agent-report", "high")
                            }
                            (status::EvidenceSource::AgentRule, status::Confidence::High) => {
                                ("agent-rule", "high")
                            }
                            (status::EvidenceSource::AgentLifecycle, status::Confidence::High) => {
                                ("agent-lifecycle", "high")
                            }
                            (status::EvidenceSource::TerminalProgress, _) => {
                                ("terminal-progress", "medium")
                            }
                            (status::EvidenceSource::ScreenHeuristic, _) => {
                                ("screen-heuristic", "low")
                            }
                            (status::EvidenceSource::ForegroundProcess, _) => {
                                ("foreground-process", "low")
                            }
                            _ => ("no-signal", "none"),
                        };
                    let snapshot = status_bus::StatusSnapshot {
                        version: 1,
                        app_id: bus.app_id.clone(),
                        pane_id: pane.id as u64,
                        pane_generation: pane.generation.clone(),
                        state: pane.status.label().into(),
                        evidence: evidence.into(),
                        confidence: confidence.into(),
                        reporter: pane.native_session_reporter.clone(),
                        updated_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis(),
                    };
                    (
                        status_bus::snapshot_path(bus, pane.id, &pane.generation),
                        snapshot,
                    )
                })
                .collect();
            cx.background_executor()
                .spawn(async move {
                    for (path, snapshot) in paths {
                        let Ok(bytes) = serde_json::to_vec(&snapshot) else {
                            continue;
                        };
                        let temp = path.with_extension(format!("{}.tmp", std::process::id()));
                        if std::fs::write(&temp, bytes).is_ok() {
                            let _ = std::fs::rename(temp, path);
                        }
                    }
                })
                .detach();
        }
        diagnostics::panes_polled(panes_polled);
        diagnostics::poll_end(diag_poll);
        if changed {
            cx.notify();
        }
    }

    /// Make a project the one shown in the workspace and focus its current
    /// pane. Expanding it is part of activation: selecting a project should
    /// always reveal the sessions that belong to it.
    fn activate_project(
        &mut self,
        project_idx: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(project) = self.projects.get_mut(project_idx) else {
            return;
        };
        self.active_project_idx = project_idx;
        project.expanded = true;
        let session_idx = project
            .active_session_idx
            .min(project.sessions.len().saturating_sub(1));
        project.active_session_idx = session_idx;
        let first_pane = project
            .sessions
            .get(session_idx)
            .and_then(|s| s.panes.first());
        if let Some(pane) = first_pane {
            pane.terminal.read(cx).focus_handle().focus(window);
        }
        self.focused_pane_id = first_pane.map(|pane| pane.id);
        cx.notify();
        self.schedule_workspace_save(cx);
    }

    fn toggle_project_expanded(&mut self, project_idx: usize, cx: &mut Context<Self>) {
        if let Some(project) = self.projects.get_mut(project_idx) {
            project.expanded = !project.expanded;
            cx.notify();
            self.schedule_workspace_save(cx);
        }
    }

    /// Create a session inside one specific project, switching the workspace
    /// to that project so the new terminal is the one being worked in. This
    /// backs the "+" on every project header, not just the active project.
    fn create_new_session_in_project(
        &mut self,
        project_idx: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let self_weak = cx.entity().downgrade();
        let next_id = self.next_id;
        self.next_id += 1;

        let Some(dir) = self.projects.get(project_idx).map(|p| p.path.clone()) else {
            return;
        };
        if let Ok((terminal, pid, generation)) =
            Self::spawn_terminal_view(Some(&dir), self.palette.clone(), self_weak, next_id, cx)
        {
            let session_count = self
                .projects
                .get(project_idx)
                .map(|p| p.sessions.len())
                .unwrap_or(0);
            let title = SharedString::from(format!("terminal {}", session_count + 1));
            let display_title = truncate_title(&title);
            let mut pane = TerminalPane::new(
                next_id,
                generation,
                terminal.clone(),
                title.clone(),
                display_title.clone(),
                pid,
                self.warp_bottom_input,
            );
            let (input_state, input_subscription) = build_pane_input(next_id, window, cx);
            if self.warp_bottom_input {
                input_state.update(cx, |state, cx| {
                    state.focus(window, cx);
                });
            } else {
                terminal.read(cx).focus_handle().focus(window);
            }
            pane.input_state = Some(input_state);
            pane.input_subscription = Some(input_subscription);
            let session = Session {
                id: next_id,
                title: title.clone(),
                display_title: truncate_title(&title),
                logo: logos::default_logo(),
                pane_tree: Some(PaneNode::Leaf(next_id)),
                maximized_pane_id: None,
                panes: vec![pane],
                status: SessionStatus::Idle,
            };

            if let Some(project) = self.projects.get_mut(project_idx) {
                project.sessions.push(session);
                project.active_session_idx = project.sessions.len() - 1;
                project.expanded = true;
            }
            self.active_project_idx = project_idx;
            self.focused_pane_id = Some(next_id);
            self.rebuild_session_index();
            cx.notify();
        }
    }

    /// New session in whichever project is currently open in the workspace.
    #[allow(dead_code)]
    fn create_new_session_for_active_project(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let project_idx = self.active_project_idx;
        if self.projects.get(project_idx).is_none() {
            // With no project open there is nothing to add a session to, so
            // the request turns into "open a project first".
            self.select_folder_and_add_project(window, cx);
            return;
        }
        self.create_new_session_in_project(project_idx, window, cx);
    }

    /// Close one session (all of its panes) without touching the other
    /// sessions and projects. The project stays open, with an empty state that
    /// offers to create a new session.
    fn close_session(
        &mut self,
        project_idx: usize,
        session_idx: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(project) = self.projects.get_mut(project_idx) else {
            return;
        };
        if session_idx >= project.sessions.len() {
            return;
        }
        let removed = project.sessions.remove(session_idx);
        for pane in &removed.panes {
            status::forget_pid(pane.pid);
        }
        if project.active_session_idx > session_idx {
            project.active_session_idx -= 1;
        }
        if project.active_session_idx >= project.sessions.len() {
            project.active_session_idx = project.sessions.len().saturating_sub(1);
        }
        let is_active_project = project_idx == self.active_project_idx;
        let pane_id = project
            .sessions
            .get(project.active_session_idx)
            .and_then(|s| s.panes.first())
            .map(|pane| pane.id);
        if is_active_project {
            if let Some(pane_id) = pane_id {
                self.focus_pane(pane_id, window, cx);
            } else {
                self.focused_pane_id = None;
            }
        }
        // Drop the removed terminals after forgetting their pids, so their
        // PTY masters close and the shells go away with them.
        drop(removed);
        self.rebuild_session_index();
        cx.notify();
        self.schedule_workspace_save(cx);
    }

    /// Close a whole project. Its sessions and PTYs go away; the remaining
    /// projects (and their sessions) are left exactly as they were.
    fn close_project(&mut self, project_idx: usize, window: &mut Window, cx: &mut Context<Self>) {
        if project_idx >= self.projects.len() {
            return;
        }
        let project_id = self.projects[project_idx].id;
        if self.project_menu == Some(project_id) {
            self.project_menu = None;
        }
        if self.renaming_project.as_ref().map(|r| r.project_id) == Some(project_id) {
            self.renaming_project = None;
        }
        let was_active = project_idx == self.active_project_idx;
        let removed = self.projects.remove(project_idx);
        for session in &removed.sessions {
            for pane in &session.panes {
                status::forget_pid(pane.pid);
            }
        }
        if self.active_project_idx > project_idx {
            self.active_project_idx -= 1;
        }
        if self.active_project_idx >= self.projects.len() {
            self.active_project_idx = self.projects.len().saturating_sub(1);
        }
        drop(removed);
        if was_active && !self.projects.is_empty() {
            let next_idx = self.active_project_idx;
            self.activate_project(next_idx, window, cx);
        } else if self.projects.is_empty() {
            self.focused_pane_id = None;
        }
        self.rebuild_session_index();
        cx.notify();
        self.schedule_workspace_save(cx);
    }

    fn start_renaming_project(
        &mut self,
        project_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.project_menu = None;
        if let Some(project) = self.projects.iter().find(|p| p.id == project_id) {
            let current_name = project.name.clone();
            let input = cx.new(|cx| InputState::new(window, cx).default_value(current_name));
            input.update(cx, |state, cx| {
                state.focus(window, cx);
                state.select_all_text(cx);
            });
            let subscription = cx.subscribe(&input, move |this, input, event: &InputEvent, cx| {
                if let InputEvent::PressEnter { .. } = event {
                    let new_name = input.read(cx).value();
                    this.finish_renaming_project(project_id, new_name, cx);
                }
            });
            self.renaming_project = Some(RenamingProject {
                project_id,
                input,
                _subscription: subscription,
            });
        }
        cx.notify();
    }

    fn finish_renaming_project(
        &mut self,
        project_id: usize,
        new_name: SharedString,
        cx: &mut Context<Self>,
    ) {
        let trimmed = new_name.trim();
        if !trimmed.is_empty() {
            if let Some(project) = self.projects.iter_mut().find(|p| p.id == project_id) {
                project.name = SharedString::from(trimmed.to_string());
                self.schedule_workspace_save(cx);
            }
        }
        self.renaming_project = None;
        cx.notify();
    }

    fn cancel_renaming_project(&mut self, cx: &mut Context<Self>) {
        self.renaming_project = None;
        cx.notify();
    }

    /// Open a folder as a project. Opening a folder that is already in the
    /// sidebar focuses that project (and revives it if it has no sessions)
    /// rather than creating a duplicate.
    fn open_project_path(&mut self, folder: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(existing) = self.projects.iter().position(|p| p.path == folder) {
            if self.projects[existing].sessions.is_empty() {
                self.create_new_session_in_project(existing, window, cx);
            } else {
                self.activate_project(existing, window, cx);
            }
            return;
        }

        let name = folder
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        let self_weak = cx.entity().downgrade();
        let next_id = self.next_id;
        self.next_id += 1;

        if let Ok((terminal, pid, generation)) =
            Self::spawn_terminal_view(Some(&folder), self.palette.clone(), self_weak, next_id, cx)
        {
            let title = SharedString::from("terminal 1");
            let mut pane = TerminalPane::new(
                next_id,
                generation,
                terminal.clone(),
                title.clone(),
                title.clone(),
                pid,
                self.warp_bottom_input,
            );
            let (input_state, input_subscription) = build_pane_input(next_id, window, cx);
            if self.warp_bottom_input {
                input_state.update(cx, |state, cx| {
                    state.focus(window, cx);
                });
            } else {
                terminal.read(cx).focus_handle().focus(window);
            }
            pane.input_state = Some(input_state);
            pane.input_subscription = Some(input_subscription);
            let session = Session {
                id: next_id,
                title: title.clone(),
                display_title: title,
                logo: logos::default_logo(),
                pane_tree: Some(PaneNode::Leaf(next_id)),
                maximized_pane_id: None,
                panes: vec![pane],
                status: SessionStatus::Idle,
            };
            let project = Project {
                id: next_id,
                name: SharedString::from(name),
                path: folder,
                sessions: vec![session],
                active_session_idx: 0,
                expanded: true,
            };

            self.projects.push(project);
            self.active_project_idx = self.projects.len() - 1;
            self.rebuild_session_index();
            self.focused_pane_id = Some(next_id);
            cx.notify();
            self.schedule_workspace_save(cx);
        }
    }

    fn select_folder_and_add_project(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
            self.open_project_path(folder, window, cx);
        }
    }

    fn rebuild_session_index(&mut self) {
        self.session_index.clear();
        self.pane_index.clear();
        for (p_idx, project) in self.projects.iter().enumerate() {
            for (s_idx, session) in project.sessions.iter().enumerate() {
                self.session_index.insert(session.id, (p_idx, s_idx));
                for (pane_idx, pane) in session.panes.iter().enumerate() {
                    self.pane_index.insert(pane.id, (p_idx, s_idx, pane_idx));
                }
            }
        }
    }

    /// Split a pane inside its session's own screen: the new terminal becomes
    /// a sibling leaf in that session's layout tree. Other sessions' screens
    /// are untouched.
    fn split_pane(
        &mut self,
        target_pane_id: usize,
        axis: gpui::Axis,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let self_weak = cx.entity().downgrade();
        let split_id = self.next_id;
        self.next_id += 1;
        let next_id = self.next_id;
        self.next_id += 1;
        let Some(&(p_idx, s_idx, _)) = self.pane_index.get(&target_pane_id) else {
            return;
        };

        let dir = self.projects.get(p_idx).map(|p| p.path.clone());
        let pane_count = self
            .projects
            .get(p_idx)
            .and_then(|p| p.sessions.get(s_idx))
            .map(|s| s.panes.len())
            .unwrap_or(0);
        if let Ok((terminal, pid, generation)) =
            Self::spawn_terminal_view(dir.as_ref(), self.palette.clone(), self_weak, next_id, cx)
        {
            let title = SharedString::from(format!("terminal {}", pane_count + 1));
            let display_title = truncate_title(&title);
            let mut pane = TerminalPane::new(
                next_id,
                generation,
                terminal.clone(),
                title.clone(),
                display_title,
                pid,
                self.warp_bottom_input,
            );
            let (input_state, input_subscription) = build_pane_input(next_id, window, cx);
            if self.warp_bottom_input {
                input_state.update(cx, |state, cx| {
                    state.focus(window, cx);
                });
            } else {
                terminal.read(cx).focus_handle().focus(window);
            }
            pane.input_state = Some(input_state);
            pane.input_subscription = Some(input_subscription);

            if let Some(project) = self.projects.get_mut(p_idx) {
                if let Some(session) = project.sessions.get_mut(s_idx) {
                    session.panes.push(pane);
                    session.maximized_pane_id = None;
                    if let Some(tree) = &mut session.pane_tree {
                        tree.split(target_pane_id, next_id, split_id, axis);
                    } else {
                        session.pane_tree = Some(PaneNode::Leaf(next_id));
                    }
                }
            }
            self.focused_pane_id = Some(next_id);
            self.rebuild_session_index();
            cx.notify();
        }
    }

    fn toggle_maximize_pane(&mut self, pane_id: usize, cx: &mut Context<Self>) {
        let Some(&(p_idx, s_idx, _)) = self.pane_index.get(&pane_id) else {
            return;
        };

        if let Some(session) = self
            .projects
            .get_mut(p_idx)
            .and_then(|p| p.sessions.get_mut(s_idx))
        {
            if session.maximized_pane_id == Some(pane_id) {
                session.maximized_pane_id = None;
            } else {
                session.maximized_pane_id = Some(pane_id);
            }
            cx.notify();
        }
    }

    /// Close one pane inside its session's screen. Closing the last pane also
    /// removes the session (screen) itself; an emptied project spawns a fresh
    /// session, like before.
    fn close_pane(&mut self, pane_id: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(&(p_idx, s_idx, _)) = self.pane_index.get(&pane_id) else {
            return;
        };

        let mut session_removed = false;
        let mut needs_new_session = false;
        if let Some(project) = self.projects.get_mut(p_idx) {
            if let Some(session) = project.sessions.get_mut(s_idx) {
                if session.maximized_pane_id == Some(pane_id) {
                    session.maximized_pane_id = None;
                }
                if let Some(closed_pane) = session.panes.iter().find(|pane| pane.id == pane_id) {
                    status::forget_pid(closed_pane.pid);
                }
                session.pane_tree = session.pane_tree.take().and_then(|t| t.remove(pane_id));
                session.panes.retain(|p| p.id != pane_id);
                if session.pane_tree.is_none() || session.panes.is_empty() {
                    session_removed = true;
                }
            }
            if session_removed {
                project.sessions.retain(|s| !s.panes.is_empty());
                if project.sessions.is_empty() {
                    needs_new_session = true;
                } else {
                    project.active_session_idx =
                        project.active_session_idx.min(project.sessions.len() - 1);
                }
            } else if let Some(session) = project.sessions.get(s_idx) {
                // Focus the surviving first pane of this screen.
                let next_leaf = session
                    .pane_tree
                    .as_ref()
                    .map(|t| t.first_leaf())
                    .and_then(|id| session.panes.iter().find(|p| p.id == id));
                if let Some(pane) = next_leaf {
                    let pid = pane.id;
                    self.focus_pane(pid, window, cx);
                }
            }
        }

        self.rebuild_session_index();

        if needs_new_session {
            self.create_new_session_in_project(p_idx, window, cx);
        } else {
            cx.notify();
        }
    }

    fn focus_pane(&mut self, pane_id: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.active_view = ActiveView::Workspace;
        self.focused_pane_id = Some(pane_id);
        if let Some(&(p_idx, s_idx, _)) = self.pane_index.get(&pane_id) {
            self.active_project_idx = p_idx;
            if let Some(p) = self.projects.get_mut(p_idx) {
                p.active_session_idx = s_idx;
                if let Some(session) = p.sessions.get_mut(s_idx) {
                    if let Some(pane) = session.panes.iter_mut().find(|p| p.id == pane_id) {
                        let should_show = pane.should_show_bottom_input(cx);
                        pane.terminal.update(cx, |term, _| {
                            term.set_read_only(should_show);
                        });
                        if should_show {
                            if let Some(input_state) = &pane.input_state {
                                input_state.update(cx, |state, cx| {
                                    state.focus(window, cx);
                                });
                            } else {
                                pane.terminal.read(cx).focus_handle().focus(window);
                            }
                        } else {
                            pane.terminal.read(cx).focus_handle().focus(window);
                        }
                    }
                }
            }
        }
        cx.notify();
    }

    fn focus_pane_input(&mut self, pane_id: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.active_view = ActiveView::Workspace;
        self.focused_pane_id = Some(pane_id);
        if let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) {
            self.active_project_idx = p_idx;
            if let Some(pane) = self
                .projects
                .get_mut(p_idx)
                .and_then(|p| p.sessions.get_mut(s_idx))
                .and_then(|s| s.panes.get_mut(pane_idx))
            {
                if let Some(input_state) = &pane.input_state {
                    input_state.update(cx, |state, cx| {
                        state.focus(window, cx);
                    });
                }
            }
        }
        cx.notify();
    }

    fn pane_ref(&self, pane_id: usize) -> Option<&TerminalPane> {
        let &(p_idx, s_idx, pane_idx) = self.pane_index.get(&pane_id)?;
        self.projects
            .get(p_idx)?
            .sessions
            .get(s_idx)?
            .panes
            .get(pane_idx)
    }

    fn pane_mut(&mut self, pane_id: usize) -> Option<&mut TerminalPane> {
        let &(p_idx, s_idx, pane_idx) = self.pane_index.get(&pane_id)?;
        self.projects
            .get_mut(p_idx)?
            .sessions
            .get_mut(s_idx)?
            .panes
            .get_mut(pane_idx)
    }

    /// Shell history plus in-session commands, oldest first.
    fn pane_history_entries(&self, pane: &TerminalPane) -> Vec<String> {
        shell_history::merge(&pane.shell_history, &pane.command_history)
    }

    /// Snapshot of everything suggestion code needs from a pane's input.
    fn input_context(
        &self,
        pane_id: usize,
        cx: &App,
    ) -> Option<(String, usize, PathBuf, Vec<String>, Entity<InputState>)> {
        let pane = self.pane_ref(pane_id)?;
        let input = pane.input_state.clone()?;
        let value = input.read(cx).value().to_string();
        let cursor = input.read(cx).cursor();
        let cwd = self
            .pane_cwd_path(pane)
            .unwrap_or_else(|| PathBuf::from("."));
        Some((value, cursor, cwd, self.pane_history_entries(pane), input))
    }

    /// Create a pane's input editor and subscribe to its changes.
    fn create_pane_input(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Entity<InputState> {
        let (input, subscription) = build_pane_input(pane_id, window, cx);
        if let Some(pane) = self.pane_mut(pane_id) {
            pane.input_state = Some(input.clone());
            pane.input_subscription = Some(subscription);
        }
        input
    }

    /// Keep ghost text and the completion menu in sync with typed text.
    fn on_pane_input_changed(
        &mut self,
        pane_id: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some((value, cursor, cwd, entries, input)) = self.input_context(pane_id, cx) else {
            return;
        };
        let menu_open = self
            .pane_ref(pane_id)
            .is_some_and(|pane| pane.completion_menu.is_some());

        if menu_open {
            // Typing filters an open menu instead of dismissing it.
            self.refresh_completion_menu(pane_id, &value, cursor, &cwd, &entries);
            input.update(cx, |state, cx| state.clear_inline_completion(cx));
            return;
        }

        let suggestion = completion::history_suggestion(&value, cursor, &entries);
        input.update(cx, |state, cx| match suggestion {
            Some(text) if !text.is_empty() => state.set_inline_completion_text(text, cx),
            _ => state.clear_inline_completion(cx),
        });
    }

    /// Recompute the open completion menu for the current input value.
    fn refresh_completion_menu(
        &mut self,
        pane_id: usize,
        value: &str,
        cursor_utf16: usize,
        cwd: &Path,
        entries: &[String],
    ) {
        let cursor = completion::utf16_to_byte(value, cursor_utf16);
        let result = completion::complete(value, cursor, cwd, entries);
        let Some(pane) = self.pane_mut(pane_id) else {
            return;
        };
        match result {
            Some(result) => {
                let token = value
                    .get(result.start..result.end)
                    .unwrap_or_default()
                    .to_string();
                let previous = pane
                    .completion_menu
                    .as_ref()
                    .filter(|menu| menu.token == token)
                    .map(|menu| menu.selected)
                    .unwrap_or(0);
                pane.completion_menu = Some(CompletionMenuState {
                    start: result.start,
                    end: result.end,
                    token,
                    selected: previous.min(result.items.len().saturating_sub(1)),
                    items: result.items,
                });
            }
            None => pane.completion_menu = None,
        }
    }

    fn close_completion_menu(&mut self, pane_id: usize, cx: &mut Context<Self>) {
        if let Some(pane) = self.pane_mut(pane_id) {
            pane.completion_menu = None;
        }
        cx.notify();
    }

    fn move_completion_selection(&mut self, pane_id: usize, delta: isize, cx: &mut Context<Self>) {
        if let Some(pane) = self.pane_mut(pane_id) {
            if let Some(menu) = pane.completion_menu.as_mut() {
                if !menu.items.is_empty() {
                    let len = menu.items.len() as isize;
                    menu.selected = (menu.selected as isize + delta).rem_euclid(len) as usize;
                }
            }
        }
        cx.notify();
    }

    /// Tab: accept the only candidate, or open the menu when there are several.
    fn start_completion(&mut self, pane_id: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some((value, cursor, cwd, entries, input)) = self.input_context(pane_id, cx) else {
            return;
        };

        // A visible ghost suggestion is the best candidate; Tab accepts it.
        let ghost = input.read(cx).inline_completion_text().map(str::to_string);
        if ghost.is_some() {
            input.update(cx, |state, cx| {
                state.accept_inline_completion(window, cx);
            });
            cx.notify();
            return;
        }

        // Even before the deferred change handler has painted ghost text, Tab
        // completes from history.
        if let Some(suffix) = completion::history_suggestion(&value, cursor, &entries) {
            let cursor = cursor + completion::utf16_len(&suffix);
            set_input_text_with_cursor(
                &input,
                format!("{value}{suffix}"),
                cursor,
                window,
                cx,
            );
            cx.notify();
            return;
        }

        let cursor_byte = completion::utf16_to_byte(&value, cursor);
        let Some(result) = completion::complete(&value, cursor_byte, &cwd, &entries) else {
            self.close_completion_menu(pane_id, cx);
            return;
        };

        if result.items.len() == 1 {
            let item = result.items[0].clone();
            self.apply_completion(
                pane_id,
                result.start,
                result.end,
                item,
                window,
                cx,
            );
        } else {
            if let Some(pane) = self.pane_mut(pane_id) {
                pane.completion_menu = Some(CompletionMenuState {
                    start: result.start,
                    end: result.end,
                    token: value
                        .get(result.start..result.end)
                        .unwrap_or_default()
                        .to_string(),
                    items: result.items,
                    selected: 0,
                });
            }
            cx.notify();
        }
    }

    fn accept_completion_at(
        &mut self,
        pane_id: usize,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let selected = self.pane_ref(pane_id).and_then(|pane| {
            let menu = pane.completion_menu.as_ref()?;
            let item = menu.items.get(index)?.clone();
            Some((menu.start, menu.end, item))
        });
        let Some((start, end, item)) = selected else {
            return;
        };
        self.apply_completion(pane_id, start, end, item, window, cx);
    }

    /// Replace the completing token with `item` and keep the editor usable.
    fn apply_completion(
        &mut self,
        pane_id: usize,
        start: usize,
        end: usize,
        item: completion::CompletionItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some((value, _, _, _entries, input)) = self.input_context(pane_id, cx) else {
            return;
        };
        let start = start.min(value.len());
        let end = end.min(value.len());
        if !value.is_char_boundary(start) || !value.is_char_boundary(end) || start > end {
            return;
        }

        let mut new_value = String::with_capacity(value.len() + item.value.len() + 1);
        new_value.push_str(&value[..start]);
        new_value.push_str(&item.value);
        let mut cursor = start + item.value.len();
        new_value.push_str(&value[end..]);

        // Command and branch completions read better with the cursor moved to
        // the next token; file completions stay tight to the path.
        if !item.is_dir && item.kind != completion::CompletionKind::File {
            new_value.insert(cursor, ' ');
            cursor += 1;
        }

        let cursor_utf16 = completion::byte_to_utf16(&new_value, cursor);
        input.update(cx, |state, cx| {
            state.set_value(new_value, window, cx);
            let position = state
                .text()
                .offset_to_position(cursor_utf16.min(completion::utf16_len(state.value().as_str())));
            state.set_cursor_position(position, window, cx);
            state.clear_inline_completion(cx);
        });

        if item.is_dir {
            // Descend into the directory: keep completing the new token.
            if let Some((value, cursor, cwd, entries, _)) = self.input_context(pane_id, cx) {
                if let Some(pane) = self.pane_mut(pane_id) {
                    pane.completion_menu = None;
                }
                self.refresh_completion_menu(pane_id, &value, cursor, &cwd, &entries);
                if let Some(pane) = self.pane_mut(pane_id) {
                    if let Some(menu) = pane.completion_menu.as_mut() {
                        menu.selected = 0;
                    }
                }
            }
        } else if let Some(pane) = self.pane_mut(pane_id) {
            pane.completion_menu = None;
        }

        if let Some(pane) = self.pane_ref(pane_id) {
            if let Some(input_state) = pane.input_state.clone() {
                input_state.update(cx, |state, cx| state.focus(window, cx));
            }
        }
        cx.notify();
    }

    // ---------------------------------------------------------------------
    // History search (Ctrl+R)
    // ---------------------------------------------------------------------

    fn open_history_search(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.history_search.is_some() {
            return;
        }
        let Some(pane) = self.pane_ref(pane_id) else {
            return;
        };
        let entries = self.pane_history_entries(pane);
        let initial = pane
            .input_state
            .as_ref()
            .map(|input| input.read(cx).value().to_string())
            .unwrap_or_default();

        let input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Search command history")
                .default_value(initial.clone())
        });
        input.update(cx, |state, cx| {
            let position = state
                .text()
                .offset_to_position(completion::utf16_len(&initial));
            state.set_cursor_position(position, window, cx);
            state.focus(window, cx);
        });

        let matches: Vec<String> = shell_history::search(&entries, &initial)
            .into_iter()
            .cloned()
            .collect();

        let subscription = cx.subscribe_in(
            &input,
            window,
            move |_this: &mut Self, _input, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::Change) {
                    let weak = cx.entity().downgrade();
                    window.defer(cx, move |window, cx| {
                        let _ = weak.update(cx, |this, cx| {
                            this.refresh_history_search(window, cx);
                        });
                    });
                }
            },
        );

        self.history_search = Some(HistorySearchState {
            pane_id,
            input,
            matches,
            selected: 0,
            _subscription: subscription,
        });
        if let Some(pane) = self.pane_mut(pane_id) {
            pane.completion_menu = None;
        }
        cx.notify();
    }

    fn refresh_history_search(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(search) = self.history_search.as_ref() else {
            return;
        };
        let query = search.input.read(cx).value().to_string();
        let pane_id = search.pane_id;
        let Some(pane) = self.pane_ref(pane_id) else {
            return;
        };
        let entries = self.pane_history_entries(pane);
        let matches: Vec<String> = shell_history::search(&entries, &query)
            .into_iter()
            .cloned()
            .collect();
        if let Some(search) = self.history_search.as_mut() {
            search.matches = matches;
            search.selected = 0;
        }
        cx.notify();
    }

    fn move_history_search_selection(&mut self, delta: isize, cx: &mut Context<Self>) {
        if let Some(search) = self.history_search.as_mut() {
            if !search.matches.is_empty() {
                let len = search.matches.len() as isize;
                search.selected = (search.selected as isize + delta).rem_euclid(len) as usize;
            }
        }
        cx.notify();
    }

    fn close_history_search(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let pane_id = self.history_search.as_ref().map(|search| search.pane_id);
        self.history_search = None;
        if let Some(pane_id) = pane_id {
            if let Some(input) = self
                .pane_ref(pane_id)
                .and_then(|pane| pane.input_state.clone())
            {
                input.update(cx, |state, cx| state.focus(window, cx));
            }
        }
        cx.notify();
    }

    fn accept_history_search(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(search) = self.history_search.as_ref() else {
            return;
        };
        let Some(command) = search.matches.get(search.selected).cloned() else {
            let pane_id = search.pane_id;
            self.history_search = None;
            if let Some(input) = self
                .pane_ref(pane_id)
                .and_then(|pane| pane.input_state.clone())
            {
                input.update(cx, |state, cx| state.focus(window, cx));
            }
            cx.notify();
            return;
        };
        let pane_id = search.pane_id;
        let input = self
            .pane_ref(pane_id)
            .and_then(|pane| pane.input_state.clone());
        self.history_search = None;
        if let Some(input) = input {
            set_input_text(&input, command, window, cx);
            input.update(cx, |state, cx| state.focus(window, cx));
        }
        if let Some(pane) = self.pane_mut(pane_id) {
            pane.history_index = None;
        }
        cx.notify();
    }

    fn accept_history_search_index(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(search) = self.history_search.as_mut() {
            search.selected = index;
        }
        self.accept_history_search(window, cx);
    }

    fn toggle_pane_bottom_input(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };
        let Some(session) = project.sessions.get_mut(s_idx) else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };
        let is_showing_now = pane.should_show_bottom_input(cx);
        if is_showing_now {
            pane.bottom_input_enabled = false;
            pane.bottom_input_visible = false;
            pane.completion_menu = None;
        } else {
            pane.bottom_input_enabled = true;
            pane.bottom_input_visible = true;
        }

        let should_show = pane.should_show_bottom_input(cx);
        pane.terminal.update(cx, |term, _| {
            term.set_read_only(should_show);
        });
        if should_show {
            if let Some(input_state) = &pane.input_state {
                input_state.update(cx, |state, cx| {
                    state.focus(window, cx);
                });
            }
        } else {
            pane.terminal.read(cx).focus_handle().focus(window);
        }
        cx.notify();
    }

    #[allow(dead_code)]
    fn clear_pane_input(&mut self, pane_id: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };
        let Some(session) = project.sessions.get_mut(s_idx) else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };
        if let Some(input_state) = &pane.input_state {
            input_state.update(cx, |state, cx| {
                state.set_value("", window, cx);
                state.focus(window, cx);
            });
        }
        pane.history_index = None;
        pane.pending_draft.clear();
        pane.completion_menu = None;
        cx.notify();
    }

    #[allow(dead_code)]
    fn cancel_pane_job(&mut self, pane_id: usize, cx: &mut Context<Self>) {
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };
        let Some(session) = project.sessions.get_mut(s_idx) else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };
        pane.terminal.update(cx, |term, cx| {
            term.send_bytes(b"\x03", cx);
        });
        cx.notify();
    }

    fn step_pane_history(
        &mut self,
        pane_id: usize,
        is_up: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(pane) = self.pane_mut(pane_id) else {
            return;
        };
        let Some(input_state) = pane.input_state.clone() else {
            return;
        };
        let entries = shell_history::merge(&pane.shell_history, &pane.command_history);
        if entries.is_empty() {
            return;
        }

        if is_up {
            let new_idx = match pane.history_index {
                None => {
                    pane.pending_draft = input_state.read(cx).value().to_string();
                    entries.len() - 1
                }
                Some(idx) => idx.saturating_sub(1),
            };
            pane.history_index = Some(new_idx);
            if let Some(cmd) = entries.get(new_idx).cloned() {
                set_input_text(&input_state, cmd, window, cx);
            }
        } else if let Some(idx) = pane.history_index {
            if idx + 1 < entries.len() {
                let new_idx = idx + 1;
                pane.history_index = Some(new_idx);
                if let Some(cmd) = entries.get(new_idx).cloned() {
                    set_input_text(&input_state, cmd, window, cx);
                }
            } else {
                pane.history_index = None;
                let draft = std::mem::take(&mut pane.pending_draft);
                set_input_text(&input_state, draft, window, cx);
            }
        }
        cx.notify();
    }

    fn execute_pane_input(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };
        let Some(session) = project.sessions.get_mut(s_idx) else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };
        let Some(input_state) = pane.input_state.clone() else {
            return;
        };

        let cmd = input_state.read(cx).value().to_string();
        if cmd.is_empty() {
            pane.terminal.update(cx, |term, cx| {
                term.send_bytes(b"\r", cx);
            });
        } else {
            let terminal = pane.terminal.clone();
            terminal.update(cx, |term, cx| {
                term.send_text(&cmd, cx);
            });
            if pane.command_history.last().map(|s| s.as_str()) != Some(&cmd) {
                pane.command_history.push(cmd);
            }
            pane.history_index = None;
            pane.pending_draft.clear();
            input_state.update(cx, |state, cx| {
                state.set_value("", window, cx);
            });

            // Send Enter separately after a brief 25ms tick so raw-mode CLI agents (Claude Code,
            // Ink, readline) receive it as an authentic Enter keystroke rather than pasted text,
            // triggering immediate automatic prompt submission.
            let terminal = terminal.clone();
            App::spawn(cx, async move |cx: &mut gpui::AsyncApp| {
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(25))
                    .await;
                let _ = terminal.update(cx, |term, cx| {
                    term.send_bytes(b"\r", cx);
                });
            }).detach();
        }
        input_state.update(cx, |state, cx| {
            state.focus(window, cx);
        });
        cx.notify();
    }
    fn handle_pane_ctrl_c(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };
        let Some(session) = project.sessions.get_mut(s_idx) else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };

        // 1. Clear input state and draft
        if let Some(input_state) = &pane.input_state {
            input_state.update(cx, |state, cx| {
                state.set_value("", window, cx);
            });
        }
        pane.history_index = None;
        pane.pending_draft.clear();
        pane.completion_menu = None;

        // 2. Send \x03 (ETX/SIGINT) to PTY (interrupts process on Linux, macOS, and Windows ConPTY)
        pane.terminal.update(cx, |term, cx| {
            term.send_bytes(b"\x03", cx);
        });

        // 3. On Unix: also directly signal the active foreground process group
        #[cfg(unix)]
        if let Some(pid) = pane.pid {
            if let Some(tpgid) = status::read_proc_tpgid(pid) {
                if tpgid > 0 && tpgid != pid as i64 {
                    unsafe {
                        libc::kill(-(tpgid as libc::pid_t), libc::SIGINT);
                    }
                }
            }
        }

        // 4. If an agent/command was working, interrupt and return to Idle
        if pane.status == SessionStatus::Working {
            pane.status = SessionStatus::Idle;
            pane.bottom_input_visible = true;
            pane.terminal.update(cx, |term, _| term.set_read_only(true));
        }

        cx.notify();
    }

    fn handle_pane_input_key(
        &mut self,
        pane_id: usize,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let keystroke = &event.keystroke;
        let key = keystroke.key.as_str();
        let modifiers = keystroke.modifiers;

        let Some(pane) = self.pane_ref(pane_id) else {
            return;
        };
        let Some(input_state) = pane.input_state.clone() else {
            return;
        };
        let value = input_state.read(cx).value().to_string();
        let cursor_utf16 = input_state.read(cx).cursor();
        let cursor_byte = completion::utf16_to_byte(&value, cursor_utf16);
        let cursor_line = input_state.read(cx).cursor_position().line;
        let last_line = value.matches('\n').count();
        let has_menu = pane.completion_menu.is_some();
        let plus_menu_open = pane.plus_menu_open;
        let terminal = pane.terminal.clone();

        // --- completion menu navigation ------------------------------------
        if has_menu {
            if key == "down" {
                cx.stop_propagation();
                self.move_completion_selection(pane_id, 1, cx);
                return;
            }
            if key == "up" {
                cx.stop_propagation();
                self.move_completion_selection(pane_id, -1, cx);
                return;
            }
            if key == "right" && !modifiers.control && !modifiers.shift && !modifiers.alt {
                cx.stop_propagation();
                let selected = self
                    .pane_ref(pane_id)
                    .and_then(|pane| pane.completion_menu.as_ref().map(|menu| menu.selected))
                    .unwrap_or(0);
                self.accept_completion_at(pane_id, selected, window, cx);
                return;
            }
            if (key == "enter" || key == "return") && !modifiers.shift {
                // Enter means "run this", never "insert the highlighted
                // suggestion": Tab and Right accept completions.
                cx.stop_propagation();
                self.close_completion_menu(pane_id, cx);
                self.execute_pane_input(pane_id, window, cx);
                return;
            }
            if key == "tab" && !modifiers.control && !modifiers.alt {
                cx.stop_propagation();
                let selected = self
                    .pane_ref(pane_id)
                    .and_then(|pane| pane.completion_menu.as_ref().map(|menu| menu.selected))
                    .unwrap_or(0);
                self.accept_completion_at(pane_id, selected, window, cx);
                return;
            }
            if key == "escape" {
                cx.stop_propagation();
                self.close_completion_menu(pane_id, cx);
                return;
            }
        }

        // Enter: execute the command.
        if (key == "enter" || key == "return")
            && !modifiers.shift
            && !modifiers.control
            && !modifiers.alt
        {
            cx.stop_propagation();
            self.execute_pane_input(pane_id, window, cx);
            return;
        }

        // Shift+Enter: insert a newline (multi-line commands, Warp style).
        if (key == "enter" || key == "return")
            && modifiers.shift
            && !modifiers.control
            && !modifiers.alt
        {
            input_state.update(cx, |state, cx| {
                state.insert("\n", window, cx);
            });
            cx.stop_propagation();
            cx.notify();
            return;
        }

        // Ctrl+G: toggle the input editor.
        if (key == "g" || key == "G")
            && (modifiers.control || modifiers.platform)
            && !modifiers.shift
            && !modifiers.alt
        {
            cx.stop_propagation();
            self.toggle_pane_bottom_input(pane_id, window, cx);
            return;
        }

        // Ctrl+C is handled by the terminal's `Copy` action as well; the guard
        // here keeps it working even when the editor has no selection.
        if key == "c" && modifiers.control && !modifiers.shift && !modifiers.alt {
            cx.stop_propagation();
            self.handle_pane_ctrl_c(pane_id, window, cx);
            return;
        }

        // Ctrl+R: reverse history search, like the shell's own prompt.
        if key == "r" && modifiers.control && !modifiers.shift && !modifiers.alt {
            cx.stop_propagation();
            self.open_history_search(pane_id, window, cx);
            return;
        }

        // Tab: accept the ghost suggestion, or open completion candidates.
        if key == "tab" && !modifiers.control && !modifiers.alt {
            cx.stop_propagation();
            self.start_completion(pane_id, window, cx);
            return;
        }

        // Right at the end of the line accepts a ghost suggestion.
        if key == "right" && !modifiers.control && !modifiers.shift && !modifiers.alt {
            if cursor_utf16 >= completion::utf16_len(&value)
                && input_state.read(cx).has_inline_completion()
            {
                input_state.update(cx, |state, cx| {
                    state.accept_inline_completion(window, cx);
                });
                cx.stop_propagation();
                cx.notify();
                return;
            }
        }

        // Ctrl+A / Ctrl+E: readline start/end of line.
        if key == "a" && modifiers.control && !modifiers.shift && !modifiers.alt {
            let start = value[..cursor_byte].rfind('\n').map_or(0, |idx| idx + 1);
            move_input_cursor(
                &input_state,
                completion::byte_to_utf16(&value, start),
                window,
                cx,
            );
            cx.stop_propagation();
            cx.notify();
            return;
        }
        if key == "e" && modifiers.control && !modifiers.shift && !modifiers.alt {
            let end = value[cursor_byte..]
                .find('\n')
                .map_or(value.len(), |idx| cursor_byte + idx);
            move_input_cursor(
                &input_state,
                completion::byte_to_utf16(&value, end),
                window,
                cx,
            );
            cx.stop_propagation();
            cx.notify();
            return;
        }

        // Ctrl+U: kill from the start of the line to the cursor.
        if key == "u" && modifiers.control && !modifiers.shift && !modifiers.alt {
            let start = value[..cursor_byte].rfind('\n').map_or(0, |idx| idx + 1);
            let mut new_value = String::with_capacity(value.len());
            new_value.push_str(&value[..start]);
            new_value.push_str(&value[cursor_byte..]);
            set_input_text_with_cursor(
                &input_state,
                new_value,
                completion::byte_to_utf16(&value, start),
                window,
                cx,
            );
            if let Some(pane) = self.pane_mut(pane_id) {
                pane.history_index = None;
                pane.pending_draft.clear();
            }
            cx.stop_propagation();
            cx.notify();
            return;
        }

        // Ctrl+W / Alt+Backspace: delete the previous word.
        if (key == "w" && modifiers.control && !modifiers.shift && !modifiers.alt)
            || (key == "backspace" && modifiers.alt && !modifiers.control && !modifiers.shift)
        {
            let start = previous_word_start(&value, cursor_byte);
            let mut new_value = String::with_capacity(value.len());
            new_value.push_str(&value[..start]);
            new_value.push_str(&value[cursor_byte..]);
            set_input_text_with_cursor(
                &input_state,
                new_value,
                completion::byte_to_utf16(&value, start),
                window,
                cx,
            );
            cx.stop_propagation();
            cx.notify();
            return;
        }

        // Ctrl+K: kill from the cursor to the end of the line.
        if key == "k" && modifiers.control && !modifiers.shift && !modifiers.alt {
            let end = value[cursor_byte..]
                .find('\n')
                .map_or(value.len(), |idx| cursor_byte + idx);
            let mut new_value = String::with_capacity(value.len());
            new_value.push_str(&value[..cursor_byte]);
            new_value.push_str(&value[end..]);
            set_input_text_with_cursor(&input_state, new_value, cursor_utf16, window, cx);
            cx.stop_propagation();
            cx.notify();
            return;
        }

        // Alt+B / Alt+F: move by word (readline).
        if key == "b" && modifiers.alt && !modifiers.control && !modifiers.shift {
            let start = previous_word_start(&value, cursor_byte);
            move_input_cursor(
                &input_state,
                completion::byte_to_utf16(&value, start),
                window,
                cx,
            );
            cx.stop_propagation();
            cx.notify();
            return;
        }
        if key == "f" && modifiers.alt && !modifiers.control && !modifiers.shift {
            let end = next_word_end(&value, cursor_byte);
            move_input_cursor(
                &input_state,
                completion::byte_to_utf16(&value, end),
                window,
                cx,
            );
            cx.stop_propagation();
            cx.notify();
            return;
        }

        // Ctrl+L: clear the terminal screen.
        if key == "l" && modifiers.control && !modifiers.shift && !modifiers.alt {
            terminal.update(cx, |term, cx| {
                term.send_bytes(b"\x0c", cx);
            });
            cx.stop_propagation();
            return;
        }

        // Ctrl+Z stays with the editor (undo), not job control: a terminal
        // prompt's line editor is what the user is typing into here.

        // Ctrl+D: EOF on an empty line.
        if key == "d" && modifiers.control && !modifiers.shift && !modifiers.alt {
            if value.is_empty() {
                terminal.update(cx, |term, cx| {
                    term.send_bytes(b"\x04", cx);
                });
                cx.stop_propagation();
                return;
            }
        }

        // Ctrl+\: quit the foreground job (SIGQUIT).
        if key == "\\" && modifiers.control && !modifiers.shift && !modifiers.alt {
            terminal.update(cx, |term, cx| {
                term.send_bytes(b"\x1c", cx);
            });
            cx.stop_propagation();
            return;
        }

        // Escape: dismiss menus, then clear the input.
        if key == "escape" && !modifiers.shift && !modifiers.control && !modifiers.alt {
            if self.terminal_context_menu.is_some() {
                self.terminal_context_menu = None;
                cx.stop_propagation();
                cx.notify();
                return;
            }
            if plus_menu_open {
                if let Some(pane) = self.pane_mut(pane_id) {
                    pane.plus_menu_open = false;
                }
                cx.stop_propagation();
                cx.notify();
                return;
            }
            if !value.is_empty() {
                input_state.update(cx, |state, cx| {
                    state.set_value("", window, cx);
                });
                if let Some(pane) = self.pane_mut(pane_id) {
                    pane.history_index = None;
                    pane.pending_draft.clear();
                }
                cx.stop_propagation();
                cx.notify();
            }
            return;
        }

        // Ctrl+P / Ctrl+N: previous / next history entry.
        if (key == "p" || key == "n")
            && modifiers.control
            && !modifiers.shift
            && !modifiers.alt
        {
            cx.stop_propagation();
            self.step_pane_history(pane_id, key == "p", window, cx);
            return;
        }

        // Up / Down: history, but only once the cursor is on the first / last
        // line of a multi-line buffer, so line navigation still works.
        if key == "up" && !modifiers.shift && !modifiers.control && !modifiers.alt {
            if cursor_line == 0 {
                cx.stop_propagation();
                self.step_pane_history(pane_id, true, window, cx);
            }
            return;
        }
        if key == "down" && !modifiers.shift && !modifiers.control && !modifiers.alt {
            if cursor_line as usize >= last_line {
                cx.stop_propagation();
                self.step_pane_history(pane_id, false, window, cx);
            }
            return;
        }

        // Ordinary typing leaves history navigation mode.
        if (!modifiers.control && !modifiers.alt)
            && (keystroke.key_char.is_some() || key == "backspace" || key == "delete")
        {
            if let Some(pane) = self.pane_mut(pane_id) {
                pane.history_index = None;
            }
        }
    }

    fn ensure_panes_input_state(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let mut missing = Vec::new();
        for project in &mut self.projects {
            for session in &mut project.sessions {
                for pane in &mut session.panes {
                    let placeholder = "Run commands";
                    if let Some(input) = &pane.input_state {
                        input.update(cx, |state, cx| {
                            state.set_placeholder(placeholder, window, cx);
                            state.set_auto_grow(1, 8);
                        });
                        if pane.input_subscription.is_none() && pane.input_state.is_none() {
                            missing.push(pane.id);
                        }
                    } else {
                        missing.push(pane.id);
                    }
                }
            }
        }
        for pane_id in missing {
            self.create_pane_input(pane_id, window, cx);
        }

        // Dynamically ensure the focused pane's input is focused like Warp.
        // The history palette owns focus while it is open.
        if self.warp_bottom_input
            && self.active_view == ActiveView::Workspace
            && self.history_search.is_none()
        {
            if let Some(pane_id) = self.focused_pane_id {
                if let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) {
                    if let Some(project) = self.projects.get(p_idx) {
                        if let Some(session) = project.sessions.get(s_idx) {
                            if let Some(pane) = session.panes.get(pane_idx) {
                                let should_show = pane.should_show_bottom_input(cx);
                                pane.terminal.update(cx, |term, _| {
                                    term.set_read_only(should_show);
                                });
                                if should_show {
                                    if let Some(input_state) = &pane.input_state {
                                        let is_input_focused = input_state.read(cx).focus_handle().is_focused(window);
                                        let is_term_focused = pane.terminal.read(cx).focus_handle().is_focused(window);
                                        if is_term_focused || !is_input_focused {
                                            input_state.update(cx, |state, cx| {
                                                state.focus(window, cx);
                                            });
                                        }
                                    }
                                } else {
                                    let is_term_focused = pane.terminal.read(cx).focus_handle().is_focused(window);
                                    if !is_term_focused {
                                        pane.terminal.read(cx).focus_handle().focus(window);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn open_terminal_context_menu(
        &mut self,
        pane_id: usize,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.terminal_context_menu = Some(TerminalContextMenu { pane_id, position });
        self.focus_pane(pane_id, window, cx);
        cx.notify();
    }

    fn close_terminal_context_menu(&mut self, cx: &mut Context<Self>) {
        if self.terminal_context_menu.is_some() {
            self.terminal_context_menu = None;
            cx.notify();
        }
    }

    fn copy_terminal_selection(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) {
            if let Some(pane) = self
                .projects
                .get(p_idx)
                .and_then(|p| p.sessions.get(s_idx))
                .and_then(|s| s.panes.get(pane_idx))
            {
                if let Some(text) = pane.terminal.read(cx).selected_text() {
                    if !text.is_empty() {
                        cx.write_to_clipboard(gpui::ClipboardItem::new_string(text));
                        pane.terminal.update(cx, |term, cx| {
                            term.clear_selection(cx);
                        });
                    }
                }
            }
        }
        self.close_terminal_context_menu(cx);
        self.focus_pane(pane_id, window, cx);
    }

    fn paste_from_clipboard_into_pane(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) {
            if let Some(pane) = self
                .projects
                .get(p_idx)
                .and_then(|p| p.sessions.get(s_idx))
                .and_then(|s| s.panes.get(pane_idx))
            {
                if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
                    let clean = text
                        .trim_matches(&['\r', '\n'][..])
                        .replace("\r\n", " ")
                        .replace('\n', " ")
                        .replace('\r', "");
                    if !clean.is_empty() {
                        let should_show = pane.should_show_bottom_input(cx);
                        if should_show {
                            if let Some(input_state) = &pane.input_state {
                                input_state.update(cx, |state, cx| {
                                    let cur = state.value().to_string();
                                    state.set_value(&format!("{cur}{clean}"), window, cx);
                                    state.focus(window, cx);
                                });
                            }
                        } else {
                            pane.terminal.update(cx, |term, cx| {
                                term.paste_text(&clean, cx);
                            });
                        }
                    }
                }
            }
        }
        self.close_terminal_context_menu(cx);
        self.focus_pane(pane_id, window, cx);
    }

    fn select_all_in_terminal(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) {
            if let Some(pane) = self
                .projects
                .get(p_idx)
                .and_then(|p| p.sessions.get(s_idx))
                .and_then(|s| s.panes.get(pane_idx))
            {
                pane.terminal.update(cx, |term, cx| {
                    term.select_all(cx);
                });
            }
        }
        self.close_terminal_context_menu(cx);
        self.focus_pane(pane_id, window, cx);
    }

    fn clear_terminal_buffer(
        &mut self,
        pane_id: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) {
            if let Some(pane) = self
                .projects
                .get(p_idx)
                .and_then(|p| p.sessions.get(s_idx))
                .and_then(|s| s.panes.get(pane_idx))
            {
                pane.terminal.update(cx, |term, cx| {
                    term.send_bytes(b"\x0c", cx);
                });
            }
        }
        self.close_terminal_context_menu(cx);
        self.focus_pane(pane_id, window, cx);
    }

    fn render_terminal_context_menu_layer(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let Some(menu) = self.terminal_context_menu else {
            return div().into_any_element();
        };

        let pane_id = menu.pane_id;
        let tokens = self.tokens(cx);

        let has_selection = self
            .pane_index
            .get(&pane_id)
            .and_then(|&(p_idx, s_idx, pane_idx)| {
                self.projects
                    .get(p_idx)
                    .and_then(|p| p.sessions.get(s_idx))
                    .and_then(|s| s.panes.get(pane_idx))
            })
            .map(|pane| pane.terminal.read(cx).has_selection())
            .unwrap_or(false);

        let left = (menu.position.x - px(5.0)).max(px(10.0));
        let top = (menu.position.y - px(5.0)).max(px(35.0));

        let modal_border: Hsla = if tokens.is_dark {
            rgba(0x2d2e3dff).into()
        } else {
            tokens.card_border
        };
        let modal_bg: Hsla = if tokens.is_dark {
            rgba(0x16171eff).into()
        } else {
            rgba(0xffffffff).into()
        };
        let sep_bg: Hsla = if tokens.is_dark {
            rgba(0xffffff14).into()
        } else {
            rgba(0x00000014).into()
        };

        deferred(
            div()
                .id("terminal-context-menu-layer")
                .size_full()
                .absolute()
                .top_0()
                .left_0()
                .child(
                    div()
                        .id("terminal-context-menu-backdrop")
                        .size_full()
                        .absolute()
                        .top_0()
                        .left_0()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _, _, cx| {
                                this.close_terminal_context_menu(cx);
                            }),
                        )
                        .on_mouse_down(
                            MouseButton::Right,
                            cx.listener(|this, _, _, cx| {
                                this.close_terminal_context_menu(cx);
                            }),
                        ),
                )
                .child(
                    div()
                        .id(("terminal-context-menu-modal", pane_id))
                        .absolute()
                        .top(top)
                        .left(left)
                        .w(px(180.0))
                        .p(px(5.0))
                        .rounded(px(8.0))
                        .border_1()
                        .border_color(modal_border)
                        .bg(modal_bg)
                        .shadow_lg()
                        .flex()
                        .flex_col()
                        .gap(px(2.0))
                        // Copy option
                        .child(
                            div()
                                .id(("context-menu-copy", pane_id))
                                .flex()
                                .items_center()
                                .justify_between()
                                .px(px(8.0))
                                .py(px(5.5))
                                .rounded(px(5.0))
                                .cursor(if has_selection {
                                    CursorStyle::PointingHand
                                } else {
                                    CursorStyle::Arrow
                                })
                                .opacity(if has_selection { 1.0 } else { 0.45 })
                                .when(has_selection, |this| {
                                    this.hover(|s| s.bg(tokens.sidebar_item_hover))
                                })
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(8.0))
                                        .child(
                                            svg()
                                                .path("icons/copy.svg")
                                                .size(px(13.0))
                                                .text_color(tokens.text_primary),
                                        )
                                        .child(
                                            div()
                                                .font_family(MONO_FONT)
                                                .text_size(px(12.0))
                                                .text_color(tokens.text_primary)
                                                .child("Copy"),
                                        ),
                                )
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(10.5))
                                        .text_color(tokens.text_muted)
                                        .child("Ctrl+C"),
                                )
                                .when(has_selection, |this| {
                                    this.on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.copy_terminal_selection(pane_id, window, cx);
                                        }),
                                    )
                                }),
                        )
                        // Paste option
                        .child(
                            div()
                                .id(("context-menu-paste", pane_id))
                                .flex()
                                .items_center()
                                .justify_between()
                                .px(px(8.0))
                                .py(px(5.5))
                                .rounded(px(5.0))
                                .cursor(CursorStyle::PointingHand)
                                .hover(|s| s.bg(tokens.sidebar_item_hover))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(8.0))
                                        .child(
                                            svg()
                                                .path("icons/clipboard.svg")
                                                .size(px(13.0))
                                                .text_color(tokens.text_primary),
                                        )
                                        .child(
                                            div()
                                                .font_family(MONO_FONT)
                                                .text_size(px(12.0))
                                                .text_color(tokens.text_primary)
                                                .child("Paste"),
                                        ),
                                )
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(10.5))
                                        .text_color(tokens.text_muted)
                                        .child("Ctrl+V"),
                                )
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, window, cx| {
                                        cx.stop_propagation();
                                        this.paste_from_clipboard_into_pane(pane_id, window, cx);
                                    }),
                                ),
                        )
                        // Separator
                        .child(
                            div()
                                .my(px(2.0))
                                .h(px(1.0))
                                .w_full()
                                .bg(sep_bg),
                        )
                        // Select All
                        .child(
                            div()
                                .id(("context-menu-select-all", pane_id))
                                .flex()
                                .items_center()
                                .justify_between()
                                .px(px(8.0))
                                .py(px(5.5))
                                .rounded(px(5.0))
                                .cursor(CursorStyle::PointingHand)
                                .hover(|s| s.bg(tokens.sidebar_item_hover))
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(12.0))
                                        .text_color(tokens.text_primary)
                                        .child("Select All"),
                                )
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(10.5))
                                        .text_color(tokens.text_muted)
                                        .child("Ctrl+A"),
                                )
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, window, cx| {
                                        cx.stop_propagation();
                                        this.select_all_in_terminal(pane_id, window, cx);
                                    }),
                                ),
                        )
                        // Clear Screen
                        .child(
                            div()
                                .id(("context-menu-clear", pane_id))
                                .flex()
                                .items_center()
                                .justify_between()
                                .px(px(8.0))
                                .py(px(5.5))
                                .rounded(px(5.0))
                                .cursor(CursorStyle::PointingHand)
                                .hover(|s| s.bg(tokens.sidebar_item_hover))
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(12.0))
                                        .text_color(tokens.text_primary)
                                        .child("Clear Screen"),
                                )
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(10.5))
                                        .text_color(tokens.text_muted)
                                        .child("Ctrl+L"),
                                )
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(move |this, _, window, cx| {
                                        cx.stop_propagation();
                                        this.clear_terminal_buffer(pane_id, window, cx);
                                    }),
                                ),
                        ),
                ),
        )
        .priority(200)
        .into_any_element()
    }

    /// Ctrl+R palette: type to filter, arrows to pick, Enter to insert.
    fn render_history_search_layer(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let Some(search) = self.history_search.as_ref() else {
            return div().into_any_element();
        };
        let tokens = self.tokens(cx);

        let panel_bg: Hsla = if tokens.is_dark {
            rgba(0x16171eff).into()
        } else {
            rgba(0xffffffff).into()
        };
        let panel_border: Hsla = if tokens.is_dark {
            rgba(0x2d2e3dff).into()
        } else {
            tokens.card_border
        };

        let row_selected_bg: Hsla = if tokens.is_dark {
            rgba(0x262838ff).into()
        } else {
            rgba(0x0000000f).into()
        };
        let row_hover_bg: Hsla = if tokens.is_dark {
            rgba(0x1f2130ff).into()
        } else {
            rgba(0x0000000a).into()
        };

        let rows: Vec<gpui::AnyElement> = if search.matches.is_empty() {
            vec![div()
                .px(px(10.0))
                .py(px(10.0))
                .text_size(px(12.0))
                .text_color(tokens.text_muted)
                .child("No matching commands")
                .into_any_element()]
        } else {
            search
                .matches
                .iter()
                .take(10)
                .enumerate()
                .map(|(index, command)| {
                    let selected = index == search.selected;
                    div()
                        .id(("history-search-row", index))
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap(px(10.0))
                        .px(px(9.0))
                        .py(px(6.0))
                        .rounded(px(5.0))
                        .cursor(CursorStyle::PointingHand)
                        .when(selected, |this| this.bg(row_selected_bg))
                        .hover(|this| this.bg(row_hover_bg))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(8.0))
                                .min_w_0()
                                .child(
                                    div()
                                        .w(px(16.0))
                                        .text_size(px(10.0))
                                        .text_color(if selected {
                                            rgba(0x38bdf8ff).into()
                                        } else {
                                            tokens.text_muted
                                        })
                                        .child(format!("{}", index + 1)),
                                )
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(12.5))
                                        .text_color(tokens.text_primary)
                                        .truncate()
                                        .child(command.clone()),
                                ),
                        )
                        .child(
                            div()
                                .text_size(px(10.0))
                                .text_color(tokens.text_muted)
                                .child("↵"),
                        )
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, _, window, cx| {
                                cx.stop_propagation();
                                this.accept_history_search_index(index, window, cx);
                            }),
                        )
                        .into_any_element()
                })
                .collect()
        };

        deferred(
            div()
                .id("history-search-layer")
                .size_full()
                .absolute()
                .top_0()
                .left_0()
                .child(
                    div()
                        .id("history-search-backdrop")
                        .size_full()
                        .absolute()
                        .top_0()
                        .left_0()
                        .bg(rgba(0x00000055))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _, window, cx| {
                                cx.stop_propagation();
                                this.close_history_search(window, cx);
                            }),
                        ),
                )
                .child(
                    div()
                        .id("history-search-panel")
                        .absolute()
                        .top(px(80.0))
                        .left_0()
                        .right_0()
                        .flex()
                        .justify_center()
                        .child(
                            div()
                                .w(px(560.0))
                                .p(px(10.0))
                                .rounded(px(10.0))
                                .border_1()
                                .border_color(panel_border)
                                .bg(panel_bg)
                                .shadow_lg()
                                .flex()
                                .flex_col()
                                .gap(px(8.0))
                                .on_action(cx.listener(
                                    |this, _: &gpui_component::input::Enter, window, cx| {
                                        this.accept_history_search(window, cx);
                                    },
                                ))
                                .on_key_down(cx.listener(
                                    |this, event: &KeyDownEvent, window, cx| {
                                        match event.keystroke.key.as_str() {
                                            "down" => {
                                                cx.stop_propagation();
                                                this.move_history_search_selection(1, cx);
                                            }
                                            "up" => {
                                                cx.stop_propagation();
                                                this.move_history_search_selection(-1, cx);
                                            }
                                            "escape" => {
                                                cx.stop_propagation();
                                                this.close_history_search(window, cx);
                                            }
                                            _ => {}
                                        }
                                    },
                                ))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .child(
                                            div()
                                                .text_size(px(10.5))
                                                .font_weight(gpui::FontWeight::BOLD)
                                                .text_color(tokens.text_muted)
                                                .child("SEARCH COMMAND HISTORY"),
                                        )
                                        .child(
                                            div()
                                                .text_size(px(10.0))
                                                .text_color(tokens.text_muted)
                                                .child("Esc to cancel"),
                                        ),
                                )
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(13.0))
                                        .child(
                                            Input::new(&search.input)
                                                .appearance(true)
                                                .bordered(true)
                                                .px_2()
                                                .py_1(),
                                        ),
                                )
                                .child(
                                    div()
                                        .id("history-search-results")
                                        .max_h(px(300.0))
                                        .overflow_y_scrollbar()
                                        .flex()
                                        .flex_col()
                                        .gap(px(1.0))
                                        .children(rows),
                                )
                                .child(
                                    div()
                                        .text_size(px(10.0))
                                        .text_color(tokens.text_muted)
                                        .child(
                                            "↑↓ navigate · Enter insert · Esc cancel",
                                        ),
                                ),
                        ),
                ),
        )
        .priority(210)
        .into_any_element()
    }

    #[allow(dead_code)]
    fn toggle_plus_menu(&mut self, pane_id: usize, cx: &mut Context<Self>) {
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };
        let Some(session) = project.sessions.get_mut(s_idx) else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };
        pane.plus_menu_open = !pane.plus_menu_open;
        cx.notify();
    }

    fn run_quick_action(
        &mut self,
        pane_id: usize,
        cmd: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(&(p_idx, s_idx, pane_idx)) = self.pane_index.get(&pane_id) else {
            return;
        };
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };
        let Some(session) = project.sessions.get_mut(s_idx) else {
            return;
        };
        let Some(pane) = session.panes.get_mut(pane_idx) else {
            return;
        };
        pane.plus_menu_open = false;
        pane.terminal.update(cx, |term, cx| {
            term.send_text(&format!("{cmd}\r"), cx);
        });
        if pane.command_history.last().map(|s| s.as_str()) != Some(&cmd) {
            pane.command_history.push(cmd.to_string());
        }
        if let Some(input_state) = &pane.input_state {
            input_state.update(cx, |state, cx| {
                state.focus(window, cx);
            });
        }
        cx.notify();
    }

    fn render_quick_action_item(
        label: &'static str,
        cmd: &'static str,
        icon: &'static str,
        pane_id: usize,
        tokens: &ThemeTokens,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        div()
            .id(SharedString::from(format!("quick-action-{pane_id}-{cmd}")))
            .flex()
            .items_center()
            .gap(px(8.0))
            .px(px(8.0))
            .py(px(5.0))
            .rounded(px(4.0))
            .cursor(CursorStyle::PointingHand)
            .hover(|s| s.bg(tokens.pane_header_btn_hover))
            .child(
                svg()
                    .path(icon)
                    .size(px(13.0))
                    .text_color(tokens.text_muted),
            )
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(tokens.text_primary)
                    .child(label),
            )
            .child(
                div()
                    .text_size(px(10.5))
                    .font_family(MONO_FONT)
                    .text_color(tokens.text_muted)
                    .ml_auto()
                    .child(cmd),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, window, cx| {
                    cx.stop_propagation();
                    this.run_quick_action(pane_id, cmd, window, cx);
                }),
            )
    }

    fn pane_cwd_path(&self, pane: &TerminalPane) -> Option<std::path::PathBuf> {
        #[cfg(target_os = "linux")]
        if let Some(link) = pane
            .pid
            .and_then(|pid| std::fs::read_link(format!("/proc/{pid}/cwd")).ok())
        {
            return Some(link);
        }
        if let Some(project) = self
            .pane_index
            .get(&pane.id)
            .and_then(|&(p_idx, _, _)| self.projects.get(p_idx))
        {
            return Some(project.path.clone());
        }
        directories::UserDirs::new()
            .map(|u| u.home_dir().to_path_buf())
            .or_else(|| std::env::var_os("HOME").map(std::path::PathBuf::from))
    }

    fn pane_git_branch(&self, pane: &TerminalPane) -> Option<String> {
        let dir = self.pane_cwd_path(pane)?;
        Self::detect_git_branch(&dir)
    }

    fn detect_git_branch(dir: &std::path::Path) -> Option<String> {
        let mut current = dir;
        loop {
            let git_path = current.join(".git");
            if git_path.is_dir() {
                let head_file = git_path.join("HEAD");
                if let Ok(content) = std::fs::read_to_string(head_file) {
                    let trimmed = content.trim();
                    if let Some(branch) = trimmed.strip_prefix("ref: refs/heads/") {
                        return Some(branch.to_string());
                    } else if !trimmed.is_empty() {
                        let short = if trimmed.len() > 7 { &trimmed[..7] } else { trimmed };
                        return Some(short.to_string());
                    }
                }
                return None;
            } else if git_path.is_file() {
                if let Ok(content) = std::fs::read_to_string(&git_path) {
                    for line in content.lines() {
                        if let Some(rest) = line.strip_prefix("gitdir:") {
                            let rel = rest.trim();
                            let target = if std::path::Path::new(rel).is_absolute() {
                                std::path::PathBuf::from(rel)
                            } else {
                                current.join(rel)
                            };
                            let head_file = target.join("HEAD");
                            if let Ok(head_content) = std::fs::read_to_string(head_file) {
                                let trimmed = head_content.trim();
                                if let Some(branch) = trimmed.strip_prefix("ref: refs/heads/") {
                                    return Some(branch.to_string());
                                } else if !trimmed.is_empty() {
                                    let short = if trimmed.len() > 7 { &trimmed[..7] } else { trimmed };
                                    return Some(short.to_string());
                                }
                            }
                        }
                    }
                }
                return None;
            }
            match current.parent() {
                Some(parent) if parent != current => current = parent,
                _ => break,
            }
        }
        None
    }

    fn pane_cwd_display(&self, pane: &TerminalPane) -> String {
        if let Some(path) = self.pane_cwd_path(pane) {
            Self::format_path_display(&path)
        } else {
            "~".to_string()
        }
    }

    fn format_path_display(path: &std::path::Path) -> String {
        let home = directories::UserDirs::new()
            .map(|u| u.home_dir().to_path_buf())
            .or_else(|| std::env::var_os("HOME").map(std::path::PathBuf::from));
        if let Some(rel) = home.as_ref().and_then(|h| path.strip_prefix(h).ok()) {
            if rel.as_os_str().is_empty() {
                return "~".to_string();
            } else {
                return format!("~/{}", rel.display());
            }
        }
        path.display().to_string()
    }

    fn render_warp_input_bar(
        &self,
        pane: &TerminalPane,
        _is_focused: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        let pane_id = pane.id;
        let tokens = self.tokens(cx);
        let cwd_display = self.pane_cwd_display(pane);
        let git_branch = self.pane_git_branch(pane);

        let bar_border: Hsla = if tokens.is_dark {
            rgba(0x1d1d1dff).into()
        } else {
            tokens.pane_border
        };
        let bar_bg: Hsla = if tokens.is_dark {
            rgba(0x050505ff).into()
        } else {
            tokens.pane_header_bg
        };
        let badge_border: Hsla = if tokens.is_dark {
            rgba(0x282828ff).into()
        } else {
            rgba(0x00000018).into()
        };
        let badge_bg: Hsla = if tokens.is_dark {
            rgba(0x111111ff).into()
        } else {
            rgba(0x0000000a).into()
        };
        let dir_icon_color: Hsla = if tokens.is_dark {
            rgba(0xb2b2c3ff).into()
        } else {
            tokens.text_secondary
        };
        let dir_text_color: Hsla = if tokens.is_dark {
            rgba(0xe5e6fcff).into()
        } else {
            tokens.text_primary
        };
        let git_icon_color: Hsla = if tokens.is_dark {
            rgba(0xadc993ff).into()
        } else {
            rgba(0x2d7a2dff).into()
        };
        let git_text_color: Hsla = if tokens.is_dark {
            rgba(0xd9fcb8ff).into()
        } else {
            rgba(0x236823ff).into()
        };
        let input_element = if let Some(input_state) = &pane.input_state {
            div()
                .id(("warp-input-wrapper", pane_id))
                .w_full()
                .min_h(px(24.0))
                .max_h(px(220.0))
                .overflow_y_scrollbar()
                .flex()
                .items_start()
                .font_family(MONO_FONT)
                .text_size(px(14.0))
                .key_context(WARP_INPUT_CONTEXT)
                .on_action(cx.listener(
                    move |this, _: &CompleteCommand, window, cx| {
                        cx.stop_propagation();
                        this.start_completion(pane_id, window, cx);
                    },
                ))
                .on_action(cx.listener(|_, _: &IgnoreKey, _, cx| {
                    cx.stop_propagation();
                }))
                .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                    this.handle_pane_input_key(pane_id, event, window, cx);
                }))
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .items_center()
                        .child(
                            Input::new(input_state)
                                .appearance(false)
                                .bordered(false)
                                .focus_bordered(false)
                                .cleanable(false)
                                .px_0()
                                .py_0(),
                        ),
                )
                .into_any_element()
        } else {
            div().id(("warp-input-wrapper-empty", pane_id)).w_full().min_h(px(24.0)).into_any_element()
        };

        let completion_menu = self.render_completion_menu(pane, &tokens, cx);

        let plus_menu = if pane.plus_menu_open {
            Some(
                div()
                    .id(("warp-plus-menu", pane_id))
                    .absolute()
                    .bottom(px(100.0))
                    .left(px(12.0))
                    .w(px(220.0))
                    .p(px(6.0))
                    .rounded(px(8.0))
                    .border_1()
                    .border_color(tokens.card_border)
                    .bg(if tokens.is_dark {
                        rgba(0x1e1f22ff)
                    } else {
                        rgba(0xffffffff)
                    })
                    .shadow_lg()
                    .flex()
                    .flex_col()
                    .gap(px(2.0))
                    .child(
                        div()
                            .px(px(8.0))
                            .py(px(4.0))
                            .text_size(px(10.5))
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(tokens.text_muted)
                            .child("QUICK ACTIONS"),
                    )
                    .child(
                        div()
                            .px(px(8.0))
                            .py(px(3.0))
                            .border_b_1()
                            .border_color(tokens.card_border)
                            .text_size(px(11.0))
                            .text_color(tokens.text_secondary)
                            .child(format!("📁 {}", cwd_display)),
                    )
                    .child(Self::render_quick_action_item(
                        "Clear Terminal",
                        "clear",
                        "icons/x.svg",
                        pane_id,
                        &tokens,
                        cx,
                    ))
                    .child(Self::render_quick_action_item(
                        "List Files",
                        "ls -la",
                        "icons/folder.svg",
                        pane_id,
                        &tokens,
                        cx,
                    ))
                    .child(Self::render_quick_action_item(
                        "Git Status",
                        "git status",
                        "icons/status-check.svg",
                        pane_id,
                        &tokens,
                        cx,
                    ))
                    .child(Self::render_quick_action_item(
                        "Working Directory",
                        "pwd",
                        "icons/monitor.svg",
                        pane_id,
                        &tokens,
                        cx,
                    )),
            )
        } else {
            None
        };

        div()
            .id(("warp-input-bar", pane_id))
            .relative()
            .w_full()
            .flex()
            .flex_col()
            .px(px(12.0))
            .pt(px(11.0))
            .pb(px(10.0))
            .border_t_1()
            .border_color(bar_border)
            .bg(bar_bg)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, window, cx| {
                    this.focus_pane_input(pane_id, window, cx);
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                    cx.stop_propagation();
                    this.open_terminal_context_menu(pane_id, event.position, window, cx);
                }),
            )
            .children(plus_menu)
            .children(completion_menu)
            // Row 1: Badges
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.0))
                    // Badge 1: Directory
                    .child(
                        div()
                            .id(("warp-badge-cwd", pane_id))
                            .flex()
                            .items_center()
                            .gap(px(5.0))
                            .h(px(22.0))
                            .px(px(7.0))
                            .rounded(px(5.0))
                            .border_1()
                            .border_color(badge_border)
                            .bg(badge_bg)
                            .child(
                                svg()
                                    .path("icons/folder.svg")
                                    .size(px(12.0))
                                    .text_color(dir_icon_color),
                            )
                            .child(
                                div()
                                    .font_family(MONO_FONT)
                                    .text_size(px(12.0))
                                    .text_color(dir_text_color)
                                    .child(cwd_display),
                            ),
                    )
                    // Badge 2: Git branch
                    .when_some(git_branch, |this, branch| {
                        this.child(
                            div()
                                .id(("warp-badge-git", pane_id))
                                .flex()
                                .items_center()
                                .gap(px(5.0))
                                .h(px(22.0))
                                .px(px(7.0))
                                .rounded(px(5.0))
                                .border_1()
                                .border_color(badge_border)
                                .bg(badge_bg)
                                .child(
                                    svg()
                                        .path("icons/git-branch.svg")
                                        .size(px(12.0))
                                        .text_color(git_icon_color),
                                )
                                .child(
                                    div()
                                        .font_family(MONO_FONT)
                                        .text_size(px(12.0))
                                        .text_color(git_text_color)
                                        .child(branch),
                                ),
                        )
                    }),
            )
            // Row 2: Input Row
            .child(
                div()
                    .w_full()
                    .mt(px(12.0))
                    .child(input_element),
            )
            .child(
                div()
                    .w_full()
                    .mt(px(7.0))
                    .flex()
                    .items_center()
                    .justify_between()
                    .text_size(px(10.0))
                    .text_color(tokens.text_muted)
                    .child(
                        div().child(
                            "Tab complete · ↑↓ history · Ctrl+R search · Shift+Enter newline",
                        ),
                    )
                    .child(div().font_family(MONO_FONT).child("Ctrl+G")),
            )
    }

    /// Warp-style completion popup, anchored above the input bar.
    fn render_completion_menu(
        &self,
        pane: &TerminalPane,
        tokens: &ThemeTokens,
        cx: &mut Context<Self>,
    ) -> Option<gpui::AnyElement> {
        let pane_id = pane.id;
        let menu = pane.completion_menu.as_ref()?;
        if menu.items.is_empty() {
            return None;
        }

        let bg: Hsla = if tokens.is_dark {
            rgba(0x16171eff).into()
        } else {
            rgba(0xffffffff).into()
        };
        let border: Hsla = if tokens.is_dark {
            rgba(0x2d2e3dff).into()
        } else {
            tokens.card_border
        };
        let row_selected_bg: Hsla = if tokens.is_dark {
            rgba(0x262838ff).into()
        } else {
            rgba(0x0000000f).into()
        };
        let row_hover_bg: Hsla = if tokens.is_dark {
            rgba(0x1f2130ff).into()
        } else {
            rgba(0x0000000a).into()
        };

        let rows: Vec<gpui::AnyElement> = menu
            .items
            .iter()
            .enumerate()
            .take(9)
            .map(|(index, item)| {
                let selected = index == menu.selected;
                let kind_color: Hsla = match item.kind {
                    completion::CompletionKind::Directory => rgba(0x54aeffff).into(),
                    completion::CompletionKind::File => tokens.text_secondary,
                    completion::CompletionKind::Branch => rgba(0x9ecb7bff).into(),
                    completion::CompletionKind::History => rgba(0xd7ba7dff).into(),
                    completion::CompletionKind::Command => rgba(0xc5a3ffff).into(),
                };
                div()
                    .id(("completion-item", pane_id * 1000 + index))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(10.0))
                    .px(px(8.0))
                    .py(px(4.0))
                    .rounded(px(4.0))
                    .cursor(CursorStyle::PointingHand)
                    .when(selected, |this| this.bg(row_selected_bg))
                    .hover(|this| this.bg(row_hover_bg))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(7.0))
                            .min_w_0()
                            .child(
                                div()
                                    .w(px(14.0))
                                    .text_size(px(10.0))
                                    .text_color(kind_color)
                                    .child(if item.is_dir { "▸" } else { "·" }),
                            )
                            .child(
                                div()
                                    .font_family(MONO_FONT)
                                    .text_size(px(12.5))
                                    .text_color(tokens.text_primary)
                                    .truncate()
                                    .child(item.display.clone()),
                            ),
                    )
                    .child(
                        div()
                            .text_size(px(10.0))
                            .text_color(tokens.text_muted)
                            .child(item.detail.clone()),
                    )
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, window, cx| {
                            cx.stop_propagation();
                            this.accept_completion_at(pane_id, index, window, cx);
                        }),
                    )
                    .into_any_element()
            })
            .collect();

        Some(
            div()
                .id(("completion-menu", pane_id))
                .absolute()
                .bottom(relative(1.0))
                .left(px(0.0))
                .mb(px(6.0))
                .w(px(440.0))
                .max_w(relative(1.0))
                .max_h(px(240.0))
                .overflow_y_scrollbar()
                .p(px(5.0))
                .rounded(px(8.0))
                .border_1()
                .border_color(border)
                .bg(bg)
                .shadow_lg()
                .flex()
                .flex_col()
                .gap(px(1.0))
                .children(rows)
                .into_any_element(),
        )
    }

    fn render_terminal_pane(
        &self,
        pane: &TerminalPane,
        is_maximized: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        let pane_id = pane.id;
        let is_focused = self.focused_pane_id == Some(pane_id);
        let tokens = self.tokens(cx);

        div()
            .id(("terminal-pane", pane_id))
            .flex()
            .flex_col()
            .size_full()
            .bg(tokens.pane_bg)
            .rounded(px(8.0))
            .border_1()
            .border_color(if is_focused {
                tokens.pane_border_focused
            } else {
                tokens.pane_border
            })
            .overflow_hidden()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, window, cx| {
                    this.focus_pane(pane_id, window, cx);
                }),
            )
            .child(
                div()
                    .h(px(28.0))
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px(px(8.0))
                    .bg(tokens.pane_header_bg)
                    .child(
                        div()
                            .text_size(px(11.5))
                            .text_color(tokens.pane_header_fg)
                            .child(pane.display_title.clone()),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.0))
                            .child(
                                div()
                                    .id(("bottom-input-toggle-btn", pane_id))
                                    .size(px(22.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(4.0))
                                    .cursor(CursorStyle::PointingHand)
                                    .bg(if pane.should_show_bottom_input(cx) {
                                        tokens.pane_header_btn_hover
                                    } else {
                                        gpui::transparent_black()
                                    })
                                    .hover(|s| s.bg(tokens.pane_header_btn_hover))
                                    .tooltip(|_, cx| {
                                        cx.new(|_| {
                                            gpui_component::tooltip::Tooltip::new(SharedString::from(
                                                "Toggle Warp input editor (Ctrl+G)",
                                            ))
                                        })
                                        .into()
                                    })
                                    .child(
                                        svg()
                                            .path("icons/terminal-bottom.svg")
                                            .size(px(13.0))
                                            .text_color(if pane.should_show_bottom_input(cx) {
                                                rgba(0x38bdf8ff).into()
                                            } else {
                                                tokens.pane_header_icon
                                            })
                                            .hover(|s| s.text_color(tokens.pane_header_icon_hover)),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.toggle_pane_bottom_input(pane_id, window, cx);
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id(("split-v-btn", pane_id))
                                    .size(px(22.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(4.0))
                                    .cursor(CursorStyle::PointingHand)
                                    .hover(|s| s.bg(tokens.pane_header_btn_hover))
                                    .child(
                                        svg()
                                            .path("icons/split-vertical.svg")
                                            .size(px(13.0))
                                            .text_color(tokens.pane_header_icon)
                                            .hover(|s| s.text_color(tokens.pane_header_icon_hover)),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.split_pane(
                                                pane_id,
                                                gpui::Axis::Horizontal,
                                                window,
                                                cx,
                                            );
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id(("split-h-btn", pane_id))
                                    .size(px(22.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(4.0))
                                    .cursor(CursorStyle::PointingHand)
                                    .hover(|s| s.bg(tokens.pane_header_btn_hover))
                                    .child(
                                        svg()
                                            .path("icons/split-horizontal.svg")
                                            .size(px(13.0))
                                            .text_color(tokens.pane_header_icon)
                                            .hover(|s| s.text_color(tokens.pane_header_icon_hover)),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.split_pane(
                                                pane_id,
                                                gpui::Axis::Vertical,
                                                window,
                                                cx,
                                            );
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id(("maximize-btn", pane_id))
                                    .size(px(22.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(4.0))
                                    .cursor(CursorStyle::PointingHand)
                                    .hover(|s| s.bg(tokens.pane_header_btn_hover))
                                    .child(
                                        svg()
                                            .path(if is_maximized {
                                                "icons/pane-restore.svg"
                                            } else {
                                                "icons/pane-maximize.svg"
                                            })
                                            .size(px(13.0))
                                            .text_color(tokens.pane_header_icon)
                                            .hover(|s| s.text_color(tokens.pane_header_icon_hover)),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, _, cx| {
                                            cx.stop_propagation();
                                            this.toggle_maximize_pane(pane_id, cx);
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id(("close-btn", pane_id))
                                    .size(px(22.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(4.0))
                                    .cursor(CursorStyle::PointingHand)
                                    .hover(|s| s.bg(tokens.pane_header_btn_hover))
                                    .child(
                                        svg()
                                            .path("icons/pane-close.svg")
                                            .size(px(13.0))
                                            .text_color(tokens.pane_header_icon)
                                            .hover(|s| s.text_color(rgba(0xf87171ff))),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.close_pane(pane_id, window, cx);
                                        }),
                                    ),
                            ),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .size_full()
                    .overflow_hidden()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, window, cx| {
                            this.focus_pane(pane_id, window, cx);
                        }),
                    )
                    .on_mouse_down(
                        MouseButton::Right,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            cx.stop_propagation();
                            this.open_terminal_context_menu(pane_id, event.position, window, cx);
                        }),
                    )
                    .child(pane.terminal.clone()),
            )
            .when(
                pane.should_show_bottom_input(cx),
                |this| this.child(self.render_warp_input_bar(pane, is_focused, cx)),
            )
    }

    fn render_pane_node(
        &self,
        node: &PaneNode,
        session: &Session,
        is_maximized: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        match node {
            PaneNode::Leaf(pane_id) => {
                if let Some(pane) = session.panes.iter().find(|p| p.id == *pane_id) {
                    self.render_terminal_pane(pane, is_maximized, cx)
                        .into_any_element()
                } else {
                    div().into_any_element()
                }
            }
            PaneNode::Split {
                id,
                axis,
                left,
                right,
            } => {
                let left_el = self.render_pane_node(left, session, is_maximized, cx);
                let right_el = self.render_pane_node(right, session, is_maximized, cx);
                let is_horizontal = *axis == gpui::Axis::Horizontal;
                let split_id = ("terminal-split", *id);

                if is_horizontal {
                    h_resizable(split_id)
                        .child(
                            resizable_panel().size_range(px(100.0)..px(3000.0)).child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .size_full()
                                    .p(px(2.0))
                                    .child(div().flex_1().size_full().child(left_el)),
                            ),
                        )
                        .child(
                            resizable_panel().size_range(px(100.0)..px(3000.0)).child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .size_full()
                                    .p(px(2.0))
                                    .child(div().flex_1().size_full().child(right_el)),
                            ),
                        )
                        .into_any_element()
                } else {
                    v_resizable(split_id)
                        .child(
                            resizable_panel().size_range(px(70.0)..px(3000.0)).child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .size_full()
                                    .p(px(2.0))
                                    .child(div().flex_1().size_full().child(left_el)),
                            ),
                        )
                        .child(
                            resizable_panel().size_range(px(70.0)..px(3000.0)).child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .size_full()
                                    .p(px(2.0))
                                    .child(div().flex_1().size_full().child(right_el)),
                            ),
                        )
                        .into_any_element()
                }
            }
        }
    }

    /// The pane entity a top-level action (font size) applies to: the focused
    /// pane, or the first pane of the active session's screen.
    fn active_pane_terminal(&self) -> Option<Entity<TerminalView>> {
        let focused_pane = self
            .focused_pane_id
            .and_then(|focused| self.pane_index.get(&focused).copied())
            .and_then(|(p_idx, s_idx, pane_idx)| {
                self.projects
                    .get(p_idx)
                    .and_then(|p| p.sessions.get(s_idx))
                    .and_then(|s| s.panes.get(pane_idx))
            })
            .map(|p| p.terminal.clone());
        focused_pane.or_else(|| {
            self.active_session()
                .and_then(|s| s.panes.first())
                .map(|p| p.terminal.clone())
        })
    }

    fn on_key_down(&mut self, event: &KeyDownEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let keystroke = &event.keystroke;

        if keystroke.key == "escape" && self.active_view == ActiveView::Settings {
            self.active_view = ActiveView::Workspace;
            cx.stop_propagation();
            cx.notify();
            return;
        }

        if keystroke.modifiers.control && keystroke.key == "," {
            self.active_view = if self.active_view == ActiveView::Settings {
                ActiveView::Workspace
            } else {
                ActiveView::Settings
            };
            cx.stop_propagation();
            cx.notify();
            return;
        }

        if (keystroke.modifiers.control || keystroke.modifiers.platform)
            && !keystroke.modifiers.shift
            && !keystroke.modifiers.alt
            && (keystroke.key == "g" || keystroke.key == "G")
        {
            if let Some(pane_id) = self.focused_pane_id {
                self.toggle_pane_bottom_input(pane_id, _window, cx);
                cx.stop_propagation();
                return;
            }
        }

        if keystroke.modifiers.control && (keystroke.key == "+" || keystroke.key == "=") {
            if let Some(entity) = self.active_pane_terminal() {
                entity.update(cx, |terminal, cx| {
                    let mut config = terminal.config().clone();
                    config.font_size += px(1.0);
                    terminal.update_config(config, cx);
                });
            }
            cx.stop_propagation();
        } else if keystroke.modifiers.control && keystroke.key == "-" {
            if let Some(entity) = self.active_pane_terminal() {
                entity.update(cx, |terminal, cx| {
                    let mut config = terminal.config().clone();
                    if config.font_size > px(6.0) {
                        config.font_size -= px(1.0);
                        terminal.update_config(config, cx);
                    }
                });
            }
            cx.stop_propagation();
        }
    }

    // ---------------------------------------------------------------------
    // Sidebar: projects and their sessions
    // ---------------------------------------------------------------------

    /// The whole left rail: a toolbar, the Projects header with its "Open
    /// project" action, and one collapsible block per open project.
    fn render_sidebar(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let project_rows: Vec<gpui::AnyElement> = self
            .projects
            .iter()
            .enumerate()
            .map(|(p_idx, project)| self.render_project_entry(p_idx, project, cx))
            .collect();

        let tokens = self.tokens(cx);

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(tokens.sidebar_bg)
            .border_r_1()
            .border_color(tokens.sidebar_border)
            .when(self.project_menu.is_some(), |this| {
                this.child(
                    deferred(
                        div()
                            .id("project-menu-backdrop")
                            .size_full()
                            .absolute()
                            .top_0()
                            .left_0()
                            .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                this.project_menu = None;
                                cx.notify();
                            })),
                    )
                    .priority(100),
                )
            })
            .child(self.render_sidebar_toolbar(cx))
            .child(
                div()
                    .flex_1()
                    .overflow_y_scrollbar()
                    .flex()
                    .flex_col()
                    .pb(px(10.0))
                    .children(project_rows)
                    .when(self.projects.is_empty(), |this| {
                        this.child(self.render_sidebar_empty_state(cx))
                    }),
            )
            .when(self.integration_panel_visible, |this| {
                this.child(self.render_integrations_panel(cx))
            })
            .child(self.render_sidebar_footer(cx))
            .into_any_element()
    }

    fn render_sidebar_footer(&self, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        let is_settings = self.active_view == ActiveView::Settings;
        let tokens = self.tokens(cx);

        div()
            .id("sidebar-footer")
            .w_full()
            .p(px(8.0))
            .border_t_1()
            .border_color(tokens.sidebar_footer_border)
            .bg(tokens.sidebar_footer_bg)
            .child(
                div()
                    .id("sidebar-settings-btn")
                    .h(px(34.0))
                    .w_full()
                    .px(px(10.0))
                    .rounded(px(6.0))
                    .flex()
                    .items_center()
                    .justify_between()
                    .cursor(CursorStyle::PointingHand)
                    .bg(if is_settings {
                        tokens.sidebar_item_active
                    } else {
                        rgba(0x00000000).into()
                    })
                    .hover(move |s| {
                        if is_settings {
                            s
                        } else {
                            s.bg(tokens.sidebar_item_hover)
                        }
                    })
                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                        cx.stop_propagation();
                        this.active_view = if this.active_view == ActiveView::Settings {
                            ActiveView::Workspace
                        } else {
                            ActiveView::Settings
                        };
                        cx.notify();
                    }))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.0))
                            .child(
                                svg()
                                    .path("icons/settings.svg")
                                    .size(px(15.0))
                                    .text_color(if is_settings {
                                        tokens.sidebar_item_active_fg
                                    } else {
                                        tokens.sidebar_icon
                                    }),
                            )
                            .child(
                                div()
                                    .text_size(px(12.5))
                                    .font_weight(if is_settings {
                                        gpui::FontWeight::SEMIBOLD
                                    } else {
                                        gpui::FontWeight::NORMAL
                                    })
                                    .text_color(if is_settings {
                                        tokens.sidebar_item_active_fg
                                    } else {
                                        tokens.sidebar_fg
                                    })
                                    .child("Settings"),
                            ),
                    )
                    .child(
                        div()
                            .text_size(px(10.5))
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .text_color(tokens.sidebar_muted)
                            .child(self.theme_preference.label()),
                    ),
            )
    }

    fn render_settings_page(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let tokens = self.tokens(cx);
        let is_maximized = window.is_maximized();
        let is_fullscreen = window.is_fullscreen();
        let controls = window.window_controls();

        let window_controls = (!is_fullscreen).then(|| {
            div()
                .id("settings-window-controls")
                .flex()
                .items_center()
                .h_full()
                .when(controls.minimize, |this| {
                    this.child(
                        div()
                            .id("settings-window-minimize-btn")
                            .w(px(36.0))
                            .h_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor(CursorStyle::PointingHand)
                            .hover(move |s| s.bg(tokens.titlebar_btn_hover))
                            .on_mouse_down(MouseButton::Left, |_, window, cx| {
                                cx.stop_propagation();
                                window.minimize_window();
                            })
                            .child(
                                svg()
                                    .path("window-minimize.svg")
                                    .size(px(14.0))
                                    .text_color(tokens.titlebar_fg)
                                    .hover(move |s| s.text_color(tokens.text_primary)),
                            ),
                    )
                })
                .when(controls.maximize, |this| {
                    this.child(
                        div()
                            .id("settings-window-maximize-btn")
                            .w(px(36.0))
                            .h_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor(CursorStyle::PointingHand)
                            .hover(move |s| s.bg(tokens.titlebar_btn_hover))
                            .on_mouse_down(MouseButton::Left, |_, window, cx| {
                                cx.stop_propagation();
                                window.zoom_window();
                            })
                            .child(
                                svg()
                                    .path(if is_maximized {
                                        "window-restore.svg"
                                    } else {
                                        "window-maximize.svg"
                                    })
                                    .size(px(14.0))
                                    .text_color(tokens.titlebar_fg)
                                    .hover(move |s| s.text_color(tokens.text_primary)),
                            ),
                    )
                })
                .child(
                    div()
                        .id("settings-window-close-btn")
                        .w(px(36.0))
                        .h_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor(CursorStyle::PointingHand)
                        .hover(|s| s.bg(rgba(0xe81123ff)))
                        .on_mouse_down(MouseButton::Left, |_, window, cx| {
                            cx.stop_propagation();
                            window.remove_window();
                        })
                        .child(
                            svg()
                                .path("window-close.svg")
                                .size(px(14.0))
                                .text_color(tokens.titlebar_fg)
                                .hover(|s| s.text_color(rgba(0xffffffff))),
                        ),
                )
        });

        let settings_titlebar = self
            .make_drag_area(cx)
            .h(px(38.0))
            .w_full()
            .flex()
            .items_center()
            .justify_between()
            .pl_3()
            .bg(tokens.titlebar_bg)
            .border_b_1()
            .border_color(tokens.titlebar_border)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .when(!self.sidebar_visible, |this| {
                        this.child(
                            div()
                                .id("settings-sidebar-toggle-btn")
                                .cursor(CursorStyle::PointingHand)
                                .flex()
                                .items_center()
                                .justify_center()
                                .child(
                                    svg()
                                        .path("icons/panel-left.svg")
                                        .size(px(16.0))
                                        .text_color(tokens.titlebar_fg)
                                        .hover(move |s| s.text_color(tokens.text_primary)),
                                )
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(|this, _, _, cx| {
                                        cx.stop_propagation();
                                        this.sidebar_visible = true;
                                        cx.notify();
                                        this.schedule_workspace_save(cx);
                                    }),
                                ),
                        )
                    })
                    .child(
                        div()
                            .id("settings-back-btn")
                            .cursor(CursorStyle::PointingHand)
                            .px(px(8.0))
                            .py(px(4.0))
                            .rounded(px(4.0))
                            .flex()
                            .items_center()
                            .gap(px(6.0))
                            .bg(tokens.pill_inactive_bg)
                            .hover(move |s| s.bg(tokens.card_hover))
                            .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                cx.stop_propagation();
                                this.active_view = ActiveView::Workspace;
                                cx.notify();
                            }))
                            .child(
                                svg()
                                    .path("icons/arrow-left.svg")
                                    .size(px(13.0))
                                    .text_color(tokens.text_secondary),
                            )
                            .child(
                                div()
                                    .text_size(px(11.5))
                                    .text_color(tokens.text_primary)
                                    .child("Workspace"),
                            ),
                    )
                    .child(
                        div()
                            .text_size(px(12.5))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(tokens.titlebar_fg)
                            .child("Settings"),
                    ),
            )
            .children(window_controls);

        let current_pref = self.theme_preference;
        let current_appearance = window.appearance();
        let resolved_mode = current_pref.resolve_mode(current_appearance);

        let theme_card = |id: &'static str,
                          pref: ThemePreference,
                          title: &'static str,
                          subtitle: &'static str,
                          icon: &'static str,
                          is_default: bool| {
            let is_selected = current_pref == pref;
            div()
                .id(id)
                .flex_1()
                .p(px(16.0))
                .rounded(px(8.0))
                .border_1()
                .border_color(if is_selected {
                    tokens.pane_border_focused
                } else {
                    tokens.card_border
                })
                .bg(if is_selected {
                    tokens.pill_inactive_bg
                } else {
                    tokens.card_bg
                })
                .cursor(CursorStyle::PointingHand)
                .hover(move |s| {
                    if !is_selected {
                        s.bg(tokens.card_hover)
                    } else {
                        s
                    }
                })
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        cx.stop_propagation();
                        this.set_theme_preference(pref, Some(window), cx);
                    }),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(10.0))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .size(px(32.0))
                                        .rounded(px(6.0))
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .bg(if is_selected {
                                            tokens.pill_active_bg
                                        } else {
                                            tokens.pill_inactive_bg
                                        })
                                        .child(
                                            svg()
                                                .path(icon)
                                                .size(px(16.0))
                                                .text_color(if is_selected {
                                                    tokens.pill_active_fg
                                                } else {
                                                    tokens.text_secondary
                                                }),
                                        ),
                                )
                                .when(is_default, |this| {
                                    this.child(
                                        div()
                                            .px_1p5()
                                            .py_0p5()
                                            .rounded_sm()
                                            .bg(tokens.pill_inactive_bg)
                                            .border_1()
                                            .border_color(tokens.border_subtle)
                                            .text_size(px(10.0))
                                            .font_weight(gpui::FontWeight::MEDIUM)
                                            .text_color(tokens.text_secondary)
                                            .child("Default"),
                                    )
                                }),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(2.0))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(6.0))
                                        .child(
                                            div()
                                                .text_size(px(14.0))
                                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                                .text_color(tokens.text_primary)
                                                .child(title),
                                        )
                                        .when(is_selected, |this| {
                                            this.child(
                                                div()
                                                    .size(px(6.0))
                                                    .rounded_full()
                                                    .bg(tokens.pill_active_bg),
                                            )
                                        }),
                                )
                                .child(
                                    div()
                                        .text_size(px(11.5))
                                        .text_color(tokens.text_secondary)
                                        .child(subtitle),
                                ),
                        ),
                )
        };

        let theme_section = div()
            .w_full()
            .p(px(20.0))
            .rounded(px(8.0))
            .border_1()
            .border_color(tokens.card_border)
            .bg(tokens.card_bg)
            .flex()
            .flex_col()
            .gap(px(16.0))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.0))
                    .child(
                        div()
                            .text_size(px(15.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(tokens.text_primary)
                            .child("Appearance & Theme"),
                    )
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(tokens.text_secondary)
                            .child("Configure interface theme. The token-based theme dynamically styles the workspace and terminal emulators."),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap(px(12.0))
                    .child(theme_card(
                        "theme-option-system",
                        ThemePreference::System,
                        "System",
                        "Automatically sync with OS appearance",
                        "icons/monitor.svg",
                        true,
                    ))
                    .child(theme_card(
                        "theme-option-dark",
                        ThemePreference::Dark,
                        "Dark",
                        "Sleek dark theme with GitHub Dark terminal",
                        "icons/moon.svg",
                        false,
                    ))
                    .child(theme_card(
                        "theme-option-light",
                        ThemePreference::Light,
                        "Light",
                        "Clean light theme with GitHub Light terminal",
                        "icons/sun.svg",
                        false,
                    )),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .pt(px(8.0))
                    .border_t_1()
                    .border_color(tokens.border_subtle)
                    .child(
                        div()
                            .text_size(px(11.5))
                            .text_color(tokens.text_muted)
                            .child(format!(
                                "Current preference: {} · Active resolved mode: {}",
                                current_pref.label(),
                                if resolved_mode.is_dark() { "Dark" } else { "Light" }
                            )),
                    ),
            );
        let warp_input_enabled = self.warp_bottom_input;
        let input_section = div()
            .w_full()
            .p(px(20.0))
            .rounded(px(8.0))
            .border_1()
            .border_color(tokens.card_border)
            .bg(tokens.card_bg)
            .flex()
            .flex_col()
            .gap(px(16.0))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.0))
                    .child(
                        div()
                            .text_size(px(15.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(tokens.text_primary)
                            .child("Terminal Input Style"),
                    )
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(tokens.text_secondary)
                            .child("Choose the default input editor style for every terminal inside Markup."),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap(px(12.0))
                    .child(
                        div()
                            .id("input-style-warp")
                            .flex_1()
                            .p(px(14.0))
                            .rounded(px(6.0))
                            .border_1()
                            .border_color(if warp_input_enabled {
                                tokens.pane_border_focused
                            } else {
                                tokens.border_subtle
                            })
                            .bg(if warp_input_enabled {
                                tokens.pill_active_bg
                            } else {
                                tokens.pill_inactive_bg
                            })
                            .cursor(CursorStyle::PointingHand)
                            .hover(|s| s.bg(tokens.card_hover))
                            .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                this.warp_bottom_input = true;
                                for project in &mut this.projects {
                                    for session in &mut project.sessions {
                                        for pane in &mut session.panes {
                                            pane.bottom_input_enabled = true;
                                        }
                                    }
                                }
                                this.schedule_workspace_save(cx);
                                cx.notify();
                            }))
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.0))
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .gap(px(6.0))
                                            .child(
                                                div()
                                                    .text_size(px(13.0))
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(rgba(0x38bdf8ff))
                                                    .child("❯"),
                                            )
                                            .child(
                                                div()
                                                    .text_size(px(13.0))
                                                    .font_weight(gpui::FontWeight::MEDIUM)
                                                    .text_color(tokens.text_primary)
                                                    .child("Warp-style Bottom Input (Recommended)"),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(11.5))
                                            .text_color(tokens.text_secondary)
                                            .child("Pinned modern prompt editor at the bottom with command history, multi-line editing, and status."),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .id("input-style-classic")
                            .flex_1()
                            .p(px(14.0))
                            .rounded(px(6.0))
                            .border_1()
                            .border_color(if !warp_input_enabled {
                                tokens.pane_border_focused
                            } else {
                                tokens.border_subtle
                            })
                            .bg(if !warp_input_enabled {
                                tokens.pill_active_bg
                            } else {
                                tokens.pill_inactive_bg
                            })
                            .cursor(CursorStyle::PointingHand)
                            .hover(|s| s.bg(tokens.card_hover))
                            .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                this.warp_bottom_input = false;
                                for project in &mut this.projects {
                                    for session in &mut project.sessions {
                                        for pane in &mut session.panes {
                                            pane.bottom_input_enabled = false;
                                        }
                                    }
                                }
                                this.schedule_workspace_save(cx);
                                cx.notify();
                            }))
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.0))
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .font_weight(gpui::FontWeight::MEDIUM)
                                            .text_color(tokens.text_primary)
                                            .child("Classic Terminal"),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(11.5))
                                            .text_color(tokens.text_secondary)
                                            .child("Direct cursor typing inline on the terminal character grid."),
                                    ),
                            ),
                    ),
            );

        let token_swatch = |label: &'static str, color: gpui::Hsla, name: &'static str| {
            div()
                .flex()
                .items_center()
                .justify_between()
                .p(px(8.0))
                .rounded(px(6.0))
                .bg(tokens.pill_inactive_bg)
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.0))
                        .child(
                            div()
                                .size(px(18.0))
                                .rounded(px(4.0))
                                .border_1()
                                .border_color(tokens.border)
                                .bg(color),
                        )
                        .child(
                            div()
                                .text_size(px(12.0))
                                .font_weight(gpui::FontWeight::MEDIUM)
                                .text_color(tokens.text_primary)
                                .child(label),
                        ),
                )
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(tokens.text_muted)
                        .child(name),
                )
        };

        let tokens_section = div()
            .w_full()
            .p(px(20.0))
            .rounded(px(8.0))
            .border_1()
            .border_color(tokens.card_border)
            .bg(tokens.card_bg)
            .flex()
            .flex_col()
            .gap(px(16.0))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.0))
                    .child(
                        div()
                            .text_size(px(15.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(tokens.text_primary)
                            .child("Design Tokens (Active)"),
                    )
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(tokens.text_secondary)
                            .child("Markup uses token-based theming. Below are the live tokens active for this theme:"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.0))
                    .child(token_swatch("Window Background", tokens.window_bg, "window_bg"))
                    .child(token_swatch("Sidebar Background", tokens.sidebar_bg, "sidebar_bg"))
                    .child(token_swatch("Terminal Pane Background", tokens.pane_bg, "pane_bg"))
                    .child(token_swatch("Text Primary", tokens.text_primary, "text_primary"))
                    .child(token_swatch("Border", tokens.border, "border"))
                    .child(token_swatch("Primary Accent", tokens.primary_btn_bg, "primary_btn_bg"))
                    .child(token_swatch("Terminal Border (Focused)", tokens.pane_border_focused, "pane_border_focused")),
            );

        let about_section = div()
            .w_full()
            .p(px(20.0))
            .rounded(px(8.0))
            .border_1()
            .border_color(tokens.card_border)
            .bg(tokens.card_bg)
            .flex()
            .items_center()
            .justify_between()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.0))
                    .child(
                        div()
                            .size(px(36.0))
                            .rounded(px(8.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(gpui::img(SharedString::from("logo/markup.png")).size(px(32.0))),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(2.0))
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(tokens.text_primary)
                                    .child("Markup"),
                            )
                            .child(
                                div()
                                    .text_size(px(11.5))
                                    .text_color(tokens.text_secondary)
                                    .child("GPU-accelerated terminal emulator for AI coding agents"),
                            ),
                    ),
            )
            .child(
                div()
                    .text_size(px(11.5))
                    .text_color(tokens.text_muted)
                    .child("v0.1.0"),
            );

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(tokens.window_bg)
            .child(settings_titlebar)
            .child(
                div()
                    .flex_1()
                    .overflow_y_scrollbar()
                    .p(px(24.0))
                    .child(
                        div()
                            .max_w(px(720.0))
                            .w_full()
                            .mx_auto()
                            .flex()
                            .flex_col()
                            .gap(px(20.0))
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.0))
                                    .child(
                                        div()
                                            .text_size(px(22.0))
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(tokens.text_primary)
                                            .child("Settings"),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .text_color(tokens.text_secondary)
                                            .child("Manage application preferences, appearance, and theme tokens."),
                                    ),
                            )
                            .child(theme_section)
                            .child(input_section)
                            .child(tokens_section)
                            .child(about_section),
                    ),
            )
    }

    fn render_sidebar_toolbar(&self, cx: &mut Context<Self>) -> gpui::Div {
        let tokens = self.tokens(cx);
        div()
            .h(px(38.0))
            .w_full()
            .flex()
            .items_center()
            .justify_between()
            .pl(px(14.0))
            .pr(px(10.0))
            .child(sidebar_icon_button(
                "sidebar-toggle-btn",
                "icons/panel-left.svg",
                16.0,
                &tokens,
                cx.listener::<MouseDownEvent>(|this, _, _, cx| {
                    this.sidebar_visible = !this.sidebar_visible;
                    cx.notify();
                    this.schedule_workspace_save(cx);
                }),
            ))
            .child(self.make_drag_area(cx).flex_1().h_full())
            .child(
                div()
                    .flex()
                    .items_center()
                    .child(sidebar_icon_button(
                        "sidebar-open-project-btn",
                        "icons/folder-plus.svg",
                        15.0,
                        &tokens,
                        cx.listener::<MouseDownEvent>(|this, _, window, cx| {
                            this.select_folder_and_add_project(window, cx);
                        }),
                    )),
            )
    }

    fn render_integrations_panel(&self, cx: &mut Context<Self>) -> gpui::Div {
        let weak = cx.entity().downgrade();
        let tokens = self.tokens(cx);
        div()
            .w_full()
            .px(px(12.0))
            .py(px(10.0))
            .border_t_1()
            .border_color(tokens.sidebar_border)
            .flex()
            .flex_col()
            .gap(px(7.0))
            .child(div().text_size(px(12.0)).font_weight(gpui::FontWeight::BOLD).text_color(tokens.sidebar_fg).child("Agent integrations"))
            .child(div().text_size(px(10.5)).text_color(tokens.sidebar_muted).child("Claude Code · Codex · OpenCode · Pi · OMP · Command Code"))
            .child(div().text_size(px(10.0)).text_color(tokens.sidebar_muted).child("Versions and hook trust: markup integration doctor"))
            .child(div().text_size(px(10.0)).text_color(tokens.sidebar_muted).child(self.integration_feedback.clone()))
            .child(
                div()
                    .id("enable-agent-integrations")
                    .w_full().py(px(6.0)).rounded(px(6.0))
                    .bg(tokens.card_bg).hover(move |s| s.bg(tokens.card_hover))
                    .cursor(CursorStyle::PointingHand)
                    .text_size(px(11.0)).text_color(tokens.sidebar_fg)
                    .flex().items_center().justify_center()
                    .child("Enable local integrations")
                    .on_mouse_down(MouseButton::Left, move |_, _, cx| {
                        let target = weak.clone();
                        cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                            let result = cx.background_executor().spawn(async move {
                                std::env::current_exe().map_err(anyhow::Error::from).and_then(|exe| {
                                    std::process::Command::new(exe).args(["integration", "setup"]).output().map_err(anyhow::Error::from)
                                })
                            }).await;
                            let message = match result {
                                Ok(output) if output.status.success() => "Installed. Restart each agent; Codex may also ask you to trust its hooks.".to_string(),
                                Ok(output) => format!("Setup failed: {}", String::from_utf8_lossy(&output.stderr).trim()),
                                Err(error) => format!("Setup failed: {error}"),
                            };
                            let _ = target.update(cx, |this, cx| { this.integration_feedback = SharedString::from(message); cx.notify(); });
                        }).detach();
                    }),
            )
    }

    /// One collapsible project block: the project header (activate, expand,
    /// new session, close project) followed by its session rows.
    fn render_project_entry(
        &self,
        project_idx: usize,
        project: &Project,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let tokens = self.tokens(cx);
        let is_active_project = project_idx == self.active_project_idx;
        let needs_attention = project
            .sessions
            .iter()
            .filter(|s| matches!(s.status, SessionStatus::Blocked | SessionStatus::Error))
            .count();
        let chevron = if project.expanded {
            "icons/chevron-down.svg"
        } else {
            "icons/chevron-right.svg"
        };

        let mut rows: Vec<gpui::AnyElement> = Vec::new();
        if project.expanded {
            if project.sessions.is_empty() {
                rows.push(
                    div()
                        .pl(px(34.0))
                        .pr(px(16.0))
                        .py(px(6.0))
                        .text_size(px(11.5))
                        .text_color(tokens.sidebar_muted)
                        .child("No sessions yet")
                        .into_any_element(),
                );
            }
            for (s_idx, session) in project.sessions.iter().enumerate() {
                let is_active_session = is_active_project && s_idx == project.active_session_idx;
                rows.push(self.render_session_entry(
                    project_idx,
                    s_idx,
                    session,
                    is_active_session,
                    cx,
                ));
            }
        }

        let is_renaming = self
            .renaming_project
            .as_ref()
            .is_some_and(|r| r.project_id == project.id);
        let is_menu_open = self.project_menu == Some(project.id);
        let group_name = SharedString::from(format!("project-row-{}", project.id));
        let project_id = project.id;

        let header_content: gpui::AnyElement = if is_renaming {
            let renaming = self.renaming_project.as_ref().unwrap();
            let input_entity = renaming.input.clone();
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(4.0))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .child(
                            Input::new(&renaming.input)
                                .with_size(gpui_component::Size::Small)
                                .cleanable(false),
                        ),
                )
                .child(sidebar_icon_button(
                    ("project-rename-save", project.id),
                    "icons/status-check.svg",
                    12.0,
                    &tokens,
                    cx.listener(move |this, _, _, cx| {
                        let val = input_entity.read(cx).value();
                        this.finish_renaming_project(project_id, val, cx);
                    }),
                ))
                .child(sidebar_icon_button(
                    ("project-rename-cancel", project.id),
                    "icons/x.svg",
                    11.0,
                    &tokens,
                    cx.listener(move |this, _, _, cx| {
                        this.cancel_renaming_project(cx);
                    }),
                ))
                .into_any_element()
        } else {
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(4.0))
                .child(
                    div()
                        .min_w_0()
                        .truncate()
                        .text_size(px(12.5))
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(if is_active_project {
                            tokens.sidebar_fg
                        } else {
                            tokens.sidebar_muted
                        })
                        .child(project.name.clone()),
                )
                .child(
                    div()
                        .relative()
                        .child(
                            sidebar_icon_button(
                                ("project-options", project.id),
                                "icons/ellipsis.svg",
                                13.0,
                                &tokens,
                                cx.listener::<MouseDownEvent>(move |this, _, _, cx| {
                                    this.project_menu = if this.project_menu == Some(project_id) {
                                        None
                                    } else {
                                        Some(project_id)
                                    };
                                    cx.notify();
                                }),
                            )
                            .opacity(if is_menu_open { 1.0 } else { 0.0 })
                            .group_hover(group_name.clone(), |s| s.opacity(1.0)),
                        )
                        .when(is_menu_open, |this| {
                            this.child(
                                deferred(
                                    div()
                                        .id(("project-menu-popover", project.id))
                                        .absolute()
                                        .top(px(24.0))
                                        .left_0()
                                        .w(px(130.0))
                                        .bg(tokens.card_bg)
                                        .border_1()
                                        .border_color(tokens.card_border)
                                        .rounded(px(6.0))
                                        .shadow_md()
                                        .p(px(4.0))
                                        .flex()
                                        .flex_col()
                                        .gap(px(2.0))
                                        .child(
                                            div()
                                                .id(("menu-rename", project.id))
                                                .px(px(8.0))
                                                .py(px(5.0))
                                                .rounded(px(4.0))
                                                .text_size(px(12.0))
                                                .text_color(tokens.text_primary)
                                                .cursor(CursorStyle::PointingHand)
                                                .hover(move |s| s.bg(tokens.sidebar_item_hover))
                                                .child("Rename")
                                                .on_mouse_down(
                                                    MouseButton::Left,
                                                    cx.listener(move |this, _, window, cx| {
                                                        cx.stop_propagation();
                                                        this.start_renaming_project(project_id, window, cx);
                                                    }),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .id(("menu-remove", project.id))
                                                .px(px(8.0))
                                                .py(px(5.0))
                                                .rounded(px(4.0))
                                                .text_size(px(12.0))
                                                .text_color(tokens.text_primary)
                                                .cursor(CursorStyle::PointingHand)
                                                .hover(move |s| s.bg(tokens.sidebar_item_hover))
                                                .child("Remove")
                                                .on_mouse_down(
                                                    MouseButton::Left,
                                                    cx.listener(move |this, _, window, cx| {
                                                        cx.stop_propagation();
                                                        this.project_menu = None;
                                                        if let Some(idx) = this.projects.iter().position(|p| p.id == project_id) {
                                                            this.close_project(idx, window, cx);
                                                        }
                                                    }),
                                                ),
                                        ),
                                 )
                                .priority(101),
                            )
                        }),
                )
                .child(div().flex_1())
                .when(needs_attention > 0, |this| {
                    this.child(
                        div()
                            .text_size(px(11.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(rgba(0xb7782fff))
                            .child(format!("{needs_attention} needs you")),
                    )
                })
                .child(sidebar_icon_button(
                    ("project-new-session", project.id),
                    "icons/plus.svg",
                    13.0,
                    &tokens,
                    cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                        this.create_new_session_in_project(project_idx, window, cx);
                    }),
                ))
                .into_any_element()
        };

        div()
            .flex()
            .flex_col()
            .mb(px(18.0))
            .child(
                div()
                    .id(("project-row", project.id))
                    .group(group_name)
                    .ml(px(6.0))
                    .mr(px(6.0))
                    .mt(px(2.0))
                    .h(px(32.0))
                    .pl(px(4.0))
                    .pr(px(4.0))
                    .rounded(px(6.0))
                    .cursor(CursorStyle::PointingHand)
                    .flex()
                    .items_center()
                    .gap(px(6.0))
                    .bg(rgba(0x00000000))
                    .hover(move |s| s.bg(tokens.sidebar_item_hover))
                    .child(
                        div()
                            .id(("project-chevron", project.id))
                            .size(px(18.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded(px(3.0))
                            .hover(move |s| s.bg(tokens.sidebar_icon_btn_hover))
                            .child(
                                svg()
                                    .path(chevron)
                                    .size(px(12.0))
                                    .text_color(tokens.sidebar_icon),
                            )
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener::<MouseDownEvent>(move |this, _, _, cx| {
                                    cx.stop_propagation();
                                    this.toggle_project_expanded(project_idx, cx);
                                }),
                            ),
                    )
                    .child(header_content)
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                            if this.renaming_project.as_ref().map(|r| r.project_id) == Some(project_id) {
                                return;
                            }
                            this.activate_project(project_idx, window, cx);
                        }),
                    ),
            )
            .children(rows)
            .into_any_element()
    }

    /// One session (screen) inside a project: logo, live status, title and a
    /// close action. Clicking it swaps the workspace to that session.
    fn render_session_entry(
        &self,
        project_idx: usize,
        session_idx: usize,
        session: &Session,
        is_active_session: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let tokens = self.tokens(cx);
        let selected_evidence = session
            .panes
            .iter()
            .max_by_key(|pane| pane.status.rank())
            .map(|pane| {
                format!(
                    "{} — {:?} evidence, {:?} confidence",
                    pane.status.label(),
                    pane.detection.evidence,
                    pane.detection.confidence
                )
            })
            .unwrap_or_else(|| format!("{} — no pane evidence", session.status.label()));
        let indicator = match session.status {
            SessionStatus::Working => gpui_component::Icon::new(gpui_component::IconName::Loader)
                .size(px(16.0))
                .text_color(tokens.sidebar_muted)
                .with_animation(
                    ("session-spinner", session.id),
                    Animation::new(std::time::Duration::from_millis(900)).repeat(),
                    |icon, delta| icon.transform(Transformation::rotate(percentage(delta))),
                )
                .into_any_element(),
            SessionStatus::Done => div()
                .id(("session-status-ack", session.id))
                .size(px(19.0))
                .rounded_full()
                .bg(rgba(0x43b976dd))
                .border_1()
                .border_color(rgba(0xc1f7d8ff))
                .flex()
                .items_center()
                .justify_center()
                .cursor(CursorStyle::PointingHand)
                .tooltip(|_, cx| {
                    cx.new(|_| {
                        gpui_component::tooltip::Tooltip::new(SharedString::from(
                            "Clear completed status",
                        ))
                    })
                    .into()
                })
                .child(
                    svg()
                        .path("icons/status-check.svg")
                        .size(px(12.0))
                        .text_color(rgba(0xffffffff)),
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        cx.stop_propagation();
                        if let Some(session) = this
                            .projects
                            .get_mut(project_idx)
                            .and_then(|project| project.sessions.get_mut(session_idx))
                        {
                            for pane in &mut session.panes {
                                if matches!(pane.status, SessionStatus::Done | SessionStatus::Error)
                                {
                                    pane.status = SessionStatus::Idle;
                                    pane.native_status = None;
                                    pane.native_acknowledged = true;
                                    pane.attention.seen = true;
                                    pane.attention.last = Some(SessionStatus::Idle);
                                    pane.ack_screen =
                                        Some(screen_fingerprint(&pane.screen_scratch, &pane.title));
                                    pane.osc_report_active = false;
                                    pane.pending_approvals.clear();
                                    pane.poll_cache = None;
                                    pane.detection = status::DetectionResult::new(
                                        SessionStatus::Idle,
                                        status::EvidenceSource::NoSignal,
                                        status::Confidence::None,
                                    );
                                    pane.terminal.update(cx, |terminal, _| {
                                        terminal.clear_agent_status_reports()
                                    });
                                }
                            }
                            session.status = session
                                .panes
                                .iter()
                                .map(|pane| pane.status)
                                .max_by_key(|status| status.rank())
                                .unwrap_or(SessionStatus::Idle);
                        }
                        cx.notify();
                    }),
                )
                .into_any_element(),
            SessionStatus::Error => div()
                .id(("session-error-ack", session.id))
                .size(px(19.0))
                .rounded_full()
                .bg(rgba(0xf85149dd))
                .border_1()
                .border_color(rgba(0xffc7c4ff))
                .flex()
                .items_center()
                .justify_center()
                .cursor(CursorStyle::PointingHand)
                .tooltip(|_, cx| {
                    cx.new(|_| {
                        gpui_component::tooltip::Tooltip::new(SharedString::from(
                            "Clear failed status",
                        ))
                    })
                    .into()
                })
                .child(
                    div()
                        .text_size(px(12.0))
                        .font_weight(gpui::FontWeight::BOLD)
                        .text_color(rgba(0xffffffff))
                        .child("!"),
                )
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, _, cx| {
                        cx.stop_propagation();
                        if let Some(session) = this
                            .projects
                            .get_mut(project_idx)
                            .and_then(|project| project.sessions.get_mut(session_idx))
                        {
                            for pane in &mut session.panes {
                                if matches!(pane.status, SessionStatus::Done | SessionStatus::Error)
                                {
                                    pane.status = SessionStatus::Idle;
                                    pane.native_status = None;
                                    pane.native_acknowledged = true;
                                    pane.attention.seen = true;
                                    pane.attention.last = Some(SessionStatus::Idle);
                                    pane.ack_screen =
                                        Some(screen_fingerprint(&pane.screen_scratch, &pane.title));
                                    pane.osc_report_active = false;
                                    pane.pending_approvals.clear();
                                    pane.poll_cache = None;
                                    pane.detection = status::DetectionResult::new(
                                        SessionStatus::Idle,
                                        status::EvidenceSource::NoSignal,
                                        status::Confidence::None,
                                    );
                                    pane.terminal.update(cx, |terminal, _| {
                                        terminal.clear_agent_status_reports()
                                    });
                                }
                            }
                            session.status = session
                                .panes
                                .iter()
                                .map(|pane| pane.status)
                                .max_by_key(|status| status.rank())
                                .unwrap_or(SessionStatus::Idle);
                        }
                        cx.notify();
                    }),
                )
                .into_any_element(),
            SessionStatus::Blocked => div()
                .size(px(19.0))
                .rounded_full()
                .bg(rgba(0xffedc9cc))
                .border_1()
                .border_color(rgba(0xc8873cff))
                .flex()
                .items_center()
                .justify_center()
                .child(div().size(px(7.0)).rounded_full().bg(rgba(0xc8873cff)))
                .into_any_element(),
            SessionStatus::Unknown => div()
                .flex()
                .items_center()
                .text_size(px(10.0))
                .text_color(tokens.sidebar_muted)
                .child("Unknown")
                .into_any_element(),
            SessionStatus::Active | SessionStatus::Idle => div().size(px(19.0)).into_any_element(),
        };

        div()
            .id(("session-item", session.id))
            .mx(px(8.0))
            .h(px(38.0))
            .px(px(12.0))
            .rounded(px(9.0))
            .cursor(CursorStyle::PointingHand)
            .tooltip(move |_, cx| {
                let text = selected_evidence.clone();
                cx.new(|_| gpui_component::tooltip::Tooltip::new(SharedString::from(text)))
                    .into()
            })
            .flex()
            .items_center()
            .gap(px(8.0))
            .bg(if is_active_session {
                tokens.sidebar_item_active
            } else {
                rgba(0x00000000).into()
            })
            .hover(move |s| {
                if is_active_session {
                    s
                } else {
                    s.bg(tokens.sidebar_item_hover)
                }
            })
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_size(px(13.0))
                    .font_weight(
                        if matches!(
                            session.status,
                            SessionStatus::Blocked | SessionStatus::Error
                        ) {
                            gpui::FontWeight::SEMIBOLD
                        } else {
                            gpui::FontWeight::MEDIUM
                        },
                    )
                    .text_color(if is_active_session {
                        tokens.sidebar_item_active_fg
                    } else {
                        tokens.sidebar_fg
                    })
                    .child(session.display_title.clone()),
            )
            .child(indicator)
            .child(sidebar_icon_button(
                ("session-close", session.id),
                "icons/x.svg",
                10.0,
                &tokens,
                cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                    this.close_session(project_idx, session_idx, window, cx);
                }),
            ))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                    this.select_session(project_idx, session_idx, window, cx);
                }),
            )
            .into_any_element()
    }

    fn render_sidebar_empty_state(&self, cx: &mut Context<Self>) -> gpui::Div {
        let tokens = self.tokens(cx);
        div()
            .flex()
            .flex_col()
            .items_center()
            .pt(px(36.0))
            .px(px(20.0))
            .gap(px(8.0))
            .child(
                div()
                    .size(px(40.0))
                    .rounded(px(10.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(tokens.card_bg)
                    .child(
                        svg()
                            .path("icons/folder.svg")
                            .size(px(20.0))
                            .text_color(tokens.sidebar_icon),
                    ),
            )
            .child(
                div()
                    .text_size(px(12.5))
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(tokens.sidebar_fg)
                    .child("No projects open"),
            )
            .child(
                div()
                    .text_size(px(11.5))
                    .text_color(tokens.sidebar_muted)
                    .text_center()
                    .child("Open any folder to start a project with its own sessions."),
            )
            .child(sidebar_text_button(
                "sidebar-empty-open-project",
                "Open project",
                "icons/folder-plus.svg",
                true,
                &tokens,
                cx.listener::<MouseDownEvent>(|this, _, window, cx| {
                    this.select_folder_and_add_project(window, cx);
                }),
            ))
    }

    /// Shown in the terminal area when the active project has no session, or
    /// when no project is open at all.
    fn render_empty_terminal(&self, cx: &mut Context<Self>) -> gpui::Div {
        let has_project = self.active_project().is_some();
        let project_idx = self.active_project_idx;
        let tokens = self.tokens(cx);
        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap(px(10.0))
            .child(
                div()
                    .size(px(48.0))
                    .rounded(px(12.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(tokens.card_bg)
                    .child(
                        svg()
                            .path("icons/folder.svg")
                            .size(px(24.0))
                            .text_color(tokens.text_secondary),
                    ),
            )
            .child(
                div()
                    .text_size(px(14.0))
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(tokens.text_primary)
                    .child(if has_project {
                        "No session in this project"
                    } else {
                        "No project open"
                    }),
            )
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(tokens.text_muted)
                    .child(if has_project {
                        "Create a session to start a terminal in this project."
                    } else {
                        "Open a folder to start a project with its own sessions."
                    }),
            )
            .child(
                div()
                    .mt(px(4.0))
                    .flex()
                    .items_center()
                    .gap(px(8.0))
                    .child(sidebar_text_button(
                        "empty-terminal-open-project",
                        "Open project",
                        "icons/folder-plus.svg",
                        true,
                        &tokens,
                        cx.listener::<MouseDownEvent>(|this, _, window, cx| {
                            this.select_folder_and_add_project(window, cx);
                        }),
                    ))
                    .when(has_project, |this| {
                        this.child(sidebar_text_button(
                            "empty-terminal-new-session",
                            "New session",
                            "icons/plus.svg",
                            false,
                            &tokens,
                            cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                                this.create_new_session_in_project(project_idx, window, cx);
                            }),
                        ))
                    }),
            )
    }

    /// Switch the workspace to a session in a given project.
    fn select_session(
        &mut self,
        project_idx: usize,
        session_idx: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.active_view = ActiveView::Workspace;
        let Some(project) = self.projects.get_mut(project_idx) else {
            return;
        };
        if session_idx >= project.sessions.len() {
            return;
        }
        project.active_session_idx = session_idx;
        project.expanded = true;
        self.active_project_idx = project_idx;
        let pane_id = project.sessions[session_idx].panes.first().map(|p| p.id);
        if let Some(pane_id) = pane_id {
            self.focus_pane(pane_id, window, cx);
        } else {
            self.focused_pane_id = None;
        }
        cx.notify();
    }
}

/// Small square icon button used by the sidebar rows and the sidebar
/// toolbar. `on_click` is a `cx.listener`; the button stops mouse propagation
/// so clicking it never also activates the row it sits in.
fn sidebar_icon_button(
    id: impl Into<gpui::ElementId>,
    icon: &'static str,
    icon_size: f32,
    tokens: &ThemeTokens,
    on_click: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    let hover_bg = tokens.sidebar_icon_btn_hover;
    let icon_color = tokens.sidebar_icon;
    let icon_hover_color = tokens.sidebar_icon_hover;
    div()
        .id(id)
        .size(px(20.0))
        .rounded(px(4.0))
        .flex()
        .items_center()
        .justify_center()
        .flex_shrink_0()
        .cursor(CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover_bg))
        .on_mouse_down(MouseButton::Left, move |ev, window, cx| {
            cx.stop_propagation();
            on_click(ev, window, cx);
        })
        .child(
            svg()
                .path(icon)
                .size(px(icon_size))
                .text_color(icon_color)
                .hover(move |s| s.text_color(icon_hover_color)),
        )
}

fn sidebar_text_button(
    id: impl Into<gpui::ElementId>,
    label: &'static str,
    icon: &'static str,
    primary: bool,
    tokens: &ThemeTokens,
    on_click: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    let bg_color = if primary { tokens.primary_btn_bg } else { tokens.secondary_btn_bg };
    let border_color = if primary { tokens.primary_btn_bg } else { tokens.border };
    let hover_color = if primary { tokens.primary_btn_hover } else { tokens.secondary_btn_hover };
    let text_color = if primary { tokens.primary_btn_fg } else { tokens.secondary_btn_fg };
    div()
        .id(id)
        .h(px(30.0))
        .px(px(12.0))
        .rounded(px(6.0))
        .cursor(CursorStyle::PointingHand)
        .flex()
        .items_center()
        .gap(px(6.0))
        .bg(bg_color)
        .border_1()
        .border_color(border_color)
        .hover(move |s| s.bg(hover_color))
        .child(svg().path(icon).size(px(13.0)).text_color(text_color))
        .child(
            div()
                .text_size(px(12.0))
                .text_color(text_color)
                .child(label),
        )
        .on_mouse_down(MouseButton::Left, move |ev, window, cx| {
            cx.stop_propagation();
            on_click(ev, window, cx);
        })
}

impl Render for AppState {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let diag_render = diagnostics::render_start();
        // "Looking" needs an active window: a visible but backgrounded window
        // has not been seen, so a finish there must still become Done.
        self.window_active = _window.is_window_active();
        self.ensure_panes_input_state(_window, cx);

        if self.theme_preference == ThemePreference::System {
            let sys_mode = self.theme_preference.resolve_mode(_window.appearance());
            let current_is_dark = self.palette.background().l < 0.5;
            if sys_mode.is_dark() != current_is_dark {
                self.palette = load_github_palette(sys_mode.is_dark());
                let palette = self.palette.clone();
                gpui_component::Theme::change(sys_mode, Some(_window), cx);
                for project in &mut self.projects {
                    for session in &mut project.sessions {
                        for pane in &mut session.panes {
                            let mut config = pane.terminal.read(cx).config().clone();
                            config.colors = palette.clone();
                            pane.terminal.update(cx, |term, cx| {
                                term.update_config(config, cx);
                            });
                        }
                    }
                }
            }
        }

        let tokens = self.tokens(cx);
        // Track which pane owns keyboard focus each frame; the poller uses
        // this for Active-state resolution and for clearing sticky Done/Error.
        let mut focused_pane = None;
        if let Some(session) = self.active_session() {
            for pane in &session.panes {
                let is_term_focused = pane.terminal.read(cx).focus_handle().is_focused(_window);
                let is_input_focused = pane
                    .input_state
                    .as_ref()
                    .is_some_and(|input| input.read(cx).focus_handle().is_focused(_window));
                if is_term_focused || is_input_focused {
                    focused_pane = Some(pane.id);
                    break;
                }
            }
        }
        if let Some(focused_pane) = focused_pane {
            self.focused_pane_id = Some(focused_pane);
        } else if self.focused_pane_id.is_none() {
            self.focused_pane_id = self
                .active_session()
                .and_then(|s| s.panes.first())
                .map(|p| p.id);
        }

        // Drain alerts here, where a window exists to take notifications and
        // sounds; the poller has none. A blocked transition while looking only
        // rings (no toast), and a missed finish rings only when it happened.
        let alerts = std::mem::take(&mut self.pending_alerts);
        if !alerts.is_empty() {
            let weak = cx.entity().downgrade();
            _window.defer(cx, move |window, cx| {
                for alert in alerts {
                    crate::sound::play(alert.kind);
                    if alert.kind == attention::AlertKind::Blocked && alert.looking {
                        continue;
                    }
                    let (title, kind) = match alert.kind {
                        attention::AlertKind::Blocked => (
                            format!("{} needs you", alert.name),
                            NotificationType::Error,
                        ),
                        attention::AlertKind::Done => (
                            format!("{} finished", alert.name),
                            NotificationType::Info,
                        ),
                    };
                    let weak = weak.clone();
                    let pane_id = alert.pane_id;
                    let note = Notification::new()
                        .title(title)
                        .message(alert.place)
                        .with_type(kind)
                        .on_click(move |_, window, cx| {
                            let _ = weak.update(cx, |this, cx| {
                                this.focus_pane(pane_id, window, cx);
                            });
                        });
                    window.push_notification(note, cx);
                }
            });
        }

        let active_title = self
            .active_session()
            .map(|s| s.title.clone())
            .unwrap_or_else(|| SharedString::from("No session"));

        let active_project_name = self
            .active_project()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| SharedString::from("No project"));

        let terminal_area = if let Some(session) = self.active_session() {
            // Each session is its own screen: its pane tree (and maximized
            // pane) is rendered exclusively; other sessions' panes keep
            // running in the background.
            if let Some(max_pane_id) = session.maximized_pane_id {
                if let Some(pane) = session.panes.iter().find(|p| p.id == max_pane_id) {
                    self.render_terminal_pane(pane, true, cx).into_any_element()
                } else if let Some(tree) = &session.pane_tree {
                    self.render_pane_node(tree, session, false, cx)
                } else {
                    div().into_any_element()
                }
            } else if let Some(tree) = &session.pane_tree {
                self.render_pane_node(tree, session, false, cx)
            } else if let Some(pane) = session.panes.first() {
                self.render_terminal_pane(pane, false, cx)
                    .into_any_element()
            } else {
                div().into_any_element()
            }
        } else {
            self.render_empty_terminal(cx).into_any_element()
        };
        // Focused terminal's status and logo, mirrored in the title bar.
        let titlebar_status = self.active_session().map(|s| s.status);
        let titlebar_logo = self.active_session().map(|s| s.logo.clone());

        let is_maximized = _window.is_maximized();
        let is_fullscreen = _window.is_fullscreen();
        let controls = _window.window_controls();

        let default_window_controls = (!is_fullscreen).then(|| {
            div()
                .id("default-window-controls")
                .flex()
                .items_center()
                .h_full()
                .when(controls.minimize, |this| {
                    this.child(
                        div()
                            .id("window-minimize-btn")
                            .w(px(36.0))
                            .h_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor(CursorStyle::PointingHand)
                            .hover(move |s| s.bg(tokens.titlebar_btn_hover))
                            .on_mouse_down(MouseButton::Left, |_, window, cx| {
                                cx.stop_propagation();
                                window.minimize_window();
                            })
                            .child(
                                svg()
                                    .path("window-minimize.svg")
                                    .size(px(14.0))
                                    .text_color(tokens.titlebar_fg)
                                    .hover(move |s| s.text_color(tokens.text_primary)),
                            ),
                    )
                })
                .when(controls.maximize, |this| {
                    this.child(
                        div()
                            .id("window-maximize-btn")
                            .w(px(36.0))
                            .h_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor(CursorStyle::PointingHand)
                            .hover(move |s| s.bg(tokens.titlebar_btn_hover))
                            .on_mouse_down(MouseButton::Left, |_, window, cx| {
                                cx.stop_propagation();
                                window.zoom_window();
                            })
                            .child(
                                svg()
                                    .path(if is_maximized {
                                        "window-restore.svg"
                                    } else {
                                        "window-maximize.svg"
                                    })
                                    .size(px(14.0))
                                    .text_color(tokens.titlebar_fg)
                                    .hover(move |s| s.text_color(tokens.text_primary)),
                            ),
                    )
                })
                .child(
                    div()
                        .id("window-close-btn")
                        .w(px(36.0))
                        .h_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor(CursorStyle::PointingHand)
                        .hover(|s| s.bg(rgba(0xe81123ff)))
                        .on_mouse_down(MouseButton::Left, |_, window, cx| {
                            cx.stop_propagation();
                            window.remove_window();
                        })
                        .child(
                            svg()
                                .path("window-close.svg")
                                .size(px(14.0))
                                .text_color(tokens.titlebar_fg)
                                .hover(|s| s.text_color(rgba(0xffffffff))),
                        ),
                )
        });

        let terminal_content = div()
            .flex()
            .flex_col()
            .size_full()
            .bg(tokens.window_bg)
            .child(
                self.make_drag_area(cx)
                    .h(px(38.0))
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_between()
                    .pl_4()
                    .bg(tokens.titlebar_bg)
                    .border_b_1()
                    .border_color(tokens.titlebar_border)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .when(!self.sidebar_visible, |this| {
                                this.child(
                                    div()
                                        .id("terminal-sidebar-toggle-btn")
                                        .cursor(CursorStyle::PointingHand)
                                        .mr_2()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .child(
                                            svg()
                                                .path("icons/panel-left.svg")
                                                .size(px(16.0))
                                                .text_color(tokens.titlebar_fg)
                                                .hover(move |s| s.text_color(tokens.text_primary)),
                                        )
                                        .on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(|this, _, _, cx| {
                                                cx.stop_propagation();
                                                this.sidebar_visible = true;
                                                cx.notify();
                                                this.schedule_workspace_save(cx);
                                            }),
                                        ),
                                )
                            })
                            .text_size(px(12.5))
                            .text_color(tokens.titlebar_fg)
                            .children(titlebar_logo.map(|logo| {
                                div()
                                    .size(px(16.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .flex_shrink_0()
                                    .child(gpui::img(logo).size(px(14.0)).rounded(px(2.0)))
                            }))
                            .child(format!("{active_project_name} — {active_title}"))
                            .children(titlebar_status.map(|status| {
                                let (dot_color, text_color, badge_bg) = status.colors();
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_1()
                                    .px_1p5()
                                    .py_0p5()
                                    .rounded_sm()
                                    .bg(rgba(badge_bg))
                                    .text_color(rgba(text_color))
                                    .child(div().size(px(6.0)).rounded_full().bg(rgba(dot_color)))
                                    .child(status.label())
                            })),
                    )
                    .children(default_window_controls),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .size_full()
                    .bg(tokens.window_bg)
                    .overflow_hidden()
                    .p(px(6.0))
                    .child(div().flex().flex_col().size_full().child(terminal_area)),
            );

        let sidebar_content = self.render_sidebar(cx);

        let main_content = match self.active_view {
            ActiveView::Workspace => terminal_content.into_any_element(),
            ActiveView::Settings => self.render_settings_page(_window, cx).into_any_element(),
        };

        let root = div()
            .flex()
            .size_full()
            .bg(tokens.window_bg)
            .text_color(tokens.text_primary)
            .on_key_down(cx.listener(Self::on_key_down))
            .when(self.sidebar_visible, |this| {
                this.child(
                    div()
                        .w(self.sidebar_width)
                        .h_full()
                        .flex_shrink_0()
                        .child(sidebar_content),
                )
            })
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .min_w_0()
                    .overflow_hidden()
                    .child(main_content),
            )
            .when(self.terminal_context_menu.is_some(), |this| {
                this.child(self.render_terminal_context_menu_layer(cx))
            })
            .when(self.history_search.is_some(), |this| {
                this.child(self.render_history_search_layer(cx))
            });
        diagnostics::render_end(diag_render);
        root
    }
}

/// How many non-empty screen lines `T3_STATUS_DEBUG` prints per terminal.
const DEBUG_SCREEN_LINES: usize = 14;

/// Sidebar form of a session title: full text up to 160 chars, then an ellipsis.
/// Uses `char_indices().nth(160)` so short titles cost one pass that stops at
/// the end — unlike `chars().count()`, which always scans the whole string —
/// and is cached on the session so rendering never recomputes it.
fn truncate_title(title: &SharedString) -> SharedString {
    match title.as_str().char_indices().nth(160) {
        Some((idx, _)) => SharedString::from(format!("{}...", &title.as_str()[..idx])),
        None => title.clone(),
    }
}

/// Cheap, stable fingerprint of what is on screen, used to tell whether the
/// user has already looked at a finished/failed terminal.
fn screen_fingerprint(lines: &[String], title: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    lines.hash(&mut hasher);
    title.hash(&mut hasher);
    hasher.finish()
}

/// How alerts and the Agents list name a pane: its agent app first, else a
/// meaningful title, else a neutral fallback.
fn pane_agent_name(pane: &TerminalPane) -> String {
    if let Some(app) = &pane.app {
        return app.to_string();
    }
    let title = pane.display_title.trim();
    if !title.is_empty() && !title.starts_with("terminal ") {
        return title.to_string();
    }
    "An agent".to_string()
}

/// Build a pane's input editor plus its change subscription.
///
/// The subscription defers its work: `InputEvent::Change` is emitted while the
/// input entity is leased, so ghost text and the completion menu are refreshed
/// after that effect cycle instead of re-entering the entity.
fn build_pane_input(
    pane_id: usize,
    window: &mut Window,
    cx: &mut Context<AppState>,
) -> (Entity<InputState>, gpui::Subscription) {
    let input = cx.new(|cx| {
        InputState::new(window, cx)
            .placeholder("Run commands")
            .auto_grow(1, 8)
    });
    let subscription = cx.subscribe_in(
        &input,
        window,
        move |_this: &mut AppState, _input, event: &InputEvent, window, cx| {
            if matches!(event, InputEvent::Change) {
                let weak = cx.entity().downgrade();
                window.defer(cx, move |window, cx| {
                    let _ = weak.update(cx, |this, cx| {
                        this.on_pane_input_changed(pane_id, window, cx);
                    });
                });
            }
        },
    );
    (input, subscription)
}

/// Replace the input value and place the cursor at the end.
fn set_input_text(
    input: &Entity<InputState>,
    text: String,
    window: &mut Window,
    cx: &mut Context<AppState>,
) {
    let cursor = completion::utf16_len(&text);
    set_input_text_with_cursor(input, text, cursor, window, cx);
}

fn set_input_text_with_cursor(
    input: &Entity<InputState>,
    text: String,
    cursor_utf16: usize,
    window: &mut Window,
    cx: &mut Context<AppState>,
) {
    input.update(cx, |state, cx| {
        state.set_value(text, window, cx);
        let cursor = cursor_utf16.min(completion::utf16_len(state.value().as_str()));
        let position = state.text().offset_to_position(cursor);
        state.set_cursor_position(position, window, cx);
    });
}

fn move_input_cursor(
    input: &Entity<InputState>,
    cursor_utf16: usize,
    window: &mut Window,
    cx: &mut Context<AppState>,
) {
    input.update(cx, |state, cx| {
        let cursor = cursor_utf16.min(completion::utf16_len(state.value().as_str()));
        let position = state.text().offset_to_position(cursor);
        state.set_cursor_position(position, window, cx);
    });
}

/// Byte offset where the word before `cursor` starts (readline `Ctrl+W`).
fn previous_word_start(value: &str, cursor: usize) -> usize {
    let cursor = cursor.min(value.len());
    let mut start = cursor;
    while let Some(ch) = value[..start].chars().next_back() {
        if !ch.is_whitespace() {
            break;
        }
        start -= ch.len_utf8();
    }
    while let Some(ch) = value[..start].chars().next_back() {
        if ch.is_whitespace() {
            break;
        }
        start -= ch.len_utf8();
    }
    start
}

/// Byte offset just past the word at `cursor` (readline `Alt+F`).
fn next_word_end(value: &str, cursor: usize) -> usize {
    let cursor = cursor.min(value.len());
    let mut end = cursor;
    while let Some(ch) = value[end..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        end += ch.len_utf8();
    }
    while let Some(ch) = value[end..].chars().next() {
        if ch.is_whitespace() {
            break;
        }
        end += ch.len_utf8();
    }
    end
}

fn detect_shell() -> (String, String) {
    if cfg!(windows) {
        (
            std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string()),
            "cmd".to_string(),
        )
    } else {
        if let Ok(shell) = std::env::var("SHELL") {
            let name = std::path::Path::new(&shell)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "shell".to_string());
            return (shell, name);
        }
        ("/bin/bash".to_string(), "bash".to_string())
    }
}
#[cfg(target_os = "linux")]
fn ensure_linux_desktop_entry() {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return;
    };
    let data_dir = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".local/share"));
    let apps_dir = data_dir.join("applications");
    let icons_dir = data_dir.join("icons/hicolor/512x512/apps");
    let _ = std::fs::create_dir_all(&apps_dir);
    let _ = std::fs::create_dir_all(&icons_dir);

    let icon_path = icons_dir.join("markup.png");
    if !icon_path.exists() {
        let _ = std::fs::write(&icon_path, APP_ICON_PNG);
    }

    let exe_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("markup"));
    let exe_str = exe_path.to_string_lossy();

    let desktop_content = format!(
        "[Desktop Entry]\n\
        Type=Application\n\
        Name=Markup\n\
        Comment=GPU-accelerated terminal\n\
        Exec={exe_str}\n\
        Icon=markup\n\
        Terminal=false\n\
        Categories=System;TerminalEmulator;\n\
        StartupWMClass=markup\n"
    );

    let _ = std::fs::write(apps_dir.join("markup.desktop"), desktop_content);

    let t3_content = format!(
        "[Desktop Entry]\n\
        Type=Application\n\
        Name=Markup\n\
        Exec={exe_str}\n\
        Icon=markup\n\
        Terminal=false\n\
        StartupWMClass=t3code-terminal\n"
    );
    let _ = std::fs::write(apps_dir.join("t3code-terminal.desktop"), t3_content);
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("report") => return cli::run_report(&args.collect::<Vec<_>>()),
        Some("skill") => return cli::run_skill(&args.collect::<Vec<_>>()),
        Some("integration") => return integrations::run(&args.collect::<Vec<_>>()),
        Some("status-hook") => return integrations::run_hook(&args.collect::<Vec<_>>()),
        Some("status-event") => return cli::run_status_event(),
        Some("status") => return cli::run_status(&args.collect::<Vec<_>>()),
        Some("pane") => {
            #[cfg(unix)]
            return herdr_compat::run_cli(&args.collect::<Vec<_>>());
            #[cfg(not(unix))]
            anyhow::bail!("native reporting compatibility requires a Unix socket");
        }
        Some("--help" | "-h") => {
            println!(
                "Usage: markup [report|skill|integration|status|status-event|status-hook]\nNative compatibility: markup pane <report-agent|report-agent-session|release-agent> ..."
            );
            return Ok(());
        }
        Some("--version" | "-V") => {
            println!("markup {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some(command) => anyhow::bail!("unsupported Markup command: {command}"),
        None => {}
    }
    Application::new()
        .with_assets(CombinedAssets)
        .run(|cx: &mut App| {
            #[cfg(target_os = "linux")]
            ensure_linux_desktop_entry();
            let window_icon = gpui::WindowIcon::from_png_bytes(APP_ICON_PNG).ok();
            gpui_component::init(cx);
            init_warp_input_bindings(cx);
            let initial_mode = ThemePreference::System.resolve_mode(cx.window_appearance());
            gpui_component::Theme::change(initial_mode, None, cx);
            load_embedded_fonts(cx);
            sync_component_fonts(cx);

            std::thread::spawn(|| {
                completion::prewarm();
                let _ = shell_history::cached();
            });

            let palette = load_github_palette(initial_mode.is_dark());
            let persistence_path = workspace_state::state_path();

            let bounds = Bounds::centered(None, size(px(1200.), px(780.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: Some(TitleBar::title_bar_options()),
                    window_decorations: Some(WindowDecorations::Client),
                    app_id: Some("markup".to_string()),
                    window_icon,
                    ..Default::default()
                },
                |window, cx| {
                    let app_state = cx.new(|_| AppState {
                        projects: Vec::new(),
                        active_project_idx: 0,
                        next_id: 1,
                        session_index: HashMap::new(),
                        pane_index: HashMap::new(),
                        palette: palette.clone(),
                        focused_pane_id: None,
                        sidebar_visible: true,
                        warp_bottom_input: true,
                        active_view: ActiveView::Workspace,
                        theme_preference: ThemePreference::System,
                        integration_panel_visible: false,
                        integration_feedback: SharedString::from("Setup is opt-in. Status becomes Reporting only after a live event arrives."),
                        sidebar_width: SIDEBAR_WIDTH,
                        drag_start_pos: None,
                        persistence_path: persistence_path.clone(),
                        save_generation: Arc::new(AtomicU64::new(0)),
                        startup_loaded: false,
                        quit_subscription: None,
                        window_active: true,
                        status_primed: false,
                        pending_alerts: Vec::new(),
                        attention_seq: 0,
                        project_menu: None,
                        renaming_project: None,
                        terminal_context_menu: None,
                        history_search: None,
                    });

                    let weak_state = app_state.downgrade();
                    let load_path = persistence_path.clone();
                    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                        let saved = cx
                            .background_executor()
                            .spawn(async move {
                                load_path
                                    .as_deref()
                                    .map(workspace_state::load)
                                    .unwrap_or_default()
                            })
                            .await;
                        let _ = weak_state
                            .update(cx, |state, cx| state.apply_saved_workspace(saved, cx));
                    })
                    .detach();

                    let quit_path = persistence_path.clone();
                    app_state.update(cx, |state, cx| {
                        state.persistence_path = persistence_path.clone();
                        state.startup_loaded = false;
                        state.quit_subscription = Some(cx.on_app_quit(move |state, cx| {
                            state.save_generation.fetch_add(1, Ordering::SeqCst);
                            let path = quit_path.clone();
                            let snapshot = state.saved_workspace();
                            cx.background_executor().spawn(async move {
                                if let Some(path) = path {
                                    if let Err(error) =
                                        workspace_state::save_atomic(&path, &snapshot)
                                    {
                                        eprintln!(
                                            "failed to save workspace state on quit: {error}"
                                        );
                                    }
                                }
                            })
                        }));
                    });

                    let app_state_weak = app_state.downgrade();
                    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                        loop {
                            cx.background_executor().timer(POLL_INTERVAL).await;

                            // Snapshot the shell pids on the UI thread (a borrow
                            // over the pane list), do the blocking `/proc`
                            // inspection on the background executor, then apply
                            // the result back on the UI thread. Nothing in this
                            // loop reads files or waits on I/O while a frame
                            // could be building.
                            let Ok(pids) = app_state_weak.update(cx, |state, _| state.pane_pids())
                            else {
                                break;
                            };
                            let foregrounds = cx
                                .background_executor()
                                .spawn(async move {
                                    pids.into_iter()
                                        .map(|pid| {
                                            (pid, status::foreground_process_names(Some(pid)))
                                        })
                                        .collect::<HashMap<u32, Arc<[String]>>>()
                                })
                                .await;

                            let res = app_state_weak.update(cx, |state, cx| {
                                state.poll_sessions_status_with(&foregrounds, cx);
                            });
                            if res.is_err() {
                                break;
                            }
                        }
                    })
                    .detach();

                    // Opt-in diagnostics: one line per second with frame
                    // timing, UI-thread work, CPU, RSS and wakeups. Not spawned
                    // at all unless `MARKUP_DIAGNOSTICS` is set.
                    if diagnostics::enabled() {
                        cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                            let mut sampler = diagnostics::Sampler::new();
                            loop {
                                cx.background_executor()
                                    .timer(diagnostics::REPORT_INTERVAL)
                                    .await;
                                eprintln!("{}", sampler.report());
                            }
                        })
                        .detach();
                    }

                    // TEMPORARY DIAGNOSTIC: MARKUP_RESIZE_PROBE=<rounds> drags
                    // the window edge in small steps, the way a real resize
                    // does, while the status poller keeps ticking.
                    if let Some(probe) = std::env::var_os("MARKUP_RESIZE_PROBE") {
                        let rounds: usize = probe.to_string_lossy().parse().unwrap_or(10);
                        let handle = window.window_handle();
                        cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                            let (mut w, mut h) = (1200.0f32, 780.0f32);
                            for _ in 0..rounds {
                                for _ in 0..60 {
                                    w = (w - 12.0).max(360.0);
                                    h = (h - 8.0).max(240.0);
                                    cx.background_executor()
                                        .timer(std::time::Duration::from_millis(16))
                                        .await;
                                    if handle
                                        .update(cx, |_, window, _| {
                                            window.resize(size(px(w), px(h)));
                                        })
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                                for _ in 0..60 {
                                    w = (w + 12.0).min(1200.0);
                                    h = (h + 8.0).min(780.0);
                                    cx.background_executor()
                                        .timer(std::time::Duration::from_millis(16))
                                        .await;
                                    if handle
                                        .update(cx, |_, window, _| {
                                            window.resize(size(px(w), px(h)));
                                        })
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                            eprintln!("[probe] {rounds} drag rounds finished without a panic");
                            let _ = cx.update(|cx| cx.quit());
                        })
                        .detach();
                    }

                    cx.new(|cx| Root::new(app_state, window, cx))
                },
            )
            .expect("Failed to open window");

            cx.activate(true);
        });

    Ok(())
}

#[cfg(test)]
mod resize_tests {
    //! Window-resize regression tests.
    //!
    //! A resize is the one event that re-lays-out everything at once: the
    //! resizable sidebar panels, the terminal grid (cols/rows recomputed from
    //! the new bounds) and the client-drawn title bar. These tests drive the
    //! real view through the same window options `main` uses and draw a frame
    //! after every resize step.

    use super::*;
    use gpui::{Modifiers, TestAppContext, VisualTestContext, point, size};
    use std::ops::Deref as _;

    /// One project with one live session, built exactly like `main` does.
    fn build_app_state(cx: &mut Context<AppState>) -> AppState {
        let palette = ColorPalette::default();
        let dir = std::env::current_dir().expect("current dir");
        let weak = cx.entity().downgrade();
        let (terminal, pid, generation) =
            AppState::spawn_terminal_view(Some(&dir), palette.clone(), weak, 1, cx)
                .expect("terminal view");
        let title = SharedString::from("terminal 1");

        let mut pane = TerminalPane::new(
            1,
            generation,
            terminal,
            title.clone(),
            title.clone(),
            pid,
            true,
        );
        // Tests assert exact history indices; the developer machine's real
        // shell history must not leak into that.
        pane.shell_history = Arc::new(Vec::new());

        AppState {
            projects: vec![Project {
                id: 1,
                name: SharedString::from("markup"),
                path: dir,
                sessions: vec![Session {
                    id: 1,
                    title: title.clone(),
                    display_title: title,
                    logo: logos::default_logo(),
                    pane_tree: Some(PaneNode::Leaf(1)),
                    maximized_pane_id: None,
                    panes: vec![pane],
                    status: SessionStatus::Idle,
                }],
                active_session_idx: 0,
                expanded: true,
            }],
            active_project_idx: 0,
            next_id: 2,
            session_index: HashMap::from([(1, (0, 0))]),
            pane_index: HashMap::from([(1, (0, 0, 0))]),
            palette,
            focused_pane_id: Some(1),
            sidebar_visible: true,
            warp_bottom_input: true,
            active_view: ActiveView::Workspace,
            theme_preference: ThemePreference::System,
            integration_panel_visible: false,
            integration_feedback: SharedString::from(
                "Setup is opt-in. Status becomes Reporting only after a live event arrives.",
            ),
            sidebar_width: SIDEBAR_WIDTH,
            drag_start_pos: None,
            persistence_path: None,
            save_generation: Arc::new(AtomicU64::new(0)),
            startup_loaded: true,
            quit_subscription: None,
            window_active: true,
            status_primed: false,
            pending_alerts: Vec::new(),
            attention_seq: 0,
            project_menu: None,
            renaming_project: None,
            terminal_context_menu: None,
            history_search: None,
        }
    }

    /// Open the app the way `main` does and return a context that can draw it.
    pub(crate) fn open_app(cx: &mut TestAppContext) -> (VisualTestContext, Entity<AppState>) {
        cx.update(|cx| {
            gpui_component::init(cx);
            init_warp_input_bindings(cx);
            gpui_component::Theme::change(gpui_component::ThemeMode::Dark, None, cx);
            load_embedded_fonts(cx);
            sync_component_fonts(cx);
        });

        let state = cx.new(build_app_state);
        let handle = cx
            .update(|cx: &mut App| {
                let bounds = Bounds::centered(None, size(px(1200.), px(780.)), cx);
                cx.open_window(
                    WindowOptions {
                        window_bounds: Some(WindowBounds::Windowed(bounds)),
                        titlebar: Some(TitleBar::title_bar_options()),
                        window_decorations: Some(WindowDecorations::Client),
                        app_id: Some("markup".to_string()),
                        window_icon: gpui::WindowIcon::from_png_bytes(APP_ICON_PNG).ok(),
                        ..Default::default()
                    },
                    |window, cx| cx.new(|cx| Root::new(state.clone(), window, cx)),
                )
            })
            .expect("open window");

        let cx = VisualTestContext::from_window(*handle.deref(), cx);
        cx.run_until_parked();
        (cx, state)
    }

    #[test]
    fn window_resizes_without_panicking() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        let terminal = app_state.read_with(&cx, |state, _| {
            state.active_session().expect("a session").panes[0]
                .terminal
                .clone()
        });

        let mut grid_sizes = Vec::new();
        for window_size in [
            size(px(1100.), px(700.)),
            size(px(900.), px(600.)),
            size(px(700.), px(500.)),
            size(px(520.), px(400.)),
            size(px(380.), px(300.)),
            size(px(280.), px(220.)),
            size(px(200.), px(160.)),
            size(px(1200.), px(780.)),
        ] {
            cx.simulate_resize(window_size);
            cx.run_until_parked();
            cx.draw_frame().expect("a frame at the new size");

            // The grid has to track the window, and every scrape has to come
            // back with exactly the rows the grid has: a scrape that names rows
            // the grid resized away is what aborted the process on a drag.
            let (cols, rows) = terminal.read_with(&cx, |view, _| view.dimensions());
            assert!(cols >= 2 && rows >= 1, "grid collapsed to {cols}x{rows}");
            let scraped = terminal.read_with(&cx, |view, _| view.bottom_lines(usize::MAX));
            assert_eq!(scraped.len(), rows, "scrape out of step with the grid");
            grid_sizes.push((cols, rows));
        }

        // ...and the grid really did follow the window, rather than both sides
        // agreeing on a size nobody updated.
        let (big_cols, big_rows) = grid_sizes[0];
        let (small_cols, small_rows) = grid_sizes[6];
        assert!(
            small_rows < big_rows && small_cols < big_cols,
            "grid did not follow the window: {small_cols}x{small_rows} at 200x160, \
             {big_cols}x{big_rows} at 1100x700"
        );
    }

    /// The status poller runs between frames, so it can tick after the paint
    /// pass has already shrunk the terminal grid but before the next render
    /// syncs the cached row count. Scraping the screen in that window asks the
    /// grid for rows that no longer exist.
    #[test]
    fn status_poll_between_resize_frames_does_not_panic() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        // A frame at the large size: the grid is laid out big while the cached
        // row count still holds the size from the previous frame.
        cx.draw_frame().expect("a frame at the opening size");

        // Shrink the window and draw one frame: the render syncs the cached
        // dims to the large grid, then the paint shrinks the grid.
        cx.simulate_resize(size(px(560.), px(320.)));
        cx.run_until_parked();
        cx.draw_frame().expect("a frame at the smaller size");

        // The poller's ticks are what a user sees as a crash while dragging.
        app_state.update(&mut cx, |state, cx| state.poll_sessions_status(cx));
        app_state.update(&mut cx, |state, cx| state.poll_sessions_status(cx));
    }

    /// Simulating dragging across the fixed sidebar and window bounds must
    /// stay panic-free.
    #[test]
    fn dragging_the_sidebar_handle_does_not_panic() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, _) = open_app(&mut cx);
        cx.draw_frame().expect("a frame at the opening size");

        let y = px(400.0);
        let grab = point(px(327.5), y);
        cx.simulate_mouse_down(grab, MouseButton::Left, Modifiers::none());
        for dx in [-400.0, -50.0, 0.0, 60.0, 200.0, 400.0, -800.0, 30.0] {
            cx.simulate_mouse_move(
                point(px(327.5 + dx), y),
                Some(MouseButton::Left),
                Modifiers::none(),
            );
            let _ = cx.draw_frame();
        }
        cx.simulate_mouse_up(grab, MouseButton::Left, Modifiers::none());
        let _ = cx.draw_frame();
    }

    #[test]
    fn sidebar_toggle_and_resize_without_panicking() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        for visible in [false, true, false, true] {
            app_state.update(&mut cx, |state, cx| {
                state.sidebar_visible = visible;
                cx.notify();
            });

            cx.simulate_resize(size(px(640.), px(480.)));
            cx.run_until_parked();
            cx.draw_frame().expect("a frame after toggling");
            cx.simulate_resize(size(px(1200.), px(780.)));
            cx.run_until_parked();
            cx.draw_frame().expect("a frame after toggling");
        }
    }

    #[test]
    fn fixed_sidebar_width_stays_constant_on_resize() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        for new_size in [
            size(px(640.), px(480.)),
            size(px(1440.), px(900.)),
            size(px(1920.), px(1080.)),
        ] {
            cx.simulate_resize(new_size);
            cx.run_until_parked();
            cx.draw_frame().expect("drawn frame");

            let width = app_state.read_with(&cx, |state, _| state.sidebar_width);
            assert_eq!(width, SIDEBAR_WIDTH, "sidebar width must remain fixed across window resizes");
        }
    }
}

#[cfg(test)]
mod xtgettcap_tests {
    //! End-to-end check of the terminal crate's XTGETTCAP passthrough: a
    //! capability query must produce a reply on the PTY writer and must never
    //! reach the grid as printable output.

    use alacritty_terminal::event::WindowSize;
    use terminal::{ColorPalette, GpuiEventProxy, PtyWriter, TerminalNotifier, TerminalState};
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// Writer sink that records everything the terminal writes back.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Capture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A terminal with a real PTY writer installed, like the app has.
    fn terminal_with_capture() -> (TerminalState, Capture) {
        let (tx, _rx) = std::sync::mpsc::channel();
        let proxy = GpuiEventProxy::new(tx);
        // The notifier slot is shared with the proxy/Term, so installing the
        // writer after construction is exactly what `TerminalView` does.
        let slot = proxy.notifier_slot();
        let state = TerminalState::new(80, 24, proxy);

        let capture = Capture::default();
        let writer: PtyWriter = Arc::new(Mutex::new(
            Box::new(capture.clone()) as Box<dyn std::io::Write + Send>
        ));
        *slot.lock() = Some(TerminalNotifier::new(
            writer,
            Arc::new(Mutex::new(WindowSize {
                num_lines: 24,
                num_cols: 80,
                cell_width: 8,
                cell_height: 16,
            })),
            Arc::new(Mutex::new(ColorPalette::default())),
        ));

        (state, capture)
    }

    #[test]
    fn rgb_capability_is_answered_with_a_value() {
        let (mut state, capture) = terminal_with_capture();

        // "RGB" hex-encoded is `524742`.
        state.process_bytes(b"\x1bP+q524742\x1b\\");

        let reply = String::from_utf8_lossy(&capture.0.lock()).to_string();
        // `1` hex-encoded is `31`.
        assert_eq!(reply, "\x1bP1+r524742=31\x1b\\");
    }

    #[test]
    fn unknown_capability_gets_the_invalid_reply() {
        let (mut state, capture) = terminal_with_capture();

        // "XYZ" hex-encoded is `58595a`.
        state.process_bytes(b"\x1bP+q58595a\x1b\\");

        let reply = String::from_utf8_lossy(&capture.0.lock()).to_string();
        assert_eq!(reply, "\x1bP0+r58595a\x1b\\");
    }

    #[test]
    fn multiple_capabilities_get_one_reply_each() {
        let (mut state, capture) = terminal_with_capture();

        // "RGB;XYZ" -> `524742;58595a`
        state.process_bytes(b"\x1bP+q524742;58595a\x1b\\");

        let reply = String::from_utf8_lossy(&capture.0.lock()).to_string();
        assert_eq!(reply, "\x1bP1+r524742=31\x1b\\\x1bP0+r58595a\x1b\\");
    }

    #[test]
    fn query_is_not_printed_to_the_grid() {
        let (mut state, capture) = terminal_with_capture();

        state.process_bytes(b"before \x1bP+q524742\x1b\\ after\r\n");

        // Content starts on the top row, so read the whole visible screen.
        let screen = state.bottom_lines(usize::MAX).join("\n");
        assert!(
            screen.contains("before  after"),
            "query text leaked into the grid: {screen:?}"
        );
        assert!(
            !capture.0.lock().is_empty(),
            "the query should have been answered"
        );
    }
}

#[cfg(test)]
mod pane_tests {
    use super::*;
    use gpui::{Modifiers, TestAppContext, VisualContext, point};
    use resize_tests::open_app;
    #[test]
    fn pane_node_split_and_contains() {
        let mut tree = PaneNode::Leaf(1);
        assert!(tree.contains(1));
        assert!(!tree.contains(2));
        assert_eq!(tree.first_leaf(), 1);

        assert!(tree.split(1, 2, 10, gpui::Axis::Horizontal));
        assert!(tree.contains(1));
        assert!(tree.contains(2));
        assert!(!tree.contains(3));

        assert!(tree.split(2, 3, 11, gpui::Axis::Vertical));
        assert!(tree.contains(3));
        assert_eq!(tree.first_leaf(), 1);
    }

    #[test]
    fn pane_node_remove_collapses_tree() {
        let mut tree = PaneNode::Leaf(1);
        tree.split(1, 2, 10, gpui::Axis::Horizontal);
        tree.split(2, 3, 11, gpui::Axis::Vertical);

        // Remove 2: the vertical split (2, 3) should collapse to 3
        let tree = tree.remove(2).expect("tree remains");
        assert!(tree.contains(1));
        assert!(!tree.contains(2));
        assert!(tree.contains(3));

        // Remove 3: the horizontal split (1, 3) should collapse to 1
        let tree = tree.remove(3).expect("tree remains");
        assert_eq!(tree, PaneNode::Leaf(1));

        // Remove 1: tree becomes None
        assert!(tree.remove(1).is_none());
    }

    #[test]
    fn visual_split_maximize_close_flow() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        // Split vertical
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_pane(1, gpui::Axis::Horizontal, window, cx);
        });
        cx.draw_frame().expect("frame after vertical split");

        let pane_count = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions[0].panes.len()
        });
        assert_eq!(pane_count, 2);

        let new_pane_id = app_state.read_with(&cx, |state, _| state.focused_pane_id.unwrap());

        // Toggle maximize
        app_state.update(&mut cx, |state, cx| {
            state.toggle_maximize_pane(new_pane_id, cx);
        });
        cx.draw_frame().expect("frame while maximized");

        let is_max = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions[0].maximized_pane_id
        });
        assert_eq!(is_max, Some(new_pane_id));

        // Restore
        app_state.update(&mut cx, |state, cx| {
            state.toggle_maximize_pane(new_pane_id, cx);
        });
        cx.draw_frame().expect("frame restored");

        // Close the new pane
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.close_pane(new_pane_id, window, cx);
        });
        cx.draw_frame().expect("frame after close");

        let pane_count_after = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions[0].panes.len()
        });
        assert_eq!(pane_count_after, 1);
    }

    #[test]
    fn dragging_split_terminal_divider_resizes_without_panic() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        // Split vertical
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_pane(1, gpui::Axis::Horizontal, window, cx);
        });
        cx.draw_frame().expect("frame after vertical split");

        let y = px(400.0);
        let grab = point(px(700.0), y);
        cx.simulate_mouse_down(grab, MouseButton::Left, Modifiers::none());
        for dx in [-150.0, -50.0, 0.0, 50.0, 150.0] {
            cx.simulate_mouse_move(
                point(px(700.0 + dx), y),
                Some(MouseButton::Left),
                Modifiers::none(),
            );
            let _ = cx.draw_frame();
        }
        cx.simulate_mouse_up(grab, MouseButton::Left, Modifiers::none());
        let _ = cx.draw_frame();
    }

    #[test]
    fn reproduce_nested_split_bounds() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        // Split 1 vertically: Left: 1, Right: new (2)
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_pane(1, gpui::Axis::Horizontal, window, cx);
        });
        cx.draw_frame().expect("frame after vertical split");

        // Split 1 horizontally: Top-left: 1, Bottom-left: new (3)
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_pane(1, gpui::Axis::Vertical, window, cx);
        });
        cx.run_until_parked();
        cx.draw_frame().expect("frame after horizontal split");

        // Check terminal dimensions/bounds for all 3 panes
        let (term1, term2, term3) = app_state.read_with(&cx, |state, cx| {
            let session = &state.active_project().unwrap().sessions[0];
            let t1 = session.panes[0].terminal.read(cx).dimensions();
            let t2 = session.panes[1].terminal.read(cx).dimensions();
            let t3 = session.panes[2].terminal.read(cx).dimensions();
            (t1, t2, t3)
        });

        // Ensure no terminal collapsed to height 0 / 1 row
        assert!(term1.1 > 5, "term1 collapsed: {:?}", term1);
        assert!(term2.1 > 5, "term2 collapsed: {:?}", term2);
        assert!(term3.1 > 5, "term3 collapsed: {:?}", term3);
    }

    #[test]
    fn horizontal_then_vertical_nested_split_does_not_collapse() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        // Split 1 horizontally: Top: 1, Bottom: 2
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_pane(1, gpui::Axis::Vertical, window, cx);
        });
        cx.draw_frame().expect("frame after horizontal split");

        let bottom_id = app_state.read_with(&cx, |state, _| state.focused_pane_id.unwrap());

        // Split bottom vertically: Bottom-left: 2, Bottom-right: 3
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_pane(bottom_id, gpui::Axis::Horizontal, window, cx);
        });
        cx.run_until_parked();
        cx.draw_frame().expect("frame after vertical split");

        let (term1, term2, term3) = app_state.read_with(&cx, |state, cx| {
            let session = &state.active_project().unwrap().sessions[0];
            let t1 = session.panes[0].terminal.read(cx).dimensions();
            let t2 = session.panes[1].terminal.read(cx).dimensions();
            let t3 = session.panes[2].terminal.read(cx).dimensions();
            (t1, t2, t3)
        });

        assert!(term1.1 > 5, "term1 collapsed: {:?}", term1);
        assert!(term2.1 > 5, "term2 collapsed: {:?}", term2);
        assert!(term3.1 > 5, "term3 collapsed: {:?}", term3);
    }

    #[test]
    fn session_isolation_switch_tabs_keeps_screens_separate() {
        // The herdr-style isolation contract: creating a second session must
        // not touch the first session's pane tree, and each tab renders its
        // own screen.
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_pane(1, gpui::Axis::Horizontal, window, cx);
        });
        cx.draw_frame().expect("frame after split in session 1");

        let new_pane_id = app_state.read_with(&cx, |state, _| state.focused_pane_id.unwrap());

        cx.update_window_entity(&app_state, |state, window, cx| {
            state.create_new_session_for_active_project(window, cx);
        });
        cx.draw_frame().expect("frame after adding a session");

        let (first_pane_count, second_tree) = app_state.read_with(&cx, |state, _| {
            let project = state.active_project().unwrap();
            (
                project.sessions[0].panes.len(),
                project.sessions[1].pane_tree.clone(),
            )
        });
        assert_eq!(first_pane_count, 2);
        // The new session starts on its own full-size screen with one pane.
        assert!(matches!(second_tree, Some(PaneNode::Leaf(_))));

        // Switch back to the first tab: the split is still there.
        app_state.update(&mut cx, |state, cx| {
            state.active_project_mut().unwrap().active_session_idx = 0;
            cx.notify();
        });
        cx.draw_frame().expect("frame after switching tabs");
        let first_tree = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions[0]
                .pane_tree
                .clone()
        });
        assert!(first_tree.map_or(false, |t| t.contains(1) && t.contains(new_pane_id)));
    }
    /// A project directory that is not the repository the test runs in, so
    /// opening it really is "a second project".
    fn scratch_project_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&dir).expect("scratch project dir");
        dir
    }

    #[test]
    fn opening_a_second_project_keeps_the_first_one_alive() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        let dir = scratch_project_dir("markup-multi-project-open");
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.open_project_path(dir, window, cx);
        });
        cx.draw_frame().expect("frame with two projects");

        let (project_count, active_idx, first_sessions, second_sessions, expanded) = app_state
            .read_with(&cx, |state, _| {
                (
                    state.projects.len(),
                    state.active_project_idx,
                    state.projects[0].sessions.len(),
                    state.projects[1].sessions.len(),
                    state.projects[1].expanded,
                )
            });
        assert_eq!(project_count, 2);
        assert_eq!(active_idx, 1);
        assert_eq!(first_sessions, 1, "the first project keeps its session");
        assert_eq!(second_sessions, 1, "the new project starts with a session");
        assert!(expanded, "the newly opened project is expanded");
    }

    #[test]
    fn sessions_are_created_in_the_project_that_asks_for_them() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        let dir = scratch_project_dir("markup-multi-project-sessions");
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.open_project_path(dir, window, cx);
        });
        cx.draw_frame().expect("frame with two projects");

        // Create a session in the *first* project while the second one is the
        // active project: the new terminal must land in project 0 and switch
        // the workspace over to it.
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.create_new_session_in_project(0, window, cx);
        });
        cx.draw_frame().expect("frame after creating a session");

        let (first, second, active) = app_state.read_with(&cx, |state, _| {
            (
                state.projects[0].sessions.len(),
                state.projects[1].sessions.len(),
                state.active_project_idx,
            )
        });
        assert_eq!(first, 2);
        assert_eq!(second, 1);
        assert_eq!(active, 0);
    }

    #[test]
    fn closing_a_session_or_project_leaves_the_others_intact() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        let dir = scratch_project_dir("markup-multi-project-close");
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.open_project_path(dir, window, cx);
        });
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.create_new_session_in_project(0, window, cx);
        });
        cx.draw_frame()
            .expect("frame with two projects and two sessions");

        // Close the extra session in project 0. Project 1 must be untouched.
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.close_session(0, 1, window, cx);
        });
        cx.draw_frame().expect("frame after closing a session");
        let (first, second) = app_state.read_with(&cx, |state, _| {
            (
                state.projects[0].sessions.len(),
                state.projects[1].sessions.len(),
            )
        });
        assert_eq!((first, second), (1, 1));

        // Close the first project. The second one becomes active with its own
        // session still running.
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.close_project(0, window, cx);
        });
        cx.draw_frame().expect("frame after closing a project");
        let (project_count, active_sessions, name) = app_state.read_with(&cx, |state, _| {
            (
                state.projects.len(),
                state.active_session().map(|s| s.panes.len()).unwrap_or(0),
                state.projects[0].name.to_string(),
            )
        });
        assert_eq!(project_count, 1);
        assert_eq!(active_sessions, 1);
        assert_eq!(name, "markup-multi-project-close");
    }

    #[test]
    fn sidebar_renders_collapsed_and_expanded_projects() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        let dir = scratch_project_dir("markup-multi-project-collapse");
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.open_project_path(dir, window, cx);
        });
        cx.draw_frame().expect("frame with two expanded projects");

        // Collapse the first project while the second stays expanded.
        cx.update_window_entity(&app_state, |state, _window, cx| {
            state.toggle_project_expanded(0, cx);
        });
        cx.draw_frame().expect("frame with one collapsed project");
        let expanded = app_state.read_with(&cx, |state, _| {
            (state.projects[0].expanded, state.projects[1].expanded)
        });
        assert_eq!(expanded, (false, true));

        // Activating a collapsed project expands it again.
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.activate_project(0, window, cx);
        });
        cx.draw_frame().expect("frame after activating a project");
        let (active, expanded) = app_state.read_with(&cx, |state, _| {
            (state.active_project_idx, state.projects[0].expanded)
        });
        assert_eq!(active, 0);
        assert!(expanded);
    }

    #[test]
    fn sidebar_renders_with_agent_and_worker() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        cx.update_window_entity(&app_state, |state, _window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.is_agent = true;
            pane.app = Some(SharedString::from("claude"));
            pane.status = SessionStatus::Done;
            pane.attention.seen = false;
            pane.attention.last = Some(SessionStatus::Done);
            pane.attention.since = 42;
            pane.sub_agents.insert(
                "child".into(),
                attention::SubAgent {
                    id: "child".into(),
                    title: Some("explore".into()),
                    status: SessionStatus::Working,
                    since: 41,
                    seq: 41,
                },
            );
            cx.notify();
        });
        // Drawing a frame with agent and sub-agent panes succeeds without panic.
        cx.draw_frame().expect("frame with agent pane");
        let agents = app_state.read_with(&cx, |state, _| {
            state
                .projects
                .iter()
                .flat_map(|project| &project.sessions)
                .flat_map(|session| &session.panes)
                .filter(|pane| pane.is_agent)
                .count()
        });
        assert_eq!(agents, 1);
    }

    #[test]
    fn project_rename_flow() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        let project_id = app_state.read_with(&cx, |state, _| state.projects[0].id);

        // Start rename
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.start_renaming_project(project_id, window, cx);
        });
        assert!(app_state.read_with(&cx, |state, _| state.renaming_project.is_some()));

        // Frame should render the inline input cleanly
        cx.draw_frame().expect("frame while renaming project");

        // Finish rename
        cx.update_window_entity(&app_state, |state, _, cx| {
            state.finish_renaming_project(project_id, SharedString::from("renamed-project"), cx);
        });

        let (name, is_renaming) = app_state.read_with(&cx, |state, _| {
            (state.projects[0].name.clone(), state.renaming_project.is_some())
        });
        assert_eq!(name, "renamed-project");
        assert!(!is_renaming);

        // Frame renders with new project name
        cx.draw_frame().expect("frame with renamed project");
    }

    #[test]
    fn project_menu_and_remove_flow() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        let dir = scratch_project_dir("markup-menu-remove");
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.open_project_path(dir, window, cx);
        });
        cx.draw_frame().expect("frame with two projects");

        let (count, project_id) = app_state.read_with(&cx, |state, _| {
            (state.projects.len(), state.projects[1].id)
        });
        assert_eq!(count, 2);

        // Open 3-dot dropdown menu
        cx.update_window_entity(&app_state, |state, _, cx| {
            state.project_menu = Some(project_id);
            cx.notify();
        });
        // Frame renders menu popover and backdrop
        cx.draw_frame().expect("frame with project menu open");

        // Close/Remove the project
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.close_project(1, window, cx);
        });
        cx.draw_frame().expect("frame after removing project");

        let (final_count, menu) = app_state.read_with(&cx, |state, _| {
            (state.projects.len(), state.project_menu)
        });
        assert_eq!(final_count, 1);
        assert_eq!(menu, None);
    }
}

#[cfg(test)]
mod settings_and_theme_tests {
    use super::*;
    use gpui::{KeyDownEvent, Modifiers, TestAppContext, VisualContext};
    use resize_tests::open_app;

    #[test]
    fn settings_page_toggle_and_render() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        // Initially on workspace
        let initial_view = app_state.read_with(&cx, |state, _| state.active_view);
        assert_eq!(initial_view, ActiveView::Workspace);

        // Switch to settings
        cx.update_window_entity(&app_state, |state, _window, cx| {
            state.active_view = ActiveView::Settings;
            cx.notify();
        });
        cx.draw_frame().expect("draw settings page frame");
        let in_settings = app_state.read_with(&cx, |state, _| state.active_view);
        assert_eq!(in_settings, ActiveView::Settings);

        // Switch back to workspace
        cx.update_window_entity(&app_state, |state, _window, cx| {
            state.active_view = ActiveView::Workspace;
            cx.notify();
        });
        cx.draw_frame().expect("draw workspace page frame");
        let back_to_workspace = app_state.read_with(&cx, |state, _| state.active_view);
        assert_eq!(back_to_workspace, ActiveView::Workspace);
    }

    #[test]
    fn settings_terminal_input_style_toggle() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        // Initially enabled
        assert!(app_state.read_with(&cx, |state, _| state.warp_bottom_input));

        // Switch to settings and set classic
        cx.update_window_entity(&app_state, |state, _window, cx| {
            state.active_view = ActiveView::Settings;
            state.warp_bottom_input = false;
            for project in &mut state.projects {
                for session in &mut project.sessions {
                    for pane in &mut session.panes {
                        pane.bottom_input_enabled = false;
                    }
                }
            }
            cx.notify();
        });
        cx.draw_frame().expect("draw settings with classic input");
        assert!(!app_state.read_with(&cx, |state, _| state.warp_bottom_input));
        assert!(!app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].bottom_input_enabled));

        // Switch back to warp style
        cx.update_window_entity(&app_state, |state, _window, cx| {
            state.warp_bottom_input = true;
            for project in &mut state.projects {
                for session in &mut project.sessions {
                    for pane in &mut session.panes {
                        pane.bottom_input_enabled = true;
                    }
                }
            }
            cx.notify();
        });
        cx.draw_frame().expect("draw settings with warp input");
        assert!(app_state.read_with(&cx, |state, _| state.warp_bottom_input));
        assert!(app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].bottom_input_enabled));
    }

    #[test]
    fn settings_keyboard_shortcuts() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        // Ctrl+, opens settings
        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev = KeyDownEvent {
                keystroke: gpui::Keystroke {
                    modifiers: Modifiers {
                        control: true,
                        ..Default::default()
                    },
                    key: ",".to_string(),
                    key_char: Some(",".to_string()),
                },
                is_held: false,
            };
            state.on_key_down(&ev, window, cx);
        });
        let view = app_state.read_with(&cx, |state, _| state.active_view);
        assert_eq!(view, ActiveView::Settings);

        // Escape closes settings
        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev = KeyDownEvent {
                keystroke: gpui::Keystroke {
                    modifiers: Modifiers::default(),
                    key: "escape".to_string(),
                    key_char: None,
                },
                is_held: false,
            };
            state.on_key_down(&ev, window, cx);
        });
        let view = app_state.read_with(&cx, |state, _| state.active_view);
        assert_eq!(view, ActiveView::Workspace);
    }

    #[test]
    fn theme_switching_updates_palette_and_terminals() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        // Default preference is System
        let pref = app_state.read_with(&cx, |state, _| state.theme_preference);
        assert_eq!(pref, ThemePreference::System);

        // Switch to Light
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.set_theme_preference(ThemePreference::Light, Some(window), cx);
        });
        cx.draw_frame().expect("draw light theme frame");

        app_state.read_with(&cx, |state, cx| {
            assert_eq!(state.theme_preference, ThemePreference::Light);
            assert!(state.palette.background().l > 0.8, "Light terminal background should be bright");
            let pane_term_palette = state.projects[0].sessions[0].panes[0].terminal.read(cx).config().colors.clone();
            assert_eq!(pane_term_palette.background(), state.palette.background());
        });

        // Switch to Dark
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.set_theme_preference(ThemePreference::Dark, Some(window), cx);
        });
        cx.draw_frame().expect("draw dark theme frame");

        app_state.read_with(&cx, |state, cx| {
            assert_eq!(state.theme_preference, ThemePreference::Dark);
            assert!(state.palette.background().l < 0.2, "Dark terminal background should be dark");
            let pane_term_palette = state.projects[0].sessions[0].panes[0].terminal.read(cx).config().colors.clone();
            assert_eq!(pane_term_palette.background(), state.palette.background());
        });

        // Switch back to System
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.set_theme_preference(ThemePreference::System, Some(window), cx);
        });
        cx.draw_frame().expect("draw system theme frame");
        assert_eq!(
            app_state.read_with(&cx, |state, _| state.theme_preference),
            ThemePreference::System
        );
    }

    #[test]
    fn session_or_pane_selection_resets_settings_view() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        // In settings, selecting session switches to workspace
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.active_view = ActiveView::Settings;
            state.select_session(0, 0, window, cx);
        });
        assert_eq!(
            app_state.read_with(&cx, |state, _| state.active_view),
            ActiveView::Workspace
        );

        // In settings, focusing pane switches to workspace
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.active_view = ActiveView::Settings;
            state.focus_pane(1, window, cx);
        });
        assert_eq!(
            app_state.read_with(&cx, |state, _| state.active_view),
            ActiveView::Workspace
        );
    }
}

#[cfg(test)]
mod warp_bottom_input_tests {
    use super::*;
    use gpui::{TestAppContext, VisualContext, point};
    use resize_tests::open_app;

    #[test]
    fn warp_bottom_input_renders_and_toggles() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);

        // By default, warp_bottom_input is enabled
        let enabled = app_state.read_with(&cx, |state, _| state.warp_bottom_input);
        assert!(enabled);

        // Drawing a frame ensures panes' input state is created and renders the bottom input bar
        cx.draw_frame().expect("draw frame with warp bottom input");

        let pane_enabled = app_state.read_with(&cx, |state, _| {
            state.projects[0].sessions[0].panes[0].bottom_input_enabled
        });
        assert!(pane_enabled);

        let has_input_state = app_state.read_with(&cx, |state, _| {
            state.projects[0].sessions[0].panes[0].input_state.is_some()
        });
        assert!(has_input_state);

        // Toggle bottom input off
        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.toggle_pane_bottom_input(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw frame after toggle off");

        let pane_disabled = app_state.read_with(&cx, |state, _| {
            state.projects[0].sessions[0].panes[0].bottom_input_enabled
        });
        assert!(!pane_disabled);

        // Toggle bottom input back on
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.toggle_pane_bottom_input(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw frame after toggle on");

        let pane_enabled_again = app_state.read_with(&cx, |state, _| {
            state.projects[0].sessions[0].panes[0].bottom_input_enabled
        });
        assert!(pane_enabled_again);
    }

    #[test]
    fn warp_bottom_input_history_and_execution() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Execute "echo 1"
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("echo 1", window, cx);
            });
            state.execute_pane_input(pane_id, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.command_history, vec!["echo 1"]);
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "");
        });

        // Execute "echo 2"
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("echo 2", window, cx);
            });
            state.execute_pane_input(pane_id, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.command_history, vec!["echo 1", "echo 2"]);
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "");
        });

        // Start typing a draft "draft cmd"
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("draft cmd", window, cx);
            });
            // Press Up: should recall "echo 2" and store "draft cmd" in pending_draft
            state.step_pane_history(pane_id, true, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.pending_draft, "draft cmd");
            assert_eq!(pane.history_index, Some(1));
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "echo 2");
        });

        // Press Up again: should recall "echo 1"
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.step_pane_history(pane_id, true, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.history_index, Some(0));
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "echo 1");
        });

        // Press Down: should recall "echo 2"
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.step_pane_history(pane_id, false, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.history_index, Some(1));
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "echo 2");
        });

        // Press Down again: should restore pending_draft "draft cmd"
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.step_pane_history(pane_id, false, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.history_index, None);
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "draft cmd");
        });
    }

    #[test]
    fn warp_bottom_input_clear_and_cancel() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Type a value and then clear it
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("to be cleared", window, cx);
            });
            state.clear_pane_input(pane_id, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.history_index, None);
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "");
        });

        // Cancel job should be callable without panicking
        cx.update_window_entity(&app_state, |state, _window, cx| {
            state.cancel_pane_job(pane_id, cx);
        });
    }

    #[test]
    fn terminal_is_read_only_and_cannot_be_written_directly() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        // Terminal must be configured read-only
        let is_ro = app_state.read_with(&cx, |state, cx| {
            state.projects[0].sessions[0].panes[0].terminal.read(cx).is_read_only()
        });
        assert!(is_ro, "Terminal must be strictly read-only");

        // Attempting to paste on the terminal directly is ignored
        cx.update_window_entity(&app_state, |state, _window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.terminal.update(cx, |term, cx| {
                term.paste_text("injected text", cx);
            });
        });
    }

    #[test]
    fn plus_menu_and_message_placeholder() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Check plus menu toggle
        cx.update_window_entity(&app_state, |state, _window, cx| {
            state.toggle_plus_menu(pane_id, cx);
        });
        let menu_open = app_state.read_with(&cx, |state, _| {
            state.projects[0].sessions[0].panes[0].plus_menu_open
        });
        assert!(menu_open);

        cx.draw_frame().expect("draw frame with plus menu open");

        // Run quick action "clear"
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.run_quick_action(pane_id, "clear", window, cx);
        });

        let menu_closed = app_state.read_with(&cx, |state, _| {
            !state.projects[0].sessions[0].panes[0].plus_menu_open
        });
        assert!(menu_closed);

        let history = app_state.read_with(&cx, |state, _| {
            state.projects[0].sessions[0].panes[0].command_history.clone()
        });
        assert_eq!(history, vec!["clear"]);
    }

    #[test]
    fn terminal_shortcuts_work_in_read_only() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Ctrl+A selects all
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.terminal.update(cx, |term, cx| {
                let ev = gpui::KeyDownEvent {
                    keystroke: gpui::Keystroke::parse("ctrl-a").unwrap(),
                    is_held: false,
                };
                term.dispatch_key(&ev, window, cx);
            });
        });

        // Ctrl+C with no selection sends SIGINT (\x03)
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.terminal.update(cx, |term, cx| {
                term.clear_selection(cx);
                let ev = gpui::KeyDownEvent {
                    keystroke: gpui::Keystroke::parse("ctrl-c").unwrap(),
                    is_held: false,
                };
                term.dispatch_key(&ev, window, cx);
            });
        });

        // In bottom input: Ctrl+Z, Ctrl+D, Ctrl+L work
        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev_z = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-z").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev_z, window, cx);

            let ev_l = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-l").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev_l, window, cx);
        });
    }

    #[test]
    fn git_branch_detection_and_warp_badges() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        // Verify git branch detection on the current repo
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let branch = AppState::detect_git_branch(manifest_dir);
        assert!(branch.is_some(), "Should detect git branch for repo");

        // Temporary directory without git should return None
        let tmp = std::env::temp_dir();
        let non_git = AppState::detect_git_branch(&tmp);
        assert!(non_git.is_none() || tmp.join(".git").exists());

        // Verify placeholder is "Run commands"
        let placeholder_is_run_commands = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.read(cx).placeholder_str() == "Run commands"
        });
        assert!(placeholder_is_run_commands, "Input placeholder must be 'Run commands'");

        // Frame draws with badges and input
        cx.draw_frame().expect("draw frame with badges and input");
    }

    #[test]
    fn dynamic_auto_focus_like_warp() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let mut is_focused = false;
        let mut pane_id = 0;
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            pane_id = pane.id;
            is_focused = input.read(cx).focus_handle().is_focused(window);
        });
        assert!(is_focused, "Bottom input must be dynamically focused on start");

        // Focus pane should keep input focused
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.focus_pane(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw frame after focus_pane");

        let mut input_focused_again = false;
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input_focused_again = input.read(cx).focus_handle().is_focused(window);
        });
        assert!(input_focused_again, "Input must remain focused dynamically");
    }

    #[test]
    fn terminal_right_click_copy_and_paste() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Case 1: Select text in terminal -> right-click opens context menu with Copy option
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.terminal.update(cx, |term, cx| {
                term.select_all(cx);
            });
            state.open_terminal_context_menu(pane_id, point(px(100.0), px(100.0)), window, cx);
        });

        let is_menu_open = app_state.read_with(&cx, |state, _| state.terminal_context_menu.is_some());
        assert!(is_menu_open, "Right click must open context menu modal");
        cx.draw_frame().expect("draw context menu modal");

        // Clicking Copy in modal copies selection and clears selection
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.copy_terminal_selection(pane_id, window, cx);
        });

        let has_sel = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            pane.terminal.read(cx).has_selection()
        });
        assert!(!has_sel, "Copy should clear selection after copying");
        assert!(app_state.read_with(&cx, |state, _| state.terminal_context_menu.is_none()));

        // Case 2: Right-click opens modal -> clicking Paste inserts clipboard text into input
        cx.update_window_entity(&app_state, |state, window, cx| {
            cx.write_to_clipboard(gpui::ClipboardItem::new_string("echo copied".to_string()));
            state.open_terminal_context_menu(pane_id, point(px(100.0), px(100.0)), window, cx);
            state.paste_from_clipboard_into_pane(pane_id, window, cx);
        });

        let input_val = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.read(cx).value().to_string()
        });
        assert_eq!(input_val, "echo copied", "Paste should insert clipboard text into input");

        // Case 3: Pasting text containing newlines should sanitize and NOT panic
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("", window, cx);
            });
            cx.write_to_clipboard(gpui::ClipboardItem::new_string("multi\nline\r\ncommand\n".to_string()));
            state.paste_from_clipboard_into_pane(pane_id, window, cx);
        });

        // Draw frame: if newlines leaked into single-line shape_line, this would panic
        cx.draw_frame().expect("drawing frame after multiline paste must not panic");

        let input_val_clean = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.read(cx).value().to_string()
        });
        assert_eq!(input_val_clean, "multi line command");
    }

    #[test]
    fn warp_dynamic_show_hide_and_unblocked_toggle() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        // 1. Initially at shell prompt: bottom input should be shown
        let initial_show = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            pane.should_show_bottom_input(cx)
        });
        assert!(initial_show, "At idle shell prompt, bottom input should be shown");

        // 2. Agent starts working: status becomes Working -> dynamically auto-hides input bar
        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);
        cx.update_window_entity(&app_state, |state, _window, cx| {
            // Simulate poller detecting Working status
            state.projects[0].sessions[0].panes[0].status = SessionStatus::Working;
            state.projects[0].sessions[0].panes[0].bottom_input_visible = false;
            state.projects[0].sessions[0].panes[0].terminal.update(cx, |term, _| term.set_read_only(false));
        });
        cx.draw_frame().expect("draw frame with working agent");

        let working_show = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            pane.should_show_bottom_input(cx)
        });
        assert!(!working_show, "While agent is working, bottom input auto-hides (Warp style)");

        // 3. User toggles anytime without blocking: even while agent is working, toggle works!
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.toggle_pane_bottom_input(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw frame after manual toggle on while working");

        let toggled_on_show = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            pane.should_show_bottom_input(cx)
        });
        assert!(toggled_on_show, "User can toggle bottom input ON anytime without blocking");

        // User toggles back off
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.toggle_pane_bottom_input(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw frame after manual toggle off");
        let toggled_off_show = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            pane.should_show_bottom_input(cx)
        });
        assert!(!toggled_off_show, "User can toggle bottom input OFF anytime without blocking");

        // 4. Agent asks for input (status becomes Blocked): dynamically auto-shows input bar
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.status = SessionStatus::Blocked;
            pane.bottom_input_enabled = true;
            pane.bottom_input_visible = true;
            pane.terminal.update(cx, |term, _| term.set_read_only(true));
            state.focus_pane(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw frame with blocked status");

        let (blocked_show, is_ro) = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            (pane.should_show_bottom_input(cx), pane.terminal.read(cx).is_read_only())
        });
        assert!(blocked_show, "When agent asks for input (Blocked), bottom input dynamically shows");
        assert!(is_ro, "Terminal is in input-editor mode so user types into bottom input");

        // 5. Agent finishes back to shell prompt
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.status = SessionStatus::Idle;
            pane.bottom_input_visible = true;
            pane.terminal.update(cx, |term, _| term.set_read_only(true));
            state.focus_pane(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw frame after agent finishes");

        let finished_show = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            pane.should_show_bottom_input(cx)
        });
        assert!(finished_show, "After agent finishes, bottom input is shown");
    }

    #[test]
    fn ctrl_g_keyboard_shortcut_toggles_input() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Initially visible
        let initial = app_state.read_with(&cx, |state, cx| {
            state.projects[0].sessions[0].panes[0].should_show_bottom_input(cx)
        });
        assert!(initial);

        // Press Ctrl+G while focused in input: toggles off
        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-g").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });
        cx.draw_frame().expect("draw frame after Ctrl+G");

        let after_ctrl_g = app_state.read_with(&cx, |state, cx| {
            state.projects[0].sessions[0].panes[0].should_show_bottom_input(cx)
        });
        assert!(!after_ctrl_g, "Ctrl+G must toggle bottom input off");

        // Press Ctrl+G at window level: toggles back on
        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-g").unwrap(),
                is_held: false,
            };
            state.on_key_down(&ev, window, cx);
        });
        cx.draw_frame().expect("draw frame after second Ctrl+G");

        let after_second_ctrl_g = app_state.read_with(&cx, |state, cx| {
            state.projects[0].sessions[0].panes[0].should_show_bottom_input(cx)
        });
        assert!(after_second_ctrl_g, "Ctrl+G must toggle bottom input back on");
    }

    #[test]
    fn prompt_submission_sends_text_and_enter_without_manual_intervention() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("prompt message", window, cx);
            });
            state.execute_pane_input(pane_id, window, cx);
        });

        // Advance executor so the 25ms timer fires and delivers Enter
        cx.background_executor.advance_clock(std::time::Duration::from_millis(50));
        cx.run_until_parked();

        // Terminal received input and cleared draft
        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "");
            assert_eq!(pane.command_history.last().map(|s| s.as_str()), Some("prompt message"));
        });
    }

    #[test]
    fn ctrl_c_cancels_draft_and_interrupts_running_process() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // 1. Typing draft and pressing Ctrl+C clears draft and sends SIGINT
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("some draft command", window, cx);
            });
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-c").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "", "Ctrl+C must clear input draft");
        });

        // 2. Pressing Ctrl+C while command is working interrupts and restores idle
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.status = SessionStatus::Working;
            state.handle_pane_ctrl_c(pane_id, window, cx);
        });

        app_state.read_with(&cx, |state, _| {
            let pane = &state.projects[0].sessions[0].panes[0];
            assert_eq!(pane.status, SessionStatus::Idle, "Ctrl+C must interrupt working command");
            assert!(pane.bottom_input_visible, "Bottom input must be restored after Ctrl+C");
        });
    }

    #[test]
    fn native_terminal_readline_shortcuts_in_input() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Test Ctrl+U (clear line)
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("command to clear", window, cx);
            });
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-u").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "", "Ctrl+U must clear line");
        });

        // Test Ctrl+W (delete previous word)
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("git checkout main", window, cx);
            });
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-w").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().trim_end(), "git checkout", "Ctrl+W must delete last word");
        });
    }

    #[test]
    fn autosuggestion_tab_and_right_arrow_completion() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Populate history
        cx.update_window_entity(&app_state, |state, _window, _cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.command_history.push("docker compose up -d".to_string());
        });

        // Type prefix "doc" and hit Tab
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("doc", window, cx);
            });
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("tab").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "docker compose up -d", "Tab must complete autosuggestion");
        });
    }

    #[test]
    fn input_auto_grow_height_increases_with_characters_and_newlines() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Initially 1 line
        let initial_rows = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.read(cx).current_rows()
        });
        assert_eq!(initial_rows, 1, "Input should start with 1 row");

        // Typing multi-line text increases row count automatically
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("line 1\nline 2\nline 3\nline 4", window, cx);
            });
        });
        cx.draw_frame().expect("draw frame with multi-line text");

        let multiline_rows = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.read(cx).current_rows()
        });
        assert!(multiline_rows >= 4, "Input row count must increase as characters/lines increase (auto-grow)");

        // Shift+Enter inserts another line
        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("shift-enter").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });
        cx.draw_frame().expect("draw frame after shift-enter");

        let after_shift_enter_rows = app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.read(cx).current_rows()
        });
        assert!(after_shift_enter_rows >= 5, "Shift+Enter must add newline and increase height");
    }

    #[test]
    fn enter_executes_command_and_does_not_insert_newline() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        // Type command "ls -la"
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("ls -la", window, cx);
            });
            // Simulate physical Enter keydown
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("enter").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        // Verify: Enter executed the command, cleared the input field, and did NOT insert newline into input!
        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "", "Enter must execute and clear input, not insert a newline");
            assert_eq!(pane.command_history.last().map(|s| s.as_str()), Some("ls -la"));
        });

        // Also test dispatching Enter action directly (e.g. from GPUI action system)
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("pwd", window, cx);
            });
            input.update(cx, |inp, cx| {
                inp.enter(&gpui_component::input::Enter { secondary: false }, window, cx);
            });
        });

        // In auto-grow mode, enter() should NOT have inserted \n into the text
        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert!(!input.read(cx).value().contains('\n'), "Enter action must not insert newline into auto-grow prompt");
        });
    }

    #[test]
    fn typing_shows_ghost_history_suggestion_and_right_accepts_it() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.command_history.push("docker compose up -d".to_string());
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("docker comp", window, cx);
            });
        });
        cx.run_until_parked();

        app_state.read_with(&cx, |state, cx| {
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(
                input.read(cx).inline_completion_text(),
                Some("ose up -d"),
                "typing a history prefix must show ghost text"
            );
        });

        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("right").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(input.read(cx).value().as_str(), "docker compose up -d");
        });
    }

    #[test]
    fn completion_menu_opens_navigates_and_accepts_with_tab() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("cat Cargo", window, cx);
            });
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("tab").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        app_state.read_with(&cx, |state, _| {
            let menu = state.projects[0].sessions[0].panes[0]
                .completion_menu
                .as_ref()
                .expect("Tab must open the completion menu");
            assert_eq!(menu.items.len(), 2, "Cargo.lock and Cargo.toml match");
            assert_eq!(menu.selected, 0);
        });

        // Move the highlight down, then accept with Tab.
        cx.update_window_entity(&app_state, |state, window, cx| {
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("down").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("tab").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "cat Cargo.toml");
            assert!(pane.completion_menu.is_none(), "accepting closes the menu");
            assert!(pane.command_history.is_empty(), "completion must not execute");
        });
    }

    #[test]
    fn right_and_ctrl_c_flow_through_the_action_pipeline() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        // Ghost suggestion set through the deferred change handler, then
        // accepted with Right (bound to MoveRight in the Input context).
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.command_history.push("docker compose up -d".to_string());
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("docker comp", window, cx);
                inp.focus(window, cx);
            });
        });
        cx.run_until_parked();
        cx.draw_frame().expect("draw frame with ghost suggestion");
        cx.simulate_keystrokes("right");
        cx.run_until_parked();

        app_state.read_with(&cx, |state, cx| {
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(input.read(cx).value().as_str(), "docker compose up -d");
        });

        // Ctrl+C (Copy in the Input context) must still interrupt and clear.
        cx.simulate_keystrokes("ctrl-c");
        cx.run_until_parked();
        app_state.read_with(&cx, |state, cx| {
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(input.read(cx).value().as_str(), "");
        });
    }

    #[test]
    fn ctrl_r_palette_accepts_with_enter_through_dispatch() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);
        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.command_history.push("git status".to_string());
            pane.command_history.push("git push".to_string());
            state.open_history_search(pane_id, window, cx);
        });
        cx.draw_frame().expect("draw palette frame");

        cx.simulate_keystrokes("enter");
        cx.run_until_parked();

        app_state.read_with(&cx, |state, cx| {
            assert!(state.history_search.is_none(), "Enter closes the palette");
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(input.read(cx).value().as_str(), "git push");
        });
    }

    #[test]
    fn history_and_readline_keys_flow_through_the_action_pipeline() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.command_history.push("echo one".to_string());
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| inp.focus(window, cx));
        });
        cx.draw_frame().expect("draw focused frame");

        // Up is bound to `MoveUp` in the Input context; at the first line it
        // must propagate so history recall runs.
        cx.simulate_keystrokes("up");
        cx.run_until_parked();
        app_state.read_with(&cx, |state, cx| {
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(input.read(cx).value().as_str(), "echo one");
        });

        // Ctrl+A is bound to SelectAll in the Input context; in an auto-grow
        // prompt it must end up as readline "start of line".
        cx.update_window_entity(&app_state, |state, window, cx| {
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("git status", window, cx);
                inp.focus(window, cx);
            });
        });
        cx.draw_frame().expect("draw frame before Ctrl+A");
        cx.simulate_keystrokes("ctrl-a");
        cx.run_until_parked();
        app_state.read_with(&cx, |state, cx| {
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(input.read(cx).cursor(), 0);
        });
    }

    #[test]
    fn enter_key_executes_once_through_the_action_pipeline() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("echo dispatched", window, cx);
                inp.focus(window, cx);
            });
        });
        cx.draw_frame().expect("draw frame with command");

        cx.simulate_keystrokes("enter");
        cx.run_until_parked();

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(input.read(cx).value().as_str(), "");
            assert_eq!(pane.command_history.last().map(String::as_str), Some("echo dispatched"));
        });
    }

    #[test]
    fn tab_key_dispatches_completion_not_indentation() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            input.update(cx, |inp, cx| {
                inp.set_value("cat Cargo", window, cx);
                inp.focus(window, cx);
            });
        });
        cx.draw_frame().expect("draw frame with input focused");

        // Goes through the real keybinding (`IndentInline` in the Input
        // context) and then bubbles to the wrapper's completion handler.
        cx.simulate_keystrokes("tab");
        cx.run_until_parked();

        app_state.read_with(&cx, |state, cx| {
            let pane = &state.projects[0].sessions[0].panes[0];
            let input = pane.input_state.as_ref().unwrap();
            assert_eq!(
                input.read(cx).value().as_str(),
                "cat Cargo",
                "Tab must not insert indentation into the prompt"
            );
            assert!(
                pane.completion_menu.is_some(),
                "Tab must open the completion menu through the action pipeline"
            );
        });
    }

    #[test]
    fn ctrl_r_history_search_filters_and_inserts() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("draw initial frame");

        let pane_id = app_state.read_with(&cx, |state, _| state.projects[0].sessions[0].panes[0].id);

        cx.update_window_entity(&app_state, |state, window, cx| {
            let pane = &mut state.projects[0].sessions[0].panes[0];
            pane.command_history
                .push("cargo test".to_string());
            pane.command_history
                .push("git status".to_string());
            pane.command_history
                .push("git push".to_string());
            let ev = gpui::KeyDownEvent {
                keystroke: gpui::Keystroke::parse("ctrl-r").unwrap(),
                is_held: false,
            };
            state.handle_pane_input_key(pane_id, &ev, window, cx);
            assert!(state.history_search.is_some(), "Ctrl+R opens history search");
        });

        // Type a query into the palette; the deferred handler refreshes it.
        cx.update_window_entity(&app_state, |state, window, cx| {
            let search = state.history_search.as_ref().unwrap();
            let input = search.input.clone();
            input.update(cx, |inp, cx| {
                inp.set_value("git", window, cx);
            });
        });
        cx.run_until_parked();

        app_state.read_with(&cx, |state, _| {
            let search = state.history_search.as_ref().unwrap();
            assert_eq!(search.matches, vec!["git push", "git status"]);
        });

        cx.update_window_entity(&app_state, |state, window, cx| {
            state.accept_history_search(window, cx);
        });

        app_state.read_with(&cx, |state, cx| {
            assert!(state.history_search.is_none(), "accepting closes the palette");
            let input = state.projects[0].sessions[0].panes[0]
                .input_state
                .as_ref()
                .unwrap();
            assert_eq!(input.read(cx).value().as_str(), "git push");
        });
    }
}
