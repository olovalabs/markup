use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use anyhow::Result;
use gpui::{
    px, size, App, AppContext, Application, Bounds, Context, CursorStyle, Edges, Entity,
    InteractiveElement as _, IntoElement, KeyDownEvent, MouseButton, ParentElement, Render,
    SharedString, Styled, Window, WindowBounds, WindowDecorations,
    WindowOptions, div, rgba, svg, MouseDownEvent, MouseMoveEvent, Pixels, Point, WindowControlArea,
};
use gpui::prelude::FluentBuilder as _;
use gpui_component::resizable::{h_resizable, resizable_panel, v_resizable};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::{Root, TitleBar};
use gpui_terminal::{ColorPalette, TerminalConfig, TerminalView};
use parking_lot::Mutex;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde_json::Value;

mod agent_rules;
mod assets;
mod logos;
mod status;

use assets::{load_embedded_fonts, sync_component_fonts, CombinedAssets, MONO_FONT};
use status::{SessionStatus, StatusInput};
pub const APP_ICON_PNG: &[u8] = include_bytes!("../assets/logo/markup.png");


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
    pub terminal: Entity<TerminalView>,
    pub status: SessionStatus,
    pub title: String,
    /// Title as shown in the pane header (truncated to 160 chars), cached so
    /// the render pass does not re-scan and re-allocate it every frame.
    pub display_title: SharedString,
    pub pid: Option<u32>,
    /// Reused scrape buffer: the poller reads the screen into this instead of
    /// allocating a fresh `Vec<String>` every 500 ms per pane.
    pub screen_scratch: Vec<String>,
    /// (screen revision, focused, exited, foreground fingerprint) from the
    /// last poll tick. The revision lets an idle pane skip rebuilding its screen
    /// strings and fingerprints entirely.
    pub poll_cache: Option<(u64, bool, bool, u64)>,
    /// Screen fingerprint at the moment the user last looked at a finished or
    /// failed terminal. While the screen still matches, the sticky Done/Error
    /// badge stays acknowledged (Idle); any new output invalidates it and lets
    /// a fresh Done/Error be raised.
    pub ack_screen: Option<u64>,
    /// A recognized agent CLI was the foreground process on the previous tick.
    /// Its disappearance means the agent finished (Done), like herdr.
    pub agent_active: bool,
    /// When the terminal first showed an unacknowledged Done, for the short
    /// auto-clear timeout in the spec.
    pub done_since: Option<std::time::Instant>,
    pub detector: status::Detector,
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

/// How long an unacknowledged Done badge lingers before it falls back to Idle.
const DONE_LINGER: std::time::Duration = std::time::Duration::from_secs(30);
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

    pub fn split(&mut self, target_id: usize, new_id: usize, split_id: usize, axis: gpui::Axis) -> bool {
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
                left.split(target_id, new_id, split_id, axis) || right.split(target_id, new_id, split_id, axis)
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
            PaneNode::Split { id, axis, left, right } => {
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
    pub sidebar_visible: bool,
    pub sidebar_width: Pixels,
    drag_start_pos: Option<Point<Pixels>>,
}

impl AppState {
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
            .on_mouse_move(
                cx.listener(|this, ev: &MouseMoveEvent, window, _| {
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
                }),
            )
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

    fn spawn_terminal_view(
        dir: Option<&PathBuf>,
        palette: ColorPalette,
        window_title_target: gpui::WeakEntity<Self>,
        pane_id: usize,
        cx: &mut Context<Self>,
    ) -> Result<(Entity<TerminalView>, Option<u32>)> {
        let (shell_cmd, _) = detect_shell();
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

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| anyhow::anyhow!("Spawn shell error: {e}"))?;
        let pid = child.process_id();

        let writer = pair.master.take_writer().map_err(|e| anyhow::anyhow!("{e}"))?;
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
            copy_on_select: false,
            right_click_paste: false,
            alternate_scroll: true,
            detect_path_links: true,
            show_scrollbar: true,
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
        Ok((terminal, pid))
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
    pub fn poll_sessions_status(&mut self, cx: &mut Context<Self>) {
        let mut changed = false;
        let focused_id = self.focused_pane_id;
        // `T3_STATUS_DEBUG=1` dumps every detection pass: the facts that went in
        // and the badge that came out. Without this the poller is invisible.
        let debug = status::debug_enabled();
        for project in &mut self.projects {
            for session in &mut project.sessions {
                let default_primary_id = session.pane_tree.as_ref().map(|t| t.first_leaf());
                let pane_count = session.panes.len();
                let mut primary_idx: Option<usize> = None;
                let mut primary_focused = false;
                let mut primary_fg: Arc<[String]> = status::foreground_process_names(None);

                for (pane_idx, pane) in session.panes.iter_mut().enumerate() {
                    let focused = focused_id == Some(pane.id);
                    // Reuse the same immutable foreground snapshot for status,
                    // cache keys, and the session logo (one /proc probe per pane).
                    let foreground = status::foreground_process_names(pane.pid);

                    // The pane the session badge/title follows: the focused
                    // pane, else the first pane of the screen.
                    let is_default_primary =
                        Some(pane.id) == default_primary_id || pane_count == 1;
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
                    let foreground_fp = screen_fingerprint(foreground.as_ref(), "");
                    let (screen_revision, exited) = {
                        let terminal = pane.terminal.read(cx);
                        (terminal.screen_revision(), terminal.has_exited())
                    };
                    let unchanged = pane.poll_cache.is_some_and(
                        |(revision, was_focused, was_exited, old_foreground_fp)| {
                            revision == screen_revision
                                && was_focused == focused
                                && was_exited == exited
                                && old_foreground_fp == foreground_fp
                        },
                    );
                    if unchanged
                        && pane.status != SessionStatus::Done
                        && pane.status != SessionStatus::Working
                    {
                        continue;
                    }

                    // Scrape into the pane's reused buffer only after the cheap
                    // revision/focus/process checks show that something changed.
                    // Ask for the whole visible screen: fresh shells can print
                    // their prompt on row 0, not just the last dozen rows.
                    pane.terminal
                        .read(cx)
                        .bottom_lines_into(usize::MAX, &mut pane.screen_scratch);

                    let screen_fp = {
                        let terminal = pane.terminal.read(cx);
                        screen_fingerprint(
                            &pane.screen_scratch,
                            terminal.title().unwrap_or(""),
                        )
                    };
                    let poll_key = (screen_revision, focused, exited, foreground_fp);

                    let (next, ack) = {
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
                        let ack = if focused
                            && (terminal.has_exited()
                                || matches!(pane.status, SessionStatus::Done | SessionStatus::Error))
                        {
                            Some(screen_fp)
                        } else {
                            pane.ack_screen
                        };
                        let acknowledged = ack == Some(screen_fp);

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
                        } else {
                            let input = StatusInput {
                                lines: bottom_lines,
                                title: Some(title),
                                foreground: &foreground,
                                detector_state: Some(detector_state),
                            };
                            // Debug builds capture the match for the dump below;
                            // release builds use the plain verdict (same work).
                            let (detected, rule_match) = if debug {
                                status::evaluate_with_match(&input, pane.status, focused)
                            } else {
                                (status::evaluate(&input, pane.status, focused), None)
                            };
                            // Stash the match for the debug dump below so the pack
                            // is not evaluated a second time to explain the badge.
                            debug_match = rule_match;
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
                            let rule = debug_match
                                .as_ref()
                                .map_or_else(
                                || "(no rule)".to_string(),
                                |m| {
                                    format!(
                                        "{}/{} v{} state={} priority={} region={} \
                                         visible(working/blocker/idle)={}/{}/{} skip_update={}",
                                        m.pack,
                                        m.rule,
                                        m.pack_version.as_deref().unwrap_or("?"),
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
                            eprintln!(
                                "[status] pane={} focused={focused} exited={} exit={:?} \
                                 pid={:?} fg={foreground:?} pack={:?} title={title:?} \
                                 prev={} next={} detector={detector_state:?} rule={rule}",
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

                        (next, ack)
                    };

                    // Agent lifecycle. Two ways a task counts as finished, because
                    // an agent CLI keeps running after it answers:
                    //
                    //  * the CLI itself exited (agent gone), or
                    //  * it stopped working and is back at rest (the response is
                    //    complete) — the usual case, and the one exit codes can
                    //    never catch, since the PTY child is the persistent shell.
                    //
                    // Either way the completion is reported as Done: herdr's "the
                    // agent finished and you have not looked at it yet". Reporting
                    // it only while unfocused is deliberate — a badge is for the
                    // tab you are *not* on.
                    let agent_now = foreground.iter().any(|name| status::is_agent_process(name));
                    let agent_exited = pane.agent_active && !agent_now;
                    pane.agent_active = agent_now;

                    let turn_finished = matches!(
                        pane.status,
                        SessionStatus::Working | SessionStatus::Blocked
                    ) && next == SessionStatus::Idle;

                    let just_finished = agent_exited || turn_finished;
                    let next = if just_finished
                        && next == SessionStatus::Idle
                        && ack != Some(screen_fp)
                        && !focused
                    {
                        SessionStatus::Done
                    } else {
                        next
                    };

                    // Spec: Done also drops back to Idle after a short timeout even
                    // if the user never focuses the terminal.
                    let next = if next == SessionStatus::Done {
                        let since = *pane.done_since.get_or_insert_with(std::time::Instant::now);
                        if since.elapsed() >= DONE_LINGER {
                            pane.done_since = None;
                            SessionStatus::Idle
                        } else {
                            next
                        }
                    } else {
                        pane.done_since = None;
                        next
                    };

                    pane.ack_screen = ack;
                    if pane.status != next {
                        pane.status = next;
                        changed = true;
                    }

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
    }

    fn toggle_project_expanded(&mut self, project_idx: usize, cx: &mut Context<Self>) {
        if let Some(project) = self.projects.get_mut(project_idx) {
            project.expanded = !project.expanded;
            cx.notify();
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
        if let Ok((terminal, pid)) = Self::spawn_terminal_view(
            Some(&dir),
            self.palette.clone(),
            self_weak,
            next_id,
            cx,
        ) {
            terminal.read(cx).focus_handle().focus(window);
            let session_count = self
                .projects
                .get(project_idx)
                .map(|p| p.sessions.len())
                .unwrap_or(0);
            let title = SharedString::from(format!("terminal {}", session_count + 1));
            let pane = TerminalPane {
                id: next_id,
                terminal,
                status: SessionStatus::Idle,
                title: title.to_string(),
                display_title: truncate_title(&title),
                pid,
                screen_scratch: Vec::new(),
                poll_cache: None,
                ack_screen: None,
                agent_active: false,
                done_since: None,
                detector: status::Detector::default(),
            };
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
                if let Some(pane) = project.sessions[project.active_session_idx].panes.first() {
                    pane.terminal.read(cx).focus_handle().focus(window);
                }
                self.focused_pane_id = Some(pane_id);
            } else {
                self.focused_pane_id = None;
            }
        }
        // Drop the removed terminals after forgetting their pids, so their
        // PTY masters close and the shells go away with them.
        drop(removed);
        self.rebuild_session_index();
        cx.notify();
    }

    /// Close a whole project. Its sessions and PTYs go away; the remaining
    /// projects (and their sessions) are left exactly as they were.
    fn close_project(
        &mut self,
        project_idx: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if project_idx >= self.projects.len() {
            return;
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
    }

    /// Open a folder as a project. Opening a folder that is already in the
    /// sidebar focuses that project (and revives it if it has no sessions)
    /// rather than creating a duplicate.
    fn open_project_path(
        &mut self,
        folder: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
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

        if let Ok((terminal, pid)) = Self::spawn_terminal_view(
            Some(&folder),
            self.palette.clone(),
            self_weak,
            next_id,
            cx,
        ) {
            terminal.read(cx).focus_handle().focus(window);
            let title = SharedString::from("terminal 1");
            let pane = TerminalPane {
                id: next_id,
                terminal,
                status: SessionStatus::Idle,
                title: title.to_string(),
                display_title: title.clone(),
                pid,
                screen_scratch: Vec::new(),
                poll_cache: None,
                ack_screen: None,
                agent_active: false,
                done_since: None,
                detector: status::Detector::default(),
            };
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
        if let Ok((terminal, pid)) = Self::spawn_terminal_view(
            dir.as_ref(),
            self.palette.clone(),
            self_weak,
            next_id,
            cx,
        ) {
            terminal.read(cx).focus_handle().focus(window);
            let title = SharedString::from(format!("terminal {}", pane_count + 1));
            let pane = TerminalPane {
                id: next_id,
                terminal,
                status: SessionStatus::Idle,
                title: title.to_string(),
                display_title: truncate_title(&title),
                pid,
                screen_scratch: Vec::new(),
                poll_cache: None,
                ack_screen: None,
                agent_active: false,
                done_since: None,
                detector: status::Detector::default(),
            };

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
                    project.active_session_idx = project
                        .active_session_idx
                        .min(project.sessions.len() - 1);
                }
            } else if let Some(session) = project.sessions.get(s_idx) {
                // Focus the surviving first pane of this screen.
                let next_leaf = session
                    .pane_tree
                    .as_ref()
                    .map(|t| t.first_leaf())
                    .and_then(|id| session.panes.iter().find(|p| p.id == id));
                if let Some(pane) = next_leaf {
                    pane.terminal.read(cx).focus_handle().focus(window);
                    self.focused_pane_id = Some(pane.id);
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
        self.focused_pane_id = Some(pane_id);
        if let Some(&(p_idx, s_idx, _)) = self.pane_index.get(&pane_id) {
            self.active_project_idx = p_idx;
            if let Some(p) = self.projects.get_mut(p_idx) {
                p.active_session_idx = s_idx;
                if let Some(session) = p.sessions.get_mut(s_idx) {
                    if let Some(pane) = session.panes.iter_mut().find(|p| p.id == pane_id) {
                        pane.terminal.read(cx).focus_handle().focus(window);
                    }
                }
            }
        }
        cx.notify();
    }

    fn render_terminal_pane(
        &self,
        pane: &TerminalPane,
        is_maximized: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        let pane_id = pane.id;
        let is_focused = self.focused_pane_id == Some(pane_id);

        div()
            .id(("terminal-pane", pane_id))
            .flex()
            .flex_col()
            .size_full()
            .bg(rgba(0x0c0d12ff))
            .rounded(px(8.0))
            .border_1()
            .border_color(if is_focused {
                rgba(0x38394eff)
            } else {
                rgba(0x22232cff)
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
                    .bg(rgba(0x0c0d12ff))
                    .child(
                        div()
                            .text_size(px(11.5))
                            .text_color(rgba(0x656679ff))
                            .child(pane.display_title.clone()),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.0))
                            .child(
                                div()
                                    .id(("split-v-btn", pane_id))
                                    .size(px(22.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(4.0))
                                    .cursor(CursorStyle::PointingHand)
                                    .hover(|s| s.bg(rgba(0x23242cff)))
                                    .child(
                                        svg()
                                            .path("icons/split-vertical.svg")
                                            .size(px(13.0))
                                            .text_color(rgba(0x8788a5ff))
                                            .hover(|s| s.text_color(rgba(0xf0f6fcff))),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.split_pane(pane_id, gpui::Axis::Horizontal, window, cx);
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
                                    .hover(|s| s.bg(rgba(0x23242cff)))
                                    .child(
                                        svg()
                                            .path("icons/split-horizontal.svg")
                                            .size(px(13.0))
                                            .text_color(rgba(0x8788a5ff))
                                            .hover(|s| s.text_color(rgba(0xf0f6fcff))),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.split_pane(pane_id, gpui::Axis::Vertical, window, cx);
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
                                    .hover(|s| s.bg(rgba(0x23242cff)))
                                    .child(
                                        svg()
                                            .path(if is_maximized {
                                                "icons/pane-restore.svg"
                                            } else {
                                                "icons/pane-maximize.svg"
                                            })
                                            .size(px(13.0))
                                            .text_color(rgba(0x8788a5ff))
                                            .hover(|s| s.text_color(rgba(0xf0f6fcff))),
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
                                    .hover(|s| s.bg(rgba(0x23242cff)))
                                    .child(
                                        svg()
                                            .path("icons/pane-close.svg")
                                            .size(px(13.0))
                                            .text_color(rgba(0x8788a5ff))
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
                    .child(pane.terminal.clone()),
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
            PaneNode::Split { id, axis, left, right } => {
                let left_el = self.render_pane_node(left, session, is_maximized, cx);
                let right_el = self.render_pane_node(right, session, is_maximized, cx);
                let is_horizontal = *axis == gpui::Axis::Horizontal;
                let split_id = ("terminal-split", *id);

                if is_horizontal {
                    h_resizable(split_id)
                        .child(
                            resizable_panel()
                                .size_range(px(100.0)..px(3000.0))
                                .child(div().flex().flex_col().size_full().p(px(2.0)).child(div().flex_1().size_full().child(left_el))),
                        )
                        .child(
                            resizable_panel()
                                .size_range(px(100.0)..px(3000.0))
                                .child(div().flex().flex_col().size_full().p(px(2.0)).child(div().flex_1().size_full().child(right_el))),
                        )
                        .into_any_element()
                } else {
                    v_resizable(split_id)
                        .child(
                            resizable_panel()
                                .size_range(px(70.0)..px(3000.0))
                                .child(div().flex().flex_col().size_full().p(px(2.0)).child(div().flex_1().size_full().child(left_el))),
                        )
                        .child(
                            resizable_panel()
                                .size_range(px(70.0)..px(3000.0))
                                .child(div().flex().flex_col().size_full().p(px(2.0)).child(div().flex_1().size_full().child(right_el))),
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

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgba(0x1b1c22ff))
            .border_r_1()
            .border_color(rgba(0x26272fff))
            .child(self.render_sidebar_toolbar(cx))
            .child(self.render_sidebar_projects_header(cx))
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
            .into_any_element()
    }

    fn render_sidebar_toolbar(&self, cx: &mut Context<Self>) -> gpui::Div {
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
                cx.listener::<MouseDownEvent>(|this, _, _, cx| {
                    this.sidebar_visible = !this.sidebar_visible;
                    cx.notify();
                }),
            ))
            .child(self.make_drag_area(cx).flex_1().h_full())
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.0))
                    .child(sidebar_icon_button(
                        "sidebar-open-project-btn",
                        "icons/folder-plus.svg",
                        15.0,
                        cx.listener::<MouseDownEvent>(|this, _, window, cx| {
                            this.select_folder_and_add_project(window, cx);
                        }),
                    ))
                    .child(sidebar_icon_button(
                        "sidebar-new-session-btn",
                        "icons/plus.svg",
                        15.0,
                        cx.listener::<MouseDownEvent>(|this, _, window, cx| {
                            this.create_new_session_for_active_project(window, cx);
                        }),
                    )),
            )
    }

    fn render_sidebar_projects_header(&self, cx: &mut Context<Self>) -> gpui::Div {
        div()
            .w_full()
            .h(px(28.0))
            .flex()
            .items_center()
            .justify_between()
            .pl(px(16.0))
            .pr(px(10.0))
            .child(
                div()
                    .text_size(px(10.0))
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(rgba(0x656679ff))
                    .child("PROJECTS"),
            )
            .child(sidebar_text_button(
                "sidebar-open-project-text-btn",
                "Open project",
                "icons/folder-plus.svg",
                false,
                cx.listener::<MouseDownEvent>(|this, _, window, cx| {
                    this.select_folder_and_add_project(window, cx);
                }),
            ))
    }

    /// One collapsible project block: the project header (activate, expand,
    /// new session, close project) followed by its session rows.
    fn render_project_entry(
        &self,
        project_idx: usize,
        project: &Project,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let is_active_project = project_idx == self.active_project_idx;
        let session_count = project.sessions.len();
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
                        .text_color(rgba(0x656679ff))
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
            rows.push(self.render_new_session_row(project_idx, project.id, cx));
        }

        div()
            .flex()
            .flex_col()
            .child(
                div()
                    .id(("project-row", project.id))
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
                    .bg(if is_active_project {
                        rgba(0x26272fff)
                    } else {
                        rgba(0x00000000)
                    })
                    .hover(|s| s.bg(rgba(0x23242cff)))
                    .child(
                        div()
                            .id(("project-chevron", project.id))
                            .size(px(18.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded(px(3.0))
                            .hover(|s| s.bg(rgba(0x32333eff)))
                            .child(
                                svg()
                                    .path(chevron)
                                    .size(px(12.0))
                                    .text_color(rgba(0x8788a5ff)),
                            )
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener::<MouseDownEvent>(move |this, _, _, cx| {
                                    cx.stop_propagation();
                                    this.toggle_project_expanded(project_idx, cx);
                                }),
                            ),
                    )
                    .child(
                        svg()
                            .path("icons/folder.svg")
                            .size(px(14.0))
                            .text_color(if is_active_project {
                                rgba(0xc3c1eeff)
                            } else {
                                rgba(0x8788a5ff)
                            }),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .text_size(px(12.5))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(if is_active_project {
                                rgba(0xe6edf3ff)
                            } else {
                                rgba(0xb9bac9ff)
                            })
                            .child(project.name.clone()),
                    )
                    .child(
                        div()
                            .flex_shrink_0()
                            .text_size(px(10.5))
                            .text_color(rgba(0x656679ff))
                            .child(format!("{session_count}")),
                    )
                    .child(sidebar_icon_button(
                        ("project-new-session", project.id),
                        "icons/plus.svg",
                        13.0,
                        cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                            this.create_new_session_in_project(project_idx, window, cx);
                        }),
                    ))
                    .child(sidebar_icon_button(
                        ("project-close", project.id),
                        "icons/x.svg",
                        12.0,
                        cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                            this.close_project(project_idx, window, cx);
                        }),
                    ))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                            // The whole header selects the project and reveals
                            // its sessions; the chevron is what collapses it.
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
        let (dot_color, text_color, badge_bg) = session.status.colors();
        let display_status = match session.status {
            SessionStatus::Active => "Active",
            SessionStatus::Idle => "Idle",
            SessionStatus::Working => "Working",
            SessionStatus::Blocked => "Blocked",
            SessionStatus::Error => "Error",
            SessionStatus::Done => "Done",
        };

        div()
            .id(("session-item", session.id))
            .ml(px(6.0))
            .mr(px(6.0))
            .h(px(34.0))
            .pl(px(16.0))
            .pr(px(5.0))
            .rounded(px(6.0))
            .cursor(CursorStyle::PointingHand)
            .flex()
            .items_center()
            .gap(px(8.0))
            .bg(if is_active_session {
                rgba(0x2b2c37ff)
            } else {
                rgba(0x00000000)
            })
            .border_1()
            .border_color(if is_active_session {
                rgba(0x353645ff)
            } else {
                rgba(0x00000000)
            })
            .hover(|s| {
                if !is_active_session {
                    s.bg(rgba(0x23242c88))
                } else {
                    s
                }
            })
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(9.0))
                    .min_w_0()
                    .flex_1()
                    .child(
                        // AI Agent / Command logo with status indicator
                        div()
                            .relative()
                            .size(px(20.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .flex_shrink_0()
                            .child(
                                gpui::img(session.logo.clone())
                                    .size(px(18.0))
                                    .rounded(px(3.5)),
                            )
                            .child(
                                div()
                                    .absolute()
                                    .bottom(-px(1.0))
                                    .right(-px(1.0))
                                    .size(px(7.0))
                                    .rounded_full()
                                    .bg(rgba(dot_color))
                                    .border_1()
                                    .border_color(if is_active_session {
                                        rgba(0x2b2c37ff)
                                    } else {
                                        rgba(0x1b1c22ff)
                                    }),
                            ),
                    )
                    .child(
                        div()
                            .text_size(px(12.5))
                            .truncate()
                            .text_color(if is_active_session {
                                rgba(0xd7d5f4ff)
                            } else {
                                rgba(0x8788a5ff)
                            })
                            .hover(|s| {
                                if !is_active_session {
                                    s.text_color(rgba(0xc3c1eeff))
                                } else {
                                    s
                                }
                            })
                            .child(session.display_title.clone()),
                    ),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .h(px(18.0))
                    .px(px(6.0))
                    .rounded_full()
                    .bg(rgba(badge_bg))
                    .border_1()
                    .border_color(rgba((dot_color & 0xffffff00) | 0x33))
                    .flex()
                    .items_center()
                    .gap(px(4.0))
                    .child(
                        div()
                            .size(px(5.0))
                            .rounded_full()
                            .bg(rgba(dot_color)),
                    )
                    .child(
                        div()
                            .text_size(px(10.0))
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .text_color(rgba(text_color))
                            .child(display_status),
                    ),
            )
            .child(sidebar_icon_button(
                ("session-close", session.id),
                "icons/x.svg",
                11.0,
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

    /// A quiet "+ New session" row under each expanded project, so creating a
    /// session never requires guessing which project is active.
    fn render_new_session_row(
        &self,
        project_idx: usize,
        project_id: usize,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        div()
            .id(("project-new-session-row", project_id))
            .ml(px(6.0))
            .mr(px(6.0))
            .mb(px(4.0))
            .h(px(26.0))
            .pl(px(17.0))
            .pr(px(6.0))
            .rounded(px(6.0))
            .cursor(CursorStyle::PointingHand)
            .flex()
            .items_center()
            .gap(px(7.0))
            .text_size(px(11.5))
            .text_color(rgba(0x656679ff))
            .hover(|s| s.bg(rgba(0x23242cff)).text_color(rgba(0xc3c1eeff)))
            .child(
                svg()
                    .path("icons/plus.svg")
                    .size(px(11.0))
                    .text_color(rgba(0x656679ff)),
            )
            .child("New session")
            .on_mouse_down(
                MouseButton::Left,
                cx.listener::<MouseDownEvent>(move |this, _, window, cx| {
                    this.create_new_session_in_project(project_idx, window, cx);
                }),
            )
            .into_any_element()
    }

    fn render_sidebar_empty_state(&self, cx: &mut Context<Self>) -> gpui::Div {
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
                    .bg(rgba(0x23242cff))
                    .child(
                        svg()
                            .path("icons/folder.svg")
                            .size(px(20.0))
                            .text_color(rgba(0x8788a5ff)),
                    ),
            )
            .child(
                div()
                    .text_size(px(12.5))
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(rgba(0xc3c1eeff))
                    .child("No projects open"),
            )
            .child(
                div()
                    .text_size(px(11.5))
                    .text_color(rgba(0x656679ff))
                    .text_center()
                    .child("Open any folder to start a project with its own sessions."),
            )
            .child(sidebar_text_button(
                "sidebar-empty-open-project",
                "Open project",
                "icons/folder-plus.svg",
                true,
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
                    .bg(rgba(0x1b1c22ff))
                    .child(
                        svg()
                            .path("icons/folder.svg")
                            .size(px(24.0))
                            .text_color(rgba(0x8788a5ff)),
                    ),
            )
            .child(
                div()
                    .text_size(px(14.0))
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(rgba(0xc3c1eeff))
                    .child(if has_project {
                        "No session in this project"
                    } else {
                        "No project open"
                    }),
            )
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(rgba(0x656679ff))
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
        if let Some(pane) = project.sessions[session_idx].panes.first() {
            pane.terminal.read(cx).focus_handle().focus(window);
        }
        self.focused_pane_id = pane_id;
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
    on_click: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .size(px(20.0))
        .rounded(px(4.0))
        .flex()
        .items_center()
        .justify_center()
        .flex_shrink_0()
        .cursor(CursorStyle::PointingHand)
        .hover(|s| s.bg(rgba(0x32333eff)))
        .on_mouse_down(MouseButton::Left, move |ev, window, cx| {
            cx.stop_propagation();
            on_click(ev, window, cx);
        })
        .child(
            svg()
                .path(icon)
                .size(px(icon_size))
                .text_color(rgba(0x8788a5ff))
                .hover(|s| s.text_color(rgba(0xf0f6fcff))),
        )
}

/// Slightly larger labelled button used for "Open project" / "New session"
/// empty-state actions.
fn sidebar_text_button(
    id: impl Into<gpui::ElementId>,
    label: &'static str,
    icon: &'static str,
    primary: bool,
    on_click: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .h(px(30.0))
        .px(px(12.0))
        .rounded(px(6.0))
        .cursor(CursorStyle::PointingHand)
        .flex()
        .items_center()
        .gap(px(6.0))
        .bg(if primary {
            rgba(0x1f6febff)
        } else {
            rgba(0x26272fff)
        })
        .border_1()
        .border_color(if primary {
            rgba(0x1f6febff)
        } else {
            rgba(0x32333eff)
        })
        .hover(move |s| {
            if primary {
                s.bg(rgba(0x2f81f7ff))
            } else {
                s.bg(rgba(0x2d2e38ff))
            }
        })
        .child(
            svg()
                .path(icon)
                .size(px(13.0))
                .text_color(rgba(0xffffffff)),
        )
        .child(
            div()
                .text_size(px(12.0))
                .text_color(rgba(0xffffffff))
                .child(label),
        )
        .on_mouse_down(MouseButton::Left, move |ev, window, cx| {
            cx.stop_propagation();
            on_click(ev, window, cx);
        })
}

impl Render for AppState {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Track which pane owns keyboard focus each frame; the poller uses
        // this for Active-state resolution and for clearing sticky Done/Error.
        let mut focused_pane = None;
        if let Some(session) = self.active_session() {
            for pane in &session.panes {
                if pane.terminal.read(cx).focus_handle().is_focused(_window) {
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
                self.render_terminal_pane(pane, false, cx).into_any_element()
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
                            .hover(|s| s.bg(rgba(0x23242cff)))
                            .on_mouse_down(MouseButton::Left, |_, window, cx| {
                                cx.stop_propagation();
                                window.minimize_window();
                            })
                            .child(
                                svg()
                                    .path("window-minimize.svg")
                                    .size(px(14.0))
                                    .text_color(rgba(0x8788a5ff))
                                    .hover(|s| s.text_color(rgba(0xf0f6fcff))),
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
                            .hover(|s| s.bg(rgba(0x23242cff)))
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
                                    .text_color(rgba(0x8788a5ff))
                                    .hover(|s| s.text_color(rgba(0xf0f6fcff))),
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
                                .text_color(rgba(0x8788a5ff))
                                .hover(|s| s.text_color(rgba(0xffffffff))),
                        ),
                )
        });

        let terminal_content = div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgba(0x14151bff))
            .child(
                self.make_drag_area(cx)
                    .h(px(38.0))
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_between()
                    .pl_4()
                    .bg(rgba(0x14151bff))
                    .border_b_1()
                    .border_color(rgba(0x20212aff))
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
                                                .text_color(rgba(0x8788a5ff))
                                                .hover(|s| s.text_color(rgba(0xf0f6fcff))),
                                        )
                                        .on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(|this, _, _, cx| {
                                                cx.stop_propagation();
                                                this.sidebar_visible = true;
                                                cx.notify();
                                            }),
                                        ),
                                )
                            })
                            .text_size(px(12.5))
                            .text_color(rgba(0x8788a5ff))
                            .children(titlebar_logo.map(|logo| {
                                div()
                                    .size(px(16.0))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .flex_shrink_0()
                                    .child(
                                        gpui::img(logo)
                                            .size(px(14.0))
                                            .rounded(px(2.0)),
                                    )
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
                                    .child(
                                        div()
                                            .size(px(6.0))
                                            .rounded_full()
                                            .bg(rgba(dot_color)),
                                    )
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
                    .bg(rgba(0x14151bff))
                    .overflow_hidden()
                    .p(px(6.0))
                    .child(div().flex().flex_col().size_full().child(terminal_area)),
            );

        let sidebar_content = self.render_sidebar(cx);

        div()
            .flex()
            .size_full()
            .bg(rgba(0x14151bff))
            .text_color(rgba(0xe6edf3ff))
            .on_key_down(cx.listener(Self::on_key_down))
            .child(
                if self.sidebar_visible {
                    h_resizable("workspace-layout-panels")
                        .child(
                            resizable_panel()
                                .size(self.sidebar_width)
                                .size_range(px(200.0)..px(500.0))
                                .child(sidebar_content),
                        )
                        .child(resizable_panel().child(terminal_content))
                        .into_any_element()
                } else {
                    terminal_content.into_any_element()
                },
            )
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

fn parse_hex(s: &str) -> Option<u32> {
    let hex = s.strip_prefix('#')?;
    if hex.len() == 6 {
        u32::from_str_radix(&format!("{hex}ff"), 16).ok()
    } else if hex.len() == 8 {
        u32::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

fn hex_to_rgb(hex: u32) -> (u8, u8, u8) {
    (
        ((hex >> 24) & 0xff) as u8,
        ((hex >> 16) & 0xff) as u8,
        ((hex >> 8) & 0xff) as u8,
    )
}

fn load_github_dark_palette() -> ColorPalette {
    let theme_data = assets::AppAssets::get("themes/github.json");
    if let Some(file) = theme_data {
        if let Ok(Value::Array(themes)) = serde_json::from_slice::<Value>(&file.data) {
            if let Some(github_dark) = themes.iter().find(|t| {
                t.get("name").and_then(Value::as_str) == Some("GitHub Dark")
            }) {
                let mut builder = ColorPalette::builder();
                let style = github_dark.get("style").and_then(Value::as_object);

                let get_color = |key: &str| -> Option<(u8, u8, u8)> {
                    style
                        .and_then(|s| s.get(key))
                        .and_then(Value::as_str)
                        .and_then(parse_hex)
                        .map(hex_to_rgb)
                };

                if let Some((r, g, b)) = get_color("terminal.background")
                    .or_else(|| get_color("editor.background"))
                    .or_else(|| get_color("background"))
                {
                    builder = builder.background(r, g, b);
                }

                if let Some((r, g, b)) = get_color("terminal.foreground")
                    .or_else(|| get_color("editor.foreground"))
                    .or_else(|| get_color("text"))
                {
                    builder = builder.foreground(r, g, b);
                }

                let cursor_color = github_dark
                    .get("players")
                    .and_then(Value::as_array)
                    .and_then(|p| p.first())
                    .and_then(|p| p.get("cursor"))
                    .and_then(Value::as_str)
                    .and_then(parse_hex)
                    .map(hex_to_rgb)
                    .or_else(|| get_color("text.accent"))
                    .or_else(|| get_color("terminal.bright_foreground"))
                    .or_else(|| get_color("terminal.foreground"));

                if let Some((r, g, b)) = cursor_color {
                    builder = builder.cursor(r, g, b);
                }

                let ansi_keys = [
                    ("terminal.ansi.black", 0),
                    ("terminal.ansi.red", 1),
                    ("terminal.ansi.green", 2),
                    ("terminal.ansi.yellow", 3),
                    ("terminal.ansi.blue", 4),
                    ("terminal.ansi.magenta", 5),
                    ("terminal.ansi.cyan", 6),
                    ("terminal.ansi.white", 7),
                    ("terminal.ansi.bright_black", 8),
                    ("terminal.ansi.bright_red", 9),
                    ("terminal.ansi.bright_green", 10),
                    ("terminal.ansi.bright_yellow", 11),
                    ("terminal.ansi.bright_blue", 12),
                    ("terminal.ansi.bright_magenta", 13),
                    ("terminal.ansi.bright_cyan", 14),
                    ("terminal.ansi.bright_white", 15),
                ];

                for (key, idx) in ansi_keys {
                    if let Some((r, g, b)) = get_color(key) {
                        builder = match idx {
                            0 => builder.black(r, g, b),
                            1 => builder.red(r, g, b),
                            2 => builder.green(r, g, b),
                            3 => builder.yellow(r, g, b),
                            4 => builder.blue(r, g, b),
                            5 => builder.magenta(r, g, b),
                            6 => builder.cyan(r, g, b),
                            7 => builder.white(r, g, b),
                            8 => builder.bright_black(r, g, b),
                            9 => builder.bright_red(r, g, b),
                            10 => builder.bright_green(r, g, b),
                            11 => builder.bright_yellow(r, g, b),
                            12 => builder.bright_blue(r, g, b),
                            13 => builder.bright_magenta(r, g, b),
                            14 => builder.bright_cyan(r, g, b),
                            15 => builder.bright_white(r, g, b),
                            _ => builder,
                        };
                    }
                }

                return builder.build();
            }
        }
    }

    ColorPalette::builder()
        .background(0x01, 0x04, 0x09)
        .foreground(0xf0, 0xf6, 0xfc)
        .cursor(0x44, 0x93, 0xf8)
        .build()
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
    Application::new()
        .with_assets(CombinedAssets)
        .run(|cx: &mut App| {
            #[cfg(target_os = "linux")]
            ensure_linux_desktop_entry();
            let window_icon = gpui::WindowIcon::from_png_bytes(APP_ICON_PNG).ok();
            gpui_component::init(cx);
            gpui_component::Theme::change(gpui_component::ThemeMode::Dark, None, cx);
            load_embedded_fonts(cx);
            sync_component_fonts(cx);

            let palette = load_github_dark_palette();
            let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let current_name = current_dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "Workspace".to_string());

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
                    let palette_clone = palette.clone();
                    let current_dir_clone = current_dir.clone();
                    let app_state = cx.new(|cx| {
                        let self_weak = cx.entity().downgrade();
                        let (initial_terminal, pid) = AppState::spawn_terminal_view(
                            Some(&current_dir_clone),
                            palette_clone.clone(),
                            self_weak,
                            1,
                            cx,
                        )
                        .expect("Failed to spawn initial terminal");

                        initial_terminal.read(cx).focus_handle().focus(window);

                        let initial_pane = TerminalPane {
                            id: 1,
                            terminal: initial_terminal,
                            status: SessionStatus::Idle,
                            title: "terminal 1".to_string(),
                            display_title: SharedString::from("terminal 1"),
                            pid,
                            screen_scratch: Vec::new(),
                            poll_cache: None,
                            ack_screen: None,
                            agent_active: false,
                            done_since: None,
                            detector: status::Detector::default(),
                        };

                        let initial_session = Session {
                            id: 1,
                            title: SharedString::from("terminal 1"),
                            display_title: SharedString::from("terminal 1"),
                            logo: logos::default_logo(),
                            pane_tree: Some(PaneNode::Leaf(1)),
                            maximized_pane_id: None,
                            panes: vec![initial_pane],
                            status: SessionStatus::Idle,
                        };

                        let initial_project = Project {
                            id: 1,
                            name: SharedString::from(current_name),
                            path: current_dir_clone,
                            sessions: vec![initial_session],
                            active_session_idx: 0,
                            expanded: true,
                        };

                        AppState {
                            projects: vec![initial_project],
                            active_project_idx: 0,
                            next_id: 2,
                            session_index: HashMap::from([(1, (0, 0))]),
                            pane_index: HashMap::from([(1, (0, 0, 0))]),
                            palette: palette_clone,
                            focused_pane_id: Some(1),
                            sidebar_visible: true,
                            sidebar_width: px(330.0),
                            drag_start_pos: None,
                        }
                    });

                    let app_state_weak = app_state.downgrade();
                    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                        loop {
                            cx.background_executor().timer(std::time::Duration::from_millis(500)).await;
                            let res = app_state_weak.update(cx, |state, cx| {
                                state.poll_sessions_status(cx);
                            });
                            if res.is_err() {
                                break;
                            }
                        }
                    })
                    .detach();

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
    use gpui::{point, size, Modifiers, TestAppContext, VisualTestContext};
    use std::ops::Deref as _;

    /// One project with one live session, built exactly like `main` does.
    fn build_app_state(cx: &mut Context<AppState>) -> AppState {
        let palette = ColorPalette::default();
        let dir = std::env::current_dir().expect("current dir");
        let weak = cx.entity().downgrade();
        let (terminal, pid) =
            AppState::spawn_terminal_view(Some(&dir), palette.clone(), weak, 1, cx)
                .expect("terminal view");
        let title = SharedString::from("terminal 1");

        let pane = TerminalPane {
            id: 1,
            terminal,
            status: SessionStatus::Idle,
            title: "terminal 1".to_string(),
            display_title: title.clone(),
            pid,
            screen_scratch: Vec::new(),
            poll_cache: None,
            ack_screen: None,
            agent_active: false,
            done_since: None,
            detector: status::Detector::default(),
        };

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
            sidebar_width: px(330.0),
            drag_start_pos: None,
        }
    }

    /// Open the app the way `main` does and return a context that can draw it.
    pub(crate) fn open_app(cx: &mut TestAppContext) -> (VisualTestContext, Entity<AppState>) {
        cx.update(|cx| {
            gpui_component::init(cx);
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
            state.active_session().expect("a session").panes[0].terminal.clone()
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

    /// The sidebar is resized by dragging its handle. That path runs its own
    /// arithmetic over the panels' sizes, so dragging it past both the min and
    /// the max has to stay panic-free.
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
}

#[cfg(test)]
mod xtgettcap_tests {
    //! End-to-end check of the terminal crate's XTGETTCAP passthrough: a
    //! capability query must produce a reply on the PTY writer and must never
    //! reach the grid as printable output.

    use alacritty_terminal::event::WindowSize;
    use gpui_terminal::{ColorPalette, GpuiEventProxy, PtyWriter, TerminalNotifier, TerminalState};
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
        assert!(!capture.0.lock().is_empty(), "the query should have been answered");
    }
}

#[cfg(test)]
mod pane_tests {
    use super::*;
    use gpui::{point, Modifiers, TestAppContext, VisualContext};
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
            state.active_project().unwrap().sessions[0].pane_tree.clone()
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

        let (project_count, active_idx, first_sessions, second_sessions, expanded) =
            app_state.read_with(&cx, |state, _| {
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
        cx.draw_frame().expect("frame with two projects and two sessions");

        // Close the extra session in project 0. Project 1 must be untouched.
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.close_session(0, 1, window, cx);
        });
        cx.draw_frame().expect("frame after closing a session");
        let (first, second) = app_state.read_with(&cx, |state, _| {
            (state.projects[0].sessions.len(), state.projects[1].sessions.len())
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

}
