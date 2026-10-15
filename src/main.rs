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
pub struct Session {
    pub id: usize,
    pub title: SharedString,
    /// Title as shown in the sidebar (truncated to 160 chars), cached so the
    /// render pass does not re-scan and re-allocate it for every session of
    /// every frame.
    pub display_title: SharedString,
    pub terminal: Entity<TerminalView>,
    pub status: SessionStatus,
    pub pid: Option<u32>,
    /// Reused scrape buffer: the poller reads the screen into this instead of
    /// allocating a fresh `Vec<String>` every 500 ms per session.
    pub screen_scratch: Vec<String>,
    /// (screen fingerprint, focused, exited, foreground fingerprint) from the
    /// last poll tick. When it matches, nothing observable changed and the
    /// whole rule-pack evaluation + logo resolution is skipped.
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
    /// Path of the AI agent / command logo shown in the sidebar.
    pub logo: SharedString,
    pub detector: status::Detector,
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
    pub pane_tree: Option<PaneNode>,
    pub maximized_session_id: Option<usize>,
}

struct AppState {
    projects: Vec<Project>,
    active_project_idx: usize,
    next_id: usize,
    palette: ColorPalette,
    /// O(1) session lookup by id: session id -> (project idx, session idx).
    session_index: HashMap<usize, (usize, usize)>,
    /// Session that currently owns keyboard focus, tracked every render frame
    /// and consumed by the status poller (Active state, sticky-state clearing).
    focused_session_id: Option<usize>,
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
        session_id: usize,
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
                        state.update_session_title(session_id, &title_str, cx);
                    });
                })
            // Exit status is intentionally not set here: the status poller
            // derives Done/Error from the real exit code (see poll_sessions_status).
        });
        Ok((terminal, pid))
    }

    fn update_session_title(&mut self, session_id: usize, title: &str, cx: &mut Context<Self>) {
        let trimmed = title.trim();
        if trimmed.is_empty() {
            return;
        }
        let Some(&(p_idx, s_idx)) = self.session_index.get(&session_id) else {
            return;
        };
        let Some(session) = self
            .projects
            .get_mut(p_idx)
            .and_then(|p| p.sessions.get_mut(s_idx))
        else {
            return;
        };
        // Unchanged titles are common (TUIs re-emit them constantly); the logo
        // depends only on title/status/logo, so there is nothing to redo.
        if session.title.as_str() == trimmed {
            return;
        }
        let next = SharedString::from(trimmed.to_string());
        session.title = next.clone();
        session.display_title = truncate_title(&next);
        let title_logo =
            logos::resolve_session_logo(&[], Some(trimmed), session.status, &session.logo);
        if title_logo != logos::default_logo() && session.logo != title_logo {
            session.logo = title_logo;
        }
        cx.notify();
    }

    /// Periodic detection tick, herdr-style: screen scrape + OSC title +
    /// foreground process tree, with sticky Done/Error and focus resolution.
    pub fn poll_sessions_status(&mut self, cx: &mut Context<Self>) {
        let mut changed = false;
        let focused_id = self.focused_session_id;
        // `T3_STATUS_DEBUG=1` dumps every detection pass: the facts that went in
        // and the badge that came out. Without this the poller is invisible.
        let debug = status::debug_enabled();
        for project in &mut self.projects {
            for session in &mut project.sessions {
                let focused = focused_id == Some(session.id);

                // Background terminals never render, so their PTY events
                // (title updates, child exit) must be drained here or the
                // status would go stale while the user is in another tab.
                if !focused {
                    session.terminal.update(cx, |term, cx| {
                        term.drain_background_events(cx);
                    });
                }

                // Reused for both the debug dump and the agent lifecycle below.
                let foreground = status::foreground_process_names(session.pid);

                // Scrape into the session's reused buffer. `bottom_lines` takes
                // the last N *rows*, clamped to the visible screen. Ask for the
                // whole screen: a fresh shell prints its prompt on row 0, well
                // above a 12-row window at the bottom. The rules skip blank
                // rows themselves.
                session
                    .terminal
                    .read(cx)
                    .bottom_lines_into(usize::MAX, &mut session.screen_scratch);

                // Idle fast path: when the screen, focus, exit state, and
                // foreground group all match last tick, the status, ack, and
                // logo cannot have changed — skip the rule-pack evaluation
                // entirely. Done is excluded: it expires on a timer even when
                // nothing else changes.
                let poll_key = {
                    let terminal = session.terminal.read(cx);
                    let fp = screen_fingerprint(
                        &session.screen_scratch,
                        terminal.title().unwrap_or(""),
                    );
                    (
                        fp,
                        focused,
                        terminal.has_exited(),
                        screen_fingerprint(&foreground, ""),
                    )
                };
                if session.poll_cache == Some(poll_key)
                    && session.status != SessionStatus::Done
                    && session.status != SessionStatus::Working
                {
                    continue;
                }

                let (next, ack, screen_fp) = {
                    let terminal = session.terminal.read(cx);
                    let bottom_lines = &session.screen_scratch;
                    let title = terminal.title().unwrap_or("");
                    let screen_fp = screen_fingerprint(bottom_lines, title);
                    let screen_text = bottom_lines.join("\n");
                    let detector_state = session.detector.update(
                        &screen_text,
                        terminal.has_exited(),
                        std::time::Instant::now(),
                    );
                    // while the screen is unchanged the sticky badge stays
                    // cleared; new output re-arms it.
                    let ack = if focused
                        && (terminal.has_exited()
                            || matches!(session.status, SessionStatus::Done | SessionStatus::Error))
                    {
                        Some(screen_fp)
                    } else {
                        session.ack_screen
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
                            status::evaluate_with_match(&input, session.status, focused)
                        } else {
                            (status::evaluate(&input, session.status, focused), None)
                        };
                        // Stash the match for the debug dump below so the pack
                        // is not evaluated a second time to explain the badge.
                        debug_match = rule_match;
                        if acknowledged && matches!(detected, SessionStatus::Done | SessionStatus::Error) {
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
                            "[status] session={} focused={focused} exited={} exit={:?} \
                             pid={:?} fg={foreground:?} pack={:?} title={title:?} \
                             prev={} next={} detector={detector_state:?} rule={rule}",
                            session.id,
                            terminal.has_exited(),
                            terminal.exit_status(),
                            session.pid,
                            status::pack_for(&foreground),
                            session.status.label(),
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

                    (next, ack, screen_fp)
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
                let agent_exited = session.agent_active && !agent_now;
                session.agent_active = agent_now;

                let turn_finished = matches!(
                    session.status,
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
                    let since = *session.done_since.get_or_insert_with(std::time::Instant::now);
                    if since.elapsed() >= DONE_LINGER {
                        session.done_since = None;
                        SessionStatus::Idle
                    } else {
                        next
                    }
                } else {
                    session.done_since = None;
                    next
                };

                session.ack_screen = ack;
                if session.status != next {
                    session.status = next;
                    changed = true;
                }

                let term_title = session.terminal.read(cx).title();
                let next_logo = logos::resolve_session_logo(
                    &foreground,
                    term_title,
                    session.status,
                    &session.logo,
                );
                if session.logo != next_logo {
                    session.logo = next_logo;
                    changed = true;
                }

                session.poll_cache = Some(poll_key);
            }
        }
        if changed {
            cx.notify();
        }
    }
    fn create_new_session_for_active_project(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let self_weak = cx.entity().downgrade();
        let next_id = self.next_id;
        self.next_id += 1;

        let dir = self.active_project().map(|p| p.path.clone());
        if let Ok((terminal, pid)) = Self::spawn_terminal_view(
            dir.as_ref(),
            self.palette.clone(),
            self_weak,
            next_id,
            cx,
        ) {
            terminal.read(cx).focus_handle().focus(window);
            let session_count = self
                .active_project()
                .map(|p| p.sessions.len())
                .unwrap_or(0);
            let title = SharedString::from(format!("terminal {}", session_count + 1));
            let session = Session {
                id: next_id,
                title: title.clone(),
                display_title: truncate_title(&title),
                terminal,
                status: SessionStatus::Idle,
                pid,
                screen_scratch: Vec::new(),
                poll_cache: None,
                ack_screen: None,
                agent_active: false,
                done_since: None,
                logo: logos::default_logo(),
                detector: status::Detector::default(),
            };

            let p_idx = self.active_project_idx;
            let pushed = if let Some(project) = self.active_project_mut() {
                let target_id = project.sessions.get(project.active_session_idx).map(|s| s.id);
                project.sessions.push(session);
                project.active_session_idx = project.sessions.len() - 1;
                if let Some(tree) = &mut project.pane_tree {
                    if let Some(target) = target_id {
                        let split_id = next_id;
                        tree.split(target, next_id, split_id, gpui::Axis::Horizontal);
                    } else {
                        *tree = PaneNode::Leaf(next_id);
                    }
                } else {
                    project.pane_tree = Some(PaneNode::Leaf(next_id));
                }
                project.maximized_session_id = None;
                Some(project.active_session_idx)
            } else {
                None
            };
            if let Some(s_idx) = pushed {
                self.session_index.insert(next_id, (p_idx, s_idx));
                self.focused_session_id = Some(next_id);
                cx.notify();
            }
        }
    }

    fn select_folder_and_add_project(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
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
                let session = Session {
                    id: next_id,
                    title: SharedString::from("terminal 1"),
                    display_title: SharedString::from("terminal 1"),
                    terminal,
                    status: SessionStatus::Idle,
                    pid,
                    screen_scratch: Vec::new(),
                    poll_cache: None,
                    ack_screen: None,
                    agent_active: false,
                    done_since: None,
                    logo: logos::default_logo(),
                    detector: status::Detector::default(),
                };
                let project = Project {
                    id: next_id,
                    name: SharedString::from(name),
                    path: folder,
                    sessions: vec![session],
                    active_session_idx: 0,
                    pane_tree: Some(PaneNode::Leaf(next_id)),
                    maximized_session_id: None,
                };

                self.projects.push(project);
                self.active_project_idx = self.projects.len() - 1;
                let p_idx = self.active_project_idx;
                self.session_index.insert(next_id, (p_idx, 0));
                self.focused_session_id = Some(next_id);
                cx.notify();
            }
        }
    }

    fn rebuild_session_index(&mut self) {
        self.session_index.clear();
        for (p_idx, project) in self.projects.iter().enumerate() {
            for (s_idx, session) in project.sessions.iter().enumerate() {
                self.session_index.insert(session.id, (p_idx, s_idx));
            }
        }
    }

    fn split_session(
        &mut self,
        target_session_id: usize,
        axis: gpui::Axis,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let self_weak = cx.entity().downgrade();
        let split_id = self.next_id;
        self.next_id += 1;
        let next_id = self.next_id;
        self.next_id += 1;
        let p_idx = self
            .session_index
            .get(&target_session_id)
            .map(|(p, _)| *p)
            .unwrap_or(self.active_project_idx);

        let dir = self.projects.get(p_idx).map(|p| p.path.clone());
        if let Ok((terminal, pid)) = Self::spawn_terminal_view(
            dir.as_ref(),
            self.palette.clone(),
            self_weak,
            next_id,
            cx,
        ) {
            terminal.read(cx).focus_handle().focus(window);
            let session_count = self
                .projects
                .get(p_idx)
                .map(|p| p.sessions.len())
                .unwrap_or(0);
            let title = SharedString::from(format!("terminal {}", session_count + 1));
            let session = Session {
                id: next_id,
                title: title.clone(),
                display_title: truncate_title(&title),
                terminal,
                status: SessionStatus::Idle,
                pid,
                screen_scratch: Vec::new(),
                poll_cache: None,
                ack_screen: None,
                agent_active: false,
                done_since: None,
                logo: logos::default_logo(),
                detector: status::Detector::default(),
            };

            if let Some(project) = self.projects.get_mut(p_idx) {
                project.sessions.push(session);
                project.active_session_idx = project.sessions.len() - 1;
                if let Some(tree) = &mut project.pane_tree {
                    tree.split(target_session_id, next_id, split_id, axis);
                } else {
                    project.pane_tree = Some(PaneNode::Leaf(next_id));
                }
                project.maximized_session_id = None;
            }
            self.focused_session_id = Some(next_id);
            self.rebuild_session_index();
            cx.notify();
        }
    }

    fn toggle_maximize_session(&mut self, session_id: usize, cx: &mut Context<Self>) {
        let p_idx = self
            .session_index
            .get(&session_id)
            .map(|(p, _)| *p)
            .unwrap_or(self.active_project_idx);

        if let Some(project) = self.projects.get_mut(p_idx) {
            if project.maximized_session_id == Some(session_id) {
                project.maximized_session_id = None;
            } else {
                project.maximized_session_id = Some(session_id);
            }
            cx.notify();
        }
    }

    fn close_session(&mut self, session_id: usize, window: &mut Window, cx: &mut Context<Self>) {
        let p_idx = self
            .session_index
            .get(&session_id)
            .map(|(p, _)| *p)
            .unwrap_or(self.active_project_idx);

        let mut needs_new_session = false;
        if let Some(project) = self.projects.get_mut(p_idx) {
            if project.maximized_session_id == Some(session_id) {
                project.maximized_session_id = None;
            }
            project.pane_tree = project.pane_tree.take().and_then(|t| t.remove(session_id));
            project.sessions.retain(|s| s.id != session_id);

            if let Some(tree) = &project.pane_tree {
                let next_active = tree.first_leaf();
                if let Some(pos) = project.sessions.iter().position(|s| s.id == next_active) {
                    project.active_session_idx = pos;
                    project.sessions[pos].terminal.read(cx).focus_handle().focus(window);
                    self.focused_session_id = Some(next_active);
                }
            } else if project.sessions.is_empty() {
                needs_new_session = true;
            } else {
                let first_id = project.sessions[0].id;
                project.pane_tree = Some(PaneNode::Leaf(first_id));
                project.active_session_idx = 0;
                project.sessions[0].terminal.read(cx).focus_handle().focus(window);
                self.focused_session_id = Some(first_id);
            }
        }

        self.rebuild_session_index();

        if needs_new_session {
            self.create_new_session_for_active_project(window, cx);
        } else {
            cx.notify();
        }
    }

    fn focus_session(&mut self, session_id: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.focused_session_id = Some(session_id);
        if let Some(&(p_idx, s_idx)) = self.session_index.get(&session_id) {
            self.active_project_idx = p_idx;
            if let Some(p) = self.projects.get_mut(p_idx) {
                p.active_session_idx = s_idx;
                if let Some(session) = p.sessions.get(s_idx) {
                    session.terminal.read(cx).focus_handle().focus(window);
                }
            }
        }
        cx.notify();
    }

    fn render_terminal_pane(
        &self,
        session: &Session,
        is_maximized: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        let session_id = session.id;
        let is_focused = self.focused_session_id == Some(session_id);

        div()
            .id(("terminal-pane", session_id))
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
                    this.focus_session(session_id, window, cx);
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
                            .child(session.display_title.clone()),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.0))
                            .child(
                                div()
                                    .id(("split-v-btn", session_id))
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
                                            this.split_session(session_id, gpui::Axis::Horizontal, window, cx);
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id(("split-h-btn", session_id))
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
                                            this.split_session(session_id, gpui::Axis::Vertical, window, cx);
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id(("maximize-btn", session_id))
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
                                            this.toggle_maximize_session(session_id, cx);
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id(("close-btn", session_id))
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
                                            this.close_session(session_id, window, cx);
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
                    .child(session.terminal.clone()),
            )
    }

    fn render_pane_node(
        &self,
        node: &PaneNode,
        project: &Project,
        is_maximized: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        match node {
            PaneNode::Leaf(session_id) => {
                if let Some(session) = project.sessions.iter().find(|s| s.id == *session_id) {
                    self.render_terminal_pane(session, is_maximized, cx)
                        .into_any_element()
                } else {
                    div().into_any_element()
                }
            }
            PaneNode::Split { id, axis, left, right } => {
                let left_el = self.render_pane_node(left, project, is_maximized, cx);
                let right_el = self.render_pane_node(right, project, is_maximized, cx);
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

    fn on_key_down(&mut self, event: &KeyDownEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let keystroke = &event.keystroke;

        if keystroke.modifiers.control && (keystroke.key == "+" || keystroke.key == "=") {
            if let Some(session) = self.active_session() {
                session.terminal.update(cx, |terminal, cx| {
                    let mut config = terminal.config().clone();
                    config.font_size += px(1.0);
                    terminal.update_config(config, cx);
                });
            }
            cx.stop_propagation();
        } else if keystroke.modifiers.control && keystroke.key == "-" {
            if let Some(session) = self.active_session() {
                session.terminal.update(cx, |terminal, cx| {
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
}

impl Render for AppState {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Track which session owns keyboard focus each frame; the poller uses
        // this for Active-state resolution and for clearing sticky Done/Error.
        if let Some(project) = self.active_project() {
            let mut focused_id = None;
            let mut focused_idx = None;
            for (idx, s) in project.sessions.iter().enumerate() {
                if s.terminal.read(cx).focus_handle().is_focused(_window) {
                    focused_id = Some(s.id);
                    focused_idx = Some(idx);
                    break;
                }
            }
            if let (Some(id), Some(idx)) = (focused_id, focused_idx) {
                self.focused_session_id = Some(id);
                if let Some(p) = self.projects.get_mut(self.active_project_idx) {
                    p.active_session_idx = idx;
                }
            } else if self.focused_session_id.is_none() {
                self.focused_session_id = self.active_session().map(|s| s.id);
            }
        }

        let active_title = self
            .active_session()
            .map(|s| s.title.clone())
            .unwrap_or_else(|| SharedString::from("Terminal"));

        let active_project_name = self
            .active_project()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| SharedString::from("No Project"));

        let terminal_area = if let Some(project) = self.active_project() {
            if let Some(max_id) = project.maximized_session_id {
                if let Some(session) = project.sessions.iter().find(|s| s.id == max_id) {
                    self.render_terminal_pane(session, true, cx).into_any_element()
                } else if let Some(tree) = &project.pane_tree {
                    self.render_pane_node(tree, project, false, cx)
                } else {
                    div().into_any_element()
                }
            } else if let Some(tree) = &project.pane_tree {
                self.render_pane_node(tree, project, false, cx)
            } else if let Some(session) = self.active_session() {
                self.render_terminal_pane(session, false, cx).into_any_element()
            } else {
                div().into_any_element()
            }
        } else {
            div().into_any_element()
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

        let sidebar_content = div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgba(0x1b1c22ff))
            .border_r_1()
            .border_color(rgba(0x26272fff))
            // Top Row: sidebar toggle, spacer, eye, cmd, plus
            .child(
                div()
                    .h(px(38.0))
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_between()
                    .pl(px(14.0))
                    .pr(px(14.0))
                    .child(
                        div()
                            .id("sidebar-toggle-btn")
                            .cursor(CursorStyle::PointingHand)
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
                                    this.sidebar_visible = !this.sidebar_visible;
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(
                        self.make_drag_area(cx)
                            .flex_1()
                            .h_full(),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(14.0))
                            .child(
                                div()
                                    .id("sidebar-view-btn")
                                    .cursor(CursorStyle::PointingHand)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(
                                        svg()
                                            .path("icons/eye.svg")
                                            .size(px(16.0))
                                            .text_color(rgba(0x8788a5ff))
                                            .hover(|s| s.text_color(rgba(0xf0f6fcff))),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        |_, _, cx| cx.stop_propagation(),
                                    ),
                            )
                            .child(
                                div()
                                    .id("sidebar-command-btn")
                                    .cursor(CursorStyle::PointingHand)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(
                                        svg()
                                            .path("icons/command.svg")
                                            .size(px(14.0))
                                            .text_color(rgba(0x8788a5ff))
                                            .hover(|s| s.text_color(rgba(0xf0f6fcff))),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.select_folder_and_add_project(window, cx);
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id("sidebar-add-btn")
                                    .cursor(CursorStyle::PointingHand)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(
                                        svg()
                                            .path("icons/plus.svg")
                                            .size(px(16.0))
                                            .text_color(rgba(0x8788a5ff))
                                            .hover(|s| s.text_color(rgba(0xf0f6fcff))),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.create_new_session_for_active_project(window, cx);
                                        }),
                                    ),
                            ),
                    ),
            )
            // Projects & Sessions List
            .child(
                div()
                    .flex_1()
                    .overflow_y_scrollbar()
                    .flex()
                    .flex_col()
                    .children(self.projects.iter().enumerate().map(|(p_idx, project)| {
                        let is_active_project = p_idx == self.active_project_idx;
                        div()
                            .flex()
                            .flex_col()
                            // Project Title Header ("icy-narrows")
                            .child(
                                div()
                                    .id(("project-header", project.id))
                                    .pt(px(14.0))
                                    .pb(px(8.0))
                                    .px(px(16.0))
                                    .cursor(CursorStyle::PointingHand)
                                    .child(
                                        div()
                                            .text_size(px(12.0))
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .text_color(rgba(0x656679ff))
                                            .hover(|s| s.text_color(rgba(0x8b949eff)))
                                            .child(project.name.clone()),
                                    )
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(move |this, _, window, cx| {
                                            this.active_project_idx = p_idx;
                                            if let Some(session) = this.active_session() {
                                                session.terminal.read(cx).focus_handle().focus(window);
                                            }
                                            cx.notify();
                                        }),
                                    ),
                            )
                            // Sessions List for this project
                            .child(
                                div()
                                    .px(px(6.0))
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.0))
                                    .children(project.sessions.iter().enumerate().map(|(s_idx, session)| {
                                        let is_active_session = is_active_project && s_idx == project.active_session_idx;
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
                                            .h(px(36.0))
                                            .px(px(10.0))
                                            .rounded_full()
                                            .cursor(CursorStyle::PointingHand)
                                            .flex()
                                            .items_center()
                                            .justify_between()
                                            .gap(px(8.0))
                                            .bg(if is_active_session {
                                                rgba(0x26272fff)
                                            } else {
                                                rgba(0x00000000)
                                            })
                                            .border_1()
                                            .border_color(if is_active_session {
                                                rgba(0x32333eff)
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
                                                    .gap(px(10.0))
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
                                                                        rgba(0x26272fff)
                                                                    } else {
                                                                        rgba(0x1b1c22ff)
                                                                    }),
                                                            ),
                                                    )
                                                    .child(
                                                        div()
                                                            .text_size(px(13.0))
                                                            .truncate()
                                                            .text_color(if is_active_session {
                                                                rgba(0xc3c1eeff)
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
                                                    .h(px(19.0))
                                                    .px(px(7.0))
                                                    .rounded_full()
                                                    .bg(rgba(badge_bg))
                                                    .border_1()
                                                    .border_color(rgba((dot_color & 0xffffff00) | 0x33))
                                                    .flex()
                                                    .items_center()
                                                    .gap(px(5.0))
                                                    .child(
                                                        div()
                                                            .size(px(5.0))
                                                            .rounded_full()
                                                            .bg(rgba(dot_color)),
                                                    )
                                                    .child(
                                                        div()
                                                            .text_size(px(10.5))
                                                            .font_weight(gpui::FontWeight::MEDIUM)
                                                            .text_color(rgba(text_color))
                                                            .child(display_status),
                                                    ),
                                            )
                                            .on_mouse_down(
                                                MouseButton::Left,
                                                cx.listener(move |this, _, window, cx| {
                                                    this.active_project_idx = p_idx;
                                                    if let Some(p) = this.projects.get_mut(p_idx) {
                                                        p.active_session_idx = s_idx;
                                                        let s_id = p.sessions[s_idx].id;
                                                        if !p.pane_tree.as_ref().map_or(false, |t| t.contains(s_id)) {
                                                            let split_id = this.next_id;
                                                            this.next_id += 1;
                                                            if let Some(tree) = &mut p.pane_tree {
                                                                let active_id = tree.first_leaf();
                                                                tree.split(active_id, s_id, split_id, gpui::Axis::Horizontal);
                                                            } else {
                                                                p.pane_tree = Some(PaneNode::Leaf(s_id));
                                                            }
                                                        }
                                                        p.maximized_session_id = None;
                                                    }
                                                    if let Some(session) = this.active_session() {
                                                        session.terminal.read(cx).focus_handle().focus(window);
                                                    }
                                                    this.focused_session_id = this.active_session().map(|s| s.id);
                                                    cx.notify();
                                                }),
                                            )
                                    }))
                            )
                    })),
            );

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

fn main() -> Result<()> {
    Application::new()
        .with_assets(CombinedAssets)
        .run(|cx: &mut App| {
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
                    app_id: Some("t3code-terminal".to_string()),
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

                        let initial_session = Session {
                            id: 1,
                            title: SharedString::from("terminal 1"),
                            display_title: SharedString::from("terminal 1"),
                            terminal: initial_terminal,
                            status: SessionStatus::Idle,
                            pid,
                            screen_scratch: Vec::new(),
                            poll_cache: None,
                            ack_screen: None,
                            agent_active: false,
                            done_since: None,
                            logo: logos::default_logo(),
                            detector: status::Detector::default(),
                        };

                        let initial_project = Project {
                            id: 1,
                            name: SharedString::from(current_name),
                            path: current_dir_clone,
                            sessions: vec![initial_session],
                            active_session_idx: 0,
                            pane_tree: Some(PaneNode::Leaf(1)),
                            maximized_session_id: None,
                        };

                        AppState {
                            projects: vec![initial_project],
                            active_project_idx: 0,
                            next_id: 2,
                            session_index: HashMap::from([(1, (0, 0))]),
                            palette: palette_clone,
                            focused_session_id: None,
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

        AppState {
            projects: vec![Project {
                id: 1,
                name: SharedString::from("markup"),
                path: dir,
                sessions: vec![Session {
                    id: 1,
                    title: title.clone(),
                    display_title: title,
                    terminal,
                    status: SessionStatus::Idle,
                    pid,
                    screen_scratch: Vec::new(),
                    poll_cache: None,
                    ack_screen: None,
                    agent_active: false,
                    done_since: None,
                    logo: logos::default_logo(),
                    detector: status::Detector::default(),
                }],
                active_session_idx: 0,
                pane_tree: Some(PaneNode::Leaf(1)),
                maximized_session_id: None,
            }],
            active_project_idx: 0,
            next_id: 2,
            session_index: HashMap::from([(1, (0, 0))]),
            palette,
            focused_session_id: None,
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
                        app_id: Some("t3code-terminal".to_string()),
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
            state.active_session().expect("a session").terminal.clone()
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
            state.split_session(1, gpui::Axis::Horizontal, window, cx);
        });
        cx.draw_frame().expect("frame after vertical split");

        let session_count = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions.len()
        });
        assert_eq!(session_count, 2);

        let new_session_id = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions.last().unwrap().id
        });

        // Toggle maximize
        app_state.update(&mut cx, |state, cx| {
            state.toggle_maximize_session(new_session_id, cx);
        });
        cx.draw_frame().expect("frame while maximized");

        let is_max = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().maximized_session_id
        });
        assert_eq!(is_max, Some(new_session_id));

        // Restore
        app_state.update(&mut cx, |state, cx| {
            state.toggle_maximize_session(new_session_id, cx);
        });
        cx.draw_frame().expect("frame restored");

        // Close the new pane
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.close_session(new_session_id, window, cx);
        });
        cx.draw_frame().expect("frame after close");

        let session_count_after = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions.len()
        });
        assert_eq!(session_count_after, 1);
    }

    #[test]
    fn dragging_split_terminal_divider_resizes_without_panic() {
        let mut cx = TestAppContext::with_headless_renderer_allowing_software(true)
            .expect("headless renderer");
        let (mut cx, app_state) = open_app(&mut cx);
        cx.draw_frame().expect("initial frame");

        // Split vertical
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_session(1, gpui::Axis::Horizontal, window, cx);
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
            state.split_session(1, gpui::Axis::Horizontal, window, cx);
        });
        cx.draw_frame().expect("frame after vertical split");

        // Split 1 horizontally: Top-left: 1, Bottom-left: new (3)
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_session(1, gpui::Axis::Vertical, window, cx);
        });
        cx.run_until_parked();
        cx.draw_frame().expect("frame after horizontal split");

        // Check terminal dimensions/bounds for all 3 sessions
        let (term1, term2, term3) = app_state.read_with(&cx, |state, cx| {
            let project = state.active_project().unwrap();
            let t1 = project.sessions[0].terminal.read(cx).dimensions();
            let t2 = project.sessions[1].terminal.read(cx).dimensions();
            let t3 = project.sessions[2].terminal.read(cx).dimensions();
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
            state.split_session(1, gpui::Axis::Vertical, window, cx);
        });
        cx.draw_frame().expect("frame after horizontal split");

        let bottom_id = app_state.read_with(&cx, |state, _| {
            state.active_project().unwrap().sessions.last().unwrap().id
        });

        // Split bottom vertically: Bottom-left: 2, Bottom-right: 3
        cx.update_window_entity(&app_state, |state, window, cx| {
            state.split_session(bottom_id, gpui::Axis::Horizontal, window, cx);
        });
        cx.run_until_parked();
        cx.draw_frame().expect("frame after vertical split");

        let (term1, term2, term3) = app_state.read_with(&cx, |state, cx| {
            let project = state.active_project().unwrap();
            let t1 = project.sessions[0].terminal.read(cx).dimensions();
            let t2 = project.sessions[1].terminal.read(cx).dimensions();
            let t3 = project.sessions[2].terminal.read(cx).dimensions();
            (t1, t2, t3)
        });

        assert!(term1.1 > 5, "term1 collapsed: {:?}", term1);
        assert!(term2.1 > 5, "term2 collapsed: {:?}", term2);
        assert!(term3.1 > 5, "term3 collapsed: {:?}", term3);
    }
}
