use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use anyhow::Result;
use gpui::{
    px, size, App, AppContext, Application, Bounds, Context, CursorStyle, Edges, Entity,
    InteractiveElement as _, IntoElement, KeyDownEvent, MouseButton, ParentElement, Render,
    SharedString, Styled, Window, WindowBounds, WindowDecorations,
    WindowOptions, div, rgba,
};
use gpui_component::resizable::{h_resizable, resizable_panel};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::{Root, TitleBar};
use gpui_terminal::{ColorPalette, TerminalConfig, TerminalView};
use parking_lot::Mutex;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde_json::Value;

mod agent_rules;
mod assets;
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
    pub terminal: Entity<TerminalView>,
    pub status: SessionStatus,
    pub pid: Option<u32>,
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
}

/// How long an unacknowledged Done badge lingers before it falls back to Idle.
const DONE_LINGER: std::time::Duration = std::time::Duration::from_secs(30);
pub struct Project {
    pub id: usize,
    pub name: String,
    pub path: PathBuf,
    pub sessions: Vec<Session>,
    pub active_session_idx: usize,
}

struct AppState {
    projects: Vec<Project>,
    active_project_idx: usize,
    next_id: usize,
    palette: ColorPalette,
    /// Session that currently owns keyboard focus, tracked every render frame
    /// and consumed by the status poller (Active state, sticky-state clearing).
    focused_session_id: Option<usize>,
}

impl AppState {
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
        for project in &mut self.projects {
            for session in &mut project.sessions {
                if session.id == session_id {
                    let next = SharedString::from(trimmed.to_string());
                    if session.title != next {
                        session.title = next;
                        cx.notify();
                    }
                    return;
                }
            }
        }
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

                let (next, ack, screen_fp) = {
                    let terminal = session.terminal.read(cx);
                    // `bottom_lines` takes the last N *rows*, clamped to the
                    // visible screen. Ask for the whole screen: a fresh shell
                    // prints its prompt on row 0, well above a 12-row window
                    // at the bottom. The rules skip blank rows themselves.
                    let bottom_lines = terminal.bottom_lines(usize::MAX);
                    let title = terminal.title().unwrap_or("");
                    let screen_fp = screen_fingerprint(&bottom_lines, title);

                    // Looking at a finished/failed terminal acknowledges it:
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

                    // Exit of the PTY child wins: the real exit code decides
                    // Done (0) vs Error (non-zero / signal), herdr-style.
                    let next = if terminal.has_exited() {
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
                            lines: &bottom_lines,
                            title: Some(title),
                            foreground: &foreground,
                        };
                        let detected = status::evaluate(&input, session.status, focused);
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
                        let rule = status::explain(&StatusInput {
                            lines: &bottom_lines,
                            title: Some(title),
                            foreground: &foreground,
                        })
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
                             prev={} next={} rule={rule}",
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
            let session = Session {
                id: next_id,
                title: SharedString::from(format!("terminal {}", session_count + 1)),
                terminal,
                status: SessionStatus::Idle,
                pid,
                ack_screen: None,
                agent_active: false,
                done_since: None,
            };

            if let Some(project) = self.active_project_mut() {
                project.sessions.push(session);
                project.active_session_idx = project.sessions.len() - 1;
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
                    terminal,
                    status: SessionStatus::Idle,
                    pid,
                    ack_screen: None,
                    agent_active: false,
                    done_since: None,
                };
                let project = Project {
                    id: next_id,
                    name,
                    path: folder,
                    sessions: vec![session],
                    active_session_idx: 0,
                };

                self.projects.push(project);
                self.active_project_idx = self.projects.len() - 1;
                cx.notify();
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
        self.focused_session_id = self.active_session().and_then(|s| {
            let focused = s.terminal.read(cx).focus_handle().is_focused(_window);
            focused.then_some(s.id)
        });

        let active_title = self
            .active_session()
            .map(|s| s.title.clone())
            .unwrap_or_else(|| SharedString::from("Terminal"));

        let active_project_name = self
            .active_project()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "No Project".to_string());

        let active_session_view = self.active_session().map(|s| s.terminal.clone());

        // Focused terminal's status, mirrored in the title bar.
        let titlebar_status = self.active_session().map(|s| s.status);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgba(0x0f1115ff))
            .text_color(rgba(0xe6edf3ff))
            .on_key_down(cx.listener(Self::on_key_down))
            .child(
                TitleBar::new().child(
                    div()
                        .w_full()
                        .h_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .gap_2()
                        .text_size(px(12.5))
                        .text_color(rgba(0x8b949eff))
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
                                        .w(px(6.0))
                                        .h(px(6.0))
                                        .rounded_full()
                                        .bg(rgba(dot_color)),
                                )
                                .child(status.label())
                        })),
                ),
            )
            .child(
                h_resizable("workspace-resizable-layout")
                    .child(
                        resizable_panel()
                            .size(px(260.0))
                            .size_range(px(180.0)..px(600.0))
                            .child(
                                div()
                                    .size_full()
                                    .flex()
                                    .flex_col()
                                    .bg(rgba(0x090a0dff))
                                    .border_r_1()
                                    .border_color(rgba(0x1e2229ff))
                            // Sidebar Header
                            .child(
                                div()
                                    .p_3()
                                    .flex()
                                    .flex_col()
                                    .gap_2()
                                    .border_b_1()
                                    .border_color(rgba(0x1e2229ff))
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .justify_between()
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(rgba(0x58a6ffff))
                                                    .child("T3 Code"),
                                            )
                                            .child(
                                                div()
                                                    .id("open-folder-btn")
                                                    .px_2()
                                                    .py_0p5()
                                                    .rounded_md()
                                                    .bg(rgba(0x1f242cff))
                                                    .hover(|s| s.bg(rgba(0x2d333bff)))
                                                    .cursor(CursorStyle::PointingHand)
                                                    .text_xs()
                                                    .text_color(rgba(0xc9d1d9ff))
                                                    .child("+ Folder")
                                                    .on_mouse_down(
                                                        MouseButton::Left,
                                                        cx.listener(|this, _, window, cx| {
                                                            this.select_folder_and_add_project(window, cx);
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
                                    .p_2()
                                    .flex()
                                    .flex_col()
                                    .gap_2()
                                    .children(self.projects.iter().enumerate().map(|(p_idx, project)| {
                                        let is_active_project = p_idx == self.active_project_idx;
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_1()
                                            .child(
                                                div()
                                                    .id(("project-header", project.id))
                                                    .px_2()
                                                    .py_1()
                                                    .rounded_md()
                                                    .bg(if is_active_project {
                                                        rgba(0x161b22ff)
                                                    } else {
                                                        rgba(0x00000000)
                                                    })
                                                    .hover(|s| s.bg(rgba(0x161b2299)))
                                                    .cursor(CursorStyle::PointingHand)
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                                            .text_color(if is_active_project {
                                                                rgba(0x58a6ffff)
                                                            } else {
                                                                rgba(0x8b949eff)
                                                            })
                                                            .child(project.name.clone()),
                                                    )
                                                    .child(
                                                        div()
                                                            .id(("add-session-btn", project.id))
                                                            .text_xs()
                                                            .text_color(rgba(0x8b949eff))
                                                            .hover(|s| s.text_color(rgba(0xffffffff)))
                                                            .child("+")
                                                            .on_mouse_down(
                                                                MouseButton::Left,
                                                                cx.listener(move |this, _, window, cx| {
                                                                    this.active_project_idx = p_idx;
                                                                    this.create_new_session_for_active_project(window, cx);
                                                                }),
                                                            ),
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
                                            // Sessions list for this project
                                            .children(project.sessions.iter().enumerate().map(|(s_idx, session)| {
                                                let is_active_session = is_active_project && s_idx == project.active_session_idx;
                                                let (dot_color, text_color, badge_bg) = session.status.colors();
                                                div()
                                                    .id(("session-item", session.id))
                                                    .ml_3()
                                                    .px_2()
                                                    .py_1()
                                                    .rounded_md()
                                                    .bg(if is_active_session {
                                                        rgba(0x21262dff)
                                                    } else {
                                                        rgba(0x00000000)
                                                    })
                                                    .hover(|s| s.bg(rgba(0x161b22ff)))
                                                    .cursor(CursorStyle::PointingHand)
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .items_center()
                                                            .gap_2()
                                                            .child(
                                                                // Status indicator dot
                                                                div()
                                                                    .w(px(6.0))
                                                                    .h(px(6.0))
                                                                    .rounded_full()
                                                                    .bg(rgba(dot_color))
                                                            )
                                                            .child(
                                                                div()
                                                                    .text_xs()
                                                                    .text_color(if is_active_session {
                                                                        rgba(0xf0f6fcff)
                                                                    } else {
                                                                        rgba(0x8b949eff)
                                                                    })
                                                                    .child(session.title.clone()),
                                                            ),
                                                    )
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .px_1p5()
                                                            .py_0p5()
                                                            .rounded_sm()
                                                            .bg(rgba(badge_bg))
                                                            .text_color(rgba(text_color))
                                                            .child(session.status.label()),
                                                    )
                                                    .on_mouse_down(
                                                        MouseButton::Left,
                                                        cx.listener(move |this, _, window, cx| {
                                                            this.active_project_idx = p_idx;
                                                            if let Some(p) = this.projects.get_mut(p_idx) {
                                                                p.active_session_idx = s_idx;
                                                            }
                                                            if let Some(session) = this.active_session() {
                                                                session.terminal.read(cx).focus_handle().focus(window);
                                                            }
                                                            cx.notify();
                                                        }),
                                                    )
                                            }))
                                    })),
                            ),
                            )
                    )
                    .child(
                        resizable_panel().child(
                            div()
                                .flex_1()
                                .size_full()
                                .bg(rgba(0x010409ff))
                                .overflow_hidden()
                                .children(active_session_view),
                        )
                    ),
            )
    }
}

/// How many non-empty screen lines `T3_STATUS_DEBUG` prints per terminal.
const DEBUG_SCREEN_LINES: usize = 14;

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
                            terminal: initial_terminal,
                            status: SessionStatus::Idle,
                            pid,
                            ack_screen: None,
                            agent_active: false,
                            done_since: None,
                        };

                        let initial_project = Project {
                            id: 1,
                            name: current_name,
                            path: current_dir_clone,
                            sessions: vec![initial_session],
                            active_session_idx: 0,
                        };

                        AppState {
                            projects: vec![initial_project],
                            active_project_idx: 0,
                            next_id: 2,
                            palette: palette_clone,
                            focused_session_id: None,
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

                    cx.new(|cx| Root::new(app_state, window, cx))
                },
            )
            .expect("Failed to open window");

            cx.activate(true);
        });

    Ok(())
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
