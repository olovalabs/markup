//! T3 Code — a GPUI terminal workspace for AI agent CLIs.
//!
//! The window is two panes: the workspace sidebar (projects → terminals, each
//! with a live status badge) and the active terminal. The sidebar is the part
//! that has to answer "which of my agents needs me?" at a glance, so the
//! status detection in `src/status.rs` feeds it a single, exact verdict per
//! terminal and everything visual here is derived from that verdict.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use gpui::prelude::FluentBuilder as _;
use gpui::{
    div, px, rgba, size, svg, AnyElement, App, AppContext, Application, Bounds, Context,
    CursorStyle, Div, Edges, ElementId, Entity, InteractiveElement as _, IntoElement, KeyDownEvent,
    MouseButton, MouseDownEvent, ParentElement, Render, SharedString, Stateful, Styled, Window,
    WindowBounds, WindowDecorations, WindowOptions,
};
use gpui_component::resizable::{h_resizable, resizable_panel};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::tooltip::Tooltip;
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
use status::{ExitInfo, SessionStatus, StatusInput};

/// Every colour the chrome uses, in one place: the palette is GitHub Dark, and
/// keeping the hex values out of the layout code makes the sidebar reviewable.
mod ui {
    /// Window background, behind everything.
    pub const BG_WINDOW: u32 = 0x0f1115ff;
    /// Sidebar background — a shade darker than the window, so the pane edge
    /// reads as a real edge without a heavy border.
    pub const BG_SIDEBAR: u32 = 0x090b0fff;
    /// Terminal background.
    pub const BG_TERMINAL: u32 = 0x010409ff;
    /// Pane border.
    pub const BORDER: u32 = 0x1e2229ff;
    /// Divider inside a pane (header/footer rules).
    pub const BORDER_SOFT: u32 = 0x16191fff;
    /// Row background for the selected session.
    pub const SELECTED: u32 = 0x1c2129ff;
    /// Row background on hover.
    pub const HOVER: u32 = 0x171c23ff;
    /// Slightly stronger hover, for icon buttons.
    pub const HOVER_STRONG: u32 = 0x21262dff;
    /// Background of an active project header.
    pub const SURFACE: u32 = 0x12161cff;
    pub const TEXT_BRIGHT: u32 = 0xf0f6fcff;
    pub const TEXT: u32 = 0xb9c1cbff;
    pub const TEXT_MUTED: u32 = 0x8b949eff;
    pub const TEXT_FAINT: u32 = 0x6e7681ff;
    pub const TEXT_DIM: u32 = 0x545d68ff;
    pub const ACCENT: u32 = 0x58a6ffff;
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

/// Everything a spawn hands back to the session it belongs to.
struct SpawnedTerminal {
    view: Entity<TerminalView>,
    pid: Option<u32>,
    /// Filled in by the waiter thread when the shell really exits; the status
    /// poller reads it to tell `exited` from `error` from `killed`.
    exit: Arc<Mutex<Option<ExitInfo>>>,
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
    /// When `status` last changed, so the sidebar can say "working · 2m".
    pub status_since: Instant,
    /// The elapsed-time label that was last rendered. The poller keeps it in
    /// sync with what the sidebar would draw, so a clock that is not on screen
    /// never causes a repaint.
    pub elapsed_label: SharedString,
    /// Plain-language evidence for the current status, shown in its tooltip.
    pub status_note: SharedString,
    /// Real exit status of the shell, recorded by the waiter thread.
    pub shell_exit: Arc<Mutex<Option<ExitInfo>>>,
    pub pid: Option<u32>,
    /// Reused scrape buffer: the poller reads the screen into this instead of
    /// allocating a fresh `Vec<String>` every 500 ms per session.
    pub screen_scratch: Vec<String>,
    /// (screen fingerprint, focused, exited, foreground fingerprint) from the
    /// last poll tick. When it matches and the badge has no running timer, the
    /// whole rule-pack evaluation is skipped.
    pub poll_cache: Option<(u64, bool, bool, u64)>,
    /// Screen fingerprint at the moment the user last looked at a finished or
    /// failed terminal. While the screen still matches, the sticky badge stays
    /// acknowledged (Idle); any new output invalidates it and lets a fresh
    /// Done/Error/Killed be raised.
    pub ack_screen: Option<u64>,
    /// A recognized agent CLI was the foreground process on the previous tick.
    /// Its disappearance means the agent finished (Done), like herdr.
    pub agent_active: bool,
    /// Last time the screen, title, or progress report changed — the stall
    /// watchdog's clock.
    pub last_output_at: Instant,
    /// Screen fingerprint from the previous tick, used to detect new output.
    pub last_screen_fp: u64,
    /// When the terminal first showed an unacknowledged Done.
    pub done_since: Option<Instant>,
    /// Path of the AI agent / command logo shown in the sidebar.
    pub logo: SharedString,
}

impl Session {
    /// A freshly spawned session: idle, with no detection history yet.
    fn new(id: usize, title: SharedString, spawned: SpawnedTerminal) -> Self {
        let now = Instant::now();
        Self {
            id,
            display_title: truncate_title(&title),
            title,
            terminal: spawned.view,
            status: SessionStatus::Idle,
            status_since: now,
            elapsed_label: SharedString::from("now"),
            status_note: SharedString::from("new terminal"),
            shell_exit: spawned.exit,
            pid: spawned.pid,
            screen_scratch: Vec::new(),
            poll_cache: None,
            ack_screen: None,
            agent_active: false,
            last_output_at: now,
            last_screen_fp: 0,
            done_since: None,
            logo: logos::default_logo(),
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
}

impl Project {
    /// The loudest status in the project, for the project row's dot.
    fn rollup(&self) -> Option<SessionStatus> {
        self.sessions
            .iter()
            .map(|session| session.status)
            .max_by_key(|status| status.rank())
    }

    /// How many of its terminals want the user (blocked, done, failed, stuck).
    fn attention_count(&self) -> usize {
        self.sessions
            .iter()
            .filter(|session| session.status.needs_attention())
            .count()
    }
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
    /// Ids of projects whose terminal lists are folded away.
    collapsed_projects: HashSet<usize>,
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
    ) -> Result<SpawnedTerminal> {
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

        // The shell's *real* exit status is the only trustworthy
        // done/failed/killed signal a terminal has, and only the process handle
        // can report it. `wait` blocks, so it runs on its own thread and the
        // result is published into a slot the poller reads.
        let exit: Arc<Mutex<Option<ExitInfo>>> = Arc::new(Mutex::new(None));
        {
            let mut child = child;
            let exit_slot = Arc::clone(&exit);
            std::thread::spawn(move || {
                if let Ok(status) = child.wait() {
                    // portable-pty keeps the signal name inside its `Display`
                    // form ("Terminated by SIGKILL", "Exited with code 1",
                    // "Success"); the numeric code alone cannot distinguish a
                    // signal death from an ordinary exit code.
                    let text = status.to_string();
                    let signal = text
                        .strip_prefix("Terminated by ")
                        .map(|signal| signal.trim().to_string());
                    *exit_slot.lock() = Some(ExitInfo::new(status.exit_code() as i32, signal));
                }
            });
        }

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
            // derives exited/error/killed from the real exit code recorded by
            // the waiter thread above.
        });

        Ok(SpawnedTerminal {
            view: terminal,
            pid,
            exit,
        })
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

    /// Rebuilds the id → (project, session) index. Cheap (a handful of
    /// entries) and far less error-prone than patching it in place when
    /// sessions and projects are removed.
    fn reindex_sessions(&mut self) {
        self.session_index.clear();
        for (p_idx, project) in self.projects.iter().enumerate() {
            for (s_idx, session) in project.sessions.iter().enumerate() {
                self.session_index.insert(session.id, (p_idx, s_idx));
            }
        }
    }

    /// Periodic detection tick, herdr-style: screen scrape + OSC title + OSC
    /// progress + foreground process tree, with sticky terminal states, a stall
    /// watchdog and focus resolution.
    pub fn poll_sessions_status(&mut self, cx: &mut Context<Self>) {
        let mut changed = false;
        let focused_id = self.focused_session_id;
        // `T3_STATUS_DEBUG=1` dumps every detection pass: the facts that went in
        // and the badge that came out. Without this the poller is invisible.
        let debug = status::debug_enabled();
        let now = Instant::now();

        for project in &mut self.projects {
            for session in &mut project.sessions {
                let focused = focused_id == Some(session.id);
                // The badge this tick started with; both the transition rules
                // and the debug dump need it after `session.status` is updated.
                let prev_status = session.status;

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

                // Scrape into the session's reused buffer. `bottom_lines_into`
                // takes the last N *rows*, clamped to the visible screen. Ask
                // for the whole screen: a fresh shell prints its prompt on row
                // 0, well above a 12-row window at the bottom. The rules skip
                // blank rows themselves.
                session
                    .terminal
                    .read(cx)
                    .bottom_lines_into(usize::MAX, &mut session.screen_scratch);

                // Title, OSC 9;4 progress and exit state, read through one
                // borrow of the terminal entity.
                let (title, progress, exited, exit_status) = {
                    let terminal = session.terminal.read(cx);
                    (
                        terminal.title().unwrap_or("").to_string(),
                        terminal.progress().map(str::to_string),
                        terminal.has_exited(),
                        terminal.exit_status(),
                    )
                };

                // Everything an agent can change without starting or finishing
                // a turn: the grid, the title, and its progress report.
                let screen_fp =
                    screen_fingerprint(&session.screen_scratch, &title, progress.as_deref());
                let fg_fp = screen_fingerprint(&foreground, "", None);

                // Any of those changing counts as output, which is what the
                // stall watchdog measures silence against.
                if screen_fp != session.last_screen_fp {
                    session.last_screen_fp = screen_fp;
                    session.last_output_at = now;
                }

                let poll_key = (screen_fp, focused, exited, fg_fp);
                let idle_like = matches!(
                    session.status,
                    SessionStatus::Active | SessionStatus::Idle
                );
                // Fast path: nothing changed and the badge has no timer running,
                // so the verdict cannot differ from last tick's.
                if idle_like && session.poll_cache == Some(poll_key) {
                    continue;
                }

                let agent_now = foreground.iter().any(|name| status::is_agent_process(name));
                let shell_exit = session.shell_exit.lock().clone();

                // Looking at a finished/failed terminal acknowledges it: while
                // the screen is unchanged the sticky badge stays cleared; new
                // output re-arms it.
                let ack = if focused && (exited || session.status.is_final()) {
                    Some(screen_fp)
                } else {
                    session.ack_screen
                };
                let acknowledged = ack == Some(screen_fp);

                // The agent rule match behind this tick's verdict, for the
                // tooltip and the debug dump.
                let mut rule = None;
                let mut reason = None;

                let mut next = if exited {
                    // The PTY child is gone, so the terminal is closed: say so
                    // and keep saying it. There is no prompt to return to, so
                    // there is nothing for the user to "acknowledge" away — the
                    // badge is the shell's real exit status, which is also the
                    // only thing left to know about this terminal.
                    match &shell_exit {
                        Some(info) => SessionStatus::from_exit(info),
                        // The PTY closed before the waiter thread could record
                        // a status; it ended either way.
                        None => SessionStatus::Exited,
                    }
                } else {
                    let input = StatusInput {
                        lines: &session.screen_scratch,
                        title: Some(&title),
                        progress: progress.as_deref(),
                        foreground: &foreground,
                    };
                    let verdict = status::verdict(&input, session.status, focused);
                    rule = verdict.rule;
                    reason = verdict.reason;
                    if acknowledged && verdict.status.is_final() {
                        if focused {
                            SessionStatus::Active
                        } else {
                            SessionStatus::Idle
                        }
                    } else {
                        verdict.status
                    }
                };

                // Stall watchdog: a recognized agent that has printed nothing
                // at all for a minute is not "working", it is stuck.
                if next == SessionStatus::Working
                    && agent_now
                    && now.saturating_duration_since(session.last_output_at) >= status::STALL_AFTER
                {
                    next = SessionStatus::Stalled;
                }

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
                let agent_exited = session.agent_active && !agent_now;
                session.agent_active = agent_now;

                let turn_finished = matches!(
                    session.status,
                    SessionStatus::Working | SessionStatus::Stalled | SessionStatus::Blocked
                ) && next == SessionStatus::Idle;

                let just_finished = agent_exited || turn_finished;
                let next = if just_finished && next == SessionStatus::Idle && !acknowledged && !focused
                {
                    SessionStatus::Done
                } else {
                    next
                };

                // Spec: Done also drops back to Idle after a short timeout even
                // if the user never focuses the terminal.
                let next = if next == SessionStatus::Done {
                    let since = *session.done_since.get_or_insert(now);
                    if now.saturating_duration_since(since) >= status::DONE_LINGER {
                        session.done_since = None;
                        SessionStatus::Idle
                    } else {
                        next
                    }
                } else {
                    session.done_since = None;
                    next
                };

                // The evidence the sidebar tooltip shows, most specific first.
                let note: SharedString = if next == SessionStatus::Stalled {
                    SharedString::from(format!(
                        "no output for {}",
                        format_elapsed(now.saturating_duration_since(session.last_output_at))
                    ))
                } else if next == SessionStatus::Done {
                    SharedString::from("turn finished")
                } else if let Some(rule) = &rule {
                    SharedString::from(format!("{} · {}", rule.pack, rule.rule))
                } else if let Some(info) = &shell_exit {
                    SharedString::from(info.note())
                } else if let Some(reason) = reason {
                    SharedString::from(reason)
                } else {
                    SharedString::from("")
                };

                session.ack_screen = ack;
                if session.status != next {
                    session.status = next;
                    session.status_since = now;
                    changed = true;
                }

                // Keep the rendered elapsed label in sync. Comparing the
                // formatted string means the window repaints when a timer on
                // screen would change, and never otherwise.
                let elapsed =
                    format_elapsed(now.saturating_duration_since(session.status_since));
                if session.elapsed_label != elapsed {
                    session.elapsed_label = elapsed;
                    changed = true;
                }
                if session.status_note != note {
                    session.status_note = note;
                    changed = true;
                }

                // `map(str::to_string)`: the title borrows the terminal, and
                // the logo resolution below needs the session mutably again.
                let term_title = session.terminal.read(cx).title().map(str::to_string);
                let next_logo = logos::resolve_session_logo(
                    &foreground,
                    term_title.as_deref(),
                    session.status,
                    &session.logo,
                );
                if session.logo != next_logo {
                    session.logo = next_logo;
                    changed = true;
                }

                session.poll_cache = Some(poll_key);

                if debug {
                    let rule_text = rule
                        .as_ref()
                        .map_or_else(
                            || match reason {
                                Some(reason) => format!("(fallback: {reason})"),
                                None => "(no rule)".to_string(),
                            },
                            |matched| {
                                format!(
                                    "{}/{} v{} state={} priority={} region={} \
                                     visible(working/blocker/idle)={}/{}/{} skip_update={}",
                                    matched.pack,
                                    matched.rule,
                                    matched.pack_version.as_deref().unwrap_or("?"),
                                    matched.state.label(),
                                    matched.priority,
                                    matched.region,
                                    matched.visible_working,
                                    matched.visible_blocker,
                                    matched.visible_idle,
                                    matched.skip_state_update,
                                )
                            },
                        );
                    eprintln!(
                        "[status] session={} focused={focused} exited={} exit={:?} shell_exit={:?} \
                         pid={:?} fg={foreground:?} pack={:?} progress={progress:?} title={title:?} \
                         silence={:?} prev={} next={} note={:?} rule={rule_text}",
                        session.id,
                        exited,
                        exit_status,
                        shell_exit.as_ref().map(ExitInfo::note),
                        session.pid,
                        status::pack_for(&foreground),
                        now.saturating_duration_since(session.last_output_at),
                        prev_status.label(),
                        session.status.label(),
                        session.status_note.as_ref(),
                    );
                    let screen: Vec<&str> = session
                        .screen_scratch
                        .iter()
                        .map(String::as_str)
                        .filter(|line| !line.trim().is_empty())
                        .collect();
                    let from = screen.len().saturating_sub(DEBUG_SCREEN_LINES);
                    for line in &screen[from..] {
                        eprintln!("[status]   | {line}");
                    }
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
        if let Ok(spawned) =
            Self::spawn_terminal_view(dir.as_ref(), self.palette.clone(), self_weak, next_id, cx)
        {
            spawned.view.read(cx).focus_handle().focus(window);
            let session_count = self
                .active_project()
                .map(|p| p.sessions.len())
                .unwrap_or(0);
            let title = SharedString::from(format!("terminal {}", session_count + 1));
            let session = Session::new(next_id, title, spawned);

            let p_idx = self.active_project_idx;
            let pushed = if let Some(project) = self.active_project_mut() {
                project.sessions.push(session);
                project.active_session_idx = project.sessions.len() - 1;
                Some(project.active_session_idx)
            } else {
                None
            };
            if let Some(s_idx) = pushed {
                self.session_index.insert(next_id, (p_idx, s_idx));
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

            if let Ok(spawned) = Self::spawn_terminal_view(
                Some(&folder),
                self.palette.clone(),
                self_weak,
                next_id,
                cx,
            ) {
                spawned.view.read(cx).focus_handle().focus(window);
                let session = Session::new(next_id, SharedString::from("terminal 1"), spawned);
                let project = Project {
                    id: next_id,
                    name: SharedString::from(name),
                    path: folder,
                    sessions: vec![session],
                    active_session_idx: 0,
                };

                self.projects.push(project);
                self.active_project_idx = self.projects.len() - 1;
                let p_idx = self.active_project_idx;
                self.session_index.insert(next_id, (p_idx, 0));
                cx.notify();
            }
        }
    }

    /// Switch projects and put the keyboard back into that project's terminal.
    fn activate_project(&mut self, p_idx: usize, window: &mut Window, cx: &mut Context<Self>) {
        if p_idx >= self.projects.len() {
            return;
        }
        self.active_project_idx = p_idx;
        // Clicking a folded project opens it as well as selecting it: the click
        // is the user asking to see what is inside.
        let id = self.projects[p_idx].id;
        self.collapsed_projects.remove(&id);
        if let Some(session) = self.active_session() {
            session.terminal.read(cx).focus_handle().focus(window);
        }
        cx.notify();
    }

    fn toggle_project_collapsed(&mut self, p_idx: usize, cx: &mut Context<Self>) {
        let Some(project) = self.projects.get(p_idx) else {
            return;
        };
        if !self.collapsed_projects.remove(&project.id) {
            self.collapsed_projects.insert(project.id);
        }
        cx.notify();
    }

    fn activate_session(
        &mut self,
        p_idx: usize,
        s_idx: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.active_project_idx = p_idx;
        if let Some(project) = self.projects.get_mut(p_idx) {
            if s_idx < project.sessions.len() {
                project.active_session_idx = s_idx;
            }
        }
        if let Some(session) = self.active_session() {
            session.terminal.read(cx).focus_handle().focus(window);
        }
        cx.notify();
    }

    /// Close one terminal. A project's *last* terminal closes the project; the
    /// window always keeps at least one project.
    fn close_session(
        &mut self,
        p_idx: usize,
        s_idx: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // A project's last terminal *is* the project: closing it removes the
        // workspace entry, so that falls through to `remove_project`.
        let is_last_terminal = self
            .projects
            .get(p_idx)
            .is_none_or(|project| project.sessions.len() <= 1);
        if is_last_terminal {
            self.remove_project(p_idx, window, cx);
            return;
        }
        let Some(project) = self.projects.get_mut(p_idx) else {
            return;
        };

        let closing = project.sessions.remove(s_idx);
        hangup_process_group(closing.pid);
        let was_focused = self.focused_session_id == Some(closing.id);
        drop(closing);

        if project.active_session_idx >= project.sessions.len() {
            project.active_session_idx = project.sessions.len() - 1;
        } else if s_idx < project.active_session_idx {
            project.active_session_idx -= 1;
        }

        self.reindex_sessions();
        if was_focused {
            if let Some(session) = self.active_session() {
                session.terminal.read(cx).focus_handle().focus(window);
            }
        }
        cx.notify();
    }

    fn remove_project(&mut self, p_idx: usize, window: &mut Window, cx: &mut Context<Self>) {
        // Never leave the window without a workspace to show.
        if self.projects.len() <= 1 || p_idx >= self.projects.len() {
            return;
        }
        let removed = self.projects.remove(p_idx);
        self.collapsed_projects.remove(&removed.id);
        for session in &removed.sessions {
            hangup_process_group(session.pid);
        }
        drop(removed);

        if self.active_project_idx >= self.projects.len() {
            self.active_project_idx = self.projects.len() - 1;
        } else if p_idx < self.active_project_idx {
            self.active_project_idx -= 1;
        }

        self.reindex_sessions();
        if let Some(session) = self.active_session() {
            session.terminal.read(cx).focus_handle().focus(window);
        }
        cx.notify();
    }

    /// Per-status terminal counts across every project, loudest first — the
    /// sidebar footer's summary line.
    fn status_summary(&self) -> Vec<(SessionStatus, usize)> {
        let loudest_first = [
            SessionStatus::Blocked,
            SessionStatus::Error,
            SessionStatus::Killed,
            SessionStatus::Stalled,
            SessionStatus::Done,
            SessionStatus::Working,
            SessionStatus::Waiting,
        ];
        let sessions = || {
            self.projects
                .iter()
                .flat_map(|project| project.sessions.iter())
        };
        loudest_first
            .iter()
            .map(|&status| {
                (
                    status,
                    sessions().filter(|session| session.status == status).count(),
                )
            })
            .filter(|(_, count)| *count > 0)
            .collect()
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

    // -----------------------------------------------------------------------
    // Sidebar rendering
    //
    // These return concrete element types (`Div`, `Stateful<Div>`) rather than
    // `impl IntoElement`: the sidebar mixes rows of different shapes, and a
    // concrete type keeps them collectable into one list.
    // -----------------------------------------------------------------------

    fn render_sidebar(&self, cx: &mut Context<Self>) -> Div {
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgba(ui::BG_SIDEBAR))
            .border_r_1()
            .border_color(rgba(ui::BORDER))
            .child(self.render_sidebar_header(cx))
            .child(
                div()
                    .id("sidebar-scroll")
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .px_2()
                    .py_2()
                    .overflow_y_scrollbar()
                    .children(self.sidebar_rows(cx)),
            )
            .child(self.render_sidebar_footer())
    }

    fn render_sidebar_header(&self, cx: &mut Context<Self>) -> Div {
        div()
            .flex_shrink_0()
            .px_3()
            .py_2()
            .flex()
            .items_center()
            .justify_between()
            .border_b_1()
            .border_color(rgba(ui::BORDER_SOFT))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(7.0)).h(px(7.0)).rounded_full().bg(rgba(ui::ACCENT)))
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgba(ui::TEXT_BRIGHT))
                            .child("T3 Code"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgba(ui::TEXT_DIM))
                            .child(format!(
                                "{} workspace{}",
                                self.projects.len(),
                                if self.projects.len() == 1 { "" } else { "s" }
                            )),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(icon_button(
                        "sidebar-new-terminal",
                        "ui_icons/add.svg",
                        "New terminal in the active project",
                        13.0,
                        cx.listener(|this, _, window, cx| {
                            let p_idx = this.active_project_idx;
                            this.activate_project(p_idx, window, cx);
                            this.create_new_session_for_active_project(window, cx);
                        }),
                    ))
                    .child(icon_button(
                        "sidebar-add-project",
                        "ui_icons/folder-opened.svg",
                        "Open a folder as a new project",
                        13.0,
                        cx.listener(|this, _, window, cx| {
                            this.select_folder_and_add_project(window, cx)
                        }),
                    )),
            )
    }

    /// All sidebar rows, in order: project header followed by its terminals.
    fn sidebar_rows(&self, cx: &mut Context<Self>) -> Vec<Stateful<Div>> {
        let mut rows = Vec::with_capacity(self.projects.len() * 2);
        for p_idx in 0..self.projects.len() {
            rows.push(self.project_row(p_idx, cx));
            let collapsed = self.collapsed_projects.contains(&self.projects[p_idx].id);
            if collapsed {
                continue;
            }
            for s_idx in 0..self.projects[p_idx].sessions.len() {
                rows.push(self.session_row(p_idx, s_idx, cx));
            }
        }
        rows
    }

    /// A project header: fold arrow, folder, name, rollup dot, terminal count,
    /// and a hover-revealed "new terminal" button.
    fn project_row(&self, p_idx: usize, cx: &mut Context<Self>) -> Stateful<Div> {
        let project = &self.projects[p_idx];
        let is_active = p_idx == self.active_project_idx;
        let collapsed = self.collapsed_projects.contains(&project.id);
        let attention = project.attention_count();
        // Only show a dot when the project has something to say: a quiet
        // project should not look busy.
        let rollup = project.rollup().filter(|_| attention > 0);
        let sessions = project.sessions.len();

        div()
            .id(("project", project.id))
            .group("project")
            .px_2()
            .py_1()
            .rounded_md()
            .bg(if is_active {
                rgba(ui::SURFACE)
            } else {
                rgba(0x00000000)
            })
            .hover(|s| s.bg(rgba(ui::HOVER)))
            .cursor(CursorStyle::PointingHand)
            .flex()
            .items_center()
            .gap_2()
            // Fold arrow. Its own hitbox and its own click handler, so folding
            // never changes which project is active.
            .child(
                div()
                    .id(("project-fold", project.id))
                    .size(px(14.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_sm()
                    .hover(|s| s.bg(rgba(ui::HOVER_STRONG)))
                    .child(
                        svg()
                            .path(if collapsed {
                                "ui_icons/chevron-right.svg"
                            } else {
                                "ui_icons/chevron-down.svg"
                            })
                            .size(px(10.0))
                            .text_color(rgba(ui::TEXT_MUTED)),
                    )
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            this.toggle_project_collapsed(p_idx, cx);
                            cx.stop_propagation();
                        }),
                    ),
            )
            .child(
                svg()
                    .path(if is_active {
                        "ui_icons/folder-opened.svg"
                    } else {
                        "ui_icons/folder.svg"
                    })
                    .size(px(13.0))
                    .text_color(rgba(if is_active { ui::ACCENT } else { ui::TEXT_MUTED })),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_xs()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(rgba(if is_active { ui::TEXT_BRIGHT } else { ui::TEXT }))
                    .child(project.name.clone()),
            )
            .when_some(rollup, |el, status| {
                el.child(div().flex_shrink_0().child(status_dot(status, 6.0)))
            })
            .child(
                div()
                    .flex_shrink_0()
                    .text_xs()
                    .text_color(rgba(if attention > 0 {
                        ui::TEXT_MUTED
                    } else {
                        ui::TEXT_DIM
                    }))
                    .child(format!("{sessions}")),
            )
            .child(
                div()
                    .id(("project-new-terminal", project.id))
                    .size(px(16.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_sm()
                    .invisible()
                    .group_hover("project", |s| s.visible())
                    .hover(|s| s.bg(rgba(ui::HOVER_STRONG)))
                    .cursor(CursorStyle::PointingHand)
                    .tooltip(|window, cx| {
                        Tooltip::new("New terminal in this project").build(window, cx)
                    })
                    .child(
                        svg()
                            .path("ui_icons/add.svg")
                            .size(px(11.0))
                            .text_color(rgba(ui::TEXT_MUTED)),
                    )
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, window, cx| {
                            this.activate_project(p_idx, window, cx);
                            this.create_new_session_for_active_project(window, cx);
                        }),
                    ),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, window, cx| {
                    this.activate_project(p_idx, window, cx);
                }),
            )
    }

    /// One terminal row: status rail, agent logo, title, and — once the status
    /// is worth reading — the status itself with how long it has been like
    /// that.
    fn session_row(&self, p_idx: usize, s_idx: usize, cx: &mut Context<Self>) -> Stateful<Div> {
        let project = &self.projects[p_idx];
        let session = &project.sessions[s_idx];
        let is_active = p_idx == self.active_project_idx && s_idx == project.active_session_idx;
        let (dot_color, status_color, _) = session.status.colors();
        // Idle/active rows stay one line tall; anything the user may need to
        // act on gets a second line with the reason and the clock.
        let show_detail = session.status.needs_attention() || session.status.is_busy();
        let tooltip: SharedString = if session.status_note.is_empty() {
            SharedString::from(session.status.description().to_string())
        } else {
            SharedString::from(format!(
                "{} {}",
                session.status.description(),
                session.status_note
            ))
        };

        div()
            .id(("session", session.id))
            .group("session")
            .relative()
            // Indented under the project it belongs to: the rail sits in the
            // gutter, the logo lines up with the project's icon.
            .pl_5()
            .pr_2()
            .py_1()
            .rounded_md()
            .cursor(CursorStyle::PointingHand)
            .bg(if is_active {
                rgba(ui::SELECTED)
            } else {
                rgba(0x00000000)
            })
            .hover(|s| s.bg(rgba(ui::HOVER)))
            .flex()
            .items_center()
            .gap_2()
            .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
            // The rail: a 2 px bar in the status colour. Scanning a column of
            // rails is faster than reading a column of words.
            .child(
                div()
                    .absolute()
                    .left_0()
                    .top_1()
                    .bottom_1()
                    .w(px(2.0))
                    .rounded_full()
                    .when(is_active, |el| el.bg(rgba(ui::ACCENT)))
                    .when(!is_active && session.status.needs_attention(), |el| {
                        el.bg(rgba(dot_color))
                    }),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .size(px(16.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(gpui::img(session.logo.clone()).size(px(15.0)).rounded(px(2.0))),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(
                        div()
                            .truncate()
                            .text_xs()
                            .text_color(rgba(if is_active { ui::TEXT_BRIGHT } else { ui::TEXT }))
                            .child(session.display_title.clone()),
                    )
                    .when(show_detail, |el| {
                        el.child(
                            div()
                                .flex()
                                .items_center()
                                .gap_1()
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgba(status_color))
                                        .child(session.status.label()),
                                )
                                .child(div().text_xs().text_color(rgba(ui::TEXT_DIM)).child("·"))
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgba(ui::TEXT_FAINT))
                                        .child(session.elapsed_label.clone()),
                                ),
                        )
                    }),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(status_dot(session.status, 6.0))
                    .child(
                        div()
                            .id(("session-close", session.id))
                            .size(px(16.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded_sm()
                            .invisible()
                            .group_hover("session", |s| s.visible())
                            .hover(|s| s.bg(rgba(ui::HOVER_STRONG)))
                            .cursor(CursorStyle::PointingHand)
                            .tooltip(|window, cx| {
                                Tooltip::new("Close terminal").build(window, cx)
                            })
                            .child(
                                svg()
                                    .path("ui_icons/panel_close.svg")
                                    .size(px(9.0))
                                    .text_color(rgba(ui::TEXT_MUTED)),
                            )
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, window, cx| {
                                    this.close_session(p_idx, s_idx, window, cx);
                                    cx.stop_propagation();
                                }),
                            ),
                    ),
            )
            .when(session.status.needs_attention(), |el| {
                el.bg(rgba(if is_active { ui::SELECTED } else { ui::HOVER }))
            })
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, window, cx| {
                    this.activate_session(p_idx, s_idx, window, cx);
                }),
            )
    }

    /// The footer: a live summary of every terminal in the workspace, so the
    /// window can be read without expanding anything.
    fn render_sidebar_footer(&self) -> Div {
        let summary = self.status_summary();
        let total: usize = self.projects.iter().map(|p| p.sessions.len()).sum();
        let quiet = total == 0 || summary.is_empty();

        div()
            .flex_shrink_0()
            .px_3()
            .py_2()
            .flex()
            .flex_col()
            .gap_1()
            .border_t_1()
            .border_color(rgba(ui::BORDER_SOFT))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .when(quiet, |el| {
                        el.child(
                            div()
                                .text_xs()
                                .text_color(rgba(ui::TEXT_DIM))
                                .child(format!("{total} idle")),
                        )
                    })
                    .children(summary.into_iter().map(|(status, count)| {
                        let (dot_color, text_color, badge_bg) = status.colors();
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .px_1p5()
                            .py_0p5()
                            .rounded_sm()
                            .bg(rgba(badge_bg))
                            .tooltip(move |window, cx| {
                                Tooltip::new(format!(
                                    "{count} {} — {}",
                                    status.label(),
                                    status.description()
                                ))
                                .build(window, cx)
                            })
                            .child(div().size(px(5.0)).rounded_full().bg(rgba(dot_color)))
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgba(text_color))
                                    .child(format!("{count} {}", status.label())),
                            )
                    })),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgba(ui::TEXT_DIM))
                    .child(format!(
                        "{} project{} · {} terminal{}",
                        self.projects.len(),
                        if self.projects.len() == 1 { "" } else { "s" },
                        total,
                        if total == 1 { "" } else { "s" },
                    )),
            )
    }

    fn render_titlebar(
        &self,
        active_project_name: &SharedString,
        active_title: &SharedString,
        titlebar_logo: Option<SharedString>,
        titlebar_status: Option<(SessionStatus, SharedString, SharedString)>,
    ) -> AnyElement {
        TitleBar::new()
            .child(
            div()
                .w_full()
                .h_full()
                .flex()
                .items_center()
                .justify_center()
                .gap_2()
                .text_size(px(12.5))
                .children(titlebar_logo.map(|logo| {
                    div()
                        .size(px(16.0))
                        .flex()
                        .items_center()
                        .justify_center()
                        .flex_shrink_0()
                        .child(gpui::img(logo).size(px(14.0)).rounded(px(2.0)))
                }))
                .child(
                    div()
                        .text_xs()
                        .text_color(rgba(ui::TEXT_MUTED))
                        .child(active_project_name.clone()),
                )
                .child(div().text_xs().text_color(rgba(ui::TEXT_DIM)).child("›"))
                .child(
                    div()
                        .max_w(px(420.0))
                        .truncate()
                        .text_xs()
                        .text_color(rgba(ui::TEXT))
                        .child(active_title.clone()),
                )
                .children(titlebar_status.map(|(status, elapsed, note)| {
                    status_chip(status, &elapsed, &note)
                })),
            )
            .into_any_element()
    }
}

impl Render for AppState {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Track which session owns keyboard focus each frame; the poller uses
        // this for Active-state resolution and for clearing sticky badges.
        self.focused_session_id = self.active_session().and_then(|s| {
            let focused = s.terminal.read(cx).focus_handle().is_focused(window);
            focused.then_some(s.id)
        });

        let active_title = self
            .active_session()
            .map(|s| s.title.clone())
            .unwrap_or_else(|| SharedString::from("Terminal"));

        let active_project_name = self
            .active_project()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| SharedString::from("No Project"));

        let active_session_view = self.active_session().map(|s| s.terminal.clone());

        // Focused terminal's badge and logo, mirrored in the title bar, plus
        // the evidence and clock the sidebar row shows.
        let titlebar = self.active_session().map(|s| {
            (
                s.status,
                s.elapsed_label.clone(),
                s.status_note.clone(),
                s.logo.clone(),
            )
        });

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgba(ui::BG_WINDOW))
            .text_color(rgba(ui::TEXT))
            .on_key_down(cx.listener(Self::on_key_down))
            .child(self.render_titlebar(
                &active_project_name,
                &active_title,
                titlebar.as_ref().map(|(_, _, _, logo)| logo.clone()),
                titlebar.as_ref().map(|(status, elapsed, note, _)| {
                    (*status, elapsed.clone(), note.clone())
                }),
            ))
            .child(
                h_resizable("workspace-resizable-layout")
                    .child(
                        resizable_panel()
                            .size(px(272.0))
                            .size_range(px(208.0)..px(640.0))
                            .child(self.render_sidebar(cx)),
                    )
                    .child(
                        resizable_panel().child(
                            div()
                                .flex_1()
                                .size_full()
                                .bg(rgba(ui::BG_TERMINAL))
                                .overflow_hidden()
                                .children(active_session_view),
                        ),
                    ),
            )
    }
}

/// How many non-empty screen lines `T3_STATUS_DEBUG` prints per terminal.
const DEBUG_SCREEN_LINES: usize = 14;

/// Sidebar/terminal titlebar button: a square hit target with a hover
/// background and a tooltip, so every icon in the chrome is self-explaining.
fn icon_button(
    id: impl Into<ElementId>,
    icon: &'static str,
    tooltip: &'static str,
    icon_size: f32,
    listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
) -> Stateful<Div> {
    div()
        .id(id)
        .size(px(icon_size + 9.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_md()
        .cursor(CursorStyle::PointingHand)
        .hover(|s| s.bg(rgba(ui::HOVER_STRONG)))
        .tooltip(move |window, cx| Tooltip::new(tooltip).build(window, cx))
        .child(
            svg()
                .path(icon)
                .size(px(icon_size))
                .text_color(rgba(ui::TEXT_MUTED)),
        )
        .on_mouse_down(MouseButton::Left, listener)
}

/// The status dot: filled when something is happening, hollow when the
/// terminal is quiet, ringed by its own badge tint when it wants attention.
fn status_dot(status: SessionStatus, diameter: f32) -> Div {
    let (dot_color, _, badge_bg) = status.colors();
    let hollow = matches!(
        status,
        SessionStatus::Idle | SessionStatus::Exited | SessionStatus::Waiting
    );
    div()
        .flex_shrink_0()
        .size(px(diameter + 7.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .when(status.needs_attention(), |el| el.bg(rgba(badge_bg)))
        .child(
            div()
                .size(px(diameter))
                .rounded_full()
                .when(hollow, |el| {
                    el.border_1().border_color(rgba(dot_color))
                })
                .when(!hollow, |el| el.bg(rgba(dot_color))),
        )
}

/// A status pill: dot, label, and the elapsed clock — the same thing the
/// sidebar row shows, in one line, for the title bar.
fn status_chip(status: SessionStatus, elapsed: &SharedString, note: &SharedString) -> Div {
    let (dot_color, text_color, badge_bg) = status.colors();
    let tooltip: SharedString = if note.is_empty() {
        SharedString::from(status.description().to_string())
    } else {
        SharedString::from(format!("{} {}", status.description(), note))
    };
    div()
        .flex_shrink_0()
        .flex()
        .items_center()
        .gap_1()
        .px_1p5()
        .py_0p5()
        .rounded_full()
        .bg(rgba(badge_bg))
        .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
        .child(div().size(px(6.0)).rounded_full().bg(rgba(dot_color)))
        .child(
            div()
                .text_xs()
                .text_color(rgba(text_color))
                .child(status.label()),
        )
        .when(!elapsed.is_empty(), |el| {
            el.child(div().text_xs().text_color(rgba(ui::TEXT_DIM)).child("·"))
                .child(
                    div()
                        .text_xs()
                        .text_color(rgba(text_color))
                        .child(elapsed.clone()),
                )
        })
}

/// Coarse elapsed-time label: `now`, `42s`, `3m`, `2h`, `4d`.
///
/// Coarse on purpose — the string is what decides whether a frame is needed, so
/// a sidebar clock does not repaint the window every tick.
fn format_elapsed(elapsed: Duration) -> SharedString {
    let seconds = elapsed.as_secs();
    let label = if seconds < 5 {
        "now".to_string()
    } else if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3600 {
        format!("{}m", seconds / 60)
    } else if seconds < 86_400 {
        format!("{}h", seconds / 3600)
    } else {
        format!("{}d", seconds / 86_400)
    };
    SharedString::from(label)
}

/// Ask a terminal's shell to go away, together with everything it started.
///
/// The PTY master makes the shell a session leader (`portable-pty` calls
/// `setsid`), so signalling the *group* (`-pid`) reaches the foreground job as
/// well: closing a terminal must not leave a build running behind it.
#[cfg(unix)]
fn hangup_process_group(pid: Option<u32>) {
    if let Some(pid) = pid {
        // Safety: a negative pid targets the process group; a stale pid simply
        // fails with ESRCH, which there is nothing to do about.
        unsafe {
            libc::kill(-(pid as i32), libc::SIGHUP);
        }
    }
}

#[cfg(not(unix))]
fn hangup_process_group(_pid: Option<u32>) {}

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
/// user has already looked at a finished/failed terminal, and whether an agent
/// has gone silent.
fn screen_fingerprint(lines: &[String], title: &str, progress: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    lines.hash(&mut hasher);
    title.hash(&mut hasher);
    progress.hash(&mut hasher);
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
                        let spawned = AppState::spawn_terminal_view(
                            Some(&current_dir_clone),
                            palette_clone.clone(),
                            self_weak,
                            1,
                            cx,
                        )
                        .expect("Failed to spawn initial terminal");

                        spawned.view.read(cx).focus_handle().focus(window);
                        let initial_session =
                            Session::new(1, SharedString::from("terminal 1"), spawned);

                        let initial_project = Project {
                            id: 1,
                            name: SharedString::from(current_name),
                            path: current_dir_clone,
                            sessions: vec![initial_session],
                            active_session_idx: 0,
                        };

                        AppState {
                            projects: vec![initial_project],
                            active_project_idx: 0,
                            next_id: 2,
                            session_index: HashMap::from([(1, (0, 0))]),
                            palette: palette_clone,
                            focused_session_id: None,
                            collapsed_projects: HashSet::new(),
                        }
                    });

                    let app_state_weak = app_state.downgrade();
                    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                        loop {
                            cx.background_executor()
                                .timer(Duration::from_millis(500))
                                .await;
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
