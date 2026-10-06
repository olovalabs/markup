use std::hash::Hasher as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
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

mod assets;
mod detect;
use assets::{load_embedded_fonts, sync_component_fonts, CombinedAssets, MONO_FONT};

/// How often terminal statuses are re-evaluated.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// How long a terminal keeps its Done/Error badge before falling back to
/// Idle (it also clears immediately when the terminal is focused).
const FINISHED_HOLD: Duration = Duration::from_secs(10);
/// A running command that produces no output for this long and shows no
/// visible working signal is considered stuck (Blocked).
const STUCK_AFTER: Duration = Duration::from_secs(45);
/// Grace window after an agent process first appears (herdr's
/// AGENT_STARTUP_GRACE_WINDOW): startup UI churn must not count as work.
const AGENT_STARTUP_GRACE: Duration = Duration::from_secs(3);
/// An agent's working stretch is only over once working evidence has been
/// absent for this long (herdr debounces Working→Idle with pending-idle
/// confirmations to ride out screen redraw gaps).
const WORK_END_CONFIRM: Duration = Duration::from_millis(1500);
/// An agent working stretch must last at least this long for its end to be
/// reported as Done; shorter bursts are startup/UI noise.
const MIN_WORK_STRETCH: Duration = Duration::from_secs(5);

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
/// Live status shown for a terminal. Variants are ordered by priority
/// (lowest → highest); when several signals apply at once the highest wins.
/// Priority: Error > Blocked > Working > Done > Active > Idle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SessionStatus {
    /// Not focused and nothing is running.
    Idle,
    /// The user is currently focused/typing in this terminal.
    Active,
    /// The last task finished successfully.
    Done,
    /// An AI agent CLI or command is running.
    Working,
    /// The process is waiting for input/approval, or is stuck.
    Blocked,
    /// The process failed (non-zero exit, crash, or error output).
    Error,
}

impl SessionStatus {
    pub fn label(self) -> &'static str {
        match self {
            SessionStatus::Idle => "idle",
            SessionStatus::Active => "active",
            SessionStatus::Done => "done",
            SessionStatus::Working => "working",
            SessionStatus::Blocked => "blocked",
            SessionStatus::Error => "error",
        }
    }

    /// Status dot / badge text color (GitHub dark palette).
    pub fn color(self) -> u32 {
        match self {
            SessionStatus::Idle => 0x8b949eff,    // gray
            SessionStatus::Active => 0x58a6ffff,  // blue
            SessionStatus::Done => 0x3fb950ff,    // green
            SessionStatus::Working => 0xd29922ff, // yellow
            SessionStatus::Blocked => 0xf0883eff, // orange
            SessionStatus::Error => 0xf85149ff,   // red
        }
    }

    /// Badge background color.
    pub fn badge_bg(self) -> u32 {
        match self {
            SessionStatus::Idle => 0x21262d88,
            SessionStatus::Active => 0x1f6feb33,
            SessionStatus::Done => 0x23863644,
            SessionStatus::Working => 0x9e6a0344,
            SessionStatus::Blocked => 0xdb6d2844,
            SessionStatus::Error => 0xda363344,
        }
    }
}

pub struct Session {
    pub id: usize,
    pub title: SharedString,
    pub terminal: Entity<TerminalView>,
    pub status: SessionStatus,
    pub pid: Option<u32>,
    /// Basename of the shell spawned in this session. Nested interactive
    /// shells of the same kind are ignored when scanning the process tree so
    /// they don't count as "work running".
    pub shell_name: String,
    /// Whether this session's terminal currently holds keyboard focus.
    pub focused: bool,
    /// AI agent CLI identified by process name (e.g. `claude`), if one is
    /// running or just finished.
    pub agent: Option<String>,
    /// An agent process was present in the process tree as of the last tick.
    pub agent_process_present: bool,
    /// Startup grace window after an agent process first appeared.
    pub agent_grace_until: Option<Instant>,
    /// Start of the agent's current uninterrupted working stretch.
    pub working_since: Option<Instant>,
    /// Last moment positive working evidence was seen in the current stretch.
    pub last_work_activity: Option<Instant>,
    /// A foreground command was running as of the last detection tick.
    pub busy: bool,
    /// The shell exited (terminal is dead); status is pinned to Done/Error.
    pub shell_exited: bool,
    /// Terminal outcome (Done/Error) recorded when the last command finished.
    pub finished_status: Option<SessionStatus>,
    pub finished_at: Option<Instant>,
    /// Screen-quietness tracking for the "stuck" heuristic.
    pub last_screen_hash: u64,
    pub last_screen_change: Option<Instant>,
    /// Focus subscriptions for the terminal view (kept alive here).
    focus_wired: bool,
    focus_subs: Vec<gpui::Subscription>,
}
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
    /// Process table used by status detection, refreshed once per poll tick.
    sys: sysinfo::System,
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
    ) -> Result<(Entity<TerminalView>, Option<u32>, String)> {
        let (shell_cmd, shell_name) = detect_shell();
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
        let target_for_exit = window_title_target.clone();
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
                .with_exit_callback(move |_window, cx| {
                    let _ = target_for_exit.update(cx, |state, cx| {
                        state.on_shell_exit(session_id, cx);
                    });
                })
        });
        Ok((terminal, pid, shell_name))
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

    fn session_mut(&mut self, session_id: usize) -> Option<&mut Session> {
        self.projects
            .iter_mut()
            .flat_map(|p| p.sessions.iter_mut())
            .find(|s| s.id == session_id)
    }

    /// Called when the session's shell exits (PTY EOF). The exit status code
    /// decides between Done (0 / unknown) and Error (non-zero): herdr treats
    /// process exit as the authoritative completion signal.
    fn on_shell_exit(&mut self, session_id: usize, cx: &mut Context<Self>) {
        let now = Instant::now();
        let Some(session) = self.session_mut(session_id) else {
            return;
        };
        if session.shell_exited {
            return;
        }
        session.shell_exited = true;
        let exit_code = session.terminal.read(cx).exit_status();
        let outcome = match exit_code {
            Some(code) if code != 0 => SessionStatus::Error,
            _ => SessionStatus::Done,
        };
        session.finished_status = Some(outcome);
        session.finished_at = Some(now);
        session.status = outcome;
        session.busy = false;
        cx.notify();
    }

    /// Focus moved into this terminal: it becomes Active unless a higher
    /// priority state (Working/Blocked/Error…) is in effect. Focusing also
    /// acknowledges and clears a Done/Error badge.
    fn on_terminal_focused(&mut self, session_id: usize, cx: &mut Context<Self>) {
        let Some(session) = self.session_mut(session_id) else {
            return;
        };
        session.focused = true;
        match session.status {
            SessionStatus::Done | SessionStatus::Error => {
                session.finished_status = None;
                session.finished_at = None;
                session.status = SessionStatus::Active;
            }
            SessionStatus::Idle => {
                session.status = SessionStatus::Active;
            }
            _ => {}
        }
        cx.notify();
    }

    /// Focus left this terminal. Working/Blocked/Error/Done are untouched —
    /// switching away must never reset them.
    fn on_terminal_blurred(&mut self, session_id: usize, cx: &mut Context<Self>) {
        let Some(session) = self.session_mut(session_id) else {
            return;
        };
        session.focused = false;
        if session.status == SessionStatus::Active {
            session.status = SessionStatus::Idle;
        }
        cx.notify();
    }


    /// Periodic detection tick. Mirrors herdr's approach: process-tree
    /// inspection (agent CLI running?), screen-tail + OSC-title pattern
    /// matching (blocked / working / prompt), and process exit as the
    /// authoritative completion signal.
    pub fn poll_sessions_status(&mut self, cx: &mut Context<Self>) {
        let now = Instant::now();
        // One process-table refresh per tick, shared by all sessions.
        detect::refresh_processes(&mut self.sys);

        let mut changed = false;
        for project in &mut self.projects {
            for session in &mut project.sessions {
                let before = session.status;

                if session.shell_exited {
                    // Terminal is dead: hold Done/Error until it decays.
                    if Self::decay_finished(session, now) {
                        changed = true;
                    }
                    continue;
                }

                let (exited, exit_code, title, lines) = {
                    let terminal = session.terminal.read(cx);
                    (
                        terminal.has_exited(),
                        terminal.exit_status(),
                        terminal.title().unwrap_or("").to_string(),
                        terminal.bottom_lines(24),
                    )
                };

                // Backstop for the exit callback (EOF on the PTY).
                if exited {
                    session.shell_exited = true;
                    let outcome = match exit_code {
                        Some(code) if code != 0 => SessionStatus::Error,
                        _ => SessionStatus::Done,
                    };
                    session.finished_status = Some(outcome);
                    session.finished_at = Some(now);
                    session.status = outcome;
                    session.busy = false;
                    if before != outcome {
                        changed = true;
                    }
                    continue;
                }

                let scan = detect::scan_session_processes(&self.sys, session.pid, &session.shell_name);
                let signals = detect::scan_screen(&lines);
                let title_working = detect::title_indicates_working(&title);

                // Track how long the visible screen has been unchanged; a
                // long-silent running command with no visible working signal
                // is considered stuck (Blocked).
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                for line in lines.iter().rev().take(16) {
                    hasher.write(line.as_bytes());
                    hasher.write_u8(0);
                }
                let screen_hash = hasher.finish();
                if screen_hash != session.last_screen_hash {
                    session.last_screen_hash = screen_hash;
                    session.last_screen_change = Some(now);
                }
                let silent_for = session
                    .last_screen_change
                    .map_or(Duration::MAX, |t| now.saturating_duration_since(t));

                // Completion detection for one-shot commands: something that
                // was running is gone (herdr: the foreground job disappearing
                // is the authoritative "finished" signal). Classify the
                // outcome by the recent output since the shell already reaped
                // the child and its exit code is not available to us.
                if session.busy && !scan.busy {
                    let outcome = if signals.error_hint {
                        SessionStatus::Error
                    } else {
                        SessionStatus::Done
                    };
                    session.finished_status = Some(outcome);
                    session.finished_at = Some(now);
                    session.working_since = None;
                    session.last_work_activity = None;
                }
                session.busy = scan.busy;
                if let Some(agent) = scan.agent {
                    session.agent = Some(agent);
                }

                // ---- Agent lifecycle (herdr-style) ----
                // An identified agent CLI is NOT "working" just because its
                // process exists: launched-but-waiting-at-its-prompt means
                // Idle (herdr's known-agent idle fallback). Working needs
                // positive screen evidence (spinner, "esc to interrupt",
                // progress bar, working verbs).
                let agent_present = scan.agent.is_some();
                if agent_present && !session.agent_process_present {
                    // Agent process just appeared: startup grace, like
                    // herdr's AGENT_STARTUP_GRACE_WINDOW.
                    session.agent_grace_until = Some(now + AGENT_STARTUP_GRACE);
                    session.working_since = None;
                    session.last_work_activity = None;
                }
                session.agent_process_present = agent_present;
                if !agent_present {
                    session.agent_grace_until = None;
                }

                let in_grace = session.agent_grace_until.is_some_and(|until| now < until);
                let working_evidence = signals.working || title_working;

                if agent_present && working_evidence && !in_grace {
                    if session.working_since.is_none() {
                        session.working_since = Some(now);
                    }
                    session.last_work_activity = Some(now);
                } else if agent_present && session.working_since.is_some() && !signals.blocked {
                    // Working evidence is gone. Debounce before declaring the
                    // stretch over (herdr's pending-idle confirmations) so a
                    // brief redraw gap doesn't flicker the badge.
                    let end_confirmed = session.last_work_activity.map_or(true, |t| {
                        now.saturating_duration_since(t) >= WORK_END_CONFIRM
                    });
                    if end_confirmed {
                        let since = session.working_since.take();
                        session.last_work_activity = None;
                        if let Some(since) = since
                            && now.saturating_duration_since(since) >= MIN_WORK_STRETCH
                        {
                            // The agent finished a real task and returned to
                            // its prompt: flash Done.
                            session.finished_status = Some(SessionStatus::Done);
                            session.finished_at = Some(now);
                        }
                    }
                }

                let mut detected = if signals.blocked {
                    SessionStatus::Blocked
                } else if agent_present {
                    if session.working_since.is_some() {
                        // Positive working evidence (or the debounce window
                        // right after it): the agent is processing.
                        SessionStatus::Working
                    } else if let Some(outcome) = session.finished_status {
                        // Fresh "task done" flash after an agent working
                        // stretch ended (or the shell-exit outcome).
                        let fresh = session
                            .finished_at
                            .map_or(false, |t| now.saturating_duration_since(t) < FINISHED_HOLD);
                        if fresh {
                            outcome
                        } else {
                            session.finished_status = None;
                            session.finished_at = None;
                            if session.focused {
                                SessionStatus::Active
                            } else {
                                SessionStatus::Idle
                            }
                        }
                    } else {
                        // Agent is alive but shows no working evidence: it
                        // sits at its input prompt waiting for the next task
                        // (herdr's DEFAULT_KNOWN_AGENT_IDLE_FALLBACK).
                        SessionStatus::Idle
                    }
                } else if scan.busy {
                    // One-shot command running: Working by process presence.
                    if !signals.working && !title_working && silent_for >= STUCK_AFTER {
                        SessionStatus::Blocked
                    } else {
                        SessionStatus::Working
                    }
                } else if let Some(outcome) = session.finished_status {
                    // Sticky Done/Error right after a command finished.
                    // Process exit is authoritative (herdr): it beats a stale
                    // spinner still visible on screen or in the OSC title.
                    // Focusing the terminal acknowledges it (see
                    // on_terminal_focused).
                    let fresh = session
                        .finished_at
                        .map_or(false, |t| now.saturating_duration_since(t) < FINISHED_HOLD);
                    if fresh {
                        outcome
                    } else {
                        session.finished_status = None;
                        session.finished_at = None;
                        session.agent = None;
                        if session.focused {
                            SessionStatus::Active
                        } else {
                            SessionStatus::Idle
                        }
                    }
                } else if (title_working || signals.working) && !signals.prompt {
                    // Spinner on screen / OSC title without any local
                    // process: keep Working for agents we can't see (ssh,
                    // container) — unless the shell prompt is already back,
                    // which wins (prompt-first idle).
                    SessionStatus::Working
                } else if session.focused {
                    SessionStatus::Active
                } else {
                    SessionStatus::Idle
                };

                // New work supersedes any stale terminal outcome. For agents
                // "new work" means a fresh working stretch (process presence
                // alone is not work); for commands it means a running process.
                let new_work_active = if agent_present {
                    session.working_since.is_some()
                } else {
                    scan.busy
                };
                if new_work_active || signals.blocked {
                    session.finished_status = None;
                    session.finished_at = None;
                }

                // Focus is only a status when nothing higher-priority applies.
                if detected == SessionStatus::Idle && session.focused {
                    detected = SessionStatus::Active;
                }

                if session.status != detected {
                    session.status = detected;
                    changed = true;
                }
            }
        }
        if changed {
            cx.notify();
        }
    }

    /// Decay a Done/Error badge to Idle/Active once FINISHED_HOLD elapsed.
    /// Returns true when the status changed.
    fn decay_finished(session: &mut Session, now: Instant) -> bool {
        let Some(at) = session.finished_at else {
            return false;
        };
        if now.saturating_duration_since(at) < FINISHED_HOLD {
            return false;
        }
        session.finished_status = None;
        session.finished_at = None;
        session.agent = None;
        let next = if session.focused {
            SessionStatus::Active
        } else {
            SessionStatus::Idle
        };
        if session.status == next {
            return false;
        }
        session.status = next;
        true
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
        if let Ok((terminal, pid, shell_name)) = Self::spawn_terminal_view(
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
                status: SessionStatus::Active,
                pid,
                shell_name,
                focused: true,
                agent: None,
                agent_process_present: false,
                agent_grace_until: None,
                working_since: None,
                last_work_activity: None,
                busy: false,
                shell_exited: false,
                finished_status: None,
                finished_at: None,
                last_screen_hash: 0,
                last_screen_change: None,
                focus_wired: false,
                focus_subs: Vec::new(),
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

            if let Ok((terminal, pid, shell_name)) = Self::spawn_terminal_view(
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
                    status: SessionStatus::Active,
                    pid,
                    shell_name,
                    focused: true,
                    agent: None,
                    agent_process_present: false,
                    agent_grace_until: None,
                    working_since: None,
                    last_work_activity: None,
                    busy: false,
                    shell_exited: false,
                    finished_status: None,
                    finished_at: None,
                    last_screen_hash: 0,
                    last_screen_change: None,
                    focus_wired: false,
                    focus_subs: Vec::new(),
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
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Lazily wire focus listeners for each terminal so the Active state
        // follows focus in real time (and Done/Error clear when focused).
        for project in &mut self.projects {
            for session in &mut project.sessions {
                if !session.focus_wired {
                    let handle = session.terminal.read(cx).focus_handle().clone();
                    let session_id = session.id;
                    session.focus_wired = true;
                    session.focus_subs.push(cx.on_focus(
                        &handle,
                        window,
                        move |this, _window, cx| {
                            this.on_terminal_focused(session_id, cx);
                        },
                    ));
                    session.focus_subs.push(cx.on_blur(
                        &handle,
                        window,
                        move |this, _window, cx| {
                            this.on_terminal_blurred(session_id, cx);
                        },
                    ));
                }
            }
        }

        let active_title = self
            .active_session()
            .map(|s| s.title.clone())
            .unwrap_or_else(|| SharedString::from("Terminal"));

        let active_project_name = self
            .active_project()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "No Project".to_string());

        let active_session_view = self.active_session().map(|s| s.terminal.clone());

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
                        .text_size(px(12.5))
                        .text_color(rgba(0x8b949eff))
                        .child(format!("{active_project_name} — {active_title}")),
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
                                                                    .bg(rgba(session.status.color()))
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
                                                            .bg(rgba(session.status.badge_bg()))
                                                            .text_color(rgba(session.status.color()))
                                                            .child({
                                                                let label = session.status.label();
                                                                match session.agent.as_deref() {
                                                                    Some(agent)
                                                                        if matches!(
                                                                            session.status,
                                                                            SessionStatus::Working
                                                                                | SessionStatus::Blocked
                                                                                | SessionStatus::Done
                                                                                | SessionStatus::Error
                                                                        ) =>
                                                                    {
                                                                        format!("{label} · {agent}")
                                                                    }
                                                                    _ => label.to_string(),
                                                                }
                                                            }),
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
                        let (initial_terminal, pid, shell_name) = AppState::spawn_terminal_view(
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
                            status: SessionStatus::Active,
                            pid,
                            shell_name,
                            focused: true,
                            agent: None,
                            agent_process_present: false,
                            agent_grace_until: None,
                            working_since: None,
                            last_work_activity: None,
                            busy: false,
                            shell_exited: false,
                            finished_status: None,
                            finished_at: None,
                            last_screen_hash: 0,
                            last_screen_change: None,
                            focus_wired: false,
                            focus_subs: Vec::new(),
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
                            sys: sysinfo::System::new(),
                        }
                    });

                    let app_state_weak = app_state.downgrade();
                    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
                        loop {
                            cx.background_executor().timer(POLL_INTERVAL).await;
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
