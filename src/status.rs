//! Live terminal status detection.
//!
//! Detection runs in two tiers, in this order:
//!
//! 1. **Agent rule packs.** If an agent CLI owns the shell's foreground process
//!    group, its herdr rule pack (`src/agent_rules.rs`) decides the state from
//!    the screen, the OSC title, and the OSC 9;4 progress field. This is where
//!    `working` and `blocked` come from, and it is the only tier that
//!    understands an agent's own UI.
//! 2. **Plain-shell fallback.** With no agent running, the state is derived
//!    from what is actually running: a foreground command means `Working`, a
//!    known interactive full-screen program means `Waiting`, a decision prompt
//!    means `Blocked`, failure output means `Error`.
//!
//! The foreground group is read from `/proc/<pid>/stat` field 8 (`tpgid`). Note
//! that a `portable-pty` shell is its own process-group leader, so `pgrp == pid`
//! while `tpgid` is the job actually running — reading the wrong field here once
//! made the whole process half of detection silently inert.
//!
//! herdr has no `error` state, so `Error`, `Killed`, `Exited` and `Stalled` are
//! ours. `Error`/`Killed`/`Exited` come from the terminal's real exit code (the
//! host applies that in `poll_sessions_status`, where a waiter thread records
//! the shell's true exit status) and, for plain shells, from failure text on
//! screen. Screen-based errors are deliberately *not* applied while an agent
//! pack is active, because agent transcripts are full of `error:` lines
//! belonging to the code being discussed rather than to the terminal.
//!
//! Sticky states: the terminal states (`Done`, `Error`, `Killed`, `Exited`)
//! persist while a terminal is unfocused and clear when the user looks at it,
//! so a completion or a failure stays visible until it has been seen.
//!
//! Platform note: the process-tree half is Linux-only (`/proc`). Elsewhere
//! [`foreground_process_names`] returns nothing and detection rests on the
//! screen and OSC rules alone.

#[cfg(unix)]
use std::collections::HashMap;
#[cfg(unix)]
use std::path::Path;
#[cfg(unix)]
use std::sync::LazyLock;
use std::time::Duration;

use crate::agent_rules::{self, DetectionInput, RuleState};

/// How long an agent may show no new output before it counts as [`Stalled`].
///
/// Agent CLIs animate a spinner or a timer while they work, so a completely
/// static screen for this long means the CLI is wedged, waiting on something
/// that never answered, or has a blocked UI. Only applied to recognized agents
/// (see `poll_sessions_status`): a quiet `cargo build --release` in a plain
/// shell is not a stall.
///
/// [`Stalled`]: SessionStatus::Stalled
pub const STALL_AFTER: Duration = Duration::from_secs(60);

/// How long an unacknowledged `Done` badge lingers before it falls back to
/// `Idle` (the host applies this; see `DONE_LINGER` in `main.rs`).
pub const DONE_LINGER: Duration = Duration::from_secs(30);

/// Foreground-group cache: shell pid -> (tpgid, process names).
///
/// The poller runs every 500 ms per session, but the foreground group only
/// changes when a command starts or ends. The `stat` read still happens each
/// tick (one small read, and it is the change detector), while the heavier
/// `/proc` walk is skipped while the group is unchanged. Entries are keyed by
/// shell pid; pids are few (one per session) and a changed group simply
/// overwrites its entry.
#[cfg(unix)]
static FG_CACHE: LazyLock<parking_lot::Mutex<HashMap<u32, (i64, Vec<String>)>>> =
    LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

/// The status a terminal session can report. Exactly one is active at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// User is currently focused/typing in this terminal.
    Active,
    /// Not focused, nothing is running.
    Idle,
    /// A command or AI agent CLI is running.
    Working,
    /// Still running, but nothing on screen has changed for a long time.
    Stalled,
    /// An interactive full-screen program owns the terminal and is waiting for
    /// keystrokes (vim, less, htop, ssh, a REPL, …).
    Waiting,
    /// Waiting for user input / approval / a decision.
    Blocked,
    /// Task finished successfully while the user was looking elsewhere.
    Done,
    /// Process failed (non-zero exit, crash, error output).
    Error,
    /// Process was terminated by a signal (crash, OOM kill, SIGHUP on close).
    Killed,
    /// The terminal's shell exited; there is nothing left to talk to.
    Exited,
}

impl SessionStatus {
    /// Render priority: what a one-line summary of a whole project should show.
    ///
    /// A blocker outranks a failure — a blocked agent is *waiting on you right
    /// now*, while a failed one is waiting for you to read it — and both outrank
    /// ordinary work.
    pub fn rank(self) -> u8 {
        match self {
            SessionStatus::Blocked => 10,
            SessionStatus::Error => 9,
            SessionStatus::Killed => 8,
            SessionStatus::Stalled => 7,
            SessionStatus::Done => 6,
            SessionStatus::Working => 5,
            SessionStatus::Waiting => 4,
            SessionStatus::Exited => 3,
            SessionStatus::Active => 2,
            SessionStatus::Idle => 1,
        }
    }

    /// Short label for the sidebar badge.
    pub fn label(self) -> &'static str {
        match self {
            SessionStatus::Active => "active",
            SessionStatus::Idle => "idle",
            SessionStatus::Working => "working",
            SessionStatus::Stalled => "stalled",
            SessionStatus::Waiting => "waiting",
            SessionStatus::Blocked => "blocked",
            SessionStatus::Done => "done",
            SessionStatus::Error => "error",
            SessionStatus::Killed => "killed",
            SessionStatus::Exited => "exited",
        }
    }

    /// One sentence explaining the state, for the badge tooltip.
    pub fn description(self) -> &'static str {
        match self {
            SessionStatus::Active => "This is the terminal you are working in.",
            SessionStatus::Idle => "Ready at the prompt, nothing running.",
            SessionStatus::Working => "A command or an AI agent is running.",
            SessionStatus::Stalled => "Still running, but nothing has been printed for a while.",
            SessionStatus::Waiting => "An interactive program is waiting for keystrokes.",
            SessionStatus::Blocked => "Waiting for you: an approval, a choice, or input.",
            SessionStatus::Done => "Finished successfully while you were looking elsewhere.",
            SessionStatus::Error => "The last command or the session failed.",
            SessionStatus::Killed => "The process was terminated by a signal.",
            SessionStatus::Exited => "The shell exited; this terminal is closed.",
        }
    }

    /// (dot, text, badge background) colors, from the spec.
    ///
    /// Hues carry the meaning: blue = you are here, amber = busy, purple =
    /// busy but silent, teal = waiting on keys, orange = decisions, green =
    /// finished, red = failed, pink = killed, slate = closed.
    pub fn colors(self) -> (u32, u32, u32) {
        match self {
            SessionStatus::Active => (0x58a6ffff, 0x58a6ffff, 0x1f6feb33),
            SessionStatus::Idle => (0x8b949eff, 0x8b949eff, 0x21262d88),
            SessionStatus::Working => (0xd29922ff, 0xd29922ff, 0x9e6a0344),
            SessionStatus::Stalled => (0xa371f7ff, 0xa371f7ff, 0x6e40c933),
            SessionStatus::Waiting => (0x39c5cfff, 0x39c5cfff, 0x1b7c8333),
            SessionStatus::Blocked => (0xdb6d28ff, 0xdb6d28ff, 0xbd561444),
            SessionStatus::Done => (0x3fb950ff, 0x3fb950ff, 0x23863644),
            SessionStatus::Error => (0xf85149ff, 0xf85149ff, 0xda363344),
            SessionStatus::Killed => (0xdb61a2ff, 0xdb61a2ff, 0xa41c4e33),
            SessionStatus::Exited => (0x6e7681ff, 0x6e7681ff, 0x21262d66),
        }
    }

    /// The terminal states: sticky badges that survive until the user looks at
    /// the terminal, because they report something that already happened.
    pub fn is_final(self) -> bool {
        matches!(
            self,
            SessionStatus::Done | SessionStatus::Error | SessionStatus::Killed | SessionStatus::Exited
        )
    }

    /// Something is running (or is supposed to be).
    pub fn is_busy(self) -> bool {
        matches!(
            self,
            SessionStatus::Working | SessionStatus::Stalled | SessionStatus::Waiting
        )
    }

    /// Worth interrupting the user for: it either needs them or already ended.
    pub fn needs_attention(self) -> bool {
        matches!(
            self,
            SessionStatus::Blocked
                | SessionStatus::Done
                | SessionStatus::Error
                | SessionStatus::Killed
                | SessionStatus::Stalled
        )
    }

    /// The state to report from a real process exit status.
    ///
    /// `Exited` is a clean `exit` at the shell (code 0) — deliberately *not*
    /// `Done`, which means "an agent finished a turn you have not read yet".
    pub fn from_exit(info: &ExitInfo) -> SessionStatus {
        match (info.signal.is_some(), info.code) {
            (true, _) => SessionStatus::Killed,
            (false, 0) => SessionStatus::Exited,
            (false, _) => SessionStatus::Error,
        }
    }
}

/// The real exit status of a session's shell, recorded by the waiter thread
/// that owns the PTY child (see `spawn_terminal_view`).
///
/// This is the most trustworthy signal a terminal has: a clean `exit`, an
/// application's own failure code, or the signal that killed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitInfo {
    pub code: i32,
    /// Signal name (`SIGKILL`, `SIGSEGV`, …) when the process was killed.
    pub signal: Option<String>,
}

impl ExitInfo {
    pub fn new(code: i32, signal: Option<String>) -> Self {
        Self { code, signal }
    }

    /// Short human-readable form for tooltips: `exit 0`, `exit 130`,
    /// `killed by SIGKILL`.
    pub fn note(&self) -> String {
        match &self.signal {
            Some(signal) => format!("killed by {signal}"),
            None if self.code == 0 => "exit 0".to_string(),
            None => format!("exit {}", self.code),
        }
    }
}

/// All inputs needed for one detection pass.
pub struct StatusInput<'a> {
    /// Visible screen rows, top to bottom.
    pub lines: &'a [String],
    /// OSC 0/2 title, if the application set one.
    pub title: Option<&'a str>,
    /// OSC 9;4 progress payload (`4;1;-1`, `4;0`, …), if the application sent
    /// one recently. This is the signal several agent CLIs use to announce
    /// "my turn is running" / "my turn is done".
    pub progress: Option<&'a str>,
    /// Names in the shell's foreground process group. Empty at a prompt.
    pub foreground: &'a [String],
}

/// A verdict plus the evidence behind it.
///
/// The evidence is what makes the badges trustworthy: the sidebar tooltip and
/// the `T3_STATUS_DEBUG` dump both render it, so a wrong badge can always be
/// traced back to the rule or the screen line that produced it.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub status: SessionStatus,
    /// The agent rule that decided the state, when a pack matched.
    pub rule: Option<agent_rules::RuleMatch>,
    /// Plain-language reason for a fallback (non-agent) verdict.
    pub reason: Option<&'static str>,
}

/// Does this process name belong to an agent CLI we have a rule pack for?
pub fn is_agent_process(name: &str) -> bool {
    agent_rules::identify(name).is_some()
}

/// The rule pack that owns any of `process_names`, for logging.
pub fn pack_for(process_names: &[String]) -> Option<&'static str> {
    agent_rules::embedded()
        .pack_for(process_names)
        .map(|pack| pack.id.as_str())
}

/// Which pack and rule produced the verdict, for `T3_STATUS_DEBUG` output.
///
/// `None` means either that no pack applies (a plain shell) or that the pack
/// matched no rule; the caller logs the pack separately to tell those apart.
pub fn explain(input: &StatusInput<'_>) -> Option<agent_rules::RuleMatch> {
    let screen = input.lines.join("\n");
    let pack = agent_rules::embedded().pack_for(input.foreground)?;
    agent_rules::evaluate(
        pack,
        DetectionInput {
            screen: &screen,
            osc_title: input.title.unwrap_or(""),
            osc_progress: input.progress.unwrap_or(""),
        },
    )
}

/// Names of the processes in the shell's foreground job, best candidates first.
///
/// Three sources, because no single one is enough:
/// * `/proc/<pid>/comm` — set from the executable name (catches symlinked CLIs),
/// * `argv[0]` basename — reflects `exec -a` renames,
/// * arguments of an interpreter — where npm-installed agents live
///   (`node …/node_modules/@anthropic-ai/claude-code/cli.js`).
///
/// The whole foreground *process group* is walked, not just its leader, so a
/// wrapper (`bash -lc …`, `sh -c …`, `npx …`) cannot hide the agent that is
/// really doing the work. That walk is the expensive part, so it only runs when
/// the group changes (see [`FG_CACHE`]).
///
/// Empty when the shell itself owns the foreground group, i.e. the user is back
/// at a prompt, or on platforms without `/proc`.
#[cfg(unix)]
pub fn foreground_process_names(pid: Option<u32>) -> Vec<String> {
    let Some(pid) = pid else {
        return Vec::new();
    };
    let Some(tpgid) = read_proc_tpgid(pid) else {
        return Vec::new();
    };
    // `tpgid == 0` means no controlling terminal; `tpgid == pid` means the
    // shell itself owns the foreground group, i.e. it is sitting at a prompt.
    if tpgid == 0 || tpgid == pid as i64 {
        return Vec::new();
    }

    // Fast path: same foreground group as last tick means the same names —
    // skip the `/proc` walk. (A wrapper that swaps its child without changing
    // groups is re-read on the next command; the worst case is a briefly stale
    // pack selection, never a stuck badge.)
    if let Some((cached_tpgid, names)) = FG_CACHE.lock().get(&pid) {
        if *cached_tpgid == tpgid {
            return names.clone();
        }
    }

    let mut names = Vec::with_capacity(4);
    let mut push = |name: String| {
        let name = name.trim().to_string();
        if !name.is_empty() && !names.contains(&name) {
            names.push(name);
        }
    };

    // Group leader first (its `comm`/`argv[0]` is the most specific name), then
    // every other member of the same process group.
    let mut pids = vec![tpgid as u32];
    for other in group_members(pid, tpgid) {
        if other != tpgid as u32 {
            pids.push(other);
        }
    }
    for proc_pid in pids {
        read_process_names(&Path::new("/proc").join(proc_pid.to_string()), &mut push);
    }

    FG_CACHE.lock().insert(pid, (tpgid, names.clone()));
    names
}

#[cfg(not(unix))]
pub fn foreground_process_names(_pid: Option<u32>) -> Vec<String> {
    Vec::new()
}

/// Every process in `tpgid`'s process group, capped so a pathological machine
/// cannot turn one poll tick into an unbounded `/proc` walk.
#[cfg(unix)]
fn group_members(_shell_pid: u32, tpgid: i64) -> Vec<u32> {
    /// Enough for any real shell pipeline; extra members add nothing but cost.
    const MAX_GROUP_MEMBERS: usize = 64;

    let mut pids = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return pids;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|name| name.parse::<u32>().ok()) else {
            // Not a pid directory (`/proc/self`, `/proc/meminfo`, …).
            continue;
        };
        if pids.len() >= MAX_GROUP_MEMBERS {
            break;
        }
        let stat_path = Path::new("/proc").join(pid.to_string()).join("stat");
        let Ok(stat) = std::fs::read_to_string(stat_path) else {
            continue;
        };
        if parse_stat_pgrp(&stat) == Some(tpgid) {
            pids.push(pid);
        }
    }
    pids
}

/// Every name worth testing against the rule packs for one process.
///
/// Written as a free function over a `push` callback so the caller controls
/// deduplication and allocation.
#[cfg(unix)]
fn read_process_names(proc_dir: &Path, push: &mut impl FnMut(String)) {
    if let Ok(comm) = std::fs::read_to_string(proc_dir.join("comm")) {
        push(comm);
    }

    let Ok(cmdline) = std::fs::read(proc_dir.join("cmdline")) else {
        return;
    };
    let args: Vec<&[u8]> = cmdline
        .split(|&b| b == 0)
        .filter(|arg| !arg.is_empty())
        .collect();
    let Some(argv0) = args.first() else {
        return;
    };

    let argv0 = String::from_utf8_lossy(argv0).to_string();
    push(basename(&argv0));
    if !is_interpreter(&basename(&argv0)) {
        return;
    }

    // Interpreters hide the real program behind a script path; see
    // [`argument_name_candidates`].
    for arg in &args[1..] {
        if arg.starts_with(b"-") {
            continue;
        }
        for candidate in argument_name_candidates(&String::from_utf8_lossy(arg)) {
            push(candidate);
        }
    }
}

/// Candidate program names hidden inside one argument of an interpreter.
///
/// npm-style installs put the agent's name in a *directory* component, so every
/// component counts, not just the file name:
/// `…/node_modules/@anthropic-ai/claude-code/cli.js` yields `@anthropic-ai`,
/// `claude-code`, `cli.js`, and the extension-stripped `cli`. A component that
/// is itself a script also contributes its stem, which is how `claude.js`
/// becomes the `claude` the packs know.
#[cfg(unix)]
fn argument_name_candidates(argument: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    for component in argument.split('/') {
        if component.is_empty() {
            continue;
        }
        let name = basename(component);
        if name.is_empty() {
            continue;
        }
        if let Some((stem, extension)) = name.rsplit_once('.') {
            if SCRIPT_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()) && !stem.is_empty()
            {
                candidates.push(stem.to_string());
            }
        }
        candidates.push(name);
    }
    candidates
}

/// Interpreters that run another program named in their arguments.
const INTERPRETERS: &[&str] = &[
    "node", "nodejs", "deno", "bun", "python", "python3", "ruby", "perl", "php", "java", "sh",
    "bash", "zsh", "dash", "fish", "npx", "bunx", "pnpm", "yarn", "npm", "sudo", "env", "doas",
    "timeout", "stdbuf", "nice",
];

/// Extensions stripped when a script name is tested against the packs.
const SCRIPT_EXTENSIONS: &[&str] = &["js", "mjs", "cjs", "ts", "py", "sh", "rb", "php", "jar", "exe"];

#[cfg(unix)]
fn is_interpreter(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    INTERPRETERS.contains(&lower.as_str())
}

/// Last path component of `arg`, with a Windows `.exe` suffix removed so the
/// same pack lookup works on both platforms.
#[cfg(unix)]
fn basename(arg: &str) -> String {
    let name = arg.rsplit(['/', '\\']).next().unwrap_or(arg);
    name.strip_suffix(".exe").unwrap_or(name).to_string()
}

/// Field 8 (`tpgid`) of `/proc/<pid>/stat`: the terminal's foreground process
/// group, i.e. whoever is *running right now* under this shell.
#[cfg(unix)]
fn read_proc_tpgid(pid: u32) -> Option<i64> {
    let stat_path = Path::new("/proc").join(pid.to_string()).join("stat");
    let stat = std::fs::read_to_string(stat_path).ok()?;
    parse_stat_tpgid(&stat)
}

/// The fields *after* the parenthesised `comm`, which may itself contain spaces
/// and parens, so the only safe split is on the **last** `)`.
#[cfg(unix)]
fn after_comm(stat: &str) -> Option<&str> {
    stat.rsplit_once(')').map(|(_, rest)| rest)
}

/// Field 8 (`tpgid`) of one `/proc/<pid>/stat` line — the foreground group.
///
/// Counting from just after `comm` the fields are 1-indexed as:
///
/// ```text
/// 3 state | 4 ppid | 5 pgrp | 6 session | 7 tty_nr | 8 tpgid
/// ```
///
/// Reading field 5 here was a real bug — that is `pgrp`, and a `portable-pty`
/// shell is its own process-group leader, so `pgrp == pid` and the caller's
/// guard rejected every result. The whole process-tree half of detection was
/// silently dead: a running CLI could never be reported as `Working`.
#[cfg(unix)]
fn parse_stat_tpgid(stat: &str) -> Option<i64> {
    // This slice starts at field 3 (state), so tpgid at field 8 is index 5.
    after_comm(stat)?.split_whitespace().nth(8 - 3)?.parse().ok()
}

/// Field 5 (`pgrp`) of one `/proc/<pid>/stat` line — the process group, which
/// identifies the foreground *job* a process belongs to.
#[cfg(unix)]
fn parse_stat_pgrp(stat: &str) -> Option<i64> {
    after_comm(stat)?.split_whitespace().nth(5 - 3)?.parse().ok()
}

/// True when `T3_STATUS_DEBUG` is set: the poller logs every detection pass
/// together with the facts that produced it.
///
/// This is the fastest way to explain a wrong badge — run the app with
/// `T3_STATUS_DEBUG=1` and the terminal shows exactly why each state changed.
pub fn debug_enabled() -> bool {
    std::env::var_os("T3_STATUS_DEBUG").is_some()
}

/// The single current status for a terminal.
///
/// `prev` is the last reported status; it drives stickiness. `focused` means
/// the user is interacting with this terminal right now.
pub fn evaluate(input: &StatusInput<'_>, prev: SessionStatus, focused: bool) -> SessionStatus {
    verdict(input, prev, focused).status
}

/// Like [`evaluate`], but also returns the agent rule match behind the verdict
/// (if any), so callers that log the explanation do not evaluate twice.
pub fn evaluate_with_match(
    input: &StatusInput<'_>,
    prev: SessionStatus,
    focused: bool,
) -> (SessionStatus, Option<agent_rules::RuleMatch>) {
    let verdict = verdict(input, prev, focused);
    (verdict.status, verdict.rule)
}

/// Full detection pass: sticky resolution, then the two detection tiers.
pub fn verdict(input: &StatusInput<'_>, prev: SessionStatus, focused: bool) -> Verdict {
    // Sticky terminal states are cleared on focus, otherwise they persist.
    if prev.is_final() {
        let status = if focused { SessionStatus::Idle } else { prev };
        return Verdict {
            status,
            rule: None,
            reason: None,
        };
    }

    let detected = detect_full(input);

    // Priority resolution with focus folded in as the lowest rung.
    let focus_status = if focused {
        SessionStatus::Active
    } else {
        SessionStatus::Idle
    };
    let status = if detected.status.rank() > focus_status.rank() {
        detected.status
    } else {
        focus_status
    };
    Verdict { status, ..detected }
}

/// Pure detection (no stickiness): agent packs, then the plain-shell fallback.
fn detect_full(input: &StatusInput<'_>) -> Verdict {
    let Some(pack) = agent_rules::embedded().pack_for(input.foreground) else {
        // No agent CLI is driving this terminal, so the only signals available
        // are generic ones. The fallback works on the recent lines only and
        // lowercases them once, shared by every rule.
        let recent = FallbackRules::recent_lower(input, FallbackRules::WINDOW);
        if FallbackRules::error_lower(&recent) {
            return Verdict {
                status: SessionStatus::Error,
                rule: None,
                reason: Some("failure output"),
            };
        }
        if FallbackRules::blocked_lower(&recent) {
            return Verdict {
                status: SessionStatus::Blocked,
                rule: None,
                reason: Some("waiting for input"),
            };
        }
        if input.foreground.is_empty() {
            return Verdict {
                status: SessionStatus::Idle,
                rule: None,
                reason: None,
            };
        }
        // A full-screen program that is up and waiting for keystrokes is not
        // "working"; it is waiting for the user, which is a different badge.
        if FallbackRules::waiting_program(input.foreground) {
            return Verdict {
                status: SessionStatus::Waiting,
                rule: None,
                reason: Some("interactive program"),
            };
        }
        return Verdict {
            status: SessionStatus::Working,
            rule: None,
            reason: Some("command running"),
        };
    };

    let screen = input.lines.join("\n");
    let matched = agent_rules::evaluate(
        pack,
        DetectionInput {
            screen: &screen,
            osc_title: input.title.unwrap_or(""),
            // OSC 9;4 progress: the terminal snoops it out of the byte stream,
            // so rules that read it (claude, grok, kiro, letta, qwen) can
            // finally fire.
            osc_progress: input.progress.unwrap_or(""),
        },
    );

    let status = match &matched {
        // A known agent with no matching rule is at rest, per upstream's
        // known-agent fallback.
        None => SessionStatus::Idle,
        // Transient overlays (transcript viewer, model picker) match only to say
        // "ignore me" — leave the badge where it was.
        Some(m) if m.skip_state_update => SessionStatus::Idle,
        Some(m) => match m.state {
            RuleState::Blocked => SessionStatus::Blocked,
            RuleState::Working => SessionStatus::Working,
            RuleState::Idle => SessionStatus::Idle,
            // Unknown means "neither" — but the agent process really is
            // running, so report that rather than claiming the pane is idle.
            RuleState::Unknown => SessionStatus::Working,
        },
    };

    Verdict {
        status,
        rule: matched,
        reason: None,
    }
}

/// Generic rules for terminals with no agent pack: a plain shell.
struct FallbackRules;

impl FallbackRules {
    /// Lines considered "recent" for the fallback rules.
    const WINDOW: usize = 12;

    /// The newest `n` non-empty lines of the visible screen, trimmed and
    /// lowercased once so every fallback rule shares the normalized form
    /// instead of each joining and lowercasing separately.
    fn recent_lower(input: &StatusInput<'_>, n: usize) -> Vec<String> {
        let mut lines: Vec<String> = input
            .lines
            .iter()
            .map(String::as_str)
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_lowercase())
            .collect();
        let excess = lines.len().saturating_sub(n);
        lines.split_off(excess)
    }

    /// The classic interactive prompt a plain command blocks on.
    ///
    /// Deliberately narrow: agents have their own packs, and vague wording here
    /// only produces false positives on ordinary output. Anchoring on how a
    /// prompt *ends* (`?`, `:`, `[y/n]`) is what keeps a transcript line that
    /// merely mentions "password" from turning a terminal orange.
    ///
    /// Takes [`Self::recent_lower`] output.
    fn blocked_lower(recent: &[String]) -> bool {
        recent.iter().any(|line| {
            let line = line.trim_end();
            // Password prompts are specific enough to match on their own.
            if line.contains("password:")
                || line.contains("passphrase:")
                || line.contains("password for ")
                || line.contains("enter passphrase")
                || line.contains("enter password")
            {
                return true;
            }
            // A question counts only when it asks for a decision.
            if line.ends_with('?') || line.ends_with(':') {
                let decisive = line.contains("proceed")
                    || line.contains("continue")
                    || line.contains("overwrite")
                    || line.contains("are you sure")
                    || line.contains("confirm")
                    || line.contains("do you want")
                    || line.contains("would you like")
                    || line.contains("approve")
                    || line.contains("allow");
                if decisive {
                    return true;
                }
            }
            line.contains("[y/n]")
                || line.contains("(y/n)")
                || line.contains("[y/n/q]")
                || line.contains("(y/n/q)")
                || line.contains("yes (y)")
                || line.contains("yes/no")
                || line.contains("press enter to continue")
                || line.contains("press any key")
        })
    }

    /// Strong failure markers on the most recent output lines.
    ///
    /// Scans a small window rather than just the last line, because a shell
    /// redraws its prompt *below* the failed command's output.
    fn error_lower(recent: &[String]) -> bool {
        let start = recent.len().saturating_sub(4);
        recent[start..].iter().any(|line| {
            let line = line.trim_start();
            line.contains("command not found")
                || line.contains("permission denied")
                || line.contains("panicked at")
                || line.contains("segmentation fault")
                || line.contains("traceback (most recent call last)")
                || line.contains("unhandled exception")
                || line.contains("assertion failed")
                || line.contains("cannot find")
                || line.contains("no such file or directory")
                || line.contains("failed to")
                || line.starts_with("error:")
                || line.starts_with("error ")
                || line.starts_with("fatal:")
                || line.starts_with("panic:")
                || line.starts_with("failed:")
                || line.starts_with("failure:")
        })
    }

    /// Is a full-screen, key-driven program the thing holding the terminal?
    ///
    /// Matched on the process name, which is exactly how the agent packs pick
    /// their agent — same idea, different list. Interpreters are included
    /// because a REPL (or `python -i`) is a prompt by another name; the cost is
    /// that a *silent* script is labelled `waiting` too, which is the same
    /// information a human would have from the outside.
    fn waiting_program(foreground: &[String]) -> bool {
        foreground.iter().any(|name| {
            let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
            let base = base.strip_suffix(".exe").unwrap_or(base).to_ascii_lowercase();
            let stem = base
                .split_once('.')
                .map(|(stem, _)| stem)
                .unwrap_or(base.as_str());
            WAITING_PROGRAMS.contains(&stem)
        })
    }
}

/// Programs that sit in the foreground and wait for keystrokes.
///
/// Kept to names that are *only* ever interactive: an editor, a pager, a
/// monitor, a remote shell, a REPL. Anything that can also run unattended
/// (`make`, `git`, `npm`, `bash -c`) stays `working` on purpose.
const WAITING_PROGRAMS: &[&str] = &[
    // editors & pagers
    "vim", "nvim", "vi", "nano", "emacs", "pico", "micro", "helix", "hx", "less", "more", "man",
    // monitors
    "htop", "btop", "top", "glances", "watch", "ncdu", "k9s", "lazydocker", "lazygit", "tig",
    "gitui", "ranger", "lf", "nnn", "mc", "visidata", "vd",
    // remote & database shells
    "ssh", "sftp", "telnet", "mosh", "mysql", "psql", "mongosh", "redis-cli", "sqlite3",
    // debuggers & REPLs
    "python", "python3", "ipython", "node", "deno", "bun", "irb", "pry", "iex", "ghci", "gdb",
    "lldb", "julia", "lua",
    // multiplexers & chat clients
    "tmux", "screen", "weechat", "irssi", "newsboat", "mutt", "neomutt",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(lines: &[&str], foreground: &[&str], focused: bool) -> SessionStatus {
        eval_with_progress(lines, foreground, focused, None)
    }

    fn eval_with_progress(
        lines: &[&str],
        foreground: &[&str],
        focused: bool,
        progress: Option<&str>,
    ) -> SessionStatus {
        let lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        let fg: Vec<String> = foreground.iter().map(|s| s.to_string()).collect();
        let input = StatusInput {
            lines: &lines,
            title: None,
            progress,
            foreground: &fg,
        };
        evaluate(&input, SessionStatus::Idle, focused)
    }

    /// A claude prompt box: two horizontal rules bracketing the input line.
    const CLAUDE_IDLE_BOX: &[&str] = &[
        "──────────────────────────────────",
        "  some conversation",
        "──────────────────────────────────",
        "❯ ",
        "──────────────────────────────────",
        "  ? for shortcuts",
    ];

    #[test]
    fn priority_order() {
        // A blocker is the loudest thing a terminal can say: it is waiting on
        // the user *right now*, ahead of anything that already finished.
        assert!(SessionStatus::Blocked.rank() > SessionStatus::Error.rank());
        assert!(SessionStatus::Error.rank() > SessionStatus::Killed.rank());
        assert!(SessionStatus::Killed.rank() > SessionStatus::Stalled.rank());
        assert!(SessionStatus::Stalled.rank() > SessionStatus::Done.rank());
        assert!(SessionStatus::Done.rank() > SessionStatus::Working.rank());
        assert!(SessionStatus::Working.rank() > SessionStatus::Waiting.rank());
        assert!(SessionStatus::Waiting.rank() > SessionStatus::Exited.rank());
        assert!(SessionStatus::Exited.rank() > SessionStatus::Active.rank());
        assert!(SessionStatus::Active.rank() > SessionStatus::Idle.rank());
    }

    #[test]
    fn final_states_are_the_sticky_ones() {
        for status in [
            SessionStatus::Done,
            SessionStatus::Error,
            SessionStatus::Killed,
            SessionStatus::Exited,
        ] {
            assert!(status.is_final(), "{status:?} should be sticky");
            assert!(status.needs_attention());
        }
        for status in [SessionStatus::Active, SessionStatus::Idle, SessionStatus::Working] {
            assert!(!status.is_final(), "{status:?} should not be sticky");
        }
    }

    #[test]
    fn exit_status_maps_to_the_right_badge() {
        assert_eq!(
            SessionStatus::from_exit(&ExitInfo::new(0, None)),
            SessionStatus::Exited
        );
        assert_eq!(
            SessionStatus::from_exit(&ExitInfo::new(1, None)),
            SessionStatus::Error
        );
        assert_eq!(
            SessionStatus::from_exit(&ExitInfo::new(137, Some("SIGKILL".to_string()))),
            SessionStatus::Killed
        );
        assert_eq!(ExitInfo::new(0, None).note(), "exit 0");
        assert_eq!(ExitInfo::new(130, None).note(), "exit 130");
        assert_eq!(
            ExitInfo::new(137, Some("SIGKILL".to_string())).note(),
            "killed by SIGKILL"
        );
    }

    #[test]
    fn plain_shell_command_is_working() {
        // No agent → anything in the foreground group is a running command.
        assert_eq!(eval(&["building..."], &["cargo"], false), SessionStatus::Working);
    }

    #[test]
    fn plain_shell_prompt_is_idle() {
        assert_eq!(eval(&["user@host ~/project ❯"], &[], false), SessionStatus::Idle);
        assert_eq!(eval(&[], &[], false), SessionStatus::Idle);
    }

    #[test]
    fn full_screen_programs_are_waiting_not_working() {
        assert_eq!(eval(&["~", "~", "-- INSERT --"], &["nvim"], false), SessionStatus::Waiting);
        assert_eq!(eval(&["lines 1-20"], &["less"], false), SessionStatus::Waiting);
        assert_eq!(eval(&[">>> "], &["python3"], false), SessionStatus::Waiting);
        assert_eq!(eval(&["[user@host ~]$ "], &["ssh"], false), SessionStatus::Waiting);
        // A build is not interactive, no matter how quiet it is.
        assert_eq!(eval(&["compiling"], &["cargo"], false), SessionStatus::Working);
    }

    #[test]
    fn claude_at_its_prompt_is_idle_even_though_the_process_runs() {
        // The pack wins over the foreground fallback: an agent sitting at its
        // prompt is Idle even though its TUI is the running process.
        assert_eq!(eval(CLAUDE_IDLE_BOX, &["claude"], false), SessionStatus::Idle);
    }

    #[test]
    fn claude_working_footer_is_working() {
        assert_eq!(
            eval(
                &["⏺ Read(src/main.rs)", "✳ Baking… (2m 14s · esc to interrupt)"],
                &["claude"],
                false
            ),
            SessionStatus::Working
        );
    }

    #[test]
    fn osc_progress_from_a_pack_reaches_the_rules() {
        // Grok announces a running turn with `4;1;-1`; with a blank screen the
        // progress report is the only signal, and it has to win.
        assert_eq!(
            eval_with_progress(&[""], &["grok"], false, Some("4;1;-1")),
            SessionStatus::Working
        );
        // `4;0;0` is its "cleared" report: back to the prompt.
        assert_eq!(
            eval_with_progress(&[""], &["grok"], false, Some("4;0;0")),
            SessionStatus::Idle
        );
    }

    #[test]
    fn claude_permission_menu_is_blocked() {
        assert_eq!(
            eval(
                &[
                    "──────────────────────────────────",
                    " Do you want to proceed?",
                    " ❯ 1. Yes",
                    "   2. No, and tell Claude what to do differently",
                    "──────────────────────────────────",
                    " esc to cancel · enter to confirm",
                ],
                &["claude"],
                false
            ),
            SessionStatus::Blocked
        );
    }

    #[test]
    fn agent_transcript_errors_do_not_paint_the_terminal_red() {
        // The code being discussed probably contains `error:`; that is not the
        // terminal failing, so the plain-shell Error rule stays out of the way
        // whenever a pack is active.
        assert_ne!(
            eval(
                &["error: could not compile `markup`", "❯ "],
                &["claude"],
                false
            ),
            SessionStatus::Error
        );
    }

    #[test]
    fn gemini_working_hint_is_not_a_blocker() {
        assert_eq!(
            eval(&[" ⠋ Thinking... (esc to cancel)"], &["gemini"], false),
            SessionStatus::Working
        );
    }

    #[test]
    fn agent_running_with_no_matching_rule_is_idle() {
        // Upstream's known-agent fallback: the agent is up but showing nothing
        // recognizable, so it is treated as idle rather than busy.
        assert_eq!(eval(&["hello"], &["claude"], false), SessionStatus::Idle);
    }

    #[test]
    fn plain_shell_error_output_is_error() {
        assert_eq!(
            eval(&["bash: foo: command not found"], &[], false),
            SessionStatus::Error
        );
        assert_eq!(
            eval(&["fatal: not a git repository"], &[], false),
            SessionStatus::Error
        );
        // The shell redraws its prompt under the failure; the error must still
        // be visible in the window.
        assert_eq!(
            eval(&["error: could not compile `markup`", "user@host ~/markup ❯"], &[], false),
            SessionStatus::Error
        );
    }

    #[test]
    fn plain_shell_confirmation_prompt_is_blocked() {
        assert_eq!(eval(&["Overwrite the file? [y/n]"], &["cp"], false), SessionStatus::Blocked);
        assert_eq!(
            eval(&["[sudo] password for user:"], &["sudo"], false),
            SessionStatus::Blocked
        );
        assert_eq!(
            eval(&["Continue? "], &["git"], false),
            SessionStatus::Blocked
        );
    }

    #[test]
    fn merely_mentioning_a_prompt_does_not_block() {
        // A transcript line that talks *about* prompts is not a prompt.
        assert_eq!(
            eval(&["the password: field is required in config.toml"], &["cat"], false),
            SessionStatus::Working
        );
        assert_eq!(
            eval(&["print 'are you sure' to ask the user"], &["cargo"], false),
            SessionStatus::Working
        );
    }

    #[test]
    fn sticky_done_until_focus() {
        let lines = vec!["$".to_string()];
        let input = StatusInput {
            lines: &lines,
            title: None,
            progress: None,
            foreground: &[],
        };
        assert_eq!(evaluate(&input, SessionStatus::Done, false), SessionStatus::Done);
        assert_eq!(evaluate(&input, SessionStatus::Done, true), SessionStatus::Idle);
        assert_eq!(evaluate(&input, SessionStatus::Error, false), SessionStatus::Error);
    }

    #[test]
    fn sticky_killed_and_exited_until_focus() {
        let lines = vec!["$".to_string()];
        let input = StatusInput {
            lines: &lines,
            title: None,
            progress: None,
            foreground: &[],
        };
        for sticky in [SessionStatus::Killed, SessionStatus::Exited] {
            assert_eq!(evaluate(&input, sticky, false), sticky);
            assert_eq!(evaluate(&input, sticky, true), SessionStatus::Idle);
        }
    }

    #[test]
    fn active_wins_when_nothing_else() {
        assert_eq!(eval(&["$"], &[], true), SessionStatus::Active);
        assert_eq!(eval(&["$"], &[], false), SessionStatus::Idle);
    }

    #[test]
    fn blocked_beats_active() {
        assert_eq!(eval(&["Delete everything? [y/n]"], &["rm"], true), SessionStatus::Blocked);
    }

    #[test]
    fn verdict_explains_itself() {
        let lines: Vec<String> = ["cli: command not found"].iter().map(|s| s.to_string()).collect();
        let input = StatusInput {
            lines: &lines,
            title: None,
            progress: None,
            foreground: &[],
        };
        let verdict = verdict(&input, SessionStatus::Idle, false);
        assert_eq!(verdict.status, SessionStatus::Error);
        assert_eq!(verdict.reason, Some("failure output"));
        assert!(verdict.rule.is_none());
    }

    #[test]
    fn agent_names_come_from_the_packs() {
        assert!(is_agent_process("claude"));
        assert!(is_agent_process("claude-code"));
        assert!(is_agent_process("/usr/local/bin/codex"));
        assert!(is_agent_process("gemini"));
        assert!(is_agent_process("opencode"));
        assert!(!is_agent_process("claudette"));
        assert!(!is_agent_process("bash"));
        assert!(!is_agent_process("sleep"));
    }

    #[test]
    fn explain_names_the_matching_rule() {
        let lines: Vec<String> = ["✳ Baking… (2m 14s · esc to interrupt)"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let fg = vec!["claude".to_string()];
        let input = StatusInput {
            lines: &lines,
            title: None,
            progress: None,
            foreground: &fg,
        };
        let m = explain(&input).expect("a rule should match");
        assert_eq!(m.pack, "claude");
        assert_eq!(m.state.label(), "working");
        assert!(m.pack_version.is_some());
    }

    #[test]
    fn pack_lookup_reports_the_agent() {
        assert_eq!(pack_for(&["opencode".to_string()]), Some("opencode"));
        assert_eq!(pack_for(&["bash".to_string()]), None);
    }

    #[cfg(unix)]
    #[test]
    fn tpgid_is_field_8_not_pgrp() {
        // Real line captured from the `portable-pty` fish shell this app spawns.
        // Field 5 (`pgrp`) equals the pid, which is why reading field 5 always
        // looked like "the shell is at a prompt"; field 8 (`tpgid`) is 380572,
        // the actual foreground job.
        let stat = "380159 (fish) S 379762 380159 380159 34822 380572 4194304 1445 \
                    5045 8 9 4 3 1 2 16 -4 1 0 5676308 300417024 2121 18446744073709551615";
        assert_eq!(parse_stat_tpgid(stat), Some(380572));
        assert_eq!(parse_stat_pgrp(stat), Some(380159));
    }

    #[cfg(unix)]
    #[test]
    fn tpgid_parses_comm_with_spaces_and_parens() {
        // `comm` is user-controlled and may contain spaces and `)`, so the split
        // has to happen at the *last* paren, not the first.
        let stat = "42 (my (weird) comm) S 1 42 42 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0 0 7";
        assert_eq!(parse_stat_tpgid(stat), Some(-1));
        assert_eq!(parse_stat_pgrp(stat), Some(42));
    }

    #[cfg(unix)]
    #[test]
    fn interpreter_arguments_are_searched_for_agent_names() {
        // npm-installed CLIs look like `node …/claude-code/cli.js`, so the
        // directory components count as candidate names. `claude-code` is an
        // alias the claude pack answers to, and `claude` comes back from the
        // script stem.
        let candidates =
            argument_name_candidates("/usr/lib/node_modules/@anthropic-ai/claude-code/cli.js");
        assert!(candidates.contains(&"claude-code".to_string()), "{candidates:?}");
        assert!(candidates.contains(&"claude".to_string()), "{candidates:?}");
        assert!(candidates.contains(&"cli.js".to_string()), "{candidates:?}");
        assert!(candidates.contains(&"cli".to_string()), "{candidates:?}");
    }

    #[cfg(unix)]
    #[test]
    fn missing_proc_entries_yield_no_names() {
        let mut names = Vec::new();
        read_process_names(Path::new("/nonexistent"), &mut |name| names.push(name));
        assert!(names.is_empty(), "a missing /proc entry yields no names");
    }

    #[cfg(unix)]
    #[test]
    fn basenames_are_portable() {
        assert_eq!(basename("/usr/bin/node"), "node");
        assert_eq!(basename(r"C:\Program Files\node\node.exe"), "node");
    }
}
