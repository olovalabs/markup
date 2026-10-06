//! Live terminal status detection.
//!
//! Detection runs in two tiers, in this order:
//!
//! 1. **Agent rule packs.** If an agent CLI owns the shell's foreground process
//!    group, its herdr rule pack (`src/agent_rules.rs`) decides the state from
//!    the screen, the OSC title, and the OSC progress field. This is where
//!    `working` and `blocked` come from, and it is the only tier that
//!    understands an agent's own UI.
//! 2. **Plain-shell fallback.** With no agent running, a foreground command
//!    means `Working`; a `[y/n]` prompt means `Blocked`; failure output means
//!    `Error`.
//!
//! The foreground group is read from `/proc/<pid>/stat` field 8 (`tpgid`). Note
//! that a `portable-pty` shell is its own process-group leader, so `pgrp == pid`
//! while `tpgid` is the job actually running — reading the wrong field here once
//! made the whole process half of detection silently inert.
//!
//! herdr has no `error` state, so `Error` is ours: it comes from the terminal's
//! real exit code (the host applies that in `poll_sessions_status`) and, for
//! plain shells, from failure text on screen. Deliberately *not* applied while
//! an agent pack is active, because agent transcripts are full of `error:` lines
//! belonging to the code being discussed rather than to the terminal.
//!
//! Sticky states: `Done` and `Error` persist while a terminal is unfocused and
//! clear when the user looks at it, so a completion stays visible until it has
//! been seen.
//!
//! Platform note: the process-tree half is Linux-only (`/proc`). Elsewhere
//! [`foreground_process_names`] returns nothing and detection rests on the
//! screen and OSC rules alone.

use std::path::Path;

use crate::agent_rules::{self, DetectionInput, RuleState};

/// The status a terminal session can report. Exactly one is active at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// User is currently focused/typing in this terminal.
    Active,
    /// Not focused, nothing is running.
    Idle,
    /// A command or AI agent CLI is running.
    Working,
    /// Waiting for user input / approval / stuck.
    Blocked,
    /// Process failed (non-zero exit, crash, error output).
    Error,
    /// Task finished successfully.
    Done,
}

impl SessionStatus {
    /// Render priority: Error > Blocked > Working > Done > Active > Idle.
    fn rank(self) -> u8 {
        match self {
            SessionStatus::Error => 6,
            SessionStatus::Blocked => 5,
            SessionStatus::Working => 4,
            SessionStatus::Done => 3,
            SessionStatus::Active => 2,
            SessionStatus::Idle => 1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SessionStatus::Active => "active",
            SessionStatus::Idle => "idle",
            SessionStatus::Working => "working",
            SessionStatus::Blocked => "blocked",
            SessionStatus::Error => "error",
            SessionStatus::Done => "done",
        }
    }

    /// (dot, text, badge background) colors, from the spec.
    pub fn colors(self) -> (u32, u32, u32) {
        match self {
            SessionStatus::Active => (0x58a6ffff, 0x58a6ffff, 0x1f6feb33),
            SessionStatus::Idle => (0x8b949eff, 0x8b949eff, 0x21262d88),
            SessionStatus::Working => (0xd29922ff, 0xd29922ff, 0x9e6a0344),
            SessionStatus::Blocked => (0xdb6d28ff, 0xdb6d28ff, 0xbd561444),
            SessionStatus::Error => (0xf85149ff, 0xf85149ff, 0xda363344),
            SessionStatus::Done => (0x3fb950ff, 0x3fb950ff, 0x23863644),
        }
    }
}

/// All inputs needed for one detection pass.
pub struct StatusInput<'a> {
    /// Visible screen rows, top to bottom.
    pub lines: &'a [String],
    /// OSC 0/2 title, if the application set one.
    pub title: Option<&'a str>,
    /// Names in the shell's foreground process group. Empty at a prompt.
    pub foreground: &'a [String],
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
            osc_progress: "",
        },
    )
}

/// Candidate names for the process in the shell's foreground process group.
///
/// Three sources, because no single one is enough:
/// * `/proc/<pid>/comm` — set from the executable name (catches symlinked CLIs),
/// * `argv[0]` basename — reflects `exec -a` renames,
/// * `argv[1]` basename — where npm-installed agents live (`node …/claude`).
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

    let dir = Path::new("/proc").join(tpgid.to_string());
    let mut names = Vec::with_capacity(3);

    let mut push = |name: String| {
        let name = name.trim().to_string();
        if !name.is_empty() && !names.contains(&name) {
            names.push(name);
        }
    };

    if let Ok(comm) = std::fs::read_to_string(dir.join("comm")) {
        push(comm);
    }

    if let Ok(cmdline) = std::fs::read(dir.join("cmdline")) {
        fn basename(arg: &[u8]) -> String {
            String::from_utf8_lossy(arg)
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_string()
        }
        let args: Vec<&[u8]> = cmdline.split(|&b| b == 0).filter(|a| !a.is_empty()).collect();
        if let Some(argv0) = args.first() {
            push(basename(argv0));
        }
        if let (Some(argv0), Some(argv1)) = (args.first(), args.get(1)) {
            const INTERPRETERS: &[&str] =
                &["node", "nodejs", "python", "python3", "deno", "bun", "ruby", "perl", "sh", "bash"];
            let interpreter = basename(argv0).to_ascii_lowercase();
            if INTERPRETERS.contains(&interpreter.as_str()) {
                push(basename(argv1));
            }
        }
    }

    names
}

#[cfg(not(unix))]
pub fn foreground_process_names(_pid: Option<u32>) -> Vec<String> {
    Vec::new()
}

/// The `tpgid` field of `/proc/<pid>/stat`: the terminal's foreground process
/// group, i.e. whoever is *running right now* under this shell.
///
/// Parsing needs care. The fields are space-separated **after** a
/// parenthesised `comm`, which may itself contain spaces and parens, so the
/// only safe split is on the **last** `)`. Counting from there the fields are
/// 1-indexed as:
///
/// ```text
/// 3 state | 4 ppid | 5 pgrp | 6 session | 7 tty_nr | 8 tpgid
/// ```
///
/// Reading field 5 here was a real bug — that is `pgrp`, and a `portable-pty`
/// shell is its own process-group leader, so `pgrp == pid` and the guard below
/// rejected every result. The whole process-tree half of detection was silently
/// dead: a running CLI could never be reported as `Working`.
#[cfg(unix)]
fn read_proc_tpgid(pid: u32) -> Option<i64> {
    let stat = std::fs::read_to_string(Path::new("/proc").join(pid.to_string()).join("stat")).ok()?;
    parse_stat_tpgid(&stat)
}

/// Field 8 (`tpgid`) of one `/proc/<pid>/stat` line.
#[cfg(unix)]
fn parse_stat_tpgid(stat: &str) -> Option<i64> {
    let after_comm = stat.rsplit_once(')')?.1;
    // This slice starts at field 3 (state), so tpgid at field 8 is index 5.
    after_comm.split_whitespace().nth(8 - 3)?.parse().ok()
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
    // Sticky states are cleared on focus, otherwise they persist.
    if matches!(prev, SessionStatus::Done | SessionStatus::Error) {
        return if focused { SessionStatus::Idle } else { prev };
    }

    let detected = detect(input, prev);

    // Priority resolution with focus folded in as the lowest rung.
    let focus_status = if focused { SessionStatus::Active } else { SessionStatus::Idle };
    if detected.rank() > focus_status.rank() {
        detected
    } else {
        focus_status
    }
}

/// Pure detection (no stickiness): agent packs, then the plain-shell fallback.
fn detect(input: &StatusInput<'_>, prev: SessionStatus) -> SessionStatus {
    let screen = input.lines.join("\n");

    let Some(pack) = agent_rules::embedded().pack_for(input.foreground) else {
        // No agent CLI is driving this terminal, so the only signals available
        // are generic ones.
        if FallbackRules::error(input) {
            return SessionStatus::Error;
        }
        if FallbackRules::plain_shell_blocked(input) {
            return SessionStatus::Blocked;
        }
        return if input.foreground.is_empty() {
            SessionStatus::Idle
        } else {
            SessionStatus::Working
        };
    };

    let matched = agent_rules::evaluate(
        pack,
        DetectionInput {
            screen: &screen,
            osc_title: input.title.unwrap_or(""),
            // OSC 9;4 progress is not parsed yet; the rules that read it simply
            // never match, which is how upstream treats a terminal that does
            // not report progress.
            osc_progress: "",
        },
    );

    match matched {
        // A known agent with no matching rule is at rest, per upstream's
        // known-agent fallback.
        None => SessionStatus::Idle,
        // Transient overlays (transcript viewer, model picker) match only to say
        // "ignore me" — leave the badge where it was.
        Some(m) if m.skip_state_update => prev,
        Some(m) => match m.state {
            RuleState::Blocked => SessionStatus::Blocked,
            RuleState::Working => SessionStatus::Working,
            RuleState::Idle => SessionStatus::Idle,
            // Unknown means "neither" — but the agent process really is
            // running, so report that rather than claiming the pane is idle.
            RuleState::Unknown => SessionStatus::Working,
        },
    }
}

/// Generic rules for terminals with no agent pack: a plain shell.
struct FallbackRules;

impl FallbackRules {
    /// Lines considered "recent" for the fallback rules.
    const WINDOW: usize = 12;

    /// The newest `n` non-empty lines of the visible screen.
    fn recent<'a>(input: &'a StatusInput<'a>, n: usize) -> Vec<&'a str> {
        let mut lines: Vec<&str> = input
            .lines
            .iter()
            .map(String::as_str)
            .filter(|line| !line.trim().is_empty())
            .collect();
        let excess = lines.len().saturating_sub(n);
        lines.split_off(excess)
    }

    /// The classic interactive prompt a plain command blocks on.
    ///
    /// Deliberately narrow: agents have their own packs, and vague wording here
    /// only produces false positives on ordinary output.
    fn plain_shell_blocked(input: &StatusInput) -> bool {
        let text = Self::recent(input, Self::WINDOW).join("\n").to_lowercase();
        text.contains("[y/n]")
            || text.contains("(y/n)")
            || text.contains("yes (y)")
            || text.contains("password:")
    }

    /// Strong failure markers on the most recent output lines.
    ///
    /// Scans a small window rather than just the last line, because a shell
    /// redraws its prompt *below* the failed command's output.
    fn error(input: &StatusInput) -> bool {
        Self::recent(input, 4).iter().any(|line| {
            let l = line.trim().to_lowercase();
            l.contains("command not found")
                || l.contains("permission denied")
                || l.contains("panicked at")
                || l.contains("segmentation fault")
                || l.contains("traceback (most recent call last)")
                || l.starts_with("error:")
                || l.starts_with("error ")
                || l.starts_with("fatal:")
                || l.starts_with("panic:")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(lines: &[&str], foreground: &[&str], focused: bool) -> SessionStatus {
        let lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        let fg: Vec<String> = foreground.iter().map(|s| s.to_string()).collect();
        let input = StatusInput {
            lines: &lines,
            title: None,
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
        assert!(SessionStatus::Error.rank() > SessionStatus::Blocked.rank());
        assert!(SessionStatus::Blocked.rank() > SessionStatus::Working.rank());
        assert!(SessionStatus::Working.rank() > SessionStatus::Done.rank());
        assert!(SessionStatus::Done.rank() > SessionStatus::Active.rank());
        assert!(SessionStatus::Active.rank() > SessionStatus::Idle.rank());
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
    fn claude_at_its_prompt_is_idle_even_though_the_process_runs() {
        // The pack wins over the foreground fallback: an agent sitting at its
        // prompt is Idle even though its TUI is the running process.
        assert_eq!(
            eval(CLAUDE_IDLE_BOX, &["claude"], false),
            SessionStatus::Idle
        );
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
    }

    #[test]
    fn sticky_done_until_focus() {
        let lines = vec!["$".to_string()];
        let input = StatusInput {
            lines: &lines,
            title: None,
            foreground: &[],
        };
        assert_eq!(evaluate(&input, SessionStatus::Done, false), SessionStatus::Done);
        assert_eq!(evaluate(&input, SessionStatus::Done, true), SessionStatus::Idle);
        assert_eq!(evaluate(&input, SessionStatus::Error, false), SessionStatus::Error);
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
    }

    #[cfg(unix)]
    #[test]
    fn tpgid_parses_comm_with_spaces_and_parens() {
        // `comm` is user-controlled and may contain spaces and `)`, so the split
        // has to happen at the *last* paren, not the first.
        let stat = "42 (my (weird) comm) S 1 42 42 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0 0 7";
        assert_eq!(parse_stat_tpgid(stat), Some(-1));
    }
}
