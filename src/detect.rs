//! Live terminal status detection, modeled after herdr's detection engine
//! (<https://github.com/herdrdev/herdr>).
//!
//! herdr classifies each terminal pane by combining three signals:
//!
//! 1. **Process monitoring** — it resolves the PTY's foreground process group
//!    (`platform::foreground_job`) and matches process names against a table of
//!    known agent CLIs (`detect::identify_agent`): claude, codex, gemini, …
//! 2. **Output parsing** — it periodically reads the live tail of the screen
//!    (`bottom_non_empty_lines`) plus the OSC title and matches them against
//!    per-agent pattern manifests: spinners (braille `⠋⠙⠹…`, half-circles
//!    `◐◓◑◒`, `✻✳·`), working verbs ("Thinking…", "esc to interrupt"),
//!    approval forms ("Do you want to proceed?", "[y/n]") and idle prompt
//!    boxes (`^\s*❯`).
//! 3. **Process exit** — the foreground job disappearing is the authoritative
//!    "finished" signal; the shell returning to an idle prompt means the agent
//!    exited.
//!
//! This module re-implements the same approach for markup: instead of the
//! PTY's foreground group it walks the process tree below each session's shell
//! using the `sysinfo` crate, and the screen/OSC-title matchers live in
//! [`scan_screen`] and [`title_indicates_working`].

use std::collections::HashMap;

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

/// Result of a process-tree scan for a single terminal session.
#[derive(Debug, Clone, Default)]
pub struct ProcessScan {
    /// A foreground command (anything other than the shell itself) is running.
    pub busy: bool,
    /// Identified AI agent CLI, if any is running (e.g. `claude`, `codex`).
    pub agent: Option<String>,
}

/// Refresh the process table with just the data status detection needs:
/// process names, parent pids (always parsed from procfs) and argv (fetched
/// once per process). Much cheaper than `System::refresh_all`.
pub fn refresh_processes(sys: &mut System) {
    sys.refresh_processes_specifics(
        ProcessesToUpdate::All,
        true,
        ProcessRefreshKind::nothing()
            .with_cmd(UpdateKind::OnlyIfNotSet)
            .without_tasks(),
    );
}

/// Walk all descendants of the session's shell and classify what is running.
pub fn scan_session_processes(
    sys: &System,
    shell_pid: Option<u32>,
    shell_name: &str,
) -> ProcessScan {
    let mut scan = ProcessScan::default();
    let Some(root) = shell_pid.map(Pid::from_u32) else {
        return scan;
    };

    let processes = sys.processes();
    if !processes.contains_key(&root) {
        // The shell itself is gone; nothing can be running in this terminal.
        return scan;
    }

    // Group the process table by parent pid once per scan.
    let mut children: HashMap<Pid, Vec<Pid>> = HashMap::new();
    for (pid, process) in processes {
        if let Some(parent) = process.parent() {
            children.entry(parent).or_default().push(*pid);
        }
    }

    let shell_name = shell_name.to_ascii_lowercase();
    let mut stack = vec![root];
    while let Some(pid) = stack.pop() {
        let Some(kids) = children.get(&pid) else {
            continue;
        };
        for kid in kids {
            stack.push(*kid);
            let Some(process) = processes.get(kid) else {
                continue;
            };
            let name = process.name().to_string_lossy().into_owned();
            let cmd: Vec<String> = process
                .cmd()
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect();
            if is_ignorable(&name, &cmd, &shell_name) {
                continue;
            }
            scan.busy = true;
            if scan.agent.is_none()
                && let Some(agent) = identify_agent_process(&name, &cmd)
            {
                scan.agent = Some(agent.to_string());
            }
        }
    }
    scan
}

/// Interactive shells and session wrappers do not count as "work running".
fn is_shell_name(name: &str) -> bool {
    matches!(
        name,
        "bash"
            | "zsh"
            | "fish"
            | "sh"
            | "dash"
            | "ksh"
            | "tcsh"
            | "csh"
            | "elvish"
            | "nu"
            | "xonsh"
            | "osh"
            | "oil"
            | "ion"
            | "murex"
            | "login"
            | "bash.exe"
            | "zsh.exe"
            | "fish.exe"
            | "sh.exe"
            | "pwsh"
            | "powershell"
            | "powershell.exe"
            | "cmd"
            | "cmd.exe"
    )
}

/// A process that should not count as running work: the session's own shell or
/// another interactive shell sitting at its prompt. A shell *with arguments*
/// (`bash script.sh`, `sh -c …`) is running something and is kept.
fn is_ignorable(name: &str, cmd: &[String], shell_name: &str) -> bool {
    let base = basename(name).to_ascii_lowercase();
    if base == shell_name || is_shell_name(&base) {
        return cmd.len() <= 1;
    }
    false
}

/// Identify which AI agent CLI a process is, mirroring herdr's
/// `detect::identify_agent` + `normalized_process_name`: match the process
/// name first, then argv0, then known runtime wrappers such as
/// `node …/claude-code/cli.js`.
pub fn identify_agent_process(name: &str, cmd: &[String]) -> Option<&'static str> {
    if let Some(agent) = agent_for_name(basename(name)) {
        return Some(agent);
    }
    if let Some(argv0) = cmd.first()
        && let Some(agent) = agent_for_name(basename(argv0))
    {
        return Some(agent);
    }
    // Generic runtime launching an agent script.
    let launcher = cmd.first().map(|arg| basename(arg).to_ascii_lowercase());
    if matches!(
        launcher.as_deref(),
        Some("node" | "nodejs" | "bun" | "deno" | "python" | "python3" | "sh" | "bash" | "zsh")
    ) {
        for arg in cmd.iter().skip(1) {
            if let Some(agent) = agent_for_name(basename(arg)) {
                return Some(agent);
            }
        }
        for (token, label) in AGENT_PATH_TOKENS {
            if cmd
                .iter()
                .skip(1)
                .any(|arg| arg.to_ascii_lowercase().contains(token))
            {
                return Some(label);
            }
        }
    }
    None
}

/// Path fragments that identify agent CLIs launched through a generic runtime
/// (e.g. `node /usr/lib/node_modules/@anthropic-ai/claude-code/cli.js`).
const AGENT_PATH_TOKENS: &[(&str, &str)] = &[
    ("claude-code", "claude"),
    ("@anthropic-ai/claude", "claude"),
    ("gemini-cli", "gemini"),
    ("@google/gemini-cli", "gemini"),
    ("codex-cli", "codex"),
    ("@openai/codex", "codex"),
];

fn basename(s: &str) -> &str {
    std::path::Path::new(s)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(s)
}

/// Canonical agent label for a process/binary basename (subset of herdr's
/// agent table in `crates/herdr/src/detect/mod.rs`).
fn agent_for_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let lower = lower.strip_suffix(".exe").unwrap_or(&lower);
    match lower {
        "claude" | "claude-code" => Some("claude"),
        "codex" => Some("codex"),
        "gemini" => Some("gemini"),
        "cursor" | "cursor-agent" => Some("cursor"),
        "copilot" | "github-copilot" | "ghcs" => Some("copilot"),
        "opencode" | "opencode2" | "open-code" => Some("opencode"),
        "amp" | "amp-local" => Some("amp"),
        "droid" => Some("droid"),
        "grok" | "grok-build" => Some("grok"),
        "qwen" | "qwen-code" => Some("qwen"),
        "kimi" | "kimi-code" => Some("kimi"),
        "kiro" | "kiro-cli" => Some("kiro"),
        "cline" => Some("cline"),
        "aider" => Some("aider"),
        "goose" => Some("goose"),
        "pi" => Some("pi"),
        "kilo" | "kilo-code" | "kilocode" => Some("kilo"),
        "devin" | "devin-cli" => Some("devin"),
        _ => None,
    }
}

/// Signals we can read from the visible tail of the terminal screen.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScreenSignals {
    /// The foreground program is asking for input/approval (herdr's
    /// `visible_blocker` family: "Do you want to proceed?", "[y/n]", …).
    pub blocked: bool,
    /// The bottom of the screen looks like an interactive prompt (herdr's
    /// idle prompt box).
    pub prompt: bool,
    /// A spinner / working verb is visible (herdr's `visible_working`).
    pub working: bool,
    /// The recent output looks like a failure (used to classify an exit).
    pub error_hint: bool,
}

/// Number of trailing non-empty lines inspected for prompt/blocked/working
/// signals (herdr mostly matches `bottom_non_empty_lines(12)`).
const SCREEN_TAIL_LINES: usize = 8;

/// Inspect the visible tail of the screen (plus OSC title is handled in
/// [`title_indicates_working`]).
pub fn scan_screen(lines: &[String]) -> ScreenSignals {
    let mut signals = ScreenSignals::default();

    let tail: Vec<&str> = lines
        .iter()
        .rev()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .take(SCREEN_TAIL_LINES)
        .collect();

    // First decisive line from the bottom wins, mirroring herdr's rule
    // priorities: a blocker beats an idle prompt beats a working spinner.
    for line in &tail {
        let lower = line.to_ascii_lowercase();
        if is_blocked_line(&lower) {
            signals.blocked = true;
            break;
        }
        if is_prompt_line(line) {
            signals.prompt = true;
            break;
        }
        if is_working_line(line, &lower) {
            signals.working = true;
            break;
        }
    }

    // Error hints are scanned over the whole recent tail, independently.
    for line in &tail {
        let lower = line.to_ascii_lowercase();
        if error_pattern_hit(&lower) {
            signals.error_hint = true;
            break;
        }
    }

    signals
}

const BLOCKED_PATTERNS: &[&str] = &[
    "esc to cancel",
    "[y/n]",
    "(y/n)",
    "[y/n",
    "allow tool execution?",
    "do you want to proceed?",
    "do you want to run:",
    "do you want to allow",
    "press enter to continue",
    "press enter to confirm",
    "permission required",
    "allow this command?",
    "waiting for permission",
    "waiting for your approval",
];

fn is_blocked_line(lower: &str) -> bool {
    BLOCKED_PATTERNS.iter().any(|p| lower.contains(p))
}

fn is_prompt_line(line: &str) -> bool {
    line.starts_with('❯')
        || line.starts_with('▸')
        || line.starts_with('➜')
        || line.ends_with('$')
        || line.ends_with('>')
        || line.ends_with('#')
        || line.ends_with('%')
}

fn is_working_line(line: &str, lower: &str) -> bool {
    if lower.contains("esc to interrupt")
        || lower.contains("working…")
        || lower.contains("working...")
        || lower.contains("thinking")
        || lower.contains("generating")
        || lower.contains("executing")
        || lower.contains("synthesizing")
        || lower.contains("baking")
        || lower.contains("running…")
        || lower.contains("running...")
    {
        return true;
    }
    line.chars().next().is_some_and(is_spinner_char)
}

/// Spinner glyphs used by agent CLIs (braille range covers Claude ≤ 2.1.227,
/// half-circles cover the newer busy spinner; `✻✳✽·` are Claude/Codex idle
/// and working marks — see herdr's `claude.toml` manifest).
pub fn is_spinner_char(c: char) -> bool {
    matches!(c, '◐' | '◓' | '◑' | '◒' | '·' | '✢' | '✳' | '✶' | '✻' | '✽' | '⏸' | '⏵')
        || ('\u{2800}'..='\u{28FF}').contains(&c)
}

/// OSC titles carrying spinner glyphs or working verbs mean the agent is
/// working even when its process lives somewhere we can't see (ssh,
/// container). Mirrors herdr's `osc_title_working` rule.
pub fn title_indicates_working(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("working")
        || lower.contains("thinking")
        || lower.contains("generating")
        || lower.contains("executing")
        || lower.contains("baking")
    {
        return true;
    }
    trimmed.chars().next().is_some_and(is_spinner_char)
}

/// Output patterns that indicate the command that just finished failed.
const ERROR_PATTERNS: &[&str] = &[
    "error:",
    "error[",
    "panic",
    "traceback (most recent call last",
    "command not found",
    "segmentation fault",
    "core dumped",
    "fatal: ",
    "assertion failed",
    "permission denied",
    "exception in thread",
    "syntax error",
    "killed",
    "interrupt",
    "^c",
    "cancelled",
    "canceled",
    "aborted",
    "zsh: exit",
];

fn error_pattern_hit(lower: &str) -> bool {
    if ERROR_PATTERNS.iter().any(|p| lower.contains(p)) {
        return true;
    }
    // "exit code N" / "exit status N" with N != 0 (success messages like
    // "exit code 0" must not classify as errors).
    for marker in ["exit code ", "exit status ", "exited with code "] {
        if let Some(idx) = lower.find(marker)
            && let Some(c) = lower[idx + marker.len()..].chars().next()
            && c.is_ascii_digit()
            && c != '0'
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn detects_blocked_approval_prompt() {
        let s = scan_screen(&lines(&["some output", "Do you want to proceed? [y/n]"]));
        assert!(s.blocked);
    }

    #[test]
    fn detects_idle_prompt() {
        let s = scan_screen(&lines(&["$ ls", "file.txt", "user@host ~ %"]));
        assert!(s.prompt);
        assert!(!s.blocked && !s.working);
    }

    #[test]
    fn detects_spinner_working() {
        let s = scan_screen(&lines(&["⠙ Thinking… (3s · esc to interrupt)"]));
        assert!(s.working);
    }

    #[test]
    fn detects_error_hint() {
        let s = scan_screen(&lines(&["error: could not compile `demo`"]));
        assert!(s.error_hint);
        let ok = scan_screen(&lines(&["exit code 0"]));
        assert!(!ok.error_hint);
    }

    #[test]
    fn title_spinner_means_working() {
        assert!(title_indicates_working("⠹ claude"));
        assert!(title_indicates_working("Working…"));
        assert!(!title_indicates_working("user@host: ~"));
    }

    #[test]
    fn identifies_agents_by_process_name() {
        assert_eq!(identify_agent_process("claude", &["claude".into()]), Some("claude"));
        assert_eq!(identify_agent_process("codex", &["codex".into()]), Some("codex"));
        assert_eq!(
            identify_agent_process(
                "node",
                &["node".into(), "/usr/lib/node_modules/@anthropic-ai/claude-code/cli.js".into()]
            ),
            Some("claude")
        );
        assert_eq!(identify_agent_process("vim", &["vim".into(), "main.rs".into()]), None);
    }
}
