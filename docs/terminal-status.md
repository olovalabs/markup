# Live terminal status badges

Every terminal shows exactly one color-coded status at a time, visible in the
sidebar list (dot + badge) so you can see what each terminal is doing at a
glance — including terminals you are not looking at.

| Status  | Meaning                                                              | Color  |
|---------|----------------------------------------------------------------------|--------|
| active  | You are currently focused/typing in this terminal                    | Blue   |
| idle    | Not focused and nothing is running                                   | Gray   |
| working | An AI agent CLI or command is running (persists across focus change) | Yellow |
| blocked | The process is waiting for your input/approval, or is stuck          | Orange |
| error   | The process failed (non-zero exit, crash, or error output)           | Red    |
| done    | The last task finished successfully                                  | Green  |

Priority when several signals apply: **Error > Blocked > Working > Done >
Active > Idle**.

## How detection works

The design mirrors [herdr](https://github.com/herdrdev/herdr), which combines
three signal sources to classify each pane:

1. **Process monitoring** — every 500 ms markup walks the process tree below
   each session's shell (`sysinfo` crate, `src/detect.rs`). Any descendant
   that is not an interactive shell means a command is running → `working`.
   Process names are matched against a table of known AI agent CLIs (herdr's
   `identify_agent` table: `claude`, `codex`, `gemini`, `cursor-agent`,
   `copilot`, `opencode`, `amp`, `aider`, …), including runtime wrappers such
   as `node …/@anthropic-ai/claude-code/cli.js`. The identified agent name is
   shown in the badge (e.g. `working · claude`).

2. **Output parsing** — the visible tail of the screen plus the OSC window
   title are matched against the same pattern families herdr uses: approval
   forms (`Do you want to proceed?`, `[y/n]`, `esc to cancel`, …) →
   `blocked`; spinners (braille `⠋⠙⠹…`, `◐◓◑◒`, `✻✳·`) and working verbs
   (`Thinking…`, `esc to interrupt`) in the output or title → `working`;
   a live prompt line (`❯`, `… $`, `… %`, …) → idle screen.

3. **Process exit** — the authoritative completion signal (herdr treats the
   foreground job disappearing as "finished"). When the running process
   disappears, the recent output is scanned for failure patterns
   (`error:`, `panic`, `Traceback…`, `^C`, …) → `error`, otherwise → `done`.
   When the shell itself exits, the PTY exit status decides: code 0 → `done`,
   non-zero → `error`.

Extra behaviors:

- **Working persists across focus changes.** Status is computed per session
  from process + screen signals; focus only ever adds/removes `active`.
- **Done/Error clear** when you focus that terminal, or automatically after
  ~10 s (`FINISHED_HOLD`).
- **Stuck heuristic**: a running command that produces no output and shows no
  visible working signal for 45 s (`STUCK_AFTER`) is flagged `blocked`.
- Spinner-only signals (OSC title or on-screen spinners) keep a terminal
  `working` even when the process lives somewhere markup cannot see
  (SSH, containers) — the same trade-off herdr makes with `osc_title_working`.

## Testing each state

See `scripts/status-demo.sh` and `scripts/fake-agents/`:

```sh
./scripts/status-demo.sh working 30   # yellow, persists while you switch terminals
./scripts/status-demo.sh blocked      # orange, waiting on [y/n]
./scripts/status-demo.sh error        # red (error output, exit 3)
./scripts/status-demo.sh done         # green, fades back to gray
./scripts/status-demo.sh stuck        # yellow -> orange after ~45s silent
PATH="$PWD/scripts/fake-agents:$PATH" claude   # "working · claude" by process name
```

`active` (blue): focus a terminal and type. `idle` (gray): leave a terminal
at its prompt and focus another one. `exit 1` in a shell produces `error` via
the real PTY exit code.
