# Agent status integrations

Command Code uses its built-in Herdr reporter through Markup's scoped reporting compatibility endpoint. New Markup panes export `HERDR_ENV`, `HERDR_SOCKET_PATH`, an opaque `HERDR_PANE_ID` bound to the pane generation, and `HERDR_BIN_PATH`. No Command Code mod is needed. Only native reporting is supported; Herdr's general pane control, metadata, and session restoration API are outside this compatibility interface. Native initial idle means ready; a working/blocked → idle transition settles a turn and latches the checkmark. The protocol does not distinguish every reason an agent became idle.

## Status semantics (Herdr parity)

The badge follows Herdr's attention rules on top of the two-tier detection:

- Priority is `blocked > done > working > idle`. `Error` ranks as blocked and
  `Unknown`/`Active` rank as idle for rollups and the Agents list.
- A busy agent that finishes while its screen is not visible (or the window is
  inactive) becomes `done`. Looking at it — selecting its session, focusing it,
  or reactivating the window — clears `done` to `idle` automatically.
- A finish seen live never becomes `done`; it goes straight to `idle`/`active`.
  `Error` is a live blocked state and does not auto-clear; use the badge or
  `markup status clear`.
- Only an agent rule pack reporting `working`, or a native/OSC `working` report,
  keeps a focused pane on `working`. Generic busy text reads `active` while you
  look at it and `working` once you switch away.
- Needing you (blocked/error) always plays the alert sound; the toast appears
  only when you were not looking. A missed finish plays the done sound and shows
  a toast. `MARKUP_NO_SOUND=1` disables sounds.
- The sidebar **Agents** section lists reporting panes under Needs you / Done /
  Working / Idle, most recently changed first. Workers reported with an
  `agent_id` are indented under their parent and never take the parent's status
  or raise their own alert.

OpenCode uses the unmodified upstream Herdr TUI integration pinned at `bc8c28988f75c538d4ca058b702f5adbea30566e`. It follows the displayed session, listens using the actual V2 `api.data.listen` interface, reads initial string-valued session status, tracks child-session approvals, and rejects stale selection work. One-click setup registers its TUI entry in OpenCode's configuration; copying plugin files alone is insufficient. V2 registration uses `cli.json.plugins`; legacy TUI configuration is preserved and migration can require a second enablement after restart. See [`assets/integrations/NOTICE.md`](assets/integrations/NOTICE.md) for source and license details.

Generic screen detection treats a static busy marker as uncertain after five seconds without changes to the marker itself. Unrelated output cannot keep an old transcript spinner active. This restores Unknown; it never fabricates a completion check from silence. Agent-specific rules and explicit reports retain their own authority.

Markup displays reported agent activity in the sidebar. A completion or error stays visible until a new task starts or its report is cleared.

Enable the native integrations from the sidebar's **Agent integrations** panel or run `markup integration setup` once, then restart the agent CLIs in a new Markup terminal. `markup integration status` and `markup integration doctor` report installation separately from live reporting; installed hooks are not considered verified until a real event arrives. `markup integration remove` removes only Markup's entries. Configuration changes are backed up before replacement. During development the command can be run as `cargo run -- integration setup`.

Setup adds Markup hooks to Claude Code's `settings.json` and Codex's `hooks.json`, and installs owned extension files for Pi, OMP, and OpenCode v2. Command Code uses its vendor-supplied reporter without a configuration edit. Existing settings and other hook entries are preserved. Codex may require reviewing and trusting the new hooks in `/hooks`; agent configuration or organizational policy can also disable them. OpenCode's TUI plugin supports its v2 plugin API. If Pi and OMP resolve to the same extension directory, setup stops and asks for a separate `OMP_AGENT_DIR` so one extension cannot overwrite the other.

Hooks and extensions report only in terminals launched by Markup. The app injects a private, per-user Unix socket address plus app, pane, and pane-generation identities. The app accepts a report only for the live pane generation and ignores duplicate/out-of-order reporter sequences. Shared services that do not inherit a pane's Markup environment cannot be routed automatically; they need a small per-TUI bridge that forwards the exact native session ID. SSH, containers, and nested tmux do not relay the local socket.

Unsupported agents can report through the compatible OSC protocol using `markup report working`, `markup report blocked`, `markup report done`, `markup report error`, and `markup report clear`. Native extensions can invoke `markup status-event` with one JSON event on stdin:

```json
{"version":1,"reporter":"my-agent","reporter_generation":"process-run-id","sequence":1,"session_id":"native-session-id","turn_id":"turn-id","event":"working"}
```

The executable supplies the app and pane identity from the environment and the app verifies those values. Supported lifecycle events are `register`, `idle`, `working`, `blocked`, `approval_resolved`, `done`, `error`, `cancelled`, `clear`, and `heartbeat`. Use an increasing `sequence` for each reporter generation. Include a native session ID and turn ID when available. Never infer `done` from silence. `markup status explain --json` prints the latest content-free state/evidence snapshot; `markup status clear` clears the pane's reported state. Unknown unfamiliar programs remain Unknown when the terminal cannot tell whether they are working, waiting, or complete.

Herdr screen manifests can be updated explicitly with `markup integration update-rules`; downloaded packs are validated against the supported engine version and loaded ahead of the bundled pack only when newer. User files under `~/.config/markup/agent-detection/` override both. The current bundled rules are attributed in [`assets/agent-detection/NOTICE.md`](assets/agent-detection/NOTICE.md).

Verification commands: `cargo test`, `cargo test -p gpui-terminal`, and `bun test tests/integrations/opencode-tui.test.ts`. The upstream TUI tests cover SDK boundaries, session changes, blockers, delayed deliveries, and stale selections. These tests validate the bridge; they do not label a locally installed agent version as live verified.
