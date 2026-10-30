# Changelog

All notable changes to Markup will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-30

### Added
- **GPU-Accelerated Terminal Emulator**:
  - Embedded `terminal` component utilizing GPUI and `alacritty_terminal` for VTE parsing.
  - High-performance 1 ms stream coalescing with a 128 KiB parse cap to preserve UI framerates at high log throughput.
  - Full support for ANSI colors (standard 16, 256-color, 24-bit RGB), text styles, and embedded Mononoki monospace fonts.
  - OSC 52 clipboard copy and paste integration.
- **Agent Attention & Status System**:
  - Parity with the Herdr agent attention model (`blocked > done > working > idle`).
  - Latching completion tracking: agent runs that finish in an unviewed tab switch to `done` and alert the user; focusing the tab clears status to `idle`.
  - Audio chimes and desktop toast notifications for blocked queries and missed finishes (`MARKUP_NO_SOUND=1` support).
  - Sub-agent hierarchy support: child sessions with `agent_id` nest neatly under parent sessions without duplicate notifications.
- **Multi-Project Workspace Management**:
  - Collapsible project folders in the sidebar grouping related terminal sessions.
  - Workspace state persistence to `~/.config/markup/workspace.json`, automatically restoring open projects and session trees across application restarts.
  - Project-level actions for adding sessions and opening directories.
- **Native Agent Integrations & CLI**:
  - One-command setup CLI (`markup integration setup`, `status`, `doctor`, `remove`).
  - Native hook support for Claude Code, Codex, Pi, OMP, OpenCode v2, and Command Code (Herdr compatibility protocol).
  - Terminal CLI commands: `markup report`, `markup skill`, `markup status`, `markup pane`, and OSC 7501 ANSI sequence reporting.
- **Desktop Integration**:
  - Custom title bar and window decoration controls.
  - Automatic Linux desktop entry (`.desktop`) creation in `~/.local/share/applications/`.
  - Taskbar icon support for Linux, macOS, and Windows.
  - GitHub Dark palette theme.
