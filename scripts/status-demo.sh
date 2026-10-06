#!/usr/bin/env bash
# Simulate each terminal status for markup's live status badges.
#
# Run these inside one of markup's terminals, then click another terminal
# (or another session in the sidebar) and watch the badge update live:
#
#   ./scripts/status-demo.sh working [seconds]  Working (yellow): spinner + progress, default 30s
#   ./scripts/status-demo.sh blocked            Blocked (orange): asks for approval and waits for input
#   ./scripts/status-demo.sh error              Error (red): fails with error output, exits 3
#   ./scripts/status-demo.sh done               Done (green): finishes successfully, exits 0
#   ./scripts/status-demo.sh stuck              Blocked (orange): runs silently >45s -> flagged stuck
#
#   Active (blue):  just focus a terminal and type — nothing else applies.
#   Idle  (gray):   leave a terminal at the shell prompt and focus another one.
#
# Notes:
# - Done/Error badges clear when you focus that terminal, or automatically
#   after ~10 seconds.
# - To test AI agent CLI detection by process name, run one of the fake
#   agents instead (badge then shows e.g. "working · claude"):
#
#       PATH="$PWD/scripts/fake-agents:$PATH" claude

set -uo pipefail

mode="${1:-working}"
frames=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)

case "$mode" in
  working)
    seconds="${2:-30}"
    echo "Simulating a working command for ${seconds}s — switch away and watch the badge."
    end=$(( $(date +%s) + seconds ))
    i=0
    SECONDS=0
    while [ "$(date +%s)" -lt "$end" ]; do
      printf '\r  %s Working… (%ss · esc to interrupt)        ' "${frames[i%10]}" "$SECONDS"
      i=$((i + 1))
      sleep 0.15
    done
    printf '\n'
    echo "✔ finished cleanly -> badge should flash done (green), then fade to idle"
    ;;

  blocked)
    echo "Simulating an agent that needs approval…"
    sleep 1
    echo "  → Deploy branch 'main' to production?"
    printf 'Do you want to proceed? [y/n] '
    read -r answer
    echo "You answered: ${answer:-<nothing>} -> badge was blocked (orange) while waiting"
    ;;

  error)
    echo "Simulating a failing command…"
    sleep 2
    echo "error: simulated failure — something exploded" >&2
    echo "Traceback (most recent call last):" >&2
    echo "  File \"task.py\", line 1, in <module>" >&2
    exit 3
    ;;

  done)
    echo "Simulating a successful task…"
    sleep 2
    echo "✔ Task completed successfully -> badge should show done (green)"
    ;;

  stuck)
    echo "Simulating a stuck/silent command (no output for 45+ seconds)…"
    echo "After ~45s of silence the badge flips from working to blocked."
    sleep 600
    ;;

  *)
    echo "usage: $0 {working [seconds]|blocked|error|done|stuck}" >&2
    exit 2
    ;;
esac
