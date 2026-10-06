# Vendored agent-detection rule packs

The `*.toml` files in this directory are copied **unmodified** from the herdr
project and are licensed under the Apache License, Version 2.0.

- Upstream: <https://github.com/herdrdev/herdr>
- Directory: `distribution/agent-detection/`
- Revision: `fabcab108f9db383e3cd6889f254c8d0d66dcd7b` (2026-10-05)
- License: Apache-2.0 — <https://github.com/herdrdev/herdr/blob/master/LICENSE>
- Copyright: the herdr authors

They are evaluated by `src/agent_rules.rs`, which is a port of herdr's
`src/detect/manifest.rs` (same manifest schema, region slicing, and gate
semantics) so that the packs keep working exactly as upstream intends.

## Updating

Drop in a newer pack (or a new agent's pack) and it is picked up with no Rust
changes, as long as its `id`/`aliases` match the process name of the agent CLI.
`index.toml` is upstream's registry; this crate instead auto-discovers every
`agent-detection/*.toml` embedded under `assets/`.

If the packs are ever distributed as part of a binary release, include the full
Apache-2.0 license text alongside them, as the license requires.
