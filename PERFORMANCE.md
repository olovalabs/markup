# Performance audit — markup / gpui-terminal

Date: 2026-10-29
Base commit: `d53bfbc` (main)

This report covers the application (`src/`), the local terminal component
(`gpui-terminal/`), the vendored GPUI patch (`components/gpui/`), and the
dependency/build setup. Linux was measured end to end; macOS and Windows were
reviewed from source only (no access to those machines).

---

## 1. Environment and method

| | |
|---|---|
| OS | CachyOS, Linux 7.3.0-4-cachyos x86_64 |
| Desktop | X11 (Xfwm4) |
| CPU / GPU | AMD Ryzen 5 5600G (6C/12T) / Radeon Vega series (amdgpu) |
| RAM | 13.5 GiB |
| Display | 1920x1080 @ 60 Hz |
| Binary | `target/release/markup`, 25 MB, stripped |
| Profile | `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `strip = true`, `panic = "unwind"` |

**Instrumentation.** Two independent measurements were used:

1. **In-app, opt-in** (`MARKUP_DIAGNOSTICS=1`): one line per second with frame
   count and worst inter-frame gap, render/paint/PTY-parse/poll/`/proc` timing,
   process CPU (utime+stime), RSS, thread count and context switches. Disabled
   by default; when off every hook is one relaxed atomic load and no sampling
   task exists.
2. **External `/proc` sampler** at 1 Hz reading utime/stime, RSS, thread count
   and voluntary+involuntary context switches of the process. This measures the
   whole process, including threads the in-app report cannot see.

Workloads were driven with `xdotool`/`wmctrl` against the real window:
idle, typing (`echo line N` every 30 ms), sustained output
(`yes markup-perf-line | head -c 40000000`, ~40 MB), resize (240 alternating
window sizes at ~30/s), scroll (600 wheel notches over 3 s after 200k lines of
history), agent-like static "working" screen, and multi-pane (3 splits + one
pane streaming).

Numbers are 1 Hz samples; CPU has 10 ms (1 tick) resolution, so "0%" means fewer
than one tick in that second. The desktop was not idle-quarantined, so treat
sub-1% differences as noise. Each workload ran 6-12 s.

---

## 2. Results

### 2.1 Idle, one pane

| Metric | Before | After |
|---|---|---|
| Process CPU | 0-1% (0 ticks most seconds) | 0-1% (0 ticks most seconds) |
| RSS | 79.5 MB | 78.9-79.6 MB |
| Threads | 23 | 23 |
| Context switches | 62-63 /s | 65 /s |
| Frames rendered | **0 /s** | **0 /s** |
| Status poll, UI thread | 0.030-0.044 ms/tick | **0.005 ms/tick** |
| `/proc` inspection | 0.013-0.018 ms x 2 calls, **on UI thread** | 0.023-0.030 ms x 1 call, **background executor** |

The idle path was already not repainting: GPUI's X11 backend wakes at monitor
refresh (`src/platform/linux/x11/client.rs`, `start_refresh_loop`) but only
draws when a view is dirty, so the 60-65 wakeups/s cost no measurable CPU.
This is upstream X11 design, not an app bug, and was left alone.

### 2.2 Sustained output (~40 MB burst)

| Metric | Before coalescing | After adaptive coalescing |
|---|---|---|
| Throughput | 38 MB/s | 39 MB/s |
| PTY parse batches /s | **34,994** | **5,552** |
| Parse avg / max | 0.02 / **2.70 ms** | 0.10 / **1.47 ms** |
| Frames in peak second | 48-49 fps | **53 fps** |
| Process CPU, peak second | 107-117% | 100-107% (unchanged within noise) |
| Terminal paint avg | 0.16-0.31 ms | 0.12 ms |
| Status poll | 0.059 ms/tick | 0.042 ms/tick |

"Before" here means the code after the poll refactor but before the coalescing
pause; the 128 KiB cap alone never bound (largest queued batch was 76 KiB), so
this is effectively the original reader behaviour. Most of the CPU is
`alacritty_terminal` parsing, not painting: ~0.1-0.7 CPU-seconds per 40 MB
depending on batch shape, while paint stays around 0.12 ms/frame.

### 2.3 Interaction

| Workload | Before | After |
|---|---|---|
| Typing, 400 lines at ~30/s | 11-12% CPU | 10-12% CPU |
| Window resize, 240 steps at ~30/s | 14-15% CPU, ~795 ctxt/s | 14-15% CPU, ~800 ctxt/s |
| Scroll, 600 notches after 200k lines | burst-bound (same paint path) | 2% CPU over the 3 s burst |
| First-frame paint at startup | 1.8-2.0 ms (max 11 ms) | same |

No regressions in input or resize. The 1 ms coalescing pause only applies when
the reader has caught up with the PTY, so a keystroke echo can pay up to +1 ms
before parsing, which is below interactive perception.

### 2.4 Multi-pane

3 panes (2 splits), one streaming 20 MB:

| Metric | Value |
|---|---|
| Threads | 25 (one PTY reader thread per pane) |
| Idle poll with 3 panes | **0.008 ms/tick** (~0.0027 ms/pane) |
| Poll while one pane streams | 0.098 ms/tick |
| Throughput / frames | 19 MB/s at 59 fps |
| RSS after one pane's scrollback filled | 88 MB |

The poll scales linearly and stays far below a frame budget at this pane count.

### 2.5 Search under streaming output

Reproducible measurement (`cargo test --release -p gpui-terminal --
--ignored --nocapture search_refresh`, ignored test in `search.rs`):

```
search refresh over 10000 history lines x 120 cols:
avg=8.04ms max=9.42ms matches=5000
```

One full-scrollback rescan costs **~8 ms**, i.e. half a 60 fps frame. The app
currently has no search UI, so this path is dormant in `markup`; it matters for
any embedder that enables search, and for the component's API contract.

---

## 3. Findings, by impact

### P1 — PTY output arrived as tens of thousands of tiny UI updates per second

**Original:** `gpui-terminal/src/view.rs:276-304`
(`read_stdout_blocking` + the reader task's coalescing loop).

**Cause.** The reader task woke for every small PTY read, drained only what was
already queued, and ran `process_bytes` inside a UI-thread entity update. At
38 MB/s that produced ~35,000 updates/s; each update also re-checked search and
called `cx.notify()`. The largest single parse reached 2.7 ms and frame pacing
fell to 48-49 fps.

**Before**

```rust
// Coalesce everything that already arrived so a firehose
// of output costs one parse + one repaint, not hundreds.
while let Ok(more) = bytes_rx.try_recv() {
    bytes.reserve(more.len());
    bytes.extend(more);
}
// ... this.update(...) { view.state.process_bytes(&bytes); ... }
```

**After**

```rust
// Wait only when the reader has caught up with the PTY; a non-empty queue
// means the producer is already ahead and parsing immediately keeps bulk
// throughput. Cap the batch so one UI-thread parse fits a frame.
if bytes.len() < MAX_COALESCE_BYTES && bytes_rx.is_empty() {
    cx.background_executor().timer(COALESCE_DELAY).await; // 1 ms
}
while bytes.len() < MAX_COALESCE_BYTES {           // 128 KiB
    let Ok(more) = bytes_rx.try_recv() else { break; };
    bytes.reserve(more.len());
    bytes.extend(more);
}
```

**Effect.** 34,994 -> 5,552 parse batches/s, worst parse 2.70 -> 1.47 ms,
48-49 -> 53 fps at the same 39 MB/s. The 128 KiB cap also bounds a backlog: a
stalled consumer can no longer hand the UI thread an 8 MiB parse. Whole-process
CPU for the 40 MB burst is unchanged within noise (100-107% for the adaptive
build vs 107-117% before): the same bytes still have to be parsed. The win is
fewer UI updates, a smaller worst-case stall, and steadier frame pacing, not
less parser work. On interactive output the 1 ms pause adds background timer
wakeups (context switches during typing rose from ~600/s to ~800/s) with no
change in measured CPU.

---

### P2 — Full-scrollback search rescans on the UI thread, even for hidden panes

**Original:** `gpui-terminal/src/view.rs:207,286-302` (150 ms throttle),
`gpui-terminal/src/search.rs:128-138` (`refresh`), `:97` (`set_query`).

**Cause.** Every PTY batch on a terminal with an active search re-ran
`RegexIter` over the whole scrollback (up to 10k lines / 5,000 matches) on the
UI thread. Measured at 8.0 ms average, 9.4 ms worst. At the old 150 ms throttle
that is ~5.3% of the UI thread and a half-frame stall roughly every 150 ms while
output streams. Background (non-rendering) terminals paid it too, for matches
nobody could see.

**Before**

```rust
if view.search.is_active() {
    let due = view.search_last_refresh
        .map(|t| now - t >= SEARCH_REFRESH_MIN_INTERVAL) // 150 ms
        .unwrap_or(true);
    if due { view.search.refresh(&view.state); /* ... */ }
}
```

**After**

```rust
// Hidden terminals: the first render after they reappear rebuilds matches.
if view.search.is_active() && view.visible_recently() {
    // ... same throttle, now 500 ms (measured cost documented above)
}
```

and in `render`:

```rust
let was_hidden = !self.visible_recently();
self.last_render = Some(Instant::now());
if was_hidden && self.search.is_active() {
    self.search.refresh(&self.state);   // catch up once, on screen
}
```

**Effect.** A visible search now spends ~1.6% of the UI thread during a firehose
instead of ~5.3%. Hidden panes spend **zero**; matches are rebuilt on the next
paint after they return. `markup` itself never activates search, so the app-level
effect is zero today; this is a component hardening fix.

---

### P3 — Blocking `/proc` inspection ran inside the 500 ms UI-thread tick

**Original:** `src/main.rs:424` (`foreground_process_names(pane.pid)` per pane)
and `:2707` (timer loop), `src/status.rs:337-424` (`/proc` reads).

**Cause.** The status loop ran on the UI/main thread and, for every pane every
500 ms, opened and read `/proc/<pid>/stat` (plus `comm`/`cmdline`/`children` on
a change). It is the only blocking I/O on that path. At one pane it measured
0.013 ms/call (cheap on this machine), but it scales with pane count and sits
between frames.

**Before**

```rust
loop {
    cx.background_executor().timer(500ms).await;
    app_state_weak.update(cx, |state, cx| {
        state.poll_sessions_status(cx);   // reads /proc inline
    });
}
```

**After**

```rust
loop {
    cx.background_executor().timer(POLL_INTERVAL).await;

    let pids = app_state_weak.update(cx, |state, _| state.pane_pids())?;
    let foregrounds = cx.background_executor().spawn(async move {
        pids.into_iter()
            .map(|pid| (pid, status::foreground_process_names(Some(pid))))
            .collect()
    }).await;                              // /proc on a background thread

    app_state_weak.update(cx, |state, cx| {
        state.poll_sessions_status_with(&foregrounds, cx);  // UI state only
    })?;
}
```

**Effect.** UI-thread poll time at idle dropped from 0.033 ms to **0.005 ms** per
tick (6x); the `/proc` cost moved off the frame entirely. Whole-process CPU is
unchanged at the measurement resolution (idle was already 0-1%), which is the
honest result: this is removal of a blocking call and a scaling hazard, not a
CPU win at one pane. The synchronous `poll_sessions_status` remains for tests.

The snapshot is keyed by pid. A pane created between the snapshot and the apply
sees an empty foreground group for one tick; a closed pane's snapshot entry is
ignored. Both degrade to "one tick late", never stale state.

---

### P4 — Finished and static-Working panes were re-scraped every tick

**Original:** `src/main.rs:483-496`. The idle fast path deliberately excluded
`Done` and `Working`, so those panes re-extracted every screen row into
`Vec<String>` and re-hashed it twice a second — for up to 30 s after a task
finished (`DONE_LINGER`), or for the whole life of a quiet agent.

**Before**

```rust
if unchanged
    && pane.status != SessionStatus::Done
    && pane.status != SessionStatus::Working
{
    continue;
}
pane.terminal.read(cx)
    .bottom_lines_into(usize::MAX, &mut pane.screen_scratch); // every tick
```

**After**

```rust
if unchanged {
    match pane.status {
        Done => { /* only the linger timer can change the badge */ continue; }
        Working => {}   // reuse the rows already in screen_scratch
        _ => continue,
    }
}
// The grid only moved if the revision did; a focus/process change can
// reuse the cached rows.
if !screen_unchanged || pane.screen_scratch.is_empty() { /* scrape */ }
```

**Effect.** On a static agent "working" screen the poll reports
`scrape=0.000 ms` on every unchanged tick; previously it paid the scrape
(0.02-0.03 ms here, plus one `String` per row) twice a second. A `Done` badge
now costs a timer comparison instead of ~60 full scrapes over its 30 s linger.
The detector still runs on `Working` so the transition to idle is unchanged.
This is small in absolute terms but removes the only per-tick allocations that
scaled with pane count.

---

### P5 — Detection joined and re-scanned the screen several times per tick

**Original:** `src/status.rs:315,504,533` (three `join("\n")` calls per
evaluation), `:266-269` (`tail_lines` collected every line, then sliced the last
12).

**Cause.** `poll_sessions_status` built the screen text, then `detect_full` built
it again for the rule pack and the tail, and `evaluate_with_match` built it a
third time for its own tail. `tail_lines` also allocated a `Vec<&str>` for the
whole screen to read 12 lines.

**Before**

```rust
let screen = input.lines.join("\n");
let tail = tail_lines(&screen, WINDOW);        // whole screen collected
// detect_full(input) joins again, evaluate_with_match joins again...
```

**After**

```rust
pub fn evaluate_screen_with_match(screen: &str, input, prev, focused) { ... }
// one join in the poller; detect_full_screen takes &str

pub fn tail_lines(screen: &str, n: usize) -> String {
    // walk backwards, stop after n non-empty lines
}
```

**Effect.** One join and one bounded tail per detection instead of three joins
and two full-screen tail builds. At a 24-row screen this is a few microseconds;
it does not move the end-to-end numbers, but it removes per-tick allocation that
scales with screen size and pane count.

---

### P6 — Process-name fingerprint was hashed every tick

**Original:** `src/main.rs:469`
(`screen_fingerprint(foreground.as_ref(), "")`).

**Cause.** The cache key hashed the foreground process names on every poll to
decide whether anything changed. The names already come from a cache that
returns the same `Arc<[String]>` allocation while the process group is stable.

**After**

```rust
Arc::ptr_eq(old_foreground, &foreground)   // O(1) instead of hashing
```

**Effect.** Included in the P3 numbers; removes one hash per pane per tick and
lets `poll_cache` store the snapshot without allocating.

---

### P7 — Dependency duplication and an unused dependency

**Original:** `Cargo.toml:14` (`portable-pty = "0.8"`),
`gpui-terminal/Cargo.toml:58-59` (`ropey = "1"`).

* `gpui-terminal` already used `portable-pty 0.9`, so the app and the component
  compiled two major versions of the same crate. The root now uses `0.9`;
  `cargo tree -d` no longer lists a second `portable-pty`.
* `ropey 1.x` was declared by `gpui-terminal` but never referenced (grep of
  `gpui-terminal/src`). It was removed. `gpui-component` still uses
  `ropey 2.0.0-beta.1`, which is a different crate version and required by that
  package.

**Effect.** One fewer crate graph in the app build, and a smaller link set. No
behaviour change (PTY tests and the full suite pass). Remaining duplicates
reported by `cargo tree -d` are transitive version pins that cannot be removed
without changing upstream dependencies: `naga` (normal vs build dep of
`blade-graphics`), `kurbo`/`svgtypes` (`resvg`/`usvg`), `arrayvec` (`naga` +
`rav1e`), `taffy` and `gpui_collections`/`zed-sum-tree` (GPUI + gpui-component).

Note: `gpui` pulls `image` -> `ravif` -> `rav1e` directly, so the large AVIF
encoder chain is not caused by `gpui-terminal`'s `arboard` default features and
cannot be removed without patching the vendored GPUI.

---

### P8 — `/proc` code compiled on every Unix, not just Linux

**Original:** `src/status.rs:34,61,337,461,486` used `#[cfg(unix)]`.

**Cause.** macOS is Unix, so the Linux `/proc/<pid>/stat` reader compiled there
and attempted a failing file open per pane every 500 ms instead of taking the
documented "no process tree" fallback.

**After.** The cache and reader are `#[cfg(target_os = "linux")]`; all other
platforms use the empty-snapshot implementation, and the pure `stat` parser and
its tests stay available only where they are used.

**Effect.** No Linux change. On macOS the poll now does no filesystem work for
process detection (screen/OSC rules only, as documented). Windows already took
the non-Unix path.

---

## 4. Reviewed and intentionally not changed

| Area | Finding | Why left alone |
|---|---|---|
| Idle repaint | 0 frames/s measured; X11 wakes 60/s but does not draw | Waking is upstream GPUI X11; changing it needs damage-driven refresh and cross-platform validation |
| Terminal paint | 0.12-0.35 ms average during output; 1.8-11 ms on the first frame only | Not a bottleneck; first-frame cost is font/shaping warm-up |
| Per-frame renderer clone | `TerminalView::render` clones a 4.4 KiB palette + font name each paint | Total render time is 0.05-0.08 ms; not measurable in end-to-end numbers |
| Scrollback retention | 10,000 lines ~= 23 MB at ~95 cols (RSS 79.5 -> 103 MB after 200k lines) | Already bounded (`Config::scrolling_history`) and configurable; changing the default is a product decision |
| PTY read allocation | One `Vec` per read | Workload is parse-bound, not allocation-bound; recycling adds a second channel for no measured gain |
| Renderer selection | Linux uses the vendored blade/Vulkan backend; present mode is the default FIFO | No tear/throughput issue measured; changing present mode needs GPU/driver coverage |
| Release profile | `opt-level=3`, thin LTO, 1 CGU, `strip=true` | Matches the requested profile. `strip=true` hides symbols from profilers; see the override in section 6 |
| Panic strategy | `panic=unwind` (default) retained | No measured benefit to `abort`; unwind keeps panic messages/diagnostics on all platforms |
| Status poll period | 500 ms retained | Process state has no portable event source; the timer is the change detector and now triggers almost no UI-thread work |

---

## 5. Diagnostics

Opt-in, off by default:

```sh
MARKUP_DIAGNOSTICS=1 ./target/release/markup
```

One line per second:

```
[diag] 1.00s frames=53 (53.0 fps, max gap 116.7ms) render avg=0.06ms max=0.13ms |
paint avg=0.12ms max=0.26ms | parse batches=5552 avg=0.10ms max=1.47ms max batch=57KB (39MB/s) |
poll ticks=2 panes=2 avg=0.042ms max=0.050ms (scrape 0.012ms, detect 0.023ms) |
fg 2 avg=0.044ms max=0.061ms | cpu=100.0% rss=103.0MB threads=23 ctxt=1389/s
```

Fields: frame cadence and worst gap, root render time, terminal paint time,
PTY-parse batches/throughput and worst batch, status poll (with scrape and
detection split out), `/proc` gather time, process CPU, RSS, thread count and
context switches per second. Hidden cost when disabled: one atomic load per
hook, no task, no syscalls. `T3_STATUS_DEBUG=1` remains the way to explain a
badge decision.

The ignored `search_refresh_cost_over_full_scrollback` test is the reproducible
measurement for the search path:

```sh
cargo test --release -p gpui-terminal -- --ignored --nocapture search_refresh
```

---

## 6. Profiling recipes

The release profile strips symbols. For any symbol-based tool, build a profiling
copy first:

```sh
CARGO_PROFILE_RELEASE_STRIP=false \
CARGO_PROFILE_RELEASE_DEBUG=1 \
cargo build --release
```

`panic=unwind` and `opt-level=3` are kept; thin LTO and one CGU make optimized
stacks less inlined than a release `fat` LTO build but keep the measurements
representative.

### Linux

**`cargo flamegraph`** (needs `perf` and `cargo install flamegraph`):

```sh
cargo install flamegraph
CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=1 \
  cargo flamegraph --release --bin markup
# or attach: cargo flamegraph --release -p <pid>
```

**samply** (no kernel perf permission needed; opens Firefox Profiler):

```sh
cargo install samply
samply record -- ./target/release/markup
```

**Linux `perf`**:

```sh
perf stat -e task-clock,context-switches,cycles,instructions,page-faults \
  ./target/release/markup
sudo perf record -g --call-graph dwarf ./target/release/markup
sudo perf report --hierarchy
```

`kernel.perf_event_paranoid` must allow user-space sampling (`<= 1`, or use
`sudo`). For the X11 wakeup question specifically:

```sh
sudo perf trace -e 'syscalls:sys_enter_poll,syscalls:sys_enter_ppoll,syscalls:sys_enter_epoll_wait' -p <pid>
```

**heaptrack**:

```sh
heaptrack ./target/release/markup        # then heaptrack_gui / heaptrack_print
```

Use it for allocation shape and peak RSS; it serializes execution, so do not
use its wall times as frame timings.

**DHAT (Valgrind)**:

```sh
valgrind --tool=dhat ./target/release/markup
# open dhat.out.<pid> with the DHAT viewer (dh_view.html)
```

For Rust-native allocation profiling, add the `dhat` crate behind a
non-default feature and call `dhat::Profiler::new_heap()` at startup; that keeps
the default build clean. DHAT is allocation-focused, not a frame profiler.

**Tracy** (frame timeline): wire the `profiling` crate with the Tracy backend
(`profiling = { version = "1", features = ["tracy"] }` plus the `tracy-client`
`enable` feature), build with
`RUSTFLAGS="-C force-frame-pointers=yes"`, and run the Tracy GUI on
`localhost:8086`. No Tracy integration exists in the tree today; the in-app
diagnostics cover frame gaps in the meantime.

### macOS (unverified — needs a Mac)

* Instruments -> **Time Profiler** attached to `markup` for CPU; **Metal System
  Trace** for GPU; **Energy Log** and **Allocations** for energy/memory.
* `powermetrics --samplers tasks --show-process-energy -n 20` for wakeups/energy
  (needs `sudo`).
* Build without stripping (same `CARGO_PROFILE_RELEASE_STRIP=false` override) so
  Instruments can symbolicate.
* Tests still needed: idle energy and wakeups, Metal frame pacing, resize,
  multi-pane throughput, input latency, and that the Linux-only `/proc` change
  behaves as intended (screen/OSC detection only).

### Windows (unverified — needs a Windows machine)

```bat
wpr -start CPU -start GPU -filemode
:: run markup and the workloads
wpr -stop perf.etl
```

Open `perf.etl` in Windows Performance Analyzer: **CPU Usage (Sampled)**,
**GPU Usage**, **Generic Events**, and **Wait Analysis** for the message loop and
the vsync thread (`components/gpui/src/platform/windows/vsync.rs`). Check DPI at
100/150/200%, per-monitor DPI changes, resize, ConPTY output throughput, and
thread counts with Process Explorer or `wpaexporter`.

---

## 7. Platform validation status

| Platform | Status | Notes |
|---|---|---|
| Linux/X11 | **Measured** | All numbers in this report. |
| Linux/Wayland | Not run (no Wayland session available) | Wayland support is preserved; `gpui` features `wayland` + `x11` are both enabled. Run the idle/typing/output/resize workloads under a Wayland compositor before claiming parity. |
| macOS | **Not run** | `/proc` reader now correctly compiles only on Linux (P8). Metal/energy/DPI behaviour unverified; see Instruments recipe. |
| Windows | **Not run** | Vendored GPUI's vsync thread and DPI handling unverified; ConPTY throughput and rendering need WPA runs. |

The off-thread `/proc` gather is a no-op outside Linux but still dispatches a
small background task every 500 ms; this is 2 tasks/s and not measurable, but a
future port could skip the dispatch when process-tree detection is unavailable.

---

## 8. Validation performed

```sh
cargo test --workspace          # 64 passed, 0 failed (status transitions,
                                # terminal output, XTGETTCAP, search, panes,
                                # resize, focus, multi-project lifecycle)
cargo build --release           # success, Linux x11
cargo check --workspace         # no warnings in changed crates
MARKUP_DIAGNOSTICS=1 ./target/release/markup   # diagnostics work
./target/release/markup         # no diagnostics output, as required
cargo tree -d                   # portable-pty duplicate gone
```

macOS/Windows toolchains were not available, so no cross-target compilation or
runtime validation was possible. The Linux-only cfg change is the main
platform-affecting edit; it removes work on macOS and leaves Windows unchanged.

---

## 9. Reproducing the workloads

```sh
# terminal 1
cd /home/sera/Desktop/markup
MARKUP_DIAGNOSTICS=1 ./target/release/markup

# terminal 2 (window must be focused)
wid=$(xdotool search --name markup | tail -1)
xdotool windowactivate --sync "$wid"

# typing
for i in $(seq 1 400); do xdotool type "echo line $i"; xdotool key Return; sleep 0.03; done

# sustained output (~40 MB)
xdotool type "yes markup-perf-line | head -c 40000000"; xdotool key Return

# resize (~30/s)
for i in $(seq 1 240); do
  if [ $((i % 2)) -eq 0 ]; then xdotool windowsize "$wid" 1100 720
  else xdotool windowsize "$wid" 760 520; fi
  sleep 0.03
done

# scroll (after building history)
xdotool type "seq 1 200000"; xdotool key Return; sleep 1
xdotool click --repeat 600 --delay 5 4

# multi-pane: click split-vertical three times (1200x780 default window)
for _ in 1 2 3; do xdotool mousemove --window "$wid" 1077 72; xdotool click 1; sleep 0.4; done
```

Compare runs by keeping the same window size, pane count and output volume;
`MARKUP_DIAGNOSTICS` prints a line a second, and the external sampler
(`/proc/<pid>/stat` utime+stime, RSS, threads, ctxt switches) gives the
process-wide view.
