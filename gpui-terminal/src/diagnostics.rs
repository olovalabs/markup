//! Paint-side counters for the app's opt-in diagnostics.
//!
//! The terminal crate cannot see the app's `diagnostics` module, so it keeps
//! its own cumulative counters here and the app drains them with
//! [`take_paint_stats`] when it prints a report. When `MARKUP_DIAGNOSTICS` is
//! unset the hooks are a single relaxed load.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("MARKUP_DIAGNOSTICS").is_some_and(|value| value != "0"));

#[inline]
pub fn enabled() -> bool {
    *ENABLED
}

static PAINTS: AtomicU64 = AtomicU64::new(0);
static PAINT_NANOS: AtomicU64 = AtomicU64::new(0);
static PAINT_MAX_NANOS: AtomicU64 = AtomicU64::new(0);

static PARSE_BATCHES: AtomicU64 = AtomicU64::new(0);
static PARSE_NANOS: AtomicU64 = AtomicU64::new(0);
static PARSE_MAX_NANOS: AtomicU64 = AtomicU64::new(0);
static PARSE_BYTES: AtomicU64 = AtomicU64::new(0);
static PARSE_MAX_BYTES: AtomicU64 = AtomicU64::new(0);

/// Start timing one terminal paint. `None` when diagnostics are off.
#[inline]
pub fn paint_start() -> Option<Instant> {
    enabled().then(Instant::now)
}

#[inline]
pub fn paint_end(start: Option<Instant>) {
    let Some(start) = start else { return };
    let nanos = start.elapsed().as_nanos() as u64;
    PAINTS.fetch_add(1, Ordering::Relaxed);
    PAINT_NANOS.fetch_add(nanos, Ordering::Relaxed);
    PAINT_MAX_NANOS.fetch_max(nanos, Ordering::Relaxed);
}

/// Count, cumulative nanos and worst-case nanos since the previous call.
pub struct PaintStats {
    pub paints: u64,
    pub nanos: u64,
    pub max_nanos: u64,
}

pub fn take_paint_stats() -> PaintStats {
    PaintStats {
        paints: PAINTS.swap(0, Ordering::Relaxed),
        nanos: PAINT_NANOS.swap(0, Ordering::Relaxed),
        max_nanos: PAINT_MAX_NANOS.swap(0, Ordering::Relaxed),
    }
}

/// Start timing one PTY parse batch on the UI thread.
#[inline]
pub fn parse_start() -> Option<Instant> {
    enabled().then(Instant::now)
}

#[inline]
pub fn parse_end(start: Option<Instant>, bytes: usize) {
    let Some(start) = start else { return };
    let nanos = start.elapsed().as_nanos() as u64;
    PARSE_BATCHES.fetch_add(1, Ordering::Relaxed);
    PARSE_NANOS.fetch_add(nanos, Ordering::Relaxed);
    PARSE_MAX_NANOS.fetch_max(nanos, Ordering::Relaxed);
    PARSE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    PARSE_MAX_BYTES.fetch_max(bytes as u64, Ordering::Relaxed);
}

/// Batches, cumulative nanos, worst case, bytes and largest batch since the
/// previous call.
pub struct ParseStats {
    pub batches: u64,
    pub nanos: u64,
    pub max_nanos: u64,
    pub bytes: u64,
    pub max_bytes: u64,
}

pub fn take_parse_stats() -> ParseStats {
    ParseStats {
        batches: PARSE_BATCHES.swap(0, Ordering::Relaxed),
        nanos: PARSE_NANOS.swap(0, Ordering::Relaxed),
        max_nanos: PARSE_MAX_NANOS.swap(0, Ordering::Relaxed),
        bytes: PARSE_BYTES.swap(0, Ordering::Relaxed),
        max_bytes: PARSE_MAX_BYTES.swap(0, Ordering::Relaxed),
    }
}
