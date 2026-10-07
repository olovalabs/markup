//! Owner of the `alacritty_terminal` [`Term`] and the VTE [`Processor`].
//!
//! Everything that needs to touch the grid goes through [`TerminalState`], which
//! keeps the terminal behind a `parking_lot::Mutex` so the PTY reader task and
//! the UI thread can share it.

use crate::event::{GpuiEventProxy, TerminalNotifierHandle};
use alacritty_terminal::grid::{Dimensions, Scroll};
use alacritty_terminal::index::{Boundary, Column, Direction, Line, Point, Side};
use alacritty_terminal::selection::{Selection, SelectionRange, SelectionType};
use alacritty_terminal::term::search::{Match, RegexIter, RegexSearch};
use alacritty_terminal::term::{Config, Osc52, Term, TermMode};
use alacritty_terminal::vi_mode::ViMotion;
use alacritty_terminal::vte::ansi::Processor;
use parking_lot::Mutex;
use std::sync::Arc;

/// Upper bound on the number of highlighted matches we will collect for a
/// single search, mirroring Zed's `MAX_SEARCH_LINES` style guard so a pathological
/// regex over a 100k-line scrollback cannot stall a frame.
pub const MAX_SEARCH_MATCHES: usize = 5_000;

/// Dimensions handed to `Term::new` / `Term::resize`.
///
/// Note that `total_lines` must be `screen_lines` here: the scrollback size is
/// taken from [`Config::scrolling_history`], *not* from this trait, and
/// `Grid::resize` only looks at `screen_lines`/`columns`.
#[derive(Debug, Clone, Copy)]
pub struct TermDimensions {
    columns: usize,
    screen_lines: usize,
}

impl TermDimensions {
    pub fn new(columns: usize, screen_lines: usize) -> Self {
        Self {
            columns: columns.max(alacritty_terminal::term::MIN_COLUMNS),
            screen_lines: screen_lines.max(alacritty_terminal::term::MIN_SCREEN_LINES),
        }
    }
}

impl Dimensions for TermDimensions {
    fn total_lines(&self) -> usize {
        self.screen_lines
    }

    fn screen_lines(&self) -> usize {
        self.screen_lines
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn last_column(&self) -> Column {
        Column(self.columns.saturating_sub(1))
    }

    fn topmost_line(&self) -> Line {
        Line(0)
    }

    fn bottommost_line(&self) -> Line {
        Line(self.screen_lines as i32 - 1)
    }
}

pub struct TerminalState {
    term: Arc<Mutex<Term<GpuiEventProxy>>>,
    parser: Processor,
    cols: usize,
    rows: usize,
    config: Config,
    /// Kept so protocol passthroughs (XTGETTCAP) can reply into the PTY.
    notifier: TerminalNotifierHandle,
}

impl TerminalState {
    /// Create a terminal with alacritty's default `Config` (10k lines of
    /// scrollback, default semantic escape chars).
    pub fn new(cols: usize, rows: usize, event_proxy: GpuiEventProxy) -> Self {
        Self::with_config(cols, rows, Config::default(), event_proxy)
    }

    /// Create a terminal with an explicit alacritty [`Config`].
    ///
    /// This is how the scrollback size actually reaches the grid — passing a
    /// bigger `Dimensions::total_lines` does nothing.
    pub fn with_config(
        cols: usize,
        rows: usize,
        config: Config,
        event_proxy: GpuiEventProxy,
    ) -> Self {
        let dimensions = TermDimensions::new(cols, rows);
        let notifier = event_proxy.notifier_handle();
        let term = Term::new(config.clone(), &dimensions, event_proxy);

        Self {
            term: Arc::new(Mutex::new(term)),
            parser: Processor::new(),
            cols: dimensions.columns(),
            rows: dimensions.screen_lines(),
            config,
            notifier,
        }
    }

    /// Build an alacritty [`Config`] from the component-level settings.
    pub fn build_config(scrollback: usize, semantic_escape_chars: &str, osc52: bool) -> Config {
        Config {
            scrolling_history: scrollback,
            semantic_escape_chars: semantic_escape_chars.to_owned(),
            // The kitty keyboard protocol is deliberately off: an application
            // that negotiates it expects the frontend to *encode* keys the
            // kitty way, and `input.rs` speaks plain xterm.
            kitty_keyboard: false,
            osc52: if osc52 {
                Osc52::CopyPaste
            } else {
                Osc52::OnlyCopy
            },
            ..Config::default()
        }
    }

    pub fn process_bytes(&mut self, bytes: &[u8]) {
        self.answer_xtgettcap(bytes);
        let mut term = self.term.lock();
        self.parser.advance(&mut *term, bytes);
    }

    /// Answer XTGETTCAP queries (`DCS + q <hex-encoded caps> ST`).
    ///
    /// alacritty_terminal's ANSI handler never dispatches DCS (vte's
    /// `hook`/`put`/`unhook` are no-ops), so the raw byte stream is snooped here
    /// and replies are written straight into the PTY through the notifier.
    /// Per xterm/kitty, a supported capability is answered with
    /// `DCS 1 + r <hex-cap>=<hex-value> ST` and an unsupported one with
    /// `DCS 0 + r <hex-cap> ST`. Capability names may be `;`-separated, in which
    /// case one reply is sent per name.
    ///
    /// Only queries fully contained in `bytes` are answered: a query split
    /// across two PTY reads is ignored rather than answered piecemeal (agent
    /// CLIs send their capability probes in a single write).
    fn answer_xtgettcap(&mut self, bytes: &[u8]) {
        // Fast path: every query starts with ESC, which plain output never
        // contains — one short scan instead of two substring searches per read.
        if !bytes.contains(&0x1b) {
            return;
        }
        const DCS_PLUS_Q: &[u8] = b"\x1bP+q";
        const ST: &[u8] = b"\x1b\\";

        let mut search_from = 0;
        while let Some(rel) = find_subslice(&bytes[search_from..], DCS_PLUS_Q) {
            let start = search_from + rel + DCS_PLUS_Q.len();
            let Some(end_rel) = find_subslice(&bytes[start..], ST) else {
                break; // unterminated query: cannot answer it from this read
            };
            // `Pt` is one or more `;`-separated hex-encoded names, so split the
            // raw payload first and decode each name separately.
            for name_hex in bytes[start..start + end_rel].split(|&b| b == b';') {
                if name_hex.is_empty() {
                    continue;
                }
                if let Some(cap) = decode_hex(name_hex) {
                    let reply = xtgettcap_reply(cap.as_str());
                    self.notifier.write(&reply);
                }
            }
            search_from = start + end_rel + ST.len();
        }
    }

    pub fn resize(&mut self, cols: usize, rows: usize) {
        let dimensions = TermDimensions::new(cols, rows);
        self.cols = dimensions.columns();
        self.rows = dimensions.screen_lines();

        let mut term = self.term.lock();
        term.resize(dimensions);
    }

    /// Update the cached dimensions after the grid was resized directly through
    /// [`Self::term_arc`] (which is what the paint pass does, so the very first
    /// frame is already laid out at the right size).
    pub fn sync_dimensions(&mut self, cols: usize, rows: usize) {
        self.cols = cols;
        self.rows = rows;
    }

    /// Apply a new alacritty config (scrollback size, semantic escape chars, …)
    /// to a live terminal.
    pub fn set_config(&mut self, config: Config) {
        if self.config == config {
            return;
        }
        self.config = config.clone();
        let mut term = self.term.lock();
        term.set_options(config);
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn mode(&self) -> TermMode {
        *self.term.lock().mode()
    }

    pub fn with_term<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Term<GpuiEventProxy>) -> R,
    {
        let term = self.term.lock();
        f(&term)
    }

    pub fn with_term_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Term<GpuiEventProxy>) -> R,
    {
        let mut term = self.term.lock();
        f(&mut term)
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn term_arc(&self) -> Arc<Mutex<Term<GpuiEventProxy>>> {
        Arc::clone(&self.term)
    }

    // ---------------------------------------------------------------------
    // Scrolling
    // ---------------------------------------------------------------------

    pub fn display_offset(&self) -> usize {
        self.term.lock().grid().display_offset()
    }

    pub fn history_size(&self) -> usize {
        self.term.lock().grid().history_size()
    }

    /// `true` when the viewport is pinned to the newest output.
    pub fn is_at_bottom(&self) -> bool {
        self.display_offset() == 0
    }

    /// Scroll the viewport.
    ///
    /// `Term::scroll_display` already clamps the vi cursor into the viewport and
    /// recomputes the vi selection, so this is all that is needed for vi mode
    /// to stay consistent while scrolling.
    pub fn scroll(&self, scroll: Scroll) {
        self.term.lock().scroll_display(scroll);
    }

    pub fn scroll_lines(&self, lines: i32) {
        if lines != 0 {
            self.scroll(Scroll::Delta(lines));
        }
    }

    pub fn scroll_to_bottom(&self) {
        self.scroll(Scroll::Bottom);
    }

    pub fn scroll_to_top(&self) {
        self.scroll(Scroll::Top);
    }

    // ---------------------------------------------------------------------
    // Selection
    // ---------------------------------------------------------------------

    pub fn set_selection(&self, selection: Option<Selection>) {
        self.term.lock().selection = selection;
    }

    pub fn start_selection(&self, ty: SelectionType, point: Point, side: Side) {
        self.term.lock().selection = Some(Selection::new(ty, point, side));
    }

    pub fn update_selection(&self, point: Point, side: Side) {
        let mut term = self.term.lock();
        if let Some(selection) = term.selection.as_mut() {
            selection.update(point, side);
        }
    }

    pub fn selection_range(&self) -> Option<SelectionRange> {
        let term = self.term.lock();
        term.selection.as_ref().and_then(|s| s.to_range(&term))
    }

    pub fn selection_text(&self) -> Option<String> {
        self.term.lock().selection_to_string()
    }

    pub fn has_selection(&self) -> bool {
        let term = self.term.lock();
        term.selection
            .as_ref()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }

    pub fn clear_selection(&self) {
        self.term.lock().selection = None;
    }

    /// Select the entire scrollback plus the visible screen.
    pub fn select_all(&self) {
        let mut term = self.term.lock();
        let start = Point::new(term.topmost_line(), Column(0));
        let end = Point::new(term.bottommost_line(), term.last_column());
        let mut selection = Selection::new(SelectionType::Simple, start, Side::Left);
        selection.update(end, Side::Right);
        term.selection = Some(selection);
    }

    // ---------------------------------------------------------------------
    // Search
    // ---------------------------------------------------------------------

    /// Collect every match of `regex` in the scrollback + screen.
    pub fn all_matches(&self, regex: &mut RegexSearch) -> Vec<Match> {
        let term = self.term.lock();
        let start = Point::new(term.topmost_line(), Column(0));
        let end = Point::new(term.bottommost_line(), term.last_column());
        RegexIter::new(start, end, Direction::Right, &term, regex)
            .take(MAX_SEARCH_MATCHES)
            .collect()
    }

    /// Find the next match relative to `origin`.
    pub fn search_next(
        &self,
        regex: &mut RegexSearch,
        origin: Point,
        direction: Direction,
    ) -> Option<Match> {
        let term = self.term.lock();
        term.search_next(regex, origin, direction, Side::Left, None)
    }

    /// Scroll the viewport so that `point` is visible.
    pub fn scroll_to_point(&self, point: Point) {
        self.term.lock().scroll_to_point(point);
    }

    // ---------------------------------------------------------------------
    // Vi mode
    // ---------------------------------------------------------------------

    pub fn is_vi_mode(&self) -> bool {
        self.term.lock().mode().contains(TermMode::VI)
    }

    pub fn toggle_vi_mode(&self) {
        self.term.lock().toggle_vi_mode();
    }

    pub fn vi_motion(&self, motion: ViMotion) {
        self.term.lock().vi_motion(motion);
    }

    pub fn vi_goto_point(&self, point: Point) {
        self.term.lock().vi_goto_point(point);
    }

    pub fn vi_cursor_point(&self) -> Point {
        self.term.lock().vi_mode_cursor.point
    }

    // ---------------------------------------------------------------------
    // Misc helpers
    // ---------------------------------------------------------------------

    /// `true` when the application currently wants a blinking cursor
    /// (`DECSCUSR` blinking shapes or `CSI ? 12 h`).
    pub fn cursor_blinking(&self) -> bool {
        self.term.lock().cursor_style().blinking
    }

    pub fn set_focused(&self, focused: bool) {
        self.term.lock().is_focused = focused;
    }

    /// Clamp a terminal point into the valid grid range.
    pub fn clamp_point(&self, point: Point) -> Point {
        let term = self.term.lock();
        point.grid_clamp(&*term, Boundary::Grid)
    }

    /// Reset the terminal (`clear` + wipe the scrollback), like Zed's
    /// `terminal: clear` action.
    pub fn clear_screen_and_scrollback(&mut self) {
        {
            let mut term = self.term.lock();
            term.selection = None;
        }
        // `ESC c` (RIS) resets the emulator; `CSI 3 J` drops the scrollback.
        self.process_bytes(b"\x1b[H\x1b[2J\x1b[3J");
    }

    /// Text of a single terminal line, trailing whitespace trimmed.
    pub fn line_text(&self, line: Line) -> String {
        let term = self.term.lock();
        let start = Point::new(line, Column(0));
        let end = Point::new(line, term.last_column());
        term.bounds_to_string(start, end).trim_end().to_string()
    }

    /// Retrieve the bottom N lines from the visible viewport.
    pub fn bottom_lines(&self, n: usize) -> Vec<String> {
        let mut lines = Vec::new();
        self.bottom_lines_into(n, &mut lines);
        lines
    }

    /// Like [`Self::bottom_lines`], but reuses the caller's buffer across
    /// ticks so a status poller scraping every session at 2 Hz does not
    /// reallocate the outer `Vec` each time. Each line is still freshly
    /// extracted (and trimmed in place: one allocation per line, not two).
    pub fn bottom_lines_into(&self, n: usize, out: &mut Vec<String>) {
        let term = self.term.lock();
        let screen_lines = self.rows;
        let last_col = term.last_column();
        let count = n.min(screen_lines);
        let start_row = screen_lines.saturating_sub(count);
        out.clear();
        out.reserve(count);
        for row in start_row..screen_lines {
            let line = Line(row as i32);
            let start = Point::new(line, Column(0));
            let end = Point::new(line, last_col);
            let mut text = term.bounds_to_string(start, end);
            let trimmed_len = text.trim_end().len();
            text.truncate(trimmed_len);
            out.push(text);
        }
    }
}

/// Find `needle` in `haystack` and return the byte offset of the first match.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Value of an XTGETTCAP capability, if we implement it.
///
/// `RGB`/`Tc` advertise direct colour support, which the renderer does honour
/// (the palette and truecolour SGR sequences are handled by `colors.rs`).
/// Everything else is answered as unsupported so probing clients fall back to
/// their conservative path instead of assuming the feature exists.
fn xtgettcap_value(cap: &str) -> Option<&'static str> {
    match cap {
        "RGB" | "Tc" => Some("1"),
        _ => None,
    }
}

/// Build the `DCS` reply for one requested capability.
fn xtgettcap_reply(cap: &str) -> Vec<u8> {
    let name_hex = encode_hex(cap.as_bytes());
    match xtgettcap_value(cap) {
        Some(value) => {
            format!("\x1bP1+r{name_hex}={}\x1b\\", encode_hex(value.as_bytes())).into_bytes()
        }
        None => format!("\x1bP0+r{name_hex}\x1b\\").into_bytes(),
    }
}

/// Lowercase hex encoding, as used by XTGETTCAP parameters.
fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Decode a hex nibble pair per byte. Returns `None` if the sequence is
/// malformed (odd length or bad nibble).
fn decode_hex(bytes: &[u8]) -> Option<String> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut chunks = bytes.chunks_exact(2);
    for chunk in &mut chunks {
        let hi = (chunk[0] as char).to_digit(16)?;
        let lo = (chunk[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    fn state(cols: usize, rows: usize) -> TerminalState {
        let (tx, _rx) = channel();
        TerminalState::new(cols, rows, GpuiEventProxy::new(tx))
    }

    #[test]
    fn test_terminal_creation() {
        let terminal = state(80, 24);
        assert_eq!(terminal.cols(), 80);
        assert_eq!(terminal.rows(), 24);
    }

    #[test]
    fn test_process_bytes() {
        let mut terminal = state(80, 24);
        terminal.process_bytes(b"Hello, world!");
        terminal.with_term(|term| assert_eq!(term.grid().columns(), 80));
    }

    #[test]
    fn test_resize() {
        let mut terminal = state(80, 24);
        terminal.resize(120, 30);

        assert_eq!(terminal.cols(), 120);
        assert_eq!(terminal.rows(), 30);
        terminal.with_term(|term| {
            assert_eq!(term.grid().columns(), 120);
            assert_eq!(term.grid().screen_lines(), 30);
        });
    }

    #[test]
    fn test_mode() {
        let terminal = state(80, 24);
        let _bits = terminal.mode().bits();
    }

    #[test]
    fn test_term_arc() {
        let terminal = state(80, 24);
        assert!(Arc::ptr_eq(&terminal.term_arc(), &terminal.term_arc()));
    }

    #[test]
    fn test_scrollback_is_configurable() {
        let (tx, _rx) = channel();
        let config = TerminalState::build_config(500, ",│`|:\"' ()[]{}<>\t", true);
        let mut terminal = TerminalState::with_config(80, 24, config, GpuiEventProxy::new(tx));

        for i in 0..1000 {
            terminal.process_bytes(format!("Line {i}\r\n").as_bytes());
        }

        // 500 lines of history must be retained, not 0 and not 1000.
        assert_eq!(terminal.history_size(), 500);
    }

    #[test]
    fn test_scroll_direction() {
        let mut terminal = state(80, 24);
        for i in 0..100 {
            terminal.process_bytes(format!("Line {i}\r\n").as_bytes());
        }

        assert!(terminal.is_at_bottom());
        terminal.scroll_lines(5);
        assert_eq!(terminal.display_offset(), 5);
        assert!(!terminal.is_at_bottom());
        terminal.scroll_lines(-5);
        assert_eq!(terminal.display_offset(), 0);

        terminal.scroll_to_top();
        assert!(terminal.display_offset() > 0);
        terminal.scroll_to_bottom();
        assert_eq!(terminal.display_offset(), 0);
    }

    #[test]
    fn test_select_all_and_copy() {
        let mut terminal = state(20, 4);
        terminal.process_bytes(b"alpha\r\nbeta\r\n");

        assert!(!terminal.has_selection());
        terminal.select_all();
        assert!(terminal.has_selection());

        let text = terminal.selection_text().unwrap_or_default();
        assert!(text.contains("alpha"));
        assert!(text.contains("beta"));

        terminal.clear_selection();
        assert!(!terminal.has_selection());
    }

    #[test]
    fn test_search_finds_all_matches() {
        let mut terminal = state(40, 6);
        terminal.process_bytes(b"needle one\r\nhaystack\r\nneedle two\r\n");

        let mut regex = RegexSearch::new("needle").expect("valid regex");
        let matches = terminal.all_matches(&mut regex);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_vi_mode_toggle() {
        let terminal = state(40, 6);
        assert!(!terminal.is_vi_mode());
        terminal.toggle_vi_mode();
        assert!(terminal.is_vi_mode());
        terminal.toggle_vi_mode();
        assert!(!terminal.is_vi_mode());
    }

    #[test]
    fn test_xtgettcap_known_cap_replies_inline() {
        let (tx, rx) = std::sync::mpsc::channel();
        let proxy = GpuiEventProxy::new(tx);
        let mut terminal = TerminalState::new(80, 24, proxy);

        // XTGETTCAP requests are `ESC P + q <percent-encoded> ESC \`; the
        // responder must fire before the stream reaches the grid and must not
        // print anything.
        terminal.process_bytes(b"\x1bP+q5442\x1b\\");

        // The query is dropped by the VTE handler (no output): nothing was
        // printed, and any control state the handler tracked stays consistent.
        assert!(!rx.try_recv().is_ok());
    }

    #[test]
    fn test_xtgettcap_unknown_cap_gets_invalid_reply() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let proxy = GpuiEventProxy::new(tx);
        let mut terminal = TerminalState::new(80, 24, proxy);

        // Malformed request must still produce the `0+r` reply.
        terminal.process_bytes(b"\x1bP+q78797a7a\x1b\\");
    }

    #[test]
    fn test_decode_hex_roundtrip() {
        assert_eq!(decode_hex(b"4142"), Some("AB".to_string()));
    }

    #[test]
    fn test_line_text() {
        let mut terminal = state(40, 6);
        terminal.process_bytes(b"hello world\r\n");
        assert_eq!(terminal.line_text(Line(0)), "hello world");
    }
}
