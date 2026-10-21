//! [`TableView`]: the widget's state, its frame, its toolbar, and its status line.

use core::ops::Range;
use core::time::Duration;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use egui::{Align, Button, Color32, Key, Layout, RichText, ScrollArea, TextEdit, TextStyle, Ui};
use egui_table_kit::interaction::{RowEventKind, SelectionInput, SelectionMode, TableOutput};
use egui_table_kit::layout::{AutoSizeMode, CellInfo, Column};
use egui_table_kit::operations::Row;
use egui_table_kit::state::TableState;
use egui_table_kit::table::TableKit;
use parquet_table_core::{LazyOptions, LazyTable, Spawn, Stats, TableSource, Viewport};

use crate::copy::{Copied, CopySummary, copy_selection};
use crate::provider::{Adapter, CellCache, Columns, Shared};
use crate::render::{self, Kind};
use crate::style::{Paint, TableStyle, chip, gauge};
use crate::text::{bytes, grouped, parse_row_number};

/// Distinguishes views: each has its own layout state (scroll position, column widths).
static NEXT_VIEW: AtomicU64 = AtomicU64::new(0);

/// The narrowest a column can be dragged.
const MIN_COLUMN_WIDTH: f32 = 24.0;

/// How a [`TableView`] looks and what it allows.
#[derive(Debug, Clone, PartialEq)]
pub struct TableViewOptions {
    /// The engine's tuning: concurrent fetches, the cache's byte budget, prefetching, and the
    /// scroll policy.
    pub lazy: LazyOptions,
    /// The height of a row, in points.
    pub row_height: f32,
    /// The most characters of a value drawn in its cell; a longer value ends with `…` and
    /// shows more on hover.
    pub cell_chars: usize,
    /// The most characters of a value shown on hover, and copied.
    pub hover_chars: usize,
    /// The most cells one Copy puts on the clipboard.
    pub copy_max_cells: usize,
    /// About the most bytes one Copy puts on the clipboard.
    pub copy_max_bytes: usize,
    /// Whether the toolbar is shown above the table.
    pub toolbar: bool,
    /// Whether the status line is shown below the table.
    pub status_line: bool,
    /// Whether rows have alternating backgrounds.
    pub striped: bool,
    /// The colours (each one the [`Ui`]'s visuals' unless set).
    pub style: TableStyle,
}

impl Default for TableViewOptions {
    fn default() -> Self {
        Self {
            lazy: LazyOptions::default(),
            row_height: 26.0,
            cell_chars: 256,
            hover_chars: 4096,
            copy_max_cells: 1_000_000,
            copy_max_bytes: 16 << 20,
            toolbar: true,
            status_line: true,
            striped: true,
            style: TableStyle::new(),
        }
    }
}

/// What a frame of a [`TableView`] did and showed. Rows are source rows (from 0); columns are
/// source columns.
#[derive(Debug)]
#[non_exhaustive]
pub struct TableViewOutput {
    /// The table's response, or `None` when it couldn't be drawn (see [`Self::error`]).
    pub response: Option<egui::Response>,
    /// The rows drawn.
    pub rows: Range<u64>,
    /// The columns drawn, in display order (the row-number column left out).
    pub columns: Vec<usize>,
    /// The row clicked, if one was.
    pub clicked: Option<u64>,
    /// The row double-clicked, if one was.
    pub activated: Option<u64>,
    /// The row right-clicked, if one was (it is selected too).
    pub secondary_clicked: Option<u64>,
    /// How many rows are selected.
    pub selected_rows: u64,
    /// What the toolbar's Copy did, when it was used.
    pub copied: Option<CopySummary>,
    /// Why the table couldn't be drawn.
    pub error: Option<String>,
}

/// A change made in the column chooser.
enum ColumnChange {
    Show(usize, bool),
    Reset,
}

/// Starting widths, measured once.
#[derive(Debug, Clone)]
struct Widths {
    row_number: f32,
    columns: Vec<f32>,
}

/// A lazily loading, virtualized table over any [`TableSource`]: only the rows and columns in
/// view (and a prefetch ring around them) are read, and nothing is read while the view flies
/// past rows.
///
/// The caller owns it with its UI state and calls [`show`](Self::show) every frame. It never
/// blocks and takes no lock: fetches run through the caller's [`Spawn`], and a finished fetch
/// asks the [`egui::Context`] for a repaint. Time comes from egui's input, so it runs on
/// `wasm32-unknown-unknown` too.
///
/// Above the table, a toolbar shows the size, goes to a row, chooses columns (hidden columns
/// are never fetched), copies the selected rows, and retries failed blocks; below it, a status
/// line shows the rows in view, the cache, the fetches, and an optional caller text. The first
/// column holds row numbers (from 1) and stays in place. Colours come from the [`Ui`]'s
/// visuals only.
pub struct TableView {
    shared: Shared,
    state: TableState,
    options: TableViewOptions,
    id: String,
    /// The engine's last viewport: what the last frame drew.
    view: Viewport,
    /// The source columns the last frame drew.
    drawn_columns: Vec<usize>,
    frame: u64,
    widths: Option<Widths>,
    go_to: String,
    scroll_to: Option<u64>,
    status_text: Option<String>,
    message: Option<String>,
    /// The status line's height last frame (it wraps onto more lines in a narrow view).
    status_height: Option<f32>,
}

impl core::fmt::Debug for TableView {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TableView")
            .field("id", &self.id)
            .field("rows", &self.shared.rows)
            .field("columns", &self.shared.columns.count())
            .field("visible", &self.shared.columns.visible.len())
            .field("view", &self.view)
            .field("engine", &self.shared.engine)
            .finish_non_exhaustive()
    }
}

/// How soon to look again while fetches are in flight (about two frames).
const IN_FLIGHT_POLL: Duration = Duration::from_millis(33);

/// What a finished fetch calls: a repaint of `ctx` natively. On WebAssembly nothing (a
/// threaded build's `Context` is not `Send`): the view polls while fetches are in flight.
#[cfg(not(target_family = "wasm"))]
fn wake(ctx: &egui::Context) -> Arc<dyn Fn() + Send + Sync> {
    let ctx = ctx.clone();
    Arc::new(move || ctx.request_repaint())
}

/// What a finished fetch calls: a repaint of `ctx` natively. On WebAssembly nothing (a
/// threaded build's `Context` is not `Send`): the view polls while fetches are in flight.
#[cfg(target_family = "wasm")]
fn wake(_ctx: &egui::Context) -> Arc<dyn Fn() + Send + Sync> {
    Arc::new(|| {})
}

impl TableView {
    /// A view of `source`. Fetches run on `spawner`. Natively each finished fetch requests a
    /// repaint of `ctx`; on WebAssembly, where a threaded build's `Context` can't be shared with
    /// workers, the view asks for frames itself while fetches are in flight (the interface
    /// thread never waits either way). Shows every column, nothing selected, at the first row.
    pub fn new(
        source: Arc<dyn TableSource>,
        spawner: Arc<dyn Spawn>,
        ctx: &egui::Context,
        options: TableViewOptions,
    ) -> Self {
        let wake = wake(ctx);
        let columns = Columns::new(source.schema());
        let rows = usize::try_from(source.row_count()).unwrap_or(usize::MAX);
        let engine = LazyTable::new(source, spawner, wake, options.lazy.clone());
        let id = format!(
            "egui-parquet-table-{}",
            NEXT_VIEW.fetch_add(1, Ordering::Relaxed)
        );
        let mut state = TableState::new(id.clone(), rows);
        state.selection_mode = SelectionMode::Explorer;
        Self {
            shared: Shared {
                engine,
                columns,
                rows,
                cell_chars: options.cell_chars,
                hover_chars: options.hover_chars,
                cells: RefCell::new(CellCache::default()),
                tooltips: RefCell::new(HashMap::new()),
                recorded: RefCell::new(None),
            },
            state,
            options,
            id,
            view: Viewport::default(),
            drawn_columns: Vec::new(),
            frame: 0,
            widths: None,
            go_to: String::new(),
            scroll_to: None,
            status_text: None,
            message: None,
            status_height: None,
        }
    }

    /// The source.
    pub fn source(&self) -> &Arc<dyn TableSource> {
        self.shared.engine.source()
    }

    /// The engine: what each cell shows ([`LazyTable::cell`]) and its counters.
    pub fn engine(&self) -> &LazyTable {
        &self.shared.engine
    }

    /// The engine's counters.
    pub fn stats(&self) -> Stats {
        self.shared.engine.stats()
    }

    /// The options.
    pub fn options(&self) -> &TableViewOptions {
        &self.options
    }

    /// The view's options, to change: its style when the host's theme changes, its limits.
    pub fn options_mut(&mut self) -> &mut TableViewOptions {
        &mut self.options
    }

    /// The table kit's state: the selection, the focused row.
    pub fn table_state(&self) -> &TableState {
        &self.state
    }

    /// The table kit's state, to select rows from code (`selected_rows` holds rows below
    /// 2^32, see [`egui_table_kit::state::BITMAP_ROWS`]).
    pub fn table_state_mut(&mut self) -> &mut TableState {
        &mut self.state
    }

    /// Sets a text shown at the end of the status line (`"page index: loaded"`), or none.
    pub fn set_status_text(&mut self, text: Option<String>) {
        self.status_text = text;
    }

    /// The source columns shown, in display order.
    pub fn visible_columns(&self) -> &[usize] {
        &self.shared.columns.visible
    }

    /// Shows `columns` in that order and hides the others (out-of-range and repeated columns
    /// are left out). Hidden columns are never fetched.
    pub fn set_visible_columns(&mut self, columns: impl IntoIterator<Item = usize>) {
        let drawn: Vec<usize> = self
            .view
            .columns
            .clone()
            .filter_map(|position| self.shared.columns.visible.get(position).copied())
            .collect();
        self.shared.columns.show(columns.into_iter().collect());
        self.shared
            .engine
            .set_column_order(self.shared.columns.order());
        // The engine's next update follows the same columns at their new positions.
        self.view.columns = self.shared.columns.position_span(drawn);
    }

    /// Shows or hides one column, keeping the others; a column shown again takes its place
    /// in schema order.
    pub fn set_column_visible(&mut self, column: usize, visible: bool) {
        let shown = self.shared.columns.positions.get(column).copied().flatten();
        match (visible, shown) {
            (true, None) if column < self.shared.columns.count() => {
                let mut columns = self.shared.columns.visible.clone();
                let at = columns.partition_point(|&c| c < column);
                columns.insert(at, column);
                self.set_visible_columns(columns);
            }
            (false, Some(_)) => {
                let columns: Vec<usize> = self
                    .shared
                    .columns
                    .visible
                    .iter()
                    .copied()
                    .filter(|&c| c != column)
                    .collect();
                self.set_visible_columns(columns);
            }
            _ => {}
        }
    }

    /// Shows every column in schema order.
    pub fn reset_columns(&mut self) {
        let count = self.shared.columns.count();
        self.set_visible_columns(0..count);
    }

    /// Scrolls row `row` (from 0) into the middle of the view on the next frame.
    pub fn scroll_to_row(&mut self, row: u64) {
        let last = u64::try_from(self.shared.rows.saturating_sub(1)).unwrap_or(u64::MAX);
        self.scroll_to = Some(row.min(last));
    }

    /// Scrolls to row `row` (from 0) and makes it the only selected row (a row past the
    /// selection's reach, 2^32, is scrolled to and nothing is selected).
    pub fn go_to_row(&mut self, row: u64) {
        if self.shared.rows == 0 {
            return;
        }
        self.scroll_to_row(row);
        let Some(row) = self.scroll_to.and_then(|row| usize::try_from(row).ok()) else {
            return;
        };
        self.state.selected_rows.clear();
        let adapter = Adapter(&self.shared);
        self.state
            .select_row(&adapter, row, egui::Modifiers::NONE, SelectionInput::Click);
    }

    /// Copies the selected rows whose shown cells are all loaded, as tab-separated text with a
    /// header row (see [`Copied`]), within the options' limits. Fetches nothing and touches no
    /// clipboard: the toolbar's Copy puts the text there.
    pub fn copy_selection(&self) -> Copied {
        copy_selection(
            &self.shared,
            &self.state,
            self.options.copy_max_cells,
            self.options.copy_max_bytes,
        )
    }

    /// Fetches failed blocks again.
    pub fn retry_failed(&mut self) {
        self.shared.engine.retry_failed();
    }

    /// Draws the view in the space `ui` has, and does the frame's work: takes finished
    /// fetches, starts the fetches the view needs, and requests repaints while it waits.
    pub fn show(&mut self, ui: &mut Ui) -> TableViewOutput {
        let now = ui.input(|input| input.time);
        self.frame = self.frame.wrapping_add(1);
        // Take finished fetches before anything is drawn.
        self.shared.engine.update(&self.view, now);
        self.shared.cells.get_mut().begin(self.frame);
        if self.widths.is_none() {
            self.widths = Some(self.measure(ui));
        }

        let copied = if self.options.toolbar {
            self.toolbar(ui)
        } else {
            None
        };

        let status_height = if self.options.status_line {
            self.status_height.unwrap_or_else(|| {
                ui.text_style_height(&TextStyle::Body) + 2.0 * ui.spacing().item_spacing.y
            })
        } else {
            0.0
        };
        // The space left, less the status line; in a space without a bound (a scroll area),
        // a screenful of rows.
        let available = ui.available_height();
        let available = if available.is_finite() {
            available
        } else {
            ui.ctx().content_rect().height()
        };
        let height = (available - status_height).max(4.0 * self.options.row_height);
        let table = self.table(ui, height);
        self.shared.cells.get_mut().end();

        // Hand what was drawn to the engine at once, so its fetches start this frame.
        if let Some(recorded) = self.shared.recorded.get_mut().take() {
            self.drawn_columns = recorded.columns;
            if recorded.view != self.view {
                self.view = recorded.view;
                self.shared.engine.update(&self.view, now);
                ui.ctx().request_repaint();
            }
        }
        if let Some(after) = self.shared.engine.repaint_after()
            && let Ok(after) = Duration::try_from_secs_f64(after.max(0.0))
        {
            ui.ctx().request_repaint_after(after);
        }
        // Fetches in flight: look again shortly, wherever a worker can't wake the interface
        // (and as a backstop where it can).
        if self.shared.engine.stats().inflight_jobs > 0 {
            ui.ctx().request_repaint_after(IN_FLIGHT_POLL);
        }

        let mut output = TableViewOutput {
            response: None,
            rows: self.view.rows.clone(),
            columns: self.drawn_columns.clone(),
            clicked: None,
            activated: None,
            secondary_clicked: None,
            selected_rows: self.state.selected_rows.len(),
            copied,
            error: None,
        };
        match table {
            Ok(table) => {
                for event in &table.events {
                    let row = u64::try_from(event.row_index).ok();
                    match event.kind {
                        RowEventKind::Clicked => output.clicked = row,
                        RowEventKind::Activated => output.activated = row,
                        RowEventKind::SecondaryClicked => output.secondary_clicked = row,
                        _ => {}
                    }
                }
                output.response = Some(table.response);
            }
            Err(error) => {
                let message = error.to_string();
                ui.colored_label(ui.visuals().error_fg_color, &message);
                output.error = Some(message);
            }
        }
        if self.options.status_line {
            let used = self.status_line(ui);
            self.status_height = Some(used + 2.0 * ui.spacing().item_spacing.y);
        }
        output
    }

    /// Starting widths: each column's kind's, widened for its name; the row numbers' fitted
    /// to the largest.
    fn measure(&self, ui: &Ui) -> Widths {
        let body = TextStyle::Body.resolve(ui.style());
        let mono = TextStyle::Monospace.resolve(ui.style());
        let painter = ui.painter();
        let width_of = |text: &str, font: &egui::FontId| {
            painter
                .layout_no_wrap(text.to_owned(), font.clone(), Color32::WHITE)
                .size()
                .x
        };
        let mut columns: Vec<f32> = self
            .shared
            .columns
            .names
            .iter()
            .zip(&self.shared.columns.kinds)
            .map(|(name, kind)| render::start_width(*kind, width_of(name, &body)))
            .collect();
        let largest = u64::try_from(self.shared.rows).unwrap_or(u64::MAX);
        let row_number = (width_of(&largest.to_string(), &mono) + 24.0).max(40.0);
        // A view wider than its columns: the text columns share the rest (once, when the view
        // first appears; the reader resizes them as before).
        let used: f32 = row_number + columns.iter().sum::<f32>();
        let spare = ui.available_width() - used - ui.spacing().scroll.bar_width - 8.0;
        let texts = self
            .shared
            .columns
            .kinds
            .iter()
            .filter(|kind| matches!(kind, Kind::Text | Kind::Nested))
            .count();
        if spare > 0.0 && texts > 0 {
            #[allow(
                clippy::cast_precision_loss,
                clippy::as_conversions,
                reason = "a handful of columns"
            )]
            let share = spare / texts as f32;
            for (width, kind) in columns.iter_mut().zip(&self.shared.columns.kinds) {
                if matches!(kind, Kind::Text | Kind::Nested) {
                    *width += share;
                }
            }
        }
        Widths {
            row_number,
            columns,
        }
    }

    /// The toolbar: the table's size on the left; Go to row, Columns, Copy, and (after a
    /// failure) Retry on the right. Returns what a copy did.
    fn toolbar(&mut self, ui: &mut Ui) -> Option<CopySummary> {
        let paint = self.options.style.resolve(ui);
        let rows = u64::try_from(self.shared.rows).unwrap_or(u64::MAX);
        let columns = u64::try_from(self.shared.columns.count()).unwrap_or(u64::MAX);
        let mut copied = None;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            strong_label(
                ui,
                &grouped(rows),
                if rows == 1 { "row" } else { "rows" },
                &paint,
            );
            ui.label(RichText::new("·").color(paint.muted));
            strong_label(
                ui,
                &grouped(columns),
                if columns == 1 { "column" } else { "columns" },
                &paint,
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if self.shared.engine.stats().failed_blocks > 0
                    && ui
                        .button("Retry failed")
                        .on_hover_text("Read the blocks that failed again")
                        .clicked()
                {
                    self.retry_failed();
                }
                if ui
                    .add_enabled(self.can_copy(), Button::new("Copy"))
                    .on_hover_text("Copy the selected rows' loaded cells as tab-separated text")
                    .on_disabled_hover_text("Select rows to copy them")
                    .clicked()
                {
                    copied = Some(self.copy_to_clipboard(ui.ctx()));
                }
                let label = format!(
                    "Columns ({}/{})",
                    self.shared.columns.visible.len(),
                    self.shared.columns.count()
                );
                ui.menu_button(label, |ui| self.column_chooser(ui));
                self.go_to_row_field(ui);
            });
        });
        ui.add_space(4.0);
        copied
    }

    /// A Go to row field (rows from 1; Enter goes there and selects the row), for the toolbar
    /// or a host's own. Returns its response.
    pub fn go_to_row_field(&mut self, ui: &mut Ui) -> egui::Response {
        let field = ui
            .add(
                TextEdit::singleline(&mut self.go_to)
                    .hint_text("Go to row…")
                    .desired_width(110.0),
            )
            .on_hover_text("Go to a row (from 1) and select it: type its number, then Enter");
        if field.lost_focus() && ui.input(|input| input.key_pressed(Key::Enter)) {
            let rows = u64::try_from(self.shared.rows).unwrap_or(u64::MAX);
            match parse_row_number(&self.go_to) {
                Some(number) => {
                    self.go_to_row(number.saturating_sub(1));
                    self.message = None;
                }
                None => {
                    self.message = Some(format!(
                        "Go to row: enter a number from 1 to {}",
                        grouped(rows)
                    ));
                }
            }
        }
        field
    }

    /// The column chooser (a checkbox per column, and Show all columns), for the toolbar's menu
    /// or a host's own. Hidden columns are never fetched.
    pub fn column_chooser(&mut self, ui: &mut Ui) {
        match column_chooser(ui, &self.shared.columns) {
            Some(ColumnChange::Show(column, visible)) => self.set_column_visible(column, visible),
            Some(ColumnChange::Reset) => self.reset_columns(),
            None => {}
        }
    }

    /// Whether Copy has something to copy: selected rows and shown columns.
    #[must_use]
    pub fn can_copy(&self) -> bool {
        !self.state.selected_rows.is_empty() && !self.shared.columns.visible.is_empty()
    }

    /// Copies the selected rows' loaded cells to the clipboard as tab-separated text (with a
    /// header row), and says what it did (on the status line too).
    pub fn copy_to_clipboard(&mut self, ctx: &egui::Context) -> CopySummary {
        let copied = self.copy_selection();
        if copied.summary.rows > 0 {
            ctx.copy_text(copied.text);
        }
        self.message = Some(copied.summary.describe());
        copied.summary
    }

    /// The table, through the kit, with the row-number column pinned on the left.
    fn table(
        &mut self,
        ui: &mut Ui,
        height: f32,
    ) -> Result<TableOutput, egui_table_kit::error::TableError> {
        let mut map = Vec::with_capacity(self.shared.columns.visible.len().saturating_add(1));
        map.push(0);
        map.extend(
            self.shared
                .columns
                .visible
                .iter()
                .filter_map(|column| column.checked_add(1)),
        );
        let widths = self.widths.as_ref();
        let columns: Vec<Column> = map
            .iter()
            .map(|&provider_column| {
                let width = match provider_column.checked_sub(1) {
                    None => widths.map_or(60.0, |w| w.row_number),
                    Some(column) => widths
                        .and_then(|w| w.columns.get(column).copied())
                        .unwrap_or_else(|| Kind::Other.width()),
                };
                Column::new(width)
                    .range(MIN_COLUMN_WIDTH..=f32::INFINITY)
                    .resizable(true)
                    .id(egui::Id::new((self.id.as_str(), provider_column)))
            })
            .collect();

        let paint = self.options.style.resolve(ui);
        let header_height = (self.options.row_height * 1.6).round().max(38.0);
        let adapter = Adapter(&self.shared);
        let mut kit = TableKit::new(self.id.clone(), &adapter, &mut self.state)
            .with_row_height(self.options.row_height)
            .with_header_height(header_height)
            .with_header_color(paint.header)
            .with_striping_color(paint.stripe)
            .with_hover_color(paint.hover)
            .with_cell_align(Align::Center)
            .with_striped(self.options.striped)
            .with_max_height(Some(height))
            .with_auto_size_mode(AutoSizeMode::Never)
            .with_focus_outline(false)
            .with_columns(columns)
            .with_column_map(map)
            .with_sticky_columns(1);
        if let Some(row) = self.scroll_to.take() {
            kit = kit.with_scroll_to_row(row, Align::Center);
        }
        kit.show_with_output(
            ui,
            |ui: &mut Ui, cell: &CellInfo, row: &dyn Row, color: Color32| {
                render::cell(ui, &adapter, &paint, cell, row, color)
            },
        )
    }

    /// The status line: rows in view, the selection, how much of the file has been read and
    /// how full the cache is (small gauges), loading, failures, and texts. In a narrow view whole
    /// items wrap. Returns its height.
    fn status_line(&self, ui: &mut Ui) -> f32 {
        let paint = self.options.style.resolve(ui);
        let stats = self.shared.engine.stats();
        let total = u64::try_from(self.shared.rows).unwrap_or(u64::MAX);
        let source = self.shared.engine.source();
        let (read, len) = (source.bytes_read(), source.byte_len());
        egui::Frame::new()
            .fill(paint.gutter)
            .stroke(egui::Stroke::new(1.0, paint.border))
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::symmetric(10, 5))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal_wrapped(|ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    ui.spacing_mut().item_spacing.x = 6.0;
                    let rows = &self.view.rows;
                    if rows.is_empty() {
                        strong_label(ui, &grouped(total), "rows", &paint);
                    } else {
                        ui.label(RichText::new("Rows").color(paint.muted));
                        ui.label(
                            RichText::new(format!(
                                "{}–{}",
                                grouped(rows.start.saturating_add(1)),
                                grouped(rows.end)
                            ))
                            .strong(),
                        );
                        ui.label(
                            RichText::new(format!("of {}", grouped(total))).color(paint.muted),
                        );
                    }
                    let selected = self.state.selected_rows.len();
                    if selected > 0 {
                        gap(ui);
                        ui.label(
                            RichText::new(format!("{} selected", grouped(selected)))
                                .color(paint.accent),
                        );
                    }
                    if let Some(read) = read {
                        gap(ui);
                        ui.label(RichText::new("Read").color(paint.muted));
                        let text = match len {
                            Some(len) => format!("{} of {}", bytes(read), bytes(len)),
                            None => bytes(read),
                        };
                        if let Some(len) = len.filter(|len| *len > 0) {
                            #[allow(
                                clippy::cast_precision_loss,
                                clippy::as_conversions,
                                reason = "a gauge's fraction"
                            )]
                            let fraction = read as f32 / len as f32;
                            gauge(
                                ui,
                                &paint,
                                fraction,
                                "How much of the file has been read: only the columns and rows \
                                 viewed, and a margin around them",
                            );
                        }
                        ui.label(RichText::new(text).monospace());
                    }
                    gap(ui);
                    ui.label(RichText::new("Cache").color(paint.muted));
                    let budget = self.options.lazy.cache_bytes;
                    if budget > 0 {
                        #[allow(
                            clippy::cast_precision_loss,
                            clippy::as_conversions,
                            reason = "a gauge's fraction"
                        )]
                        let fraction = stats.cached_bytes as f32 / budget as f32;
                        gauge(
                            ui,
                            &paint,
                            fraction,
                            &format!(
                                "{} of decoded values cached, of {}",
                                bytes(stats.cached_bytes),
                                bytes(budget)
                            ),
                        );
                    }
                    ui.label(RichText::new(bytes(stats.cached_bytes)).monospace());
                    if stats.inflight_jobs > 0 {
                        gap(ui);
                        ui.add(egui::Spinner::new().size(12.0).color(paint.accent));
                        ui.label(RichText::new("Loading").color(paint.muted));
                    }
                    if stats.failed_blocks > 0 {
                        gap(ui);
                        ui.label(
                            RichText::new(format!(
                                "{} {} failed",
                                grouped(u64::try_from(stats.failed_blocks).unwrap_or(u64::MAX)),
                                if stats.failed_blocks == 1 {
                                    "block"
                                } else {
                                    "blocks"
                                }
                            ))
                            .color(paint.danger),
                        );
                    }
                    if let Some(text) = &self.status_text {
                        gap(ui);
                        chip(ui, &paint, text);
                    }
                    if let Some(message) = &self.message {
                        gap(ui);
                        ui.label(RichText::new(message.as_str()).color(paint.muted));
                    }
                });
            })
            .response
            .rect
            .height()
    }
}

/// A count in bold followed by its noun, muted.
fn strong_label(ui: &mut Ui, count: &str, noun: &str, paint: &Paint) {
    ui.label(RichText::new(count).strong());
    ui.label(RichText::new(noun).color(paint.muted));
}

/// The space between items of the status line.
fn gap(ui: &mut Ui) {
    ui.add_space(10.0);
}

/// The column chooser's menu: a checkbox per column (only those scrolled into view are laid
/// out) and a reset.
fn column_chooser(ui: &mut Ui, columns: &Columns) -> Option<ColumnChange> {
    let mut change = None;
    ui.set_min_width(220.0);
    if ui.button("Show all columns").clicked() {
        change = Some(ColumnChange::Reset);
    }
    ui.separator();
    let row_height = ui.spacing().interact_size.y;
    ScrollArea::vertical().max_height(320.0).show_rows(
        ui,
        row_height,
        columns.count(),
        |ui, range| {
            for column in range {
                let Some(name) = columns.names.get(column) else {
                    continue;
                };
                let mut shown = columns.positions.get(column).copied().flatten().is_some();
                if ui.checkbox(&mut shown, name.as_str()).changed() {
                    change = Some(ColumnChange::Show(column, shown));
                }
            }
        },
    );
    change
}
