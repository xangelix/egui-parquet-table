//! The table kit's provider over a [`LazyTable`]: what the kit asks of each cell, answered from
//! the engine's cache and never from the source.
//!
//! Provider column 0 is the row-number column (sticky, never fetched); provider column `p > 0`
//! is source column `p - 1`. Provider rows are source rows. The provider is rebuilt every frame
//! over the view's [`Shared`] state; its [`prepare`](TableProvider::prepare) records what the
//! kit draws, for the engine's next [`update`](LazyTable::update).

use core::cell::RefCell;
use core::ops::Range;
use std::borrow::Cow;
use std::collections::HashMap;

use egui_table_kit::error::TableError;
use egui_table_kit::operations::{
    BorrowedRow, CellStatus, HeaderIter, RowCallback, TableCell, TableProvider, Viewport as Drawn,
};
use egui_table_kit::state::TableState;
use parquet_table_core::arrow::schema::SchemaRef;
use parquet_table_core::{Cell, CellText, LazyTable, Viewport, format_cell};

use crate::copy::{loaded_ranges, selected_in};
use crate::render::Kind;
use crate::text::{column_tooltip, display_name};

/// The most formatted cells kept between the kit's question and the renderer's (a frame draws
/// far fewer; a caller walking rows through the provider formats the rest without keeping
/// them).
const MAX_CACHED_CELLS: usize = 16_384;

/// The columns: their names and kinds, and which are shown in what order.
#[derive(Debug, Clone)]
pub(crate) struct Columns {
    /// Each source column's name as shown.
    pub names: Vec<String>,
    /// Each source column's kind of values.
    pub kinds: Vec<Kind>,
    /// Each source column's short type name, under its name in the header.
    pub types: Vec<String>,
    /// The source columns shown, in display order.
    pub visible: Vec<usize>,
    /// Each source column's display position, `None` when hidden.
    pub positions: Vec<Option<usize>>,
}

impl Columns {
    /// Every column of `schema`, shown in schema order.
    pub(crate) fn new(schema: &SchemaRef) -> Self {
        let names = schema
            .fields()
            .iter()
            .map(|field| display_name(field.name()))
            .collect::<Vec<_>>();
        let kinds = schema
            .fields()
            .iter()
            .map(|field| Kind::of(field.data_type()))
            .collect();
        let types = schema
            .fields()
            .iter()
            .map(|field| crate::render::type_name(field.data_type()))
            .collect();
        let count = names.len();
        let mut columns = Self {
            names,
            kinds,
            types,
            visible: Vec::new(),
            positions: Vec::new(),
        };
        columns.show((0..count).collect());
        columns
    }

    /// The number of source columns.
    pub(crate) fn count(&self) -> usize {
        self.names.len()
    }

    /// Shows `visible` in that order (out-of-range and repeated columns are left out).
    pub(crate) fn show(&mut self, visible: Vec<usize>) {
        let mut positions = vec![None; self.count()];
        let mut kept = Vec::with_capacity(visible.len());
        for column in visible {
            if let Some(slot) = positions.get_mut(column)
                && slot.is_none()
            {
                *slot = Some(kept.len());
                kept.push(column);
            }
        }
        self.visible = kept;
        self.positions = positions;
    }

    /// The engine's column order: `None` while every column is shown in schema order.
    pub(crate) fn order(&self) -> Option<Vec<usize>> {
        let identity = self.visible.len() == self.count()
            && self.visible.iter().enumerate().all(|(i, &c)| i == c);
        (!identity).then(|| self.visible.clone())
    }

    /// The display positions of `sources` (hidden ones left out) as one range, or empty.
    pub(crate) fn position_span(&self, sources: impl IntoIterator<Item = usize>) -> Range<usize> {
        let mut span: Option<(usize, usize)> = None;
        for source in sources {
            if let Some(Some(position)) = self.positions.get(source) {
                span = Some(match span {
                    Some((low, high)) => (low.min(*position), high.max(*position)),
                    None => (*position, *position),
                });
            }
        }
        span.map_or(0..0, |(low, high)| low..high.saturating_add(1))
    }
}

/// What the kit drew in the last pass: the engine's viewport, and the source columns drawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Recorded {
    /// Rows and display positions, as the engine takes them.
    pub view: Viewport,
    /// The source columns drawn, in order (the row-number column left out).
    pub columns: Vec<usize>,
}

#[derive(Debug)]
struct Cached {
    text: CellText,
    /// The longer text shown on hover, once asked for.
    full: Option<CellText>,
    /// The frame the cell was last drawn in.
    used: u64,
}

/// Formatted cells of the frame being drawn: the kit asks for a cell's text (for its label),
/// then the renderer draws it, and each is formatted once.
#[derive(Debug, Default)]
pub(crate) struct CellCache {
    frame: u64,
    entries: HashMap<(u64, usize), Cached>,
}

impl CellCache {
    /// Starts a frame.
    pub(crate) fn begin(&mut self, frame: u64) {
        self.frame = frame;
    }

    /// Ends a frame: forgets the cells it didn't draw.
    pub(crate) fn end(&mut self) {
        let frame = self.frame;
        self.entries.retain(|_, cell| cell.used == frame);
    }
}

/// What the provider reads: everything of the view but the kit's own state.
pub(crate) struct Shared {
    /// The engine.
    pub engine: LazyTable,
    /// The columns.
    pub columns: Columns,
    /// The number of rows the kit shows (the source's, clamped to `usize`).
    pub rows: usize,
    /// Characters drawn of a value in its cell.
    pub cell_chars: usize,
    /// Characters of a value shown on hover.
    pub hover_chars: usize,
    /// Formatted cells of the frame.
    pub cells: RefCell<CellCache>,
    /// Header tooltips, built when first shown.
    pub tooltips: RefCell<HashMap<usize, String>>,
    /// What the last pass drew, from [`TableProvider::prepare`].
    pub recorded: RefCell<Option<Recorded>>,
}

/// The kit's provider for one frame.
pub(crate) struct Adapter<'a>(pub &'a Shared);

impl Adapter<'_> {
    /// The kind of a source column's values.
    pub(crate) fn kind(&self, column: usize) -> Option<Kind> {
        self.0.columns.kinds.get(column).copied()
    }

    fn cell_chars(&self, column: usize) -> usize {
        self.kind(column)
            .map_or(self.0.cell_chars, |kind| kind.cell_chars(self.0.cell_chars))
    }

    /// The text a loaded cell shows, or `None` while it isn't loaded. Reads the engine's
    /// cache only.
    pub(crate) fn text(&self, row: u64, column: usize) -> Option<CellText> {
        let Cell::Ready { array, index } = self.0.engine.cell(row, column) else {
            return None;
        };
        let mut cache = self.0.cells.borrow_mut();
        let frame = cache.frame;
        if let Some(cached) = cache.entries.get_mut(&(row, column)) {
            cached.used = frame;
            return Some(cached.text.clone());
        }
        let text = format_cell(array.as_ref(), index, self.cell_chars(column));
        if cache.entries.len() < MAX_CACHED_CELLS {
            cache.entries.insert(
                (row, column),
                Cached {
                    text: text.clone(),
                    full: None,
                    used: frame,
                },
            );
        }
        Some(text)
    }

    /// The longer text of a loaded cell, shown on hover when its cell's text was cut.
    pub(crate) fn full_text(&self, row: u64, column: usize) -> Option<CellText> {
        let Cell::Ready { array, index } = self.0.engine.cell(row, column) else {
            return None;
        };
        let mut cache = self.0.cells.borrow_mut();
        if let Some(full) = cache
            .entries
            .get(&(row, column))
            .and_then(|cached| cached.full.clone())
        {
            return Some(full);
        }
        let full = format_cell(array.as_ref(), index, self.0.hover_chars);
        if let Some(cached) = cache.entries.get_mut(&(row, column)) {
            cached.full = Some(full.clone());
        }
        Some(full)
    }

    /// The rows whose shown cells are all loaded, ascending.
    fn loaded(&self) -> Vec<Range<u64>> {
        loaded_ranges(&self.0.engine, &self.0.columns.visible)
    }
}

impl TableProvider for Adapter<'_> {
    /// Records what the kit is about to draw; the view hands it to the engine after the
    /// frame. Never fetches.
    fn prepare(&self, drawn: &Drawn<'_>) {
        let rows = drawn.rows.span().map_or(0..0, |span| {
            let start = u64::try_from(*span.start()).unwrap_or(u64::MAX);
            let end = u64::try_from(*span.end()).unwrap_or(u64::MAX);
            start..end.saturating_add(1)
        });
        let columns: Vec<usize> = drawn
            .columns
            .iter()
            .filter_map(|&column| column.checked_sub(1))
            .collect();
        let positions = self.0.columns.position_span(columns.iter().copied());
        *self.0.recorded.borrow_mut() = Some(Recorded {
            view: Viewport {
                rows,
                columns: positions,
            },
            columns,
        });
    }

    fn cell_status(&self, row_index: usize, col_index: usize) -> CellStatus<'_> {
        let Some(column) = col_index.checked_sub(1) else {
            return CellStatus::Ready;
        };
        let row = u64::try_from(row_index).unwrap_or(u64::MAX);
        match self.0.engine.cell(row, column) {
            Cell::Ready { .. } | Cell::OutOfRange => CellStatus::Ready,
            Cell::Loading | Cell::Deferred => CellStatus::Loading,
            Cell::Failed(error) => CellStatus::Failed(Cow::Borrowed(error.message())),
        }
    }

    fn is_lazy(&self) -> bool {
        true
    }

    fn can_sort(&self, _col_index: usize) -> bool {
        false
    }

    fn can_filter(&self, _col_index: usize) -> bool {
        false
    }

    fn column_count(&self) -> usize {
        self.0.columns.count().saturating_add(1)
    }

    fn header(&self, index: usize) -> Option<Cow<'_, str>> {
        match index.checked_sub(1) {
            None => Some(Cow::Borrowed("#")),
            Some(column) => self
                .0
                .columns
                .names
                .get(column)
                .map(|name| Cow::Borrowed(name.as_str())),
        }
    }

    /// The column's types, codecs, sizes, nulls, and range, from the source's
    /// [`column_info`](parquet_table_core::TableSource::column_info), built the first time
    /// its header is hovered.
    fn header_tooltip(&self, index: usize) -> Option<Cow<'_, str>> {
        let Some(column) = index.checked_sub(1) else {
            return Some(Cow::Borrowed("Row number (from 1)"));
        };
        if column >= self.0.columns.count() {
            return None;
        }
        let mut tooltips = self.0.tooltips.borrow_mut();
        let text = tooltips
            .entry(column)
            .or_insert_with(|| column_tooltip(&self.0.engine.source().column_info(column)));
        Some(Cow::Owned(text.clone()))
    }

    /// The column's short type name, under its name (headers two lines tall).
    fn header_subtitle(&self, index: usize) -> Option<Cow<'_, str>> {
        let column = index.checked_sub(1)?;
        self.0
            .columns
            .types
            .get(column)
            .map(|name| Cow::Borrowed(name.as_str()))
    }

    fn headers(&self) -> HeaderIter<'_> {
        HeaderIter::new(self)
    }

    fn row_count(&self) -> usize {
        self.0.rows
    }

    /// The cell's text: the row number in column 0; elsewhere the value (`null` for a null,
    /// `?` for a value that can't be shown), or nothing while it isn't loaded. A value cut at
    /// the cell's length comes with a longer one (up to the hover length) as its hover text.
    /// Reads the engine's cache only: never the source.
    fn cell_at(
        &self,
        row_index: usize,
        col_index: usize,
    ) -> Result<Option<TableCell<'_>>, TableError> {
        let row = u64::try_from(row_index).unwrap_or(u64::MAX);
        let Some(column) = col_index.checked_sub(1) else {
            return Ok(row
                .checked_add(1)
                .map(|number| (Cow::Owned(number.to_string()), None)));
        };
        Ok(self.text(row, column).map(|text| match text {
            CellText::Null => (Cow::Borrowed("null"), None),
            CellText::Invalid => (Cow::Borrowed("?"), None),
            CellText::Text { text, truncated } => {
                let hover = truncated
                    .then(|| self.full_text(row, column))
                    .flatten()
                    .map(|full| Cow::Owned(full.as_str().to_owned()));
                (Cow::Owned(text), hover)
            }
        }))
    }

    /// Visits the selected rows whose shown cells are all loaded, ascending. Rows that aren't
    /// loaded are left out, and nothing is fetched: the walk takes time bounded by the cache,
    /// however many rows are selected.
    fn for_selected_rows(
        &self,
        state: &TableState,
        f: &mut RowCallback<'_>,
    ) -> Result<(), TableError> {
        for range in self.loaded() {
            for row in selected_in(state, &range) {
                f(&BorrowedRow {
                    provider: self,
                    row_index: usize::try_from(row).unwrap_or(usize::MAX),
                })?;
            }
        }
        Ok(())
    }

    /// Visits the rows whose shown cells are all loaded, ascending. Rows that aren't loaded are
    /// left out, and nothing is fetched: the walk is bounded by the cache.
    fn for_all_rows(&self, f: &mut RowCallback<'_>) -> Result<(), TableError> {
        for range in self.loaded() {
            for row in range {
                let Ok(row_index) = usize::try_from(row) else {
                    return Ok(());
                };
                f(&BorrowedRow {
                    provider: self,
                    row_index,
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{ArrayRef, StringArray};
    use egui_table_kit::operations::VisibleRows;
    use parquet_table_core::arrow::schema::{DataType, Field, Schema};
    use parquet_table_core::{
        BlockId, CancelToken, ColumnInfo, FetchError, LazyOptions, Spawn, TableSource,
    };

    use super::*;

    /// Three Utf8 columns of four rows, one block each; the first value of column 1 is long.
    struct Words {
        schema: SchemaRef,
        fetches: AtomicUsize,
    }

    impl TableSource for Words {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }
        fn row_count(&self) -> u64 {
            4
        }
        fn block_of(&self, column: usize, row: u64) -> Option<BlockId> {
            (column < 3 && row < 4).then_some(BlockId {
                column,
                group: 0,
                index: 0,
            })
        }
        fn block_rows(&self, _block: BlockId) -> Range<u64> {
            0..4
        }
        fn estimated_bytes(&self, _block: BlockId) -> u64 {
            64
        }
        fn fetch(
            &self,
            blocks: &[BlockId],
            _cancel: &CancelToken,
        ) -> Result<Vec<ArrayRef>, FetchError> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            Ok(blocks
                .iter()
                .map(|block| {
                    let long = "y".repeat(1_000);
                    let values = [long.as_str(), "b", "c", "d"]
                        .map(|value| format!("{}{value}", block.column));
                    Arc::new(StringArray::from(values.to_vec())) as ArrayRef
                })
                .collect())
        }
        fn column_info(&self, column: usize) -> ColumnInfo {
            ColumnInfo {
                name: format!("w{column}"),
                ..ColumnInfo::default()
            }
        }
    }

    struct Inline;

    impl Spawn for Inline {
        fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) {
            job();
        }
    }

    fn shared() -> (Shared, Arc<Words>) {
        let schema = Arc::new(Schema::new(
            (0..3)
                .map(|c| Field::new(format!("w{c}"), DataType::Utf8, false))
                .collect::<Vec<_>>(),
        ));
        let source = Arc::new(Words {
            schema: Arc::clone(&schema),
            fetches: AtomicUsize::new(0),
        });
        let engine = LazyTable::new(
            source.clone(),
            Arc::new(Inline),
            Arc::new(|| {}),
            LazyOptions::default(),
        );
        let shared = Shared {
            engine,
            columns: Columns::new(&schema),
            rows: 4,
            cell_chars: 256,
            hover_chars: 4096,
            cells: RefCell::new(CellCache::default()),
            tooltips: RefCell::new(HashMap::new()),
            recorded: RefCell::new(None),
        };
        (shared, source)
    }

    #[test]
    fn columns_map_display_positions() {
        let (mut shared, _) = shared();
        assert_eq!(shared.columns.order(), None);
        shared.columns.show(vec![2, 0, 2, 7]);
        assert_eq!(shared.columns.visible, [2, 0]);
        assert_eq!(shared.columns.positions, [Some(1), None, Some(0)]);
        assert_eq!(shared.columns.order(), Some(vec![2, 0]));
        assert_eq!(shared.columns.position_span([0, 1]), 1..2);
        assert_eq!(shared.columns.position_span([1]), 0..0);
    }

    #[test]
    fn prepare_records_rows_and_display_positions() {
        let (mut shared, _) = shared();
        shared.columns.show(vec![2, 0]);
        let adapter = Adapter(&shared);
        // Provider columns: the row numbers, then source columns 2 and 0.
        adapter.prepare(&Drawn::new(1..3, VisibleRows::Identity(1..3), &[0, 3, 1]));
        assert_eq!(
            shared.recorded.borrow().clone(),
            Some(Recorded {
                view: Viewport {
                    rows: 1..3,
                    columns: 0..2
                },
                columns: vec![2, 0],
            })
        );
    }

    #[test]
    fn cells_come_from_the_cache_with_cut_values_carrying_longer_text() {
        let (mut shared, source) = shared();
        let adapter = Adapter(&shared);
        assert_eq!(adapter.cell_status(0, 2), CellStatus::Loading);
        assert!(adapter.cell_at(0, 2).unwrap().is_none());
        assert_eq!(
            source.fetches.load(Ordering::SeqCst),
            0,
            "asking never fetches"
        );
        assert_eq!(
            adapter.cell_at(41, 0).unwrap().unwrap().0,
            "42",
            "row numbers from 1"
        );

        let view = Viewport {
            rows: 0..4,
            columns: 0..3,
        };
        shared.engine.update(&view, 0.0);
        shared.engine.update(&view, 0.01);
        let fetches = source.fetches.load(Ordering::SeqCst);
        let adapter = Adapter(&shared);
        assert_eq!(adapter.cell_status(0, 2), CellStatus::Ready);
        let (text, hover) = adapter.cell_at(0, 2).unwrap().unwrap();
        assert_eq!(text.chars().count(), 257);
        assert!(text.ends_with('…'));
        assert_eq!(hover.unwrap().chars().count(), 1_001);
        let (text, hover) = adapter.cell_at(1, 2).unwrap().unwrap();
        assert_eq!((text.as_ref(), hover), ("1b", None));
        assert_eq!(source.fetches.load(Ordering::SeqCst), fetches);
        assert!(adapter.header_tooltip(2).unwrap().starts_with("w1"));
        assert_eq!(adapter.header(0).unwrap(), "#");
    }
}
