//! The lazy table driven with synthetic time: what it fetches while scrolling and settling,
//! cancellation, the cache budget, retries, and failures.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

mod common;

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, Int64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet_table_core::{
    BlockId, CancelToken, Cell, CellText, ColumnInfo, FetchError, FetchErrorKind, LazyOptions,
    LazyTable, MemorySource, ParquetLimits, ParquetSource, Spawn, TableSource, Viewport,
    format_cell,
};

/// A table of `columns` Int64 columns whose cell (row, column) is `row * 1000 + column`,
/// in blocks of `block_rows`.
struct Mock {
    schema: SchemaRef,
    rows: u64,
    block_rows: u64,
    fetched: Mutex<Vec<Vec<BlockId>>>,
    script: Mutex<HashMap<BlockId, VecDeque<FetchErrorKind>>>,
}

impl Mock {
    fn new(columns: usize, rows: u64, block_rows: u64) -> Arc<Self> {
        let fields: Vec<Field> = (0..columns)
            .map(|c| Field::new(format!("c{c}"), DataType::Int64, false))
            .collect();
        Arc::new(Self {
            schema: Arc::new(Schema::new(fields)),
            rows,
            block_rows,
            fetched: Mutex::new(Vec::new()),
            script: Mutex::new(HashMap::new()),
        })
    }

    fn fail(&self, block: BlockId, kinds: &[FetchErrorKind]) {
        self.script
            .lock()
            .unwrap()
            .insert(block, kinds.iter().copied().collect());
    }

    fn fetched_blocks(&self) -> Vec<BlockId> {
        self.fetched
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect()
    }

    fn block(&self, column: usize, index: u32) -> BlockId {
        BlockId {
            column,
            group: 0,
            index,
        }
    }
}

impl TableSource for Mock {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn row_count(&self) -> u64 {
        self.rows
    }

    fn block_of(&self, column: usize, row: u64) -> Option<BlockId> {
        (column < self.schema.fields().len() && row < self.rows)
            .then(|| self.block(column, (row / self.block_rows) as u32))
    }

    fn block_rows(&self, block: BlockId) -> Range<u64> {
        let start = u64::from(block.index) * self.block_rows;
        start..(start + self.block_rows).min(self.rows)
    }

    fn estimated_bytes(&self, block: BlockId) -> u64 {
        let rows = self.block_rows(block);
        (rows.end - rows.start) * 8
    }

    fn fetch(&self, blocks: &[BlockId], cancel: &CancelToken) -> Result<Vec<ArrayRef>, FetchError> {
        self.fetched.lock().unwrap().push(blocks.to_vec());
        if cancel.is_cancelled() {
            return Err(FetchError::cancelled());
        }
        for block in blocks {
            let next = self
                .script
                .lock()
                .unwrap()
                .get_mut(block)
                .and_then(VecDeque::pop_front);
            match next {
                Some(FetchErrorKind::Retryable) => return Err(FetchError::retryable("flaky")),
                Some(FetchErrorKind::Permanent) => return Err(FetchError::permanent("broken")),
                Some(FetchErrorKind::Limit) => return Err(FetchError::limit("too big")),
                Some(FetchErrorKind::Cancelled) => return Err(FetchError::cancelled()),
                None => {}
            }
        }
        Ok(blocks
            .iter()
            .map(|b| {
                let rows = self.block_rows(*b);
                Arc::new(
                    rows.map(|r| r as i64 * 1000 + b.column as i64)
                        .collect::<Int64Array>(),
                ) as ArrayRef
            })
            .collect())
    }

    fn column_info(&self, _column: usize) -> ColumnInfo {
        ColumnInfo::default()
    }
}

/// Runs each job at once, inside `spawn`.
struct Inline;

impl Spawn for Inline {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) {
        job();
    }
}

/// Holds jobs until the test runs them.
#[derive(Default)]
struct Manual(Mutex<Vec<Box<dyn FnOnce() + Send + 'static>>>);

impl Manual {
    fn run_all(&self) -> usize {
        let jobs: Vec<_> = std::mem::take(&mut *self.0.lock().unwrap());
        let n = jobs.len();
        for job in jobs {
            job();
        }
        n
    }
}

impl Spawn for Manual {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) {
        self.0.lock().unwrap().push(job);
    }
}

fn table(
    source: Arc<dyn TableSource>,
    spawner: Arc<dyn Spawn>,
    options: LazyOptions,
) -> (LazyTable, Arc<AtomicUsize>) {
    let wakes = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&wakes);
    let wake = Arc::new(move || {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    (LazyTable::new(source, spawner, wake, options), wakes)
}

fn view(top: u64, height: u64, columns: Range<usize>) -> Viewport {
    Viewport {
        rows: top..top + height,
        columns,
    }
}

fn value(table: &LazyTable, row: u64, column: usize) -> Option<i64> {
    match table.cell(row, column) {
        Cell::Ready { array, index } => Some(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(index),
        ),
        _ => None,
    }
}

const FRAME: f64 = 1.0 / 60.0;

#[test]
fn a_fast_scroll_fetches_nothing_until_it_settles() {
    let mock = Mock::new(10, 1_000_000, 1_000);
    let (mut table, _) = table(mock.clone(), Arc::new(Inline), LazyOptions::default());
    table.update(&view(0, 40, 0..10), 0.0);
    let at_top = table.stats().fetches_started;
    assert!(at_top > 0, "the first view is fetched");
    table.update(&view(0, 40, 0..10), FRAME);
    assert_eq!(value(&table, 5, 3), Some(5_003));

    // A fling to the bottom in one second.
    let mut now = FRAME;
    for frame in 1..=60u64 {
        now = FRAME + frame as f64 * FRAME;
        let top = frame * (1_000_000 - 40) / 60;
        table.update(&view(top, 40, 0..10), now);
        assert!(table.stats().moving, "frame {frame}");
        if frame < 60 {
            assert!(matches!(table.cell(top, 0), Cell::Deferred));
        }
    }
    assert_eq!(
        table.stats().fetches_started,
        at_top,
        "nothing fetched while moving"
    );
    assert!(table.repaint_after().is_some_and(|t| t > 0.0 && t <= 0.12));

    // Resting at the bottom: still deferred until the settle time has passed.
    let bottom = 1_000_000 - 40;
    table.update(&view(bottom, 40, 0..10), now + 0.05);
    assert_eq!(table.stats().fetches_started, at_top);
    table.update(&view(bottom, 40, 0..10), now + 0.13);
    assert!(!table.stats().moving);
    assert!(table.stats().fetches_started > at_top);
    assert!(table.repaint_after().is_none());
    table.update(&view(bottom, 40, 0..10), now + 0.14);
    assert_eq!(
        value(&table, bottom + 39, 9),
        Some((bottom as i64 + 39) * 1000 + 9)
    );
    // Nothing between top and bottom was fetched.
    let indices: Vec<u32> = mock.fetched_blocks().iter().map(|b| b.index).collect();
    assert!(indices.iter().all(|&i| i <= 1 || i >= 998), "{indices:?}");
}

#[test]
fn settling_fetches_visible_blocks_nearest_the_centre_first_then_prefetches() {
    let mock = Mock::new(10, 100_000, 1_000);
    let manual = Arc::new(Manual::default());
    let options = LazyOptions {
        max_jobs: 100,
        max_columns_per_fetch: 1,
        ..LazyOptions::default()
    };
    let (mut table, _) = table(mock.clone(), manual.clone(), options);
    // Rows 5000..5040 lie in block 5; columns 3, 4, 5 are visible.
    table.update(&view(5_000, 40, 3..6), 0.0);
    assert!(matches!(table.cell(5_000, 4), Cell::Loading));
    assert!(matches!(table.cell(50_000, 4), Cell::Deferred));
    manual.run_all();
    let order = mock.fetched_blocks();
    assert_eq!(order[0], mock.block(4, 5), "the centre first");
    let visible: Vec<_> = order[..3].to_vec();
    for column in 3..6 {
        assert!(visible.contains(&mock.block(column, 5)));
    }
    // Then the ring: rows one viewport up (block 4) and down, two columns either side.
    let rest: Vec<_> = order[3..].to_vec();
    assert!(rest.contains(&mock.block(4, 4)));
    assert!(rest.contains(&mock.block(1, 5)));
    assert!(rest.contains(&mock.block(7, 4)));
    assert!(!rest.contains(&mock.block(0, 5)));
    assert!(!rest.contains(&mock.block(8, 5)));
    assert!(rest.iter().all(|b| !visible.contains(b)));
}

#[test]
fn blocks_with_the_same_rows_share_a_fetch() {
    let mock = Mock::new(20, 100_000, 1_000);
    let manual = Arc::new(Manual::default());
    let options = LazyOptions {
        max_columns_per_fetch: 4,
        prefetch_columns: 0,
        prefetch_viewports: 0.0,
        prefetch_bias_viewports: 0.0,
        ..LazyOptions::default()
    };
    let (mut table, _) = table(mock.clone(), manual.clone(), options);
    table.update(&view(10, 40, 0..10), 0.0);
    manual.run_all();
    let calls = mock.fetched.lock().unwrap().clone();
    assert_eq!(calls.len(), 3, "10 columns in fetches of at most 4");
    assert!(calls.iter().all(|c| c.len() <= 4));
    table.update(&view(10, 40, 0..10), 0.01);
    for column in 0..10 {
        assert_eq!(value(&table, 20, column), Some(20_000 + column as i64));
    }
}

#[test]
fn moving_away_cancels_stale_jobs_and_drops_queued_work() {
    let mock = Mock::new(10, 1_000_000, 1_000);
    let manual = Arc::new(Manual::default());
    let options = LazyOptions {
        max_jobs: 2,
        max_columns_per_fetch: 1,
        ..LazyOptions::default()
    };
    let (mut table, _) = table(mock.clone(), manual.clone(), options);
    table.update(&view(0, 40, 0..6), 0.0);
    assert_eq!(table.stats().inflight_jobs, 2);
    // Jump far away and settle there.
    table.update(&view(500_000, 40, 0..6), 0.1);
    table.update(&view(500_000, 40, 0..6), 0.3);
    let stats = table.stats();
    assert_eq!(stats.fetches_cancelled, 2);
    assert_eq!(
        stats.inflight_jobs, 2,
        "cancelled jobs keep their slots until they end"
    );
    assert_eq!(stats.fetches_started, 2);
    // The cancelled jobs end without fetching; their slots go to the new view.
    assert_eq!(manual.run_all(), 2);
    assert!(mock.fetched_blocks().is_empty());
    table.update(&view(500_000, 40, 0..6), 0.31);
    assert_eq!(table.stats().fetches_started, 4);
    while manual.run_all() > 0 {
        table.update(&view(500_000, 40, 0..6), 0.32);
    }
    let fetched = mock.fetched_blocks();
    assert!(fetched.iter().all(|b| b.index >= 499), "{fetched:?}");
    assert_eq!(value(&table, 500_010, 5), Some(500_010 * 1000 + 5));
    assert!(matches!(table.cell(10, 0), Cell::Deferred));
}

#[test]
fn the_cache_stays_within_its_budget_and_keeps_what_is_in_view() {
    let mock = Mock::new(10, 1_000_000, 1_000);
    let block_bytes = Arc::new(Int64Array::from(vec![0i64; 1000])).get_array_memory_size() as u64;
    let options = LazyOptions {
        cache_bytes: block_bytes * 5,
        ..LazyOptions::default()
    };
    let (mut table, _) = table(mock.clone(), Arc::new(Inline), options);
    let mut now = 0.0;
    for step in 0..30u64 {
        let top = step * 7_919 % 990_000;
        let columns = (step as usize % 7)..(step as usize % 7 + 3);
        // Arrive, settle, and receive.
        for _ in 0..3 {
            now += 0.2;
            table.update(&view(top, 40, columns.clone()), now);
        }
        let stats = table.stats();
        for column in columns.clone() {
            for row in [top, top + 39] {
                assert!(
                    value(&table, row, column).is_some(),
                    "step {step}: row {row} column {column}"
                );
            }
        }
        let visible_blocks = columns.len() as u64 * 2;
        assert!(
            stats.cached_bytes <= options_cache(block_bytes).max(visible_blocks * block_bytes),
            "step {step}: {} cached",
            stats.cached_bytes
        );
    }
    assert!(table.stats().evictions > 0);
}

fn options_cache(block_bytes: u64) -> u64 {
    block_bytes * 5
}

#[test]
fn retryable_failures_back_off_then_recover() {
    let mock = Mock::new(1, 10_000, 1_000);
    mock.fail(
        mock.block(0, 0),
        &[FetchErrorKind::Retryable, FetchErrorKind::Retryable],
    );
    let options = LazyOptions {
        prefetch_viewports: 0.0,
        prefetch_bias_viewports: 0.0,
        ..LazyOptions::default()
    };
    let (mut table, _) = table(mock.clone(), Arc::new(Inline), options);
    let v = view(0, 40, 0..1);
    table.update(&v, 0.0); // fetch 1 fails
    table.update(&v, 0.01); // receives the failure: retry at 0.26
    assert!(matches!(table.cell(0, 0), Cell::Loading));
    let wait = table.repaint_after().unwrap();
    assert!((wait - 0.25).abs() < 1e-9, "{wait}");
    table.update(&v, 0.2);
    assert_eq!(mock.fetched_blocks().len(), 1, "not before the backoff");
    table.update(&v, 0.27); // fetch 2 fails
    table.update(&v, 0.28); // retry at 0.78 (doubled)
    assert!((table.repaint_after().unwrap() - 0.5).abs() < 1e-9);
    table.update(&v, 0.5);
    assert_eq!(mock.fetched_blocks().len(), 2);
    table.update(&v, 0.79); // fetch 3 succeeds
    table.update(&v, 0.8);
    assert_eq!(value(&table, 3, 0), Some(3_000));
    let stats = table.stats();
    assert_eq!((stats.retries, stats.fetch_errors), (2, 2));
}

#[test]
fn retries_are_bounded() {
    let mock = Mock::new(1, 10_000, 1_000);
    mock.fail(mock.block(0, 0), &[FetchErrorKind::Retryable; 10]);
    let options = LazyOptions {
        max_retries: 2,
        prefetch_viewports: 0.0,
        prefetch_bias_viewports: 0.0,
        ..LazyOptions::default()
    };
    let (mut table, _) = table(mock.clone(), Arc::new(Inline), options);
    let mut now = 0.0;
    for _ in 0..50 {
        now += 0.1;
        table.update(&view(0, 40, 0..1), now);
    }
    assert_eq!(mock.fetched_blocks().len(), 3, "one try and two retries");
    assert!(matches!(table.cell(0, 0), Cell::Failed(e) if e.kind() == FetchErrorKind::Retryable));
}

#[test]
fn permanent_failures_and_limits_are_shown_and_not_retried() {
    let mock = Mock::new(2, 10_000, 1_000);
    mock.fail(mock.block(0, 0), &[FetchErrorKind::Permanent]);
    mock.fail(mock.block(1, 0), &[FetchErrorKind::Limit]);
    let options = LazyOptions {
        max_columns_per_fetch: 1,
        prefetch_viewports: 0.0,
        prefetch_bias_viewports: 0.0,
        ..LazyOptions::default()
    };
    let (mut table, _) = table(mock.clone(), Arc::new(Inline), options);
    for i in 0..10 {
        table.update(&view(0, 40, 0..2), f64::from(i) * 0.5);
    }
    assert!(matches!(table.cell(0, 0), Cell::Failed(e) if e.kind() == FetchErrorKind::Permanent));
    assert!(matches!(table.cell(39, 1), Cell::Failed(e) if e.kind() == FetchErrorKind::Limit));
    assert_eq!(mock.fetched_blocks().len(), 2);
    assert_eq!(table.stats().failed_blocks, 2);
    // Asked to, the table tries again.
    table.retry_failed();
    table.update(&view(0, 40, 0..2), 10.0);
    table.update(&view(0, 40, 0..2), 10.1);
    assert_eq!(value(&table, 0, 0), Some(0));
}

#[test]
fn cells_out_of_range() {
    let mock = Mock::new(2, 100, 10);
    let (mut table, _) = table(mock, Arc::new(Inline), LazyOptions::default());
    table.update(&view(90, 40, 0..5), 0.0);
    table.update(&view(90, 40, 0..5), 0.01);
    assert!(matches!(table.cell(100, 0), Cell::OutOfRange));
    assert!(matches!(table.cell(0, 2), Cell::OutOfRange));
    assert_eq!(value(&table, 99, 1), Some(99_001));
}

#[test]
fn a_parquet_file_through_threads() {
    let bytes = common::flat_file(30_000, &common::Options::default());
    let expected = common::eager(&bytes);
    let limits = ParquetLimits {
        target_block_bytes: 16 << 10,
        ..ParquetLimits::default()
    };
    let source = Arc::new(ParquetSource::open(MemorySource::new(bytes), &limits).unwrap());
    let spawner = Arc::new(|job: Box<dyn FnOnce() + Send + 'static>| {
        std::thread::spawn(job);
    });
    let (mut table, wakes) = table(source, spawner, LazyOptions::default());
    let v = view(12_345, 50, 0..7);
    let mut now = 0.0;
    let mut ready = false;
    for _ in 0..2_000 {
        table.update(&v, now);
        ready = (12_345..12_395)
            .all(|row| (0..7).all(|column| matches!(table.cell(row, column), Cell::Ready { .. })));
        if ready {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
        now += 0.002;
    }
    assert!(ready);
    assert!(wakes.load(Ordering::Relaxed) > 0);
    for row in 12_345..12_395u64 {
        for column in 0..7 {
            let Cell::Ready { array, index } = table.cell(row, column) else {
                unreachable!();
            };
            let got = format_cell(array.as_ref(), index, 100);
            let want = format_cell(expected.column(column).as_ref(), row as usize, 100);
            assert_eq!(got, want, "row {row} column {column}");
            if column == 1 && row % 1000 % 7 == 0 {
                assert_eq!(got, CellText::Null);
            }
        }
    }
}

#[test]
fn the_engine_can_move_between_threads_and_sources_are_shared() {
    fn send<T: Send>() {}
    fn shared<T: Send + Sync>() {}
    send::<LazyTable>();
    shared::<ParquetSource<MemorySource>>();
    shared::<ParquetSource<parquet_table_core::FileSource>>();
    shared::<CancelToken>();
}
