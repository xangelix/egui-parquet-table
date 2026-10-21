//! The lazy table: what a widget drives once per frame on its UI thread.
//!
//! [`LazyTable`] owns the block cache and the bookkeeping of fetch jobs. It is `Send` but
//! never shared: the UI thread calls [`update`](LazyTable::update) each frame and
//! [`cell`](LazyTable::cell) for each visible cell, and nothing it touches is locked.
//! Fetches run on workers handed out through [`Spawn`]; a worker delivers its result
//! through a bounded lock-free queue and calls `wake`, and the next `update` drains the
//! queue without blocking. The queue holds one slot per job the table counts as running
//! (cancelled jobs included, until their result arrives), so a push never fails.
//!
//! The [`policy`] decides when to fetch: while the view moves fast nothing new is fetched
//! (uncached cells are [`Cell::Deferred`]); once it settles, the visible blocks are fetched
//! nearest the centre first, then a prefetch ring around them. Every update recomputes
//! what is wanted, so work for rows the user left behind is cancelled.

pub mod policy;

use core::ops::Range;
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use concurrent_queue::ConcurrentQueue;

use self::policy::{Direction, Motion, PolicyConfig, ScrollPolicy};
use crate::cancel::CancelToken;
use crate::error::{FetchError, FetchErrorKind};
use crate::table::{BlockId, TableSource};

/// Runs fetch jobs off the UI thread.
///
/// Every job handed to `spawn` must eventually run (a dropped job keeps its slot forever).
/// Jobs may block; they never touch the UI thread's state.
pub trait Spawn: Send + Sync {
    /// Runs `job` on a worker.
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>);
}

impl<F> Spawn for F
where
    F: Fn(Box<dyn FnOnce() + Send + 'static>) + Send + Sync,
{
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) {
        self(job);
    }
}

/// Tuning of a [`LazyTable`].
#[derive(Debug, Clone, PartialEq)]
pub struct LazyOptions {
    /// Fetch jobs running at once.
    pub max_jobs: usize,
    /// Estimated decoded bytes of running jobs (one job may exceed it when it runs alone).
    pub max_inflight_bytes: u64,
    /// Decoded bytes kept in the cache (exceeded only by blocks in view).
    pub cache_bytes: u64,
    /// Columns fetched together in one job.
    pub max_columns_per_fetch: usize,
    /// Rows prefetched above and below the view, in viewport heights.
    pub prefetch_viewports: f64,
    /// Extra rows prefetched in the scroll direction, in viewport heights.
    pub prefetch_bias_viewports: f64,
    /// Columns prefetched on either side of the view.
    pub prefetch_columns: usize,
    /// The scroll policy's tuning.
    pub policy: PolicyConfig,
    /// Retries of a block after a retryable failure.
    pub max_retries: u32,
    /// The first retry's delay in seconds; each further retry doubles it.
    pub retry_backoff_seconds: f64,
    /// Failed blocks remembered (beyond those in view).
    pub max_failed_blocks: usize,
    /// Blocks considered per update (bounds work for tiny blocks and huge views).
    pub max_blocks_per_update: usize,
}

impl Default for LazyOptions {
    fn default() -> Self {
        Self {
            max_jobs: 4,
            max_inflight_bytes: 64 << 20,
            cache_bytes: 128 << 20,
            max_columns_per_fetch: 8,
            prefetch_viewports: 1.0,
            prefetch_bias_viewports: 1.0,
            prefetch_columns: 2,
            policy: PolicyConfig::default(),
            max_retries: 3,
            retry_backoff_seconds: 0.25,
            max_failed_blocks: 4096,
            max_blocks_per_update: 4096,
        }
    }
}

/// The part of the table about to be drawn.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Viewport {
    /// Visible rows.
    pub rows: Range<u64>,
    /// Visible column positions (indices into the column order, which is the schema
    /// order unless [`LazyTable::set_column_order`] changed it).
    pub columns: Range<usize>,
}

/// What a cell shows.
#[derive(Debug, Clone, Copy)]
pub enum Cell<'a> {
    /// The value is at `index` of `array`.
    Ready {
        /// The block's array.
        array: &'a ArrayRef,
        /// The row within the block.
        index: usize,
    },
    /// Being fetched (or waiting for a slot or a retry).
    Loading,
    /// Not fetched while the view moves fast.
    Deferred,
    /// The block failed.
    Failed(&'a FetchError),
    /// No such row or column.
    OutOfRange,
}

/// Counters for a status line or a debugging view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Decoded bytes in the cache.
    pub cached_bytes: u64,
    /// Blocks in the cache.
    pub cached_blocks: usize,
    /// Blocks remembered as failed.
    pub failed_blocks: usize,
    /// Jobs running (cancelled ones included until they end).
    pub inflight_jobs: usize,
    /// Estimated decoded bytes of running jobs.
    pub inflight_bytes: u64,
    /// Decoded bytes received from fetches.
    pub bytes_fetched: u64,
    /// Jobs started.
    pub fetches_started: u64,
    /// Jobs that returned data.
    pub fetches_completed: u64,
    /// Jobs cancelled because their rows were no longer wanted.
    pub fetches_cancelled: u64,
    /// Jobs that failed (retryable failures included).
    pub fetch_errors: u64,
    /// Retries scheduled.
    pub retries: u64,
    /// Blocks evicted from the cache.
    pub evictions: u64,
    /// Whether the view is moving fast (no new fetches).
    pub moving: bool,
}

#[derive(Debug)]
enum Entry {
    Ready { array: ArrayRef, bytes: u64 },
    Failed(FetchError),
    Retry { attempts: u32, at: f64 },
}

#[derive(Debug)]
struct Slot {
    entry: Entry,
    /// The last frame the block was in view (or arrived).
    used: u64,
}

#[derive(Debug)]
struct Job {
    blocks: Vec<BlockId>,
    cancel: CancelToken,
    bytes: u64,
    cancelled: bool,
}

struct Completion {
    id: u64,
    result: Result<Vec<ArrayRef>, FetchError>,
}

/// A wanted block and its priority (lower first).
#[derive(Debug, Clone, Copy)]
struct Want {
    block: BlockId,
    priority: f64,
    visible: bool,
}

/// The lazy, cached view of a [`TableSource`].
pub struct LazyTable {
    source: Arc<dyn TableSource>,
    spawner: Arc<dyn Spawn>,
    wake: Arc<dyn Fn() + Send + Sync>,
    options: LazyOptions,
    policy: ScrollPolicy,
    results: Arc<ConcurrentQueue<Completion>>,
    cache: HashMap<BlockId, Slot>,
    cached_bytes: u64,
    jobs: HashMap<u64, Job>,
    /// Blocks being fetched by a job that is not cancelled.
    pending: HashMap<BlockId, u64>,
    next_job: u64,
    visible: HashSet<BlockId>,
    wanted: HashSet<BlockId>,
    order: Option<Vec<usize>>,
    frame: u64,
    now: f64,
    motion: Motion,
    repaint_after: Option<f64>,
    stats: Stats,
}

impl core::fmt::Debug for LazyTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LazyTable")
            .field("options", &self.options)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl Drop for LazyTable {
    fn drop(&mut self) {
        for job in self.jobs.values() {
            job.cancel.cancel();
        }
    }
}

/// Converts with saturation (row counts and positions to floating point for ordering).
#[allow(clippy::cast_precision_loss)]
fn as_f64(n: u64) -> f64 {
    n as f64
}

impl LazyTable {
    /// A table over `source`. Fetch jobs are handed to `spawner`; `wake` is called by a
    /// worker after it delivered a result (typically: request a repaint).
    pub fn new(
        source: Arc<dyn TableSource>,
        spawner: Arc<dyn Spawn>,
        wake: Arc<dyn Fn() + Send + Sync>,
        options: LazyOptions,
    ) -> Self {
        let capacity = options.max_jobs.max(1);
        Self {
            source,
            spawner,
            wake,
            policy: ScrollPolicy::new(options.policy),
            options,
            results: Arc::new(ConcurrentQueue::bounded(capacity)),
            cache: HashMap::new(),
            cached_bytes: 0,
            jobs: HashMap::new(),
            pending: HashMap::new(),
            next_job: 0,
            visible: HashSet::new(),
            wanted: HashSet::new(),
            order: None,
            frame: 0,
            now: 0.0,
            motion: Motion::Settled,
            repaint_after: None,
            stats: Stats::default(),
        }
    }

    /// The source.
    pub fn source(&self) -> &Arc<dyn TableSource> {
        &self.source
    }

    /// The options.
    pub fn options(&self) -> &LazyOptions {
        &self.options
    }

    /// Sets the display order of columns: position `i` shows source column `order[i]`.
    /// `None` restores the schema order. Out-of-range entries are ignored.
    pub fn set_column_order(&mut self, order: Option<Vec<usize>>) {
        let count = self.source.column_count();
        self.order = order.map(|o| o.into_iter().filter(|&c| c < count).collect());
    }

    /// The source column at a display position.
    pub fn column_at(&self, position: usize) -> Option<usize> {
        match &self.order {
            Some(order) => order.get(position).copied(),
            None => (position < self.source.column_count()).then_some(position),
        }
    }

    /// The number of display positions.
    pub fn column_positions(&self) -> usize {
        match &self.order {
            Some(order) => order.len(),
            None => self.source.column_count(),
        }
    }

    /// Forgets failed blocks so they are fetched again.
    pub fn retry_failed(&mut self) {
        self.cache
            .retain(|_, slot| !matches!(slot.entry, Entry::Failed(_) | Entry::Retry { .. }));
    }

    /// Once per frame: the rows and columns about to be drawn, and the caller's clock in
    /// seconds. Never blocks.
    pub fn update(&mut self, view: &Viewport, now: f64) {
        if now.is_finite() && now > self.now {
            self.now = now;
        }
        let now = self.now;
        self.frame = self.frame.saturating_add(1);
        self.drain();
        let height = view.rows.end.saturating_sub(view.rows.start);
        self.motion = self.policy.observe(view.rows.start, height, now);

        let wants = self.wants(view);
        self.visible = wants
            .iter()
            .filter(|w| w.visible)
            .map(|w| w.block)
            .collect();
        for block in &self.visible {
            if let Some(slot) = self.cache.get_mut(block) {
                slot.used = self.frame;
            }
        }
        let settled = self.motion == Motion::Settled;
        let fetchable: Vec<Want> = if settled {
            wants
                .into_iter()
                .filter(|w| match self.cache.get(&w.block) {
                    None => true,
                    Some(slot) => matches!(slot.entry, Entry::Retry { at, .. } if at <= now),
                })
                .collect()
        } else {
            Vec::new()
        };
        self.wanted = fetchable.iter().map(|w| w.block).collect();
        self.cancel_stale();
        self.start_jobs(&fetchable);
        self.evict();
        self.repaint_after = self.next_timer(now);
    }

    /// What the cell at `row` of source column `column` shows.
    pub fn cell(&self, row: u64, column: usize) -> Cell<'_> {
        let Some(block) = self.source.block_of(column, row) else {
            return Cell::OutOfRange;
        };
        match self.cache.get(&block).map(|slot| &slot.entry) {
            Some(Entry::Ready { array, .. }) => {
                let start = self.source.block_rows(block).start;
                match row.checked_sub(start).and_then(|i| usize::try_from(i).ok()) {
                    Some(index) if index < array.len() => Cell::Ready { array, index },
                    _ => Cell::Loading,
                }
            }
            Some(Entry::Failed(error)) => Cell::Failed(error),
            Some(Entry::Retry { .. }) => Cell::Loading,
            None if self.pending.contains_key(&block) || self.wanted.contains(&block) => {
                Cell::Loading
            }
            None => Cell::Deferred,
        }
    }

    /// The blocks whose data is cached (those [`cell`](Self::cell) shows as ready), in no
    /// particular order. Never fetches: a widget uses it to visit only loaded rows (copying a
    /// selection of billions of rows, for instance) in time bounded by the cache.
    pub fn ready_blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.cache
            .iter()
            .filter(|(_, slot)| matches!(slot.entry, Entry::Ready { .. }))
            .map(|(block, _)| *block)
    }

    /// Seconds until the table wants another [`update`](Self::update) without new input
    /// (the settle timer or a retry), or `None`. Results arriving from workers call `wake`
    /// instead.
    pub fn repaint_after(&self) -> Option<f64> {
        self.repaint_after
    }

    /// Counters.
    pub fn stats(&self) -> Stats {
        let failed = self
            .cache
            .values()
            .filter(|s| matches!(s.entry, Entry::Failed(_)))
            .count();
        Stats {
            cached_bytes: self.cached_bytes,
            cached_blocks: self.cache.len().saturating_sub(failed),
            failed_blocks: failed,
            inflight_jobs: self.jobs.len(),
            inflight_bytes: self.inflight_bytes(),
            moving: self.motion != Motion::Settled,
            ..self.stats
        }
    }

    fn inflight_bytes(&self) -> u64 {
        self.jobs
            .values()
            .map(|j| j.bytes)
            .fold(0u64, u64::saturating_add)
    }

    /// Takes every delivered result, without blocking.
    fn drain(&mut self) {
        while let Ok(completion) = self.results.pop() {
            self.complete(completion);
        }
    }

    fn insert(&mut self, block: BlockId, entry: Entry) {
        let bytes = match &entry {
            Entry::Ready { bytes, .. } => *bytes,
            _ => 0,
        };
        let slot = Slot {
            entry,
            used: self.frame,
        };
        if let Some(old) = self.cache.insert(block, slot)
            && let Entry::Ready { bytes, .. } = old.entry
        {
            self.cached_bytes = self.cached_bytes.saturating_sub(bytes);
        }
        self.cached_bytes = self.cached_bytes.saturating_add(bytes);
    }

    fn complete(&mut self, completion: Completion) {
        let Some(job) = self.jobs.remove(&completion.id) else {
            return;
        };
        for block in &job.blocks {
            if self.pending.get(block) == Some(&completion.id) {
                self.pending.remove(block);
            }
        }
        let result = completion.result.and_then(|arrays| {
            if arrays.len() == job.blocks.len() {
                Ok(arrays)
            } else {
                Err(FetchError::permanent(
                    "the source returned the wrong number of columns",
                ))
            }
        });
        match result {
            Ok(arrays) => {
                self.stats.fetches_completed = self.stats.fetches_completed.saturating_add(1);
                for (block, array) in job.blocks.iter().zip(arrays) {
                    let rows = self.source.block_rows(*block);
                    let expected = rows.end.saturating_sub(rows.start);
                    if u64::try_from(array.len()).ok() != Some(expected) {
                        self.insert(
                            *block,
                            Entry::Failed(FetchError::permanent(
                                "the source returned the wrong number of rows",
                            )),
                        );
                        continue;
                    }
                    let bytes = u64::try_from(array.get_array_memory_size()).unwrap_or(u64::MAX);
                    self.stats.bytes_fetched = self.stats.bytes_fetched.saturating_add(bytes);
                    self.insert(*block, Entry::Ready { array, bytes });
                }
            }
            Err(error) => match error.kind() {
                FetchErrorKind::Cancelled => {}
                FetchErrorKind::Retryable => {
                    self.stats.fetch_errors = self.stats.fetch_errors.saturating_add(1);
                    for block in &job.blocks {
                        let attempts = match self.cache.get(block).map(|s| &s.entry) {
                            Some(Entry::Retry { attempts, .. }) => attempts.saturating_add(1),
                            Some(Entry::Ready { .. }) => continue,
                            _ => 1,
                        };
                        if attempts > self.options.max_retries {
                            self.insert(*block, Entry::Failed(error.clone()));
                        } else {
                            self.stats.retries = self.stats.retries.saturating_add(1);
                            let exponent = i32::try_from(attempts.saturating_sub(1)).unwrap_or(30);
                            let delay = self.options.retry_backoff_seconds * 2f64.powi(exponent);
                            let at = self.now + delay;
                            self.insert(*block, Entry::Retry { attempts, at });
                        }
                    }
                }
                FetchErrorKind::Permanent | FetchErrorKind::Limit => {
                    self.stats.fetch_errors = self.stats.fetch_errors.saturating_add(1);
                    for block in &job.blocks {
                        self.insert(*block, Entry::Failed(error.clone()));
                    }
                }
            },
        }
    }

    /// The blocks of the given rows and column positions, with priorities.
    #[allow(clippy::too_many_arguments)]
    fn collect(
        &self,
        rows: Range<u64>,
        positions: Range<usize>,
        center: (f64, f64),
        scale: (f64, f64),
        visible: bool,
        out: &mut Vec<Want>,
        seen: &mut HashSet<BlockId>,
    ) {
        for position in positions {
            let Some(column) = self.column_at(position) else {
                continue;
            };
            let mut row = rows.start;
            while row < rows.end && out.len() < self.options.max_blocks_per_update {
                let Some(block) = self.source.block_of(column, row) else {
                    break;
                };
                let block_rows = self.source.block_rows(block);
                if block_rows.end <= row {
                    break;
                }
                if seen.insert(block) {
                    let lo = block_rows.start.max(rows.start);
                    let hi = block_rows.end.min(rows.end);
                    let mid = (as_f64(lo) + as_f64(hi)) / 2.0;
                    let distance = ((mid - center.0) / scale.0).abs()
                        + ((as_f64(u64::try_from(position).unwrap_or(u64::MAX)) - center.1)
                            / scale.1)
                            .abs();
                    out.push(Want {
                        block,
                        priority: if visible { distance } else { 1e9 + distance },
                        visible,
                    });
                }
                row = block_rows.end;
            }
        }
    }

    /// Visible blocks, then the prefetch ring, in priority order.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn wants(&self, view: &Viewport) -> Vec<Want> {
        let total_rows = self.source.row_count();
        let positions = self.column_positions();
        let rows = view.rows.start.min(total_rows)..view.rows.end.min(total_rows);
        let columns = view.columns.start.min(positions)..view.columns.end.min(positions);
        if rows.is_empty() || columns.is_empty() {
            return Vec::new();
        }
        let height = rows.end.saturating_sub(rows.start);
        let width = columns.end.saturating_sub(columns.start);
        let center = (
            (as_f64(rows.start) + as_f64(rows.end)) / 2.0,
            (as_f64(u64::try_from(columns.start).unwrap_or(0))
                + as_f64(u64::try_from(columns.end).unwrap_or(0))
                - 1.0)
                / 2.0,
        );
        let scale = (
            as_f64(height).max(1.0),
            as_f64(u64::try_from(width).unwrap_or(1)).max(1.0),
        );
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        self.collect(
            rows.clone(),
            columns.clone(),
            center,
            scale,
            true,
            &mut out,
            &mut seen,
        );
        let ring = |viewports: f64| -> u64 {
            let rows = (as_f64(height) * viewports.max(0.0)).ceil();
            if rows >= as_f64(u64::MAX) {
                u64::MAX
            } else {
                rows as u64
            }
        };
        let (above, below) = match self.policy.direction() {
            Direction::Up => (
                self.options.prefetch_viewports + self.options.prefetch_bias_viewports,
                self.options.prefetch_viewports,
            ),
            Direction::Down => (
                self.options.prefetch_viewports,
                self.options.prefetch_viewports + self.options.prefetch_bias_viewports,
            ),
            Direction::None => (
                self.options.prefetch_viewports,
                self.options.prefetch_viewports,
            ),
        };
        let ring_rows = rows.start.saturating_sub(ring(above))
            ..rows.end.saturating_add(ring(below)).min(total_rows);
        let ring_columns = columns.start.saturating_sub(self.options.prefetch_columns)
            ..columns
                .end
                .saturating_add(self.options.prefetch_columns)
                .min(positions);
        self.collect(
            ring_rows,
            ring_columns,
            center,
            scale,
            false,
            &mut out,
            &mut seen,
        );
        out.sort_by(|a, b| a.priority.total_cmp(&b.priority));
        out
    }

    /// Cancels jobs none of whose blocks are in view or wanted.
    fn cancel_stale(&mut self) {
        let mut cancelled = Vec::new();
        for (&id, job) in &mut self.jobs {
            if job.cancelled {
                continue;
            }
            let keep = job
                .blocks
                .iter()
                .any(|b| self.visible.contains(b) || self.wanted.contains(b));
            if !keep {
                job.cancel.cancel();
                job.cancelled = true;
                cancelled.push((id, job.blocks.clone()));
            }
        }
        for (id, blocks) in cancelled {
            self.stats.fetches_cancelled = self.stats.fetches_cancelled.saturating_add(1);
            for block in blocks {
                if self.pending.get(&block) == Some(&id) {
                    self.pending.remove(&block);
                }
            }
        }
    }

    /// Groups the wanted blocks into multi-column fetches and starts as many as the
    /// limits allow, in priority order.
    fn start_jobs(&mut self, wants: &[Want]) {
        // Group blocks covering the same rows, keeping the order of first appearance.
        let mut groups: Vec<(Vec<BlockId>, bool)> = Vec::new();
        let mut open: HashMap<(u64, u64), usize> = HashMap::new();
        for want in wants {
            if self.pending.contains_key(&want.block) {
                continue;
            }
            let rows = self.source.block_rows(want.block);
            let key = (rows.start, rows.end);
            let max = self.options.max_columns_per_fetch.max(1);
            match open.get(&key).copied() {
                Some(index) if groups.get(index).is_some_and(|g| g.0.len() < max) => {
                    if let Some(group) = groups.get_mut(index) {
                        group.0.push(want.block);
                        group.1 |= want.visible;
                    }
                }
                _ => {
                    open.insert(key, groups.len());
                    groups.push((vec![want.block], want.visible));
                }
            }
        }
        for (blocks, visible) in groups {
            if self.jobs.len() >= self.options.max_jobs.max(1) {
                break;
            }
            let bytes = blocks
                .iter()
                .map(|b| self.source.estimated_bytes(*b))
                .fold(0u64, u64::saturating_add);
            let inflight = self.inflight_bytes();
            if inflight > 0 && inflight.saturating_add(bytes) > self.options.max_inflight_bytes {
                continue;
            }
            // Prefetch only into room the cache has.
            if !visible
                && self
                    .cached_bytes
                    .saturating_add(inflight)
                    .saturating_add(bytes)
                    > self.options.cache_bytes
            {
                continue;
            }
            self.spawn(blocks, bytes);
        }
    }

    fn spawn(&mut self, blocks: Vec<BlockId>, bytes: u64) {
        let id = self.next_job;
        self.next_job = self.next_job.wrapping_add(1);
        let cancel = CancelToken::new();
        for block in &blocks {
            self.pending.insert(*block, id);
        }
        self.jobs.insert(
            id,
            Job {
                blocks: blocks.clone(),
                cancel: cancel.clone(),
                bytes,
                cancelled: false,
            },
        );
        self.stats.fetches_started = self.stats.fetches_started.saturating_add(1);
        let source = Arc::clone(&self.source);
        let results = Arc::clone(&self.results);
        let wake = Arc::clone(&self.wake);
        self.spawner.spawn(Box::new(move || {
            let result = if cancel.is_cancelled() {
                Err(FetchError::cancelled())
            } else {
                catch_unwind(AssertUnwindSafe(|| source.fetch(&blocks, &cancel)))
                    .unwrap_or_else(|_| Err(FetchError::undecodable()))
            };
            // One slot per counted job: this never finds the queue full.
            let _ = results.push(Completion { id, result });
            wake();
        }));
    }

    /// Evicts least recently viewed blocks until the cache fits, never one in view; and
    /// forgets the oldest failures beyond the cap.
    fn evict(&mut self) {
        if self.cached_bytes > self.options.cache_bytes {
            let mut candidates: Vec<(u64, BlockId)> = self
                .cache
                .iter()
                .filter(|(b, s)| {
                    matches!(s.entry, Entry::Ready { .. }) && !self.visible.contains(b)
                })
                .map(|(b, s)| (s.used, *b))
                .collect();
            candidates.sort_unstable();
            for (_, block) in candidates {
                if self.cached_bytes <= self.options.cache_bytes {
                    break;
                }
                if let Some(Slot {
                    entry: Entry::Ready { bytes, .. },
                    ..
                }) = self.cache.remove(&block)
                {
                    self.cached_bytes = self.cached_bytes.saturating_sub(bytes);
                    self.stats.evictions = self.stats.evictions.saturating_add(1);
                }
            }
        }
        let failures = self
            .cache
            .values()
            .filter(|s| !matches!(s.entry, Entry::Ready { .. }))
            .count();
        if failures > self.options.max_failed_blocks {
            let mut candidates: Vec<(u64, BlockId)> = self
                .cache
                .iter()
                .filter(|(b, s)| {
                    !matches!(s.entry, Entry::Ready { .. }) && !self.visible.contains(b)
                })
                .map(|(b, s)| (s.used, *b))
                .collect();
            candidates.sort_unstable();
            let excess = failures.saturating_sub(self.options.max_failed_blocks);
            for (_, block) in candidates.into_iter().take(excess) {
                self.cache.remove(&block);
            }
        }
    }

    /// The settle timer or the earliest retry.
    fn next_timer(&self, now: f64) -> Option<f64> {
        let settle = match self.motion {
            Motion::Moving { settles_in } => Some(settles_in),
            Motion::Settled => None,
        };
        let retry = self
            .cache
            .iter()
            .filter(|(b, _)| self.visible.contains(b))
            .filter_map(|(_, s)| match s.entry {
                Entry::Retry { at, .. } => Some((at - now).max(0.0)),
                _ => None,
            })
            .reduce(f64::min);
        match (settle, retry) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
}
