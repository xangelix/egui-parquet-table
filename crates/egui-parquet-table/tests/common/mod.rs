//! Shared helpers: a frame driver over `egui::Context::run_ui`, the accessibility tree and the
//! text drawn, spawners (inline and held), a synthetic source, and Parquet files in memory.

#![allow(
    dead_code,
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use bytes::Bytes;
use egui::accesskit::Role;
use egui::{Event, Modifiers, PointerButton, Pos2, Rect, Vec2};
use egui_parquet_table::parquet_table_core::arrow::schema::{DataType, Field, Schema, SchemaRef};
use egui_parquet_table::parquet_table_core::{
    BlockId, CancelToken, ColumnInfo, FetchError, MemorySource, ParquetLimits, ParquetSource,
    Spawn, TableSource,
};
use egui_parquet_table::{TableView, TableViewOptions, TableViewOutput};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};

/// Drives frames of one context with a synthetic clock.
pub struct Frames {
    pub ctx: egui::Context,
    pub time: f64,
    pub size: Vec2,
}

impl Frames {
    pub fn new() -> Self {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        ctx.global_style_mut(|style| {
            style.scroll_animation = egui::style::ScrollAnimation::none();
        });
        Self {
            ctx,
            time: 0.0,
            size: egui::vec2(1000.0, 600.0),
        }
    }

    /// One frame, 1/60 s after the last, with `events`.
    pub fn frame(
        &mut self,
        view: &mut TableView,
        events: Vec<Event>,
    ) -> (TableViewOutput, egui::FullOutput) {
        self.time += 1.0 / 60.0;
        let mut output = None;
        let mut full = self.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, self.size)),
                time: Some(self.time),
                events,
                ..Default::default()
            },
            |ui| output = Some(view.show(ui)),
        );
        full.textures_delta.clear();
        (output.expect("shown"), full)
    }

    /// `n` frames without input.
    pub fn frames(
        &mut self,
        view: &mut TableView,
        n: usize,
    ) -> (TableViewOutput, egui::FullOutput) {
        let mut last = self.frame(view, vec![]);
        for _ in 1..n {
            last = self.frame(view, vec![]);
        }
        last
    }

    /// Lets `seconds` pass before the next frame.
    pub fn wait(&mut self, seconds: f64) {
        self.time += seconds;
    }

    /// Clicks at `pos` (press in one frame, release in the next); returns the second frame.
    pub fn click(
        &mut self,
        view: &mut TableView,
        pos: Pos2,
    ) -> (TableViewOutput, egui::FullOutput) {
        self.frame(view, vec![Event::PointerMoved(pos), press(pos, true)]);
        self.frame(view, vec![press(pos, false)])
    }
}

pub fn press(pos: Pos2, pressed: bool) -> Event {
    Event::PointerButton {
        pos,
        pressed,
        button: PointerButton::Primary,
        modifiers: Modifiers::NONE,
    }
}

/// One node of the accessibility tree.
#[derive(Debug, Clone)]
pub struct Node {
    pub role: Role,
    pub label: Option<String>,
    pub value: Option<String>,
    pub bounds: Option<Rect>,
}

impl Node {
    /// The label, or a label's text (egui puts it in the value).
    pub fn text(&self) -> Option<&str> {
        self.label.as_deref().or(self.value.as_deref())
    }

    pub fn center(&self) -> Pos2 {
        self.bounds.expect("bounds").center()
    }
}

/// The accessibility tree's nodes of a frame.
pub fn nodes(full: &egui::FullOutput) -> Vec<Node> {
    let update = full
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("accesskit is on");
    update
        .nodes
        .iter()
        .map(|(_, node)| Node {
            role: node.role(),
            label: node.label().map(str::to_owned),
            value: node.value().map(str::to_owned),
            bounds: node.bounds().map(|b| {
                Rect::from_min_max(
                    Pos2::new(b.x0 as f32, b.y0 as f32),
                    Pos2::new(b.x1 as f32, b.y1 as f32),
                )
            }),
        })
        .collect()
}

/// The texts of the accessibility tree.
pub fn labels(full: &egui::FullOutput) -> HashSet<String> {
    nodes(full)
        .iter()
        .filter_map(|node| node.text().map(str::to_owned))
        .collect()
}

/// The node with this role and text.
pub fn find(full: &egui::FullOutput, role: Role, text: &str) -> Option<Node> {
    nodes(full)
        .into_iter()
        .find(|node| node.role == role && node.text() == Some(text))
}

/// A text drawn, with its style.
#[derive(Debug, Clone)]
pub struct Drawn {
    pub text: String,
    pub italics: bool,
    pub color: egui::Color32,
    pub rect: Rect,
}

/// Every text drawn in a frame.
pub fn drawn(full: &egui::FullOutput) -> Vec<Drawn> {
    fn collect(shape: &egui::Shape, out: &mut Vec<Drawn>) {
        match shape {
            egui::Shape::Text(text) => {
                let format = text.galley.job.sections.first().map(|s| s.format.clone());
                out.push(Drawn {
                    text: text.galley.text().to_owned(),
                    italics: format.as_ref().is_some_and(|f| f.italics),
                    color: format.map_or(text.fallback_color, |f| f.color),
                    rect: text.galley.rect.translate(text.pos.to_vec2()),
                });
            }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| collect(shape, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for clipped in &full.shapes {
        collect(&clipped.shape, &mut out);
    }
    out
}

/// The text put on the clipboard in a frame.
pub fn clipboard(full: &egui::FullOutput) -> Option<String> {
    full.platform_output
        .commands
        .iter()
        .find_map(|command| match command {
            egui::OutputCommand::CopyText(text) => Some(text.clone()),
            _ => None,
        })
}

/// Runs each job at once, inside `spawn`.
pub struct Inline;

impl Spawn for Inline {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) {
        job();
    }
}

/// Holds jobs until the test runs them.
#[derive(Default)]
pub struct Held(Mutex<Vec<Box<dyn FnOnce() + Send + 'static>>>);

impl Held {
    pub fn take(&self) -> Vec<Box<dyn FnOnce() + Send + 'static>> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }

    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

impl Spawn for Held {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) {
        self.0.lock().unwrap().push(job);
    }
}

/// A table of `columns` Int64 columns `c0`, `c1`, … whose cell (row, column) is
/// `row * 1000 + column`, in blocks of `block_rows`. Records every fetch and the thread it ran
/// on; fails every fetch while `failing` is set.
pub struct Synthetic {
    schema: SchemaRef,
    rows: u64,
    block_rows: u64,
    pub fetches: Mutex<Vec<(Vec<BlockId>, ThreadId)>>,
    pub failing: AtomicBool,
}

impl Synthetic {
    pub fn new(columns: usize, rows: u64, block_rows: u64) -> Arc<Self> {
        let fields: Vec<Field> = (0..columns)
            .map(|c| Field::new(format!("c{c}"), DataType::Int64, false))
            .collect();
        Arc::new(Self {
            schema: Arc::new(Schema::new(fields)),
            rows,
            block_rows,
            fetches: Mutex::new(Vec::new()),
            failing: AtomicBool::new(false),
        })
    }

    /// Every block fetched so far.
    pub fn fetched(&self) -> Vec<BlockId> {
        self.fetches
            .lock()
            .unwrap()
            .iter()
            .flat_map(|(blocks, _)| blocks.iter().copied())
            .collect()
    }

    pub fn fetch_calls(&self) -> usize {
        self.fetches.lock().unwrap().len()
    }

    pub fn threads(&self) -> Vec<ThreadId> {
        self.fetches
            .lock()
            .unwrap()
            .iter()
            .map(|(_, thread)| *thread)
            .collect()
    }
}

impl TableSource for Synthetic {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn row_count(&self) -> u64 {
        self.rows
    }

    fn block_of(&self, column: usize, row: u64) -> Option<BlockId> {
        (column < self.schema.fields().len() && row < self.rows).then(|| BlockId {
            column,
            group: 0,
            index: u32::try_from(row / self.block_rows).unwrap(),
        })
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
        self.fetches
            .lock()
            .unwrap()
            .push((blocks.to_vec(), std::thread::current().id()));
        if cancel.is_cancelled() {
            return Err(FetchError::cancelled());
        }
        if self.failing.load(Ordering::SeqCst) {
            return Err(FetchError::permanent("checksum mismatch"));
        }
        Ok(blocks
            .iter()
            .map(|block| {
                let rows = self.block_rows(*block);
                Arc::new(
                    rows.map(|row| row as i64 * 1000 + block.column as i64)
                        .collect::<Int64Array>(),
                ) as ArrayRef
            })
            .collect())
    }

    fn column_info(&self, column: usize) -> ColumnInfo {
        ColumnInfo {
            name: format!("c{column}"),
            data_type: "Int64".to_owned(),
            ..ColumnInfo::default()
        }
    }
}

/// What row `row` of [`people`] holds in `name`: a null every fifth row from 3, the string
/// `"null"` every fifth row from 4.
pub fn name_of(row: usize) -> Option<String> {
    match row % 5 {
        3 => None,
        4 => Some("null".to_owned()),
        _ => Some(format!("name {row}")),
    }
}

/// A Parquet file of `rows` rows: `id` (Int64, the row), `name` (see [`name_of`]), `value`
/// (Float64, half the row, null every seventh), `flag` (Boolean), and `note` (a 300-character
/// string every eleventh row). Row groups of 4,000 rows, pages of 500, with the page index.
pub fn people(rows: usize) -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("value", DataType::Float64, true),
        Field::new("flag", DataType::Boolean, false),
        Field::new("note", DataType::Utf8, false),
    ]));
    let properties = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_max_row_group_row_count(Some(4_000))
        .set_data_page_row_count_limit(500)
        .set_write_batch_size(500)
        .build();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, Arc::clone(&schema), Some(properties)).unwrap();
    for start in (0..rows).step_by(1000) {
        let range = start..(start + 1000).min(rows);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(range.clone().map(|r| r as i64).collect::<Int64Array>()),
            Arc::new(range.clone().map(name_of).collect::<StringArray>()),
            Arc::new(
                range
                    .clone()
                    .map(|r| (r % 7 != 6).then_some(r as f64 / 2.0))
                    .collect::<Float64Array>(),
            ),
            Arc::new(
                range
                    .clone()
                    .map(|r| Some(r % 2 == 0))
                    .collect::<BooleanArray>(),
            ),
            Arc::new(
                range
                    .map(|r| {
                        Some(if r % 11 == 0 {
                            format!("long {r} {}", "x".repeat(300))
                        } else {
                            format!("note {r}")
                        })
                    })
                    .collect::<StringArray>(),
            ),
        ];
        writer
            .write(&RecordBatch::try_new(Arc::clone(&schema), columns).unwrap())
            .unwrap();
    }
    writer.close().unwrap();
    Bytes::from(out)
}

/// [`people`] opened from memory, in blocks of a few hundred rows.
pub fn people_source(rows: usize) -> Arc<ParquetSource<MemorySource>> {
    let limits = ParquetLimits {
        target_block_bytes: 4 << 10,
        ..ParquetLimits::default()
    };
    Arc::new(ParquetSource::open(MemorySource::new(people(rows)), &limits).unwrap())
}

/// A view of `source` drawn by `frames`' context.
pub fn view(
    frames: &Frames,
    source: Arc<dyn TableSource>,
    spawner: Arc<dyn Spawn>,
    options: TableViewOptions,
) -> TableView {
    TableView::new(source, spawner, &frames.ctx, options)
}

/// Frames until `ready` holds (at most `max` frames, each `step` seconds of synthetic time
/// after the last, plus a real pause of at most a millisecond for worker threads).
pub fn until(
    frames: &mut Frames,
    view: &mut TableView,
    max: usize,
    pause: bool,
    mut ready: impl FnMut(&TableView, &TableViewOutput, &egui::FullOutput) -> bool,
) -> (TableViewOutput, egui::FullOutput) {
    for _ in 0..max {
        let (output, full) = frames.frame(view, vec![]);
        if ready(view, &output, &full) {
            return (output, full);
        }
        if pause {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    panic!("not ready after {max} frames");
}

/// Whether every cell of the rows and columns drawn is loaded.
pub fn all_loaded(view: &TableView, output: &TableViewOutput) -> bool {
    !output.rows.is_empty()
        && !output.columns.is_empty()
        && output.rows.clone().all(|row| {
            output.columns.iter().all(|&column| {
                matches!(
                    view.engine().cell(row, column),
                    egui_parquet_table::parquet_table_core::Cell::Ready { .. }
                )
            })
        })
}

/// Counts of the drawn texts.
pub fn text_counts(full: &egui::FullOutput) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for text in drawn(full) {
        *counts.entry(text.text).or_insert(0) += 1;
    }
    counts
}

/// A table over arrays in memory, each column one block.
pub struct Arrays {
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    rows: u64,
}

impl Arrays {
    pub fn new(columns: Vec<(&str, ArrayRef)>) -> Arc<Self> {
        let rows = columns.first().map_or(0, |(_, array)| array.len() as u64);
        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect();
        Arc::new(Self {
            schema: Arc::new(Schema::new(fields)),
            columns: columns.into_iter().map(|(_, array)| array).collect(),
            rows,
        })
    }
}

impl TableSource for Arrays {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn row_count(&self) -> u64 {
        self.rows
    }

    fn block_of(&self, column: usize, row: u64) -> Option<BlockId> {
        (column < self.columns.len() && row < self.rows).then_some(BlockId {
            column,
            group: 0,
            index: 0,
        })
    }

    fn block_rows(&self, _block: BlockId) -> Range<u64> {
        0..self.rows
    }

    fn estimated_bytes(&self, _block: BlockId) -> u64 {
        self.rows * 8
    }

    fn fetch(
        &self,
        blocks: &[BlockId],
        _cancel: &CancelToken,
    ) -> Result<Vec<ArrayRef>, FetchError> {
        Ok(blocks
            .iter()
            .map(|block| Arc::clone(&self.columns[block.column]))
            .collect())
    }

    fn column_info(&self, column: usize) -> ColumnInfo {
        ColumnInfo {
            name: self.schema.field(column).name().clone(),
            data_type: self.schema.field(column).data_type().to_string(),
            ..ColumnInfo::default()
        }
    }
}
