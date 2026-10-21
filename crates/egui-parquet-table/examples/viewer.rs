//! A Parquet viewer: opens the file given on the command line and shows it in a
//! [`TableView`] filling the window. Fetches run on a small pool of threads.
//!
//! ```text
//! cargo run --release --example viewer -- data.parquet
//! cargo run --release --example viewer -- --demo [ROWS]
//! ```
//!
//! Without a file (or with `--demo`) it writes a demo file to the temporary directory first,
//! once: 2,000,000 rows (or `ROWS`) of 30 columns of mixed types, with nulls, long strings,
//! timestamps, dates, binary, a list, a struct, and decimals, with the page index.

// Demo code: plain arithmetic and indexing on small, known numbers.
#![allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
    TimestampMicrosecondArray, UInt16Array,
};
use eframe::egui;
use egui_parquet_table::parquet_table_core::arrow::schema::{
    DataType, Field, Fields, Schema, TimeUnit,
};
use egui_parquet_table::parquet_table_core::{
    FileSource, PageIndexStatus, ParquetLimits, ParquetSource, Spawn,
};
use egui_parquet_table::{TableView, TableViewOptions};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};

/// Rows of the demo file unless told otherwise.
const DEMO_ROWS: usize = 2_000_000;
/// Rows written at a time.
const BATCH_ROWS: usize = 65_536;
/// Threads that run fetches.
const WORKERS: usize = 4;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A fixed pool of threads running jobs from a channel. (Example code: the widget takes any
/// [`Spawn`], such as a runtime's blocking pool or web workers.)
fn pool(threads: usize) -> Arc<dyn Spawn> {
    let (sender, receiver) = mpsc::channel::<Job>();
    let receiver: Arc<Mutex<Receiver<Job>>> = Arc::new(Mutex::new(receiver));
    for _ in 0..threads {
        let receiver = Arc::clone(&receiver);
        std::thread::spawn(move || {
            loop {
                let job = match receiver.lock() {
                    Ok(receiver) => receiver.recv(),
                    Err(_) => return,
                };
                match job {
                    Ok(job) => job(),
                    Err(_) => return,
                }
            }
        });
    }
    Arc::new(move |job: Job| {
        // The workers live as long as the process: a send never fails.
        let _ = sender.send(job);
    })
}

/// The demo schema: 30 columns of mixed types.
fn demo_schema() -> Arc<Schema> {
    let point = Fields::from(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]);
    let mut fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int32, true),
        Field::new("score", DataType::Float64, true),
        Field::new("category", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("comment", DataType::Utf8, true),
        Field::new("active", DataType::Boolean, true),
        Field::new(
            "created",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("day", DataType::Date32, false),
        Field::new("checksum", DataType::Binary, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
            true,
        ),
        Field::new("point", DataType::Struct(point), true),
        Field::new("amount", DataType::Decimal128(12, 2), true),
        Field::new("level", DataType::Int8, false),
        Field::new("port", DataType::UInt16, false),
        Field::new("ratio", DataType::Float32, true),
    ];
    for i in 0..14 {
        fields.push(match i % 3 {
            0 => Field::new(format!("metric_{i:02}"), DataType::Int64, true),
            1 => Field::new(format!("metric_{i:02}"), DataType::Float64, true),
            _ => Field::new(format!("label_{i:02}"), DataType::Utf8, true),
        });
    }
    Arc::new(Schema::new(fields))
}

/// SplitMix64 of `seed`: a value per cell, the same each run.
fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Rows `start..start + rows` of the demo file.
fn demo_batch(
    schema: &Arc<Schema>,
    start: usize,
    rows: usize,
) -> Result<RecordBatch, Box<dyn Error>> {
    let range = start..start + rows;
    let r = |i: usize, column: u64| mix(i as u64 * 64 + column);
    let categories = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
    let words = ["red", "green", "blue", "north", "south", "fast", "slow"];

    let mut tags = ListBuilder::new(StringBuilder::new());
    for i in range.clone() {
        if i % 10 == 9 {
            tags.append(false);
            continue;
        }
        for j in 0..(r(i, 10) % 4) {
            tags.values()
                .append_value(words[((r(i, 11) >> (j * 8)) % words.len() as u64) as usize]);
        }
        tags.append(true);
    }
    let xs: Float64Array = range
        .clone()
        .map(|i| (r(i, 20) % 10_000) as f64 / 100.0)
        .collect();
    let ys: Float64Array = range
        .clone()
        .map(|i| (r(i, 21) % 10_000) as f64 / 100.0)
        .collect();
    let point = StructArray::try_new(
        Fields::from(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]),
        vec![Arc::new(xs) as ArrayRef, Arc::new(ys)],
        None,
    )?;
    let checksums: Vec<Option<[u8; 8]>> = range
        .clone()
        .map(|i| (i % 23 != 0).then(|| r(i, 9).to_le_bytes()))
        .collect();

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(range.clone().map(|i| i as i64).collect::<Int64Array>()),
        Arc::new(
            range
                .clone()
                .map(|i| (i % 17 != 0).then_some((r(i, 1) % 1_000_000) as i32))
                .collect::<Int32Array>(),
        ),
        Arc::new(
            range
                .clone()
                .map(|i| (i % 7 != 0).then(|| (r(i, 2) % 1_000_000) as f64 / 1_000.0))
                .collect::<Float64Array>(),
        ),
        Arc::new(
            range
                .clone()
                .map(|i| Some(categories[(r(i, 3) % categories.len() as u64) as usize]))
                .collect::<StringArray>(),
        ),
        Arc::new(
            range
                .clone()
                .map(|i| (i % 11 != 4).then(|| format!("user {}", r(i, 4) % 100_000)))
                .collect::<StringArray>(),
        ),
        Arc::new(
            range
                .clone()
                .map(|i| {
                    (i % 13 != 6).then(|| {
                        let length = if i % 50 == 0 {
                            400
                        } else {
                            (r(i, 5) % 12) as usize
                        };
                        (0..length)
                            .map(|j| words[((r(i, 5) >> (j % 8)) as usize + j) % words.len()])
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                })
                .collect::<StringArray>(),
        ),
        Arc::new(
            range
                .clone()
                .map(|i| (i % 19 != 0).then_some(r(i, 6) % 2 == 0))
                .collect::<BooleanArray>(),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                range
                    .clone()
                    .map(|i| 1_700_000_000_000_000 + i as i64 * 1_000_003)
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(Date32Array::from(
            range
                .clone()
                .map(|i| 19_000 + (i / 1_000) as i32)
                .collect::<Vec<_>>(),
        )),
        Arc::new(
            checksums
                .iter()
                .map(|bytes| bytes.as_ref().map(|b| &b[..]))
                .collect::<BinaryArray>(),
        ),
        Arc::new(tags.finish()),
        Arc::new(point),
        Arc::new(
            range
                .clone()
                .map(|i| (i % 29 != 0).then_some((r(i, 12) % 100_000_000) as i128))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(12, 2)?,
        ),
        Arc::new(
            range
                .clone()
                .map(|i| ((r(i, 13) % 200) as i16 - 100) as i8)
                .collect::<Int8Array>(),
        ),
        Arc::new(
            range
                .clone()
                .map(|i| (r(i, 14) % 65_536) as u16)
                .collect::<UInt16Array>(),
        ),
        Arc::new(
            range
                .clone()
                .map(|i| (i % 5 != 2).then(|| (r(i, 15) % 1_000) as f32 / 1_000.0))
                .collect::<Float32Array>(),
        ),
    ];
    for metric in 0..14u64 {
        let column = 16 + metric;
        let array: ArrayRef = match metric % 3 {
            0 => Arc::new(
                range
                    .clone()
                    .map(|i| (r(i, column) % 31 != 0).then_some((r(i, column) % 1_000_000) as i64))
                    .collect::<Int64Array>(),
            ),
            1 => Arc::new(
                range
                    .clone()
                    .map(|i| {
                        (r(i, column) % 37 != 0).then(|| (r(i, column) % 1_000_000) as f64 / 7.0)
                    })
                    .collect::<Float64Array>(),
            ),
            _ => Arc::new(
                range
                    .clone()
                    .map(|i| {
                        (r(i, column) % 41 != 0)
                            .then(|| format!("{}-{}", words[(r(i, column) % 7) as usize], i % 997))
                    })
                    .collect::<StringArray>(),
            ),
        };
        columns.push(array);
    }
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// Writes the demo file of `rows` rows to `path` (through a temporary name, so a cut-short run
/// leaves no file behind).
fn write_demo(path: &Path, rows: usize) -> Result<(), Box<dyn Error>> {
    let schema = demo_schema();
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_max_row_group_row_count(Some(250_000))
        .set_data_page_row_count_limit(10_000)
        .build();
    let partial = path.with_extension("partial");
    let file = std::fs::File::create(&partial)?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties))?;
    let mut start = 0;
    while start < rows {
        let n = BATCH_ROWS.min(rows - start);
        writer.write(&demo_batch(&schema, start, n)?)?;
        start += n;
    }
    writer.close()?;
    std::fs::rename(&partial, path)?;
    Ok(())
}

/// The demo file of `rows` rows in the temporary directory, written if it isn't there yet.
fn demo_file(rows: usize) -> Result<PathBuf, Box<dyn Error>> {
    let path = std::env::temp_dir().join(format!("egui-parquet-table-demo-{rows}.parquet"));
    if !path.exists() {
        eprintln!("writing {rows} rows of demo data to {}", path.display());
        write_demo(&path, rows)?;
    }
    Ok(path)
}

/// A Parquet file opened for the view.
type Opened = (Arc<ParquetSource<FileSource>>, String);

/// Opens `path` as a table source, with the page index's status for the status line.
fn open(path: &Path) -> Result<Opened, Box<dyn Error>> {
    let source = ParquetSource::open(FileSource::open(path)?, &ParquetLimits::default())?;
    let index = match source.page_index() {
        PageIndexStatus::Loaded => "loaded",
        PageIndexStatus::Absent => "absent",
        PageIndexStatus::TooLarge => "too large",
        PageIndexStatus::Invalid => "invalid",
    };
    Ok((Arc::new(source), format!("page index: {index}")))
}

struct Viewer {
    view: TableView,
}

impl Viewer {
    fn new(ctx: &egui::Context, path: &Path) -> Result<Self, Box<dyn Error>> {
        let (source, status) = open(path)?;
        let mut view = TableView::new(source, pool(WORKERS), ctx, TableViewOptions::default());
        view.set_status_text(Some(status));
        Ok(Self { view })
    }
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            self.view.show(ui);
        });
    }
}

/// The file to show: the one given, or the demo file.
fn file_from_args(args: &[String]) -> Result<PathBuf, Box<dyn Error>> {
    match args {
        [] => demo_file(DEMO_ROWS),
        [flag] if flag == "--demo" => demo_file(DEMO_ROWS),
        [flag, rows] if flag == "--demo" => demo_file(rows.replace('_', "").parse()?),
        [path] => Ok(PathBuf::from(path)),
        _ => Err("usage: viewer [FILE.parquet | --demo [ROWS]]".into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = file_from_args(&args)?;
    let title = format!(
        "{} — Parquet viewer",
        path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned()
        )
    );
    eframe::run_native(
        &title,
        eframe::NativeOptions::default(),
        Box::new(move |cc| {
            let viewer = Viewer::new(&cc.egui_ctx, &path).map_err(|error| error.to_string())?;
            Ok(Box::new(viewer))
        }),
    )?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// A small demo file opens and the viewer's view fills from the thread pool, headless.
    #[test]
    fn the_demo_opens_and_fills_from_the_pool() {
        let dir =
            std::env::temp_dir().join(format!("egui-parquet-table-viewer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("demo.parquet");
        write_demo(&path, 70_000).unwrap();

        let ctx = egui::Context::default();
        let mut viewer = Viewer::new(&ctx, &path).unwrap();
        let schema = viewer.view.source().schema().clone();
        assert_eq!(schema.fields().len(), 30);
        assert_eq!(viewer.view.source().row_count(), 70_000);

        let mut time = 0.0;
        let mut ready = false;
        for _ in 0..5_000 {
            time += 1.0 / 60.0;
            let mut output = None;
            let mut full = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1200.0, 700.0),
                    )),
                    time: Some(time),
                    ..Default::default()
                },
                |ui| output = Some(viewer.view.show(ui)),
            );
            full.textures_delta.clear();
            let output = output.unwrap();
            ready = !output.rows.is_empty()
                && !output.columns.is_empty()
                && output.rows.clone().all(|row| {
                    output.columns.iter().all(|&column| {
                        matches!(
                            viewer.view.engine().cell(row, column),
                            egui_parquet_table::parquet_table_core::Cell::Ready { .. }
                        )
                    })
                });
            if ready {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(ready, "the view filled");
        assert_eq!(viewer.view.stats().failed_blocks, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn arguments_choose_the_file() {
        assert_eq!(
            file_from_args(&["data.parquet".to_owned()]).unwrap(),
            PathBuf::from("data.parquet")
        );
        assert!(file_from_args(&["a".to_owned(), "b".to_owned(), "c".to_owned()]).is_err());
    }
}
