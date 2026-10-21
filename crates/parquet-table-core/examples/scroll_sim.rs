//! Writes a large Parquet file to the temporary directory, then simulates a user who opens
//! it, flings the view from the first row to the last, and stops there. Prints what was
//! read and fetched at each stage.
//!
//! ```text
//! cargo run --release --example scroll_sim [rows] [columns]
//! ```

// Demo code: plain arithmetic on small, known numbers.
#![allow(clippy::arithmetic_side_effects)]

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet_table_core::{
    Cell, CountingSource, FileSource, LazyOptions, LazyTable, ParquetLimits, ParquetSource, Stats,
    TableSource, Viewport,
};

const BATCH: usize = 100_000;
const VIEW_ROWS: u64 = 40;
const VIEW_COLUMNS: usize = 8;

fn schema(columns: usize) -> Arc<Schema> {
    let fields: Vec<Field> = (0..columns)
        .map(|c| match c % 4 {
            0 => Field::new(format!("id{c}"), DataType::Int64, false),
            1 => Field::new(format!("x{c}"), DataType::Float64, true),
            2 => Field::new(format!("tag{c}"), DataType::Utf8, false),
            _ => Field::new(format!("note{c}"), DataType::Utf8, true),
        })
        .collect();
    Arc::new(Schema::new(fields))
}

fn batch(schema: &Arc<Schema>, start: usize, rows: usize) -> Result<RecordBatch, Box<dyn Error>> {
    let tags = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let columns: Vec<ArrayRef> = (0..schema.fields().len())
        .map(|c| -> ArrayRef {
            let rows = start..start + rows;
            match c % 4 {
                0 => Arc::new(rows.map(|r| (r * 7 + c) as i64).collect::<Int64Array>()),
                1 => Arc::new(
                    rows.map(|r| {
                        (r % 13 != 0).then(|| ((r * 2_654_435_761) % 1_000_003) as f64 / 7.0)
                    })
                    .collect::<Float64Array>(),
                ),
                2 => Arc::new(
                    rows.map(|r| tags.get((r + c) % tags.len()).copied())
                        .collect::<StringArray>(),
                ),
                _ => Arc::new(
                    rows.map(|r| (r % 5 != 0).then(|| format!("note {r} of column {c}")))
                        .collect::<StringArray>(),
                ),
            }
        })
        .collect();
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

fn write(path: &std::path::Path, rows: usize, columns: usize) -> Result<(), Box<dyn Error>> {
    let schema = schema(columns);
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_max_row_group_row_count(Some(1 << 20))
        .build();
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties))?;
    let mut start = 0;
    while start < rows {
        let n = BATCH.min(rows - start);
        writer.write(&batch(&schema, start, n)?)?;
        start += n;
    }
    writer.close()?;
    Ok(())
}

fn all_ready(table: &LazyTable, view: &Viewport) -> bool {
    view.rows.clone().all(|row| {
        view.columns
            .clone()
            .all(|column| matches!(table.cell(row, column), Cell::Ready { .. }))
    })
}

fn line(stage: &str, read: u64, stats: &Stats, previous: &Stats) {
    println!(
        "{stage:<28} read {:>12} bytes | fetches started {:>4} completed {:>4} cancelled {:>3} | cache {:>10} bytes in {:>4} blocks",
        read,
        stats.fetches_started - previous.fetches_started,
        stats.fetches_completed - previous.fetches_completed,
        stats.fetches_cancelled - previous.fetches_cancelled,
        stats.cached_bytes,
        stats.cached_blocks,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let rows: usize = args
        .next()
        .map(|a| a.parse())
        .transpose()?
        .unwrap_or(5_000_000);
    let columns: usize = args.next().map(|a| a.parse()).transpose()?.unwrap_or(20);
    let path = std::env::temp_dir().join(format!(
        "parquet-table-core-scroll-sim-{}.parquet",
        std::process::id()
    ));

    let started = Instant::now();
    write(&path, rows, columns)?;
    let size = std::fs::metadata(&path)?.len();
    println!(
        "wrote {rows} rows x {columns} columns: {size} bytes in {:.1} s",
        started.elapsed().as_secs_f64()
    );

    let result = simulate(&path, size);
    std::fs::remove_file(&path)?;
    result
}

fn simulate(path: &std::path::Path, size: u64) -> Result<(), Box<dyn Error>> {
    let counting = Arc::new(CountingSource::new(FileSource::open(path)?));
    let source = Arc::new(ParquetSource::open(
        Arc::clone(&counting),
        &ParquetLimits::default(),
    )?);
    println!(
        "opened: page index {:?}, {} row groups; read {} of {size} bytes",
        source.page_index(),
        source.row_groups(),
        counting.bytes()
    );
    let total_rows = source.row_count();
    let spawner = Arc::new(|job: Box<dyn FnOnce() + Send + 'static>| {
        std::thread::spawn(job);
    });
    let mut table = LazyTable::new(source, spawner, Arc::new(|| {}), LazyOptions::default());

    let clock = Instant::now();
    let now = || clock.elapsed().as_secs_f64();
    let wait = |table: &mut LazyTable, view: &Viewport| -> Duration {
        let begin = Instant::now();
        while !all_ready(table, view) {
            std::thread::sleep(Duration::from_millis(1));
            table.update(view, now());
        }
        // Let the prefetch ring land too.
        while table.stats().inflight_jobs > 0 {
            std::thread::sleep(Duration::from_millis(1));
            table.update(view, now());
        }
        begin.elapsed()
    };

    // The first view.
    let top = Viewport {
        rows: 0..VIEW_ROWS,
        columns: 0..VIEW_COLUMNS,
    };
    let before = counting.bytes();
    let stats0 = table.stats();
    table.update(&top, now());
    let took = wait(&mut table, &top);
    let stats1 = table.stats();
    line(
        &format!("first view ({} ms)", took.as_millis()),
        counting.bytes() - before,
        &stats1,
        &stats0,
    );

    // A fling to the last row in 1.5 s at 60 frames per second (synthetic frames, real
    // clock for the policy: the frames are paced).
    let before = counting.bytes();
    let frames = 90u64;
    let last_top = total_rows.saturating_sub(VIEW_ROWS);
    for frame in 1..=frames {
        let top_row = last_top * frame / frames;
        let view = Viewport {
            rows: top_row..top_row + VIEW_ROWS,
            columns: 0..VIEW_COLUMNS,
        };
        table.update(&view, now());
        std::thread::sleep(Duration::from_millis(16));
    }
    let stats2 = table.stats();
    line(
        "fling to the bottom",
        counting.bytes() - before,
        &stats2,
        &stats1,
    );

    // Resting at the bottom: the view settles, then fetches.
    let bottom = Viewport {
        rows: last_top..last_top + VIEW_ROWS,
        columns: 0..VIEW_COLUMNS,
    };
    let before = counting.bytes();
    table.update(&bottom, now());
    let took = wait(&mut table, &bottom);
    let stats3 = table.stats();
    line(
        &format!("settle at the bottom ({} ms)", took.as_millis()),
        counting.bytes() - before,
        &stats3,
        &stats2,
    );
    println!(
        "total read {} of {size} bytes ({:.3}%) in {} source reads",
        counting.bytes(),
        counting.bytes() as f64 * 100.0 / size as f64,
        counting.calls()
    );
    Ok(())
}
