//! What opening and fetching read: only the footer (and the offset index) to open, only the
//! requested columns' chunks to fetch, and with the offset index only the pages holding the
//! requested rows.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

mod common;

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use common::{Options, flat_file, wide_file};
use parquet_table_core::{
    BlockId, CancelToken, CountingSource, MemorySource, PageIndexStatus, ParquetLimits,
    ParquetSource, TableSource,
};

type Counted = Arc<CountingSource<MemorySource>>;

fn open(bytes: &Bytes, limits: &ParquetLimits) -> (Counted, ParquetSource<Counted>) {
    let counting = Arc::new(CountingSource::recording(MemorySource::new(bytes.clone())));
    let source = ParquetSource::open(Arc::clone(&counting), limits).unwrap();
    (counting, source)
}

/// The union of every column chunk's offset-index range.
fn index_span(source: &ParquetSource<Counted>) -> Option<Range<u64>> {
    source
        .metadata()
        .row_groups()
        .iter()
        .flat_map(|g| g.columns())
        .filter_map(|c| c.offset_index_range())
        .reduce(|a, b| a.start.min(b.start)..a.end.max(b.end))
}

#[test]
fn opening_reads_only_the_footer_and_the_offset_index() {
    let bytes = flat_file(50_000, &Options::default());
    let len = bytes.len() as u64;

    let (counting, source) = open(&bytes, &ParquetLimits::default());
    assert_eq!(source.page_index(), PageIndexStatus::Loaded);
    let span = index_span(&source).unwrap();
    let footer = source.footer_bytes();
    assert_eq!(counting.bytes(), 8 + footer + (span.end - span.start));
    for range in counting.ranges() {
        assert!(range.start >= span.start && range.end <= len, "{range:?}");
    }
    eprintln!(
        "open with offset index: {} bytes of {} ({} footer, {} index)",
        counting.bytes(),
        len,
        footer,
        span.end - span.start
    );

    // Over the cap: the index is not read.
    let limits = ParquetLimits {
        max_page_index_bytes: 16,
        ..ParquetLimits::default()
    };
    let (counting, source) = open(&bytes, &limits);
    assert_eq!(source.page_index(), PageIndexStatus::TooLarge);
    assert_eq!(counting.bytes(), 8 + source.footer_bytes());
    assert_eq!(counting.calls(), 2);

    // A file without one.
    let bytes = flat_file(
        20_000,
        &Options {
            page_index: false,
            ..Options::default()
        },
    );
    let (counting, source) = open(&bytes, &ParquetLimits::default());
    assert_eq!(source.page_index(), PageIndexStatus::Absent);
    assert_eq!(counting.bytes(), 8 + source.footer_bytes());
}

/// The bytes of a leaf chunk's dictionary page and of the data pages that hold a block's
/// rows.
fn page_bytes(source: &ParquetSource<Counted>, block: BlockId, leaf: usize) -> (u64, u64) {
    let group = block.group as usize;
    let group_start = source.row_group_rows(group).start;
    let rows = source.block_rows(block);
    let (first, last) = (rows.start - group_start, rows.end - group_start);
    let chunk = source.column_chunk_range(group, leaf).unwrap();
    let index = source.metadata().page_index().unwrap();
    let locations = index.page_locations(group, leaf).unwrap();
    let group_rows = source.row_group_rows(group).end - group_start;
    let dictionary = (locations[0].offset as u64).saturating_sub(chunk.start);
    let mut total = 0;
    for (i, page) in locations.iter().enumerate() {
        let page_first = page.first_row_index as u64;
        let page_end = locations
            .get(i + 1)
            .map_or(group_rows, |next| next.first_row_index as u64);
        if page_first < last && page_end > first {
            total += page.compressed_page_size as u64;
        }
    }
    (dictionary, total)
}

#[test]
fn a_block_reads_only_the_pages_holding_its_rows() {
    let options = Options {
        row_group_rows: 1 << 20,
        page_rows: 2_000,
        ..Options::default()
    };
    let bytes = flat_file(200_000, &options);
    let len = bytes.len() as u64;
    let limits = ParquetLimits {
        target_block_bytes: 64 << 10,
        ..ParquetLimits::default()
    };
    let (counting, source) = open(&bytes, &limits);
    assert_eq!(source.page_index(), PageIndexStatus::Loaded);
    let cancel = CancelToken::new();
    for (column, name) in [(0, "id"), (2, "color"), (3, "text")] {
        let leaf = source.leaf_columns(column).start;
        let block = source.block_of(column, 123_456).unwrap();
        let rows = source.block_rows(block);
        counting.reset();
        let arrays = TableSource::fetch(&source, &[block], &cancel).unwrap();
        assert_eq!(arrays[0].len() as u64, rows.end - rows.start);
        let (dictionary, pages) = page_bytes(&source, block, leaf);
        let chunk = source.column_chunk_range(0, leaf).unwrap();
        for range in counting.ranges() {
            assert!(range.start >= chunk.start && range.end <= chunk.end);
        }
        assert_eq!(
            counting.bytes(),
            dictionary + pages,
            "{name}: read exactly the dictionary and the pages holding the rows"
        );
        assert!(pages * 20 < chunk.end - chunk.start, "{name}");
        let (first_bytes, first_reads) = (counting.bytes(), counting.calls());
        // The next block reuses the dictionary page.
        let next = source.block_of(column, rows.end).unwrap();
        counting.reset();
        TableSource::fetch(&source, &[next], &cancel).unwrap();
        let (_, next_pages) = page_bytes(&source, next, leaf);
        assert_eq!(counting.bytes(), next_pages, "{name}: dictionary cached");
        eprintln!(
            "{name}: block of {} rows read {first_bytes} bytes in {first_reads} reads \
             ({dictionary} dictionary + {pages} pages), the next block {} bytes; \
             chunk {} bytes; file {len} bytes",
            rows.end - rows.start,
            counting.bytes(),
            chunk.end - chunk.start
        );
    }
}

#[test]
fn without_an_offset_index_reads_stay_inside_the_chunk() {
    let options = Options {
        row_group_rows: 1 << 20,
        page_rows: 2_000,
        page_index: false,
        ..Options::default()
    };
    let bytes = flat_file(200_000, &options);
    let limits = ParquetLimits {
        target_block_bytes: 64 << 10,
        ..ParquetLimits::default()
    };
    let (counting, source) = open(&bytes, &limits);
    assert_eq!(source.page_index(), PageIndexStatus::Absent);
    let cancel = CancelToken::new();
    let chunk = source.column_chunk_range(0, 0).unwrap();
    let blocks: Vec<BlockId> = source.blocks(0).collect();
    for (label, block) in [("first", blocks[0]), ("last", *blocks.last().unwrap())] {
        counting.reset();
        TableSource::fetch(&source, &[block], &cancel).unwrap();
        for range in counting.ranges() {
            assert!(range.start >= chunk.start && range.end <= chunk.end);
        }
        assert!(counting.bytes() <= chunk.end - chunk.start + 4096);
        eprintln!(
            "no index, {label} block of id: read {} bytes in {} reads; chunk {} bytes",
            counting.bytes(),
            counting.calls(),
            chunk.end - chunk.start
        );
    }
}

#[test]
fn a_wide_file_reads_only_the_requested_columns() {
    let bytes = wide_file(1_000, 5_000, &Options::default());
    let len = bytes.len() as u64;
    let (counting, source) = open(&bytes, &ParquetLimits::default());
    assert_eq!(source.schema().fields().len(), 1_000);
    let opened = counting.bytes();
    counting.reset();
    let columns = [3usize, 500, 998];
    let blocks: Vec<BlockId> = columns
        .iter()
        .map(|&c| source.block_of(c, 2_500).unwrap())
        .collect();
    assert!(
        blocks
            .iter()
            .all(|b| source.block_rows(*b) == source.block_rows(blocks[0]))
    );
    let arrays = TableSource::fetch(&source, &blocks, &CancelToken::new()).unwrap();
    assert_eq!(arrays.len(), 3);
    let chunks: Vec<Range<u64>> = columns
        .iter()
        .map(|&c| source.column_chunk_range(0, c).unwrap())
        .collect();
    for range in counting.ranges() {
        assert!(
            chunks
                .iter()
                .any(|c| range.start >= c.start && range.end <= c.end),
            "{range:?} outside the requested chunks"
        );
    }
    let total: u64 = chunks.iter().map(|c| c.end - c.start).sum();
    assert!(counting.bytes() <= total);
    eprintln!(
        "1000 columns: open read {opened} bytes; 3 columns read {} bytes in {} reads; file {len} bytes",
        counting.bytes(),
        counting.calls()
    );
}

/// Through the format-independent trait, a Parquet source reports what it has read and the
/// file's length, so a view can show how little of a file it has read.
#[test]
fn the_table_trait_reports_bytes_read_and_the_files_length() {
    let bytes = flat_file(50_000, &Options::default());
    let (counting, source) = open(&bytes, &ParquetLimits::default());
    let table: &dyn TableSource = &source;
    assert_eq!(table.byte_len(), Some(bytes.len() as u64));
    assert_eq!(table.bytes_read(), Some(counting.bytes()));
    let block = table.block_of(0, 0).unwrap();
    let before = counting.bytes();
    table.fetch(&[block], &CancelToken::new()).unwrap();
    assert!(counting.bytes() > before);
    assert_eq!(table.bytes_read(), Some(counting.bytes()));
}
