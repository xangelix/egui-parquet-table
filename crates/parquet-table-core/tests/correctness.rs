//! Every block of every column equals the same rows read eagerly, across codecs, page
//! versions, with and without the page index, flat and nested.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

mod common;

use std::collections::BTreeMap;

use bytes::Bytes;
use common::{Options, eager, flat_file, nested_file};
use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use parquet_table_core::{
    CancelToken, MemorySource, PageIndexStatus, ParquetLimits, ParquetSource, TableSource,
};

/// Small blocks, so every file has many of them and their sizes differ per column.
fn limits() -> ParquetLimits {
    ParquetLimits {
        target_block_bytes: 4 << 10,
        min_block_rows: 16,
        max_block_rows: 1024,
        ..ParquetLimits::default()
    }
}

fn check(bytes: &Bytes, page_index: bool, label: &str) -> usize {
    let expected = eager(bytes);
    let source = ParquetSource::open(MemorySource::new(bytes.clone()), &limits())
        .unwrap_or_else(|e| panic!("{label}: open failed: {e}"));
    assert_eq!(
        source.page_index(),
        if page_index {
            PageIndexStatus::Loaded
        } else {
            PageIndexStatus::Absent
        },
        "{label}"
    );
    assert_eq!(source.row_count(), expected.num_rows() as u64, "{label}");
    assert_eq!(source.schema(), &expected.schema(), "{label}");
    let cancel = CancelToken::new();
    let mut blocks = 0;
    for column in 0..expected.num_columns() {
        for block in source.blocks(column) {
            let rows = source.block_rows(block);
            let arrays = TableSource::fetch(&source, &[block], &cancel)
                .unwrap_or_else(|e| panic!("{label}: column {column} rows {rows:?}: {e}"));
            let want = expected
                .column(column)
                .slice(rows.start as usize, (rows.end - rows.start) as usize);
            assert_eq!(
                arrays[0].as_ref(),
                want.as_ref(),
                "{label}: column {column} rows {rows:?}"
            );
            blocks += 1;
        }
    }
    // Multi-column fetches: group the blocks of every column by the rows they cover.
    let mut by_rows: BTreeMap<(u64, u64), Vec<_>> = BTreeMap::new();
    for column in 0..expected.num_columns() {
        for block in source.blocks(column) {
            let rows = source.block_rows(block);
            by_rows
                .entry((rows.start, rows.end))
                .or_default()
                .push(block);
        }
    }
    let mut shared = 0;
    for ((start, end), group) in by_rows.iter().filter(|(_, g)| g.len() > 1) {
        // Reversed, to check that arrays come back in the order asked.
        let mut group = group.clone();
        group.reverse();
        let arrays = TableSource::fetch(&source, &group, &cancel).unwrap();
        for (block, array) in group.iter().zip(&arrays) {
            let want = expected
                .column(block.column)
                .slice(*start as usize, (end - start) as usize);
            assert_eq!(array.as_ref(), want.as_ref(), "{label}: shared fetch");
        }
        shared += 1;
    }
    assert!(shared > 0, "{label}: no blocks shared their rows");
    blocks
}

fn codecs() -> Vec<(Compression, &'static str)> {
    vec![
        (Compression::UNCOMPRESSED, "uncompressed"),
        (Compression::SNAPPY, "snappy"),
        (Compression::ZSTD(ZstdLevel::default()), "zstd"),
        (Compression::GZIP(GzipLevel::default()), "gzip"),
        (Compression::BROTLI(BrotliLevel::default()), "brotli"),
        (Compression::LZ4, "lz4"),
        (Compression::LZ4_RAW, "lz4_raw"),
    ]
}

#[test]
fn flat_files_match_an_eager_read() {
    for (compression, name) in codecs() {
        for page_index in [true, false] {
            let options = Options {
                compression,
                page_index,
                ..Options::default()
            };
            let bytes = flat_file(10_000, &options);
            let label = format!("flat {name} index={page_index}");
            let blocks = check(&bytes, page_index, &label);
            assert!(blocks > 50, "{label}: {blocks} blocks");
        }
    }
}

#[test]
fn nested_files_match_an_eager_read() {
    for (compression, name) in [
        (Compression::SNAPPY, "snappy"),
        (Compression::ZSTD(ZstdLevel::default()), "zstd"),
        (Compression::GZIP(GzipLevel::default()), "gzip"),
    ] {
        for page_index in [true, false] {
            let options = Options {
                compression,
                page_index,
                ..Options::default()
            };
            let bytes = nested_file(9_000, &options);
            check(
                &bytes,
                page_index,
                &format!("nested {name} index={page_index}"),
            );
        }
    }
}

#[test]
fn v2_pages_and_plain_encoding_match_an_eager_read() {
    for (compression, name) in [
        (Compression::ZSTD(ZstdLevel::default()), "zstd"),
        (Compression::GZIP(GzipLevel::default()), "gzip"),
        (Compression::BROTLI(BrotliLevel::default()), "brotli"),
        (Compression::SNAPPY, "snappy"),
    ] {
        for (page_index, dictionary) in [(true, true), (false, false), (true, false)] {
            let options = Options {
                compression,
                page_index,
                v2_pages: true,
                dictionary,
                ..Options::default()
            };
            let label = format!("v2 {name} index={page_index} dict={dictionary}");
            check(&flat_file(7_000, &options), page_index, &label);
            check(&nested_file(5_000, &options), page_index, &label);
        }
    }
}

#[test]
fn one_large_row_group() {
    let options = Options {
        row_group_rows: 1 << 20,
        page_rows: 1000,
        ..Options::default()
    };
    check(&flat_file(20_000, &options), true, "one row group");
}

#[test]
fn column_info_reports_statistics() {
    let bytes = flat_file(10_000, &Options::default());
    let source = ParquetSource::open(MemorySource::new(bytes), &limits()).unwrap();
    let id = source.column_info(0);
    assert_eq!(id.name, "id");
    assert_eq!(id.physical_type.as_deref(), Some("INT64"));
    assert_eq!(id.min.as_deref(), Some("0"));
    assert_eq!(id.max.as_deref(), Some("9999"));
    assert_eq!(id.null_count, Some(0));
    assert_eq!(id.codecs, vec!["SNAPPY".to_owned()]);
    assert!(id.compressed_bytes > 0 && id.uncompressed_bytes > 0);
    let value = source.column_info(1);
    assert_eq!(value.null_count, Some(1430));
    let color = source.column_info(2);
    assert_eq!(color.logical_type.as_deref(), Some("STRING"));
    assert_eq!(color.min.as_deref(), Some(""));
    assert_eq!(color.max.as_deref(), Some("red"));
    let time = source.column_info(5);
    assert!(
        time.min
            .as_deref()
            .is_some_and(|m| m.starts_with("2023-11-14T22:13:20")),
        "{time:?}"
    );
    assert_eq!(source.column_info(99), Default::default());
}
