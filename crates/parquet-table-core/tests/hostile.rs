//! Hostile files: truncated, lying about their sizes, corrupted at random, and footers built
//! to exhaust the decoder. Every case ends in an error; none panics, aborts, or allocates
//! without bound.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

mod common;

use std::sync::{Arc, Once};

use bytes::Bytes;
use common::{Options, Rng, flat_file, nested_file};
use parquet::basic::{Compression, GzipLevel};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader, ParquetMetaDataWriter};
use parquet_table_core::{
    CancelToken, FetchErrorKind, MemorySource, OpenError, ParquetLimits, ParquetSource, TableSource,
};

/// Keeps the default panic message for the tests' own panics but not for panics the
/// decoder raises (and the engine catches) on corrupted input.
fn quiet_dependency_panics() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let from_dependency = info
                .location()
                .is_some_and(|l| l.file().contains(".cargo/registry"));
            if !from_dependency {
                default(info);
            }
        }));
    });
}

fn open(bytes: &Bytes) -> Result<ParquetSource<MemorySource>, OpenError> {
    ParquetSource::open(MemorySource::new(bytes.clone()), &ParquetLimits::default())
}

/// Opens and fetches every block of every column; returns (opened, blocks, failed).
fn exercise(bytes: &Bytes) -> (bool, usize, usize) {
    let Ok(source) = open(bytes) else {
        return (false, 0, 0);
    };
    let cancel = CancelToken::new();
    let (mut blocks, mut failed) = (0, 0);
    for column in 0..source.schema().fields().len() {
        for block in source.blocks(column).take(64) {
            blocks += 1;
            match TableSource::fetch(&source, &[block], &cancel) {
                Ok(arrays) => {
                    let rows = source.block_rows(block);
                    assert_eq!(arrays[0].len() as u64, rows.end - rows.start);
                }
                Err(e) => {
                    failed += 1;
                    assert_ne!(e.kind(), FetchErrorKind::Cancelled);
                }
            }
        }
        // Statistics come from the (possibly corrupted) footer too.
        let _ = source.column_info(column);
    }
    (true, blocks, failed)
}

fn small_file(compression: Compression, page_index: bool) -> Bytes {
    let options = Options {
        compression,
        page_index,
        row_group_rows: 400,
        page_rows: 100,
        ..Options::default()
    };
    flat_file(1_200, &options)
}

#[test]
fn truncated_files_are_refused() {
    let bytes = small_file(Compression::SNAPPY, true);
    for len in 0..bytes.len() {
        let truncated = bytes.slice(..len);
        assert!(open(&truncated).is_err(), "truncated to {len}");
    }
    // The tail kept but the start cut: chunks point at the wrong bytes.
    for cut in [1usize, 4, 100, 1000] {
        let cut_bytes = bytes.slice(cut..);
        let (opened, blocks, failed) = exercise(&cut_bytes);
        assert!(
            !opened || failed == blocks,
            "cut {cut}: {failed} of {blocks} failed"
        );
    }
}

#[test]
fn footer_lengths_are_checked() {
    let bytes = small_file(Compression::SNAPPY, true);
    let len = bytes.len();
    let footer = u32::from_le_bytes(bytes[len - 8..len - 4].try_into().unwrap());

    let mut lying = bytes.to_vec();
    lying[len - 8..len - 4].copy_from_slice(&(len as u32).to_le_bytes());
    assert_eq!(
        open(&Bytes::from(lying)).err(),
        Some(OpenError::Malformed("the footer length exceeds the file"))
    );
    let mut lying = bytes.to_vec();
    lying[len - 8..len - 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        open(&Bytes::from(lying)),
        Err(OpenError::Limit(_))
    ));

    let limits = ParquetLimits {
        max_footer_bytes: u64::from(footer) - 1,
        ..ParquetLimits::default()
    };
    let refused = ParquetSource::open(MemorySource::new(bytes.clone()), &limits).err();
    assert_eq!(
        refused,
        Some(OpenError::Limit("the footer exceeds the size limit"))
    );

    let mut encrypted = bytes.to_vec();
    encrypted[len - 4..].copy_from_slice(b"PARE");
    assert_eq!(
        open(&Bytes::from(encrypted)).err(),
        Some(OpenError::Encrypted)
    );
    assert_eq!(
        open(&Bytes::from_static(b"PAR1PAR1")).err(),
        Some(OpenError::NotParquet)
    );
}

#[test]
fn random_byte_flips_never_panic() {
    quiet_dependency_panics();
    let mut rng = Rng(0x5eed);
    let mut summary = (0usize, 0usize, 0usize, 0usize);
    let files = [
        small_file(Compression::SNAPPY, true),
        small_file(Compression::GZIP(GzipLevel::default()), true),
        small_file(Compression::LZ4, false),
        nested_file(
            800,
            &Options {
                row_group_rows: 300,
                page_rows: 100,
                ..Options::default()
            },
        ),
    ];
    for bytes in &files {
        let len = bytes.len() as u64;
        let footer = u64::from(u32::from_le_bytes(
            bytes[bytes.len() - 8..bytes.len() - 4].try_into().unwrap(),
        ));
        let footer_start = len - 8 - footer;
        // `HOSTILE_ROUNDS` runs a longer campaign.
        let rounds: u64 = std::env::var("HOSTILE_ROUNDS")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(100);
        for round in 0..rounds {
            let mut mutated = bytes.to_vec();
            for _ in 0..1 + rng.below(4) {
                // Half the flips land in the footer, where they matter most.
                let at = if round % 2 == 0 {
                    footer_start + rng.below(footer)
                } else {
                    rng.below(len)
                };
                mutated[at as usize] ^= 1 << rng.below(8);
            }
            let (opened, blocks, failed) = exercise(&Bytes::from(mutated));
            summary.0 += 1;
            summary.1 += usize::from(opened);
            summary.2 += blocks;
            summary.3 += failed;
        }
    }
    eprintln!(
        "{} mutations: {} opened, {} block fetches, {} failed",
        summary.0, summary.1, summary.2, summary.3
    );
}

/// Replaces the footer of `bytes` with `metadata` encoded by the crate's writer.
fn with_metadata(bytes: &Bytes, metadata: &ParquetMetaData) -> Bytes {
    let len = bytes.len();
    let footer = u32::from_le_bytes(bytes[len - 8..len - 4].try_into().unwrap()) as usize;
    let mut out = bytes[..len - 8 - footer].to_vec();
    ParquetMetaDataWriter::new(&mut out, metadata)
        .finish()
        .unwrap();
    Bytes::from(out)
}

fn metadata(bytes: &Bytes) -> ParquetMetaData {
    ParquetMetaDataReader::new()
        .parse_and_finish(bytes)
        .unwrap()
}

#[test]
fn huge_row_counts_are_bounded() {
    let bytes = small_file(Compression::SNAPPY, true);
    let original = metadata(&bytes);

    // A row group claiming 2^40 rows (and the file, consistently).
    let mut groups = original.row_groups().to_vec();
    groups[0] = groups[0]
        .clone()
        .into_builder()
        .set_num_rows(1 << 40)
        .build()
        .unwrap();
    let huge = original
        .clone()
        .into_builder()
        .set_row_groups(groups)
        .build();
    let hostile = with_metadata(&bytes, &huge);
    let source = open(&hostile).unwrap();
    assert!(source.row_count() > 1 << 40);
    for row in [0u64, 50_000, 1 << 39] {
        let block = source.block_of(0, row).unwrap();
        let result = TableSource::fetch(&source, &[block], &CancelToken::new());
        assert_eq!(
            result.unwrap_err().kind(),
            FetchErrorKind::Permanent,
            "row {row}"
        );
    }

    // A negative count.
    let mut groups = original.row_groups().to_vec();
    groups[1] = groups[1]
        .clone()
        .into_builder()
        .set_num_rows(-5)
        .build()
        .unwrap();
    let negative = original
        .clone()
        .into_builder()
        .set_row_groups(groups)
        .build();
    assert!(matches!(
        open(&with_metadata(&bytes, &negative)),
        Err(OpenError::Malformed(_))
    ));

    // The file's count disagreeing with its row groups.
    let footer_bytes = {
        let mut out = Vec::new();
        ParquetMetaDataWriter::new(&mut out, &original)
            .finish()
            .unwrap();
        out
    };
    let total = original.file_metadata().num_rows();
    let mut patched = footer_bytes.clone();
    let needle = [0x16u8, ((total << 1) & 0x7f | 0x80) as u8];
    let at = patched
        .windows(2)
        .position(|w| w == needle)
        .expect("the file's num_rows field");
    patched[at + 1] ^= 0x02;
    let len = bytes.len();
    let footer = u32::from_le_bytes(bytes[len - 8..len - 4].try_into().unwrap()) as usize;
    let mut out = bytes[..len - 8 - footer].to_vec();
    out.extend_from_slice(&patched);
    assert_eq!(
        open(&Bytes::from(out)).err(),
        Some(OpenError::Malformed(
            "the row count disagrees with the row groups"
        ))
    );
}

#[test]
fn column_chunks_outside_the_file_are_refused() {
    let bytes = small_file(Compression::SNAPPY, false);
    let original = metadata(&bytes);
    for (offset, size) in [(i64::MAX - 10, 100), (10, 1 << 40), (-1, 10), (0, 10)] {
        let mut groups = original.row_groups().to_vec();
        let mut columns = groups[0].columns().to_vec();
        columns[2] = columns[2]
            .clone()
            .into_builder()
            .set_dictionary_page_offset(None)
            .set_data_page_offset(offset)
            .set_total_compressed_size(size)
            .build()
            .unwrap();
        groups[0] = groups[0]
            .clone()
            .into_builder()
            .set_column_metadata(columns)
            .build()
            .unwrap();
        let hostile = original
            .clone()
            .into_builder()
            .set_row_groups(groups)
            .build();
        assert_eq!(
            open(&with_metadata(&bytes, &hostile)).err(),
            Some(OpenError::Malformed("a column chunk lies outside the data")),
            "offset {offset} size {size}"
        );
    }
}

/// A compact-protocol writer for hand-built footers.
#[derive(Default)]
struct Thrift(Vec<u8>);

impl Thrift {
    fn field(&mut self, delta: u8, ty: u8) -> &mut Self {
        self.0.push(delta << 4 | ty);
        self
    }
    fn varint(&mut self, mut v: u64) -> &mut Self {
        while v >= 0x80 {
            self.0.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
        self
    }
    fn i32(&mut self, v: i32) -> &mut Self {
        self.varint(((v << 1) ^ (v >> 31)) as u32 as u64)
    }
    fn binary(&mut self, b: &[u8]) -> &mut Self {
        self.varint(b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }
    fn list(&mut self, element: u8, size: u64) -> &mut Self {
        if size < 15 {
            self.0.push((size as u8) << 4 | element);
        } else {
            self.0.push(0xF0 | element);
            self.varint(size);
        }
        self
    }
    fn stop(&mut self) -> &mut Self {
        self.0.push(0);
        self
    }
    /// A schema element: group (`children`) or INT32 leaf.
    fn element(&mut self, name: &[u8], children: Option<i32>) -> &mut Self {
        match children {
            Some(n) => {
                self.field(3, 5).i32(1); // repetition OPTIONAL
                self.field(1, 8).binary(name);
                self.field(1, 5).i32(n);
            }
            None => {
                self.field(1, 5).i32(1); // INT32
                self.field(2, 5).i32(0); // REQUIRED
                self.field(1, 8).binary(name);
            }
        }
        self.stop()
    }
    fn file(&self) -> Bytes {
        let mut out = b"PAR1".to_vec();
        out.extend_from_slice(&self.0);
        out.extend_from_slice(&(self.0.len() as u32).to_le_bytes());
        out.extend_from_slice(b"PAR1");
        Bytes::from(out)
    }
}

const LIST: u8 = 9;
const STRUCT: u8 = 12;

#[test]
fn footers_built_to_exhaust_the_decoder_are_refused() {
    // A schema nested 100,000 deep (the crate builds it recursively).
    let depth = 100_000u64;
    let mut t = Thrift::default();
    t.field(1, 5).i32(1).field(1, LIST).list(STRUCT, depth + 1);
    for _ in 0..depth {
        t.element(b"g", Some(1));
    }
    t.element(b"x", None).stop();
    assert_eq!(
        open(&t.file()).err(),
        Some(OpenError::Limit("the schema is nested too deeply"))
    );

    // A group claiming i32::MAX children (the crate reserves a vector that large).
    let mut t = Thrift::default();
    t.field(1, 5).i32(1).field(1, LIST).list(STRUCT, 2);
    t.element(b"root", Some(i32::MAX))
        .element(b"x", None)
        .stop();
    assert!(matches!(open(&t.file()), Err(OpenError::Malformed(_))));

    // A row-group list claiming 2^31 - 1 entries, first or repeated.
    let schema = |t: &mut Thrift| {
        t.field(1, 5).i32(1).field(1, LIST).list(STRUCT, 2);
        t.element(b"root", Some(1)).element(b"x", None);
        t.field(1, 6).varint(0); // num_rows
    };
    let mut t = Thrift::default();
    schema(&mut t);
    t.field(1, LIST).list(STRUCT, (1 << 31) - 1).stop();
    assert!(matches!(open(&t.file()), Err(OpenError::Malformed(_))));
    let mut t = Thrift::default();
    schema(&mut t);
    t.field(1, LIST).list(STRUCT, 0);
    t.0.extend_from_slice(&[LIST, 0x08]); // field 4 again (full id)
    t.list(STRUCT, (1 << 31) - 1).stop();
    assert!(matches!(open(&t.file()), Err(OpenError::Malformed(_))));

    // Too many columns, and row groups times columns.
    let mut t = Thrift::default();
    t.field(1, 5).i32(1).field(1, LIST).list(STRUCT, 40_001);
    t.element(b"root", Some(40_000));
    for _ in 0..40_000 {
        t.element(b"c", None);
    }
    t.stop();
    assert_eq!(
        open(&t.file()).err(),
        Some(OpenError::Limit("too many columns"))
    );

    // A known field with another declared type, and unknown field types.
    let mut t = Thrift::default();
    t.field(1, 8).binary(b"version?").stop();
    assert!(matches!(open(&t.file()), Err(OpenError::Malformed(_))));
    let mut t = Thrift::default();
    t.field(1, 14).stop();
    assert!(matches!(open(&t.file()), Err(OpenError::Malformed(_))));
}

/// Rewrites the varint of field 2 (`uncompressed_page_size`) of the page header at `at`
/// to `value`, keeping its length (overlong varints are valid).
fn patch_uncompressed_size(bytes: &mut [u8], at: usize, value: u32) {
    assert_eq!(bytes[at], 0x15, "field 1, i32");
    let mut i = at + 2;
    assert_eq!(bytes[i - 1] & 0x80, 0, "a one-byte page type");
    assert_eq!(bytes[i], 0x15, "field 2, i32");
    i += 1;
    let start = i;
    while bytes[i] & 0x80 != 0 {
        i += 1;
    }
    let len = i + 1 - start;
    let mut v = u64::from(value) << 1;
    for (k, byte) in bytes[start..=i].iter_mut().enumerate() {
        let last = k + 1 == len;
        *byte = (v & 0x7f) as u8 | if last { 0 } else { 0x80 };
        v >>= 7;
    }
    assert_eq!(v, 0, "the value fits the old length");
}

#[test]
fn pages_lying_about_their_size_are_refused() {
    quiet_dependency_panics();
    for page_index in [true, false] {
        let options = Options {
            compression: Compression::GZIP(GzipLevel::default()),
            page_index,
            dictionary: false,
            row_group_rows: 100_000,
            page_rows: 20_000,
            ..Options::default()
        };
        let bytes = flat_file(20_000, &options);
        let source = open(&bytes).unwrap();
        let start = source.column_chunk_range(0, 0).unwrap().start as usize;
        let block = source.block_of(0, 0).unwrap();

        // Declares 100 bytes, decompresses to 160 KB: refused before the crate inflates it.
        let mut bomb = bytes.to_vec();
        patch_uncompressed_size(&mut bomb, start, 100);
        let source = open(&Bytes::from(bomb)).unwrap();
        let error = TableSource::fetch(&source, &[block], &CancelToken::new()).unwrap_err();
        assert_eq!(error.kind(), FetchErrorKind::Permanent);
        assert_eq!(
            error.message(),
            "a page decompresses beyond its declared size"
        );

        // Declares more than the page limit.
        let mut huge = bytes.to_vec();
        patch_uncompressed_size(&mut huge, start, 1_000_000);
        let limits = ParquetLimits {
            max_page_bytes: 512 << 10,
            ..ParquetLimits::default()
        };
        let source = ParquetSource::open(MemorySource::new(Bytes::from(huge)), &limits).unwrap();
        let error = TableSource::fetch(&source, &[block], &CancelToken::new()).unwrap_err();
        assert_eq!(error.kind(), FetchErrorKind::Limit, "index={page_index}");
    }
}

#[test]
fn cancelled_fetches_read_nothing() {
    let bytes = small_file(Compression::SNAPPY, true);
    let counting = Arc::new(parquet_table_core::CountingSource::new(MemorySource::new(
        bytes,
    )));
    let source = ParquetSource::open(Arc::clone(&counting), &ParquetLimits::default()).unwrap();
    let opened = counting.bytes();
    let cancel = CancelToken::new();
    cancel.cancel();
    let block = source.block_of(0, 0).unwrap();
    let error = TableSource::fetch(&source, &[block], &cancel).unwrap_err();
    assert_eq!(error.kind(), FetchErrorKind::Cancelled);
    assert_eq!(counting.bytes(), opened);
}
