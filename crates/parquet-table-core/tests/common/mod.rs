//! Shared helpers: Parquet files written with `ArrowWriter`, eager reads, a seeded PRNG.

#![allow(
    dead_code,
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder, StringBuilder};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, StructArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// How a test file is written.
#[derive(Clone, Debug)]
pub struct Options {
    pub compression: Compression,
    pub page_index: bool,
    pub row_group_rows: usize,
    pub page_rows: usize,
    pub v2_pages: bool,
    pub dictionary: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            compression: Compression::SNAPPY,
            page_index: true,
            row_group_rows: 4_000,
            page_rows: 500,
            v2_pages: false,
            dictionary: true,
        }
    }
}

pub fn properties(options: &Options) -> WriterProperties {
    WriterProperties::builder()
        .set_compression(options.compression)
        .set_statistics_enabled(if options.page_index {
            EnabledStatistics::Page
        } else {
            EnabledStatistics::Chunk
        })
        .set_offset_index_disabled(!options.page_index)
        .set_max_row_group_row_count(Some(options.row_group_rows))
        .set_data_page_row_count_limit(options.page_rows)
        .set_write_batch_size(options.page_rows.min(1024))
        .set_dictionary_enabled(options.dictionary)
        .set_writer_version(if options.v2_pages {
            WriterVersion::PARQUET_2_0
        } else {
            WriterVersion::PARQUET_1_0
        })
        .build()
}

pub fn write(batches: &[RecordBatch], options: &Options) -> Bytes {
    let schema = batches[0].schema();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, schema, Some(properties(options))).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    Bytes::from(out)
}

/// Flat columns of many kinds: ints, floats with nulls, a low-cardinality (dictionary)
/// string, unique strings with nulls, booleans, timestamps, binary.
pub fn flat_batch(start: i64, rows: usize, rng: &mut Rng) -> RecordBatch {
    let ids: Int64Array = (start..start + rows as i64).collect();
    let floats: Float64Array = (0..rows)
        .map(|i| (i % 7 != 0).then(|| rng.next() as f64 / 1e15))
        .collect();
    let colors = ["red", "green", "blue", "null", ""];
    let names: StringArray = (0..rows)
        .map(|i| Some(colors[(start as usize + i) % colors.len()]))
        .collect();
    let texts: StringArray = (0..rows)
        .map(|i| {
            (i % 11 != 3).then(|| {
                format!(
                    "row {} {}",
                    start + i as i64,
                    "x".repeat(rng.below(40) as usize)
                )
            })
        })
        .collect();
    let flags: BooleanArray = (0..rows)
        .map(|i| Some(rng.below(3) == 0 || i % 5 == 0))
        .collect();
    let times = TimestampMicrosecondArray::from(
        (0..rows)
            .map(|i| 1_700_000_000_000_000 + (start + i as i64) * 1_000_003)
            .collect::<Vec<_>>(),
    )
    .with_timezone("UTC");
    let blobs: BinaryArray = (0..rows)
        .map(|i| (i % 13 != 0).then(|| (start as u64 + i as u64).to_le_bytes().to_vec()))
        .collect::<Vec<_>>()
        .iter()
        .map(|v| v.as_deref())
        .collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(ids),
        Arc::new(floats),
        Arc::new(names),
        Arc::new(texts),
        Arc::new(flags),
        Arc::new(times),
        Arc::new(blobs),
    ];
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
        Field::new("color", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("blob", DataType::Binary, true),
    ]);
    RecordBatch::try_new(Arc::new(schema), columns).unwrap()
}

/// Nested columns: a list of ints (with null lists and null items), a struct with a null
/// child, and an id.
pub fn nested_batch(start: i64, rows: usize, rng: &mut Rng) -> RecordBatch {
    let ids: Int64Array = (start..start + rows as i64).collect();
    let mut lists = ListBuilder::new(Int32Builder::new());
    for i in 0..rows {
        if i % 9 == 4 {
            lists.append(false);
            continue;
        }
        for j in 0..rng.below(5) {
            if j == 2 {
                lists.values().append_null();
            } else {
                lists
                    .values()
                    .append_value((start as i32).wrapping_add(i as i32 * 3 + j as i32));
            }
        }
        lists.append(true);
    }
    let lists = lists.finish();
    let a: Int32Array = (0..rows)
        .map(|i| (i % 4 != 0).then_some(i as i32))
        .collect();
    let mut b = StringBuilder::new();
    for i in 0..rows {
        b.append_value(format!("s{}", start + i as i64));
    }
    let structs = StructArray::new(
        Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, false),
        ]),
        vec![Arc::new(a), Arc::new(b.finish())],
        None,
    );
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "list",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
            true,
        ),
        Field::new("record", structs.data_type().clone(), false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(ids), Arc::new(lists), Arc::new(structs)],
    )
    .unwrap()
}

/// `rows` rows of flat data in batches of 1,000.
pub fn flat_file(rows: usize, options: &Options) -> Bytes {
    let mut rng = Rng(7);
    let batches: Vec<_> = (0..rows)
        .step_by(1000)
        .map(|start| flat_batch(start as i64, 1000.min(rows - start), &mut rng))
        .collect();
    write(&batches, options)
}

/// `rows` rows of nested data in batches of 1,000.
pub fn nested_file(rows: usize, options: &Options) -> Bytes {
    let mut rng = Rng(11);
    let batches: Vec<_> = (0..rows)
        .step_by(1000)
        .map(|start| nested_batch(start as i64, 1000.min(rows - start), &mut rng))
        .collect();
    write(&batches, options)
}

/// `columns` Int64 columns of `rows` rows.
pub fn wide_file(columns: usize, rows: usize, options: &Options) -> Bytes {
    let fields: Vec<Field> = (0..columns)
        .map(|c| Field::new(format!("c{c}"), DataType::Int64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = (0..columns)
        .map(|c| {
            Arc::new(
                (0..rows as i64)
                    .map(|r| r * 1000 + c as i64)
                    .collect::<Int64Array>(),
            ) as ArrayRef
        })
        .collect();
    let batch = RecordBatch::try_new(schema, arrays).unwrap();
    write(&[batch], options)
}

/// The whole file read eagerly into one batch.
pub fn eager(bytes: &Bytes) -> RecordBatch {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
        .unwrap()
        .with_batch_size(1 << 20)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
}
