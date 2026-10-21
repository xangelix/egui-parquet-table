//! [`ColumnInfo`] from the footer: sizes, codecs, types, and statistics aggregated over all
//! row groups.

use arrow_array::{Array, ArrayRef};
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::basic::LogicalType;

use super::{ParquetSource, guarded, nonnegative};
use crate::format::{CellText, format_cell};
use crate::source::ByteSource;
use crate::table::ColumnInfo;

/// Characters kept of a formatted statistic.
const STAT_CHARS: usize = 64;

fn logical_name(logical: &LogicalType) -> String {
    match logical {
        LogicalType::String => "STRING".to_owned(),
        LogicalType::Map => "MAP".to_owned(),
        LogicalType::List => "LIST".to_owned(),
        LogicalType::Enum => "ENUM".to_owned(),
        LogicalType::Decimal(_) => "DECIMAL".to_owned(),
        LogicalType::Date => "DATE".to_owned(),
        LogicalType::Time(_) => "TIME".to_owned(),
        LogicalType::Timestamp(_) => "TIMESTAMP".to_owned(),
        LogicalType::Integer(int) => format!(
            "INT({}, {})",
            int.bit_width,
            if int.is_signed { "signed" } else { "unsigned" }
        ),
        LogicalType::Json => "JSON".to_owned(),
        LogicalType::Bson => "BSON".to_owned(),
        LogicalType::Uuid => "UUID".to_owned(),
        LogicalType::Float16 => "FLOAT16".to_owned(),
        other => format!("{other:?}"),
    }
}

/// The index of the smallest (or largest) non-null value.
fn extreme(values: &ArrayRef, largest: bool) -> Option<usize> {
    let compare = make_comparator(values.as_ref(), values.as_ref(), SortOptions::default()).ok()?;
    let mut best: Option<usize> = None;
    for i in 0..values.len() {
        if values.is_null(i) {
            continue;
        }
        best = match best {
            None => Some(i),
            Some(b) => {
                let ordering = compare(i, b);
                let better = if largest {
                    ordering.is_gt()
                } else {
                    ordering.is_lt()
                };
                Some(if better { i } else { b })
            }
        };
    }
    best
}

fn display(values: &ArrayRef, index: Option<usize>) -> Option<String> {
    match format_cell(values.as_ref(), index?, STAT_CHARS) {
        CellText::Text { text, .. } => Some(text),
        CellText::Null | CellText::Invalid => None,
    }
}

pub(super) fn column_info<B: ByteSource>(source: &ParquetSource<B>, column: usize) -> ColumnInfo {
    let Some(field) = source.schema.fields().get(column) else {
        return ColumnInfo::default();
    };
    let metadata = source.metadata();
    let leaves = source.leaf_columns(column);
    let mut info = ColumnInfo {
        name: field.name().clone(),
        data_type: field.data_type().to_string(),
        leaf_columns: leaves.len(),
        ..ColumnInfo::default()
    };
    for group in metadata.row_groups() {
        for chunk in leaves.clone().filter_map(|leaf| group.columns().get(leaf)) {
            info.compressed_bytes = info
                .compressed_bytes
                .saturating_add(nonnegative(chunk.compressed_size()));
            info.uncompressed_bytes = info
                .uncompressed_bytes
                .saturating_add(nonnegative(chunk.uncompressed_size()));
            let codec = chunk.compression().to_string();
            if !info.codecs.contains(&codec) && info.codecs.len() < 8 {
                info.codecs.push(codec);
            }
        }
    }
    let schema = metadata.file_metadata().schema_descr();
    let leaf = leaves.start;
    let flat = leaves.len() == 1 && schema.get_column_root(leaf).is_primitive();
    if !flat {
        return info;
    }
    let descr = schema.column(leaf);
    info.physical_type = Some(format!("{:?}", descr.physical_type()));
    info.logical_type = descr.logical_type_ref().map(logical_name).or_else(|| {
        let converted = descr.converted_type();
        (converted != parquet::basic::ConvertedType::NONE).then(|| format!("{converted:?}"))
    });
    // Statistics conversion runs on values a hostile file chose: keep panics out.
    let stats = guarded(|| -> Result<_, ()> {
        let converter = StatisticsConverter::from_column_index(leaf, field.as_ref(), schema)
            .map_err(|_| ())?
            .with_missing_null_counts_as_zero(false);
        let groups = metadata.row_groups();
        let nulls = converter.row_group_null_counts(groups.iter()).ok();
        let mins = converter.row_group_mins(groups.iter()).ok();
        let maxes = converter.row_group_maxes(groups.iter()).ok();
        let null_count = nulls.and_then(|n| {
            (n.null_count() == 0).then(|| n.values().iter().fold(0u64, |a, &b| a.saturating_add(b)))
        });
        let min = mins.and_then(|m| display(&m, extreme(&m, false)));
        let max = maxes.and_then(|m| display(&m, extreme(&m, true)));
        Ok((null_count, min, max))
    });
    if let Ok((null_count, min, max)) = stats {
        info.null_count = null_count;
        info.min = min;
        info.max = max;
    }
    info
}
