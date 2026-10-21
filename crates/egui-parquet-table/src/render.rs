//! How cells look: what kind of value a column holds (from its Arrow type), its default width,
//! and how a value, a null, an undisplayable value, or a row number is drawn.

use egui::{Align, Color32, Layout, Response, RichText, Ui};
use egui_table_kit::layout::CellInfo;
use egui_table_kit::operations::Row;
use parquet_table_core::CellText;
use parquet_table_core::arrow::schema::DataType;

use crate::provider::Adapter;
use crate::style::Paint;

/// The most characters of a binary value's hexadecimal drawn in its cell (16 bytes); the rest
/// is on hover.
const BINARY_CHARS: usize = 32;

/// The widest a column starts, however long its name.
const MAX_START_WIDTH: f32 = 320.0;

/// What a column holds, for its look: alignment, font, and starting width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Signed and unsigned integers.
    Integer,
    /// Floating point and decimal numbers.
    Float,
    /// Booleans.
    Boolean,
    /// Strings.
    Text,
    /// Bytes, shown as hexadecimal.
    Binary,
    /// Dates.
    Date,
    /// Times of day.
    Time,
    /// Timestamps.
    Timestamp,
    /// Durations and intervals.
    Duration,
    /// Lists, structs, maps, and unions, shown compactly.
    Nested,
    /// Anything else.
    Other,
}

impl Kind {
    /// The kind of values of an Arrow type (a dictionary or run-end encoded column is the kind
    /// of its values).
    pub(crate) fn of(data_type: &DataType) -> Self {
        match data_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => Self::Integer,
            DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..) => Self::Float,
            DataType::Boolean => Self::Boolean,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Self::Text,
            DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_) => Self::Binary,
            DataType::Date32 | DataType::Date64 => Self::Date,
            DataType::Time32(_) | DataType::Time64(_) => Self::Time,
            DataType::Timestamp(..) => Self::Timestamp,
            DataType::Duration(_) | DataType::Interval(_) => Self::Duration,
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::FixedSizeList(..)
            | DataType::Struct(_)
            | DataType::Map(..)
            | DataType::Union(..) => Self::Nested,
            DataType::Dictionary(_, values) => Self::of(values),
            DataType::RunEndEncoded(_, values) => Self::of(values.data_type()),
            _ => Self::Other,
        }
    }

    /// The width a column of this kind starts at, in points.
    pub(crate) const fn width(self) -> f32 {
        match self {
            Self::Boolean => 64.0,
            Self::Integer => 90.0,
            Self::Float | Self::Date => 110.0,
            Self::Time => 120.0,
            Self::Duration | Self::Other => 140.0,
            Self::Binary => 160.0,
            Self::Timestamp => 180.0,
            Self::Text => 200.0,
            Self::Nested => 240.0,
        }
    }

    /// Whether values are right-aligned (numbers).
    pub(crate) const fn right_aligned(self) -> bool {
        matches!(self, Self::Integer | Self::Float)
    }

    /// Whether values are drawn in the monospace font.
    pub(crate) const fn monospace(self) -> bool {
        !matches!(self, Self::Text | Self::Nested | Self::Other)
    }

    /// The most characters of a value drawn in its cell.
    pub(crate) fn cell_chars(self, cell_chars: usize) -> usize {
        match self {
            Self::Binary => cell_chars.min(BINARY_CHARS),
            _ => cell_chars,
        }
    }
}

/// A short name of an Arrow type for a column's header: `int64`, `text`, `timestamp (UTC)`,
/// `list<int32>`. Dictionary and run-end encodings are storage: a column is named by its values.
pub(crate) fn type_name(data_type: &DataType) -> String {
    match data_type {
        DataType::Null => "null".to_owned(),
        DataType::Boolean => "bool".to_owned(),
        DataType::Int8 => "int8".to_owned(),
        DataType::Int16 => "int16".to_owned(),
        DataType::Int32 => "int32".to_owned(),
        DataType::Int64 => "int64".to_owned(),
        DataType::UInt8 => "uint8".to_owned(),
        DataType::UInt16 => "uint16".to_owned(),
        DataType::UInt32 => "uint32".to_owned(),
        DataType::UInt64 => "uint64".to_owned(),
        DataType::Float16 => "float16".to_owned(),
        DataType::Float32 => "float32".to_owned(),
        DataType::Float64 => "float64".to_owned(),
        DataType::Decimal32(precision, scale)
        | DataType::Decimal64(precision, scale)
        | DataType::Decimal128(precision, scale)
        | DataType::Decimal256(precision, scale) => format!("decimal({precision}, {scale})"),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "text".to_owned(),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => "binary".to_owned(),
        DataType::FixedSizeBinary(size) => format!("binary[{size}]"),
        DataType::Date32 | DataType::Date64 => "date".to_owned(),
        DataType::Time32(_) | DataType::Time64(_) => "time".to_owned(),
        DataType::Timestamp(_, None) => "timestamp".to_owned(),
        DataType::Timestamp(_, Some(zone)) => format!("timestamp ({zone})"),
        DataType::Duration(_) => "duration".to_owned(),
        DataType::Interval(_) => "interval".to_owned(),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field) => format!("list<{}>", type_name(field.data_type())),
        DataType::FixedSizeList(field, size) => {
            format!("list<{}>[{size}]", type_name(field.data_type()))
        }
        DataType::Struct(fields) => format!("struct ({} fields)", fields.len()),
        DataType::Map(..) => "map".to_owned(),
        DataType::Union(..) => "union".to_owned(),
        DataType::Dictionary(_, values) => type_name(values),
        DataType::RunEndEncoded(_, values) => type_name(values.data_type()),
    }
}

/// A column's starting width: its kind's, widened to show a name of `name_width` points (up to
/// a bound).
pub(crate) fn start_width(kind: Kind, name_width: f32) -> f32 {
    // The header's padding on both sides, and the gap the kit keeps beside its title.
    let header = name_width + 28.0;
    kind.width().max(header.min(MAX_START_WIDTH))
}

/// Draws `text` in the cell, on the left or the right, cut to the cell's width. When
/// `elided_tooltip` is set, a value cut by the width shows itself on hover.
fn label(ui: &mut Ui, text: RichText, right: bool, elided_tooltip: bool) -> Response {
    let layout = if right {
        Layout::right_to_left(Align::Center)
    } else {
        Layout::left_to_right(Align::Center)
    };
    ui.with_layout(layout, |ui| {
        ui.add(
            egui::Label::new(text)
                .selectable(false)
                .truncate()
                .show_tooltip_when_elided(elided_tooltip),
        )
    })
    .inner
}

/// The custom cell renderer handed to the kit: row numbers in column 0, values elsewhere. Asks
/// nothing of the source: values come from the engine's cache through the adapter.
pub(crate) fn cell(
    ui: &mut Ui,
    adapter: &Adapter<'_>,
    paint: &Paint,
    cell: &CellInfo,
    row: &dyn Row,
    color: Color32,
) -> Option<Response> {
    let row = u64::try_from(row.row_index()?).ok()?;
    let Some(column) = cell.col_nr.checked_sub(1) else {
        // The gutter: its own background and a hairline on its right edge.
        let spacing = ui.spacing().item_spacing;
        let rect = ui
            .max_rect()
            .expand2(egui::vec2(spacing.x * 0.5, spacing.y * 0.5));
        ui.painter().rect_filled(rect, 0.0, paint.gutter);
        ui.painter().vline(
            rect.right() - 0.5,
            rect.y_range(),
            egui::Stroke::new(1.0, paint.border),
        );
        let number = row.checked_add(1)?.to_string();
        let text = RichText::new(number).small().monospace().color(paint.muted);
        return Some(label(ui, text, true, true));
    };
    let kind = adapter.kind(column)?;
    let value = adapter.text(row, column)?;
    Some(match value {
        CellText::Null => {
            let text = RichText::new("null").italics().color(paint.muted);
            label(ui, text, kind.right_aligned(), false)
        }
        CellText::Invalid => {
            let text = RichText::new("⚠").color(ui.visuals().warn_fg_color);
            label(ui, text, kind.right_aligned(), false).on_hover_text("This value can't be shown")
        }
        CellText::Text { text, truncated } => {
            let color = match kind {
                Kind::Boolean if text == "true" => paint.accent,
                Kind::Boolean | Kind::Binary => paint.muted,
                _ => color,
            };
            let mut rich = RichText::new(text).color(color);
            if kind.monospace() {
                rich = rich.monospace();
            }
            let response = label(ui, rich, kind.right_aligned(), !truncated);
            if truncated && response.hovered() {
                let full = adapter.full_text(row, column);
                response.on_hover_ui(|ui| {
                    ui.set_max_width(ui.spacing().tooltip_width.max(480.0));
                    match &full {
                        Some(CellText::Text { text, .. }) => {
                            ui.label(text.as_str());
                        }
                        _ => {
                            ui.label("This value can't be shown");
                        }
                    }
                })
            } else {
                response
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use parquet_table_core::arrow::schema::{Field, TimeUnit};

    #[test]
    fn kinds_follow_the_arrow_type() {
        assert_eq!(Kind::of(&DataType::Int64), Kind::Integer);
        assert_eq!(Kind::of(&DataType::Decimal128(10, 2)), Kind::Float);
        assert_eq!(Kind::of(&DataType::Boolean), Kind::Boolean);
        assert_eq!(
            Kind::of(&DataType::Timestamp(TimeUnit::Microsecond, None)),
            Kind::Timestamp
        );
        assert_eq!(
            Kind::of(&DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Utf8)
            )),
            Kind::Text
        );
        let list = DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true)));
        assert_eq!(Kind::of(&list), Kind::Nested);
        assert_eq!(Kind::of(&DataType::Null), Kind::Other);
    }

    #[test]
    fn types_have_short_names_by_their_values() {
        assert_eq!(type_name(&DataType::Int64), "int64");
        assert_eq!(type_name(&DataType::Utf8View), "text");
        assert_eq!(type_name(&DataType::Decimal128(10, 2)), "decimal(10, 2)");
        assert_eq!(
            type_name(&DataType::Timestamp(TimeUnit::Second, Some("UTC".into()))),
            "timestamp (UTC)"
        );
        let list = DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true)));
        assert_eq!(type_name(&list), "list<int32>");
        assert_eq!(
            type_name(&DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Utf8)
            )),
            "text"
        );
    }

    #[test]
    fn widths_follow_the_kind_and_the_name() {
        assert!(Kind::Boolean.width() < Kind::Integer.width());
        assert!(Kind::Integer.width() < Kind::Float.width());
        assert!(Kind::Float.width() < Kind::Timestamp.width());
        assert!(Kind::Timestamp.width() < Kind::Text.width());
        assert!(Kind::Text.width() < Kind::Nested.width());
        assert!((start_width(Kind::Boolean, 10.0) - 64.0).abs() < f32::EPSILON);
        assert!((start_width(Kind::Boolean, 100.0) - 128.0).abs() < f32::EPSILON);
        assert!((start_width(Kind::Text, 10_000.0) - MAX_START_WIDTH).abs() < f32::EPSILON);
        assert!(Kind::Integer.right_aligned() && !Kind::Text.right_aligned());
        assert_eq!(Kind::Binary.cell_chars(256), BINARY_CHARS);
        assert_eq!(Kind::Text.cell_chars(256), 256);
    }
}
