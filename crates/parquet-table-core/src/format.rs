//! Cell text for the widget.
//!
//! [`format_cell`] renders one value with `arrow-cast`'s display, bounded: formatting stops
//! as soon as the text reaches `max_chars` (a long string or a large nested value is never
//! rendered in full). Text is one line: control characters, and the bidirectional controls
//! that could reorder what is displayed around them, are shown as escapes. Binary values
//! are hexadecimal, nested values compact (`[1, 2]`, `{a: 1}`, with strings quoted inside
//! them), and a null is [`CellText::Null`], never the text `"null"`.
//!
//! Timestamps with a named time zone (`"UTC"`, `"Europe/Berlin"`) are shown as the UTC
//! instant with a `Z` suffix: the crate is built without a time-zone database (only fixed
//! offsets such as `"+02:00"` are applied). Inside nested values such timestamps cannot be
//! shown and the cell is [`CellText::Invalid`].

use core::fmt::{self, Write};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::timezone::Tz;
use arrow_array::types::{
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use arrow_array::{Array, ArrayRef};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{DataType, TimeUnit};

/// The display text of one cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellText {
    /// The value is null.
    Null,
    /// The value's text.
    Text {
        /// The text, ending with `…` when truncated.
        text: String,
        /// Whether the text was cut at `max_chars`.
        truncated: bool,
    },
    /// The value cannot be displayed (index out of range or an unsupported type).
    Invalid,
}

impl CellText {
    /// The text to draw: nulls as an empty string, invalid cells as `?`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Null => "",
            Self::Text { text, .. } => text,
            Self::Invalid => "?",
        }
    }
}

/// Whether a value of this type is shown as a nested value.
fn is_nested(data_type: &DataType) -> bool {
    match data_type {
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Struct(_)
        | DataType::Map(_, _)
        | DataType::Union(_, _) => true,
        DataType::Dictionary(_, value) => is_nested(value),
        DataType::RunEndEncoded(_, value) => is_nested(value.data_type()),
        _ => false,
    }
}

/// A timestamp array with a time zone the build cannot resolve, re-typed without it.
fn without_named_zone(array: &dyn Array) -> Option<ArrayRef> {
    let DataType::Timestamp(unit, Some(zone)) = array.data_type() else {
        return None;
    };
    if zone.parse::<Tz>().is_ok() {
        return None;
    }
    let none = None::<String>;
    Some(match unit {
        TimeUnit::Second => Arc::new(
            array
                .as_primitive_opt::<TimestampSecondType>()?
                .clone()
                .with_timezone_opt(none),
        ),
        TimeUnit::Millisecond => Arc::new(
            array
                .as_primitive_opt::<TimestampMillisecondType>()?
                .clone()
                .with_timezone_opt(none),
        ),
        TimeUnit::Microsecond => Arc::new(
            array
                .as_primitive_opt::<TimestampMicrosecondType>()?
                .clone()
                .with_timezone_opt(none),
        ),
        TimeUnit::Nanosecond => Arc::new(
            array
                .as_primitive_opt::<TimestampNanosecondType>()?
                .clone()
                .with_timezone_opt(none),
        ),
    })
}

/// A writer that keeps at most `max` characters, escaping as it goes.
struct Bounded {
    out: String,
    chars: usize,
    max: usize,
    truncated: bool,
}

impl Bounded {
    fn push(&mut self, c: char) -> fmt::Result {
        if self.chars >= self.max {
            self.truncated = true;
            return Err(fmt::Error);
        }
        self.out.push(c);
        self.chars = self.chars.saturating_add(1);
        Ok(())
    }

    fn push_str(&mut self, s: &str) -> fmt::Result {
        s.chars().try_for_each(|c| self.push(c))
    }
}

/// Characters shown as escapes: C0 and C1 controls, DEL, and bidirectional controls.
fn needs_escape(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
        )
}

impl Write for Bounded {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            match c {
                '\n' => self.push_str("\\n")?,
                '\r' => self.push_str("\\r")?,
                '\t' => self.push_str("\\t")?,
                c if needs_escape(c) => {
                    let mut escaped = String::with_capacity(10);
                    write!(escaped, "\\u{{{:x}}}", u32::from(c))?;
                    self.push_str(&escaped)?;
                }
                c => self.push(c)?,
            }
        }
        Ok(())
    }
}

/// Formats the value at `index` of `array` in at most `max_chars` characters (plus a
/// trailing `…` when cut).
///
/// Builds a formatter per call; a widget drawing many cells per frame should cache the
/// text per visible cell.
pub fn format_cell(array: &dyn Array, index: usize, max_chars: usize) -> CellText {
    if index >= array.len() {
        return CellText::Invalid;
    }
    if array
        .logical_nulls()
        .is_some_and(|nulls| nulls.is_null(index))
    {
        return CellText::Null;
    }
    let naive = without_named_zone(array);
    let shown: &dyn Array = naive.as_deref().unwrap_or(array);
    let options = FormatOptions::new()
        .with_null("null")
        .with_display_error(false)
        .with_quoted_strings(is_nested(array.data_type()));
    let Ok(formatter) = ArrayFormatter::try_new(shown, &options) else {
        return CellText::Invalid;
    };
    let mut out = Bounded {
        out: String::new(),
        chars: 0,
        max: max_chars,
        truncated: false,
    };
    let written = formatter.value(index).write(&mut out).is_ok()
        && (naive.is_none() || out.write_char('Z').is_ok());
    if !written && !out.truncated {
        return CellText::Invalid;
    }
    let mut text = out.out;
    if out.truncated {
        text.push('…');
    }
    CellText::Text {
        text,
        truncated: out.truncated,
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{BinaryArray, Int32Array, StringArray, StructArray};
    use arrow_schema::{Field, Fields};

    fn text(s: &str) -> CellText {
        CellText::Text {
            text: s.to_owned(),
            truncated: false,
        }
    }

    #[test]
    fn nulls_differ_from_the_string_null() {
        let array = StringArray::from(vec![Some("null"), None]);
        assert_eq!(format_cell(&array, 0, 10), text("null"));
        assert_eq!(format_cell(&array, 1, 10), CellText::Null);
        assert_eq!(format_cell(&array, 2, 10), CellText::Invalid);
    }

    #[test]
    fn text_is_bounded_and_one_line() {
        let long = "x".repeat(10_000);
        let array = StringArray::from(vec![long.as_str(), "a\nb\u{202E}c"]);
        let CellText::Text { text, truncated } = format_cell(&array, 0, 8) else {
            panic!("expected text");
        };
        assert!(truncated);
        assert_eq!(text, "xxxxxxxx…");
        assert_eq!(format_cell(&array, 1, 64), text_of("a\\nb\\u{202e}c"));
    }

    fn text_of(s: &str) -> CellText {
        text(s)
    }

    #[test]
    fn binary_is_hex_and_nested_is_compact() {
        let array = BinaryArray::from(vec![&[0xde_u8, 0xad][..]]);
        assert_eq!(format_cell(&array, 0, 64), text("dead"));

        let mut list = ListBuilder::new(StringBuilder::new());
        list.values().append_value("a");
        list.values().append_null();
        list.append(true);
        let list = list.finish();
        assert_eq!(format_cell(&list, 0, 64), text("[\"a\", null]"));

        let ints: Arc<dyn Array> = Arc::new(Int32Array::from(vec![7]));
        let structs = StructArray::new(
            Fields::from(vec![Field::new("n", arrow_schema::DataType::Int32, false)]),
            vec![ints],
            None,
        );
        assert_eq!(format_cell(&structs, 0, 64), text("{n: 7}"));
    }

    #[test]
    fn named_time_zones_show_the_utc_instant() {
        use arrow_array::TimestampSecondArray;
        let utc = TimestampSecondArray::from(vec![0]).with_timezone("UTC");
        assert_eq!(format_cell(&utc, 0, 64), text("1970-01-01T00:00:00Z"));
        let offset = TimestampSecondArray::from(vec![0]).with_timezone("+02:00");
        assert_eq!(
            format_cell(&offset, 0, 64),
            text("1970-01-01T02:00:00+02:00")
        );
    }
}
