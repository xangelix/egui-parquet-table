//! Text for the toolbar, the status line, headers, and tooltips: grouped numbers, byte sizes,
//! one-line column names, and the column description shown on a header.

use core::fmt::Write as _;

use parquet_table_core::ColumnInfo;

/// The most characters of a column name shown (a longer one ends with `…`).
pub(crate) const NAME_CHARS: usize = 256;

/// `n` with its thousands separated by commas: `2,000,000`.
pub(crate) fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len().saturating_mul(4) / 3);
    let mut left = digits.len();
    for c in digits.chars() {
        if !out.is_empty() && left.is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
        left = left.saturating_sub(1);
    }
    out
}

/// A size in binary units: `512 B`, `1.5 KiB`, `12.3 MiB`.
pub(crate) fn bytes(n: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    #[allow(clippy::cast_precision_loss)] // A display value.
    let mut value = n as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len().saturating_sub(1) {
        value /= 1024.0;
        unit = unit.saturating_add(1);
    }
    format!("{value:.1} {}", UNITS.get(unit).copied().unwrap_or("B"))
}

/// Whether a character is shown as an escape: controls, and the bidirectional controls that
/// could reorder the text around them (the same set as cell text).
fn needs_escape(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
        )
}

/// A name as shown: one line, controls and bidirectional controls escaped (a file may name a
/// column anything), at most [`NAME_CHARS`] characters.
pub(crate) fn display_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len().min(NAME_CHARS));
    let mut chars = 0usize;
    for c in name.chars() {
        if chars >= NAME_CHARS {
            out.push('…');
            break;
        }
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if needs_escape(c) => {
                let _ = write!(out, "\\u{{{:x}}}", u32::from(c));
            }
            c => out.push(c),
        }
        chars = chars.saturating_add(1);
    }
    out
}

/// The description of a column shown on its header: its types, codecs, sizes, nulls, and
/// range, as far as the source knows them.
pub(crate) fn column_tooltip(info: &ColumnInfo) -> String {
    let mut out = display_name(&info.name);
    let _ = write!(out, "\nType: {}", info.data_type);
    match (&info.physical_type, &info.logical_type) {
        (Some(physical), Some(logical)) => {
            let _ = write!(out, "\nStored as: {physical} ({logical})");
        }
        (Some(physical), None) => {
            let _ = write!(out, "\nStored as: {physical}");
        }
        (None, Some(logical)) => {
            let _ = write!(out, "\nLogical type: {logical}");
        }
        (None, None) => {}
    }
    if info.leaf_columns > 1 {
        let _ = write!(out, "\nStored in {} leaf columns", info.leaf_columns);
    }
    if !info.codecs.is_empty() {
        let _ = write!(out, "\nCompression: {}", info.codecs.join(", "));
    }
    if info.compressed_bytes > 0 || info.uncompressed_bytes > 0 {
        let _ = write!(
            out,
            "\nSize: {} compressed, {} uncompressed",
            bytes(info.compressed_bytes),
            bytes(info.uncompressed_bytes)
        );
    }
    if let Some(nulls) = info.null_count {
        let _ = write!(out, "\nNulls: {}", grouped(nulls));
    }
    if let Some(min) = &info.min {
        let _ = write!(out, "\nMin: {min}");
    }
    if let Some(max) = &info.max {
        let _ = write!(out, "\nMax: {max}");
    }
    out
}

/// A row number typed by the user (1-based): digits with optional `,`, `_`, or space
/// separators.
pub(crate) fn parse_row_number(text: &str) -> Option<u64> {
    let digits: String = text
        .trim()
        .chars()
        .filter(|c| !matches!(c, ',' | '_' | ' '))
        .collect();
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok().filter(|&n| n >= 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_grouped() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(2_000_000), "2,000,000");
        assert_eq!(grouped(u64::MAX), "18,446,744,073,709,551,615");
    }

    #[test]
    fn sizes_are_in_binary_units() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(1536), "1.5 KiB");
        assert_eq!(bytes(128 << 20), "128.0 MiB");
        assert_eq!(bytes(u64::MAX), "16384.0 PiB");
    }

    #[test]
    fn names_are_one_line_and_bounded() {
        assert_eq!(display_name("a\tb\nc\u{202E}d"), "a\\tb\\nc\\u{202e}d");
        let long = "x".repeat(1_000);
        let shown = display_name(&long);
        assert_eq!(shown.chars().count(), NAME_CHARS + 1);
        assert!(shown.ends_with('…'));
    }

    #[test]
    fn row_numbers_parse_leniently_but_only_as_positive_numbers() {
        assert_eq!(parse_row_number(" 1,234_567 "), Some(1_234_567));
        assert_eq!(parse_row_number("0"), None);
        assert_eq!(parse_row_number("-5"), None);
        assert_eq!(parse_row_number("12a"), None);
        assert_eq!(parse_row_number(""), None);
        assert_eq!(parse_row_number("99999999999999999999999"), None);
    }

    #[test]
    fn the_tooltip_says_what_is_known() {
        let info = ColumnInfo {
            name: "price".to_owned(),
            data_type: "Float64".to_owned(),
            physical_type: Some("DOUBLE".to_owned()),
            logical_type: None,
            leaf_columns: 1,
            compressed_bytes: 2048,
            uncompressed_bytes: 4096,
            null_count: Some(12_345),
            min: Some("0.5".to_owned()),
            max: Some("99.5".to_owned()),
            codecs: vec!["SNAPPY".to_owned()],
        };
        assert_eq!(
            column_tooltip(&info),
            "price\nType: Float64\nStored as: DOUBLE\nCompression: SNAPPY\n\
             Size: 2.0 KiB compressed, 4.0 KiB uncompressed\nNulls: 12,345\nMin: 0.5\nMax: 99.5"
        );
        let bare = ColumnInfo {
            name: "x".to_owned(),
            data_type: "Int32".to_owned(),
            ..ColumnInfo::default()
        };
        assert_eq!(column_tooltip(&bare), "x\nType: Int32");
    }
}
