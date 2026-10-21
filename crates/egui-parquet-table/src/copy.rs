//! Loaded rows and copying: which rows have every shown cell in the engine's cache, and the
//! selection's loaded rows as tab-separated text.
//!
//! Everything here reads the cache only. The rows visited are those of cached blocks, so the
//! work is bounded by the cache's budget however many rows are selected (a Select All of a
//! billion rows copies the loaded ones and counts the rest).

use core::ops::{Range, RangeInclusive};
use std::collections::HashMap;

use egui_table_kit::state::TableState;
use parquet_table_core::{Cell, CellText, LazyTable, format_cell};

use crate::provider::Shared;
use crate::text::grouped;

/// What a copy of the selection did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CopySummary {
    /// Rows copied.
    pub rows: u64,
    /// Selected rows left out because some of their shown cells weren't loaded.
    pub not_loaded: u64,
    /// Selected, loaded rows left out because the copy reached its cell or byte limit.
    pub over_limit: u64,
}

impl CopySummary {
    /// One sentence for a status line: `Copied 12 rows; 3 selected rows weren't loaded`.
    pub fn describe(&self) -> String {
        let mut out = if self.rows == 0 {
            "Nothing copied".to_owned()
        } else {
            format!("Copied {}", rows(self.rows))
        };
        if self.not_loaded > 0 {
            out.push_str(&format!(
                "; {} selected {} loaded",
                rows(self.not_loaded),
                if self.not_loaded == 1 {
                    "wasn't"
                } else {
                    "weren't"
                }
            ));
        }
        if self.over_limit > 0 {
            out.push_str(&format!("; {} over the copy limit", rows(self.over_limit)));
        }
        out
    }
}

/// `n rows` (or `1 row`), grouped.
fn rows(n: u64) -> String {
    if n == 1 {
        "1 row".to_owned()
    } else {
        format!("{} rows", grouped(n))
    }
}

/// A copy of the selection: tab-separated text and what it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Copied {
    /// A header row of the shown columns' names, then one line per copied row, each ending
    /// with a newline; empty when no row was copied. Fields are separated by tabs; a null or a
    /// value that can't be shown is an empty field. Values are one line (controls are escaped)
    /// and cut at the view's hover length.
    pub text: String,
    /// What was copied and what was left out.
    pub summary: CopySummary,
}

/// The selection bits of the rows `rows`, if any can be selected (the kit's bitmaps hold rows
/// below 2^32).
fn bits(rows: &Range<u64>) -> Option<RangeInclusive<u32>> {
    let last = rows.end.checked_sub(1).filter(|&last| last >= rows.start)?;
    let start = u32::try_from(rows.start).ok()?;
    Some(start..=u32::try_from(last).unwrap_or(u32::MAX))
}

/// The selected rows among `rows`, ascending.
pub(crate) fn selected_in<'a>(
    state: &'a TableState,
    rows: &Range<u64>,
) -> impl Iterator<Item = u64> + 'a {
    bits(rows)
        .into_iter()
        .flat_map(move |bits| state.selected_rows.range(bits))
        .map(u64::from)
}

/// How many of `rows` are selected.
fn selected_count(state: &TableState, rows: &Range<u64>) -> u64 {
    bits(rows).map_or(0, |bits| state.selected_rows.range_cardinality(bits))
}

/// Joins sorted ranges that overlap or touch.
fn merge(mut ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut out: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match out.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => out.push(range),
        }
    }
    out
}

/// The rows in both lists of sorted, disjoint ranges.
fn intersect(a: &[Range<u64>], b: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    let (mut left, mut right) = (a.iter().peekable(), b.iter().peekable());
    while let (Some(x), Some(y)) = (left.peek(), right.peek()) {
        let start = x.start.max(y.start);
        let end = x.end.min(y.end);
        if start < end {
            out.push(start..end);
        }
        if x.end <= y.end {
            left.next();
        } else {
            right.next();
        }
    }
    out
}

/// The rows whose cells in every one of `columns` (source columns) are loaded, as sorted,
/// disjoint ranges. Empty without columns. Reads the cache only.
pub(crate) fn loaded_ranges(engine: &LazyTable, columns: &[usize]) -> Vec<Range<u64>> {
    if columns.is_empty() {
        return Vec::new();
    }
    let index: HashMap<usize, usize> = columns
        .iter()
        .enumerate()
        .map(|(position, &column)| (column, position))
        .collect();
    let mut per_column: Vec<Vec<Range<u64>>> = vec![Vec::new(); columns.len()];
    let source = engine.source();
    for block in engine.ready_blocks() {
        if let Some(list) = index
            .get(&block.column)
            .and_then(|&position| per_column.get_mut(position))
        {
            let rows = source.block_rows(block);
            if !rows.is_empty() {
                list.push(rows);
            }
        }
    }
    let mut loaded: Option<Vec<Range<u64>>> = None;
    for list in per_column {
        let list = merge(list);
        let next = match loaded {
            None => list,
            Some(so_far) => intersect(&so_far, &list),
        };
        if next.is_empty() {
            return next;
        }
        loaded = Some(next);
    }
    loaded.unwrap_or_default()
}

/// Copies the selected rows whose shown cells are all loaded, as tab-separated text with a
/// header row, up to `max_cells` cells and about `max_bytes` bytes (a row is copied whole or
/// not at all).
pub(crate) fn copy_selection(
    shared: &Shared,
    state: &TableState,
    max_cells: usize,
    max_bytes: usize,
) -> Copied {
    let columns = &shared.columns.visible;
    let width = columns.len();
    let loaded = loaded_ranges(&shared.engine, columns);
    let selected = state.selected_rows.len();
    let selected_loaded = loaded
        .iter()
        .map(|range| selected_count(state, range))
        .fold(0u64, u64::saturating_add);

    let mut text = String::new();
    for (position, &column) in columns.iter().enumerate() {
        if position > 0 {
            text.push('\t');
        }
        if let Some(name) = shared.columns.names.get(column) {
            text.push_str(name);
        }
    }
    text.push('\n');

    let mut copied = 0u64;
    let mut cells = 0usize;
    'rows: for range in &loaded {
        for row in selected_in(state, range) {
            if cells.saturating_add(width) > max_cells || text.len() >= max_bytes {
                break 'rows;
            }
            for (position, &column) in columns.iter().enumerate() {
                if position > 0 {
                    text.push('\t');
                }
                if let Cell::Ready { array, index } = shared.engine.cell(row, column)
                    && let CellText::Text { text: value, .. } =
                        format_cell(array.as_ref(), index, shared.hover_chars)
                {
                    text.push_str(&value);
                }
            }
            text.push('\n');
            copied = copied.saturating_add(1);
            cells = cells.saturating_add(width);
        }
    }
    if copied == 0 {
        text.clear();
    }
    Copied {
        text,
        summary: CopySummary {
            rows: copied,
            not_loaded: selected.saturating_sub(selected_loaded),
            over_limit: selected_loaded.saturating_sub(copied),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_and_intersect() {
        assert_eq!(merge(vec![10..20, 0..5, 5..8, 15..30]), vec![0..8, 10..30]);
        assert_eq!(
            intersect(&[0..10, 20..30, 40..50], &[5..25, 28..45]),
            vec![5..10, 20..25, 28..30, 40..45]
        );
        let (first, second) = (vec![0..10, 30..40], vec![10..20, 40..50]);
        assert!(
            intersect(&first, &second).is_empty(),
            "touching isn't overlapping"
        );
        assert!(intersect(&[], &second).is_empty());
    }

    #[test]
    fn selection_bits_stop_at_the_bitmaps() {
        assert_eq!(bits(&(0..10)), Some(0..=9));
        assert_eq!(bits(&(5..5)), None);
        assert_eq!(
            bits(&((u64::from(u32::MAX) - 1)..(1 << 40))),
            Some(u32::MAX - 1..=u32::MAX)
        );
        assert_eq!(bits(&((1 << 32)..(1 << 33))), None);
    }

    #[test]
    fn summaries_read_as_sentences() {
        let summary = CopySummary {
            rows: 1_200,
            not_loaded: 1,
            over_limit: 3,
        };
        assert_eq!(
            summary.describe(),
            "Copied 1,200 rows; 1 row selected wasn't loaded; 3 rows over the copy limit"
        );
        assert_eq!(
            CopySummary {
                rows: 0,
                not_loaded: 4,
                over_limit: 0
            }
            .describe(),
            "Nothing copied; 4 rows selected weren't loaded"
        );
    }
}
