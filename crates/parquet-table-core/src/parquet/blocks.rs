//! Row blocks: how each column of each row group is divided for fetching.
//!
//! Every (column, row group) pair gets one fixed block length, chosen so a block's
//! estimated decoded size is about the target, rounded down to a power of two and clamped
//! to the configured bounds. Blocks start at the row group's first row, so with power-of-two
//! lengths the blocks of two columns in one row group either cover the same rows or nest,
//! and neighbouring columns of similar width share fetches. Block boundaries are computed
//! arithmetically; nothing is stored per block or per page.

use core::ops::Range;

use crate::table::BlockId;

/// The block layout of a file.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    /// The first row of each row group, then the total row count.
    starts: Vec<u64>,
    groups: usize,
    columns: usize,
    /// Rows per block, indexed `column * groups + group`.
    rows_per_block: Vec<u64>,
    /// Estimated decoded bytes of the whole column chunk, indexed like `rows_per_block`.
    chunk_bytes: Vec<u64>,
}

/// The sizing rules.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sizing {
    pub target_block_bytes: u64,
    pub min_block_rows: u64,
    pub max_block_rows: u64,
}

/// The largest power of two not above `n` (`n` > 0).
fn floor_pow2(n: u64) -> u64 {
    1u64.checked_shl(63u32.saturating_sub(n.leading_zeros()))
        .unwrap_or(1)
}

/// Rows per block for a chunk of `rows` rows estimated at `bytes` decoded bytes.
pub(crate) fn rows_per_block(bytes: u64, rows: u64, sizing: &Sizing) -> u64 {
    let min = sizing.min_block_rows.max(1);
    let max = sizing.max_block_rows.max(min);
    let wanted = if bytes == 0 {
        max
    } else {
        // target * rows / bytes, in u128 so nothing overflows.
        let n = u128::from(sizing.target_block_bytes)
            .saturating_mul(u128::from(rows))
            .checked_div(u128::from(bytes))
            .unwrap_or(0);
        u64::try_from(n).unwrap_or(u64::MAX)
    };
    let mut per_block = floor_pow2(wanted.max(1)).clamp(min, max);
    // Block indices are u32: widen blocks of absurdly large row groups.
    let needed = rows.div_ceil(u64::from(u32::MAX)).max(1);
    if per_block < needed {
        per_block = needed;
    }
    per_block
}

impl Layout {
    /// A layout for `columns` columns over row groups of the given row counts.
    /// `estimate(column, group)` is the estimated decoded size of that column chunk.
    /// Returns `None` if the counts overflow.
    pub(crate) fn new(
        group_rows: &[u64],
        columns: usize,
        sizing: &Sizing,
        mut estimate: impl FnMut(usize, usize) -> u64,
    ) -> Option<Self> {
        let groups = group_rows.len();
        let mut starts = Vec::with_capacity(groups.checked_add(1)?);
        let mut total = 0u64;
        for &rows in group_rows {
            starts.push(total);
            total = total.checked_add(rows)?;
        }
        starts.push(total);
        let cells = columns.checked_mul(groups)?;
        let mut rows_per = Vec::with_capacity(cells);
        let mut chunk_bytes = Vec::with_capacity(cells);
        for column in 0..columns {
            for (group, &rows) in group_rows.iter().enumerate() {
                let bytes = estimate(column, group);
                rows_per.push(rows_per_block(bytes, rows, sizing));
                chunk_bytes.push(bytes);
            }
        }
        Some(Self {
            starts,
            groups,
            columns,
            rows_per_block: rows_per,
            chunk_bytes,
        })
    }

    /// Total rows.
    pub(crate) fn rows(&self) -> u64 {
        self.starts.last().copied().unwrap_or(0)
    }

    /// Number of row groups.
    pub(crate) fn groups(&self) -> usize {
        self.groups
    }

    /// The rows of a row group.
    pub(crate) fn group_rows(&self, group: usize) -> Range<u64> {
        let start = self.starts.get(group).copied().unwrap_or(0);
        let end = self
            .starts
            .get(group.saturating_add(1))
            .copied()
            .unwrap_or(start);
        start..end
    }

    /// The row group holding `row`.
    pub(crate) fn group_of(&self, row: u64) -> Option<usize> {
        if row >= self.rows() {
            return None;
        }
        let first_starts = self.starts.get(..self.groups)?;
        // The last group starting at or before `row`; empty groups share their start with
        // the next one, so this is never an empty group.
        first_starts.partition_point(|&s| s <= row).checked_sub(1)
    }

    fn cell(&self, column: usize, group: usize) -> Option<usize> {
        if column >= self.columns || group >= self.groups {
            return None;
        }
        column.checked_mul(self.groups)?.checked_add(group)
    }

    /// Rows per block of a column in a row group.
    pub(crate) fn rows_per_block(&self, column: usize, group: usize) -> Option<u64> {
        self.rows_per_block.get(self.cell(column, group)?).copied()
    }

    /// The estimated decoded bytes of a whole column chunk.
    pub(crate) fn chunk_bytes(&self, column: usize, group: usize) -> Option<u64> {
        self.chunk_bytes.get(self.cell(column, group)?).copied()
    }

    /// The block of `column` that holds `row`.
    pub(crate) fn block_of(&self, column: usize, row: u64) -> Option<BlockId> {
        let group = self.group_of(row)?;
        let per = self.rows_per_block(column, group)?;
        let within = row.checked_sub(self.group_rows(group).start)?;
        let index = within.checked_div(per)?;
        Some(BlockId {
            column,
            group: u32::try_from(group).ok()?,
            index: u32::try_from(index).ok()?,
        })
    }

    /// The rows a block covers (empty if it does not exist).
    pub(crate) fn block_rows(&self, block: BlockId) -> Range<u64> {
        let group = usize::try_from(block.group).unwrap_or(usize::MAX);
        let Some(per) = self.rows_per_block(block.column, group) else {
            return 0..0;
        };
        let rows = self.group_rows(group);
        let Some(start) = u64::from(block.index)
            .checked_mul(per)
            .and_then(|offset| rows.start.checked_add(offset))
        else {
            return 0..0;
        };
        if start >= rows.end {
            return 0..0;
        }
        start..start.saturating_add(per).min(rows.end)
    }

    /// The estimated decoded bytes of `rows` rows of a column in a row group.
    pub(crate) fn estimate_rows(&self, column: usize, group: usize, rows: u64) -> u64 {
        let total = self.group_rows(group);
        let total = total.end.saturating_sub(total.start);
        let chunk = self.chunk_bytes(column, group).unwrap_or(0);
        if total == 0 {
            return 0;
        }
        let n = u128::from(chunk)
            .saturating_mul(u128::from(rows))
            .checked_div(u128::from(total))
            .unwrap_or(0);
        u64::try_from(n).unwrap_or(u64::MAX)
    }

    /// The estimated decoded bytes of a block.
    pub(crate) fn estimated_bytes(&self, block: BlockId) -> u64 {
        let rows = self.block_rows(block);
        let group = usize::try_from(block.group).unwrap_or(usize::MAX);
        self.estimate_rows(block.column, group, rows.end.saturating_sub(rows.start))
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    const SIZING: Sizing = Sizing {
        target_block_bytes: 1000,
        min_block_rows: 4,
        max_block_rows: 64,
    };

    #[test]
    fn block_lengths() {
        assert_eq!(floor_pow2(1), 1);
        assert_eq!(floor_pow2(1000), 512);
        assert_eq!(floor_pow2(u64::MAX), 1 << 63);
        // 10 bytes per row -> 100 rows wanted -> 64 (max).
        assert_eq!(rows_per_block(10_000, 1000, &SIZING), 64);
        // 100 bytes per row -> 10 rows -> 8.
        assert_eq!(rows_per_block(100_000, 1000, &SIZING), 8);
        // 10 KB per row -> 0 rows -> min 4.
        assert_eq!(rows_per_block(10_000_000, 1000, &SIZING), 4);
        assert_eq!(rows_per_block(0, 1000, &SIZING), 64);
        // Huge row counts still fit u32 block indices.
        let per = rows_per_block(1, u64::MAX, &SIZING);
        assert!(u64::MAX.div_ceil(per) <= u64::from(u32::MAX));
    }

    #[test]
    fn blocks_cover_every_row_once() {
        // Groups: 100 rows, 0 rows, 37 rows.
        let layout = Layout::new(&[100, 0, 37], 2, &SIZING, |column, _| {
            if column == 0 { 1_000 } else { 100_000 }
        })
        .unwrap();
        assert_eq!(layout.rows(), 137);
        for column in 0..2 {
            let mut row = 0;
            while row < layout.rows() {
                let block = layout.block_of(column, row).unwrap();
                let rows = layout.block_rows(block);
                assert_eq!(rows.start, row, "column {column}");
                assert!(!rows.is_empty());
                for r in rows.clone() {
                    assert_eq!(layout.block_of(column, r), Some(block));
                }
                row = rows.end;
            }
            assert_eq!(layout.block_of(column, 137), None);
        }
        assert_eq!(layout.block_of(2, 0), None);
        assert_eq!(layout.group_of(100), Some(2));
        let bogus = BlockId {
            column: 0,
            group: 9,
            index: 0,
        };
        assert!(layout.block_rows(bogus).is_empty());
    }
}
