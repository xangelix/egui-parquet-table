//! The format-independent table the engine and the widget work with.
//!
//! A [`TableSource`] divides every column into row blocks. A block is the unit that is
//! fetched, cached, and evicted. Blocks of different columns that cover the same rows can
//! be fetched together.

use core::ops::Range;

use arrow_array::ArrayRef;
use arrow_schema::SchemaRef;

use crate::cancel::CancelToken;
use crate::error::FetchError;

/// One block: a range of rows of one column.
///
/// `group` and `index` are the source's own coordinates (for Parquet: the row group and
/// the block within it). Two blocks cover the same rows exactly when
/// [`TableSource::block_rows`] returns the same range for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId {
    /// The column (an index into the schema's top-level fields).
    pub column: usize,
    /// The partition of the rows the block lies in (a Parquet row group).
    pub group: u32,
    /// The block within the partition.
    pub index: u32,
}

/// Static facts about a column, for headers and tooltips.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColumnInfo {
    /// The field name.
    pub name: String,
    /// The Arrow type the column decodes to, as text.
    pub data_type: String,
    /// The storage type (Parquet physical type), when the column is a single leaf.
    pub physical_type: Option<String>,
    /// The logical or converted type, when the column is a single leaf and has one.
    pub logical_type: Option<String>,
    /// How many stored leaf columns make up this column (more than one when nested).
    pub leaf_columns: usize,
    /// Stored (compressed) bytes over all partitions.
    pub compressed_bytes: u64,
    /// Uncompressed bytes over all partitions.
    pub uncompressed_bytes: u64,
    /// Nulls over all partitions, when every partition records them.
    pub null_count: Option<u64>,
    /// The smallest value from the statistics, formatted for display.
    pub min: Option<String>,
    /// The largest value from the statistics, formatted for display.
    pub max: Option<String>,
    /// The compression codecs used, as text (one entry per distinct codec).
    pub codecs: Vec<String>,
}

/// A table whose columns can be fetched block by block.
///
/// Implementations are shared between the UI thread, which only calls the cheap
/// arithmetic methods, and worker threads, which call [`fetch`](Self::fetch).
pub trait TableSource: Send + Sync + 'static {
    /// The table's schema: one field per column.
    fn schema(&self) -> &SchemaRef;

    /// The number of rows.
    fn row_count(&self) -> u64;

    /// The block of `column` that holds `row`, or `None` when either is out of range.
    fn block_of(&self, column: usize, row: u64) -> Option<BlockId>;

    /// The rows a block covers (empty for a block that does not exist).
    fn block_rows(&self, block: BlockId) -> Range<u64>;

    /// An estimate of the block's decoded size in memory.
    fn estimated_bytes(&self, block: BlockId) -> u64;

    /// Fetches the given blocks, which must cover the same rows in different columns.
    /// Returns one array per block, in the order given, each with exactly as many rows as
    /// the block covers. Runs on a worker; may block.
    fn fetch(&self, blocks: &[BlockId], cancel: &CancelToken) -> Result<Vec<ArrayRef>, FetchError>;

    /// Facts about a column for display. May walk all partitions: callers should cache it.
    fn column_info(&self, column: usize) -> ColumnInfo;

    /// The number of columns.
    fn column_count(&self) -> usize {
        self.schema().fields().len()
    }
}
