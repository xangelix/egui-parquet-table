//! Parquet files as a [`TableSource`]: lazy in columns (only the requested column chunks
//! are read) and in rows (with a page index, only the pages that hold the requested rows).
//!
//! [`ParquetSource::open`] reads the 8-byte trailer and the footer, pre-checks the footer
//! against the limits (with a small Thrift reader of its own), decodes it, validates every row count and column
//! chunk range, and loads the offset index (the page index part that locates pages) when it
//! fits under its cap. [`ParquetSource::fetch`] decodes one row range of one row group for
//! a set of columns through the `parquet` crate's synchronous Arrow reader.
//!
//! Without an offset index, the reader cannot jump to the page holding a row: it reads every
//! page header of the column chunk from the start and decodes (then discards) the values
//! before the requested rows. Fetching the last block of a large row group then costs about
//! as much as reading the chunk once. Files written with page statistics enabled (the
//! default in most writers today) carry an offset index.

mod blocks;
mod expansion;
mod info;
mod reader;
mod thrift;

use core::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelectionPolicy, RowSelector,
};
use parquet::basic::Type as PhysicalType;
use parquet::file::metadata::{
    ColumnChunkMetaData, PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader,
};

use self::blocks::{Layout, Sizing};
use self::reader::{Adapter, ChunkGuard, ReadCaps, Region, Shared};
use self::thrift::{FooterLimits, ThriftError};
use crate::cancel::CancelToken;
use crate::error::{FetchError, FetchErrorKind, OpenError};
use crate::source::{ByteSource, SourceError};
use crate::table::{BlockId, ColumnInfo, TableSource};

/// Limits and sizing for opening and reading a Parquet file.
///
/// Every limit exists because a hostile file can declare any size it likes: the defaults
/// admit ordinary files and refuse absurd ones before anything large is allocated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetLimits {
    /// The largest footer (file metadata) read and decoded.
    pub max_footer_bytes: u64,
    /// The largest offset index loaded; a larger one is ignored (reads then go without it).
    pub max_page_index_bytes: u64,
    /// The most stored (leaf) columns.
    pub max_columns: usize,
    /// The most row groups.
    pub max_row_groups: usize,
    /// The most column chunks (row groups times leaf columns): bounds the decoded
    /// metadata's memory.
    pub max_column_chunks: u64,
    /// The deepest schema nesting.
    pub max_schema_depth: usize,
    /// The decoded size a block aims for.
    pub target_block_bytes: u64,
    /// The fewest rows in a block.
    pub min_block_rows: u64,
    /// The most rows in a block.
    pub max_block_rows: u64,
    /// The largest estimated decoded size of one fetch; larger fetches fail with
    /// [`FetchErrorKind::Limit`].
    pub max_block_bytes: u64,
    /// The largest page size (compressed or uncompressed) a page header may declare.
    pub max_page_bytes: u64,
    /// The largest page header.
    pub max_page_header_bytes: u64,
    /// The largest span read at once when the source prefers aligned ranges.
    pub read_window_bytes: u64,
    /// Dictionary pages kept across fetches, in bytes (each fetch of a dictionary-encoded
    /// chunk needs its dictionary page; zero disables the cache).
    pub dictionary_cache_bytes: u64,
}

impl Default for ParquetLimits {
    fn default() -> Self {
        Self {
            max_footer_bytes: 32 << 20,
            max_page_index_bytes: 16 << 20,
            max_columns: 32_768,
            max_row_groups: 1 << 20,
            max_column_chunks: 1 << 20,
            max_schema_depth: 64,
            target_block_bytes: 1 << 20,
            min_block_rows: 16,
            max_block_rows: 1 << 18,
            max_block_bytes: 256 << 20,
            max_page_bytes: 64 << 20,
            max_page_header_bytes: 8 << 20,
            read_window_bytes: 8 << 20,
            dictionary_cache_bytes: 16 << 20,
        }
    }
}

/// Whether reads can use the offset index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageIndexStatus {
    /// Loaded: fetches read only the pages holding the requested rows.
    Loaded,
    /// The file has none.
    Absent,
    /// The file has one larger than [`ParquetLimits::max_page_index_bytes`].
    TooLarge,
    /// The file's index is inconsistent with its column chunks, so it is not used.
    Invalid,
}

/// One fetch: a row range inside one row group, for a set of columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchRequest {
    /// The row group.
    pub row_group: usize,
    /// Rows of the file (not of the row group); must lie inside the row group.
    pub rows: Range<u64>,
    /// Top-level columns. The batch holds them in ascending order, without duplicates.
    pub columns: Vec<usize>,
}

/// A Parquet file opened for lazy reading.
#[derive(Debug)]
pub struct ParquetSource<B: ByteSource> {
    shared: Arc<Shared<B>>,
    limits: ParquetLimits,
    arrow: ArrowReaderMetadata,
    schema: SchemaRef,
    layout: Layout,
    /// The leaf columns of each top-level column (contiguous in schema order).
    leaves: Vec<Range<usize>>,
    /// Each column chunk's byte range, indexed `group * leaves + leaf`.
    chunks: Vec<Range<u64>>,
    leaf_count: usize,
    page_index: PageIndexStatus,
    footer_bytes: u64,
}

fn thrift_error(error: ThriftError) -> OpenError {
    match error {
        ThriftError::Eof => OpenError::Malformed("the footer is truncated"),
        ThriftError::Invalid(m) => OpenError::Malformed(m),
        ThriftError::Limit(m) => OpenError::Limit(m),
    }
}

/// Runs `f`, turning a panic into `Err(None)` and an error into `Err(Some(e))`.
fn guarded<T, E>(f: impl FnOnce() -> Result<T, E>) -> Result<T, Option<E>> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(Some(error)),
        Err(_) => Err(None),
    }
}

fn nonnegative(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

/// A column chunk's byte range as the reader uses it, or `None` if it is invalid.
fn chunk_range(chunk: &ColumnChunkMetaData, data_end: u64) -> Option<Range<u64>> {
    let start = chunk
        .dictionary_page_offset()
        .unwrap_or_else(|| chunk.data_page_offset());
    let start = u64::try_from(start).ok()?;
    let len = u64::try_from(chunk.compressed_size()).ok()?;
    let end = start.checked_add(len)?;
    let data = u64::try_from(chunk.data_page_offset()).ok()?;
    (start >= 4 && end <= data_end && data >= start && (data < end || len == 0))
        .then_some(start..end)
}

/// The estimated decoded (Arrow) size of a column chunk.
fn estimate_chunk(chunk: &ColumnChunkMetaData) -> u64 {
    let uncompressed = nonnegative(chunk.uncompressed_size());
    let values = nonnegative(chunk.num_values());
    let descr = chunk.column_descr();
    let width: u64 = match descr.physical_type() {
        PhysicalType::BOOLEAN => 0,
        PhysicalType::INT32 | PhysicalType::FLOAT => 4,
        PhysicalType::INT64 | PhysicalType::DOUBLE | PhysicalType::INT96 => 8,
        PhysicalType::FIXED_LEN_BYTE_ARRAY => u64::try_from(descr.type_length()).unwrap_or(0),
        PhysicalType::BYTE_ARRAY => 4,
    };
    let mut decoded = values.saturating_mul(width);
    if descr.physical_type() == PhysicalType::BYTE_ARRAY {
        let data = chunk
            .unencoded_byte_array_data_bytes()
            .map_or(uncompressed, nonnegative);
        decoded = decoded.saturating_add(data);
    }
    // Validity bitmaps (and the booleans themselves).
    decoded = decoded.saturating_add(values.div_ceil(8));
    if descr.max_rep_level() > 0 {
        // List offsets.
        decoded = decoded.saturating_add(values.saturating_mul(4));
    }
    decoded.max(uncompressed)
}

impl<B: ByteSource> ParquetSource<B> {
    /// Opens a Parquet file: reads and checks the footer and, when it fits under the cap,
    /// the offset index. Nothing else is read.
    pub fn open(source: B, limits: &ParquetLimits) -> Result<Self, OpenError> {
        let shared = Arc::new(Shared::new(source, limits.dictionary_cache_bytes));
        let len = shared.len;
        // "PAR1", the footer length, "PAR1".
        if len < 12 {
            return Err(OpenError::NotParquet);
        }
        let trailer_start = len.saturating_sub(8);
        let trailer = shared.read(trailer_start..len)?;
        match trailer.get(4..8) {
            Some(b"PAR1") => {}
            Some(b"PARE") => return Err(OpenError::Encrypted),
            _ => return Err(OpenError::NotParquet),
        }
        let footer_len = trailer
            .get(..4)
            .and_then(|b| <[u8; 4]>::try_from(b).ok())
            .map(u32::from_le_bytes)
            .ok_or(OpenError::NotParquet)?;
        let footer_len = u64::from(footer_len);
        if footer_len > limits.max_footer_bytes {
            return Err(OpenError::Limit("the footer exceeds the size limit"));
        }
        let footer_start = trailer_start
            .checked_sub(footer_len)
            .filter(|&start| start >= 4)
            .ok_or(OpenError::Malformed("the footer length exceeds the file"))?;
        let footer = shared.read(footer_start..trailer_start)?;

        let footer_limits = FooterLimits {
            max_columns: limits.max_columns,
            max_schema_depth: limits.max_schema_depth,
            max_row_groups: limits.max_row_groups,
            max_column_chunks: limits.max_column_chunks,
        };
        let summary = thrift::scan_footer(&footer, &footer_limits).map_err(thrift_error)?;
        let metadata = guarded(|| ParquetMetaDataReader::decode_metadata(&footer))
            .map_err(|_| OpenError::Malformed("the footer could not be decoded"))?;
        drop(footer);

        let checked = guarded(|| Self::check_metadata(&metadata, &summary, footer_start))
            .map_err(|e| e.unwrap_or(OpenError::Malformed("the footer could not be decoded")))?;
        let (group_rows, chunks, leaves) = checked;
        let leaf_count = summary.leaves;

        let (metadata, page_index) = Self::load_offset_index(
            &shared,
            metadata,
            limits,
            footer_start,
            &chunks,
            &group_rows,
            leaf_count,
        )?;
        let metadata = Arc::new(metadata);
        let arrow = guarded(|| {
            ArrowReaderMetadata::try_new(Arc::clone(&metadata), ArrowReaderOptions::new())
        })
        .map_err(|_| OpenError::Unsupported("the schema cannot be read as Arrow"))?;
        let schema = Arc::clone(arrow.schema());
        if schema.fields().len() != leaves.len() {
            return Err(OpenError::Unsupported("the schema cannot be read as Arrow"));
        }

        let sizing = Sizing {
            target_block_bytes: limits.target_block_bytes,
            min_block_rows: limits.min_block_rows,
            max_block_rows: limits.max_block_rows,
        };
        let layout = Layout::new(&group_rows, leaves.len(), &sizing, |column, group| {
            let Some(range) = leaves.get(column) else {
                return 0;
            };
            let Some(row_group) = metadata.row_groups().get(group) else {
                return 0;
            };
            range
                .clone()
                .filter_map(|leaf| row_group.columns().get(leaf))
                .map(estimate_chunk)
                .fold(0u64, u64::saturating_add)
        })
        .ok_or(OpenError::Malformed("the row counts overflow"))?;

        Ok(Self {
            shared,
            limits: limits.clone(),
            arrow,
            schema,
            layout,
            leaves,
            chunks,
            leaf_count,
            page_index,
            footer_bytes: footer_len,
        })
    }

    /// Validates the decoded metadata. Returns the rows of each row group, every column
    /// chunk's range, and each top-level column's leaves.
    #[allow(clippy::type_complexity)]
    fn check_metadata(
        metadata: &ParquetMetaData,
        summary: &thrift::FooterSummary,
        data_end: u64,
    ) -> Result<(Vec<u64>, Vec<Range<u64>>, Vec<Range<usize>>), OpenError> {
        let schema = metadata.file_metadata().schema_descr();
        let leaf_count = schema.num_columns();
        if metadata.num_row_groups() != summary.row_groups || leaf_count != summary.leaves {
            return Err(OpenError::Malformed("the footer could not be decoded"));
        }
        if u32::try_from(metadata.num_row_groups()).is_err() {
            return Err(OpenError::Limit("too many row groups"));
        }
        // Leaves of each top-level column; they are contiguous in schema order.
        let roots = schema.root_schema().get_fields().len();
        let mut leaves: Vec<Range<usize>> = Vec::with_capacity(roots);
        for leaf in 0..leaf_count {
            let root = schema.get_column_root_idx(leaf);
            let next = leaf.saturating_add(1);
            if root.checked_add(1) == Some(leaves.len()) {
                if let Some(range) = leaves.last_mut() {
                    range.end = next;
                }
            } else if root == leaves.len() {
                leaves.push(leaf..next);
            } else {
                return Err(OpenError::Unsupported("a column without stored data"));
            }
        }
        if leaves.len() != roots {
            return Err(OpenError::Unsupported("a column without stored data"));
        }

        let mut group_rows = Vec::with_capacity(metadata.num_row_groups());
        let cells = metadata.num_row_groups().saturating_mul(leaf_count);
        let mut chunks = Vec::with_capacity(cells);
        let mut total = 0u64;
        for group in metadata.row_groups() {
            let rows = u64::try_from(group.num_rows())
                .map_err(|_| OpenError::Malformed("a row group has a negative row count"))?;
            total = total
                .checked_add(rows)
                .ok_or(OpenError::Malformed("the row counts overflow"))?;
            group_rows.push(rows);
            if group.columns().len() != leaf_count {
                return Err(OpenError::Malformed("a row group lacks columns"));
            }
            for chunk in group.columns() {
                if chunk.file_path().is_some() {
                    return Err(OpenError::Unsupported("column data in another file"));
                }
                if chunk.num_values() < 0 || chunk.uncompressed_size() < 0 {
                    return Err(OpenError::Malformed(
                        "a column chunk declares a negative size",
                    ));
                }
                let range = chunk_range(chunk, data_end)
                    .ok_or(OpenError::Malformed("a column chunk lies outside the data"))?;
                chunks.push(range);
            }
        }
        if u64::try_from(metadata.file_metadata().num_rows()).ok() != Some(total) {
            return Err(OpenError::Malformed(
                "the row count disagrees with the row groups",
            ));
        }
        Ok((group_rows, chunks, leaves))
    }

    /// Loads the offset index when it exists, fits under the cap, and lies before the
    /// footer; validates it against the column chunks.
    fn load_offset_index(
        shared: &Arc<Shared<B>>,
        metadata: ParquetMetaData,
        limits: &ParquetLimits,
        data_end: u64,
        chunks: &[Range<u64>],
        group_rows: &[u64],
        leaf_count: usize,
    ) -> Result<(ParquetMetaData, PageIndexStatus), OpenError> {
        let span = metadata
            .row_groups()
            .iter()
            .flat_map(|group| group.columns())
            .filter_map(ColumnChunkMetaData::offset_index_range)
            .reduce(|a, b| a.start.min(b.start)..a.end.max(b.end));
        let Some(span) = span else {
            return Ok((metadata, PageIndexStatus::Absent));
        };
        let bytes = span.end.saturating_sub(span.start);
        if bytes > limits.max_page_index_bytes {
            return Ok((metadata, PageIndexStatus::TooLarge));
        }
        if span.start < 4 || span.end > data_end {
            return Ok((metadata, PageIndexStatus::Invalid));
        }
        let adapter = Adapter::new(
            Arc::clone(shared),
            CancelToken::new(),
            ReadCaps {
                max_read: bytes,
                max_page_bytes: 0,
                max_header_bytes: 0,
                window_bytes: limits.read_window_bytes,
            },
            vec![Region {
                range: span,
                chunk: None,
            }],
        );
        let candidate = metadata.clone();
        let loaded = guarded(|| {
            let mut reader = ParquetMetaDataReader::new_with_metadata(candidate)
                .with_column_index_policy(PageIndexPolicy::Skip)
                .with_offset_index_policy(PageIndexPolicy::Optional);
            reader.read_page_indexes(&adapter)?;
            reader.finish()
        });
        let loaded = match loaded {
            Ok(loaded) => loaded,
            Err(_) => {
                // A read failure is the source's; anything else is a bad index.
                return match adapter.failure() {
                    Some(e) if e.kind() == FetchErrorKind::Retryable => {
                        Err(OpenError::Source(SourceError::retryable(e.message())))
                    }
                    Some(e) if e.kind() == FetchErrorKind::Permanent => {
                        Err(OpenError::Source(SourceError::permanent(e.message())))
                    }
                    _ => Ok((metadata, PageIndexStatus::Invalid)),
                };
            }
        };
        let valid = guarded(|| -> Result<bool, ()> {
            Ok(Self::offset_index_is_valid(
                &loaded, chunks, group_rows, leaf_count,
            ))
        })
        .unwrap_or(false);
        if valid {
            Ok((loaded, PageIndexStatus::Loaded))
        } else {
            Ok((metadata, PageIndexStatus::Invalid))
        }
    }

    /// Every page location lies in its chunk, in order, without overlap, and the first
    /// rows start at zero and increase within the row group.
    fn offset_index_is_valid(
        metadata: &ParquetMetaData,
        chunks: &[Range<u64>],
        group_rows: &[u64],
        leaf_count: usize,
    ) -> bool {
        let Some(index) = metadata.page_index() else {
            return false;
        };
        for (group, &rows) in group_rows.iter().enumerate() {
            for leaf in 0..leaf_count {
                let Some(offsets) = index.offset_index(group, leaf) else {
                    continue;
                };
                let Some(chunk) = group
                    .checked_mul(leaf_count)
                    .and_then(|i| i.checked_add(leaf))
                    .and_then(|i| chunks.get(i))
                else {
                    return false;
                };
                let locations = offsets.page_locations();
                if locations.is_empty() && rows > 0 {
                    return false;
                }
                let mut next_offset = chunk.start;
                let mut previous_row: Option<u64> = None;
                for location in locations {
                    let (Ok(offset), Ok(size), Ok(first_row)) = (
                        u64::try_from(location.offset),
                        u64::try_from(location.compressed_page_size),
                        u64::try_from(location.first_row_index),
                    ) else {
                        return false;
                    };
                    let Some(end) = offset.checked_add(size) else {
                        return false;
                    };
                    if size == 0 || offset < next_offset || end > chunk.end || first_row >= rows {
                        return false;
                    }
                    match previous_row {
                        None if first_row != 0 => return false,
                        Some(previous) if first_row <= previous => return false,
                        _ => {}
                    }
                    next_offset = end;
                    previous_row = Some(first_row);
                }
            }
        }
        true
    }

    /// The decoded file metadata (with the offset index when it is loaded).
    pub fn metadata(&self) -> &Arc<ParquetMetaData> {
        self.arrow.metadata()
    }

    /// Whether fetches can use the offset index.
    pub fn page_index(&self) -> PageIndexStatus {
        self.page_index
    }

    /// The footer's size in bytes.
    pub fn footer_bytes(&self) -> u64 {
        self.footer_bytes
    }

    /// The limits the file was opened with.
    pub fn limits(&self) -> &ParquetLimits {
        &self.limits
    }

    /// The byte source.
    pub fn source(&self) -> &B {
        &self.shared.source
    }

    /// Bytes read from the source so far, by every open and fetch of this file.
    pub fn bytes_read(&self) -> u64 {
        self.shared.bytes_read.load(Ordering::Relaxed)
    }

    /// Source reads so far.
    pub fn reads(&self) -> u64 {
        self.shared.reads.load(Ordering::Relaxed)
    }

    /// Number of row groups.
    pub fn row_groups(&self) -> usize {
        self.layout.groups()
    }

    /// The rows of a row group (empty if it does not exist).
    pub fn row_group_rows(&self, group: usize) -> Range<u64> {
        self.layout.group_rows(group)
    }

    /// The leaf (stored) columns that make up a top-level column.
    pub fn leaf_columns(&self, column: usize) -> Range<usize> {
        self.leaves.get(column).cloned().unwrap_or(0..0)
    }

    /// The byte range of a leaf column's chunk in a row group.
    pub fn column_chunk_range(&self, group: usize, leaf: usize) -> Option<Range<u64>> {
        if leaf >= self.leaf_count {
            return None;
        }
        let index = group.checked_mul(self.leaf_count)?.checked_add(leaf)?;
        self.chunks.get(index).cloned()
    }

    /// All blocks of a column, in row order.
    pub fn blocks(&self, column: usize) -> impl Iterator<Item = BlockId> + '_ {
        let mut row = 0u64;
        core::iter::from_fn(move || {
            let block = self.layout.block_of(column, row)?;
            row = self.layout.block_rows(block).end;
            Some(block)
        })
    }

    /// The block of `column` holding `row`.
    pub fn block_of(&self, column: usize, row: u64) -> Option<BlockId> {
        self.layout.block_of(column, row)
    }

    /// The rows of a block.
    pub fn block_rows(&self, block: BlockId) -> Range<u64> {
        self.layout.block_rows(block)
    }

    /// The estimated decoded size of a request.
    pub fn estimate(&self, request: &FetchRequest) -> u64 {
        let rows = request.rows.end.saturating_sub(request.rows.start);
        request
            .columns
            .iter()
            .map(|&column| self.layout.estimate_rows(column, request.row_group, rows))
            .fold(0u64, u64::saturating_add)
    }

    /// Fetches one row range of one row group for a set of columns.
    ///
    /// With the offset index loaded, only the pages covering the rows are read (plus each
    /// chunk's dictionary page). Without it, every page header of each requested chunk up
    /// to the last requested row is read and the values before the first requested row are
    /// decoded and discarded. Returns a batch with the requested columns in ascending
    /// order.
    pub fn fetch(
        &self,
        request: &FetchRequest,
        cancel: &CancelToken,
    ) -> Result<RecordBatch, FetchError> {
        if cancel.is_cancelled() {
            return Err(FetchError::cancelled());
        }
        let group = request.row_group;
        let group_range = self.layout.group_rows(group);
        if group >= self.layout.groups()
            || request.rows.start < group_range.start
            || request.rows.end > group_range.end
            || request.rows.start >= request.rows.end
        {
            return Err(FetchError::permanent("the requested rows are out of range"));
        }
        let mut columns = request.columns.clone();
        columns.sort_unstable();
        columns.dedup();
        if columns.is_empty() || columns.iter().any(|&c| c >= self.leaves.len()) {
            return Err(FetchError::permanent(
                "the requested columns are out of range",
            ));
        }
        let normalized = FetchRequest {
            row_group: group,
            rows: request.rows.clone(),
            columns,
        };
        if self.estimate(&normalized) > self.limits.max_block_bytes {
            return Err(FetchError::limit(
                "the block exceeds the decoded size limit",
            ));
        }
        let too_large = || FetchError::limit("the row group is too large for this platform");
        let skip = usize::try_from(normalized.rows.start.saturating_sub(group_range.start))
            .map_err(|_| too_large())?;
        let select = usize::try_from(normalized.rows.end.saturating_sub(normalized.rows.start))
            .map_err(|_| too_large())?;
        let rest = usize::try_from(group_range.end.saturating_sub(normalized.rows.end))
            .map_err(|_| too_large())?;

        let regions = self.regions(group, &normalized.columns)?;
        let adapter = Adapter::new(
            Arc::clone(&self.shared),
            cancel.clone(),
            ReadCaps {
                max_read: self
                    .limits
                    .max_page_bytes
                    .saturating_add(self.limits.max_page_header_bytes),
                max_page_bytes: self.limits.max_page_bytes,
                max_header_bytes: self.limits.max_page_header_bytes,
                window_bytes: self.limits.read_window_bytes,
            },
            regions,
        );
        let metadata = self.arrow.metadata();
        let schema = metadata.file_metadata().schema_descr();
        let decoded = guarded(|| -> Result<Vec<RecordBatch>, ()> {
            let mask = ProjectionMask::roots(schema, normalized.columns.iter().copied());
            let selectors: Vec<RowSelector> = [
                RowSelector::skip(skip),
                RowSelector::select(select),
                RowSelector::skip(rest),
            ]
            .into_iter()
            .filter(|s| s.row_count > 0)
            .collect();
            let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
                adapter.clone(),
                self.arrow.clone(),
            )
            .with_projection(mask)
            .with_row_groups(vec![group])
            .with_row_selection(RowSelection::from(selectors))
            .with_row_selection_policy(RowSelectionPolicy::Selectors)
            .with_batch_size(select)
            .build()
            .map_err(|_| ())?;
            let mut batches = Vec::with_capacity(1);
            for batch in reader {
                batches.push(batch.map_err(|_| ())?);
            }
            Ok(batches)
        });
        let batches = match decoded {
            Ok(batches) => batches,
            Err(_) => return Err(adapter.failure().unwrap_or_else(FetchError::undecodable)),
        };
        let batch = match batches.len() {
            1 => batches
                .into_iter()
                .next()
                .ok_or_else(FetchError::undecodable)?,
            _ => {
                let schema = batches
                    .first()
                    .map(RecordBatch::schema)
                    .ok_or_else(FetchError::undecodable)?;
                guarded(|| arrow_select::concat::concat_batches(&schema, &batches))
                    .map_err(|_| FetchError::undecodable())?
            }
        };
        if batch.num_rows() != select || batch.num_columns() != normalized.columns.len() {
            return Err(FetchError::permanent(
                "the decoded rows do not match the column chunk",
            ));
        }
        Ok(batch)
    }

    /// The regions a fetch of `columns` in `group` may read, with their page guards.
    fn regions(&self, group: usize, columns: &[usize]) -> Result<Vec<Region>, FetchError> {
        let metadata = self.arrow.metadata();
        let row_group = metadata
            .row_groups()
            .get(group)
            .ok_or(FetchError::permanent("the requested rows are out of range"))?;
        let group_rows = {
            let rows = self.layout.group_rows(group);
            rows.end.saturating_sub(rows.start)
        };
        let index = metadata.page_index();
        let mut regions = Vec::new();
        for &column in columns {
            for leaf in self.leaf_columns(column) {
                let chunk = row_group.columns().get(leaf).ok_or(FetchError::permanent(
                    "the requested columns are out of range",
                ))?;
                let range = self
                    .column_chunk_range(group, leaf)
                    .ok_or(FetchError::permanent(
                        "the requested columns are out of range",
                    ))?;
                let locations = index.and_then(|index| index.page_locations(group, leaf));
                // The crate reads the bytes before the first page as a dictionary page.
                let dictionary_first = locations.is_some_and(|l| {
                    l.first().and_then(|p| u64::try_from(p.offset).ok()) != Some(range.start)
                });
                let page_starts = locations.map(|locations| {
                    let mut starts: Vec<u64> =
                        Vec::with_capacity(locations.len().saturating_add(1));
                    if dictionary_first {
                        starts.push(range.start);
                    }
                    starts.extend(
                        locations
                            .iter()
                            .filter_map(|l| u64::try_from(l.offset).ok()),
                    );
                    starts
                });
                regions.push(Region {
                    range,
                    chunk: Some(ChunkGuard {
                        codec: chunk.compression(),
                        flat: chunk.column_descr().max_rep_level() == 0,
                        group_rows,
                        page_starts,
                        dictionary_first,
                    }),
                });
            }
        }
        regions.sort_by_key(|r| r.range.start);
        let overlapping = regions
            .windows(2)
            .any(|w| matches!(w, [a, b] if b.range.start < a.range.end));
        if overlapping {
            return Err(FetchError::permanent("column chunks overlap"));
        }
        Ok(regions)
    }
}

impl<B: ByteSource> TableSource for ParquetSource<B> {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn bytes_read(&self) -> Option<u64> {
        Some(Self::bytes_read(self))
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.source().len())
    }

    fn row_count(&self) -> u64 {
        self.layout.rows()
    }

    fn block_of(&self, column: usize, row: u64) -> Option<BlockId> {
        self.layout.block_of(column, row)
    }

    fn block_rows(&self, block: BlockId) -> Range<u64> {
        self.layout.block_rows(block)
    }

    fn estimated_bytes(&self, block: BlockId) -> u64 {
        self.layout.estimated_bytes(block)
    }

    fn fetch(&self, blocks: &[BlockId], cancel: &CancelToken) -> Result<Vec<ArrayRef>, FetchError> {
        let first = blocks
            .first()
            .ok_or(FetchError::permanent("no blocks were requested"))?;
        let rows = self.layout.block_rows(*first);
        if rows.is_empty()
            || blocks
                .iter()
                .any(|b| b.group != first.group || self.layout.block_rows(*b) != rows)
        {
            return Err(FetchError::permanent("the blocks cover different rows"));
        }
        let request = FetchRequest {
            row_group: usize::try_from(first.group)
                .map_err(|_| FetchError::permanent("the requested rows are out of range"))?,
            rows,
            columns: blocks.iter().map(|b| b.column).collect(),
        };
        let batch = ParquetSource::fetch(self, &request, cancel)?;
        let mut columns: Vec<usize> = request.columns;
        columns.sort_unstable();
        columns.dedup();
        blocks
            .iter()
            .map(|b| {
                let position = columns
                    .binary_search(&b.column)
                    .map_err(|_| FetchError::undecodable())?;
                batch
                    .columns()
                    .get(position)
                    .cloned()
                    .ok_or_else(FetchError::undecodable)
            })
            .collect()
    }

    fn column_info(&self, column: usize) -> ColumnInfo {
        info::column_info(self, column)
    }
}
