//! An engine for virtualized table views over Parquet data, loading lazily in both
//! directions: only the columns in view and only the rows in view (plus a prefetch ring)
//! are read and decoded. It knows nothing about any UI toolkit; a widget drives it.
//!
//! The layers, bottom up:
//!
//! * [`source`]: [`ByteSource`], positional reads from wherever the file lives, with
//!   [`MemorySource`], `FileSource` (native targets), and [`CountingSource`] for
//!   diagnostics. Errors say whether a read may be retried.
//! * [`parquet`]: [`ParquetSource`], which opens a file by its footer (checking it against
//!   [`ParquetLimits`] before anything large is allocated: files may be hostile), divides
//!   each column into row blocks, and decodes one row range of several columns per
//!   [`fetch`](ParquetSource::fetch). Panics in the decoder become errors.
//! * [`table`]: [`TableSource`], the format-independent interface the engine and the widget
//!   use.
//! * [`engine`]: [`LazyTable`], owned by the UI thread, which never blocks: it decides from
//!   the scroll motion what to fetch, runs fetches through a caller-provided [`Spawn`],
//!   collects results from a lock-free queue, and caches decoded blocks under a byte budget.
//! * [`format`](mod@format): bounded, one-line cell text.
//!
//! Nothing here starts a thread, takes a lock on the UI thread, or reads a clock: the
//! caller passes the time to [`LazyTable::update`]. The crate builds for
//! `wasm32-unknown-unknown`.

pub mod cancel;
pub mod engine;
pub mod error;
pub mod format;
pub mod parquet;
pub mod source;
pub mod table;

pub use crate::cancel::CancelToken;
pub use crate::engine::{Cell, LazyOptions, LazyTable, Spawn, Stats, Viewport};
pub use crate::error::{FetchError, FetchErrorKind, OpenError};
pub use crate::format::{CellText, format_cell};
pub use crate::parquet::{FetchRequest, PageIndexStatus, ParquetLimits, ParquetSource};
#[cfg(not(target_arch = "wasm32"))]
pub use crate::source::FileSource;
pub use crate::source::{ByteSource, CountingSource, MemorySource, SourceError, SourceErrorKind};
pub use crate::table::{BlockId, ColumnInfo, TableSource};

/// Re-exports of the Arrow crates in the public API, so callers use matching versions.
pub mod arrow {
    pub use arrow_array as array;
    pub use arrow_schema as schema;
}
