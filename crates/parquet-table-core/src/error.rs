//! Errors of opening a file and of fetching a block.
//!
//! Every message is static text: errors are shown in the UI, so they never carry a path, a
//! URL, or bytes of the file (a library error message could quote a malformed value).

use core::fmt;

use crate::source::{SourceError, SourceErrorKind};

/// What the engine does with a failed fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FetchErrorKind {
    /// Transient: the engine tries again after a backoff (a bounded number of times).
    Retryable,
    /// The block cannot be produced (malformed data, failed verification). Never retried.
    Permanent,
    /// The fetch was cancelled; the engine drops it without showing anything.
    Cancelled,
    /// The block exceeds a configured limit (its estimated or declared size). Not retried.
    Limit,
}

/// A failed fetch: its kind and a static description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchError {
    kind: FetchErrorKind,
    message: &'static str,
}

impl FetchError {
    /// A new error.
    pub const fn new(kind: FetchErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }

    /// A transient failure.
    pub const fn retryable(message: &'static str) -> Self {
        Self::new(FetchErrorKind::Retryable, message)
    }

    /// A failure that must not be retried.
    pub const fn permanent(message: &'static str) -> Self {
        Self::new(FetchErrorKind::Permanent, message)
    }

    /// A cancelled fetch.
    pub const fn cancelled() -> Self {
        Self::new(FetchErrorKind::Cancelled, "the fetch was cancelled")
    }

    /// A refusal by a configured limit.
    pub const fn limit(message: &'static str) -> Self {
        Self::new(FetchErrorKind::Limit, message)
    }

    /// The error a decoder failure or panic becomes.
    pub(crate) const fn undecodable() -> Self {
        Self::permanent("the column data could not be decoded")
    }

    /// What kind of failure this is.
    pub const fn kind(&self) -> FetchErrorKind {
        self.kind
    }

    /// The static description.
    pub const fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for FetchError {}

impl From<SourceError> for FetchError {
    fn from(error: SourceError) -> Self {
        let kind = match error.kind() {
            SourceErrorKind::Retryable => FetchErrorKind::Retryable,
            SourceErrorKind::Permanent => FetchErrorKind::Permanent,
            SourceErrorKind::Cancelled => FetchErrorKind::Cancelled,
        };
        Self::new(kind, error.message())
    }
}

/// Why a file could not be opened.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OpenError {
    /// Reading the file failed.
    #[error("the file could not be read: {0}")]
    Source(SourceError),
    /// The file does not end with the Parquet magic.
    #[error("the file is not a Parquet file")]
    NotParquet,
    /// The file uses Parquet modular encryption, which is not supported.
    #[error("encrypted Parquet files are not supported")]
    Encrypted,
    /// The file's structure is invalid.
    #[error("the file is malformed: {0}")]
    Malformed(&'static str),
    /// The file exceeds a configured limit.
    #[error("the file exceeds a limit: {0}")]
    Limit(&'static str),
    /// The file uses something this build cannot read.
    #[error("the file is not supported: {0}")]
    Unsupported(&'static str),
}

impl OpenError {
    /// Whether opening again later may succeed (the source reported a transient failure).
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Source(e) if e.kind() == SourceErrorKind::Retryable)
    }
}

impl From<SourceError> for OpenError {
    fn from(error: SourceError) -> Self {
        Self::Source(error)
    }
}
