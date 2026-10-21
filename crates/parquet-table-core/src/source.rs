//! The byte level: positional reads from wherever the file lives.
//!
//! A [`ByteSource`] is the only thing an integrator implements to read a new kind of storage.
//! Reads are positional and go through `&self`, so one source serves many concurrent fetches.
//! Errors carry a [`SourceErrorKind`] that decides what the engine does next: a
//! [`Retryable`](SourceErrorKind::Retryable) failure is tried again after a backoff, a
//! [`Permanent`](SourceErrorKind::Permanent) one (an integrity failure, for example) never is.

use core::fmt;
use core::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

/// Positional, concurrent read access to the bytes of one file.
///
/// Implementations must be cheap to call from several worker threads at once. Every read
/// is exact: either the whole buffer is filled or an error is returned.
pub trait ByteSource: Send + Sync + 'static {
    /// The length of the file in bytes.
    fn len(&self) -> u64;

    /// Whether the file is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fills `buf` with the bytes at `offset..offset + buf.len()`.
    ///
    /// Concurrent positional reads through `&self`. On error the buffer's contents are
    /// unspecified.
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError>;

    /// Optional ranges the source prefers reads to be aligned to (for example the records
    /// of an encrypted store, which are verified as a whole).
    ///
    /// Called once when a file is opened. When present, reads are widened to the
    /// preferred ranges they touch (up to a bound) and the extra bytes are kept for the
    /// rest of the fetch, so neighbouring reads in one record cost one source read. The
    /// ranges must be sorted, non-empty, non-overlapping, and inside the file; otherwise
    /// they are ignored. Default `None`.
    fn preferred_ranges(&self) -> Option<Vec<Range<u64>>> {
        None
    }
}

/// What the engine should do about a failed read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceErrorKind {
    /// A transient failure (a network hiccup): the read may be tried again later.
    Retryable,
    /// The bytes cannot be read, now or later (missing, truncated, failed verification).
    /// Never retried.
    Permanent,
    /// The read was abandoned because its job was cancelled.
    Cancelled,
}

/// A failed read: its kind and a static description.
///
/// The message is `&'static str` on purpose: source errors end up in the UI, and a static
/// message cannot carry a path, a URL, or file contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceError {
    kind: SourceErrorKind,
    message: &'static str,
}

impl SourceError {
    /// A new error of the given kind.
    pub const fn new(kind: SourceErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }

    /// A transient failure.
    pub const fn retryable(message: &'static str) -> Self {
        Self::new(SourceErrorKind::Retryable, message)
    }

    /// A failure that must not be retried.
    pub const fn permanent(message: &'static str) -> Self {
        Self::new(SourceErrorKind::Permanent, message)
    }

    /// A read abandoned by cancellation.
    pub const fn cancelled() -> Self {
        Self::new(SourceErrorKind::Cancelled, "the read was cancelled")
    }

    /// What kind of failure this is.
    pub const fn kind(&self) -> SourceErrorKind {
        self.kind
    }

    /// The static description.
    pub const fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for SourceError {}

/// Returns the range `offset..offset + len` as `usize` indices into a buffer of `total`
/// bytes, or `None` when it does not fit.
fn checked_span(offset: u64, len: usize, total: usize) -> Option<Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(len)?;
    (end <= total).then_some(start..end)
}

/// A file held in memory. Cloning shares the bytes.
#[derive(Debug, Clone, Default)]
pub struct MemorySource {
    bytes: Bytes,
}

impl MemorySource {
    /// A source over `bytes`.
    pub fn new(bytes: impl Into<Bytes>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    /// The bytes.
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }
}

impl ByteSource for MemorySource {
    fn len(&self) -> u64 {
        // usize -> u64 never loses information on supported targets.
        u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        let span = checked_span(offset, buf.len(), self.bytes.len())
            .ok_or(SourceError::permanent("read past the end of the file"))?;
        let src = self
            .bytes
            .get(span)
            .ok_or(SourceError::permanent("read past the end of the file"))?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use file::FileSource;

#[cfg(not(target_arch = "wasm32"))]
mod file {
    use super::{ByteSource, SourceError};
    use std::fs::File;
    use std::io;
    use std::path::Path;

    /// A local file read with positional reads (`pread` on Unix, `ReadFile` with an
    /// offset on Windows). The length is taken once, when the source is created.
    #[derive(Debug)]
    pub struct FileSource {
        file: File,
        len: u64,
    }

    impl FileSource {
        /// Opens `path` for reading.
        pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
            Self::new(File::open(path)?)
        }

        /// A source over an open file.
        pub fn new(file: File) -> io::Result<Self> {
            let len = file.metadata()?.len();
            Ok(Self { file, len })
        }
    }

    fn map_io(error: &io::Error) -> SourceError {
        match error.kind() {
            io::ErrorKind::UnexpectedEof => SourceError::permanent("read past the end of the file"),
            io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
                SourceError::retryable("the read was interrupted")
            }
            _ => SourceError::permanent("the file could not be read"),
        }
    }

    impl ByteSource for FileSource {
        fn len(&self) -> u64 {
            self.len
        }

        fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
            let len = u64::try_from(buf.len())
                .map_err(|_| SourceError::permanent("read past the end of the file"))?;
            match offset.checked_add(len) {
                Some(end) if end <= self.len => {}
                _ => return Err(SourceError::permanent("read past the end of the file")),
            }
            read_at(&self.file, offset, buf).map_err(|e| map_io(&e))
        }
    }

    #[cfg(unix)]
    fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }

    #[cfg(windows)]
    fn read_at(file: &File, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match file.seek_read(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    let rest = buf
                        .get_mut(n..)
                        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
                    buf = rest;
                    offset = offset
                        .checked_add(u64::try_from(n).map_err(|_| io::ErrorKind::InvalidData)?)
                        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    #[cfg(not(any(unix, windows)))]
    fn read_at(_file: &File, _offset: u64, _buf: &mut [u8]) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// Wraps a source and counts what is read through it: a diagnostics helper for tests,
/// benchmarks, and debugging views.
///
/// Counting is lock-free. Recording the individual ranges (off by default) takes a lock
/// on every read: only use it where reads come from workers, never the UI thread.
#[derive(Debug)]
pub struct CountingSource<B> {
    inner: B,
    bytes: AtomicU64,
    calls: AtomicU64,
    ranges: Option<Mutex<Vec<Range<u64>>>>,
}

impl<B: ByteSource> CountingSource<B> {
    /// Counts bytes and calls.
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            bytes: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            ranges: None,
        }
    }

    /// Counts bytes and calls and records every range read.
    pub fn recording(inner: B) -> Self {
        Self {
            ranges: Some(Mutex::new(Vec::new())),
            ..Self::new(inner)
        }
    }

    /// Bytes read so far.
    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Successful and failed read calls so far.
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    /// The ranges read so far, in order (empty unless created with [`Self::recording`]).
    pub fn ranges(&self) -> Vec<Range<u64>> {
        self.ranges
            .as_ref()
            .and_then(|r| r.lock().ok().map(|r| r.clone()))
            .unwrap_or_default()
    }

    /// Resets the counters and the recorded ranges.
    pub fn reset(&self) {
        self.bytes.store(0, Ordering::Relaxed);
        self.calls.store(0, Ordering::Relaxed);
        if let Some(Ok(mut ranges)) = self.ranges.as_ref().map(Mutex::lock) {
            ranges.clear();
        }
    }

    /// The wrapped source.
    pub fn inner(&self) -> &B {
        &self.inner
    }
}

impl<B: ByteSource> ByteSource for CountingSource<B> {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(len, Ordering::Relaxed);
        if let Some(Ok(mut ranges)) = self.ranges.as_ref().map(Mutex::lock) {
            ranges.push(offset..offset.saturating_add(len));
        }
        self.inner.read_exact_at(offset, buf)
    }

    fn preferred_ranges(&self) -> Option<Vec<Range<u64>>> {
        self.inner.preferred_ranges()
    }
}

impl<B: ByteSource> ByteSource for std::sync::Arc<B> {
    fn len(&self) -> u64 {
        (**self).len()
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        (**self).read_exact_at(offset, buf)
    }

    fn preferred_ranges(&self) -> Option<Vec<Range<u64>>> {
        (**self).preferred_ranges()
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    #[test]
    fn memory_reads_are_exact() {
        let source = MemorySource::new(vec![1u8, 2, 3, 4, 5]);
        let mut buf = [0u8; 2];
        source.read_exact_at(3, &mut buf).unwrap();
        assert_eq!(buf, [4, 5]);
        assert_eq!(
            source.read_exact_at(4, &mut buf).unwrap_err().kind(),
            SourceErrorKind::Permanent
        );
        assert!(source.read_exact_at(u64::MAX, &mut buf).is_err());
    }

    #[test]
    fn counting_counts() {
        let source = CountingSource::recording(MemorySource::new(vec![0u8; 100]));
        let mut buf = [0u8; 10];
        source.read_exact_at(5, &mut buf).unwrap();
        source.read_exact_at(50, &mut buf).unwrap();
        assert_eq!(source.bytes(), 20);
        assert_eq!(source.calls(), 2);
        assert_eq!(source.ranges(), vec![5..15, 50..60]);
        source.reset();
        assert_eq!(source.bytes(), 0);
        assert!(source.ranges().is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn file_reads_are_positional() {
        let path = std::env::temp_dir().join(format!(
            "parquet-table-core-source-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, (0u8..=255).collect::<Vec<_>>()).unwrap();
        let source = FileSource::open(&path).unwrap();
        let mut buf = [0u8; 4];
        source.read_exact_at(250, &mut buf).unwrap();
        assert_eq!(buf, [250, 251, 252, 253]);
        assert!(source.read_exact_at(253, &mut buf).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
