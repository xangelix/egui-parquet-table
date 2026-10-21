//! The bridge from [`ByteSource`] to the `parquet` crate's `ChunkReader`, with the guards
//! that keep a hostile file's declared sizes from driving allocations.
//!
//! An [`Adapter`] serves one fetch (or the page-index load). It only reads inside the
//! regions it was planned for (the requested column chunks), checks the cancel token
//! before every read, and records the first failure so the fetch can report its real kind
//! (`parquet` turns every error into text on the way out).
//!
//! Page guards: the crate reads page headers through `get_read` (without a page index) or
//! as the start of a `get_bytes` page read (with one). Each header is parsed here first
//! and refused when its sizes exceed the limits or contradict its column chunk, and the
//! pages of codecs whose decoders read to the end are trial-decompressed with a bound (see
//! [`super::expansion`]).

use core::ops::Range;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use bytes::{Buf, Bytes};
use parquet::basic::Compression;
use parquet::errors::ParquetError;
use parquet::file::reader::{ChunkReader, Length};

use super::expansion::{self, ExpansionError};
use super::thrift::{self, HeaderError, PageHeaderInfo};
use crate::cancel::CancelToken;
use crate::error::FetchError;
use crate::source::{ByteSource, SourceError};

/// A file and its counters, shared by every fetch.
#[derive(Debug)]
pub(crate) struct Shared<B> {
    pub source: B,
    pub len: u64,
    /// Validated preferred read ranges (sorted, disjoint, inside the file).
    pub preferred: Option<Vec<Range<u64>>>,
    pub bytes_read: AtomicU64,
    pub reads: AtomicU64,
    /// Dictionary pages kept across fetches (read by workers only).
    dictionaries: Mutex<Dictionaries>,
}

/// Recently read dictionary pages, by file offset, oldest first, within a byte budget.
/// Every fetch of a dictionary-encoded chunk starts with its dictionary page, which can be
/// far larger than the data pages a block needs.
#[derive(Debug, Default)]
struct Dictionaries {
    pages: VecDeque<(u64, Bytes)>,
    bytes: u64,
    budget: u64,
}

impl<B: ByteSource> Shared<B> {
    pub(crate) fn new(source: B, dictionary_cache_bytes: u64) -> Self {
        let len = source.len();
        let preferred = source
            .preferred_ranges()
            .filter(|ranges| valid_preferred(ranges, len));
        Self {
            source,
            len,
            preferred,
            bytes_read: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            dictionaries: Mutex::new(Dictionaries {
                budget: dictionary_cache_bytes,
                ..Dictionaries::default()
            }),
        }
    }

    /// A dictionary page read before at `start` with `len` bytes (already checked).
    fn cached_dictionary(&self, start: u64, len: u64) -> Option<Bytes> {
        let cache = self.dictionaries.lock().ok()?;
        cache
            .pages
            .iter()
            .find(|(at, bytes)| *at == start && u64::try_from(bytes.len()).ok() == Some(len))
            .map(|(_, bytes)| bytes.clone())
    }

    /// Keeps a checked dictionary page.
    fn keep_dictionary(&self, start: u64, bytes: &Bytes) {
        let Ok(mut cache) = self.dictionaries.lock() else {
            return;
        };
        let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if len > cache.budget || cache.pages.iter().any(|(at, _)| *at == start) {
            return;
        }
        cache.bytes = cache.bytes.saturating_add(len);
        // A copy: `bytes` may be a slice of a larger read window.
        cache
            .pages
            .push_back((start, Bytes::copy_from_slice(bytes)));
        while cache.bytes > cache.budget {
            match cache.pages.pop_front() {
                Some((_, old)) => {
                    let old = u64::try_from(old.len()).unwrap_or(0);
                    cache.bytes = cache.bytes.saturating_sub(old);
                }
                None => break,
            }
        }
    }

    /// Reads exactly `range` (whose length the caller has bounded).
    pub(crate) fn read(&self, range: Range<u64>) -> Result<Bytes, SourceError> {
        if range.end > self.len || range.start > range.end {
            return Err(SourceError::permanent("read past the end of the file"));
        }
        let len = usize::try_from(range.end.saturating_sub(range.start))
            .map_err(|_| SourceError::permanent("a read is too large for this platform"))?;
        let mut buf = vec![0u8; len];
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(range.end.saturating_sub(range.start), Ordering::Relaxed);
        self.source.read_exact_at(range.start, &mut buf)?;
        Ok(Bytes::from(buf))
    }
}

fn valid_preferred(ranges: &[Range<u64>], len: u64) -> bool {
    let mut previous_end = 0u64;
    for range in ranges {
        if range.start >= range.end || range.start < previous_end || range.end > len {
            return false;
        }
        previous_end = range.end;
    }
    true
}

/// Byte caps of one adapter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReadCaps {
    /// The largest single `get_bytes`.
    pub max_read: u64,
    /// The largest page (compressed or uncompressed) a header may declare.
    pub max_page_bytes: u64,
    /// The largest page header.
    pub max_header_bytes: u64,
    /// The largest preferred-range window kept for reuse.
    pub window_bytes: u64,
}

/// What a column chunk's guard needs.
#[derive(Debug, Clone)]
pub(crate) struct ChunkGuard {
    pub codec: Compression,
    /// No repetition: a data page holds at most one value per row.
    pub flat: bool,
    /// Rows in the row group.
    pub group_rows: u64,
    /// With an offset index: the sorted starts of every page the crate reads whole
    /// (including a dictionary page before the first data page).
    pub page_starts: Option<Vec<u64>>,
    /// With an offset index: the chunk starts with a dictionary page.
    pub dictionary_first: bool,
}

/// A region reads may touch.
#[derive(Debug, Clone)]
pub(crate) struct Region {
    pub range: Range<u64>,
    /// `None` for page-index bytes (no page checks).
    pub chunk: Option<ChunkGuard>,
}

/// A page header read through `get_read` whose data has not been read yet.
#[derive(Debug, Clone, Copy)]
struct Pending {
    data_start: u64,
    compressed: u64,
    /// Bytes at the start of the data that are not compressed (v2 levels).
    levels: usize,
    /// Expand-check the data against this many bytes, or `None` for no check.
    limit: Option<usize>,
    /// The page is a dictionary page.
    dictionary: bool,
}

#[derive(Debug, Default)]
struct State {
    /// Preferred-range windows kept for this fetch, oldest first.
    windows: VecDeque<(u64, Bytes)>,
    window_total: u64,
    /// One pending header per region.
    pending: Vec<Option<Pending>>,
}

#[derive(Debug)]
struct Inner<B> {
    shared: Arc<Shared<B>>,
    cancel: CancelToken,
    caps: ReadCaps,
    regions: Vec<Region>,
    state: Mutex<State>,
    failure: OnceLock<FetchError>,
}

/// A `ChunkReader` for one fetch. Cloning shares the fetch's state.
#[derive(Debug)]
pub(crate) struct Adapter<B>(Arc<Inner<B>>);

impl<B> Clone for Adapter<B> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

fn malformed(message: &'static str) -> FetchError {
    FetchError::permanent(message)
}

impl<B: ByteSource> Adapter<B> {
    /// An adapter that may read only inside `regions` (sorted by start, disjoint).
    pub(crate) fn new(
        shared: Arc<Shared<B>>,
        cancel: CancelToken,
        caps: ReadCaps,
        regions: Vec<Region>,
    ) -> Self {
        let pending = vec![None; regions.len()];
        Self(Arc::new(Inner {
            shared,
            cancel,
            caps,
            regions,
            state: Mutex::new(State {
                pending,
                ..State::default()
            }),
            failure: OnceLock::new(),
        }))
    }

    /// The first failure recorded by a read, if any.
    pub(crate) fn failure(&self) -> Option<FetchError> {
        self.0.failure.get().cloned()
    }

    fn fail(&self, error: FetchError) -> ParquetError {
        let message = error.message();
        let _ = self.0.failure.set(error);
        ParquetError::General(message.to_owned())
    }

    fn check_cancel(&self) -> Result<(), FetchError> {
        if self.0.cancel.is_cancelled() {
            Err(FetchError::cancelled())
        } else {
            Ok(())
        }
    }

    /// The region that wholly contains `range`.
    fn region(&self, range: &Range<u64>) -> Result<(usize, &Region), FetchError> {
        let regions = &self.0.regions;
        let index = regions
            .partition_point(|r| r.range.start <= range.start)
            .checked_sub(1)
            .ok_or(malformed("a read outside the planned column chunks"))?;
        let region = regions
            .get(index)
            .ok_or(malformed("a read outside the planned column chunks"))?;
        if range.start < region.range.start || range.end > region.range.end {
            return Err(malformed("a read outside the planned column chunks"));
        }
        Ok((index, region))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>, FetchError> {
        self.0
            .state
            .lock()
            .map_err(|_| FetchError::permanent("an earlier read failed"))
    }

    /// Reads `range`, widened to the source's preferred ranges when that is cheap, and
    /// served from this fetch's windows when possible.
    fn read(&self, range: Range<u64>) -> Result<Bytes, FetchError> {
        {
            let state = self.lock()?;
            for (start, bytes) in &state.windows {
                let end = start.saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
                if *start <= range.start && range.end <= end {
                    let from = usize::try_from(range.start.saturating_sub(*start))
                        .map_err(|_| malformed("a read is too large"))?;
                    let to = usize::try_from(range.end.saturating_sub(*start))
                        .map_err(|_| malformed("a read is too large"))?;
                    return Ok(bytes.slice(from..to));
                }
            }
        }
        if let Some(window) = self.window_for(&range) {
            let bytes = self.0.shared.read(window.clone())?;
            let from = usize::try_from(range.start.saturating_sub(window.start))
                .map_err(|_| malformed("a read is too large"))?;
            let to = usize::try_from(range.end.saturating_sub(window.start))
                .map_err(|_| malformed("a read is too large"))?;
            let slice = bytes.slice(from..to);
            let mut state = self.lock()?;
            state.window_total = state
                .window_total
                .saturating_add(window.end.saturating_sub(window.start));
            state.windows.push_back((window.start, bytes));
            let cap = self.0.caps.window_bytes.saturating_mul(2);
            while state.window_total > cap {
                match state.windows.pop_front() {
                    Some((_, old)) => {
                        state.window_total = state
                            .window_total
                            .saturating_sub(u64::try_from(old.len()).unwrap_or(0));
                    }
                    None => break,
                }
            }
            return Ok(slice);
        }
        Ok(self.0.shared.read(range)?)
    }

    /// The preferred-range span covering `range`, when it is larger than `range` and within
    /// the window cap.
    fn window_for(&self, range: &Range<u64>) -> Option<Range<u64>> {
        let preferred = self.0.shared.preferred.as_ref()?;
        let first = preferred.partition_point(|r| r.end <= range.start);
        let mut span: Option<Range<u64>> = None;
        for r in preferred.get(first..)? {
            if r.start >= range.end {
                break;
            }
            span = Some(match span {
                None => r.start.min(range.start)..r.end.max(range.end),
                Some(s) => s.start..s.end.max(r.end),
            });
        }
        let span = span?;
        let span = span.start.min(range.start)..span.end.max(range.end);
        let len = span.end.saturating_sub(span.start);
        (len > range.end.saturating_sub(range.start) && len <= self.0.caps.window_bytes)
            .then_some(span)
    }

    /// Checks a parsed header against the limits and its column chunk. `available` is the
    /// number of bytes from the header's start to the end of what may hold the page.
    fn validate(
        &self,
        info: &PageHeaderInfo,
        guard: &ChunkGuard,
        available: u64,
    ) -> Result<(), FetchError> {
        let (Ok(uncompressed), Ok(compressed)) = (
            u64::try_from(info.uncompressed),
            u64::try_from(info.compressed),
        ) else {
            return Err(malformed("a page header declares a negative size"));
        };
        if uncompressed > self.0.caps.max_page_bytes || compressed > self.0.caps.max_page_bytes {
            return Err(FetchError::limit("a page exceeds the page size limit"));
        }
        let header = u64::try_from(info.header_len).unwrap_or(u64::MAX);
        if header.saturating_add(compressed) > available {
            return Err(malformed("a page extends past its column chunk"));
        }
        if let Some(values) = info.dictionary_values {
            // Every dictionary entry takes at least a byte.
            match u64::try_from(values) {
                Ok(values) if values <= uncompressed => {}
                _ => return Err(malformed("a dictionary page declares too many values")),
            }
        }
        if let Some(values) = info.data_values {
            let values = u64::try_from(values)
                .map_err(|_| malformed("a data page declares a negative value count"))?;
            if guard.flat && values > guard.group_rows {
                return Err(malformed("a data page declares more values than rows"));
            }
        }
        if let Some(rows) = info.data_rows {
            match u64::try_from(rows) {
                Ok(rows) if rows <= guard.group_rows => {}
                _ => {
                    return Err(malformed(
                        "a data page declares more rows than its row group",
                    ));
                }
            }
        }
        if let Some((def, rep)) = info.v2_levels {
            let levels = i64::from(def).saturating_add(i64::from(rep));
            if def < 0
                || rep < 0
                || levels > i64::from(info.compressed)
                || levels > i64::from(info.uncompressed)
            {
                return Err(malformed("a data page declares invalid level lengths"));
            }
        }
        Ok(())
    }

    /// The trial decompression to run on a page's data: (uncompressed prefix, limit).
    fn expansion_plan(info: &PageHeaderInfo, guard: &ChunkGuard) -> Option<(usize, usize)> {
        if !expansion::needs_check(guard.codec) {
            return None;
        }
        let uncompressed = usize::try_from(info.uncompressed).ok()?;
        match info.page_type {
            // Data page v1 and dictionary page: the whole data is compressed.
            0 | 2 => Some((0, uncompressed)),
            // Data page v2: the levels are stored uncompressed before the values.
            3 if info.v2_compressed => {
                let (def, rep) = info.v2_levels.unwrap_or((0, 0));
                let levels = usize::try_from(def)
                    .ok()?
                    .checked_add(usize::try_from(rep).ok()?)?;
                Some((levels, uncompressed.saturating_sub(levels)))
            }
            _ => None,
        }
    }

    fn check_expansion(
        codec: Compression,
        data: &[u8],
        levels: usize,
        limit: usize,
    ) -> Result<(), FetchError> {
        let compressed = data.get(levels..).unwrap_or(&[]);
        expansion::check(codec, compressed, limit).map_err(|e| match e {
            ExpansionError::TooLarge => malformed("a page decompresses beyond its declared size"),
            ExpansionError::Corrupt => malformed("a page could not be decompressed"),
        })
    }

    /// A page header at `start`, parsed and checked: exactly its bytes.
    fn header(&self, start: u64) -> Result<Bytes, FetchError> {
        self.check_cancel()?;
        let probe_end = start
            .checked_add(1)
            .ok_or(malformed("a read outside the planned column chunks"))?;
        let (index, region) = self.region(&(start..probe_end))?;
        let guard = region
            .chunk
            .as_ref()
            .ok_or(malformed("a read outside the planned column chunks"))?;
        let available = region.range.end.saturating_sub(start);
        let cap = available.min(self.0.caps.max_header_bytes.max(1));
        // Most headers are a few dozen bytes; larger ones (with statistics) are re-read.
        let mut want = cap.min(128);
        loop {
            self.check_cancel()?;
            let bytes = self.read(start..start.saturating_add(want))?;
            match thrift::parse_page_header(&bytes) {
                Ok(info) => {
                    self.validate(&info, guard, available)?;
                    let header = u64::try_from(info.header_len).unwrap_or(u64::MAX);
                    let pending = Pending {
                        data_start: start.saturating_add(header),
                        compressed: u64::try_from(info.compressed).unwrap_or(0),
                        levels: Self::expansion_plan(&info, guard).map_or(0, |p| p.0),
                        limit: Self::expansion_plan(&info, guard).map(|p| p.1),
                        dictionary: info.page_type == 2,
                    };
                    if let Some(slot) = self.lock()?.pending.get_mut(index) {
                        *slot = Some(pending);
                    }
                    return Ok(bytes.slice(..info.header_len));
                }
                Err(HeaderError::Incomplete) if want < cap => {
                    want = want.saturating_mul(2).min(cap);
                }
                Err(HeaderError::Incomplete) if cap < available => {
                    return Err(FetchError::limit("a page header exceeds the size limit"));
                }
                Err(HeaderError::Incomplete) => {
                    return Err(malformed("a page header is truncated"));
                }
                Err(HeaderError::Malformed(_)) => {
                    return Err(malformed("a page header is malformed"));
                }
            }
        }
    }

    /// `get_bytes`: exactly the requested range, with page checks.
    fn exact(&self, start: u64, length: usize) -> Result<Bytes, FetchError> {
        self.check_cancel()?;
        let len = u64::try_from(length).map_err(|_| FetchError::limit("a read is too large"))?;
        if len > self.0.caps.max_read {
            return Err(FetchError::limit("a read exceeds the size limit"));
        }
        let end = start
            .checked_add(len)
            .filter(|&end| end <= self.0.shared.len)
            .ok_or(malformed("a read past the end of the file"))?;
        let (index, region) = self.region(&(start..end))?;
        let Some(guard) = region.chunk.as_ref() else {
            return self.read(start..end);
        };
        let shared = &self.0.shared;
        if let Some(starts) = &guard.page_starts {
            // With an offset index every read is a whole page: header and data.
            if starts.binary_search(&start).is_err() {
                return Err(malformed("an unexpected read inside a column chunk"));
            }
            let dictionary = guard.dictionary_first && start == region.range.start;
            if dictionary && let Some(bytes) = shared.cached_dictionary(start, len) {
                return Ok(bytes);
            }
            let bytes = self.read(start..end)?;
            let info = match thrift::parse_page_header(&bytes) {
                Ok(info) => info,
                Err(HeaderError::Incomplete) => return Err(malformed("a page is truncated")),
                Err(HeaderError::Malformed(_)) => {
                    return Err(malformed("a page header is malformed"));
                }
            };
            self.validate(&info, guard, len)?;
            if let Some((levels, limit)) = Self::expansion_plan(&info, guard) {
                let data = bytes.get(info.header_len..).unwrap_or(&[]);
                Self::check_expansion(guard.codec, data, levels, limit)?;
            }
            if dictionary {
                shared.keep_dictionary(start, &bytes);
            }
            return Ok(bytes);
        }
        // Without one, a data read follows the header read of the same page.
        let pending = self.lock()?.pending.get_mut(index).and_then(Option::take);
        match pending {
            Some(p) if p.data_start == start && p.compressed == len => {
                if p.dictionary
                    && let Some(bytes) = shared.cached_dictionary(start, len)
                {
                    return Ok(bytes);
                }
                let bytes = self.read(start..end)?;
                if let Some(limit) = p.limit {
                    Self::check_expansion(guard.codec, &bytes, p.levels, limit)?;
                }
                if p.dictionary {
                    shared.keep_dictionary(start, &bytes);
                }
                Ok(bytes)
            }
            _ if expansion::needs_check(guard.codec) => {
                Err(malformed("an unexpected read inside a column chunk"))
            }
            _ => self.read(start..end),
        }
    }
}

impl<B: ByteSource> Length for Adapter<B> {
    fn len(&self) -> u64 {
        self.0.shared.len
    }
}

/// The reader `get_read` returns: it parses and checks the page header at its start when
/// it is first read from, and then yields exactly the header's bytes.
///
/// It is lazy because the crate also calls `get_read` at offsets it then never reads from
/// (when it has already peeked the header there), and those offsets are not headers.
#[derive(Debug)]
pub(crate) struct HeaderReader<B> {
    adapter: Adapter<B>,
    start: u64,
    header: Option<bytes::buf::Reader<Bytes>>,
}

impl<B: ByteSource> std::io::Read for HeaderReader<B> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.header.is_none() {
            match self.adapter.header(self.start) {
                Ok(bytes) => self.header = Some(bytes.reader()),
                Err(error) => {
                    let message = error.message();
                    let _ = self.adapter.0.failure.set(error);
                    return Err(std::io::Error::other(message));
                }
            }
        }
        match self.header.as_mut() {
            Some(header) => header.read(buf),
            None => Ok(0),
        }
    }
}

impl<B: ByteSource> ChunkReader for Adapter<B> {
    type T = HeaderReader<B>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        Ok(HeaderReader {
            adapter: self.clone(),
            start,
            header: None,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        self.exact(start, length).map_err(|e| self.fail(e))
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crate::source::MemorySource;

    fn caps() -> ReadCaps {
        ReadCaps {
            max_read: 1 << 20,
            max_page_bytes: 1 << 20,
            max_header_bytes: 1 << 10,
            window_bytes: 64,
        }
    }

    #[test]
    fn reads_stay_inside_their_regions() {
        let shared = Arc::new(Shared::new(MemorySource::new(vec![0u8; 100]), 0));
        let adapter = Adapter::new(
            shared,
            CancelToken::new(),
            caps(),
            vec![Region {
                range: 10..20,
                chunk: None,
            }],
        );
        assert!(adapter.get_bytes(10, 10).is_ok());
        assert!(adapter.get_bytes(15, 10).is_err());
        assert!(adapter.get_bytes(5, 2).is_err());
        assert!(adapter.get_bytes(u64::MAX, 2).is_err());
        let mut header = adapter.get_read(12).unwrap();
        let mut buf = [0u8; 1];
        assert!(
            std::io::Read::read(&mut header, &mut buf).is_err(),
            "index regions have no page headers"
        );
    }

    #[test]
    fn cancellation_is_checked_before_reading() {
        let shared = Arc::new(Shared::new(MemorySource::new(vec![0u8; 100]), 0));
        let cancel = CancelToken::new();
        let adapter = Adapter::new(
            Arc::clone(&shared),
            cancel.clone(),
            caps(),
            vec![Region {
                range: 0..100,
                chunk: None,
            }],
        );
        cancel.cancel();
        assert!(adapter.get_bytes(0, 10).is_err());
        assert_eq!(
            adapter.failure().map(|e| e.kind()),
            Some(crate::FetchErrorKind::Cancelled)
        );
        assert_eq!(shared.bytes_read.load(Ordering::Relaxed), 0);
    }

    #[derive(Debug)]
    struct Records(MemorySource);

    impl ByteSource for Records {
        fn len(&self) -> u64 {
            self.0.len()
        }
        fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
            self.0.read_exact_at(offset, buf)
        }
        fn preferred_ranges(&self) -> Option<Vec<Range<u64>>> {
            Some((0..10).map(|i| i * 10..i * 10 + 10).collect())
        }
    }

    #[test]
    fn preferred_ranges_widen_and_are_reused() {
        let shared = Arc::new(Shared::new(
            Records(MemorySource::new((0u8..100).collect::<Vec<_>>())),
            0,
        ));
        let adapter = Adapter::new(
            Arc::clone(&shared),
            CancelToken::new(),
            caps(),
            vec![Region {
                range: 0..100,
                chunk: None,
            }],
        );
        assert_eq!(&adapter.get_bytes(12, 3).unwrap()[..], &[12, 13, 14]);
        assert_eq!(&adapter.get_bytes(17, 2).unwrap()[..], &[17, 18]);
        assert_eq!(shared.reads.load(Ordering::Relaxed), 1);
        assert_eq!(shared.bytes_read.load(Ordering::Relaxed), 10);
        // A read across two records widens to both.
        assert_eq!(&adapter.get_bytes(28, 4).unwrap()[..], &[28, 29, 30, 31]);
        assert_eq!(shared.bytes_read.load(Ordering::Relaxed), 30);
    }
}
