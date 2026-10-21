//! Decompression-bomb checks for the codecs whose `parquet` decoders read to the end.
//!
//! The `parquet` crate decompresses GZIP and BROTLI pages with `read_to_end`, and LZ4
//! (Hadoop) pages fall back to the LZ4 frame decoder, which also reads to the end. None of
//! them stop at the page's declared uncompressed size, so a small hostile page could
//! expand to gigabytes before the size check runs. Before such a page is handed to the
//! crate, it is decompressed here into a small scratch buffer, counting, and refused as
//! soon as it exceeds its declared size. SNAPPY, ZSTD, and LZ4_RAW decode into a buffer of
//! the declared size and need no check. The cost is decoding those pages twice.

use std::io::Read;

use parquet::basic::Compression;

/// Whether pages of this codec need a trial decompression.
pub(crate) fn needs_check(codec: Compression) -> bool {
    matches!(
        codec,
        Compression::GZIP(_) | Compression::BROTLI(_) | Compression::LZ4
    )
}

/// Why a page failed the check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpansionError {
    /// The page expands beyond its declared size.
    TooLarge,
    /// The page does not decompress.
    Corrupt,
}

/// Reads `reader` to the end, refusing more than `limit` bytes.
fn drain(mut reader: impl Read, limit: usize) -> Result<(), ExpansionError> {
    let mut scratch = vec![0u8; 64 * 1024];
    let mut total = 0usize;
    loop {
        match reader.read(&mut scratch) {
            Ok(0) => return Ok(()),
            Ok(n) => {
                total = total.saturating_add(n);
                if total > limit {
                    return Err(ExpansionError::TooLarge);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return Err(ExpansionError::Corrupt),
        }
    }
}

/// Checks that `input`, compressed with `codec`, expands to at most `limit` bytes.
///
/// For LZ4 the crate first tries the Hadoop framing (bounded by the declared size) and only
/// then the frame format; checking the frame format alone is enough to bound the fallback.
/// A page that is not an LZ4 frame fails that decoder at once and passes.
pub(crate) fn check(codec: Compression, input: &[u8], limit: usize) -> Result<(), ExpansionError> {
    match codec {
        Compression::GZIP(_) => drain(flate2::read::MultiGzDecoder::new(input), limit),
        Compression::BROTLI(_) => drain(brotli::Decompressor::new(input, 4096), limit),
        Compression::LZ4 => match drain(lz4_flex::frame::FrameDecoder::new(input), limit) {
            Err(ExpansionError::Corrupt) => Ok(()),
            other => other,
        },
        _ => Ok(()),
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn gzip_bomb_is_refused() {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        encoder.write_all(&vec![0u8; 4 << 20]).unwrap();
        let bomb = encoder.finish().unwrap();
        assert!(bomb.len() < 64 * 1024);
        let codec = Compression::GZIP(Default::default());
        assert_eq!(check(codec, &bomb, 1 << 20), Err(ExpansionError::TooLarge));
        assert_eq!(check(codec, &bomb, 4 << 20), Ok(()));
        assert_eq!(
            check(codec, b"not gzip", 1 << 20),
            Err(ExpansionError::Corrupt)
        );
    }

    #[test]
    fn lz4_frame_bomb_is_refused() {
        let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
        encoder.write_all(&vec![7u8; 2 << 20]).unwrap();
        let bomb = encoder.finish().unwrap();
        assert_eq!(
            check(Compression::LZ4, &bomb, 1 << 20),
            Err(ExpansionError::TooLarge)
        );
        // Hadoop-framed input is not an LZ4 frame and passes.
        assert_eq!(
            check(Compression::LZ4, &[0, 0, 0, 4, 0, 0, 0, 1, 9], 16),
            Ok(())
        );
    }
}
