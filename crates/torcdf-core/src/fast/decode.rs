//! Undoing HDF5 filters and converting stored bytes to numbers.
//!
//! Pure functions on one chunk's bytes; no HDF5 calls and no lock.

use super::h5::{ByteOrder, Filter};
use crate::DType;
use std::cell::RefCell;

thread_local! {
    /// One inflate state per decode thread, reused across chunks.
    static INFLATE: RefCell<libdeflater::Decompressor> =
        RefCell::new(libdeflater::Decompressor::new());
}

/// Undoes `filters` (in write order) on one chunk's stored bytes: last
/// filter first, skipping each filter whose bit is set in `filter_mask`.
///
/// `elem_size` is the shuffle element size. `max_len` bounds the inflated
/// size (a chunk's raw bytes plus any checksum); inflating to more is an
/// error. A Fletcher32 checksum mismatch is an error too.
pub(crate) fn unfilter(
    raw: Vec<u8>,
    filters: &[Filter],
    filter_mask: u32,
    elem_size: usize,
    max_len: usize,
) -> Result<Vec<u8>, String> {
    let mut buf = raw;
    for (i, f) in filters.iter().enumerate().rev() {
        let skipped = u32::try_from(i)
            .ok()
            .and_then(|i| filter_mask.checked_shr(i))
            .is_some_and(|m| m & 1 != 0);
        if skipped {
            continue;
        }
        buf = match f {
            Filter::Deflate => inflate(&buf, max_len)?,
            Filter::Shuffle => unshuffle(buf, elem_size),
            Filter::Fletcher32 => strip_fletcher32(buf)?,
        };
    }
    Ok(buf)
}

fn inflate(src: &[u8], max_len: usize) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; max_len];
    let n = INFLATE
        .with_borrow_mut(|d| d.zlib_decompress(src, &mut out))
        .map_err(|e| format!("inflate failed: {e:?}"))?;
    out.truncate(n);
    Ok(out)
}

/// HDF5's shuffle, undone: byte `b` of element `j` was stored at
/// `b * n + j`. Trailing `len % elem_size` bytes were not shuffled.
fn unshuffle(src: Vec<u8>, elem_size: usize) -> Vec<u8> {
    let n = src.len() / elem_size.max(1);
    if elem_size <= 1 || n <= 1 {
        return src;
    }
    let body = n * elem_size;
    let mut out = vec![0u8; src.len()];
    match elem_size {
        2 => unshuffle_n::<2>(&src[..body], &mut out[..body], n),
        4 => unshuffle_n::<4>(&src[..body], &mut out[..body], n),
        8 => unshuffle_n::<8>(&src[..body], &mut out[..body], n),
        _ => {
            for (j, dst) in out[..body].chunks_exact_mut(elem_size).enumerate() {
                for (b, o) in dst.iter_mut().enumerate() {
                    *o = src[b * n + j];
                }
            }
        }
    }
    out[body..].copy_from_slice(&src[body..]);
    out
}

/// `unshuffle` for `N`-byte elements; `src` and `out` hold `n` elements.
fn unshuffle_n<const N: usize>(src: &[u8], out: &mut [u8], n: usize) {
    let planes: [&[u8]; N] = std::array::from_fn(|b| &src[b * n..(b + 1) * n]);
    for (j, dst) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        for (o, plane) in dst.iter_mut().zip(&planes) {
            *o = plane[j];
        }
    }
}

/// Checks and removes the trailing checksum HDF5's Fletcher32 filter adds.
fn strip_fletcher32(mut buf: Vec<u8>) -> Result<Vec<u8>, String> {
    let body = buf
        .len()
        .checked_sub(4)
        .ok_or("chunk is too short for its checksum")?;
    let stored = u32::from_le_bytes([buf[body], buf[body + 1], buf[body + 2], buf[body + 3]]);
    let sum = fletcher32(&buf[..body]);
    // HDF5 before 1.6.3 wrote the checksum with the bytes of each 16-bit
    // half swapped; HDF5 still accepts both.
    let swapped = ((sum & 0x00ff_00ff) << 8) | ((sum >> 8) & 0x00ff_00ff);
    if stored != sum && stored != swapped {
        return Err("chunk checksum mismatch".into());
    }
    buf.truncate(body);
    Ok(buf)
}

/// HDF5's `H5_checksum_fletcher32`: 16-bit big-endian words, an odd final
/// byte padded with zero, sums folded every 360 words (wrapping as in C).
fn fletcher32(data: &[u8]) -> u32 {
    let fold = |s: u32| (s & 0xffff) + (s >> 16);
    let (mut s1, mut s2) = (0u32, 0u32);
    let (words, rest) = data.as_chunks::<2>();
    for block in words.chunks(360) {
        for w in block {
            s1 = s1.wrapping_add(u32::from(u16::from_be_bytes(*w)));
            s2 = s2.wrapping_add(s1);
        }
        s1 = fold(s1);
        s2 = fold(s2);
    }
    if let [last] = rest {
        s1 = s1.wrapping_add(u32::from(*last) << 8);
        s2 = s2.wrapping_add(s1);
        s1 = fold(s1);
        s2 = fold(s2);
    }
    (fold(s2) << 16) | fold(s1)
}

/// Converts stored elements of `dtype` in byte order `order` to `f64`, as
/// netCDF-C's conversion to double does. Fills `out.len()` elements.
pub(crate) fn bytes_to_f64(bytes: &[u8], dtype: &DType, order: ByteOrder, out: &mut [f64]) {
    macro_rules! conv {
        ($t:ty, $n:literal) => {{
            let words = bytes.as_chunks::<$n>().0;
            match order {
                ByteOrder::Little => {
                    for (o, w) in out.iter_mut().zip(words) {
                        *o = <$t>::from_le_bytes(*w) as f64;
                    }
                }
                ByteOrder::Big => {
                    for (o, w) in out.iter_mut().zip(words) {
                        *o = <$t>::from_be_bytes(*w) as f64;
                    }
                }
            }
        }};
    }
    match dtype {
        DType::I8 => conv!(i8, 1),
        DType::U8 => conv!(u8, 1),
        DType::I16 => conv!(i16, 2),
        DType::U16 => conv!(u16, 2),
        DType::I32 => conv!(i32, 4),
        DType::U32 => conv!(u32, 4),
        DType::I64 => conv!(i64, 8),
        DType::U64 => conv!(u64, 8),
        DType::F32 => conv!(f32, 4),
        DType::F64 => conv!(f64, 8),
        DType::Char | DType::Str | DType::Other(_) => out.fill(f64::NAN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_roundtrip() {
        let plain: Vec<u8> = (1..=14).collect();
        // 3 elements of 4 bytes: byte b of element j goes to b * 3 + j.
        let mut shuffled = vec![0u8; 14];
        for j in 0..3 {
            for b in 0..4 {
                shuffled[b * 3 + j] = plain[j * 4 + b];
            }
        }
        shuffled[12..].copy_from_slice(&plain[12..]);
        assert_ne!(shuffled, plain);
        assert_eq!(
            unfilter(shuffled, &[Filter::Shuffle], 0, 4, 14).unwrap(),
            plain
        );
    }

    #[test]
    fn filter_mask_skips_filter() {
        let plain: Vec<u8> = (0..400u32).flat_map(|i| (i * 7).to_le_bytes()).collect();
        let mut c = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let mut packed = vec![0u8; c.zlib_compress_bound(plain.len())];
        let n = c.zlib_compress(&plain, &mut packed).unwrap();
        packed.truncate(n);
        let out = unfilter(
            packed,
            &[Filter::Shuffle, Filter::Deflate],
            0b1,
            4,
            plain.len(),
        );
        assert_eq!(out.unwrap(), plain);
    }

    #[test]
    fn fletcher32_checked_and_stripped() {
        // Odd length: the final byte is padded.
        let data: Vec<u8> = (0..1001u32).map(|i| (i * 31 % 251) as u8).collect();
        let mut buf = data.clone();
        buf.extend(fletcher32(&data).to_le_bytes());
        let out = unfilter(buf.clone(), &[Filter::Fletcher32], 0, 4, 0);
        assert_eq!(out.unwrap(), data);
        buf[10] ^= 1;
        assert!(unfilter(buf.clone(), &[Filter::Fletcher32], 0, 4, 0).is_err());
        assert!(unfilter(vec![1, 2, 3], &[Filter::Fletcher32], 0, 4, 0).is_err());
    }
}
