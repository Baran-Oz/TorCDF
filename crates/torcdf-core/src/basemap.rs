//! Compact binary format for embedded basemap polylines.
//!
//! Layout (little-endian): `b"TCNE"`, u32 version (1), u32 n, n x u32 point
//! counts, then all points as (f32 lon, f32 lat).

use crate::CoreError;

pub type Polyline = Vec<[f32; 2]>;

const MAGIC: &[u8; 4] = b"TCNE";
const VERSION: u32 = 1;

pub fn encode(lines: &[Polyline]) -> Vec<u8> {
    let pts: usize = lines.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(12 + lines.len() * 4 + pts * 8);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(lines.len() as u32).to_le_bytes());
    for l in lines {
        out.extend_from_slice(&(l.len() as u32).to_le_bytes());
    }
    for p in lines.iter().flatten() {
        out.extend_from_slice(&p[0].to_le_bytes());
        out.extend_from_slice(&p[1].to_le_bytes());
    }
    out
}

fn err(msg: &str) -> CoreError {
    CoreError::Read { var: "basemap".into(), msg: msg.into() }
}

pub fn decode(bytes: &[u8]) -> Result<Vec<Polyline>, CoreError> {
    if bytes.len() < 12 || &bytes[0..4] != MAGIC {
        return Err(err("bad magic"));
    }
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    if u32_at(4) != VERSION {
        return Err(err("unsupported version"));
    }
    let n = u32_at(8) as usize;
    let header = n
        .checked_mul(4)
        .and_then(|h| h.checked_add(12))
        .filter(|&h| h <= bytes.len())
        .ok_or_else(|| err("truncated header"))?;
    let counts: Vec<usize> = (0..n).map(|i| u32_at(12 + i * 4) as usize).collect();
    let total = counts
        .iter()
        .try_fold(0usize, |a, &c| a.checked_add(c))
        .and_then(|t| t.checked_mul(8))
        .ok_or_else(|| err("bad length"))?;
    if bytes.len() - header != total {
        return Err(err("bad length"));
    }
    let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let mut off = header;
    let mut out = Vec::with_capacity(n);
    for c in counts {
        let mut l = Vec::with_capacity(c);
        for _ in 0..c {
            l.push([f(off), f(off + 4)]);
            off += 8;
        }
        out.push(l);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basemap_round_trip() {
        let l = vec![vec![[0.0, 1.0], [2.0, 3.0]], vec![[-180.0, -90.0]]];
        assert_eq!(decode(&encode(&l)).unwrap(), l);
    }

    #[test]
    fn basemap_rejects_garbage() {
        assert!(decode(b"nope").is_err());
        let mut b = encode(&[vec![[0.0, 0.0]]]);
        b.truncate(b.len() - 1);
        assert!(decode(&b).is_err());
    }
}
