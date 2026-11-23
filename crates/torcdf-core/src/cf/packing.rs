//! `scale_factor` / `add_offset` unpacking and missing-value detection.
//!
//! Fill, `missing_value` and valid-range checks all compare against the
//! packed (raw) value, as the CF conventions require.

use crate::model::{DType, VarInfo};

const FILL_FLOAT: f64 = 9.969209968386869e36;

/// Per-variable unpacking rules; build once, apply per element.
#[derive(Debug, Clone)]
pub struct Unpack {
    scale: f64,
    offset: f64,
    /// Values (packed) that mean "missing".
    missing: Vec<f64>,
    min: Option<f64>,
    max: Option<f64>,
    /// Compare in f32 precision (the raw data is f32).
    single: bool,
    /// `_Unsigned = "true"` on a signed integer: `2^bits`, added to raw
    /// values below zero.
    wrap: Option<f64>,
}

/// `2^bits` for a signed integer type.
fn signed_modulus(dt: &DType) -> Option<f64> {
    match dt {
        DType::I8 => Some(256.0),
        DType::I16 => Some(65536.0),
        DType::I32 => Some(4294967296.0),
        DType::I64 => Some(18446744073709551616.0),
        _ => None,
    }
}

fn default_fill(dt: &DType) -> Option<f64> {
    match dt {
        DType::I16 => Some(-32767.0),
        DType::U16 => Some(65535.0),
        DType::I32 => Some(-2147483647.0),
        DType::U32 => Some(4294967295.0),
        DType::I64 => Some(-9223372036854775806.0),
        DType::U64 => Some(18446744073709551614.0),
        DType::F32 | DType::F64 => Some(FILL_FLOAT),
        DType::I8 | DType::U8 | DType::Char | DType::Str | DType::Other(_) => None,
    }
}

impl Unpack {
    pub fn for_var(v: &VarInfo) -> Unpack {
        let unsigned = v.attrs.iter().any(|a| {
            a.name == "_Unsigned" && a.value.as_str().is_some_and(|s| s.trim().eq_ignore_ascii_case("true"))
        });
        let wrap = if unsigned { signed_modulus(&v.dtype) } else { None };
        let fix = |x: f64| match wrap {
            Some(m) if x < 0.0 => x + m,
            _ => x,
        };
        let nums = |name: &str| {
            v.attrs
                .iter()
                .find(|a| a.name == name)
                .and_then(|a| a.value.as_f64s())
                .unwrap_or_default()
        };
        let scale = nums("scale_factor").first().copied().unwrap_or(1.0);
        let offset = nums("add_offset").first().copied().unwrap_or(0.0);

        let mut missing: Vec<f64> = nums("missing_value").into_iter().map(fix).collect();
        match nums("_FillValue").first() {
            Some(&x) => missing.push(fix(x)),
            None => match wrap {
                // The unsigned type's default fill: 2^bits - 1, or 2^64 - 2.
                Some(m) => missing.push(if m > 1e19 { m - 2.0 } else { m - 1.0 }),
                None => missing.extend(default_fill(&v.dtype)),
            },
        }

        let range = nums("valid_range");
        let (min, max) = if range.len() >= 2 {
            (Some(fix(range[0])), Some(fix(range[1])))
        } else {
            (nums("valid_min").first().copied().map(fix), nums("valid_max").first().copied().map(fix))
        };

        Unpack { scale, offset, missing, min, max, single: v.dtype == DType::F32, wrap }
    }

    /// Unpacked value, or `f32::NAN` if `raw` is missing.
    pub fn apply(&self, raw: f64) -> f32 {
        let raw = self.unsign(raw);
        if raw.is_nan() || self.is_missing(raw) {
            return f32::NAN;
        }
        (raw * self.scale + self.offset) as f32
    }

    /// [`apply`](Self::apply) in full precision: unpacked value, or
    /// `f64::NAN` if `raw` is missing.
    pub fn apply_f64(&self, raw: f64) -> f64 {
        let raw = self.unsign(raw);
        if raw.is_nan() || self.is_missing(raw) {
            return f64::NAN;
        }
        raw * self.scale + self.offset
    }

    fn unsign(&self, raw: f64) -> f64 {
        match self.wrap {
            Some(m) if raw < 0.0 => raw + m,
            _ => raw,
        }
    }

    fn is_missing(&self, raw: f64) -> bool {
        let eq = |a: f64, b: f64| if self.single { a as f32 == b as f32 } else { a == b };
        self.missing.iter().any(|&m| eq(raw, m))
            || self.min.is_some_and(|m| raw < m)
            || self.max.is_some_and(|m| raw > m)
    }
}

#[cfg(test)]
mod tests {
    use crate::fixtures;
    use crate::{NcFile, SliceRequest};

    fn whole(nc: &NcFile, var: &str, lib: bool) -> Vec<f32> {
        let r = SliceRequest { var: var.into(), start: vec![0, 0], count: vec![2, 3], stride: vec![1, 1] };
        if lib { nc.read_slice_via_library(&r) } else { nc.read_slice(&r) }.unwrap().data
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    fn open() -> (tempfile::TempDir, NcFile) {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_unsigned(dir.path())).unwrap();
        (dir, nc)
    }

    #[test]
    fn unsigned_byte_fill_is_missing() {
        let (_d, nc) = open();
        let got = whole(&nc, "/ub", false);
        assert!(same(&got, &[0.0, 1.0, 127.0, 128.0, 254.0, f32::NAN]), "{got:?}");
    }

    #[test]
    fn unsigned_short_scaled() {
        let (_d, nc) = open();
        let got = whole(&nc, "/us", false);
        assert!(same(&got, &[0.0, 0.5, 16383.5, 16384.0, 32767.0, 16384.5]), "{got:?}");
    }

    #[test]
    fn signed_unchanged_without_attr() {
        let (_d, nc) = open();
        let got = whole(&nc, "/plain", false);
        assert!(same(&got, &[0.0, 1.0, 127.0, -128.0, -2.0, -1.0]), "{got:?}");
    }

    #[test]
    fn unsigned_fast_path_matches_library() {
        let (_d, nc) = open();
        for v in ["/ub", "/us", "/plain"] {
            assert!(nc.fast_eligible(v), "{v}");
            assert!(same(&whole(&nc, v, true), &whole(&nc, v, false)), "{v}");
        }
    }
}
