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
        let nums = |name: &str| {
            v.attrs
                .iter()
                .find(|a| a.name == name)
                .and_then(|a| a.value.as_f64s())
                .unwrap_or_default()
        };
        let scale = nums("scale_factor").first().copied().unwrap_or(1.0);
        let offset = nums("add_offset").first().copied().unwrap_or(0.0);

        let mut missing = nums("missing_value");
        match nums("_FillValue").first() {
            Some(&x) => missing.push(x),
            None => missing.extend(default_fill(&v.dtype)),
        }

        let range = nums("valid_range");
        let (min, max) = if range.len() >= 2 {
            (Some(range[0]), Some(range[1]))
        } else {
            (nums("valid_min").first().copied(), nums("valid_max").first().copied())
        };

        Unpack { scale, offset, missing, min, max, single: v.dtype == DType::F32 }
    }

    /// Unpacked value, or `f32::NAN` if `raw` is missing.
    pub fn apply(&self, raw: f64) -> f32 {
        if raw.is_nan() || self.is_missing(raw) {
            return f32::NAN;
        }
        (raw * self.scale + self.offset) as f32
    }

    /// [`apply`](Self::apply) in full precision: unpacked value, or
    /// `f64::NAN` if `raw` is missing.
    pub fn apply_f64(&self, raw: f64) -> f64 {
        if raw.is_nan() || self.is_missing(raw) {
            return f64::NAN;
        }
        raw * self.scale + self.offset
    }

    fn is_missing(&self, raw: f64) -> bool {
        let eq = |a: f64, b: f64| if self.single { a as f32 == b as f32 } else { a == b };
        self.missing.iter().any(|&m| eq(raw, m))
            || self.min.is_some_and(|m| raw < m)
            || self.max.is_some_and(|m| raw > m)
    }
}
