//! Unpacked hyperslab reads.
//!
//! netCDF-C strided reads are never used (they are very slow on chunked,
//! compressed data). A strided request is served by reading contiguous
//! hyperslabs and thinning along the last dimension here in Rust.

use crate::cf::Unpack;
use crate::{CoreError, DType, NcFile, VarInfo, VectorParts};
use netcdf::Extent;
use std::os::raw::c_char;

/// Largest f64 read buffer, in bytes, used for one netCDF call.
const BATCH_BYTES: usize = 64 * 1024 * 1024;
const BATCH_ELEMS: usize = BATCH_BYTES / std::mem::size_of::<f64>();

/// A hyperslab request: for each dimension, `count` indices starting at
/// `start`, of which every `stride`-th is returned.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SliceRequest {
    pub var: String,
    pub start: Vec<usize>,
    pub count: Vec<usize>,
    pub stride: Vec<usize>,
}

/// Row-major, unpacked values; `NaN` marks missing data.
pub struct Slice {
    /// `ceil(count[i] / stride[i])` per dimension.
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    /// Number of non-NaN values in `data`.
    pub valid_count: usize,
    /// East/north components, shaped like `data`, when the slice is of a
    /// vector field.
    pub vector: Option<VectorParts>,
}

impl Slice {
    pub fn byte_size(&self) -> usize {
        let parts = self.vector.as_ref().map_or(0, |v| v.east.len() + v.north.len());
        (self.data.len() + parts) * std::mem::size_of::<f32>()
    }
}

impl NcFile {
    /// Reads the requested hyperslab, unpacking every value.
    ///
    /// Variables the fast reader can decode are read by it; if it fails for
    /// any reason, the read is silently repeated through netCDF-C, so a
    /// fast-reader bug cannot cost the user data.
    pub fn read_slice(&self, r: &SliceRequest) -> Result<Slice, CoreError> {
        if self.fast_eligible(&r.var)
            && let Ok(s) = self.read_fast(r)
        {
            return Ok(s);
        }
        self.read_slice_via_library(r)
    }

    /// `read_slice` always through netCDF-C (diagnostics and benchmarks).
    pub fn read_slice_via_library(&self, r: &SliceRequest) -> Result<Slice, CoreError> {
        self.read_slice_batched(r, BATCH_ELEMS)
    }

    /// `read_slice_via_library` with the per-read element limit given (testable).
    fn read_slice_batched(&self, r: &SliceRequest, batch: usize) -> Result<Slice, CoreError> {
        let (info, shape, total) = self.check_request(r)?;
        let n = shape.len();
        let unpack = Unpack::for_var(info);
        let mut data = vec![f32::NAN; total];
        if n == 0 {
            let raw = self.read_raw(&r.var, &[], &[])?;
            data[0] = unpack.apply(raw[0]);
        } else {
            self.read_blocks(r, &shape, &unpack, batch, &mut data)?;
        }
        let valid_count = data.iter().filter(|v| !v.is_nan()).count();
        Ok(Slice { shape, data, valid_count, vector: None })
    }

    /// Validates `r` against its variable. Returns the variable, the output
    /// shape (`ceil(count / stride)` per dimension) and its element count.
    pub(crate) fn check_request(
        &self,
        r: &SliceRequest,
    ) -> Result<(&VarInfo, Vec<usize>, usize), CoreError> {
        let err = |msg: &str| CoreError::Read { var: r.var.clone(), msg: msg.into() };
        let info = self.numeric_var(&r.var)?;
        let n = info.shape.len();
        if r.start.len() != n || r.count.len() != n || r.stride.len() != n {
            return Err(err(&format!("request has the wrong rank (variable has {n} dimensions)")));
        }
        let mut shape = Vec::with_capacity(n);
        for d in 0..n {
            let (start, count, stride) = (r.start[d], r.count[d], r.stride[d]);
            if count == 0 || stride == 0 {
                return Err(err("count and stride must be at least 1"));
            }
            if start.checked_add(count).is_none_or(|end| end > info.shape[d]) {
                return Err(err(&format!(
                    "dimension {d}: start {start} + count {count} exceeds length {}",
                    info.shape[d]
                )));
            }
            shape.push(count.div_ceil(stride));
        }
        let total = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .filter(|t| t.checked_mul(4).is_some())
            .ok_or_else(|| err("slice is too large"))?;
        Ok((info, shape, total))
    }

    /// Fills `out` (row-major over `shape`) from contiguous hyperslabs.
    ///
    /// Dimensions `m..` form one block read per iteration of the outer
    /// dimensions `..m`. `m` is lowered while every dimension from `m-1` on
    /// has stride 1 and the block fits the batch size, so unstrided reads are
    /// few large reads rather than many row reads. The last dimension is
    /// split into segments when a single row would exceed the batch size.
    fn read_blocks(
        &self,
        r: &SliceRequest,
        shape: &[usize],
        unpack: &Unpack,
        batch: usize,
        out: &mut [f32],
    ) -> Result<(), CoreError> {
        let n = shape.len();
        let last = n - 1;
        let mut m = last;
        if r.stride[last] == 1 {
            let mut block = r.count[last];
            while m > 0 && r.stride[m - 1] == 1 {
                match block.checked_mul(r.count[m - 1]) {
                    Some(b) if b <= batch => {
                        block = b;
                        m -= 1;
                    }
                    _ => break,
                }
            }
        }
        // Output elements per iteration of the outer dimensions.
        let block_out: usize = shape[m..].iter().product();
        // Output elements of the last dimension per segment (1+ elements,
        // spanning (k-1)*stride+1 source elements, within the batch size).
        let stride_last = r.stride[last];
        let seg = if m == last {
            (((batch - 1) / stride_last) + 1).min(shape[last])
        } else {
            shape[last]
        };

        let outer_total: usize = shape[..m].iter().product();
        let mut idx = vec![0usize; m];
        let mut start = vec![0usize; n];
        let mut count = vec![0usize; n];
        let mut raw: Vec<f64> = Vec::new();
        for outer in 0..outer_total {
            for d in 0..m {
                start[d] = r.start[d] + idx[d] * r.stride[d];
                count[d] = 1;
            }
            start[m..last].copy_from_slice(&r.start[m..last]);
            count[m..last].copy_from_slice(&r.count[m..last]);
            let base = outer * block_out;
            let mut done = 0;
            while done < shape[last] {
                let k = seg.min(shape[last] - done);
                start[last] = r.start[last] + done * stride_last;
                count[last] = if m == last { (k - 1) * stride_last + 1 } else { r.count[last] };
                let len: usize = count.iter().product();
                raw.resize(len, 0.0);
                self.read_raw_into(&r.var, &start, &count, &mut raw)?;
                let dst = &mut out[base + done..];
                if m == last {
                    for (o, v) in dst.iter_mut().take(k).zip(raw.iter().step_by(stride_last)) {
                        *o = unpack.apply(*v);
                    }
                } else {
                    for (o, v) in dst.iter_mut().take(len).zip(raw.iter()) {
                        *o = unpack.apply(*v);
                    }
                }
                done += k;
            }
            // Advance the odometer over the outer dimensions.
            for d in (0..m).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(())
    }

    /// Reads a 1-D coordinate variable, or a 2-D one flattened row-major, as
    /// raw `f64` (no unpacking, no longitude wrapping).
    pub fn read_coord(&self, var_path: &str) -> Result<Vec<f64>, CoreError> {
        let info = self.numeric_var(var_path)?;
        if !(1..=2).contains(&info.shape.len()) {
            return Err(CoreError::Unsupported(format!(
                "{var_path}: coordinate variables must be 1-D or 2-D"
            )));
        }
        let start = vec![0; info.shape.len()];
        let count = info.shape.clone();
        self.read_raw(var_path, &start, &count)
    }

    /// The (y, x) plane of a 2-D lat/lon variable, at index 0 of any other
    /// dims (decision P6, e.g. WRF's `XLAT(Time, south_north, west_east)`),
    /// unpacked to f64 (missing = NaN), row-major over (y, x) whatever the
    /// variable's own dim order.
    pub fn read_horizontal(
        &self,
        var_path: &str,
        y_dim: &str,
        x_dim: &str,
    ) -> Result<Vec<f64>, CoreError> {
        let err = |msg: String| CoreError::Read { var: var_path.into(), msg };
        let info = self.numeric_var(var_path)?;
        let pos = |d: &str| {
            info.dims
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| err(format!("has no dimension {d}")))
        };
        let (py, px) = (pos(y_dim)?, pos(x_dim)?);
        if py == px {
            return Err(err("y and x must be different dimensions".into()));
        }
        let (ny, nx) = (info.shape[py], info.shape[px]);
        ny.checked_mul(nx)
            .filter(|n| n.checked_mul(std::mem::size_of::<f64>()).is_some_and(|b| b <= isize::MAX as usize))
            .ok_or_else(|| err("plane is too large".into()))?;
        if info.shape.contains(&0) {
            return Ok(Vec::new());
        }
        let mut count = vec![1; info.shape.len()];
        count[py] = ny;
        count[px] = nx;
        let unpack = Unpack::for_var(info);
        let raw = self.read_raw(var_path, &vec![0; count.len()], &count)?;
        // `raw` is row-major in the variable's dim order: (y, x) or (x, y).
        Ok(if py < px {
            raw.iter().map(|&v| unpack.apply_f64(v)).collect()
        } else {
            let mut out = Vec::with_capacity(ny * nx);
            for y in 0..ny {
                for x in 0..nx {
                    out.push(unpack.apply_f64(raw[x * ny + y]));
                }
            }
            out
        })
    }

    /// Reads text values of a string (`NC_STRING`) or char (`NC_CHAR`)
    /// variable over the hyperslab `start`/`count` (stride 1).
    ///
    /// A string variable gives one `String` per element. A char variable is
    /// an array of fixed-width strings whose last dimension is the string
    /// length: each run of `count[last]` characters becomes one `String`,
    /// with trailing NULs removed (a scalar char variable gives one string).
    /// Invalid UTF-8 is replaced, not rejected.
    pub fn read_strings(
        &self,
        var_path: &str,
        start: &[usize],
        count: &[usize],
    ) -> Result<Vec<String>, CoreError> {
        let err = |msg: String| CoreError::Read { var: var_path.into(), msg };
        let info = self.meta.var(var_path).ok_or_else(|| err("no such variable".into()))?;
        if !matches!(info.dtype, DType::Str | DType::Char) {
            return Err(CoreError::Unsupported(format!(
                "{var_path}: {:?} variables are not text",
                info.dtype
            )));
        }
        let n = info.shape.len();
        if start.len() != n || count.len() != n {
            return Err(err(format!("request has the wrong rank (variable has {n} dimensions)")));
        }
        for d in 0..n {
            if count[d] == 0 {
                return Err(err("count must be at least 1".into()));
            }
            if start[d].checked_add(count[d]).is_none_or(|end| end > info.shape[d]) {
                return Err(err(format!(
                    "dimension {d}: start {} + count {} exceeds length {}",
                    start[d], count[d], info.shape[d]
                )));
            }
        }
        let total = count
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| err("request is too large".into()))?;
        let extents: Vec<Extent> = start
            .iter()
            .zip(count)
            .map(|(&start, &count)| Extent::SliceCount { start, count, stride: 1 })
            .collect();
        let is_str = info.dtype == DType::Str;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let var = self
                .file
                .variable(var_path.trim_start_matches('/'))
                .ok_or_else(|| "variable not found".to_string())?;
            let elem = if is_str { std::mem::size_of::<*mut c_char>() } else { 1 };
            let mut raw = vec![0u8; total * elem];
            if extents.is_empty() {
                var.get_raw_values_into(&mut raw, ..)
            } else {
                var.get_raw_values_into(&mut raw, extents)
            }
            .map_err(|e| e.to_string())?;
            Ok::<_, String>(if is_str {
                // SAFETY: netCDF filled `raw` with `total` string pointers that
                // it allocated; each is read once, then all are freed.
                unsafe { take_nc_strings(&raw) }
            } else {
                let width = count.last().copied().unwrap_or(1);
                raw.chunks(width)
                    .map(|s| {
                        let end = s.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
                        String::from_utf8_lossy(&s[..end]).into_owned()
                    })
                    .collect()
            })
        }));
        match res {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(msg)) => Err(err(msg)),
            Err(_) => Err(err("netCDF read failed unexpectedly".into())),
        }
    }

    fn numeric_var(&self, path: &str) -> Result<&VarInfo, CoreError> {
        let info = self.meta.var(path).ok_or_else(|| CoreError::Read {
            var: path.into(),
            msg: "no such variable".into(),
        })?;
        if !info.dtype.is_numeric() {
            return Err(CoreError::Unsupported(format!(
                "{path}: {:?} variables cannot be read as numbers",
                info.dtype
            )));
        }
        Ok(info)
    }

    fn read_raw(&self, path: &str, start: &[usize], count: &[usize]) -> Result<Vec<f64>, CoreError> {
        let mut buf = vec![0.0; count.iter().product()];
        self.read_raw_into(path, start, count, &mut buf)?;
        Ok(buf)
    }

    /// One contiguous (stride 1) netCDF read of `count` elements at `start`.
    /// `buf.len()` must equal the product of `count`.
    fn read_raw_into(
        &self,
        path: &str,
        start: &[usize],
        count: &[usize],
        buf: &mut [f64],
    ) -> Result<(), CoreError> {
        let err = |msg: String| CoreError::Read { var: path.into(), msg };
        let extents: Vec<Extent> = start
            .iter()
            .zip(count)
            .map(|(&start, &count)| Extent::SliceCount { start, count, stride: 1 })
            .collect();
        // The netcdf crate panics on some lookup failures.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let var = self
                .file
                .variable(path.trim_start_matches('/'))
                .ok_or_else(|| "variable not found".to_string())?;
            if extents.is_empty() {
                var.get_values_into(buf, ..)
            } else {
                var.get_values_into(buf, extents)
            }
            .map_err(|e| e.to_string())
        }));
        match res {
            Ok(Ok(())) => Ok(()),
            Ok(Err(msg)) => Err(err(msg)),
            Err(_) => Err(err("netCDF read failed unexpectedly".into())),
        }
    }
}

/// Converts a buffer of netCDF-allocated `char*` (native-endian pointer
/// words) to owned strings and frees them with `nc_free_string`.
///
/// # Safety
/// `raw` must hold only pointers returned by a netCDF string read that have
/// not been freed yet; null pointers are allowed and read as "".
unsafe fn take_nc_strings(raw: &[u8]) -> Vec<String> {
    const W: usize = std::mem::size_of::<*mut c_char>();
    let mut ptrs: Vec<*mut c_char> = raw
        .as_chunks::<W>()
        .0
        .iter()
        .map(|c| usize::from_ne_bytes(*c) as *mut c_char)
        .collect();
    let out = ptrs
        .iter()
        .map(|&p| {
            if p.is_null() {
                String::new()
            } else {
                unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        })
        .collect();
    let _g = netcdf_sys::libnetcdf_lock.lock();
    unsafe { netcdf_sys::nc_free_string(ptrs.len(), ptrs.as_mut_ptr()) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    /// `read_horizontal` takes index 0 of other dims, reorders (x, y)
    /// storage to row-major (y, x) and unpacks to f64.
    #[test]
    fn read_horizontal_plane() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("plane.nc");
        {
            let mut f = netcdf::create(&path).unwrap();
            f.add_dimension("t", 2).unwrap();
            f.add_dimension("x", 3).unwrap();
            f.add_dimension("y", 2).unwrap();
            let mut v = f.add_variable::<i16>("lat", &["t", "x", "y"]).unwrap();
            v.set_fill_value(-999i16).unwrap();
            v.put_attribute("scale_factor", 0.5f64).unwrap();
            v.put_attribute("add_offset", 1.0f64).unwrap();
            let mut raw = Vec::new();
            for t in 0..2i16 {
                for x in 0..3i16 {
                    for y in 0..2i16 {
                        raw.push(if (t, x, y) == (0, 1, 0) { -999 } else { t * 100 + x * 10 + y });
                    }
                }
            }
            v.put_values(&raw, ..).unwrap();
        }
        let f = NcFile::open(&path).unwrap();
        let got = f.read_horizontal("/lat", "y", "x").unwrap();
        assert_eq!(got.len(), 6);
        for y in 0..2 {
            for x in 0..3 {
                let g = got[y * 3 + x];
                if (x, y) == (1, 0) {
                    assert!(g.is_nan());
                } else {
                    assert_eq!(g, (x * 10 + y) as f64 * 0.5 + 1.0, "y {y} x {x}");
                }
            }
        }
        assert!(f.read_horizontal("/nope", "y", "x").is_err());
    }

    /// Tiny batch limits force row splitting and last-dimension segmenting;
    /// results must equal the default path.
    #[test]
    fn batching_does_not_change_results() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
        let cases = [
            ([0, 0, 0], [24, 10, 20], [1, 1, 1]),
            ([0, 0, 0], [24, 10, 20], [5, 3, 4]),
            ([2, 1, 3], [7, 8, 15], [3, 2, 1]),
            ([2, 1, 3], [7, 8, 15], [1, 1, 3]),
        ];
        for (start, count, stride) in cases {
            let r = SliceRequest {
                var: "/ocean/uo".into(),
                start: start.to_vec(),
                count: count.to_vec(),
                stride: stride.to_vec(),
            };
            let want = f.read_slice(&r).unwrap();
            for batch in [1, 3, 7, 20, 25, 199, 200, 201, 4000] {
                let got = f.read_slice_batched(&r, batch).unwrap();
                assert_eq!(got.shape, want.shape);
                assert_eq!(got.valid_count, want.valid_count, "batch {batch}");
                let same = got
                    .data
                    .iter()
                    .zip(&want.data)
                    .all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b);
                assert!(same, "batch {batch} {r:?}");
            }
        }
    }
}
