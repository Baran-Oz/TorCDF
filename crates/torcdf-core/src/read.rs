//! Unpacked hyperslab reads.
//!
//! netCDF-C strided reads are never used (they are very slow on chunked,
//! compressed data). A strided request is served by reading contiguous
//! hyperslabs and thinning along the last dimension here in Rust.
//!
//! Files on the HDF5 backend are read with `H5Dread` of a strided
//! hyperslab, converted by HDF5 to native `double`: one read per batch.

use crate::cf::Unpack;
use crate::fast::h5::H5File;
use crate::h5meta::Held;
use crate::open::Backend;
use crate::{CoreError, DType, NcFile, VarInfo, VectorParts};
use hdf5_metno_sys::h5::hsize_t;
use hdf5_metno_sys::h5d::{H5Dclose, H5Dget_space, H5Dopen2, H5Dread};
use hdf5_metno_sys::h5p::H5P_DEFAULT;
use hdf5_metno_sys::h5s::{
    H5S_seloper_t, H5Sclose, H5Screate_simple, H5Sget_select_npoints, H5Sget_simple_extent_ndims,
    H5Sselect_hyperslab,
};
use hdf5_metno_sys::h5t::H5T_NATIVE_DOUBLE;
use netcdf::Extent;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr::null;

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
    ///
    /// A registered vector pair path is read on its centre grid: `data` is
    /// the magnitude and `vector` holds east/north.
    pub fn read_slice(&self, r: &SliceRequest) -> Result<Slice, CoreError> {
        if self.meta.var(&r.var).is_none()
            && let Some(pair) = self.pair(&r.var)
        {
            return self.read_pair(&pair, r);
        }
        if self.meta.var(&r.var).is_none()
            && let Some((var, pair)) = crate::vector::split_layer(&r.var)
        {
            return self.read_layer(var, pair, r);
        }
        if self.fast_eligible(&r.var)
            && let Ok(s) = self.read_fast(r)
        {
            return Ok(s);
        }
        self.read_slice_via_library(r)
    }

    /// An arrow layer: `var`'s values with the registered pair's arrows,
    /// both over the request's indices.
    fn read_layer(&self, var: &str, pair: &str, r: &SliceRequest) -> Result<Slice, CoreError> {
        let misfit = || CoreError::Read { var: r.var.clone(), msg: format!("arrows {pair} do not fit {var}") };
        let p = self.pair(pair).filter(|p| self.layer_fits(var, p)).ok_or_else(misfit)?;
        let x = self.read_slice(&SliceRequest { var: var.into(), ..r.clone() })?;
        let v = self.read_pair(&p, &SliceRequest { var: p.path.clone(), ..r.clone() })?;
        Ok(Slice { shape: x.shape, data: x.data, valid_count: x.valid_count, vector: v.vector })
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
        } else if let Backend::Hdf5(h5) = &self.backend {
            read_h5_blocks(h5, r, &shape, &unpack, batch, &mut data)?;
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
        let info = self.numeric_var(&r.var)?;
        let (shape, total) = check_shape(r, &info.shape)?;
        Ok((info, shape, total))
    }

    /// Validates `r` against a variable or a registered vector pair (its
    /// centre grid). Returns the output shape.
    pub(crate) fn check_any_request(&self, r: &SliceRequest) -> Result<Vec<usize>, CoreError> {
        if self.meta.var(&r.var).is_none()
            && let Some(pair) = self.pair(&r.var)
        {
            return check_shape(r, &pair.centre_shape).map(|(shape, _)| shape);
        }
        if self.meta.var(&r.var).is_none()
            && let Some((var, _)) = crate::vector::split_layer(&r.var)
        {
            let x = SliceRequest { var: var.into(), ..r.clone() };
            return self.check_request(&x).map(|(_, shape, _)| shape);
        }
        self.check_request(r).map(|(_, shape, _)| shape)
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
                .netcdf()?
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

    pub(crate) fn numeric_var(&self, path: &str) -> Result<&VarInfo, CoreError> {
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

    /// One contiguous (stride 1) read of `count` elements at `start`, through
    /// the file's backend. `buf.len()` must equal the product of `count`.
    fn read_raw_into(
        &self,
        path: &str,
        start: &[usize],
        count: &[usize],
        buf: &mut [f64],
    ) -> Result<(), CoreError> {
        let err = |msg: String| CoreError::Read { var: path.into(), msg };
        if let Backend::Hdf5(h5) = &self.backend {
            return h5_read(h5, path, start, count, &vec![1; start.len()], buf).map_err(err);
        }
        let extents: Vec<Extent> = start
            .iter()
            .zip(count)
            .map(|(&start, &count)| Extent::SliceCount { start, count, stride: 1 })
            .collect();
        // The netcdf crate panics on some lookup failures.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let var = self
                .netcdf()?
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

/// Validates `r` against a variable of shape `dims`. Returns the output shape
/// (`ceil(count / stride)` per dimension) and its element count.
pub(crate) fn check_shape(r: &SliceRequest, dims: &[usize]) -> Result<(Vec<usize>, usize), CoreError> {
    let err = |msg: &str| CoreError::Read { var: r.var.clone(), msg: msg.into() };
    let n = dims.len();
    if r.start.len() != n || r.count.len() != n || r.stride.len() != n {
        return Err(err(&format!("request has the wrong rank (variable has {n} dimensions)")));
    }
    let mut shape = Vec::with_capacity(n);
    for (d, &len) in dims.iter().enumerate() {
        let (start, count, stride) = (r.start[d], r.count[d], r.stride[d]);
        if count == 0 || stride == 0 {
            return Err(err("count and stride must be at least 1"));
        }
        if start.checked_add(count).is_none_or(|end| end > len) {
            return Err(err(&format!(
                "dimension {d}: start {start} + count {count} exceeds length {len}"
            )));
        }
        shape.push(count.div_ceil(stride));
    }
    let total = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .filter(|t| t.checked_mul(4).is_some())
        .ok_or_else(|| err("slice is too large"))?;
    Ok((shape, total))
}

/// Fills `out` (row-major over `shape`, the output of the validated request
/// `r`) through the HDF5 backend, one strided `H5Dread` per block.
///
/// A block is whole output dimensions `m+1..`, up to `g` output indices of
/// dimension `m` and one index of each dimension before it, at most `batch`
/// elements (`g` is at least 1), so it is contiguous in `out`.
fn read_h5_blocks(
    h5: &H5File,
    r: &SliceRequest,
    shape: &[usize],
    unpack: &Unpack,
    batch: usize,
    out: &mut [f32],
) -> Result<(), CoreError> {
    let err = |msg: String| CoreError::Read { var: r.var.clone(), msg };
    let n = shape.len();
    // `inner`: output elements of dimensions `m+1..` (fits: `out` does).
    let (mut m, mut inner) = (n - 1, 1usize);
    while m > 0 && inner * shape[m] <= batch {
        inner *= shape[m];
        m -= 1;
    }
    let g = (batch / inner).clamp(1, shape[m]);
    let outer_total: usize = shape[..m].iter().product();
    let mut idx = vec![0usize; m];
    let mut start = r.start.clone();
    let mut count = shape.to_vec();
    let mut raw: Vec<f64> = Vec::new();
    for outer in 0..outer_total {
        for d in 0..m {
            start[d] = r.start[d] + idx[d] * r.stride[d];
            count[d] = 1;
        }
        let mut j = 0;
        while j < shape[m] {
            let k = g.min(shape[m] - j);
            start[m] = r.start[m] + j * r.stride[m];
            count[m] = k;
            // Zeroed each time: nothing of an earlier block can show through.
            raw.clear();
            raw.resize(k * inner, 0.0);
            h5_read(h5, &r.var, &start, &count, &r.stride, &mut raw).map_err(err)?;
            let base = (outer * shape[m] + j) * inner;
            for (o, v) in out[base..base + raw.len()].iter_mut().zip(&raw) {
                *o = unpack.apply(*v);
            }
            j += k;
        }
        // Advance the odometer over dimensions `..m`.
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

/// One `H5Dread` of dataset `path`: per dimension, `count` indices from
/// `start`, every `stride`-th, converted to `f64` into `buf` (row-major).
/// `buf.len()` must equal the number of selected elements; a scalar
/// dataset takes empty slices and a buffer of 1.
fn h5_read(
    h5: &H5File,
    path: &str,
    start: &[usize],
    count: &[usize],
    stride: &[usize],
    buf: &mut [f64],
) -> Result<(), String> {
    let name = CString::new(path).map_err(|e| e.to_string())?;
    let hs = |v: &[usize]| -> Vec<hsize_t> { v.iter().map(|&x| x as hsize_t).collect() };
    let (start, count, stride) = (hs(start), hs(count), hs(stride));
    let len = [buf.len() as hsize_t];
    // Every `Held` below is declared after the guard, so it closes first,
    // still under the lock.
    let _g = netcdf_sys::libnetcdf_lock.lock();
    let dset = Held::new(unsafe { H5Dopen2(h5.id(), name.as_ptr(), H5P_DEFAULT) }, H5Dclose, "open the dataset")?;
    let file_space = Held::new(unsafe { H5Dget_space(dset.0) }, H5Sclose, "read the dataset's shape")?;
    let rank = unsafe { H5Sget_simple_extent_ndims(file_space.0) };
    if usize::try_from(rank).ok() != Some(start.len()) || count.len() != start.len() || stride.len() != start.len() {
        return Err(format!("request has the wrong rank (dataset has {rank} dimensions)"));
    }
    if rank > 0 {
        let sel = unsafe {
            H5Sselect_hyperslab(
                file_space.0,
                H5S_seloper_t::H5S_SELECT_SET,
                start.as_ptr(),
                stride.as_ptr(),
                count.as_ptr(),
                null(),
            )
        };
        if sel < 0 {
            return Err("HDF5 could not select the requested values".into());
        }
    }
    if unsafe { H5Sget_select_npoints(file_space.0) } != buf.len() as i64 {
        return Err("the selection does not match the read buffer".into());
    }
    let mem_space = Held::new(unsafe { H5Screate_simple(1, len.as_ptr(), null()) }, H5Sclose, "create a dataspace")?;
    let rc = unsafe {
        H5Dread(dset.0, *H5T_NATIVE_DOUBLE, mem_space.0, file_space.0, H5P_DEFAULT, buf.as_mut_ptr().cast())
    };
    if rc < 0 {
        return Err("HDF5 could not read the values as numbers".into());
    }
    Ok(())
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

    fn req(var: &str, start: &[usize], count: &[usize], stride: &[usize]) -> SliceRequest {
        SliceRequest { var: var.into(), start: start.to_vec(), count: count.to_vec(), stride: stride.to_vec() }
    }

    /// `plain.h5` through the HDF5 backend (forced) and through netCDF-C
    /// (what `open` keeps for it), so every read is checked on both.
    fn plain_both(dir: &std::path::Path) -> [NcFile; 2] {
        let p = fixtures::hdf5::write_plain_h5(dir);
        let h = NcFile::open_hdf5(&p).unwrap();
        assert!(h.backend_is_hdf5());
        let nc = NcFile::open(&p).unwrap();
        assert!(!nc.backend_is_hdf5());
        [h, nc]
    }

    fn floats(r: std::ops::Range<i32>) -> Vec<f32> {
        r.map(|k| k as f32).collect()
    }

    #[test]
    fn h5_reads_known_values() {
        let d = tempfile::tempdir().unwrap();
        let both = plain_both(d.path());
        for f in &both {
            let t = f.read_slice(&req("/obs/temp", &[0, 0], &[3, 4], &[1, 1])).unwrap();
            assert_eq!(t.shape, [3, 4]);
            assert_eq!(t.data, floats(0..12));
            assert_eq!(t.valid_count, 12);
            let g = f.read_slice(&req("/grid", &[1, 0, 0], &[1, 3, 5], &[1, 1, 1])).unwrap();
            assert_eq!(g.shape, [1, 3, 5]);
            assert_eq!(g.data, floats(15..30));
            let c = f.read_slice(&req("/obs/count", &[1], &[3], &[1])).unwrap();
            assert_eq!(c.data, [20.0, 30.0, 40.0]);
            assert_eq!(f.read_coord("/obs/count").unwrap(), [10.0, 20.0, 30.0, 40.0]);
        }
        // The HDF5 backend without the fast reader: H5Dread itself.
        let [h, _] = &both;
        let g = h.read_slice_via_library(&req("/grid", &[1, 0, 0], &[1, 3, 5], &[1, 1, 1])).unwrap();
        assert_eq!(g.data, floats(15..30));
        let plane = h.read_horizontal("/obs/temp", "phony_dim_0", "phony_dim_1").unwrap();
        assert_eq!(plane, (0..12).map(f64::from).collect::<Vec<_>>());

        // hidden.h5: `open` lands on the HDF5 backend.
        let hn = NcFile::open(&fixtures::hdf5::write_hidden_h5(d.path())).unwrap();
        assert!(hn.backend_is_hdf5());
        let a = hn.read_slice(&req("/a", &[0, 0], &[2, 3], &[1, 1])).unwrap();
        assert_eq!(a.data, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let b = hn.read_slice(&req("/sub/b", &[0], &[3], &[2])).unwrap();
        assert_eq!(b.data, [7.0, 9.0]);
        assert_eq!(hn.read_coord("/x").unwrap(), [10.0, 20.0, 30.0]);
    }

    #[test]
    fn h5_strided_equals_subsampled() {
        let d = tempfile::tempdir().unwrap();
        let both = plain_both(d.path());
        let r = req("/grid", &[0, 0, 1], &[2, 3, 4], &[1, 2, 3]);
        let mut want = Vec::new();
        for i in 0..2 {
            for j in [0, 2] {
                for k in [1, 4] {
                    want.push((i * 15 + j * 5 + k) as f32);
                }
            }
        }
        for f in &both {
            let full = f.read_slice(&req("/grid", &[0, 0, 0], &[2, 3, 5], &[1, 1, 1])).unwrap();
            assert_eq!(full.data, floats(0..30));
            let s = f.read_slice(&r).unwrap();
            assert_eq!(s.shape, [2, 2, 2]);
            assert_eq!(s.data, want);
            let t = f.read_slice(&req("/obs/temp", &[0, 1], &[3, 3], &[2, 2])).unwrap();
            assert_eq!(t.shape, [2, 2]);
            assert_eq!(t.data, [1.0, 3.0, 9.0, 11.0]);
        }
        // HDF5 backend: the fast reader serves chunked datasets; H5Dread
        // (in batches of every size) gives the same values.
        let [h, nc] = &both;
        assert!(h.fast_eligible("/grid"));
        assert!(!h.fast_eligible("/obs/temp"), "contiguous");
        let before = h.fast_reads();
        h.read_slice(&r).unwrap();
        assert_eq!(h.fast_reads(), before + 1);
        let cases = [
            r.clone(),
            req("/grid", &[0, 0, 0], &[2, 3, 5], &[1, 1, 1]),
            req("/grid", &[1, 1, 0], &[1, 2, 5], &[1, 1, 2]),
            req("/obs/temp", &[0, 0], &[3, 4], &[1, 3]),
        ];
        for c in &cases {
            let want = nc.read_slice(c).unwrap();
            for batch in [1, 2, 3, 4, 5, 7, 15, 29, 30, 1000] {
                let got = h.read_slice_batched(c, batch).unwrap();
                assert_eq!(got.shape, want.shape, "batch {batch} {c:?}");
                assert_eq!(got.data, want.data, "batch {batch} {c:?}");
            }
        }
    }

    #[test]
    fn h5_compound_read_errors() {
        let d = tempfile::tempdir().unwrap();
        let both = plain_both(d.path());
        for f in &both {
            assert!(!f.fast_eligible("/table"));
            assert!(f.read_slice(&req("/table", &[0], &[2], &[1])).is_err());
            assert!(f.read_slice_via_library(&req("/table", &[0], &[2], &[1])).is_err());
            assert!(f.read_coord("/table").is_err());
        }
        // H5Dread itself refuses to convert a compound to doubles.
        let [h, _] = &both;
        let e = h.read_raw("/table", &[0], &[2]).unwrap_err();
        assert!(matches!(&e, CoreError::Read { var, .. } if var == "/table"), "{e}");
        assert!(h.read_raw("/nope", &[0], &[1]).is_err());
    }

    #[test]
    fn h5_series_works() {
        use crate::{SeriesRequest, SliceService};
        use std::sync::Arc;
        let d = tempfile::tempdir().unwrap();
        let svc = SliceService::new(1 << 20, 1, Arc::new(|| {}));
        for f in plain_both(d.path()) {
            let f = Arc::new(f);
            let sr = SeriesRequest { var: "/grid".into(), point: vec![0, 1, 2], along: 0, centre: 1 };
            let mut h = svc.request_series(&f, sr);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            while h.finished().is_none() {
                assert!(std::time::Instant::now() < deadline, "series timed out");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            h.finished().unwrap().unwrap();
            let mut blocks = h.take_blocks();
            blocks.sort_by_key(|b| b.start);
            let values: Vec<f32> = blocks.into_iter().flat_map(|b| b.values).collect();
            assert_eq!(values, [7.0, 22.0], "HDF5 backend: {}", f.backend_is_hdf5());
        }
    }
}
