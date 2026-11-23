//! HDF5 chunk locations for netCDF-4 variables.
//!
//! A second, read-only HDF5 handle on a file netCDF-C already has open. It
//! answers where each chunk's bytes lie and how they were filtered, so the
//! bytes can be read and decoded without netCDF-C. Every HDF5 call is made
//! under `netcdf_sys::libnetcdf_lock`, the lock netCDF-C's calls use.

use hdf5_metno_sys::h5::{H5open, HADDR_UNDEF, haddr_t, herr_t, hsize_t};
use hdf5_metno_sys::h5d::{
    H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS, H5D_fill_value_t, H5D_layout_t, H5Dchunk_iter, H5Dclose,
    H5Dget_chunk_info_by_coord, H5Dget_create_plist, H5Dget_num_chunks, H5Dget_space, H5Dget_type,
    H5Dopen2,
};
use hdf5_metno_sys::h5f::{H5F_ACC_RDONLY, H5Fclose, H5Fget_create_plist, H5Fopen};
use hdf5_metno_sys::h5i::hid_t;
use hdf5_metno_sys::h5p::{
    H5P_DEFAULT, H5Pclose, H5Pfill_value_defined, H5Pget_chunk, H5Pget_chunk_opts,
    H5Pget_fill_value, H5Pget_filter2, H5Pget_layout, H5Pget_nfilters, H5Pget_userblock,
};
use hdf5_metno_sys::h5s::{H5Sclose, H5Sget_simple_extent_dims, H5Sget_simple_extent_ndims};
use hdf5_metno_sys::h5t::{
    H5T_IEEE_F32BE, H5T_IEEE_F32LE, H5T_IEEE_F64BE, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, H5T_class_t,
    H5T_order_t, H5Tclose, H5Tequal, H5Tget_class, H5Tget_offset, H5Tget_order, H5Tget_precision,
    H5Tget_size,
};
use hdf5_metno_sys::h5z::{H5Z_FILTER_DEFLATE, H5Z_FILTER_FLETCHER32, H5Z_FILTER_SHUFFLE};
use netcdf_sys::libnetcdf_lock;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::os::raw::{c_int, c_uint};
use std::path::Path;
use std::ptr::null_mut;
use std::sync::{Arc, Mutex};

/// Up to this many chunks the whole index is read in one `H5Dchunk_iter`
/// pass; above it, chunks are looked up one at a time as they are asked for.
const ITER_LIMIT: u64 = 1_000_000;

#[cfg(test)]
thread_local! {
    /// Replaces `ITER_LIMIT` for `H5Var::open` calls on this thread, so a
    /// test can force the one-lookup-at-a-time index.
    pub(crate) static TEST_ITER_LIMIT: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
}

/// An HDF5 identifier, closed under the lock when dropped.
struct Id {
    id: hid_t,
    close: unsafe extern "C" fn(hid_t) -> herr_t,
}

impl Id {
    /// Wraps the result of an HDF5 call that returns an identifier.
    fn new(
        id: hid_t,
        close: unsafe extern "C" fn(hid_t) -> herr_t,
        what: &str,
    ) -> Result<Id, String> {
        if id < 0 {
            Err(format!("HDF5 could not {what}"))
        } else {
            Ok(Id { id, close })
        }
    }
}

impl Drop for Id {
    fn drop(&mut self) {
        let _g = libnetcdf_lock.lock();
        unsafe { (self.close)(self.id) };
    }
}

/// A read-only HDF5 handle on a netCDF-4 or plain HDF5 file.
pub(crate) struct H5File {
    file: Arc<Id>,
    /// Userblock size: HDF5 addresses are relative to the end of it.
    base: u64,
}

impl H5File {
    /// The HDF5 file identifier. Use it only under the netCDF lock.
    pub(crate) fn id(&self) -> hid_t {
        self.file.id
    }

    /// Opens `c_path`, the path netCDF-C was given for the same file (or,
    /// for the HDF5 backend, the first path candidate HDF5 accepts).
    pub(crate) fn open(c_path: &Path) -> Result<H5File, String> {
        let c = CString::new(c_path.to_str().ok_or("path is not valid UTF-8")?)
            .map_err(|e| e.to_string())?;
        let _g = libnetcdf_lock.lock();
        if unsafe { H5open() } < 0 {
            return Err("HDF5 could not initialise".into());
        }
        // A second open must use the file close degree of the first. netCDF-C
        // sets H5F_CLOSE_WEAK, which is also what the default list gives.
        let file = Id::new(
            unsafe { H5Fopen(c.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT) },
            H5Fclose,
            "open the file",
        )?;
        let fcpl = Id::new(
            unsafe { H5Fget_create_plist(file.id) },
            H5Pclose,
            "read file creation properties",
        )?;
        let mut base: hsize_t = 0;
        if unsafe { H5Pget_userblock(fcpl.id, &mut base) } < 0 {
            return Err("HDF5 could not read the userblock size".into());
        }
        Ok(H5File {
            file: Arc::new(file),
            base,
        })
    }
}

/// A filter in a chunk's pipeline that the fast reader can undo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Filter {
    Deflate,
    Shuffle,
    Fletcher32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteOrder {
    Little,
    Big,
}

/// Where one chunk's stored (filtered) bytes lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkLoc {
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub size: u64,
    /// Bit `i` set: filter `i` of the pipeline was skipped for this chunk.
    pub filter_mask: u32,
}

/// A chunked variable whose chunks the fast reader can decode itself.
pub(crate) struct H5Var {
    /// Chunk shape, in elements.
    pub chunk: Vec<usize>,
    /// Pipeline in write order, as `H5Pget_filter2` reports it.
    pub filters: Vec<Filter>,
    pub order: ByteOrder,
    pub elem_size: usize,
    /// The dataset's extent when opened. Along an unlimited dimension it can
    /// be shorter than the netCDF length: records past it were never written.
    pub shape: Vec<usize>,
    /// The dataset's user-defined HDF5 fill value as `f64` (what reading a
    /// never-written chunk returns); None when HDF5's default applies.
    pub fill: Option<f64>,
    index: Index,
    base: u64,
    // Fields drop in declaration order: the dataset closes before the file.
    dset: Id,
    _file: Arc<Id>,
}

/// Chunk origin (in elements) to location.
enum Index {
    /// Every written chunk, keyed by its row-major linear chunk number;
    /// absent means never written. Holds the chunk-grid strides.
    Full {
        strides: Vec<u64>,
        map: HashMap<u64, ChunkLoc>,
    },
    /// Filled one lookup at a time (too many chunks to list up front).
    ByCoord(Mutex<HashMap<Vec<usize>, Option<ChunkLoc>>>),
}

/// Row-major strides, in chunks, of the chunk grid over `shape`; None if
/// the grid has more than `u64::MAX` chunks.
fn grid_strides(shape: &[usize], chunk: &[usize]) -> Option<Vec<u64>> {
    let mut strides = vec![1u64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        let n = u64::try_from(shape[d + 1].div_ceil(chunk[d + 1])).ok()?;
        strides[d] = strides[d + 1].checked_mul(n)?;
    }
    // The outermost axis must fit too.
    let n0 = u64::try_from(shape.first()?.div_ceil(*chunk.first()?)).ok()?;
    strides.first()?.checked_mul(n0)?;
    Some(strides)
}

/// Linear chunk number of the chunk at element origin `coord`; None if
/// `coord` is not a chunk origin inside `shape`.
fn linear(coord: &[usize], chunk: &[usize], shape: &[usize], strides: &[u64]) -> Option<u64> {
    if coord.len() != chunk.len() || coord.len() != shape.len() {
        return None;
    }
    let mut lin = 0u64;
    for d in 0..coord.len() {
        if !coord[d].is_multiple_of(chunk[d]) || coord[d] >= shape[d] {
            return None;
        }
        let k = u64::try_from(coord[d] / chunk[d]).ok()?;
        lin = lin.checked_add(k.checked_mul(strides[d])?)?;
    }
    Some(lin)
}

impl H5Var {
    /// `var_path` is the variable's full netCDF path, e.g. `/ocean/uo`.
    ///
    /// Ok(None): not eligible (contiguous/compact, other filters,
    /// DONT_FILTER_PARTIAL_CHUNKS set, or not a plain number type).
    pub(crate) fn open(f: &H5File, var_path: &str) -> Result<Option<H5Var>, String> {
        #[cfg(test)]
        let limit = TEST_ITER_LIMIT.get().unwrap_or(ITER_LIMIT);
        #[cfg(not(test))]
        let limit = ITER_LIMIT;
        Self::open_with_limit(f, var_path, limit)
    }

    fn open_with_limit(
        f: &H5File,
        var_path: &str,
        iter_limit: u64,
    ) -> Result<Option<H5Var>, String> {
        let c = CString::new(var_path).map_err(|e| e.to_string())?;
        let _g = libnetcdf_lock.lock();
        let dset = Id::new(
            unsafe { H5Dopen2(f.file.id, c.as_ptr(), H5P_DEFAULT) },
            H5Dclose,
            "open the variable",
        )?;

        let dcpl = Id::new(
            unsafe { H5Dget_create_plist(dset.id) },
            H5Pclose,
            "read the variable's creation properties",
        )?;
        if unsafe { H5Pget_layout(dcpl.id) } != H5D_layout_t::H5D_CHUNKED {
            return Ok(None);
        }
        let mut opts: c_uint = 0;
        if unsafe { H5Pget_chunk_opts(dcpl.id, &mut opts) } < 0 {
            return Err("HDF5 could not read the chunk options".into());
        }
        if opts & H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS != 0 {
            return Ok(None);
        }
        let Some(filters) = filters(&dcpl)? else {
            return Ok(None);
        };

        let dtype = Id::new(
            unsafe { H5Dget_type(dset.id) },
            H5Tclose,
            "read the variable's type",
        )?;
        let class = unsafe { H5Tget_class(dtype.id) };
        if !matches!(class, H5T_class_t::H5T_INTEGER | H5T_class_t::H5T_FLOAT) {
            return Ok(None);
        }
        let elem_size = unsafe { H5Tget_size(dtype.id) };
        if elem_size == 0 {
            return Err("HDF5 could not read the type size".into());
        }
        if !standard_layout(dtype.id, class, elem_size) {
            return Ok(None);
        }
        let order = match unsafe { H5Tget_order(dtype.id) } {
            H5T_order_t::H5T_ORDER_LE => ByteOrder::Little,
            H5T_order_t::H5T_ORDER_BE => ByteOrder::Big,
            H5T_order_t::H5T_ORDER_NONE if elem_size == 1 => ByteOrder::Little,
            _ => return Ok(None),
        };

        let space = Id::new(
            unsafe { H5Dget_space(dset.id) },
            H5Sclose,
            "read the variable's shape",
        )?;
        let rank = unsafe { H5Sget_simple_extent_ndims(space.id) };
        if rank <= 0 {
            return Ok(None);
        }
        let mut dims = vec![0 as hsize_t; rank as usize];
        if unsafe { H5Sget_simple_extent_dims(space.id, dims.as_mut_ptr(), null_mut()) } != rank {
            return Err("HDF5 could not read the variable's shape".into());
        }
        let mut cdims = vec![0 as hsize_t; rank as usize];
        if unsafe { H5Pget_chunk(dcpl.id, rank, cdims.as_mut_ptr()) } != rank {
            return Err("HDF5 could not read the chunk shape".into());
        }
        if cdims.contains(&0) {
            return Err("HDF5 reported an empty chunk shape".into());
        }
        let fill = user_fill(&dcpl)?;
        let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let chunk: Vec<usize> = cdims.iter().map(|&d| d as usize).collect();

        let mut nchunks: hsize_t = 0;
        if unsafe { H5Dget_num_chunks(dset.id, space.id, &mut nchunks) } < 0 {
            return Err("HDF5 could not count the chunks".into());
        }
        let strides = grid_strides(&shape, &chunk);
        let index = match strides {
            Some(strides) if nchunks <= iter_limit => {
                let mut ctx = IterCtx {
                    chunk: &chunk,
                    shape: &shape,
                    strides: &strides,
                    base: f.base,
                    map: HashMap::new(),
                    err: None,
                };
                let op_data = (&raw mut ctx).cast::<c_void>();
                let rc =
                    unsafe { H5Dchunk_iter(dset.id, H5P_DEFAULT, Some(collect_chunk), op_data) };
                if let Some(e) = ctx.err {
                    return Err(e);
                }
                if rc < 0 {
                    return Err("HDF5 could not list the chunks".into());
                }
                let map = ctx.map;
                Index::Full { strides, map }
            }
            _ => Index::ByCoord(Mutex::new(HashMap::new())),
        };

        Ok(Some(H5Var {
            chunk,
            filters,
            order,
            elem_size,
            shape,
            fill,
            index,
            base: f.base,
            dset,
            _file: Arc::clone(&f.file),
        }))
    }

    /// `coord` is the chunk's element origin (multiples of `chunk`).
    /// Ok(None): the chunk was never written.
    pub(crate) fn chunk_loc(&self, coord: &[usize]) -> Result<Option<ChunkLoc>, String> {
        let not_origin = || format!("{coord:?} is not a chunk origin of this variable");
        match &self.index {
            Index::Full { strides, map } => {
                let lin =
                    linear(coord, &self.chunk, &self.shape, strides).ok_or_else(not_origin)?;
                Ok(map.get(&lin).copied())
            }
            Index::ByCoord(cache) => {
                let valid = coord.len() == self.chunk.len()
                    && coord
                        .iter()
                        .zip(&self.chunk)
                        .zip(&self.shape)
                        .all(|((&c, &k), &n)| c % k == 0 && c < n);
                if !valid {
                    return Err(not_origin());
                }
                if let Some(&hit) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(coord) {
                    return Ok(hit);
                }
                let loc = self.query(coord)?;
                cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(coord.to_vec(), loc);
                Ok(loc)
            }
        }
    }

    /// Whether chunks are looked up one at a time (`Index::ByCoord`).
    #[cfg(test)]
    pub(crate) fn uses_lookup_index(&self) -> bool {
        matches!(self.index, Index::ByCoord(_))
    }

    /// One chunk's location straight from HDF5.
    fn query(&self, coord: &[usize]) -> Result<Option<ChunkLoc>, String> {
        let offset: Vec<hsize_t> = coord.iter().map(|&c| c as hsize_t).collect();
        let (mut mask, mut addr, mut size): (c_uint, haddr_t, hsize_t) = (0, 0, 0);
        let rc = {
            let _g = libnetcdf_lock.lock();
            unsafe {
                H5Dget_chunk_info_by_coord(
                    self.dset.id,
                    offset.as_ptr(),
                    &mut mask,
                    &mut addr,
                    &mut size,
                )
            }
        };
        if rc < 0 {
            return Err(format!("HDF5 could not locate chunk {coord:?}"));
        }
        chunk_loc(self.base, mask, addr, size)
    }
}

/// Ok(None) when the chunk has no storage (never written); Err if the
/// absolute offset does not fit in 64 bits.
fn chunk_loc(
    base: u64,
    filter_mask: c_uint,
    addr: haddr_t,
    size: hsize_t,
) -> Result<Option<ChunkLoc>, String> {
    if addr == HADDR_UNDEF || size == 0 {
        return Ok(None);
    }
    let offset = base
        .checked_add(addr)
        .ok_or("chunk address is past the end of any file")?;
    Ok(Some(ChunkLoc {
        offset,
        size,
        filter_mask,
    }))
}

/// Whether every bit of an element of type `t` is its value in the
/// standard layout the decoder assumes: integers without padding, IEEE
/// single or double floats. (netCDF types always are; a plain HDF5 file
/// can hold others, which only HDF5 converts correctly.) Caller holds the lock.
fn standard_layout(t: hid_t, class: H5T_class_t, size: usize) -> bool {
    match class {
        H5T_class_t::H5T_INTEGER => unsafe { H5Tget_offset(t) == 0 && H5Tget_precision(t) == size * 8 },
        H5T_class_t::H5T_FLOAT => [*H5T_IEEE_F32LE, *H5T_IEEE_F32BE, *H5T_IEEE_F64LE, *H5T_IEEE_F64BE]
            .into_iter()
            .any(|s| unsafe { H5Tequal(t, s) } > 0),
        _ => false,
    }
}

/// The dataset's user-defined fill value converted by HDF5 to `f64`, as
/// netCDF-C converts what it reads; None when no user value is set (HDF5's
/// default). Caller holds the lock.
fn user_fill(dcpl: &Id) -> Result<Option<f64>, String> {
    let mut status = H5D_fill_value_t::H5D_FILL_VALUE_ERROR;
    if unsafe { H5Pfill_value_defined(dcpl.id, &mut status) } < 0 {
        return Err("HDF5 could not read the fill value setting".into());
    }
    if status != H5D_fill_value_t::H5D_FILL_VALUE_USER_DEFINED {
        return Ok(None);
    }
    let mut value = 0.0f64;
    let rc = unsafe { H5Pget_fill_value(dcpl.id, *H5T_NATIVE_DOUBLE, (&raw mut value).cast()) };
    if rc < 0 {
        return Err("HDF5 could not read the fill value".into());
    }
    Ok(Some(value))
}

/// The filter pipeline, or None if it holds a filter the fast reader cannot
/// undo. Caller holds the lock.
fn filters(dcpl: &Id) -> Result<Option<Vec<Filter>>, String> {
    let n = unsafe { H5Pget_nfilters(dcpl.id) };
    if n < 0 {
        return Err("HDF5 could not read the filter pipeline".into());
    }
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n as c_uint {
        let (mut flags, mut nvals): (c_uint, usize) = (0, 0);
        let id = unsafe {
            H5Pget_filter2(
                dcpl.id,
                i,
                &mut flags,
                &mut nvals,
                null_mut(),
                0,
                null_mut(),
                null_mut(),
            )
        };
        out.push(match id {
            H5Z_FILTER_DEFLATE => Filter::Deflate,
            H5Z_FILTER_SHUFFLE => Filter::Shuffle,
            H5Z_FILTER_FLETCHER32 => Filter::Fletcher32,
            x if x < 0 => return Err("HDF5 could not read a filter".into()),
            _ => return Ok(None),
        });
    }
    Ok(Some(out))
}

struct IterCtx<'a> {
    chunk: &'a [usize],
    shape: &'a [usize],
    strides: &'a [u64],
    base: u64,
    map: HashMap<u64, ChunkLoc>,
    /// Set when a chunk cannot be recorded; iteration then stops.
    err: Option<String>,
}

/// `H5Dchunk_iter` callback: records one chunk in the `IterCtx` behind
/// `op_data`. `offset` is the chunk origin in elements.
unsafe extern "C" fn collect_chunk(
    offset: *const hsize_t,
    filter_mask: c_uint,
    addr: haddr_t,
    size: hsize_t,
    op_data: *mut c_void,
) -> c_int {
    let ctx = unsafe { &mut *op_data.cast::<IterCtx>() };
    let origin: Vec<usize> = unsafe { std::slice::from_raw_parts(offset, ctx.chunk.len()) }
        .iter()
        .map(|&o| o as usize)
        .collect();
    let Some(lin) = linear(&origin, ctx.chunk, ctx.shape, ctx.strides) else {
        ctx.err = Some(format!(
            "HDF5 listed a chunk at {origin:?}, outside the variable"
        ));
        return -1; // stop: H5Dchunk_iter returns failure
    };
    match chunk_loc(ctx.base, filter_mask, addr, size) {
        Ok(Some(loc)) => {
            ctx.map.insert(lin, loc);
        }
        Ok(None) => {}
        Err(e) => {
            ctx.err = Some(e);
            return -1;
        }
    }
    0 // H5_ITER_CONT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cf::Unpack;
    use crate::fixtures;
    use crate::{NcFile, SliceRequest};

    #[test]
    fn second_open_alongside_netcdf() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let f = H5File::open(&nc.c_path).unwrap();
        let files = {
            let _g = netcdf_sys::libnetcdf_lock.lock();
            unsafe {
                hdf5_metno_sys::h5f::H5Fget_obj_count(
                    hdf5_metno_sys::h5f::H5F_OBJ_ALL as hdf5_metno_sys::h5i::hid_t,
                    hdf5_metno_sys::h5f::H5F_OBJ_FILE,
                )
            }
        };
        assert!(files >= 2, "HDF5 sees {files} open file ids");
        drop(f);
        let req = SliceRequest {
            var: "/ocean/uo".into(),
            start: vec![0, 0, 0],
            count: vec![1, 10, 20],
            stride: vec![1, 1, 1],
        };
        let s = nc.read_slice(&req).unwrap();
        assert_eq!(s.data.len(), 200);
        assert_eq!(s.valid_count, 170);
    }

    #[test]
    fn uo_is_eligible() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let f = H5File::open(&nc.c_path).unwrap();
        let v = H5Var::open(&f, "/ocean/uo")
            .unwrap()
            .expect("uo is eligible");
        assert_eq!(v.chunk, vec![1, 10, 20]);
        assert_eq!(v.filters, vec![Filter::Deflate]);
        assert_eq!(v.order, ByteOrder::Little);
        assert_eq!(v.elem_size, 4);
    }

    #[test]
    fn raw_chunk_inflates_to_library_values() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let f = H5File::open(&nc.c_path).unwrap();
        let v = H5Var::open(&f, "/ocean/uo").unwrap().unwrap();
        let loc = v.chunk_loc(&[3, 0, 0]).unwrap().expect("chunk was written");
        assert_eq!(loc.filter_mask, 0);

        let file = std::fs::File::open(&nc.c_path).unwrap();
        let mut packed = vec![0u8; loc.size as usize];
        super::super::read_exact_at(&file, loc.offset, &mut packed).unwrap();
        let mut raw = vec![0u8; 200 * 4];
        let n = libdeflater::Decompressor::new()
            .zlib_decompress(&packed, &mut raw)
            .unwrap();
        assert_eq!(n, raw.len());

        let unpack = Unpack::for_var(nc.meta.var("/ocean/uo").unwrap());
        let ours: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| unpack.apply(f64::from(f32::from_le_bytes(b))))
            .collect();
        let req = SliceRequest {
            var: "/ocean/uo".into(),
            start: vec![3, 0, 0],
            count: vec![1, 10, 20],
            stride: vec![1, 1, 1],
        };
        let lib = nc.read_slice(&req).unwrap().data;
        assert_eq!(ours.len(), lib.len());
        for (i, (a, b)) in ours.iter().zip(&lib).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{i}: {a} != {b}"
            );
        }
        assert!(ours.iter().any(|x| !x.is_nan()));
    }

    #[test]
    fn contiguous_not_eligible() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_bare(dir.path())).unwrap();
        let f = H5File::open(&nc.c_path).unwrap();
        assert!(H5Var::open(&f, "/data").unwrap().is_none());
    }

    /// The one-pass index and per-chunk lookups give the same locations,
    /// with origins in elements (chunks wider than one element per axis).
    #[test]
    fn iter_and_by_coord_agree() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grid.nc");
        {
            let mut nc = netcdf::create(&path).unwrap();
            nc.add_dimension("y", 6).unwrap();
            nc.add_dimension("x", 8).unwrap();
            let mut v = nc.add_variable::<i16>("z", &["y", "x"]).unwrap();
            v.set_chunking(&[2, 4]).unwrap();
            v.set_compression(1, true).unwrap();
            // Only the first chunk row is written.
            let vals: Vec<i16> = (0..16).collect();
            v.put_values(&vals, (0..2, 0..8)).unwrap();
        }
        let nc = NcFile::open(&path).unwrap();
        let f = H5File::open(&nc.c_path).unwrap();
        let full = H5Var::open_with_limit(&f, "/z", u64::MAX).unwrap().unwrap();
        let lazy = H5Var::open_with_limit(&f, "/z", 0).unwrap().unwrap();
        assert!(matches!(full.index, Index::Full { .. }));
        assert!(matches!(lazy.index, Index::ByCoord(_)));
        assert_eq!(full.filters, vec![Filter::Shuffle, Filter::Deflate]);
        assert_eq!(full.elem_size, 2);
        for y in [0, 2, 4] {
            for x in [0, 4] {
                let a = full.chunk_loc(&[y, x]).unwrap();
                assert_eq!(a, lazy.chunk_loc(&[y, x]).unwrap(), "chunk [{y}, {x}]");
                assert_eq!(
                    lazy.chunk_loc(&[y, x]).unwrap(),
                    a,
                    "cached chunk [{y}, {x}]"
                );
                assert_eq!(a.is_some(), y == 0, "chunk [{y}, {x}] written");
            }
        }
        assert!(full.chunk_loc(&[1, 0]).is_err());
        assert!(full.chunk_loc(&[6, 0]).is_err());
    }

    #[test]
    fn chunk_address_overflow_is_an_error() {
        assert!(chunk_loc(u64::MAX, 0, 1, 10).is_err());
        assert_eq!(chunk_loc(0, 0, HADDR_UNDEF, 10), Ok(None));
        let loc = ChunkLoc {
            offset: 612,
            size: 10,
            filter_mask: 2,
        };
        assert_eq!(chunk_loc(512, 2, 100, 10), Ok(Some(loc)));
    }

    #[test]
    fn linear_chunk_numbers_are_row_major() {
        let (shape, chunk) = ([7, 13, 17], [2, 5, 6]); // a 4 x 3 x 3 grid
        let s = grid_strides(&shape, &chunk).unwrap();
        assert_eq!(s, vec![9, 3, 1]);
        assert_eq!(linear(&[0, 0, 0], &chunk, &shape, &s), Some(0));
        assert_eq!(linear(&[6, 10, 12], &chunk, &shape, &s), Some(35));
        assert_eq!(linear(&[1, 0, 0], &chunk, &shape, &s), None);
        assert_eq!(linear(&[8, 0, 0], &chunk, &shape, &s), None);
        assert!(grid_strides(&[usize::MAX, usize::MAX, 2], &[1, 1, 1]).is_none());
    }
}
