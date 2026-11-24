//! Metadata-only open of a netCDF/HDF5 file. No variable data is read.

use crate::fast::h5::H5File;
use crate::model::*;
use netcdf::AttributeValue;
use netcdf::types::{FloatType, IntType, NcVariableType};
use std::ffi::CString;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};

/// Where an open file's variable data is read from.
pub(crate) enum Backend {
    Netcdf(netcdf::File),
    /// Plain HDF5 that netCDF-C refused, or hid some datasets of.
    Hdf5(H5File),
}

/// An open file: its structure plus the handle used for later reads.
pub struct NcFile {
    pub meta: Dataset,
    /// The fast chunk reader's handles and caches; dropped (closed) before
    /// the backend's handle.
    pub(crate) fast: crate::fast::FastState,
    pub(crate) backend: Backend,
    /// The exact path the backend opened (the real path, or its 8.3 form on
    /// Windows if the real one was refused), so a second HDF5 open names the same file.
    pub(crate) c_path: PathBuf,
    open_id: u64,
    /// Vector pairs the user made (none at open). They
    /// last for the session only.
    pub(crate) pairs: std::sync::RwLock<Vec<std::sync::Arc<crate::vector::VectorPair>>>,
    /// Calculated variables the user made (none at open), in the order
    /// made. They last for the session only.
    pub(crate) derived: std::sync::RwLock<Vec<std::sync::Arc<crate::calc::Derived>>>,
    /// Goes up whenever a calculated variable is added, changed or removed,
    /// so a read that started before cannot be cached after.
    pub(crate) derived_epoch: std::sync::atomic::AtomicU64,
    /// Resolved staggering and rotation per pair path, with the pair they
    /// were resolved for (a pair replaced at the same path resolves again).
    pub(crate) resolved: std::sync::Mutex<crate::vector::ResolvedCache>,
}

/// Source of [`NcFile::open_id`] values.
static NEXT_OPEN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl NcFile {
    /// Unique per successful `open` call within this process. Unlike the
    /// path, it changes when a file is closed, rewritten and reopened, so it
    /// is safe to key caches by.
    pub fn open_id(&self) -> u64 {
        self.open_id
    }

    /// Whether the file is read through the HDF5 library rather than
    /// netCDF-C (`meta.format` is then [`FileFormat::Hdf5`]).
    pub fn backend_is_hdf5(&self) -> bool {
        matches!(self.backend, Backend::Hdf5(_))
    }

    /// The netCDF-C handle; an error for files read through HDF5 (only
    /// their numbers can be read, through [`NcFile::read_slice`] and co.).
    pub(crate) fn netcdf(&self) -> Result<&netcdf::File, String> {
        match &self.backend {
            Backend::Netcdf(f) => Ok(f),
            Backend::Hdf5(_) => Err("text values of plain HDF5 files cannot be read yet".into()),
        }
    }

    /// Opens `path` and reads its structure (dimensions, variables,
    /// attributes, groups). Variable data is not touched.
    ///
    /// netCDF-C is tried first. When it refuses the file, or when the HDF5
    /// library finds more plottable datasets than netCDF-C shows numeric
    /// variables (it hides some), the file is read through HDF5.
    pub fn open(path: &Path) -> Result<NcFile, CoreError> {
        Self::open_netcdf(path).or_else(|nc_err| Self::open_hdf5(path).map_err(|_| nc_err))
    }

    fn open_netcdf(path: &Path) -> Result<NcFile, CoreError> {
        let open_err = |msg: String| CoreError::Open { path: path.to_path_buf(), msg };
        // The real path first; the 8.3 form only if netCDF-C refuses it.
        // `cpath` is whichever candidate worked, for every later open.
        let mut last_err = None;
        let mut opened = None;
        for cand in c_path_candidates(path) {
            let tried = netcdf::open(&cand)
                .map_err(|e| e.to_string())
                .and_then(|file| RawHandle::open(&cand).map(|raw| (file, raw)));
            match tried {
                Ok((file, raw)) => {
                    opened = Some((cand, file, raw));
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        let Some((cpath, file, raw)) = opened else {
            return Err(open_err(last_err.unwrap_or_else(|| "file could not be opened".into())));
        };
        let format = raw.format().map_err(open_err)?;
        let file_size = std::fs::metadata(path).map_err(|e| open_err(e.to_string()))?.len();

        // The netcdf crate panics on some metadata errors; a failed open must
        // not take the app down. `raw` is dropped (handle closed) on return.
        let mut root = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_root(&file)))
            .map_err(|_| open_err("file metadata could not be read (corrupt or unsupported)".into()))?;
        if matches!(format, FileFormat::Netcdf4 | FileFormat::Netcdf4Classic) {
            raw.fill_storage(&mut root);
            // Does netCDF-C hide datasets? A cheap count first; the full
            // HDF5 metadata only when the backend switches. Any error or
            // panic here means "not hidden": netCDF-C stays.
            let hdf5 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let h5 = H5File::open(&cpath).ok()?;
                let count = crate::h5meta::count_plottable(&h5).ok()?;
                if !crate::h5meta::hides_datasets(&root, count) {
                    return None;
                }
                let h5_root = crate::h5meta::read_h5_meta(&h5, path).ok()?;
                Some((h5, h5_root))
            }));
            if let Ok(Some((h5, h5_root))) = hdf5 {
                let meta = Dataset { path: path.to_path_buf(), file_size, format: FileFormat::Hdf5, root: h5_root };
                return Ok(Self::assemble(meta, Backend::Hdf5(h5), cpath));
            }
        }
        let meta = Dataset { path: path.to_path_buf(), file_size, format, root };
        Ok(Self::assemble(meta, Backend::Netcdf(file), cpath))
    }

    /// Opens `path` through the HDF5 library alone: the first path
    /// candidate HDF5 accepts.
    pub(crate) fn open_hdf5(path: &Path) -> Result<NcFile, CoreError> {
        let open_err = |msg: String| CoreError::Open { path: path.to_path_buf(), msg };
        let mut last_err = None;
        for cand in c_path_candidates(path) {
            match H5File::open(&cand) {
                Ok(h5) => {
                    let root = crate::h5meta::read_h5_meta(&h5, path).map_err(open_err)?;
                    let file_size = std::fs::metadata(path).map_err(|e| open_err(e.to_string()))?.len();
                    let meta = Dataset { path: path.to_path_buf(), file_size, format: FileFormat::Hdf5, root };
                    return Ok(Self::assemble(meta, Backend::Hdf5(h5), cand));
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(open_err(last_err.unwrap_or_else(|| "file could not be opened".into())))
    }

    fn assemble(meta: Dataset, backend: Backend, c_path: PathBuf) -> NcFile {
        let open_id = NEXT_OPEN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // No pair is made at open: the user pairs variables themselves.
        NcFile {
            meta,
            fast: Default::default(),
            backend,
            c_path,
            open_id,
            pairs: Default::default(),
            derived: Default::default(),
            derived_epoch: Default::default(),
            resolved: Default::default(),
        }
    }
}

/// The paths to try with netCDF-C (or HDF5), in order: the real path, then (Windows,
/// non-ASCII only) its ASCII-only short (8.3) form, in case netCDF-C cannot
/// open the real one through the ANSI code page.
fn c_path_candidates(p: &Path) -> Vec<PathBuf> {
    let mut out = vec![p.to_path_buf()];
    if p.to_str().is_some_and(str::is_ascii) {
        return out;
    }
    #[cfg(windows)]
    if let Some(short) = short_path(p) {
        out.push(short);
    }
    out
}

/// The 8.3 form of `p`, if the file exists, has one, and it is ASCII.
#[cfg(windows)]
fn short_path(p: &Path) -> Option<PathBuf> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    unsafe extern "system" {
        fn GetShortPathNameW(long: *const u16, short: *mut u16, len: u32) -> u32;
    }
    let long: Vec<u16> = p.as_os_str().encode_wide().chain(Some(0)).collect();
    let needed = unsafe { GetShortPathNameW(long.as_ptr(), std::ptr::null_mut(), 0) };
    if needed == 0 {
        return None;
    }
    let mut buf = vec![0u16; needed as usize];
    let n = unsafe { GetShortPathNameW(long.as_ptr(), buf.as_mut_ptr(), needed) } as usize;
    if n == 0 || n >= buf.len() {
        return None;
    }
    let short = PathBuf::from(std::ffi::OsString::from_wide(&buf[..n]));
    short.to_str().is_some_and(str::is_ascii).then_some(short)
}

// ---- structure via the netcdf crate ---------------------------------------

fn read_root(file: &netcdf::File) -> Group {
    let dims = file.dimensions().map(|d| dim_of(&d)).collect();
    let vars = file.variables().map(|v| var_of(&v, "/")).collect();
    let attrs = file.attributes().filter_map(|a| attr_of(&a)).collect();
    let groups = file
        .groups()
        .map(|it| it.map(|g| read_group(&g, "/")).collect())
        .unwrap_or_default();
    Group { name: "/".into(), path: "/".into(), dims, vars, attrs, groups }
}

fn read_group(g: &netcdf::Group, parent_path: &str) -> Group {
    let name = g.name();
    let path = join(parent_path, &name);
    Group {
        dims: g.dimensions().map(|d| dim_of(&d)).collect(),
        vars: g.variables().map(|v| var_of(&v, &path)).collect(),
        attrs: g.attributes().filter_map(|a| attr_of(&a)).collect(),
        groups: g.groups().map(|s| read_group(&s, &path)).collect(),
        name,
        path,
    }
}

fn join(parent: &str, name: &str) -> String {
    if parent == "/" { format!("/{name}") } else { format!("{parent}/{name}") }
}

fn dim_of(d: &netcdf::Dimension) -> Dim {
    Dim { name: d.name(), len: d.len(), unlimited: d.is_unlimited() }
}

fn var_of(v: &netcdf::Variable, group_path: &str) -> VarInfo {
    let name = v.name();
    VarInfo {
        path: join(group_path, &name),
        dims: v.dimensions().iter().map(|d| d.name()).collect(),
        shape: v.dimensions().iter().map(|d| d.len()).collect(),
        dtype: dtype_of(&v.vartype()),
        attrs: v.attributes().filter_map(|a| attr_of(&a)).collect(),
        storage: Storage::Contiguous,
        name,
    }
}

fn dtype_of(t: &NcVariableType) -> DType {
    match t {
        NcVariableType::Int(IntType::I8) => DType::I8,
        NcVariableType::Int(IntType::U8) => DType::U8,
        NcVariableType::Int(IntType::I16) => DType::I16,
        NcVariableType::Int(IntType::U16) => DType::U16,
        NcVariableType::Int(IntType::I32) => DType::I32,
        NcVariableType::Int(IntType::U32) => DType::U32,
        NcVariableType::Int(IntType::I64) => DType::I64,
        NcVariableType::Int(IntType::U64) => DType::U64,
        NcVariableType::Float(FloatType::F32) => DType::F32,
        NcVariableType::Float(FloatType::F64) => DType::F64,
        NcVariableType::Char => DType::Char,
        NcVariableType::String => DType::Str,
        NcVariableType::Compound(_) => DType::Other("compound".into()),
        NcVariableType::Opaque(_) => DType::Other("opaque".into()),
        NcVariableType::Enum(_) => DType::Other("enum".into()),
        NcVariableType::Vlen(_) => DType::Other("vlen".into()),
    }
}

/// Attributes whose value cannot be read are omitted.
fn attr_of(a: &netcdf::Attribute) -> Option<Attr> {
    use AttributeValue as V;
    let ints = |v: Vec<i64>| AttrValue::Ints(v);
    let value = match a.value().ok()? {
        V::Uchar(x) => ints(vec![x.into()]),
        V::Uchars(x) => ints(x.into_iter().map(i64::from).collect()),
        V::Schar(x) => ints(vec![x.into()]),
        V::Schars(x) => ints(x.into_iter().map(i64::from).collect()),
        V::Ushort(x) => ints(vec![x.into()]),
        V::Ushorts(x) => ints(x.into_iter().map(i64::from).collect()),
        V::Short(x) => ints(vec![x.into()]),
        V::Shorts(x) => ints(x.into_iter().map(i64::from).collect()),
        V::Uint(x) => ints(vec![x.into()]),
        V::Uints(x) => ints(x.into_iter().map(i64::from).collect()),
        V::Int(x) => ints(vec![x.into()]),
        V::Ints(x) => ints(x.into_iter().map(i64::from).collect()),
        V::Ulonglong(x) => u64s_to_attr(vec![x]),
        V::Ulonglongs(x) => u64s_to_attr(x),
        V::Longlong(x) => ints(vec![x]),
        V::Longlongs(x) => ints(x),
        V::Float(x) => AttrValue::Floats(vec![x.into()]),
        V::Floats(x) => AttrValue::Floats(x.into_iter().map(f64::from).collect()),
        V::Double(x) => AttrValue::Floats(vec![x]),
        V::Doubles(x) => AttrValue::Floats(x),
        V::Str(s) => AttrValue::Text(s),
        V::Strs(s) => AttrValue::Texts(s),
    };
    Some(Attr { name: a.name().to_string(), value })
}

/// Unsigned 64-bit values stay exact `Ints` when they all fit in `i64`;
/// otherwise the whole attribute becomes `Floats` rather than wrapping.
pub(crate) fn u64s_to_attr(v: Vec<u64>) -> AttrValue {
    match v.iter().map(|&x| i64::try_from(x)).collect::<Result<Vec<_>, _>>() {
        Ok(ints) => AttrValue::Ints(ints),
        Err(_) => AttrValue::Floats(v.into_iter().map(|x| x as f64).collect()),
    }
}

// ---- format and storage via the raw C API ---------------------------------
//
// The netcdf crate exposes neither the file format nor deflate/filter info,
// so a second metadata-only handle answers those. Every raw call is made
// under the same lock the netcdf crate uses.

struct RawHandle(c_int);

impl RawHandle {
    fn open(path: &Path) -> Result<RawHandle, String> {
        let c = CString::new(path.to_str().ok_or("path is not valid UTF-8")?)
            .map_err(|e| e.to_string())?;
        let _g = netcdf_sys::libnetcdf_lock.lock();
        let mut ncid = 0;
        let rc = unsafe { netcdf_sys::nc_open(c.as_ptr(), netcdf_sys::NC_NOWRITE, &mut ncid) };
        if rc == netcdf_sys::NC_NOERR { Ok(RawHandle(ncid)) } else { Err(nc_message(rc)) }
    }

    fn format(&self) -> Result<FileFormat, String> {
        let _g = netcdf_sys::libnetcdf_lock.lock();
        let mut f = 0;
        let rc = unsafe { netcdf_sys::nc_inq_format(self.0, &mut f) };
        if rc != netcdf_sys::NC_NOERR {
            return Err(nc_message(rc));
        }
        match f {
            netcdf_sys::NC_FORMAT_CLASSIC => Ok(FileFormat::Classic),
            netcdf_sys::NC_FORMAT_64BIT_OFFSET => Ok(FileFormat::Offset64),
            netcdf_sys::NC_FORMAT_CDF5 => Ok(FileFormat::Cdf5),
            netcdf_sys::NC_FORMAT_NETCDF4 => Ok(FileFormat::Netcdf4),
            netcdf_sys::NC_FORMAT_NETCDF4_CLASSIC => Ok(FileFormat::Netcdf4Classic),
            other => Err(format!("unknown netCDF format code {other}")),
        }
    }

    /// Sets `storage` on every variable of `g` and its subgroups.
    fn fill_storage(&self, g: &mut Group) {
        let _guard = netcdf_sys::libnetcdf_lock.lock();
        self.fill_group(g);
    }

    fn fill_group(&self, g: &mut Group) {
        let gid = if g.path == "/" {
            Some(self.0)
        } else {
            CString::new(g.path.as_str()).ok().and_then(|p| {
                let mut gid = 0;
                let rc = unsafe { netcdf_sys::nc_inq_grp_full_ncid(self.0, p.as_ptr(), &mut gid) };
                (rc == netcdf_sys::NC_NOERR).then_some(gid)
            })
        };
        for v in &mut g.vars {
            v.storage = gid.and_then(|gid| var_storage(gid, v)).unwrap_or(Storage::Unknown);
        }
        for sub in &mut g.groups {
            self.fill_group(sub);
        }
    }
}

impl Drop for RawHandle {
    fn drop(&mut self) {
        let _g = netcdf_sys::libnetcdf_lock.lock();
        unsafe { netcdf_sys::nc_close(self.0) };
    }
}

/// Caller holds the netCDF lock.
fn var_storage(gid: c_int, v: &VarInfo) -> Option<Storage> {
    use netcdf_sys::*;
    let name = CString::new(v.name.as_str()).ok()?;
    let mut varid = 0;
    if unsafe { nc_inq_varid(gid, name.as_ptr(), &mut varid) } != NC_NOERR {
        return None;
    }
    let mut kind = 0;
    let mut chunk = vec![0usize; v.shape.len()];
    if unsafe { nc_inq_var_chunking(gid, varid, &mut kind, chunk.as_mut_ptr()) } != NC_NOERR {
        return None;
    }
    match kind {
        NC_CONTIGUOUS => Some(Storage::Contiguous),
        NC_COMPACT => Some(Storage::Compact),
        NC_CHUNKED => {
            let (mut shuffle, mut deflate, mut level) = (0, 0, 0);
            let rc = unsafe { nc_inq_var_deflate(gid, varid, &mut shuffle, &mut deflate, &mut level) };
            let mut nfilters = 0usize;
            let rc2 = unsafe {
                nc_inq_var_filter_ids(gid, varid, &mut nfilters, std::ptr::null_mut())
            };
            let compressed = (rc == NC_NOERR && deflate != 0) || (rc2 == NC_NOERR && nfilters > 0);
            Some(Storage::Chunked { chunk, compressed })
        }
        _ => Some(Storage::Unknown),
    }
}

fn nc_message(code: c_int) -> String {
    let p = unsafe { netcdf_sys::nc_strerror(code) };
    if p.is_null() {
        return format!("netCDF error {code}");
    }
    unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u64_attrs_do_not_wrap() {
        assert_eq!(u64s_to_attr(vec![5, 7]), AttrValue::Ints(vec![5, 7]));
        assert_eq!(u64s_to_attr(vec![i64::MAX as u64]), AttrValue::Ints(vec![i64::MAX]));
        let big = u64::MAX - 1;
        assert_eq!(u64s_to_attr(vec![1, big]), AttrValue::Floats(vec![1.0, big as f64]));
    }

    #[test]
    fn real_path_tried_first() {
        let ascii = Path::new("data/file.nc");
        assert_eq!(c_path_candidates(ascii), vec![ascii.to_path_buf()]);
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("Ölçüm 测试").join("a.nc");
        std::fs::create_dir_all(real.parent().unwrap()).unwrap();
        std::fs::write(&real, b"x").unwrap();
        let c = c_path_candidates(&real);
        assert_eq!(c[0], real);
        #[cfg(windows)]
        if c.len() > 1 {
            assert_eq!(c.len(), 2);
            assert!(c[1].to_str().unwrap().is_ascii());
        }
        #[cfg(not(windows))]
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn non_ascii_path_reads() {
        use crate::{NcFile, SliceRequest};
        let tmp = tempfile::tempdir().unwrap();
        let ascii_dir = tmp.path().join("plain");
        let uni_dir = tmp.path().join("Ölçüm 测试");
        std::fs::create_dir_all(&ascii_dir).unwrap();
        std::fs::create_dir_all(&uni_dir).unwrap();
        let a = crate::fixtures::vectors::write_arome_like(&ascii_dir);
        let u = uni_dir.join("arome.nc");
        std::fs::copy(&a, &u).unwrap();
        let req = SliceRequest {
            var: "/wind_speed".into(),
            start: vec![0, 0, 0, 0],
            count: vec![2, 1, 4, 5],
            stride: vec![1, 1, 1, 1],
        };
        let want = NcFile::open(&a).unwrap().read_slice(&req).unwrap().data;
        let nc = NcFile::open(&u).unwrap();
        let got = nc.read_slice(&req).unwrap().data;
        assert_eq!(want.len(), got.len());
        assert!(want.iter().zip(&got).all(|(x, y)| x == y || (x.is_nan() && y.is_nan())));
        eprintln!(
            "non_ascii_path_reads: netCDF-C {} the real path (c_path = {})",
            if nc.c_path == u { "accepted" } else { "refused" },
            nc.c_path.display()
        );
        eprintln!("HDF5 on the real path: {}", crate::fast::h5::H5File::open(&u).is_ok());
        // The fast reader opens the same stored path.
        assert!(crate::fast::h5::H5File::open(&nc.c_path).is_ok());
    }
}
