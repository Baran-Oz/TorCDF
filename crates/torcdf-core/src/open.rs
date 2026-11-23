//! Metadata-only open of a netCDF/HDF5 file. No variable data is read.

use crate::model::*;
use netcdf::AttributeValue;
use netcdf::types::{FloatType, IntType, NcVariableType};
use std::ffi::CString;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};

/// An open file: its structure plus the netCDF handle used for later reads.
pub struct NcFile {
    pub meta: Dataset,
    /// The fast chunk reader's handles and caches; dropped (closed) before
    /// the netCDF handle.
    pub(crate) fast: crate::fast::FastState,
    pub(crate) file: netcdf::File,
    /// The exact path given to netCDF-C (the 8.3 form on Windows when
    /// needed), so a second HDF5 open names the same file.
    pub(crate) c_path: PathBuf,
    open_id: u64,
    /// Vector pairs: detected at open, then added or removed by hand. They
    /// last for the session only.
    pub(crate) pairs: std::sync::RwLock<Vec<std::sync::Arc<crate::vector::VectorPair>>>,
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

    /// Opens `path` and reads its structure (dimensions, variables,
    /// attributes, groups). Variable data is not touched.
    pub fn open(path: &Path) -> Result<NcFile, CoreError> {
        let open_err = |msg: String| CoreError::Open { path: path.to_path_buf(), msg };
        let cpath = c_path(path)?;
        let file = netcdf::open(&cpath).map_err(|e| open_err(e.to_string()))?;
        let raw = RawHandle::open(&cpath).map_err(open_err)?;
        let format = raw.format().map_err(open_err)?;
        let file_size = std::fs::metadata(path).map_err(|e| open_err(e.to_string()))?.len();

        // The netcdf crate panics on some metadata errors; a failed open must
        // not take the app down. `raw` is dropped (handle closed) on return.
        let mut root = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_root(&file)))
            .map_err(|_| open_err("file metadata could not be read (corrupt or unsupported)".into()))?;
        if matches!(format, FileFormat::Netcdf4 | FileFormat::Netcdf4Classic) {
            raw.fill_storage(&mut root);
        }
        let meta = Dataset { path: path.to_path_buf(), file_size, format, root };
        let open_id = NEXT_OPEN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Metadata only: no variable data is read.
        let pairs = crate::vector::detect_pairs(&meta).into_iter().map(std::sync::Arc::new).collect();
        Ok(NcFile {
            meta,
            fast: Default::default(),
            file,
            c_path: cpath,
            open_id,
            pairs: std::sync::RwLock::new(pairs),
        })
    }
}

/// The path to hand to netCDF-C. On Windows netCDF-C opens files through the
/// ANSI code page, so a path with non-ASCII characters (which would otherwise
/// not be found) is replaced by its ASCII-only short (8.3) form.
fn c_path(p: &Path) -> Result<PathBuf, CoreError> {
    let open_err = |msg: &str| CoreError::Open { path: p.to_path_buf(), msg: msg.into() };
    if p.to_str().is_some_and(str::is_ascii) {
        return Ok(p.to_path_buf());
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        unsafe extern "system" {
            fn GetShortPathNameW(long: *const u16, short: *mut u16, len: u32) -> u32;
        }
        let long: Vec<u16> = p.as_os_str().encode_wide().chain(Some(0)).collect();
        let needed = unsafe { GetShortPathNameW(long.as_ptr(), std::ptr::null_mut(), 0) };
        if needed == 0 {
            return Err(open_err("file not found or no short path available"));
        }
        let mut buf = vec![0u16; needed as usize];
        let n = unsafe { GetShortPathNameW(long.as_ptr(), buf.as_mut_ptr(), needed) } as usize;
        if n == 0 || n >= buf.len() {
            return Err(open_err("could not form a short path"));
        }
        let short = PathBuf::from(std::ffi::OsString::from_wide(&buf[..n]));
        if short.to_str().is_some_and(str::is_ascii) {
            return Ok(short);
        }
        Err(open_err("path has non-ASCII characters and no ASCII short form"))
    }
    #[cfg(not(windows))]
    {
        p.to_str().map(|_| p.to_path_buf()).ok_or_else(|| open_err("path is not valid UTF-8"))
    }
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
fn u64s_to_attr(v: Vec<u64>) -> AttrValue {
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
}
