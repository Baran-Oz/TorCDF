//! Structure of a plain HDF5 file, read through the HDF5 library. No
//! dataset values are read.
//!
//! Used when netCDF-C refuses a file or hides some of its datasets.
//! Dimensions come from attached dimension scales; an axis without one gets
//! a phony dimension `phony_dim_<n>`, shared within a group by length
//! (netCDF-C's convention). Integer and float datasets are numeric, strings
//! are `Str`, every other type is `DType::Other` (metadata only).
//! Attributes are kept when numeric or string, scalar or 1-D.

use crate::fast::h5::H5File;
use crate::model::*;
use hdf5_metno_sys::h5::{H5_INDEX_CRT_ORDER, H5_INDEX_NAME, H5_ITER_INC, herr_t, hsize_t};
use hdf5_metno_sys::h5a::{H5Aclose, H5Aget_name, H5Aget_space, H5Aget_type, H5Aopen_by_idx, H5Aread};
use hdf5_metno_sys::h5d::{H5D_layout_t, H5Dget_create_plist, H5Dget_space, H5Dget_type};
use hdf5_metno_sys::h5g::H5Gget_create_plist;
use hdf5_metno_sys::h5i::hid_t;
use hdf5_metno_sys::h5l::{H5L_info2_t, H5L_type_t, H5Literate2};
use hdf5_metno_sys::h5o::{
    H5O_INFO_BASIC, H5O_INFO_NUM_ATTRS, H5O_info2_t, H5O_token_t, H5O_type_t, H5Oclose, H5Oget_info3, H5Oopen,
};
use hdf5_metno_sys::h5p::{
    H5P_CRT_ORDER_TRACKED, H5P_DEFAULT, H5Pclose, H5Pget_chunk, H5Pget_layout, H5Pget_link_creation_order,
    H5Pget_nfilters,
};
use hdf5_metno_sys::h5s::{
    H5S_UNLIMITED, H5S_class_t, H5Sclose, H5Sget_simple_extent_dims, H5Sget_simple_extent_ndims,
    H5Sget_simple_extent_type,
};
use hdf5_metno_sys::h5t::{
    H5T_C_S1, H5T_NATIVE_DOUBLE, H5T_NATIVE_LLONG, H5T_NATIVE_ULLONG, H5T_SGN_NONE, H5T_STR_SPACEPAD,
    H5T_VARIABLE, H5T_class_t, H5Tclose, H5Tcopy, H5Tget_class, H5Tget_cset, H5Tget_sign, H5Tget_size,
    H5Tget_strpad, H5Tis_variable_str, H5Treclaim, H5Tset_cset, H5Tset_size,
};
use netcdf_sys::libnetcdf_lock;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_void};
use std::os::raw::{c_char, c_uint};
use std::path::Path;
use std::ptr::null_mut;

/// HDF5 high-level dimension-scale calls (the `hl` library is linked for
/// netCDF-C). Callers hold `netcdf_sys::libnetcdf_lock`.
pub(crate) mod ffi {
    use hdf5_metno_sys::h5::{herr_t, htri_t};
    use hdf5_metno_sys::h5i::hid_t;
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int, c_uint};

    pub(crate) type H5DSIterate = Option<unsafe extern "C" fn(hid_t, c_uint, hid_t, *mut c_void) -> herr_t>;

    unsafe extern "C" {
        pub(crate) fn H5DSis_scale(did: hid_t) -> htri_t;
        pub(crate) fn H5DSget_scale_name(did: hid_t, name: *mut c_char, size: usize) -> isize;
        pub(crate) fn H5DSget_num_scales(did: hid_t, dim: c_uint) -> c_int;
        pub(crate) fn H5DSiterate_scales(
            did: hid_t,
            dim: c_uint,
            idx: *mut c_int,
            visitor: H5DSIterate,
            visitor_data: *mut c_void,
        ) -> herr_t;
        #[cfg(any(test, feature = "fixtures"))]
        pub(crate) fn H5DSset_scale(dsid: hid_t, dimname: *const c_char) -> herr_t;
        #[cfg(any(test, feature = "fixtures"))]
        pub(crate) fn H5DSattach_scale(did: hid_t, dsid: hid_t, idx: c_uint) -> herr_t;
    }
}

/// netCDF-C names a dimension scale this way when it is a dimension without
/// a coordinate variable; netCDF-C does not list it as a variable.
const DIM_WITHOUT_VARIABLE: &str = "This is a netCDF dimension but not a netCDF variable.";

/// Bookkeeping attributes of dimension scales and netCDF-4, never shown.
const HIDDEN_ATTRS: [&str; 8] = [
    "DIMENSION_LIST",
    "REFERENCE_LIST",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_NCProperties",
    "_nc3_strict",
    "_IsNetcdf4",
    "_SuperblockVersion",
];

/// The root group of the HDF5 file `f`, opened from `path`.
pub(crate) fn read_h5_meta(f: &H5File, path: &Path) -> Result<Group, String> {
    let _g = libnetcdf_lock.lock();
    let mut w = Walker::default();
    let mut root = Held::new(unsafe { H5Oopen(f.id(), c"/".as_ptr(), H5P_DEFAULT) }, H5Oclose, "open the root group")
        .and_then(|g| w.group(g.0, "/", "/"))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    w.resolve_scales(&mut root);
    let mut added = HashSet::new();
    for (group_path, dim) in w.named {
        if !added.insert((group_path.clone(), dim.name.clone())) {
            continue;
        }
        let g = find_group(&mut root, &group_path);
        if !g.dims.iter().any(|d| d.name == dim.name) {
            g.dims.push(dim);
        }
    }
    Ok(root)
}

/// Variables of `g` and its subgroups that can be plotted (numeric).
pub(crate) fn plottable(g: &Group) -> usize {
    g.vars.iter().filter(|v| v.dtype.is_numeric()).count() + g.groups.iter().map(plottable).sum::<usize>()
}

/// Whether HDF5 finds more plottable datasets (`h5_count`, from
/// [`count_plottable`]) than netCDF-C shows numeric variables in `nc`:
/// netCDF-C hid some.
pub(crate) fn hides_datasets(nc: &Group, h5_count: usize) -> bool {
    h5_count > plottable(nc)
}

/// The number of datasets [`read_h5_meta`] would list as numeric, found
/// cheaply: links, object types, type classes and dimension-scale names
/// only (no attributes, no dimension names).
pub(crate) fn count_plottable(f: &H5File) -> Result<usize, String> {
    fn walk(gid: hid_t, seen: &mut HashSet<[u8; 16]>) -> Result<usize, String> {
        seen.insert(token_key(&obj_info(gid)?.token));
        let mut n = 0;
        for link in link_names(gid)? {
            let Ok(obj) = Held::new(unsafe { H5Oopen(gid, link.as_ptr(), H5P_DEFAULT) }, H5Oclose, "open") else {
                continue;
            };
            let Ok(oi) = obj_info(obj.0) else { continue };
            match oi.type_ {
                H5O_type_t::H5O_TYPE_GROUP if !seen.contains(&token_key(&oi.token)) => {
                    n += walk(obj.0, seen).unwrap_or(0);
                }
                H5O_type_t::H5O_TYPE_DATASET => n += usize::from(dataset_is_plottable(obj.0)),
                _ => {}
            }
        }
        Ok(n)
    }
    let _g = libnetcdf_lock.lock();
    let root = Held::new(unsafe { H5Oopen(f.id(), c"/".as_ptr(), H5P_DEFAULT) }, H5Oclose, "open the root group")?;
    walk(root.0, &mut HashSet::new())
}

/// Whether [`read_h5_meta`] lists dataset `did` as numeric.
fn dataset_is_plottable(did: hid_t) -> bool {
    let Ok(space) = Held::new(unsafe { H5Dget_space(did) }, H5Sclose, "read the dataset's shape") else {
        return false;
    };
    if !matches!(extent(space.0), Ok((_, _, false))) {
        return false;
    }
    let Ok(ty) = Held::new(unsafe { H5Dget_type(did) }, H5Tclose, "read the dataset's type") else {
        return false;
    };
    if !dtype_of(ty.0).is_numeric() {
        return false;
    }
    !(unsafe { ffi::H5DSis_scale(did) } > 0 && scale_name(did).is_some_and(|n| n.starts_with(DIM_WITHOUT_VARIABLE)))
}

/// An object token as a hashable key.
fn token_key(t: &H5O_token_t) -> [u8; 16] {
    // SAFETY: H5O_token_t is a `repr(C)` wrapper of 16 bytes.
    unsafe { std::mem::transmute::<H5O_token_t, [u8; 16]>(*t) }
}

/// The group at `path` below `root`; `root` itself if there is none.
fn find_group<'a>(root: &'a mut Group, path: &str) -> &'a mut Group {
    fn has(g: &Group, path: &str) -> bool {
        g.path == path || g.groups.iter().any(|s| has(s, path))
    }
    if root.path == path || !has(root, path) {
        return root;
    }
    let i = root.groups.iter().position(|s| has(s, path)).unwrap_or_default();
    find_group(&mut root.groups[i], path)
}

/// An HDF5 identifier closed when dropped. Unlike `fast::h5`'s, its drop
/// does not take the lock: it is only used while the lock is held.
pub(crate) struct Held(pub(crate) hid_t, unsafe extern "C" fn(hid_t) -> herr_t);

impl Held {
    pub(crate) fn new(id: hid_t, close: unsafe extern "C" fn(hid_t) -> herr_t, what: &str) -> Result<Held, String> {
        if id < 0 { Err(format!("HDF5 could not {what}")) } else { Ok(Held(id, close)) }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        unsafe { (self.1)(self.0) };
    }
}

#[derive(Default)]
struct Walker {
    /// Number of the next phony dimension, counted over the whole file.
    next_phony: usize,
    /// Dimensions named by dimension scales, with the path of the group
    /// that holds the scale; added to the groups after the walk.
    named: Vec<(String, Dim)>,
    /// Groups already walked: hard links can form cycles.
    seen: HashSet<[u8; 16]>,
    /// Every dataset walked, by object token: its group path and name.
    datasets: HashMap<[u8; 16], (String, String)>,
    /// Axes on a dimension scale, by variable path; named once the walk has
    /// seen every dataset (the scale may come later).
    scaled: HashMap<String, Vec<ScaledAxis>>,
}

/// An axis whose dimension is the attached scale `token`.
struct ScaledAxis {
    axis: usize,
    token: [u8; 16],
    unlimited: bool,
    /// The scale's `NAME`, used if the scale itself was not walked (e.g.
    /// reachable only through a soft link).
    fallback: String,
}

impl Walker {
    /// Group `gid` and everything below it. Members that cannot be opened or
    /// described are left out.
    fn group(&mut self, gid: hid_t, name: &str, path: &str) -> Result<Group, String> {
        let info = obj_info(gid)?;
        self.seen.insert(token_key(&info.token));
        let mut g = Group {
            name: name.into(),
            path: path.into(),
            dims: Vec::new(),
            vars: Vec::new(),
            attrs: read_attrs(gid, info.num_attrs, false),
            groups: Vec::new(),
        };
        for link in link_names(gid)? {
            let Ok(obj) = Held::new(unsafe { H5Oopen(gid, link.as_ptr(), H5P_DEFAULT) }, H5Oclose, "open") else {
                continue;
            };
            let Ok(oi) = obj_info(obj.0) else { continue };
            let link = link.to_string_lossy();
            let child = join(path, &link);
            match oi.type_ {
                H5O_type_t::H5O_TYPE_GROUP if !self.seen.contains(&token_key(&oi.token)) => {
                    if let Ok(sub) = self.group(obj.0, &link, &child) {
                        g.groups.push(sub);
                    }
                }
                H5O_type_t::H5O_TYPE_DATASET => {
                    let key = token_key(&oi.token);
                    self.datasets.entry(key).or_insert_with(|| (g.path.clone(), link.to_string()));
                    if let Ok(Some(v)) = self.dataset(obj.0, &link, &child, oi.num_attrs, &mut g) {
                        g.vars.push(v);
                    }
                }
                _ => {}
            }
        }
        Ok(g)
    }

    /// Dataset `did` of group `g`; None for a netCDF dimension without a
    /// variable. New phony dimensions are added to `g.dims`.
    fn dataset(
        &mut self,
        did: hid_t,
        name: &str,
        path: &str,
        nattrs: hsize_t,
        g: &mut Group,
    ) -> Result<Option<VarInfo>, String> {
        let space = Held::new(unsafe { H5Dget_space(did) }, H5Sclose, "read the dataset's shape")?;
        let (shape, unlimited, empty) = extent(space.0)?;
        let is_scale = unsafe { ffi::H5DSis_scale(did) } > 0;
        if is_scale && scale_name(did).is_some_and(|n| n.starts_with(DIM_WITHOUT_VARIABLE)) {
            if let (Some(&len), Some(&unl)) = (shape.first(), unlimited.first()) {
                self.named.push((g.path.clone(), Dim { name: name.into(), len, unlimited: unl }));
            }
            return Ok(None);
        }
        let ty = Held::new(unsafe { H5Dget_type(did) }, H5Tclose, "read the dataset's type")?;
        let dtype = if empty { DType::Other("empty".into()) } else { dtype_of(ty.0) };

        let mut dims: Vec<String> = Vec::with_capacity(shape.len());
        for (d, (&len, &unl)) in shape.iter().zip(&unlimited).enumerate() {
            let dim_name = if let Some((token, fallback)) = (!is_scale).then(|| attached_scale(did, d)).flatten() {
                // Named by `resolve_scales` after the walk.
                let axis = ScaledAxis { axis: d, token, unlimited: unl, fallback: fallback.clone() };
                self.scaled.entry(path.to_string()).or_default().push(axis);
                fallback
            } else if is_scale && shape.len() == 1 {
                self.named.push((g.path.clone(), Dim { name: name.into(), len, unlimited: unl }));
                name.to_string()
            } else {
                self.phony(&mut g.dims, &dims, len, unl)
            };
            dims.push(dim_name);
        }
        Ok(Some(VarInfo {
            name: name.into(),
            path: path.into(),
            dims,
            shape,
            dtype,
            attrs: read_attrs(did, nattrs, is_scale),
            storage: storage(did),
        }))
    }

    /// Names every axis on a dimension scale after the scale dataset, and
    /// records the dimension for the scale's group.
    fn resolve_scales(&mut self, g: &mut Group) {
        for v in &mut g.vars {
            for s in self.scaled.remove(&v.path).unwrap_or_default() {
                let (group_path, name) =
                    self.datasets.get(&s.token).cloned().unwrap_or_else(|| (g.path.clone(), s.fallback));
                if let (Some(slot), Some(&len)) = (v.dims.get_mut(s.axis), v.shape.get(s.axis)) {
                    slot.clone_from(&name);
                    self.named.push((group_path, Dim { name, len, unlimited: s.unlimited }));
                }
            }
        }
        for sub in &mut g.groups {
            self.resolve_scales(sub);
        }
    }

    /// A phony dimension of `len` in `group_dims` that `used` (the axes of
    /// the same dataset so far) does not already use; created if needed.
    fn phony(&mut self, group_dims: &mut Vec<Dim>, used: &[String], len: usize, unlimited: bool) -> String {
        let reuse = group_dims.iter().find(|d| d.len == len && d.unlimited == unlimited && !used.contains(&d.name));
        if let Some(d) = reuse {
            return d.name.clone();
        }
        let name = format!("phony_dim_{}", self.next_phony);
        self.next_phony += 1;
        group_dims.push(Dim { name: name.clone(), len, unlimited });
        name
    }
}

fn join(parent: &str, name: &str) -> String {
    if parent == "/" { format!("/{name}") } else { format!("{parent}/{name}") }
}

fn obj_info(id: hid_t) -> Result<H5O_info2_t, String> {
    let mut info: H5O_info2_t = unsafe { std::mem::zeroed() };
    if unsafe { H5Oget_info3(id, &mut info, H5O_INFO_BASIC | H5O_INFO_NUM_ATTRS) } < 0 {
        return Err("HDF5 could not describe an object".into());
    }
    Ok(info)
}

/// Names of the hard links in group `gid`, in creation order when the group
/// tracks it, otherwise by name. Soft and external links are skipped.
fn link_names(gid: hid_t) -> Result<Vec<CString>, String> {
    let gcpl = Held::new(unsafe { H5Gget_create_plist(gid) }, H5Pclose, "read group properties")?;
    let mut flags: c_uint = 0;
    let tracked = unsafe { H5Pget_link_creation_order(gcpl.0, &mut flags) } >= 0 && flags & H5P_CRT_ORDER_TRACKED != 0;
    let index = if tracked { H5_INDEX_CRT_ORDER } else { H5_INDEX_NAME };
    let mut names: Vec<CString> = Vec::new();
    let mut idx: hsize_t = 0;
    let op_data = (&raw mut names).cast::<c_void>();
    if unsafe { H5Literate2(gid, index, H5_ITER_INC, &mut idx, Some(collect_link), op_data) } < 0 {
        return Err("HDF5 could not list a group".into());
    }
    Ok(names)
}

/// `H5Literate2` callback: records hard link names in the `Vec<CString>`
/// behind `op_data`.
unsafe extern "C" fn collect_link(
    _group: hid_t,
    name: *const c_char,
    info: *const H5L_info2_t,
    op_data: *mut c_void,
) -> herr_t {
    let names = unsafe { &mut *op_data.cast::<Vec<CString>>() };
    if unsafe { (*info).type_ } == H5L_type_t::H5L_TYPE_HARD {
        names.push(unsafe { CStr::from_ptr(name) }.to_owned());
    }
    0 // H5_ITER_CONT
}

/// Shape, per-axis unlimited flags, and whether the dataspace is empty
/// (`H5S_NULL`: no values at all). Scalars have rank 0.
fn extent(space: hid_t) -> Result<(Vec<usize>, Vec<bool>, bool), String> {
    match unsafe { H5Sget_simple_extent_type(space) } {
        H5S_class_t::H5S_SCALAR => Ok((vec![], vec![], false)),
        H5S_class_t::H5S_NULL => Ok((vec![], vec![], true)),
        H5S_class_t::H5S_SIMPLE => {
            let rank = unsafe { H5Sget_simple_extent_ndims(space) };
            if rank < 0 {
                return Err("HDF5 could not read a shape".into());
            }
            let mut dims = vec![0 as hsize_t; rank as usize];
            let mut max = vec![0 as hsize_t; rank as usize];
            if unsafe { H5Sget_simple_extent_dims(space, dims.as_mut_ptr(), max.as_mut_ptr()) } != rank {
                return Err("HDF5 could not read a shape".into());
            }
            let shape = dims.iter().map(|&d| usize::try_from(d).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
            Ok((shape, max.iter().map(|&m| m == H5S_UNLIMITED).collect(), false))
        }
        _ => Err("HDF5 could not read a shape".into()),
    }
}

/// The `NAME` of dimension scale `did`, if it has one.
fn scale_name(did: hid_t) -> Option<String> {
    let n = unsafe { ffi::H5DSget_scale_name(did, null_mut(), 0) };
    if n <= 0 {
        return None;
    }
    let mut buf = vec![0u8; n as usize + 1];
    if unsafe { ffi::H5DSget_scale_name(did, buf.as_mut_ptr().cast(), buf.len()) } < 0 {
        return None;
    }
    CStr::from_bytes_until_nul(&buf).ok().map(|s| s.to_string_lossy().into_owned())
}

/// The first dimension scale attached to axis `d` of `did`: its object
/// token (cheap, unlike its path, which HDF5 finds by searching the file)
/// and its `NAME` (`"scale"` if none).
fn attached_scale(did: hid_t, d: usize) -> Option<([u8; 16], String)> {
    let d = c_uint::try_from(d).ok()?;
    if unsafe { ffi::H5DSget_num_scales(did, d) } <= 0 {
        return None;
    }
    let mut out: Option<([u8; 16], String)> = None;
    let data = (&raw mut out).cast::<c_void>();
    unsafe { ffi::H5DSiterate_scales(did, d, null_mut(), Some(first_scale), data) };
    out
}

/// `H5DSiterate_scales` visitor: stores the scale's token and `NAME` in
/// the `Option<([u8; 16], String)>` behind `data` and stops.
unsafe extern "C" fn first_scale(_did: hid_t, _dim: c_uint, dsid: hid_t, data: *mut c_void) -> herr_t {
    let out = unsafe { &mut *data.cast::<Option<([u8; 16], String)>>() };
    let Ok(info) = obj_info(dsid) else { return -1 };
    let name = scale_name(dsid).filter(|n| !n.is_empty()).unwrap_or_else(|| "scale".into());
    *out = Some((token_key(&info.token), name));
    1 // stop
}

fn class_name(c: H5T_class_t) -> &'static str {
    match c {
        H5T_class_t::H5T_TIME => "time",
        H5T_class_t::H5T_BITFIELD => "bitfield",
        H5T_class_t::H5T_OPAQUE => "opaque",
        H5T_class_t::H5T_COMPOUND => "compound",
        H5T_class_t::H5T_REFERENCE => "reference",
        H5T_class_t::H5T_ENUM => "enum",
        H5T_class_t::H5T_VLEN => "vlen",
        H5T_class_t::H5T_ARRAY => "array",
        // H5T_COMPLEX (HDF5 2.0), not named by every binding version.
        c if c as i32 == 11 => "complex",
        _ => "unknown",
    }
}

/// Integer and float classes are numeric: an unusual size maps to the
/// nearest type that holds every value (HDF5 converts on read).
fn dtype_of(t: hid_t) -> DType {
    let size = unsafe { H5Tget_size(t) };
    match unsafe { H5Tget_class(t) } {
        H5T_class_t::H5T_INTEGER => {
            let signed = unsafe { H5Tget_sign(t) } != H5T_SGN_NONE;
            match (size, signed) {
                (..=1, true) => DType::I8,
                (..=1, false) => DType::U8,
                (2, true) => DType::I16,
                (2, false) => DType::U16,
                (3..=4, true) => DType::I32,
                (3..=4, false) => DType::U32,
                (_, true) => DType::I64,
                (_, false) => DType::U64,
            }
        }
        H5T_class_t::H5T_FLOAT if size <= 4 => DType::F32,
        H5T_class_t::H5T_FLOAT => DType::F64,
        H5T_class_t::H5T_STRING => DType::Str,
        c => DType::Other(class_name(c).into()),
    }
}

/// Chunking and compression from the dataset's creation properties.
fn storage(did: hid_t) -> Storage {
    let Ok(dcpl) = Held::new(unsafe { H5Dget_create_plist(did) }, H5Pclose, "read dataset properties") else {
        return Storage::Unknown;
    };
    match unsafe { H5Pget_layout(dcpl.0) } {
        H5D_layout_t::H5D_CONTIGUOUS => Storage::Contiguous,
        H5D_layout_t::H5D_COMPACT => Storage::Compact,
        H5D_layout_t::H5D_CHUNKED => {
            let mut chunk = vec![0 as hsize_t; 32]; // H5S_MAX_RANK
            let rank = unsafe { H5Pget_chunk(dcpl.0, chunk.len() as i32, chunk.as_mut_ptr()) };
            if rank < 0 {
                return Storage::Unknown;
            }
            chunk.truncate(rank as usize);
            let compressed = unsafe { H5Pget_nfilters(dcpl.0) } > 0;
            Storage::Chunked { chunk: chunk.into_iter().map(|c| c as usize).collect(), compressed }
        }
        _ => Storage::Unknown,
    }
}

/// The `n` attributes of object `oid`, by name. Unreadable ones (other
/// types or shapes) and bookkeeping ones are left out; `scale`: also the
/// dimension-scale `CLASS` and `NAME`.
fn read_attrs(oid: hid_t, n: hsize_t, scale: bool) -> Vec<Attr> {
    (0..n)
        .filter_map(|i| {
            let a = unsafe { H5Aopen_by_idx(oid, c".".as_ptr(), H5_INDEX_NAME, H5_ITER_INC, i, H5P_DEFAULT, H5P_DEFAULT) };
            let a = Held::new(a, H5Aclose, "open an attribute").ok()?;
            let name = attr_name(a.0)?;
            let hidden = HIDDEN_ATTRS.contains(&name.as_str()) || (scale && (name == "CLASS" || name == "NAME"));
            if hidden {
                return None;
            }
            Some(Attr { value: attr_value(a.0)?, name })
        })
        .collect()
}

fn attr_name(a: hid_t) -> Option<String> {
    let n = unsafe { H5Aget_name(a, 0, null_mut()) };
    if n < 0 {
        return None;
    }
    let mut buf = vec![0u8; n as usize + 1];
    if unsafe { H5Aget_name(a, buf.len(), buf.as_mut_ptr().cast()) } < 0 {
        return None;
    }
    CStr::from_bytes_until_nul(&buf).ok().map(|s| s.to_string_lossy().into_owned())
}

/// A numeric or string attribute that is a scalar or 1-D.
fn attr_value(a: hid_t) -> Option<AttrValue> {
    let space = Held::new(unsafe { H5Aget_space(a) }, H5Sclose, "read an attribute's shape").ok()?;
    let (shape, _, empty) = extent(space.0).ok()?;
    if empty || shape.len() > 1 {
        return None;
    }
    let n = shape.first().copied().unwrap_or(1);
    let ty = Held::new(unsafe { H5Aget_type(a) }, H5Tclose, "read an attribute's type").ok()?;
    match unsafe { H5Tget_class(ty.0) } {
        H5T_class_t::H5T_INTEGER if unsafe { H5Tget_sign(ty.0) } == H5T_SGN_NONE => {
            let mut v = vec![0u64; n];
            (unsafe { H5Aread(a, *H5T_NATIVE_ULLONG, v.as_mut_ptr().cast()) } >= 0)
                .then(|| crate::open::u64s_to_attr(v))
        }
        H5T_class_t::H5T_INTEGER => {
            let mut v = vec![0i64; n];
            (unsafe { H5Aread(a, *H5T_NATIVE_LLONG, v.as_mut_ptr().cast()) } >= 0).then_some(AttrValue::Ints(v))
        }
        H5T_class_t::H5T_FLOAT => {
            let mut v = vec![0f64; n];
            (unsafe { H5Aread(a, *H5T_NATIVE_DOUBLE, v.as_mut_ptr().cast()) } >= 0).then_some(AttrValue::Floats(v))
        }
        H5T_class_t::H5T_STRING => {
            let mut v = read_strings(a, ty.0, space.0, n)?;
            if shape.is_empty() || v.len() == 1 { v.pop().map(AttrValue::Text) } else { Some(AttrValue::Texts(v)) }
        }
        _ => None,
    }
}

/// The `n` strings of string attribute `a` (type `ty`, dataspace `space`),
/// fixed or variable length.
fn read_strings(a: hid_t, ty: hid_t, space: hid_t, n: usize) -> Option<Vec<String>> {
    if unsafe { H5Tis_variable_str(ty) } > 0 {
        let mem = Held::new(unsafe { H5Tcopy(*H5T_C_S1) }, H5Tclose, "make a string type").ok()?;
        let set = unsafe { H5Tset_size(mem.0, H5T_VARIABLE) >= 0 && H5Tset_cset(mem.0, H5Tget_cset(ty)) >= 0 };
        if !set {
            return None;
        }
        let mut ptrs: Vec<*mut c_char> = vec![null_mut(); n];
        if unsafe { H5Aread(a, mem.0, ptrs.as_mut_ptr().cast()) } < 0 {
            return None;
        }
        let out = ptrs
            .iter()
            .map(|&p| if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() })
            .collect();
        unsafe { H5Treclaim(mem.0, space, H5P_DEFAULT, ptrs.as_mut_ptr().cast()) };
        return Some(out);
    }
    let size = unsafe { H5Tget_size(ty) };
    let mut buf = vec![0u8; size.checked_mul(n)?];
    if size == 0 || unsafe { H5Aread(a, ty, buf.as_mut_ptr().cast()) } < 0 {
        return None;
    }
    let spaces = unsafe { H5Tget_strpad(ty) } == H5T_STR_SPACEPAD;
    let out = buf
        .chunks(size)
        .map(|s| {
            let end = s.iter().position(|&b| b == 0).unwrap_or(s.len());
            let t = String::from_utf8_lossy(&s[..end]);
            if spaces { t.trim_end_matches(' ').to_string() } else { t.into_owned() }
        })
        .collect();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{self, hdf5 as fx, vectors};
    use crate::{NcFile, SliceRequest};

    fn attr<'a>(attrs: &'a [Attr], name: &str) -> &'a AttrValue {
        &attrs.iter().find(|a| a.name == name).unwrap_or_else(|| panic!("no attribute {name}")).value
    }

    fn dim(name: &str, len: usize) -> Dim {
        Dim { name: name.into(), len, unlimited: false }
    }

    fn names(v: &VarInfo) -> Vec<&str> {
        v.dims.iter().map(String::as_str).collect()
    }

    #[test]
    fn plain_h5_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let p = fx::write_plain_h5(dir.path());
        // D7: netCDF-C opens this file and hides none of its datasets, so
        // `open` keeps netCDF-C.
        let nc = NcFile::open(&p).unwrap();
        assert!(!nc.backend_is_hdf5());
        assert_eq!(nc.meta.format, FileFormat::Netcdf4);

        let h = NcFile::open_hdf5(&p).unwrap();
        assert!(h.backend_is_hdf5());
        let m = &h.meta;
        assert_eq!(m.format, FileFormat::Hdf5);
        assert_eq!(attr(&m.root.attrs, "title"), &AttrValue::Text("plain".into()));

        let temp = m.var("/obs/temp").unwrap();
        assert_eq!(names(temp), ["phony_dim_0", "phony_dim_1"]);
        assert_eq!(temp.shape, [3, 4]);
        assert_eq!(temp.dtype, DType::F32);
        assert_eq!(temp.storage, Storage::Contiguous);
        assert_eq!(attr(&temp.attrs, "units"), &AttrValue::Text("degC".into()));
        let count = m.var("/obs/count").unwrap();
        assert_eq!(names(count), ["phony_dim_1"]);
        assert_eq!(count.shape, [4]);
        assert_eq!(count.dtype, DType::I32);
        assert_eq!(m.group_of("/obs/temp").dims, [dim("phony_dim_0", 3), dim("phony_dim_1", 4)]);

        let grid = m.var("/grid").unwrap();
        assert_eq!(grid.dtype, DType::F64);
        assert_eq!(grid.shape, [2, 3, 5]);
        assert_eq!(names(grid), ["phony_dim_2", "phony_dim_3", "phony_dim_4"]);
        assert_eq!(grid.storage, Storage::Chunked { chunk: vec![1, 3, 5], compressed: true });
        let table = m.var("/table").unwrap();
        assert_eq!(table.dtype, DType::Other("compound".into()));
        assert_eq!(names(table), ["phony_dim_2"]);
        let paths: Vec<&str> = m.all_vars().iter().map(|v| v.path.as_str()).collect();
        assert_eq!(paths, ["/grid", "/table", "/obs/temp", "/obs/count"]);
        assert_eq!(m.root.groups[0].path, "/obs");
        assert_eq!(m.root.groups[0].name, "obs");
    }

    #[test]
    fn hidden_dataset_uses_hdf5_backend() {
        let dir = tempfile::tempdir().unwrap();
        let p = fx::write_hidden_h5(dir.path());
        assert!(netcdf::open(&p).is_err(), "netCDF-C refuses hidden.h5");
        let nc = NcFile::open(&p).unwrap();
        assert!(nc.backend_is_hdf5());
        let m = &nc.meta;
        assert_eq!(m.format, FileFormat::Hdf5);
        let paths: Vec<&str> = m.all_vars().iter().map(|v| v.path.as_str()).collect();
        assert_eq!(paths, ["/a", "/x", "/sub/b"]);
        let a = m.var("/a").unwrap();
        assert_eq!(names(a), ["phony_dim_0", "x"]);
        assert_eq!(a.shape, [2, 3]);
        assert!(a.attrs.is_empty(), "dimension scale bookkeeping is hidden: {:?}", a.attrs);
        let x = m.var("/x").unwrap();
        assert_eq!(names(x), ["x"]);
        assert_eq!(x.dtype, DType::F64);
        assert!(x.attrs.is_empty(), "dimension scale bookkeeping is hidden: {:?}", x.attrs);
        assert_eq!(m.root.dims, [dim("phony_dim_0", 2), dim("x", 3)]);
        let b = m.var("/sub/b").unwrap();
        assert_eq!(names(b), ["phony_dim_1"]);
        assert_eq!(b.dtype, DType::I32);
        assert_eq!(m.group_of("/sub/b").dims, [dim("phony_dim_1", 3)]);

        // Read through the HDF5 backend.
        let req = SliceRequest { var: "/a".into(), start: vec![0, 0], count: vec![2, 3], stride: vec![1, 1] };
        assert_eq!(nc.read_slice(&req).unwrap().data, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        // Neither the refused netCDF-C open nor the HDF5 one leaves the file open.
        drop(nc);
        std::fs::remove_file(&p).expect("hidden.h5 is closed");
    }

    #[test]
    fn netcdf_files_still_use_netcdf() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        for p in [vectors::write_nemo_like(d), vectors::write_arome_like(d), fixtures::write_nc4(d)] {
            let nc = NcFile::open(&p).unwrap();
            assert_eq!(nc.meta.format, FileFormat::Netcdf4, "{}", p.display());
            assert!(!nc.backend_is_hdf5(), "{}", p.display());
            // The HDF5 walk sees the same plottable variables (dimensions
            // without coordinate variables are not datasets to it either).
            let h5 = H5File::open(&nc.c_path).unwrap();
            let root = read_h5_meta(&h5, &p).unwrap();
            assert_eq!(plottable(&root), plottable(&nc.meta.root), "{}", p.display());
            assert_eq!(count_plottable(&h5).unwrap(), plottable(&root), "{}", p.display());
            let h5_meta = Dataset { path: p.clone(), file_size: 0, format: FileFormat::Hdf5, root };
            for v in nc.meta.all_vars() {
                let h = h5_meta.var(&v.path).unwrap_or_else(|| panic!("{} missing", v.path));
                assert_eq!(h.dims, v.dims, "{} dims", v.path);
            }
        }
        let classic = NcFile::open(&fixtures::write_classic(d)).unwrap();
        assert_eq!(classic.meta.format, FileFormat::Classic);
        assert!(!classic.backend_is_hdf5());
    }

    #[test]
    fn more_plottable_datasets_means_hidden() {
        let var = |name: &str, dtype: DType| VarInfo {
            name: name.into(),
            path: format!("/{name}"),
            dims: vec![],
            shape: vec![],
            dtype,
            attrs: vec![],
            storage: Storage::Unknown,
        };
        let group = |vars: Vec<VarInfo>| Group {
            name: "/".into(),
            path: "/".into(),
            dims: vec![],
            vars,
            attrs: vec![],
            groups: vec![],
        };
        let nc = group(vec![var("a", DType::F32), var("s", DType::Str)]);
        let same = group(vec![var("a", DType::F32), var("t", DType::Other("compound".into()))]);
        let more = group(vec![var("a", DType::F32), var("h", DType::I16)]);
        assert!(!hides_datasets(&nc, plottable(&same)));
        assert!(hides_datasets(&nc, plottable(&more)));
    }

    /// A netCDF-4 file with many variables: every axis has a dimension
    /// scale, so the hidden-dataset check must not resolve scale paths.
    #[test]
    fn many_variables_open_quickly() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("many.nc");
        {
            let mut f = netcdf::create(&p).unwrap();
            f.add_dimension("time", 2).unwrap();
            f.add_dimension("y", 3).unwrap();
            f.add_dimension("x", 4).unwrap();
            for i in 0..2000 {
                f.add_variable::<f32>(&format!("v{i}"), &["time", "y", "x"]).unwrap();
            }
        }
        let secs = std::time::Duration::from_secs;
        let t = std::time::Instant::now();
        let nc = NcFile::open(&p).unwrap();
        let took = t.elapsed();
        assert!(!nc.backend_is_hdf5());
        assert_eq!(plottable(&nc.meta.root), 2000);
        // With a full walk resolving scale paths, the open took about 85 s
        // (debug build); now about 5 s, nearly all of it outside the check.
        assert!(took < secs(30), "open took {took:?}");

        let h5 = H5File::open(&nc.c_path).unwrap();
        let t = std::time::Instant::now();
        assert_eq!(count_plottable(&h5).unwrap(), 2000, "the hidden check counts what netCDF-C shows");
        let count = t.elapsed();
        assert!(count < secs(2), "count took {count:?}");

        // The full walk (used when the backend switches) names dimensions by
        // scale token, not by searching the file for each scale's path.
        let t = std::time::Instant::now();
        let root = read_h5_meta(&h5, &p).unwrap();
        let full = t.elapsed();
        assert_eq!(root.dims.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), ["time", "y", "x"]);
        assert!(root.vars.iter().all(|v| v.dims == ["time", "y", "x"]));
        assert!(full < secs(10), "full walk took {full:?}");
        eprintln!("many_variables_open_quickly: open {took:?}, count {count:?}, full walk {full:?}");
    }
}
