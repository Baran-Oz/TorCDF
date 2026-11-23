//! Plain HDF5 fixtures (no netCDF conventions), written through the HDF5
//! library under the netCDF lock.

use crate::h5meta::ffi::{H5DSattach_scale, H5DSset_scale};
use hdf5_metno_sys::h5::{H5open, herr_t, hsize_t};
use hdf5_metno_sys::h5a::{H5Aclose, H5Acreate2, H5Awrite};
use hdf5_metno_sys::h5d::{H5Dclose, H5Dcreate2, H5Dwrite};
use hdf5_metno_sys::h5f::{H5F_ACC_TRUNC, H5Fclose, H5Fcreate};
use hdf5_metno_sys::h5g::{H5Gclose, H5Gcreate2};
use hdf5_metno_sys::h5i::hid_t;
use hdf5_metno_sys::h5l::H5Lcreate_soft;
use hdf5_metno_sys::h5p::{
    H5P_CLS_DATASET_CREATE, H5P_CLS_FILE_CREATE, H5P_CLS_GROUP_CREATE, H5P_CRT_ORDER_INDEXED,
    H5P_CRT_ORDER_TRACKED, H5P_DEFAULT, H5Pclose, H5Pcreate, H5Pset_chunk, H5Pset_deflate,
    H5Pset_link_creation_order,
};
use hdf5_metno_sys::h5s::{H5S_ALL, H5S_class_t, H5Sclose, H5Screate, H5Screate_simple};
use hdf5_metno_sys::h5t::{
    H5T_C_S1, H5T_IEEE_F32LE, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, H5T_NATIVE_FLOAT, H5T_NATIVE_INT,
    H5T_STD_I32LE, H5T_VARIABLE, H5T_class_t, H5T_cset_t, H5T_str_t, H5Tclose, H5Tcopy, H5Tcreate,
    H5Tinsert, H5Tset_cset, H5Tset_size, H5Tset_strpad,
};
use netcdf_sys::libnetcdf_lock;
use std::ffi::{CString, c_void};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr::null;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Asserts an HDF5 identifier or status is not an error.
fn ok<T: PartialOrd + Default + Copy>(v: T, what: &str) -> T {
    assert!(v >= T::default(), "HDF5 could not {what}");
    v
}

/// A property list of class `cls` that tracks and indexes link creation
/// order, so its group lists members in the order they were made.
unsafe fn ordered(cls: hid_t) -> hid_t {
    unsafe {
        let p = ok(H5Pcreate(cls), "create a property list");
        let flags = H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED;
        ok(H5Pset_link_creation_order(p, flags), "track creation order");
        p
    }
}

/// A new file at `path` (truncated). `in_order`: the root group lists its
/// members in creation order. Caller holds the lock.
unsafe fn create(path: &Path, in_order: bool) -> hid_t {
    let p = c(path.to_str().unwrap());
    unsafe {
        ok(H5open(), "initialise");
        let fcpl = if in_order { ordered(*H5P_CLS_FILE_CREATE) } else { H5P_DEFAULT };
        let f = ok(H5Fcreate(p.as_ptr(), H5F_ACC_TRUNC, fcpl, H5P_DEFAULT), "create the file");
        if in_order {
            H5Pclose(fcpl);
        }
        f
    }
}

/// A group that lists its members in creation order.
unsafe fn group(loc: hid_t, name: &str) -> hid_t {
    let n = c(name);
    unsafe {
        let gcpl = ordered(*H5P_CLS_GROUP_CREATE);
        let g = ok(H5Gcreate2(loc, n.as_ptr(), H5P_DEFAULT, gcpl, H5P_DEFAULT), "create a group");
        H5Pclose(gcpl);
        g
    }
}

/// Creates dataset `name` of file type `ftype` and shape `dims`, writes
/// `data` (memory type `mtype`) and returns it open. `chunk`: chunked +
/// deflate level 4.
unsafe fn dataset(
    loc: hid_t,
    name: &str,
    (ftype, dims): (hid_t, &[hsize_t]),
    chunk: Option<&[hsize_t]>,
    (mtype, data): (hid_t, *const c_void),
) -> hid_t {
    let n = c(name);
    unsafe {
        let space = ok(H5Screate_simple(dims.len() as i32, dims.as_ptr(), null()), "create a dataspace");
        let dcpl = match chunk {
            Some(ch) => {
                let p = ok(H5Pcreate(*H5P_CLS_DATASET_CREATE), "create a property list");
                ok(H5Pset_chunk(p, ch.len() as i32, ch.as_ptr()), "set chunking");
                ok(H5Pset_deflate(p, 4), "set deflate");
                p
            }
            None => H5P_DEFAULT,
        };
        let d = ok(H5Dcreate2(loc, n.as_ptr(), ftype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT), "create a dataset");
        ok(H5Dwrite(d, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "write a dataset");
        if chunk.is_some() {
            H5Pclose(dcpl);
        }
        H5Sclose(space);
        d
    }
}

/// A scalar string attribute: variable-length when `vlen`, else fixed-length.
unsafe fn str_attr(loc: hid_t, name: &str, value: &str, vlen: bool) {
    let n = c(name);
    let v = c(value);
    unsafe {
        let t = ok(H5Tcopy(*H5T_C_S1), "copy the string type");
        ok(H5Tset_cset(t, H5T_cset_t::H5T_CSET_UTF8), "set the character set");
        let space = ok(H5Screate(H5S_class_t::H5S_SCALAR), "create a dataspace");
        if vlen {
            ok(H5Tset_size(t, H5T_VARIABLE), "set the string size");
        } else {
            ok(H5Tset_size(t, value.len()), "set the string size");
            ok(H5Tset_strpad(t, H5T_str_t::H5T_STR_NULLPAD), "set string padding");
        }
        let a = ok(H5Acreate2(loc, n.as_ptr(), t, space, H5P_DEFAULT, H5P_DEFAULT), "create an attribute");
        let ptr: *const c_char = v.as_ptr();
        let rc: herr_t = if vlen { H5Awrite(a, t, (&raw const ptr).cast()) } else { H5Awrite(a, t, ptr.cast()) };
        ok(rc, "write an attribute");
        H5Aclose(a);
        H5Sclose(space);
        H5Tclose(t);
    }
}

/// Plain HDF5 `plain.h5`, no dimension scales, members in creation order:
/// - group `/obs`: `temp` f32 `[3,4]` = 0..12 with `units="degC"`
///   (variable-length string), then `count` i32 `[4]` = 10, 20, 30, 40;
/// - `/grid` f64 `[2,3,5]` = 0..30, chunked `[1,3,5]` + deflate;
/// - `/table`: two compound records `{a: i32, b: f64}` = (1, 1.5), (2, 2.5);
/// - root attribute `title="plain"` (fixed-length string).
pub fn write_plain_h5(dir: &Path) -> PathBuf {
    let path = dir.join("plain.h5");
    let _g = libnetcdf_lock.lock();
    unsafe {
        let f = create(&path, true);
        str_attr(f, "title", "plain", false);

        let obs = group(f, "obs");
        let temp: Vec<f32> = (0..12).map(|k| k as f32).collect();
        let d = dataset(obs, "temp", (*H5T_IEEE_F32LE, &[3, 4]), None, (*H5T_NATIVE_FLOAT, temp.as_ptr().cast()));
        str_attr(d, "units", "degC", true);
        H5Dclose(d);
        let count: [i32; 4] = [10, 20, 30, 40];
        let d = dataset(obs, "count", (*H5T_STD_I32LE, &[4]), None, (*H5T_NATIVE_INT, count.as_ptr().cast()));
        H5Dclose(d);
        H5Gclose(obs);

        let grid: Vec<f64> = (0..30).map(f64::from).collect();
        let chunk: &[hsize_t] = &[1, 3, 5];
        let d = dataset(f, "grid", (*H5T_IEEE_F64LE, &[2, 3, 5]), Some(chunk), (*H5T_NATIVE_DOUBLE, grid.as_ptr().cast()));
        H5Dclose(d);

        #[repr(C)]
        struct Rec {
            a: i32,
            b: f64,
        }
        let recs = [Rec { a: 1, b: 1.5 }, Rec { a: 2, b: 2.5 }];
        let t = ok(H5Tcreate(H5T_class_t::H5T_COMPOUND, size_of::<Rec>()), "create a compound type");
        ok(H5Tinsert(t, c("a").as_ptr(), std::mem::offset_of!(Rec, a), *H5T_NATIVE_INT), "insert a member");
        ok(H5Tinsert(t, c("b").as_ptr(), std::mem::offset_of!(Rec, b), *H5T_NATIVE_DOUBLE), "insert a member");
        H5Dclose(dataset(f, "table", (t, &[2]), None, (t, recs.as_ptr().cast())));
        H5Tclose(t);

        H5Fclose(f);
    }
    path
}

/// Plain HDF5 `hidden.h5` (members listed by name, the HDF5 default):
/// - `/a` f32 `[2,3]` = 0, 0.5, .. 2.5, its second axis on dimension scale `/x`;
/// - `/x` f64 `[3]` = 10, 20, 30, a dimension scale;
/// - `/sub/b` i32 `[3]` = 7, 8, 9;
/// - `/dangling`: a soft link to a missing object.
///
/// netCDF-C 4.9 refuses the file: it opens every link a group lists and
/// fails on the dangling one. (A file it opens but hides a plottable dataset
/// in cannot be made: it gives every integer and float dataset a netCDF type
/// and skips only other classes, which are not plottable.)
pub fn write_hidden_h5(dir: &Path) -> PathBuf {
    let path = dir.join("hidden.h5");
    let _g = libnetcdf_lock.lock();
    unsafe {
        let f = create(&path, false);
        let a: Vec<f32> = (0..6).map(|k| k as f32 * 0.5).collect();
        let da = dataset(f, "a", (*H5T_IEEE_F32LE, &[2, 3]), None, (*H5T_NATIVE_FLOAT, a.as_ptr().cast()));
        let x = [10.0f64, 20.0, 30.0];
        let dx = dataset(f, "x", (*H5T_IEEE_F64LE, &[3]), None, (*H5T_NATIVE_DOUBLE, x.as_ptr().cast()));
        ok(H5DSset_scale(dx, c("x").as_ptr()), "make a dimension scale");
        ok(H5DSattach_scale(da, dx, 1), "attach a dimension scale");
        H5Dclose(dx);
        H5Dclose(da);
        let sub = group(f, "sub");
        let b: [i32; 3] = [7, 8, 9];
        H5Dclose(dataset(sub, "b", (*H5T_STD_I32LE, &[3]), None, (*H5T_NATIVE_INT, b.as_ptr().cast())));
        H5Gclose(sub);
        let (target, link) = (c("/nowhere"), c("dangling"));
        ok(H5Lcreate_soft(target.as_ptr(), f, link.as_ptr(), H5P_DEFAULT, H5P_DEFAULT), "create a soft link");
        H5Fclose(f);
    }
    path
}
