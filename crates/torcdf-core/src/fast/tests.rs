//! The fast reader against the netCDF-C library path, bit for bit.

use crate::fixtures;
use crate::{NcFile, Slice, SliceRequest};

/// Every eligible variable of `fixtures::write_fast_matrix`.
const ELIGIBLE: [&str; 14] = [
    "sd_f32",
    "d_f64",
    "s_i32",
    "fl_f32",
    "p_i16",
    "mv_f32",
    "be_f32",
    "t_i8",
    "t_u8",
    "t_u16",
    "t_u32",
    "t_i64",
    "t_u64",
    "sparse_f32",
];

/// (start, count, stride) on `(time, y, x)` = `[7, 13, 17]`.
const REQUESTS: [([usize; 3], [usize; 3], [usize; 3]); 4] = [
    ([0, 0, 0], [7, 13, 17], [1, 1, 1]),
    ([1, 2, 3], [5, 9, 11], [2, 3, 4]),
    ([0, 6, 8], [7, 1, 1], [1, 1, 1]),
    ([6, 10, 12], [1, 3, 5], [1, 1, 1]),
];

fn req(var: &str, (start, count, stride): ([usize; 3], [usize; 3], [usize; 3])) -> SliceRequest {
    SliceRequest {
        var: format!("/{var}"),
        start: start.to_vec(),
        count: count.to_vec(),
        stride: stride.to_vec(),
    }
}

fn assert_same(a: &Slice, b: &Slice, what: &str) {
    assert_eq!(a.shape, b.shape, "{what}: shape");
    assert_eq!(a.valid_count, b.valid_count, "{what}: valid count");
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        assert!(
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            "{what}: element {i}: {x} != {y}"
        );
    }
}

/// Every eligible variable and request: `read_slice` equals the library.
fn assert_matrix_matches_library(nc: &NcFile) {
    for var in ELIGIBLE {
        for r in REQUESTS {
            let r = req(var, r);
            let fast = nc.read_slice(&r).unwrap();
            let lib = nc.read_slice_via_library(&r).unwrap();
            assert_same(&fast, &lib, &format!("{r:?}"));
            if var != "sparse_f32" {
                assert!(fast.valid_count > 0, "{r:?} has no data");
            }
        }
    }
}

#[test]
fn fast_equals_library_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let nc = NcFile::open(&fixtures::write_fast_matrix(dir.path())).unwrap();
    assert_matrix_matches_library(&nc);
}

/// The same matrix with chunks looked up one at a time, so pool threads
/// make HDF5 calls (under the lock) while they decode.
#[test]
fn fast_equals_library_with_lookup_index() {
    let dir = tempfile::tempdir().unwrap();
    let nc = NcFile::open(&fixtures::write_fast_matrix(dir.path())).unwrap();
    super::h5::TEST_ITER_LIMIT.set(Some(0));
    for var in ELIGIBLE {
        let v = nc.fast_var(&format!("/{var}")).expect("eligible");
        assert!(v.h5.uses_lookup_index(), "{var}");
    }
    super::h5::TEST_ITER_LIMIT.set(None);
    let before = nc.fast_reads();
    assert_matrix_matches_library(&nc);
    assert_eq!(nc.fast_reads(), before + 4 * ELIGIBLE.len());
}

#[test]
fn fast_path_is_used() {
    let dir = tempfile::tempdir().unwrap();
    let nc = NcFile::open(&fixtures::write_fast_matrix(dir.path())).unwrap();
    for var in ELIGIBLE {
        assert!(nc.fast_eligible(&format!("/{var}")), "{var} is eligible");
        let before = nc.fast_reads();
        for r in REQUESTS {
            nc.read_slice(&req(var, r)).unwrap();
        }
        assert_eq!(nc.fast_reads(), before + 4, "{var}");
    }

    assert!(!nc.fast_eligible("/contig_f32"));
    assert!(!nc.fast_eligible("/no_such_var"));
    let before = nc.fast_reads();
    let r = SliceRequest {
        var: "/contig_f32".into(),
        start: vec![0, 0],
        count: vec![13, 17],
        stride: vec![1, 1],
    };
    let s = nc.read_slice(&r).unwrap();
    assert!(s.valid_count > 0);
    assert_eq!(nc.fast_reads(), before);
}

#[test]
fn never_written_chunks_read_as_fill() {
    let dir = tempfile::tempdir().unwrap();
    let nc = NcFile::open(&fixtures::write_fast_matrix(dir.path())).unwrap();
    let r = req("sparse_f32", REQUESTS[0]);
    let before = nc.fast_reads();
    let fast = nc.read_slice(&r).unwrap();
    assert_eq!(nc.fast_reads(), before + 1);
    assert_same(&fast, &nc.read_slice_via_library(&r).unwrap(), "sparse_f32");
    let mut written = 0;
    for t in 0..7 {
        for y in 0..13 {
            for x in 0..17 {
                let v = fast.data[(t * 13 + y) * 17 + x];
                if t < 2 && y < 5 && x < 6 {
                    written += usize::from(!v.is_nan());
                } else {
                    assert!(v.is_nan(), "[{t}, {y}, {x}] = {v}");
                }
            }
        }
    }
    assert!(written > 50, "{written} written values");
}

/// Other files still read: netCDF-3 never takes the fast path.
#[test]
fn classic_file_is_not_eligible() {
    let dir = tempfile::tempdir().unwrap();
    let nc = NcFile::open(&fixtures::write_classic(dir.path())).unwrap();
    assert!(!nc.fast_eligible("/t"));
}

/// A plain HDF5 file (as h5py would write it, no netCDF attributes): `/v`,
/// f32 `[4, 6]` in `[2, 3]` chunks with HDF5 fill value 42 and no
/// `_FillValue`. Only the first chunk is written.
fn write_hdf5_user_fill(dir: &std::path::Path) -> std::path::PathBuf {
    use hdf5_metno_sys::h5::{H5open, hsize_t};
    use hdf5_metno_sys::h5d::{H5Dclose, H5Dcreate2, H5Dwrite};
    use hdf5_metno_sys::h5f::{H5F_ACC_TRUNC, H5Fclose, H5Fcreate};
    use hdf5_metno_sys::h5p::{
        H5P_CLS_DATASET_CREATE, H5P_DEFAULT, H5Pclose, H5Pcreate, H5Pset_chunk, H5Pset_fill_value,
    };
    use hdf5_metno_sys::h5s::{H5S_SELECT_SET, H5Sclose, H5Screate_simple, H5Sselect_hyperslab};
    use hdf5_metno_sys::h5t::{H5T_IEEE_F32LE, H5T_NATIVE_FLOAT};
    use std::ffi::CString;
    use std::ptr::null;

    let path = dir.join("h5fill.h5");
    let c = CString::new(path.to_str().unwrap()).unwrap();
    let name = CString::new("v").unwrap();
    let _g = netcdf_sys::libnetcdf_lock.lock();
    unsafe {
        assert!(H5open() >= 0);
        let f = H5Fcreate(c.as_ptr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        assert!(f >= 0);
        let dims: [hsize_t; 2] = [4, 6];
        let space = H5Screate_simple(2, dims.as_ptr(), null());
        let dcpl = H5Pcreate(*H5P_CLS_DATASET_CREATE);
        let chunk: [hsize_t; 2] = [2, 3];
        assert!(H5Pset_chunk(dcpl, 2, chunk.as_ptr()) >= 0);
        let fill = 42.0f32;
        assert!(H5Pset_fill_value(dcpl, *H5T_NATIVE_FLOAT, (&raw const fill).cast()) >= 0);
        let d = H5Dcreate2(
            f,
            name.as_ptr(),
            *H5T_IEEE_F32LE,
            space,
            H5P_DEFAULT,
            dcpl,
            H5P_DEFAULT,
        );
        assert!(d >= 0);
        let mem = H5Screate_simple(2, chunk.as_ptr(), null());
        let origin: [hsize_t; 2] = [0, 0];
        let sel = H5Sselect_hyperslab(
            space,
            H5S_SELECT_SET,
            origin.as_ptr(),
            null(),
            chunk.as_ptr(),
            null(),
        );
        assert!(sel >= 0);
        let vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(
            H5Dwrite(
                d,
                *H5T_NATIVE_FLOAT,
                mem,
                space,
                H5P_DEFAULT,
                vals.as_ptr().cast()
            ) >= 0
        );
        H5Sclose(mem);
        H5Dclose(d);
        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(f);
    }
    path
}

/// Never-written chunks read as HDF5's fill value. A file from another tool
/// may set one that is not `_FillValue` (or the netCDF default); the fast
/// path cannot match the library there, so the variable is not eligible.
#[test]
fn hdf5_fill_other_than_fillvalue_is_not_eligible() {
    let dir = tempfile::tempdir().unwrap();
    let nc = NcFile::open(&write_hdf5_user_fill(dir.path())).unwrap();
    assert!(nc.meta.var("/v").unwrap().attrs.is_empty());
    assert!(!nc.fast_eligible("/v"));
    let r = SliceRequest {
        var: "/v".into(),
        start: vec![0, 0],
        count: vec![4, 6],
        stride: vec![1, 1],
    };
    let before = nc.fast_reads();
    let s = nc.read_slice(&r).unwrap();
    assert_eq!(nc.fast_reads(), before);
    assert_eq!(&s.data[..3], &[1.0, 2.0, 3.0]);
    assert_eq!(s.data[3], 42.0, "an unwritten chunk reads as HDF5's fill");
    assert_eq!(s.data[23], 42.0);
}

/// On the HDF5 backend the reference is H5Dread, which gives HDF5's fill
/// for never-written chunks: the fast reader serves the dataset and agrees.
#[test]
fn h5_backend_fast_reader_uses_hdf5_fill() {
    let dir = tempfile::tempdir().unwrap();
    let h = NcFile::open_hdf5(&write_hdf5_user_fill(dir.path())).unwrap();
    assert!(h.fast_eligible("/v"));
    for stride in [vec![1, 1], vec![3, 2]] {
        let r = SliceRequest { var: "/v".into(), start: vec![0, 0], count: vec![4, 6], stride };
        let before = h.fast_reads();
        let fast = h.read_slice(&r).unwrap();
        assert_eq!(h.fast_reads(), before + 1);
        let lib = h.read_slice_via_library(&r).unwrap();
        assert_eq!(fast.data, lib.data, "{r:?}");
    }
    let r = SliceRequest { var: "/v".into(), start: vec![0, 0], count: vec![4, 6], stride: vec![1, 1] };
    let s = h.read_slice(&r).unwrap();
    assert_eq!(&s.data[..4], &[1.0, 2.0, 3.0, 42.0]);
    assert_eq!(s.data[23], 42.0);
}
