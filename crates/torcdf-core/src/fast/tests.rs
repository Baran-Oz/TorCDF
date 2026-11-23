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

#[test]
fn fast_equals_library_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let nc = NcFile::open(&fixtures::write_fast_matrix(dir.path())).unwrap();
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
