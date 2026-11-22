use approx::assert_relative_eq;
use torcdf_core::fixtures;
use torcdf_core::{CoreError, NcFile, SliceRequest};

fn req(var: &str, start: &[usize], count: &[usize], stride: &[usize]) -> SliceRequest {
    SliceRequest {
        var: var.into(),
        start: start.to_vec(),
        count: count.to_vec(),
        stride: stride.to_vec(),
    }
}

fn classic() -> (tempfile::TempDir, NcFile) {
    let d = tempfile::tempdir().unwrap();
    let f = NcFile::open(&fixtures::write_classic(d.path())).unwrap();
    (d, f)
}

fn nc4() -> (tempfile::TempDir, NcFile) {
    let d = tempfile::tempdir().unwrap();
    let f = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
    (d, f)
}

#[test]
fn reads_unpacked_slice() {
    let (_d, f) = classic();
    let s = f.read_slice(&req("/t", &[1, 0, 0], &[1, 4, 8], &[1, 1, 1])).unwrap();
    assert_eq!(s.shape, vec![1, 4, 8]);
    assert_relative_eq!(s.data[9], ((100 + 10 + 1) as f64 * 0.1 + 273.15) as f32, epsilon = 1e-3);
    assert_eq!(s.valid_count, 32);
    assert_eq!(s.byte_size(), 32 * 4);
}

#[test]
fn fill_becomes_nan() {
    let (_d, f) = classic();
    let s = f.read_slice(&req("/t", &[0, 0, 0], &[1, 1, 1], &[1, 1, 1])).unwrap();
    assert!(s.data[0].is_nan());
    assert_eq!(s.valid_count, 0);
}

#[test]
fn strided_matches_full_read_thinned() {
    let (_d, f) = nc4();
    let full = f.read_slice(&req("/ocean/uo", &[0, 0, 0], &[24, 10, 20], &[1, 1, 1])).unwrap();
    let thin = f.read_slice(&req("/ocean/uo", &[0, 0, 0], &[24, 10, 20], &[5, 3, 4])).unwrap();
    assert_eq!(thin.shape, vec![5, 4, 5]);
    let mut nan_seen = false;
    for i in 0..5 {
        for j in 0..4 {
            for k in 0..5 {
                let a = thin.data[(i * 4 + j) * 5 + k];
                let b = full.data[((i * 5) * 10 + j * 3) * 20 + k * 4];
                nan_seen |= a.is_nan();
                assert!((a.is_nan() && b.is_nan()) || a == b, "({i},{j},{k}): {a} vs {b}");
            }
        }
    }
    assert!(nan_seen, "fixture has fill at lon < 3, so k=0 must be NaN");
}

#[test]
fn strided_with_offset_and_partial_window() {
    let (_d, f) = nc4();
    let full = f.read_slice(&req("/ocean/uo", &[0, 0, 0], &[24, 10, 20], &[1, 1, 1])).unwrap();
    let s = f.read_slice(&req("/ocean/uo", &[2, 1, 3], &[7, 8, 15], &[3, 2, 1])).unwrap();
    assert_eq!(s.shape, vec![3, 4, 15]);
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..15 {
                let a = s.data[(i * 4 + j) * 15 + k];
                let b = full.data[(((2 + i * 3) * 10) + 1 + j * 2) * 20 + 3 + k];
                assert!((a.is_nan() && b.is_nan()) || a == b);
            }
        }
    }
}

#[test]
fn point_series() {
    let (_d, f) = nc4();
    let s = f.read_slice(&req("/ocean/uo", &[0, 5, 10], &[24, 1, 1], &[1, 1, 1])).unwrap();
    assert_eq!(s.shape, vec![24, 1, 1]);
    assert_eq!(s.data.len(), 24);
}

#[test]
fn string_var_unsupported() {
    let (_d, f) = nc4();
    assert!(matches!(
        f.read_slice(&req("/ocean/label", &[0], &[1], &[1])),
        Err(CoreError::Unsupported(_))
    ));
}

#[test]
fn bad_requests_are_errors_not_panics() {
    let (_d, f) = classic();
    let bad = [
        req("/t", &[0, 0, 0], &[4, 4, 8], &[1, 1, 1]),          // count past end
        req("/t", &[3, 0, 0], &[1, 1, 1], &[1, 1, 1]),          // start past end
        req("/t", &[0, 0], &[1, 1], &[1, 1]),                   // wrong rank
        req("/t", &[0, 0, 0], &[1, 1, 1], &[1, 0, 1]),          // zero stride
        req("/t", &[0, 0, 0], &[1, 0, 1], &[1, 1, 1]),          // zero count
        req("/nope", &[0], &[1], &[1]),                         // unknown var
        req("/t", &[usize::MAX, 0, 0], &[2, 1, 1], &[1, 1, 1]), // overflow
    ];
    for r in &bad {
        assert!(f.read_slice(r).is_err(), "{r:?}");
    }
}

#[test]
fn reads_coordinates() {
    let (_d, f) = classic();
    assert_eq!(f.read_coord("/lon").unwrap().len(), 8);
    assert!(f.read_coord("/t").is_err()); // rank 3
    assert!(f.read_coord("/nope").is_err());

    let d = tempfile::tempdir().unwrap();
    let c = NcFile::open(&fixtures::write_curvilinear(d.path())).unwrap();
    let v = c.meta.all_vars().into_iter().find(|v| v.shape.len() == 2).unwrap();
    let n: usize = v.shape.iter().product();
    assert_eq!(c.read_coord(&v.path.clone()).unwrap().len(), n);
}

#[test]
fn read_strings_nc_string() {
    let (_d, f) = nc4();
    let all = f.read_strings("/ocean/label", &[0], &[24]).unwrap();
    let want: Vec<String> = (0..24).map(|i| format!("step-{i}")).collect();
    assert_eq!(all, want);
    assert_eq!(f.read_strings("/ocean/label", &[5], &[3]).unwrap(), ["step-5", "step-6", "step-7"]);
}

/// NC_CHAR element for writing a char-array fixture.
#[repr(transparent)]
#[derive(Copy, Clone)]
struct NcChar(u8);
unsafe impl netcdf::types::NcTypeDescriptor for NcChar {
    fn type_descriptor() -> netcdf::types::NcVariableType {
        netcdf::types::NcVariableType::Char
    }
}

#[test]
fn read_strings_char_array() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("chars.nc");
    {
        let mut nc = netcdf::create(&path).unwrap();
        nc.add_dimension("n", 3).unwrap();
        nc.add_dimension("strlen", 5).unwrap();
        let mut v = nc.add_variable::<NcChar>("names", &["n", "strlen"]).unwrap();
        // NUL-padded rows; the last uses the full width.
        let raw = b"ab\0\0\0xyz\0\0hello";
        let vals: Vec<NcChar> = raw.iter().map(|&b| NcChar(b)).collect();
        v.put_values(&vals, ..).unwrap();
    }
    let f = NcFile::open(&path).unwrap();
    assert_eq!(f.read_strings("/names", &[0, 0], &[3, 5]).unwrap(), ["ab", "xyz", "hello"]);
    assert_eq!(f.read_strings("/names", &[1, 0], &[2, 5]).unwrap(), ["xyz", "hello"]);
}

#[test]
fn read_strings_rejects_numeric_and_bad_requests() {
    let (_d, f) = nc4();
    assert!(matches!(
        f.read_strings("/ocean/uo", &[0, 0, 0], &[1, 1, 1]),
        Err(CoreError::Unsupported(_))
    ));
    assert!(f.read_strings("/ocean/label", &[20], &[5]).is_err()); // past end
    assert!(f.read_strings("/ocean/label", &[0, 0], &[1, 1]).is_err()); // wrong rank
    assert!(f.read_strings("/ocean/label", &[0], &[0]).is_err()); // zero count
    assert!(f.read_strings("/nope", &[0], &[1]).is_err());
}
