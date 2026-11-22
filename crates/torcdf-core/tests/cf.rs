use approx::assert_relative_eq;
use torcdf_core::cf::Unpack;
use torcdf_core::fixtures;
use torcdf_core::{Attr, AttrValue, DType, NcFile, Storage, VarInfo};

fn var(dtype: DType, attrs: Vec<(&str, AttrValue)>) -> VarInfo {
    VarInfo {
        name: "v".into(),
        path: "/v".into(),
        dims: vec![],
        shape: vec![],
        dtype,
        attrs: attrs.into_iter().map(|(n, value)| Attr { name: n.into(), value }).collect(),
        storage: Storage::Contiguous,
    }
}

fn f(v: &[f64]) -> AttrValue {
    AttrValue::Floats(v.to_vec())
}

#[test]
fn unpack_scale_offset_fill() {
    let d = tempfile::tempdir().unwrap();
    let file = NcFile::open(&fixtures::write_classic(d.path())).unwrap();
    let u = Unpack::for_var(file.meta.var("/t").unwrap());
    assert_relative_eq!(u.apply(100.0), 283.15, epsilon = 1e-4);
    assert!(u.apply(-999.0).is_nan());
}

#[test]
fn default_fill_applies_without_attr() {
    let d = tempfile::tempdir().unwrap();
    let file = NcFile::open(&fixtures::write_bare(d.path())).unwrap();
    let u = Unpack::for_var(file.meta.var("/data").unwrap());
    assert!(u.apply(9.969209968386869e36).is_nan());
    assert_eq!(u.apply(5.0), 5.0);
    let u = Unpack::for_var(&var(DType::F32, vec![]));
    assert!(u.apply(9.969209968386869e36).is_nan());
    let u = Unpack::for_var(&var(DType::I16, vec![]));
    assert!(u.apply(-32767.0).is_nan());
    assert_eq!(u.apply(-32766.0), -32766.0);
    let u = Unpack::for_var(&var(DType::I32, vec![]));
    assert!(u.apply(-2147483647.0).is_nan());
}

#[test]
fn no_default_fill_for_byte_types() {
    for dt in [DType::I8, DType::U8] {
        let u = Unpack::for_var(&var(dt, vec![]));
        assert_eq!(u.apply(-127.0), -127.0);
        assert_eq!(u.apply(255.0), 255.0);
    }
}

#[test]
fn explicit_fill_replaces_default() {
    let u = Unpack::for_var(&var(DType::I16, vec![("_FillValue", AttrValue::Ints(vec![-1]))]));
    assert!(u.apply(-1.0).is_nan());
    assert_eq!(u.apply(-32767.0), -32767.0);
}

#[test]
fn valid_range_on_packed() {
    let u = Unpack::for_var(&var(
        DType::I16,
        vec![("valid_range", f(&[0.0, 100.0])), ("scale_factor", f(&[2.0]))],
    ));
    assert!(u.apply(101.0).is_nan());
    assert!(u.apply(-1.0).is_nan());
    assert_eq!(u.apply(100.0), 200.0);
    assert_eq!(u.apply(0.0), 0.0);
}

#[test]
fn valid_min_max() {
    let u = Unpack::for_var(&var(
        DType::I16,
        vec![("valid_min", f(&[10.0])), ("valid_max", f(&[20.0]))],
    ));
    assert!(u.apply(9.0).is_nan());
    assert!(u.apply(21.0).is_nan());
    assert_eq!(u.apply(15.0), 15.0);
}

#[test]
fn missing_value_list() {
    let u = Unpack::for_var(&var(DType::I16, vec![("missing_value", f(&[-1.0, -2.0]))]));
    assert!(u.apply(-1.0).is_nan());
    assert!(u.apply(-2.0).is_nan());
    assert!(!u.apply(-3.0).is_nan());
    let u = Unpack::for_var(&var(DType::I16, vec![("missing_value", AttrValue::Ints(vec![7]))]));
    assert!(u.apply(7.0).is_nan());
}

#[test]
fn offset_only_and_u64_fill() {
    let u = Unpack::for_var(&var(DType::F64, vec![("add_offset", f(&[10.0]))]));
    assert_eq!(u.apply(1.0), 11.0);
    // u64 attrs above i64::MAX arrive as Floats; compare cast the same way.
    let big = u64::MAX - 1;
    let u = Unpack::for_var(&var(DType::U64, vec![("_FillValue", f(&[big as f64]))]));
    assert!(u.apply(big as f64).is_nan());
}
