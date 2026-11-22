use approx::assert_relative_eq;
use torcdf_core::cf::{detect_axes, detect_geo, wrap_lon, Geo, TimeUnits, Unpack};
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

fn dt(units: &str, cal: Option<&str>, v: f64) -> String {
    TimeUnits::parse(units, cal).unwrap().decode(v).to_string()
}

#[test]
fn regular_geo() {
    let d = tempfile::tempdir().unwrap();
    let file = NcFile::open(&fixtures::write_classic(d.path())).unwrap();
    let t = file.meta.var("/t").unwrap();
    match detect_geo(&file.meta, t) {
        Geo::Regular { lat_dim: 1, lon_dim: 2, lat_var, lon_var } => {
            assert_eq!((lat_var.as_str(), lon_var.as_str()), ("/lat", "/lon"));
        }
        g => panic!("{g:?}"),
    }
}

#[test]
fn regular_geo_in_group() {
    let d = tempfile::tempdir().unwrap();
    let file = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
    let uo = file.meta.var("/ocean/uo").unwrap();
    assert!(matches!(
        detect_geo(&file.meta, uo),
        Geo::Regular { lat_dim: 1, lon_dim: 2, .. }
    ));
    assert_eq!(detect_axes(&file.meta, uo).time, Some(0));
}

#[test]
fn curvilinear_geo() {
    let d = tempfile::tempdir().unwrap();
    let file = NcFile::open(&fixtures::write_curvilinear(d.path())).unwrap();
    let sst = file.meta.var("/sst").unwrap();
    assert!(matches!(
        detect_geo(&file.meta, sst),
        Geo::Curvilinear { y_dim: 0, x_dim: 1, .. }
    ));
}

#[test]
fn bare_has_no_geo() {
    let d = tempfile::tempdir().unwrap();
    let file = NcFile::open(&fixtures::write_bare(d.path())).unwrap();
    let data = file.meta.var("/data").unwrap();
    assert!(matches!(detect_geo(&file.meta, data), Geo::None));
    let ax = detect_axes(&file.meta, data);
    assert!(ax.time.is_none() && ax.vertical.is_none());
}

#[test]
fn time_axis_found() {
    let d = tempfile::tempdir().unwrap();
    let file = NcFile::open(&fixtures::write_classic(d.path())).unwrap();
    let t = file.meta.var("/t").unwrap();
    assert_eq!(detect_axes(&file.meta, t).time, Some(0));
    assert_eq!(detect_axes(&file.meta, t).vertical, None);
}

#[test]
fn wrap() {
    assert_eq!(wrap_lon(315.0), -45.0);
    assert_eq!(wrap_lon(180.0), -180.0);
    assert_eq!(wrap_lon(-180.0), -180.0);
    assert_eq!(wrap_lon(0.0), 0.0);
    assert_eq!(wrap_lon(540.0), -180.0);
    assert_eq!(wrap_lon(-190.0), 170.0);
}

#[test]
fn noleap_skips_feb29() {
    assert_eq!(dt("days since 2000-02-28", Some("noleap"), 1.0), "2000-03-01 00:00");
    assert_eq!(dt("days since 2000-01-01", Some("365_day"), 365.0), "2001-01-01 00:00");
    assert_eq!(dt("days since 2000-03-01", Some("noleap"), -1.0), "2000-02-28 00:00");
}

#[test]
fn standard_has_feb29() {
    assert_eq!(dt("days since 2000-02-28", None, 1.0), "2000-02-29 00:00");
    assert_eq!(dt("days since 1900-02-28", None, 1.0), "1900-03-01 00:00");
    assert_eq!(dt("days since 1900-02-28", Some("proleptic_gregorian"), 1.0), "1900-03-01 00:00");
}

#[test]
fn all_leap() {
    assert_eq!(dt("days since 2001-02-28", Some("all_leap"), 1.0), "2001-02-29 00:00");
    assert_eq!(dt("days since 2001-01-01", Some("366_day"), 366.0), "2002-01-01 00:00");
}

#[test]
fn day360() {
    assert_eq!(dt("days since 2000-01-30", Some("360_day"), 1.0), "2000-02-01 00:00");
    assert_eq!(dt("days since 2000-01-01", Some("360_day"), 360.0), "2001-01-01 00:00");
    assert_eq!(dt("days since 2000-01-01", Some("360_day"), 59.0), "2000-02-30 00:00");
}

#[test]
fn hours_iso_epoch() {
    assert_eq!(
        dt("hours since 1970-01-01T00:00:00Z", Some("gregorian"), 492_756.0),
        "2026-03-19 12:00"
    );
}

#[test]
fn julian_gregorian_switch() {
    assert_eq!(dt("days since 1582-10-04", Some("standard"), 1.0), "1582-10-15 00:00");
    // proleptic Gregorian has no gap
    assert_eq!(
        dt("days since 1582-10-04", Some("proleptic_gregorian"), 1.0),
        "1582-10-05 00:00"
    );
    // pure Julian has a Feb 29 in 1900
    assert_eq!(dt("days since 1900-02-28", Some("julian"), 1.0), "1900-02-29 00:00");
    assert_eq!(dt("days since 1582-10-15", Some("standard"), -1.0), "1582-10-04 00:00");
}

#[test]
fn unit_variants_and_seconds_display() {
    assert_eq!(dt("seconds since 2000-01-01 00:00:00", None, 90.0), "2000-01-01 00:01:30");
    assert_eq!(dt("minutes since 2000-01-01", None, 90.0), "2000-01-01 01:30");
    assert_eq!(dt("hr since 2000-01-01 06:00 UTC", None, 1.0), "2000-01-01 07:00");
    assert_eq!(dt("d since 2000-01-01", Some("STANDARD"), 0.5), "2000-01-01 12:00");
    assert_eq!(dt("days since 2000-1-1", None, 0.1), "2000-01-01 02:24");
    assert_eq!(dt("days since 2000-01-01", None, -0.5), "1999-12-31 12:00");
    assert_eq!(dt("days since 2000-01-01", None, 0.999999999), "2000-01-02 00:00");
}

#[test]
fn month_year_units() {
    // udunits year = 365.242198781 days = 31556925.9747 s
    assert_eq!(dt("years since 2000-01-01", None, 1.0), "2000-12-31 05:48:45.975");
    assert_eq!(dt("months since 2000-01-01", Some("standard"), 12.0), "2000-12-31 05:48:45.975");
}

#[test]
fn bad_units_is_none() {
    assert!(TimeUnits::parse("furlongs", None).is_none());
    assert!(TimeUnits::parse("fortnights since 2000-01-01", None).is_none());
    assert!(TimeUnits::parse("days since garbage", None).is_none());
    assert!(TimeUnits::parse("days since 2000-13-01", None).is_none());
    assert!(TimeUnits::parse("days since 2000-01-01", Some("martian")).is_none());
}

#[test]
fn calendar_round_trip() {
    // decode -> format -> parse as epoch -> decode(0) must be the identity,
    // and successive samples must differ, for every calendar across eras.
    for cal in ["standard", "proleptic_gregorian", "julian", "noleap", "all_leap", "360_day"] {
        let base = TimeUnits::parse("days since 1500-01-01", Some(cal)).unwrap();
        let mut prev = String::new();
        for n in (0..200_000).step_by(7) {
            let s = base.decode(n as f64).to_string();
            assert_ne!(s, prev);
            let date = &s[..10];
            let back = dt(&format!("days since {date}"), Some(cal), 0.0);
            assert_eq!(back, format!("{date} 00:00"), "{cal} day {n}");
            prev = s;
        }
    }
}
