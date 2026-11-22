//! Verifies that the shared fixture writers produce exactly what later tests
//! rely on. A single test function: it calls the raw netCDF C API (not
//! thread-safe) for format and deflate checks.

use netcdf::AttributeValue;
use std::ffi::CString;
use std::path::Path;
use torcdf_core::fixtures;

fn str_attr(v: &netcdf::Variable, name: &str) -> String {
    match v.attribute(name).unwrap_or_else(|| panic!("missing {name}")).value().unwrap() {
        AttributeValue::Str(s) => s,
        other => panic!("{name}: expected string, got {other:?}"),
    }
}

fn f64_attr(v: &netcdf::Variable, name: &str) -> f64 {
    match v.attribute(name).unwrap_or_else(|| panic!("missing {name}")).value().unwrap() {
        AttributeValue::Double(x) => x,
        other => panic!("{name}: expected double, got {other:?}"),
    }
}

/// Returns (nc_inq_format, deflate flag, deflate level) of `var` in `group`
/// (empty group = root) using the raw C API.
fn raw_inspect(path: &Path, group: &str, var: &str) -> (i32, i32, i32) {
    use netcdf_sys::*;
    let cpath = CString::new(path.to_str().unwrap()).unwrap();
    unsafe {
        let mut ncid = 0;
        assert_eq!(nc_open(cpath.as_ptr(), 0, &mut ncid), NC_NOERR);
        let mut format = 0;
        assert_eq!(nc_inq_format(ncid, &mut format), NC_NOERR);
        let (mut shuffle, mut deflate, mut level) = (0, 0, 0);
        if !var.is_empty() {
            let mut gid = ncid;
            if !group.is_empty() {
                let g = CString::new(group).unwrap();
                assert_eq!(nc_inq_grp_ncid(ncid, g.as_ptr(), &mut gid), NC_NOERR);
            }
            let v = CString::new(var).unwrap();
            let mut varid = 0;
            assert_eq!(nc_inq_varid(gid, v.as_ptr(), &mut varid), NC_NOERR);
            assert_eq!(
                nc_inq_var_deflate(gid, varid, &mut shuffle, &mut deflate, &mut level),
                NC_NOERR
            );
        }
        assert_eq!(nc_close(ncid), NC_NOERR);
        (format, deflate, level)
    }
}

#[test]
fn fixture_contents() {
    let dir = tempfile::tempdir().unwrap();

    // classic.nc
    let p = fixtures::write_classic(dir.path());
    assert_eq!(raw_inspect(&p, "", "").0, netcdf_sys::NC_FORMAT_CLASSIC);
    let f = netcdf::open(&p).unwrap();
    let dim = |n: &str| f.dimension(n).unwrap();
    assert_eq!(dim("time").len(), 3);
    assert!(dim("time").is_unlimited());
    assert_eq!(dim("lat").len(), 4);
    assert!(!dim("lat").is_unlimited());
    assert_eq!(dim("lon").len(), 8);
    let time = f.variable("time").unwrap();
    assert_eq!(str_attr(&time, "units"), "days since 2000-02-28");
    assert_eq!(str_attr(&time, "calendar"), "noleap");
    assert_eq!(time.get_values::<f64, _>(..).unwrap(), vec![0.0, 1.0, 2.0]);
    let lat = f.variable("lat").unwrap();
    assert_eq!(str_attr(&lat, "units"), "degrees_north");
    assert_eq!(lat.get_values::<f64, _>(..).unwrap(), vec![-45.0, -15.0, 15.0, 45.0]);
    let lon = f.variable("lon").unwrap();
    assert_eq!(str_attr(&lon, "units"), "degrees_east");
    let lons = lon.get_values::<f64, _>(..).unwrap();
    assert_eq!(lons, (0..8).map(|i| i as f64 * 45.0).collect::<Vec<_>>());
    let t = f.variable("t").unwrap();
    let t_dims: Vec<_> = t.dimensions().iter().map(|d| d.name()).collect();
    assert_eq!(t_dims, ["time", "lat", "lon"]);
    assert_eq!(f64_attr(&t, "scale_factor"), 0.1);
    assert_eq!(f64_attr(&t, "add_offset"), 273.15);
    assert_eq!(t.fill_value::<i16>().unwrap(), Some(-999));
    let raw = t.get_values::<i16, _>(..).unwrap();
    assert_eq!(raw.len(), 96);
    assert_eq!(raw[0], -999);
    assert_eq!(raw[1], 1); // t[0,0,1]
    assert_eq!(raw[2 * 32 + 3 * 8 + 5], 235); // t[2,3,5]
    drop(f);

    // nc4.nc
    let p = fixtures::write_nc4(dir.path());
    let (format, deflate, level) = raw_inspect(&p, "ocean", "uo");
    assert_eq!(format, netcdf_sys::NC_FORMAT_NETCDF4);
    assert_eq!((deflate, level), (1, 4));
    let (_, deflate, level) = raw_inspect(&p, "ocean", "vo");
    assert_eq!((deflate, level), (1, 4));
    let f = netcdf::open(&p).unwrap();
    match f.attribute("title").unwrap().value().unwrap() {
        AttributeValue::Str(s) => assert_eq!(s, "fixture"),
        other => panic!("title: {other:?}"),
    }
    let g = f.group("ocean").unwrap().unwrap();
    assert_eq!(g.dimension("time").unwrap().len(), 24);
    assert_eq!(g.dimension("lat").unwrap().len(), 10);
    assert_eq!(g.dimension("lon").unwrap().len(), 20);
    for (name, std_name) in [
        ("uo", "eastward_sea_water_velocity"),
        ("vo", "northward_sea_water_velocity"),
    ] {
        let v = g.variable(name).unwrap();
        assert_eq!(str_attr(&v, "standard_name"), std_name);
        assert_eq!(v.chunking().unwrap(), Some(vec![1, 10, 20]));
        assert_eq!(v.fill_value::<f32>().unwrap(), Some(1e20f32));
        let data = v.get_values::<f32, _>(..).unwrap();
        assert_eq!(data.len(), 24 * 10 * 20);
        for t in 0..24 {
            for j in 0..10 {
                for k in 0..20 {
                    let x = data[t * 200 + j * 20 + k];
                    assert_eq!(x == 1e20f32, k < 3, "{name}[{t},{j},{k}] = {x}");
                }
            }
        }
        assert_eq!(data[0], 1e20f32);
        assert_ne!(data[3], 1e20f32);
    }
    for c in ["time", "lat", "lon"] {
        assert_eq!(g.variable(c).unwrap().dimensions().len(), 1);
    }
    let label = g.variable("label").unwrap();
    assert_eq!(label.dimensions()[0].name(), "time");
    assert_eq!(label.get_string(0).unwrap(), "step-0");
    assert_eq!(label.get_string(23).unwrap(), "step-23");
    drop(f);

    // curv.nc
    let p = fixtures::write_curvilinear(dir.path());
    let f = netcdf::open(&p).unwrap();
    assert_eq!(f.dimension("y").unwrap().len(), 5);
    assert_eq!(f.dimension("x").unwrap().len(), 6);
    for n in ["nav_lat", "nav_lon"] {
        let v = f.variable(n).unwrap();
        let dims: Vec<_> = v.dimensions().iter().map(|d| d.name()).collect();
        assert_eq!(dims, ["y", "x"]);
        assert_eq!(v.attributes().count(), 0, "{n} must have no units");
    }
    let sst = f.variable("sst").unwrap();
    assert_eq!(str_attr(&sst, "coordinates"), "nav_lat nav_lon");
    drop(f);

    // bare.nc
    let p = fixtures::write_bare(dir.path());
    let f = netcdf::open(&p).unwrap();
    assert_eq!(f.attributes().count(), 0);
    assert_eq!(f.dimension("a").unwrap().len(), 3);
    assert_eq!(f.dimension("b").unwrap().len(), 4);
    assert_eq!(f.variables().count(), 1);
    let v = f.variable("data").unwrap();
    assert_eq!(v.attributes().count(), 0);
    assert_eq!(
        v.get_values::<f64, _>(..).unwrap(),
        (0..12).map(f64::from).collect::<Vec<_>>()
    );
}
