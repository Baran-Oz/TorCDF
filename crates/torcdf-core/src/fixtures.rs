//! Fixture netCDF writers for tests. Enabled by the `fixtures` cargo feature.
//!
//! Every writer takes a directory and returns the path of the file it created.

use std::path::{Path, PathBuf};

/// netCDF-3 classic `classic.nc`: `t(time,lat,lon)` i16 packed with
/// scale/offset/fill on a CF-style grid with an unlimited `time` dimension.
pub fn write_classic(dir: &Path) -> PathBuf {
    let path = dir.join("classic.nc");
    let mut f = netcdf::create_with(&path, netcdf::Options::default()).unwrap();
    f.add_unlimited_dimension("time").unwrap();
    f.add_dimension("lat", 4).unwrap();
    f.add_dimension("lon", 8).unwrap();

    // Classic files must finish all definitions before any data is written.
    {
        let mut time = f.add_variable::<f64>("time", &["time"]).unwrap();
        time.put_attribute("units", "days since 2000-02-28").unwrap();
        time.put_attribute("calendar", "noleap").unwrap();
    }
    {
        let mut lat = f.add_variable::<f64>("lat", &["lat"]).unwrap();
        lat.put_attribute("units", "degrees_north").unwrap();
    }
    {
        let mut lon = f.add_variable::<f64>("lon", &["lon"]).unwrap();
        lon.put_attribute("units", "degrees_east").unwrap();
    }
    {
        let mut t = f
            .add_variable::<i16>("t", &["time", "lat", "lon"])
            .unwrap();
        t.set_fill_value(-999i16).unwrap();
        t.put_attribute("scale_factor", 0.1f64).unwrap();
        t.put_attribute("add_offset", 273.15f64).unwrap();
    }
    f.enddef().unwrap();

    f.variable_mut("time")
        .unwrap()
        .put_values(&[0.0f64, 1.0, 2.0], 0..3)
        .unwrap();
    f.variable_mut("lat")
        .unwrap()
        .put_values(&[-45.0f64, -15.0, 15.0, 45.0], ..)
        .unwrap();
    let lons: Vec<f64> = (0..8).map(|i| i as f64 * 45.0).collect();
    f.variable_mut("lon").unwrap().put_values(&lons, ..).unwrap();

    let mut raw = Vec::with_capacity(3 * 4 * 8);
    for i in 0..3i16 {
        for j in 0..4i16 {
            for k in 0..8i16 {
                raw.push(i * 100 + j * 10 + k);
            }
        }
    }
    raw[0] = -999;
    f.variable_mut("t")
        .unwrap()
        .put_values(&raw, (0..3, 0..4, 0..8))
        .unwrap();
    path
}

/// netCDF-4 `nc4.nc`: group `ocean` with chunked, deflated `uo`/`vo`, a string
/// variable and a global title.
pub fn write_nc4(dir: &Path) -> PathBuf {
    let path = dir.join("nc4.nc");
    let mut f = netcdf::create(&path).unwrap();
    f.add_attribute("title", "fixture").unwrap();
    let mut g = f.add_group("ocean").unwrap();
    g.add_dimension("time", 24).unwrap();
    g.add_dimension("lat", 10).unwrap();
    g.add_dimension("lon", 20).unwrap();

    let mut time = g.add_variable::<f64>("time", &["time"]).unwrap();
    time.put_attribute("units", "hours since 2000-01-01").unwrap();
    let times: Vec<f64> = (0..24).map(f64::from).collect();
    time.put_values(&times, ..).unwrap();

    let mut lat = g.add_variable::<f64>("lat", &["lat"]).unwrap();
    lat.put_attribute("units", "degrees_north").unwrap();
    let lats: Vec<f64> = (0..10).map(|i| -9.0 + 2.0 * f64::from(i)).collect();
    lat.put_values(&lats, ..).unwrap();

    let mut lon = g.add_variable::<f64>("lon", &["lon"]).unwrap();
    lon.put_attribute("units", "degrees_east").unwrap();
    let lons: Vec<f64> = (0..20).map(f64::from).collect();
    lon.put_values(&lons, ..).unwrap();

    for (name, std_name, scale) in [
        ("uo", "eastward_sea_water_velocity", 1.0f32),
        ("vo", "northward_sea_water_velocity", -1.0f32),
    ] {
        let mut v = g
            .add_variable::<f32>(name, &["time", "lat", "lon"])
            .unwrap();
        v.set_chunking(&[1, 10, 20]).unwrap();
        v.set_compression(4, false).unwrap();
        v.set_fill_value(1e20f32).unwrap();
        v.put_attribute("standard_name", std_name).unwrap();
        v.put_attribute("units", "m s-1").unwrap();
        let mut data = Vec::with_capacity(24 * 10 * 20);
        for t in 0..24 {
            for j in 0..10 {
                for k in 0..20 {
                    data.push(if k < 3 {
                        1e20f32
                    } else {
                        scale * (t as f32 * 0.01 + j as f32 * 0.1 + k as f32 * 0.001)
                    });
                }
            }
        }
        v.put_values(&data, ..).unwrap();
    }

    let mut label = g.add_string_variable("label", &["time"]).unwrap();
    for i in 0..24usize {
        label.put_string(&format!("step-{i}"), i).unwrap();
    }
    path
}

/// `curv.nc`: curvilinear grid with 2-D `nav_lat`/`nav_lon` and `sst`.
pub fn write_curvilinear(dir: &Path) -> PathBuf {
    let path = dir.join("curv.nc");
    let mut f = netcdf::create(&path).unwrap();
    f.add_dimension("y", 5).unwrap();
    f.add_dimension("x", 6).unwrap();
    let mut lat = f.add_variable::<f32>("nav_lat", &["y", "x"]).unwrap();
    let lats: Vec<f32> = (0..30).map(|n| (n / 6) as f32 * 2.0 + (n % 6) as f32 * 0.1).collect();
    lat.put_values(&lats, ..).unwrap();
    let mut lon = f.add_variable::<f32>("nav_lon", &["y", "x"]).unwrap();
    let lons: Vec<f32> = (0..30).map(|n| (n % 6) as f32 * 2.0 + (n / 6) as f32 * 0.3).collect();
    lon.put_values(&lons, ..).unwrap();
    let mut sst = f.add_variable::<f32>("sst", &["y", "x"]).unwrap();
    sst.put_attribute("coordinates", "nav_lat nav_lon").unwrap();
    let vals: Vec<f32> = (0..30).map(|n| 10.0 + n as f32 * 0.5).collect();
    sst.put_values(&vals, ..).unwrap();
    path
}

/// `bare.nc`: a single `data(a,b)` f64 variable with no attributes anywhere.
pub fn write_bare(dir: &Path) -> PathBuf {
    let path = dir.join("bare.nc");
    let mut f = netcdf::create(&path).unwrap();
    f.add_dimension("a", 3).unwrap();
    f.add_dimension("b", 4).unwrap();
    let mut v = f.add_variable::<f64>("data", &["a", "b"]).unwrap();
    let vals: Vec<f64> = (0..12).map(f64::from).collect();
    v.put_values(&vals, ..).unwrap();
    path
}
