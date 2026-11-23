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

/// Records, rows and columns of [`write_fast_matrix`]'s variables.
const FAST_DIMS: [usize; 3] = [7, 13, 17];

/// The smooth field of [`write_fast_matrix`]; `None` marks a missing value.
fn fast_field(t: usize, y: usize, x: usize) -> Option<f64> {
    if (t * 7 + y * 3 + x).is_multiple_of(13) {
        return None;
    }
    Some(10.0 + 3.0 * (y as f64 * 0.3).sin() + 2.0 * (x as f64 * 0.25).cos() + t as f64 * 0.5)
}

/// The field over all of `FAST_DIMS`, row-major, through `map`.
fn fast_grid<T>(map: impl Fn(Option<f64>) -> T) -> Vec<T> {
    let [nt, ny, nx] = FAST_DIMS;
    let mut out = Vec::with_capacity(nt * ny * nx);
    for t in 0..nt {
        for y in 0..ny {
            for x in 0..nx {
                out.push(map(fast_field(t, y, x)));
            }
        }
    }
    out
}

/// netCDF-4 `fast.nc` for the fast chunk reader: `time` (unlimited, 7
/// records written), `y` = 13, `x` = 17. Every variable but `contig_f32` is
/// `(time, y, x)` with chunks `[2, 5, 6]`, so every axis ends in a partial
/// chunk:
/// - filters: `sd_f32` shuffle + deflate, `d_f64` deflate, `s_i32` shuffle
///   only, `fl_f32` Fletcher32 + deflate;
/// - packing: `p_i16` (scale 0.01, offset 20, `_FillValue` -999), `mv_f32`
///   (`missing_value` -1, no `_FillValue`);
/// - `be_f32` big-endian; `t_i8` .. `t_u64` deflated, one per integer type;
/// - `sparse_f32`: only `time 0..2, y 0..5, x 0..6` (one chunk) written;
/// - `contig_f32 (y, x)`: contiguous, so not eligible (HDF5 cannot store a
///   variable with an unlimited dimension contiguously).
pub fn write_fast_matrix(dir: &Path) -> PathBuf {
    use netcdf::{Endianness, FileMut, NcTypeDescriptor, VariableMut};

    const DIMS: [&str; 3] = ["time", "y", "x"];
    const CHUNK: [usize; 3] = [2, 5, 6];
    let [_, ny, nx] = FAST_DIMS;

    fn add<T: NcTypeDescriptor>(
        f: &mut FileMut,
        name: &str,
        setup: impl FnOnce(&mut VariableMut),
        data: &[T],
    ) {
        let mut v = f.add_variable::<T>(name, &DIMS).unwrap();
        v.set_chunking(&CHUNK).unwrap();
        setup(&mut v);
        let [nt, ny, nx] = FAST_DIMS;
        v.put_values(data, (0..nt, 0..ny, 0..nx)).unwrap();
    }

    let path = dir.join("fast.nc");
    let mut f = netcdf::create(&path).unwrap();
    f.add_unlimited_dimension("time").unwrap();
    f.add_dimension("y", ny).unwrap();
    f.add_dimension("x", nx).unwrap();

    let f32_or = |fill: f32| move |v: Option<f64>| v.map_or(fill, |v| v as f32);
    const NC_FILL_F64: f64 = 9.969_209_968_386_869e36;
    const NC_FILL_F32: f32 = NC_FILL_F64 as f32;

    add(
        &mut f,
        "sd_f32",
        |v| {
            v.set_compression(4, true).unwrap();
            v.set_fill_value(1e20f32).unwrap();
        },
        &fast_grid(f32_or(1e20)),
    );
    add(
        &mut f,
        "d_f64",
        |v| v.set_compression(4, false).unwrap(),
        &fast_grid(|v| v.unwrap_or(NC_FILL_F64)),
    );
    add(
        &mut f,
        "p_i16",
        |v| {
            v.set_compression(2, true).unwrap();
            v.set_fill_value(-999i16).unwrap();
            v.put_attribute("scale_factor", 0.01f64).unwrap();
            v.put_attribute("add_offset", 20.0f64).unwrap();
        },
        &fast_grid(|v| v.map_or(-999i16, |v| ((v - 20.0) / 0.01).round() as i16)),
    );
    add(
        &mut f,
        "mv_f32",
        |v| {
            v.set_compression(4, false).unwrap();
            v.put_attribute("missing_value", -1.0f32).unwrap();
        },
        &fast_grid(f32_or(-1.0)),
    );
    add(
        &mut f,
        "be_f32",
        |v| {
            v.set_compression(4, true).unwrap();
            v.set_endianness(Endianness::Big).unwrap();
        },
        &fast_grid(f32_or(NC_FILL_F32)),
    );
    let deflate = |v: &mut VariableMut| v.set_compression(4, true).unwrap();
    add(&mut f, "t_i8", deflate, &fast_grid(|v| v.map_or(-127i8, |v| (v * 5.0) as i8)));
    add(&mut f, "t_u8", deflate, &fast_grid(|v| v.map_or(255u8, |v| (v * 10.0) as u8)));
    add(&mut f, "t_u16", deflate, &fast_grid(|v| v.map_or(65535u16, |v| (v * 1e3) as u16)));
    add(
        &mut f,
        "t_u32",
        deflate,
        &fast_grid(|v| v.map_or(4_294_967_295u32, |v| (v * 1e6) as u32)),
    );
    // Beyond 2^53, so conversion to f64 rounds.
    add(
        &mut f,
        "t_i64",
        deflate,
        &fast_grid(|v| v.map_or(-9_223_372_036_854_775_806i64, |v| -(1i64 << 60) + (v * 1e6) as i64 * 7)),
    );
    add(
        &mut f,
        "t_u64",
        deflate,
        &fast_grid(|v| v.map_or(18_446_744_073_709_551_614u64, |v| (1u64 << 63) + (v * 1e6) as u64 * 7)),
    );
    {
        let mut v = f.add_variable::<f32>("sparse_f32", &DIMS).unwrap();
        v.set_chunking(&CHUNK).unwrap();
        v.set_compression(4, true).unwrap();
        let block: Vec<f32> = (0..2 * 5 * 6)
            .map(|i| fast_field(i / 30, i / 6 % 5, i % 6).map_or(NC_FILL_F32, |v| v as f32))
            .collect();
        v.put_values(&block, (0..2, 0..5, 0..6)).unwrap();
    }
    {
        let mut v = f.add_variable::<f32>("contig_f32", &["y", "x"]).unwrap();
        let vals = fast_grid(f32_or(NC_FILL_F32));
        v.put_values(&vals[..ny * nx], (0..ny, 0..nx)).unwrap();
    }
    drop(f);

    // Shuffle without deflate, and Fletcher32, have no netcdf crate setter.
    add_raw_fast_vars(
        &path,
        &fast_grid(|v| v.map_or(-9999i32, |v| (v * 100.0).round() as i32)),
        &fast_grid(f32_or(NC_FILL_F32)),
    );
    path
}

/// Adds `s_i32` (shuffle only, `_FillValue` -9999) and `fl_f32` (Fletcher32
/// + deflate) to `fast.nc` through netCDF-C, under the netCDF lock.
fn add_raw_fast_vars(path: &Path, s_i32: &[i32], fl_f32: &[f32]) {
    use netcdf_sys::*;
    use std::ffi::CString;
    use std::os::raw::c_int;

    let ok = |rc: c_int| assert_eq!(rc, NC_NOERR, "netCDF error {rc}");
    let c = CString::new(path.to_str().unwrap()).unwrap();
    let _g = libnetcdf_lock.lock();
    unsafe {
        let mut ncid = 0;
        ok(nc_open(c.as_ptr(), NC_WRITE, &mut ncid));
        ok(nc_redef(ncid));
        let mut dims = [0 as c_int; 3];
        for (id, name) in dims.iter_mut().zip(["time", "y", "x"]) {
            let n = CString::new(name).unwrap();
            ok(nc_inq_dimid(ncid, n.as_ptr(), id));
        }
        let chunk = [2usize, 5, 6];
        let def = |name: &str, xtype: nc_type| {
            let n = CString::new(name).unwrap();
            let mut varid = 0;
            ok(nc_def_var(ncid, n.as_ptr(), xtype, 3, dims.as_ptr(), &mut varid));
            ok(nc_def_var_chunking(ncid, varid, NC_CHUNKED, chunk.as_ptr()));
            varid
        };
        let s_id = def("s_i32", NC_INT);
        ok(nc_def_var_deflate(ncid, s_id, 1, 0, 0));
        let fill: i32 = -9999;
        ok(nc_def_var_fill(ncid, s_id, 0, (&raw const fill).cast()));
        let fl_id = def("fl_f32", NC_FLOAT);
        ok(nc_def_var_fletcher32(ncid, fl_id, 1));
        ok(nc_def_var_deflate(ncid, fl_id, 0, 1, 4));
        ok(nc_enddef(ncid));
        let start = [0usize; 3];
        let count = FAST_DIMS;
        ok(nc_put_vara_int(ncid, s_id, start.as_ptr(), count.as_ptr(), s_i32.as_ptr()));
        ok(nc_put_vara_float(ncid, fl_id, start.as_ptr(), count.as_ptr(), fl_f32.as_ptr()));
        ok(nc_close(ncid));
    }
}
