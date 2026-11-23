//! Vector fixtures: model-like files whose components hold a known field.
//!
//! At centre-grid index position (p, q) (x, y; half-integers for staggered
//! points) the field is east `E = known_east(p)` and north
//! `N = known_north(q)`, in m s-1. Files store the grid components
//! `u = E cos α + N sin α` and `v = −E sin α + N cos α`, each at its own
//! point. Turned grids are straight in Mercator coordinates (conformal, so
//! the local grid angle on the sphere is θ everywhere, α = θ): with
//! `k = 0.1° / cos 40°` (in radians), position (p, q) sits at
//! `lon = 5° + k (p cos θ − q sin θ)` and
//! `merc(lat) = merc(40°) + k (p sin θ + q cos θ)`, where
//! `merc(φ) = ln tan(π/4 + φ/2)`: about 0.1° per cell around (40°N, 5°E).

use netcdf::{FileMut, NcTypeDescriptor};
use std::path::{Path, PathBuf};

/// Eastward component at centre-grid x position `p`.
pub fn known_east(p: f64) -> f64 {
    1.0 + 0.1 * p
}

/// Northward component at centre-grid y position `q`.
pub fn known_north(q: f64) -> f64 {
    0.5 + 0.05 * q
}

/// Latitude and longitude of position (p, q) on a grid turned by `theta`
/// (straight in Mercator coordinates; see the module docs).
fn lat_lon(theta: f64, p: f64, q: f64) -> (f64, f64) {
    let (s, c) = theta.sin_cos();
    let k = 0.1f64.to_radians() / 40f64.to_radians().cos();
    let merc = |lat: f64| (std::f64::consts::FRAC_PI_4 + lat / 2.0).tan().ln();
    let y = merc(40f64.to_radians()) + k * (p * s + q * c);
    let lat = (2.0 * y.exp().atan() - std::f64::consts::FRAC_PI_2).to_degrees();
    let lon = 5.0 + (k * (p * c - q * s)).to_degrees();
    (lat, lon)
}

/// Grid components (u, v) of the known field at (p, q), with α = `theta`.
fn grid_uv(theta: f64, p: f64, q: f64) -> (f64, f64) {
    let (s, c) = theta.sin_cos();
    let (e, n) = (known_east(p), known_north(q));
    (e * c + n * s, -e * s + n * c)
}

/// `f(p, q)` over `nt` repeats of an `ny` × `nx` grid, row-major, where
/// index (j, i) sits at position (p, q) = (i + dp, j + dq).
fn grid<T>(nt: usize, ny: usize, nx: usize, dp: f64, dq: f64, f: impl Fn(f64, f64) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(nt * ny * nx);
    for _ in 0..nt {
        for j in 0..ny {
            for i in 0..nx {
                out.push(f(i as f64 + dp, j as f64 + dq));
            }
        }
    }
    out
}

fn put<T: NcTypeDescriptor + Copy>(
    f: &mut FileMut,
    name: &str,
    dims: &[&str],
    attrs: &[(&str, &str)],
    fill: Option<T>,
    data: &[T],
) {
    let mut v = f.add_variable::<T>(name, dims).unwrap();
    if let Some(fill) = fill {
        v.set_fill_value(fill).unwrap();
    }
    for (k, val) in attrs {
        v.put_attribute(k, *val).unwrap();
    }
    v.put_values(data, ..).unwrap();
}

/// Latitude and longitude variables (f64) for points at offset (dp, dq).
#[allow(clippy::too_many_arguments)]
fn put_lat_lon(
    f: &mut FileMut,
    names: [&str; 2],
    dims: &[&str],
    shape: [usize; 3],
    theta: f64,
    (dp, dq): (f64, f64),
    attrs: [&[(&str, &str)]; 2],
) {
    let [nt, ny, nx] = shape;
    let lat = grid(nt, ny, nx, dp, dq, |p, q| lat_lon(theta, p, q).0);
    let lon = grid(nt, ny, nx, dp, dq, |p, q| lat_lon(theta, p, q).1);
    put(f, names[0], dims, attrs[0], None, &lat);
    put(f, names[1], dims, attrs[1], None, &lon);
}

const STD_LAT: &[(&str, &str)] = &[("standard_name", "latitude")];
const STD_LON: &[(&str, &str)] = &[("standard_name", "longitude")];
const DEG_LAT: &[(&str, &str)] = &[("units", "degree_north")];
const DEG_LON: &[(&str, &str)] = &[("units", "degree_east")];

/// NEMO-like `nemo.nc` (θ = 30°): `time_counter` = 2 (its variable in
/// "seconds since 1900-01-01 00:00:00"), `y` = 20, `x` = 30.
/// T points `nav_lat`/`nav_lon`, U points `nav_lat_u`/`nav_lon_u` at
/// (j, i+½), V points `nav_lat_v`/`nav_lon_v` at (j+½, i), all with
/// latitude/longitude standard names. `thetao`, `uo`, `vo` (units `m/s`, no
/// standard names) on `(time_counter, y, x)` with `_FillValue` 1e20;
/// `uo[:, 5, 10]` is fill.
pub fn write_nemo_like(dir: &Path) -> PathBuf {
    let theta = 30f64.to_radians();
    let (nt, ny, nx) = (2, 20, 30);
    let path = dir.join("nemo.nc");
    let mut f = netcdf::create(&path).unwrap();
    f.add_dimension("time_counter", nt).unwrap();
    f.add_dimension("y", ny).unwrap();
    f.add_dimension("x", nx).unwrap();
    let tc = [("units", "seconds since 1900-01-01 00:00:00"), ("axis", "T")];
    put(&mut f, "time_counter", &["time_counter"], &tc, None, &[0f64, 86_400.0]);
    let yx = ["y", "x"];
    let shape = [1, ny, nx];
    put_lat_lon(&mut f, ["nav_lat", "nav_lon"], &yx, shape, theta, (0.0, 0.0), [STD_LAT, STD_LON]);
    put_lat_lon(&mut f, ["nav_lat_u", "nav_lon_u"], &yx, shape, theta, (0.5, 0.0), [STD_LAT, STD_LON]);
    put_lat_lon(&mut f, ["nav_lat_v", "nav_lon_v"], &yx, shape, theta, (0.0, 0.5), [STD_LAT, STD_LON]);

    let dims = ["time_counter", "y", "x"];
    let fill = Some(1e20f32);
    let thetao = grid(nt, ny, nx, 0.0, 0.0, |p, q| (10.0 + 0.01 * (p + q)) as f32);
    put(&mut f, "thetao", &dims, &[("coordinates", "nav_lat nav_lon")], fill, &thetao);
    let mut uo = grid(nt, ny, nx, 0.5, 0.0, |p, q| grid_uv(theta, p, q).0 as f32);
    for t in 0..nt {
        uo[(t * ny + 5) * nx + 10] = 1e20;
    }
    let uo_attrs = [("units", "m/s"), ("coordinates", "nav_lat_u nav_lon_u")];
    put(&mut f, "uo", &dims, &uo_attrs, fill, &uo);
    let vo = grid(nt, ny, nx, 0.0, 0.5, |p, q| grid_uv(theta, p, q).1 as f32);
    let vo_attrs = [("units", "m/s"), ("coordinates", "nav_lat_v nav_lon_v")];
    put(&mut f, "vo", &dims, &vo_attrs, fill, &vo);
    path
}

/// ROMS-like `roms.nc` (θ = 0.5 rad): `ocean_time` = 2, rho points
/// `(eta_rho, xi_rho)` = 12 × 16, U points `(eta_u, xi_u)` = 12 × 15 at
/// (j, i+½), V points `(eta_v, xi_v)` = 11 × 16 at (j+½, i), with
/// `lat_*`/`lon_*` for each (units degree_north/east). `angle` = 0.5 on
/// rho points, `temp` on rho points, `u`/`v` (units `meter second-1`).
pub fn write_roms_like(dir: &Path) -> PathBuf {
    let theta = 0.5;
    let nt = 2;
    let path = dir.join("roms.nc");
    let mut f = netcdf::create(&path).unwrap();
    f.add_dimension("ocean_time", nt).unwrap();
    for (name, len) in
        [("eta_rho", 12), ("xi_rho", 16), ("eta_u", 12), ("xi_u", 15), ("eta_v", 11), ("xi_v", 16)]
    {
        f.add_dimension(name, len).unwrap();
    }
    let ll = [DEG_LAT, DEG_LON];
    put_lat_lon(&mut f, ["lat_rho", "lon_rho"], &["eta_rho", "xi_rho"], [1, 12, 16], theta, (0.0, 0.0), ll);
    put_lat_lon(&mut f, ["lat_u", "lon_u"], &["eta_u", "xi_u"], [1, 12, 15], theta, (0.5, 0.0), ll);
    put_lat_lon(&mut f, ["lat_v", "lon_v"], &["eta_v", "xi_v"], [1, 11, 16], theta, (0.0, 0.5), ll);

    put(&mut f, "angle", &["eta_rho", "xi_rho"], &[("units", "radians")], None, &[0.5f64; 12 * 16]);
    let temp = grid(nt, 12, 16, 0.0, 0.0, |p, q| (15.0 + 0.01 * (p - q)) as f32);
    let rho = ["ocean_time", "eta_rho", "xi_rho"];
    put(&mut f, "temp", &rho, &[("coordinates", "lon_rho lat_rho")], None, &temp);
    let units = ("units", "meter second-1");
    let u = grid(nt, 12, 15, 0.5, 0.0, |p, q| grid_uv(theta, p, q).0 as f32);
    let u_attrs = [units, ("coordinates", "lon_u lat_u")];
    put(&mut f, "u", &["ocean_time", "eta_u", "xi_u"], &u_attrs, None, &u);
    let v = grid(nt, 11, 16, 0.0, 0.5, |p, q| grid_uv(theta, p, q).1 as f32);
    let v_attrs = [units, ("coordinates", "lon_v lat_v")];
    put(&mut f, "v", &["ocean_time", "eta_v", "xi_v"], &v_attrs, None, &v);
    path
}

/// WRF-like `wrf.nc` (θ = −0.2 rad): `Time` = 2 with no `Time` variable but
/// `XTIME(Time)` = [0, 60] ("minutes since 2000-01-01 00:00:00"), mass points
/// `(south_north, west_east)` = 10 × 14, `west_east_stag` = 15,
/// `south_north_stag` = 11. `XLAT`/`XLONG` (mass), `XLAT_U`/`XLONG_U` at
/// (j, i−½) and `XLAT_V`/`XLONG_V` at (j−½, i), all `(Time, ..)` with
/// degree units and the same plane at both times. `COSALPHA`, `SINALPHA`
/// and `T2` on mass points; `U` on `(Time, south_north, west_east_stag)`,
/// `V` on `(Time, south_north_stag, west_east)` (units `m s-1`).
pub fn write_wrf_like(dir: &Path) -> PathBuf {
    let theta: f64 = -0.2;
    let (nt, ny, nx) = (2, 10, 14);
    let path = dir.join("wrf.nc");
    let mut f = netcdf::create(&path).unwrap();
    f.add_dimension("Time", nt).unwrap();
    f.add_dimension("south_north", ny).unwrap();
    f.add_dimension("west_east", nx).unwrap();
    f.add_dimension("south_north_stag", ny + 1).unwrap();
    f.add_dimension("west_east_stag", nx + 1).unwrap();
    let mass = ["Time", "south_north", "west_east"];
    let u_dims = ["Time", "south_north", "west_east_stag"];
    let v_dims = ["Time", "south_north_stag", "west_east"];
    let xtime = [("units", "minutes since 2000-01-01 00:00:00"), ("description", "minutes since simulation start")];
    put(&mut f, "XTIME", &["Time"], &xtime, None, &[0f32, 60.0]);
    let ll = [DEG_LAT, DEG_LON];
    put_lat_lon(&mut f, ["XLAT", "XLONG"], &mass, [nt, ny, nx], theta, (0.0, 0.0), ll);
    put_lat_lon(&mut f, ["XLAT_U", "XLONG_U"], &u_dims, [nt, ny, nx + 1], theta, (-0.5, 0.0), ll);
    put_lat_lon(&mut f, ["XLAT_V", "XLONG_V"], &v_dims, [nt, ny + 1, nx], theta, (0.0, -0.5), ll);

    let n = nt * ny * nx;
    put(&mut f, "COSALPHA", &mass, &[], None, &vec![theta.cos() as f32; n]);
    put(&mut f, "SINALPHA", &mass, &[], None, &vec![theta.sin() as f32; n]);
    let t2 = grid(nt, ny, nx, 0.0, 0.0, |p, q| (285.0 + 0.1 * (p + q)) as f32);
    put(&mut f, "T2", &mass, &[("units", "K"), ("coordinates", "XLONG XLAT")], None, &t2);
    let units = ("units", "m s-1");
    let u = grid(nt, ny, nx + 1, -0.5, 0.0, |p, q| grid_uv(theta, p, q).0 as f32);
    put(&mut f, "U", &u_dims, &[units, ("coordinates", "XLONG_U XLAT_U")], None, &u);
    let v = grid(nt, ny + 1, nx, 0.0, -0.5, |p, q| grid_uv(theta, p, q).1 as f32);
    put(&mut f, "V", &v_dims, &[units, ("coordinates", "XLONG_V XLAT_V")], None, &v);
    path
}

/// `noclue_flat.nc` (θ = 0.005 rad) or `noclue_turned.nc` (θ = 20°):
/// `y` = 15, `x` = 20, variables `h, lat, lon, u, v` in that order. 2-D
/// `lat`/`lon` without attributes; `h`, `u`, `v` carry
/// `coordinates "lon lat"` and nothing else (no units, no standard names).
pub fn write_noclue(dir: &Path, turned: bool) -> PathBuf {
    if turned {
        write_noclue_as(dir, "noclue_turned.nc", 20f64.to_radians(), ["u", "v"])
    } else {
        write_noclue_as(dir, "noclue_flat.nc", 0.005, ["u", "v"])
    }
}

/// `noclue_renamed.nc`: `noclue_flat.nc` with `u`/`v` named `aa`/`bb`, so
/// nothing is detected and the pair must be made by hand.
pub fn write_noclue_renamed(dir: &Path) -> PathBuf {
    write_noclue_as(dir, "noclue_renamed.nc", 0.005, ["aa", "bb"])
}

fn write_noclue_as(dir: &Path, file: &str, theta: f64, names: [&str; 2]) -> PathBuf {
    let (ny, nx) = (15, 20);
    let path = dir.join(file);
    let mut f = netcdf::create(&path).unwrap();
    f.add_dimension("y", ny).unwrap();
    f.add_dimension("x", nx).unwrap();
    let yx = ["y", "x"];
    let coords = &[("coordinates", "lon lat")];
    let h = grid(1, ny, nx, 0.0, 0.0, |p, q| (100.0 + p + 2.0 * q) as f32);
    put(&mut f, "h", &yx, coords, None, &h);
    put_lat_lon(&mut f, ["lat", "lon"], &yx, [1, ny, nx], theta, (0.0, 0.0), [&[], &[]]);
    let u = grid(1, ny, nx, 0.0, 0.0, |p, q| grid_uv(theta, p, q).0 as f32);
    put(&mut f, names[0], &yx, coords, None, &u);
    let v = grid(1, ny, nx, 0.0, 0.0, |p, q| grid_uv(theta, p, q).1 as f32);
    put(&mut f, names[1], &yx, coords, None, &v);
    path
}

#[cfg(test)]
mod tests {
    /// Writes the vector fixtures to `C:/t/fixtures` for checking by hand:
    /// `cargo test -p torcdf-core write_vector_fixtures_to -- --ignored`.
    #[test]
    #[ignore = "writes files for the manual checklist"]
    fn write_vector_fixtures_to() {
        let dir = std::path::Path::new("C:/t/fixtures");
        std::fs::create_dir_all(dir).unwrap();
        for path in [
            super::write_nemo_like(dir),
            super::write_roms_like(dir),
            super::write_wrf_like(dir),
            super::write_noclue(dir, false),
            super::write_noclue(dir, true),
        ] {
            assert!(path.is_file(), "{} not written", path.display());
            println!("wrote {}", path.display());
        }
    }
}
