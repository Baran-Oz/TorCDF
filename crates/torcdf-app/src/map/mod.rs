pub mod projection;
pub mod camera;
pub mod basemap;
pub mod graticule;
pub mod locate;
pub mod mesh;
pub mod tiles;

/// Longitudes beyond this are fill values, not coordinates.
const MAX_ABS_LON: f64 = 720.0;

/// A usable curvilinear cell coordinate: fill values (1e20 and friends)
/// and NaN count as missing.
pub(crate) fn valid_lonlat(lon: f64, lat: f64) -> bool {
    lon.is_finite() && lat.is_finite() && lat.abs() <= 90.0 && lon.abs() <= MAX_ABS_LON
}
