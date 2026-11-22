//! Map projections on the unit sphere. World units are radians for the
//! cylindrical projections and a stereographic radius (equator at 2) for the
//! polar ones. The WGSL in `projection.wgsl` mirrors these formulas.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

/// Mercator is cut off at the latitude where y = pi (a square world).
pub const MERCATOR_LIMIT_DEG: f64 = 85.051_129;
/// Mercator inverse accepts |y| up to pi plus a tolerance, so the limit
/// latitude (rounded up, y a hair above pi) round-trips.
const MERCATOR_Y_MAX: f64 = PI + 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Projection {
    #[default]
    Equirectangular,
    Mercator,
    NorthPolarStereo,
    SouthPolarStereo,
}

/// Wrap a longitude in radians to [-pi, pi).
fn wrap_lon(lon: f64) -> f64 {
    let w = (lon + PI).rem_euclid(TAU) - PI;
    // rem_euclid can round up to exactly TAU for tiny negative inputs.
    if w >= PI {
        w - TAU
    } else {
        w
    }
}

impl Projection {
    pub fn all() -> &'static [Projection] {
        &[
            Projection::Equirectangular,
            Projection::Mercator,
            Projection::NorthPolarStereo,
            Projection::SouthPolarStereo,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Projection::Equirectangular => "Equirectangular",
            Projection::Mercator => "Mercator",
            Projection::NorthPolarStereo => "North polar stereographic",
            Projection::SouthPolarStereo => "South polar stereographic",
        }
    }

    /// Must match `projection.wgsl`.
    pub fn shader_id(&self) -> u32 {
        match self {
            Projection::Equirectangular => 0,
            Projection::Mercator => 1,
            Projection::NorthPolarStereo => 2,
            Projection::SouthPolarStereo => 3,
        }
    }

    /// The world repeats every 2*pi in x.
    pub fn wraps_x(&self) -> bool {
        matches!(self, Projection::Equirectangular | Projection::Mercator)
    }

    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> Option<[f64; 2]> {
        if !lon_deg.is_finite() || !lat_deg.is_finite() || lat_deg.abs() > 90.0 {
            return None;
        }
        let lam = wrap_lon(lon_deg.to_radians());
        let phi = lat_deg.to_radians();
        match self {
            Projection::Equirectangular => Some([lam, phi]),
            Projection::Mercator => {
                if lat_deg.abs() > MERCATOR_LIMIT_DEG {
                    return None;
                }
                Some([lam, (FRAC_PI_4 + phi / 2.0).tan().ln()])
            }
            Projection::NorthPolarStereo => {
                if lat_deg < 0.0 {
                    return None;
                }
                let rho = 2.0 * (FRAC_PI_4 - phi / 2.0).tan();
                Some([rho * lam.sin(), -rho * lam.cos()])
            }
            Projection::SouthPolarStereo => {
                if lat_deg > 0.0 {
                    return None;
                }
                let rho = 2.0 * (FRAC_PI_4 + phi / 2.0).tan();
                Some([rho * lam.sin(), rho * lam.cos()])
            }
        }
    }

    /// Returns (lon, lat) in degrees with lon in [-180, 180).
    pub fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        match self {
            Projection::Equirectangular => {
                if y.abs() > FRAC_PI_2 {
                    return None;
                }
                Some((wrap_lon(x).to_degrees(), y.to_degrees()))
            }
            Projection::Mercator => {
                if y.abs() > MERCATOR_Y_MAX {
                    return None;
                }
                let phi = 2.0 * y.exp().atan() - FRAC_PI_2;
                Some((wrap_lon(x).to_degrees(), phi.to_degrees()))
            }
            Projection::NorthPolarStereo | Projection::SouthPolarStereo => {
                let north = *self == Projection::NorthPolarStereo;
                let rho = x.hypot(y);
                // Tolerance so the equator (rho = 2) round-trips.
                if rho > 2.0 + 1e-9 {
                    return None;
                }
                let c = 2.0 * (rho / 2.0).atan();
                let (lam, phi) = if north {
                    (x.atan2(-y), FRAC_PI_2 - c)
                } else {
                    (x.atan2(y), c - FRAC_PI_2)
                };
                Some((wrap_lon(lam).to_degrees(), phi.to_degrees()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn equirect_is_radians() {
        let p = Projection::Equirectangular.forward(90.0, 45.0).unwrap();
        assert_relative_eq!(p[0], FRAC_PI_2);
        assert_relative_eq!(p[1], FRAC_PI_4);
    }

    #[test]
    fn mercator_y_at_45() {
        assert_relative_eq!(
            Projection::Mercator.forward(0.0, 45.0).unwrap()[1],
            0.881_373_587,
            epsilon = 1e-8
        );
    }

    #[test]
    fn mercator_rejects_beyond_limit() {
        assert!(Projection::Mercator.forward(0.0, 86.0).is_none());
        assert!(Projection::Mercator.forward(0.0, 90.0).is_none());
    }

    #[test]
    fn mercator_inverse_rejects_beyond_square_world() {
        let m = Projection::Mercator;
        let top = m.forward(0.0, MERCATOR_LIMIT_DEG).unwrap()[1];
        assert!(m.inverse(0.0, top).is_some(), "the limit round-trips");
        assert!(m.inverse(0.0, 3.0).is_some());
        assert!(m.inverse(0.0, PI + 0.01).is_none());
        assert!(m.inverse(1.0, -PI - 0.01).is_none());
    }

    #[test]
    fn polar_north_values() {
        let n = Projection::NorthPolarStereo;
        assert_eq!(n.forward(0.0, 90.0).unwrap(), [0.0, 0.0]);
        assert_relative_eq!(n.forward(0.0, 0.0).unwrap()[1], -2.0);
        assert_relative_eq!(n.forward(90.0, 0.0).unwrap()[0], 2.0);
    }

    #[test]
    fn polar_rejects_other_hemisphere() {
        assert!(Projection::NorthPolarStereo.forward(0.0, -1.0).is_none());
        assert!(Projection::SouthPolarStereo.forward(0.0, 1.0).is_none());
        assert!(Projection::NorthPolarStereo.inverse(2.5, 0.0).is_none());
        assert!(Projection::SouthPolarStereo.inverse(0.0, 2.5).is_none());
    }

    #[test]
    fn metadata() {
        let ids: Vec<u32> = Projection::all().iter().map(|p| p.shader_id()).collect();
        assert_eq!(ids, [0, 1, 2, 3]);
        assert_eq!(Projection::NorthPolarStereo.name(), "North polar stereographic");
        assert!(Projection::Mercator.wraps_x() && !Projection::SouthPolarStereo.wraps_x());
    }

    #[test]
    fn round_trip_all() {
        for &p in Projection::all() {
            let (lo, hi) = match p {
                Projection::Equirectangular => (-90, 90),
                Projection::Mercator => (-85, 85),
                Projection::NorthPolarStereo => (0, 90),
                Projection::SouthPolarStereo => (-90, 0),
            };
            let mut lon = -179;
            while lon <= 179 {
                let mut lat: i32 = lo;
                while lat <= hi {
                    let (lon_f, lat_f) = (lon as f64, lat as f64);
                    let xy = p.forward(lon_f, lat_f).unwrap();
                    let (lo2, la2) = p.inverse(xy[0], xy[1]).unwrap();
                    let dlon = (lo2 - lon_f + 180.0).rem_euclid(360.0) - 180.0;
                    // Longitude is undefined at the poles.
                    if lat.abs() != 90 {
                        assert!(dlon.abs() < 1e-9, "{p:?} {lon},{lat}: lon {lo2}");
                    }
                    assert!((la2 - lat_f).abs() < 1e-9, "{p:?} {lon},{lat}: lat {la2}");
                    lat += 5;
                }
                lon += 7;
            }
        }
    }

    #[test]
    fn non_finite_and_out_of_range_rejected() {
        for &p in Projection::all() {
            assert!(p.forward(f64::NAN, 10.0).is_none(), "{p:?}");
            assert!(p.forward(10.0, f64::NAN).is_none(), "{p:?}");
            assert!(p.forward(f64::INFINITY, 10.0).is_none(), "{p:?}");
            assert!(p.forward(10.0, 100.0).is_none(), "{p:?}");
            assert!(p.forward(10.0, -100.0).is_none(), "{p:?}");
            assert!(p.inverse(f64::NAN, 0.5).is_none(), "{p:?}");
            assert!(p.inverse(0.5, f64::NAN).is_none(), "{p:?}");
            assert!(p.inverse(f64::INFINITY, 0.5).is_none(), "{p:?}");
        }
    }

    #[test]
    fn longitude_wraps_into_half_open_range() {
        for lon in [180.0, -180.0, 540.0, -540.0] {
            let xy = Projection::Equirectangular.forward(lon, 0.0).unwrap();
            assert_eq!(xy[0], -PI, "forward lon {lon}");
            let (back, _) = Projection::Equirectangular.inverse(xy[0], 0.0).unwrap();
            assert_eq!(back, -180.0, "inverse lon {lon}");
        }
        // Just below -pi must not round up to +pi.
        let (lon, _) = Projection::Mercator.inverse(-PI - 1e-16, 0.0).unwrap();
        assert!((-180.0..180.0).contains(&lon), "{lon}");
        let (lon, _) = Projection::Mercator.inverse(-1e-17 - PI, 0.0).unwrap();
        assert!((-180.0..180.0).contains(&lon), "{lon}");
    }

    #[test]
    fn cylindrical_inverse_wraps() {
        let (lon, _) = Projection::Equirectangular.inverse(PI + 0.1, 0.0).unwrap();
        assert_relative_eq!(lon, -180.0 + 0.1f64.to_degrees(), epsilon = 1e-9);
    }
}
