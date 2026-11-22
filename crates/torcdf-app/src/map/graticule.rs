//! Lat/lon grid lines in geographic degrees (projected later by the map).

use torcdf_core::basemap::Polyline;

const STEPS: [f64; 8] = [90.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25];
const MIN_SPACING_PX: f64 = 80.0;
/// Most points one generated set holds; densification coarsens to stay below.
pub const MAX_POINTS: usize = 50_000;
/// Parallels stop short of the poles.
const MAX_PARALLEL: f64 = 80.0;

/// Smallest step whose on-screen spacing is at least 80 px, given the
/// current scale in degrees per 100 px. Falls back to the coarsest step for
/// a non-positive or non-finite scale, and to the finest when zoomed further.
pub fn graticule_step(deg_per_100px: f64) -> f64 {
    if !(deg_per_100px.is_finite() && deg_per_100px > 0.0) {
        return STEPS[0];
    }
    let px_per_deg = 100.0 / deg_per_100px;
    STEPS
        .iter()
        .rev()
        .copied()
        .find(|s| s * px_per_deg >= MIN_SPACING_PX)
        .unwrap_or(STEPS[0])
}

/// The whole-globe set: [`graticule_window`] over -180..=180, -90..=90.
#[cfg(test)]
pub fn graticule_lines(step_deg: f64) -> Vec<Polyline> {
    graticule_window(step_deg, [-180.0, 180.0], [-90.0, 90.0])
}

/// Meridians at multiples of `step_deg` inside the `lon` window (spanning the
/// `lat` window), and parallels inside the `lat` window (within -80..=80,
/// spanning the `lon` window). `lon` may run past ±180 (e.g. 170..200); a
/// window wider than 360 degrees is cut to 360 about its centre. Lines are
/// densified every min(step, 1) degree, coarser when the set would exceed
/// [`MAX_POINTS`]. A non-positive or non-finite step, or a non-finite
/// window, gives no lines.
pub fn graticule_window(step_deg: f64, lon: [f64; 2], lat: [f64; 2]) -> Vec<Polyline> {
    let finite = lon.iter().chain(&lat).all(|v| v.is_finite());
    if !(step_deg.is_finite() && step_deg > 0.0) || !finite {
        return Vec::new();
    }
    let (mut lo0, mut lo1) = (lon[0].min(lon[1]), lon[0].max(lon[1]));
    if lo1 - lo0 > 360.0 {
        let c = (lo0 + lo1) / 2.0;
        (lo0, lo1) = (c - 180.0, c + 180.0);
    }
    let (la0, la1) = (lat[0].min(lat[1]).max(-90.0), lat[0].max(lat[1]).min(90.0));
    if la0 > la1 {
        return Vec::new();
    }
    let mer = multiples(step_deg, lo0, lo1);
    let par = multiples(step_deg, la0.max(-MAX_PARALLEL), la1.min(MAX_PARALLEL));
    let lines = mer.len() + par.len();
    if lines == 0 || lines > MAX_POINTS / 4 {
        return Vec::new();
    }
    // Each line holds about span / dens + 2 points.
    let length = mer.len() as f64 * (la1 - la0) + par.len() as f64 * (lo1 - lo0);
    let budget = (MAX_POINTS - 2 * lines) as f64;
    let dens = step_deg.min(1.0).max(length / budget);

    let mut out = Vec::with_capacity(lines);
    for &lon in &mer {
        out.push(densify(la0, la1, dens, |t| [lon as f32, t as f32]));
    }
    for &lat in &par {
        out.push(densify(lo0, lo1, dens, |t| [t as f32, lat as f32]));
    }
    out
}

/// Multiples of `step` in [a, b]; empty if there would be more than
/// `MAX_POINTS` of them.
fn multiples(step: f64, a: f64, b: f64) -> Vec<f64> {
    let (k0, k1) = ((a / step).ceil(), (b / step).floor());
    if k1 < k0 || k1 - k0 >= MAX_POINTS as f64 {
        return Vec::new();
    }
    (k0 as i64..=k1 as i64).map(|k| k as f64 * step).collect()
}

fn densify(a: f64, b: f64, dens: f64, pt: impl Fn(f64) -> [f32; 2]) -> Polyline {
    let n = ((b - a) / dens).round().max(1.0) as usize;
    (0..=n).map(|k| pt(a + (b - a) * k as f64 / n as f64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(g: &[Polyline]) -> usize {
        g.iter().map(Vec::len).sum()
    }

    #[test]
    fn graticule_step_spacing() {
        assert_eq!(graticule_step(100.0), 90.0);
        assert_eq!(graticule_step(1.0), 1.0);
        assert_eq!(graticule_step(0.1), 0.25);
    }

    #[test]
    fn graticule_lines_cover_globe() {
        let g = graticule_lines(30.0);
        assert_eq!(g.len(), 18);
    }

    #[test]
    fn bad_steps_are_harmless() {
        for s in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(graticule_step(s), 90.0, "{s}");
            assert!(graticule_lines(s).is_empty(), "{s}");
        }
        assert!(graticule_window(1.0, [f64::NAN, 10.0], [0.0, 10.0]).is_empty());
        // A step so fine the lines alone would blow the budget.
        assert!(graticule_lines(1e-6).is_empty());
    }

    #[test]
    fn sets_stay_bounded() {
        // The finest step over the whole globe used to be ~2 M points.
        assert!(points(&graticule_lines(0.25)) <= MAX_POINTS);
        for step in [90.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25] {
            assert!(points(&graticule_lines(step)) <= MAX_POINTS, "{step}");
        }
    }

    #[test]
    fn window_across_the_dateline() {
        let g = graticule_window(1.0, [170.0, 200.0], [0.0, 10.0]);
        // 31 meridians (170..=200) and 11 parallels (0..=10).
        assert_eq!(g.len(), 42);
        assert_eq!(g[0][0], [170.0, 0.0]);
        assert_eq!(g[30][0], [200.0, 0.0]);
        // Parallels span the window, densified every degree.
        assert_eq!(g[31].len(), 31);
        assert_eq!((g[31][0], g[31][30]), ([170.0, 0.0], [200.0, 0.0]));
    }
}
