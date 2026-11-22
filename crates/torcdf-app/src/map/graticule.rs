//! Lat/lon grid lines in geographic degrees (projected later by the map).

use torcdf_core::basemap::Polyline;

const STEPS: [f64; 8] = [90.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25];
const MIN_SPACING_PX: f64 = 80.0;

/// Smallest step whose on-screen spacing is at least 80 px, given the
/// current scale in degrees per 100 px. Falls back to the finest step.
pub fn graticule_step(deg_per_100px: f64) -> f64 {
    let px_per_deg = 100.0 / deg_per_100px;
    STEPS
        .iter()
        .rev()
        .copied()
        .find(|s| s * px_per_deg >= MIN_SPACING_PX)
        .unwrap_or(STEPS[0])
}

/// Meridians over -180..=180 (full latitude span) and parallels over
/// -80..=80, at multiples of `step_deg`, densified every min(step, 1) degree.
pub fn graticule_lines(step_deg: f64) -> Vec<Polyline> {
    let dens = step_deg.min(1.0);
    let mut out = Vec::new();

    let n_mer = (180.0 / step_deg).floor() as i32;
    for i in -n_mer..=n_mer {
        let lon = (i as f64 * step_deg) as f32;
        out.push(densify(-90.0, 90.0, dens, |t| [lon, t as f32]));
    }
    let n_par = (80.0 / step_deg).floor() as i32;
    for i in -n_par..=n_par {
        let lat = (i as f64 * step_deg) as f32;
        out.push(densify(-180.0, 180.0, dens, |t| [t as f32, lat]));
    }
    out
}

fn densify(a: f64, b: f64, dens: f64, pt: impl Fn(f64) -> [f32; 2]) -> Polyline {
    let n = ((b - a) / dens).round().max(1.0) as usize;
    (0..=n).map(|k| pt(a + (b - a) * k as f64 / n as f64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
