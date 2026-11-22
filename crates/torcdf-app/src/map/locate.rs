//! Maps a lon/lat to the grid cell under it (hover readout, inspector) and
//! supplies sample points for camera fitting. Pure CPU and `Send`, so large
//! curvilinear locators can be built on a helper job.

use rayon::prelude::*;

use super::valid_lonlat as valid;
use crate::render::map_regular::{lon_window_start, strictly_monotonic, unwrap_lon};

/// Finds the cell containing or nearest to a lon/lat.
#[derive(Debug, Clone)]
pub enum CellLocator {
    /// Rectilinear grid: `lat` strictly monotonic, `lon` unwrapped (continuous)
    /// and strictly monotonic, both either direction.
    Regular { lat: Vec<f64>, lon: Vec<f64>, lon_start: f64 },
    /// Curvilinear grid with a lon/lat bucket index over its valid cells.
    Curvilinear(Curvi),
}

/// Bucket index of a curvilinear grid (row-major `lat`/`lon`, `ny * nx`).
#[derive(Debug, Clone)]
pub struct Curvi {
    lat: Vec<f64>,
    lon: Vec<f64>,
    ny: usize,
    nx: usize,
    /// Bounding box origin of the valid cells (lon wrapped into [-180, 180)).
    lon0: f64,
    lat0: f64,
    lon_step: f64,
    lat_step: f64,
    /// Buckets per axis (`nb` x `nb`).
    nb: usize,
    /// CSR layout: cells of bucket `b` are `cells[start[b]..start[b + 1]]`.
    start: Vec<u32>,
    cells: Vec<u32>,
    /// Great-circle distance (radians) from each cell to its nearest valid
    /// 4-neighbour (up/down/left/right); 0 for isolated or invalid cells.
    nn: Vec<f32>,
    /// Search radius (radians): twice the 99.9th-percentile `nn`. A lookup
    /// finds the true nearest cell whenever it lies within this radius, and
    /// nothing otherwise. A percentile, not the maximum, so a few cells with
    /// junk coordinates cannot widen every search to the whole grid.
    reach: f64,
}

/// [`Curvi::reach`] of the neighbour distances `nn`.
fn reach_of(nn: &[f32]) -> f64 {
    let mut d: Vec<f32> = nn.iter().copied().filter(|&v| v > 0.0).collect();
    if d.is_empty() {
        return 0.0;
    }
    let k = ((d.len() - 1) as f64 * 0.999).round() as usize;
    let (_, v, _) = d.select_nth_unstable_by(k, f32::total_cmp);
    2.0 * *v as f64
}

/// Longitude wrapped into [-180, 180).
fn wrap180(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

fn unit(lon: f64, lat: f64) -> [f64; 3] {
    let (lo, la) = (lon.to_radians(), lat.to_radians());
    [la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()]
}

/// Great-circle angle (radians) between two unit vectors.
fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let chord = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    2.0 * (chord / 2.0).min(1.0).asin()
}

/// Index of the entry of strictly monotonic `c` nearest `x`; `None` when `x`
/// lies beyond an end by more than half that end cell's own spacing (a single
/// entry counts as a 1 degree cell).
fn nearest_axis(c: &[f64], x: f64) -> Option<usize> {
    let n = c.len();
    if n == 0 || x.is_nan() {
        return None;
    }
    let asc = n < 2 || c[n - 1] > c[0];
    let p = c.partition_point(|&v| if asc { v < x } else { v > x });
    let half = |a: usize, b: usize| if n < 2 { 0.5 } else { 0.5 * (c[a] - c[b]).abs() };
    if p == 0 {
        return ((x - c[0]).abs() <= half(1.min(n - 1), 0)).then_some(0);
    }
    if p == n {
        return ((x - c[n - 1]).abs() <= half(n - 1, n.saturating_sub(2))).then_some(n - 1);
    }
    Some(if (x - c[p - 1]).abs() <= (c[p] - x).abs() { p - 1 } else { p })
}

/// `k`-th of `m` indices spread evenly over `0..len`.
fn spread(k: usize, m: usize, len: usize) -> usize {
    if m < 2 { len / 2 } else { k * (len - 1) / (m - 1) }
}

impl CellLocator {
    /// `None` unless `lat` and `lon` are non-empty, finite and strictly
    /// monotonic (`lon` after unwrapping the 0/360 or dateline seam).
    pub fn regular(lat: Vec<f64>, lon: Vec<f64>) -> Option<CellLocator> {
        if lat.is_empty() || lon.is_empty() || !strictly_monotonic(&lat) {
            return None;
        }
        if lon.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let lon = unwrap_lon(&lon);
        if !strictly_monotonic(&lon) {
            return None;
        }
        let lon_start = lon_window_start(&lon);
        Some(CellLocator::Regular { lat, lon, lon_start })
    }

    /// Locator for row-major `lat`/`lon` of shape `[ny, nx]`. Mismatched
    /// lengths or no valid cell give a locator that finds nothing.
    pub fn curvilinear(lat: Vec<f64>, lon: Vec<f64>, ny: usize, nx: usize) -> CellLocator {
        let n = ny
            .checked_mul(nx)
            .filter(|&n| n == lat.len() && n == lon.len() && n < u32::MAX as usize);
        let Some(n) = n else {
            return CellLocator::Curvilinear(Curvi::empty());
        };
        let ok = |i: usize| valid(lon[i], lat[i]);
        // Bounding box of the valid cells.
        let bbox = (0..n)
            .into_par_iter()
            .filter(|&i| ok(i))
            .map(|i| {
                let (lo, la) = (wrap180(lon[i]), lat[i]);
                (lo, lo, la, la)
            })
            .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3)));
        let Some((lo_min, lo_max, la_min, la_max)) = bbox else {
            return CellLocator::Curvilinear(Curvi::empty());
        };
        let nb = ((n as f64).sqrt().round() as usize).clamp(1, 4096);
        let step = |lo: f64, hi: f64| ((hi - lo) / nb as f64).max(1e-9);
        let mut c = Curvi {
            ny,
            nx,
            lon0: lo_min,
            lat0: la_min,
            lon_step: step(lo_min, lo_max),
            lat_step: step(la_min, la_max),
            nb,
            ..Curvi::empty()
        };
        // Bucket of every valid cell (in parallel), then a counting sort.
        let bucket: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| if ok(i) { c.bucket_of(lon[i], lat[i]) as u32 } else { u32::MAX })
            .collect();
        let mut start = vec![0u32; nb * nb + 1];
        for &b in bucket.iter().filter(|&&b| b != u32::MAX) {
            start[b as usize + 1] += 1;
        }
        for b in 0..nb * nb {
            start[b + 1] += start[b];
        }
        let mut fill = start.clone();
        let mut cells = vec![0u32; start[nb * nb] as usize];
        for (i, &b) in bucket.iter().enumerate().filter(|&(_, &b)| b != u32::MAX) {
            cells[fill[b as usize] as usize] = i as u32;
            fill[b as usize] += 1;
        }
        let nn: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !ok(i) {
                    return 0.0;
                }
                let (iy, ix) = (i / nx, i % nx);
                let here = unit(lon[i], lat[i]);
                let mut best = f64::INFINITY;
                let mut look = |j: usize| {
                    if ok(j) {
                        best = best.min(angle(here, unit(lon[j], lat[j])));
                    }
                };
                if iy > 0 {
                    look(i - nx);
                }
                if iy + 1 < ny {
                    look(i + nx);
                }
                if ix > 0 {
                    look(i - 1);
                }
                if ix + 1 < nx {
                    look(i + 1);
                }
                if best.is_finite() { best as f32 } else { 0.0 }
            })
            .collect();
        c.start = start;
        c.cells = cells;
        c.reach = reach_of(&nn);
        c.nn = nn;
        c.lat = lat;
        c.lon = lon;
        CellLocator::Curvilinear(c)
    }

    /// `(iy, ix)` of the cell under (`lon`, `lat`) in degrees, if any.
    pub fn nearest(&self, lon: f64, lat: f64) -> Option<(usize, usize)> {
        if !lon.is_finite() || !lat.is_finite() {
            return None;
        }
        match self {
            CellLocator::Regular { lat: la, lon: lo, lon_start } => {
                let iy = nearest_axis(la, lat)?;
                let q = lon_start + (lon - lon_start).rem_euclid(360.0);
                let ix = nearest_axis(lo, q)?;
                Some((iy, ix))
            }
            CellLocator::Curvilinear(c) => c.nearest(lon, lat),
        }
    }

    /// Up to `n * n` valid cell centres `(lon, lat)` spread evenly over the
    /// grid index space (lon in [-180, 180)), for fitting a camera.
    pub fn sample_lonlat(&self, n: usize) -> Vec<(f64, f64)> {
        let (ny, nx) = self.shape();
        let (my, mx) = (n.min(ny), n.min(nx));
        let mut out = Vec::with_capacity(my * mx);
        for ky in 0..my {
            for kx in 0..mx {
                let (lo, la) = self.centre(spread(ky, my, ny), spread(kx, mx, nx));
                if valid(lo, la) {
                    out.push((wrap180(lo), la));
                }
            }
        }
        out
    }

    /// True for a regular grid whose lon axis covers (nearly) the globe:
    /// `|last - first| + s >= 360 - 1.5 * s` on the unwrapped axis, where
    /// `s = |last - first| / (nx - 1)` is the mean cell spacing.
    pub fn wraps_lon(&self) -> bool {
        match self {
            CellLocator::Regular { lon, .. } if lon.len() >= 2 => {
                let span = (lon[lon.len() - 1] - lon[0]).abs();
                let s = span / (lon.len() - 1) as f64;
                span + s >= 360.0 - 1.5 * s
            }
            _ => false,
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            CellLocator::Regular { lat, lon, .. } => (lat.len(), lon.len()),
            CellLocator::Curvilinear(c) => (c.ny, c.nx),
        }
    }

    fn centre(&self, iy: usize, ix: usize) -> (f64, f64) {
        match self {
            CellLocator::Regular { lat, lon, .. } => (lon[ix], lat[iy]),
            CellLocator::Curvilinear(c) => (c.lon[iy * c.nx + ix], c.lat[iy * c.nx + ix]),
        }
    }
}

impl Curvi {
    fn empty() -> Curvi {
        Curvi {
            lat: Vec::new(),
            lon: Vec::new(),
            ny: 0,
            nx: 0,
            lon0: 0.0,
            lat0: 0.0,
            lon_step: 1.0,
            lat_step: 1.0,
            nb: 1,
            start: vec![0, 0],
            cells: Vec::new(),
            nn: Vec::new(),
            reach: 0.0,
        }
    }

    fn bucket_of(&self, lon: f64, lat: f64) -> usize {
        self.lat_bucket(lat) * self.nb + self.lon_bucket(wrap180(lon))
    }

    /// Bucket index of `v` on an axis starting at `v0`, clamped to the grid.
    fn ax(&self, v: f64, v0: f64, step: f64) -> usize {
        (((v - v0) / step).floor().max(0.0) as usize).min(self.nb - 1)
    }

    fn lat_bucket(&self, lat: f64) -> usize {
        self.ax(lat, self.lat0, self.lat_step)
    }

    /// `lon` already wrapped into [-180, 180).
    fn lon_bucket(&self, lon: f64) -> usize {
        self.ax(lon, self.lon0, self.lon_step)
    }

    /// Bucket columns (inclusive ranges, at most two) holding every cell
    /// within `r` degrees of (`lon`, `lat`). Buckets are fixed in degrees,
    /// so the lon half-width grows towards the poles: a point within `r` of
    /// the query has `sin(dlon/2) <= sin(r/2) / sqrt(cos(lat) cos(lat'))`,
    /// `lat'` the band edge nearer the pole. A band reaching a pole, or a
    /// half-width of 180 or more, takes all columns. Otherwise the lon
    /// window is split at the dateline, so the last and first columns of a
    /// grid crossing it are searched together.
    fn lon_ranges(&self, lon: f64, lat: f64, r: f64) -> Vec<(usize, usize)> {
        let all = vec![(0, self.nb - 1)];
        let edge = lat.abs() + r;
        if edge >= 90.0 {
            return all;
        }
        let c = (lat.to_radians().cos() * edge.to_radians().cos()).sqrt();
        let s = (0.5 * r.to_radians()).sin() / c;
        if s >= 1.0 {
            return all;
        }
        let half = 2.0 * s.asin().to_degrees();
        if half >= 180.0 {
            return all;
        }
        let lon = wrap180(lon);
        let (a, b) = (lon - half, lon + half);
        let span = |lo: f64, hi: f64| (self.lon_bucket(lo), self.lon_bucket(hi));
        if a < -180.0 {
            vec![span(a + 360.0, 180.0), span(-180.0, b)]
        } else if b >= 180.0 {
            vec![span(a, 180.0), span(-180.0, b - 360.0)]
        } else {
            vec![span(a, b)]
        }
    }

    /// The valid cell nearest (`lon`, `lat`), searched within [`Curvi::reach`]
    /// (see [`Curvi::lon_ranges`]), and accepted only within twice its own
    /// neighbour distance.
    fn nearest(&self, lon: f64, lat: f64) -> Option<(usize, usize)> {
        if self.cells.is_empty() {
            return None;
        }
        let q = unit(lon, lat);
        let r = self.reach.to_degrees();
        let rows = self.lat_bucket(lat - r)..=self.lat_bucket(lat + r);
        let cols = self.lon_ranges(lon, lat, r);
        let mut best: Option<(f64, usize)> = None;
        for y in rows {
            for &(x0, x1) in &cols {
                let b0 = y * self.nb + x0;
                let b1 = y * self.nb + x1;
                // Buckets x0..=x1 of a row are contiguous in the CSR layout.
                for &i in &self.cells[self.start[b0] as usize..self.start[b1 + 1] as usize] {
                    let i = i as usize;
                    let d = angle(q, unit(self.lon[i], self.lat[i]));
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, i));
                    }
                }
            }
        }
        // Beyond `reach` a nearer cell may lie in a bucket not searched.
        let (d, i) = best?;
        (d <= self.reach.min(2.0 * self.nn[i] as f64)).then_some((i / self.nx, i % self.nx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    /// 20 x 30 grid of 0.5 degree cells rotated 30 degrees about (10E, 20N).
    fn rotated() -> (Vec<f64>, Vec<f64>) {
        let (ny, nx) = (20, 30);
        let (s, c) = 30f64.to_radians().sin_cos();
        let (mut lat, mut lon) = (Vec::new(), Vec::new());
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = ((ix as f64 - 15.0) * 0.5, (iy as f64 - 10.0) * 0.5);
                lon.push(10.0 + x * c - y * s);
                lat.push(20.0 + x * s + y * c);
            }
        }
        (lat, lon)
    }

    #[test]
    fn regular_nearest_exact() {
        let l = CellLocator::regular(vec![-10.0, 0.0, 10.0], vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        assert_eq!(l.nearest(21.0, 1.0), Some((1, 2)));
        assert!(!l.wraps_lon());
    }

    #[test]
    fn regular_decreasing_lat() {
        let l = CellLocator::regular(vec![10.0, 0.0, -10.0], vec![0.0, 10.0]).unwrap();
        assert_eq!(l.nearest(0.0, 9.0), Some((0, 0)));
        assert_eq!(l.nearest(0.0, -9.0), Some((2, 0)));
    }

    #[test]
    fn regular_0_360_wrap() {
        let lon: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let l = CellLocator::regular(vec![-10.0, 0.0, 10.0], lon).unwrap();
        assert_eq!(l.nearest(-10.0, 0.0).map(|p| p.1), Some(35));
        assert_eq!(l.nearest(181.0, 0.0), Some((1, 18)));
        assert!(l.wraps_lon());
    }

    #[test]
    fn regular_dateline_and_decreasing_lon() {
        let lon: Vec<f64> = (0..36).map(|i| -180.0 + i as f64 * 10.0).collect();
        let l = CellLocator::regular(vec![0.0, 10.0], lon).unwrap();
        assert_eq!(l.nearest(190.0, 0.0), Some((0, 1)));
        let l = CellLocator::regular(vec![0.0, 10.0], vec![20.0, 10.0, 0.0, 350.0]).unwrap();
        assert_eq!(l.nearest(-9.0, 0.0), Some((0, 3)));
        assert_eq!(l.nearest(19.0, 0.0), Some((0, 0)));
        assert!(!l.wraps_lon());
    }

    #[test]
    fn regular_outside_is_none() {
        let l = CellLocator::regular(vec![40.0, 50.0], vec![10.0, 20.0]).unwrap();
        assert_eq!(l.nearest(30.0, 45.0), None);
        assert_eq!(l.nearest(15.0, 56.0), None);
        assert_eq!(l.nearest(15.0, 54.0), Some((1, 0)));
        assert_eq!(l.nearest(f64::NAN, 45.0), None);
    }

    #[test]
    fn regular_single_cell_axis_and_rejects() {
        let l = CellLocator::regular(vec![5.0], vec![10.0]).unwrap();
        assert_eq!(l.nearest(10.4, 5.4), Some((0, 0)));
        assert_eq!(l.nearest(10.6, 5.0), None);
        assert!(CellLocator::regular(vec![], vec![1.0]).is_none());
        assert!(CellLocator::regular(vec![1.0, 1.0], vec![1.0, 2.0]).is_none());
        assert!(CellLocator::regular(vec![1.0, 2.0], vec![1.0, f64::NAN]).is_none());
        assert!(CellLocator::regular(vec![0.0, 2.0, 1.0], vec![1.0, 2.0]).is_none());
    }

    #[test]
    fn curvilinear_nearest() {
        let (lat, lon) = rotated();
        let l = CellLocator::curvilinear(lat.clone(), lon.clone(), 20, 30);
        assert!(!l.wraps_lon());
        for iy in 0..20 {
            for ix in 0..30 {
                let i = iy * 30 + ix;
                assert_eq!(l.nearest(lon[i], lat[i]), Some((iy, ix)));
            }
        }
    }

    #[test]
    fn locator_skips_nan_coords() {
        let (mut lat, mut lon) = rotated();
        let (old_lon, old_lat) = (lon[5 * 30 + 5], lat[5 * 30 + 5]);
        lat[5 * 30 + 5] = f64::NAN;
        lon[5 * 30 + 5] = f64::NAN;
        let l = CellLocator::curvilinear(lat, lon, 20, 30);
        assert_ne!(l.nearest(old_lon, old_lat), Some((5, 5)));
    }

    #[test]
    fn curvilinear_far_is_none() {
        let (lat, lon) = rotated();
        let l = CellLocator::curvilinear(lat, lon, 20, 30);
        assert_eq!(l.nearest(10.0 + 50.0, 20.0), None);
        assert_eq!(l.nearest(10.0, 20.0 + 50.0), None);
    }

    #[test]
    fn curvilinear_degenerate_inputs() {
        assert_send::<CellLocator>();
        let l = CellLocator::curvilinear(vec![], vec![], 0, 0);
        assert_eq!(l.nearest(0.0, 0.0), None);
        assert!(l.sample_lonlat(4).is_empty());
        let l = CellLocator::curvilinear(vec![f64::NAN; 6], vec![f64::NAN; 6], 2, 3);
        assert_eq!(l.nearest(0.0, 0.0), None);
        assert!(l.sample_lonlat(4).is_empty());
        let l = CellLocator::curvilinear(vec![0.0; 5], vec![0.0; 6], 2, 3);
        assert_eq!(l.nearest(0.0, 0.0), None);
        // Fill values are missing, not coordinates.
        let l = CellLocator::curvilinear(vec![1e20, 0.0], vec![1e20, 0.0], 1, 2);
        assert_eq!(l.nearest(0.0, 0.0), Some((0, 1)));
        assert_eq!(l.sample_lonlat(4), vec![(0.0, 0.0)]);
        // A fill longitude with a plausible latitude is missing too (as in
        // the mesh).
        let l = CellLocator::curvilinear(vec![0.0, 0.0], vec![1e20, 0.0], 1, 2);
        assert_eq!(l.sample_lonlat(4), vec![(0.0, 0.0)]);
    }

    #[test]
    fn curvilinear_crossing_dateline() {
        // 4 x 40 cells of 1 degree, lon 160..199 (180 and beyond wrap to
        // -180..-161): 13 x 13 buckets, so lon 179 and 180 sit in the first
        // and last bucket columns.
        let (ny, nx) = (4, 40);
        let (mut lat, mut lon) = (Vec::new(), Vec::new());
        for iy in 0..ny {
            for ix in 0..nx {
                lat.push(iy as f64);
                lon.push(160.0 + ix as f64);
            }
        }
        let l = CellLocator::curvilinear(lat.clone(), lon.clone(), ny, nx);
        let CellLocator::Curvilinear(c) = &l else { unreachable!() };
        assert!(c.nb > 3, "nb = {}", c.nb);
        for i in 0..ny * nx {
            assert_eq!(l.nearest(lon[i], lat[i]), Some((i / nx, i % nx)));
        }
        assert_eq!(l.nearest(179.6, 1.0), Some((1, 20)));
        assert_eq!(l.nearest(-179.6, 1.0), Some((1, 20)));
        assert_eq!(l.nearest(179.4, 1.0), Some((1, 19)));
        assert_eq!(l.nearest(179.4 - 360.0, 2.0), Some((2, 19)));
    }

    /// Spherical north polar stereographic grid: `n` x `n` cells of `km`
    /// kilometres centred on the pole (an even `n` puts the pole between
    /// four cells), as sea-ice products ship it.
    fn polar_stereo(n: usize, km: f64) -> (Vec<f64>, Vec<f64>) {
        let r = 6371.0;
        let (mut lat, mut lon) = (Vec::new(), Vec::new());
        for iy in 0..n {
            for ix in 0..n {
                let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * km;
                let y = (iy as f64 - (n as f64 - 1.0) / 2.0) * km;
                let c = 2.0 * (x.hypot(y) / (2.0 * r)).atan();
                lat.push(90.0 - c.to_degrees());
                lon.push(x.atan2(-y).to_degrees());
            }
        }
        (lat, lon)
    }

    /// Distance (radians) from (`lon`, `lat`) to the nearest valid cell, by
    /// brute force.
    fn brute(c: &Curvi, lon: f64, lat: f64) -> f64 {
        let q = unit(lon, lat);
        (0..c.lat.len())
            .filter(|&i| valid(c.lon[i], c.lat[i]))
            .map(|i| angle(q, unit(c.lon[i], c.lat[i])))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn curvilinear_polar_stereo_near_pole() {
        let n = 100;
        let (lat, lon) = polar_stereo(n, 25.0);
        assert!(lat.iter().filter(|&&la| la > 85.0).count() > 100);
        let l = CellLocator::curvilinear(lat.clone(), lon.clone(), n, n);
        for i in 0..n * n {
            assert_eq!(l.nearest(lon[i], lat[i]), Some((i / n, i % n)), "cell at lat {}", lat[i]);
        }
        // Points between cells, all around the pole, find a nearest cell (by
        // distance: at the pole itself four cells tie).
        let CellLocator::Curvilinear(c) = &l else { unreachable!() };
        for la in [84.0, 87.5, 89.0, 89.7, 89.95, 90.0] {
            for k in 0..72 {
                let lo = -180.0 + 5.0 * k as f64 + 0.3;
                let (iy, ix) = l.nearest(lo, la).unwrap_or_else(|| panic!("({lo}, {la}) found nothing"));
                let got = angle(unit(lo, la), unit(lon[iy * n + ix], lat[iy * n + ix]));
                assert!((got - brute(c, lo, la)).abs() < 1e-12, "({lo}, {la}) -> ({iy}, {ix})");
            }
        }
    }

    #[test]
    fn sample_lonlat_bounded() {
        let (lat, lon) = rotated();
        let l = CellLocator::curvilinear(lat.clone(), lon.clone(), 20, 30);
        let s = l.sample_lonlat(8);
        assert!(!s.is_empty() && s.len() <= 64);
        for (lo, la) in s {
            assert!(
                lon.iter().zip(&lat).any(|(&a, &b)| (a - lo).abs() < 1e-12 && (b - la).abs() < 1e-12),
                "({lo}, {la}) is not a cell centre"
            );
        }
        let r = CellLocator::regular(vec![0.0, 1.0], vec![0.0, 1.0, 2.0]).unwrap();
        assert!(r.sample_lonlat(8).len() <= 6);
        assert!(r.sample_lonlat(0).is_empty());
    }
}
