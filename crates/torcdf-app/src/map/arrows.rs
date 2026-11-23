//! Where vector arrows go and which way they point (pure logic; the GPU
//! drawing is `render::arrows`, the background job `view::arrow_state`).
//!
//! Two world spaces ([`ArrowSpace`]):
//! - Map: positions are projected world units and directions come from the
//!   projection's tangents ([`screen_dir`]).
//! - Image: positions are cell centres `(ix + 0.5, iy + 0.5)` in the image
//!   view's cell units (x = column, y = row index), which
//!   `render::arrows::image_affine` maps to the screen. No latitudes are
//!   involved, so every row gets an arrow. North is drawn up on screen, as
//!   the image view puts north up: +row when rows grow upwards (`flip_y`),
//!   -row otherwise. East is divided by the cell aspect (screen width over
//!   height of a cell) so the on-screen angle is that of `(east, north)`.

use super::projection::Projection;
use super::tiles::TileSpec;
use std::collections::HashSet;
use std::f64::consts::{PI, TAU};
use std::sync::Arc;
use torcdf_core::Slice;

/// Least on-screen distance between neighbouring arrows.
pub const ARROW_SPACING_PT: f64 = 30.0;
/// Hard cap on arrow instances.
pub const MAX_ARROWS: usize = 20_000;
/// Arrows closer to a pole than this (degrees) are skipped.
const POLE_LIMIT_DEG: f64 = 89.9;
/// Step for the numeric map tangents, degrees.
const DELTA_DEG: f64 = 1e-3;

/// Per axis the smallest power of two `k` with `k * cell_pt >= 30` (min 1).
pub fn arrow_step(cell_pt: [f64; 2]) -> [usize; 2] {
    cell_pt.map(|c| {
        let mut k = 1usize;
        // NaN or tiny cells run to the cap.
        while (c.is_nan() || (k as f64) * c < ARROW_SPACING_PT) && k < 1 << 30 {
            k *= 2;
        }
        k
    })
}

/// Multiples of `k` in `[start, start + count)`.
fn multiples(start: usize, count: usize, k: usize) -> usize {
    if count == 0 {
        return 0;
    }
    (start + count - 1) / k + 1 - start.div_ceil(k)
}

/// Cells `(iy, ix)` with `iy % k[0] == 0` and `ix % k[1] == 0` inside the
/// windows (start/count of each); `k` is doubled on both axes until at most
/// `MAX_ARROWS` remain. Returns the cells and the final `k`.
pub fn lattice(windows: &[TileSpec], k: [usize; 2]) -> (Vec<(usize, usize)>, [usize; 2]) {
    let mut k = [k[0].max(1), k[1].max(1)];
    loop {
        let total: usize = windows
            .iter()
            .map(|w| {
                multiples(w.start[0], w.count[0], k[0])
                    .saturating_mul(multiples(w.start[1], w.count[1], k[1]))
            })
            .fold(0, usize::saturating_add);
        if total <= MAX_ARROWS || k[0] >= 1 << 30 || k[1] >= 1 << 30 {
            break;
        }
        k = [k[0] * 2, k[1] * 2];
    }
    let mut cells = Vec::new();
    for w in windows {
        let first = |a: usize| w.start[a].div_ceil(k[a]) * k[a];
        let mut iy = first(0);
        while iy < w.start[0] + w.count[0] {
            let mut ix = first(1);
            while ix < w.start[1] + w.count[1] {
                cells.push((iy, ix));
                ix += k[1];
            }
            iy += k[0];
        }
    }
    cells.truncate(MAX_ARROWS);
    (cells, k)
}

/// Unit screen direction (world axes, y up) of a vector with the given
/// east/north components at `(lon, lat)` degrees. The projection's scaled
/// tangents are taken numerically: `e = dF/dlon / cos(lat)`, `n = dF/dlat`.
/// `None` near the poles (|lat| > 89.9), where `forward` fails, or where the
/// result is not a usable direction.
pub fn screen_dir(proj: Projection, lon: f64, lat: f64, east: f32, north: f32) -> Option<[f32; 2]> {
    if !lon.is_finite() || !lat.is_finite() || lat.abs() > POLE_LIMIT_DEG {
        return None;
    }
    let f0 = proj.forward(lon, lat)?;
    let fe = proj.forward(lon + DELTA_DEG, lat)?;
    let fn_ = proj.forward(lon, lat + DELTA_DEG)?;
    let d = DELTA_DEG.to_radians();
    let mut dx = fe[0] - f0[0];
    if proj.wraps_x() {
        // The step may cross the seam: undo the 2*pi jump.
        if dx > PI {
            dx -= TAU;
        } else if dx < -PI {
            dx += TAU;
        }
    }
    let cos_phi = lat.to_radians().cos();
    let e = [dx / (d * cos_phi), (fe[1] - f0[1]) / (d * cos_phi)];
    let n = [(fn_[0] - f0[0]) / d, (fn_[1] - f0[1]) / d];
    let (east, north) = (east as f64, north as f64);
    let v = [east * e[0] + north * n[0], east * e[1] + north * n[1]];
    let len = v[0].hypot(v[1]);
    if !len.is_finite() || len < 1e-12 {
        return None;
    }
    let out = [(v[0] / len) as f32, (v[1] / len) as f32];
    (out[0].is_finite() && out[1].is_finite()).then_some(out)
}

/// 95th percentile of the finite values above zero.
pub fn p95(mags: impl Iterator<Item = f32>) -> Option<f32> {
    let mut v: Vec<f32> = mags.filter(|m| m.is_finite() && *m > 0.0).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_unstable_by(f32::total_cmp);
    Some(v[((v.len() - 1) as f64 * 0.95).floor() as usize])
}

/// One arrow, laid out for the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArrowInstance {
    /// World position (x in [-pi, pi) for wrapping projections).
    pub pos: [f32; 2],
    pub dir: [f32; 2],
    pub mag: f32,
    pub _pad: f32,
}

/// Centre `(lon, lat)` of a full-resolution cell.
pub type LonLatFn = Arc<dyn Fn(usize, usize) -> Option<(f64, f64)> + Send + Sync>;

/// The world arrows are placed in (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArrowSpace {
    Map(Projection),
    Image {
        /// Rows grow upwards on screen.
        flip_y: bool,
        /// Screen width over height of one cell.
        aspect: f32,
    },
}

/// Owned inputs, so the build can run in a background job.
pub struct ArrowInput {
    pub space: ArrowSpace,
    /// Finest first. Texel `(ty, tx)` of a tile is element
    /// `ty * spec.shape()[1] + tx` of its slice.
    pub tiles: Vec<(TileSpec, Arc<Slice>)>,
    /// Centre coordinates `(lon, lat)` of a full-resolution cell (map only).
    pub lonlat: LonLatFn,
    pub cells: Vec<(usize, usize)>,
}

pub struct ArrowSet {
    pub instances: Vec<ArrowInstance>,
    pub p95: Option<f32>,
}

/// Samples each lattice cell from the finest tile covering it, at that
/// texel's own full-resolution cell. Cells with missing components or
/// coordinates, or no usable direction, are skipped; so are duplicates.
pub fn build_arrows(input: &ArrowInput) -> ArrowSet {
    let mut seen = HashSet::new();
    let mut instances = Vec::new();
    for &(iy, ix) in &input.cells {
        if instances.len() >= MAX_ARROWS {
            break;
        }
        let Some((spec, slice, t)) = input
            .tiles
            .iter()
            .find_map(|(s, sl)| s.nearest_texel(iy, ix).map(|t| (s, sl, t)))
        else {
            continue;
        };
        let cell = [0, 1].map(|a| spec.start[a] + t[a] * spec.stride[a].max(1));
        if !seen.insert((cell[0], cell[1])) {
            continue;
        }
        let Some(parts) = slice.vector.as_ref() else { continue };
        // Not `slice.shape`: a tile may hold the request's slice, [1, ny, nx].
        let at = t[0] * spec.shape()[1] + t[1];
        let (Some(&east), Some(&north)) = (parts.east.get(at), parts.north.get(at)) else {
            continue;
        };
        if east.is_nan() || north.is_nan() {
            continue;
        }
        let Some((pos, dir)) = place(input, cell, east, north) else { continue };
        let mag = match slice.data.get(at) {
            Some(&m) if m.is_finite() => m,
            _ => east.hypot(north),
        };
        instances.push(ArrowInstance { pos, dir, mag, _pad: 0.0 });
    }
    let p95 = p95(instances.iter().map(|i| i.mag));
    ArrowSet { instances, p95 }
}

/// World position and unit world direction of the arrow at full-resolution
/// `cell` (`[iy, ix]`).
fn place(input: &ArrowInput, cell: [usize; 2], east: f32, north: f32) -> Option<([f32; 2], [f32; 2])> {
    match input.space {
        ArrowSpace::Map(proj) => {
            let (lon, lat) = (input.lonlat)(cell[0], cell[1])?;
            let w = proj.forward(lon, lat)?;
            let dir = screen_dir(proj, lon, lat, east, north)?;
            Some(([w[0] as f32, w[1] as f32], dir))
        }
        ArrowSpace::Image { flip_y, aspect } => {
            let pos = [cell[1] as f32 + 0.5, cell[0] as f32 + 0.5];
            let aspect = if aspect.is_finite() && aspect > 0.0 { aspect } else { 1.0 };
            let v = [east / aspect, if flip_y { north } else { -north }];
            let len = v[0].hypot(v[1]);
            (len.is_finite() && len > 0.0).then_some((pos, [v[0] / len, v[1] / len]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::camera::MapCamera;
    use crate::map::locate::CellLocator;
    use crate::map::tiles::{plan_detail, visible_windows, TileSpec};
    use eframe::egui::{pos2, Rect};
    use std::sync::Arc;
    use torcdf_core::{Slice, VectorParts};

    #[test]
    fn arrow_step_powers_of_two() {
        assert_eq!(arrow_step([30.0, 4.0]), [1, 8]);
        assert_eq!(arrow_step([0.5, 100.0]), [64, 1]);
        assert_eq!(arrow_step([29.9, 31.0]), [2, 1]);
    }

    #[test]
    fn lattice_anchored_and_capped() {
        let w = TileSpec { start: [3, 5], count: [20, 20], stride: [1, 1] };
        let (cells, k) = lattice(&[w], [4, 4]);
        assert_eq!(k, [4, 4]);
        assert_eq!(cells.len(), 25);
        assert_eq!(cells[0], (4, 8));
        assert!(cells.iter().all(|&(y, x)| y % 4 == 0 && x % 4 == 0));
        let big = TileSpec { start: [0, 0], count: [1000, 1000], stride: [1, 1] };
        let (cells, k) = lattice(&[big], [1, 1]);
        assert_eq!(k, [8, 8]);
        assert!(cells.len() <= MAX_ARROWS);
    }

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
    }

    #[test]
    fn screen_dir_equirect_and_polar() {
        let eq = Projection::Equirectangular;
        assert!(close(screen_dir(eq, 10.0, 0.0, 1.0, 0.0).unwrap(), [1.0, 0.0]));
        let s = 5f32.sqrt();
        assert!(close(screen_dir(eq, 10.0, 60.0, 1.0, 1.0).unwrap(), [2.0 / s, 1.0 / s]));
        let n = screen_dir(Projection::NorthPolarStereo, 0.0, 80.0, 0.0, 1.0).unwrap();
        assert!(close(n, [0.0, 1.0]), "{n:?}");
    }

    #[test]
    fn screen_dir_polar_and_mercator_limits() {
        assert!(screen_dir(Projection::NorthPolarStereo, 0.0, 89.95, 1.0, 1.0).is_none());
        assert!(screen_dir(Projection::Mercator, 0.0, 86.0, 1.0, 1.0).is_none());
        for &p in Projection::all() {
            for lat in -90..=90 {
                for lon in (-180..=180).step_by(15) {
                    if let Some(d) = screen_dir(p, lon as f64, lat as f64, 1.0, 1.0) {
                        assert!(d.iter().all(|v| v.is_finite()), "{p:?} {lon} {lat} {d:?}");
                    }
                }
            }
        }
        // Dateline step must not blow up.
        let d = screen_dir(Projection::Equirectangular, 179.9995, 0.0, 1.0, 0.0).unwrap();
        assert!(close(d, [1.0, 0.0]), "{d:?}");
    }

    fn tile(spec: TileSpec, east: Vec<f32>, north: Vec<f32>) -> (TileSpec, Arc<Slice>) {
        let shape = spec.shape();
        let data: Vec<f32> = east.iter().zip(&north).map(|(e, n)| e.hypot(*n)).collect();
        let valid_count = data.iter().filter(|v| !v.is_nan()).count();
        let s = Slice {
            shape: shape.to_vec(),
            data,
            valid_count,
            vector: Some(VectorParts { east, north }),
        };
        (spec, Arc::new(s))
    }

    #[test]
    fn build_skips_missing_and_uses_finest() {
        let ov = TileSpec { start: [0, 0], count: [16, 16], stride: [4, 4] };
        let de = TileSpec { start: [0, 0], count: [8, 8], stride: [1, 1] };
        let mut de_e = vec![1.0f32; 64];
        de_e[2 * 8 + 2] = 3.0;
        de_e[5 * 8 + 5] = f32::NAN;
        let tiles = vec![
            tile(de, de_e, vec![0.0; 64]),
            tile(ov, vec![2.0; 16], vec![0.0; 16]),
        ];
        let input = ArrowInput {
            space: ArrowSpace::Map(Projection::Equirectangular),
            tiles,
            lonlat: Arc::new(|iy, ix| Some((ix as f64, iy as f64))),
            cells: vec![(2, 2), (12, 12), (5, 5)],
        };
        let set = build_arrows(&input);
        assert_eq!(set.instances.len(), 2);
        let a = set.instances[0];
        assert_eq!(a.mag, 3.0);
        assert!((a.pos[0] - 2f32.to_radians()).abs() < 1e-6);
        let b = set.instances[1];
        assert_eq!(b.mag, 2.0);
        assert!((b.pos[0] - 12f32.to_radians()).abs() < 1e-6);
        assert!((b.pos[1] - 12f32.to_radians()).abs() < 1e-6);
        assert!(set.p95.is_some());
    }

    #[test]
    fn build_reads_texels_by_tile_width() {
        // A tile holding the request's slice keeps its full shape, [1, ny, nx].
        let spec = TileSpec { start: [0, 0], count: [2, 3], stride: [1, 1] };
        let (_, s) = tile(spec, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![0.0; 6]);
        let mut s = Arc::try_unwrap(s).ok().expect("sole owner");
        s.shape = vec![1, 2, 3];
        let input = ArrowInput {
            space: ArrowSpace::Map(Projection::Equirectangular),
            tiles: vec![(spec, Arc::new(s))],
            lonlat: Arc::new(|iy, ix| Some((ix as f64, iy as f64))),
            cells: vec![(1, 2)],
        };
        assert_eq!(build_arrows(&input).instances[0].mag, 6.0);
    }

    #[test]
    fn image_mode_places_cells_and_keeps_north_up() {
        let spec = TileSpec { start: [0, 0], count: [600, 2], stride: [1, 1] };
        let n = 1200;
        let (_, s) = tile(spec, vec![0.0; n], vec![1.0; n]);
        let image = |flip_y, aspect| ArrowInput {
            space: ArrowSpace::Image { flip_y, aspect },
            tiles: vec![(spec, s.clone())],
            lonlat: Arc::new(|_, _| None),
            cells: vec![(500, 1)],
        };
        // Row 500 is far past any latitude, yet it gets an arrow.
        let set = build_arrows(&image(true, 1.0));
        assert_eq!(set.instances.len(), 1);
        let a = set.instances[0];
        assert_eq!(a.pos, [1.5, 500.5]);
        // North is up on screen: +row when rows grow upwards, -row otherwise.
        assert!(close(a.dir, [0.0, 1.0]), "{:?}", a.dir);
        let a = build_arrows(&image(false, 1.0)).instances[0];
        assert!(close(a.dir, [0.0, -1.0]), "{:?}", a.dir);
        // Cells twice as wide as tall: x is halved so the screen angle holds.
        let (_, s2) = tile(spec, vec![1.0; n], vec![1.0; n]);
        let input = ArrowInput { tiles: vec![(spec, s2)], ..image(true, 2.0) };
        let d = build_arrows(&input).instances[0].dir;
        let s5 = 5f32.sqrt();
        assert!(close(d, [1.0 / s5, 2.0 / s5]), "{d:?}");
    }

    #[test]
    fn visible_windows_matches_plan_core() {
        let lat = (0..43200).map(|i| -90.0 + i as f64 / 240.0).collect();
        let lon = (0..86400).map(|i| -180.0 + i as f64 / 240.0).collect();
        let loc = CellLocator::regular(lat, lon).unwrap();
        let full = [43200, 86400];
        let r = Rect::from_min_max(pos2(0.0, 0.0), pos2(1000.0, 800.0));
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.fit(&[(5.0, 40.0), (15.0, 50.0)], r);
        let ov = TileSpec::overview(full[0], full[1]);
        let plan = plan_detail(&loc, full, &cam, r, 1.0, &ov);
        let w = visible_windows(&loc, full, &cam, r).unwrap();
        assert_eq!(plan.core.len(), w.x.len());
        for (c, x) in plan.core.iter().zip(&w.x) {
            assert_eq!((c.start[0], c.start[0] + c.count[0] - 1), w.y);
            assert_eq!((c.start[1], c.start[1] + c.count[1] - 1), *x);
        }
        assert!(w.cell_pt.iter().all(|&c| c > 0.1 && c < 1.0), "{:?}", w.cell_pt);
    }
}
