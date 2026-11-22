//! Which part of a grid a GPU tile holds.

use super::camera::MapCamera;
use super::locate::CellLocator;
use crate::view::defaults::{overview_stride, OVERVIEW_TARGET};
use eframe::egui::{pos2, Rect};

/// A strided window of the plotted dimensions: tile texel `(ty, tx)` holds the
/// full-resolution cell `(start_y + ty * stride_y, start_x + tx * stride_x)`.
/// All arrays are `[y, x]`, in full-resolution indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileSpec {
    pub start: [usize; 2],
    pub count: [usize; 2],
    pub stride: [usize; 2],
}

impl TileSpec {
    /// The whole `ny` x `nx` grid thinned to at most `OVERVIEW_TARGET` per side.
    pub fn overview(ny: usize, nx: usize) -> TileSpec {
        TileSpec {
            start: [0, 0],
            count: [ny, nx],
            stride: [overview_stride(ny, OVERVIEW_TARGET), overview_stride(nx, OVERVIEW_TARGET)],
        }
    }

    /// Texture size `[ny, nx]` of the tile.
    pub fn shape(&self) -> [usize; 2] {
        [0, 1].map(|a| self.count[a].div_ceil(self.stride[a].max(1)))
    }

    /// True when `other`'s index window lies inside this tile's.
    pub fn contains(&self, other: &TileSpec) -> bool {
        (0..2).all(|a| {
            self.start[a] <= other.start[a]
                && other.start[a] + other.count[a] <= self.start[a] + self.count[a]
        })
    }

    /// Tile texel nearest to full-resolution cell `(iy, ix)`, if the cell is
    /// inside the tile's extent.
    pub fn nearest_texel(&self, iy: usize, ix: usize) -> Option<[usize; 2]> {
        let shape = self.shape();
        let mut out = [0; 2];
        for (a, i) in [iy, ix].into_iter().enumerate() {
            let d = i.checked_sub(self.start[a])?;
            let s = self.stride[a].max(1);
            if d >= self.count[a] {
                return None;
            }
            out[a] = ((d + s / 2) / s).min(shape[a] - 1);
        }
        Some(out)
    }
}

/// Detail tiles are planned only when the overview is coarser than the
/// on-screen need by more than this factor on some axis.
pub const DETAIL_TRIGGER: f64 = 1.5;

/// Largest texture side of a tile.
const MAX_SIDE: usize = 2048;
const _: () = assert!(OVERVIEW_TARGET <= MAX_SIDE);
/// Sample points per screen axis (edges included).
const SAMPLES: usize = 33;

/// Plans 0, 1 or 2 strided detail tiles covering the visible window.
///
/// The rect is sampled on a 33 x 33 grid (points evenly spaced, edges
/// included) through the camera and `loc.nearest`; misses are skipped. The
/// window is the hit bounding box (on a lon-wrapping grid: the complement of
/// the largest circular gap in the hit columns, split in two when it crosses
/// the seam), widened by 10% per side and clamped to `full`. The plan also
/// keeps each tile's window before the margin (`core`).
///
/// `window_px`, the framebuffer extent the window covers on an axis, is
/// `rect_px * hit_fraction * window_cells / hit_cells`, where `hit_fraction`
/// is the share of sample rows (columns) that hit the grid.
pub fn plan_detail(
    loc: &CellLocator,
    full: [usize; 2],
    cam: &MapCamera,
    rect: Rect,
    pixels_per_point: f32,
    overview: &TileSpec,
) -> DetailPlan {
    let ppp = pixels_per_point as f64;
    let rect_px = [rect.height() as f64 * ppp, rect.width() as f64 * ppp];
    let sane = |v: f64| v.is_finite() && v > 0.0;
    if full.contains(&0) || !sane(rect_px[0]) || !sane(rect_px[1]) {
        return DetailPlan::default();
    }
    let (mut ys, mut xs) = (Vec::new(), Vec::new());
    // Sample rows / columns that hit: [min, max].
    let mut span = [[usize::MAX, 0usize]; 2];
    for r in 0..SAMPLES {
        for c in 0..SAMPLES {
            let p = pos2(
                rect.min.x + rect.width() * c as f32 / (SAMPLES - 1) as f32,
                rect.min.y + rect.height() * r as f32 / (SAMPLES - 1) as f32,
            );
            let hit = cam
                .screen_to_lonlat(p, rect)
                .and_then(|(lo, la)| loc.nearest(lo, la))
                .filter(|&(iy, ix)| iy < full[0] && ix < full[1]);
            if let Some((iy, ix)) = hit {
                ys.push(iy);
                xs.push(ix);
                for (a, k) in [r, c].into_iter().enumerate() {
                    span[a] = [span[a][0].min(k), span[a][1].max(k)];
                }
            }
        }
    }
    if ys.is_empty() {
        return DetailPlan::default();
    }
    let frac = |a: usize| (span[a][1] - span[a][0] + 1) as f64 / SAMPLES as f64;

    // Core windows as inclusive (lo, hi) index ranges.
    let y_core = (*ys.iter().min().unwrap_or(&0), *ys.iter().max().unwrap_or(&0));
    xs.sort_unstable();
    let (x_lo, x_hi) = (xs[0], xs[xs.len() - 1]);
    let mut x_core = vec![(x_lo, x_hi)];
    if loc.wraps_lon() {
        // Largest circular gap between consecutive sorted hit columns.
        let mut gap = x_lo + full[1] - x_hi;
        let mut at = None;
        for i in 1..xs.len() {
            if xs[i] - xs[i - 1] > gap {
                gap = xs[i] - xs[i - 1];
                at = Some(i);
            }
        }
        if let Some(i) = at {
            x_core = vec![(xs[i], full[1] - 1), (0, xs[i - 1])];
        }
    }

    // 10% margin on each side of the whole window, clamped to the grid.
    let widen = |(lo, hi): (usize, usize), total: usize, n: usize| {
        let m = total.div_ceil(10);
        (lo.saturating_sub(m), (hi + m).min(n - 1))
    };
    let y_win = widen(y_core, y_core.1 - y_core.0 + 1, full[0]);
    let x_total: usize = x_core.iter().map(|&(l, h)| h - l + 1).sum();
    let x_win: Vec<(usize, usize)> = x_core.iter().map(|&w| widen(w, x_total, full[1])).collect();

    let cells = |ws: &[(usize, usize)]| ws.iter().map(|&(l, h)| h - l + 1).sum::<usize>();
    let core_cells = [y_core.1 - y_core.0 + 1, x_total];
    let win_cells = [cells(&[y_win]), cells(&x_win)];
    let mut stride = [1usize; 2];
    let longest = [y_win.1 - y_win.0 + 1, x_win.iter().map(|&(l, h)| h - l + 1).max().unwrap_or(1)];
    for a in 0..2 {
        let px = (rect_px[a] * frac(a) * win_cells[a] as f64 / core_cells[a] as f64).max(1.0);
        stride[a] = ((win_cells[a] as f64 / px).floor() as usize).max(1);
        // Never coarser than the overview on any axis, then the texture cap.
        // For the app's overview (OVERVIEW_TARGET <= MAX_SIDE per side) the
        // cap cannot undo the first: the window lies inside the grid.
        stride[a] = stride[a].min(overview.stride[a].max(1));
        stride[a] = stride[a].max(longest[a].div_ceil(MAX_SIDE));
    }

    let needed = (0..2).any(|a| overview.stride[a] as f64 > DETAIL_TRIGGER * stride[a] as f64);
    if !needed {
        return DetailPlan::default();
    }
    let spec = |y: (usize, usize), (l, h): (usize, usize)| TileSpec {
        start: [y.0, l],
        count: [y.1 - y.0 + 1, h - l + 1],
        stride,
    };
    DetailPlan {
        tiles: x_win.iter().map(|&x| spec(y_win, x)).collect(),
        core: x_core.iter().map(|&x| spec(y_core, x)).collect(),
    }
}

/// The detail tiles to read and, for each, the window it must cover: the
/// same tile before the 10% margin.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetailPlan {
    pub tiles: Vec<TileSpec>,
    pub core: Vec<TileSpec>,
}

impl DetailPlan {
    /// True when the tiles `held` (planned earlier) already give what this
    /// plan needs: every core window lies inside a held tile of the same
    /// stride. Reading this plan would then add no detail.
    pub fn covered_by(&self, held: &[TileSpec]) -> bool {
        !self.core.is_empty()
            && self.core.iter().all(|c| held.iter().any(|h| h.stride == c.stride && h.contains(c)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::map::projection::Projection;
    use eframe::egui::{pos2, Rect};

    fn regular(nlat: usize, nlon: usize, dlat: f64, lon0: f64, dlon: f64, lat0: f64) -> CellLocator {
        let lat = (0..nlat).map(|i| lat0 + i as f64 * dlat).collect();
        let lon = (0..nlon).map(|i| lon0 + i as f64 * dlon).collect();
        CellLocator::regular(lat, lon).unwrap()
    }

    fn rect(w: f32, h: f32) -> Rect {
        Rect::from_min_max(pos2(0.0, 0.0), pos2(w, h))
    }

    #[test]
    fn world_view_needs_no_detail() {
        let loc = regular(43200, 86400, 1.0 / 240.0, -180.0, 1.0 / 240.0, -90.0);
        let full = [43200, 86400];
        let r = rect(1600.0, 800.0);
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.fit(&[(-180.0, -90.0), (179.99, 90.0)], r);
        let ov = TileSpec::overview(full[0], full[1]);
        assert!(plan_detail(&loc, full, &cam, r, 1.0, &ov).tiles.is_empty());
    }

    #[test]
    fn zoomed_view_requests_window() {
        let loc = regular(43200, 86400, 1.0 / 240.0, -180.0, 1.0 / 240.0, -90.0);
        let full = [43200, 86400];
        let r = rect(1000.0, 800.0);
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.fit(&[(5.0, 40.0), (15.0, 50.0)], r);
        let ov = TileSpec::overview(full[0], full[1]);
        let t = plan_detail(&loc, full, &cam, r, 1.0, &ov).tiles;
        assert_eq!(t.len(), 1);
        let t = t[0];
        // lat 40..50 is rows 31200..33600; margin is about 10% of the window.
        assert!(t.start[0] <= 31200 && t.start[0] >= 31200 - 600, "{t:?}");
        assert!(t.start[0] + t.count[0] >= 33600 && t.start[0] + t.count[0] <= 33600 + 600, "{t:?}");
        assert!(t.stride[0] <= 3 && t.stride[1] <= 3 && t.stride[0] >= 1, "{t:?}");
        assert!(t.shape().iter().all(|&n| n <= 2048), "{t:?}");
    }

    #[test]
    fn seam_splits_detail_window() {
        let loc = regular(18001, 36000, 0.01, 0.0, 0.01, -90.0);
        let full = [18001, 36000];
        let r = rect(1000.0, 800.0);
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.centre = [0.0, 0.0];
        cam.scale = 1000.0 / 20f64.to_radians();
        let ov = TileSpec::overview(full[0], full[1]);
        assert_eq!(ov.stride, [9, 18]);
        let t = plan_detail(&loc, full, &cam, r, 1.0, &ov).tiles;
        assert_eq!(t.len(), 2, "{t:?}");
        assert!(t.iter().any(|t| t.start[1] == 0), "{t:?}");
        assert!(t.iter().any(|t| t.start[1] + t.count[1] == 36000), "{t:?}");
        assert_eq!(t[0].stride, t[1].stride);
    }

    #[test]
    fn small_grid_never_details() {
        let loc = regular(380, 1016, 0.1, -10.0, 0.1, 30.0);
        let full = [380, 1016];
        let r = rect(1000.0, 800.0);
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.fit(&[(-10.0, 30.0), (91.0, 68.0)], r);
        cam.zoom_at(r.center(), 10.0, r);
        let ov = TileSpec::overview(full[0], full[1]);
        assert_eq!(ov.stride, [1, 1]);
        assert!(plan_detail(&loc, full, &cam, r, 1.0, &ov).tiles.is_empty());
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let loc = regular(10, 10, 1.0, 0.0, 1.0, 0.0);
        let cam = MapCamera::new(Projection::Equirectangular);
        let ov = TileSpec::overview(10, 10);
        assert!(plan_detail(&loc, [10, 10], &cam, rect(0.0, 0.0), 1.0, &ov).tiles.is_empty());
        assert!(plan_detail(&loc, [0, 0], &cam, rect(100.0, 100.0), 1.0, &ov).tiles.is_empty());
        assert!(plan_detail(&loc, [10, 10], &cam, rect(100.0, 100.0), 0.0, &ov).tiles.is_empty());
    }

    #[test]
    fn tile_spec_overview_and_texels() {
        let t = TileSpec::overview(43200, 86400);
        assert_eq!(t.stride, [22, 43]);
        assert_eq!(t.nearest_texel(22, 43), Some([1, 1]));
        assert_eq!(t.nearest_texel(23, 44), Some([1, 1]));
        assert_eq!(t.nearest_texel(43200, 0), None);
        assert!(t.shape().iter().all(|&n| n <= OVERVIEW_TARGET));
    }

    #[test]
    fn detail_stride_never_coarser_than_overview() {
        // Rows dense (1/240 degree, overview stride 22), columns sparse
        // enough for the overview to hold them all (2000 of 0.005 degree):
        // at 50 px per degree the rows need detail; the columns, several per pixel,
        // must not get a tile coarser than the overview.
        let loc = regular(43200, 2000, 1.0 / 240.0, 0.0, 0.005, -90.0);
        let full = [43200, 2000];
        let r = rect(1000.0, 800.0);
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.centre = [10f64.to_radians(), 0.0];
        cam.scale = 1000.0 / 20f64.to_radians();
        let ov = TileSpec::overview(full[0], full[1]);
        assert_eq!(ov.stride, [22, 1]);
        let t = plan_detail(&loc, full, &cam, r, 1.0, &ov).tiles;
        assert_eq!(t.len(), 1, "{t:?}");
        assert!(t[0].stride[0] < 22, "{t:?}");
        assert_eq!(t[0].stride[1], 1, "{t:?}");
        assert!(t[0].shape().iter().all(|&n| n <= MAX_SIDE), "{t:?}");
    }

    #[test]
    fn held_tiles_cover_small_pans_only() {
        let loc = regular(43200, 86400, 1.0 / 240.0, -180.0, 1.0 / 240.0, -90.0);
        let full = [43200, 86400];
        let r = rect(1000.0, 800.0);
        let ov = TileSpec::overview(full[0], full[1]);
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.fit(&[(5.0, 40.0), (15.0, 50.0)], r);
        let held = plan_detail(&loc, full, &cam, r, 1.0, &ov);
        assert_eq!(held.tiles.len(), 1);
        assert!(held.covered_by(&held.tiles));
        assert!(!DetailPlan::default().covered_by(&held.tiles), "an empty plan clears");
        // A small pan moves the window but stays inside the margin.
        let mut c = cam;
        c.pan(eframe::egui::vec2(20.0, -15.0));
        let p = plan_detail(&loc, full, &c, r, 1.0, &ov);
        assert_ne!(p.tiles, held.tiles);
        assert!(p.covered_by(&held.tiles), "{p:?} vs {held:?}");
        // A pan beyond the held tile needs a read.
        let mut c = cam;
        c.pan(eframe::egui::vec2(300.0, 0.0));
        assert!(!plan_detail(&loc, full, &c, r, 1.0, &ov).covered_by(&held.tiles));
        // Zooming in needs a finer stride.
        let mut c = cam;
        c.zoom_at(r.center(), 3.0, r);
        let p = plan_detail(&loc, full, &c, r, 1.0, &ov);
        assert!(p.tiles[0].stride[0] < held.tiles[0].stride[0], "{p:?} vs {held:?}");
        assert!(!p.covered_by(&held.tiles));
    }
}
