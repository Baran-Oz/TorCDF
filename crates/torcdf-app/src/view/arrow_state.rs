//! Arrow settings of a vector view and the background job that places the
//! arrows.
//!
//! A build runs in a [`Job`] from data already held (no I/O). It is keyed by
//! what it was built from ([`ArrowKey`]: the tiles' slices, their specs, the
//! lattice step `k` and the world space) and covers an index window: the
//! visible core window widened by 50% per side (plan P4). A new build starts
//! only when the key changes or the visible core leaves the built window.

use std::ops::RangeInclusive;
use std::sync::{Arc, Weak};
use std::time::Instant;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use torcdf_core::{Slice, VectorParts};

use crate::app::Job;
use crate::map::arrows::{
    ARROW_SPACING_PT, ArrowInput, ArrowInstance, ArrowSet, ArrowSpace, LonLatFn, arrow_step, build_arrows,
    lattice,
};
use crate::map::tiles::{TileSpec, VisibleWindows};
use crate::map::valid_lonlat;
use crate::render::arrows::{ARROW_IMAGE_ID, ARROW_MAP_ID, upload_arrows};
use crate::ui::plot_area::Grid;
use crate::view::defaults::AxisChoice;
use crate::view::map_state::DETAIL_DELAY;
use crate::view::var_view::Coords;

/// Range of the "Arrow length" slider (a multiplier).
pub const LENGTH_RANGE: RangeInclusive<f32> = 0.25..=4.0;
const BUILD_PROBLEM: &str = "Arrows could not be prepared, so they are hidden.";

/// What an arrow set is built from. Slices compare by identity.
#[derive(Clone, Debug)]
pub struct ArrowKey {
    pub slices: Vec<Weak<Slice>>,
    pub specs: Vec<TileSpec>,
    /// Lattice step `[y, x]` asked for (before thinning to `MAX_ARROWS`).
    pub k: [usize; 2],
    pub space: ArrowSpace,
}

impl PartialEq for ArrowKey {
    fn eq(&self, o: &Self) -> bool {
        self.slices.len() == o.slices.len()
            && self.slices.iter().zip(&o.slices).all(|(a, b)| a.ptr_eq(b))
            && self.specs == o.specs
            && self.k == o.k
            && self.space == o.space
    }
}

impl ArrowKey {
    /// The key of `tiles` (finest first) at step `k` in `space`.
    pub fn of(tiles: &[(TileSpec, Arc<Slice>)], k: [usize; 2], space: ArrowSpace) -> ArrowKey {
        ArrowKey {
            slices: tiles.iter().map(|t| Arc::downgrade(&t.1)).collect(),
            specs: tiles.iter().map(|t| t.0).collect(),
            k,
            space,
        }
    }

    fn gpu_id(&self) -> u64 {
        match self.space {
            ArrowSpace::Map(_) => ARROW_MAP_ID,
            ArrowSpace::Image { .. } => ARROW_IMAGE_ID,
        }
    }
}

/// Inclusive full-resolution index windows: one row window and one or two
/// column windows (two when the view crosses a wrapping seam).
#[derive(Clone, Debug, PartialEq)]
pub struct IndexWindow {
    pub y: (usize, usize),
    pub x: Vec<(usize, usize)>,
}

impl IndexWindow {
    /// The whole `full` grid.
    #[cfg(test)]
    pub fn whole(full: [usize; 2]) -> IndexWindow {
        IndexWindow { y: (0, full[0].saturating_sub(1)), x: vec![(0, full[1].saturating_sub(1))] }
    }

    pub fn from_visible(v: &VisibleWindows) -> IndexWindow {
        IndexWindow { y: v.y, x: v.x.clone() }
    }

    /// Widened by half its span on each side (the x margin from the total
    /// column span), clamped to `full`.
    pub fn widened(&self, full: [usize; 2]) -> IndexWindow {
        let widen = |(lo, hi): (usize, usize), total: usize, n: usize| {
            let m = total.div_ceil(2);
            (lo.saturating_sub(m), (hi + m).min(n.saturating_sub(1)))
        };
        let x_total: usize = self.x.iter().map(|&(l, h)| h - l + 1).sum();
        IndexWindow {
            y: widen(self.y, self.y.1 - self.y.0 + 1, full[0]),
            x: self.x.iter().map(|&w| widen(w, x_total, full[1])).collect(),
        }
    }

    /// True when every window of `inner` lies inside one of `self`.
    pub fn contains(&self, inner: &IndexWindow) -> bool {
        let within = |o: (usize, usize), i: (usize, usize)| o.0 <= i.0 && i.1 <= o.1;
        within(self.y, inner.y) && inner.x.iter().all(|&ix| self.x.iter().any(|&ox| within(ox, ix)))
    }

    fn specs(&self) -> Vec<TileSpec> {
        let (y0, y1) = self.y;
        self.x
            .iter()
            .map(|&(x0, x1)| TileSpec { start: [y0, x0], count: [y1 - y0 + 1, x1 - x0 + 1], stride: [1, 1] })
            .collect()
    }
}

/// Arrow settings and the arrow set of a vector view.
pub struct ArrowState {
    pub on: bool,
    /// Length multiplier ([`LENGTH_RANGE`], logarithmic slider).
    pub length: f32,
    /// 95th-percentile magnitude of the arrows on show.
    pub p95: Option<f32>,
    /// Map: the lattice step and visible core window at the last rest.
    pub rest: Option<([usize; 2], IndexWindow)>,
    /// Image view: the same, refreshed by [`image_rest_due`](Self::image_rest_due).
    pub image_rest: Option<([usize; 2], IndexWindow)>,
    /// Image view rest timer: (view, panel, grid) last seen, since when,
    /// and whether `image_rest` was refreshed for it.
    image_seen: Option<(egui::Rect, egui::Rect, [usize; 2])>,
    image_seen_at: Instant,
    image_planned: bool,
    /// One-line message when a build failed (the arrows are hidden).
    pub problem: Option<String>,
    built: Option<(ArrowKey, IndexWindow)>,
    job: Option<(ArrowKey, IndexWindow, Job<ArrowSet>)>,
    /// A finished set waiting for upload: (GPU id, instances).
    ready: Option<(u64, Vec<ArrowInstance>)>,
    /// Key whose build died: not started again.
    failed: Option<ArrowKey>,
}

impl Default for ArrowState {
    fn default() -> Self {
        ArrowState {
            on: true,
            length: 1.0,
            p95: None,
            rest: None,
            image_rest: None,
            image_seen: None,
            image_seen_at: Instant::now(),
            image_planned: false,
            problem: None,
            built: None,
            job: None,
            ready: None,
            failed: None,
        }
    }
}

impl ArrowState {
    /// px_per_unit = 30 pt * ppp * length / p95.
    pub fn px_per_unit(&self, ppp: f32) -> Option<f32> {
        let p = self.p95.filter(|p| p.is_finite() && *p > 0.0)?;
        let v = ARROW_SPACING_PT as f32 * ppp * self.length / p;
        v.is_finite().then_some(v)
    }

    /// The reference arrow: a round value near p95 (1, 2 or 5 x 10^n,
    /// nearest on a log scale) and its length in points.
    pub fn reference(&self) -> Option<(f32, f32)> {
        let p = self.p95.filter(|p| p.is_finite() && *p > 0.0)? as f64;
        let e = p.log10().floor() as i32;
        let scaled = |m: f64| if e >= 0 { m * 10f64.powi(e) } else { m / 10f64.powi(-e) };
        let value = [1.0, 2.0, 5.0, 10.0]
            .map(scaled)
            .into_iter()
            .min_by(|a, b| (a / p).ln().abs().total_cmp(&(b / p).ln().abs()))?;
        let len = ARROW_SPACING_PT * self.length as f64 * value / p;
        Some((value as f32, len as f32))
    }

    /// The world space of the arrows on show.
    pub fn built_space(&self) -> Option<ArrowSpace> {
        self.built.as_ref().map(|b| b.0.space)
    }

    /// Arrows are on and the set on show was built for exactly `space`, so
    /// the affine for `space` places it correctly.
    pub fn drawable_in(&self, space: ArrowSpace) -> bool {
        self.on && self.built_space() == Some(space)
    }

    /// Image view: once `view` (cell units), the panel `rect` and the grid
    /// `full` have stayed the same for [`DETAIL_DELAY`] (no drag or wheel
    /// gesture in progress), refreshes `image_rest` from them, once per
    /// rest. Until then a repaint is scheduled for that moment.
    pub fn image_rest_due(&mut self, ctx: &egui::Context, view: egui::Rect, rect: egui::Rect, full: [usize; 2]) {
        let now = (view, rect, full);
        if self.image_seen != Some(now) {
            self.image_seen = Some(now);
            self.image_seen_at = Instant::now();
            self.image_planned = false;
        }
        if self.image_planned {
            return;
        }
        let left = DETAIL_DELAY.saturating_sub(self.image_seen_at.elapsed());
        if left.is_zero() {
            self.image_planned = true;
            if let Some(plan) = image_plan(view, rect, full) {
                self.image_rest = Some(plan);
            }
        } else {
            ctx.request_repaint_after(left);
        }
    }

    /// True when `key` over the visible `core` needs a new build: the key
    /// differs from the one built (or building), or `core` leaves its window.
    pub fn wants(&self, key: &ArrowKey, core: &IndexWindow) -> bool {
        if self.failed.as_ref() == Some(key) {
            return false;
        }
        let current = match &self.job {
            Some((k, w, _)) => Some((k, w)),
            None => self.built.as_ref().map(|(k, w)| (k, w)),
        };
        !current.is_some_and(|(k, w)| k == key && w.contains(core))
    }

    /// Starts building `key` over `core` widened by 50% per side (clamped
    /// to `full`); `make` runs in the job with that window and gives the
    /// input (its cells are filled in there). A build in flight is superseded.
    pub fn start(
        &mut self,
        ctx: &egui::Context,
        key: ArrowKey,
        core: &IndexWindow,
        full: [usize; 2],
        make: impl FnOnce(&IndexWindow) -> ArrowInput + Send + 'static,
    ) {
        let window = core.widened(full);
        let (specs, k, w) = (window.specs(), key.k, window.clone());
        let job = Job::spawn(ctx, move || {
            let mut input = make(&w);
            input.cells = lattice(&specs, k).0;
            build_arrows(&input)
        });
        self.job = Some((key, window, job));
    }

    /// [`wants`](Self::wants), then [`start`](Self::start). True when a job started.
    pub fn update(
        &mut self,
        ctx: &egui::Context,
        key: ArrowKey,
        core: &IndexWindow,
        full: [usize; 2],
        make: impl FnOnce(&IndexWindow) -> ArrowInput + Send + 'static,
    ) -> bool {
        let want = self.wants(&key, core);
        if want {
            self.start(ctx, key, core, full, make);
        }
        want
    }

    /// `key` cannot be built (its input is missing): say so, and do not
    /// try it again.
    pub fn give_up(&mut self, key: ArrowKey) {
        self.failed = Some(key);
        self.problem = Some(BUILD_PROBLEM.into());
    }

    /// Takes a finished build in. True when a new set is ready for upload.
    pub fn poll_finished(&mut self) -> bool {
        let Some(r) = self.job.as_ref().and_then(|j| j.2.poll()) else { return false };
        let Some((key, window, _)) = self.job.take() else { return false };
        match r {
            Ok(set) => {
                self.p95 = set.p95;
                self.ready = Some((key.gpu_id(), set.instances));
                self.built = Some((key, window));
                self.problem = None;
                true
            }
            Err(()) => {
                self.built = None;
                self.give_up(key);
                false
            }
        }
    }

    /// Uploads a finished set, if any.
    pub fn upload_ready(&mut self, rs: &RenderState) {
        if let Some((id, instances)) = self.ready.take() {
            upload_arrows(rs, id, &instances);
        }
    }
}

/// Map cell centres `(lon, lat)`: the 1-D coordinates on regular grids, the
/// 2-D ones (missing values skipped) on curvilinear grids.
pub fn map_lonlat(curvilinear: bool, coords: &Coords, axes: AxisChoice, full: [usize; 2]) -> Option<LonLatFn> {
    if curvilinear {
        let (lat, lon) = coords.curv.clone()?;
        let nx = full[1];
        return Some(Arc::new(move |iy, ix| {
            let i = iy.checked_mul(nx)?.checked_add(ix)?;
            let (lo, la) = (*lon.get(i)?, *lat.get(i)?);
            valid_lonlat(lo, la).then_some((lo, la))
        }));
    }
    let lat = Arc::new(coords.dims.get(axes.y)?.clone()?);
    let lon = Arc::new(coords.dims.get(axes.x)?.clone()?);
    Some(Arc::new(move |iy, ix| Some((*lon.get(ix)?, *lat.get(iy)?))))
}

/// Image view: the lattice step for the cell size on screen and the cells
/// `view` (cell units) shows in `rect`, clamped to the `full` grid. `None`
/// when no cell is on show.
pub fn image_plan(view: egui::Rect, rect: egui::Rect, full: [usize; 2]) -> Option<([usize; 2], IndexWindow)> {
    let span = |lo: f32, hi: f32, n: usize| -> Option<(usize, usize)> {
        if n == 0 || !(lo.is_finite() && hi.is_finite()) {
            return None;
        }
        let (a, b) = (lo.floor().max(0.0), (hi.ceil() - 1.0).min(n as f32 - 1.0));
        (a <= b).then_some((a as usize, b as usize))
    };
    let core = IndexWindow {
        y: span(view.min.y, view.max.y, full[0])?,
        x: vec![span(view.min.x, view.max.x, full[1])?],
    };
    let cell_pt = [rect.height() / view.height(), rect.width() / view.width()].map(f64::from);
    Some((arrow_step(cell_pt), core))
}

/// Image view input over `window` (image rows and columns). A plain slice
/// is used as it is (one `[ny, nx]` tile); an x-before-y one has only the
/// window's cells turned to `[y, x]`. Runs in the job.
pub fn image_input(slice: Arc<Slice>, grid: Grid, space: ArrowSpace, window: &IndexWindow) -> ArrowInput {
    let whole = TileSpec { start: [0, 0], count: [grid.ny, grid.nx], stride: [1, 1] };
    let mut tiles = Vec::new();
    if !grid.transposed {
        tiles.push((whole, slice));
    } else {
        let x0 = window.x.iter().map(|w| w.0).min().unwrap_or(0);
        let x1 = window.x.iter().map(|w| w.1).max().unwrap_or(0).min(grid.nx.saturating_sub(1));
        let (y0, y1) = (window.y.0, window.y.1.min(grid.ny.saturating_sub(1)));
        if y0 <= y1 && x0 <= x1 {
            // `d` is nx rows (x outer) of ny values.
            let pick = |d: &[f32]| -> Vec<f32> {
                (y0..=y1)
                    .flat_map(|r| (x0..=x1).map(move |c| d.get(c * grid.ny + r).copied().unwrap_or(f32::NAN)))
                    .collect()
            };
            let data = pick(&slice.data);
            let part = Slice {
                shape: vec![y1 - y0 + 1, x1 - x0 + 1],
                valid_count: data.iter().filter(|v| !v.is_nan()).count(),
                data,
                vector: slice.vector.as_ref().map(|p| VectorParts { east: pick(&p.east), north: pick(&p.north) }),
            };
            let spec = TileSpec { start: [y0, x0], count: [y1 - y0 + 1, x1 - x0 + 1], stride: [1, 1] };
            tiles.push((spec, Arc::new(part)));
        }
    }
    ArrowInput { space, tiles, lonlat: Arc::new(|_, _| None), cells: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::projection::Projection;

    #[test]
    fn px_per_unit_from_p95() {
        let mut s = ArrowState::default();
        assert_eq!(s.px_per_unit(1.0), None);
        s.p95 = Some(2.0);
        assert_eq!(s.px_per_unit(1.0), Some(15.0));
        s.length = 2.0;
        assert_eq!(s.px_per_unit(1.5), Some(45.0));
    }

    #[test]
    fn reference_is_round() {
        let mut s = ArrowState { p95: Some(0.37), ..ArrowState::default() };
        let (value, len) = s.reference().expect("reference");
        assert_eq!(value, 0.5);
        assert!((len - 30.0 * 0.5 / 0.37).abs() < 1e-4, "{len}");
        s.p95 = Some(180.0);
        assert_eq!(s.reference().map(|r| r.0), Some(200.0));
        s.p95 = Some(1.2);
        assert_eq!(s.reference().map(|r| r.0), Some(1.0));
        s.p95 = None;
        assert_eq!(s.reference(), None);
    }

    fn input(_: &IndexWindow) -> ArrowInput {
        // The whole 1000 x 1000 grid, every 100th cell.
        let spec = TileSpec { start: [0, 0], count: [1000, 1000], stride: [100, 100] };
        let slice = Slice {
            shape: vec![10, 10],
            data: vec![1.0; 100],
            valid_count: 100,
            vector: Some(VectorParts { east: vec![1.0; 100], north: vec![0.0; 100] }),
        };
        ArrowInput {
            space: ArrowSpace::Map(Projection::Equirectangular),
            tiles: vec![(spec, Arc::new(slice))],
            // 0.1 degree cells from (-50, -50).
            lonlat: Arc::new(|iy, ix| Some((ix as f64 * 0.1 - 50.0, iy as f64 * 0.1 - 50.0))),
            cells: Vec::new(),
        }
    }

    fn key(k: usize) -> ArrowKey {
        ArrowKey {
            slices: Vec::new(),
            specs: Vec::new(),
            k: [k, k],
            space: ArrowSpace::Map(Projection::Equirectangular),
        }
    }

    fn window(y: (usize, usize), x: (usize, usize)) -> IndexWindow {
        IndexWindow { y, x: vec![x] }
    }

    #[test]
    fn rebuild_only_on_key_change() {
        let ctx = egui::Context::default();
        let full = [1000, 1000];
        let core = window((400, 499), (400, 499));
        let mut s = ArrowState::default();
        assert!(s.update(&ctx, key(4), &core, full, input), "first build");
        assert!(!s.update(&ctx, key(4), &core, full, input), "same key: no second job");
        // A pan inside the built window (50% margin per side) changes nothing.
        let panned = window((440, 539), (370, 469));
        assert!(!s.update(&ctx, key(4), &panned, full, input), "pan inside the window");
        // Leaving the window, or a new key, builds again.
        let far = window((700, 799), (400, 499));
        assert!(s.update(&ctx, key(4), &far, full, input), "pan out of the window");
        assert!(s.update(&ctx, key(8), &far, full, input), "new k");
        // The finished job is taken in and its p95 kept.
        let t0 = std::time::Instant::now();
        while !s.poll_finished() {
            assert!(t0.elapsed().as_secs() < 20, "arrow job timed out");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(s.p95, Some(1.0));
        assert!(!s.update(&ctx, key(8), &far, full, input), "built key: nothing to do");
        // Only a set built for exactly this space may be drawn.
        let built = ArrowSpace::Map(Projection::Equirectangular);
        assert!(s.drawable_in(built));
        assert!(!s.drawable_in(ArrowSpace::Map(Projection::Mercator)));
        assert!(!s.drawable_in(ArrowSpace::Image { flip_y: true, aspect: 1.0 }));
        s.on = false;
        assert!(!s.drawable_in(built), "switched off");
    }

    #[test]
    fn image_gate_rejects_old_flip_or_aspect() {
        let ctx = egui::Context::default();
        let grid = Grid { ny: 4, nx: 4, transposed: false };
        let slice = Arc::new(Slice {
            shape: vec![4, 4],
            data: vec![1.0; 16],
            valid_count: 16,
            vector: Some(VectorParts { east: vec![1.0; 16], north: vec![0.0; 16] }),
        });
        let space = ArrowSpace::Image { flip_y: true, aspect: 1.0 };
        let tile = TileSpec { start: [0, 0], count: [4, 4], stride: [1, 1] };
        let key = ArrowKey::of(&[(tile, slice.clone())], [1, 1], space);
        let mut s = ArrowState::default();
        let core = IndexWindow::whole([4, 4]);
        assert!(s.update(&ctx, key, &core, [4, 4], move |w| image_input(slice, grid, space, w)));
        let t0 = std::time::Instant::now();
        while !s.poll_finished() {
            assert!(t0.elapsed().as_secs() < 20, "arrow job timed out");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(s.drawable_in(space));
        assert!(!s.drawable_in(ArrowSpace::Image { flip_y: false, aspect: 1.0 }), "old flip_y");
        assert!(!s.drawable_in(ArrowSpace::Image { flip_y: true, aspect: 2.0 }), "old aspect");
    }

    #[test]
    fn image_window_follows_the_view_when_zoomed() {
        let full = [2048, 2048];
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 800.0));
        // 16 x 16 cells on show: 50 pt per cell, so k = 1.
        let view = egui::Rect::from_min_max(egui::pos2(1000.0, 500.0), egui::pos2(1016.0, 516.0));
        let (k, core) = image_plan(view, rect, full).expect("visible");
        assert_eq!(k, arrow_step([50.0, 50.0]));
        assert_eq!(k, [1, 1]);
        assert_eq!(core, IndexWindow { y: (500, 515), x: vec![(1000, 1015)] });
        // Built over the visible window plus 50%, the lattice keeps k (no thinning).
        let window = core.widened(full);
        assert_eq!(window, IndexWindow { y: (492, 523), x: vec![(992, 1023)] });
        let (cells, k_used) = lattice(&window.specs(), k);
        assert_eq!(k_used, k);
        assert_eq!(cells.len(), 32 * 32);
        // The whole grid would be thinned to every 16th cell: 800 pt apart.
        assert_eq!(lattice(&IndexWindow::whole(full).specs(), k).1, [16, 16]);
        // Partly off the image: clamped; wholly off: nothing.
        let edge = egui::Rect::from_min_max(egui::pos2(-4.5, 2040.2), egui::pos2(3.5, 2050.0));
        let (_, core) = image_plan(edge, rect, full).expect("partly visible");
        assert_eq!(core, IndexWindow { y: (2040, 2047), x: vec![(0, 3)] });
        let off = egui::Rect::from_min_max(egui::pos2(-20.0, 0.0), egui::pos2(-10.0, 10.0));
        assert!(image_plan(off, rect, full).is_none());
    }

    #[test]
    fn transposed_image_input_holds_only_the_window() {
        // x before y: data is [x = 3][y = 2].
        let grid = Grid { ny: 2, nx: 3, transposed: true };
        let d: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let parts = VectorParts { east: d.iter().map(|v| v + 10.0).collect(), north: d.clone() };
        let slice = Arc::new(Slice { shape: vec![3, 2], data: d, valid_count: 6, vector: Some(parts) });
        let space = ArrowSpace::Image { flip_y: true, aspect: 1.0 };
        let w = IndexWindow { y: (1, 1), x: vec![(1, 2)] };
        let input = image_input(slice, grid, space, &w);
        let (spec, s) = &input.tiles[0];
        assert_eq!(*spec, TileSpec { start: [1, 1], count: [1, 2], stride: [1, 1] });
        // (row 1, col 1) = data[1 * 2 + 1], (row 1, col 2) = data[2 * 2 + 1].
        assert_eq!(s.data, vec![3.0, 5.0]);
        assert_eq!(s.vector.as_ref().map(|p| p.east.clone()), Some(vec![13.0, 15.0]));
    }

    #[test]
    fn window_widens_by_half_per_side() {
        let w = window((400, 499), (0, 99)).widened([1000, 1000]);
        assert_eq!(w, window((350, 549), (0, 149)));
        assert!(w.contains(&window((360, 540), (10, 140))));
        assert!(!w.contains(&window((360, 560), (10, 140))));
    }
}
