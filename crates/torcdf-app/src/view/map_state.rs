//! Map view state: camera, overlay switches, the cell locator, the GPU data
//! tile and line sets, and the background jobs that prepare them.
//!
//! Nothing here reads a file. Heavy CPU work (locator, curvilinear meshes,
//! projecting coastlines and grid lines) runs in `Job`s; the UI thread only
//! uploads finished results. Panning and zooming change `camera` alone.

use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{Rect, pos2};
use eframe::egui_wgpu::RenderState;
use torcdf_core::cf::Geo;
use torcdf_core::{CoreError, RequestHandle, Slice};

use crate::app::Job;
use crate::map::basemap::{borders, coastlines};
use crate::map::camera::MapCamera;
use crate::map::graticule::{graticule_step, graticule_window};
use crate::map::locate::CellLocator;
use crate::map::mesh::{CurvMesh, MESH_MAX_SIDE, MeshVertex, build_mesh, project_mesh};
use crate::map::projection::Projection;
use crate::map::tiles::{TileSpec, plan_detail};
use crate::render::image::UploadError;
use crate::render::lines::{project_polylines, upload_lines};
use crate::render::map_curv::{set_curvilinear_mesh, upload_curvilinear};
use crate::render::map_regular::{MapTile, upload_regular};
use crate::view::defaults::AxisChoice;
use crate::view::var_view::Coords;

/// The camera must rest this long before detail tiles are planned.
pub const DETAIL_DELAY: Duration = Duration::from_millis(300);
const DETAIL_PROBLEM: &str = "Detail could not be loaded; showing the overview.";
pub const MAP_OVERVIEW_ID: u64 = 10;
pub const MAP_DETAIL_IDS: [u64; 2] = [11, 12];
pub const COAST_ID: u64 = 20;
pub const BORDERS_ID: u64 = 21;
pub const GRID_ID: u64 = 22;

/// Cell centres per axis the camera fits to.
const FIT_SAMPLES: usize = 32;
/// Sample points per screen axis when finding the visible lat range.
const WINDOW_SAMPLES: usize = 17;

/// The value at full-resolution cell `(iy, ix)`. `tiles[0]` is the overview;
/// every slice holds its tile's values row-major as `[ty, tx]`. The finest
/// tile (smallest stride) whose extent holds the cell gives its nearest
/// texel, as drawn (the overview is simply the coarsest tile); `None`
/// outside every tile. Returns the value and the full-resolution cell
/// that texel holds (`start + t * stride`), so a hover index names the
/// cell whose value it shows.
pub fn value_at(tiles: &[(TileSpec, &Slice)], iy: usize, ix: usize) -> Option<(f32, (usize, usize))> {
    let get = |spec: &TileSpec, s: &Slice, t: [usize; 2]| {
        let v = s.data.get(t[0] * spec.shape()[1] + t[1]).copied()?;
        Some((v, (spec.start[0] + t[0] * spec.stride[0], spec.start[1] + t[1] * spec.stride[1])))
    };
    // Finest first; on a tie a detail tile beats the overview (`tiles[0]`).
    let mut finest: Vec<(usize, &(TileSpec, &Slice))> = tiles.iter().enumerate().collect();
    finest.sort_by_key(|(i, (spec, _))| (spec.stride[0].saturating_mul(spec.stride[1]), *i == 0));
    finest
        .into_iter()
        .find_map(|(_, (spec, s))| spec.nearest_texel(iy, ix).map(|t| (spec, s, t)))
        .and_then(|(spec, s, t)| get(spec, s, t))
}

/// [`value_at`] for the hover readout: a missing value (NaN or ±inf) counts
/// as no value, so it reads "—" like a point outside the data.
pub fn hover_value(tiles: &[(TileSpec, &Slice)], iy: usize, ix: usize) -> Option<(f32, (usize, usize))> {
    value_at(tiles, iy, ix).filter(|(v, _)| v.is_finite())
}

/// A lon/lat window in degrees. `lon` may run past ±180 (it follows the
/// camera across the seam); a span of 360 or more means "all longitudes".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoWindow {
    pub lon: [f64; 2],
    pub lat: [f64; 2],
}

impl GeoWindow {
    fn full_lon(&self) -> bool {
        self.lon[1] - self.lon[0] >= 360.0 - 1e-9
    }

    /// `self` widened by half its span on every side (lat clamped to ±90,
    /// lon to all longitudes once it would reach 360 degrees).
    pub fn widened(&self) -> GeoWindow {
        let half = |a: [f64; 2]| 0.5 * (a[1] - a[0]).max(1e-6);
        let (hy, hx) = (half(self.lat), half(self.lon));
        let lat = [(self.lat[0] - hy).max(-90.0), (self.lat[1] + hy).min(90.0)];
        let lon = if self.lon[1] - self.lon[0] + 2.0 * hx >= 360.0 {
            [-180.0, 180.0]
        } else {
            [self.lon[0] - hx, self.lon[1] + hx]
        };
        GeoWindow { lon, lat }
    }

    /// True when `inner` lies inside `self`, allowing a whole-turn shift in
    /// longitude (the camera centre wraps at ±180).
    pub fn contains(&self, inner: &GeoWindow) -> bool {
        let lat_ok = self.lat[0] <= inner.lat[0] && inner.lat[1] <= self.lat[1];
        let lon_ok = self.full_lon()
            || [-360.0, 0.0, 360.0]
                .iter()
                .any(|k| self.lon[0] <= inner.lon[0] + k && inner.lon[1] + k <= self.lon[1]);
        lat_ok && lon_ok
    }
}

/// The lon/lat window `cam` shows in `rect`; `None` when nothing of the
/// globe is on screen. Cylindrical projections read it off the rect edges;
/// polar ones sample the rect (all longitudes, and the pole when visible).
pub fn visible_window(cam: &MapCamera, rect: Rect) -> Option<GeoWindow> {
    let (w, h) = (rect.width() as f64, rect.height() as f64);
    if !(w > 0.0 && h > 0.0 && cam.scale > 0.0) {
        return None;
    }
    if cam.proj.wraps_x() {
        let (hw, hh) = (0.5 * w / cam.scale, 0.5 * h / cam.scale);
        let x = [cam.centre[0] - hw, cam.centre[0] + hw].map(f64::to_degrees);
        let lon = if x[1] - x[0] >= 360.0 { [-180.0, 180.0] } else { x };
        let lat_at = |y: f64| match cam.proj.inverse(0.0, y) {
            Some((_, la)) => la,
            None => 90f64.copysign(y),
        };
        let lat = [lat_at(cam.centre[1] - hh), lat_at(cam.centre[1] + hh)];
        return Some(GeoWindow { lon, lat });
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let n = WINDOW_SAMPLES - 1;
    for r in 0..=n {
        for c in 0..=n {
            let p = pos2(
                rect.min.x + rect.width() * c as f32 / n as f32,
                rect.min.y + rect.height() * r as f32 / n as f32,
            );
            if let Some((_, la)) = cam.screen_to_lonlat(p, rect) {
                (lo, hi) = (lo.min(la), hi.max(la));
            }
        }
    }
    if rect.contains(cam.world_to_screen([0.0, 0.0], rect)) {
        if cam.proj == Projection::NorthPolarStereo {
            hi = 90.0;
        } else {
            lo = -90.0;
        }
    }
    (lo <= hi).then_some(GeoWindow { lon: [-180.0, 180.0], lat: [lo, hi] })
}

/// The grid-line set on the GPU (or being prepared): its projection, step
/// and the window it covers.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GridKey {
    proj: Projection,
    step: f64,
    window: GeoWindow,
}

/// Projected coastlines and borders, as line segment vertex lists.
type BasemapLines = (Vec<[f32; 2]>, Vec<[f32; 2]>);

/// A curvilinear tile's mesh (geographic) and its projection for `proj`.
struct MeshSet {
    spec: TileSpec,
    geo: Arc<CurvMesh>,
    proj: Projection,
    verts: Vec<MeshVertex>,
    indices: Vec<u32>,
}

/// A data tile on the GPU.
pub struct Tile {
    pub spec: TileSpec,
    /// The tile's values as `[ty, tx]` row-major (for hover).
    pub data: Arc<Slice>,
    pub gpu: MapTile,
}

/// The finished locator job.
struct Located {
    locator: CellLocator,
    fit: Vec<(f64, f64)>,
}

/// Why an overview upload did not happen.
pub enum UploadFailure {
    /// The coordinates cannot be drawn as a map: fall back to the image.
    Coords(String),
    /// Anything else (texture too large, no GPU): a message in the plot area.
    Other(String),
}

pub struct MapState {
    pub camera: MapCamera,
    pub coast: bool,
    pub borders: bool,
    pub grid: bool,
    /// Set once the locator job finishes.
    pub locator: Option<CellLocator>,
    fit_points: Vec<(f64, f64)>,
    locator_job: Option<Job<Result<Located, String>>>,
    needs_fit: bool,
    curvilinear: bool,
    /// The overview tile on the GPU.
    pub overview: Option<Tile>,
    /// The slice last offered to the overview upload (so a failure is not
    /// retried every frame).
    tried: Option<Arc<Slice>>,
    mesh: Option<MeshSet>,
    mesh_job: Option<((TileSpec, Projection), Job<MeshSet>)>,
    /// Projection of the curvilinear mesh on the GPU.
    mesh_on_gpu: Option<Projection>,
    /// Projection of the coastline and border sets on the GPU.
    basemap_on_gpu: Option<Projection>,
    basemap_job: Option<(Projection, Job<BasemapLines>)>,
    grid_on_gpu: Option<GridKey>,
    grid_job: Option<(GridKey, Job<Vec<[f32; 2]>>)>,
    /// Keys whose job died: never started again for the same key.
    basemap_failed: Option<Projection>,
    grid_failed: Option<GridKey>,
    mesh_failed: Option<(TileSpec, Projection)>,
    /// One-line message when the data tile cannot be prepared.
    pub tile_problem: Option<String>,
    /// One-line message when an overlay cannot be prepared (it is skipped).
    pub overlay_problem: Option<String>,
    /// One-line message when detail tiles could not be loaded (the overview stays).
    pub detail_problem: Option<String>,
    /// The pointer is over the map this frame.
    pub hovering: bool,
    /// Lon/lat under the pointer, when it is on the globe.
    pub pointer: Option<(f64, f64)>,
    /// The detail tiles asked for (at most 2), whether or not they arrived.
    plan: Vec<TileSpec>,
    /// Reads in flight for `plan`; dropping them cancels the unstarted ones.
    handles: Vec<(TileSpec, RequestHandle)>,
    /// Tiles of `plan` read but not yet on the GPU (uploaded together).
    arrived: Vec<(TileSpec, Arc<Slice>)>,
    detail_requested: Instant,
    /// Detail tiles on screen; held until the next plan's tiles all arrive.
    detail: Vec<DetailTile>,
    detail_meshes: Vec<MeshSet>,
    detail_mesh_jobs: Vec<((TileSpec, Projection), Job<MeshSet>)>,
    detail_mesh_failed: Vec<(TileSpec, Projection)>,
    /// Camera and rect size last seen, and when they last changed.
    seen: Option<(MapCamera, [f32; 2])>,
    seen_at: Instant,
    /// The plan for the current resting camera was made.
    planned: bool,
    /// A plan waiting for `VarView::start_detail` to request it.
    wanted: Option<Vec<TileSpec>>,
    /// Debug log lines (`TORCDF_LOG`); `None` when logging is off.
    pub events: Option<Vec<String>>,
    /// What the map last did this frame, for the slow-frame log.
    pub activity: &'static str,
}

/// A detail tile on the GPU, with the projection of its mesh (curvilinear).
struct DetailTile {
    tile: Tile,
    mesh_proj: Option<Projection>,
}

impl MapState {
    pub fn new(geo: &Geo) -> MapState {
        MapState {
            camera: MapCamera::new(Projection::default()),
            coast: true,
            borders: true,
            grid: true,
            locator: None,
            fit_points: Vec::new(),
            locator_job: None,
            needs_fit: true,
            curvilinear: matches!(geo, Geo::Curvilinear { .. }),
            overview: None,
            tried: None,
            mesh: None,
            mesh_job: None,
            mesh_on_gpu: None,
            basemap_on_gpu: None,
            basemap_job: None,
            grid_on_gpu: None,
            grid_job: None,
            basemap_failed: None,
            grid_failed: None,
            mesh_failed: None,
            tile_problem: None,
            overlay_problem: None,
            detail_problem: None,
            hovering: false,
            pointer: None,
            plan: Vec::new(),
            handles: Vec::new(),
            arrived: Vec::new(),
            detail_requested: Instant::now(),
            detail: Vec::new(),
            detail_meshes: Vec::new(),
            detail_mesh_jobs: Vec::new(),
            detail_mesh_failed: Vec::new(),
            seen: None,
            seen_at: Instant::now(),
            planned: false,
            wanted: None,
            events: None,
            activity: "",
        }
    }

    fn note(&mut self, msg: impl FnOnce() -> String) {
        if let Some(e) = &mut self.events {
            e.push(msg());
        }
    }

    /// True once per rest: [`DETAIL_DELAY`] after the camera (or the rect)
    /// last changed. Until then a repaint is scheduled for that moment.
    pub fn detail_due(&mut self, ctx: &eframe::egui::Context, rect: Rect) -> bool {
        let now = (self.camera, [rect.width(), rect.height()]);
        if self.seen != Some(now) {
            self.seen = Some(now);
            self.seen_at = Instant::now();
            self.planned = false;
        }
        if self.planned {
            return false;
        }
        let left = DETAIL_DELAY.saturating_sub(self.seen_at.elapsed());
        if left.is_zero() {
            self.planned = true;
            true
        } else {
            ctx.request_repaint_after(left);
            false
        }
    }

    /// Plans the detail tiles for the resting camera; the plan is kept for
    /// `VarView::start_detail` unless it equals the current one or the
    /// current tiles already cover it (same stride, every visible window
    /// inside a current tile: a pan within the margin re-reads nothing).
    pub fn plan_detail_for(
        &mut self,
        full: [usize; 2],
        rect: Rect,
        ppp: f32,
        overview: &TileSpec,
    ) {
        let Some(loc) = &self.locator else { return };
        let t = Instant::now();
        let detail = plan_detail(loc, full, &self.camera, rect, ppp, overview);
        let covered = detail.covered_by(&self.plan);
        let mut plan = detail.tiles;
        plan.truncate(MAP_DETAIL_IDS.len());
        self.activity = "detail plan";
        if plan != self.plan && !covered {
            self.note(|| {
                format!("detail plan {plan:?} ({:.1} ms)", t.elapsed().as_secs_f64() * 1e3)
            });
            self.wanted = Some(plan);
        }
    }

    pub fn take_wanted(&mut self) -> Option<Vec<TileSpec>> {
        self.wanted.take()
    }

    /// Makes `plan` current: reads for the old one are dropped. The tiles on
    /// show stay until the new ones arrive; an empty plan clears them.
    pub fn set_plan(&mut self, mut plan: Vec<TileSpec>) {
        plan.truncate(MAP_DETAIL_IDS.len());
        self.detail_problem = None;
        self.plan = plan;
        self.handles.clear();
        self.arrived.clear();
        let keep = |s: &TileSpec| self.plan.contains(s) || self.detail.iter().any(|d| d.tile.spec == *s);
        self.detail_meshes.retain(|m| keep(&m.spec));
        self.detail_mesh_failed.retain(|(s, _)| keep(s));
        if self.plan.is_empty() {
            self.detail.clear();
            self.detail_meshes.clear();
            self.detail_mesh_jobs.clear();
        }
    }

    /// Forgets all detail tiles and plans (the view kind changed).
    fn clear_detail(&mut self) {
        self.set_plan(Vec::new());
        self.wanted = None;
        self.planned = false;
    }

    /// Takes the read of detail tile `spec` unless it is no longer in the
    /// plan (a stale result). True when stored.
    pub fn accept_detail(&mut self, spec: TileSpec, slice: Arc<Slice>) -> bool {
        if !self.plan.contains(&spec) || self.arrived.iter().any(|a| a.0 == spec) {
            return false;
        }
        self.arrived.push((spec, slice));
        true
    }

    /// Starts tracking the read of detail tile `spec`; a cache hit arrives at
    /// once.
    pub fn track_detail(&mut self, spec: TileSpec, handle: RequestHandle) {
        self.note(|| format!("detail request {spec:?}"));
        self.handles.push((spec, handle));
        self.poll_detail();
    }

    /// Collects finished detail reads. A failed read leaves its tile out of
    /// the plan (and is not retried until the camera next rests).
    pub fn poll_detail(&mut self) {
        let mut i = 0;
        while i < self.handles.len() {
            let Some(r) = self.handles[i].1.try_take() else {
                i += 1;
                continue;
            };
            let (spec, _) = self.handles.remove(i);
            let ms = self.detail_requested.elapsed().as_secs_f64() * 1e3;
            self.on_detail_result(spec, r, ms);
        }
    }

    /// A finished read of detail tile `spec`. A failed read leaves the tile
    /// out of the plan and reports a message.
    fn on_detail_result(&mut self, spec: TileSpec, r: Result<Arc<Slice>, CoreError>, ms: f64) {
        match r {
            Ok(slice) => {
                let stored = self.accept_detail(spec, slice);
                self.note(|| format!("detail tile {spec:?} arrived {ms:.1} ms after request, kept {stored}"));
            }
            Err(e) => {
                self.plan.retain(|s| *s != spec);
                self.detail_failed();
                self.note(|| format!("detail tile {spec:?} failed after {ms:.1} ms: {e}"));
            }
        }
    }

    /// Detail could not be loaded: say so, and drop the held tiles when no
    /// tile of the plan is left.
    fn detail_failed(&mut self) {
        self.detail_problem = Some(DETAIL_PROBLEM.into());
        if self.plan.is_empty() {
            self.detail.clear();
            self.detail_meshes.clear();
            self.detail_mesh_jobs.clear();
            self.detail_mesh_failed.clear();
        }
    }

    /// Marks the time detail requests were issued (for the debug log).
    pub fn detail_requested_now(&mut self) {
        self.detail_requested = Instant::now();
    }

    /// Drops reads and arrivals for a re-request of the same plan (slider
    /// move); returns the plan. The tiles on show stay.
    pub fn restart_detail(&mut self) -> Vec<TileSpec> {
        self.handles.clear();
        self.arrived.clear();
        self.plan.clone()
    }

    pub fn is_curvilinear(&self) -> bool {
        self.curvilinear
    }

    /// Switches projection; the camera re-fits and overlays and meshes are
    /// re-projected (in jobs) on the next frames.
    pub fn set_projection(&mut self, proj: Projection) {
        if proj != self.camera.proj {
            self.camera = MapCamera::new(proj);
            self.needs_fit = true;
            self.overlay_problem = None;
        }
    }

    /// Re-fit the camera to the data on the next frame.
    pub fn request_fit(&mut self) {
        self.needs_fit = true;
    }

    /// Forget the uploaded data tile (the view kind changed).
    pub fn reset_tiles(&mut self) {
        self.overview = None;
        self.tried = None;
        self.mesh_on_gpu = None;
        self.clear_detail();
    }

    /// Starts building the locator from the coordinates just read. `Err`
    /// says why the variable cannot be mapped.
    pub fn start_locator(
        &mut self,
        ctx: &eframe::egui::Context,
        geo: &Geo,
        coords: &Coords,
        shape: &[usize],
    ) -> Result<(), String> {
        let dim = |d: usize| coords.dims.get(d).and_then(Option::clone);
        let job = match geo {
            Geo::Regular { lat_dim, lon_dim, .. } => {
                let (Some(lat), Some(lon)) = (dim(*lat_dim), dim(*lon_dim)) else {
                    return Err("its latitude/longitude coordinates could not be read".into());
                };
                Job::spawn(ctx, move || {
                    let locator = CellLocator::regular(lat, lon).ok_or_else(|| {
                        "its latitude/longitude must be finite and strictly increasing or decreasing"
                            .to_string()
                    })?;
                    let fit = locator.sample_lonlat(FIT_SAMPLES);
                    Ok(Located { locator, fit })
                })
            }
            Geo::Curvilinear { y_dim, x_dim, .. } => {
                let Some((lat, lon)) = coords.curv.clone() else {
                    return Err("its 2-D latitude/longitude could not be read".into());
                };
                let (ny, nx) = (shape[*y_dim], shape[*x_dim]);
                Job::spawn(ctx, move || {
                    let locator = CellLocator::curvilinear((*lat).clone(), (*lon).clone(), ny, nx);
                    let fit = locator.sample_lonlat(FIT_SAMPLES);
                    if fit.is_empty() {
                        return Err("its 2-D latitude/longitude has no valid values".into());
                    }
                    Ok(Located { locator, fit })
                })
            }
            Geo::None => return Err("it has no latitude/longitude coordinates".into()),
        };
        self.locator_job = Some(job);
        Ok(())
    }

    /// Collects the locator job. `Some(reason)` when the variable cannot be
    /// mapped after all.
    pub fn poll_locator(&mut self) -> Option<String> {
        let r = self.locator_job.as_ref()?.poll()?;
        self.locator_job = None;
        match r {
            Ok(Ok(l)) => {
                self.locator = Some(l.locator);
                self.fit_points = l.fit;
                None
            }
            Ok(Err(why)) => Some(why),
            Err(()) => Some("building its cell index failed unexpectedly".into()),
        }
    }

    /// Fits the camera to the data when asked to (first show, projection
    /// change, double-click). A grid spanning all longitudes is centred on
    /// Greenwich.
    pub fn fit_if_needed(&mut self, rect: Rect) {
        let Some(loc) = &self.locator else { return };
        if !self.needs_fit || !(rect.width() > 0.0 && rect.height() > 0.0) {
            return;
        }
        self.camera.fit(&self.fit_points, rect);
        if loc.wraps_lon() && self.camera.proj.wraps_x() {
            self.camera.centre[0] = 0.0;
        }
        self.needs_fit = false;
    }

    /// Per-frame bookkeeping: collects finished jobs, uploads finished line
    /// sets and meshes, and starts the jobs the current camera needs. Work
    /// is started only when the projection, the grid step, the tile or the
    /// visible window (beyond the generated margin) changes.
    pub fn prepare(
        &mut self,
        ctx: &eframe::egui::Context,
        rs: &RenderState,
        rect: Rect,
        coords: &Coords,
        full: [usize; 2],
        spec: Option<TileSpec>,
    ) {
        let proj = self.camera.proj;

        // Coastlines and borders: once per projection.
        if self.basemap_on_gpu != Some(proj)
            && self.basemap_failed != Some(proj)
            && self.basemap_job.as_ref().map(|j| j.0) != Some(proj)
        {
            let job = Job::spawn(ctx, move || {
                (project_polylines(proj, coastlines()), project_polylines(proj, borders()))
            });
            self.basemap_job = Some((proj, job));
        }
        if let Some((p, job)) = &self.basemap_job
            && let Some(r) = job.poll()
        {
            let p = *p;
            self.basemap_job = None;
            match r {
                Ok((coast, borders)) => {
                    self.activity = "line upload";
                    upload_lines(rs, COAST_ID, &coast);
                    upload_lines(rs, BORDERS_ID, &borders);
                    self.basemap_on_gpu = Some(p);
                }
                Err(()) => {
                    self.basemap_failed = Some(p);
                    self.overlay_problem =
                        Some("Coastlines and borders could not be prepared, so they are hidden.".into());
                }
            }
        }

        // Grid lines: per projection, step and visible window.
        if self.grid
            && let Some(vis) = visible_window(&self.camera, rect)
        {
            let step = graticule_step((100.0 / self.camera.scale).to_degrees());
            let covers =
                |k: &GridKey| k.proj == proj && k.step == step && k.window.contains(&vis);
            let have = self.grid_on_gpu.as_ref().is_some_and(covers)
                || self.grid_failed.as_ref().is_some_and(covers)
                || self.grid_job.as_ref().is_some_and(|(k, _)| covers(k));
            if !have {
                let key = GridKey { proj, step, window: vis.widened() };
                let job = Job::spawn(ctx, move || {
                    project_polylines(proj, &graticule_window(step, key.window.lon, key.window.lat))
                });
                self.grid_job = Some((key, job));
            }
        }
        if let Some((key, job)) = &self.grid_job
            && let Some(r) = job.poll()
        {
            let key = *key;
            self.grid_job = None;
            match r {
                Ok(segments) => {
                    self.activity = "line upload";
                    upload_lines(rs, GRID_ID, &segments);
                    self.grid_on_gpu = Some(key);
                }
                Err(()) => {
                    self.grid_failed = Some(key);
                    self.overlay_problem =
                        Some("Grid lines could not be prepared, so they are hidden.".into());
                }
            }
        }

        // Curvilinear mesh of the overview tile: per tile and projection.
        if let (true, Some(spec), Some((lat, lon))) = (self.curvilinear, spec, &coords.curv) {
            let want = (spec, proj);
            let have = self.mesh.as_ref().is_some_and(|m| (m.spec, m.proj) == want)
                || self.mesh_failed == Some(want)
                || self.mesh_job.as_ref().is_some_and(|(k, _)| *k == want);
            if !have {
                let geo = self.mesh.as_ref().filter(|m| m.spec == spec).map(|m| m.geo.clone());
                let (lat, lon) = (lat.clone(), lon.clone());
                let job = Job::spawn(ctx, move || {
                    let geo = geo.unwrap_or_else(|| {
                        Arc::new(build_mesh(&lat, &lon, full, &spec, MESH_MAX_SIDE))
                    });
                    let (verts, indices) = project_mesh(proj, &geo);
                    MeshSet { spec, geo, proj, verts, indices }
                });
                self.mesh_job = Some((want, job));
            }
        }
        if let Some((key, job)) = &self.mesh_job
            && let Some(r) = job.poll()
        {
            let key = *key;
            self.mesh_job = None;
            match r {
                Ok(m) => {
                    self.mesh = Some(m);
                    self.tile_problem = None;
                }
                Err(()) => {
                    self.mesh_failed = Some(key);
                    self.tile_problem = Some(
                        "Cannot draw this map: preparing its curvilinear grid failed unexpectedly."
                            .into(),
                    );
                }
            }
        }
        // A re-projected mesh for the tile already on the GPU.
        if let (Some(t), Some(m)) = (&self.overview, &self.mesh)
            && self.curvilinear
            && t.spec == m.spec
            && m.proj == proj
            && self.mesh_on_gpu != Some(proj)
        {
            set_curvilinear_mesh(rs, MAP_OVERVIEW_ID, &m.verts, &m.indices);
            self.mesh_on_gpu = Some(proj);
        }
    }

    /// Uploads `slice` (the overview read for `spec`) unless it is already
    /// on the GPU, or was tried and failed. `Ok(true)` when it was uploaded
    /// now; `Ok(false)` when nothing was to do or a curvilinear tile is
    /// still waiting for its mesh.
    pub fn upload_overview(
        &mut self,
        rs: &RenderState,
        slice: &Arc<Slice>,
        spec: TileSpec,
        axes: AxisChoice,
        coords: &Coords,
    ) -> Result<bool, UploadFailure> {
        if self.tried.as_ref().is_some_and(|t| Arc::ptr_eq(t, slice)) {
            return Ok(false);
        }
        let mesh = if self.curvilinear {
            let proj = self.camera.proj;
            let Some(m) = self.mesh.as_ref().filter(|m| m.spec == spec && m.proj == proj) else {
                return Ok(false);
            };
            Some(m)
        } else {
            None
        };
        let t = Instant::now();
        let result = build_tile(rs, MAP_OVERVIEW_ID, slice, spec, axes, coords, mesh);
        self.tried = Some(slice.clone());
        self.activity = "overview upload";
        let took = t.elapsed().as_secs_f64() * 1e3;
        match result {
            Ok(tile) => {
                self.overview = Some(tile);
                if self.curvilinear {
                    self.mesh_on_gpu = Some(self.camera.proj);
                }
                self.note(|| format!("overview uploaded in {took:.1} ms"));
                Ok(true)
            }
            Err(f) => {
                self.overview = None;
                Err(f)
            }
        }
    }

    /// Per frame: starts mesh jobs for curvilinear detail tiles, and once
    /// every planned tile has arrived (and has its mesh) uploads them all
    /// at once, replacing the tiles on show. A tile that cannot be uploaded
    /// is skipped with a log line. `Ok`-path result: true when uploaded.
    pub fn prepare_detail(
        &mut self,
        ctx: &eframe::egui::Context,
        rs: &RenderState,
        axes: AxisChoice,
        coords: &Coords,
        full: [usize; 2],
    ) -> bool {
        let proj = self.camera.proj;
        if self.curvilinear
            && let Some((lat, lon)) = &coords.curv
        {
            // Meshes for tiles that arrived and for those on show (re-projection).
            let specs: Vec<TileSpec> = self
                .arrived
                .iter()
                .map(|a| a.0)
                .chain(self.detail.iter().map(|d| d.tile.spec))
                .collect();
            for spec in specs {
                let want = (spec, proj);
                let have = self.detail_meshes.iter().any(|m| (m.spec, m.proj) == want)
                    || self.detail_mesh_failed.contains(&want)
                    || self.detail_mesh_jobs.iter().any(|(k, _)| *k == want);
                if have {
                    continue;
                }
                let geo = self.detail_meshes.iter().find(|m| m.spec == spec).map(|m| m.geo.clone());
                let (lat, lon) = (lat.clone(), lon.clone());
                let job = Job::spawn(ctx, move || {
                    let geo = geo.unwrap_or_else(|| {
                        Arc::new(build_mesh(&lat, &lon, full, &spec, MESH_MAX_SIDE))
                    });
                    let (verts, indices) = project_mesh(proj, &geo);
                    MeshSet { spec, geo, proj, verts, indices }
                });
                self.detail_mesh_jobs.push((want, job));
            }
            let mut i = 0;
            while i < self.detail_mesh_jobs.len() {
                let Some(r) = self.detail_mesh_jobs[i].1.poll() else {
                    i += 1;
                    continue;
                };
                let (key, _) = self.detail_mesh_jobs.remove(i);
                match r {
                    Ok(m) => {
                        self.detail_meshes.retain(|o| o.spec != m.spec);
                        self.detail_meshes.push(m);
                    }
                    Err(()) => {
                        self.detail_mesh_failed.push(key);
                        self.detail_problem = Some(DETAIL_PROBLEM.into());
                        self.note(|| format!("detail mesh {key:?} failed"));
                    }
                }
            }
            // Tiles on show get their re-projected mesh.
            for d in &mut self.detail {
                if d.mesh_proj != Some(proj)
                    && let Some(m) = self.detail_meshes.iter().find(|m| m.spec == d.tile.spec && m.proj == proj)
                {
                    set_curvilinear_mesh(rs, d.tile.gpu.id, &m.verts, &m.indices);
                    d.mesh_proj = Some(proj);
                    self.activity = "detail mesh upload";
                }
            }
        }

        if self.plan.is_empty() || self.arrived.len() != self.plan.len() {
            return false;
        }
        let meshes: Vec<Option<&MeshSet>> = self
            .arrived
            .iter()
            .map(|(spec, _)| self.detail_meshes.iter().find(|m| m.spec == *spec && m.proj == proj))
            .collect();
        if self.curvilinear && meshes.iter().any(Option::is_none) {
            // Wait for the mesh jobs; a failed mesh drops its tile.
            let failed: Vec<TileSpec> = self
                .arrived
                .iter()
                .map(|a| a.0)
                .filter(|s| self.detail_mesh_failed.contains(&(*s, proj)))
                .collect();
            if !failed.is_empty() {
                self.plan.retain(|s| !failed.contains(s));
                self.arrived.retain(|a| !failed.contains(&a.0));
                self.detail_failed();
            }
            return false;
        }
        let t = Instant::now();
        let mut tiles = Vec::new();
        let mut notes = Vec::new();
        for (i, (spec, slice)) in self.arrived.iter().enumerate() {
            match build_tile(rs, MAP_DETAIL_IDS[i], slice, *spec, axes, coords, meshes[i]) {
                Ok(tile) => {
                    tiles.push(DetailTile { tile, mesh_proj: self.curvilinear.then_some(proj) })
                }
                Err(UploadFailure::Coords(e) | UploadFailure::Other(e)) => {
                    notes.push(format!("detail tile {spec:?} not uploaded: {e}"));
                }
            }
        }
        self.arrived.clear();
        self.activity = "detail upload";
        let took = t.elapsed().as_secs_f64() * 1e3;
        notes.push(format!("detail uploaded: {} tiles in {took:.1} ms", tiles.len()));
        if notes.iter().any(|n| n.contains("not uploaded")) {
            self.detail_problem = Some(DETAIL_PROBLEM.into());
        } else {
            self.detail_problem = None;
        }
        self.detail = tiles;
        for n in notes {
            self.note(|| n);
        }
        true
    }

    /// Detail tiles that can be drawn with the current projection, in order.
    pub fn drawable_details(&self) -> impl Iterator<Item = &Tile> {
        let proj = self.camera.proj;
        self.detail
            .iter()
            .filter(move |d| !self.curvilinear || d.mesh_proj == Some(proj))
            .map(|d| &d.tile)
    }

    /// The overview tile, when it can be drawn with the current projection.
    pub fn drawable_overview(&self) -> Option<&Tile> {
        let t = self.overview.as_ref()?;
        let ok = !self.curvilinear || self.mesh_on_gpu == Some(self.camera.proj);
        ok.then_some(t)
    }

    /// Coastlines and borders are on the GPU for the current projection.
    pub fn basemap_ready(&self) -> bool {
        self.basemap_on_gpu == Some(self.camera.proj)
    }

    /// Grid lines are on the GPU for the current projection.
    pub fn grid_ready(&self) -> bool {
        self.grid_on_gpu.is_some_and(|k| k.proj == self.camera.proj)
    }

    /// Hover value of full-resolution cell `(iy, ix)` from the tiles on
    /// show, with the cell actually sampled; `None` for a missing value.
    pub fn value(&self, iy: usize, ix: usize) -> Option<(f32, (usize, usize))> {
        let t = self.overview.as_ref()?;
        let mut tiles = vec![(t.spec, &*t.data)];
        tiles.extend(self.detail.iter().map(|d| (d.tile.spec, &*d.tile.data)));
        hover_value(&tiles, iy, ix)
    }
}

/// Uploads one data tile as texture `id`. Curvilinear tiles need their
/// `mesh`. Transposed (x-before-y) variables are turned to `[y, x]`; the
/// returned tile's `data` is row-major as `[ty, tx]`.
fn build_tile(
    rs: &RenderState,
    id: u64,
    slice: &Arc<Slice>,
    spec: TileSpec,
    axes: AxisChoice,
    coords: &Coords,
    mesh: Option<&MeshSet>,
) -> Result<Tile, UploadFailure> {
    let [ny, nx] = spec.shape();
    let grid = crate::ui::plot_area::Grid { ny, nx, transposed: axes.y > axes.x };
    let data = crate::ui::plot_area::image_data(&slice.data, &grid);
    let result = if let Some(m) = mesh {
        upload_curvilinear(rs, id, &data, ny, nx, &m.verts, &m.indices)
    } else {
        let strided = |d: usize, a: usize| -> Option<Vec<f64>> {
            let c = coords.dims.get(d)?.as_ref()?;
            (0..spec.shape()[a]).map(|k| c.get(spec.start[a] + k * spec.stride[a]).copied()).collect()
        };
        let (Some(lat), Some(lon)) = (strided(axes.y, 0), strided(axes.x, 1)) else {
            return Err(UploadFailure::Coords("its coordinates do not match its shape".into()));
        };
        upload_regular(rs, id, &data, ny, nx, &lat, &lon)
    };
    match result {
        Ok(gpu) => {
            let data = match data {
                std::borrow::Cow::Borrowed(_) => slice.clone(),
                std::borrow::Cow::Owned(v) => {
                    Arc::new(Slice {
                        shape: vec![ny, nx],
                        valid_count: slice.valid_count,
                        data: v,
                        vector: None,
                    })
                }
            };
            Ok(Tile { spec, data, gpu })
        }
        Err(UploadError::BadCoords(why)) => Err(UploadFailure::Coords(why.into())),
        Err(e) => Err(UploadFailure::Other(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::vec2;

    fn filled(spec: &TileSpec, v: f32) -> Slice {
        let [ny, nx] = spec.shape();
        Slice { shape: vec![ny, nx], data: vec![v; ny * nx], valid_count: ny * nx, vector: None }
    }

    #[test]
    fn value_at_prefers_detail() {
        let overview = TileSpec { start: [0, 0], count: [32, 32], stride: [4, 4] };
        let detail = TileSpec { start: [8, 8], count: [8, 8], stride: [1, 1] };
        let (o, d) = (filled(&overview, 1.0), filled(&detail, 2.0));
        let tiles = [(overview, &o), (detail, &d)];
        assert_eq!(value_at(&tiles, 10, 10), Some((2.0, (10, 10))));
        // Outside the detail tile: the overview's nearest texel, and its cell.
        assert_eq!(value_at(&tiles, 1, 1), Some((1.0, (0, 0))));
        assert_eq!(value_at(&tiles, 3, 30), Some((1.0, (4, 28))));
        assert_eq!(value_at(&tiles, 10_000, 0), None);
    }

    #[test]
    fn value_at_uses_detail_off_its_lattice() {
        // A stride-2 detail tile is drawn over its whole extent, so a cell
        // between its texels reads the detail's nearest texel, not the overview.
        let overview = TileSpec { start: [0, 0], count: [64, 64], stride: [8, 8] };
        let detail = TileSpec { start: [8, 8], count: [9, 9], stride: [2, 2] };
        let (o, d) = (filled(&overview, 1.0), filled(&detail, 2.0));
        let tiles = [(overview, &o), (detail, &d)];
        assert_eq!(value_at(&tiles, 9, 11), Some((2.0, (10, 12))));
        assert_eq!(value_at(&tiles, 16, 16), Some((2.0, (16, 16))));
        // Just past the detail extent: the overview again.
        assert_eq!(value_at(&tiles, 17, 9), Some((1.0, (16, 8))));
        // The order of the tiles does not matter.
        let tiles = [(detail, &d), (overview, &o)];
        assert_eq!(value_at(&tiles, 9, 11), Some((2.0, (10, 12))));
    }

    #[test]
    fn value_at_reads_row_major_texels() {
        let spec = TileSpec { start: [0, 0], count: [2, 3], stride: [1, 1] };
        let s = Slice {
            shape: vec![2, 3],
            data: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            valid_count: 6,
            vector: None,
        };
        assert_eq!(value_at(&[(spec, &s)], 1, 2), Some((5.0, (1, 2))));
        assert_eq!(value_at(&[(spec, &s)], 0, 1), Some((1.0, (0, 1))));
        assert_eq!(value_at(&[], 0, 0), None);
    }

    fn rect() -> Rect {
        Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 400.0))
    }

    #[test]
    fn window_follows_camera_and_wraps() {
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.scale = 800.0 / 0.5; // 0.5 rad across
        cam.centre = [179f64.to_radians(), 0.0];
        let w = visible_window(&cam, rect()).unwrap();
        assert!((w.lon[0] - (179.0 - 14.3239)).abs() < 1e-3 && w.lon[1] > 180.0, "{w:?}");
        let g = w.widened();
        assert!(g.contains(&w));
        // Panning across the dateline wraps the centre to -179.5: still inside.
        cam.centre = [-179.5f64.to_radians(), 0.0];
        assert!(g.contains(&visible_window(&cam, rect()).unwrap()));
        // Far away: not inside.
        cam.centre = [0.0, 0.0];
        assert!(!g.contains(&visible_window(&cam, rect()).unwrap()));
    }

    #[test]
    fn polar_window_includes_pole() {
        let cam = MapCamera { proj: Projection::NorthPolarStereo, centre: [0.0, 0.0], scale: 100.0 };
        let w = visible_window(&cam, rect()).unwrap();
        assert_eq!(w.lat[1], 90.0);
        assert_eq!(w.lon, [-180.0, 180.0]);
        assert!(w.lat[0] < 90.0);
    }

    // Headless flow checks: real files, real jobs, the software GPU adapter.

    use crate::render::gpu_common::headless;
    use crate::view::defaults::ViewKind;
    use crate::view::var_view::VarView;
    use eframe::egui;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};
    use torcdf_core::{NcFile, SliceService, fixtures};

    fn wait(what: &str, mut done: impl FnMut() -> bool) {
        let t = Instant::now();
        while !done() {
            assert!(t.elapsed() < Duration::from_secs(30), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The view of `var`, polled until its locator and first slice are in.
    fn open_view(
        write: fn(&Path) -> PathBuf,
        var: &str,
    ) -> (VarView, egui::Context, SliceService, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let nc = Arc::new(NcFile::open(&write(dir.path())).unwrap());
        let info = nc.meta.var(var).unwrap().clone();
        let ctx = egui::Context::default();
        let service = SliceService::new(64 << 20, 2, Arc::new(|| {}));
        let mut v = VarView::new(&ctx, 1, nc, info);
        v.refresh(&ctx, &service);
        wait("locator and slice", || {
            v.poll(&ctx);
            v.map.as_ref().is_some_and(|m| m.locator.is_some()) && v.shown.is_some()
        });
        (v, ctx, service, dir)
    }


    #[test]
    fn map_flow_regular() {
        let (mut v, ctx, _service, _dir) = open_view(fixtures::write_classic, "/t");
        assert_eq!(v.kind, ViewKind::Map);
        let rs = headless::render_state();
        let axes = v.axes.unwrap();
        let shown = v.shown.as_ref().unwrap();
        let spec = TileSpec::overview(4, 8);
        let map = v.map.as_mut().unwrap();

        // First show fits the camera; a global-in-lon grid is centred on 0E.
        let before = map.camera;
        map.fit_if_needed(rect());
        assert_ne!(map.camera, before);
        assert_eq!(map.camera.centre[0], 0.0);

        assert!(matches!(map.upload_overview(&rs, &shown.slice, spec, axes, &v.coords), Ok(true)));
        // Same slice again: nothing to do.
        assert!(matches!(map.upload_overview(&rs, &shown.slice, spec, axes, &v.coords), Ok(false)));
        assert_eq!(map.value(1, 2), Some((shown.slice.data[8 + 2], (1, 2))));
        // Cell (0, 0) holds the fill value: hover reads it as no value.
        assert!(shown.slice.data[0].is_nan());
        assert_eq!(map.value(0, 0), None);
        assert!(map.drawable_overview().is_some());

        // Coastlines, borders and grid lines arrive through jobs.
        wait("overlays", || {
            map.prepare(&ctx, &rs, rect(), &v.coords, [4, 8], Some(spec));
            map.basemap_ready() && map.grid_ready()
        });
        // Panning a little stays inside the generated grid window.
        let key = map.grid_on_gpu;
        map.camera.pan(egui::vec2(20.0, 0.0));
        map.prepare(&ctx, &rs, rect(), &v.coords, [4, 8], Some(spec));
        assert!(map.grid_job.is_none() && map.grid_on_gpu == key);

        // A projection change re-fits and re-projects the overlays.
        map.set_projection(Projection::NorthPolarStereo);
        assert!(!map.basemap_ready() && !map.grid_ready());
        map.fit_if_needed(rect());
        wait("re-projected overlays", || {
            map.prepare(&ctx, &rs, rect(), &v.coords, [4, 8], Some(spec));
            map.basemap_ready() && map.grid_ready()
        });
        assert!(map.drawable_overview().is_some());
    }

    #[test]
    fn map_flow_curvilinear() {
        let (mut v, ctx, _service, _dir) = open_view(fixtures::write_curvilinear, "/sst");
        assert_eq!(v.kind, ViewKind::Map);
        assert!(v.coords.curv.is_some(), "2-D lat/lon read through read_slice");
        let rs = headless::render_state();
        let axes = v.axes.unwrap();
        let shown = v.shown.as_ref().unwrap();
        let spec = TileSpec::overview(5, 6);
        let map = v.map.as_mut().unwrap();
        assert!(map.is_curvilinear());
        map.fit_if_needed(rect());

        // The tile waits for its mesh (built in a job).
        assert!(matches!(map.upload_overview(&rs, &shown.slice, spec, axes, &v.coords), Ok(false)));
        wait("mesh and upload", || {
            map.prepare(&ctx, &rs, rect(), &v.coords, [5, 6], Some(spec));
            matches!(map.upload_overview(&rs, &shown.slice, spec, axes, &v.coords), Ok(true))
        });
        assert!(map.drawable_overview().is_some());
        assert_eq!(map.value(2, 3), Some((shown.slice.data[2 * 6 + 3], (2, 3))));

        // A projection change hides the tile until its mesh is re-projected.
        map.set_projection(Projection::Mercator);
        assert!(map.drawable_overview().is_none());
        wait("re-projected mesh", || {
            map.prepare(&ctx, &rs, rect(), &v.coords, [5, 6], Some(spec));
            map.drawable_overview().is_some()
        });
    }

    #[test]
    fn hover_treats_missing_values_as_none() {
        let spec = TileSpec { start: [0, 0], count: [1, 4], stride: [1, 1] };
        let s = Slice {
            shape: vec![1, 4],
            data: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 7.0],
            valid_count: 3,
            vector: None,
        };
        let tiles = [(spec, &s)];
        assert!(value_at(&tiles, 0, 0).is_some_and(|(v, _)| v.is_nan()));
        assert_eq!(hover_value(&tiles, 0, 0), None);
        assert_eq!(hover_value(&tiles, 0, 1), None);
        assert_eq!(hover_value(&tiles, 0, 2), None);
        assert_eq!(hover_value(&tiles, 0, 3), Some((7.0, (0, 3))));
        assert_eq!(hover_value(&tiles, 5, 0), None);
    }

    #[test]
    fn failed_jobs_are_reported_once_and_not_respawned() {
        let ctx = egui::Context::default();
        let rs = headless::render_state();
        let geo = Geo::Curvilinear {
            y_dim: 0,
            x_dim: 1,
            lat_var: "/lat".into(),
            lon_var: "/lon".into(),
        };
        let mut m = MapState::new(&geo);
        let coords = Coords {
            dims: Vec::new(),
            curv: Some((Arc::new(vec![0.0; 30]), Arc::new(vec![0.0; 30]))),
        };
        let spec = TileSpec::overview(5, 6);
        let proj = m.camera.proj;
        // Jobs that die, under the keys `prepare` would use.
        m.mesh_job = Some(((spec, proj), Job::spawn(&ctx, move || -> MeshSet {
            panic!("injected job failure")
        })));
        m.basemap_job = Some((proj, Job::spawn(&ctx, move || -> BasemapLines {
            panic!("injected job failure")
        })));
        let step = graticule_step((100.0 / m.camera.scale).to_degrees());
        let window = visible_window(&m.camera, rect()).unwrap().widened();
        let key = GridKey { proj, step, window };
        m.grid_job = Some((key, Job::spawn(&ctx, move || -> Vec<[f32; 2]> {
            panic!("injected job failure")
        })));

        wait("the injected failures", || {
            m.prepare(&ctx, &rs, rect(), &coords, [5, 6], Some(spec));
            m.mesh_failed.is_some() && m.basemap_failed.is_some() && m.grid_failed.is_some()
        });
        assert!(m.tile_problem.as_deref().is_some_and(|p| p.starts_with("Cannot draw this map")));
        assert!(m.overlay_problem.is_some());
        // Later frames do not start the same jobs again.
        for _ in 0..5 {
            m.prepare(&ctx, &rs, rect(), &coords, [5, 6], Some(spec));
            assert!(m.mesh_job.is_none() && m.basemap_job.is_none() && m.grid_job.is_none());
        }
        assert_eq!(m.mesh_failed, Some((spec, proj)));
        assert_eq!(m.grid_failed, Some(key));
        assert!(m.drawable_overview().is_none() && !m.basemap_ready() && !m.grid_ready());
    }

    #[test]
    fn stale_detail_is_ignored() {
        let geo = Geo::Regular {
            lat_dim: 0,
            lon_dim: 1,
            lat_var: "/lat".into(),
            lon_var: "/lon".into(),
        };
        let mut m = MapState::new(&geo);
        let a = TileSpec { start: [0, 0], count: [4, 4], stride: [1, 1] };
        let b = TileSpec { start: [10, 10], count: [4, 4], stride: [1, 1] };
        m.set_plan(vec![a]);
        assert!(!m.accept_detail(b, Arc::new(filled(&b, 1.0))), "not in the plan");
        assert!(m.arrived.is_empty());
        assert!(m.accept_detail(a, Arc::new(filled(&a, 2.0))));
        assert_eq!(m.arrived.len(), 1);
        // A new plan drops what arrived for the old one.
        m.set_plan(vec![b]);
        assert!(m.arrived.is_empty());
        assert!(!m.accept_detail(a, Arc::new(filled(&a, 2.0))));
    }

    #[test]
    fn detail_flow_holds_old_tiles_and_empty_plan_clears() {
        let (mut v, ctx, service, _dir) = open_view(fixtures::write_classic, "/t");
        let rs = headless::render_state();
        let axes = v.axes.unwrap();
        let ov = TileSpec::overview(4, 8);
        let a = TileSpec { start: [0, 0], count: [2, 4], stride: [1, 1] };
        let b = TileSpec { start: [1, 2], count: [2, 3], stride: [1, 1] };
        {
            let shown = v.shown.as_ref().unwrap().slice.clone();
            let map = v.map.as_mut().unwrap();
            assert!(matches!(map.upload_overview(&rs, &shown, ov, axes, &v.coords), Ok(true)));
            map.set_plan(vec![a]);
        }
        let full = [4, 8];
        let run = |v: &mut VarView| {
            v.refresh(&ctx, &service); // re-requests the overview and the plan
            wait("detail upload", || {
                v.poll(&ctx);
                v.map.as_mut().unwrap().prepare_detail(&ctx, &rs, axes, &v.coords, full)
            });
        };
        run(&mut v);
        let map = v.map.as_mut().unwrap();
        assert_eq!(map.drawable_details().count(), 1);
        let truth = map.value(1, 1);
        assert!(truth.is_some());
        // A new plan: the old tile stays until the new one arrives.
        map.set_plan(vec![b]);
        assert_eq!(map.drawable_details().count(), 1);
        assert_eq!(map.drawable_details().next().unwrap().spec, a);
        run(&mut v);
        let map = v.map.as_mut().unwrap();
        assert_eq!(map.drawable_details().next().unwrap().spec, b);
        // An empty plan clears the detail tiles.
        map.set_plan(Vec::new());
        assert_eq!(map.drawable_details().count(), 0);
    }

    /// Measures the detail reads the app would issue on a big regular grid:
    /// `TORCDF_GEBCO=<file> cargo test -p torcdf-app detail_read_timings -- --ignored --nocapture`.
    /// Camera fitted to a lon/lat box in a 1600x900 rect at ppp 1, then
    /// `plan_detail`, then each tile through `SliceService` (first touch) and
    /// `read_slice` again (warm: the file cache holds it, the service cache is
    /// bypassed).
    #[test]
    #[ignore = "needs TORCDF_GEBCO=<path to GEBCO_2025_sub_ice.nc>"]
    fn detail_read_timings() {
        use crate::map::locate::CellLocator;
        use crate::map::tiles::plan_detail;
        use crate::view::defaults::{choose_axes, tile_request};
        let Some(path) = std::env::var_os("TORCDF_GEBCO") else { return };
        let nc = Arc::new(NcFile::open(Path::new(&path)).unwrap());
        let var = nc.meta.var("/elevation").unwrap().clone();
        let axes = choose_axes(&nc.meta, &var).unwrap();
        let Geo::Regular { lat_dim, lon_dim, .. } = torcdf_core::cf::detect_geo(&nc.meta, &var)
        else {
            panic!("not a regular grid")
        };
        let coord = |d: usize| nc.read_coord(&format!("/{}", var.dims[d])).unwrap();
        let t = Instant::now();
        let loc = CellLocator::regular(coord(lat_dim), coord(lon_dim)).unwrap();
        println!("locator built in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        let full = [var.shape[axes.y], var.shape[axes.x]];
        let overview = TileSpec::overview(full[0], full[1]);
        let service = SliceService::new(256 << 20, 2, Arc::new(|| {}));
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(1600.0, 900.0));
        let mut boxes: Vec<(String, [f64; 4])> = [360.0f64, 240.0, 180.0, 120.0, 90.0, 60.0, 30.0, 20.0]
            .into_iter()
            .map(|w| {
                let h = (w * 9.0 / 16.0).min(170.0);
                (format!("{w} deg wide, centred 8E 46N"), [8.0 - w / 2.0, 46.0 - h / 2.0, 8.0 + w / 2.0, 46.0 + h / 2.0])
            })
            .collect();
        let named: [(&str, [f64; 4]); 5] = [
            ("10x10 deg, 0-10E 40-50N", [0.0, 40.0, 10.0, 50.0]),
            ("2x2 deg, 7-9E 45-47N", [7.0, 45.0, 9.0, 47.0]),
            ("0.5x0.5 deg Alps, 8-8.5E 46-46.5N", [8.0, 46.0, 8.5, 46.5]),
            ("0.1x0.1 deg Alps, 8-8.1E 46-46.1N", [8.0, 46.0, 8.1, 46.1]),
            ("2x2 deg Norway coast, 5-7E 61-63N", [5.0, 61.0, 7.0, 63.0]),
        ];
        boxes.extend(named.map(|(n, b)| (n.to_string(), b)));
        for (name, [lo0, la0, lo1, la1]) in boxes {
            let mut cam = MapCamera::new(Projection::Equirectangular);
            cam.fit(&[(lo0, la0), (lo1, la1)], rect);
            let t = Instant::now();
            let plan = plan_detail(&loc, full, &cam, rect, 1.0, &overview).tiles;
            println!("{name}: plan {plan:?} in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            for spec in plan {
                let req = tile_request(&var, &axes, &vec![0; var.shape.len()], &spec);
                let t = Instant::now();
                let mut h = service.request(&nc, req.clone());
                let first = wait_take(&mut h);
                let first_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let _ = nc.read_slice(&req).unwrap();
                let warm_ms = t.elapsed().as_secs_f64() * 1e3;
                println!(
                    "  tile {spec:?} shape {:?}: first touch {first_ms:.1} ms, warm {warm_ms:.1} ms",
                    first.shape
                );
            }
        }
    }

    fn wait_take(h: &mut torcdf_core::RequestHandle) -> Arc<Slice> {
        loop {
            if let Some(r) = h.try_take() {
                return r.unwrap();
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn failed_detail_shows_a_message_until_a_new_plan() {
        let geo = Geo::Regular { lat_dim: 0, lon_dim: 1, lat_var: "/lat".into(), lon_var: "/lon".into() };
        let mut m = MapState::new(&geo);
        let a = TileSpec { start: [0, 0], count: [4, 4], stride: [1, 1] };
        let b = TileSpec { start: [10, 10], count: [4, 4], stride: [1, 1] };
        m.set_plan(vec![a]);
        m.on_detail_result(a, Err(torcdf_core::CoreError::Unsupported("boom".into())), 1.0);
        assert!(m.detail_problem.as_deref().is_some_and(|p| p.starts_with("Detail could not")));
        assert!(m.plan.is_empty() && m.arrived.is_empty());
        // A later plan starts clean; its tile arriving keeps no message.
        m.set_plan(vec![b]);
        assert!(m.detail_problem.is_none());
        m.on_detail_result(b, Ok(Arc::new(filled(&b, 1.0))), 1.0);
        assert!(m.detail_problem.is_none() && m.arrived.len() == 1);
        // An empty plan after a failure clears the message too.
        m.on_detail_result(b, Err(torcdf_core::CoreError::Unsupported("x".into())), 1.0);
        assert!(m.detail_problem.is_some());
        m.set_plan(Vec::new());
        assert!(m.detail_problem.is_none());
    }

    #[test]
    fn camera_must_rest_before_detail_is_due() {
        let ctx = egui::Context::default();
        let geo = Geo::Regular {
            lat_dim: 0,
            lon_dim: 1,
            lat_var: "/lat".into(),
            lon_var: "/lon".into(),
        };
        let mut m = MapState::new(&geo);
        assert!(!m.detail_due(&ctx, rect()), "first frame starts the clock");
        std::thread::sleep(DETAIL_DELAY + Duration::from_millis(20));
        assert!(m.detail_due(&ctx, rect()));
        assert!(!m.detail_due(&ctx, rect()), "due once per rest");
        m.camera.pan(vec2(5.0, 0.0));
        std::thread::sleep(DETAIL_DELAY + Duration::from_millis(20));
        assert!(!m.detail_due(&ctx, rect()), "moving restarts the clock");
        std::thread::sleep(DETAIL_DELAY + Duration::from_millis(20));
        assert!(m.detail_due(&ctx, rect()));
    }

    #[test]
    fn small_pan_keeps_the_detail_plan() {
        let geo = Geo::Regular { lat_dim: 0, lon_dim: 1, lat_var: "/lat".into(), lon_var: "/lon".into() };
        let mut m = MapState::new(&geo);
        let (ny, nx) = (43200, 86400);
        let lat = (0..ny).map(|i| -90.0 + i as f64 / 240.0).collect();
        let lon = (0..nx).map(|i| -180.0 + i as f64 / 240.0).collect();
        m.locator = CellLocator::regular(lat, lon);
        let (full, ov) = ([ny, nx], TileSpec::overview(ny, nx));
        let r = Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 800.0));
        m.camera = MapCamera::new(Projection::Equirectangular);
        m.camera.fit(&[(5.0, 40.0), (15.0, 50.0)], r);
        m.plan_detail_for(full, r, 1.0, &ov);
        let first = m.take_wanted().expect("zoomed in: detail wanted");
        assert_eq!(first.len(), 1);
        m.set_plan(first.clone());
        // Small pan: the held tile covers the view, nothing is re-read.
        m.camera.pan(vec2(20.0, -15.0));
        m.plan_detail_for(full, r, 1.0, &ov);
        assert_eq!(m.take_wanted(), None);
        // Pan beyond the held tile: a new read.
        m.camera.pan(vec2(300.0, 0.0));
        m.plan_detail_for(full, r, 1.0, &ov);
        let moved = m.take_wanted().expect("pan beyond the tile re-plans");
        assert_eq!(moved[0].stride, first[0].stride);
        m.set_plan(moved.clone());
        // Zoom in: a finer stride is needed.
        m.camera.zoom_at(r.center(), 3.0, r);
        m.plan_detail_for(full, r, 1.0, &ov);
        let zoomed = m.take_wanted().expect("zoom in re-plans");
        assert!(zoomed[0].stride[0] < moved[0].stride[0], "{zoomed:?} vs {moved:?}");
    }

    #[test]
    fn unusable_coords_are_reported() {
        let ctx = egui::Context::default();
        let geo = Geo::Regular {
            lat_dim: 0,
            lon_dim: 1,
            lat_var: "/lat".into(),
            lon_var: "/lon".into(),
        };
        let mut m = MapState::new(&geo);
        // Coordinates that could not be read.
        assert!(m.start_locator(&ctx, &geo, &Coords::default(), &[3, 4]).is_err());
        // Latitudes that are not monotonic.
        let coords = Coords {
            dims: vec![Some(vec![0.0, 10.0, 5.0]), Some(vec![0.0, 1.0, 2.0, 3.0])],
            curv: None,
        };
        m.start_locator(&ctx, &geo, &coords, &[3, 4]).unwrap();
        let mut why = None;
        wait("locator job", || {
            why = m.poll_locator();
            why.is_some()
        });
        assert!(m.locator.is_none());
    }

    #[test]
    fn bad_coords_fall_back_to_image() {
        let (mut v, _ctx, _service, _dir) = open_view(fixtures::write_classic, "/t");
        v.fall_back_to_image("its coordinates are unusable");
        assert_eq!(v.kind, ViewKind::Image);
        assert!(v.map.is_none() && v.image.upload_needed);
        assert!(v.notice.as_deref().is_some_and(|n| n.contains("unusable")));
        assert!(!v.kinds().contains(&ViewKind::Map));
    }
}
