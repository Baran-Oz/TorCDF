//! The selected variable: its view kind, slider positions, coordinates and
//! the slice on display, with the background jobs that fill them in.

use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use torcdf_core::cf::{Geo, TimeUnits, detect_axes, detect_geo};
use torcdf_core::{
    CoreError, DType, NcFile, RequestHandle, Resolved, Slice, SliceRequest, SliceService, VarInfo,
    VectorPair, layer_path,
};

use crate::app::Job;
use crate::map::tiles::TileSpec;
use crate::render::colormap::Colormap;
use crate::render::image::ImageLayer;
use crate::ui::{inspector, plot_area};
use crate::view::arrow_state::ArrowState;
use crate::view::click::{ClickTarget, PendingClick};
use crate::view::defaults::{
    AxisChoice, ViewKind, auto_range, build_request, choose_axes, choose_view, tile_request,
};
use crate::view::map_state::MapState;

/// Slices up to this many values get their statistics on the UI thread
/// (well under a millisecond); larger ones on a helper thread.
const INLINE_STATS_MAX: usize = 1 << 16;
/// Cells shown by the table view.
pub const TABLE_MAX_CELLS: usize = 10_000;
/// Columns of a 2-D table (the rest of the cell budget goes to rows).
const TABLE_MAX_COLS: usize = 100;
/// Points in a line plot; longer lines are thinned by the core.
const LINE_MAX_POINTS: usize = 100_000;
/// Largest curvilinear grid (cells) whose 2-D lat/lon are read (for the map
/// and hover); larger grids are shown as an image.
const CURV_MAX_CELLS: usize = 10_000_000;

/// Colour settings shared by every variable.
pub struct Display {
    pub colormap: Colormap,
    pub log: bool,
    /// Keep `range` instead of each slice's automatic range.
    pub lock: bool,
    pub range: (f32, f32),
}

impl Display {
    /// The range to colour with: the locked range, or the slice's automatic
    /// range (centred on zero for the diverging and topography maps).
    pub fn effective(&self, stats: Option<&Stats>) -> Option<(f32, f32)> {
        if self.lock {
            return Some(self.range);
        }
        let auto = stats?.auto?;
        Some(match self.colormap {
            Colormap::Diverging => inspector::centre_on_zero(auto),
            Colormap::Topo => inspector::sea_level_range(auto),
            Colormap::Viridis => auto,
        })
    }
}

/// Per-slice statistics, computed once when the slice arrives.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    /// `None` when every value is missing.
    pub auto: Option<(f32, f32)>,
    pub min_positive: Option<f32>,
}

impl Stats {
    fn of(s: &Slice) -> Stats {
        let min_positive =
            s.data.iter().copied().filter(|v| *v > 0.0 && v.is_finite()).reduce(f32::min);
        Stats { auto: auto_range(s), min_positive }
    }
}

/// The slice on display.
pub struct Shown {
    pub slice: Arc<Slice>,
    pub req: SliceRequest,
    pub stats: Stats,
    /// Line view points (missing values skipped), built once.
    pub line: Vec<egui_plot::PlotPoint>,
}

/// Curvilinear 2-D latitude and longitude, shared with the map's jobs.
pub type CurvLatLon = (Arc<Vec<f64>>, Arc<Vec<f64>>);

/// Coordinate values of a variable, read off the UI thread.
#[derive(Default)]
pub struct Coords {
    /// Per dimension of the variable: its 1-D coordinate values.
    pub dims: Vec<Option<Vec<f64>>>,
    /// Curvilinear 2-D latitude and longitude, row-major over (y, x),
    /// unpacked (missing values are NaN). Shared with the map's jobs.
    pub curv: Option<CurvLatLon>,
    /// Vector views: the pair's resolution (or why it failed), set by the
    /// coordinates job; moved to `VarView::resolved` on arrival.
    pub(super) vector: Option<Result<Arc<Resolved>, String>>,
}

/// Image layer and navigation state of the image view.
#[derive(Default)]
pub struct ImageState {
    pub layer: Option<ImageLayer>,
    pub upload_needed: bool,
    pub grid: Option<plot_area::Grid>,
    /// Visible cell-space rect; `None` until fitted to the panel.
    pub view: Option<egui::Rect>,
    /// Panel size the view was fitted to.
    pub fitted_to: egui::Vec2,
    /// The user panned or zoomed, so new slices and resizes keep the view.
    pub user_moved: bool,
    /// Where a Shift+drag index box started.
    pub band: Option<egui::Pos2>,
}

/// The value under the cursor, with original (unstrided) indices on the
/// plotted y and x dimensions.
#[derive(Debug, Clone, Copy)]
pub struct Hover {
    pub value: f32,
    pub iy: usize,
    pub ix: usize,
    /// East and north of the arrow drawn at that cell (after rotation),
    /// when the view has vector parts; `NaN` where one is missing.
    pub arrow: Option<(f32, f32)>,
}

/// Text values of a string or char variable for the table.
pub struct TextTable {
    pub values: Vec<String>,
    pub truncated: bool,
}

/// Everything about the selected variable.
pub struct VarView {
    pub file_id: u64,
    pub nc: Arc<NcFile>,
    /// What the view reads: a variable, a pair, or (layered views) a
    /// variable with a pair's arrows, at the layer path.
    pub var: VarInfo,
    /// Layered views: the coloured variable's own path; `None` otherwise.
    pub base: Option<String>,
    pub kind: ViewKind,
    default_kind: ViewKind,
    pub axes: Option<AxisChoice>,
    pub geo: Geo,
    pub time_dim: Option<usize>,
    pub time_units: Option<TimeUnits>,
    pub slider_pos: Vec<usize>,
    pub coords: Coords,
    coords_job: Option<Job<Coords>>,
    pending: Option<RequestHandle>,
    pub requested_at: Instant,
    stats_job: Option<(Arc<Slice>, SliceRequest, Job<Stats>)>,
    pub shown: Option<Shown>,
    pub text: Option<TextTable>,
    text_job: Option<Job<Result<TextTable, CoreError>>>,
    pub error: Option<String>,
    pub image: ImageState,
    /// Map view state; `None` when the variable cannot be mapped.
    pub map: Option<MapState>,
    /// What a map made later for this view starts with (Settings).
    pub plot_defaults: crate::settings::PlotDefaults,
    /// One-line notice in the plot area (e.g. why the map fell back to the image).
    pub notice: Option<String>,
    /// A passing message (a pop-up click that opened nothing) and when it
    /// goes; shown under `notice`, which it never replaces.
    pub flash: Option<(String, Instant)>,
    pub hover: Option<Hover>,
    /// A click waiting out the double-click time before it opens a pop-up.
    pub pending_click: PendingClick<ClickTarget>,
    /// The vector pair shown, when this view is of one.
    pub vector: Option<Arc<VectorPair>>,
    /// Its staggering, rotation and centre coordinates, once resolved.
    pub resolved: Option<Arc<Resolved>>,
    /// Arrow settings and set; vector views only.
    pub arrows: Option<ArrowState>,
}

impl VarView {
    pub(crate) fn new(ctx: &egui::Context, file_id: u64, nc: Arc<NcFile>, var: VarInfo) -> VarView {
        VarView::build(ctx, file_id, nc, var, None)
    }

    /// Plain variable `x` coloured, with `pair`'s arrows on top: the view
    /// reads the layer path, so slices carry `x`'s values and the pair's
    /// east/north. `x`'s coordinates stay the view's; the coords job also
    /// resolves the pair (for its notes and the rotation switch).
    pub(crate) fn new_layered(
        ctx: &egui::Context,
        file_id: u64,
        nc: Arc<NcFile>,
        x: VarInfo,
        pair: Arc<VectorPair>,
    ) -> VarView {
        let base = x.path.clone();
        let mut v = VarView::build(ctx, file_id, nc, x, Some(pair.clone()));
        v.var.path = layer_path(&base, &pair.path);
        v.base = Some(base);
        v.vector = Some(pair);
        let mut arrows = ArrowState::default();
        arrows.on = true;
        v.arrows = Some(arrows);
        v
    }

    /// The variable's own path: `x`'s for a layered view, else the view's.
    pub fn plain_path(&self) -> &str {
        self.base.as_deref().unwrap_or(&self.var.path)
    }

    /// A view of plain variable `var`; its coords job also resolves `pair`
    /// when there is one.
    fn build(
        ctx: &egui::Context,
        file_id: u64,
        nc: Arc<NcFile>,
        var: VarInfo,
        pair: Option<Arc<VectorPair>>,
    ) -> VarView {
        let ds = &nc.meta;
        let kind = choose_view(ds, &var);
        let geo = detect_geo(ds, &var);
        let mut coord_paths = dim_coord_paths(&nc, &var, &geo);
        let (time_dim, time_units) = time_axis(ds, &var, &mut coord_paths);
        let curv = match &geo {
            Geo::Curvilinear { y_dim, x_dim, lat_var, lon_var }
                if var.shape[*y_dim].saturating_mul(var.shape[*x_dim]) <= CURV_MAX_CELLS =>
            {
                Some(CurvPaths {
                    lat: lat_var.clone(),
                    lon: lon_var.clone(),
                    y_dim: var.dims[*y_dim].clone(),
                    x_dim: var.dims[*x_dim].clone(),
                })
            }
            _ => None,
        };
        let mut kind = kind;
        let mut notice = None;
        if kind == ViewKind::Map && matches!(geo, Geo::Curvilinear { .. }) && curv.is_none() {
            kind = ViewKind::Image;
            notice = Some(map_notice(&format!(
                "its 2-D latitude/longitude has more than {} million cells",
                CURV_MAX_CELLS / 1_000_000
            )));
        }
        let map = (kind == ViewKind::Map).then(|| MapState::new(&geo));
        let job_nc = nc.clone();
        let coords_job = Some(Job::spawn(ctx, move || {
            let mut coords = load_coords(&job_nc, &coord_paths, curv);
            if let Some(p) = pair {
                coords.vector = Some(job_nc.resolve_vector(&p).map_err(|e| e.to_string()));
            }
            coords
        }));
        VarView {
            file_id,
            base: None,
            plot_defaults: Default::default(),
            axes: choose_axes(ds, &var),
            slider_pos: vec![0; var.shape.len()],
            nc,
            var,
            kind,
            default_kind: kind,
            geo,
            time_dim,
            time_units,
            coords: Coords::default(),
            coords_job,
            pending: None,
            requested_at: Instant::now(),
            stats_job: None,
            shown: None,
            text: None,
            text_job: None,
            error: None,
            image: ImageState::default(),
            map,
            notice,
            flash: None,
            hover: None,
            pending_click: PendingClick::default(),
            vector: None,
            resolved: None,
            arrows: None,
        }
    }

    /// A view of a vector pair: its magnitude on the centre grid. The pair's
    /// resolution (which reads coordinates and angles) runs in the coords job.
    pub(crate) fn new_vector(
        ctx: &egui::Context,
        file_id: u64,
        nc: Arc<NcFile>,
        pair: Arc<VectorPair>,
    ) -> VarView {
        let ds = &nc.meta;
        let var = pair.var_info();
        let [hy, hx] = pair.horiz;
        let regular =
            pair.centre_lat.as_deref().and_then(|p| ds.var(p)).is_some_and(|v| v.shape.len() == 1);
        let geo = geo_of_pair(&pair, regular);
        let mut coord_paths = dim_coord_paths(&nc, &var, &Geo::None);
        coord_paths[hy] = None;
        coord_paths[hx] = None;
        let (time_dim, time_units) = time_axis(ds, &var, &mut coord_paths);
        let mut kind = ViewKind::Map;
        let mut notice = None;
        if pair.centre_lat.is_none() {
            // Nothing says where it is: an image, without a map notice.
            kind = ViewKind::Image;
        } else if !regular && var.shape[hy].saturating_mul(var.shape[hx]) > CURV_MAX_CELLS {
            kind = ViewKind::Image;
            notice = Some(map_notice(&format!(
                "its 2-D latitude/longitude has more than {} million cells",
                CURV_MAX_CELLS / 1_000_000
            )));
        }
        let map = (kind == ViewKind::Map).then(|| MapState::new(&geo));
        let (job_nc, job_pair) = (nc.clone(), pair.clone());
        let coords_job =
            Some(Job::spawn(ctx, move || load_vector_coords(&job_nc, &coord_paths, &job_pair)));
        VarView {
            file_id,
            base: None,
            plot_defaults: Default::default(),
            axes: Some(AxisChoice { y: hy, x: hx }),
            slider_pos: vec![0; var.shape.len()],
            nc,
            var,
            kind,
            default_kind: kind,
            geo,
            time_dim,
            time_units,
            coords: Coords::default(),
            coords_job,
            pending: None,
            requested_at: Instant::now(),
            stats_job: None,
            shown: None,
            text: None,
            text_job: None,
            error: None,
            image: ImageState::default(),
            map,
            notice,
            flash: None,
            hover: None,
            pending_click: PendingClick::default(),
            vector: Some(pair),
            resolved: None,
            arrows: Some(ArrowState::default()),
        }
    }

    /// Settings defaults for this view and for any map it makes later.
    pub fn apply_plot_defaults(&mut self, d: &crate::settings::PlotDefaults) {
        if let Some(m) = &mut self.map {
            m.apply_defaults(d);
        }
        self.plot_defaults = d.clone();
    }

    /// The map cannot be drawn: show the image view instead, with a notice.
    pub fn fall_back_to_image(&mut self, why: &str) {
        self.kind = ViewKind::Image;
        self.default_kind = ViewKind::Image;
        self.map = None;
        self.notice = Some(map_notice(why));
        self.hover = None;
        self.image = ImageState { upload_needed: self.shown.is_some(), ..ImageState::default() };
    }

    pub fn is_text(&self) -> bool {
        matches!(self.var.dtype, DType::Str | DType::Char)
    }

    /// A read (or the statistics of one) is still running.
    pub fn loading(&self) -> bool {
        self.pending.is_some() || self.stats_job.is_some() || self.text_job.is_some()
    }

    /// View kinds that make sense for this variable.
    pub fn kinds(&self) -> Vec<ViewKind> {
        match self.default_kind {
            ViewKind::Map => vec![ViewKind::Map, ViewKind::Image, ViewKind::Table],
            ViewKind::Image => vec![ViewKind::Image, ViewKind::Table],
            ViewKind::Line => vec![ViewKind::Line, ViewKind::Table],
            ViewKind::Table => vec![ViewKind::Table],
        }
    }

    /// Dimensions shown by the current view; every other dimension longer
    /// than 1 gets a slider.
    pub fn plotted_dims(&self) -> Vec<usize> {
        let n = self.var.shape.len();
        if self.is_text() {
            // A char array's last dimension is the string length.
            return match (self.var.dtype == DType::Char, n) {
                (_, 0) => vec![],
                (true, 1) => vec![0],
                (true, _) => vec![n - 2, n - 1],
                (false, _) => vec![n - 1],
            };
        }
        let line_dim = self.var.shape.iter().position(|&len| len > 1);
        match (self.kind, self.axes, line_dim) {
            (ViewKind::Map | ViewKind::Image | ViewKind::Table, Some(a), _) => vec![a.y, a.x],
            (_, _, Some(d)) => vec![d],
            _ => vec![],
        }
    }

    /// The user may choose the plotted dimensions: a plain numeric variable
    /// (no arrows) in the image or table view, with at least two dimensions
    /// longer than 1. Maps keep lat/lon; vector views their pair's axes.
    pub fn axes_choosable(&self) -> bool {
        self.vector.is_none()
            && self.var.dtype.is_numeric()
            && matches!(self.kind, ViewKind::Image | ViewKind::Table)
            && self.var.shape.iter().filter(|&&n| n > 1).count() >= 2
    }

    /// Puts `axes` on the plotted y and x and reads the slice for them.
    /// Refused when the axes cannot be chosen, when y == x or when either is
    /// not a dimension longer than 1. Sliders that stay keep their
    /// positions; a dimension leaving an axis gets a slider at 0.
    pub fn set_axes(&mut self, ctx: &egui::Context, service: &SliceService, axes: AxisChoice) {
        let long = |d: usize| self.var.shape.get(d).is_some_and(|&n| n > 1);
        if !self.axes_choosable() || axes.y == axes.x || !long(axes.y) || !long(axes.x) || self.axes == Some(axes) {
            return;
        }
        self.move_axes(axes);
        // Every dimension's coordinates are already loaded (see
        // `dim_coord_paths`), so the new axes have theirs.
        self.refresh(ctx, service);
    }

    /// Back on the map: the geographic dimensions become the axes again (the
    /// caller refreshes).
    pub fn restore_geo_axes(&mut self) {
        let axes = match self.geo {
            Geo::Regular { lat_dim, lon_dim, .. } => AxisChoice { y: lat_dim, x: lon_dim },
            Geo::Curvilinear { y_dim, x_dim, .. } => AxisChoice { y: y_dim, x: x_dim },
            Geo::None => return,
        };
        if self.axes.is_some() && self.axes != Some(axes) {
            self.move_axes(axes);
        }
    }

    /// The plotted axes are the grid's latitude and longitude (or its
    /// curvilinear y and x), in that order.
    pub fn on_geo_axes(&self) -> bool {
        let Some(a) = self.axes else { return false };
        match self.geo {
            Geo::Regular { lat_dim, lon_dim, .. } => (a.y, a.x) == (lat_dim, lon_dim),
            Geo::Curvilinear { y_dim, x_dim, .. } => (a.y, a.x) == (y_dim, x_dim),
            Geo::None => false,
        }
    }

    /// Sets the axes, moving sliders: a dimension leaving an axis gets a
    /// slider at 0, one moving onto an axis loses its own. The slice and
    /// image of the old axes are dropped (the image is fitted afresh).
    fn move_axes(&mut self, axes: AxisChoice) {
        let old = self.axes.map_or_else(Vec::new, |a| vec![a.y, a.x]);
        for (d, pos) in self.slider_pos.iter_mut().enumerate() {
            if old.contains(&d) || d == axes.y || d == axes.x {
                *pos = 0;
            }
        }
        self.axes = Some(axes);
        self.shown = None;
        self.stats_job = None;
        self.hover = None;
        self.pending_click.cancel();
        self.image = ImageState::default();
    }

    /// Shows `text` as a passing message for `for_how_long` from now.
    pub fn set_flash(&mut self, text: String, for_how_long: std::time::Duration) {
        self.flash = Some((text, Instant::now() + for_how_long));
    }

    /// Drops the passing message once its time is up at `now`; else the
    /// time it has left (`None` when there is none).
    pub fn expire_flash(&mut self, now: Instant) -> Option<std::time::Duration> {
        let (_, until) = self.flash.as_ref()?;
        let left = until.saturating_duration_since(now);
        if left.is_zero() {
            self.flash = None;
            return None;
        }
        Some(left)
    }

    pub fn slider_dims(&self) -> Vec<usize> {
        let plotted = self.plotted_dims();
        (0..self.var.shape.len())
            .filter(|d| !plotted.contains(d) && self.var.shape[*d] > 1)
            .collect()
    }

    /// The hyperslab the current view needs.
    fn request(&self) -> SliceRequest {
        self.request_at(&self.slider_pos)
    }

    /// Reads the view needs to show slider positions `pos` (the playback
    /// hook): the main slice, plus the map's current detail plan. Empty when
    /// the view reads nothing through the slice service.
    pub fn requests_at(&self, pos: &[usize]) -> Vec<SliceRequest> {
        let unreadable = matches!(self.var.dtype, DType::Other(_)) || self.var.shape.contains(&0);
        if unreadable || self.is_text() || pos.len() != self.var.shape.len() {
            return Vec::new();
        }
        let mut reqs = vec![self.request_at(pos)];
        if let (ViewKind::Map, Some(map), Some(a)) = (self.kind, &self.map, &self.axes) {
            reqs.extend(map.plan().iter().map(|s| tile_request(&self.var, a, pos, s)));
        }
        reqs
    }

    /// The slice on show is the one this view reads at slider positions `pos`.
    pub fn shows_position(&self, pos: &[usize]) -> bool {
        pos.len() == self.var.shape.len()
            && self.shown.as_ref().is_some_and(|s| s.req == self.request_at(pos))
    }

    /// The hyperslab the current view needs at slider positions `pos`.
    fn request_at(&self, pos: &[usize]) -> SliceRequest {
        let v = &self.var;
        match (self.kind, &self.axes, self.is_text()) {
            (ViewKind::Image, Some(a), false) => return build_request(v, a, pos),
            (ViewKind::Map, Some(a), false) => {
                let spec = TileSpec::overview(v.shape[a.y], v.shape[a.x]);
                return tile_request(v, a, pos, &spec);
            }
            _ => {}
        }
        let n = v.shape.len();
        let mut r = SliceRequest {
            var: v.path.clone(),
            start: pos.to_vec(),
            count: vec![1; n],
            stride: vec![1; n],
        };
        let plotted = self.plotted_dims();
        for &d in &plotted {
            r.start[d] = 0;
        }
        match plotted[..] {
            [d] if self.kind == ViewKind::Line && !self.is_text() => {
                r.count[d] = v.shape[d];
                r.stride[d] = v.shape[d].div_ceil(LINE_MAX_POINTS).max(1);
            }
            [d] => r.count[d] = v.shape[d].min(TABLE_MAX_CELLS),
            [y, x] if v.dtype == DType::Char => {
                r.count[x] = v.shape[x];
                r.count[y] = v.shape[y].min(TABLE_MAX_CELLS);
            }
            [y, x] => {
                r.count[x] = v.shape[x].min(TABLE_MAX_COLS);
                r.count[y] = v.shape[y].min(TABLE_MAX_CELLS / r.count[x].max(1));
            }
            _ => {}
        }
        r
    }

    /// Asks for the data of the current view. The superseded request is
    /// dropped first, which cancels it if no worker has started it.
    pub fn refresh(&mut self, ctx: &egui::Context, service: &SliceService) {
        self.refresh_with(ctx, service, &[]);
    }

    /// [`Self::refresh`], taking any read in `ready` (a played frame, held
    /// by the playback) instead of asking the service for it, so a frame
    /// read ahead is shown even if the cache has evicted it since.
    pub fn refresh_with(
        &mut self,
        ctx: &egui::Context,
        service: &SliceService,
        ready: &[(SliceRequest, Arc<Slice>)],
    ) {
        self.pending = None;
        self.stats_job = None;
        self.text_job = None;
        self.error = None;
        self.hover = None;
        self.requested_at = Instant::now();
        if let DType::Other(t) = &self.var.dtype {
            self.error = Some(format!("Values of {t} type cannot be shown; see its attributes."));
            return;
        }
        if self.var.shape.contains(&0) {
            self.error = Some("This variable has no values (a dimension has length 0).".into());
            return;
        }
        let r = self.request();
        if self.is_text() {
            let truncated = self.plotted_dims().iter().any(|&d| r.count[d] < self.var.shape[d]);
            let nc = self.nc.clone();
            self.text_job = Some(Job::spawn(ctx, move || {
                let values = nc.read_strings(&r.var, &r.start, &r.count)?;
                Ok(TextTable { values, truncated })
            }));
            return;
        }
        if let Some(s) = held(ready, &r) {
            self.on_slice(ctx, r, Ok(s));
        } else {
            let mut handle = service.request(&self.nc, r);
            // A cache hit is ready at once and will not call on_ready.
            match handle.try_take() {
                Some(result) => {
                    let req = handle.req().clone();
                    self.on_slice(ctx, req, result);
                }
                // Keeping the handle keeps the request alive.
                None => self.pending = Some(handle),
            }
        }
        // A slider move re-reads the detail tiles for the new position.
        self.request_detail(service, ready);
    }

    /// Requests the detail plan the map just made, if any (the map is shown
    /// and the camera has rested). Reads run on the service's workers.
    pub fn start_detail(&mut self, service: &SliceService) {
        let Some(map) = self.map.as_mut() else { return };
        let Some(plan) = map.take_wanted() else { return };
        map.set_plan(plan);
        self.request_detail(service, &[]);
    }

    /// (Re)requests every tile of the map's current detail plan at the
    /// current slider position (taking those in `ready` as read). Tiles on
    /// show stay until the new ones arrive.
    fn request_detail(&mut self, service: &SliceService, ready: &[(SliceRequest, Arc<Slice>)]) {
        let (Some(map), Some(axes)) = (self.map.as_mut(), self.axes) else { return };
        if self.kind != ViewKind::Map {
            return;
        }
        let plan = map.restart_detail();
        map.detail_requested_now();
        for spec in plan {
            let req = tile_request(&self.var, &axes, &self.slider_pos, &spec);
            match held(ready, &req) {
                Some(s) => {
                    map.accept_detail(spec, s);
                }
                None => map.track_detail(spec, service.request(&self.nc, req)),
            }
        }
    }

    /// Takes a finished read; true if it went on show at once.
    fn on_slice(
        &mut self,
        ctx: &egui::Context,
        req: SliceRequest,
        r: Result<Arc<Slice>, CoreError>,
    ) -> bool {
        match r {
            Err(e) => {
                self.error = Some(e.to_string());
                false
            }
            Ok(slice) if slice.data.len() <= INLINE_STATS_MAX => {
                let stats = Stats::of(&slice);
                self.show(slice, req, stats);
                true
            }
            Ok(slice) => {
                let s = slice.clone();
                self.stats_job = Some((slice, req, Job::spawn(ctx, move || Stats::of(&s))));
                false
            }
        }
    }

    fn show(&mut self, slice: Arc<Slice>, req: SliceRequest, stats: Stats) {
        let line = match (self.kind, &self.plotted_dims()[..]) {
            (ViewKind::Line, &[d]) => {
                let xs = self.coords.dims.get(d).and_then(|c| c.as_ref());
                let point = |(i, &v): (usize, &f32)| {
                    let idx = req.start[d] + i * req.stride[d];
                    let x = xs.and_then(|c| c.get(idx).copied()).unwrap_or(idx as f64);
                    egui_plot::PlotPoint::new(x, v)
                };
                slice.data.iter().enumerate().filter(|(_, v)| v.is_finite()).map(point).collect()
            }
            _ => Vec::new(),
        };
        self.shown = Some(Shown { slice, req, stats, line });
        self.image.upload_needed = true;
    }

    /// Collects finished background work. Returns (new data on show,
    /// coordinates just arrived).
    pub(crate) fn poll(&mut self, ctx: &egui::Context) -> (bool, bool) {
        let mut arrived = false;
        let coords_arrived = if let Some(r) = self.coords_job.as_ref().and_then(Job::poll) {
            self.coords = r.unwrap_or_default();
            self.coords_job = None;
            let resolve_failed = self.take_resolution();
            // Line x values come from the coordinates.
            if self.kind == ViewKind::Line
                && let Some(s) = self.shown.take()
            {
                self.show(s.slice, s.req, s.stats);
            }
            // Orientation and aspect may have changed.
            if !self.image.user_moved {
                self.image.view = None;
            }
            // A pair that cannot be resolved is shown as an image, with why.
            if let Some(e) = resolve_failed {
                self.fall_back_to_image("");
                self.notice = Some(e);
            }
            // The map's cell index is built off the UI thread.
            if let Some(m) = &mut self.map
                && let Err(why) = m.start_locator(ctx, &self.geo, &self.coords, &self.var.shape)
            {
                self.fall_back_to_image(&why);
            }
            true
        } else {
            false
        };
        if let Some(why) = self.map.as_mut().and_then(MapState::poll_locator) {
            self.fall_back_to_image(&why);
        }
        if let Some(m) = &mut self.map {
            m.poll_detail();
        }
        if let Some(r) = self.text_job.as_ref().and_then(Job::poll) {
            self.text_job = None;
            match r {
                Ok(Ok(t)) => self.text = Some(t),
                Ok(Err(e)) => self.error = Some(e.to_string()),
                Err(()) => self.error = Some("The read failed unexpectedly.".into()),
            }
            arrived = true;
        }
        if let Some(h) = &mut self.pending
            && let Some(r) = h.try_take()
        {
            let req = h.req().clone();
            self.pending = None;
            arrived |= self.on_slice(ctx, req, r);
        }
        if let Some((_, _, job)) = &self.stats_job
            && let Some(r) = job.poll()
        {
            let (slice, req, _) = self.stats_job.take().expect("checked above");
            match r {
                Ok(stats) => {
                    self.show(slice, req, stats);
                    arrived = true;
                }
                Err(()) => self.error = Some("Range statistics failed unexpectedly.".into()),
            }
        }
        (arrived, coords_arrived)
    }

    /// Moves a vector view's resolution out of the freshly arrived coords.
    /// Returns the error to show when the pair could not be resolved. The
    /// grid kind follows the resolved centre coordinates.
    fn take_resolution(&mut self) -> Option<String> {
        let pair = self.vector.as_ref()?;
        let layered = self.base.is_some();
        match self.coords.vector.take()? {
            // A layered view keeps its variable's grid and view; only the
            // arrows' notes are missing.
            Err(e) if layered => {
                self.notice.get_or_insert(e);
                None
            }
            Ok(r) => {
                let regular = matches!(self.geo, Geo::Regular { .. });
                if !layered && r.regular != regular && self.map.is_some() {
                    self.geo = geo_of_pair(pair, r.regular);
                    let mut m = MapState::new(&self.geo);
                    m.apply_defaults(&self.plot_defaults);
                    self.map = Some(m);
                }
                self.resolved = Some(r);
                None
            }
            Err(e) => Some(e),
        }
    }

    /// Draw row index increasing upwards unless the y coordinate decreases
    /// (north up for both south-to-north and north-to-south grids).
    pub fn flip_y(&self) -> bool {
        let Some(a) = self.axes else { return true };
        if let (Geo::Curvilinear { .. }, Some((lat, _)), true) = (&self.geo, &self.coords.curv, self.on_geo_axes()) {
            let (ny, nx) = (self.var.shape[a.y], self.var.shape[a.x]);
            let mid = nx / 2;
            return match (lat.get(mid), lat.get((ny - 1) * nx + mid)) {
                // Deliberate: NaN endpoints count as "not decreasing" (north up).
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                (Some(first), Some(last)) => !(last < first),
                _ => true,
            };
        }
        match self.coords.dims.get(a.y).and_then(|c| c.as_ref()) {
            // Deliberate: NaN endpoints count as "not decreasing" (north up).
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            Some(c) if c.len() >= 2 => !(c[c.len() - 1] < c[0]),
            _ => true,
        }
    }

    /// Screen width/height of one image cell: the stride ratio, times the
    /// lon/lat spacing ratio when a regular grid's lat/lon are the axes.
    pub fn cell_aspect(&self) -> f32 {
        let (Some(a), Some(s)) = (self.axes, &self.shown) else { return 1.0 };
        let spacing = |d: usize| -> f64 {
            if !matches!(self.geo, Geo::Regular { .. }) || !self.on_geo_axes() {
                return 1.0;
            }
            match self.coords.dims.get(d).and_then(|c| c.as_ref()) {
                Some(c) if c.len() >= 2 => {
                    let step = ((c[c.len() - 1] - c[0]) / (c.len() - 1) as f64).abs();
                    if step.is_finite() && step > 0.0 { step } else { 1.0 }
                }
                _ => 1.0,
            }
        };
        let x = s.req.stride[a.x] as f64 * spacing(a.x);
        let y = s.req.stride[a.y] as f64 * spacing(a.y);
        ((x / y) as f32).clamp(1e-3, 1e3)
    }
}

/// Per dimension, the 1-D coordinate variable to read: the regular lat/lon
/// pair on geographic grids, otherwise a numeric variable named like the
/// dimension (in the variable's group, then the root).
fn dim_coord_paths(nc: &NcFile, v: &VarInfo, geo: &Geo) -> Vec<Option<String>> {
    let ds = &nc.meta;
    let group = ds.group_of(&v.path);
    (0..v.dims.len())
        .map(|i| {
            if let Geo::Regular { lat_dim, lon_dim, lat_var, lon_var } = geo {
                if i == *lat_dim {
                    return Some(lat_var.clone());
                }
                if i == *lon_dim {
                    return Some(lon_var.clone());
                }
            }
            let dim = &v.dims[i];
            group
                .vars
                .iter()
                .chain(ds.root.vars.iter())
                .find(|c| c.name == *dim && c.dims.len() == 1 && c.dims[0] == *dim)
                .filter(|c| c.dtype.is_numeric())
                .map(|c| c.path.clone())
        })
        .collect()
}

/// The grid of a pair's centre over its horizontal dimensions.
fn geo_of_pair(pair: &VectorPair, regular: bool) -> Geo {
    let [y, x] = pair.horiz;
    let lat_var = pair.centre_lat.clone().unwrap_or_default();
    let lon_var = pair.centre_lon.clone().unwrap_or_default();
    if regular {
        Geo::Regular { lat_dim: y, lon_dim: x, lat_var, lon_var }
    } else {
        Geo::Curvilinear { y_dim: y, x_dim: x, lat_var, lon_var }
    }
}

/// Runs on a helper thread: the non-horizontal coordinates, then the pair's
/// resolution, whose centre latitude/longitude become the horizontal ones.
fn load_vector_coords(nc: &NcFile, paths: &[Option<String>], pair: &VectorPair) -> Coords {
    let mut coords = load_coords(nc, paths, None);
    let r = nc.resolve_vector(pair).map_err(|e| e.to_string());
    if let Ok(r) = &r {
        let [hy, hx] = pair.horiz;
        if r.regular {
            coords.dims[hy] = Some((*r.lat).clone());
            coords.dims[hx] = Some((*r.lon).clone());
        } else if !r.lat.is_empty() {
            coords.curv = Some((r.lat.clone(), r.lon.clone()));
        }
    }
    coords.vector = Some(r);
    coords
}

/// The slice read for `req` among `ready`, if there.
fn held(ready: &[(SliceRequest, Arc<Slice>)], req: &SliceRequest) -> Option<Arc<Slice>> {
    ready.iter().find(|(r, _)| r == req).map(|(_, s)| s.clone())
}

/// The time dimension of `var` and its decoded units. The variable giving
/// time values (`/time`, or WRF's `/XTIME` over `Time`) becomes that
/// dimension's coordinate in `coord_paths`. Unknown units or calendar: no
/// decoding, raw values are shown.
fn time_axis(
    ds: &torcdf_core::Dataset,
    var: &VarInfo,
    coord_paths: &mut [Option<String>],
) -> (Option<usize>, Option<TimeUnits>) {
    let axes = detect_axes(ds, var);
    let Some(d) = axes.time else { return (None, None) };
    let Some(c) = axes.time_var.as_deref().and_then(|p| ds.var(p)).filter(|c| c.dtype.is_numeric())
    else {
        return (Some(d), None);
    };
    if let Some(slot) = coord_paths.get_mut(d) {
        *slot = Some(c.path.clone());
    }
    let attr = |n: &str| c.attrs.iter().find(|a| a.name == n).and_then(|a| a.value.as_str());
    (Some(d), attr("units").and_then(|u| TimeUnits::parse(u, attr("calendar"))))
}

/// "Map unavailable" notice for `why` (a clause about the variable).
fn map_notice(why: &str) -> String {
    format!("Shown as an image: this variable cannot be drawn as a map because {why}.")
}

/// The 2-D latitude/longitude of a curvilinear variable and its (y, x) dims.
struct CurvPaths {
    lat: String,
    lon: String,
    y_dim: String,
    x_dim: String,
}

/// Runs on a helper thread. Coordinates that cannot be read are left out.
fn load_coords(nc: &NcFile, paths: &[Option<String>], curv: Option<CurvPaths>) -> Coords {
    // A coordinate this long would risk an allocation abort; labels then fall
    // back to indices.
    const MAX_COORD_LEN: usize = 10_000_000;
    let read = |p: &str| {
        let len: usize = nc.meta.var(p)?.shape.iter().product();
        if len > MAX_COORD_LEN { None } else { nc.read_coord(p).ok() }
    };
    // 2-D lat/lon are unpacked, so fill values become NaN. Only their (y, x)
    // plane is read: index 0 of any other dims, such as WRF's `Time`.
    let read_2d = |p: &str, c: &CurvPaths| -> Option<Arc<Vec<f64>>> {
        let v = nc.meta.var(p)?;
        let len = |d: &str| v.dims.iter().position(|x| x == d).map(|i| v.shape[i]);
        if len(&c.y_dim)?.saturating_mul(len(&c.x_dim)?) > CURV_MAX_CELLS {
            return None;
        }
        nc.read_horizontal(p, &c.y_dim, &c.x_dim).ok().map(Arc::new)
    };
    let dims = paths.iter().map(|p| p.as_deref().and_then(read)).collect();
    let curv = curv.and_then(|c| Some((read_2d(&c.lat, &c)?, read_2d(&c.lon, &c)?)));
    Coords { dims, curv, vector: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torcdf_core::{Basis, fixtures};

    fn nemo_view() -> (tempfile::TempDir, VarView) {
        let d = tempfile::tempdir().unwrap();
        let nc = Arc::new(fixtures::paired(NcFile::open(&fixtures::vectors::write_nemo_like(d.path())).unwrap()));
        let pair = nc.pairs().into_iter().next().expect("nemo pair detected");
        let v = VarView::new_vector(&egui::Context::default(), 1, nc, pair);
        (d, v)
    }

    fn poll_coords(v: &mut VarView) {
        let ctx = egui::Context::default();
        let t0 = Instant::now();
        loop {
            if v.poll(&ctx).1 {
                return;
            }
            assert!(t0.elapsed().as_secs() < 20, "coords job timed out");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn pair_without_latlon_starts_as_image() {
        let d = tempfile::tempdir().unwrap();
        let nc = Arc::new(fixtures::paired(NcFile::open(&fixtures::vectors::write_noclue(d.path(), false)).unwrap()));
        // The fixture carries 2-D lat/lon; drop them from the pair, as when none is found.
        let mut pair = (*nc.pairs().into_iter().next().expect("u/v pair")).clone();
        pair.centre_lat = None;
        pair.centre_lon = None;
        let pair = Arc::new(pair);
        let v = VarView::new_vector(&egui::Context::default(), 1, nc, pair);
        assert_eq!(v.kind, ViewKind::Image);
        assert!(v.map.is_none());
    }

    #[test]
    fn vector_view_is_map() {
        let (_d, v) = nemo_view();
        assert_eq!(v.kind, ViewKind::Map);
        assert_eq!(v.axes, Some(AxisChoice { y: 1, x: 2 }));
        assert_eq!(v.slider_dims(), vec![0]);
    }

    /// A layered view keeps its variable's coordinates and also resolves
    /// the pair (for its notes and the rotation switch).
    #[test]
    fn layered_view_resolves_pair_on_own_grid() {
        let d = tempfile::tempdir().unwrap();
        let nc = Arc::new(fixtures::paired(NcFile::open(&fixtures::vectors::write_nemo_like(d.path())).unwrap()));
        let pair = nc.pairs().into_iter().next().expect("nemo pair detected");
        let x = nc.meta.var("/thetao").unwrap().clone();
        let mut v = VarView::new_layered(&egui::Context::default(), 1, nc, x, pair.clone());
        assert_eq!(v.var.path, layer_path("/thetao", &pair.path));
        assert_eq!(v.plain_path(), "/thetao");
        let geo_before = format!("{:?}", v.geo);
        poll_coords(&mut v);
        assert!(v.resolved.is_some(), "pair resolved");
        assert_eq!(format!("{:?}", v.geo), geo_before, "the variable's own grid");
        assert_eq!(v.coords.curv.as_ref().expect("thetao's lat/lon").0.len(), 600);
        assert!(v.arrows.as_ref().is_some_and(|a| a.on));
    }

    #[test]
    fn vector_view_resolves() {
        let (_d, mut v) = nemo_view();
        poll_coords(&mut v);
        assert_eq!(v.resolved.as_ref().expect("resolved").basis, Basis::Likely);
        assert_eq!(v.coords.curv.as_ref().expect("curv").0.len(), 600);
    }
}
