//! The selected variable: its view kind, slider positions, coordinates and
//! the slice on display, with the background jobs that fill them in.

use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use torcdf_core::cf::{Geo, TimeUnits, detect_axes, detect_geo};
use torcdf_core::{
    CoreError, DType, NcFile, RequestHandle, Slice, SliceRequest, SliceService, VarInfo,
};

use crate::app::Job;
use crate::render::colormap::Colormap;
use crate::render::image::ImageLayer;
use crate::ui::{inspector, plot_area};
use crate::view::defaults::{
    AxisChoice, ViewKind, auto_range, build_request, choose_axes, choose_view,
};

/// Slices up to this many values get their statistics on the UI thread
/// (well under a millisecond); larger ones on a helper thread.
const INLINE_STATS_MAX: usize = 1 << 16;
/// Cells shown by the table view.
pub const TABLE_MAX_CELLS: usize = 10_000;
/// Columns of a 2-D table (the rest of the cell budget goes to rows).
const TABLE_MAX_COLS: usize = 100;
/// Points in a line plot; longer lines are thinned by the core.
const LINE_MAX_POINTS: usize = 100_000;
/// Largest curvilinear grid (cells) whose 2-D lat/lon are read for hover.
const CURV_MAX_CELLS: usize = 25_000_000;

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

/// Coordinate values of a variable, read off the UI thread.
#[derive(Default)]
pub struct Coords {
    /// Per dimension of the variable: its 1-D coordinate values.
    pub dims: Vec<Option<Vec<f64>>>,
    /// Curvilinear 2-D latitude and longitude, row-major over (y, x).
    pub curv: Option<(Vec<f64>, Vec<f64>)>,
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
}

/// The value under the cursor, with original (unstrided) indices on the
/// plotted y and x dimensions.
#[derive(Debug, Clone, Copy)]
pub struct Hover {
    pub value: f32,
    pub iy: usize,
    pub ix: usize,
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
    pub var: VarInfo,
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
    pub hover: Option<Hover>,
}

impl VarView {
    pub(crate) fn new(ctx: &egui::Context, file_id: u64, nc: Arc<NcFile>, var: VarInfo) -> VarView {
        let ds = &nc.meta;
        let kind = choose_view(ds, &var);
        let geo = detect_geo(ds, &var);
        let time_dim = detect_axes(ds, &var).time;
        let coord_paths = dim_coord_paths(&nc, &var, &geo);
        // Unknown units or calendar: no decoding, raw values are shown.
        let time_units = time_dim
            .and_then(|d| coord_paths[d].as_deref())
            .and_then(|p| ds.var(p))
            .and_then(|c| {
                let attr = |n: &str| {
                    c.attrs.iter().find(|a| a.name == n).and_then(|a| a.value.as_str())
                };
                TimeUnits::parse(attr("units")?, attr("calendar"))
            });
        let curv = match &geo {
            Geo::Curvilinear { y_dim, x_dim, lat_var, lon_var }
                if var.shape[*y_dim].saturating_mul(var.shape[*x_dim]) <= CURV_MAX_CELLS =>
            {
                Some((lat_var.clone(), lon_var.clone()))
            }
            _ => None,
        };
        let job_nc = nc.clone();
        let coords_job = Some(Job::spawn(ctx, move || load_coords(&job_nc, &coord_paths, curv)));
        VarView {
            file_id,
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
            hover: None,
        }
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
            (ViewKind::Image | ViewKind::Table, Some(a), _) => vec![a.y, a.x],
            (_, _, Some(d)) => vec![d],
            _ => vec![],
        }
    }

    pub fn slider_dims(&self) -> Vec<usize> {
        let plotted = self.plotted_dims();
        (0..self.var.shape.len())
            .filter(|d| !plotted.contains(d) && self.var.shape[*d] > 1)
            .collect()
    }

    /// The hyperslab the current view needs.
    fn request(&self) -> SliceRequest {
        let v = &self.var;
        if let (ViewKind::Image, Some(a), false) = (self.kind, &self.axes, self.is_text()) {
            return build_request(v, a, &self.slider_pos);
        }
        let n = v.shape.len();
        let mut r = SliceRequest {
            var: v.path.clone(),
            start: self.slider_pos.clone(),
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
            true
        } else {
            false
        };
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

    /// Draw row index increasing upwards unless the y coordinate decreases
    /// (north up for both south-to-north and north-to-south grids).
    pub fn flip_y(&self) -> bool {
        let Some(a) = self.axes else { return true };
        if let (Geo::Curvilinear { .. }, Some((lat, _))) = (&self.geo, &self.coords.curv) {
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
    /// lon/lat spacing ratio on regular lat/lon grids.
    pub fn cell_aspect(&self) -> f32 {
        let (Some(a), Some(s)) = (self.axes, &self.shown) else { return 1.0 };
        let spacing = |d: usize| -> f64 {
            if !matches!(self.geo, Geo::Regular { .. }) {
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

/// Runs on a helper thread. Coordinates that cannot be read are left out.
fn load_coords(nc: &NcFile, paths: &[Option<String>], curv: Option<(String, String)>) -> Coords {
    // A coordinate this long would risk an allocation abort; labels then fall
    // back to indices.
    const MAX_COORD_LEN: usize = 10_000_000;
    let read = |p: &str| {
        let len: usize = nc.meta.var(p)?.shape.iter().product();
        if len > MAX_COORD_LEN { None } else { nc.read_coord(p).ok() }
    };
    let dims = paths.iter().map(|p| p.as_deref().and_then(read)).collect();
    let curv = curv.and_then(|(lat, lon)| Some((read(&lat)?, read(&lon)?)));
    Coords { dims, curv }
}
