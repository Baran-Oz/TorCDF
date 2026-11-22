//! Application state and wiring: open files, the selection, and every
//! background job. The UI thread never touches a file: opens, slice reads,
//! coordinate reads, text reads and large-slice statistics all run on other
//! threads and report back through channels, then wake the UI.

use std::io::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::time::Instant;

use eframe::egui;
use torcdf_core::cf::{Geo, TimeUnits, detect_axes, detect_geo};
use torcdf_core::{
    Attr, CoreError, DEFAULT_WORKERS, DType, NcFile, RequestHandle, Slice, SliceRequest,
    SliceService, VarInfo, default_cache_budget_bytes,
};

use crate::render::colormap::Colormap;
use crate::render::image::ImageLayer;
use crate::ui::{self, inspector, plot_area, tree};
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
/// Frames slower than this are logged (UI-thread stalls).
const SLOW_FRAME_MS: f64 = 30.0;

/// A result produced on a helper thread.
pub struct Job<T> {
    rx: Receiver<T>,
}

impl<T: Send + 'static> Job<T> {
    /// Runs `f` on a new thread; the UI is woken when it finishes.
    pub fn spawn(ctx: &egui::Context, f: impl FnOnce() -> T + Send + 'static) -> Job<T> {
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        // If the thread cannot start, `tx` is dropped and `poll` reports it.
        let _ = std::thread::Builder::new().name("torcdf-job".into()).spawn(move || {
            if let Ok(v) = catch_unwind(AssertUnwindSafe(f)) {
                let _ = tx.send(v);
            }
            drop(tx);
            ctx.request_repaint();
        });
        Job { rx }
    }

    /// `Some(Ok)` once finished, `Some(Err)` if the job died, `None` while running.
    pub fn poll(&self) -> Option<Result<T, ()>> {
        match self.rx.try_recv() {
            Ok(v) => Some(Ok(v)),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(())),
        }
    }
}

/// Optional timing log for headless checks: `TORCDF_LOG=<file>` appends
/// timestamped lines (written by a logger thread, not the UI thread); with
/// it, `TORCDF_SELECT=<var path>` selects that variable when a file opens,
/// so first-plot times can be measured without clicks.
struct DebugLog {
    tx: Option<Sender<String>>,
    t0: Instant,
    select: Option<String>,
}

impl DebugLog {
    fn from_env() -> DebugLog {
        let tx = std::env::var_os("TORCDF_LOG").map(|path| {
            let (tx, rx) = channel::<String>();
            std::thread::spawn(move || {
                let file = std::fs::OpenOptions::new().create(true).append(true).open(path);
                if let Ok(mut f) = file {
                    for line in rx {
                        let _ = writeln!(f, "{line}");
                    }
                }
            });
            tx
        });
        let select = std::env::var("TORCDF_SELECT").ok().filter(|_| tx.is_some());
        DebugLog { tx, t0: Instant::now(), select }
    }

    fn log(&mut self, msg: impl std::fmt::Display) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(format!("[{:9.1} ms] {msg}", ms(self.t0)));
        }
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

pub struct OpenFile {
    pub id: u64,
    pub name: String,
    pub nc: Arc<NcFile>,
}

pub struct Opening {
    pub name: String,
    path: PathBuf,
    started: Instant,
    job: Job<Result<NcFile, CoreError>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    File(u64),
    Group(u64, String),
    Var(u64, String),
}

/// What the panels asked for this frame; applied after drawing.
pub enum Action {
    OpenDialog,
    Select(Selection),
    CloseFile(u64),
    DismissError(usize),
    SliderMoved,
    SetKind(ViewKind),
    FitView,
    Exit,
}

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
    fn new(ctx: &egui::Context, file_id: u64, nc: Arc<NcFile>, var: VarInfo) -> VarView {
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
    fn poll(&mut self, ctx: &egui::Context) -> (bool, bool) {
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

fn file_name(p: &Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

pub struct TorApp {
    pub files: Vec<OpenFile>,
    next_id: u64,
    pub opening: Vec<Opening>,
    dialog: Option<Job<Vec<PathBuf>>>,
    pub errors: Vec<String>,
    pub selection: Option<Selection>,
    pub view: Option<VarView>,
    pub display: Display,
    service: SliceService,
    log: DebugLog,
    first_image_logged: bool,
    ctx: egui::Context,
}

impl TorApp {
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> TorApp {
        let ctx = cc.egui_ctx.clone();
        let wake = ctx.clone();
        let service = SliceService::new(
            default_cache_budget_bytes(),
            DEFAULT_WORKERS,
            Arc::new(move || wake.request_repaint()),
        );
        let mut app = TorApp {
            files: Vec::new(),
            next_id: 1,
            opening: Vec::new(),
            dialog: None,
            errors: Vec::new(),
            selection: None,
            view: None,
            display: Display {
                colormap: Colormap::Viridis,
                log: false,
                lock: false,
                range: (0.0, 1.0),
            },
            service,
            log: DebugLog::from_env(),
            first_image_logged: false,
            ctx,
        };
        app.log.log("window created");
        if cc.wgpu_render_state.is_none() {
            app.errors.push("GPU rendering is unavailable, so images cannot be drawn.".into());
        }
        if let Some(p) = path {
            app.open(p);
        }
        app
    }

    /// Opens `path` on a helper thread.
    fn open(&mut self, path: PathBuf) {
        self.log.log(format_args!("open {}", path.display()));
        let p = path.clone();
        let job = Job::spawn(&self.ctx, move || NcFile::open(&p));
        self.opening.push(Opening { name: file_name(&path), path, started: Instant::now(), job });
    }

    fn poll(&mut self) {
        let mut i = 0;
        while i < self.opening.len() {
            let Some(r) = self.opening[i].job.poll() else {
                i += 1;
                continue;
            };
            let o = self.opening.remove(i);
            match r {
                Ok(Ok(nc)) => {
                    self.log.log(format_args!("tree ready {:.1} ms after open", ms(o.started)));
                    let id = self.next_id;
                    self.next_id += 1;
                    self.files.push(OpenFile { id, name: o.name, nc: Arc::new(nc) });
                    if let Some(var) = self.log.select.take() {
                        self.select(Selection::Var(id, var));
                    }
                }
                Ok(Err(e)) => {
                    self.log.log(format_args!("open failed: {e}"));
                    self.errors.push(e.to_string());
                }
                Err(()) => self.errors.push(format!(
                    "cannot open {}: the open failed unexpectedly",
                    o.path.display()
                )),
            }
        }
        if let Some(r) = self.dialog.as_ref().and_then(Job::poll) {
            self.dialog = None;
            for p in r.unwrap_or_default() {
                self.open(p);
            }
        }
        if let Some(v) = &mut self.view {
            let error_before = v.error.is_some();
            let (arrived, coords) = v.poll(&self.ctx);
            if arrived {
                let shape = v.shown.as_ref().map(|s| s.slice.shape.clone()).unwrap_or_default();
                self.log.log(format_args!(
                    "data ready {:.1} ms after request, shape {shape:?}",
                    ms(v.requested_at)
                ));
            }
            if let (true, Some(d)) = (coords, v.time_dim) {
                let n = v.var.shape[d];
                let label = |i: usize| {
                    let c = v.coords.dims.get(d).and_then(|c| c.as_ref()?.get(i).copied());
                    ui::slider_label(i, c, v.time_units.as_ref())
                };
                let (first, last) = (label(0), label(n.saturating_sub(1)));
                self.log.log(format_args!("time axis {}: {first} .. {last} ({n} steps)", v.var.dims[d]));
            }
            if !error_before
                && let Some(e) = &v.error
            {
                self.log.log(format_args!("view error: {e}"));
            }
        }
    }

    fn select(&mut self, sel: Selection) {
        if self.selection.as_ref() == Some(&sel) {
            return;
        }
        if let Selection::Var(id, path) = &sel {
            let var = self.files.iter().find(|f| f.id == *id).and_then(|f| {
                f.nc.meta.var(path).map(|v| (f.nc.clone(), v.clone()))
            });
            // Dropping the old view drops its pending request (cancelling it).
            self.view = None;
            if let Some((nc, var)) = var {
                self.log.log(format_args!("select {}", var.path));
                let mut v = VarView::new(&self.ctx, *id, nc, var);
                v.refresh(&self.ctx, &self.service);
                self.view = Some(v);
                self.display.lock = false;
                self.first_image_logged = false;
            }
        }
        self.selection = Some(sel);
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::OpenDialog => {
                if self.dialog.is_none() {
                    // The native dialog blocks its thread, so it gets its own.
                    self.dialog = Some(Job::spawn(&self.ctx, || {
                        rfd::FileDialog::new()
                            .set_title("Open netCDF / HDF5 files")
                            .add_filter("netCDF / HDF5", &["nc", "nc4", "cdf", "h5", "hdf5", "he5"])
                            .add_filter("All files", &["*"])
                            .pick_files()
                            .unwrap_or_default()
                    }));
                }
            }
            Action::Select(sel) => self.select(sel),
            Action::CloseFile(id) => {
                let (closed, kept): (Vec<_>, Vec<_>) =
                    std::mem::take(&mut self.files).into_iter().partition(|f| f.id == id);
                self.files = kept;
                let view = self.view.take_if(|v| v.file_id == id);
                // The last handle closes the file (I/O, and it may wait for
                // the netCDF lock), so it is dropped off the UI thread.
                std::thread::spawn(move || drop((closed, view)));
                let on_file = match &self.selection {
                    Some(Selection::File(f) | Selection::Group(f, _) | Selection::Var(f, _)) => {
                        *f == id
                    }
                    None => false,
                };
                if on_file {
                    self.selection = None;
                }
            }
            Action::DismissError(i) => {
                if i < self.errors.len() {
                    self.errors.remove(i);
                }
            }
            Action::SliderMoved => {
                if let Some(v) = &mut self.view {
                    v.refresh(&self.ctx, &self.service);
                }
            }
            Action::SetKind(kind) => {
                if let Some(v) = &mut self.view
                    && v.kind != kind
                {
                    v.kind = kind;
                    v.shown = None;
                    v.image = ImageState::default();
                    v.refresh(&self.ctx, &self.service);
                }
            }
            Action::FitView => {
                if let Some(v) = &mut self.view {
                    v.image.view = None;
                    v.image.user_moved = false;
                }
            }
            Action::Exit => self.ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

}

/// Title and attributes of the selected item.
fn selected_attrs<'a>(files: &'a [OpenFile], sel: Option<&Selection>) -> Option<(String, &'a [Attr])> {
    let file = |id: &u64| files.iter().find(|f| f.id == *id);
    match sel? {
        Selection::File(id) => {
            let f = file(id)?;
            Some((format!("{} (global)", f.name), &f.nc.meta.root.attrs[..]))
        }
        Selection::Group(id, path) => {
            let g = find_group(&file(id)?.nc.meta.root, path)?;
            Some((format!("group {}", g.path), &g.attrs[..]))
        }
        Selection::Var(id, path) => {
            let v = file(id)?.nc.meta.var(path)?;
            Some((v.path.clone(), &v.attrs[..]))
        }
    }
}

fn find_group<'a>(g: &'a torcdf_core::Group, path: &str) -> Option<&'a torcdf_core::Group> {
    if g.path == path {
        return Some(g);
    }
    g.groups.iter().find_map(|s| find_group(s, path))
}

impl eframe::App for TorApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let frame_start = Instant::now();
        self.poll();
        let dropped: Vec<PathBuf> = ui.ctx().input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect()
        });
        for p in dropped {
            self.open(p);
        }

        let mut actions = Vec::new();
        egui::Panel::top("menu").show(ui, |ui| ui::menu_bar(ui, self.view.as_ref(), &mut actions));
        if !self.errors.is_empty() {
            egui::Panel::top("errors").show(ui, |ui| ui::error_banners(ui, &self.errors, &mut actions));
        }
        egui::Panel::left("tree").default_size(300.0).show(ui, |ui| {
            tree::show(ui, &self.files, &self.opening, self.selection.as_ref(), &mut actions)
        });
        let attrs = selected_attrs(&self.files, self.selection.as_ref());
        let var_selected = matches!(self.selection, Some(Selection::Var(..)));
        egui::Panel::right("inspector").default_size(300.0).show(ui, |ui| {
            inspector::show(ui, &mut self.display, self.view.as_ref(), var_selected, attrs)
        });
        if let Some(v) = &mut self.view
            && !v.slider_dims().is_empty()
        {
            egui::Panel::bottom("sliders").show(ui, |ui| ui::slider_strip(ui, v, &mut actions));
        }
        let uploaded = egui::CentralPanel::default()
            .show(ui, |ui| {
                plot_area::show(ui, self.view.as_mut(), &self.display, frame.wgpu_render_state())
            })
            .inner;
        ui::drop_overlay(ui.ctx());

        if uploaded && !self.first_image_logged {
            self.first_image_logged = true;
            if let Some(v) = &self.view {
                let range = self.display.effective(v.shown.as_ref().map(|s| &s.stats));
                self.log.log(format_args!(
                    "first image uploaded {:.1} ms after request; grid {:?}, flip_y (north up) {}, range {range:?}",
                    ms(v.requested_at),
                    v.image.grid,
                    v.flip_y(),
                ));
            }
        }
        for a in actions {
            self.apply(a);
        }
        let took = ms(frame_start);
        if took > SLOW_FRAME_MS {
            self.log.log(format_args!("slow frame: {took:.1} ms"));
        }
    }
}
