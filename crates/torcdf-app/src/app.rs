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
use torcdf_core::{
    Attr, CoreError, DEFAULT_WORKERS, DirConvention, NcFile, PairForm, RotateChoice, SliceService, VarInfo,
    VectorPair, default_cache_budget_bytes, layer_path, pair_by_hand, pair_magdir_by_hand,
};

use crate::settings::{Loaded, Look, PlotDefaults, RecentFormula, Settings, TextSize, WindowGeom};
use crate::ui::shortcuts::{self, Command};
use crate::ui::windows::{Win, Windows};
use crate::ui::inspector::{PanelInput, PanelTab};
use crate::ui::{self, inspector, plot_area, tree};
use crate::view::arrow_layer::{ArrowLayer, ArrowMode, PICK_TWO, candidates};
use crate::view::defaults::{AxisChoice, ViewKind};
use crate::view::playback::{FPS_RANGE, Playback, Tick, frame_bytes};
use crate::view::popup::{Popup, PopupKind, PopupSet};
use crate::view::png_export;
use crate::view::extract::{ExtractForm, RegionText, load_coords};
use crate::view::var_view::{Display, ImageState, VarView};

/// A CSV save: `None` when the dialog was cancelled, else the write result.
type SaveOutcome = Option<Result<(), String>>;

/// A PNG save: `None` when the dialog was cancelled, else the saved file's
/// name or the error.
type PngOutcome = Option<Result<String, String>>;

/// A screenshot asked for and not yet delivered: what to cut out of it.
struct Shot {
    rx: Receiver<Arc<egui::ColorImage>>,
    rect: egui::Rect,
    ppp: f32,
    name: String,
    /// The pop-up being saved; `None` for the plot area.
    popup: Option<u64>,
}

/// How long a pop-up click message (`VarView::flash`) stays in the plot area.
const NOTICE_TIME: std::time::Duration = std::time::Duration::from_secs(4);

/// Frames slower than this are logged (UI-thread stalls).
const SLOW_FRAME_MS: f64 = 20.0;

/// While playback waits for a frame, the UI looks again this often (reads
/// finishing wake it sooner).
const WAIT_RECHECK: std::time::Duration = std::time::Duration::from_millis(50);

/// The dimension to animate by default: the time axis when it has a slider,
/// else the first slider (0 when there is none).
fn default_play_dim(v: &VarView) -> usize {
    let dims = v.slider_dims();
    match v.time_dim {
        Some(t) if dims.contains(&t) => t,
        _ => dims.first().copied().unwrap_or(0),
    }
}

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

    fn enabled(&self) -> bool {
        self.tx.is_some()
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

/// The Calculate window's fields.
pub struct CalcForm {
    pub file: u64,
    pub name: String,
    pub formula: String,
    pub units: String,
    /// The calculated variable being edited.
    pub replacing: Option<String>,
    /// The result's description, or what is wrong.
    pub check: Result<String, String>,
}

/// What the panels asked for this frame; applied after drawing.
pub enum Action {
    OpenDialog,
    Select(Selection),
    CloseFile(u64),
    DismissError(usize),
    SliderMoved,
    SetKind(ViewKind),
    /// Plot tab "Axes": the dimensions on the plotted y and x.
    SetAxes { y: usize, x: usize },
    /// Remove a pair (and its rotation variants) for this session.
    Unpair { file: u64, path: String },
    SetRotate(RotateChoice),
    /// Arrows section: Off, x & y, or magnitude & direction.
    SetArrowMode(ArrowMode),
    /// Arrows section drop-down: slot 0 = x / magnitude, 1 = y / direction.
    PickArrowVar { slot: usize, path: String },
    /// Direction convention of the arrows' magnitude/direction pair.
    SetConvention(DirConvention),
    /// The right panel's tab.
    SetPanelTab(PanelTab),
    FitView,
    /// A click on the map or image: pop up the series or profile there.
    OpenPopup { iy: usize, ix: usize, lonlat: Option<(f64, f64)> },
    ClosePopup(u64),
    SetPopupKind(u64, PopupKind),
    /// A click in a pop-up: move the slider of that dimension (plan P7).
    JumpTo { file: u64, dim_name: String, index: usize },
    SavePopupCsv(u64),
    /// Save a pop-up's point (its whole series) as netCDF.
    SavePopupNetcdf(u64),
    /// Save the plot area with its colour bar as a PNG.
    SaveImage,
    SavePopupPng(u64),
    /// Animate the playback dimension (locks the colour range as shown).
    Play,
    Pause,
    SetFps(f32),
    SetLoop(bool),
    SetPlayDim(usize),
    Exit,
    OpenRecent(PathBuf),
    ShowWindow(Win),
    SetLook(Look),
    SetTextSize(TextSize),
    /// What new plots start with; plots already shown keep theirs.
    SetPlotDefaults(PlotDefaults),
    SetRecentMax(usize),
    SetReopenLast(bool),
    /// Settings back to defaults (recent files and window geometry stay).
    ResetSettings,
    /// Open the Calculate window for `file`; `edit`: a calculated variable to change.
    OpenCalculate { file: u64, edit: Option<String> },
    /// Remove a calculated variable (refused while something uses it).
    DeleteDerived { file: u64, path: String },
    /// A field of the Calculate window changed: check again.
    CalcEdited,
    /// The Calculate window's Create (or Apply when editing).
    CreateDerived,
    CloseCalculate,
    /// Fill the Calculate window from recent formula `i`.
    UseRecentFormula(usize),
    /// Open the Extract window for `file`, ticking `var` (else the variable on screen).
    OpenExtract { file: u64, var: Option<String> },
    /// A field or tick of the Extract window changed: check again.
    ExtractEdited,
    /// Region from the map as shown.
    ExtractUseView,
    ExtractWholeGrid,
    /// Row `row` takes the slider's step (`end`: its To, else its From).
    ExtractFromSlider { row: usize, end: bool },
    SaveExtract,
    CancelExtract,
    CloseExtract,
    /// Shift+drag on the map: the region box (degrees).
    ExtractRegionDrawn { south: f64, north: f64, west: f64, east: f64 },
    /// Shift+drag on an image: (first, last) indices along its plotted y and x.
    ExtractIndexBoxDrawn { y: (usize, usize), x: (usize, usize) },
    /// Shift+drag across a time series of `file`: axis values.
    ExtractTimeDragged { file: u64, from: f64, to: f64 },
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
    pub popups: PopupSet,
    /// The window's arrow slot (the Plot tab's Arrows section).
    pub arrow_layer: Option<ArrowLayer>,
    /// The right panel's tab.
    pub panel_tab: PanelTab,
    /// Pop-up id and its CSV save: `None` when the dialog was cancelled.
    saves: Vec<(u64, Job<SaveOutcome>)>,
    /// The central plot area (colour bar included) as drawn last frame.
    plot_rect: Option<egui::Rect>,
    /// Screenshots requested, waiting to arrive.
    shots: Vec<Shot>,
    /// PNG encodes and saves in progress.
    png_saves: Vec<(Option<u64>, Job<PngOutcome>)>,
    service: SliceService,
    /// The service's cache budget (read-ahead holds at most an eighth).
    cache_budget: usize,
    /// Animation of the current view; dropped with the view.
    pub playback: Option<Playback>,
    /// When a slider last moved and the positions it moved to, until a
    /// slice for those positions is uploaded (for the step log).
    slider_changed_at: Option<(Instant, Vec<usize>)>,
    log: DebugLog,
    first_image_logged: bool,
    ctx: egui::Context,
    /// Preferences, recent files and window geometry (saved between runs).
    pub settings: Settings,
    /// Where `settings` is saved; `None` never saves (tests).
    settings_path: Option<PathBuf>,
    /// A failed save was reported (once per run).
    save_failed: bool,
    /// The window's last un-maximised geometry and whether it is maximised.
    last_geom: Option<WindowGeom>,
    /// Settings, About, licences and shortcut windows.
    pub windows: Windows,
    /// The graphics adapter, for "Copy version info".
    renderer: String,
    /// Scroll the tree to the selection next frame (it moved by keyboard).
    scroll_tree: bool,
    /// The Calculate window, when open.
    pub calc: Option<CalcForm>,
    /// The Extract window, when open (it outlives its file while a save runs).
    pub extract: Option<ExtractForm>,
}

/// The geometry to save from egui's view of the window. egui reports it
/// in points divided by the text `zoom`; the window is rebuilt at zoom 1,
/// so the zoom is undone here. While maximised, the size and position
/// from before (`prev`) are kept.
fn window_geom(
    outer: Option<egui::Rect>,
    inner: Option<egui::Rect>,
    maximized: bool,
    zoom: f32,
    prev: Option<WindowGeom>,
) -> Option<WindowGeom> {
    match (maximized, inner) {
        (false, Some(inner)) => Some(WindowGeom {
            pos: outer.map(|o| [o.min.x * zoom, o.min.y * zoom]),
            size: [inner.width() * zoom, inner.height() * zoom],
            maximized: false,
        }),
        _ => prev.map(|g| WindowGeom { maximized, ..g }),
    }
}

/// The colour settings a new window starts with.
fn initial_display(s: &Settings) -> Display {
    Display { colormap: s.plot.colormap, log: false, lock: false, range: (0.0, 1.0) }
}

impl TorApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        cli_path: Option<PathBuf>,
        settings: Settings,
        loaded: Loaded,
    ) -> TorApp {
        let ctx = cc.egui_ctx.clone();
        ui::theme::install_fonts(&ctx);
        ui::theme::apply_look(&ctx, settings.look, settings.text_size);
        let wake = ctx.clone();
        let cache_budget = default_cache_budget_bytes();
        let service = SliceService::new(
            cache_budget,
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
            display: initial_display(&settings),
            popups: PopupSet::default(),
            arrow_layer: None,
            panel_tab: PanelTab::Plot,
            saves: Vec::new(),
            plot_rect: None,
            shots: Vec::new(),
            png_saves: Vec::new(),
            service,
            cache_budget,
            playback: None,
            slider_changed_at: None,
            log: DebugLog::from_env(),
            first_image_logged: false,
            ctx,
            settings,
            settings_path: crate::settings::settings_path(),
            save_failed: false,
            last_geom: None,
            windows: Windows::default(),
            renderer: "none".into(),
            scroll_tree: false,
            calc: None,
            extract: None,
        };
        if let Loaded::Recovered(msg) = loaded {
            app.errors.push(msg);
        }
        app.log.log("window created");
        if let Some(rs) = &cc.wgpu_render_state {
            let info = rs.adapter.get_info();
            app.renderer = format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);
            app.log.log(format_args!(
                "adapter: {} backend={:?} type={:?}",
                info.name, info.backend, info.device_type
            ));
        }
        if cc.wgpu_render_state.is_none() {
            app.errors.push("GPU rendering is unavailable, so images cannot be drawn.".into());
        }
        app.last_geom = app.settings.window;
        app.start_files(cli_path);
        app
    }

    /// Opens the command-line file, or else (when Settings ask for it) the
    /// files open when the app last closed; those now missing get one
    /// banner each.
    pub(crate) fn start_files(&mut self, cli_path: Option<PathBuf>) {
        if let Some(p) = cli_path {
            self.open(p);
            return;
        }
        if !self.settings.reopen_last {
            return;
        }
        for p in self.settings.last_open.clone() {
            if p.exists() {
                self.open(p);
            } else {
                self.errors.push(format!("{} from last time was not found", file_name(&p)));
            }
        }
    }

    /// Saves the settings file; a failure is reported once per run.
    pub(crate) fn save_settings(&mut self) {
        let Some(path) = &self.settings_path else { return };
        if let Err(e) = self.settings.save_to(path)
            && !self.save_failed
        {
            self.save_failed = true;
            self.errors.push(format!("Settings could not be saved: {e}"));
        }
    }

    /// Notes the open files, in tree order, for "reopen at start-up".
    pub(crate) fn remember_open_files(&mut self) {
        self.settings.last_open = self.files.iter().map(|f| f.nc.meta.path.clone()).collect();
    }

    /// Follows the window's size and position (kept from before it was
    /// maximised) for saving at exit.
    fn track_window(&mut self) {
        let (outer, inner, maximized) =
            self.ctx.input(|i| (i.viewport().outer_rect, i.viewport().inner_rect, i.viewport().maximized));
        let Some(maximized) = maximized else { return };
        let zoom = self.ctx.zoom_factor();
        if let Some(g) = window_geom(outer, inner, maximized, zoom, self.last_geom) {
            self.last_geom = Some(g);
        }
    }

    /// Opens `path` on a helper thread.
    fn open(&mut self, path: PathBuf) {
        self.log.log(format_args!("open {}", path.display()));
        let p = path.clone();
        let job = Job::spawn(&self.ctx, move || NcFile::open(&p));
        self.opening.push(Opening { name: file_name(&path), path, started: Instant::now(), job });
    }

    fn poll(&mut self) {
        self.poll_extract();
        self.poll_popup_saves();
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
                    self.settings.push_recent(&o.path);
                    self.save_settings();
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
            // A recent file that no longer opens leaves the list.
            if !self.files.iter().any(|f| f.nc.meta.path == o.path) {
                let before = self.settings.recent.len();
                self.settings.remove_recent(&o.path);
                if self.settings.recent.len() != before {
                    self.save_settings();
                }
            }
        }
        if let Some(r) = self.dialog.as_ref().and_then(Job::poll) {
            self.dialog = None;
            for p in r.unwrap_or_default() {
                self.open(p);
            }
        }
        self.popups.poll();
        self.poll_png();
        let mut i = 0;
        while i < self.saves.len() {
            let Some(r) = self.saves[i].1.poll() else {
                i += 1;
                continue;
            };
            let (id, _) = self.saves.remove(i);
            let failed = match r {
                Ok(Some(Err(e))) => Some(e),
                Err(()) => Some("The CSV could not be saved.".to_string()),
                _ => None,
            };
            if let (Some(e), Some(p)) = (failed, self.popups.list.iter_mut().find(|p| p.id == id)) {
                p.error = Some(e);
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
            let found = self.files.iter().find(|f| f.id == *id);
            let pair = found.and_then(|f| Some((f.nc.clone(), f.nc.pair(path)?)));
            let var = found.and_then(|f| Some((f.nc.clone(), f.nc.var_info_any(path).filter(|_| f.nc.pair(path).is_none())?)));
            // Dropping the old view drops its pending request (cancelling it);
            // its map settings may carry over to a map of the same file.
            let old_map = self.view.take().filter(|v| v.file_id == *id).and_then(|v| v.map);
            // Its playback (and so its read-ahead) goes with it.
            self.playback = None;
            if let Some((nc, pair)) = pair {
                self.log.log(format_args!("select {}", pair.path));
                // A pair's own entry draws its own arrows.
                self.arrow_layer = Some(ArrowLayer::of_pair(*id, pair.clone()));
                let mut v = VarView::new_vector(&self.ctx, *id, nc, pair);
                keep_map_settings(&mut v, old_map.as_ref(), &self.settings.plot);
                self.show_view(v);
            } else if let Some((nc, var)) = var {
                self.log.log(format_args!("select {}", var.path));
                // Arrows never carry over between files; while Off, the
                // pre-fill follows the selected variable.
                let keep = self.arrow_layer.as_ref().is_some_and(|l| l.file == *id && l.mode != ArrowMode::Off);
                if !keep {
                    self.arrow_layer = Some(ArrowLayer::fresh(*id, &nc, &var));
                }
                // Picks refused over the previous variable may fit this one.
                let retry = keep && self.arrow_layer.as_ref().is_some_and(|l| l.pair.is_none() && !l.missing_picks());
                let mut v = self.x_view(*id, nc, var);
                keep_map_settings(&mut v, old_map.as_ref(), &self.settings.plot);
                self.show_view(v);
                if retry {
                    self.pair_from_picks(true);
                }
            }
        }
        self.selection = Some(sel);
    }

    /// The view of plain variable `x` under the arrow slot: layered when
    /// the slot's pair fits `x`'s grid; plain otherwise, with a note when
    /// arrows are on but hidden.
    fn x_view(&self, file: u64, nc: Arc<NcFile>, x: VarInfo) -> VarView {
        match self.slot_pair(file) {
            Some(p) if nc.layer_fits(&x.path, &p) => VarView::new_layered(&self.ctx, file, nc, x, p),
            Some(_) => {
                let note = hidden_note(&x.name);
                let mut v = VarView::new(&self.ctx, file, nc, x);
                v.notice = Some(match v.notice.take() {
                    Some(n) => format!("{n}\n{note}"),
                    None => note,
                });
                v
            }
            None => VarView::new(&self.ctx, file, nc, x),
        }
    }

    /// The pair the slot draws on `file`, when arrows are on.
    fn slot_pair(&self, file: u64) -> Option<Arc<VectorPair>> {
        self.arrow_layer.as_ref().filter(|l| l.file == file && l.mode != ArrowMode::Off)?.pair.clone()
    }

    /// The coloured plain variable of the current view (plain or layered);
    /// `None` for a pair's own entry or no view.
    fn current_x(&self) -> Option<(u64, Arc<NcFile>, VarInfo)> {
        let v = self.view.as_ref()?;
        if v.base.is_none() && v.vector.is_some() {
            return None;
        }
        let x = v.nc.var_info_any(v.plain_path())?;
        Some((v.file_id, v.nc.clone(), x))
    }

    /// The variables the Arrows section's drop-downs offer in the slot's
    /// mode, over the coloured variable (a pair's own entry: its centre
    /// grid). Only worked out while the Plot tab shows them.
    fn arrow_candidates(&self) -> Vec<String> {
        let (Some(v), Some(l)) = (&self.view, &self.arrow_layer) else { return Vec::new() };
        if self.panel_tab != PanelTab::Plot || l.file != v.file_id || l.mode == ArrowMode::Off {
            return Vec::new();
        }
        let x = match self.current_x() {
            Some((_, _, x)) => x,
            None => match &v.vector {
                Some(p) => p.var_info(),
                None => return Vec::new(),
            },
        };
        candidates(&v.nc.meta_with_derived(), &x, l.mode)
    }

    /// Brings the current view in line with the arrow slot: a plain or
    /// layered view of the coloured variable, or a pair's own entry.
    fn show_slot(&mut self) {
        let Some(v) = &self.view else { return };
        if let Some((file, nc, x)) = self.current_x() {
            let target = match self.slot_pair(file) {
                Some(p) if nc.layer_fits(&x.path, &p) => layer_path(&x.path, &p.path),
                _ => x.path.clone(),
            };
            let hidden = self.slot_pair(file).is_some() && target == x.path;
            let shown_hidden = v.notice.as_deref().is_some_and(|n| n.contains(&hidden_note(&x.name)));
            if target != v.var.path || hidden != shown_hidden {
                let new = self.x_view(file, nc, x);
                self.replace_view(new);
            }
            return;
        }
        // A pair's own entry: it shows the slot's pair, or no arrows when Off.
        let (file, nc) = (v.file_id, v.nc.clone());
        let Some(l) = self.arrow_layer.as_ref().filter(|l| l.file == file) else { return };
        let on = l.mode != ArrowMode::Off;
        match l.pair.clone() {
            Some(p) if on && p.path != v.var.path => {
                let new = VarView::new_vector(&self.ctx, file, nc, p);
                self.replace_view(new);
            }
            _ => {
                if let Some(a) = self.view.as_mut().and_then(|v| v.arrows.as_mut()) {
                    a.on = on && l.pair.is_some();
                }
            }
        }
    }

    /// Pairs the slot's picks in its mode and shows the result. With
    /// `replace` (a drop-down edit), the pair being edited is unregistered
    /// (with its rotation variants) once a new one is made; it is kept
    /// through failed picks until then. Any failure leaves the slot without
    /// a pair and says why.
    fn pair_from_picks(&mut self, replace: bool) {
        let x = self.current_x();
        let Some(l) = self.arrow_layer.as_mut() else { return };
        let Some(nc) = self.files.iter().find(|f| f.id == l.file).map(|f| f.nc.clone()) else { return };
        let old = l.pair.take();
        if !replace {
            // A form switch keeps the old pair registered: no edit pending.
            l.editing = None;
        } else if old.is_some() {
            l.editing = old;
        }
        let [Some(a), Some(b)] = l.picks.clone() else {
            l.problem = Some(PICK_TWO.into());
            self.show_slot();
            return;
        };
        let mode = l.mode;
        // A registered pair of these two in this form is used as it is
        // (it keeps its settings), preferring the one on show
        // (a pair's own entry keeps its rotation variant).
        let same = |p: &VectorPair| p.east == a && p.north == b && ArrowMode::of_form(p.form) == mode;
        let shown = self.view.as_ref().and_then(|v| v.vector.clone());
        let existing = shown
            .filter(|p| same(p) && nc.pair(&p.path).is_some())
            .or_else(|| nc.pairs().into_iter().find(|p| same(p)));
        let (pair, registered) = match existing {
            Some(p) => (p, true),
            None => {
                let made = match mode {
                    ArrowMode::EastNorth => pair_by_hand(&nc.meta_with_derived(), &a, &b),
                    ArrowMode::MagDir => pair_magdir_by_hand(&nc.meta_with_derived(), &a, &b),
                    ArrowMode::Off => return,
                };
                match made {
                    Ok(p) => (Arc::new(p), false),
                    Err(e) => {
                        l.problem = Some(e);
                        self.show_slot();
                        return;
                    }
                }
            }
        };
        // Nothing is registered for a pair off the coloured variable's grid.
        if let Some((_, _, x)) = &x
            && !nc.layer_fits(&x.path, &pair)
        {
            l.problem = Some(off_grid(&x.name));
            self.show_slot();
            return;
        }
        if let Some(o) = l.editing.take().filter(|o| family(&o.path) != family(&pair.path)) {
            remove_family(&nc, &o.path);
        }
        let pair = if registered { pair } else { register(&nc, Arc::unwrap_or_clone(pair)) };
        l.pair = Some(pair);
        l.problem = None;
        l.remember_picks();
        self.show_slot();
    }

    /// Reads the first slice of a new view and puts it on show.
    fn show_view(&mut self, mut v: VarView) {
        if let (true, Some(m)) = (self.log.enabled(), &mut v.map) {
            m.events = Some(Vec::new());
        }
        v.refresh(&self.ctx, &self.service);
        // A new view starts paused; the old view's read-ahead is dropped.
        self.playback = Some(Playback::new(default_play_dim(&v)));
        self.view = Some(v);
        self.display.lock = false;
        self.first_image_logged = false;
        self.slider_changed_at = None;
    }

    /// Advances the animation: keeps its read-ahead going and, when a frame
    /// is due and read, moves the slider to it. While the view is still
    /// taking in the frame on show, the next one waits.
    fn tick_playback(&mut self) {
        let (Some(pb), Some(v)) = (&mut self.playback, &mut self.view) else { return };
        let dims = v.slider_dims();
        if !dims.contains(&pb.dim) {
            // The view kind changed which dimensions have sliders.
            pb.stop();
            pb.dim = default_play_dim(v);
        }
        if !pb.playing {
            return;
        }
        let now = Instant::now();
        let dim = pb.dim;
        let (len, current) = (v.var.shape[dim], v.slider_pos[dim]);
        let components = if v.vector.is_some() { 3 } else { 1 };
        let bytes = frame_bytes(&v.requests_at(&v.slider_pos), components);
        let base = v.slider_pos.clone();
        let requests = |i: usize| {
            let mut pos = base.clone();
            pos[dim] = i;
            v.requests_at(&pos)
        };
        let budget = self.cache_budget;
        if v.loading() {
            // The frame on show is still being taken in: read ahead, but hold the step.
            pb.keep_ahead(now, current, len, bytes, budget, &self.service, &v.nc, &requests);
            let after = if pb.waiting { WAIT_RECHECK } else { pb.due_in(now) };
            self.ctx.request_repaint_after(after);
            return;
        }
        let tick = pb.tick(now, current, len, bytes, budget, &self.service, &v.nc, &requests);
        match tick {
            Tick::Show(i) => {
                v.slider_pos[dim] = i;
                let ready = pb.take_ready();
                v.refresh_with(&self.ctx, &self.service, &ready);
                self.slider_changed_at = Some((now, v.slider_pos.clone()));
            }
            Tick::Stop => pb.stop(),
            Tick::Idle | Tick::Wait => {}
        }
        if pb.playing {
            // While waiting, finished reads wake the UI; this is a fallback.
            let after = if pb.waiting { WAIT_RECHECK } else { pb.due_in(Instant::now()) };
            self.ctx.request_repaint_after(after);
        }
    }

    /// Milliseconds from the last slider change to this upload, once the
    /// slice on show is the one for the positions moved to (an upload of an
    /// earlier step, whose read finished late, is not counted).
    fn step_shown(&mut self, uploaded: bool) -> Option<f64> {
        let (t, pos) = self.slider_changed_at.as_ref()?;
        if !uploaded || !self.view.as_ref()?.shows_position(pos) {
            return None;
        }
        let took = ms(*t);
        self.slider_changed_at = None;
        Some(took)
    }

    /// Shows `v` in place of the current view of the same grid (another
    /// rotation, convention or arrow layer): slider positions, the camera,
    /// the arrow length, the colour lock and the playback settings stay.
    fn replace_view(&mut self, mut v: VarView) {
        self.selection = Some(Selection::Var(v.file_id, v.plain_path().to_string()));
        v.apply_plot_defaults(&self.settings.plot);
        let Some(old) = self.view.take() else {
            self.show_view(v);
            return;
        };
        if v.slider_pos.len() == old.slider_pos.len() {
            v.slider_pos = old.slider_pos.clone();
        }
        // The view kind the user chose stays when it suits the new view.
        if v.var.shape == old.var.shape && v.kinds().contains(&old.kind) {
            v.kind = old.kind;
        }
        // A pan or zoom over other axes (chosen in the old view) is not kept.
        if v.axes == old.axes {
            v.image.view = old.image.view;
            v.image.fitted_to = old.image.fitted_to;
            v.image.user_moved = old.image.user_moved;
        }
        if let (Some(new), Some(prev)) = (&mut v.map, &old.map) {
            new.inherit_view(prev);
        }
        if let (Some(new), Some(prev)) = (&mut v.arrows, &old.arrows) {
            new.length = prev.length;
            // A pair's own entry keeps its switch (a layered view is on).
            if v.base.is_none() {
                new.on = prev.on;
            }
        }
        // The slider range and colours stay as the user left them.
        let lock = self.display.lock;
        let settings = self.playback.as_ref().map(|p| (p.dim, p.fps, p.looping));
        self.show_view(v);
        self.display.lock = lock;
        // The new view starts paused, with the same playback settings.
        if let (Some(p), Some((dim, fps, looping))) = (&mut self.playback, settings) {
            (p.dim, p.fps, p.looping) = (dim, fps, looping);
        }
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
                // The window closes with its file, unless a save is running.
                self.extract.take_if(|f| f.file == id && f.job.is_none() && f.dialog.is_none());
                self.arrow_layer.take_if(|l| l.file == id);
                let view = self.view.take_if(|v| v.file_id == id);
                if view.is_some() {
                    // Its read-ahead stops with it.
                    self.playback = None;
                }
                let gone: Vec<u64> =
                    self.popups.list.iter().filter(|p| p.file_id == id).map(|p| p.id).collect();
                for p in gone {
                    self.popups.close(p);
                }
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
                    self.slider_changed_at = Some((Instant::now(), v.slider_pos.clone()));
                }
            }
            Action::SetKind(kind) => {
                if let Some(v) = &mut self.view
                    && v.kind != kind
                {
                    v.kind = kind;
                    // The map draws lat/lon, whatever the image had on its axes.
                    if kind == ViewKind::Map {
                        v.restore_geo_axes();
                    }
                    v.shown = None;
                    v.image = ImageState::default();
                    // The map keeps its camera and cell index, not its tile.
                    if let Some(m) = &mut v.map {
                        m.reset_tiles();
                    }
                    v.refresh(&self.ctx, &self.service);
                }
            }
            Action::SetAxes { y, x } => {
                if let Some(v) = &mut self.view {
                    v.set_axes(&self.ctx, &self.service, AxisChoice { y, x });
                }
            }
            Action::Unpair { file, path } => {
                let Some(f) = self.files.iter().find(|f| f.id == file) else { return };
                for tag in ["", "#on", "#off"] {
                    f.nc.remove_pair(&format!("{path}{tag}"));
                }
                let of_it = |p: &str| p.strip_prefix(&path).is_some_and(|t| t.is_empty() || t.starts_with('#'));
                let shown = matches!(&self.selection, Some(Selection::Var(id, p)) if *id == file && of_it(p));
                if shown {
                    self.view = None;
                    self.playback = None;
                    self.selection = None;
                }
                // The live arrow layer of it turns Off; a layered view of
                // it becomes the plain view of its variable.
                if let Some(l) = self.arrow_layer.as_mut().filter(|l| l.file == file)
                    && l.editing.as_ref().is_some_and(|p| of_it(&p.path))
                {
                    l.editing = None;
                }
                if let Some(l) = self.arrow_layer.as_mut().filter(|l| l.file == file)
                    && l.pair.as_ref().is_some_and(|p| of_it(&p.path))
                {
                    l.mode = ArrowMode::Off;
                    l.pair = None;
                    self.show_slot();
                }
            }
            Action::SetRotate(choice) => {
                let Some(old) = self.view.as_ref().and_then(|v| Some((v.nc.clone(), v.vector.clone()?)))
                else {
                    return;
                };
                let (nc, pair) = old;
                let new = pair.with_rotate(choice);
                if new.path != pair.path {
                    let new = nc.add_pair(new);
                    // The slot follows the pair drawn (rotation is per pair).
                    if let Some(l) = self.arrow_layer.as_mut()
                        && l.pair.as_ref().is_some_and(|p| p.path == pair.path)
                    {
                        l.pair = Some(new.clone());
                    }
                    let (file, layered) = self.view.as_ref().map_or((0, false), |v| (v.file_id, v.base.is_some()));
                    if layered {
                        if let Some((file, nc, x)) = self.current_x() {
                            let v = VarView::new_layered(&self.ctx, file, nc, x, new);
                            self.replace_view(v);
                        }
                    } else {
                        let v = VarView::new_vector(&self.ctx, file, nc, new);
                        self.replace_view(v);
                    }
                }
            }
            Action::SetArrowMode(mode) => {
                let Some(l) = self.arrow_layer.as_mut() else { return };
                if l.mode == mode {
                    return;
                }
                l.mode = mode;
                // The old pair stays registered; the slot no longer draws it.
                l.pair = None;
                l.editing = None;
                if mode == ArrowMode::Off {
                    l.problem = l.missing_picks().then(|| PICK_TWO.to_string());
                    self.show_slot();
                } else {
                    // Each form starts from its own picks.
                    l.load_prefill(mode);
                    self.pair_from_picks(false);
                }
            }
            Action::PickArrowVar { slot, path } => {
                let Some(l) = self.arrow_layer.as_mut() else { return };
                if slot > 1 || (l.picks[slot].as_deref() == Some(path.as_str()) && l.pair.is_some()) {
                    return;
                }
                l.picks[slot] = Some(path);
                // While Off the picks are only kept.
                if l.mode != ArrowMode::Off {
                    self.pair_from_picks(true);
                }
            }
            Action::SetConvention(c) => {
                let Some(l) = self.arrow_layer.as_mut() else { return };
                let Some(old) = l.pair.clone().filter(|p| matches!(p.form, PairForm::MagDir(_))) else { return };
                let Some(nc) = self.files.iter().find(|f| f.id == l.file).map(|f| f.nc.clone()) else { return };
                let new = old.with_convention(c);
                if new == *old {
                    return;
                }
                // The pair is replaced at its new path, never added beside it.
                remove_family(&nc, &old.path);
                l.pair = Some(register(&nc, new));
                self.show_slot();
            }
            Action::SetPanelTab(tab) => self.panel_tab = tab,
            Action::FitView => {
                if let Some(v) = &mut self.view {
                    v.image.view = None;
                    v.image.user_moved = false;
                    if let Some(m) = &mut v.map {
                        m.request_fit();
                    }
                }
            }
            Action::OpenPopup { iy, ix, lonlat } => {
                let Some(v) = &mut self.view else { return };
                let id = self.popups.next_id();
                match Popup::open(v, iy, ix, lonlat, id, &self.service) {
                    Ok(p) => self.popups.place(p),
                    Err(e) => v.set_flash(e, NOTICE_TIME),
                }
            }
            Action::ClosePopup(id) => self.popups.close(id),
            Action::SetPopupKind(id, kind) => {
                if let Some(p) = self.popups.list.iter_mut().find(|p| p.id == id) {
                    p.set_kind(kind, &self.service);
                }
            }
            Action::JumpTo { file, dim_name, index } => {
                let Some(v) = self.view.as_mut().filter(|v| v.file_id == file) else { return };
                let Some(d) = v.var.dims.iter().position(|n| *n == dim_name) else { return };
                if index >= v.var.shape[d] || !v.slider_dims().contains(&d) || v.slider_pos[d] == index {
                    return;
                }
                v.slider_pos[d] = index;
                v.refresh(&self.ctx, &self.service);
                self.slider_changed_at = Some((Instant::now(), v.slider_pos.clone()));
            }
            Action::SavePopupNetcdf(id) => self.save_popup_netcdf_dialog(id),
            Action::SavePopupCsv(id) => {
                let Some(p) = self.popups.list.iter().find(|p| p.id == id) else { return };
                if self.saves.iter().any(|(s, _)| *s == id) {
                    return;
                }
                let (csv, name) = (p.csv_file_bytes(), format!("{}.csv", p.var_name));
                // The dialog and the write block, so they get their own thread.
                let job = Job::spawn(&self.ctx, move || {
                    let path = rfd::FileDialog::new()
                        .set_title("Save the series as CSV")
                        .set_file_name(name)
                        .add_filter("CSV", &["csv"])
                        .save_file()?;
                    Some(std::fs::write(&path, csv).map_err(|e| format!("Cannot save {}: {e}", path.display())))
                });
                self.saves.push((id, job));
            }
            Action::SaveImage => {
                let (Some(v), Some(rect)) = (&self.view, self.plot_rect) else { return };
                self.request_shot(rect, png_export::default_name(v), None);
            }
            Action::SavePopupPng(id) => {
                let Some(p) = self.popups.list.iter().find(|p| p.id == id) else { return };
                let Some(rect) = p.rect else { return };
                self.request_shot(rect, png_export::popup_name(&p.var_name), Some(id));
            }
            Action::Play => {
                let (Some(pb), Some(v)) = (&mut self.playback, &mut self.view) else { return };
                if !v.slider_dims().contains(&pb.dim) {
                    return;
                }
                // At the end without looping, play starts over from the first step.
                let d = pb.dim;
                if !pb.looping && v.slider_pos[d] + 1 >= v.var.shape[d] {
                    v.slider_pos[d] = 0;
                    v.refresh(&self.ctx, &self.service);
                    self.slider_changed_at = Some((Instant::now(), v.slider_pos.clone()));
                }
                // Frames keep the colours they start with (unlocking stays possible).
                if let Some(range) = self.display.effective(v.shown.as_ref().map(|s| &s.stats)) {
                    self.display.range = range;
                    self.display.lock = true;
                }
                pb.start(Instant::now());
                self.ctx.request_repaint();
            }
            Action::Pause => {
                if let Some(pb) = &mut self.playback {
                    pb.stop();
                }
            }
            Action::SetFps(fps) => {
                if let Some(pb) = &mut self.playback
                    && fps.is_finite()
                {
                    pb.fps = fps.clamp(*FPS_RANGE.start(), *FPS_RANGE.end());
                }
            }
            Action::SetLoop(looping) => {
                if let Some(pb) = &mut self.playback {
                    pb.looping = looping;
                }
            }
            Action::SetPlayDim(d) => {
                let (Some(pb), Some(v)) = (&mut self.playback, &self.view) else { return };
                if pb.dim == d || !v.slider_dims().contains(&d) {
                    return;
                }
                // Read-ahead along the old dimension is dropped.
                let playing = pb.playing;
                pb.stop();
                pb.dim = d;
                if playing {
                    pb.start(Instant::now());
                }
            }
            Action::Exit => self.ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            Action::OpenRecent(p) => self.open(p),
            Action::ShowWindow(w) => self.windows.open(w),
            Action::SetLook(look) => {
                self.settings.look = look;
                self.apply_look_and_save();
            }
            Action::SetTextSize(size) => {
                self.settings.text_size = size;
                self.apply_look_and_save();
            }
            Action::SetPlotDefaults(d) => {
                self.settings.plot = d;
                self.save_settings();
            }
            Action::SetRecentMax(n) => {
                self.settings.recent_max = n;
                self.settings.recent.truncate(n);
                self.save_settings();
            }
            Action::SetReopenLast(on) => {
                self.settings.reopen_last = on;
                self.save_settings();
            }
            Action::ResetSettings => {
                self.settings.reset_preferences();
                self.settings.recent.truncate(self.settings.recent_max);
                self.apply_look_and_save();
            }
            Action::OpenCalculate { file, edit } => self.open_calculate(file, edit.as_deref()),
            Action::CalcEdited => self.recheck_calc(),
            Action::CreateDerived => self.create_derived(),
            Action::CloseCalculate => self.calc = None,
            Action::UseRecentFormula(i) => {
                if let (Some(form), Some(r)) = (self.calc.as_mut(), self.settings.recent_formulas.get(i)) {
                    if form.replacing.is_none() {
                        form.name = r.name.clone();
                    }
                    form.formula = r.formula.clone();
                    form.units = r.units.clone();
                }
                self.recheck_calc();
            }
            Action::DeleteDerived { file, path } => self.delete_derived(file, &path),
            Action::OpenExtract { file, var } => self.open_extract(file, var),
            Action::ExtractEdited => self.recheck_extract(),
            Action::ExtractUseView => {
                let r = self.view_region();
                if let (Some(f), Some(r)) = (self.extract.as_mut(), r) {
                    f.region = Some(f.clamped(r));
                }
                self.recheck_extract();
            }
            Action::ExtractWholeGrid => {
                if let Some(f) = self.extract.as_mut() {
                    f.region = f.whole_region();
                }
                self.recheck_extract();
            }
            Action::ExtractFromSlider { row, end } => self.extract_from_slider(row, end),
            Action::ExtractRegionDrawn { south, north, west, east } => {
                let Some(file) = self.view.as_ref().map(|v| v.file_id) else { return };
                let n = crate::view::extract::fmt_num;
                let r = RegionText { south: n(south), north: n(north), west: n(west), east: n(east) };
                if let Some(f) = self.extract_for(file) {
                    f.pending_region = Some((r, false));
                    f.apply_pending();
                }
                self.recheck_extract();
            }
            Action::ExtractIndexBoxDrawn { y, x } => {
                let Some(v) = self.view.as_ref() else { return };
                let Some(a) = v.axes else { return };
                let key = |d: usize| v.nc.dim_key(v.plain_path(), &v.var.dims[d]);
                let (Some(ky), Some(kx), file) = (key(a.y), key(a.x), v.file_id) else { return };
                if let Some(f) = self.extract_for(file) {
                    f.pending_index.extend([(ky, y.0, y.1), (kx, x.0, x.1)]);
                    f.apply_pending();
                }
                self.recheck_extract();
            }
            Action::ExtractTimeDragged { file, from, to } => {
                if let Some(f) = self.extract_for(file) {
                    f.pending_time = Some((from, to));
                    f.apply_pending();
                }
                self.recheck_extract();
            }
            Action::SaveExtract => self.save_extract_dialog(),
            Action::CancelExtract => {
                if let Some(j) = self.extract.as_ref().and_then(|f| f.job.as_ref()) {
                    j.cancel();
                }
            }
            Action::CloseExtract => {
                // A running save is cancelled (its partial file is removed).
                if let Some(j) = self.extract.take().and_then(|f| f.job) {
                    j.cancel();
                }
            }
        }
    }

    /// Opens the Calculate window for `file`: empty, or filled from the
    /// calculated variable `edit`.
    fn open_calculate(&mut self, file: u64, edit: Option<&str>) {
        let Some(f) = self.files.iter().find(|f| f.id == file) else { return };
        let mut form = CalcForm {
            file,
            name: String::new(),
            formula: String::new(),
            units: String::new(),
            replacing: None,
            check: Err(String::new()),
        };
        if let Some(d) = edit.and_then(|p| f.nc.derived(p)) {
            form.name = d.name.clone();
            form.formula = d.formula.clone();
            form.units = d.units.clone().unwrap_or_default();
            form.replacing = Some(d.path.clone());
        }
        self.calc = Some(form);
        self.recheck_calc();
    }

    /// Checks the Calculate window's formula again (metadata only).
    fn recheck_calc(&mut self) {
        let Some(form) = self.calc.as_mut() else { return };
        let Some(f) = self.files.iter().find(|f| f.id == form.file) else { return };
        form.check = match f.nc.check_formula(&form.name, &form.formula, Some(&form.units), form.replacing.as_deref()) {
            Ok(d) => {
                let shape: Vec<String> = d.info.shape.iter().map(usize::to_string).collect();
                Ok(format!("{} ({})  {}", d.name, d.info.dims.join(", "), shape.join(" × ")))
            }
            Err(e) => Err(e.to_string()),
        };
    }

    /// Registers the Calculate window's variable (or the edit of one), shows
    /// it and closes the window. After an edit nothing computed from the
    /// old formula stays: cached slices go, pop-ups of it close and the view
    /// is rebuilt.
    fn create_derived(&mut self) {
        // A save computes calculated variables as it goes; changing one
        // under it would mix old and new values in the file.
        let saving = self.extract.as_ref().is_some_and(|f| f.job.is_some())
            || self.popups.list.iter().any(|p| p.nc_job.is_some());
        let Some(form) = self.calc.as_mut() else { return };
        if saving && form.replacing.is_some() {
            form.check = Err("A save is running; apply this change once it has finished.".into());
            return;
        }
        let Some(f) = self.files.iter().find(|f| f.id == form.file) else {
            self.calc = None;
            return;
        };
        let nc = f.nc.clone();
        let d = match nc.check_formula(&form.name, &form.formula, Some(&form.units), form.replacing.as_deref()) {
            Ok(d) => d,
            Err(e) => {
                form.check = Err(e.to_string());
                return;
            }
        };
        let (file, path, editing) = (form.file, d.path.clone(), form.replacing.is_some());
        let recent = RecentFormula { name: d.name.clone(), formula: d.formula.clone(), units: form.units.trim().to_string() };
        nc.add_derived(d);
        // Slices cached under this path may hold another formula's values
        // (an edit, or a deleted variable of the same name).
        self.service.forget_file(&nc);
        self.calc = None;
        if editing {
            let affected = nc.affected_by(&path);
            let gone: Vec<u64> = self
                .popups
                .list
                .iter()
                .filter(|p| p.file_id == file && affected.contains(&p.var_path))
                .map(|p| p.id)
                .collect();
            for id in gone {
                self.popups.close(id);
            }
            if let Some(sel @ Selection::Var(id, _)) = self.selection.clone()
                && id == file
            {
                self.selection = None;
                self.select(sel);
            }
        } else {
            self.select(Selection::Var(file, path));
            self.scroll_tree = true;
        }
        self.settings.push_formula(recent);
        self.save_settings();
    }

    /// Removes a calculated variable; refused (with a banner) while
    /// another calculated variable or an arrow pair uses it.
    fn delete_derived(&mut self, file: u64, path: &str) {
        let Some(f) = self.files.iter().find(|f| f.id == file) else { return };
        if let Err(e) = f.nc.remove_derived(path) {
            self.errors.push(e);
            return;
        }
        self.service.forget_file(&f.nc);
        let gone: Vec<u64> =
            self.popups.list.iter().filter(|p| p.file_id == file && p.var_path == path).map(|p| p.id).collect();
        for id in gone {
            self.popups.close(id);
        }
        if matches!(&self.selection, Some(Selection::Var(id, p)) if *id == file && p == path) {
            self.view = None;
            self.playback = None;
            self.selection = None;
        }
    }

    /// Carries out a keyboard shortcut.
    pub(crate) fn command(&mut self, c: Command) {
        match c {
            Command::Open => self.apply(Action::OpenDialog),
            Command::CloseFile => {
                if let Some(id) = self.current_file() {
                    self.apply(Action::CloseFile(id));
                }
            }
            Command::ExportPng => {
                if self.view.is_some() {
                    self.apply(Action::SaveImage);
                }
            }
            Command::Settings => self.windows.open(Win::Settings),
            Command::ShortcutHelp => self.windows.open(Win::Shortcuts),
            Command::Calculate => {
                if let Some(id) = self.current_file() {
                    self.open_calculate(id, None);
                }
            }
            Command::Extract => {
                if let Some(id) = self.current_file() {
                    self.open_extract(id, None);
                }
            }
            Command::StepBack | Command::StepForward | Command::FirstStep | Command::LastStep => self.step(c),
            Command::PlayPause => {
                let Some(pb) = &self.playback else { return };
                if self.view.as_ref().is_some_and(|v| v.slider_dims().contains(&pb.dim)) {
                    self.apply(if pb.playing { Action::Pause } else { Action::Play });
                }
            }
            Command::Fit => {
                if self.view.as_ref().is_some_and(|v| matches!(v.kind, ViewKind::Image | ViewKind::Map)) {
                    self.apply(Action::FitView);
                }
            }
            Command::PrevVar | Command::NextVar => {
                let step = if c == Command::NextVar { 1 } else { -1 };
                let order = tree::var_order(&self.files);
                if let Some(s) = shortcuts::neighbour(&order, self.selection.as_ref(), step)
                    && self.selection.as_ref() != Some(&s)
                {
                    self.apply(Action::Select(s));
                    self.scroll_tree = true;
                }
            }
        }
    }

    /// Moves the playback dimension's slider (or the first slider's, when
    /// that one is not a slider of the view) one step or to an end.
    fn step(&mut self, c: Command) {
        let Some(v) = &mut self.view else { return };
        let dims = v.slider_dims();
        let Some(dim) = self.playback.as_ref().map(|p| p.dim).filter(|d| dims.contains(d)).or(dims.first().copied())
        else {
            return;
        };
        let last = v.var.shape[dim] - 1;
        let pos = v.slider_pos[dim].min(last);
        let to = match c {
            Command::StepBack => pos.saturating_sub(1),
            Command::StepForward => (pos + 1).min(last),
            Command::FirstStep => 0,
            _ => last,
        };
        if to != v.slider_pos[dim] {
            v.slider_pos[dim] = to;
            self.apply(Action::SliderMoved);
        }
    }

    /// The file of the selection, or else of the view (for Close file).
    pub(crate) fn current_file(&self) -> Option<u64> {
        match &self.selection {
            Some(Selection::File(id) | Selection::Group(id, _) | Selection::Var(id, _)) => Some(*id),
            None => self.view.as_ref().map(|v| v.file_id),
        }
    }

    fn apply_look_and_save(&mut self) {
        ui::theme::apply_look(&self.ctx, self.settings.look, self.settings.text_size);
        self.save_settings();
    }

}

/// The Extract window.
impl TorApp {
    /// Opens the Extract window for `file` (netCDF files only), ticking
    /// `var`, else the variable on screen (both components of an arrow
    /// pair). An open window for the same file keeps its choices.
    fn open_extract(&mut self, file: u64, var: Option<String>) {
        let Some(f) = self.files.iter().find(|f| f.id == file) else { return };
        if !f.nc.is_netcdf() {
            return;
        }
        let mut ticks: Vec<String> = Vec::new();
        match var {
            Some(v) => ticks.push(v),
            None => {
                if let Some(v) = self.view.as_ref().filter(|v| v.file_id == file) {
                    match (&v.vector, &v.base) {
                        (Some(p), None) => ticks.extend([p.east.clone(), p.north.clone()]),
                        _ => ticks.push(v.plain_path().to_string()),
                    }
                }
            }
        }
        ticks.retain(|p| f.nc.meta.var(p).is_some() || f.nc.derived(p).is_some());
        let view_region = self.view_region();
        match self.extract.as_mut() {
            Some(form) if form.file == file => form.ticked.extend(ticks),
            Some(form) if form.job.is_some() => {
                self.errors.push(format!("A save from {} is running; wait for it or cancel it first.", form.file_name));
                return;
            }
            _ => {
                let mut form = ExtractForm::new(file, f.name.clone(), ticks.into_iter().collect());
                form.pending_region = view_region.map(|r| (r, true));
                self.extract = Some(form);
            }
        }
        self.recheck_extract();
    }

    /// The Extract window for `file`, opened (with the variable on screen) if needed.
    fn extract_for(&mut self, file: u64) -> Option<&mut ExtractForm> {
        if self.extract.as_ref().is_none_or(|f| f.file != file) {
            self.open_extract(file, None);
        }
        self.extract.as_mut().filter(|f| f.file == file)
    }

    /// The map's visible lon/lat window as region text.
    fn view_region(&self) -> Option<RegionText> {
        let v = self.view.as_ref().filter(|v| v.kind == ViewKind::Map)?;
        let m = v.map.as_ref()?;
        let w = crate::view::map_state::visible_window(&m.camera, m.last_rect?)?;
        let n = crate::view::extract::fmt_num;
        Some(RegionText { south: n(w.lat[0]), north: n(w.lat[1]), west: n(w.lon[0]), east: n(w.lon[1]) })
    }

    /// Checks the Extract window again, first reading the coordinates of
    /// newly ticked variables (off the UI thread).
    fn recheck_extract(&mut self) {
        let Some(form) = self.extract.as_mut() else { return };
        let Some(f) = self.files.iter().find(|f| f.id == form.file) else { return };
        if !form.ticked.is_empty() && form.coords_stale() {
            if form.loading.is_none() {
                let (nc, ticked) = (f.nc.clone(), form.ticked.clone());
                form.loading = Some(Job::spawn(&self.ctx, move || load_coords(&nc, &ticked)));
            }
            form.check = Err("reading coordinates…".into());
            return;
        }
        form.recheck(&f.nc);
    }

    /// Coordinates arriving, the Save dialog closing and the save finishing.
    fn poll_extract(&mut self) {
        let Some(form) = self.extract.as_mut() else { return };
        if let Some(r) = form.loading.as_ref().and_then(|j| j.poll()) {
            form.loading = None;
            let c = r.unwrap_or_else(|()| crate::view::extract::FormCoords {
                ticked: form.ticked.clone(),
                dims: Vec::new(),
                region: None,
            });
            form.set_coords(c);
            self.recheck_extract();
        }
        let Some(form) = self.extract.as_mut() else { return };
        if let Some(r) = form.dialog.as_ref().and_then(|j| j.poll()) {
            form.dialog = None;
            if let Ok(Some(path)) = r {
                self.start_extract(path);
            }
        }
        let Some(form) = self.extract.as_mut() else { return };
        let Some(job) = &form.job else { return };
        match job.finished() {
            Some(Ok(path)) => {
                let skipped = job.skipped();
                let mut line = format!("Saved to {}", path.display());
                if !skipped.is_empty() {
                    line += &format!(". Not saved: {}", skipped.join(", "));
                }
                form.status = Some(line);
                form.job = None;
            }
            Some(Err(e)) => {
                form.status = Some(format!("Not saved: {e}"));
                form.job = None;
            }
            None => self.ctx.request_repaint_after(std::time::Duration::from_millis(100)),
        }
    }

    /// Opens the Save dialog for the checked extract.
    fn save_extract_dialog(&mut self) {
        self.recheck_extract();
        let Some(form) = self.extract.as_mut() else { return };
        if form.busy() || form.check.is_err() {
            return;
        }
        let Some(f) = self.files.iter().find(|f| f.id == form.file) else { return };
        let src = f.nc.meta.path.clone();
        let dir = src.parent().map(Path::to_path_buf);
        let stem = src.file_stem().map_or_else(|| "extract".into(), |s| s.to_string_lossy().into_owned());
        let name = format!("{stem}_extract.nc");
        form.dialog = Some(Job::spawn(&self.ctx, move || {
            let mut d = rfd::FileDialog::new()
                .set_title("Save the extract as netCDF")
                .set_file_name(name)
                .add_filter("netCDF", &["nc"]);
            if let Some(dir) = dir {
                d = d.set_directory(dir);
            }
            d.save_file()
        }));
    }

    /// Starts saving the checked extract to `path` (never the file being read).
    pub(crate) fn start_extract(&mut self, path: PathBuf) {
        let Some(form) = self.extract.as_mut() else { return };
        if form.job.is_some() {
            return;
        }
        let Some(f) = self.files.iter().find(|f| f.id == form.file) else { return };
        let plan = match &form.check {
            Ok(p) => p.clone(),
            Err(e) => {
                form.status = Some(format!("Not saved: {e}"));
                return;
            }
        };
        let src = &f.nc.meta.path;
        let same = path == *src
            || matches!((std::fs::canonicalize(&path), std::fs::canonicalize(src)), (Ok(a), Ok(b)) if a == b);
        if same {
            form.status = Some("Not saved: that is the file being read; choose another name".into());
            return;
        }
        let line = torcdf_core::extract::history_line(&f.name, &form.summary());
        form.status = None;
        form.job = Some(torcdf_core::extract::ExtractJob::start(f.nc.clone(), plan, path, line));
        self.ctx.request_repaint();
    }

    /// Row `row` takes the step the view's slider of that dimension is at.
    fn extract_from_slider(&mut self, row: usize, end: bool) {
        let (Some(form), Some(v)) = (self.extract.as_mut(), self.view.as_ref()) else { return };
        if v.file_id != form.file {
            return;
        }
        let Some(key) = form.dims.get(row).map(|r| r.key.clone()) else { return };
        let Some(d) = slider_dim_of(v, &key) else { return };
        let Some(text) = form.text_at(&key, v.slider_pos[d]) else { return };
        if end {
            form.dims[row].to = text;
        } else {
            form.dims[row].from = text;
        }
        self.recheck_extract();
    }
}

/// The slider dimension of `v` that is the dimension `key`.
fn slider_dim_of(v: &VarView, key: &torcdf_core::extract::DimKey) -> Option<usize> {
    let path = v.plain_path();
    v.slider_dims().into_iter().find(|&d| v.nc.dim_key(path, &v.var.dims[d]).as_ref() == Some(key))
}

/// A pop-up's Save netCDF.
impl TorApp {
    /// Opens the Save dialog for pop-up `id`'s point.
    fn save_popup_netcdf_dialog(&mut self, id: u64) {
        let Some(p) = self.popups.list.iter_mut().find(|p| p.id == id) else { return };
        if p.saving() || !p.is_netcdf() {
            return;
        }
        let Some(f) = self.files.iter().find(|f| f.id == p.file_id) else { return };
        let src = f.nc.meta.path.clone();
        let name = popup_file_name(&src, &p.var_name, p.lonlat);
        let dir = src.parent().map(Path::to_path_buf);
        p.nc_dialog = Some(Job::spawn(&self.ctx, move || {
            let mut d = rfd::FileDialog::new()
                .set_title("Save the point as netCDF")
                .set_file_name(name)
                .add_filter("netCDF", &["nc"]);
            if let Some(dir) = dir {
                d = d.set_directory(dir);
            }
            d.save_file()
        }));
    }

    /// Starts saving pop-up `id`'s point to `path`.
    pub(crate) fn start_popup_extract(&mut self, id: u64, path: PathBuf) {
        let Some(p) = self.popups.list.iter_mut().find(|p| p.id == id) else { return };
        if p.nc_job.is_some() {
            return;
        }
        let Some(f) = self.files.iter().find(|f| f.id == p.file_id) else { return };
        let plan = match f.nc.plan_extract(&crate::view::popup::popup_selection(p, &f.nc)) {
            Ok(plan) => plan,
            Err(e) => {
                p.nc_status = Some(format!("Not saved: {e}"));
                return;
            }
        };
        let src = &f.nc.meta.path;
        if path == *src || matches!((std::fs::canonicalize(&path), std::fs::canonicalize(src)), (Ok(a), Ok(b)) if a == b) {
            p.nc_status = Some("Not saved: that is the file being read; choose another name".into());
            return;
        }
        let summary = match p.lonlat {
            Some((lon, lat)) => format!("point lat {lat:.4}, lon {lon:.4}"),
            None => format!("point {}", p.title),
        };
        let line = torcdf_core::extract::history_line(&f.name, &summary);
        p.nc_status = None;
        p.nc_job = Some(torcdf_core::extract::ExtractJob::start(f.nc.clone(), plan, path, line));
        self.ctx.request_repaint();
    }

    /// Pop-up Save dialogs closing and saves finishing.
    fn poll_popup_saves(&mut self) {
        let mut chosen = Vec::new();
        let mut running = false;
        for p in &mut self.popups.list {
            if let Some(r) = p.nc_dialog.as_ref().and_then(|j| j.poll()) {
                p.nc_dialog = None;
                if let Ok(Some(path)) = r {
                    chosen.push((p.id, path));
                }
            }
            let Some(job) = &p.nc_job else { continue };
            match job.finished() {
                Some(Ok(path)) => {
                    p.nc_status = Some(format!("Saved to {}", path.display()));
                    p.nc_job = None;
                }
                Some(Err(e)) => {
                    p.nc_status = Some(format!("Not saved: {e}"));
                    p.nc_job = None;
                }
                None => running = true,
            }
        }
        if running {
            self.ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        for (id, path) in chosen {
            self.start_popup_extract(id, path);
        }
    }
}

/// `<stem>_<var>_<lat>_<lon>.nc` (4 decimals, `-` as `m`), or
/// `<stem>_<var>_point.nc` without a lon/lat.
fn popup_file_name(src: &Path, var: &str, lonlat: Option<(f64, f64)>) -> String {
    let stem = src.file_stem().map_or_else(|| "point".into(), |s| s.to_string_lossy().into_owned());
    match lonlat {
        Some((lon, lat)) => {
            let n = |v: f64| format!("{v:.4}").replace('-', "m");
            format!("{stem}_{var}_{}_{}.nc", n(lat), n(lon))
        }
        None => format!("{stem}_{var}_point.nc"),
    }
}

/// Names the Calculate window offers: the file's numeric variables and its
/// calculated ones, as a backtick path where the name is not unique.
fn calc_names(nc: &NcFile) -> Vec<String> {
    let ds = nc.meta_with_derived();
    let vars: Vec<&VarInfo> = ds.all_vars().into_iter().filter(|v| v.dtype.is_numeric()).collect();
    vars.iter()
        .map(|v| {
            let plain = v.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && v.name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
            let unique = vars.iter().filter(|w| w.name == v.name).count() == 1;
            match (plain, unique) {
                (true, true) => v.name.clone(),
                (false, true) => format!("`{}`", v.name),
                _ => format!("`{}`", v.path),
            }
        })
        .collect()
}

/// Title and attributes of the selected item.
fn selected_attrs(files: &[OpenFile], sel: Option<&Selection>) -> Option<(String, Vec<Attr>)> {
    let file = |id: &u64| files.iter().find(|f| f.id == *id);
    match sel? {
        Selection::File(id) => {
            let f = file(id)?;
            Some((format!("{} (global)", f.name), f.nc.meta.root.attrs.clone()))
        }
        Selection::Group(id, path) => {
            let g = find_group(&file(id)?.nc.meta.root, path)?;
            Some((format!("group {}", g.path), g.attrs.clone()))
        }
        Selection::Var(id, path) => {
            let v = file(id)?.nc.var_info_any(path)?;
            Some((v.path, v.attrs))
        }
    }
}

impl TorApp {
    /// Asks for a screenshot; `rect` is cut out of it when it arrives.
    fn request_shot(&mut self, rect: egui::Rect, name: String, popup: Option<u64>) {
        if png_pending(self.shots.iter().map(|s| s.popup).chain(self.png_saves.iter().map(|(t, _)| *t)), popup) {
            return;
        }
        // The previous "Saved x.png" message and the hover readout must not
        // end up in the picture.
        if let Some(v) = &mut self.view {
            v.flash = None;
            v.hover = None;
        }
        let (tx, rx) = channel();
        let wake = self.ctx.clone();
        self.ctx.request_screenshot(move |image| {
            let _ = tx.send(image);
            wake.request_repaint();
        });
        self.shots.push(Shot { rx, rect, ppp: self.ctx.pixels_per_point(), name, popup });
    }

    /// Screenshots that arrived go to a save job (crop, encode, dialog,
    /// write); finished jobs report in the plot area or the error banner.
    fn poll_png(&mut self) {
        let mut i = 0;
        while i < self.shots.len() {
            match self.shots[i].rx.try_recv() {
                Err(TryRecvError::Empty) => i += 1,
                Err(TryRecvError::Disconnected) => {
                    self.shots.remove(i);
                    self.errors.push("The screenshot could not be taken.".into());
                }
                Ok(image) => {
                    let s = self.shots.remove(i);
                    let target = s.popup;
                    let job = Job::spawn(&self.ctx, move || {
                        let cut = png_export::crop(&image, s.rect, s.ppp);
                        let png = match png_export::encode_png(&cut) {
                            Ok(b) => b,
                            Err(e) => return Some(Err(e)),
                        };
                        let path = rfd::FileDialog::new()
                            .set_title("Save as PNG")
                            .set_file_name(s.name)
                            .add_filter("PNG image", &["png"])
                            .save_file()?;
                        Some(match std::fs::write(&path, png) {
                            Ok(()) => Ok(file_name(&path)),
                            Err(e) => Err(format!("Cannot save {}: {e}", path.display())),
                        })
                    });
                    self.png_saves.push((target, job));
                }
            }
        }
        let mut i = 0;
        while i < self.png_saves.len() {
            let Some(r) = self.png_saves[i].1.poll() else {
                i += 1;
                continue;
            };
            self.png_saves.remove(i);
            match r {
                Ok(Some(Ok(name))) => {
                    if let Some(v) = &mut self.view {
                        v.set_flash(format!("Saved {name}"), NOTICE_TIME);
                    }
                }
                Ok(Some(Err(e))) => self.errors.push(e),
                Err(()) => self.errors.push("The PNG could not be saved.".into()),
                Ok(None) => {}
            }
        }
    }
}

/// Whether a PNG save for `target` (the plot, or a pop-up id) is already
/// pending, among the targets of the screenshots and dialogs in progress.
fn png_pending(pending: impl IntoIterator<Item = Option<u64>>, target: Option<u64>) -> bool {
    pending.into_iter().any(|t| t == target)
}

/// A new map view of the same file keeps the projection, overlay switches
/// and camera of the map it replaces; any other view starts from defaults.
/// A new view starts from the Settings defaults; a map of the same file
/// shown before keeps its camera and overlay switches instead.
fn keep_map_settings(v: &mut VarView, old: Option<&crate::view::map_state::MapState>, d: &PlotDefaults) {
    v.apply_plot_defaults(d);
    if let (Some(new), Some(old)) = (&mut v.map, old) {
        new.inherit_view(old);
    }
}

/// The note on a plain view while the arrow layer cannot sit on `var`.
pub(crate) fn hidden_note(var: &str) -> String {
    format!("Arrows hidden: {var} is on a different grid")
}

/// Why picked variables were not paired over `var`.
fn off_grid(var: &str) -> String {
    format!("These two do not sit on {var}'s grid")
}

/// A pair path without its rotation tag.
fn family(path: &str) -> &str {
    path.strip_suffix("#on").or_else(|| path.strip_suffix("#off")).unwrap_or(path)
}

/// Unregisters the pair at `path` with its rotation variants.
fn remove_family(nc: &NcFile, path: &str) {
    let base = family(path);
    for tag in ["", "#on", "#off"] {
        nc.remove_pair(&format!("{base}{tag}"));
    }
}

/// Registers `p`; a rotation variant also registers its untagged pair,
/// which is the tree's row for it.
fn register(nc: &NcFile, p: VectorPair) -> Arc<VectorPair> {
    if p.rotate != RotateChoice::Auto {
        nc.add_pair(p.with_rotate(RotateChoice::Auto));
    }
    nc.add_pair(p)
}

fn find_group<'a>(g: &'a torcdf_core::Group, path: &str) -> Option<&'a torcdf_core::Group> {
    if g.path == path {
        return Some(g);
    }
    g.groups.iter().find_map(|s| find_group(s, path))
}

impl eframe::App for TorApp {
    fn on_exit(&mut self) {
        self.remember_open_files();
        if self.last_geom.is_some() {
            self.settings.window = self.last_geom;
        }
        self.save_settings();
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let frame_start = Instant::now();
        self.poll();
        self.tick_playback();
        self.track_window();
        for c in shortcuts::pressed(ui.ctx()) {
            self.command(c);
        }
        let dropped: Vec<PathBuf> = ui.ctx().input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect()
        });
        for p in dropped {
            self.open(p);
        }

        let mut actions = Vec::new();
        // The plot's own PNG button hides while its picture is being taken.
        let capturing = self.shots.iter().any(|s| s.popup.is_none());
        let current_file = self.current_file();
        let extract_ok = current_file.and_then(|id| self.files.iter().find(|f| f.id == id)).is_some_and(|f| f.nc.is_netcdf());
        let recent = &self.settings.recent;
        egui::Panel::top("menu").show(ui, |ui| ui::menu_bar(ui, self.view.as_ref(), recent, current_file, extract_ok, &mut actions));
        if !self.errors.is_empty() {
            egui::Panel::top("errors").show(ui, |ui| ui::error_banners(ui, &self.errors, &mut actions));
        }
        let scroll_tree = std::mem::take(&mut self.scroll_tree);
        egui::Panel::left("tree").default_size(300.0).show(ui, |ui| {
            tree::show(ui, &self.files, &self.opening, self.selection.as_ref(), scroll_tree, &mut actions)
        });
        // A vector pair is no variable of the file: its attributes come from the view.
        let pair_attrs =
            self.view.as_ref().filter(|v| v.vector.is_some() && v.base.is_none()).map(|v| v.var.attrs.clone());
        let attrs_owned = match (&self.selection, pair_attrs) {
            (Some(Selection::Var(_, path)), Some(a)) => Some((path.clone(), a)),
            _ => selected_attrs(&self.files, self.selection.as_ref()),
        };
        let attrs = attrs_owned.as_ref().map(|(t, a)| (t.clone(), &a[..]));
        let var_selected = matches!(self.selection, Some(Selection::Var(..)));
        let offered = self.arrow_candidates();
        let file = self.view.as_ref().map(|v| v.file_id);
        let arrows = self.arrow_layer.as_ref().filter(|l| Some(l.file) == file);
        let input = PanelInput {
            tab: self.panel_tab,
            var_selected,
            attrs,
            arrows: arrows.map(|l| (l, &offered[..])),
        };
        egui::Panel::right("inspector").default_size(300.0).show(ui, |ui| {
            inspector::show(ui, input, &mut self.display, self.view.as_mut(), &mut actions)
        });
        if let Some(v) = &mut self.view
            && !v.slider_dims().is_empty()
        {
            let pb = self.playback.as_ref();
            egui::Panel::bottom("sliders").show(ui, |ui| ui::slider_strip(ui, v, pb, &mut actions));
        }
        // The Extract window's box and span; never in a PNG being taken.
        let marks = if self.shots.is_empty() { self.extract.as_ref().and_then(|f| f.overlay()) } else { None };
        let central = egui::CentralPanel::default().show(ui, |ui| {
            plot_area::show(
                ui,
                self.view.as_mut(),
                &self.display,
                frame.wgpu_render_state(),
                &self.popups,
                capturing,
                marks.as_ref(),
                &mut actions,
            )
        });
        let uploaded = central.inner;
        self.plot_rect = Some(central.response.rect);
        let current = self.view.as_ref().map(|v| (v.file_id, &v.var.dims[..], &v.slider_pos[..]));
        ui::popup::show_all(ui.ctx(), &mut self.popups, current, marks.as_ref(), &mut actions);
        ui::windows::show_all(ui.ctx(), &mut self.windows, &self.settings, &self.renderer, &mut actions);
        if let Some(form) = self.calc.as_mut()
            && let Some(f) = self.files.iter().find(|f| f.id == form.file)
        {
            let vars = calc_names(&f.nc);
            ui::calc::show(ui.ctx(), form, &f.name, &vars, &self.settings.recent_formulas, &mut actions);
        }
        if let Some(form) = self.extract.as_mut() {
            let nc = self.files.iter().find(|f| f.id == form.file).map(|f| f.nc.clone());
            let sliders: Vec<torcdf_core::extract::DimKey> = self
                .view
                .as_ref()
                .filter(|v| v.file_id == form.file)
                .map(|v| v.slider_dims().into_iter().filter_map(|d| v.nc.dim_key(v.plain_path(), &v.var.dims[d])).collect())
                .unwrap_or_default();
            ui::extract::show(ui.ctx(), form, nc.as_deref(), &sliders, &mut actions);
        }
        ui::drop_overlay(ui.ctx());
        // Detail reads start only after the map has planned them for a resting camera.
        if let Some(v) = &mut self.view {
            v.start_detail(&self.service);
        }

        if let Some(took) = self.step_shown(uploaded) {
            self.log.log(format_args!("step shown {took:.1} ms after slider change"));
        }
        if uploaded && !self.first_image_logged {
            self.first_image_logged = true;
            if let Some(v) = &self.view {
                let range = self.display.effective(v.shown.as_ref().map(|s| &s.stats));
                if let (ViewKind::Map, Some(m)) = (v.kind, &v.map) {
                    self.log.log(format_args!(
                        "first map tile uploaded {:.1} ms after request; tile {:?}, camera {:?}, range {range:?}",
                        ms(v.requested_at),
                        m.overview.as_ref().map(|t| t.spec),
                        m.camera,
                    ));
                } else {
                    self.log.log(format_args!(
                        "first image uploaded {:.1} ms after request; grid {:?}, flip_y (north up) {}, range {range:?}",
                        ms(v.requested_at),
                        v.image.grid,
                        v.flip_y(),
                    ));
                }
            }
        }
        for a in actions {
            self.apply(a);
        }
        let took = ms(frame_start);
        if let Some(m) = self.view.as_mut().and_then(|v| v.map.as_mut())
            && let Some(events) = m.events.as_mut()
        {
            for e in std::mem::take(events) {
                self.log.log(e);
            }
        }
        if took > SLOW_FRAME_MS && self.log.enabled() {
            let doing = self.view.as_ref().map_or(String::new(), |v| {
                let activity = v.map.as_ref().map_or("", |m| m.activity);
                format!("view {:?}, map activity \"{activity}\", uploaded {uploaded}", v.kind)
            });
            self.log.log(format_args!("slow frame: {took:.1} ms ({doing})"));
        }
    }
}

#[cfg(test)]
impl TorApp {
    /// An app without a window, with `path` already open as file 1 (opened
    /// on the calling thread). Panics if the file cannot be opened.
    pub fn headless(path: &Path) -> TorApp {
        TorApp::headless_with(path, Settings::default())
    }

    /// As [`TorApp::headless`], with `settings` (never saved).
    pub fn headless_with(path: &Path, settings: Settings) -> TorApp {
        let ctx = egui::Context::default();
        let cache_budget = default_cache_budget_bytes();
        let service = SliceService::new(
            cache_budget,
            DEFAULT_WORKERS,
            Arc::new(|| {}),
        );
        let nc = NcFile::open(path).expect("fixture opens");
        TorApp {
            files: vec![OpenFile { id: 1, name: file_name(path), nc: Arc::new(nc) }],
            next_id: 2,
            opening: Vec::new(),
            dialog: None,
            errors: Vec::new(),
            selection: None,
            view: None,
            display: initial_display(&settings),
            popups: PopupSet::default(),
            arrow_layer: None,
            panel_tab: PanelTab::Plot,
            saves: Vec::new(),
            plot_rect: None,
            shots: Vec::new(),
            png_saves: Vec::new(),
            service,
            cache_budget,
            playback: None,
            slider_changed_at: None,
            log: DebugLog { tx: None, t0: Instant::now(), select: None },
            first_image_logged: false,
            ctx,
            settings,
            settings_path: None,
            save_failed: false,
            last_geom: None,
            windows: Windows::default(),
            renderer: "none".into(),
            scroll_tree: false,
            calc: None,
            extract: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_save_is_skipped_only_for_the_same_pending_target() {
        assert!(!png_pending([], None));
        assert!(png_pending([None], None));
        assert!(png_pending([Some(3), None], None));
        assert!(!png_pending([Some(3)], None));
        assert!(png_pending([Some(3)], Some(3)));
        assert!(!png_pending([None, Some(4)], Some(3)));
    }
    use crate::view::arrow_layer::ArrowMode;
    use std::time::Duration;
    use torcdf_core::fixtures::vectors::{write_arome_like, write_nemo_like};
    use torcdf_core::{AngleRef, AngleUnit, DirConvention, Guessed, SliceRequest, Towards, layer_path};

    #[test]
    fn rotate_toggle_reselects() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        torcdf_core::fixtures::pair_detected(&app.files[0].nc);
        let pair = app.files[0].nc.pairs().remove(0);
        app.apply(Action::Select(Selection::Var(1, pair.path.clone())));
        app.view.as_mut().unwrap().slider_pos[0] = 1;
        app.apply(Action::SetRotate(RotateChoice::Off));
        let Some(Selection::Var(_, path)) = &app.selection else { panic!("no selection") };
        assert!(path.ends_with("#off"), "{path}");
        let v = app.view.as_ref().expect("view");
        assert_eq!(v.var.path, *path);
        assert_eq!(v.slider_pos[0], 1);
        // The tagged variant is registered; the tree still shows one row.
        assert!(app.files[0].nc.pair(path).is_some());
    }

    #[test]
    fn pair_by_hand_and_unpair() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::vectors::write_noclue_renamed(dir.path());
        let mut app = TorApp::headless(&path);
        assert!(app.files[0].nc.pairs().is_empty());
        app.apply(Action::Select(Selection::Var(1, "/h".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!(slot.problem.as_deref(), Some("Pick two variables"));
        assert!(slot.pair.is_none());
        app.apply(Action::PickArrowVar { slot: 0, path: "/aa".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/bb".into() });
        let p = app.files[0].nc.pairs().remove(0);
        assert_eq!((p.east.as_str(), p.north.as_str()), ("/aa", "/bb"));
        assert_eq!(app.selection, Some(Selection::Var(1, "/h".into())));
        assert_eq!(app.view.as_ref().unwrap().var.path, layer_path("/h", &p.path));
        // Its own tree entry, then Unpair: the view goes with the pair.
        app.apply(Action::Select(Selection::Var(1, p.path.clone())));
        app.apply(Action::Unpair { file: 1, path: p.path.clone() });
        assert!(app.files[0].nc.pairs().is_empty());
        assert!(app.view.is_none() && app.selection.is_none());
        // A variable cannot be paired with itself: the slot says why.
        app.apply(Action::Select(Selection::Var(1, "/h".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        app.apply(Action::PickArrowVar { slot: 0, path: "/aa".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/aa".into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert!(slot.problem.is_some() && slot.pair.is_none());
        assert!(app.files[0].nc.pairs().is_empty());
        assert!(app.errors.is_empty());
    }

    const AROME_PAIR: &str = "/wind_speed@wind_direction~from.cwn.deg";

    /// AROME-like file 1, `/wind_speed` coloured, its own magnitude/direction
    /// pair as arrows (picked in the Plot tab).
    fn arome_layer(dir: &Path) -> TorApp {
        let mut app = TorApp::headless(&write_arome_like(dir));
        app.apply(Action::Select(Selection::Var(1, "/wind_speed".into())));
        app.apply(Action::SetArrowMode(ArrowMode::MagDir));
        app.apply(Action::PickArrowVar { slot: 0, path: "/wind_speed".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/wind_direction".into() });
        app
    }

    #[test]
    fn magdir_layer_from_picks() {
        let dir = tempfile::tempdir().unwrap();
        let app = arome_layer(dir.path());
        let v = app.view.as_ref().expect("view");
        assert_eq!(v.var.path, format!("/wind_speed|{AROME_PAIR}"));
        assert_eq!(v.base.as_deref(), Some("/wind_speed"));
        assert_eq!(v.plain_path(), "/wind_speed");
        assert!(v.arrows.as_ref().is_some_and(|a| a.on));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!(slot.mode, ArrowMode::MagDir);
        assert_eq!(slot.pair.as_ref().map(|p| p.path.as_str()), Some(AROME_PAIR));
        assert_eq!(slot.problem, None);
        assert_eq!(app.selection, Some(Selection::Var(1, "/wind_speed".into())));
    }

    /// Each form has its own pre-fill: x & y has none on this file, and
    /// switching to it never pairs speed/direction as east/north.
    #[test]
    fn mode_switch_uses_that_forms_picks() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_arome_like(dir.path()));
        torcdf_core::fixtures::pair_detected(&app.files[0].nc);
        app.apply(Action::Select(Selection::Var(1, "/wind_speed".into())));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!((slot.file, slot.mode), (1, ArrowMode::Off));
        assert_eq!(slot.picks, [Some("/wind_speed".to_string()), Some("/wind_direction".to_string())]);
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.picks, [None, None]);
        assert_eq!(slot.problem.as_deref(), Some("Pick two variables"));
        assert!(slot.pair.is_none());
        assert_eq!(app.view.as_ref().unwrap().var.path, "/wind_speed");
        assert_eq!(app.files[0].nc.pairs().len(), 1);
        app.apply(Action::SetArrowMode(ArrowMode::MagDir));
        assert_eq!(app.view.as_ref().unwrap().var.path, format!("/wind_speed|{AROME_PAIR}"));
        app.apply(Action::SetArrowMode(ArrowMode::Off));
        let v = app.view.as_ref().unwrap();
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/wind_speed", None));
        assert!(v.vector.is_none());
        assert_eq!(app.arrow_layer.as_ref().unwrap().mode, ArrowMode::Off);
    }

    #[test]
    fn layer_over_own_component() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        let pos = app.view.as_ref().unwrap().slider_pos.clone();
        wait_shown(&mut app, &pos);
        let v = app.view.as_ref().unwrap();
        let shown = v.shown.as_ref().expect("shown");
        assert!(shown.slice.vector.is_some(), "the layer carries the arrows");
        let plain = SliceRequest { var: "/wind_speed".into(), ..shown.req.clone() };
        let speed = v.nc.read_slice(&plain).expect("plain read");
        let same = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
        };
        assert!(same(&shown.slice.data, &speed.data), "colours are wind_speed's");
    }

    #[test]
    fn layer_kept_on_same_grid_hidden_on_other() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        torcdf_core::fixtures::pair_detected(&app.files[0].nc);
        let pair = app.files[0].nc.pairs().remove(0);
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        let layered = layer_path("/thetao", &pair.path);
        assert_eq!(app.view.as_ref().unwrap().var.path, layered);
        // uo sits on the u-grid: plain, with the note; the slot stays.
        app.apply(Action::Select(Selection::Var(1, "/uo".into())));
        let v = app.view.as_ref().unwrap();
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/uo", None));
        assert!(v.vector.is_none());
        assert_eq!(v.notice.as_deref(), Some("Arrows hidden: uo is on a different grid"));
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.mode, ArrowMode::EastNorth);
        assert_eq!(slot.pair.as_ref().map(|p| p.path.clone()), Some(pair.path.clone()));
        // Back on the centre grid: the arrows come back.
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        assert_eq!(app.view.as_ref().unwrap().var.path, layered);
    }

    /// A pair whose centre is not the coloured variable's grid is refused.
    #[test]
    fn picks_off_grid_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        let before: Vec<String> = app.files[0].nc.pairs().iter().map(|p| p.path.clone()).collect();
        app.apply(Action::Select(Selection::Var(1, "/uo".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        assert_eq!(app.arrow_layer.as_ref().unwrap().problem.as_deref(), Some("Pick two variables"));
        app.apply(Action::PickArrowVar { slot: 0, path: "/uo".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/vo".into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.problem.as_deref(), Some("These two do not sit on uo's grid"));
        assert!(slot.pair.is_none());
        let after: Vec<String> = app.files[0].nc.pairs().iter().map(|p| p.path.clone()).collect();
        assert_eq!(after, before);
        assert_eq!(app.view.as_ref().unwrap().var.path, "/uo");
    }

    /// Swapping the two drop-downs passes through a failed pick (a
    /// variable with itself); the pair being edited still goes once the
    /// next pick pairs, so the tree holds no stale entry.
    #[test]
    fn failed_pick_still_replaces_edited_pair() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        let nc = app.files[0].nc.clone();
        assert_eq!(nc.pairs().len(), 1);
        app.apply(Action::PickArrowVar { slot: 0, path: "/wind_direction".into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert!(slot.problem.is_some() && slot.pair.is_none());
        assert!(nc.pair(AROME_PAIR).is_some(), "kept while the edit is unfinished");
        app.apply(Action::PickArrowVar { slot: 1, path: "/wind_speed".into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.problem, None);
        let new = slot.pair.clone().expect("paired");
        assert_eq!((new.east.as_str(), new.north.as_str()), ("/wind_direction", "/wind_speed"));
        assert!(nc.pair(AROME_PAIR).is_none(), "the edited pair is gone");
        assert_eq!(nc.pairs().len(), 1);
    }

    #[test]
    fn convention_change_replaces_pair() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        let to = DirConvention {
            towards: Towards::To,
            reference: AngleRef::ClockwiseFromNorth,
            unit: AngleUnit::Degrees,
            guessed: Guessed::default(),
        };
        app.apply(Action::SetConvention(to));
        let nc = app.files[0].nc.clone();
        assert!(nc.pair(AROME_PAIR).is_none(), "the old path is gone");
        let new = app.arrow_layer.as_ref().unwrap().pair.clone().expect("pair");
        assert!(new.path.ends_with("~to.cwn.deg"), "{}", new.path);
        assert!(nc.pair(&new.path).is_some());
        assert_eq!(nc.pairs().len(), 1);
        assert_eq!(app.view.as_ref().unwrap().var.path, layer_path("/wind_speed", &new.path));
    }

    #[test]
    fn other_file_clears_slot() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        let nemo = NcFile::open(&write_nemo_like(dir.path())).unwrap();
        app.files.push(OpenFile { id: 2, name: "nemo.nc".into(), nc: Arc::new(nemo) });
        app.apply(Action::Select(Selection::Var(2, "/thetao".into())));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!((slot.file, slot.mode), (2, ArrowMode::Off));
        assert!(slot.pair.is_none());
        let v = app.view.as_ref().unwrap();
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/thetao", None));
        assert!(v.vector.is_none());
    }

    #[test]
    fn unpair_turns_layer_off() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        app.apply(Action::Unpair { file: 1, path: AROME_PAIR.into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.mode, ArrowMode::Off);
        assert!(slot.pair.is_none());
        let v = app.view.as_ref().expect("the plain view");
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/wind_speed", None));
        assert!(v.vector.is_none());
        assert_eq!(app.selection, Some(Selection::Var(1, "/wind_speed".into())));
    }

    /// Toggling arrows keeps the view kind the user chose.
    #[test]
    fn arrow_toggle_keeps_view_kind() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        assert_eq!(app.view.as_ref().unwrap().kind, ViewKind::Map);
        app.apply(Action::SetKind(ViewKind::Image));
        app.apply(Action::SetArrowMode(ArrowMode::Off));
        let v = app.view.as_ref().unwrap();
        assert_eq!((v.var.path.as_str(), v.kind), ("/wind_speed", ViewKind::Image));
        app.apply(Action::SetArrowMode(ArrowMode::MagDir));
        let v = app.view.as_ref().unwrap();
        assert!(v.base.is_some(), "layered again");
        assert_eq!(v.kind, ViewKind::Image);
    }

    /// On a pair's own entry, switching to a form that has no pair leaves
    /// no arrows drawn.
    #[test]
    fn pair_entry_mode_switch_turns_arrows_off() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_arome_like(dir.path()));
        torcdf_core::fixtures::pair_detected(&app.files[0].nc);
        app.apply(Action::Select(Selection::Var(1, AROME_PAIR.into())));
        let v = app.view.as_ref().expect("view");
        assert!(v.arrows.as_ref().is_some_and(|a| a.on), "the pair entry draws its arrows");
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        assert_eq!(app.arrow_layer.as_ref().unwrap().problem.as_deref(), Some("Pick two variables"));
        let v = app.view.as_ref().expect("view");
        assert!(!v.arrows.as_ref().is_some_and(|a| a.on), "no pair, no arrows");
    }

    /// A rotation change on a layered view keeps it layered over the new
    /// (tagged) pair, which the slot now holds.
    #[test]
    fn rotate_on_layered_view() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        app.apply(Action::SetRotate(RotateChoice::On));
        let want = format!("{AROME_PAIR}#on");
        let slot = app.arrow_layer.as_ref().unwrap();
        let p = slot.pair.as_ref().expect("paired");
        assert_eq!((p.path.as_str(), p.rotate), (want.as_str(), RotateChoice::On));
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.base.as_deref(), Some("/wind_speed"), "still layered");
        assert_eq!(v.var.path, format!("/wind_speed|{want}"));
        assert_eq!(v.vector.as_ref().map(|p| p.path.as_str()), Some(want.as_str()));
    }

    /// Picks refused over one variable pair once a variable they fit is
    /// selected: the stale "do not sit on" message goes.
    #[test]
    fn select_repairs_from_picks() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/uo".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        app.apply(Action::PickArrowVar { slot: 0, path: "/uo".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/vo".into() });
        assert!(app.arrow_layer.as_ref().unwrap().problem.is_some());
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.problem, None);
        let pair = slot.pair.clone().expect("paired");
        assert_eq!((pair.east.as_str(), pair.north.as_str()), ("/uo", "/vo"));
        assert_eq!(app.view.as_ref().unwrap().var.path, layer_path("/thetao", &pair.path));
        assert_eq!(app.selection, Some(Selection::Var(1, "/thetao".into())));
    }

    /// The AROME-like file has a time axis but no vertical one, and its pair
    /// is magnitude/direction; this test wants plain `uo`/`vo` arrows, so it
    /// uses the NEMO-like `thetao`.
    fn derive(app: &TorApp, name: &str, formula: &str) -> String {
        let nc = &app.files[0].nc;
        nc.add_derived(nc.check_formula(name, formula, None, None).unwrap()).path.clone()
    }

    #[test]
    fn derived_is_listed_and_selectable() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        let p = derive(&app, "speed", "sqrt(uo^2 + vo^2)");
        assert!(crate::ui::tree::var_order(&app.files).contains(&Selection::Var(1, p.clone())));
        app.apply(Action::Select(Selection::Var(1, p.clone())));
        let v = app.view.as_ref().expect("view");
        assert_eq!(v.var.path, p);
        assert_eq!(v.kind, ViewKind::Map);
        let (title, attrs) = selected_attrs(&app.files, app.selection.as_ref()).expect("attrs");
        assert_eq!(title, p);
        assert!(attrs.iter().any(|a| a.name == "formula"));
    }

    #[test]
    fn derived_arrow_component() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        let u2 = derive(&app, "u2", "uo*2");
        let v2 = derive(&app, "v2", "vo*2");
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        app.apply(Action::PickArrowVar { slot: 0, path: u2.clone() });
        app.apply(Action::PickArrowVar { slot: 1, path: v2 });
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!(slot.pair.as_ref().map(|p| p.east.clone()), Some(u2), "{:?}", slot.problem);
    }

    /// Types into the open Calculate window and re-checks.
    fn fill(app: &mut TorApp, name: &str, formula: &str, units: &str) {
        let f = app.calc.as_mut().expect("Calculate window open");
        f.name = name.into();
        f.formula = formula.into();
        f.units = units.into();
        app.apply(Action::CalcEdited);
    }

    #[test]
    fn calculate_creates_and_selects() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::OpenCalculate { file: 1, edit: None });
        fill(&mut app, "speed", "sqrt(uo^2+vo^2)", "m s-1");
        assert_eq!(app.calc.as_ref().unwrap().check, Ok("speed (time, lat, lon)  24 × 10 × 20".to_string()));
        app.apply(Action::CreateDerived);
        assert!(app.calc.is_none(), "the window closes");
        assert!(app.files[0].nc.derived("/ocean/speed").is_some());
        assert_eq!(app.selection, Some(Selection::Var(1, "/ocean/speed".into())));
        let r = &app.settings.recent_formulas[0];
        assert_eq!((r.name.as_str(), r.formula.as_str(), r.units.as_str()), ("speed", "sqrt(uo^2+vo^2)", "m s-1"));
        // Recent fills the form.
        app.apply(Action::OpenCalculate { file: 1, edit: None });
        app.apply(Action::UseRecentFormula(0));
        assert_eq!(app.calc.as_ref().unwrap().formula, "sqrt(uo^2+vo^2)");
        assert_eq!(app.calc.as_ref().unwrap().check, Err("speed already exists".to_string()));
    }

    #[test]
    fn bad_formula_keeps_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::OpenCalculate { file: 1, edit: None });
        fill(&mut app, "x", "uo +", "");
        assert!(app.calc.as_ref().unwrap().check.is_err());
        app.apply(Action::CreateDerived);
        assert!(app.calc.is_some());
        assert!(app.files[0].nc.derived_vars().is_empty());
    }

    #[test]
    fn ctrl_k_without_file_does_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::Select(Selection::File(1)));
        app.command(Command::Calculate);
        assert_eq!(app.calc.as_ref().map(|c| c.file), Some(1));
        app.apply(Action::CloseCalculate);
        app.apply(Action::CloseFile(1));
        app.command(Command::Calculate);
        assert!(app.calc.is_none());
    }

    #[test]
    fn edit_refreshes_view_and_closes_popups() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::OpenCalculate { file: 1, edit: None });
        fill(&mut app, "s", "uo*1", "");
        app.apply(Action::CreateDerived);
        wait_shown(&mut app, &[0, 0, 0]);
        app.apply(Action::OpenPopup { iy: 1, ix: 1, lonlat: None });
        assert_eq!(app.popups.list.len(), 1);
        app.apply(Action::OpenCalculate { file: 1, edit: Some("/ocean/s".into()) });
        assert_eq!(app.calc.as_ref().unwrap().formula, "uo*1");
        fill(&mut app, "s", "uo*2", "");
        app.apply(Action::CreateDerived);
        assert!(app.popups.list.is_empty(), "pop-ups of the old formula close");
        wait_shown(&mut app, &[0, 0, 0]);
        let v = app.view.as_ref().unwrap();
        let shown = v.shown.as_ref().unwrap();
        let uo = app.files[0].nc.read_slice(&SliceRequest { var: "/ocean/uo".into(), ..shown.req.clone() }).unwrap();
        for (a, b) in shown.slice.data.iter().zip(&uo.data) {
            assert!((a.is_nan() && b.is_nan()) || (a - 2.0 * b).abs() < 1e-5, "{a} vs 2×{b}");
        }
    }

    #[test]
    fn recreated_name_shows_new_values() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::OpenCalculate { file: 1, edit: None });
        fill(&mut app, "s", "uo*1", "");
        app.apply(Action::CreateDerived);
        wait_shown(&mut app, &[0, 0, 0]);
        app.apply(Action::DeleteDerived { file: 1, path: "/ocean/s".into() });
        app.apply(Action::OpenCalculate { file: 1, edit: None });
        fill(&mut app, "s", "uo*2", "");
        app.apply(Action::CreateDerived);
        wait_shown(&mut app, &[0, 0, 0]);
        let shown = app.view.as_ref().unwrap().shown.as_ref().unwrap();
        let uo = app.files[0].nc.read_slice(&SliceRequest { var: "/ocean/uo".into(), ..shown.req.clone() }).unwrap();
        for (a, b) in shown.slice.data.iter().zip(&uo.data) {
            assert!((a.is_nan() && b.is_nan()) || (a - 2.0 * b).abs() < 1e-5, "{a} vs 2×{b}");
        }
    }

    #[test]
    fn delete_refused_while_used() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        let a = derive(&app, "a", "uo*2");
        let b = derive(&app, "b", "a+1");
        app.apply(Action::DeleteDerived { file: 1, path: a.clone() });
        assert_eq!(app.errors, ["a is used by b"]);
        assert!(app.files[0].nc.derived(&a).is_some());
        app.apply(Action::Select(Selection::Var(1, b.clone())));
        app.apply(Action::DeleteDerived { file: 1, path: b.clone() });
        assert!(app.files[0].nc.derived(&b).is_none());
        assert!(app.view.is_none() && app.selection.is_none());
    }

    #[test]
    fn nothing_is_paired_until_the_user_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        // The tree lists registered pairs only, so no vector row either.
        assert!(app.files[0].nc.pairs().is_empty());
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!(slot.mode, ArrowMode::Off);
        assert_eq!(slot.picks, [None, None]);
    }

    #[test]
    fn popup_on_layer_reads_plain_series() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        torcdf_core::fixtures::pair_detected(&app.files[0].nc);
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        assert!(app.view.as_ref().unwrap().base.is_some(), "layered");
        app.apply(Action::OpenPopup { iy: 1, ix: 1, lonlat: None });
        assert_eq!(app.popups.list.len(), 1);
        let p = &app.popups.list[0];
        assert_eq!(p.var_path, "/thetao");
        assert!(!p.has_components());
        let t0 = Instant::now();
        while app.popups.list[0].values.iter().any(Option::is_none) && app.popups.list[0].error.is_none() {
            assert!(t0.elapsed().as_secs() < 20, "the series never arrived");
            std::thread::sleep(Duration::from_millis(2));
            app.popups.poll();
        }
        let p = &app.popups.list[0];
        assert_eq!(p.error, None);
        // thetao = 10 + 0.01 (p + q) at every time step.
        assert_eq!(p.values.len(), 2);
        assert!(p.values.iter().all(|v| v.is_some_and(|x| (x - 10.02).abs() < 1e-4)), "{:?}", p.values);
    }

    fn wait_opened(app: &mut TorApp) {
        let start = Instant::now();
        while !app.opening.is_empty() {
            assert!(start.elapsed() < Duration::from_secs(20), "open timed out");
            std::thread::sleep(Duration::from_millis(10));
            app.poll();
        }
    }

    #[test]
    fn new_view_uses_plot_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Settings::default();
        s.plot.projection = crate::map::projection::Projection::Mercator;
        s.plot.coast = false;
        let mut app = TorApp::headless_with(&torcdf_core::fixtures::write_nc4(dir.path()), s);
        app.apply(Action::Select(Selection::Var(1, "/ocean/uo".into())));
        let m = app.view.as_ref().unwrap().map.as_ref().expect("uo is a map");
        assert_eq!(m.camera.proj, crate::map::projection::Projection::Mercator);
        assert!(!m.coast);
        assert!(m.borders && m.grid);
    }

    #[test]
    fn colormap_default_from_settings() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Settings::default();
        s.plot.colormap = crate::render::colormap::Colormap::Topo;
        let app = TorApp::headless_with(&torcdf_core::fixtures::write_nc4(dir.path()), s);
        assert_eq!(app.display.colormap, crate::render::colormap::Colormap::Topo);
    }

    #[test]
    fn reopen_last_skips_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::write_nc4(dir.path());
        let s = Settings {
            reopen_last: true,
            last_open: vec![path.clone(), PathBuf::from("C:/nope/x.nc")],
            ..Settings::default()
        };
        let mut app = TorApp::headless_with(&path, s);
        app.start_files(None);
        assert_eq!(app.opening.len(), 1);
        assert_eq!(app.errors.len(), 1);
        assert!(app.errors[0].contains("x.nc"), "{:?}", app.errors);
        wait_opened(&mut app); // netCDF must not be mid-open at process exit
    }

    #[test]
    fn reopen_last_off_opens_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::write_nc4(dir.path());
        let s = Settings { last_open: vec![path.clone()], ..Settings::default() };
        let mut app = TorApp::headless_with(&path, s);
        app.start_files(None);
        assert!(app.opening.is_empty() && app.errors.is_empty());
    }

    #[test]
    fn command_line_path_wins_over_reopen_last() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::write_nc4(dir.path());
        let s = Settings { reopen_last: true, last_open: vec![PathBuf::from("C:/nope/x.nc")], ..Settings::default() };
        let mut app = TorApp::headless_with(&path, s);
        app.start_files(Some(path.clone()));
        assert_eq!(app.opening.len(), 1);
        assert!(app.errors.is_empty());
        wait_opened(&mut app);
    }

    #[test]
    fn open_success_adds_recent() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::write_nc4(dir.path());
        let mut app = TorApp::headless_with(&path, Settings::default());
        app.open(path.clone());
        wait_opened(&mut app);
        assert_eq!(app.settings.recent.first(), Some(&path));
    }

    #[test]
    fn open_failure_removes_recent() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::write_nc4(dir.path());
        let gone = dir.path().join("gone.nc");
        let mut s = Settings::default();
        s.push_recent(&gone);
        let mut app = TorApp::headless_with(&path, s);
        app.open(gone.clone());
        wait_opened(&mut app);
        assert!(app.settings.recent.is_empty());
        assert_eq!(app.errors.len(), 1);
    }

    #[test]
    fn last_open_lists_files_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::write_nc4(dir.path());
        let mut app = TorApp::headless_with(&path, Settings::default());
        app.remember_open_files();
        assert_eq!(app.settings.last_open, vec![path]);
    }

    #[test]
    fn set_look_saves_and_applies() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless_with(&torcdf_core::fixtures::write_nc4(dir.path()), Settings::default());
        app.settings_path = Some(dir.path().join("TorCDF").join("settings.json"));
        app.apply(Action::SetLook(crate::settings::Look::Dark));
        assert_eq!(app.settings.look, crate::settings::Look::Dark);
        assert_eq!(app.ctx.options(|o| o.theme_preference), egui::ThemePreference::Dark);
        let (saved, _) = Settings::load_from(app.settings_path.as_ref().unwrap());
        assert_eq!(saved.look, crate::settings::Look::Dark);
    }

    #[test]
    fn set_plot_defaults_leaves_current_view() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let d = PlotDefaults { projection: crate::map::projection::Projection::Mercator, ..Default::default() };
        app.apply(Action::SetPlotDefaults(d.clone()));
        assert_eq!(app.settings.plot, d);
        let m = app.view.as_ref().unwrap().map.as_ref().unwrap();
        assert_eq!(m.camera.proj, crate::map::projection::Projection::Equirectangular);
    }

    #[test]
    fn set_recent_max_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless_with(&torcdf_core::fixtures::write_nc4(dir.path()), Settings::default());
        for i in 0..5 {
            app.settings.push_recent(Path::new(&format!("C:/d/{i}.nc")));
        }
        app.apply(Action::SetRecentMax(0));
        assert_eq!(app.settings.recent_max, 0);
        assert!(app.settings.recent.is_empty());
    }

    #[test]
    fn reset_settings_keeps_recent() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless_with(&torcdf_core::fixtures::write_nc4(dir.path()), Settings::default());
        app.settings.push_recent(Path::new("C:/d/a.nc"));
        app.apply(Action::SetLook(crate::settings::Look::Dark));
        app.apply(Action::SetReopenLast(true));
        app.apply(Action::ResetSettings);
        assert_eq!(app.settings.look, crate::settings::Look::Light);
        assert!(!app.settings.reopen_last);
        assert_eq!(app.settings.recent.len(), 1);
        assert_eq!(app.ctx.options(|o| o.theme_preference), egui::ThemePreference::Light);
    }

    #[test]
    fn show_window_opens_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless_with(&torcdf_core::fixtures::write_nc4(dir.path()), Settings::default());
        app.apply(Action::ShowWindow(ui::windows::Win::About));
        assert!(app.windows.about);
        assert!(!app.windows.settings);
    }

    #[test]
    fn step_keys_move_playback_dim() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let dim = app.playback.as_ref().unwrap().dim;
        let len = app.view.as_ref().unwrap().var.shape[dim];
        assert!(len > 1);
        app.command(Command::StepForward);
        assert_eq!(app.view.as_ref().unwrap().slider_pos[dim], 1);
        app.command(Command::LastStep);
        assert_eq!(app.view.as_ref().unwrap().slider_pos[dim], len - 1);
        app.command(Command::StepForward);
        assert_eq!(app.view.as_ref().unwrap().slider_pos[dim], len - 1);
        app.command(Command::FirstStep);
        assert_eq!(app.view.as_ref().unwrap().slider_pos[dim], 0);
        app.command(Command::StepBack);
        assert_eq!(app.view.as_ref().unwrap().slider_pos[dim], 0);
    }

    #[test]
    fn keys_without_view_do_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        for c in [Command::StepForward, Command::LastStep, Command::PlayPause, Command::Fit, Command::ExportPng] {
            app.command(c);
        }
        assert!(app.errors.is_empty());
        assert!(app.view.is_none());
    }

    #[test]
    fn next_var_follows_tree_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        let order = tree::var_order(&app.files);
        assert!(order.len() >= 2);
        app.command(Command::NextVar);
        assert_eq!(app.selection.as_ref(), Some(&order[0]));
        app.command(Command::NextVar);
        assert_eq!(app.selection.as_ref(), Some(&order[1]));
        app.command(Command::PrevVar);
        assert_eq!(app.selection.as_ref(), Some(&order[0]));
    }

    #[test]
    fn ctrl_w_closes_current_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        app.command(Command::CloseFile);
        assert!(app.files.is_empty());
        assert!(app.view.is_none());
    }

    #[test]
    fn f1_and_settings_open_windows() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.command(Command::ShortcutHelp);
        app.command(Command::Settings);
        assert!(app.windows.shortcuts && app.windows.settings);
    }

    #[test]
    fn window_geometry_undoes_text_zoom() {
        // egui reports the window in zoomed points; the builder at start-up
        // works at zoom 1, so geometry is saved at zoom 1.
        let r = |x: f32, y: f32, w: f32, h: f32| Some(egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h)));
        let g = window_geom(r(80.0, 40.0, 820.0, 580.0), r(84.0, 72.0, 800.0, 560.0), false, 1.25, None).unwrap();
        assert_eq!(g, WindowGeom { pos: Some([100.0, 50.0]), size: [1000.0, 700.0], maximized: false });
        // Maximised: the size from before stays.
        let m = window_geom(r(0.0, 0.0, 2000.0, 1000.0), r(0.0, 0.0, 2000.0, 1000.0), true, 1.0, Some(g)).unwrap();
        assert_eq!(m, WindowGeom { maximized: true, ..g });
        assert_eq!(window_geom(None, None, true, 1.0, None), None);
    }

    #[test]
    fn version_info_has_version() {
        let s = ui::windows::version_info("Test GPU");
        assert!(s.contains(env!("CARGO_PKG_VERSION")));
        assert!(s.contains("Test GPU"));
        assert!(s.contains("Windows"));
    }

    fn uo_app(dir: &Path) -> TorApp {
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir));
        app.apply(Action::Select(Selection::Var(1, "/ocean/uo".into())));
        app
    }

    #[test]
    fn map_settings_survive_variable_switch() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let default_proj = app.view.as_ref().unwrap().map.as_ref().expect("uo is a map").camera.proj;
        {
            let m = app.view.as_mut().unwrap().map.as_mut().unwrap();
            m.set_projection(crate::map::projection::Projection::Mercator);
            m.coast = !m.coast;
            m.borders = !m.borders;
            m.grid = !m.grid;
        }
        let before = {
            let m = app.view.as_ref().unwrap().map.as_ref().unwrap();
            (m.camera, m.coast, m.borders, m.grid)
        };
        assert_ne!(before.0.proj, default_proj);
        app.apply(Action::Select(Selection::Var(1, "/ocean/vo".into())));
        let m = app.view.as_ref().unwrap().map.as_ref().expect("vo is a map");
        assert_eq!((m.camera, m.coast, m.borders, m.grid), before);
    }

    #[test]
    fn jump_sets_matching_slider() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let before = app.view.as_ref().unwrap().requested_at;
        std::thread::sleep(std::time::Duration::from_millis(5));
        app.apply(Action::JumpTo { file: 2, dim_name: "time".into(), index: 9 });
        assert_eq!(app.view.as_ref().unwrap().slider_pos[0], 0);
        app.apply(Action::JumpTo { file: 1, dim_name: "nope".into(), index: 9 });
        assert_eq!(app.view.as_ref().unwrap().slider_pos[0], 0);
        assert_eq!(app.view.as_ref().unwrap().requested_at, before);
        app.apply(Action::JumpTo { file: 1, dim_name: "time".into(), index: 9 });
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.slider_pos[0], 9);
        assert!(v.requested_at > before, "the view was not refreshed");
        // Past the end: ignored.
        app.apply(Action::JumpTo { file: 1, dim_name: "time".into(), index: 999 });
        assert_eq!(app.view.as_ref().unwrap().slider_pos[0], 9);
    }

    #[test]
    fn play_locks_range() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let t0 = Instant::now();
        while app.view.as_ref().unwrap().shown.is_none() {
            assert!(t0.elapsed().as_secs() < 20, "the first slice never arrived");
            std::thread::sleep(std::time::Duration::from_millis(2));
            app.poll();
        }
        let stats = app.view.as_ref().unwrap().shown.as_ref().map(|s| s.stats);
        let range = app.display.effective(stats.as_ref()).expect("a range");
        assert!(!app.display.lock);
        app.apply(Action::Play);
        assert!(app.display.lock);
        assert_eq!(app.display.range, range);
        assert!(app.playback.as_ref().is_some_and(|p| p.playing));
        app.apply(Action::Pause);
        assert!(!app.playback.as_ref().unwrap().playing);
        // Another variable drops the playback.
        app.apply(Action::Play);
        app.apply(Action::Select(Selection::Var(1, "/ocean/vo".into())));
        assert!(app.playback.as_ref().is_none_or(|p| !p.playing));
    }

    fn wait_shown(app: &mut TorApp, pos: &[usize]) {
        let t0 = Instant::now();
        while !app.view.as_ref().unwrap().shows_position(pos) {
            assert!(t0.elapsed().as_secs() < 20, "the slice for {pos:?} never arrived");
            std::thread::sleep(std::time::Duration::from_millis(2));
            app.poll();
        }
    }

    #[test]
    fn step_log_times_the_step_moved_to() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        wait_shown(&mut app, &[0, 0, 0]);
        // The slider moved to step 1, but the upload is of step 0 (a read
        // that finished late): not logged, and the change stays pending.
        app.slider_changed_at = Some((Instant::now(), vec![1, 0, 0]));
        assert_eq!(app.step_shown(true), None);
        assert!(app.slider_changed_at.is_some());
        app.apply(Action::JumpTo { file: 1, dim_name: "time".into(), index: 1 });
        wait_shown(&mut app, &[1, 0, 0]);
        assert_eq!(app.step_shown(false), None, "no upload this frame");
        assert!(app.step_shown(true).is_some(), "step 1 is uploaded");
        assert!(app.slider_changed_at.is_none());
        assert_eq!(app.step_shown(true), None, "logged once");
    }

    #[test]
    fn open_popup_action_places() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        app.apply(Action::OpenPopup { iy: 4, ix: 7, lonlat: None });
        assert_eq!(app.popups.list.len(), 1);
        let first = app.popups.list[0].id;
        app.apply(Action::OpenPopup { iy: 5, ix: 6, lonlat: None });
        assert_eq!(app.popups.list.len(), 1);
        assert_ne!(app.popups.list[0].id, first);
        app.popups.list[0].pinned = true;
        app.apply(Action::OpenPopup { iy: 1, ix: 1, lonlat: None });
        assert_eq!(app.popups.list.len(), 2);
        // Off the grid: nothing opens, a passing message appears.
        app.apply(Action::OpenPopup { iy: 99, ix: 99, lonlat: None });
        assert_eq!(app.popups.list.len(), 2);
        assert!(app.view.as_ref().unwrap().flash.is_some());
        let id = app.popups.list[1].id;
        app.apply(Action::ClosePopup(id));
        assert_eq!(app.popups.list.len(), 1);
    }

    #[test]
    fn failed_popup_click_keeps_the_notice() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let why = "Shown as an image: this variable cannot be drawn as a map because of a test.";
        app.view.as_mut().unwrap().notice = Some(why.into());
        let clicked = Instant::now();
        app.apply(Action::OpenPopup { iy: 99, ix: 99, lonlat: None });
        let v = app.view.as_mut().unwrap();
        assert_eq!(v.notice.as_deref(), Some(why));
        let (text, _) = v.flash.clone().expect("click error shown");
        assert_eq!(text, "That point is outside the grid.");
        // Still there within 4 s; gone after, and the notice stays.
        assert!(v.expire_flash(clicked + Duration::from_millis(3900)).is_some());
        assert!(v.flash.is_some());
        assert!(v.expire_flash(Instant::now() + NOTICE_TIME).is_none());
        assert!(v.flash.is_none());
        assert_eq!(v.notice.as_deref(), Some(why));
    }

    /// `profile.nc`'s `t(depth, y, x)`, `t[k, j, i] = k * 100 + j * 10 + i`,
    /// shown as an image on (y, x).
    fn profile_app(dir: &Path) -> TorApp {
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_profile(dir));
        app.apply(Action::Select(Selection::Var(1, "/t".into())));
        app
    }

    /// The image on show, `ny` rows of `nx` values.
    fn shown_image(app: &TorApp) -> (usize, usize, Vec<f32>) {
        let v = app.view.as_ref().unwrap();
        let s = v.shown.as_ref().expect("a slice on show");
        let grid = plot_area::plot_grid(&s.slice.shape, v.axes.expect("axes"));
        (grid.ny, grid.nx, plot_area::image_data(&s.slice.data, &grid).into_owned())
    }

    #[test]
    fn swap_axes_transposes() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = profile_app(dir.path());
        assert_eq!(app.view.as_ref().unwrap().kind, ViewKind::Image);
        app.apply(Action::JumpTo { file: 1, dim_name: "depth".into(), index: 1 });
        wait_shown(&mut app, &[1, 0, 0]);
        let (ny, nx, before) = shown_image(&app);
        assert_eq!((ny, nx), (3, 4));
        assert_eq!(before[2 * 4 + 3], 123.0);
        app.apply(Action::SetAxes { y: 2, x: 1 });
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.axes, Some(AxisChoice { y: 2, x: 1 }));
        assert_eq!(v.slider_dims(), vec![0]);
        assert_eq!(v.slider_pos[0], 1, "the depth slider stays");
        wait_shown(&mut app, &[1, 0, 0]);
        let (ny, nx, after) = shown_image(&app);
        assert_eq!((ny, nx), (4, 3));
        for r in 0..4 {
            for c in 0..3 {
                assert_eq!(after[r * 3 + c], before[c * 4 + r], "cell ({r}, {c})");
            }
        }
    }

    #[test]
    fn axis_onto_slider_dim() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = profile_app(dir.path());
        app.apply(Action::JumpTo { file: 1, dim_name: "depth".into(), index: 2 });
        app.apply(Action::SetAxes { y: 0, x: 2 });
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.axes, Some(AxisChoice { y: 0, x: 2 }));
        assert_eq!(v.slider_dims(), vec![1], "y gets a slider, depth has none");
        assert_eq!(v.slider_pos[1], 0);
        wait_shown(&mut app, &[0, 0, 0]);
        let (ny, nx, img) = shown_image(&app);
        assert_eq!((ny, nx), (5, 4));
        for r in 0..5 {
            for c in 0..4 {
                assert_eq!(img[r * 4 + c], (r * 100 + c) as f32, "cell ({r}, {c})");
            }
        }
    }

    #[test]
    fn same_dim_on_both_axes_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = profile_app(dir.path());
        let before = app.view.as_ref().unwrap().requested_at;
        app.apply(Action::SetAxes { y: 2, x: 2 });
        app.apply(Action::SetAxes { y: 1, x: 7 });
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.axes, Some(AxisChoice { y: 1, x: 2 }));
        assert_eq!(v.requested_at, before, "nothing re-read");
    }

    #[test]
    fn axes_not_choosable_for_vectors_or_maps() {
        let dir = tempfile::tempdir().unwrap();
        // A pair's own view keeps its horizontal axes.
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        torcdf_core::fixtures::pair_detected(&app.files[0].nc);
        let pair = app.files[0].nc.pairs().remove(0);
        app.apply(Action::Select(Selection::Var(1, pair.path.clone())));
        let v = app.view.as_ref().unwrap();
        assert!(!v.axes_choosable());
        let axes = v.axes;
        app.apply(Action::SetAxes { y: 0, x: 2 });
        assert_eq!(app.view.as_ref().unwrap().axes, axes);
        // So does a layered view, in any view kind.
        let mut app = arome_layer(dir.path());
        app.apply(Action::SetKind(ViewKind::Image));
        let v = app.view.as_ref().unwrap();
        assert!(v.base.is_some());
        assert!(!v.axes_choosable());
        // A map keeps lat/lon; its image and table views can choose.
        let mut app = uo_app(dir.path());
        assert!(!app.view.as_ref().unwrap().axes_choosable());
        app.apply(Action::SetKind(ViewKind::Image));
        assert!(app.view.as_ref().unwrap().axes_choosable());
        app.apply(Action::SetKind(ViewKind::Table));
        assert!(app.view.as_ref().unwrap().axes_choosable());
        // Plain image: yes; one long dimension (a line): no.
        let app = profile_app(dir.path());
        assert!(app.view.as_ref().unwrap().axes_choosable());
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_long_series(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/v".into())));
        assert!(!app.view.as_ref().unwrap().axes_choosable());
    }

    /// Arrows on over chosen axes: the layered view is on the grid's axes,
    /// fitted afresh rather than at the old axes' pan and zoom.
    #[test]
    fn arrows_over_chosen_axes_refit() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_arome_like(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/wind_speed".into())));
        app.apply(Action::SetKind(ViewKind::Image));
        app.apply(Action::SetAxes { y: 0, x: 3 });
        app.view.as_mut().unwrap().image.user_moved = true;
        app.apply(Action::SetArrowMode(ArrowMode::MagDir));
        app.apply(Action::PickArrowVar { slot: 0, path: "/wind_speed".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/wind_direction".into() });
        let v = app.view.as_ref().unwrap();
        assert!(v.base.is_some(), "layered");
        assert_eq!(v.axes, Some(AxisChoice { y: 2, x: 3 }));
        assert!(!v.image.user_moved);
    }

    /// Back on the map, lat/lon are the axes again.
    #[test]
    fn map_view_restores_lat_lon_axes() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let geo = app.view.as_ref().unwrap().axes;
        assert_eq!(geo, Some(AxisChoice { y: 1, x: 2 }));
        app.apply(Action::JumpTo { file: 1, dim_name: "time".into(), index: 5 });
        app.apply(Action::SetKind(ViewKind::Image));
        app.apply(Action::SetAxes { y: 0, x: 2 });
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.slider_dims(), vec![1]);
        assert!(!v.on_geo_axes());
        app.apply(Action::SetKind(ViewKind::Map));
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.axes, geo);
        assert!(v.on_geo_axes());
        assert_eq!(v.slider_dims(), vec![0]);
        assert_eq!(v.slider_pos[0], 0, "time left an axis: back at 0");
    }

    // ---- Extract -------------------------------------------------------

    /// Polls until the Extract form has its coordinates and no save runs.
    fn wait_extract(app: &mut TorApp) {
        let start = Instant::now();
        while app.extract.as_ref().is_some_and(|f| f.busy()) {
            assert!(start.elapsed() < Duration::from_secs(20), "extract timed out");
            std::thread::sleep(Duration::from_millis(10));
            app.poll();
        }
    }

    fn nc4_extract(dir: &Path) -> TorApp {
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir));
        app.apply(Action::Select(Selection::Var(1, "/ocean/uo".into())));
        app.apply(Action::OpenExtract { file: 1, var: None });
        wait_extract(&mut app);
        app
    }

    fn region(app: &mut TorApp, s: &str, n: &str, w: &str, e: &str) {
        let r = app.extract.as_mut().unwrap().region.as_mut().expect("region rows");
        (r.south, r.north, r.west, r.east) = (s.into(), n.into(), w.into(), e.into());
        app.apply(Action::ExtractEdited);
    }

    #[test]
    fn extract_defaults_from_view() {
        let dir = tempfile::tempdir().unwrap();
        let app = nc4_extract(dir.path());
        let f = app.extract.as_ref().expect("form open");
        assert_eq!(f.ticked.iter().collect::<Vec<_>>(), ["/ocean/uo"]);
        let time = &f.dims[0];
        assert_eq!(time.label, "time");
        assert_eq!((time.from.as_str(), time.to.as_str()), ("2000-01-01 00:00", "2000-01-01 23:00"));
        assert_eq!(f.dims.len(), 1, "lat/lon are the region");
        let r = f.region.as_ref().expect("region");
        assert_eq!((r.south.as_str(), r.north.as_str(), r.west.as_str(), r.east.as_str()), ("-9", "9", "0", "19"));
        let plan = f.check.as_ref().expect("check ok");
        assert_eq!(plan.vars[0].count, [24, 10, 20]);
    }

    #[test]
    fn typed_ranges_snap_and_check() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = nc4_extract(dir.path());
        app.extract.as_mut().unwrap().dims[0].from = "2000-01-01 02:30".into();
        app.apply(Action::ExtractEdited);
        let f = app.extract.as_ref().unwrap();
        assert_eq!(f.dims[0].from, "2000-01-01 03:00", "snapped to the next step");
        assert_eq!(f.check.as_ref().unwrap().vars[0].count[0], 21);

        region(&mut app, "50", "60", "0", "19");
        assert_eq!(app.extract.as_ref().unwrap().check.as_ref().unwrap_err(), "the selection holds no cells");
        region(&mut app, "-9", "9", "15", "2");
        assert_eq!(app.extract.as_ref().unwrap().check.as_ref().unwrap_err(), torcdf_core::extract::range::SEAM);
        region(&mut app, "-3", "3", "4", "6");
        let plan = app.extract.as_ref().unwrap().check.clone().unwrap();
        assert_eq!(plan.vars[0].count, [21, 4, 3]);
        assert_eq!(app.extract.as_ref().unwrap().summary(), "time 2000-01-01 03:00..2000-01-01 23:00, lat -3..3, lon 4..6");
    }

    #[test]
    fn save_writes_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = nc4_extract(dir.path());
        region(&mut app, "-3", "3", "4", "6");
        let out = dir.path().join("cut.nc");
        app.start_extract(out.clone());
        wait_extract(&mut app);
        let status = app.extract.as_ref().unwrap().status.clone().unwrap_or_default();
        assert!(status.starts_with("Saved to"), "{status}");
        let o = NcFile::open(&out).unwrap();
        assert_eq!(o.meta.var("/ocean/uo").unwrap().shape, [24, 4, 3]);
        let h = o.meta.root.attrs.iter().find(|a| a.name == "history").and_then(|a| a.value.as_str()).unwrap();
        assert!(h.contains("TorCDF: extracted from nc4.nc: lat -3..3, lon 4..6"), "{h}");
    }

    #[test]
    fn save_refuses_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = nc4_extract(dir.path());
        let src = app.files[0].nc.meta.path.clone();
        app.start_extract(src.clone());
        wait_extract(&mut app);
        let status = app.extract.as_ref().unwrap().status.clone().unwrap_or_default();
        assert!(status.starts_with("Not saved"), "{status}");
        assert!(NcFile::open(&src).is_ok());
    }

    #[test]
    fn closing_file_does_not_stop_save() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = nc4_extract(dir.path());
        let out = dir.path().join("whole.nc");
        app.start_extract(out.clone());
        app.apply(Action::CloseFile(1));
        wait_extract(&mut app);
        let status = app.extract.as_ref().unwrap().status.clone().unwrap_or_default();
        assert!(status.starts_with("Saved to"), "{status}");
        assert!(NcFile::open(&out).is_ok());
    }

    #[test]
    fn ctrl_shift_e_opens_extract_for_netcdf_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::Select(Selection::File(1)));
        app.command(Command::Extract);
        assert_eq!(app.extract.as_ref().map(|f| f.file), Some(1));
        let mut h5 = TorApp::headless(&torcdf_core::fixtures::hdf5::write_hidden_h5(dir.path()));
        h5.apply(Action::Select(Selection::File(1)));
        h5.command(Command::Extract);
        assert!(h5.extract.is_none());
    }

    #[test]
    fn drawn_region_fills_form() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/ocean/uo".into())));
        app.apply(Action::ExtractRegionDrawn { south: -10.0, north: 10.0, west: 5.0, east: 25.0 });
        wait_extract(&mut app);
        let f = app.extract.as_ref().expect("the window opens");
        let r = f.region.as_ref().expect("region");
        assert_eq!((r.south.as_str(), r.north.as_str(), r.west.as_str(), r.east.as_str()), ("-10", "10", "5", "25"));
        assert_eq!(f.check.as_ref().expect("check ok").vars[0].count, [24, 10, 15]);
    }

    #[test]
    fn dragged_time_fills_form() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/ocean/uo".into())));
        app.apply(Action::ExtractTimeDragged { file: 1, from: 7.0, to: 3.0 });
        wait_extract(&mut app);
        let f = app.extract.as_ref().expect("the window opens");
        assert_eq!((f.dims[0].from.as_str(), f.dims[0].to.as_str()), ("2000-01-01 03:00", "2000-01-01 07:00"));
        assert_eq!(f.check.as_ref().expect("check ok").vars[0].count[0], 5);
    }

    #[test]
    fn index_box_on_image() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_bare(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/data".into())));
        assert_eq!(app.view.as_ref().unwrap().kind, ViewKind::Image);
        app.apply(Action::ExtractIndexBoxDrawn { y: (1, 2), x: (0, 3) });
        wait_extract(&mut app);
        let f = app.extract.as_ref().expect("the window opens");
        let rows: Vec<(&str, &str, &str)> = f.dims.iter().map(|r| (r.label.as_str(), r.from.as_str(), r.to.as_str())).collect();
        assert_eq!(rows, [("a", "1", "2"), ("b", "0", "3")]);
        assert_eq!(f.check.as_ref().expect("check ok").vars[0].count, [2, 4]);
    }

    #[test]
    fn drawn_box_on_lat_lon_image_sets_region() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = nc4_extract(dir.path());
        // lat index 2..=4 is -5..-1; lon index 5..=7 is 5..7.
        app.apply(Action::SetKind(ViewKind::Image));
        app.apply(Action::ExtractIndexBoxDrawn { y: (2, 4), x: (5, 7) });
        let r = app.extract.as_ref().unwrap().region.clone().unwrap();
        assert_eq!((r.south.as_str(), r.north.as_str(), r.west.as_str(), r.east.as_str()), ("-5", "-1", "5", "7"));
    }

    /// Polls until pop-up `id`'s series is read and any netCDF save is done.
    fn wait_popup(app: &mut TorApp, id: u64) {
        let start = Instant::now();
        loop {
            app.poll();
            let p = app.popups.list.iter().find(|p| p.id == id).expect("pop-up open");
            let (done, total) = p.progress();
            if total > 0 && done == total && !p.saving() {
                return;
            }
            assert!(start.elapsed() < Duration::from_secs(20), "pop-up timed out");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn popup_netcdf_is_one_cell_series() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        app.apply(Action::OpenPopup { iy: 2, ix: 5, lonlat: Some((5.0, -5.0)) });
        let id = app.popups.list[0].id;
        wait_popup(&mut app, id);
        let out = dir.path().join("point.nc");
        app.start_popup_extract(id, out.clone());
        wait_popup(&mut app, id);
        let p = app.popups.list.iter().find(|p| p.id == id).unwrap();
        assert!(p.nc_status.as_deref().is_some_and(|s| s.starts_with("Saved to")), "{:?}", p.nc_status);
        let o = NcFile::open(&out).unwrap();
        assert_eq!(o.meta.var("/ocean/uo").unwrap().shape, [24, 1, 1]);
        assert_eq!(o.read_coord("/ocean/lat").unwrap(), [-5.0]);
        assert_eq!(o.read_coord("/ocean/lon").unwrap(), [5.0]);
        let got = o
            .read_slice(&torcdf_core::SliceRequest {
                var: "/ocean/uo".into(),
                start: vec![0, 0, 0],
                count: vec![24, 1, 1],
                stride: vec![1, 1, 1],
            })
            .unwrap()
            .data;
        let want: Vec<f32> = p.values.iter().map(|v| v.unwrap_or(f32::NAN)).collect();
        assert_eq!(got, want);
        assert_eq!(popup_file_name(Path::new("/d/nc4.nc"), "uo", Some((-12.5, 45.1234))), "nc4_uo_45.1234_m12.5000.nc");
    }

    #[test]
    fn pair_popup_saves_both_components() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir.path()));
        torcdf_core::fixtures::pair_detected(&app.files[0].nc);
        let pair = app.files[0].nc.pairs()[0].path.clone();
        app.apply(Action::Select(Selection::Var(1, pair)));
        app.apply(Action::OpenPopup { iy: 2, ix: 5, lonlat: None });
        let id = app.popups.list[0].id;
        wait_popup(&mut app, id);
        let out = dir.path().join("pair.nc");
        app.start_popup_extract(id, out.clone());
        wait_popup(&mut app, id);
        let o = NcFile::open(&out).unwrap();
        assert_eq!(o.meta.var("/ocean/uo").unwrap().shape, [24, 1, 1]);
        assert_eq!(o.meta.var("/ocean/vo").unwrap().shape, [24, 1, 1]);
    }

    #[test]
    fn calculated_edits_wait_for_a_running_save() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = nc4_extract(dir.path());
        let b = derive(&app, "b", "uo*2");
        app.start_extract(dir.path().join("running.nc"));
        app.apply(Action::OpenCalculate { file: 1, edit: Some(b.clone()) });
        fill(&mut app, "b", "uo*3", "");
        app.apply(Action::CreateDerived);
        let form = app.calc.as_ref().expect("the window stays open");
        assert!(form.check.as_ref().is_err_and(|e| e.contains("save")), "{:?}", form.check);
        assert_eq!(app.files[0].nc.derived(&b).unwrap().formula, "uo*2");
        wait_extract(&mut app);
        app.apply(Action::CreateDerived);
        assert_eq!(app.files[0].nc.derived(&b).unwrap().formula, "uo*3");
    }
}
