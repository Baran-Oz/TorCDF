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
    Attr, CoreError, DEFAULT_WORKERS, NcFile, SliceService, default_cache_budget_bytes,
};

use crate::render::colormap::Colormap;
use crate::ui::{self, inspector, plot_area, tree};
use crate::view::defaults::ViewKind;
use crate::view::var_view::{Display, ImageState, VarView};

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
