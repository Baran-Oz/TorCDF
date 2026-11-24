//! Centre panel: map, image, line or table view of the selected variable.

use std::borrow::Cow;

use eframe::egui::{self, Pos2, Rect, Vec2};
use eframe::egui_wgpu::RenderState;
use egui_extras::{Column, TableBuilder};

use super::{colorbar, fmt_value, inspector::log_range, map_view, slider_label};
use crate::map::arrows::ArrowSpace;
use crate::map::tiles::TileSpec;
use crate::render::arrows::{ARROW_IMAGE_ID, ArrowParams, image_affine, paint_arrows};
use crate::render::image::{self as gpu, ImageParams};
use crate::view::click::ClickStep;
use crate::view::arrow_state::{ArrowKey, image_input};
use crate::view::defaults::{AxisChoice, ViewKind};
use crate::app::Action;
use crate::ui::popup::draw_markers;
use crate::view::popup::{Popup, PopupSet};
use crate::view::var_view::{Display, Hover, Stats, VarView};

/// The image panel's one GPU layer (layers are never freed, so the id is fixed).
const IMAGE_LAYER_ID: u64 = 1;
/// Wheel zoom speed: view scale factor per scrolled point.
pub const WHEEL_ZOOM: f32 = 0.0025;
/// Width of the colour bar strip right of a map or image (points).
const COLORBAR_WIDTH: f32 = 130.0;

/// Draws the selected variable. Returns true when a new image was uploaded.
pub fn show(
    ui: &mut egui::Ui,
    view: Option<&mut VarView>,
    display: &Display,
    rs: Option<&RenderState>,
    popups: &PopupSet,
    capturing: bool,
    actions: &mut Vec<Action>,
) -> bool {
    let Some(v) = view else {
        ui.centered_and_justified(|ui| {
            ui.weak("Open a file (File > Open, or drop it on the window), then select a variable.");
        });
        return false;
    };
    header(ui, v, capturing, actions);
    if let Some(n) = &v.notice {
        ui.colored_label(ui.visuals().warn_fg_color, n);
    }
    if let Some(left) = v.expire_flash(std::time::Instant::now()) {
        ui.ctx().request_repaint_after(left);
        if let Some((text, _)) = &v.flash {
            ui.colored_label(ui.visuals().warn_fg_color, text);
        }
    }
    if let Some(e) = &v.error {
        ui.colored_label(ui.visuals().error_fg_color, e);
    }
    if let Some(p) = v.arrows.as_ref().filter(|a| a.on).and_then(|a| a.problem.as_ref()) {
        ui.colored_label(ui.visuals().warn_fg_color, p);
    }
    let uploaded = match v.kind {
        ViewKind::Map => map_view::show(ui, v, display, rs, popups),
        ViewKind::Image => image(ui, v, display, rs, popups),
        ViewKind::Line => {
            line(ui, v);
            false
        }
        ViewKind::Table => {
            table(ui, v);
            false
        }
    };
    // A single click opens its pop-up once no second click can follow.
    let double_click_time = ui.ctx().options(|o| o.input_options.max_double_click_delay);
    match v.pending_click.poll(std::time::Instant::now(), double_click_time) {
        Ok((iy, ix, lonlat)) => actions.push(Action::OpenPopup { iy, ix, lonlat }),
        Err(ClickStep::Wait(left)) => ui.ctx().request_repaint_after(left),
        Err(_) => {}
    }
    uploaded
}

fn attr<'a>(v: &'a VarView, name: &str) -> Option<&'a str> {
    v.var.attrs.iter().find(|a| a.name == name).and_then(|a| a.value.as_str())
}

/// Name, long name and units, with a spinner while data is loading, and the
/// PNG button on the right. The button keeps its space but is not drawn while
/// the plot's picture is taken, so it stays out of the image.
fn header(ui: &mut egui::Ui, v: &VarView, capturing: bool, actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.strong(&v.var.name);
        if let Some(long) = attr(v, "long_name").or_else(|| attr(v, "standard_name")) {
            ui.label(long);
        }
        if let Some(u) = attr(v, "units") {
            ui.weak(format!("[{u}]"));
        }
        if v.loading() {
            ui.spinner();
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let png = ui
                .add_visible(!capturing, egui::Button::new("PNG"))
                .on_hover_text("Save the plot as a PNG picture");
            if png.clicked() {
                actions.push(Action::SaveImage);
            }
        });
    });
}

fn image(
    ui: &mut egui::Ui,
    v: &mut VarView,
    display: &Display,
    rs: Option<&RenderState>,
    popups: &PopupSet,
) -> bool {
    let Some(axes) = v.axes else {
        ui.weak("This variable has no two dimensions to draw as an image.");
        return false;
    };
    let mut uploaded = false;
    if let (true, Some(shown)) = (v.image.upload_needed, &v.shown) {
        v.image.upload_needed = false;
        let grid = plot_grid(&shown.slice.shape, axes);
        let data = image_data(&shown.slice.data, &grid);
        let result = match rs {
            Some(rs) => gpu::upload(rs, IMAGE_LAYER_ID, &data, grid.ny, grid.nx).map_err(|e| e.to_string()),
            None => Err("GPU rendering is unavailable".into()),
        };
        match result {
            Ok(layer) => {
                v.image.layer = Some(layer);
                uploaded = true;
            }
            Err(e) => {
                v.image.layer = None;
                v.error = Some(format!("Cannot draw this slice: {e}"));
            }
        }
        if v.image.grid != Some(grid) && !v.image.user_moved {
            v.image.view = None;
        }
        v.image.grid = Some(grid);
    }

    let (rect, bar, resp) = plot_rects(ui, "image");
    v.hover = None;
    let (Some(layer), Some(grid), Some(shown)) = (v.image.layer, v.image.grid, &v.shown) else {
        if v.loading() {
            ui.put(Rect::from_center_size(rect.center(), Vec2::splat(32.0)), egui::Spinner::new());
        }
        return uploaded;
    };
    let slice = shown.slice.clone();
    let stats = shown.stats;
    let (ny, nx) = layer.shape();
    let (y0, sy) = (shown.req.start[axes.y], shown.req.stride[axes.y]);
    let (x0, sx) = (shown.req.start[axes.x], shown.req.stride[axes.x]);

    // Pan and zoom only change the view; they never re-request data.
    let flip_y = v.flip_y();
    if !v.image.user_moved && v.image.fitted_to != rect.size() {
        v.image.view = None;
    }
    let fit = |v: &VarView| fit_view(rect.size(), ny, nx, v.cell_aspect());
    let mut view = match v.image.view {
        Some(view) => view,
        None => {
            v.image.fitted_to = rect.size();
            fit(v)
        }
    };
    if resp.dragged() {
        view = pan_view(view, resp.drag_delta(), rect.size(), flip_y);
        v.image.user_moved = true;
    }
    if let Some(pos) = resp.hover_pos() {
        let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let factor = (-scroll * WHEEL_ZOOM).exp() / zoom;
        let zoomed = zoom_view(view, view_point(pos, rect, view, flip_y), factor);
        let limit = 100.0 * ny.max(nx) as f32;
        if factor != 1.0 && zoomed.width() > 0.5 && zoomed.width() < limit {
            view = zoomed;
            v.image.user_moved = true;
        }
        if let Some((r, c)) = cell_at(pos, rect, view, flip_y, ny, nx) {
            let i = if grid.transposed { c * ny + r } else { r * nx + c };
            if let Some(&value) = slice.data.get(i) {
                // The vector parts are laid out like the values.
                let arrow = slice.vector.as_ref().and_then(|p| Some((*p.east.get(i)?, *p.north.get(i)?)));
                v.hover = Some(Hover { value, iy: y0 + r * sy, ix: x0 + c * sx, arrow });
            }
        }
    }
    // A click (no drag) opens a pop-up at the cell under it.
    // It waits out the double-click time first (see `show`): a double-click
    // only fits the view.
    if resp.clicked()
        && !resp.double_clicked()
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some((r, c)) = cell_at(pos, rect, view, flip_y, ny, nx)
    {
        v.pending_click.click(std::time::Instant::now(), (y0 + r * sy, x0 + c * sx, None));
    }
    if resp.double_clicked() {
        v.pending_click.cancel();
        view = fit(v);
        v.image.fitted_to = rect.size();
        v.image.user_moved = false;
    }
    v.image.view = Some(view);

    let Some(range) = draw_range(display, &stats) else {
        all_missing(ui, rect);
        return uploaded;
    };
    let params = ImageParams { range, colormap: display.colormap, log: display.log, view, flip_y };
    gpu::paint(ui, rect, &layer, &params);
    let arrows_drawn = rs.is_some_and(|rs| image_arrows(ui, v, rs, rect, view, flip_y, grid));
    let marker = |p: &Popup| {
        let r = cell_index(*p.point.get(axes.y)?, y0, sy, ny)?;
        let c = cell_index(*p.point.get(axes.x)?, x0, sx, nx)?;
        Some(cell_centre_screen(r, c, rect, view, flip_y))
    };
    draw_markers(&ui.painter().with_clip_rect(rect), popups, v.file_id, v.plain_path(), &marker);
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    ui.painter().rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Inside);
    let arrow = reference_arrow(v, arrows_drawn);
    colorbar::show(ui, bar, display.colormap, range, display.log, &colorbar::colorbar_label(&v.var), arrow);
    uploaded
}

/// A vector view's arrows over its image, built in a job over the visible
/// cells plus 50% per side (as on the map, plan P4). The step and window
/// are refreshed only once the view has rested; a new slice rebuilds at
/// once over the last window. True when drawn.
fn image_arrows(
    ui: &mut egui::Ui,
    v: &mut VarView,
    rs: &RenderState,
    rect: Rect,
    view: Rect,
    flip_y: bool,
    grid: Grid,
) -> bool {
    let aspect = v.cell_aspect();
    let Some(slice) = v.shown.as_ref().map(|s| s.slice.clone()) else { return false };
    let Some(a) = v.arrows.as_mut().filter(|a| a.on) else { return false };
    a.poll_finished();
    a.upload_ready(rs);
    let full = [grid.ny, grid.nx];
    a.image_rest_due(ui.ctx(), view, rect, full);
    let space = ArrowSpace::Image { flip_y, aspect };
    if let Some((k, core)) = a.image_rest.clone() {
        let tile = TileSpec { start: [0, 0], count: full, stride: [1, 1] };
        let key = ArrowKey::of(&[(tile, slice.clone())], k, space);
        a.update(ui.ctx(), key, &core, full, move |w| image_input(slice, grid, space, w));
    }
    // Never a set built for another flip or aspect under this affine.
    if !a.drawable_in(space) {
        return false;
    }
    let ppp = ui.ctx().pixels_per_point();
    let Some(px_per_unit) = a.px_per_unit(ppp) else { return false };
    let r = rect * ppp;
    let p = ArrowParams {
        affine: image_affine(view, rect, flip_y, ppp),
        px_per_unit,
        wrap: false,
        rect_px: [r.min.x, r.min.y, r.width(), r.height()],
    };
    paint_arrows(ui, rect, ARROW_IMAGE_ID, &p);
    true
}

/// The reference arrow under the colour bar, when arrows are `drawn`.
pub fn reference_arrow(v: &VarView, drawn: bool) -> Option<colorbar::ReferenceArrow<'_>> {
    let (value, len_pt) = v.arrows.as_ref().filter(|_| drawn)?.reference()?;
    Some(colorbar::ReferenceArrow { value, len_pt, units: attr(v, "units").unwrap_or("") })
}

/// Allocates the rest of the panel: the plot rect (click and drag) and the
/// colour bar strip to its right.
pub fn plot_rects(ui: &mut egui::Ui, id: &str) -> (Rect, Rect, egui::Response) {
    let (full, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let w = COLORBAR_WIDTH.min(full.width() * 0.3);
    let plot = Rect::from_min_max(full.min, egui::pos2(full.max.x - w, full.max.y));
    let bar = Rect::from_min_max(egui::pos2(full.max.x - w + 8.0, full.min.y), full.max);
    let resp = ui.interact(plot, ui.make_persistent_id(id), egui::Sense::click_and_drag());
    ui.painter().rect_filled(plot, 0.0, ui.visuals().extreme_bg_color);
    (plot, bar, resp)
}

/// The value range to colour with (log-adjusted in log mode); `None` when
/// every value is missing.
pub fn draw_range(display: &Display, stats: &Stats) -> Option<(f32, f32)> {
    let range = display.effective(Some(stats))?;
    Some(if display.log { log_range(range, stats.min_positive).unwrap_or(range) } else { range })
}

/// "All values missing" across `rect`.
pub fn all_missing(ui: &egui::Ui, rect: Rect) {
    let font = egui::FontId::proportional(18.0);
    let color = ui.visuals().warn_fg_color;
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "All values missing", font, color);
}

/// Label for index `i` of dimension `d`: its coordinate (decoded if time).
fn coord_label(v: &VarView, d: usize, i: usize) -> String {
    let coord = v.coords.dims.get(d).and_then(|c| c.as_ref()?.get(i).copied());
    let time = if v.time_dim == Some(d) { v.time_units.as_ref() } else { None };
    slider_label(i, coord, time)
}

fn line(ui: &mut egui::Ui, v: &VarView) {
    let Some(shown) = &v.shown else { return };
    if shown.line.is_empty() {
        ui.colored_label(ui.visuals().warn_fg_color, "All values missing");
    }
    let d = v.plotted_dims().first().copied();
    let x_label = match d {
        Some(d) if v.coords.dims.get(d).is_some_and(Option::is_some) => v.var.dims[d].clone(),
        Some(d) => format!("{} (index)", v.var.dims[d]),
        None => "index".into(),
    };
    let units = attr(v, "units").map(|u| format!(" [{u}]")).unwrap_or_default();
    let time = d.filter(|d| v.time_dim == Some(*d)).and(v.time_units.clone());
    let name = v.var.name.clone();
    let ticks = time.clone();
    egui_plot::Plot::new(("line", &v.var.path))
        .x_axis_label(x_label)
        .x_axis_formatter(move |m, _| match &ticks {
            Some(t) => super::time_tick(t, m.value, m.step_size),
            None => fmt_value(m.value),
        })
        .y_axis_label(format!("{name}{units}"))
        .label_formatter(move |h| {
            let p = match h {
                egui_plot::HoverPosition::NearDataPoint { position, .. }
                | egui_plot::HoverPosition::Elsewhere { position } => position,
            };
            let x = time.as_ref().map_or_else(|| fmt_value(p.x), |t| t.decode(p.x).to_string());
            Some(format!("x = {x}\ny = {}", fmt_value(p.y)))
        })
        .show(ui, |plot| {
            plot.line(egui_plot::Line::new(name.clone(), egui_plot::PlotPoints::from(&shown.line[..])));
        });
}

fn table(ui: &mut egui::Ui, v: &VarView) {
    if v.is_text() {
        text_table(ui, v);
        return;
    }
    let Some(shown) = &v.shown else { return };
    let (req, data) = (&shown.req, &shown.slice.data);
    let plotted = v.plotted_dims();
    let (rows, cols, ydim, xdim) = match plotted[..] {
        [y, x] => (req.count[y], req.count[x], Some(y), Some(x)),
        [d] => (req.count[d], 1, Some(d), None),
        _ => (1, 1, None, None),
    };
    let transposed = matches!((ydim, xdim), (Some(y), Some(x)) if y > x);
    if plotted.iter().any(|&d| req.count[d] < v.var.shape[d]) {
        let total: usize = plotted.iter().map(|&d| v.var.shape[d]).product();
        ui.weak(format!("Showing the first {rows} × {cols} of {total} values in this slice."));
    }
    let header = |d: Option<usize>| d.map_or_else(String::new, |d| v.var.dims[d].clone());
    egui::ScrollArea::horizontal().show(ui, |ui| {
        TableBuilder::new(ui)
            .id_salt(("table", &v.var.path))
            .striped(true)
            .resizable(true)
            .column(Column::auto().at_least(80.0))
            .columns(Column::auto().at_least(70.0), cols)
            .header(20.0, |mut h| {
                h.col(|ui| {
                    ui.strong(header(ydim));
                });
                for c in 0..cols {
                    h.col(|ui| {
                        let text = match xdim {
                            Some(x) => coord_label(v, x, req.start[x] + c),
                            None => "value".into(),
                        };
                        ui.strong(text);
                    });
                }
            })
            .body(|body| {
                body.rows(18.0, rows, |mut row| {
                    let r = row.index();
                    row.col(|ui| {
                        let text = ydim.map_or_else(String::new, |y| coord_label(v, y, req.start[y] + r));
                        ui.label(text);
                    });
                    for c in 0..cols {
                        let i = if transposed { c * rows + r } else { r * cols + c };
                        row.col(|ui| {
                            ui.label(data.get(i).map_or_else(String::new, |x| fmt_value(*x as f64)));
                        });
                    }
                });
            });
    });
}

fn text_table(ui: &mut egui::Ui, v: &VarView) {
    let Some(t) = &v.text else { return };
    if t.truncated {
        ui.weak(format!("Showing the first {} strings.", t.values.len()));
    }
    TableBuilder::new(ui)
        .id_salt(("text", &v.var.path))
        .striped(true)
        .column(Column::auto().at_least(60.0))
        .column(Column::remainder())
        .header(20.0, |mut h| {
            h.col(|ui| {
                ui.strong("index");
            });
            h.col(|ui| {
                ui.strong("value");
            });
        })
        .body(|body| {
            body.rows(18.0, t.values.len(), |mut row| {
                let i = row.index();
                row.col(|ui| {
                    ui.label(i.to_string());
                });
                row.col(|ui| {
                    ui.label(&t.values[i]);
                });
            });
        });
}

/// Image layout of a slice: `ny` rows by `nx` columns. `transposed` when the
/// y dimension comes after the x dimension in the variable, so the
/// row-major slice must be transposed to get rows along y.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub ny: usize,
    pub nx: usize,
    pub transposed: bool,
}

/// Squeezes a slice `shape` (count 1 on every non-plotted dimension) to the
/// plotted dimensions.
pub fn plot_grid(shape: &[usize], axes: AxisChoice) -> Grid {
    Grid { ny: shape[axes.y], nx: shape[axes.x], transposed: axes.y > axes.x }
}

/// `data` laid out as `g.ny` rows of `g.nx` values.
pub fn image_data<'a>(data: &'a [f32], g: &Grid) -> Cow<'a, [f32]> {
    if !g.transposed {
        return Cow::Borrowed(data);
    }
    // `data` is nx rows (x outer) of ny values.
    let mut out = vec![f32::NAN; g.ny * g.nx];
    for (c, col) in data.chunks_exact(g.ny).enumerate().take(g.nx) {
        for (r, &v) in col.iter().enumerate() {
            out[r * g.nx + c] = v;
        }
    }
    Cow::Owned(out)
}

/// Cell-space point (x = column, y = row, fractional) under screen `pos`
/// when `view` (cell units) is drawn into `rect`. Mirrors `image.wgsl`.
pub fn view_point(pos: Pos2, rect: Rect, view: Rect, flip_y: bool) -> Pos2 {
    let uv = (pos - rect.min) / rect.size();
    let x = view.min.x + uv.x * view.width();
    let y = if flip_y {
        view.max.y - uv.y * view.height()
    } else {
        view.min.y + uv.y * view.height()
    };
    Pos2::new(x, y)
}

/// Row or column of full-grid index `i` in a slice starting at `start` with
/// `stride` and `n` cells, when `i` is one of them.
fn cell_index(i: usize, start: usize, stride: usize, n: usize) -> Option<usize> {
    let stride = stride.max(1);
    let d = i.checked_sub(start)?;
    (d % stride == 0 && d / stride < n).then_some(d / stride)
}

/// Screen position of the centre of cell `(row, col)`; the inverse of `view_point`.
pub fn cell_centre_screen(row: usize, col: usize, rect: Rect, view: Rect, flip_y: bool) -> Pos2 {
    let (x, y) = (col as f32 + 0.5, row as f32 + 0.5);
    let u = (x - view.min.x) / view.width();
    let v = if flip_y { (view.max.y - y) / view.height() } else { (y - view.min.y) / view.height() };
    rect.min + Vec2::new(u, v) * rect.size()
}

/// `(row, col)` of the cell under `pos`, if it lies inside the image.
pub fn cell_at(
    pos: Pos2,
    rect: Rect,
    view: Rect,
    flip_y: bool,
    ny: usize,
    nx: usize,
) -> Option<(usize, usize)> {
    let p = view_point(pos, rect, view, flip_y);
    let inside = p.x >= 0.0 && p.y >= 0.0 && p.x < nx as f32 && p.y < ny as f32;
    inside.then(|| (p.y.floor() as usize, p.x.floor() as usize))
}

/// The view (cell units) showing the whole `ny` x `nx` image centred in a
/// panel of `size`, with cells `cell_aspect` times as wide as tall on screen.
pub fn fit_view(size: Vec2, ny: usize, nx: usize, cell_aspect: f32) -> Rect {
    let (w, h) = (nx as f32 * cell_aspect, ny as f32);
    let whole = Rect::from_min_max(Pos2::ZERO, Pos2::new(nx as f32, ny as f32));
    if !(size.x > 0.0 && size.y > 0.0 && w > 0.0 && h > 0.0 && w.is_finite()) {
        return whole;
    }
    // Screen points per world unit (one cell height).
    let s = (size.x / w).min(size.y / h);
    Rect::from_center_size(whole.center(), Vec2::new(size.x / s / cell_aspect, size.y / s))
}

/// `view` scaled by `factor` about the cell-space point `anchor`, which
/// stays fixed on screen (`factor < 1` zooms in).
pub fn zoom_view(view: Rect, anchor: Pos2, factor: f32) -> Rect {
    Rect::from_min_max(anchor + (view.min - anchor) * factor, anchor + (view.max - anchor) * factor)
}

/// `view` after dragging the image by `delta` screen points in a panel of `size`.
pub fn pan_view(view: Rect, delta: Vec2, size: Vec2, flip_y: bool) -> Rect {
    let dx = -delta.x * view.width() / size.x;
    let dy = delta.y * view.height() / size.y;
    view.translate(Vec2::new(dx, if flip_y { dy } else { -dy }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{pos2, vec2};

    #[test]
    fn marker_position_inverts_cell_at() {
        let rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(400.0, 300.0));
        let view = Rect::from_min_max(pos2(2.0, 1.0), pos2(12.0, 8.0));
        for flip in [false, true] {
            let p = cell_centre_screen(3, 5, rect, view, flip);
            assert_eq!(cell_at(p, rect, view, flip, 8, 12), Some((3, 5)));
        }
        assert_eq!(cell_index(14, 4, 5, 3), Some(2));
        assert_eq!(cell_index(15, 4, 5, 3), None);
        assert_eq!(cell_index(3, 4, 5, 3), None);
    }

    #[test]
    fn grid_squeezes_singleton_dims() {
        // (time, lat, lon) slice at one time step.
        let g = plot_grid(&[1, 4, 8], AxisChoice { y: 1, x: 2 });
        assert_eq!(g, Grid { ny: 4, nx: 8, transposed: false });
        // (lon, lat): y comes after x.
        let g = plot_grid(&[8, 1, 4], AxisChoice { y: 2, x: 0 });
        assert_eq!(g, Grid { ny: 4, nx: 8, transposed: true });
        // A single-row geo grid keeps its plotted length-1 dimension.
        let g = plot_grid(&[1, 1, 5], AxisChoice { y: 1, x: 2 });
        assert_eq!(g, Grid { ny: 1, nx: 5, transposed: false });
    }

    #[test]
    fn image_data_transposes_when_needed() {
        let d = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // [x=2][y=3] row-major
        let plain = Grid { ny: 2, nx: 3, transposed: false };
        assert_eq!(&*image_data(&d, &plain), &d);
        let t = Grid { ny: 3, nx: 2, transposed: true };
        assert_eq!(&*image_data(&d, &t), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    fn panel() -> Rect {
        Rect::from_min_size(pos2(100.0, 50.0), vec2(200.0, 100.0))
    }

    #[test]
    fn hover_maps_screen_to_cell() {
        let view = Rect::from_min_max(pos2(0.0, 0.0), pos2(4.0, 2.0));
        // Top-left pixel is row 0 unflipped, the last row when flipped.
        assert_eq!(cell_at(pos2(100.5, 50.5), panel(), view, false, 2, 4), Some((0, 0)));
        assert_eq!(cell_at(pos2(100.5, 50.5), panel(), view, true, 2, 4), Some((1, 0)));
        assert_eq!(cell_at(pos2(299.0, 149.0), panel(), view, false, 2, 4), Some((1, 3)));
        assert_eq!(cell_at(pos2(299.0, 149.0), panel(), view, true, 2, 4), Some((0, 3)));
        // Zoomed into the cell (row 1, col 2): the whole panel is that cell.
        let zoomed = Rect::from_min_max(pos2(2.0, 1.0), pos2(3.0, 2.0));
        assert_eq!(cell_at(pos2(250.0, 100.0), panel(), zoomed, false, 2, 4), Some((1, 2)));
        // Outside the image.
        let wide = Rect::from_min_max(pos2(-4.0, 0.0), pos2(8.0, 2.0));
        assert_eq!(cell_at(pos2(101.0, 60.0), panel(), wide, false, 2, 4), None);
        assert_eq!(view_point(pos2(200.0, 100.0), panel(), view, false), pos2(2.0, 1.0));
    }

    #[test]
    fn fit_centres_whole_image() {
        let v = fit_view(vec2(200.0, 100.0), 10, 10, 1.0);
        assert_eq!(v, Rect::from_min_max(pos2(-5.0, 0.0), pos2(15.0, 10.0)));
        // Cells twice as wide as tall: 10 columns span 20 units, filling the width.
        let v = fit_view(vec2(200.0, 100.0), 10, 10, 2.0);
        assert_eq!(v, Rect::from_min_max(pos2(0.0, 0.0), pos2(10.0, 10.0)));
        // Tall panel: letterboxed vertically.
        let v = fit_view(vec2(100.0, 200.0), 10, 10, 1.0);
        assert_eq!(v, Rect::from_min_max(pos2(0.0, -5.0), pos2(10.0, 15.0)));
    }

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let view = Rect::from_min_max(pos2(0.0, 0.0), pos2(10.0, 10.0));
        let z = zoom_view(view, pos2(2.0, 8.0), 0.5);
        assert_eq!(z, Rect::from_min_max(pos2(1.0, 4.0), pos2(6.0, 9.0)));
        // The anchor's relative position in the view is unchanged.
        let rel = |r: Rect, p: Pos2| ((p.x - r.min.x) / r.width(), (p.y - r.min.y) / r.height());
        assert_eq!(rel(view, pos2(2.0, 8.0)), rel(z, pos2(2.0, 8.0)));
    }

    #[test]
    fn pan_follows_the_pointer() {
        let view = Rect::from_min_max(pos2(0.0, 0.0), pos2(4.0, 2.0));
        let size = vec2(200.0, 100.0);
        // Dragging right/down by a quarter panel moves the view left/up one cell
        // (unflipped: rows grow downwards, so the view moves to lower rows).
        let p = pan_view(view, vec2(50.0, 50.0), size, false);
        assert_eq!(p, Rect::from_min_max(pos2(-1.0, -1.0), pos2(3.0, 1.0)));
        // Flipped: rows grow upwards, dragging down shows higher rows.
        let p = pan_view(view, vec2(50.0, 50.0), size, true);
        assert_eq!(p, Rect::from_min_max(pos2(-1.0, 1.0), pos2(3.0, 3.0)));
    }
}
