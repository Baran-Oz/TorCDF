//! Map plot area: the data tile, grid lines, borders and coastlines through
//! the map camera, pan/zoom/hover input, and the colour bar.
//!
//! Drag pans, the wheel zooms about the cursor and double-click re-fits: all
//! three change `MapState::camera` only (no read, no upload). Uploads happen
//! when new data or a finished background job arrives.

use eframe::egui::{self, Color32, Rect, Vec2};
use eframe::egui_wgpu::RenderState;
use torcdf_core::SliceRequest;

use super::colorbar;
use super::plot_area::{WHEEL_ZOOM, all_missing, draw_range, plot_rects, reference_arrow};
use crate::map::arrows::{ArrowInput, ArrowSpace, arrow_step};
use crate::map::tiles::{TileSpec, visible_windows};
use crate::render::arrows::{ARROW_MAP_ID, ArrowParams, map_affine, paint_arrows};
use crate::render::gpu_common::camera_uniform;
use crate::render::lines::{LineParams, paint_lines};
use crate::render::map_curv::paint_curvilinear;
use crate::render::map_regular::{MapParams, paint_regular};
use crate::view::arrow_state::{ArrowKey, IndexWindow, map_lonlat};
use crate::view::defaults::AxisChoice;
use crate::ui::popup::draw_markers;
use crate::view::popup::{Popup, PopupSet};
use crate::view::map_state::{BORDERS_ID, COAST_ID, GRID_ID, UploadFailure};
use crate::view::var_view::{Display, VarView};


/// The tile a read covers on the plotted axes.
fn spec_of(req: &SliceRequest, a: AxisChoice) -> TileSpec {
    TileSpec {
        start: [req.start[a.y], req.start[a.x]],
        count: [req.count[a.y], req.count[a.x]],
        stride: [req.stride[a.y], req.stride[a.x]],
    }
}

fn centred_text(ui: &egui::Ui, rect: Rect, text: &str, colour: Color32) {
    let font = egui::FontId::proportional(16.0);
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, font, colour);
}

/// Draws the map of `v`. Returns true when a new data tile was uploaded.
pub fn show(
    ui: &mut egui::Ui,
    v: &mut VarView,
    display: &Display,
    rs: Option<&RenderState>,
    popups: &PopupSet,
) -> bool {
    // A failed background job shows once here; it is not retried.
    if let Some(m) = &v.map {
        if let Some(p) = &m.tile_problem {
            ui.colored_label(ui.visuals().error_fg_color, p);
        }
        if let Some(p) = &m.overlay_problem {
            ui.colored_label(ui.visuals().warn_fg_color, p);
        }
        if let Some(p) = &m.detail_problem {
            ui.colored_label(ui.visuals().warn_fg_color, p);
        }
    }
    let (rect, bar, resp) = plot_rects(ui, "map");
    let Some(rs) = rs else {
        centred_text(ui, rect, "GPU rendering is unavailable, so maps cannot be drawn.", ui.visuals().error_fg_color);
        return false;
    };
    let loading = v.loading();
    let (Some(axes), Some(map)) = (v.axes, v.map.as_mut()) else { return false };
    v.hover = None;
    map.activity = "";
    map.hovering = false;
    map.pointer = None;
    if map.locator.is_none() {
        // Coordinates are being read, or the cell index built.
        ui.put(Rect::from_center_size(rect.center(), Vec2::splat(32.0)), egui::Spinner::new());
        let below = rect.translate(Vec2::new(0.0, 36.0));
        centred_text(ui, below, "Preparing the map…", ui.visuals().weak_text_color());
        return false;
    }
    map.fit_if_needed(rect);

    // Input: camera only.
    if resp.dragged() {
        map.camera.pan(resp.drag_delta());
    }
    if let Some(pos) = resp.hover_pos() {
        let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let factor = (scroll * WHEEL_ZOOM).exp() * zoom;
        if factor != 1.0 {
            map.camera.zoom_at(pos, factor as f64, rect);
        }
    }
    // A click (no drag) opens a pop-up at the cell under it; off the grid it does nothing.
    // It waits out the double-click time first (see `plot_area::show`): a
    // double-click only fits the view.
    if resp.clicked()
        && !resp.double_clicked()
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some((lon, lat)) = map.camera.screen_to_lonlat(pos, rect)
        && let Some(loc) = &map.locator
        && let Some((iy, ix)) = loc.nearest(lon, lat)
    {
        v.pending_click.click(std::time::Instant::now(), (iy, ix, Some((lon, lat))));
    }
    if resp.double_clicked() {
        v.pending_click.cancel();
        map.request_fit();
        map.fit_if_needed(rect);
    }

    // Finished jobs, new jobs, uploads.
    let full = [v.var.shape[axes.y], v.var.shape[axes.x]];
    let spec = v.shown.as_ref().map(|s| spec_of(&s.req, axes));
    map.prepare(ui.ctx(), rs, rect, &v.coords, full, spec);
    let mut uploaded = false;
    let mut failure = None;
    if let (Some(s), Some(spec)) = (&v.shown, spec) {
        match map.upload_overview(rs, &s.slice, spec, axes, &v.coords) {
            Ok(u) => uploaded = u,
            Err(f) => failure = Some(f),
        }
    }

    // Detail tiles: planned once the camera has rested, then read and
    // uploaded as a set. Moving the camera never reads anything. The rest is
    // only consumed once there is an overview to plan against.
    if let Some(spec) = spec
        && map.detail_due(ui.ctx(), rect)
    {
        map.plan_detail_for(full, rect, ui.ctx().pixels_per_point(), &spec);
        // Arrows: the step and visible window at rest (plan P4).
        if let (Some(a), Some(loc)) = (v.arrows.as_mut(), &map.locator)
            && let Some(vis) = visible_windows(loc, full, &map.camera, rect)
        {
            a.rest = Some((arrow_step(vis.cell_pt), IndexWindow::from_visible(&vis)));
        }
    }
    map.prepare_detail(ui.ctx(), rs, axes, &v.coords, full);

    // Arrows are rebuilt in a job when the tiles on show, the step, the
    // projection or the resting window change; panning inside the built
    // window changes only the camera.
    if let Some(a) = v.arrows.as_mut().filter(|a| a.on) {
        a.poll_finished();
        a.upload_ready(rs);
        if let (Some((k, core)), Some(ov)) = (&a.rest, map.drawable_overview()) {
            let mut tiles = map.detail_tiles();
            tiles.push((ov.spec, ov.data.clone()));
            let space = ArrowSpace::Map(map.camera.proj);
            let key = ArrowKey::of(&tiles, *k, space);
            if a.wants(&key, core) {
                let core = core.clone();
                match map_lonlat(map.is_curvilinear(), &v.coords, axes, full) {
                    Some(lonlat) => {
                        let make = move |_: &_| ArrowInput { space, tiles, lonlat, cells: Vec::new() };
                        a.start(ui.ctx(), key, &core, full, make);
                    }
                    None => a.give_up(key),
                }
            }
        }
    }

    // Hover: pointer -> lon/lat -> cell -> value. The index shown is the
    // cell the drawn tile actually sampled, so index and value agree.
    if let Some(pos) = resp.hover_pos() {
        map.hovering = true;
        map.pointer = map.camera.screen_to_lonlat(pos, rect);
        if let (Some((lon, lat)), Some(loc)) = (map.pointer, &map.locator)
            && let Some((iy, ix)) = loc.nearest(lon, lat)
            && let Some(h) = map.hover(iy, ix)
        {
            v.hover = Some(h);
        }
    }

    // Data tile, then grid lines, borders and coastlines on top.
    let camera = camera_uniform(&map.camera, rect, ui.ctx().pixels_per_point());
    let range = v.shown.as_ref().and_then(|s| draw_range(display, &s.stats));
    if let (Some(t), Some(range)) = (map.drawable_overview(), range) {
        let p = MapParams { range, colormap: display.colormap, log: display.log, camera };
        if map.is_curvilinear() {
            paint_curvilinear(ui, rect, &t.gpu, &p);
        } else {
            paint_regular(ui, rect, &t.gpu, &p);
        }
    }
    if let Some(range) = range {
        let p = MapParams { range, colormap: display.colormap, log: display.log, camera };
        for t in map.drawable_details() {
            if map.is_curvilinear() {
                paint_curvilinear(ui, rect, &t.gpu, &p);
            } else {
                paint_regular(ui, rect, &t.gpu, &p);
            }
        }
    }
    let wrap = map.camera.proj.wraps_x();
    let ppp = ui.ctx().pixels_per_point();
    let arrows = v.arrows.as_ref().filter(|a| a.drawable_in(ArrowSpace::Map(map.camera.proj)));
    let px_per_unit = arrows.and_then(|a| a.px_per_unit(ppp));
    if let Some(px_per_unit) = px_per_unit {
        let p = ArrowParams { affine: map_affine(&camera), px_per_unit, wrap, rect_px: camera.rect_px };
        paint_arrows(ui, rect, ARROW_MAP_ID, &p);
    }
    let overlay = crate::ui::theme::overlay_colours(ui.visuals().dark_mode);
    let line = |colour| LineParams { camera, colour, wrap };
    if map.grid && map.grid_ready() {
        paint_lines(ui, rect, GRID_ID, &line(overlay.grid));
    }
    if map.basemap_ready() {
        if map.borders {
            paint_lines(ui, rect, BORDERS_ID, &line(overlay.borders));
        }
        if map.coast {
            paint_lines(ui, rect, COAST_ID, &line(overlay.coast));
        }
    }
    let marker = |p: &Popup| {
        let (lon, lat) = p.lonlat?;
        let w = map.camera.proj.forward(lon, lat)?;
        Some(map.camera.world_to_screen(w, rect)).filter(|s| rect.contains(*s))
    };
    // Field by field: `map` is still borrowed (a layered view marks its own variable).
    draw_markers(ui.painter(), popups, v.file_id, v.base.as_deref().unwrap_or(&v.var.path), &marker);
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    ui.painter().rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Inside);
    if v.shown.is_some() && range.is_none() {
        all_missing(ui, rect);
    }
    if loading && map.overview.is_none() {
        ui.put(Rect::from_center_size(rect.center(), Vec2::splat(32.0)), egui::Spinner::new());
    }
    if let Some(range) = range {
        let arrow = reference_arrow(v, px_per_unit.is_some());
        colorbar::show(ui, bar, display.colormap, range, display.log, &colorbar::colorbar_label(&v.var), arrow);
    }

    match failure {
        Some(UploadFailure::Coords(why)) => v.fall_back_to_image(&why),
        Some(UploadFailure::Other(e)) => v.error = Some(format!("Cannot draw this map: {e}")),
        None => {}
    }
    uploaded
}
