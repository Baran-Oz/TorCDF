//! The Extract window: choose variables and ranges, then save them as a
//! smaller netCDF file.

use eframe::egui;
use torcdf_core::NcFile;
use torcdf_core::extract::DimKey;

use crate::app::{Action, SelWin};
use crate::view::extract::ExtractForm;

/// Variables offered: the file's (coordinate variables are kept
/// automatically, so not listed) and its calculated ones.
fn offered(nc: &NcFile) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = nc
        .meta
        .all_vars()
        .into_iter()
        .filter(|v| !(v.dims.len() == 1 && v.dims[0] == v.name))
        .map(|v| (v.path.clone(), v.path.trim_start_matches('/').to_string()))
        .collect();
    out.extend(nc.derived_vars().iter().map(|d| (d.path.clone(), format!("ƒ {}", d.path.trim_start_matches('/')))));
    out
}

/// Draws the window. `nc`: the file, while it is open. `sliders`: the
/// dimensions the view has a slider for (rows of those get "◀ slider").
pub fn show(
    ctx: &egui::Context,
    form: &mut ExtractForm,
    nc: Option<&NcFile>,
    sliders: &[DimKey],
    actions: &mut Vec<Action>,
) {
    let mut open = true;
    let shown = egui::Window::new("Extract")
        .id(egui::Id::NULL.with("extract"))
        .open(&mut open)
        .default_width(560.0)
        .collapsible(false)
        .show(ctx, |ui| contents(ui, form, nc, sliders, actions));
    if shown.is_some_and(|r| r.response.contains_pointer()) && ctx.input(|i| i.pointer.any_pressed()) {
        actions.push(Action::TouchSelection(SelWin::Extract));
    }
    if !open {
        actions.push(Action::CloseExtract);
    }
}

fn contents(ui: &mut egui::Ui, form: &mut ExtractForm, nc: Option<&NcFile>, sliders: &[DimKey], actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.label("File:");
        ui.strong(&form.file_name);
    });
    let running = form.job.is_some();
    if let Some(nc) = nc {
        ui.add_enabled_ui(!running, |ui| fields(ui, form, nc, sliders, actions));
    } else {
        ui.weak("The file was closed.");
    }

    ui.add_space(6.0);
    match &form.check {
        Ok(plan) => {
            let mb = plan.bytes as f64 / 1e6;
            ui.label(format!("About {} before compression", if mb < 1.0 { format!("{:.2} MB", mb) } else { format!("{mb:.0} MB") }));
            if !plan.skipped.is_empty() {
                ui.colored_label(ui.visuals().warn_fg_color, format!("Will not be saved: {}", plan.skipped.join(", ")));
            }
        }
        Err(e) => {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
    }
    ui.horizontal(|ui| {
        let can_save = nc.is_some() && form.check.is_ok() && !form.busy();
        if ui.add_enabled(can_save, egui::Button::new("Save netCDF…")).clicked() {
            actions.push(Action::SaveExtract);
        }
        if ui.button("Close").clicked() {
            actions.push(Action::CloseExtract);
        }
    });
    if let Some(job) = &form.job {
        ui.horizontal(|ui| {
            let p = job.progress();
            ui.add(egui::ProgressBar::new(p).desired_width(320.0).show_percentage());
            if ui.button("Cancel").clicked() {
                actions.push(Action::CancelExtract);
            }
        });
    }
    if let Some(s) = &form.status {
        let colour = if s.starts_with("Saved") { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
        ui.add(egui::Label::new(egui::RichText::new(s).color(colour)).wrap());
    }
}

fn fields(ui: &mut egui::Ui, form: &mut ExtractForm, nc: &NcFile, sliders: &[DimKey], actions: &mut Vec<Action>) {
    ui.label("Variables (their coordinates are always kept)");
    egui::ScrollArea::vertical().id_salt("extract vars").max_height(140.0).show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            for (path, label) in offered(nc) {
                let mut on = form.ticked.contains(&path);
                if ui.checkbox(&mut on, label).changed() {
                    if on {
                        form.ticked.insert(path);
                    } else {
                        form.ticked.remove(&path);
                    }
                    actions.push(Action::SelectionEdited(SelWin::Extract));
                }
            }
        });
    });
    ui.add_space(4.0);
    // Edits are checked (and snapped) when a box loses focus or Enter is pressed.
    let mut edited = false;
    let kept: Vec<Option<usize>> = form.dims.iter().map(|r| form.kept(&r.key)).collect();
    egui::Grid::new("extract rows").num_columns(5).spacing([6.0, 6.0]).show(ui, |ui| {
        for (i, row) in form.dims.iter_mut().enumerate() {
            let slider = sliders.contains(&row.key);
            ui.label(&row.label);
            for (end, text) in [(false, &mut row.from), (true, &mut row.to)] {
                ui.horizontal(|ui| {
                    ui.label(if end { "To" } else { "From" });
                    edited |= ui.add(egui::TextEdit::singleline(text).desired_width(130.0)).lost_focus();
                    if slider && ui.small_button("◀ slider").on_hover_text("Use the slider's step").clicked() {
                        actions.push(Action::FromSlider { win: SelWin::Extract, row: i, end });
                    }
                });
            }
            match kept.get(i).copied().flatten() {
                Some(n) => ui.weak(n.to_string()),
                None => ui.weak(""),
            };
            ui.end_row();
        }
        if let Some(r) = &mut form.region {
            ui.label("Latitude");
            for (label, text) in [("From", &mut r.south), ("To", &mut r.north)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    edited |= ui.add(egui::TextEdit::singleline(text).desired_width(80.0)).lost_focus();
                });
            }
            ui.end_row();
            ui.label("Longitude");
            for (label, text) in [("West", &mut r.west), ("East", &mut r.east)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    edited |= ui.add(egui::TextEdit::singleline(text).desired_width(80.0)).lost_focus();
                });
            }
            ui.end_row();
        }
    });
    if form.region.is_some() {
        ui.horizontal(|ui| {
            if ui.button("Use current map view").clicked() {
                actions.push(Action::UseView(SelWin::Extract));
            }
            if ui.button("Whole grid").clicked() {
                actions.push(Action::WholeGrid(SelWin::Extract));
            }
        });
    }
    if edited {
        actions.push(Action::SelectionEdited(SelWin::Extract));
    }
    ui.weak("Tip: Shift+drag on the map to draw the region; Shift+drag on a time series to set the time range.");
}

// ---- On-screen selection: Shift+drag and overlays --------------------------

/// The accent colour of Extract boxes and spans.
fn accent(ui: &egui::Ui) -> egui::Color32 {
    ui.visuals().selection.stroke.color
}

/// Shift+drag rubber band on a map or image. `start` holds the press
/// point while a band is drawn. Returns the band while dragging, and the
/// finished band (at least 3 points each way) on release.
pub fn band(
    ui: &egui::Ui,
    resp: &egui::Response,
    start: &mut Option<egui::Pos2>,
    rect: egui::Rect,
) -> (Option<egui::Rect>, Option<egui::Rect>) {
    if resp.drag_started() && ui.input(|i| i.modifiers.shift) {
        *start = ui.input(|i| i.pointer.press_origin());
    }
    let Some(a) = *start else { return (None, None) };
    let b = ui.input(|i| i.pointer.latest_pos()).unwrap_or(a);
    let r = egui::Rect::from_two_pos(a, b).intersect(rect);
    if resp.drag_stopped() || !ui.input(|i| i.pointer.primary_down()) {
        *start = None;
        return (None, (r.width() >= 3.0 && r.height() >= 3.0).then_some(r));
    }
    (Some(r), None)
}

/// Draws a band being dragged.
pub fn paint_band(ui: &egui::Ui, r: egui::Rect) {
    let c = accent(ui);
    let p = ui.painter();
    p.rect_filled(r, 0.0, c.gamma_multiply(0.15));
    p.rect_stroke(r, 0.0, egui::Stroke::new(1.5, c), egui::StrokeKind::Inside);
}

/// Draws the outline of a polygon given in screen points; `None` points
/// (off the projection) break it. With `wrap`, a jump wider than half the
/// panel (round the globe) breaks it too.
pub fn paint_outline(ui: &egui::Ui, clip: egui::Rect, pts: &[Option<egui::Pos2>], wrap: bool) {
    let painter = ui.painter().with_clip_rect(clip);
    let stroke = egui::Stroke::new(2.0, accent(ui));
    // A jump wider than half the panel is a wrap round the globe, not an edge.
    let far = clip.width().max(clip.height()) / 2.0;
    for w in pts.windows(2) {
        if let (Some(a), Some(b)) = (w[0], w[1])
            && (!wrap || (a - b).length() < far)
        {
            painter.line_segment([a, b], stroke);
        }
    }
}

/// The region box's outline in lon/lat: each edge sampled at 16 points,
/// east of west (a box across the seam runs past 180).
pub fn region_outline(s: f64, n: f64, w: f64, e: f64) -> Vec<(f64, f64)> {
    let e = if e < w { e + 360.0 } else { e };
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let mut out = Vec::new();
    for (a, b) in [((w, s), (e, s)), ((e, s), (e, n)), ((e, n), (w, n)), ((w, n), (w, s))] {
        for k in 0..16 {
            let t = k as f64 / 16.0;
            out.push((lerp(a.0, b.0, t), lerp(a.1, b.1, t)));
        }
    }
    out.push((w, s));
    out
}

/// Whether a Shift+drag is (or may start) on the plot with `id`: panning
/// is switched off then.
pub fn span_drag_active(ui: &egui::Ui, id: egui::Id) -> bool {
    ui.input(|i| i.modifiers.shift) || ui.data(|d| d.get_temp::<f64>(id).is_some())
}

/// Shift+drag across an egui_plot along x. Shades `shown` (the window's
/// time range) and the span being dragged; returns the finished (from, to).
pub fn plot_span(
    ui: &egui::Ui,
    id: egui::Id,
    out: &egui_plot::PlotResponse<()>,
    shown: Option<(f64, f64)>,
) -> Option<(f64, f64)> {
    let x_at = |p: egui::Pos2| out.transform.value_from_position(p).x;
    if out.response.drag_started()
        && ui.input(|i| i.modifiers.shift)
        && let Some(p) = ui.input(|i| i.pointer.press_origin())
    {
        ui.data_mut(|d| d.insert_temp(id, x_at(p)));
    }
    let frame = *out.transform.frame();
    let shade = |a: f64, b: f64, alpha: f32| {
        let (xa, xb) = (out.transform.position_from_point_x(a), out.transform.position_from_point_x(b));
        let r = egui::Rect::from_x_y_ranges(xa.min(xb)..=xa.max(xb).max(xa.min(xb) + 1.0), frame.y_range());
        ui.painter().with_clip_rect(frame).rect_filled(r, 0.0, accent(ui).gamma_multiply(alpha));
    };
    if let Some((a, b)) = shown {
        shade(a, b, 0.12);
    }
    let start = ui.data(|d| d.get_temp::<f64>(id))?;
    let now = ui.input(|i| i.pointer.latest_pos()).map(x_at).unwrap_or(start);
    if out.response.drag_stopped() || !ui.input(|i| i.pointer.primary_down()) {
        ui.data_mut(|d| d.remove::<f64>(id));
        return Some((start, now));
    }
    shade(start, now, 0.25);
    ui.ctx().request_repaint();
    None
}
