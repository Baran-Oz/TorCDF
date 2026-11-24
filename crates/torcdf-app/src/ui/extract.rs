//! The Extract window: choose variables and ranges, then save them as a
//! smaller netCDF file.

use eframe::egui;
use torcdf_core::NcFile;
use torcdf_core::extract::DimKey;

use crate::app::Action;
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
    egui::Window::new("Extract")
        .id(egui::Id::NULL.with("extract"))
        .open(&mut open)
        .default_width(560.0)
        .collapsible(false)
        .show(ctx, |ui| contents(ui, form, nc, sliders, actions));
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
                    actions.push(Action::ExtractEdited);
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
                        actions.push(Action::ExtractFromSlider { row: i, end });
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
                actions.push(Action::ExtractUseView);
            }
            if ui.button("Whole grid").clicked() {
                actions.push(Action::ExtractWholeGrid);
            }
        });
    }
    if edited {
        actions.push(Action::ExtractEdited);
    }
    ui.weak("Tip: Shift+drag on the map to draw the region; Shift+drag on a time series to set the time range.");
}
