//! The Statistics window: choose data, then counts, extremes, mean,
//! spread, percentiles and a histogram.

use eframe::egui;
use torcdf_core::NcFile;
use torcdf_core::extract::DimKey;

use super::extract::{offered, selection_fields};
use crate::app::{Action, SelWin};
use crate::view::stats::{ROWS, StatsForm};

/// Draws the window. `nc`: the file, while it is open. `sliders`: the
/// dimensions the view has a slider for.
pub fn show(ctx: &egui::Context, form: &mut StatsForm, nc: Option<&NcFile>, sliders: &[DimKey], actions: &mut Vec<Action>) {
    let mut open = true;
    let shown = egui::Window::new("Statistics")
        .id(egui::Id::NULL.with("statistics"))
        .open(&mut open)
        .default_width(680.0)
        .min_width(560.0)
        .resizable(true)
        .collapsible(false)
        // Long results (many variables, the histogram) scroll instead of running off the screen.
        .vscroll(true)
        .show(ctx, |ui| contents(ui, form, nc, sliders, actions));
    if shown.is_some_and(|r| r.response.contains_pointer()) && ctx.input(|i| i.pointer.any_pressed()) {
        actions.push(Action::TouchSelection(SelWin::Stats));
    }
    if !open {
        actions.push(Action::CloseStats);
    }
}

fn contents(ui: &mut egui::Ui, form: &mut StatsForm, nc: Option<&NcFile>, sliders: &[DimKey], actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.label("File:");
        ui.strong(&form.file_name);
    });
    let running = form.job.is_some();
    match nc {
        Some(nc) => {
            ui.add_enabled_ui(!running, |ui| {
                selection_fields(ui, &mut form.sel, offered(nc, true), sliders, SelWin::Stats, "Variables", actions);
            });
        }
        None => {
            ui.weak("The file was closed.");
        }
    }
    if form.region.is_some() {
        let mut w = form.weighted;
        if ui.checkbox(&mut w, "Weight by cell area (cos latitude)").on_hover_text("Applies to the mean and standard deviation").changed() {
            actions.push(Action::SetWeighted(w));
        }
    }
    let request = form.stats_request(form.weighted);
    if let Err(e) = &request {
        ui.colored_label(ui.visuals().error_fg_color, e);
    }
    ui.horizontal(|ui| {
        let can = nc.is_some() && request.is_ok() && !form.busy();
        if ui.add_enabled(can, egui::Button::new("Compute")).clicked() {
            actions.push(Action::ComputeStats);
        }
        if ui.button("Close").clicked() {
            actions.push(Action::CloseStats);
        }
    });
    if let Some(job) = &form.job {
        ui.horizontal(|ui| {
            ui.add(egui::ProgressBar::new(job.progress()).desired_width(320.0).show_percentage());
            if ui.button("Cancel").clicked() {
                actions.push(Action::CancelStats);
            }
        });
    }
    if let Some(s) = &form.status {
        let colour = if s.starts_with("Saved") { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
        ui.add(egui::Label::new(egui::RichText::new(s).color(colour)).wrap());
    }
    results(ui, form, actions);
}

fn results(ui: &mut egui::Ui, form: &mut StatsForm, actions: &mut Vec<Action>) {
    let stale = form.out_of_date();
    let Some((_, table, outcome)) = &form.results else { return };
    ui.separator();
    let heading = if stale { "for the previous selection" } else { "for this selection" };
    ui.weak(format!("{} — {heading}", table.title));
    let weak = ui.visuals().weak_text_color();
    let text = |s: &str| if stale { egui::RichText::new(s).color(weak) } else { egui::RichText::new(s) };
    egui::ScrollArea::horizontal().id_salt("stats table").show(ui, |ui| {
        egui::Grid::new("stats grid").striped(true).spacing([14.0, 4.0]).show(ui, |ui| {
            ui.label("");
            for c in &table.columns {
                ui.strong(text(c));
            }
            ui.end_row();
            for (label, cells) in ROWS.iter().zip(&table.cells) {
                ui.label(text(label));
                for c in cells {
                    ui.label(text(c));
                }
                ui.end_row();
            }
        });
    });

    // Histogram of one variable.
    let ok: Vec<(usize, &torcdf_core::stats::VarStats)> =
        outcome.iter().enumerate().filter_map(|(i, r)| r.as_ref().ok().map(|s| (i, s))).collect();
    if !ok.is_empty() {
        let pick = ok.iter().position(|(i, _)| *i == form.hist_var).unwrap_or(0);
        let (i, s) = ok[pick];
        if ok.len() > 1 {
            egui::ComboBox::from_id_salt("stats hist var").selected_text(&table.columns[i]).show_ui(ui, |ui| {
                for (j, _) in &ok {
                    ui.selectable_value(&mut form.hist_var, *j, &table.columns[*j]);
                }
            });
        }
        let n = s.histogram.len().max(1) as f64;
        let width = (s.max.value - s.min.value) / n;
        let bars: Vec<egui_plot::Bar> = s
            .histogram
            .iter()
            .enumerate()
            .map(|(k, &c)| {
                let centre = s.min.value + (k as f64 + 0.5) * width;
                egui_plot::Bar::new(centre, c as f64).width(width.max(f64::EPSILON))
            })
            .collect();
        egui_plot::Plot::new(("stats histogram", i))
            .height(160.0)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .y_axis_label("count")
            .x_axis_label(table.columns[i].clone())
            .show(ui, |p| p.bar_chart(egui_plot::BarChart::new("count", bars)));
    }
    ui.horizontal(|ui| {
        if ui.button("Copy").on_hover_text("Tab-separated, for Excel or a text file").clicked() {
            actions.push(Action::CopyStats);
        }
        if ui.add_enabled(form.csv_dialog.is_none(), egui::Button::new("Save CSV…")).clicked() {
            actions.push(Action::SaveStatsCsv);
        }
    });
}
