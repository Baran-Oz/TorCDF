//! The Calculate window: a new variable from a formula.

use eframe::egui;
use torcdf_core::FUNCTIONS;

use crate::app::{Action, CalcForm};
use crate::settings::RecentFormula;

/// Draws the window; what the user did is added to `actions`.
/// `vars`: names to insert (backtick paths where a name is not unique).
pub fn show(
    ctx: &egui::Context,
    form: &mut CalcForm,
    file_name: &str,
    vars: &[String],
    recent: &[RecentFormula],
    actions: &mut Vec<Action>,
) {
    let mut open = true;
    let title = if form.replacing.is_some() { "Edit calculated variable" } else { "Calculate new variable" };
    egui::Window::new(title)
        .id(egui::Id::NULL.with("calculate"))
        .open(&mut open)
        .default_width(520.0)
        .collapsible(false)
        .show(ctx, |ui| contents(ui, form, file_name, vars, recent, actions));
    if !open {
        actions.push(Action::CloseCalculate);
    }
}

fn contents(
    ui: &mut egui::Ui,
    form: &mut CalcForm,
    file_name: &str,
    vars: &[String],
    recent: &[RecentFormula],
    actions: &mut Vec<Action>,
) {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("File:");
        ui.strong(file_name);
    });
    egui::Grid::new("calc fields").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
        ui.label("Name");
        let editing = form.replacing.is_some();
        let name = ui.add_enabled(!editing, egui::TextEdit::singleline(&mut form.name).desired_width(200.0));
        let name = if editing { name.on_disabled_hover_text("The name stays while editing") } else { name };
        changed |= name.changed();
        ui.end_row();
        ui.label("Units");
        changed |= ui.add(egui::TextEdit::singleline(&mut form.units).hint_text("optional").desired_width(200.0)).changed();
        ui.end_row();
    });
    ui.label("Formula");
    let formula_id = egui::Id::NULL.with("calc formula");
    let r = ui.add(
        egui::TextEdit::singleline(&mut form.formula)
            .id(formula_id)
            .hint_text("e.g. sqrt(uo^2 + vo^2)")
            .font(egui::TextStyle::Monospace)
            .desired_width(f32::INFINITY),
    );
    changed |= r.changed();
    if !recent.is_empty() {
        egui::ComboBox::from_id_salt("calc recent").selected_text("Recent").show_ui(ui, |ui| {
            for (i, f) in recent.iter().enumerate() {
                if ui.selectable_label(false, format!("{} = {}", f.name, f.formula)).clicked() {
                    actions.push(Action::UseRecentFormula(i));
                }
            }
        });
    }
    ui.add_space(4.0);
    match &form.check {
        Ok(text) => {
            ui.colored_label(ui.visuals().selection.stroke.color, format!("✔ {text}"));
        }
        Err(e) if form.formula.trim().is_empty() => {
            ui.weak(e);
        }
        Err(e) => {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
    }
    ui.add_space(4.0);
    ui.separator();
    let mut insert: Option<String> = None;
    ui.columns(2, |cols| {
        cols[0].label("Variables (click to insert)");
        egui::ScrollArea::vertical().id_salt("calc vars").max_height(140.0).show(&mut cols[0], |ui| {
            ui.horizontal_wrapped(|ui| {
                for v in vars {
                    if ui.small_button(v).clicked() {
                        insert = Some(v.clone());
                    }
                }
            });
        });
        cols[1].label("Functions");
        egui::ScrollArea::vertical().id_salt("calc funcs").max_height(140.0).show(&mut cols[1], |ui| {
            ui.horizontal_wrapped(|ui| {
                for (name, help) in FUNCTIONS {
                    if ui.small_button(*name).on_hover_text(*help).clicked() {
                        insert = Some(format!("{name}("));
                    }
                }
            });
        });
    });
    if let Some(text) = insert {
        insert_at_cursor(ui.ctx(), formula_id, &mut form.formula, &text);
        changed = true;
    }
    if changed {
        actions.push(Action::CalcEdited);
    }
    ui.separator();
    ui.horizontal(|ui| {
        let verb = if form.replacing.is_some() { "Apply" } else { "Create" };
        if ui.add_enabled(form.check.is_ok(), egui::Button::new(verb)).clicked() {
            actions.push(Action::CreateDerived);
        }
        if ui.button("Close").clicked() {
            actions.push(Action::CloseCalculate);
        }
    });
}

/// Inserts `text` at the formula's cursor (at the end if it has none) and
/// puts the cursor after it.
fn insert_at_cursor(ctx: &egui::Context, id: egui::Id, formula: &mut String, text: &str) {
    let mut state = egui::TextEdit::load_state(ctx, id).unwrap_or_default();
    let chars = formula.chars().count();
    let at = state.cursor.char_range().map_or(chars, |r| usize::from(r.primary.index).min(chars));
    let byte = formula.char_indices().nth(at).map_or(formula.len(), |(b, _)| b);
    formula.insert_str(byte, text);
    let after = egui::text::CCursor::new(at + text.chars().count());
    state.cursor.set_char_range(Some(egui::text::CCursorRange::one(after)));
    state.store(ctx, id);
    ctx.memory_mut(|m| m.request_focus(id));
}
