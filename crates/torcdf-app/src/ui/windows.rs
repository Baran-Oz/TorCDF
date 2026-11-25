//! The Settings, About, licences and keyboard-shortcut windows.

use eframe::egui;

use super::shortcuts::SHORTCUTS;
use crate::app::Action;
use crate::map::projection::Projection;
use crate::render::colormap::Colormap;
use crate::settings::{Look, Settings, TextSize};

/// Which window to open.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Win {
    Settings,
    About,
    Licences,
    Shortcuts,
}

/// The windows' open state (not remembered between runs).
#[derive(Default)]
pub struct Windows {
    pub settings: bool,
    pub about: bool,
    pub licences: bool,
    pub shortcuts: bool,
    icon: Option<egui::TextureHandle>,
}

impl Windows {
    pub fn open(&mut self, w: Win) {
        *match w {
            Win::Settings => &mut self.settings,
            Win::About => &mut self.about,
            Win::Licences => &mut self.licences,
            Win::Shortcuts => &mut self.shortcuts,
        } = true;
    }
}

const LICENCES: &str = include_str!("../../assets/THIRD_PARTY_LICENSES.txt");
/// Copyright and licence line shown in the About window.
pub const NOTICE: &str = "Copyright (C) 2026 Baran Oz · Licensed under GPL-3.0-or-later · no warranty";
/// The project page (the workspace's `repository`).
pub const HOMEPAGE: &str = env!("CARGO_PKG_REPOSITORY");
const RECENT_COUNTS: [usize; 4] = [0, 5, 10, 20];
const PROJECTIONS: [Projection; 4] = [
    Projection::Equirectangular,
    Projection::Mercator,
    Projection::NorthPolarStereo,
    Projection::SouthPolarStereo,
];

/// Version, renderer and Windows version, for bug reports.
pub fn version_info(renderer: &str) -> String {
    format!(
        "TorCDF {}\nRenderer: {renderer}\nWindows: {}",
        env!("CARGO_PKG_VERSION"),
        windows_version().unwrap_or_else(|| "unknown".into())
    )
}

#[cfg(windows)]
fn windows_version() -> Option<String> {
    use windows_sys::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW};
    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }
    let key = wide(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion");
    let read = |name: &str| -> Option<String> {
        let name = wide(name);
        let mut buf = [0u16; 128];
        let mut len = (buf.len() * 2) as u32;
        // SAFETY: the buffers outlive the call and `len` is the byte size of `buf`.
        let rc = unsafe {
            RegGetValueW(
                HKEY_LOCAL_MACHINE,
                key.as_ptr(),
                name.as_ptr(),
                RRF_RT_REG_SZ,
                std::ptr::null_mut(),
                buf.as_mut_ptr().cast(),
                &mut len,
            )
        };
        (rc == 0).then(|| {
            let n = (len as usize / 2).saturating_sub(1).min(buf.len());
            String::from_utf16_lossy(&buf[..n])
        })
    };
    let build = read("CurrentBuild")?;
    Some(match read("DisplayVersion") {
        Some(v) => format!("{v} (build {build})"),
        None => format!("build {build}"),
    })
}

#[cfg(not(windows))]
fn windows_version() -> Option<String> {
    None
}

pub fn show_all(
    ctx: &egui::Context,
    w: &mut Windows,
    settings: &Settings,
    renderer: &str,
    actions: &mut Vec<Action>,
) {
    let mut open = w.settings;
    egui::Window::new("Settings")
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| settings_body(ui, settings, actions, &mut w.settings));
    w.settings &= open;

    let mut open = w.about;
    let icon = w.icon.get_or_insert_with(|| {
        let img = crate::window_icon().map_or_else(
            || egui::ColorImage::filled([1, 1], egui::Color32::TRANSPARENT),
            |i| egui::ColorImage::from_rgba_unmultiplied([i.width as usize, i.height as usize], &i.rgba),
        );
        ctx.load_texture("torcdf-icon", img, egui::TextureOptions::LINEAR)
    });
    let mut show_licences = false;
    egui::Window::new("About TorCDF")
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add(egui::Image::new((icon.id(), egui::vec2(72.0, 72.0))));
                ui.heading("TorCDF");
                ui.weak(format!("Version {} · fast viewer for netCDF and HDF5 files", env!("CARGO_PKG_VERSION")));
                ui.add_space(8.0);
                ui.small("Uses netCDF-C (Unidata), HDF5 (The HDF Group), egui and wgpu.");
                ui.small("Coastlines: Natural Earth (public domain).");
                ui.add_space(4.0);
                ui.small(NOTICE);
                ui.hyperlink_to("github.com/Baran-Oz/TorCDF", HOMEPAGE);
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.link("Third-party licences…").clicked() {
                        show_licences = true;
                    }
                    ui.weak("·");
                    if ui.link("Copy version info").clicked() {
                        ui.ctx().copy_text(version_info(renderer));
                    }
                });
            });
        });
    w.about &= open;
    if show_licences {
        w.open(Win::Licences);
    }

    egui::Window::new("Third-party licences")
        .open(&mut w.licences)
        .default_size([560.0, 460.0])
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(LICENCES).monospace().small()).wrap());
            });
        });

    egui::Window::new("Keyboard shortcuts")
        .open(&mut w.shortcuts)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            egui::Grid::new("shortcuts").num_columns(2).striped(true).spacing([24.0, 6.0]).show(ui, |ui| {
                for s in SHORTCUTS {
                    ui.monospace(s.keys);
                    ui.label(s.label);
                    ui.end_row();
                }
            });
            ui.add_space(6.0);
            ui.weak("Ctrl+S stays free for saving edited files later.");
        });
}

/// One row of mutually exclusive choices.
fn segmented<T: Copy + PartialEq>(ui: &mut egui::Ui, current: T, choices: &[(T, &str)]) -> Option<T> {
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for (value, name) in choices {
            if ui.selectable_label(current == *value, *name).clicked() && current != *value {
                picked = Some(*value);
            }
        }
    });
    picked
}

fn settings_body(ui: &mut egui::Ui, s: &Settings, actions: &mut Vec<Action>, open: &mut bool) {
    let heading = |ui: &mut egui::Ui, text: &str| {
        ui.add_space(6.0);
        ui.label(egui::RichText::new(text).small().strong().color(ui.visuals().weak_text_color()));
    };
    egui::Grid::new("settings").num_columns(2).spacing([24.0, 8.0]).show(ui, |ui| {
        heading(ui, "APPEARANCE");
        ui.end_row();
        ui.label("Look");
        let looks = [(Look::Light, "Light"), (Look::Dark, "Dark"), (Look::System, "Match Windows")];
        if let Some(l) = segmented(ui, s.look, &looks) {
            actions.push(Action::SetLook(l));
        }
        ui.end_row();
        ui.label("Text size");
        let sizes = [(TextSize::Smaller, "Smaller"), (TextSize::Normal, "Normal"), (TextSize::Larger, "Larger")];
        if let Some(t) = segmented(ui, s.text_size, &sizes) {
            actions.push(Action::SetTextSize(t));
        }
        ui.end_row();

        heading(ui, "NEW PLOTS START WITH");
        ui.end_row();
        let mut d = s.plot.clone();
        ui.label("Colour map");
        egui::ComboBox::from_id_salt("default colormap").selected_text(d.colormap.name()).show_ui(ui, |ui| {
            for cm in Colormap::all() {
                ui.selectable_value(&mut d.colormap, *cm, cm.name());
            }
        });
        ui.end_row();
        ui.label("Map projection");
        egui::ComboBox::from_id_salt("default projection").selected_text(d.projection.name()).show_ui(ui, |ui| {
            for p in PROJECTIONS {
                ui.selectable_value(&mut d.projection, p, p.name());
            }
        });
        ui.end_row();
        ui.label("Map overlays");
        ui.horizontal(|ui| {
            ui.checkbox(&mut d.coast, "Coastlines");
            ui.checkbox(&mut d.borders, "Borders");
            ui.checkbox(&mut d.grid, "Grid lines");
        });
        ui.end_row();
        if d != s.plot {
            actions.push(Action::SetPlotDefaults(d));
        }

        heading(ui, "FILES");
        ui.end_row();
        ui.label("Recent files to remember");
        let mut n = s.recent_max;
        egui::ComboBox::from_id_salt("recent count").selected_text(n.to_string()).show_ui(ui, |ui| {
            for c in RECENT_COUNTS {
                ui.selectable_value(&mut n, c, c.to_string());
            }
        });
        if n != s.recent_max {
            actions.push(Action::SetRecentMax(n));
        }
        ui.end_row();
        ui.label("At start-up");
        let mut reopen = s.reopen_last;
        if ui.checkbox(&mut reopen, "Reopen the files open last time").changed() {
            actions.push(Action::SetReopenLast(reopen));
        }
        ui.end_row();
    });
    ui.add_space(10.0);
    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Reset to defaults").clicked() {
            actions.push(Action::ResetSettings);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Close").clicked() {
                *open = false;
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn about_names_licence_and_homepage() {
        assert!(NOTICE.contains("Copyright (C) 2026 Baran Oz"));
        assert!(NOTICE.contains("GPL-3.0-or-later"));
        assert!(NOTICE.contains("no warranty"));
        assert_eq!(HOMEPAGE, "https://github.com/Baran-Oz/TorCDF");
    }

    #[test]
    fn licences_carry_full_crate_texts() {
        assert!(LICENCES.contains("netCDF-C"));
        assert!(LICENCES.contains("Natural Earth"));
        assert!(LICENCES.contains("Permission is hereby granted, free of charge")); // MIT text
        assert!(LICENCES.contains("Apache License"));
        assert!(LICENCES.contains("eframe"));
    }
}
