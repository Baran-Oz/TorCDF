// Release builds are a GUI program: no console window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod render;
mod ui;
mod view;

use eframe::egui;

fn main() -> eframe::Result {
    // `args_os`: a path that is not valid Unicode must not panic.
    let path = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("TorCDF")
            .with_inner_size([1400.0, 850.0])
            .with_min_inner_size([640.0, 400.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "TorCDF",
        options,
        Box::new(move |cc| Ok(Box::new(app::TorApp::new(cc, path)))),
    )
}
