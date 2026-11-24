// Release builds are a GUI program: no console window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod map;
mod render;
mod settings;
mod ui;
mod view;

use eframe::egui;
use std::sync::Arc;

/// True when `TORCDF_SOFTWARE_RENDERER` is exactly "1".
fn software_renderer_requested(var: Option<&std::ffi::OsStr>) -> bool {
    var == Some(std::ffi::OsStr::new("1"))
}

/// Selects the CPU (WARP) adapter that can present to `surface`, falling back
/// to the first adapter that can when there is none.
fn pick_cpu_adapter(
    adapters: &[eframe::wgpu::Adapter],
    surface: Option<&eframe::wgpu::Surface<'_>>,
) -> Result<eframe::wgpu::Adapter, String> {
    let usable = |a: &&eframe::wgpu::Adapter| surface.is_none_or(|s| a.is_surface_supported(s));
    adapters
        .iter()
        .filter(usable)
        .find(|a| a.get_info().device_type == eframe::wgpu::DeviceType::Cpu)
        .or_else(|| adapters.iter().find(usable))
        .cloned()
        .ok_or_else(|| "no suitable graphics adapter".to_string())
}

/// The 256 px app icon for the title bar and taskbar.
fn window_icon() -> Option<egui::IconData> {
    let bytes: &[u8] = include_bytes!("../assets/torcdf-256.png");
    let mut r = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    (info.color_type == png::ColorType::Rgba && info.bit_depth == png::BitDepth::Eight).then(|| {
        buf.truncate(info.width as usize * info.height as usize * 4);
        egui::IconData { rgba: buf, width: info.width, height: info.height }
    })
}

/// `pos` when at least a 100 × 40 strip of the window's top edge (its
/// title bar) at `pos` lies inside `screen`; `None` otherwise.
fn visible_position(pos: [f32; 2], size: [f32; 2], screen: egui::Rect) -> Option<[f32; 2]> {
    let bar = egui::Rect::from_min_size(egui::pos2(pos[0], pos[1]), egui::vec2(size[0], 40.0));
    let seen = bar.intersect(screen);
    (seen.is_positive() && seen.width() >= 100.0 && seen.height() >= 40.0).then_some(pos)
}

/// The desktop spanning all monitors, in points at the system scale.
#[cfg(windows)]
fn virtual_screen() -> egui::Rect {
    use windows_sys::Win32::UI::HiDpi::GetDpiForSystem;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
    };
    // SAFETY: plain queries without arguments to check.
    let (x, y, w, h, dpi) = unsafe {
        (
            GetSystemMetrics(SM_XVIRTUALSCREEN),
            GetSystemMetrics(SM_YVIRTUALSCREEN),
            GetSystemMetrics(SM_CXVIRTUALSCREEN),
            GetSystemMetrics(SM_CYVIRTUALSCREEN),
            GetDpiForSystem(),
        )
    };
    let scale = if dpi == 0 { 1.0 } else { dpi as f32 / 96.0 };
    egui::Rect::from_min_size(
        egui::pos2(x as f32 / scale, y as f32 / scale),
        egui::vec2(w as f32 / scale, h as f32 / scale),
    )
}

#[cfg(not(windows))]
fn virtual_screen() -> egui::Rect {
    egui::Rect::EVERYTHING
}

fn main() -> eframe::Result {
    // `args_os`: a path that is not valid Unicode must not panic.
    let path = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    let software = software_renderer_requested(std::env::var_os("TORCDF_SOFTWARE_RENDERER").as_deref());
    let title = if software { "TorCDF (software rendering)" } else { "TorCDF" };
    let mut options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size([1400.0, 850.0])
            .with_min_inner_size([640.0, 400.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    if let Some(icon) = window_icon() {
        options.viewport = options.viewport.with_icon(Arc::new(icon));
    }
    let (settings, loaded) = match settings::settings_path() {
        Some(p) => settings::Settings::load_from(&p),
        None => (settings::Settings::default(), settings::Loaded::Fresh),
    };
    if let Some(g) = settings.window {
        let size = [g.size[0].max(640.0), g.size[1].max(400.0)];
        options.viewport = options.viewport.with_inner_size(size).with_maximized(g.maximized);
        if let Some(pos) = g.pos.and_then(|p| visible_position(p, size, virtual_screen())) {
            options.viewport = options.viewport.with_position(pos);
        }
    }
    if software
        && let eframe::egui_wgpu::WgpuSetup::CreateNew(c) = &mut options.wgpu_options.wgpu_setup
    {
        c.native_adapter_selector = Some(Arc::new(pick_cpu_adapter));
    }
    eframe::run_native(
        "TorCDF",
        options,
        Box::new(move |cc| Ok(Box::new(app::TorApp::new(cc, path, settings, loaded)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    fn screen(x0: f32, y0: f32, x1: f32, y1: f32) -> egui::Rect {
        egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1))
    }

    #[test]
    fn visible_position_on_screen() {
        let s = screen(0.0, 0.0, 1920.0, 1080.0);
        assert_eq!(visible_position([100.0, 100.0], [800.0, 600.0], s), Some([100.0, 100.0]));
    }

    #[test]
    fn visible_position_off_screen() {
        let s = screen(0.0, 0.0, 1920.0, 1080.0);
        assert_eq!(visible_position([3000.0, 100.0], [800.0, 600.0], s), None);
        // Only a 10 px strip of the title bar would show.
        assert_eq!(visible_position([-790.0, 100.0], [800.0, 600.0], s), None);
        // Title bar above the top of the screen.
        assert_eq!(visible_position([100.0, -300.0], [800.0, 600.0], s), None);
    }

    #[test]
    fn visible_position_second_monitor() {
        let s = screen(-1920.0, 0.0, 1920.0, 1080.0);
        assert_eq!(visible_position([-1500.0, 50.0], [800.0, 600.0], s), Some([-1500.0, 50.0]));
    }

    #[test]
    fn window_icon_decodes() {
        let icon = window_icon().expect("icon");
        assert_eq!((icon.width, icon.height), (256, 256));
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
        assert_eq!(icon.rgba[3], 0, "corner transparent");
    }

    #[test]
    fn software_flag_parsing() {
        assert!(software_renderer_requested(Some(OsStr::new("1"))));
        assert!(!software_renderer_requested(None));
        assert!(!software_renderer_requested(Some(OsStr::new("0"))));
        assert!(!software_renderer_requested(Some(OsStr::new(""))));
    }
}
