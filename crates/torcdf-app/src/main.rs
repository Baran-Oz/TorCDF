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
    if software
        && let eframe::egui_wgpu::WgpuSetup::CreateNew(c) = &mut options.wgpu_options.wgpu_setup
    {
        c.native_adapter_selector = Some(Arc::new(pick_cpu_adapter));
    }
    eframe::run_native(
        "TorCDF",
        options,
        Box::new(move |cc| Ok(Box::new(app::TorApp::new(cc, path)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

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
