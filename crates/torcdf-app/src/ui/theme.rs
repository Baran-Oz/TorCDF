//! The Light (default) and Dark grey looks, the UI font, and the map
//! overlay colours that follow the look.
//!
//! Only the interface is styled: colour-mapped data never passes through
//! here.

use eframe::egui;
use egui::{Color32, CornerRadius, Stroke, Theme, ThemePreference};

use crate::settings::{Look, TextSize};

pub struct Palette {
    pub window: Color32,
    pub panel: Color32,
    pub line: Color32,
    pub text: Color32,
    pub weak: Color32,
    pub accent: Color32,
    pub selection: Color32,
    pub plot_bg: Color32,
    pub warn: Color32,
    pub error: Color32,
    /// Button and other widget background at rest.
    pub widget: Color32,
    /// The same under the pointer.
    pub widget_hover: Color32,
}

pub const LIGHT: Palette = Palette {
    window: Color32::from_rgb(0xf4, 0xf6, 0xf9),
    panel: Color32::from_rgb(0xff, 0xff, 0xff),
    line: Color32::from_rgb(0xd9, 0xde, 0xe6),
    text: Color32::from_rgb(0x1b, 0x25, 0x33),
    weak: Color32::from_rgb(0x6b, 0x77, 0x87),
    accent: Color32::from_rgb(0x1f, 0x6f, 0xd1),
    selection: Color32::from_rgb(0xe3, 0xee, 0xfc),
    plot_bg: Color32::from_rgb(0xee, 0xf1, 0xf5),
    warn: Color32::from_rgb(0x9a, 0x67, 0x00),
    error: Color32::from_rgb(0xc6, 0x28, 0x28),
    widget: Color32::from_rgb(0xee, 0xf1, 0xf5),
    widget_hover: Color32::from_rgb(0xe2, 0xe7, 0xee),
};

pub const DARK: Palette = Palette {
    window: Color32::from_rgb(0x1e, 0x1f, 0x22),
    panel: Color32::from_rgb(0x2a, 0x2c, 0x30),
    line: Color32::from_rgb(0x3a, 0x3d, 0x42),
    text: Color32::from_rgb(0xdc, 0xdf, 0xe4),
    weak: Color32::from_rgb(0x8b, 0x90, 0x99),
    accent: Color32::from_rgb(0x5a, 0xa0, 0xff),
    selection: Color32::from_rgb(0x34, 0x46, 0x5f),
    plot_bg: Color32::from_rgb(0x17, 0x18, 0x1a),
    warn: Color32::from_rgb(0xe0, 0xb0, 0x50),
    error: Color32::from_rgb(0xff, 0x6b, 0x6b),
    widget: Color32::from_rgb(0x34, 0x37, 0x3c),
    widget_hover: Color32::from_rgb(0x40, 0x44, 0x4a),
};

pub fn palette(dark: bool) -> &'static Palette {
    if dark { &DARK } else { &LIGHT }
}

/// egui's own light or dark style, recoloured with `p`.
pub fn style_for(p: &Palette, dark: bool) -> egui::Style {
    let mut s = egui::Style {
        visuals: if dark { egui::Visuals::dark() } else { egui::Visuals::light() },
        ..Default::default()
    };
    let v = &mut s.visuals;
    v.panel_fill = p.panel;
    v.window_fill = p.panel;
    v.window_stroke = Stroke::new(1.0, p.line);
    v.extreme_bg_color = p.plot_bg;
    v.faint_bg_color = p.window;
    v.selection.bg_fill = p.selection;
    v.selection.stroke = Stroke::new(1.0, p.accent);
    v.hyperlink_color = p.accent;
    v.warn_fg_color = p.warn;
    v.error_fg_color = p.error;
    v.override_text_color = None;
    v.weak_text_color = Some(p.weak);
    v.text_cursor.stroke = Stroke::new(2.0, p.accent);
    v.window_corner_radius = CornerRadius::same(6);
    v.menu_corner_radius = CornerRadius::same(6);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = p.panel;
    w.noninteractive.weak_bg_fill = p.window;
    w.noninteractive.bg_stroke = Stroke::new(1.0, p.line);
    w.noninteractive.fg_stroke = Stroke::new(1.0, p.text);
    w.inactive.bg_fill = p.widget;
    w.inactive.weak_bg_fill = p.widget;
    w.inactive.bg_stroke = Stroke::new(1.0, p.line);
    w.inactive.fg_stroke = Stroke::new(1.0, p.text);
    w.hovered.bg_fill = p.widget_hover;
    w.hovered.weak_bg_fill = p.widget_hover;
    w.hovered.bg_stroke = Stroke::new(1.0, p.accent);
    w.hovered.fg_stroke = Stroke::new(1.5, p.text);
    w.active.bg_fill = p.accent;
    w.active.weak_bg_fill = p.accent;
    w.active.bg_stroke = Stroke::new(1.0, p.accent);
    w.active.fg_stroke = Stroke::new(2.0, Color32::WHITE);
    w.open.bg_fill = p.widget_hover;
    w.open.weak_bg_fill = p.widget_hover;
    w.open.bg_stroke = Stroke::new(1.0, p.line);
    w.open.fg_stroke = Stroke::new(1.0, p.text);
    for wv in [&mut w.noninteractive, &mut w.inactive, &mut w.hovered, &mut w.active, &mut w.open] {
        wv.corner_radius = CornerRadius::same(4);
    }
    s.spacing.item_spacing = egui::vec2(8.0, 5.0);
    s.spacing.button_padding = egui::vec2(8.0, 3.0);
    s
}

/// Installs both looks, picks one, and sets the text size.
pub fn apply_look(ctx: &egui::Context, look: Look, size: TextSize) {
    ctx.set_style_of(Theme::Light, style_for(&LIGHT, false));
    ctx.set_style_of(Theme::Dark, style_for(&DARK, true));
    ctx.set_theme(match look {
        Look::Light => ThemePreference::Light,
        Look::Dark => ThemePreference::Dark,
        Look::System => ThemePreference::System,
    });
    ctx.set_zoom_factor(size.zoom());
}

/// Puts Windows' Segoe UI first for ordinary text, keeping egui's fonts
/// behind it for symbols. Does nothing when the font cannot be read.
pub fn install_fonts(ctx: &egui::Context) {
    let windir = std::env::var_os("WINDIR").unwrap_or_else(|| "C:\\Windows".into());
    let path = std::path::Path::new(&windir).join("Fonts").join("segoeui.ttf");
    let Ok(bytes) = std::fs::read(&path) else { return };
    let mut defs = egui::FontDefinitions::default();
    defs.font_data.insert("segoe".into(), std::sync::Arc::new(egui::FontData::from_owned(bytes)));
    defs.families.entry(egui::FontFamily::Proportional).or_default().insert(0, "segoe".into());
    ctx.set_fonts(defs);
}

/// Map overlay line colours for a look.
pub struct OverlayColours {
    pub coast: Color32,
    pub borders: Color32,
    pub grid: Color32,
}

pub fn overlay_colours(dark: bool) -> OverlayColours {
    if dark {
        OverlayColours {
            coast: Color32::from_gray(235),
            borders: Color32::from_gray(160),
            grid: Color32::from_white_alpha(60),
        }
    } else {
        OverlayColours {
            coast: LIGHT.text,
            borders: Color32::from_gray(110),
            grid: Color32::from_rgba_unmultiplied(0x1b, 0x25, 0x33, 70),
        }
    }
}

fn luminance(c: Color32) -> f32 {
    let lin = |u: u8| {
        let s = u as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
}

/// WCAG contrast ratio of two opaque colours (1 to 21).
pub fn contrast_ratio(a: Color32, b: Color32) -> f32 {
    let (la, lb) = (luminance(a), luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{Look, TextSize};

    #[test]
    fn text_contrast_both_looks() {
        for p in [&LIGHT, &DARK] {
            assert!(contrast_ratio(p.text, p.panel) >= 4.5);
            assert!(contrast_ratio(p.weak, p.panel) >= 3.0);
            assert!(contrast_ratio(p.text, p.window) >= 4.5);
        }
    }

    #[test]
    fn coast_visible_on_plot_bg() {
        for (dark, p) in [(false, &LIGHT), (true, &DARK)] {
            assert!(contrast_ratio(overlay_colours(dark).coast, p.plot_bg) >= 3.0, "dark {dark}");
        }
    }

    #[test]
    fn accent_on_panel_visible() {
        for p in [&LIGHT, &DARK] {
            assert!(contrast_ratio(p.accent, p.panel) >= 3.0);
        }
    }

    #[test]
    fn contrast_ratio_extremes() {
        assert!((contrast_ratio(Color32::BLACK, Color32::WHITE) - 21.0).abs() < 0.01);
        assert!((contrast_ratio(Color32::WHITE, Color32::WHITE) - 1.0).abs() < 0.01);
    }

    #[test]
    fn style_uses_palette() {
        let l = style_for(&LIGHT, false);
        assert_eq!(l.visuals.panel_fill, LIGHT.panel);
        assert!(!l.visuals.dark_mode);
        assert_eq!(l.visuals.selection.bg_fill, LIGHT.selection);
        let d = style_for(&DARK, true);
        assert!(d.visuals.dark_mode);
        assert_eq!(d.visuals.panel_fill, DARK.panel);
    }

    #[test]
    fn apply_look_sets_preference_and_zoom() {
        let ctx = egui::Context::default();
        apply_look(&ctx, Look::System, TextSize::Larger);
        ctx.run_ui(Default::default(), |_| {}).textures_delta.clear(); // zoom applies at the next pass
        assert_eq!(ctx.options(|o| o.theme_preference), egui::ThemePreference::System);
        assert_eq!(ctx.zoom_factor(), 1.25);
        apply_look(&ctx, Look::Dark, TextSize::Normal);
        ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
        assert_eq!(ctx.options(|o| o.theme_preference), egui::ThemePreference::Dark);
        assert_eq!(ctx.zoom_factor(), 1.0);
    }

    #[test]
    fn palette_by_mode() {
        assert_eq!(palette(false).panel, LIGHT.panel);
        assert_eq!(palette(true).panel, DARK.panel);
    }
}
