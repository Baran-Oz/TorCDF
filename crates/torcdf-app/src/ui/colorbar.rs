//! Colour bar: tick computation, label text and an egui-drawn gradient.

use std::sync::Arc;

use eframe::egui;
use torcdf_core::VarInfo;

use crate::render::colormap::Colormap;
use crate::ui::fmt_value;

const BAR_WIDTH: f32 = 16.0;

fn pow10(n: i32) -> f64 {
    if n >= 0 { 10f64.powi(n) } else { 1.0 / 10f64.powi(-n) }
}

/// "Nice" tick values inside [lo, hi]: steps of 1, 2 or 5 x 10^n (linear), or
/// the decades (log, lo > 0, at least two decades). At most `max_ticks` ticks.
pub fn nice_ticks(lo: f64, hi: f64, max_ticks: usize, log: bool) -> Vec<f64> {
    if !lo.is_finite() || !hi.is_finite() || max_ticks == 0 {
        return Vec::new();
    }
    let (lo, hi) = if lo > hi { (hi, lo) } else { (lo, hi) };
    if lo == hi {
        return vec![lo];
    }
    if log && lo > 0.0 {
        let eps = 1e-9;
        let first = (lo.log10() - eps).ceil() as i32;
        let last = (hi.log10() + eps).floor() as i32;
        if last > first {
            let n = (last - first + 1) as usize;
            let stride = n.div_ceil(max_ticks).max(1);
            return (first..=last).step_by(stride).map(pow10).collect();
        }
    }
    linear_ticks(lo, hi, max_ticks)
}

fn linear_ticks(lo: f64, hi: f64, max_ticks: usize) -> Vec<f64> {
    let span = hi - lo;
    let eps = span * 1e-9;
    let start = (span / max_ticks as f64).log10().floor() as i32 - 1;
    for n in start..start + 40 {
        for m in [1i64, 2, 5] {
            let step = m as f64 * pow10(n);
            let k0 = ((lo - eps) / step).ceil() as i64;
            let k1 = ((hi + eps) / step).floor() as i64;
            if k1 < k0 || (k1 - k0 + 1) as usize > max_ticks {
                continue;
            }
            // Integer multiple of m, scaled by an exact power of ten: no float noise.
            return (k0..=k1)
                .map(|k| {
                    let q = (k * m) as f64;
                    if n >= 0 { q * pow10(n) } else { q / pow10(-n) }
                })
                .collect();
        }
    }
    Vec::new()
}

/// "long_name [units]", falling back to standard_name, then the variable name.
pub fn colorbar_label(v: &VarInfo) -> String {
    let attr = |name: &str| {
        v.attrs.iter().find(|a| a.name == name).and_then(|a| a.value.as_str()).filter(|s| !s.is_empty())
    };
    let base = attr("long_name").or_else(|| attr("standard_name")).unwrap_or(&v.name);
    match attr("units") {
        Some(u) => format!("{base} [{u}]"),
        None => base.to_string(),
    }
}

/// Space between the label and the bar top, and below the bar: over half a
/// 12 pt text row, so tick labels centred on the bar ends stay clear of the
/// label and inside the strip.
const END_GAP: f32 = 8.0;

/// The label laid out for `rect`, wrapped to its width so the units are
/// never clipped, and the bar below it.
fn layout(
    painter: &egui::Painter,
    rect: egui::Rect,
    label: &str,
    font: egui::FontId,
    colour: egui::Color32,
) -> (Arc<egui::Galley>, egui::Rect) {
    let galley = painter.layout(label.to_owned(), font, colour, rect.width());
    let bar = egui::Rect::from_min_max(
        egui::pos2(rect.left(), rect.top() + galley.size().y + END_GAP),
        egui::pos2(rect.left() + BAR_WIDTH, rect.bottom() - END_GAP),
    );
    (galley, bar)
}

/// Height of the reference arrow row under the bar (arrow and its text).
const ARROW_ROW: f32 = 34.0;

/// A vector view's reference arrow: its value, length in points and units.
pub struct ReferenceArrow<'a> {
    pub value: f32,
    pub len_pt: f32,
    pub units: &'a str,
}

/// `(value, len)` stepped down the 1-2-5 sequence until `len <= max`, the
/// length shrinking with the value.
pub fn fit_reference(value: f64, len: f32, max: f32) -> (f64, f32) {
    let (mut value, mut len) = (value, len);
    for _ in 0..40 {
        if len <= max || !(len > 0.0 && value > 0.0 && value.is_finite()) {
            break;
        }
        let e = value.log10().floor() as i32;
        let m = (value / pow10(e)).round();
        let next = if m >= 5.0 { 2.0 * pow10(e) } else if m >= 2.0 { pow10(e) } else { 5.0 * pow10(e - 1) };
        len *= (next / value) as f32;
        value = next;
    }
    (value, len)
}

/// A horizontal arrow from `x0` of `len` points at height `y`, with
/// "value units" below it.
fn reference_arrow(painter: &egui::Painter, rect: egui::Rect, r: &ReferenceArrow, colour: egui::Color32) {
    let (value, len) = fit_reference(r.value as f64, r.len_pt, rect.width() - 4.0);
    if !(len.is_finite() && len >= 1.0) {
        return;
    }
    let (x0, y) = (rect.left() + 1.0, rect.top() + 8.0);
    let head = (0.3 * len).min(8.0);
    let tip = egui::pos2(x0 + len, y);
    painter.line_segment([egui::pos2(x0, y), egui::pos2(tip.x - head, y)], egui::Stroke::new(1.5, colour));
    let tri = vec![egui::pos2(tip.x - head, y - 0.5 * head), tip, egui::pos2(tip.x - head, y + 0.5 * head)];
    painter.add(egui::Shape::convex_polygon(tri, colour, egui::Stroke::NONE));
    let text = format!("{} {}", fmt_value(value), r.units);
    let font = egui::FontId::proportional(12.0);
    painter.text(egui::pos2(x0, y + 8.0), egui::Align2::LEFT_TOP, text.trim_end(), font, colour);
}

/// Draw the bar (low at the bottom) with tick labels and the label above,
/// inside `rect`; with `arrow`, the reference arrow under the bar.
pub fn show(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    colormap: Colormap,
    range: (f32, f32),
    log: bool,
    label: &str,
    arrow: Option<ReferenceArrow>,
) {
    let painter = ui.painter_at(rect);
    let text_color = ui.visuals().text_color();
    let font = egui::FontId::proportional(12.0);

    let rect = match &arrow {
        Some(a) if rect.height() > 3.0 * ARROW_ROW => {
            let row = egui::Rect::from_min_max(egui::pos2(rect.left(), rect.bottom() - ARROW_ROW), rect.max);
            reference_arrow(&painter, row, a, text_color);
            egui::Rect::from_min_max(rect.min, egui::pos2(rect.right(), row.top()))
        }
        _ => rect,
    };
    let (galley, bar) = layout(&painter, rect, label, font.clone(), text_color);
    if bar.height() < 8.0 {
        return;
    }
    painter.galley(rect.left_top(), galley, text_color);

    let lut = colormap.lut();
    let mut mesh = egui::Mesh::default();
    let n = lut.len();
    for (i, c) in lut.iter().enumerate() {
        let y1 = bar.bottom() - bar.height() * i as f32 / n as f32;
        let y0 = bar.bottom() - bar.height() * (i + 1) as f32 / n as f32;
        let col = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
        mesh.add_colored_rect(egui::Rect::from_min_max(egui::pos2(bar.left(), y0), egui::pos2(bar.right(), y1)), col);
    }
    painter.add(egui::Shape::mesh(mesh));

    let (lo, hi) = (range.0 as f64, range.1 as f64);
    let max_ticks = ((bar.height() / 28.0) as usize).clamp(2, 10);
    let use_log = log && lo > 0.0 && hi > 0.0;
    let frac = |t: f64| -> f32 {
        let f = if use_log { (t.ln() - lo.ln()) / (hi.ln() - lo.ln()) } else { (t - lo) / (hi - lo) };
        if f.is_finite() { f as f32 } else { 0.5 }
    };
    for t in nice_ticks(lo, hi, max_ticks, use_log) {
        let y = bar.bottom() - bar.height() * frac(t).clamp(0.0, 1.0);
        painter.line_segment(
            [egui::pos2(bar.right(), y), egui::pos2(bar.right() + 4.0, y)],
            egui::Stroke::new(1.0, text_color),
        );
        painter.text(egui::pos2(bar.right() + 6.0, y), egui::Align2::LEFT_CENTER, fmt_value(t), font.clone(), text_color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torcdf_core::{Attr, AttrValue, DType, Storage};

    fn var(attrs: Vec<(&str, &str)>) -> VarInfo {
        VarInfo {
            name: "z".into(),
            path: "/z".into(),
            dims: vec![],
            shape: vec![],
            dtype: DType::F32,
            attrs: attrs
                .into_iter()
                .map(|(n, v)| Attr { name: n.into(), value: AttrValue::Text(v.into()) })
                .collect(),
            storage: Storage::Contiguous,
        }
    }

    #[test]
    fn linear_ticks() {
        assert_eq!(nice_ticks(0.0, 1.0, 6, false), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    }
    #[test]
    fn linear_ticks_negative() {
        assert_eq!(
            nice_ticks(-1234.0, 5678.0, 8, false),
            vec![-1000.0, 0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        );
    }
    #[test]
    fn log_ticks() {
        assert_eq!(nice_ticks(0.01, 1000.0, 8, true), vec![0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]);
    }
    #[test]
    fn flat_range_tick() {
        assert_eq!(nice_ticks(3.0, 3.0, 5, false), vec![3.0]);
    }
    #[test]
    fn degenerate_inputs() {
        assert!(nice_ticks(f64::NAN, 1.0, 5, false).is_empty());
        assert!(nice_ticks(0.0, 1.0, 0, false).is_empty());
        assert_eq!(nice_ticks(1.0, 0.0, 6, false), nice_ticks(0.0, 1.0, 6, false));
        // log with non-positive lo, or under two decades, falls back to linear
        assert_eq!(nice_ticks(0.0, 1.0, 6, true), nice_ticks(0.0, 1.0, 6, false));
        assert_eq!(nice_ticks(2.0, 5.0, 4, true), nice_ticks(2.0, 5.0, 4, false));
    }
    #[test]
    fn reference_steps_down_to_fit() {
        let (v, len) = fit_reference(0.5, 300.0, 100.0);
        assert!((v - 0.1).abs() < 1e-12 && (len - 60.0).abs() < 1e-3, "{v} {len}");
        let (v, len) = fit_reference(200.0, 90.0, 100.0);
        assert_eq!((v, len), (200.0, 90.0), "fits already");
        let (v, len) = fit_reference(1.0, 150.0, 100.0);
        assert!((v - 0.5).abs() < 1e-12 && (len - 75.0).abs() < 1e-3, "{v} {len}");
    }

    #[test]
    fn label_text() {
        let v = var(vec![("long_name", "Elevation relative to sea level"), ("units", "m")]);
        assert_eq!(colorbar_label(&v), "Elevation relative to sea level [m]");
        assert_eq!(colorbar_label(&var(vec![])), "z");
        assert_eq!(colorbar_label(&var(vec![("standard_name", "height")])), "height");
    }

    #[test]
    fn long_label_wraps_inside_the_strip_above_the_bar() {
        // The strip the plot area gives the colour bar (130 pt less margins).
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(122.0, 400.0));
        let font = egui::FontId::proportional(12.0);
        let ctx = egui::Context::default();
        let mut out = ctx.run_ui(Default::default(), |ui| {
            let painter = ui.painter_at(rect);
            let lay = |s: &str| layout(&painter, rect, s, font.clone(), egui::Color32::WHITE);
            let (short, _) = lay("z");
            for long in ["Elevation relative to sea level [m]", "Eastward Velocity [m s-1]"] {
                let (g, bar) = lay(long);
                assert!(g.size().x <= rect.width(), "{long}: {:?}", g.size());
                assert!(g.size().y > short.size().y, "{long} wraps to more lines");
                assert!(g.text().ends_with(']'), "units kept");
                assert!(bar.top() >= rect.top() + g.size().y, "{long}: bar {bar:?}");
            }
            // Tick labels (centred on the bar ends) stay inside the strip.
            let (_, bar) = lay("z");
            let half_row = 0.5 * short.size().y;
            assert!(bar.bottom() + half_row <= rect.bottom(), "{bar:?}");
            assert!(bar.top() - half_row >= rect.top() + short.size().y, "{bar:?}");
        });
        out.textures_delta.clear(); // no renderer in this test
    }
}
