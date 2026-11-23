//! PNG export: cropping a screenshot to a rectangle, encoding it, and the
//! default file name.

use eframe::egui;

use super::var_view::VarView;

/// The part of `image` under `rect` (in points), at `ppp` pixels per point.
/// The rectangle is clipped to the image.
pub fn crop(image: &egui::ColorImage, rect: egui::Rect, ppp: f32) -> egui::ColorImage {
    let [w, h] = image.size;
    let edge = |v: f32, max: usize| ((v * ppp).round().max(0.0) as usize).min(max);
    let (x0, x1) = (edge(rect.min.x, w), edge(rect.max.x, w));
    let (y0, y1) = (edge(rect.min.y, h), edge(rect.max.y, h));
    let (cw, ch) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let mut pixels = Vec::with_capacity(cw * ch);
    for y in y0..y0 + ch {
        pixels.extend_from_slice(&image.pixels[y * w + x0..y * w + x0 + cw]);
    }
    egui::ColorImage { size: [cw, ch], source_size: egui::vec2(cw as f32, ch as f32), pixels }
}

/// The image as an 8-bit RGBA PNG.
pub fn encode_png(image: &egui::ColorImage) -> Result<Vec<u8>, String> {
    let [w, h] = image.size;
    if w == 0 || h == 0 {
        return Err("There is nothing to save: the image is empty.".into());
    }
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| format!("Cannot encode the PNG: {e}"))?;
    let bytes: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_srgba_unmultiplied()).collect();
    writer.write_image_data(&bytes).map_err(|e| format!("Cannot encode the PNG: {e}"))?;
    writer.finish().map_err(|e| format!("Cannot encode the PNG: {e}"))?;
    Ok(out)
}

/// Letters, digits, `-` and `.` stay; every run of anything else is one `_`.
pub fn sanitise(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '-' || c == '.' {
            out.push(c);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

/// `<var leaf>[_<dim>-<index>…].png`, one part per slider dimension.
pub fn default_name(view: &VarView) -> String {
    let mut name = sanitise(&view.var.name);
    if name.is_empty() {
        name = "plot".into();
    }
    for d in view.slider_dims() {
        let dim = sanitise(&view.var.dims[d]);
        name.push_str(&format!("_{dim}-{}", view.slider_pos[d]));
    }
    name.push_str(".png");
    name
}

/// `<var name>.png` for a pop-up.
pub fn popup_name(var_name: &str) -> String {
    let n = sanitise(var_name);
    format!("{}.png", if n.is_empty() { "popup" } else { &n })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use egui::Color32;
    use torcdf_core::{NcFile, fixtures};

    use super::*;

    fn gradient(w: usize, h: usize) -> egui::ColorImage {
        let mut img = egui::ColorImage::filled([w, h], Color32::BLACK);
        for y in 0..h {
            for x in 0..w {
                img.pixels[y * w + x] = Color32::from_rgb(x as u8, y as u8, 7);
            }
        }
        img
    }

    #[test]
    fn crop_scales_by_ppp() {
        let c = crop(&gradient(20, 10), egui::Rect::from_min_max(egui::pos2(2.0, 1.0), egui::pos2(6.0, 4.0)), 2.0);
        assert_eq!(c.size, [8, 6]);
        assert_eq!(c.pixels[0], Color32::from_rgb(4, 2, 7));
        assert_eq!(c.pixels[c.pixels.len() - 1], Color32::from_rgb(11, 7, 7));
    }

    #[test]
    fn crop_clamps_to_image() {
        let img = gradient(20, 10);
        let c = crop(&img, egui::Rect::from_min_max(egui::pos2(8.0, 3.0), egui::pos2(50.0, 50.0)), 1.0);
        assert_eq!(c.size, [12, 7]);
        let off = crop(&img, egui::Rect::from_min_max(egui::pos2(30.0, 30.0), egui::pos2(40.0, 40.0)), 1.0);
        assert_eq!(off.size, [0, 0]);
        let neg = crop(&img, egui::Rect::from_min_max(egui::pos2(-5.0, -5.0), egui::pos2(2.0, 2.0)), 1.0);
        assert_eq!(neg.size, [2, 2]);
    }

    #[test]
    fn png_round_trips() {
        let img = gradient(13, 5);
        let bytes = encode_png(&img).unwrap();
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (13, 5));
        assert_eq!(info.color_type, png::ColorType::Rgba);
        let want: Vec<u8> = img.pixels.iter().flat_map(|p| p.to_array()).collect();
        assert_eq!(&buf[..info.buffer_size()], &want[..]);
        assert!(encode_png(&egui::ColorImage::filled([0, 3], Color32::BLACK)).is_err());
    }

    #[test]
    fn default_name_has_slider_positions() {
        let d = tempfile::tempdir().unwrap();
        let nc = Arc::new(NcFile::open(&fixtures::write_classic(d.path())).unwrap());
        let info = nc.meta.all_vars().into_iter().find(|v| v.dims.len() == 3).expect("a 3-D variable").clone();
        let mut v = VarView::new(&egui::Context::default(), 1, nc, info);
        // Make it 4-D: a trailing dimension that is not plotted.
        v.var.name = "thetao".into();
        v.var.dims.push("depth".into());
        v.var.shape.push(5);
        v.slider_pos.push(0);
        let time = v.var.dims.iter().position(|n| n == "time").unwrap();
        v.slider_pos[time] = 3;
        assert_eq!(default_name(&v), "thetao_time-3_depth-0.png");
        v.var.name = "wind (wind_speed, wind_direction)".into();
        assert_eq!(default_name(&v), "wind_wind_speed_wind_direction_time-3_depth-0.png");
    }
}
