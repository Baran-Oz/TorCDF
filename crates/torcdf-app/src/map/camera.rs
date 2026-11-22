//! Map camera: pan, zoom and fit in world units. `scale` is in screen points
//! (egui logical units) per world unit.

use super::projection::Projection;
use eframe::egui::{Pos2, Rect, Vec2};
use std::f64::consts::{PI, TAU};

pub const MAX_SCALE: f64 = 1.0e6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapCamera {
    pub proj: Projection,
    pub centre: [f64; 2],
    pub scale: f64,
}

/// Smallest allowed scale: cylindrical worlds stay at least as wide as the
/// rect; the polar disc (4 world units across) at least as large as the
/// smaller rect side.
pub fn min_scale(proj: Projection, rect: Rect) -> f64 {
    let (w, h) = (rect.width() as f64, rect.height() as f64);
    // The floor is a backstop so the scale can never be zero or negative.
    let s = if proj.wraps_x() { w / TAU } else { w.min(h) / 4.0 };
    if s.is_finite() {
        s.max(1e-9)
    } else {
        1e-9
    }
}

fn valid_rect(rect: Rect) -> bool {
    rect.width() > 0.0
        && rect.height() > 0.0
        && rect.width().is_finite()
        && rect.height().is_finite()
}

fn wrap_pi(x: f64) -> f64 {
    let w = (x + PI).rem_euclid(TAU) - PI;
    if w >= PI {
        w - TAU
    } else {
        w
    }
}

impl MapCamera {
    pub fn new(proj: Projection) -> MapCamera {
        let scale = if proj.wraps_x() {
            // World is 2*pi wide; height is pi (equirect) or 2*pi (Mercator).
            let h = if proj == Projection::Equirectangular { PI } else { TAU };
            (1000.0 / TAU).min(500.0 / h)
        } else {
            500.0 / 4.0
        };
        MapCamera { proj, centre: [0.0, 0.0], scale }
    }

    pub fn world_to_screen(&self, w: [f64; 2], rect: Rect) -> Pos2 {
        let c = rect.center();
        Pos2::new(
            (c.x as f64 + (w[0] - self.centre[0]) * self.scale) as f32,
            (c.y as f64 - (w[1] - self.centre[1]) * self.scale) as f32,
        )
    }

    pub fn screen_to_world(&self, p: Pos2, rect: Rect) -> [f64; 2] {
        let c = rect.center();
        [
            self.centre[0] + (p.x as f64 - c.x as f64) / self.scale,
            self.centre[1] - (p.y as f64 - c.y as f64) / self.scale,
        ]
    }

    pub fn screen_to_lonlat(&self, p: Pos2, rect: Rect) -> Option<(f64, f64)> {
        let w = self.screen_to_world(p, rect);
        self.proj.inverse(w[0], w[1])
    }

    /// Content follows the pointer. Only the centre wrap applies here (the
    /// scale is unchanged), so no rect is needed.
    pub fn pan(&mut self, delta: Vec2) {
        if !delta.x.is_finite() || !delta.y.is_finite() {
            return;
        }
        self.centre[0] -= delta.x as f64 / self.scale;
        self.centre[1] += delta.y as f64 / self.scale;
        self.wrap_centre();
    }

    pub fn zoom_at(&mut self, p: Pos2, factor: f64, rect: Rect) {
        if !valid_rect(rect)
            || !p.x.is_finite()
            || !p.y.is_finite()
            || !(factor.is_finite() && factor > 0.0)
        {
            return;
        }
        let w = self.screen_to_world(p, rect);
        self.clamp_scale(self.scale * factor, rect);
        let c = rect.center();
        self.centre[0] = w[0] - (p.x as f64 - c.x as f64) / self.scale;
        self.centre[1] = w[1] + (p.y as f64 - c.y as f64) / self.scale;
        self.clamp(rect);
    }

    /// Frame the projectable points with a 5% margin. Unprojectable points are
    /// skipped; if none remain the camera is unchanged. A single point
    /// centres on it and keeps the current scale.
    pub fn fit(&mut self, lonlat: &[(f64, f64)], rect: Rect) {
        let pts: Vec<[f64; 2]> =
            lonlat.iter().filter_map(|&(lo, la)| self.proj.forward(lo, la)).collect();
        if pts.is_empty() || !valid_rect(rect) {
            return;
        }
        let mut xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
        if self.proj.wraps_x() {
            let (s, c) = xs.iter().fold((0.0, 0.0), |(s, c), x| (s + x.sin(), c + x.cos()));
            let mean = if s.hypot(c) > 1e-9 { s.atan2(c) } else { xs[0] };
            for x in &mut xs {
                *x = mean + wrap_pi(*x - mean);
            }
        }
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for (x, p) in xs.iter().zip(&pts) {
            x0 = x0.min(*x);
            x1 = x1.max(*x);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
        let (dx, dy) = (x1 - x0, y1 - y0);
        let mut best = f64::INFINITY;
        if dx > 1e-12 {
            best = best.min(rect.width() as f64 / (dx * 1.05));
        }
        if dy > 1e-12 {
            best = best.min(rect.height() as f64 / (dy * 1.05));
        }
        self.centre = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
        if best.is_finite() {
            self.scale = best;
        }
        self.clamp(rect);
    }

    pub fn clamp(&mut self, rect: Rect) {
        if !valid_rect(rect) {
            return;
        }
        self.clamp_scale(self.scale, rect);
        self.wrap_centre();
    }

    fn clamp_scale(&mut self, s: f64, rect: Rect) {
        self.scale = s.max(min_scale(self.proj, rect)).min(MAX_SCALE);
    }

    fn wrap_centre(&mut self) {
        if self.proj.wraps_x() {
            self.centre[0] = wrap_pi(self.centre[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> Rect {
        Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(800.0, 600.0))
    }

    #[test]
    fn screen_world_round_trip() {
        let cams = [
            MapCamera::new(Projection::Equirectangular),
            MapCamera { proj: Projection::Mercator, centre: [1.0, 0.5], scale: 300.0 },
            MapCamera { proj: Projection::NorthPolarStereo, centre: [0.3, -0.2], scale: 150.0 },
        ];
        let pts = [(10.0, 20.0), (400.0, 300.0), (810.0, 620.0), (123.0, 456.0), (700.0, 50.0)];
        for cam in cams {
            for (x, y) in pts {
                let p = Pos2::new(x, y);
                let q = cam.world_to_screen(cam.screen_to_world(p, rect()), rect());
                assert!((p.x - q.x).abs() < 1e-3 && (p.y - q.y).abs() < 1e-3, "{cam:?} {p:?} {q:?}");
            }
        }
    }

    #[test]
    fn zoom_keeps_anchor() {
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.scale = 200.0;
        let p = Pos2::new(250.0, 180.0);
        let w = cam.screen_to_world(p, rect());
        cam.zoom_at(p, 2.0, rect());
        let w2 = cam.screen_to_world(p, rect());
        assert!((w[0] - w2[0]).abs() < 1e-9 && (w[1] - w2[1]).abs() < 1e-9);
        assert_eq!(cam.scale, 400.0);
    }

    #[test]
    fn pan_moves_content() {
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.scale = 200.0;
        let c0 = cam.centre;
        cam.pan(Vec2::new(10.0, 0.0));
        assert!((cam.centre[0] - (c0[0] - 10.0 / 200.0)).abs() < 1e-12);
        let c1 = cam.centre;
        cam.pan(Vec2::new(0.0, 10.0));
        assert!((cam.centre[1] - (c1[1] + 10.0 / 200.0)).abs() < 1e-12);
    }

    fn corners(l0: f64, l1: f64, a0: f64, a1: f64) -> Vec<(f64, f64)> {
        vec![(l0, a0), (l1, a0), (l0, a1), (l1, a1)]
    }

    #[test]
    fn fit_frames_points() {
        let r = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut cam = MapCamera::new(Projection::Equirectangular);
        let pts = corners(-6.0, 36.0, 30.0, 46.0);
        cam.fit(&pts, r);
        let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for (lo, la) in pts {
            let p = cam.world_to_screen(cam.proj.forward(lo, la).unwrap(), r);
            assert!(r.contains(p), "{p:?}");
            x0 = x0.min(p.x);
            x1 = x1.max(p.x);
            y0 = y0.min(p.y);
            y1 = y1.max(p.y);
        }
        assert!((x1 - x0) >= 0.9 * 800.0 || (y1 - y0) >= 0.9 * 600.0 * 0.95);
    }

    #[test]
    fn fit_across_dateline() {
        let r = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.fit(&[(170.0, 10.0), (-170.0, 10.0), (170.0, -10.0), (-170.0, -10.0)], r);
        let w = 20.0f64.to_radians() * cam.scale;
        let h = w; // 20 degrees tall as well
        assert!(w <= 800.0 + 1e-6 && h <= 600.0 + 1e-6);
        assert!(w >= 0.8 * 800.0 || h >= 0.8 * 600.0, "scale {}", cam.scale);
        // Framed around the dateline.
        let p = cam.world_to_screen([-PI, 0.0], r);
        assert!((p.x - 400.0).abs() < 1.0, "{p:?}");
    }

    #[test]
    fn centre_wraps_for_cylindrical() {
        let mut cam = MapCamera::new(Projection::Equirectangular);
        for _ in 0..500 {
            cam.pan(Vec2::new(-300.0, 0.0));
            assert!((-PI..PI).contains(&cam.centre[0]), "{}", cam.centre[0]);
        }
    }

    #[test]
    fn scale_is_clamped() {
        let r = rect();
        let mut cam = MapCamera::new(Projection::Equirectangular);
        let p = Pos2::new(300.0, 250.0);
        let w = cam.screen_to_world(p, r);
        cam.zoom_at(p, 1e12, r);
        assert_eq!(cam.scale, MAX_SCALE);
        let w2 = cam.screen_to_world(p, r);
        assert!((w[0] - w2[0]).abs() < 1e-9 && (w[1] - w2[1]).abs() < 1e-9);
        cam.zoom_at(p, 1e-12, r);
        assert_eq!(cam.scale, min_scale(Projection::Equirectangular, r));
        let w3 = cam.screen_to_world(p, r);
        assert!((w[0] - w3[0]).abs() < 1e-9 && (w[1] - w3[1]).abs() < 1e-9);
    }

    #[test]
    fn empty_rect_is_harmless() {
        let neg = Rect::from_min_max(Pos2::new(10.0, 10.0), Pos2::new(-5.0, -5.0));
        for bad in [Rect::ZERO, neg] {
            let mut cam = MapCamera::new(Projection::Equirectangular);
            let before = cam;
            cam.fit(&[(0.0, 0.0), (10.0, 10.0)], bad);
            cam.zoom_at(Pos2::new(1.0, 1.0), 2.0, bad);
            cam.clamp(bad);
            assert_eq!(cam, before);
        }
        let mut cam = MapCamera::new(Projection::Equirectangular);
        cam.fit(&[(0.0, 0.0), (10.0, 10.0)], Rect::ZERO);
        let r = rect();
        cam.fit(&[(0.0, 0.0), (10.0, 10.0)], r);
        assert!(cam.scale > 0.0 && cam.scale.is_finite() && cam.centre.iter().all(|c| c.is_finite()));
        cam.pan(Vec2::new(5.0, 5.0));
        assert!(cam.centre.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn non_finite_inputs_ignored() {
        let r = rect();
        let mut cam = MapCamera::new(Projection::Mercator);
        let before = cam;
        cam.pan(Vec2::new(f32::NAN, 1.0));
        cam.pan(Vec2::new(1.0, f32::INFINITY));
        cam.zoom_at(Pos2::new(f32::NAN, 5.0), 2.0, r);
        for f in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            cam.zoom_at(r.center(), f, r);
        }
        assert_eq!(cam, before);
    }
}
