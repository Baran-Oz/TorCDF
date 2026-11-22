// Map projections on the unit sphere and the map camera. Mirrors
// src/map/projection.rs (forward and inverse, rejections and the longitude
// wrap to [-pi, pi)) and gpu_common::CameraUniform. Binding-free: shaders
// include this with concat!(include_str!(..)).
//
// Projection ids (Projection::shader_id): 0 equirectangular, 1 Mercator,
// 2 north polar stereographic, 3 south polar stereographic.
// Results are vec3: .xy the value, .z = 1 when valid and 0 otherwise.

const PROJ_PI: f32 = 3.14159265358979;
const PROJ_TAU: f32 = 6.28318530717959;
const PROJ_FRAC_PI_2: f32 = 1.57079632679490;
const PROJ_FRAC_PI_4: f32 = 0.78539816339745;
const MERCATOR_LIMIT_DEG: f32 = 85.051129;
// Polar inverse accepts rho <= 2 plus a tolerance so the equator (rho = 2)
// round-trips: 1e-9 on the CPU (f64), one f32 ulp at 2 here.
const POLAR_RHO_MAX: f32 = 2.0000002;

// gpu_common::CameraUniform (32 bytes).
struct Camera {
    // Camera centre in world units.
    centre: vec2<f32>,
    // Framebuffer pixels per world unit.
    scale_px: f32,
    // Projection id.
    proj: u32,
    // Map rect in framebuffer pixels, unclamped: x, y, w, h.
    rect_px: vec4<f32>,
};

// Framebuffer pixel position of world point `w` (world y up, screen y down).
fn world_to_px(cam: Camera, w: vec2<f32>) -> vec2<f32> {
    let c = cam.rect_px.xy + 0.5 * cam.rect_px.zw;
    let d = (w - cam.centre) * cam.scale_px;
    return vec2<f32>(c.x + d.x, c.y - d.y);
}

// World point under framebuffer pixel position `px`.
fn px_to_world(cam: Camera, px: vec2<f32>) -> vec2<f32> {
    let c = cam.rect_px.xy + 0.5 * cam.rect_px.zw;
    let d = (px - c) / cam.scale_px;
    return vec2<f32>(cam.centre.x + d.x, cam.centre.y - d.y);
}

fn proj_non_finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u;
}

// atan to ~2e-7 relative error (Cephes atanf). Built-in atan is only
// approximate on some adapters (about 2e-5 rad on WARP), which is visible at
// deep zoom.
fn proj_atan(v: f32) -> f32 {
    let a = abs(v);
    var base = 0.0;
    var x = a;
    if a > 2.414213562373095 {
        base = PROJ_FRAC_PI_2;
        x = -1.0 / a;
    } else if a > 0.4142135623730950 {
        base = PROJ_FRAC_PI_4;
        x = (a - 1.0) / (a + 1.0);
    }
    let z = x * x;
    let p = (((0.0805374449538 * z - 0.138776856032) * z + 0.199777106478) * z
        - 0.333329491539) * z * x + x;
    return sign(v) * (base + p);
}

// atan2 on proj_atan; 0 at the origin (where longitude is undefined).
fn proj_atan2(y: f32, x: f32) -> f32 {
    if x == 0.0 {
        if y == 0.0 {
            return 0.0;
        }
        return sign(y) * PROJ_FRAC_PI_2;
    }
    let t = proj_atan(y / x);
    if x > 0.0 {
        return t;
    }
    if y < 0.0 {
        return t - PROJ_PI;
    }
    return t + PROJ_PI;
}

// Wraps radians into [-pi, pi).
fn wrap_lon(lon: f32) -> f32 {
    let t = lon + PROJ_PI;
    var w = t - PROJ_TAU * floor(t / PROJ_TAU) - PROJ_PI;
    // f32 rounding can land just outside the half-open range.
    if w >= PROJ_PI {
        w = w - PROJ_TAU;
    }
    if w < -PROJ_PI {
        w = -PROJ_PI;
    }
    return w;
}

// (lon, lat) in degrees -> world xy.
fn forward(proj: u32, lonlat: vec2<f32>) -> vec3<f32> {
    let lon = lonlat.x;
    let lat = lonlat.y;
    if proj_non_finite(lon) || proj_non_finite(lat) || abs(lat) > 90.0 {
        return vec3<f32>(0.0);
    }
    let lam = wrap_lon(radians(lon));
    let phi = radians(lat);
    var out = vec3<f32>(0.0);
    switch proj {
        case 0u: {
            out = vec3<f32>(lam, phi, 1.0);
        }
        case 1u: {
            if abs(lat) <= MERCATOR_LIMIT_DEG {
                out = vec3<f32>(lam, log(tan(PROJ_FRAC_PI_4 + phi / 2.0)), 1.0);
            }
        }
        case 2u: {
            if lat >= 0.0 {
                let rho = 2.0 * tan(PROJ_FRAC_PI_4 - phi / 2.0);
                out = vec3<f32>(rho * sin(lam), -rho * cos(lam), 1.0);
            }
        }
        case 3u: {
            if lat <= 0.0 {
                let rho = 2.0 * tan(PROJ_FRAC_PI_4 + phi / 2.0);
                out = vec3<f32>(rho * sin(lam), rho * cos(lam), 1.0);
            }
        }
        default: {}
    }
    return out;
}

// World xy -> (lon, lat) in degrees, lon in [-180, 180).
fn inverse(proj: u32, xy: vec2<f32>) -> vec3<f32> {
    let x = xy.x;
    let y = xy.y;
    if proj_non_finite(x) || proj_non_finite(y) {
        return vec3<f32>(0.0);
    }
    var out = vec3<f32>(0.0);
    switch proj {
        case 0u: {
            if abs(y) <= PROJ_FRAC_PI_2 {
                out = vec3<f32>(degrees(wrap_lon(x)), degrees(y), 1.0);
            }
        }
        case 1u: {
            let phi = 2.0 * proj_atan(exp(y)) - PROJ_FRAC_PI_2;
            out = vec3<f32>(degrees(wrap_lon(x)), degrees(phi), 1.0);
        }
        case 2u, 3u: {
            let rho = length(xy);
            if rho <= POLAR_RHO_MAX {
                let c = 2.0 * proj_atan(rho / 2.0);
                var lam: f32;
                var phi: f32;
                if proj == 2u {
                    lam = proj_atan2(x, -y);
                    phi = PROJ_FRAC_PI_2 - c;
                } else {
                    lam = proj_atan2(x, y);
                    phi = c - PROJ_FRAC_PI_2;
                }
                out = vec3<f32>(degrees(wrap_lon(lam)), degrees(phi), 1.0);
            }
        }
        default: {}
    }
    return out;
}
