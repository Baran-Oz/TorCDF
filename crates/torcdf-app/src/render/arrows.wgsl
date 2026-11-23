// Vector arrows: one instance per arrow, a 9-vertex glyph (a shaft quad and
// a head triangle) built in framebuffer pixels around the projected tail.
// Drawn twice: a light outline one pixel wider, then the dark arrow.
// Prepended at build time: colour.wgsl (linear_from_gamma_rgb).

struct Arrows {
    // Framebuffer px = [a b; c d] * world + [e f]: lin = (a, b, c, d).
    lin: vec4<f32>,
    off: vec2<f32>,
    // Framebuffer size; the paint callback sets a full-screen viewport.
    screen_px: vec2<f32>,
    // The plot rect in framebuffer pixels (x, y, w, h), for culling.
    rect_px: vec4<f32>,
    // Arrow length in framebuffer pixels per unit of magnitude.
    px_per_unit: f32,
    // Non-zero: vertex copies 0, 1, 2 draw shifted by -2pi, 0, +2pi in x.
    wrap: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> arrows: Arrows;

const ARROW_TAU: f32 = 6.28318530717959;
const GLYPH_VERTS: u32 = 9u;
const HEAD_MAX_PX: f32 = 8.0;
const SHAFT_HALF_PX: f32 = 0.75;
// Every corner here: a degenerate triangle outside clip space.
const OUTSIDE: vec4<f32> = vec4<f32>(2.0, 2.0, 0.0, 1.0);

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) colour: vec4<f32>,
};

fn lin_apply(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(arrows.lin.x * v.x + arrows.lin.y * v.y, arrows.lin.z * v.x + arrows.lin.w * v.y);
}

// Corner `k` of a glyph `len` px long in arrow space (x along the arrow,
// y across), grown by `grow` px for the outline.
fn corner(k: u32, len: f32, grow: f32) -> vec2<f32> {
    let head = min(0.3 * len, HEAD_MAX_PX);
    let hs = SHAFT_HALF_PX + 0.5 * grow;
    let x0 = -0.5 * grow;
    let x1 = len - head;
    let hw = 0.5 * head + grow;
    let hx = x1 - 0.5 * grow;
    let tip = len + grow;
    var c = array<vec2<f32>, 9>(
        vec2<f32>(x0, -hs), vec2<f32>(x1, -hs), vec2<f32>(x1, hs),
        vec2<f32>(x0, -hs), vec2<f32>(x1, hs), vec2<f32>(x0, hs),
        vec2<f32>(hx, -hw), vec2<f32>(tip, 0.0), vec2<f32>(hx, hw),
    );
    return c[k];
}

fn glyph(vi: u32, pos: vec2<f32>, dir: vec2<f32>, mag: f32, grow: f32) -> vec4<f32> {
    let len = mag * arrows.px_per_unit;
    // Under a pixel (or NaN): not drawn.
    if !(len >= 1.0) {
        return OUTSIDE;
    }
    var shift = 0.0;
    if arrows.wrap != 0u {
        shift = (f32(vi / GLYPH_VERTS) - 1.0) * ARROW_TAU;
    }
    let base = lin_apply(vec2<f32>(pos.x + shift, pos.y)) + arrows.off;
    // World y up becomes framebuffer y down through the linear part.
    let sd = lin_apply(dir);
    let sl = length(sd);
    if !(sl > 0.0) {
        return OUTSIDE;
    }
    let reach = len + 2.0 * HEAD_MAX_PX;
    let r = arrows.rect_px;
    if base.x < r.x - reach || base.x > r.x + r.z + reach || base.y < r.y - reach || base.y > r.y + r.w + reach {
        return OUTSIDE;
    }
    let u = sd / sl;
    let v = vec2<f32>(-u.y, u.x);
    let c = corner(vi % GLYPH_VERTS, len, grow);
    let px = base + u * c.x + v * c.y;
    return vec4<f32>(px.x / arrows.screen_px.x * 2.0 - 1.0, 1.0 - px.y / arrows.screen_px.y * 2.0, 0.0, 1.0);
}

@vertex
fn vs_outline(
    @builtin(vertex_index) vi: u32,
    @location(0) pos: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) mag: f32,
) -> VsOut {
    // #F0F0F0
    return VsOut(glyph(vi, pos, dir, mag, 1.0), vec4<f32>(vec3<f32>(240.0 / 255.0), 1.0));
}

@vertex
fn vs_fill(
    @builtin(vertex_index) vi: u32,
    @location(0) pos: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) mag: f32,
) -> VsOut {
    // #303030
    return VsOut(glyph(vi, pos, dir, mag, 0.0), vec4<f32>(vec3<f32>(48.0 / 255.0), 1.0));
}

@fragment
fn fs_main_gamma_framebuffer(in: VsOut) -> @location(0) vec4<f32> {
    return in.colour;
}

@fragment
fn fs_main_linear_framebuffer(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(linear_from_gamma_rgb(in.colour.rgb), in.colour.a);
}
