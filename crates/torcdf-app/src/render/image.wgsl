// Colour-mapped image layer. Data is an unfilterable R32Float texture and the
// colour map a 256 x 1 Rgba8Unorm LUT; both are read with textureLoad only
// (no samplers, no float filtering), which keeps this within downlevel limits.

struct Params {
    // t = (w - offset) * scale, where w = v, or log2(v) in log mode.
    // scale == 0 marks a degenerate range: every valid value maps to LUT[0].
    offset: f32,
    scale: f32,
    log_scale: u32,
    flip_y: u32,
    // Data-space rectangle shown, in cell units (x = column, y = row index).
    view_min: vec2<f32>,
    view_max: vec2<f32>,
    // The callback rectangle in framebuffer pixels, unclamped. egui clamps the
    // viewport to the screen, so positions are mapped from this instead.
    rect_min_px: vec2<f32>,
    rect_size_px: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var data_tex: texture_2d<f32>;
@group(0) @binding(2) var lut_tex: texture_2d<f32>;

// One triangle covering the whole viewport.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// NaN or +-inf, tested on the bits so no compiler can fold `v != v` away.
fn non_finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u;
}

// Premultiplied, gamma-space colour of the cell under framebuffer position `pos`.
fn shade(pos: vec2<f32>) -> vec4<f32> {
    let uv = (pos - params.rect_min_px) / params.rect_size_px;
    let size = params.view_max - params.view_min;
    let x = params.view_min.x + uv.x * size.x;
    var y = params.view_min.y + uv.y * size.y;
    if params.flip_y != 0u {
        y = params.view_max.y - uv.y * size.y;
    }
    let dims = vec2<f32>(textureDimensions(data_tex));
    if !(x >= 0.0 && y >= 0.0 && x < dims.x && y < dims.y) {
        return vec4<f32>(0.0);
    }
    let v = textureLoad(data_tex, vec2<i32>(i32(floor(x)), i32(floor(y))), 0).r;
    if non_finite(v) {
        return vec4<f32>(0.0);
    }
    var w = v;
    if params.log_scale != 0u {
        if v <= 0.0 {
            return vec4<f32>(0.0);
        }
        w = log2(v);
    }
    let t = clamp((w - params.offset) * params.scale, 0.0, 1.0);
    let index = min(i32(t * 256.0), 255);
    return textureLoad(lut_tex, vec2<i32>(index, 0), 0);
}

// egui's preferred target: Rgba8Unorm / Bgra8Unorm holding gamma-space colour.
@fragment
fn fs_main_gamma_framebuffer(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return shade(pos.xy);
}

fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// sRGB targets encode on write, so hand them linear colour.
@fragment
fn fs_main_linear_framebuffer(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = shade(pos.xy);
    return vec4<f32>(linear_from_gamma_rgb(c.rgb), c.a);
}
