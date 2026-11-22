// Colour-mapped image layer. Data is an unfilterable R32Float texture and the
// colour map a 256 x 1 Rgba8Unorm LUT; both are read with textureLoad only
// (no samplers, no float filtering), which keeps this within downlevel limits.
// Prepended at build time: colour.wgsl (non_finite, colour_of,
// linear_from_gamma_rgb).

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
    return colour_of(v, params.offset, params.scale, params.log_scale, lut_tex);
}

// egui's preferred target: Rgba8Unorm / Bgra8Unorm holding gamma-space colour.
@fragment
fn fs_main_gamma_framebuffer(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return shade(pos.xy);
}

// sRGB targets encode on write, so hand them linear colour.
@fragment
fn fs_main_linear_framebuffer(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = shade(pos.xy);
    return vec4<f32>(linear_from_gamma_rgb(c.rgb), c.a);
}
