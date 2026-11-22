// Regular lat/lon grid drawn through the map camera. Per pixel: framebuffer
// position -> world -> inverse projection -> (lon, lat) -> nearest cell by
// binary search over the coordinate arrays (1-row R32Float textures, read with
// textureLoad only) -> colour. Points off the projection, off the grid by more
// than half an end cell, or on a missing value are transparent.
// Prepended at build time: colour.wgsl, projection.wgsl.

struct Params {
    camera: Camera,
    // t = (w - offset) * scale, as in image.wgsl.
    offset: f32,
    scale: f32,
    log_scale: u32,
    // Start of the 360 degree longitude window: min(lon) - half end spacing.
    lon_start: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var data_tex: texture_2d<f32>;
@group(0) @binding(2) var lut_tex: texture_2d<f32>;
@group(0) @binding(3) var lat_tex: texture_2d<f32>;
@group(0) @binding(4) var lon_tex: texture_2d<f32>;

// One triangle covering the whole viewport.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn coord(tex: texture_2d<f32>, i: u32) -> f32 {
    return textureLoad(tex, vec2<i32>(i32(i), 0), 0).r;
}

// Index of the coordinate nearest `x` in the strictly monotonic array `tex`
// (either direction), or -1 when `x` lies beyond an end by more than half
// that end cell's spacing (a one-cell axis has spacing 1).
fn nearest(tex: texture_2d<f32>, x: f32) -> i32 {
    let n = textureDimensions(tex).x;
    let first = coord(tex, 0u);
    var sgn = 1.0;
    var half_first = 0.5;
    var half_last = 0.5;
    var last = first;
    if n > 1u {
        last = coord(tex, n - 1u);
        sgn = select(-1.0, 1.0, last >= first);
        half_first = 0.5 * abs(coord(tex, 1u) - first);
        half_last = 0.5 * abs(last - coord(tex, n - 2u));
    }
    // Work in the increasing direction.
    let v = sgn * x;
    if v < sgn * first - half_first || v > sgn * last + half_last {
        return -1;
    }
    // Largest lo with sgn * c[lo] <= v (0 when none).
    var lo = 0u;
    var hi = n - 1u;
    while lo < hi {
        let mid = (lo + hi + 1u) / 2u;
        if sgn * coord(tex, mid) <= v {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    if lo + 1u < n {
        let a = sgn * coord(tex, lo);
        let b = sgn * coord(tex, lo + 1u);
        if v - a > b - v {
            lo = lo + 1u;
        }
    }
    return i32(lo);
}

// Premultiplied, gamma-space colour at framebuffer position `pos`.
fn shade(pos: vec2<f32>) -> vec4<f32> {
    let cam = params.camera;
    let world = px_to_world(cam, pos);
    // The inverse wraps cylindrical x into [-pi, pi).
    let ll = inverse(cam.proj, world);
    if ll.z == 0.0 {
        return vec4<f32>(0.0);
    }
    let d = ll.x - params.lon_start;
    let lon = params.lon_start + d - 360.0 * floor(d / 360.0);
    let col = nearest(lon_tex, lon);
    let row = nearest(lat_tex, ll.y);
    if col < 0 || row < 0 {
        return vec4<f32>(0.0);
    }
    let v = textureLoad(data_tex, vec2<i32>(col, row), 0).r;
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
