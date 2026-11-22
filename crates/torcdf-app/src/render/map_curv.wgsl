// Curvilinear grid drawn as a triangle mesh. The CPU projects the mesh to
// world units (map::mesh::project_mesh); each vertex carries its tile texel
// [x, y] (column, row), which the rasteriser interpolates. The fragment
// shader rounds it to the nearest texel and reads the data with textureLoad.
// Prepended at build time: colour.wgsl, projection.wgsl.

struct Params {
    camera: Camera,
    // t = (w - offset) * scale, as in image.wgsl.
    offset: f32,
    scale: f32,
    log_scale: u32,
    // Framebuffer size in pixels; the paint callback sets a full-screen
    // viewport so framebuffer pixels map to clip space through this.
    screen_px: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var data_tex: texture_2d<f32>;
@group(0) @binding(2) var lut_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) texel: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) w: vec2<f32>,
    @location(1) texel: vec2<f32>,
    @builtin(instance_index) inst: u32,
) -> VsOut {
    // Cylindrical projections (ids 0 and 1) draw three copies at -2pi, 0, +2pi.
    var shift = 0.0;
    if params.camera.proj <= 1u {
        shift = (f32(inst) - 1.0) * PROJ_TAU;
    }
    let px = world_to_px(params.camera, vec2<f32>(w.x + shift, w.y));
    let ndc = vec2<f32>(px.x / params.screen_px.x * 2.0 - 1.0, 1.0 - px.y / params.screen_px.y * 2.0);
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.texel = texel;
    return out;
}

// Premultiplied, gamma-space colour for an interpolated texel coordinate.
fn shade(texel: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(data_tex));
    let t = clamp(vec2<i32>(floor(texel + vec2<f32>(0.5))), vec2<i32>(0), dims - vec2<i32>(1));
    let v = textureLoad(data_tex, t, 0).r;
    return colour_of(v, params.offset, params.scale, params.log_scale, lut_tex);
}

// egui's preferred target: Rgba8Unorm / Bgra8Unorm holding gamma-space colour.
@fragment
fn fs_main_gamma_framebuffer(in: VsOut) -> @location(0) vec4<f32> {
    return shade(in.texel);
}

// sRGB targets encode on write, so hand them linear colour.
@fragment
fn fs_main_linear_framebuffer(in: VsOut) -> @location(0) vec4<f32> {
    let c = shade(in.texel);
    return vec4<f32>(linear_from_gamma_rgb(c.rgb), c.a);
}
