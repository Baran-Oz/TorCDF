// Line overlays (coastlines, borders, graticule): a LineList of world-space
// endpoints placed by the map camera. 1 px lines, no wide-line features.
// Prepended at build time: colour.wgsl, projection.wgsl (Camera, world_to_px).

struct Lines {
    camera: Camera,
    // Premultiplied gamma-space colour (egui's Color32).
    colour: vec4<f32>,
    // Framebuffer size in pixels; the paint callback sets a full-screen
    // viewport so framebuffer pixels map to clip space through this.
    screen_px: vec2<f32>,
    // Non-zero: instances 0, 1, 2 draw shifted by -2pi, 0, +2pi in x.
    wrap: u32,
};

@group(0) @binding(0) var<uniform> lines: Lines;

@vertex
fn vs_main(@location(0) w: vec2<f32>, @builtin(instance_index) inst: u32) -> @builtin(position) vec4<f32> {
    var shift = 0.0;
    if lines.wrap != 0u {
        shift = (f32(inst) - 1.0) * PROJ_TAU;
    }
    // Snap to the centre of the pixel containing the point, so a 1 px line
    // lights a definite pixel instead of straddling a pixel boundary.
    let px = floor(world_to_px(lines.camera, vec2<f32>(w.x + shift, w.y))) + 0.5;
    let ndc = vec2<f32>(px.x / lines.screen_px.x * 2.0 - 1.0, 1.0 - px.y / lines.screen_px.y * 2.0);
    return vec4<f32>(ndc, 0.0, 1.0);
}

@fragment
fn fs_main_gamma_framebuffer() -> @location(0) vec4<f32> {
    return lines.colour;
}

@fragment
fn fs_main_linear_framebuffer() -> @location(0) vec4<f32> {
    let c = lines.colour;
    return vec4<f32>(linear_from_gamma_rgb(c.rgb), c.a);
}
