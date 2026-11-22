// Shared colouring (plan 1a): value -> premultiplied gamma-space colour via a
// 256 x 1 Rgba8Unorm LUT read with textureLoad. Binding-free: shaders include
// this with concat!(include_str!(..)) and pass their own LUT texture.

// NaN or +-inf, tested on the bits so no compiler can fold `v != v` away.
fn non_finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u;
}

// Colour of data value `v`, transparent when missing. t = (w - offset) * scale,
// where w = v, or log2(v) in log mode (v <= 0 is then transparent).
// scale == 0 marks a degenerate range: every valid value maps to LUT[0].
fn colour_of(v: f32, offset: f32, scale: f32, log_scale: u32, lut: texture_2d<f32>) -> vec4<f32> {
    if non_finite(v) {
        return vec4<f32>(0.0);
    }
    var w = v;
    if log_scale != 0u {
        if v <= 0.0 {
            return vec4<f32>(0.0);
        }
        w = log2(v);
    }
    let t = clamp((w - offset) * scale, 0.0, 1.0);
    let index = min(i32(t * 256.0), 255);
    return textureLoad(lut, vec2<i32>(index, 0), 0);
}

// sRGB targets encode on write, so they are handed linear colour.
fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}
