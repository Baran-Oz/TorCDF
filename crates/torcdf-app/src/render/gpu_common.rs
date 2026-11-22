//! GPU helpers shared by the image, map and line layers: the camera uniform,
//! data and LUT textures, and (in tests) a headless render harness.
//!
//! WGSL snippets shared the same way: `colour.wgsl` (value to colour) and
//! `projection.wgsl` (map projections and the `Camera` uniform). Shaders
//! include them with `concat!(include_str!(..))`.

use eframe::egui;
use eframe::egui_wgpu::wgpu;

use crate::map::camera::MapCamera;

/// The `Camera` struct of `projection.wgsl`. WGSL layout: `vec2<f32>` at 0,
/// `f32` at 8, `u32` at 12, `vec4<f32>` at 16; 32 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// Camera centre in world units.
    pub centre: [f32; 2],
    /// Framebuffer pixels per world unit.
    pub scale_px: f32,
    /// `Projection::shader_id`.
    pub proj: u32,
    /// The map rect in framebuffer pixels, unclamped: x, y, w, h.
    pub rect_px: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<CameraUniform>() == 32);

/// The camera uniform for drawing `cam` into `rect` (points).
pub fn camera_uniform(cam: &MapCamera, rect: egui::Rect, pixels_per_point: f32) -> CameraUniform {
    let r = rect * pixels_per_point;
    CameraUniform {
        centre: [cam.centre[0] as f32, cam.centre[1] as f32],
        scale_px: (cam.scale * pixels_per_point as f64) as f32,
        proj: cam.proj.shader_id(),
        rect_px: [r.min.x, r.min.y, r.width(), r.height()],
    }
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// An `nx` x `ny` `R32Float` texture (unfilterable, read with `textureLoad`).
/// Callers bound both sides by the texture limit, so the casts are lossless.
pub(crate) fn create_data_texture(device: &wgpu::Device, ny: usize, nx: usize) -> wgpu::Texture {
    texture(device, "torcdf data", nx as u32, ny as u32, wgpu::TextureFormat::R32Float)
}

/// Writes row-major `data` of shape `[ny, nx]` into a data texture.
pub(crate) fn write_data(queue: &wgpu::Queue, tex: &wgpu::Texture, data: &[f32], ny: usize, nx: usize) {
    write_texture(queue, tex, bytemuck::cast_slice(data), nx, ny, 4);
}

/// A 256 x 1 `Rgba8Unorm` colour LUT.
pub(crate) fn lut_texture(device: &wgpu::Device) -> wgpu::Texture {
    texture(device, "torcdf lut", 256, 1, wgpu::TextureFormat::Rgba8Unorm)
}

/// Writes a colour map's 256 entries into a LUT texture.
pub(crate) fn write_lut(queue: &wgpu::Queue, tex: &wgpu::Texture, lut: &[[u8; 4]; 256]) {
    write_texture(queue, tex, lut.as_flattened(), 256, 1, 4);
}

fn write_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: usize,
    height: usize,
    texel_bytes: usize,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((width * texel_bytes) as u32),
            rows_per_image: None,
        },
        wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
    );
}

/// Headless rendering on the software fallback adapter for GPU tests.
#[cfg(test)]
pub(crate) mod headless {
    use eframe::egui;
    use eframe::egui_wgpu::{self, Renderer, RendererOptions, ScreenDescriptor, wgpu};
    use std::sync::Arc;

    /// egui prefers a gamma-space (non-sRGB) target; this is what the app gets.
    pub(crate) const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// A render state on the software fallback adapter (WARP on Windows), with
    /// downlevel limits so the test sees the same constraints as weak machines.
    pub(crate) fn render_state() -> egui_wgpu::RenderState {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: true,
            ..Default::default()
        }))
        .expect("no fallback (software) wgpu adapter on this machine");
        eprintln!("adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("torcdf-test"),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .expect("device on fallback adapter");
        let renderer = Renderer::new(&device, FORMAT, RendererOptions::PREDICTABLE);
        egui_wgpu::RenderState {
            adapter,
            available_adapters: Vec::new(),
            instance,
            device,
            queue,
            target_format: FORMAT,
            renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
        }
    }

    /// Runs one egui frame over a whole `w` x `h` offscreen target, calling
    /// `draw` with the ui and the screen rect, renders it through egui_wgpu's
    /// renderer and reads the pixels back row-major.
    pub(crate) fn render(
        rs: &egui_wgpu::RenderState,
        w: u32,
        h: u32,
        mut draw: impl FnMut(&mut egui::Ui, egui::Rect),
    ) -> Vec<[u8; 4]> {
        let (device, queue) = (&rs.device, &rs.queue);
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
        let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| draw(ui, screen));
        let textures = std::mem::take(&mut out.textures_delta.set);
        out.textures_delta.clear(); // nothing to free in a single-frame context
        let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: out.pixels_per_point };

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut renderer = rs.renderer.write();
        for (id, deltas) in &textures {
            for delta in deltas {
                renderer.update_texture(device, queue, *id, delta);
            }
        }
        let mut cmds = renderer.update_buffers(device, queue, &mut encoder, &prims, &sd);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            renderer.render(&mut pass.forget_lifetime(), &prims, &sd);
        }

        let row_bytes = 4 * w.div_ceil(64) * 64; // rows padded to 256 bytes
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row_bytes * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        cmds.push(encoder.finish());
        queue.submit(cmds);
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map readback"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let bytes = readback.get_mapped_range(..).expect("mapped range");
        (0..h as usize)
            .flat_map(|y| (0..w as usize).map(move |x| (y, x)))
            .map(|(y, x)| {
                let o = y * row_bytes as usize + 4 * x;
                [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::projection::Projection;

    #[test]
    fn camera_uniform_converts_points_to_px() {
        let cam = MapCamera { proj: Projection::Mercator, centre: [0.5, -0.25], scale: 100.0 };
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(300.0, 200.0));
        let one = camera_uniform(&cam, rect, 1.0);
        assert_eq!(
            one,
            CameraUniform {
                centre: [0.5, -0.25],
                scale_px: 100.0,
                proj: 1,
                rect_px: [10.0, 20.0, 300.0, 200.0]
            }
        );
        let two = camera_uniform(&cam, rect, 2.0);
        assert_eq!(two.centre, one.centre, "centre is in world units");
        assert_eq!(two.proj, 1);
        assert_eq!(two.scale_px, 200.0);
        assert_eq!(two.rect_px, [20.0, 40.0, 600.0, 400.0]);
    }

    /// Entry point appended to `projection.wgsl`: each case is
    /// (proj, 0 forward / 1 inverse, argument) and yields the vec3 result.
    const PROJECTION_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> cases: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4<f32>>;

@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = cases[id.x];
    let proj = u32(c.x);
    var r: vec3<f32>;
    if c.y == 0.0 {
        r = forward(proj, c.zw);
    } else {
        r = inverse(proj, c.zw);
    }
    results[id.x] = vec4<f32>(r, 0.0);
}
"#;

    /// Runs `projection.wgsl` on the fallback adapter for each case.
    fn run_projection_probe(cases: &[[f32; 4]]) -> Vec<[f32; 4]> {
        use wgpu::util::DeviceExt;
        let rs = headless::render_state();
        let device = &rs.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("projection probe"),
            source: wgpu::ShaderSource::Wgsl(
                [include_str!("projection.wgsl"), PROJECTION_PROBE].concat().into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("projection probe"),
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bytes = std::mem::size_of_val(cases) as u64;
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cases"),
            contents: bytemuck::cast_slice(cases),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("projection probe"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        rs.queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map readback"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mapped = readback.get_mapped_range(..).expect("mapped range");
        mapped
            .chunks_exact(16)
            .map(|c| std::array::from_fn(|i| bytemuck::pod_read_unaligned(&c[4 * i..4 * i + 4])))
            .collect()
    }

    /// f32 shader results agree with the f64 CPU projection to within these.
    const FORWARD_TOL: f64 = 1e-5;
    const INVERSE_TOL_DEG: f64 = 1e-4;

    #[test]
    fn projection_wgsl_matches_rust() {
        let mut cases = Vec::new();
        // Forward: a lon/lat grid (including out-of-range and other-hemisphere
        // points), lon wrap past 180, and non-finite input.
        for &p in Projection::all() {
            let id = p.shader_id() as f32;
            for lon in [-180.0, -135.0, -20.0, 0.0, 45.0, 179.0, 180.0, 190.0, 400.0] {
                for lat in [-95.0, -89.0, -86.0, -60.0, -10.0, 0.0, 10.0, 60.0, 86.0, 89.0, 95.0] {
                    cases.push([id, 0.0, lon, lat]);
                }
            }
            cases.push([id, 0.0, f32::NAN, 10.0]);
            cases.push([id, 0.0, 10.0, f32::INFINITY]);
            // Inverse: points inside and outside each world.
            for x in [-4.0, -2.5, -1.0, -0.3, 0.0, 0.7, 1.5, 3.0, 3.5] {
                for y in [-3.0, -1.7, -1.0, -0.2, 0.0, 0.4, 1.2, 1.6, 2.5] {
                    cases.push([id, 1.0, x, y]);
                }
            }
            cases.push([id, 1.0, f32::NAN, 0.5]);
            cases.push([id, 1.0, 0.5, f32::NEG_INFINITY]);
        }
        let results = run_projection_probe(&cases);
        let mut bad = Vec::new();
        let mut worst = [0f64; 2];
        for (c, r) in cases.iter().zip(&results) {
            let p = Projection::all()[c[0] as usize];
            let (a, b) = (c[2] as f64, c[3] as f64);
            let inverse = c[1] != 0.0;
            let want = if inverse {
                p.inverse(a, b).map(|(lon, lat)| [lon, lat])
            } else {
                p.forward(a, b)
            };
            let ok = match want {
                None => r[2] == 0.0,
                Some(w) => {
                    let mut err = [(r[0] as f64 - w[0]).abs(), (r[1] as f64 - w[1]).abs()];
                    // Longitude is undefined at the poles.
                    if inverse && a == 0.0 && b == 0.0 && !p.wraps_x() {
                        err[0] = 0.0;
                    }
                    let e = err[0].max(err[1]);
                    worst[inverse as usize] = worst[inverse as usize].max(e);
                    r[2] == 1.0 && e < if inverse { INVERSE_TOL_DEG } else { FORWARD_TOL }
                }
            };
            if !ok {
                bad.push(format!("{p:?} case {c:?}: GPU {r:?}, CPU {want:?}"));
            }
        }
        eprintln!("worst error: forward {:e} world units, inverse {:e} deg", worst[0], worst[1]);
        assert!(bad.is_empty(), "{} of {} cases differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
    }
}
