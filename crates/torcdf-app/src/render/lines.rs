//! GPU line overlays (coastlines, borders, graticule) drawn through the map
//! camera inside an egui paint callback.
//!
//! Polylines are projected to world units on the CPU once per projection
//! ([`project_polylines`]) and uploaded as a `LineList` vertex buffer
//! ([`upload_lines`]). Panning and zooming only rewrite a 64-byte uniform.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};
use std::num::NonZeroU64;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

use super::gpu_common::CameraUniform;
use crate::map::projection::Projection;
use torcdf_core::basemap::Polyline;

/// LineList endpoints in world units: one segment per consecutive valid pair
/// of points. A segment is dropped when either end does not project. For the
/// cylindrical projections a segment crossing the seam (|dx| > pi) is kept:
/// its second endpoint is shifted by 2*pi so it stays short and continuous,
/// and the part beyond +-pi is covered by the renderer's wrap copies.
pub fn project_polylines(proj: Projection, lines: &[Polyline]) -> Vec<[f32; 2]> {
    let mut out = Vec::new();
    for line in lines {
        let mut prev: Option<[f64; 2]> = None;
        for p in line {
            let cur = proj.forward(p[0] as f64, p[1] as f64);
            if let (Some(a), Some(mut b)) = (prev, cur) {
                if proj.wraps_x() {
                    if b[0] - a[0] > PI {
                        b[0] -= TAU;
                    } else if b[0] - a[0] < -PI {
                        b[0] += TAU;
                    }
                }
                out.push([a[0] as f32, a[1] as f32]);
                out.push([b[0] as f32, b[1] as f32]);
            }
            prev = cur;
        }
    }
    out
}

/// Uploads LineList endpoints as line set `id`, replacing any previous set.
/// The vertex buffer is reused when it is large enough. An odd trailing
/// endpoint is ignored.
pub fn upload_lines(render_state: &egui_wgpu::RenderState, id: u64, segments: &[[f32; 2]]) {
    let device = &render_state.device;
    let segments = &segments[..segments.len() / 2 * 2];
    let bytes: &[u8] = bytemuck::cast_slice(segments);
    let mut renderer = render_state.renderer.write();
    let res = renderer
        .callback_resources
        .entry::<LineResources>()
        .or_insert_with(|| LineResources::new(device, render_state.target_format));
    let reusable = res.sets.get(&id).is_some_and(|s| s.vertices.size() >= bytes.len() as u64);
    if !reusable {
        let set = LineSet::new(device, &res.bind_group_layout, bytes.len() as u64);
        res.sets.insert(id, set);
    }
    if let Some(set) = res.sets.get_mut(&id) {
        if !bytes.is_empty() {
            render_state.queue.write_buffer(&set.vertices, 0, bytes);
        }
        set.count = segments.len() as u32;
    }
}

/// How to draw a line set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineParams {
    pub camera: CameraUniform,
    pub colour: egui::Color32,
    /// Also draw the set shifted by -2pi and +2pi in x (cylindrical worlds).
    pub wrap: bool,
}

/// Draws line set `id` into `rect` (points) of `ui`.
/// Paint one line set at most once per frame: its uniforms are per set.
pub fn paint_lines(ui: &mut egui::Ui, rect: egui::Rect, id: u64, p: &LineParams) {
    let callback = LineCallback { id, params: *p };
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, callback));
}

/// The `Lines` uniform of `lines.wgsl`: `Camera` (32 bytes) at 0, `vec4` colour
/// at 32, `vec2` screen size at 48, `u32` wrap at 56; 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct LineUniform {
    camera: CameraUniform,
    colour: [f32; 4],
    screen_px: [f32; 2],
    wrap: u32,
    _pad: u32,
}

const UNIFORM_BYTES: u64 = std::mem::size_of::<LineUniform>() as u64;
const _: () = assert!(UNIFORM_BYTES == 64);

/// Everything this module keeps in `CallbackResources`.
struct LineResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sets: HashMap<u64, LineSet>,
}

impl LineResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("torcdf lines"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("colour.wgsl"),
                    include_str!("projection.wgsl"),
                    include_str!("lines.wgsl")
                )
                .into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("torcdf lines"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(UNIFORM_BYTES),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("torcdf lines"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        // Mirror egui: gamma-space colour for Unorm targets, linear for sRGB.
        let fragment_entry = if target_format.is_srgb() {
            "fs_main_linear_framebuffer"
        } else {
            "fs_main_gamma_framebuffer"
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("torcdf lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            // egui's pass (eframe defaults): no depth buffer, no MSAA.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, bind_group_layout, sets: HashMap::new() }
    }
}

/// GPU objects of one line set.
struct LineSet {
    vertices: wgpu::Buffer,
    /// Endpoints to draw (even).
    count: u32,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl LineSet {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bytes: u64) -> Self {
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torcdf lines vertices"),
            // Never zero-sized, so an empty set still has a valid buffer.
            size: bytes.max(8),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torcdf lines params"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("torcdf lines"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        Self { vertices, count: 0, uniforms, bind_group }
    }
}

/// One frame's draw of a line set.
struct LineCallback {
    id: u64,
    params: LineParams,
}

impl egui_wgpu::CallbackTrait for LineCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let set = resources.get::<LineResources>().and_then(|res| res.sets.get(&self.id));
        if let Some(set) = set {
            let p = &self.params;
            let u = LineUniform {
                camera: p.camera,
                colour: p.colour.to_normalized_gamma_f32(),
                screen_px: [screen.size_in_pixels[0] as f32, screen.size_in_pixels[1] as f32],
                wrap: u32::from(p.wrap),
                _pad: 0,
            };
            queue.write_buffer(&set.uniforms, 0, bytemuck::bytes_of(&u));
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<LineResources>() else { return };
        let Some(set) = res.sets.get(&self.id) else { return };
        if set.count == 0 {
            return;
        }
        // egui's viewport is the callback rect clamped to the screen. Use the
        // whole framebuffer instead so the shader maps framebuffer pixels to
        // clip space directly; egui's scissor still clips to the rect, and
        // egui restores its own viewport after the callback.
        let [w, h] = info.screen_size_px;
        render_pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &set.bind_group, &[]);
        render_pass.set_vertex_buffer(0, set.vertices.slice(..));
        let instances = if self.params.wrap { 3 } else { 1 };
        render_pass.draw(0..set.count, 0..instances);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::camera::MapCamera;
    use crate::render::gpu_common::{camera_uniform, headless};
    use std::f64::consts::{PI, TAU};

    #[test]
    fn project_polylines_handles_seam_and_bad_points() {
        let eq = Projection::Equirectangular;
        // Seam crossings are unwrapped so the segment stays short and continuous.
        let east = project_polylines(eq, &[vec![[179.0, 0.0], [-179.0, 0.0]]]);
        assert_eq!(east.len(), 2, "seam segment kept");
        assert!((east[0][0] - 179f32.to_radians()).abs() < 1e-5, "{east:?}");
        assert!((east[1][0] - 181f32.to_radians()).abs() < 1e-5, "{east:?}");
        let west = project_polylines(eq, &[vec![[-179.0, 0.0], [179.0, 0.0]]]);
        assert_eq!(west.len(), 2, "mirror seam segment kept");
        assert!((west[0][0] - (-179f32).to_radians()).abs() < 1e-5, "{west:?}");
        assert!((west[1][0] - (-181f32).to_radians()).abs() < 1e-5, "{west:?}");
        let one = project_polylines(eq, &[vec![[0.0, 0.0], [10.0, 0.0]]]);
        assert_eq!(one.len(), 2, "one segment = two endpoints");
        assert!((one[1][0] - 10f32.to_radians()).abs() < 1e-6);
        let np = Projection::NorthPolarStereo;
        assert_eq!(project_polylines(np, &[vec![[0.0, 10.0], [0.0, -10.0]]]).len(), 0);
        let merc = Projection::Mercator;
        assert_eq!(project_polylines(merc, &[vec![[0.0, 80.0], [0.0, 89.0]]]).len(), 0);
        // A bad point only removes the segments that touch it.
        let three = vec![[0.0, 0.0], [1.0, f32::NAN], [2.0, 0.0], [3.0, 0.0]];
        assert_eq!(project_polylines(eq, &[three]).len(), 2);
    }

    fn lit(px: [u8; 4]) -> bool {
        px[3] > 0
    }

    #[test]
    fn lines_render_headless() {
        const W: u32 = 64;
        const H: u32 = 32;
        let rs = headless::render_state();
        let white = egui::Color32::WHITE;
        let scale = W as f64 / TAU; // 2*pi spans the target (pixels_per_point 1)
        let draw = |id: u64, centre_x: f64, wrap: bool| {
            let cam = MapCamera { proj: Projection::Equirectangular, centre: [centre_x, 0.0], scale };
            headless::render(&rs, W, H, |ui, rect| {
                let p = LineParams { camera: camera_uniform(&cam, rect, 1.0), colour: white, wrap };
                paint_lines(ui, rect, id, &p);
            })
        };
        let at = |px: &[[u8; 4]], x: u32, y: u32| px[(y * W + x) as usize];

        let eq = Projection::Equirectangular;
        upload_lines(&rs, 1, &project_polylines(eq, &[vec![[-90.0, 0.0], [90.0, 0.0]]]));
        let px = draw(1, 0.0, false);
        assert_eq!(at(&px, 32, 16)[3], 255, "on the line: {:?}", at(&px, 32, 16));
        assert_eq!(at(&px, 32, 4)[3], 0, "off the line: {:?}", at(&px, 32, 4));

        // Just east of the dateline, lon 170..179 lies one world to the left.
        upload_lines(&rs, 2, &project_polylines(eq, &[vec![[170.0, 0.0], [179.0, 0.0]]]));
        let row = |px: &[[u8; 4]]| (0..W).filter(|&x| lit(at(px, x, 16))).collect::<Vec<_>>();
        let wrapped = draw(2, -PI + 0.01, true);
        // Copy spans x = -3.316..-3.159 rad, i.e. about columns 30..=31 (centre -pi+0.01).
        let cols = row(&wrapped);
        assert!(cols.contains(&30), "wrap draws the copy: {cols:?}");
        assert!(cols.iter().all(|&x| (29..=32).contains(&x)), "copy stays near x=31: {cols:?}");
        let unwrapped = draw(2, -PI + 0.01, false);
        assert!(row(&unwrapped).is_empty(), "no wrap, nothing in view: {:?}", row(&unwrapped));

        // A line crossing the dateline (170 -> -170) has no gap at the target centre.
        upload_lines(&rs, 3, &project_polylines(eq, &[vec![[170.0, 0.0], [-170.0, 0.0]]]));
        let seam = draw(3, PI, true);
        let cols = row(&seam);
        assert!((31..=32).all(|x| cols.contains(&x)), "no gap at the dateline: {cols:?}");
    }
}
