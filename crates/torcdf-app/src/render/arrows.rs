//! GPU vector arrows: instanced glyphs drawn inside an egui paint callback,
//! on the map (through the camera) and on the image view.
//!
//! Instances ([`ArrowInstance`]) hold world positions and directions; an
//! affine map from world to framebuffer pixels places them, so panning and
//! zooming only rewrite a 64-byte uniform. The glyph has a constant pixel
//! size per unit of magnitude.

use std::collections::HashMap;
use std::num::NonZeroU64;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

use super::gpu_common::CameraUniform;
use crate::map::arrows::ArrowInstance;

pub const ARROW_MAP_ID: u64 = 30;
pub const ARROW_IMAGE_ID: u64 = 31;

/// Vertices per glyph: a shaft quad (two triangles) and a head triangle.
const GLYPH_VERTS: u32 = 9;

/// How to draw an arrow set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrowParams {
    /// Framebuffer px = `[a b; c d]·world + [e f]`, as `(a, b, c, d, e, f)`.
    pub affine: [f32; 6],
    /// Arrow length in framebuffer px per unit of magnitude.
    pub px_per_unit: f32,
    /// Also draw at world x ± 2π.
    pub wrap: bool,
    /// The plot rect in framebuffer px (x, y, w, h); arrows far outside are culled.
    pub rect_px: [f32; 4],
}

/// The world-to-framebuffer affine of the map camera.
pub fn map_affine(cam: &CameraUniform) -> [f32; 6] {
    let s = cam.scale_px;
    let [x, y, w, h] = cam.rect_px;
    let (cx, cy) = (x + 0.5 * w, y + 0.5 * h);
    [s, 0.0, 0.0, -s, cx - cam.centre[0] * s, cy + cam.centre[1] * s]
}

/// The affine of the image view showing `view` (cell units: x the column,
/// y the row index) in `rect` (points); mirrors `image.wgsl`.
pub fn image_affine(view: egui::Rect, rect: egui::Rect, flip_y: bool, ppp: f32) -> [f32; 6] {
    let sx = ppp * rect.width() / view.width();
    let sy = ppp * rect.height() / view.height();
    let e = ppp * rect.min.x - view.min.x * sx;
    if flip_y {
        [sx, 0.0, 0.0, -sy, e, ppp * rect.min.y + view.max.y * sy]
    } else {
        [sx, 0.0, 0.0, sy, e, ppp * rect.min.y - view.min.y * sy]
    }
}

/// Uploads `instances` as arrow set `id`, replacing any previous set. The
/// buffer is reused when large enough.
pub fn upload_arrows(rs: &egui_wgpu::RenderState, id: u64, instances: &[ArrowInstance]) {
    let device = &rs.device;
    let bytes: &[u8] = bytemuck::cast_slice(instances);
    let mut renderer = rs.renderer.write();
    let res = renderer
        .callback_resources
        .entry::<ArrowResources>()
        .or_insert_with(|| ArrowResources::new(device, rs.target_format));
    let reusable = res.sets.get(&id).is_some_and(|s| s.instances.size() >= bytes.len() as u64);
    if !reusable {
        let set = ArrowGpu::new(device, &res.bind_group_layout, bytes.len() as u64);
        res.sets.insert(id, set);
    }
    if let Some(set) = res.sets.get_mut(&id) {
        if !bytes.is_empty() {
            rs.queue.write_buffer(&set.instances, 0, bytes);
        }
        set.count = instances.len() as u32;
    }
}

/// Draws arrow set `id` into `rect` (points). Paint one set at most once per
/// frame: its uniforms are per set.
pub fn paint_arrows(ui: &mut egui::Ui, rect: egui::Rect, id: u64, p: &ArrowParams) {
    let callback = ArrowCallback { id, params: *p };
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, callback));
}

/// The `Arrows` uniform of `arrows.wgsl`: `vec4` at 0, `vec2` at 16 and 24,
/// `vec4` at 32, `f32` at 48, `u32` at 52; 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ArrowUniform {
    lin: [f32; 4],
    off: [f32; 2],
    screen_px: [f32; 2],
    rect_px: [f32; 4],
    px_per_unit: f32,
    wrap: u32,
    _pad: [u32; 2],
}

const UNIFORM_BYTES: u64 = std::mem::size_of::<ArrowUniform>() as u64;
const _: () = assert!(UNIFORM_BYTES == 64);
const INSTANCE_BYTES: u64 = std::mem::size_of::<ArrowInstance>() as u64;
const _: () = assert!(INSTANCE_BYTES == 24);

/// Everything this module keeps in `CallbackResources`.
struct ArrowResources {
    /// The light outline, drawn first.
    outline: wgpu::RenderPipeline,
    /// The dark arrow on top.
    fill: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sets: HashMap<u64, ArrowGpu>,
}

impl ArrowResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("torcdf arrows"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(include_str!("colour.wgsl"), include_str!("arrows.wgsl")).into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("torcdf arrows"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(UNIFORM_BYTES),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("torcdf arrows"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        // Mirror egui: gamma-space colour for Unorm targets, linear for sRGB.
        let fragment_entry = if target_format.is_srgb() {
            "fs_main_linear_framebuffer"
        } else {
            "fs_main_gamma_framebuffer"
        };
        let pipeline = |vertex_entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("torcdf arrows"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vertex_entry),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: INSTANCE_BYTES,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32],
                    })],
                },
                primitive: wgpu::PrimitiveState::default(),
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
            })
        };
        Self {
            outline: pipeline("vs_outline"),
            fill: pipeline("vs_fill"),
            bind_group_layout,
            sets: HashMap::new(),
        }
    }
}

/// GPU objects of one arrow set.
struct ArrowGpu {
    instances: wgpu::Buffer,
    count: u32,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ArrowGpu {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bytes: u64) -> Self {
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torcdf arrow instances"),
            // Never zero-sized, so an empty set still has a valid buffer.
            size: bytes.max(INSTANCE_BYTES),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torcdf arrow params"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("torcdf arrows"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        Self { instances, count: 0, uniforms, bind_group }
    }
}

/// One frame's draw of an arrow set.
struct ArrowCallback {
    id: u64,
    params: ArrowParams,
}

impl egui_wgpu::CallbackTrait for ArrowCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let set = resources.get::<ArrowResources>().and_then(|res| res.sets.get(&self.id));
        if let Some(set) = set {
            let p = &self.params;
            let a = p.affine;
            let u = ArrowUniform {
                lin: [a[0], a[1], a[2], a[3]],
                off: [a[4], a[5]],
                screen_px: [screen.size_in_pixels[0] as f32, screen.size_in_pixels[1] as f32],
                rect_px: p.rect_px,
                px_per_unit: p.px_per_unit,
                wrap: u32::from(p.wrap),
                _pad: [0; 2],
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
        let Some(res) = resources.get::<ArrowResources>() else { return };
        let Some(set) = res.sets.get(&self.id) else { return };
        if set.count == 0 {
            return;
        }
        // Full-framebuffer viewport (as in `lines`): the shader maps
        // framebuffer pixels to clip space; egui's scissor clips to the rect.
        let [w, h] = info.screen_size_px;
        render_pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
        let verts = GLYPH_VERTS * if self.params.wrap { 3 } else { 1 };
        render_pass.set_bind_group(0, &set.bind_group, &[]);
        render_pass.set_vertex_buffer(0, set.instances.slice(..));
        for pipeline in [&res.outline, &res.fill] {
            render_pass.set_pipeline(pipeline);
            render_pass.draw(0..verts, 0..set.count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::gpu_common::headless;
    use std::f32::consts::PI;

    const W: u32 = 200;

    /// Camera at world (`centre_x`, 0), 200 framebuffer px per radian.
    fn camera(centre_x: f32) -> CameraUniform {
        CameraUniform { centre: [centre_x, 0.0], scale_px: 200.0, proj: 0, rect_px: [0.0, 0.0, 200.0, 200.0] }
    }

    fn draw(rs: &egui_wgpu::RenderState, id: u64, inst: ArrowInstance, centre_x: f32, wrap: bool) -> Vec<[u8; 4]> {
        upload_arrows(rs, id, &[inst]);
        let p = ArrowParams {
            affine: map_affine(&camera(centre_x)),
            px_per_unit: 40.0,
            wrap,
            rect_px: [0.0, 0.0, 200.0, 200.0],
        };
        headless::render(rs, W, W, |ui, rect| paint_arrows(ui, rect, id, &p))
    }

    fn dark(px: &[[u8; 4]], x: u32, y: u32) -> bool {
        let p = px[(y * W + x) as usize];
        p[3] > 200 && p[0] < 100 && p[1] < 100 && p[2] < 100
    }

    /// Some dark pixel at column `x` within rows `y - 1 ..= y + 1`.
    fn dark_near_row(px: &[[u8; 4]], x: u32, y: u32) -> bool {
        (y - 1..=y + 1).any(|y| dark(px, x, y))
    }

    fn dark_near_col(px: &[[u8; 4]], x: u32, y: u32) -> bool {
        (x - 1..=x + 1).any(|x| dark(px, x, y))
    }

    fn arrow(pos: [f32; 2], dir: [f32; 2], mag: f32) -> ArrowInstance {
        ArrowInstance { pos, dir, mag, _pad: 0.0 }
    }

    fn any_dark(px: &[[u8; 4]]) -> bool {
        (0..W).any(|y| (0..W).any(|x| dark(px, x, y)))
    }

    #[test]
    fn eastward_arrow_points_right() {
        let rs = headless::render_state();
        let px = draw(&rs, 101, arrow([0.0, 0.0], [1.0, 0.0], 1.0), 0.0, false);
        let missing: Vec<u32> = (100..135).filter(|&x| !dark_near_row(&px, x, 100)).collect();
        assert!(missing.is_empty(), "no dark pixel near row 100 at columns {missing:?}");
        let behind: Vec<u32> = (60..95).filter(|&x| dark_near_row(&px, x, 100)).collect();
        assert!(behind.is_empty(), "dark pixels behind the tail at {behind:?}");
    }

    #[test]
    fn northward_arrow_points_up() {
        let rs = headless::render_state();
        let px = draw(&rs, 102, arrow([0.0, 0.0], [0.0, 1.0], 1.0), 0.0, false);
        let missing: Vec<u32> = (65..100).filter(|&y| !dark_near_col(&px, 100, y)).collect();
        assert!(missing.is_empty(), "no dark pixel near column 100 at rows {missing:?}");
        let below: Vec<u32> = (105..140).filter(|&y| dark_near_col(&px, 100, y)).collect();
        assert!(below.is_empty(), "dark pixels below the tail at {below:?}");
    }

    #[test]
    fn dateline_arrow_wraps() {
        let rs = headless::render_state();
        let inst = arrow([-PI + 0.05, 0.0], [1.0, 0.0], 1.0);
        let px = draw(&rs, 103, inst, PI, true);
        let missing: Vec<u32> = (110..145).filter(|&x| !dark_near_row(&px, x, 100)).collect();
        assert!(missing.is_empty(), "wrapped copy missing at columns {missing:?}");
        let px = draw(&rs, 103, inst, PI, false);
        assert!(!any_dark(&px), "without wrap nothing is in view");
    }

    #[test]
    fn short_arrows_not_drawn() {
        let rs = headless::render_state();
        let px = draw(&rs, 104, arrow([0.0, 0.0], [1.0, 0.0], 0.01), 0.0, false);
        assert!(!any_dark(&px), "a 0.4 px arrow is discarded");
    }

    #[test]
    fn image_affine_maps_cells() {
        let view = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(4.0, 2.0));
        let rect = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(200.0, 100.0));
        let apply = |a: [f32; 6], w: [f32; 2]| [a[0] * w[0] + a[1] * w[1] + a[4], a[2] * w[0] + a[3] * w[1] + a[5]];
        // Unflipped, 2 px per point: row 0 at the top; 50 pt per cell.
        let a = image_affine(view, rect, false, 2.0);
        assert_eq!(apply(a, [0.5, 0.5]), [250.0, 150.0]);
        assert_eq!(apply(a, [4.0, 2.0]), [600.0, 300.0]);
        // Flipped: row index grows upwards.
        let a = image_affine(view, rect, true, 1.0);
        assert_eq!(apply(a, [0.0, 0.0]), [100.0, 150.0]);
        assert_eq!(apply(a, [4.0, 2.0]), [300.0, 50.0]);
    }
}
