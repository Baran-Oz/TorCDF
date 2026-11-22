//! GPU map layer for curvilinear grids: the data tile is a texture and a
//! coarse projected mesh (see `map::mesh`) interpolates the tile texel across
//! the screen. Panning and zooming rewrite a uniform only; a projection change
//! calls [`set_curvilinear_mesh`].

use std::collections::HashMap;
use std::num::NonZeroU64;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

use super::colormap::Colormap;
use super::gpu_common::{self};
use super::image::{check_shape, normalisation, UploadError};
use super::map_regular::{MapParams, MapTile};
use crate::map::mesh::MeshVertex;

/// Checks that `indices` forms whole triangles over `verts`.
fn check_mesh(verts: &[MeshVertex], indices: &[u32]) -> Result<(), UploadError> {
    if indices.len() % 3 != 0 {
        return Err(UploadError::BadCoords("mesh indices are not whole triangles"));
    }
    if indices.iter().any(|&i| i as usize >= verts.len()) {
        return Err(UploadError::BadCoords("mesh index out of range"));
    }
    Ok(())
}

/// Uploads row-major `data` of shape `[ny, nx]` and its projected mesh as
/// tile `id`. Re-uploading an id reuses its texture and buffers when large
/// enough. Nothing is touched on error.
pub fn upload_curvilinear(
    render_state: &egui_wgpu::RenderState,
    id: u64,
    data: &[f32],
    ny: usize,
    nx: usize,
    verts: &[MeshVertex],
    indices: &[u32],
) -> Result<MapTile, UploadError> {
    let (device, queue) = (&render_state.device, &render_state.queue);
    let max = device.limits().max_texture_dimension_2d.min(2048) as usize;
    check_shape(data.len(), ny, nx, max)?;
    check_mesh(verts, indices)?;

    let mut renderer = render_state.renderer.write();
    let res = renderer
        .callback_resources
        .entry::<CurvResources>()
        .or_insert_with(|| CurvResources::new(device, render_state.target_format));
    let reusable = res.tiles.get(&id).is_some_and(|t| (t.ny, t.nx) == (ny, nx));
    if !reusable {
        let tile = TileGpu::new(device, &res.bind_group_layout, ny, nx);
        res.tiles.insert(id, tile);
    }
    if let Some(tile) = res.tiles.get_mut(&id) {
        gpu_common::write_data(queue, &tile.data, data, ny, nx);
        tile.write_mesh(device, queue, verts, indices);
    }
    Ok(MapTile { id, ny, nx })
}

/// Replaces the mesh of tile `id` (after a projection change). An unknown id
/// or an invalid mesh is ignored.
pub fn set_curvilinear_mesh(
    render_state: &egui_wgpu::RenderState,
    id: u64,
    verts: &[MeshVertex],
    indices: &[u32],
) {
    if check_mesh(verts, indices).is_err() {
        return;
    }
    let mut renderer = render_state.renderer.write();
    if let Some(tile) = renderer
        .callback_resources
        .get_mut::<CurvResources>()
        .and_then(|res| res.tiles.get_mut(&id))
    {
        tile.write_mesh(&render_state.device, &render_state.queue, verts, indices);
    }
}

/// Draws `tile` into `rect` (points) of `ui`.
/// Paint each tile id at most once per frame: its uniforms are per tile.
pub fn paint_curvilinear(ui: &mut egui::Ui, rect: egui::Rect, tile: &MapTile, p: &MapParams) {
    let callback = CurvCallback { id: tile.id, params: *p };
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, callback));
}

/// The `Params` uniform of `map_curv.wgsl`: `Camera` (32 bytes) at 0, `f32`
/// offset at 32, `f32` scale at 36, `u32` log at 40, `vec2` screen size at 48;
/// 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct CurvUniform {
    camera: gpu_common::CameraUniform,
    offset: f32,
    scale: f32,
    log: u32,
    _pad: u32,
    screen_px: [f32; 2],
    _pad2: [u32; 2],
}

const UNIFORM_BYTES: u64 = std::mem::size_of::<CurvUniform>() as u64;
const _: () = assert!(UNIFORM_BYTES == 64);

/// Everything this module keeps in `CallbackResources`.
struct CurvResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tiles: HashMap<u64, TileGpu>,
}

impl CurvResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("torcdf map curvilinear"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("colour.wgsl"),
                    include_str!("projection.wgsl"),
                    include_str!("map_curv.wgsl")
                )
                .into(),
            ),
        });
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("torcdf map curvilinear"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(UNIFORM_BYTES),
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("torcdf map curvilinear"),
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
            label: Some("torcdf map curvilinear"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
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
        });
        Self { pipeline, bind_group_layout, tiles: HashMap::new() }
    }
}

/// GPU objects of one tile.
struct TileGpu {
    ny: usize,
    nx: usize,
    data: wgpu::Texture,
    lut: wgpu::Texture,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// Colour map currently in `lut`; `None` until first painted.
    colormap: Option<Colormap>,
}

fn mesh_buffer(device: &wgpu::Device, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        // Never zero-sized, so an empty mesh still has valid buffers.
        size: bytes.max(16),
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl TileGpu {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, ny: usize, nx: usize) -> Self {
        // check_shape bounds both sides by the texture limit.
        let data = gpu_common::create_data_texture(device, ny, nx);
        let lut = gpu_common::lut_texture(device);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torcdf map curvilinear params"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let views: Vec<wgpu::TextureView> =
            [&data, &lut].iter().map(|t| t.create_view(&Default::default())).collect();
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }];
        entries.extend(views.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: wgpu::BindingResource::TextureView(v),
        }));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("torcdf map curvilinear"),
            layout,
            entries: &entries,
        });
        Self {
            ny,
            nx,
            data,
            lut,
            uniforms,
            bind_group,
            vertices: mesh_buffer(device, "torcdf curv vertices", 0, wgpu::BufferUsages::VERTEX),
            indices: mesh_buffer(device, "torcdf curv indices", 0, wgpu::BufferUsages::INDEX),
            index_count: 0,
            colormap: None,
        }
    }

    /// Writes an already validated mesh, growing the buffers only when needed.
    fn write_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[MeshVertex], indices: &[u32]) {
        let vb: &[u8] = bytemuck::cast_slice(verts);
        let ib: &[u8] = bytemuck::cast_slice(indices);
        if self.vertices.size() < vb.len() as u64 {
            self.vertices = mesh_buffer(device, "torcdf curv vertices", vb.len() as u64, wgpu::BufferUsages::VERTEX);
        }
        if self.indices.size() < ib.len() as u64 {
            self.indices = mesh_buffer(device, "torcdf curv indices", ib.len() as u64, wgpu::BufferUsages::INDEX);
        }
        if !vb.is_empty() {
            queue.write_buffer(&self.vertices, 0, vb);
        }
        if !ib.is_empty() {
            queue.write_buffer(&self.indices, 0, ib);
        }
        self.index_count = indices.len() as u32;
    }
}

/// One frame's draw of a tile.
struct CurvCallback {
    id: u64,
    params: MapParams,
}

impl egui_wgpu::CallbackTrait for CurvCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let tile = resources.get_mut::<CurvResources>().and_then(|res| res.tiles.get_mut(&self.id));
        if let Some(tile) = tile {
            let p = &self.params;
            let (offset, scale) = normalisation(p.range, p.log);
            let u = CurvUniform {
                camera: p.camera,
                offset,
                scale,
                log: u32::from(p.log),
                _pad: 0,
                screen_px: [screen.size_in_pixels[0] as f32, screen.size_in_pixels[1] as f32],
                _pad2: [0; 2],
            };
            queue.write_buffer(&tile.uniforms, 0, bytemuck::bytes_of(&u));
            if tile.colormap != Some(p.colormap) {
                gpu_common::write_lut(queue, &tile.lut, &p.colormap.lut());
                tile.colormap = Some(p.colormap);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<CurvResources>() else { return };
        let Some(tile) = res.tiles.get(&self.id) else { return };
        if tile.index_count == 0 {
            return;
        }
        // Full-framebuffer viewport (egui clamps the callback's own to the
        // screen); egui's scissor still clips to the rect. See lines.rs.
        let [w, h] = info.screen_size_px;
        render_pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &tile.bind_group, &[]);
        render_pass.set_vertex_buffer(0, tile.vertices.slice(..));
        render_pass.set_index_buffer(tile.indices.slice(..), wgpu::IndexFormat::Uint32);
        // Cylindrical projections (ids 0, 1) draw three wrap copies.
        let instances = if self.params.camera.proj <= 1 { 3 } else { 1 };
        render_pass.draw_indexed(0..tile.index_count, 0, 0..instances);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::camera::MapCamera;
    use crate::map::mesh::{build_mesh, project_mesh};
    use crate::map::projection::Projection;
    use crate::map::tiles::TileSpec;
    use crate::render::gpu_common::{camera_uniform, headless};

    const W: u32 = 64;
    const H: u32 = 32;

    #[test]
    fn curvilinear_renders_headless() {
        let rs = headless::render_state();
        // 3 x 3 grid at 10 degree spacing, rotated 45 degrees about (0, 0).
        let (s, c) = (std::f64::consts::FRAC_PI_4.sin(), std::f64::consts::FRAC_PI_4.cos());
        let mut lon = Vec::new();
        let mut lat = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                let (x, y) = ((i as f64 - 1.0) * 10.0, (j as f64 - 1.0) * 10.0);
                lon.push(x * c - y * s);
                lat.push(x * s + y * c);
            }
        }
        let tile = TileSpec::overview(3, 3);
        let mesh = build_mesh(&lat, &lon, [3, 3], &tile, 257);
        let proj = Projection::Equirectangular;
        let (verts, idx) = project_mesh(proj, &mesh);
        let data: Vec<f32> = (0..9).map(|k| k as f32 + 0.5).collect();
        let t = upload_curvilinear(&rs, 1, &data, 3, 3, &verts, &idx).expect("upload");

        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(W as f32, H as f32));
        let mut cam = MapCamera::new(proj);
        let pts: Vec<(f64, f64)> = mesh.lonlat.iter().map(|p| (p[0], p[1])).collect();
        cam.fit(&pts, rect);
        let px = headless::render(&rs, W, H, |ui, rect| {
            let p = MapParams {
                range: (0.0, 9.0),
                colormap: Colormap::Viridis,
                log: false,
                camera: camera_uniform(&cam, rect, 1.0),
            };
            paint_curvilinear(ui, rect, &t, &p);
        });
        let want = Colormap::Viridis.lut()[(4.5 / 9.0 * 256.0) as usize];
        let got = px[(16 * W + 32) as usize];
        assert!((0..4).all(|k| (got[k] as i32 - want[k] as i32).abs() <= 1), "{got:?} vs {want:?}");
        assert_eq!(px[0][3], 0, "outside the grid: {:?}", px[0]);

        // Invalid meshes are refused before touching the GPU.
        assert!(matches!(
            upload_curvilinear(&rs, 2, &data, 3, 3, &verts, &[0, 1]),
            Err(UploadError::BadCoords(_))
        ));
        assert!(matches!(
            upload_curvilinear(&rs, 2, &data, 3, 3, &verts, &[0, 1, 9999]),
            Err(UploadError::BadCoords(_))
        ));
        // A new mesh (projection change) can be swapped in.
        set_curvilinear_mesh(&rs, 1, &verts, &[]);
        let px = headless::render(&rs, W, H, |ui, rect| {
            let p = MapParams {
                range: (0.0, 9.0),
                colormap: Colormap::Viridis,
                log: false,
                camera: camera_uniform(&cam, rect, 1.0),
            };
            paint_curvilinear(ui, rect, &t, &p);
        });
        assert_eq!(px[(16 * W + 32) as usize][3], 0, "empty mesh draws nothing");
    }
}
