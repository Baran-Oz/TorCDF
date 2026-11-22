//! GPU colour-mapped image layer: an `R32Float` data texture drawn through a
//! 256-entry colour LUT inside an egui paint callback.
//!
//! GPU state lives in egui_wgpu's `CallbackResources`: one pipeline, plus a
//! data texture, LUT, uniform buffer and bind group per layer id. Changing the
//! colour map or range rewrites only the uniforms and the LUT, never the data.

use std::collections::HashMap;
use std::num::NonZeroU64;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

use super::colormap::Colormap;

/// Largest texture side this plan uses: the wgpu downlevel default.
const MAX_TEXTURE_DIM: u32 = 2048;

/// Size of the `Params` uniform in `image.wgsl`.
const UNIFORM_BYTES: u64 = 48;

/// Handle to a data texture uploaded with [`upload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLayer {
    id: u64,
    ny: usize,
    nx: usize,
}

impl ImageLayer {
    pub fn id(&self) -> u64 {
        self.id
    }

    /// `(ny, nx)` of the uploaded data.
    pub fn shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }
}

/// How to draw an [`ImageLayer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageParams {
    /// Data values mapped to the first and last colour; values outside clamp.
    /// In log mode both ends must be positive, otherwise everything valid
    /// draws in the first colour.
    pub range: (f32, f32),
    pub colormap: Colormap,
    /// Colour by `log10(v)`; `v <= 0` draws transparent.
    pub log: bool,
    /// Data-space rect shown, in cell units: x is the column, y the row index
    /// (`0..nx`, `0..ny` shows the whole image). Cells outside are transparent.
    pub view: egui::Rect,
    /// Draw row index increasing upwards (true when latitude increases with
    /// index, so north is up). False draws row 0 at the top.
    pub flip_y: bool,
}

/// Why [`upload`] refused the data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadError {
    Empty,
    TooLarge { ny: usize, nx: usize, max: usize },
    LengthMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for UploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UploadError::Empty => write!(f, "image has no cells"),
            UploadError::TooLarge { ny, nx, max } => {
                write!(f, "image {ny} x {nx} exceeds the {max} x {max} texture limit")
            }
            UploadError::LengthMismatch { expected, got } => {
                write!(f, "image data has {got} values, expected {expected}")
            }
        }
    }
}

impl std::error::Error for UploadError {}

/// Uploads row-major `data` of shape `[ny, nx]` as layer `id`.
///
/// Re-uploading an existing id replaces its data (reusing the texture when
/// the shape is unchanged), so callers should use a stable id per displayed
/// image rather than a fresh id per slice. Sides above 2048 (or the device
/// limit, if lower) are refused; callers stride down first.
pub fn upload(
    render_state: &egui_wgpu::RenderState,
    id: u64,
    data: &[f32],
    ny: usize,
    nx: usize,
) -> Result<ImageLayer, UploadError> {
    let (device, queue) = (&render_state.device, &render_state.queue);
    let max = device.limits().max_texture_dimension_2d.min(MAX_TEXTURE_DIM) as usize;
    check_shape(data.len(), ny, nx, max)?;

    let mut renderer = render_state.renderer.write();
    let images = renderer
        .callback_resources
        .entry::<ImageResources>()
        .or_insert_with(|| ImageResources::new(device, render_state.target_format));
    let reusable = images.layers.get(&id).is_some_and(|l| (l.ny, l.nx) == (ny, nx));
    if !reusable {
        let layer = LayerGpu::new(device, &images.bind_group_layout, ny, nx);
        images.layers.insert(id, layer);
    }
    if let Some(layer) = images.layers.get(&id) {
        layer.write_data(queue, data);
    }
    Ok(ImageLayer { id, ny, nx })
}

/// Draws `layer` into `rect` (in points) of `ui`.
pub fn paint(ui: &mut egui::Ui, rect: egui::Rect, layer: &ImageLayer, p: &ImageParams) {
    let callback = ImageCallback { id: layer.id, rect, params: *p };
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, callback));
}

fn check_shape(len: usize, ny: usize, nx: usize, max: usize) -> Result<(), UploadError> {
    if ny == 0 || nx == 0 {
        return Err(UploadError::Empty);
    }
    if ny > max || nx > max {
        return Err(UploadError::TooLarge { ny, nx, max });
    }
    if len != ny * nx {
        return Err(UploadError::LengthMismatch { expected: ny * nx, got: len });
    }
    Ok(())
}

/// `(offset, scale)` such that `t = (w - offset) * scale`, where `w` is the
/// value, or its log2 in log mode (normalising log2 equals normalising log10).
/// A degenerate range gives scale 0, mapping every valid value to `t = 0`.
fn normalisation(range: (f32, f32), log: bool) -> (f32, f32) {
    let (lo, hi) = if log { (range.0.log2(), range.1.log2()) } else { range };
    let span = hi - lo;
    if lo.is_finite() && span.is_finite() && span != 0.0 {
        (lo, 1.0 / span)
    } else {
        (0.0, 0.0)
    }
}

/// The `Params` uniform of `image.wgsl`, as raw words.
fn uniform_words(p: &ImageParams, rect: egui::Rect, pixels_per_point: f32) -> [u32; 12] {
    let (offset, scale) = normalisation(p.range, p.log);
    let rect_px = rect * pixels_per_point;
    let f = f32::to_bits;
    [
        f(offset),
        f(scale),
        u32::from(p.log),
        u32::from(p.flip_y),
        f(p.view.min.x),
        f(p.view.min.y),
        f(p.view.max.x),
        f(p.view.max.y),
        f(rect_px.min.x),
        f(rect_px.min.y),
        f(rect_px.width()),
        f(rect_px.height()),
    ]
}

/// Everything this module keeps in `CallbackResources`.
struct ImageResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    layers: HashMap<u64, LayerGpu>,
}

impl ImageResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("torcdf image"),
            source: wgpu::ShaderSource::Wgsl(include_str!("image.wgsl").into()),
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
            label: Some("torcdf image"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("torcdf image"),
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
            label: Some("torcdf image"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
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
        Self { pipeline, bind_group_layout, layers: HashMap::new() }
    }
}

/// GPU objects of one layer.
struct LayerGpu {
    ny: usize,
    nx: usize,
    data: wgpu::Texture,
    lut: wgpu::Texture,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Colour map currently in `lut`; `None` until first painted.
    colormap: Option<Colormap>,
}

impl LayerGpu {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, ny: usize, nx: usize) -> Self {
        let texture = |label, width, height, format| {
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
        };
        // check_shape bounds both sides by MAX_TEXTURE_DIM, so the casts are lossless.
        let data = texture("torcdf image data", nx as u32, ny as u32, wgpu::TextureFormat::R32Float);
        let lut = texture("torcdf image lut", 256, 1, wgpu::TextureFormat::Rgba8Unorm);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torcdf image params"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let data_view = data.create_view(&Default::default());
        let lut_view = lut.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("torcdf image"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&data_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
            ],
        });
        Self { ny, nx, data, lut, uniforms, bind_group, colormap: None }
    }

    fn write_data(&self, queue: &wgpu::Queue, data: &[f32]) {
        write_texture(queue, &self.data, bytemuck::cast_slice(data), self.nx, self.ny, 4);
    }

    /// Updates uniforms, and the LUT when the colour map changed.
    fn write_params(&mut self, queue: &wgpu::Queue, words: &[u32; 12], colormap: Colormap) {
        queue.write_buffer(&self.uniforms, 0, bytemuck::cast_slice(words));
        if self.colormap != Some(colormap) {
            let lut = colormap.lut();
            write_texture(queue, &self.lut, lut.as_flattened(), 256, 1, 4);
            self.colormap = Some(colormap);
        }
    }
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

/// One frame's draw of a layer.
struct ImageCallback {
    id: u64,
    rect: egui::Rect,
    params: ImageParams,
}

impl egui_wgpu::CallbackTrait for ImageCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let layer = resources
            .get_mut::<ImageResources>()
            .and_then(|images| images.layers.get_mut(&self.id));
        if let Some(layer) = layer {
            let words = uniform_words(&self.params, self.rect, screen.pixels_per_point);
            layer.write_params(queue, &words, self.params.colormap);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(images) = resources.get::<ImageResources>() else { return };
        let Some(layer) = images.layers.get(&self.id) else { return };
        render_pass.set_pipeline(&images.pipeline);
        render_pass.set_bind_group(0, &layer.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor, wgpu};
    use std::sync::Arc;

    /// egui prefers a gamma-space (non-sRGB) target; this is what the app gets.
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// A render state on the software fallback adapter (WARP on Windows), with
    /// downlevel limits so the test sees the same constraints as weak machines.
    fn render_state() -> egui_wgpu::RenderState {
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

    /// Runs one egui frame that paints `layer` over a whole `w` x `h` offscreen
    /// target through egui_wgpu's renderer, and reads the pixels back row-major.
    fn render(
        rs: &egui_wgpu::RenderState,
        w: u32,
        h: u32,
        layer: &ImageLayer,
        p: &ImageParams,
    ) -> Vec<[u8; 4]> {
        let (device, queue) = (&rs.device, &rs.queue);
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
        let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| paint(ui, screen, layer, p));
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

        let row_bytes = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // >= 4 * w for small tests
        assert!(4 * w <= row_bytes);
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

    fn params(flip_y: bool) -> ImageParams {
        ImageParams {
            range: (0.0, 1.0),
            colormap: Colormap::Viridis,
            log: false,
            view: egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(2.0, 2.0)),
            flip_y,
        }
    }

    fn near(got: [u8; 4], want: [u8; 4]) -> bool {
        (0..4).all(|c| (got[c] as i32 - want[c] as i32).abs() <= 1)
    }

    #[test]
    fn render_two_by_two_viridis_on_fallback_adapter() {
        let rs = render_state();
        let layer = upload(&rs, 7, &[0.0, 1.0, f32::NAN, 0.5], 2, 2).expect("upload");
        let px = render(&rs, 2, 2, &layer, &params(false));
        let lut = Colormap::Viridis.lut();
        // Pixel (row, col) = px[row * 2 + col]; row 0 is the top of the screen.
        assert_eq!(px[0], lut[0], "pixel (0,0)");
        assert_eq!(px[1], lut[255], "pixel (0,1)");
        assert_eq!(px[2][3], 0, "pixel (1,0) is NaN, must be transparent: {:?}", px[2]);
        assert!(near(px[3], lut[128]), "pixel (1,1): {:?} vs {:?}", px[3], lut[128]);
    }

    #[test]
    fn render_flip_y_puts_last_row_on_top() {
        let rs = render_state();
        let layer = upload(&rs, 8, &[0.0, 1.0, f32::NAN, 0.5], 2, 2).expect("upload");
        let px = render(&rs, 2, 2, &layer, &params(true));
        let lut = Colormap::Viridis.lut();
        assert_eq!(px[0][3], 0, "top-left is data row 1, col 0 (NaN): {:?}", px[0]);
        assert!(near(px[1], lut[128]), "top-right: {:?}", px[1]);
        assert_eq!(px[2], lut[0], "bottom-left");
        assert_eq!(px[3], lut[255], "bottom-right");
    }

    #[test]
    fn render_log_scale_and_non_positive_transparent() {
        let rs = render_state();
        let layer = upload(&rs, 9, &[1.0, 100.0, 0.0, 10.0], 2, 2).expect("upload");
        let p = ImageParams { range: (1.0, 100.0), log: true, ..params(false) };
        let px = render(&rs, 2, 2, &layer, &p);
        let lut = Colormap::Viridis.lut();
        assert_eq!(px[0], lut[0]);
        assert_eq!(px[1], lut[255]);
        assert_eq!(px[2][3], 0, "v <= 0 is transparent in log mode: {:?}", px[2]);
        assert!(near(px[3], lut[128]), "{:?} vs {:?}", px[3], lut[128]);
    }

    #[test]
    fn render_colormap_change_redraws_same_layer() {
        let rs = render_state();
        let layer = upload(&rs, 10, &[0.0, 1.0, f32::NAN, 0.5], 2, 2).expect("upload");
        let _ = render(&rs, 2, 2, &layer, &params(false));
        let p = ImageParams { colormap: Colormap::Diverging, ..params(false) };
        let px = render(&rs, 2, 2, &layer, &p);
        let lut = Colormap::Diverging.lut();
        assert_eq!(px[0], lut[0]);
        assert_eq!(px[1], lut[255]);
    }

    #[test]
    fn render_normalisation_linear_log_and_degenerate() {
        assert_eq!(normalisation((2.0, 6.0), false), (2.0, 0.25));
        // Log mode: 10 sits halfway between 1 and 100.
        let (offset, scale) = normalisation((1.0, 100.0), true);
        assert!(((10f32.log2() - offset) * scale - 0.5).abs() < 1e-6);
        assert_eq!(normalisation((3.0, 3.0), false), (0.0, 0.0));
        assert_eq!(normalisation((0.0, 100.0), true), (0.0, 0.0));
        assert_eq!(normalisation((f32::NAN, 1.0), false), (0.0, 0.0));
    }

    #[test]
    fn render_upload_rejects_bad_shapes() {
        let rs = render_state();
        assert_eq!(upload(&rs, 1, &[], 0, 0), Err(UploadError::Empty));
        assert_eq!(
            upload(&rs, 1, &[0.0; 3], 2, 2),
            Err(UploadError::LengthMismatch { expected: 4, got: 3 })
        );
        let wide = vec![0.0; 2049];
        assert_eq!(
            upload(&rs, 1, &wide, 1, 2049),
            Err(UploadError::TooLarge { ny: 1, nx: 2049, max: 2048 })
        );
    }
}
