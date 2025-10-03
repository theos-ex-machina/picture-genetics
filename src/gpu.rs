use bytemuck::{Pod, Zeroable};
use image::RgbaImage;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuBlock {
    x: f32,
    y: f32,
    scale: f32,
    texture_idx: u32,
    score: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    tex_width: u32,
    tex_height: u32,
    base_ssd: f32,
    sampling: u32,
    _pad: [u32; 2],
}

pub struct GpuScorer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl GpuScorer {
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await?;

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()?;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/score_calculation.wgsl").into(),
                ),
            });

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

            Some(Self {
                device,
                queue,
                pipeline,
                layout,
            })
        })
    }

    pub fn score_blocks(
        &self,
        target: &RgbaImage,
        canvas: &RgbaImage,
        texture: &RgbaImage,
        blocks: &mut [(f32, f32, f32)], // (x, y, scale)
        base_ssd: f32,
    ) -> Option<Vec<f32>> {
        // Check if image buffers would exceed GPU limits (128MB per buffer)
        const MAX_BUFFER_SIZE: usize = 128 * 1024 * 1024; // 128MB
        let target_size = (target.width() * target.height() * 4 * 4) as usize; // RGBA, f32
        let canvas_size = (canvas.width() * canvas.height() * 4 * 4) as usize;

        if target_size > MAX_BUFFER_SIZE || canvas_size > MAX_BUFFER_SIZE {
            // Images too large for GPU buffers, use CPU instead
            return None;
        }

        pollster::block_on(async {
            let target_data: Vec<[f32; 4]> = target
                .pixels()
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                        p[3] as f32 / 255.0,
                    ]
                })
                .collect();

            let canvas_data: Vec<[f32; 4]> = canvas
                .pixels()
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                        p[3] as f32 / 255.0,
                    ]
                })
                .collect();

            let texture_data: Vec<[f32; 4]> = texture
                .pixels()
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                        p[3] as f32 / 255.0,
                    ]
                })
                .collect();

            let mut gpu_blocks: Vec<GpuBlock> = blocks
                .iter()
                .map(|(x, y, scale)| GpuBlock {
                    x: *x,
                    y: *y,
                    scale: *scale,
                    texture_idx: 0,
                    score: 0.0,
                    _pad: [0.0; 3],
                })
                .collect();

            let uniforms = Uniforms {
                width: target.width(),
                height: target.height(),
                tex_width: texture.width(),
                tex_height: texture.height(),
                base_ssd,
                sampling: crate::SCORE_SAMPLING_RATE as u32,
                _pad: [0; 2],
            };

            let target_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&target_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let canvas_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&canvas_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let texture_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&texture_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let blocks_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&gpu_blocks),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

            let uniforms_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (gpu_blocks.len() * std::mem::size_of::<GpuBlock>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: target_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: canvas_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: texture_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: blocks_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: uniforms_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = ((blocks.len() as u32 + 255) / 256).max(1);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(
                &blocks_buf,
                0,
                &staging_buf,
                0,
                (gpu_blocks.len() * std::mem::size_of::<GpuBlock>()) as u64,
            );

            self.queue.submit(Some(encoder.finish()));

            let slice = staging_buf.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).ok();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().ok();

            let data = slice.get_mapped_range();
            let result: &[GpuBlock] = bytemuck::cast_slice(&data);
            gpu_blocks.copy_from_slice(result);
            drop(data);
            staging_buf.unmap();

            Some(gpu_blocks.iter().map(|b| b.score).collect())
        })
    }
}
