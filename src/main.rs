use anyhow::Result;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};
use cgmath::{Point3, Vector3};

mod camera;
mod controller;
mod material;
mod model;

use camera::{Camera, CameraUniform};
use controller::InputState;
use material::Material;
use model::{Model, Vertex};
use std::time::Instant;
use cgmath::InnerSpace;
use half::f16;

fn pick_env_hdr_path(model_paths: &[String]) -> Option<PathBuf> {
    fn score(name: &str) -> i32 {
        let n = name.to_ascii_lowercase();
        if n.contains("skybox") {
            300
        } else if n.contains("reflection") && n.contains("interior") {
            250
        } else if n.contains("reflection") {
            200
        } else {
            100
        }
    }

    let mut best: Option<(i32, PathBuf)> = None;
    for p in model_paths {
        let path = Path::new(p);
        let dirs = [path.parent(), path.parent().and_then(|d| d.parent())];
        for dir in dirs.into_iter().flatten() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let ep = entry.path();
                    if ep.extension().and_then(|s| s.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("hdr")) {
                        let name = ep.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        let s = score(name);
                        if best.as_ref().is_none_or(|(bs, _)| s > *bs) {
                            best = Some((s, ep));
                        }
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

fn opengl_to_wgpu_matrix() -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    )
}

fn compute_light_view_proj(light_dir: Vector3<f32>, scene_min: Point3<f32>, scene_max: Point3<f32>) -> cgmath::Matrix4<f32> {
    let center = Point3::new(
        (scene_min.x + scene_max.x) * 0.5,
        (scene_min.y + scene_max.y) * 0.5,
        (scene_min.z + scene_max.z) * 0.5,
    );
    let extent = Vector3::new(
        scene_max.x - scene_min.x,
        scene_max.y - scene_min.y,
        scene_max.z - scene_min.z,
    );
    let radius = (extent.magnitude() * 0.5).max(1.0);

    let up_l = if light_dir.y.abs() > 0.95 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let light_pos = center - light_dir * (radius * 3.0 + 10.0);
    let light_view = cgmath::Matrix4::look_at_rh(light_pos, center, up_l);

    let corners = [
        Point3::new(scene_min.x, scene_min.y, scene_min.z),
        Point3::new(scene_min.x, scene_min.y, scene_max.z),
        Point3::new(scene_min.x, scene_max.y, scene_min.z),
        Point3::new(scene_min.x, scene_max.y, scene_max.z),
        Point3::new(scene_max.x, scene_min.y, scene_min.z),
        Point3::new(scene_max.x, scene_min.y, scene_max.z),
        Point3::new(scene_max.x, scene_max.y, scene_min.z),
        Point3::new(scene_max.x, scene_max.y, scene_max.z),
    ];

    let mut min_ls = cgmath::Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_ls = cgmath::Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in corners {
        let lp = light_view * cgmath::Vector4::new(p.x, p.y, p.z, 1.0);
        min_ls.x = min_ls.x.min(lp.x);
        min_ls.y = min_ls.y.min(lp.y);
        min_ls.z = min_ls.z.min(lp.z);
        max_ls.x = max_ls.x.max(lp.x);
        max_ls.y = max_ls.y.max(lp.y);
        max_ls.z = max_ls.z.max(lp.z);
    }

    let half_x = ((max_ls.x - min_ls.x) * 0.5).max(0.01);
    let half_y = ((max_ls.y - min_ls.y) * 0.5).max(0.01);
    let half_size = half_x.max(half_y) * 1.05;

    let center_x = (min_ls.x + max_ls.x) * 0.5;
    let center_y = (min_ls.y + max_ls.y) * 0.5;

    let shadow_res = 4096.0;
    let texel = (2.0 * half_size) / shadow_res;
    let snapped_x = (center_x / texel).floor() * texel;
    let snapped_y = (center_y / texel).floor() * texel;

    let left = snapped_x - half_size;
    let right_o = snapped_x + half_size;
    let bottom = snapped_y - half_size;
    let top = snapped_y + half_size;

    let z_margin = radius * 0.6 + 10.0;
    let near_z = (-max_ls.z - z_margin).max(0.1);
    let far_z = (-min_ls.z + z_margin).max(near_z + 0.1);

    let light_proj = cgmath::ortho(left, right_o, bottom, top, near_z, far_z);
    opengl_to_wgpu_matrix() * light_proj * light_view
}

fn compute_cascade_view_proj(
    light_dir: Vector3<f32>,
    camera: &Camera,
    near: f32,
    far: f32,
    scene_min: Point3<f32>,
    scene_max: Point3<f32>,
) -> cgmath::Matrix4<f32> {
    use cgmath::Matrix4;

    let up_l = if light_dir.y.abs() > 0.95 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };

    let forward = camera.forward();
    let center = camera.position + forward * ((near + far) * 0.5);
    let center = Point3::new(center.x, center.y, center.z);

    let tan_half_v = (camera.fovy.to_radians() * 0.5).tan();
    let tan_half_h = tan_half_v * camera.aspect;

    let far_half_h = far * tan_half_h;
    let far_half_v = far * tan_half_v;
    let near_half_h = near * tan_half_h;
    let near_half_v = near * tan_half_v;

    let far_radius = (far * far + far_half_h * far_half_h + far_half_v * far_half_v).sqrt();
    let near_radius = (near * near + near_half_h * near_half_h + near_half_v * near_half_v).sqrt();
    let mut radius = far_radius.max(near_radius);

    let shadow_res = 4096.0;
    let texel = (2.0 * radius) / shadow_res;
    radius = (radius / texel).ceil() * texel;

    let light_pos = center - light_dir * (radius * 4.0 + 200.0);
    let light_view = Matrix4::look_at_rh(light_pos, center, up_l);

    let center_ls = light_view * cgmath::Vector4::new(center.x, center.y, center.z, 1.0);
    let snapped_x = (center_ls.x / texel).round() * texel;
    let snapped_y = (center_ls.y / texel).round() * texel;

    let min_x = snapped_x - radius;
    let max_x = snapped_x + radius;
    let min_y = snapped_y - radius;
    let max_y = snapped_y + radius;

    let scene_corners = [
        Point3::new(scene_min.x, scene_min.y, scene_min.z),
        Point3::new(scene_min.x, scene_min.y, scene_max.z),
        Point3::new(scene_min.x, scene_max.y, scene_min.z),
        Point3::new(scene_min.x, scene_max.y, scene_max.z),
        Point3::new(scene_max.x, scene_min.y, scene_min.z),
        Point3::new(scene_max.x, scene_min.y, scene_max.z),
        Point3::new(scene_max.x, scene_max.y, scene_min.z),
        Point3::new(scene_max.x, scene_max.y, scene_max.z),
    ];

    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for p in &scene_corners {
        let lp = light_view * cgmath::Vector4::new(p.x, p.y, p.z, 1.0);
        min_z = min_z.min(lp.z);
        max_z = max_z.max(lp.z);
    }
    let margin_z = radius * 2.0 + 200.0;
    min_z -= margin_z;
    max_z += margin_z;

    let light_proj = cgmath::ortho(min_x, max_x, min_y, max_y, -max_z, -min_z);
    opengl_to_wgpu_matrix() * light_proj * light_view
}

struct SceneMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    material_index: usize,
}

#[derive(Copy, Clone)]
struct MaterialMeta {
    alpha_mode: model::AlphaMode,
    double_sided: bool,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    render_pipeline_opaque_cull: wgpu::RenderPipeline,
    render_pipeline_opaque_nocull: wgpu::RenderPipeline,
    render_pipeline_alpha_cull: wgpu::RenderPipeline,
    render_pipeline_alpha_nocull: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    shadow_camera_buffers: [wgpu::Buffer; 4],
    shadow_camera_bind_groups: [wgpu::BindGroup; 4],
    input: InputState,
    last_frame: Instant,
    meshes: Vec<SceneMesh>,
    materials: Vec<Material>,
    material_meta: Vec<MaterialMeta>,
    light_dir: Vector3<f32>,
    light_view_proj: cgmath::Matrix4<f32>,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    shadow_texture: wgpu::Texture,
    shadow_texture_view: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    env_texture: wgpu::Texture,
    env_texture_view: wgpu::TextureView,
    env_sampler: wgpu::Sampler,
    scene_center: Point3<f32>,
    scene_radius: f32,
    scene_min: Point3<f32>,
    scene_max: Point3<f32>,
}

impl State {
    async fn new(window: Window) -> Result<Self> {
        let window = Arc::new(window);
        let size = window.inner_size();
        
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(window.clone())?;
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                    label: None,
                },
                None,
            )
            .await?;
        
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut model_paths: Vec<String> = std::env::args().skip(1).collect();
        if model_paths.is_empty() {
            model_paths.push("assets/models/environment/IntelSponza/NewSponza_Main_glTF_003.gltf".to_string());
        }

        let mut loaded_models: Vec<Model> = Vec::new();
        let mut offset_x = 0.0f32;
        let padding = 2.0f32;

        let mut scene_min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut scene_max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for path in &model_paths {
            let mut m = Model::load(path)?;

            let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
            let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
            for mesh in &m.meshes {
                for v in &mesh.vertices {
                    min.x = min.x.min(v.position[0]);
                    min.y = min.y.min(v.position[1]);
                    min.z = min.z.min(v.position[2]);
                    max.x = max.x.max(v.position[0]);
                    max.y = max.y.max(v.position[1]);
                    max.z = max.z.max(v.position[2]);
                }
            }
            let width = (max.x - min.x).max(1.0);

            if offset_x != 0.0 {
                for mesh in &mut m.meshes {
                    for v in &mut mesh.vertices {
                        v.position[0] += offset_x;
                    }
                }
                min.x += offset_x;
                max.x += offset_x;
            }

            scene_min.x = scene_min.x.min(min.x);
            scene_min.y = scene_min.y.min(min.y);
            scene_min.z = scene_min.z.min(min.z);
            scene_max.x = scene_max.x.max(max.x);
            scene_max.y = scene_max.y.max(max.y);
            scene_max.z = scene_max.z.max(max.z);

            loaded_models.push(m);
            offset_x += width + padding;
        }

        let scene_center = Point3::new(
            (scene_min.x + scene_max.x) * 0.5,
            (scene_min.y + scene_max.y) * 0.5,
            (scene_min.z + scene_max.z) * 0.5,
        );
        let extent = Vector3::new(
            scene_max.x - scene_min.x,
            scene_max.y - scene_min.y,
            scene_max.z - scene_min.z,
        );
        let scene_radius = (extent.magnitude() * 0.5).max(1.0);

        let mut camera = Camera::new(size.width, size.height);
        camera.set_look_at(
            scene_center + Vector3::new(0.0, scene_radius * 0.5 + 1.0, scene_radius * 2.0 + 2.0),
            scene_center,
        );
        
        let mut camera_uniform = CameraUniform::new();

        let light_dir = Vector3::new(0.0f32, -1.0f32, 0.0f32);
        let light_view_proj = compute_light_view_proj(light_dir, scene_min, scene_max);

        camera_uniform.update(&camera, light_view_proj, light_dir, 1.0);
        
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let shadow_camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("shadow_camera_bind_group_layout"),
            });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("camera_bind_group_layout"),
            });
        
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Texture Array"),
            size: wgpu::Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_texture_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Texture View"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(4),
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let (env_texture, env_texture_view, env_sampler) = {
            let fallback_hdr = PathBuf::from("assets/models/environment/IntelSponza/textures/kloppenheim_05_4k.hdr");
            let hdr_path = pick_env_hdr_path(&model_paths).unwrap_or(fallback_hdr);
            let bytes = std::fs::read(&hdr_path).unwrap_or_default();
            let mut width = 1u32;
            let mut height = 1u32;
            let mut rgba16: Vec<u16> = vec![f16::from_f32(0.0).to_bits(), f16::from_f32(0.0).to_bits(), f16::from_f32(0.0).to_bits(), f16::from_f32(1.0).to_bits()];
            if !bytes.is_empty() {
                if let Ok(img) = image::load_from_memory(&bytes) {
                    match img {
                        image::DynamicImage::ImageRgb32F(buf) => {
                            width = buf.width();
                            height = buf.height();
                            rgba16 = Vec::with_capacity((width * height * 4) as usize);
                            for p in buf.pixels() {
                                rgba16.push(f16::from_f32(p.0[0]).to_bits());
                                rgba16.push(f16::from_f32(p.0[1]).to_bits());
                                rgba16.push(f16::from_f32(p.0[2]).to_bits());
                                rgba16.push(f16::from_f32(1.0).to_bits());
                            }
                        }
                        image::DynamicImage::ImageRgba32F(buf) => {
                            width = buf.width();
                            height = buf.height();
                            rgba16 = Vec::with_capacity((width * height * 4) as usize);
                            for p in buf.pixels() {
                                rgba16.push(f16::from_f32(p.0[0]).to_bits());
                                rgba16.push(f16::from_f32(p.0[1]).to_bits());
                                rgba16.push(f16::from_f32(p.0[2]).to_bits());
                                rgba16.push(f16::from_f32(p.0[3]).to_bits());
                            }
                        }
                        other => {
                            let rgba = other.to_rgba8();
                            width = rgba.width();
                            height = rgba.height();
                            rgba16 = Vec::with_capacity((width * height * 4) as usize);
                            for p in rgba.pixels() {
                                let r = (p.0[0] as f32) / 255.0;
                                let g = (p.0[1] as f32) / 255.0;
                                let b = (p.0[2] as f32) / 255.0;
                                let a = (p.0[3] as f32) / 255.0;
                                rgba16.push(f16::from_f32(r).to_bits());
                                rgba16.push(f16::from_f32(g).to_bits());
                                rgba16.push(f16::from_f32(b).to_bits());
                                rgba16.push(f16::from_f32(a).to_bits());
                            }
                        }
                    }
                }
            }

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Env Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&rgba16),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(8 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (texture, view, sampler)
        };

        let shadow_camera_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            let mut u = camera_uniform;
            u.light_view_proj = light_view_proj.into();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Shadow Camera Buffer {}", i)),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        let shadow_camera_bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &shadow_camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_camera_buffers[i].as_entire_binding(),
                }],
                label: Some(&format!("shadow_camera_bind_group {}", i)),
            })
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&env_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&env_sampler),
                },
            ],
            label: Some("camera_bind_group"),
        });
        
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("material_bind_group_layout"),
            });
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &material_bind_group_layout],
                push_constant_ranges: &[],
            });

        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&shadow_camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_state = wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        };

        let make_pipeline = |label: &str,
                             blend: wgpu::BlendState,
                             depth_write: bool,
                             depth_compare: wgpu::CompareFunction,
                             cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_pipeline_layout),
                cache: None,
                vertex: vertex_state.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: depth_write,
                    depth_compare,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };

        let render_pipeline_opaque_cull = make_pipeline(
            "Render Pipeline Opaque Cull",
            wgpu::BlendState::REPLACE,
            true,
            wgpu::CompareFunction::Less,
            Some(wgpu::Face::Back),
        );
        let render_pipeline_opaque_nocull = make_pipeline(
            "Render Pipeline Opaque NoCull",
            wgpu::BlendState::REPLACE,
            true,
            wgpu::CompareFunction::Less,
            None,
        );
        let render_pipeline_alpha_cull = make_pipeline(
            "Render Pipeline Alpha Cull",
            wgpu::BlendState::ALPHA_BLENDING,
            false,
            wgpu::CompareFunction::LessEqual,
            Some(wgpu::Face::Back),
        );
        let render_pipeline_alpha_nocull = make_pipeline(
            "Render Pipeline Alpha NoCull",
            wgpu::BlendState::ALPHA_BLENDING,
            false,
            wgpu::CompareFunction::LessEqual,
            None,
        );

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_shadow",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&sky_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let default_base_color_texture = material::create_default_texture_pixel(
            &device,
            &queue,
            [255, 255, 255, 255],
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let default_metallic_roughness_texture = material::create_default_texture_pixel(
            &device,
            &queue,
            [0, 255, 0, 255],
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let mut meshes: Vec<SceneMesh> = Vec::new();
        let mut materials: Vec<Material> = Vec::new();
        let mut material_meta: Vec<MaterialMeta> = Vec::new();

        for model in loaded_models {
            let material_offset = materials.len();
            for mat in &model.materials {
                materials.push(Material::from_model_material(
                    &device,
                    &queue,
                    &material_bind_group_layout,
                    mat,
                    &model.textures,
                    &default_base_color_texture,
                    &default_metallic_roughness_texture,
                ));
                material_meta.push(MaterialMeta {
                    alpha_mode: mat.alpha_mode,
                    double_sided: mat.double_sided,
                });
            }

            for mesh in &model.meshes {
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                meshes.push(SceneMesh {
                    vertex_buffer,
                    index_buffer,
                    index_count: mesh.indices.len() as u32,
                    material_index: material_offset + mesh.material_index,
                });
            }
        }
        
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline_opaque_cull,
            render_pipeline_opaque_nocull,
            render_pipeline_alpha_cull,
            render_pipeline_alpha_nocull,
            sky_pipeline,
            shadow_pipeline,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            shadow_camera_buffers,
            shadow_camera_bind_groups,
            input: InputState::new(),
            last_frame: Instant::now(),
            meshes,
            materials,
            material_meta,
            light_dir,
            light_view_proj,
            depth_texture,
            depth_texture_view,
            shadow_texture,
            shadow_texture_view,
            shadow_sampler,
            env_texture,
            env_texture_view,
            env_sampler,
            scene_center,
            scene_radius,
            scene_min,
            scene_max,
        })
    }
    
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            
            self.camera.update_aspect(new_size.width, new_size.height);
            
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            
            self.depth_texture_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }
    
    fn input(&mut self, event: &WindowEvent) -> bool {
        let used = self.input.on_window_event(event);
        if self.input.mouse_captured {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
            self.window.set_cursor_visible(false);
        }
        used
    }
    
    fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        let (dx, dy) = self.input.take_mouse_delta();
        if self.input.mouse_captured {
            self.camera.apply_mouse_look(dx, dy, 0.002);
        }

        let mut wish = Vector3::new(0.0, 0.0, 0.0);
        if self.input.forward {
            wish += self.camera.forward();
        }
        if self.input.back {
            wish -= self.camera.forward();
        }
        if self.input.right {
            wish += self.camera.right();
        }
        if self.input.left {
            wish -= self.camera.right();
        }
        if self.input.up {
            wish += self.camera.up;
        }
        if self.input.down {
            wish -= self.camera.up;
        }
        if wish.magnitude2() > 0.0 {
            wish = wish.normalize();
        }

        let speed = if self.input.sprint { 18.0 } else { 6.0 };
        self.camera.move_fly(wish, dt, speed);

        let cascade_splits = [
            self.camera.znear + 0.05 * (self.camera.zfar - self.camera.znear),
            self.camera.znear + 0.15 * (self.camera.zfar - self.camera.znear),
            self.camera.znear + 0.40 * (self.camera.zfar - self.camera.znear),
            self.camera.zfar,
        ];

        let light_view_projs = [
            compute_cascade_view_proj(
                self.light_dir,
                &self.camera,
                self.camera.znear,
                cascade_splits[0],
                self.scene_min,
                self.scene_max,
            ),
            compute_cascade_view_proj(
                self.light_dir,
                &self.camera,
                cascade_splits[0],
                cascade_splits[1],
                self.scene_min,
                self.scene_max,
            ),
            compute_cascade_view_proj(
                self.light_dir,
                &self.camera,
                cascade_splits[1],
                cascade_splits[2],
                self.scene_min,
                self.scene_max,
            ),
            compute_cascade_view_proj(
                self.light_dir,
                &self.camera,
                cascade_splits[2],
                cascade_splits[3],
                self.scene_min,
                self.scene_max,
            ),
        ];

        let env_intensity = self.camera_uniform.env_intensity[0];
        self.camera_uniform.update_with_cascades(
            &self.camera,
            light_view_projs,
            cascade_splits,
            self.light_dir,
            env_intensity,
        );
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        for i in 0..4 {
            let mut u = self.camera_uniform;
            u.light_view_proj = light_view_projs[i].into();
            self.queue.write_buffer(
                &self.shadow_camera_buffers[i],
                0,
                bytemuck::cast_slice(&[u]),
            );
        }
    }
    
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        for cascade in 0..4 {
            let shadow_layer_view = self.shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Shadow Layer {}", cascade)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: cascade,
                array_layer_count: Some(1),
            });

            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Shadow Pass Cascade {}", cascade)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shadow_layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(0, &self.shadow_camera_bind_groups[cascade as usize], &[]);
            for mesh in &self.meshes {
                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                shadow_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            render_pass.set_pipeline(&self.sky_pipeline);
            render_pass.draw(0..3, 0..1);

            for mesh in &self.meshes {
                let material_index = mesh.material_index.min(self.materials.len().saturating_sub(1));
                let meta = self
                    .material_meta
                    .get(material_index)
                    .copied()
                    .unwrap_or(MaterialMeta {
                        alpha_mode: model::AlphaMode::Opaque,
                        double_sided: false,
                    });
                if meta.alpha_mode == model::AlphaMode::Blend {
                    continue;
                }
                let pipeline = if meta.double_sided {
                    &self.render_pipeline_opaque_nocull
                } else {
                    &self.render_pipeline_opaque_cull
                };
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(1, &self.materials[material_index].bind_group, &[]);
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            for mesh in &self.meshes {
                let material_index = mesh.material_index.min(self.materials.len().saturating_sub(1));
                let meta = self
                    .material_meta
                    .get(material_index)
                    .copied()
                    .unwrap_or(MaterialMeta {
                        alpha_mode: model::AlphaMode::Opaque,
                        double_sided: false,
                    });
                if meta.alpha_mode != model::AlphaMode::Blend {
                    continue;
                }
                let pipeline = if meta.double_sided {
                    &self.render_pipeline_alpha_nocull
                } else {
                    &self.render_pipeline_alpha_cull
                };
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(1, &self.materials[material_index].bind_group, &[]);
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }
}

fn main() -> Result<()> {
    env_logger::init();
    
    let event_loop = EventLoop::new()?;
    let window = event_loop.create_window(
        WindowAttributes::default()
            .with_title("Dusk Engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
    )?;
    
    let mut state = pollster::block_on(State::new(window))?;
    
    event_loop.run(move |event, elwt| {
        match event {
            Event::DeviceEvent { event, .. } => {
                if let DeviceEvent::MouseMotion { delta } = event {
                    state.input.on_mouse_motion(delta);
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    state: ElementState::Pressed,
                                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                                    ..
                                },
                            ..
                        } => elwt.exit(),
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                state.window.request_redraw();
            }
            _ => {}
        }
    })?;
    
    Ok(())
}
