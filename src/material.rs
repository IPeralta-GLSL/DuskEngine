use wgpu::util::DeviceExt;
use crate::model::{Material as ModelMaterial, Texture as ModelTexture};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    pub metallic_roughness: [f32; 4],
    pub alpha_cutoff_flags: [f32; 4],
}

pub struct Material {
    pub uniform: MaterialUniform,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn from_model_material(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        material: &ModelMaterial,
        textures: &[ModelTexture],
        default_base_color_texture: &wgpu::Texture,
        default_metallic_roughness_texture: &wgpu::Texture,
    ) -> Self {
        let alpha_mode = match material.alpha_mode {
            crate::model::AlphaMode::Opaque => 0.0,
            crate::model::AlphaMode::Mask => 1.0,
            crate::model::AlphaMode::Blend => 2.0,
        };
        let double_sided = if material.double_sided { 1.0 } else { 0.0 };

        let uniform = MaterialUniform {
            base_color: material.base_color,
            metallic_roughness: [material.metallic, material.roughness, 0.0, 0.0],
            alpha_cutoff_flags: [material.alpha_cutoff, alpha_mode, double_sided, 0.0],
        };
        
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let base_color_view = if let Some(img_idx) = material.base_color_image {
            if let Some(model_texture) = textures.get(img_idx) {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Base Color Texture"),
                    size: wgpu::Extent3d {
                        width: model_texture.width,
                        height: model_texture.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: model_texture.format,
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
                    &model_texture.data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * model_texture.width),
                        rows_per_image: Some(model_texture.height),
                    },
                    wgpu::Extent3d {
                        width: model_texture.width,
                        height: model_texture.height,
                        depth_or_array_layers: 1,
                    },
                );
                
                texture.create_view(&wgpu::TextureViewDescriptor::default())
            } else {
                default_base_color_texture.create_view(&wgpu::TextureViewDescriptor::default())
            }
        } else {
            default_base_color_texture.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let metallic_roughness_view = if let Some(img_idx) = material.metallic_roughness_image {
            if let Some(model_texture) = textures.get(img_idx) {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Metallic Roughness Texture"),
                    size: wgpu::Extent3d {
                        width: model_texture.width,
                        height: model_texture.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: model_texture.format,
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
                    &model_texture.data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * model_texture.width),
                        rows_per_image: Some(model_texture.height),
                    },
                    wgpu::Extent3d {
                        width: model_texture.width,
                        height: model_texture.height,
                        depth_or_array_layers: 1,
                    },
                );

                texture.create_view(&wgpu::TextureViewDescriptor::default())
            } else {
                default_metallic_roughness_texture.create_view(&wgpu::TextureViewDescriptor::default())
            }
        } else {
            default_metallic_roughness_texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&base_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&metallic_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        
        Self {
            uniform,
            bind_group,
        }
    }
}

pub fn create_default_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Default Texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
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
        &[255u8, 255, 255, 255],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    
    texture
}

pub fn create_default_texture_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixel: [u8; 4],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Default Texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
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
        &pixel,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    texture
}
