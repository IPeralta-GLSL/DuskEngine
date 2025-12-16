use anyhow::{Context, Result};
use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, SquareMatrix, Vector3, Vector4};
use std::io::Cursor;
use std::{fs, path::Path};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

pub struct Material {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub base_color_image: Option<usize>,
    pub metallic_roughness_image: Option<usize>,
    pub normal_image: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub base_color_texcoord_set: u32,
    pub metallic_roughness_texcoord_set: u32,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_index: usize,
}

pub struct Texture {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub has_alpha: bool,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
}

impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext.eq_ignore_ascii_case("fbx") {
                anyhow::bail!(
                    "FBX is not supported natively. Convert it to glTF/GLB first (e.g. with Blender export or 'FBX2glTF'), then load the .gltf/.glb file: {}",
                    path.display()
                );
            }
        }
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

        let gltf = match path.extension().and_then(|s| s.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("glb") => {
                let bytes = fs::read(path).with_context(|| format!("read GLB: {}", path.display()))?;
                gltf::Gltf::from_slice(&bytes).with_context(|| format!("parse GLB: {}", path.display()))?
            }
            _ => gltf::Gltf::open(path).with_context(|| format!("open glTF: {}", path.display()))?,
        };
        let document = gltf.document;

        let mut buffers: Vec<Vec<u8>> = Vec::new();
        for buffer in document.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Uri(uri) => {
                    let buf_path = base_dir.join(uri);
                    let data = fs::read(&buf_path).with_context(|| format!("read buffer: {}", buf_path.display()))?;
                    buffers.push(data);
                }
                gltf::buffer::Source::Bin => {
                    let blob = gltf
                        .blob
                        .as_ref()
                        .context("missing .blob for BIN buffer")?;
                    buffers.push(blob.clone());
                }
            }
        }

        fn try_read_uri(base_dir: &Path, uri: &str) -> Option<Vec<u8>> {
            let mut candidate = uri.replace('\\', "/");
            if candidate.contains("%20") {
                candidate = candidate.replace("%20", " ");
            }

            let parent_dir = base_dir.parent();
            let mut roots: Vec<&Path> = vec![base_dir];
            if let Some(p) = parent_dir {
                roots.push(p);
            }

            // 1) Try the full URI path relative to common roots.
            for root in &roots {
                let direct = root.join(&candidate);
                if let Ok(bytes) = fs::read(&direct) {
                    return Some(bytes);
                }
            }

            // 2) Try common texture folders (both casing variants), with full URI.
            for root in &roots {
                for folder in ["textures", "Textures"] {
                    let direct = root.join(folder).join(&candidate);
                    if let Ok(bytes) = fs::read(&direct) {
                        return Some(bytes);
                    }
                }
            }

            // 3) Fall back to filename-only search inside common texture folders.
            let file_name = Path::new(&candidate).file_name()?.to_string_lossy().to_string();
            for root in &roots {
                for folder in ["textures", "Textures"] {
                    let direct = root.join(folder).join(&file_name);
                    if let Ok(bytes) = fs::read(&direct) {
                        return Some(bytes);
                    }
                }
            }

            let file_name_lower = file_name.to_ascii_lowercase();
            for root in &roots {
                for dir in [
                    root.to_path_buf(),
                    root.join("textures"),
                    root.join("Textures"),
                ] {
                    if let Ok(entries) = fs::read_dir(&dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if !path.is_file() {
                                continue;
                            }
                            let name = match path.file_name().and_then(|s| s.to_str()) {
                                Some(s) => s,
                                None => continue,
                            };
                            if name.to_ascii_lowercase() == file_name_lower {
                                if let Ok(bytes) = fs::read(&path) {
                                    return Some(bytes);
                                }
                            }
                        }
                    }
                }
            }

            None
        }

        let mut textures: Vec<Texture> = Vec::new();
        for image in document.images() {
            let bytes_opt: Option<Vec<u8>> = match image.source() {
                gltf::image::Source::Uri { uri, .. } => {
                    try_read_uri(base_dir, uri)
                }
                gltf::image::Source::View { view, .. } => {
                    let buffer_data = &buffers[view.buffer().index()];
                    let start = view.offset();
                    let end = start + view.length();
                    Some(buffer_data[start..end].to_vec())
                }
            };

            let (data, width, height) = if let Some(bytes) = bytes_opt {
                let is_dds = bytes.len() >= 4 && &bytes[0..4] == b"DDS ";

                if is_dds {
                    let mut cur = Cursor::new(&bytes);
                    if let Ok(dds) = image_dds::ddsfile::Dds::read(&mut cur) {
                        if let Ok(img) = image_dds::image_from_dds(&dds, 0) {
                            let (w, h) = img.dimensions();
                            (img.into_raw(), w, h)
                        } else {
                            log::warn!("Failed to decode DDS image (mip0). Using fallback 1x1 white.");
                            (vec![255u8, 255, 255, 255], 1, 1)
                        }
                    } else {
                        log::warn!("Failed to parse DDS header. Using fallback 1x1 white.");
                        (vec![255u8, 255, 255, 255], 1, 1)
                    }
                } else if let Ok(img) = image::load_from_memory(&bytes) {
                    let rgba = img.to_rgba8();
                    let (w, h) = rgba.dimensions();
                    (rgba.into_raw(), w, h)
                } else {
                    log::warn!("Failed to decode image bytes (non-DDS). Using fallback 1x1 white.");
                    (vec![255u8, 255, 255, 255], 1, 1)
                }
            } else {
                (vec![255u8, 255, 255, 255], 1, 1)
            };

            let mut has_alpha = false;
            if data.len() >= 4 {
                for a in data.iter().skip(3).step_by(4) {
                    if *a != 255 {
                        has_alpha = true;
                        break;
                    }
                }
            }

            textures.push(Texture {
                data,
                width,
                height,
                format: wgpu::TextureFormat::Rgba8Unorm,
                has_alpha,
            });
        }

        let mut materials = Vec::new();
        for material in document.materials() {
            let pbr = material.pbr_metallic_roughness();
            let base_color_image = pbr
                .base_color_texture()
                .map(|t| t.texture().source().index());
            let metallic_roughness_image = pbr
                .metallic_roughness_texture()
                .map(|t| t.texture().source().index());
            let base_color_texcoord_set = pbr.base_color_texture().map(|t| t.tex_coord()).unwrap_or(0);
            let metallic_roughness_texcoord_set = pbr
                .metallic_roughness_texture()
                .map(|t| t.tex_coord())
                .unwrap_or(0);
            let normal_image = material
                .normal_texture()
                .map(|t| t.texture().source().index());

            let mut alpha_mode = match material.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            };

            if alpha_mode == AlphaMode::Opaque {
                if let Some(img_idx) = base_color_image {
                    if textures.get(img_idx).is_some_and(|t| t.has_alpha) {
                        alpha_mode = AlphaMode::Blend;
                    }
                }
            }

            let alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);
            let double_sided = material.double_sided();

            materials.push(Material {
                base_color: pbr.base_color_factor(),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                base_color_image,
                metallic_roughness_image,
                normal_image,
                alpha_mode,
                alpha_cutoff,
                double_sided,
                base_color_texcoord_set,
                metallic_roughness_texcoord_set,
            });
        }

        if materials.is_empty() {
            materials.push(Material {
                base_color: [1.0, 1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                base_color_image: None,
                metallic_roughness_image: None,
                normal_image: None,
                alpha_mode: AlphaMode::Opaque,
                alpha_cutoff: 0.5,
                double_sided: false,
                base_color_texcoord_set: 0,
                metallic_roughness_texcoord_set: 0,
            });
        }

        for mat in &materials {
            if let Some(img_idx) = mat.base_color_image {
                if let Some(tex) = textures.get_mut(img_idx) {
                    tex.format = wgpu::TextureFormat::Rgba8UnormSrgb;
                }
            }
        }

        let mut meshes: Vec<Mesh> = Vec::new();
        let scene = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .context("glTF has no scene")?;

        fn mat4_from_cols(cols: [[f32; 4]; 4]) -> Matrix4<f32> {
            Matrix4::from(cols)
        }

        fn normal_matrix(m: Matrix4<f32>) -> Matrix3<f32> {
            let a = Matrix3::new(
                m.x.x, m.x.y, m.x.z,
                m.y.x, m.y.y, m.y.z,
                m.z.x, m.z.y, m.z.z,
            );
            a.invert().unwrap_or(Matrix3::from_scale(1.0)).transpose()
        }

        fn traverse<'a>(
            node: gltf::scene::Node<'a>,
            parent: Matrix4<f32>,
            buffers: &'a [Vec<u8>],
            materials: &'a [Material],
            meshes_out: &mut Vec<Mesh>,
        ) {
            let local = mat4_from_cols(node.transform().matrix());
            let world = parent * local;
            let nmat = normal_matrix(world);

            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                    let material_index = primitive.material().index().unwrap_or(0);
                    let uv_set = materials
                        .get(material_index)
                        .map(|m| {
                            if m.base_color_image.is_some() {
                                m.base_color_texcoord_set
                            } else {
                                m.metallic_roughness_texcoord_set
                            }
                        })
                        .unwrap_or(0);

                    let positions: Vec<[f32; 3]> = reader
                        .read_positions()
                        .map(|iter| iter.collect())
                        .unwrap_or_default();

                    let normals: Vec<[f32; 3]> = reader
                        .read_normals()
                        .map(|iter| iter.collect())
                        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

                    let tex_coords: Vec<[f32; 2]> = reader
                        .read_tex_coords(uv_set)
                        .or_else(|| reader.read_tex_coords(0))
                        .map(|iter| iter.into_f32().collect())
                        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                    let mut vertices: Vec<Vertex> = Vec::with_capacity(positions.len());
                    for ((pos, norm), uv) in positions.iter().zip(normals.iter()).zip(tex_coords.iter()) {
                        let wp = world * Vector4::new(pos[0], pos[1], pos[2], 1.0);
                        let nn = nmat * Vector3::new(norm[0], norm[1], norm[2]);
                        let nn = nn.normalize();
                        vertices.push(Vertex {
                            position: [wp.x, wp.y, wp.z],
                            normal: [nn.x, nn.y, nn.z],
                            tex_coords: *uv,
                        });
                    }

                    let indices: Vec<u32> = reader
                        .read_indices()
                        .map(|iter| iter.into_u32().collect())
                        .unwrap_or_default();

                    meshes_out.push(Mesh {
                        vertices,
                        indices,
                        material_index,
                    });
                }
            }

            for child in node.children() {
                traverse(child, world, buffers, materials, meshes_out);
            }
        }

        for node in scene.nodes() {
            traverse(
                node,
                Matrix4::from_scale(1.0),
                &buffers,
                &materials,
                &mut meshes,
            );
        }

        Ok(Model {
            meshes,
            materials,
            textures,
        })
    }
}
