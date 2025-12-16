# DuskEngine

Visor 3D en Rust + wgpu que carga glTF 2.0 (`.gltf` y `.glb`) y renderiza con PBR (baseColor + metallicRoughness).

## Ejecutar

```bash
cd /home/iperalta/Documents/Git/DuskEngine
cargo run --release
```

Por defecto carga:

`assets/models/environment/IntelSponza/NewSponza_Main_glTF_003.gltf`

Para cargar otro modelo:

```bash
cargo run --release -- path/al/modelo.glb
```

## Notas

- Si el `.gltf` referencia texturas faltantes, se usa una textura por defecto.
- wgpu estable no expone DXR/VKRT; este proyecto usa rasterización. Si quieres, puedo iterar a un path tracer por compute shader (ray tracing) como siguiente paso.
