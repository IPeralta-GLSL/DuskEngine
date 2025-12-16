# DuskEngine

Visor 3D en Rust + wgpu que carga glTF 2.0 (`.gltf` y `.glb`) y renderiza con PBR (baseColor + metallicRoughness).

## Ejecutar

```bash
cd /home/iperalta/Documents/Git/DuskEngine
cargo run --release
```

Controles:


## Converting FBX (FBX2glTF)

This engine loads `.gltf` / `.glb` (not `.fbx`).

If you have an FBX (example: `assets/models/environment/SunTemple/SunTemple.fbx`), convert it first using `FBX2glTF` and then load the resulting `.gltf`/`.glb`.

```bash
# 1) Get FBX2glTF (build or download from: https://github.com/facebookincubator/FBX2glTF)
#    If you drop the binary into ./tools/, the script will auto-detect it.
#    Example filename supported: tools/FBX2glTF-linux-x64
# 2) Convert FBX -> GLB (recommended)
chmod +x tools/convert_fbx2gltf.sh
chmod +x tools/FBX2glTF-linux-x64 || true
tools/convert_fbx2gltf.sh assets/models/environment/SunTemple/SunTemple.fbx assets/models/environment/SunTemple/converted --glb

# 3) Run the engine
cargo run --release -- assets/models/environment/SunTemple/converted/SunTemple.glb
```

Por defecto carga:

Para cargar otro modelo:

```bash
cargo run --release -- path/al/modelo.glb
```

## Notas

- Si el `.gltf` referencia texturas faltantes, se usa una textura por defecto.
- wgpu estable no expone DXR/VKRT; este proyecto usa rasterización. Si quieres, puedo iterar a un path tracer por compute shader (ray tracing) como siguiente paso.
