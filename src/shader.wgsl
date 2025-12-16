struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) light_clip: vec4<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    position: vec4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    env_intensity: vec4<f32>,
};

struct Material {
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    alpha_cutoff_flags: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var shadow_map: texture_depth_2d;

@group(0) @binding(2)
var shadow_sampler: sampler_comparison;

@group(0) @binding(3)
var env_map: texture_2d<f32>;

@group(0) @binding(4)
var env_sampler: sampler;

@group(1) @binding(0)
var<uniform> material: Material;

@group(1) @binding(1)
var base_color_texture: texture_2d<f32>;

@group(1) @binding(2)
var metallic_roughness_texture: texture_2d<f32>;

@group(1) @binding(3)
var material_sampler: sampler;

const PI: f32 = 3.14159265359;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = position;
    out.normal = normal;
    out.tex_coords = tex_coords;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.light_clip = camera.light_view_proj * vec4<f32>(position, 1.0);
    return out;
}

@vertex
fn vs_shadow(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
) -> @builtin(position) vec4<f32> {
    return camera.light_view_proj * vec4<f32>(position, 1.0);
}

fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

fn shadow_pcf(light_clip: vec4<f32>, depth_bias: f32) -> f32 {
    if light_clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = light_clip.xyz / light_clip.w;
    let uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return 1.0;
    }
    let depth = ndc.z;
    // Smaller kernel reduces shadow bleeding/light leaks through thin geometry.
    let texel = 1.0 / vec2<f32>(2048.0, 2048.0);
    var sum = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let o = vec2<f32>(f32(x), f32(y)) * texel;
            sum = sum + textureSampleCompare(shadow_map, shadow_sampler, uv + o, depth - depth_bias);
        }
    }
    return sum / 9.0;
}

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    
    let nom = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / denom;
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    
    let nom = NdotV;
    let denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = geometry_schlick_ggx(NdotV, roughness);
    let ggx1 = geometry_schlick_ggx(NdotL, roughness);
    
    return ggx1 * ggx2;
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_sample = textureSample(base_color_texture, material_sampler, in.tex_coords);
    let albedo = base_sample.rgb * material.base_color.rgb;
    let alpha = base_sample.a * material.base_color.a;

    if material.alpha_cutoff_flags.y >= 0.5 && material.alpha_cutoff_flags.y < 1.5 {
        if alpha < material.alpha_cutoff_flags.x {
            discard;
        }
    }

    let mr_sample = textureSample(metallic_roughness_texture, material_sampler, in.tex_coords).rgb;
    let metallic = clamp(mr_sample.b * material.metallic_roughness.r, 0.0, 1.0);
    let roughness = clamp(mr_sample.g * material.metallic_roughness.g, 0.04, 1.0);
    
    let N = normalize(in.normal);
    let V = normalize(camera.position.xyz - in.world_position);

    let ndotl = max(dot(N, normalize(-camera.light_dir.xyz)), 0.0);
    // Conservative receiver bias to reduce light leaking/peter-panning.
    let bias = max(0.00008, 0.0012 * (1.0 - ndotl));
    let shadow = shadow_pcf(in.light_clip, bias);
    
    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);
    
    var Lo = vec3<f32>(0.0);

    let L = normalize(-camera.light_dir.xyz);
    let H = normalize(V + L);
    let radiance = vec3<f32>(6.0, 6.0, 6.0);

    let NDF = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD = kD * (1.0 - metallic);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    let specular = numerator / denominator;

    let NdotL = max(dot(N, L), 0.0);
    Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow;
    
    let env_uv = dir_to_equirect_uv(N);
    let env_col = textureSample(env_map, env_sampler, env_uv).rgb;
    let ambient = env_col * albedo * camera.env_intensity.rgb;
    var color = ambient + Lo;
    
    color = color / (color + vec3<f32>(1.0));

    if material.alpha_cutoff_flags.y >= 1.5 {
        return vec4<f32>(color, alpha);
    }

    return vec4<f32>(color, 1.0);
}
