struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) view_depth: f32,
};

struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) dir: vec3<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    position: vec4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    env_intensity: vec4<f32>,
    cascade_splits: vec4<f32>,
    light_view_proj_cascade1: mat4x4<f32>,
    light_view_proj_cascade2: mat4x4<f32>,
    light_view_proj_cascade3: mat4x4<f32>,
};

struct Material {
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    alpha_cutoff_flags: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var shadow_map: texture_depth_2d_array;

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
const SHADOW_MAP_SIZE: f32 = 4096.0;

const POISSON_DISK: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.094184101, -0.92938870),
    vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),
    vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554),
    vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),
    vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367),
    vec2<f32>(0.14383161, -0.14100790)
);

fn select_cascade(view_depth: f32) -> i32 {
    if view_depth < camera.cascade_splits.x {
        return 0;
    } else if view_depth < camera.cascade_splits.y {
        return 1;
    } else if view_depth < camera.cascade_splits.z {
        return 2;
    }
    return 3;
}

struct CascadeBlend {
    c0: i32,
    c1: i32,
    t: f32,
};

fn cascade_blend(d: f32) -> CascadeBlend {
    let s0 = camera.cascade_splits.x;
    let s1 = camera.cascade_splits.y;
    let s2 = camera.cascade_splits.z;

    let w0 = max(1.0, 0.1 * s0);
    let w1 = max(1.0, 0.1 * (s1 - s0));
    let w2 = max(1.0, 0.1 * (s2 - s1));

    if d < s0 {
        let t = smoothstep(s0 - w0, s0, d);
        return CascadeBlend(0, 1, t);
    } else if d < s1 {
        let t = smoothstep(s1 - w1, s1, d);
        return CascadeBlend(1, 2, t);
    } else if d < s2 {
        let t = smoothstep(s2 - w2, s2, d);
        return CascadeBlend(2, 3, t);
    }
    return CascadeBlend(3, 3, 0.0);
}

fn get_light_view_proj(cascade: i32) -> mat4x4<f32> {
    if cascade == 0 {
        return camera.light_view_proj;
    } else if cascade == 1 {
        return camera.light_view_proj_cascade1;
    } else if cascade == 2 {
        return camera.light_view_proj_cascade2;
    }
    return camera.light_view_proj_cascade3;
}

fn shadow_pcf_cascade(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>, cascade: i32) -> f32 {
    let light_vp = get_light_view_proj(cascade);
    
    let NdotL = max(dot(N, L), 0.0);
    let slope_bias = 0.0002 * sqrt(1.0 - NdotL * NdotL) / max(NdotL, 0.001);
    let normal_offset = 0.004 * (1.0 - NdotL);
    let offset_pos = world_pos + N * normal_offset;
    
    let light_clip = light_vp * vec4<f32>(offset_pos, 1.0);
    
    if light_clip.w <= 0.0 {
        return 1.0;
    }
    
    let ndc = light_clip.xyz / light_clip.w;
    let uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
    
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return 1.0;
    }
    
    let depth = ndc.z - slope_bias;
    
    let texel_size = 1.2 / SHADOW_MAP_SIZE;
    var shadow_sum = 0.0;
    
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[0] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[1] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[2] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[3] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[4] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[5] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[6] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[7] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[8] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[9] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[10] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[11] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[12] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[13] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[14] * texel_size, cascade, depth);
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + POISSON_DISK[15] * texel_size, cascade, depth);
    
    return shadow_sum / 16.0;
}

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
    let clip_pos = camera.view_proj * vec4<f32>(position, 1.0);
    out.clip_position = clip_pos;
    out.view_depth = distance(position, camera.position.xyz);
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

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let clip = vec4<f32>(p[vid], 1.0, 1.0);

    let view_h = camera.proj_inv * clip;
    let view_dir = normalize(view_h.xyz / view_h.w);
    let world_dir = normalize((camera.view_inv * vec4<f32>(view_dir, 0.0)).xyz);

    var o: SkyOut;
    o.pos = vec4<f32>(p[vid], 1.0, 1.0);
    o.dir = world_dir;
    return o;
}

fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
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
    let L = normalize(-camera.light_dir.xyz);

    let cb = cascade_blend(in.view_depth);
    let s0 = shadow_pcf_cascade(in.world_position, N, L, cb.c0);
    let s1 = shadow_pcf_cascade(in.world_position, N, L, cb.c1);
    let shadow = s0 * (1.0 - cb.t) + s1 * cb.t;
    
    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);
    
    var Lo = vec3<f32>(0.0);

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

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let uv = dir_to_equirect_uv(in.dir);
    var col = textureSample(env_map, env_sampler, uv).rgb * camera.env_intensity.rgb;
    col = col / (col + vec3<f32>(1.0));
    return vec4<f32>(col, 1.0);
}
