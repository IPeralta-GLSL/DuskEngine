struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    position: vec4<f32>,
};

struct Material {
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    alpha_cutoff_flags: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

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
    return out;
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
    
    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);
    
    var Lo = vec3<f32>(0.0);

    let light_color = vec3<f32>(300.0, 300.0, 300.0);

    let lp0 = vec3<f32>(10.0, 10.0, 10.0);
    let lp1 = vec3<f32>(-10.0, 10.0, 10.0);
    let lp2 = vec3<f32>(10.0, 10.0, -10.0);
    let lp3 = vec3<f32>(-10.0, 10.0, -10.0);

    {
        let L = normalize(lp0 - in.world_position);
        let H = normalize(V + L);
        let distance = length(lp0 - in.world_position);
        let attenuation = 1.0 / (distance * distance);
        let radiance = light_color * attenuation;

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
        Lo = Lo + (kD * albedo / PI + specular) * radiance * NdotL;
    }
    {
        let L = normalize(lp1 - in.world_position);
        let H = normalize(V + L);
        let distance = length(lp1 - in.world_position);
        let attenuation = 1.0 / (distance * distance);
        let radiance = light_color * attenuation;

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
        Lo = Lo + (kD * albedo / PI + specular) * radiance * NdotL;
    }
    {
        let L = normalize(lp2 - in.world_position);
        let H = normalize(V + L);
        let distance = length(lp2 - in.world_position);
        let attenuation = 1.0 / (distance * distance);
        let radiance = light_color * attenuation;

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
        Lo = Lo + (kD * albedo / PI + specular) * radiance * NdotL;
    }
    {
        let L = normalize(lp3 - in.world_position);
        let H = normalize(V + L);
        let distance = length(lp3 - in.world_position);
        let attenuation = 1.0 / (distance * distance);
        let radiance = light_color * attenuation;

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
        Lo = Lo + (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    let ambient = vec3<f32>(0.03) * albedo;
    var color = ambient + Lo;
    
    color = color / (color + vec3<f32>(1.0));

    if material.alpha_cutoff_flags.y >= 1.5 {
        return vec4<f32>(color, alpha);
    }

    return vec4<f32>(color, 1.0);
}
