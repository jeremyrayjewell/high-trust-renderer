#version 150

uniform sampler2D scene_color_tex;
uniform sampler2D front_depth_tex;
uniform sampler2D back_depth_tex;
uniform sampler2D front_normal_tex;
uniform sampler2D front_local_tex;
uniform vec4 absorption_color;
uniform float absorption_density;
uniform float transmission_gain;
uniform float scattering_strength;
uniform float cloudiness;
uniform float refraction_strength;
uniform float ior;
uniform float surface_opacity;
uniform float specular_strength;
uniform float fresnel_strength;
uniform float surface_reflection_strength;
uniform float thickness_scale;
uniform float view_mode;
uniform float debug_depth_max;
uniform float debug_thickness_max;
uniform vec4 texture_uv_scale;

in vec2 v_uv;

out vec4 fragColor;

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

void main() {
    vec2 sampleUv = clamp(v_uv * texture_uv_scale.xy, vec2(0.0), texture_uv_scale.xy);
    vec3 sceneColor = texture(scene_color_tex, sampleUv).rgb;
    float frontDepth = texture(front_depth_tex, sampleUv).r;
    float backDepth = texture(back_depth_tex, sampleUv).r;
    vec4 normalSample = texture(front_normal_tex, sampleUv);
    vec4 localSample = texture(front_local_tex, sampleUv);

    if (frontDepth <= 0.0001 || backDepth <= frontDepth) {
        fragColor = vec4(sceneColor, 1.0);
        return;
    }

    float thickness = max(backDepth - frontDepth, 0.0) * thickness_scale;
    vec3 normal = normalize(normalSample.rgb * 2.0 - 1.0);
    vec3 bodyColor = clamp(localSample.rgb, vec3(0.04), vec3(1.0));
    float localZ = localSample.a * 4.0 - 2.0;

    float bend = clamp((ior - 1.0) * refraction_strength * thickness * 2.4, 0.0, 0.24);
    vec2 refractOffset = normal.xy * bend;
    vec2 uv = clamp(sampleUv + refractOffset, vec2(0.001), texture_uv_scale.xy - vec2(0.001));
    vec3 refractedScene = texture(scene_color_tex, uv).rgb;

    vec3 mediumColor = mix(absorption_color.rgb, bodyColor, 0.78);
    vec3 absorptionCoeff = mix(vec3(1.18, 1.18, 1.18), vec3(0.08, 0.08, 0.08), clamp(mediumColor, 0.0, 1.0));
    vec3 transmittance = exp(-absorptionCoeff * thickness * absorption_density);
    float cloud = hash31(vec3(bodyColor.rg * 1.6, localZ * 0.32) + vec3(normal.xy * 0.3, -0.15));
    cloud += 0.18 * sin(localZ * 1.7 + dot(normal.xy, vec2(1.3, 1.1)));
    cloud = clamp(cloud * 0.5 + 0.5, 0.0, 1.0);

    float thinBoost = exp(-thickness * 1.2);
    float thickBoost = 1.0 - thinBoost;
    vec3 scattering = mediumColor * (0.64 + 0.92 * cloudiness) * thickBoost * (0.76 + 0.42 * cloud) * scattering_strength;

    vec3 lightDir = normalize(vec3(-0.35, -0.45, 0.82));
    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 10.0) * specular_strength;
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.2) * fresnel_strength;

    vec3 transmitted = refractedScene * transmittance * transmission_gain;
    vec3 edgeLift = mediumColor * thinBoost * 0.28;
    vec3 interiorLift = mediumColor * (0.24 + 0.32 * cloud) * (0.34 + 0.76 * thickBoost);
    vec3 reflection = mix(sceneColor, vec3(1.0), 0.35) * (fresnel * surface_reflection_strength);
    vec3 bodyFloor = mediumColor * (0.12 + 0.14 * thickBoost);
    vec3 composite = transmitted + scattering + edgeLift + interiorLift + bodyFloor + reflection + vec3(spec);
    composite = mix(sceneColor, composite, surface_opacity);
    composite = max(composite, sceneColor * 0.48);

    if (view_mode < 0.5) {
        fragColor = vec4(composite, 1.0);
        return;
    }
    if (view_mode < 1.5) {
        float v = clamp(frontDepth / debug_depth_max, 0.0, 1.0);
        fragColor = vec4(v, v, v, 1.0);
        return;
    }
    if (view_mode < 2.5) {
        float v = clamp(backDepth / debug_depth_max, 0.0, 1.0);
        fragColor = vec4(v, v, v, 1.0);
        return;
    }
    if (view_mode < 3.5) {
        float v = clamp(thickness / debug_thickness_max, 0.0, 1.0);
        fragColor = vec4(v, v, v, 1.0);
        return;
    }
    if (view_mode < 4.5) {
        fragColor = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }
    if (view_mode < 5.5) {
        fragColor = vec4(refractOffset.x * 8.0 + 0.5, refractOffset.y * 8.0 + 0.5, clamp(length(refractOffset) * 12.0, 0.0, 1.0), 1.0);
        return;
    }
    fragColor = vec4(transmittance, 1.0);
}
