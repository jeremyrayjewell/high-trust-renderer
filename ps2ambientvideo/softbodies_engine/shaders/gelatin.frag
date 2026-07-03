#version 150

uniform vec4 base_color;
uniform float opacity;
uniform float transmission_strength;
uniform float cloudiness;
uniform float fresnel_strength;
uniform float specular_strength;
uniform float emissive_strength;
uniform float absorption_strength;
uniform float rear_face_strength;
uniform float thickness_absorption;
uniform float minimum_body_light;
uniform float exposure;
uniform float inner_layer_opacity;
uniform float layer_scale;
uniform float time;
uniform float audio_high;
uniform float audio_rms;
uniform vec3 camera_world_position;
uniform vec3 key_light_position;
uniform vec3 fill_light_position;
uniform vec3 back_light_position;
uniform vec3 key_light_color;
uniform vec3 fill_light_color;
uniform vec3 back_light_color;

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec3 v_local_pos;
in vec4 v_seed_color;
in vec2 v_uv;

out vec4 fragColor;

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(camera_world_position - v_world_pos);
    vec3 Lk = normalize(key_light_position - v_world_pos);
    vec3 Lf = normalize(fill_light_position - v_world_pos);
    vec3 Lb = normalize(back_light_position - v_world_pos);

    bool frontFacing = gl_FrontFacing;
    float NdV = clamp(dot(N, V), 0.0, 1.0);
    float backView = 1.0 - NdV;
    float radial = clamp(length(v_local_pos), 0.0, 1.6);
    float normalizedRadius = clamp(radial / 1.25, 0.0, 1.0);
    float centerMass = 1.0 - normalizedRadius;
    float edgeThin = pow(normalizedRadius, 1.15);
    float cloudyMask = smoothstep(0.08, 0.92, centerMass);
    float thicknessFactor = mix(1.0, 1.0 + thickness_absorption, cloudyMask);

    float diffuseKey = max(dot(N, Lk), 0.0);
    float diffuseFill = max(dot(N, Lf), 0.0);
    float backFacing = max(dot(-N, Lb), 0.0);
    float throughCamera = clamp(dot(Lb, V) * 0.5 + 0.5, 0.0, 1.0);
    float transmissionWindow = edgeThin * (1.0 - 0.45 * cloudyMask);
    float transmission = transmission_strength * pow(backFacing, 1.35) * (0.08 + 0.22 * transmissionWindow) * (0.62 + 0.20 * throughCamera);
    float fresnel = pow(backView, 4.2) * fresnel_strength * (0.22 + 0.28 * edgeThin);

    vec3 Hk = normalize(Lk + V);
    vec3 Hf = normalize(Lf + V);
    float spec = specular_strength * (
        pow(max(dot(N, Hk), 0.0), 4.0) +
        0.42 * pow(max(dot(N, Hf), 0.0), 3.4)
    );

    float cloudNoise = hash31(v_local_pos * 0.65 + vec3(0.1, time * 0.04, -0.12));
    float driftNoise = hash31(v_local_pos * 1.15 + vec3(time * 0.05, -time * 0.03, 0.07));
    float internalVariation = mix(cloudNoise, driftNoise, 0.38);
    internalVariation += 0.03 * sin((v_local_pos.x * 0.8 + v_local_pos.y * 0.6 + v_local_pos.z * 0.5) * 4.0 + time * (0.4 + audio_high * 0.5));
    float cloudyCore = clamp(cloudiness * (0.58 + 0.42 * internalVariation) * cloudyMask, 0.0, 1.0);

    float isInner = 1.0 - step(0.985, layer_scale);
    vec3 denseTint = mix(base_color.rgb * 0.98, min(base_color.rgb * 1.30, vec3(1.0)), cloudyMask);
    vec3 transmittedTint = mix(base_color.rgb * 1.02, min(base_color.rgb * 1.20, vec3(1.0)), transmissionWindow);

    vec3 lightField = vec3(minimum_body_light);
    lightField += diffuseKey * key_light_color * 0.38;
    lightField += diffuseFill * fill_light_color * 0.34;
    lightField += transmission * back_light_color * 0.42;

    vec3 bodyColor = denseTint * lightField;
    bodyColor = mix(bodyColor, denseTint * (minimum_body_light + 0.26), cloudyCore * 0.54);
    bodyColor += cloudyCore * denseTint * 0.58;
    bodyColor += transmittedTint * transmission * (0.16 + 0.08 * transmissionWindow);
    bodyColor += spec * vec3(0.98, 0.99, 1.0) * (1.0 - isInner * 0.92);
    bodyColor += fresnel * transmittedTint * 0.03 * (1.0 - isInner * 0.85);
    bodyColor += emissive_strength * (0.24 + 0.36 * audio_high) * denseTint;

    bodyColor *= mix(1.0, 1.0 - absorption_strength * 0.22, 1.0 - edgeThin);
    bodyColor *= mix(1.0, 0.92, isInner);

    float alpha = opacity;
    alpha += cloudyCore * 0.28;
    alpha += centerMass * 0.20 * thicknessFactor;
    alpha += (1.0 - edgeThin) * 0.10;
    alpha -= transmissionWindow * 0.03;
    alpha *= mix(1.0, inner_layer_opacity, isInner);

    if (!frontFacing) {
        bodyColor *= rear_face_strength;
        alpha *= rear_face_strength;
    }

    bodyColor = max(bodyColor, denseTint * minimum_body_light);
    bodyColor *= exposure;
    alpha = clamp(alpha, 0.72, 0.995);

    fragColor = vec4(bodyColor, alpha);
}
