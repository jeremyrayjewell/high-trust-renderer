#version 150

uniform vec4 base_color;

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec4 v_color;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 L1 = normalize(vec3(-0.4, -0.5, 1.0));
    vec3 L2 = normalize(vec3(0.3, 0.8, 0.6));
    vec3 V = normalize(vec3(0.0, 0.0, 1.0));
    float key = max(dot(N, L1), 0.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);
    float fill = max(dot(N, L2), 0.0);
    vec3 H = normalize(L1 + V);
    float spec = pow(max(dot(N, H), 0.0), 12.0);
    vec3 albedo = base_color.rgb * v_color.rgb;
    vec3 color = albedo * (0.42 + 0.50 * key + 0.18 * fill);
    color += albedo * rim * 0.18;
    color += vec3(1.0) * spec * 0.28;
    fragColor = vec4(color, 1.0);
}
