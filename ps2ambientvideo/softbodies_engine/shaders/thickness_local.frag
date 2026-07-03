#version 150

in vec3 v_local_pos;
in vec3 v_body_color;

out vec4 fragColor;

void main() {
    fragColor = vec4(clamp(v_body_color, vec3(0.0), vec3(1.0)), clamp(v_local_pos.z * 0.25 + 0.5, 0.0, 1.0));
}
