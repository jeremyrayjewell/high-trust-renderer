#version 150

in vec3 v_normal;

out vec4 fragColor;

void main() {
    fragColor = vec4(normalize(v_normal) * 0.5 + 0.5, 1.0);
}
