#version 150

in float v_linear_depth;

out vec4 fragColor;

void main() {
    fragColor = vec4(v_linear_depth, 0.0, 0.0, 1.0);
}
