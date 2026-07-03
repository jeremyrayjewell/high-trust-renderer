#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;

in vec4 p3d_Vertex;

out float v_linear_depth;

void main() {
    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    v_linear_depth = -view_pos.z;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
