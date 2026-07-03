#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec4 p3d_Color;

out vec3 v_local_pos;
out vec3 v_body_color;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_local_pos = p3d_Vertex.xyz;
    v_body_color = p3d_Color.rgb;
}
