#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec3 v_local_pos;
out vec4 v_seed_color;
out vec2 v_uv;

void main() {
    vec4 world_pos = p3d_ModelMatrix * p3d_Vertex;
    v_world_pos = world_pos.xyz;
    v_world_normal = normalize(p3d_NormalMatrix * p3d_Normal);
    v_local_pos = p3d_Vertex.xyz;
    v_seed_color = p3d_Color;
    v_uv = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}

