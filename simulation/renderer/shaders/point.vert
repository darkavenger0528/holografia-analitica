#version 330

in vec3 in_position;
in vec3 in_color;
in float in_size;

out vec3 v_color;

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(in_position, 1.0);
    gl_PointSize = in_size;

    v_color = in_color;
}