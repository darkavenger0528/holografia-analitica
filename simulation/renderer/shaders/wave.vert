#version 330

in vec3 in_position;
in float in_phase;
in float in_density;

out float v_phase;
out float v_density;

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(in_position, 1.0);

    gl_PointSize = 6.0;

    v_phase = in_phase;
    v_density = in_density;
}