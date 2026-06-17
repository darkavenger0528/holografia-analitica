#version 330

in vec3 v_color;

out vec4 fragColor;

void main()
{
    vec2 c = gl_PointCoord - vec2(0.5);

    if (length(c) > 0.5)
        discard;

    fragColor = vec4(v_color, 1.0);
}