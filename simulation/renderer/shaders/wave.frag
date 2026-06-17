#version 330

in float v_phase;
in float v_density;

out vec4 fragColor;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(
        1.0,
        2.0/3.0,
        1.0/3.0,
        3.0
    );

    vec3 p =
        abs(fract(c.xxx + K.xyz) * 6.0 - K.www);

    return c.z *
        mix(
            K.xxx,
            clamp(p - K.xxx, 0.0, 1.0),
            c.y
        );
}

void main()
{
    vec2 p = gl_PointCoord - vec2(0.5);

    if(length(p) > 0.5)
        discard;

    vec3 rgb = hsv2rgb(
        vec3(
            v_phase,
            0.85,
            0.45 + 0.55 * v_density
        )
    );

    fragColor = vec4(rgb, 1.0);
}