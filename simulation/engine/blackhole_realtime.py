import argparse
import math
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../models/blackhole"))
from blackhole_core import shader_params

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_mass;
uniform float u_spin;
uniform float u_event_horizon;
uniform float u_photon_sphere;
uniform float u_isco;
uniform float u_disk_inner;
uniform float u_disk_outer;
uniform float u_camera_distance;
uniform float u_tilt;
const float PI = 3.141592653589793;

mat3 rot_x(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c
    );
}

mat3 rot_y(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    );
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p *= 2.03;
        a *= 0.5;
    }
    return v;
}

vec3 starfield(vec3 rd) {
    vec2 sph = vec2(atan(rd.z, rd.x), asin(clamp(rd.y, -1.0, 1.0)));
    vec2 uv = sph * vec2(0.1591, 0.3183);
    float scale = 220.0;
    vec2 gv = uv * scale;
    vec2 cell = floor(gv);
    vec2 local = fract(gv) - 0.5;          // posición dentro de la celda, centrada
    float h = hash21(cell);
    float present = smoothstep(0.9975, 1.0, h);  // ¿esta celda tiene estrella?
    // Jitter del centro de la estrella dentro de la celda + tamaño puntual
    vec2 jitter = (vec2(hash21(cell + 1.7), hash21(cell + 8.3)) - 0.5) * 0.7;
    float d = length(local - jitter);
    float starShape = exp(-d * d * 90.0);        // núcleo puntual suave
    float star = present * starShape;
    vec3 tint = mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.92, 0.8), hash21(cell + 4.7));
    float neb = fbm(uv * 6.0 + vec2(0.0, u_time * 0.01));
    vec3 bg = mix(vec3(0.005, 0.006, 0.012), vec3(0.015, 0.012, 0.025), neb * 0.65);
    return bg + star * tint * (0.6 + 1.2 * hash21(cell + 2.3));
}

float bend_strength(float b) {
    float farTerm = (0.25 * u_mass) / (b + 0.35);
    float ringTerm = 0.95 * exp(-pow((b - u_photon_sphere) * 1.6, 2.0));
    return farTerm + ringTerm;
}

vec3 lens_ray(vec3 ro, vec3 rd) {
    vec3 p = ro;
    vec3 d = normalize(rd);
    for (int i = 0; i < 96; i++) {
        float r = length(p);
        if (r < u_event_horizon * 0.92) {
            return vec3(1e9);
        }
        vec3 toCenter = -p / max(r, 1e-4);
        float impact = length(cross(p, d));
        float bend = bend_strength(impact) * 0.012;
        vec3 dragAxis = vec3(0.0, 1.0, 0.0);
        vec3 frameDrag = normalize(cross(dragAxis, p + vec3(1e-5))) * (u_spin * 0.0015 / (r + 0.5));
        d = normalize(d + toCenter * bend + frameDrag);
        // Paso adaptativo: más fino cerca del horizonte para suavizar
        // el borde de la sombra (evita el aliasing "dentado").
        float step = mix(0.035, 0.10, smoothstep(u_event_horizon * 0.92, u_event_horizon * 2.2, r));
        p += d * step;
    }
    return normalize(d);
}

bool disk_intersection(vec3 ro, vec3 rd, out vec3 hit) {
    vec3 p = ro;
    vec3 d = normalize(rd);
    float prevY = p.y;
    for (int i = 0; i < 180; i++) {
        float r = length(p);
        if (r < u_event_horizon * 0.92) return false;
        float impact = length(cross(p, d));
        vec3 toCenter = -p / max(r, 1e-4);
        float bend = bend_strength(impact) * 0.010;
        vec3 dragAxis = vec3(0.0, 1.0, 0.0);
        vec3 frameDrag = normalize(cross(dragAxis, p + vec3(1e-5))) * (u_spin * 0.0013 / (r + 0.5));
        d = normalize(d + toCenter * bend + frameDrag);
        float step = mix(0.028, 0.08, smoothstep(u_event_horizon * 0.92, u_event_horizon * 2.2, r));
        p += d * step;
        if ((prevY > 0.0 && p.y <= 0.0) || (prevY < 0.0 && p.y >= 0.0)) {
            vec3 h = p;
            float rr = length(h.xz);
            if (rr > u_disk_inner && rr < u_disk_outer) {                hit = h;
                return true;
            }
        }
        prevY = p.y;
    }
    return false;
}

vec3 disk_color(vec3 hit, vec3 rd) {
    float r = length(hit.xz);
    float a = atan(hit.z, hit.x);
    float orbit = 1.9 / sqrt(r + 0.15);
    vec2 tangent = normalize(vec2(-sin(a), cos(a)));
    vec2 view2 = normalize(rd.xz + vec2(1e-6));
    float toward = dot(tangent, -view2);
    float beta = clamp(toward * orbit * 0.55, -0.92, 0.92);
    float gamma = 1.0 / sqrt(1.0 - beta * beta);
    float doppler = 1.0 / (gamma * (1.0 - beta));
    float beaming = clamp(pow(doppler, 2.4), 0.35, 5.0);
    float innerHeat = pow(1.0 / max(r - u_event_horizon, 0.18), 0.82);
    float band = exp(-pow((r - (u_isco * 1.15)) / 0.65, 2.0));
    float ring = exp(-pow((r - u_photon_sphere) / 0.18, 2.0));
    float swirl = fbm(vec2(a * 3.5 + u_time * 0.6, r * 1.8 - u_time * 0.35));
    float streaks = 0.5 + 0.5 * sin(a * 22.0 - u_time * (3.5 + 0.4 / max(r, 0.2)) + swirl * 4.0);
    float textureMask = mix(0.7, 1.35, swirl) * mix(0.8, 1.25, streaks);
    // Paleta blanco-azulada (plasma a muy alta temperatura, estilo
    // "Gargantua"): casi monocromo, con un leve viraje a perla/violeta
    // en los bordes exteriores en vez de naranja.
    vec3 hot  = vec3(1.55, 1.50, 1.45);   // núcleo: blanco puro, liger. cálido
    vec3 warm = vec3(1.35, 1.28, 1.30);   // banda intermedia: blanco-perla
    vec3 cool = vec3(0.85, 0.88, 1.05);   // borde exterior: blanco-azulado
    float radialMix = clamp((r - u_disk_inner) / (u_disk_outer - u_disk_inner), 0.0, 1.0);
    vec3 base = mix(hot, cool, radialMix * 0.55);
    base = mix(base, warm, band * 0.45);
    float intensity = (0.40 * innerHeat + 1.35 * band + 1.15 * ring) * textureMask * beaming;
    intensity *= exp(-0.06 * (r - u_disk_inner));
    // Exposición elevada: sobre-expone el disco para el look "quemado
    // de blanco" característico, en vez de bandas de color separadas.
    intensity *= 1.8;
    vec3 col = base * intensity;
    col += vec3(1.3, 1.28, 1.25) * ring * 0.7;
    return col;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
    vec3 ro = vec3(0.0, 0.55, u_camera_distance);
    vec3 rd = normalize(vec3(uv.x, uv.y, -1.55));
    rd = rot_x(u_tilt) * rot_y(0.08 * sin(u_time * 0.15)) * rd;
    vec3 bent = lens_ray(ro, rd);
    if (bent.x > 1e8) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec3 hit;
    vec3 color = vec3(0.0);
    if (disk_intersection(ro, rd, hit)) {
        color += disk_color(hit, bent);
    }
    color += starfield(bent);
    float center = length(uv);
    float shadow = smoothstep(0.20, 0.165, center);
    float glow = exp(-pow((center - 0.195) / 0.032, 2.0)) * 0.6;
    color = mix(color, vec3(0.0), shadow);
    color += vec3(1.25, 1.22, 1.30) * glow;
    // Exposición global elevada (look "sobre-expuesto" tipo Gargantua)
    color *= 1.35;
    color = color / (1.0 + color);
    color = pow(color, vec3(0.55));
    // Dithering para evitar banding visible en degradados suaves
    // (horizonte, disco) en displays/GPUs de menor precisión de color.
    float dither = (hash21(gl_FragCoord.xy + fract(u_time) * 37.0) - 0.5) / 255.0;
    color += vec3(dither);
    f_color = vec4(color, 1.0);
}
"""

def build_gl_engine(args):
    import moderngl
    if getattr(args, "display", 0) == 1:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{args.width + 10},0"
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    pygame.display.set_mode((args.width, args.height), flags)
    
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    
    # Quad optimizado de 4 vértices para TRIANGLE_STRIP usando NumPy
    quad_vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0
    ], dtype='float32')
    
    quad = ctx.buffer(quad_vertices)
    vao = ctx.simple_vertex_array(prog, quad, "in_pos")
    
    params = shader_params(mass=1.0, spin=0.72, disk_outer=6.2, camera_distance=11.0, tilt_deg=4.0)
    prog["u_resolution"].value = (args.width, args.height)
    prog["u_mass"].value = params["mass"]
    prog["u_spin"].value = params["spin"]
    prog["u_event_horizon"].value = params["event_horizon"]
    prog["u_photon_sphere"].value = params["photon_sphere"]
    prog["u_isco"].value = params["isco"]
    prog["u_disk_inner"].value = params["disk_inner"]
    prog["u_disk_outer"].value = params["disk_outer"]
    prog["u_camera_distance"].value = params["camera_distance"]
    prog["u_tilt"].value = math.radians(params["tilt_deg"])
    
    return pygame, ctx, prog, vao, moderngl

def run_gl(args):
    pygame, ctx, prog, vao, moderngl = build_gl_engine(args)
    clock = pygame.time.Clock()
    t = 0.0
    paused = False
    speed = 1.0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if not paused:
            t += dt * speed
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed *= 1.15
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed /= 1.15
                elif e.key == pygame.K_r:
                    t = 0.0
                    
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        prog["u_time"].value = t
        vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()
    pygame.quit()

def run_fallback(args, exc):
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height), pygame.FULLSCREEN if args.fullscreen else 0)
    font = pygame.font.SysFont("Arial", 22)
    small = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(30)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
        screen.fill((5, 5, 8))
        y = 80
        for line in [
            "No se pudo iniciar ModernGL para la version 2 del agujero negro.",
            "Instala: pip install moderngl",
            "Detalle:",
            str(exc),
            "",
            "Esta version requiere contexto OpenGL 3.3+ y soporte de shaders.",
            "ESC o Q para salir.",
        ]:
            surf = (font if y < 170 else small).render(line, True, (220, 220, 220))
            screen.blit(surf, (60, y))
            y += 34 if y < 170 else 24
        pygame.display.flip()
    pygame.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display", type=int, default=0)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()
    try:
        run_gl(args)
    except Exception as exc:
        run_fallback(args, exc)

if __name__ == "__main__":
    main()