"""
simulation/engine/universe_realtime.py

Renderizado en tiempo real de la expansión del universo (FLRW)
usando pygame. Diseñado para proyección Pepper's Ghost con
vidrio horizontal a 45°.

Versión 2 — mejoras visuales:
  - Cuatro categorías de objetos con color e iluminación propios:
      · Soles (estrellas)  → grandes, brillantes, con halo (glow)
      · Planetas           → medianos, paleta variada
      · Asteroides         → diminutos, grises/marrones, en gran cantidad
      · Galaxias           → discos elípticos difusos de fondo, violeta/magenta
  - Supersampling (renderizado a mayor resolución y reescalado)
    para suavizar bordes y mejorar la calidad visual.

Controles:
  ESPACIO  → pausar / reanudar
  R        → reiniciar desde el Big Bang
  + / -    → acelerar / ralentizar
  F        → fullscreen
  ESC / Q  → salir

Uso:
  python universe_realtime.py              # pantalla principal
  python universe_realtime.py --display 1  # pantalla HDMI (Android)
  python universe_realtime.py --display 1 --fullscreen
"""

import sys
import os
import argparse
import math
import numpy as np
import pygame
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(__file__))
from pepper_layout import blit_to_pepper

# ── Parámetros cosmológicos (Planck 2018) ──────────────────
H0_GYR  = 67.4 * 1.022e-3
OMEGA_M = 0.315
OMEGA_R = 9.24e-5
OMEGA_L = 0.685
T_HOY   = 13.8

# ── Parámetros de simulación ────────────────────────────────
N_SOLES      = 60     # estrellas: grandes, brillantes
N_PLANETAS   = 260    # cuerpos medianos de colores variados
N_ASTEROIDES = 1100   # objetos diminutos, "polvo" cósmico
N_GALAXIAS   = 18     # estructuras grandes y difusas, de fondo
T_MIN        = 0.5
T_MAX        = 30.0
DURACION_SEG = 60.0
FPS_TARGET   = 60

# Supersampling: se renderiza a SUPERSAMPLE * (W,H) y se reescala
# hacia abajo para suavizar bordes (antialiasing barato).
SUPERSAMPLE = 1.75

COLOR_BG         = (0, 0, 0)
COLOR_OBSERVADOR = (255, 255, 255)
COLOR_HOY_LINE   = (255, 200, 50)

# Paletas de color por tipo de objeto
PALETA_SOLES = [
    (255, 244, 200),
    (255, 210, 130),
    (255, 140,  90),
    (210, 225, 255),
    (255, 255, 255),
]
PESOS_SOLES = [0.35, 0.25, 0.20, 0.10, 0.10]

PALETA_PLANETAS = [
    (100, 150, 255),
    (130, 200, 230),
    (200, 130,  90),
    (230, 180, 120),
    (120, 220, 140),
    (220,  90,  90),
    (180, 140, 220),
]

PALETA_ASTEROIDES = [
    (140, 130, 115),
    (120, 105,  90),
    (160, 150, 135),
    (105, 100,  95),
]

PALETA_GALAXIAS = [
    (170, 110, 230),   # violeta
    (220, 110, 200),   # magenta
    (130, 100, 220),   # azul-violeta
    (230, 160, 255),   # lila claro
    (255, 140, 200),   # rosado
]


# ═══════════════════════════════════════════════════════════════════════════
# Precálculo de a(t)
# ═══════════════════════════════════════════════════════════════════════════

def precompute_scale_factor(n=1000):
    def rhs(t, y):
        a = max(y[0], 1e-10)
        H = math.sqrt(OMEGA_R/a**4 + OMEGA_M/a**3 + OMEGA_L)
        return [H0_GYR * a * H]

    t_eval = np.linspace(T_MIN, T_MAX, n)
    sol = solve_ivp(rhs, (T_MIN, T_MAX), [1e-3], t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-11)
    idx = np.argmin(np.abs(sol.t - T_HOY))
    a_norm = sol.y[0] / sol.y[0, idx]
    return sol.t, a_norm


# ═══════════════════════════════════════════════════════════════════════════
# Generación de objetos (soles, planetas, asteroides)
# ═══════════════════════════════════════════════════════════════════════════

def _posiciones_esfera(n, rng, r_min=0.05, r_max=1.0):
    """N posiciones comóviles uniformemente distribuidas en una esfera."""
    theta = np.arccos(rng.uniform(-1, 1, n))
    phi   = rng.uniform(0, 2 * math.pi, n)
    r     = (r_min**3 + rng.uniform(0, 1, n) * (r_max**3 - r_min**3)) ** (1/3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1), r


def generar_universo(seed=42):
    """
    Genera las tres poblaciones de objetos.

    Retorna un dict con, para cada categoría:
        pos     : (N,3) posiciones comóviles
        colores : (N,3) uint8
        tamanos : (N,)  tamaño base en píxeles
        es_sol  : bool — si lleva halo de brillo
    """
    rng = np.random.default_rng(seed)
    datos = {}

    # ── Soles ────────────────────────────────────────────────────────────
    pos, r = _posiciones_esfera(N_SOLES, rng, r_min=0.05, r_max=1.0)
    idx_color = rng.choice(len(PALETA_SOLES), size=N_SOLES, p=PESOS_SOLES)
    colores = np.array([PALETA_SOLES[i] for i in idx_color], dtype=np.uint8)
    tamanos = rng.uniform(4.0, 8.0, N_SOLES)
    datos["soles"] = dict(pos=pos, r=r, colores=colores,
                          tamanos=tamanos, es_sol=True)

    # ── Planetas ─────────────────────────────────────────────────────────
    pos, r = _posiciones_esfera(N_PLANETAS, rng, r_min=0.03, r_max=1.0)
    idx_color = rng.integers(0, len(PALETA_PLANETAS), N_PLANETAS)
    colores = np.array([PALETA_PLANETAS[i] for i in idx_color], dtype=np.uint8)
    tamanos = rng.uniform(1.5, 3.5, N_PLANETAS)
    datos["planetas"] = dict(pos=pos, r=r, colores=colores,
                             tamanos=tamanos, es_sol=False)

    # ── Asteroides ───────────────────────────────────────────────────────
    pos, r = _posiciones_esfera(N_ASTEROIDES, rng, r_min=0.02, r_max=1.0)
    idx_color = rng.integers(0, len(PALETA_ASTEROIDES), N_ASTEROIDES)
    colores = np.array([PALETA_ASTEROIDES[i] for i in idx_color], dtype=np.uint8)
    tamanos = rng.uniform(0.6, 1.4, N_ASTEROIDES)
    datos["asteroides"] = dict(pos=pos, r=r, colores=colores,
                               tamanos=tamanos, es_sol=False)

    # ── Galaxias ─────────────────────────────────────────────────────────
    # Estructuras grandes y difusas (discos elípticos), preferentemente
    # más lejanas (r_min alto) para dar sensación de profundidad/fondo.
    pos, r = _posiciones_esfera(N_GALAXIAS, rng, r_min=0.35, r_max=1.0)
    idx_color = rng.integers(0, len(PALETA_GALAXIAS), N_GALAXIAS)
    colores = np.array([PALETA_GALAXIAS[i] for i in idx_color], dtype=np.uint8)
    ancho  = rng.uniform(16.0, 28.0, N_GALAXIAS)
    alto   = ancho * rng.uniform(0.25, 0.5, N_GALAXIAS)
    angulo = rng.uniform(0, 2 * math.pi, N_GALAXIAS)
    datos["galaxias"] = dict(pos=pos, r=r, colores=colores,
                             ancho=ancho, alto=alto, angulo=angulo,
                             es_sol=False)

    return datos


# ═══════════════════════════════════════════════════════════════════════════
# Proyección 3D → 2D (perspectiva simple)
# ═══════════════════════════════════════════════════════════════════════════

def proyectar(pos3d, a, angulo_rot, cx, cy, escala):
    pos = pos3d * a
    cos_a, sin_a = math.cos(angulo_rot), math.sin(angulo_rot)
    x =  pos[:, 0] * cos_a + pos[:, 2] * sin_a
    z = -pos[:, 0] * sin_a + pos[:, 2] * cos_a
    y =  pos[:, 1]

    d_foco = 2.5
    perspectiva = d_foco / (z + d_foco + 1e-5)
    px = (cx + x * escala * perspectiva)
    py = (cy - y * escala * perspectiva)
    return px, py, perspectiva, z


# ═══════════════════════════════════════════════════════════════════════════
# Halo / glow para soles
# ═══════════════════════════════════════════════════════════════════════════

def crear_glow_template(size=40):
    """Superficie con degradado radial horneado directamente en RGB
    (premultiplicado), pensada para blitear con BLEND_ADD sobre
    superficies sin canal alpha."""
    surf = pygame.Surface((size, size))
    arr = pygame.surfarray.pixels3d(surf)
    cx, cy = size / 2, size / 2
    for x in range(size):
        for y in range(size):
            d = math.hypot(x - cx, y - cy) / (size / 2)
            grad = max(0.0, 1.0 - d) ** 2
            arr[x, y, :] = int(grad * 255)
    del arr
    return surf


def crear_glow_eliptico(w=24, h=10):
    """Igual que crear_glow_template pero elíptico, para representar
    el disco difuso de una galaxia."""
    w, h = max(2, w), max(2, h)
    surf = pygame.Surface((w, h))
    arr = pygame.surfarray.pixels3d(surf)
    cx, cy = w / 2, h / 2
    for x in range(w):
        for y in range(h):
            dx = (x - cx) / (w / 2)
            dy = (y - cy) / (h / 2)
            d = math.hypot(dx, dy)
            grad = max(0.0, 1.0 - d) ** 1.5
            arr[x, y, :] = int(grad * 255)
    del arr
    return surf


# ═══════════════════════════════════════════════════════════════════════════
# Renderer principal
# ═══════════════════════════════════════════════════════════════════════════

class UniverseRenderer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.RW = int(W * SUPERSAMPLE)
        self.RH = int(H * SUPERSAMPLE)

        print("Precalculando a(t)...")
        self.t_arr, self.a_arr = precompute_scale_factor()

        print("Generando universo (soles, planetas, asteroides, galaxias)...")
        self.universo = generar_universo()

        self.t_sim      = T_MIN
        self.pausado    = False
        self.velocidad  = 1.0
        self.angulo_rot = 0.0

        # Superficie de render interno (mayor resolución)
        self.render_surf = pygame.Surface((self.RW, self.RH))
        # Superficie final del tamaño de pantalla
        self.model_surf  = pygame.Surface((self.W, self.H))

        # Glow precoloreado por cada sol
        print("Preparando halos de brillo...")
        glow_base = crear_glow_template(40)
        soles = self.universo["soles"]
        self.glows = []
        for color in soles["colores"]:
            g = glow_base.copy()
            g.fill((int(color[0]), int(color[1]), int(color[2])),
                  special_flags=pygame.BLEND_RGB_MULT)
            self.glows.append(g)

        # Glow elíptico precoloreado y rotado por cada galaxia
        print("Preparando discos de galaxias...")
        galaxias = self.universo["galaxias"]
        self.glows_gal = []
        for i in range(N_GALAXIAS):
            w_base = int(galaxias["ancho"][i])
            h_base = max(2, int(galaxias["alto"][i]))
            color = galaxias["colores"][i]
            g = crear_glow_eliptico(w_base, h_base)
            g.fill((int(color[0]), int(color[1]), int(color[2])),
                  special_flags=pygame.BLEND_RGB_MULT)
            ang_deg = math.degrees(galaxias["angulo"][i])
            g = pygame.transform.rotate(g, ang_deg)
            self.glows_gal.append(g)

    def a_en(self, t):
        return float(np.interp(t, self.t_arr, self.a_arr))

    def avanzar(self, dt_real):
        if self.pausado:
            return
        rango = T_MAX - T_MIN
        self.t_sim += (rango / DURACION_SEG) * dt_real * self.velocidad
        self.angulo_rot += dt_real * 0.25 * self.velocidad
        if self.t_sim >= T_MAX:
            self.t_sim = T_MIN

    # ── Dibujo de una categoría de objetos ─────────────────────────────────
    def _dibujar_categoria(self, surf, datos, a, cx, cy, escala, glows=None):
        px, py, persp, z = proyectar(datos["pos"], a, self.angulo_rot,
                                     cx, cy, escala)
        orden = np.argsort(z)

        W, H = surf.get_size()
        brillo_global = min(1.0, a * 0.8 + 0.25)

        for i in orden:
            x, y = px[i], py[i]
            if not (-20 <= x < W + 20 and -20 <= y < H + 20):
                continue
            sz = max(1, int(datos["tamanos"][i] * persp[i] * SUPERSAMPLE))
            color = datos["colores"][i]
            color_b = tuple(int(c * brillo_global) for c in color)

            if glows is not None:
                # Halo aditivo
                glow_sz = max(2, sz * 6)
                glow_scaled = pygame.transform.smoothscale(
                    glows[i], (glow_sz, glow_sz))
                surf.blit(glow_scaled, (int(x - glow_sz/2), int(y - glow_sz/2)),
                         special_flags=pygame.BLEND_ADD)
                # Núcleo brillante
                pygame.draw.circle(surf, (255, 255, 255),
                                  (int(x), int(y)), max(1, sz // 2))
            else:
                pygame.draw.circle(surf, color_b, (int(x), int(y)), sz)

    # ── Dibujo de galaxias (discos elípticos difusos) ───────────────────────
    def _dibujar_galaxias(self, surf, a, cx, cy, escala):
        datos = self.universo["galaxias"]
        px, py, persp, z = proyectar(datos["pos"], a, self.angulo_rot,
                                     cx, cy, escala)
        orden = np.argsort(z)
        W, H = surf.get_size()

        for i in orden:
            x, y = px[i], py[i]
            if not (-60 <= x < W + 60 and -60 <= y < H + 60):
                continue
            factor = max(0.05, persp[i] * SUPERSAMPLE)
            g = self.glows_gal[i]
            gw, gh = g.get_size()
            sw = max(2, int(gw * factor))
            sh = max(2, int(gh * factor))
            g_scaled = pygame.transform.smoothscale(g, (sw, sh))
            surf.blit(g_scaled, (int(x - sw / 2), int(y - sh / 2)),
                     special_flags=pygame.BLEND_ADD)

    def dibujar_modelo(self, a):
        surf = self.render_surf
        surf.fill(COLOR_BG)
        cx, cy = self.RW // 2, self.RH // 2
        escala = min(self.RW, self.RH) * 0.35

        # Galaxias primero (fondo difuso, más lejanas)
        self._dibujar_galaxias(surf, a, cx, cy, escala)

        # Asteroides (más numerosos y pequeños)
        self._dibujar_categoria(surf, self.universo["asteroides"], a,
                                cx, cy, escala)
        # Planetas
        self._dibujar_categoria(surf, self.universo["planetas"], a,
                                cx, cy, escala)
        # Soles con halo
        self._dibujar_categoria(surf, self.universo["soles"], a,
                                cx, cy, escala, glows=self.glows)

        # Observador central
        pygame.draw.circle(surf, COLOR_OBSERVADOR, (cx, cy),
                           max(2, int(3 * SUPERSAMPLE)))

        # Horizonte de Hubble
        radio_h = int(escala * a * 0.95)
        if radio_h > 0:
            color_h = COLOR_HOY_LINE if abs(self.t_sim - T_HOY) < 0.5 \
                      else (35, 35, 90)
            grosor = max(1, int(SUPERSAMPLE))
            pygame.draw.circle(surf, color_h, (cx, cy), radio_h, grosor)

        # Reescalar a tamaño de pantalla (antialiasing)
        pygame.transform.smoothscale(surf, (self.W, self.H), self.model_surf)

    def dibujar_hud(self, screen, a, fps):
        if not pygame.font.get_init():
            return
        try:
            font = pygame.font.SysFont("monospace", 16)
        except Exception:
            return

        lineas = [
            f"t = {self.t_sim:5.1f} Gyr",
            f"a(t) = {a:.4f}",
            f"{'◀ HOY' if abs(self.t_sim - T_HOY) < 1 else ''}",
            f"vel x{self.velocidad:.1f}  {'⏸' if self.pausado else '▶'}",
            f"FPS {fps:.0f}",
        ]
        y = 10
        for l in lineas:
            if l.strip():
                screen.blit(font.render(l, True, (180, 180, 180)), (10, y))
            y += 20

        # Leyenda de colores
        leyenda = [
            ("Soles",      (255, 244, 200)),
            ("Planetas",   (130, 200, 230)),
            ("Asteroides", (140, 130, 115)),
            ("Galaxias",   (220, 140, 230)),
        ]
        y = 10
        for nombre, color in leyenda:
            pygame.draw.circle(screen, color, (self.W - 130, y + 8), 5)
            txt = font.render(nombre, True, (170, 170, 170))
            screen.blit(txt, (self.W - 115, y))
            y += 20

        bw = 200
        bx, by = 10, self.H - 30
        prog = (self.t_sim - T_MIN) / (T_MAX - T_MIN)
        pygame.draw.rect(screen, (40, 40, 40), (bx, by, bw, 8))
        pygame.draw.rect(screen, (80, 160, 255), (bx, by, int(bw*prog), 8))
        x_hoy = bx + int(bw * (T_HOY - T_MIN) / (T_MAX - T_MIN))
        pygame.draw.line(screen, COLOR_HOY_LINE, (x_hoy, by-3), (x_hoy, by+11), 2)

        ctrl = font.render(
            "ESPACIO:pausa  R:reiniciar  +/-:vel  F:fullscreen  ESC:salir",
            True, (70, 70, 70))
        screen.blit(ctrl, (10, self.H - 48))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display",    type=int,  default=0)
    parser.add_argument("--width",      type=int,  default=1280)
    parser.add_argument("--height",     type=int,  default=720)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    if args.display == 1:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{args.width + 10},0"

    pygame.init()
    try:
        pygame.font.init()
    except Exception as e:
        print(f"Aviso: fuentes no disponibles ({e}). El HUD se omitirá.")

    flags = (pygame.FULLSCREEN | pygame.NOFRAME) if args.fullscreen else 0
    screen = pygame.display.set_mode((args.width, args.height), flags)
    pygame.display.set_caption("Expansión del Universo — Pepper Ghost")

    renderer = UniverseRenderer(args.width, args.height)
    clock    = pygame.time.Clock()
    fullscreen = args.fullscreen

    print("\n✓ Listo. ESPACIO=pausa  R=reiniciar  +/-=velocidad  F=fullscreen  ESC=salir\n")

    running = True
    while running:
        dt = clock.tick(FPS_TARGET) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    renderer.pausado = not renderer.pausado
                elif k == pygame.K_r:
                    renderer.t_sim = T_MIN
                    renderer.angulo_rot = 0.0
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    renderer.velocidad = min(renderer.velocidad * 1.5, 16.0)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    renderer.velocidad = max(renderer.velocidad / 1.5, 0.25)
                elif k == pygame.K_f:
                    fullscreen = not fullscreen
                    flags = (pygame.FULLSCREEN | pygame.NOFRAME) if fullscreen else 0
                    screen = pygame.display.set_mode((args.width, args.height), flags)

        renderer.avanzar(dt)
        a = renderer.a_en(renderer.t_sim)
        renderer.dibujar_modelo(a)
        blit_to_pepper(screen, renderer.model_surf, args.width, args.height)
        renderer.dibujar_hud(screen, a, clock.get_fps())
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
