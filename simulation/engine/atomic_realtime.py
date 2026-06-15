"""
simulation/engine/atomic_realtime.py

Renderizado en tiempo real de orbitales atómicos del hidrógeno
(densidad de probabilidad |psi_nlm|^2) usando pygame. Diseñado para
proyección Pepper's Ghost con vidrio horizontal a 45°.

Técnica de visualización: "nube de puntos" (dot-density plot).
Para cada orbital se muestrean miles de puntos (x,y,z) con
probabilidad proporcional a |psi|^2, usando el hecho de que para
el átomo de hidrógeno la distribución radial y la angular son
independientes:

    |psi_nlm|^2 dV = [R_nl(r)^2 r^2 dr] · [Y(theta,phi)^2 sin(theta) dtheta dphi]

Se usan armónicos esféricos REALES (combinaciones de Y_l^m) para
obtener las formas clásicas de libro de texto (p_x, p_y, p_z, d_xy,
d_z2, etc.), cuyo signo determina el color del punto (lóbulo de fase
positiva / negativa).

El programa recorre una secuencia de orbitales (1s, 2s, 2p_z, 2p_x,
3d_z2, 3d_xy) en un ciclo de 60 s, con rotación 3D continua.

Controles:
  ESPACIO     → pausar / reanudar
  R           → reiniciar el ciclo
  ←  /  →     → orbital anterior / siguiente (manual)
  + / -       → acelerar / ralentizar
  F           → fullscreen
  ESC / Q     → salir

Uso:
  python atomic_realtime.py
  python atomic_realtime.py --display 1 --fullscreen
"""

import sys
import os
import argparse
import math
import numpy as np
import pygame
from scipy.special import genlaguerre, factorial

sys.path.insert(0, os.path.dirname(__file__))
from pepper_layout import blit_to_pepper

# ── Parámetros de simulación ────────────────────────────────
N_PUNTOS       = 6000
DURACION_ORB   = 10.0     # segundos por orbital
FPS_TARGET     = 60
SUPERSAMPLE    = 1.75
TILT           = math.radians(22)   # inclinación fija de cámara

COLOR_BG     = (0, 0, 0)
COLOR_POS    = (110, 200, 255)   # lóbulo de fase positiva (cian)
COLOR_NEG    = (255, 150,  90)   # lóbulo de fase negativa (naranja)
COLOR_NUCLEO = (255, 250, 220)   # núcleo (protón)
COLOR_HUD    = (180, 180, 180)


# ═══════════════════════════════════════════════════════════════════════════
# Funciones de onda del hidrógeno
# ═══════════════════════════════════════════════════════════════════════════

def radial_wavefunction(n, l, r):
    """Parte radial R_nl(r), en unidades del radio de Bohr a0=1."""
    rho = 2 * r / n
    norm = np.sqrt((2/n)**3 * factorial(n-l-1) / (2*n*factorial(n+l)**3))
    L = genlaguerre(n-l-1, 2*l+1)(rho)
    return norm * np.exp(-rho/2) * rho**l * L


def radial_extent(n, l, factor=0.97):
    """Radio r tal que la probabilidad acumulada radial = factor."""
    r_coarse = np.linspace(1e-4, 80*n**2, 6000)
    R = radial_wavefunction(n, l, r_coarse)
    P = R**2 * r_coarse**2
    dr = r_coarse[1] - r_coarse[0]
    P = P / (P.sum() * dr)
    cdf = np.cumsum(P) * dr
    idx = np.searchsorted(cdf, factor)
    return r_coarse[min(idx, len(r_coarse) - 1)]


def angular_real(l, tipo, theta, phi):
    """Armónicos esféricos reales (formas clásicas de libro de texto)."""
    if l == 0:
        return np.ones_like(theta)
    if l == 1:
        if tipo == "pz": return np.cos(theta)
        if tipo == "px": return np.sin(theta) * np.cos(phi)
        if tipo == "py": return np.sin(theta) * np.sin(phi)
    if l == 2:
        if tipo == "dz2":   return (3*np.cos(theta)**2 - 1)
        if tipo == "dxz":   return np.sin(theta)*np.cos(theta)*np.cos(phi)
        if tipo == "dyz":   return np.sin(theta)*np.cos(theta)*np.sin(phi)
        if tipo == "dx2y2": return np.sin(theta)**2 * np.cos(2*phi)
        if tipo == "dxy":   return np.sin(theta)**2 * np.sin(2*phi)
    raise ValueError(f"Combinación no soportada: l={l}, tipo={tipo}")


def generar_nube_puntos(n, l, tipo, n_puntos=N_PUNTOS, seed=0):
    """
    Muestrea n_puntos posiciones (x,y,z) con densidad |psi_nlm|^2,
    aprovechando la separabilidad radial/angular.

    Retorna
    -------
    pos    : (N,3) posiciones normalizadas (r_ref = 1 contiene ~90% prob.)
    signo  : (N,)  +1/-1 según el lóbulo (fase del armónico real)
    r_norm : (N,)  radio normalizado de cada punto (para sombreado)
    """
    rng = np.random.default_rng(seed)

    # ── Distribución radial ─────────────────────────────────
    r_max = radial_extent(n, l, 0.985) * 1.1
    r_grid = np.linspace(1e-4, r_max, 800)
    R = radial_wavefunction(n, l, r_grid)
    Pr = R**2 * r_grid**2
    Pr = Pr / Pr.sum()
    r_s = rng.choice(r_grid, size=n_puntos, p=Pr)

    # ── Distribución angular ────────────────────────────────
    n_th, n_ph = 90, 180
    th_grid = np.linspace(0, np.pi, n_th)
    ph_grid = np.linspace(0, 2*np.pi, n_ph, endpoint=False)
    TH, PH = np.meshgrid(th_grid, ph_grid, indexing="ij")
    Yr = angular_real(l, tipo, TH, PH)
    Pang = (Yr**2) * np.sin(TH)
    Pang_flat = Pang.flatten()
    Pang_flat = Pang_flat / Pang_flat.sum()
    idx = rng.choice(len(Pang_flat), size=n_puntos, p=Pang_flat)
    th_s = TH.flatten()[idx]
    ph_s = PH.flatten()[idx]

    signo = np.sign(Yr.flatten()[idx])
    signo[signo == 0] = 1

    x = r_s * np.sin(th_s) * np.cos(ph_s)
    y = r_s * np.sin(th_s) * np.sin(ph_s)
    z = r_s * np.cos(th_s)

    # Normalización: r_ref contiene ~90% de la probabilidad radial,
    # así todos los orbitales se ven con un tamaño comparable.
    r_ref = radial_extent(n, l, 0.90)
    pos = np.stack([x, y, z], axis=1) / r_ref
    r_norm = r_s / r_ref

    return pos, signo, r_norm


# ═══════════════════════════════════════════════════════════════════════════
# Catálogo de orbitales en el ciclo de demostración
# ═══════════════════════════════════════════════════════════════════════════

ORBITALES = [
    dict(n=1, l=0, tipo="s",  nombre="1s",
         descripcion="Esfera uniforme — estado fundamental"),
    dict(n=2, l=0, tipo="s",  nombre="2s",
         descripcion="Capa con nodo radial"),
    dict(n=2, l=1, tipo="pz", nombre="2p_z",
         descripcion="Lóbulos a lo largo del eje z"),
    dict(n=2, l=1, tipo="px", nombre="2p_x",
         descripcion="Lóbulos a lo largo del eje x"),
    dict(n=3, l=2, tipo="dz2", nombre="3d_z2",
         descripcion="Anillo ecuatorial + lóbulos polares"),
    dict(n=3, l=2, tipo="dxy", nombre="3d_xy",
         descripcion="Cuatro lóbulos tipo trébol"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Proyección 3D → 2D con inclinación fija + rotación continua
# ═══════════════════════════════════════════════════════════════════════════

def proyectar(pos3d, angulo_rot, tilt, cx, cy, escala):
    x0, y0, z0 = pos3d[:, 0], pos3d[:, 1], pos3d[:, 2]

    # Inclinación fija de cámara (rotación en X)
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    y1 = y0 * cos_t - z0 * sin_t
    z1 = y0 * sin_t + z0 * cos_t
    x1 = x0

    # Rotación continua en Y
    cos_a, sin_a = math.cos(angulo_rot), math.sin(angulo_rot)
    x2 = x1 * cos_a + z1 * sin_a
    z2 = -x1 * sin_a + z1 * cos_a
    y2 = y1

    d_foco = 2.5
    persp = d_foco / (z2 + d_foco + 1e-5)
    px = cx + x2 * escala * persp
    py = cy - y2 * escala * persp
    return px, py, persp, z2


# ═══════════════════════════════════════════════════════════════════════════
# Halo / glow (reutilizado del modelo de universo)
# ═══════════════════════════════════════════════════════════════════════════

def crear_glow_template(size=40):
    """Degradado radial horneado en RGB, para BLEND_ADD."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Renderer principal
# ═══════════════════════════════════════════════════════════════════════════

class AtomicRenderer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.RW = int(W * SUPERSAMPLE)
        self.RH = int(H * SUPERSAMPLE)

        print("Generando nubes de probabilidad |psi|^2 para cada orbital...")
        self.nubes = []
        for i, orb in enumerate(ORBITALES):
            pos, signo, r_norm = generar_nube_puntos(
                orb["n"], orb["l"], orb["tipo"], seed=100 + i)
            self.nubes.append((pos, signo, r_norm))
            print(f"  · {orb['nombre']:<6} listo "
                  f"({(signo>0).mean()*100:5.1f}% lóbulo +)")

        self.t_sim      = 0.0
        self.pausado    = False
        self.velocidad  = 1.0
        self.angulo_rot = 0.0
        self.indice_manual = None   # si no es None, fija el orbital

        self.render_surf = pygame.Surface((self.RW, self.RH))
        self.model_surf  = pygame.Surface((self.W, self.H))

        # Halo del núcleo (precoloreado, blanco cálido)
        print("Preparando halo del núcleo...")
        glow = crear_glow_template(36)
        glow.fill(COLOR_NUCLEO, special_flags=pygame.BLEND_RGB_MULT)
        self.glow_nucleo = glow

    @property
    def duracion_ciclo(self):
        return DURACION_ORB * len(ORBITALES)

    def indice_actual(self):
        if self.indice_manual is not None:
            return self.indice_manual
        return int(self.t_sim // DURACION_ORB) % len(ORBITALES)

    def progreso_orbital(self):
        """Fracción [0,1) transcurrida del orbital actual."""
        return (self.t_sim % DURACION_ORB) / DURACION_ORB

    def avanzar(self, dt_real):
        if self.pausado:
            return
        self.t_sim += dt_real * self.velocidad
        if self.t_sim >= self.duracion_ciclo:
            self.t_sim = 0.0
        self.angulo_rot += dt_real * 0.35 * self.velocidad

    def cambiar_orbital(self, delta):
        """Navegación manual con flechas."""
        idx = self.indice_actual()
        nuevo = (idx + delta) % len(ORBITALES)
        self.indice_manual = nuevo
        self.t_sim = nuevo * DURACION_ORB

    def reanudar_automatico(self):
        self.indice_manual = None

    def dibujar_modelo(self, idx):
        surf = self.render_surf
        surf.fill(COLOR_BG)
        cx, cy = self.RW // 2, self.RH // 2
        escala = min(self.RW, self.RH) * 0.32

        pos, signo, r_norm = self.nubes[idx]
        px, py, persp, z = proyectar(pos, self.angulo_rot, TILT,
                                     cx, cy, escala)
        orden = np.argsort(z)

        Wd, Hd = surf.get_size()

        for i in orden:
            x, y = px[i], py[i]
            if not (-10 <= x < Wd + 10 and -10 <= y < Hd + 10):
                continue
            base = COLOR_POS if signo[i] > 0 else COLOR_NEG
            # Sombreado: puntos cercanos al núcleo, ligeramente más claros
            brillo = 0.55 + 0.45 * max(0.0, 1.0 - r_norm[i])
            color = tuple(min(255, int(c * brillo)) for c in base)
            sz = max(1, int(1.4 * persp[i] * SUPERSAMPLE))
            pygame.draw.circle(surf, color, (int(x), int(y)), sz)

        # Núcleo (protón) en el origen, con halo
        nx, ny, nper, _ = proyectar(np.array([[0.0, 0.0, 0.0]]),
                                    self.angulo_rot, TILT, cx, cy, escala)
        nx, ny = int(nx[0]), int(ny[0])
        glow_sz = max(4, int(14 * SUPERSAMPLE))
        glow_scaled = pygame.transform.smoothscale(
            self.glow_nucleo, (glow_sz, glow_sz))
        surf.blit(glow_scaled, (nx - glow_sz//2, ny - glow_sz//2),
                 special_flags=pygame.BLEND_ADD)
        pygame.draw.circle(surf, (255, 255, 255), (nx, ny),
                           max(1, int(2 * SUPERSAMPLE)))

        # Reescalar a tamaño de pantalla (antialiasing)
        pygame.transform.smoothscale(surf, (self.W, self.H), self.model_surf)

    def dibujar_hud(self, screen, idx, fps):
        if not pygame.font.get_init():
            return
        try:
            font  = pygame.font.SysFont("monospace", 16)
            font2 = pygame.font.SysFont("monospace", 14)
        except Exception:
            return

        orb = ORBITALES[idx]
        n, l = orb["n"], orb["l"]
        m_label = {"s": 0, "pz": 0, "px": "±1", "py": "±1",
                  "dz2": 0, "dxz": "±1", "dyz": "±1",
                  "dx2y2": "±2", "dxy": "±2"}[orb["tipo"]]
        energia = -13.6 / n**2

        lineas = [
            f"Orbital {orb['nombre']}",
            f"n={n}  l={l}  m={m_label}",
            f"E_{n} = {energia:.2f} eV",
            "",
            f"vel x{self.velocidad:.1f}  {'⏸' if self.pausado else '▶'}",
            f"FPS {fps:.0f}",
        ]
        y = 10
        for li in lineas:
            if li:
                screen.blit(font.render(li, True, COLOR_HUD), (10, y))
            y += 20

        # Descripción del orbital
        desc = font2.render(orb["descripcion"], True, (140, 140, 140))
        screen.blit(desc, (10, y + 4))

        # Leyenda de fase
        leyenda = [
            ("Fase +",  COLOR_POS),
            ("Fase -",  COLOR_NEG),
            ("Núcleo",  COLOR_NUCLEO),
        ]
        y = 10
        for nombre, color in leyenda:
            pygame.draw.circle(screen, color, (self.W - 110, y + 8), 5)
            txt = font.render(nombre, True, (170, 170, 170))
            screen.blit(txt, (self.W - 95, y))
            y += 20

        # Barra de progreso del orbital actual + marcas de cada orbital
        bw = 240
        bx, by = 10, self.H - 30
        n_orb = len(ORBITALES)
        prog_total = self.t_sim / self.duracion_ciclo
        pygame.draw.rect(screen, (40, 40, 40), (bx, by, bw, 8))
        pygame.draw.rect(screen, (80, 160, 255),
                         (bx, by, int(bw * prog_total), 8))
        for k in range(1, n_orb):
            xk = bx + int(bw * k / n_orb)
            pygame.draw.line(screen, (90, 90, 90), (xk, by-2), (xk, by+10), 1)

        ctrl = font.render(
            "ESPACIO:pausa  R:reinicio  ←/→:orbital  +/-:vel  F:fullscreen  ESC:salir",
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
    pygame.display.set_caption("Orbitales Atómicos — Pepper Ghost")

    renderer = AtomicRenderer(args.width, args.height)
    clock    = pygame.time.Clock()
    fullscreen = args.fullscreen

    print("\n✓ Listo. ESPACIO=pausa  R=reinicio  ←/→=orbital  +/-=velocidad  "
          "F=fullscreen  ESC=salir\n")

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
                    renderer.t_sim = 0.0
                    renderer.angulo_rot = 0.0
                    renderer.reanudar_automatico()
                elif k == pygame.K_RIGHT:
                    renderer.cambiar_orbital(+1)
                elif k == pygame.K_LEFT:
                    renderer.cambiar_orbital(-1)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    renderer.velocidad = min(renderer.velocidad * 1.5, 8.0)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    renderer.velocidad = max(renderer.velocidad / 1.5, 0.25)
                elif k == pygame.K_f:
                    fullscreen = not fullscreen
                    flags = (pygame.FULLSCREEN | pygame.NOFRAME) if fullscreen else 0
                    screen = pygame.display.set_mode((args.width, args.height), flags)

        renderer.avanzar(dt)
        idx = renderer.indice_actual()
        renderer.dibujar_modelo(idx)
        blit_to_pepper(screen, renderer.model_surf, args.width, args.height)
        renderer.dibujar_hud(screen, idx, clock.get_fps())
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
