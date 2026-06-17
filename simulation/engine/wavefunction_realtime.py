"""
simulation/engine/wavefunction_realtime.py

Renderizado en tiempo real de paquetes de onda cuánticos 1D
Psi(x,t) = sum_n c_n psi_n(x) e^{-i E_n t}, usando pygame.
Diseñado para proyección Pepper's Ghost con vidrio horizontal a 45°.

La función de onda compleja se visualiza como una "cinta" 3D:
    eje X      -> posición x
    eje Y      -> Re[Psi(x,t)]
    eje Z      -> Im[Psi(x,t)]
cuyo color codifica la FASE local (hue = arg(Psi)/2pi) y cuyo brillo
codifica la densidad |Psi|^2. Bajo la cinta se dibuja el "paisaje" de
densidad de probabilidad |Psi(x,t)|^2 y el potencial V(x) de referencia,
además de un marcador para <x>(t) (valor esperado de la posición).

Se recorren tres sistemas, 20 s cada uno (ciclo de 60 s, igual que los
otros dos modelos):
  1. Pozo infinito   -> paquete de onda rebotando entre paredes
  2. Oscilador armónico (estado coherente) -> oscilación tipo clásica
  3. Doble pozo      -> oscilación túnel entre los dos mínimos

Controles:
  ESPACIO     → pausar / reanudar
  R           → reiniciar el ciclo
  ←  /  →     → sistema anterior / siguiente (manual)
  + / -       → acelerar / ralentizar
  F           → fullscreen
  ESC / Q     → salir

Uso:
  python wavefunction_realtime.py
  python wavefunction_realtime.py --display 1 --fullscreen
"""

import sys
import os
import argparse
import math
import colorsys
import numpy as np
import pygame
from scipy.special import eval_hermite, factorial
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.dirname(__file__))
from pepper_layout import blit_to_pepper

# ── Parámetros generales ─────────────────────────────────────
N_X          = 400
T_SISTEMA    = 20.0     # segundos por sistema
CICLOS_OBJ   = 2.5      # oscilaciones visuales por sistema en T_SISTEMA
FPS_TARGET   = 60
SUPERSAMPLE  = 1.75
TILT         = math.radians(18)

HBAR = MASS = 1.0

COLOR_BG    = (0, 0, 0)
COLOR_V     = (90, 90, 100)
COLOR_X     = (255, 250, 220)
COLOR_AXIS  = (40, 40, 50)
COLOR_HUD   = (180, 180, 180)


# ═══════════════════════════════════════════════════════════════════════════
# Autoestados de cada sistema
# ═══════════════════════════════════════════════════════════════════════════

def autoestados_pozo(n_states=6, L=10.0):
    x = np.linspace(0, L, N_X)
    psis = np.zeros((N_X, n_states))
    Es = np.zeros(n_states)
    for n in range(1, n_states + 1):
        psis[:, n-1] = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        Es[n-1] = (n**2 * np.pi**2) / (2 * MASS * L**2)
    V = np.zeros(N_X)
    x_norm = (x / L - 0.5) * 2.0
    return x_norm, psis, Es, V


def autoestados_oscilador(n_states=14, x_max=7.0):
    x = np.linspace(-x_max, x_max, N_X)
    psis = np.zeros((N_X, n_states))
    Es = np.zeros(n_states)
    for n in range(n_states):
        norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
        psis[:, n] = norm * eval_hermite(n, x) * np.exp(-x**2 / 2)
        Es[n] = n + 0.5
    V = 0.5 * x**2
    x_norm = x / x_max
    return x_norm, psis, Es, V


def autoestados_doble_pozo(n_states=4, x_max=7.0, a=0.5, b=0.06):
    x = np.linspace(-x_max, x_max, N_X)
    dx = x[1] - x[0]
    V = -a * x**2 + b * x**4
    diag_main = HBAR**2 / (MASS * dx**2) + V
    diag_off = -HBAR**2 / (2 * MASS * dx**2) * np.ones(N_X - 1)
    E, psi = eigh_tridiagonal(diag_main, diag_off,
                              select="i", select_range=(0, n_states-1))
    for n in range(n_states):
        norm = np.sqrt(np.sum(psi[:, n]**2) * dx)
        psi[:, n] /= norm
    x_norm = x / x_max
    return x_norm, psi, E, V


# ═══════════════════════════════════════════════════════════════════════════
# Catálogo de sistemas en el ciclo de demostración
# ═══════════════════════════════════════════════════════════════════════════

def coef_coherente(alpha, n_states):
    c = np.array([np.exp(-alpha**2/2) * alpha**n / np.sqrt(float(factorial(n)))
                  for n in range(n_states)])
    return c / np.sqrt(np.sum(np.abs(c)**2))


SISTEMAS = [
    dict(nombre="Pozo infinito de potencial",
         clave="pozo",
         descripcion="Paquete de onda (n=1,2) rebotando entre paredes",
         build=lambda: autoestados_pozo(),
         coef=lambda E: _coef_dos_niveles_arr(0, 1, len(E))),
    dict(nombre="Oscilador armónico",
         clave="oscilador",
         descripcion="Estado coherente: oscilación tipo clásica",
         build=lambda: autoestados_oscilador(),
         coef=lambda E: coef_coherente(2.5, len(E))),
    dict(nombre="Doble pozo de potencial",
         clave="doble_pozo",
         descripcion="Oscilación túnel entre los dos mínimos (n=0,1)",
         build=lambda: autoestados_doble_pozo(),
         coef=lambda E: _coef_dos_niveles_arr(0, 1, len(E))),
]


def _coef_dos_niveles_arr(i, j, n_total):
    c = np.zeros(n_total)
    c[i] = c[j] = 1/np.sqrt(2)
    return c


# ═══════════════════════════════════════════════════════════════════════════
# Proyección 3D → 2D con inclinación fija + rotación continua
# (idéntica a atomic_realtime.py)
# ═══════════════════════════════════════════════════════════════════════════

def proyectar(pos3d, angulo_rot, tilt, cx, cy, escala):
    x0, y0, z0 = pos3d[:, 0], pos3d[:, 1], pos3d[:, 2]
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    y1 = y0 * cos_t - z0 * sin_t
    z1 = y0 * sin_t + z0 * cos_t
    x1 = x0
    cos_a, sin_a = math.cos(angulo_rot), math.sin(angulo_rot)
    x2 = x1 * cos_a + z1 * sin_a
    z2 = -x1 * sin_a + z1 * cos_a
    y2 = y1
    d_foco = 2.5
    persp = d_foco / (z2 + d_foco + 1e-5)
    px = cx + x2 * escala * persp
    py = cy - y2 * escala * persp
    return px, py, persp, z2


def crear_glow_template(size=40):
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

class WaveRenderer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.RW = int(W * SUPERSAMPLE)
        self.RH = int(H * SUPERSAMPLE)

        print("Calculando autoestados de los 3 sistemas...")
        self.sistemas = []
        for sys_def in SISTEMAS:
            x_norm, psis, Es, V = sys_def["build"]()
            c = sys_def["coef"](Es)
            dE = self._delta_e_dominante(Es, c)
            scale = CICLOS_OBJ * 2*np.pi / (dE * T_SISTEMA)

            datos = dict(x_norm=x_norm, psis=psis, Es=Es, c=c,
                         V=V, scale=scale, dE=dE,
                         nombre=sys_def["nombre"],
                         descripcion=sys_def["descripcion"])
            datos["amp_re"], datos["amp_dens"] = self._normalizar(datos)
            self.sistemas.append(datos)
            print(f"  · {sys_def['nombre']:<28} dE={dE:.4f}  scale={scale:.3f}")

        self.t_sim         = 0.0
        self.pausado       = False
        self.velocidad     = 1.0
        self.angulo_rot    = 0.0
        self.indice_manual = None

        self.render_surf = pygame.Surface((self.RW, self.RH))
        self.model_surf  = pygame.Surface((self.W, self.H))

        glow = crear_glow_template(36)
        glow.fill(COLOR_X, special_flags=pygame.BLEND_RGB_MULT)
        self.glow_marcador = glow

    @staticmethod
    def _delta_e_dominante(Es, c):
        """Diferencia de energía dominante (mayor peso) para fijar la
        escala temporal de la animación."""
        pesos = np.abs(c)**2
        idx = np.argsort(pesos)[::-1]
        n_top = idx[:2]
        if len(n_top) < 2:
            return max(Es[0], 1e-3)
        dE = abs(Es[n_top[0]] - Es[n_top[1]])
        return dE if dE > 1e-6 else max(Es[idx[0]], 1e-3)

    def _evaluar(self, datos, t_eff):
        fases = np.exp(-1j * datos["Es"] * t_eff)
        Psi = datos["psis"] @ (datos["c"] * fases)
        return Psi

    def _normalizar(self, datos):
        """Muestrea Psi(x,t) en un ciclo completo para fijar amplitudes
        de la cinta (Re/Im) y del paisaje de densidad."""
        T = 2*np.pi / datos["dE"]
        max_re, max_dens = 0.0, 0.0
        for t in np.linspace(0, T, 24):
            Psi = self._evaluar(datos, t)
            max_re = max(max_re, np.max(np.abs(Psi.real)), np.max(np.abs(Psi.imag)))
            max_dens = max(max_dens, np.max(np.abs(Psi)**2))
        amp_re = 0.40 / max(max_re, 1e-9)
        amp_dens = 0.40 / max(max_dens, 1e-9)
        return amp_re, amp_dens

    @property
    def duracion_ciclo(self):
        return T_SISTEMA * len(self.sistemas)

    def indice_actual(self):
        if self.indice_manual is not None:
            return self.indice_manual
        return int(self.t_sim // T_SISTEMA) % len(self.sistemas)

    def t_local(self):
        return self.t_sim % T_SISTEMA

    def avanzar(self, dt_real):
        if self.pausado:
            return
        self.t_sim += dt_real * self.velocidad
        if self.t_sim >= self.duracion_ciclo:
            self.t_sim = 0.0
        self.angulo_rot += dt_real * 0.18 * self.velocidad

    def cambiar_sistema(self, delta):
        idx = self.indice_actual()
        nuevo = (idx + delta) % len(self.sistemas)
        self.indice_manual = nuevo
        self.t_sim = nuevo * T_SISTEMA

    def reanudar_automatico(self):
        self.indice_manual = None

    def dibujar_modelo(self, idx):
        surf = self.render_surf
        surf.fill(COLOR_BG)
        cx, cy = self.RW // 2, self.RH // 2
        escala = min(self.RW, self.RH) * 0.40

        datos = self.sistemas[idx]
        t_eff = self.t_local() * datos["scale"]
        Psi = self._evaluar(datos, t_eff)

        x_norm = datos["x_norm"]
        re = Psi.real * datos["amp_re"]
        im = Psi.imag * datos["amp_re"]
        dens = (np.abs(Psi)**2) * datos["amp_dens"]
        fase = (np.angle(Psi) / (2*np.pi)) % 1.0
        dens_norm = dens / max(np.max(dens), 1e-9)

        # ── Eje / línea base ────────────────────────────────────────────
        base_pts = np.stack([x_norm, np.zeros_like(x_norm),
                             np.zeros_like(x_norm)], axis=1)
        pxb, pyb, _, _ = proyectar(base_pts, self.angulo_rot, TILT,
                                   cx, cy, escala)
        pts_axis = list(zip(pxb.astype(int), pyb.astype(int)))
        if len(pts_axis) > 1:
            pygame.draw.lines(surf, COLOR_AXIS, False, pts_axis,
                             max(1, int(SUPERSAMPLE)))

        # ── Potencial V(x), normalizado a una franja inferior ───────────
        Vn = datos["V"]
        if np.max(Vn) - np.min(Vn) > 1e-9:
            V_disp = (Vn - np.min(Vn)) / (np.max(Vn) - np.min(Vn)) * 0.28
        else:
            V_disp = np.zeros_like(Vn)
        v_pts = np.stack([x_norm, -0.85 + V_disp, np.zeros_like(x_norm)], axis=1)
        pxv, pyv, _, _ = proyectar(v_pts, self.angulo_rot, TILT, cx, cy, escala)
        pts_v = list(zip(pxv.astype(int), pyv.astype(int)))
        if len(pts_v) > 1:
            pygame.draw.lines(surf, COLOR_V, False, pts_v,
                             max(1, int(SUPERSAMPLE)))

        # ── Paisaje de densidad |Psi(x,t)|^2 ─────────────────────────────
        d_pts = np.stack([x_norm, -0.55 + dens, np.zeros_like(x_norm)], axis=1)
        pxd, pyd, perspd, _ = proyectar(d_pts, self.angulo_rot, TILT,
                                        cx, cy, escala)
        for i in range(len(x_norm) - 1):
            brillo = 0.25 + 0.75 * dens_norm[i]
            color = (int(255*brillo), int(120*brillo), int(60*brillo))
            p1 = (int(pxd[i]), int(pyd[i]))
            p2 = (int(pxd[i+1]), int(pyd[i+1]))
            pygame.draw.line(surf, color, p1, p2, max(1, int(2*SUPERSAMPLE)))

        # ── Cinta compleja Psi(x,t): Re/Im, color por fase ──────────────
        psi_pts = np.stack([x_norm, re, im], axis=1)
        pxp, pyp, perspp, _ = proyectar(psi_pts, self.angulo_rot, TILT,
                                        cx, cy, escala)
        for i in range(len(x_norm)):
            h = fase[i]
            v = 0.45 + 0.55 * dens_norm[i]
            r, g, bch = colorsys.hsv_to_rgb(h, 0.85, v)
            color = (int(r*255), int(g*255), int(bch*255))
            sz = max(1, int(1.6 * perspp[i] * SUPERSAMPLE))
            pygame.draw.circle(surf, color, (int(pxp[i]), int(pyp[i])), sz)
        for i in range(len(x_norm)-1):
            p1 = (int(pxp[i]), int(pyp[i]))
            p2 = (int(pxp[i+1]), int(pyp[i+1]))
            h = fase[i]
            v = 0.45 + 0.55*dens_norm[i]
            r,g,bch = colorsys.hsv_to_rgb(h, 0.7, v)
            pygame.draw.line(surf, (int(r*255),int(g*255),int(bch*255)), p1, p2,
                            max(1,int(SUPERSAMPLE)))

        # ── Marcador <x>(t) sobre el paisaje de densidad ────────────────
        x_exp = np.sum(x_norm * (np.abs(Psi)**2)) / np.sum(np.abs(Psi)**2)
        mk = np.array([[x_exp, -0.55, 0.0]])
        pxm, pym, _, _ = proyectar(mk, self.angulo_rot, TILT, cx, cy, escala)
        mxx, myy = int(pxm[0]), int(pym[0])
        glow_sz = max(4, int(14*SUPERSAMPLE))
        glow_scaled = pygame.transform.smoothscale(self.glow_marcador,
                                                    (glow_sz, glow_sz))
        surf.blit(glow_scaled, (mxx-glow_sz//2, myy-glow_sz//2),
                 special_flags=pygame.BLEND_ADD)
        pygame.draw.circle(surf, (255,255,255), (mxx, myy), max(1,int(2*SUPERSAMPLE)))

        pygame.transform.smoothscale(surf, (self.W, self.H), self.model_surf)

    def dibujar_hud(self, screen, idx, fps):
        if not pygame.font.get_init():
            return
        try:
            font  = pygame.font.SysFont("monospace", 16)
            font2 = pygame.font.SysFont("monospace", 14)
        except Exception:
            return

        datos = self.sistemas[idx]
        c = datos["c"]
        idx_top = np.argsort(np.abs(c)**2)[::-1][:2]
        idx_top = sorted(idx_top)
        E_str = ", ".join(f"E_{n}={datos['Es'][n]:.3f}" for n in idx_top)

        lineas = [
            datos["nombre"],
            E_str,
            f"vel x{self.velocidad:.1f}  {'⏸' if self.pausado else '▶'}",
            f"FPS {fps:.0f}",
        ]
        y = 10
        for li in lineas:
            screen.blit(font.render(li, True, COLOR_HUD), (10, y))
            y += 20

        desc = font2.render(datos["descripcion"], True, (140, 140, 140))
        screen.blit(desc, (10, y+4))

        leyenda = [
            ("Psi(x,t)", (110, 200, 255)),
            ("|Psi|^2",  (255, 150,  60)),
            ("V(x)",     COLOR_V),
            ("<x>(t)",   COLOR_X),
        ]
        y = 10
        for nombre, color in leyenda:
            pygame.draw.circle(screen, color, (self.W - 120, y+8), 5)
            screen.blit(font.render(nombre, True, (170,170,170)), (self.W-105, y))
            y += 20

        bw = 240
        bx, by = 10, self.H - 30
        n_sis = len(self.sistemas)
        prog_total = self.t_sim / self.duracion_ciclo
        pygame.draw.rect(screen, (40,40,40), (bx, by, bw, 8))
        pygame.draw.rect(screen, (80,160,255), (bx, by, int(bw*prog_total), 8))
        for k in range(1, n_sis):
            xk = bx + int(bw*k/n_sis)
            pygame.draw.line(screen, (90,90,90), (xk,by-2), (xk,by+10), 1)

        ctrl = font.render(
            "ESPACIO:pausa  R:reinicio  ←/→:sistema  +/-:vel  F:fullscreen  ESC:salir",
            True, (70,70,70))
        screen.blit(ctrl, (10, self.H-48))


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
    pygame.display.set_caption("Funciones de Onda 1D — Pepper Ghost")

    renderer = WaveRenderer(args.width, args.height)
    clock    = pygame.time.Clock()
    fullscreen = args.fullscreen

    print("\n✓ Listo. ESPACIO=pausa  R=reinicio  ←/→=sistema  +/-=velocidad  "
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
                    renderer.cambiar_sistema(+1)
                elif k == pygame.K_LEFT:
                    renderer.cambiar_sistema(-1)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    renderer.velocidad = min(renderer.velocidad*1.5, 8.0)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    renderer.velocidad = max(renderer.velocidad/1.5, 0.25)
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
