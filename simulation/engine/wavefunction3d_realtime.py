"""
simulation/engine/wavefunction3d_realtime.py

Renderizado en tiempo real de funciones de onda 3D Psi(x,y,z,t)
usando pygame. Diseñado para proyección Pepper's Ghost con vidrio
horizontal a 45°.

Técnica de visualización: nube de puntos muestreados de |Psi(x,y,z,t)|²
mediante MCMC (Metropolis-Hastings), igual que atomic_realtime.py pero
para potenciales arbitrarios en 3D sin separabilidad garantizada.

Sistemas incluidos (ciclo de 60s, 20s cada uno):
  1. Pozo esférico infinito  — estados (n,l,m) analíticos, misma física
                               que el hidrógeno pero con energías E_nl
                               proporcionales a ceros de Bessel j_nl.
  2. Oscilador armónico 3D   — estados producto (nx,ny,nz), separable.
                               Muestra modos (1,0,0), (1,1,0), (1,1,1).
  3. Paquete de onda 3D      — superposición de dos estados del oscilador
                               que produce un paquete que oscila y rota.

Controles:
  ESPACIO   → pausar / reanudar
  R         → reiniciar el ciclo
  ←  /  →  → sistema anterior / siguiente
  + / -     → acelerar / ralentizar
  F         → fullscreen
  ESC / Q   → salir

Uso:
  python wavefunction3d_realtime.py
  python wavefunction3d_realtime.py --display 1 --fullscreen
"""

import sys
import os
import argparse
import math
import numpy as np
import pygame
from scipy.special import spherical_jn, sph_harm_y, genlaguerre, factorial
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from pepper_layout import blit_to_pepper

# ── Parámetros generales ─────────────────────────────────────
N_PUNTOS     = 3000    # puntos en la nube
N_BURN       = 400     # pasos de quemado MCMC
N_THIN       = 3       # adelgazamiento MCMC
N_FRAMES_CACHE = 12    # frames temporales precalculados por sistema
T_SISTEMA    = 20.0    # segundos por sistema
FPS_TARGET   = 60
SUPERSAMPLE  = 1.75
TILT         = math.radians(20)

HBAR = MASS = 1.0

COLOR_BG     = (0, 0, 0)
COLOR_POS    = (110, 200, 255)
COLOR_NEG    = (255, 150,  90)
COLOR_NUCLEO = (255, 250, 220)
COLOR_HUD    = (180, 180, 180)


# ═══════════════════════════════════════════════════════════════════════════
# Sistema 1: Pozo esférico infinito
# Psi_nlm(r,theta,phi) = A * j_l(z_nl * r/R) * Y_l^m(theta,phi)
# donde z_nl es el n-ésimo cero de j_l y R es el radio del pozo.
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_CEROS_BESSEL = {}

def cero_bessel(l, n, bracket=(0.1, 50.0)):
    """n-ésimo cero positivo de j_l(x), con cache."""
    key = (l, n)
    if key in _CACHE_CEROS_BESSEL:
        return _CACHE_CEROS_BESSEL[key]
    xs = np.linspace(bracket[0], bracket[1], 2000)
    vals = spherical_jn(l, xs)
    sign_changes = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_changes) < n:
        raise ValueError(f"No se encontró el cero {n} de j_{l}")
    x0, x1 = xs[sign_changes[n-1]], xs[sign_changes[n-1]+1]
    z = brentq(lambda x: spherical_jn(l, x), x0, x1)
    _CACHE_CEROS_BESSEL[key] = z
    return z


def psi_pozo_esferico(r, theta, phi, n=1, l=0, m=0, R=8.0):
    """
    Función de onda del pozo esférico infinito.
    Usa armónico esférico real (misma convención que atomic_realtime.py).
    """
    z_nl = cero_bessel(l, n)
    rho = z_nl * r / R
    radial = spherical_jn(l, rho)
    # Normalización radial aproximada
    radial = np.where(r < R, radial, 0.0)

    # Armónico esférico real (sph_harm_y usa orden: l, m, theta, phi)
    if m == 0:
        Y = np.real(sph_harm_y(l, 0, theta, phi))
    elif m > 0:
        Y = np.sqrt(2) * np.real(sph_harm_y(l, m, theta, phi))
    else:
        Y = np.sqrt(2) * np.imag(sph_harm_y(l, -m, theta, phi))

    return radial * Y


def energia_pozo(n, l, R=8.0):
    z_nl = cero_bessel(l, n)
    return HBAR**2 * z_nl**2 / (2 * MASS * R**2)


# ═══════════════════════════════════════════════════════════════════════════
# Sistema 2 y 3: Oscilador armónico 3D (separable)
# Psi_{nx,ny,nz}(x,y,z) = psi_nx(x) * psi_ny(y) * psi_nz(z)
# E_{nx,ny,nz} = (nx+ny+nz+3/2) * hbar * omega
# ═══════════════════════════════════════════════════════════════════════════

def psi_osc_1d(n, x, omega=1.0):
    """Autofunción del oscilador armónico 1D normalizada."""
    norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi / omega))
    xi = np.sqrt(omega) * x
    from scipy.special import eval_hermite
    return norm * eval_hermite(n, xi) * np.exp(-xi**2 / 2)


def psi_osc_3d(x, y, z, nx=0, ny=0, nz=0, omega=1.0):
    return psi_osc_1d(nx, x, omega) * psi_osc_1d(ny, y, omega) * psi_osc_1d(nz, z, omega)


def energia_osc(nx, ny, nz, omega=1.0):
    return (nx + ny + nz + 1.5) * HBAR * omega


# ═══════════════════════════════════════════════════════════════════════════
# Muestreo MCMC (Metropolis-Hastings) de |Psi(x,y,z)|^2
# ═══════════════════════════════════════════════════════════════════════════

def mcmc_sample_3d(psi_fn, n_points, burn=N_BURN, thin=N_THIN,
                   step=0.6, x0=None, seed=42):
    """
    Muestrea n_points puntos de la distribución |psi_fn(x,y,z)|^2
    mediante Metropolis-Hastings con propuesta gaussiana isótropa.

    Retorna array (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    pos = np.zeros(3) if x0 is None else np.array(x0, dtype=float)
    prob = psi_fn(*pos)**2

    samples = np.zeros((n_points, 3))
    total_steps = burn + n_points * thin
    count = 0

    for step_i in range(total_steps):
        proposal = pos + rng.normal(0, step, 3)
        prob_new  = psi_fn(*proposal)**2
        if prob_new > 0 and rng.random() < prob_new / max(prob, 1e-300):
            pos  = proposal
            prob = prob_new
        if step_i >= burn and (step_i - burn) % thin == 0:
            samples[count] = pos
            count += 1
            if count >= n_points:
                break

    return samples


# ═══════════════════════════════════════════════════════════════════════════
# Catálogo de sistemas
# ═══════════════════════════════════════════════════════════════════════════

def build_pozo_esferico():
    """
    Superposición de dos estados del pozo esférico para tener dinámica:
    (n=1,l=0,m=0) + (n=1,l=1,m=0). Rota entre forma esférica y dipolar.
    """
    E0 = energia_pozo(1, 0)
    E1 = energia_pozo(1, 1)
    dE = E1 - E0
    r_ref = 6.0

    # Factor que comprime el período físico real (2*pi/dE, muy largo)
    # al tiempo de demo deseado, de modo que se vea ~2.5 oscilaciones
    # completas en T_SISTEMA segundos.
    t_period_demo = T_SISTEMA / 2.5
    factor_tiempo = (2*np.pi / dE) / t_period_demo

    def psi_t(t_demo):
        t = t_demo * factor_tiempo
        c0 = np.cos(dE * t / 2)
        c1 = np.sin(dE * t / 2)
        def fn(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
            theta = np.arccos(np.clip(z/r, -1, 1))
            phi   = np.arctan2(y, x)
            return c0 * psi_pozo_esferico(r, theta, phi, 1, 0, 0) + \
                   c1 * psi_pozo_esferico(r, theta, phi, 1, 1, 0)
        return fn

    def signo_t(x, y, z, t_demo):
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
        theta = np.arccos(np.clip(z/r, -1, 1))
        phi   = np.arctan2(y, x)
        val = psi_pozo_esferico(r, theta, phi, 1, 0, 0) + \
              psi_pozo_esferico(r, theta, phi, 1, 1, 0)
        return np.where(np.abs(val) > 1e-12, np.sign(val), 1.0)

    return dict(
        nombre     = "Pozo esférico infinito 3D",
        descripcion= "Superposición (1,0,0)+(1,1,0) — oscila entre esfera y dipolo",
        psi_t      = psi_t,
        signo_t    = signo_t,
        dE         = dE,
        r_ref      = r_ref,
        step_mcmc  = 1.8,
        t_period   = t_period_demo,
    )


def build_oscilador_3d():
    """
    Estado (1,0,0) del oscilador armónico 3D — dos lóbulos en X.
    Estático pero se rota, por eso se ve en 3D.
    """
    E = energia_osc(1, 0, 0)
    r_ref = 4.0

    def psi_t(t):
        def fn(x, y, z): return psi_osc_3d(x, y, z, 1, 0, 0)
        return fn

    def signo_t(x, y, z, t):
        return np.sign(psi_osc_3d(x, y, z, 1, 0, 0))

    return dict(
        nombre     = "Oscilador armónico 3D — modo (1,0,0)",
        descripcion= "Dos lóbulos a lo largo del eje X, similar al orbital 2p_x",
        psi_t      = psi_t,
        signo_t    = signo_t,
        dE         = 1.0,
        r_ref      = r_ref,
        step_mcmc  = 1.2,
        t_period   = 2*np.pi,
    )


def build_paquete_3d():
    """
    Paquete oscilante 3D: superposición de (1,0,0) y (0,1,0).
    El centroide rota en el plano XY.
    """
    E10 = energia_osc(1, 0, 0)
    E01 = energia_osc(0, 1, 0)
    omega_x, omega_y = 1.0, 1.6   # anisotropía fuerte: período corto y visible
    dE  = abs(omega_x - omega_y)
    r_ref = 4.5

    def psi_t(t):
        phase = np.exp(-1j * dE * t)
        def fn(x, y, z):
            p1 = psi_osc_1d(1, x, omega_x) * psi_osc_1d(0, y, omega_y) * psi_osc_1d(0, z)
            p2 = psi_osc_1d(0, x, omega_x) * psi_osc_1d(1, y, omega_y) * psi_osc_1d(0, z)
            return np.real(p1 + phase * p2)
        return fn

    def signo_t(x, y, z, t):
        phase = np.cos(dE * t)
        p1 = psi_osc_1d(1, x, omega_x) * psi_osc_1d(0, y, omega_y) * psi_osc_1d(0, z)
        p2 = psi_osc_1d(0, x, omega_x) * psi_osc_1d(1, y, omega_y) * psi_osc_1d(0, z)
        val = p1 + phase * p2
        return np.where(np.abs(val) > 1e-12, np.sign(val), 1.0)

    return dict(
        nombre     = "Paquete de onda 3D — oscilador anisótropo",
        descripcion= "Superposición (1,0,0)+(0,1,0): centroide rota en plano XY",
        psi_t      = psi_t,
        signo_t    = signo_t,
        dE         = max(dE, 0.05),
        r_ref      = r_ref,
        step_mcmc  = 1.3,
        t_period   = 2*np.pi / max(dE, 0.05),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Proyección 3D → 2D (idéntica a atomic_realtime.py)
# ═══════════════════════════════════════════════════════════════════════════

def proyectar(pos3d, angulo_rot, tilt, cx, cy, escala):
    x0, y0, z0 = pos3d[:, 0], pos3d[:, 1], pos3d[:, 2]
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    y1 =  y0 * cos_t - z0 * sin_t
    z1 =  y0 * sin_t + z0 * cos_t
    cos_a, sin_a = math.cos(angulo_rot), math.sin(angulo_rot)
    x2 =  x0 * cos_a + z1 * sin_a
    z2 = -x0 * sin_a + z1 * cos_a
    y2 =  y1
    d_foco = 2.5
    persp = d_foco / (z2 + d_foco + 1e-5)
    px = cx + x2 * escala * persp
    py = cy - y2 * escala * persp
    return px, py, persp, z2


def crear_glow_template(size=40):
    surf = pygame.Surface((size, size))
    arr  = pygame.surfarray.pixels3d(surf)
    cx, cy = size/2, size/2
    for x in range(size):
        for y in range(size):
            d = math.hypot(x-cx, y-cy) / (size/2)
            arr[x, y, :] = int(max(0.0, 1.0-d)**2 * 255)
    del arr
    return surf


# ═══════════════════════════════════════════════════════════════════════════
# Renderer principal
# ═══════════════════════════════════════════════════════════════════════════

class Wave3DRenderer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.RW = int(W * SUPERSAMPLE)
        self.RH = int(H * SUPERSAMPLE)

        print("Construyendo sistemas 3D...")
        self.sistemas = [
            build_pozo_esferico(),
            build_oscilador_3d(),
            build_paquete_3d(),
        ]

        print("Precalculando nubes MCMC (secuencia de t para cada sistema)...")
        for i, s in enumerate(self.sistemas):
            t_vals = np.linspace(0, s["t_period"], N_FRAMES_CACHE, endpoint=False)
            nubes, signos_list = [], []
            for j, tv in enumerate(t_vals):
                psi_fn = s["psi_t"](tv)
                nube   = mcmc_sample_3d(psi_fn, N_PUNTOS,
                                        burn=N_BURN if j==0 else 150,
                                        step=s["step_mcmc"],
                                        seed=200 + i*100 + j)
                signos = s["signo_t"](nube[:,0], nube[:,1], nube[:,2], tv)
                nubes.append(nube / s["r_ref"])
                signos_list.append(np.asarray(signos, dtype=float))
            s["nubes_cache"]  = nubes
            s["signos_cache"] = signos_list
            s["t_vals_cache"] = t_vals
            print(f"  · {s['nombre'][:38]:<38} ({N_FRAMES_CACHE} frames) OK")

        self.t_sim         = 0.0
        self.pausado       = False
        self.velocidad     = 1.0
        self.angulo_rot    = 0.0
        self.indice_manual = None

        self.render_surf = pygame.Surface((self.RW, self.RH))
        self.model_surf  = pygame.Surface((self.W, self.H))

        glow = crear_glow_template(36)
        glow.fill(COLOR_NUCLEO, special_flags=pygame.BLEND_RGB_MULT)
        self.glow_nucleo = glow

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
        self.t_sim      += dt_real * self.velocidad
        self.angulo_rot += dt_real * 0.30 * self.velocidad
        if self.t_sim >= self.duracion_ciclo:
            self.t_sim = 0.0

    def cambiar_sistema(self, delta):
        idx  = self.indice_actual()
        nuevo = (idx + delta) % len(self.sistemas)
        self.indice_manual = nuevo
        self.t_sim = nuevo * T_SISTEMA

    def reanudar_automatico(self):
        self.indice_manual = None

    def _obtener_nube(self, idx, t_local):
        """Selecciona la nube precalculada más cercana al tiempo actual."""
        s      = self.sistemas[idx]
        t_eff  = t_local * (2*np.pi / s["t_period"]) % s["t_period"]
        t_vals = s["t_vals_cache"]
        j      = int(np.argmin(np.abs(t_vals - t_eff)))
        return s["nubes_cache"][j], s["signos_cache"][j]

    def dibujar_modelo(self, idx):
        nube, signos = self._obtener_nube(idx, self.t_local())
        s    = self.sistemas[idx]

        surf = self.render_surf
        surf.fill(COLOR_BG)
        cx, cy = self.RW//2, self.RH//2
        escala = min(self.RW, self.RH) * 0.32

        px, py, persp, z = proyectar(nube, self.angulo_rot, TILT,
                                     cx, cy, escala)
        orden = np.argsort(z)
        Wd, Hd = surf.get_size()

        for i in orden:
            x, y = px[i], py[i]
            if not (-10 <= x < Wd+10 and -10 <= y < Hd+10):
                continue
            base   = COLOR_POS if signos[i] > 0 else COLOR_NEG
            brillo = 0.5 + 0.5 * max(0.0, 1.0 - np.linalg.norm(nube[i]))
            color  = tuple(min(255, int(c*brillo)) for c in base)
            sz     = max(1, int(1.4 * persp[i] * SUPERSAMPLE))
            pygame.draw.circle(surf, color, (int(x), int(y)), sz)

        # Núcleo / origen
        orig = np.array([[0.0, 0.0, 0.0]])
        pxo, pyo, _, _ = proyectar(orig, self.angulo_rot, TILT, cx, cy, escala)
        nx, ny = int(pxo[0]), int(pyo[0])
        glow_sz = max(4, int(12*SUPERSAMPLE))
        gs = pygame.transform.smoothscale(self.glow_nucleo, (glow_sz, glow_sz))
        surf.blit(gs, (nx-glow_sz//2, ny-glow_sz//2),
                 special_flags=pygame.BLEND_ADD)
        pygame.draw.circle(surf, (255,255,255), (nx,ny), max(1,int(2*SUPERSAMPLE)))

        pygame.transform.smoothscale(surf, (self.W, self.H), self.model_surf)

    def dibujar_hud(self, screen, idx, fps):
        if not pygame.font.get_init():
            return
        try:
            font  = pygame.font.SysFont("monospace", 16)
            font2 = pygame.font.SysFont("monospace", 14)
        except Exception:
            return

        s = self.sistemas[idx]
        lineas = [
            s["nombre"],
            f"T_periodo = {s['t_period']:.2f}",
            f"vel x{self.velocidad:.1f}  {'⏸' if self.pausado else '▶'}",
            f"FPS {fps:.0f}",
        ]
        y = 10
        for li in lineas:
            screen.blit(font.render(li, True, COLOR_HUD), (10, y))
            y += 20
        screen.blit(font2.render(s["descripcion"], True, (140,140,140)), (10, y+4))

        leyenda = [("Fase +", COLOR_POS), ("Fase -", COLOR_NEG)]
        y = 10
        for nombre, color in leyenda:
            pygame.draw.circle(screen, color, (self.W-110, y+8), 5)
            screen.blit(font.render(nombre, True, (170,170,170)), (self.W-95, y))
            y += 20

        bw, bx, by = 240, 10, self.H-30
        prog = self.t_sim / self.duracion_ciclo
        pygame.draw.rect(screen, (40,40,40), (bx, by, bw, 8))
        pygame.draw.rect(screen, (80,160,255), (bx, by, int(bw*prog), 8))
        for k in range(1, len(self.sistemas)):
            xk = bx + int(bw*k/len(self.sistemas))
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

    flags  = (pygame.FULLSCREEN | pygame.NOFRAME) if args.fullscreen else 0
    screen = pygame.display.set_mode((args.width, args.height), flags)
    pygame.display.set_caption("Funciones de Onda 3D — Pepper Ghost")

    renderer   = Wave3DRenderer(args.width, args.height)
    clock      = pygame.time.Clock()
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
