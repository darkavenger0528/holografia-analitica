#!/usr/bin/env bash
# ============================================================
# setup_proyecto.sh
# Crea todos los archivos nuevos del proyecto holografia-analitica
# y los commitea al repo.
# Ejecutar desde: ~/holografia-analitica
# ============================================================

set -e

REPO_DIR="$HOME/holografia-analitica"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "ERROR: No se encontró el repo en $REPO_DIR"
  echo "Asegúrate de estar dentro de ~/holografia-analitica"
  exit 1
fi

cd "$REPO_DIR"
echo "==> Trabajando en: $REPO_DIR"

# ── Crear directorios necesarios ─────────────────────────────
mkdir -p simulation/engine

# ============================================================
# ARCHIVO 1: simulation/engine/pepper_layout.py
# ============================================================
cat > simulation/engine/pepper_layout.py << 'EOF'
"""
simulation/engine/pepper_layout.py

Composición del layout para proyección Pepper's Ghost con vidrio
horizontal a 45°. La pantalla está boca abajo, el vidrio encima.

Para este setup solo se necesita UNA vista centrada sobre fondo negro.
El vidrio a 45° refleja la imagen hacia el observador que mira de frente.

La imagen proyectada debe estar:
- Fondo completamente negro (el negro no refleja → transparente)
- Contenido brillante centrado (colores saturados, alto contraste)
- Orientación: normal si la pantalla está boca arriba mirando al vidrio
"""

import pygame

# Proporción del área activa respecto al canvas total
CONTENT_RATIO = 0.75   # 75% del ancho/alto para el modelo


def create_pepper_surface(screen_w: int, screen_h: int):
    """
    Crea la superficie base negra y el rectángulo donde
    debe dibujarse el modelo.

    Retorna
    -------
    base    : Surface negra del tamaño de la pantalla
    content : Rect centrado donde va el modelo
    """
    base = pygame.Surface((screen_w, screen_h))
    base.fill((0, 0, 0))

    cw = int(screen_w * CONTENT_RATIO)
    ch = int(screen_h * CONTENT_RATIO)
    cx = (screen_w - cw) // 2
    cy = (screen_h - ch) // 2

    content_rect = pygame.Rect(cx, cy, cw, ch)
    return base, content_rect


def blit_to_pepper(screen, model_surface, screen_w: int, screen_h: int):
    """
    Pega el modelo renderizado en el centro de la pantalla
    sobre fondo negro. Llama a esto en cada frame.
    """
    screen.fill((0, 0, 0))

    cw = int(screen_w * CONTENT_RATIO)
    ch = int(screen_h * CONTENT_RATIO)
    scaled = pygame.transform.smoothscale(model_surface, (cw, ch))

    cx = (screen_w - cw) // 2
    cy = (screen_h - ch) // 2
    screen.blit(scaled, (cx, cy))
EOF

# ============================================================
# ARCHIVO 2: simulation/engine/universe_realtime.py
# ============================================================
cat > simulation/engine/universe_realtime.py << 'EOF'
"""
simulation/engine/universe_realtime.py

Renderizado en tiempo real de la expansión del universo (FLRW)
usando pygame. Diseñado para proyección Pepper's Ghost con
vidrio horizontal a 45°.

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

# Importar layout (mismo directorio)
sys.path.insert(0, os.path.dirname(__file__))
from pepper_layout import blit_to_pepper

# ── Parámetros cosmológicos (Planck 2018) ──────────────────
H0_GYR  = 67.4 * 1.022e-3
OMEGA_M = 0.315
OMEGA_R = 9.24e-5
OMEGA_L = 0.685
T_HOY   = 13.8

# ── Parámetros de simulación ────────────────────────────────
N_GALAXIAS   = 300
T_MIN        = 0.5
T_MAX        = 30.0
DURACION_SEG = 60.0
FPS_TARGET   = 60

COLOR_BG         = (0, 0, 0)
COLOR_OBSERVADOR = (255, 255, 255)
COLOR_HOY_LINE   = (255, 200, 50)


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


def generar_galaxias(n=N_GALAXIAS, seed=42):
    rng = np.random.default_rng(seed)
    theta = np.arccos(rng.uniform(-1, 1, n))
    phi   = rng.uniform(0, 2 * math.pi, n)
    r     = rng.uniform(0.05, 1.0, n) ** (1/3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pos = np.stack([x, y, z], axis=1)

    colores = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        t_c = r[i]
        colores[i] = (int(255*t_c), int(100*(1-t_c)), int(255*(1-t_c)))
    return pos, colores


def proyectar(pos3d, a, angulo_rot, cx, cy, escala):
    pos = pos3d * a
    cos_a, sin_a = math.cos(angulo_rot), math.sin(angulo_rot)
    x =  pos[:, 0] * cos_a + pos[:, 2] * sin_a
    z = -pos[:, 0] * sin_a + pos[:, 2] * cos_a
    y =  pos[:, 1]

    d_foco = 2.5
    perspectiva = d_foco / (z + d_foco + 1e-5)
    px = (cx + x * escala * perspectiva).astype(int)
    py = (cy - y * escala * perspectiva).astype(int)
    sz = np.clip(perspectiva * 4, 1, 8).astype(int)
    return np.stack([px, py, sz], axis=1)


class UniverseRenderer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        print("Precalculando a(t)...")
        self.t_arr, self.a_arr = precompute_scale_factor()
        print("Generando galaxias...")
        self.pos_comov, self.colores = generar_galaxias()
        self.t_sim     = T_MIN
        self.pausado   = False
        self.velocidad = 1.0
        self.angulo_rot = 0.0
        self.model_surf = pygame.Surface((W, H))

    def a_en(self, t):
        return float(np.interp(t, self.t_arr, self.a_arr))

    def avanzar(self, dt_real):
        if self.pausado:
            return
        rango = T_MAX - T_MIN
        self.t_sim += (rango / DURACION_SEG) * dt_real * self.velocidad
        self.angulo_rot += dt_real * 0.3 * self.velocidad
        if self.t_sim >= T_MAX:
            self.t_sim = T_MIN

    def dibujar_modelo(self, a):
        surf = self.model_surf
        surf.fill(COLOR_BG)
        cx, cy = self.W // 2, self.H // 2
        escala = min(self.W, self.H) * 0.35

        pts = proyectar(self.pos_comov, a, self.angulo_rot, cx, cy, escala)
        z_vals = self.pos_comov[:, 2] * a
        orden  = np.argsort(z_vals)

        for i in orden:
            px, py, sz = pts[i]
            if 0 <= px < self.W and 0 <= py < self.H:
                brillo = min(1.0, a * 0.8 + 0.2)
                color  = tuple(int(c * brillo) for c in self.colores[i])
                pygame.draw.circle(surf, color, (px, py), sz)

        pygame.draw.circle(surf, COLOR_OBSERVADOR, (cx, cy), 5)

        radio_h = int(escala * a * 0.95)
        if radio_h > 0:
            color_h = COLOR_HOY_LINE if abs(self.t_sim - T_HOY) < 0.5 \
                      else (30, 30, 80)
            pygame.draw.circle(surf, color_h, (cx, cy), radio_h, 2)

    def dibujar_hud(self, screen, a, fps):
        font = pygame.font.SysFont("monospace", 16)
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
    pygame.font.init()

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
EOF

# ============================================================
# ARCHIVO 3: run_model.py
# ============================================================
cat > run_model.py << 'EOF'
"""
run_model.py — Launcher del sistema de proyección holográfica

Uso:
  python run_model.py                       # menú interactivo
  python run_model.py universe              # pantalla principal
  python run_model.py universe --display 1  # pantalla HDMI
  python run_model.py universe --display 1 --fullscreen
"""

import sys
import os
import subprocess
import argparse

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "simulation", "engine")

MODELOS = {
    "universe":     ("universe_realtime.py",    "Expansión del universo FLRW"),
    "atomic":       ("atomic_realtime.py",       "Orbital atómico del hidrógeno"),
    "wavefunction": ("wavefunction_realtime.py", "Función de onda 1D"),
}


def menu():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   PROYECCIÓN HOLOGRÁFICA — ÓPTICA CHAPTER    ║")
    print("╚══════════════════════════════════════════════╝\n")
    for i, (key, (script, desc)) in enumerate(MODELOS.items(), 1):
        ok = "✓" if os.path.exists(os.path.join(ENGINE_DIR, script)) \
             else "⏳ próximamente"
        print(f"  [{i}] {key:<14} {desc}  {ok}")
    print()
    display = input("Pantalla (0=principal, 1=HDMI) [0]: ").strip() or "0"
    full    = input("Fullscreen? (s/n) [n]: ").strip().lower() == "s"
    opcion  = input("Elige modelo (nombre o número): ").strip()
    keys = list(MODELOS.keys())
    if opcion.isdigit() and 1 <= int(opcion) <= len(keys):
        return keys[int(opcion)-1], int(display), full
    elif opcion in MODELOS:
        return opcion, int(display), full
    print("Modelo no reconocido.")
    sys.exit(1)


def lanzar(modelo, display, fullscreen):
    script = os.path.join(ENGINE_DIR, MODELOS[modelo][0])
    if not os.path.exists(script):
        print(f"⏳ El modelo '{modelo}' aún no está implementado.")
        sys.exit(0)
    cmd = [sys.executable, script, "--display", str(display)]
    if fullscreen:
        cmd.append("--fullscreen")
    print(f"\nLanzando: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelo", nargs="?", choices=list(MODELOS.keys()))
    parser.add_argument("--display",    type=int, default=0)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()
    if args.modelo:
        lanzar(args.modelo, args.display, args.fullscreen)
    else:
        lanzar(*menu())


if __name__ == "__main__":
    main()
EOF

# ── Crear venv e instalar dependencias ───────────────────────
echo ""
echo "==> Creando entorno virtual..."
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install numpy scipy matplotlib pygame -q
echo "==> Dependencias instaladas."

# ── Git: agregar y commitear ─────────────────────────────────
echo ""
echo "==> Commiteando archivos nuevos..."
git add simulation/engine/pepper_layout.py \
        simulation/engine/universe_realtime.py \
        run_model.py \
        requirements.txt \
        README.md

git commit -m "feat(engine): renderer en tiempo real para Pepper's Ghost

- pepper_layout.py: layout para vidrio horizontal 45°
- universe_realtime.py: universo FLRW en tiempo real con pygame
  · 300 galaxias con corrimiento al rojo
  · Rotación 3D automática con perspectiva
  · Horizonte de Hubble visual
  · Controles: pausa, reinicio, velocidad, fullscreen
  · Soporte --display 1 para HDMI (Android)
  · Ciclo de 60s del Big Bang hasta t=30 Gyr
- run_model.py: launcher con menú interactivo
- requirements.txt: agrega pygame>=2.6.0"

git push

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✓ Todo listo. Para ejecutar:                ║"
echo "║    source .venv/bin/activate                 ║"
echo "║    python run_model.py universe              ║"
echo "╚══════════════════════════════════════════════╝"
