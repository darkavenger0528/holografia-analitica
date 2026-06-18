"""
run_model.py  —  Launcher del sistema de proyección holográfica

Uso:
  python run_model.py                           # menú interactivo
  python run_model.py universe                  # universo, pantalla principal
  python run_model.py atomic --display 1        # orbital, pantalla HDMI
  python run_model.py wavefunction --fullscreen # función de onda, fullscreen
  python run_model.py demo --display 1          # demo: los 3 modelos en secuencia

Argumentos opcionales:
  --display N       pantalla de salida (0=principal, 1=HDMI). Default: 0
  --width W         ancho en píxeles.  Default: 1280
  --height H        alto en píxeles.   Default: 720
  --fullscreen      modo pantalla completa
"""

import sys
import os
import subprocess
import argparse
import time

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "simulation", "engine")

# ── Catálogo de modelos ───────────────────────────────────────────────────
MODELOS = {
    "universe": dict(
        script  = "universe_realtime.py",
        desc    = "Expansión del universo FLRW",
        detalle = "Soles, planetas, asteroides y galaxias · Big Bang → t=30 Gyr · 60s",
        controles = "ESPACIO=pausa  R=reiniciar  +/-=velocidad",
    ),
    "atomic": dict(
        script  = "atomic_realtime.py",
        desc    = "Orbitales atómicos del hidrógeno",
        detalle = "Nube |ψ_nlm|² · 1s→2s→2p→3d · 10s por orbital · ←/→=cambiar",
        controles = "ESPACIO=pausa  R=reiniciar  ←/→=orbital  +/-=velocidad",
    ),
    "wavefunction": dict(
        script  = "wavefunction_realtime.py",
        desc    = "Función de onda Ψ(x,t) — 1D",
        detalle = "Pozo infinito · Oscilador armónico · Doble pozo · 20s c/u",
        controles = "ESPACIO=pausa  R=reiniciar  ←/→=sistema  +/-=velocidad",
    ),
    "wavefunction3d": dict(
        script  = "wavefunction3d_realtime.py",
        desc    = "Función de onda Ψ(x,y,z,t) — 3D",
        detalle = "Pozo esférico · Oscilador 3D · Paquete anisótropo · 20s c/u (MCMC)",
        controles = "ESPACIO=pausa  R=reiniciar  ←/→=sistema  +/-=velocidad",
    ),
}

DEMO_ORDEN   = ["universe", "atomic", "wavefunction", "wavefunction3d"]
DEMO_DURACION = 60  # segundos por modelo en modo demo (usar Ctrl+C para avanzar)


# ── Utilidades ────────────────────────────────────────────────────────────

def disponible(modelo: str) -> bool:
    return os.path.exists(os.path.join(ENGINE_DIR, MODELOS[modelo]["script"]))


def banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║      PROYECCIÓN HOLOGRÁFICA  —  Capítulo de Óptica  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def menu_interactivo():
    banner()
    print("  Modelos disponibles:\n")
    keys = list(MODELOS.keys())
    for i, key in enumerate(keys, 1):
        m = MODELOS[key]
        estado = "✓" if disponible(key) else "⏳"
        print(f"  [{i}] {estado}  {key:<14}  {m['desc']}")
        print(f"        {m['detalle']}")
        print()

    print("  [D]  Demo automático (los 3 modelos en secuencia)")
    print()

    display_str = input("  Pantalla (0=principal, 1=HDMI) [0]: ").strip() or "0"
    width_str   = input("  Ancho px [1280]: ").strip() or "1280"
    height_str  = input("  Alto px  [720]:  ").strip() or "720"
    full_str    = input("  Fullscreen? (s/n) [n]: ").strip().lower()
    print()
    opcion = input("  Elige modelo (nombre, número o D para demo): ").strip()

    try:
        display = int(display_str)
        width   = int(width_str)
        height  = int(height_str)
    except ValueError:
        print("Valor inválido.")
        sys.exit(1)

    fullscreen = full_str == "s"

    if opcion.lower() == "d":
        return "demo", display, width, height, fullscreen

    if opcion.isdigit() and 1 <= int(opcion) <= len(keys):
        modelo = keys[int(opcion) - 1]
    elif opcion in MODELOS:
        modelo = opcion
    else:
        print(f"Opción no reconocida: '{opcion}'")
        sys.exit(1)

    return modelo, display, width, height, fullscreen


# ── Lanzador ──────────────────────────────────────────────────────────────

def build_cmd(modelo: str, display: int, width: int, height: int,
              fullscreen: bool) -> list:
    script = os.path.join(ENGINE_DIR, MODELOS[modelo]["script"])
    cmd = [sys.executable, script,
           "--display", str(display),
           "--width",   str(width),
           "--height",  str(height)]
    if fullscreen:
        cmd.append("--fullscreen")
    return cmd


def lanzar(modelo: str, display: int, width: int, height: int,
           fullscreen: bool):
    if not disponible(modelo):
        print(f"⏳  El modelo '{modelo}' todavía no está implementado.")
        sys.exit(0)

    m = MODELOS[modelo]
    print(f"  Modelo   : {m['desc']}")
    print(f"  Detalle  : {m['detalle']}")
    print(f"  Controles: {m['controles']}  F=fullscreen  ESC=salir")
    print(f"  Pantalla : display={display}  {width}×{height}"
          f"  {'FULLSCREEN' if fullscreen else 'ventana'}")
    print()

    cmd = build_cmd(modelo, display, width, height, fullscreen)
    print(f"  Ejecutando: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def lanzar_demo(display: int, width: int, height: int, fullscreen: bool):
    """Lanza los 3 modelos en secuencia. Ctrl+C en cualquier momento avanza
    al siguiente. Al terminar el último, el ciclo termina."""
    banner()
    print(f"  Modo DEMO — {len(DEMO_ORDEN)} modelos, {DEMO_DURACION}s por modelo")
    print(f"  Presiona Ctrl+C dentro de cada modelo para avanzar al siguiente.\n")

    for i, modelo in enumerate(DEMO_ORDEN, 1):
        if not disponible(modelo):
            print(f"  [{i}/{len(DEMO_ORDEN)}] ⏳  '{modelo}' no disponible, saltando...")
            continue

        m = MODELOS[modelo]
        print(f"  [{i}/{len(DEMO_ORDEN)}] Iniciando: {m['desc']}...")
        cmd = build_cmd(modelo, display, width, height, fullscreen)
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print(f"\n  Avanzando al siguiente modelo...")
        print()

    print("  ✓ Demo completado.")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launcher holográfico — Capítulo de Óptica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  python run_model.py                        menú interactivo
  python run_model.py universe               universo en pantalla principal
  python run_model.py atomic --display 1     orbitales en pantalla HDMI
  python run_model.py wavefunction --fullscreen
  python run_model.py demo --display 1       demo automático en HDMI
        """)

    parser.add_argument("modelo", nargs="?",
                        choices=list(MODELOS.keys()) + ["demo"],
                        help="modelo a ejecutar (o 'demo')")
    parser.add_argument("--display",    type=int,  default=0,
                        help="índice de pantalla (0=principal, 1=HDMI)")
    parser.add_argument("--width",      type=int,  default=1280)
    parser.add_argument("--height",     type=int,  default=720)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    if args.modelo:
        if args.modelo == "demo":
            lanzar_demo(args.display, args.width, args.height, args.fullscreen)
        else:
            lanzar(args.modelo, args.display, args.width, args.height,
                   args.fullscreen)
    else:
        resultado = menu_interactivo()
        modelo, display, width, height, fullscreen = resultado
        if modelo == "demo":
            lanzar_demo(display, width, height, fullscreen)
        else:
            lanzar(modelo, display, width, height, fullscreen)


if __name__ == "__main__":
    main()
