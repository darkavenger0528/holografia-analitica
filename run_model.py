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
