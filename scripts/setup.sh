#!/usr/bin/env bash
# setup.sh — Configura el entorno de desarrollo del proyecto

set -e

echo "==> Verificando Python 3.11+..."
python3 --version

echo "==> Creando entorno virtual..."
python3 -m venv .venv

echo "==> Activando entorno e instalando dependencias..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Entorno listo."
echo "  Activa el entorno con: source .venv/bin/activate"
