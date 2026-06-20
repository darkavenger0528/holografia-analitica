# Holografía Analógica — Capítulo Estudiantil de Óptica

Proyecto de visualización holográfica de modelos matemáticos complejos. El objetivo es proyectar representaciones tridimensionales de fenómenos físicos difíciles de comprender visualmente —como orbitales atómicos, expansión del universo y funciones de onda cuánticas— en tiempo real mediante holografía analógica.

---

## Estructura del repositorio

```
holografia-analitica/
├── models/               # Modelos matemáticos y simulaciones numéricas
│   ├── atomic/           # Modelo atómico y orbitales
│   ├── universe/         # Expansión del universo (métrica FLRW, etc.)
│   ├── wavefunction/     # Funciones de onda / Mecánica cuántica
│   └── other/            # Otros modelos en exploración
├── simulation/           # Motor de simulación y renderizado
│   ├── engine/           # Core del motor (proyección, cálculo de franjas)
│   ├── output/           # Salidas generadas (patrones de interferencia, etc.)
│   └── tests/            # Tests unitarios y de integración
├── assets/               # Recursos estáticos
│   ├── shaders/          # Shaders para visualización en tiempo real
│   ├── textures/         # Texturas auxiliares
│   └── exports/          # Exportaciones listas para proyección holográfica
├── docs/                 # Documento de investigación
│   ├── paper/            # Fuente LaTeX del artículo/informe
│   ├── references/       # Bibliografía (.bib) y PDFs de referencia
│   └── figures/          # Figuras generadas para el paper
└── scripts/              # Scripts de utilidad (setup, build, export)
```

---

## Modelos implementados

| Modelo | Estado | Descripción |
|---|---|---|
| Expansión del universo | ✅ Listo (tiempo real) | Métrica FLRW, soles/planetas/asteroides/galaxias, factor de escala a(t) |
| Orbital atómico (H) | ✅ Listo (tiempo real) | Nube de probabilidad \|ψ_nlm\|², ciclo 1s→2s→2p→3d |
| Función de onda 1D | ✅ Listo (tiempo real) | Ψ(x,t) compleja: pozo infinito, oscilador armónico, doble pozo |
| Función de onda 3D | ✅ Listo (tiempo real) | Ψ(x,y,z,t) por muestreo MCMC: pozo esférico, oscilador 3D, paquete anisótropo |
| Agujero negro | ✅ Listo (tiempo real, GPU) | Lente gravitacional, horizonte de eventos, disco de acreción con Doppler (shader GLSL) |

---

## Requisitos

```bash
# Python 3.11+
pip install -r requirements.txt
```

Ver `scripts/setup.sh` para instalación completa.

---

## Cómo contribuir

1. Clona el repo: `git clone git@github.com:TU_USER/holografia-analitica.git`
2. Crea una rama para tu feature: `git checkout -b feature/nombre-del-modelo`
3. Haz commit de tus cambios con mensajes descriptivos
4. Abre un Pull Request hacia `main`

Convención de commits:
- `feat:` nueva funcionalidad o modelo
- `fix:` corrección de errores
- `docs:` cambios en documentación o paper
- `sim:` cambios en el motor de simulación
- `refactor:` refactorización sin cambio funcional

---

## Equipo

Capítulo Estudiantil de Óptica — [Universidad]

---

## Licencia

MIT License — ver `LICENSE`

---

## Cómo ejecutar los modelos

```bash
# Activar entorno
source .venv/bin/activate

# Menú interactivo
python run_model.py

# Directo en pantalla principal (desarrollo/pruebas)
python run_model.py universe

# En pantalla HDMI (Android como display externo)
python run_model.py universe --display 1 --fullscreen
```

### Controles durante la proyección

| Tecla | Acción |
|---|---|
| `ESPACIO` | Pausar / reanudar |
| `R` | Reiniciar desde el Big Bang |
| `+` / `-` | Acelerar / ralentizar la simulación |
| `F` | Fullscreen |
| `ESC` | Salir |
