"""
models/wavefunction/schrodinger_1d.py

Solución numérica de la ecuación de Schrödinger independiente del tiempo en 1D:

    -ħ²/(2m) · d²ψ/dx² + V(x)·ψ = E·ψ

Mediante el método de diferencias finitas (matriz tridiagonal).
Incluye potenciales: pozo infinito, oscilador armónico y doble pozo.

Referencia:
    Griffiths, D.J. (2018). Introduction to Quantum Mechanics, 3rd ed.
    Capítulos 2-3.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal


# Constantes (unidades atómicas: ħ=1, m=1, a0=1)
HBAR = 1.0
MASS = 1.0


def infinite_well(x: np.ndarray, L: float = 10.0) -> np.ndarray:
    """Pozo de potencial infinito de ancho L."""
    V = np.full_like(x, np.inf)
    inside = (x > 0) & (x < L)
    V[inside] = 0.0
    return V


def harmonic_oscillator(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """Oscilador armónico V(x) = ½mω²x²."""
    return 0.5 * MASS * omega**2 * x**2


def double_well(
    x: np.ndarray, a: float = 2.0, b: float = 0.1
) -> np.ndarray:
    """Doble pozo V(x) = -a·x² + b·x⁴."""
    return -a * x**2 + b * x**4


def solve_schrodinger_1d(
    potential_fn,
    x_range: tuple = (-10.0, 10.0),
    n_points: int = 500,
    n_states: int = 6,
    **potential_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve la ecuación de Schrödinger 1D por diferencias finitas.

    Parámetros
    ----------
    potential_fn    : función V(x, **kwargs)
    x_range         : dominio espacial
    n_points        : número de puntos de la grilla
    n_states        : número de eigenstados a calcular
    **potential_kwargs : parámetros adicionales para potential_fn

    Retorna
    -------
    x          : array de posiciones
    energies   : eigenvalores E_n (n_states,)
    wavefuncs  : eigenfunciones ψ_n(x) (n_points, n_states)
    """
    x = np.linspace(*x_range, n_points)
    dx = x[1] - x[0]

    V = potential_fn(x, **potential_kwargs)
    # Reemplazar infinitos por barrera alta finita
    V_finite = np.where(np.isinf(V), 1e6, V)

    # Diagonal principal y superdiagonal del Hamiltoniano tridiagonal
    diag_main = HBAR**2 / (MASS * dx**2) + V_finite
    diag_off = -HBAR**2 / (2 * MASS * dx**2) * np.ones(n_points - 1)

    energies, wavefuncs = eigh_tridiagonal(
        diag_main, diag_off, select="i", select_range=(0, n_states - 1)
    )

    # Normalizar
    for i in range(n_states):
        norm = np.sqrt(np.trapz(wavefuncs[:, i] ** 2, x))
        wavefuncs[:, i] /= norm

    return x, energies, wavefuncs


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    configs = [
        (infinite_well, {"L": 10.0}, (-0.5, 10.5), "Pozo infinito"),
        (harmonic_oscillator, {"omega": 1.0}, (-8, 8), "Oscilador armónico"),
        (double_well, {"a": 2.0, "b": 0.1}, (-6, 6), "Doble pozo"),
    ]

    for ax, (pot_fn, kwargs, x_range, title) in zip(axes, configs):
        x, E, psi = solve_schrodinger_1d(
            pot_fn, x_range=x_range, n_states=4, **kwargs
        )
        V = pot_fn(x, **kwargs)
        V_plot = np.clip(V, -20, 40)

        ax.plot(x, V_plot, "k-", lw=1.5, label="V(x)")
        colors = ["royalblue", "tomato", "seagreen", "darkorange"]
        for i in range(4):
            offset = E[i]
            ax.plot(x, 3 * psi[:, i] + offset, color=colors[i],
                    label=f"n={i}, E={E[i]:.2f}")
            ax.axhline(offset, color=colors[i], ls="--", alpha=0.4)

        ax.set_title(title)
        ax.set_xlabel("x (a₀)")
        ax.set_ylabel("Energía (Eₕ)")
        ax.legend(fontsize=7)
        ax.set_ylim(-5, max(E) * 1.5)

    plt.tight_layout()
    plt.savefig("simulation/output/schrodinger_1d.png", dpi=150)
    print("Figura guardada en simulation/output/schrodinger_1d.png")
