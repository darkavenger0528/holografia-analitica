"""
models/universe/flrw_expansion.py

Modelo de expansión cosmológica basado en la métrica FLRW
(Friedmann-Lemaître-Robertson-Walker).

La ecuación de Friedmann describe la evolución del factor de escala a(t):

    H²(t) = (ȧ/a)² = (8πG/3)ρ - k/a² + Λ/3

Para un universo plano (k=0) dominado por materia, radiación y energía oscura.

Referencia:
    Ryden, B. (2017). Introduction to Cosmology, 2nd ed. Cambridge.
"""

import numpy as np
from scipy.integrate import solve_ivp


# Constantes cosmológicas (Planck 2018)
H0 = 67.4          # km/s/Mpc — constante de Hubble actual
OMEGA_M = 0.315    # parámetro de densidad de materia
OMEGA_R = 9.24e-5  # parámetro de densidad de radiación
OMEGA_LAMBDA = 0.685  # parámetro de energía oscura
OMEGA_K = 1 - OMEGA_M - OMEGA_R - OMEGA_LAMBDA  # curvatura (~0)


def hubble_parameter(a: float | np.ndarray) -> float | np.ndarray:
    """
    H(a) / H0 — parámetro de Hubble normalizado en función del factor de escala.

    Parámetros
    ----------
    a : factor de escala (a=1 hoy, a<1 en el pasado)
    """
    return np.sqrt(
        OMEGA_R / a**4
        + OMEGA_M / a**3
        + OMEGA_K / a**2
        + OMEGA_LAMBDA
    )


def friedmann_rhs(t: float, y: list) -> list:
    """RHS del sistema ODE: dy/dt = [ȧ, ä] → usado por solve_ivp."""
    a = y[0]
    a = max(a, 1e-10)
    adot = H0 * a * hubble_parameter(a)  # ȧ = H(a)·a   (simplificado)
    return [adot]


def integrate_scale_factor(
    a0: float = 1e-4,
    t_span_Gyr: tuple = (0.0, 30.0),
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integra la ecuación de Friedmann para obtener a(t).

    Parámetros
    ----------
    a0        : factor de escala inicial
    t_span_Gyr: intervalo de tiempo en Giga-años
    n_points  : número de puntos de evaluación

    Retorna
    -------
    t   : array de tiempos en Gyr
    a   : array del factor de escala a(t)
    """
    # Conversión de H0 a unidades de 1/Gyr
    H0_inv_Gyr = H0 * 1.022e-3  # km/s/Mpc → 1/Gyr

    t_eval = np.linspace(*t_span_Gyr, n_points)
    sol = solve_ivp(
        friedmann_rhs,
        t_span_Gyr,
        [a0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.t, sol.y[0]


def comoving_distance_grid(
    grid_size: int = 60,
    t_steps: int = 50,
    max_dist_Mpc: float = 4000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera una grilla 3D de partículas comoving que se expanden con a(t).
    Útil para visualización holográfica de la expansión.

    Retorna
    -------
    t        : tiempos en Gyr (t_steps,)
    a        : factor de escala a(t)
    positions: posiciones físicas (t_steps, N_particles, 3) en Mpc
    """
    t, a = integrate_scale_factor(n_points=t_steps)

    # Distribución inicial aleatoria de "galaxias" en coordenadas comoving
    rng = np.random.default_rng(seed=42)
    n_particles = grid_size**3
    comoving_pos = rng.uniform(-max_dist_Mpc, max_dist_Mpc, (n_particles, 3))

    # Posiciones físicas: r_phys = a(t) * r_comov
    positions = a[:, None, None] * comoving_pos[None, :, :]  # (t, N, 3)

    return t, a, positions


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, a = integrate_scale_factor()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, a, color="steelblue", lw=2)
    axes[0].axvline(13.8, color="gray", ls="--", label="Hoy (13.8 Gyr)")
    axes[0].set_xlabel("Tiempo (Gyr)")
    axes[0].set_ylabel("Factor de escala a(t)")
    axes[0].set_title("Expansión del universo FLRW")
    axes[0].legend()

    H_vals = hubble_parameter(a) * 67.4
    axes[1].plot(t, H_vals, color="darkorange", lw=2)
    axes[1].axvline(13.8, color="gray", ls="--")
    axes[1].set_xlabel("Tiempo (Gyr)")
    axes[1].set_ylabel("H(t) [km/s/Mpc]")
    axes[1].set_title("Parámetro de Hubble H(t)")

    plt.tight_layout()
    plt.savefig("simulation/output/flrw_expansion.png", dpi=150)
    print("Figura guardada en simulation/output/flrw_expansion.png")
