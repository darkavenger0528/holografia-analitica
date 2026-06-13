"""
models/atomic/hydrogen_orbital.py

Calcula la densidad de probabilidad |ψ(r,θ,φ)|² de los orbitales
del átomo de hidrógeno a partir de los números cuánticos (n, l, m).

Referencia:
    Griffiths, D.J. (2018). Introduction to Quantum Mechanics, 3rd ed.
    Capítulo 4, sección 4.2.
"""

import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial


def radial_wavefunction(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """
    Parte radial de la función de onda R_nl(r).

    Parámetros
    ----------
    n : número cuántico principal (n >= 1)
    l : número cuántico azimutal (0 <= l < n)
    r : array de distancias radiales en unidades de a0 (radio de Bohr)

    Retorna
    -------
    R_nl(r) : array de la misma forma que r
    """
    assert 1 <= n, "n debe ser >= 1"
    assert 0 <= l < n, "l debe cumplir 0 <= l < n"

    # Prefactor de normalización
    rho = 2 * r / n
    norm = np.sqrt(
        (2 / n) ** 3
        * factorial(n - l - 1)
        / (2 * n * factorial(n + l) ** 3)
    )
    laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return norm * np.exp(-rho / 2) * rho ** l * laguerre_poly


def orbital_density(
    n: int,
    l: int,
    m: int,
    grid_size: int = 100,
    extent: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula |ψ_nlm(x,y,z)|² en una grilla 3D cartesiana.

    Parámetros
    ----------
    n, l, m : números cuánticos
    grid_size : número de puntos por eje
    extent    : extensión de la grilla en unidades de a0

    Retorna
    -------
    X, Y, Z : mallas cartesianas
    density  : densidad de probabilidad |ψ|²
    """
    assert -l <= m <= l, "m debe cumplir -l <= m <= l"

    coords = np.linspace(-extent, extent, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Coordenadas esféricas
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r = np.where(r == 0, 1e-10, r)          # evitar división por cero
    theta = np.arccos(np.clip(Z / r, -1, 1))
    phi = np.arctan2(Y, X)

    R = radial_wavefunction(n, l, r)
    Y_lm = sph_harm(m, l, phi, theta)       # scipy: sph_harm(m, l, phi, theta)

    psi = R * Y_lm
    density = np.abs(psi) ** 2

    return X, Y, Z, density


if __name__ == "__main__":
    # Ejemplo: orbital 2p_z (n=2, l=1, m=0)
    import matplotlib.pyplot as plt

    X, Y, Z, density = orbital_density(n=2, l=1, m=0, grid_size=80, extent=15.0)

    # Corte en el plano XZ (y=0, índice central)
    mid = density.shape[1] // 2
    slice_xz = density[:, mid, :]

    plt.figure(figsize=(6, 6))
    plt.title("Orbital 2p_z — corte plano XZ")
    plt.imshow(
        slice_xz.T,
        origin="lower",
        extent=[-15, 15, -15, 15],
        cmap="inferno",
    )
    plt.colorbar(label="|ψ|²")
    plt.xlabel("x (a₀)")
    plt.ylabel("z (a₀)")
    plt.tight_layout()
    plt.savefig("simulation/output/orbital_2p0_xz.png", dpi=150)
    print("Figura guardada en simulation/output/orbital_2p0_xz.png")
