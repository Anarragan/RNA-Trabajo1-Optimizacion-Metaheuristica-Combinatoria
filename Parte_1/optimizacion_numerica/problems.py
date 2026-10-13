"""Funciones objetivo: cada problema agrupa f, su gradiente analítico, dominio y óptimo."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

C_SCHWEFEL = 418.9828872724338
X_OPT_SCHWEFEL = 420.96874636


def schwefel(x):
    """Schwefel en [-500, 500]^d. Mínimo global: x_i = 420.9687 -> f ≈ 0."""
    x = np.asarray(x, dtype=float)
    return C_SCHWEFEL * x.shape[-1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)


def gradiente_schwefel(x):
    """d/dx [x sin(sqrt|x|)] = sin(sqrt|x|) + (sqrt|x|/2) cos(sqrt|x|); f usa el signo opuesto."""
    s = np.sqrt(np.abs(np.asarray(x, dtype=float)))
    return np.where(s == 0, 0.0, -np.sin(s) - 0.5 * s * np.cos(s))


def rosenbrock(x):
    """Rosenbrock en [-5, 10]^d (d >= 2). Mínimo global: x_i = 1 -> f = 0."""
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2, axis=-1)


def gradiente_rosenbrock(x):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    d = x[..., 1:] - x[..., :-1] ** 2
    g[..., :-1] = -400.0 * x[..., :-1] * d - 2.0 * (1 - x[..., :-1])
    g[..., 1:] += 200.0 * d
    return g


@dataclass(frozen=True)
class Problema:
    nombre: str
    f: Callable
    grad: Callable
    bounds: tuple
    x_opt: float            # valor óptimo de cada coordenada
    log_color: bool = False  # escala log1p en el mapa 2D (rango dinámico enorme)
    dim_min: int = 1


PROBLEMAS = {
    "schwefel": Problema("Schwefel", schwefel, gradiente_schwefel, (-500.0, 500.0), X_OPT_SCHWEFEL),
    "rosenbrock": Problema("Rosenbrock", rosenbrock, gradiente_rosenbrock, (-5.0, 10.0), 1.0,
                           log_color=True, dim_min=2),
}
