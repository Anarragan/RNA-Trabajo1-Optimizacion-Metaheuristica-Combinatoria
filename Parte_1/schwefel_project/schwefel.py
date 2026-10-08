import numpy as np

C_SCHWEFEL = 418.9829
X_OPT = 420.96874636

def schwefel(x):
    """
    Función de Schwefel estándar.

    Dominio: [-500, 500]^n
    Mínimo global:
        x_i = 420.96874636
        f(x*) ≈ 0
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        return C_SCHWEFEL * x.size - np.sum(
            x * np.sin(np.sqrt(np.abs(x)))
        )
    else:
        n = x.shape[1]
        return C_SCHWEFEL * n - np.sum(
            x * np.sin(np.sqrt(np.abs(x))),
            axis=1
        )