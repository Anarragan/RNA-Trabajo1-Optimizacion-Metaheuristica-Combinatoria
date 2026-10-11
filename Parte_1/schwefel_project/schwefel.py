import numpy as np

C_SCHWEFEL = 418.9828872724338 
X_OPT = 420.96874636

def schwefel(x):
    """
    Función de Schwefel estándar.
    Dominio: [-500, 500]^d
    Mínimo global: x_i = 420.96874636 -> f(x*) ≈ 0
    """
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    
    return C_SCHWEFEL * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)