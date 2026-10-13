# IMportamos NumPy para tarabajar con arreglos y operaciones matemáticas
import numpy as np

# Coeficiente estándar de rosenbrock(el "100" de la formula)
C_ROSENBROCK = 100

# Valor de cada coordenada en el mínimo global: [1, 1, ..., 1]
X_OPT = 1.0

# Definimos la función de Rosenbrock que recibe una arreglo x
def rosenbrock(x):

    """
    Función de Rosenbrock estándar

    Dominio típico: [-5, 10]
     Dominio usado en este trabajo: [-2, 2].
    Mínimo global:
        x_i = 1.0 para todo i
        f(x*) = 0.0    
    """

    # Convertimos x a un arreglo de NumPy (por si llega como lista)
    x = np.asarray(x, dtype=float)

    # Si x es un vector (una sola dimensión)
    if x.ndim == 1:

        # Fórmula de Rosenbrock
        return np.sum(C_ROSENBROCK * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    # Si x es una matriz (varias filas a la vez)
    else:

        # Aplicamos la fórmula fila por fila
        return np.sum(C_ROSENBROCK * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

# Definimos el gradiente analitico de rosenbrock
def gradiente_rosenbrock(x):

    """
    Gradiente analítico de la función de Rosenbrock.

    Funciona tanto para un punto (vector de forma (n,))
    como para una matriz de puntos (forma (m, n)).
    """

    # Convertimos x a un arreglo de NumPy
    x = np.asarray(x, dtype=float)

    # Caso 1: x es un vector (un solo punto)
    if x.ndim == 1:
        n = x.size
        grad = np.zeros_like(x)
        grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        for i in range(1, n - 1):
            grad[i] = (200 * (x[i] - x[i-1]**2)
                       - 400 * x[i] * (x[i+1] - x[i]**2)
                       - 2 * (1 - x[i]))
        grad[-1] = 200 * (x[-1] - x[-2]**2)
        return grad

    # Caso 2: x es una matriz (varios puntos, uno por fila)
    n = x.shape[1]
    grad = np.zeros_like(x)
    grad[:, 0] = -400 * x[:, 0] * (x[:, 1] - x[:, 0]**2) - 2 * (1 - x[:, 0])
    for i in range(1, n - 1):
        grad[:, i] = (200 * (x[:, i] - x[:, i-1]**2)
                      - 400 * x[:, i] * (x[:, i+1] - x[:, i]**2)
                      - 2 * (1 - x[:, i]))
    grad[:, -1] = 200 * (x[:, -1] - x[:, -2]**2)
    return grad