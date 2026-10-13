# Importamos os para manejar rutas y crear carpetas
import os

# Importamos json para guardar los resúmenes de los experimentos
import json

# Importamos inspect para analizar los parámetros de las funciones
import inspect

# Importamos NumPy para cálculo numérico
import numpy as np

# Importamos la función y su gradiente
from rosenbrock import rosenbrock, gradiente_rosenbrock

# Importamos los 4 algoritmos
from optimizers import (descenso_gradiente, pso, evolucion_diferencial, algoritmo_evolutivo)


# Carpeta donde guardaremos todos los resultados
RESULTS_DIR = "results"

# Umbral de éxito por dimensión (Rosenbrock es exigente)
UMBRALES_EXITO = {2: 1e-2, 3: 1e-1}


def ejecutar_optimizador(algoritmo_fn, dim=2, seed=None, f=rosenbrock,
                         bounds=(-2.0, 2.0), **kwargs):
    """
    Ejecuta una ÚNICA corrida individual de cualquier optimizador de manera transparente.

    Detecta automáticamente qué parámetros acepta cada algoritmo (x0 para GD,
    rng para PSO/DE/EA, etc.) y le pasa los correctos.
    """
    rng = np.random.default_rng(seed)  # Generador aleatorio con la semilla
    lo, hi = bounds  # Límites del espacio

    sig = inspect.signature(algoritmo_fn)  # Firma del algoritmo
    params = sig.parameters  # Sus parámetros

    opt_args = {}  # Argumentos que le pasaremos

    if "f" in params:  # Si acepta función objetivo
        opt_args["f"] = f
    if "dim" in params:  # Si acepta dimensión
        opt_args["dim"] = dim
    if "bounds" in params:  # Si acepta límites
        opt_args["bounds"] = bounds
    if "rng" in params:  # Si acepta generador aleatorio (PSO, DE, EA)
        opt_args["rng"] = rng
    elif "x0" in params:  # Si acepta punto inicial (GD)
        opt_args["x0"] = rng.uniform(lo, hi, size=dim)

    for k, v in kwargs.items():  # Hiperparámetros adicionales (lr, pop_size, etc.)
        if k not in params:  # Si el algoritmo no acepta ese parámetro
            raise ValueError(f"El optimizador {algoritmo_fn.__name__} no acepta el parámetro '{k}'.")
        opt_args[k] = v

    return algoritmo_fn(**opt_args)  # Ejecutamos el optimizador


def run_experiments(algoritmo_fn, dim=2, n_runs=30, base_seed=0, tag=None,
                    umbral_exito=None, bounds=(-2.0, 2.0), **opt_kwargs):
    """
    Ejecuta el protocolo de N corridas independientes sobre cualquier optimizador,
    almacenando estadísticas globales e historiales detallados.
    """
    if n_runs < 1:
        raise ValueError("n_runs debe ser al menos 1.")

    nombre_algo = getattr(algoritmo_fn, "__name__", "optimizador")
    if tag is None:
        tag = f"{nombre_algo}_{dim}d"

    if umbral_exito is None:
        umbral_exito = UMBRALES_EXITO.get(dim, 1e-1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    f_finales, x_finales, iteraciones, n_evals, exitos, convergio = [], [], [], [], [], []

    n_evals_grad_list = []

    for run in range(n_runs):
        seed = base_seed + run
        res = ejecutar_optimizador(
            algoritmo_fn=algoritmo_fn,
            dim=dim,
            seed=seed,
            bounds=bounds,
            **opt_kwargs
        )

        save_data = {
            "x_final": res["x_final"],
            "f_final": res["f_final"],
            "historial_x": res["historial_x"],
            "historial_f": res["historial_f"],
            "iteraciones": res["iteraciones"],
            "n_evals_f": res["n_evals_f"],
            "n_evals_grad": res["n_evals_grad"],
            "seed": seed,
            "convergio": res["convergio"],
        }
        if "historial_poblacion" in res:
            save_data["historial_poblacion"] = res["historial_poblacion"]

        np.savez(os.path.join(RESULTS_DIR, f"{tag}_run{run:02d}.npz"), **save_data)

        f_finales.append(res["f_final"])
        x_finales.append(res["x_final"])
        iteraciones.append(res["iteraciones"])
        n_evals.append(res["n_evals_f"])
        n_evals_grad_list.append(res.get("n_evals_grad", 0))
        exitos.append(res["f_final"] <= umbral_exito)
        convergio.append(res["convergio"])

    f_finales = np.array(f_finales)

    resumen = {
        "tag": tag,
        "algoritmo": nombre_algo,
        "dim": dim,
        "n_runs": n_runs,
        "umbral_exito": umbral_exito,
        "media": float(np.mean(f_finales)),
        "std": float(np.std(f_finales, ddof=1)),
        "mejor": float(np.min(f_finales)),
        "peor": float(np.max(f_finales)),
        "tasa_exito": float(np.mean(exitos)),
        "media_evals_f": float(np.mean(n_evals)),
        "media_evals_grad":float(np.mean(n_evals_grad_list)),
        "media_iteraciones": float(np.mean(iteraciones)),
        "tasa_convergencia": float(np.mean(convergio)),
        "f_finales": [float(v) for v in f_finales],
        "config": {
            "base_seed": base_seed,
            "bounds": list(bounds),
            **opt_kwargs
        },
    }

    with open(os.path.join(RESULTS_DIR, f"{tag}_summary.json"), "w") as fp:
        json.dump(resumen, fp, indent=2)

    return resumen


def cargar_corrida(tag, run):
    """Carga y retorna los datos completos guardados de una corrida específica."""
    filepath = os.path.join(RESULTS_DIR, f"{tag}_run{run:02d}.npz")
    return np.load(filepath)

