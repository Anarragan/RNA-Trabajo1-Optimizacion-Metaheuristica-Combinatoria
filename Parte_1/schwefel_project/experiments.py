import os
import json
import inspect
import numpy as np

from schwefel import schwefel
import optimizers

RESULTS_DIR = "results"
UMBRALES_EXITO = {
    2: 1e-3,
    3: 1e-2,
    4: 5e-2,
    5: 1e-1,
}

def ejecutar_optimizador(algoritmo_fn, dim=2, seed=None, f=schwefel, bounds=(-500.0, 500.0), **kwargs):
    """
    Ejecuta una ÚNICA corrida individual de cualquier optimizador de manera transparente.
    
    Parámetros:
        - algoritmo_fn: función del optimizador (e.g., optimizers.pso, optimizers.descenso_gradiente).
        - dim: dimensión del problema.
        - seed: entero o None. Si se proporciona, se crea un np.random.default_rng(seed).
        - f: función objetivo.
        - bounds: límites del espacio de búsqueda.
        - **kwargs: hiperparámetros adicionales específicos del optimizador (e.g., pop_size, learning_rate).
    
    Retorna:
        Dict estándar retornado por el optimizador con historial, métricas y metadatos.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds

    # Inspeccionamos la firma del algoritmo para enviarle solo los parámetros que soporta
    sig = inspect.signature(algoritmo_fn)
    params = sig.parameters

    opt_args = {}
    
    if "f" in params:
        opt_args["f"] = f
    if "dim" in params:
        opt_args["dim"] = dim
    if "bounds" in params:
        opt_args["bounds"] = bounds
    if "rng" in params:
        opt_args["rng"] = rng
    elif "x0" in params:
        # Algoritmos punto a punto como Gradient Descent requieren un punto x0 inicial
        opt_args["x0"] = rng.uniform(lo, hi, size=dim)

    # Incorporamos hiperparámetros adicionales pasados por el usuario
    for k, v in kwargs.items():
        if k not in params:
            raise ValueError(f"El optimizador {algoritmo_fn.__name__} no acepta el parámetro '{k}'.")
        else:
            opt_args[k] = v

    return algoritmo_fn(**opt_args)

def run_experiments(
    algoritmo_fn,
    dim=2,
    n_runs=30,
    base_seed=0,
    tag=None,
    umbral_exito=None,
    bounds=(-500.0, 500.0),
    **opt_kwargs
):
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

    for run in range(n_runs):
        seed = base_seed + run
        
        # Reutilizamos exactamente la misma función de corrida individual
        res = ejecutar_optimizador(
            algoritmo_fn=algoritmo_fn,
            dim=dim,
            seed=seed,
            bounds=bounds,
            **opt_kwargs
        )

        # Guardar historial individual
        save_data = {
            "x_final": res["x_final"],
            "f_final": res["f_final"],
            "historial_x": res["historial_x"],
            "historial_f": res["historial_f"],
            "iteraciones": res["iteraciones"],
            "n_evals_f": res["n_evals_f"],
            "seed": seed,
            "convergio": res["convergio"],
        }
        if "historial_poblacion" in res:
            save_data["historial_poblacion"] = res["historial_poblacion"]

        np.savez(os.path.join(RESULTS_DIR, f"{tag}_run{run:02d}.npz"), **save_data)

        # Recopilación de estadísticas
        f_finales.append(res["f_final"])
        x_finales.append(res["x_final"])
        iteraciones.append(res["iteraciones"])
        n_evals.append(res["n_evals_f"])
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