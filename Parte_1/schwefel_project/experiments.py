"""
Orquestador de experimentos: corre múltiples semillas, guarda resultados,
calcula estadísticas.
"""
import os
import json
import numpy as np

from schwefel import schwefel
from optimizers import descenso_gradiente, gradiente_schwefel


RESULTS_DIR = "results"
UMBRALES_EXITO = {
    2: 1e-3,
    3: 1e-2,
    4: 5e-2,
    5: 1e-1,
}

def run_gd_experiments(
    dim=2,
    n_runs=30,
    learning_rate=0.1,
    max_iter=200,
    tol=1e-6,
    bounds=(-500.0, 500.0),
    use_analytic_grad=True,
    record_every=1,
    base_seed=0,
    tag=None,
    umbral_exito=None,
):
    """
    Corre n_runs corridas independientes de GD sobre Schwefel.

    Parámetros clave:
      - dim: dimensión del problema. Acepta 2, 3, ...
      - tag: nombre base para los archivos. Si None, se deriva: f"gd_{dim}d".
      - base_seed: semilla base. Cada corrida usa base_seed + run.
      - umbral_exito: umbral para contar éxito. Si None, se toma de UMBRALES_EXITO[dim].

    Ejemplo:
        run_gd_experiments(dim=2, base_seed=0)
        run_gd_experiments(dim=3, base_seed=1000)
    """
    if tag is None:
        tag = f"gd_{dim}d"
    if umbral_exito is None:
        umbral_exito = UMBRALES_EXITO.get(dim, 1e-1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    f_finales, x_finales, iteraciones, n_evals, exitos, convergio = [], [], [], [], [], []

    for run in range(n_runs):
        seed = base_seed + run
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(bounds[0], bounds[1], size=dim)

        res = descenso_gradiente(
            f=schwefel,
            x0=x0,
            grad_fn=gradiente_schwefel if use_analytic_grad else None,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            bounds=bounds,
            record_every=record_every,
        )

        np.savez(
            os.path.join(RESULTS_DIR, f"{tag}_run{run:02d}.npz"),
            x0=x0,
            x_final=res["x_final"],
            f_final=res["f_final"],
            historial_x=res["historial_x"],
            historial_f=res["historial_f"],
            iteraciones=res["iteraciones"],
            n_evals_f=res["n_evals_f"],
            seed=seed,
            convergio=res["convergio"],
        )

        f_finales.append(res["f_final"])
        x_finales.append(res["x_final"])
        iteraciones.append(res["iteraciones"])
        n_evals.append(res["n_evals_f"])
        exitos.append(res["f_final"] <= umbral_exito)
        convergio.append(res["convergio"])

    f_finales = np.array(f_finales)
    resumen = {
        "tag": tag,
        "dim": dim,
        "n_runs": n_runs,
        "umbral_exito": umbral_exito,
        "media": float(np.mean(f_finales)),
        "std": float(np.std(f_finales)),
        "mejor": float(np.min(f_finales)),
        "peor": float(np.max(f_finales)),
        "tasa_exito": float(np.mean(exitos)),
        "media_evals_f": float(np.mean(n_evals)),
        "media_iteraciones": float(np.mean(iteraciones)),
        "tasa_convergencia": float(np.mean(convergio)),
        "f_finales": [float(v) for v in f_finales],
        "config": {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "tol": tol,
            "use_analytic_grad": use_analytic_grad,
            "record_every": record_every,
            "base_seed": base_seed,
        },
    }

    with open(os.path.join(RESULTS_DIR, f"{tag}_summary.json"), "w") as fp:
        json.dump(resumen, fp, indent=2)

    return resumen

def cargar_corrida(tag, run):
    """
    Carga los resultados de una corrida específica.
    """
    d = np.load(os.path.join(RESULTS_DIR, f"{tag}_run{run:02d}.npz"))
    return d["historial_x"], d["historial_f"]