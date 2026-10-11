import os
import json
import time
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

def evals_equivalentes(res, dim, costo_grad=None):
    """
    Evaluaciones totales expresadas en 'evaluaciones de f':

        equivalentes = n_evals_f + costo_grad * n_evals_grad

    Por defecto costo_grad = 2*dim (gradiente por diferencias centrales:
    2 evaluaciones de f por coordenada).
    """
    if costo_grad is None:
        costo_grad = 2 * dim
    return res["n_evals_f"] + costo_grad * res.get("n_evals_grad", 0)


def ejecutar_optimizador(algoritmo_fn, dim=2, seed=None, f=schwefel,
                         bounds=(-500.0, 500.0), **kwargs):
    """
    Ejecuta UNA corrida de cualquier optimizador y mide su tiempo.

    Retorna el dict del optimizador más la clave 'tiempo_s'.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds

    # Solo se envían los parámetros que la función soporta.
    params = inspect.signature(algoritmo_fn).parameters

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
        # Métodos de un solo punto (GD) necesitan un x0 inicial.
        opt_args["x0"] = rng.uniform(lo, hi, size=dim)

    for k, v in kwargs.items():
        if k not in params:
            raise ValueError(
                f"El optimizador {algoritmo_fn.__name__} no acepta el parámetro '{k}'."
            )
        opt_args[k] = v

    t0 = time.perf_counter()
    res = dict(algoritmo_fn(**opt_args))
    res["tiempo_s"] = time.perf_counter() - t0
    return res


def run_experiments(
    algoritmo_fn,
    dim=2,
    n_runs=30,
    base_seed=0,
    tag=None,
    umbral_exito=None,
    bounds=(-500.0, 500.0),
    costo_grad=None,
    guardar_historial=True,
    **opt_kwargs
):
    """
    Protocolo de N corridas independientes (semillas base_seed + run).

    costo_grad: costo de un gradiente en evaluaciones de f (por defecto 2*dim).
    guardar_historial: si es False no se escriben los .npz por corrida
                       (el resumen JSON siempre se guarda).
    """
    if n_runs < 1:
        raise ValueError("n_runs debe ser al menos 1.")

    nombre_algo = getattr(algoritmo_fn, "__name__", "optimizador")
    tag = tag or f"{nombre_algo}_{dim}d"
    if umbral_exito is None:
        umbral_exito = UMBRALES_EXITO.get(dim, 1e-1)
    if costo_grad is None:
        costo_grad = 2 * dim

    os.makedirs(RESULTS_DIR, exist_ok=True)

    f_fin, x_fin, iters = [], [], []
    ev_f, ev_g, ev_eq, tiempos = [], [], [], []
    exitos, convergio = [], []
    reporta_grad = True

    for run in range(n_runs):
        seed = base_seed + run
        res = ejecutar_optimizador(
            algoritmo_fn=algoritmo_fn, dim=dim, seed=seed,
            bounds=bounds, **opt_kwargs
        )

        if guardar_historial:
            datos = {
                "x_final": res["x_final"],
                "f_final": res["f_final"],
                "historial_x": res["historial_x"],
                "historial_f": res["historial_f"],
                "iteraciones": res["iteraciones"],
                "n_evals_f": res["n_evals_f"],
                "n_evals_grad": res.get("n_evals_grad", 0),
                "tiempo_s": res["tiempo_s"],
                "seed": seed,
                "convergio": res["convergio"],
            }
            if "historial_poblacion" in res:
                datos["historial_poblacion"] = res["historial_poblacion"]
            np.savez(os.path.join(RESULTS_DIR, f"{tag}_run{run:02d}.npz"), **datos)

        reporta_grad &= "n_evals_grad" in res
        f_fin.append(float(res["f_final"]))
        x_fin.append(np.asarray(res["x_final"], dtype=float))
        iters.append(res["iteraciones"])
        ev_f.append(res["n_evals_f"])
        ev_g.append(res.get("n_evals_grad", 0))
        ev_eq.append(evals_equivalentes(res, dim, costo_grad))
        tiempos.append(res["tiempo_s"])
        exitos.append(res["f_final"] <= umbral_exito)
        convergio.append(res["convergio"])

    f_fin = np.array(f_fin)
    i_mejor = int(np.argmin(f_fin))

    resumen = {
        "tag": tag,
        "algoritmo": nombre_algo,
        "dim": dim,
        "n_runs": n_runs,
        "umbral_exito": umbral_exito,
        # Calidad de la solución
        "media": float(f_fin.mean()),
        "mediana": float(np.median(f_fin)),
        "std": float(f_fin.std(ddof=1)) if n_runs > 1 else 0.0,
        "mejor": float(f_fin.min()),
        "peor": float(f_fin.max()),
        "mejor_x": x_fin[i_mejor].tolist(),
        "tasa_exito": float(np.mean(exitos)),
        # Costo computacional (evaluaciones por separado + equivalencia)
        "media_evals_f": float(np.mean(ev_f)),
        "media_evals_grad": float(np.mean(ev_g)),
        "costo_grad_en_f": costo_grad,
        "media_evals_equiv": float(np.mean(ev_eq)),
        "std_evals_equiv": float(np.std(ev_eq, ddof=1)) if n_runs > 1 else 0.0,
        "grad_reportado": bool(reporta_grad),
        # Otros
        "media_iteraciones": float(np.mean(iters)),
        "media_tiempo_s": float(np.mean(tiempos)),
        "tasa_convergencia": float(np.mean(convergio)),
        "f_finales": f_fin.tolist(),
        "config": {"base_seed": base_seed, "bounds": list(bounds), **opt_kwargs},
    }

    with open(os.path.join(RESULTS_DIR, f"{tag}_summary.json"), "w") as fp:
        json.dump(resumen, fp, indent=2)

    return resumen

def cargar_corrida(tag, run):
    """Carga los datos guardados de una corrida específica."""
    return np.load(os.path.join(RESULTS_DIR, f"{tag}_run{run:02d}.npz"))