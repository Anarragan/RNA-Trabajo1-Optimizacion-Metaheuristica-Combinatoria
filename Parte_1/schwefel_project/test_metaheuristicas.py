"""
Prueba rápida de EA, PSO y DE sobre Schwefel.
"""
import numpy as np
from schwefel import schwefel
from optimizers import pso, evolucion_diferencial, algoritmo_evolutivo

def main():
    dim = 2
    n_runs = 10
    bounds = (-500.0, 500.0)
    umbral = 1e-3  # éxito para 2D

    configs = {
        "PSO": lambda rng: pso(
            schwefel, dim, rng,
            pop_size=40, max_iter=500,
            bounds=bounds, record_population=False,
        ),
        "DE": lambda rng: evolucion_diferencial(
            schwefel, dim, rng,
            pop_size=40, max_iter=500,
            F=0.6, CR=0.9,
            bounds=bounds, record_population=False,
        ),
        "EA": lambda rng: algoritmo_evolutivo(
            schwefel, dim, rng,
            pop_size=40, max_iter=500,
            pc=0.9, pm=0.2, elite=2,
            bounds=bounds, record_population=False,
        ),
    }

    print(f"\n=== Schwefel {dim}D | {n_runs} corridas | umbral éxito = {umbral} ===\n")
    print(f"{'Algoritmo':<6} {'Mejor':>12} {'Media':>12} {'Peor':>12} "
          f"{'Tasaéxito':>10} {'Iter media':>10} {'Evals media':>12}")
    print("-" * 80)

    for nombre, runner in configs.items():
        fs, its, evs, exitos = [], [], [], []
        for run in range(n_runs):
            rng = np.random.default_rng(1000 + run)
            res = runner(rng)
            fs.append(res["f_final"])
            its.append(res["iteraciones"])
            evs.append(res["n_evals_f"])
            exitos.append(res["f_final"] <= umbral)

        fs = np.array(fs)
        print(f"{nombre:<6} {fs.min():>12.4e} {fs.mean():>12.4e} {fs.max():>12.4e} "
              f"{np.mean(exitos)*100:>9.0f}% {np.mean(its):>10.1f} {np.mean(evs):>12.0f}")

    print("\n--- Detalle por corrida (PSO, primera corrida) ---")
    rng = np.random.default_rng(1000)
    res = pso(schwefel, dim, rng, pop_size=40, max_iter=500, bounds=bounds)
    print(f"x0 aleatorio   : {rng.uniform(*bounds, dim)}")
    print(f"x* final       : {res['x_final']}")
    print(f"f(x*)          : {res['f_final']:.6e}")
    print(f"Iteraciones    : {res['iteraciones']}")
    print(f"Convergió      : {res['convergio']}")
    print(f"Óptimo teórico : ~420.9687 en cada componente, f=0")

if __name__ == "__main__":
    main()