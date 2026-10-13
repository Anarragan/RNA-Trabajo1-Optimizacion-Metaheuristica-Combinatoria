import argparse
import json
import sys
from pathlib import Path

import numpy as np

import optimizers
from experiments import ejecutar_optimizador, evals_equivalentes, run_experiments
from problems import PROBLEMAS
from visualize import make_gd_gif, make_metaheuristic_gif

RAIZ = Path(__file__).resolve().parent

MIN_CORRIDAS = 30

ALGORITMOS = {
    "gd": optimizers.descenso_gradiente,
    "ea": optimizers.algoritmo_evolutivo,
    "pso": optimizers.pso,
    "de": optimizers.evolucion_diferencial,
}

# Hiperparámetros propios de cada algoritmo (los flags se generan a partir de aquí).
HIPERPARAMETROS = {
    "gd": ("learning_rate", "tol"),
    "ea": ("pop_size", "mutation_rate", "mutation_scale"),
    "pso": ("pop_size", "w", "c1", "c2"),
    "de": ("pop_size", "F", "CR"),
}


def cargar_config(ruta):
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta}")
    with ruta.open("r", encoding="utf-8") as f:
        return json.load(f)


def crear_parser():
    p = argparse.ArgumentParser(
        description="Ejecutor de optimizadores para funciones de prueba (Schwefel, Rosenbrock).",
        epilog=(
            "Ejemplos:\n"
            "  python run.py --algo pso                    # 1 corrida de PSO (Schwefel)\n"
            "  python run.py --problem rosenbrock --algo de  # igual, en Rosenbrock\n"
            "  python run.py --algo pso --gif              # 1 corrida + GIF (dim 2)\n"
            "  python run.py --algo pso de --experiment    # 30 corridas de PSO y DE + comparación\n"
            "  python run.py --compare --runs 50           # los 4 algoritmos, 50 corridas c/u\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--problem", choices=list(PROBLEMAS), default="schwefel",
                   help="Función objetivo (por defecto: schwefel).")
    p.add_argument("--config", default=None,
                   help="Ruta del JSON de configuración (por defecto configs/CLI_defaults_<problem>.json).")
    p.add_argument("--algo", nargs="+", choices=[*ALGORITMOS, "all"],
                   help="Uno o varios algoritmos (gd ea pso de, o 'all').")
    p.add_argument("--experiment", action="store_true",
                   help="Corre cada algoritmo elegido >= 30 veces y muestra la comparación.")
    p.add_argument("--compare", action="store_true",
                   help="Igual que --experiment; si no pones --algo, compara los 4.")
    p.add_argument("--save-runs", action="store_true",
                   help="Guarda el historial de cada corrida (.npz) en results/.")
    p.add_argument("--grad-cost", type=float, default=None,
                   help="Costo de un gradiente en evaluaciones de f "
                        "(por defecto 2*dim, diferencias centrales).")

    p.add_argument("--gif", action="store_true",
                   help="Genera el GIF (solo una corrida individual en dim = 2).")
    p.add_argument("--gif-dir", default=str(RAIZ / "gifs"),
                   help="Directorio de salida del GIF (por defecto: gifs/).")

    p.add_argument("--dim", type=int)
    p.add_argument("--runs", type=int, help="Corridas por algoritmo (mínimo 30).")
    p.add_argument("--seed", type=int)
    p.add_argument("--max-iter", type=int)
    p.add_argument("--bounds", type=float, nargs=2)
    p.add_argument("--threshold", type=float)
    p.add_argument("--tag")

    vistos = set()
    for nombres in HIPERPARAMETROS.values():
        for nombre in nombres:
            if nombre in vistos:
                continue
            vistos.add(nombre)
            flag = "--" + (nombre if nombre in ("F", "CR") else nombre.replace("_", "-"))
            p.add_argument(flag, dest=nombre,
                           type=int if nombre == "pop_size" else float)
    return p

def construir_parametros(args, config, algo, problema):
    """Precedencia: CLI > sección del algoritmo > sección common."""
    exp, common = config["experiment"], config["common"]

    def elegir(cli, valor):
        return valor if cli is None else cli

    dim = elegir(args.dim, exp["dim"])
    seed = elegir(args.seed, exp["seed"])
    runs = elegir(args.runs, exp["runs"])
    bounds = tuple(elegir(args.bounds, problema.bounds))
    threshold = elegir(args.threshold, exp["success_threshold"])

    params = dict(config[algo])
    params.update({
        k: getattr(args, k)
        for k in HIPERPARAMETROS[algo]
        if getattr(args, k) is not None
    })
    params["max_iter"] = elegir(args.max_iter, common["max_iter"])

    if algo != "gd":
        params.setdefault("pop_size", common["pop_size"])
        params["record_population"] = args.gif  # solo si hay GIF que dibujar

    if dim < problema.dim_min:
        raise ValueError(f"{problema.nombre} requiere dim >= {problema.dim_min}.")
    return dim, seed, runs, bounds, threshold, params

# --------------------------------------------------------------------------
# Salida en consola
# --------------------------------------------------------------------------
def mostrar_resultado(res, dim, costo_grad):
    print("\n--- Resultado de la optimización ---")
    print(f"Punto final: {res['x_final']}")
    print(f"Valor final de f: {float(res['f_final']):.10g}")
    print(f"Iteraciones: {res['iteraciones']}")
    print(f"Evaluaciones de f: {res['n_evals_f']}")
    print(f"Evaluaciones de gradiente: {res.get('n_evals_grad', 0)}")
    print(f"Evaluaciones equivalentes (en f): "
          f"{evals_equivalentes(res, dim, costo_grad):.0f}")
    print(f"Tiempo: {res['tiempo_s']:.4f} s")
    print(f"Convergió según su criterio: {res['convergio']}")

def imprimir_comparacion(etiquetas, resumenes):
    r0 = resumenes[0]
    print(f"\n=== Comparación | dim={r0['dim']} | corridas por algoritmo={r0['n_runs']} "
          f"| éxito si f <= {r0['umbral_exito']:g} ===\n")

    cols = [
        ("Algo", "{:<5}", lambda e, r: e.upper()),
        ("Mejor f", "{:>12.5g}", lambda e, r: r["mejor"]),
        ("Media f", "{:>12.5g}", lambda e, r: r["media"]),
        ("Std f", "{:>10.4g}", lambda e, r: r["std"]),
        ("Éxito", "{:>7.0%}", lambda e, r: r["tasa_exito"]),
        ("Iter", "{:>8.0f}", lambda e, r: r["media_iteraciones"]),
        ("Evals f", "{:>10.0f}", lambda e, r: r["media_evals_f"]),
        ("Evals ∇f", "{:>9.0f}", lambda e, r: r["media_evals_grad"]),
        ("Evals eq.", "{:>10.0f}", lambda e, r: r["media_evals_equiv"]),
        ("Tiempo s", "{:>9.4f}", lambda e, r: r["media_tiempo_s"]),
    ]

    def ancho(fmt):  # "{:>12.5g}" -> 12
        return int("".join(c for c in fmt.split(":")[1].split(".")[0] if c.isdigit()))

    cabecera = " ".join(
        f"{n:<{ancho(fmt)}}" if fmt.startswith("{:<") else f"{n:>{ancho(fmt)}}"
        for n, fmt, _ in cols
    )
    print(cabecera)
    print("-" * len(cabecera))
    for e, r in zip(etiquetas, resumenes):
        print(" ".join(fmt.format(get(e, r)) for _, fmt, get in cols))

    print(f"\nEvals eq. = Evals f + {r0['costo_grad_en_f']:g} x Evals ∇f "
          f"(1 gradiente = {r0['costo_grad_en_f']:g} evaluaciones de f). "
          "Los valores son promedios por corrida.")

    print("\nMejor solución encontrada (la mejor de todas las corridas):")
    for e, r in zip(etiquetas, resumenes):
        x = np.array2string(np.array(r["mejor_x"]), precision=4, suppress_small=True)
        print(f"  {e.upper():<4} f={r['mejor']:.6g}  x={x}")

    if not all(r["grad_reportado"] for r in resumenes):
        print("\nAviso: algún optimizador no devolvió 'n_evals_grad'; "
              "se asumió 0 y 'Evals eq.' puede subestimar su costo.")

def generar_gif(problema, algo, res, umbral, directorio):
    comunes = dict(
        problema=problema,
        hist_x=res["historial_x"],
        hist_f=res["historial_f"],
        output_path=str(Path(directorio) / f"{problema.nombre.lower()}_{algo}_2d.gif"),
        umbral=umbral,
    )
    if algo == "gd":
        make_gd_gif(**comunes)
        return
    poblacion = res.get("historial_poblacion")
    if poblacion is None:
        raise ValueError("El optimizador no devolvió 'historial_poblacion'.")
    make_metaheuristic_gif(hist_pob=poblacion, nombre=algo.upper(), **comunes)

# --------------------------------------------------------------------------
# Modos
# --------------------------------------------------------------------------
def resolver_algoritmos(parser, args):
    if not args.algo:
        if args.compare:
            return list(ALGORITMOS)
        parser.error("Indica --algo (uno o varios) o usa --compare.")
    if "all" in args.algo:
        return list(ALGORITMOS)
    return list(dict.fromkeys(args.algo))  # sin repetidos, conserva el orden

def modo_comparacion(args, config, algos, problema):
    resumenes = []
    for algo in algos:
        dim, seed, runs, bounds, thr, params = construir_parametros(args, config, algo, problema)
        tag = f"{args.tag}_{algo}" if args.tag else f"{problema.nombre.lower()}_{algo}_{dim}d_seed{seed}"
        print(f"Corriendo {algo.upper()}: {runs} corridas...", flush=True)
        resumenes.append(run_experiments(
            algoritmo_fn=ALGORITMOS[algo], problema=problema, dim=dim, n_runs=runs, base_seed=seed,
            tag=tag, umbral_exito=thr, bounds=bounds,
            costo_grad=args.grad_cost, guardar_historial=args.save_runs, **params,
        ))
    imprimir_comparacion(algos, resumenes)
    print("\nResúmenes completos guardados en results/<tag>_summary.json")


def modo_individual(args, config, algo, problema):
    dim, seed, _, bounds, threshold, params = construir_parametros(args, config, algo, problema)
    res = ejecutar_optimizador(
        algoritmo_fn=ALGORITMOS[algo], problema=problema, dim=dim, seed=seed,
        bounds=bounds, **params
    )
    mostrar_resultado(res, dim, args.grad_cost)
    if args.gif:
        generar_gif(problema, algo, res, threshold, args.gif_dir)

def main():
    if hasattr(sys.stdout, "reconfigure"):      # consolas Windows (cp1252) fallan con "∇"
        sys.stdout.reconfigure(encoding="utf-8")
    parser = crear_parser()
    args = parser.parse_args()
    problema = PROBLEMAS[args.problem]
    config = cargar_config(args.config or RAIZ / "configs" / f"CLI_defaults_{args.problem}.json")
    algos = resolver_algoritmos(parser, args)

    try:
        if args.experiment or args.compare:
            if args.gif:
                parser.error("--gif solo se usa en una corrida individual (sin --experiment/--compare).")
            runs = construir_parametros(args, config, algos[0], problema)[2]
            if runs < MIN_CORRIDAS:
                parser.error(f"Se requieren al menos {MIN_CORRIDAS} corridas por algoritmo (pediste {runs}).")
            modo_comparacion(args, config, algos, problema)
            return

        if len(algos) > 1:
            parser.error("Para varios algoritmos usa --experiment o --compare.")
        dim = construir_parametros(args, config, algos[0], problema)[0]
        if args.gif and dim != 2:
            parser.error("--gif solo está disponible para dim = 2.")
        modo_individual(args, config, algos[0], problema)
    except ValueError as e:
        parser.error(str(e))

if __name__ == "__main__":
    main()
