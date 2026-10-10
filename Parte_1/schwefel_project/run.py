import argparse
import json
from pathlib import Path

import optimizers
from experiments import ejecutar_optimizador, run_experiments
from visualize import make_gd_gif, make_metaheuristic_gif

# Los valores por defecto viven en ../configs/ respecto a este archivo.
CONFIG_POR_DEFECTO = (
    Path(__file__).resolve().parent / "configs" / "CLI_defaults_schwefel.json"
)

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
    """Lee el JSON de configuración; falla con un mensaje claro si no existe."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta}")
    with ruta.open("r", encoding="utf-8") as f:
        return json.load(f)


def crear_parser():
    p = argparse.ArgumentParser(
        description="Ejecutor de optimizadores para la función de Schwefel."
    )
    p.add_argument("--config", default=str(CONFIG_POR_DEFECTO),
                   help="Ruta del JSON de configuración.")
    p.add_argument("--algo", required=True, choices=list(ALGORITMOS),
                   help="Algoritmo que se ejecutará.")
    p.add_argument("--experiment", action="store_true",
                   help="Activa el experimento de múltiples corridas.")

    # Generación de GIF: solo cuando se pide explícitamente.
    p.add_argument("--gif", action="store_true",
                   help="Genera el GIF (solo corrida individual en 2D).")
    p.add_argument("--gif-dir", default="gifs",
                   help="Directorio de salida del GIF (por defecto: gifs).")

    # Parámetros generales.
    p.add_argument("--dim", type=int)
    p.add_argument("--runs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-iter", type=int)
    p.add_argument("--bounds", type=float, nargs=2)
    p.add_argument("--threshold", type=float)
    p.add_argument("--tag")

    # Hiperparámetros: un flag por nombre único (pop_size se comparte).
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


def construir_parametros(args, config):
    """Precedencia: CLI > sección del algoritmo > sección common."""
    exp, common = config["experiment"], config["common"]

    def elegir(cli, valor):
        return valor if cli is None else cli

    dim = elegir(args.dim, exp["dimensions"][0])
    seed = elegir(args.seed, exp["seed"])
    runs = elegir(args.runs, exp["runs"])
    bounds = tuple(elegir(args.bounds, common["bounds"]))
    threshold = elegir(args.threshold, exp["success_threshold"])

    params = dict(config[args.algo])
    params.update({
        k: getattr(args, k)
        for k in HIPERPARAMETROS[args.algo]
        if getattr(args, k) is not None
    })
    params["max_iter"] = elegir(args.max_iter, common["max_iter"])

    if args.algo != "gd":
        params.setdefault("pop_size", common["pop_size"])
        # Guardar la población solo cuando hay GIF que dibujar.
        params["record_population"] = args.gif

    return dim, seed, runs, bounds, threshold, params


def mostrar_resultado(res):
    print("\n--- Resultado de la optimización ---")
    print(f"Punto final: {res['x_final']}")
    print(f"Valor final de f: {float(res['f_final']):.10g}")
    print(f"Iteraciones: {res['iteraciones']}")
    print(f"Evaluaciones de f: {res['n_evals_f']}")
    print(f"Convergió según su criterio: {res['convergio']}")


def generar_gif(algo, res, umbral, directorio):
    salida = str(Path(directorio) / f"{algo}_2d.gif")
    comunes = dict(
        hist_x=res["historial_x"],
        hist_f=res["historial_f"],
        output_path=salida,
        umbral=umbral,
    )
    if algo == "gd":
        make_gd_gif(**comunes)
        return

    poblacion = res.get("historial_poblacion")
    if poblacion is None:
        raise ValueError("El optimizador no devolvió 'historial_poblacion'.")
    make_metaheuristic_gif(hist_pob=poblacion, nombre=algo.upper(), **comunes)


def main():
    parser = crear_parser()
    args = parser.parse_args()
    config = cargar_config(args.config)

    dim, seed, runs, bounds, threshold, params = construir_parametros(args, config)
    algoritmo_fn = ALGORITMOS[args.algo]

    if args.gif and args.experiment:
        parser.error("--gif no se puede usar con --experiment.")
    if args.gif and dim != 2:
        parser.error("--gif solo está disponible para dim = 2.")

    if args.experiment:
        if runs < 30:
            parser.error("El modo experimental requiere al menos 30 corridas.")

        resumen = run_experiments(
            algoritmo_fn=algoritmo_fn,
            dim=dim,
            n_runs=runs,
            base_seed=seed,
            tag=args.tag or f"{args.algo}_{dim}d_seed{seed}",
            umbral_exito=threshold,
            bounds=bounds,
            **params,
        )
        print("\n--- Resumen del experimento ---")
        print(json.dumps(resumen, indent=2, ensure_ascii=False))
        return

    # Corrida individual: no activa el protocolo estadístico.
    res = ejecutar_optimizador(
        algoritmo_fn=algoritmo_fn,
        dim=dim,
        seed=seed,
        bounds=bounds,
        **params,
    )
    mostrar_resultado(res)

    if args.gif:
        generar_gif(args.algo, res, threshold, args.gif_dir)


if __name__ == "__main__":
    main()