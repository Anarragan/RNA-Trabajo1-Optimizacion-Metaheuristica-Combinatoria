import argparse  # Para leer argumentos desde la terminal
import json  # Para leer y escribir JSON
from pathlib import Path  # Manejo moderno de rutas

import optimizers  # Importamos todos los optimizadores
from experiments import ejecutar_optimizador, run_experiments  # Funciones de experiments
from visualize import make_gd_gif, make_metaheuristic_gif  # Funciones de visualize

# Ruta por defecto del archivo de configuración
CONFIG_POR_DEFECTO = (
    Path(__file__).resolve().parent / "configs" / "CLI_defaults_rosenbrock.json"
)

# Diccionario que mapea nombres a funciones de optimización
ALGORITMOS = {
    "gd": optimizers.descenso_gradiente,  # Descenso por gradiente
    "ea": optimizers.algoritmo_evolutivo,  # Algoritmo evolutivo
    "pso": optimizers.pso,  # PSO
    "de": optimizers.evolucion_diferencial,  # Evolución diferencial
}

# Hiperparámetros propios de cada algoritmo (para generar los flags)
HIPERPARAMETROS = {
    "gd": ("learning_rate", "tol"),  # GD: tasa de aprendizaje y tolerancia
    "ea": ("pop_size", "mutation_rate", "mutation_scale"),  # EA: población, tasa y escala de mutación
    "pso": ("pop_size", "w", "c1", "c2"),  # PSO: población, inercia, coeficientes
    "de": ("pop_size", "F", "CR"),  # DE: población, factor F, tasa cruce
}


def cargar_config(ruta):  # Lee el JSON de configuración
    ruta = Path(ruta)  # Convertimos a Path
    if not ruta.is_file():  # Si el archivo no existe...
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta}")  # Error claro
    with ruta.open("r", encoding="utf-8") as f:  # Abrimos el archivo
        return json.load(f)  # Devolvemos el diccionario


def crear_parser():  # Construye el parser de argumentos
    p = argparse.ArgumentParser(description="Ejecutor de optimizadores para Rosenbrock.")
    p.add_argument("--config", default=str(CONFIG_POR_DEFECTO),  # --config
                   help="Ruta del JSON de configuración.")
    p.add_argument("--algo", required=True, choices=list(ALGORITMOS),  # --algo (obligatorio)
                   help="Algoritmo que se ejecutará.")
    p.add_argument("--experiment", action="store_true",  # --experiment (modo 30 corridas)
                   help="Activa el experimento de múltiples corridas.")
    p.add_argument("--gif", action="store_true",  # --gif
                   help="Genera el GIF (solo corrida individual en 2D).")
    p.add_argument("--gif-dir", default="gifs",  # --gif-dir (carpeta)
                   help="Directorio de salida del GIF (por defecto: gifs).")

    # Parámetros generales
    p.add_argument("--dim", type=int)  # Dimensión
    p.add_argument("--runs", type=int)  # Número de corridas
    p.add_argument("--seed", type=int)  # Semilla base
    p.add_argument("--max-iter", type=int)  # Máximo de iteraciones
    p.add_argument("--bounds", type=float, nargs=2)  # Límites del espacio
    p.add_argument("--threshold", type=float)  # Umbral de éxito
    p.add_argument("--tag")  # Nombre de salida

    # Hiperparámetros: un flag por nombre único
    vistos = set()  # Para no repetir flags
    for nombres in HIPERPARAMETROS.values():  # Recorremos todos los hiperparámetros
        for nombre in nombres:  # Uno por uno
            if nombre in vistos:  # Si ya lo agregamos...
                continue  # Lo saltamos
            vistos.add(nombre)  # Lo marcamos como visto
            flag = "--" + (nombre if nombre in ("F", "CR") else nombre.replace("_", "-"))  # --nombre-con-guiones
            p.add_argument(flag, dest=nombre,  # Agregamos el flag
                           type=int if nombre == "pop_size" else float)  # int para pop_size, float para el resto
    return p  # Devolvemos el parser


def construir_parametros(args, config):  # Combina CLI > config > defaults
    exp, common = config["experiment"], config["common"]  # Secciones del JSON

    def elegir(cli, valor):  # Función auxiliar: CLI tiene prioridad
        return valor if cli is None else cli

    dim = elegir(args.dim, exp["dimensions"][0])  # Dimensión
    seed = elegir(args.seed, exp["seed"])  # Semilla
    runs = elegir(args.runs, exp["runs"])  # Corridas
    bounds = tuple(elegir(args.bounds, common["bounds"]))  # Límites
    threshold = elegir(args.threshold, exp["success_threshold"])  # Umbral

    params = dict(config[args.algo])  # Copiamos los parámetros del algoritmo
    params.update({  # Actualizamos con los de CLI
        k: getattr(args, k) for k in HIPERPARAMETROS[args.algo]
        if getattr(args, k) is not None
    })
    params["max_iter"] = elegir(args.max_iter, common["max_iter"])  # max_iter

    if args.algo != "gd":  # Si es metaheurística...
        params.setdefault("pop_size", common["pop_size"])  # pop_size por defecto
        params["record_population"] = args.gif  # Solo guardamos población si hay GIF

    return dim, seed, runs, bounds, threshold, params  # Devolvemos todo


def mostrar_resultado(res):  # Imprime el resultado en consola
    print("\n--- Resultado de la optimización ---")
    print(f"Punto final: {res['x_final']}")  # Punto final
    print(f"Valor final de f: {float(res['f_final']):.10g}")  # Valor de f
    print(f"Iteraciones: {res['iteraciones']}")  # Iteraciones
    print(f"Evaluaciones de f: {res['n_evals_f']}")  # Evaluaciones
    print(f"Convergió según su criterio: {res['convergio']}")  # Convergencia


def generar_gif(algo, res, umbral, directorio):  # Genera el GIF según el algoritmo
    salida = str(Path(directorio) / f"{algo}_2d.gif")  # Ruta de salida
    comunes = dict(hist_x=res["historial_x"], hist_f=res["historial_f"],  # Parámetros comunes
                   output_path=salida, umbral=umbral)
    if algo == "gd":  # Si es GD...
        make_gd_gif(**comunes)  # GIF de GD
        return

    poblacion = res.get("historial_poblacion")  # Si es metaheurística, necesitamos la población
    if poblacion is None:  # Si no se guardó...
        raise ValueError("El optimizador no devolvió 'historial_poblacion'.")
    make_metaheuristic_gif(hist_pob=poblacion, nombre=algo.upper(), **comunes)  # GIF de metaheurística


def main():  # Función principal
    parser = crear_parser()  # Creamos el parser
    args = parser.parse_args()  # Leemos argumentos
    config = cargar_config(args.config)  # Cargamos configuración

    dim, seed, runs, bounds, threshold, params = construir_parametros(args, config)  # Combinamos
    algoritmo_fn = ALGORITMOS[args.algo]  # Elegimos la función del algoritmo

    if args.gif and args.experiment:  # Validaciones
        parser.error("--gif no se puede usar con --experiment.")
    if args.gif and dim != 2:  # Solo GIF en 2D
        parser.error("--gif solo está disponible para dim = 2.")

    if args.experiment:  # Modo experimento (30 corridas)
        if runs < 30:
            parser.error("El modo experimental requiere al menos 30 corridas.")

        # Corremos los experimentos
        resumen = run_experiments(  
            algoritmo_fn=algoritmo_fn, dim=dim, n_runs=runs,
            base_seed=seed,
            tag=args.tag or f"{args.algo}_{dim}d_seed{seed}",
            umbral_exito=threshold, bounds=bounds, **params,
        )
        print("\n--- Resumen del experimento ---")  # Imprimimos el resumen
        print(json.dumps(resumen, indent=2, ensure_ascii=False))
        return

    res = ejecutar_optimizador(  # Modo corrida individual
        algoritmo_fn=algoritmo_fn, dim=dim, seed=seed,
        bounds=bounds, **params,
    )
    mostrar_resultado(res)  # Mostramos el resultado

    if args.gif:  # Si se pidió GIF...
        generar_gif(args.algo, res, threshold, args.gif_dir)  # Lo generamos


if __name__ == "__main__":  # Si ejecutamos este archivo directamente...
    main()  # Llamamos a main