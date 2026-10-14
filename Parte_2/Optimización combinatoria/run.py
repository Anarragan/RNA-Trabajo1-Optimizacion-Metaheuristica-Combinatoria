import argparse
import json
import sys
import time
from pathlib import Path

# Asegurar que el directorio de módulos de la Parte 2 esté en el path
RAIZ_PARTE2 = Path(__file__).resolve().parent
DIR_PREPARADA = RAIZ_PARTE2 / "Informacion_Preparada"
if str(DIR_PREPARADA) not in sys.path:
    sys.path.insert(0, str(DIR_PREPARADA))

from algoritmos import ejecutar_experimentos_comparativos
from generar_matrices import generar_todas_las_matrices
from generar_tsp import generar_instancias_tsp
from visualizador_tsp import ejecutar_orquestador_visual

ALGORITMOS_VALIDOS = ["aco", "ga", "both", "all"]


def cargar_config(ruta: Path) -> dict:
    """
    Carga y valida un archivo de configuración en formato JSON.

    Parámetros:
        ruta (Path): Ruta al archivo .json de configuración.

    Retorna:
        dict: Diccionario parseado con las secciones de configuración ('experiment', 'common', 'aco', 'ga').
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta}")
    with ruta.open("r", encoding="utf-8") as f:
        return json.load(f)


def crear_parser() -> argparse.ArgumentParser:
    """
    Construye y configura el analizador de argumentos de línea de comandos (CLI) para el TSP.

    Define las opciones para selección de algoritmos, valores horarios del vendedor,
    hiperparámetros de metaheurísticas (población, iteraciones, feromonas, mutación),
    semilla de reproducibilidad, archivos de configuración y banderas utilitarias/visuales.

    Parámetros:
        Ninguno.

    Retorna:
        argparse.ArgumentParser: Objeto analizador de argumentos listo para procesar la entrada de consola.
    """
    p = argparse.ArgumentParser(
        description="Optimizador de rutas del TSP en España Peninsular (Colonia de Hormigas y Algoritmo Genético).",
        epilog=(
            "Ejemplos:\n"
            "  python run.py --demo                      # Corrida rápida de demostración (< 30s)\n"
            "  python run.py                             # Corrida estándar (ACO y GA, todas las horas)\n"
            "  python run.py --algo aco                  # Solo Colonia de Hormigas (ACO)\n"
            "  python run.py --algo ga --horas 25.0      # Algoritmo Genético a 25.0 EUR/h\n"
            "  python run.py --demo --gif                # Demostración + regeneración de GIFs\n"
            "  python run.py --horas 0 15 25 --pop-size 80 --max-iter 200 --seed 123\n"
            "  python run.py --algo aco --alpha 1.2 --beta 2.5 --rho 0.15\n"
            "  python run.py --config configs/demo.json  # Cargar parámetros desde archivo JSON\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--demo",
        action="store_true",
        help="Modo demostración en vivo (valor de hora INE (ETCL) 25.0 EUR/h).",
    )
    p.add_argument(
        "--algo",
        choices=ALGORITMOS_VALIDOS,
        default="both",
        help="Algoritmo a ejecutar: 'aco', 'ga', 'both' o 'all' (por defecto: both).",
    )
    p.add_argument(
        "--horas",
        nargs="+",
        type=float,
        default=None,
        help="Uno o varios valores de hora del vendedor en EUR/h (ej. --horas 0.0 25.0 60.0).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Ruta a un archivo JSON de configuración personalizada.",
    )
    p.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Tamaño de población / número de hormigas por iteración.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Número máximo de iteraciones.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla del generador pseudoaleatorio para reproducibilidad.",
    )

    # Hiperparámetros específicos de ACO
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="[ACO] Exponente del rastro de feromona (por defecto: 1.0).",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=None,
        help="[ACO] Exponente de la visibilidad heurística (por defecto: 2.0).",
    )
    p.add_argument(
        "--rho",
        type=float,
        default=None,
        help="[ACO] Tasa de evaporación de feromonas (por defecto: 0.1).",
    )

    # Hiperparámetros específicos de GA
    p.add_argument(
        "--prob-mut",
        type=float,
        default=None,
        help="[GA] Probabilidad de mutación genética (por defecto: 0.1).",
    )

    # Flags visuales y utilitarios
    p.add_argument(
        "--gif",
        action="store_true",
        help="Regenera las animaciones GIF y el mosaico comparativo en GIFS/.",
    )
    p.add_argument(
        "--generar-matrices",
        action="store_true",
        help="Regenera las matrices base en Informacion_Preparada/data/.",
    )
    p.add_argument(
        "--generar-instancias",
        action="store_true",
        help="Regenera los archivos .tsp en Informacion_Preparada/instancias_tsp/.",
    )

    return p


def resolver_parametros(args: argparse.Namespace, config: dict) -> dict:
    """
    Resuelve y consolida los hiperparámetros finales de ejecución aplicando la jerarquía:
    Línea de comandos (CLI) > Archivo de configuración (JSON) > Valores por defecto.

    Parámetros:
        args (argparse.Namespace): Argumentos parseados provistos por el usuario en consola.
        config (dict): Diccionario cargado desde el archivo JSON de configuración.

    Retorna:
        dict: Diccionario consolidado con las claves:
            - 'valores_hora' (list[float]): Lista de tarifas horarias del vendedor (€/h).
            - 'semilla' (int): Semilla para generación pseudoaleatoria.
            - 'tamano_poblacion' (int): Tamaño de población de hormigas.
            - 'max_iteraciones' (int): Número máximo de iteraciones.
            - 'alpha' (float): Exponente del rastro de feromonas (ACO).
            - 'beta' (float): Exponente de la visibilidad heurística (ACO).
            - 'rho' (float): Tasa de evaporación de feromonas (ACO).
            - 'prob_mutacion' (float): Probabilidad de mutación genética (GA).
            - 'algoritmos' (list[str]): Lista normalizada con los algoritmos a ejecutar (['ACO'], ['GA'] o ambos).
    """
    exp = config.get("experiment", {})
    common = config.get("common", {})
    aco_cfg = config.get("aco", {})
    ga_cfg = config.get("ga", {})

    # Jerarquia
    def elegir(val_cli, val_cfg, fallback):
        if val_cli is not None:
            return val_cli
        if val_cfg is not None:
            return val_cfg
        return fallback

    valores_hora = elegir(args.horas, exp.get("valores_hora"), [0.0, 15.0, 25.0, 35.0, 60.0])
    seed = elegir(args.seed, exp.get("seed"), 42)
    pop_size = elegir(args.pop_size, common.get("pop_size"), 60)
    max_iter = elegir(args.max_iter, common.get("max_iter"), 150)

    alpha = elegir(args.alpha, aco_cfg.get("alpha"), 1.0)
    beta = elegir(args.beta, aco_cfg.get("beta"), 2.0)
    rho = elegir(args.rho, aco_cfg.get("rho"), 0.1)
    prob_mut = elegir(args.prob_mut, ga_cfg.get("prob_mut"), 0.1)

    # Normalizar algoritmos
    if args.algo in ("both", "all"):
        algos = ["ACO", "GA"]
    elif args.algo == "aco":
        algos = ["ACO"]
    elif args.algo == "ga":
        algos = ["GA"]
    else:
        algos = ["ACO", "GA"]

    return {
        "valores_hora": [float(h) for h in valores_hora],
        "semilla": int(seed),
        "tamano_poblacion": int(pop_size),
        "max_iteraciones": int(max_iter),
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "prob_mutacion": float(prob_mut),
        "algoritmos": algos,
    }


def main() -> None:
    """
    Punto de entrada principal para la ejecución por línea de comandos.

    Flujo de ejecución:
    1. Configura la codificación de la consola para soporte de caracteres UTF-8.
    2. Parsea los argumentos de línea de comandos.
    3. Regenera opcionalmente las matrices base o instancias TSPLIB si se solicitó.
    4. Carga la configuración base (JSON o valores por defecto) y resuelve los parámetros finales.
    5. Invoca la batería de experimentos comparativos (ACO y/o GA) imprimiendo la tabla resumen.
    6. Genera animaciones y figuras en caso de incluir '--gif'.
    7. Mide y muestra el tiempo total de cómputo.

    Parámetros:
        Ninguno.

    Retorna:
        None.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = crear_parser()
    args = parser.parse_args()

    # Regeneración de matrices base si se solicita explícitamente
    if args.generar_matrices:
        print("[INFO] Regenerando matrices base de España...")
        generar_todas_las_matrices()

    # Regeneración de instancias TSPLIB si se solicita explícitamente
    if args.generar_instancias:
        print("[INFO] Regenerando instancias TSPLIB (.tsp)...")
        generar_instancias_tsp()

    # Selección de archivo de configuración
    if args.config:
        ruta_cfg = Path(args.config)
    elif args.demo:
        ruta_cfg = RAIZ_PARTE2 / "configs" / "demo.json"
    else:
        ruta_cfg = RAIZ_PARTE2 / "configs" / "CLI_defaults_tsp.json"

    config = cargar_config(ruta_cfg)
    params = resolver_parametros(args, config)

    t0 = time.time()

    # Ejecutar experimentos principales
    ejecutar_experimentos_comparativos(
        valores_hora=params["valores_hora"],
        tamano_poblacion=params["tamano_poblacion"],
        max_iteraciones=params["max_iteraciones"],
        semilla=params["semilla"],
        alpha=params["alpha"],
        beta=params["beta"],
        rho=params["rho"],
        prob_mutacion=params["prob_mutacion"],
        algoritmos=params["algoritmos"],
    )

    # Generación de visualizaciones si se incluye el flag --gif
    if args.gif:
        print("\n[INFO] Generando visualizaciones y GIFs...")
        sufijos_horas = [
            f"hora_{int(h)}" if float(h).is_integer() else f"hora_{h:.1f}".replace(".", "_")
            for h in params["valores_hora"]
        ]
        ejecutar_orquestador_visual(
            algoritmos_filtrar=params["algoritmos"],
            experimentos_filtrar=sufijos_horas,
        )

    t_total = time.time() - t0
    print(f"\n[OK] Ejecución completada en {t_total:.2f} segundos.")


if __name__ == "__main__":
    main()
