import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tsplib95

# Compatibilidad con NumPy 2.x
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

from sko.ACA import ACA_TSP
from sko.GA import GA_TSP

# El valor de hora dado por el INE (ETCL) esta alrededor de 25 euros la hora
VALORES_HORA_DEFAULT: List[float] = [0.0, 15.0, 25.0, 35.0, 50.0, 75.0]


def cargar_instancia_tsp(ruta_tsp: Path) -> Tuple[tsplib95.models.StandardProblem, np.ndarray]:
    """
    Carga una instancia .tsp usando tsplib95 y extrae la matriz de costos como array np.

    parametros:
        ruta_tsp (Path): Ruta al archivo .tsp.

    retorna:
        Tuple[tsplib95.models.StandardProblem, np.ndarray]:
            - problema: Objeto StandardProblem parseado por la librería tsplib95.
            - matriz_costos: Matriz bidimensional (N x N) con los costos de viaje entre pares de ciudades.

    """
    problema = tsplib95.load(str(ruta_tsp))
    matriz_costos = np.array(problema.edge_weights, dtype=float)
    return problema, matriz_costos


def funcion_objetivo(recorrido: List[int], matriz_costos: np.ndarray) -> float:
    """
    Calcula el costo total de un recorrido cíclico sobre una matriz de costos.

    parametros:
        recorrido (List[int]): Secuencia ordenada de índices de ciudades que componen la ruta.
        matriz_costos (np.ndarray): Matriz (N x N) con los costos entre todas las ciudades.

    retorna:
        float: Costo total acumulado de la ruta cíclica.
    """
    n = len(recorrido)
    # % para conectar la última ciudad con la primera
    return sum(matriz_costos[recorrido[i % n], recorrido[(i + 1) % n]] for i in range(n))


def calcular_desglose_costos(
    ruta_indices: List[int],
    df_combustible: pd.DataFrame,
    df_peajes: pd.DataFrame,
    df_tiempos: pd.DataFrame,
    df_distancias: pd.DataFrame,
    valor_hora: float,
) -> Dict[str, float]:
    """
    Calcula el desglose económico y logístico (combustible, peajes, tiempo y distancia)
    para un recorrido cíclico dado (por iteracion).

    parametros:
        ruta_indices (List[int]): Lista con la secuencia ordenada de índices de ciudades que componen el recorrido.
        df_combustible (pd.DataFrame): Matriz (N x N) con los costos en combustible (€) entre cada par de ciudades.
        df_peajes (pd.DataFrame): Matriz (N x N) con los costos de peajes de autopista (€) entre cada par de ciudades.
        df_tiempos (pd.DataFrame): Matriz (N x N) con los tiempos de viaje en carretera (horas) entre cada par de ciudades.
        df_distancias (pd.DataFrame): Matriz (N x N) con las distancias de conducción (km) entre cada par de ciudades.
        valor_hora (float): Valor de la hora del vendedor (€/h).

    retorna:
        Dict[str, float]: Diccionario con el desglose acumulado de:
            - 'distancia_total_km': Kilómetros totales recorridos en el recorrido.
            - 'costo_combustible_eur': Gasto total acumulado en combustible (€).
            - 'costo_peajes_eur': Gasto total acumulado en peajes (€).
            - 'tiempo_total_horas': Horas totales de conducción estimadas.
            - 'costo_tiempo_vendedor_eur': Costo laboral imputado (tiempo_total_horas * valor_hora) (€).
            - 'costo_total_eur': Suma total del costo general de la ruta (€).

    """
    n = len(ruta_indices)
    costo_combustible = 0.0
    costo_peajes = 0.0
    tiempo_horas = 0.0
    distancia_km = 0.0

    for i in range(n):
        u = ruta_indices[i]
        v = ruta_indices[(i + 1) % n] 
        costo_combustible += df_combustible.iloc[u, v]
        costo_peajes += df_peajes.iloc[u, v]
        tiempo_horas += df_tiempos.iloc[u, v]
        distancia_km += df_distancias.iloc[u, v]

    costo_tiempo = tiempo_horas * valor_hora
    costo_total = costo_combustible + costo_peajes + costo_tiempo

    return {
        "distancia_total_km": round(float(distancia_km), 2),
        "costo_combustible_eur": round(float(costo_combustible), 2),
        "costo_peajes_eur": round(float(costo_peajes), 2),
        "tiempo_total_horas": round(float(tiempo_horas), 2),
        "costo_tiempo_vendedor_eur": round(float(costo_tiempo), 2),
        "costo_total_eur": round(float(costo_total), 2),
    }


def ejecutar_colonia_hormigas(
    matriz_costos: np.ndarray,
    tamano_poblacion: int = 60,
    max_iteraciones: int = 150,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    semilla: int = 42,
) -> Tuple[List[int], float, List[List[int]], List[float], float]:
    """
    Ejecuta el algoritmo ACA_TSP de scikit-opt sobre una matriz de costos.

    parametros:
        matriz_costos (np.ndarray): Matriz (N x N) con los costos entre todas las ciudades.
        tamano_poblacion (int): Número de hormigas generadas por iteración.
        max_iteraciones (int): Número máximo de iteraciones.
        alpha (float): Exponente que controla la influencia del rastro de feromona en la transición.
        beta (float): Exponente que controla la visibilidad heurística (inverso del costo de la arista).
        rho (float): Tasa de evaporación de feromonas.
        semilla (int): Semilla del generador pseudoaleatorio.

    Retorna:
        mejor_ruta (list): Índices de la mejor ruta encontrada.
        mejor_costo (float): Costo total de la mejor ruta.
        historial_rutas (list): Mejor ruta en cada iteración.
        historial_costos (list): Mejor costo en cada iteración.
        tiempo_segundos (float): Tiempo de cómputo en segundos, redondeado a 2 decimales.
    """
    np.random.seed(semilla)
    n_ciudades = len(matriz_costos)

    t_inicio = time.time()
    aca = ACA_TSP(
        func=lambda recorrido: funcion_objetivo(recorrido, matriz_costos),
        n_dim=n_ciudades,
        size_pop=tamano_poblacion,
        max_iter=max_iteraciones,
        distance_matrix=matriz_costos,
        alpha=alpha,
        beta=beta,
        rho=rho,
    )
    mejor_x, mejor_y = aca.run()
    t_fin = time.time()

    historial_rutas = [arr.tolist() for arr in aca.x_best_history]
    historial_costos = [float(costo) for costo in aca.y_best_history]

    return mejor_x.tolist(), float(mejor_y), historial_rutas, historial_costos, round(t_fin - t_inicio, 2)


def ejecutar_algoritmo_genetico(
    matriz_costos: np.ndarray,
    tamano_poblacion: int = 60,
    max_iteraciones: int = 150,
    prob_cruce: float = 0.8,
    prob_mutacion: float = 0.1,
    semilla: int = 42,
) -> Tuple[List[int], float, List[List[int]], List[float], float]:
    """
    Ejecuta el algoritmo GA_TSP (Genetico) de scikit-opt sobre una matriz de costos dada.

    parametros:
        matriz_costos (np.ndarray): Matriz (N x N) con los costos entre todas las ciudades.
        tamano_poblacion (int): Número de individuos generados por iteración.
        max_iteraciones (int): Número máximo de iteraciones.
        prob_cruce (float): Probabilidad de cruce genético entre individuos.
        prob_mutacion (float): Probabilidad de mutación por individuo.
        semilla (int): Semilla del generador pseudoaleatorio.

    Retorna:
        mejor_ruta (list): Índices de la mejor ruta encontrada.
        mejor_costo (float): Costo total de la mejor ruta.
        historial_rutas (list): Mejor ruta en cada iteración.
        historial_costos (list): Mejor costo en cada iteración.
        tiempo_segundos (float): Tiempo de cómputo en segundos, redondeado a 2 decimales.
    """
    np.random.seed(semilla)
    n_ciudades = len(matriz_costos)

    t_inicio = time.time()
    ga = GA_TSP(
        func=lambda recorrido: funcion_objetivo(recorrido, matriz_costos),
        n_dim=n_ciudades,
        size_pop=tamano_poblacion,
        max_iter=max_iteraciones,
        prob_mut=prob_mutacion
    )
    mejor_x, mejor_y = ga.run()
    t_fin = time.time()

    historial_rutas = [arr.tolist() for arr in ga.generation_best_X]
    # En GA_TSP, generation_best_Y contiene escalares, pero run() siempre devuelve mejor_y como array (1,)
    historial_costos = [float(costo) for costo in ga.generation_best_Y]
    mejor_y_val = float(mejor_y[0])

    return mejor_x.tolist(), mejor_y_val, historial_rutas, historial_costos, round(t_fin - t_inicio, 2)


def ejecutar_experimentos_comparativos(
    carpeta_instancias: str = "instancias_tsp",
    carpeta_salida_datos: str = "datos_graficas",
    valores_hora: Optional[List[float]] = None,
    tamano_poblacion: int = 60,
    max_iteraciones: int = 150,
    semilla: int = 42,
) -> Dict[str, Any]:
    """
    Resuelve instancias TSP almacenando historiales completos tanto para la Colonia de 
    Hormigas (ACO) como para el Algoritmo Genético (GA), exportando los datos consolidados
    necesarios para análisis y visualización comparativa.

    parametros:
        carpeta_instancias (str): Directorio que contiene los .tsp.
        carpeta_salida_datos (str): Directorio donde se guardan los archivos de resultados (.json y .csv).
        valores_hora (list de float, opcional): Lista de valores (€/h) del valor de la hora del vendedor a procesar.
        tamano_poblacion (int): Número de individuos/hormigas por iteración.
        max_iteraciones (int): Número máximo de iteraciones.
        semilla (int): Semilla del generador pseudoaleatorio.

    Retorna:
        Dict[str, Any]: Diccionario con los resultados de los experimentos, estructurado en:
            - 'metadata' (dict):
                * 'algoritmos_comparados': Lista con los nombres de las metaheurísticas ejecutadas (ACO y GA).
                * 'ciudades' (list[str]): Lista con los nombres de las 47 ciudades.
                * 'tamano_poblacion' (int): Número de individuos / hormigas configurado.
                * 'max_iteraciones' (int): Cantidad máxima de iteraciones ejecutadas.
                * 'fecha' (str): Marca de tiempo del experimento.
            - 'experimentos' (dict): Diccionario indexado por cada instancia ('hora_{valor_hora}'), donde cada una contiene:
                * 'valor_hora_vendedor': Valor de la hora del vendedor en EUR/h.
                * 'ACO' y 'GA' (dict): Resultados de cada algoritmo:
                    - 'mejor_costo_total' (float): Costo monetario total de la mejor ruta hallada (€).
                    - 'tiempo_computo_s' (float): Tiempo de cómputo en segundos.
                    - 'desglose' (dict): Desglose logístico y económico (distancia_total_km, costo_combustible_eur, costo_peajes_eur, tiempo_total_horas, costo_tiempo_vendedor_eur, costo_total_eur).
                    - 'mejor_ruta_indices' (list[int]): Lista de índices de ciudades que componen el mejor circuito.
                    - 'mejor_ruta_ciudades' (list[str]): Lista de nombres de ciudades correspondientes a la mejor ruta.
                    - 'historial_costos' (list[float]): Evolución del costo mínimo en cada iteración.
                    - 'historial_rutas' (list[list[int]]): Historial de las mejores trayectorias por iteración.
    """
    directorio_base = Path(__file__).resolve().parent
    dir_instancias = directorio_base / carpeta_instancias
    dir_salida = directorio_base / carpeta_salida_datos
    dir_salida.mkdir(parents=True, exist_ok=True)

    # Cargar matrices base para desglose
    df_ciudades = pd.read_csv(directorio_base / "ciudades_espana.csv")
    df_combustible = pd.read_csv(directorio_base / "combustible_espana_eur.csv", index_col=0)
    df_peajes = pd.read_csv(directorio_base / "peajes_espana_eur.csv", index_col=0)
    df_tiempos = pd.read_csv(directorio_base / "tiempos_espana_h.csv", index_col=0)
    df_distancias = pd.read_csv(directorio_base / "distancias_espana_km.csv", index_col=0)

    nombres_ciudades = list(df_ciudades["ciudad"])

    if valores_hora is None:
        valores_hora = VALORES_HORA_DEFAULT

    print("=" * 80)
    print("EJECUTANDO ACO Y GA SOBRE INSTANCIAS TSP DE ESPAÑA")
    print(f"Hormigas por generación: {tamano_poblacion} | Iteraciones: {max_iteraciones}")
    print("=" * 80)

    resultados_globales: Dict[str, Any] = {
        "metadata": {
            "algoritmos_comparados": ["Colonia de Hormigas (ACO)", "Algoritmo Genético (GA)"],
            "ciudades": nombres_ciudades,
            "tamano_poblacion": tamano_poblacion,
            "max_iteraciones": max_iteraciones,
        },
        "experimentos": {},
    }

    resumen_filas = []
    convergencia_dict: Dict[str, List[float]] = {}

    for valor_hora in valores_hora:
        # Cambiar el . por un _ si es decimal
        sufijo = f"{int(valor_hora)}" if float(valor_hora).is_integer() else f"{valor_hora:.1f}".replace(".", "_")
        ruta_tsp = dir_instancias / f"espana47_hora_{sufijo}.tsp"
        
        if not ruta_tsp.exists():
            print(f"No se encontró la instancia '{ruta_tsp}'.")
            continue

        print(f"\nInstancia: '{ruta_tsp.name}' (Valor hora: {valor_hora:.1f} EUR/h)")
        problema, matriz_costos = cargar_instancia_tsp(ruta_tsp)

        # Ejecutar ACO
        mejor_ruta_aco, mejor_costo_aco, hist_rutas_aco, hist_costos_aco, t_computo_aco = ejecutar_colonia_hormigas(
            matriz_costos=matriz_costos, tamano_poblacion=tamano_poblacion, max_iteraciones=max_iteraciones, semilla=semilla
        )
        desglose_aco = calcular_desglose_costos(mejor_ruta_aco, df_combustible, df_peajes, df_tiempos, df_distancias, valor_hora)
        nombres_ruta_aco = [nombres_ciudades[idx] for idx in mejor_ruta_aco]

        # Ejecutar GA
        mejor_ruta_ga, mejor_costo_ga, hist_rutas_ga, hist_costos_ga, t_computo_ga = ejecutar_algoritmo_genetico(
            matriz_costos=matriz_costos, tamano_poblacion=tamano_poblacion, max_iteraciones=max_iteraciones, semilla=semilla
        )
        desglose_ga = calcular_desglose_costos(mejor_ruta_ga, df_combustible, df_peajes, df_tiempos, df_distancias, valor_hora)
        nombres_ruta_ga = [nombres_ciudades[idx] for idx in mejor_ruta_ga]

        print(f"  [ACO] Costo: {mejor_costo_aco:.2f} EUR | Tiempo de cómputo: {t_computo_aco} s")
        print(f"  [GA]  Costo: {mejor_costo_ga:.2f} EUR | Tiempo de cómputo: {t_computo_ga} s")

        # Almacenamiento de datos (Ambos algoritmos)
        clave_exp = f"hora_{sufijo}"
        convergencia_dict[f"{clave_exp}_ACO"] = hist_costos_aco
        convergencia_dict[f"{clave_exp}_GA"] = hist_costos_ga

        resultados_globales["experimentos"][clave_exp] = {
            "valor_hora_vendedor": valor_hora,
            "ACO": {
                "mejor_costo_total": mejor_costo_aco,
                "tiempo_computo_s": t_computo_aco,
                "desglose": desglose_aco,
                "mejor_ruta_indices": mejor_ruta_aco,
                "mejor_ruta_ciudades": nombres_ruta_aco,
                "historial_costos": hist_costos_aco,
                "historial_rutas": hist_rutas_aco,
            },
            "GA": {
                "mejor_costo_total": mejor_costo_ga,
                "tiempo_computo_s": t_computo_ga,
                "desglose": desglose_ga,
                "mejor_ruta_indices": mejor_ruta_ga,
                "mejor_ruta_ciudades": nombres_ruta_ga,
                "historial_costos": hist_costos_ga,
                "historial_rutas": hist_rutas_ga,
            }
        }

        # Filas para el dataframe resumen
        resumen_filas.append({"Algoritmo": "ACO", "valor_hora_eur": valor_hora, "tiempo_computo_s": t_computo_aco, **desglose_aco})
        resumen_filas.append({"Algoritmo": "GA", "valor_hora_eur": valor_hora, "tiempo_computo_s": t_computo_ga, **desglose_ga})

    # Guardar resultados en el disco
    ruta_json = dir_salida / "resultados_comparativos.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados_globales, f, indent=2, ensure_ascii=False)

    columnas_orden = [
        "Algoritmo",
        "valor_hora_eur",
        "costo_total_eur",
        "distancia_total_km",
        "tiempo_total_horas",
        "costo_combustible_eur",
        "costo_peajes_eur",
        "costo_tiempo_vendedor_eur",
        "tiempo_computo_s",
    ]
    df_resumen = pd.DataFrame(resumen_filas)[columnas_orden]
    ruta_resumen = dir_salida / "resumen_comparativo.csv"
    df_resumen.to_csv(ruta_resumen, index=False)

    df_convergencia = pd.DataFrame(convergencia_dict)
    df_convergencia.index.name = "iteracion"
    ruta_convergencia = dir_salida / "convergencia_comparativa.csv"
    df_convergencia.to_csv(ruta_convergencia)

    # Formatear tabla (pandas la formatea raro en la terminal)
    header = f"{'Algo':<5} {'Hora(EUR)':>9} {'Total(EUR)':>11} {'Dist(km)':>10} {'Tpo(h)':>8} {'Comb(EUR)':>10} {'Peaje(EUR)':>10} {'C.Tpo(EUR)':>11} {'CPU(s)':>8}"
    sep_line = "-" * len(header)
    sep_double = "=" * len(header)

    print("\n" + sep_double)
    print("RESUMEN COMPARATIVO (ACO vs GA) X VALOR DE HORA DEL VENDEDOR:")
    print(sep_double)
    print(header)
    print(sep_double)

    last_hora = None
    for _, fila in df_resumen.iterrows():
        hora = fila["valor_hora_eur"]
        if last_hora is not None and hora != last_hora:
            print(sep_line)
        last_hora = hora
        print(
            f"{fila['Algoritmo']:<5} "
            f"{hora:>9.1f} "
            f"{fila['costo_total_eur']:>11.2f} "
            f"{fila['distancia_total_km']:>10.2f} "
            f"{fila['tiempo_total_horas']:>8.2f} "
            f"{fila['costo_combustible_eur']:>10.2f} "
            f"{fila['costo_peajes_eur']:>10.2f} "
            f"{fila['costo_tiempo_vendedor_eur']:>11.2f} "
            f"{fila['tiempo_computo_s']:>8.2f}"
        )

    print(sep_double)
    print(f"Archivos exportados exitosamente en: '{dir_salida.relative_to(directorio_base)}'")
    print(sep_double)

    return resultados_globales


if __name__ == "__main__":
    ejecutar_experimentos_comparativos()