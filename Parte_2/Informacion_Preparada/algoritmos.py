"""
Optimización del Problema del Viajante (TSP) para España peninsular (47 ciudades)
mediante Algoritmo de Colonia de Hormigas (ACO) y Algoritmo Genético (GA) usando scikit-opt.

Resuelve las instancias generadas con diferentes valores de la hora del vendedor
y exporta los datos estructurados unificados necesarios para análisis y visualización.
"""

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

VALORES_HORA_DEFAULT: List[float] = [0.0, 15.0, 25.0, 35.0, 50.0, 75.0]

def cargar_instancia_tsp(ruta_tsp: Path) -> Tuple[tsplib95.models.StandardProblem, np.ndarray]:
    """Carga una instancia .tsp usando tsplib95 y extrae la matriz de costos como array np."""
    problema = tsplib95.load(str(ruta_tsp))
    matriz_costos = np.array(problema.edge_weights, dtype=float)
    return problema, matriz_costos


def calcular_desglose_costos(
    ruta_indices: List[int],
    df_combustible: pd.DataFrame,
    df_peajes: pd.DataFrame,
    df_tiempos: pd.DataFrame,
    df_distancias: pd.DataFrame,
    valor_hora: float,
) -> Dict[str, float]:
    """Calcula el desglose económico y logístico detallado para un recorrido cíclico dado."""
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
    """Ejecuta el algoritmo ACA_TSP (Hormigas) de scikit-opt sobre una matriz de costos."""
    np.random.seed(semilla)
    n_ciudades = len(matriz_costos)

    def funcion_objetivo(recorrido):
        n = len(recorrido)
        return sum(matriz_costos[recorrido[i % n], recorrido[(i + 1) % n]] for i in range(n))

    t_inicio = time.time()
    aca = ACA_TSP(
        func=funcion_objetivo,
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
    """Ejecuta el algoritmo GA_TSP (Genético) de scikit-opt sobre una matriz de costos."""
    np.random.seed(semilla)
    n_ciudades = len(matriz_costos)

    def funcion_objetivo(recorrido):
        n = len(recorrido)
        return sum(matriz_costos[recorrido[i % n], recorrido[(i + 1) % n]] for i in range(n))

    t_inicio = time.time()
    ga = GA_TSP(
        func=funcion_objetivo,
        n_dim=n_ciudades,
        size_pop=tamano_poblacion,
        max_iter=max_iteraciones,
        prob_mut=prob_mutacion
    )
    mejor_x, mejor_y = ga.run()
    t_fin = time.time()

    historial_rutas = [arr.tolist() for arr in ga.generation_best_X]
    # scikit-opt puede devolver mejor_y como un array unidimensional en GA, por eso se extrae el primer valor
    historial_costos = [float(costo[0] if isinstance(costo, (list, np.ndarray)) else costo) for costo in ga.generation_best_Y]
    mejor_y_val = float(mejor_y[0] if isinstance(mejor_y, (list, np.ndarray)) else mejor_y)

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
    Hormigas como para el Algoritmo Genético, exportando los datos consolidados.
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
    print(f"Individuos/Hormigas por generación: {tamano_poblacion} | Iteraciones: {max_iteraciones}")
    print("=" * 80)

    resultados_globales: Dict[str, Any] = {
        "metadata": {
            "algoritmos_comparados": ["Colonia de Hormigas (ACO)", "Algoritmo Genético (GA)"],
            "ciudades": nombres_ciudades,
            "tamano_poblacion": tamano_poblacion,
            "max_iteraciones": max_iteraciones,
            "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "experimentos": {},
    }

    resumen_filas = []
    convergencia_dict: Dict[str, List[float]] = {}

    for valor_hora in valores_hora:
        sufijo = f"{int(valor_hora)}" if float(valor_hora).is_integer() else f"{valor_hora:.1f}".replace(".", "_")
        ruta_tsp = dir_instancias / f"espana47_hora_{sufijo}.tsp"
        
        if not ruta_tsp.exists():
            print(f"Advertencia: No se encontró la instancia '{ruta_tsp}'. Saltando...")
            continue

        print(f"\nProcesando Instancia: '{ruta_tsp.name}' (Valor hora: {valor_hora:.1f} EUR/h)")
        problema, matriz_costos = cargar_instancia_tsp(ruta_tsp)

        # 1. EJECUCIÓN ACO
        mejor_ruta_aco, mejor_costo_aco, hist_rutas_aco, hist_costos_aco, t_computo_aco = ejecutar_colonia_hormigas(
            matriz_costos=matriz_costos, tamano_poblacion=tamano_poblacion, max_iteraciones=max_iteraciones, semilla=semilla
        )
        desglose_aco = calcular_desglose_costos(mejor_ruta_aco, df_combustible, df_peajes, df_tiempos, df_distancias, valor_hora)
        nombres_ruta_aco = [nombres_ciudades[idx] for idx in mejor_ruta_aco]

        # 2. EJECUCIÓN GA
        mejor_ruta_ga, mejor_costo_ga, hist_rutas_ga, hist_costos_ga, t_computo_ga = ejecutar_algoritmo_genetico(
            matriz_costos=matriz_costos, tamano_poblacion=tamano_poblacion, max_iteraciones=max_iteraciones, semilla=semilla
        )
        desglose_ga = calcular_desglose_costos(mejor_ruta_ga, df_combustible, df_peajes, df_tiempos, df_distancias, valor_hora)
        nombres_ruta_ga = [nombres_ciudades[idx] for idx in mejor_ruta_ga]

        print(f"  [ACO] Costo: {mejor_costo_aco:.2f} EUR | Tiempo de cómputo: {t_computo_aco} s")
        print(f"  [GA]  Costo: {mejor_costo_ga:.2f} EUR | Tiempo de cómputo: {t_computo_ga} s")

        # 3. ALMACENAMIENTO UNIFICADO
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

    df_resumen = pd.DataFrame(resumen_filas)
    ruta_resumen = dir_salida / "resumen_comparativo.csv"
    df_resumen.to_csv(ruta_resumen, index=False)

    df_convergencia = pd.DataFrame(convergencia_dict)
    df_convergencia.index.name = "iteracion"
    ruta_convergencia = dir_salida / "convergencia_comparativa.csv"
    df_convergencia.to_csv(ruta_convergencia)

    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO (ACO vs GA) X VALOR DE HORA DEL VENDEDOR:")
    print("=" * 80)
    print(df_resumen.to_string(index=False))
    print("=" * 80)
    print(f"Archivos exportados exitosamente en: '{dir_salida.relative_to(directorio_base)}'")
    print("=" * 80)

    return resultados_globales


if __name__ == "__main__":
    ejecutar_experimentos_comparativos()