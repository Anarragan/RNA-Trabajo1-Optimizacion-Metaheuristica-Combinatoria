"""
Optimización del Problema del Viajante (TSP) para España peninsular (47 ciudades)
mediante el Algoritmo de Colonia de Hormigas (ACO) de scikit-opt (sko.ACA.ACA_TSP).

Resuelve las instancias generadas con diferentes valores de la hora del vendedor
y exporta los datos estructurados necesarios para análisis y visualización (GIFs).
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tsplib95

# Compatibilidad con NumPy 2.x: la librería scikit-opt usa internamente np.int y np.float (removidos en NumPy 2.0+)
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

from sko.ACA import ACA_TSP

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


def calcular_desglose_costos(
    ruta_indices: List[int],
    df_combustible: pd.DataFrame,
    df_peajes: pd.DataFrame,
    df_tiempos: pd.DataFrame,
    df_distancias: pd.DataFrame,
    valor_hora: float,
) -> Dict[str, float]:
    """
    Calcula el desglose económico y logístico detallado (combustible, peajes, tiempo y distancia)
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
        v = ruta_indices[(i + 1) % n] # % para el ultimo tramo vuelve a la primera ciudad
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


def ejecutar_colonia_hormigas( # Retorna lo necesario para hacer los gifts
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
        alpha (float): Exponente que pondera la influencia del rastro de feromona en la transición.
        beta (float): Exponente que pondera la visibilidad heurística (inverso del costo de la arista).
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

    def funcion_objetivo(recorrido):
        n = len(recorrido)
        # Costo total del recorrido usando csv de costos totales
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


def ejecutar_experimentos_aco(
    carpeta_instancias: str = "instancias_tsp",
    carpeta_salida_datos: str = "datos_graficas",
    valores_hora: Optional[List[float]] = None,
    tamano_poblacion: int = 60,
    max_iteraciones: int = 150,
    semilla: int = 42,
) -> Dict[str, Any]:
    """
    Resuelve instancias TSP en el directorio especificado, almacena historiales completos
    y exporta los datos necesarios para visualización y análisis.

    parametros:
        carpeta_instancias (str): Directorio que contiene los .tsp.
        carpeta_salida_datos (str): Directorio donde se guardan los resultados.
        valores_hora (list de float, opcional): Lista de valores (€/h) a procesar.
        tamano_poblacion (int): Número de hormigas por iteración.
        max_iteraciones (int): Número máximo de iteraciones.
        semilla (int): Semilla del generador pseudoaleatorio.

    Retorna:
        dict: Diccionario con los resultados de todos los experimentos.
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
    print("EJECUTANDO COLONIA DE HORMIGAS (scikit-opt) SOBRE INSTANCIAS TSP DE ESPAÑA")
    print(f"Hormigas por generación: {tamano_poblacion} | Iteraciones máximas: {max_iteraciones}")
    print("=" * 80)

    resultados_globales: Dict[str, Any] = {
        "metadata": {
            "algoritmo": "Colonia de Hormigas (scikit-opt, ACA_TSP)",
            "ciudades": nombres_ciudades,
            "tamano_poblacion": tamano_poblacion,
            "max_iteraciones": max_iteraciones,
            "alpha": 1.0,
            "beta": 2.0,
            "rho": 0.1,
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
            raise FileNotFoundError(f"No se encontró la instancia: '{ruta_tsp}'")

        print(f"\nInstancia: '{ruta_tsp.name}' (Valor hora: {valor_hora:.1f} EUR/h)")

        problema, matriz_costos = cargar_instancia_tsp(ruta_tsp)

        mejor_ruta, mejor_costo, hist_rutas, hist_costos, t_computo = ejecutar_colonia_hormigas(
            matriz_costos=matriz_costos,
            tamano_poblacion=tamano_poblacion,
            max_iteraciones=max_iteraciones,
            semilla=semilla,
        )

        desglose = calcular_desglose_costos( # Solo para mejor ruta
            ruta_indices=mejor_ruta,
            df_combustible=df_combustible,
            df_peajes=df_peajes,
            df_tiempos=df_tiempos,
            df_distancias=df_distancias,
            valor_hora=valor_hora,
        )

        nombres_ruta = [nombres_ciudades[idx] for idx in mejor_ruta]

        print(f"    Mejor costo total: {mejor_costo:.2f} EUR (en {t_computo} s)")
        print(f"    Distancia: {desglose['distancia_total_km']:,.2f} km | Tiempo: {desglose['tiempo_total_horas']:.2f} h")
        print(f"    Combustible: {desglose['costo_combustible_eur']:.2f} EUR | Peajes: {desglose['costo_peajes_eur']:.2f} EUR | Tiempo: {desglose['costo_tiempo_vendedor_eur']:.2f} EUR")

        # Data para grafico de convergencia
        clave_exp = f"hora_{sufijo}"
        convergencia_dict[clave_exp] = hist_costos

        resultados_globales["experimentos"][clave_exp] = {
            "archivo_tsp": ruta_tsp.name,
            "valor_hora_vendedor": valor_hora,
            "mejor_costo_total": mejor_costo,
            "tiempo_computo_s": t_computo,
            "desglose": desglose,
            "mejor_ruta_indices": mejor_ruta,
            "mejor_ruta_ciudades": nombres_ruta,
            "historial_costos": hist_costos,
            "historial_rutas": hist_rutas, 
        }

        resumen_filas.append({ # Data para tabla
            "valor_hora_eur": valor_hora,
            "costo_total_eur": desglose["costo_total_eur"],
            "distancia_km": desglose["distancia_total_km"],
            "tiempo_viaje_horas": desglose["tiempo_total_horas"],
            "costo_combustible_eur": desglose["costo_combustible_eur"],
            "costo_peajes_eur": desglose["costo_peajes_eur"],
            "costo_tiempo_eur": desglose["costo_tiempo_vendedor_eur"],
            "tiempo_computo_s": t_computo,
        })

    # Guardar resultados como JSON 
    ruta_json = dir_salida / "resultados_aco.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados_globales, f, indent=2, ensure_ascii=False)

    df_resumen = pd.DataFrame(resumen_filas)
    ruta_resumen = dir_salida / "resumen_experimentos_aco.csv"
    df_resumen.to_csv(ruta_resumen, index=False)

    df_convergencia = pd.DataFrame(convergencia_dict)
    df_convergencia.index.name = "iteracion"
    ruta_convergencia = dir_salida / "convergencia_aco.csv"
    df_convergencia.to_csv(ruta_convergencia)

    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO X VALOR DE HORA DEL VENDEDOR:")
    print("=" * 80)
    print(df_resumen.to_string(index=False))
    print("=" * 80)
    print(f"Datos para graficar en: '{dir_salida.relative_to(directorio_base)}'")
    print(f"     - JSON completo con trayectorias: {ruta_json.name}") # Globales
    print(f"     - Tabla resumen:   {ruta_resumen.name}")
    print(f"     - Data para curva:         {ruta_convergencia.name}")
    print("=" * 80)

    return resultados_globales


if __name__ == "__main__":
    ejecutar_experimentos_aco()
