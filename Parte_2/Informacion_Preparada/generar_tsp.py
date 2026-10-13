from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import tsplib95


VALORES_HORA_DEFAULT: List[float] = [0.0, 15.0, 25.0, 35.0, 50.0, 75.0]


def crear_problema_tsp(
    df_costos: pd.DataFrame,
    nombre: str = "espana47",
    tipo_problema: str = "ATSP",
    comentario: Optional[str] = None,
) -> tsplib95.models.StandardProblem:
    """
    Construye un objeto StandardProblem de tsplib95 en modo "EXPLICIT FULL_MATRIX".

    Parámetros:
        df_costos (pd.DataFrame): Matriz de costos totales (47x47) en euros.
        nombre (str): Nombre de la instancia TSP.
        tipo_problema (str): Tipo de problema TSPLIB ('TSP' o 'Asymmetric TSP').
        comentario (str, opcional): Comentario descriptivo del problema.

    Retorna:
        tsplib95.models.StandardProblem: Instancia de TSP en TSPLIB95.
    """
    dimension = len(df_costos)

    if comentario is None:
        comentario = (
            "47 capitales provinciales de Espana peninsular.\n"
            "Matriz de costos totales en EUR calculada a partir del Modelo de Costo:\n"
            "C_total = C_combustible + C_peajes + (Tiempo_h * Valor_Hora).\n"
            "Vehiculo de referencia: Toyota Corolla Berlina 1995 (E100, consumo 7.3 L/100 km).\n"
            "Fuentes: OSRM Table API, Red de Carreteras del Estado (MITMA), INE (ETCL)."
        )

    problema = tsplib95.models.StandardProblem()
    problema.name = nombre
    problema.comment = comentario
    problema.type = tipo_problema
    problema.dimension = dimension
    problema.edge_weight_type = "EXPLICIT"
    problema.edge_weight_format = "FULL_MATRIX"

    # Matriz completa de costos (47x47)
    problema.edge_weights = df_costos.values.tolist()

    return problema


def guardar_archivo_tsp(
    problema: tsplib95.models.StandardProblem,
    ruta_salida: Path,
) -> None:
    """
    Exporta la instancia TSP a un archivo .tsp.
    Garantiza el salto de línea final tras la etiqueta EOF para compatibilidad estricta.

    Parámetros:
        problema (tsplib95.models.StandardProblem): Instancia a exportar.
        ruta_salida (Path): Ruta del archivo destino .tsp.
    retorna:
        na
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True) # Crear directorio si no existe
    problema.save(str(ruta_salida))

    # Asegurar que termine con '\n' tras 'EOF' por error (compatiblidad) de carga
    with open(ruta_salida, "r+", encoding="utf-8") as f:
        contenido = f.read()
        if not contenido.endswith("\n"):
            f.write("\n")


def calcular_matriz_costos_hora(
    directorio_base: Path,
    valor_hora: float,
) -> pd.DataFrame:
    """
    Calcula la matriz de costos totales en EUR combinando combustible, peajes
    y el costo temporal del vendedor para un valor horario específico.

    Parámetros:
        directorio_base (Path): Directorio donde estan los CSV con las matrices de costos.
        valor_hora (float): Valor de la hora del vendedor (€/h).

    Retorna:
        pd.DataFrame: Matriz (47x47) con el costo monetario total consolidado entre cada par de ciudades.
    """
    ruta_combustible = directorio_base / "combustible_espana_eur.csv"
    ruta_peajes = directorio_base / "peajes_espana_eur.csv"
    ruta_tiempos = directorio_base / "tiempos_espana_h.csv"

    df_combustible = pd.read_csv(ruta_combustible, index_col=0)
    df_peajes = pd.read_csv(ruta_peajes, index_col=0)
    df_tiempos = pd.read_csv(ruta_tiempos, index_col=0)

    df_costo_tiempo = df_tiempos * valor_hora
    df_costos_totales = df_combustible + df_peajes + df_costo_tiempo

    np.fill_diagonal(df_costos_totales.values, 0.0) # Costo de i -> i debe ser 0
    return df_costos_totales.round(2)


def generar_instancias_tsp(
    valores_hora: Optional[List[float]] = None,
    carpeta_instancias: str = "instancias_tsp",
) -> List[Path]:
    """
    Genera y exporta múltiples archivos .tsp en un directorio "instancias",
    variando el valor de la hora del vendedor.

    Parámetros:
        valores_hora (list de float, opcional): Lista de valores (€/h) a usar.
        carpeta_instancias (str): Nombre del subdirectorio destino.

    Retorna:
        list de Path: Rutas absolutas a los archivos .tsp generados.
    """
    directorio_base = Path(__file__).resolve().parent # Directorio base

    dir_instancias = directorio_base / carpeta_instancias # Directorio de instancias
    dir_instancias.mkdir(parents=True, exist_ok=True)

    if valores_hora is None:
        valores_hora = VALORES_HORA_DEFAULT

    rutas_generadas: List[Path] = []

    # Generar distintos .tsp para diferentes valores de hora
    for v_hora in valores_hora:
        # espana47_hora_0.tsp, espana47_hora_15.tsp, etc.
        sufijo = f"{int(v_hora)}" if float(v_hora).is_integer() else f"{v_hora:.1f}".replace(".", "_")
        nombre_instancia = f"espana47_hora_{sufijo}" # Nombre de instancias
        ruta_tsp = dir_instancias / f"{nombre_instancia}.tsp" # Ruta de instancias

        # matriz de costos
        df_costos = calcular_matriz_costos_hora(directorio_base, v_hora)

        comentario = (
            f"47 capitales de Espana peninsular. Valor hora vendedor: {v_hora:.2f} EUR/h.\n"
            f"Formula: C_total = Combustible + Peajes + (Tiempo_h * {v_hora:.2f})."
        )

        problema = crear_problema_tsp(
            df_costos=df_costos,
            nombre=nombre_instancia,
            tipo_problema="TSP",
            comentario=comentario,
        )

        guardar_archivo_tsp(problema, ruta_tsp)
        rutas_generadas.append(ruta_tsp)
        print(f"{nombre_instancia}.tsp  ({v_hora:.1f} EUR/h) -> '{ruta_tsp.name}'")

    return rutas_generadas


if __name__ == "__main__":
    generar_instancias_tsp()
