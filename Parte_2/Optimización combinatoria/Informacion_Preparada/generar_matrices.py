from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Valor de la hora laboral del vendedor en euros (€/h) respaldado por el INE (ETCL)
VALOR_HORA_VENDEDOR_DEFAULT = 25.00  # Según la ETCL hecha por el INE, se sitúa en torno a 24.50 € - 26.18 €/hora.

# Directorio por defecto para almacenar y leer matrices y datos
DIRECTORIO_DATOS_DEFAULT = Path(__file__).resolve().parent / "data"


def cargar_matriz_peajes(
    dir_datos: Optional[Path] = None,
    nombre_archivo_peajes: str = "peajes_espana_eur.csv",
) -> pd.DataFrame:
    """
    Carga y valida la matriz de costos de peaje entre las capitales peninsulares
    previamente consultada y guardada mediante la API de HERE Routing v8.

    Procedimiento:
    1. Verifica que exista el archivo CSV de peajes en 'data/'.
    2. Lee la matriz cuadrada (47x47) usando los nombres de las ciudades como índices y columnas.
    3. Valida alineación con 'ciudades_espana.csv' (si existe), limpia posibles nulos
       y asegura que la diagonal principal sea 0.0.
    4. Muestra un resumen estadístico de los peajes cargados sin sobrescribir el CSV.

    Parámetros:
        dir_datos (Path, opcional): Ruta al directorio de datos. Por defecto 'Informacion_Preparada/data'.
        nombre_archivo_peajes (str): Nombre del archivo CSV de peajes. Por defecto 'peajes_espana_eur.csv'.

    Retorna:
        pd.DataFrame: Matriz simétrica (47x47) con los costos de peaje reales en euros.
    """
    if dir_datos is None:
        dir_datos = DIRECTORIO_DATOS_DEFAULT

    ruta_peajes = dir_datos / nombre_archivo_peajes
    ruta_ciudades = dir_datos / "ciudades_espana.csv"

    if not ruta_peajes.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de peajes en '{ruta_peajes}'. "
            "Ejecuta primero el script de extracción con la API (matrizpeajes.py)."
        )

    df_peajes = pd.read_csv(ruta_peajes, index_col=0)
    df_peajes = df_peajes.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Validar y alinear el orden de ciudades con ciudades_espana.csv si está disponible
    if ruta_ciudades.exists():
        nombres_ciudades = pd.read_csv(ruta_ciudades)["ciudad"].tolist()
        faltantes = set(nombres_ciudades) - set(df_peajes.index)
        if faltantes:
            raise ValueError(
                f"La matriz de peajes no contiene todas las ciudades esperadas. Faltan: {sorted(faltantes)}"
            )
        df_peajes = df_peajes.loc[nombres_ciudades, nombres_ciudades]

    # Asegurar que la diagonal principal (ciudad -> misma ciudad) sea 0.0
    matriz_peajes = df_peajes.to_numpy(copy=True)
    np.fill_diagonal(matriz_peajes, 0.0)
    df_peajes = pd.DataFrame(matriz_peajes, index=df_peajes.index, columns=df_peajes.columns)
    df_peajes = df_peajes.round(2)

    # Estadísticas rápidas de verificación (solo triángulo superior para no contar doble)
    n = len(df_peajes)
    triangulo_superior = df_peajes.values[np.triu_indices(n, k=1)]
    total_pares = len(triangulo_superior)
    pares_con_peaje = int(np.sum(triangulo_superior > 0))
    peaje_max = float(np.max(triangulo_superior)) if total_pares > 0 else 0.0

    print(f"Matriz de peajes cargada exitosamente desde: '{ruta_peajes}'")
    print(f"     Dimensiones: {df_peajes.shape}")
    print(f"     Pares únicos con peaje > 0 €: {pares_con_peaje} / {total_pares}")
    print(f"     Peaje máximo registrado: {peaje_max:.2f} €")

    return df_peajes


def generar_matriz_combustible(
    dir_datos: Optional[Path] = None,
    nombre_archivo_salida: str = "combustible_espana_eur.csv",
) -> Optional[pd.DataFrame]:
    """
    Lee la matriz de distancias y genera una nueva matriz (N x N) con el
    costo monetario del combustible en euros para cada trayecto entre capitales.

    Aplica la siguiente fórmula:
        Costo (€) = (Distancia [km] / 100) * Consumo [L/100km] * Precio [€/L]

    Parámetros:
        dir_datos (Path, opcional): Ruta al directorio de datos. Por defecto 'Informacion_Preparada/data'.
        nombre_archivo_salida (str): Nombre del archivo CSV de salida. Por defecto 'combustible_espana_eur.csv'.

    Retorna:
        Optional[pd.DataFrame]: Matriz con los costos de combustible redondeados a 2 decimales.
    """
    if dir_datos is None:
        dir_datos = DIRECTORIO_DATOS_DEFAULT
    dir_datos.mkdir(parents=True, exist_ok=True)

    ruta_distancias = dir_datos / "distancias_espana_km.csv"
    ruta_salida = dir_datos / nombre_archivo_salida

    # Parámetros del vehículo y mercado (¡Documentar en reporte APA!)
    consumo_l_100km = 5.5     # Litros cada 100 km (Seat León 1.5 TSI)
    precio_euro_litro = 1.55  # Precio por litro (Gasolina 95)

    if not ruta_distancias.exists():
        print(f"Error: No se encontró '{ruta_distancias}'.")
        return None

    # Leer la matriz de distancias que obtuvimos de OSRM
    df_distancias = pd.read_csv(ruta_distancias, index_col=0)

    # Vectorización pura: Cálculo matemático simultáneo para las 2.209 combinaciones
    df_combustible = (df_distancias / 100) * consumo_l_100km * precio_euro_litro
    df_combustible = df_combustible.round(2)

    # Guardar en la carpeta data/
    df_combustible.to_csv(ruta_salida)

    print(f"Matriz de combustible generada exitosamente en: '{ruta_salida}'")
    print(f"     Dimensiones: {df_combustible.shape}")
    print("\nMuestra (Primeras 3x3 ciudades en Euros):")
    print(df_combustible.iloc[:3, :3])
    return df_combustible


def calcular_matriz_costos(
    df_combustible: pd.DataFrame,
    df_peajes: pd.DataFrame,
    df_tiempos: pd.DataFrame,
    valor_hora_vendedor: float = VALOR_HORA_VENDEDOR_DEFAULT,
) -> pd.DataFrame:
    """
    Calcula la matriz de costos totales en EUR combinando combustible, peajes
    y el costo temporal del vendedor para un valor horario específico.

    Fórmula:
        C_total(i, j) = C_combustible(i, j) + C_peaje(i, j) + (Tiempo(i, j) * Valor_Hora)

    Parámetros:
        df_combustible (pd.DataFrame): Matriz de combustible en euros.
        df_peajes (pd.DataFrame): Matriz de peajes en euros.
        df_tiempos (pd.DataFrame): Matriz de tiempos en horas.
        valor_hora_vendedor (float): Costo monetario por hora efectiva del vendedor en euros (€/h).

    Retorna:
        pd.DataFrame: Matriz (47x47) de costos totales redondeada a 2 decimales, diagonal en 0.0.
    """
    df_costo_tiempo = df_tiempos * valor_hora_vendedor
    df_costos_totales = df_combustible + df_peajes + df_costo_tiempo

    # Asegurar que la diagonal principal (i -> i) sea 0.0
    matriz_costos = df_costos_totales.to_numpy(copy=True)
    np.fill_diagonal(matriz_costos, 0.0)
    df_costos_totales = pd.DataFrame(
        matriz_costos, index=df_costos_totales.index, columns=df_costos_totales.columns
    )

    return df_costos_totales.round(2)


def generar_matriz_costos(
    valor_hora_vendedor: float = VALOR_HORA_VENDEDOR_DEFAULT,
    nombre_archivo_salida: str = "costos_totales_espana_eur.csv",
    dir_datos: Optional[Path] = None,
    nombre_archivo_peajes: str = "peajes_espana_eur.csv",
) -> pd.DataFrame:
    """
    Genera y exporta la matriz de costos combinados (combustible, peajes y tiempo del vendedor)
    para las 47 capitales de España peninsular.
    Se usa el Modelo de Costo Generalizado de Transporte:
    C_total(i, j) = C_combustible(i, j) + C_peaje(i, j) + (Tiempo(i, j) * Valor_Hora)

    Parámetros:
        valor_hora_vendedor (float): Costo monetario por hora efectiva del vendedor en euros (€/h).
        nombre_archivo_salida (str): Nombre del archivo CSV a generar.
        dir_datos (Path, opcional): Ruta al directorio de datos. Por defecto 'Informacion_Preparada/data'.
        nombre_archivo_peajes (str): Nombre del archivo CSV de peajes a leer.

    Retorna:
        pd.DataFrame: Matriz de costos totales simétrica con dimensiones 47x47 en euros.
    """
    if dir_datos is None:
        dir_datos = DIRECTORIO_DATOS_DEFAULT
    dir_datos.mkdir(parents=True, exist_ok=True)

    ruta_combustible = dir_datos / "combustible_espana_eur.csv"
    ruta_tiempos = dir_datos / "tiempos_espana_h.csv"
    ruta_salida = dir_datos / nombre_archivo_salida

    # Cargar matrices de costos, tiempo y peajes reales desde disco
    df_combustible = pd.read_csv(ruta_combustible, index_col=0)
    df_tiempos = pd.read_csv(ruta_tiempos, index_col=0)
    df_peajes = cargar_matriz_peajes(
        dir_datos=dir_datos,
        nombre_archivo_peajes=nombre_archivo_peajes,
    )

    df_costos_totales = calcular_matriz_costos(
        df_combustible=df_combustible,
        df_peajes=df_peajes,
        df_tiempos=df_tiempos,
        valor_hora_vendedor=valor_hora_vendedor,
    )

    # Exportar como CSV
    df_costos_totales.to_csv(ruta_salida)

    print(f"Matriz de costos generada en: '{ruta_salida}'")
    print(f"     Dimensiones: {df_costos_totales.shape}")
    print(f"     Valor hora del vendedor aplicado: {valor_hora_vendedor:.2f} EUR/h")
    print("\nPreview:")
    print(df_costos_totales.iloc[:3, :3])

    return df_costos_totales


def generar_todas_las_matrices(
    dir_datos: Optional[Path] = None,
    valor_hora_vendedor: float = VALOR_HORA_VENDEDOR_DEFAULT,
    nombre_archivo_peajes: str = "peajes_espana_eur.csv",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Pipeline de generación de matrices en orden:
    1. Lectura y validación de peajes reales desde CSV ('peajes_espana_eur.csv').
    2. Estimación del consumo y costo de combustible ('combustible_espana_eur.csv').
    3. Consolidación de la matriz de costos totales ('costos_totales_espana_eur.csv').

    Parámetros:
        dir_datos (Path, opcional): Ruta al directorio de datos donde se guardan las matrices.
        valor_hora_vendedor (float): Valor de la hora laboral del vendedor en euros.
        nombre_archivo_peajes (str): Nombre del archivo CSV de peajes consultado con la API.

    Retorna:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
            DataFrames correspondientes a peajes, combustible y costos totales.
    """
    if dir_datos is None:
        dir_datos = DIRECTORIO_DATOS_DEFAULT

    print("=" * 60)
    print("EJECUTANDO GENERADOR UNIFICADO DE MATRICES")
    print(f"Directorio de almacenamiento: '{dir_datos}'")
    print("=" * 60)

    print("\n1. Matriz de peajes (desde CSV de la API)")
    df_peajes = cargar_matriz_peajes(
        dir_datos=dir_datos,
        nombre_archivo_peajes=nombre_archivo_peajes,
    )

    print("\n2. Matriz de combustible")
    df_combustible = generar_matriz_combustible(dir_datos=dir_datos)

    print("\n3. Matriz de costos totales")
    df_costos = generar_matriz_costos(
        valor_hora_vendedor=valor_hora_vendedor,
        dir_datos=dir_datos,
        nombre_archivo_peajes=nombre_archivo_peajes,
    )

    print("\n" + "=" * 60)

    return df_peajes, df_combustible, df_costos


if __name__ == "__main__":
    generar_todas_las_matrices()
