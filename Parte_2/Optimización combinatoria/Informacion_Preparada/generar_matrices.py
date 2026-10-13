from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Valor de la hora laboral del vendedor en euros (€/h) respaldado por el INE (ETCL)
VALOR_HORA_VENDEDOR_DEFAULT = 25.00 # Segun la ETCL hecha por el INE, se situa en torno a 24.50 € - 26.18 €/hora.

# Directorio por defecto para almacenar y leer matrices y datos
DIRECTORIO_DATOS_DEFAULT = Path(__file__).resolve().parent / "data"


def inyectar_peajes_peninsula(
    dir_datos: Optional[Path] = None,
    nombre_archivo_salida: str = "peajes_espana_eur.csv",
) -> pd.DataFrame:
    """
    Construye la matriz de costos de peaje entre las capitales peninsulares.

    Procedimiento:
    1. Lee los nombres de las 47 capitales desde 'data/ciudades_espana.csv'.
    2. Crea una matriz cuadrada inicializada en ceros (0.0 €) con las ciudades como índices y columnas.
    3. Itera sobre el catálogo de rutas con peaje activo e inyecta simétricamente los costos.
    4. Exporta la matriz resultante a la carpeta 'data/'.

    Parámetros:
        dir_datos (Path, opcional): Ruta al directorio de datos. Por defecto 'Informacion_Preparada/data'.
        nombre_archivo_salida (str): Nombre del archivo CSV de salida. Por defecto 'peajes_espana_eur.csv'.

    Retorna:
        pd.DataFrame: Matriz simétrica (47x47) con los costos de peaje en euros.
    """
    if dir_datos is None:
        dir_datos = DIRECTORIO_DATOS_DEFAULT
    dir_datos.mkdir(parents=True, exist_ok=True)

    ruta_ciudades = dir_datos / "ciudades_espana.csv"
    ruta_salida = dir_datos / nombre_archivo_salida

    # Leer la lista oficial de ciudades
    df_ciudades = pd.read_csv(ruta_ciudades)
    nombres_ciudades = df_ciudades['ciudad'].tolist()
    
    # Crear una matriz de ceros limpia
    df_peajes = pd.DataFrame(0.0, index=nombres_ciudades, columns=nombres_ciudades)

    # Base de datos exhaustiva de los peajes vigentes en España Peninsular (Euros)
    # Valores de referencia para turismos ligeros (Actualizados post-liberación)
    rutas_con_peaje = [
        # Radial Noroeste / AP-6 / AP-51 / AP-61
        ("Madrid", "Segovia", 7.45),
        ("Madrid", "Ávila", 10.50),
        
        # AP-66 y AP-71 (Asturias y León)
        ("León", "Oviedo", 15.00),
        
        # AP-9 y AP-53 (Galicia)
        ("A Coruña", "Pontevedra", 13.85),
        ("A Coruña", "Vigo", 18.15), # Asumiendo paso por Pontevedra
        ("Pontevedra", "Ourense", 6.40),
        
        # AP-68 (Eje del Ebro)
        ("Bilbao", "Logroño", 19.50),
        ("Logroño", "Zaragoza", 17.80),
        ("Bilbao", "Zaragoza", 37.25),
        
        # AP-8 y AP-1 (País Vasco)
        ("Bilbao", "San Sebastián", 13.10),
        ("Vitoria-Gasteiz", "San Sebastián", 12.50),
        ("Vitoria-Gasteiz", "Bilbao", 6.25),
        
        # C-32 / C-16 (Cataluña)
        ("Barcelona", "Girona", 10.20),
        
        # AP-46 (Andalucía)
        ("Málaga", "Córdoba", 5.60) # Tramo Las Pedrizas
    ]

    print("Inyectando tarifas de peaje en la matriz local...")
    contador = 0
    
    for origen, destino, costo in rutas_con_peaje:
        if origen in df_peajes.columns and destino in df_peajes.columns:
            # Asignar costo simétricamente (ida y vuelta)
            df_peajes.loc[origen, destino] = costo
            df_peajes.loc[destino, origen] = costo
            contador += 1
        else:
            print(f"Advertencia: No se encontró '{origen}' o '{destino}' en la matriz.")

    # Guardar el CSV definitivo
    df_peajes.to_csv(ruta_salida)
    
    print(f"Se inyectaron {contador} rutas de peaje exitosamente.")
    print("El resto de las 2,162 combinaciones mantienen costo 0.00 € por la red gratuita.")
    print(f"Archivo guardado en: {ruta_salida}")
    return df_peajes


def generar_matriz_combustible(
    dir_datos: Optional[Path] = None,
    nombre_archivo_salida: str = "combustible_espana_eur.csv",
) -> Optional[pd.DataFrame]:
    """
    Lee la matriz de distancias y genera una nueva matriz (N x N) con el 
    costo monetario del combustible en euros para cada trayecto entre capitales.

    Aplica la siguiente formula:
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
    consumo_l_100km = 5.5    # Litros cada 100 km (Seat León 1.5 TSI)
    precio_euro_litro = 1.55 # Precio por litro (Gasolina 95)

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
    print(f"Dimensiones: {df_combustible.shape}")
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
    np.fill_diagonal(df_costos_totales.values, 0.0)

    return df_costos_totales.round(2)


def generar_matriz_costos(
    valor_hora_vendedor: float = VALOR_HORA_VENDEDOR_DEFAULT,
    nombre_archivo_salida: str = "costos_totales_espana_eur.csv",
    dir_datos: Optional[Path] = None,
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

    Retorna:
        pd.DataFrame: Matriz de costos totales simétrica con dimensiones 47x47 en euros.
    """
    if dir_datos is None:
        dir_datos = DIRECTORIO_DATOS_DEFAULT
    dir_datos.mkdir(parents=True, exist_ok=True)

    ruta_combustible = dir_datos / "combustible_espana_eur.csv"
    ruta_tiempos = dir_datos / "tiempos_espana_h.csv"
    ruta_peajes = dir_datos / "peajes_espana_eur.csv"
    ruta_salida = dir_datos / nombre_archivo_salida

    # Cargar matrices de costos y tiempo
    df_combustible = pd.read_csv(ruta_combustible, index_col=0)
    df_tiempos = pd.read_csv(ruta_tiempos, index_col=0)
    df_peajes = pd.read_csv(ruta_peajes, index_col=0)

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
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Pipeline de generación de matrices en orden:
    1. Inyección de tarifas de peajes peninsulares ('peajes_espana_eur.csv').
    2. Estimación del consumo y costo de combustible ('combustible_espana_eur.csv').
    3. Consolidación de la matriz de costos totales ('costos_totales_espana_eur.csv').

    Parámetros:
        dir_datos (Path, opcional): Ruta al directorio de datos donde se guardan las matrices.
        valor_hora_vendedor (float): Valor de la hora laboral del vendedor en euros.

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

    print("\n Matriz de peajes")
    df_peajes = inyectar_peajes_peninsula(dir_datos=dir_datos)

    print("\n Matriz de combustible")
    df_combustible = generar_matriz_combustible(dir_datos=dir_datos)

    print("\n Matriz de costos totales")
    df_costos = generar_matriz_costos(
        valor_hora_vendedor=valor_hora_vendedor,
        dir_datos=dir_datos,
    )

    print("\n" + "=" * 60)


    return df_peajes, df_combustible, df_costos


if __name__ == "__main__":
    generar_todas_las_matrices()
