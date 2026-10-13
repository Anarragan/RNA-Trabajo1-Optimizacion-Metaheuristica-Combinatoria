"""
Cálculo de la Matriz de Costos de Combustible para el TSP de España Peninsular (47 ciudades).

Lee la matriz de distancias por carretera generada mediante OSRM y calcula
el costo monetario directo en combustible (€) entre cada par de capitales,
asumiendo el consumo medio de un turismo y el precio de mercado de la gasolina.
"""

import pandas as pd
from pathlib import Path


def generar_matriz_combustible():
    """
    Lee la matriz de distancias y genera una nueva matriz bidimensional (N x N) con el 
    costo monetario del combustible en euros (€) para cada trayecto entre capitales.

    Aplica un cálculo vectorizado basado en los parámetros de consumo del vehículo y
    precio de mercado del combustible:
        Costo (€) = (Distancia [km] / 100) * Consumo [L/100km] * Precio [€/L]

    parametros:
        Ninguno. Lee 'distancias_espana_km.csv' desde el directorio 'data/'.

    retorna:
        None. Exporta la matriz resultante redondeada a 2 decimales a 'data/combustible_espana_eur.csv'.
    """
    ruta_distancias = Path('data') / 'distancias_espana_km.csv'
    ruta_salida = Path('data') / 'combustible_espana_eur.csv'

    # Parámetros del vehículo y mercado (¡Documentar en reporte APA!)
    consumo_l_100km = 5.5    # Litros cada 100 km (Seat León 1.5 TSI)
    precio_euro_litro = 1.55 # Precio por litro (Gasolina 95)

    if not ruta_distancias.exists():
        print(f"Error: No se encontró '{ruta_distancias}'.")
        return

    # 1. Leer la matriz de distancias que obtuvimos de OSRM
    # index_col=0 asegura que los nombres de las ciudades sean el índice (las filas)
    df_distancias = pd.read_csv(ruta_distancias, index_col=0)

    # 2. Vectorización pura: Cálculo matemático simultáneo para las 2.209 combinaciones
    # Fórmula: (Distancia / 100) * Consumo * Precio
    df_combustible = (df_distancias / 100) * consumo_l_100km * precio_euro_litro

    # Redondeamos a 2 decimales para que represente centavos de Euro
    df_combustible = df_combustible.round(2)

    # 3. Guardar en la carpeta data/
    df_combustible.to_csv(ruta_salida)
    
    print(f"✅ Matriz de combustible generada exitosamente en: '{ruta_salida}'")
    print(f"   Dimensiones: {df_combustible.shape}")
    print("\n🔍 Muestra (Primeras 3x3 ciudades en Euros):")
    print(df_combustible.iloc[:3, :3])

if __name__ == "__main__":
    generar_matriz_combustible()