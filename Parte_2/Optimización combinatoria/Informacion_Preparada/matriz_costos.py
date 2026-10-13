from pathlib import Path
import numpy as np
import pandas as pd

# Valor de la hora laboral del vendedor en euros (€/h) respaldado por el INE (ETCL)
VALOR_HORA_VENDEDOR_DEFAULT = 25.00 # Segun la ETCL hecha or el INE, se situa en torno a 24.50 € - 26.18 €/hora.


def generar_matriz_costos(
    valor_hora_vendedor: float = VALOR_HORA_VENDEDOR_DEFAULT,
    nombre_archivo_salida: str = "costos_totales_espana_eur.csv",
) -> pd.DataFrame:
    """
    Genera y exporta la matriz de costos combinados (combustible, peajes y tiempo del vendedor)
    para las 47 capitales de España peninsular.
    Se usa el Modelo de Costo Generalizado de Transporte:
    C_total(i, j) = C_combustible(i, j) + C_peaje(i, j) + (Tiempo(i, j) * Valor_Hora)
    
    Parámetros:
        valor_hora_vendedor (float): Costo monetario por hora efectiva del vendedor en euros (€/h).
        nombre_archivo_salida (str): Nombre del archivo CSV a generar.

    Retorna:
        pd.DataFrame: Matriz de costos totales simétrica con dimensiones 47x47 en euros.
    """
    directorio_base = Path(__file__).resolve().parent

    ruta_combustible = directorio_base / "combustible_espana_eur.csv"
    ruta_tiempos = directorio_base / "tiempos_espana_h.csv"
    ruta_peajes = directorio_base / "peajes_espana_eur.csv"
    ruta_salida = directorio_base / nombre_archivo_salida

    # Cargar matrices de costos y tiempo
    df_combustible = pd.read_csv(ruta_combustible, index_col=0)
    df_tiempos = pd.read_csv(ruta_tiempos, index_col=0)
    df_peajes = pd.read_csv(ruta_peajes, index_col=0)


    # Calculo matriz de costos totales
    # C_total = Combustible + Peajes + (Tiempo_horas * Valor_hora_vendedor)
    df_costo_tiempo = df_tiempos * valor_hora_vendedor
    df_costos_totales = df_combustible + df_peajes + df_costo_tiempo

    # Asegurar que la diagonal principal (i -> i) sea 0.0
    np.fill_diagonal(df_costos_totales.values, 0.0)

    # Redondear a 2 decimales (centavos de euro)
    df_costos_totales = df_costos_totales.round(2)

    # Exporta como CSV
    df_costos_totales.to_csv(ruta_salida)

    print(f"Matriz de costos generada en: '{ruta_salida}'")
    print(f"     Dimensiones: {df_costos_totales.shape}")
    print(f"     Valor hora del vendedor aplicado: {valor_hora_vendedor:.2f} EUR/h")
    print("\nPreview:")
    print(df_costos_totales.iloc[:3, :3])

    return df_costos_totales


if __name__ == "__main__":
    generar_matriz_costos()
