import json
import time
from pathlib import Path
import pandas as pd
import requests

# 1. Definición de las 47 capitales peninsulares con coordenadas (Latitud, Longitud)
# Fuente de coordenadas: Nomenclátor Geográfico Básico de España (IGN)
CAPITALES = [
    {"ciudad": "A Coruña", "lat": 43.3713, "lon": -8.3960},
    {"ciudad": "Albacete", "lat": 38.9943, "lon": -1.8585},
    {"ciudad": "Alicante", "lat": 38.3452, "lon": -0.4810},
    {"ciudad": "Almería", "lat": 36.8381, "lon": -2.4597},
    {"ciudad": "Ávila", "lat": 40.6565, "lon": -4.6818},
    {"ciudad": "Badajoz", "lat": 38.8794, "lon": -6.9706},
    {"ciudad": "Barcelona", "lat": 41.3879, "lon": 2.1699},
    {"ciudad": "Bilbao", "lat": 43.2630, "lon": -2.9350},
    {"ciudad": "Burgos", "lat": 42.3439, "lon": -3.6969},
    {"ciudad": "Cáceres", "lat": 39.4753, "lon": -6.3723},
    {"ciudad": "Cádiz", "lat": 36.5298, "lon": -6.2926},
    {"ciudad": "Castellón de la Plana", "lat": 39.9864, "lon": -0.0513},
    {"ciudad": "Ciudad Real", "lat": 38.9861, "lon": -3.9274},
    {"ciudad": "Córdoba", "lat": 37.8882, "lon": -4.7794},
    {"ciudad": "Cuenca", "lat": 40.0704, "lon": -2.1374},
    {"ciudad": "Girona", "lat": 41.9794, "lon": 2.8214},
    {"ciudad": "Granada", "lat": 37.1773, "lon": -3.5986},
    {"ciudad": "Guadalajara", "lat": 40.6337, "lon": -3.1674},
    {"ciudad": "Huelva", "lat": 37.2614, "lon": -6.9447},
    {"ciudad": "Huesca", "lat": 42.1362, "lon": -0.4087},
    {"ciudad": "Jaén", "lat": 37.7796, "lon": -3.7849},
    {"ciudad": "León", "lat": 42.5987, "lon": -5.5671},
    {"ciudad": "Lleida", "lat": 41.6176, "lon": 0.6200},
    {"ciudad": "Logroño", "lat": 42.4658, "lon": -2.4499},
    {"ciudad": "Lugo", "lat": 43.0097, "lon": -7.5568},
    {"ciudad": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"ciudad": "Málaga", "lat": 36.7213, "lon": -4.4214},
    {"ciudad": "Murcia", "lat": 37.9922, "lon": -1.1307},
    {"ciudad": "Ourense", "lat": 42.3358, "lon": -7.8639},
    {"ciudad": "Oviedo", "lat": 43.3619, "lon": -5.8494},
    {"ciudad": "Palencia", "lat": 42.0095, "lon": -4.5288},
    {"ciudad": "Pamplona", "lat": 42.8125, "lon": -1.6458},
    {"ciudad": "Pontevedra", "lat": 42.4310, "lon": -8.6444},
    {"ciudad": "Salamanca", "lat": 40.9701, "lon": -5.6635},
    {"ciudad": "San Sebastián", "lat": 43.3183, "lon": -1.9812},
    {"ciudad": "Santander", "lat": 43.4623, "lon": -3.8099},
    {"ciudad": "Segovia", "lat": 40.9429, "lon": -4.1088},
    {"ciudad": "Sevilla", "lat": 37.3891, "lon": -5.9845},
    {"ciudad": "Soria", "lat": 41.7666, "lon": -2.4779},
    {"ciudad": "Tarragona", "lat": 41.1189, "lon": 1.2445},
    {"ciudad": "Teruel", "lat": 40.3456, "lon": -1.1072},
    {"ciudad": "Toledo", "lat": 39.8628, "lon": -4.0273},
    {"ciudad": "Valencia", "lat": 39.4699, "lon": -0.3763},
    {"ciudad": "Valladolid", "lat": 41.6523, "lon": -4.7245},
    {"ciudad": "Vitoria-Gasteiz", "lat": 42.8469, "lon": -2.6716},
    {"ciudad": "Zamora", "lat": 41.5033, "lon": -5.7446},
    {"ciudad": "Zaragoza", "lat": 41.6488, "lon": -0.8891},
]


def construir_url_osrm(coordenadas):
    """OSRM exige el formato: {longitud},{latitud} separados por punto y coma."""
    coords_str = ";".join([f"{c['lon']},{c['lat']}" for c in coordenadas])
    base_url = "http://router.project-osrm.org/table/v1/driving/"
    # Pedimos explicitamente anotaciones de distancia y duracion
    url = f"{base_url}{coords_str}?annotations=distance,duration"
    return url


def obtener_matrices():
    print(f"Obteniendo datos de ruta para {len(CAPITALES)} ciudades...")

    url = construir_url_osrm(CAPITALES)
    headers = {"User-Agent": "OptimizacionTSP-Espana/1.0"}

    respuesta = requests.get(url, headers=headers, timeout=30)

    if respuesta.status_code != 200:
        raise RuntimeError(
            f"Error al consultar OSRM API (Status: {respuesta.status_code}): {respuesta.text}"
        )

    datos = respuesta.json()

    if datos.get("code") != "Ok":
        raise RuntimeError(f"Error devuelto por el motor OSRM: {datos.get('message')}")

    # OSRM entrega:
    # - 'durations': matriz en segundos -> convertimos a horas
    # - 'distances': matriz en metros -> convertimos a kilómetros
    duraciones_segundos = datos["durations"]
    distancias_metros = datos["distances"]

    nombres = [c["ciudad"] for c in CAPITALES]

    df_tiempos_h = pd.DataFrame(duraciones_segundos, index=nombres, columns=nombres) / 3600.0
    df_distancias_km = pd.DataFrame(distancias_metros, index=nombres, columns=nombres) / 1000.0
    df_ciudades = pd.DataFrame(CAPITALES)

    # Crear carpeta de datos si no existe
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Exportar datos limpios
    df_ciudades.to_csv(output_dir / "ciudades_espana.csv", index=False)
    df_distancias_km.to_csv(output_dir / "distancias_espana_km.csv")
    df_tiempos_h.to_csv(output_dir / "tiempos_espana_h.csv")

    print("Matrices generadas exitosamente en la carpeta 'data/':")
    print(f"- data/ciudades_espana.csv ({len(df_ciudades)} ciudades)")
    print(f"- data/distancias_espana_km.csv (Matriz {df_distancias_km.shape})")
    print(f"- data/tiempos_espana_h.csv (Matriz {df_tiempos_h.shape})")


if __name__ == "__main__":
    obtener_matrices()
    