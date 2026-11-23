

import json
import time
from pathlib import Path
import pandas as pd
import requests


def extraer_costo_peajes_here(datos: dict) -> float:
    """Extrae el costo total de peajes en EUR desde la estructura correcta de HERE Routing API v8."""
    rutas = datos.get("routes", [])
    if not rutas:
        return 0.0

    total_eur = 0.0
    for seccion in rutas[0].get("sections", []):
        # 1. Intentar leer el total consolidado de la sección
        total_obj = seccion.get("travelSummary", {}).get("tolls", {}).get("total")
        if total_obj and "value" in total_obj:
            total_eur += float(total_obj["value"])
            continue

        # 2. Fallback: sumar tarifa por tarifa en section['tolls']
        for peaje in seccion.get("tolls", []):
            fares = peaje.get("fares", [])
            if fares:
                # Tomar el menor precio disponible (p. ej. turismo estándar / telepeaje o efectivo)
                precios = [
                    float(f["price"]["value"])
                    for f in fares
                    if "price" in f and "value" in f["price"]
                ]
                if precios:
                    total_eur += min(precios)

    return round(total_eur, 2)


def consultar_peaje_here(lat1, lon1, lat2, lon2, api_key) -> float:
    url = "https://router.hereapi.com/v8/routes"
    params = {
        "apiKey": api_key,
        "transportMode": "car",
        "origin": f"{lat1},{lon1}",
        "destination": f"{lat2},{lon2}",
        "return": "tolls,travelSummary",
        "currency": "EUR",
    }

    for intento in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return extraer_costo_peajes_here(resp.json())
            elif resp.status_code == 429:
                print("⚠️ Límite de velocidad (429). Esperando 15s...")
                time.sleep(15)
            else:
                print(f"⚠️ HTTP {resp.status_code}: {resp.text[:120]}")
                return 0.0
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error de red (intento {intento+1}/3): {e}")
            time.sleep(5)
    return 0.0


def construir_matriz_peajes_here():
    API_KEY = ""

    base_dir = Path(__file__).resolve().parent
    ruta_ciudades = base_dir / "data" / "ciudades_espana.csv"
    ruta_salida = base_dir / "data" / "peajes_espana_eur.csv"
    ruta_progreso = base_dir / "data" / "progreso_peajes_here.json"

    df_ciudades = pd.read_csv(ruta_ciudades)
    ciudades = df_ciudades.to_dict("records")
    nombres = [c["ciudad"] for c in ciudades]
    n = len(ciudades)

    # Cargar progreso previo si existe
    if ruta_salida.exists() and ruta_progreso.exists():
        df_peajes = pd.read_csv(ruta_salida, index_col=0)
        with open(ruta_progreso, "r", encoding="utf-8") as f:
            completados = set(json.load(f))
        print(f"🔄 Reanudando desde {len(completados)}/1081 trayectos ya guardados...")
    else:
        df_peajes = pd.DataFrame(0.0, index=nombres, columns=nombres)
        completados = set()

    total_pares = n * (n - 1) // 2
    consultas_sesion = 0

    for i in range(n):
        for j in range(i + 1, n):
            clave = f"{i}-{j}"
            if clave in completados:
                continue

            c1, c2 = ciudades[i], ciudades[j]
            costo = consultar_peaje_here(c1["lat"], c1["lon"], c2["lat"], c2["lon"], API_KEY)

            df_peajes.iloc[i, j] = costo
            df_peajes.iloc[j, i] = costo
            completados.add(clave)
            consultas_sesion += 1

            if costo > 0:
                print(f"💰 [{len(completados)}/{total_pares}] {c1['ciudad']} <-> {c2['ciudad']}: {costo:.2f} €")

            time.sleep(0.35)

            if consultas_sesion % 25 == 0:
                df_peajes.to_csv(ruta_salida)
                with open(ruta_progreso, "w", encoding="utf-8") as f:
                    json.dump(list(completados), f)
                print(f"💾 Progreso guardado: {len(completados)}/{total_pares}")

    df_peajes.to_csv(ruta_salida)
    with open(ruta_progreso, "w", encoding="utf-8") as f:
        json.dump(list(completados), f)
    print(f"✅ Matriz real de peajes completada y guardada en: {ruta_salida}")


if __name__ == "__main__":
    construir_matriz_peajes_here()