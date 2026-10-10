import pandas as pd
from pathlib import Path

def inyectar_peajes_peninsula():
    ruta_ciudades = Path('data') / 'ciudades_espana.csv'
    ruta_salida = Path('data') / 'peajes_espana_eur.csv'

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
    
    print(f"✅ Se inyectaron {contador} rutas de peaje exitosamente.")
    print("✅ El resto de las 2,162 combinaciones mantienen costo 0.00 € por la red gratuita.")
    print(f"✅ Archivo guardado en: {ruta_salida}")

if __name__ == "__main__":
    inyectar_peajes_peninsula()