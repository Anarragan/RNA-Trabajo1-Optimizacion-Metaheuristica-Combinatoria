"""
Módulo de Visualización Avanzado para el Problema del Viajante (TSP) en España.
Genera GIFs con gráfica de convergencia (mapa a la izq, gráfica a la der)
y una imagen comparativa final estilo mosaico para ACO y GA.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# Estilo global para las gráficas
plt.style.use('dark_background')
sns.set_palette("husl")

def generar_gif_avanzado(
    experimento_key: str,
    algoritmo: str,
    datos: dict,
    lons: list,
    lats: list,
    nombres: list,
    dir_gifs: Path,
    fps: int = 10
):
    """Genera un GIF con el mapa a la izquierda y la gráfica de costo a la derecha."""
    
    try:
        data_algoritmo = datos["experimentos"][experimento_key][algoritmo]
        historial_rutas = data_algoritmo["historial_rutas"]
        historial_costos = data_algoritmo["historial_costos"]
        valor_hora = datos["experimentos"][experimento_key]["valor_hora_vendedor"]
    except KeyError:
        print(f"  -> No se encontraron datos para {algoritmo} en {experimento_key}. Saltando.")
        return

    num_frames = len(historial_rutas)
    if num_frames == 0:
        return

    # Configurar la figura (Ancho x Alto) y dividirla en 2 espacios
    fig = plt.figure(figsize=(16, 7), facecolor='#111827')
    fig.suptitle(f"{algoritmo} - Evolución (Valor Hora: {valor_hora} €/h)", 
                 fontsize=18, color='white', weight='bold')
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
    ax_mapa = plt.subplot(gs[0])
    ax_costo = plt.subplot(gs[1])
    
    # Pre-configurar gráfica de costo (Derecha)
    ax_costo.set_title("Evolución del Costo Mínimo", color='white', fontsize=14)
    ax_costo.set_xlabel("Iteración", color='white', fontsize=12)
    ax_costo.set_ylabel("Costo Total (€)", color='white', fontsize=12)
    ax_costo.grid(True, linestyle=':', alpha=0.3)
    ax_costo.set_xlim(0, num_frames)
    
    min_cost = min(historial_costos)
    max_cost = max(historial_costos)
    ax_costo.set_ylim(min_cost * 0.95, max_cost * 1.05)
    
    # Objetos vacíos que se llenarán frame a frame
    linea_costo, = ax_costo.plot([], [], color='#00d2ff', linewidth=2.5)
    punto_costo, = ax_costo.plot([], [], 'o', color='#ff0055', markersize=8)
    
    # Trazo fantasma de todo el historial de costos para contexto
    ax_costo.plot(range(num_frames), historial_costos, color='gray', linestyle=':', alpha=0.4)

    # Índice de las ciudades a etiquetar (para evitar sobreposición, etiquetamos menos y las desplazamos un poco)
    # Ajusta estos índices según las ciudades que quieras destacar en tu CSV
    ciudades_a_etiquetar = [0, 5, 12, 25, 38] 

    def actualizar_frame(frame):
        # 1. Actualizar Mapa (Izquierda)
        ax_mapa.clear()
        ax_mapa.set_facecolor('#1e293b')
        ax_mapa.set_xticks([])
        ax_mapa.set_yticks([])
        ax_mapa.set_xlim(-9.5, 3.5)
        ax_mapa.set_ylim(35.5, 44.5)
        
        # Etiqueta flotante con el estado actual
        ax_mapa.text(-9.0, 43.5, 
                     f"Iteración: {frame+1}/{num_frames}\nCosto: {historial_costos[frame]:,.2f} €", 
                     color='white', fontsize=12, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))

        ruta_indices = historial_rutas[frame]
        ruta_lons = [lons[i] for i in ruta_indices] + [lons[ruta_indices[0]]]
        ruta_lats = [lats[i] for i in ruta_indices] + [lats[ruta_indices[0]]]

        # Dibujar líneas de ruta
        ax_mapa.plot(ruta_lons, ruta_lats, color='orange', linewidth=2, alpha=0.9, zorder=1)
        # Dibujar puntos (ciudades)
        ax_mapa.scatter(lons, lats, color='#00d2ff', s=35, edgecolor='black', zorder=2)
        
        # Etiquetas de las ciudades seleccionadas, con un pequeño desplazamiento
        for idx in ciudades_a_etiquetar:
            if idx < len(lons): # Verificación por seguridad
                 # Se suma un pequeño offset a x e y para que el texto no tape el punto
                 ax_mapa.text(lons[idx] + 0.15, lats[idx] + 0.15, nombres[idx], 
                              fontsize=7, color='white', weight='bold', zorder=3)

        # 2. Actualizar Gráfica de Costo (Derecha)
        x_data = list(range(frame + 1))
        y_data = historial_costos[:frame + 1]
        
        linea_costo.set_data(x_data, y_data)
        punto_costo.set_data([frame], [historial_costos[frame]])

    print(f"Generando GIF animado para {experimento_key} ({algoritmo})...")
    anim = FuncAnimation(fig, actualizar_frame, frames=num_frames, interval=1000/fps, repeat=False)
    
    ruta_salida = dir_gifs / f"{algoritmo.lower()}_espana47_{experimento_key}.gif"
    anim.save(ruta_salida, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"  -> Guardado: {ruta_salida.name}")


def generar_imagen_comparativa_final(
    datos: dict, 
    algoritmo: str,
    lons: list, 
    lats: list,
    dir_salida: Path
):
    """Genera una imagen estática con las 6 mejores rutas, una para cada valor de hora."""
    
    experimentos = []
    for key, val in datos["experimentos"].items():
        if algoritmo in val:
             experimentos.append((val["valor_hora_vendedor"], key, val[algoritmo]))
    
    experimentos.sort(key=lambda x: x[0]) 
    
    if not experimentos:
        print(f"No hay experimentos para armar el grid final de {algoritmo}.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='#111827')
    fig.suptitle(f"Comparativa Final - Mejor Ruta TSP - {algoritmo}", 
                 fontsize=22, color='white', weight='bold')
    
    axes = axes.flatten()
    colores_rutas = ['#00d2ff', '#20c997', '#fd7e14', '#dc3545', '#6f42c1', '#fd7e14']

    for i, (valor_hora, key, datos_alg) in enumerate(experimentos):
        if i >= 6: break 
        
        ax = axes[i]
        ax.set_facecolor('#1e293b')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-9.5, 3.5)
        ax.set_ylim(35.5, 44.5)
        
        mejor_ruta = datos_alg["mejor_ruta_indices"]
        desglose = datos_alg["desglose"]
        
        titulo_stats = (f"Hora: {valor_hora} €/h | Costo: {datos_alg['mejor_costo_total']:,.2f} €\n"
                        f"Distancia: {desglose['distancia_total_km']:,.1f} km | Tiempo: {desglose['tiempo_total_horas']:.1f} h\n"
                        f"(Comb: {desglose['costo_combustible_eur']:,.0f} € | Peajes: {desglose['costo_peajes_eur']:,.0f} €)")
        
        ax.set_title(titulo_stats, color='white', fontsize=11, weight='bold', pad=10)

        ruta_lons = [lons[idx] for idx in mejor_ruta] + [lons[mejor_ruta[0]]]
        ruta_lats = [lats[idx] for idx in mejor_ruta] + [lats[mejor_ruta[0]]]
        
        ax.plot(ruta_lons, ruta_lats, color=colores_rutas[i % len(colores_rutas)], linewidth=2.5, alpha=0.9)
        ax.scatter(lons, lats, color='gray', s=20, zorder=2)
        
        ax.plot(lons[mejor_ruta[0]], lats[mejor_ruta[0]], marker='*', color='red', markersize=15, zorder=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    ruta_img = dir_salida / f"comparativa_final_{algoritmo.lower()}.png"
    plt.savefig(ruta_img, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Imagen comparativa final guardada: {ruta_img.name}")


def ejecutar_orquestador_visual():
    """Ejecuta la generación visual para todos los algoritmos disponibles."""
    directorio_base = Path(__file__).resolve().parent
    
    dir_datos = directorio_base / "Parte_2" / "Informacion_Preparada" / "datos_graficas"
    ruta_ciudades = directorio_base / "Parte_2" / "Informacion_Preparada" / "ciudades_espana.csv"
    
    dir_gifs = directorio_base / "Parte_2" / "gifs2"
    dir_gifs.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("INICIANDO RENDERIZADO VISUAL PARA ACO Y GA")
    print("="*60)

    try:
        # Asegúrate de usar los nombres de columnas correctos (ciudad, lat, lon) según tu CSV
        df_ciudades = pd.read_csv(ruta_ciudades)
        nombres = df_ciudades['ciudad'].tolist()
        lons = df_ciudades['lon'].tolist() 
        lats = df_ciudades['lat'].tolist()
    except Exception as e:
        print(f"Error cargando CSV de ciudades: {e}")
        return

    ruta_json = dir_datos / "resultados_comparativos.json"
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos_json = json.load(f)
    except Exception as e:
        print(f"Error cargando el JSON de resultados: {e}")
        return

    # Extraer automáticamente los algoritmos presentes en el JSON (ej. "ACO" y "GA")
    algoritmos_presentes = datos_json["metadata"].get("algoritmos_comparados", [])
    
    # Mapeo simple: si en el JSON dice "Colonia de Hormigas (ACO)", extraer solo "ACO"
    algoritmos_claves = []
    for alg_nombre in algoritmos_presentes:
        if "ACO" in alg_nombre or "ACA" in alg_nombre:
             algoritmos_claves.append("ACO")
        elif "GA" in alg_nombre:
             algoritmos_claves.append("GA")

    # Si no se detectaron automáticamente, forzamos ambos
    if not algoritmos_claves:
         algoritmos_claves = ["ACO", "GA"]

    for algoritmo in algoritmos_claves:
        print(f"\nProcesando gráficos para {algoritmo}...")
        for clave_exp in datos_json["experimentos"].keys():
            generar_gif_avanzado(
                experimento_key=clave_exp,
                algoritmo=algoritmo,
                datos=datos_json,
                lons=lons,
                lats=lats,
                nombres=nombres,
                dir_gifs=dir_gifs,
                fps=10
            )
            
        generar_imagen_comparativa_final(
            datos=datos_json,
            algoritmo=algoritmo,
            lons=lons,
            lats=lats,
            dir_salida=dir_gifs
        )
        
    print("\n¡Proceso Finalizado Exitosamente!")

if __name__ == "__main__":
    # Ahora el orquestador se encarga de ejecutar ambos automáticamente
    ejecutar_orquestador_visual()