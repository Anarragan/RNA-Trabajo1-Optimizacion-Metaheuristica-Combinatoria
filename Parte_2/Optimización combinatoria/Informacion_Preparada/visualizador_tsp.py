import json
from pathlib import Path
from typing import Optional
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns

# Estilo global para las gráficas
plt.style.use('dark_background')
sns.set_palette("husl")


def cargar_capa_carreteras(ruta_zip: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Carga el catálogo de la Red de Carreteras del Estado (RCE) desde el archivo .zip
    utilizando geopandas y lo reproyecta a coordenadas geográficas WGS84 (EPSG:4326).

    Parámetros:
        ruta_zip (Path): Ruta al archivo comprimido con el shapefile de carreteras.

    Retorna:
        Optional[gpd.GeoDataFrame]: Capa geográfica de carreteras o None si no se encuentra.
    """
    if not ruta_zip.exists():
        print(f"Advertencia: No se encontró '{ruta_zip.name}'. Se continuará sin capa de carreteras.")
        return None
    try:
        print(f"Cargando capa de Red de Carreteras del Estado '{ruta_zip.name}'...")
        gdf = gpd.read_file(f"zip://{ruta_zip}")
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        print(f" Capa cargada exitosamente ({len(gdf)} tramos viales).")
        return gdf
    except Exception as e:
        print(f"Advertencia: Error al cargar '{ruta_zip.name}': {e}. Se continuará sin carreteras.")
        return None


def generar_gif_avanzado(
    experimento_key: str,
    algoritmo: str,
    datos: dict,
    lons: list,
    lats: list,
    nombres: list,
    dir_gifs: Path,
    gdf_carreteras: Optional[gpd.GeoDataFrame] = None,
    fps: int = 10
):
    """
    Genera una animación GIF sincronizada de dos paneles para un experimento y algoritmo dados.

    Estructura de la visualización:
    - Panel izquierdo: Mapa geográfico de España peninsular que traza paso a paso la evolución
      del ciclo cerrado sobre las 47 ciudades, indicando el progreso de iteraciones y datos actuales.
    - Panel derecho: Curva de convergencia que muestra el descenso del costo mínimo a lo largo
      de las iteraciones frente al historial completo.

    parametros:
        experimento_key (str): Clave del experimento a procesar (ej. 'hora_0', 'hora_15').
        algoritmo (str): Nombre o identificador del algoritmo ('ACO' o 'GA').
        datos (dict): Diccionario cargado desde 'resultados_comparativos.json'.
        lons (list): Coordenadas de longitud geográfica de las 47 capitales.
        lats (list): Coordenadas de latitud geográfica de las 47 capitales.
        nombres (list): Nombres de las 47 capitales peninsulares.
        dir_gifs (Path): Directorio de salida donde se guardará el GIF animado.
        fps (int, opcional): Cuadros por segundo (frames per second) de la animación. Por defecto es 10.

    retorna:
        None. Guarda el archivo '{algoritmo}_espana47_{experimento_key}.gif' en el directorio especificado.
    """
    
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
    ax_costo.set_ylabel("Costo Total (EUR)", color='white', fontsize=12)
    ax_costo.grid(True, linestyle=':', alpha=0.3)
    ax_costo.set_xlim(0, num_frames)
    
    min_cost = min(historial_costos)
    max_cost = max(historial_costos)
    ax_costo.set_ylim(min_cost * 0.95, max_cost * 1.05)
    
    # Objetos vacíos que se llenarán frame a frame
    linea_costo, = ax_costo.plot([], [], color='#00d2ff', linewidth=2.5)
    punto_costo, = ax_costo.plot([], [], 'o', color='#ff0055', markersize=8)
    
    # Trazo opaco de todo el historial de costos
    ax_costo.plot(range(num_frames), historial_costos, color='gray', linestyle=':', alpha=0.4)

    # Configurar Mapa Base (Izquierda)
    ax_mapa.set_facecolor('#0f172a')
    ax_mapa.set_xticks([])
    ax_mapa.set_yticks([])
    ax_mapa.set_xlim(-9.5, 3.5)
    ax_mapa.set_ylim(35.5, 44.5)

    # Dibujar la Red de Carreteras del Estado (RCE) como fondo si está disponible
    if gdf_carreteras is not None:
        gdf_carreteras.plot(ax=ax_mapa, color='#334155', linewidth=0.45, alpha=0.28, zorder=1)

    # Dibujar puntos de ciudades y etiquetas fijas
    ax_mapa.scatter(lons, lats, color='#00d2ff', s=35, edgecolor='black', zorder=3)
    ciudades_a_etiquetar = [0, 5, 12, 25, 38]
    for idx in ciudades_a_etiquetar:
        if idx < len(lons):
            ax_mapa.text(lons[idx] + 0.15, lats[idx] + 0.15, nombres[idx], # +0.15 para alejar las tags
                         fontsize=7, color='white', weight='bold', zorder=4)

    # Elementos dinámicos de la ruta que cambian en cada frame
    linea_ruta, = ax_mapa.plot([], [], color='orange', linewidth=2, alpha=0.75, zorder=2)
    punto_inicio, = ax_mapa.plot([], [], marker='*', color='#ef4444', markersize=12, zorder=5)
    texto_iter = ax_mapa.text(-9.0, 43.5, "",
                              color='white', fontsize=12, weight='bold',
                              bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'),
                              zorder=6)

    def actualizar_frame(frame):
        # 1. Actualizar Ruta en Mapa (Izquierda)
        ruta_indices = historial_rutas[frame]
        ruta_lons = [lons[i] for i in ruta_indices] + [lons[ruta_indices[0]]]
        ruta_lats = [lats[i] for i in ruta_indices] + [lats[ruta_indices[0]]]

        linea_ruta.set_data(ruta_lons, ruta_lats)
        punto_inicio.set_data([lons[ruta_indices[0]]], [lats[ruta_indices[0]]])
        texto_iter.set_text(
            f"Iteración: {frame+1}/{num_frames}\nCosto: {historial_costos[frame]:,.2f} €"
        )

        # 2. Actualizar Gráfica de Costo (Derecha)
        x_data = list(range(frame + 1))
        y_data = historial_costos[:frame + 1]

        linea_costo.set_data(x_data, y_data)
        punto_costo.set_data([frame], [historial_costos[frame]])

    print(f"GIF animado para {experimento_key} ({algoritmo})")
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
    dir_salida: Path,
    gdf_carreteras: Optional[gpd.GeoDataFrame] = None,
):
    """
    Genera una imagen estática en cuadrícula (mosaico 2x3) con las mejores rutas finales encontradas
    por el algoritmo para cada valor de tarifa horaria del vendedor.

    Para cada instancia, grafica:
    - La Red de Carreteras del Estado como fondo territorial.
    - La trayectoria óptima final en el mapa peninsular indicando la ciudad origen con un marcador destacado.
    - Las métricas clave en el encabezado: tarifa horaria, costo total (€), distancia (km),
      tiempo de viaje (h) y desglose de combustible y peajes (€).

    Parámetros:
        datos (dict): Diccionario consolidado cargado desde 'resultados_comparativos.json'.
        algoritmo (str): Algoritmo a visualizar ('ACO' o 'GA').
        lons (list): Coordenadas de longitud de las 47 ciudades.
        lats (list): Coordenadas de latitud de las 47 ciudades.
        dir_salida (Path): Directorio donde se guardará la imagen comparativa.
        gdf_carreteras (Optional[gpd.GeoDataFrame]): Capa geográfica RCE para el fondo.

    Retorna:
        None. Exporta la imagen 'comparativa_final_{algoritmo}.png' en alta resolución (DPI 150).
    """
    
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
        ax.set_facecolor('#0f172a')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-9.5, 3.5)
        ax.set_ylim(35.5, 44.5)
        
        # Dibujar carreteras de la Red del Estado como fondo
        if gdf_carreteras is not None:
            gdf_carreteras.plot(ax=ax, color='#334155', linewidth=0.40, alpha=0.20, zorder=1)

        mejor_ruta = datos_alg["mejor_ruta_indices"]
        desglose = datos_alg["desglose"]
        
        titulo_stats = (f"Hora: {valor_hora} €/h | Costo: {datos_alg['mejor_costo_total']:,.2f} €\n"
                        f"Distancia: {desglose['distancia_total_km']:,.1f} km | Tiempo: {desglose['tiempo_total_horas']:.1f} h\n"
                        f"(Comb: {desglose['costo_combustible_eur']:,.0f} € | Peajes: {desglose['costo_peajes_eur']:,.0f} €)")
        
        ax.set_title(titulo_stats, color='white', fontsize=11, weight='bold', pad=10)

        ruta_lons = [lons[idx] for idx in mejor_ruta] + [lons[mejor_ruta[0]]]
        ruta_lats = [lats[idx] for idx in mejor_ruta] + [lats[mejor_ruta[0]]]
        
        ax.plot(ruta_lons, ruta_lats, color=colores_rutas[i % len(colores_rutas)], linewidth=2.2, alpha=0.75, zorder=2)
        ax.scatter(lons, lats, color='#00d2ff', s=25, edgecolor='black', zorder=3)
        ax.plot(lons[mejor_ruta[0]], lats[mejor_ruta[0]], marker='*', color='#ef4444', markersize=14, zorder=4)

    # Ocultar subplots sobrantes en el grid
    for j in range(len(experimentos), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    ruta_img = dir_salida / f"comparativa_final_{algoritmo.lower()}.png"
    plt.savefig(ruta_img, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Imagen comparativa final guardada: {ruta_img.name}")


def ejecutar_orquestador_visual(
    algoritmos_filtrar: Optional[list] = None,
    experimentos_filtrar: Optional[list] = None,
    dir_salida_gifs: Optional[Path] = None,
):
    """
    Orquesta y ejecuta de forma automatizada la generación visual completa para ACO y GA.

    Procedimiento:
    1. Carga las coordenadas de las ciudades desde 'data/ciudades_espana.csv'.
    2. Carga la capa geográfica de carreteras desde 'data/260306-catalogorce2025.zip'.
    3. Lee los resultados unificados de 'resultados_comparativos.json'.
    4. Detecta automáticamente los algoritmos disponibles en los metadatos ('ACO' y 'GA').
    5. Itera sobre cada algoritmo y para cada tarifa de hora del vendedor genera:
       - Animación GIF de convergencia y mapa con la red de carreteras (mediante generar_gif_avanzado).
       - Mosaico comparativo de las rutas finales (mediante generar_imagen_comparativa_final).

    Parámetros:
        algoritmos_filtrar (list, opcional): Lista de algoritmos a graficar (ej. ['ACO']).
        experimentos_filtrar (list, opcional): Lista de claves de experimentos a graficar (ej. ['hora_25']).
        dir_salida_gifs (Path, opcional): Carpeta destino para los GIFs e imágenes.

    Retorna:
        None. Genera los archivos en disco dentro de la carpeta 'GIFS/'.
    """
    directorio_base = Path(__file__).resolve().parent
    
    dir_datos = directorio_base / "datos_graficas"
    ruta_ciudades = directorio_base / "data" / "ciudades_espana.csv"
    ruta_carreteras_zip = directorio_base / "data" / "260306-catalogorce2025.zip"
    
    dir_gifs = dir_salida_gifs if dir_salida_gifs is not None else (directorio_base.parent / "GIFS")
    dir_gifs.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("INICIANDO RENDERIZADO VISUAL PARA ACO Y GA")
    print(f"Directorio de salida: '{dir_gifs}'")
    print("="*60)

    try:
        df_ciudades = pd.read_csv(ruta_ciudades)
        nombres = df_ciudades['ciudad'].tolist()
        lons = df_ciudades['lon'].tolist() 
        lats = df_ciudades['lat'].tolist()
    except Exception as e:
        print(f"Error cargando CSV de ciudades: {e}")
        return

    # Cargar shapefile de la Red de Carreteras del Estado con geopandas
    gdf_carreteras = cargar_capa_carreteras(ruta_carreteras_zip)

    ruta_json = dir_datos / "resultados_comparativos.json"
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos_json = json.load(f)
    except Exception as e:
        print(f"Error cargando el JSON de resultados: {e}")
        return

    # Extraer automáticamente los algoritmos presentes en el JSON (ej. "ACO" y "GA")
    algoritmos_presentes = datos_json["metadata"].get("algoritmos_comparados", [])
    
    algoritmos_claves = []
    for alg_nombre in algoritmos_presentes:
        if "ACO" in alg_nombre or "ACA" in alg_nombre:
             algoritmos_claves.append("ACO")
        elif "GA" in alg_nombre:
             algoritmos_claves.append("GA")

    if not algoritmos_claves:
         algoritmos_claves = ["ACO", "GA"]

    if algoritmos_filtrar:
        algoritmos_claves = [a for a in algoritmos_claves if a.upper() in [x.upper() for x in algoritmos_filtrar]]

    claves_disponibles = list(datos_json["experimentos"].keys())
    if experimentos_filtrar:
        claves_disponibles = [k for k in claves_disponibles if k in experimentos_filtrar]

    for algoritmo in algoritmos_claves:
        print(f"\nProcesando gráficos para {algoritmo}...")
        for clave_exp in claves_disponibles:
            generar_gif_avanzado(
                experimento_key=clave_exp,
                algoritmo=algoritmo,
                datos=datos_json,
                lons=lons,
                lats=lats,
                nombres=nombres,
                dir_gifs=dir_gifs,
                gdf_carreteras=gdf_carreteras,
                fps=10
            )
            
        generar_imagen_comparativa_final(
            datos=datos_json,
            algoritmo=algoritmo,
            lons=lons,
            lats=lats,
            dir_salida=dir_gifs,
            gdf_carreteras=gdf_carreteras,
        )
        
    print("\n¡Proceso Finalizado Exitosamente!")

if __name__ == "__main__":
    # Ahora el orquestador se encarga de ejecutar ambos automáticamente
    ejecutar_orquestador_visual()