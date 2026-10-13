"""
Módulo Estandarizado de Visualización y Generación de GIFs para el TSP en España.

Diseñado de forma agnóstica al algoritmo metaheurístico utilizado:
puede visualizar historiales y rutas generadas por Colonia de Hormigas (ACO),
Algoritmos Genéticos (GA), o cualquier otra metaheurística.

Genera:
1. GIFs animados con panel sincronizado (Mapa Geográfico de España + Curva de Convergencia).
2. Imagen estática comparativa de las mejores rutas para diferentes valores de hora del vendedor.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para entornos sin display

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


def cargar_coordenadas_ciudades(directorio_base: Path) -> pd.DataFrame:
    """
    Carga el archivo de ciudades con sus coordenadas geográficas (latitud y longitud).

    Parámetros:
        directorio_base (Path): Directorio donde se encuentra el archivo 'ciudades_espana.csv'.

    Retorna:
        pd.DataFrame: DataFrame con columnas ['ciudad', 'lat', 'lon'] para las 47 capitales provinciales.
    """
    ruta_ciudades = directorio_base / "ciudades_espana.csv"
    if not ruta_ciudades.exists():
        raise FileNotFoundError(f"No se encontró el archivo de ciudades: '{ruta_ciudades}'")
    return pd.read_csv(ruta_ciudades)


def crear_animacion_tsp(
    historial_rutas: List[List[int]],
    historial_costos: List[float],
    df_ciudades: pd.DataFrame,
    titulo_problema: str,
    ruta_salida_gif: Path,
    distancia_final_km: Optional[float] = None,
    fps: int = 6,
    submuestreo_pasos: int = 1,
) -> Path:
    """
    Crea y guarda un GIF animado sincronizado de 2 paneles:
    - Izquierda: Mapa de España peninsular con la ruta evolucionando entre las 47 ciudades.
    - Derecha: Curva de convergencia del costo total en cada iteración.

    Parámetros:
        historial_rutas (list): Lista de listas con la secuencia de índices de ciudades por iteración.
        historial_costos (list): Lista de costos totales (€) en cada iteración.
        df_ciudades (pd.DataFrame): DataFrame con latitud y longitud de las 47 ciudades.
        titulo_problema (str): Título identificador (ej. 'Colonia de Hormigas - 25.0 EUR/h').
        ruta_salida_gif (Path): Ruta donde se guardará el archivo GIF.
        fps (int): Cuadros por segundo de la animación.
        submuestreo_pasos (int): Frecuencia de frames para controlar el peso del GIF.

    Retorna:
        Path: Ruta del GIF generado.
    """
    ruta_salida_gif.parent.mkdir(parents=True, exist_ok=True)

    lons = df_ciudades["lon"].values
    lats = df_ciudades["lat"].values
    nombres = df_ciudades["ciudad"].values
    n_ciudades = len(df_ciudades)

    # Submuestreo de frames si hay muchas iteraciones para un GIF fluido y ligero
    total_iter = len(historial_rutas)
    indices_frames = list(range(0, total_iter, submuestreo_pasos))
    if indices_frames[-1] != total_iter - 1:
        indices_frames.append(total_iter - 1)

    # Configuración de figura con estética cuidada
    fig, (ax_mapa, ax_costo) = plt.subplots(1, 2, figsize=(15, 7), dpi=100)
    fig.patch.set_facecolor("#0f172a")  # Fondo azul oscuro elegante (slate-900)

    # Panel Mapa
    ax_mapa.set_facecolor("#1e293b")
    ax_mapa.set_aspect("equal")
    ax_mapa.set_xlim(lons.min() - 1.0, lons.max() + 1.0)
    ax_mapa.set_ylim(lats.min() - 0.8, lats.max() + 0.8)
    ax_mapa.tick_params(colors="#94a3b8", labelsize=9)
    ax_mapa.set_xlabel("Longitud (°)", color="#94a3b8", fontsize=10)
    ax_mapa.set_ylabel("Latitud (°)", color="#94a3b8", fontsize=10)
    ax_mapa.grid(True, linestyle="--", alpha=0.2, color="#64748b")

    # Dibujar ciudades fijas
    ax_mapa.scatter(lons, lats, color="#38bdf8", s=35, zorder=4, edgecolor="#0284c7", linewidth=1.2)

    # Destacar algunas capitales representativas con etiquetas
    ciudades_clave = {"Madrid", "Barcelona", "Sevilla", "Valencia", "Bilbao", "A Coruña", "Zaragoza"}
    for i, nombre in enumerate(nombres):
        if nombre in ciudades_clave:
            ax_mapa.annotate(
                nombre,
                (lons[i], lats[i]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="#f8fafc",
                weight="bold",
                zorder=5,
            )

    (linea_ruta,) = ax_mapa.plot([], [], color="#f59e0b", linewidth=1.8, alpha=0.85, zorder=3)
    punto_inicio = ax_mapa.scatter([], [], color="#ef4444", s=90, marker="*", zorder=6, label="Ciudad Inicial")
    texto_mapa = ax_mapa.text(
        0.03, 0.94, "", transform=ax_mapa.transAxes, color="#f8fafc", fontsize=11,
        weight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", alpha=0.8, edgecolor="#334155")
    )

    # Panel Curva de Costo
    ax_costo.set_facecolor("#1e293b")
    ax_costo.set_xlim(0, max(1, total_iter - 1))
    costo_min = min(historial_costos)
    costo_max = max(historial_costos)
    margen = (costo_max - costo_min) * 0.08 if costo_max != costo_min else 50.0
    ax_costo.set_ylim(costo_min - margen, costo_max + margen)
    ax_costo.tick_params(colors="#94a3b8", labelsize=9)
    ax_costo.set_xlabel("Iteración", color="#94a3b8", fontsize=10)
    ax_costo.set_ylabel("Costo Total (€)", color="#94a3b8", fontsize=10)
    ax_costo.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax_costo.set_title("Evolución del Costo Mínimo", color="#f8fafc", fontsize=12, pad=10)

    # Línea de costo tenue de fondo completo
    ax_costo.plot(range(total_iter), historial_costos, color="#475569", linewidth=1.2, linestyle=":")
    (linea_progreso,) = ax_costo.plot([], [], color="#10b981", linewidth=2.2, zorder=3)
    (punto_actual_costo,) = ax_costo.plot([], [], marker="o", color="#34d399", markersize=6, zorder=4)

    if distancia_final_km is not None:
        fig.suptitle(f"{titulo_problema} — Distancia: {distancia_final_km:,.1f} km", color="#f8fafc", fontsize=13, weight="bold", y=0.98)
    else:
        fig.suptitle(titulo_problema, color="#f8fafc", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def init():
        linea_ruta.set_data([], [])
        linea_progreso.set_data([], [])
        punto_actual_costo.set_data([], [])
        texto_mapa.set_text("")
        return linea_ruta, linea_progreso, punto_actual_costo, texto_mapa

    def update(frame_idx):
        it = indices_frames[frame_idx]
        ruta = historial_rutas[it]
        costo_it = historial_costos[it]

        # Cerrar el circuito (retornar a la primera ciudad)
        ruta_cerrada = list(ruta) + [ruta[0]]
        x_coords = [lons[idx] for idx in ruta_cerrada]
        y_coords = [lats[idx] for idx in ruta_cerrada]

        linea_ruta.set_data(x_coords, y_coords)

        # Actualizar curva de costo hasta la iteración actual
        iteraciones_actuales = list(range(it + 1))
        costos_actuales = historial_costos[: it + 1]
        linea_progreso.set_data(iteraciones_actuales, costos_actuales)
        punto_actual_costo.set_data([it], [costo_it])

        texto_mapa.set_text(f"Iteración: {it + 1}/{total_iter}\nCosto: {costo_it:,.2f} €")

        return linea_ruta, linea_progreso, punto_actual_costo, texto_mapa

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(indices_frames),
        interval=1000 // fps,
        blit=False,
    )

    writer = PillowWriter(fps=fps)
    anim.save(str(ruta_salida_gif), writer=writer)
    plt.close(fig)

    return ruta_salida_gif


def generar_comparativa_estatica(
    datos_experimentos: Dict[str, Any],
    df_ciudades: pd.DataFrame,
    ruta_salida_png: Path,
) -> Path:
    """
    Genera un panel comparativo estático (imagen PNG de alta resolución) mostrando las mejores rutas
    alcanzadas para cada uno de los valores de valor de hora del vendedor.

    Parámetros:
        datos_experimentos (Dict[str, Any]): Diccionario estructurado con los resultados de todas las instancias
                                             (cargado desde 'resultados_aco.json').
        df_ciudades (pd.DataFrame): DataFrame con latitud y longitud de las 47 ciudades provinciales.
        ruta_salida_png (Path): Ruta donde se guardará la imagen comparativa final.

    Retorna:
        Path: Ruta del archivo PNG guardado.
    """
    ruta_salida_png.parent.mkdir(parents=True, exist_ok=True)
    experimentos = datos_experimentos.get("experimentos", {})
    claves = list(experimentos.keys())
    n_exp = len(claves)

    if n_exp == 0:
        return ruta_salida_png

    lons = df_ciudades["lon"].values
    lats = df_ciudades["lat"].values

    cols = 3
    filas = (n_exp + cols - 1) // cols

    fig, axes = plt.subplots(filas, cols, figsize=(18, 5.5 * filas), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    axes = np.array(axes).reshape(-1)

    colores_ruta = ["#38bdf8", "#34d399", "#f59e0b", "#fb7185", "#c084fc", "#fb923c"]

    for idx, clave in enumerate(claves):
        ax = axes[idx]
        exp = experimentos[clave]
        v_hora = exp["valor_hora_vendedor"]
        costo = exp["mejor_costo_total"]
        desglose = exp.get("desglose", {})
        ruta = exp["mejor_ruta_indices"]
        color = colores_ruta[idx % len(colores_ruta)]

        ax.set_facecolor("#1e293b")
        ax.set_aspect("equal")
        ax.set_xlim(lons.min() - 0.8, lons.max() + 0.8)
        ax.set_ylim(lats.min() - 0.6, lats.max() + 0.6)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.15, color="#64748b")

        # Puntos de ciudades
        ax.scatter(lons, lats, color="#64748b", s=25, zorder=3)

        # Ruta cerrada
        ruta_c = list(ruta) + [ruta[0]]
        ax.plot([lons[i] for i in ruta_c], [lats[i] for i in ruta_c], color=color, linewidth=1.6, zorder=4)
        ax.scatter([lons[ruta[0]]], [lats[ruta[0]]], color="#ef4444", s=70, marker="*", zorder=5)

        t_horas = desglose.get("tiempo_total_horas", 0.0)
        d_km = desglose.get("distancia_total_km", 0.0)
        c_peajes = desglose.get("costo_peajes_eur", 0.0)
        c_comb = desglose.get("costo_combustible_eur", 0.0)

        ax.set_title(
            f"Hora: {v_hora:.1f} €/h | Costo: {costo:,.2f} €\n"
            f"Distancia: {d_km:,.1f} km | Tiempo: {t_horas:.1f} h\n"
            f"(Comb: {c_comb:,.0f} € | Peajes: {c_peajes:,.0f} €)",
            color="#f8fafc",
            fontsize=9.5,
            weight="bold",
            pad=6,
        )

    # Ocultar ejes sobrantes si los hay
    for idx in range(n_exp, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "Comparativa de Rutas Óptimas según Valor de la Hora del Vendedor (España Peninsular 47)",
        color="#f8fafc",
        fontsize=14,
        weight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(str(ruta_salida_png), facecolor=fig.get_facecolor())
    plt.close(fig)

    return ruta_salida_png


def generar_todos_los_gifs(
    archivo_json: str = "datos_graficas/resultados_aco.json",
    carpeta_salida_gifs: str = "gifs",
    fps: int = 8,
    submuestreo_pasos: int = 2,
) -> List[Path]:
    """
    Función principal de visualización:
    Carga los resultados exportados (JSON) y produce un GIF animado para cada experimento,
    además de la imagen estática comparativa global.

    Parámetros:
        archivo_json (str): Ruta relativa al archivo JSON con los resultados de la metaheurística.
                            Por defecto "datos_graficas/resultados_aco.json".
        carpeta_salida_gifs (str): Subdirectorio de salida para almacenar las animaciones y la imagen estática.
                                   Por defecto "gifs".
        fps (int): Cuadros por segundo de las animaciones generadas. Por defecto 8.
        submuestreo_pasos (int): Frecuencia de frames para controlar el tamaño del archivo GIF. Por defecto 2.

    Retorna:
        List[Path]: Lista con las rutas a todos los archivos GIF creados.
    """
    directorio_base = Path(__file__).resolve().parent
    ruta_json = directorio_base / archivo_json
    dir_gifs = directorio_base / carpeta_salida_gifs
    dir_gifs.mkdir(parents=True, exist_ok=True)

    if not ruta_json.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: '{ruta_json}'")

    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    df_ciudades = cargar_coordenadas_ciudades(directorio_base)
    experimentos = datos.get("experimentos", {})

    print("=" * 80)
    print("GENERANDO ANIMACIONES GIF Y COMPARATIVAS DE RUTAS TSP")
    print(f"Directorio de salida: '{dir_gifs.relative_to(directorio_base)}'")
    print("=" * 80)

    gifs_generados: List[Path] = []

    for clave, exp in experimentos.items():
        v_hora = exp["valor_hora_vendedor"]
        hist_rutas = exp["historial_rutas"]
        hist_costos = exp["historial_costos"]

        sufijo = f"{int(v_hora)}" if float(v_hora).is_integer() else f"{v_hora:.1f}".replace(".", "_")
        nombre_gif = f"aco_espana47_hora_{sufijo}.gif"
        ruta_gif = dir_gifs / nombre_gif

        titulo = f"Colonia de Hormigas - España 47 Capitales (Valor Hora: {v_hora:.1f} €/h)"
        print(f">> Creando GIF: '{nombre_gif}' ({len(hist_rutas)} iteraciones)...")

        d_km = exp.get("desglose", {}).get("distancia_total_km") or exp.get("desglose_optimo", {}).get("distancia_total_km")
        crear_animacion_tsp(
            historial_rutas=hist_rutas,
            historial_costos=hist_costos,
            df_ciudades=df_ciudades,
            titulo_problema=titulo,
            ruta_salida_gif=ruta_gif,
            distancia_final_km=d_km,
            fps=fps,
            submuestreo_pasos=submuestreo_pasos,
        )
        gifs_generados.append(ruta_gif)
        print(f"   [OK] Guardado: {ruta_gif.name}")

    # Generar comparativa estática en PNG
    ruta_comparativa_png = dir_gifs / "comparativa_rutas_horas.png"
    print("\n>> Generando imagen comparativa estática...")
    generar_comparativa_estatica(datos, df_ciudades, ruta_comparativa_png)
    print(f"   [OK] Guardado: {ruta_comparativa_png.name}")

    print("=" * 80)
    print(f"Proceso completado: {len(gifs_generados)} GIFs y 1 comparativa generados exitosamente.")
    print("=" * 80)

    return gifs_generados


if __name__ == "__main__":
    generar_todos_los_gifs()
