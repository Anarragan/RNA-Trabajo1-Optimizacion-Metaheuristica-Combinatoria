from dataclasses import dataclass          # dataclass: forma corta de crear clases con atributos
from functools import lru_cache            # lru_cache: guarda resultados de funciones para no recalcular
from pathlib import Path                   # Path: manejo moderno de rutas de archivos

import matplotlib                          # Importamos matplotlib (librería de gráficas)
matplotlib.use("Agg")                      # Modo sin pantalla (necesario para guardar GIFs sin abrir ventanas)

import matplotlib.patheffects as pe        # Efectos visuales (halo alrededor de líneas)
import matplotlib.pyplot as plt            # pyplot: funciones de dibujo
import numpy as np                         # NumPy para cálculos numéricos
from matplotlib.animation import FuncAnimation, PillowWriter  # Animación y escritor de GIF
from matplotlib.gridspec import GridSpec   # GridSpec: para hacer sub-gráficas de tamaño variable
from matplotlib.ticker import FormatStrFormatter, NullFormatter  # Formato de ejes logarítmicos

from rosenbrock import rosenbrock, X_OPT   # Importamos la función y su óptimo (1.0)
from rosenbrock import gradiente_rosenbrock  # Importamos el gradiente (para mostrarlo en el GIF)
LIM = 2        # Rango del mapa de Rosenbrock que vamos a dibujar (de -2 a 2)
EPS = 1e-8     # Número muy pequeño para evitar log(0)


# ============================================================================
# UTILIDADES
# ============================================================================

@lru_cache(maxsize=None)  # Cachea el resultado: solo calcula la malla la primera vez
def _malla(n=250):  # Función que calcula la superficie de Rosenbrock sobre una rejilla
    """Superficie de Rosenbrock 2D (se calcula una sola vez)."""
    g = np.linspace(-LIM, LIM, n)  # Genera n puntos entre -LIM y LIM
    XX, YY = np.meshgrid(g, g)  # Crea una malla 2D con esos puntos
    ZZ = rosenbrock(np.column_stack((XX.ravel(), YY.ravel()))).reshape(XX.shape)  # Evalúa f en toda la malla
    return XX, YY, ZZ  # Devuelve la malla y los valores


def _validar(hist_x, hist_f):  # Verifica que los historiales sean válidos
    hist_x = np.asarray(hist_x, dtype=float)  # Convierte a arreglo de floats
    hist_f = np.asarray(hist_f, dtype=float).reshape(-1)  # Convierte a vector
    if hist_x.ndim != 2 or hist_x.shape[1] != 2:  # Debe ser 2D (T, 2)
        raise ValueError("El GIF solo está definido para trayectorias 2D.")
    if len(hist_x) != len(hist_f) or len(hist_f) == 0:  # Deben tener misma longitud
        raise ValueError("Los historiales deben ser no vacíos y de igual longitud.")
    if not (np.isfinite(hist_x).all() and np.isfinite(hist_f).all()):  # No debe haber NaN ni inf
        raise ValueError("Los historiales contienen valores no finitos.")
    return hist_x, hist_f


def _ajustar_f(hist_f):  # Desplaza f para que sea siempre positiva (por el log)
    return hist_f - min(0.0, float(hist_f.min())) + EPS  # Asegura que mínimo sea >= EPS


def _estado(f, k, T, umbral, activo):  # Determina el texto de estado del GIF
    if f <= umbral:  # Si ya alcanzó el umbral de éxito
        return "ÉXITO"  # Estado: ÉXITO
    return "FIN" if k == T else activo  # Si es el último frame, "FIN"; si no, el estado "activo"


def _fondo_2d(ax):  # Dibuja el mapa de fondo con la superficie de Rosenbrock
    XX, YY, ZZ = _malla()  # Obtenemos la malla (cacheada)
    im = ax.imshow(ZZ, origin="lower", extent=[-LIM, LIM, -LIM, LIM],  # Dibuja la superficie como imagen
                   cmap="viridis", aspect="equal", interpolation="bilinear")  # Con colores viridis
    ax.contour(XX, YY, ZZ, levels=12, colors="k", linewidths=0.3, alpha=0.25)  # Curvas de nivel tenues
    ax.plot(X_OPT, X_OPT, "*", color="red", markersize=15,  # Marca el óptimo global con una estrella roja
            markeredgecolor="white", label="Óptimo global", zorder=6)  # Con borde blanco
    ax.set_xlabel("x1")  # Etiqueta eje x
    ax.set_ylabel("x2")  # Etiqueta eje y
    ax.figure.colorbar(im, ax=ax, label="f(x)", shrink=0.8, pad=0.02)  # Barra de colores lateral


# ============================================================================
# MOTOR DE ANIMACIÓN
# ============================================================================

@dataclass  # Decorador que convierte esto en una clase con atributos automáticos
class Panel:  # Clase que representa un panel de la parte derecha del GIF
    titulo: str                     # Título del panel
    ylabel: str                     # Etiqueta del eje y
    series: list                    # Lista de series: [(etiqueta, datos, color), ...]
    log: bool = False               # Si usa escala logarítmica
    ylim: tuple | None = None       # Límites del eje y (opcional)
    referencia: float | None = None # Línea horizontal de referencia (opcional)


def _ylim_auto(panel):  # Calcula automáticamente los límites del eje y
    datos = np.concatenate([np.asarray(s[1], float) for s in panel.series])  # Junta todos los datos
    if panel.log:  # Si es escala log...
        pos = datos[datos > 0]  # Solo los valores positivos
        return (pos.min()/2, pos.max()*2) if len(pos) else (EPS, 1.0)  # Rango razonable
    lo, hi = float(datos.min()), float(datos.max())  # Si no es log, rango min/max
    pad = 0.08 * (hi - lo or 1.0)  # Añade un 8% de padding
    return lo - pad, hi + pad  # Devuelve el rango con padding


def _animar(hist_x, paneles, titulo, output_path, *, extra=None,  # Función principal de animación
            etiqueta_punto="Posición actual", n_frames=60, fps=10,
            dpi=90, pausa_final=10):
    """Función genérica que crea una animación con el mapa a la izquierda y paneles a la derecha."""
    T = len(hist_x) - 1  # Número total de frames
    fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)  # Creamos la figura
    gs = GridSpec(len(paneles), 2, width_ratios=[1.3, 1], figure=fig)  # Grid: mapa (izq) + paneles (der)

    # ---- MAPA (izquierda) ----
    ax_m = fig.add_subplot(gs[:, 0])  # Subplot que ocupa toda la columna izquierda
    _fondo_2d(ax_m)  # Dibujamos el fondo con la superficie
    ax_m.plot(*hist_x[0], "o", color="white", markeredgecolor="black",  # Marcamos el punto de inicio
              markersize=8, label="Inicio", zorder=5)
    halo = [pe.withStroke(linewidth=4, foreground="black")]  # Efecto halo (borde negro)
    tray, = ax_m.plot([], [], "-", color="orange", lw=1.8, path_effects=halo)  # Línea de trayectoria
    punto, = ax_m.plot([], [], "D", color="yellow", markeredgecolor="black",  # Punto actual (rombo amarillo)
                       markersize=8, zorder=7, label=etiqueta_punto)
    upd_extra = extra(ax_m) if extra else (lambda k: None)  # Elementos extra (ej: nube de partículas)
    ax_m.legend(loc="upper left", fontsize=8, framealpha=0.85)  # Leyenda

    # ---- PANELES (derecha) ----
    lineas, ax0 = [], None  # Lista de líneas y primer eje (para compartir eje x)
    for i, p in enumerate(paneles):  # Recorremos los paneles
        ax = fig.add_subplot(gs[i, 1], sharex=ax0)  # Cada panel en una fila de la columna derecha
        ax0 = ax0 or ax  # Guardamos el primer eje
        if p.log:  # Si es log...
            ax.set_yscale("log")  # Ponemos escala log
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))  # Formato de números
            ax.yaxis.set_minor_formatter(NullFormatter())  # Sin etiquetas menores
        grupo = []  # Grupo de líneas del panel
        for etiqueta, datos, color in p.series:  # Recorremos las series
            datos = np.maximum(datos, EPS) if p.log else np.asarray(datos)  # Ajustamos para log
            ln, = ax.plot([], [], color=color, lw=1.6, label=etiqueta)  # Línea vacía (se llena al animar)
            grupo.append((ln, datos))  # Guardamos la línea y sus datos
        if p.referencia is not None:  # Si hay línea de referencia
            ax.axhline(p.referencia, ls="--", color="red", lw=1, label="Óptimo")  # La dibujamos
        ax.set_xlim(0, max(T, 1))  # Límite del eje x (iteraciones)
        ax.set_ylim(*(p.ylim or _ylim_auto(p)))  # Límite del eje y
        ax.set_title(p.titulo, fontsize=9, loc="left")  # Título
        ax.set_ylabel(p.ylabel, fontsize=8)  # Etiqueta y
        ax.grid(alpha=0.3)  # Rejilla
        if len(p.series) > 1 or p.referencia is not None:  # Si hay leyenda que mostrar
            ax.legend(fontsize=7, ncol=3, loc="upper right")  # Mostramos leyenda
        if i < len(paneles) - 1:  # Si no es el último panel
            ax.tick_params(labelbottom=False)  # Ocultamos etiquetas x (comparten eje)
        lineas.append(grupo)  # Guardamos las líneas del panel
    ax.set_xlabel("Iteración")  # Etiqueta x del último panel

    txt = fig.suptitle("", fontsize=10, family="monospace")  # Título superior (se actualiza en cada frame)

    def update(k):  # Función que actualiza el frame k
        tray.set_data(hist_x[:k+1, 0], hist_x[:k+1, 1])  # Actualizamos la trayectoria (línea naranja)
        punto.set_data([hist_x[k, 0]], [hist_x[k, 1]])  # Actualizamos el punto actual
        it = np.arange(k+1)  # Iteraciones hasta k
        for grupo in lineas:  # Recorremos los paneles
            for ln, datos in grupo:  # Recorremos las series
                ln.set_data(it, datos[:k+1])  # Actualizamos los datos de cada línea
        upd_extra(k)  # Actualizamos elementos extra (partículas)
        txt.set_text(titulo(k))  # Actualizamos el título con la info del frame

    frames = np.unique(np.linspace(0, T, min(n_frames, T+1)).astype(int))  # Elegimos n_frames distribuidos
    frames = np.concatenate([frames, np.full(pausa_final, T)])  # Añadimos pausa al final

    out = Path(output_path)  # Ruta de salida
    out.parent.mkdir(parents=True, exist_ok=True)  # Creamos la carpeta si no existe
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps,  # Creamos la animación
                         blit=False, repeat=False)
    try:
        anim.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)  # Guardamos como GIF
    finally:
        plt.close(fig)  # Cerramos la figura (libera memoria)
    print(f"Animación guardada en: {out}")  # Avisamos al usuario


# ============================================================================
# API PÚBLICA: las dos funciones que usa run.py
# ============================================================================

def make_gd_gif(hist_x, hist_f, output_path, n_frames=60, fps=10, dpi=90,
                umbral=1e-3):  # Función que genera el GIF de descenso por gradiente
    """Anima una ejecución de GD en 2D."""
    hist_x, hist_f = _validar(hist_x, hist_f)  # Validamos los historiales
    T = len(hist_f) - 1  # Número de frames
    grad = np.linalg.norm(gradiente_rosenbrock(hist_x), axis=1)  # Norma del gradiente en cada paso
    dist = np.linalg.norm(hist_x - X_OPT, axis=1)  # Distancia al óptimo en cada paso
    paso = np.r_[0.0, np.linalg.norm(np.diff(hist_x, axis=0), axis=1)]  # Tamaño del paso

    paneles = [  # Definimos 3 paneles a la derecha
        Panel("Valor objetivo", "f(x) ajustada",  # Panel 1: valor de f
              [("f", _ajustar_f(hist_f), "tab:blue")], log=True),  # Escala log
        Panel("Coordenadas", "x",  # Panel 2: coordenadas
              [("x1", hist_x[:, 0], "tab:blue"),
               ("x2", hist_x[:, 1], "tab:orange")],
              ylim=(-LIM, LIM), referencia=X_OPT),  # Referencia: óptimo x=1
        Panel("Gradiente y tamaño de paso", "magnitud",  # Panel 3: gradiente y paso
              [("||∇f||", grad, "tab:red"), ("paso", paso, "tab:green")],
              log=True),
    ]

    def titulo(k):  # Función que genera el título de cada frame
        e = _estado(hist_f[k], k, T, umbral, "OPTIMIZANDO")  # Estado actual
        return (f"GD | iter {k}/{T} | f={hist_f[k]:.5g} | "  # Texto del título
                f"||∇f||={grad[k]:.1e} | dist={dist[k]:.1f} | {e}")

    _animar(hist_x, paneles, titulo, output_path, n_frames=n_frames, fps=fps, dpi=dpi)  # Lanzamos la animación


def make_metaheuristic_gif(hist_pob, hist_x, hist_f, output_path, n_frames=60,
                           fps=10, dpi=90, umbral=1e-3, nombre="PSO"):  # Función que genera el GIF de PSO/DE/EA
    """Anima un optimizador poblacional en 2D."""
    hist_x, hist_f = _validar(hist_x, hist_f)  # Validamos
    hist_pob = np.asarray(hist_pob, dtype=float)  # Convertimos la población a arreglo
    if hist_pob.ndim != 3 or hist_pob.shape[2] != 2 \
            or hist_pob.shape[0] != len(hist_f):  # Verificamos forma (T+1, pop, 2)
        raise ValueError("hist_pob debe tener forma (T+1, población, 2).")

    T = len(hist_f) - 1  # Número de frames
    diversidad = hist_pob.std(axis=1).mean(axis=1)  # Diversidad de la población por iteración
    dist = np.linalg.norm(hist_x - X_OPT, axis=1)  # Distancia al óptimo

    paneles = [  # 3 paneles a la derecha
        Panel("Mejor valor encontrado", "f(mejor) ajustada",  # Panel 1: mejor f
              [("f*", _ajustar_f(hist_f), "tab:blue")], log=True),
        Panel("Dispersión de la población", "diversidad",  # Panel 2: diversidad
              [("diversidad", diversidad, "tab:green")],
              ylim=(0, max(float(diversidad.max())*1.1, EPS))),
        Panel("Coordenadas del mejor", "x",  # Panel 3: coordenadas del mejor
              [("x1", hist_x[:, 0], "tab:blue"),
               ("x2", hist_x[:, 1], "tab:orange")],
              ylim=(-LIM, LIM), referencia=X_OPT),
    ]

    def extra(ax):  # Función que agrega la nube de partículas al mapa
        nube = ax.scatter([], [], s=18, c="orange", alpha=0.7,  # Puntos naranjas (partículas)
                          edgecolors="black", linewidths=0.3, zorder=4,
                          label="Partículas")
        return lambda k: nube.set_offsets(hist_pob[k])  # Actualiza las posiciones en cada frame

    def titulo(k):  # Título de cada frame
        e = _estado(hist_f[k], k, T, umbral, "EXPLORANDO")  # Estado
        return (f"{nombre} | gen {k}/{T} | f*={hist_f[k]:.5g} | "  # Texto del título
                f"diversidad={diversidad[k]:.1f} | dist={dist[k]:.1f} | {e}")

    _animar(hist_x, paneles, titulo, output_path, extra=extra,  # Lanzamos la animación con nube de partículas
            etiqueta_punto="Mejor solución", n_frames=n_frames, fps=fps, dpi=dpi)