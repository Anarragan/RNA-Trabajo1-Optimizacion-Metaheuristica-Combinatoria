from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, NullFormatter

from schwefel import schwefel, X_OPT
from optimizers import gradiente_schwefel

LIM = 500
EPS = 1e-8
# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _malla(n=250):
    """Superficie de Schwefel 2D (se calcula una sola vez)."""
    g = np.linspace(-LIM, LIM, n)
    XX, YY = np.meshgrid(g, g)
    ZZ = schwefel(np.column_stack((XX.ravel(), YY.ravel()))).reshape(XX.shape)
    return XX, YY, ZZ

def _validar(hist_x, hist_f):
    hist_x = np.asarray(hist_x, dtype=float)
    hist_f = np.asarray(hist_f, dtype=float).reshape(-1)
    if hist_x.ndim != 2 or hist_x.shape[1] != 2:
        raise ValueError("El GIF solo está definido para trayectorias 2D.")
    if len(hist_x) != len(hist_f) or len(hist_f) == 0:
        raise ValueError("Los historiales deben ser no vacíos y de igual longitud.")
    if not (np.isfinite(hist_x).all() and np.isfinite(hist_f).all()):
        raise ValueError("Los historiales contienen valores no finitos.")
    return hist_x, hist_f

def _ajustar_f(hist_f):
    """Desplaza f para poder usar escala log aunque f llegue a 0 o sea negativa."""
    return hist_f - min(0.0, float(hist_f.min())) + EPS

def _estado(f, k, T, umbral, activo):
    if f <= umbral:
        return "ÉXITO"
    return "FIN" if k == T else activo

def _fondo_2d(ax):
    """Mapa de la función + óptimo global."""
    XX, YY, ZZ = _malla()
    im = ax.imshow(ZZ, origin="lower", extent=[-LIM, LIM, -LIM, LIM],
                   cmap="viridis", aspect="equal", interpolation="bilinear")
    # Pocos contornos y tenues: con muchos, la paleta de 256 colores del GIF
    # produce ruido y bandas visibles.
    ax.contour(XX, YY, ZZ, levels=12, colors="k", linewidths=0.3, alpha=0.25)
    ax.plot(X_OPT, X_OPT, "*", color="red", markersize=15,
            markeredgecolor="white", label="Óptimo global", zorder=6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.figure.colorbar(im, ax=ax, label="f(x)", shrink=0.8, pad=0.02)

# --------------------------------------------------------------------------
# Motor de animación común
# --------------------------------------------------------------------------
@dataclass
class Panel:
    titulo: str
    ylabel: str
    series: list                 # [(etiqueta, datos, color), ...]
    log: bool = False
    ylim: tuple | None = None
    referencia: float | None = None

def _ylim_auto(panel):
    datos = np.concatenate([np.asarray(s[1], float) for s in panel.series])
    if panel.log:
        pos = datos[datos > 0]
        return (pos.min() / 2, pos.max() * 2) if len(pos) else (EPS, 1.0)
    lo, hi = float(datos.min()), float(datos.max())
    pad = 0.08 * (hi - lo or 1.0)
    return lo - pad, hi + pad

def _animar(hist_x, paneles, titulo, output_path, *, extra=None,
            etiqueta_punto="Posición actual", n_frames=60, fps=10,
            dpi=90, pausa_final=10):
    """
    hist_x   : (T+1, 2) trayectoria a dibujar sobre el mapa.
    paneles  : lista de Panel (columna derecha).
    titulo   : función k -> str.
    extra    : función ax_mapa -> callable(k) para dibujar elementos extra.
    """
    T = len(hist_x) - 1
    fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)
    gs = GridSpec(len(paneles), 2, width_ratios=[1.3, 1], figure=fig)

    # ---- mapa ----
    ax_m = fig.add_subplot(gs[:, 0])
    _fondo_2d(ax_m)
    ax_m.plot(*hist_x[0], "o", color="white", markeredgecolor="black",
              markersize=8, label="Inicio", zorder=5)
    halo = [pe.withStroke(linewidth=4, foreground="black")]
    tray, = ax_m.plot([], [], "-", color="orange", lw=1.8, path_effects=halo)
    punto, = ax_m.plot([], [], "D", color="yellow", markeredgecolor="black",
                       markersize=8, zorder=7, label=etiqueta_punto)
    upd_extra = extra(ax_m) if extra else (lambda k: None)
    ax_m.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ---- paneles de series ----
    lineas, ax0 = [], None
    for i, p in enumerate(paneles):
        ax = fig.add_subplot(gs[i, 1], sharex=ax0)
        ax0 = ax0 or ax
        if p.log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        grupo = []
        for etiqueta, datos, color in p.series:
            datos = np.maximum(datos, EPS) if p.log else np.asarray(datos)
            ln, = ax.plot([], [], color=color, lw=1.6, label=etiqueta)
            grupo.append((ln, datos))
        if p.referencia is not None:
            ax.axhline(p.referencia, ls="--", color="red", lw=1, label="Óptimo")
        ax.set_xlim(0, max(T, 1))
        ax.set_ylim(*(p.ylim or _ylim_auto(p)))
        ax.set_title(p.titulo, fontsize=9, loc="left")
        ax.set_ylabel(p.ylabel, fontsize=8)
        ax.grid(alpha=0.3)
        if len(p.series) > 1 or p.referencia is not None:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
        if i < len(paneles) - 1:
            ax.tick_params(labelbottom=False)
        lineas.append(grupo)
    ax.set_xlabel("Iteración")

    txt = fig.suptitle("", fontsize=10, family="monospace")

    def update(k):
        tray.set_data(hist_x[:k + 1, 0], hist_x[:k + 1, 1])
        punto.set_data([hist_x[k, 0]], [hist_x[k, 1]])
        it = np.arange(k + 1)
        for grupo in lineas:
            for ln, datos in grupo:
                ln.set_data(it, datos[:k + 1])
        upd_extra(k)
        txt.set_text(titulo(k))

    frames = np.unique(np.linspace(0, T, min(n_frames, T + 1)).astype(int))
    frames = np.concatenate([frames, np.full(pausa_final, T)])  # pausa al final

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                         blit=False, repeat=False)
    try:
        # PillowWriter elige el formato por extensión: .gif, .webp o .png (APNG).
        anim.save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)
    print(f"Animación guardada en: {out}")

# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------
def make_gd_gif(hist_x, hist_f, output_path, n_frames=60, fps=10, dpi=90,
                umbral=1e-3):
    """Anima una ejecución de GD en 2D."""
    hist_x, hist_f = _validar(hist_x, hist_f)
    T = len(hist_f) - 1
    grad = np.linalg.norm(gradiente_schwefel(hist_x), axis=1)
    dist = np.linalg.norm(hist_x - X_OPT, axis=1)
    paso = np.r_[0.0, np.linalg.norm(np.diff(hist_x, axis=0), axis=1)]

    paneles = [
        Panel("Valor objetivo", "f(x) ajustada",
              [("f", _ajustar_f(hist_f), "tab:blue")], log=True),
        Panel("Coordenadas", "x",
              [("x1", hist_x[:, 0], "tab:blue"),
               ("x2", hist_x[:, 1], "tab:orange")],
              ylim=(-LIM, LIM), referencia=X_OPT),
        Panel("Gradiente y tamaño de paso", "magnitud",
              [("||∇f||", grad, "tab:red"), ("paso", paso, "tab:green")],
              log=True),
    ]

    def titulo(k):
        e = _estado(hist_f[k], k, T, umbral, "OPTIMIZANDO")
        return (f"GD | iter {k}/{T} | f={hist_f[k]:.5g} | "
                f"||∇f||={grad[k]:.1e} | dist={dist[k]:.1f} | {e}")

    _animar(hist_x, paneles, titulo, output_path,
            n_frames=n_frames, fps=fps, dpi=dpi)

def make_metaheuristic_gif(hist_pob, hist_x, hist_f, output_path, n_frames=60,
                           fps=10, dpi=90, umbral=1e-3, nombre="PSO"):
    """
    Anima un optimizador poblacional en 2D.
    hist_pob: (T+1, población, 2); hist_x: mejor individuo por generación.
    """
    hist_x, hist_f = _validar(hist_x, hist_f)
    hist_pob = np.asarray(hist_pob, dtype=float)
    if hist_pob.ndim != 3 or hist_pob.shape[2] != 2 \
            or hist_pob.shape[0] != len(hist_f):
        raise ValueError("hist_pob debe tener forma (T+1, población, 2).")

    T = len(hist_f) - 1
    diversidad = hist_pob.std(axis=1).mean(axis=1)
    dist = np.linalg.norm(hist_x - X_OPT, axis=1)

    paneles = [
        Panel("Mejor valor encontrado", "f(mejor) ajustada",
              [("f*", _ajustar_f(hist_f), "tab:blue")], log=True),
        Panel("Dispersión de la población", "diversidad",
              [("diversidad", diversidad, "tab:green")],
              ylim=(0, max(float(diversidad.max()) * 1.1, EPS))),
        Panel("Coordenadas del mejor", "x",
              [("x1", hist_x[:, 0], "tab:blue"),
               ("x2", hist_x[:, 1], "tab:orange")],
              ylim=(-LIM, LIM), referencia=X_OPT),
    ]

    def extra(ax):
        nube = ax.scatter([], [], s=18, c="orange", alpha=0.7,
                          edgecolors="black", linewidths=0.3, zorder=4,
                          label="Partículas")
        return lambda k: nube.set_offsets(hist_pob[k])

    def titulo(k):
        e = _estado(hist_f[k], k, T, umbral, "EXPLORANDO")
        return (f"{nombre} | gen {k}/{T} | f*={hist_f[k]:.5g} | "
                f"diversidad={diversidad[k]:.1f} | dist={dist[k]:.1f} | {e}")

    _animar(hist_x, paneles, titulo, output_path, extra=extra,
            etiqueta_punto="Mejor solución",
            n_frames=n_frames, fps=fps, dpi=dpi)