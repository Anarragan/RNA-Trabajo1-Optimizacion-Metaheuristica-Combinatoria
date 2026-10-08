import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from pathlib import Path
from schwefel import schwefel, X_OPT
from optimizers import gradiente_schwefel

LIM = 500

PARES = {2: [(0, 1)], 3: [(0, 1), (0, 2), (1, 2)]}   # planos (xi, xj) a mostrar

def _malla(n):
    g = np.linspace(-LIM, LIM, n)
    XX, YY = np.meshgrid(g, g)
    ZZ = schwefel(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(XX.shape)
    return XX, YY, ZZ

def _fondo_2d(ax, nombres, colorbar=False, n=200):
    XX, YY, ZZ = _malla(n)
    im = ax.imshow(ZZ, origin="lower", extent=[-LIM, LIM, -LIM, LIM], cmap="viridis", aspect="auto")
    ax.contour(XX, YY, ZZ, levels=20, colors="k", linewidths=0.4, alpha=0.5)   # curvas de nivel
    ax.plot(X_OPT, X_OPT, "r*", ms=14, mec="white")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(nombres[0]); ax.set_ylabel(nombres[1])
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="f(x)", shrink=0.8)
def make_gd_gif(hist_x, hist_f, output_path, n_frames=60, fps=10, dpi=70, umbral=1e-3):
    if hist_x.shape[1] != 2:
        raise ValueError("El GIF solo está definido para 2 variables (mapa de niveles 2D).")
    T = len(hist_f) - 1
    hist_g = np.linalg.norm(gradiente_schwefel(hist_x), axis=1)
    hist_d = np.linalg.norm(hist_x - X_OPT, axis=1)
    hist_p = np.r_[np.inf, np.linalg.norm(np.diff(hist_x, axis=0), axis=1)]  # tamaño del paso
    frames = np.unique(np.linspace(0, T, n_frames).astype(int))

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(3, 2, width_ratios=[1.3, 1], figure=fig)
    ax_m = fig.add_subplot(gs[:, 0])
    _fondo_2d(ax_m, ("x1", "x2"), colorbar=True)
    ax_m.plot(*hist_x[0], "wo", mec="k", ms=7)
    tray, = ax_m.plot([], [], "-", color="orange", lw=2)
    punto, = ax_m.plot([], [], "o", color="orange", mec="k", ms=7, zorder=5)

    ax_f = fig.add_subplot(gs[0, 1]); ax_x = fig.add_subplot(gs[1, 1]); ax_g = fig.add_subplot(gs[2, 1])
    fp = np.maximum(hist_f, 1e-6); gp = np.maximum(hist_g, 1e-6)
    lf, = ax_f.semilogy([], [], color="tab:blue")
    ax_f.set_xlim(0, T); ax_f.set_ylim(fp.min() * 0.5, fp.max() * 2); ax_f.set_ylabel("f(x)")
    lg, = ax_g.semilogy([], [], color="tab:red")
    ax_g.set_xlim(0, T); ax_g.set_ylim(gp.min() * 0.5, gp.max() * 2)
    ax_g.set_ylabel("|grad f|"); ax_g.set_xlabel("iteración")
    lxs = [ax_x.plot([], [], label=f"x{i+1}")[0] for i in range(2)]
    ax_x.axhline(X_OPT, ls="--", c="r", lw=1, label="X_OPT")
    ax_x.set_xlim(0, T); ax_x.set_ylim(-LIM, LIM); ax_x.set_ylabel("x")
    ax_x.legend(fontsize=7, ncol=3)
    for a in (ax_f, ax_g):    # ejes log con texto plano: evita el mathtext lento
        a.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        a.yaxis.set_minor_formatter(NullFormatter())
    for a in (ax_f, ax_x, ax_g): a.grid(alpha=.3)
    titulo = fig.suptitle("", fontsize=11, family="monospace")

    def update(k):
        h = hist_x[:k + 1]
        tray.set_data(h[:, 0], h[:, 1]); punto.set_data([h[-1, 0]], [h[-1, 1]])
        it = np.arange(k + 1)
        lf.set_data(it, fp[:k + 1]); lg.set_data(it, gp[:k + 1])
        for i, l in enumerate(lxs): l.set_data(it, h[:, i])
        if hist_f[k] <= umbral:   estado = "ÓPTIMO GLOBAL"
        elif hist_p[k] < 1e-3:    estado = "DETENIDO (mín. local o frontera)"
        else:                     estado = "descendiendo"
        titulo.set_text(f"iter {k}/{T} | f={hist_f[k]:.4g} | |grad|={hist_g[k]:.2e} "
                        f"| dist={hist_d[k]:.1f} | {estado}")

    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=frames).save(str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

def make_metaheuristic_gif(hist_pob, hist_x, hist_f, output_path,
                           n_frames=60, fps=10, dpi=70, umbral=1e-3,
                           nombre="PSO"):
    if hist_x.shape[1] != 2:
        raise ValueError("El GIF solo está definido para 2 variables.")

    T = len(hist_f) - 1
    diversidad = np.std(hist_pob, axis=1).mean(axis=1)      # (T+1,)
    frames = np.unique(np.linspace(0, T, n_frames).astype(int))

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(3, 2, width_ratios=[1.3, 1], figure=fig)

    ax_m = fig.add_subplot(gs[:, 0])
    _fondo_2d(ax_m, ("x1", "x2"), colorbar=True)
    nube = ax_m.scatter([], [], s=18, c="orange", alpha=0.6,
                        edgecolors="k", linewidths=0.3, zorder=4)
    mejor, = ax_m.plot([], [], "*", color="red", mec="white", ms=16, zorder=5)

    ax_f = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[2, 1])

    fp = np.maximum(hist_f, 1e-6)
    lf, = ax_f.semilogy([], [], color="tab:blue")
    ax_f.set_xlim(0, T); ax_f.set_ylim(fp.min()*0.5, fp.max()*2)
    ax_f.set_ylabel("f(mejor)")

    ld, = ax_d.plot([], [], color="tab:green")
    ax_d.set_xlim(0, T); ax_d.set_ylim(0, diversidad.max()*1.1)
    ax_d.set_ylabel("diversidad")

    lx1, = ax_x.plot([], [], label="x1*")
    lx2, = ax_x.plot([], [], label="x2*")
    ax_x.axhline( X_OPT, ls="--", c="r", lw=1, label="X_OPT")
    ax_x.axhline(-X_OPT, ls="--", c="r", lw=1)
    ax_x.set_xlim(0, T); ax_x.set_ylim(-LIM, LIM)
    ax_x.set_xlabel("iteración"); ax_x.set_ylabel("x*")
    ax_x.legend(fontsize=7, ncol=3)

    for a in (ax_f, ax_d, ax_x): a.grid(alpha=.3)
    ax_f.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax_f.yaxis.set_minor_formatter(NullFormatter())

    titulo = fig.suptitle("", fontsize=11, family="monospace")

    def update(k):
        nube.set_offsets(hist_pob[k])
        mejor.set_data([hist_x[k, 0]], [hist_x[k, 1]])
        it = np.arange(k + 1)
        lf.set_data(it, fp[:k+1])
        ld.set_data(it, diversidad[:k+1])
        lx1.set_data(it, hist_x[:k+1, 0])
        lx2.set_data(it, hist_x[:k+1, 1])
        estado = "ÓPTIMO" if hist_f[k] <= umbral else "explorando"
        titulo.set_text(f"{nombre} | iter {k}/{T} | f*={hist_f[k]:.4g} "
                        f"| div={diversidad[k]:.1f} | {estado}")

    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=frames).save(
        str(out), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)