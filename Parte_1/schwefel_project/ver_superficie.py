"""Visor interactivo de la función de Schwefel (rotar con el mouse)."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from schwefel import schwefel, X_OPT

LIM = 500


def mostrar_superficie(n=120, elev=45, azim=45, guardar=None):
    g = np.linspace(-LIM, LIM, n)
    XX, YY = np.meshgrid(g, g)
    ZZ = schwefel(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(XX.shape)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    sup = ax.plot_surface(XX, YY, ZZ, cmap="viridis", rstride=2, cstride=2,
                          linewidth=0, antialiased=False)
    fig.colorbar(sup, ax=ax, shrink=0.6, label="f(x)")
    ax.scatter([X_OPT], [X_OPT], [0], c="red", marker="*", s=250, edgecolors="white",
               depthshade=False, label=f"Óptimo global ({X_OPT:.2f}, {X_OPT:.2f})")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f(x1, x2)")
    ax.set_title("Función de Schwefel (2 variables) - arrastra con el mouse para rotar")
    ax.legend(loc="upper left")
    ax.view_init(elev=elev, azim=azim)

    if guardar:
        fig.savefig(guardar, dpi=120, bbox_inches="tight")
        print("Imagen guardada en", guardar)
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Superficie 3D interactiva de Schwefel")
    p.add_argument("--n", type=int, default=120, help="puntos por lado (más = más detalle, rotación más lenta)")
    p.add_argument("--elev", type=float, default=45, help="elevación inicial de la cámara")
    p.add_argument("--azim", type=float, default=45, help="azimut inicial de la cámara")
    p.add_argument("--guardar", metavar="RUTA.png", help="en vez de abrir la ventana, guarda una imagen")
    a = p.parse_args()
    mostrar_superficie(a.n, a.elev, a.azim, a.guardar)