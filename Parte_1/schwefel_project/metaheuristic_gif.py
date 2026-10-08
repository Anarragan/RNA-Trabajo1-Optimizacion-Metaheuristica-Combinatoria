from visualize import make_metaheuristic_gif
from optimizers import evolucion_diferencial
from schwefel import schwefel
import numpy as np

rng = np.random.default_rng(0)
res = evolucion_diferencial(schwefel, dim=2, rng=rng,
                            pop_size=40, max_iter=500,
                            record_population=True)   # ← CLAVE

make_metaheuristic_gif(
    hist_pob=res["historial_poblacion"],
    hist_x  =res["historial_x"],
    hist_f  =res["historial_f"],
    output_path="results/de_2d.gif",
    n_frames=80, fps=12, nombre="DE",
)