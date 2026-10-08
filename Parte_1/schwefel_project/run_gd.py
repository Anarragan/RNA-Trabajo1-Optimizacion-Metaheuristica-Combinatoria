import argparse
import json
import os
import numpy as np
from schwefel import schwefel
from optimizers import gradiente_schwefel
from experiments import run_gd_experiments, RESULTS_DIR

def build_parser():
    p = argparse.ArgumentParser(description="GD sobre Schwefel: experimento + GIFs")
    p.add_argument("--config", help="archivo JSON con parámetros (la línea de comandos tiene prioridad)")
    p.add_argument("--dim", type=int, nargs="+", help="Dimensión o dimensiones a evaluar (ej: --dim 2 o --dim 2 3)")
    p.add_argument("--x0", nargs="+", type=float, default= None, help="Condición inicial (ej: --x0 400.0 -350.0). Si se omite, se genera aleatoria."),
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje")
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--seed", type=int, default=None, help="semilla base; la corrida i usa seed+i")
    p.add_argument("--tag", default=None, help="nombre de salida; {dim} se reemplaza")
    p.add_argument("--gif-run", type=int, default=0, help="índice de la corrida a animar")
    p.add_argument("--gif-frames", type=int, default=60)
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--algo", choices=["gd", "pso", "de", "ea"], default="gd", help="Algoritmo a ejecutar"),
    p.add_argument("--pop-size", type=int, default=None,help="Tamaño de población (PSO/DE/EA)")

    return p
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # 1. Cargar configuración base del JSON si existe
    config_base = {}
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_base = json.load(f)

    # 2. Mezclar (Prioridad Terminal > JSON > Hardcoded)
    dims = args.dim if args.dim is not None else config_base.get("dim", [2, 3])
    runs = args.runs if args.runs is not None else config_base.get("runs", 30)
    lr = args.lr if args.lr is not None else config_base.get("lr", 0.5)
    max_iter = args.max_iter if args.max_iter is not None else config_base.get("max_iter", 2000)
    tol = args.tol if args.tol is not None else config_base.get("tol", 1e-6)
    seed = args.seed if args.seed is not None else config_base.get("seed", 0)
    tag_template = args.tag if args.tag is not None else config_base.get("tag", "gd_{dim}d")

    # El bloque de algoritmos alternativos usa la primera dimensión solicitada.
    for d in dims:
        current_tag = tag_template.format(dim=d)
        print(f"... Ejecutando {current_tag}")
   # si el usuario no especifica un algoritmo, usamos GD por defecto; si especifica otro, usamos el runner correspondiente
        if args.algo != "gd":
            from optimizers import pso, evolucion_diferencial, algoritmo_evolutivo
            pop_size = args.pop_size if args.pop_size is not None else config_base.get("pop_size", 40)
            runners = {
                "pso": lambda rng: pso(schwefel, d, rng, pop_size=pop_size,
                                    max_iter=max_iter, bounds=(-500.0, 500.0)),
                "de":  lambda rng: evolucion_diferencial(schwefel, d, rng, pop_size=pop_size,
                                                        max_iter=max_iter, bounds=(-500.0, 500.0)),
                "ea":  lambda rng: algoritmo_evolutivo(schwefel, d, rng, pop_size=pop_size,
                                                    max_iter=max_iter, bounds=(-500.0, 500.0)),
            }
            fs, its, evs, exitos = [], [], [], []
            umbral = {2: 1e-3, 3: 1e-2, 4: 5e-2, 5: 1e-1}.get(d, 1e-1)
            for run in range(runs):
                rng = np.random.default_rng(seed + run)
                res = runners[args.algo](rng)
                fs.append(res["f_final"]); its.append(res["iteraciones"])
                evs.append(res["n_evals_f"]); exitos.append(res["f_final"] <= umbral)

            fs = np.array(fs)
            print(f" -> Algoritmo: {args.algo.upper()}")
            print(f" -> Mejor f(x*): {fs.min():.6e}")
            print(f" -> Media f(x*): {fs.mean():.6e}")
            print(f" -> Peor f(x*) : {fs.max():.6e}")
            print(f" -> Tasa éxito : {np.mean(exitos)*100:.0f}%")
            print(f" -> Iter media : {np.mean(its):.1f}")
            print(f" -> Evals media: {np.mean(evs):.0f}")
            continue  # Saltar la ejecución de GD si se usó otro algoritmo


    # 3. Ajuste si el usuario fuerza una condición inicial estática (--x0)
    if args.x0 is not None:
        runs = 1  # Forzamos 1 sola corrida ya que no es aleatoria
        # Si el usuario no especificó la dimensión en comandos, la inferimos del tamaño de x0
        if args.dim is None:
            dims = [len(args.x0)]
        else:
            if len(args.x0) != dims[0]:
                parser.error(f"El tamaño de --x0 ({len(args.x0)}) no coincide con la dimensión dada ({dims[0]})")

    # 4. Iterar sobre las dimensiones solicitadas (ej: primero 2D, luego 3D)
    for d in dims:
        current_tag = tag_template.format(dim=d)
        print(f"\n" + "="*50)
        print(f" EJECUTANDO EXPERIMENTO {d}D | Tag: {current_tag}")
        print(f" LR: {lr} | Max Iter: {max_iter} | Corrida(s): {runs}")
        print("="*50)

        # Si hay x0 manual, modificamos temporalmente la lógica aleatoria interfiriendo en experiments o llamando directo
        if args.x0 is not None:
            # Re-escribimos la lógica para una sola corrida fija usando tu optimizer directo
            from optimizers import descenso_gradiente
            res = descenso_gradiente(
                f=schwefel,
                x0=np.array(args.x0, dtype=float),
                grad_fn=gradiente_schwefel,
                learning_rate=lr,
                max_iter=max_iter,
                tol=tol,
                bounds=(-500.0, 500.0)
            )
            # Imprimir directamente la revisión de consola requerida
            grad_final = gradiente_schwefel(res["x_final"])
            norma_grad = np.linalg.norm(grad_final)
            
            print(f" -> Condición Inicial manual: {args.x0}")
            print(f" -> Iteraciones realizadas: {res['iteraciones']}")
            print(f" -> Valor de la función objetivo f(x*): {res['f_final']:.6f}")
            print(f" -> Valor final de las variables x*: {res['x_final']}")
            print(f" -> Norma del gradiente final: {norma_grad:.2e}")
            print(f" -> ¿Convergió por tolerancia?: {res['convergio']}")
        else:
            # Ejecución por defecto en lote (30 corridas aleatorias) usando tu orquestador
            resumen = run_gd_experiments(
                dim=d, n_runs=runs, learning_rate=lr, max_iter=max_iter,
                tol=tol, base_seed=seed, tag=current_tag
            )
            # Mostrar la revisión de consola histórica del lote completo
            print(f" -> Éxito del lote (Tasa): {resumen['tasa_exito']*100}%")
            print(f" -> Promedio de iteraciones: {resumen['media_iteraciones']:.2f}")
            print(f" -> Mejor f(x*) encontrado: {resumen['mejor']:.6f}")
            print(f" -> Peor f(x*) encontrado: {resumen['peor']:.6f}")
            print(f" -> Tasa de convergencia: {resumen['tasa_convergencia']*100}%")
            
            # Cargar la última corrida guardada para mostrar una muestra individual en consola
            ultima_run_path = os.path.join(RESULTS_DIR, f"{current_tag}_run{runs-1:02d}.npz")
            if os.path.exists(ultima_run_path):
                data = np.load(ultima_run_path)
                grad_final = gradiente_schwefel(data['x_final'])
                print(f"\n [Muestra de la última corrida aleatoria (Run {runs-1})]:")
                print(f"   - X0 generado: {data['x0']}")
                print(f"   - Iteraciones: {data['iteraciones']}")
                print(f"   - f(x*) final: {data['f_final']:.6f}")
                print(f"   - x* final   : {data['x_final']}")
                print(f"   - Norma grad : {np.linalg.norm(grad_final):.2e}")