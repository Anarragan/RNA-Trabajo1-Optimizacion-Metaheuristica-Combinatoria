import numpy as np
from schwefel import schwefel

def gradiente_schwefel(x):
    """
    Gradiente analítico de la función de Schwefel.
    """
    x = np.asarray(x, dtype=float)

    sqrt_abs_x = np.sqrt(np.abs(x))

    grad = (
        -np.sin(sqrt_abs_x)
        - (sqrt_abs_x / 2) * np.cos(sqrt_abs_x)
    )

    # En x = 0, el valor del gradiente es 0
    grad = np.where(x == 0, 0.0, grad)

    return grad

def descenso_gradiente(
    f,
    x0,
    grad_fn=None,
    learning_rate=0.1,
    max_iter=1000,
    tol=1e-6,
    bounds=(-500.0, 500.0),
    record_every=1,
):
    """
    Descenso por gradiente.

    Si grad_fn es None, usa diferencias centrales (gradiente numérico).
    El parámetro 'record_every' controla cada cuántas iteraciones se
    guarda en el historial (para no saturar el GIF).
    """
    x = np.array(x0, dtype=float)
    n = x.size
    lo, hi = bounds
    hx, hf, n_evals = [x.copy()], [f(x)], 1
    convergio = False

    for it in range(1, max_iter + 1):
        x_new = np.clip(x - learning_rate * grad_fn(x), lo, hi)
        convergio = np.linalg.norm(x_new - x) < tol
        x = x_new

        if it % record_every == 0 or convergio or it == max_iter:
            hx.append(x.copy()); hf.append(f(x)); n_evals += 1
        if convergio:
            break

    return {"x_final": x, "f_final": hf[-1],
            "historial_x": np.array(hx), "historial_f": np.array(hf),
            "iteraciones": it, "convergio": bool(convergio),
            "n_evals_f": n_evals
    }

def _diversidad(x):
    """Dispersión media de la población (desv. estándar por variable, promediada)."""
    return float(np.mean(np.std(x, axis=0)))


def _resultado(nombre, hx, hf, it, convergio, n_evals, meta, pob=None):
    out = {"x_final": hx[-1].copy(), "f_final": float(hf[-1]),
           "historial_x": np.array(hx),          # MEJOR individuo en cada iteración
           "historial_f": np.array(hf),          # f del mejor individuo en cada iteración
           "iteraciones": it, "convergio": bool(convergio), "n_evals_f": n_evals,
           "metadata": {"algoritmo": nombre, **meta}}
    if pob is not None:
        out["historial_poblacion"] = np.array(pob)   # (T+1, pop, dim): para animar
    return out

def pso(f, dim, rng, pop_size=40, max_iter=300, w=0.7298, c1=1.49618, c2=1.49618,
        vmax_frac=0.2, tol=1e-6, bounds=(-500.0, 500.0), record_population=True):
    """Optimización por enjambre de partículas (PSO) con coeficientes de constricción."""
    lo, hi = bounds
    vmax = vmax_frac * (hi - lo)
    x = rng.uniform(lo, hi, (pop_size, dim))
    v = rng.uniform(-vmax, vmax, (pop_size, dim))
    fx = f(x); n_evals = pop_size
    pbest, pbest_f = x.copy(), fx.copy()
    i = pbest_f.argmin(); gbest, gbest_f = pbest[i].copy(), pbest_f[i]
    hx, hf = [gbest.copy()], [gbest_f]
    pob = [x.copy()] if record_population else None
    convergio = False

    for it in range(1, max_iter + 1):
        r1, r2 = rng.random((pop_size, dim)), rng.random((pop_size, dim))
        v = np.clip(w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x), -vmax, vmax)
        x = np.clip(x + v, lo, hi)
        fx = f(x); n_evals += pop_size
        mejor = fx < pbest_f
        pbest[mejor], pbest_f[mejor] = x[mejor], fx[mejor]
        i = pbest_f.argmin()
        if pbest_f[i] < gbest_f:
            gbest, gbest_f = pbest[i].copy(), pbest_f[i]
        hx.append(gbest.copy()); hf.append(gbest_f)
        if record_population: pob.append(x.copy())
        if _diversidad(x) < tol:
            convergio = True
            break

    return _resultado("PSO", hx, hf, it, convergio, n_evals,
                      dict(pop_size=pop_size, max_iter=max_iter, w=w, c1=c1, c2=c2,
                           vmax_frac=vmax_frac, tol=tol), pob)

def evolucion_diferencial(f, dim, rng, pop_size=40, max_iter=300, F=0.6, CR=0.9,
                          tol=1e-6, bounds=(-500.0, 500.0), record_population=True):
    """Evolución diferencial DE/rand/1/bin."""
    if pop_size < 4:
        raise ValueError("DE necesita pop_size >= 4")
    lo, hi = bounds
    idx = np.arange(pop_size)
    x = rng.uniform(lo, hi, (pop_size, dim))
    fx = f(x); n_evals = pop_size
    i = fx.argmin()
    hx, hf = [x[i].copy()], [fx[i]]
    pob = [x.copy()] if record_population else None
    convergio = False

    for it in range(1, max_iter + 1):
        # tres índices distintos entre sí y distintos del individuo i
        r = rng.random((pop_size, pop_size)); r[idx, idx] = np.inf
        a, b, c = np.argsort(r, axis=1)[:, :3].T
        mutante = np.clip(x[a] + F * (x[b] - x[c]), lo, hi)
        cruce = rng.random((pop_size, dim)) < CR
        cruce[idx, rng.integers(dim, size=pop_size)] = True      # al menos una variable del mutante
        trial = np.where(cruce, mutante, x)
        ft = f(trial); n_evals += pop_size
        mejor = ft <= fx                                          # selección uno a uno
        x[mejor], fx[mejor] = trial[mejor], ft[mejor]
        i = fx.argmin()
        hx.append(x[i].copy()); hf.append(fx[i])
        if record_population: pob.append(x.copy())
        if _diversidad(x) < tol:
            convergio = True
            break

    return _resultado("DE", hx, hf, it, convergio, n_evals,
                      dict(pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, tol=tol), pob)

def algoritmo_evolutivo(f, dim, rng, pop_size=40, max_iter=300, pc=0.9, pm=0.2,
                        sigma_frac=0.1, elite=2, torneo_k=3, alpha=0.5, tol=1e-6,
                        bounds=(-500.0, 500.0), record_population=True):
    """Algoritmo genético de codificación real: torneo, cruce BLX-alpha,
    mutación gaussiana decreciente y elitismo."""
    lo, hi = bounds
    x = rng.uniform(lo, hi, (pop_size, dim))
    fx = f(x); n_evals = pop_size
    i = fx.argmin()
    hx, hf = [x[i].copy()], [fx[i]]
    pob = [x.copy()] if record_population else None
    convergio = False

    for it in range(1, max_iter + 1):
        sigma = sigma_frac * (hi - lo) * (1 - it / max_iter) + 1e-4     # mutación que decrece
        orden = np.argsort(fx)
        elites, elites_f = x[orden[:elite]].copy(), fx[orden[:elite]].copy()

        cand = rng.integers(pop_size, size=(pop_size, torneo_k))         # selección por torneo
        padres = x[cand[np.arange(pop_size), fx[cand].argmin(axis=1)]]

        pareja = padres[rng.permutation(pop_size)]                       # cruce BLX-alpha
        d = np.abs(padres - pareja)
        hijos = rng.uniform(np.minimum(padres, pareja) - alpha * d,
                            np.maximum(padres, pareja) + alpha * d)
        hijos = np.where((rng.random(pop_size) < pc)[:, None], hijos, padres)

        mut = rng.random((pop_size, dim)) < pm                           # mutación gaussiana
        hijos = np.clip(hijos + mut * rng.normal(0, sigma, (pop_size, dim)), lo, hi)

        nuevos = hijos[elite:]                                           # los élites no se re-evalúan
        f_nuevos = f(nuevos); n_evals += len(nuevos)
        x = np.vstack([elites, nuevos]); fx = np.concatenate([elites_f, f_nuevos])
        i = fx.argmin()
        hx.append(x[i].copy()); hf.append(fx[i])
        if record_population: pob.append(x.copy())
        if _diversidad(x) < tol:
            convergio = True
            break

    return _resultado("EA", hx, hf, it, convergio, n_evals,
                      dict(pop_size=pop_size, max_iter=max_iter, pc=pc, pm=pm,
                           sigma_frac=sigma_frac, elite=elite, torneo_k=torneo_k,
                           alpha=alpha, tol=tol), pob) 
