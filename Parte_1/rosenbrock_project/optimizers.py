# Importamos NumPy para trabajr con   arreglos y operaciones matemáticas
import numpy as np

# Importamos la función de Rosenbrock desde nuestro archivo
from rosenbrock import rosenbrock

# Definimos la función de descenso por gradiente
def descenso_gradiente(f, x0, grad_fn=None, learning_rate=0.001, max_iter=1000, tol=1e-6, bounds=(-2.0,2.0), record_every=1):
    """
    Descenso por gradiente

    si grad_fn es None, usa diferencias centrales(gradiente numérico)
    El parámetro reord_every indica cada cuántas iteraciones se guarda en el historial (para no saturar el Gif)

    """
    # Copiamos el punto de partida como arreglo de flotantes
    x = np.asarray(x0, dtype=float) 

    # Guardamos el número de dimensiones del problema
    n = x.size 

    # Desempaquetamos los lïmites del espacio de búsqueda
    lo, hi = bounds

    # Guardamos posición inicial y valor inicial
    hx, hf, n_evals = [x.copy()], [f(x)], 1

    n_evals_grad = 0

    # Bandera: no hemos convergido
    convergio = False

    # Bucle principal: iteramos hasta max_iter veces
    for it in range(1, max_iter + 1):

        # Si no nos pasaron gradiente analítico
        if grad_fn is None:

            # Gradiente numérico por diferencias centrales

            # Creamos un arreglo de ceros para el gradiente
            grad = np.zeros(n)

            # Recorremos cada dimensión
            for i in range(n):

                # Copiamos x para evaluar a la derecha y a la izquierda
                xp, xm = x.copy(), x.copy()

                # Sumam0s un pequeño incremento a la dimensión i
                xp[i] += 1e-5

                # Sumamos un pequeño incremento a la dimensión i
                xm[i] -= 1e-5

                # Aproximación de la derivada parcial
                grad[i] = (f(xp) - f(xm)) / 2e-5

                # Sumamos 2 evaluaciones de f (una por xp y otra por xm)
                n_evals += 2
            
            #contamos 1 evaluación del gradiente(númerico)
            n_evals_grad += 1

        # Si nos pasaron gradiente analítico
        else:
            # Calculamos el gradiente con la funcion analítica
            grad = grad_fn(x)

            # contamos 1 evaluación del gradiente (ánalitico)
            n_evals_grad += 1

        # Actualizamos posición y recortamos al rango
        x_new = np.clip(x - learning_rate * grad, lo, hi)

        # Verificamos si el cambio es pequeño
        convergio = np.linalg.norm(x_new - x) < tol

        # Actualizamos x con la nueva posición
        x = x_new

        # ¿ Guardamos esta iteración?
        if it % record_every == 0 or convergio  or it == max_iter:

            # Guardamos la posición actual en el historial
            hx.append(x.copy())

            # Guardamos el valor de f en la posición actual
            hf.append(f(x))

            # Sumamos 1 al contador de evaluaciones de f
            n_evals += 1

        # Si ya convergimos
        if convergio:
            # Salimos del bucle
            break

        
    
    # Devolvemos un diccionario con todos los resultados
    return {
            
        # Posición final  donde terminó el algoritmo
        "x_final": x,

        # Valor de f en la posición final
        "f_final": hf[-1],

        # Historial de posiciones  (para el GIF)
        "historial_x": np.array(hx),

        # Historial de valores de f 
        "historial_f": np.array(hf),

        # Número total de iteraciones realizadas
        "iteraciones": it,

        #¿Convergió por tolerancia?
        "convergio":bool(convergio),

        # Número total de evaluaciones de f 
        "n_evals_f": n_evals, 

        "n_evals_grad": n_evals_grad,

        }



# Calcula qué tan dispersa está la población
def _diversidad(x):

    """
    Dispersión media de la población (desv. estándar por variable, promediada).
   """
    # Desviación estándar media
    return float(np.mean(np.std(x, axis=0)))


# Empaqueta resultados
def _resultado(nombre, hx, hf, it, convergio, n_evals, meta, pob=None):

# Creamos el diccionario de salida
    out = {

    # Mejor posición final encontrada
    "x_final": hx[-1].copy(),

    # Mejor valor de f encontrado
    "f_final": hf[-1],

    # Historial del mejor individuo por iteración
    "historial_x": np.array(hx),

    # Historial del mejor valor de f por iteración
    "historial_f": np.array(hf),

    # Número total de iteraciones
    "iteraciones": it,

    # ¿Convergió el algoritmo?
    "convergio": bool(convergio),

    # Número total de evaluaciones de f
    "n_evals_f": n_evals,

     # Los metaheurísticos no usan gradiente
    "n_evals_grad": 0,

    # Info adicional del algoritmo
    "metadata": {"algoritmo": nombre, **meta},

    }


    # Si nos pasaron historial de población
    if pob is not None:

        # Guardamos el historial de la población
        out["historial_poblacion"] = np.array(pob)
        

    # Devolvemos el diccionario de resultados
    return out


#PSO: firma parte 1
def pso(f, dim, rng, pop_size=40, max_iter=300, w=0.7298, c1=1.49618, c2=1.49618, vmax_frac = 0.2, tol=1e-6, bounds=(-2.0,2.0), record_population=True):
    
    # Desempaquetamos los límites del espacio de búsqueda
    lo, hi = bounds

    # Velocidad máxima permitida
    vmax = vmax_frac * (hi - lo)

    # Posiciones iniciales aleatorias de las particulas 
    x = rng.uniform(lo, hi, size=(pop_size, dim))

    # Velocidades iniciales aleatorias
    v = rng.uniform(-vmax, vmax, size=(pop_size, dim))

    # Evaluamos f en todas las partículas iniciales
    fx = f(x)

    # Contador de evaluaciones de f (ya evaluamos todas las particulas)
    n_evals = pop_size

    # mejor posición y valor personal de cada partícula
    pbest, pbest_f = x.copy(), fx.copy()

    # índice de la particula con el mejor valor personal
    i = pbest_f.argmin()

    # mejor posición y valor global
    gbest, gbest_f = pbest[i].copy(), pbest_f[i]

    # Historial de la mejor global (para el GIF)
    hx, hf = [gbest.copy()], [gbest_f]

    # Historial de poblacion
    pob = [x.copy()] if record_population else None

    # Bander de convergencia
    convergio = False

    # bucle pricipaal de PSO
    for it in range(1, max_iter + 1):

        # Número aleatorios para el componente cognitivo
        r1 = rng.random((pop_size, dim))

        # Números aleatorios para el componente social
        r2 = rng.random((pop_size, dim))

        # Actualizamos velocidades
        v = np.clip(w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x), -vmax, vmax)

        # Actualizamos posiciones y recortamos el rango
        x = np.clip(x + v, lo, hi)

        # evaluamos f en las nuevas posiciones de las particulas
        fx = f(x)

        # Sumamos las evaluaciones de esta iteración
        n_evals += pop_size

        # Mascara booleana: True donde la particula mejoró 
        mejoras = fx < pbest_f

        # Actualizamos récords personales  
        pbest[mejoras], pbest_f[mejoras] = x[mejoras], fx[mejoras]

        # indice de la mejor particula según su récord personal
        i = pbest_f.argmin()

        # ¿Hay un nuevo mejor global?
        if pbest_f[i] < gbest_f:

            # Actualizamos el mejor global
            gbest, gbest_f = pbest[i].copy(), pbest_f[i]

         # Guardamos la mejor posición global actual en el historial
        hx.append(gbest.copy())

         # Guardamos el mejor valor global actual en el historial
        hf.append(gbest_f)

         # si estamos guardadno la poblacion completa
        if record_population:

            # Guardamos una copia de la poblacion 
            pob.append(x.copy())
        
        # ¿La población está suficientemente concentrada?
        if _diversidad(x) < tol:

            # Marcamos que el algoritmo convergió
            convergio = True

            # Salimos del bucle
            break

    # Empaquetamos los resultados de PSO
    return _resultado("PSO", hx, hf, it, convergio, n_evals,
                    dict(pop_size = pop_size, max_iter=max_iter, w=w, c1=c1, c2=c2,
                    vmax_frac=vmax_frac, tol=tol), pob)

# DE: firma parte 1
def evolucion_diferencial(f, dim, rng, pop_size=40, max_iter=300, F=0.6, CR=0.9, tol=1e-6, bounds=(-2.0,2.0), record_population= True):

    # DE necesita al menos 4 individuos
    if pop_size < 4:

        # Lanza un error
        raise ValueError("DE necesita pop_size >=4 ")

    # Desempaquetamos los limites del espacio de búsqueda
    lo, hi = bounds

    # índices de los individuos: 0, 1, 2, . . ., po_size-1
    idx = np.arange(pop_size)

    # Población inicial aleatoria
    x = rng.uniform(lo, hi, (pop_size, dim))

    #Evaluamos f en toda la población inicial
    fx = f(x)

    # Contador de evaluaciones de f
    n_evals = pop_size

    # índice del mejor individuo inicial
    i = fx.argmin()

    # Historial del mejor individuo
    hx, hf = [x[i].copy()], [fx[i]]

    # Historial de población (opcional)
    pob = [x.copy()] if record_population else None

    # Bandera de convergencia
    convergio = False

    # Bucle principal de DE
    for it in range(1, max_iter +1):

         #Matriz de Números aleatorios
        r = rng.random((pop_size, pop_size ))

        # En la diagonal ponemos infinito para no elegir al mismo individuo
        r[idx, idx] = np.inf

        # Elegimos 3 Índices distintos por individuo
        a, b, c = np.argsort(r, axis=1)[:, :3].T 

        # Vector mutante
        mutante = np.clip(x[a] + F*(x[b] - x[c]), lo, hi)

        # Máscara booleana de cruce
        cruce = rng.random((pop_size, dim)) < CR

        # Garantizamos al menos 1 variable del mutante
        cruce[idx, rng.integers(dim, size=pop_size)] = True

        # Individuo prueba: mezcla de mutante y padre
        trial = np.where(cruce, mutante, x)

        # Evaluamos f en todos los individuos 
        ft = f(trial)

        # Sumamos las evaluaciones de f en esta iteracion
        n_evals += pop_size

        # Máscara: True donde el trial es mejor o igual que el padre
        mejor = ft <= fx

        # Reemplazamos padres por trials mejores
        x[mejor], fx[mejor] = trial[mejor], ft[mejor]

        # índice del mejor individuo actual
        i = fx.argmin()

        # Guardamos la mejor posición actual en el historial
        hx.append(x[i].copy())

        # Guardamos el mejor valor actual en el historial
        hf.append(fx[i])

        # Si estamos guardando la población
        if record_population: 

            # Guardamos la población inicial
            pob.append(x.copy())
                
        # ¿ La población ya convergió?
        if _diversidad(x) < tol:   

            # Marcamos que el algoritmo convergió
            convergio = True
                
            # Salimos del bucle
            break               
 
 # Empaquetamos resultados de DE

    return _resultado("DE", hx, hf, it, convergio, n_evals, 
                    dict(pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, tol=tol), pob)

#EA: firma parte 1
def algoritmo_evolutivo(f, dim, rng, pop_size=40, max_iter=300, pc=0.9,
                        mutation_rate=0.2, mutation_scale=0.1, elite=2,
                        torneo_k=3, alpha=0.5, tol=1e-6,
                        bounds=(-2.0, 2.0), record_population=True):

    # Desempaquetamos los límites
    lo, hi = bounds

    # Población inicial aleatoria
    x = rng.uniform(lo, hi, (pop_size, dim))

    # Evaluamos f en toda la población
    fx = f(x)

    # Contador de evaluaciones de f
    n_evals = pop_size

    # Índice del mejor individuo inicial
    i = fx.argmin()

    # Historial del mejor individuo
    hx, hf = [x[i].copy()], [fx[i]]

    # Historial de población (opcional)
    pob = [x.copy()] if record_population else None

    # Bandera de convergencia
    convergio = False

    # Bucle principal del algoritmo evolutivo
    for it in range(1, max_iter + 1):

        # Mutación decreciente (usa mutation_scale)
        sigma = mutation_scale * (hi - lo) * (1 - it/max_iter) + 1e-4

        # Índice de los individuos ordenados por valor de f
        orden = np.argsort(fx)

        # Los mejores pasan intactos
        elites, elites_f = x[orden[:elite]].copy(), fx[orden[:elite]].copy()

        # Candidatos para torneos
        cand = rng.integers(pop_size, size=(pop_size, torneo_k))

        # Selección por torneo
        padres = x[cand[np.arange(pop_size), fx[cand].argmin(axis=1)]]

        # Pareja: padres mezclados aleatoriamente
        pareja = padres[rng.permutation(pop_size)]

        # Distancia entre cada padre y su pareja
        d = np.abs(padres - pareja)

        # Cruce BLX-α (extremo inferior)
        hijos = rng.uniform(np.minimum(padres, pareja) - alpha*d,
                            np.maximum(padres, pareja) + alpha*d)

        # Aplicamos cruce con probabilidad pc
        hijos = np.where((rng.random(pop_size) < pc)[:, None], hijos, padres)

        # Máscara de mutación (usa mutation_rate)
        mut = rng.random((pop_size, dim)) < mutation_rate

        # Mutación gaussiana
        hijos = np.clip(hijos + mut * rng.normal(0, sigma, (pop_size, dim)), lo, hi)

        # Los hijos que no son élites (el primero se descarta)
        nuevos = hijos[elite:]

        # Evaluamos f en los hijos nuevos
        f_nuevos = f(nuevos)

        # Sumamos las evaluaciones de los hijos nuevos
        n_evals += len(nuevos)

        # Nueva población: élites + hijos
        x = np.vstack([elites, nuevos])

        # Valores correspondientes de f
        fx = np.concatenate([elites_f, f_nuevos])

        # Índice del mejor individuo actual
        i = fx.argmin()

        # Guardamos la mejor posición actual en el historial
        hx.append(x[i].copy())

        # Guardamos el mejor valor actual en el historial
        hf.append(fx[i])

        # Si estamos guardando la población
        if record_population:
            pob.append(x.copy())

        # ¿La población ya convergió?
        if _diversidad(x) < tol:
            convergio = True
            break

    # Empaquetamos resultados de EA (con los nombres nuevos)
    return _resultado("EA", hx, hf, it, convergio, n_evals,
                      dict(pop_size=pop_size, max_iter=max_iter, pc=pc,
                           mutation_rate=mutation_rate, mutation_scale=mutation_scale,
                           elite=elite, torneo_k=torneo_k, alpha=alpha, tol=tol), pob)