# RNA-Trabajo1-Optimizacion-Metaheuristica-Combinatoria

## Parte 1: Optimización Numérica (Schwefel y Rosenbrock)

Optimización de funciones de prueba (**Schwefel** y **Rosenbrock**) con cuatro algoritmos:
descenso por gradiente (`gd`), algoritmo evolutivo (`ea`), enjambre de partículas (`pso`) y evolución diferencial (`de`).

## Instalación

**Requisitos:** Python 3.10 o superior (probado con 3.11) y estas librerías (ver [requirements.txt](requirements.txt)):

| Librería   | Para qué se usa                      | Versión probada |
|------------|--------------------------------------|-----------------|
| numpy      | cálculo numérico y optimizadores     | 2.4.6           |
| matplotlib | gráficas y animaciones               | 3.11.2          |
| pillow     | escritura de los GIF                 | 12.3.0          |

No se necesita GPU ni descargar datos: todo corre en CPU en segundos (los GIF tardan 1-2 min).

```bash
# 1. Crear y activar el entorno virtual (desde la raíz del repo)
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/CMD)
# source .venv/bin/activate     # Linux / macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrar a la carpeta del proyecto y ejecutar
cd Parte_1/optimizacion_numerica
python run.py --algo pso
```

**Salidas:** los resúmenes (`*_summary.json`) y corridas guardadas van a `Parte_1/optimizacion_numerica/results/`; los GIF a `Parte_1/optimizacion_numerica/gifs/`.

## Guía del CLI (`run.py`)

Estructura general: `python run.py [--problem P] --algo A [opciones]`

- `--problem schwefel|rosenbrock` (por defecto `schwefel`). Rosenbrock exige `--dim >= 2`.
- `--algo gd ea pso de` (uno, varios, o `all`).
- Sin `--experiment`/`--compare` hace **1 corrida**; con ellos hace **≥ 30 corridas** por algoritmo y muestra una tabla comparativa.

### Corrida individual

```bash
python run.py --algo pso                               # Schwefel, 2D
python run.py --algo de --dim 3                        # Schwefel, 3D
python run.py --problem rosenbrock --algo pso          # Rosenbrock, 2D
python run.py --problem rosenbrock --algo de --dim 3   # Rosenbrock, 3D
python run.py --algo ea --seed 7                       # otra semilla
```

### GIF (solo 2D, corrida individual)

```bash
python run.py --algo pso --gif
python run.py --problem rosenbrock --algo gd --gif
python run.py --algo de --gif --gif-dir mis_gifs       # carpeta de salida
```

### Experimentos y comparación (mínimo 30 corridas)

```bash
python run.py --algo pso --experiment                  # 30 corridas de PSO
python run.py --algo pso de --experiment --runs 50     # compara PSO y DE
python run.py --compare                                # los 4 algoritmos, Schwefel 2D
python run.py --compare --dim 3                        # los 4 algoritmos, 3D
python run.py --problem rosenbrock --compare           # los 4 en Rosenbrock 2D
python run.py --problem rosenbrock --compare --dim 3 --runs 50
python run.py --compare --save-runs                    # guarda el historial de cada corrida (.npz)
python run.py --compare --tag prueba1                  # nombre de los archivos de salida
```

### Hiperparámetros por optimizador

Los valores por defecto están en `configs/CLI_defaults_<problema>.json`; cualquier flag de consola los sobrescribe.

| Algoritmo | Flags                                                        |
|-----------|--------------------------------------------------------------|
| `gd`      | `--learning-rate`, `--tol`                                   |
| `ea`      | `--pop-size`, `--patience`, `--mutation-rate`, `--mutation-scale`          |
| `pso`     | `--pop-size`, `--patience`, `--w`, `--c1`, `--c2`                          |
| `de`      | `--pop-size`, `--patience`, `--F`, `--CR`                                  |
| todos     | `--max-iter`, `--dim`, `--seed`, `--bounds LO HI`, `--threshold` (umbral de éxito), `--runs` |

```bash
# Descenso por gradiente
python run.py --algo gd --learning-rate 0.05 --max-iter 3000
python run.py --problem rosenbrock --algo gd --learning-rate 1e-4 --max-iter 20000   # GD necesita lr pequeño aquí

# Algoritmo evolutivo
python run.py --algo ea --pop-size 80 --mutation-rate 0.2 --mutation-scale 0.05

# PSO
python run.py --algo pso --pop-size 60 --w 0.6 --c1 1.8 --c2 1.2

# Evolución diferencial
python run.py --algo de --F 0.5 --CR 0.8 --pop-size 60
python run.py --problem rosenbrock --algo de --dim 3 --F 0.7 --CR 0.9 --max-iter 3000

# Hiperparámetros dentro de una comparación (afectan solo al algoritmo que los tiene)
python run.py --compare --pop-size 60 --max-iter 3000 --threshold 1e-2

# Otro archivo de configuración
python run.py --algo pso --config configs/CLI_defaults_rosenbrock.json
```

### Notas

- **Parada por estancamiento:** `ea`, `pso` y `de` se detienen si el mejor valor no mejora en `--patience` iteraciones (Schwefel: 50 por defecto; Rosenbrock: desactivado. `--patience N` lo cambia y `--patience 0` lo desactiva). `gd` no la usa.
- La tabla de `--compare` reporta mejor/media/desv. de f, tasa de éxito (f ≤ umbral), iteraciones, evaluaciones de f y de gradiente, y evaluaciones equivalentes (1 gradiente = `2·dim` evaluaciones de f; cámbialo con `--grad-cost`).
- Con las mismas semillas los resultados son reproducibles (`--seed`, por defecto 42).
- En 3D o más suba `--max-iter` y/o `--pop-size` si la tasa de éxito baja.
- Detalles de las funciones y del diseño: [Parte_1/optimizacion_numerica/Documentacion.md](Parte_1/optimizacion_numerica/Documentacion.md).

## Parte 2: Optimización Combinatoria (TSP en España Peninsular)

Resolución del Problema del Viajante de Comercio Asimétrico (**ATSP**) sobre las 47 capitales peninsulares usando dos metaheurísticas bioinspiradas: Colonia de Hormigas (`ACO`) y Algoritmo Genético (`GA`).
El modelo optimiza el costo económico total considerando:
- **Combustible:** Toyota Corolla 1995 E100 (consumo mixto 7.3 L/100 km).
- **Peajes:** Tarifas oficiales de autopistas en España.
- **Tiempo del vendedor:** Coste horario laboral (referencia INE: 25.0 €/h).

Los datos y matrices están en `Parte_2/Optimización combinatoria/Informacion_Preparada/data/`.



### Opciones del CLI (`run.py`)

Estructura general: `python run.py [--algo {aco,ga,both}] [--horas H1 H2...] [opciones]`

| Flag | Descripción | Por defecto |
|---|---|---|
| `--demo` | Demostración en vivo en segundos (tarifa 25.0 €/h) | Desactivado |
| `--algo` | Algoritmo(s) a ejecutar: `aco`, `ga`, o `both` | `both` |
| `--horas` | Tarifas horarias del vendedor en €/h (ej. `--horas 0 15 25 35 60`) | `[0.0, 15.0, 25.0, 35.0, 60.0]` |
| `--pop-size` | Tamaño de población / número de hormigas | 60 |
| `--max-iter` | Iteraciones máximas | 150 |
| `--alpha` | [ACO] Peso de feromona | 1.0 |
| `--beta` | [ACO] Peso de visibilidad heurística | 2.0 |
| `--rho` | [ACO] Tasa de evaporación de feromonas | 0.1 |
| `--prob-mut` | [GA] Probabilidad de mutación genética | 0.1 |
| `--seed` | Semilla aleatoria reproducible | 42 |
| `--gif` | Genera animaciones GIF y mosaico sobre la red de carreteras | Desactivado |
| `--config` | Ruta a archivo JSON con parámetros personalizados | `configs/CLI_defaults_tsp.json` |
| `--generar-matrices` | Regenera matrices base en data/ | Desactivado |
| `--generar-instancias` | Regenera archivos .tsp en instancias_tsp/ | Desactivado |

```bash
# Ejemplos de personalización:
python run.py --algo aco --horas 25.0 --pop-size 80 --max-iter 200
python run.py --algo ga --prob-mut 0.15 --seed 123
python run.py --config configs/demo.json
```

**Salidas:** Resultados numéricos (`.json`, `.csv`) en `Parte_2/Optimización combinatoria/Informacion_Preparada/datos_graficas/`; animaciones GIF y figuras en `Parte_2/Optimización combinatoria/GIFS/`.

Detalles del modelo y diseño: [Parte_2/Optimización combinatoria/Documentacion.md](Parte_2/Optimización%20combinatoria/Documentacion.md).

# Guía inicial de GitFlow Simplificado

> **Spoiler:** El único comando que uso es:
> ```bash
> git merge develop
> ```

---

## Paso 1: Crear tu rama de trabajo

Al clonar el repositorio, por defecto estarás en la rama `main`. Debes crear tu rama propia **a partir de `develop`** (puedes usar las ayudas visuales de la extensión de Git en VS Code).

* **Resultado:** Tendrás una rama propia (`tu-nombre`) basada en `develop`.

---

## Paso 2: Guardar tus cambios locales

Mientras trabajas en tu rama, guarda el avance haciendo un `commit` estándar hacia tu rama personal.

---

## Paso 3: Subir tu tarea y notificar al equipo

Cuando completes tu tarea o ítem:
1. Haz un **Pull Request (PR)** dirigido a la rama `develop`.
2. Notifica a tus compañeros para que sincronicen su rama local con el nuevo `develop`.

> ⚠️ **Aclaración importante para los PR:**
> En GitHub, tras hacer *commit* y *push* en tu rama personal, aparecerá un recuadro naranja con el botón **"Compare & pull request"**.
> 
> Al hacer clic, **asegúrate de verificar las ramas de destino:**
> * **`base`:** `develop`
> * **`compare`:** `tu-nombre`

---

## ¿Cómo actualizar tu rama con los cambios de `develop`?

Cuando un compañero notifique que subió cambios a `develop`:

1. Ve a la rama `develop` local en VS Code.
2. Haz un **Fetch / Pull** para traer los cambios remotos.
3. Cambia nuevamente a tu rama personal (`tu-nombre`).
4. Ejecuta en la terminal:
   ```bash
   git merge develop
   ```
* **Resultado:** Tu rama quedará completamente sincronizada con `develop`.

---

## ¿Qué pasa cuando actualizan `develop` y tengo cambios sin guardar?

Si alguien actualizó `develop`, tienes cambios en una versión antigua y **no los quieres perder**, puedes usar el **Stash** para guardar una especie de *checkpoint*.

### Pasos en VS Code:
1. En el menú lateral izquierdo de Git (Control de código fuente), haz clic en los **tres puntos (`...`)**.
2. Ve a **Stash** $\rightarrow$ **Stash (Include Untracked)** y asígnale un nombre a tu trabajo.
3. Actualiza tu rama `develop` local (como se indicó en la sección anterior).
4. Cambia a tu rama personal y ejecuta `git merge develop`.
5. Ve nuevamente a los tres puntos (`...`) de la interfaz gráfica de Git:
   * Busca **Stash** $\rightarrow$ **Apply Latest Stash**.
6. Si hay **conflictos**, resuélvelos decidiendo qué cambios entrantes (*Incoming*) mantienes o cuáles de tus cambios (*Current*) prefieres conservar.

### Al terminar el trabajo
Se actualiza main con la version final de develop, se sigue este flujo para poder mantener un correcto manejo de versiones.
