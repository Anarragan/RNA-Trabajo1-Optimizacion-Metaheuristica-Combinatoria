# Documentación - Parte 2: Optimización Combinatoria (TSP en España Peninsular)

Este módulo aborda la optimización del **Problema del Viajante de Comercio Asimétrico (ATSP)** para recorrer las 47 capitales de provincia de España peninsular, modelando costos reales de transporte, peajes y tiempo laboral.

---

## 1. Planteamiento del Problema

El objetivo es encontrar un circuito cerrado (ciclo hamiltoniano) que visite cada una de las 47 capitales peninsulares exactamente una vez y regrese a la ciudad de origen con el **mínimo costo monetario total**.

### ¿Por qué ATSP (Asimétrico)?
En redes de transporte reales, la distancia y el tiempo de viaje entre dos puntos $A \to B$ y $B \to A$ **no son idénticos**:
- Sentidos únicos de circulación urbana y enlaces viales unidireccionales.
- Circunvalaciones y trazados de autovía que obligan a desvíos asimétricos según el sentido.
- Tarifas de peaje y tiempos de paso que pueden diferir por peajes de tramo unidireccional.

Por tanto, el problema se modela estrictamente como un **ATSP** con matrices completas explícitas de dimensión 47 en formato **TSPLIB95** (`instancias_tsp/*.tsp`).

---

## 2. Modelo Económico y Vehículo de Referencia

### Vehículo seleccionado
- **Modelo:** Toyota Corolla Berlina 1995 (E100), motor 1.6 Si (114 CV).
- **Consumo combinado:** $7.3\text{ L} / 100\text{ km}$ (gasolina regular).
- **Fuente de consumo:** [Auto-Data.net (Toyota Corolla VII E100)](https://www.auto-data.net/es/toyota-corolla-vii-e100-1.6-si-114hp-3351).

### Función de Costo Total
El costo monetario total de un recorrido se compone de tres pilares:

$$\text{Costo Total} = C_{\text{combustible}} + C_{\text{peajes}} + (T_{\text{horas}} \times \text{Valor\_Hora})$$

1. **Costo de Combustible ($C_{\text{combustible}}$):**
   $$C_{\text{combustible}} = \text{Distancia (km)} \times \left(\frac{7.3}{100}\right) \times \text{Precio Medio Gasolina (€/L)}$$
   Precios de combustible tomados de las estadísticas oficiales del Geoportal del Ministerio para la Transición Ecológica y el Reto Demográfico.

2. **Costo de Peajes ($C_{\text{peajes}}$):**
   Tarifas oficiales de las autopistas de peaje peninsulares recopiladas del catálogo de la Red de Carreteras del Estado (MITMA) y concesionarias viales.

3. **Costo del Tiempo del Vendedor ($T_{\text{horas}} \times \text{Valor\_Hora}$):**
   Cuantifica el coste laboral directo del conductor (salario, seguridad social y dietas) más el costo de oportunidad comercial (ventas no ejecutadas durante el traslado):
   - **Tarifa base:** $25.00\text{ €/h}$, estimacion de el rango resltante en la Encuesta Trimestral de Coste Laboral (ETCL) del INE para el sector comercial.
   - **Análisis paramétrico:** Se evalúan cinco escenarios: $0.0$, $15.0$, $25.0$, $35.0$ y $60.0\text{ €/h}$.

### Impacto del Valor de la Hora en la Solución
- **A $0.0\text{ €/h}$:** El tiempo no cuesta. La metaheurística prioriza minimizar la distancia kilométrica y evitar totalmente los peajes, aun si las carreteras convencionales aumentan las horas al volante.
- **A valores altos ($35.0$ y $60.0\text{ €/h}$):** El tiempo domina la función objetivo. Resulta rentable pagar peajes o asumir más kilómetros si se transita por autopistas rápidas que reduzcan las horas de conducción.

---

## 3. Datos y Reproducibilidad Offline

Todos los datos están congelados en `Parte_2/Optimización combinatoria/Informacion_Preparada/data/`:

| Archivo CSV | Descripción |
|---|---|
| `ciudades_espana.csv` | Coordenadas $(x,y) = (\text{lon}, \text{lat})$ y nombres de las 47 capitales peninsulares. |
| `distancias_espana_km.csv` | Matriz $47 \times 47$ de distancias reales de conducción en carretera (km). |
| `tiempos_espana_h.csv` | Matriz $47 \times 47$ de tiempos reales de viaje (horas). |
| `combustible_espana_eur.csv` | Matriz $47 \times 47$ con el costo en combustible (€) según el Toyota Corolla. |
| `peajes_espana_eur.csv` | Matriz $47 \times 47$ con los costos de peajes de autopista (€). |
| `costos_totales_espana_eur.csv` | Matriz $47 \times 47$ consolidada para el valor de hora por defecto (25.0 €/h). |
| `260306-catalogorce2025.zip` | Shapefile oficial del Catálogo de la Red de Carreteras del Estado (MITMA) para renderizado geográfico con GeoPandas. |

---

## 4. Metaheurísticas Implementadas (`algoritmos.py`)

Se comparan dos metaheurísticas bioinspiradas aplicadas al TSP:

### A. Colonia de Hormigas (ACO - Ant Colony Optimization)
- **Implementación:** `sko.ACA.ACA_TSP`.
- **Regla de transición probabilística:** La hormiga elige la siguiente ciudad $j$ desde $i$ proporcional a $[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta$, donde $\tau$ es el rastro de feromona y $\eta = 1 / c_{ij}$ es la visibilidad heurística.
- **Evaporación de feromonas:** $\rho$ controla el decaimiento de las feromonas entre iteraciones para evitar estancamiento prematuro.
- **Comportamiento:** Logra rutas finales de menor costo global gracias a la memoria colectiva depositada en las aristas del grafo.

### B. Algoritmo Genético (GA - Genetic Algorithm)
- **Implementación:** `sko.GA.GA_TSP`.
- **Representación:** Cromosomas de permutación única de 47 genes.
- **Operadores:** Cruce especializado para permutaciones sin ciudades repetidas y mutación con probabilidad $p_{\text{mut}}$.
- **Comportamiento:** Cómputo significativamente más rápido por iteración (~0.9 s vs ~16 s de ACO), pero con mayor probabilidad de quedar atrapado en mínimos locales.

---

## 5. Guía del CLI (`run.py`)

La ejecución de experimentos se realiza con un único comando mediante `run.py`.

### Parámetros Disponibles

| Parámetro | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--demo` | Flag | `False` | Corrida rápida en vivo (< 30s) con valor horario INE (25.0 €/h). |
| `--algo` | `aco`, `ga`, `both` | `both` | Algoritmo(s) a ejecutar. |
| `--horas` | Lista de floats | `[0.0, 15.0, 25.0, 35.0, 60.0]` | Valores horarios del vendedor (€/h). |
| `--pop-size` | Entero | `60` | Tamaño de población / número de hormigas. |
| `--max-iter` | Entero | `150` | Máximo número de iteraciones. |
| `--alpha` | Float | `1.0` | [ACO] Exponente de feromona. |
| `--beta` | Float | `2.0` | [ACO] Exponente de visibilidad heurística. |
| `--rho` | Float | `0.1` | [ACO] Tasa de evaporación. |
| `--prob-mut` | Float | `0.1` | [GA] Probabilidad de mutación. |
| `--seed` | Entero | `42` | Semilla aleatoria fija. |
| `--gif` | Flag | `False` | Genera los GIFs de evolución y mosaicos comparativos en `GIFS/`. |
| `--config` | Ruta | `configs/CLI_defaults_tsp.json` | Carga de parámetros desde archivo JSON. |
| `--generar-matrices` | Flag | `False` | Regenera las matrices base en `Informacion_Preparada/data/`. |
| `--generar-instancias` | Flag | `False` | Regenera las instancias `.tsp` en `Informacion_Preparada/instancias_tsp/`. |

### Ejemplos de Ejecución

```bash
# 1. Demostración en vivo frente al profesor (< 30 segundos)
python run.py --demo

# 2. Demostración con generación de GIF animado sobre la red de carreteras
python run.py --demo --gif

# 3. Corrida completa estándar (ACO y GA para las 5 tarifas horarias)
python run.py

# 4. Ejecutar únicamente Colonia de Hormigas
python run.py --algo aco

# 5. Modificar hiperparámetros desde la consola
python run.py --horas 0.0 25.0 60.0 --pop-size 80 --max-iter 200 --alpha 1.2 --beta 2.5

# 6. Uso de archivo de configuración externo
python run.py --config configs/demo.json
```

---

## 6. Bibliografía y Fuentes Oficiales

1. **Ministerio de Transportes y Movilidad Sostenible (MITMA):**
   *Red de Carreteras del Estado (RCE): Archivos de Geometrías y Catálogo 2025.* [Enlace oficial](https://www.transportes.gob.es/carreteras/catalogo-y-evolucion-de-la-red-de-carreteras/archivos-geometrias-rce)
2. **Instituto Geográfico Nacional (IGN):**
   *Base Topográfica Nacional (Red de Transportes).* [Centro de Descargas CNIG](https://centrodedescargas.cnig.es/CentroDescargas/index.jsp)
3. **Instituto Nacional de Estadística (INE):**
   *Encuesta Trimestral de Coste Laboral (ETCL): Coste laboral por hora efectiva (CNAE-09 Sección G: Comercio).* [Enlace INE](https://www.ine.es/dynt3/inebase/es/index.htm?padre=457&capsel=458)
4. **Ministerio para la Transición Ecológica y el Reto Demográfico:**
   *Geoportal de Gasolineras: Precios oficiales de combustibles en España.* [Enlace](https://geoportalgasolineras.es)
5. **Wardman, M., Chintakayala, V. P., & de Jong, G. (2016):**
   *Values of European travel time savings: A meta-analysis.* Transportation Research Part A: Policy and Practice, 94, 213-228.
