# Informe técnico: clasificación del nivel de ingreso de los países

**Proyecto:** Fundamentos matemáticos y algorítmicos aplicados a Inteligencia
Artificial en Python — DataExpert.
**Fecha de ejecución:** 2026-10-01 · **Python:** 3.14.7 · **Semilla:** 42.

## Resumen ejecutivo

Se construyó un clasificador de **tres niveles de ingreso** (bajo / medio /
alto) a partir de un panel macroeconómico de **10.512 registros** (220 países,
1970-2021, 26 columnas). El mejor modelo es la **regresión logística
estandarizada**, con **accuracy 0,7694** y **F1 macro 0,7629** en prueba y
**0,7625 ± 0,0106** en validación cruzada de 5 pliegues. El experimento de
dispersión (requisito 10.1) demostró que, sin escalar, la distancia del KNN la
domina una única variable (`Total Value Added`, σ = 954.458.105.484,8854 en
entrenamiento) y que, una vez estandarizadas las numéricas y **sin el bloque
one-hot de `Year`**, el KNN alcanza **0,9301** de accuracy. Las 22 pruebas de
`pytest` pasan y `main.py` genera **11 figuras**.

| Dato | Valor |
|------|-------|
| Dataset | 10.512 filas × 26 columnas, 0 duplicados, 2.374 celdas nulas |
| Objetivo | Terciles de `Per capita GNI`: bajo ≤ 1.042,00 USD < medio ≤ 5.485,67 USD < alto |
| División | 8.409 train / 2.103 test (80/20, estratificado, semilla 42) |
| Modelos | KNN (k=5) y regresión logística, con preprocesamiento en `Pipeline` |
| Modelo principal | Regresión logística (mayor accuracy en validación cruzada) |
| Salidas | 11 figuras PNG + este informe |

---

## 1. Introducción

DataExpert necesita consolidar una base sólida de fundamentos matemáticos para
desarrollar soluciones de Machine Learning. Este proyecto educativo recorre la
cadena completa de un problema real de aprendizaje automático —explorar datos,
analizarlos estadísticamente, resolver operaciones de álgebra lineal a mano,
preprocesar sin fugas, entrenar, evaluar y visualizar— con el doble objetivo
de **entender los fundamentos** y **dejar un programa que se ejecute de
principio a fin sin errores**.

Todo el código está en español (comentarios, docstrings y mensajes), se
organiza en módulos de una sola responsabilidad bajo `src/` y `main.py` se
limita a orquestar las siete etapas con `logging`.

## 2. Descripción del problema

El dataset contiene observaciones país-año de agregados macroeconómicos
(producto, consumo, comercio, población, tipos de cambio…). Se desea predecir
el **nivel de ingreso de un país** a partir de esa información.

Tras la exploración (sección 7) se decidió un problema de **CLASIFICACIÓN
MULTICLASE** con tres clases equilibradas derivadas de `Per capita GNI`
(PNB per cápita, USD corrientes):

| Clase | Condición | Registros |
|-------|-----------|-----------|
| bajo | `Per capita GNI` ≤ 1.042,00 USD | 3.505 (33,3 %) |
| medio | 1.042,00 < `Per capita GNI` ≤ 5.485,67 USD | 3.503 (33,3 %) |
| alto | `Per capita GNI` > 5.485,67 USD | 3.504 (33,3 %) |

**Por qué terciles y no cortes fijos en dólares.** El dataset cubre
1970-2021 en USD nominales: un corte fijo mezclaría épocas (1.042 USD en 1970
era una renta alta; en 2021 no lo es) y, además, los terciles dejan las tres
clases prácticamente idénticas (3.505 / 3.503 / 3.504), mientras que un corte
tipo Banco Mundial habría dado clases más desiguales. Un objetivo equilibrado
permite interpretar el accuracy sin que una clase mayoritaria lo inflée.

## 3. Objetivos del proyecto

1. Leer y explorar un CSV real comprobando estructura, tipos, nulos y duplicados.
2. Calcular media, mediana, varianza y desviación estándar **a mano** y
   contrastarlos con NumPy usando `ddof` explícito.
3. Detectar valores atípicos con puntuación Z y tomar una decisión justificada.
4. Demostrar operaciones de álgebra lineal: vectores, matrices, producto punto,
   norma y sistemas de ecuaciones.
5. Preprocesar los datos **sin fuga de información** y entrenar modelos de
   clasificación con scikit-learn.
6. Medir el **impacto de la dispersión** de las variables en el modelo
   (requisito 10.1).
7. Generar visualizaciones y este informe con valores reales de la ejecución.

## 4. Dataset utilizado

`data/data.csv` (2,6 MB), datos macroeconómicos tipo Banco Mundial. La cabecera
trae espacios al inicio y al final en los nombres de columna, que se recortan
al cargar.

- **10.512 filas × 26 columnas**; 20 `float64`, 4 `int64` y 2 `str`.
- **Unidad de observación:** país-año (panel). 220 países, 52 años (1970-2021),
  153 monedas distintas.
- **2.374 celdas nulas** en 11 columnas (1.993 filas con al menos un nulo,
  18,96 % del total). La peor es `Changes in inventories` con 1.841 nulos
  (17,51 %).
- **0 registros duplicados.**
- `Per capita GNI`: mín. 34 USD, máx. 234.317 USD, media 8.965,56 USD,
  mediana 2.316,50 USD, desviación típica 17.070,21 USD; cuartiles
  730, 2.316,50 y 8.965,75 USD.

## 5. Tecnologías y librerías utilizadas

| Librería | Versión instalada | Uso en el proyecto |
|----------|-------------------|--------------------|
| Python | 3.14.7 | Lenguaje y entorno |
| pandas | 3.0.6 | Lectura, exploración y manipulación |
| numpy | 2.5.3 | Vectores, matrices y estadística |
| scikit-learn | 1.9.1 | Preprocesamiento, modelos y métricas |
| matplotlib | 3.11.2 | Figuras (backend `Agg`, sin pantalla) |
| seaborn | 0.13.2 | Histogramas, heatmap y boxplots |
| pytest | 9.1.1 | Pruebas automáticas |

Todas las dependencias están fijadas en `requirements.txt` (resultado de
`pip freeze` dentro del entorno virtual `.venv`).

## 6. Estructura del proyecto y cómo ejecutarlo

```text
main.py                   # Orquesta las 7 etapas con logging (sin lógica de negocio)
requirements.txt          # Versiones fijadas
README.md                 # Instrucciones de instalación y ejecución
.gitignore                # .venv/, __pycache__/, .atl, etc.
data/data.csv             # Dataset (no se modifica)
src/
  config.py               # Rutas (pathlib), semilla, umbrales, TARGET, exclusiones
  exploracion.py          # Requisito 1
  estadistica.py          # Requisitos 8, 9 y 10
  algebra_lineal.py       # Requisitos 5, 6 y 7
  preprocesamiento.py     # Requisito 2
  modelo.py               # Requisitos 3, 4 y 10.1
  visualizacion.py        # Requisitos 11-14
tests/test_estadistica.py # 22 pruebas con pytest
reports/figures/          # 11 figuras PNG
reports/informe.md        # Este documento
```

Instalación y ejecución (Windows / PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python main.py     # ejecuta las 7 etapas y genera figuras + informe
pytest             # 22 pruebas
```

## 7. Exploración de los datos (requisito 1)

Ejecutada por `src/exploracion.py`. Resultados:

| Comprobación | Resultado |
|--------------|-----------|
| Forma | 10.512 × 26 |
| Nombres de columna | recortados (venían con espacios) |
| Tipos | 20 float64, 4 int64, 2 str |
| Columnas con nulos | 11 (2.374 celdas; 1.993 filas afectadas) |
| Nulos más abundantes | `Changes in inventories` 1.841 (17,51 %), `Agriculture…` 121 (1,15 %) |
| Duplicados | 0 |
| Cardinalidad | `Country` 220, `Currency` 153, `CountryID` 220, `Year` 52 |
| `Per capita GNI` | 34 – 234.317 USD; media 8.965,56; mediana 2.316,50 |

**Decisión sobre el tipo de problema y el objetivo.** El dataset no trae una
columna objetivo: se deriva de `Per capita GNI` (columna categórica de
niveles → clasificación) y se fija en `config.TARGET` **después** de la
exploración, como exige el enunciado.

**Predictores excluidos y por qué** (6 de 26 columnas):

| Columna excluida | Motivo |
|------------------|--------|
| `Per capita GNI` | Fuga directa: es la definición literal del objetivo |
| `Gross National Income(GNI) in USD` | Identidad aritmética: `GNI / población` reproduce el objetivo con correlación 0,9999996 |
| `Gross Domestic Product (GDP)` | Proxy del mismo constructo: `GDP / población` correlaciona 0,991 con el objetivo; con `Population` en el modelo el target sería reconstruible y la evaluación, inútil |
| `Country` | Cardinalidad 220: memorizaría el país en lugar de aprender |
| `Currency` | Cardinalidad 153: mismo problema |
| `CountryID` | Identificador, no una magnitud |

Quedan **19 predictoras numéricas** y **1 categórica (`Year`)**, que se codifica
con one-hot como efectos fijos de año (práctica habitual en econometría de
panel) y que además permite ejemplificar el `OneHotEncoder` exigido.

## 8. Proceso de preprocesamiento (requisito 2)

`src/preprocesamiento.py` aplica el orden correcto, sin fugas:

1. Eliminar duplicados (0 en este dataset).
2. Separar `X` e `y` (objetivo creado por terciles).
3. **Dividir primero**: 8.409 registros de entrenamiento (80 %) y 2.103 de
   prueba (20 %), con `stratify=y` y `random_state=42`.
4. Ajustar (`fit`) los transformadores **solo en entrenamiento**, dentro del
   `Pipeline`, y aplicar `transform` a ambos conjuntos.

| Rama del `ColumnTransformer` | Columnas | Transformaciones |
|------------------------------|----------|------------------|
| `num` | 19 numéricas | `SimpleImputer(strategy="median")` → `StandardScaler()` |
| `cat` | `Year` | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")` |

Distribución de clases tras la división:

| Conjunto | bajo | medio | alto |
|----------|------|-------|------|
| train | 2.804 (33,3 %) | 2.802 (33,3 %) | 2.803 (33,3 %) |
| test | 701 (33,3 %) | 701 (33,3 %) | 701 (33,3 %) |

**Hallazgo de estructura de panel:** el 100,0 % de los registros de prueba
pertenecen a un país que también está en entrenamiento (misma economía
observada en otros años). Se documenta en las limitaciones (sección 21).

## 9. Fundamentos de álgebra lineal aplicados

Un dataset es una matriz `X` de forma (n_muestras, n_características) y un
modelo lineal predice con `X @ w + b`. El módulo `src/algebra_lineal.py`
demuestra cada operación y la relaciona con IA: el producto punto es la
combinación ponderada de características y la norma de los pesos es lo que
penalizan las regularizaciones L2.

## 10. Operaciones con vectores y matrices (requisitos 5 y 6)

Salida real de la etapa 4:

| Operación | Entrada | Resultado |
|-----------|---------|-----------|
| Suma de vectores | `a = [1,2,3]`, `b = [4,5,6]` | `[5, 7, 9]` |
| Resta de vectores | `a - b` | `[-3, -3, -3]` |
| Escalar | `3 * a` | `[3, 6, 9]` |
| Acceso / modificación | `a[1]`, `a[0] = 99` | `2.0` / `[99, 2, 3]` |
| Suma de matrices | `A + B` | `[[6, 8], [10, 12]]` |
| Resta de matrices | `A - B` | `[[-4, -4], [-4, -4]]` |
| **Multiplicación** | `A @ B` | `[[19, 22], [43, 50]]` |
| Producto elemento a elemento | `A * B` | `[[5, 12], [21, 32]]` (¡distinto!) |
| Transpuesta | `A.T` | `[[1, 3], [2, 4]]` |
| Escalar × matriz | `2 * A` | `[[2, 4], [6, 8]]` |

con `A = [[1,2],[3,4]]` y `B = [[5,6],[7,8]]`. Se usa el operador `@` para la
multiplicación matricial: `A * B` multiplica elemento a elemento y confundirlos
es el error más común.

**Producto punto y norma, manual frente a NumPy:**

| Magnitud | Manual | NumPy | `np.isclose` |
|----------|--------|-------|--------------|
| `a · b` | 32,000000 | `np.dot` 32,000000 | True |
| `‖a‖₂` | 3,741657 | `np.linalg.norm` 3,741657 | True |

Las comparaciones se hacen con `np.isclose` (nunca con `==`) por el redondeo
de punto flotante. El producto punto manual valida además que ambos vectores
tengan la misma longitud y la norma que el vector no esté vacío, con mensajes
de error en español.

## 11. Resolución de sistemas de ecuaciones (requisito 7)

Sistema `A x = b` con `2x + y = 5` y `x + 3y = 10`:

| Comprobación | Resultado |
|--------------|-----------|
| ¿Matriz cuadrada? | True |
| Rango (`matrix_rank`) | 2 / 2 → solución única |
| Solución (`np.linalg.solve`) | `x = [1.0, 3.0]` |
| Comprobación `A @ x ≈ b` | True (residuo máximo 0,00e+00) |

El código comprueba la unicidad con `matrix_rank` (más fiable que mirar si el
determinante "es cero") y **captura `LinAlgError`** con un mensaje claro. La
demostración incluye además una matriz singular `[[1,2],[2,4]]`, que produce:
*"La matriz de coeficientes es singular (rango incompleto): el sistema no
tiene solución única."* — el programa continúa sin caerse. Se prefiere `solve`
a `inv(A) @ b` por estabilidad numérica.

## 12. Análisis estadístico (requisito 8)

Se analizan las **24 variables numéricas** con `describe()`, media, mediana y
cuartiles. Indicador de asimetría usado: `|media − mediana| / desviación`
(cuanto mayor, más asimétrica es la distribución).

| Variable | Media | Mediana | Distancia relativa |
|----------|-------|---------|--------------------|
| `Per capita GNI` | 8.965,57 | 2.316,50 | **0,390** |
| `Exports of goods and services` | 46.711.757.655,54 | 2.407.905.656,00 | 0,253 |
| `Imports of goods and services` | 45.915.038.487,03 | 2.891.919.117,00 | 0,238 |
| `Population` | 28.515.225,81 | 5.051.556,00 | 0,206 |
| `Construction (ISIC F)` | 10.021.989.861,23 | 371.770.132,50 | 0,194 |
| `Gross Domestic Product (GDP)` | 182.876.492.569,56 | 8.070.571.563,00 | 0,178 |

Todas las variables monetarias tienen media muy superior a la mediana:
distribuciones fuertemente **asimétricas a la derecha** con cola larga (países
de gran tamaño económico). La mediana es más robusta que la media ante esos
valores extremos, de ahí que se marque siempre en los histogramas.

Cuartiles de `Per capita GNI`: Q25 = 730,00 USD · Q50 = 2.316,50 USD ·
Q75 = 8.965,75 USD.

## 13. Cálculo de varianza y desviación estándar (requisito 9)

Implementadas **a mano** en `src/estadistica.py` (`varianza(datos, muestral)`
y `desviacion_estandar(datos, muestral)`) y contrastadas con NumPy sobre
`Per capita GNI` (10.512 valores):

| Magnitud | Fórmula | Manual | NumPy (`ddof` explícito) | ¿Coincide? |
|----------|---------|--------|--------------------------|------------|
| Varianza **poblacional** | divide entre N | 291.364.209,3792 | `np.var(ddof=0)` → 291.364.209,3792 | True |
| Varianza **muestral** | divide entre n − 1 | 291.391.929,3116 | `np.var(ddof=1)` → 291.391.929,3116 | True |
| Desviación **poblacional** | √varianza poblacional | 17.069,3939 | `np.std(ddof=0)` → 17.069,3939 | True |

**Trampa documentada en el docstring:** NumPy y Pandas no usan el mismo
`ddof` por defecto — `np.var`/`np.std` usan `ddof=0` (poblacional) y
`pd.Series.var/std` usan `ddof=1` (muestral). Comparar sin fijarlo hace parecer
que la implementación manual está mal. Por eso **siempre se pasa `ddof` de
forma explícita**.

**Validaciones** de las funciones manuales: lista vacía → `ValueError`;
varianza muestral con un solo dato → `ValueError` (n − 1 = 0); valores no
numéricos o NaN → `ValueError`; la varianza poblacional de un único dato es 0.
Las 22 pruebas de `tests/test_estadistica.py` verifican todo esto contra NumPy,
incluido el dataset clásico `[2, 4, 4, 4, 5, 5, 7, 9]` (varianza poblacional
exacta 4,0) y valores negativos.

## 14. Identificación de valores atípicos y decisión tomada (requisito 10)

Criterio: puntuación Z con `Z = (x − μ) / σ`, `σ` calculada con `ddof=0`
explícito, umbral `|Z| > 3` (`config.Z_UMBRAL`), aplicado **solo a variables
numéricas** y excluyendo los identificadores `CountryID` y `Year` (parecen
números pero no son magnitudes). Se evaluaron **22 variables**.

| Resultado | Valor |
|-----------|-------|
| Atípicos detectados | **2.572** |
| Variable con más atípicos | `Per capita GNI` (201; z máx. 13,202) |
| Otros | `Exports…` 185 (19,918), `Imports…` 161 (18,590), `IMF based exchange rate` 151 (21,454), `AMA exchange rate` 137 (48,572) |
| Atípico más extremo | `AMA exchange rate`, fila 4316: 111.636,628 (z = 48,572) |

**Decisión: NO se eliminan los atípicos.** Razones:

1. Son valores macroeconómicos legítimos (tipos de cambio de economías con
   hiperinflación, economías de billones), no errores de captura.
2. Eliminarlos de forma automática sesgaría la muestra hacia economías
   "normales" y destruiría información relevante para predecir niveles de
   ingreso.
3. El propio método tiene las limitaciones conocidas: los atípicos inflan la
   media y la desviación con la que se detectan; con pocas observaciones el
   umbral 3 no detecta nada; y funciona mejor con distribuciones simétricas.
4. Los nulos, en cambio, **sí** se tratan (imputación por mediana/moda dentro
   del `Pipeline`, ajustada solo con entrenamiento).

Los atípicos se visualizan en `reports/figures/boxplot_valores_atipicos.png`.

## 15. Impacto de la dispersión en el modelo (requisito 10.1)

Este es el hallazgo más interesante del proyecto. Desviación estándar de las
19 numéricas en entrenamiento (`ddof=0`):

| Extremo | Variable | Desviación estándar |
|---------|----------|---------------------|
| Mayor | `Total Value Added` | 954.458.105.484,8854 |
| Mayor | `Final consumption expenditure` | 744.761.900.259,3745 |
| Mayor | `Household consumption expenditure…` | 596.017.820.694,3943 |
| Menor | `AMA exchange rate` | 1.946,7952 |
| Menor | `IMF based exchange rate` | 1.914,5043 |

La variable más dispersa tiene una desviación estándar unas **498 millones de
veces** mayor que la menos dispersa. Experimento con el MISMO modelo, solo
cambiando el escalado (todas las cifras en prueba):

| Modelo | Sin escalar | Con `StandardScaler` | Escalado solo con numéricas* |
|--------|-------------|----------------------|------------------------------|
| **KNN (k=5)** | acc 0,8597 · F1 0,8589 | acc 0,7323 · F1 0,7275 | **acc 0,9301 · F1 0,9301** |
| **Regresión logística** | acc 0,7418 · F1 0,7365 | acc 0,7694 · F1 0,7629 | acc 0,8112 · F1 0,8112 |

\* Diagnóstico: mismo pipeline escalado pero sin el bloque one-hot de `Year`.

Mejora de F1 al escalar: **KNN −0,1314**; **regresión logística +0,0264**.

**¿Por qué el KNN empeora al escalar si la teoría dice lo contrario?** El
propio programa lo mide con el diagnóstico de vecinos (fracción de la
distancia² que aporta el bloque one-hot y año del vecino más cercano):

| Variante | Distancia² media | Peso del bloque one-hot | Vecinos del mismo año |
|----------|------------------|-------------------------|-----------------------|
| crudo | 9,62 × 10²¹ | 0,0000 | 0,0067 |
| escalado | 1,0579 | 0,1322 | 0,9301 |

Tres hechos encadenados, todos medidos:

1. **Sin escalar, la predicción de la teoría se cumple**: la distancia la
   domina la variable con mayor dispersión (la distancia² media es del orden
   de 10²¹ y el bloque one-hot aporta 0,0000). El KNN crudo funciona razonable
   bien (0,8597): su vecino más cercano cae casi nunca en el mismo año
   (0,67 %), es decir, en otra observación de una economía de estructura
   parecida —y el 100 % de los países de prueba ya aparecen en entrenamiento,
   como se midió en la sección 8.
2. **Al escalar, todas las numéricas pesan lo mismo (eso es lo que hace
   `StandardScaler`)**, pero las 52 columnas one-hot de `Year` NO pasan por el
   escalador y pasan a pesar 0,1322 de la distancia. Ese bloque actúa como
   "puerta de año": el 93,01 % de los vecinos pasa a ser del mismo año y de
   otro país, y el KNN completo baja a 0,7323.
3. **Con el bloque one-hot fuera, escalar mejora a los dos modelos** (KNN
   0,9301 = mejor resultado de todo el proyecto; regresión logística 0,8112).
   La regresión logística, que no usa distancias, ya mejora con escalar en la
   comparación completa (+0,0264 de F1), porque sin estandarizar sus
   gradientes los dominan las variables de escala gigante.

**Conclusión de la sección:** escalar es necesario —las variables están en
escalas que difieren hasta 498 millones de veces— pero un bloque one-hot
grande interactúa con la distancia euclídea y hay que vigilarlo. La mejora
real del escalado se ve comparando 0,7323 contra 0,9301 (mismo modelo, mismos
datos, única diferencia: qué columnas pasan por el escalador).

## 16. Modelo de Machine Learning (requisito 3)

| Decisión | Elección |
|----------|----------|
| Tipo de problema | Clasificación multiclase (3 clases balanceadas) |
| Algoritmos | KNN (k=5, basado en distancias) y regresión logística (basada en optimización) |
| Preprocesamiento | `ColumnTransformer` + `Pipeline` dentro del modelo |
| Codificación del objetivo | Terciles de `Per capita GNI` |
| Selección de características | 19 numéricas + `Year` (one-hot); 6 columnas excluidas (sección 7) |
| Reproducibilidad | `random_state = 42` en la división y en la regresión logística |

Se entrenaron los dos modelos porque tienen sensibilidades opuestas: el KNN
mide distancias ( sensible a la escala y a la métrica) y la regresión
logística optimiza una función de pérdida sobre variables ponderadas.

## 17. Entrenamiento y evaluación (requisito 4)

Métricas de **clasificación** (accuracy, precision, recall y F1 con promedio
macro) más matriz de confusión. No se usan métricas de regresión.

### KNN (k = 5)

| Métrica | Entrenamiento | Prueba |
|---------|---------------|--------|
| Accuracy | 0,8223 | 0,7323 |
| Precision (macro) | 0,8218 | 0,7268 |
| Recall (macro) | 0,8223 | 0,7323 |
| F1 (macro) | 0,8206 | 0,7275 |

Brecha train − test: **+0,0900** → posible sobreajuste.

### Regresión logística

| Métrica | Entrenamiento | Prueba |
|---------|---------------|--------|
| Accuracy | 0,7856 | 0,7694 |
| Precision (macro) | 0,7903 | 0,7736 |
| Recall (macro) | 0,7855 | 0,7694 |
| F1 (macro) | 0,7808 | 0,7629 |

Brecha train − test: **+0,0162** → sin sobreajuste apreciable.

### Validación cruzada (5 pliegues, sobre entrenamiento)

| Modelo | Pliegues | Accuracy media |
|--------|----------|----------------|
| KNN | 0,7039 · 0,7241 · 0,7087 · 0,7176 · 0,6936 | 0,7096 ± 0,0106 |
| Regresión logística | 0,7574 · 0,7503 · 0,7717 · 0,7782 · 0,7549 | **0,7625 ± 0,0106** |

**Modelo principal: regresión logística**, por tener la mayor accuracy en
validación cruzada (estimación más robusta que un solo corte de prueba) y la
menor brecha de sobreajuste. El preprocesamiento se ajusta dentro de cada
pliegue, por lo que la validación cruzada no tiene fuga de información.

## 18. Visualizaciones (requisitos 11 a 14)

Las 11 figuras se generan en `reports/figures/` con `dpi=150`,
`bbox_inches="tight"` y `plt.close()`, sobre el backend `Agg` (sin pantalla).
Títulos, ejes y unidades en español.

| Figura | Contenido |
|--------|-----------|
| `histograma_ingreso_per_capita.png` | Distribución del PNB per cápita con media (roja) y mediana (verde) marcadas |
| `histogramas_predictores.png` | 4 histogramas (población, exportaciones, valor añadido total, consumo) con media y mediana |
| `matriz_correlacion.png` | Heatmap de Pearson de las 12 numéricas más ligadas al ingreso |
| `dispersion_pairplot.png` | Pairplot por pares coloreado por nivel de ingreso (muestra de 1.500) |
| `boxplot_valores_atipicos.png` | Boxplots de variables estandarizadas con atípicos en rojo y umbral \|Z\| = 3 |
| `distribucion_clases.png` | Balance de las 3 clases del objetivo |
| `matriz_confusion_knn.png` | Matriz de confusión del KNN en prueba |
| `matriz_confusion_regresion_logistica.png` | Matriz de confusión de la regresión logística en prueba |
| `metricas_train_vs_test.png` | Comparación de las 4 métricas en train y test por modelo |
| `experimento_dispersion_metricas.png` | Barras del experimento: sin escalar vs escalado vs escalado solo numéricas |
| `experimento_dispersion_desviaciones.png` | Desviación estándar de cada variable (escala logarítmica) |

## 19. Resultados obtenidos

1. **Modelo principal (regresión logística, prueba):** accuracy **0,7694**,
   precision **0,7736**, recall **0,7694**, F1 **0,7629**; validación cruzada
   **0,7625 ± 0,0106**.
2. **KNN (prueba):** accuracy **0,7323**, F1 **0,7275**; validación cruzada
   **0,7096 ± 0,0106**.
3. **Experimento de dispersión:** KNN 0,8597 sin escalar → 0,7323 escalado →
   **0,9301 escalado sin bloque one-hot**; regresión logística 0,7418 →
   0,7694 → 0,8112.
4. **Estadística manual:** varianza poblacional 291.364.209,3792 y muestral
   291.391.929,3116, idénticas a NumPy con `ddof` explícito.
5. **Atípicos:** 2.572 detectados con \|Z\| > 3; decisión: conservarlos.
6. **Álgebra:** producto punto y norma manuales coinciden con NumPy;
   sistema resuelto con residuo 0,00e+00; caso singular capturado.
7. **Pruebas:** `pytest` → **22 passed**.

## 20. Interpretación de los resultados

- **El accuracy del 76,94 % es sólido para 3 clases balanceadas** (un modelo
  aleatorio rondaría el 33 %). Como las clases están equilibradas, el accuracy
  es una métrica fiable y el F1 macro confirma que no hay una clase que se
  arrastre sola.
- **La regresión logística generaliza mejor que el KNN** en esta configuración
  (brecha +0,0162 frente a +0,0900 y mejor validación cruzada): el KNN
  memoriza mejor los registros que ya vio.
- **La dispersión manda.** Con variables cuya desviación estándar difiere
  hasta 498 millones de veces, cualquier modelo que combine por distancias o
  por pesos necesita estandarización; el experimento lo cuantifica en las
  tres variantes de la sección 15.
- **El one-hot de `Year` no es inocuo en KNN**: al escalarse el resto, sus
  columnas binarias ganan peso real en la distancia (0,1322 del total) y
  cambian a los vecinos: de caer en años distintos (0,67 % del mismo año) a
  caer en el mismo año (93,01 %).
- **La correlación visible no implica causalidad**: el heatmap muestra
  asociaciones (p. ej. exportaciones con producto), no relaciones causales.

## 21. Limitaciones del análisis

1. **Estructura de panel con división aleatoria:** el 100,0 % de los registros
   de prueba pertenecen a un país presente en entrenamiento. Un país se
   repite ~48 años y sus agregados cambian poco de un año a otro, así que
   parte del rendimiento proviene de "reconocer economías ya vistas" y no
   solo de aprender la relación ingreso-estructura. La solución rigurosa sería
   dividir por grupos (`GroupShuffleSplit` por país) o por tiempo (entrenar
   hasta un año y testear los siguientes).
2. **Objetivo nominal en dólares corrientes:** los terciles mezclan la
   posición relativa con la inflación acumulada de 51 años. Una alternativa es
   usar dólares constantes o rendimientos.
3. **Atípicos conservados:** no se hizo el análisis opcional de rendimiento
   "con y sin" atípicos; solo se detectaron y visualizaron.
4. **`Year` como categórica penaliza al KNN escalado** (medido: 0,9301 sin el
   bloque frente a 0,7323 con él). Se mantiene porque el enunciado exige
   ejemplificar la codificación one-hot, y el coste queda documentado.
5. **Imputación por mediana/moda** asume que los nulos son aproximadamente
   aleatorios; con `Changes in inventories` (17,51 % de nulos) esa suposición
   es discutible.
6. **Dos modelos sencillos:** no se exploraron árboles, SVM ni ajuste de
   hiperparámetros (búsqueda de k, regularización), que probablemente
   mejorarían las métricas.

## 22. Conclusiones

1. El proyecto cumple el circuito completo —exploración, estadística manual,
   álgebra lineal, preprocesamiento sin fugas, dos modelos, evaluación y
   visualización— en una estructura modular ejecutable con `python main.py` y
   verificable con `pytest` (22 pruebas).
2. Se resolvió un problema de **clasificación multiclase balanceado** (33,3 %
   por clase) eligiendo el objetivo **después** de explorar los datos, con 6
   columnas excluidas documentadas por fuga de información o cardinalidad.
3. Las funciones manuales de varianza y desviación coinciden con NumPy
   **solo cuando el `ddof` es explícito**: la trampa de los valores por defecto
   de NumPy (0) y Pandas (1) es real y está documentada en el código y aquí.
4. El **impacto de la dispersión es medible y no siempre intuitivo**: escalar
   es imprescindible con desviaciones estándar que difieren 498 millones de
   veces, pero su efecto neto depende de qué otras columnas acompañan (bloque
   one-hot) — el experimento 10.1 lo deja cuantificado en tres variantes.
5. La **regresión logística escalada** es el modelo recomendado: mejor
   validación cruzada (0,7625), menor sobreajuste (+0,0162) y métricas de
   prueba equilibradas (accuracy 0,7694 / F1 0,7629).

---

*Cifras obtenidas de la ejecución de `python main.py` el 2026-10-01 con
Python 3.14.7, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1,
matplotlib 3.11.2, seaborn 0.13.2 y pytest 9.1.1.*
