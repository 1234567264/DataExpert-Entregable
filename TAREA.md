# Fundamentos Matemáticos y Algorítmicos Aplicados a Inteligencia Artificial en Python

## Tema

**Fundamentos Matemáticos y Algorítmicos Aplicados a Inteligencia Artificial en Python**

---

## Contexto

La empresa **DataExpert**, especializada en el desarrollo de soluciones basadas en **Inteligencia Artificial (IA)**, enfrenta dificultades en el desarrollo de modelos debido a la falta de una base sólida en los fundamentos matemáticos y algorítmicos.

Esta situación limita su capacidad para desarrollar, optimizar y escalar soluciones eficientes basadas en **Machine Learning** y **Deep Learning**.

Para resolver estos problemas, la empresa requiere implementar algoritmos aplicados a IA que integren conocimientos de **Machine Learning, estadística y álgebra lineal** utilizando Python.

El objetivo es desarrollar una solución educativa que permita comprender y aplicar operaciones matemáticas fundamentales, analizar datos y entrenar modelos básicos de aprendizaje automático.

---

## Requerimientos principales

La solución debe incluir:

* Implementación de un modelo básico de **Machine Learning** para resolver un problema de clasificación o regresión utilizando **Scikit-learn**.
* Aplicación de operaciones de **álgebra lineal** utilizando **NumPy**, incluyendo vectores, matrices, producto punto, normas y sistemas de ecuaciones lineales.
* Desarrollo de un análisis estadístico de datos utilizando **Pandas**, **NumPy** y herramientas de visualización.
* Implementación de funciones para calcular manualmente la **varianza y desviación estándar**, comparando los resultados con los obtenidos mediante **NumPy**.
* Análisis del **impacto de la dispersión de los datos** en el comportamiento de un modelo de IA.
* Identificación de valores atípicos utilizando medidas estadísticas.
* Creación de visualizaciones para interpretar distribuciones y relaciones entre variables utilizando **Matplotlib** y **Seaborn**.

---

## Fuente de datos

La información puede obtenerse de fuentes de datos abiertas como:

* [Kaggle - Datasets](https://www.kaggle.com/datasets)
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

También se pueden generar datos sintéticos utilizando Python.

Para este proyecto, el dataset se encuentra disponible en:

```text
data/data.csv
```

En caso de no disponer de un dataset, se deberá generar o seleccionar uno apropiado para el problema de clasificación o regresión que se desea resolver.

El proyecto debe trabajar con los datos seleccionados y adaptar el procesamiento a sus características reales.

> **Nota:** No se deben asumir columnas, tipos de datos ni características específicas sin comprobar previamente la estructura del dataset. En particular, la **variable objetivo y el tipo de problema (clasificación o regresión) se definen después de la exploración**, no antes.

---

## Estructura del proyecto

El proyecto se organiza de forma modular: cada archivo de `src/` agrupa una responsabilidad, y `main.py` en la raíz es el **punto de entrada único** que ejecuta cada etapa una por una, en orden.

```text
proyecto-ia-fundamentos/
├── main.py                    # Punto de entrada: orquesta todas las etapas en orden
├── requirements.txt           # Dependencias con versiones fijadas
├── README.md                  # Cómo instalar y ejecutar el proyecto
├── .gitignore                 # Excluye .venv/, __pycache__/, etc. y .atl
├── data/
│   └── data.csv               # Dataset del proyecto
├── src/
│   ├── __init__.py
│   ├── config.py              # Constantes: rutas, semilla aleatoria, umbral Z, etc.
│   ├── exploracion.py         # Lectura y exploración inicial (requisito 1)
│   ├── estadistica.py         # Media, mediana, varianza/desviación manual, outliers (8, 9, 10)
│   ├── algebra_lineal.py      # Vectores, matrices, producto punto, norma, sistemas (5, 6, 7)
│   ├── preprocesamiento.py    # Nulos, duplicados, codificación, escalado, split (2)
│   ├── modelo.py              # Entrenamiento, predicción y evaluación (3, 4)
│   └── visualizacion.py       # Histogramas, dispersión, boxplots, gráficos del modelo (11-14)
├── tests/
│   └── test_estadistica.py    # Pruebas: funciones manuales vs NumPy
└── reports/
    ├── figures/               # Gráficos generados (PNG)
    └── informe.md             # Informe técnico final
```

### Diseño de `main.py`

`main.py` no contiene lógica de negocio: solo **importa las funciones de `src/` y las llama en orden**, pasándose los resultados de una etapa a la siguiente. Así cada módulo se puede probar y reutilizar por separado.

Orden de ejecución:

| Paso | Etapa | Módulo | Qué hace |
|------|-------|--------|----------|
| 1 | Exploración | `exploracion.py` | Lee el CSV y muestra estructura, tipos, nulos y duplicados |
| 2 | Estadística | `estadistica.py` | Media, mediana, varianza y desviación manual vs NumPy |
| 3 | Valores atípicos | `estadistica.py` | Detección con puntuación Z |
| 4 | Álgebra lineal | `algebra_lineal.py` | Vectores, matrices, producto punto, norma, sistema lineal |
| 5 | Preprocesamiento | `preprocesamiento.py` | Limpieza, codificación, split y escalado (sin fuga de datos) |
| 6 | Modelo | `modelo.py` | Entrena, predice y evalúa |
| 7 | Visualización | `visualizacion.py` | Genera y guarda todos los gráficos en `reports/figures/` |

Esqueleto sugerido:

```python
"""Punto de entrada del proyecto: ejecuta cada etapa en orden."""
import logging

from src import (
    algebra_lineal,
    config,
    estadistica,
    exploracion,
    modelo,
    preprocesamiento,
    visualizacion,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Paso 1/7: exploración de datos")
    df = exploracion.cargar_datos(config.DATA_PATH)
    exploracion.resumen(df)

    logger.info("Paso 2/7: análisis estadístico")
    estadistica.analizar(df)

    logger.info("Paso 3/7: valores atípicos")
    atipicos = estadistica.detectar_atipicos(df, umbral=config.Z_UMBRAL)

    logger.info("Paso 4/7: álgebra lineal")
    algebra_lineal.ejecutar_demostraciones()

    logger.info("Paso 5/7: preprocesamiento")
    X_train, X_test, y_train, y_test = preprocesamiento.preparar(df)

    logger.info("Paso 6/7: modelo")
    resultados = modelo.entrenar_y_evaluar(X_train, X_test, y_train, y_test)

    logger.info("Paso 7/7: visualizaciones")
    visualizacion.generar_todas(df, atipicos, resultados)

    logger.info("Proyecto completado. Figuras en %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
```

> **Nota de buena práctica:** la protección `if __name__ == "__main__":` evita que el código se ejecute al importar el archivo. Las constantes (rutas, semilla, umbral) viven en `config.py` para no tener "números mágicos" repartidos por el código.

```python
# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RANDOM_STATE = 42     # Reproducibilidad: mismos resultados en cada ejecución
TEST_SIZE = 0.2
Z_UMBRAL = 3.0

# Se define DESPUÉS de explorar el dataset (no asumir columnas):
TARGET: str | None = None
```

---

## Requisitos funcionales

### 1. Lectura y exploración de datos

**Módulo:** `src/exploracion.py`

El proyecto debe permitir:

* Leer archivos en formato CSV.
* Inspeccionar la estructura del dataset.
* Identificar la cantidad de filas y columnas.
* Examinar los nombres de las variables.
* Identificar los tipos de datos.
* Detectar valores faltantes.
* Detectar registros duplicados.
* Obtener una vista preliminar de los datos.
* Generar un resumen inicial del dataset.

Se debe utilizar principalmente:

* Pandas
* NumPy

Ejemplo básico:

```python
import pandas as pd

df = pd.read_csv("data/data.csv")
print(df.shape)            # (filas, columnas)
print(df.dtypes)           # tipo de cada columna
print(df.head())           # vista preliminar
print(df.isna().sum())     # nulos por columna
print(df.duplicated().sum())  # filas duplicadas
```

> **Decisión que sale de este paso:** con la estructura y la variable que se quiere predecir se define el tipo de problema. Objetivo **categórico** → clasificación; objetivo **numérico continuo** → regresión. Con ello se fijan el algoritmo y las métricas de los requisitos 3 y 4.

Documentación oficial: [`pandas.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), [`DataFrame.info`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html), [`DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html).

### 2. Preprocesamiento de datos

**Módulo:** `src/preprocesamiento.py`

Antes de entrenar los modelos, se deben aplicar técnicas de preprocesamiento adecuadas.

Entre ellas pueden incluirse:

* Tratamiento de valores nulos.
* Eliminación de registros duplicados cuando corresponda.
* Codificación de variables categóricas.
* Normalización de variables numéricas.
* Estandarización de características.
* Selección de características relevantes.
* Separación de variables predictoras y variable objetivo.
* División de los datos en conjuntos de entrenamiento y prueba.

Se utilizará principalmente:

* Pandas
* NumPy
* Scikit-learn

> El preprocesamiento debe adaptarse a las características reales del dataset. **Las transformaciones que aprenden parámetros de los datos (imputación, escalado, codificación) deben ajustarse (`fit`) únicamente con el conjunto de entrenamiento** y luego solo aplicarse (`transform`) al de prueba, para evitar la fuga de información (*data leakage*).

**Orden correcto:**

1. Separar `X` (predictoras) e `y` (objetivo).
2. Dividir en entrenamiento y prueba **primero**.
3. Ajustar los transformadores con el entrenamiento.
4. Aplicarlos al entrenamiento y a la prueba.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,   # reproducibilidad
    # stratify=y,      # solo en clasificación: conserva la proporción de clases
)
```

**Buena práctica en producción:** encapsular el preprocesamiento y el modelo en un `Pipeline` con `ColumnTransformer`. Así el `fit` solo ve datos de entrenamiento (también dentro de validación cruzada) y el mismo objeto se puede guardar y reutilizar:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numericas = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
])
categoricas = Pipeline([
    ("imputar", SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore")),
])

preprocesador = ColumnTransformer([
    ("num", numericas, columnas_numericas),     # definidas tras explorar el dataset
    ("cat", categoricas, columnas_categoricas),
])
```

> **Nota:** el escalado importa para modelos sensibles a la escala (KNN, regresión logística, regresión lineal con regularización) y no es necesario para árboles de decisión. Elegir las transformaciones según el algoritmo.

Documentación oficial: [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).

### 3. Modelo básico de Machine Learning

**Módulo:** `src/modelo.py`

Se debe implementar un modelo básico de aprendizaje automático que permita resolver un problema de clasificación o regresión.

El modelo debe incluir:

* Selección del dataset.
* Identificación del problema.
* Exploración de las variables.
* Selección de características.
* Definición de la variable objetivo.
* Preparación de los datos.
* División del dataset.
* Selección del algoritmo.
* Entrenamiento del modelo.
* Generación de predicciones.
* Evaluación de los resultados.

Se utilizará principalmente:

* Scikit-learn

Se pueden utilizar algoritmos sencillos como:

* Regresión lineal.
* Regresión logística.
* Árboles de decisión.
* K-Nearest Neighbors (KNN).

La elección del algoritmo dependerá del tipo de problema y de las características del dataset.

| Tipo de problema | Algoritmos sencillos |
|------------------|----------------------|
| Clasificación | Regresión logística, árbol de decisión, KNN |
| Regresión | Regresión lineal, árbol de decisión (regresor), KNN (regresor) |

> **Nota:** la regresión logística, a pesar del nombre, es un modelo de **clasificación**.

Documentación oficial: [Guía de usuario de scikit-learn](https://scikit-learn.org/stable/user_guide.html).

### 4. Validación y evaluación del modelo

**Módulo:** `src/modelo.py`

El modelo implementado debe evaluarse utilizando métricas apropiadas al problema.

Para clasificación, se pueden utilizar:

* Accuracy.
* Precision.
* Recall.
* F1-score.
* Matriz de confusión.

Para regresión, se pueden utilizar:

* Error absoluto medio (MAE).
* Error cuadrático medio (MSE).
* Raíz del error cuadrático medio (RMSE).
* Coeficiente de determinación (R²).

La evaluación debe permitir identificar:

* El rendimiento del modelo.
* La precisión de las predicciones.
* Los errores cometidos.
* La capacidad de generalización.

> Las métricas deben seleccionarse de acuerdo con el tipo de problema. No se deben aplicar métricas de clasificación a modelos de regresión ni viceversa.

**Notas importantes:**

* **Accuracy puede engañar con clases desbalanceadas.** Si el 95 % de los registros es de una clase, un modelo que siempre predice esa clase tiene 95 % de accuracy y no sirve. Revisar la distribución de clases y apoyarse en precision, recall y F1.
* **Generalización:** comparar el rendimiento en entrenamiento y en prueba. Una diferencia grande indica sobreajuste (*overfitting*). Opcionalmente, usar validación cruzada (`cross_val_score`) para una estimación más estable.
* **RMSE:** en versiones recientes de scikit-learn existe `root_mean_squared_error`; en versiones antiguas se calcula como `np.sqrt(mean_squared_error(...))`. Verificar la versión instalada, porque el parámetro `squared=False` de `mean_squared_error` fue deprecado.

Documentación oficial: [Métricas de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).

### 5. Operaciones con álgebra lineal

**Módulo:** `src/algebra_lineal.py`

Se deben implementar operaciones fundamentales de álgebra lineal utilizando Python.

El proyecto debe incluir:

* Creación de vectores.
* Creación de matrices.
* Acceso y modificación de elementos.
* Suma y resta de vectores.
* Suma y resta de matrices.
* Multiplicación de matrices.
* Multiplicación de un vector por un escalar.
* Transposición de matrices.
* Producto punto de vectores.
* Cálculo de normas vectoriales.

Se utilizará principalmente:

* NumPy

Las operaciones deben acompañarse de ejemplos que expliquen su funcionamiento y su aplicación en problemas de Inteligencia Artificial.

Ejemplo simple:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(a + b)      # suma de vectores
print(3 * a)      # multiplicación por escalar
print(A.T)        # transpuesta
print(A @ B)      # multiplicación de matrices (no es A * B, que es elemento a elemento)
```

> **Nota:** `A * B` multiplica **elemento a elemento**; la multiplicación matricial es `A @ B` (o `np.matmul`). Confundirlos es un error muy común.

**Aplicación en IA:** un dataset se representa como una matriz `X` de forma `(n_muestras, n_características)`; las predicciones de un modelo lineal se calculan como `X @ w + b`.

Documentación oficial: [NumPy: guía de álgebra lineal](https://numpy.org/doc/stable/reference/routines.linalg.html).

### 6. Producto punto y norma de un vector

**Módulo:** `src/algebra_lineal.py`

Se debe demostrar el cálculo del producto punto entre dos vectores y la obtención de la norma de un vector.

El producto punto puede utilizarse para representar operaciones como la combinación ponderada de características en modelos de Machine Learning.

Se debe implementar:

* Definición de dos vectores.
* Cálculo del producto punto.
* Interpretación del resultado.
* Cálculo de la norma euclidiana.
* Comparación entre el cálculo manual y el resultado obtenido con NumPy.

Se utilizarán principalmente:

* NumPy

Las operaciones pueden expresarse matemáticamente como:

Producto punto:

$$
\mathbf{a}\cdot\mathbf{b}=\sum_{i=1}^{n}a_i b_i
$$

Norma euclidiana:

$$
\|\mathbf{a}\|_2=\sqrt{\sum_{i=1}^{n}a_i^2}
$$

Ejemplo (manual vs NumPy):

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

producto_manual = sum(x * y for x, y in zip(a, b))
norma_manual = sum(x ** 2 for x in a) ** 0.5

assert np.isclose(producto_manual, np.dot(a, b))
assert np.isclose(norma_manual, np.linalg.norm(a))
```

> **Nota:** comparar números decimales con `np.isclose` o `np.allclose`, no con `==`, por los errores de redondeo de punto flotante.

**Interpretación en IA:** en un modelo lineal, `pesos · características` es la combinación ponderada que produce la predicción; la norma de los pesos se usa en técnicas de regularización.

Documentación oficial: [`numpy.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html), [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).

### 7. Resolución de sistemas de ecuaciones lineales

**Módulo:** `src/algebra_lineal.py`

Se debe implementar la resolución de sistemas de ecuaciones lineales mediante operaciones matriciales.

El proyecto debe incluir:

* Definición de un sistema de ecuaciones.
* Representación matricial del sistema.
* Identificación de la matriz de coeficientes.
* Definición del vector de términos independientes.
* Resolución del sistema.
* Comprobación de la solución obtenida.

Se utilizará principalmente:

* NumPy

Se puede utilizar la función `numpy.linalg.solve()` para resolver sistemas compatibles con una solución única.

La representación general de un sistema lineal es:

$$
A\mathbf{x}=\mathbf{b}
$$

> Se debe comprobar que la matriz de coeficientes permita obtener una solución única antes de utilizar el método de resolución directa.

Ejemplo:

```python
import numpy as np

# 2x + y = 5
#  x + 3y = 10
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])
b = np.array([5.0, 10.0])

# Comprobar que la solución es única: la matriz debe ser cuadrada y de rango completo
assert A.shape[0] == A.shape[1]
assert np.linalg.matrix_rank(A) == A.shape[0]

x = np.linalg.solve(A, b)

# Comprobación: A @ x debe reproducir b
assert np.allclose(A @ x, b)
print(x)
```

**Notas importantes:**

* Si la matriz es singular, `np.linalg.solve` lanza `LinAlgError`. Conviene capturarlo en lugar de dejar que el programa se caiga.
* Para comprobar la unicidad es más fiable `np.linalg.matrix_rank` (o el número de condición `np.linalg.cond`) que mirar si el determinante "es cero", porque el determinante con decimales puede dar valores minúsculos que no son exactamente cero.
* En la práctica se prefiere `solve` sobre calcular la inversa (`np.linalg.inv(A) @ b`): es más estable numéricamente y más eficiente.

Documentación oficial: [`numpy.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), [`numpy.linalg.matrix_rank`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html).

### 8. Análisis estadístico de datos

**Módulo:** `src/estadistica.py`

Se debe realizar un análisis estadístico de las variables numéricas del dataset.

Debe incluir, según las características de los datos:

* Cálculo de la media.
* Cálculo de la mediana.
* Identificación de valores mínimos y máximos.
* Obtención de cuartiles.
* Análisis de distribuciones.
* Interpretación de medidas de tendencia central.
* Comparación de variables numéricas.

Se utilizarán principalmente:

* Pandas
* NumPy

El análisis debe permitir comprender las características generales de los datos y su comportamiento estadístico.

```python
columnas_numericas = df.select_dtypes(include="number")

print(columnas_numericas.describe())   # media, std, min, cuartiles, max
print(columnas_numericas.median())
```

> **Nota de interpretación:** si la media y la mediana de una variable difieren mucho, la distribución probablemente es asimétrica o tiene valores extremos. La mediana es más robusta frente a atípicos que la media.

Documentación oficial: [`DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html), [`DataFrame.quantile`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html).

### 9. Cálculo de varianza y desviación estándar

**Módulo:** `src/estadistica.py`

Se deben implementar funciones en Python para calcular manualmente la varianza y la desviación estándar de un conjunto de datos.

El proyecto debe incluir:

* Definición de un conjunto de datos numéricos.
* Cálculo de la media.
* Cálculo de las diferencias respecto a la media.
* Elevación al cuadrado de las diferencias.
* Cálculo de la varianza.
* Cálculo de la desviación estándar.
* Comparación de los resultados manuales con NumPy.
* Interpretación de la dispersión de los datos.

Se utilizarán principalmente:

* Python
* NumPy

Se deben considerar las diferencias entre varianza poblacional y varianza muestral.

Varianza poblacional:

$$
\sigma^2=\frac{\sum_{i=1}^{N}(x_i-\mu)^2}{N}
$$

Varianza muestral:

$$
s^2=\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}
$$

Desviación estándar:

$$
\sigma=\sqrt{\sigma^2}
$$

> Es importante especificar si los datos representan una población completa o una muestra, ya que el denominador utilizado cambia el resultado.

> ⚠️ **Trampa frecuente al comparar con las librerías:** NumPy y Pandas usan valores por defecto **distintos**.
>
> | Función | Valor por defecto | Tipo de varianza |
> |---------|-------------------|------------------|
> | `np.var`, `np.std` | `ddof=0` | Poblacional (divide entre N) |
> | `pd.Series.var`, `pd.Series.std` | `ddof=1` | Muestral (divide entre n−1) |
>
> Si se compara la función manual con una y con la otra sin fijar el parámetro, los resultados no coinciden y parece un error de la implementación. **Siempre pasar `ddof` de forma explícita.**

Implementación sugerida (con el tipo de varianza explícito):

```python
import numpy as np


def varianza(datos, muestral: bool = False) -> float:
    """Varianza calculada manualmente.

    muestral=False -> poblacional (divide entre N)
    muestral=True  -> muestral    (divide entre n - 1)
    """
    n = len(datos)
    if n == 0:
        raise ValueError("El conjunto de datos no puede estar vacío.")
    if muestral and n < 2:
        raise ValueError("La varianza muestral requiere al menos 2 datos.")

    media = sum(datos) / n
    suma_cuadrados = sum((x - media) ** 2 for x in datos)
    return suma_cuadrados / (n - 1 if muestral else n)


def desviacion_estandar(datos, muestral: bool = False) -> float:
    return varianza(datos, muestral) ** 0.5


datos = [2, 4, 4, 4, 5, 5, 7, 9]

assert np.isclose(varianza(datos), np.var(datos, ddof=0))
assert np.isclose(varianza(datos, muestral=True), np.var(datos, ddof=1))
assert np.isclose(desviacion_estandar(datos), np.std(datos, ddof=0))
```

**Buena práctica en producción:** estas funciones críticas deben tener pruebas automáticas (`tests/test_estadistica.py`, con `pytest`) que comparen el resultado manual contra NumPy con `ddof` explícito, incluyendo casos borde (lista vacía, un solo dato, valores negativos).

Documentación oficial: [`numpy.var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html), [`numpy.std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html), [`pandas.Series.var`](https://pandas.pydata.org/docs/reference/api/pandas.Series.var.html).

### 10. Identificación de valores atípicos

**Módulo:** `src/estadistica.py`

Se deben identificar posibles valores atípicos mediante técnicas estadísticas basadas en la desviación estándar.

El proceso debe incluir:

* Selección de variables numéricas.
* Cálculo de la media.
* Cálculo de la desviación estándar.
* Definición de un umbral de detección.
* Identificación de observaciones que superen dicho umbral.
* Visualización de los valores atípicos.
* Interpretación de los resultados.

Se puede utilizar un criterio basado en puntuaciones Z.

$$
Z=\frac{x-\mu}{\sigma}
$$

Se pueden considerar como posibles valores atípicos aquellos registros cuyo valor absoluto de Z sea superior a 3.

Se utilizarán principalmente:

* NumPy
* Pandas

> La identificación de valores atípicos no implica necesariamente su eliminación. Se debe evaluar si representan errores, casos poco frecuentes o información relevante para el problema.

Ejemplo:

```python
def detectar_atipicos_z(serie, umbral: float = 3.0):
    """Devuelve los valores cuyo |Z| supera el umbral."""
    media = serie.mean()
    desviacion = serie.std(ddof=0)   # fijar ddof de forma explícita y consistente
    if desviacion == 0:
        return serie.iloc[0:0]       # sin dispersión no hay atípicos
    z = (serie - media) / desviacion
    return serie[z.abs() > umbral]
```

**Limitaciones que conviene mencionar en el informe:**

* **Los atípicos distorsionan la media y la desviación** que se usan para detectarlos (un valor muy extremo infla la desviación y puede "esconderse" a sí mismo).
* **Con pocos datos, el umbral 3 puede no detectar nada:** el |Z| máximo posible depende del tamaño de la muestra y, con muestras muy pequeñas, ningún valor puede superar 3.
* El criterio Z funciona mejor con distribuciones aproximadamente simétricas. Como contraste, se puede comparar con el método del rango intercuartílico (IQR), más robusto.
* Aplicar el análisis **solo a variables numéricas** y no a identificadores o códigos que parecen números.

### 10.1 Impacto de la dispersión en el modelo

**Módulo:** `src/modelo.py` (y gráficos en `src/visualizacion.py`)

El caso práctico pide analizar cómo la dispersión de los datos afecta a los modelos de IA. Se puede cubrir con un experimento sencillo y comparable:

1. Entrenar el mismo modelo **sin escalar** las variables.
2. Entrenar el mismo modelo **con `StandardScaler`** (ajustado solo con entrenamiento).
3. Comparar las métricas de ambos y relacionarlas con la desviación estándar de cada variable.

> **Por qué importa:** en algoritmos basados en distancias, como KNN, una variable con una escala mucho mayor (mayor desviación estándar) domina el cálculo de distancias y las demás pierden influencia. Estandarizar pone las variables en una escala comparable (media 0, desviación 1).

Complemento opcional: mostrar cómo la presencia de atípicos cambia el rendimiento (por ejemplo, con y sin los registros marcados en el requisito 10) y discutir el resultado, sin eliminar datos de forma automática.

---

## Visualización de datos

**Módulo:** `src/visualizacion.py`

Se deben crear visualizaciones que permitan analizar e interpretar los datos y los resultados del modelo.

Se utilizarán:

* Matplotlib
* Seaborn

> **Buena práctica:** guardar cada figura en `reports/figures/` con `plt.savefig(..., dpi=150, bbox_inches="tight")` y cerrarla con `plt.close()` al terminar. Así `main.py` genera los gráficos sin depender de ventanas interactivas y no se acumulan figuras abiertas en memoria. Cada gráfico debe tener título, nombres de ejes y unidades cuando apliquen.

Documentación oficial: [Matplotlib](https://matplotlib.org/stable/), [Seaborn](https://seaborn.pydata.org/).

### 11. Histogramas

Se deben implementar histogramas para analizar la distribución de variables numéricas relevantes.

Por ejemplo:

* Distribución de edades.
* Distribución de ingresos.
* Distribución de precios.
* Distribución de características numéricas del dataset.

Los histogramas deben permitir observar:

* Concentración de valores.
* Dispersión.
* Asimetría.
* Posibles valores extremos.
* Forma de la distribución.

La selección de variables dependerá de las características reales del dataset.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x="columna_numerica", kde=True)
plt.title("Distribución de columna_numerica")
plt.savefig("reports/figures/hist_columna_numerica.png", dpi=150, bbox_inches="tight")
plt.close()
```

> **Nota:** el número de *bins* cambia la forma aparente de la distribución. Probar varios valores antes de sacar conclusiones.

### 12. Gráficos de dispersión

Se deben implementar gráficos de dispersión para analizar la relación entre dos variables numéricas.

Por ejemplo:

```text
Variable X

     ↕

Variable Y
```

Los gráficos deben permitir identificar:

* Relaciones entre variables.
* Tendencias.
* Agrupaciones.
* Posibles valores atípicos.
* Patrones que puedan resultar útiles para el modelo.

Se utilizarán principalmente:

* Matplotlib
* Seaborn

> **Nota:** una correlación visible no implica causalidad. Si se calcula el coeficiente de correlación, interpretarlo junto al gráfico y no de forma aislada.

### 13. Visualización de medidas estadísticas

Se deben desarrollar visualizaciones que permitan representar las principales medidas estadísticas calculadas.

Se pueden utilizar:

* Diagramas de caja (*boxplots*).
* Histogramas con líneas de media y mediana.
* Gráficos comparativos de distribuciones.
* Gráficos de dispersión con variables relevantes.

Estas visualizaciones deben facilitar la interpretación de la distribución de los datos y la identificación de posibles anomalías.

```python
sns.histplot(data=df, x="columna_numerica")
plt.axvline(df["columna_numerica"].mean(), color="red", linestyle="--", label="Media")
plt.axvline(df["columna_numerica"].median(), color="green", linestyle="-", label="Mediana")
plt.legend()
```

### 14. Visualización de resultados del modelo

Se deben generar gráficos que permitan interpretar los resultados obtenidos durante el entrenamiento y la evaluación del modelo.

Según el tipo de problema, se pueden incluir:

Para clasificación:

* Matriz de confusión.
* Comparación entre clases reales y predichas.
* Visualización de métricas de evaluación.

Para regresión:

* Valores reales frente a valores predichos.
* Gráfico de residuos.
* Comparación de errores de predicción.

Se utilizarán principalmente:

* Matplotlib
* Seaborn
* Scikit-learn

Ejemplo (clasificación):

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

---

## Entrenamiento y evaluación

El modelo implementado debe entrenarse utilizando los datos preparados previamente.

Posteriormente, debe evaluarse mediante métricas apropiadas al problema.

La evaluación debe permitir determinar:

* Qué tan bien funciona el modelo.
* Qué tan precisas son sus predicciones.
* Qué errores presenta.
* Qué tan bien generaliza sobre datos no utilizados durante el entrenamiento.
* Cómo influyen las características seleccionadas en los resultados.

Los resultados deben acompañarse de gráficos cuando sea apropiado.

Además, se deben interpretar las operaciones matemáticas realizadas y explicar su importancia en los procesos de Machine Learning.

---

## Requisitos de implementación

### Librerías principales

El proyecto debe utilizar las siguientes librerías:

| Librería     | Uso                                                     |
| ------------ | ------------------------------------------------------- |
| Pandas       | Lectura, manipulación y análisis de datos               |
| NumPy        | Operaciones numéricas, vectores, matrices y estadística |
| Scikit-learn | Preprocesamiento, entrenamiento y evaluación de modelos |
| Matplotlib   | Visualización de datos y resultados                     |
| Seaborn      | Visualización estadística                               |
| Pytest       | Pruebas automáticas de las funciones manuales           |

También se pueden utilizar las librerías estándar de Python para implementar manualmente operaciones matemáticas y funciones estadísticas.

### Entorno y dependencias

* Trabajar dentro de un **entorno virtual** (`python -m venv .venv`) para aislar las dependencias del proyecto.
* Registrar las dependencias en `requirements.txt` **con versiones fijadas** (`pip freeze` o escritas a mano), para que el proyecto sea reproducible en otra máquina.
* Ejecución completa del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
pip install -r requirements.txt
python main.py
pytest                          # ejecuta las pruebas
```

### Dataset del proyecto

El dataset utilizado por el proyecto se encuentra en:

```text
data/data.csv
```

Si no existe un dataset disponible, se deberá seleccionar una fuente pública o generar datos sintéticos.

La implementación debe inspeccionar primero los datos y adaptar los procedimientos a sus columnas, tipos y características reales.

> No se deben inventar variables ni asumir que el dataset contiene características específicas sin haberlo inspeccionado.

### Criterios de calidad del código

* **Modularidad:** una responsabilidad por archivo; `main.py` solo orquesta.
* **Reproducibilidad:** semilla fija (`RANDOM_STATE`) en todas las operaciones aleatorias.
* **Legibilidad:** nombres descriptivos, *docstrings* en cada función pública y anotaciones de tipo.
* **Sin números mágicos:** constantes en `config.py`.
* **Rutas portables:** usar `pathlib.Path`, no rutas escritas a mano con `\` o `/`.
* **Trazabilidad:** usar `logging` en lugar de `print` para los mensajes de avance de `main.py`.
* **Manejo de errores:** validar entradas (archivo inexistente, listas vacías, matriz singular) con mensajes claros.
* **Pruebas:** al menos las funciones estadísticas manuales comparadas contra NumPy.

---

## Entregable

La solución debe presentarse como un proyecto educativo de Python que incluya:

* Lectura y exploración de datos CSV.
* Análisis de la estructura del dataset.
* Preprocesamiento de datos.
* Manejo de valores nulos y duplicados.
* Normalización y codificación de variables cuando corresponda.
* Implementación de un modelo básico de Machine Learning.
* Entrenamiento y evaluación del modelo.
* Creación y manipulación de vectores y matrices.
* Operaciones de álgebra lineal.
* Cálculo de producto punto.
* Cálculo de normas vectoriales.
* Resolución de sistemas de ecuaciones lineales.
* Cálculo de media y mediana.
* Cálculo manual de varianza y desviación estándar.
* Comparación de resultados estadísticos con NumPy.
* Análisis del impacto de la dispersión en el modelo.
* Identificación de valores atípicos.
* Generación de histogramas.
* Creación de gráficos de dispersión.
* Visualización e interpretación de resultados.
* Estructura modular (`src/`) con punto de entrada único (`main.py`).
* Informe técnico con los procedimientos realizados.

### Informe técnico

El proyecto debe finalizar con un informe técnico (`reports/informe.md`) que explique los procedimientos realizados.

El informe debe incluir, como mínimo:

* Introducción.
* Descripción del problema.
* Objetivos del proyecto.
* Dataset utilizado.
* Tecnologías y librerías utilizadas.
* Estructura del proyecto y cómo ejecutarlo.
* Exploración de los datos.
* Proceso de preprocesamiento.
* Fundamentos de álgebra lineal aplicados.
* Operaciones con vectores y matrices.
* Resolución de sistemas de ecuaciones.
* Análisis estadístico.
* Cálculo de varianza y desviación estándar (indicando si es poblacional o muestral).
* Identificación de valores atípicos y decisión tomada sobre ellos.
* Impacto de la dispersión en el modelo.
* Modelo de Machine Learning.
* Entrenamiento y evaluación.
* Visualizaciones.
* Resultados obtenidos.
* Interpretación de los resultados.
* Limitaciones del análisis.
* Conclusiones.

---