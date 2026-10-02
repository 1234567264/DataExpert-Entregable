"""Constantes globales del proyecto.

Aquí viven rutas, semillas y umbrales para no tener números mágicos repartidos
por el código (criterio de calidad de TAREA.md). Todas las rutas se construyen
con `pathlib.Path` para que funcionen igual en Windows, Linux o macOS.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto (portables: nunca se escriben a mano con \ o /)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
INFORME_PATH = BASE_DIR / "reports" / "informe.md"

# ---------------------------------------------------------------------------
# Reproducibilidad y división de los datos
# ---------------------------------------------------------------------------
RANDOM_STATE = 42  # semilla fija: mismos resultados en cada ejecución
TEST_SIZE = 0.2    # 20 % de los registros se reserva para prueba
CV_FOLDS = 5       # pliegues de la validación cruzada

# ---------------------------------------------------------------------------
# Estadística y detección de atípicos
# ---------------------------------------------------------------------------
Z_UMBRAL = 3.0  # se consideran atípicos los valores con |Z| > 3

# ---------------------------------------------------------------------------
# Variable objetivo (se fija DESPUÉS de explorar el dataset, no antes)
# ---------------------------------------------------------------------------
# Tras explorar data/data.csv se decidió un problema de CLASIFICACIÓN:
# predecir el nivel de ingreso de un país derivando terciles de "Per capita GNI".
TARGET = "Nivel de ingreso"                 # nombre de la columna objetivo
COL_INGRESO = "Per capita GNI"              # columna de la que se deriva
CLASES = ("bajo", "medio", "alto")          # orden de las tres clases
CUANTIL_INFERIOR = 1 / 3                    # corte entre bajo y medio
CUANTIL_SUPERIOR = 2 / 3                    # corte entre medio y alto

# ---------------------------------------------------------------------------
# Selección de predictores (decisiones tomadas tras la exploración)
# ---------------------------------------------------------------------------
COLUMNAS_EXCLUIDAS: tuple[str, ...] = (
    "Per capita GNI",  # fuga directa de información: es la definición del objetivo
    "Gross National Income(GNI) in USD",  # identidad aritmética: GNI / población ≈ objetivo
    "Gross Domestic Product (GDP)",  # proxy del mismo constructo (corr. 0.991 vía GDP/población)
    "Country",   # alta cardinalidad (220 valores únicos)
    "Currency",  # alta cardinalidad (153 valores únicos)
    "CountryID", # identificador del país, no una magnitud
)

# "Year" se trata como categórica (efectos fijos de año, codificados con one-hot)
COLUMNAS_CATEGORICAS: tuple[str, ...] = ("Year",)

# Identificadores/códigos que se excluyen del análisis de atípicos porque no
# representan magnitudes sino posiciones o códigos.
COLUMNAS_NO_ATIPICAS: tuple[str, ...] = ("CountryID", "Year")

# ---------------------------------------------------------------------------
# Hiperparámetros de los modelos (evita números mágicos en modelo.py)
# ---------------------------------------------------------------------------
N_VECINOS = 5        # k por defecto del clasificador KNN
MAX_ITER_LOGREG = 10000  # iteraciones máximas de la regresión logística (evita ConvergenceWarning en CV)

# Brecha máxima entre accuracy de entrenamiento y de prueba por debajo de la
# cual NO se considera sobreajuste.
UMBRAL_SOBREAJUSTE = 0.05
