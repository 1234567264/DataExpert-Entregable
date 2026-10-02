"""Requisito 2: preprocesamiento de datos sin fuga de información.

Orden correcto (obligatorio):
    1. Separar X (predictoras) e y (objetivo).
    2. Dividir en entrenamiento y prueba PRIMERO.
    3. Ajustar (`fit`) los transformadores solo con el entrenamiento.
    4. Aplicarlos (`transform`) al entrenamiento y a la prueba.

Ese orden evita la fuga de datos (*data leakage*): la imputación, el escalado
y la codificación aprenden parámetros (mediana, media, desviación, categorías)
del conjunto de entrenamiento y nunca ven los datos de prueba.

El preprocesamiento y el modelo viven dentro de un `Pipeline` con
`ColumnTransformer`, de modo que el `fit` solo ve entrenamiento (también
dentro de la validación cruzada) y el mismo objeto se puede guardar y reusar.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def umbrales_ingreso(df: pd.DataFrame) -> tuple[float, float]:
    """Calcula los dos cortes que dividen el ingreso per cápita en terciles.

    Args:
        df: DataFrame con la columna "Per capita GNI".

    Returns:
        Tupla (corte_inferior, corte_superior) en USD.

    Raises:
        ValueError: si falta la columna de ingreso.
    """
    if config.COL_INGRESO not in df.columns:
        raise ValueError(
            f"No se pueden calcular los umbrales: falta la columna '{config.COL_INGRESO}'."
        )
    ingreso = df[config.COL_INGRESO]
    return (
        float(ingreso.quantile(config.CUANTIL_INFERIOR)),
        float(ingreso.quantile(config.CUANTIL_SUPERIOR)),
    )


def crear_variable_objetivo(df: pd.DataFrame) -> pd.Series:
    """Deriva la variable objetivo de clasificación a partir del ingreso per cápita.

    Divide "Per capita GNI" en tres terciles (bajo / medio / alto). Se eligieron
    umbrales por cuantiles —y no cortes fijos en dólares— porque el dataset cubre
    1970-2021 en USD nominales: un corte fijo mezclaría épocas (mil dólares en
    970 era mucho; en 2021 es poco) y además los terciles dejan las tres clases
    perfectamente equilibradas (un tercio cada una).

    Args:
        df: DataFrame con la columna "Per capita GNI".

    Returns:
        Serie categórica ordenada con los valores "bajo", "medio" y "alto".

    Raises:
        ValueError: si no existe la columna de ingreso o todos sus valores son nulos.
    """
    if config.COL_INGRESO not in df.columns:
        raise ValueError(
            f"No se puede crear el objetivo: falta la columna '{config.COL_INGRESO}'."
        )

    ingreso = df[config.COL_INGRESO]
    if ingreso.notna().sum() == 0:
        raise ValueError(f"La columna '{config.COL_INGRESO}' solo contiene valores nulos.")

    corte_inferior, corte_superior = umbrales_ingreso(df)

    objetivo = pd.cut(
        ingreso,
        bins=[-float("inf"), corte_inferior, corte_superior, float("inf")],
        labels=list(config.CLASES),
        ordered=True,
    )
    objetivo.name = config.TARGET
    return objetivo


def columnas_predictoras(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa las columnas predictoras en numéricas y categóricas.

    Excluye las columnas documentadas en `config.COLUMNAS_EXCLUIDAS`
    (fuga de información, alta cardinalidad o identificadores).

    Args:
        df: DataFrame completo.

    Returns:
        Tupla (columnas_numéricas, columnas_categóricas).

    Raises:
        ValueError: si no queda ninguna predictora tras las exclusiones.
    """
    candidatas = [c for c in df.columns if c not in config.COLUMNAS_EXCLUIDAS]
    candidatas = [c for c in candidatas if c != config.TARGET]

    categoricas = [c for c in candidatas if c in config.COLUMNAS_CATEGORICAS]
    numericas = [c for c in candidatas if c not in config.COLUMNAS_CATEGORICAS]

    if not numericas and not categoricas:
        raise ValueError("No queda ninguna columna predictora tras aplicar las exclusiones.")
    return numericas, categoricas


def construir_preprocesador(
    numericas: list[str],
    categoricas: list[str],
    escalar: bool = True,
) -> ColumnTransformer:
    """Crea el `ColumnTransformer` con imputación, escalado y codificación.

    * Numéricas: imputación por mediana + `StandardScaler` (si `escalar=True`).
    * Categóricas: imputación por moda + `OneHotEncoder(handle_unknown="ignore")`.

    Los parámetros solo se ajustan con `fit` en entrenamiento (dentro del
    `Pipeline`), nunca con datos de prueba.

    Args:
        numericas: nombres de las columnas numéricas.
        categoricas: nombres de las columnas categóricas.
        escalar: si es False se omite el escalado (para el experimento 10.1).

    Returns:
        Transformador listo para usar dentro de un `Pipeline`.
    """
    tuberia_numerica = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        tuberia_numerica.append(("escalar", StandardScaler()))

    transformadores: list[tuple[str, Pipeline | SimpleImputer, list[str]]] = [
        ("num", Pipeline(tuberia_numerica), numericas),
    ]
    if categoricas:
        transformadores.append((
            "cat",
            Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("codificar", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas,
        ))

    return ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=False)


def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepara los datos para el modelo (requisito 2).

    Elimina duplicados, crea el objetivo, separa X e y, divide en
    entrenamiento/prueba con estratificación y devuelve los conjuntos ya
    separados. El ajuste de transformadores ocurre después, dentro del
    `Pipeline` de `modelo.py`, y solo ve el conjunto de entrenamiento.

    Args:
        df: DataFrame original con los datos ya explorados.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: si el DataFrame está vacío o no se puede crear el objetivo.
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío: no se puede dividir.")

    print("=" * 78)
    print("PASO 5 · PREPROCESAMIENTO (sin fuga de datos)")
    print("=" * 78)

    duplicados = int(df.duplicated().sum())
    if duplicados:
        df = df.drop_duplicates()
    print(f"Registros duplicados eliminados: {duplicados}")

    objetivo = crear_variable_objetivo(df)
    corte_inferior, corte_superior = umbrales_ingreso(df)
    print(f"Objetivo '{config.TARGET}' = terciles de {config.COL_INGRESO}: "
          f"bajo <= {corte_inferior:,.2f} USD < medio <= {corte_superior:,.2f} USD < alto")

    columnas_excluidas_presentes = [c for c in config.COLUMNAS_EXCLUIDAS if c in df.columns]
    X = df.drop(columns=columnas_excluidas_presentes + [config.TARGET], errors="ignore")
    y = objetivo.loc[X.index]

    # Filas sin objetivo (nulos en la variable de ingreso) no sirven para supervisado
    mask = y.notna()
    if not bool(mask.all()):
        print(f"Registros descartados por nulos en el objetivo: {int((~mask).sum())}")
        X, y = X[mask], y[mask]

    numericas, categoricas = columnas_predictoras(df)
    print(f"Predictoras excluidas ({len(columnas_excluidas_presentes)}): "
          f"{', '.join(columnas_excluidas_presentes)}")
    print(f"Predictoras numéricas ({len(numericas)}): {', '.join(numericas)}")
    print(f"Predictoras categóricas ({len(categoricas)}): {', '.join(categoricas) or 'ninguna'}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # clasificación: conserva la proporción de clases
    )

    print(f"\nDivisión: {len(X_train)} registros de entrenamiento "
          f"({100 * (1 - config.TEST_SIZE):.0f} %) y {len(X_test)} de prueba "
          f"({100 * config.TEST_SIZE:.0f} %); semilla={config.RANDOM_STATE}")

    # Diagnóstico de estructura de panel: cada país aparece en muchos años, así
    # que un corte aleatorio deja el mismo país en ambos conjuntos y sus filas
    # son casi idénticas entre años consecutivos. Se documenta como limitación.
    if "Country" in df.columns:
        paises_entrenamiento = set(df.loc[X_train.index, "Country"])
        vistos_tambien_en_train = df.loc[X_test.index, "Country"].isin(paises_entrenamiento)
        print(f"Estructura de panel: {100 * float(vistos_tambien_en_train.mean()):.1f} % de los "
              "registros de prueba corresponden a un país que también está en entrenamiento "
              "(misma economía observada en otros años).")

    print("\n-- Distribución de clases --")
    for nombre, serie in (("train", y_train), ("test", y_test)):
        conteos = serie.value_counts().reindex(config.CLASES)
        total = int(conteos.sum())
        detalle = ", ".join(
            f"{clase}={int(cantidad)} ({100 * cantidad / total:.1f} %)"
            for clase, cantidad in conteos.items()
        )
        print(f"  {nombre:<5} -> {detalle}")

    print("\nOrden aplicado: separar X/y -> dividir PRIMERO -> ajustar transformadores "
          "solo en el entrenamiento (dentro del Pipeline) -> transformar ambos.\n")

    return X_train, X_test, y_train, y_test
