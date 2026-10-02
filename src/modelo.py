"""Requisitos 3, 4 y 10.1: entrenamiento, evaluación y experimento de dispersión.

Se entrena un clasificador KNN (basado en distancias) y una regresión logística
(basada en optimización), ambos con el preprocesamiento dentro de un `Pipeline`
para que la imputación, el escalado y la codificación solo se ajusten con datos
de entrenamiento. Se comparan métricas de entrenamiento y prueba para detectar
sobreajuste y se añade validación cruzada como estimación más estable.

Métricas: como el problema es de CLASIFICACIÓN se usan accuracy, precision,
recall y F1 (promedio macro) más la matriz de confusión. Nunca se mezclan con
métricas de regresión (MAE, RMSE, R²).

El requisito 10.1 implementa el experimento de dispersión: el mismo modelo
entrenado sin escalar y con `StandardScaler` (ajustado solo en entrenamiento),
comparando las métricas y relacionándolas con la desviación estándar de cada
variable, más un diagnóstico de qué depende la distancia del KNN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline

from . import config, preprocesamiento

MODELOS_CONOCIDOS = ("knn", "regresion_logistica")

ETIQUETAS_MODELO = {
    "knn": "KNN",
    "regresion_logistica": "Regresión logística",
}


def _estimador(nombre: str):
    """Crea el clasificador de sklearn correspondiente al nombre pedido.

    Args:
        nombre: "knn" o "regresion_logistica".

    Returns:
        Instancia del clasificador con sus hiperparámetros de `config`.

    Raises:
        ValueError: si el nombre del modelo no está soportado.
    """
    if nombre == "knn":
        return KNeighborsClassifier(n_neighbors=config.N_VECINOS)
    if nombre == "regresion_logistica":
        return LogisticRegression(
            max_iter=config.MAX_ITER_LOGREG,
            random_state=config.RANDOM_STATE,
            solver="saga",  # mejor convergencia para multinomial con muchas features (one-hot de Year)
        )
    raise ValueError(
        f"Modelo desconocido: '{nombre}'. Opciones válidas: {', '.join(MODELOS_CONOCIDOS)}."
    )


def construir_modelo(X: pd.DataFrame, nombre: str, escalar: bool = True) -> Pipeline:
    """Construye el `Pipeline` completo (preprocesamiento + clasificador).

    Args:
        X: DataFrame de predictores (solo para conocer las columnas).
        nombre: "knn" o "regresion_logistica".
        escalar: si es False se omite `StandardScaler` (experimento 10.1).

    Returns:
        Pipeline listo para `fit`/`predict`.

    Raises:
        ValueError: si el nombre del modelo no está soportado.
    """
    _estimador(nombre)  # valida el nombre antes de montar nada

    # A partir de aquí X solo contiene predictoras: las categóricas se
    # identifican por nombre (config) y el resto se trata como numéricas.
    categoricas = [c for c in X.columns if c in config.COLUMNAS_CATEGORICAS]
    numericas = [c for c in X.columns if c not in config.COLUMNAS_CATEGORICAS]

    preprocesador = preprocesamiento.construir_preprocesador(numericas, categoricas, escalar=escalar)
    return Pipeline([("preprocesador", preprocesador), ("clasificador", _estimador(nombre))])


def _metricas(y_real: pd.Series, y_predicho: np.ndarray) -> dict[str, float]:
    """Calcula las métricas de clasificación con promedio macro.

    Args:
        y_real: etiquetas verdaderas.
        y_predicho: etiquetas predichas.

    Returns:
        Diccionario con accuracy, precision, recall y f1 (todas entre 0 y 1).
    """
    return {
        "accuracy": float(accuracy_score(y_real, y_predicho)),
        "precision": float(precision_score(y_real, y_predicho, average="macro", zero_division=0)),
        "recall": float(recall_score(y_real, y_predicho, average="macro", zero_division=0)),
        "f1": float(f1_score(y_real, y_predicho, average="macro", zero_division=0)),
    }


def _imprimir_metricas(titulo: str, metricas: dict[str, float]) -> None:
    """Imprime un bloque de métricas con formato tabular en español.

    Args:
        titulo: encabezado del bloque.
        metricas: diccionario nombre -> valor (0-1).
    """
    print(f"\n  {titulo}")
    print("  " + "-" * 46)
    for nombre, valor in metricas.items():
        print(f"  {nombre:<12} {valor:>10.4f}  ({100 * valor:5.2f} %)")


def _denso(matriz) -> np.ndarray:
    """Convierte la salida del preprocesador (a veces dispersa) en ndarray.

    El `ColumnTransformer` devuelve matrices dispersas cuando el bloque
    one-hot es poco denso; para medir distancias necesitamos una matriz densa.

    Args:
        matriz: matriz devuelta por `ColumnTransformer.transform`.

    Returns:
        Matriz NumPy densa de dos dimensiones.
    """
    if hasattr(matriz, "toarray"):
        matriz = matriz.toarray()
    return np.asarray(matriz)


def _diagnostico_vecinos(X_train: pd.DataFrame, X_test: pd.DataFrame, escalar: bool) -> dict[str, float]:
    """Mide de qué depende la distancia del KNN: magnitud vs bloque one-hot.

    Calcula, para el vecino más cercano de cada registro de prueba, la
    distancia² media, qué fracción de ella aporta el bloque one-hot de las
    categóricas y cuántos vecinos caen en el mismo año. Sirve para explicar
    por qué escalar puede cambiar el resultado del KNN aunque la teoría diga
    que estandarizar siempre ayuda.

    Args:
        X_train: predictores de entrenamiento.
        X_test: predictores de prueba.
        escalar: si es True usa el preprocesador con `StandardScaler`.

    Returns:
        Diccionario con `distancia2_media`, `peso_bloque_categorico` y
        `vecinos_mismo_anio`.
    """
    pipeline = construir_modelo(X_train, "knn", escalar=escalar)
    transformador = pipeline.named_steps["preprocesador"]
    transformador.fit(X_train)  # solo con entrenamiento: el test únicamente transforma
    Z_train = _denso(transformador.transform(X_train))
    Z_test = _denso(transformador.transform(X_test))
    nombres = [str(c) for c in transformador.get_feature_names_out()]

    vecinos = NearestNeighbors(n_neighbors=1).fit(Z_train).kneighbors(Z_test, return_distance=False).ravel()
    distancias = (Z_test - Z_train[vecinos]) ** 2
    total = float(distancias.sum())

    columnas_categoricas = [
        indice for indice, nombre in enumerate(nombres)
        if nombre.split("_")[0] in config.COLUMNAS_CATEGORICAS
    ]
    anios_train = X_train["Year"].to_numpy() if "Year" in X_train.columns else None
    anios_test = X_test["Year"].to_numpy() if "Year" in X_test.columns else None

    return {
        "distancia2_media": total / len(Z_test),
        "peso_bloque_categorico": (
            float(distancias[:, columnas_categoricas].sum()) / total if columnas_categoricas and total else 0.0
        ),
        "vecinos_mismo_anio": (
            float(np.mean(anios_train[vecinos] == anios_test))
            if anios_train is not None and anios_test is not None
            else float("nan")
        ),
    }


def experimento_dispersion(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Requisito 10.1: mide el impacto de la dispersión (escalado) en el modelo.

    Entrena el MISMO modelo en cuatro combinaciones (KNN y regresión logística,
    cada uno sin escalar y con `StandardScaler` ajustado solo en entrenamiento)
    y devuelve las métricas de prueba de cada una junto con la desviación
    estándar de cada variable numérica. Añade dos diagnósticos: el mismo modelo
    escalado pero SOLO con numéricas (sin el bloque one-hot) y el análisis de
    la distancia del KNN.

    Por qué importa: en algoritmos basados en distancias como KNN, una variable
    con desviación estándar mucho mayor domina la distancia y las demás pierden
    influencia. Estandarizar (media 0, desviación 1) pone las variables en una
    escala comparable.

    Args:
        X_train: predictores de entrenamiento.
        X_test: predictores de prueba.
        y_train: objetivo de entrenamiento.
        y_test: objetivo de prueba.

    Returns:
        Diccionario con `combinaciones` (las 4 combinaciones del experimento),
        `solo_numericas` (diagnóstico), `diagnostico_vecinos`,
        `desviaciones_estandar` y `mejora_f1_al_escalar`.
    """
    combinaciones: dict[str, dict[str, float]] = {}
    for nombre in MODELOS_CONOCIDOS:
        for escalar, sufijo in ((False, "sin_escalar"), (True, "escalado")):
            pipeline = construir_modelo(X_train, nombre, escalar=escalar)
            pipeline.fit(X_train, y_train)
            predicciones = pipeline.predict(X_test)
            combinaciones[f"{nombre}_{sufijo}"] = _metricas(y_test, predicciones)

    # Diagnóstico: mismo modelo escalado, pero sin el bloque one-hot de Year.
    # Sirve para separar el efecto de escalar del efecto del codificador.
    numericas = [c for c in X_train.columns if c not in config.COLUMNAS_CATEGORICAS]
    solo_numericas: dict[str, dict[str, float]] = {}
    for nombre in MODELOS_CONOCIDOS:
        pipeline = Pipeline([
            ("preprocesador", preprocesamiento.construir_preprocesador(numericas, [], escalar=True)),
            ("clasificador", _estimador(nombre)),
        ])
        pipeline.fit(X_train, y_train)
        solo_numericas[f"{nombre}_escalado_solo_numericas"] = _metricas(y_test, pipeline.predict(X_test))

    columnas_categoricas = [c for c in X_train.columns if c in config.COLUMNAS_CATEGORICAS]
    numericas_df = X_train.select_dtypes(include="number").drop(
        columns=columnas_categoricas, errors="ignore"
    )
    desviaciones = (
        numericas_df.std(ddof=0)  # ddof=0 explícito: dispersión poblacional
        .sort_values(ascending=False)
        .to_dict()
    )

    mejoras = {
        nombre: combinaciones[f"{nombre}_escalado"]["f1"]
        - combinaciones[f"{nombre}_sin_escalar"]["f1"]
        for nombre in MODELOS_CONOCIDOS
    }

    return {
        "combinaciones": combinaciones,
        "solo_numericas": solo_numericas,
        "diagnostico_vecinos": {
            "crudo": _diagnostico_vecinos(X_train, X_test, escalar=False),
            "escalado": _diagnostico_vecinos(X_train, X_test, escalar=True),
        },
        "desviaciones_estandar": desviaciones,
        "mejora_f1_al_escalar": mejoras,
    }


def entrenar_y_evaluar(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Entrena los modelos, los evalúa y ejecuta el experimento de dispersión.

    Requisitos 3 (modelo), 4 (validación/evaluación) y 10.1 (impacto de la
    dispersión). Compara siempre entrenamiento vs prueba para poder detectar
    sobreajuste: si el rendimiento de train es muy superior al de test, el
    modelo ha memorizado y no generaliza. El modelo principal se elige con la
    validación cruzada (estimación más robusta que un solo corte de prueba).

    Args:
        X_train: predictores de entrenamiento.
        X_test: predictores de prueba.
        y_train: objetivo de entrenamiento.
        y_test: objetivo de prueba.

    Returns:
        Diccionario con los modelos entrenados (pipeline, métricas de train y
        test, predicciones y matriz de confusión), la validación cruzada de
        cada modelo, el experimento de dispersión, las clases y el modelo
        principal.
    """
    print("=" * 78)
    print("PASO 6 · ENTRENAMIENTO Y EVALUACIÓN DEL MODELO")
    print("=" * 78)
    print(f"Clasificación multiclase -> clases: {', '.join(config.CLASES)}")
    print(f"Modelos: KNN (k={config.N_VECINOS}) y regresión logística, "
          f"con imputación + escalado (numéricas) y one-hot (categóricas) dentro del Pipeline.")

    modelos: dict[str, dict[str, object]] = {}
    for nombre in MODELOS_CONOCIDOS:
        pipeline = construir_modelo(X_train, nombre, escalar=True)
        pipeline.fit(X_train, y_train)

        pred_train = pipeline.predict(X_train)
        pred_test = pipeline.predict(X_test)

        metricas_train = _metricas(y_train, pred_train)
        metricas_test = _metricas(y_test, pred_test)

        modelos[nombre] = {
            "pipeline": pipeline,
            "metricas_train": metricas_train,
            "metricas_test": metricas_test,
            "predicciones_train": pred_train,
            "predicciones_test": pred_test,
            "matriz_confusion": confusion_matrix(y_test, pred_test, labels=list(config.CLASES)),
        }

        print(f"\n-- {ETIQUETAS_MODELO[nombre]} --")
        _imprimir_metricas("Entrenamiento", metricas_train)
        _imprimir_metricas("Prueba", metricas_test)

        brecha = metricas_train["accuracy"] - metricas_test["accuracy"]
        veredicto = (
            "sin sobreajuste apreciable"
            if brecha < config.UMBRAL_SOBREAJUSTE
            else "posible sobreajuste: el modelo rinde mucho mejor en lo que ya vio"
        )
        print(f"  Brecha de accuracy train - test: {brecha:+.4f} -> {veredicto}")

    # Validación cruzada (opcional recomendada): estimación más estable
    validacion: dict[str, dict[str, object]] = {}
    for nombre in MODELOS_CONOCIDOS:
        puntajes = cross_val_score(
            construir_modelo(X_train, nombre, escalar=True),
            X_train,
            y_train,
            cv=config.CV_FOLDS,
            scoring="accuracy",
        )
        validacion[nombre] = {
            "pliegues": config.CV_FOLDS,
            "accuracy_por_pliegue": puntajes.tolist(),
            "accuracy_media": float(puntajes.mean()),
            "accuracy_desviacion": float(puntajes.std()),
        }
        print(f"\n-- Validación cruzada: {ETIQUETAS_MODELO[nombre]} "
              f"({config.CV_FOLDS} pliegues, sobre train) --")
        print("  " + ", ".join(f"{puntaje:.4f}" for puntaje in puntajes))
        print(f"  accuracy media = {puntajes.mean():.4f} ± {puntajes.std():.4f}")

    modelo_principal = max(validacion, key=lambda n: validacion[n]["accuracy_media"])
    print(f"\n  Modelo principal (mayor accuracy en validación cruzada): "
          f"{ETIQUETAS_MODELO[modelo_principal]}")

    experimento = experimento_dispersion(X_train, X_test, y_train, y_test)

    print("\n-- Experimento de dispersión (requisito 10.1): sin escalar vs escalado --")
    print(f"  {'combinación':<36} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    print("  " + "-" * 78)
    for clave, metricas in experimento["combinaciones"].items():
        print(f"  {clave:<36} {metricas['accuracy']:>10.4f} {metricas['precision']:>10.4f} "
              f"{metricas['recall']:>10.4f} {metricas['f1']:>10.4f}")

    print("\n  Diagnóstico: mismo modelo ESCALADO pero solo con variables numéricas "
          "(sin el bloque one-hot de Year):")
    for clave, metricas in experimento["solo_numericas"].items():
        print(f"  {clave:<36} {metricas['accuracy']:>10.4f} {metricas['precision']:>10.4f} "
              f"{metricas['recall']:>10.4f} {metricas['f1']:>10.4f}")

    print("\n  Desviación estándar de las variables numéricas (entrenamiento, ddof=0):")
    desviaciones_lista = list(experimento["desviaciones_estandar"].items())
    mayores, menores = desviaciones_lista[:6], desviaciones_lista[-3:]
    for variable, desviacion in mayores:
        print(f"    {variable[:60]:<60} {desviacion:>18,.4f}")
    print("    ...")
    for variable, desviacion in menores:
        print(f"    {variable[:60]:<60} {desviacion:>18,.4f}")
    print(f"    ({len(desviaciones_lista)} variables numéricas en total)")

    print("\n  Diagnóstico de la distancia del KNN (vecino más cercano de cada prueba):")
    print(f"    {'variante':<12} {'distancia² media':>20} {'peso bloque one-hot':>22} "
          f"{'vecinos mismo año':>20}")
    for variante, datos in experimento["diagnostico_vecinos"].items():
        print(f"    {variante:<12} {datos['distancia2_media']:>20.4f} "
              f"{datos['peso_bloque_categorico']:>22.4f} {datos['vecinos_mismo_anio']:>20.4f}")

    for nombre, mejora in experimento["mejora_f1_al_escalar"].items():
        print(f"  Mejora de F1 al escalar ({ETIQUETAS_MODELO[nombre]}): {mejora:+.4f}")
    print("  Interpretación: sin escalar, la distancia la dominan las variables con mayor "
          "desviación estándar (el bloque one-hot pesa 0); al escalar, las numéricas pesan "
          "igual y el bloque one-hot gana peso real en la distancia. Con el bloque one-hot "
          "fuera, escalar mejora a los dos modelos.\n")

    return {
        "modelos": modelos,
        "validacion_cruzada": validacion,
        "experimento": experimento,
        "clases": list(config.CLASES),
        "y_test": y_test,
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "modelo_principal": modelo_principal,
    }
