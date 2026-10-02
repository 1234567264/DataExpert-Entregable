"""Visualización de datos y resultados (requisitos 11-14).

Todas las figuras se guardan en `reports/figures/` con `dpi=150` y
`bbox_inches="tight"`, y se cierran con `plt.close()` para no acumular figuras
en memoria. Se usa el backend `Agg` (sin pantalla) para que funcione en
cualquier entorno, incluidos servidores sin monitor.

Titular, ejes y unidades siempre en español.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla: debe ir ANTES de importar pyplot

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from . import config, preprocesamiento  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")

# Nombre legible de cada modelo entrenado
ETIQUETAS: dict[str, str] = {
    "knn": "KNN",
    "regresion_logistica": "Regresión logística",
}

# Variables elegidas para los gráficos (se comprobaron en la exploración)
PREDICTORES_HISTOGRAMA: tuple[str, ...] = (
    "Population",
    "Exports of goods and services",
    "Total Value Added",
    "Household consumption expenditure (including Non-profit institutions serving households)",
)

VARIABLES_DISPERSION: tuple[str, ...] = (
    "Per capita GNI",
    "Population",
    "Exports of goods and services",
    "Total Value Added",
)


def guardar_figura(nombre: str, figura: Figure | None = None) -> Path:
    """Guarda la figura en `reports/figures/` y la cierra.

    Args:
        nombre: nombre del archivo PNG (p. ej. "histograma_ingreso.png").
        figura: figura a guardar; si es None se usa la figura activa.

    Returns:
        Ruta completa donde se guardó la figura.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.FIGURES_DIR / nombre
    figura_actual = figura if figura is not None else plt.gcf()
    figura_actual.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura_actual)
    print(f"  figura guardada -> {ruta.relative_to(config.BASE_DIR)}")
    return ruta


def histograma_ingreso(df: pd.DataFrame) -> Path:
    """Histograma del ingreso per cápita con media y mediana (req. 11 y 13)."""
    serie = df[config.COL_INGRESO].dropna()

    plt.figure(figsize=(9, 5))
    sns.histplot(serie, bins=40, kde=True, color="steelblue")
    plt.axvline(serie.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Media: {serie.mean():,.0f} USD".replace(",", "."))
    plt.axvline(serie.median(), color="green", linestyle="-", linewidth=2,
                label=f"Mediana: {serie.median():,.0f} USD".replace(",", "."))
    rango = f"({int(df['Year'].min())}–{int(df['Year'].max())})" if "Year" in df.columns else ""
    plt.title(f"Distribución del ingreso per cápita {rango}")
    plt.xlabel("PNB per cápita (USD corrientes)")
    plt.ylabel("Número de registros (país-año)")
    plt.legend()
    plt.tight_layout()
    return guardar_figura("histograma_ingreso_per_capita.png")


def histogramas_predictores(df: pd.DataFrame) -> Path:
    """Histogramas de predictores numéricas con media y mediana (req. 11 y 13)."""
    candidatas = [c for c in df.columns if c in PREDICTORES_HISTOGRAMA][:4]
    fig, ejes = plt.subplots(2, 2, figsize=(12, 8))

    for eje, columna in zip(ejes.ravel(), candidatas):
        serie = df[columna].dropna()
        sns.histplot(serie, bins=40, kde=True, color="slategray", ax=eje)
        eje.axvline(serie.mean(), color="red", linestyle="--",
                    label=f"Media: {serie.mean():,.0f}".replace(",", "."))
        eje.axvline(serie.median(), color="green", linestyle="-",
                    label=f"Mediana: {serie.median():,.0f}".replace(",", "."))
        eje.set_title(columna[:52], fontsize=10)
        eje.set_xlabel("Valor (USD corrientes)")
        eje.set_ylabel("Frecuencia")
        eje.legend(fontsize=8)

    fig.suptitle("Distribución de predictores numéricas (media vs mediana)")
    fig.tight_layout()
    return guardar_figura("histogramas_predictores.png", fig)


def matriz_correlacion(df: pd.DataFrame) -> Path:
    """Heatmap de correlación de las 12 numéricas más ligadas al ingreso (req. 12)."""
    numericas = df.select_dtypes(include="number")
    correlaciones = numericas.corr()
    objetivo = correlaciones[config.COL_INGRESO].drop(config.COL_INGRESO).abs()
    seleccion = objetivo.sort_values(ascending=False).head(12).index.tolist()

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        correlaciones.loc[seleccion, seleccion],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlación de Pearson"},
    )
    plt.title("Correlación de Pearson entre las 12 variables numéricas\n"
              "más relacionadas con el ingreso per cápita")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    return guardar_figura("matriz_correlacion.png")


def dispersion(df: pd.DataFrame, tamano_muestra: int = 1500) -> Path:
    """Pairplot de dispersión coloreado por nivel de ingreso (req. 12).

    Args:
        df: DataFrame original.
        tamano_muestra: tamaño de la muestra aleatoria (con 10 mil filas el
            pairplot es lento y no aporta más información).

    Returns:
        Ruta de la figura guardada.
    """
    parejas = [c for c in VARIABLES_DISPERSION if c in df.columns]
    objetivo = preprocesamiento.crear_variable_objetivo(df)
    disponibles = df[parejas].dropna()
    muestra = disponibles.sample(n=min(tamano_muestra, len(disponibles)),
                                 random_state=config.RANDOM_STATE).copy()
    muestra[config.TARGET] = objetivo.loc[muestra.index]

    figura = sns.pairplot(
        muestra,
        vars=parejas,
        hue=config.TARGET,
        hue_order=list(config.CLASES),
        corner=True,
        height=2.6,
        plot_kws={"alpha": 0.5, "s": 18},
    )
    figura.fig.suptitle(
        "Dispersión por pares de variables numéricas, coloreadas por nivel de ingreso "
        f"(muestra de {len(muestra)} registros)",
        y=1.02,
    )
    return guardar_figura("dispersion_pairplot.png", figura.fig)


def boxplot_atipicos(df: pd.DataFrame, atipicos: pd.DataFrame) -> Path:
    """Boxplots de variables estandarizadas con atípicos resaltados (req. 10 y 13).

    Se grafican valores Z (media 0, desviación 1) para que todas las variables
    quepan en la misma escala; los puntos fuera de los bigotes se pintan en
    rojo y las líneas punteadas marcan el umbral |Z| = 3 usado para detectarlos.
    """
    numericas = df.select_dtypes(include="number").drop(
        columns=[c for c in config.COLUMNAS_NO_ATIPICAS if c in df.columns],
        errors="ignore",
    )
    z = (numericas - numericas.mean()) / numericas.std(ddof=0)
    largo = z.melt(var_name="variable", value_name="valor_z").dropna()

    plt.figure(figsize=(11, 9))
    sns.boxplot(
        data=largo,
        y="variable",
        x="valor_z",
        orient="h",
        color="lightsteelblue",
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "crimson", "alpha": 0.6},
    )
    plt.axvline(-config.Z_UMBRAL, color="red", linestyle="--", linewidth=1.5,
                label=f"Umbral |Z| = {config.Z_UMBRAL:g}")
    plt.axvline(config.Z_UMBRAL, color="red", linestyle="--", linewidth=1.5)
    plt.title("Valores atípicos por puntuación Z (variables estandarizadas)")
    plt.xlabel("Puntuación Z (desviaciones estándar respecto a la media)")
    plt.ylabel("Variable numérica")
    plt.legend()
    plt.tight_layout()
    ruta = guardar_figura("boxplot_valores_atipicos.png")
    print(f"  atípicos detectados y representados: {len(atipicos)}")
    return ruta


def distribucion_clases(df: pd.DataFrame) -> Path:
    """Barras con la frecuencia de cada clase del objetivo (req. 14)."""
    objetivo = preprocesamiento.crear_variable_objetivo(df).dropna()
    conteos = objetivo.value_counts().reindex(config.CLASES).astype(int)
    total = int(conteos.sum())

    plt.figure(figsize=(7, 5))
    plt.bar([str(c) for c in conteos.index], conteos.values, color=sns.color_palette("viridis", 3))
    for posicion, valor in enumerate(conteos.values):
        plt.text(posicion, valor + total * 0.01,
                 f"{valor}\n({100 * valor / total:.1f} %)", ha="center", fontsize=10)
    plt.title("Distribución de la variable objetivo (balance de clases)")
    plt.xlabel("Nivel de ingreso (terciles del PNB per cápita)")
    plt.ylabel("Número de registros (país-año)")
    plt.ylim(0, conteos.max() * 1.18)
    plt.tight_layout()
    return guardar_figura("distribucion_clases.png")


def matriz_confusion(resultados: dict[str, object], nombre: str) -> Path:
    """Matriz de confusión de un modelo sobre el conjunto de prueba (req. 14).

    Args:
        resultados: diccionario devuelto por `modelo.entrenar_y_evaluar`.
        nombre: clave del modelo ("knn" o "regresion_logistica").

    Returns:
        Ruta de la figura guardada.
    """
    modelo = resultados["modelos"][nombre]
    etiquetas = resultados["clases"]

    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        resultados["y_test"],
        modelo["predicciones_test"],
        labels=etiquetas,
        cmap="Blues",
        colorbar=False,
        ax=plt.gca(),
    )
    plt.title(f"Matriz de confusión en prueba — {ETIQUETAS[nombre]}")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    return guardar_figura(f"matriz_confusion_{nombre}.png")


def metricas_train_vs_test(resultados: dict[str, object]) -> Path:
    """Compara accuracy, precision, recall y F1 en train y test (req. 4 y 14)."""
    metricas = ["accuracy", "precision", "recall", "f1"]

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for eje, (clave, etiqueta) in zip(ejes, ETIQUETAS.items()):
        modelo = resultados["modelos"][clave]
        train = [modelo["metricas_train"][m] for m in metricas]
        test = [modelo["metricas_test"][m] for m in metricas]
        posiciones = list(range(len(metricas)))

        eje.bar([p - 0.2 for p in posiciones], train, width=0.4,
                label="Entrenamiento", color="steelblue")
        eje.bar([p + 0.2 for p in posiciones], test, width=0.4,
                label="Prueba", color="darkorange")
        eje.set_xticks(posiciones)
        eje.set_xticklabels([m.capitalize() for m in metricas])
        eje.set_ylim(0, 1.08)
        eje.set_title(etiqueta)
        eje.set_ylabel("Puntuación (0-1)")
        for posicion, valor in zip(posiciones, test):
            eje.text(posicion + 0.2, valor + 0.02, f"{valor:.3f}", ha="center", fontsize=8)
        eje.legend()

    fig.suptitle("Rendimiento en entrenamiento frente a prueba (detectar sobreajuste)")
    fig.tight_layout()
    return guardar_figura("metricas_train_vs_test.png", fig)


def experimento_metricas(resultados: dict[str, object]) -> Path:
    """Barras del experimento 10.1: mismo modelo sin escalar vs escalado."""
    experimento = resultados["experimento"]
    metricas_disponibles = {**experimento["combinaciones"], **experimento["solo_numericas"]}
    etiquetas = {
        "knn_sin_escalar": "KNN\nsin escalar",
        "knn_escalado": "KNN\nescalado",
        "knn_escalado_solo_numericas": "KNN escalado\nsolo numéricas",
        "regresion_logistica_sin_escalar": "Reg. logística\nsin escalar",
        "regresion_logistica_escalado": "Reg. logística\nescalado",
        "regresion_logistica_escalado_solo_numericas": "Reg. logística escal.\nsolo numéricas",
    }
    claves = [c for c in etiquetas if c in metricas_disponibles]
    accuracy = [metricas_disponibles[c]["accuracy"] for c in claves]
    f1 = [metricas_disponibles[c]["f1"] for c in claves]
    posiciones = list(range(len(claves)))

    plt.figure(figsize=(13, 5))
    plt.bar([p - 0.2 for p in posiciones], accuracy, width=0.4,
            label="Accuracy", color="steelblue")
    plt.bar([p + 0.2 for p in posiciones], f1, width=0.4, label="F1 (macro)", color="seagreen")
    plt.xticks(posiciones, [etiquetas[c] for c in claves], fontsize=8)
    plt.ylim(0, 1.08)
    plt.ylabel("Puntuación en prueba (0-1)")
    plt.title("Impacto de la dispersión: mismo modelo con y sin escalado")
    for posicion, valor in zip(posiciones, accuracy):
        plt.text(posicion - 0.2, valor + 0.02, f"{valor:.3f}", ha="center", fontsize=8)
    for posicion, valor in zip(posiciones, f1):
        plt.text(posicion + 0.2, valor + 0.02, f"{valor:.3f}", ha="center", fontsize=8)
    plt.legend()
    plt.tight_layout()
    return guardar_figura("experimento_dispersion_metricas.png")


def experimento_desviaciones(resultados: dict[str, object]) -> Path:
    """Desviación estándar de cada variable numérica: la causa del efecto 10.1."""
    desviaciones = resultados["experimento"]["desviaciones_estandar"]
    variables = [nombre[:46] for nombre in desviaciones][::-1]
    valores = list(desviaciones.values())[::-1]

    plt.figure(figsize=(9, 7))
    plt.barh(variables, valores, color="slategray")
    plt.xscale("log")
    plt.xlabel("Desviación estándar (USD corrientes, escala logarítmica, ddof=0)")
    plt.ylabel("Variable numérica")
    plt.title("Dispersión de cada variable predictora (entrenamiento)")
    plt.tight_layout()
    return guardar_figura("experimento_dispersion_desviaciones.png")


def generar_todas(
    df: pd.DataFrame,
    atipicos: pd.DataFrame,
    resultados: dict[str, object],
) -> list[Path]:
    """Genera y guarda todas las figuras del proyecto (requisitos 11-14).

    Args:
        df: DataFrame original con los datos.
        atipicos: DataFrame de atípicos detectados por puntuación Z.
        resultados: diccionario devuelto por `modelo.entrenar_y_evaluar`.

    Returns:
        Lista de rutas de las figuras generadas.
    """
    print("=" * 78)
    print("PASO 7 · VISUALIZACIONES")
    print("=" * 78)

    rutas = [
        histograma_ingreso(df),
        histogramas_predictores(df),
        matriz_correlacion(df),
        dispersion(df),
        boxplot_atipicos(df, atipicos),
        distribucion_clases(df),
        *(matriz_confusion(resultados, nombre) for nombre in ETIQUETAS),
        metricas_train_vs_test(resultados),
        experimento_metricas(resultados),
        experimento_desviaciones(resultados),
    ]
    print(f"Total de figuras generadas: {len(rutas)}\n")
    return rutas
