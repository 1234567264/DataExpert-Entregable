"""Requisitos 8, 9 y 10: análisis estadístico, varianza/desviación manuales y
detección de valores atípicos.

Las funciones `varianza` y `desviacion_estandar` se implementan a mano (sin
NumPy) para entender la fórmula y luego se comparan con NumPy usando `ddof`
explícito.

⚠️ Trampa de los `ddof` por defecto:
    | Función                  | ddof por defecto | Tipo de varianza |
    |--------------------------|------------------|------------------|
    | `np.var`, `np.std`       | 0                | poblacional (N)  |
    | `pd.Series.var/std`      | 1                | muestral (n - 1) |
Si se comparan sin fijar el parámetro, los resultados no coinciden y parece
un bug de la implementación. Siempre hay que pasar `ddof` de forma explícita.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from . import config


def _convertir_a_valores(datos: Iterable[float]) -> list[float]:
    """Convierte cualquier iterable en una lista de floats validada.

    Args:
        datos: iterable de valores numéricos.

    Returns:
        Lista de números en punto flotante.

    Raises:
        ValueError: si contiene valores no numéricos, NaN o infinitos.
    """
    try:
        valores = [float(x) for x in datos]
    except (TypeError, ValueError) as exc:
        raise ValueError("El conjunto de datos debe contener solo valores numéricos.") from exc

    if any(math.isnan(v) or math.isinf(v) for v in valores):
        raise ValueError("El conjunto de datos contiene valores NaN o infinitos.")

    return valores


def varianza(datos: Iterable[float], muestral: bool = False) -> float:
    """Calcula la varianza a mano, sin NumPy.

    Fórmulas:
        poblacional: σ² = Σ(xᵢ - μ)² / N          (muestral=False, por defecto)
        muestral:    s² = Σ(xᵢ - x̄)² / (n - 1)    (muestral=True)

    Args:
        datos: iterable con los valores numéricos.
        muestral: si es True divide entre n - 1; si es False entre N.

    Returns:
        Varianza calculada.

    Raises:
        ValueError: si la lista está vacía, si no contiene números, si
            contiene NaN/infinitos o si se pide varianza muestral con
            menos de 2 datos.
    """
    valores = _convertir_a_valores(datos)
    n = len(valores)

    if n == 0:
        raise ValueError("El conjunto de datos no puede estar vacío.")
    if muestral and n < 2:
        raise ValueError("La varianza muestral requiere al menos 2 datos.")

    media = sum(valores) / n
    suma_cuadrados = sum((x - media) ** 2 for x in valores)
    return suma_cuadrados / (n - 1 if muestral else n)


def desviacion_estandar(datos: Iterable[float], muestral: bool = False) -> float:
    """Calcula la desviación estándar a mano (raíz de la varianza).

    Args:
        datos: iterable con los valores numéricos.
        muestral: si es True usa el denominador n - 1 (muestra); si es False N.

    Returns:
        Desviación estándar calculada.

    Raises:
        ValueError: si procede de `varianza` (lista vacía, no numérica, etc.).
    """
    return varianza(datos, muestral=muestral) ** 0.5


def analizar(df: pd.DataFrame) -> dict[str, Any]:
    """Ejecuta el análisis estadístico de las variables numéricas (requisito 8).

    Calcula medidas de tendencia central, cuartiles y comparación manual de
    varianza/desviación frente a NumPy con `ddof` explícito.

    Args:
        df: DataFrame con los datos ya explorados.

    Returns:
        Diccionario con el `describe` por variable, la tabla de asimetría
        media/mediana y las varianzas de la variable de ingreso.

    Raises:
        ValueError: si el DataFrame no tiene variables numéricas.
    """
    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        raise ValueError("El DataFrame no contiene variables numéricas que analizar.")

    describe = numericas.describe().T[["count", "mean", "50%", "std", "min", "25%", "75%", "max"]]

    # Distancia media-mediana normalizada por la desviación: si es grande, la
    # distribución es asimétrica o tiene valores extremos.
    asimetria = pd.DataFrame({
        "media": numericas.mean(),
        "mediana": numericas.median(),
        "distancia_relativa": (numericas.mean() - numericas.median()).abs()
        / numericas.std(ddof=0).replace(0, np.nan),
    }).sort_values("distancia_relativa", ascending=False)

    print("=" * 78)
    print("PASO 2 · ANÁLISIS ESTADÍSTICO (variables numéricas)")
    print("=" * 78)
    print(f"Variables numéricas analizadas: {len(numericas.columns)}")

    print("\n-- Medidas de tendencia central: media vs mediana --")
    print("   (distancia_relativa = |media - mediana| / desviación; cuanto mayor, "
          "más asimétrica)")
    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:,.3f}"):
        print(asimetria.head(8).to_string())

    if "Per capita GNI" in numericas.columns:
        ingreso = numericas["Per capita GNI"].dropna()
        print("\n-- Cuartiles de Per capita GNI (USD) --")
        for cuantil, valor in ingreso.quantile([0.25, 0.5, 0.75]).items():
            print(f"   Q{int(cuantil * 100):>2} = {valor:,.2f} USD")

        valores = ingreso.tolist()
        v_pob_manual = varianza(valores, muestral=False)
        v_mue_manual = varianza(valores, muestral=True)
        d_pob_manual = desviacion_estandar(valores, muestral=False)

        print("\n-- Varianza y desviación manual vs NumPy (Per capita GNI) --")
        print(f"   Varianza POBLACIONAL manual : {v_pob_manual:,.4f}")
        print(f"   np.var(ddof=0)              : {np.var(valores, ddof=0):,.4f}  "
              f"-> coincide: {np.isclose(v_pob_manual, np.var(valores, ddof=0))}")
        print(f"   Varianza MUESTRAL manual    : {v_mue_manual:,.4f}")
        print(f"   np.var(ddof=1)              : {np.var(valores, ddof=1):,.4f}  "
              f"-> coincide: {np.isclose(v_mue_manual, np.var(valores, ddof=1))}")
        print(f"   Desviación POBLACIONAL manual: {d_pob_manual:,.4f}")
        print(f"   np.std(ddof=0)              : {np.std(valores, ddof=0):,.4f}  "
              f"-> coincide: {np.isclose(d_pob_manual, np.std(valores, ddof=0))}")
        print("   Nota: pd.Series.var() usa ddof=1 por defecto y np.var() usa 0; "
              "por eso se fija siempre el ddof.")

        varianza_info = {
            "poblacional_manual": v_pob_manual,
            "poblacional_numpy": float(np.var(valores, ddof=0)),
            "muestral_manual": v_mue_manual,
            "muestral_numpy": float(np.var(valores, ddof=1)),
            "desviacion_poblacional_manual": d_pob_manual,
            "desviacion_poblacional_numpy": float(np.std(valores, ddof=0)),
        }
    else:
        varianza_info = {}

    print("\n-- Estadísticos descriptivos completos (por variable) --")
    with pd.option_context("display.width", 200, "display.max_columns", 10,
                           "display.float_format", lambda v: f"{v:,.2f}"):
        print(describe.to_string())
    print()

    return {"describe": describe, "asimetria": asimetria, "varianza_ingreso": varianza_info}


def detectar_atipicos(df: pd.DataFrame, umbral: float = config.Z_UMBRAL) -> pd.DataFrame:
    """Detecta valores atípicos por puntuación Z (requisito 10).

    Z = (x - μ) / σ con σ calculada con ``ddof=0`` de forma explícita y
    consistente. Se considera atípico todo registro con |Z| > `umbral`.

    Se aplica SOLO a variables numéricas y se omiten identificadores/códigos
    (p. ej. CountryID o Year), que parecen números pero no son magnitudes.

    Limitaciones del método (para el informe):
      * los propios atípicos inflan la media y la desviación que se usan para
        detectarlos (un valor extremo puede "esconderse");
      * con muestras pequeñas ningún valor puede superar 3 desviaciones;
      * funciona mejor con distribuciones simétricas que con asimétricas;
      * no implica eliminar los registros: hay que juzgarlos caso a caso.

    Args:
        df: DataFrame con los datos.
        umbral: valor absoluto de Z a superar (por defecto `config.Z_UMBRAL`).

    Returns:
        DataFrame largo con columnas `variable`, `fila`, `valor` y `z`,
        ordenado por valor absoluto de Z descendente.

    Raises:
        ValueError: si el DataFrame está vacío o el umbral no es positivo.
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío: no se pueden detectar atípicos.")
    if umbral <= 0:
        raise ValueError(f"El umbral de la puntuación Z debe ser positivo (recibido: {umbral}).")

    numericas = df.select_dtypes(include="number")
    excluir = [c for c in config.COLUMNAS_NO_ATIPICAS if c in numericas.columns]
    numericas = numericas.drop(columns=excluir, errors="ignore")

    registros: list[dict[str, Any]] = []
    for columna in numericas.columns:
        serie = numericas[columna].dropna()
        if serie.empty:
            continue
        desviacion = float(serie.std(ddof=0))
        if desviacion == 0:
            continue  # sin dispersión no hay atípicos
        z = (serie - serie.mean()) / desviacion
        atipicos = serie[z.abs() > umbral]
        for indice, valor in atipicos.items():
            registros.append({
                "variable": columna,
                "fila": indice,
                "valor": float(valor),
                "z": float(z.loc[indice]),
            })

    resultado = (
        pd.DataFrame(registros, columns=["variable", "fila", "valor", "z"])
        .assign(abs_z=lambda t: t["z"].abs())
        .sort_values("abs_z", ascending=False)
        .drop(columns="abs_z")
        .reset_index(drop=True)
    )

    print("=" * 78)
    print(f"PASO 3 · VALORES ATÍPICOS (puntuación Z, |Z| > {umbral}, ddof=0)")
    print("=" * 78)
    print(f"Variables numéricas evaluadas: {len(numericas.columns)} "
          f"(excluidas como identificadores: {', '.join(excluir) or 'ninguna'})")
    print(f"Atípicos detectados: {len(resultado)}")

    if resultado.empty:
        print("No se detectó ningún valor atípico con este umbral.\n")
        return resultado

    por_variable = (
        resultado.groupby("variable")
        .agg(atípicos=("valor", "size"), z_máximo=("z", lambda s: s.abs().max()))
        .sort_values("atípicos", ascending=False)
    )
    print("\n-- Atípicos por variable (hasta 10) --")
    print(por_variable.head(10).to_string(float_format=lambda v: f"{v:,.3f}"))
    print("\n-- 5 atípicos más extremos --")
    print(resultado.head(5).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("  Decisión: NO se eliminan automáticamente; se analizan en el informe.\n")

    return resultado
