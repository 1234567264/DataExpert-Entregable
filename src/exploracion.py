"""Requisito 1: lectura e exploración inicial del dataset.

Este módulo solo observa los datos: los cambia en nada. Su trabajo es leer el
CSV, limpiar los nombres de columna (el dataset trae espacios al principio y
al final) y mostrar la estructura, tipos, nulos y duplicados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lee un archivo CSV y normaliza los nombres de sus columnas.

    Los nombres de data/data.csv llegan con espacios al inicio y al final
    (por ejemplo ``"Country "``); se recortan para trabajar cómodos.

    Args:
        ruta: ruta completa del archivo CSV.

    Returns:
        DataFrame con los datos leídos y los nombres de columna limpios.

    Raises:
        FileNotFoundError: si la ruta no existe o no apunta a un archivo.
        ValueError: si el CSV está vacío o no tiene columnas.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {ruta}. "
            "Comprueba que data/data.csv existe."
        )

    df = pd.read_csv(ruta)

    if df.empty or df.shape[1] == 0:
        raise ValueError(f"El archivo {ruta} está vacío o no tiene columnas.")

    df.columns = df.columns.str.strip()
    return df


def resumen(df: pd.DataFrame) -> dict[str, Any]:
    """Muestra un resumen inicial del dataset y devuelve sus cifras clave.

    Cubre el requisito 1: forma, nombres de variables, tipos de datos,
    valores faltantes, registros duplicados, cardinalidad y vista preliminar.

    Args:
        df: DataFrame ya cargado con los nombres de columna limpios.

    Returns:
        Diccionario con las cifras del resumen (forma, nulos, duplicados,
        cardinalidad y estadísticos de la variable de ingreso).
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío: no hay nada que explorar.")

    filas, columnas = df.shape
    nulos = df.isna().sum()
    columnas_con_nulos = nulos[nulos > 0]
    duplicados = int(df.duplicated().sum())

    print("=" * 78)
    print("PASO 1 · EXPLORACIÓN DE LOS DATOS")
    print("=" * 78)
    print(f"Forma del dataset (filas, columnas): {df.shape}")
    print(f"Memoria en uso: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    print("\n-- Variables (nombres ya recortados) --")
    for i, columna in enumerate(df.columns, start=1):
        print(f"  {i:>2}. {columna}")

    print("\n-- Tipos de datos --")
    for tipo, cantidad in df.dtypes.astype(str).value_counts().items():
        print(f"  {tipo}: {cantidad} columna(s)")
    print("\nDetalle de tipos:")
    for columna, tipo in df.dtypes.items():
        print(f"  {columna:<80} {tipo}")

    print("\n-- Valores faltantes --")
    if columnas_con_nulos.empty:
        print("  No hay valores nulos.")
    else:
        for columna, cantidad in columnas_con_nulos.items():
            print(f"  {columna:<80} {cantidad:>6} ({100 * cantidad / filas:5.2f} %)")
        print(f"  Total de celdas nulas: {int(nulos.sum())}")
        print(f"  Filas con al menos un nulo: {int(df.isna().any(axis=1).sum())}")

    print(f"\n-- Registros duplicados: {duplicados} --")

    print("\n-- Cardinalidad de las variables de identificación --")
    for columna in ("Country", "Currency", "CountryID", "Year"):
        if columna in df.columns:
            print(f"  {columna:<12} {df[columna].nunique()} valores únicos")

    ingreso = None
    if "Per capita GNI" in df.columns:
        ingreso = df["Per capita GNI"]
        print("\n-- Variable de ingreso per cápita (Per capita GNI, USD) --")
        print(f"  mín : {ingreso.min():>12,.0f}")
        print(f"  máx : {ingreso.max():>12,.0f}")
        print(f"  media: {ingreso.mean():>12,.2f}")
        print(f"  mediana: {ingreso.median():>12,.2f}")
        print(f"  desv. típica: {ingreso.std(ddof=1):>9,.2f}")
        print("  Cuartiles (USD):")
        for cuantil, valor in ingreso.quantile([0.25, 0.5, 0.75]).items():
            print(f"    Q{int(cuantil * 100):>2}: {valor:>12,.2f}")

    print("\n-- Vista preliminar (5 primeras filas) --")
    with pd.option_context("display.width", 160, "display.max_columns", 8):
        print(df.head().to_string(index=True))

    print(f"\nResumen: {filas} filas × {columnas} columnas, "
          f"{len(columnas_con_nulos)} columnas con nulos, {duplicados} duplicados.\n")

    return {
        "filas": filas,
        "columnas": columnas,
        "nulos_totales": int(nulos.sum()),
        "columnas_con_nulos": columnas_con_nulos.to_dict(),
        "filas_con_nulos": int(df.isna().any(axis=1).sum()),
        "duplicados": duplicados,
        "cardinalidad": {
            c: int(df[c].nunique()) for c in ("Country", "Currency", "CountryID", "Year")
            if c in df.columns
        },
        "ingreso": None if ingreso is None else ingreso.describe().to_dict(),
    }
