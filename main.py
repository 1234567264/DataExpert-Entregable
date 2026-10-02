"""Punto de entrada del proyecto: ejecuta cada etapa en orden.

No contiene lógica de negocio: solo importa las funciones de `src/` y las
llama en orden, pasándose los resultados de una etapa a la siguiente. Usa
`logging` (nunca `print`) para dejar constancia trazable de cada paso.
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Ejecuta las 7 etapas del proyecto en orden."""
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
