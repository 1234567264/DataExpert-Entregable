"""Pruebas de las funciones estadísticas y de álgebra lineal manuales.

Comprueban que los cálculos hechos a mano coinciden con NumPy usando `ddof`
explícito (poblacional ddof=0, muestral ddof=1) y los casos borde: lista
vacía, un solo dato con varianza muestral, valores negativos y comparaciones
con `np.isclose` (nunca con `==`, por el redondeo de punto flotante).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Permite importar `src` cuando pytest se ejecuta desde la raíz del proyecto
# sin instalarlo como paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algebra_lineal import (  # noqa: E402
    multiplicar_matrices,
    norma_euclidiana,
    producto_punto,
    resolver_sistema,
)
from src.estadistica import desviacion_estandar, varianza  # noqa: E402

# Dataset clásico de ejemplo: media 5, varianza poblacional 4
DATOS_EJEMPLO = [2, 4, 4, 4, 5, 5, 7, 9]


def test_varianza_ejemplo_tipico_poblacional() -> None:
    """La varianza poblacional de [2,4,4,4,5,5,7,9] es 4 y coincide con np.var(ddof=0)."""
    resultado = varianza(DATOS_EJEMPLO, muestral=False)
    assert np.isclose(resultado, 4.0)
    assert np.isclose(resultado, np.var(DATOS_EJEMPLO, ddof=0))


def test_varianza_ejemplo_tipico_muestral() -> None:
    """La varianza muestral divide entre n-1 y coincide con np.var(ddof=1)."""
    resultado = varianza(DATOS_EJEMPLO, muestral=True)
    esperado = 32 / 7  # 4.571428...
    assert np.isclose(resultado, esperado)
    assert np.isclose(resultado, np.var(DATOS_EJEMPLO, ddof=1))


def test_desviacion_estandar_poblacional() -> None:
    """La desviación poblacional es la raíz de la varianza poblacional."""
    resultado = desviacion_estandar(DATOS_EJEMPLO, muestral=False)
    assert np.isclose(resultado, 2.0)
    assert np.isclose(resultado, np.std(DATOS_EJEMPLO, ddof=0))


def test_desviacion_estandar_muestral() -> None:
    """La desviación muestral coincide con np.std(ddof=1)."""
    resultado = desviacion_estandar(DATOS_EJEMPLO, muestral=True)
    assert np.isclose(resultado, np.std(DATOS_EJEMPLO, ddof=1))


def test_ddof_distintos_no_coinciden() -> None:
    """Poblacional y muestral dan resultados distintos: por eso el ddof es explícito."""
    poblacional = varianza(DATOS_EJEMPLO, muestral=False)
    muestral = varianza(DATOS_EJEMPLO, muestral=True)
    assert not np.isclose(poblacional, muestral)
    assert np.isclose(muestral, poblacional * len(DATOS_EJEMPLO) / (len(DATOS_EJEMPLO) - 1))


def test_numpy_y_pandas_tienen_ddof_por_defecto_distinto() -> None:
    """np.var usa ddof=0 y pd.Series.var usa ddof=1: comparar sin fijarlo es un error."""
    serie = np.array(DATOS_EJEMPLO, dtype=float)
    assert not np.isclose(np.var(serie), np.var(serie, ddof=1))
    assert np.isclose(np.var(serie, ddof=0), np.var(serie))  # mismo por defecto


def test_varianza_lista_vacia() -> None:
    """Una lista vacía no tiene varianza: debe lanzar ValueError."""
    with pytest.raises(ValueError, match="vacío"):
        varianza([])


def test_varianza_muestral_un_solo_dato() -> None:
    """La varianza muestral con n = 1 no está definida: debe lanzar ValueError."""
    with pytest.raises(ValueError, match="al menos 2 datos"):
        varianza([7.0], muestral=True)


def test_varianza_poblacional_un_solo_dato_es_cero() -> None:
    """Con un solo dato la varianza poblacional es 0 (no es un error)."""
    assert varianza([7.0], muestral=False) == 0.0


def test_desviacion_estandar_lista_vacia() -> None:
    """La desviación hereda la validación de la varianza."""
    with pytest.raises(ValueError, match="vacío"):
        desviacion_estandar([])


def test_valores_negativos() -> None:
    """El cálculo manual también vale para valores negativos."""
    datos = [-3.0, -1.0, 1.0, 3.0]
    assert np.isclose(varianza(datos), np.var(datos, ddof=0))
    assert np.isclose(varianza(datos, muestral=True), np.var(datos, ddof=1))
    assert np.isclose(desviacion_estandar(datos), np.std(datos, ddof=0))


def test_valores_no_numericos() -> None:
    """Contaminar los datos con texto debe lanzar ValueError, no devolver basura."""
    with pytest.raises(ValueError, match="solo valores numéricos"):
        varianza([1.0, "dos", 3.0])


def test_valores_nan() -> None:
    """Un NaN en los datos invalida el cálculo manual."""
    with pytest.raises(ValueError, match="NaN"):
        varianza([1.0, float("nan"), 3.0])


def test_producto_punto_manual_vs_numpy() -> None:
    """El producto punto manual coincide con np.dot."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert np.isclose(producto_punto(a, b), np.dot(a, b))
    assert np.isclose(producto_punto(a, b), 32.0)


def test_producto_punto_dimensiones_distintas() -> None:
    """Vectores de distinta longitud no se pueden multiplicar."""
    with pytest.raises(ValueError, match="misma longitud"):
        producto_punto(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_norma_manual_vs_numpy() -> None:
    """La norma euclidiana manual coincide con np.linalg.norm."""
    vector = np.array([3.0, 4.0])
    assert np.isclose(norma_euclidiana(vector), np.linalg.norm(vector))
    assert np.isclose(norma_euclidiana(vector), 5.0)


def test_norma_vector_vacio() -> None:
    """La norma de un vector vacío es un error de entrada."""
    with pytest.raises(ValueError, match="vacío"):
        norma_euclidiana(np.array([]))


def test_multiplicar_matrices_con_arroba() -> None:
    """La multiplicación matricial con @ no es la elemento a elemento."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    esperado = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(multiplicar_matrices(a, b), esperado)
    assert not np.allclose(multiplicar_matrices(a, b), a * b)


def test_multiplicar_matrices_dimensiones_invalidas() -> None:
    """Se exige A.cols == B.filas."""
    with pytest.raises(ValueError, match="incompatibles"):
        multiplicar_matrices(np.ones((2, 3)), np.ones((2, 3)))


def test_resolver_sistema_correcto() -> None:
    """A x = b se resuelve y A @ x ≈ b."""
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 10.0])
    x = resolver_sistema(a, b)
    assert np.allclose(a @ x, b)
    assert np.allclose(x, [1.0, 3.0])


def test_resolver_sistema_singular() -> None:
    """Una matriz singular (rango incompleto) se rechaza con un mensaje claro."""
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="singular"):
        resolver_sistema(a, b)


def test_resolver_sistema_dimensiones_invalidas() -> None:
    """La matriz debe ser cuadrada y b debe tener la misma longitud."""
    with pytest.raises(ValueError, match="cuadrada"):
        resolver_sistema(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="elementos"):
        resolver_sistema(np.eye(2), np.ones(3))
