"""Requisitos 5, 6 y 7: operaciones de álgebra lineal con NumPy.

Vectores, matrices, acceso/modificación, suma y resta, multiplicación
matricial (`@`, no `*`), escalar, transpuesta, producto punto y norma
manuales frente a NumPy, y sistemas de ecuaciones `A x = b` con comprobación
de unicidad mediante `np.linalg.matrix_rank`.

Aplicación en IA: un dataset es una matriz X de forma (n_muestras,
n_características) y las predicciones de un modelo lineal son `X @ w + b`.
"""

from __future__ import annotations

import numpy as np

from . import config


def producto_punto(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula el producto punto de dos vectores a mano (sum(aᵢ · bᵢ)).

    En IA representa la combinación ponderada de características:
    `pesos · características` produce la predicción de un modelo lineal.

    Args:
        a: primer vector.
        b: segundo vector.

    Returns:
        Producto punto escalar.

    Raises:
        ValueError: si los vectores no tienen la misma longitud.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"Los vectores deben tener la misma longitud: {a.shape[0]} frente a {b.shape[0]}."
        )
    return float(sum(x * y for x, y in zip(a, b)))


def norma_euclidiana(vector: np.ndarray) -> float:
    """Calcula la norma euclidiana (L2) de un vector a mano: √Σaᵢ².

    En IA la norma de los pesos se usa en técnicas de regularización
    (ridge/L2 penaliza los pesos con gran norma).

    Args:
        vector: vector de entrada.

    Returns:
        Norma euclidiana.

    Raises:
        ValueError: si el vector está vacío.
    """
    vector = np.asarray(vector, dtype=float).ravel()
    if vector.size == 0:
        raise ValueError("No se puede calcular la norma de un vector vacío.")
    return float(sum(float(x) ** 2 for x in vector) ** 0.5)


def sumar_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Suma (o resta, con `a - b`) dos matrices elemento a elemento.

    Args:
        a: primera matriz.
        b: segunda matriz, de las mismas dimensiones que `a`.

    Returns:
        Matriz resultado de la suma.

    Raises:
        ValueError: si las dimensiones no coinciden.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"Las matrices deben tener la misma dimensión para sumarse: "
            f"{a.shape} frente a {b.shape}."
        )
    return a + b


def multiplicar_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiplica dos matrices con el operador `@` (producto matricial).

    ⚠️ `A * B` multiplica elemento a elemento; `A @ B` es la multiplicación
    matricial. Confundirlos es el error más común de este tema.

    Args:
        a: matriz de forma (m, k).
        b: matriz de forma (k, n).

    Returns:
        Matriz de forma (m, n).

    Raises:
        ValueError: si las dimensiones interiores no coinciden (k₁ ≠ k₂).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Se esperaban dos matrices bidimensionales.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Dimensiones incompatibles para la multiplicación: "
            f"A es {a.shape} y B es {b.shape} (se necesita A.cols == B.filas)."
        )
    return a @ b


def resolver_sistema(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve el sistema lineal A x = b con `numpy.linalg.solve`.

    Antes de resolver se comprueba que la solución sea única: la matriz debe
    ser cuadrada y de rango completo (`matrix_rank` es más fiable que mirar
    el determinante, que con decimales nunca da exactamente cero).

    Args:
        a: matriz de coeficientes cuadrada de forma (n, n).
        b: vector de términos independientes de longitud n.

    Returns:
        Vector solución x tal que A @ x ≈ b.

    Raises:
        ValueError: si las dimensiones no cuadran, si la matriz es singular
            (rango incompleto) o si `solve` lanza `LinAlgError`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).ravel()

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"La matriz de coeficientes debe ser cuadrada; recibida con forma {a.shape}.")
    if b.shape[0] != a.shape[0]:
        raise ValueError(
            f"El vector b debe tener {a.shape[0]} elementos; recibido con {b.shape[0]}."
        )
    if np.linalg.matrix_rank(a) != a.shape[0]:
        raise ValueError(
            "La matriz de coeficientes es singular (rango incompleto): "
            "el sistema no tiene solución única."
        )

    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        # Capturamos el error en lugar de dejar caer el programa.
        raise ValueError(
            f"No se pudo resolver el sistema A x = b: {exc}"
        ) from exc


def ejecutar_demostraciones() -> dict[str, object]:
    """Ejecuta la demostración completa de álgebra lineal (requisitos 5, 6 y 7).

    Recorre creación de vectores/matrices, acceso y modificación de elementos,
    suma, resta, multiplicación matricial, escalar, transpuesta, producto punto
    manual vs NumPy, norma manual vs NumPy y la resolución de un sistema
    lineal (incluido un caso singular que se captura sin romper el programa).

    Returns:
        Diccionario con todos los resultados numéricos de la demostración.
    """
    print("=" * 78)
    print("PASO 4 · ÁLGEBRA LINEAL")
    print("=" * 78)

    # -- Vectores y matrices ------------------------------------------------
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    matriz_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    matriz_b = np.array([[5.0, 6.0], [7.0, 8.0]])

    print("\n-- Vectores --")
    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"  suma a + b = {(a + b).tolist()}")
    print(f"  resta a - b = {(a - b).tolist()}")
    print(f"  3 * a (escalar) = {(3 * a).tolist()}")

    # Acceso y modificación de elementos
    copia = a.copy()
    copia[0] = 99.0
    print(f"  acceso a[1] = {a[1]:.1f}; modificación a[0] = 99 -> {copia.tolist()}")

    # Acceso y modificación de elementos de una copia, para no alterar A
    matriz_modificada = matriz_a.copy()
    matriz_modificada[0, 1] = 7.0

    print("\n-- Matrices --")
    print(f"  A = {matriz_a.tolist()}")
    print(f"  B = {matriz_b.tolist()}")
    print(f"  acceso A[1,0] = {matriz_a[1, 0]:.1f} · modificación A[0,1] = 7 -> "
          f"{matriz_modificada.tolist()}")
    print(f"  suma A + B = {(matriz_a + matriz_b).tolist()}")
    print(f"  resta A - B = {(matriz_a - matriz_b).tolist()}")
    producto = multiplicar_matrices(matriz_a, matriz_b)
    print(f"  multiplicación A @ B = {producto.tolist()}  "
          f"(A * B elemento a elemento sería {(matriz_a * matriz_b).tolist()})")
    print(f"  transpuesta A.T = {matriz_a.T.tolist()}")
    print(f"  escalar 2 * A = {(2 * matriz_a).tolist()}")

    # -- Producto punto y norma (manual vs NumPy) ---------------------------
    punto_manual = producto_punto(a, b)
    punto_numpy = float(np.dot(a, b))
    norma_manual = norma_euclidiana(a)
    norma_numpy = float(np.linalg.norm(a))

    print("\n-- Producto punto y norma: manual vs NumPy --")
    print(f"  a · b manual = {punto_manual:.6f} | np.dot = {punto_numpy:.6f} | "
          f"np.isclose = {np.isclose(punto_manual, punto_numpy)}")
    print(f"  ||a|| manual   = {norma_manual:.6f} | np.linalg.norm = {norma_numpy:.6f} | "
          f"np.isclose = {np.isclose(norma_manual, norma_numpy)}")
    print("  (se comparan decimales con np.isclose, nunca con ==, por el redondeo "
          "de punto flotante)")

    # -- Sistema de ecuaciones ---------------------------------------------
    #   2x +  y = 5
    #    x + 3y = 10
    coeficientes = np.array([[2.0, 1.0], [1.0, 3.0]])
    terminos = np.array([5.0, 10.0])

    es_cuadrada = coeficientes.shape[0] == coeficientes.shape[1]
    rango = int(np.linalg.matrix_rank(coeficientes))
    solucion = resolver_sistema(coeficientes, terminos)
    residuo = float(np.max(np.abs(coeficientes @ solucion - terminos)))

    print("\n-- Sistema de ecuaciones A x = b --")
    print("  2x + y = 5 · x + 3y = 10")
    print(f"  matriz cuadrada: {es_cuadrada} · rango: {rango}/{coeficientes.shape[0]} "
          f"(solución única: {es_cuadrada and rango == coeficientes.shape[0]})")
    print(f"  solución x = {np.round(solucion, 6).tolist()}")
    print(f"  comprobación A @ x ~ b -> {np.allclose(coeficientes @ solucion, terminos)} "
          f"(residuo máx. {residuo:.2e})")

    # Caso singular: se captura el error en lugar de romper el programa
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    mensaje_singular: str
    try:
        resolver_sistema(singular, np.array([1.0, 2.0]))
        mensaje_singular = "sin error (inesperado)"
    except ValueError as exc:
        mensaje_singular = str(exc)
    print(f"\n-- Caso singular capturado (matriz [[1,2],[2,4]]) --")
    print(f"  {mensaje_singular}")

    print("\nAplicación en IA: el dataset es la matriz X (n_muestras, n_características) "
          "y un modelo lineal predice con X @ w + b.\n")

    return {
        "producto_punto_manual": punto_manual,
        "producto_punto_numpy": punto_numpy,
        "norma_manual": norma_manual,
        "norma_numpy": norma_numpy,
        "producto_matrices": producto.tolist(),
        "transpuesta": matriz_a.T.tolist(),
        "sistema_solucion": solucion.tolist(),
        "sistema_residuo": residuo,
        "sistema_rango": rango,
        "mensaje_singular": mensaje_singular,
        "config_random_state": config.RANDOM_STATE,
    }
