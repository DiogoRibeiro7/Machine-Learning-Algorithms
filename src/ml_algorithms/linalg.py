"""Small linear-algebra helpers used across estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class LeastSquaresResult:
    """Least-squares solution together with numerical diagnostics."""

    solution: NDArray[np.float64]
    residual_sum_squares: float
    rank: int
    singular_values: NDArray[np.float64]
    condition_number: float


def least_squares(
    matrix: ArrayLike,
    target: ArrayLike,
) -> LeastSquaresResult:
    """Solve a linear least-squares problem with SVD-based diagnostics.

    Parameters
    ----------
    matrix:
        Two-dimensional design or coefficient matrix.
    target:
        One-dimensional right-hand-side vector.

    Returns
    -------
    LeastSquaresResult
        Minimum-norm solution, residual sum of squares, rank, singular
        values, and a condition-number diagnostic.
    """
    design = _validate_matrix(matrix, name="matrix")
    rhs = _validate_vector(target, name="target")
    if design.shape[0] != rhs.shape[0]:
        raise ValueError("matrix and target must have the same number of rows.")

    solution, _, rank, singular_values = np.linalg.lstsq(
        design,
        rhs,
        rcond=None,
    )
    solution_array = np.asarray(solution, dtype=np.float64)
    singular_array = np.asarray(singular_values, dtype=np.float64)
    residuals = rhs - design @ solution_array

    return LeastSquaresResult(
        solution=solution_array.copy(),
        residual_sum_squares=float(residuals @ residuals),
        rank=int(rank),
        singular_values=singular_array.copy(),
        condition_number=_condition_number(
            singular_array,
            rank=int(rank),
            n_columns=design.shape[1],
        ),
    )


def pairwise_squared_distances(
    left: ArrayLike,
    right: ArrayLike,
) -> NDArray[np.float64]:
    """Return pairwise squared Euclidean distances between row vectors."""
    left_array = _validate_matrix(left, name="left")
    right_array = _validate_matrix(right, name="right")
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("left and right must have the same number of columns.")

    differences = left_array[:, None, :] - right_array[None, :, :]
    squared = np.sum(differences * differences, axis=2)
    return np.asarray(squared, dtype=np.float64)


def pairwise_distances(
    left: ArrayLike,
    right: ArrayLike,
) -> NDArray[np.float64]:
    """Return pairwise Euclidean distances between row vectors."""
    return np.sqrt(pairwise_squared_distances(left, right))


def canonicalize_row_signs(
    vectors: ArrayLike,
) -> NDArray[np.float64]:
    """Choose deterministic signs for rows of sign-indeterminate vectors."""
    array = _validate_matrix(vectors, name="vectors").copy()
    for row in array:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return array


def _condition_number(
    singular_values: NDArray[np.float64],
    *,
    rank: int,
    n_columns: int,
) -> float:
    """Return a 2-norm condition diagnostic from singular values."""
    if singular_values.size == 0 or rank < n_columns:
        return float("inf")
    smallest = float(singular_values[-1])
    if smallest <= np.finfo(np.float64).eps:
        return float("inf")
    return float(singular_values[0] / smallest)


def _validate_matrix(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return a finite two-dimensional float matrix."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if 0 in array.shape:
        raise ValueError(f"{name} must be non-empty.")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return a finite one-dimensional float vector."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return array


__all__ = [
    "LeastSquaresResult",
    "canonicalize_row_signs",
    "least_squares",
    "pairwise_distances",
    "pairwise_squared_distances",
]
