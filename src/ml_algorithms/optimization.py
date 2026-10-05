"""Small numerical optimization utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Objective = Callable[[NDArray[np.float64]], float]
type Gradient = Callable[[NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True, slots=True)
class OptimizationStep:
    """Diagnostics recorded after one gradient-descent update."""

    iteration: int
    objective: float
    gradient_norm: float
    step_norm: float


@dataclass(frozen=True, slots=True)
class GradientDescentResult:
    """Result returned by :func:`gradient_descent`."""

    parameters: NDArray[np.float64]
    converged: bool
    n_iter: int
    objective: float
    gradient_norm: float
    history: tuple[OptimizationStep, ...]


def gradient_descent(
    objective: Objective,
    gradient: Gradient,
    initial: ArrayLike,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 1_000,
    tol: float = 1e-8,
) -> GradientDescentResult:
    """Minimize a differentiable objective with full-batch gradient descent.

    Parameters
    ----------
    objective:
        Scalar objective evaluated at the current parameter vector.
    gradient:
        Gradient of ``objective`` with respect to the parameter vector.
    initial:
        One-dimensional initial parameter vector.
    learning_rate:
        Positive constant step size.
    max_iter:
        Maximum number of parameter updates.
    tol:
        Convergence tolerance on the Euclidean gradient norm.

    Returns
    -------
    GradientDescentResult
        Final parameters together with convergence diagnostics and history.
    """
    rate = _validate_positive_real(learning_rate, name="learning_rate")
    tolerance = _validate_positive_real(tol, name="tol")
    iterations = _validate_positive_integer(max_iter, name="max_iter")
    parameters = _validate_parameter_vector(initial)

    history: list[OptimizationStep] = []
    converged = False
    n_iter = 0

    current_objective = _evaluate_objective(objective, parameters)
    current_gradient = _evaluate_gradient(gradient, parameters)
    gradient_norm = float(np.linalg.norm(current_gradient))

    if gradient_norm <= tolerance:
        converged = True

    while not converged and n_iter < iterations:
        step = -rate * current_gradient
        candidate = parameters + step
        if not bool(np.isfinite(candidate).all()):
            raise ValueError("Gradient descent produced non-finite parameters.")

        n_iter += 1
        parameters = candidate
        current_objective = _evaluate_objective(objective, parameters)
        current_gradient = _evaluate_gradient(gradient, parameters)
        gradient_norm = float(np.linalg.norm(current_gradient))
        step_norm = float(np.linalg.norm(step))

        history.append(
            OptimizationStep(
                iteration=n_iter,
                objective=current_objective,
                gradient_norm=gradient_norm,
                step_norm=step_norm,
            )
        )

        if gradient_norm <= tolerance:
            converged = True

    return GradientDescentResult(
        parameters=parameters.copy(),
        converged=converged,
        n_iter=n_iter,
        objective=current_objective,
        gradient_norm=gradient_norm,
        history=tuple(history),
    )


def _validate_parameter_vector(initial: ArrayLike) -> NDArray[np.float64]:
    """Return a finite one-dimensional float parameter vector."""
    try:
        parameters = np.asarray(initial, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial must contain numeric values.") from exc

    if parameters.ndim != 1:
        raise ValueError("initial must be one-dimensional.")
    if parameters.size == 0:
        raise ValueError("initial must contain at least one parameter.")
    if not bool(np.isfinite(parameters).all()):
        raise ValueError("initial must contain only finite values.")
    return parameters.copy()


def _evaluate_objective(
    objective: Objective,
    parameters: NDArray[np.float64],
) -> float:
    """Evaluate and validate the scalar objective."""
    value = float(objective(parameters.copy()))
    if not np.isfinite(value):
        raise ValueError("objective must return a finite scalar value.")
    return value


def _evaluate_gradient(
    gradient: Gradient,
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate and validate a gradient vector."""
    try:
        value = np.asarray(gradient(parameters.copy()), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient must return numeric values.") from exc

    if value.ndim != 1:
        raise ValueError("gradient must return a one-dimensional vector.")
    if value.shape != parameters.shape:
        raise ValueError("gradient must have the same shape as the parameters.")
    if not bool(np.isfinite(value).all()):
        raise ValueError("gradient must contain only finite values.")
    return value


def _validate_positive_real(value: float, *, name: str) -> float:
    """Validate a positive finite real hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _validate_positive_integer(value: int, *, name: str) -> int:
    """Validate a strictly positive integer hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


__all__ = [
    "GradientDescentResult",
    "OptimizationStep",
    "gradient_descent",
]
