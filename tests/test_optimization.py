"""Tests for numerical optimization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms.optimization import gradient_descent


def test_gradient_descent_solves_one_dimensional_quadratic() -> None:
    def objective(theta: np.ndarray) -> float:
        return float((theta[0] - 3.0) ** 2)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (theta[0] - 3.0)])

    result = gradient_descent(
        objective,
        gradient,
        [0.0],
        learning_rate=0.25,
        tol=1e-10,
    )

    assert result.converged is True
    assert result.parameters.tolist() == pytest.approx([3.0], abs=1e-9)
    assert result.objective == pytest.approx(0.0, abs=1e-18)
    assert result.gradient_norm <= 1e-10
    assert result.n_iter == len(result.history)


def test_gradient_descent_solves_multivariate_quadratic() -> None:
    matrix = np.array([[4.0, 0.0], [0.0, 2.0]])
    optimum = np.array([1.5, -2.0])

    def objective(theta: np.ndarray) -> float:
        error = theta - optimum
        return float(0.5 * error @ matrix @ error)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return matrix @ (theta - optimum)

    result = gradient_descent(
        objective,
        gradient,
        [8.0, 5.0],
        learning_rate=0.2,
        tol=1e-9,
    )

    assert result.converged is True
    assert bool(np.allclose(result.parameters, optimum, atol=1e-8, rtol=0.0))
    assert all(step.iteration >= 1 for step in result.history)
    assert all(step.gradient_norm >= 0.0 for step in result.history)
    assert all(step.step_norm > 0.0 for step in result.history)


def test_already_optimal_initial_point_converges_without_updates() -> None:
    def objective(theta: np.ndarray) -> float:
        return float(theta @ theta)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return 2.0 * theta

    result = gradient_descent(objective, gradient, [0.0, 0.0])

    assert result.converged is True
    assert result.n_iter == 0
    assert result.history == ()
    assert result.objective == 0.0


def test_iteration_limit_is_reported() -> None:
    def objective(theta: np.ndarray) -> float:
        return float((theta[0] - 1.0) ** 2)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (theta[0] - 1.0)])

    result = gradient_descent(
        objective,
        gradient,
        [10.0],
        learning_rate=0.01,
        max_iter=2,
        tol=1e-15,
    )

    assert result.converged is False
    assert result.n_iter == 2
    assert len(result.history) == 2


def test_rejects_gradient_with_wrong_shape() -> None:
    def objective(theta: np.ndarray) -> float:
        return float(theta @ theta)

    def gradient(_: np.ndarray) -> np.ndarray:
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        gradient_descent(objective, gradient, [0.0])


def test_rejects_non_finite_objective_and_gradient() -> None:
    def finite_gradient(theta: np.ndarray) -> np.ndarray:
        return np.ones_like(theta)

    with pytest.raises(ValueError, match="objective"):
        gradient_descent(lambda _: float("nan"), finite_gradient, [0.0])

    with pytest.raises(ValueError, match="gradient"):
        gradient_descent(
            lambda theta: float(theta @ theta),
            lambda _: np.array([np.inf]),
            [0.0],
        )


def test_rejects_invalid_configuration() -> None:
    objective = lambda theta: float(theta @ theta)
    gradient = lambda theta: 2.0 * theta

    with pytest.raises(ValueError, match="learning_rate"):
        gradient_descent(objective, gradient, [1.0], learning_rate=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        gradient_descent(objective, gradient, [1.0], max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        gradient_descent(objective, gradient, [1.0], tol=0.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        gradient_descent(objective, gradient, [[1.0]], tol=1e-8)
