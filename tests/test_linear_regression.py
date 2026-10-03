"""Tests for ordinary least-squares linear regression."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms import LinearRegression, NotFittedError


def test_exact_multivariate_fit_recovers_coefficients() -> None:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, -1.0],
        ],
        dtype=np.float64,
    )
    expected_coef = np.array([2.0, -3.0], dtype=np.float64)
    expected_intercept = 1.5
    y = expected_intercept + X @ expected_coef

    estimator = LinearRegression().fit(X, y)

    assert estimator.intercept_ == pytest.approx(expected_intercept)
    assert estimator.coef_ is not None
    assert estimator.coef_.tolist() == pytest.approx(expected_coef.tolist(), abs=1e-12)
    assert estimator.predict(X).tolist() == pytest.approx(y.tolist(), abs=1e-12)
    assert estimator.residual_sum_squares_ == pytest.approx(0.0, abs=1e-24)
    assert estimator.rank_ == 3


def test_fit_without_intercept() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = 4.0 * X[:, 0]

    estimator = LinearRegression(fit_intercept=False).fit(X, y)

    assert estimator.intercept_ == 0.0
    assert estimator.coef_ is not None
    assert estimator.coef_.tolist() == pytest.approx([4.0], abs=1e-12)
    assert estimator.predict([[5.0]]).tolist() == pytest.approx([20.0], abs=1e-12)


def test_rank_deficient_design_returns_valid_least_squares_solution() -> None:
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ],
        dtype=np.float64,
    )
    y = np.array([4.0, 7.0, 10.0, 13.0], dtype=np.float64)

    estimator = LinearRegression().fit(X, y)

    assert estimator.rank_ is not None
    assert estimator.rank_ < 3
    assert estimator.predict(X).tolist() == pytest.approx(y.tolist(), abs=1e-12)
    assert estimator.residual_sum_squares_ == pytest.approx(0.0, abs=1e-24)


def test_refit_replaces_previous_parameters() -> None:
    estimator = LinearRegression().fit([[0.0], [1.0]], [1.0, 2.0])
    first_coef = estimator.coef_.copy() if estimator.coef_ is not None else None

    estimator.fit([[0.0], [1.0], [2.0]], [3.0, 5.0, 7.0])

    assert first_coef is not None
    assert first_coef.tolist() == pytest.approx([1.0], abs=1e-12)
    assert estimator.coef_ is not None
    assert estimator.coef_.tolist() == pytest.approx([2.0], abs=1e-12)
    assert estimator.intercept_ == pytest.approx(3.0)


def test_prediction_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        LinearRegression().predict([[1.0]])


def test_non_numeric_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="numeric"):
        LinearRegression().fit([[1.0], [2.0]], ["a", "b"])


def test_non_finite_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        LinearRegression().fit([[1.0], [2.0]], [1.0, np.inf])


def test_invalid_fit_intercept_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="boolean"):
        LinearRegression(fit_intercept=1)  # type: ignore[arg-type]
