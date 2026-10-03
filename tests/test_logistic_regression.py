"""Tests for binary logistic regression."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms import LogisticRegression, NotFittedError


def test_matches_reference_coefficients_on_small_dataset() -> None:
    X = np.array([[-2.0], [-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5], [2.0]])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])

    estimator = LogisticRegression(tol=1e-10).fit(X, y)

    assert estimator.converged_ is True
    assert estimator.coef_ is not None
    assert estimator.intercept_ == pytest.approx(0.64999681, abs=1e-6)
    assert estimator.coef_.tolist() == pytest.approx([2.58451424], abs=1e-6)
    probability = estimator.predict_proba([[0.25]])[0, 1]
    assert probability == pytest.approx(0.78518217, abs=1e-6)


def test_probabilities_sum_to_one() -> None:
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    estimator = LogisticRegression(l2=1.0).fit(X, y)
    probabilities = estimator.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0] * 4)


def test_predict_uses_original_binary_labels() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([2, 2, 7, 7])

    estimator = LogisticRegression(l2=0.5).fit(X, y)
    predictions = estimator.predict([[-3.0], [3.0]])

    assert predictions.tolist() == [2, 7]


def test_prediction_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        LogisticRegression().predict([[0.0]])


def test_requires_exactly_two_classes() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        LogisticRegression().fit([[0.0], [1.0], [2.0]], [0, 1, 2])


def test_regularization_keeps_parameters_finite_for_separable_data() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    estimator = LogisticRegression(l2=1.0).fit(X, y)

    assert estimator.coef_ is not None
    assert np.isfinite(estimator.coef_).all()
    assert estimator.intercept_ is not None
    assert np.isfinite(estimator.intercept_)


def test_invalid_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LogisticRegression(l2=-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        LogisticRegression(max_iter=0)
    with pytest.raises(ValueError, match="positive"):
        LogisticRegression(tol=0.0)
