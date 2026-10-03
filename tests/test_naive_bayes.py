"""Tests for Gaussian Naive Bayes."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms import GaussianNB, NotFittedError


def test_estimates_expected_gaussian_parameters() -> None:
    X = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.2],
            [0.0, -0.2],
            [0.5, 0.1],
            [0.8, 0.9],
            [1.0, 1.2],
            [1.4, 0.8],
            [1.6, 1.1],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    estimator = GaussianNB().fit(X, y)

    assert estimator.theta_ is not None
    assert estimator.var_ is not None
    assert estimator.class_prior_ is not None
    assert estimator.theta_[0].tolist() == pytest.approx([-0.25, 0.025])
    assert estimator.theta_[1].tolist() == pytest.approx([1.2, 1.0])
    assert estimator.var_[0].tolist() == pytest.approx(
        [0.3125, 0.021875],
        abs=1e-8,
    )
    assert estimator.var_[1].tolist() == pytest.approx(
        [0.1, 0.025],
        abs=1e-8,
    )
    assert estimator.class_prior_.tolist() == pytest.approx([0.5, 0.5])


def test_probabilities_match_reference_values() -> None:
    X = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.2],
            [0.0, -0.2],
            [0.5, 0.1],
            [0.8, 0.9],
            [1.0, 1.2],
            [1.4, 0.8],
            [1.6, 1.1],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    estimator = GaussianNB().fit(X, y)
    probabilities = estimator.predict_proba([[0.4, 0.2], [0.8, 0.6]])

    assert probabilities[0].tolist() == pytest.approx(
        [0.9999992633, 0.0000007367],
        abs=1e-8,
    )
    assert probabilities[1].tolist() == pytest.approx(
        [0.0029467711, 0.9970532289],
        abs=1e-8,
    )


def test_predict_preserves_string_labels() -> None:
    X = np.array([[0.0], [0.2], [3.0], [3.2]], dtype=np.float64)
    y = np.array(["cold", "cold", "hot", "hot"])

    estimator = GaussianNB().fit(X, y)

    assert estimator.predict([[0.1], [3.1]]).tolist() == ["cold", "hot"]


def test_probabilities_sum_to_one() -> None:
    X = np.array([[0.0], [0.2], [1.0], [1.2]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])

    probabilities = GaussianNB().fit(X, y).predict_proba([[0.1], [0.8]])

    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_variance_smoothing_handles_constant_features() -> None:
    X = np.ones((4, 2), dtype=np.float64)
    y = np.array([0, 0, 1, 1])

    estimator = GaussianNB(var_smoothing=1e-6).fit(X, y)

    assert estimator.epsilon_ is not None
    assert estimator.epsilon_ > 0.0
    assert estimator.var_ is not None
    assert bool(np.all(estimator.var_ > 0.0))
    assert bool(np.isfinite(estimator.predict_proba([[1.0, 1.0]])).all())


def test_prediction_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        GaussianNB().predict([[0.0]])


def test_invalid_var_smoothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        GaussianNB(var_smoothing=0.0)
    with pytest.raises(TypeError, match="real number"):
        GaussianNB(var_smoothing=True)
