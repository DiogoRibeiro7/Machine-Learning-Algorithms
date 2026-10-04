"""Tests for the linear support-vector machine."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms import LinearSVM, NotFittedError


def test_symmetric_problem_matches_closed_form_reference() -> None:
    X = np.array(
        [[-2.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1])

    estimator = LinearSVM(C=1.0, tol=1e-12).fit(X, y)

    assert estimator.coef_ is not None
    assert estimator.intercept_ == pytest.approx(0.0, abs=1e-12)
    assert estimator.coef_.tolist() == pytest.approx([1 / 3, 1 / 3], abs=1e-10)
    assert estimator.loss_ == pytest.approx(1 / 6, abs=1e-10)
    assert estimator.converged_ is True


def test_predicts_linearly_separable_labels() -> None:
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    estimator = LinearSVM(C=10.0).fit(X, y)

    assert estimator.predict(X).tolist() == y.tolist()
    scores = estimator.decision_function([[-2.5], [2.5]])
    assert scores[0] < 0.0
    assert scores[1] > 0.0


def test_preserves_original_string_labels() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array(["left", "left", "right", "right"])

    estimator = LinearSVM(C=2.0).fit(X, y)

    assert estimator.predict([[-3.0], [3.0]]).tolist() == ["left", "right"]


def test_fit_without_intercept_keeps_zero_intercept() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    estimator = LinearSVM(fit_intercept=False).fit(X, y)

    assert estimator.intercept_ == 0.0


def test_requires_exactly_two_classes() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        LinearSVM().fit([[0.0], [1.0], [2.0]], [0, 1, 2])


def test_decision_function_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        LinearSVM().decision_function([[0.0]])


def test_invalid_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        LinearSVM(C=0.0)
    with pytest.raises(ValueError, match="at least 1"):
        LinearSVM(max_iter=0)
    with pytest.raises(ValueError, match="positive"):
        LinearSVM(tol=0.0)
