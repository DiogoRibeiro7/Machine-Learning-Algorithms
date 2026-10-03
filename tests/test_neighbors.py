"""Tests for the k-nearest-neighbours classifier."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms import KNeighborsClassifier, NotFittedError


def test_predicts_expected_classes() -> None:
    X = np.array([[0.0], [1.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])

    estimator = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    predictions = estimator.predict([[0.2], [4.8]])

    assert predictions.tolist() == [0, 1]


def test_predict_proba_uses_vote_frequencies() -> None:
    X = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1])

    estimator = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    probabilities = estimator.predict_proba([[1.1]])

    assert probabilities.shape == (1, 2)
    assert probabilities[0].tolist() == pytest.approx([2 / 3, 1 / 3])


def test_equal_vote_tie_uses_sorted_class_order() -> None:
    X = np.array([[0.0], [2.0]], dtype=np.float64)
    y = np.array([7, 2])

    estimator = KNeighborsClassifier(n_neighbors=2).fit(X, y)

    assert estimator.classes_ is not None
    assert estimator.classes_.tolist() == [2, 7]
    assert estimator.predict([[1.0]]).tolist() == [2]


def test_equal_distances_preserve_training_order() -> None:
    X = np.array([[-1.0], [1.0], [3.0]], dtype=np.float64)
    y = np.array([10, 20, 30])

    estimator = KNeighborsClassifier(n_neighbors=2).fit(X, y)
    distances, indices = estimator.kneighbors([[0.0]])

    assert indices.tolist() == [[0, 1]]
    assert distances[0].tolist() == pytest.approx([1.0, 1.0])


def test_prediction_preserves_string_labels() -> None:
    X = np.array([[0.0], [1.0], [9.0], [10.0]], dtype=np.float64)
    y = np.array(["left", "left", "right", "right"])

    estimator = KNeighborsClassifier(n_neighbors=1).fit(X, y)

    assert estimator.predict([[0.2], [9.8]]).tolist() == ["left", "right"]


def test_kneighbors_returns_expected_distances_and_indices() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    y = np.array([0, 1, 1])

    estimator = KNeighborsClassifier(n_neighbors=2).fit(X, y)
    distances, indices = estimator.kneighbors([[0.0, 0.0]])

    assert indices.tolist() == [[0, 1]]
    assert distances[0].tolist() == pytest.approx([0.0, 5.0])


def test_prediction_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        KNeighborsClassifier().predict([[0.0]])


def test_too_many_neighbors_is_rejected_during_fit() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        KNeighborsClassifier(n_neighbors=3).fit([[0.0], [1.0]], [0, 1])


def test_invalid_neighbor_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        KNeighborsClassifier(n_neighbors=0)
    with pytest.raises(TypeError, match="integer"):
        KNeighborsClassifier(n_neighbors=True)
