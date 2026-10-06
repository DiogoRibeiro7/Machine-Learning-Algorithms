"""Tests for common metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml_algorithms.metrics import (
    accuracy_score,
    binary_log_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def test_regression_metrics_match_hand_computed_values() -> None:
    y_true = [1.0, 2.0, 4.0]
    y_pred = [1.0, 3.0, 2.0]

    assert mean_squared_error(y_true, y_pred) == pytest.approx(5.0 / 3.0)
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.0)
    assert r2_score(y_true, y_pred) == pytest.approx(-1.0 / 14.0)


def test_r2_perfect_prediction_is_one() -> None:
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_r2_constant_target_has_explicit_degenerate_behavior() -> None:
    assert r2_score([2.0, 2.0, 2.0], [2.0, 2.0, 2.0]) == 1.0
    assert r2_score([2.0, 2.0, 2.0], [2.0, 2.0, 3.0]) == 0.0


def test_accuracy_supports_arbitrary_labels() -> None:
    assert accuracy_score(["cat", "dog", "dog"], ["cat", "cat", "dog"]) == pytest.approx(2 / 3)


def test_binary_log_loss_matches_hand_computed_value() -> None:
    actual = binary_log_loss([1, 0], [0.8, 0.25])
    expected = -(math.log(0.8) + math.log(0.75)) / 2.0

    assert actual == pytest.approx(expected)


def test_binary_log_loss_clips_zero_and_one_probabilities() -> None:
    value = binary_log_loss([1, 0], [1.0, 0.0])

    assert np.isfinite(value)
    assert value >= 0.0


def test_binary_log_loss_validates_inputs() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        binary_log_loss([0, 2], [0.2, 0.8])
    with pytest.raises(ValueError, match="between 0 and 1"):
        binary_log_loss([0, 1], [-0.1, 1.1])
    with pytest.raises(ValueError, match=r"smaller than 0\.5"):
        binary_log_loss([0, 1], [0.2, 0.8], epsilon=0.5)


def test_confusion_matrix_returns_matrix_and_label_order() -> None:
    matrix, labels = confusion_matrix(["cat", "dog", "dog"], ["cat", "cat", "dog"])

    assert labels.tolist() == ["cat", "dog"]
    assert matrix.tolist() == [[1, 0], [1, 1]]


def test_confusion_matrix_respects_explicit_label_order() -> None:
    matrix, labels = confusion_matrix([0, 1, 1], [0, 0, 1], labels=[1, 0])

    assert labels.tolist() == [1, 0]
    assert matrix.tolist() == [[1, 1], [0, 1]]


def test_confusion_matrix_rejects_missing_and_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="missing from labels"):
        confusion_matrix([0, 1], [0, 1], labels=[0])
    with pytest.raises(ValueError, match="duplicates"):
        confusion_matrix([0, 1], [0, 1], labels=[0, 0, 1])


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        mean_squared_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="inconsistent"):
        accuracy_score(["a", "b"], ["a"])


def test_regression_metrics_reject_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="numeric"):
        mean_absolute_error(["a"], ["b"])
