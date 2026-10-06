"""Common regression and classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._validation import validate_positive_real, validate_target


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the mean squared prediction error."""
    true, predicted = _validate_numeric_pair(y_true, y_pred)
    residual = true - predicted
    return float(np.mean(residual * residual))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the mean absolute prediction error."""
    true, predicted = _validate_numeric_pair(y_true, y_pred)
    return float(np.mean(np.abs(true - predicted)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the coefficient of determination.

    A constant target has zero total sum of squares. For that degenerate
    case, perfect predictions return 1.0 and all imperfect predictions
    return 0.0.
    """
    true, predicted = _validate_numeric_pair(y_true, y_pred)
    residual = true - predicted
    residual_sum_squares = float(residual @ residual)
    centered = true - float(np.mean(true))
    total_sum_squares = float(centered @ centered)

    if total_sum_squares == 0.0:
        return 1.0 if residual_sum_squares == 0.0 else 0.0
    return float(1.0 - residual_sum_squares / total_sum_squares)


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of exactly matching labels."""
    true, predicted = _validate_label_pair(y_true, y_pred)
    return float(np.mean(true == predicted))


def binary_log_loss(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    epsilon: float = 1e-15,
) -> float:
    """Return mean binary cross-entropy for positive-class probabilities."""
    stabilizer = validate_positive_real(epsilon, name="epsilon")
    if stabilizer >= 0.5:
        raise ValueError("epsilon must be smaller than 0.5.")

    true = _validate_binary_target(y_true)
    probability = _validate_probability_vector(
        y_probability,
        n_samples=true.shape[0],
    )
    clipped = np.clip(probability, stabilizer, 1.0 - stabilizer)
    losses = -(true * np.log(clipped) + (1.0 - true) * np.log1p(-clipped))
    return float(np.mean(losses))


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = None,
) -> tuple[NDArray[np.int64], NDArray[Any]]:
    """Return a confusion matrix and the label order used for its axes.

    Rows correspond to true labels and columns to predicted labels.
    """
    true, predicted = _validate_label_pair(y_true, y_pred)
    label_order = _resolve_labels(true, predicted, labels=labels)
    matrix = np.zeros((label_order.shape[0], label_order.shape[0]), dtype=np.int64)

    for true_value, predicted_value in zip(true, predicted, strict=True):
        true_index = _label_index(label_order, true_value)
        predicted_index = _label_index(label_order, predicted_value)
        matrix[true_index, predicted_index] += 1

    return matrix, label_order.copy()


def _validate_numeric_pair(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate paired finite numeric vectors."""
    true_raw, predicted_raw = _validate_label_pair(y_true, y_pred)
    try:
        true = np.asarray(true_raw, dtype=np.float64)
        predicted = np.asarray(predicted_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and y_pred must contain numeric values.") from exc

    if not bool(np.isfinite(true).all()) or not bool(np.isfinite(predicted).all()):
        raise ValueError("y_true and y_pred must contain only finite values.")
    return true, predicted


def _validate_label_pair(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Validate paired one-dimensional target/prediction vectors."""
    true = validate_target(y_true)
    predicted = validate_target(y_pred, n_samples=true.shape[0])
    return np.asarray(true), np.asarray(predicted)


def _validate_binary_target(y_true: ArrayLike) -> NDArray[np.float64]:
    """Validate a binary target encoded as zero and one."""
    raw = validate_target(y_true)
    try:
        target = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must be encoded with numeric 0 and 1 values.") from exc

    if not bool(np.isfinite(target).all()):
        raise ValueError("y_true must contain only finite values.")
    if not bool(np.isin(target, [0.0, 1.0]).all()):
        raise ValueError("y_true must contain only 0 and 1.")
    return target


def _validate_probability_vector(
    values: ArrayLike,
    *,
    n_samples: int,
) -> NDArray[np.float64]:
    """Validate a one-dimensional finite probability vector."""
    try:
        probability = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_probability must contain numeric values.") from exc

    if probability.ndim != 1:
        raise ValueError("y_probability must be one-dimensional.")
    if probability.shape[0] != n_samples:
        raise ValueError("y_true and y_probability must have the same length.")
    if not bool(np.isfinite(probability).all()):
        raise ValueError("y_probability must contain only finite values.")
    if bool(np.any((probability < 0.0) | (probability > 1.0))):
        raise ValueError("y_probability values must lie between 0 and 1.")
    return probability


def _resolve_labels(
    true: NDArray[Any],
    predicted: NDArray[Any],
    *,
    labels: ArrayLike | None,
) -> NDArray[Any]:
    """Resolve and validate the confusion-matrix label order."""
    if labels is None:
        try:
            return np.unique(np.concatenate((true, predicted)))
        except TypeError as exc:
            raise ValueError("Observed labels must be mutually comparable.") from exc

    label_order = validate_target(labels)
    try:
        unique = np.unique(label_order)
    except TypeError as exc:
        raise ValueError("labels must be mutually comparable.") from exc
    if unique.shape[0] != label_order.shape[0]:
        raise ValueError("labels must not contain duplicates.")

    for value in np.concatenate((true, predicted)):
        _label_index(label_order, value)
    return np.asarray(label_order)


def _label_index(labels: NDArray[Any], value: Any) -> int:
    """Return the index of one label or raise a clear validation error."""
    matches = np.flatnonzero(labels == value)
    if matches.size == 0:
        raise ValueError(f"Observed label {value!r} is missing from labels.")
    return int(matches[0])


__all__ = [
    "accuracy_score",
    "binary_log_loss",
    "confusion_matrix",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
]
