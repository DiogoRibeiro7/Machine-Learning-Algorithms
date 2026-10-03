"""Validation helpers shared by estimator implementations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ml_algorithms.types import FeatureMatrix, TargetVector


def validate_features(
    X: ArrayLike,
    *,
    expected_n_features: int | None = None,
    ensure_finite: bool = True,
) -> FeatureMatrix:
    """Validate and normalize a two-dimensional feature matrix."""
    try:
        array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric values.") from exc

    if array.ndim != 2:
        raise ValueError(
            f"X must be two-dimensional; received an array with {array.ndim} dimensions."
        )

    n_samples, n_features = array.shape
    if n_samples == 0:
        raise ValueError("X must contain at least one sample.")
    if n_features == 0:
        raise ValueError("X must contain at least one feature.")

    if expected_n_features is not None and n_features != expected_n_features:
        raise ValueError(
            "X has an incompatible number of features: "
            f"expected {expected_n_features}, received {n_features}."
        )

    if ensure_finite and not bool(np.isfinite(array).all()):
        raise ValueError("X must contain only finite values.")

    return array


def validate_target(y: ArrayLike, *, n_samples: int | None = None) -> TargetVector:
    """Validate a one-dimensional target vector without changing its dtype."""
    array = np.asarray(y)

    if array.ndim != 1:
        raise ValueError(
            f"y must be one-dimensional; received an array with {array.ndim} dimensions."
        )
    if array.shape[0] == 0:
        raise ValueError("y must contain at least one value.")
    if n_samples is not None and array.shape[0] != n_samples:
        raise ValueError(
            "X and y have inconsistent numbers of samples: "
            f"{n_samples} and {array.shape[0]}."
        )

    return array


def validate_X_y(X: ArrayLike, y: ArrayLike) -> tuple[FeatureMatrix, TargetVector]:
    """Validate a supervised learning feature matrix and target vector."""
    features = validate_features(X)
    target = validate_target(y, n_samples=features.shape[0])
    return features, target


__all__ = ["validate_X_y", "validate_features", "validate_target"]
