"""Tests for common estimator infrastructure."""

from __future__ import annotations

from typing import Self

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from ml_algorithms import BaseEstimator, NotFittedError, PredictorMixin
from ml_algorithms._random import resolve_random_state
from ml_algorithms._validation import validate_X_y, validate_features


class MeanRegressor(BaseEstimator, PredictorMixin):
    """Minimal estimator used to exercise the common contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        self._reset_fit_state()
        if y is None:
            raise ValueError("y is required.")

        features, target = validate_X_y(X, y)
        self.mean_ = float(np.asarray(target, dtype=np.float64).mean())
        self._mark_fitted(n_features=features.shape[1])
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        features = self._validate_inference_features(X)
        if self.mean_ is None:
            raise RuntimeError("Estimator has no fitted mean.")
        return np.full(features.shape[0], self.mean_, dtype=np.float64)


def test_predict_before_fit_raises() -> None:
    estimator = MeanRegressor()

    with pytest.raises(NotFittedError):
        estimator.predict([[1.0]])


def test_fit_records_feature_count_and_returns_self() -> None:
    estimator = MeanRegressor()

    fitted = estimator.fit([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0])

    assert fitted is estimator
    assert estimator.is_fitted
    assert estimator.n_features_in_ == 2


def test_prediction_checks_feature_count() -> None:
    estimator = MeanRegressor().fit([[1.0, 2.0]], [3.0])

    with pytest.raises(ValueError, match="incompatible number of features"):
        estimator.predict([[1.0]])


def test_validate_features_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_features([[1.0, np.nan]])


def test_validate_X_y_rejects_inconsistent_sample_counts() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        validate_X_y([[1.0], [2.0]], [1.0])


def test_integer_random_state_is_reproducible() -> None:
    first = resolve_random_state(42).normal(size=5)
    second = resolve_random_state(42).normal(size=5)

    np.testing.assert_array_equal(first, second)


def test_existing_generator_is_reused() -> None:
    generator = np.random.default_rng(42)

    assert resolve_random_state(generator) is generator


def test_boolean_random_state_is_rejected() -> None:
    with pytest.raises(TypeError, match="boolean"):
        resolve_random_state(True)
