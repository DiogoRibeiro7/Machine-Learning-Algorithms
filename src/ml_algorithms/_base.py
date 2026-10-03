"""Common estimator contracts and fitted-state handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from numpy.typing import ArrayLike, NDArray

from ml_algorithms._validation import validate_features
from ml_algorithms.exceptions import NotFittedError
from ml_algorithms.types import FeatureMatrix


class BaseEstimator(ABC):
    """Base class for estimators implemented by this package.

    Estimators use fit(X, y=None) -> self. Fitted estimators record
    n_features_in_, and inference-time methods validate feature count.
    Subclasses should reset state at the beginning of fit and mark the
    estimator fitted only after all learned parameters are available.
    """

    def __init__(self) -> None:
        self._is_fitted = False
        self.n_features_in_: int | None = None

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the estimator and return itself."""

    @property
    def is_fitted(self) -> bool:
        """Return whether fitted parameters are available."""
        return self._is_fitted

    def _reset_fit_state(self) -> None:
        """Clear fitted-state metadata before refitting."""
        self._is_fitted = False
        self.n_features_in_ = None

    def _mark_fitted(self, *, n_features: int) -> None:
        """Record successful fitting and the expected feature count."""
        if n_features < 1:
            raise ValueError("n_features must be at least 1.")

        self.n_features_in_ = n_features
        self._is_fitted = True

    def _check_is_fitted(self) -> None:
        """Raise if fitted parameters are unavailable."""
        if not self._is_fitted:
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted. Call fit before using it."
            )

        if self.n_features_in_ is None:
            raise RuntimeError(
                "Estimator state is inconsistent: fitted estimator has no "
                "n_features_in_."
            )

    def _validate_fit_features(self, X: ArrayLike) -> FeatureMatrix:
        """Validate feature data supplied to fit."""
        return validate_features(X)

    def _validate_inference_features(self, X: ArrayLike) -> FeatureMatrix:
        """Validate feature data supplied after fitting."""
        self._check_is_fitted()
        if self.n_features_in_ is None:
            raise RuntimeError("Estimator has no fitted feature-count metadata.")

        return validate_features(X, expected_n_features=self.n_features_in_)


class PredictorMixin(ABC):
    """Interface for estimators that produce per-sample predictions."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict one output value for each input sample."""


class TransformerMixin(ABC):
    """Interface for estimators that transform feature matrices."""

    @abstractmethod
    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Transform input samples into a new representation."""


__all__ = ["BaseEstimator", "PredictorMixin", "TransformerMixin"]
