"""Naive Bayes estimators."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._base import BaseEstimator, PredictorMixin
from ml_algorithms._validation import validate_X_y


class GaussianNB(BaseEstimator, PredictorMixin):
    """Gaussian Naive Bayes classifier for continuous features.

    Parameters
    ----------
    var_smoothing:
        Fraction of the largest global feature variance added to each
        class-conditional variance for numerical stability.
    """

    def __init__(self, *, var_smoothing: float = 1e-9) -> None:
        if isinstance(var_smoothing, bool) or not isinstance(
            var_smoothing, (int, float)
        ):
            raise TypeError("var_smoothing must be a real number.")
        if var_smoothing <= 0.0:
            raise ValueError("var_smoothing must be positive.")

        super().__init__()
        self.var_smoothing = float(var_smoothing)
        self.classes_: NDArray[Any] | None = None
        self.class_count_: NDArray[np.int64] | None = None
        self.class_prior_: NDArray[np.float64] | None = None
        self.theta_: NDArray[np.float64] | None = None
        self.var_: NDArray[np.float64] | None = None
        self.epsilon_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Estimate class priors and Gaussian feature parameters."""
        self._reset_parameters()
        if y is None:
            raise ValueError("y is required for Gaussian Naive Bayes.")

        features, target = validate_X_y(X, y)
        classes, encoded, counts = np.unique(
            target,
            return_inverse=True,
            return_counts=True,
        )

        n_classes = classes.shape[0]
        n_features = features.shape[1]
        theta = np.empty((n_classes, n_features), dtype=np.float64)
        variances = np.empty((n_classes, n_features), dtype=np.float64)

        global_variance = np.var(features, axis=0)
        variance_scale = max(
            float(np.max(global_variance)),
            float(np.finfo(np.float64).eps),
        )
        epsilon = self.var_smoothing * variance_scale

        for class_index in range(n_classes):
            class_samples = features[encoded == class_index]
            theta[class_index] = np.mean(class_samples, axis=0)
            variances[class_index] = np.var(class_samples, axis=0) + epsilon

        self.classes_ = np.asarray(classes).copy()
        self.class_count_ = np.asarray(counts, dtype=np.int64)
        self.class_prior_ = np.asarray(
            counts / features.shape[0],
            dtype=np.float64,
        )
        self.theta_ = theta
        self.var_ = variances
        self.epsilon_ = epsilon
        self._mark_fitted(n_features=n_features)
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict the class with largest posterior probability."""
        joint = self._joint_log_likelihood(X)
        if self.classes_ is None:
            raise RuntimeError("GaussianNB fitted classes are unavailable.")
        return self.classes_[np.argmax(joint, axis=1)]

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return normalized log posterior probabilities."""
        joint = self._joint_log_likelihood(X)
        normalizer = _logsumexp(joint, axis=1)
        return joint - normalizer[:, None]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return posterior class probabilities."""
        return np.exp(self.predict_log_proba(X))

    def _joint_log_likelihood(self, X: ArrayLike) -> NDArray[np.float64]:
        """Compute log P(class) + log P(features | class)."""
        features = self._validate_inference_features(X)
        if (
            self.class_prior_ is None
            or self.theta_ is None
            or self.var_ is None
        ):
            raise RuntimeError("GaussianNB fitted parameters are unavailable.")

        n_classes = self.class_prior_.shape[0]
        joint = np.empty((features.shape[0], n_classes), dtype=np.float64)

        for class_index in range(n_classes):
            variance = self.var_[class_index]
            mean = self.theta_[class_index]
            log_prior = np.log(self.class_prior_[class_index])
            log_density = -0.5 * np.sum(
                np.log(2.0 * np.pi * variance)
                + ((features - mean) ** 2) / variance,
                axis=1,
            )
            joint[:, class_index] = log_prior + log_density

        return joint

    def _reset_parameters(self) -> None:
        """Clear learned parameters before fitting or refitting."""
        self._reset_fit_state()
        self.classes_ = None
        self.class_count_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.epsilon_ = None


def _logsumexp(
    values: NDArray[np.float64],
    *,
    axis: int,
) -> NDArray[np.float64]:
    """Compute log(sum(exp(values))) stably along one axis."""
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    summed = np.sum(np.exp(shifted), axis=axis)
    return np.squeeze(maximum, axis=axis) + np.log(summed)


__all__ = ["GaussianNB"]
