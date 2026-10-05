"""Matrix decomposition estimators."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._base import BaseEstimator, TransformerMixin
from ml_algorithms.linalg import canonicalize_row_signs


class PCA(BaseEstimator, TransformerMixin):
    """Principal component analysis using centered singular value decomposition.

    Parameters
    ----------
    n_components:
        Number of principal components to retain. ``None`` keeps all
        components available from the compact SVD.
    """

    def __init__(self, *, n_components: int | None = None) -> None:
        if n_components is not None:
            if isinstance(n_components, bool) or not isinstance(n_components, int):
                raise TypeError("n_components must be an integer or None.")
            if n_components < 1:
                raise ValueError("n_components must be at least 1.")

        super().__init__()
        self.n_components = n_components
        self.n_components_: int | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.n_samples_seen_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit principal components to a feature matrix."""
        del y
        self._reset_parameters()
        features = self._validate_fit_features(X)

        n_samples, n_features = features.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")

        max_components = min(n_samples, n_features)
        if self.n_components is None:
            n_components = max_components
        else:
            n_components = self.n_components
            if n_components > max_components:
                raise ValueError(
                    "n_components cannot exceed min(n_samples, n_features)."
                )

        mean = np.mean(features, axis=0)
        centered = features - mean
        _, singular_values, right_vectors = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        components = canonicalize_row_signs(right_vectors)
        all_explained_variance = singular_values**2 / (n_samples - 1)
        total_variance = float(np.sum(all_explained_variance))
        if total_variance > 0.0:
            all_ratios = all_explained_variance / total_variance
        else:
            all_ratios = np.zeros_like(all_explained_variance)

        self.n_components_ = n_components
        self.mean_ = np.asarray(mean, dtype=np.float64).copy()
        self.components_ = np.asarray(
            components[:n_components],
            dtype=np.float64,
        ).copy()
        self.explained_variance_ = np.asarray(
            all_explained_variance[:n_components],
            dtype=np.float64,
        ).copy()
        self.explained_variance_ratio_ = np.asarray(
            all_ratios[:n_components],
            dtype=np.float64,
        ).copy()
        self.singular_values_ = np.asarray(
            singular_values[:n_components],
            dtype=np.float64,
        ).copy()
        self.n_samples_seen_ = n_samples
        self._mark_fitted(n_features=n_features)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project centered samples onto the fitted principal components."""
        features = self._validate_inference_features(X)
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA fitted parameters are unavailable.")
        return (features - self.mean_) @ self.components_.T

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Map principal-component coordinates back to feature space."""
        self._check_is_fitted()
        if self.mean_ is None or self.components_ is None or self.n_components_ is None:
            raise RuntimeError("PCA fitted parameters are unavailable.")

        transformed = _validate_component_coordinates(
            X,
            expected_n_components=self.n_components_,
        )
        return transformed @ self.components_ + self.mean_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit PCA and return transformed training coordinates."""
        self.fit(X)
        return self.transform(X)

    def _reset_parameters(self) -> None:
        """Clear learned parameters before fitting or refitting."""
        self._reset_fit_state()
        self.n_components_ = None
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_samples_seen_ = None


def _validate_component_coordinates(
    X: ArrayLike,
    *,
    expected_n_components: int,
) -> NDArray[np.float64]:
    """Validate coordinates supplied to inverse_transform."""
    try:
        array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric component coordinates.") from exc

    if array.ndim != 2:
        raise ValueError("Component coordinates must be two-dimensional.")
    if array.shape[0] == 0:
        raise ValueError("Component coordinates must contain at least one sample.")
    if array.shape[1] != expected_n_components:
        raise ValueError(
            "Component coordinates have an incompatible number of columns."
        )
    if not bool(np.isfinite(array).all()):
        raise ValueError("Component coordinates must contain only finite values.")
    return array


__all__ = ["PCA"]
