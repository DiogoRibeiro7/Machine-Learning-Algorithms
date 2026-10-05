"""Clustering estimators."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._base import BaseEstimator, PredictorMixin, TransformerMixin
from ml_algorithms._random import resolve_random_state
from ml_algorithms.linalg import pairwise_distances, pairwise_squared_distances


class KMeans(BaseEstimator, PredictorMixin, TransformerMixin):
    """k-means clustering with deterministic initialization controls.

    Parameters
    ----------
    n_clusters:
        Number of clusters.
    init:
        Either "k-means++" or "random".
    max_iter:
        Maximum number of centroid-update iterations.
    tol:
        Convergence tolerance on maximum centroid displacement.
    random_state:
        Integer seed, NumPy Generator, or None.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
            raise TypeError("n_clusters must be an integer.")
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if init not in {"k-means++", "random"}:
            raise ValueError('init must be "k-means++" or "random".')
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")

        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_: NDArray[np.float64] | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.converged_: bool | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit k-means to the feature matrix."""
        del y
        self._reset_parameters()
        features = self._validate_fit_features(X)
        if self.n_clusters > features.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples.")

        rng = resolve_random_state(self.random_state)
        centers = self._initialize_centers(features, rng)
        labels = np.zeros(features.shape[0], dtype=np.int64)
        converged = False
        n_iter = 0

        for _ in range(self.max_iter):
            n_iter += 1
            labels = _assign_labels(features, centers)
            new_centers = self._recompute_centers(features, labels, centers)
            displacement = np.linalg.norm(new_centers - centers, axis=1)
            centers = new_centers
            if float(np.max(displacement)) <= self.tol:
                converged = True
                break

        labels = _assign_labels(features, centers)
        inertia = _compute_inertia(features, centers, labels)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        self.converged_ = converged
        self._mark_fitted(n_features=features.shape[1])
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Assign each sample to its nearest fitted centroid."""
        features = self._validate_inference_features(X)
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans cluster centers are unavailable.")
        return _assign_labels(features, self.cluster_centers_)

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return Euclidean distances from samples to all centroids."""
        features = self._validate_inference_features(X)
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans cluster centers are unavailable.")
        return _pairwise_distances(features, self.cluster_centers_)

    def fit_predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Fit the model and return training-set cluster labels."""
        self.fit(X)
        if self.labels_ is None:
            raise RuntimeError("KMeans fitted labels are unavailable.")
        return self.labels_.copy()

    def _initialize_centers(
        self,
        features: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Initialize centroids using the configured strategy."""
        if self.init == "random":
            indices = rng.choice(
                features.shape[0],
                size=self.n_clusters,
                replace=False,
            )
            return features[indices].copy()

        centers = np.empty(
            (self.n_clusters, features.shape[1]),
            dtype=np.float64,
        )
        first_index = int(rng.integers(features.shape[0]))
        centers[0] = features[first_index]

        closest_sq = pairwise_squared_distances(features, centers[[0]])[:, 0]
        for center_index in range(1, self.n_clusters):
            total = float(np.sum(closest_sq))
            if total <= 0.0:
                remaining = np.arange(features.shape[0])
                chosen = int(rng.choice(remaining))
            else:
                probabilities = closest_sq / total
                chosen = int(rng.choice(features.shape[0], p=probabilities))

            centers[center_index] = features[chosen]
            new_sq = pairwise_squared_distances(features, centers[[center_index]])[:, 0]
            closest_sq = np.minimum(closest_sq, new_sq)

        return centers

    def _recompute_centers(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        previous_centers: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Recompute centroids and recover empty clusters deterministically."""
        centers = np.empty_like(previous_centers)
        empty_clusters: list[int] = []

        for cluster_index in range(self.n_clusters):
            members = features[labels == cluster_index]
            if members.shape[0] == 0:
                empty_clusters.append(cluster_index)
            else:
                centers[cluster_index] = np.mean(members, axis=0)

        if empty_clusters:
            assigned_centers = previous_centers[labels]
            squared_distance = np.sum((features - assigned_centers) ** 2, axis=1)
            order = np.argsort(-squared_distance, kind="stable")
            used_samples: set[int] = set()

            for cluster_index in empty_clusters:
                for sample_index in order:
                    candidate = int(sample_index)
                    if candidate not in used_samples:
                        centers[cluster_index] = features[candidate]
                        used_samples.add(candidate)
                        break

        return centers

    def _reset_parameters(self) -> None:
        """Clear learned parameters before fitting or refitting."""
        self._reset_fit_state()
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.converged_ = None


def _assign_labels(
    features: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return nearest-centroid labels."""
    distances = pairwise_distances(features, centers)
    return np.argmin(distances, axis=1).astype(np.int64, copy=False)


def _compute_inertia(
    features: NDArray[np.float64],
    centers: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    """Return within-cluster sum of squared distances."""
    residuals = features - centers[labels]
    return float(np.sum(residuals * residuals))


__all__ = ["KMeans"]
