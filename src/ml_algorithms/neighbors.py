"""Nearest-neighbour estimators."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._base import BaseEstimator, PredictorMixin
from ml_algorithms._validation import validate_X_y, validate_positive_integer
from ml_algorithms.linalg import pairwise_distances
from ml_algorithms.types import FeatureMatrix


class KNeighborsClassifier(BaseEstimator, PredictorMixin):
    """k-nearest-neighbours classifier using Euclidean distance.

    Parameters
    ----------
    n_neighbors:
        Number of training samples used for each prediction.
    """

    def __init__(self, *, n_neighbors: int = 5) -> None:
        n_neighbors = validate_positive_integer(n_neighbors, name="n_neighbors")

        super().__init__()
        self.n_neighbors = n_neighbors
        self.X_train_: FeatureMatrix | None = None
        self.y_train_: NDArray[Any] | None = None
        self.classes_: NDArray[Any] | None = None
        self._encoded_y: NDArray[np.int64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Store the training data used for neighbour lookup."""
        self._reset_parameters()
        if y is None:
            raise ValueError("y is required for k-nearest neighbours.")

        features, target = validate_X_y(X, y)
        if self.n_neighbors > features.shape[0]:
            raise ValueError(
                "n_neighbors cannot exceed the number of training samples."
            )

        classes, encoded = np.unique(target, return_inverse=True)
        self.X_train_ = features.copy()
        self.y_train_ = np.asarray(target).copy()
        self.classes_ = np.asarray(classes).copy()
        self._encoded_y = np.asarray(encoded, dtype=np.int64)

        self._mark_fitted(n_features=features.shape[1])
        return self

    def kneighbors(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Return distances and indices for the nearest neighbours."""
        queries = self._validate_inference_features(X)
        if self.X_train_ is None:
            raise RuntimeError("Training features are unavailable.")

        distances = pairwise_distances(queries, self.X_train_)

        neighbor_indices = np.argsort(distances, axis=1, kind="stable")[
            :, : self.n_neighbors
        ]
        neighbor_distances = np.take_along_axis(
            distances,
            neighbor_indices,
            axis=1,
        )
        return neighbor_distances, neighbor_indices.astype(np.int64, copy=False)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimate class probabilities from neighbour vote frequencies."""
        _, neighbor_indices = self.kneighbors(X)
        if self.classes_ is None or self._encoded_y is None:
            raise RuntimeError("Fitted class information is unavailable.")

        probabilities = np.zeros(
            (neighbor_indices.shape[0], self.classes_.shape[0]),
            dtype=np.float64,
        )

        for row_index, indices in enumerate(neighbor_indices):
            counts = np.bincount(
                self._encoded_y[indices],
                minlength=self.classes_.shape[0],
            )
            probabilities[row_index] = counts / self.n_neighbors

        return probabilities

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict the class with the largest neighbour vote count."""
        probabilities = self.predict_proba(X)
        if self.classes_ is None:
            raise RuntimeError("Fitted classes are unavailable.")

        winning_indices = np.argmax(probabilities, axis=1)
        return self.classes_[winning_indices]

    def _reset_parameters(self) -> None:
        """Clear stored training data before fitting or refitting."""
        self._reset_fit_state()
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self._encoded_y = None


__all__ = ["KNeighborsClassifier"]
