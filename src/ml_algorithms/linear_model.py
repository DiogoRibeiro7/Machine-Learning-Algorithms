"""Ordinary least-squares linear regression."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._base import BaseEstimator, PredictorMixin
from ml_algorithms._validation import validate_X_y


class LinearRegression(BaseEstimator, PredictorMixin):
    """Ordinary least-squares linear regression.

    The model minimizes the residual sum of squares using NumPy's least-squares
    solver. The implementation supports an optional intercept and remains
    well-defined for rank-deficient design matrices by returning the
    minimum-norm least-squares solution.

    Parameters
    ----------
    fit_intercept:
        Whether to estimate an intercept term. If False, the fitted intercept
        is exactly zero.
    """

    def __init__(self, *, fit_intercept: bool = True) -> None:
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")

        super().__init__()
        self.fit_intercept = fit_intercept
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.rank_: int | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.residual_sum_squares_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the ordinary least-squares model."""
        self._reset_parameters()

        if y is None:
            raise ValueError("y is required for linear regression.")

        features, target = validate_X_y(X, y)

        try:
            target_array = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must contain numeric values.") from exc

        if not bool(np.isfinite(target_array).all()):
            raise ValueError("y must contain only finite values.")

        if self.fit_intercept:
            intercept_column = np.ones((features.shape[0], 1), dtype=np.float64)
            design = np.concatenate((intercept_column, features), axis=1)
        else:
            design = features

        solution, _, rank, singular_values = np.linalg.lstsq(
            design,
            target_array,
            rcond=None,
        )

        fitted_values = design @ solution
        residuals = target_array - fitted_values

        if self.fit_intercept:
            self.intercept_ = float(solution[0])
            self.coef_ = np.asarray(solution[1:], dtype=np.float64).copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(solution, dtype=np.float64).copy()

        self.rank_ = int(rank)
        self.singular_values_ = np.asarray(
            singular_values,
            dtype=np.float64,
        ).copy()
        self.residual_sum_squares_ = float(residuals @ residuals)

        self._mark_fitted(n_features=features.shape[1])
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict continuous target values."""
        features = self._validate_inference_features(X)

        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LinearRegression fitted parameters are unavailable.")

        return features @ self.coef_ + self.intercept_

    def _reset_parameters(self) -> None:
        """Clear learned parameters before fitting or refitting."""
        self._reset_fit_state()
        self.coef_ = None
        self.intercept_ = None
        self.rank_ = None
        self.singular_values_ = None
        self.residual_sum_squares_ = None


__all__ = ["LinearRegression"]
