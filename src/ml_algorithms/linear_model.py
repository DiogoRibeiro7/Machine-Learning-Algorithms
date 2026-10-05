"""Linear models implemented from first principles."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._base import BaseEstimator, PredictorMixin
from ml_algorithms._validation import (
    validate_X_y,
    validate_non_negative_real,
    validate_positive_integer,
    validate_positive_real,
)
from ml_algorithms.linalg import least_squares


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

        least_squares_result = least_squares(design, target_array)
        solution = least_squares_result.solution

        if self.fit_intercept:
            self.intercept_ = float(solution[0])
            self.coef_ = np.asarray(solution[1:], dtype=np.float64).copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(solution, dtype=np.float64).copy()

        self.rank_ = least_squares_result.rank
        self.singular_values_ = least_squares_result.singular_values.copy()
        self.residual_sum_squares_ = least_squares_result.residual_sum_squares

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


class LogisticRegression(BaseEstimator, PredictorMixin):
    """Binary logistic regression fitted with Newton iterations.

    Parameters
    ----------
    fit_intercept:
        Whether to estimate an intercept.
    l2:
        L2 penalty strength applied to slope coefficients only.
    max_iter:
        Maximum number of Newton iterations.
    tol:
        Absolute tolerance on the largest parameter update.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        l2: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        l2 = validate_non_negative_real(l2, name="l2")
        max_iter = validate_positive_integer(max_iter, name="max_iter")
        tol = validate_positive_real(tol, name="tol")

        super().__init__()
        self.fit_intercept = fit_intercept
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.classes_: NDArray[np.int64] | None = None
        self.n_iter_: int | None = None
        self.converged_: bool | None = None
        self.loss_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit binary logistic regression."""
        self._reset_parameters()
        if y is None:
            raise ValueError("y is required for logistic regression.")

        features, target = validate_X_y(X, y)
        target_array = np.asarray(target)
        classes = np.unique(target_array)
        if classes.shape[0] != 2:
            raise ValueError("y must contain exactly two distinct classes.")

        negative_class, positive_class = classes[0], classes[1]
        binary = (target_array == positive_class).astype(np.float64)

        if self.fit_intercept:
            design = np.concatenate(
                (np.ones((features.shape[0], 1), dtype=np.float64), features),
                axis=1,
            )
        else:
            design = features

        theta = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.full(design.shape[1], self.l2, dtype=np.float64)
        if self.fit_intercept:
            penalty[0] = 0.0

        converged = False
        n_samples = float(design.shape[0])
        n_iter = 0

        for _ in range(self.max_iter):
            n_iter += 1
            logits = design @ theta
            probabilities = _stable_sigmoid(logits)
            gradient = design.T @ (probabilities - binary) / n_samples
            gradient = gradient + penalty * theta

            weights = probabilities * (1.0 - probabilities)
            hessian = design.T @ (design * weights[:, None]) / n_samples
            hessian = hessian + np.diag(penalty)

            step = least_squares(hessian, gradient).solution
            current_loss = _logistic_loss(design, binary, theta, penalty)

            scale = 1.0
            candidate = theta - step
            for _ in range(25):
                candidate = theta - scale * step
                candidate_loss = _logistic_loss(
                    design,
                    binary,
                    candidate,
                    penalty,
                )
                if candidate_loss <= current_loss:
                    break
                scale *= 0.5

            update = candidate - theta
            theta = candidate
            if float(np.max(np.abs(update))) <= self.tol:
                converged = True
                break

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = np.asarray(theta[1:], dtype=np.float64).copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(theta, dtype=np.float64).copy()

        self.classes_ = np.asarray([negative_class, positive_class], dtype=np.int64)
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.loss_ = _logistic_loss(design, binary, theta, penalty)
        self._mark_fitted(n_features=features.shape[1])
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return class probabilities in classes_ order."""
        features = self._validate_inference_features(X)
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LogisticRegression fitted parameters are unavailable.")

        positive = _stable_sigmoid(features @ self.coef_ + self.intercept_)
        negative = 1.0 - positive
        return np.column_stack((negative, positive))

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Predict binary class labels using a 0.5 probability threshold."""
        probabilities = self.predict_proba(X)[:, 1]
        if self.classes_ is None:
            raise RuntimeError("LogisticRegression fitted classes are unavailable.")

        indices = (probabilities >= 0.5).astype(np.int64)
        return self.classes_[indices]

    def _reset_parameters(self) -> None:
        """Clear learned parameters before fitting or refitting."""
        self._reset_fit_state()
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_iter_ = None
        self.converged_ = None
        self.loss_ = None


def _stable_sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the logistic sigmoid without overflow."""
    output = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return output


def _logistic_loss(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    theta: NDArray[np.float64],
    penalty: NDArray[np.float64],
) -> float:
    """Return mean logistic loss plus an L2 penalty."""
    logits = design @ theta
    data_loss = np.mean(np.logaddexp(0.0, logits) - target * logits)
    regularization = 0.5 * np.sum(penalty * theta * theta)
    return float(data_loss + regularization)


__all__ = ["LinearRegression", "LogisticRegression"]
