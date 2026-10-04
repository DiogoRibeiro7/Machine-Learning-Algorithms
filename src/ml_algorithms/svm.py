"""Support-vector machine estimators."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_algorithms._base import BaseEstimator, PredictorMixin
from ml_algorithms._validation import validate_X_y


class LinearSVM(BaseEstimator, PredictorMixin):
    """Binary linear soft-margin SVM with squared hinge loss.

    The estimator minimizes

        0.5 * ||w||^2 + C * mean(max(0, 1 - y f(x))^2),

    where the intercept is not regularized.

    Parameters
    ----------
    C:
        Weight of the squared-hinge data-fit term.
    fit_intercept:
        Whether to estimate an intercept.
    max_iter:
        Maximum number of Newton iterations.
    tol:
        Absolute tolerance on the largest parameter update.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        if isinstance(C, bool) or not isinstance(C, (int, float)):
            raise TypeError("C must be a real number.")
        if C <= 0.0:
            raise ValueError("C must be positive.")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")

        super().__init__()
        self.C = float(C)
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = float(tol)
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.classes_: NDArray[Any] | None = None
        self.n_iter_: int | None = None
        self.converged_: bool | None = None
        self.loss_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the binary linear SVM."""
        self._reset_parameters()
        if y is None:
            raise ValueError("y is required for LinearSVM.")

        features, target = validate_X_y(X, y)
        classes = np.unique(target)
        if classes.shape[0] != 2:
            raise ValueError("y must contain exactly two distinct classes.")

        negative_class, positive_class = classes[0], classes[1]
        signed_target = np.where(target == positive_class, 1.0, -1.0)

        if self.fit_intercept:
            design = np.concatenate(
                (np.ones((features.shape[0], 1), dtype=np.float64), features),
                axis=1,
            )
            regularization = np.ones(design.shape[1], dtype=np.float64)
            regularization[0] = 0.0
        else:
            design = features
            regularization = np.ones(design.shape[1], dtype=np.float64)

        theta = np.zeros(design.shape[1], dtype=np.float64)
        converged = False
        n_iter = 0

        for _ in range(self.max_iter):
            n_iter += 1
            scores = design @ theta
            margins = signed_target * scores
            active = margins < 1.0

            gradient = regularization * theta
            hessian = np.diag(regularization)

            if bool(np.any(active)):
                active_design = design[active]
                active_target = signed_target[active]
                residual = 1.0 - margins[active]
                gradient -= (
                    2.0
                    * self.C
                    / features.shape[0]
                    * (active_design.T @ (active_target * residual))
                )
                hessian += (
                    2.0
                    * self.C
                    / features.shape[0]
                    * (active_design.T @ active_design)
                )

            step, _, _, _ = np.linalg.lstsq(hessian, gradient, rcond=None)
            current_loss = _squared_hinge_objective(
                design,
                signed_target,
                theta,
                regularization,
                self.C,
            )

            scale = 1.0
            candidate = theta - step
            for _ in range(25):
                candidate = theta - scale * step
                candidate_loss = _squared_hinge_objective(
                    design,
                    signed_target,
                    candidate,
                    regularization,
                    self.C,
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

        self.classes_ = np.asarray([negative_class, positive_class]).copy()
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.loss_ = _squared_hinge_objective(
            design,
            signed_target,
            theta,
            regularization,
            self.C,
        )
        self._mark_fitted(n_features=features.shape[1])
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return signed distances up to the scale of the fitted coefficients."""
        features = self._validate_inference_features(X)
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LinearSVM fitted parameters are unavailable.")
        return features @ self.coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict the original binary class labels."""
        scores = self.decision_function(X)
        if self.classes_ is None:
            raise RuntimeError("LinearSVM fitted classes are unavailable.")
        indices = (scores >= 0.0).astype(np.int64)
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


def _squared_hinge_objective(
    design: NDArray[np.float64],
    signed_target: NDArray[np.float64],
    theta: NDArray[np.float64],
    regularization: NDArray[np.float64],
    C: float,
) -> float:
    """Return the regularized mean squared-hinge objective."""
    margins = signed_target * (design @ theta)
    residual = np.maximum(0.0, 1.0 - margins)
    penalty = 0.5 * np.sum(regularization * theta * theta)
    data_fit = C * np.mean(residual * residual)
    return float(penalty + data_fit)


__all__ = ["LinearSVM"]
