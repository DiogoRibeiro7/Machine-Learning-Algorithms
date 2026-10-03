"""Smoke tests for the package foundation."""

from __future__ import annotations

import ml_algorithms


def test_package_imports() -> None:
    """The top-level package should expose its core estimator API."""
    expected_exports = {
        "BaseEstimator",
        "NotFittedError",
        "PredictorMixin",
        "TransformerMixin",
    }

    assert expected_exports <= set(ml_algorithms.__all__)
