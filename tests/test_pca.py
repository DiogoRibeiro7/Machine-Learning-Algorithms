"""Tests for principal component analysis."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms import NotFittedError, PCA


def test_single_component_matches_known_axis() -> None:
    X = np.array(
        [[8.0, 5.0], [9.0, 5.0], [11.0, 5.0], [12.0, 5.0]],
        dtype=np.float64,
    )

    estimator = PCA(n_components=1).fit(X)

    assert estimator.mean_ is not None
    assert estimator.components_ is not None
    assert estimator.explained_variance_ is not None
    assert estimator.explained_variance_ratio_ is not None
    assert estimator.mean_.tolist() == pytest.approx([10.0, 5.0])
    assert estimator.components_[0].tolist() == pytest.approx([1.0, 0.0])
    assert estimator.explained_variance_[0] == pytest.approx(10.0 / 3.0)
    assert estimator.explained_variance_ratio_[0] == pytest.approx(1.0)
    assert estimator.transform(X)[:, 0].tolist() == pytest.approx(
        [-2.0, -1.0, 1.0, 2.0]
    )


def test_full_components_reconstruct_original_data() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3)) + np.array([3.0, -2.0, 5.0])

    estimator = PCA(n_components=3).fit(X)
    reconstructed = estimator.inverse_transform(estimator.transform(X))

    assert reconstructed.tolist() == pytest.approx(X.tolist(), abs=1e-12)


def test_full_explained_variance_ratio_sums_to_one() -> None:
    X = np.array(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0], [4.0, 3.0]],
        dtype=np.float64,
    )

    estimator = PCA().fit(X)

    assert estimator.explained_variance_ratio_ is not None
    assert float(np.sum(estimator.explained_variance_ratio_)) == pytest.approx(1.0)


def test_reduced_transform_has_requested_shape() -> None:
    X = np.arange(30, dtype=np.float64).reshape(10, 3)

    transformed = PCA(n_components=2).fit_transform(X)

    assert transformed.shape == (10, 2)


def test_constant_data_has_zero_explained_variance_ratios() -> None:
    X = np.ones((5, 3), dtype=np.float64)

    estimator = PCA(n_components=2).fit(X)

    assert estimator.explained_variance_ is not None
    assert estimator.explained_variance_ratio_ is not None
    assert estimator.explained_variance_.tolist() == pytest.approx([0.0, 0.0])
    assert estimator.explained_variance_ratio_.tolist() == pytest.approx([0.0, 0.0])
    assert bool(np.isfinite(estimator.components_).all())


def test_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        PCA(n_components=1).transform([[1.0, 2.0]])


def test_inverse_transform_validates_component_count() -> None:
    estimator = PCA(n_components=1).fit([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="incompatible"):
        estimator.inverse_transform([[0.0, 1.0]])


def test_invalid_component_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        PCA(n_components=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        PCA(n_components=3).fit([[0.0, 1.0], [1.0, 0.0]])


def test_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        PCA().fit([[1.0, 2.0]])
