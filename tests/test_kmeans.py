"""Tests for k-means clustering."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms import KMeans, NotFittedError


def test_finds_two_obvious_clusters() -> None:
    X = np.array(
        [[0.0], [0.2], [0.4], [9.6], [9.8], [10.0]],
        dtype=np.float64,
    )

    estimator = KMeans(n_clusters=2, random_state=7).fit(X)

    assert estimator.cluster_centers_ is not None
    centers = np.sort(estimator.cluster_centers_[:, 0])
    assert centers.tolist() == pytest.approx([0.2, 9.8], abs=1e-12)
    assert estimator.inertia_ == pytest.approx(0.16, abs=1e-12)
    assert estimator.converged_ is True


def test_random_state_is_reproducible() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 2))

    first = KMeans(n_clusters=3, random_state=42).fit(X)
    second = KMeans(n_clusters=3, random_state=42).fit(X)

    assert first.cluster_centers_ is not None
    assert second.cluster_centers_ is not None
    assert bool(
        np.allclose(
            first.cluster_centers_,
            second.cluster_centers_,
            atol=1e-12,
            rtol=0.0,
        )
    )
    assert first.labels_ is not None
    assert second.labels_ is not None
    assert first.labels_.tolist() == second.labels_.tolist()


def test_transform_returns_distances_to_all_centroids() -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float64)
    estimator = KMeans(n_clusters=2, random_state=0).fit(X)

    distances = estimator.transform([[0.5]])

    assert distances.shape == (1, 2)
    assert float(np.min(distances)) == pytest.approx(0.0, abs=1e-12)


def test_fit_predict_matches_stored_labels() -> None:
    X = np.array([[0.0], [0.1], [5.0], [5.1]], dtype=np.float64)
    estimator = KMeans(n_clusters=2, random_state=0)

    labels = estimator.fit_predict(X)

    assert estimator.labels_ is not None
    assert labels.tolist() == estimator.labels_.tolist()


def test_empty_cluster_recovery_keeps_centers_finite() -> None:
    X = np.array([[0.0], [0.0], [0.0], [10.0], [10.0]], dtype=np.float64)

    estimator = KMeans(
        n_clusters=3,
        init="random",
        random_state=1,
        max_iter=20,
    ).fit(X)

    assert estimator.cluster_centers_ is not None
    assert bool(np.isfinite(estimator.cluster_centers_).all())
    assert estimator.labels_ is not None
    assert estimator.labels_.shape == (5,)


def test_prediction_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        KMeans(n_clusters=2).predict([[0.0]])


def test_more_clusters_than_samples_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        KMeans(n_clusters=3).fit([[0.0], [1.0]])


def test_invalid_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError, match="k-means"):
        KMeans(init="first")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        KMeans(tol=0.0)
