"""Property-based tests for numerical invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from ml_algorithms import PCA, KMeans, LinearRegression, LogisticRegression
from ml_algorithms._validation import validate_features


FINITE_FLOAT = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@st.composite
def full_rank_matrix(
    draw: st.DrawFn,
    *,
    min_samples: int = 2,
    max_samples: int = 8,
    min_features: int = 1,
    max_features: int = 4,
) -> NDArray[np.float64]:
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_features = draw(
        st.integers(
            min_value=min_features,
            max_value=min(max_features, n_samples),
        )
    )
    values = draw(
        st.lists(
            FINITE_FLOAT,
            min_size=n_samples * n_features,
            max_size=n_samples * n_features,
        )
    )
    return np.asarray(values, dtype=np.float64).reshape(n_samples, n_features)


@given(
    st.lists(
        FINITE_FLOAT,
        min_size=4,
        max_size=12,
    )
)
@settings(max_examples=25, deadline=None)
def test_logistic_probabilities_sum_to_one(values: list[float]) -> None:
    X = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    split = max(1, X.shape[0] // 2)
    y = np.zeros(X.shape[0], dtype=np.int64)
    y[split:] = 1
    if bool(np.all(y == y[0])):
        y[-1] = 1

    model = LogisticRegression(l2=1.0).fit(X, y)
    probabilities = model.predict_proba(X)

    assert probabilities.shape == (X.shape[0], 2)
    assert bool(
        np.allclose(
            probabilities.sum(axis=1),
            np.ones(X.shape[0]),
            atol=1e-12,
            rtol=0.0,
        )
    )
    assert bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))


@given(full_rank_matrix())
@settings(max_examples=25, deadline=None)
def test_pca_full_components_reconstruct_input(X: NDArray[np.float64]) -> None:
    model = PCA().fit(X)
    reconstructed = model.inverse_transform(model.transform(X))

    assert bool(np.allclose(reconstructed, X, atol=1e-9, rtol=1e-9))


@given(full_rank_matrix(min_samples=3, max_samples=10))
@settings(max_examples=25, deadline=None)
def test_kmeans_inertia_is_non_negative_and_seeded_fit_is_deterministic(
    X: NDArray[np.float64],
) -> None:
    n_clusters = min(3, X.shape[0])
    first = KMeans(n_clusters=n_clusters, random_state=11).fit(X)
    second = KMeans(n_clusters=n_clusters, random_state=11).fit(X)

    assert first.inertia_ is not None
    assert second.inertia_ is not None
    assert first.inertia_ >= 0.0
    assert second.inertia_ >= 0.0
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
    assert np.array_equal(first.labels_, second.labels_)


@given(
    full_rank_matrix(min_samples=2, max_samples=8),
    st.lists(FINITE_FLOAT, min_size=8, max_size=8),
)
@settings(max_examples=20, deadline=None)
def test_linear_regression_prediction_shape_matches_query_rows(
    X: NDArray[np.float64],
    target_values: list[float],
) -> None:
    y = np.asarray(target_values[: X.shape[0]], dtype=np.float64)
    if y.shape[0] < X.shape[0]:
        y = np.resize(y, X.shape[0])

    model = LinearRegression().fit(X, y)
    query = X[: min(3, X.shape[0])]
    predictions = model.predict(query)

    assert predictions.shape == (query.shape[0],)


@given(full_rank_matrix(min_samples=2, max_samples=6))
@settings(max_examples=20, deadline=None)
def test_refit_replaces_linear_regression_feature_contract(X: NDArray[np.float64]) -> None:
    first_features = X.shape[1]
    y = np.arange(X.shape[0], dtype=np.float64)
    model = LinearRegression().fit(X, y)

    replacement = np.column_stack((X, np.ones(X.shape[0], dtype=np.float64)))
    model.fit(replacement, y)

    assert model.n_features_in_ == first_features + 1
    with pytest.raises(ValueError, match="incompatible number of features"):
        model.predict(X)


@given(st.integers(min_value=1, max_value=4))
@settings(max_examples=10, deadline=None)
def test_feature_validation_rejects_non_matrix_dimensions(n_values: int) -> None:
    one_dimensional = np.arange(n_values, dtype=np.float64)

    with pytest.raises(ValueError, match="two-dimensional"):
        validate_features(one_dimensional)
