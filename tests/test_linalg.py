"""Tests for shared linear-algebra helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ml_algorithms.linalg import (
    canonicalize_row_signs,
    least_squares,
    pairwise_distances,
    pairwise_squared_distances,
)


def test_least_squares_reports_full_rank_diagnostics() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    target = np.array([1.0, 4.0, 5.0])

    result = least_squares(matrix, target)

    assert result.solution.tolist() == pytest.approx([1.0, 2.0], abs=1e-12)
    assert result.residual_sum_squares == pytest.approx(0.0, abs=1e-24)
    assert result.rank == 2
    assert result.singular_values.shape == (2,)
    assert np.isfinite(result.condition_number)
    assert result.condition_number >= 1.0


def test_least_squares_rank_deficiency_returns_minimum_norm_solution() -> None:
    matrix = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    target = np.array([2.0, 4.0, 6.0])

    result = least_squares(matrix, target)

    assert result.solution.tolist() == pytest.approx([1.0, 1.0], abs=1e-12)
    assert result.rank == 1
    assert result.condition_number == float("inf")


def test_least_squares_exposes_large_condition_number() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1e-10]])
    target = np.array([1.0, 1e-10])

    result = least_squares(matrix, target)

    assert result.rank == 2
    assert result.condition_number == pytest.approx(1e10, rel=1e-12)


def test_pairwise_distance_helpers_match_known_geometry() -> None:
    left = np.array([[0.0, 0.0], [3.0, 4.0]])
    right = np.array([[0.0, 0.0], [6.0, 8.0]])

    squared = pairwise_squared_distances(left, right)
    distances = pairwise_distances(left, right)

    assert squared.tolist() == [[0.0, 100.0], [25.0, 25.0]]
    assert distances.tolist() == [[0.0, 10.0], [5.0, 5.0]]


def test_canonicalize_row_signs_is_deterministic() -> None:
    vectors = np.array([[-1.0, 0.2], [0.1, -3.0], [0.0, 0.0]])

    oriented = canonicalize_row_signs(vectors)

    assert oriented.tolist() == [[1.0, -0.2], [-0.1, 3.0], [0.0, 0.0]]
    assert bool(np.allclose(np.abs(oriented), np.abs(vectors)))


def test_distance_helpers_validate_feature_dimension() -> None:
    with pytest.raises(ValueError, match="same number of columns"):
        pairwise_distances([[0.0, 1.0]], [[0.0]])


def test_least_squares_validates_shapes_and_finite_values() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        least_squares([[1.0], [2.0]], [1.0])
    with pytest.raises(ValueError, match="finite"):
        least_squares([[1.0], [np.inf]], [1.0, 2.0])
