"""Tests for benchmark-suite plumbing."""

from __future__ import annotations

from pathlib import Path

from benchmarks.run_benchmarks import BenchmarkConfig, run_benchmarks


def test_benchmark_suite_returns_expected_rows(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        sample_sizes=(12,),
        feature_counts=(2,),
        fixed_samples=12,
        fixed_features=2,
        n_eval=6,
        repeat=1,
        seed=7,
        output=tmp_path / "benchmarks.csv",
    )

    results = run_benchmarks(config)

    assert len(results) == 14
    assert {result.algorithm for result in results} == {
        "linear_regression",
        "logistic_regression",
        "knn_classifier",
        "gaussian_nb",
        "kmeans",
        "pca",
        "linear_svm",
    }
    assert {result.sweep for result in results} == {
        "sample_count",
        "feature_count",
    }
    assert all(result.fit_seconds >= 0.0 for result in results)
    assert all(result.inference_seconds >= 0.0 for result in results)
