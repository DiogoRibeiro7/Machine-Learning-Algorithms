"""Run deterministic local benchmarks for maintained estimators."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ml_algorithms import (
    PCA,
    GaussianNB,
    KMeans,
    KNeighborsClassifier,
    LinearRegression,
    LinearSVM,
    LogisticRegression,
)
from ml_algorithms.metrics import accuracy_score, mean_squared_error, r2_score

type TimedOperation = Callable[[], object]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for one benchmark run."""

    sample_sizes: tuple[int, ...]
    feature_counts: tuple[int, ...]
    fixed_samples: int
    fixed_features: int
    n_eval: int
    repeat: int
    seed: int
    output: Path | None


@dataclass(frozen=True, slots=True)
class BenchmarkContext:
    """Dataset dimensions and timing controls for one benchmark case."""

    n_samples: int
    n_features: int
    n_eval: int
    repeat: int
    seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Timing and numerical-quality measurements for one estimator."""

    fit_seconds: float
    inference_seconds: float
    quality_metric: str
    quality_value: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One row in the benchmark report."""

    algorithm: str
    sweep: str
    n_samples: int
    n_features: int
    fit_seconds: float
    inference_seconds: float
    quality_metric: str
    quality_value: float


def run_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run both scaling sweeps for every maintained estimator."""
    runners: tuple[
        tuple[str, Callable[[BenchmarkContext], BenchmarkMeasurement]], ...
    ] = (
        ("linear_regression", _benchmark_linear_regression),
        ("logistic_regression", _benchmark_logistic_regression),
        ("knn_classifier", _benchmark_knn),
        ("gaussian_nb", _benchmark_gaussian_nb),
        ("kmeans", _benchmark_kmeans),
        ("pca", _benchmark_pca),
        ("linear_svm", _benchmark_linear_svm),
    )

    results: list[BenchmarkResult] = []
    for algorithm, runner in runners:
        for n_samples in config.sample_sizes:
            context = BenchmarkContext(
                n_samples=n_samples,
                n_features=config.fixed_features,
                n_eval=config.n_eval,
                repeat=config.repeat,
                seed=config.seed,
            )
            results.append(
                _result_from_measurement(
                    algorithm,
                    "sample_count",
                    context,
                    runner(context),
                )
            )

        for n_features in config.feature_counts:
            context = BenchmarkContext(
                n_samples=config.fixed_samples,
                n_features=n_features,
                n_eval=config.n_eval,
                repeat=config.repeat,
                seed=config.seed,
            )
            results.append(
                _result_from_measurement(
                    algorithm,
                    "feature_count",
                    context,
                    runner(context),
                )
            )

    return results


def _benchmark_linear_regression(
    context: BenchmarkContext,
) -> BenchmarkMeasurement:
    X_train, y_train, X_eval, y_eval = _regression_data(context)

    fit_seconds = _median_seconds(
        lambda: LinearRegression().fit(X_train, y_train),
        context.repeat,
    )
    model = LinearRegression().fit(X_train, y_train)
    inference_seconds = _median_seconds(
        lambda: model.predict(X_eval),
        context.repeat,
    )
    quality = r2_score(y_eval, model.predict(X_eval))
    return BenchmarkMeasurement(
        fit_seconds,
        inference_seconds,
        "r2",
        quality,
    )


def _benchmark_logistic_regression(
    context: BenchmarkContext,
) -> BenchmarkMeasurement:
    X_train, y_train, X_eval, y_eval = _classification_data(context)

    fit_seconds = _median_seconds(
        lambda: LogisticRegression(l2=0.1).fit(X_train, y_train),
        context.repeat,
    )
    model = LogisticRegression(l2=0.1).fit(X_train, y_train)
    inference_seconds = _median_seconds(
        lambda: model.predict(X_eval),
        context.repeat,
    )
    quality = accuracy_score(y_eval, model.predict(X_eval))
    return BenchmarkMeasurement(
        fit_seconds,
        inference_seconds,
        "accuracy",
        quality,
    )


def _benchmark_knn(context: BenchmarkContext) -> BenchmarkMeasurement:
    X_train, y_train, X_eval, y_eval = _classification_data(context)

    fit_seconds = _median_seconds(
        lambda: KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        context.repeat,
    )
    model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    inference_seconds = _median_seconds(
        lambda: model.predict(X_eval),
        context.repeat,
    )
    quality = accuracy_score(y_eval, model.predict(X_eval))
    return BenchmarkMeasurement(
        fit_seconds,
        inference_seconds,
        "accuracy",
        quality,
    )


def _benchmark_gaussian_nb(
    context: BenchmarkContext,
) -> BenchmarkMeasurement:
    X_train, y_train, X_eval, y_eval = _classification_data(context)

    fit_seconds = _median_seconds(
        lambda: GaussianNB().fit(X_train, y_train),
        context.repeat,
    )
    model = GaussianNB().fit(X_train, y_train)
    inference_seconds = _median_seconds(
        lambda: model.predict(X_eval),
        context.repeat,
    )
    quality = accuracy_score(y_eval, model.predict(X_eval))
    return BenchmarkMeasurement(
        fit_seconds,
        inference_seconds,
        "accuracy",
        quality,
    )


def _benchmark_kmeans(context: BenchmarkContext) -> BenchmarkMeasurement:
    X_train, X_eval = _clustering_data(context)

    fit_seconds = _median_seconds(
        lambda: KMeans(n_clusters=3, random_state=context.seed).fit(X_train),
        context.repeat,
    )
    model = KMeans(n_clusters=3, random_state=context.seed).fit(X_train)
    inference_seconds = _median_seconds(
        lambda: model.transform(X_eval),
        context.repeat,
    )
    if model.inertia_ is None:
        raise RuntimeError("KMeans benchmark did not produce inertia.")
    quality = model.inertia_ / (context.n_samples * context.n_features)
    return BenchmarkMeasurement(
        fit_seconds,
        inference_seconds,
        "inertia_per_element",
        quality,
    )


def _benchmark_pca(context: BenchmarkContext) -> BenchmarkMeasurement:
    X_train, X_eval = _pca_data(context)
    n_components = min(5, context.n_features)

    fit_seconds = _median_seconds(
        lambda: PCA(n_components=n_components).fit(X_train),
        context.repeat,
    )
    model = PCA(n_components=n_components).fit(X_train)
    inference_seconds = _median_seconds(
        lambda: model.transform(X_eval),
        context.repeat,
    )
    transformed = model.transform(X_eval)
    reconstructed = model.inverse_transform(transformed)
    quality = mean_squared_error(X_eval.ravel(), reconstructed.ravel())
    return BenchmarkMeasurement(
        fit_seconds,
        inference_seconds,
        "reconstruction_mse",
        quality,
    )


def _benchmark_linear_svm(
    context: BenchmarkContext,
) -> BenchmarkMeasurement:
    X_train, y_train, X_eval, y_eval = _classification_data(context)

    fit_seconds = _median_seconds(
        lambda: LinearSVM(C=1.0).fit(X_train, y_train),
        context.repeat,
    )
    model = LinearSVM(C=1.0).fit(X_train, y_train)
    inference_seconds = _median_seconds(
        lambda: model.predict(X_eval),
        context.repeat,
    )
    quality = accuracy_score(y_eval, model.predict(X_eval))
    return BenchmarkMeasurement(
        fit_seconds,
        inference_seconds,
        "accuracy",
        quality,
    )


def _regression_data(
    context: BenchmarkContext,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Generate deterministic noisy linear-regression data."""
    rng = np.random.default_rng(context.seed)
    coefficients = np.linspace(0.5, 1.5, context.n_features, dtype=np.float64)
    X_train = rng.normal(size=(context.n_samples, context.n_features))
    X_eval = rng.normal(size=(context.n_eval, context.n_features))
    y_train = X_train @ coefficients + rng.normal(
        scale=0.1,
        size=context.n_samples,
    )
    y_eval = X_eval @ coefficients + rng.normal(
        scale=0.1,
        size=context.n_eval,
    )
    return X_train, y_train, X_eval, y_eval


def _classification_data(
    context: BenchmarkContext,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Generate deterministic approximately linearly separable data."""
    rng = np.random.default_rng(context.seed)
    weights = np.linspace(-1.0, 1.0, context.n_features, dtype=np.float64)
    X_train = rng.normal(size=(context.n_samples, context.n_features))
    X_eval = rng.normal(size=(context.n_eval, context.n_features))
    train_score = X_train @ weights + rng.normal(
        scale=0.35,
        size=context.n_samples,
    )
    eval_score = X_eval @ weights + rng.normal(
        scale=0.35,
        size=context.n_eval,
    )
    y_train = (train_score >= 0.0).astype(np.int64)
    y_eval = (eval_score >= 0.0).astype(np.int64)
    _ensure_two_classes(y_train, train_score)
    return X_train, y_train, X_eval, y_eval


def _ensure_two_classes(
    target: NDArray[np.int64],
    score: NDArray[np.float64],
) -> None:
    """Ensure synthetic training labels contain both binary classes."""
    if np.unique(target).shape[0] == 2:
        return
    target[int(np.argmin(score))] = 0
    target[int(np.argmax(score))] = 1


def _clustering_data(
    context: BenchmarkContext,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate three deterministic spherical clusters."""
    rng = np.random.default_rng(context.seed)
    centers = np.vstack(
        (
            np.full(context.n_features, -3.0),
            np.zeros(context.n_features),
            np.full(context.n_features, 3.0),
        )
    )
    train_labels = np.arange(context.n_samples) % 3
    eval_labels = np.arange(context.n_eval) % 3
    X_train = centers[train_labels] + rng.normal(
        scale=0.5,
        size=(context.n_samples, context.n_features),
    )
    X_eval = centers[eval_labels] + rng.normal(
        scale=0.5,
        size=(context.n_eval, context.n_features),
    )
    rng.shuffle(X_train)
    rng.shuffle(X_eval)
    return X_train, X_eval


def _pca_data(
    context: BenchmarkContext,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate deterministic low-rank data with small isotropic noise."""
    rng = np.random.default_rng(context.seed)
    latent_dim = min(3, context.n_features)
    loadings = rng.normal(size=(latent_dim, context.n_features))
    train_latent = rng.normal(size=(context.n_samples, latent_dim))
    eval_latent = rng.normal(size=(context.n_eval, latent_dim))
    X_train = train_latent @ loadings + rng.normal(
        scale=0.05,
        size=(context.n_samples, context.n_features),
    )
    X_eval = eval_latent @ loadings + rng.normal(
        scale=0.05,
        size=(context.n_eval, context.n_features),
    )
    return X_train, X_eval


def _median_seconds(operation: TimedOperation, repeat: int) -> float:
    """Return the median wall-clock duration over repeated calls."""
    durations: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        operation()
        durations.append(perf_counter() - start)
    return float(median(durations))


def _result_from_measurement(
    algorithm: str,
    sweep: str,
    context: BenchmarkContext,
    measurement: BenchmarkMeasurement,
) -> BenchmarkResult:
    """Attach benchmark metadata to one measurement."""
    return BenchmarkResult(
        algorithm=algorithm,
        sweep=sweep,
        n_samples=context.n_samples,
        n_features=context.n_features,
        fit_seconds=measurement.fit_seconds,
        inference_seconds=measurement.inference_seconds,
        quality_metric=measurement.quality_metric,
        quality_value=measurement.quality_value,
    )


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    """Print a compact human-readable benchmark table."""
    header = (
        f"{'algorithm':<20} {'sweep':<14} {'n':>6} {'p':>5} "
        f"{'fit ms':>10} {'infer ms':>10} {'metric':<20} {'value':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.algorithm:<20} {result.sweep:<14} "
            f"{result.n_samples:>6d} {result.n_features:>5d} "
            f"{1_000.0 * result.fit_seconds:>10.3f} "
            f"{1_000.0 * result.inference_seconds:>10.3f} "
            f"{result.quality_metric:<20} {result.quality_value:>10.4f}"
        )


def _write_csv(results: Sequence[BenchmarkResult], output: Path) -> None:
    """Write benchmark results to a deterministic CSV schema."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "algorithm",
                "sweep",
                "n_samples",
                "n_features",
                "fit_seconds",
                "inference_seconds",
                "quality_metric",
                "quality_value",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.algorithm,
                    result.sweep,
                    result.n_samples,
                    result.n_features,
                    f"{result.fit_seconds:.9f}",
                    f"{result.inference_seconds:.9f}",
                    result.quality_metric,
                    f"{result.quality_value:.9f}",
                ]
            )


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer command-line value."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> BenchmarkConfig:
    """Parse command-line arguments into a typed benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Benchmark maintained machine-learning estimators.",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=_positive_int,
        default=[100, 500, 1_000],
    )
    parser.add_argument(
        "--feature-counts",
        nargs="+",
        type=_positive_int,
        default=[2, 8, 32],
    )
    parser.add_argument("--fixed-samples", type=_positive_int, default=500)
    parser.add_argument("--fixed-features", type=_positive_int, default=8)
    parser.add_argument("--n-eval", type=_positive_int, default=128)
    parser.add_argument("--repeat", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)

    namespace = parser.parse_args(argv)
    return BenchmarkConfig(
        sample_sizes=tuple(cast(list[int], namespace.sample_sizes)),
        feature_counts=tuple(cast(list[int], namespace.feature_counts)),
        fixed_samples=cast(int, namespace.fixed_samples),
        fixed_features=cast(int, namespace.fixed_features),
        n_eval=cast(int, namespace.n_eval),
        repeat=cast(int, namespace.repeat),
        seed=cast(int, namespace.seed),
        output=cast(Path | None, namespace.output),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured benchmark suite."""
    config = parse_args(argv)
    results = run_benchmarks(config)
    _print_results(results)
    if config.output is not None:
        _write_csv(results, config.output)
        print(f"\nWrote {len(results)} rows to {config.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
