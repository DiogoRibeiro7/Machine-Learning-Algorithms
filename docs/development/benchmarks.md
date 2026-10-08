# Benchmark suite

The benchmark suite measures runtime and numerical quality for every maintained estimator on deterministic synthetic data.

Run the default suite with:

```bash
poetry run python -m benchmarks.run_benchmarks
```

To write machine-readable output:

```bash
poetry run python -m benchmarks.run_benchmarks --output benchmark-results.csv
```

## What is measured

Each estimator is evaluated under two independent scaling sweeps:

- **sample scaling**: sample count changes while feature count stays fixed;
- **feature scaling**: feature count changes while sample count stays fixed.

For every case the suite records:

- median fit time;
- median inference or transform time;
- sample count;
- feature count;
- one algorithm-appropriate quality metric.

The quality metrics are:

| Estimator | Quality metric |
| --- | --- |
| Linear regression | R² |
| Logistic regression | accuracy |
| k-nearest neighbours | accuracy |
| Gaussian Naive Bayes | accuracy |
| k-means | inertia per matrix element |
| PCA | reconstruction MSE |
| Linear SVM | accuracy |

## Reproducibility

All generated data use an explicit integer seed. The benchmark dimensions and repeat count are configurable from the command line.

The default run uses medians over repeated calls to reduce sensitivity to transient scheduler noise.

## Interpretation

Timing results are **observational**, not correctness thresholds.

Do not compare absolute timings from different machines as if they were directly equivalent. The useful signals are:

- relative behavior among algorithms on the same machine;
- how fit cost changes as sample count grows;
- how fit cost changes as feature count grows;
- whether numerical quality remains stable while dimensions change.

The benchmark suite is intentionally excluded from pass/fail CI timing gates. CI only smoke-tests the benchmark plumbing so that imports, generated data, and estimator calls do not silently break.


## v0.1.0 reference baseline

The first modern release records one reference point from the default benchmark suite at **500 training samples and 8 features**.

Environment:

- GitHub Actions `ubuntu-latest`;
- Python 3.12.14;
- NumPy 2.5.3;
- seed 42;
- 128 evaluation samples;
- median of 3 timing repetitions.

| Estimator | Fit (ms) | Inference / transform (ms) | Quality |
| --- | ---: | ---: | ---: |
| Linear regression | 0.116 | 0.006 | R² = 0.9986 |
| Logistic regression | 0.899 | 0.037 | accuracy = 0.9062 |
| k-nearest neighbours | 0.042 | 7.960 | accuracy = 0.8359 |
| Gaussian Naive Bayes | 0.153 | 0.038 | accuracy = 0.8594 |
| k-means | 0.499 | 0.024 | inertia/element = 0.2489 |
| PCA | 0.114 | 0.007 | reconstruction MSE = 0.0010 |
| Linear SVM | 0.614 | 0.015 | accuracy = 0.8906 |

These timings are a release reference, not a performance contract. Future comparisons should use the same benchmark configuration and a comparable execution environment.
