# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-07

### Added

- Modern Python package under `src/ml_algorithms/`.
- Shared estimator, validation, random-state, optimization, linear-algebra, and metrics infrastructure.
- Maintained implementations of linear regression, logistic regression, k-nearest neighbours, Gaussian Naive Bayes, k-means, PCA, and linear SVM.
- Gradient descent and AdaGrad numerical optimization utilities.
- Property-based numerical tests and project-wide coverage reporting.
- Deterministic benchmark suite for all maintained estimators.
- MkDocs Material documentation with mathematical notes, API reference, examples, and development guides.
- Semantic-versioning release checks that align tags, package versions, and changelog entries.

### Changed

- Repository development moved to `main`.
- Historical notebooks were separated from maintained package examples.
- Supported Python versions are 3.12, 3.13, and 3.14.

### Quality baseline

- Project-wide statement coverage baseline: 93%.
- Enforced minimum statement coverage: 92%.
- Deterministic benchmark suite records fit time, inference/transform time, and numerical-quality metrics across sample-count and feature-count sweeps.
