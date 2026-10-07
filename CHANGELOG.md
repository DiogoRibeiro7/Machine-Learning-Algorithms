# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Modern Python package under `src/ml_algorithms/`.
- Shared estimator, validation, random-state, optimization, linear-algebra, and metrics infrastructure.
- Maintained implementations of linear regression, logistic regression, k-nearest neighbours, Gaussian Naive Bayes, k-means, PCA, and linear SVM.
- Gradient descent and AdaGrad numerical optimization utilities.
- Property-based numerical tests and project-wide coverage reporting.
- Deterministic benchmark suite for all maintained estimators.
- MkDocs Material documentation with mathematical notes, API reference, examples, and development guides.

### Changed

- Repository development moved to `main`.
- Historical notebooks were separated from maintained package examples.
- Supported Python versions are 3.12, 3.13, and 3.14.

### Notes

The first modern release has not been tagged yet. Issue #41 will convert this Unreleased section into the dated `0.1.0` release entry.
