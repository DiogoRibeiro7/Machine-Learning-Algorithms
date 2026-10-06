# Roadmap

## Goal

Modernize this repository into a focused, testable and documented collection of machine-learning algorithms implemented from first principles.

The repository should prioritize mathematical clarity, NumPy-based implementations, reproducible examples, and explicit links between theory and code.

## Phase 1 — Repository foundation

- [x] Move the default development workflow from `master` to `main`
- [x] Introduce a Python package under `src/`
- [x] Add `pyproject.toml` with Python 3.12 support
- [x] Add Ruff, mypy, pytest and pre-commit
- [x] Add GitHub Actions for linting, typing and tests
- [x] Add contribution and development documentation
- [x] Define repository scope and remove or relocate unrelated notebooks

## Phase 2 — Core algorithms

Implement small, readable estimators with a consistent API:

- [x] Linear regression
- [x] Logistic regression
- [x] k-nearest neighbours
- [x] Gaussian Naive Bayes
- [x] k-means
- [x] PCA
- [x] linear SVM

Each implementation should include:

- mathematical formulation;
- typed Python implementation;
- unit tests;
- numerical edge-case tests;
- comparison against a trusted reference implementation where appropriate;
- one compact example notebook.

## Phase 3 — Numerical foundations

- [x] Gradient descent utilities
- [x] Adaptive optimization, including AdaGrad
- [x] Stable linear algebra helpers
- [x] Input validation and deterministic random-state handling
- [x] Common metrics

## Phase 4 — Documentation

- [x] MkDocs Material site
- [x] Mathematical notes for each algorithm
- [x] API reference
- [x] Reproducible examples
- [x] Design notes explaining implementation choices and numerical trade-offs
- [x] GitHub Pages deployment

## Phase 5 — Quality and releases

- [ ] Property-based and regression tests where useful
- [ ] Coverage reporting
- [ ] Benchmark suite
- [ ] Changelog and semantic versioning
- [ ] First modern release

## Out of scope

The repository should not become a generic collection of unrelated notebooks. Time-series forecasting, deep learning, and domain-specific analyses should live in dedicated repositories unless they directly support the educational purpose of this project.
