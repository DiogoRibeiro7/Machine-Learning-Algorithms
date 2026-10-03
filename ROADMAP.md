# Roadmap

## Goal

Modernize this repository into a focused, testable and documented collection of machine-learning algorithms implemented from first principles.

The repository should prioritize mathematical clarity, NumPy-based implementations, reproducible examples, and explicit links between theory and code.

## Phase 1 — Repository foundation

- [ ] Move the default development workflow from `master` to `main`
- [x] Introduce a Python package under `src/`
- [x] Add `pyproject.toml` with Python 3.12 support
- [x] Add Ruff, mypy, pytest and pre-commit
- [x] Add GitHub Actions for linting, typing and tests
- [ ] Add contribution and development documentation
- [x] Define repository scope and remove or relocate unrelated notebooks

## Phase 2 — Core algorithms

Implement small, readable estimators with a consistent API:

- [x] Linear regression
- [x] Logistic regression
- [ ] k-nearest neighbours
- [ ] Gaussian Naive Bayes
- [ ] k-means
- [ ] PCA
- [ ] linear SVM

Each implementation should include:

- mathematical formulation;
- typed Python implementation;
- unit tests;
- numerical edge-case tests;
- comparison against a trusted reference implementation where appropriate;
- one compact example notebook.

## Phase 3 — Numerical foundations

- [ ] Gradient descent utilities
- [ ] Adaptive optimization, including AdaGrad
- [ ] Stable linear algebra helpers
- [ ] Input validation and deterministic random-state handling
- [ ] Common metrics

## Phase 4 — Documentation

- [ ] MkDocs Material site
- [ ] Mathematical notes for each algorithm
- [ ] API reference
- [ ] Reproducible examples
- [ ] Design notes explaining implementation choices and numerical trade-offs
- [ ] GitHub Pages deployment

## Phase 5 — Quality and releases

- [ ] Property-based and regression tests where useful
- [ ] Coverage reporting
- [ ] Benchmark suite
- [ ] Changelog and semantic versioning
- [ ] First modern release

## Out of scope

The repository should not become a generic collection of unrelated notebooks. Time-series forecasting, deep learning, and domain-specific analyses should live in dedicated repositories unless they directly support the educational purpose of this project.
