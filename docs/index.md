# Machine Learning Algorithms

<div class="hero" markdown>

<span class="hero-eyebrow">FROM FIRST PRINCIPLES</span>

## Mathematical models. Transparent NumPy. Tested implementations.

A compact collection of classical machine-learning algorithms implemented to keep the mathematics visible. The package favors explicit numerical methods, small APIs, deterministic behavior, and tests over framework machinery.

<div class="hero-actions">
<a class="md-button md-button--primary" href="getting-started/">Get started</a>
<a class="md-button" href="algorithms/">Explore the algorithms</a>
</div>

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### Seven maintained estimators

Linear and logistic regression, k-nearest neighbours, Gaussian Naive Bayes, k-means, PCA, and a linear SVM.
</div>

<div class="feature-card" markdown>
### Mathematics beside code

Each algorithm documents its objective, assumptions, numerical method, fitted quantities, and known limitations.
</div>

<div class="feature-card" markdown>
### Small engineering surface

NumPy is the only runtime dependency. Ruff, mypy, pytest, pre-commit, and GitHub Actions enforce the development contract.
</div>

</div>

## Project philosophy

This repository is not intended to compete with production libraries such as scikit-learn. Its purpose is narrower: make the path from mathematical formulation to working implementation easy to inspect.

The maintained code follows a few rules:

- estimators expose small, consistent `fit`, `predict`, and transformation interfaces;
- numerical choices are documented rather than hidden;
- deterministic behavior is preferred when algorithms contain ambiguity;
- edge cases are tested explicitly;
- notebooks demonstrate the package instead of containing the implementation.

## Current algorithm set

| Area | Estimator | Numerical core |
| --- | --- | --- |
| Regression | `LinearRegression` | Least squares via SVD-based solver |
| Classification | `LogisticRegression` | Newton updates + stable log-loss |
| Classification | `KNeighborsClassifier` | Brute-force Euclidean search |
| Classification | `GaussianNB` | Log Gaussian likelihoods |
| Clustering | `KMeans` | Lloyd iterations + k-means++ |
| Decomposition | `PCA` | Centered compact SVD |
| Classification | `LinearSVM` | Squared-hinge soft margin |

## Where to go next

- [Getting started](getting-started.md) for installation and a first estimator.
- [Algorithms](algorithms/index.md) for the mathematical notes.
- [Architecture](architecture.md) for repository design and estimator contracts.
- [API reference](api-reference.md) for the public package surface.
- [Examples](examples.md) for maintained notebooks.
