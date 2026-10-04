# Getting started

## Requirements

- Python 3.12, 3.13, or 3.14
- NumPy 2.x

## Install for development

The repository uses Poetry for project metadata and dependency management.

```bash
git clone https://github.com/DiogoRibeiro7/Machine-Learning-Algorithms.git
cd Machine-Learning-Algorithms
poetry install
```

Enter the environment or prefix commands with `poetry run`.

```bash
poetry run pytest
poetry run ruff check src tests
poetry run mypy src tests
```

## First model

```python
import numpy as np

from ml_algorithms import LinearRegression

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])

model = LinearRegression().fit(X, y)

print(model.coef_)
print(model.intercept_)
print(model.predict([[4.0]]))
```

The estimator records the fitted feature count and rejects inference data with an incompatible number of columns.

## Build the documentation locally

```bash
poetry run mkdocs serve
```

For the same validation used in CI:

```bash
poetry run mkdocs build --strict
```

## Package conventions

All maintained estimators:

1. validate a two-dimensional feature matrix;
2. return `self` from `fit`;
3. expose `n_features_in_` after fitting;
4. raise `NotFittedError` when inference is attempted before fitting;
5. preserve original class labels where classification semantics require it.
