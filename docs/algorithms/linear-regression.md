# Linear regression

## Model

For a feature matrix \(X \in \mathbb{R}^{n \times p}\) and target vector
\(y \in \mathbb{R}^{n}\), ordinary least squares estimates coefficients by
minimizing

\[
\min_{\beta_0,\beta}
\lVert y - \beta_0\mathbf{1} - X\beta \rVert_2^2.
\]

When `fit_intercept=True`, the implementation augments the design matrix with
a column of ones. If the augmented matrix is denoted by \(A\), the numerical
problem becomes

\[
\min_{\theta} \lVert A\theta-y\rVert_2^2.
\]

## Numerical method

The implementation uses `numpy.linalg.lstsq` rather than explicitly computing

\[
(A^\top A)^{-1}A^\top y.
\]

This avoids explicitly inverting the normal equations and remains well-defined
for rank-deficient designs, where the solver returns a minimum-norm
least-squares solution.

## Fitted attributes

- `coef_`: slope coefficients.
- `intercept_`: fitted intercept, or exactly zero when `fit_intercept=False`.
- `rank_`: numerical rank of the design matrix.
- `singular_values_`: singular values reported by the least-squares solver.
- `residual_sum_squares_`: residual sum of squares.
- `n_features_in_`: number of fitted input features.

## Example

```python
import numpy as np

from ml_algorithms import LinearRegression

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])

model = LinearRegression().fit(X, y)

print(model.intercept_)       # 1.0
print(model.coef_)            # [2.0]
print(model.predict([[4.0]])) # [9.0]
```

## Design note

The original repository demonstrated linear regression with gradient descent.
Gradient descent remains useful as an optimization topic, but ordinary least
squares has a direct least-squares formulation. The maintained estimator keeps
the statistical model separate from iterative optimization.
