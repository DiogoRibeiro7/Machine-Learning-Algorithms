# Logistic regression

## Model

For binary targets, logistic regression models the positive-class probability as

[
P(Y=1mid x)=sigma(eta_0+x^	opeta),
]

where

[
sigma(z)=rac{1}{1+e^{-z}}.
]

The estimator minimizes the mean logistic loss, optionally with an L2 penalty on
the slope coefficients.

## Numerical method

The implementation uses Newton updates. At each iteration it computes the
gradient and Hessian of the penalized logistic loss, solves the Newton system
with `numpy.linalg.lstsq`, and applies a short backtracking line search.

The sigmoid is evaluated piecewise to avoid overflow for large positive or
negative logits. The objective uses `numpy.logaddexp` for numerical stability.

## Fitted attributes

- `coef_`: slope coefficients.
- `intercept_`: fitted intercept.
- `classes_`: the two observed labels, in sorted order.
- `n_iter_`: number of Newton iterations.
- `converged_`: whether the parameter update fell below tolerance.
- `loss_`: final penalized logistic loss.
- `n_features_in_`: number of fitted input features.

## Example

```python
import numpy as np

from ml_algorithms import LogisticRegression

X = np.array([[-2.0], [-1.0], [0.5], [1.0], [2.0]])
y = np.array([0, 0, 1, 1, 1])

model = LogisticRegression(l2=0.1).fit(X, y)

print(model.predict([[0.25]]))
print(model.predict_proba([[0.25]]))
```

## Design note

This implementation is binary only. Multiclass extensions would introduce a
different optimization problem and are intentionally deferred rather than hidden
behind extra abstraction.
