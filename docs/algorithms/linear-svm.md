# Linear support-vector machine

## Objective

The maintained estimator is a binary linear soft-margin SVM with squared hinge
loss. For labels encoded internally as (y_iin{-1,+1}),

[
min_{w,b}
rac{1}{2}lVert wVert_2^2
+
Crac{1}{n}sum_{i=1}^{n}
max(0, 1-y_i(w^	op x_i+b))^2.
]

The intercept (b) is not regularized.

This is the L2-loss linear SVM objective. It differs from the original notebook,
which used the ordinary hinge loss with an epoch-dependent regularization term.

## Optimization

The squared hinge loss is piecewise quadratic. On each iteration the
implementation identifies samples inside the margin, builds the gradient and
Hessian for that active set, solves the Newton system with
`numpy.linalg.lstsq`, and uses a short backtracking line search.

This gives deterministic full-batch updates and avoids introducing another
optimization dependency.

## Decision function

[
f(x)=w^	op x+b.
]

`decision_function` returns (f(x)). Predictions use the sign of that score,
then map (-1,+1) back to the two original class labels stored in
`classes_`.

## Fitted attributes

- `coef_`: separating hyperplane coefficients.
- `intercept_`: fitted intercept.
- `classes_`: the two original labels in sorted order.
- `n_iter_`: number of Newton iterations.
- `converged_`: whether the largest parameter update fell below `tol`.
- `loss_`: final regularized squared-hinge objective.
- `n_features_in_`: fitted feature count.

## Numerical scope

This implementation is deliberately limited to:

- binary classification;
- a linear decision boundary;
- dense NumPy arrays;
- squared hinge loss.

Kernel SVMs, multiclass decompositions, and large-scale coordinate-descent
solvers are separate topics.

## Example

```python
import numpy as np

from ml_algorithms import LinearSVM

X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LinearSVM(C=10.0).fit(X, y)

print(model.coef_)
print(model.intercept_)
print(model.predict([[-2.5], [2.5]]))
```
