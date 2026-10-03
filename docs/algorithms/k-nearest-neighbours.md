# k-nearest neighbours

## Method

For a query point (x), k-nearest neighbours finds the (k) training samples
with smallest Euclidean distance

[
d(x,x_i)=sqrt{sum_j (x_j-x_{ij})^2}.
]

The predicted class is the class receiving the largest number of votes among
those neighbours.

## Deterministic ties

Two kinds of ties are handled explicitly:

1. If training samples have exactly equal distance to the query, neighbour
   selection uses a stable sort, so their original training order is preserved.
2. If two or more classes receive the same number of votes, the first class in
   sorted `classes_` order is selected.

This makes repeated predictions deterministic for fixed training data.

## Probabilities

`predict_proba` returns empirical neighbour vote frequencies. For example, if
three neighbours contain two samples from class A and one from class B, the
estimated probabilities are (2/3) and (1/3).

## Complexity

This implementation deliberately uses brute-force neighbour search:

- fit: (O(1)) apart from validation and storing the data;
- prediction for (m) queries: (O(mnp)) distance work for (n) training
  samples and (p) features, plus sorting.

That is appropriate for an educational implementation. Spatial indexes and
approximate-neighbour methods are separate algorithmic topics.

## Example

```python
import numpy as np

from ml_algorithms import KNeighborsClassifier

X = np.array([[0.0], [1.0], [4.0], [5.0]])
y = np.array([0, 0, 1, 1])

model = KNeighborsClassifier(n_neighbors=3).fit(X, y)

print(model.predict([[0.2], [4.8]]))
print(model.predict_proba([[0.2], [4.8]]))
```
