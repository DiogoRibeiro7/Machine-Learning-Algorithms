# k-means

## Objective

k-means partitions the observations into (K) clusters by minimizing the
within-cluster sum of squared Euclidean distances,

[
sum_{i=1}^{n}
leftlVert x_i-mu_{z_i}ightVert_2^2,
]

where (mu_k) is the centroid of cluster (k).

The fitted `inertia_` is exactly this objective evaluated at the final
assignments.

## Iteration

Each iteration alternates between:

1. assigning every sample to its nearest centroid;
2. replacing each centroid with the arithmetic mean of its assigned samples.

Convergence is declared when the maximum Euclidean displacement of any centroid
is at most `tol`.

## Initialization

Two strategies are supported:

- `k-means++`: pick the first centroid randomly, then sample subsequent
  centroids with probability proportional to squared distance from the nearest
  selected centroid;
- `random`: choose distinct training observations uniformly without
  replacement.

An integer `random_state` makes initialization reproducible.

## Empty clusters

If a centroid receives no observations, the implementation repositions it to a
training sample with a large current reconstruction error. Ties are resolved
stably, making this recovery deterministic for a fixed initialization.

## Fitted attributes

- `cluster_centers_`: final centroids.
- `labels_`: training-set cluster assignments.
- `inertia_`: within-cluster sum of squares.
- `n_iter_`: number of centroid-update iterations.
- `converged_`: whether the displacement criterion was met.
- `n_features_in_`: fitted feature count.

## Example

```python
import numpy as np

from ml_algorithms import KMeans

X = np.array([[0.0], [0.2], [0.4], [9.6], [9.8], [10.0]])

model = KMeans(n_clusters=2, random_state=42).fit(X)

print(model.cluster_centers_)
print(model.labels_)
print(model.inertia_)
```

## Historical note

The legacy notebook initialized centroids from the first (K) rows and used a
relative coordinate-wise movement calculation. The maintained implementation
uses explicit random initialization controls and an absolute Euclidean
displacement criterion, avoiding division by zero when old centroid coordinates
are zero.
