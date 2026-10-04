# Principal component analysis

## Model

PCA finds orthonormal directions that successively maximize the variance of
centered data. Given

[
X_c = X - mathbf{1}ar{x}^{	op},
]

the maintained implementation computes the compact singular value decomposition

[
X_c = USigma V^{	op}.
]

The rows of (V^{	op}) are the principal directions stored in
`components_`.

## Why SVD?

The legacy notebook explicitly formed the covariance matrix and called
`numpy.linalg.eig`. The maintained implementation applies SVD directly to the
centered data. This avoids squaring the condition number through
(X_c^{	op}X_c) and guarantees real orthonormal components for real input
data.

## Explained variance

If (sigma_k) is the (k)-th singular value and there are (n) samples,

[
lambda_k = rac{sigma_k^2}{n-1}.
]

These values are stored in `explained_variance_`. Their fraction of total
variance is stored in `explained_variance_ratio_`.

For data with exactly zero total variance, the explained-variance ratios are
defined as zeros rather than NaNs.

## Centering

`transform` always projects centered data:

[
Z = (X-ar{x})V_r.
]

This matters. The original notebook centered the data to estimate covariance,
but its projection method then multiplied the uncentered input by the component
matrix.

## Sign indeterminacy

If (v) is a principal direction, (-v) is an equally valid direction. PCA
components are therefore mathematically sign-indeterminate.

For reproducibility, this implementation chooses a canonical sign: the loading
with largest absolute magnitude in each component is forced to be non-negative.
This does not change the represented subspace or explained variance.

## Fitted attributes

- `mean_`: feature means used for centering.
- `components_`: retained principal directions.
- `explained_variance_`: variance explained by each retained component.
- `explained_variance_ratio_`: fraction of total variance explained.
- `singular_values_`: retained singular values.
- `n_components_`: resolved number of retained components.
- `n_samples_seen_`: number of fitting samples.

## Example

```python
import numpy as np

from ml_algorithms import PCA

X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
])

pca = PCA(n_components=1).fit(X)

Z = pca.transform(X)
X_approx = pca.inverse_transform(Z)

print(pca.components_)
print(pca.explained_variance_ratio_)
```
