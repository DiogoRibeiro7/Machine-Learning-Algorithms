# Stable linear algebra

The shared linear-algebra layer contains only operations that recur across multiple estimators and benefit from one consistent numerical contract.

## Least squares

`least_squares(A, b)` solves

[
min_x lVert Ax-bVert_2^2
]

with `numpy.linalg.lstsq`, which uses an SVD-based least-squares solver rather than explicitly forming or inverting (A^	op A).

The returned diagnostics include:

- minimum-norm solution;
- residual sum of squares computed explicitly;
- numerical rank;
- singular values;
- a 2-norm condition-number diagnostic.

Rank-deficient systems report an infinite condition number. This is intentional: the returned minimum-norm solution may still be perfectly useful, but the system is not uniquely determined by the columns of (A).

## Pairwise distances

`pairwise_squared_distances` computes squared Euclidean distances directly from row-wise differences:

[
d^2(x_i,y_j)
=
sum_k (x_{ik}-y_{jk})^2.
]

`pairwise_distances` applies the square root to that result.

The direct-difference formulation is used rather than the algebraic expansion
(lVert xVert^2+lVert yVert^2-2x^	op y), avoiding small negative values from cancellation before the square root.

These helpers are shared by k-nearest neighbours and k-means.

## Sign canonicalization

Singular vectors and eigenvectors are only defined up to sign. If (v) is valid, so is (-v).

`canonicalize_row_signs` makes this ambiguity deterministic by forcing the loading with largest absolute magnitude in each row to be non-negative. PCA uses this helper so repeated fits return a stable orientation without changing the represented subspace.

## Scope

The module does not wrap arbitrary NumPy linear algebra. Decompositions, matrix products, and algorithm-specific constructions remain visible in their estimators unless there is concrete duplication or a numerical contract worth centralizing.
