# Algorithms

The maintained implementations are organized by statistical task rather than by notebook history.

## Supervised learning

### Regression

- [Linear regression](linear-regression.md) — ordinary least squares solved with `numpy.linalg.lstsq`.

### Classification

- [Logistic regression](logistic-regression.md) — binary logistic loss with Newton updates.
- [k-nearest neighbours](k-nearest-neighbours.md) — deterministic brute-force neighbour voting.
- [Gaussian Naive Bayes](gaussian-naive-bayes.md) — class-conditional Gaussian densities evaluated in log space.
- [Linear SVM](linear-svm.md) — binary squared-hinge soft-margin classifier.

## Unsupervised learning

- [k-means](k-means.md) — Lloyd iterations with k-means++ or random initialization.
- [PCA](pca.md) — centered singular value decomposition.

## Reading the notes

Each algorithm page describes four layers:

1. **model** — the mathematical object being estimated;
2. **numerical method** — how the implementation computes it;
3. **fitted attributes** — what is learned and exposed;
4. **scope** — choices intentionally excluded from the current implementation.

The implementation source remains intentionally small enough to read alongside these notes.
