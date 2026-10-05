# Validation and random state

The package uses one small set of contracts for estimator inputs and numeric hyperparameters.

## Feature matrices

Feature input `X` is normalized to a two-dimensional `float64` array.

The shared contract requires:

- at least one sample;
- at least one feature;
- numeric values;
- finite values;
- the fitted feature count during inference.

All maintained estimators use this contract either directly or through `BaseEstimator`.

## Targets

Targets `y` must be one-dimensional and match the number of feature rows.

Classification labels may remain strings or other non-numeric values. Numeric targets and numeric class labels additionally must contain only finite values.

This means NaN and infinity are rejected consistently before estimator-specific fitting logic runs.

## Scalar hyperparameters

Numeric constructor parameters use shared validators:

- positive real;
- non-negative real;
- positive integer.

Python booleans are rejected explicitly even though `bool` is a subclass of `int`. NaN and infinity are rejected for real-valued hyperparameters.

This keeps `tol=True`, `max_iter=True`, and similar accidental values from being silently interpreted as numbers.

## Random state

The package accepts three forms:

### Integer seed

An integer seed creates a **new generator on each resolution**.

```python
first = KMeans(random_state=42).fit(X)
second = KMeans(random_state=42).fit(X)
```

The two fits restart the same random sequence and are reproducible.

### Existing Generator

A supplied `numpy.random.Generator` is reused by identity.

```python
rng = np.random.default_rng(42)

first = KMeans(random_state=rng).fit(X)
second = KMeans(random_state=rng).fit(X)
```

The second fit continues from the generator state left by the first. This is intentional and is useful when one caller owns a longer random stream.

### None

`None` creates a fresh generator and does not promise reproducibility.

## Why this distinction matters

Treating integer seeds and generator objects identically would hide state ownership. The explicit contract makes it clear whether an estimator should restart a sequence or consume an existing one.
