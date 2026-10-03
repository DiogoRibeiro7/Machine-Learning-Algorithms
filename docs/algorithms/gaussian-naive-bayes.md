# Gaussian Naive Bayes

## Model

Gaussian Naive Bayes assumes features are conditionally independent given the
class and that each feature follows a class-specific Gaussian distribution.

For class (c) and feature (j),

[
X_j mid Y=c sim mathcal{N}(mu_{cj},sigma_{cj}^2).
]

The classifier chooses the class maximizing

[
log P(Y=c)
-rac{1}{2}sum_j
left[
log(2pisigma_{cj}^2)
+
rac{(x_j-mu_{cj})^2}{sigma_{cj}^2}
ight].
]

## Numerical stability

The implementation works in log space rather than multiplying densities
directly. Posterior probabilities are normalized with a stable log-sum-exp
calculation.

A small variance floor is added to every class-feature variance:

[
epsilon =
	ext{var_smoothing}
	imes
max(max_j operatorname{Var}(X_j), epsilon_{	ext{machine}}).
]

The machine-epsilon fallback means even a dataset whose features are globally
constant retains strictly positive variances.

## Fitted attributes

- `classes_`: observed classes in sorted order.
- `class_count_`: number of training samples per class.
- `class_prior_`: empirical class probabilities.
- `theta_`: class-feature means.
- `var_`: smoothed class-feature variances.
- `epsilon_`: variance-smoothing term.
- `n_features_in_`: fitted feature count.

## Example

```python
import numpy as np

from ml_algorithms import GaussianNB

X = np.array([[0.0], [0.2], [3.0], [3.2]])
y = np.array(["cold", "cold", "hot", "hot"])

model = GaussianNB().fit(X, y)

print(model.predict([[0.1], [3.1]]))
print(model.predict_proba([[0.1], [3.1]]))
```

## Historical note

The original repository's Naive Bayes notebook was a word-count spam
classifier. That is a discrete text model, not Gaussian Naive Bayes. It remains
in the legacy notebooks for provenance rather than being treated as the source
of this maintained implementation.
