# Common metrics

The metrics module provides a small NumPy-only set of measures for examples, tests, and benchmarks.

## Regression

### Mean squared error

[
operatorname{MSE}
=
rac{1}{n}sum_{i=1}^{n}(y_i-hat y_i)^2.
]

### Mean absolute error

[
operatorname{MAE}
=
rac{1}{n}sum_{i=1}^{n}|y_i-hat y_i|.
]

### Coefficient of determination

[
R^2
=
1-
rac{sum_i(y_i-hat y_i)^2}
{sum_i(y_i-ar y)^2}.
]

For a constant target, the denominator is zero. This implementation defines the degenerate case explicitly:

- perfect prediction: (R^2=1);
- imperfect prediction: (R^2=0).

## Classification

### Accuracy

Accuracy is the fraction of labels predicted exactly correctly. Labels may be strings or other comparable values.

### Binary log loss

For binary targets (y_iin{0,1}) and positive-class probabilities (p_i),

[
-rac{1}{n}
sum_i
left[
y_ilog p_i
+
(1-y_i)log(1-p_i)
ight].
]

Probabilities are validated to lie in ([0,1]), then clipped by an explicit `epsilon` before logarithms are evaluated.

### Confusion matrix

`confusion_matrix` returns both the count matrix and the exact label order used for its rows and columns.

Rows are true labels; columns are predicted labels.

When `labels` is omitted, the order is the sorted unique union of observed true and predicted labels. An explicit `labels` argument can be supplied when another order is required.

## Validation

All metrics require one-dimensional, non-empty vectors with matching sample counts. Regression metrics require finite numeric inputs. Classification metrics preserve arbitrary labels where the mathematics does not require numeric encoding.
