# Gradient descent

Gradient descent is the first reusable optimization primitive in the numerical-foundations layer.

For a differentiable objective (f(	heta)), the update is

[
	heta_{t+1}
=
	heta_t-eta
abla f(	heta_t),
]

where (eta>0) is a fixed learning rate.

## Design

The implementation is deliberately small:

- full-batch and deterministic;
- one-dimensional parameter vectors;
- constant learning rate;
- Euclidean gradient-norm convergence criterion;
- explicit finite-value and shape validation;
- immutable diagnostics for every update.

It is not intended to become a general optimization framework. Its purpose is to make first-order optimization reusable in examples and future estimator implementations without hiding the update rule.

## Convergence

The optimizer stops when

[
leftlVert 
abla f(	heta_t)ightVert_2 leq arepsilon,
]

or after `max_iter` updates.

A result records:

- final parameters;
- final objective value;
- final gradient norm;
- whether the tolerance was reached;
- number of updates;
- an immutable history of objective, gradient norm, and step norm.

## Example

```python
import numpy as np

from ml_algorithms.optimization import gradient_descent

def objective(theta: np.ndarray) -> float:
    return float((theta[0] - 3.0) ** 2)

def gradient(theta: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (theta[0] - 3.0)])

result = gradient_descent(
    objective,
    gradient,
    [0.0],
    learning_rate=0.25,
)

print(result.parameters)
print(result.converged)
```

## Numerical limitations

A fixed learning rate can converge slowly on badly scaled objectives and can diverge if it is too large. The function intentionally does not add line search, momentum, stochastic batches, or adaptive scaling.

Adaptive scaling is introduced separately by the AdaGrad roadmap item.
