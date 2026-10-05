# AdaGrad

AdaGrad adapts the effective learning rate separately for each parameter by accumulating squared gradients.

For gradient component (g_{t,j}),

[
G_{t,j} = G_{t-1,j} + g_{t,j}^2,
]

and the update is

[
	heta_{t+1,j}
=
	heta_{t,j}
-
rac{eta}{sqrt{G_{t,j}}+arepsilon}
g_{t,j}.
]

The global learning-rate multiplier (eta) is therefore modulated by the history of each coordinate.

## Why adaptive scaling helps

A fixed learning rate must be small enough for the most strongly curved direction of an objective. On anisotropic problems, that same small rate can make progress painfully slow along flatter directions.

AdaGrad reduces this mismatch by scaling coordinates independently according to their accumulated gradient magnitudes.

The test suite includes a two-dimensional quadratic with a 100:1 curvature ratio and compares AdaGrad against ordinary fixed-rate gradient descent.

## API

The function mirrors `gradient_descent`:

- the same objective callable;
- the same gradient callable;
- the same one-dimensional parameter representation;
- the same gradient-norm convergence criterion;
- the same immutable per-iteration diagnostics.

In addition, `AdaGradResult` exposes `accumulated_squared_gradients`, making the final adaptive state inspectable.

## Example

```python
import numpy as np

from ml_algorithms.optimization import adagrad

matrix = np.array([[100.0, 0.0], [0.0, 1.0]])

def objective(theta: np.ndarray) -> float:
    return float(0.5 * theta @ matrix @ theta)

def gradient(theta: np.ndarray) -> np.ndarray:
    return matrix @ theta

result = adagrad(
    objective,
    gradient,
    [10.0, 10.0],
    learning_rate=2.0,
)

print(result.parameters)
print(result.accumulated_squared_gradients)
```

## Numerical limitations

AdaGrad's accumulator only grows. Consequently, effective learning rates decrease monotonically and may become very small during long optimization runs.

That behavior is intentional here: this implementation demonstrates AdaGrad itself rather than extending immediately to RMSProp, Adam, or accumulator decay.
