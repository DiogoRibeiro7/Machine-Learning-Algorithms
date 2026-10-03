"""Machine-learning algorithms implemented from first principles.

The package intentionally keeps runtime dependencies minimal and focuses on
transparent NumPy implementations whose mathematical behaviour is easy to
inspect and test.
"""

from ml_algorithms._base import BaseEstimator, PredictorMixin, TransformerMixin
from ml_algorithms.exceptions import NotFittedError

__all__ = [
    "BaseEstimator",
    "NotFittedError",
    "PredictorMixin",
    "TransformerMixin",
]
