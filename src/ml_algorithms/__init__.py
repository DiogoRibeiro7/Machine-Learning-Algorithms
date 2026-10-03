"""Machine-learning algorithms implemented from first principles.

The package intentionally keeps runtime dependencies minimal and focuses on
transparent NumPy implementations whose mathematical behaviour is easy to
inspect and test.
"""

from ml_algorithms._base import BaseEstimator, PredictorMixin, TransformerMixin
from ml_algorithms.cluster import KMeans
from ml_algorithms.exceptions import NotFittedError
from ml_algorithms.linear_model import LinearRegression, LogisticRegression
from ml_algorithms.naive_bayes import GaussianNB
from ml_algorithms.decomposition import PCA
from ml_algorithms.neighbors import KNeighborsClassifier

__all__ = [
    "BaseEstimator",
    "GaussianNB",
    "KMeans",
    "KNeighborsClassifier",
    "LinearRegression",
    "LogisticRegression",
    "NotFittedError",
    "PCA",
    "PredictorMixin",
    "TransformerMixin",
]
