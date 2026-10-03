"""Shared type aliases for estimator implementations."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FeatureMatrix: TypeAlias = NDArray[np.float64]
TargetVector: TypeAlias = NDArray[Any]
PredictionArray: TypeAlias = NDArray[Any]
RandomState: TypeAlias = int | np.random.Generator | None

__all__ = [
    "FeatureMatrix",
    "PredictionArray",
    "RandomState",
    "TargetVector",
]
