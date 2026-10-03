"""Shared type aliases for estimator implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

type FeatureMatrix = NDArray[np.float64]
type TargetVector = NDArray[Any]
type PredictionArray = NDArray[Any]
type RandomState = int | np.random.Generator | None

__all__ = [
    "FeatureMatrix",
    "PredictionArray",
    "RandomState",
    "TargetVector",
]
