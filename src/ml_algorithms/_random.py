"""Random-number generator helpers."""

from __future__ import annotations

import numpy as np

from ml_algorithms.types import RandomState


def resolve_random_state(random_state: RandomState) -> np.random.Generator:
    """Return a NumPy generator for a package-level random-state value."""
    if random_state is None:
        return np.random.default_rng()

    if isinstance(random_state, np.random.Generator):
        return random_state

    if isinstance(random_state, bool):
        raise TypeError("random_state must not be a boolean.")

    if isinstance(random_state, int):
        return np.random.default_rng(random_state)

    raise TypeError(
        "random_state must be None, an integer seed, or numpy.random.Generator."
    )


__all__ = ["resolve_random_state"]
