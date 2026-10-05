"""Random-number generator helpers."""

from __future__ import annotations

import numpy as np

from ml_algorithms.types import RandomState


def resolve_random_state(random_state: RandomState) -> np.random.Generator:
    """Resolve package random-state values to a NumPy Generator.

    An integer seed creates a new generator on every call, so repeated fits
    with the same integer restart the same random sequence. An existing
    numpy.random.Generator is returned unchanged, so repeated use advances
    that generator's state. None creates a fresh non-deterministic generator.
    """
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
