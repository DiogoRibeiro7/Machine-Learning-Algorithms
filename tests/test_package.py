"""Smoke tests for the package foundation."""

from __future__ import annotations

import ml_algorithms


def test_package_imports() -> None:
    """The top-level package should import successfully."""
    assert ml_algorithms.__all__ == []
