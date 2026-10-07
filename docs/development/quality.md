# Quality checks

The repository uses a deliberately small quality stack.

## Local checks

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy src tests
poetry run pytest
poetry run pytest --cov=ml_algorithms --cov-report=term-missing --cov-report=xml:coverage.xml
poetry run mkdocs build --strict
```

## Ruff

Ruff handles linting and formatting for Python 3.12 syntax.

## mypy

mypy runs in strict mode against both `src/` and `tests/`. Public code should keep types precise enough that tests do not need broad ignores.

## pytest

Tests cover ordinary behavior, fitted-state errors, deterministic behavior, and numerical edge cases. The test matrix runs against Python 3.12, 3.13, and 3.14.

## Hypothesis

Hypothesis adds bounded property-based tests for invariants such as probability normalization, PCA reconstruction, non-negative k-means inertia, seeded determinism, prediction shapes, refit state replacement, and validation behavior. Strategies use small finite arrays and explicit example limits so CI remains fast and reproducible.

## Coverage

The Python 3.12 coverage job runs the full test suite with `pytest-cov`, prints missing lines in the terminal, and writes `coverage.xml`. CI uploads that XML file as the `coverage-xml` artifact so an external reporting service can consume it later without changing the test command.

The minimum project-wide threshold is set from the measured baseline of the current suite rather than from an arbitrary target.

## Documentation

MkDocs builds with `strict: true`. Broken links, invalid navigation, or documentation warnings should fail CI rather than silently reaching GitHub Pages.
