# Quality checks

The repository uses a deliberately small quality stack.

## Local checks

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy src tests
poetry run pytest
poetry run mkdocs build --strict
```

## Ruff

Ruff handles linting and formatting for Python 3.12 syntax.

## mypy

mypy runs in strict mode against both `src/` and `tests/`. Public code should keep types precise enough that tests do not need broad ignores.

## pytest

Tests cover ordinary behavior, fitted-state errors, deterministic behavior, and numerical edge cases. The test matrix runs against Python 3.12, 3.13, and 3.14.

## Documentation

MkDocs builds with `strict: true`. Broken links, invalid navigation, or documentation warnings should fail CI rather than silently reaching GitHub Pages.
