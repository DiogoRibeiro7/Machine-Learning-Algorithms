# Contributing

Contributions should preserve the repository's main goal: transparent, mathematically grounded implementations that remain easy to read and test.

## Development setup

```bash
git clone https://github.com/DiogoRibeiro7/Machine-Learning-Algorithms.git
cd Machine-Learning-Algorithms
poetry install
poetry run pre-commit install
```

The project targets Python 3.12+.

## Workflow

Use a focused branch and open a pull request against the default development branch. Keep changes scoped: algorithm work, numerical infrastructure, tests, documentation, or repository maintenance should be reviewable independently.

Before opening or updating a pull request, run:

```bash
poetry run ruff check src tests benchmarks scripts
poetry run ruff format --check src tests benchmarks scripts
poetry run mypy
poetry run pytest
poetry run mkdocs build --strict
```

## Algorithm contributions

A maintained estimator should include:

- a clear mathematical objective or probabilistic model;
- a typed NumPy implementation;
- explicit input and fitted-state validation;
- ordinary unit tests and numerical edge-case tests;
- deterministic behavior where the algorithm permits it;
- mathematical documentation under `docs/algorithms/`;
- a compact maintained example under `notebooks/examples/`.

Avoid introducing a dependency when a short NumPy implementation keeps the method clearer.

## Numerical behavior

Do not hide numerical trade-offs. If an implementation uses regularization, variance smoothing, a convergence tolerance, a decomposition choice, or a deterministic tie rule, document it and test it.

## Legacy notebooks

Historical notebooks under `notebooks/legacy/` are retained for provenance. New functionality should not be implemented there.

## Releases

The release process is documented in [docs/development/releases.md](docs/development/releases.md).

Release preparation must keep `pyproject.toml`, the Git tag, and `CHANGELOG.md` aligned. Before tagging a version, validate the metadata locally:

```bash
poetry run python scripts/check_release.py --tag vX.Y.Z
```

Breaking API or numerical-contract changes must be recorded explicitly in the changelog and release notes.

## Pull requests

Do not merge a pull request until CI passes. Prefer small pull requests with a single purpose and a description that explains the mathematical or engineering choice being made.
