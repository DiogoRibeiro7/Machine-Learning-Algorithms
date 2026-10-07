# Releases and versioning

The project uses Semantic Versioning and keeps one authoritative package version in `pyproject.toml`.

## Version policy

Versions use `MAJOR.MINOR.PATCH`.

- **MAJOR**: incompatible public API or numerical-contract changes after the project reaches a stable 1.x release.
- **MINOR**: backward-compatible algorithms, metrics, capabilities, or meaningful numerical extensions.
- **PATCH**: backward-compatible bug fixes, documentation corrections, test improvements, and numerical fixes that restore documented behavior.

While the project remains in the 0.x series, compatibility is still taken seriously. A breaking API or numerical-contract change must be called out explicitly in the changelog and release notes, even when it is represented by a minor-version increment under Semantic Versioning's pre-1.0 rules.

## What counts as a breaking numerical change

Breaking changes are not limited to Python signatures. They include changes such as:

- altering the mathematical objective of an estimator;
- changing convergence semantics or default tolerances in a way that materially changes results;
- changing deterministic tie-breaking or seeded behavior;
- changing label ordering, output shapes, or fitted-attribute meaning;
- changing the documented treatment of degenerate numerical cases.

Such changes must appear under a **Changed** section in `CHANGELOG.md` with an explicit compatibility note.

## Changelog

Development changes accumulate under `## [Unreleased]`.

For a release, move the relevant entries into a dated heading:

```text
## [0.1.0] - 2026-10-07
```

Release notes should be based on that changelog section so the repository history and GitHub release description remain aligned.

## Release checklist

Before creating a tag:

1. Confirm CI, documentation, coverage, and benchmark smoke tests are green.
2. Choose the Semantic Versioning increment.
3. Update `[project].version` in `pyproject.toml`.
4. Move the relevant `Unreleased` entries into a dated version section.
5. Leave a fresh `## [Unreleased]` section at the top of the changelog.
6. Run the full local quality suite.
7. Validate metadata locally:

   ```bash
   poetry run python scripts/check_release.py --tag vX.Y.Z
   ```

8. Build the package locally if desired:

   ```bash
   poetry build
   ```

9. Commit the version and changelog changes.
10. Create the annotated release tag `vX.Y.Z`.
11. Push the tag and confirm the **Release check** workflow succeeds.
12. Create the GitHub release using the matching changelog section as the release notes.

## Release workflow

A tag matching `v*.*.*` triggers `.github/workflows/release-check.yml`.

The workflow verifies that:

- the tag is valid `vMAJOR.MINOR.PATCH`;
- the tag version equals `[project].version`;
- `CHANGELOG.md` contains a dated heading for that exact version;
- the Python package can be built successfully.

The built wheel and source distribution are uploaded as workflow artifacts for inspection.

**The workflow does not publish the package to PyPI or any other registry.** Publication must be added explicitly in a later change if the project needs it.
