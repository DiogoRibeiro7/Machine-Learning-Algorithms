# Notebooks

This directory separates maintained examples from historical material.

## Maintained examples

New examples belong in `examples/`. They should:

- import implementations from `ml_algorithms` rather than redefining them;
- be reproducible from a clean environment;
- use small, documented datasets;
- focus on one algorithm or concept;
- avoid becoming the primary implementation or test surface.

## Legacy notebooks

Historical notebooks are preserved under `legacy/` for provenance. They are not part
of the supported package API and are not expected to satisfy the current linting,
typing, testing, or dependency standards.

Some legacy notebooks are explicitly marked as out of scope because the modern
repository focuses on classical machine-learning algorithms implemented from first
principles.
