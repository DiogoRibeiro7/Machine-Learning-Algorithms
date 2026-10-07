"""Tests for release metadata validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.check_release import validate_release


def _write_release_files(
    tmp_path: Path,
    *,
    version: str = "1.2.3",
    changelog_version: str = "1.2.3",
) -> tuple[Path, Path]:
    """Create minimal release metadata files for one test."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        f"# Changelog\n\n## [{changelog_version}] - 2026-10-07\n\n- Release.\n",
        encoding="utf-8",
    )
    return pyproject, changelog


def test_release_metadata_must_align(tmp_path: Path) -> None:
    pyproject, changelog = _write_release_files(tmp_path)

    version = validate_release(
        "v1.2.3",
        pyproject_path=pyproject,
        changelog_path=changelog,
    )

    assert version == "1.2.3"


def test_release_tag_must_match_package_version(tmp_path: Path) -> None:
    pyproject, changelog = _write_release_files(tmp_path, version="1.2.4")

    with pytest.raises(ValueError, match="does not match"):
        validate_release(
            "v1.2.3",
            pyproject_path=pyproject,
            changelog_path=changelog,
        )


def test_release_requires_dated_changelog_heading(tmp_path: Path) -> None:
    pyproject, changelog = _write_release_files(
        tmp_path,
        changelog_version="1.2.2",
    )

    with pytest.raises(ValueError, match="dated release heading"):
        validate_release(
            "v1.2.3",
            pyproject_path=pyproject,
            changelog_path=changelog,
        )


def test_release_tag_must_use_v_semver_form(tmp_path: Path) -> None:
    pyproject, changelog = _write_release_files(tmp_path)

    with pytest.raises(ValueError, match=r"vMAJOR\.MINOR\.PATCH"):
        validate_release(
            "1.2.3",
            pyproject_path=pyproject,
            changelog_path=changelog,
        )
