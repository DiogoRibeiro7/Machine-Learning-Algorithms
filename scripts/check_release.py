"""Validate release tag, package version, and changelog alignment."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


VERSION_PATTERN = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")
CHANGELOG_PATTERN_TEMPLATE = r"^## \[{version}\] - \d{{4}}-\d{{2}}-\d{{2}}$"


def package_version(pyproject_path: Path) -> str:
    """Read the project version from pyproject.toml."""
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no [project] table.")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml has no valid project version.")
    return version


def validate_release(
    tag: str,
    *,
    pyproject_path: Path = Path("pyproject.toml"),
    changelog_path: Path = Path("CHANGELOG.md"),
) -> str:
    """Validate one semantic-version release tag against repository metadata."""
    match = VERSION_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError("Release tag must use the form vMAJOR.MINOR.PATCH.")

    version = match.group("version")
    project_version = package_version(pyproject_path)
    if project_version != version:
        raise ValueError(
            f"Tag {tag} does not match pyproject.toml version {project_version}."
        )

    changelog = changelog_path.read_text(encoding="utf-8")
    heading_pattern = re.compile(
        CHANGELOG_PATTERN_TEMPLATE.format(version=re.escape(version)),
        flags=re.MULTILINE,
    )
    if heading_pattern.search(changelog) is None:
        raise ValueError(
            f"CHANGELOG.md has no dated release heading for version {version}."
        )

    return version


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate release metadata before tagging or publishing.",
    )
    parser.add_argument("--tag", required=True, help="Release tag, for example v0.1.0.")
    return parser.parse_args()


def main() -> int:
    """Validate release metadata and report the aligned version."""
    args = parse_args()
    version = validate_release(args.tag)
    print(f"Release metadata aligned for version {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
