from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from Packaging.common import version as packaging_version


REPO_ROOT = Path(__file__).resolve().parents[2]


def _package_version_metadata() -> tuple[str, tuple[int, ...]]:
    tree = ast.parse((REPO_ROOT / "tldw_chatbook" / "__init__.py").read_text())
    assignments: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "__version__",
                "VERSION_TUPLE",
            }:
                assignments[target.id] = ast.literal_eval(node.value)

    package_version_tuple = assignments["VERSION_TUPLE"]
    assert isinstance(package_version_tuple, tuple)

    return (str(assignments["__version__"]), package_version_tuple)


def test_release_version_metadata_stays_in_lockstep() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    project_version = project["version"]
    package_version, package_version_tuple = _package_version_metadata()

    assert package_version == project_version
    assert packaging_version.VERSION == project_version
    assert packaging_version.VERSION_TUPLE == package_version_tuple


def test_pypi_release_scripts_match_packaged_entry_points() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        scripts = tomllib.load(stream)["project"]["scripts"]

    assert scripts == {
        "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
        "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
    }
