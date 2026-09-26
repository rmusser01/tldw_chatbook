"""The Character Creator built-in skill ships in the wheel (TASK-32954).

Mirrors ``test_canvas_guide_manifest.py``: the distribution checker
(``Packaging/check_manifest.py``) is the gate a built wheel/sdist must pass,
so these tests pin that it REQUIRES the skill file and does not reject it as
development Markdown. A separate check proves the setuptools package-data
configuration actually selects the file.
"""

import fnmatch
import tomllib
from pathlib import Path

import pytest

from Packaging import check_manifest

SKILL = "tldw_chatbook/assets/skills/character-creator/SKILL.md"
REPO = Path(__file__).resolve().parents[2]


def _errors(kind, members):
    required = getattr(check_manifest, f"REQUIRED_{kind.upper()}_PATHS")
    return check_manifest._validate_content(
        kind, members, required_paths=required, required_globs=set()
    )


def test_builtin_skill_accepted_in_wheel():
    assert not _errors("wheel", check_manifest.REQUIRED_WHEEL_PATHS | {SKILL})


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_builtin_skill_is_required(kind):
    members = getattr(check_manifest, f"REQUIRED_{kind.upper()}_PATHS") - {SKILL}
    assert f"{kind}: missing required path: {SKILL}" in _errors(kind, members)


def test_package_data_selects_the_skill_file():
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["tldw_chatbook"]
    relative = SKILL.removeprefix("tldw_chatbook/")
    assert (REPO / SKILL).is_file()
    assert any(fnmatch.fnmatchcase(relative, p) for p in patterns)
