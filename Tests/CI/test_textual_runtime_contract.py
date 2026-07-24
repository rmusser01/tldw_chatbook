from collections.abc import Iterable
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXTUAL_8_X = SpecifierSet(">=8.0.0,<9")


def _textual_requirement(entries: Iterable[str]) -> Requirement:
    for entry in entries:
        candidate = entry.split("#", 1)[0].strip()
        if not candidate:
            continue
        requirement = Requirement(candidate)
        if requirement.name.lower() == "textual":
            return requirement
    raise AssertionError("Textual requirement is missing")


def _assert_textual_8_only(requirement: Requirement) -> None:
    assert requirement.specifier == TEXTUAL_8_X
    assert Version("7.999.999") not in requirement.specifier
    assert Version("8.0.0") in requirement.specifier
    assert Version("8.999.999") in requirement.specifier
    assert Version("9.0.0") not in requirement.specifier


def test_pyproject_supports_only_textual_8_x() -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    requirement = _textual_requirement(pyproject["project"]["dependencies"])

    _assert_textual_8_only(requirement)


def test_development_requirements_support_only_textual_8_x() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    requirement = _textual_requirement(requirements.splitlines())

    _assert_textual_8_only(requirement)
