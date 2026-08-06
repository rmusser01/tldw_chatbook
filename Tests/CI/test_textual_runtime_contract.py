from collections.abc import Iterable
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    assert Version("7.999.999") not in requirement.specifier
    assert Version("9.0.0") not in requirement.specifier
    exact_pins = [spec for spec in requirement.specifier if spec.operator == "=="]
    if exact_pins:
        # Exact pin (e.g. ==8.2.8 per TASK-1353): must stay within the 8.x line.
        assert all(Version(spec.version).major == 8 for spec in exact_pins)
    else:
        assert Version("8.0.0") in requirement.specifier
        assert Version("8.999.999") in requirement.specifier


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
