"""Declared encoding detector versions must remain compatible with Requests."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize("manifest", ["pyproject.toml", "requirements.txt"])
def test_chardet_requirement_excludes_requests_incompatible_major(manifest):
    root = Path(__file__).resolve().parents[2]
    source = (root / manifest).read_text()
    entries = (
        tomllib.loads(source)["project"]["dependencies"]
        if manifest == "pyproject.toml"
        else [line for line in source.splitlines() if line.startswith("chardet")]
    )
    requirement = next(
        Requirement(entry) for entry in entries if entry.startswith("chardet")
    )
    assert "5.2.0" in requirement.specifier
    assert "6.0.0" not in requirement.specifier
    assert "3.0.1" not in requirement.specifier
