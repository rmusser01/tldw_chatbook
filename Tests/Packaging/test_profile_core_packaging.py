import tomllib
from pathlib import Path

from setuptools import find_packages


def test_root_distribution_discovers_profile_core_package():
    packages = {
        package
        for source_root in (".", "packages/tldw_profile_core/src")
        for package in find_packages(
            where=source_root, include=["tldw_chatbook*", "tldw_profile_core*"]
        )
    }
    assert "tldw_profile_core" in packages


def test_root_and_standalone_distributions_pin_profile_canonicalizer():
    root = Path(__file__).parents[2]
    root_project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    standalone = tomllib.loads(
        (root / "packages/tldw_profile_core/pyproject.toml").read_text(encoding="utf-8")
    )
    expected = "rfc8785==0.1.4"
    assert expected in root_project["project"]["dependencies"]
    assert expected in standalone["project"]["dependencies"]
