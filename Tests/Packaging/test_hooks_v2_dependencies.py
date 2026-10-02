"""The installed base distribution includes H3's offline schema runtime."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from setuptools.config.pyprojecttoml import apply_configuration
from setuptools.dist import Distribution

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture(scope="module")
def base_requirements():
    distribution = apply_configuration(
        Distribution(), str(Path(__file__).parents[2] / "pyproject.toml")
    )
    return {
        requirement.name: requirement
        for requirement in map(Requirement, distribution.install_requires)
    }


@pytest.mark.parametrize(
    ("name", "bounds"),
    [
        ("jsonschema", ">=4.26,<5"),
        ("referencing", ">=0.37,<1"),
        ("pydantic", ">=2.4,<3"),
    ],
)
def test_base_distribution_declares_offline_schema_runtime(
    base_requirements, name, bounds
):
    expected = Requirement(name + bounds)
    assert name in base_requirements
    requirement = base_requirements[name]
    assert requirement.marker is None
    assert requirement.specifier == expected.specifier
