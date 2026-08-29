from setuptools import find_packages


def test_root_distribution_discovers_profile_core_package():
    packages = {
        package
        for source_root in (".", "packages/tldw_profile_core/src")
        for package in find_packages(where=source_root, include=["tldw_chatbook*", "tldw_profile_core*"])
    }
    assert "tldw_profile_core" in packages
