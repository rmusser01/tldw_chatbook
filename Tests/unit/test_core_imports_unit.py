# test_core_imports_unit.py
# Unit tests for core import functionality
#
"""
Unit tests for verifying core module imports work without optional dependencies.

These tests verify that the core modules can be imported successfully
without requiring optional dependencies to be installed.
"""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_core_modules_import():
    """The core modules import cleanly.

    Renamed from `test_core_imports_without_optional_deps`, which is the claim
    this cannot make: every optional group is installed in the environments this
    runs in, so importing successfully here says nothing about a machine without
    them. `Utils/optional_deps.py` is where that property is actually decided.

    What it can check, and now does, is that these four import at all. The
    previous version had had every import deleted down to a bare comment and
    ended in `assert True`, so it passed even while claiming to exercise them --
    a module could have been renamed or made unimportable and this would still
    have been green.
    """
    import importlib

    for module in (
        "tldw_chatbook.config",
        "tldw_chatbook.DB.ChaChaNotes_DB",
        "tldw_chatbook.Chat.Chat_Functions",
        "tldw_chatbook.Utils.optional_deps",
    ):
        assert importlib.import_module(module) is not None, module
