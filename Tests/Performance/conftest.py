"""Shared fixtures for the performance guards.

The four boot-budget ratchet guards (TASK-23029 / ADR-097) share their policy
text, snapshot IO and diff formatting through ``boot_budget_ratchet.py`` in
this directory. ``Tests`` is not a package (``--import-mode=importlib``), so
the module is loaded by file path here, once, and handed to tests as a
fixture. ``scripts/update_boot_budget_snapshots.py`` loads it the same way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_RATCHET_MODULE_NAME = "tldw_tests_boot_budget_ratchet"


def load_boot_budget_ratchet() -> ModuleType:
    """Load (or return the already-loaded) shared ratchet helper module."""
    module = sys.modules.get(_RATCHET_MODULE_NAME)
    if module is not None:
        return module
    path = Path(__file__).with_name("boot_budget_ratchet.py")
    spec = importlib.util.spec_from_file_location(_RATCHET_MODULE_NAME, path)
    assert spec is not None and spec.loader is not None, f"unloadable: {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RATCHET_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ratchet() -> ModuleType:
    """The shared boot-budget ratchet helper (policy text, snapshots, diffs)."""
    return load_boot_budget_ratchet()
