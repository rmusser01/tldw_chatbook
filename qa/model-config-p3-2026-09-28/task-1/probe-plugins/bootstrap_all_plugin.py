"""Scratch plugin: apply the repo's own `bootstrap_profile` opt-in marker
(Tests/conftest.py, TASK-32873) to every collected node, so each keeps the
sandboxed collection-time profile under a scratch TLDW_TEST_CONFIG_ROOT
instead of tripping ADR-126 under the per-test redirect. No test is edited;
HOME/XDG/TLDW_CONFIG_PATH still point into the scratch root (asserted)."""
import os

import pytest


def pytest_collection_modifyitems(items):
    root = os.environ["TLDW_TEST_CONFIG_ROOT"]
    assert "scratchpad" in root and os.environ["HOME"].startswith(root), root
    for item in items:
        item.add_marker(pytest.mark.bootstrap_profile)
