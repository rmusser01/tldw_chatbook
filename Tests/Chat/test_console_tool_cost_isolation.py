"""TASK-33621.4: every test saves Console agent tool costs to its own file.

Review 2026-11-23 (round 6): the per-test cost file was a Tests/UI fixture,
but about 580 bootstrap-profile test files outside Tests/UI also drive the
Console (``Tests/Chat/test_console_light_surface_context_meter.py`` among
them) and share one data folder per pytest process. The fixture now lives in
``Tests/conftest.py``; this checks it from outside Tests/UI.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_a_non_ui_test_saves_tool_costs_to_its_own_file(
    console_tool_cost_file, tmp_path
) -> None:
    from tldw_chatbook import console_tool_costs as store

    assert store.tool_cost_path() == console_tool_cost_file
    assert tmp_path in console_tool_cost_file.parents
    store.save_costs({"key": 7})
    assert store.load_costs(console_tool_cost_file) == {"key": 7}
