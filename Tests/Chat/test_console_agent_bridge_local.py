"""Tests for the bridge-side local-tool wiring (Task 6).

Covers: LocalToolProvider registration in the per-run registry/allow-list
(builtin -> local -> skill -> MCP shadowing order) and the combined
review_state_scope that isolates BOTH providers' approval stamps around
nested sub-agent runs (ADR-032's third mechanism).
"""

from contextlib import contextmanager

from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Chat.console_agent_bridge import (
    _combined_review_state_scope,
    _compose_run_registry_and_allowed,
)


def test_run_registry_includes_local_tools(tmp_path):
    local = LocalToolProvider(workspace_root=tmp_path)
    registry, allowed, builtin_names = _compose_run_registry_and_allowed(
        {}, local_provider=local
    )
    names = [e.name for e in registry.list_catalog()]
    assert "fs_list" in names and "calculator" in names
    assert "fs_list" in allowed
    assert "fs_list" not in builtin_names  # skills never narrow/grant local tools


def test_combined_stamp_scope_isolates_both(tmp_path):
    class FakeProvider:
        def __init__(self):
            self.stamps = {"x": "approve_once"}
            self.log = []

        @contextmanager
        def stamp_scope(self):
            saved = self.stamps
            self.stamps = {}
            self.log.append("enter")
            try:
                yield
            finally:
                self.stamps = saved
                self.log.append("exit")

    p1, p2 = FakeProvider(), FakeProvider()
    scope = _combined_review_state_scope(p1, p2)
    assert scope is not None
    with scope():
        assert p1.stamps == {} and p2.stamps == {}
    assert p1.stamps == {"x": "approve_once"} and p2.stamps == {"x": "approve_once"}
    assert p1.log == ["enter", "exit"] and p2.log == ["enter", "exit"]
    assert _combined_review_state_scope(None, None) is None
    assert _combined_review_state_scope(p1, None) is not None  # Nones skipped
