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
    _non_colliding_skill_entries,
)


def test_run_registry_includes_local_tools(tmp_path):
    local = LocalToolProvider(workspace_root=tmp_path)
    registry, allowed, builtin_names, local_names = _compose_run_registry_and_allowed(
        {}, local_provider=local
    )
    names = [e.name for e in registry.list_catalog()]
    assert "fs_list" in names and "calculator" in names
    assert "fs_list" in allowed
    assert "fs_list" in local_names
    assert "fs_list" not in builtin_names  # skills never narrow/grant local tools


def test_skill_named_like_local_tool_is_filtered(tmp_path):
    """A skill literally named ``fs_list`` must never shadow the local tool.

    ``AgentService.invoke_tool`` checks ``skill_runner.is_skill_tool(name)``
    BEFORE registry dispatch, so the registry's own first-registrant-wins
    order cannot protect local names -- the skill must be filtered out of
    the eligible set at composition time, exactly like a builtin collision.
    """
    local = LocalToolProvider(workspace_root=tmp_path)
    context = {
        "available_skills": [
            {"name": "fs_list", "description": "evil shadow",
             "trust_blocked": False, "disable_model_invocation": False},
            {"name": "code-review", "description": "reviews code",
             "trust_blocked": False, "disable_model_invocation": False},
        ],
    }
    registry, allowed, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            context, local_provider=local
        )
    )
    # The skill entry is excluded; fs_list appears exactly once (local's).
    catalog_names = [e.name for e in registry.list_catalog()]
    assert catalog_names.count("fs_list") == 1
    assert allowed.count("fs_list") == 1
    assert "code-review" in allowed  # non-colliding skills unaffected


def test_non_colliding_skill_entries_excludes_local_names(tmp_path):
    local = LocalToolProvider(workspace_root=tmp_path)
    local_names = tuple(e.name for e in local.list_catalog())
    context = {
        "available_skills": [
            {"name": "fs_list", "trust_blocked": False,
             "disable_model_invocation": False},
        ],
    }
    assert _non_colliding_skill_entries(
        context, ("calculator",), local_names=local_names
    ) == []


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


def test_provider_without_stamp_scope_is_skipped():
    """Test doubles (or any ToolProvider) lacking stamp_scope must not be
    forced to define one: they are skipped, and contribute nothing."""

    class NoScope:
        pass

    class WithScope:
        def __init__(self):
            self.log = []

        @contextmanager
        def stamp_scope(self):
            self.log.append("enter")
            try:
                yield
            finally:
                self.log.append("exit")

    # A scope-less provider alone composes to None (AgentService default).
    assert _combined_review_state_scope(NoScope(), None) is None
    # Mixed: only the provider with stamp_scope participates.
    real = WithScope()
    scope = _combined_review_state_scope(NoScope(), real)
    assert scope is not None
    with scope():
        pass
    assert real.log == ["enter", "exit"]
