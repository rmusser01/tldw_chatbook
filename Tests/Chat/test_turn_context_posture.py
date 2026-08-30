"""Task 7 (workspace assistant defaults): posture in the turn context and the
composition advertising filter.

Two seams:

- ``ConsoleTurnExecutionContext.capture`` gains ``persona_policy_rules`` and
  ``tool_policy_profile_id`` (frozen/detached like every other mapping the
  context holds; identity defaults of ``()`` / ``"default"``).
- ``_compose_run_registry_and_allowed`` gains ``persona_policy_rules`` and
  applies the persona policy as a NARROWING-ONLY advertising filter over the
  run's assembled allow-list (skill rules filter skill-provider names only;
  every other name evaluates under the ``mcp_tool`` kind), plus per-run call
  caps enforced at ``ToolCatalogRegistry.invoke_by_name``.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.agent_models import (
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Agents.run_tool_policy import PERSONA_POLICY_CALL_CAP_REFUSAL
from tldw_chatbook.Chat.console_agent_bridge import _compose_run_registry_and_allowed
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext


class _FakeToolsProvider:
    """Minimal ``ToolProvider`` double offering fixed names (mirrors the
    ``_FakeMCPProvider`` harness in ``Tests/Chat/test_console_agent_bridge.py``
    -- catalog/invoke seam only)."""

    def __init__(self, names: list[str], source: str = "mcp") -> None:
        self._names = list(names)
        self._source = source
        self.invoke_calls: list[tuple[str, dict]] = []

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=name, name=name, one_line_description="", source=self._source
            )
            for name in self._names
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id,
            description="",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content=f"result:{tool_id}")


def _selection() -> ConsoleProviderSelection:
    return ConsoleProviderSelection(provider="openai")


# -- turn-context capture -----------------------------------------------------


def test_capture_freezes_posture_values():
    rules = [{"rule_kind": "mcp_tool", "rule_name": "fs_*", "allowed": False}]
    context = ConsoleTurnExecutionContext.capture(
        session_id="session-1",
        provider_selection=_selection(),
        persona_policy_rules=rules,
        tool_policy_profile_id="ws-profile-1",
    )
    assert context.tool_policy_profile_id == "ws-profile-1"
    assert isinstance(context.persona_policy_rules, tuple)
    assert dict(context.persona_policy_rules[0]) == rules[0]
    # Detached: later mutation of the caller's structures must not leak in.
    rules[0]["allowed"] = True
    rules.append({"rule_kind": "skill", "rule_name": "x", "allowed": False})
    assert dict(context.persona_policy_rules[0])["allowed"] is False
    assert len(context.persona_policy_rules) == 1
    # Frozen: the stored rule mapping is itself immutable.
    with pytest.raises(TypeError):
        context.persona_policy_rules[0]["allowed"] = True


def test_capture_posture_defaults_are_identity():
    context = ConsoleTurnExecutionContext.capture(
        session_id="session-1",
        provider_selection=_selection(),
    )
    assert context.persona_policy_rules == ()
    assert context.tool_policy_profile_id == "default"


# -- composition advertising filter -------------------------------------------


def test_compose_filter_drops_denied_mcp_kind_tool():
    mcp = _FakeToolsProvider(["fs_write", "web_search"])
    _registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {},
            mcp_provider=mcp,
            persona_policy_rules=[
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "fs_*",
                    "allowed": False,
                }
            ],
        )
    )
    assert "fs_write" not in allowed_tools
    assert "web_search" in allowed_tools


def test_compose_filter_skill_rules_filter_only_skill_names():
    context = {
        "available_skills": [
            {
                "name": "code-review",
                "trust_blocked": False,
                "disable_model_invocation": False,
            }
        ]
    }
    mcp = _FakeToolsProvider(["fs_write"])
    _registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            context,
            mcp_provider=mcp,
            persona_policy_rules=[
                {
                    "rule_kind": "skill",
                    "rule_name": "code-review",
                    "allowed": False,
                }
            ],
        )
    )
    assert "code-review" not in allowed_tools
    # The skill-kind rule never touches non-skill names.
    assert "fs_write" in allowed_tools


def test_compose_filter_without_rules_is_identity():
    mcp = _FakeToolsProvider(["fs_write", "web_search"])
    _registry, baseline, _b, _l = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp
    )
    _registry, with_none, _b, _l = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp, persona_policy_rules=None
    )
    _registry, with_empty, _b, _l = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp, persona_policy_rules=[]
    )
    assert baseline == with_none == with_empty


# -- per-run call caps at the registry choke point ----------------------------


def test_compose_builds_run_call_caps_from_rule_verdicts():
    mcp = _FakeToolsProvider(["web_search", "fs_read"])
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            {},
            mcp_provider=mcp,
            persona_policy_rules=[
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "web_search",
                    "allowed": True,
                    "max_calls_per_turn": 1,
                }
            ],
        )
    )
    assert "web_search" in allowed_tools  # capped but still advertised
    with use_run_id("run-caps"):
        first = registry.invoke_by_name("web_search", {})
        assert first.ok is True
        second = registry.invoke_by_name("web_search", {})
        assert second.ok is False
        assert second.error == PERSONA_POLICY_CALL_CAP_REFUSAL.format(
            name="web_search"
        )
        # An uncapped sibling tool is untouched by the cap refusal.
        assert registry.invoke_by_name("fs_read", {}).ok is True


def test_registry_without_policy_invokes_uncapped():
    """No rules -> no policy set -> invoke_by_name behavior is unchanged."""
    mcp = _FakeToolsProvider(["web_search"])
    registry, _allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({}, mcp_provider=mcp)
    )
    with use_run_id("run-plain"):
        assert registry.invoke_by_name("web_search", {}).ok is True
        assert registry.invoke_by_name("web_search", {}).ok is True
