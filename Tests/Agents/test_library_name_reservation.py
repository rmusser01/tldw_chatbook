"""Permanent ADR-079 Library-name reservation across provider modes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.library_rag_tool_provider import (
    LibraryRagToolProvider,
    RAG_TOOL_NAME,
)
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_agent_bridge import (
    SPAWN_TOOL_NAME,
    _BridgeSkillRunner,
    _compose_run_registry_and_allowed,
    build_console_first_request_plan,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
)
from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS


class _DirectService:
    def invoke(self, _name, _arguments):
        return {"items": [], "total": 0}


class _MCPProvider:
    def __init__(self, names):
        self._names = tuple(names)
        self.invoke_calls = []

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"mcp:{name}",
                name=name,
                one_line_description="MCP test tool",
                source="mcp",
            )
            for name in self._names
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id.split(":", 1)[1],
            description="MCP test tool",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, args))
        return ToolResult(ok=True, content="mcp")


def _reservation():
    from tldw_chatbook.Agents import tool_catalog

    return tool_catalog.LIBRARY_RESERVED_TOOL_NAMES


def _provider_for(mode):
    if mode == "direct":
        provider = LibraryToolProvider(_DirectService())
    elif mode == "rag":
        provider = LibraryRagToolProvider(None)
    else:
        return None, None
    authority = provider.issue_builtin_authority(
        reserved_names=_reservation(),
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )
    return provider, authority


def test_library_reservation_is_one_derived_immutable_union():
    reserved = _reservation()

    assert isinstance(reserved, frozenset)
    assert len(LIBRARY_TOOL_DESCRIPTORS) == 24
    assert len(reserved) == 25
    assert reserved == frozenset((*LIBRARY_TOOL_DESCRIPTORS.keys(), RAG_TOOL_NAME))


@pytest.mark.parametrize(
    "mode",
    ["blocked", "unavailable", "provider-not-registered", "direct", "rag"],
)
def test_every_reserved_skill_collision_is_filtered_without_registration(mode):
    reserved = _reservation()
    context = {
        "available_skills": [
            {
                "name": name,
                "description": "collision",
                "trust_blocked": False,
                "disable_model_invocation": False,
            }
            for name in sorted(reserved)
        ]
        + [
            {
                "name": "safe_skill",
                "description": "safe",
                "trust_blocked": False,
                "disable_model_invocation": False,
            }
        ]
    }
    provider, authority = _provider_for(mode)

    registry, allowed, _builtin, _local = _compose_run_registry_and_allowed(
        context,
        library_provider=provider,
        library_authority=authority,
    )
    skill_names = {
        entry.name for entry in registry.list_catalog() if entry.source == "skill"
    }

    assert skill_names == {"safe_skill"}
    assert "safe_skill" in allowed
    assert allowed[-1] == SPAWN_TOOL_NAME
    assert not (reserved & skill_names)


def test_reserved_skill_name_never_reaches_skill_runner_dispatch_when_blocked():
    context = {
        "available_skills": [
            {
                "name": "library_list_notes",
                "description": "collision",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
            {
                "name": "safe_skill",
                "description": "safe",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ]
    }
    plan = build_console_first_request_plan(
        shared_registry=ToolCatalogRegistry(),
        shared_allowed_tools=(),
        context=context,
        skills_present=True,
        mcp_provider=None,
        builtin_gate=None,
        local_provider=None,
        library_provider=None,
        library_authority=None,
        workspace_id=None,
        ephemeral=False,
        diff_sink=None,
        scratch_root=None,
        scratch_lease=None,
        resolution=SimpleNamespace(model="model-a", execution_key="openai"),
        fallback_model="model-a",
        session_system_prompt="",
        native_tools=True,
        turn_skill_bindings=(),
        turn_bundle_block="",
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        agent_messages=[],
    )
    runner = _BridgeSkillRunner(
        skills_service=SimpleNamespace(),
        skill_names=plan.skill_names,
        builtin_names=plan.builtin_names,
    )

    assert runner.is_skill_tool("library_list_notes") is False
    assert runner.is_skill_tool("safe_skill") is True


@pytest.mark.parametrize(
    "mode",
    ["blocked", "unavailable", "provider-not-registered", "direct", "rag"],
)
def test_exact_reserved_mcp_collisions_are_filtered_but_unrelated_names_survive(mode):
    reserved = _reservation()
    unrelated = "library_custom_export"
    mcp = _MCPProvider((*sorted(reserved), unrelated))
    provider, authority = _provider_for(mode)

    registry, allowed, _builtin, _local = _compose_run_registry_and_allowed(
        {},
        mcp_provider=mcp,
        library_provider=provider,
        library_authority=authority,
    )
    mcp_names = {
        entry.name for entry in registry.list_catalog() if entry.source == "mcp"
    }

    assert mcp_names == {unrelated}
    assert unrelated in allowed
    assert not (reserved & mcp_names)
    assert registry.invoke_by_name(unrelated, {}).content == "mcp"
