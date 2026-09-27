"""TASK-32956: character_* tools in the MCP hub per-tool Permissions catalog.

End to end over real objects: the hub's own catalog method
(``UnifiedMCPControlPlaneService.local_hub_tools``), the hub's own write path
(``set_tool_state``) into a real ``MCPPermissionStore``, and the Console's own
provider composition (``ConsoleChatController._compose_local_provider``) over a
real in-memory character database. Only config reads are pinned (never the
real config loader -- ADR-126 / test isolation).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tldw_chatbook.Agents.local_tool_provider as ltp
import tldw_chatbook.Chat.console_chat_controller as controller_mod
import tldw_chatbook.MCP.unified_control_plane_service as control_plane_module
from Tests.Chat.test_console_local_review_hook import (
    _bare_controller,
    _compose_local_provider,
)
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.MCP import local_server_tools
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)

CHARACTER_TOOLS = {"character_search", "character_get", "character_save"}


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A real control-plane service over a real permission store."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    gate = {"on": True}

    def tool_setting(section, key, default=None):
        if (section, key) == ("tools", ltp.CHARACTER_TOOLS_GATE_KEY):
            return gate["on"]
        return default

    monkeypatch.setattr(ltp, "get_cli_setting", tool_setting)
    monkeypatch.setattr(
        control_plane_module, "get_cli_setting", lambda s, k, d=None: d
    )
    monkeypatch.setattr(
        local_server_tools, "resolve_server_workspace_root", lambda: workspace
    )
    service = UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=None, server_service=None
    )
    service._permission_store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    return SimpleNamespace(service=service, gate=gate, workspace=workspace)


def _rows(service):
    return {tool.name: tool for tool in service.local_hub_tools()}


def test_rows_listed_when_gate_on(hub):
    rows = _rows(hub.service)
    assert CHARACTER_TOOLS <= set(rows)
    for name in CHARACTER_TOOLS:
        assert rows[name].server_key == "local:__local__"
        # Console-only: listed for permissions, never executable from the hub.
        assert rows[name].executable is False
    assert rows["character_save"].tags == ("mutates",)


def test_rows_absent_when_gate_off(hub):
    hub.gate["on"] = False
    assert CHARACTER_TOOLS.isdisjoint(_rows(hub.service))


def _console(monkeypatch, hub, answer=None):
    """The Console's real local-provider composition over a real card DB.

    ``answer`` is what the approval card returns for every pending row;
    ``None`` means nobody answers (the call times out).
    """
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        lambda s, k=None, d=None: True if (s, k) == ("console", "local_tools_enabled") else d,
    )
    db = CharactersRAGDB(":memory:", client_id="test")
    local = LocalCharacterPersonaService(db=db)
    card_id = local.create_character({"name": "Aria", "description": "A pilot"})["id"]
    controller = _bare_controller(
        SimpleNamespace(
            unified_mcp_service=hub.service,
            local_character_persona_service=local,
            post_message=lambda _message: True,
        )
    )
    controller.store = persisted_console_store()
    controller._character_read_guards = {}
    session = controller.store.create_session(runtime_backend="local")
    approvals: list = []

    def request_mcp_approvals(pending, *, session_id=None):
        approvals.extend(pending)
        return {} if answer is None else {p.llm_name: answer for p in pending}

    controller.request_mcp_approvals = request_mcp_approvals
    provider, _hook = _compose_local_provider(controller, session.id)
    assert provider is not None
    return provider, approvals, card_id


def _allow_in_hub(service, name):
    """Set Allow exactly the way the hub's Permissions row does."""
    row = _rows(service)[name]
    service.set_tool_state(row.server_key, row.name, "allow", tool=row)


@pytest.mark.parametrize("name", ["character_search", "character_get"])
def test_hub_allow_lets_the_read_run_in_the_console_without_a_card(
    monkeypatch, hub, name
):
    provider, approvals, card_id = _console(monkeypatch, hub)
    args = {"id": card_id} if name == "character_get" else {}

    # Default posture: the read asks.
    assert provider.pending_gate_for(name, args) is not None

    _allow_in_hub(hub.service, name)

    assert provider.pending_gate_for(name, args) is None
    result = provider.invoke(f"local:{name}", args)
    assert approvals == []
    assert result.ok is True
    assert json.loads(result.content)["status"] == "ok"


def test_character_save_still_asks_every_call_when_set_to_allow(monkeypatch, hub):
    provider, approvals, _card_id = _console(monkeypatch, hub)
    _allow_in_hub(hub.service, "character_save")

    # The hub shows the save floored to Ask, not a plain Allow.
    row = _rows(hub.service)["character_save"]
    state = hub.service.gate_tool_test(row)
    assert state.state == "ask" and state.risk_floored is True

    for attempt in range(2):
        gate = provider.pending_gate_for("character_save", {"name": f"Nova{attempt}"})
        assert gate is not None and gate.reason == "risk_floored"
        result = provider.invoke("local:character_save", {"name": f"Nova{attempt}"})
        assert result.ok is False  # unanswered card: nothing saved
    assert [p.tool_name for p in approvals] == ["character_save", "character_save"]


def test_hub_inspector_notice_names_the_every_call_floor():
    # An explicit Allow floored to Ask must not claim the default was inherited.
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import _risk_floored_notice

    explicit = EffectiveToolState("ask", "tool_override", risk_floored=True)
    inherited = EffectiveToolState("ask", "global_default", risk_floored=True)
    assert _risk_floored_notice(explicit) == "Asks on every call, even when set to Allow."
    assert "inherited default" in _risk_floored_notice(inherited)


# -- Fix round: the save card offers only Approve once / Deny ---------------

_SAVE = ("local:__local__", "character_save")


def _no_standing_save_grant(service):
    assert service.is_session_approved(*_SAVE) is False
    assert service.permission_store.get_tool_entry(*_SAVE) is None


def test_save_card_offers_only_approve_once_or_deny(monkeypatch, hub):
    provider, _approvals, card_id = _console(monkeypatch, hub)
    save = provider.pending_gate_for("character_save", {"name": "Nova"})
    assert save.options == ("approve_once", "deny")
    # The reads keep the ordinary whole-tool choices.
    read = provider.pending_gate_for("character_get", {"id": card_id})
    assert "approve_session" in read.options and "always_allow" in read.options
    # A session grant from any other path is not honoured for the save.
    hub.service.approve_for_session(*_SAVE)
    assert provider.pending_gate_for("character_save", {"name": "Nova"}) is not None


@pytest.mark.parametrize("answer", ["approve_once", "approve_session", "always_allow"])
def test_the_next_save_still_asks_after_one_is_approved(monkeypatch, hub, answer):
    # approve_session / always_allow cannot be picked on the narrowed card;
    # a stale or forged one must still count as this call only.
    provider, approvals, _card_id = _console(monkeypatch, hub, answer=answer)
    for attempt in range(2):
        result = provider.invoke("local:character_save", {"name": f"Nova{attempt}"})
        assert result.ok is True
        assert json.loads(result.content)["status"] == "saved"
    assert [p.tool_name for p in approvals] == ["character_save", "character_save"]
    _no_standing_save_grant(hub.service)


@pytest.mark.parametrize("forged", ["approve_session", "always_allow"])
def test_forged_stamp_writes_no_grant_and_the_next_turn_asks(monkeypatch, hub, forged):
    # The Console's batch-review path: a stamped verdict applied at invoke.
    from tldw_chatbook.Agents.run_context import use_run_id

    provider, approvals, _card_id = _console(monkeypatch, hub)
    provider.apply_batch_decisions("run-forged", {"character_save": forged})
    with use_run_id("run-forged"):
        first = provider.invoke("local:character_save", {"name": "Nova"})
    assert first.ok is True
    _no_standing_save_grant(hub.service)

    # Next turn: the review hook clears the run's stamps on entry.
    provider.apply_batch_decisions("run-forged", {})
    with use_run_id("run-forged"):
        second = provider.invoke("local:character_save", {"name": "Vega"})
    assert second.ok is False  # asked again; nobody answered
    assert [p.tool_name for p in approvals] == ["character_save"]


# -- Fix round: Tool Pack export carries the character rules ----------------


class _NoServers:
    """The v1 registry's two local-control reads, with nothing registered."""

    def get_inventory(self):
        return {"tools": []}

    def get_external_servers(self):
        return []


def test_tool_pack_export_carries_a_stored_character_rule(hub):
    from tldw_chatbook.Tool_Packs.catalog_snapshot import PermissionInventoryRegistry
    from tldw_chatbook.Tool_Packs.export import ToolPackExportService

    _allow_in_hub(hub.service, "character_get")
    registry = PermissionInventoryRegistry.v1(_NoServers(), fallback_root=hub.workspace)
    review = ToolPackExportService(hub.service.permission_store, registry).capture(
        profile_id="default", display_name="Characters", suggested_id="characters"
    )
    rules = {
        (rule.server_key, rule.tool_name): rule.state for rule in review.payload.tools
    }
    assert ("local:__local__", "character_get") not in review.omitted_allow_ask
    assert rules[("local:__local__", "character_get")] == "allow"
    assert rules[("local:__local__", "character_save")] == "ask"
