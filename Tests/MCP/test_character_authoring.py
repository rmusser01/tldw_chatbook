"""Character authoring exercises real persistence and both MCP entry points."""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.MCP.hub_tool_catalog import builtin_tools_from_inventory
from tldw_chatbook.MCP.permission_store import (
    MCPPermissionStore,
    resolve_effective_state,
    resolve_effective_state_by_key,
)
from tldw_chatbook.MCP.server import describe_local_mcp_capabilities

# Real config consumers retain the isolated profile selected during collection.
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def character_tools(tmp_path, monkeypatch):
    from tldw_chatbook.MCP import tools as module

    monkeypatch.setattr(module, "SimplifiedRAGSearchService", lambda _: None)
    db = CharactersRAGDB(str(tmp_path / "characters.sqlite"), "mcp-authoring")
    tools = module.MCPTools(db, None)
    yield tools
    db.close()


@pytest.mark.asyncio
async def test_create_list_update_and_stale_version_preserve_card(character_tools):
    created = await character_tools.create_character(
        "Ada", {"description": "Original", "tags": ["math"]}
    )
    assert created == {"id": created["id"], "name": "Ada", "version": 1}
    roster = await character_tools.list_available_characters()
    assert next(row for row in roster if row["id"] == created["id"])["version"] == 1
    updated = await character_tools.update_character(
        created["id"], 1, {"description": "Updated"}
    )
    assert updated["version"] == 2
    stale = await character_tools.update_character(
        created["id"], 1, {"description": "Lost update"}
    )
    assert stale["error_code"] == "conflict"
    record = character_tools.chachanotes_db.get_character_card_by_id(created["id"])
    assert record["description"] == "Updated"
    assert record["tags"] == ["math"]
    duplicate = await character_tools.create_character("Ada")
    assert duplicate["error_code"] == "conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,fields",
    [
        (" ", None),
        ("Ada", {"image_base64": "abc"}),
        ("Ada", {"unknown": "value"}),
        ("Ada", {"name": "Override"}),
        ("Ada", {"description": "x" * 50001}),
        ("Ada", []),
    ],
)
async def test_invalid_create_is_explicit_and_does_not_write(
    character_tools, name, fields
):
    before = character_tools.chachanotes_db.list_character_cards()
    result = await character_tools.create_character(name, fields)
    assert result["error_code"] == "invalid_arguments"
    assert character_tools.chachanotes_db.list_character_cards() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version,fields",
    [
        (True, {"name": "B"}),
        (0, {"name": "B"}),
        ("1", {"name": "B"}),
        (1, {}),
        (1, {"name": " "}),
        (1, {"description": None}),
    ],
)
async def test_invalid_update_preserves_card(character_tools, version, fields):
    created = await character_tools.create_character("Ada")
    result = await character_tools.update_character(created["id"], version, fields)
    assert result["error_code"] == "invalid_arguments"
    assert (
        character_tools.chachanotes_db.get_character_card_by_id(created["id"])[
            "version"
        ]
        == 1
    )


@pytest.mark.asyncio
async def test_missing_update_has_structured_error(character_tools):
    result = await character_tools.update_character(
        99999, 1, {"description": "Missing"}
    )
    assert result["error_code"] == "not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("tags", "null"),
        ("tags", '"science"'),
        ("alternate_greetings", "null"),
        ("extensions", "null"),
    ],
)
async def test_normalized_empty_or_scalar_json_fields_cannot_report_success(
    character_tools, field, value
):
    created = await character_tools.create_character("Ada", {"tags": ["math"]})
    before = character_tools.chachanotes_db.get_character_card_by_id(created["id"])
    result = await character_tools.update_character(created["id"], 1, {field: value})
    assert result["error_code"] == "invalid_arguments"
    assert (
        character_tools.chachanotes_db.get_character_card_by_id(created["id"]) == before
    )


@pytest.mark.asyncio
async def test_encoded_json_collections_are_normalized_before_authoring(
    character_tools,
):
    created = await character_tools.create_character(
        "Ada", {"tags": '["math"]', "extensions": '{"topic":"arithmetic"}'}
    )
    record = character_tools.chachanotes_db.get_character_card_by_id(created["id"])
    assert record["tags"] == ["math"]
    assert record["extensions"] == {"topic": "arithmetic"}


@pytest.mark.parametrize("name", ["create_character", "update_character"])
def test_character_write_permissions_match_with_and_without_catalog(name, tmp_path):
    catalog = builtin_tools_from_inventory(describe_local_mcp_capabilities())
    tool = next(tool for tool in catalog if tool.name == name)
    assert tool.tags == ("mutates",)
    store = MCPPermissionStore(tmp_path / "permissions.json")
    store.set_server_default(tool.server_key, "allow")
    for expected in ("ask", "allow", "deny"):
        if expected != "ask":
            store.set_tool_state(tool.server_key, name, expected)
        payload = store.load()
        live = resolve_effective_state(payload, tool)
        by_key = resolve_effective_state_by_key(payload, tool.server_key, name)
        assert live.state == by_key.state == expected
        assert live.risk_floored == by_key.risk_floored


@pytest.mark.asyncio
async def test_in_process_runtime_authoring_parity(character_tools):
    from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate

    delegate = LocalMCPRuntimeDelegate(
        manifest_provider=describe_local_mcp_capabilities
    )
    delegate._tools = character_tools
    created = await delegate.execute_tool("create_character", {"name": "Ada"})
    updated = await delegate.execute_tool(
        "update_character",
        {
            "character_id": created["id"],
            "expected_version": 1,
            "fields": {"personality": "Curious"},
        },
    )
    assert updated["version"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name,action",
    [
        ("create_character", "character.persona.create.local"),
        ("update_character", "character.persona.update.local"),
    ],
)
async def test_runtime_policy_refuses_character_writes(
    character_tools, tool_name, action
):
    from Tests.MCP.test_local_control_service import FakeLocalStore, FakeMCPClient
    from tldw_chatbook.MCP.local_control_service import (
        LocalMCPControlService,
        MCPGovernanceDenied,
    )
    from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate

    delegate = LocalMCPRuntimeDelegate(
        manifest_provider=describe_local_mcp_capabilities
    )
    delegate._tools = character_tools
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        client=FakeMCPClient(),
        manifest_provider=describe_local_mcp_capabilities,
        runtime_delegate=delegate,
    )
    created = await character_tools.create_character("Ada")
    service.save_governance_rule(
        {"rule_id": "deny-card-write", "capability_id": action, "decision": "deny"}
    )
    before = character_tools.chachanotes_db.list_character_cards()
    with pytest.raises(MCPGovernanceDenied, match=action):
        await service.execute_tool(
            tool_name,
            {
                "name": "Grace",
                "character_id": created["id"],
                "expected_version": 1,
                "fields": {"description": "Blocked"},
            },
        )
    assert character_tools.chachanotes_db.list_character_cards() == before


@pytest.mark.asyncio
async def test_standalone_requires_grant_and_honors_revocation(
    character_tools, tmp_path, monkeypatch
):
    gateway = pytest.importorskip("mcp_unified.gateway")
    from tldw_chatbook import config
    from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime
    from tldw_chatbook.MCP.server import TldwMCPServer, _describe_local_tools

    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)
    instance = TldwMCPServer.__new__(TldwMCPServer)
    instance.tools = character_tools
    instance.mcp = ChatbookGatewayRuntime(
        name="test", version="1", tool_descriptors=_describe_local_tools()
    )
    instance._register_tools()
    instance.mcp.finalize()
    context = gateway.GatewayRequestContext(request_id="character-authoring")
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    before = character_tools.chachanotes_db.list_character_cards()
    denied = await instance.mcp.call_tool("create_character", {"name": "Ada"}, context)
    assert denied["error_code"] == "permission_required"
    assert character_tools.chachanotes_db.list_character_cards() == before
    store.set_tool_state("builtin:tldw_chatbook", "create_character", "allow")
    created = await instance.mcp.call_tool("create_character", {"name": "Ada"}, context)
    assert created["version"] == 1
    store.set_tool_state("builtin:tldw_chatbook", "update_character", "allow")
    updated = await instance.mcp.call_tool(
        "update_character",
        {
            "character_id": created["id"],
            "expected_version": 1,
            "fields": {"name": "Grace"},
        },
        context,
    )
    assert updated["name"] == "Grace"
    store.set_kill_switch(True)
    refused = await instance.mcp.call_tool(
        "create_character", {"name": "Blocked"}, context
    )
    assert refused["error_code"] == "permission_denied"
    store.set_kill_switch(False)
    store.set_tool_state("builtin:tldw_chatbook", "create_character", "deny")
    refused = await instance.mcp.call_tool(
        "create_character", {"name": "Blocked"}, context
    )
    assert refused["error_code"] == "permission_denied"
    assert len(character_tools.chachanotes_db.list_character_cards()) == len(before) + 1
