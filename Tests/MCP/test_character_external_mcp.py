"""TASK-32955: character reads for external MCP clients, and the guard and
server-mode refusal on ADR-183's writes.

Real SQLite, the real ``LocalCharacterPersonaService`` behind the real
``CharacterToolService``, the real gateway runtime, and a real
``MCPPermissionStore`` under a temp user-data dir.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tldw_chatbook.Agents.local_tool_provider as ltp
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.MCP import local_server_tools
from tldw_chatbook.MCP.builtin_tool_policy import BUILTIN_MCP_SERVER_KEY
from tldw_chatbook.MCP.permission_store import MCPPermissionStore, definition_hash
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Tools.character_tool_service import (
    CHARACTER_FIELD_READ_BOUND,
    SERVER_REFUSAL,
)

# Real config consumers retain the isolated profile selected during collection.
pytestmark = pytest.mark.bootstrap_profile

LONG = "The tutor explains every step twice. " * 200  # far past one card page
READS = {"character_search", "character_get"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    pytest.importorskip("mcp_unified.gateway")
    from tldw_chatbook import config
    from tldw_chatbook.MCP import tools as tools_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(tools_module, "SimplifiedRAGSearchService", lambda _: None)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)
    settings: dict[tuple[str, str], object] = {}
    monkeypatch.setattr(
        config, "get_cli_setting", lambda s, k, d=None: settings.get((s, k), d)
    )
    db = CharactersRAGDB(str(tmp_path / "characters.sqlite"), "mcp-external")
    yield SimpleNamespace(
        settings=settings,
        db=db,
        store=MCPPermissionStore(tmp_path / "mcp_permissions.json"),
        monkeypatch=monkeypatch,
    )
    db.close()


def _serve(env):
    """The real constructor, over this test's database instead of the profile's."""
    from tldw_chatbook.MCP.server import TldwMCPServer

    def _init_databases(server):
        server.chachanotes_db = env.db
        server.media_db = None
        server.notes_service = None

    env.monkeypatch.setattr(TldwMCPServer, "_init_databases", _init_databases)
    server = TldwMCPServer(version="1")
    env.tools = server.tools
    return server


def _context():
    gateway = pytest.importorskip("mcp_unified.gateway")
    return gateway.GatewayRequestContext(request_id="character-external")


async def _names(server):
    return {d["name"] for d in await server.mcp.list_tools(_context())}


async def _allow_reads(env, server):
    """An operator Allow on each read's own row (as the Hub sets it)."""
    for descriptor in await server.mcp.list_tools(_context()):
        if descriptor["name"] in READS:
            env.store.set_tool_state(
                "local:__local__",
                descriptor["name"],
                "allow",
                definition_hash=definition_hash(
                    descriptor["description"], descriptor["inputSchema"]
                ),
            )


async def _call(server, name, arguments):
    result = await server.mcp.call_tool(name, arguments, _context())
    return json.loads(result) if isinstance(result, str) else result


def _card(env, **fields):
    return env.db.add_character_card({"name": "Ada", **fields})


def _pin_source(env, source):
    from tldw_chatbook.runtime_policy import bootstrap

    env.monkeypatch.setattr(
        bootstrap,
        "load_default_runtime_source_state",
        lambda: RuntimeSourceState(active_source=source),
    )


# -- AC#1 / AC#4: exposure ----------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("local_tools", [False, True])
async def test_switch_off_publishes_no_character_tool(env, local_tools):
    env.settings[("mcp", "expose_local_tools")] = local_tools
    names = await _names(_serve(env))
    assert not {n for n in names if n.startswith("character_")}
    assert ("fs_read" in names) is local_tools  # control: the other switch works


@pytest.mark.asyncio
async def test_switch_on_publishes_callable_reads_and_never_save(env):
    env.settings[("mcp", "expose_character_tools")] = True
    character_id = _card(env, description="A patient tutor")
    server = _serve(env)
    names = await _names(server)
    assert READS <= names
    assert "character_save" not in names
    assert "fs_read" not in names  # independent of expose_local_tools
    from tldw_chatbook.MCP.gateway_runtime import GatewayToolExecutionError

    with pytest.raises(GatewayToolExecutionError) as unpublished:
        await server.mcp.call_tool("character_save", {"name": "X"}, _context())
    assert unpublished.value.reason_code == "tool_not_found"

    await _allow_reads(env, server)
    found = await _call(server, "character_search", {"query": "Ada"})
    assert [item["id"] for item in found["items"]] == [character_id]
    card = await _call(server, "character_get", {"id": character_id})
    assert card["fields"]["description"]["text"] == "A patient tutor"


@pytest.mark.asyncio
async def test_reads_ask_by_default_like_every_external_local_tool(env):
    env.settings[("mcp", "expose_character_tools")] = True
    server = _serve(env)
    from tldw_chatbook.MCP.gateway_runtime import GatewayToolExecutionError

    for name, arguments in (("character_search", {}), ("character_get", {"id": 1})):
        with pytest.raises(GatewayToolExecutionError) as refused:
            await server.mcp.call_tool(name, arguments, _context())
        assert refused.value.reason_code == "operator_approval_required"


@pytest.mark.asyncio
async def test_switch_is_independent_of_the_console_gate(env):
    """Console off + MCP on still publishes; Console on + MCP off does not."""
    env.monkeypatch.setattr(
        ltp,
        "get_cli_setting",
        lambda s, k, d=None: False if k == ltp.CHARACTER_TOOLS_GATE_KEY else d,
    )
    env.settings[("mcp", "expose_character_tools")] = True
    assert READS <= await _names(_serve(env))


def test_hub_local_composition_is_unchanged(tmp_path):
    """TASK-32956's rows stay Console-only and non-executable in the Hub."""
    root = tmp_path / "hub"
    root.mkdir()
    with local_server_tools.build_hub_local_inspection_provider(
        root, resolve_state=lambda _hub: None
    ) as inspection:
        console_only = inspection.provider.specs_for_exposure(
            ltp.LocalToolExposure.CONSOLE_ONLY
        )
        external = inspection.provider.specs_for_exposure(
            ltp.LocalToolExposure.CONSOLE_AND_EXTERNAL_MCP
        )
    rows = READS | {"character_save"}
    assert rows <= {s.name for s in console_only}
    assert not rows & {s.name for s in external}
    with local_server_tools.build_hub_local_provider(
        root, resolve_state=lambda _hub: None, approval_callback=None
    ) as shared:
        executable = shared.provider.specs_for_exposure(
            ltp.LocalToolExposure.CONSOLE_AND_EXTERNAL_MCP
        )
        assert executable  # control: the shared projection is not empty
        assert not rows & {s.name for s in executable}


# -- AC#3: truncation guard ---------------------------------------------------


def _grant_update(env):
    env.store.set_tool_state(BUILTIN_MCP_SERVER_KEY, "update_character", "allow")


@pytest.mark.asyncio
async def test_update_of_a_long_field_needs_a_full_character_get_first(env):
    assert len(LONG) > CHARACTER_FIELD_READ_BOUND
    env.settings[("mcp", "expose_character_tools")] = True
    character_id = _card(env, description=LONG, personality="Kind")
    server = _serve(env)
    await _allow_reads(env, server)
    _grant_update(env)
    patch = {"character_id": character_id, "expected_version": 1}

    refused = await _call(
        server, "update_character", {**patch, "fields": {"description": "Short"}}
    )
    assert refused == {
        "error_code": "read_full_field_first",
        "error": "Read the full 'description' field with character_get before "
        "changing it.",
    }
    assert env.db.get_character_card_by_id(character_id)["description"] == LONG

    # Control: a short field needs no prior read.
    ok = await _call(server, "update_character", {**patch, "fields": {"personality": "Warm"}})
    assert ok["version"] == 2
    patch["expected_version"] = 2

    # The whole-card read truncates the long field, so it does not unlock it.
    card = await _call(server, "character_get", {"id": character_id})
    assert card["fields"]["description"]["truncated"] is True
    still = await _call(server, "update_character", {**patch, "fields": {"description": "S"}})
    assert still["error_code"] == "read_full_field_first"

    offset = 0
    while True:
        page = await _call(
            server,
            "character_get",
            {"id": character_id, "field": "description", "offset": offset},
        )
        if "next_offset" not in page:
            break
        offset = page["next_offset"]
    updated = await _call(server, "update_character", {**patch, "fields": {"description": "S"}})
    assert updated["version"] == 3
    assert env.db.get_character_card_by_id(character_id)["description"] == "S"


@pytest.mark.asyncio
async def test_switch_off_keeps_adr_183_long_field_updates(env):
    """No read tool is exposed to satisfy a guard, so none applies (as on dev)."""
    character_id = _card(env, description=LONG)
    server = _serve(env)
    _grant_update(env)
    updated = await _call(
        server,
        "update_character",
        {"character_id": character_id, "expected_version": 1, "fields": {"description": "S"}},
    )
    assert updated.get("version") == 2, updated
    assert env.db.get_character_card_by_id(character_id)["description"] == "S"


@pytest.mark.asyncio
async def test_failed_read_registration_applies_no_guard(env):
    """A guard without its published read tool would strand long-field edits."""
    env.settings[("mcp", "expose_character_tools")] = True
    env.monkeypatch.setattr(
        local_server_tools, "_local_agent_tool_registrations", lambda _p: [None]
    )
    character_id = _card(env, description=LONG)
    server = _serve(env)
    assert not READS & await _names(server)
    _grant_update(env)
    updated = await _call(
        server,
        "update_character",
        {"character_id": character_id, "expected_version": 1, "fields": {"description": "S"}},
    )
    assert updated.get("version") == 2, updated


@pytest.mark.asyncio
async def test_in_process_runtime_applies_no_guard(env):
    """No read tool exists in-process; the operator runs the write by hand."""
    from tldw_chatbook.MCP.tools import MCPTools

    character_id = _card(env, description=LONG)
    tools = MCPTools(env.db, None)
    assert tools.character_read_guard is None
    updated = await tools.update_character(character_id, 1, {"description": "S"})
    assert updated["version"] == 2


# -- AC#3: server-mode refusal ------------------------------------------------


@pytest.mark.asyncio
async def test_server_mode_refuses_writes_and_reads(env):
    env.settings[("mcp", "expose_character_tools")] = True
    character_id = _card(env, description="A patient tutor")
    server = _serve(env)
    await _allow_reads(env, server)
    for name in ("create_character", "update_character"):
        env.store.set_tool_state(BUILTIN_MCP_SERVER_KEY, name, "allow")
    _pin_source(env, "server")
    before = env.db.list_character_cards()

    created = await _call(server, "create_character", {"name": "Grace"})
    updated = await _call(
        server,
        "update_character",
        {"character_id": character_id, "expected_version": 1, "fields": {"scenario": "x"}},
    )
    unsupported = {"error_code": "unsupported", "error": SERVER_REFUSAL}
    assert created == updated == unsupported
    for name, arguments in (("character_search", {}), ("character_get", {"id": character_id})):
        result = await _call(server, name, arguments)
        assert (result["status"], result["message"]) == ("unsupported", SERVER_REFUSAL)
    assert env.db.list_character_cards() == before
    assert env.db.get_character_card_by_id(character_id)["version"] == 1

    _pin_source(env, "local")  # control: the same calls succeed locally
    assert (await _call(server, "create_character", {"name": "Grace"}))["version"] == 1


# -- AC#2: explicit operator grant only ---------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["create_character", "update_character"])
async def test_writes_need_an_explicit_tool_grant(env, name):
    character_id = _card(env)
    before = len(env.db.list_character_cards())
    server = _serve(env)
    arguments = (
        {"name": "Grace"}
        if name == "create_character"
        else {"character_id": character_id, "expected_version": 1, "fields": {"scenario": "x"}}
    )
    refused = await _call(server, name, arguments)  # fresh store: ask
    assert refused["error_code"] == "permission_required"
    env.store.set_server_default(BUILTIN_MCP_SERVER_KEY, "allow")
    assert (await _call(server, name, arguments))["error_code"] == "permission_required"
    env.store.set_global_default("allow")
    assert (await _call(server, name, arguments))["error_code"] == "permission_required"
    assert env.db.get_character_card_by_id(character_id)["version"] == 1
    assert len(env.db.list_character_cards()) == before

    env.store.set_tool_state(BUILTIN_MCP_SERVER_KEY, name, "allow")  # control
    assert "error_code" not in await _call(server, name, arguments)
