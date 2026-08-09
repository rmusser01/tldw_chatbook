"""Tests for MCP/local_server_tools.py — external-client local tool composition.

The composition builds a LocalToolProvider for non-Console MCP serving:
permission state resolved FRESH per call from a real MCPPermissionStore,
no approval callback (ask fails closed with the external refusal copy),
kill switch honored, and no Console-only seams (todo store, session
approvals, persistence).
"""

import pytest
from loguru import logger

from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_GATE_ERROR_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
)
from tldw_chatbook.MCP.local_server_tools import (
    EXTERNAL_NO_CALLBACK_REFUSAL,
    LocalToolRegistration,
    _local_agent_tool_registrations,
    build_server_local_provider,
)
from tldw_chatbook.MCP.permission_store import MCPPermissionStore, definition_hash
from tldw_chatbook.MCP.server import TldwMCPServer, _describe_local_tools


BUILTIN_TOOL_NAMES = [descriptor["name"] for descriptor in _describe_local_tools()]


def _context():
    gateway = pytest.importorskip("mcp_unified.gateway")
    return gateway.GatewayRequestContext(request_id="test-request")


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "hello.txt").write_text("hello world\n")
    return root


@pytest.fixture
def store(tmp_path):
    return MCPPermissionStore(tmp_path / "mcp_permissions.json")


def _grant(store, provider, name):
    """Persist an operator allow the way Console "Always allow" does."""
    hub = provider.hub_tool_for(name)
    store.set_tool_state(
        hub.server_key,
        hub.name,
        "allow",
        definition_hash=definition_hash(hub.description, hub.input_schema),
    )


def test_granted_tool_executes(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert r.ok and "hello world" in r.content


def test_default_ask_fails_closed_with_external_refusal(workspace, store):
    # Nothing granted (an untouched store file) -> global default "ask",
    # no approval callback -> external no-callback refusal.
    provider = build_server_local_provider(workspace, store)
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == EXTERNAL_NO_CALLBACK_REFUSAL


def test_missing_store_file_fails_closed(workspace, tmp_path):
    missing = MCPPermissionStore(tmp_path / "no-such-dir" / "mcp_permissions.json")
    provider = build_server_local_provider(workspace, missing)
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == EXTERNAL_NO_CALLBACK_REFUSAL


def test_kill_switch_refuses_even_granted_tools(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    store.set_kill_switch(True)
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL


def test_kill_switch_read_failure_fails_closed(workspace):
    sentinel = "SENTINEL /private/path API_KEY=secret"

    class RaisingStore:
        def load(self):
            return {}

        def get_kill_switch(self):
            raise RuntimeError(sentinel)

    provider = build_server_local_provider(workspace, RaisingStore())
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    finally:
        logger.remove(sink_id)
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL
    assert all(sentinel not in record for record in records)


@pytest.mark.parametrize("failure_stage", ["load", "resolution"])
def test_permission_store_read_failure_is_payload_free(workspace, failure_stage):
    sentinel = "SENTINEL /private/path API_KEY=secret"

    class RaisingPayload(dict):
        def get(self, _key, _default=None):
            raise RuntimeError(sentinel)

    class RaisingStore:
        def load(self):
            if failure_stage == "load":
                raise RuntimeError(sentinel)
            return RaisingPayload()

        def get_kill_switch(self):
            return False

    provider = build_server_local_provider(workspace, RaisingStore())
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        result = provider.invoke("local:fs_read", {"path": sentinel})
    finally:
        logger.remove(sink_id)

    assert not result.ok and result.error == LOCAL_GATE_ERROR_REFUSAL
    assert all(sentinel not in record for record in records)


def test_deny_state_refuses(workspace, store):
    provider = build_server_local_provider(workspace, store)
    hub = provider.hub_tool_for("fs_read")
    store.set_tool_state(hub.server_key, hub.name, "deny")
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


def test_resolve_state_reads_store_fresh_per_call(workspace, store):
    # Operator changes take effect immediately: grant -> executes, revoke
    # -> fails closed, all against the same composed provider.
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    assert provider.invoke("local:fs_read", {"path": "hello.txt"}).ok
    hub = provider.hub_tool_for("fs_read")
    store.set_tool_state(hub.server_key, hub.name, None)  # operator revokes
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == EXTERNAL_NO_CALLBACK_REFUSAL


def test_todo_write_absent_from_catalog(workspace, store):
    # No todo_store is handed in (Console-session-scoped state), so the
    # todo_write spec must not be registered at all.
    provider = build_server_local_provider(workspace, store)
    names = [e.name for e in provider.list_catalog()]
    assert "todo_write" not in names
    assert "fs_read" in names


# -- _local_agent_tool_registrations (pure builder, no mcp package needed) --


def _registrations(provider):
    return {r.name: r for r in _local_agent_tool_registrations(provider)}


def test_granted_tool_registration_handler_executes(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    regs = _registrations(provider)
    assert "fs_read" in regs
    reg = regs["fs_read"]
    # Introspection fields come from the provider's load_schema.
    assert reg.description
    assert reg.parameters.get("type") == "object"
    result = reg.handler({"path": "hello.txt"})
    assert isinstance(result, ToolResult)
    assert result.ok and "hello world" in result.content


def test_ask_state_handler_returns_tool_result(workspace, store):
    # fs_write with nothing granted -> ask default -> external refusal.
    provider = build_server_local_provider(workspace, store)
    result = _registrations(provider)["fs_write"].handler(
        {"path": "x.txt", "content": "y"}
    )
    assert result == ToolResult(ok=False, error=EXTERNAL_NO_CALLBACK_REFUSAL)


def test_deny_state_handler_returns_tool_result(workspace, store):
    provider = build_server_local_provider(workspace, store)
    hub = provider.hub_tool_for("fs_glob")
    store.set_tool_state(hub.server_key, hub.name, "deny")
    result = _registrations(provider)["fs_glob"].handler({"pattern": "*.py"})
    assert result == ToolResult(ok=False, error=LOCAL_DENY_REFUSAL)


def test_kill_switch_handler_returns_tool_result(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    store.set_kill_switch(True)
    result = _registrations(provider)["fs_read"].handler({"path": "hello.txt"})
    assert result == ToolResult(ok=False, error=LOCAL_KILL_SWITCH_REFUSAL)


def test_todo_write_absent_from_registrations(workspace, store):
    provider = build_server_local_provider(workspace, store)
    regs = _registrations(provider)
    assert "todo_write" not in regs
    # The rest of the default catalog IS present.
    assert "fs_read" in regs
    assert "git_status" in regs
    assert "web_fetch" in regs


def _bare_server() -> TldwMCPServer:
    """Return a DB-free server with the real gateway registration seam."""
    pytest.importorskip("mcp_unified.gateway")
    from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime

    server = TldwMCPServer.__new__(TldwMCPServer)
    server.mcp = ChatbookGatewayRuntime(
        name="test",
        version="0.1.0",
        tool_descriptors=_describe_local_tools(),
    )
    server._register_tools()
    return server


@pytest.mark.asyncio
async def test_flag_on_registers_local_tool_names(monkeypatch, tmp_path):
    import tldw_chatbook.config as config

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("mcp", "expose_local_tools"):
            return True
        return default

    monkeypatch.setattr(config, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    server = _bare_server()
    server._register_local_agent_tools()
    server.mcp.finalize()

    names = {
        descriptor["name"] for descriptor in await server.mcp.list_tools(_context())
    }
    assert "fs_read" in names
    assert "fs_write" in names
    assert "git_status" in names
    assert "web_fetch" in names
    assert "todo_write" not in names


@pytest.mark.asyncio
async def test_flag_off_registers_nothing(monkeypatch, tmp_path):
    import tldw_chatbook.config as config

    monkeypatch.setattr(
        config, "get_cli_setting", lambda section, key, default=None: default
    )
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    server = _bare_server()
    server._register_local_agent_tools()
    server.mcp.finalize()

    assert [
        descriptor["name"] for descriptor in await server.mcp.list_tools(_context())
    ] == BUILTIN_TOOL_NAMES


@pytest.mark.parametrize("failure_stage", ["construction", "staging"])
@pytest.mark.asyncio
async def test_registration_failure_keeps_builtins_and_emits_fixed_stderr_once(
    failure_stage, monkeypatch, tmp_path, capsys
):
    import tldw_chatbook.config as config
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    sentinel = "SENTINEL /private/path API_KEY=secret"

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("mcp", "expose_local_tools"):
            return True
        return default

    monkeypatch.setattr(config, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError(sentinel)

    if failure_stage == "construction":
        monkeypatch.setattr(local_server_tools, "build_server_local_provider", boom)
    else:
        monkeypatch.setattr(
            local_server_tools,
            "_local_agent_tool_registrations",
            lambda _provider: [
                LocalToolRegistration(
                    "valid_first",
                    "valid",
                    {"type": "object", "properties": {}},
                    lambda _arguments: ToolResult(ok=True, content="ok"),
                ),
                LocalToolRegistration(
                    "invalid_second",
                    "invalid",
                    {"type": "array"},
                    lambda _arguments: ToolResult(ok=True, content="ok"),
                ),
            ],
        )

    server = _bare_server()
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        server._register_local_agent_tools()
    finally:
        logger.remove(sink_id)
    server.mcp.finalize()

    captured = capsys.readouterr()
    assert all("Local registration failed" not in record for record in records)
    assert all(sentinel not in record for record in records)
    assert captured.out == ""
    assert captured.err == (
        "Local MCP tools unavailable; continuing with built-in tools.\n"
    )
    assert sentinel not in captured.err
    assert [
        descriptor["name"] for descriptor in await server.mcp.list_tools(_context())
    ] == BUILTIN_TOOL_NAMES


def test_fastmcp_parameter_summary_workaround_is_removed() -> None:
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    assert not hasattr(local_server_tools, "_parameter_summary")
