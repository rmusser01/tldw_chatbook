"""Tests for MCP/local_server_tools.py — external-client local tool composition.

The composition builds a LocalToolProvider for non-Console MCP serving:
permission state resolved FRESH per call from a real MCPPermissionStore,
no approval callback (ask fails closed with the external refusal copy),
kill switch honored, and no Console-only seams (todo store, session
approvals, persistence).
"""

import pytest

from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
)
from tldw_chatbook.MCP.local_server_tools import (
    EXTERNAL_NO_CALLBACK_REFUSAL,
    _local_agent_tool_registrations,
    build_server_local_provider,
)
from tldw_chatbook.MCP.permission_store import MCPPermissionStore, definition_hash
from tldw_chatbook.MCP.server import MCP_AVAILABLE


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
    class RaisingStore:
        def load(self):
            return {}

        def get_kill_switch(self):
            raise RuntimeError("disk gone")

    provider = build_server_local_provider(workspace, RaisingStore())
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL


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
    assert "hello world" in reg.handler({"path": "hello.txt"})


def test_ask_state_handler_returns_error_dict(workspace, store):
    # fs_write with nothing granted -> ask default -> external refusal,
    # shaped per the server.py {"error": str} convention.
    provider = build_server_local_provider(workspace, store)
    result = _registrations(provider)["fs_write"].handler(
        {"path": "x.txt", "content": "y"}
    )
    assert result == {"error": EXTERNAL_NO_CALLBACK_REFUSAL}


def test_deny_state_handler_returns_error_dict(workspace, store):
    provider = build_server_local_provider(workspace, store)
    hub = provider.hub_tool_for("fs_glob")
    store.set_tool_state(hub.server_key, hub.name, "deny")
    result = _registrations(provider)["fs_glob"].handler({"pattern": "*.py"})
    assert result == {"error": LOCAL_DENY_REFUSAL}


def test_kill_switch_handler_returns_error_dict(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    store.set_kill_switch(True)
    result = _registrations(provider)["fs_read"].handler({"path": "hello.txt"})
    assert result == {"error": LOCAL_KILL_SWITCH_REFUSAL}


def test_todo_write_absent_from_registrations(workspace, store):
    provider = build_server_local_provider(workspace, store)
    regs = _registrations(provider)
    assert "todo_write" not in regs
    # The rest of the default catalog IS present.
    assert "fs_read" in regs
    assert "git_status" in regs
    assert "web_fetch" in regs


# -- FastMCP binding layer (skip when the mcp package is unavailable) --------


def _fastmcp_tool_names(mcp):
    return {t.name for t in mcp._tool_manager.list_tools()}


@pytest.fixture
def bare_server():
    """A TldwMCPServer with only its FastMCP instance (no DB init)."""
    from mcp.server.fastmcp import FastMCP

    from tldw_chatbook.MCP.server import TldwMCPServer

    server = TldwMCPServer.__new__(TldwMCPServer)
    server.mcp = FastMCP("test")
    return server


@pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp package not installed")
def test_flag_on_registers_local_tool_names(bare_server, monkeypatch, tmp_path):
    import tldw_chatbook.config as config

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("mcp", "expose_local_tools"):
            return True
        return default

    monkeypatch.setattr(config, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    bare_server._register_local_agent_tools()

    names = _fastmcp_tool_names(bare_server.mcp)
    assert "fs_read" in names
    assert "fs_write" in names
    assert "git_status" in names
    assert "web_fetch" in names
    assert "todo_write" not in names


@pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp package not installed")
def test_flag_off_registers_nothing(bare_server, monkeypatch, tmp_path):
    import tldw_chatbook.config as config

    @bare_server.mcp.tool()
    async def existing_tool() -> str:
        """Pre-existing server tool."""
        return "ok"

    monkeypatch.setattr(
        config, "get_cli_setting", lambda section, key, default=None: default
    )
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    bare_server._register_local_agent_tools()

    # Default flag (False) -> no-op; pre-existing tools untouched.
    assert _fastmcp_tool_names(bare_server.mcp) == {"existing_tool"}
