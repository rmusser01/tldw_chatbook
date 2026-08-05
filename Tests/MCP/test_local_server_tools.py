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
    build_server_local_provider,
)
from tldw_chatbook.MCP.permission_store import MCPPermissionStore, definition_hash


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
