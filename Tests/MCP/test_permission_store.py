"""Tests for the schema-versioned MCP permission store (Phase 4, Task 1).

Covers: fresh-default payload shape, corrupt/unknown-version backup-and-reset
policy, kill switch + global default round-trips, server/tool state
set/inherit-prune semantics, the allow-requires-hash guard, the
config_changed clearing/marking contract, and the atomic-write pattern.
"""
from __future__ import annotations

import json

import pytest

from tldw_chatbook.MCP.permission_store import (
    DEFAULT_GLOBAL,
    SCHEMA_VERSION,
    STORE_STATES,
    MCPPermissionStore,
)


def _fresh_payload_shape() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kill_switch": False,
        "profiles": {
            "default": {
                "global_default": DEFAULT_GLOBAL,
                "servers": {},
            }
        },
    }


def test_load_returns_fresh_default_payload_when_file_missing(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    payload = store.load()

    assert payload["schema_version"] == 1
    assert payload["kill_switch"] is False
    assert payload["profiles"]["default"]["global_default"] == "ask"
    assert payload["profiles"]["default"]["servers"] == {}
    assert not (tmp_path / "mcp_permissions.json").exists()


def test_load_backs_up_corrupt_json_and_returns_fresh_default(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    path.write_text("{not valid json", encoding="utf-8")

    store = MCPPermissionStore(path)
    payload = store.load()

    assert payload == _fresh_payload_shape()
    backup_path = tmp_path / "mcp_permissions.json.bak"
    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == "{not valid json"
    assert not path.exists()


def test_load_backs_up_non_dict_json_and_returns_fresh_default(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    store = MCPPermissionStore(path)
    payload = store.load()

    assert payload == _fresh_payload_shape()
    assert (tmp_path / "mcp_permissions.json.bak").exists()
    assert not path.exists()


def test_load_backs_up_unknown_schema_version_and_returns_fresh_default(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    path.write_text(json.dumps({"schema_version": 2, "kill_switch": True}), encoding="utf-8")

    store = MCPPermissionStore(path)
    payload = store.load()

    assert payload == _fresh_payload_shape()
    backup_path = tmp_path / "mcp_permissions.json.bak"
    assert backup_path.exists()
    assert json.loads(backup_path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert not path.exists()


def test_load_backup_replaces_prior_bak_file(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    backup_path = tmp_path / "mcp_permissions.json.bak"
    backup_path.write_text("stale backup contents", encoding="utf-8")
    path.write_text("still not json", encoding="utf-8")

    store = MCPPermissionStore(path)
    store.load()

    assert backup_path.read_text(encoding="utf-8") == "still not json"


def test_kill_switch_round_trip(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    assert store.get_kill_switch() is False

    store.set_kill_switch(True)

    assert store.get_kill_switch() is True
    on_disk = json.loads((tmp_path / "mcp_permissions.json").read_text(encoding="utf-8"))
    assert on_disk["kill_switch"] is True


def test_global_default_validates_and_round_trips(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    assert store.get_global_default() == "ask"

    store.set_global_default("deny")

    assert store.get_global_default() == "deny"

    with pytest.raises(ValueError):
        store.set_global_default("nonsense")


def test_set_server_default_and_inherit_prunes_entry(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    assert store.get_server_entry(server_key) is None

    store.set_server_default(server_key, "allow")

    entry = store.get_server_entry(server_key)
    assert entry is not None
    assert entry["default"] == "allow"

    store.set_server_default(server_key, None)

    assert store.get_server_entry(server_key) is None


def test_set_tool_state_and_inherit_prunes_entry(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    assert store.get_tool_entry(server_key, "search") is None

    store.set_tool_state(server_key, "search", "ask")

    tool_entry = store.get_tool_entry(server_key, "search")
    assert tool_entry is not None
    assert tool_entry["state"] == "ask"

    store.set_tool_state(server_key, "search", None)

    assert store.get_tool_entry(server_key, "search") is None
    # Pruning the sole tool must also prune the now-empty server entry.
    assert store.get_server_entry(server_key) is None


def test_set_tool_state_inherit_prunes_tool_but_keeps_server_default(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    store.set_server_default(server_key, "ask")
    store.set_tool_state(server_key, "search", "allow", definition_hash="hash-1")

    store.set_tool_state(server_key, "search", None)

    assert store.get_tool_entry(server_key, "search") is None
    entry = store.get_server_entry(server_key)
    assert entry is not None
    assert entry["default"] == "ask"


def test_set_tool_state_allow_without_hash_raises_value_error(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    with pytest.raises(ValueError):
        store.set_tool_state("local:demo-server", "search", "allow")


def test_set_tool_state_allow_stores_hash_and_clears_config_changed(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    store.set_tool_state(server_key, "search", "ask")
    assert store.mark_config_changed(server_key, "search") is True
    tool_entry = store.get_tool_entry(server_key, "search")
    assert tool_entry.get("config_changed") is True

    store.set_tool_state(server_key, "search", "allow", definition_hash="abc123")

    tool_entry = store.get_tool_entry(server_key, "search")
    assert tool_entry["state"] == "allow"
    assert tool_entry["definition_hash"] == "abc123"
    assert "config_changed" not in tool_entry


def test_mark_config_changed_returns_true_then_false(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"
    store.set_tool_state(server_key, "search", "allow", definition_hash="abc123")

    first = store.mark_config_changed(server_key, "search")
    second = store.mark_config_changed(server_key, "search")

    assert first is True
    assert second is False


def test_save_atomic_write_leaves_no_tmp_behind(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    store = MCPPermissionStore(path)

    store.set_kill_switch(True)

    assert path.exists()
    assert not (tmp_path / "mcp_permissions.json.tmp").exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "updated_at" in on_disk
    assert on_disk["schema_version"] == 1


def test_store_states_and_default_global_constants():
    assert STORE_STATES == ("allow", "ask", "deny")
    assert DEFAULT_GLOBAL == "ask"
    assert SCHEMA_VERSION == 1
