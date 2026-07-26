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
    path.write_text(
        json.dumps({"schema_version": 2, "kill_switch": True}), encoding="utf-8"
    )

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
    on_disk = json.loads(
        (tmp_path / "mcp_permissions.json").read_text(encoding="utf-8")
    )
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


def test_mark_config_changed_second_call_does_not_rewrite_file(tmp_path):
    """Perf: the control plane calls `mark_config_changed()` on every
    resolution pass for every tool that already carries the marker (any
    ⚠ tool in the catalog) -- the second-and-later calls must be a no-op
    read, not a load()+save() that rewrites the store file every time."""
    path = tmp_path / "mcp_permissions.json"
    store = MCPPermissionStore(path)
    server_key = "local:demo-server"
    store.set_tool_state(server_key, "search", "allow", definition_hash="abc123")

    assert store.mark_config_changed(server_key, "search") is True
    before_bytes = path.read_bytes()
    before_mtime_ns = path.stat().st_mtime_ns

    assert store.mark_config_changed(server_key, "search") is False

    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime_ns


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


# -- Task 2: GatedToolRef + resolve_builtin_state -----------------------------

from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    GatedToolRef,
    resolve_builtin_state,
)

BUILTIN_KEY = "agent:builtin"


def _ref(name="calculator", tags=()):
    return GatedToolRef(
        server_key=BUILTIN_KEY,
        name=name,
        description="d",
        input_schema={"type": "object"},
        tags=tuple(tags),
    )


def _payload(*, global_default=None, server_default=None, tool_state=None):
    server: dict = {}
    if server_default is not None:
        server["default"] = server_default
    if tool_state is not None:
        server["tools"] = {"calculator": {"state": tool_state}}
    profile: dict = {"servers": {BUILTIN_KEY: server} if server else {}}
    if global_default is not None:
        profile["global_default"] = global_default
    return {"profiles": {"default": profile}}


def test_builtin_floor_is_allow_not_the_mcp_global_default():
    # MCP's global default is "ask"; built-ins must NOT inherit it, or
    # calculator would prompt on every use.
    eff = resolve_builtin_state(_payload(global_default="ask"), _ref())
    assert eff.state == "allow"
    assert eff.risk_floored is False


def test_empty_payload_resolves_to_allow_floor():
    eff = resolve_builtin_state({}, _ref())
    assert eff.state == "allow"


def test_high_risk_tag_floors_inherited_allow_to_ask():
    eff = resolve_builtin_state({}, _ref(tags=("mutates",)))
    assert eff.state == "ask"
    assert eff.risk_floored is True


def test_explicit_tool_override_allow_is_not_floored():
    eff = resolve_builtin_state(
        _payload(tool_state="allow"), _ref(tags=("mutates",))
    )
    assert eff.state == "allow"
    assert eff.origin == "tool_override"
    assert eff.risk_floored is False


def test_server_default_beats_the_builtin_floor():
    eff = resolve_builtin_state(_payload(server_default="deny"), _ref())
    assert eff.state == "deny"
    assert eff.origin == "server_default"


def test_no_hash_comparison_for_builtins():
    # An allow override with a STALE/absent definition_hash must stay
    # allow -- the rug-pull guard is deliberately not applied to
    # in-process code (it would re-prompt on every app upgrade).
    payload = _payload(tool_state="allow")
    entry = payload["profiles"]["default"]["servers"][BUILTIN_KEY]["tools"]
    entry["calculator"]["definition_hash"] = "stale-and-wrong"
    eff = resolve_builtin_state(payload, _ref())
    assert eff.state == "allow"
    assert eff.config_changed is False


def test_builtin_and_mcp_builtin_server_namespaces_are_disjoint():
    """A decision for the built-in MCP SERVER must not govern the
    agent-runtime tool of the same name, or vice versa."""
    from tldw_chatbook.MCP.readiness import BUILTIN_SERVER_KEY

    assert BUILTIN_SERVER_KEY != BUILTIN_TOOL_SERVER_KEY
    # A deny recorded against the MCP built-in server leaves the
    # agent-runtime tool of the same name on its own floor.
    payload = {
        "profiles": {
            "default": {
                "servers": {
                    BUILTIN_SERVER_KEY: {"tools": {"calculator": {"state": "deny"}}}
                }
            }
        }
    }
    assert resolve_builtin_state(payload, _ref("calculator")).state == "allow"


def test_mcp_resolver_is_unaffected_by_the_builtin_floor():
    # Guard the "do not modify resolve_effective_state" constraint: an
    # MCP tool with no entries still inherits the MCP global default.
    from tldw_chatbook.MCP.permission_store import resolve_effective_state
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        server_key="local:x", server_label="x", source="local",
        name="t", description="d", input_schema=None, tags=(),
        stale=False, executable=True,
    )
    assert resolve_effective_state({}, tool).state == "ask"


# -- Task 1: HASH_FREE_SERVER_KEYS exemption ----------------------------------

from tldw_chatbook.MCP.permission_store import HASH_FREE_SERVER_KEYS


def test_hash_free_keys_contains_exactly_the_builtin_namespace():
    """Pin the CONTENTS. Adding a remote namespace here would silently
    disable the rug-pull guard for it -- the one way this change could
    become a real weakening."""
    assert HASH_FREE_SERVER_KEYS == frozenset({BUILTIN_TOOL_SERVER_KEY})


def test_allow_without_hash_is_permitted_for_the_builtin_namespace(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "allow")

    entry = store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "write_thing")
    assert entry["state"] == "allow"
    assert not entry.get("definition_hash")


def test_allow_without_hash_still_raises_for_an_mcp_server(tmp_path):
    """MCP's guard is unchanged."""
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    with pytest.raises(ValueError, match="definition_hash"):
        store.set_tool_state("local:docs", "some_tool", "allow")


def test_deny_and_clear_need_no_hash_for_either_namespace(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "deny")
    store.set_tool_state("local:docs", "some_tool", "deny")
    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", None)

    assert store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "write_thing") is None
