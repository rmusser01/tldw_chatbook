"""Tests for the control plane's typed permission methods (Phase 4, Task 4).

Covers: the lazy `permission_store` property (path derivation + None
fallback), `effective_tool_states()` (batch resolution, the rug-pull
downgrade audit -- emitted exactly once and only for tools with an explicit
tool-level `allow` entry), the typed state setters (`set_tool_state`,
`set_server_default`, `set_global_default`, kill-switch get/set) and their
no-store no-op fallbacks, and `gate_tool_test()` (single-tool resolution for
the Test Tool button, no audit emission, kill switch deliberately ignored).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_SERVER_KEY,
    LocalToolProvider,
)
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
from tldw_chatbook.MCP.execution_log import MCPExecutionLog
from tldw_chatbook.MCP.hub_test_execution import (
    ToolTestAdmissionBlocked,
    ToolTestAdmissionPreview,
    ToolTestAdmissionStale,
    authority_fingerprint,
)
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    MCPPermissionStore,
    definition_hash,
)
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain


def _tool(
    *,
    server_key: str = "local:demo",
    name: str = "search",
    description: str = "Search docs",
    input_schema: dict | None = None,
    tags: tuple[str, ...] = (),
) -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label="demo",
        source="local",
        name=name,
        description=description,
        input_schema=input_schema,
        tags=tags,
        stale=False,
        executable=True,
    )


def _service(tmp_path: Path) -> tuple[UnifiedMCPControlPlaneService, LocalMCPStore]:
    store = LocalMCPStore(tmp_path / "store.json")
    fake_local_service = SimpleNamespace(store=store)
    service = UnifiedMCPControlPlaneService(
        local_service=fake_local_service,
        server_service=None,
        target_store=None,
        context_store=None,
    )
    return service, store


def _service_without_store() -> UnifiedMCPControlPlaneService:
    # No `.store` attribute at all -- mirrors `getattr(..., "store", None)`.
    fake_local_service = SimpleNamespace()
    return UnifiedMCPControlPlaneService(
        local_service=fake_local_service,
        server_service=None,
        target_store=None,
        context_store=None,
    )


def _permission_log_records(store: LocalMCPStore) -> list[dict]:
    log_path = Path(store.path).with_name("mcp_execution_log.jsonl")
    return MCPExecutionLog(log_path).read_recent()


_TODO_TOOL_NAMES = ("todo_create", "todo_update", "todo_get", "todo_list")


def _todo_hubs(tmp_path: Path) -> dict[str, HubTool]:
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        todo_store=SessionTodoStore(),
    )
    return {name: provider.hub_tool_for(name) for name in _TODO_TOOL_NAMES}


# -- permission_store lazy property ------------------------------------------


def test_permission_store_derives_path_from_local_service_store(tmp_path):
    service, store = _service(tmp_path)

    permission_store = service.permission_store

    assert isinstance(permission_store, MCPPermissionStore)
    assert permission_store.path == Path(store.path).with_name("mcp_permissions.json")


def test_permission_store_is_cached_across_accesses(tmp_path):
    service, _store = _service(tmp_path)

    first = service.permission_store
    second = service.permission_store

    assert first is second


def test_permission_store_is_none_when_local_service_has_no_store():
    service = _service_without_store()

    assert service.permission_store is None


# -- effective_tool_states: no-store fallback --------------------------------


def test_effective_tool_states_no_store_returns_ask_global_default_for_every_tool():
    service = _service_without_store()
    tools = [_tool(name="search"), _tool(name="write", server_key="local:other")]

    result = service.effective_tool_states(tools)

    assert set(result.keys()) == {("local:demo", "search"), ("local:other", "write")}
    for key, effective in result.items():
        assert effective.state == "ask", key
        assert effective.origin == "global_default", key
        assert effective.config_changed is False, key


# -- effective_tool_states: precedence resolution ----------------------------


def test_effective_tool_states_resolves_per_precedence_with_real_store(tmp_path):
    service, store = _service(tmp_path)
    tool_override = _tool(name="search", server_key="local:demo")
    server_default_tool = _tool(name="write", server_key="local:demo")
    global_default_tool = _tool(name="fetch", server_key="local:other")

    current_hash = definition_hash(
        tool_override.description, tool_override.input_schema
    )
    permission_store = MCPPermissionStore(
        Path(store.path).with_name("mcp_permissions.json")
    )
    permission_store.set_global_default("deny")
    permission_store.set_server_default("local:demo", "ask")
    permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash=current_hash
    )

    result = service.effective_tool_states(
        [tool_override, server_default_tool, global_default_tool]
    )

    assert result[("local:demo", "search")].state == "allow"
    assert result[("local:demo", "search")].origin == "tool_override"
    assert result[("local:demo", "write")].state == "ask"
    assert result[("local:demo", "write")].origin == "server_default"
    assert result[("local:other", "fetch")].state == "deny"
    assert result[("local:other", "fetch")].origin == "global_default"


# -- effective_tool_states: rug-pull downgrade audit -------------------------


def test_effective_tool_states_fresh_mismatch_emits_exactly_one_downgraded_record_across_two_calls(
    tmp_path,
):
    service, store = _service(tmp_path)
    original_tool = _tool(name="search", description="Search docs")
    permission_store = service.permission_store
    original_hash = definition_hash(
        original_tool.description, original_tool.input_schema
    )
    permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash=original_hash
    )

    # Simulate a rug-pull: the tool's live definition has changed since the
    # user allowed it, so the stored hash no longer matches.
    changed_tool = _tool(name="search", description="Search docs AND delete them")

    first_result = service.effective_tool_states([changed_tool])
    assert first_result[("local:demo", "search")].state == "ask"
    assert first_result[("local:demo", "search")].config_changed is True

    records = _permission_log_records(store)
    assert len(records) == 1
    record = records[0]
    assert record["server_key"] == "local:demo"
    assert record["tool_name"] == "search"
    assert record["initiator"] == "system"
    assert record["decision"] == "downgraded"
    assert record["ok"] is False
    assert record["duration_ms"] == 0
    assert record["status"] == "blocked"
    assert record["error_category"] == "definition_changed"
    assert "definition changed since" not in repr(record)

    # The marker is now persisted -- a second resolution pass must not
    # append a second audit record.
    second_result = service.effective_tool_states([changed_tool])
    assert second_result[("local:demo", "search")].state == "ask"
    assert second_result[("local:demo", "search")].config_changed is True

    records_after_second_call = _permission_log_records(store)
    assert len(records_after_second_call) == 1


def test_effective_tool_states_no_explicit_entry_never_marks_or_audits(tmp_path):
    """CROSS-TASK INVARIANT (T2 review): `mark_config_changed` uses
    `setdefault` and CAN create a stateless `{"config_changed": true}` entry
    that resolution then silently ignores. A tool with no explicit
    tool-level entry (state inherited from the global default) must never
    trigger a marker or an audit record, no matter what its live definition
    looks like -- there is nothing to "rug-pull" against."""
    service, store = _service(tmp_path)
    permission_store = service.permission_store
    permission_store.set_global_default("allow")
    tool = _tool(name="search", description="Whatever the live definition is today")

    result = service.effective_tool_states([tool])

    assert result[("local:demo", "search")].state == "allow"
    assert result[("local:demo", "search")].origin == "global_default"
    assert result[("local:demo", "search")].config_changed is False
    assert permission_store.get_tool_entry("local:demo", "search") is None
    assert _permission_log_records(store) == []


def test_effective_tool_states_matching_hash_does_not_mark_or_audit(tmp_path):
    service, store = _service(tmp_path)
    tool = _tool(name="search")
    permission_store = service.permission_store
    current_hash = definition_hash(tool.description, tool.input_schema)
    permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash=current_hash
    )

    result = service.effective_tool_states([tool])

    assert result[("local:demo", "search")].state == "allow"
    assert result[("local:demo", "search")].config_changed is False
    entry = permission_store.get_tool_entry("local:demo", "search")
    assert not entry.get("config_changed")
    assert _permission_log_records(store) == []


def test_effective_tool_states_downgrade_audit_survives_execution_log_failure(
    tmp_path, monkeypatch
):
    """Best-effort contract: a failure while appending the audit record
    must not prevent `effective_tool_states()` from returning its result
    (mirrors `_record_tool_execution`'s never-raise contract)."""
    import tldw_chatbook.MCP.unified_control_plane_service as control_plane_module

    class _RaisingExecutionLog(MCPExecutionLog):
        def append(self, record):
            raise OSError("disk full")

    monkeypatch.setattr(control_plane_module, "MCPExecutionLog", _RaisingExecutionLog)
    service, store = _service(tmp_path)
    tool = _tool(name="search")
    permission_store = service.permission_store
    permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash="stale-hash"
    )

    result = service.effective_tool_states([tool])

    assert result[("local:demo", "search")].state == "ask"
    assert result[("local:demo", "search")].config_changed is True


# -- set_tool_state -----------------------------------------------------------


def test_set_tool_state_allow_computes_and_stores_definition_hash_and_clears_marker(
    tmp_path,
):
    service, _store = _service(tmp_path)
    permission_store = service.permission_store
    permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash="stale-hash"
    )
    permission_store.mark_config_changed("local:demo", "search")
    tool = _tool(name="search", description="Search docs")

    service.set_tool_state("local:demo", "search", "allow", tool=tool)

    entry = permission_store.get_tool_entry("local:demo", "search")
    assert entry["state"] == "allow"
    assert entry["definition_hash"] == definition_hash(
        tool.description, tool.input_schema
    )
    assert "config_changed" not in entry


def test_set_tool_state_allow_without_tool_raises_value_error(tmp_path):
    service, _store = _service(tmp_path)

    with pytest.raises(ValueError):
        service.set_tool_state("local:demo", "search", "allow", tool=None)


def test_set_tool_state_ask_does_not_require_tool(tmp_path):
    service, _store = _service(tmp_path)

    service.set_tool_state("local:demo", "search", "ask")

    entry = service.permission_store.get_tool_entry("local:demo", "search")
    assert entry == {"state": "ask"}


def test_set_tool_state_none_clears_entry(tmp_path):
    service, _store = _service(tmp_path)
    permission_store = service.permission_store
    permission_store.set_tool_state("local:demo", "search", "ask")

    service.set_tool_state("local:demo", "search", None)

    assert permission_store.get_tool_entry("local:demo", "search") is None


def test_set_tool_state_no_store_is_a_noop():
    service = _service_without_store()

    service.set_tool_state(
        "local:demo", "search", "allow", tool=_tool()
    )  # must not raise


def test_set_tool_state_allow_for_builtin_namespace_needs_no_tool(tmp_path):
    """`agent:builtin` is hash-free (TASK-627 Task 1): `allow` must succeed
    with no `tool=` argument at all, unlike every MCP `server_key`."""
    service, _store = _service(tmp_path)

    service.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "allow")

    entry = service.permission_store.get_tool_entry(
        BUILTIN_TOOL_SERVER_KEY, "write_thing"
    )
    assert entry["state"] == "allow"


def test_set_tool_state_allow_still_requires_tool_for_mcp_server(tmp_path):
    """The exemption is namespace-scoped: an MCP `server_key` must still
    raise without a `tool=` argument."""
    service, _store = _service(tmp_path)

    with pytest.raises(ValueError):
        service.set_tool_state("local:demo", "search", "allow")


# -- set_server_default / set_global_default / kill switch --------------------


def test_set_server_default_round_trips(tmp_path):
    service, _store = _service(tmp_path)

    service.set_server_default("local:demo", "deny")

    assert service.permission_store.get_server_entry("local:demo")["default"] == "deny"

    service.set_server_default("local:demo", None)

    assert service.permission_store.get_server_entry("local:demo") is None


def test_set_global_default_round_trips(tmp_path):
    service, _store = _service(tmp_path)

    service.set_global_default("deny")

    assert service.permission_store.get_global_default() == "deny"


def test_kill_switch_get_defaults_false_and_set_round_trips(tmp_path):
    service, _store = _service(tmp_path)

    assert service.get_kill_switch() is False

    service.set_kill_switch(True)

    assert service.get_kill_switch() is True


def test_no_store_fallbacks_for_setters_are_noops_and_kill_switch_is_false():
    service = _service_without_store()

    service.set_server_default("local:demo", "deny")  # must not raise
    service.set_global_default("deny")  # must not raise
    service.set_kill_switch(True)  # must not raise

    assert service.get_kill_switch() is False


# -- gate_tool_test -------------------------------------------------------------


def test_gate_tool_test_no_store_returns_ask_global_default():
    service = _service_without_store()

    result = service.gate_tool_test(_tool())

    assert result.state == "ask"
    assert result.origin == "global_default"


@pytest.mark.parametrize(
    "stored_state,expected_state",
    [("deny", "deny"), ("ask", "ask"), ("allow", "allow")],
)
def test_gate_tool_test_returns_state_per_store(tmp_path, stored_state, expected_state):
    service, _store = _service(tmp_path)
    tool = _tool(name="search")
    kwargs = {}
    if stored_state == "allow":
        kwargs["definition_hash"] = definition_hash(tool.description, tool.input_schema)
    service.permission_store.set_tool_state(
        "local:demo", "search", stored_state, **kwargs
    )

    result = service.gate_tool_test(tool)

    assert result.state == expected_state


def test_gate_tool_test_ignores_kill_switch(tmp_path):
    service, _store = _service(tmp_path)
    tool = _tool(name="search")
    service.permission_store.set_tool_state(
        "local:demo",
        "search",
        "allow",
        definition_hash=definition_hash(tool.description, tool.input_schema),
    )
    service.permission_store.set_kill_switch(True)

    result = service.gate_tool_test(tool)

    assert result.state == "allow"


def test_gate_tool_test_does_not_emit_audit_record_on_fresh_mismatch(tmp_path):
    service, store = _service(tmp_path)
    tool = _tool(name="search")
    service.permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash="stale-hash"
    )

    result = service.gate_tool_test(tool)

    assert result.state == "ask"
    assert result.config_changed is True
    assert _permission_log_records(store) == []
    entry = service.permission_store.get_tool_entry("local:demo", "search")
    assert not entry.get("config_changed")  # gate must not persist the marker either


# -- I1: gate_tool_test_by_key (no live HubTool) -----------------------------
#
# `MCPWorkbench._resolve_test_gate()`'s fallback for a tool that dropped out
# of the catalog snapshot (`_tool_for()` returned None): no `HubTool` is
# available to hash-compare, so this resolves deny/ask straight through and
# downgrades any "allow" verdict to "ask" (see
# `resolve_effective_state_by_key`'s own docstring for the full rationale).


def test_gate_tool_test_by_key_no_store_returns_ask_global_default():
    service = _service_without_store()

    result = service.gate_tool_test_by_key("local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_gate_tool_test_by_key_deny_passes_through(tmp_path):
    service, _store = _service(tmp_path)
    service.permission_store.set_tool_state("local:demo", "search", "deny")

    result = service.gate_tool_test_by_key("local:demo", "search")

    assert result.state == "deny"


def test_gate_tool_test_by_key_ask_passes_through(tmp_path):
    service, _store = _service(tmp_path)
    service.permission_store.set_tool_state("local:demo", "search", "ask")

    result = service.gate_tool_test_by_key("local:demo", "search")

    assert result.state == "ask"


def test_gate_tool_test_by_key_allow_downgrades_to_ask_without_live_tool(tmp_path):
    """The core I1 fix: an explicit "allow" resolved WITHOUT a live
    `HubTool` to hash-check must never be trusted as-is -- this is what
    lets the gate say "ask"/"deny" for a vanished tool instead of `None`
    (which used to mean "run immediately, ungated")."""
    service, _store = _service(tmp_path)
    tool = _tool(name="search")
    service.permission_store.set_tool_state(
        "local:demo",
        "search",
        "allow",
        definition_hash=definition_hash(tool.description, tool.input_schema),
    )

    result = service.gate_tool_test_by_key("local:demo", "search")

    assert result.state == "ask"
    assert result.config_changed is True


def test_gate_tool_test_by_key_does_not_emit_audit_record(tmp_path):
    service, store = _service(tmp_path)
    service.permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash="stale-hash"
    )

    service.gate_tool_test_by_key("local:demo", "search")

    assert _permission_log_records(store) == []


# -- task-2838: local agent tools resolve through the SAME shared store -------


def test_local_agent_tool_allow_round_trips_between_console_and_hub(tmp_path):
    """A Console "Always allow" write resolves identically hub-side.

    Both surfaces talk to the same UnifiedMCPControlPlaneService methods
    (console_chat_controller._compose_local_provider and the Hub workbench
    permission cycle) against the same derived mcp_permissions.json, so a
    grant made on either side must be honored by the other's resolution.
    """
    service, _store = _service(tmp_path)
    provider = LocalToolProvider(workspace_root=tmp_path)
    hub = provider.hub_tool_for("fs_read")

    # The Console-side write shape (its _persist_approval always_allow path).
    service.set_tool_state(hub.server_key, hub.name, "allow", tool=hub)

    # The Hub-side batch resolution (Tools State column / Permissions matrix).
    result = service.effective_tool_states([hub])
    assert result[(hub.server_key, hub.name)].state == "allow"
    assert result[(hub.server_key, hub.name)].origin == "tool_override"

    # Persisted under the synthetic local server key WITH the rug-pull hash.
    payload = service.permission_store.load()
    entry = payload["profiles"]["default"]["servers"]["local:__local__"]["tools"][
        "fs_read"
    ]
    assert entry["state"] == "allow"
    assert entry["definition_hash"] == definition_hash(
        hub.description, hub.input_schema
    )


def test_local_agent_server_default_allow_is_risk_floored_for_mutates_tools(
    tmp_path,
):
    """Spec §3.2: an INHERITED allow floors to ask for `mutates`-tagged tools.

    Setting the `local:__local__` server default to "allow" must NOT wave
    fs_write/fs_edit/fs_patch through -- only an explicit tool-level
    "Always allow" escapes the floor.
    """
    service, _store = _service(tmp_path)
    provider = LocalToolProvider(workspace_root=tmp_path)
    write_hub = provider.hub_tool_for("fs_write")
    read_hub = provider.hub_tool_for("fs_read")

    service.set_server_default("local:__local__", "allow")

    result = service.effective_tool_states([write_hub, read_hub])
    floored = result[("local:__local__", "fs_write")]
    assert floored.state == "ask"
    assert floored.risk_floored is True
    assert result[("local:__local__", "fs_read")].state == "allow"

    # An explicit tool-level allow is never floored (spec §3.2).
    service.set_tool_state(
        write_hub.server_key, write_hub.name, "allow", tool=write_hub
    )
    result = service.effective_tool_states([write_hub])
    explicit = result[("local:__local__", "fs_write")]
    assert explicit.state == "allow"
    assert explicit.risk_floored is False


def test_obsolete_todo_write_allow_does_not_authorize_replacement_tools(tmp_path):
    """The retired broad grant is inert; only inherited read tools stay allowed."""
    service, _store = _service(tmp_path)
    hubs = _todo_hubs(tmp_path)
    legacy = _tool(
        server_key=LOCAL_SERVER_KEY,
        name="todo_write",
        description="Replace the complete session todo list.",
        input_schema={"type": "object", "properties": {"todos": {"type": "array"}}},
        tags=("mutates",),
    )
    service.set_tool_state(
        legacy.server_key,
        legacy.name,
        "allow",
        tool=legacy,
    )

    inherited_ask = service.effective_tool_states(list(hubs.values()))
    for name in _TODO_TOOL_NAMES:
        state = inherited_ask[(LOCAL_SERVER_KEY, name)]
        assert state.state == "ask"
        assert state.origin == "global_default"
        assert state.config_changed is False
        assert state.risk_floored is False

    payload = service.permission_store.load()
    stored_tools = payload["profiles"]["default"]["servers"][LOCAL_SERVER_KEY]["tools"]
    assert set(stored_tools) == {"todo_write"}

    service.set_server_default(LOCAL_SERVER_KEY, "allow")
    inherited_allow = service.effective_tool_states(list(hubs.values()))
    for name in ("todo_create", "todo_update"):
        state = inherited_allow[(LOCAL_SERVER_KEY, name)]
        assert state.state == "ask"
        assert state.origin == "server_default"
        assert state.risk_floored is True
    for name in ("todo_get", "todo_list"):
        state = inherited_allow[(LOCAL_SERVER_KEY, name)]
        assert state.state == "allow"
        assert state.origin == "server_default"
        assert state.risk_floored is False


@pytest.mark.parametrize("tool_name", _TODO_TOOL_NAMES)
def test_todo_replacement_tool_allow_requires_current_definition_hash(
    tmp_path, tool_name
):
    """Each replacement grant is bound to that tool's live definition."""
    service, _store = _service(tmp_path)
    hub = _todo_hubs(tmp_path)[tool_name]
    legacy = _tool(
        server_key=LOCAL_SERVER_KEY,
        name="todo_write",
        description="Replace the complete session todo list.",
        input_schema={"type": "object", "properties": {"todos": {"type": "array"}}},
        tags=("mutates",),
    )
    service.set_tool_state(
        legacy.server_key,
        legacy.name,
        "allow",
        tool=legacy,
    )
    legacy_entry = service.permission_store.get_tool_entry(
        LOCAL_SERVER_KEY, "todo_write"
    )
    legacy_hash = legacy_entry["definition_hash"]

    service.permission_store.set_tool_state(
        LOCAL_SERVER_KEY,
        tool_name,
        "allow",
        definition_hash=legacy_hash,
    )
    stale = service.effective_tool_states([hub])[(LOCAL_SERVER_KEY, tool_name)]
    assert stale.state == "ask"
    assert stale.origin == "tool_override"
    assert stale.config_changed is True

    service.set_tool_state(LOCAL_SERVER_KEY, tool_name, "allow", tool=hub)
    current_hash = definition_hash(hub.description, hub.input_schema)
    current_entry = service.permission_store.get_tool_entry(LOCAL_SERVER_KEY, tool_name)
    assert current_entry == {
        "state": "allow",
        "definition_hash": current_hash,
    }
    assert current_hash != legacy_hash
    fresh = service.effective_tool_states([hub])[(LOCAL_SERVER_KEY, tool_name)]
    assert fresh.state == "allow"
    assert fresh.origin == "tool_override"
    assert fresh.config_changed is False
    assert fresh.risk_floored is False


# -- workspace assistant defaults (Task 6): named-profile resolution ---------
#
# Task 5 gave the store named permission profiles (profile-major chains:
# the named profile's tool/server/global levels settle before the default
# profile's). Task 6 threads `profile_id` through this service funnel and
# adds the `gate_tool_test_for_profile` alias Task 7's Console closure
# consumes. Every keyword defaults to `"default"` -- byte-identical to the
# single-profile behavior the tests above pin.


def test_gate_tool_test_for_profile_respects_named_profile(tmp_path):
    """The Console's per-workspace gate seam: a deny recorded only in the
    named profile is visible through `gate_tool_test_for_profile` while
    the default-profile `gate_tool_test` call is unchanged."""
    service, _store = _service(tmp_path)
    store = service.permission_store
    store.ensure_profile("ws-w-1")
    store.set_tool_state("local:__local__", "fs_write", "deny", profile_id="ws-w-1")
    hub = _tool(server_key="local:__local__", name="fs_write")

    assert service.gate_tool_test_for_profile(hub, "ws-w-1").state == "deny"
    assert service.gate_tool_test(hub).state != "deny"


def test_effective_tool_states_named_profile_shadows_and_inherits(tmp_path):
    """Batch resolution threads `profile_id`: the named profile's tool
    override shadows the default profile's, and tools the named profile
    leaves unset inherit from the default profile (profile-major chain)."""
    service, _store = _service(tmp_path)
    store = service.permission_store
    shadowed = _tool(name="search", server_key="local:demo")
    inherited = _tool(name="fetch", server_key="local:other")
    store.ensure_profile("ws-w-1")
    store.set_tool_state("local:demo", "search", "deny", profile_id="ws-w-1")
    store.set_tool_state(
        "local:demo",
        "search",
        "allow",
        definition_hash=definition_hash(
            shadowed.description, shadowed.input_schema
        ),
    )

    named = service.effective_tool_states([shadowed, inherited], profile_id="ws-w-1")
    assert named[("local:demo", "search")].state == "deny"
    assert named[("local:demo", "search")].origin == "tool_override"
    # Nothing in the named profile for this tool: inherit from default.
    assert named[("local:other", "fetch")].state == "ask"
    assert named[("local:other", "fetch")].origin == "global_default"

    # The default-profile call is untouched by the named profile's data.
    default = service.effective_tool_states([shadowed, inherited])
    assert default[("local:demo", "search")].state == "allow"


def test_effective_tool_states_named_profile_rug_pull_marks_named_entry(tmp_path):
    """The downgrade audit writes its `config_changed` marker into the
    profile the resolution ran under, not the default profile."""
    service, _store = _service(tmp_path)
    store = service.permission_store
    store.ensure_profile("ws-w-1")
    store.set_tool_state(
        "local:demo", "search", "allow", profile_id="ws-w-1", definition_hash="stale"
    )
    changed_tool = _tool(name="search", description="Search docs AND delete them")

    result = service.effective_tool_states([changed_tool], profile_id="ws-w-1")

    assert result[("local:demo", "search")].state == "ask"
    assert result[("local:demo", "search")].config_changed is True
    payload = store.load()
    named_entry = payload["profiles"]["ws-w-1"]["servers"]["local:demo"]["tools"][
        "search"
    ]
    assert named_entry.get("config_changed") is True
    assert "local:demo" not in payload["profiles"]["default"]["servers"]


def test_set_tool_state_allow_hashes_under_named_profile(tmp_path):
    """The service's own hash computation (for `allow` writes) follows the
    `profile_id` too: the named profile's entry carries the definition
    hash and resolves fresh-allow without a rug-pull downgrade."""
    service, _store = _service(tmp_path)
    service.permission_store.ensure_profile("ws-w-1")
    tool = _tool(name="search")

    service.set_tool_state(
        "local:demo", "search", "allow", tool=tool, profile_id="ws-w-1"
    )

    entry = service.permission_store.load()["profiles"]["ws-w-1"]["servers"][
        "local:demo"
    ]["tools"]["search"]
    assert entry == {
        "state": "allow",
        "definition_hash": definition_hash(tool.description, tool.input_schema),
    }
    assert service.gate_tool_test(tool, profile_id="ws-w-1").state == "allow"


def test_set_server_and_global_defaults_write_to_named_profile(tmp_path):
    """`set_server_default`/`set_global_default` thread `profile_id` to the
    store: the named profile receives the write, the default profile does
    not."""
    service, _store = _service(tmp_path)
    store = service.permission_store
    store.ensure_profile("ws-w-1")

    service.set_server_default("local:other", "deny", profile_id="ws-w-1")
    service.set_global_default("ask", profile_id="ws-w-1")

    payload = store.load()
    named = payload["profiles"]["ws-w-1"]
    assert named["servers"]["local:other"]["default"] == "deny"
    assert named["global_default"] == "ask"
    assert payload["profiles"]["default"]["servers"] == {}


# -- immutable Hub Test Tool admission --------------------------------------


def test_admission_preview_resolves_live_exact_definition_gate_and_authority(
    tmp_path, monkeypatch
):
    import tldw_chatbook.MCP.local_server_tools as local_server_tools
    import tldw_chatbook.MCP.unified_control_plane_service as service_module

    service, _store = _service(tmp_path)
    rendered = _tool(
        server_key=LOCAL_SERVER_KEY,
        name="fs_read",
        description="stale panel definition",
    )
    live = _tool(
        server_key=LOCAL_SERVER_KEY,
        name="fs_read",
        description="current provider definition",
    )
    authority = capture_directory_chain(tmp_path)

    class _Provider:
        def hub_tools(self):
            return [live]

    class _Handle:
        provider = _Provider()

        def __init__(self):
            self.authority = authority

        def close(self):
            return None

    monkeypatch.setattr(
        local_server_tools,
        "build_hub_local_inspection_provider",
        lambda *a, **k: _Handle(),
    )
    monkeypatch.setattr(
        local_server_tools, "build_hub_local_provider", lambda *a, **k: _Handle()
    )
    monkeypatch.setattr(
        local_server_tools, "resolve_server_workspace_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        service_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            True if (section, key) == ("console", "local_tools_enabled") else default
        ),
    )
    service.set_tool_state(LOCAL_SERVER_KEY, "fs_read", "allow", tool=live)

    preview = service.prepare_hub_test(rendered)

    assert isinstance(preview, ToolTestAdmissionPreview)
    assert (preview.server_key, preview.tool_name) == (LOCAL_SERVER_KEY, "fs_read")
    assert preview.definition_hash == definition_hash(
        live.description, live.input_schema
    )
    assert preview.rendered_gate == "allow"
    assert preview.authority_fingerprint == authority_fingerprint(authority)
    assert preview.safe_authority_label == "Selected workspace"
    assert str(tmp_path) not in repr(preview)


@pytest.mark.asyncio
async def test_preview_nonce_revoke_and_reuse_return_typed_stale_outcomes(tmp_path):
    service, _store = _service(tmp_path)
    tool = _tool()
    service.local_service.get_external_servers = lambda: [
        {
            "profile_id": "demo",
            "is_connected": True,
            "discovery_snapshot": {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                    }
                ]
            },
        }
    ]
    preview = service.prepare_hub_test(tool)
    service.revoke_hub_test_preview(preview.nonce)

    revoked = await service.execute_prepared_hub_test(preview.nonce, "run", {})
    reused = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(revoked, ToolTestAdmissionStale)
    assert revoked.reason == "preview_unavailable"
    assert isinstance(reused, ToolTestAdmissionStale)
    assert service.local_service.get_external_servers()  # low-level seams untouched


@pytest.mark.asyncio
async def test_rendered_allow_requires_run_and_fresh_allow(tmp_path):
    service, _store = _service(tmp_path)
    tool = _tool()
    service.local_service.get_external_servers = lambda: [
        {
            "profile_id": "demo",
            "is_connected": True,
            "discovery_snapshot": {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                    }
                ]
            },
        }
    ]
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    service.test_hub_tool = AsyncMock(return_value={"ok": True})

    wrong_intent_preview = service.prepare_hub_test(tool)
    wrong_intent = await service.execute_prepared_hub_test(
        wrong_intent_preview.nonce, "approve_once", {}
    )
    assert isinstance(wrong_intent, ToolTestAdmissionBlocked)
    assert wrong_intent.reason == "intent_mismatch"
    service.test_hub_tool.assert_not_awaited()

    changed_preview = service.prepare_hub_test(tool)
    service.set_tool_state(tool.server_key, tool.name, "ask")
    changed = await service.execute_prepared_hub_test(changed_preview.nonce, "run", {})
    assert isinstance(changed, ToolTestAdmissionStale)
    assert changed.reason == "gate_changed"
    assert changed.refreshed_preview is not None
    assert changed.refreshed_preview.rendered_gate == "ask"
    service.test_hub_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_rendered_ask_approve_once_accepts_fresh_ask_or_allow_without_persisting(
    tmp_path,
):
    service, _store = _service(tmp_path)
    tool = _tool()
    service.local_service.get_external_servers = lambda: [
        {
            "profile_id": "demo",
            "is_connected": True,
            "discovery_snapshot": {
                "tools": [{"name": tool.name, "description": tool.description}]
            },
        }
    ]
    service.test_hub_tool = AsyncMock(side_effect=[{"ask": True}, {"allow": True}])

    ask_preview = service.prepare_hub_test(tool)
    ask_result = await service.execute_prepared_hub_test(
        ask_preview.nonce, "approve_once", {"b": 2, "a": 1}
    )
    assert ask_result == {"ask": True}
    assert service.permission_store.get_tool_entry(tool.server_key, tool.name) is None

    allow_preview = service.prepare_hub_test(tool)
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    allow_result = await service.execute_prepared_hub_test(
        allow_preview.nonce, "approve_once", {"x": True}
    )
    assert allow_result == {"allow": True}
    assert service.test_hub_tool.await_args_list[0].kwargs["decision"] == "approved"
    assert service.test_hub_tool.await_args_list[1].kwargs["decision"] == "allowed"


@pytest.mark.asyncio
async def test_definition_change_downgrades_stored_allow_without_dispatch(tmp_path):
    service, _store = _service(tmp_path)
    live = {"description": "original"}
    tool = _tool(description=live["description"])

    def _catalog():
        return [
            {
                "profile_id": "demo",
                "is_connected": True,
                "discovery_snapshot": {
                    "tools": [{"name": tool.name, "description": live["description"]}]
                },
            }
        ]

    service.local_service.get_external_servers = _catalog
    service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    service.test_hub_tool = AsyncMock(return_value={"should": "not run"})
    preview = service.prepare_hub_test(tool)
    live["description"] = "changed after render"

    result = await service.execute_prepared_hub_test(preview.nonce, "run", {})

    assert isinstance(result, ToolTestAdmissionStale)
    assert result.reason == "definition_changed"
    assert result.refreshed_preview is not None
    assert result.refreshed_preview.rendered_gate == "ask"
    service.test_hub_tool.assert_not_awaited()
