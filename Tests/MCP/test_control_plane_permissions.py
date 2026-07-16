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

import pytest

from tldw_chatbook.MCP.execution_log import MCPExecutionLog
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.permission_store import MCPPermissionStore, definition_hash
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService


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
        local_service=fake_local_service, server_service=None, target_store=None, context_store=None
    )
    return service, store


def _service_without_store() -> UnifiedMCPControlPlaneService:
    # No `.store` attribute at all -- mirrors `getattr(..., "store", None)`.
    fake_local_service = SimpleNamespace()
    return UnifiedMCPControlPlaneService(
        local_service=fake_local_service, server_service=None, target_store=None, context_store=None
    )


def _permission_log_records(store: LocalMCPStore) -> list[dict]:
    log_path = Path(store.path).with_name("mcp_execution_log.jsonl")
    return MCPExecutionLog(log_path).read_recent()


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

    assert result == {
        ("local:demo", "search"): result[("local:demo", "search")],
        ("local:other", "write"): result[("local:other", "write")],
    }
    for effective in result.values():
        assert effective.state == "ask"
        assert effective.origin == "global_default"
        assert effective.config_changed is False


# -- effective_tool_states: precedence resolution ----------------------------


def test_effective_tool_states_resolves_per_precedence_with_real_store(tmp_path):
    service, store = _service(tmp_path)
    tool_override = _tool(name="search", server_key="local:demo")
    server_default_tool = _tool(name="write", server_key="local:demo")
    global_default_tool = _tool(name="fetch", server_key="local:other")

    current_hash = definition_hash(tool_override.description, tool_override.input_schema)
    permission_store = MCPPermissionStore(Path(store.path).with_name("mcp_permissions.json"))
    permission_store.set_global_default("deny")
    permission_store.set_server_default("local:demo", "ask")
    permission_store.set_tool_state("local:demo", "search", "allow", definition_hash=current_hash)

    result = service.effective_tool_states([tool_override, server_default_tool, global_default_tool])

    assert result[("local:demo", "search")].state == "allow"
    assert result[("local:demo", "search")].origin == "tool_override"
    assert result[("local:demo", "write")].state == "ask"
    assert result[("local:demo", "write")].origin == "server_default"
    assert result[("local:other", "fetch")].state == "deny"
    assert result[("local:other", "fetch")].origin == "global_default"


# -- effective_tool_states: rug-pull downgrade audit -------------------------


def test_effective_tool_states_fresh_mismatch_emits_exactly_one_downgraded_record_across_two_calls(tmp_path):
    service, store = _service(tmp_path)
    original_tool = _tool(name="search", description="Search docs")
    permission_store = service.permission_store
    original_hash = definition_hash(original_tool.description, original_tool.input_schema)
    permission_store.set_tool_state("local:demo", "search", "allow", definition_hash=original_hash)

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
    assert record["error"] == "search definition changed since you allowed it — review and re-allow"

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
    permission_store.set_tool_state("local:demo", "search", "allow", definition_hash=current_hash)

    result = service.effective_tool_states([tool])

    assert result[("local:demo", "search")].state == "allow"
    assert result[("local:demo", "search")].config_changed is False
    entry = permission_store.get_tool_entry("local:demo", "search")
    assert not entry.get("config_changed")
    assert _permission_log_records(store) == []


def test_effective_tool_states_downgrade_audit_survives_execution_log_failure(tmp_path, monkeypatch):
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
    permission_store.set_tool_state("local:demo", "search", "allow", definition_hash="stale-hash")

    result = service.effective_tool_states([tool])

    assert result[("local:demo", "search")].state == "ask"
    assert result[("local:demo", "search")].config_changed is True


# -- set_tool_state -----------------------------------------------------------


def test_set_tool_state_allow_computes_and_stores_definition_hash_and_clears_marker(tmp_path):
    service, _store = _service(tmp_path)
    permission_store = service.permission_store
    permission_store.set_tool_state("local:demo", "search", "allow", definition_hash="stale-hash")
    permission_store.mark_config_changed("local:demo", "search")
    tool = _tool(name="search", description="Search docs")

    service.set_tool_state("local:demo", "search", "allow", tool=tool)

    entry = permission_store.get_tool_entry("local:demo", "search")
    assert entry["state"] == "allow"
    assert entry["definition_hash"] == definition_hash(tool.description, tool.input_schema)
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

    service.set_tool_state("local:demo", "search", "allow", tool=_tool())  # must not raise


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
    service.permission_store.set_tool_state("local:demo", "search", stored_state, **kwargs)

    result = service.gate_tool_test(tool)

    assert result.state == expected_state


def test_gate_tool_test_ignores_kill_switch(tmp_path):
    service, _store = _service(tmp_path)
    tool = _tool(name="search")
    service.permission_store.set_tool_state(
        "local:demo", "search", "allow", definition_hash=definition_hash(tool.description, tool.input_schema)
    )
    service.permission_store.set_kill_switch(True)

    result = service.gate_tool_test(tool)

    assert result.state == "allow"


def test_gate_tool_test_does_not_emit_audit_record_on_fresh_mismatch(tmp_path):
    service, store = _service(tmp_path)
    tool = _tool(name="search")
    service.permission_store.set_tool_state("local:demo", "search", "allow", definition_hash="stale-hash")

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
        "local:demo", "search", "allow", definition_hash=definition_hash(tool.description, tool.input_schema)
    )

    result = service.gate_tool_test_by_key("local:demo", "search")

    assert result.state == "ask"
    assert result.config_changed is True


def test_gate_tool_test_by_key_does_not_emit_audit_record(tmp_path):
    service, store = _service(tmp_path)
    service.permission_store.set_tool_state("local:demo", "search", "allow", definition_hash="stale-hash")

    service.gate_tool_test_by_key("local:demo", "search")

    assert _permission_log_records(store) == []
