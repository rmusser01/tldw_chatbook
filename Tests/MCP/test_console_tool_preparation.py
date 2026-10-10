"""Fresh preparation owns I/O; consumers receive detached policy/catalog data."""

import asyncio
import threading
import time
from dataclasses import FrozenInstanceError

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission

_MaximumProbe = snapshot_controls._MaximumProbe
_loop_projection = snapshot_controls._loop_projection
catalog_controls = snapshot_controls.catalog_controls
catalog_store = snapshot_controls.catalog_store
local_root = snapshot_controls.local_root
mcp_sources = snapshot_controls.mcp_sources
snapshot_case = snapshot_controls.snapshot_case


async def _prepare(case, *, catalog=True, local=True, profile="default", maximum=None):
    from tldw_chatbook.MCP import console_tool_preparation

    callback = getattr(console_tool_preparation, "prepare_console_tools", None)
    assert callable(callback), "shared tool preparation has not been implemented"
    return await callback(
        case.service,
        profile_id=profile,
        include_mcp_catalog=catalog,
        need_local_switch=local,
        builtin_raw_name_exclusions=frozenset(),
        owned_profile_ids=frozenset(),
        **({"maximum_tool_ids": maximum} if maximum is not None else {}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "catalog,local,killed,policy_reads,catalog_reads",
    [
        (True, True, False, 1, 1),
        (True, False, False, 1, 1),
        (False, True, False, 1, 0),
        (False, False, False, 0, 0),
        (True, True, True, 1, 0),
    ],
)
async def test_native_read_counts_and_loop_inventory(
    snapshot_case, catalog, local, killed, policy_reads, catalog_reads
):
    case = snapshot_case
    _, observed = _loop_projection(case)
    if killed:
        case.permissions.set_kill_switch(True)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        prepared = await _prepare(case, catalog=catalog, local=local)
    assert len(probe.permission_threads) == policy_reads
    assert len(probe.read_threads) == catalog_reads
    assert all(
        thread is not threading.current_thread()
        for thread in probe.permission_threads + probe.read_threads
    )
    if not catalog and not local:
        assert prepared is None and observed == []
    else:
        prepared.require_current(case.service)
        assert prepared.kill_switch is killed
        assert observed == ([True] if catalog_reads else [])
        assert [row.name for row in prepared.tools] == (
            ["first", "sample_builtin"] if catalog_reads else []
        )
        assert all(
            lease not in storage_admission._live_leases for lease in probe.leases
        )


@pytest.mark.asyncio
async def test_prepared_schema_is_detached_for_each_consumer(snapshot_case):
    case = snapshot_case
    schema = {
        "type": "object",
        "properties": {"z": {"enum": ["a", "b"]}, "a": {"type": "string"}},
    }
    case.source.save_discovery_snapshot(
        "one", {"tools": [{"name": "first", "inputSchema": schema}]}
    )
    stored_schema = case.source.get_discovery_snapshot("one")["tools"][0]["inputSchema"]
    case.local.manifest_provider = lambda: {
        "tools": [{"name": "ordered", "inputSchema": schema}]
    }
    prepared = await _prepare(case)
    assert list(prepared.tools[0].to_hub_tool().input_schema["properties"]) == list(
        stored_schema["properties"]
    )
    row = prepared.tools[-1]
    first, second = row.to_hub_tool(), row.to_hub_tool()
    assert list(first.input_schema["properties"]) == ["z", "a"]
    first.input_schema["properties"]["z"]["enum"].append("changed")
    assert second.input_schema["properties"]["z"]["enum"] == ["a", "b"]
    with pytest.raises(FrozenInstanceError):
        row.name = "changed"
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        assert row.to_hub_tool().input_schema == second.input_schema
    assert not probe.read_threads and not probe.permission_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "restricted"])
async def test_named_profile_projection_matches_public_fresh_resolution(
    snapshot_case, profile
):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.ensure_profile(profile)
    case.permissions.set_global_default("deny", profile_id=profile)
    prepared = await _prepare(case, profile=profile)
    tools = [row.to_hub_tool() for row in prepared.tools]
    expected = await asyncio.to_thread(
        case.service.effective_tool_states, tools, profile_id=profile
    )
    assert all(
        row.effective == expected[(row.server_key, row.name)] for row in prepared.tools
    )
    assert [row.effective.state for row in prepared.tools] == ["deny", "deny"]


@pytest.mark.asyncio
async def test_replaced_reader_declines_without_invoking_custom_source(snapshot_case):
    case = snapshot_case
    calls = []
    case.service.effective_tool_states = lambda *a, **k: calls.append(True)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        assert await _prepare(case) is None
    assert calls == [] and not probe.read_threads and not probe.permission_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancel", "source"])
async def test_preparation_holds_native_work_and_refuses_changed_source(
    snapshot_case, tmp_path, change
):
    case = snapshot_case
    _loop_projection(case)
    probe = snapshot_controls._PermissionProbe(case.source, case.permissions)
    old_path = case.source.path
    with probe.installed():
        task = asyncio.create_task(_prepare(case))
        try:
            await catalog_controls._worker_entered(probe, task)
            if change == "cancel":
                case.service._maintenance_close_admission()
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert not await case.service._maintenance_drain(
                    time.monotonic() + 0.03
                )
                assert probe.leases and all(
                    lease in storage_admission._live_leases for lease in probe.leases
                )
            else:
                case.source.path = tmp_path / "redirected.json"
            probe.release.set()
            if change == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert await case.service._maintenance_drain(time.monotonic() + 0.5)
            else:
                with pytest.raises(bootstrap.RecoveryRequired):
                    await task
                assert not case.source.path.exists()
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            case.source.path = old_path
            await catalog_controls._settle(task, probe)
            if case.service._producer_lifetime.closed:
                case.service._maintenance_resume()


@pytest.mark.asyncio
async def test_result_cannot_be_adopted_by_another_service(snapshot_case):
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    case = snapshot_case
    prepared = await _prepare(case)
    other = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=case.local,
        server_service=None,
    )
    other._permission_store = case.permissions
    with pytest.raises(bootstrap.RecoveryRequired):
        prepared.require_current(other)


@pytest.mark.asyncio
async def test_changed_definition_audits_once_and_keeps_fresh_invocation_gate(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash="0" * 64
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        prepared = await _prepare(case)
    row = prepared.tools[0]
    assert row.effective.state == "ask" and row.effective.config_changed
    assert (
        case.permissions.get_tool_entry("local:one", "first")["config_changed"] is True
    )
    assert len(probe.audit_threads) == 1
    assert all(
        thread is not threading.current_thread() for thread in probe.audit_threads
    )
    assert (
        len(
            [
                record
                for record in case.service.execution_log.read_recent()
                if record["decision"] == "downgraded"
            ]
        )
        == 1
    )
    await _prepare(case)
    assert (
        len(
            [
                record
                for record in case.service.execution_log.read_recent()
                if record["decision"] == "downgraded"
            ]
        )
        == 1
    )
    case.permissions.set_tool_state("local:one", "first", "deny")
    assert case.service.gate_tool_test(row.to_hub_tool()).state == "deny"
    assert row.effective.state == "ask"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["helper", "resolver"])
async def test_replaced_pure_resolver_declines_without_native_reads(
    snapshot_case, monkeypatch, kind
):
    from tldw_chatbook.MCP import unified_control_plane_service as control

    case = snapshot_case
    calls = []
    name = (
        "_resolve_tool_states_from_payload"
        if kind == "helper"
        else "resolve_effective_state"
    )
    monkeypatch.setattr(control, name, lambda *args, **kwargs: calls.append(True))
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        assert await _prepare(case) is None
    assert calls == [] and not probe.read_threads and not probe.permission_threads


@pytest.mark.asyncio
async def test_unissued_copy_cannot_be_adopted(snapshot_case):
    from dataclasses import replace

    prepared = await _prepare(snapshot_case)
    copy = replace(prepared)
    with pytest.raises(bootstrap.RecoveryRequired):
        copy.require_current(snapshot_case.service)


@pytest.mark.asyncio
async def test_non_json_schema_retains_ordinary_route(snapshot_case):
    case = snapshot_case
    case.local.manifest_provider = lambda: {
        "tools": [
            {"name": "custom", "inputSchema": {"type": "object", "extra": {"a", "b"}}}
        ]
    }
    assert await _prepare(case) is None


@pytest.mark.asyncio
async def test_replaced_resolver_dependency_retains_ordinary_route(
    snapshot_case, monkeypatch
):
    from tldw_chatbook.MCP import permission_store

    calls = []
    monkeypatch.setattr(
        permission_store, "_profile_chain", lambda *a, **k: calls.append(True)
    )
    probe = _MaximumProbe(snapshot_case.source, snapshot_case.permissions)
    with probe.installed():
        assert await _prepare(snapshot_case) is None
    assert calls == [] and not probe.read_threads and not probe.permission_threads


@pytest.mark.asyncio
async def test_public_resolution_audits_each_duplicate_definition_using_its_own_state(
    snapshot_case,
):
    from dataclasses import replace
    from tldw_chatbook.MCP.hub_tool_catalog import local_tools_from_record
    from tldw_chatbook.MCP.permission_store import definition_hash

    case = snapshot_case
    tool = local_tools_from_record(
        {"profile_id": "one", "discovery_snapshot": {"tools": [{"name": "first"}]}}
    )[0]
    changed = replace(tool, description="changed")
    case.permissions.set_tool_state(
        "local:one",
        "first",
        "allow",
        definition_hash=definition_hash(tool.description, tool.input_schema),
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        result = await asyncio.to_thread(
            case.service.effective_tool_states, [tool, changed]
        )
    assert result[("local:one", "first")].config_changed
    assert len(probe.audit_threads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["converter", "result", "schema"])
async def test_replaced_preparation_helper_declines_before_native_reads(
    snapshot_case, monkeypatch, kind
):
    from tldw_chatbook.MCP import console_tool_preparation as preparation

    calls = []
    owner, name = {
        "converter": (preparation.PreparedConsoleTool, "to_hub_tool"),
        "result": (preparation.ConsoleToolPreparation, "require_current"),
        "schema": (preparation, "_schema_json"),
    }[kind]
    monkeypatch.setattr(owner, name, lambda *args, **kwargs: calls.append(True))
    probe = _MaximumProbe(snapshot_case.source, snapshot_case.permissions)
    with probe.installed():
        assert await _prepare(snapshot_case) is None
    assert calls == [] and not probe.permission_threads and not probe.read_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["converter", "schema"])
async def test_issued_result_refuses_replaced_pure_helper_before_use(
    snapshot_case, monkeypatch, kind
):
    from tldw_chatbook.MCP import console_tool_preparation as preparation

    prepared = await _prepare(snapshot_case)
    calls = []
    owner, name = (
        (preparation.PreparedConsoleTool, "to_hub_tool")
        if kind == "converter"
        else (preparation, "_schema_json")
    )
    monkeypatch.setattr(owner, name, lambda *args, **kwargs: calls.append(True))
    probe = _MaximumProbe(snapshot_case.source, snapshot_case.permissions)
    with probe.installed():
        with pytest.raises(bootstrap.RecoveryRequired):
            prepared.require_current(snapshot_case.service)
    assert calls == [] and not probe.permission_threads and not probe.read_threads


class _NormalizationProbe(_MaximumProbe):
    """Observe real preparation work without replacing qualified helpers."""

    def __init__(self, store, permissions):
        super().__init__(store, permissions)
        from tldw_chatbook.MCP import console_tool_preparation as preparation
        from tldw_chatbook.MCP import hub_tool_catalog as catalog
        from tldw_chatbook.MCP import unified_control_plane_service as control

        self.extra_codes = {
            preparation._schema_json.__code__,
            preparation.PreparedConsoleTool.to_hub_tool.__code__,
            control._resolve_tool_states_from_payload.__code__,
            catalog.local_tools_from_record.__code__,
            catalog.builtin_tools_from_inventory.__code__,
        }
        self.normalization_calls = []

    def observe(self, frame, event, arg):
        super().observe(frame, event, arg)
        if event == "call" and frame.f_code in self.extra_codes:
            self.normalization_calls.append(frame.f_code.co_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "stored_string",
        "deep",
        "aggregate_bytes",
        "tool_count",
        "aggregate_nodes",
        "description",
    ],
)
async def test_oversized_catalog_declines_before_new_normalization_and_preserves_ordinary_output(
    snapshot_case, kind
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.MCP.client import (
        MAX_DESCRIPTOR_STRING_LENGTH,
        MAX_SCHEMA_BYTES,
        MAX_SCHEMA_DEPTH,
    )

    case = snapshot_case
    expected = []
    if kind == "stored_string":
        raw = {
            "name": "first",
            "inputSchema": {"type": "object", "large": "x" * (MAX_SCHEMA_BYTES + 1)},
        }
        case.source.save_discovery_snapshot("one", {"tools": [raw]})
        expected.append(raw)
        inventory_tools = []
    else:
        expected.append({"name": "first"})
        if kind == "deep":
            schema = {}
            child = schema
            for _ in range(MAX_SCHEMA_DEPTH):
                child["nested"] = {}
                child = child["nested"]
            inventory_tools = [{"name": "deep", "inputSchema": schema}]
        elif kind == "aggregate_bytes":
            inventory_tools = [
                {"name": f"aggregate_{index}", "inputSchema": {"large": "x" * 32_768}}
                for index in range(40)
            ]
        elif kind == "tool_count":
            inventory_tools = [
                {"name": f"tool_{index}", "inputSchema": {"type": "object"}}
                for index in range(257)
            ]
        elif kind == "aggregate_nodes":
            inventory_tools = [
                {"name": f"nodes_{index}", "inputSchema": {"enum": [None] * 10_000}}
                for index in range(2)
            ]
        else:
            inventory_tools = [
                {
                    "name": "description",
                    "description": "x" * (MAX_DESCRIPTOR_STRING_LENGTH + 1),
                    "inputSchema": {"type": "object"},
                }
            ]
        expected.extend(inventory_tools)
    case.local.manifest_provider = lambda: {"tools": inventory_tools}
    probe = _NormalizationProbe(case.source, case.permissions)
    with probe.installed():
        prepared = await _prepare(case)
    assert type(prepared) is type(
        None
    ), "oversized data must decline sharing, never truncate"
    assert probe.normalization_calls == []
    assert len(probe.permission_threads) == 1 and len(probe.read_threads) == 1
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)

    # The optimization's limits are eligibility only. The existing provider
    # still advertises exactly the accepted tool/schema/description output.
    provider = MCPToolProvider(
        service=case.service, main_loop=asyncio.get_running_loop()
    )
    await provider.compose_catalog()
    actual = [provider.load_schema(entry.id) for entry in provider.list_catalog()]
    assert len(actual) == len(expected)
    for actual_tool, expected_tool in zip(actual, expected):
        assert actual_tool.description == expected_tool.get("description", "")
        assert actual_tool.parameters == expected_tool.get(
            "inputSchema", {"type": "object", "properties": {}}
        )


_BUILTIN_MAXIMUM = frozenset({"builtin:tldw_chatbook::sample_builtin"})
_MIXED_MAXIMUM = _BUILTIN_MAXIMUM | {"local:one::first"}


@pytest.mark.parametrize(
    "maximum,expected",
    [
        (None, True),
        (frozenset(), False),
        (_BUILTIN_MAXIMUM, False),
        (_MIXED_MAXIMUM, True),
        (frozenset({"local:one::first"}), True),
        (frozenset({"server:remote::first"}), True),
        (frozenset({"builtin:other::first"}), True),
        (frozenset({"builtin:tldw_chatbook::"}), True),
        (frozenset({"unknown"}), True),
        (frozenset({1}), True),
        (set(_BUILTIN_MAXIMUM), True),
        (list(_BUILTIN_MAXIMUM), True),
    ],
)
def test_external_catalog_requirement_is_conservative(maximum, expected):
    from tldw_chatbook.MCP.hub_tool_catalog import maximum_needs_external_catalog

    assert maximum_needs_external_catalog(maximum) is expected


def test_external_catalog_requirement_does_not_consume_custom_values():
    from tldw_chatbook.MCP.hub_tool_catalog import maximum_needs_external_catalog

    class CustomMaximum(frozenset):
        def __iter__(self):
            pytest.fail("custom maximum was consumed during eligibility")

    class CustomId(str):
        def startswith(self, *args, **kwargs):
            pytest.fail("custom ID callback was consumed during eligibility")

    assert maximum_needs_external_catalog(CustomMaximum(_BUILTIN_MAXIMUM)) is True
    assert (
        maximum_needs_external_catalog(
            frozenset({CustomId("builtin:tldw_chatbook::sample_builtin")})
        )
        is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "maximum,external", [(_BUILTIN_MAXIMUM, False), (_MIXED_MAXIMUM, True)]
)
async def test_frozen_maximum_omits_only_unneeded_catalog_read(
    snapshot_case, maximum, external
):
    case = snapshot_case
    _, inventory = _loop_projection(case)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        prepared = await _prepare(case, maximum=maximum)
    prepared.require_current(case.service)
    assert not prepared.kill_switch
    assert [row.name for row in prepared.tools] == (
        ["first", "sample_builtin"] if external else ["sample_builtin"]
    )
    assert prepared._includes_external_catalog is external
    assert len(probe.permission_threads) == 1
    assert len(probe.read_threads) == int(external)
    assert inventory == [True]
    assert all(
        thread is not threading.current_thread()
        for thread in probe.permission_threads + probe.read_threads
    )
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)


@pytest.mark.asyncio
async def test_excluded_external_definition_is_audited_when_next_needed(snapshot_case):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash="0" * 64
    )
    unused = _MaximumProbe(case.source, case.permissions)
    with unused.installed():
        prepared = await _prepare(case, maximum=_BUILTIN_MAXIMUM)
    assert [row.name for row in prepared.tools] == ["sample_builtin"]
    assert len(unused.permission_threads) == 1
    assert not unused.read_threads and not unused.audit_threads
    assert (
        case.permissions.get_tool_entry("local:one", "first").get("config_changed")
        is not True
    )
    needed = _MaximumProbe(case.source, case.permissions)
    with needed.installed():
        prepared = await _prepare(case, maximum=_MIXED_MAXIMUM)
    external = next(row for row in prepared.tools if row.server_key == "local:one")
    assert external.effective.state == "ask" and external.effective.config_changed
    assert len(needed.read_threads) == len(needed.audit_threads) == 1
    assert case.permissions.get_tool_entry("local:one", "first")["config_changed"]
    assert (
        len(
            [
                row
                for row in case.service.execution_log.read_recent()
                if row["decision"] == "downgraded"
            ]
        )
        == 1
    )
    assert all(lease not in storage_admission._live_leases for lease in needed.leases)


@pytest.mark.asyncio
async def test_unused_corrupt_catalog_stays_unread_until_external_is_needed(
    snapshot_case,
):
    from tldw_chatbook.MCP.local_store import LocalMCPStoreLoadError

    case = snapshot_case
    _loop_projection(case)
    original = case.source.path.read_bytes()
    corrupt = b"{not valid catalog JSON"
    case.source.path.write_bytes(corrupt)
    try:
        unused = _MaximumProbe(case.source, case.permissions)
        with unused.installed():
            prepared = await _prepare(case, maximum=_BUILTIN_MAXIMUM)
        assert [row.name for row in prepared.tools] == ["sample_builtin"]
        assert len(unused.permission_threads) == 1
        assert not unused.read_threads
        assert case.source.path.read_bytes() == corrupt
        needed = _MaximumProbe(case.source, case.permissions)
        with needed.installed(), pytest.raises(LocalMCPStoreLoadError):
            await _prepare(case, maximum=_MIXED_MAXIMUM)
        assert len(needed.permission_threads) == len(needed.read_threads) == 1
        assert case.source.path.read_bytes() == corrupt
        assert needed.leases and all(
            lease not in storage_admission._live_leases for lease in needed.leases
        )
    finally:
        case.source.path.write_bytes(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["deny", "kill"])
async def test_builtin_only_preparation_rereads_common_policy(snapshot_case, change):
    case = snapshot_case
    _, inventory = _loop_projection(case)
    first = await _prepare(case, maximum=_BUILTIN_MAXIMUM)
    assert first.tools[0].effective.state != "deny" and not first.kill_switch
    inventory.clear()
    if change == "kill":
        case.permissions.set_kill_switch(True)
    else:
        case.permissions.set_tool_state(
            "builtin:tldw_chatbook", "sample_builtin", "deny"
        )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        current = await _prepare(case, maximum=_BUILTIN_MAXIMUM)
    current.require_current(case.service)
    assert len(probe.permission_threads) == 1 and not probe.read_threads
    if change == "kill":
        assert current.kill_switch and current.tools == () and inventory == []
    else:
        assert not current.kill_switch and inventory == [True]
        assert current.tools[0].effective.state == "deny"
    assert first.tools[0].effective.state != "deny" and not first.kill_switch
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)


@pytest.mark.asyncio
async def test_builtin_only_preparation_keeps_builtin_hash_free_policy(snapshot_case):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state(
        "builtin:tldw_chatbook", "sample_builtin", "allow", definition_hash="0" * 64
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        prepared = await _prepare(case, maximum=_BUILTIN_MAXIMUM)
    assert [row.name for row in prepared.tools] == ["sample_builtin"]
    # The builtin namespace is already hash-free; preserve that policy rather
    # than inventing an external-definition downgrade for an installed tool.
    assert prepared.tools[0].effective.state == "allow"
    assert not prepared.tools[0].effective.config_changed
    assert not probe.read_threads and not probe.audit_threads
    assert probe.permission_threads and all(
        thread is not threading.current_thread()
        for thread in probe.permission_threads + probe.audit_threads
    )
    assert not case.permissions.get_tool_entry(
        "builtin:tldw_chatbook", "sample_builtin"
    ).get("config_changed", False)
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)
