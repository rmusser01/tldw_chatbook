"""Shared stock composition observes original native policy once."""

import asyncio
import threading
import time
from types import SimpleNamespace
from dataclasses import replace

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from Tests.Chat.test_console_async_mcp_snapshot import (
    _MaximumProbe,
    _PermissionProbe,
    _loop_projection,
)
from Tests.Chat.test_console_local_review_hook import _test_execution_context
from Tests.MCP import test_external_catalog_worker_ownership as catalog_controls
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.MCP.permission_store import definition_hash
from tldw_chatbook.Agents.tool_refusals import TOOL_KILL_SWITCH_REFUSAL

mcp_sources = snapshot_controls.mcp_sources
local_root = snapshot_controls.local_root
catalog_store = snapshot_controls.catalog_store
snapshot_case = snapshot_controls.snapshot_case


def _context(case, *, local_enabled=True):
    context = _test_execution_context(
        case.controller._scratch_spaces.snapshot(case.session.id),
        session_id=case.session.id,
        tool_configuration={"local_tools_enabled": local_enabled},
    )
    return replace(
        context,
        configuration=replace(
            context.configuration,
            mcp_tool_maximum=frozenset(
                {"local:one::first", "builtin:tldw_chatbook::sample_builtin"}
            ),
            mcp_definition_maximum={
                "local:one::first": definition_hash("", None),
                "builtin:tldw_chatbook::sample_builtin": definition_hash(
                    "builtin", {"type": "object"}
                ),
            },
        ),
    )


@pytest.mark.asyncio
async def test_actual_composition_observes_policy_once_for_both_consumers(
    snapshot_case,
):
    case = snapshot_case
    _, inventory_calls = _loop_projection(case)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, _, local, _ = await case.controller._compose_agent_request_providers(
            session_id=case.session.id,
            project_selection=None,
            project_authority_guard=None,
            turn_context=_context(case),
            admitted_roots=(),
        )
    assert mcp is not None and local is not None
    assert len(mcp.list_catalog()) == 2
    assert inventory_calls == [True]
    assert len(probe.read_threads) == 1
    assert all(
        thread is not threading.current_thread() for thread in probe.permission_threads
    )
    assert (
        len(probe.permission_threads) == 1
    ), "MCP and local composition repeat original permission loads"


async def _compose(case, context, **kwargs):
    return await case.controller._compose_agent_request_providers(
        session_id=case.session.id,
        project_selection=None,
        project_authority_guard=None,
        turn_context=context,
        admitted_roots=(),
        **kwargs,
    )


def _configure(context, **kwargs):
    return replace(context, configuration=replace(context.configuration, **kwargs))


def _catalog_projection(provider):
    return [
        (entry, provider.load_schema(entry.id)) for entry in provider.list_catalog()
    ]


@pytest.mark.asyncio
async def test_shared_catalog_matches_standalone_and_pure_consumers_do_not_read(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    context = _context(case)
    standalone = await case.controller._compose_mcp_provider(
        case.session.id,
        maximum_tool_ids=context.mcp_tool_maximum,
        maximum_definition_hashes=context.mcp_definition_maximum,
    )
    expected = _catalog_projection(standalone)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, _, local, review = await _compose(case, context)
        assert _catalog_projection(mcp) == expected
        assert _catalog_projection(mcp) == expected
        assert local.list_catalog()
        assert callable(review)
    assert len(probe.permission_threads) == len(probe.read_threads) == 1
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)
    assert mcp.not_connected_count == standalone.not_connected_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "maximum,local_enabled,killed,permission_reads,catalog_reads,has_mcp,has_local",
    [
        (None, True, False, 1, 1, True, True),
        (frozenset(), True, False, 1, 0, False, True),
        (frozenset(), False, False, 0, 0, False, False),
        (None, False, False, 1, 1, True, False),
        (None, True, True, 1, 0, False, False),
        (frozenset(), True, True, 1, 0, False, False),
    ],
    ids=["unset", "local-only", "no-consumers", "mcp-only", "killed", "killed-local"],
)
async def test_conditional_consumers_do_only_required_native_work(
    snapshot_case,
    maximum,
    local_enabled,
    killed,
    permission_reads,
    catalog_reads,
    has_mcp,
    has_local,
):
    case = snapshot_case
    _, inventory_calls = _loop_projection(case)
    case.permissions.set_kill_switch(killed)
    context = _configure(
        _context(case, local_enabled=local_enabled),
        mcp_tool_maximum=maximum,
        mcp_definition_maximum=None,
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, _, local, _ = await _compose(case, context)
    assert (mcp is not None) is has_mcp
    assert (local is not None) is has_local
    assert len(probe.permission_threads) == permission_reads
    assert len(probe.read_threads) == catalog_reads
    assert len(inventory_calls) == catalog_reads


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_counts", [True, False], ids=["dispatch", "preview"])
async def test_shared_counts_preserve_live_and_disposable_preview(
    snapshot_case, publish_counts
):
    case = snapshot_case
    _loop_projection(case)
    case.app.console_mcp_tool_count = 7
    case.app.console_mcp_not_connected_count = 4
    mcp, _, _, _ = await _compose(
        case, _context(case), publish_mcp_counts=publish_counts
    )
    assert mcp is not None
    assert (
        case.app.console_mcp_tool_count,
        case.app.console_mcp_not_connected_count,
    ) == ((2, 1) if publish_counts else (7, 4))


@pytest.mark.asyncio
async def test_named_profile_inheritance_and_definition_maximum_remain_narrowing(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.ensure_profile("focused")
    case.permissions.set_tool_state("local:one", "first", "deny")
    context = _configure(_context(case), tool_policy_profile_id="focused")
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, _, local, _ = await _compose(case, context)
    assert [hub.name for hub, _ in mcp._entry_by_llm_name.values()] == [
        "sample_builtin"
    ]
    assert mcp._profile_id() == "focused" and local is not None
    assert len(probe.permission_threads) == 1
    changed = _configure(
        context,
        mcp_definition_maximum={"builtin:tldw_chatbook::sample_builtin": "0" * 64},
    )
    mcp, _, local, _ = await _compose(case, changed)
    assert mcp is None and local is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["factory", "wrapper", "reader"])
async def test_custom_routes_keep_ordinary_composition_shape(
    snapshot_case, monkeypatch, kind
):
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.Chat import console_chat_controller as controller_source

    case = snapshot_case
    _loop_projection(case)
    observed = []
    if kind == "factory":

        class CustomProvider(provider_source.MCPToolProvider):
            pass

        monkeypatch.setattr(controller_source, "MCPToolProvider", CustomProvider)
    elif kind == "wrapper":
        original = case.controller._compose_mcp_provider

        async def custom(
            session_id=None,
            *,
            publish_counts=True,
            maximum_tool_ids=None,
            maximum_definition_hashes=None,
        ):
            observed.append(True)
            return await original(
                session_id,
                publish_counts=publish_counts,
                maximum_tool_ids=maximum_tool_ids,
                maximum_definition_hashes=maximum_definition_hashes,
            )

        monkeypatch.setattr(case.controller, "_compose_mcp_provider", custom)
    else:
        original = case.service.effective_tool_states

        def custom(tools, *, profile_id="default"):
            observed.append(True)
            return original(tools, profile_id=profile_id)

        monkeypatch.setattr(case.service, "effective_tool_states", custom)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, _, local, _ = await _compose(case, _context(case))
    assert mcp is not None and local is not None
    assert [hub.name for hub, _ in mcp._entry_by_llm_name.values()] == (
        ["first"] if kind == "reader" else ["first", "sample_builtin"]
    )
    assert len(probe.permission_threads) >= 3
    if kind != "factory":
        assert observed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["service", "session", "settings", "config", "scratch", "root-guard"]
)
async def test_changed_owner_refuses_shared_publication_after_native_read(
    snapshot_case, change
):
    case = snapshot_case
    _loop_projection(case)
    case.app.console_mcp_tool_count = 7
    guard = {"current": True}
    probe = _PermissionProbe(case.source, case.permissions)
    with probe.installed():
        pending = asyncio.create_task(
            case.controller._compose_agent_request_providers(
                session_id=case.session.id,
                project_selection=None,
                project_authority_guard=lambda: guard["current"],
                turn_context=_context(case),
                admitted_roots=(),
            )
        )
        try:
            await catalog_controls._worker_entered(probe, pending)
            if change == "service":
                case.app.unified_mcp_service = SimpleNamespace()
            elif change == "session":
                case.store._sessions[case.session.id] = replace(case.session)
            elif change == "settings":
                case.store._bump_settings_revision(case.session.id)
            elif change == "config":
                case.app.app_config = {}
            elif change == "scratch":
                case.controller._scratch_spaces = object()
            else:
                guard["current"] = False
            probe.release.set()
            with pytest.raises(
                RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await pending
            assert case.app.console_mcp_tool_count == 7
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(pending, probe)


@pytest.mark.asyncio
async def test_switch_flip_after_shared_composition_refuses_real_invocation(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    mcp, _, local, _ = await _compose(case, _context(case))
    case.permissions.set_kill_switch(True)
    from tldw_chatbook.Agents.run_context import use_run_id

    def invoke():
        with use_run_id("shared-compose-live-gate"):
            return mcp.invoke(mcp.list_catalog()[0].id, {})

    result = await asyncio.to_thread(invoke)
    assert result.ok is False
    assert result.error == TOOL_KILL_SWITCH_REFUSAL
    # The stock local invocation closure remains live too; an observation
    # captured for composition cannot supply a later clearance.
    assert await asyncio.to_thread(local._kill_switch) is True
    from tldw_chatbook.Agents.local_tool_provider import _PATH_AUTHORITY_LOCAL_NAMES

    local_name = next(
        name for name in local._specs if name not in _PATH_AUTHORITY_LOCAL_NAMES
    )
    outcome = await asyncio.to_thread(local.invoke_detailed, f"local:{local_name}", {})
    assert outcome.final_gate == "kill_switch" and not outcome.dispatch_started
    assert outcome.result.error == TOOL_KILL_SWITCH_REFUSAL


@pytest.mark.asyncio
async def test_persona_confirmation_floor_reaches_shared_mcp_and_local_gates(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash=definition_hash("", None)
    )
    context = _configure(
        _context(case),
        persona_policy_rules=(
            {
                "rule_kind": "mcp_tool",
                "rule_name": "first",
                "allowed": True,
                "require_confirmation": True,
            },
            {
                "rule_kind": "mcp_tool",
                "rule_name": "web_fetch",
                "allowed": True,
                "require_confirmation": True,
            },
        ),
    )
    mcp, _, local, _ = await _compose(case, context)
    entry = next(
        entry
        for entry in mcp.list_catalog()
        if mcp._entry_by_llm_name[entry.id][0].name == "first"
    )
    local_hub = local.hub_tool_for("web_fetch")
    case.permissions.set_tool_state(
        local_hub.server_key,
        local_hub.name,
        "allow",
        definition_hash=definition_hash(local_hub.description, local_hub.input_schema),
    )
    pending = await asyncio.to_thread(mcp.pending_gate_for, entry.id, {})
    assert pending is not None and pending.reason == "ask"
    state = await asyncio.to_thread(local._resolve_state, local_hub)
    assert state.state == "ask" and state.origin == "persona_policy"


@pytest.mark.asyncio
async def test_plugin_maximum_even_empty_keeps_ordinary_route(snapshot_case):
    case = snapshot_case
    _loop_projection(case)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        result = await case.controller._compose_shared_tool_providers(
            session_id=case.session.id,
            project_selection=None,
            project_authority_guard=None,
            turn_context=_context(case),
            publish_mcp_counts=True,
            admitted_roots=(),
            mcp_profile_kwargs={"plugin_maximum": {}},
        )
    assert result is None and not probe.permission_threads and not probe.read_threads


@pytest.mark.asyncio
async def test_repeated_cancel_drains_combined_native_work_before_return(snapshot_case):
    case = snapshot_case
    _loop_projection(case)
    probe = _PermissionProbe(case.source, case.permissions)
    with probe.installed():
        pending = asyncio.create_task(_compose(case, _context(case)))
        try:
            await catalog_controls._worker_entered(probe, pending)
            case.service._maintenance_close_admission()
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            assert not await case.service._maintenance_drain(time.monotonic() + 0.03)
            assert probe.leases and all(
                lease in storage_admission._live_leases for lease in probe.leases
            )
            probe.release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert await case.service._maintenance_drain(time.monotonic() + 0.5)
            assert probe.leases and all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(pending, probe)
            if case.service._producer_lifetime.closed:
                case.service._maintenance_resume()


@pytest.mark.asyncio
async def test_adoption_requires_one_operation_profile_and_run_owned_schema(
    snapshot_case,
):
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.Chat.console_chat_controller import (
        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    from tldw_chatbook.MCP.console_tool_preparation import prepare_console_tools

    case = snapshot_case
    _loop_projection(case)
    preparation = await prepare_console_tools(
        case.service,
        include_mcp_catalog=True,
        need_local_switch=True,
        builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        owned_profile_ids=frozenset(),
    )
    composition = provider_source.capture_standard_controller_composition(
        provider_source.MCPToolProvider, case.service
    )
    provider = provider_source.MCPToolProvider(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    provider._stamped_decisions[("old-run", "old-name")] = object()
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        provider.adopt_console_preparation(
            preparation, _controller_composition=composition
        )
        assert not provider._stamped_decisions
        replacement = provider_source.capture_standard_controller_composition(
            provider_source.MCPToolProvider, case.service
        )
        with pytest.raises(PermissionError, match="mcp_catalog_operation_changed"):
            provider.adopt_console_preparation(
                preparation, _controller_composition=replacement
            )
    assert not probe.read_threads and not probe.permission_threads
    original_row = next(
        row for row in preparation.tools if row.name == "sample_builtin"
    )
    hub = next(
        hub
        for hub, _ in provider._entry_by_llm_name.values()
        if hub.name == "sample_builtin"
    )
    hub.input_schema["mutated"] = {"nested": [True]}
    assert original_row.to_hub_tool().input_schema == {"type": "object"}
    named = provider_source.MCPToolProvider(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        profile_id_provider=lambda: "other",
    )
    with pytest.raises(PermissionError, match="mcp_catalog_profile_changed"):
        named.adopt_console_preparation(
            preparation, _controller_composition=composition
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_method", ["result-current", "converter"])
async def test_adoption_refuses_replaced_data_methods_without_executing_them(
    snapshot_case,
    monkeypatch,
    changed_method,
):
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.Chat.console_chat_controller import (
        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    from tldw_chatbook.MCP import console_tool_preparation as preparation_source

    case = snapshot_case
    _loop_projection(case)
    preparation = await preparation_source.prepare_console_tools(
        case.service,
        include_mcp_catalog=True,
        need_local_switch=True,
        builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        owned_profile_ids=frozenset(),
    )
    composition = provider_source.capture_standard_controller_composition(
        provider_source.MCPToolProvider, case.service
    )
    provider = provider_source.MCPToolProvider(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    unexpected_calls = []
    if changed_method == "result-current":
        monkeypatch.setattr(
            preparation_source.ConsoleToolPreparation,
            "require_current",
            lambda self, service: unexpected_calls.append("current"),
        )
    else:
        monkeypatch.setattr(
            preparation_source.PreparedConsoleTool,
            "to_hub_tool",
            lambda self: unexpected_calls.append("converter"),
        )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        with pytest.raises(PermissionError, match="mcp_catalog_source_changed"):
            provider.adopt_console_preparation(
                preparation, _controller_composition=composition
            )
    assert provider.list_catalog() == [] and not unexpected_calls
    assert not probe.permission_threads and not probe.read_threads


@pytest.mark.asyncio
async def test_large_valid_schema_retains_full_tools_through_ordinary_fallback(
    snapshot_case,
    record_testsuite_property,
):
    from tldw_chatbook.MCP.client import MAX_SCHEMA_BYTES

    case = snapshot_case
    _loop_projection(case)
    schema = {"type": "object", "$comment": "x" * (MAX_SCHEMA_BYTES + 1)}
    case.source.save_discovery_snapshot(
        "one", {"tools": [{"name": "first", "inputSchema": schema}]}
    )
    context = _configure(
        _context(case),
        mcp_definition_maximum={
            "local:one::first": definition_hash("", schema),
            "builtin:tldw_chatbook::sample_builtin": definition_hash(
                "builtin", {"type": "object"}
            ),
        },
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, _, local, _ = await _compose(case, context)
    assert mcp is not None and local is not None
    assert [hub.name for hub, _ in mcp._entry_by_llm_name.values()] == [
        "first",
        "sample_builtin",
    ]
    first = next(
        entry
        for entry in mcp.list_catalog()
        if mcp._entry_by_llm_name[entry.id][0].name == "first"
    )
    assert mcp.load_schema(first.id).parameters == schema
    record_testsuite_property(
        "fallback_permission_loads", len(probe.permission_threads)
    )
    record_testsuite_property("fallback_catalog_loads", len(probe.read_threads))
    # The declined domain observation and the existing ordinary route are
    # counted separately from the one-load bounded stock acceptance path.
    assert len(probe.permission_threads) == 4
    assert len(probe.read_threads) == 2
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)

@pytest.mark.asyncio
async def test_builtin_only_controller_ceiling_keeps_policy_without_external_read(
    snapshot_case,
):
    case = snapshot_case
    _, inventory_calls = _loop_projection(case)
    context = _configure(
        _context(case),
        mcp_tool_maximum=frozenset({"builtin:tldw_chatbook::sample_builtin"}),
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, builtin_gate, local, review = await _compose(case, context)
    assert mcp is not None and builtin_gate is not None and local is not None
    assert callable(review)
    assert [hub.tool_id for hub, _ in mcp._entry_by_llm_name.values()] == [
        "builtin:tldw_chatbook::sample_builtin"
    ]
    assert len(mcp.list_catalog()) == 1 and mcp.not_connected_count == 0
    assert (
        case.app.console_mcp_tool_count,
        case.app.console_mcp_not_connected_count,
    ) == (
        1,
        0,
    )
    assert inventory_calls == [True]
    assert len(probe.permission_threads) == 1
    assert all(
        thread is not threading.current_thread() for thread in probe.permission_threads
    )
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)
    assert not probe.audit_threads
    assert (
        not probe.read_threads
    ), "builtin-only composition read the excluded external catalog"


@pytest.mark.asyncio
async def test_builtin_only_schema_fallback_keeps_external_catalog_unused(
    snapshot_case,
):
    from tldw_chatbook.MCP.client import MAX_SCHEMA_BYTES

    case = snapshot_case
    schema = {"type": "object", "$comment": "x" * (MAX_SCHEMA_BYTES + 1)}
    inventory_threads = []

    def manifest():
        inventory_threads.append(threading.current_thread())
        return {
            "tools": [
                {
                    "name": "sample_builtin",
                    "description": "builtin",
                    "inputSchema": schema,
                }
            ]
        }

    case.local.manifest_provider = manifest
    context = _configure(
        _context(case),
        mcp_tool_maximum=frozenset({"builtin:tldw_chatbook::sample_builtin"}),
        mcp_definition_maximum={
            "builtin:tldw_chatbook::sample_builtin": definition_hash("builtin", schema)
        },
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        mcp, _, local, _ = await _compose(case, context)
    assert mcp is not None and local is not None
    assert len(mcp.list_catalog()) == 1
    assert mcp.load_schema(mcp.list_catalog()[0].id).parameters == schema
    assert inventory_threads == [threading.current_thread()] * 2
    assert len(probe.permission_threads) == 4
    assert not probe.read_threads
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "stock",
        "unset",
        "set",
        "frozen-subclass",
        "provider-subclass",
        "catalog-callback",
    ],
)
async def test_ordinary_builtin_ceiling_preserves_custom_routes(
    snapshot_case, monkeypatch, kind
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    case = snapshot_case
    _, inventory_calls = _loop_projection(case)
    maximum = frozenset({"builtin:tldw_chatbook::sample_builtin"})
    factory = MCPToolProvider
    catalog_calls = []
    if kind == "unset":
        maximum = None
    elif kind == "set":
        maximum = set(maximum)
    elif kind == "frozen-subclass":

        class CustomMaximum(frozenset):
            pass

        maximum = CustomMaximum(maximum)
    elif kind == "provider-subclass":

        class CustomProvider(MCPToolProvider):
            pass

        factory = CustomProvider
    elif kind == "catalog-callback":
        original = case.service.local_external_catalog

        async def custom_catalog():
            catalog_calls.append(True)
            return await original()

        monkeypatch.setattr(case.service, "local_external_catalog", custom_catalog)
    provider = factory(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        maximum_tool_ids=maximum,
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        await provider.compose_catalog()
    assert [hub.tool_id for hub, _ in provider._entry_by_llm_name.values()] == (
        ["local:one::first", "builtin:tldw_chatbook::sample_builtin"]
        if kind == "unset"
        else []
        if kind == "catalog-callback"
        else ["builtin:tldw_chatbook::sample_builtin"]
    )
    assert len(probe.read_threads) == (0 if kind == "stock" else 1)
    assert len(probe.permission_threads) == 2
    # A custom service keeps the original worker inventory route; this
    # fixture's loop-only governance declines that optional inventory.
    assert inventory_calls == ([] if kind == "catalog-callback" else [True])
    assert catalog_calls == ([True] if kind == "catalog-callback" else [])
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "maximum",
    [
        None,
        frozenset({"local:one::first"}),
        frozenset({"local:one::first", "builtin:tldw_chatbook::sample_builtin"}),
    ],
)
async def test_builtin_only_preparation_cannot_be_adopted_by_external_consumer(
    snapshot_case, maximum
):
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from Tests.MCP.test_console_tool_preparation import _prepare

    case = snapshot_case
    _loop_projection(case)
    preparation = await _prepare(
        case, maximum=frozenset({"builtin:tldw_chatbook::sample_builtin"})
    )
    assert not preparation._includes_external_catalog
    composition = provider_source.capture_standard_controller_composition(
        provider_source.MCPToolProvider, case.service
    )
    provider = provider_source.MCPToolProvider(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        maximum_tool_ids=maximum,
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed(), pytest.raises(
        PermissionError, match="mcp_catalog_maximum_changed"
    ):
        provider.adopt_console_preparation(
            preparation, _controller_composition=composition
        )
    assert not provider.list_catalog()
    assert not probe.permission_threads and not probe.read_threads


@pytest.mark.asyncio
async def test_initial_maximum_stays_fresh_before_builtin_only_composition(
    snapshot_case,
):
    from tldw_chatbook.MCP.console_snapshot import capture_console_definition_maximum

    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state("local:one", "first", "deny")
    initial = _MaximumProbe(case.source, case.permissions)
    with initial.installed():
        maximum = await capture_console_definition_maximum(case.service, frozenset())
    assert set(maximum) == {"builtin:tldw_chatbook::sample_builtin"}
    assert len(initial.read_threads) == len(initial.permission_threads) == 1
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash=definition_hash("", None)
    )
    context = _configure(
        _context(case),
        mcp_tool_maximum=frozenset(maximum),
        mcp_definition_maximum=maximum,
    )
    composition = _MaximumProbe(case.source, case.permissions)
    with composition.installed():
        mcp, _, _, _ = await _compose(case, context)
    assert mcp is not None and len(mcp.list_catalog()) == 1
    assert not composition.read_threads and len(composition.permission_threads) == 1
    following = _MaximumProbe(case.source, case.permissions)
    with following.installed():
        fresh = await capture_console_definition_maximum(case.service, frozenset())
    assert set(fresh) == {"local:one::first", "builtin:tldw_chatbook::sample_builtin"}
    assert len(following.read_threads) == len(following.permission_threads) == 1
    assert all(
        lease not in storage_admission._live_leases
        for probe in (initial, composition, following)
        for lease in probe.leases
    )
