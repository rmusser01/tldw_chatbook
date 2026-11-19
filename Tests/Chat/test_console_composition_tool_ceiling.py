"""Stock execution consumers establish their own fresh, non-widening ceiling."""

import asyncio
from dataclasses import replace
import threading

import pytest

from Tests.Chat import test_console_configuration_capture as capture_controls
from Tests.Chat import test_console_shared_tool_preparation as shared_controls
from Tests.Chat.test_console_async_mcp_snapshot import _MaximumProbe
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.MCP.permission_store import definition_hash

catalog_store = capture_controls.catalog_store
local_root = capture_controls.local_root
mcp_sources = capture_controls.mcp_sources
snapshot_case = capture_controls.snapshot_case
runtime_case = capture_controls.runtime_case


@pytest.fixture
def owned_capture_case(runtime_case):
    case = runtime_case
    # This existing runtime fixture has no persistence repository. Select the
    # ordinary capture route through the public session API, without claiming
    # that this capture-only fixture verifies a saved turn.
    case.session = case.store.create_session(
        title="Ordinary composition capture",
        workspace_id="global",
        settings=case.store.effective_session_settings(case.session.id),
        ephemeral=False,
        activate=False,
    )
    assert not case.store.session_is_ephemeral(case.session.id)
    case.store.set_session_draft(
        case.session.id,
        "Keep this draft until durable acceptance.",
        authored_token=(1, 1),
    )
    return case


@pytest.mark.asyncio
async def test_stock_owned_capture_defers_original_mcp_source_reads(
    owned_capture_case, request
):
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )

    case = owned_capture_case
    assert case.controller._turn_context_provider is None
    assert standard_console_configuration_sources(
        case.app, case.store, case.controller, session_id=case.session.id
    )
    before_inputs = case.store.session_input_snapshot(case.session.id)
    before_leases = set(storage_admission._live_leases)
    before_catalog = case.source.path.read_bytes()
    before_policy = case.permissions.path.read_bytes()
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )

    assert captured.session_id == case.session.id
    assert case.store.active_session_id != case.session.id
    assert captured.provider_selection.provider == "deepseek"
    assert captured.session_settings == case.store.effective_session_settings(
        case.session.id
    )
    assert case.store.session_inputs_are_current(before_inputs)
    assert case.source.path.read_bytes() == before_catalog
    assert case.permissions.path.read_bytes() == before_policy
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)
    assert set(storage_admission._live_leases) == before_leases
    counts = len(probe.permission_threads), len(probe.read_threads)
    request.node.user_properties.extend(
        [
            ("original_initial_permission_reads", counts[0]),
            ("original_initial_catalog_reads", counts[1]),
            ("original_initial_audit_calls", len(probe.audit_threads)),
            ("retired_observed_source_leases", len(probe.leases)),
        ]
    )
    # Original source must fail on actual redundant reads, not a missing API.
    assert counts == (0, 0)
    assert captured.mcp_definition_capture == "composition"
    assert captured.mcp_tool_maximum is None
    assert dict(captured.mcp_definition_maximum) == {}
    assert not probe.audit_threads


def _execution_context(case, captured):
    request = ConsoleTurnCustodyRequest(
        turn_id="composition-ceiling-turn",
        session_id=case.session.id,
        draft=case.session.draft,
        configuration=captured,
    )
    context = replace(
        shared_controls._context(case, local_enabled=False),
        configuration=request.configuration,
    )
    assert context.configuration is not captured
    assert context.mcp_definition_capture == captured.mcp_definition_capture
    return context


def _ids(provider):
    return {tool.tool_id for tool, _state in provider._entry_by_llm_name.values()}


def _provider(case, **kwargs):
    return MCPToolProvider(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        **kwargs,
    )


def _retired(probe):
    assert probe.leases
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)


@pytest.mark.asyncio
async def test_owned_capture_reaches_fresh_composition_and_freezes_only_that_consumer(
    owned_capture_case,
):
    case = owned_capture_case
    case.permissions.set_tool_state("local:one", "first", "deny")
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    assert captured.mcp_definition_capture == "composition"
    context = _execution_context(case, captured)
    assert context.mcp_tool_maximum is None and not context.mcp_definition_maximum
    before_inputs = case.store.session_input_snapshot(case.session.id)

    # Changes before this consumer's composition intentionally belong to it.
    case.source.save_discovery_snapshot(
        "one",
        {
            "tools": [
                {"name": "first", "description": "before composition"},
                {"name": "second"},
                {"name": "denied_at_composition"},
            ]
        },
    )
    case.permissions.set_tool_state("local:one", "denied_at_composition", "deny")
    case.permissions.set_tool_state(
        "local:one",
        "first",
        "allow",
        definition_hash=definition_hash("before composition", None),
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        provider, _, _, _ = await shared_controls._compose(case, context)
    assert provider is not None
    assert _ids(provider) == {
        "local:one::first",
        "local:one::second",
        "builtin:tldw_chatbook::sample_builtin",
    }
    assert len(probe.permission_threads) == len(probe.read_threads) == 1
    _retired(probe)
    first = next(
        entry.id
        for entry in provider.list_catalog()
        if provider._entry_by_llm_name[entry.id][0].name == "first"
    )
    assert provider.load_schema(first).description == "before composition"

    # A denial narrows; new or redefined rows cannot replace this issued pair.
    case.permissions.set_tool_state("local:one", "first", "deny")
    case.permissions.set_tool_state(
        "local:one",
        "denied_at_composition",
        "allow",
        definition_hash=definition_hash("", None),
    )
    case.source.save_discovery_snapshot(
        "one",
        {
            "tools": [
                {"name": "first", "description": "before composition"},
                {"name": "second", "description": "after publication"},
                {"name": "denied_at_composition"},
                {"name": "third"},
            ]
        },
    )
    await provider.compose_catalog()
    assert _ids(provider) == {"builtin:tldw_chatbook::sample_builtin"}
    assert case.store.session_inputs_are_current(before_inputs)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["empty", "killed"])
async def test_successful_empty_provider_ceiling_cannot_reopen(snapshot_case, initial):
    case = snapshot_case
    shared_controls._loop_projection(case)
    if initial == "killed":
        case.permissions.set_kill_switch(True)
    else:
        case.permissions.set_global_default("deny")
    provider = _provider(case, _freeze_definition_maximum_on_compose=True)
    await provider.compose_catalog()
    assert provider.list_catalog() == []
    case.permissions.set_kill_switch(False)
    case.permissions.set_global_default("ask")
    case.source.save_discovery_snapshot("one", {"tools": [{"name": "newly_visible"}]})
    await provider.compose_catalog()
    assert provider.list_catalog() == []
    # A distinct execution consumer is deliberately fresh, not attempt-cached.
    following = _provider(case, _freeze_definition_maximum_on_compose=True)
    await following.compose_catalog()
    assert _ids(following) == {
        "local:one::newly_visible",
        "builtin:tldw_chatbook::sample_builtin",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [None, frozenset()])
async def test_legacy_none_and_explicit_empty_keep_their_meaning(snapshot_case, bound):
    case = snapshot_case
    shared_controls._loop_projection(case)
    provider = _provider(case, maximum_tool_ids=bound)
    await provider.compose_catalog()
    assert bool(provider.list_catalog()) is (bound is None)
    case.source.save_discovery_snapshot("one", {"tools": [{"name": "later"}]})
    await provider.compose_catalog()
    assert _ids(provider) == (
        {"local:one::later", "builtin:tldw_chatbook::sample_builtin"}
        if bound is None
        else set()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["kill", "deny"])
async def test_composition_ceiling_never_supplies_invocation_permission(
    snapshot_case, change
):
    from tldw_chatbook.Agents.run_context import use_run_id
    from tldw_chatbook.Agents.mcp_tool_provider import DENY_REFUSAL
    from tldw_chatbook.Agents.tool_refusals import TOOL_KILL_SWITCH_REFUSAL

    case = snapshot_case
    shared_controls._loop_projection(case)
    provider = _provider(case, _freeze_definition_maximum_on_compose=True)
    await provider.compose_catalog()
    entry = next(
        entry
        for entry in provider.list_catalog()
        if provider._entry_by_llm_name[entry.id][0].tool_id == "local:one::first"
    )
    if change == "kill":
        case.permissions.set_kill_switch(True)
    else:
        case.permissions.set_tool_state("local:one", "first", "deny")

    def invoke():
        with use_run_id("composition-ceiling-fresh-invocation"):
            return provider.invoke(entry.id, {})

    result = await asyncio.to_thread(invoke)
    assert result.ok is False
    if change == "kill":
        assert result.error == TOOL_KILL_SWITCH_REFUSAL
    else:
        assert result.error == DENY_REFUSAL
    assert result.dispatch_state == "not_started"


@pytest.mark.asyncio
async def test_preview_and_live_consumers_have_separate_fresh_ceilings(
    owned_capture_case,
):
    case = owned_capture_case
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    context = _execution_context(case, captured)
    case.app.console_mcp_tool_count, case.app.console_mcp_not_connected_count = 7, 4
    preview, _, _, _ = await shared_controls._compose(
        case, context, publish_mcp_counts=False
    )
    assert preview is not None and "local:one::first" in _ids(preview)
    assert (
        case.app.console_mcp_tool_count,
        case.app.console_mcp_not_connected_count,
    ) == (7, 4)
    case.source.save_discovery_snapshot("one", {"tools": [{"name": "live_only"}]})
    live, _, _, _ = await shared_controls._compose(case, context)
    assert live is not None and live is not preview
    assert "local:one::live_only" in _ids(live)
    assert "local:one::live_only" not in _ids(preview)
    assert case.app.console_mcp_tool_count == len(live.list_catalog())


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ephemeral", "no_service"])
async def test_excluded_capture_is_explicitly_empty(
    owned_capture_case, monkeypatch, route
):
    case = owned_capture_case
    if route == "ephemeral":
        case.session = case.store.create_session(
            title="Temporary",
            ephemeral=True,
            activate=False,
            settings=case.store.effective_session_settings(case.session.id),
        )
    else:
        monkeypatch.setattr(case.app, "unified_mcp_service", None)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
    assert captured.mcp_definition_capture == "captured"
    assert captured.mcp_tool_maximum == frozenset()
    assert dict(captured.mcp_definition_maximum) == {}
    assert (
        not probe.permission_threads
        and not probe.read_threads
        and not probe.audit_threads
    )


@pytest.mark.asyncio
async def test_custom_initial_capture_keeps_its_original_sync_route(owned_capture_case):
    case = owned_capture_case
    original = case.controller.resolve_runtime_turn_configuration_snapshot
    owner_loop, owner_thread = asyncio.get_running_loop(), threading.current_thread()
    calls = []

    def custom(session_id):
        calls.append(
            (session_id, asyncio.get_running_loop(), threading.current_thread())
        )
        return original(session_id)

    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id, context_provider=custom
        )
    assert calls == [(case.session.id, owner_loop, owner_thread)]
    assert captured.mcp_definition_capture == "captured"
    assert "local:one::first" in captured.mcp_definition_maximum
    assert probe.permission_threads and probe.read_threads
    assert all(
        thread is owner_thread
        for thread in probe.permission_threads + probe.read_threads
    )
    _retired(probe)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["factory", "catalog"])
async def test_deferred_capture_refuses_late_unsupported_custom_route(
    owned_capture_case, monkeypatch, change
):
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    case = owned_capture_case
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    context = _execution_context(case, captured)
    calls = []

    def factory(*args, **kwargs):
        calls.append("factory")
        raise AssertionError("deferred ceiling reached an unsupported factory")

    async def catalog():
        calls.append("catalog")
        raise AssertionError("deferred ceiling reached an unsupported catalog")

    if change == "factory":
        monkeypatch.setattr(controller_module, "MCPToolProvider", factory)
    else:
        monkeypatch.setattr(case.service, "local_external_catalog", catalog)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed(), pytest.raises((RecoveryRequired, PermissionError)):
        await shared_controls._compose(case, context)
    assert calls == []
    assert not probe.permission_threads and not probe.read_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["method", "service"])
async def test_prospective_hook_refuses_late_custom_composition_before_call(
    owned_capture_case, monkeypatch, change
):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle

    case = owned_capture_case
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    assert captured.mcp_definition_capture == "composition"
    engine = HookEngine((), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, case.session.id)
    scope = lifecycle.open_scope()
    lifecycle.turn_scope = scope
    task = asyncio.current_task()
    assert task is not None and not task.cancelling()
    reservation = (lifecycle, scope, case.session.id)
    assert not getattr(case.controller, "_hooks_v2_submissions", {})
    monkeypatch.setattr(
        case.controller, "_hooks_v2_submissions", {task: reservation}, raising=False
    )
    calls = []

    async def custom(*args, **kwargs):
        calls.append(True)
        raise AssertionError("prospective hook invoked changed composition callback")

    if change == "method":
        monkeypatch.setattr(case.controller, "_compose_mcp_provider", custom)
    else:
        monkeypatch.setattr(case.app, "unified_mcp_service", None)
    probe = _MaximumProbe(case.source, case.permissions)
    try:
        assert lifecycle.current() and lifecycle.checkpoints.is_current(scope)
        assert case.controller._hooks_v2_submissions[task] == reservation
        with probe.installed(), pytest.raises(
            PermissionError, match="^mcp_catalog_source_changed$"
        ):
            await case.controller.compose_prospective_hook_context(
                captured, lifecycle, scope
            )
        assert calls == []
        assert not probe.permission_threads and not probe.read_threads
        assert not probe.leases
    finally:
        case.controller._hooks_v2_submissions.pop(task, None)
        lifecycle.close_scope(scope)
        lifecycle.turn_scope = None
        lifecycle.seal()
        await engine.close()
    assert not case.controller._hooks_v2_submissions


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "source"])
async def test_unpublished_composition_drains_native_work_without_freezing(
    snapshot_case, tmp_path, failure
):
    case = snapshot_case
    shared_controls._loop_projection(case)
    provider = _provider(case, _freeze_definition_maximum_on_compose=True)
    old_path = case.source.path
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        task = asyncio.create_task(provider.compose_catalog())
        try:
            await capture_controls.catalog_controls._worker_entered(probe, task)
            assert probe.leases and all(
                lease in storage_admission._live_leases for lease in probe.leases
            )
            if failure == "cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            else:
                case.source.path = tmp_path / "unadmitted-catalog.json"
            probe.release.set()
            with pytest.raises(
                asyncio.CancelledError
                if failure == "cancel"
                else (RecoveryRequired, PermissionError)
            ):
                await task
            assert provider.list_catalog() == []
            _retired(probe)
            if failure == "source":
                assert not case.source.path.exists()
        finally:
            probe.release.set()
            await capture_controls.catalog_controls._settle(task, probe)
            case.source.path = old_path
    case.source.save_discovery_snapshot(
        "one", {"tools": [{"name": "after_failed_publication"}]}
    )
    await provider.compose_catalog()
    assert "local:one::after_failed_publication" in _ids(provider)


@pytest.mark.parametrize(
    "changes",
    [
        {"mcp_definition_capture": "later"},
        {"mcp_definition_capture": True},
        {"mcp_definition_capture": "composition", "mcp_tool_maximum": frozenset()},
        {
            "mcp_definition_capture": "composition",
            "mcp_definition_maximum": {"local:one::first": "hash"},
        },
    ],
)
def test_deferred_mode_rejects_ambiguous_or_already_bound_shapes(changes):
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot

    original = ConsoleTurnConfigurationSnapshot.capture(
        session_id="ceiling-shape",
        provider_selection=ConsoleProviderSelection(provider="deepseek"),
    )
    assert original.mcp_definition_capture == "captured"
    with pytest.raises((TypeError, ValueError)):
        replace(original, **changes)
