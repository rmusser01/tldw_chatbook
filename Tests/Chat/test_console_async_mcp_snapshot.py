"""Actual native MCP preparation for loop-owned Console turn snapshots."""

import asyncio
import copy
from dataclasses import replace
import inspect
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.MCP import test_external_catalog_worker_ownership as catalog_controls
from Tests.Chat.test_console_turn_execution_context import ConsoleChatStore
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.MCP.hub_tool_catalog import local_tools_from_record
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

mcp_sources = catalog_controls.mcp_sources
local_root = catalog_controls.local_root
catalog_store = catalog_controls.catalog_store


class _MaximumProbe(catalog_controls._ReadProbe):
    def __init__(self, store, permissions, **kwargs):
        super().__init__(store, **kwargs)
        self.permissions = permissions
        # Both the public and owned routes execute this original payload body.
        self.permission_code = inspect.unwrap(MCPPermissionStore._load_locked).__code__
        self.permission_threads = []
        self.permission_calls = []
        self.permission_callers = []
        self.audit_threads = []
        self.audit_code = (
            UnifiedMCPControlPlaneService._audit_downgrade_if_fresh.__code__
        )

    def observe(self, frame, event, arg):
        if event == "call":
            if frame.f_code is self.permission_code:
                self.permission_calls.append(
                    (frame.f_locals.get("self"), threading.current_thread())
                )
                caller = frame.f_back
                path = []
                while caller is not None and len(path) < 12:
                    path.append(
                        (
                            caller.f_code.co_name,
                            caller.f_code.co_filename,
                            caller.f_code.co_firstlineno,
                        )
                    )
                    caller = caller.f_back
                self.permission_callers.append(path)
            if (
                frame.f_code is self.permission_code
                and frame.f_locals.get("self") is self.permissions
            ):
                self.permission_threads.append(threading.current_thread())
            if frame.f_code is self.audit_code:
                self.audit_threads.append(threading.current_thread())
        super().observe(frame, event, arg)


@pytest.fixture
def snapshot_case(catalog_store, mcp_sources, monkeypatch):
    from tldw_chatbook import config

    # The source fixture installs a new real module. Imported consumer aliases
    # must point to its original installed functions, never bypass old guards.
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("tldw_chatbook.") or module is config:
            continue
        for alias, value in tuple(vars(module).items()):
            if (
                callable(value)
                and getattr(value, "__module__", None) == config.__name__
            ):
                original = getattr(config, getattr(value, "__name__", ""), None)
                if callable(original):
                    monkeypatch.setattr(module, alias, original)
    permissions = mcp_sources[3]
    permissions.set_kill_switch(True)
    permissions.set_kill_switch(False)
    store = ConsoleChatStore()
    session = store.create_session(title="Native snapshot", workspace_id="global")
    local = LocalMCPControlService(store=catalog_store, manifest_provider=lambda: {})
    service = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=local,
        server_service=None,
    )
    service._permission_store = permissions
    app = SimpleNamespace(unified_mcp_service=service)
    selection = ConsoleProviderSelection(
        provider="deepseek", explicit_model="deepseek-chat"
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(),
        provider="deepseek",
        model="deepseek-chat",
        provider_config=lambda: {},
    )
    controller.app = app
    return SimpleNamespace(
        controller=controller,
        session=session,
        store=store,
        local=local,
        service=service,
        permissions=permissions,
        source=catalog_store,
        app=app,
        selection=selection,
    )


def _loop_projection(case):
    loop = asyncio.get_running_loop()
    thread = threading.current_thread()
    governance = catalog_controls._LoopGovernance()
    case.local.policy_enforcer = governance
    observed = []

    def inventory():
        assert asyncio.get_running_loop() is loop
        assert threading.current_thread() is thread
        observed.append(True)
        return {
            "tools": [
                {
                    "name": "sample_builtin",
                    "description": "builtin",
                    "inputSchema": {"type": "object"},
                }
            ]
        }

    case.local.manifest_provider = inventory
    return governance, observed


@pytest.mark.asyncio
async def test_queue_capture_reads_native_mcp_sources_off_caller_loop(snapshot_case):
    case = snapshot_case
    _loop_projection(case)
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        await case.controller.queue_prompt(
            case.session.id, text="queued message", expected_revision=0
        )
    assert probe.read_threads and probe.permission_threads
    assert all(
        thread is not threading.current_thread()
        for thread in probe.read_threads + probe.permission_threads
    ), "actual queue caller still blocks on native MCP source reads"
    assert len(probe.read_threads) == len(probe.permission_threads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("screen", [False, True])
async def test_async_snapshot_joins_sources_and_composes_on_actual_loop(
    snapshot_case, screen
):
    case = snapshot_case
    _, observed = _loop_projection(case)
    if screen:
        owner = ConsoleSessionController.__new__(ConsoleSessionController)
        owner.app_instance = case.app
        owner._provider_readiness_app_config_fn = lambda: {}
        owner._build_provider_selection_fn = lambda session_id: case.selection
        owner._current_chat_store_accessor = lambda: case.store
        owner._chat_store_accessor = lambda: case.store
        owner._rag_source_types_accessor = lambda: []
        owner._rag_top_k_accessor = lambda: 4
        owner._scratch_snapshot_provider = lambda session_id: None
        case.controller._turn_context_provider = (
            owner._build_console_turn_execution_context
        )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
    assert captured.session_id == case.session.id
    assert captured.provider_selection.provider == "deepseek"
    assert set(captured.mcp_definition_maximum) == {
        "local:one::first",
        "builtin:tldw_chatbook::sample_builtin",
    }
    assert len(probe.read_threads) == len(probe.permission_threads) == 1
    assert all(
        thread is not threading.current_thread()
        for thread in probe.read_threads + probe.permission_threads
    )
    assert observed == [True]
    assert probe.leases and all(
        lease not in storage_admission._live_leases for lease in probe.leases
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["settings", "session", "service", "configuration", "app_config"]
)
async def test_async_snapshot_refuses_changed_actor_before_publication(
    snapshot_case, change
):
    case = snapshot_case
    case.app.app_config = {}
    _loop_projection(case)
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        task = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            if change == "settings":
                settings = case.store.effective_session_settings(case.session.id)
                case.store.replace_session_settings(
                    case.session.id,
                    replace(
                        settings
                        or ConsoleSessionSettings(
                            provider="deepseek", model="deepseek-chat"
                        ),
                        temperature=0.15,
                    ),
                )
            elif change == "session":
                case.store._sessions[case.session.id] = copy.copy(case.session)
            elif change == "service":
                case.app.unified_mcp_service = SimpleNamespace()
            elif change == "app_config":
                case.app.app_config = {}
            else:
                from tldw_chatbook import config

                assert config.save_setting_to_cli_config(
                    "console", "native_tool_calls", False
                )
            probe.release.set()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await task
        finally:
            await catalog_controls._settle(task, probe)


@pytest.mark.asyncio
async def test_async_snapshot_wrong_source_during_read_freezes_empty_maximum(
    snapshot_case, tmp_path
):
    case = snapshot_case
    _loop_projection(case)
    old_path = case.source.path
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        task = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            case.source.path = tmp_path / "unadmitted.json"
            probe.release.set()
            captured = await task
            assert not captured.mcp_definition_maximum
            assert not case.source.path.exists()
            assert probe.leases and all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            case.source.path = old_path
            await catalog_controls._settle(task, probe)


@pytest.mark.asyncio
async def test_async_snapshot_preserves_default_downgrade_audit_and_fresh_gate(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash="0" * 64
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
    assert "local:one::first" in captured.mcp_definition_maximum
    entry = case.permissions.get_tool_entry("local:one", "first", profile_id="default")
    assert entry["config_changed"] is True
    assert (
        len(probe.audit_threads) == 1
        and probe.audit_threads[0] is not threading.current_thread()
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
    case.permissions.set_tool_state("local:one", "first", "deny", profile_id="default")
    tool = local_tools_from_record(case.local.get_external_servers()[0])[0]
    assert case.service.gate_tool_test(tool).state == "deny"
    assert "local:one::first" in captured.mcp_definition_maximum
    later = await case.controller.capture_turn_configuration_snapshot(case.session.id)
    assert "local:one::first" not in later.mcp_definition_maximum


@pytest.mark.asyncio
async def test_async_snapshot_cancel_drains_real_producer_and_native_read(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        task = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            case.service._maintenance_close_admission()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not await case.service._maintenance_drain(time.monotonic() + 0.03)
            assert probe.leases and all(
                lease in storage_admission._live_leases for lease in probe.leases
            )
            probe.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await case.service._maintenance_drain(time.monotonic() + 0.5)
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(task, probe)
            if case.service._producer_lifetime.closed:
                case.service._maintenance_resume()


@pytest.mark.asyncio
async def test_custom_injected_snapshot_provider_preserves_sync_contract(snapshot_case):
    case = snapshot_case
    original = case.controller.resolve_turn_configuration_snapshot(case.session.id)
    calls = []

    def custom(session_id):
        assert session_id == case.session.id
        calls.append(True)
        return original

    case.controller._turn_context_provider = custom
    del case.controller.app
    with _MaximumProbe(case.source, case.permissions).installed() as probe:
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
    assert captured is original and calls == [True]
    assert not probe.read_threads and not probe.permission_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["cold", "kill", "corrupt"])
async def test_async_snapshot_preserves_original_permission_source_semantics(
    snapshot_case, state
):
    case = snapshot_case
    _, observed = _loop_projection(case)
    if state == "cold":
        case.service._permission_store = None
    elif state == "kill":
        case.permissions.set_kill_switch(True)
    else:
        case.permissions.path.write_bytes(b"{")
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        snapshot = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
    assert len(probe.permission_calls) == 1
    assert all(
        thread is not threading.current_thread() for _, thread in probe.permission_calls
    )
    if state == "kill":
        assert (
            not snapshot.mcp_definition_maximum
            and not probe.read_threads
            and not observed
        )
    else:
        assert "local:one::first" in snapshot.mcp_definition_maximum
        assert len(probe.read_threads) == 1 and observed == [True]
        if state == "cold":
            assert case.service._permission_store is not case.permissions
            assert case.service._permission_store.path == case.permissions.path
        else:
            assert case.permissions.get_global_default() == "ask"


@pytest.mark.asyncio
async def test_async_snapshot_permission_retarget_after_read_never_redirects(
    snapshot_case, tmp_path
):
    case = snapshot_case
    _loop_projection(case)
    old_path = case.permissions.path
    replacement = tmp_path / "other-permissions.json"
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        pending = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            await catalog_controls._worker_entered(probe, pending)
            case.permissions.path = replacement
            probe.release.set()
            snapshot = await pending
            assert not snapshot.mcp_definition_maximum
            assert not replacement.exists()
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            case.permissions.path = old_path
            await catalog_controls._settle(pending, probe)


@pytest.mark.asyncio
async def test_standard_provider_inventory_governance_stays_on_actual_loop(
    snapshot_case,
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    case = snapshot_case
    _, observed = _loop_projection(case)
    provider = MCPToolProvider(
        service=case.service, main_loop=asyncio.get_running_loop()
    )
    await provider.compose_catalog()
    assert observed == [
        True
    ], "actual manifest and governance were moved off their owner loop"
    assert any("sample_builtin" in row.name for row in provider.list_catalog())


@pytest.mark.asyncio
async def test_actual_async_local_composer_reads_switch_off_loop(snapshot_case):
    from Tests.Chat.test_console_local_review_hook import _test_execution_context

    case = snapshot_case

    async def no_external_catalog(*args, **kwargs):
        return None

    case.controller._compose_mcp_provider = no_external_catalog
    context = _test_execution_context(
        case.controller._scratch_spaces.snapshot(case.session.id),
        session_id=case.session.id,
        tool_configuration={"local_tools_enabled": True},
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        _, _, local, _ = await case.controller._compose_agent_request_providers(
            session_id=case.session.id,
            project_selection=None,
            project_authority_guard=None,
            turn_context=context,
            admitted_roots=(),
        )
    assert local is not None
    assert len(probe.permission_threads) == 1
    assert (
        probe.permission_threads[0] is not threading.current_thread()
    ), "actual async local composer blocks on native permission state"


class _PermissionProbe(_MaximumProbe):
    """Passive barrier in the original installed permission-reader body."""

    def observe(self, frame, event, arg):
        super().observe(frame, event, arg)
        if (
            event != "call"
            or frame.f_code is not self.permission_code
            or frame.f_locals.get("self") is not self.permissions
        ):
            return
        from tldw_chatbook.Backup_Recovery import raw_participants

        thread = threading.current_thread()
        self.read_threads.append(thread)
        with storage_admission._changed:
            for state in tuple(raw_participants._states.values()):
                if state.source is self.permissions:
                    self.leases.extend(state.leases)
        self.entered.set()
        if thread is not self.loop_thread:
            assert self.release.wait(8)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancel", "path", "session", "app_config"])
async def test_local_async_composer_retains_native_custody_and_refuses_changed_owner(
    snapshot_case, tmp_path, change
):
    from Tests.Chat.test_console_local_review_hook import _test_execution_context

    case = snapshot_case
    case.app.app_config = {}
    context = _test_execution_context(
        case.controller._scratch_spaces.snapshot(case.session.id),
        session_id=case.session.id,
        tool_configuration={"local_tools_enabled": True},
    )
    old_path = case.permissions.path
    replacement = tmp_path / "redirected-permissions.json"
    probe = _PermissionProbe(case.source, case.permissions)
    with probe.installed():
        task = asyncio.create_task(
            case.controller._compose_local_provider_async(
                session_id=case.session.id,
                turn_context=context,
                admitted_roots=(),
            )
        )
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
            elif change == "path":
                case.permissions.path = replacement
            elif change == "app_config":
                case.app.app_config = {}
            else:
                case.store._sessions[case.session.id] = copy.copy(case.session)
            probe.release.set()
            if change == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert await case.service._maintenance_drain(time.monotonic() + 0.5)
            elif change in {"session", "app_config"}:
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="console_snapshot_owner_changed"
                ):
                    await task
            else:
                assert await task == (None, None)
                assert not replacement.exists()
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            case.permissions.path = old_path
            await catalog_controls._settle(task, probe)
            if case.service._producer_lifetime.closed:
                case.service._maintenance_resume()


@pytest.mark.asyncio
async def test_standard_provider_cancel_retains_whole_composition_until_native_retirement(
    snapshot_case,
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    case = snapshot_case
    _loop_projection(case)
    provider = MCPToolProvider(
        service=case.service, main_loop=asyncio.get_running_loop()
    )
    probe = _PermissionProbe(case.source, case.permissions)
    with probe.installed():
        task = asyncio.create_task(provider.compose_catalog())
        try:
            await catalog_controls._worker_entered(probe, task)
            case.service._maintenance_close_admission()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not await case.service._maintenance_drain(time.monotonic() + 0.03)
            probe.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await case.service._maintenance_drain(time.monotonic() + 0.5)
            assert not provider.list_catalog()
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(task, probe)
            if case.service._producer_lifetime.closed:
                case.service._maintenance_resume()


@pytest.fixture
def wired_case(snapshot_case, monkeypatch):
    """Construct the real ChatScreen and its production controller wiring."""
    from Tests.UI import app_factory
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monkeypatch.setattr(app_factory, "load_settings", config.load_settings)
    case = snapshot_case
    app = app_factory._build_test_app(
        config_overrides={"console": {"agent_runtime": False}}
    )
    app.unified_mcp_service = case.service
    screen = ChatScreen(app)
    screen._console_chat_store = case.store
    screen._console_chat_controller = case.controller
    case.controller.app = app
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(provider="deepseek", model="deepseek-chat"),
    )
    screen._console_visible_draft_session_id = case.session.id
    case.screen = screen
    case.app = app
    return case


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["normal", "queued"])
async def test_actual_screen_wiring_prepares_native_mcp_sources_off_ui_loop(
    wired_case, route
):
    case = wired_case
    _loop_projection(case)
    queue = case.screen._prompt_queue
    probe = _MaximumProbe(case.source, case.permissions)
    runtime = case.screen._console_runtime()
    try:
        with probe.installed():
            if route == "normal":
                launch = (
                    getattr(queue, "_launch_chain_async", None) or queue._launch_chain
                )
                result = launch("actual wired input", case.session.id)
                if inspect.isawaitable(result):
                    await result
            else:
                capture_async = getattr(
                    queue, "_capture_configuration_for_dispatch", None
                )
                if capture_async is not None:
                    await capture_async(case.session.id)
                else:
                    queue._capture_configuration(case.session.id)
    finally:
        # Custody is real, but this test ends before running the unrelated send.
        # Cancel and drain its actual owned tasks rather than abandoning a native worker.
        tasks = tuple(
            record.task for record in runtime._turn_custody.values() if record.task
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose(timeout_seconds=3)
    assert probe.read_threads and probe.permission_threads
    assert all(
        thread is not threading.current_thread()
        for thread in probe.read_threads + probe.permission_threads
    ), "production ChatScreen wiring still captures MCP sources synchronously before runtime custody"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "cancel",
        "draft",
        "prefill",
        "evidence",
        "session_controller",
        "app_config",
        "stash",
        "attachment",
        "builder",
    ],
)
async def test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss(
    wired_case, change
):
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

    case = wired_case
    _loop_projection(case)
    runtime = case.screen._console_runtime()
    case.store.set_session_draft(case.session.id, "owning draft")
    case.store.set_session_one_shot_prefill(case.session.id, "prefill")
    stash = ConsoleDraftStash(segments=[], text="owning draft", has_paste=False)
    queue = case.screen._prompt_queue
    if change == "attachment":
        from tldw_chatbook.Chat.attachment_core import PendingAttachment

        attachment = PendingAttachment(
            file_path="capture.txt",
            display_name="capture.txt",
            file_type="text",
            insert_mode="attachment",
            text_content="captured text",
        )
        assert case.store.add_pending_attachment(case.session.id, attachment)
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        task = asyncio.create_task(
            queue._launch_chain_async(
                "owning draft", case.session.id, stash, case.controller
            )
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            assert not runtime.has_custodied_turns()
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
            elif change == "draft":
                case.store.set_session_draft(case.session.id, "newer draft")
            elif change == "prefill":
                case.store.set_session_one_shot_prefill(
                    case.session.id, "newer prefill"
                )
            elif change == "evidence":
                from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch

                runtime.stage_console_staged_evidence(
                    ConsoleLiveWorkLaunch(
                        source="notes", title="replacement", payload={}
                    )
                )
            elif change == "session_controller":
                case.screen._session = copy.copy(case.screen._session)
            elif change == "app_config":
                case.app.app_config = copy.deepcopy(case.app.app_config)
            elif change == "attachment":
                attachment.attachment_id = "changed-attachment-owner"
            elif change == "builder":

                class EqualBuilder:
                    def __eq__(self, other):
                        return True

                    def __call__(self, *args, **kwargs):
                        raise AssertionError("replacement builder must not execute")

                case.screen._session._build_console_turn_execution_context = (
                    EqualBuilder()
                )
            else:
                stash.edit_serial += 1
            probe.release.set()
            expected = (
                asyncio.CancelledError
                if change == "cancel"
                else bootstrap.RecoveryRequired
            )
            with pytest.raises(expected):
                await task
            assert not runtime.has_custodied_turns()
            assert case.store.session_draft(case.session.id) == (
                "newer draft" if change == "draft" else "owning draft"
            )
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(task, probe)
            if case.service._producer_lifetime.closed:
                case.service._maintenance_resume()
            runtime.stage_console_staged_evidence(None)
            await runtime.dispose(timeout_seconds=3)


@pytest.mark.asyncio
async def test_actual_queued_capture_rejects_replaced_controller_before_queue_or_draft_commit(
    wired_case, monkeypatch
):
    from Tests.UI.test_console_prompt_queue import _activity
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptDispatchStatus,
    )
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

    case = wired_case
    _loop_projection(case)
    runtime = case.screen._console_runtime()
    queue = case.screen._prompt_queue
    case.store.set_session_draft(case.session.id, "queue draft")
    stash = ConsoleDraftStash(segments=[], text="queue draft", has_paste=False)
    committed = []
    monkeypatch.setattr(queue, "_blocked_reason_accessor", lambda: "")
    monkeypatch.setattr(queue, "_notify", lambda *args: None)
    monkeypatch.setattr(
        queue, "_commit_queued_draft", lambda *args: committed.append(args)
    )
    monkeypatch.setattr(
        case.controller, "activity_for", lambda sid: _activity(sid, accepted=True)
    )
    before = case.controller.prompt_queue_registry.snapshot(case.session.id)
    # Hold the first actual admitted permission body. The owner fence applies
    # throughout preparation; a later store-body deadline included a complete
    # preceding native permission phase and was not an owner-retarget witness.
    probe = _PermissionProbe(case.source, case.permissions)
    with probe.installed():
        task = asyncio.create_task(
            queue.dispatch("queue draft", session_id=case.session.id, stash=stash)
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            assert probe.permission_calls[0][0] is case.permissions
            replacement = copy.copy(case.controller)
            runtime.set_chat_controller(replacement)
            probe.release.set()
            result = await task
            assert result.status is ConsolePromptDispatchStatus.REFUSED
            assert (
                result.detail
                == "Chat or settings changed while preparing Send. Your draft was kept; send again."
            )
            assert (
                case.controller.prompt_queue_registry.snapshot(case.session.id)
                == before
            )
            assert case.store.session_draft(case.session.id) == "queue draft"
            assert stash.text == "queue draft" and not committed
            assert not runtime.has_custodied_turns()
            assert probe.leases and all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(task, probe)
            runtime.set_chat_controller(case.controller)
            await runtime.dispose(timeout_seconds=3)


@pytest.mark.asyncio
async def test_async_maximum_rejects_replaced_inventory_callback_during_native_read(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    _loop_projection(case)
    invoked = []
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        task = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            await catalog_controls._worker_entered(probe, task)

            def replacement():
                invoked.append(True)
                return []

            monkeypatch.setattr(case.local, "get_inventory", replacement)
            probe.release.set()
            context = await task
            assert (
                not invoked
            ), "a replaced service callback executed after native worker acceptance"
            assert not context.mcp_definition_maximum
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(task, probe)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["binding", "governance"])
async def test_async_maximum_rejects_field_equal_source_owner_replacement(
    snapshot_case, change
):
    from tldw_chatbook.Backup_Recovery import mcp_source_participants

    class EqualGovernance(catalog_controls._LoopGovernance):
        def __eq__(self, other):
            return (
                type(other) is type(self)
                and self.loop is other.loop
                and self.thread is other.thread
            )

    case = snapshot_case
    _loop_projection(case)
    governance = EqualGovernance()
    case.local.policy_enforcer = governance
    binding = mcp_source_participants._BINDINGS[case.source]
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        task = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            if change == "binding":
                replacement = replace(binding)
                assert replacement == binding and replacement is not binding
                mcp_source_participants._BINDINGS[case.source] = replacement
            else:
                replacement = copy.copy(governance)
                assert replacement == governance and replacement is not governance
                case.local.policy_enforcer = replacement
            probe.release.set()
            captured = await task
            assert (
                not captured.mcp_definition_maximum
            ), "field-equal replacement was accepted as the original source owner"
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(task, probe)
            mcp_source_participants._BINDINGS[case.source] = binding
            case.local.policy_enforcer = governance


@pytest.mark.asyncio
@pytest.mark.parametrize("callback", ["mark_config_changed", "append"])
async def test_custom_audit_callbacks_keep_original_caller_loop_contract(
    snapshot_case, mcp_sources, monkeypatch, callback
):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash="0" * 64
    )
    case.service._execution_log = mcp_sources[4]
    owner = (
        case.permissions
        if callback == "mark_config_changed"
        else case.service._execution_log
    )
    original = getattr(owner, callback)
    thread = threading.current_thread()
    loop = asyncio.get_running_loop()
    seen = []

    def caller_owned(*args, **kwargs):
        seen.append(threading.current_thread())
        assert threading.current_thread() is thread
        assert asyncio.get_running_loop() is loop
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, callback, caller_owned)
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    assert captured.mcp_definition_maximum
    assert seen and all(
        item is thread for item in seen
    ), "custom caller-owned audit callback moved into native worker"
    assert (
        case.permissions.get_tool_entry("local:one", "first")["config_changed"] is True
    )


@pytest.mark.parametrize(
    "method",
    [
        "inventory",
        "permission_load",
        "permission_marker",
        "log_append",
        "catalog",
        "kill",
    ],
)
def test_standard_source_qualification_rejects_borrowed_actual_method_receiver(
    snapshot_case, mcp_sources, monkeypatch, method
):
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    case = snapshot_case
    case.service._execution_log = mcp_sources[4]
    assert standard_console_sources(case.service)
    if method == "inventory":
        owner = case.local
        other = LocalMCPControlService(store=case.source, manifest_provider=lambda: {})
        name = "get_inventory"
    elif method == "permission_load" or method == "permission_marker":
        owner = case.permissions
        other = MCPPermissionStore(case.permissions.path)
        name = "load" if method == "permission_load" else "mark_config_changed"
    elif method == "log_append":
        owner = case.service._execution_log
        other = type(owner)(owner.path)
        name = "append"
    elif method == "catalog":
        owner = case.source
        other = type(owner)(owner.path)
        name = "get_catalog_bundle"
    else:
        owner = case.service
        other = UnifiedMCPControlPlaneService(
            target_store=None,
            context_store=None,
            local_service=case.local,
            server_service=None,
        )
        other._permission_store = case.permissions
        name = "get_kill_switch"
    original = getattr(owner, name)
    borrowed = getattr(other, name)
    assert borrowed.__func__ is original.__func__
    assert borrowed.__self__ is not owner
    monkeypatch.setattr(owner, name, borrowed)
    assert not standard_console_sources(
        case.service
    ), "a different actual method receiver qualified as the captured standard owner"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["inventory", "governance", "binding"])
async def test_standard_provider_retains_original_owners_across_catalog_await(
    snapshot_case, monkeypatch, change
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.Backup_Recovery import mcp_source_participants

    class EqualGovernance(catalog_controls._LoopGovernance):
        def __eq__(self, other):
            return type(other) is type(self) and self.loop is other.loop

    case = snapshot_case
    _loop_projection(case)
    governance = EqualGovernance()
    case.local.policy_enforcer = governance
    binding = mcp_source_participants._BINDINGS[case.source]
    replacement_calls = []
    provider = MCPToolProvider(
        service=case.service, main_loop=asyncio.get_running_loop()
    )
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        pending = asyncio.create_task(provider.compose_catalog())
        try:
            # Observe the first real finite permission read separately; the
            # existing catalog readiness bound then measures its own phase.
            deadline = time.monotonic() + 4
            while not probe.permission_calls and not pending.done():
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
            assert probe.permission_calls
            await catalog_controls._worker_entered(probe, pending)
            if change == "inventory":

                def replacement():
                    replacement_calls.append(True)
                    return {"tools": [{"name": "replacement"}]}

                monkeypatch.setattr(case.local, "get_inventory", replacement)
            elif change == "governance":
                replacement = copy.copy(governance)
                assert replacement == governance and replacement is not governance
                case.local.policy_enforcer = replacement
            else:
                replacement = replace(binding)
                assert replacement == binding and replacement is not binding
                mcp_source_participants._BINDINGS[case.source] = replacement
            probe.release.set()
            with pytest.raises((PermissionError, bootstrap.RecoveryRequired)):
                await pending
            assert not replacement_calls
            assert not provider.list_catalog()
            assert probe.leases and all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(pending, probe)
            mcp_source_participants._BINDINGS[case.source] = binding
            case.local.policy_enforcer = governance


@pytest.mark.asyncio
async def test_standard_provider_shared_keeper_accepts_only_its_lazy_publications(
    snapshot_case, mcp_sources, record_property
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    case = snapshot_case
    _, observed = _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash="0" * 64
    )
    case.service._permission_store = None
    assert case.service._execution_log is None
    provider = MCPToolProvider(
        service=case.service, main_loop=asyncio.get_running_loop()
    )
    probe = _MaximumProbe(case.source, case.permissions)
    with probe.installed():
        await provider.compose_catalog()
    assert provider.list_catalog() and observed == [True]
    assert case.service._permission_store is not case.permissions
    assert case.service._permission_store.path == case.permissions.path
    assert case.service._execution_log is not mcp_sources[4]
    assert case.service._execution_log.path == mcp_sources[4].path
    assert case.permissions.get_tool_entry("local:one", "first")["config_changed"]
    record_property("permission_callers", repr(probe.permission_callers))
    # Kill-switch, effective state, and downgrade RMW each require a fresh load.
    assert len(probe.permission_calls) == 3 and len(probe.read_threads) == 1
    assert all(
        owner is case.service._permission_store for owner, _ in probe.permission_calls
    )
    assert "effective_tool_states" in {
        name for name, _, _ in probe.permission_callers[1]
    }
    assert {"mark_config_changed", "_mutate_profile_locked"} <= {
        name for name, _, _ in probe.permission_callers[2]
    }
    assert all(
        thread is not threading.current_thread() for _, thread in probe.permission_calls
    )
    assert probe.audit_threads and all(
        thread is not threading.current_thread() for thread in probe.audit_threads
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["inventory", "permission_marker"])
async def test_async_maximum_rejects_equal_callable_in_retained_binding_slot(
    snapshot_case, monkeypatch, method
):
    class EqualCallable:
        def __eq__(self, other):
            return True

        def __call__(self, *args, **kwargs):
            raise AssertionError("replacement callback must not execute")

    case = snapshot_case
    _loop_projection(case)
    probe = _MaximumProbe(case.source, case.permissions, hold=True)
    with probe.installed():
        pending = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            await catalog_controls._worker_entered(probe, pending)
            owner, name = (
                (case.local, "get_inventory")
                if method == "inventory"
                else (case.permissions, "mark_config_changed")
            )
            original = getattr(owner, name)
            replacement = EqualCallable()
            assert replacement == original and replacement is not original
            monkeypatch.setattr(owner, name, replacement)
            probe.release.set()
            captured = await pending
            assert not captured.mcp_definition_maximum
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await catalog_controls._settle(pending, probe)


@pytest.mark.asyncio
async def test_class_custom_audit_keeps_original_caller_loop_contract(
    snapshot_case, monkeypatch
):
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash="0" * 64
    )
    original = UnifiedMCPControlPlaneService._audit_downgrade_if_fresh
    thread = threading.current_thread()
    loop = asyncio.get_running_loop()
    observed = []

    def caller_owned(owner, *args, **kwargs):
        result = original(owner, *args, **kwargs)
        observed.append(threading.current_thread())
        assert threading.current_thread() is thread
        assert asyncio.get_running_loop() is loop
        return result

    monkeypatch.setattr(
        UnifiedMCPControlPlaneService, "_audit_downgrade_if_fresh", caller_owned
    )
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    assert captured.mcp_definition_maximum
    assert observed and all(item is thread for item in observed)
    assert not standard_console_sources(case.service)
    assert case.permissions.get_tool_entry("local:one", "first")["config_changed"]


@pytest.fixture
def wired_composer_case(wired_case):
    """Add a real composer through Textual's declared test-only DOM helper."""
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    case = wired_case
    composer = ConsoleComposerBar(id="console-native-composer")
    case.screen._add_children(composer)
    assert case.screen._console_composer_or_none() is composer
    assert case.screen._session._console_composer_or_none() is composer
    case.composer = composer
    return case


async def _retire_wired_preparation(case, task, probe):
    """Retire a held read and any actually accepted test turn before disposal."""
    await catalog_controls._settle(task, probe)
    runtime = case.screen._console_runtime()
    tasks = tuple(
        record.task for record in runtime._turn_custody.values() if record.task
    )
    for accepted in tasks:
        accepted.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await runtime.dispose(timeout_seconds=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_text", ["", "previous draft", "unchanged input"])
async def test_actual_wired_preparation_accepts_only_mirroring_of_unchanged_composer(
    wired_composer_case, stored_text
):
    """A delayed original mirror must not turn an unchanged Send into drift."""
    from tldw_chatbook.UI.Console_Modules import wiring

    case = wired_composer_case
    _loop_projection(case)
    runtime = case.screen._console_runtime()
    draft = "unchanged input"
    case.composer.load_draft(draft)
    case.store.set_session_draft(case.session.id, stored_text)
    stash = case.composer.capture_draft_for_send()
    snapshot = case.composer.capture_draft_snapshot()
    settings_revision = case.store.session_settings_revision(case.session.id)
    probe = _PermissionProbe(case.source, case.permissions)
    with probe.installed():
        task = asyncio.create_task(
            case.screen._prompt_queue._launch_chain_async(
                draft, case.session.id, stash, case.controller
            )
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            assert not runtime.has_custodied_turns()
            assert (
                task.get_coro().cr_code
                is wiring._prepare_console_turn_to_runtime.__code__
            )
            frame = task.get_coro().cr_frame
            captured = dict(frame.f_locals)
            assert captured["session"] is case.session
            assert captured["store"] is case.store
            assert captured["runtime"] is runtime
            assert captured["composer"] is case.composer
            assert captured["composer_snapshot"] == snapshot
            assert probe.leases and all(
                lease in storage_admission._live_leases for lease in probe.leases
            )

            # This is the same original sync used by the ordinary transcript poll.
            # It may update only the text mirror; it must not edit the composer.
            case.screen._session._sync_console_session_draft()
            assert case.store.session_draft(case.session.id) == draft
            assert case.composer.capture_draft_snapshot() == snapshot
            assert (
                case.store.session_settings_revision(case.session.id)
                == settings_revision
            )
            assert (
                next(s for s in case.store.sessions() if s.id == case.session.id)
                is captured["session"]
            )
            assert runtime._chat_store is captured["store"]
            assert (
                case.store.session_one_shot_prefill_snapshot(case.session.id)
                == captured["prefill"]
            )
            assert (
                tuple(case.store.pending_attachments(case.session.id))
                == captured["attachments"]
            )
            evidence = runtime.snapshot_console_staged_evidence()
            assert evidence[0] is captured["evidence"][0]
            assert evidence[1:] == captured["evidence"][1:]
            assert (
                stash.text,
                stash.edit_serial,
                stash.generation,
                tuple(stash.segments),
            ) == captured["stash_identity"]
            assert case.screen._console_visible_draft_session_id == case.session.id
            assert case.screen._console_composer_or_none() is captured["composer"]
            mirror_changed = captured["stored_draft"] != case.store.session_draft(
                case.session.id
            )
            probe.release.set()
            try:
                turn_id = await task
            except bootstrap.RecoveryRequired as error:
                raise AssertionError(
                    f"unchanged composer refused after original mirror; mirror_changed={mirror_changed}"
                ) from error
            assert turn_id in runtime._turn_custody
            assert runtime._turn_custody[turn_id].request.draft == draft
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await _retire_wired_preparation(case, task, probe)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["edit", "edit_away_and_back", "same_text_scope", "store_only"]
)
async def test_actual_wired_preparation_still_refuses_real_draft_drift(
    wired_composer_case, change
):
    """Mirroring never authorizes a changed draft revision or store owner text."""
    case = wired_composer_case
    _loop_projection(case)
    runtime = case.screen._console_runtime()
    draft = "owning input"
    case.composer.load_draft(draft)
    case.store.set_session_draft(case.session.id, draft)
    stash = case.composer.capture_draft_for_send()
    snapshot = case.composer.capture_draft_snapshot()
    probe = _PermissionProbe(case.source, case.permissions)
    with probe.installed():
        task = asyncio.create_task(
            case.screen._prompt_queue._launch_chain_async(
                draft, case.session.id, stash, case.controller
            )
        )
        try:
            await catalog_controls._worker_entered(probe, task)
            assert not runtime.has_custodied_turns()
            assert probe.leases and all(
                lease in storage_admission._live_leases for lease in probe.leases
            )
            if change == "store_only":
                case.store.set_session_draft(case.session.id, "independent newer draft")
                assert case.composer.capture_draft_snapshot() == snapshot
            elif change == "same_text_scope":
                case.composer.load_draft(draft)
                assert case.composer.draft_text() == draft
                assert (
                    case.composer.capture_draft_snapshot().generation
                    != snapshot.generation
                )
            else:
                case.composer.insert_text("x")
                if change == "edit_away_and_back":
                    case.composer.delete_left()
                    assert case.composer.draft_text() == draft
                assert case.composer.edit_serial != snapshot.edit_serial
                case.screen._session._sync_console_session_draft()
            probe.release.set()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await task
            assert not runtime.has_custodied_turns()
            assert case.composer.draft_text() == (
                draft + "x" if change == "edit" else draft
            )
            assert case.store.session_draft(case.session.id) == (
                "independent newer draft"
                if change == "store_only"
                else case.composer.draft_text()
            )
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await _retire_wired_preparation(case, task, probe)
