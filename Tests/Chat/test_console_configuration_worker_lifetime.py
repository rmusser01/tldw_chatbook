"""Stock configuration capture retains native readers away from the input loop."""

import asyncio
import contextlib
import copy
import inspect
import sqlite3
import sys
import threading

import pytest
import pytest_asyncio

from Tests.Chat import test_console_configuration_capture as capture_controls
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.console_turn_context import resolve_turn_tool_policy_profile_id
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

catalog_store = capture_controls.catalog_store
local_root = capture_controls.local_root
mcp_sources = capture_controls.mcp_sources
snapshot_case = capture_controls.snapshot_case
runtime_case = capture_controls.runtime_case


async def _until(predicate, seconds=10):
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class _OriginalConfigurationWorkspaceRead:
    """Hold the original SQL reader after its real connection is admitted."""

    def __init__(self, registry):
        self.registry = registry
        self.code = inspect.unwrap(type(registry).get_workspace).__code__
        self.caller_code = resolve_turn_tool_policy_profile_id.__code__
        self.entered = threading.Event()
        self.release = threading.Event()
        self.connection = self.lease = self.thread = None
        self.live_at_entry = False
        self.release_timed_out = False
        self.foreign_registry = None
        self.foreign_calls = 0
        self.observe_entry = None
        self.entry_observation_error = None

    def _line(self, code, _line):
        if code is not self.code:
            return
        frame = sys._getframe(1)
        if (
            self.foreign_registry is not None
            and frame.f_locals.get("self") is self.foreign_registry
        ):
            self.foreign_calls += 1
        if self.entered.is_set():
            return
        if frame.f_locals.get("self") is not self.registry:
            return
        connection = frame.f_locals.get("conn")
        if connection is None:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not self.caller_code:
            parent = parent.f_back
        if parent is None:
            return
        self.connection = connection
        self.thread = threading.current_thread()
        participant = self.registry.db._maintenance_participant
        with storage._lock:
            self.lease = participant.connections.get(connection)
            self.live_at_entry = (
                self.lease in storage._live_leases
                and self.lease.resource_thread is self.thread
            )
        if self.observe_entry is not None:
            try:
                self.observe_entry(parent)
            except (
                Exception
            ) as error:  # Preserve the original reader on diagnostic failure.
                self.entry_observation_error = type(error).__name__
        self.entered.set()
        self.release_timed_out = not self.release.wait(8)

    @contextlib.contextmanager
    def installed(self):
        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "configuration-original-workspace")
        try:
            monitoring.register_callback(tool, monitoring.events.LINE, self._line)
            monitoring.set_local_events(tool, self.code, monitoring.events.LINE)
            yield self
        finally:
            self.release.set()
            monitoring.set_local_events(tool, self.code, 0)
            monitoring.register_callback(tool, monitoring.events.LINE, None)
            monitoring.free_tool_id(tool)

    def retired(self):
        if self.connection is None:
            return False
        try:
            sqlite3.Connection.in_transaction.__get__(self.connection)
        except sqlite3.ProgrammingError:
            closed = True
        else:
            closed = False
        with storage._lock:
            return closed and self.lease not in storage._live_leases


def _mounted_capture_provider(case):
    from functools import partial
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Console_Modules.wiring import _stock_console_scratch_snapshot

    owner = ConsoleSessionController.__new__(ConsoleSessionController)
    owner.app_instance = case.app
    owner._provider_readiness_app_config_fn = lambda: case.app.app_config
    owner._build_provider_selection_fn = case.controller._provider_selection_for_session
    owner._current_chat_store_accessor = lambda: case.store
    owner._chat_store_accessor = lambda: case.store
    owner._rag_source_types_accessor = lambda: ["notes"]
    owner._rag_top_k_accessor = lambda: 17
    screen = SimpleNamespace(
        _session=owner, app_instance=case.app, _console_runtime=lambda: case.runtime
    )
    owner._scratch_snapshot_provider = partial(_stock_console_scratch_snapshot, screen)
    return owner._build_console_turn_execution_context


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["runtime", "mounted", "received"])
async def test_original_configuration_workspace_read_does_not_block_input_loop(
    runtime_case,
    route,
):
    """The existing public capture must not execute real Workspace SQL on its loop."""
    from tldw_chatbook import config

    case = runtime_case
    database = WorkspaceDB(
        config.get_user_data_dir() / "configuration-worker.sqlite",
        client_id="configuration-worker",
    )
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="capture-owned", name="Capture owned")
    case.app.workspace_registry_service = registry
    case.session.workspace_id = "capture-owned"
    database.close()
    capture_kwargs = {}
    if route == "mounted":
        capture_kwargs["context_provider"] = _mounted_capture_provider(case)
    elif route == "received":
        from Tests.Chat.test_console_configuration_preparation import _selection

        capture_kwargs["selection"] = _selection(case)
    probe = _OriginalConfigurationWorkspaceRead(registry)
    loop = asyncio.get_running_loop()
    caller_thread = threading.current_thread()
    loop_progress = threading.Event()
    stop = threading.Event()
    progress_while_held = []

    def mark_loop_progress():
        progress_while_held.append(not probe.release.is_set())
        loop_progress.set()

    def release_independently():
        try:
            # An original loop-thread read cannot await its own async releaser.
            # The bounded independent thread lets that RED finish and clean up.
            while not probe.entered.wait(0.01):
                if stop.is_set():
                    return
            loop.call_soon_threadsafe(mark_loop_progress)
            loop_progress.wait(0.5)
        finally:
            probe.release.set()

    releaser = threading.Thread(target=release_independently)
    started = False
    try:
        with probe.installed():
            releaser.start()
            started = True
            captured = await asyncio.wait_for(
                case.controller.capture_turn_configuration_snapshot(
                    case.session.id, **capture_kwargs
                ),
                15,
            )
            await asyncio.sleep(0)
            assert probe.entered.is_set(), "original Workspace reader was not reached"
            assert probe.live_at_entry and not probe.release_timed_out
            assert captured.session_id == case.session.id
            assert captured.provider_selection.provider == "deepseek"
            assert (
                probe.thread is not caller_thread
            ), "original configuration Workspace SQL still executes on the input loop"
            assert progress_while_held == [
                True
            ], "input loop stalled during native capture"
            assert (
                probe.retired()
            ), "configuration returned before physical reader retirement"
    finally:
        stop.set()
        probe.release.set()
        if started:
            releaser.join(timeout=2)
            assert not releaser.is_alive()
        # Baseline may have allocated its original handle on the loop. Clean that
        # fixture owner explicitly, without treating cleanup as product ownership.
        database.close()


@pytest_asyncio.fixture
async def configuration_workspace(runtime_case):
    from tldw_chatbook import config

    case = runtime_case
    database = WorkspaceDB(
        config.get_user_data_dir() / "configuration-lifetime.sqlite",
        client_id="configuration-lifetime",
    )
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="capture-owned", name="Capture owned")
    case.app.workspace_registry_service = registry
    case.session.workspace_id = "capture-owned"
    case.store.switch_session(case.session.id)
    case.database, case.registry = database, registry
    database.close()
    try:
        yield case
    finally:
        database.close()


async def _wait_for_original_reader(probe, task):
    assert await _until(lambda: probe.entered.is_set() or task.done(), 10)
    if task.done() and not probe.entered.is_set():
        task.result()
    assert probe.entered.is_set(), "original configuration reader was not reached"
    assert probe.thread is not threading.current_thread()
    assert probe.live_at_entry and not probe.release_timed_out
    with storage._lock:
        assert probe.lease in storage._live_leases
        assert (
            probe.registry.db._maintenance_participant.connections.get(probe.connection)
            is probe.lease
        )
    assert not probe.retired()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "close", "dispose"])
async def test_original_configuration_reader_survives_repeated_cancel_and_lifecycle(
    configuration_workspace, action, monkeypatch
):
    from tldw_chatbook.Chat.console_preparation_reads import preparation_reads_for
    from tldw_chatbook.Chat.console_runtime import (
        CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS,
        CONSOLE_SESSION_CLOSE_GRACE_SECONDS,
    )

    case = configuration_workspace
    probe = _OriginalConfigurationWorkspaceRead(case.registry)
    task = closing = read = None
    ended = []
    original_end = case.store.end_app_runtime

    def end_after_original_native_retirement():
        assert (
            probe.retired()
        ), "store teardown preceded original configuration SQL retirement"
        ended.append(True)
        return original_end()

    monkeypatch.setattr(
        case.store, "end_app_runtime", end_after_original_native_retirement
    )
    with probe.installed():
        try:
            task = asyncio.create_task(
                case.controller.capture_turn_configuration_snapshot(case.session.id)
            )
            await _wait_for_original_reader(probe, task)
            (read,) = preparation_reads_for(
                case.controller._preparation_reads, case.session.id
            )
            assert read in case.runtime._preparation_reads
            assert read.task is task and not read.retired.done()
            # This is a configuration-only caller, before complete-request custody.
            assert not case.runtime.has_custodied_turns(case.session.id)
            if action == "stop":
                assert case.controller.stop_active_run()
            elif action == "close":
                closing = asyncio.create_task(
                    case.runtime.close_session(
                        case.session.id,
                        expected_revision=case.controller.lifecycle_impact(
                            session_id=case.session.id
                        ).revision,
                    )
                )
                await asyncio.sleep(CONSOLE_SESSION_CLOSE_GRACE_SECONDS + 0.05)
            else:
                closing = asyncio.create_task(case.runtime.dispose())
                await asyncio.sleep(CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS + 0.05)
            for _ in range(2):
                task.cancel()
                if closing is not None:
                    closing.cancel()
                await asyncio.sleep(0.05)
                assert (
                    not task.done()
                ), "caller cancellation escaped a live configuration reader"
                assert not read.retired.done()
                assert read in case.runtime._preparation_reads
                assert any(item is case.session for item in case.store.sessions())
                assert ended == []
                if closing is not None:
                    assert (
                        not closing.done()
                    ), "lifecycle completed before configuration retirement"
                with storage._lock:
                    assert probe.lease in storage._live_leases
                assert not probe.retired()
        finally:
            probe.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            if probe.entered.is_set():
                assert await _until(probe.retired, 5)
    assert task.cancelled()
    assert read.retired.done() and not read.retired.cancelled()
    assert preparation_reads_for(case.runtime._preparation_reads, case.session.id) == ()
    assert not case.runtime.has_custodied_turns(case.session.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["registry", "session"])
@pytest.mark.parametrize("route", ["runtime", "received"])
async def test_original_configuration_refuses_source_replacement_without_successor_read(
    configuration_workspace, replacement, route
):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    case = configuration_workspace
    successor_database = WorkspaceDB(
        config.get_user_data_dir() / "configuration-successor.sqlite",
        client_id="configuration-successor",
    )
    successor = LocalWorkspaceRegistryService(successor_database)
    successor.create_workspace(workspace_id="capture-owned", name="Successor")
    successor_database.close()
    probe = _OriginalConfigurationWorkspaceRead(case.registry)
    probe.foreign_registry = successor
    task = None
    capture_kwargs = {}
    if route == "received":
        from Tests.Chat.test_console_configuration_preparation import _selection

        capture_kwargs["selection"] = _selection(case)
    with probe.installed():
        try:
            task = asyncio.create_task(
                case.controller.capture_turn_configuration_snapshot(
                    case.session.id, **capture_kwargs
                )
            )
            await _wait_for_original_reader(probe, task)
            if replacement == "registry":
                case.app.workspace_registry_service = successor
            else:
                case.store._sessions[case.session.id] = copy.copy(case.session)
            probe.release.set()
            with pytest.raises(RecoveryRequired):
                await task
            assert (
                probe.foreign_calls == 0
            ), "old capture entered successor Workspace storage"
            assert await _until(probe.retired, 5)
            assert not case.runtime.has_custodied_turns(case.session.id)
        finally:
            probe.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            case.app.workspace_registry_service = case.registry
            case.store._sessions[case.session.id] = case.session
            successor_database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "held"])
async def test_mounted_policy_callback_change_retains_affinity_and_refuses_stale_result(
    configuration_workspace, monkeypatch, phase
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.UI.Console_Modules import turn_admission

    case = configuration_workspace
    provider = _mounted_capture_provider(case)
    observed = []

    def custom_profile(app, workspace_id):
        observed.append(threading.current_thread())
        return "custom-current-policy"

    async def capture():
        return await case.controller.capture_turn_configuration_snapshot(
            case.session.id, context_provider=provider
        )

    if phase == "before":
        monkeypatch.setattr(
            turn_admission, "resolve_turn_tool_policy_profile_id", custom_profile
        )
        result = await capture()
        assert result.tool_policy_profile_id == "custom-current-policy"
        assert observed == [threading.current_thread()]
        return
    probe = _OriginalConfigurationWorkspaceRead(case.registry)
    with probe.installed():
        task = asyncio.create_task(capture())
        try:
            await _wait_for_original_reader(probe, task)
            monkeypatch.setattr(
                turn_admission, "resolve_turn_tool_policy_profile_id", custom_profile
            )
            probe.release.set()
            with pytest.raises(RecoveryRequired):
                await task
            assert observed == []
            assert probe.retired()
        finally:
            probe.release.set()
            with contextlib.suppress(asyncio.CancelledError, RecoveryRequired):
                await task
