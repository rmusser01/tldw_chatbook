"""Resume retains the real registry read while allowing loop progress."""

import asyncio
from contextlib import contextmanager
import inspect
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_preparation_reads import _source_binding
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.attach_visit import ConsoleAttachVisit
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@contextmanager
def _held_original_read(registry):
    """Hold the actual SELECT result without replacing the source callback."""
    entered = threading.Event()
    release = threading.Event()
    state = SimpleNamespace(entered=entered, release=release, thread=None, read=None)
    code = inspect.unwrap(LocalWorkspaceRegistryService.get_active_workspace).__code__
    monitoring = sys.monitoring
    tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)

    def hold(_code, _line):
        frame = sys._getframe(1)
        if frame.f_locals.get("self") is not registry or entered.is_set():
            return
        if "row" not in frame.f_locals:
            return
        state.thread = threading.current_thread()
        binding = _source_binding.get()
        state.read = None if binding is None else binding.read
        state.operation = getattr(storage._operation_local, "operation", None)
        state.connection = frame.f_locals["conn"]
        state.participant = registry.db._maintenance_participant
        with storage._lock:
            state.lease = state.participant.connections.get(state.connection)
            state.native_live = (
                state.operation in storage._operations
                and state.lease in storage._live_leases
            )
        entered.set()
        state.released = release.wait(5)

    monitoring.use_tool_id(tool, "workspace-reconciliation-read")
    monitoring.register_callback(tool, monitoring.events.LINE, hold)
    monitoring.set_local_events(tool, code, monitoring.events.LINE)
    try:
        yield state
    finally:
        release.set()
        monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.LINE, None)
        monitoring.free_tool_id(tool)


def _case(tmp_path):
    database = WorkspaceDB(tmp_path / "reconcile.sqlite", client_id="reconcile-worker")
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(
        workspace_id="workspace-a", name="A", assistant_defaults=None
    )
    registry.set_active_workspace("workspace-a")
    store = ConsoleChatStore()
    session = store.create_session(title="A", workspace_id="workspace-a")
    app = SimpleNamespace(workspace_registry_service=registry)
    app.console_runtime = ConsoleRuntime(app, canvas_enabled_reader=lambda: False)
    view = SimpleNamespace(token=object(), running=True)
    controller = _workspace_controller(
        app_instance=app,
        chat_store_accessor=lambda: store,
        current_chat_store_accessor=lambda: store,
        screen_lifecycle_token_accessor=lambda: view.token,
        screen_running_accessor=lambda: view.running,
    )
    return database, registry, controller, store, session, view


async def _wait_for(predicate):
    async with asyncio.timeout(8):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_original_reconcile_read_allows_loop_progress_and_is_one_use(tmp_path):
    database, registry, controller, _store, _session, _view = _case(tmp_path)
    published = []

    async def resume():
        async with controller.prepare_console_workspace_reconciliation():
            controller._reconcile_console_session_with_registry()
            published.append("resume")

    task = None
    try:
        with _held_original_read(registry) as probe:
            task = asyncio.create_task(resume())
            await _wait_for(lambda: probe.entered.is_set() or task.done())
            if task.done():
                await task
            assert probe.thread is not threading.current_thread()
            assert probe.read in controller._preparation_reads
            assert (
                probe.read in controller.app_instance.console_runtime._preparation_reads
            )
            assert probe.native_live
            assert not task.done() and not published
            probe.release.set()
            await task
            assert probe.released
        assert published == ["resume"]
        assert not controller._preparation_reads
        with storage._lock:
            assert probe.operation not in storage._operations
            assert probe.lease not in storage._live_leases
        # The result must not survive this exact resume stack.
        with _held_original_read(registry) as fresh:
            fresh.release.set()
            controller._reconcile_console_session_with_registry()
            assert fresh.entered.is_set()
            assert fresh.thread is threading.current_thread()
    finally:
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await controller.app_instance.console_runtime.dispose()
        database.close()


@pytest.mark.parametrize(
    "drift",
    [
        "session",
        "workspace",
        "view",
        "source",
        "registry",
        "generation",
        "runtime",
        "cancel",
    ],
)
async def test_held_reconciliation_never_resumes_after_drift(
    tmp_path, monkeypatch, drift
):
    database, registry, controller, store, session, view = _case(tmp_path)
    published = []

    async def resume():
        async with controller.prepare_console_workspace_reconciliation():
            controller._reconcile_console_session_with_registry()
            published.append("resume")

    task = disposal = None
    try:
        with _held_original_read(registry) as probe:
            task = asyncio.create_task(resume())
            await _wait_for(lambda: probe.entered.is_set() or task.done())
            if task.done():
                await task
            if drift == "session":
                store.create_session(title="New", workspace_id="workspace-a")
            elif drift == "workspace":
                session.workspace_id = "workspace-b"
            elif drift == "view":
                view.token = object()
            elif drift == "source":
                monkeypatch.setattr(registry, "get_active_workspace", lambda: None)
            elif drift == "registry":
                controller.app_instance.workspace_registry_service = None
            elif drift == "generation":
                registry.clear_active_workspace()
            elif drift == "runtime":
                disposal = asyncio.create_task(
                    controller.app_instance.console_runtime.dispose()
                )
                await _wait_for(
                    lambda: controller.app_instance.console_runtime._disposed
                )
                assert not disposal.done()
            else:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert probe.read in controller._preparation_reads
            probe.release.set()
            result = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result[0], BaseException)
        assert not published
        assert not controller._preparation_reads
    finally:
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if disposal is not None:
            await disposal
        await controller.app_instance.console_runtime.dispose()
        database.close()


async def test_custom_active_workspace_callback_keeps_original_loop_affinity(
    tmp_path, monkeypatch
):
    database, registry, controller, _store, _session, _view = _case(tmp_path)
    original = registry.get_active_workspace
    threads = []

    def custom():
        threads.append(threading.current_thread())
        return original()

    monkeypatch.setattr(registry, "get_active_workspace", custom)
    try:
        async with controller.prepare_console_workspace_reconciliation():
            assert not threads
            controller._reconcile_console_session_with_registry()
        assert threads == [threading.current_thread()]
        assert not controller._preparation_reads
    finally:
        await controller.app_instance.console_runtime.dispose()
        database.close()


async def test_custom_reconciler_does_not_prepare_unused_registry_read(
    tmp_path, monkeypatch
):
    database, registry, controller, _store, _session, _view = _case(tmp_path)
    calls = []
    monkeypatch.setattr(
        controller,
        "_reconcile_console_session_with_registry",
        lambda: calls.append("custom"),
    )
    try:
        with _held_original_read(registry) as probe:
            probe.release.set()
            async with controller.prepare_console_workspace_reconciliation():
                controller._reconcile_console_session_with_registry()
            assert not probe.entered.is_set()
        assert calls == ["custom"]
    finally:
        await controller.app_instance.console_runtime.dispose()
        database.close()


async def test_custom_connection_configuration_keeps_original_loop_affinity(
    tmp_path, monkeypatch
):
    database, _registry, controller, _store, _session, _view = _case(tmp_path)
    original = database._configure_connection
    threads = []

    def configure(connection):
        threads.append(threading.current_thread())
        return original(connection)

    monkeypatch.setattr(database, "_configure_connection", configure)
    database.close()
    try:
        async with controller.prepare_console_workspace_reconciliation():
            assert not threads
            controller._reconcile_console_session_with_registry()
        assert threads == [threading.current_thread()]
    finally:
        await controller.app_instance.console_runtime.dispose()
        database.close()


async def test_prepared_registry_change_repairs_session_on_original_loop(tmp_path):
    database, registry, controller, store, session, _view = _case(tmp_path)
    registry.create_workspace(
        workspace_id="workspace-b", name="B", assistant_defaults=None
    )
    registry.set_active_workspace("workspace-b")
    target = store.create_session(title="B", workspace_id="workspace-b")
    store.switch_session(session.id)
    calls = []

    def activate(workspace_id):
        assert threading.current_thread() is threading.main_thread()
        assert workspace_id == "workspace-b"
        store.switch_session(target.id)
        calls.append("activate")

    async def sync_ui():
        pass

    def run_worker(coroutine, **_kwargs):
        coroutine.close()
        calls.append("worker")

    controller._activate_console_session_for_workspace = activate
    controller._sync_chat_core_state_fn = lambda: calls.append("core")
    controller._sync_workspace_context_fn = lambda: calls.append("context")
    controller._sync_native_console_chat_ui_fn = sync_ui
    controller._screen.run_worker = run_worker
    try:
        async with controller.prepare_console_workspace_reconciliation():
            controller._reconcile_console_session_with_registry()
        assert store.ensure_session() is target
        assert calls == ["core", "activate", "context", "worker"]
    finally:
        await controller.app_instance.console_runtime.dispose()
        database.close()


@pytest.mark.parametrize("outcome", ["resume", "new_visit", "retry"])
async def test_new_attach_visit_defers_resume_while_original_read_is_held(
    tmp_path, outcome
):
    database, registry, controller, _store, _session, view = _case(tmp_path)
    callbacks = []
    resumed = []
    reconciled = []
    runtime = SimpleNamespace(_attached_generation=1)
    runtime.finish_view_reconciliation = lambda *_: reconciled.append("finish") or True
    screen = SimpleNamespace(
        app_instance=controller.app_instance,
        _workspace=controller,
        _console_attach_reconciled=False,
        _console_attach_reconcile_running=False,
        _console_attach_sync_complete=True,
        _console_attach_runtime_reconciled=False,
        _console_attach_view_started=True,
        _console_attach_resume_in_progress=False,
        _console_resume_after_reconcile=True,
        _resume_navigation_startup_in_progress=False,
        _console_mount_visit_refreshed=True,
        _console_chat_controller=None,
        _pending_character_return_focus_id=None,
        _console_attach_reconcile_retry_exhausted=False,
        _console_attach_reconcile_retry_count=0,
        call_after_refresh=callbacks.append,
        set_timer=lambda _delay, callback: callbacks.append(callback),
    )
    runtime.view = screen
    view.visit = 1
    screen._current_console_attach_visit = lambda: ConsoleAttachVisit(
        runtime, screen, 1, view.visit, True
    )
    screen._reconcile_console_after_attach = (
        lambda: ChatScreen._reconcile_console_after_attach(screen)
    )

    def resumed_body():
        controller._reconcile_console_session_with_registry()
        resumed.append("resume")
        if outcome == "retry" and len(resumed) == 1:
            raise RuntimeError("resume failed after runtime became answerable")

    screen.on_screen_resume = resumed_body
    task = None
    try:
        with _held_original_read(registry) as probe:
            task = asyncio.create_task(screen._reconcile_console_after_attach())
            await _wait_for(probe.entered.is_set)
            assert not reconciled
            # A suspend/new resume changes the visit while the physical read
            # is still held. The real resume guard must queue reconciliation.
            if outcome == "new_visit":
                view.visit += 1
                view.token = object()
                ChatScreen.on_screen_resume(screen)
                assert screen._reconcile_console_after_attach in callbacks
            assert not resumed
            probe.release.set()
            await task
        assert resumed == ([] if outcome == "new_visit" else ["resume"])
        assert reconciled == ([] if outcome == "new_visit" else ["finish"])
        assert screen._console_attach_reconciled is (outcome == "resume")
        if outcome == "retry":
            with _held_original_read(registry) as retried:
                retried.release.set()
                await screen._reconcile_console_after_attach()
                assert retried.entered.is_set()
                assert retried.thread is threading.current_thread()
                assert retried.read is None
            assert screen._console_attach_reconciled
            assert resumed == ["resume", "resume"]
            assert reconciled == ["finish"]
    finally:
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await controller.app_instance.console_runtime.dispose()
        database.close()
