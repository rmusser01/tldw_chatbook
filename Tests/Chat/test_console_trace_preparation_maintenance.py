"""Trace maintenance must respect a real Console's whole submit ownership."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_runtime_trace_maintenance_recognizes_provider_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved-revision writer exists before the first provider stream starts."""
    entered_readiness = asyncio.Event()
    maintenance_started = asyncio.Event()
    observed_activity: list[bool] = []

    class Gateway:
        async def resolve_for_send(self, _selection):
            entered_readiness.set()
            await asyncio.Event().wait()

    class Maintenance:
        def __init__(self, _database, *, provider_active, **_kwargs):
            observed_activity.append(provider_active())
            maintenance_started.set()

        def run_batch(self):
            return SimpleNamespace(logical_complete=False, admitted=False)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = Maintenance
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS",
        0.0,
    )
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(store=store, provider_gateway=Gateway())
    runtime = ConsoleRuntime(SimpleNamespace(_ui_ready=True))
    runtime.set_chat_controller(controller)
    submit = asyncio.create_task(controller.submit_draft("synthetic preparation"))
    try:
        await asyncio.wait_for(entered_readiness.wait(), timeout=5)
        assert controller._submit_tasks_snapshot() == {submit: session.id}
        assert controller._active_stream_tasks == {}
        runtime._schedule_legacy_trace_maintenance(object(), object)
        await asyncio.wait_for(maintenance_started.wait(), timeout=5)
        assert observed_activity == [True]
    finally:
        submit.cancel()
        await asyncio.gather(submit, return_exceptions=True)
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_submit", [False, True])
async def test_runtime_gc_fences_new_submits_before_its_first_database_job(
    monkeypatch: pytest.MonkeyPatch,
    cancel_submit: bool,
) -> None:
    """A submit admitted after the idle check must wait before creating revisions."""
    import threading

    loop = asyncio.get_running_loop()
    epoch_started = asyncio.Event()
    release_epoch = threading.Event()
    entered_readiness = asyncio.Event()
    submit_registered = asyncio.Event()

    class Gateway:
        async def resolve_for_send(self, _selection):
            entered_readiness.set()
            await asyncio.Event().wait()

    class Maintenance:
        def __init__(self, _database, **_kwargs):
            pass

        def run_batch(self):
            return SimpleNamespace(logical_complete=True, admitted=True)

    class Collector:
        def __init__(self, _database):
            pass

        def current_graph_epoch(self):
            loop.call_soon_threadsafe(epoch_started.set)
            assert release_epoch.wait(timeout=5)
            return 7

        def collect(self, *, request_id):
            return SimpleNamespace(marked_epoch=7)

    class Compactor:
        def __init__(self, _database, **_kwargs):
            pass

        def run_after_gc(self, _result):
            return SimpleNamespace(completed=True, reason_code="complete")

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = Maintenance
    module.TraceGarbageCollector = Collector
    module.PhysicalTraceCompactor = Compactor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS",
        0.0,
    )
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(store=store, provider_gateway=Gateway())
    register = controller._register_submit_task

    def register_and_signal(task, owner):
        register(task, owner)
        submit_registered.set()

    monkeypatch.setattr(controller, "_register_submit_task", register_and_signal)
    runtime = ConsoleRuntime(SimpleNamespace(_ui_ready=True, app_config={}))
    runtime.set_chat_controller(controller)
    submit = None
    try:
        runtime._schedule_legacy_trace_maintenance(object(), object)
        await asyncio.wait_for(epoch_started.wait(), timeout=5)
        assert controller._trace_maintenance_dispatch_paused.is_set()
        submit = asyncio.create_task(controller.submit_draft("synthetic GC overlap"))
        await asyncio.wait_for(submit_registered.wait(), timeout=5)
        assert controller._submit_tasks_snapshot() == {submit: session.id}
        assert store.messages_for_session(session.id) == []
        assert not entered_readiness.is_set()
        if cancel_submit:
            submit.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submit
            assert controller._submit_tasks_snapshot() == {}
            assert store.messages_for_session(session.id) == []
            assert not entered_readiness.is_set()
        release_epoch.set()
        if not cancel_submit:
            await asyncio.wait_for(entered_readiness.wait(), timeout=5)
    finally:
        release_epoch.set()
        if submit is not None:
            submit.cancel()
            await asyncio.gather(submit, return_exceptions=True)
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_maintenance", [False, True])
async def test_runtime_gc_releases_dispatch_fence_on_failure_or_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    cancel_maintenance: bool,
) -> None:
    """Maintenance cannot strand later registered submits behind its pause."""
    import threading

    from tldw_chatbook.Chat import console_runtime as runtime_module

    loop = asyncio.get_running_loop()
    epoch_started = asyncio.Event()
    failure_observed = asyncio.Event()
    release_epoch = threading.Event()

    class Maintenance:
        def __init__(self, _database, **_kwargs):
            pass

        def run_batch(self):
            return SimpleNamespace(logical_complete=True, admitted=True)

    class Collector:
        def __init__(self, _database):
            pass

        def current_graph_epoch(self):
            return 7

        def collect(self, *, request_id):
            loop.call_soon_threadsafe(epoch_started.set)
            assert release_epoch.wait(timeout=5)
            if not cancel_maintenance:
                raise RuntimeError("synthetic maintenance failure")
            return SimpleNamespace(marked_epoch=7)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = Maintenance
    module.TraceGarbageCollector = Collector
    module.PhysicalTraceCompactor = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        runtime_module, "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS", 0.0
    )
    monkeypatch.setattr(
        runtime_module, "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS", 0.0
    )
    monkeypatch.setattr(
        runtime_module,
        "logger",
        SimpleNamespace(warning=lambda *_args: failure_observed.set()),
    )
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=SimpleNamespace()
    )
    runtime = ConsoleRuntime(SimpleNamespace(_ui_ready=True, app_config={}))
    runtime.set_chat_controller(controller)
    try:
        runtime._schedule_legacy_trace_maintenance(object(), object)
        await asyncio.wait_for(epoch_started.wait(), timeout=5)
        assert controller._trace_maintenance_dispatch_paused.is_set()
        task = runtime._legacy_trace_maintenance_task
        assert task is not None
        if cancel_maintenance:
            for _ in range(2):
                task.cancel()
                cancellation_delivered = asyncio.Event()
                loop.call_soon(cancellation_delivered.set)
                await cancellation_delivered.wait()
                assert not task.done()
                assert controller._trace_maintenance_dispatch_paused.is_set()
        release_epoch.set()
        if cancel_maintenance:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(failure_observed.wait(), timeout=5)
        assert not controller._trace_maintenance_dispatch_paused.is_set()
    finally:
        release_epoch.set()
        await runtime.dispose()


@pytest.mark.asyncio
async def test_runtime_gc_rechecks_submit_ownership_after_pausing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pause boundary must not trust an earlier idle observation."""
    rechecked = asyncio.Event()
    owner_task = asyncio.current_task()
    assert owner_task is not None

    class Maintenance:
        def __init__(self, _database, **_kwargs):
            pass

        def run_batch(self):
            return SimpleNamespace(logical_complete=True, admitted=True)

    class Collector:
        def __init__(self, _database):
            raise AssertionError("GC must defer for the newly registered submit")

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = Maintenance
    module.TraceGarbageCollector = Collector
    module.PhysicalTraceCompactor = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS",
        0.0,
    )
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    pause = controller.pause_trace_maintenance_dispatch
    snapshot = controller._submit_tasks_snapshot

    def pause_and_register():
        pause()
        controller._register_submit_task(owner_task, session.id)

    def observe_snapshot():
        result = snapshot()
        if controller._trace_maintenance_dispatch_paused.is_set() and result:
            rechecked.set()
        return result

    monkeypatch.setattr(
        controller, "pause_trace_maintenance_dispatch", pause_and_register
    )
    monkeypatch.setattr(controller, "_submit_tasks_snapshot", observe_snapshot)
    runtime = ConsoleRuntime(SimpleNamespace(_ui_ready=True, app_config={}))
    runtime.set_chat_controller(controller)
    try:
        runtime._schedule_legacy_trace_maintenance(object(), object)
        await asyncio.wait_for(rechecked.wait(), timeout=5)
        assert not controller._trace_maintenance_dispatch_paused.is_set()
    finally:
        controller._unregister_submit_task(owner_task)
        await runtime.dispose()
