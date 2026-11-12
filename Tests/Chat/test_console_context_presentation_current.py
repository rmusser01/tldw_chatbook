"""Current memory skipping and exact finite context-display owners."""

import asyncio
import sys
import threading

import pytest

from Tests.Chat.test_console_memory_presence import _reset
from Tests.UI.test_console_shared_context_policy import _screen

pytestmark = pytest.mark.bootstrap_profile


def _memory_owner(store, controller):
    session = store.sessions()[0]
    row = store.messages_for_session(session.id)[0]
    controller._context_repository.insert_memory_selection(
        _reset(session.persisted_conversation_id, row.persisted_message_id)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", [False, True])
async def test_display_preserves_actual_no_memory_lineage_skip(tmp_path, memory):
    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    if memory:
        _memory_owner(store, controller)
    code = controller._durable_context_snapshots.__func__.__code__
    reads = []
    main = threading.get_ident()
    previous, previous_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code is code:
            reads.append(threading.get_ident())

    threading.setprofile_all_threads(observe)
    try:
        shown = await controller.context_control_presentation_inputs("session-1")
        assert len(reads) == int(memory)
        assert all(thread != main for thread in reads)
        assert shown == controller.context_control_inputs("session-1")
        assert not controller._preparation_reads
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("all_tasks", [False, True])
async def test_context_display_retains_exact_native_handle_through_cancel(tmp_path, all_tasks):
    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    _memory_owner(store, controller)
    code = store.persistence.get_message_versions.__func__.__code__
    entered, release, retired = (threading.Event() for _ in range(3))
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    existing = asyncio.all_tasks()
    baseline = database._connection_quiescence.connection_count()

    def observe(frame, event, _argument):
        if frame.f_code is code and event == "return":
            # The original version query has acquired its real worker handle;
            # the enclosing finite callback still owns it until this returns.
            entered.set()
            assert release.wait(5)
            retired.set()

    threading.setprofile_all_threads(observe)
    task = asyncio.create_task(controller.context_control_presentation_inputs("session-1"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert controller._preparation_reads
        assert database._connection_quiescence.connection_count() == baseline + 1
        for _ in range(3):
            selected = asyncio.all_tasks() - existing if all_tasks else [task]
            for pending in selected:
                pending.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not retired.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert retired.is_set() and not controller._preparation_reads
        assert database._connection_quiescence.connection_count() == baseline
    finally:
        release.set()
        await asyncio.gather(task, *tasks, return_exceptions=True)
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        database.close()


@pytest.mark.asyncio
async def test_custom_context_reader_keeps_issuing_thread_and_live_result(tmp_path, monkeypatch):
    database, _store, controller, _screen_owner, tasks = _screen(tmp_path)
    calls = []
    result = object()

    def custom(session_id):
        calls.append((threading.get_ident(), session_id))
        return result

    monkeypatch.setattr(controller, "context_control_inputs", custom)
    try:
        assert await controller.context_control_presentation_inputs("session-1") is result
        assert calls == [(threading.get_ident(), "session-1")]
        assert not controller._preparation_reads
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["warm", "queued"])
@pytest.mark.parametrize("closed", ["controller", "screen", "runtime"])
async def test_no_config_display_never_calls_custom_reader_after_closure(
    tmp_path, monkeypatch, entry, closed
):
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleContextReadSnapshot,
    )

    database, _store, controller, screen, tasks = _screen(tmp_path)
    runtime = ConsoleRuntime(app=None)
    screen._console_runtime_fallback = runtime
    screen.app_instance.app_config = {}
    snapshot = ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
    assert snapshot.config_projection is None
    result = controller.context_control_inputs("session-1")
    calls = []

    def custom(session_id):
        calls.append((threading.get_ident(), session_id))
        return result

    monkeypatch.setattr(controller, "context_control_inputs", custom)
    locked = False
    try:
        assert await snapshot.warm(controller, "session-1")
        assert calls == [(threading.get_ident(), "session-1")]
        snapshot.at -= 2
        if entry == "queued":
            await snapshot.lock.acquire()
            locked = True
            assert snapshot.inputs(controller, "session-1") is result
            assert len(tasks) == 1
            await asyncio.sleep(0)
        if closed == "controller":
            await controller.shutdown()
            assert controller._disposed
        elif closed == "runtime":
            await runtime.dispose()
            assert runtime._disposed
        else:
            screen._closing = True
        if locked:
            snapshot.lock.release()
            locked = False
            await asyncio.gather(*tasks)
        else:
            assert not await snapshot.warm(controller, "session-1")
        assert calls == [(threading.get_ident(), "session-1")]
        # Direct explicit action readers retain their existing callable ABI.
        assert controller.context_control_inputs("session-1") is result
        assert len(calls) == 2
    finally:
        if locked:
            snapshot.lock.release()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose()
        await controller.shutdown()
        database.close()
