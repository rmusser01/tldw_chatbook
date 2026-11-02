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


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_stock_context_display_read_retires_before_session_close(
    tmp_path, cancel_close
):
    from tldw_chatbook.Chat.console_runtime import (
        CONSOLE_SESSION_CLOSE_GRACE_SECONDS,
        ConsoleRuntime,
    )

    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    _memory_owner(store, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    code = store.persistence.get_message_versions.__func__.__code__
    entered, release, returned = (threading.Event() for _ in range(3))
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    baseline = database._connection_quiescence.connection_count()
    read_task = closing = None

    def observe(frame, event, _argument):
        if frame.f_code is code and event == "return":
            # Original query has returned inside the stock finite callback;
            # its real worker connection remains owned until callback exit.
            entered.set()
            assert release.wait(5)
            returned.set()

    threading.setprofile_all_threads(observe)
    try:
        read_task = asyncio.create_task(
            controller.context_control_presentation_inputs("session-1")
        )
        assert await asyncio.to_thread(entered.wait, 5)
        (read,) = tuple(controller._preparation_reads)
        assert not read.retired.done()
        assert database._connection_quiescence.connection_count() == baseline + 1
        closing = asyncio.create_task(
            runtime.close_session(
                "session-1",
                expected_revision=controller.lifecycle_impact(
                    session_id="session-1"
                ).revision,
            )
        )
        assert CONSOLE_SESSION_CLOSE_GRACE_SECONDS == 2.0
        await asyncio.sleep(CONSOLE_SESSION_CLOSE_GRACE_SECONDS + 0.05)
        assert not closing.done(), (
            "Close escaped before original context-read retirement",
            closing.exception() if closing.done() else None,
        )
        assert read in controller._preparation_reads
        assert read in runtime._preparation_reads
        assert not read.retired.done() and not returned.is_set()
        assert any(row.id == "session-1" for row in store.sessions())
        if cancel_close:
            for _ in range(3):
                closing.cancel()
                await asyncio.sleep(0)
                assert not closing.done()
                assert not read.retired.done()
                assert (
                    database._connection_quiescence.connection_count() == baseline + 1
                )
        release.set()
        if cancel_close:
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await closing
        await asyncio.gather(read_task, return_exceptions=True)
        assert returned.is_set() and read.retired.done()
        assert not controller._preparation_reads and not runtime._preparation_reads
        assert database._connection_quiescence.connection_count() == baseline
        assert not any(row.id == "session-1" for row in store.sessions())
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (read_task, closing, *tasks) if task is not None),
            return_exceptions=True,
        )
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        await runtime.dispose()
        database.close()


@pytest.mark.asyncio
async def test_stock_context_display_refuses_existing_session_close_fence(tmp_path):
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    database, _store, controller, _screen_owner, tasks = _screen(tmp_path)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(controller.store)
    runtime.set_chat_controller(controller)
    ticket = controller.begin_session_close(
        "session-1",
        expected_revision=controller.lifecycle_impact(session_id="session-1").revision,
    )
    try:
        with pytest.raises(RuntimeError, match="Console context display owner changed"):
            await controller.context_control_presentation_inputs("session-1")
        assert not controller._preparation_reads and not runtime._preparation_reads
        # Explicit live action reads keep their preceding callable contract.
        assert len(controller.context_control_inputs("session-1")) == 3
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        controller.finalize_session_close(ticket)
        await runtime.dispose()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["standalone-attach", "controller-rebind", "store-rebind"]
)
async def test_first_stock_context_probe_refuses_changed_runtime_binding(
    tmp_path, change
):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_context_repository import may_hold_branch_memory

    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    _memory_owner(store, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    if change != "standalone-attach":
        runtime.set_chat_controller(controller)
    code = may_hold_branch_memory.__code__
    versions_code = store.persistence.get_message_versions.__func__.__code__
    lineage_calls = []
    entered, release, returned = (threading.Event() for _ in range(3))
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    baseline = database._connection_quiescence.connection_count()
    task = replacement_controller = replacement_store = None

    def observe(frame, event, _argument):
        if frame.f_code is versions_code and event == "call":
            lineage_calls.append(threading.get_ident())
        if frame.f_code is code and event == "return":
            # Hold the first stock memory probe inside its original finite
            # callback and native connection, without replacing the reader.
            entered.set()
            assert release.wait(5)
            returned.set()

    threading.setprofile_all_threads(observe)
    try:
        task = asyncio.create_task(
            controller.context_control_presentation_inputs("session-1")
        )
        assert await asyncio.to_thread(entered.wait, 5)
        (read,) = tuple(controller._preparation_reads)
        assert read.creator is controller and not read.retired.done()
        assert database._connection_quiescence.connection_count() == baseline + 1
        if change == "standalone-attach":
            assert getattr(controller, "_hooks_v2_runtime", None) is None
            runtime.set_chat_controller(controller)
            assert read in runtime._preparation_reads
        elif change == "controller-rebind":
            replacement_controller = ConsoleChatController(
                store=store,
                provider_gateway=controller.provider_gateway,
                context_repository=controller._context_repository,
            )
            runtime.set_chat_controller(replacement_controller)
            assert controller._hooks_v2_runtime is runtime
            assert runtime.chat_controller is replacement_controller
        else:
            replacement_store = ConsoleChatStore()
            runtime.set_chat_store(replacement_store)
            assert controller._hooks_v2_runtime is runtime
            assert controller.store is store
            assert runtime.chat_store is replacement_store
        assert not task.done() and not read.retired.done()
        assert not returned.is_set()
        assert database._connection_quiescence.connection_count() == baseline + 1
        release.set()
        with pytest.raises(RuntimeError, match="Console context display owner changed"):
            await task
        assert lineage_calls == [], "stale first probe reached a second lineage read"
        assert returned.is_set() and read.retired.done()
        assert not controller._preparation_reads and not runtime._preparation_reads
        assert database._connection_quiescence.connection_count() == baseline
    finally:
        release.set()
        await asyncio.gather(
            *(pending for pending in (task, *tasks) if pending is not None),
            return_exceptions=True,
        )
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        # Restore only after the original task's outcome/native retirement;
        # this cleanup cannot make the stale probe appear current again.
        runtime.set_chat_store(store)
        runtime.set_chat_controller(controller)
        await runtime.dispose()
        if replacement_controller is not None:
            await replacement_controller.shutdown()
        if replacement_store is not None:
            replacement_store.end_app_runtime()
        database.close()
