"""Real finite context reads respect the session close lifetime."""

import asyncio
import sys
import threading

import pytest

from Tests.Chat.test_console_context_presentation_current import _memory_owner
from Tests.UI.test_console_shared_context_policy import _screen
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["observe", "close"])
async def test_context_read_stays_owned_until_physical_retirement(tmp_path, route):
    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    _memory_owner(store, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    code = store.persistence.get_message_versions.__func__.__code__
    entered, release, retired = (threading.Event() for _ in range(3))
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    baseline = database._connection_quiescence.connection_count()

    def observe(frame, event, _argument):
        if frame.f_code is code and event == "return":
            entered.set()
            assert release.wait(5)
            retired.set()

    threading.setprofile_all_threads(observe)
    read_task = asyncio.create_task(
        controller.context_control_presentation_inputs("session-1")
    )
    close_task = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        (read,) = controller._preparation_reads
        assert read.task is read_task and not read.retired.done()
        assert database._connection_quiescence.connection_count() == baseline + 1
        if route == "observe":
            assert read in runtime._preparation_reads
        else:
            revision = controller.lifecycle_impact(session_id="session-1").revision
            close_task = asyncio.create_task(
                runtime.close_session(
                    "session-1", expected_revision=revision, timeout_seconds=0
                )
            )
            await asyncio.sleep(0.05)
            assert "session-1" in controller._session_close_generations
            assert not close_task.done(), "Close escaped while its native read was held"
            assert any(row.id == "session-1" for row in store.sessions())
            assert not retired.is_set() and not read.retired.done()
            close_task.cancel()
            await asyncio.sleep(0)
            assert not close_task.done()
        release.set()
        if route == "close":
            with pytest.raises(asyncio.CancelledError):
                await close_task
            with pytest.raises(asyncio.CancelledError):
                await read_task
            assert all(row.id != "session-1" for row in store.sessions())
        else:
            assert await read_task == controller.context_control_inputs("session-1")
        assert retired.is_set() and read.retired.done()
        assert not controller._preparation_reads and not runtime._preparation_reads
        assert database._connection_quiescence.connection_count() == baseline
    finally:
        release.set()
        await asyncio.gather(
            read_task,
            *tasks,
            *([close_task] if close_task else []),
            return_exceptions=True,
        )
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        await runtime.dispose(timeout_seconds=0)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [False, True])
async def test_close_fence_refuses_new_context_display_before_dispatch(
    tmp_path, monkeypatch, custom
):
    database, store, controller, _screen_owner, tasks = _screen(tmp_path)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    calls = []
    result = object()
    if custom:

        def custom_reader(session_id):
            calls.append(session_id)
            return result

        monkeypatch.setattr(controller, "context_control_inputs", custom_reader)
    try:
        normal = await controller.context_control_presentation_inputs("session-1")
        if custom:
            assert normal is result and calls == ["session-1"]
        revision = controller.lifecycle_impact(session_id="session-1").revision
        ticket = controller.begin_session_close("session-1", expected_revision=revision)
        assert any(row.id == "session-1" for row in store.sessions())
        with pytest.raises(RuntimeError, match="Console context display owner changed"):
            await controller.context_control_presentation_inputs("session-1")
        assert calls == (["session-1"] if custom else [])
        assert not controller._preparation_reads and not runtime._preparation_reads
        controller.finalize_session_close(ticket)
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.dispose(timeout_seconds=0)
        database.close()
