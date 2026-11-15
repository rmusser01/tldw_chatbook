"""Causal close/read candidates; copy under Tests/Chat before collecting.

These isolate lifetime defects; they do not attribute a prior mounted failure.
"""

import asyncio
import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Chat.test_console_context_presentation_current import _memory_owner
from Tests.UI.test_console_shared_context_policy import _screen
from tldw_chatbook.Backup_Recovery import runtime_producer_lifetime as lifetime
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.timeout(45)]


def _attach(store, controller):
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    return runtime


@contextmanager
def _monitor(callbacks):
    monitor = sys.monitoring
    slot = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
    monitor.use_tool_id(slot, "context-close-candidate")
    events = {}
    codes = {}
    for code, event, callback in callbacks:
        events.setdefault(event, {})[code] = callback
        codes[code] = codes.get(code, 0) | event
    registered = {}
    for event, selected in events.items():

        def dispatch(code, offset, *arguments, selected=selected):
            selected[code](sys._getframe(1), *arguments)

        registered[event] = dispatch
        assert monitor.register_callback(slot, event, dispatch) is None
    for code, mask in codes.items():
        monitor.set_local_events(slot, code, mask)
    try:
        yield
    finally:
        for code in codes:
            monitor.set_local_events(slot, code, 0)
        for event, callback in registered.items():
            assert monitor.register_callback(slot, event, None) is callback
        monitor.free_tool_id(slot)


async def _cleanup(runtime, database, tasks):
    try:
        try:
            if tasks:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), 10
                )
        finally:
            await asyncio.wait_for(runtime.dispose(timeout_seconds=0), 10)
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_close_waits_for_existing_original_context_native_retirement(
    tmp_path, cancel_close
):
    database, store, controller, _view, tasks = _screen(tmp_path)
    _memory_owner(store, controller)
    runtime = _attach(store, controller)
    session_id = store.active_session_id
    loop = asyncio.get_running_loop()
    entered, bounded_returned, draining = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    release = threading.Event()
    query = store.persistence.get_message_versions.__func__
    query_code = query.__code__
    held = []
    closing = reading = drain_waiter = None

    def query_return(frame, value):
        if frame.f_locals.get("self") is not store.persistence or held:
            return
        read = lifetime.preparation_read_for(controller)
        assert read is not None and read.session_id == session_id
        held.append(read)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(
            20
        ), "candidate cleanup never released original native query"

    def bounded_return(frame, value):
        if frame.f_locals.get("self") is runtime:
            bounded_returned.set()

    def drain_yield(frame, value):
        if (
            frame.f_locals.get("session_id") == session_id
            and held
            and any(read is held[0] for read in frame.f_locals.get("pending", ()))
        ):
            draining.set()

    with _monitor(
        (
            (query_code, sys.monitoring.events.PY_RETURN, query_return),
            (
                ConsoleRuntime._bounded_wait.__code__,
                sys.monitoring.events.PY_RETURN,
                bounded_return,
            ),
            (
                lifetime.drain_preparation_reads.__code__,
                sys.monitoring.events.PY_YIELD,
                drain_yield,
            ),
        )
    ):
        try:
            # Attach before issuance: copying already-existing handles must not
            # accidentally supply the missing observation for this new read.
            reading = asyncio.create_task(
                controller.context_control_presentation_inputs(session_id)
            )
            await asyncio.wait_for(entered.wait(), 10)
            read = held[0]
            assert read in controller._preparation_reads and not read.retired.done()
            closing = asyncio.create_task(
                runtime.close_session(
                    session_id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=session_id
                    ).revision,
                    timeout_seconds=0,
                )
            )
            await asyncio.wait_for(bounded_returned.wait(), 5)
            drain_waiter = asyncio.create_task(draining.wait())
            done, _ = await asyncio.wait(
                {closing, drain_waiter},
                timeout=5,
                return_when=asyncio.FIRST_COMPLETED,
            )
            assert done, "close neither reached physical drain nor completed"
            if closing.done():
                closing.result()  # Preserve the original finalize exception as RED.
            assert draining.is_set() and not closing.done()
            assert not read.retired.done() and any(
                s.id == session_id for s in store.sessions()
            )
            if cancel_close:
                for _ in range(2):
                    closing.cancel()
                    await asyncio.sleep(0)  # Deliver cancellation to the actual drain.
                    assert not closing.done() and not read.retired.done()
                    assert any(s.id == session_id for s in store.sessions())
            release.set()
            if cancel_close:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(closing, 10)
            else:
                await asyncio.wait_for(closing, 10)
            await asyncio.gather(reading, return_exceptions=True)
            assert read.retired.done() and not read.retired.cancelled()
            assert not lifetime.preparation_reads_for(
                controller._preparation_reads, session_id
            )
            assert all(s.id != session_id for s in store.sessions())
            assert query.__code__ is query_code
        finally:
            release.set()
            if drain_waiter is not None:
                drain_waiter.cancel()
            await _cleanup(
                runtime,
                database,
                [t for t in (reading, closing, drain_waiter, *tasks) if t is not None],
            )


class _HookCleanupBarrier:
    """Control the existing engine cleanup seam, retaining real runtime close."""

    lifecycle_owner = None

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cleanup_pending = True

    def begin_close(self):
        pass

    async def close(self):
        self.entered.set()
        await self.release.wait()
        self.cleanup_pending = False


@pytest.mark.asyncio
async def test_original_context_read_refuses_admission_during_session_close(tmp_path):
    database, store, controller, _view, tasks = _screen(tmp_path)
    runtime = _attach(store, controller)
    session_id = store.active_session_id
    hooks = _HookCleanupBarrier()
    runtime._hooks_v2_engines[session_id] = hooks
    admitted = []
    closing = None

    def observe_handle(frame):
        read = frame.f_locals.get("read")
        if (
            read is not None
            and read.creator is controller
            and read.session_id == session_id
        ):
            admitted.append(read)

    with _monitor(
        (
            (
                lifetime._observe_locked.__code__,
                sys.monitoring.events.PY_START,
                observe_handle,
            ),
        )
    ):
        try:
            closing = asyncio.create_task(
                runtime.close_session(
                    session_id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=session_id
                    ).revision,
                    timeout_seconds=0,
                )
            )
            await asyncio.wait_for(hooks.entered.wait(), 10)
            assert session_id in runtime._admission_fenced_sessions
            assert session_id in controller._session_close_generations
            assert any(s.id == session_id for s in store.sessions())
            assert not lifetime.preparation_reads_for(
                controller._preparation_reads, session_id
            )
            # Observe the unmodified real presentation entry. On the defect it
            # succeeds and records a physically retired newly admitted handle.
            refused = False
            try:
                await controller.context_control_presentation_inputs(session_id)
            except RuntimeError:
                refused = True
            assert (
                refused and not admitted
            ), f"closing context admitted {len(admitted)} native handle observations; refused={refused}"
        finally:
            hooks.release.set()
            await _cleanup(
                runtime, database, [t for t in (closing, *tasks) if t is not None]
            )
