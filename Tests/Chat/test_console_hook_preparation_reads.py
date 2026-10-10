"""Finite hook-read ownership helpers without application or permission doubles."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, copy_context
import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.asyncio


async def _until(predicate, timeout=5):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate() and loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert predicate(), "The bounded original worker barrier did not arrive."


def _current():
    return None


@pytest.mark.parametrize("global_cancel", [False, True])
async def test_repeated_cancellation_keeps_same_physical_read_in_both_observers(
    global_cancel,
):
    from tldw_chatbook.Chat.console_hook_preparation import (
        hook_preparation_reads_for,
        run_hook_preparation_read,
    )

    creator = object()
    reads, runtime_reads = set(), set()
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    selected = ContextVar("hook-read-test-context", default="missing")
    token = selected.set("captured")
    observed = []

    def original():
        observed.append((selected.get(), threading.get_ident()))
        entered.set()
        try:
            assert release.wait(10)
            return "original-result"
        finally:
            exited.set()

    existing_tasks = asyncio.all_tasks()
    task = asyncio.create_task(
        run_hook_preparation_read(
            original,
            creator=creator,
            session_id="session-one",
            reads=reads,
            observers=(runtime_reads,),
            require_current=_current,
        )
    )
    try:
        await _until(entered.is_set)
        (record,) = hook_preparation_reads_for(reads, "session-one")
        assert hook_preparation_reads_for(runtime_reads, "session-one") == (record,)
        assert record.task is task
        for _ in range(2):
            targets = asyncio.all_tasks() - existing_tasks if global_cancel else (task,)
            for target in targets:
                target.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert not exited.is_set()
            assert not record.retired.done()
            assert record in reads and record in runtime_reads
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 5)
        assert exited.is_set()
        assert record.retired.done() and not record.retired.cancelled()
        assert reads == runtime_reads == set()
        assert observed[0][0] == "captured"
        assert observed[0][1] != threading.get_ident()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        selected.reset(token)


@pytest.mark.parametrize("cancelled_error", [False, True])
async def test_original_callback_failure_retires_and_preserves_error_identity(
    cancelled_error,
):
    from tldw_chatbook.Chat.console_hook_preparation import run_hook_preparation_read

    reads = set()
    error = (
        asyncio.CancelledError("native callback")
        if cancelled_error
        else ValueError("native callback")
    )
    if cancelled_error:
        # A stale task counter must not turn the callback's exception into an
        # endless caller-cancellation drain.
        asyncio.current_task().cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)

    def original():
        raise error

    with pytest.raises(type(error)) as caught:
        await run_hook_preparation_read(
            original,
            creator=object(),
            session_id=None,
            reads=reads,
            require_current=_current,
        )
    assert caught.value is error
    assert reads == set()


async def test_caller_cancellation_wins_only_after_native_failure_and_cleanup():
    from tldw_chatbook.Chat.console_hook_preparation import run_hook_preparation_read

    reads = set()
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    failure = ValueError("original native failure")

    def original():
        entered.set()
        try:
            assert release.wait(10)
            raise failure
        finally:
            exited.set()

    task = asyncio.create_task(
        run_hook_preparation_read(
            original,
            creator=object(),
            session_id="session-one",
            reads=reads,
            require_current=_current,
        )
    )
    try:
        await _until(entered.is_set)
        (record,) = tuple(reads)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done() and not record.retired.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 5)
        assert exited.is_set() and record.retired.done()
        assert reads == set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("queues_first", [False, True])
async def test_submission_failure_cannot_enter_late_callback(monkeypatch, queues_first):
    from tldw_chatbook.Chat.console_hook_preparation import run_hook_preparation_read

    reads, runtime_reads, queued, records, calls = set(), set(), [], [], []
    error = RuntimeError("worker startup failed")

    def failed_submit(_executor, callback, *args):
        (record,) = tuple(reads)
        records.append(record)
        assert record in runtime_reads
        assert not record.retired.done()
        if queues_first:
            queued.append((callback, args))
        raise error

    loop = asyncio.get_running_loop()
    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_in_executor", failed_submit)
        with pytest.raises(RuntimeError) as caught:
            await run_hook_preparation_read(
                lambda: calls.append("native entered"),
                creator=object(),
                session_id=None,
                reads=reads,
                observers=(runtime_reads,),
                require_current=_current,
            )
    assert caught.value is error
    for callback, args in queued:
        callback(*args)
    assert calls == []
    assert records[0].retired.done()
    assert reads == runtime_reads == set()


async def test_late_runtime_observation_retires_exact_handles_before_submit_tail():
    from tldw_chatbook.Chat.console_hook_preparation import (
        drain_hook_preparation_reads,
        hook_preparation_reads_for,
        observe_hook_preparation_reads,
        run_hook_preparation_read,
    )

    reads, original_runtime, captured_runtime, replacement_runtime = (
        set(),
        set(),
        set(),
        set(),
    )
    entered, release = threading.Event(), threading.Event()
    native_done, tail = asyncio.Event(), asyncio.Event()

    def original():
        entered.set()
        assert release.wait(10)
        return "original"

    async def submit():
        result = await run_hook_preparation_read(
            original,
            creator=object(),
            session_id="session-one",
            reads=reads,
            observers=(original_runtime,),
            require_current=_current,
        )
        native_done.set()
        await tail.wait()
        return result

    task = asyncio.create_task(submit())
    try:
        await _until(entered.is_set)
        (record,) = tuple(reads)
        assert hook_preparation_reads_for(original_runtime, "session-one") == (record,)
        observe_hook_preparation_reads(reads, captured_runtime)
        observe_hook_preparation_reads(reads, captured_runtime)
        assert hook_preparation_reads_for(captured_runtime, "session-one") == (record,)
        assert replacement_runtime == set()
        release.set()
        await asyncio.wait_for(native_done.wait(), 5)
        assert record.retired.done()
        assert (
            reads
            == original_runtime
            == captured_runtime
            == replacement_runtime
            == set()
        )
        assert not task.done()
        assert (
            await asyncio.wait_for(drain_hook_preparation_reads(captured_runtime), 0.5)
            is False
        )
        tail.set()
        assert await asyncio.wait_for(task, 5) == "original"
    finally:
        release.set()
        tail.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_app_scoped_read_is_included_in_session_drain_and_survives_cancel():
    from tldw_chatbook.Chat.console_hook_preparation import (
        drain_hook_preparation_reads,
        hook_preparation_reads_for,
        run_hook_preparation_read,
    )

    reads = set()
    entered, release = threading.Event(), threading.Event()

    def original():
        entered.set()
        assert release.wait(10)
        return None

    task = asyncio.create_task(
        run_hook_preparation_read(
            original,
            creator=object(),
            session_id=None,
            reads=reads,
            require_current=_current,
        )
    )
    drain = None
    try:
        await _until(entered.is_set)
        (record,) = tuple(reads)
        assert hook_preparation_reads_for(reads, "any-session") == (record,)
        drain = asyncio.create_task(drain_hook_preparation_reads(reads, "any-session"))
        await asyncio.sleep(0)  # Enter the drain before cancelling its waiter.
        for _ in range(2):
            drain.cancel()
            await asyncio.sleep(0.02)
            assert not drain.done() and not record.retired.done()
        release.set()
        assert await asyncio.wait_for(asyncio.shield(drain), 5) is True
        await asyncio.wait_for(task, 5)
        assert reads == set()
    finally:
        release.set()
        await asyncio.gather(
            *(item for item in (task, drain) if item is not None),
            return_exceptions=True,
        )


async def test_session_binding_is_exact_task_and_restores_nested_scopes():
    from tldw_chatbook.Chat.console_hook_preparation import (
        bind_hook_preparation_session,
        hook_preparation_session_for,
    )

    creator, other = object(), object()

    async def child():
        return hook_preparation_session_for(creator)

    assert hook_preparation_session_for(creator) is None
    with bind_hook_preparation_session(creator, "session-one"):
        assert hook_preparation_session_for(creator) == "session-one"
        assert await asyncio.create_task(child()) is None
        with pytest.raises(RuntimeError):
            hook_preparation_session_for(other)
        with pytest.raises(ValueError):
            with bind_hook_preparation_session(creator, "session-two"):
                assert hook_preparation_session_for(creator) == "session-two"
                raise ValueError("scope ended")
        assert hook_preparation_session_for(creator) == "session-one"
    assert hook_preparation_session_for(creator) is None


async def test_captured_source_is_available_only_inside_original_worker_actor():
    from tldw_chatbook.Chat.console_hook_preparation import (
        hook_preparation_source_for,
        run_hook_preparation_read,
    )

    creator, source, reads = object(), object(), set()
    observed = []

    def original():
        observed.append(hook_preparation_source_for(creator))
        assert hook_preparation_source_for(object()) is None
        copied = copy_context()
        foreign = threading.Thread(
            target=lambda: observed.append(
                copied.run(hook_preparation_source_for, creator)
            )
        )
        foreign.start()
        foreign.join(5)
        assert not foreign.is_alive()
        return "original"

    assert hook_preparation_source_for(creator) is None
    assert (
        await run_hook_preparation_read(
            original,
            creator=creator,
            session_id=None,
            reads=reads,
            require_current=_current,
            source=source,
        )
        == "original"
    )
    assert observed == [source, None]
    assert hook_preparation_source_for(creator) is None
    assert reads == set()


@pytest.mark.parametrize("displacement", ["before", "worker", "after"])
async def test_original_source_check_refuses_before_effect_or_publication(
    monkeypatch,
    displacement,
):
    from tldw_chatbook.Chat.console_hook_preparation import run_hook_preparation_read

    reads, queued, calls = set(), [], []
    source = SimpleNamespace(current=displacement != "before")
    error = RuntimeError("original source changed")

    def require_current():
        if not source.current:
            raise error

    def original():
        calls.append("original")
        if displacement == "after":
            source.current = False
        return "obsolete"

    if displacement != "worker":
        with pytest.raises(RuntimeError) as caught:
            await run_hook_preparation_read(
                original,
                creator=object(),
                session_id="session-one",
                reads=reads,
                require_current=require_current,
            )
    else:
        loop = asyncio.get_running_loop()
        executor = loop.run_in_executor
        pending = loop.create_future()

        def defer(pool, callback, *args):
            queued.append((pool, callback, args))
            return pending

        with monkeypatch.context() as patch:
            patch.setattr(loop, "run_in_executor", defer)
            task = asyncio.create_task(
                run_hook_preparation_read(
                    original,
                    creator=object(),
                    session_id="session-one",
                    reads=reads,
                    require_current=require_current,
                )
            )
            await _until(lambda: bool(queued))
        source.current = False
        pool, callback, args = queued[0]
        try:
            pending.set_result(await executor(pool, callback, *args))
        except BaseException as failure:
            pending.set_exception(failure)
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(task, 5)
    assert caught.value is error
    assert calls == (["original"] if displacement == "after" else [])
    assert reads == set()


async def test_stop_iteration_callback_becomes_coroutine_error_and_retires():
    from tldw_chatbook.Chat.console_hook_preparation import (
        hook_preparation_reads_for,
        run_hook_preparation_read,
    )

    reads, records = set(), []
    original_error = StopIteration("original synchronous callback")

    def original():
        (record,) = hook_preparation_reads_for(reads)
        records.append(record)
        assert not record.retired.done()
        raise original_error

    with pytest.raises(RuntimeError, match="coroutine raised StopIteration") as caught:
        await asyncio.wait_for(
            run_hook_preparation_read(
                original,
                creator=object(),
                session_id=None,
                reads=reads,
                require_current=_current,
            ),
            5,
        )
    assert caught.value.__cause__ is original_error
    assert records[0].retired.done()
    assert reads == set()
