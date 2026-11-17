"""Interpreter handoffs belong to physical finite reads, including overlap."""

import asyncio
import sys
import threading

import pytest

from Tests.Chat.test_console_hook_preparation_reads import _until
from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def interval():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(0.005)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


async def _read(
    callback, *, reads=None, require_current=lambda: None, submission_slot=None
):
    return await run_preparation_read(
        callback,
        creator=callback,
        session_id=None,
        reads=set() if reads is None else reads,
        require_current=require_current,
        **({"submission_slot": submission_slot} if submission_slot is not None else {}),
    )


async def test_original_callback_gets_short_handoffs_then_restores_interval():
    original = sys.getswitchinterval()
    observed = await _read(sys.getswitchinterval)
    assert observed <= 0.001
    assert sys.getswitchinterval() == original


async def test_an_already_shorter_interval_is_preserved():
    sys.setswitchinterval(0.00005)
    original = sys.getswitchinterval()
    assert await _read(sys.getswitchinterval) == original
    assert sys.getswitchinterval() == original


@pytest.mark.parametrize("where", ["callback", "source"])
@pytest.mark.parametrize("scheduled", [False, True])
async def test_original_error_restores_interval_after_physical_exit(where, scheduled):
    original = sys.getswitchinterval()
    error = ValueError("original failure")
    observed = []
    slot = asyncio.Semaphore(1) if scheduled else None

    def callback():
        observed.append(sys.getswitchinterval())
        if where == "callback":
            raise error

    def current():
        if where == "source" and observed:
            raise error

    with pytest.raises(ValueError) as raised:
        await _read(callback, require_current=current, submission_slot=slot)
    assert raised.value is error
    assert sys.getswitchinterval() == original
    assert observed[0] <= 0.001
    assert slot is None or not slot.locked()


async def test_overlapping_callbacks_restore_only_after_last_physical_exit():
    original = sys.getswitchinterval()
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def callback(index):
        entered[index].set()
        assert release[index].wait(5)

    tasks = [asyncio.create_task(_read(lambda i=i: callback(i))) for i in range(2)]
    try:
        await _until(lambda: all(event.is_set() for event in entered))
        assert sys.getswitchinterval() <= 0.001
        release[0].set()
        await tasks[0]
        assert not tasks[1].done()
        assert sys.getswitchinterval() <= 0.001
        release[1].set()
        await tasks[1]
        assert sys.getswitchinterval() == original
    finally:
        for event in release:
            event.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_repeated_cancellation_keeps_handoffs_until_callback_retires():
    original = sys.getswitchinterval()
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    reads = set()

    def callback():
        entered.set()
        try:
            assert release.wait(5)
        finally:
            exited.set()

    task = asyncio.create_task(_read(callback, reads=reads))
    try:
        await _until(entered.is_set)
        assert sys.getswitchinterval() <= 0.001
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not exited.is_set()
            assert sys.getswitchinterval() <= 0.001
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set() and not reads
        assert sys.getswitchinterval() == original
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_later_different_external_setting_is_not_overwritten():
    observed = []

    def callback():
        observed.append(sys.getswitchinterval())
        sys.setswitchinterval(0.002)

    await _read(callback)
    assert sys.getswitchinterval() == 0.002
    assert observed[0] <= 0.001


@pytest.mark.parametrize("enqueue", [False, True])
@pytest.mark.parametrize("scheduled", [False, True])
async def test_failed_submission_never_changes_interval_or_enters_late_callback(
    monkeypatch, enqueue, scheduled
):
    original = sys.getswitchinterval()
    loop = asyncio.get_running_loop()
    queued, called, reads = [], [], set()
    error = RuntimeError("original submission failure")
    slot = asyncio.Semaphore(1) if scheduled else None

    def submit(executor, callback, *args):
        if enqueue:
            queued.append((callback, args))
        raise error

    monkeypatch.setattr(loop, "run_in_executor", submit)
    with pytest.raises(RuntimeError) as raised:
        await _read(lambda: called.append(True), reads=reads, submission_slot=slot)
    assert raised.value is error
    for callback, args in queued:
        assert callback(*args) is None
    assert not called and not reads
    assert sys.getswitchinterval() == original
    if slot is not None:
        assert not slot.locked(), "submission failure kept the display permit"


@pytest.mark.parametrize("retirement", ["cancel", "source_closed"])
async def test_queued_display_retires_without_submitting_callback(retirement):
    slot = asyncio.Semaphore(1)
    await slot.acquire()
    reads, called = set(), []
    closed = False

    def current():
        if closed:
            raise RuntimeError("source closed while queued")

    task = asyncio.create_task(
        _read(
            lambda: called.append(True),
            reads=reads,
            require_current=current,
            submission_slot=slot,
        )
    )
    try:
        await _until(lambda: bool(reads) or task.done())
        assert not task.done()
        (read,) = reads
        assert read._producer is None and not called
        if retirement == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert slot.locked(), "queued cancellation released another reader's permit"
            slot.release()
        else:
            closed = True
            slot.release()
            with pytest.raises(RuntimeError, match="source closed while queued"):
                await task
        assert not called and not reads and read.retired.done()
        assert await _read(lambda: "next", submission_slot=slot) == "next"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_active_display_keeps_permit_through_repeated_cancel_until_physical_return():
    slot = asyncio.Semaphore(1)
    entered, release = threading.Event(), threading.Event()
    reads, later_reads, later_called = set(), set(), []

    def callback():
        entered.set()
        assert release.wait(10), "active display callback was not released"

    first = asyncio.create_task(_read(callback, reads=reads, submission_slot=slot))
    later = None
    try:
        await _until(lambda: entered.is_set() or first.done())
        assert entered.is_set()
        later = asyncio.create_task(
            _read(
                lambda: later_called.append(True),
                reads=later_reads,
                submission_slot=slot,
            )
        )
        await _until(lambda: bool(later_reads))
        for _ in range(2):
            first.cancel()
            await asyncio.sleep(0)
            assert not first.done() and reads and slot.locked()
            assert not later_called
            assert next(iter(later_reads))._producer is None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await later
        assert later_called == [True] and not reads and not later_reads
        assert not slot.locked()
    finally:
        release.set()
        await asyncio.gather(
            first, *([later] if later is not None else []), return_exceptions=True
        )
