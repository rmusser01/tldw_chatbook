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


async def _read(callback, *, reads=None, require_current=lambda: None):
    return await run_preparation_read(
        callback,
        creator=callback,
        session_id=None,
        reads=set() if reads is None else reads,
        require_current=require_current,
    )


async def test_original_callback_gets_short_handoffs_then_restores_interval():
    original = sys.getswitchinterval()
    observed = await _read(sys.getswitchinterval)
    assert observed <= 0.001
    assert sys.getswitchinterval() == original


async def test_an_already_shorter_interval_is_preserved():
    sys.setswitchinterval(0.0005)
    original = sys.getswitchinterval()
    assert await _read(sys.getswitchinterval) == original
    assert sys.getswitchinterval() == original


@pytest.mark.parametrize("where", ["callback", "source"])
async def test_original_error_restores_interval_after_physical_exit(where):
    original = sys.getswitchinterval()
    error = ValueError("original failure")
    observed = []

    def callback():
        observed.append(sys.getswitchinterval())
        if where == "callback":
            raise error

    def current():
        if where == "source" and observed:
            raise error

    with pytest.raises(ValueError) as raised:
        await _read(callback, require_current=current)
    assert raised.value is error
    assert sys.getswitchinterval() == original
    assert observed[0] <= 0.001


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
async def test_failed_submission_never_changes_interval_or_enters_late_callback(
    monkeypatch, enqueue
):
    original = sys.getswitchinterval()
    loop = asyncio.get_running_loop()
    queued, called, reads = [], [], set()
    error = RuntimeError("original submission failure")

    def submit(executor, callback, *args):
        if enqueue:
            queued.append((callback, args))
        raise error

    monkeypatch.setattr(loop, "run_in_executor", submit)
    with pytest.raises(RuntimeError) as raised:
        await _read(lambda: called.append(True), reads=reads)
    assert raised.value is error
    for callback, args in queued:
        assert callback(*args) is None
    assert not called and not reads
    assert sys.getswitchinterval() == original
