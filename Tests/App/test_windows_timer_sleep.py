"""Textual timer sleeps stay off the default executor on Windows (TASK-38462).

Textual 8.2.8 runs every timer sleep on Windows as a blocking job on the loop's
default executor, and every cancel as a second job (``textual/_win_sleep.py``),
so 24 h timers pin the pool that ``thread=True`` workers need (TASK-38206).
These tests simulate win32 on any platform by patching ``sys.platform`` around
the install call only. Intervals use ``time.perf_counter``: Windows
``time.monotonic`` ticks every 15.625 ms, too coarse for a 30 ms wait.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import textual
import textual._time
import textual.timer
from textual.app import App

from tldw_chatbook.Utils import text_selection_crash_guard as guard


class _RecordingPool(ThreadPoolExecutor):
    """A real pool that remembers what it was asked to run."""

    def __init__(self, max_workers: int = 4) -> None:
        super().__init__(max_workers=max_workers)
        self.jobs: list[object] = []

    def submit(self, fn, /, *args, **kwargs):
        self.jobs.append(fn)
        return super().submit(fn, *args, **kwargs)


def _install_as_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as platform:
        platform.setattr(sys, "platform", "win32")
        guard.install_windows_timer_sleep()


@pytest.fixture
def windows_timer_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[_RecordingPool]:
    """Install the Windows timer sleep with a recording pool; undo both after."""
    monkeypatch.setattr(textual.timer, "sleep", textual.timer.sleep)
    pool = _RecordingPool()
    monkeypatch.setattr(guard, "_windows_timer_pool", pool)
    _install_as_windows(monkeypatch)
    yield pool
    pool.shutdown(wait=True)


async def _default_executor_spy() -> _RecordingPool:
    spy = _RecordingPool()
    asyncio.get_running_loop().set_default_executor(spy)
    return spy


@pytest.mark.asyncio
async def test_long_timer_waits_hold_no_thread(windows_timer_sleep, monkeypatch):
    default = await _default_executor_spy()
    _install_as_windows(monkeypatch)  # a second Load keeps the same sleep
    day = asyncio.create_task(textual.timer.sleep(86400))
    second = asyncio.create_task(textual.timer.sleep(guard._WINDOWS_COARSE_TIMER_WAIT))
    await asyncio.sleep(0.05)
    assert not day.done() and not second.done()
    started = time.perf_counter()
    day.cancel()
    second.cancel()
    for task in (day, second):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert time.perf_counter() - started < 0.5
    assert default.jobs == [] and windows_timer_sleep.jobs == []


@pytest.mark.asyncio
async def test_short_timer_waits_use_the_dedicated_pool(windows_timer_sleep):
    default = await _default_executor_spy()
    started = time.perf_counter()
    await textual.timer.sleep(0.03)
    assert time.perf_counter() - started >= 0.025
    await textual.timer.sleep(0.0004)  # under Textual's 0.5 ms floor: no wait
    assert windows_timer_sleep.jobs == [time.sleep]
    assert default.jobs == []


@pytest.mark.asyncio
async def test_cancelled_short_wait_returns_at_once_and_frees_its_thread(
    windows_timer_sleep, monkeypatch
):
    single = _RecordingPool(max_workers=1)
    monkeypatch.setattr(guard, "_windows_timer_pool", single)
    default = await _default_executor_spy()
    wait = asyncio.create_task(textual.timer.sleep(0.2))
    await asyncio.sleep(0.02)  # the one pool thread is now inside time.sleep
    started = time.perf_counter()
    wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait
    assert time.perf_counter() - started < 0.05, "a cancel must not wait for a thread"
    # The thread finishes its bounded sleep and serves the next wait.
    await asyncio.wait_for(textual.timer.sleep(0.01), timeout=1.0)
    assert single.jobs == [time.sleep, time.sleep] and default.jobs == []
    single.shutdown(wait=True)


@pytest.mark.asyncio
async def test_textual_timers_never_reach_the_default_executor(windows_timer_sleep):
    ticks: list[str] = []
    app = App()
    async with app.run_test() as pilot:
        default = await _default_executor_spy()
        day = app.set_interval(86400, lambda: ticks.append("day"))
        frame = app.set_interval(1 / 60, lambda: ticks.append("frame"))
        await pilot.pause(0.3)
        day.stop()
        frame.stop()
        await pilot.pause()
    assert "frame" in ticks and "day" not in ticks
    assert time.sleep not in default.jobs
    assert windows_timer_sleep.jobs and set(windows_timer_sleep.jobs) == {time.sleep}


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_other_platforms_keep_textuals_sleep(monkeypatch, platform):
    # On win32 a TldwCli booted earlier in the same process leaves the Windows
    # sleep installed process-wide; start from Textual's own sleep rather than
    # from whatever is installed. (Off win32 nothing ever installs it, so no
    # earlier-boot case exists to test, and this test creates no pool.)
    monkeypatch.setattr(textual.timer, "sleep", textual._time.sleep)
    monkeypatch.setattr(guard, "_windows_timer_pool", None)
    with monkeypatch.context() as patched:
        patched.setattr(sys, "platform", platform)
        guard.install_windows_timer_sleep()
    assert textual.timer.sleep is textual._time.sleep
    assert guard._windows_timer_pool is None


def test_load_installs_the_windows_timer_sleep(monkeypatch):
    monkeypatch.setattr(textual.timer, "sleep", textual.timer.sleep)
    monkeypatch.setattr(guard, "_windows_timer_pool", None)

    async def load() -> None:
        with monkeypatch.context() as platform:
            platform.setattr(sys, "platform", "win32")
            guard.ThreadWorkerContextGuard().on_load()

    asyncio.run(load())
    assert textual.timer.sleep is guard._windows_timer_sleep
    assert guard._windows_timer_pool is not None
    guard._windows_timer_pool.shutdown(wait=True)


def test_textual_internals_the_windows_sleep_replaces_are_unchanged():
    """Fail loudly when a Textual upgrade moves what the fix replaces.

    The fix swaps ``textual.timer.sleep``. That holds only while Timer._run
    awaits that module global, no other Textual module binds
    ``textual._time.sleep`` or ``textual._win_sleep``, and the Windows sleep
    still parks a default-executor thread. Once upstream stops doing that,
    drop ``install_windows_timer_sleep`` instead of updating this test.
    """
    assert "sleep" in textual.timer.Timer._run.__code__.co_names
    assert hasattr(textual.timer, "sleep")
    root = Path(textual.__file__).parent
    binders = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if re.search(
            r"from textual\._time import [^\n]*\bsleep\b|_win_sleep|_time\.sleep\b",
            path.read_text(encoding="utf-8"),
        )
    )
    assert binders == ["_time.py", "timer.py"]
    win_sleep = (root / "_win_sleep.py").read_text(encoding="utf-8")
    assert win_sleep.count("run_in_executor(None,") == 2, (
        "Textual's Windows sleep no longer uses the default executor for its "
        "wait and cancel; check whether install_windows_timer_sleep is still needed"
    )
