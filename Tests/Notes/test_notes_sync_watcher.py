"""Hint-only polling watcher contracts for lasting Notes sync."""

from __future__ import annotations

import asyncio
import inspect
import threading

import pytest


pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


@pytest.mark.asyncio
async def test_polling_watcher_emits_only_unique_root_ids_per_interval() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    clock = _Clock()
    batches = iter((("root-a", "root-a", "root-b"), ("root-a",), ()))
    emitted: list[str] = []
    watcher = PollingNotesSyncWatcher(
        lambda: next(batches),
        emitted.append,
        interval_seconds=5.0,
        clock=clock,
    )

    await watcher.poll_once()
    clock.value = 1.0
    await watcher.poll_once()
    clock.value = 5.0
    await watcher.poll_once()

    assert emitted == ["root-a", "root-b", "root-a"]
    assert all(type(root_id) is str for root_id in emitted)


@pytest.mark.asyncio
async def test_polling_watcher_treats_a_missing_root_as_a_hint_only_gap() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    attempts = 0

    def changed_roots() -> tuple[str, ...]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError
        return ("root-a",)

    emitted: list[str] = []
    watcher = PollingNotesSyncWatcher(changed_roots, emitted.append)

    await watcher.poll_once()
    await watcher.poll_once()

    assert emitted == ["root-a"]


@pytest.mark.asyncio
async def test_missing_root_during_batch_iteration_keeps_its_root_hint() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    def changed_roots():
        yield "root-a"
        raise FileNotFoundError

    emitted: list[str] = []
    watcher = PollingNotesSyncWatcher(changed_roots, emitted.append)

    await watcher.poll_once()

    assert emitted == ["root-a"]


@pytest.mark.asyncio
async def test_polling_watcher_stops_without_emitting_after_stop() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    emitted: list[str] = []
    release = asyncio.Event()

    async def sleep(_seconds: float) -> None:
        await release.wait()

    watcher = PollingNotesSyncWatcher(
        lambda: ("root-a",),
        emitted.append,
        sleep=sleep,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    await watcher.stop()
    release.set()
    await task

    assert emitted == []


@pytest.mark.asyncio
async def test_stop_before_run_cannot_reopen_the_watcher() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    sleeps = 0

    async def sleep(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1

    watcher = PollingNotesSyncWatcher(lambda: (), lambda _root_id: None, sleep=sleep)

    await watcher.stop()
    await watcher.run()

    assert sleeps == 0


@pytest.mark.asyncio
async def test_blocking_hint_source_never_blocks_the_event_loop() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    started = threading.Event()
    release = threading.Event()

    def changed_roots() -> tuple[str, ...]:
        started.set()
        release.wait()
        return ("root-a",)

    emitted: list[str] = []
    watcher = PollingNotesSyncWatcher(changed_roots, emitted.append)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    poll = asyncio.create_task(watcher.poll_once())
    await asyncio.to_thread(started.wait)

    heartbeat = asyncio.create_task(asyncio.sleep(0))
    await heartbeat
    assert not poll.done()

    await poll
    timer.join()
    assert emitted == ["root-a"]


def test_watcher_module_has_no_planner_executor_or_filesystem_dependency() -> None:
    import tldw_chatbook.Notes.notes_sync_watcher as module

    source = inspect.getsource(module)

    assert "notes_sync_reconciler" not in source
    assert "notes_sync_executor" not in source
    assert "notes_sync_filesystem" not in source
