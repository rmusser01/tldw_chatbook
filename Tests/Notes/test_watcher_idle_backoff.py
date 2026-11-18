"""B12: the idle sync watcher backs off to a 30 s ceiling.

The watcher doubles its sleep on every no-change poll but used to cap at
10 s, stat-walking every sync root six times per idle minute (60 walks per
10 idle minutes). The ceiling rises to 30 s -- doubling per quiet tick,
1-2-4-8-16-30 -- while an active (changing) vault keeps polling at the base
interval and any detected change resets the backoff exactly as before.
Cadence only: per-directory mtime checkpointing is unsound (content edits
change file mtime without touching any directory mtime).
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.unit


async def _run_polls(watcher, count: int) -> list[float]:
    """Drive ``count`` run-loop sleeps through the real poll path."""

    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= count:
            await watcher.stop()

    watcher._sleep = sleep
    await watcher.run()
    return sleeps


async def test_idle_backoff_doubles_to_the_thirty_second_ceiling() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    watcher = PollingNotesSyncWatcher(
        lambda: (),
        lambda _root_id: None,
        interval_seconds=1.0,
        jitter=lambda: 1.0,
    )

    sleeps = await _run_polls(watcher, 7)

    # Double per idle tick, capping at the new 30 s ceiling.
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


async def test_any_detected_change_resets_the_backoff_to_the_base_interval() -> None:
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    batches = iter(((), (), ("root-a",), (), ()))
    watcher = PollingNotesSyncWatcher(
        lambda: next(batches),
        lambda _root_id: None,
        interval_seconds=1.0,
        jitter=lambda: 1.0,
    )

    sleeps = await _run_polls(watcher, 6)

    # sleep 1, poll(no change), sleep 2, poll(no change), sleep 4,
    # poll(change -> reset), sleep 1, poll(no change), sleep 2, ...
    assert sleeps == [1.0, 2.0, 4.0, 1.0, 2.0, 4.0]


async def test_idle_poller_walks_the_roots_at_most_twice_per_idle_minute() -> None:
    """Evidence: quiet-root stat walks per 10 idle minutes.

    The 10 s ceiling walked the roots 60 times per 10 idle minutes; the 30 s
    ceiling with the doubled growth curve lands at 24.
    """

    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    watcher = PollingNotesSyncWatcher(
        lambda: (),
        lambda _root_id: None,
        interval_seconds=1.0,
        jitter=lambda: 1.0,
    )

    walks = 0
    elapsed = 0.0
    while elapsed < 600.0:  # 10 idle minutes
        walks += 1
        await watcher.poll_once()  # detects no change -> doubles the interval
        elapsed += watcher._next_sleep_seconds()

    assert walks <= 30, f"{walks} walks per 10 idle minutes (was 60)"
    assert elapsed >= 600.0


def test_default_watch_configuration_caps_idle_polling_at_thirty_seconds() -> None:
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    watcher = PollingNotesSyncWatcher(lambda: (), lambda _root_id: None)
    assert watcher._interval == 1.0
    assert watcher._max_interval == 30.0
    assert config_module.get_notes_sync_watcher_intervals({}) == (1.0, 30.0)
    assert config_module.get_notes_sync_watcher_intervals({"notes": {}}) == (
        1.0,
        30.0,
    )
