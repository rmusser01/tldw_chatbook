"""A send's durable commit keeps the Console's whole syncs quiet.

TASK-33620.15.4 (measured live 2026-10-23, 30 sends per arm, Anthropic haiku
through the review proxy): the send's durable turn commit runs on a worker
thread for 140-180 ms while the send waits on it. On the base build a
deferral left the UI loop idle for the rest of that commit after the run
poll's first tick (its whole-sync replay painted 0.2 s later). Once the
projection stopped deferring, a second poll tick and a re-armed whole sync
painted inside the commit (UI-loop callbacks in it 86 -> 102 ms), the commit
took 218 ms instead of 180 ms and warm sends reached the provider a median
31 ms later.

Now, while the run poll ticks and the commit is in flight, the commit's
first whole sync runs as before and opens a hold until the poll's next tick:
explicit syncs meanwhile leave their request for later, that next tick skips
and ends the hold, and the tick after it serves what was held (never two
skipped ticks; each commit holds once). These tests drive the real
ChatScreen on the mounted send harness. Its in-memory commit is over long
before a tick could land inside it, so the hold tests open the commit's
window by hand around the ticks. One test pins that a real send's commit
opens that window on the very store the screen's hold reads, and
``Tests/Chat/test_console_durable_commit_window.py`` pins the window on a
file-backed commit held on its thread.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from Tests.UI.test_console_config_sync_cadence import _held_run
from Tests.UI.test_console_send_acknowledgement import (
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)
from tldw_chatbook.Chat import console_native_commit
from tldw_chatbook.UI.Console_Modules import config_sync

pytestmark = pytest.mark.bootstrap_profile


def _commit_window(store):
    """The commit's in-flight window (none on the base: nothing to open)."""
    window = getattr(console_native_commit, "durable_commit_window", None)
    return window(store) if window is not None else contextlib.nullcontext()


class _Syncs:
    """Whole syncs that got past their entry in the test's own task."""

    def __init__(self, monkeypatch, console) -> None:
        self.owner = asyncio.current_task()
        self.count = 0
        real = config_sync.project_console_tick

        def counting(screen, *args, **kwargs):
            if screen is console and asyncio.current_task() is self.owner:
                self.count += 1
            return real(screen, *args, **kwargs)

        monkeypatch.setattr(config_sync, "project_console_tick", counting)


@pytest.mark.asyncio
async def test_a_durable_commit_runs_its_first_sync_then_holds_until_the_next_tick(
    monkeypatch,
):
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            timer = console._console_transcript_sync_timer
            assert timer is not None, "the run start started no poll"
            timer.pause()
            syncs = _Syncs(monkeypatch, console)

            async def settled() -> None:
                await until(lambda: not console._console_sync_in_progress, timeout=30)

            async def tick() -> None:
                await settled()
                await timer._callback()

            async def explicit() -> bool:
                await settled()
                return await console._sync_native_console_chat_ui()

            try:
                await tick()
                assert syncs.count == 1, "the poll's tick ran no sync"
                with _commit_window(console._console_chat_store):
                    # The commit's first sync runs (the base painted it too).
                    await tick()
                    assert syncs.count == 2, "the commit's first tick was held"

                    # Until the poll's next tick, an explicit sync waits.
                    console._console_sync_requested = False
                    result = await explicit()
                    assert syncs.count == 2, (
                        "an explicit sync ran while the durable commit was in flight"
                    )
                    assert result is False
                    assert console._console_sync_requested, (
                        "the held explicit sync left no request for a later tick"
                    )

                    # The next tick skips: the rest of the commit stays quiet.
                    await tick()
                    assert syncs.count == 2, (
                        "a second poll tick synced while the durable commit ran"
                    )

                    # The tick after it runs and serves the held request (never
                    # two skipped ticks in a row: one poll interval at most).
                    await tick()
                    assert syncs.count == 3, "two poll ticks in a row were held"

                    # Each commit holds once: later syncs in it run.
                    await settled()
                    await explicit()
                    assert syncs.count == 4, "the commit held a second time"

                # The commit ended: nothing is held.
                await tick()
                await tick()
                assert syncs.count == 6
                await settled()
                console._console_sync_requested = False
                await explicit()
                assert syncs.count == 7
            finally:
                gateway.validation_release.set()
                timer.resume()


@pytest.mark.asyncio
async def test_without_a_run_poll_a_durable_commit_holds_no_sync(monkeypatch):
    """Nothing else would serve a held request: with no poll, syncs run."""
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            timer = console._console_transcript_sync_timer
            timer.pause()
            syncs = _Syncs(monkeypatch, console)
            try:
                console._stop_console_transcript_sync_timer()
                with _commit_window(console._console_chat_store):
                    for expected in (1, 2):
                        await until(
                            lambda: not console._console_sync_in_progress, timeout=30
                        )
                        await console._sync_native_console_chat_ui()
                        assert syncs.count == expected
            finally:
                gateway.validation_release.set()


@pytest.mark.asyncio
async def test_a_real_sends_commit_opens_the_window_on_the_store_the_screen_reads(
    monkeypatch,
):
    """The store a send's commit marks is the one the Console's hold reads.

    The tests above open the window by hand. Here a real send's ordinary
    commit (``commit_durable_turn_owned``, on its executor thread) opens it,
    and the screen's own store (``_console_chat_store``, which the hold reads)
    reports it while the commit runs, then closes it.
    """
    started_at = getattr(
        console_native_commit, "durable_commit_started_at", lambda _store: None
    )
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            store = console._console_chat_store
            assert store is not None, "the ready Console has no store"
            real = store.commit_durable_turn
            seen: list[float | None] = []

            def commit(acceptance):
                seen.append(started_at(console._console_chat_store))
                return real(acceptance)

            monkeypatch.setattr(store, "commit_durable_turn", commit)
            press(host, "enter", "\r")
            try:
                # Provider resolution comes before the turn's durable commit.
                await until(gateway.validation_started.is_set, timeout=30)
                gateway.validation_release.set()
                await until(lambda: gateway.stream_calls >= 1, timeout=30)
                assert len(seen) == 1, f"{len(seen)} durable commits for one send"
                assert seen[0] is not None, (
                    "a real send's commit opened no window on the screen's store"
                )
                assert started_at(store) is None, "the commit window stayed open"
            finally:
                gateway.validation_release.set()


def test_each_commit_holds_once_until_the_next_tick_and_never_two_ticks():
    """The hold's bookkeeping, by itself: per commit, bounded, and reset."""
    from types import SimpleNamespace

    store = object()
    screen = SimpleNamespace(
        _console_transcript_sync_timer=object(), _console_chat_store=store
    )
    # (Nothing is ever held on the base.)
    held = getattr(
        config_sync, "_held_for_durable_commit", lambda _screen, *, poll: False
    )
    assert held(screen, poll=True) is False  # No commit in flight.
    with _commit_window(store):
        assert held(screen, poll=False) is False  # The commit's first sync.
        assert held(screen, poll=False) is True  # Held until the next tick...
        assert held(screen, poll=False) is True
        assert held(screen, poll=True) is True  # ...which skips and ends it.
        assert held(screen, poll=True) is False  # Never two in a row.
        assert held(screen, poll=False) is False  # Each commit holds once.
        with _commit_window(store):  # A second commit while the first runs:
            assert held(screen, poll=False) is False  # the same window.
    assert held(screen, poll=False) is False
    with _commit_window(store):  # A later commit gets its own first sync.
        assert held(screen, poll=True) is False
        assert held(screen, poll=True) is True
        assert held(screen, poll=True) is False
