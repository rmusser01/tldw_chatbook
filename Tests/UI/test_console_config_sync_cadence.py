"""The Console's 0.2 s poll pays the checked config operation only when it must.

TASK-33620.15.1 (owner decision 2026-10-09, "Pay the storage-admission check
less often"). Measured live on the parent build 1c6172e611: each poll tick ran
the Console's config-dependent projections (rail state, settings summary,
control bar) inside one checked config operation, median 6-10 ms, p90 31-40,
max 44-80 ms per 0.2 s tick, 53-60% of it the operation's entry and exit
(storage admission, the pause probe). Those stretches landed in a run's
busiest seconds, while input waited.

The poll now runs them only when an input changed (the installed config or
settings, the config selector, a local storage pause, the active chat, a run
state) or an explicit sync asked, and during a run at least once a second as
a backstop. When it runs, it runs as before: same locks, same refusal and
deferral during a local storage pause.

These tests drive the real ChatScreen poll callback (the timer's own
coroutine) on the mounted send harness and count, per tick, the calls into
``run_console_config_sync`` made by the tick itself. The clock the backstop
reads is replaced, so no test depends on real elapsed time.
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_console_send_acknowledgement import (
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)

pytestmark = pytest.mark.bootstrap_profile


class _Clock:
    """The backstop's clock, advanced by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Projections:
    """Counts the projections the test's own ticks attempt and render.

    A tick awaited by the test runs in the test's task; syncs the app starts
    on its own (workers, the send path) run in other tasks and are not
    counted, so a count is exactly what the ticks under test did.
    """

    def __init__(self, monkeypatch, console) -> None:
        from tldw_chatbook.UI.Console_Modules import config_sync

        self.task = asyncio.current_task()
        self.calls = 0
        self.deferred = 0
        self.renders = 0
        self.renders_anywhere = 0
        real_sync = config_sync.run_console_config_sync
        real_render = console._sync_console_control_bar_under_config

        def counting_sync(sync, **kwargs):
            if asyncio.current_task() is not self.task:
                return real_sync(sync, **kwargs)
            self.calls += 1
            result = real_sync(sync, **kwargs)
            if result is False:
                self.deferred += 1
            return result

        def counting_render(*args, **kwargs):
            self.renders_anywhere += 1
            if asyncio.current_task() is self.task:
                self.renders += 1
            return real_render(*args, **kwargs)

        monkeypatch.setattr(config_sync, "run_console_config_sync", counting_sync)
        console._sync_console_control_bar_under_config = counting_render


async def _settle(console) -> None:
    """Let any in-flight (or re-armed) sync finish before the next tick.

    Ends on the check itself, with no await after it, so the caller's next
    sync starts before any other task can.
    """
    for _ in range(3):
        await asyncio.sleep(0.01)
        await until(lambda: not console._console_sync_in_progress, timeout=30)


async def _explicit_sync(console, seen) -> None:
    """Run one explicit sync in the test's task (retrying one that deferred).

    A sync the app re-armed can still start between ``_settle`` and the
    call when it was queued behind a deferred control-bar replay; the
    explicit call then hands its work to that sync, so try again.
    """
    for _ in range(20):
        await _settle(console)
        before = seen.calls
        await console._sync_native_console_chat_ui()
        if seen.calls > before:
            return
    raise AssertionError("no explicit sync ran in the test's task")


class _Poll:
    """The 0.2 s poll's own callback, called by hand while its timer is held.

    No tick runs that the test did not ask for: the timer is stopped and
    restarting it is suppressed until ``release`` hands it back (the
    transcript only paints a reply through it).
    """

    def __init__(self, console) -> None:
        self.console = console
        console._start_console_transcript_sync_timer()
        timer = console._console_transcript_sync_timer
        assert timer is not None
        self._poll = timer._callback
        console._stop_console_transcript_sync_timer()
        console._start_console_transcript_sync_timer = lambda: None

    async def __call__(self) -> None:
        await _settle(self.console)
        await self._poll()

    async def right_after(self, change):
        """Settle, make ``change``, and tick before any other task can sync.

        The tick's sync marks itself in progress before its first await, so
        a sync the app starts meanwhile defers to it rather than projecting
        the change first.

        Returns:
            What ``change`` returned.
        """
        await _settle(self.console)
        result = change()
        await self._poll()
        return result

    def release(self) -> None:
        del self.console._start_console_transcript_sync_timer
        self.console._start_console_transcript_sync_timer()


async def _run_over(console) -> None:
    """Wait until no chat has a run going (the backstop's gate is closed)."""
    from tldw_chatbook.Chat.console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES

    controller = console._console_chat_controller

    def over() -> bool:
        return not console._console_run_active() and not any(
            state.status in FEEDBACK_ACTIVE_RUN_STATUSES
            for state in controller.run_states().values()
        )

    await until(over, timeout=60)


async def _held_run(host, pilot, gateway):
    """A ready Console whose first send is held at provider validation."""
    console, _composer = await ready_console(host, pilot, gateway)
    press(host, "enter", "\r")
    await until(gateway.validation_started.is_set, timeout=30)
    assert console._console_run_active()
    return console


@pytest.mark.asyncio
async def test_unchanged_poll_ticks_skip_the_operation_but_once_a_second_in_a_run(
    monkeypatch,
):
    """(a) unchanged ticks skip it; (c) the backstop runs during a run only."""
    from tldw_chatbook.UI.Console_Modules import config_sync

    clock = _Clock()
    # raising=False: the parent build has no backstop clock (and fails below).
    monkeypatch.setattr(config_sync, "_clock", clock, raising=False)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            tick = _Poll(console)
            seen = _Projections(monkeypatch, console)
            try:
                await _explicit_sync(console, seen)
                assert (seen.calls, seen.renders) == (1, 1)

                # (a) Within a run's second, ticks with no input change skip.
                for _ in range(5):
                    await tick()
                assert seen.calls == 1, (
                    f"{seen.calls - 1} checked config operations in 5 unchanged ticks"
                )

                # (c) The backstop: once a second during a run, not before.
                clock.now += 0.6
                await tick()
                assert seen.calls == 1, "a backstop projection before a second"
                clock.now += 0.4
                await tick()
                await tick()
                assert seen.calls == 2, "no backstop projection a second later"
                clock.now += 1.0
                await tick()
                assert (seen.calls, seen.renders) == (3, 3)
            finally:
                tick.release()
                gateway.validation_release.set()
            await _run_over(console)

            # (a) Outside a run there is no backstop: only a change projects.
            tick = _Poll(console)
            await _explicit_sync(console, seen)
            idle = seen.calls
            clock.now += 10.0
            for _ in range(3):
                await tick()
            tick.release()
            assert seen.calls == idle, (
                f"{seen.calls - idle} checked config operations in 3 idle ticks"
            )


@pytest.mark.asyncio
async def test_each_input_change_is_one_projection_on_the_next_poll_tick(monkeypatch):
    """(b) A reinstalled config or settings, or an explicit sync: one each.

    The clock stands still, so no backstop runs; the run is held at provider
    validation, so nothing else changes between the ticks.
    """
    from tldw_chatbook import config
    from tldw_chatbook.UI.Console_Modules import config_sync

    monkeypatch.setattr(config_sync, "_clock", _Clock(), raising=False)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            tick = _Poll(console)
            seen = _Projections(monkeypatch, console)
            try:
                await _explicit_sync(console, seen)
                assert seen.calls == 1

                async def projections_in_two_ticks_after(change) -> int:
                    before = seen.calls
                    await tick.right_after(change)
                    await tick()
                    return seen.calls - before

                for name, change in (
                    (
                        "a reinstalled config",
                        lambda: config.load_cli_config_and_ensure_existence(
                            force_reload=True
                        ),
                    ),
                    (
                        "reinstalled settings",
                        lambda: config.load_settings(force_reload=True),
                    ),
                ):
                    count = await projections_in_two_ticks_after(change)
                    assert count == 1, (
                        f"{count} projections in the two ticks after {name}; want 1"
                    )
                before = seen.calls
                await _explicit_sync(console, seen)
                await tick()
                assert seen.calls - before == 1, (
                    f"{seen.calls - before} projections for an explicit sync and "
                    "the tick after it; want 1"
                )
                assert seen.renders == seen.calls
            finally:
                tick.release()
                gateway.validation_release.set()
            await _run_over(console)


@pytest.mark.asyncio
async def test_a_local_storage_pause_still_defers_the_tick_projection(monkeypatch):
    """(d) A pause makes the next tick refuse and defer exactly as before.

    The tick enters the checked operation, renders nothing, marks the whole
    sync for replay and arms one delayed retry; once the pause ends, that
    retry renders. Skipping the operation because "nothing changed" would
    instead leave no retry behind.
    """
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.UI.Console_Modules import config_sync

    monkeypatch.setattr(config_sync, "_clock", _Clock(), raising=False)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            tick = _Poll(console)
            seen = _Projections(monkeypatch, console)
            await _explicit_sync(console, seen)
            for _ in range(3):
                await tick()
            assert (seen.calls, seen.renders) == (1, 1), (
                f"{seen.calls - 1} checked config operations in 3 unchanged ticks"
            )
            assert not console._console_control_bar_replay_whole_sync
            pause = await tick.right_after(storage_admission._begin_local_pause)
            try:
                assert (seen.calls, seen.deferred, seen.renders) == (2, 1, 1)
                assert console._console_control_bar_replay_whole_sync
                assert console._console_control_bar_sync_scheduled
                await tick()  # The replay owns the refresh; the tick waits.
                assert seen.calls == 2
            finally:
                pause.resume()
                tick.release()
            rendered = seen.renders_anywhere
            await until(
                lambda: seen.renders_anywhere > rendered
                and not console._console_control_bar_replay_whole_sync,
                timeout=30,
            )
