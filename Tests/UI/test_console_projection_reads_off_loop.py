"""The run's config projection pays no checked config operation on the UI loop.

TASK-33620.15.4 (measured live 2026-10-23 on base 5b5920b8e2, Anthropic haiku
through the review proxy, 160x45): a send window's projections cost 54-57 ms
of UI-loop time. An idle Console projects nothing, so when a send started its
checked config display (refreshed at most once a second, PR #3023) had always
expired. The poll's first tick then ran the native checked config operation
on the UI loop (storage admission and the config file posture), and the same
tick's later presentation read found the refresh pending and deferred to a
whole-sync replay 0.2 s later, which projected everything again and whose
own follow-up sync projected a third time. Each refreshed context snapshot
also started a whole sync of its own.

Now the run poll's start re-reads the display, a projection that finds it
past half its life re-reads it ahead while the current one renders, and a
snapshot published while the poll runs is a cadence input the next tick
projects. These tests drive the real ChatScreen on the mounted send harness
with the checked display's clock replaced, and count the checked config
operation's entries on the UI loop's thread inside the projection itself,
observed with a profiler (nothing is replaced).
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from Tests.UI.test_console_config_sync_cadence import _install_clock
from Tests.UI.test_console_send_acknowledgement import (
    build,
    eager_tasks,
    ready_console,
    until,
)
from tldw_chatbook.Backup_Recovery.config_participants import operation
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = pytest.mark.bootstrap_profile


class _NativeOperations:
    """Checked config operations the UI loop's thread enters (observed only)."""

    def __init__(self) -> None:
        self._code = operation.__wrapped__.__code__
        self._frames: list = []
        self.main = 0
        self.other = 0

    def __enter__(self):
        self._previous = sys.getprofile(), threading.getprofile()
        main = threading.main_thread().ident

        def observe(frame, event, _argument):
            # A generator's frame reports "call" again at every resume.
            if event != "call" or frame.f_code is not self._code:
                return
            if any(frame is seen for seen in self._frames):
                return
            self._frames.append(frame)
            if threading.get_ident() == main:
                self.main += 1
            else:
                self.other += 1

        threading.setprofile_all_threads(observe)
        return self

    def __exit__(self, *_exc):
        threading.setprofile_all_threads(self._previous[1])
        sys.setprofile(self._previous[0])
        self._frames.clear()


def _project(console) -> tuple[bool, _NativeOperations]:
    """Run the tick's projection once, counting its native operations."""
    with _NativeOperations() as ops:
        result = console._sync_console_rail_and_controls()
    return result, ops


async def _fresh_display(console):
    """The Console's checked display, published and not being re-read."""
    projection = spend.ConsoleReadinessConfigProjection.for_screen(console)

    def published() -> bool:
        projection.run(lambda: None)
        return (
            projection.value is not None
            and not projection.pending
            and projection._display_proof is not None
        )

    await until(published, timeout=30)
    return projection


@pytest.mark.asyncio
async def test_run_start_rereads_an_expired_display_so_the_first_projection_pays_no_operation(
    monkeypatch,
):
    clock = _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            projection = await _fresh_display(console)
            console._stop_console_transcript_sync_timer()
            clock.now += 30.0  # The Console idled; its display expired.

            # Run work starts: the poll's timer is created. (Stopped again at
            # once: this test ticks nothing; the warm-up is the start's own.)
            console._start_console_transcript_sync_timer()
            console._stop_console_transcript_sync_timer()
            assert projection.pending, (
                "run start left the expired display for the first tick to find"
            )
            await until(lambda: not projection.pending, timeout=30)
            assert projection.at == clock.now

            result, ops = _project(console)
            assert result is True
            assert ops.main == 0, (
                f"the first projection after run start entered {ops.main} checked "
                "config operation(s) on the UI loop"
            )


@pytest.mark.asyncio
async def test_a_projection_rereads_an_aging_display_ahead_of_its_expiry(
    monkeypatch,
):
    clock = _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            projection = await _fresh_display(console)
            # Run work: the poll's timer exists (held: this test ticks nothing).
            console._start_console_transcript_sync_timer()
            console._console_transcript_sync_timer.pause()
            await until(lambda: not projection.pending, timeout=30)

            # Past half its life: the projection renders the current display
            # and starts the next read.
            clock.now += 0.6
            result, ops = _project(console)
            assert result is True and ops.main == 0
            assert projection.pending, "no read started ahead of the expiry"
            await until(lambda: not projection.pending, timeout=30)

            # 1.2 s after the first read, 0.6 s after the second: fresh.
            clock.now += 0.6
            result, ops = _project(console)
            assert result is True
            assert ops.main == 0, (
                f"an expired display cost {ops.main} checked config operation(s) "
                "on the UI loop"
            )
            console._stop_console_transcript_sync_timer()
            await until(lambda: not projection.pending, timeout=30)

            # With no run poll the display is re-read only once it expires.
            clock.now += 0.6
            assert _project(console)[0] is True
            assert not projection.pending, "an idle projection re-read ahead"


@pytest.mark.asyncio
async def test_a_context_snapshot_published_during_a_run_projects_on_the_next_tick(
    monkeypatch,
):
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await _fresh_display(console)
            snapshot = spend.ConsoleContextReadSnapshot.for_screen(
                console, max_age=60.0
            )
            controller = console._ensure_console_chat_controller()
            session_id = console._ensure_console_chat_store().active_session_id
            await until(
                lambda: snapshot.inputs(controller, session_id) is not None
                and snapshot.value is not None
                and snapshot.pending_key is None,
                timeout=30,
            )
            # Run work is live: the poll's timer exists (held, so this test
            # decides when it ticks).
            console._start_console_transcript_sync_timer()
            timer = console._console_transcript_sync_timer
            timer.pause()
            groups: list[str | None] = []
            real_run_worker = console.run_worker

            def recording_run_worker(work, *args, **kwargs):
                groups.append(kwargs.get("group"))
                return real_run_worker(work, *args, **kwargs)

            monkeypatch.setattr(console, "run_worker", recording_run_worker)
            # The snapshot captured the screen's scheduler when it was built.
            monkeypatch.setattr(snapshot, "schedule", recording_run_worker)
            cadence = config_sync._cadence(console)
            published = getattr(cadence, "published", 0)  # (None on the base.)
            seen: list[int] = []  # The publication count each projection saw.
            real_project = console._sync_console_rail_and_controls

            def counting_project():
                seen.append(getattr(cadence, "published", 0))
                return real_project()

            monkeypatch.setattr(
                console, "_sync_console_rail_and_controls", counting_project
            )
            _objects, before = config_sync._projection_inputs(console)

            snapshot.key = snapshot.value = None  # Its owner moved: re-read.
            snapshot.inputs(controller, session_id)
            await until(lambda: snapshot.pending_key is None, timeout=30)
            await until(lambda: snapshot.value is not None, timeout=30)

            assert "console-context-publication" not in groups, (
                "the refreshed snapshot started a whole sync during a run"
            )
            assert getattr(cadence, "published", 0) == published + 1
            _objects, after = config_sync._projection_inputs(console)
            assert after != before, "the publication is not a cadence input"
            assert console._console_transcript_poll_asked

            # Freshness: the next tick (or a sync before it) projects it.
            await until(lambda: not console._console_sync_in_progress, timeout=30)
            await timer._callback()
            assert published + 1 in seen, (
                "no projection rendered the published snapshot by the next tick"
            )
            console._stop_console_transcript_sync_timer()


@pytest.mark.asyncio
async def test_explicit_syncs_of_a_run_start_share_the_next_ticks_projection(
    monkeypatch,
):
    """AC#3 (coalescing) and AC#5 (freshness: one poll interval at most).

    The run is held at provider validation (its run start) and its poll's
    timer is held too, so this test decides when it ticks; the backstop's
    clock stands still unless the test moves it. Projections are counted in
    the test's own task only.
    """
    from Tests.UI.test_console_config_sync_cadence import _held_run

    clock = _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            timer = console._console_transcript_sync_timer
            assert timer is not None, "the run start started no poll"
            timer.pause()
            owner = asyncio.current_task()
            calls: list[bool] = []
            real_project = console._sync_console_rail_and_controls

            def counting_project():
                result = real_project()
                if asyncio.current_task() is owner:
                    calls.append(result)
                return result

            monkeypatch.setattr(
                console, "_sync_console_rail_and_controls", counting_project
            )

            async def settled_sync() -> None:
                await until(lambda: not console._console_sync_in_progress, timeout=30)
                await console._sync_native_console_chat_ui()

            async def tick() -> None:
                await until(lambda: not console._console_sync_in_progress, timeout=30)
                await timer._callback()

            try:
                await tick()  # The run start's own tick projects.
                await until(lambda: True in calls, timeout=30)
                cadence = config_sync._cadence(console)
                before = len(calls)

                # Three syncs requested within the poll interval: none projects.
                for _ in range(3):
                    await settled_sync()
                assert len(calls) == before, (
                    f"{len(calls) - before} explicit sync(s) projected inside the "
                    "run start's poll interval"
                )
                assert cadence.requested > cadence.served

                # The next tick projects them, once.
                await tick()
                assert len(calls) == before + 1
                assert cadence.requested == cadence.served

                # Freshness: a request with no projection for a poll interval
                # projects at once, and so does one left waiting that long.
                clock.now += config_sync.RUN_COALESCE_SECONDS
                await settled_sync()
                assert len(calls) == before + 2
                await settled_sync()  # Within the interval again: it waits...
                assert len(calls) == before + 2
                clock.now += config_sync.RUN_COALESCE_SECONDS
                await settled_sync()  # ...a poll interval at most.
                assert len(calls) == before + 3
                assert cadence.requested == cadence.served
            finally:
                gateway.validation_release.set()
                timer.resume()
