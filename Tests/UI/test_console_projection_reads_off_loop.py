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

from Tests.private_profile import private_profile_test
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
            assert cadence.owed, "a poll stopping now would not render it"

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


def test_a_stopping_poll_hands_a_request_it_was_left_to_the_running_sync():
    """AC#5: a request coalesced into a poll that then stops is still rendered.

    The poll stops on a tick that skipped while another sync is running; that
    sync left its projection to this poll. With no poll left, the running
    sync's re-armed follow-up (requested here) projects at once.
    """
    from types import SimpleNamespace

    screen = SimpleNamespace(
        _console_sync_maintenance_paused=False,
        _console_sync_in_progress=True,
        _console_control_bar_replay_whole_sync=False,
        _console_sync_requested=False,
    )
    cadence = config_sync._cadence(screen)
    cadence.owed = True
    cadence.waiting_since = 1000.0  # A request left to the poll.
    config_sync.settle_stopped_poll(screen, lambda _screen: False)
    assert screen._console_sync_requested, (
        "the coalesced request was dropped when the poll stopped"
    )


def test_the_core_state_push_never_waits_for_or_rebuilds_config(monkeypatch):
    """TASK-36511: the full sync's core-state push only reads warm config it holds.

    ``sync_console_core_state`` probed ``config_read_would_wait()`` and then
    pushed; a worker that took the config locks in between made the push's
    ``load_settings()`` wait out its whole read on the UI loop (the compact
    gate's 238 ms probe stall, main thread in ``config_participants.operation``),
    and a cold cache made the push rebuild config on the UI loop. The push now
    holds both locks, so no read starts under it, and replays the whole sync
    when another thread holds a lock or a live read would be cold.
    """
    from types import SimpleNamespace

    from tldw_chatbook import config

    locks = (config._settings_rebuild_lock(), config._config_file_lock())
    warm = {"settings": {}, "cli": {}}
    monkeypatch.setattr(config, "_settings_cache_hit", lambda _path: warm["settings"])
    monkeypatch.setattr(config, "_warm_config_cache_hit", lambda: warm["cli"])
    taken_mid_push = []

    class Screen:
        _console_control_bar_replay_whole_sync = False
        app_instance = app = SimpleNamespace(app_config={"disk-loaded": True})

        def __init__(self):
            self.pushes, self.replays = 0, []

        def _console_config_snapshot_is_disk_loaded(self, value):
            return bool(value)

        def _sync_console_chat_core_state(self):
            self.pushes += 1

            def worker_read():  # A worker's checked read starts mid-push.
                got = [lock.acquire(blocking=False) for lock in locks]
                for lock, ok in zip(locks, got):
                    if ok:
                        lock.release()
                taken_mid_push.append(any(got))

            reader = threading.Thread(target=worker_read)
            reader.start()
            reader.join(5)

        def _request_console_control_bar_sync(self, *, delayed=False):
            self.replays.append(delayed)

    screen = Screen()
    config_sync.sync_console_core_state(screen)
    assert screen.pushes == 1
    assert taken_mid_push == [False], "a worker's config read started under the push"

    held, release = threading.Event(), threading.Event()

    def hold_file_lock():  # The race: the read took a lock after the probe.
        with locks[1]:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold_file_lock)
    holder.start()
    assert held.wait(5)
    try:
        config_sync.sync_console_core_state(screen)
    finally:
        release.set()
        holder.join(5)
    assert screen.pushes == 1, "the push ran while another thread held a config lock"
    assert screen.replays == [True] and screen._console_control_bar_replay_whole_sync

    warm["settings"] = None  # An external edit or a save left the cache cold.
    config_sync.sync_console_core_state(screen)
    assert screen.pushes == 1, "the push would rebuild config on the UI loop"
    assert screen.replays == [True, True]

    # An unparseable config.toml stays cold for good: one deferral per episode.
    config_sync.sync_console_core_state(screen)  # The replay pushes live.
    assert screen.pushes == 2, "a still-cold replay deferred again (replay loop)"
    config_sync.sync_console_core_state(screen)  # Same episode: no new replay.
    assert screen.pushes == 3 and screen.replays == [True, True]
    warm["settings"] = {}
    config_sync.sync_console_core_state(screen)  # A warm push ends the episode,
    warm["settings"] = None
    config_sync.sync_console_core_state(screen)  # so a later cold one defers once.
    assert screen.pushes == 4 and screen.replays == [True, True, True]


def test_a_mount_refresh_defers_a_cold_live_read_and_warms_it_off_the_loop(monkeypatch):
    """TASK-36511: ``retry_after_config_rebuild`` defers while a live read is cold.

    The identity refresh's core-state push rebuilt config on the UI loop
    (79.5 ms CPU, 139.6 ms wall in the compact gate). It now has the readiness
    projection read config off the loop and renders on a later retry.
    """
    from types import SimpleNamespace

    from tldw_chatbook import config

    warm = {"settings": None, "cli": {}}
    monkeypatch.setattr(config, "_settings_cache_hit", lambda _path: warm["settings"])
    monkeypatch.setattr(config, "_warm_config_cache_hit", lambda: warm["cli"])
    monkeypatch.setattr(config, "config_read_would_wait", lambda: False)
    kicked, timers, calls = [], [], []
    monkeypatch.setattr(
        spend.ConsoleReadinessConfigProjection,
        "for_screen",
        classmethod(
            lambda cls, screen: SimpleNamespace(run=lambda body: kicked.append(screen))
        ),
    )

    class Screen:
        app_instance = app = SimpleNamespace(app_config={"disk-loaded": True})

        def _console_config_snapshot_is_disk_loaded(self, value):
            return bool(value)

        def set_timer(self, delay, callback):
            timers.append(callback)

    @spend.retry_after_config_rebuild
    def refresh(screen):
        calls.append(True)

    screen = Screen()
    refresh(screen)
    assert calls == [], "the refresh rebuilt cold config on the UI loop"
    assert kicked == [screen] and len(timers) == 1
    warm["settings"] = {}  # The projection's off-loop read warmed it.
    timers.pop()()
    assert calls == [True] and not timers

    # An unparseable config.toml stays cold for good: one deferral per episode.
    warm["settings"] = None
    refresh(screen)
    assert calls == [True] and len(timers) == 1 and kicked == [screen] * 2
    timers.pop()()  # The retry runs although config is still cold,
    assert calls == [True] * 2, "a still-cold retry deferred again (retry chain)"
    refresh(screen)  # and so does any later call in the same cold episode.
    assert calls == [True] * 3 and not timers and kicked == [screen] * 2
    warm["settings"] = {}  # Any owner's warm read ends the episode for all.
    assert not spend.defer_cold_live_read(screen, "another owner")
    assert not screen._console_cold_read_deferrals, "a stale cold-episode entry"


class _UiLoopStarts:
    """Count starts of named code objects on the UI loop's thread (observed only).

    A coroutine's frame reports "call" again at every resume, so a start is a
    frame not seen before. Each name maps to ``(code, caller code or None)``.
    """

    def __init__(self, **codes):
        self._codes = codes
        self._frames: list = []
        self.counts = dict.fromkeys(codes, 0)

    def __enter__(self):
        self._previous = sys.getprofile()

        def observe(frame, event, _argument):
            if event != "call":
                return
            for name, (code, caller) in self._codes.items():
                if frame.f_code is not code:
                    continue
                if caller is not None and frame.f_back.f_code is not caller:
                    continue
                if not any(frame is seen for seen in self._frames):
                    self._frames.append(frame)
                    self.counts[name] += 1

        sys.setprofile(observe)  # This thread only: the UI loop's.
        return self

    def __exit__(self, *_exc):
        sys.setprofile(self._previous)
        self._frames.clear()

    def take(self) -> dict:
        counts, self.counts = self.counts, dict.fromkeys(self.counts, 0)
        return counts


@pytest.mark.asyncio
@private_profile_test
async def test_an_unparseable_config_leaves_an_idle_console_idle(request):
    """TASK-36511 review: a config that stays cold never replays the Console in a loop.

    An unparseable config.toml installs neither config cache, so a live read
    stays cold for good. Deferring every cold read replayed the idle Console
    5 full syncs a second (26 cold core-state reads a second) and re-armed the
    identity refresh every 0.2 s without running it. Each owner now defers
    once per cold episode: the core-state push and the identity refresh run,
    and the idle Console goes quiet again. Under load the first runs can take
    seconds (a retry also waits out the off-loop read's locks), so the test
    polls for them and then judges the steady state, which is what tells a
    bounded deferral from a replay loop.
    """
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            assert console._console_config_snapshot_is_disk_loaded(
                console.app_instance.app_config
            )
            path = config._get_effective_config_path()  # This child's own profile.
            original = path.read_bytes()
            await pilot.pause(1.0)
            starts = _UiLoopStarts(
                full_sync=(ChatScreen._sync_native_console_chat_ui.__code__, None),
                core_push=(
                    ChatScreen._sync_console_chat_core_state.__code__,
                    config_sync.sync_console_core_state.__code__,
                ),
                identity=(
                    ChatScreen._sync_console_identity_surfaces.__wrapped__.__code__,
                    None,
                ),
            )
            with starts:
                await pilot.pause(2.0)
                idle = starts.take()
                path.write_bytes(original + b"\n[unterminated\n")
                try:
                    await asyncio.to_thread(config.load_settings, True)
                    assert config._settings_cache_hit(path) is None
                    starts.take()
                    console.run_worker(
                        console._sync_native_console_chat_ui(),
                        exclusive=True,
                        group="console-sync",
                    )
                    console._sync_console_identity_surfaces()
                    first = dict.fromkeys(idle, 0)
                    deadline = asyncio.get_running_loop().time() + 10.0
                    while not (first["core_push"] and first["identity"]):
                        if asyncio.get_running_loop().time() > deadline:
                            break  # The steady state below names the loop.
                        await pilot.pause(0.1)
                        for name, count in starts.take().items():
                            first[name] += count
                    await pilot.pause(0.5)  # The single replay's own pass.
                    starts.take()
                    await pilot.pause(2.0)
                    later = starts.take()
                finally:
                    path.write_bytes(original)
                    await asyncio.to_thread(config.load_settings, True)
    assert later["full_sync"] <= idle["full_sync"] + 1, (
        f"the idle Console kept replaying: idle {idle}, cold config {later}"
    )
    assert first["core_push"] >= 1, f"the full sync never pushed: {first}"
    assert first["identity"] >= 1, f"the identity refresh never ran: {first}"
