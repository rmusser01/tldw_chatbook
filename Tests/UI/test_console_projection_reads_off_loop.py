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
from Tests.UI.test_console_config_sync_cadence import _install_clock, _replay_pending
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
async def test_a_warm_up_rereads_a_display_near_expiry_so_its_render_accepts_it(
    monkeypatch,
):
    """TASK-38967: a FULL sync's warm-up never hands back a proof about to expire.

    ``warm()`` returned True for any display younger than its 1 s life, while
    ``run()`` re-reads one only once it is that old, so a 0.99 s proof came
    back as warm. The render that followed refused it as expired, and the
    whole sync replayed 0.2 s later (11 of 191 warm-ups in three mounted files
    at load 9-11). The warm-up now re-reads a display past half its life, the
    share the run poll's projections already refresh ahead at.
    """
    clock = _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            projection = await _fresh_display(console)
            console._stop_console_transcript_sync_timer()  # No run poll.
            await until(lambda: not projection.pending, timeout=30)

            clock.now += projection.max_age - 0.01
            assert await projection.warm()
            clock.now += 0.05  # The FULL sync's own render takes time.
            result, _ops = _project(console)
            assert result is True, (
                "the render refused the display its warm-up had just handed back"
            )

            # While reads fail, each failed re-read arms a retry sync, so the
            # warm-up keeps the base rule: re-read only an expired display.
            # Reading ahead there replayed an unparseable config's idle
            # Console twice in 2 s (test_an_unparseable_config_...).
            await until(lambda: not projection.pending, timeout=30)
            projection._retry_delay = spend.RETRY_FIRST_DELAY  # A read failed.
            read_at = projection.at
            clock.now += projection.max_age * 0.6
            assert await projection.warm()
            assert projection.at == read_at and not projection.pending, (
                "the warm-up read ahead while the display's reads were failing"
            )


@pytest.mark.asyncio
async def test_a_full_sync_slower_than_the_display_life_does_not_replay_itself(
    monkeypatch,
):
    """A FULL sync's own awaits must not expire the display its rail step reads.

    The FULL warms the checked display before its projection, then awaits the
    transcript and the archive view before its rail-visibility step. Under load
    those awaits outlived the display's 1 s life, the rail step refused the
    expired display, and the whole sync replayed. The replay did the same, so
    an idle Console kept syncing about once a second for as long as the load
    lasted. That was the flake in
    ``test_an_unparseable_config_leaves_an_idle_console_idle``: 4 refusals in
    4 s, all on proof age (1.20-1.33 s), at load about 70. The FULL now warms
    the display again before its rail step.
    """
    clock = _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            projection = await _fresh_display(console)
            console._stop_console_transcript_sync_timer()  # No run poll.
            await until(lambda: not projection.pending, timeout=30)
            real = console._sync_native_console_transcript

            stretch = {"share": 1.2}
            reads: list = []
            after_transcript: list = []

            async def slow_transcript(*args, **kwargs):
                # Load stretched the FULL's awaits.
                clock.now += projection.max_age * stretch["share"]
                result = await real(*args, **kwargs)
                after_transcript.append(len(reads))
                return result

            console._sync_native_console_transcript = slow_transcript
            console._console_control_bar_replay_whole_sync = False
            result = await console._sync_native_console_chat_ui()
            assert (
                result is True and not console._console_control_bar_replay_whole_sync
            ), "a FULL sync whose awaits outlived the display replayed itself"

            # A display still live at the rail step is not read again: with
            # config unparseable each read is a rebuild under the config locks.
            await until(lambda: not projection.pending, timeout=30)
            stretch["share"] = 0.6
            read_at = projection.at
            assert await console._sync_native_console_chat_ui() is True
            assert projection.at == read_at, "the rail step re-read a live display"

            # While reads fail, the rail step starts no read either: one failed
            # read was followed by a second in the same sync, and the retry
            # backoff doubled (r11c-rev3).
            from tldw_chatbook.Chat import console_preparation_reads

            async def unparseable(*args, **kwargs):
                raise RuntimeError("config.toml is unparseable")

            monkeypatch.setattr(
                console_preparation_reads, "run_preparation_read", unparseable
            )
            await until(lambda: not projection.pending, timeout=30)
            projection._retry_delay = spend.RETRY_FIRST_DELAY  # A read failed.
            refresh = projection._refresh

            def counting_refresh(key):
                reads.append(key)
                return refresh(key)

            projection._refresh = counting_refresh
            at_rail: list = []
            rail = config_sync.sync_console_rail_visibility

            def counting_rail(screen):
                at_rail.append(len(reads))
                return rail(screen)

            monkeypatch.setattr(
                config_sync, "sync_console_rail_visibility", counting_rail
            )
            stretch["share"] = 1.2
            await console._sync_native_console_chat_ui()
            assert at_rail and at_rail[-1] == after_transcript[-1], (
                "the rail step started a read while the display's reads were failing"
            )


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
                lambda: (
                    snapshot.inputs(controller, session_id) is not None
                    and snapshot.value is not None
                    and snapshot.pending_key is None
                ),
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
            owed_at_publication: list[bool] = []
            publish = snapshot.publish
            assert publish is not None

            def observe_publication():
                publish()
                owed_at_publication.append(cadence.owed)

            monkeypatch.setattr(snapshot, "publish", observe_publication)
            seen: list[int] = []  # The publication count each render saw.
            real_project = console._sync_console_rail_and_controls

            def counting_project():
                publication = cadence.published
                result = real_project()
                if result is True:
                    seen.append(publication)
                return result

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
            assert owed_at_publication == [True], (
                "the publication did not leave a final render owed"
            )

            # Freshness: the next eligible tick (or a sync before it) projects it.
            # A queued whole replay or presentation read still owns this render.
            await until(
                lambda: (
                    not console._console_sync_in_progress
                    and not _replay_pending(console)
                ),
                timeout=30,
            )
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

    The run is held at provider validation. Once its natural projection settles,
    its poll timer is held so this test decides when it ticks; the backstop's
    clock stands still unless the test moves it. Projections are counted in
    the test's own task only.
    """
    from Tests.UI.test_console_config_sync_cadence import _held_run, _replay_pending

    clock = _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            timer = console._console_transcript_sync_timer
            assert timer is not None, "the run start started no poll"
            cadence = config_sync._cadence(console)

            def sync_settled() -> bool:
                return not console._console_sync_in_progress and not _replay_pending(
                    console
                )

            def run_start_projected() -> bool:
                if not sync_settled() or cadence.at != clock.now:
                    return False
                _objects, values = config_sync._projection_inputs(console)
                return cadence.values == values

            # Let the natural poll serve startup work before measuring coalescing.
            await until(run_start_projected, timeout=30)
            assert console._console_run_active(), "the held run ended during setup"
            timer.pause()
            owner = asyncio.current_task()
            calls: list[bool] = []
            projected_requests: list[int] = []
            real_project = console._sync_console_rail_and_controls

            def counting_project():
                requested = config_sync._cadence(console).requested
                result = real_project()
                if asyncio.current_task() is owner:
                    calls.append(result)
                    projected_requests.append(requested)
                return result

            monkeypatch.setattr(
                console, "_sync_console_rail_and_controls", counting_project
            )

            async def settled_sync() -> None:
                await until(sync_settled, timeout=30)
                await console._sync_native_console_chat_ui()

            async def tick() -> None:
                await until(sync_settled, timeout=30)
                await timer._callback()

            try:
                # A poll may skip work already served by the natural startup poll.
                await tick()
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
                # FULL can receive newer requests after its synchronous projection.
                assert calls[-1] is True
                assert cadence.served >= projected_requests[-1]

                # Freshness: a request with no projection for a poll interval
                # projects at once, and so does one left waiting that long.
                clock.now += config_sync.RUN_COALESCE_SECONDS
                await settled_sync()
                assert len(calls) == before + 2
                await settled_sync()  # Within the interval again: it waits...
                assert len(calls) == before + 2
                clock.now += config_sync.RUN_COALESCE_SECONDS
                real_transcript = console._sync_native_console_transcript
                late_requests: list[tuple[int, int]] = []

                async def transcript_with_late_request(*args, **kwargs):
                    result = await real_transcript(*args, **kwargs)
                    if asyncio.current_task() is owner:
                        requested = cadence.requested
                        await asyncio.create_task(
                            console._sync_native_console_chat_ui()
                        )
                        late_requests.append((requested, cadence.requested))
                    return result

                # A real overlapping FULL arrives after this projection. It is
                # owed to a later tick, not retroactively to the completed one.
                with monkeypatch.context() as patch:
                    patch.setattr(
                        console,
                        "_sync_native_console_transcript",
                        transcript_with_late_request,
                    )
                    await settled_sync()  # ...a poll interval at most.
                assert late_requests and late_requests[-1][1] > late_requests[-1][0]
                assert len(calls) == before + 3
                # FULL can receive newer requests after its synchronous projection.
                assert calls[-1] is True
                assert cadence.served >= projected_requests[-1]
            finally:
                gateway.validation_release.set()
                timer.resume()


@pytest.mark.asyncio
async def test_run_poll_repaints_send_when_a_completed_turn_releases_its_queue(
    monkeypatch,
):
    """Queue release changes the composer after the run status is already final."""
    from textual.widgets import Button

    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmissionOrigin,
    )

    _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            await _fresh_display(console)
            controller = console._ensure_console_chat_controller()
            store = controller.store
            session_id = store.active_session_id
            composer.load_draft("")
            button = composer.query_one("#console-send-message", Button)
            queue = controller.prompt_queue_coordinator
            completed = ConsoleRunStatus.COMPLETED

            async def project_poll(screen):
                config_sync.begin_console_sync(screen)
                return config_sync.project_console_tick(screen)

            async def complete_turn():
                queue.turn_accepted(
                    session_id,
                    origin=ConsoleSubmissionOrigin.MANUAL,
                    context_epoch=store.conversation_context_epoch(session_id),
                )
                controller._set_run_state(
                    ConsoleRunState(completed), session_id=session_id
                )
                assert await config_sync.run_periodic_console_sync(
                    project_poll, console
                )
                assert button.label.plain == "Queue"
                assert button.disabled
                return ConsoleSubmitResult(True, True, terminal_status=completed)

            result = await controller.run_prompt_chain(
                session_id=session_id, initial_turn=complete_turn
            )
            assert result.terminal_status is completed
            assert not controller.activity_for(session_id).accepted_live_turn
            assert not controller.prompt_queue_registry.snapshot(session_id).total_count

            assert await config_sync.run_periodic_console_sync(project_poll, console)
            assert button.label.plain == "Send"
            assert button.disabled


@pytest.mark.asyncio
async def test_run_poll_settles_streaming_header_before_ready_composer(monkeypatch):
    """The composer and the final assistant row settle in the same order."""
    from textual.widgets import Static

    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunState,
        ConsoleRunStatus,
    )
    from tldw_chatbook.UI.Console_Modules import composer_run_controls
    from tldw_chatbook.UI.Console_Modules.transcript import (
        ConsoleChangeReviewProjection,
    )

    _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            await _fresh_display(console)
            console._stop_console_transcript_sync_timer()
            controller = console._ensure_console_chat_controller()
            store = controller.store
            session_id = store.active_session_id
            assistant = store.append_message(
                session_id, role=ConsoleMessageRole.ASSISTANT, content=""
            )
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING), session_id=session_id
            )
            store.append_stream_chunk(assistant.id, "Scripted planning complete.")
            await config_sync.run_periodic_console_sync(
                composer_run_controls.sync_poll_display_ui, console
            )
            selector = (
                f"#console-message-header-{assistant.id} "
                ".console-transcript-speaker-label"
            )

            def speaker_text():
                speaker = console.query_one_optional(selector, Static)
                return "" if speaker is None else speaker.render().plain

            await until(lambda: composer.run_active and "Streaming…" in speaker_text())

            ready_headers = []
            sync_action_state = composer.sync_action_state

            def observe_action_state(**kwargs):
                sync_action_state(**kwargs)
                if not kwargs["stop_available"] and kwargs["send_label"] == "Send":
                    ready_headers.append(speaker_text())

            monkeypatch.setattr(composer, "sync_action_state", observe_action_state)
            store.mark_message_complete(assistant.id)
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id=session_id
            )
            prepare = ConsoleChangeReviewProjection.prepare
            obsolete = asyncio.Event()

            async def defer_once(projection):
                if not obsolete.is_set():
                    obsolete.set()
                    return False
                return await prepare(projection)

            with monkeypatch.context() as deferred:
                deferred.setattr(ConsoleChangeReviewProjection, "prepare", defer_once)
                await config_sync.run_periodic_console_sync(
                    composer_run_controls.sync_poll_display_ui, console
                )
                await asyncio.wait_for(obsolete.wait(), timeout=15)
                assert not ready_headers
                assert composer.run_active
                assert "Streaming…" in speaker_text()

            # The coalesced replay reads the now-current projection.
            await until(lambda: bool(ready_headers))
            assert ready_headers[0] == "Assistant"


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
                for lock, ok in zip(locks, got, strict=False):
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


class _CorePushScreen:
    """The parts of a Console screen the core-state push touches (TASK-39145)."""

    _console_control_bar_replay_whole_sync = False

    def __init__(self):
        from types import SimpleNamespace

        self.app_instance = self.app = SimpleNamespace(app_config={"disk-loaded": True})
        self.pushes, self.replays = 0, []

    def _console_config_snapshot_is_disk_loaded(self, value):
        return bool(value)

    def _sync_console_chat_core_state(self):
        self.pushes += 1

    def _request_console_control_bar_sync(self, *, delayed=False):
        self.replays.append(delayed)

    def query(self, _widget_type):
        return []


def _hold_config_file_lock():
    """Hold the config file lock on another thread; returns its release."""
    from tldw_chatbook import config

    held, release = threading.Event(), threading.Event()

    def hold():
        with config._config_file_lock():
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(5)

    def stop():
        release.set()
        holder.join(5)

    return stop


def _refusal_clock(monkeypatch, config, failure):
    """Warm caches, a replaceable episode clock and a fixed parse failure."""
    monkeypatch.setattr(config, "_settings_cache_hit", lambda _path: {})
    monkeypatch.setattr(config, "_warm_config_cache_hit", dict)
    monkeypatch.setattr(config, "get_config_load_failure", failure)
    now = [1000.0]
    monkeypatch.setattr(config_sync, "_clock", lambda: now[0])
    return now


def test_parseable_contention_never_stops_the_whole_replay(monkeypatch):
    """TASK-39145: only an unreadable config.toml bounds the refused pushes.
    Ordinary contention keeps owing a whole replay for every refusal, past
    the episode length, and never shows the notice."""
    from tldw_chatbook import config

    now = _refusal_clock(monkeypatch, config, lambda: None)
    screen = _CorePushScreen()
    release = _hold_config_file_lock()
    try:
        for _ in range(40):  # 8 s of refusals 0.2 s apart.
            config_sync.sync_console_core_state(screen)
            now[0] += 0.2
    finally:
        release()
    assert screen.replays == [True] * 40, (
        f"parseable contention stopped replaying after {len(screen.replays)}"
    )
    assert getattr(screen, "_console_config_unreadable_installs", None) is None
    assert screen.pushes == 0


def test_a_won_push_starts_a_new_refusal_episode(monkeypatch):
    """TASK-39145: the episode counts consecutive refusals; a push in between
    ends it, so a refusal after it starts a new one instead of stopping."""
    from tldw_chatbook import config

    now = _refusal_clock(monkeypatch, config, object)
    screen = _CorePushScreen()
    release = _hold_config_file_lock()
    try:
        config_sync.sync_console_core_state(screen)  # Episode starts at 1000.
        now[0] += 2.9
        config_sync.sync_console_core_state(screen)
    finally:
        release()
    config_sync.sync_console_core_state(screen)  # Wins the probe and pushes.
    assert screen.pushes == 1
    now[0] += 0.6  # 3.5 s after the first refusal.
    release = _hold_config_file_lock()
    try:
        config_sync.sync_console_core_state(screen)
    finally:
        release()
    assert screen.replays == [True] * 3, (
        "a refusal after a won push was charged to the old episode"
    )
    assert getattr(screen, "_console_config_unreadable_installs", None) is None


def test_an_install_landing_at_the_stop_never_leaves_the_notice_up(monkeypatch):
    """TASK-39145 review: the stop reads the install count before the parse
    failure, so a good install landing between them clears the notice."""
    from tldw_chatbook import config

    monkeypatch.setattr(config, "_CONFIG_CACHE_INSTALLS", 41)

    def failure_then_good_install():
        config._CONFIG_CACHE_INSTALLS += 1  # A good load lands here.
        return object()

    now = _refusal_clock(monkeypatch, config, failure_then_good_install)
    screen = _CorePushScreen()
    release = _hold_config_file_lock()
    try:
        config_sync.sync_console_core_state(screen)
        now[0] += config_sync.CORE_REFUSAL_EPISODE_SECONDS
        config_sync.sync_console_core_state(screen)  # The stop.
    finally:
        release()
    assert len(screen.replays) == 1
    config_sync.sync_config_unreadable_notice(screen)
    assert screen._console_config_unreadable_installs is None, (
        "the notice outlived the good install that landed at the stop"
    )


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


def test_a_warm_load_no_owner_saw_still_ends_the_cold_episode(monkeypatch):
    """TASK-38949: the episode ends when config loads warm, whoever loads it.

    The deferral set was cleared only when some owner was called while config
    was warm. The credential poll returns before its locked section unless its
    pre-gate changed, so after a broken config was fixed an owner's entry could
    outlive the fix, and that owner's first read in the next cold episode ran
    live on the UI loop (147-222 ms for the poll, about 80 ms for the identity
    refresh). A successful cache install now ends the episode for every owner.
    """
    from types import SimpleNamespace

    from tldw_chatbook import config

    warm = {"settings": None}
    monkeypatch.setattr(config, "_settings_cache_hit", lambda _path: warm["settings"])
    monkeypatch.setattr(config, "_warm_config_cache_hit", dict)
    installs = getattr(config, "_CONFIG_CACHE_INSTALLS", 0)
    monkeypatch.setattr(config, "_CONFIG_CACHE_INSTALLS", installs, raising=False)

    class Screen:
        app_instance = SimpleNamespace(app_config={"disk-loaded": True})

        def _console_config_snapshot_is_disk_loaded(self, value):
            return bool(value)

    screen = Screen()
    owners = (
        "_poll_console_credential_readiness",
        "sync_console_core_state",
        "_sync_console_identity_surfaces",
    )
    assert [spend.defer_cold_live_read(screen, owner) for owner in owners] == [True] * 3
    # An unparseable config.toml installs nothing, so the episode holds.
    assert [spend.defer_cold_live_read(screen, owner) for owner in owners] == [
        False
    ] * 3

    # The file is fixed and another path (the readiness projection's off-loop
    # read, a Settings save) loads it warm; no owner runs meanwhile.
    warm["settings"] = {}
    monkeypatch.setattr(config, "_CONFIG_CACHE_INSTALLS", installs + 1, raising=False)
    warm["settings"] = None  # Cold again: a new episode.
    deferred = [spend.defer_cold_live_read(screen, owner) for owner in owners]
    assert deferred == [True] * 3, (
        f"an owner read cold config live on its first call of a new episode: {deferred}"
    )
    assert [spend.defer_cold_live_read(screen, owner) for owner in owners] == [
        False
    ] * 3


@pytest.mark.asyncio
@private_profile_test
async def test_only_a_successful_config_load_counts_as_a_cache_install(request):
    """TASK-38949: the install count moves on a good load and holds on a broken one.

    ``defer_cold_live_read`` ends a cold episode when this count moves. A
    broken config.toml must leave it where it is, or each owner would defer on
    every call while the file stays unparseable.
    """
    from tldw_chatbook import config

    path = config._get_effective_config_path()  # This child's own profile.
    await asyncio.to_thread(config.load_settings, True)
    good = path.read_bytes()
    before = config._CONFIG_CACHE_INSTALLS
    await asyncio.to_thread(config.load_settings, True)
    loaded = config._CONFIG_CACHE_INSTALLS
    try:
        path.write_bytes(good + b"\n[unterminated\n")
        await asyncio.to_thread(config.load_settings, True)
        broken = config._CONFIG_CACHE_INSTALLS
        await asyncio.to_thread(config.load_settings, True)
        still_broken = config._CONFIG_CACHE_INSTALLS
    finally:
        path.write_bytes(good)
        await asyncio.to_thread(config.load_settings, True)
    fixed = config._CONFIG_CACHE_INSTALLS
    assert loaded > before, f"a good load installed nothing: {before} -> {loaded}"
    assert broken == still_broken == loaded, (
        f"an unparseable file counted as an install: {loaded} -> {broken} -> {still_broken}"
    )
    assert fixed > still_broken, f"the fixed file installed nothing: {fixed}"


def test_the_credential_poll_defers_a_cold_read_once_and_then_stays_idle(monkeypatch):
    """TASK-36511.4: the 0.25 s credential poll warms cold config off the loop.

    A poll tick that found config cold rebuilt it on the UI loop through
    ``run_console_config_sync`` (147-192 ms in a mounted Console at load 8).
    Its first cold read per episode now defers and starts the readiness
    projection's off-loop read, leaving the pre-gate open so the next tick
    renders. A config that stays cold (an unparseable file) reads live on the
    next tick and commits the pre-gate, and idle ticks then do nothing. That
    held only when the snapshot changed: a pass whose snapshot was unchanged
    returned before committing the pre-gate, so a background status change
    the readiness snapshot does not carry rebuilt cold config every tick
    (10 UI-loop rebuilds in 2.5 s, mounted, dev and the deferral alone).
    """
    from types import SimpleNamespace

    from Tests.UI.test_console_credential_poll_gate import _screen
    from tldw_chatbook import config
    from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription

    screen, builds, _syncs, _state, _revision = _screen(monkeypatch)
    screen._sync_console_control_bar = lambda: True  # The snapshot commits.
    screen.app_instance = SimpleNamespace(app_config={"disk-loaded": True})
    screen._console_config_snapshot_is_disk_loaded = bool
    warm = {"settings": None}
    monkeypatch.setattr(config, "_settings_cache_hit", lambda _path: warm["settings"])
    monkeypatch.setattr(config, "_warm_config_cache_hit", dict)
    live, kicked = [], []

    def live_sync(sync, **_kwargs):  # The UI-loop rebuild route.
        live.append(True)
        sync()
        return True

    monkeypatch.setattr(config_sync, "run_console_config_sync", live_sync)
    monkeypatch.setattr(
        spend.ConsoleReadinessConfigProjection,
        "for_screen",
        classmethod(lambda cls, s: SimpleNamespace(run=lambda body: kicked.append(s))),
    )

    screen._poll_console_credential_readiness()
    assert (len(builds), live) == (0, []), "the poll rebuilt cold config on the UI loop"
    assert kicked == [screen], "the poll did not start the off-loop config read"
    warm["settings"] = {}  # The projection's off-loop read warmed config.
    screen._poll_console_credential_readiness()
    assert (len(builds), live) == (1, []), "the warm retry did not render"

    # An unparseable config.toml stays cold for good: one deferral per episode.
    warm["settings"] = None
    # A background status change reopens the pre-gate; the snapshot keeps it.
    monkeypatch.setattr(
        subscription,
        "subscription_credential_status",
        lambda *, background=False: "pending",
    )
    screen._poll_console_credential_readiness()
    assert (len(builds), live, kicked) == (1, [], [screen] * 2)
    screen._poll_console_credential_readiness()
    assert (len(builds), live) == (2, [True]), "a still-cold tick deferred again"
    for _ in range(3):
        screen._poll_console_credential_readiness()
    assert (len(builds), live, kicked) == (2, [True], [screen] * 2), (
        "idle ticks kept reading cold config"
    )


class _UiLoopStarts:
    """Count starts of named code objects on the UI loop's thread (observed only).

    PY_START excludes coroutine resumes. Each name maps to
    ``(code, caller code or None)``.
    """

    def __init__(self, **codes):
        self._codes = codes
        self.counts = dict.fromkeys(codes, 0)

    def __enter__(self):
        thread = threading.get_ident()
        monitoring = sys.monitoring
        self._tool_id = next(
            value for value in range(1, 6) if monitoring.get_tool(value) is None
        )
        monitoring.use_tool_id(self._tool_id, "console-ui-loop-starts")

        def observe(observed_code, _offset):
            if threading.get_ident() != thread:
                return
            frame = sys._getframe(1)
            for name, (code, caller) in self._codes.items():
                if observed_code is not code:
                    continue
                if caller is not None and frame.f_back.f_code is not caller:
                    continue
                self.counts[name] += 1

        # Profiling every UI call distorts the cold rebuild this test measures.
        event = monitoring.events.PY_START
        monitoring.register_callback(self._tool_id, event, observe)
        for code, _caller in self._codes.values():
            monitoring.set_local_events(self._tool_id, code, event)
        return self

    def __exit__(self, *_exc):
        monitoring = sys.monitoring
        for code, _caller in self._codes.values():
            monitoring.set_local_events(self._tool_id, code, 0)
        monitoring.register_callback(self._tool_id, monitoring.events.PY_START, None)
        monitoring.free_tool_id(self._tool_id)

    def take(self) -> dict:
        counts, self.counts = self.counts, dict.fromkeys(self.counts, 0)
        return counts


@pytest.mark.asyncio
async def test_ui_loop_starts_counts_fresh_calls_on_only_its_thread_and_caller():
    def called():
        pass

    def caller():
        called()

    async def resumed():
        await asyncio.sleep(0)
        called()

    starts = _UiLoopStarts(
        called=(called.__code__, caller.__code__),
        coroutine=(resumed.__code__, None),
    )
    previous = sys.getprofile()
    with pytest.raises(RuntimeError, match="observer cleanup"):
        with starts:
            assert sys.getprofile() is previous
            caller()
            called()
            await resumed()
            await resumed()
            await asyncio.to_thread(caller)
            assert starts.take() == {"called": 1, "coroutine": 2}
            raise RuntimeError("observer cleanup")
    assert sys.getprofile() is previous
    assert sys.monitoring.get_tool(starts._tool_id) is None
    caller()
    await resumed()
    assert starts.take() == {"called": 0, "coroutine": 0}


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


@pytest.mark.asyncio
@private_profile_test
async def test_a_finite_checked_reader_leaves_a_core_push_release_window(request):
    """A finite real checked reader defers the push, then releases normal replay."""
    from loguru import logger

    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            projection = await _fresh_display(console)
            path = config._get_effective_config_path()
            original = path.read_bytes()
            entered, release, expired = (
                threading.Event(),
                threading.Event(),
                threading.Event(),
            )
            reader = projection.read_current
            gate_errors = []

            def hold_one_reader(message):
                if entered.is_set():
                    return
                try:
                    frame = sys._getframe()
                    for _ in range(64):
                        if frame is None:
                            return
                        if (
                            frame.f_code is reader.__code__
                            and frame.f_globals is reader.__globals__
                            and frame.f_locals.get("projection") is projection
                        ):
                            entered.set()
                            if not release.wait(5):
                                expired.set()
                            return
                        frame = frame.f_back
                except Exception as error:
                    gate_errors.append(type(error).__name__)

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
            sink = None
            with starts:
                await pilot.pause(2.0)
                idle = starts.take()
                path.write_bytes(original + b"\n[unterminated\n")
                try:
                    await asyncio.to_thread(config.load_settings, True)
                    assert config._settings_cache_hit(path) is None
                    # Filter before Loguru's sink lock: other logging does not
                    # enter this gate. It blocks only this actual checked reader.
                    sink = logger.add(
                        hold_one_reader,
                        level="ERROR",
                        filter=lambda record: (
                            record["name"] == "tldw_chatbook.config"
                            and record["function"]
                            == "_load_cli_config_bootstrap_unlocked"
                        ),
                        backtrace=False,
                        diagnose=False,
                    )
                    projection.run(lambda: None)
                    await until(entered.is_set, timeout=30)
                    assert projection.pending, (
                        "gate did not hold an actual pending read"
                    )
                    rebuild = config._settings_rebuild_lock()
                    acquired = rebuild.acquire(blocking=False)
                    if acquired:
                        rebuild.release()
                    assert not acquired, "the real checked reader did not own rebuild"
                    starts.take()
                    config_sync.sync_console_core_state(console)
                    held = starts.take()
                    assert held["core_push"] == 0, "push entered the held rebuild lock"
                    assert console._console_control_bar_replay_whole_sync
                    worker = console.run_worker(
                        console._sync_native_console_chat_ui(),
                        exclusive=True,
                        group="console-sync",
                    )
                    console._sync_console_identity_surfaces()
                    await asyncio.wait_for(worker.wait(), timeout=5)
                    release.set()
                    await until(lambda: not projection.pending, timeout=30)
                    assert not expired.is_set(), (
                        "finite reader gate expired before release"
                    )
                    assert not gate_errors, gate_errors
                    first = starts.take()
                    deadline = asyncio.get_running_loop().time() + 10.0
                    while not (first["core_push"] and first["identity"]):
                        if asyncio.get_running_loop().time() > deadline:
                            break
                        await pilot.pause(0.1)
                        for name, count in starts.take().items():
                            first[name] += count
                    await pilot.pause(0.5)
                    starts.take()
                    await pilot.pause(2.0)
                    later = starts.take()
                finally:
                    release.set()
                    if sink is not None:
                        logger.remove(sink)
                    path.write_bytes(original)
                    await asyncio.to_thread(config.load_settings, True)
    assert later["full_sync"] <= idle["full_sync"] + 1, (
        f"finite reader left replaying: idle {idle}, later {later}"
    )
    assert first["core_push"] >= 1, (
        f"finite reader released but push never ran: {first}"
    )
    assert first["identity"] >= 1, (
        f"finite reader released but identity never ran: {first}"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_refused_core_pushes_stop_replaying_an_unreadable_config_and_say_so(
    request, monkeypatch
):
    """TASK-39145 (owner decision 2026-11-09: "Dev's rule, keep stop+notice").

    With config.toml unparseable, a failing read holds both config locks
    through its rebuild. Each refused core push still owes a whole replay, as
    on dev, until the episode has refused for ``CORE_REFUSAL_EPISODE_SECONDS``.
    Then the refusals stop owing replays and the status strip says the file
    cannot be read, once. A good load clears the notice and the next push runs.
    The episode clock is replaced and every probe is made by hand.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleStatusChips,
    )

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await pilot.pause(0.5)

            async def no_full_sync(*_args, **_kwargs):  # Only this test probes.
                return False

            monkeypatch.setattr(console, "_sync_native_console_chat_ui", no_full_sync)
            now = [1000.0]
            monkeypatch.setattr(config_sync, "_clock", lambda: now[0])
            pushes, replays, shown = [], [], []
            real_push = console._sync_console_chat_core_state

            def push():
                if (
                    sys._getframe(1).f_code
                    is config_sync.sync_console_core_state.__code__
                ):
                    pushes.append(1)
                return real_push()

            monkeypatch.setattr(console, "_sync_console_chat_core_state", push)
            real_replay = config_sync.replay_whole_console_sync
            monkeypatch.setattr(
                config_sync,
                "replay_whole_console_sync",
                lambda screen: replays.append(screen) or real_replay(screen),
            )
            chips = console.query_one(ConsoleStatusChips)
            real_notice = getattr(chips, "sync_config_unreadable", None)
            if real_notice is not None:

                def notice(on):  # Records each change the strip shows.
                    if on != chips._config_unreadable:
                        shown.append(on)
                    real_notice(on)

                monkeypatch.setattr(chips, "sync_config_unreadable", notice)

            def notice_visible() -> bool:
                return any(
                    segment.name == "config-unreadable" and segment.visible
                    for segment in chips._build_segments()
                )

            path = config._get_effective_config_path()
            original = path.read_bytes()
            path.write_bytes(original + b"\n[unterminated\n")
            await asyncio.to_thread(config.load_settings, True)  # Fails; recorded.
            assert config.get_config_load_failure() is not None
            entered, release = threading.Event(), threading.Event()
            real_aside = config._preserve_corrupt_config_aside

            def parked_aside(config_path):  # Runs inside the failing rebuild.
                entered.set()
                release.wait(30)
                return real_aside(config_path)

            monkeypatch.setattr(config, "_preserve_corrupt_config_aside", parked_aside)
            reader = threading.Thread(target=config.load_settings, args=(True,))
            reader.start()
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                episode = getattr(config_sync, "CORE_REFUSAL_EPISODE_SECONDS", 3.0)
                steps = round(episode / 0.2)
                for step in range(steps):  # Refusals 0.2 s apart, inside it.
                    config_sync.sync_console_core_state(console)
                    assert len(replays) == step + 1, "a refusal owed no whole replay"
                    now[0] += 0.2
                assert not notice_visible(), "the notice showed inside the episode"
                # The replayed FULL's warm-up begins the owed CORE handoff, as
                # its prepare does, before its own probe is refused.
                [(_core, token)] = config_sync._begin_console_config_handoffs(
                    console, spend.ConsoleReadinessConfigProjection.for_screen(console)
                )
                assert token is not None and token.active, "no Canvas poll handoff"
                config_sync.sync_console_core_state(console)  # Past the episode.
                assert len(replays) == steps, (
                    "refused pushes kept owing whole replays past the episode"
                )
                assert notice_visible(), "the refusals stopped without saying why"
                # r12p-rev4 I1: a dropped replay retires the handoff, or the
                # Canvas policy poll stays declined while its token lives.
                from tldw_chatbook.Canvas import config_policy

                assert "core" not in console._console_config_handoffs, (
                    "the stop left the CORE handoff owed"
                )
                assert not token.active and not any(
                    issued.owner is console for issued in config_policy._poll_handoffs
                ), "the stop left a Canvas poll handoff registered"
                config_sync.sync_console_core_state(console)  # A later refusal.
                assert len(replays) == steps and shown == [True], shown
                assert pushes == []
            finally:
                release.set()
                await asyncio.to_thread(reader.join, 10)
                path.write_bytes(original)
            await asyncio.to_thread(config.load_settings, True)  # A good load.
            for _ in range(40):  # The live Console's reads can hold a lock.
                config_sync.sync_console_core_state(console)
                if pushes:
                    break
                await pilot.pause(0.05)
            assert pushes == [1], "the first push after a good load did not run"
            assert not notice_visible(), "the notice outlived a good config load"
            assert shown == [True, False]


def _refused_warm_up(monkeypatch, console):
    """Observe FULL's warm-up, replay timers and results (TASK-38696).

    Returns ``(refusing, warms, armed, fulls, during, delays)``: set
    ``refusing[0]`` to make the presentation warm-up refuse, as a lasting
    storage pause or a replaced reader does; otherwise it warms for real.
    ``warms`` holds the time of each FULL warm-up, ``armed`` each
    coalesced-replay timer as ``(time, delay, timer)``, ``fulls`` each
    non-poll full sync's ``(end time, result)``, and ``delays`` each delay a
    refused FULL asked for. Callables put in ``during`` run once, inside the
    next FULL warm-up, before it answers.
    """
    loop = asyncio.get_running_loop()
    refusing, warms, armed, fulls, during, delays = [True], [], [], [], [], []
    real_warm = spend.ConsoleReadinessConfigProjection.warm
    prepare = config_sync.prepare_console_presentation_sync.__code__

    async def warm(projection):
        if sys._getframe(1).f_code is prepare:  # FULL's own warm-up only.
            warms.append(loop.time())
            while during:
                during.pop(0)()
            if refusing[0]:
                return False
        return await real_warm(projection)

    real_set_timer = console.set_timer

    def set_timer(delay, callback=None, *args, **kwargs):
        timer = real_set_timer(delay, callback, *args, **kwargs)
        if getattr(callback, "__name__", "") == "_run_coalesced_control_bar_sync":
            armed.append((loop.time(), delay, timer))
        return timer

    real_delay = config_sync.full_sync_replay_delay

    def replay_delay(screen):
        delays.append(real_delay(screen))
        return delays[-1]

    real_full = console._sync_native_console_chat_ui

    async def full(*args, **kwargs):
        result = await real_full(*args, **kwargs)
        if not kwargs.get("_preparing_poll"):
            fulls.append((loop.time(), result))
        return result

    monkeypatch.setattr(spend.ConsoleReadinessConfigProjection, "warm", warm)
    monkeypatch.setattr(console, "set_timer", set_timer)
    monkeypatch.setattr(console, "_sync_native_console_chat_ui", full)
    monkeypatch.setattr(config_sync, "full_sync_replay_delay", replay_delay)
    return refusing, warms, armed, fulls, during, delays


async def _fresh_backoff(armed, minimum):
    """The replay timer just armed with at least ``minimum`` seconds to wait."""
    loop = asyncio.get_running_loop()

    def fresh():
        return (
            bool(armed) and armed[-1][1] >= minimum and loop.time() - armed[-1][0] < 0.3
        )

    await until(fresh, timeout=20)
    return armed[-1]


@pytest.mark.asyncio
@private_profile_test
async def test_a_refused_full_sync_backs_off_and_renders_once_the_refusal_clears(
    request, monkeypatch
):
    """TASK-38696: a refused warm-up replays FULL on a backoff, not every 0.2 s.

    FULL awaits ``prepare_console_presentation_sync`` before it renders; a
    refusal handed the whole sync to the coalesced replay on a fixed 0.2 s
    timer, so a lasting refusal (a storage pause, a replaced reader) paid a
    full sync and a config read about five times a second. The replay now
    backs off on TASK-38649's schedule (0.2 s doubling to 5 s). Once the
    refusal clears, the next explicit full sync (any of the Console's direct
    callers, here a plain worker) renders at once instead of waiting out the
    backoff, and the schedule starts over.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await pilot.pause(1.0)  # The mount's own leftover syncs settle.
            loop = asyncio.get_running_loop()
            refusing, warms, armed, fulls, _during, _delays = _refused_warm_up(
                monkeypatch, console
            )
            start = loop.time()
            console.run_worker(
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            await pilot.pause(3.0)
            window = [round(when - start, 2) for when in warms if when - start < 3.0]
            # 0.2 s replays fit ~15 FULLs in 3 s; a backoff fits 4 or 5.
            assert 3 <= len(window) <= 6, (
                f"a 3 s warm-up refusal ran {len(window)} full syncs at {window}"
            )

            armed_at, delay, _timer = await _fresh_backoff(armed, 1.6)
            refusing[0] = False
            asked = loop.time()
            console.run_worker(  # Any explicit full sync, as a user's action makes.
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            await until(
                lambda: any(ok for when, ok in fulls if when >= asked), timeout=30
            )
            rendered = next(when for when, ok in fulls if ok and when >= asked)
            assert rendered < armed_at + delay, (
                f"the cleared refusal rendered {rendered - asked:.2f} s after it "
                f"was asked, after its pending {delay} s backoff had run out"
            )
            assert console._console_warm_refusals == 0

            refusing[0] = True  # A later refusal starts the schedule over.
            seen = len(armed)
            console.run_worker(
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            await until(lambda: len(armed) > seen, timeout=30)
            assert armed[seen][1] == pytest.approx(0.2), armed[seen:]
            refusing[0] = False


@pytest.mark.asyncio
@private_profile_test
async def test_a_successful_readiness_read_during_a_backoff_renders_at_once(
    request, monkeypatch
):
    """TASK-38696 review: the readiness projection's own successful read renders.

    A read that lands a changed value publishes a full sync. During a refused
    warm-up's backoff that full sync used to defer to the pending replay, so
    the Console stayed unrendered until the backoff ran out (up to 5 s).
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await pilot.pause(1.0)
            loop = asyncio.get_running_loop()
            refusing, _warms, armed, fulls, _during, _delays = _refused_warm_up(
                monkeypatch, console
            )
            console.run_worker(
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            armed_at, delay, _timer = await _fresh_backoff(armed, 1.6)
            refusing[0] = False
            projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
            projection.key = None  # The read lands as a change, so it publishes.
            asked = loop.time()
            projection.run(lambda: None)  # The projection's own read starts.
            await until(
                lambda: any(ok for when, ok in fulls if when >= asked), timeout=30
            )
            rendered = next(when for when, ok in fulls if ok and when >= asked)
            assert rendered < armed_at + delay, (
                f"the published read rendered {rendered - asked:.2f} s after it "
                f"started, after its pending {delay} s backoff had run out"
            )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("when", ["before_the_read", "during_the_read"])
async def test_a_chat_switch_restarts_the_refusal_backoff(request, monkeypatch, when):
    """TASK-38696 review: only refusals of one unmoved owner count toward the backoff.

    A switch to another chat moves the readiness owner. A refusal for the new
    chat, or one whose owner moved while its warm-up read ran, is not the
    lasting refusal the backoff is for: the next replay waits the fixed
    0.2 s, not a doubled wait.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await pilot.pause(1.0)
            refusing, _warms, armed, _fulls, during, delays = _refused_warm_up(
                monkeypatch, console
            )
            console.run_worker(
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            await _fresh_backoff(armed, 0.8)
            store = console._console_chat_store

            def switch() -> None:
                store.create_session(title="Other chat", activate=True)

            seen = len(delays)
            if when == "before_the_read":
                switch()
                console.run_worker(
                    console._sync_native_console_chat_ui(),
                    exclusive=True,
                    group="console-sync",
                )
            else:
                during.append(switch)  # The next replay's warm-up sees it.
            await until(lambda: len(delays) > seen, timeout=30)
            assert delays[seen] is True, (
                f"a refusal after a chat switch backed off: {delays[seen:]}"
            )
            refusing[0] = False


@pytest.mark.asyncio
@private_profile_test
async def test_the_fleet_survivor_tick_does_not_cut_a_warm_up_backoff(
    request, monkeypatch
):
    """TASK-38696 review: a periodic tick is no explicit sync for the backoff.

    The one-second fleet survivor tick runs a full sync while a background
    sub-agent outlives the run poll. Counted as explicit, it cut a lasting
    refusal's backoff every second, a replay and a config read each time.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await pilot.pause(1.0)
            loop = asyncio.get_running_loop()
            refusing, warms, armed, fulls, _during, _delays = _refused_warm_up(
                monkeypatch, console
            )
            console.run_worker(
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            armed_at, delay, _timer = await _fresh_backoff(armed, 1.6)
            seen = len(warms)
            synced = len(fulls)
            await console._fleet._console_fleet_survivor_tick()
            await pilot.pause(0.3)
            assert loop.time() < armed_at + delay, "the check ran after the backoff"
            # The tick must reach the full sync, or the next check passes vacuously.
            assert len(fulls) > synced, "the survivor tick never ran a full sync"
            assert len(warms) == seen, (
                f"the survivor tick ran the replay {len(warms) - seen} time(s) "
                f"inside its pending {delay} s backoff"
            )
            refusing[0] = False


@pytest.mark.asyncio
@private_profile_test
async def test_a_full_sync_asked_for_during_a_pause_runs_once_after_resume(
    request, monkeypatch
):
    """TASK-38696: the replay's backoff never delays or repeats a resume's sync.

    A maintenance pause leaves a requested full sync to the coalesced replay
    timer. When that timer is a refused warm-up's backoff, the resume cuts it
    short: the requested sync runs once, at resume, and the stopped backoff
    timer runs nothing later.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await pilot.pause(1.0)
            loop = asyncio.get_running_loop()
            refusing, _warms, armed, fulls, _during, _delays = _refused_warm_up(
                monkeypatch, console
            )
            console.run_worker(
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            armed_at, delay, timer = await _fresh_backoff(armed, 0.8)
            backoff_deliveries = []
            backoff_callback = timer._callback

            def record_backoff():
                backoff_deliveries.append(loop.time())
                return backoff_callback()

            monkeypatch.setattr(timer, "_callback", record_backoff)
            console._console_sync_maintenance_close_admission()
            console.run_worker(  # Asked for during the pause.
                console._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            await until(lambda: console._console_sync_requested, timeout=10)
            refusing[0] = False

            # CORE can independently request another replay during this FULL.
            # Follow the replacement scheduled by resume and the old timer.
            launched = []
            deferred_rail_reads = set()
            sync_rail_visibility = config_sync.sync_console_rail_visibility

            def observe_rail_visibility(screen):
                result = sync_rail_visibility(screen)
                if result is False:
                    deferred_rail_reads.add(asyncio.current_task())
                return result

            monkeypatch.setattr(
                config_sync, "sync_console_rail_visibility", observe_rail_visibility
            )
            replay = console._run_coalesced_control_bar_sync
            after_refresh = console.call_after_refresh

            def record_resume():
                run_worker = console.run_worker

                def record_full(work, *args, **kwargs):
                    started_at = loop.time()
                    worker = run_worker(work, *args, **kwargs)
                    if work is console._sync_native_console_chat_ui and (
                        kwargs.get("group") == "console-sync"
                    ):
                        launched.append((started_at, worker))
                    return worker

                with monkeypatch.context() as observer:
                    observer.setattr(console, "run_worker", record_full)
                    return replay()

            def observe_schedule(callback, *args, **kwargs):
                if callback == replay:
                    callback = record_resume
                return after_refresh(callback, *args, **kwargs)

            resumed = loop.time()
            with monkeypatch.context() as observer:
                observer.setattr(console, "call_after_refresh", observe_schedule)
                console._console_sync_maintenance_resume()
            await until(lambda: bool(launched), timeout=30)
            started_at, worker = launched[0]
            assert started_at < armed_at + delay, (
                f"the resume's sync waited {started_at - resumed:.2f} s, "
                f"out to its pending {delay} s backoff"
            )
            async with asyncio.timeout_at(resumed + 30):
                if await worker.wait() is not True:
                    # FULL already projected. Its later rail read can defer
                    # independently while storage is busy.
                    assert worker._task in deferred_rail_reads
                    await until(
                        lambda: any(ok for when, ok in fulls if when >= started_at),
                        timeout=30,
                    )
            await pilot.pause(max(0.0, armed_at + delay + 0.5 - loop.time()))
            assert len(launched) == 1, (
                f"the paused request replayed {len(launched)} times after resume"
            )
            assert not backoff_deliveries, "the stopped backoff timer still delivered"


def test_a_closing_console_arms_no_replay_and_counts_no_refusal():
    """TASK-38696: a closing Console's refused FULL schedules no replay.

    Preparation refuses at once for a closing screen without counting a
    warm-up refusal, and the replay request it leads to arms no timer.
    """
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    scheduled = []
    screen = SimpleNamespace(
        _closing=True,
        _console_warm_refusals=3,
        set_timer=lambda *args, **kwargs: scheduled.append(args),
        call_after_refresh=lambda *args: scheduled.append(args),
        _run_coalesced_control_bar_sync=lambda: None,
    )
    assert asyncio.run(config_sync.prepare_console_presentation_sync(screen)) is False
    assert screen._console_warm_refusals == 3
    delay = config_sync.full_sync_replay_delay(screen)
    ChatScreen._request_console_control_bar_sync(screen, delayed=delay)
    ChatScreen._request_console_control_bar_sync(screen)
    assert scheduled == [], "a closing Console armed a replay"


@pytest.mark.asyncio
async def test_a_control_bar_sync_reads_the_label_and_builds_no_workspace_context(
    monkeypatch,
):
    """TASK-36511.2: a control-bar sync shares one workspace-context build.

    Outside a run tick, the sync's Inspector leg and its rail-state leg each
    built the workspace context from the same inputs: 2 builds per sync,
    5.8-6.9 ms of its 27-31 ms of CPU (mounted, load 15). The sync now opens
    the tick's fingerprint-validated build scope, as ``settle_stopped_poll``
    does, so the second leg reuses the first build. Each sync still builds
    afresh: the scope ends with the sync.

    TASK-36571.1: both legs read only the workspace label now, so a sync
    builds no workspace context at all (renamed from
    ``test_a_control_bar_sync_builds_the_workspace_context_once``).
    """
    from Tests.UI.test_console_native_chat_flow import _sync_console_control_bar_now
    from tldw_chatbook.UI.Console_Modules import workspace

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        console, _composer = await ready_console(host, pilot, gateway)
        await _sync_console_control_bar_now(console)
        builds = []
        real = workspace.build_console_workspace_state

        def counted(*args, **kwargs):
            builds.append(True)
            return real(*args, **kwargs)

        monkeypatch.setattr(workspace, "build_console_workspace_state", counted)
        syncs = (
            console._sync_console_control_bar,
            console._sync_console_rail_and_controls,
        )
        for sync in syncs:
            for _ in range(2):
                builds.clear()
                assert sync() is not False, f"{sync.__name__} deferred"
                assert len(builds) == 0, (
                    f"{sync.__name__} built the workspace context {len(builds)} times"
                )
