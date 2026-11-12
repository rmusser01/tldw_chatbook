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
state) or an explicit sync asked, and at least once a second as a backstop
while it ticks (only while run work is live); when the poll stops itself on
a skipped tick, the stop renders. When it runs, it runs as before: same
locks, same refusal and deferral during a local storage pause.

These tests drive the real ChatScreen poll callback (the timer's own
coroutine) on the mounted send harness and count, per tick, the tick's own
projection attempts (``_sync_console_rail_and_controls``, which enters the
checked config read: dev's checked display or the native operation). The
clock the backstop reads is replaced, so no test depends on real elapsed
time.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from Tests.private_profile import private_profile_test
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


def _install_clock(monkeypatch) -> _Clock:
    """Replace the backstop's clock, and dev's checked-display clock with it.

    On dev (PR #3023) the tick's projection renders from a checked display
    that a worker refreshes once a (monotonic) second; a stale one is
    re-read and, while that read is pending, defers the projection to the
    delayed replay. On a loaded runner real seconds pass between ticks, so
    the same clock drives both: the display goes stale exactly when the
    test moves the backstop's clock a second on.
    """
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import config_sync, console_spend_projection

    clock = _Clock()
    # raising=False: the parent build has no backstop clock (and fails below).
    monkeypatch.setattr(config_sync, "_clock", clock, raising=False)
    monkeypatch.setattr(
        console_spend_projection, "time", SimpleNamespace(monotonic=clock)
    )
    return clock


class _Projections:
    """Counts the projections the test's own ticks attempt and render.

    A tick awaited by the test runs in the test's task; syncs the app starts
    on its own (workers, the send path) run in other tasks and are not
    counted, so a count is exactly what the ticks under test did.
    """

    def __init__(self, monkeypatch, console) -> None:
        self.task = asyncio.current_task()
        self.calls = 0
        self.deferred = 0
        self.renders = 0
        self.renders_anywhere = 0
        # Rebased onto dev (PR #3023): a sync now makes other checked config
        # calls the cadence does not gate (rail visibility, the cost chip),
        # so count the tick's projection itself, the one entry it gates.
        real_project = console._sync_console_rail_and_controls
        real_render = console._sync_console_control_bar_under_config

        def counting_project():
            if asyncio.current_task() is not self.task:
                return real_project()
            self.calls += 1
            result = real_project()
            if result is False:
                self.deferred += 1
            return result

        def counting_render(*args, **kwargs):
            self.renders_anywhere += 1
            if asyncio.current_task() is self.task:
                self.renders += 1
            return real_render(*args, **kwargs)

        console._sync_console_rail_and_controls = counting_project
        console._sync_console_control_bar_under_config = counting_render


async def _settle(console) -> None:
    """Let any in-flight (or re-armed) sync finish before the next tick.

    Ends on the check itself, with no await after it, so the caller's next
    sync starts before any other task can.
    """
    for _ in range(3):
        await asyncio.sleep(0.01)
        await until(lambda: not console._console_sync_in_progress, timeout=30)


def _replay_pending(console) -> bool:
    context = getattr(console, "_console_context_read_snapshot", None)
    config = getattr(console, "_console_readiness_config_projection", None)
    return (
        getattr(console, "_console_control_bar_replay_whole_sync", False)
        or getattr(config, "pending", False)
        or getattr(context, "pending_key", None) is not None
    )


async def _explicit_sync(console, seen) -> None:
    """Run one explicit sync in the test's task (retrying one that deferred).

    A sync the app re-armed can still start between ``_settle`` and the
    call when it was queued behind a deferred control-bar replay; the
    explicit call then hands its work to that sync, so try again. On dev
    (PR #3023) the projection itself defers while its checked display is
    cold, handing the render to the delayed whole-sync replay; wait that
    out and try again, until one renders.
    """
    for _ in range(20):
        await until(lambda: not _replay_pending(console), timeout=30)
        await _settle(console)
        before = (seen.calls, seen.deferred)
        await console._sync_native_console_chat_ui()
        if seen.calls > before[0] and seen.deferred == before[1]:
            return
    raise AssertionError("no explicit sync rendered in the test's task")


class _Poll:
    """The 0.2 s poll's own callback, called by hand while its timer is held.

    No tick runs that the test did not ask for: the timer is stopped and
    restarting it is suppressed until ``release`` hands it back (the
    transcript only paints a reply through it).

    Nor does any sync the app starts from another task (a worker, a replay,
    the send path), until ``unhold`` or ``release``: those share the
    cadence's state with the test's ticks, and one landing between two
    counted ticks resets when the last projection ran and what it saw, so a
    loaded runner could move a backstop count (review MINOR 6). Held syncs
    are dropped and counted in ``dropped``.
    """

    def __init__(self, console) -> None:
        self.console = console
        console._start_console_transcript_sync_timer()
        timer = console._console_transcript_sync_timer
        assert timer is not None
        self._poll = timer._callback
        console._stop_console_transcript_sync_timer()
        console._start_console_transcript_sync_timer = lambda: None
        owner = asyncio.current_task()
        real = console._sync_native_console_chat_ui
        self.dropped = 0

        async def held() -> None:
            if asyncio.current_task() is owner:
                await real()
            else:
                self.dropped += 1

        console._sync_native_console_chat_ui = held

    async def __call__(self, *, wait_replay: bool = True) -> None:
        if wait_replay:  # Else the sync refuses at entry and the tick is void.
            await until(lambda: not _replay_pending(self.console), timeout=30)
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
        await until(lambda: not _replay_pending(self.console), timeout=30)
        await _settle(self.console)
        result = change()
        await self._poll()
        return result

    def unhold(self) -> None:
        """Let syncs the app starts from other tasks run again."""
        vars(self.console).pop("_sync_native_console_chat_ui", None)

    def release(self) -> None:
        self.unhold()
        del self.console._start_console_transcript_sync_timer
        self.console._start_console_transcript_sync_timer()


async def _quiesce(tick, seen, limit: int = 10) -> None:
    """Tick until one projects nothing (the clock stands still, no backstop).

    After a run, real work lands on its own: the conversation is saved and
    its dictionary scope refreshed, continuation and usage writes move the
    display revision. Each is an input change and projects, correctly; a
    count of "unchanged" ticks starts once they have.

    The same holds after any projection on dev (PR #3023): one whose
    presentation reads find a changed owner (a run status, a revision)
    refreshes them on a worker and publishes a whole sync with the result,
    which the next tick serves (a sync other than the poll's asked).
    """
    from tldw_chatbook.UI.Console_Modules import config_sync

    cadence = config_sync._cadence(tick.console)
    for _ in range(limit):
        # A tick refused while a replay is pending would look quiet.
        await until(lambda: not _replay_pending(tick.console), timeout=30)
        before = seen.calls
        await tick()
        objects, values = config_sync._projection_inputs(tick.console)
        if (
            seen.calls == before
            and not _replay_pending(tick.console)
            and cadence.owed
            and cadence.requested == cadence.served
            and values == cadence.values
            and len(objects) == len(cadence.objects)
            and all(now is then for now, then in zip(objects, cadence.objects))
        ):
            return
    # Without the cadence (dev) every poll tick enters the projection, so
    # this is where a build with no skipping fails.
    raise AssertionError(
        f"every one of {limit} poll ticks entered the projection"
        " (inputs still moving, or no tick skips it)"
    )


async def _until_a_tick_skips(tick, seen, clock, limit: int = 10):
    """Tick a live poll until a tick skipped after a projection that rendered.

    The clock stands still. A projection that defers records nothing, so
    the next tick still owes the backstop and projects again; a tick refused
    at entry projects nothing and skips nothing. Either can end a
    ``_quiesce``, so check the cadence itself.

    Returns:
        The screen's cadence: a projection rendered within the backstop's
        second, and a tick has skipped since.
    """
    from tldw_chatbook.UI.Console_Modules import config_sync

    cadence = config_sync._cadence(tick.console)
    for _ in range(limit):
        await _quiesce(tick, seen)
        if cadence.owed and clock.now - cadence.at < config_sync.RUN_BACKSTOP_SECONDS:
            return cadence
    raise AssertionError(f"no poll tick skipped a fresh projection in {limit} rounds")


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
    """(a) unchanged ticks skip it; (c) the backstop, in a run and its tail; (e) stop.

    The poll ticks only while run work is live, which outlasts the run
    status; ticks in that tail get the backstop too, and the tick on which
    the poll stops itself leaves the controls rendered.
    """
    from tldw_chatbook import config
    from tldw_chatbook.UI.Console_Modules import config_sync

    clock = _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            tick = _Poll(console)
            seen = _Projections(monkeypatch, console)
            try:
                await _explicit_sync(console, seen)
                await _quiesce(tick, seen)  # The sync's published follow-up.
                base = seen.calls

                # (a) Within a run's second, ticks with no input change skip.
                for _ in range(5):
                    await tick()
                assert (
                    seen.calls == base
                ), f"{seen.calls - base} projections in 5 unchanged ticks"

                # (c) The backstop: once a second during a run, not before.
                clock.now += 0.6
                await tick()
                assert seen.calls == base, "a backstop projection before a second"
                clock.now += 0.4
                await tick()
                assert seen.calls == base + 1, "no backstop projection a second later"
                await _quiesce(tick, seen)
                base = seen.calls
                clock.now += 1.0
                await tick()
                assert seen.calls == base + 1, "no second backstop projection"
                assert seen.renders == seen.calls - seen.deferred
            finally:
                tick.release()
                gateway.validation_release.set()
            await _run_over(console)

            # (c) The run's tail: its status is terminal, but the poll still
            # ticks (a custodied turn, a wake delivery, a finishing run
            # worker). Unchanged ticks skip; the backstop still runs.
            tick = _Poll(console)
            console._console_transcript_poll_needed = lambda: True
            try:
                await _explicit_sync(console, seen)
                await _quiesce(tick, seen)
                tail = seen.calls
                for _ in range(3):
                    await tick()
                assert seen.calls == tail, (
                    f"{seen.calls - tail} checked config operations in 3 "
                    "unchanged ticks of the run's tail"
                )
                source = config.current_config_identity()
                cadence = config_sync._cadence(console)
                projected_at = cadence.at
                await tick.right_after(lambda: setattr(clock, "now", clock.now + 1.0))
                if seen.calls == tail:
                    # FULL now prepares before its projection. A real config
                    # generation change during that native await must refuse
                    # the captured source, without consuming the backstop.
                    assert config.current_config_identity() != source
                    assert console._console_control_bar_replay_whole_sync
                    assert cadence.at == projected_at
                    await tick()
                assert seen.calls == tail + 1, "no backstop in the run's tail"
            finally:
                del console._console_transcript_poll_needed

            # (e) The poll stops itself on a tick that skips (nothing moved
            # within the second); the stop renders what it skipped, so the
            # Console is left as current as an always-rendering last tick
            # left it. Run work is over, but the poll is kept live until a
            # tick skips after a projection that rendered: the tail's
            # backstop above can defer, and a deferral records nothing, so
            # without this the stop tick would still owe the backstop and
            # project on its own, with nothing left for the stop to render.
            await until(
                lambda: not console._console_transcript_poll_needed(), timeout=30
            )
            console._console_transcript_poll_needed = lambda: True
            try:
                cadence = await _until_a_tick_skips(tick, seen, clock)
            finally:
                del console._console_transcript_poll_needed
            stops: list[None] = []
            stop = console._stop_console_transcript_sync_timer
            console._stop_console_transcript_sync_timer = lambda: (
                stops.append(None),
                stop(),
            )
            forced: list[bool] = []
            project = config_sync.project_console_tick

            def spy(screen, *, force: bool = False) -> bool:
                result = project(screen, force=force)
                if force:
                    forced.append(result)
                return result

            monkeypatch.setattr(config_sync, "project_console_tick", spy)
            try:
                before = (seen.calls, seen.renders)
                await tick()
                assert stops, "the poll did not stop itself"
                assert forced == [True], (
                    f"the stop forced {len(forced)} projections; want the one"
                    " that renders what the skipped last tick left"
                )
                assert (seen.calls, seen.renders) == (
                    before[0] + 1,
                    before[1] + 1,
                ), "the stopping poll's last tick left the controls unrendered"
                assert not cadence.owed
            finally:
                monkeypatch.setattr(config_sync, "project_console_tick", project)
                del console._stop_console_transcript_sync_timer
                tick.release()


@pytest.mark.asyncio
async def test_each_input_change_is_one_projection_on_the_next_poll_tick(monkeypatch):
    """(b) A reinstalled config or settings: the very next tick projects.

    The clock stands still, so no backstop runs; the run is held at provider
    validation, so nothing else changes between the ticks. On dev (PR
    #3023) a projection after a config change can be followed by the
    whole sync its refreshed presentation reads publish (or by the replay
    of a cold checked display's deferral), so after the change's own tick
    the inputs are left to settle, and must.
    """
    from tldw_chatbook import config

    _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            tick = _Poll(console)
            seen = _Projections(monkeypatch, console)
            try:
                await _explicit_sync(console, seen)

                async def projections_on_the_tick_after(change) -> int:
                    await _quiesce(tick, seen)
                    before = seen.calls
                    await tick.right_after(change)
                    projected = seen.calls - before
                    await _quiesce(tick, seen)
                    return projected

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
                    count = await projections_on_the_tick_after(change)
                    assert (
                        count == 1
                    ), f"{count} projections on the tick after {name}; want 1"
                assert seen.renders == seen.calls - seen.deferred
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

    _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            tick = _Poll(console)
            # A live poll (it would stop itself, and settle, with no run work).
            monkeypatch.setattr(
                console, "_console_transcript_poll_needed", lambda: True
            )
            seen = _Projections(monkeypatch, console)
            await _explicit_sync(console, seen)
            await _quiesce(tick, seen)  # The sync's published follow-up.
            base, rendered, deferred = seen.calls, seen.renders, seen.deferred
            for _ in range(3):
                await tick()
            assert (seen.calls, seen.renders) == (
                base,
                rendered,
            ), f"{seen.calls - base} projections in 3 unchanged ticks"
            assert not getattr(console, "_console_control_bar_replay_whole_sync", False)
            # The pause half needs the app's own delayed replay to run.
            tick.unhold()
            pause = await tick.right_after(storage_admission._begin_local_pause)
            try:
                assert (seen.calls, seen.renders) == (base + 1, rendered)
                assert seen.deferred == deferred + 1
                assert console._console_control_bar_replay_whole_sync
                assert console._console_control_bar_sync_scheduled
                # The replay owns the refresh; a tick while it is pending waits.
                await tick(wait_replay=False)
                assert seen.calls == base + 1
            finally:
                pause.resume()
                tick.release()
            rendered = seen.renders_anywhere
            await until(
                lambda: (
                    seen.renders_anywhere > rendered
                    and not console._console_control_bar_replay_whole_sync
                ),
                timeout=30,
            )


@pytest.mark.asyncio
async def test_agent_progress_and_cost_total_project_on_the_next_tick(monkeypatch):
    """What changes on its own during a run keeps the parent's tick rate.

    Review round (2026-10-10): the Agent rail's status and step lines and
    the cost chip's running total render only inside these projections, so
    a skipped tick left them up to a second late. Both are inputs now. The
    clock stands still, so only the change itself can project.
    """
    _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            tick = _Poll(console)
            seen = _Projections(monkeypatch, console)
            agent = console._agent
            try:
                await _explicit_sync(console, seen)
                await _quiesce(tick, seen)  # The sync's published follow-up.
                base = seen.calls
                await tick()
                assert seen.calls == base, "an unchanged tick projected"

                # A step: the bridge publishes the run's next step. The send
                # harness has no agent bridge, so the rail's own reader
                # answers what a published step would make it show.
                step = ("Agent: running · step 2", "· ⚙ web_search", "")
                await tick.right_after(
                    lambda: monkeypatch.setattr(
                        agent, "_console_agent_section_lines", lambda: step
                    )
                )
                assert seen.calls == base + 1, "an Agent rail step change waited"
                await tick()
                assert seen.calls == base + 1

                # The running total: a sub-agent bills tokens.
                monkeypatch.setattr(
                    agent, "_console_agent_fleet_token_total", lambda: 1_204
                )
                await tick()
                assert seen.calls == base + 2, "a cost-total change waited"
                await tick()
                assert seen.calls == base + 2
                assert seen.renders == seen.calls - seen.deferred
            finally:
                tick.release()
                gateway.validation_release.set()
            await _run_over(console)


@pytest.mark.asyncio
@private_profile_test
async def test_a_legacy_tool_approval_projects_on_the_next_tick(
    request: pytest.FixtureRequest, tmp_path, monkeypatch
):
    """A legacy unscoped tool round's arrival and resolution keep the tick rate.

    Rebase onto dev 7700c63a68 (2026-10-22): such a round mounts its card
    with no kind-registry entry, so no pending-round count moves, yet
    ``_console_pending_approval_count`` counts it from the resume state for
    the controls, rail and Inspector. The clock stands still, so only the
    round itself can project. The send harness cannot take a round armed on
    a worker thread (its controller posts to the unstarted inner app), so
    this runs on the navigation app the mounted approval journeys use.
    """
    from Tests.UI.test_console_headless_approval import _arm, _risk_row
    from Tests.UI.test_console_pending_interrupt_projection import (
        _build_app,
        _seed_console,
        _start_live_turn,
    )

    _install_clock(monkeypatch)
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        console, controller, store, session_id = await _seed_console(app, pilot)
        _start_live_turn(console, controller, store, session_id)
        tick = _Poll(console)
        # A live poll (it would stop itself, and settle, with no run work).
        monkeypatch.setattr(console, "_console_transcript_poll_needed", lambda: True)
        seen = _Projections(monkeypatch, console)
        worker = None

        def resume_approval():
            return console._task_resume_state.pending_approval

        try:
            await _explicit_sync(console, seen)
            await _quiesce(tick, seen)
            base = seen.calls
            worker, result = _arm(controller, None, call=_risk_row())
            await until(lambda: bool(resume_approval()), timeout=30)
            assert controller.pending_round_count(session_id) == 0
            assert console._console_pending_approval_count() == 1
            await tick()
            assert seen.calls == base + 1, "a legacy approval waited"
            await tick()
            assert seen.calls == base + 1, "the tick after it projected"

            controller.resolve_pending_approval(
                {"builtin__write_file": "deny"},
                round_id=resume_approval()["round_id"],
            )
            await until(lambda: not worker.is_alive(), timeout=30)
            assert result["decisions"] == {"builtin__write_file": "deny"}
            await until(
                lambda: console._console_pending_approval_count() == 0, timeout=30
            )
            await tick()
            assert seen.calls == base + 2, "a resolved legacy approval waited"
        finally:
            if worker is not None and worker.is_alive() and resume_approval():
                controller.resolve_pending_approval(
                    {"builtin__write_file": "deny"},
                    round_id=resume_approval()["round_id"],
                )
            tick.release()
            controller.begin_shutdown()
            if worker is not None:
                await until(lambda: not worker.is_alive(), timeout=30)


@pytest.mark.asyncio
async def test_a_skipped_tick_under_a_native_pause_request_runs_its_later_steps_safely(
    monkeypatch,
):
    """A tick that skips the projection still runs the sync's later steps.

    The parent's tick always entered the checked operation first, so a
    pause requested by another process refused it at entry and the sync
    returned before its later steps. A skipped tick runs them with no
    operation around them. They read only warm state: no raw-participant
    scope (the only place a native pause request refuses) is entered, so
    nothing raises. A local pause or a config cache miss are inputs, so
    those ticks still attempt (and defer) the projection as before.

    Only the loop thread's scope checks count: on dev (PR #3023) the checked
    display's once-a-second refresh reads config on a worker thread, under
    its own scope, and may run while the tick awaits; a pause refuses that
    read inside the worker, which stays cold and retries, as without this
    cadence.
    """
    from tldw_chatbook.Backup_Recovery import admission

    _install_clock(monkeypatch)
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = await _held_run(host, pilot, gateway)
            tick = _Poll(console)
            seen = _Projections(monkeypatch, console)
            try:
                await _explicit_sync(console, seen)
                await _quiesce(tick, seen)  # The sync's published follow-up.
                base = seen.calls
                probes: list[tuple[str, ...]] = []

                def requested(self, names):
                    if threading.current_thread() is threading.main_thread():
                        probes.append(names)
                    return True  # Another process asked to pause storage.

                monkeypatch.setattr(admission.Admission, "pause_requested", requested)
                # The heaviest later path: re-render the transcript (receipts).
                console._last_native_transcript_refresh_key = None
                await tick()
                assert seen.calls == base, "the tick did not skip"
                assert probes == [], f"later steps entered {len(probes)} scopes"
            finally:
                monkeypatch.undo()
                tick.release()
                gateway.validation_release.set()
            await _run_over(console)
