"""Lever L2: routine Console poll ticks stay light between full reconciliations.

Two layers, deliberately:

* A rig around a bare ``ChatScreen`` that captures the REAL ``_poll_transcript``
  closure from ``_start_console_transcript_sync_timer`` and runs it against the
  real ``_console_transcript_poll_needed`` predicate, with the full and light
  bodies replaced by recorders and the routing clock pinned. It proves the
  routing decisions exactly, without wall-clock races.
* Mounted controls on the qualified received-Send fixture (real app, real store,
  real config admission, Send held at the provider-preparation boundary). They
  prove the real light pass publishes transcript rows without entering the
  config-locked syncs, that a full reconciliation still lands at least every
  ``CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS`` and on the stopping tick, and that
  direct callers keep the full pass.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from textual.css.query import NoMatches

from Tests.UI.test_console_hook_review_send_freeze import _until
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Console_Modules import poll_cadence
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

INTERVAL = poll_cadence.CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS
#: Rig tick spacing. A binary fraction, so cadence arithmetic is exact.
ADVANCE = 0.25
#: A full pass is admitted strictly after the tick that routed it; the real
#: pass stamps ``time.perf_counter()`` after the routing read.
ADMISSION = 2.0**-20
LIGHT = ["tabs", "transcript"]


class _Clock:
    """The routing clock, advanced by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _FakeTimer:
    def __init__(self, stopped: list) -> None:
        self._stopped = stopped

    def stop(self) -> None:
        self._stopped.append(True)


class _PollRig:
    """A bare screen whose poll routes through the real closure and predicate."""

    def __init__(self, monkeypatch) -> None:
        self.clock = _Clock()
        monkeypatch.setattr(poll_cadence, "_clock", self.clock)
        self.events: list[str] = []
        self.full_results: list[bool] = []
        self.full_hold: asyncio.Event | None = None
        self.light_hold: asyncio.Event | None = None
        self.workers: list[tuple[str, str | None]] = []
        self.stopped: list[bool] = []
        self.sessions = [
            SimpleNamespace(id="viewed", settings=object()),
            SimpleNamespace(id="other", settings=object()),
        ]
        self.store = SimpleNamespace(
            active_session_id="viewed", sessions=lambda: list(self.sessions)
        )
        self.controller = SimpleNamespace(
            run_state=SimpleNamespace(status=ConsoleRunStatus.STREAMING),
            in_flight=1,
            fleet_wake=None,
        )
        self.controller.in_flight_run_count = lambda: self.controller.in_flight
        self.custody = True
        runtime = SimpleNamespace(
            chat_store=self.store,
            chat_controller=self.controller,
            change_review_coordinator=None,
            has_custodied_turns=lambda: self.custody,
        )
        screen = ChatScreen.__new__(ChatScreen)
        screen._closing = False
        screen._closed = False
        screen.app_instance = SimpleNamespace(ui_responsiveness_monitor=None)
        screen._console_runtime = lambda: runtime
        screen._console_sync_in_progress = False
        screen._console_sync_requested = False
        screen._console_transcript_sync_timer = None
        screen._workspace = SimpleNamespace(
            _invalidate_console_persisted_rows_cache=lambda: self.events.append(
                "invalidate"
            )
        )
        screen._fleet = SimpleNamespace(
            _maybe_start_console_fleet_survivor_tick=lambda: self.events.append(
                "survivor"
            )
        )

        def run_worker(coroutine, **kwargs):
            self.workers.append((coroutine.cr_code.co_name, kwargs.get("group")))
            coroutine.close()

        async def full() -> bool:
            self.events.append("full")
            if screen._console_sync_in_progress:
                screen._console_sync_requested = True
                return False
            screen._console_sync_in_progress = True
            self.clock.now += ADMISSION
            started = self.clock.now
            try:
                if self.full_hold is not None:
                    await self.full_hold.wait()
                result = self.full_results.pop(0) if self.full_results else True
                if result:
                    # The record the real full pass writes when it completes.
                    store = self.store
                    screen._console_full_sync_completed = (
                        started,
                        store,
                        (store.active_session_id, tuple(store.sessions())),
                    )
                return result
            finally:
                screen._console_sync_in_progress = False

        async def tabs() -> None:
            self.events.append("tabs")
            if self.light_hold is not None:
                await self.light_hold.wait()

        async def transcript() -> None:
            self.events.append("transcript")

        def query_one(*_args, **_kwargs):
            raise NoMatches()  # Unmounted: the run chip has nowhere to paint.

        captured = {}

        def set_interval(interval, callback):
            captured["interval"], captured["callback"] = interval, callback
            return _FakeTimer(self.stopped)

        screen.run_worker = run_worker
        screen.set_interval = set_interval
        screen._sync_native_console_chat_ui = full
        screen._sync_console_native_session_tabs = tabs
        screen._sync_native_console_transcript = transcript
        screen.query_one = query_one
        self.screen = screen
        ChatScreen._start_console_transcript_sync_timer(screen)
        assert captured["interval"] == 0.2
        self.poll = captured["callback"]

    def take(self) -> list[str]:
        events, self.events[:] = list(self.events), []
        return events

    async def tick(self) -> list[str]:
        self.clock.now += ADVANCE
        await self.poll()
        return self.take()

    def settle_viewed(self) -> None:
        self.controller.run_state.status = ConsoleRunStatus.COMPLETED
        self.controller.in_flight -= 1


TICKS_PER_INTERVAL = int(INTERVAL / ADVANCE)


async def _rig_after_first_full(monkeypatch) -> _PollRig:
    rig = _PollRig(monkeypatch)
    # No pass has completed for any owner yet: the first tick is full.
    assert await rig.tick() == ["full"]
    return rig


@pytest.mark.asyncio
async def test_routine_ticks_within_the_interval_publish_without_full_sync(
    monkeypatch,
):
    rig = await _rig_after_first_full(monkeypatch)
    # Every tick strictly inside the interval after that pass's start is
    # light; the tick that reaches the interval reconciles fully.
    for _ in range(TICKS_PER_INTERVAL - 1):
        assert await rig.tick() == LIGHT
    assert await rig.tick() == ["full"]
    assert rig.screen._console_transcript_sync_timer is not None
    assert rig.stopped == [] and rig.workers == []


@pytest.mark.asyncio
async def test_full_reconciliation_runs_every_interval_while_polling(monkeypatch):
    rig = await _rig_after_first_full(monkeypatch)
    full_at = [rig.clock.now]
    for _ in range(5 * TICKS_PER_INTERVAL):
        events = await rig.tick()
        assert events in (LIGHT, ["full"])
        if events == ["full"]:
            full_at.append(rig.clock.now)
    gaps = [later - earlier for earlier, later in zip(full_at, full_at[1:])]
    assert len(gaps) >= 4
    # At most once per interval from the poll, and at most one tick late.
    assert all(INTERVAL <= gap <= INTERVAL + ADVANCE + ADMISSION for gap in gaps)


@pytest.mark.asyncio
async def test_a_new_poll_stays_light_for_its_first_interval(monkeypatch):
    """A Send's first seconds carry no routine full pass for a known owner."""
    rig = await _rig_after_first_full(monkeypatch)
    rig.clock.now += 60.0  # The last full pass is long past.
    rig.screen._console_transcript_sync_timer = None
    ChatScreen._start_console_transcript_sync_timer(rig.screen)
    for _ in range(TICKS_PER_INTERVAL):
        assert await rig.tick() == LIGHT
    assert await rig.tick() == ["full"]


@pytest.mark.asyncio
async def test_a_completed_direct_full_pass_defers_the_routine_one(monkeypatch):
    rig = await _rig_after_first_full(monkeypatch)
    # The very next tick would be due by the poll's own last full pass...
    rig.clock.now += INTERVAL - ADVANCE
    # ...but a direct caller (a send/stop transition, a publication worker)
    # reconciled first, and the cadence counts from that pass's start.
    assert await rig.screen._sync_native_console_chat_ui() is True
    rig.take()
    for _ in range(TICKS_PER_INTERVAL - 1):
        assert await rig.tick() == LIGHT
    assert await rig.tick() == ["full"]


@pytest.mark.asyncio
async def test_stopping_tick_runs_full_and_only_then_stops(monkeypatch):
    rig = await _rig_after_first_full(monkeypatch)
    assert await rig.tick() == LIGHT
    rig.settle_viewed()
    rig.custody = False
    assert await rig.tick() == ["full", "invalidate", "survivor"]
    assert rig.screen._console_transcript_sync_timer is None
    assert rig.stopped == [True]


@pytest.mark.asyncio
async def test_deferred_final_full_refresh_keeps_the_poll_alive(monkeypatch):
    """Integration 116628850f: a deferred last pass may not stop the poll."""
    rig = await _rig_after_first_full(monkeypatch)
    rig.settle_viewed()
    rig.custody = False
    rig.full_results = [False, False]
    assert await rig.tick() == ["full"]
    assert await rig.tick() == ["full"]
    assert rig.screen._console_transcript_sync_timer is not None
    assert rig.stopped == []
    assert await rig.tick() == ["full", "invalidate", "survivor"]
    assert rig.stopped == [True]


@pytest.mark.asyncio
async def test_torn_down_tick_goes_straight_to_the_full_pass_entry_guard(
    monkeypatch,
):
    rig = await _rig_after_first_full(monkeypatch)
    rig.screen._closing = True

    def gone():
        raise AssertionError("routing read the runtime of a torn-down screen")

    rig.screen._console_runtime = gone
    rig.full_results = [False]  # What the real pass's teardown guard returns.
    assert await rig.tick() == ["full"]
    assert rig.stopped == [] and rig.workers == []


@pytest.mark.asyncio
async def test_light_tick_never_reaches_the_stop_tail(monkeypatch):
    rig = await _rig_after_first_full(monkeypatch)
    rig.light_hold = asyncio.Event()
    tick = asyncio.create_task(rig.tick())
    assert await _until(lambda: rig.events == ["tabs"], 1)
    # Work drains while the light pass is suspended: the stop edge belongs
    # to the NEXT tick, and that tick must be a full pass.
    rig.settle_viewed()
    rig.custody = False
    rig.light_hold.set()
    assert await tick == LIGHT
    assert rig.screen._console_transcript_sync_timer is not None
    assert await rig.tick() == ["full", "invalidate", "survivor"]


@pytest.mark.asyncio
async def test_viewed_settle_while_background_runs_reconciles_on_that_tick(
    monkeypatch,
):
    """ADR-226 item 6: the stop edge is not the only terminal transition."""
    rig = await _rig_after_first_full(monkeypatch)
    rig.controller.in_flight = 2
    assert await rig.tick() == LIGHT
    rig.settle_viewed()  # In flight 2 -> 1: the poll stays alive.
    assert await rig.tick() == ["full"]
    assert rig.screen._console_transcript_sync_timer is not None
    assert await rig.tick() == LIGHT


@pytest.mark.asyncio
async def test_settle_edge_is_not_discharged_by_a_pass_admitted_before_it(
    monkeypatch,
):
    rig = await _rig_after_first_full(monkeypatch)
    rig.controller.in_flight = 2
    assert await rig.tick() == LIGHT
    rig.full_hold = asyncio.Event()
    direct = asyncio.create_task(rig.screen._sync_native_console_chat_ui())
    assert await _until(lambda: rig.screen._console_sync_in_progress, 1)
    rig.take()
    rig.settle_viewed()
    # The tick observes the edge while the earlier-admitted pass runs, so
    # it requests a trailing pass behind it instead of waiting on it.
    assert await rig.tick() == ["full"]
    assert rig.screen._console_sync_requested is True
    rig.full_hold.set()
    assert await direct is True
    rig.full_hold = None
    # Drop that request so only the edge itself can route the next tick: the
    # early pass COMPLETED after the edge but was admitted before it.
    rig.screen._console_sync_requested = False
    assert await rig.tick() == ["full"]
    assert await rig.tick() == LIGHT


@pytest.mark.parametrize("change", ["session", "membership", "replaced", "settings"])
@pytest.mark.asyncio
async def test_an_unreconciled_owner_routes_the_tick_to_full(monkeypatch, change):
    rig = await _rig_after_first_full(monkeypatch)
    assert await rig.tick() == LIGHT
    if change == "session":
        rig.store.active_session_id = "other"
    elif change == "membership":
        rig.sessions.append(SimpleNamespace(id="new", settings=object()))
    elif change == "replaced":
        rig.sessions[0] = SimpleNamespace(id="viewed", settings=object())
    else:
        rig.sessions[0].settings = None
    assert await rig.tick() == ["full"]


@pytest.mark.parametrize(
    "flag",
    [
        "_console_sync_requested",
        "_console_sync_maintenance_paused",
        "_console_control_bar_replay_whole_sync",
    ],
)
@pytest.mark.asyncio
async def test_deferred_or_stranded_full_demand_keeps_the_full_route(monkeypatch, flag):
    rig = await _rig_after_first_full(monkeypatch)
    setattr(rig.screen, flag, True)
    assert (await rig.tick())[0] == "full"


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.asyncio
async def test_routine_tick_during_a_running_full_pass_is_absorbed(
    monkeypatch, requested
):
    rig = await _rig_after_first_full(monkeypatch)
    rig.clock.now += 10 * INTERVAL  # Even an overdue cadence defers to it.
    rig.screen._console_sync_in_progress = True
    rig.screen._console_sync_requested = requested
    assert await rig.tick() == []
    assert rig.screen._console_sync_requested is requested
    assert rig.workers == []


@pytest.mark.asyncio
async def test_full_request_during_light_pass_replays_once_after_it(monkeypatch):
    rig = await _rig_after_first_full(monkeypatch)
    rig.light_hold = asyncio.Event()
    tick = asyncio.create_task(rig.tick())
    assert await _until(lambda: rig.events == ["tabs"], 1)
    assert rig.screen._console_sync_in_progress is True
    # A direct full request (a session switch, say) lands across the await.
    assert await rig.screen._sync_native_console_chat_ui() is False
    assert rig.screen._console_sync_requested is True
    rig.light_hold.set()
    await tick
    assert rig.workers == [("full", "console-sync")]
    assert rig.screen._console_sync_requested is False
    assert rig.screen._console_sync_in_progress is False


@pytest.mark.asyncio
async def test_maintenance_keeps_full_request_for_its_resume_owner(monkeypatch):
    rig = await _rig_after_first_full(monkeypatch)
    rig.light_hold = asyncio.Event()
    tick = asyncio.create_task(rig.tick())
    assert await _until(lambda: rig.events == ["tabs"], 1)
    rig.screen._console_sync_requested = True
    rig.screen._console_sync_maintenance_paused = True
    rig.light_hold.set()
    await tick
    assert rig.workers == []
    assert rig.screen._console_sync_requested is True


# -- mounted controls ---------------------------------------------------------


def _record_console_passes(monkeypatch, console):
    """Record full passes (any caller) and the poll task's config/transcript work.

    Full rows are ``("full", started, result, polled, config_entries)``; the
    last field counts the poll task's config-locked entries inside that pass,
    so a pass that deferred at entry (zero) is told apart from one that ran.
    """
    rows = []
    config_entries = [0]
    original_config_sync = spend.run_console_config_sync
    original_full = ChatScreen._sync_native_console_chat_ui
    original_transcript = ChatScreen._sync_native_console_transcript

    def poll_task() -> bool:
        timer = console._console_transcript_sync_timer
        return timer is not None and asyncio.current_task() is timer._task

    def config_sync(*args, **kwargs):
        if poll_task():
            config_entries[0] += 1
            rows.append(("config", time.perf_counter()))
        return original_config_sync(*args, **kwargs)

    async def full(self):
        if self is not console:
            return await original_full(self)
        polled = poll_task()
        started, before = time.perf_counter(), config_entries[0]
        result = await original_full(self)
        rows.append(("full", started, result, polled, config_entries[0] - before))
        return result

    async def transcript(self):
        polled = self is console and poll_task()
        await original_transcript(self)
        if polled:
            rows.append(("transcript", time.perf_counter()))

    monkeypatch.setattr(spend, "run_console_config_sync", config_sync)
    monkeypatch.setattr(ChatScreen, "_sync_native_console_chat_ui", full)
    monkeypatch.setattr(ChatScreen, "_sync_native_console_transcript", transcript)
    return rows


def _sync_idle(console) -> bool:
    """No pass running and no full demand pending, deferred or replaying."""
    return not (
        console._console_sync_in_progress
        or console._console_sync_requested
        or getattr(console, "_console_control_bar_replay_whole_sync", False)
        or vars(console).get("_console_sync_maintenance_paused", False)
    )


async def _settled_direct_full(console) -> float:
    """Complete one direct full pass and return its recorded start."""
    previous = vars(console).get("_console_full_sync_completed")
    assert await _until(lambda: _sync_idle(console), 10)
    # No await between the idle check and admission. A deferral inside the
    # pass hands it to the original replay owner; wait for that to land.
    await console._sync_native_console_chat_ui()
    assert await _until(
        lambda: vars(console).get("_console_full_sync_completed") is not previous
        and _sync_idle(console),
        10,
    ), "No direct full reconciliation completed"
    return console._console_full_sync_completed[0]


async def _held_send(case, *, preparing=False):
    """Hold a live turn; by default model ticks OUTSIDE a held Preparing receipt.

    An unchanged manual Preparing receipt keeps #3023's own narrowed pass
    (``_sync_console_poll_display_ui``); lever L2's light cadence governs every
    other routine tick of a live turn. These controls hold the Send at its
    received record only to keep the turn live, so unless ``preparing`` is set
    they report no Preparing receipt to measure the L2 window.
    """
    from Tests.UI.test_console_poll_reconciliation import (
        _qualify_warm_capture_sources,
    )
    from Tests.UI.test_console_received_intent_feedback import (
        _held_received_record,
        _send,
    )

    await _qualify_warm_capture_sources(case)
    # The saved direct-provider route, as the original terminal control uses.
    case.console.app_instance.app_config["console"]["agent_runtime"] = False
    _send(case, "enter")
    await _held_received_record(case)
    assert case.console._console_transcript_sync_timer is not None
    if not preparing:
        case.console._console_preparing_poll_record = lambda: None


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_mounted_preparing_ticks_keep_the_narrowed_preparing_pass(monkeypatch):
    """While an unchanged manual Preparing receipt is held, ticks are not light."""
    from Tests.UI.test_console_received_intent_feedback import _received_console_case

    monkeypatch.setattr(poll_cadence, "CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS", 3600.0)
    async with _received_console_case(
        monkeypatch, "cadence-preparing", durable=True
    ) as case:
        await _held_send(case, preparing=True)
        console = case.console
        await _settled_direct_full(console)
        assert console._console_preparing_poll_record() is not None
        light, preparing = [], []
        original_light = poll_cadence.sync_console_poll_display
        original_preparing = console._sync_console_poll_display_ui

        async def light_pass(screen):
            light.append(screen)
            return await original_light(screen)

        async def preparing_pass():
            preparing.append(True)
            return await original_preparing()

        monkeypatch.setattr(poll_cadence, "sync_console_poll_display", light_pass)
        console._sync_console_poll_display_ui = preparing_pass
        assert await _until(lambda: len(preparing) >= 2, 5), (light, preparing)
        assert light == []
        assert case.provider_calls == []


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_mounted_routine_polls_publish_rows_without_config_locked_sync(
    monkeypatch,
):
    from Tests.UI.test_console_received_intent_feedback import _received_console_case
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    async with _received_console_case(
        monkeypatch, "cadence-light", durable=True
    ) as case:
        await _held_send(case)
        console = case.console
        rows = _record_console_passes(monkeypatch, console)
        started = await _settled_direct_full(console)
        rows.clear()
        message = case.store.append_message(
            case.session.id,
            role=ConsoleMessageRole.SYSTEM,
            content="Routine poll publication marker",
        )
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert await _until(
            lambda: message.id in transcript.mounted_message_content_ids()
            and any(row[0] == "transcript" for row in rows),
            1.5,
        ), f"No routine poll tick published the new transcript row: {rows!r}"
        assert time.perf_counter() - started < INTERVAL
        polled = [
            row for row in rows if row[0] == "config" or row[0] == "full" and row[3]
        ]
        assert not polled, (
            "Routine poll ticks inside the interval entered the full pass or its "
            f"config-locked syncs: {rows!r}"
        )
        assert case.provider_calls == []


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_mounted_poll_reconciles_every_interval_and_on_the_stopping_tick(
    monkeypatch,
):
    from Tests.UI.test_console_received_intent_feedback import _received_console_case

    async with _received_console_case(
        monkeypatch, "cadence-full", durable=True
    ) as case:
        await _held_send(case)
        console = case.console
        rows = _record_console_passes(monkeypatch, console)
        await _settled_direct_full(console)
        observed_from = console._console_full_sync_completed[0]
        await asyncio.sleep(2 * INTERVAL + 1.0)
        observed_to = time.perf_counter()
        # The poll ATTEMPTS its routine pass; on this held Send a pass can
        # defer (configuration busy) and its existing coalesced replay owner
        # then completes it -- so completion is counted for any caller.
        attempts = [row for row in rows if row[0] == "full" and row[3]]
        assert attempts, f"The poll never attempted a routine full pass: {rows!r}"
        completed = sorted(
            [observed_from]
            + [row[1] for row in rows if row[0] == "full" and row[2] is True]
        )
        gaps = [
            later - earlier
            for earlier, later in zip(completed, [*completed[1:], observed_to])
        ]
        # Upper bound: a completed reconciliation at least every interval, plus
        # tick granularity, a deferral's replay, and a loaded Windows host.
        assert all(gap <= INTERVAL + 1.5 for gap in gaps), (gaps, rows)
        # Lower bound: the poll never runs its pass sooner than the interval
        # after the start of the previous completed pass of any caller. (An
        # attempt that deferred at entry -- a pending replay -- did no work.)
        for row in (row for row in attempts if row[4]):
            previous = max(start for start in completed if start < row[1])
            assert row[1] - previous >= INTERVAL, (previous, row, rows)
        light_ticks = sum(row[0] == "transcript" for row in rows)
        assert light_ticks > len(attempts), rows
        # Release the held Send. The turn settles, and the poll may stop only
        # behind a completed full pass that its own stopping tick ran.
        rows.clear()
        case.probe.release.set()
        assert await _until(lambda: console._console_transcript_sync_timer is None, 15)
        assert case.provider_calls, "The held Send never reached the provider"
        polled = [row for row in rows if row[0] != "full" or row[3]]
        assert polled and polled[-1][0] == "full" and polled[-1][2] is True, rows
        assert not console._console_transcript_poll_needed()


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_mounted_direct_full_sync_calls_stay_full_while_polling(monkeypatch):
    from Tests.UI.test_console_received_intent_feedback import _received_console_case

    async with _received_console_case(
        monkeypatch, "cadence-direct", durable=True
    ) as case:
        await _held_send(case)
        console = case.console
        test_task = asyncio.current_task()
        entries, light = [], []
        original = spend.run_console_config_sync
        original_light = poll_cadence.sync_console_poll_display

        def config_sync(*args, **kwargs):
            if asyncio.current_task() is test_task:
                entries.append(args[0])
            return original(*args, **kwargs)

        async def light_pass(screen):
            if asyncio.current_task() is test_task:
                light.append(screen)
            return await original_light(screen)

        monkeypatch.setattr(spend, "run_console_config_sync", config_sync)
        monkeypatch.setattr(poll_cadence, "sync_console_poll_display", light_pass)
        await _settled_direct_full(console)
        for _ in range(2):
            # Back to back, well inside the poll interval: a direct call is
            # never rate-limited or narrowed. It may still DEFER under its
            # original checks (busy configuration on this held Send); that
            # deferral happens inside the full body, after live admission.
            assert await _until(lambda: _sync_idle(console), 10)
            entries.clear()
            await console._sync_native_console_chat_ui()
            assert entries, "A direct call skipped live-state config admission"
        assert light == []
        assert console._console_transcript_sync_timer is not None


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_mounted_full_request_during_light_pass_replays_with_current_owner(
    monkeypatch,
):
    """A session activation landing across a light pass's await is not lost."""
    from textual.widgets import Button

    from Tests.UI.test_console_received_intent_feedback import _received_console_case
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )

    # Keep every routine tick on the light route for this control.
    monkeypatch.setattr(poll_cadence, "CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS", 3600.0)
    async with _received_console_case(
        monkeypatch, "cadence-late-full", durable=True
    ) as case:
        successor = case.store.create_session(
            title="Current polling owner",
            workspace_id=case.session.workspace_id,
            settings=case.session.settings,
            activate=False,
        )
        case.store.set_session_draft(successor.id, "Current owner draft")
        await _held_send(case)
        console = case.console
        await _settled_direct_full(console)
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        lock = surface._session_sync_lock
        running = [0]
        original_light = poll_cadence.sync_console_poll_display

        async def light(screen):
            running[0] += 1
            try:
                return await original_light(screen)
            finally:
                running[0] -= 1

        monkeypatch.setattr(poll_cadence, "sync_console_poll_display", light)
        activation = None
        await lock.acquire()
        held = True
        try:
            # A routine light pass is suspended on the real tab-surface lock.
            assert await _until(
                lambda: running[0] == 1
                and console._console_sync_in_progress
                and bool(lock._waiters),
                5,
            ), "No routine light pass reached the tab-surface lock"
            activation = asyncio.create_task(
                console._session._activate_native_console_session(successor.id)
            )
            assert await _until(
                lambda: case.store.active_session_id == successor.id
                and console._console_sync_requested,
                5,
            ), "The activation's full request did not coalesce behind the light pass"
            assert not activation.done() and running[0] == 1
            lock.release()
            held = False
            assert await _until(
                lambda: activation.done()
                and not console._console_sync_in_progress
                and not console._console_sync_requested
                and not getattr(
                    console, "_console_control_bar_replay_whole_sync", False
                )
                and vars(console).get(
                    "_console_full_sync_completed", (0, None, (None, ()))
                )[2][0]
                == successor.id,
                10,
            ), "The coalesced full request did not settle on the current owner"
            activation.result()
            assert console.query_one(
                f"#console-session-tab-{successor.id}", Button
            ).has_class("console-session-tab-active")
            assert console._console_visible_draft_session_id == successor.id
            assert case.composer.draft_text() == "Current owner draft"
        finally:
            if held:
                lock.release()
            if activation is not None and not activation.done():
                activation.cancel()
                await asyncio.gather(activation, return_exceptions=True)
