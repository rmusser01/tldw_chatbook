"""Owner suffix + notification-triggered results pull (schedules-handoff
PR-6 task 4) -- the first-ever caller of the dormant
`ServerNotificationEventObserver.observe()`.

Two layers, matching this package's own split convention
(`test_schedules_results_tab.py`/`test_schedules_transfer_actions.py`):

- Owner-suffix rendering: a lightweight mock scheduling service (no
  server I/O needed), two widths via `run_test(size=...)`.
- Observer lifecycle + notification-triggered pull: a REAL
  `SchedulingService` over a tmp_path `ScheduledTasksDB`, with a
  duck-typed `notifications_scope_service`/`server_client` standing in
  for the SSE transport and the `/results` endpoint -- `_run_phase`/
  `_pull_results` (Task 1/`sync_now`'s own containment) run for real;
  these fakes only stand in for the network boundary.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger
from textual.widgets import DataTable

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import (
    MockSchedulingServiceMixin,
    rendered_row_cells,
)
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.Scheduling.services import SchedulingService
import tldw_chatbook.UI.Screens.scheduling.schedules_workbench as schedules_workbench_module
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    RESULTS_PULL_DEBOUNCE_SECONDS,
    SchedulesWorkbench,
)

_WAIT = RESULTS_PULL_DEBOUNCE_SECONDS + 0.2


async def _wait_until(pilot, predicate, *, timeout: float = 2.0, step: float = 0.05) -> None:
    """Poll ``predicate`` via real Pilot pauses instead of `App.workers.
    wait_for_complete()` -- that call waits for EVERY active worker,
    including the notification-observer's own long-lived one (which is
    deliberately still parked on `cancel_event.wait()` in several tests
    below, simulating an open SSE connection); waiting on it there would
    hang forever instead of proving the one worker under test finished.
    """
    elapsed = 0.0
    while elapsed < timeout:
        if predicate():
            return
        await pilot.pause(step)
        elapsed += step
    assert predicate(), f"condition not met within {timeout}s"


# ---------------------------------------------------------------------------
# Owner suffix on queue rows (plan ruling 4)
# ---------------------------------------------------------------------------


class _OwnerSuffixService(MockSchedulingServiceMixin):
    """Stub service returning one server-owned reminder."""

    async def list_reminders(self):
        return [
            ReminderTask(
                id="task-1",
                title="Nightly digest",
                owner_id="server:1",
                schedule_kind=ScheduleKind.RECURRING,
                cron="0 3 * * *",
                timezone="UTC",
            )
        ]

    async def list_tasks(self):
        return await self.list_reminders()


class _OwnerSuffixApp(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _OwnerSuffixService()


@pytest.mark.asyncio
async def test_queue_owner_suffix_shown_at_wide_width():
    async with _OwnerSuffixApp().run_test(size=(160, 42)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert "(server: 1)" in str(table.get_cell_at((0, 0)))


@pytest.mark.asyncio
async def test_queue_owner_suffix_hidden_at_compact_width():
    async with _OwnerSuffixApp().run_test(size=(100, 30)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert "(server: 1)" not in str(table.get_cell_at((0, 0)))
        assert "Nightly digest" in str(table.get_cell_at((0, 0)))


class _BracketTitleService(MockSchedulingServiceMixin):
    """Stub service whose reminder title carries a markup-shaped token."""

    async def list_reminders(self):
        return [
            ReminderTask(
                id="task-1",
                title="Nightly [bold] digest",
                owner_id="server:1",
                schedule_kind=ScheduleKind.RECURRING,
                cron="0 3 * * *",
                timezone="UTC",
            )
        ]

    async def list_tasks(self):
        return await self.list_reminders()


class _BracketTitleApp(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _BracketTitleService()


@pytest.mark.asyncio
async def test_queue_title_renders_brackets_literally():
    """Task 6 round 2, D8's class on the Queue table.

    The owner suffix survived live only because it uses parentheses; a
    user-authored title carrying a lowercase tag token would still have
    been eaten by `DataTable`'s `rich.text.Text.from_markup` formatting
    of string cells. Asserted on the painted cell, not the stored one.
    """
    async with _BracketTitleApp().run_test(size=(160, 42)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert rendered_row_cells(table, 0)[0] == "Nightly [bold] digest (server: 1)"


# ---------------------------------------------------------------------------
# Observer lifecycle + notification-triggered pull
# ---------------------------------------------------------------------------


class _ConnectedServerClient:
    """Duck-typed `SchedulingServerClient` stand-in: satisfies
    `_server_available`'s gate (`notifications_service is not None`) and
    `SyncEngine._pull_results`'s one call (`list_automation_results`).
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.notifications_service = object()
        self.calls = 0
        self.fail = fail

    async def list_automation_results(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("results endpoint down")
        return {"items": [], "has_more": False}


class _GatedServerClient(_ConnectedServerClient):
    """Blocks its first call until released, so a test can observe "a
    pull is in flight" deterministically instead of racing real time."""

    def __init__(self) -> None:
        super().__init__()
        self.call_started = asyncio.Event()
        self.release = asyncio.Event()

    async def list_automation_results(self, **kwargs):
        self.calls += 1
        self.call_started.set()
        await self.release.wait()
        return {"items": [], "has_more": False}


class FakeNotificationsScopeService:
    """Duck-typed `NotificationsScopeService` stand-in for
    `observe_server_feed_events` -- delivers a queued burst of fake
    events through the handler (mirroring `EventObserver.run`'s "await
    handler(event)" call shape) with no intervening real time, then
    blocks on the real `cancel_event` it is given, exactly like a
    long-lived SSE connection would, until `on_unmount` sets it.
    """

    def __init__(self, burst: list[dict] | None = None) -> None:
        self._burst = list(burst or [])
        self.call_count = 0

    async def observe_server_feed_events(
        self, *, handler, cancel_event, max_reconnects=0, **kwargs
    ):
        self.call_count += 1
        for raw in self._burst:
            await handler(SimpleNamespace(payload={"data": raw}))
        await cancel_event.wait()
        return SimpleNamespace(cancelled=True, handled_events=len(self._burst), reset=None)


class NotificationWorkbenchTestApp(ConsolidatedCSSApp):
    """A real Textual test app wired to a REAL `SchedulingService` over a
    tmp_path DB (matches `TransferWorkbenchTestApp`/`ResultsWorkbenchTestApp`
    in the sibling test files), plus an injectable `notifications_scope_
    service` -- the seam `_start_server_notification_observer` reads.
    """

    def __init__(
        self,
        db,
        *args,
        server_client=None,
        scope_service=None,
        runtime_source: str = "local",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        active_server_id = "1" if server_client is not None else None
        self.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(active_server_id=active_server_id)
        )
        self.scheduling_service = SchedulingService(
            db=db,
            server_client=server_client,
            runtime_source=runtime_source,
            app_getter=lambda: self,
        )
        self.notifications_scope_service = scope_service


@pytest.fixture
def notif_db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


@pytest.mark.asyncio
async def test_observer_not_started_without_a_server_connection(notif_db):
    """No server configured -- the dormant observer must stay dormant."""
    scope_service = FakeNotificationsScopeService()
    app = NotificationWorkbenchTestApp(notif_db, scope_service=scope_service)
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()
        assert scope_service.call_count == 0
        assert screen._notification_cancel_event is None


@pytest.mark.asyncio
async def test_observer_starts_on_mount_and_stops_on_unmount_double_cycle(notif_db):
    """First-ever caller of `ServerNotificationEventObserver.observe()`
    (via `NotificationsScopeService.observe_server_feed_events`):
    started on mount when a server connection is configured, stopped
    cleanly on unmount -- and a second mount/unmount cycle starts a
    genuinely fresh observer, proving nothing leaked from the first.
    """
    server_client = _ConnectedServerClient()
    scope_service = FakeNotificationsScopeService()
    app = NotificationWorkbenchTestApp(
        notif_db, server_client=server_client, scope_service=scope_service
    )
    async with app.run_test() as pilot:
        screen1 = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen1)
        await pilot.pause()
        assert scope_service.call_count == 1
        cancel_event_1 = screen1._notification_cancel_event
        assert cancel_event_1 is not None
        assert not cancel_event_1.is_set()

        await pilot.app.pop_screen()  # real unmount
        await pilot.pause()
        assert cancel_event_1.is_set()
        await pilot.app.workers.wait_for_complete()

        screen2 = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen2)
        await pilot.pause()
        assert scope_service.call_count == 2
        cancel_event_2 = screen2._notification_cancel_event
        assert cancel_event_2 is not None
        assert cancel_event_2 is not cancel_event_1
        assert not cancel_event_2.is_set()

        await pilot.app.pop_screen()
        await pilot.pause()
        assert cancel_event_2.is_set()
        await pilot.app.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_non_automation_kind_is_ignored(notif_db):
    """A non-`automation_run_*` kind must not schedule a pull."""
    app = NotificationWorkbenchTestApp(notif_db)
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()

        event = SimpleNamespace(payload={"data": {"kind": "notification.created"}})
        acked = await screen._on_server_notification_event(event)
        assert acked is True
        # Ack contract holds even for an ignored kind, but nothing was
        # scheduled -- deterministic, no waiting required.
        assert screen._results_pull_debounce_timer is None
        assert screen._results_pull_running is False


@pytest.mark.asyncio
async def test_automation_event_burst_of_three_triggers_exactly_one_pull(notif_db):
    """A burst of `automation_run_*` events delivered by the observer
    (no intervening real time) collapses into ONE debounced pull."""
    server_client = _ConnectedServerClient()
    burst = [{"kind": "automation_run_completed"}] * 3
    scope_service = FakeNotificationsScopeService(burst=burst)
    app = NotificationWorkbenchTestApp(
        notif_db, server_client=server_client, scope_service=scope_service
    )
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()
        assert scope_service.call_count == 1  # observer delivered the burst

        assert server_client.calls == 0  # still inside the debounce window
        await pilot.pause(_WAIT)
        # Not `workers.wait_for_complete()`: the observer's own worker is
        # deliberately still running (blocked on cancel_event, an open
        # "SSE connection") -- waiting on it here would hang forever.
        await _wait_until(pilot, lambda: server_client.calls >= 1)
        assert server_client.calls == 1


@pytest.mark.asyncio
async def test_trigger_during_in_flight_pull_queues_exactly_one_follow_up(notif_db):
    """A trigger that lands while a pull is already running absorbs into
    a single follow-up pull -- never a second concurrent worker, never
    more than one queued rerun (single-flight, no pile-up)."""
    server_client = _GatedServerClient()
    app = NotificationWorkbenchTestApp(notif_db, server_client=server_client)
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()

        event = SimpleNamespace(payload={"data": {"kind": "automation_run_failed"}})
        await screen._on_server_notification_event(event)
        await pilot.pause(_WAIT)  # debounce settles, pull #1 starts and blocks
        await asyncio.wait_for(server_client.call_started.wait(), timeout=2.0)
        assert screen._results_pull_running is True
        assert server_client.calls == 1

        # A second trigger while pull #1 is in flight.
        await screen._on_server_notification_event(event)
        await pilot.pause(_WAIT)  # its own debounce settles too
        assert server_client.calls == 1  # absorbed, no second worker/call yet
        assert screen._results_pull_rerun_requested is True

        server_client.release.set()  # let pull #1 finish
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert server_client.calls == 2  # exactly one follow-up pull
        assert screen._results_pull_running is False
        assert screen._results_pull_rerun_requested is False


@pytest.mark.asyncio
async def test_pull_failure_surfaces_via_sync_error_path_without_killing_observer(
    notif_db,
):
    """A failed pull is contained by `SyncEngine._run_phase` (the same
    path `sync_now` uses): recorded as a persisted sync error -- the
    existing sync-error path `_refresh_owner_select` already renders --
    and never reaches the observer's own coroutine, which is a wholly
    separate worker."""
    server_client = _ConnectedServerClient(fail=True)
    scope_service = FakeNotificationsScopeService()
    app = NotificationWorkbenchTestApp(
        notif_db, server_client=server_client, scope_service=scope_service
    )
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()
        assert scope_service.call_count == 1  # observer is up

        event = SimpleNamespace(payload={"data": {"kind": "automation_run_failed"}})
        await screen._on_server_notification_event(event)
        await pilot.pause(_WAIT)
        # Not `workers.wait_for_complete()` -- see `_wait_until`'s docstring:
        # the observer's own worker is deliberately still parked here.
        await _wait_until(pilot, lambda: not screen._results_pull_running)

        assert server_client.calls == 1
        state = notif_db.get_sync_state("local") or {}
        assert state.get("sync_errors"), "pull failure must surface as a sync error"
        assert "results endpoint down" in state["sync_errors"][-1]["message"]

        # The observer's own coroutine was never touched by the pull
        # failure: its cancel event is still unset, and the app is alive.
        assert screen._notification_cancel_event is not None
        assert not screen._notification_cancel_event.is_set()
        assert pilot.app.is_running


class _ScriptedScopeService:
    """Fix round 1 log-discipline test double: scripted sequence of
    connection outcomes for `_run_server_notification_observer`'s outer
    retry loop --

    1. RuntimeError (first failure -- should WARN)
    2. RuntimeError (repeat, same class -- should DEBUG)
    3. RuntimeError (repeat, same class -- should DEBUG)
    4. clean non-cancelled return (a "success" -- should INFO "reconnected")
    5. ValueError (a NEW class after a success -- should WARN again)
    6. blocks on cancel_event (an open connection, ends the script)
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def observe_server_feed_events(
        self, *, handler, cancel_event, max_reconnects=0, **kwargs
    ):
        self.call_count += 1
        if self.call_count <= 3:
            raise RuntimeError(f"boom {self.call_count}")
        if self.call_count == 4:
            return SimpleNamespace(cancelled=False, handled_events=0, reset=None)
        if self.call_count == 5:
            raise ValueError("a different failure class")
        await cancel_event.wait()
        return SimpleNamespace(cancelled=True, handled_events=0, reset=None)


@pytest.mark.asyncio
async def test_sustained_failure_warns_once_then_debugs_then_info_on_reconnect(
    notif_db, monkeypatch
):
    """Fix round 1 (task-4-review.md Medium): a sustained failure of the
    SAME exception class must log exactly ONE warning (not an ERROR-level
    traceback per retry attempt); identical-class repeats log at debug; a
    successful reconnect logs one info and clears the remembered class,
    so a later DIFFERENT failure class warns again instead of staying
    silent at debug forever.
    """
    # The 5s cadence between restart attempts is unchanged in production
    # (see the module constant); shrunk here only so this test does not
    # take 5+ seconds per scripted attempt.
    monkeypatch.setattr(
        schedules_workbench_module,
        "_NOTIFICATION_OBSERVER_RESTART_DELAY_SECONDS",
        0.02,
    )

    server_client = _ConnectedServerClient()
    scope_service = _ScriptedScopeService()
    app = NotificationWorkbenchTestApp(
        notif_db, server_client=server_client, scope_service=scope_service
    )

    records: list[tuple[str, str]] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="DEBUG",
    )
    try:
        async with app.run_test() as pilot:
            screen = SchedulesWorkbench(app_instance=pilot.app)
            await pilot.app.push_screen(screen)
            await _wait_until(
                pilot, lambda: scope_service.call_count >= 6, timeout=5.0
            )

            observer_records = [
                (level, msg) for level, msg in records if "notification observer" in msg
            ]
            warnings = [r for r in observer_records if r[0] == "WARNING"]
            debugs = [r for r in observer_records if r[0] == "DEBUG"]
            infos = [r for r in observer_records if r[0] == "INFO"]
            errors = [r for r in observer_records if r[0] == "ERROR"]

            # Exactly one warning per distinct failure class (RuntimeError,
            # then ValueError after the reconnect reset the memory) --
            # class-change re-warns.
            assert len(warnings) == 2, warnings
            assert "RuntimeError" in warnings[0][1]
            assert "ValueError" in warnings[1][1]
            # The two REPEAT RuntimeError failures (attempts 2 and 3) log
            # at debug, not warning.
            assert len(debugs) == 2, debugs
            # Exactly one reconnect info, once the run finally succeeds.
            assert infos == [
                ("INFO", "Schedules notification observer reconnected")
            ]
            # Never a full-traceback ERROR dump per retry -- the bug this
            # fix round exists to close.
            assert errors == []

            await pilot.app.pop_screen()
            await pilot.pause()
    finally:
        loguru_logger.remove(sink_id)


@pytest.mark.asyncio
async def test_mount_pulls_results_once_to_catch_up_on_acked_events(notif_db):
    """HIGH (Qodo): `_on_server_notification_event` acks BEFORE the pull it
    schedules has run, so an event acked just before this screen went away
    advanced the observer's durable cursor and was never redelivered --
    its results were lost until some later trigger happened to arrive.

    Closed by catching up on mount rather than by re-timing the ack (the
    ack-then-advance contract is what stops an unrelated stream replaying
    forever). The pull is a newest-window walk, i.e. idempotent, so one at
    mount recovers whatever any number of lost events would have fetched.
    """
    server_client = _ConnectedServerClient()
    scope_service = FakeNotificationsScopeService()
    app = NotificationWorkbenchTestApp(
        notif_db, server_client=server_client, scope_service=scope_service
    )
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()
        assert server_client.calls == 0  # still inside the debounce window

        await pilot.pause(_WAIT)
        await _wait_until(pilot, lambda: server_client.calls >= 1)
        # Exactly one -- mounting must not fan out into repeated pulls.
        assert server_client.calls == 1
        assert screen._results_pull_running is False


@pytest.mark.asyncio
async def test_mount_catch_up_and_an_event_burst_stay_single_flight(notif_db):
    """The catch-up pull shares the live-event path, so mounting straight
    into a burst of `automation_run_*` events costs ONE pull, not one per
    trigger plus one for the mount."""
    server_client = _ConnectedServerClient()
    scope_service = FakeNotificationsScopeService(
        burst=[{"kind": "automation_run_completed"}] * 3
    )
    app = NotificationWorkbenchTestApp(
        notif_db, server_client=server_client, scope_service=scope_service
    )
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()
        assert scope_service.call_count == 1  # observer delivered the burst

        await pilot.pause(_WAIT)
        await _wait_until(pilot, lambda: server_client.calls >= 1)
        assert server_client.calls == 1
        assert screen._results_pull_rerun_requested is False


@pytest.mark.asyncio
async def test_no_catch_up_pull_without_a_server_connection(notif_db):
    """No server configured -- nothing to catch up on, and `_run_phase`
    would only record a sync error. Same gate as the observer's."""
    app = NotificationWorkbenchTestApp(notif_db)
    async with app.run_test() as pilot:
        screen = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(screen)
        await pilot.pause()
        assert screen._results_pull_debounce_timer is None
        await pilot.pause(_WAIT)
        assert screen._results_pull_running is False
