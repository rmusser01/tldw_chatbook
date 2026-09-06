"""task-23105: the sync surface must not over-promise.

Pressing s used to toast "Sync completed." even when the engine's policy
refusal meant nothing was pulled or pushed and the bar still read
"Last pull: — Last push: —"; the owner bar always showed Server(url) +
Clear even for local-only profiles; Clear's disabled state was
color-only.
"""

from datetime import datetime, timezone

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.sync_status_widget import (
    SyncStatusWidget,
)


from types import SimpleNamespace

from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB,
    MockSchedulingServiceMixin,
    MockServerClient,
)
from tldw_chatbook.Scheduling.services.sync_engine import SyncOutcome


class _SyncService(MockSchedulingServiceMixin):
    """Service whose sync_now returns a configurable SyncOutcome."""

    server_client = MockServerClient(notifications_service=object())

    def __init__(self, outcome: SyncOutcome | None) -> None:
        self._outcome = outcome
        self.db = MockSchedulingDB()

    async def list_tasks(self, owner_id=None, include_projections=True):
        return [
            ReminderTask(
                id="task-1",
                title="Test",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            )
        ]

    async def sync_now(self, owner_id=None):
        if self._outcome is not None and self._outcome.status == "ok" and (
            self._outcome.pulled or self._outcome.pushed
        ):
            # A real transfer records the pull timestamp, like the engine.
            self.db.update_sync_state(
                "local", last_pull_at="2026-08-28T12:00:00+00:00"
            )
        return self._outcome


class _App(ConsolidatedCSSApp):
    def __init__(self, service, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = service
        # The s-key gate uses the same predicate as the bar collapse
        # (review F10): server client AND an active server id.
        self.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(active_server_id="example.com")
        )


async def _sync_via_action(pilot, workbench):
    # Fix round 1: action_sync_now is now async (it re-probes reachability
    # before deciding, matching _run_owner_transfer/_on_owner_server).
    await workbench.action_sync_now()
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_sync_that_transfers_nothing_says_so():
    app = _App(_SyncService(SyncOutcome("ok", pulled=0, pushed=0)))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        messages = [n.message for n in pilot.app._notifications]
        assert any("nothing to pull or push" in m for m in messages), messages
        assert not any(m.startswith("Sync completed") for m in messages), messages


@pytest.mark.asyncio
async def test_sync_that_transfers_reports_counts():
    app = _App(_SyncService(SyncOutcome("ok", pulled=2, pushed=1)))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        messages = [n.message for n in pilot.app._notifications]
        assert "Sync completed — pulled 2, pushed 1." in messages, messages
        # And the bar shows the recorded pull timestamp.
        pull = workbench.query_one("#scheduling-last-pull", Static)
        assert "2026-08-28" in str(pull.render())


@pytest.mark.asyncio
async def test_failed_sync_reports_failure_not_a_noop():
    """Review F3: the engine swallows server errors internally; the UI
    must report them as failures, never as an info-severity no-op."""
    app = _App(_SyncService(SyncOutcome("error", error="connection refused")))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        notifications = list(pilot.app._notifications)
        failure = [n for n in notifications if "Sync failed" in n.message]
        assert failure and failure[0].severity == "error", [
            (n.message, n.severity) for n in notifications
        ]
        assert "connection refused" in failure[0].message
        assert not any(
            "nothing to pull or push" in n.message for n in notifications
        )


@pytest.mark.asyncio
async def test_failed_sync_also_surfaces_phase_errors():
    """Final review finding 6: `_on_sync_failed` used to drop
    `phase_errors` outright -- an automation phase (definition push/pull,
    results pull) that failed in the SAME cycle as the reminder-phase
    failure this toast already reports never reached the user at all on
    THIS path (only `_on_sync_completed`'s success path read them)."""
    app = _App(
        _SyncService(
            SyncOutcome(
                "error",
                error="connection refused",
                phase_errors=("Automation results pull: scheduled_task_not_found",),
            )
        )
    )
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        notifications = list(pilot.app._notifications)
        failure = [n for n in notifications if "Sync failed" in n.message]
        assert failure and "connection refused" in failure[0].message
        also = [n for n in notifications if "Automation results pull" in n.message]
        assert also, [
            (n.message, n.severity) for n in notifications
        ]
        assert also[0].severity == "warning"


@pytest.mark.asyncio
async def test_not_applicable_sync_says_so():
    app = _App(_SyncService(SyncOutcome("not_applicable")))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        messages = [n.message for n in pilot.app._notifications]
        assert any("Sync skipped" in m for m in messages), messages


@pytest.mark.asyncio
async def test_sync_key_agrees_with_the_collapsed_bar():
    """Review F10: with a notifications service but NO active server id,
    the bar collapses ('sync is off') -- the s key must refuse with the
    same predicate instead of running a sync underneath it."""
    service = _SyncService(SyncOutcome("ok", pulled=1, pushed=0))
    app = _App(service)
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_server_id=None)
    )
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        messages = [n.message for n in pilot.app._notifications]
        assert any("nothing to sync" in m for m in messages), messages
        assert not any(m.startswith("Sync completed") for m in messages)


# --- local-owner collapse + Clear visibility ------------------------------


class _BarHarness(ConsolidatedCSSApp):
    def __init__(self, **widget_kwargs) -> None:
        super().__init__()
        self._widget_kwargs = widget_kwargs

    def compose(self) -> ComposeResult:
        yield SyncStatusWidget(**self._widget_kwargs)


@pytest.mark.asyncio
async def test_local_owner_without_server_collapses_to_one_line():
    app = _BarHarness(
        current_owner="local", active_server_id=None, server_available=False
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query_one("#scheduling-owner-server", Button).display
        assert not app.query_one("#scheduling-owner-local", Button).display
        assert not app.query_one("#scheduling-last-pull", Static).display
        assert not app.query_one("#scheduling-last-push", Static).display
        note = app.query_one("#scheduling-sync-local-note", Static)
        assert note.display
        assert "Local schedules" in str(note.render())
        assert "server" in str(note.render()).lower()


@pytest.mark.asyncio
async def test_server_available_keeps_owner_controls_visible():
    app = _BarHarness(
        current_owner="local",
        active_server_id="http://127.0.0.1:8000",
        server_available=True,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#scheduling-owner-server", Button).display
        assert app.query_one("#scheduling-owner-local", Button).display
        assert app.query_one("#scheduling-last-pull", Static).display
        assert not app.query_one("#scheduling-sync-local-note", Static).display


@pytest.mark.asyncio
async def test_collapsed_bar_still_shows_a_persisted_error_and_clear():
    """Review F11 (deliberate, not accidental): honesty beats compactness.
    A stale sync error persisted by a since-removed server must stay
    visible AND clearable on a collapsed local-only bar."""
    app = _BarHarness(
        current_owner="local", active_server_id=None, server_available=False
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(SyncStatusWidget)
        bar.update_status(None, None, [{"message": "old server: boom"}])
        await pilot.pause()

        # Collapsed: owner buttons/timestamps hidden, local note shown...
        assert not app.query_one("#scheduling-owner-server", Button).display
        assert app.query_one("#scheduling-sync-local-note", Static).display
        # ...but the error text and Clear stay visible and usable.
        error = app.query_one("#scheduling-sync-error", Static)
        assert error.display
        assert "boom" in str(error.render())
        assert app.query_one("#scheduling-clear-error", Button).display

        bar.update_status(None, None, [])
        await pilot.pause()
        assert not app.query_one("#scheduling-clear-error", Button).display


@pytest.mark.asyncio
async def test_clear_is_hidden_until_an_error_exists():
    app = _BarHarness(
        current_owner="local",
        active_server_id="http://127.0.0.1:8000",
        server_available=True,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(SyncStatusWidget)
        clear = app.query_one("#scheduling-clear-error", Button)

        bar.update_status(None, None, [])
        await pilot.pause()
        assert not clear.display

        bar.update_status(None, None, [{"message": "boom"}])
        await pilot.pause()
        assert clear.display

        bar.update_status(None, None, [])
        await pilot.pause()
        assert not clear.display


# --- redesign PR-2, Task 3: width-triggered compact path -------------------


@pytest.mark.asyncio
async def test_set_compact_hides_timestamps_but_keeps_owner_and_error_visible():
    """`set_compact` is additive and independent of `_apply_collapse`'s
    own owner/server-based collapse: with a live server (so the owner
    buttons render normally), compact hides only the last-pull/last-push
    timestamps -- the owner indicator (plan ruling 4's "(b)") and the
    error/Clear pair stay visible, same honesty-over-compactness carve-out
    `_apply_collapse` already documents."""
    app = _BarHarness(
        current_owner="local",
        active_server_id="http://127.0.0.1:8000",
        server_available=True,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(SyncStatusWidget)
        bar.update_status(None, None, [{"message": "boom"}])
        await pilot.pause()

        bar.set_compact(True)
        await pilot.pause()
        assert not app.query_one("#scheduling-last-pull", Static).display
        assert not app.query_one("#scheduling-last-push", Static).display
        assert app.query_one("#scheduling-owner-local", Button).display
        assert app.query_one("#scheduling-owner-server", Button).display
        assert app.query_one("#scheduling-sync-error", Static).display
        assert app.query_one("#scheduling-clear-error", Button).display

        bar.set_compact(False)
        await pilot.pause()
        assert app.query_one("#scheduling-last-pull", Static).display
        assert app.query_one("#scheduling-last-push", Static).display
