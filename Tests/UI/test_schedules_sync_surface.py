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


from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB,
    MockSchedulingServiceMixin,
    MockServerClient,
)


class _SyncService(MockSchedulingServiceMixin):
    """Service whose sync_now can either record a pull or do nothing."""

    server_client = MockServerClient(notifications_service=object())

    def __init__(self, records_pull: bool) -> None:
        self._records_pull = records_pull
        self.db = MockSchedulingDB()

    async def list_tasks(self):
        return [
            ReminderTask(
                id="task-1",
                title="Test",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            )
        ]

    async def sync_now(self, owner_id=None):
        if self._records_pull:
            self.db.update_sync_state(
                "local", last_pull_at="2026-08-28T12:00:00+00:00"
            )


class _App(ConsolidatedCSSApp):
    def __init__(self, service, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = service


async def _sync_via_action(pilot, workbench):
    workbench.action_sync_now()
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_sync_that_transfers_nothing_says_so():
    app = _App(_SyncService(records_pull=False))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        messages = [n.message for n in pilot.app._notifications]
        assert any("nothing was pulled or pushed" in m for m in messages), messages
        assert "Sync completed." not in messages, messages


@pytest.mark.asyncio
async def test_sync_that_records_a_pull_reports_completed():
    app = _App(_SyncService(records_pull=True))
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        await _sync_via_action(pilot, workbench)

        messages = [n.message for n in pilot.app._notifications]
        assert "Sync completed." in messages, messages
        # And the bar shows the recorded pull timestamp.
        pull = workbench.query_one("#scheduling-last-pull", Static)
        assert "2026-08-28" in str(pull.render())


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
