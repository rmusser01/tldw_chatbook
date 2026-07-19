"""Tests for the SchedulesWorkbench shell."""

from datetime import datetime, timezone

import pytest
from textual.app import App
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Input, Static

from tldw_chatbook.Scheduling.events import DeleteTaskRequested, SyncCompleted, SyncFailed
from tldw_chatbook.Scheduling.models import (
    ReminderTask,
    ScheduledTask,
    ScheduleKind,
    TaskStatus,
)
from tldw_chatbook.UI.Screens.scheduling.conflicts_tab import ConflictsTab
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
from tldw_chatbook.UI.Screens.scheduling.task_detail import (
    TaskDetail,
    TaskInspector,
    _STATUS_BADGE_CLASSES,
    _humanize_cron,
)
from tldw_chatbook.Widgets.delete_confirmation_dialog import DeleteConfirmationDialog


class WorkbenchTestApp(App):
    """Minimal test app that may not expose a real SchedulingService."""

    scheduling_service = None


class MockSchedulingService:
    """Stub service returning a single reminder task."""

    def __init__(self) -> None:
        self.updated: list[tuple[str, dict]] = []
        self.created: list[dict] = []
        self.deleted_ids: list[str] = []

    async def list_reminders(self):
        return [
            ReminderTask(
                id="task-1",
                title="Test",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime.now(timezone.utc),
                next_run_at=datetime.now(timezone.utc),
            )
        ]

    async def list_tasks(self):
        return await self.list_reminders()

    async def create_reminder(self, payload: dict):
        self.created.append(payload)
        return ReminderTask(**payload)

    async def update_reminder(self, task_id: str, fields: dict):
        self.updated.append((task_id, fields))
        reminders = await self.list_reminders()
        task = reminders[0]
        for key, value in fields.items():
            setattr(task, key, value)
        return task

    async def delete_reminder(self, task_id: str):
        self.deleted_ids.append(task_id)
        return True


class WorkbenchTestAppWithService(App):
    """Test app with a mock scheduling service."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = MockSchedulingService()


class MockSchedulingServiceWithWatchlist:
    """Stub service returning one reminder and one watchlist projection."""

    async def list_reminders(self):
        return [
            ReminderTask(
                id="task-1",
                title="Reminder",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc),
                next_run_at=datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc),
            )
        ]

    async def list_tasks(self):
        reminder_tasks = await self.list_reminders()
        return reminder_tasks + [
            ScheduledTask(
                id="watchlist:1",
                title="Watchlist Title",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                schedule_summary="Every 1h",
                next_run_at=datetime(2026, 7, 20, 11, 0, tzinfo=timezone.utc),
                owner_id="local",
            )
        ]


class WorkbenchTestAppWithMixedService(App):
    """Test app with a mixed reminder + watchlist scheduling service."""

    scheduling_service = MockSchedulingServiceWithWatchlist()


@pytest.mark.asyncio
async def test_schedules_workbench_renders_panes():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        assert isinstance(pilot.app.screen, SchedulesWorkbench)
        assert pilot.app.screen.query_one("#scheduling-workbench") is not None
        assert pilot.app.screen.query_one("#scheduling-list-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-detail-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-inspector-pane") is not None


@pytest.mark.asyncio
async def test_select_task_updates_detail():
    """Selecting a task row updates the detail pane with task information."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        title = detail.query_one("#scheduling-task-detail-title", Static)
        assert "Test" in title.visual.plain


@pytest.mark.asyncio
async def test_console_follow_selector_exists():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        assert pilot.app.screen.query_one("#schedules-follow-in-console") is not None


@pytest.mark.asyncio
async def test_task_detail_renders_selected_task():
    """The TaskDetail widget renders the selected reminder's metadata."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        title = detail.query_one("#scheduling-task-detail-title", Static)
        kind = detail.query_one("#scheduling-task-detail-type", Static)
        status_badge = detail.query_one("#scheduling-task-status-badge", Static)
        schedule = detail.query_one("#scheduling-task-detail-schedule", Static)
        next_run = detail.query_one("#scheduling-task-detail-next-run", Static)
        enable_button = detail.query_one("#scheduling-enable-task", Button)
        disable_button = detail.query_one("#scheduling-disable-task", Button)
        delete_button = detail.query_one("#scheduling-delete-task", Button)
        follow_button = detail.query_one("#schedules-follow-in-console", Button)

        assert "Test" in title.visual.plain
        assert "One-time" in kind.visual.plain
        assert "Waiting" in status_badge.visual.plain
        assert "One-time at" in schedule.visual.plain
        assert "UTC" in next_run.visual.plain
        assert enable_button.label.plain == "Enable"
        assert disable_button.label.plain == "Disable"
        assert delete_button.label.plain == "Delete"
        assert follow_button.label.plain == "Follow in Console"


@pytest.mark.asyncio
async def test_task_inspector_renders_metadata():
    """The TaskInspector widget shows sync, last-run, owner, and conflict text."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        inspector = pilot.app.screen.query_one(
            "#scheduling-task-inspector", TaskInspector
        )
        sync = inspector.query_one("#scheduling-inspector-sync", Static)
        last_run = inspector.query_one("#scheduling-inspector-last-run", Static)
        owner = inspector.query_one("#scheduling-inspector-owner", Static)
        conflict_card = inspector.query_one("#scheduling-conflict-card")
        conflict_text = inspector.query_one("#scheduling-conflict-text", Static)

        assert "version 0 (local)" in sync.visual.plain
        assert "Never run" in last_run.visual.plain
        assert "local" in owner.visual.plain
        assert "No conflict" in conflict_text.visual.plain
        assert "conflict" not in conflict_card.classes


class EmptyMockSchedulingService:
    """Stub service returning no reminder tasks."""

    async def list_reminders(self):
        return []

    async def list_tasks(self):
        return await self.list_reminders()


class DistinctMetadataMockSchedulingService:
    """Stub service returning a task with sync and last-run metadata."""

    async def list_reminders(self):
        return [
            ReminderTask(
                id="task-2",
                title="Synced Task",
                schedule_kind=ScheduleKind.RECURRING,
                cron="0 9 * * *",
                timezone="UTC",
                next_run_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
                last_run_at=datetime(2026, 7, 19, 9, 0, tzinfo=timezone.utc),
                server_id="srv-123",
                owner_id="user-1",
                sync_version=3,
            )
        ]

    async def list_tasks(self):
        return await self.list_reminders()


class FailingMockSchedulingService:
    """Stub service that raises on list_reminders."""

    async def list_reminders(self):
        raise RuntimeError("service unavailable")

    async def list_tasks(self):
        raise RuntimeError("service unavailable")


class WorkbenchTestAppWithEmptyService(App):
    """Test app with an empty scheduling service."""

    scheduling_service = EmptyMockSchedulingService()


class WorkbenchTestAppWithDistinctMetadata(App):
    """Test app with a scheduling service returning synced metadata."""

    scheduling_service = DistinctMetadataMockSchedulingService()


class WorkbenchTestAppWithFailingService(App):
    """Test app with a failing scheduling service."""

    scheduling_service = FailingMockSchedulingService()


@pytest.mark.asyncio
async def test_delete_button_opens_confirmation_dialog():
    """Clicking the Delete button opens the delete confirmation dialog."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        delete_button = pilot.app.screen.query_one("#scheduling-delete-task", Button)
        assert not delete_button.disabled
        assert delete_button.display
        delete_button.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(pilot.app.screen, DeleteConfirmationDialog)


@pytest.mark.asyncio
async def test_ctrl_d_opens_confirmation_dialog():
    """The Ctrl+D binding opens the delete confirmation dialog for the selected task."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        await pilot.press("ctrl+d")
        await pilot.pause()

        assert isinstance(pilot.app.screen, DeleteConfirmationDialog)


@pytest.mark.asyncio
async def test_empty_queue_shows_friendly_empty_state():
    """An empty queue shows the friendly empty-queue copy."""
    async with WorkbenchTestAppWithEmptyService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        empty_state = pilot.app.screen.query_one(
            "#scheduling-task-detail-empty-state", Static
        )
        assert "No scheduled tasks yet" in empty_state.visual.plain
        assert "Ctrl+C" in empty_state.visual.plain


@pytest.mark.asyncio
async def test_no_task_selected_shows_friendly_copy():
    """With no scheduling service, the detail pane prompts task selection."""
    async with WorkbenchTestApp().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        empty_state = pilot.app.screen.query_one(
            "#scheduling-task-detail-empty-state", Static
        )
        assert "Select a scheduled task" in empty_state.visual.plain
        assert "Ctrl+C" in empty_state.visual.plain


@pytest.mark.asyncio
async def test_status_badge_has_expected_class_for_waiting_task():
    """The status badge carries the CSS class matching the task status."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        badge = pilot.app.screen.query_one("#scheduling-task-status-badge", Static)
        assert "waiting" in badge.classes


@pytest.mark.asyncio
async def test_inspector_shows_distinct_metadata():
    """The inspector surfaces sync version, server id, last run, and owner."""
    async with WorkbenchTestAppWithDistinctMetadata().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        inspector = pilot.app.screen.query_one(
            "#scheduling-task-inspector", TaskInspector
        )
        sync = inspector.query_one("#scheduling-inspector-sync", Static)
        last_run = inspector.query_one("#scheduling-inspector-last-run", Static)
        owner = inspector.query_one("#scheduling-inspector-owner", Static)

        assert "version 3 (server srv-123)" in sync.visual.plain
        assert "2026-07-19 09:00 UTC" in last_run.visual.plain
        assert "user-1 / server srv-123" in owner.visual.plain


@pytest.mark.asyncio
async def test_conflict_card_shows_for_conflict_status():
    """The inspector conflict card renders when the task status is CONFLICT."""

    class ConflictMockSchedulingService:
        async def list_reminders(self):
            return [
                ReminderTask(
                    id="task-3",
                    title="Conflicted Task",
                    schedule_kind=ScheduleKind.ONE_TIME,
                    run_at=datetime.now(timezone.utc),
                    next_run_at=datetime.now(timezone.utc),
                    last_status=TaskStatus.CONFLICT,
                )
            ]

        async def list_tasks(self):
            return await self.list_reminders()

    class WorkbenchTestAppWithConflict(App):
        scheduling_service = ConflictMockSchedulingService()

    async with WorkbenchTestAppWithConflict().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        inspector = pilot.app.screen.query_one(
            "#scheduling-task-inspector", TaskInspector
        )
        conflict_card = inspector.query_one("#scheduling-conflict-card")
        conflict_text = inspector.query_one("#scheduling-conflict-text", Static)

        assert "conflict" in conflict_card.classes
        assert "Conflict detected" in conflict_text.visual.plain
        assert "Conflicted Task" in conflict_text.visual.plain


@pytest.mark.asyncio
async def test_follow_console_ignored_when_disabled():
    """Pressing the disabled Follow-in-Console button does nothing."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        follow_button = pilot.app.screen.query_one(
            "#schedules-follow-in-console", Button
        )
        follow_button.disabled = True
        # Directly invoke the handler as if a press event fired on the disabled button.
        pilot.app.screen.follow_latest_schedule_run_in_console(
            Button.Pressed(follow_button)
        )
        await pilot.pause()

        assert pilot.app.screen is not None
        assert not isinstance(pilot.app.screen, DeleteConfirmationDialog)


@pytest.mark.asyncio
async def test_load_tasks_service_error_notifies_and_uses_empty_state():
    """A service failure surfaces an error notification and consistent empty copy."""
    async with WorkbenchTestAppWithFailingService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        empty_state = pilot.app.screen.query_one(
            "#scheduling-task-detail-empty-state", Static
        )
        assert "No scheduled tasks yet" in empty_state.visual.plain


@pytest.mark.asyncio
async def test_workbench_renders_watchlist_job_row():
    """The workbench renders both reminders and watchlist projection rows."""
    async with WorkbenchTestAppWithMixedService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 2

        watchlist_row = table.get_row_at(1)
        assert "Watchlist Title" in str(watchlist_row[0])
        assert "Watchlist Job" in str(watchlist_row[1])
        assert "Waiting" in str(watchlist_row[2])
        assert "2026-07-20 11:00 UTC" in str(watchlist_row[3])


@pytest.mark.asyncio
async def test_select_watchlist_task_updates_detail():
    """Selecting a watchlist row updates the detail pane with projection metadata."""
    async with WorkbenchTestAppWithMixedService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (1, 0)
        pilot.app.screen._update_detail_for_index(1)
        await pilot.pause()

        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        type_label = detail.query_one("#scheduling-task-detail-type", Static)
        schedule = detail.query_one("#scheduling-task-detail-schedule", Static)

        assert "Watchlist Job" in type_label.visual.plain
        assert "Every 1h" in schedule.visual.plain


@pytest.mark.asyncio
async def test_inspector_shows_read_only_projection_for_watchlist():
    """The inspector surfaces the read-only projection state for watchlist jobs."""
    async with WorkbenchTestAppWithMixedService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (1, 0)
        pilot.app.screen._update_detail_for_index(1)
        await pilot.pause()

        inspector = pilot.app.screen.query_one(
            "#scheduling-task-inspector", TaskInspector
        )
        sync = inspector.query_one("#scheduling-inspector-sync", Static)
        last_run = inspector.query_one("#scheduling-inspector-last-run", Static)
        owner = inspector.query_one("#scheduling-inspector-owner", Static)

        assert "local (read-only projection)" in sync.visual.plain
        assert "-" == last_run.visual.plain.strip()
        assert "local" in owner.visual.plain


@pytest.mark.asyncio
async def test_watchlist_task_hides_lifecycle_actions():
    """Watchlist projections do not expose reminder lifecycle buttons."""
    async with WorkbenchTestAppWithMixedService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (1, 0)
        pilot.app.screen._update_detail_for_index(1)
        await pilot.pause()

        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        lifecycle = detail.query_one("#scheduling-task-detail-lifecycle", Horizontal)
        assert lifecycle.display is False


def test_humanize_cron_daily_pattern():
    """A standard daily cron pattern is summarized as 'Daily at HH:MM UTC'."""
    assert _humanize_cron("0 9 * * *") == "Daily at 09:00 UTC"
    assert (
        _humanize_cron("30 14 * * *", timezone="America/New_York")
        == "Daily at 14:30 America/New_York"
    )


def test_status_badge_classes_use_dedicated_css():
    """Each status maps to a dedicated CSS class so the TCSS can style it independently."""
    assert _STATUS_BADGE_CLASSES[TaskStatus.COMPLETED] == "completed"
    assert _STATUS_BADGE_CLASSES[TaskStatus.FOUND_RESULTS] == "found-results"
    assert _STATUS_BADGE_CLASSES[TaskStatus.ARCHIVED] == "archived"
    assert _STATUS_BADGE_CLASSES[TaskStatus.MISSED] == "missed"


class ToggleFailingMockSchedulingService:
    """Stub service that succeeds once, then fails on subsequent calls."""

    def __init__(self):
        self._calls = 0

    async def list_reminders(self):
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("service unavailable")
        return [
            ReminderTask(
                id="task-1",
                title="Morning digest",
                schedule_kind=ScheduleKind.RECURRING,
                cron="0 9 * * *",
                timezone="UTC",
                next_run_at=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
            )
        ]

    async def list_tasks(self):
        return await self.list_reminders()


class WorkbenchTestAppWithToggleFailingService(App):
    """Test app whose service succeeds on first load, then fails."""

    scheduling_service = ToggleFailingMockSchedulingService()


@pytest.mark.asyncio
async def test_load_tasks_service_error_clears_stale_rows():
    """A service failure after data was loaded clears the table and internal task list."""
    async with WorkbenchTestAppWithToggleFailingService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1

        await pilot.app.screen.load_tasks()
        await pilot.pause()

        assert table.row_count == 0
        empty_state = pilot.app.screen.query_one(
            "#scheduling-task-detail-empty-state", Static
        )
        assert "No scheduled tasks yet" in empty_state.visual.plain


class RecordingMockSchedulingService:
    """Stub service that records delete calls and their arguments."""

    def __init__(self, fail_delete: bool = False):
        self.deleted_ids: list[str] = []
        self.fail_delete = fail_delete
        self._deleted = False

    async def list_reminders(self):
        if self._deleted:
            return []
        return [
            ReminderTask(
                id="task-1",
                title="Test",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime.now(timezone.utc),
                next_run_at=datetime.now(timezone.utc),
            )
        ]

    async def list_tasks(self):
        return await self.list_reminders()

    async def delete_reminder(self, task_id: str) -> None:
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted_ids.append(task_id)
        self._deleted = True


class WorkbenchTestAppWithRecordingService(App):
    """Test app with a recording scheduling service."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduling_service = RecordingMockSchedulingService()


@pytest.mark.asyncio
async def test_delete_confirmation_runs_delete_requested_flow():
    """Confirming the delete dialog triggers the full DeleteTaskRequested flow."""
    app = WorkbenchTestAppWithRecordingService()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        detail.request_delete()
        await pilot.pause()

        assert isinstance(pilot.app.screen, DeleteConfirmationDialog)
        pilot.app.screen.dismiss(True)
        await pilot.pause()
        # Wait for the delete worker and the follow-up refresh.
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app.scheduling_service.deleted_ids == ["task-1"]


@pytest.mark.asyncio
async def test_workbench_deletes_task_and_notifies_on_success():
    """The workbench calls delete_reminder and surfaces a success notification."""
    app = WorkbenchTestAppWithRecordingService()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        task = pilot.app.screen._tasks[0]
        pilot.app.screen.post_message(DeleteTaskRequested(task))
        await pilot.pause()
        # Wait for the exclusive delete worker to finish and refresh.
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app.scheduling_service.deleted_ids == ["task-1"]
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_workbench_notifies_on_delete_failure():
    """The workbench surfaces an error notification when delete_reminder fails."""
    app = WorkbenchTestAppWithRecordingService()
    app.scheduling_service.fail_delete = True
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        task = pilot.app.screen._tasks[0]
        pilot.app.screen.post_message(DeleteTaskRequested(task))
        await pilot.pause()
        # Wait for the exclusive delete worker to finish.
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app.scheduling_service.deleted_ids == []
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_unimplemented_action_bindings_notify_user():
    """Stub action bindings notify the user instead of silently doing nothing."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        for action in (
            pilot.app.screen.action_run_now,
            pilot.app.screen.action_pause_resume,
            pilot.app.screen.action_sync_now,
        ):
            action()
            await pilot.pause()

        assert len(pilot.app._notifications) == 3
        for notification in pilot.app._notifications:
            assert notification.message == "Not yet available"
            assert notification.severity == "warning"


@pytest.mark.asyncio
async def test_enable_disable_buttons_update_reminder():
    """Enable/Disable buttons call the scheduling service and notify the user."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        enable_button = detail.query_one("#scheduling-enable-task", Button)
        disable_button = detail.query_one("#scheduling-disable-task", Button)

        detail.on_button_pressed(Button.Pressed(enable_button))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        detail.on_button_pressed(Button.Pressed(disable_button))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service = pilot.app.scheduling_service
        assert service.updated == [
            ("task-1", {"enabled": True}),
            ("task-1", {"enabled": False}),
        ]
        notifications = list(pilot.app._notifications)
        assert len(notifications) == 2
        assert notifications[0].severity == "information"
        assert notifications[1].severity == "information"


@pytest.mark.asyncio
async def test_create_reminder_action_saves_new_reminder():
    """Ctrl+C opens the reminder form; saving calls the scheduling service."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        pilot.app.scheduling_service = MockSchedulingService()
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        pilot.app.screen.action_create_reminder()
        await pilot.pause()

        assert isinstance(pilot.app.screen, ReminderForm)

        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "New reminder"
        run_at_input = pilot.app.screen.query_one("#reminder-run-at", Input)
        run_at_input.value = "2026-07-20T14:00:00+00:00"

        await pilot.click("#reminder-save")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service = pilot.app.scheduling_service
        assert len(service.created) == 1
        assert service.created[0]["title"] == "New reminder"
        assert service.created[0]["schedule_kind"] == "one_time"
        notifications = list(pilot.app._notifications)
        assert any(n.message == "Reminder created." for n in notifications)


@pytest.mark.asyncio
async def test_edit_reminder_action_updates_existing_reminder():
    """Clicking Edit opens the form pre-filled; saving calls update_reminder."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        pilot.app.scheduling_service = MockSchedulingService()
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
        edit_button = detail.query_one("#scheduling-edit-task", Button)
        detail.on_button_pressed(Button.Pressed(edit_button))
        await pilot.pause()

        assert isinstance(pilot.app.screen, ReminderForm)
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        assert title_input.value == "Test"
        title_input.value = "Updated title"

        await pilot.click("#reminder-save")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service = pilot.app.scheduling_service
        assert len(service.updated) == 1
        assert service.updated[0][0] == "task-1"
        assert service.updated[0][1]["title"] == "Updated title"
        notifications = list(pilot.app._notifications)
        assert any(n.message == "Reminder updated." for n in notifications)


def test_sync_completed_event():
    msg = SyncCompleted("server:1", conflict_count=2)
    assert msg.owner_id == "server:1"
    assert msg.conflict_count == 2


def test_sync_failed_event():
    msg = SyncFailed("server:1", error="timeout")
    assert msg.owner_id == "server:1"
    assert msg.error == "timeout"


@pytest.mark.asyncio
async def test_sync_status_widget_renders_mode_and_timestamps():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        widget = SyncStatusWidget(
            current_owner="server:example.com",
            server_available=True,
        )
        await pilot.app.mount(widget)
        await pilot.pause()

        local_btn = widget.query_one("#scheduling-owner-local", Button)
        server_btn = widget.query_one("#scheduling-owner-server", Button)
        assert local_btn.variant != "primary"
        assert server_btn.variant == "primary"

        widget.update_status(
            last_pull_at="2026-07-19T10:00:00+00:00",
            last_push_at="2026-07-19T10:05:00+00:00",
            sync_errors=[],
        )
        await pilot.pause()
        pull = widget.query_one("#scheduling-last-pull", Static)
        push = widget.query_one("#scheduling-last-push", Static)
        assert "Last pull" in pull.visual.plain
        assert "Last push" in push.visual.plain


@pytest.mark.asyncio
async def test_sync_status_widget_disables_server_button_when_unavailable():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        widget = SyncStatusWidget(
            current_owner="local",
            server_available=False,
        )
        await pilot.app.mount(widget)
        await pilot.pause()
        server_btn = widget.query_one("#scheduling-owner-server", Button)
        assert server_btn.disabled


@pytest.mark.asyncio
async def test_conflicts_tab_renders_rows_and_resolves():
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def resolve_conflict(self, conflict_id, resolution):
            self.calls.append((conflict_id, resolution))
            return True

    class CapturingConflictsTab(ConflictsTab):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.posted_messages: list[ConflictsTab.ConflictResolved] = []

        def post_message(self, message):
            if isinstance(message, ConflictsTab.ConflictResolved):
                self.posted_messages.append(message)
            return super().post_message(message)

    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        engine = FakeEngine()
        tab = CapturingConflictsTab(sync_engine=engine)
        await pilot.app.mount(tab)
        await pilot.pause()
        tab.populate([
            {
                "id": "c1",
                "local_id": "l1",
                "server_state": {},
                "local_state": {"record": {"title": "Local"}},
            },
        ])
        await pilot.pause()

        table = tab.query_one("#scheduling-conflicts-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.click("#scheduling-use-server")
        await pilot.pause()

        assert engine.calls == [("c1", "server")]
        assert len(tab.posted_messages) == 1
        msg = tab.posted_messages[0]
        assert msg.conflict_id == "c1"
        assert msg.resolution == "server"
        assert table.row_count == 0



@pytest.mark.asyncio
async def test_conflicts_tab_resolve_false_does_not_post_message():
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def resolve_conflict(self, conflict_id, resolution):
            self.calls.append((conflict_id, resolution))
            return False

    class CapturingConflictsTab(ConflictsTab):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.posted_messages: list[ConflictsTab.ConflictResolved] = []

        def post_message(self, message):
            if isinstance(message, ConflictsTab.ConflictResolved):
                self.posted_messages.append(message)
            return super().post_message(message)

    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        engine = FakeEngine()
        tab = CapturingConflictsTab(sync_engine=engine)
        await pilot.app.mount(tab)
        await pilot.pause()
        tab.populate([
            {
                "id": "c1",
                "local_id": "l1",
                "server_state": {},
                "local_state": {"record": {"title": "Local"}},
            },
        ])
        await pilot.pause()

        table = tab.query_one("#scheduling-conflicts-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.click("#scheduling-use-server")
        await pilot.pause()

        assert engine.calls == [("c1", "server")]
        assert len(tab.posted_messages) == 0
        assert table.row_count == 1
