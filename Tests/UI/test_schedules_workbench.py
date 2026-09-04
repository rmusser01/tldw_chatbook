"""Tests for the SchedulesWorkbench shell."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Select,
    Static,
    TabbedContent,
)
from textual.widgets._collapsible import CollapsibleTitle

from tldw_chatbook.Scheduling.events import (
    DeleteTaskRequested,
    SyncCompleted,
    SyncFailed,
)
from tldw_chatbook.Scheduling.models import (
    ReminderTask,
    ScheduledTask,
    ScheduleKind,
    TaskStatus,
)
from tldw_chatbook.UI.Screens.scheduling.conflicts_tab import ConflictsTab
from tldw_chatbook.UI.Screens.scheduling.definition_detail import (
    DefinitionDetail,
    _definition_owner_label,
)
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
from tldw_chatbook.UI.Screens.scheduling.task_detail import (
    TaskDetail,
    TaskInspector,
    _STATUS_BADGE_CLASSES,
    _humanize_cron,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.delete_confirmation_dialog import DeleteConfirmationDialog
from tldw_chatbook.Widgets.detail_value_row import DetailValueRow


# Shared across the Schedules UI test files (task-23106 review round F15).
from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB as _MockSchedulingDB,
    MockSchedulingServiceMixin as _MockSchedulingServiceMixin,
    MockServerClient as _MockServerClient,
    rendered_row_cells,
)


class WorkbenchTestApp(ConsolidatedCSSApp):
    """Minimal test app that may not expose a real SchedulingService."""

    scheduling_service = None


def test_static_content_gate_skips_equal_copy_but_updates_changed_copy() -> None:
    target = Static("unchanged")

    with patch.object(target, "update") as update:
        SchedulesWorkbench._update_static_content(target, "unchanged")
        update.assert_not_called()

        SchedulesWorkbench._update_static_content(target, "changed")
        update.assert_called_once_with("changed")


class MockSchedulingService(_MockSchedulingServiceMixin):
    """Stub service returning a single reminder task."""

    def __init__(self) -> None:
        super().__init__()
        self.updated: list[tuple[str, dict]] = []
        self.created: list[dict] = []
        self.deleted_ids: list[str] = []
        # Mirrors the real signature's threaded owner (never a `set_owner`
        # flip), so a cross-owner save is observable here.
        self.created_owners: list[str | None] = []
        self.updated_owners: list[str | None] = []
        self.deleted_owners: list[str | None] = []

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

    async def list_tasks(self, owner_id=None, include_projections=True):
        return await self.list_reminders()

    async def create_reminder(self, payload: dict, *, owner_id: str | None = None):
        self.created.append(payload)
        self.created_owners.append(owner_id)
        return ReminderTask(**payload)

    async def update_reminder(
        self, task_id: str, fields: dict, *, owner_id: str | None = None
    ):
        self.updated.append((task_id, fields))
        self.updated_owners.append(owner_id)
        reminders = await self.list_reminders()
        task = reminders[0]
        for key, value in fields.items():
            setattr(task, key, value)
        return task

    async def delete_reminder(self, task_id: str, *, owner_id: str | None = None):
        self.deleted_ids.append(task_id)
        self.deleted_owners.append(owner_id)
        return True


class WorkbenchTestAppWithService(ConsolidatedCSSApp):
    """Test app with a mock scheduling service."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = MockSchedulingService()


class MockSchedulingServiceWithWatchlist(_MockSchedulingServiceMixin):
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

    async def list_tasks(self, owner_id=None, include_projections=True):
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


class WorkbenchTestAppWithMixedService(ConsolidatedCSSApp):
    """Test app with a mixed reminder + watchlist scheduling service."""

    scheduling_service = MockSchedulingServiceWithWatchlist()


@pytest.mark.asyncio
async def test_schedules_workbench_renders_panes():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        assert isinstance(pilot.app.screen, SchedulesWorkbench)
        assert pilot.app.screen.query_one("#schedules-shell") is not None
        assert pilot.app.screen.query_one("#scheduling-workbench") is not None
        assert pilot.app.screen.query_one("#scheduling-list-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-detail-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-inspector-pane") is not None


@pytest.mark.asyncio
async def test_server_owner_requires_an_active_server_id():
    """A lazy server-service wrapper is not itself an available connection."""
    app = WorkbenchTestAppWithService()
    app.scheduling_service.server_client = _MockServerClient(
        notifications_service=object()
    )
    app.runtime_policy = SimpleNamespace(state=SimpleNamespace(active_server_id=None))

    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()

        server_button = workbench.query_one("#scheduling-owner-server", Button)
        assert server_button.disabled
        assert str(server_button.label) == "Server (unavailable)"
        assert (
            str(server_button.tooltip)
            == "Connect a scheduling server before switching Schedules ownership."
        )

        app.runtime_policy.state.active_server_id = "example.com"
        workbench._refresh_owner_select()
        await pilot.pause()

        assert not server_button.disabled
        assert str(server_button.label) == "Server (example.com)"
        assert (
            str(server_button.tooltip)
            == "Use the connected server as the Schedules owner."
        )


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
        status_badge = detail.query_one("#scheduling-task-status-badge", Static)
        next_run = detail.query_one("#scheduling-task-detail-next-run", Static)
        enable_button = detail.query_one("#scheduling-enable-task", Button)
        disable_button = detail.query_one("#scheduling-disable-task", Button)
        delete_button = detail.query_one("#scheduling-delete-task", Button)
        follow_button = detail.query_one("#schedules-follow-in-console", Button)

        assert "Test" in title.visual.plain
        assert "Waiting" in status_badge.visual.plain
        assert "UTC" in next_run.visual.plain
        assert enable_button.label.plain == "Enable"
        assert disable_button.label.plain == "Disable"
        assert delete_button.label.plain == "Delete"
        assert follow_button.label.plain == "Follow 'Test' in Console"

        # schedules-redesign PR-1, task 3: the old combined Type/Schedule
        # rows are hidden for a reminder now (they only remain visible for
        # a `ScheduledTask` projection); the same "One-time"/"One-time at"
        # facts render through the new Frequency group's Repeat/At rows
        # instead -- painted-output assertions, not stored-attribute ones
        # (last program's lesson), since a hidden widget's stored text
        # would pass even if nothing were actually on screen.
        legacy_fields = detail.query_one("#scheduling-task-detail-legacy-fields")
        groups = detail.query_one("#scheduling-task-detail-groups")
        assert legacy_fields.display is False
        assert groups.display is True
        repeat_value = detail.query_one("#scheduling-detail-repeat", Static)
        at_value = detail.query_one("#scheduling-detail-at", Static)
        assert "One-time" in repeat_value.render_line(0).text
        assert "One-time at" in at_value.render_line(0).text


class _BareTaskDetailApp(ConsolidatedCSSApp):
    """Bare app mounting one `TaskDetail` (schedules-redesign PR-1, task 3),
    matching `test_schedules_missed_notice.py`'s `_DetailHarnessApp`
    pattern. `CSS_PATH` is pinned to the app bundle (not just the screen
    sheets `ConsolidatedCSSApp` loads by default) so `DetailValueRow`/
    `DetailGroup`'s real `css/features/_scheduling.tcss` styling resolves
    -- Task 1's own harness precedent (`test_detail_value_row.py`).
    """

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self):
        yield TaskDetail()


def _frequency_reminder(**kwargs) -> ReminderTask:
    """A representative recurring reminder covering every §5 Frequency/
    Details/History value: weekly cadence, a non-UTC timezone, and a
    recorded last dispatch."""
    defaults = dict(
        id="task-freq",
        title="Weekly digest",
        schedule_kind=ScheduleKind.RECURRING,
        cron="0 9 * * 1",
        timezone="America/New_York",
        last_run_at=datetime(2026, 8, 24, 9, 0, tzinfo=timezone.utc),
        last_status=TaskStatus.COMPLETED,
    )
    defaults.update(kwargs)
    return ReminderTask(**defaults)


@pytest.mark.asyncio
async def test_task_detail_groups_render_every_frequency_and_details_value():
    """Every §5 reminder-column value paints through the new grouped rows
    for a representative recurring reminder (task-3 brief AC)."""
    async with _BareTaskDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder())
        await pilot.pause()

        def _text(widget_id: str) -> str:
            return detail.query_one(f"#{widget_id}", Static).render_line(0).text.strip()

        # Final review F6/F7: the prose owner vocabulary, shared with the
        # definitions pane (`owner_display_label`) -- not the raw metadata
        # string this row used to render.
        assert _text("scheduling-detail-runs-on") == "This device"
        assert _text("scheduling-detail-repeat") == "Recurring"
        assert _text("scheduling-detail-at") == "Weekly on Monday at 09:00 America/New_York"
        assert _text("scheduling-detail-timezone") == "America/New_York"
        assert _text("scheduling-detail-notifications") == "Inbox + toast"

        # The History group starts collapsed (spec §5); expand it to read
        # the painted "Last fire" value.
        history_group = detail.query_one("#scheduling-detail-group-history")
        assert history_group.collapsed is True
        history_group.collapsed = False
        await pilot.pause()
        assert _text("scheduling-detail-last-fire") == "2026-08-24 09:00 UTC — Completed"
        assert _text("scheduling-detail-history-link") == "See list below"


@pytest.mark.asyncio
async def test_task_detail_history_group_starts_collapsed_and_expands_on_click():
    """The collapsed History group hides its rows; a real click on its
    title expands it and repaints them (task-3 brief AC)."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder())
        await pilot.pause()

        history_group = detail.query_one("#scheduling-detail-group-history")
        last_fire_row = detail.query_one("#scheduling-detail-last-fire", Static)
        assert history_group.collapsed is True
        assert last_fire_row.region.height == 0

        await pilot.click(history_group.query_one(CollapsibleTitle))
        await pilot.pause()
        assert history_group.collapsed is False
        assert last_fire_row.region.height > 0
        assert "Completed" in last_fire_row.render_line(0).text

        await pilot.click(history_group.query_one(CollapsibleTitle))
        await pilot.pause()
        assert history_group.collapsed is True
        assert last_fire_row.region.height == 0


@pytest.mark.asyncio
async def test_task_detail_runs_on_shows_transfer_badge_when_in_flight():
    """'Runs on' appends the existing in-flight transfer badge text to the
    owner label, same wording as the queue row's transfer suffix
    (task-3 brief: 'the existing badge text joins the value')."""
    async with _BareTaskDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder(transfer_state="to_server_pending"))
        await pilot.pause()

        runs_on = detail.query_one("#scheduling-detail-runs-on", Static)
        assert (
            runs_on.render_line(0).text.strip()
            == "This device (Moving to server\u2026)"
        )


@pytest.mark.asyncio
async def test_task_detail_runs_on_speaks_the_definitions_pane_vocabulary():
    """Final review F6/F7: one owner vocabulary across BOTH detail panes.

    The reminder pane rendered the raw metadata string (`local`,
    `server:1 / server <id>`) while the definitions pane rendered
    `This device` / the bare server id -- two dialects for the flagship
    row of the redesign, and the User Guide described only the second.
    Both now go through `task_detail.owner_display_label`; this pins the
    reminder side against the definitions side's own helper.
    """
    async with _BareTaskDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)

        detail.set_task(
            _frequency_reminder(owner_id="server:srv-9", server_id="remote-42")
        )
        await pilot.pause()
        runs_on = detail.query_one("#scheduling-detail-runs-on", Static)
        assert runs_on.render_line(0).text.strip() == "srv-9"
        assert runs_on.render_line(0).text.strip() == _definition_owner_label(
            {"owner_id": "server:srv-9"}
        )

        detail.set_task(_frequency_reminder(owner_id="local"))
        await pilot.pause()
        assert runs_on.render_line(0).text.strip() == _definition_owner_label(
            {"owner_id": "local"}
        )


@pytest.mark.asyncio
async def test_task_detail_shows_the_reminder_body_card():
    """Final review F10: spec §5 wants the body text in a card above the
    groups (the definitions pane has had one all along); `ReminderTask.
    body` was rendered nowhere. Brackets stay literal, and a body-less
    reminder shows no empty card."""
    async with _BareTaskDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder(body="Ship the [bold] digest"))
        await pilot.pause()

        card = detail.query_one("#scheduling-task-detail-body-card", Static)
        assert card.display is True
        assert card.render_line(0).text.strip() == "Ship the [bold] digest"

        detail.set_task(_frequency_reminder(body=None))
        await pilot.pause()
        assert card.display is False


# --- redesign PR-3, task 3: reminder-pane in-pane Frequency-row editing ----


def _real_scheduling_service(tmp_path):
    """A real (in-memory-file) `ScheduledTasksDB` + `SchedulingService`,
    no server -- for tests that need genuine persistence/validation
    (Task 2's bridge), not a hand-rolled stub of it."""
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService

    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    service = SchedulingService(db=db, runtime_source="local")
    return db, service


@pytest.mark.asyncio
async def test_repeat_row_editor_opens_with_current_preset_preselected():
    """Activating a recurring reminder's Repeat row mounts a Select
    preloaded with the CURRENT cron's preset (task-3 brief AC)."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        # `_frequency_reminder`'s default cron "0 9 * * 1" == Monday 09:00.
        detail.set_task(_frequency_reminder())
        await pilot.pause()

        repeat_row = detail._repeat_row
        assert repeat_row.affordance is True
        await pilot.click(repeat_row)
        await pilot.pause()

        editor = repeat_row.query_one(Select)
        assert editor.value == "monday"


@pytest.mark.asyncio
async def test_at_row_editor_opens_with_current_run_at_preselected():
    """A one-time reminder's At row opens an Input preloaded with the
    task's own `run_at.isoformat()` -- the same prefill shape the
    create/edit modal uses."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        run_at = datetime(2030, 1, 1, 9, 0, tzinfo=timezone.utc)
        detail.set_task(
            _frequency_reminder(
                schedule_kind=ScheduleKind.ONE_TIME,
                cron=None,
                timezone=None,
                run_at=run_at,
            )
        )
        await pilot.pause()

        at_row = detail._at_row
        assert at_row.affordance is True
        await pilot.click(at_row)
        await pilot.pause()

        editor = at_row.query_one(Input)
        assert editor.value == run_at.isoformat()


@pytest.mark.asyncio
async def test_timezone_row_editor_opens_with_current_zone_preselected():
    """A recurring reminder's Timezone row opens a Select preloaded with
    the task's OWN stored zone."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder())  # timezone="America/New_York"
        await pilot.pause()

        tz_row = detail._timezone_row
        assert tz_row.affordance is True
        await pilot.click(tz_row)
        await pilot.pause()

        editor = tz_row.query_one(Select)
        assert editor.value == "America/New_York"


@pytest.mark.asyncio
async def test_frequency_row_affordance_matches_schedule_kind():
    """Repeat/Timezone only apply to a recurring schedule, At only to a
    one-time one (survey §2: the other combination is silently clobbered
    by `update_reminder`'s own recompute step) -- the non-applicable row
    stays read-only instead of offering a guaranteed-to-fail control."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder())  # recurring
        await pilot.pause()
        assert detail._repeat_row.affordance is True
        assert detail._at_row.affordance is False
        assert detail._timezone_row.affordance is True

        detail.set_task(
            _frequency_reminder(
                schedule_kind=ScheduleKind.ONE_TIME,
                cron=None,
                timezone=None,
                run_at=datetime(2030, 1, 1, 9, 0, tzinfo=timezone.utc),
            )
        )
        await pilot.pause()
        assert detail._repeat_row.affordance is False
        assert detail._at_row.affordance is True
        assert detail._timezone_row.affordance is False


@pytest.mark.asyncio
async def test_escape_cancels_open_editor_without_committing():
    """Escape closes the open editor via `end_edit` -- no field-edit
    request is posted, and the row's old value is still shown."""

    class _CapturingApp(_BareTaskDetailApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.requests: list = []

        def on_reminder_field_edit_requested(self, event) -> None:
            self.requests.append(event)

    async with _CapturingApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder())
        await pilot.pause()

        repeat_row = detail._repeat_row
        await pilot.click(repeat_row)
        await pilot.pause()
        assert repeat_row.query(Select)

        await pilot.press("escape")
        await pilot.pause()

        assert not repeat_row.query(Select)
        assert pilot.app.requests == []
        assert (
            detail.query_one("#scheduling-detail-repeat", Static)
            .render_line(0)
            .text.strip()
            == "Recurring"
        )


@pytest.mark.asyncio
async def test_repeat_row_selecting_custom_target_refuses_without_a_bridge_call():
    """Picking "Custom cron..." as a NEW Repeat target has no single-value
    edit shape here (the raw cron field only exists in the full modal),
    so it is refused client-side rather than sent to the bridge (ruling
    2: never silent) -- distinct from "custom" merely being the row's
    CURRENT (round-tripped) value, which never triggers a commit at all."""

    class _CapturingApp(_BareTaskDetailApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.requests: list = []

        def on_reminder_field_edit_requested(self, event) -> None:
            self.requests.append(event)

    async with _CapturingApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        # `_frequency_reminder`'s default cron "0 9 * * 1" == "monday".
        detail.set_task(_frequency_reminder())
        await pilot.pause()

        repeat_row = detail._repeat_row
        await pilot.click(repeat_row)
        await pilot.pause()
        editor = repeat_row.query_one(Select)
        assert editor.value == "monday"

        editor.value = "custom"  # a genuine change to an unsupported target
        await pilot.pause()

        assert not repeat_row.query(Select)  # editor closed
        assert pilot.app.requests == []  # never reached the bridge
        error = repeat_row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert "full Edit form" in error.render_line(0).text
        # The stored cron is untouched -- nothing was persisted.
        assert (
            detail.query_one("#scheduling-detail-at", Static)
            .render_line(0)
            .text.strip()
            == "Weekly on Monday at 09:00 America/New_York"
        )


@pytest.mark.asyncio
async def test_locked_row_activation_shows_lock_reason_and_opens_no_editor():
    """Ruling 2 (never silent): a locked row's Frequency rows keep their
    affordance ON so activation still responds -- with the lock reason
    via `show_error`, never an editor."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder())
        detail.set_lifecycle_lock(
            "This row is moving between this device and the server -- it "
            "is read-only until the move finishes. Cancel the transfer first."
        )
        await pilot.pause()

        repeat_row = detail._repeat_row
        assert repeat_row.affordance is True  # still responsive, not silently off
        await pilot.click(repeat_row)
        await pilot.pause()

        assert not repeat_row.query(Select)
        error = repeat_row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert "moving between this device and the server" in error.render_line(0).text


@pytest.mark.asyncio
async def test_committing_repeat_edit_persists_and_repaints_pane_and_queue_list(
    tmp_path,
):
    """Commit persists via Task 2's bridge; success repaints the pane from
    a fresh read AND the unified Queue list's own row data (task-3 brief
    AC: 'unified-list row updates after edit')."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        reminder = await service.create_reminder(
            {
                "title": "Weekly digest",
                "schedule_kind": "recurring",
                "run_at": None,
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            }
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()

            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            at_before = detail.query_one("#scheduling-detail-at", Static)
            assert (
                at_before.render_line(0).text.strip()
                == "Weekly on Monday at 09:00 America/New_York"
            )

            # `pilot.click` targets a screen coordinate computed from
            # `Widget.region`, which the Frequency group's real position
            # inside the full 3-pane workbench does not reliably match
            # (Task 1's own click-mechanics tests are BARE-harness only,
            # never embedded here) -- posting `Activated` directly drives
            # the exact same handler a real click reaches, without
            # depending on pixel-perfect layout in a pane this narrow.
            detail._repeat_row.post_message(
                DetailValueRow.Activated(detail._repeat_row)
            )
            await pilot.pause()
            select = detail._repeat_row.query_one(Select)
            select.value = "daily"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_reminder_task(reminder.id)["cron"] == "0 9 * * *"

            at_after = detail.query_one("#scheduling-detail-at", Static)
            assert (
                at_after.render_line(0).text.strip()
                == "Daily at 09:00 America/New_York"
            )

            # The unified Queue list's own underlying row data, not just
            # the detail pane, reflects the persisted edit -- proven via
            # the workbench's own refreshed `_tasks` (deterministic; the
            # rendered relative-time subtitle is wall-clock-dependent and
            # not a safe assertion here).
            workbench = pilot.app.screen
            assert workbench._tasks[0].cron == "0 9 * * *"
            assert rendered_row_cells(table, 0)[1] == "Weekly digest"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_edit_on_a_server_owned_row_persists_under_its_own_owner(
    tmp_path,
):
    """PR-2's Queue spans owners -- a server-owned row's edit must persist
    (row-owner threading is the bridge's own job, Task 2) without ever
    repointing the service's active ('local') owner (task-3 brief AC)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        task_id = db.create_reminder_task(
            owner_id="server:example.com",
            title="Server reminder",
            schedule_kind="recurring",
            run_at=None,
            cron="0 9 * * 1",
            timezone="America/New_York",
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()

            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            detail._timezone_row.post_message(
                DetailValueRow.Activated(detail._timezone_row)
            )
            await pilot.pause()
            select = detail._timezone_row.query_one(Select)
            select.value = "UTC"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            row = db.get_reminder_task(task_id)
            assert row["timezone"] == "UTC"
            assert row["owner_id"] == "server:example.com"
        assert service.owner_id == "local"  # never repointed by the edit
    finally:
        db.close()


@pytest.mark.asyncio
async def test_junk_at_value_shows_inline_error_and_restores_old_display(tmp_path):
    """A junk At submission surfaces the bridge's field error inline and
    leaves the OLD display + DB value untouched (task-3 brief AC)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="One-off",
            schedule_kind="one_time",
            run_at="2030-01-01T09:00:00+00:00",
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()

            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            at_row = detail._at_row
            at_static = detail.query_one("#scheduling-detail-at", Static)
            original_text = at_static.render_line(0).text.strip()

            at_row.post_message(DetailValueRow.Activated(at_row))
            await pilot.pause()
            input_widget = at_row.query_one(Input)
            input_widget.value = "not-a-date"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            error = at_row.query_one(".detail-value-row-error", Static)
            assert error.display is True
            assert "Run At must be a date and time" in error.render_line(0).text

            assert (
                db.get_reminder_task(task_id)["run_at"]
                == "2030-01-01T09:00:00+00:00"
            )
            assert at_static.render_line(0).text.strip() == original_text
    finally:
        db.close()


@pytest.mark.asyncio
async def test_locked_row_via_real_service_refuses_edit_and_leaves_row_unchanged(
    tmp_path,
):
    """The lock guard holds end-to-end through the real bridge too: a
    locked row's Repeat editor never opens, and the DB row is untouched."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="Locked",
            schedule_kind="recurring",
            run_at=None,
            cron="0 9 * * 1",
            timezone="America/New_York",
        )
        db.set_transfer_state(
            "reminder_task", task_id, "to_server_pending", expected=(None,)
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()

            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            repeat_row = detail._repeat_row
            assert repeat_row.affordance is True

            repeat_row.post_message(DetailValueRow.Activated(repeat_row))
            await pilot.pause()

            assert not repeat_row.query(Select)
            error = repeat_row.query_one(".detail-value-row-error", Static)
            assert error.display is True
            assert "moving between this device and the server" in error.render_line(0).text
            assert db.get_reminder_task(task_id)["cron"] == "0 9 * * 1"
    finally:
        db.close()


# --- redesign PR-3, task 4: definition-pane in-pane editing + lifecycle ----


class _BareDefinitionDetailApp(ConsolidatedCSSApp):
    """Bare app mounting one `DefinitionDetail` (schedules-redesign PR-3,
    task 4), matching `_BareTaskDetailApp`'s own pattern."""

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self):
        yield DefinitionDetail()


def _editable_definition(**overrides) -> dict:
    """A representative local `recurring_question` definition dict
    covering every task-4 editable Details/Frequency row: a recurring
    cron schedule, a pinned model, a non-default generation mode, an
    explicit sources scope, and an on notification policy."""
    base: dict = {
        "id": "def-1",
        "owner_id": "local",
        "family": "recurring_question",
        "name": "Daily standup question",
        "lifecycle": "configured",
        "schedule": {
            "kind": "cron",
            "cron": "0 9 * * 1",
            "timezone": "America/New_York",
        },
        "input": {
            "question": "What shipped?",
            "provider": "openai",
            "model": "gpt-5",
        },
        "config": {
            "generation_mode": "required",
            "scope": {"mode": "sources", "sources": ["media_db", "notes"]},
            "finding_policy": {"preset": "high_confidence_only"},
        },
        "finding_policy": {"preset": "high_confidence_only"},
        "notification_policy": {"on_success": True, "on_failure": True},
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_definition_details_rows_are_always_editable_regardless_of_schedule_kind():
    """Model/Generation/Finding policy/Sources/Notifications don't depend
    on schedule kind (unlike Repeat/At/Timezone) -- affordance stays on
    for both kinds."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        await pilot.pause()
        for row in (
            detail._model_row,
            detail._generation_row,
            detail._finding_policy_row,
            detail._sources_row,
            detail._notifications_row,
        ):
            assert row.affordance is True

        detail.set_definition(
            _editable_definition(
                schedule={"kind": "one_time", "run_at": "2030-01-01T09:00:00+00:00"}
            )
        )
        await pilot.pause()
        for row in (
            detail._model_row,
            detail._generation_row,
            detail._finding_policy_row,
            detail._sources_row,
            detail._notifications_row,
        ):
            assert row.affordance is True


@pytest.mark.asyncio
async def test_definition_frequency_row_affordance_gated_by_schedule_kind():
    """Repeat/Timezone only apply to a cron schedule, At only to a
    one_time one -- same reasoning `task_detail.py`'s reminder Frequency
    rows already use."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # cron
        await pilot.pause()
        assert detail._repeat_row.affordance is True
        assert detail._at_row.affordance is False
        assert detail._timezone_row.affordance is True

        detail.set_definition(
            _editable_definition(
                schedule={"kind": "one_time", "run_at": "2030-01-01T09:00:00+00:00"}
            )
        )
        await pilot.pause()
        assert detail._repeat_row.affordance is False
        assert detail._at_row.affordance is True
        assert detail._timezone_row.affordance is False


@pytest.mark.asyncio
async def test_model_row_editor_preselects_provider_slash_model_and_is_blank_when_not_set():
    """Task-4 brief: 'blank = provider default, the "Not set" honesty
    preserved' -- the Input opens blank, not literal 'auto'."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        await pilot.pause()

        model_row = detail._model_row
        await pilot.click(model_row)
        await pilot.pause()
        editor = model_row.query_one(Input)
        assert editor.value == "openai/gpt-5"
        await pilot.press("escape")
        await pilot.pause()

        detail.set_definition(_editable_definition(input={"question": "What shipped?"}))
        await pilot.pause()
        model_row = detail._model_row
        await pilot.click(model_row)
        await pilot.pause()
        editor = model_row.query_one(Input)
        assert editor.value == ""


@pytest.mark.asyncio
async def test_generation_row_editor_preselects_current_value_defaulting_to_optional():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # generation_mode="required"
        await pilot.pause()
        row = detail._generation_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "required"
        await pilot.press("escape")
        await pilot.pause()

        detail.set_definition(
            _editable_definition(config={"scope": {"mode": "all_searchable_library"}})
        )
        await pilot.pause()
        row = detail._generation_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "optional"


@pytest.mark.asyncio
async def test_finding_policy_row_editor_preselects_current_value():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # high_confidence_only
        await pilot.pause()
        row = detail._finding_policy_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "high_confidence_only"


@pytest.mark.asyncio
async def test_sources_row_editor_checks_stored_sources_and_all_when_library_wide():
    """The 3-checkbox mini-editor (task-4 brief's own suggested Sources
    shape) prechecks the stored subset, or every box when the scope is
    library-wide/unset -- visually 'everything', matching the row's own
    displayed value."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # sources: media_db, notes
        await pilot.pause()
        row = detail._sources_row
        await pilot.click(row)
        await pilot.pause()
        checkboxes = {cb.id: cb.value for cb in row.query(Checkbox)}
        assert checkboxes["scheduling-automation-detail-sources-media_db"] is True
        assert checkboxes["scheduling-automation-detail-sources-notes"] is True
        assert checkboxes["scheduling-automation-detail-sources-chats"] is False
        await pilot.press("escape")
        await pilot.pause()

        detail.set_definition(
            _editable_definition(config={"scope": {"mode": "all_searchable_library"}})
        )
        await pilot.pause()
        row = detail._sources_row
        await pilot.click(row)
        await pilot.pause()
        checkboxes = {cb.id: cb.value for cb in row.query(Checkbox)}
        assert all(checkboxes.values())


@pytest.mark.asyncio
async def test_notifications_row_editor_preselects_on_off():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # on
        await pilot.pause()
        row = detail._notifications_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "on"
        await pilot.press("escape")
        await pilot.pause()

        detail.set_definition(
            _editable_definition(
                notification_policy={"on_success": False, "on_failure": False}
            )
        )
        await pilot.pause()
        row = detail._notifications_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "off"


@pytest.mark.asyncio
async def test_definition_repeat_row_editor_preselects_current_preset():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # "0 9 * * 1" == monday
        await pilot.pause()
        row = detail._repeat_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "monday"


@pytest.mark.asyncio
async def test_definition_at_row_editor_preselects_current_run_at():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(
            _editable_definition(
                schedule={"kind": "one_time", "run_at": "2030-01-01T09:00:00+00:00"}
            )
        )
        await pilot.pause()
        row = detail._at_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Input)
        assert editor.value == "2030-01-01T09:00:00+00:00"


@pytest.mark.asyncio
async def test_definition_timezone_row_editor_preselects_current_zone():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # America/New_York
        await pilot.pause()
        row = detail._timezone_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "America/New_York"


@pytest.mark.asyncio
async def test_definition_escape_cancels_open_editor_without_committing():
    class _CapturingApp(_BareDefinitionDetailApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.requests: list = []

        def on_definition_field_edit_requested(self, event) -> None:
            self.requests.append(event)

    async with _CapturingApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        await pilot.pause()

        row = detail._generation_row
        await pilot.click(row)
        await pilot.pause()
        assert row.query(Select)

        await pilot.press("escape")
        await pilot.pause()

        assert not row.query(Select)
        assert pilot.app.requests == []


@pytest.mark.asyncio
async def test_definition_repeat_row_custom_target_refuses_client_side_without_bridge_call():
    class _CapturingApp(_BareDefinitionDetailApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.requests: list = []

        def on_definition_field_edit_requested(self, event) -> None:
            self.requests.append(event)

    async with _CapturingApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        await pilot.pause()

        row = detail._repeat_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        editor.value = "custom"
        await pilot.pause()

        assert not row.query(Select)
        assert pilot.app.requests == []
        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert "full Edit form" in error.render_line(0).text


@pytest.mark.asyncio
async def test_sources_editor_apply_with_nothing_checked_refuses_client_side():
    """Sources editor honesty (task-4 brief): unchecking every box and
    applying is refused inline rather than sent to the bridge -- an
    empty scope is a guaranteed `scope_empty` server refusal anyway."""

    class _CapturingApp(_BareDefinitionDetailApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.requests: list = []

        def on_definition_field_edit_requested(self, event) -> None:
            self.requests.append(event)

    async with _CapturingApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        await pilot.pause()

        row = detail._sources_row
        await pilot.click(row)
        await pilot.pause()
        for checkbox in row.query(Checkbox):
            checkbox.value = False
        await pilot.pause()
        apply_button = row.query_one(
            "#scheduling-automation-detail-sources-apply", Button
        )
        await pilot.click(apply_button)
        await pilot.pause()

        assert not row.query(Checkbox)  # editor closed
        assert pilot.app.requests == []
        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert "at least one source" in error.render_line(0).text


@pytest.mark.asyncio
async def test_definition_locked_row_activation_shows_lock_reason_and_opens_no_editor():
    """Survey point 10: `DefinitionDetail` gains the SAME transfer-lock
    wiring `TaskDetail` has -- a locked row keeps its affordance ON so
    activation still responds, with the lock reason, never an editor."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        detail.set_lifecycle_lock(
            "This row is moving between this device and the server -- it "
            "is read-only until the move finishes. Cancel the transfer first."
        )
        await pilot.pause()

        row = detail._generation_row
        assert row.affordance is True  # still responsive, not silently off
        await pilot.click(row)
        await pilot.pause()

        assert not row.query(Select)
        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert "moving between this device and the server" in error.render_line(0).text


class _LifecycleCapturingApp(_BareDefinitionDetailApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list = []

    def on_definition_lifecycle_toggle_requested(self, event) -> None:
        self.requests.append(event)


@pytest.mark.asyncio
async def test_pause_resume_button_shows_pause_and_posts_pause_when_configured():
    async with _LifecycleCapturingApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition(lifecycle="configured"))
        await pilot.pause()
        button = detail.query_one("#scheduling-automation-pause-resume", Button)
        assert button.label.plain == "Pause"
        await pilot.click(button)
        await pilot.pause()
        assert len(pilot.app.requests) == 1
        assert pilot.app.requests[0].action == "pause"


@pytest.mark.asyncio
async def test_pause_resume_button_shows_resume_and_posts_resume_when_paused():
    async with _LifecycleCapturingApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition(lifecycle="paused"))
        await pilot.pause()
        button = detail.query_one("#scheduling-automation-pause-resume", Button)
        assert button.label.plain == "Resume"
        await pilot.click(button)
        await pilot.pause()
        assert len(pilot.app.requests) == 1
        assert pilot.app.requests[0].action == "resume"


@pytest.mark.asyncio
async def test_pause_resume_button_disabled_and_reason_shown_when_locked():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        detail.set_lifecycle_lock("Read-only mid-transfer.")
        await pilot.pause()
        button = detail.query_one("#scheduling-automation-pause-resume", Button)
        assert button.disabled is True
        assert button.tooltip == "Read-only mid-transfer."
        why = detail.query_one("#scheduling-automation-detail-why", Static)
        assert why.render_line(0).text.strip() == "Read-only mid-transfer."

        detail.set_lifecycle_lock(None)
        await pilot.pause()
        assert button.disabled is False


# --- integration: real DB + service + full workbench -----------------------


@pytest.mark.asyncio
async def test_committing_generation_edit_persists_and_repaints_automations_pane(
    tmp_path,
):
    """Commit persists via `save_definition`; success repaints the
    Automations-tab pane from a fresh read AND arms the unified Queue
    list's own (lazy, tab-activation-gated) refresh -- the same
    `_definitions_stale` seam every other definition mutation in this
    file uses (run-now, transfer begin/cancel)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily standup question",
            schedule={
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            },
            input={"question": "What shipped?"},
            config={
                "generation_mode": "optional",
                "scope": {"mode": "all_searchable_library"},
            },
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()

            table = workbench.query_one("#scheduling-automations-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._generation_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "required"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert (
                db.get_automation_definition(definition_id)["config"][
                    "generation_mode"
                ]
                == "required"
            )
            generation_static = detail.query_one(
                "#scheduling-automation-detail-generation", Static
            )
            assert (
                generation_static.render_line(0).text.strip()
                == "Always generate a draft"
            )
            # The unified Queue list's own next (lazy) refresh is armed.
            assert workbench._definitions_stale is True
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_repeat_edit_resends_whole_schedule_preserving_timezone(
    tmp_path,
):
    """Pinned (task-4 brief): schedule edits RESEND THE WHOLE schedule
    dict -- an edited Repeat (cron) must not drop the recurring
    schedule's own timezone."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Weekly digest",
            schedule={
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            },
            input={"question": "What shipped?"},
            config={},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()
            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._repeat_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "daily"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["schedule"]["cron"] == "0 9 * * *"
            assert stored["schedule"]["kind"] == "cron"
            assert stored["schedule"]["timezone"] == "America/New_York"  # not dropped
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_timezone_edit_preserves_cron(tmp_path):
    """Companion pin: a Timezone edit must not drop the schedule's cron."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Weekly digest",
            schedule={
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            },
            input={"question": "What shipped?"},
            config={},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()
            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._timezone_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "UTC"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["schedule"]["timezone"] == "UTC"
            assert stored["schedule"]["cron"] == "0 9 * * 1"  # not dropped
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_model_edit_persists_and_repaints_automations_pane(
    tmp_path,
):
    """Model row commit persists `input.provider`/`model` and repaints the
    pane (task-4 review Finding 2: this row type had no real-DB persist
    test in the original diff)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily standup question",
            schedule={
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            },
            input={"question": "What shipped?"},
            config={"generation_mode": "optional"},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()

            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._model_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            input_widget = row.query_one(Input)
            input_widget.value = "openai/gpt-5"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["input"]["provider"] == "openai"
            assert stored["input"]["model"] == "gpt-5"
            assert stored["input"]["question"] == "What shipped?"  # not dropped
            model_static = detail.query_one(
                "#scheduling-automation-detail-model", Static
            )
            assert model_static.render_line(0).text.strip() == "openai/gpt-5"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_finding_policy_edit_persists_and_repaints_automations_pane(
    tmp_path,
):
    """Finding-policy row commit writes `config.finding_policy.preset`
    AND persists it back to the TOP-LEVEL `finding_policy` column this
    row reads for display (task-4 review Finding 2)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily standup question",
            schedule={
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            },
            input={"question": "What shipped?"},
            config={"finding_policy": {"preset": "balanced_findings"}},
            finding_policy={"preset": "balanced_findings"},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()

            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._finding_policy_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "high_confidence_only"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert (
                stored["config"]["finding_policy"]["preset"]
                == "high_confidence_only"
            )
            assert stored["finding_policy"]["preset"] == "high_confidence_only"
            finding_static = detail.query_one(
                "#scheduling-automation-detail-finding-policy", Static
            )
            assert (
                finding_static.render_line(0).text.strip()
                == "High confidence only"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_sources_edit_persists_and_repaints_automations_pane(
    tmp_path,
):
    """Sources editor Apply persists the explicit `{"mode": "sources", ...}`
    shape and repaints the pane (task-4 review Finding 2)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily standup question",
            schedule={
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            },
            input={"question": "What shipped?"},
            config={"scope": {"mode": "all_searchable_library"}},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()

            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._sources_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            for checkbox in row.query(Checkbox):
                checkbox.value = (
                    checkbox.id == "scheduling-automation-detail-sources-notes"
                )
            await pilot.pause()
            apply_button = row.query_one(
                "#scheduling-automation-detail-sources-apply", Button
            )
            # A real pixel click on a widget nested this deep inside the
            # full 3-pane embedded workbench is exactly the layout quirk
            # Task 3's own report documents (bare-harness clicks land
            # fine, embedded ones do not reliably match `Widget.region`)
            # -- post the Pressed message directly, same workaround this
            # file already uses for row Activation.
            apply_button.post_message(Button.Pressed(apply_button))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["config"]["scope"] == {
                "mode": "sources",
                "sources": ["notes"],
            }
            sources_static = detail.query_one(
                "#scheduling-automation-detail-sources", Static
            )
            assert sources_static.render_line(0).text.strip() == "Notes"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_notifications_edit_persists_and_repaints_automations_pane(
    tmp_path,
):
    """Notifications row commit writes the boolean on/off shape for BOTH
    outcomes and repaints the pane (task-4 review Finding 2)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily standup question",
            schedule={
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "America/New_York",
            },
            input={"question": "What shipped?"},
            config={},
            notification_policy={"on_success": False, "on_failure": False},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()

            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._notifications_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "on"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["notification_policy"] == {
                "on_success": True,
                "on_failure": True,
            }
            notif_static = detail.query_one(
                "#scheduling-automation-detail-notifications", Static
            )
            assert notif_static.render_line(0).text.strip() == "On"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_committing_at_edit_persists_and_repaints_automations_pane(tmp_path):
    """At row commit persists `run_at` on a one-time definition and
    repaints the pane (task-4 review Finding 2)."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "One-off digest",
            schedule={"kind": "one_time", "run_at": "2030-01-01T09:00:00+00:00"},
            input={"question": "What shipped?"},
            config={},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()

            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            row = detail._at_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            new_run_at = datetime(2031, 6, 15, 9, 0, tzinfo=timezone.utc)
            input_widget = row.query_one(Input)
            input_widget.value = new_run_at.isoformat()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["schedule"]["kind"] == "one_time"
            assert stored["schedule"]["run_at"] == new_run_at.isoformat()
            at_static = detail.query_one(
                "#scheduling-automation-detail-at", Static
            )
            assert (
                at_static.render_line(0).text.strip()
                == "One-time at 2031-06-15 09:00 UTC"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_not_set_model_preserved_across_an_unrelated_edit(tmp_path):
    """The Model row's 'Not set'/blank honesty (task-4 brief AC) survives
    an edit to an UNRELATED row: `_definition_edit_payload`'s empty-dict
    `input`/`notification_policy` groups round-trip through `save_
    definition`'s one-level merge untouched.

    Scoped to `input`/`notification_policy` deliberately, NOT `config`'s
    own generation_mode/scope/finding_policy trio: traced (empirically,
    via `preview_automation_definition` -> `validate_recurring_question_
    config`) that those three are unconditionally NORMALIZED WITH
    DEFAULTS on every local save regardless of which row was actually
    edited -- a PRE-EXISTING behavior of the shared preview pipeline the
    create/edit modal never exercises (it always sends explicit values),
    now newly reachable because task 4 is the first caller that can send
    a payload leaving them genuinely absent. Not a task-4 regression and
    out of this task's scope to change; flagged in the report instead."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Bare-model digest",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What shipped?"},  # no provider/model
            config={"generation_mode": "optional"},
            notification_policy={"on_success": True, "on_failure": False},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()
            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            model_static = detail.query_one(
                "#scheduling-automation-detail-model", Static
            )
            assert model_static.render_line(0).text.strip() == "auto"

            row = detail._generation_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "required"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["config"]["generation_mode"] == "required"
            assert "provider" not in stored["input"]
            assert "model" not in stored["input"]
            assert stored["notification_policy"] == {
                "on_success": True,
                "on_failure": False,
            }
            assert model_static.render_line(0).text.strip() == "auto"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_edit_on_server_owned_row_with_unreachable_seam_queues_a_mutation(
    tmp_path,
):
    """A server-owned row whose seam is unreachable still writes locally
    and queues ONE pending mutation, under the ROW's OWN owner -- never
    the service's active owner (mirrors Task 2's reminder-side row-owner
    threading test) -- and `outcome.status == "queued"` is treated as UI
    success (no inline error), same as `"saved"`."""
    from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Scheduling.services.scheduling_service import (
        SchedulingService,
    )
    from tldw_chatbook.UI.Screens.scheduling.definition_detail import (
        _definition_edit_payload,
    )

    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        server_client = AsyncMock()
        server_client.preview_automation_definition.side_effect = (
            ServerUnavailableError("offline")
        )
        service = SchedulingService(
            db=db, server_client=server_client, runtime_source="local"
        )
        # Shaped like a genuine `_load_server_automations` item (`id` is
        # the SERVER's own id) -- the shape `_resolve_local_definition_id`
        # expects. A row read back via `db.get_automation_definition`
        # instead (LOCAL id in `id`) is a DIFFERENT shape and trips
        # `_resolve_local_definition_id`'s server/local branch into
        # mirroring a SECOND, bogus row (its own "server_id" would be
        # the already-local uuid) instead of resolving the real one.
        server_item = {
            "id": "srv-def-1",
            "owner_id": "server:1",
            "family": "recurring_question",
            "name": "Server automation",
            "lifecycle": "configured",
            "schedule": {
                "kind": "cron",
                "cron": "0 9 * * 1",
                "timezone": "UTC",
            },
            "input": {"question": "What shipped?"},
            "config": {"generation_mode": "optional"},
            "version": 3,
        }
        db.upsert_automation_definitions_from_server("server:1", [server_item])
        definition_id = db.get_automation_definition_by_server_id(
            "server:1", "srv-def-1"
        )["id"]

        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            workbench = SchedulesWorkbench(app_instance=pilot.app)
            await pilot.app.push_screen(workbench)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = pilot.app.screen.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            detail.set_definition(server_item)
            await pilot.pause()

            row = detail._generation_row
            payload = _definition_edit_payload(
                server_item, config={"generation_mode": "required"}
            )
            workbench._edit_definition_field(server_item, payload, row)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["config"]["generation_mode"] == "required"
            # Final review C1 precedent: the mirror's `version` must not
            # drift locally -- the server checks it for exact equality.
            assert stored["version"] == 3
            assert service.owner_id == "local"  # never repointed by the edit
            mutation = db.get_pending_mutation_for_local_id(
                definition_id, "automation_definition"
            )
            assert mutation is not None
            assert mutation["payload"]["action"] == "update"
            pending = db.get_pending_mutations(
                "server:1", primitive="automation_definition"
            )
            assert len(pending) == 1  # queued under the ROW's own owner

            error = row.query_one(".detail-value-row-error", Static)
            assert error.display is False  # "queued" treated as success
    finally:
        db.close()


@pytest.mark.asyncio
async def test_lifecycle_toggle_pauses_a_local_automation_and_the_button_flips(
    tmp_path,
):
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Local automation",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What shipped?"},
            config={},
        )
        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            workbench = pilot.app.screen
            workbench.query_one("#scheduling-tabs", TabbedContent).active = (
                "scheduling-automations-tab"
            )
            await pilot.pause()
            table = workbench.query_one("#scheduling-automations-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            button = detail.query_one("#scheduling-automation-pause-resume", Button)
            assert button.label.plain == "Pause"
            await pilot.click(button)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_automation_definition(definition_id)["lifecycle"] == "paused"
            assert button.label.plain == "Resume"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_lifecycle_toggle_on_server_owned_row_survives_a_racing_pull(tmp_path):
    """Pinned (task-4 brief): the pause/resume toggle's DB write races a
    sync pull -- Task 2's own `upsert_automation_definitions_from_server`
    lifecycle pull-guard must keep that pull from reverting the pause,
    and that guarantee must be VISIBLE at the UI level (this pane's own
    button) after the next paint, not merely true in the DB."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        # Shaped like a genuine `_load_server_automations` item (`id` is
        # the SERVER's own id, `owner_id` stamped by that fetch) -- the
        # shape `_resolve_local_definition_id` expects and the shape
        # `DefinitionDetail._definition`/the toggle event actually carry
        # in real usage. A row read back via `db.get_automation_
        # definition` instead (LOCAL id in `id`) is a DIFFERENT shape and
        # trips `_resolve_local_definition_id`'s own server/local branch.
        server_item = {
            "id": "srv-def-1",
            "owner_id": "server:1",
            "family": "recurring_question",
            "name": "Server automation",
            "lifecycle": "configured",
            "schedule": {"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            "input": {"question": "What shipped?"},
            "config": {},
            "version": 1,
        }
        db.upsert_automation_definitions_from_server("server:1", [server_item])
        definition_id = db.get_automation_definition_by_server_id(
            "server:1", "srv-def-1"
        )["id"]

        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            workbench = SchedulesWorkbench(app_instance=pilot.app)
            await pilot.app.push_screen(workbench)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = pilot.app.screen.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            )
            detail.set_definition(server_item)
            await pilot.pause()

            workbench._toggle_definition_lifecycle(server_item, "pause")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_automation_definition(definition_id)["lifecycle"] == "paused"
            # Optimistic repaint already applied, ahead of/independent of
            # the background `_request_automations_refresh` reload.
            button = detail.query_one("#scheduling-automation-pause-resume", Button)
            assert button.label.plain == "Resume"

            mutation = db.get_pending_mutation_for_local_id(
                definition_id, "automation_definition"
            )
            assert mutation is not None
            assert mutation["payload"]["action"] == "pause"

            # A racing pull that still thinks the row is "configured" --
            # nothing has replayed the pending mutation yet, so Task 2's
            # guard is live for it.
            db.upsert_automation_definitions_from_server("server:1", [server_item])
            assert (
                db.get_automation_definition(definition_id)["lifecycle"] == "paused"
            )  # guarded, not reverted

            # The UI-visible check: a fresh paint from this (guarded) DB
            # state still shows Resume -- no flicker back to Pause.
            detail.set_definition(db.get_automation_definition(definition_id))
            await pilot.pause()
            assert button.label.plain == "Resume"
    finally:
        db.close()


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


class EmptyMockSchedulingService(_MockSchedulingServiceMixin):
    """Stub service returning no reminder tasks."""

    async def list_reminders(self):
        return []

    async def list_tasks(self, owner_id=None, include_projections=True):
        return await self.list_reminders()


class DistinctMetadataMockSchedulingService(_MockSchedulingServiceMixin):
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

    async def list_tasks(self, owner_id=None, include_projections=True):
        return await self.list_reminders()


class FailingMockSchedulingService(_MockSchedulingServiceMixin):
    """Stub service that raises on list_reminders."""

    async def list_reminders(self):
        raise RuntimeError("service unavailable")

    async def list_tasks(self, owner_id=None, include_projections=True):
        raise RuntimeError("service unavailable")


class WorkbenchTestAppWithEmptyService(ConsolidatedCSSApp):
    """Test app with an empty scheduling service."""

    scheduling_service = EmptyMockSchedulingService()


class WorkbenchTestAppWithDistinctMetadata(ConsolidatedCSSApp):
    """Test app with a scheduling service returning synced metadata."""

    scheduling_service = DistinctMetadataMockSchedulingService()


class WorkbenchTestAppWithFailingService(ConsolidatedCSSApp):
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
    """The d binding opens the delete confirmation dialog for the selected task."""
    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        await pilot.press("d")
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
        assert "Press c" in empty_state.visual.plain


@pytest.mark.asyncio
async def test_no_task_selected_shows_friendly_copy():
    """With no scheduling service, the detail pane prompts task selection."""
    async with WorkbenchTestApp().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        empty_state = pilot.app.screen.query_one(
            "#scheduling-task-detail-empty-state", Static
        )
        assert "Select a task" in empty_state.visual.plain
        assert "press c" in empty_state.visual.plain


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

    class ConflictMockSchedulingService(_MockSchedulingServiceMixin):
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

        async def list_tasks(self, owner_id=None, include_projections=True):
            return await self.list_reminders()

    class WorkbenchTestAppWithConflict(ConsolidatedCSSApp):
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


# redesign PR-2, Task 2: the four watchlist-row tests that used to live
# here (renders/selects/inspects/hides-lifecycle for a watchlist
# projection AT ROW 1 of the queue table) pinned the OLD single-primitive
# table's shape -- a flat reminder+projection list with Title/Type/
# Status/Next-Run columns. Spec S2 locked decision 2 ("Briefing and
# watchlist projections stay out") and Task 1's own report ("Task 2 is
# expected to filter list_tasks' spans-owners result down to real
# ReminderTask instances") retire that: the unified Queue list is
# reminders + automation definitions only. Replaced by the single
# exclusion test below; the detail-pane/inspector rendering for a
# `ScheduledTask` projection stays covered directly by `task_detail.py`'s
# own unit tests (`TaskDetail.set_task`/`TaskInspector.set_task` are
# still general-purpose, just no longer reachable with a projection FROM
# this table).
@pytest.mark.asyncio
async def test_watchlist_projection_excluded_from_unified_queue():
    """A watchlist projection never appears in the unified Queue list."""
    async with WorkbenchTestAppWithMixedService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        assert "Reminder" in str(table.get_row_at(0)[1])
        assert not any(
            row.kind != "reminder" for row in workbench._visible_rows
        ), "only reminder rows are unified-list eligible until definitions exist"
        assert all(
            task.id != "watchlist:1" for task in workbench._tasks
        ), "the projection must never enter the reminder-only _tasks list"


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


class ToggleFailingMockSchedulingService(_MockSchedulingServiceMixin):
    """Stub service that succeeds once, then fails on subsequent calls."""

    def __init__(self):
        super().__init__()
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

    async def list_tasks(self, owner_id=None, include_projections=True):
        return await self.list_reminders()


class WorkbenchTestAppWithToggleFailingService(ConsolidatedCSSApp):
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


class RecordingMockSchedulingService(_MockSchedulingServiceMixin):
    """Stub service that records delete calls and their arguments."""

    def __init__(self, fail_delete: bool = False):
        super().__init__()
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

    async def list_tasks(self, owner_id=None, include_projections=True):
        return await self.list_reminders()

    async def delete_reminder(self, task_id: str, *, owner_id: str | None = None) -> None:
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted_ids.append(task_id)
        self._deleted = True


class WorkbenchTestAppWithRecordingService(ConsolidatedCSSApp):
    """Test app with a recording scheduling service."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduling_service = RecordingMockSchedulingService()


class ControlledRefreshSchedulingService(_MockSchedulingServiceMixin):
    """Return controlled snapshots for a delete/user-refresh overlap."""

    def __init__(self) -> None:
        self.deleted_ids: list[str] = []
        self.delete_completed = asyncio.Event()
        self.mutation_refresh_started = asyncio.Event()
        self.user_refresh_started = asyncio.Event()
        self.release_mutation_refresh = asyncio.Event()
        self.release_user_refresh = asyncio.Event()
        self._list_calls = 0
        timestamp = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
        self.initial_task = ReminderTask(
            id="task-to-delete",
            title="Delete me",
            schedule_kind=ScheduleKind.ONE_TIME,
            run_at=timestamp,
            next_run_at=timestamp,
        )
        self.stale_task = ReminderTask(
            id="stale-task",
            title="Stale mutation snapshot",
            schedule_kind=ScheduleKind.ONE_TIME,
            run_at=timestamp,
            next_run_at=timestamp,
        )
        self.newest_task = ReminderTask(
            id="newest-task",
            title="Newest user snapshot",
            schedule_kind=ScheduleKind.ONE_TIME,
            run_at=timestamp,
            next_run_at=timestamp,
        )

    async def list_tasks(self, owner_id=None, include_projections=True) -> list[ReminderTask]:
        """Return the next controlled task snapshot.

        Returns:
            list[ReminderTask]: Snapshot for the current load step.

        Raises:
            AssertionError: If the workbench performs an unexpected extra load.
        """
        self._list_calls += 1
        if self._list_calls == 1:
            return [self.initial_task]
        if self._list_calls == 2:
            self.mutation_refresh_started.set()
            await self.release_mutation_refresh.wait()
            return [self.stale_task]
        if self._list_calls == 3:
            self.user_refresh_started.set()
            await self.release_user_refresh.wait()
            return [self.newest_task]
        raise AssertionError(f"Unexpected list_tasks call {self._list_calls}")

    async def delete_reminder(
        self, task_id: str, *, owner_id: str | None = None
    ) -> None:
        """Record completion of the controlled delete.

        Args:
            task_id: Identifier of the deleted reminder.
            owner_id: The row's own owner, threaded by the workbench
                (final review F4); recorded only, not acted on.
        """
        self.deleted_ids.append(task_id)
        self.delete_completed.set()


class WorkbenchTestAppWithControlledRefresh(ConsolidatedCSSApp):
    """Test app with event-controlled task snapshots."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = ControlledRefreshSchedulingService()


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
async def test_delete_mutation_refresh_cannot_repaint_after_newer_user_refresh():
    """A delete reconciliation cannot overwrite a newer grouped refresh."""
    app = WorkbenchTestAppWithControlledRefresh()
    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.app.workers.wait_for_complete()

        service = app.scheduling_service
        async with asyncio.timeout(2):
            workbench.post_message(DeleteTaskRequested(workbench._tasks[0]))
            await service.delete_completed.wait()
            await service.mutation_refresh_started.wait()

            # Resolving a conflict is a user action whose production handler
            # requests a grouped task-list refresh.
            workbench._on_conflict_resolved(
                ConflictsTab.ConflictResolved("conflict-1", "local")
            )
            await service.user_refresh_started.wait()
            user_worker = next(
                worker
                for worker in pilot.app.workers
                if worker.node is workbench and worker.group == "schedules-load-tasks"
            )

            service.release_user_refresh.set()
            await pilot.app.workers.wait_for_complete([user_worker])
            assert [task.id for task in workbench._tasks] == ["newest-task"]

            service.release_mutation_refresh.set()
            await pilot.app.workers.wait_for_complete()

        assert service.deleted_ids == ["task-to-delete"]
        assert [task.id for task in workbench._tasks] == ["newest-task"]
        table = workbench.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        # redesign PR-2, Task 2: column 0 is now the glyph, column 1 the
        # title (old single-primitive shape was Title/Type/Status/Next Run).
        assert "Newest user snapshot" in str(table.get_row_at(0)[1])


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
    """The c binding opens the reminder form; saving calls the scheduling service."""
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
        run_at_input.value = "2030-07-20T14:00:00+00:00"

        await pilot.click("#reminder-save")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service = pilot.app.scheduling_service
        assert len(service.created) == 1
        assert service.created[0]["title"] == "New reminder"
        assert service.created[0]["schedule_kind"] == "one_time"
        # Owner threaded through the call, never a `set_owner` flip.
        assert service.created_owners == [service.owner_id]
        notifications = list(pilot.app._notifications)
        # Same-owner save: the plain toast, no "switch owner" hint.
        assert any(n.message == "Scheduled task created." for n in notifications)


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
        assert any(n.message == "Scheduled task updated." for n in notifications)


@pytest.mark.asyncio
async def test_create_reminder_for_a_different_runs_on_owner_queues_a_mutation(
    tmp_path,
):
    """task-5: picking a non-default "Runs on" owner in the create form
    rides the EXISTING `create_reminder` server-fallback/mutation path
    (no new persistence code) -- and the service's shared `owner_id` is
    never repointed at the owner that was only meant for this one save
    (Qodo HIGH: the owner is threaded through the call, so no concurrent
    worker can observe a temporary flip).

    The toast is also checked here (Qodo MEDIUM): this list is owner-scoped,
    so a reminder created for another owner cannot appear in it -- a bare
    "Scheduled task created." reads as a lost save."""
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
    from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError

    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    server_client = AsyncMock()
    server_client.notifications_service = object()
    server_client.create_reminder.side_effect = ServerUnavailableError("offline")
    # A "server available" app also triggers the (unrelated) Automations-tab
    # load on mount; give it a real return value so it settles cleanly
    # instead of leaving an un-awaited AsyncMock coroutine at teardown.
    server_client.list_automation_definitions = AsyncMock(
        return_value={"items": [], "total": 0}
    )
    service = SchedulingService(db=db, server_client=server_client, runtime_source="local")

    app = WorkbenchTestApp()
    app.scheduling_service = service
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_server_id="example.com")
    )
    try:
        async with app.run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()

            pilot.app.screen.action_create_reminder()
            await pilot.pause()
            assert isinstance(pilot.app.screen, ReminderForm)

            runs_on = pilot.app.screen.query_one("#reminder-runs-on", Select)
            option_values = [value for _label, value in runs_on._options]
            assert option_values == ["local", "server:example.com"]
            assert runs_on.value == "local"  # default = current screen owner
            runs_on.value = "server:example.com"

            pilot.app.screen.query_one("#reminder-title", Input).value = "Server reminder"
            pilot.app.screen.query_one(
                "#reminder-run-at", Input
            ).value = "2099-08-28 09:00"

            await pilot.click("#reminder-save")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            toasts = [n.message for n in pilot.app._notifications]
            assert any(
                "Server (example.com)" in message for message in toasts
            ), f"the toast must name the owner it was created for; got {toasts}"
            # Final review F7: the queue spans owners since redesign PR-2
            # Task 1, so the old "switch to that owner to see it"
            # instruction told the user to do something unnecessary.
            assert not any("switch to that owner" in m for m in toasts), (
                f"the cross-owner instruction must be gone; got {toasts}"
            )

        assert service.owner_id == "local"  # never repointed by the save
        assert service.sync_engine.owner_id == "local"
        pending = db.get_pending_mutations(
            "server:example.com", primitive="reminder_task"
        )
        assert len(pending) == 1
        assert pending[0]["payload"]["action"] == "create"
        rows = db.list_reminder_tasks(owner_id="server:example.com")
        assert len(rows) == 1
        assert rows[0]["title"] == "Server reminder"
        assert db.list_reminder_tasks(owner_id="local") == []
    finally:
        db.close()


def test_sync_completed_event():
    msg = SyncCompleted("server:1", conflict_count=2)
    assert msg.owner_id == "server:1"
    assert msg.conflict_count == 2


def test_sync_failed_event():
    msg = SyncFailed("server:1", error="timeout")
    assert msg.owner_id == "server:1"
    assert msg.error == "timeout"


@pytest.mark.asyncio
async def test_action_sync_now_notifies_when_no_service():
    app = WorkbenchTestApp()
    workbench = SchedulesWorkbench(app)
    # Should not crash and should not start a worker
    workbench.action_sync_now()


def test_action_sync_now_guard_prevents_duplicate_workers():
    class FakeService:
        def __init__(self):
            self.owner_id = "local"
            self.server_client = None
            self.sync_now = AsyncMock()
            self.db = None

    app = WorkbenchTestAppWithService()
    app.scheduling_service = FakeService()
    workbench = SchedulesWorkbench(app)
    workbench._sync_running = True
    workbench.action_sync_now()
    # The app should have received a warning notification.
    # Exact assertion depends on the test harness; at minimum it must not start a second worker.


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
        clear_btn = widget.query_one("#scheduling-clear-error", Button)
        assert local_btn.variant != "primary"
        assert server_btn.variant == "primary"
        assert str(local_btn.tooltip) == "Use local storage as the Schedules owner."
        assert (
            str(server_btn.tooltip)
            == "Use the connected server as the Schedules owner."
        )
        assert str(clear_btn.tooltip) == "Clear the latest scheduling sync error."

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
        assert (
            str(server_btn.tooltip)
            == "Connect a scheduling server before switching Schedules ownership."
        )

        widget.set_owner_state(
            current_owner="server:example.com",
            active_server_id="example.com",
            server_available=True,
        )
        await pilot.pause()
        assert not server_btn.disabled
        assert (
            str(server_btn.tooltip)
            == "Use the connected server as the Schedules owner."
        )


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
        tab.populate(
            [
                {
                    "id": "c1",
                    "local_id": "l1",
                    "server_state": {},
                    "local_state": {"record": {"title": "Local"}},
                },
            ]
        )
        await pilot.pause()

        table = tab.query_one("#scheduling-conflicts-table", DataTable)
        assert table.row_count == 1
        server_button = tab.query_one("#scheduling-use-server", Button)
        local_button = tab.query_one("#scheduling-use-local", Button)
        # The Lab/Schedules/Logs UX overhaul (9dd2374b5, ADR-031) replaced the
        # per-button resolve tooltips with one guidance line set by
        # `_set_actions_enabled` whenever rows exist; this pin tracks that
        # shipped copy (the old per-version tooltips only exist pre-populate).
        assert (
            str(server_button.tooltip)
            == "Select a conflict above, then choose which version to keep."
        )
        assert (
            str(local_button.tooltip)
            == "Select a conflict above, then choose which version to keep."
        )
        table.cursor_coordinate = (0, 0)
        await pilot.click("#scheduling-use-server")
        await pilot.pause()

        # Resolution is guarded by a confirmation dialog (UX-007).
        assert isinstance(pilot.app.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
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
        tab.populate(
            [
                {
                    "id": "c1",
                    "local_id": "l1",
                    "server_state": {},
                    "local_state": {"record": {"title": "Local"}},
                },
            ]
        )
        await pilot.pause()

        table = tab.query_one("#scheduling-conflicts-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.click("#scheduling-use-server")
        await pilot.pause()

        # Confirmation is required before the (failing) resolution runs.
        assert isinstance(pilot.app.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert engine.calls == [("c1", "server")]
        assert len(tab.posted_messages) == 0
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_persisted_policy_refusal_is_not_surfaced_as_sync_error():
    """task-2722: older builds persisted the local-mode policy refusal as a
    sync error; profiles carrying those rows must not wear the error badge."""
    app = WorkbenchTestAppWithService()
    app.scheduling_service.db = _MockSchedulingDB(
        sync_state={
            "sync_errors": [
                {
                    "message": (
                        "notifications.reminders.list.server requires server mode."
                    ),
                    "timestamp": "2026-08-02T22:36:00+00:00",
                }
            ]
        }
    )

    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()

        header_calls: list[tuple[str, str]] = []
        workbench._sync_header_status = lambda status, label: header_calls.append(
            (status, label)
        )
        workbench._refresh_owner_select()
        await pilot.pause()

        error_widget = workbench.query_one("#scheduling-sync-error", Static)
        assert str(error_widget.renderable) == "", (
            f"refusal shown as sync error: {error_widget.renderable!r}"
        )
        assert not [c for c in header_calls if c[0] == "error"], (
            f"header wears the error badge for a policy refusal: {header_calls}"
        )


@pytest.mark.asyncio
async def test_real_sync_error_still_surfaces():
    """Guard for task-2722: only policy refusals are filtered from display."""
    app = WorkbenchTestAppWithService()
    app.scheduling_service.db = _MockSchedulingDB(
        sync_state={
            "sync_errors": [
                {
                    "message": "server unreachable: connection refused",
                    "timestamp": "2026-08-02T22:36:00+00:00",
                }
            ]
        }
    )

    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()

        error_widget = workbench.query_one("#scheduling-sync-error", Static)
        assert "connection refused" in str(error_widget.renderable)


@pytest.mark.asyncio
async def test_sync_strip_fields_do_not_abut():
    """task-2723: 'Last pull: —Last push: —<error text>' rendered as one
    unbroken run. Assert real layout geometry: each field ends at least one
    cell before the next begins."""
    app = WorkbenchTestAppWithService()
    app.scheduling_service.db = _MockSchedulingDB(
        sync_state={
            "sync_errors": [
                {
                    "message": "server unreachable: connection refused",
                    "timestamp": "2026-08-02T22:36:00+00:00",
                }
            ]
        }
    )
    # task-23105: a local-owner bar with no server collapses these fields
    # away entirely; give the harness a live server so they render and the
    # original task-2723 geometry claim stays testable.
    app.scheduling_service.server_client = _MockServerClient(
        notifications_service=object()
    )
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_server_id="example.com")
    )

    async with app.run_test(size=(200, 40)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()

        pull = workbench.query_one("#scheduling-last-pull", Static)
        push = workbench.query_one("#scheduling-last-push", Static)
        error = workbench.query_one("#scheduling-sync-error", Static)

        assert pull.region.right < push.region.x, (
            f"pull {pull.region} abuts push {push.region}"
        )
        assert push.region.right < error.region.x, (
            f"push {push.region} abuts error {error.region}"
        )


# --- redesign PR-2, Task 3: rail chrome + bottom status strip --------------


def _definition_row(
    def_id: str, *, name: str = "Digest", lifecycle: str = "configured"
) -> dict:
    return {
        "id": def_id,
        "server_id": None,
        "owner_id": "local",
        "name": name,
        # Qodo MEDIUM (schedules-redesign PR-2): `build_unified_rows`
        # filters definitions to `family == "recurring_question"` --
        # every real definition row carries a family, so this fixture
        # must too.
        "family": "recurring_question",
        "lifecycle": lifecycle,
        "schedule": {"kind": "one_time", "run_at": "2099-01-01T00:00:00+00:00"},
        "input": {"question": "What changed?"},
        "updated_at": "2026-08-01T00:00:00+00:00",
    }


class _RailService(_MockSchedulingServiceMixin):
    """One definition, optionally carrying an unread result -- drives the
    rail's `Mark all read` visibility (redesign PR-2, Task 3)."""

    def __init__(self, *, unread: bool = False, conflicts: list | None = None) -> None:
        self.owner_id = "local"
        self.db = _MockSchedulingDB(
            conflicts=conflicts,
            automation_definitions=[_definition_row("def-1")],
            automation_results=(
                [
                    {
                        "id": "result-1",
                        "definition_id": "def-1",
                        "owner_id": "local",
                        "review_state": "unread",
                        "kind": "finding",
                        "created_at": "2026-08-20T00:00:00+00:00",
                    }
                ]
                if unread
                else []
            ),
        )

    async def list_tasks(self, owner_id=None, include_projections=True):
        return []


class _RailTestApp(ConsolidatedCSSApp):
    def __init__(self, service: _RailService, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = service


@pytest.mark.asyncio
async def test_mark_all_read_hidden_when_nothing_unread():
    app = _RailTestApp(_RailService(unread=False))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        button = pilot.app.screen.query_one("#scheduling-mark-all-read", Button)
        assert button.display is False


@pytest.mark.asyncio
async def test_mark_all_read_visible_when_a_definition_has_unread_results():
    """redesign PR-2, Task 3: the rail action is shown only once the
    unified rows' total unread count is > 0 -- summed across
    `UnifiedRow.unread_count`, not a new query."""
    app = _RailTestApp(_RailService(unread=True))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        button = pilot.app.screen.query_one("#scheduling-mark-all-read", Button)
        assert button.display is True


@pytest.mark.asyncio
async def test_conflicts_badge_shows_count_and_switches_to_conflicts_tab():
    """redesign PR-2, Task 3, plan ruling 4: the status strip's conflicts
    badge mirrors `_refresh_conflicts_tab`'s own existing count (no new
    query) and switches to the Conflicts tab on click -- no overlay."""
    app = _RailTestApp(
        _RailService(conflicts=[{"id": "c1"}, {"id": "c2"}])
    )
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        badge = workbench.query_one("#scheduling-conflicts-badge", Button)
        assert str(badge.label) == "Conflicts (2)"
        assert "scheduled task" in str(badge.tooltip).lower()

        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        assert tabs.active != "scheduling-conflicts-tab"

        badge.press()
        await pilot.pause()

        assert tabs.active == "scheduling-conflicts-tab"


@pytest.mark.asyncio
async def test_conflicts_badge_defaults_to_no_count():
    app = _RailTestApp(_RailService())
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        badge = pilot.app.screen.query_one("#scheduling-conflicts-badge", Button)
        assert str(badge.label) == "Conflicts"


@pytest.mark.asyncio
async def test_status_strip_sync_widget_is_compact_at_narrow_width():
    """redesign PR-2, Task 3: the strip's width-triggered compact path --
    the default (80, 24) test size is well under `SCHEDULES_COMPACT_
    WORKBENCH_MAX_WIDTH` (120), so `_sync_responsive_workbench` (run at
    `on_mount`) should already have applied it."""
    app = _RailTestApp(_RailService())
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        sync_status = pilot.app.screen.query_one(
            "#scheduling-sync-status", SyncStatusWidget
        )
        assert "compact" in sync_status.classes


@pytest.mark.asyncio
async def test_status_strip_sync_widget_is_not_compact_at_wide_width():
    app = _RailTestApp(_RailService())
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        sync_status = pilot.app.screen.query_one(
            "#scheduling-sync-status", SyncStatusWidget
        )
        assert "compact" not in sync_status.classes
