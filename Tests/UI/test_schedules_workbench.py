"""Tests for the SchedulesWorkbench shell."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger as _loguru_logger

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
)
from textual.widgets._collapsible import CollapsibleTitle

from textual import on

from tldw_chatbook.Scheduling.events import (
    DefinitionRunNowRequested,
    DeleteTaskRequested,
    ReminderDispatched,
    ReminderFieldEditRequested,
    ReminderOwnerActionRequested,
    SyncCompleted,
    SyncFailed,
    ViewDefinitionAuditRequested,
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
from tldw_chatbook.Scheduling.services import (
    scheduling_service as scheduling_service_module,
)
from tldw_chatbook.Scheduling.services.sync_engine import SyncOutcome
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
from tldw_chatbook.UI.Screens.scheduling.results_tab import ResultsHostScreen, ResultsTab
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    NEXT_RUN_REFRESH_SECONDS,
    SCHEDULER_LIVENESS_REFRESH_SECONDS,
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
from tldw_chatbook.UI.Screens.scheduling.task_detail import (
    TaskDetail,
    TaskInspector,
    _STATUS_BADGE_CLASSES,
    _humanize_cron,
)
from tldw_chatbook.UI.Screens.scheduling.workbench_host_screen import (
    WorkbenchHostScreen,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.delete_confirmation_dialog import DeleteConfirmationDialog
from tldw_chatbook.Widgets.detail_value_row import DetailValueRow


# Shared across the Schedules UI test files (task-23106 review round F15).
from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB as _MockSchedulingDB,
    MockSchedulingServiceMixin as _MockSchedulingServiceMixin,
    MockServerClient as _MockServerClient,
    painted_glyphs_at,
    rendered_row_cells,
    settle_schedules_workbench,
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
async def test_repeat_row_editor_paints_its_current_value_when_focused():
    """Geometry pin (schedules-UAT-remediation, finding 1a): the test
    above only proved ``editor.value`` -- a stored attribute the widget
    can carry correctly while showing nothing at all. Uncompacted, a
    `Select`'s own `border: tall` (3 rows) is clipped to 1 row by
    `DetailValueRow`'s fixed-height line container
    (`.detail-value-row-line { height: 1 }`), and the surviving row is
    the border's TOP EDGE, not the label -- `editor.value` still reads
    "monday" while the compositor paints only border glyphs. Fails
    against the unfixed code (bare `Select(...)`, no `compact=True`) --
    see the task-1 report's revert-check."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder())
        await pilot.pause()

        repeat_row = detail._repeat_row
        await pilot.click(repeat_row)
        await pilot.pause()

        editor = repeat_row.query_one(Select)
        assert editor.value == "monday"
        assert editor.has_class("-textual-compact"), (
            "the Repeat row's Select editor was constructed without "
            "compact=True"
        )
        painted = painted_glyphs_at(pilot.app, editor)
        assert "Every Monday" in painted, (
            f"the Repeat editor's own value is not painted: {painted!r}"
        )


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
async def test_at_row_editor_paints_its_current_value_when_focused():
    """Geometry pin (schedules-UAT-remediation, finding 1a): an
    uncompacted `Input`'s DEFAULT_CSS is `height: 3` (border + content +
    border), clipped to 1 row by the same `.detail-value-row-line`
    container -- the surviving row is the top border, so `editor.value`
    reads the ISO timestamp correctly while the compositor paints only
    border glyphs. Fails against the unfixed code (bare `Input(...)`, no
    `compact=True`)."""
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
        await pilot.click(at_row)
        await pilot.pause()

        editor = at_row.query_one(Input)
        assert editor.value == run_at.isoformat()
        assert editor.has_class("-textual-compact"), (
            "the At row's Input editor was constructed without compact=True"
        )
        painted = painted_glyphs_at(pilot.app, editor)
        assert run_at.isoformat() in painted, (
            f"the At editor's own value is not painted: {painted!r}"
        )


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


class _CapturingDefinitionDetailApp(ConsolidatedCSSApp):
    """`_BareDefinitionDetailApp`'s twin with its own message capture
    (redesign PR-4, task 3): a plain bare harness cannot observe a
    posted `Message` bubbling past it -- an `@on` handler on the App
    itself records what `DefinitionDetail` posts, no workbench needed."""

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.run_now_events: list = []
        self.audit_events: list = []

    def compose(self):
        yield DefinitionDetail()

    @on(DefinitionRunNowRequested)
    def _capture_run_now(self, event: DefinitionRunNowRequested) -> None:
        self.run_now_events.append(event.definition)

    @on(ViewDefinitionAuditRequested)
    def _capture_audit(self, event: ViewDefinitionAuditRequested) -> None:
        self.audit_events.append(event.definition)


@pytest.mark.asyncio
async def test_run_now_button_posts_definition_run_now_requested():
    """redesign PR-4, task 3, ruling 2: the header `Run now` button
    (the retired Automations-tab `r` key's live replacement) posts
    `DefinitionRunNowRequested` carrying the painted definition -- never
    gated on the lifecycle lock or family note, same as the tab's own
    `r` key never was."""
    async with _CapturingDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        definition = _editable_definition()
        detail.set_definition(definition)
        await pilot.pause()

        run_now = detail.query_one("#scheduling-automation-run-now", Button)
        assert run_now.disabled is False
        detail.on_button_pressed(Button.Pressed(run_now))
        await pilot.pause()

        assert pilot.app.run_now_events == [definition]


@pytest.mark.asyncio
async def test_run_now_button_is_a_no_op_with_nothing_painted():
    async with _CapturingDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail._request_run_now()
        await pilot.pause()
        assert pilot.app.run_now_events == []


@pytest.mark.asyncio
async def test_last_run_row_activation_posts_view_definition_audit_requested():
    """redesign PR-4, task 3 (audit-view relocation): the `Last run` row
    -- whose own copy already says "...see Run history" for a
    server-owned definition -- is a live activation now, posting
    `ViewDefinitionAuditRequested` rather than opening an editor.
    Unconditionally activatable regardless of family/lifecycle lock,
    same as `Unread results` (task 2's own precedent): viewing history
    is not an edit."""
    async with _CapturingDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        # A non-`recurring_question` row -- editors are gated for this
        # family, but the audit activation must still fire.
        definition = _editable_definition(family="agent_task")
        detail.set_definition(definition)
        await pilot.pause()

        row = detail._last_run_row
        assert row.affordance is True
        assert row.can_focus is True
        row.post_message(DetailValueRow.Activated(row))
        await pilot.pause()

        assert pilot.app.audit_events == [definition]
        # Never opened an in-place editor for this row.
        assert not row.query(Input)
        assert not row.query(Select)


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
async def test_model_row_editor_paints_its_current_value_when_focused():
    """Geometry pin (schedules-UAT-remediation, finding 1a), the
    `DefinitionDetail` twin of `test_at_row_editor_paints_its_current_
    value_when_focused` -- same `.detail-value-row-line { height: 1 }`
    clip, a different pane class entirely, so the fix (`compact=True` at
    this construction site) is pinned independently of `TaskDetail`'s
    own."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())
        await pilot.pause()

        model_row = detail._model_row
        await pilot.click(model_row)
        await pilot.pause()
        editor = model_row.query_one(Input)
        assert editor.value == "openai/gpt-5"
        assert editor.has_class("-textual-compact"), (
            "the Model row's Input editor was constructed without "
            "compact=True"
        )
        painted = painted_glyphs_at(pilot.app, editor)
        assert "openai/gpt-5" in painted, (
            f"the Model editor's own value is not painted: {painted!r}"
        )


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
async def test_generation_row_editor_paints_its_current_value_when_focused():
    """Geometry pin (schedules-UAT-remediation, finding 1a) -- the
    `Select` case for `DefinitionDetail`, mirroring `TaskDetail`'s Repeat
    row pin."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition())  # generation_mode="required"
        await pilot.pause()
        row = detail._generation_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Select)
        assert editor.value == "required"
        assert editor.has_class("-textual-compact"), (
            "the Generation row's Select editor was constructed without "
            "compact=True"
        )
        painted = painted_glyphs_at(pilot.app, editor)
        assert "Always generate a draft" in painted, (
            f"the Generation editor's own value is not painted: {painted!r}"
        )


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
async def test_definition_timezone_editor_says_automation_not_task(monkeypatch):
    """Final review F10: this pane's inline Timezone editor is one of the
    two surfaces that got the reminder wording from task 3's option-source
    consolidation -- it passes `noun="automation"` now."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        definition = _editable_definition()
        definition["schedule"]["timezone"] = "Mars/Olympus"
        detail.set_definition(definition)
        await pilot.pause()
        row = detail._timezone_row
        await pilot.click(row)
        await pilot.pause()
        labels = [str(prompt) for prompt, _value in row.query_one(Select)._options]
        assert "Mars/Olympus — stored on this automation, not recognized here" in labels


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
async def test_committing_generation_edit_persists_and_repaints_the_definition_pane(
    tmp_path,
):
    """Commit persists via `save_definition`; success repaints the
    definition pane from a fresh read AND refreshes the unified list --
    the same `_definitions_stale` seam every other definition mutation in
    this file uses (run-now, transfer begin/cancel).

    redesign PR-4 task 5: the refresh used to be LAZY (marked stale, then
    picked up whenever the user next arrived on the Queue tab) because
    the eager half went to the Automations tab's own list. That tab is
    retired and the unified list is the only definitions surface, so the
    mutation refreshes it directly and the flag it sets is consumed in
    the same beat -- which is what the last assertion now reads. The
    set-site itself is unchanged (brief: "the staleness machine keeps
    every set-site"), so a concurrent reminder-only reload still upgrades
    itself."""
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
            # The unified list's own refresh ran and consumed the flag.
            assert workbench._definitions_stale is False
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
async def test_committing_model_edit_persists_and_repaints_the_definition_pane(
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
async def test_committing_finding_policy_edit_persists_and_repaints_the_definition_pane(
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
async def test_committing_sources_edit_persists_and_repaints_the_definition_pane(
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
async def test_committing_notifications_edit_persists_and_repaints_the_definition_pane(
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
async def test_committing_at_edit_persists_and_repaints_the_definition_pane(tmp_path):
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
async def test_at_edit_on_a_daily_automation_edits_time_of_day_not_run_at(tmp_path):
    """Qodo finding 8: the `At` row commits the field THIS schedule kind
    owns and never converts the kind.

    A `daily` schedule's `At` used to be editable (the row was gated on
    "not cron") while the commit hard-coded `{"kind": "one_time",
    "run_at": ...}` -- so setting the time on a daily automation turned it
    into a one-shot and dropped `time_of_day`. The pin is the whole
    schedule dict, `weekday` included: only the edited field moves."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Weekly digest",
            schedule={
                "kind": "weekly",
                "time_of_day": "09:00",
                "weekday": 2,
                "timezone": "UTC",
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            row = detail._at_row
            assert row.affordance is True, "a weekly schedule has a time target"
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            input_widget = row.query_one(Input)
            # Seeded from the field the kind owns, not from `run_at`.
            assert input_widget.value == "09:00"
            input_widget.value = "7:30"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)["schedule"]
            assert stored["time_of_day"] == "07:30"  # normalized, zero-padded
            assert stored["kind"] == "weekly"  # NEVER converted
            assert stored["weekday"] == 2  # sibling fields preserved
            assert stored["timezone"] == "UTC"
            assert "run_at" not in stored

            # Ruling 2: the row must repaint to the AUTHORITATIVE value.
            # `_definition_at_label` used to answer "-" for every kind but
            # cron/one_time, so this row went on reading "-" after a
            # successful edit -- the dishonest-repaint class this program
            # polices, on the row task 4 had just made editable.
            at_static = detail.query_one(
                "#scheduling-automation-detail-at", Static
            )
            assert (
                at_static.render_line(0).text.strip()
                == "Weekly on Wednesday at 07:30 UTC"
            )

            # SECOND surface: `_definition_at_label` also feeds the
            # unified-row subtitle (`build_unified_rows`'s
            # `schedule_summary` -> `_row_subtitle`). Painted, so the
            # shared-helper change is verified where it actually lands
            # rather than assumed. (redesign PR-4 task 5: no tab flip back
            # -- the row and the pane are on the same surface now.)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            queue_table = workbench.query_one("#scheduling-task-table", DataTable)
            painted = " ".join(
                str(cell)
                for index in range(queue_table.row_count)
                for cell in queue_table.get_row_at(index)
            )
            assert "Weekly on Wednesday at 07:30 UTC" in painted
    finally:
        db.close()


@pytest.mark.asyncio
async def test_at_row_is_read_only_for_an_interval_automation(tmp_path):
    """Qodo finding 8, the other half: an `interval` schedule has no
    single time field for `At` to edit, so the row offers no affordance
    rather than an editor that would rewrite the kind."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        db.create_automation_definition(
            "local",
            "recurring_question",
            "Every hour",
            schedule={"kind": "interval", "every_seconds": 3600},
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            assert detail._at_row.affordance is False
            assert detail._at_row.can_focus is False
            detail._at_row.post_message(DetailValueRow.Activated(detail._at_row))
            await pilot.pause()
            assert not detail._at_row.query(Input)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_two_fields_of_one_definition_both_land_without_a_lost_update(tmp_path):
    """Qodo finding 7: the edit worker groups per DEFINITION, not per
    field.

    With a per-FIELD group the two commits below ran concurrently, and
    `save_definition` merges each payload onto the row it read at entry
    -- so whichever finished second wrote back a snapshot taken before
    the first landed, silently reverting it. Serialized per definition,
    both survive."""
    from tldw_chatbook.UI.Screens.scheduling.definition_detail import (
        _definition_edit_payload,
    )

    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily standup question",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            definition = dict(detail._definition)

            # Two DIFFERENT fields of the SAME definition, dispatched
            # back to back with nothing awaited in between -- the shape
            # a fast typist produces.
            workbench._edit_definition_field(
                definition,
                _definition_edit_payload(definition, input={"model": "gpt-5"}),
                detail._model_row,
            )
            workbench._edit_definition_field(
                definition,
                _definition_edit_payload(
                    definition, config={"generation_mode": "required"}
                ),
                detail._generation_row,
            )
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)
            assert stored["config"]["generation_mode"] == "required"
            assert stored["input"]["model"] == "gpt-5"  # not reverted by the second
            assert stored["input"]["question"] == "What shipped?"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_model_only_target_round_trips_through_an_unchanged_submit(tmp_path):
    """Qodo finding 10: a stored `model` with no `provider` must survive
    being opened and submitted unchanged.

    The editor seeds `/model` for that shape -- the exact inverse of the
    commit's parse, where bare text means "provider". Seeding the bare
    model instead (what the row's display label shows) silently moved the
    value into the provider slot on the next Enter."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily standup question",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What shipped?", "model": "gpt-5"},
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            # The row still READS as a bare model -- the display label is
            # unchanged, only the editor's seed is the parse's inverse.
            model_static = detail.query_one(
                "#scheduling-automation-detail-model", Static
            )
            assert model_static.render_line(0).text.strip() == "gpt-5"

            row = detail._model_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            assert row.query_one(Input).value == "/gpt-5"
            await pilot.press("enter")  # submitted UNCHANGED
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            stored = db.get_automation_definition(definition_id)["input"]
            assert stored["model"] == "gpt-5"
            assert not stored.get("provider")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_non_recurring_question_definition_exposes_no_editors(tmp_path):
    """Qodo finding 11: the pane only authors `recurring_question`, and
    every payload it builds declares that family -- so an `agent_task`
    row renders every row read-only and says why, instead of offering
    editors whose commit would push it through the wrong family's
    normalizer."""
    db, service = _real_scheduling_service(tmp_path)
    try:
        db.create_automation_definition(
            "local",
            "agent_task",
            "Nightly agent run",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={},
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            assert detail._definition["family"] == "agent_task"
            assert [row.row_key for row in detail._editable_rows() if row.affordance] == []
            for row in detail._editable_rows():
                row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            assert not detail.query(Input)
            assert not detail.query(Select)
            why = detail.query_one("#scheduling-automation-detail-why", Static)
            assert "isn't a recurring question" in why.render_line(0).text
    finally:
        db.close()


def test_stale_definition_row_error_is_not_painted_on_another_row(tmp_path):
    """Qodo finding 9: the edit worker holds a `DetailValueRow` across an
    `await`, so a failure arriving after the pane moved on must not stamp
    its message under a different automation's field. The commit path
    already validates the captured identity (`_editing_definition`); this
    is the same check on the error path."""
    detail = DefinitionDetail()
    detail._definition = {"id": "def-B"}
    row = DetailValueRow("Model", "-", row_key="model")
    # Not mounted under the pane: the helper reads the row's ancestors,
    # so drive it through the pane it would really have found.
    workbench = SchedulesWorkbench.__new__(SchedulesWorkbench)

    painted: list[str] = []
    row.show_error = painted.append  # type: ignore[method-assign]

    with patch.object(
        type(row), "ancestors", property(lambda _self: [detail])
    ):
        workbench._show_definition_row_error(row, {"def-A"}, "boom")
        assert painted == [], "an error for def-A must not land while def-B shows"
        workbench._show_definition_row_error(row, {"def-B"}, "boom")
        assert painted == ["boom"]


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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
            # Fix wave F4: `settle_schedules_workbench` (not a bare
            # pause + wait) -- the mount-time catch-up results pull is a
            # `set_timer` callback `wait_for_complete` does not cover, and
            # its `_request_tasks_refresh` wipes a hand-set definition off
            # the live `#scheduling-queue-definition-detail` pane.
            await settle_schedules_workbench(pilot, workbench)

            detail = pilot.app.screen.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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
            # Fix wave F4: `settle_schedules_workbench` (not a bare
            # pause + wait) -- the mount-time catch-up results pull is a
            # `set_timer` callback `wait_for_complete` does not cover, and
            # its `_request_tasks_refresh` wipes a hand-set definition off
            # the live `#scheduling-queue-definition-detail` pane.
            await settle_schedules_workbench(pilot, workbench)

            detail = pilot.app.screen.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
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

            # Qodo findings 5+6: the toggle queues under the LIFECYCLE
            # primitive now, which is the same slot the pull guard reads.
            # The optimistic path and the guard must stay keyed on the
            # SAME primitive, or the racing pull below reverts the pause.
            mutation = db.get_pending_mutation_for_local_id(
                definition_id, "automation_lifecycle"
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
        # redesign PR-4 task 4: create rebound from c to n (spec §12).
        assert "Press n" in empty_state.visual.plain


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
        # redesign PR-4 task 4: create rebound from c to n (spec §12).
        assert "press n" in empty_state.visual.plain


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
    """A service failure surfaces an error notification; on the very
    first (never-succeeded) load there is no prior state to preserve, so
    the detail pane simply keeps its pre-load compose-time copy (UAT
    Major 5: the fix is to stop DESTROYING good state on a later
    failure, not to paint a bespoke first-failure copy). Review round 1
    finding 1: the toast text itself must not claim a "last-loaded
    queue" that never existed."""
    async with WorkbenchTestAppWithFailingService().run_test() as pilot:
        notify_calls: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, severity="information", **_: (
            notify_calls.append((severity, str(message)))
        )
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        errors = [message for severity, message in notify_calls if severity == "error"]
        assert errors, notify_calls
        assert "last-loaded queue" not in errors[0], (
            "the first-ever load has no last-good queue to claim it is showing"
        )
        assert "Could not load tasks" in errors[0]
        empty_state = pilot.app.screen.query_one(
            "#scheduling-task-detail-empty-state", Static
        )
        assert "Select a task from the queue" in empty_state.visual.plain


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
async def test_load_tasks_service_error_keeps_the_last_good_rows():
    """UAT Major 5: a service failure AFTER data was loaded must not
    destroy the last-good display -- a read failure is not evidence the
    queue is empty. This used to clear the table and the internal task
    list on ANY exception here, with nothing short of a fresh mount able
    to restore it; this test's own former name/assertions pinned that
    bug."""
    async with WorkbenchTestAppWithToggleFailingService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1
        workbench = pilot.app.screen
        tasks_before = list(workbench._tasks)
        rows_before = list(workbench._all_rows)

        notify_calls: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, severity="information", **_: (
            notify_calls.append((severity, str(message)))
        )
        await workbench.load_tasks()
        await pilot.pause()

        assert table.row_count == 1, "the table must keep the last-good rows"
        assert workbench._tasks == tasks_before
        assert workbench._all_rows == rows_before
        errors = [message for severity, message in notify_calls if severity == "error"]
        assert errors, notify_calls
        assert "last-loaded queue" in errors[0], (
            "a failure AFTER a real load must say so -- there IS a last-good queue"
        )


@pytest.mark.asyncio
async def test_load_tasks_failure_logs_which_step_raised():
    """UAT Major 5's logging hook: a bare `logger.exception("Failed to
    load tasks")` covering three distinct calls (reminders listing,
    definitions listing, row build) left no way to pin the raiser
    without reproducing it live. The log line must now name the step,
    the exception type, and enough context to find it without a repro."""
    records: list[str] = []
    sink_id = _loguru_logger.add(
        lambda message: records.append(message.record["message"]), level="ERROR"
    )
    try:
        async with WorkbenchTestAppWithFailingService().run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
    finally:
        _loguru_logger.remove(sink_id)

    assert any(
        "listing tasks" in r and "RuntimeError" in r and "owner_id=" in r
        for r in records
    ), records


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
            # requests a grouped task-list refresh. Fix wave F3: POSTED, not
            # called -- a direct `_on_conflict_resolved(...)` call is what let
            # the DISPATCH itself break unnoticed when task 5 moved the
            # `ConflictsTab` onto a pushed screen (the live-path pin now lives
            # in `test_schedules_responsive_floor.py`).
            workbench.post_message(
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

        await pilot.app.screen.action_create_reminder()
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

            await pilot.app.screen.action_create_reminder()
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
async def test_mixed_sync_cycle_toasts_success_and_surfaces_the_phase_failure():
    """UAT finding 3c pin: a cycle whose reminder phase succeeded (and,
    per the same-cycle scenario, a definition push too) alongside an
    UNRELATED phase's failure (a business 404) must never toast "Sync
    failed" -- and must never silently drop the failure either. Both
    truths, as separate notices."""
    app = WorkbenchTestAppWithService()
    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()

        notify_calls: list[tuple[str, str]] = []
        pilot.app.notify = lambda message, severity="information", **_: (
            notify_calls.append((severity, str(message)))
        )
        # Isolate the toast-mapping logic under test from the rest of
        # this handler's refresh side effects (already covered by other
        # tests in this file).
        workbench._refresh_owner_select = lambda: None
        workbench._request_tasks_refresh = lambda *a, **k: None
        workbench._refresh_conflicts_badge = lambda: None
        workbench._refresh_results_badge = lambda: None

        outcome = SyncOutcome(
            "ok",
            pulled=0,
            pushed=1,
            phase_errors=("Automation results pull: business 404 (stale server)",),
        )
        workbench._on_sync_completed(SyncCompleted("local", conflict_count=0, outcome=outcome))

        assert len(notify_calls) == 2, notify_calls
        success_severity, success_message = notify_calls[0]
        assert success_severity == "information"
        assert "failed" not in success_message.lower(), (
            "the success toast must never say 'failed'"
        )
        failure_severity, failure_message = notify_calls[1]
        assert failure_severity == "warning"
        assert "Automation results pull" in failure_message
        assert "business 404" in failure_message


def test_reminder_dispatched_relays_to_the_mounted_workbench():
    """UAT finding 3a pin: `TldwCli.on_reminder_dispatched` (the `post_
    message` fallback bridge -- `on_queue_changed` only reaches the
    scheduler loop's own in-memory queue, never a screen) must find a
    live `SchedulesWorkbench` on the screen stack and trigger a refresh,
    the exact route a scheduled reminder fire otherwise has none of."""
    from tldw_chatbook.app import TldwCli

    workbench = SchedulesWorkbench(app_instance=SimpleNamespace(scheduling_service=None))
    refresh_calls: list[bool] = []
    workbench._request_tasks_refresh = lambda *, refresh_definitions=True: (
        refresh_calls.append(refresh_definitions)
    )
    # A plain fake, not a real `App` -- `screen_stack` is a real Textual
    # App's own read-only property, and this relay's only real contract
    # is "reads `self.screen_stack`, finds the workbench, calls its
    # refresh" -- no live app/message pump needed to pin that.
    fake_app = SimpleNamespace(screen_stack=[object(), workbench])

    TldwCli.on_reminder_dispatched(fake_app, ReminderDispatched("task-1"))

    assert refresh_calls == [False], (
        "a fired reminder must trigger a reminder-only refresh without navigating away and back"
    )


def test_reminder_dispatched_is_a_no_op_with_no_workbench_mounted():
    """The relay must not raise when Schedules isn't the active screen."""
    from tldw_chatbook.app import TldwCli

    fake_app = SimpleNamespace(screen_stack=[object()])

    TldwCli.on_reminder_dispatched(fake_app, ReminderDispatched("task-1"))


def test_post_reminder_dispatched_posts_the_message():
    """`SchedulerLoop.on_reminder_dispatched` is wired to this bound
    method (`app.py`'s scheduler-loop construction) -- it must post a
    real `ReminderDispatched` carrying the fired task's id."""
    from tldw_chatbook.app import TldwCli

    posted: list[object] = []
    app = WorkbenchTestAppWithService()
    app.post_message = lambda message: posted.append(message)

    TldwCli._post_reminder_dispatched(app, "task-7")

    assert len(posted) == 1
    assert isinstance(posted[0], ReminderDispatched)
    assert posted[0].task_id == "task-7"


@pytest.mark.asyncio
async def test_reminder_dispatched_message_reaches_the_workbench_through_a_live_pump():
    """Review round 1 finding 3: the three other fanout tests each pin
    one hop in isolation via direct calls (the loop's callback fires;
    `TldwCli.on_reminder_dispatched` called directly against a fake app;
    `_post_reminder_dispatched` checked to call `post_message`) -- none
    of them actually run a message through a live Textual pump. This
    posts a REAL `ReminderDispatched` on a RUNNING app and confirms
    Textual's naming-convention dispatch (`on_reminder_dispatched`, no
    `@on` decorator -- `Message.handler_name` resolves it purely by
    class-name convention) really connects `TldwCli`'s production method
    to a mounted `SchedulesWorkbench` at runtime, not just in theory."""
    from tldw_chatbook.app import TldwCli

    class _FanoutTestApp(WorkbenchTestAppWithService):
        # The REAL production method, unbound onto this lightweight test
        # app -- if Textual's dispatch did not route by name convention,
        # this handler would simply never run.
        on_reminder_dispatched = TldwCli.on_reminder_dispatched

    app = _FanoutTestApp()
    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        refresh_calls: list[bool] = []
        workbench._request_tasks_refresh = lambda *, refresh_definitions=True: (
            refresh_calls.append(refresh_definitions)
        )
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        # `on_mount` itself calls `_request_tasks_refresh()` (default
        # True) as part of normal mounting -- irrelevant to this pin.
        refresh_calls.clear()

        pilot.app.post_message(ReminderDispatched("task-1"))
        await pilot.pause()

        assert refresh_calls == [False], (
            "the live message pump must dispatch ReminderDispatched to "
            "TldwCli.on_reminder_dispatched by naming convention alone"
        )


@pytest.mark.asyncio
async def test_liveness_strip_gets_its_own_faster_interval_timer():
    """UAT finding 3d pin: the scheduler-liveness strip used to be
    repainted only as a side effect of the 60s next-run ticker, so a
    0-30s value was static for up to a full minute between samples. It
    must now be scheduled on its OWN interval, faster than that ticker,
    calling `_refresh_scheduler_liveness` directly."""
    app = WorkbenchTestAppWithService()
    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        scheduled: list[tuple[float, object]] = []
        real_set_interval = workbench.set_interval

        def fake_set_interval(seconds, callback, *a, **k):
            scheduled.append((seconds, callback))
            return real_set_interval(seconds, callback, *a, **k)

        workbench.set_interval = fake_set_interval

        await pilot.app.push_screen(workbench)
        await pilot.pause()

        liveness_timers = [s for s, cb in scheduled if cb == workbench._refresh_scheduler_liveness]
        next_run_timers = [s for s, cb in scheduled if cb == workbench._refresh_next_run_rendering]
        assert liveness_timers == [SCHEDULER_LIVENESS_REFRESH_SECONDS]
        assert next_run_timers == [NEXT_RUN_REFRESH_SECONDS]
        assert SCHEDULER_LIVENESS_REFRESH_SECONDS < NEXT_RUN_REFRESH_SECONDS


@pytest.mark.asyncio
async def test_suspend_and_resume_pause_and_restart_the_liveness_timer_too():
    """The liveness timer must follow the same hidden-clock discipline
    (TASK-23022) the next-run timer already has -- paused while covered,
    resumed (and refreshed immediately) on uncover."""
    app = WorkbenchTestAppWithService()
    async with app.run_test() as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()

        paused = []
        resumed = []
        workbench._liveness_refresh_timer.pause = lambda: paused.append(True)
        workbench._liveness_refresh_timer.resume = lambda: resumed.append(True)
        refreshed = []
        workbench._refresh_scheduler_liveness = lambda: refreshed.append(True)

        workbench.on_screen_suspend()
        assert paused == [True]

        workbench.on_screen_resume()
        assert resumed == [True]
        assert refreshed == [True]


@pytest.mark.asyncio
async def test_action_sync_now_notifies_when_no_service():
    app = WorkbenchTestApp()
    workbench = SchedulesWorkbench(app)
    # Should not crash and should not start a worker
    await workbench.action_sync_now()


@pytest.mark.asyncio
async def test_action_sync_now_guard_prevents_duplicate_workers():
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
    # action_sync_now is now async (fix round 1) -- the _sync_running guard
    # is the very first check, before any await, so this still exercises
    # exactly what the test claims: the guard fires before anything else.
    await workbench.action_sync_now()
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


# --- redesign PR-4, task 5: the retirement -------------------------------


@pytest.mark.asyncio
async def test_the_workbench_composes_one_surface_with_no_tab_chrome():
    """The retirement's headline claim, asserted on the real widget tree.

    The Automations/Conflicts/Results `TabPane`s and the `TabbedContent`
    that held them are gone -- and with only the Queue pane left there
    was no reason to keep the container either, so the queue's three
    panes now hang directly off `#schedules-shell`. That structural
    simplification is the point of the task, so it is pinned rather than
    inferred from the absence of failures: a future edit re-wrapping the
    content in tab chrome would put every one of the retired surfaces'
    problems back.

    Painted, not merely present (`region.width`): a container mounted
    inside a dead `TabPane` would still answer `query_one` while
    rendering at a zero region -- the exact false pass
    `test_schedules_automations_tab.py`'s own detail assertions had to be
    taught about.
    """
    from textual.widgets import TabbedContent, TabPane

    app = _RailTestApp(_RailService())
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        assert list(workbench.query(TabbedContent)) == []
        assert list(workbench.query(TabPane)) == []

        shell = workbench.query_one("#schedules-shell")
        row = workbench.query_one("#scheduling-workbench")
        assert row.parent is shell, (
            "the queue row hangs directly off the shell, not off tab chrome"
        )
        for pane_id in (
            "scheduling-list-pane",
            "scheduling-detail-pane",
            "scheduling-inspector-pane",
        ):
            pane = workbench.query_one(f"#{pane_id}")
            assert pane.region.width > 0, pane_id

        # The retired surfaces' own mounted widgets are gone with them --
        # both views exist only as fresh instances inside a pushed screen.
        assert list(workbench.query(ConflictsTab)) == []
        assert list(workbench.query(ResultsTab)) == []
        # ...and only ONE DefinitionDetail is left (the Automations tab's
        # sibling instance retired with its pane).
        assert [d.id for d in workbench.query(DefinitionDetail)] == [
            "scheduling-queue-definition-detail"
        ]

        # The badges that reach the pushed views are on the surface.
        assert workbench.query_one("#scheduling-conflicts-badge", Button).region.width
        assert workbench.query_one("#scheduling-results-badge", Button).region.width


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
async def test_conflicts_badge_pushes_conflicts_overlay_with_painted_content():
    """redesign PR-4, Task 1: the status strip's conflicts badge mirrors
    `_refresh_conflicts_tab`'s own existing count (no new query) and
    pushes a fresh `ConflictsTab` overlay via `WorkbenchHostScreen` on
    click -- the tab-flip this replaced could not survive the tab bar,
    which Task 5 has now retired. With no tab bar there is nothing left
    to assert a non-flip against; what remains is the claim that always
    mattered: the badge pushes a screen carrying PAINTED conflict rows,
    and Esc returns to the same workbench instance."""
    app = _RailTestApp(
        _RailService(
            conflicts=[{"id": "c1", "local_state": {"title": "Digest"}}]
        )
    )
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        badge = workbench.query_one("#scheduling-conflicts-badge", Button)
        assert str(badge.label) == "Conflicts (1)"
        assert "scheduled task" in str(badge.tooltip).lower()

        badge.press()
        await pilot.pause()

        # A fresh screen is pushed on top of the workbench.
        assert pilot.app.screen is not workbench
        assert isinstance(pilot.app.screen, WorkbenchHostScreen)

        overlay_tab = pilot.app.screen.query_one(
            "#scheduling-conflicts-overlay", ConflictsTab
        )
        overlay_table = overlay_tab.query_one(
            "#scheduling-conflicts-table", DataTable
        )
        # Painted, not stored: what the table would actually render for
        # row 0 (schedules_test_helpers.rendered_row_cells).
        assert rendered_row_cells(overlay_table, 0)[0] == "Digest"

        await pilot.press("escape")
        await pilot.pause()

        # Esc pops back to the same underlying workbench instance.
        assert pilot.app.screen is workbench


@pytest.mark.asyncio
async def test_conflict_resolution_buttons_click_target_is_already_3_rows():
    """Finding 6/Major 10 (root-causes.md, task-3): root-causes' own
    verdict already REFUTES "clicks don't work" (a real synthesized
    mouse click lands fine). It also proposed a residual worth-fixing
    claim -- the buttons are "1 row tall... a 1-row-tall target at a
    screen edge" (`btn.size == Size(16, 1)`, and its own report also
    states `region: height=1`) -- and recommended widening via
    `min-height: 3`.

    Direct measurement against the CURRENT real bundle REFUTES that
    residual claim too: `.size.height` is indeed 1 (content box only),
    but `.region.height` -- the compositor's actual laid-out click
    target, the attribute that matters for "does a click land" -- is
    ALREADY 3 at every screen size probed (80x24, 120x40, 235x52,
    verified interactively). The reason: Textual's own `Button.-style-
    default` DEFAULT_CSS sets `border-top: tall` + `border-bottom: tall`
    (1 row each) around the 1-row content, giving 3 laid-out rows
    regardless of this app's CSS. Adding `min-height: 3` here was
    therefore a no-op (confirmed by toggling it and rebuilding the CSS
    bundle both ways -- `.region.height` stayed 3 either way), so no
    source change was made; this test pins the ACTUAL state (a
    comfortable click target) as a regression guard, not a fix."""
    app = _RailTestApp(
        _RailService(
            conflicts=[{"id": "c1", "local_state": {"title": "Digest"}}]
        )
    )
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        badge = pilot.app.screen.query_one("#scheduling-conflicts-badge", Button)
        badge.press()
        await pilot.pause()

        overlay_tab = pilot.app.screen.query_one(
            "#scheduling-conflicts-overlay", ConflictsTab
        )
        use_server = overlay_tab.query_one("#scheduling-use-server", Button)
        use_local = overlay_tab.query_one("#scheduling-use-local", Button)

        # `.region`, not `.size`: a click's screen coordinates land in
        # `.region` (the laid-out box including border chrome), which is
        # what actually determines "does this click hit the button".
        assert use_server.region.height >= 3, f"got {use_server.region.height}"
        assert use_local.region.height >= 3, f"got {use_local.region.height}"

        # Post-compositor confirmation (root-causes.md's own oracle): both
        # buttons' labels are actually painted, not merely sized on paper.
        strips = pilot.app.screen._compositor.render_strips()
        painted = "\n".join(
            "".join(seg.text for seg in strip) for strip in strips
        )
        assert "Use server" in painted
        assert "Use local" in painted


@pytest.mark.asyncio
async def test_conflicts_badge_defaults_to_no_count():
    app = _RailTestApp(_RailService())
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        badge = pilot.app.screen.query_one("#scheduling-conflicts-badge", Button)
        assert str(badge.label) == "Conflicts"


@pytest.mark.asyncio
async def test_results_badge_shows_unread_count_and_pushes_global_overlay():
    """redesign PR-4, task 2: the rail's `Results (N)` affordance mirrors
    `_refresh_results_badge`'s own existing unread count (no new query,
    the status strip's Conflicts badge's own idiom) and pushes a fresh
    `ResultsHostScreen` overlay. redesign PR-4 task 5 retired the Results
    tab, so this button (and a definition pane's unread row) is the only
    route to the view -- and with no tab bar there is nothing left to
    assert a non-flip against."""
    app = _RailTestApp(_RailService(unread=True))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        badge = workbench.query_one("#scheduling-results-badge", Button)
        assert str(badge.label) == "Results (1)"

        badge.press()
        await pilot.pause()

        # A fresh screen is pushed on top of the workbench.
        assert pilot.app.screen is not workbench
        assert isinstance(pilot.app.screen, ResultsHostScreen)

        overlay_tab = pilot.app.screen.query_one(ResultsTab)
        overlay_table = overlay_tab.query_one("#scheduling-results-table", DataTable)
        assert overlay_table.row_count == 1

        await pilot.press("escape")
        await pilot.pause()

        # Esc pops back to the same underlying workbench instance.
        assert pilot.app.screen is workbench


@pytest.mark.asyncio
async def test_results_badge_defaults_to_no_count():
    app = _RailTestApp(_RailService())
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        badge = pilot.app.screen.query_one("#scheduling-results-badge", Button)
        assert str(badge.label) == "Results"


@pytest.mark.asyncio
async def test_unread_row_activation_pushes_definition_filtered_results_overlay(
    tmp_path,
):
    """redesign PR-4, task 2: the definition pane's `Unread results` row
    is the live replacement for the retired "See Results tab" pointer
    (survey :734-736) -- activating it pushes a `ResultsHostScreen`
    scoped to ONLY that definition's results, and `a` inside it only
    marks THIS definition's unread results read (a sibling definition's
    unread result must survive). Popping (Esc) re-syncs the rail badge
    from the DB (`dismissed`)."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        target_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Weekly digest",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What changed?"},
        )
        other_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Other automation",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "Other?"},
        )
        target_result_id = db.create_automation_result(
            owner_id="local",
            definition_id=target_id,
            run_id="run-1",
            kind="finding",
            title="Digest finding",
            summary="s",
            dedupe_key="d1",
            answer="a",
            source_refs=[],
            review_state="unread",
        )
        db.create_automation_result(
            owner_id="local",
            definition_id=other_id,
            run_id="run-1",
            kind="finding",
            title="Other finding",
            summary="s",
            dedupe_key="d2",
            answer="a",
            source_refs=[],
            review_state="unread",
        )

        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            workbench = pilot.app.screen
            # Fix wave F4: `settle_schedules_workbench` (not a bare
            # pause + wait) -- the mount-time catch-up results pull is a
            # `set_timer` callback `wait_for_complete` does not cover, and
            # its `_request_tasks_refresh` wipes a hand-set definition off
            # the live `#scheduling-queue-definition-detail` pane.
            await settle_schedules_workbench(pilot, workbench)

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            definition = db.get_automation_definition(target_id)
            detail.set_definition(definition, unread_count=1)
            await pilot.pause()

            row = detail._unread_row
            assert row.affordance is True
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()

            assert isinstance(pilot.app.screen, ResultsHostScreen)
            overlay_tab = pilot.app.screen.query_one(ResultsTab)
            overlay_table = overlay_tab.query_one("#scheduling-results-table", DataTable)
            assert overlay_table.row_count == 1
            assert rendered_row_cells(overlay_table, 0)[1] == "Digest finding"

            await pilot.press("a")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_automation_result(target_result_id)["review_state"] == "read"
            other_result = db.list_automation_results(
                owner_id=None, definition_id=other_id
            )[0]
            assert other_result["review_state"] == "unread"

            await pilot.press("escape")
            await pilot.pause()

            assert pilot.app.screen is workbench
            # `dismissed` re-synced the rail badge from the DB: the
            # target definition's unread result is now read, but the
            # other definition's is still unread -> total unread == 1.
            badge = workbench.query_one("#scheduling-results-badge", Button)
            assert str(badge.label) == "Results (1)"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_definition_filtered_overlay_heading_escapes_the_definition_name(
    tmp_path,
):
    """redesign PR-4, task 2: the definition-filtered heading is rendered
    through the same `Static.update(str)` -> `Content.from_markup` parser
    the detail pane's own `escape_markup` guards against (results_tab.py's
    docstring) -- a bracket-bearing definition name must render
    literally, not get eaten as a markup tag."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Digest [urgent]",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What changed?"},
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            workbench = pilot.app.screen

            definition = db.get_automation_definition(definition_id)
            workbench._push_results_overlay(definition=definition)
            await pilot.pause()

            overlay_tab = pilot.app.screen.query_one(ResultsTab)
            heading = str(
                overlay_tab.query_one("#scheduling-results-heading").render()
            ).strip()
            assert "Digest [urgent]" in heading
    finally:
        db.close()


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


# --- redesign PR-3, task 5: owner-row transfer dropdown ---------------------


class _FakeConnectedServerClient:
    """Reads as "a server is connected" (`notifications_service is not
    None`) without making any real network call -- local duplicate of
    `test_schedules_transfer_actions.py`'s own `_FakeServerClient`,
    matching this file's own no-cross-file-test-coupling precedent."""

    def __init__(self) -> None:
        self.notifications_service = object()

    async def get_capabilities(self, *, force: bool = False):
        """task-3 (ruling 4/5): `SchedulesWorkbench.on_mount` kicks a real
        `refresh_server_reachability` probe, which calls this -- without
        it, the mount-time worker's `AttributeError` gets caught as
        "unreachable" and silently overwrites this fixture's whole
        "looks connected" premise (`_server_reachable` back to `False`)
        moments after `_connected_service` pre-seeds it `True`. Accepts
        `force` (Qodo fix round finding 1) since the real probe always
        passes it -- this fake has no cache to bypass."""
        return {}


def _connected_service(tmp_path, app):
    """A real `ScheduledTasksDB` + `SchedulingService`, wired to LOOK
    connected to a server (`active_server_id="1"`, a fake `server_
    client`) -- for tests where the owner-row dropdown must actually
    OFFER a `Server (1)` option (`SchedulesWorkbench._server_available`/
    `SchedulingService._active_server_owner_id` both gate on this)."""
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService

    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    app.active_server_id = "1"
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_server_id="1")
    )
    service = SchedulingService(
        db=db,
        server_client=_FakeConnectedServerClient(),
        runtime_source="local",
        app_getter=lambda: app,
    )
    # task-3 (ruling 4): `transfer_refusal`/`_server_available` now also
    # gate on a real `refresh_server_reachability` probe (default
    # `False`) -- this fixture's whole point is "look connected", so
    # pre-seed the same verdict a probe would reach (same precedent as
    # `test_scheduling_service.py`'s `_transfer_service`).
    service._server_reachable = True
    app.scheduling_service = service
    return db, service


# -- Final review finding 2: the header must not assert "not reachable"
# before any probe has run. ---------------------------------------------


class _SlowProbeServerClient:
    """A server client whose `get_capabilities` blocks on a controllable
    `asyncio.Event` -- lets a test observe the mount-time probe's
    still-pending window deterministically instead of racing real time
    (same idiom `test_schedules_notification_observer.py`'s own
    `_GatedServerClient` uses)."""

    def __init__(self) -> None:
        self.notifications_service = object()
        self.release = asyncio.Event()

    async def get_capabilities(self, *, force: bool = False):
        await self.release.wait()
        return {}


@pytest.mark.asyncio
async def test_header_paints_checking_not_a_false_unreachable_during_mount_probe(
    tmp_path,
):
    """Final review finding 2: `on_mount` calls `_refresh_owner_select()`
    one line before `_refresh_server_reachability()` kicks the probe
    worker, and `server_reachable` used to default `False` (a two-state
    bool with no "unprobed" value) -- so the header asserted "Server
    configured but not reachable" (a negative network fact nothing had
    established) for the whole still-pending window, then silently
    corrected once the worker resolved. `server_reachable` is now
    tri-state (`None` unprobed); the header must paint the existing
    honest "Checking sync status…" copy instead, and correct to the true
    state once the probe (blocked here on a controllable event, not real
    time) actually resolves."""
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
    from tldw_chatbook.UI.Workbench.workbench_widgets import DestinationHeader

    app = WorkbenchTestApp()
    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    app.active_server_id = "1"
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_server_id="1")
    )
    server_client = _SlowProbeServerClient()
    service = SchedulingService(
        db=db,
        server_client=server_client,
        runtime_source="local",
        app_getter=lambda: app,
    )
    app.scheduling_service = service
    try:
        async with app.run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            # No pause yet -- the mount-time probe worker has been
            # scheduled but cannot possibly have resolved (it's blocked
            # on `release`, which nothing has set).
            header = pilot.app.screen.query_one(
                "#schedules-destination-header", DestinationHeader
            )
            assert header.state.status_label == "Checking sync status…", (
                f"got {header.state.status_label!r} -- must not assert "
                "reachability nothing has established yet"
            )
            assert "not reachable" not in header.state.status_label

            server_client.release.set()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert header.state.status_label == "Local schedules", (
                f"got {header.state.status_label!r} -- must correct once "
                "the probe actually resolves (owner stayed 'local' "
                "throughout this test, so a resolved reachable server "
                "reads 'Local schedules', not 'Synced with server' -- the "
                "point under test is that it is no longer 'Server "
                "configured but not reachable')"
            )
    finally:
        db.close()


# -- Fix round 1, finding 1: _on_owner_server / action_sync_now must
# re-probe like _run_owner_transfer already did, not trust a stale
# mount-time `server_reachable` while the background probe is still
# pending. -------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_owner_server_reprobes_during_the_mount_probe_window(tmp_path):
    """A genuinely configured, about-to-be-confirmed server must not get a
    false "No server connection" refusal during the on-mount probe's
    still-pending window. `_connected_service` wires a REAL, working
    probe (`_FakeConnectedServerClient.get_capabilities` succeeds) --
    `server_reachable` is forced back to its honest pre-probe default
    right before the click (the mount-time background worker may already
    have resolved it by then; this isolates the handler's OWN re-probe as
    what's under test). `_set_owner` itself is spied out: its downstream
    runtime-policy write needs a real `RuntimePolicyContext` this test's
    fixtures don't build, and that machinery is unrelated to what's under
    test here -- whether `_on_owner_server` reaches `_set_owner` at all,
    not whether the write it performs afterward succeeds."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        async with app.run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            service._server_reachable = False  # simulate: probe still pending

            workbench = pilot.app.screen
            set_owner_calls: list[str] = []
            workbench._set_owner = set_owner_calls.append
            notify_calls: list[str] = []
            app.notify = lambda message, **kwargs: notify_calls.append(message)

            server_button = workbench.query_one("#scheduling-owner-server", Button)
            server_button.press()
            await pilot.pause()

            assert set_owner_calls == ["server:1"], (
                "the re-probe must resolve to reachable before deciding, "
                "not refuse on the stale pending default"
            )
            assert notify_calls == [], (
                f"must not show the false refusal notice, got {notify_calls!r}"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_action_sync_now_reprobes_during_the_mount_probe_window(tmp_path):
    """Same probe-window bug as `_on_owner_server`, for the `s` key --
    `action_sync_now` used to refuse ("Local only -- nothing to sync")
    off a stale `server_reachable=False` without re-probing first."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        async with app.run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            service._server_reachable = False  # simulate: probe still pending
            workbench = pilot.app.screen

            await workbench.action_sync_now()

            assert workbench._sync_running is True, (
                "the re-probe must resolve to reachable before deciding, "
                "not refuse on the stale pending default"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_action_create_reminder_reprobes_during_the_mount_probe_window(
    tmp_path,
):
    """Final review finding 3: `_runs_on_options` (read by `action_
    create_reminder`/`action_create_automation`) was the one gate site
    fix round 1 did not re-probe -- a create during the mount probe's
    still-pending window silently offered only "This device" in the
    runs-on Select, with no notice and no correction while the form
    stayed open."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        async with app.run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            service._server_reachable = False  # simulate: probe still pending

            await pilot.app.screen.action_create_reminder()
            await pilot.pause()

            form = pilot.app.screen
            assert isinstance(form, ReminderForm)
            runs_on = form.query_one("#reminder-runs-on", Select)
            option_values = [value for _label, value in runs_on._options]
            assert "server:1" in option_values, (
                f"got {option_values!r} -- the re-probe must resolve to "
                "reachable before building the runs-on options, not "
                "silently offer only 'This device'"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_on_mount_settles_orphaned_transfer_without_any_sync_running(
    tmp_path,
):
    """Final review finding 1's own pinned scenario: a reminder is queued
    `to_server_pending` while a server was configured, then the server is
    REMOVED entirely (`active_server_id -> None`) -- the UAT's actual
    core symptom, not merely a switch to another reachable server.

    The row must settle to `to_server_failed` from `on_mount` ALONE, with
    no sync ever running: no server is configured, so `_server_
    configured`/`_server_available` both read `False`, `_start_server_
    notification_observer`/`_schedule_catch_up_results_pull` both no-op,
    and nothing in this test ever calls `action_sync_now`/presses `s` --
    `_settle_orphaned_transfers` (kicked unconditionally from `on_mount`,
    running on its own deferred worker since the Qodo fix round finding 2
    fix -- `wait_for_complete()` below lets it actually finish) is the
    only mechanism that could possibly settle it.
    """
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    local_id = db.create_reminder_task(
        owner_id="local",
        title="Standup",
        schedule_kind="one_time",
        run_at="2099-01-01T00:00:00+00:00",
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",  # the server _connected_service just configured
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
        },
    )
    # The server is removed entirely.
    app.runtime_policy.state.active_server_id = None
    try:
        async with app.run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            row = db.get_reminder_task(local_id)
            assert row["transfer_state"] == "to_server_failed", (
                "must settle from on_mount alone -- no server is "
                "configured, so no sync could ever have run"
            )
            pending = db.get_pending_mutations(
                "server:1", primitive="reminder_task"
            )
            assert len(pending) == 1
            assert pending[0]["payload"]["transfer_errors"]
    finally:
        db.close()


def test_settle_orphaned_transfers_defers_to_a_worker_not_inline(tmp_path):
    """Qodo fix round finding 2 (MEDIUM) pin: `_settle_orphaned_transfers`
    must hand the actual local-DB sweep to a worker and return
    immediately, never run it inline on the calling (UI) thread --
    `run_worker` itself is substituted here so the assertion holds
    regardless of when a real worker would actually get scheduled to
    run (the concern the app-level test above can only prove after the
    fact, via `wait_for_complete()`)."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        workbench = SchedulesWorkbench(app_instance=app)

        calls: list[str | None] = []
        service.sync_engine._settle_orphaned_transfer_mutations = (
            lambda target_owner: calls.append(target_owner)
        )

        captured = {}

        def _fake_run_worker(work, *, exclusive=False, group=None, **_kwargs):
            captured["work"] = work
            captured["group"] = group

        workbench.run_worker = _fake_run_worker

        workbench._settle_orphaned_transfers()

        assert calls == [], "must not run the sweep inline before returning"
        assert captured["group"] == "schedules-orphan-sweep"
        assert captured["work"] is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_action_sync_now_splits_configured_unreachable_from_unconfigured(
    tmp_path,
):
    """Final review finding 4: `action_sync_now`'s refusal used the
    conflated "Local only -- nothing to sync (no server connection)."
    copy even when a server WAS configured (merely unreachable) -- the
    exact conflation the branch removed two surfaces away (the header's
    own split, `transfer_refusal`'s own distinct reason). A configured-
    but-unreachable server must name THAT, not "no server connection"."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    service._server_reachable = False  # confirmed unreachable, not unprobed
    try:
        async with app.run_test() as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            service._server_reachable = False  # re-assert after mount's probe

            # `_FakeConnectedServerClient.get_capabilities` always
            # succeeds, so force the re-probe itself to fail this once --
            # a genuinely unreachable server, not merely an unprobed one.
            async def _fail_reachability():
                service._server_reachable = False
                return False

            service.refresh_server_reachability = _fail_reachability

            workbench = pilot.app.screen
            await workbench.action_sync_now()
            await pilot.pause()

            messages = [n.message for n in pilot.app._notifications]
            assert any(
                "Server configured but not reachable" in m for m in messages
            ), messages
            assert not any("no server connection" in m for m in messages), messages
    finally:
        db.close()


# -- bare-harness: widget mechanics, no service needed -----------------------


@pytest.mark.asyncio
async def test_runs_on_dropdown_opens_with_current_owner_preselected():
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _frequency_reminder(owner_id="local"),
            runs_on_options=[("This device", "local"), ("Server (1)", "server:1")],
        )
        await pilot.pause()

        row = detail._runs_on_row
        await pilot.click(row)
        await pilot.pause()
        select = row.query_one(Select)
        assert select.value == "local"


@pytest.mark.asyncio
async def test_definition_runs_on_dropdown_opens_with_current_owner_preselected():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(
            _editable_definition(owner_id="server:9"),
            runs_on_options=[("This device", "local")],
        )
        await pilot.pause()

        row = detail._runs_on_row
        await pilot.click(row)
        await pilot.pause()
        select = row.query_one(Select)
        # `server:9` isn't in the base options -- the row's own current
        # owner is appended as a fallback (survey §7's "never a value
        # outside a Select's own options" precedent) so it still preselects.
        assert select.value == "server:9"


@pytest.mark.asyncio
async def test_runs_on_same_owner_selection_is_a_no_op():
    """The mount-time synthetic `Changed` `begin_edit` posts the moment
    it preselects the row's OWN current owner (same trap Task 3's Repeat/
    Timezone commits already guard against, reused verbatim) must not
    close the editor or post a transfer request -- same as those rows,
    the editor stays open for a real pick rather than self-closing.

    Final review F8/M8 ADJUDICATION. The controller's ruling was "the
    CODE is wrong -- close the editor on a same-owner pick, and update
    this test"; re-probed against Textual 8.2.8, that ruling is not
    implementable and the shipped behavior stands:

    * the synthetic mount `Changed` is real -- `Select.value` is
      `var(NULL, init=False)`, so `_on_mount`'s `_init_selected_option`
      assignment is a genuine change and posts `Changed`. Probed: the
      commit handler fires with `"local"` before any user input, and an
      `end_edit()` there leaves `editor still open: False` -- the
      dropdown shuts the instant it opens and the owner picker becomes
      unusable;
    * a genuine re-pick of the SAME option posts nothing at all
      (`Select._update_selection` assigns only `if value != self.value`),
      so a "close on a same-owner pick" branch has no reachable caller to
      hang off in the first place.

    The docstring that claimed otherwise is what got fixed instead
    (`TaskDetail._commit_runs_on_edit`). Escape is the cancel path.
    """
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _frequency_reminder(owner_id="local"),
            runs_on_options=[("This device", "local"), ("Server (1)", "server:1")],
        )
        await pilot.pause()

        commits: list[str] = []
        real_commit = TaskDetail._commit_runs_on_edit

        def _spy(self, event):
            commits.append(str(event.value))
            return real_commit(self, event)

        with patch.object(TaskDetail, "_commit_runs_on_edit", _spy), patch.object(
            detail, "post_message", wraps=detail.post_message
        ) as post_spy:
            row = detail._runs_on_row
            await pilot.click(row)
            await pilot.pause()

            # The mount echo is real and DOES reach the commit path with
            # the row's own current owner -- the whole reason that branch
            # exists. Pinned, so a later "simplification" that closes on
            # it fails here rather than in the user's hands.
            assert commits == ["local"]
            assert row.query(Select)
            assert not any(
                isinstance(call.args[0], ReminderOwnerActionRequested)
                for call in post_spy.call_args_list
            )


@pytest.mark.asyncio
async def test_runs_on_escape_cancels_dropdown_without_posting():
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _frequency_reminder(owner_id="local"),
            runs_on_options=[("This device", "local"), ("Server (1)", "server:1")],
        )
        await pilot.pause()

        with patch.object(
            detail, "post_message", wraps=detail.post_message
        ) as post_spy:
            row = detail._runs_on_row
            await pilot.click(row)
            await pilot.pause()
            assert row.query(Select)
            await pilot.press("escape")
            await pilot.pause()

            assert not row.query(Select)
            assert not any(
                isinstance(call.args[0], ReminderOwnerActionRequested)
                for call in post_spy.call_args_list
            )


@pytest.mark.asyncio
async def test_runs_on_row_reflects_transfer_state():
    """Cancel/Retry button visibility across all four `transfer_state`
    values (task-5 brief), mirroring the existing per-button visibility
    rules `test_schedules_transfer_actions.py` already pins. Affordance
    stays ON unconditionally in EVERY state (fix round 1 finding 2:
    ruling 3's "dropdown always renders" -- a locked/failed row's
    activation still reaches `on_detail_value_row_activated`, which is
    what decides dropdown-vs-show-why now, not a disabled affordance;
    see the dedicated activation tests below)."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)

        for transfer_state, cancel_visible, retry_visible in (
            (None, False, False),
            ("to_server_pending", True, False),
            ("to_server_failed", True, True),
            ("from_server_pending", True, False),
            (None, False, False),  # back to normal
        ):
            detail.set_task(_frequency_reminder(transfer_state=transfer_state))
            await pilot.pause()
            assert detail._runs_on_row.affordance is True
            assert detail._runs_on_cancel_button.display is cancel_visible
            assert detail._runs_on_retry_button.display is retry_visible


@pytest.mark.asyncio
async def test_runs_on_activation_on_in_flight_row_shows_lock_reason_not_dropdown():
    """Fix round 1 finding 2: an in-flight row's activation now shows
    its `_lifecycle_lock_reason` -- the SAME reason
    `transfer_lock_reason` computes and the Frequency rows already
    surface on click -- instead of going silently inert or opening a
    dropdown with nothing sensible to offer."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder(transfer_state="to_server_pending"))
        detail.set_lifecycle_lock(
            "This row is moving between this device and the server."
        )
        await pilot.pause()

        row = detail._runs_on_row
        await pilot.click(row)
        await pilot.pause()

        assert not row.query(Select)
        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert (
            "moving between this device and the server"
            in error.render_line(0).text
        )


@pytest.mark.asyncio
async def test_runs_on_activation_on_failed_row_shows_stored_transfer_error():
    """Fix round 1 finding 2: the SAME "Last transfer error: …" copy the
    legacy Retry button already renders (`set_transfer_reasons`), now on
    the Runs-on row too -- fed via `set_runs_on_transfer_errors`."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder(transfer_state="to_server_failed"))
        detail.set_runs_on_transfer_errors(["Connection refused"])
        await pilot.pause()

        row = detail._runs_on_row
        await pilot.click(row)
        await pilot.pause()

        assert not row.query(Select)
        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert (
            error.render_line(0).text.strip()
            == "Last transfer error: Connection refused"
        )


@pytest.mark.asyncio
async def test_runs_on_activation_on_failed_row_without_stored_errors_falls_back():
    """No stored `transfer_errors` (e.g. a pre-existing failed row) still
    surfaces SOMETHING rather than going silent."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder(transfer_state="to_server_failed"))
        await pilot.pause()

        row = detail._runs_on_row
        await pilot.click(row)
        await pilot.pause()

        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert "Transfer failed" in error.render_line(0).text


@pytest.mark.asyncio
async def test_definition_runs_on_row_reflects_transfer_state():
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)

        detail.set_definition(_editable_definition(transfer_state=None))
        await pilot.pause()
        assert detail._runs_on_row.affordance is True
        assert detail._runs_on_cancel_button.display is False
        assert detail._runs_on_retry_button.display is False

        detail.set_definition(
            _editable_definition(transfer_state="from_server_pending")
        )
        await pilot.pause()
        assert detail._runs_on_row.affordance is True
        assert detail._runs_on_cancel_button.display is True
        assert detail._runs_on_retry_button.display is False

        detail.set_definition(_editable_definition(transfer_state="to_server_failed"))
        await pilot.pause()
        assert detail._runs_on_row.affordance is True
        assert detail._runs_on_cancel_button.display is True
        assert detail._runs_on_retry_button.display is True


@pytest.mark.asyncio
async def test_definition_runs_on_activation_on_in_flight_row_shows_lock_reason():
    """Definition-pane counterpart of the reminder-pane lock-reason test
    (fix round 1 finding 2)."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(
            _editable_definition(transfer_state="from_server_pending")
        )
        detail.set_lifecycle_lock("Read-only mid-transfer.")
        await pilot.pause()

        row = detail._runs_on_row
        await pilot.click(row)
        await pilot.pause()

        assert not row.query(Select)
        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert error.render_line(0).text.strip() == "Read-only mid-transfer."


@pytest.mark.asyncio
async def test_definition_runs_on_activation_on_failed_row_shows_stored_transfer_error():
    """Definition-pane counterpart of the reminder-pane stored-error test
    (fix round 1 finding 2)."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition(transfer_state="to_server_failed"))
        detail.set_runs_on_transfer_errors(["Server rejected the request"])
        await pilot.pause()

        row = detail._runs_on_row
        await pilot.click(row)
        await pilot.pause()

        assert not row.query(Select)
        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is True
        assert (
            error.render_line(0).text.strip()
            == "Last transfer error: Server rejected the request"
        )


# -- integration: real DB + service + full workbench -------------------------


@pytest.mark.asyncio
async def test_runs_on_dropdown_refusal_renders_inline_with_health_reason(tmp_path):
    """A refused target renders via `row.show_error` (this row's OWN
    surface, never a pane-level shared notice) -- health-quoting
    preserved verbatim (spec §6.4/§7), and nothing is written.

    `transfer_refusal`'s FIRST gate ("No server connection is
    configured.") applies to EVERY direction, not only `to_server` --
    `_connected_service` is required here too, purely to get PAST that
    gate and reach the `to_local`/`recurring_question` health check this
    test actually targets.
    """
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        definition_id = db.create_automation_definition(
            "server:1",
            "recurring_question",
            "Server automation",
            server_id="srv-1",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What shipped?"},
            config={
                "generation_mode": "optional",
                "scope": {"mode": "all_searchable_library"},
            },
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()

            # redesign PR-4 task 5: this dict is painted onto the pane by
            # HAND, and that pane is now the same instance the QUEUE's own
            # loader repaints. This fixture's queue is empty (a
            # `server_id`-carrying row belongs to the server half, which
            # `_FakeConnectedServerClient` cannot list), so any background
            # reload lands `set_definition(None)` on the pane -- clearing
            # the hand-painted definition and closing the open editor
            # mid-test. The mount-time catch-up results pull is exactly
            # such a reload (a 0.3s debounce timer, so `wait_for_complete`
            # does not cover it). The retired Automations-tab
            # `DefinitionDetail` sat outside that loader's reach, which is
            # why none of this was needed before. redesign PR-4 task 6 hit
            # the same race with its pushed panes, which is where the
            # inline fix this test used to carry graduated into the shared
            # `settle_schedules_workbench` helper.
            workbench = pilot.app.screen
            await settle_schedules_workbench(pilot, workbench)

            # A server-fetch-shaped dict (Task 4's own documented trap,
            # its report's "trap hit while writing the lifecycle/offline
            # tests": `id` IS the server's own id) -- painted directly
            # onto the `DefinitionDetail` instance rather than driving a
            # real server-list fetch (out of this test's scope), matching
            # what a pure server row hands this pane.
            server_item = dict(db.get_automation_definition(definition_id) or {})
            server_item["id"] = "srv-1"
            detail = pilot.app.screen.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            detail.set_definition(
                server_item, runs_on_options=[("This device", "local")]
            )
            await pilot.pause()

            row = detail._runs_on_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            # "This device" (`to_local`) is always offered, server or not.
            select.value = "local"
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            error = row.query_one(".detail-value-row-error", Static)
            assert error.display is True
            assert (
                "Library RAG search is not available"
                in error.render_line(0).text
            )
            assert db.get_automation_definition(definition_id)["transfer_state"] is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runs_on_dropdown_confirm_dialog_lists_warnings(tmp_path):
    """Allowed -> the SAME `ConfirmationDialog` + `transfer_warnings`
    shape the legacy buttons already use (imminent run_at + the
    non-transferring `timeout_seconds` field, spec §6.4)."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        imminent = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
        db.create_reminder_task(
            owner_id="local",
            title="Almost due",
            schedule_kind="one_time",
            run_at=imminent,
            timeout_seconds=30,
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            row = detail._runs_on_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "server:1"
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            message = pilot.app.screen.message
            assert "Almost due" in message
            assert "unverified" in message
            assert "timeout_seconds" in message
            pilot.app.screen.dismiss(False)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runs_on_dropdown_confirm_begins_to_local_transfer_with_dormant_copy_id(
    tmp_path,
):
    """Task-5 brief pinned case: `begin_transfer_to_local`'s outcome
    carries the NEW dormant copy's own id, DISTINCT from the mirror's --
    the mirror itself stays untouched (survey §3's `create_local_copy_
    from_mirror` precedent). `_connected_service` -- `transfer_refusal`'s
    FIRST gate ("No server connection is configured.") applies to `to_
    local` too, not only `to_server`."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        mirror_id = db.create_reminder_task(
            owner_id="server:1",
            server_id="srv-9",
            title="Server task",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            row = detail._runs_on_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "local"
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            pilot.app.screen.dismiss(True)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            mirror = db.get_reminder_task(mirror_id)
            assert mirror["transfer_state"] is None
            assert mirror["owner_id"] == "server:1"

            local_rows = [
                row_dict
                for row_dict in db.list_reminder_tasks(owner_id="local")
                if row_dict["id"] != mirror_id
            ]
            assert len(local_rows) == 1
            copy = local_rows[0]
            assert copy["id"] != mirror_id
            assert copy["transfer_state"] == "from_server_pending"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runs_on_dropdown_confirm_begins_to_server_transfer(tmp_path):
    """The other direction: confirm fires `begin_transfer_to_server` with
    the currently-displayed row's own local id (task-5 brief: 'the right
    facade call with the right row id per direction')."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="Nightly check",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            row = detail._runs_on_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "server:1"
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            pilot.app.screen.dismiss(True)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            row_after = db.get_reminder_task(task_id)
            assert row_after["transfer_state"] == "to_server_pending"
            assert row_after["owner_id"] == "local"  # still local while queued
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runs_on_cancel_button_cancels_the_dormant_copy_using_its_own_id(
    tmp_path,
):
    """Same pinned mechanism `test_cancel_on_dormant_copy_uses_the_copys_
    own_id` (`test_schedules_transfer_actions.py`) proves for the legacy
    Cancel button -- proven again here for the NEW row-level Cancel
    affordance (task-5 brief: 'assert it'). `_connected_service` is only
    needed for the SETUP call below (`begin_transfer_to_local` itself
    checks `transfer_refusal`'s server-connection gate); the actual
    cancel path (`cancel_transfer`) never checks it."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        mirror_id = db.create_reminder_task(
            owner_id="server:1",
            server_id="srv-9",
            title="Server task",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
        )
        outcome = await service.begin_transfer_to_local("reminder_task", mirror_id)
        assert outcome.status == "pending"
        copy_id = outcome.row_id
        assert copy_id is not None
        assert copy_id != mirror_id

        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 2
            workbench = pilot.app.screen
            copy_index = next(
                index
                for index, row in enumerate(workbench._visible_rows)
                if row.kind == "reminder" and row.source_row.id == copy_id
            )
            table.cursor_coordinate = (copy_index, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            assert detail._runs_on_cancel_button.display is True
            detail._runs_on_cancel_button.press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_reminder_task(copy_id) is None
            mirror = db.get_reminder_task(mirror_id)
            assert mirror is not None
            assert mirror["transfer_state"] is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runs_on_retry_button_rebegins_a_failed_transfer(tmp_path):
    """`to_server_failed`: Retry = re-begin (the PR-5 retry leg) -- same
    facade call a first-time `to_server` begin makes."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="Nightly check",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
        )
        db.set_transfer_state(
            "reminder_task", task_id, "to_server_failed", expected=(None,)
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            assert detail._runs_on_retry_button.display is True
            detail._runs_on_retry_button.press()
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            pilot.app.screen.dismiss(True)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_reminder_task(task_id)["transfer_state"] == "to_server_pending"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_legacy_transfer_buttons_are_retired(tmp_path):
    """redesign PR-4 task 4 (ruling 2): supersedes `test_runs_on_dropdown_
    and_legacy_transfer_buttons_coexist` -- the PR-3 task 5 "coexistence
    pinned" window (task-5 brief) is over, and the legacy Move/Retry/
    Cancel buttons this test used to exercise ALONGSIDE the dropdown are
    deleted. Begin/cancel via the dropdown itself stays pinned by `test_
    runs_on_dropdown_confirm_begins_to_server_transfer`/`test_runs_on_
    cancel_button_cancels_the_dormant_copy_using_its_own_id` (this file);
    this only pins that the retired ids are actually gone and the
    dropdown's own affordance is unaffected."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        db.create_reminder_task(
            owner_id="local",
            title="Nightly check",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            for legacy_id in (
                "scheduling-transfer-to-server",
                "scheduling-transfer-to-local",
                "scheduling-retry-transfer",
                "scheduling-cancel-transfer",
            ):
                assert not detail.query(f"#{legacy_id}")
            # The Runs-on row's own affordance is the one transfer
            # surface now, unaffected by the legacy buttons' removal.
            assert detail._runs_on_row.affordance is True
    finally:
        db.close()


# -- definition-pane twins of the deep reminder-side tests above (fix
# round 1, finding 3: coverage gap -- finding 1 lived exactly here) --------


def _stub_ready_health(monkeypatch) -> None:
    """A `recurring_question` `to_local` refusal also gates on local
    health (spec §6.4/§7.4) -- irrelevant to what these tests exercise
    (the transfer machine's own routing) -- stubbed ready, mirroring
    `test_schedules_transfer_actions.py`'s own `_stub_ready_health`
    exactly (duplicated locally per this file's own no-cross-file-test-
    coupling precedent)."""
    monkeypatch.setattr(
        scheduling_service_module,
        "compute_local_health",
        lambda app, row: ("ready", ""),
    )


@pytest.mark.asyncio
async def test_definition_owner_action_marks_stale_before_resolving(
    tmp_path, monkeypatch
):
    """Fix round 1 finding 1, pinned via ORDERING, not a post-hoc flag
    read: a real mounted workbench has its own background refreshes
    (mount-time `load_tasks`, the catch-up results pull, `on_screen_
    resume`'s `_consume_definitions_stale` -- redesign PR-4 task 5 retired
    the `TabActivated` refresh this used to name) that legitimately
    consume/clear `_definitions_stale` too, so asserting the flag's
    value some time AFTER the action is racy against those -- this pins
    the actual claim instead: `self._definitions_stale` is ALREADY
    `True` by the time `_resolve_local_definition_id` runs, proving the
    write happens unconditionally BEFORE resolving (the exact ordering
    `_begin_automation_transfer`/`_cancel_automation_transfer` already
    use, schedules_workbench.py:2766-2778) rather than only in `_refresh`
    on a later success path.

    `_resolve_local_definition_id` can itself mirror a brand-new local
    row (`upsert_automation_definitions_from_server`) the FIRST time a
    pure server-fetch definition is touched -- no existing local shadow
    for this definition id, forcing that mirror path (every OTHER test
    in this file happens to hit the "already mirrored" fast path, which
    is exactly why this gap went untested)."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()

            workbench = pilot.app.screen
            observed_stale_at_resolve: list[bool] = []
            original_resolve = workbench._resolve_local_definition_id

            async def _spy_resolve(service_arg, definition_arg):
                observed_stale_at_resolve.append(workbench._definitions_stale)
                return await original_resolve(service_arg, definition_arg)

            monkeypatch.setattr(
                workbench, "_resolve_local_definition_id", _spy_resolve
            )

            # A pure server-fetch dict (Task 4's own documented trap):
            # `id` IS the server's own id, and NO local row exists for it
            # yet anywhere in the DB.
            server_item = {
                "id": "srv-brand-new",
                "owner_id": "server:1",
                "server_id": "srv-brand-new",
                "family": "recurring_question",
                "name": "Brand new server automation",
                "lifecycle": "configured",
                "schedule": {"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
                "config": {"scope": {"mode": "all_searchable_library"}},
            }
            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            row = detail._runs_on_row

            workbench._definition_owner_action(server_item, "to_local", row)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            # The spy saw `_definitions_stale` already True the moment
            # `_resolve_local_definition_id` was entered -- proves the
            # unconditional-before-resolving placement, not just "True
            # at some point".
            assert observed_stale_at_resolve == [True]

            # Refused (no ready health wired in this bare app), and the
            # mirror this call created is real.
            error = row.query_one(".detail-value-row-error", Static)
            assert error.display is True
            mirrored = db.get_automation_definition_by_server_id(
                "server:1", "srv-brand-new"
            )
            assert mirrored is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_definition_runs_on_dropdown_confirm_dialog_lists_warnings(tmp_path):
    """Definition-pane twin of `test_runs_on_dropdown_confirm_dialog_
    lists_warnings` -- `to_server` never triggers the family/health
    check (only `to_local` does), so no health stub is needed here."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        imminent = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
        db.create_automation_definition(
            "local",
            "recurring_question",
            "Almost due automation",
            schedule={"kind": "one_time", "run_at": imminent},
            input={"question": "What shipped?"},
            config={"scope": {"mode": "all_searchable_library"}},
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            row = detail._runs_on_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "server:1"
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            message = pilot.app.screen.message
            assert "Almost due automation" in message
            assert "unverified" in message
            pilot.app.screen.dismiss(False)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_definition_runs_on_dropdown_confirm_begins_to_local_transfer_with_dormant_copy_id(
    tmp_path, monkeypatch
):
    """Definition-pane twin of the reminder-side dormant-copy-id test --
    `to_local` on a `recurring_question` row needs `_stub_ready_health`
    (irrelevant to what this test targets)."""
    _stub_ready_health(monkeypatch)
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        mirror_id = db.create_automation_definition(
            "server:1",
            "recurring_question",
            "Server automation",
            server_id="srv-9",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What shipped?"},
            config={"scope": {"mode": "all_searchable_library"}},
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            # Fix wave F4: this body was repointed onto the LIVE
            # `#scheduling-queue-definition-detail` pane in task 5, which
            # the workbench's own mount-time `load_tasks` repaints -- with
            # an empty queue that means `set_definition(None)`, wiping the
            # definition set below (and its editable rows) before the test
            # activates the Runs-on row. The final review measured this
            # body at 7 failed / 3 passed over 10 solo runs without this
            # line (10 passed at BASE, where it still pointed at the
            # retired pane); 10/10 green with it.
            await settle_schedules_workbench(pilot, pilot.app.screen)

            mirror = db.get_automation_definition(mirror_id)
            detail = pilot.app.screen.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            detail.set_definition(mirror, runs_on_options=[("This device", "local")])
            await pilot.pause()

            row = detail._runs_on_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "local"
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            pilot.app.screen.dismiss(True)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            mirror_after = db.get_automation_definition(mirror_id)
            assert mirror_after["transfer_state"] is None
            assert mirror_after["owner_id"] == "server:1"

            local_rows = [
                row_dict
                for row_dict in db.list_automation_definitions(owner_id="local")
                if row_dict["id"] != mirror_id
            ]
            assert len(local_rows) == 1
            copy = local_rows[0]
            assert copy["id"] != mirror_id
            assert copy["transfer_state"] == "from_server_pending"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_definition_runs_on_dropdown_confirm_begins_to_server_transfer(tmp_path):
    """Definition-pane twin of the reminder-side `to_server` test."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest automation",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={"scope": {"mode": "all_searchable_library"}},
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            row = detail._runs_on_row
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            select = row.query_one(Select)
            select.value = "server:1"
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            pilot.app.screen.dismiss(True)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            row_after = db.get_automation_definition(definition_id)
            assert row_after["transfer_state"] == "to_server_pending"
            assert row_after["owner_id"] == "local"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_definition_runs_on_cancel_button_cancels_the_dormant_copy_using_its_own_id(
    tmp_path, monkeypatch
):
    """Definition-pane twin of the reminder-side dormant-copy Cancel
    test."""
    _stub_ready_health(monkeypatch)
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        mirror_id = db.create_automation_definition(
            "server:1",
            "recurring_question",
            "Server automation",
            server_id="srv-9",
            schedule={"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            input={"question": "What shipped?"},
            config={"scope": {"mode": "all_searchable_library"}},
        )
        outcome = await service.begin_transfer_to_local(
            "automation_definition", mirror_id
        )
        assert outcome.status == "pending"
        copy_id = outcome.row_id
        assert copy_id is not None
        assert copy_id != mirror_id

        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            # The local half only ever shows a `server_id`-less row
            # (survey §3): the copy, not the mirror -- the mirror is the
            # SERVER fetch half's row, and this harness's fake server
            # client has no `list_automation_definitions` (out of this
            # test's scope; the mirror-side assertion below reads the DB
            # directly instead of depending on the table showing it).
            assert table.row_count == 1
            copy_index = next(
                index
                for index, unified_row in enumerate(workbench._visible_rows)
                if unified_row.kind == "definition"
                and str(unified_row.source_row.get("id")) == copy_id
            )
            table.cursor_coordinate = (copy_index, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            assert detail._runs_on_cancel_button.display is True
            detail._runs_on_cancel_button.press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_automation_definition(copy_id) is None
            mirror = db.get_automation_definition(mirror_id)
            assert mirror is not None
            assert mirror["transfer_state"] is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_definition_runs_on_retry_button_rebegins_a_failed_transfer(tmp_path):
    """Definition-pane twin of the reminder-side Retry test."""
    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest automation",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={"scope": {"mode": "all_searchable_library"}},
        )
        db.set_transfer_state(
            "automation_definition", definition_id, "to_server_failed",
            expected=(None,),
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            assert table.row_count == 1
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            assert detail._runs_on_retry_button.display is True
            detail._runs_on_retry_button.press()
            await pilot.pause()

            assert isinstance(pilot.app.screen, ConfirmationDialog)
            pilot.app.screen.dismiss(True)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            row_after = db.get_automation_definition(definition_id)
            assert row_after["transfer_state"] == "to_server_pending"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_automations_tab_transfer_keybindings_are_retired(tmp_path):
    """redesign PR-4 task 4 (ruling 2): supersedes `test_definition_runs_
    on_dropdown_and_automations_tab_keybindings_coexist` -- the
    Automations-tab-only `m`/`M`/`y`/`k` keybindings and their `action_
    move_automation_to_local`/`_to_server`/`action_retry_automation_
    transfer`/`action_cancel_automation_transfer`/`_begin_automation_
    transfer`/`_cancel_automation_transfer` flow are genuinely gone --
    the Runs-on dropdown (already exercised end-to-end by `test_
    definition_runs_on_dropdown_confirm_begins_to_server_transfer`/`test_
    definition_runs_on_cancel_button_cancels_the_dormant_copy_using_its_
    own_id` above, including on this SAME Automations-tab `DefinitionDetail`
    instance) is the one transfer surface now."""
    for legacy_action in (
        "action_move_automation_to_local",
        "action_move_automation_to_server",
        "action_retry_automation_transfer",
        "action_cancel_automation_transfer",
    ):
        assert not hasattr(SchedulesWorkbench, legacy_action)
    bound_keys = {binding.key for binding in SchedulesWorkbench.BINDINGS}
    assert bound_keys.isdisjoint({"M", "y", "k"})

    app = WorkbenchTestApp()
    db, service = _connected_service(tmp_path, app)
    try:
        db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest automation",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={"scope": {"mode": "all_searchable_library"}},
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            workbench = pilot.app.screen
            # redesign PR-4 task 5: the Automations tab this used to
            # activate first is retired -- the definition row is
            # selected in the unified queue table instead, and the
            # pane under test is the queue's own `DefinitionDetail`
            # (the sibling instance that always shared this code).
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.cursor_coordinate = (0, 0)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            # The Runs-on dropdown still activates fine on the
            # Automations tab's own DefinitionDetail instance -- the
            # retired keybindings' removal did not touch it.
            detail = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            assert detail._runs_on_row.affordance is True
    finally:
        db.close()


# --- redesign PR-3, final review fix wave (I2/I3/I4, M5, M12) --------------


@pytest.mark.asyncio
async def test_repaint_with_another_task_closes_the_open_editor():
    """Final review F2/I2. The reminder pane repaints on every tick and
    `_update_detail_for_index` falls back to index 0 whenever the selected
    row leaves the filter -- so an open editor could end up mounted over a
    DIFFERENT reminder and commit its typed value onto that one.

    Probe output against the branch before this fix:
    `editor still open after repaint with another task: True` /
    `edit posted for task id: ['task-B']`.
    """
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        task_a = _frequency_reminder(id="task-A", title="A")
        task_b = _frequency_reminder(id="task-B", title="B")
        detail.set_task(task_a)
        await pilot.pause()

        row = detail._timezone_row
        await pilot.click(row)
        await pilot.pause()
        assert row.query(Select)

        detail.set_task(task_b)
        await pilot.pause()

        assert not row.query(Select), "editor survived a repaint with another task"

        # Belt: a commit that crossed the repaint in flight is discarded,
        # never written onto whichever task is painted now.
        with patch.object(
            detail, "post_message", wraps=detail.post_message
        ) as post_spy:
            detail._commit_timezone_edit(
                Select.Changed(Select([("UTC", "UTC")], value="UTC"), "UTC")
            )
            assert not any(
                isinstance(call.args[0], ReminderFieldEditRequested)
                for call in post_spy.call_args_list
            )


@pytest.mark.asyncio
async def test_same_row_repaint_preserves_an_open_editor_and_its_typed_text():
    """The other half of I2: the reminder pane repaints on EVERY tick, so
    closing the editor on a same-row repaint would make typing
    impossible. The value region is skipped for the editing row instead
    (`DetailValueRow.update_value`)."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        task = _frequency_reminder(
            id="task-A",
            schedule_kind=ScheduleKind.ONE_TIME,
            cron=None,
            timezone=None,
            run_at=datetime(2099, 9, 9, 9, 0, tzinfo=timezone.utc),
        )
        detail.set_task(task)
        await pilot.pause()

        row = detail._at_row
        await pilot.click(row)
        await pilot.pause()
        editor = row.query_one(Input)
        editor.value = "2026-09-09 08:30"

        # Five ticks' worth of same-row repaints.
        for _ in range(5):
            detail.set_task(task)
            await pilot.pause()

        assert row.query_one(Input) is editor
        assert editor.value == "2026-09-09 08:30"


@pytest.mark.asyncio
async def test_repaint_with_another_task_clears_a_stale_row_error():
    """Final review F3/I3: `show_error` was only ever cleared by
    reactivating that row or by a successful commit, so B's Frequency
    group kept accusing a value of A's that B never had. Probe output
    before the fix: `stale error still displayed on the new task's row:
    True | Unknown timezone: Mars/Olympus`."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder(id="task-A"))
        await pilot.pause()

        row = detail._timezone_row
        row.show_error("Unknown timezone: Mars/Olympus")
        await pilot.pause()
        assert row.query_one(".detail-value-row-error", Static).display is True

        detail.set_task(_frequency_reminder(id="task-B"))
        await pilot.pause()

        error = row.query_one(".detail-value-row-error", Static)
        assert error.display is False
        assert "Mars/Olympus" not in str(error.renderable)


@pytest.mark.asyncio
async def test_definition_pane_repaint_with_another_row_closes_editor_and_error():
    """The definition pane's twin of I2/I3 -- same mechanism, same
    row-identity trigger."""
    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_editable_definition(id="def-A"))
        await pilot.pause()

        row = detail._generation_row
        await pilot.click(row)
        await pilot.pause()
        assert row.query(Select)
        detail._sources_row.show_error("Choose at least one source.")

        detail.set_definition(_editable_definition(id="def-B"))
        await pilot.pause()

        assert not row.query(Select)
        assert (
            detail._sources_row.query_one(
                ".detail-value-row-error", Static
            ).display
            is False
        )


@pytest.mark.asyncio
async def test_owner_actions_row_reserves_no_line_without_a_transfer():
    """Final review F5/M5: `.detail-value-row-owner-actions` had a fixed
    `height: 1` on an always-mounted container whose two buttons are
    `.display`-toggled, so every reminder and every definition carried one
    dead line inside its Details group. Probe before the fix:
    `Region(x=3, y=12, width=75, height=1)`."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_frequency_reminder(transfer_state=None))
        await pilot.pause()

        actions = detail.query_one(".detail-value-row-owner-actions")
        assert actions.region.height == 0

        detail.set_task(
            _frequency_reminder(id="task-moving", transfer_state="to_server_pending")
        )
        await pilot.pause()
        assert detail.query_one(".detail-value-row-owner-actions").region.height == 1


@pytest.mark.asyncio
async def test_editable_rows_carry_a_row_key_for_per_row_worker_groups():
    """Final review F12/M12: both edit workers group per ROW now, so
    committing a second row's editor cannot cancel the first commit
    mid-write. The group name is built from `row.row_key`, so every
    editable row needs a distinct one -- a `None` key would collapse them
    all back into one group without failing anything visibly."""
    async with _BareTaskDetailApp().run_test(size=(80, 60)) as pilot:
        keys = [row.row_key for row in pilot.app.query_one(TaskDetail)._editable_rows()]
    assert keys and None not in keys and len(set(keys)) == len(keys)

    async with _BareDefinitionDetailApp().run_test(size=(80, 60)) as pilot:
        keys = [
            row.row_key
            for row in pilot.app.query_one(DefinitionDetail)._editable_rows()
        ]
    assert keys and None not in keys and len(set(keys)) == len(keys)


@pytest.mark.asyncio
async def test_queue_definition_pane_repaints_after_a_successful_in_pane_edit(tmp_path):
    """Final review F4/I4. `DefinitionDetail` is mounted TWICE and the
    edit success path only refreshed the Automations one; the Queue tab's
    instance is painted from `_update_detail_for_index`, which
    early-returns for the same row on a tick. So a user editing from the
    Queue tab saw the editor close, the OLD value come back, and no error
    -- indefinitely, even though the edit had persisted."""
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
    from tldw_chatbook.UI.Screens.scheduling.definition_detail import (
        _definition_edit_payload,
    )

    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        service = SchedulingService(db=db, server_client=None, runtime_source="local")
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Daily Q",
            schedule={"kind": "interval", "every_seconds": 3600},
            input={"question": "What shipped?"},
            config={"generation_mode": "optional"},
        )
        definition = db.get_automation_definition(definition_id)

        app = WorkbenchTestApp()
        app.scheduling_service = service
        async with app.run_test(size=(220, 60)) as pilot:
            workbench = SchedulesWorkbench(app_instance=pilot.app)
            await pilot.app.push_screen(workbench)
            # Fix wave F4: `settle_schedules_workbench` (not a bare
            # pause + wait) -- the mount-time catch-up results pull is a
            # `set_timer` callback `wait_for_complete` does not cover, and
            # its `_request_tasks_refresh` wipes a hand-set definition off
            # the live `#scheduling-queue-definition-detail` pane.
            await settle_schedules_workbench(pilot, workbench)

            queue_detail = pilot.app.screen.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            # The Queue tab has this definition's row selected and painted.
            workbench._selected_row_id = f"definition:{definition_id}"
            queue_detail.set_definition(definition)
            await pilot.pause()
            row = queue_detail._generation_row
            assert "Only when something new is found" == str(
                row.query_one(
                    "#scheduling-automation-detail-generation", Static
                ).renderable
            )

            payload = _definition_edit_payload(
                definition, config={"generation_mode": "required"}
            )
            workbench._edit_definition_field(definition, payload, row)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert (
                db.get_automation_definition(definition_id)["config"][
                    "generation_mode"
                ]
                == "required"
            )
            painted = str(
                row.query_one(
                    "#scheduling-automation-detail-generation", Static
                ).renderable
            )
            assert painted == "Always generate a draft", (
                f"the Queue tab's definition pane still shows {painted!r}"
            )
    finally:
        db.close()
