"""Tests for the unified Schedules Queue list (redesign PR-2, Task 2).

Covers what `Tests/UI/test_schedules_workbench.py` does not: a MIXED
listing (reminders + automation definitions, both id-spaces), each
chip's bucket (including the fired-one-time-reminder-under-Completed and
to_server_pending-stays-Active cases the brief calls out by name),
detail routing both directions, a reminder-action preservation smoke
test, and definition rows exposing no actions. `Tests/Scheduling/
test_unified_rows.py` (Task 1) already exhaustively covers the bucket/
sort/filter PURE functions this file builds on -- these tests are about
the workbench's wiring, not re-proving Task 1's own math.
"""

from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import Button, DataTable, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import MockSchedulingDB, MockSchedulingServiceMixin
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.definition_detail import DefinitionDetail
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail

_NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)


def _reminder(
    task_id: str,
    title: str,
    *,
    enabled: bool = True,
    next_run_at: datetime | None = None,
    last_run_at: datetime | None = None,
    transfer_state: str | None = None,
) -> ReminderTask:
    return ReminderTask(
        id=task_id,
        title=title,
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=next_run_at or (_NOW + timedelta(hours=1)),
        next_run_at=next_run_at,
        enabled=enabled,
        last_run_at=last_run_at,
        transfer_state=transfer_state,
    )


def _definition(def_id: str, name: str, *, lifecycle: str = "configured") -> dict:
    return {
        "id": def_id,
        "server_id": None,
        "owner_id": "local",
        "name": name,
        "lifecycle": lifecycle,
        "schedule": {"kind": "one_time", "run_at": "2099-01-01T00:00:00+00:00"},
        "input": {"question": f"Question for {name}?"},
        "updated_at": "2026-08-01T00:00:00+00:00",
    }


class _MixedService(MockSchedulingServiceMixin):
    """Reminders + automation definitions spanning every chip bucket.

    Reminder side: `task-active` (armed), `task-pending` (`enabled` with
    `transfer_state="to_server_pending"` -- spec S3: stays Active, "they
    still execute locally"), `task-paused` (disabled, not fired),
    `task-fired` (disabled, `next_run_at=None`, `last_run_at` set -- the
    fired-one-time predicate, `reminder_has_fired`).

    Definition side: `def-active` (configured), `def-paused` (paused
    lifecycle), `def-archived` (archived lifecycle), `def-unread` (
    configured, with one UNREAD `automation_results` row so the title
    cell's unread dot has something to paint).
    """

    def __init__(self) -> None:
        self.updated: list = []
        self.db = MockSchedulingDB(
            automation_definitions=[
                _definition("def-active", "Active definition"),
                _definition("def-paused", "Paused definition", lifecycle="paused"),
                _definition(
                    "def-archived", "Archived definition", lifecycle="archived"
                ),
                _definition("def-unread", "Unread definition"),
            ],
            automation_results=[
                {
                    "id": "result-1",
                    "definition_id": "def-unread",
                    "owner_id": "local",
                    "review_state": "unread",
                    "kind": "finding",
                    "created_at": "2026-08-20T00:00:00+00:00",
                }
            ],
        )

    async def list_tasks(self, owner_id=None):
        return [
            _reminder(
                "task-active", "Active reminder", next_run_at=_NOW + timedelta(hours=2)
            ),
            _reminder(
                "task-pending",
                "Pending-transfer reminder",
                next_run_at=_NOW + timedelta(hours=3),
                transfer_state="to_server_pending",
            ),
            _reminder(
                "task-paused",
                "Paused reminder",
                enabled=False,
                next_run_at=_NOW + timedelta(hours=4),
            ),
            _reminder(
                "task-fired",
                "Fired reminder",
                enabled=False,
                next_run_at=None,
                last_run_at=_NOW - timedelta(hours=1),
            ),
        ]

    async def update_reminder(self, task_id, fields, *, owner_id=None):
        self.updated.append((task_id, fields))
        return None


class _App(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _MixedService()


async def _mounted(pilot):
    workbench = SchedulesWorkbench(app_instance=pilot.app)
    await pilot.app.push_screen(workbench)
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return workbench


def _row_ids(workbench, kind: str | None = None) -> set[str]:
    return {
        row.row_id.split(":", 1)[1]
        for row in workbench._visible_rows
        if kind is None or row.kind == kind
    }


# ---------------------------------------------------------------------------
# Mixed rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_listing_renders_both_kinds_painted():
    """Reminders and definitions both render, each with a real glyph and
    the title/subtitle content the spec names (painted, not the stored
    object -- D8)."""
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)

        kinds = {row.row_id.split(":", 1)[0] for row in workbench._visible_rows}
        assert kinds == {"reminder", "definition"}
        # Default chip is "all" (Active+Paused) -- Completed rows (the
        # fired reminder, the archived definition) are excluded by design.
        assert "task-fired" not in _row_ids(workbench)
        assert "def-archived" not in _row_ids(workbench)

        for index, row in enumerate(workbench._visible_rows):
            painted = table.get_row_at(index)
            assert str(painted[0]) == row.glyph
            # `row.title` is the bare title for both kinds -- a definition
            # cell wraps it in the automation_name_cell owner-prefix
            # ("[This device] <name>"), checked precisely below.
            assert row.title in str(painted[1])

        # The definition row's title carries the automation_name_cell
        # owner-prefix rendering (reused verbatim, per the brief).
        def_index = next(
            i
            for i, row in enumerate(workbench._visible_rows)
            if row.kind == "definition"
        )
        assert "[This device]" in str(table.get_row_at(def_index)[1])

        # The unread definition's title carries the bold unread dot.
        unread_index = next(
            i
            for i, row in enumerate(workbench._visible_rows)
            if row.kind == "definition" and row.source_row.get("id") == "def-unread"
        )
        assert "●" in str(table.get_row_at(unread_index)[1])


# ---------------------------------------------------------------------------
# Chip buckets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_chip_includes_to_server_pending_reminder():
    """spec S3: `to_server_pending` stays Active -- "they still execute
    locally" -- not dropped into Paused with the dormant transfer states."""
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench._set_queue_chip("active")
        await pilot.pause()

        reminder_ids = _row_ids(workbench, "reminder")
        assert reminder_ids == {"task-active", "task-pending"}
        definition_ids = _row_ids(workbench, "definition")
        assert definition_ids == {"def-active", "def-unread"}


@pytest.mark.asyncio
async def test_paused_chip_includes_disabled_reminder_and_paused_definition():
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench._set_queue_chip("paused")
        await pilot.pause()

        assert _row_ids(workbench, "reminder") == {"task-paused"}
        assert _row_ids(workbench, "definition") == {"def-paused"}


@pytest.mark.asyncio
async def test_completed_chip_includes_fired_reminder_and_archived_definition():
    """The brief's own two named cases: a fired one-time reminder AND an
    archived definition both surface only under Completed."""
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench._set_queue_chip("completed")
        await pilot.pause()

        assert _row_ids(workbench, "reminder") == {"task-fired"}
        assert _row_ids(workbench, "definition") == {"def-archived"}


@pytest.mark.asyncio
async def test_all_chip_excludes_completed_rows():
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        workbench._set_queue_chip("all")
        await pilot.pause()

        assert "task-fired" not in _row_ids(workbench, "reminder")
        assert "def-archived" not in _row_ids(workbench, "definition")


@pytest.mark.asyncio
async def test_chip_button_press_switches_the_active_chip():
    """The chip row is real buttons, not just the pure `_set_queue_chip`
    seam -- pressing one drives the same narrowing."""
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        completed_button = workbench.query_one("#scheduling-chip-completed", Button)
        completed_button.press()
        await pilot.pause()

        assert workbench._chip == "completed"
        assert str(completed_button.variant) == "primary"
        all_button = workbench.query_one("#scheduling-chip-all", Button)
        assert str(all_button.variant) == "default"
        assert _row_ids(workbench, "reminder") == {"task-fired"}


# ---------------------------------------------------------------------------
# Detail routing, both directions
# ---------------------------------------------------------------------------


def _pane_hidden(widget) -> bool:
    return "pane-hidden" in widget.classes


@pytest.mark.asyncio
async def test_highlighting_routes_to_the_matching_detail_pane_both_directions():
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        task_detail = workbench.query_one("#scheduling-task-detail", TaskDetail)
        definition_detail = workbench.query_one(
            "#scheduling-queue-definition-detail", DefinitionDetail
        )

        reminder_index = next(
            i for i, row in enumerate(workbench._visible_rows) if row.kind == "reminder"
        )
        definition_index = next(
            i
            for i, row in enumerate(workbench._visible_rows)
            if row.kind == "definition"
        )

        # Reminder -> definition.
        table.move_cursor(row=reminder_index)
        await pilot.pause()
        assert not _pane_hidden(task_detail)
        assert _pane_hidden(definition_detail)

        table.move_cursor(row=definition_index)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert _pane_hidden(task_detail)
        assert not _pane_hidden(definition_detail)
        assert (
            definition_detail.query_one(
                "#scheduling-automation-detail-empty-state", Static
            ).display
            is False
        )

        # Definition -> reminder, back again.
        table.move_cursor(row=reminder_index)
        await pilot.pause()
        assert not _pane_hidden(task_detail)
        assert _pane_hidden(definition_detail)


# ---------------------------------------------------------------------------
# Preservation gate: reminder actions still fire amid a mixed list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reminder_actions_still_fire_amid_a_mixed_list():
    """Smoke test: edit/mark/toggle on a REMINDER row still work exactly
    as before, even though the table now also contains definition rows.
    """
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        reminder_index = next(
            i
            for i, row in enumerate(workbench._visible_rows)
            if row.kind == "reminder" and row.source_row.id == "task-active"
        )
        table.move_cursor(row=reminder_index)
        await pilot.pause()

        workbench.action_mark_task()
        await pilot.pause()
        assert workbench._marked_ids == {"task-active"}

        workbench.action_clear_marks()
        await pilot.pause()
        assert not workbench._marked_ids


# ---------------------------------------------------------------------------
# Definition rows: no actions in this PR
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_definition_rows_expose_no_actions():
    """Edit/mark/toggle/delete on a definition row must no-op -- plan
    ruling 1: definition rows are viewable + detail only until PR-4."""
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        definition_index = next(
            i
            for i, row in enumerate(workbench._visible_rows)
            if row.kind == "definition"
        )
        table.move_cursor(row=definition_index)
        await pilot.pause()

        assert workbench._selected_task_id is None
        assert workbench._selected_task() is None

        workbench.action_mark_task()
        await pilot.pause()
        assert not workbench._marked_ids

        workbench.action_edit_task()
        await pilot.pause()
        assert isinstance(pilot.app.screen, SchedulesWorkbench)  # no form pushed

        workbench.action_toggle_enabled()
        await pilot.pause()
        assert workbench._scheduling_service.updated == []
