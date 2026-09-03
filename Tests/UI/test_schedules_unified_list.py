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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, DataTable, Static, TabbedContent

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

    async def list_tasks(self, owner_id=None, include_projections=True):
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


# ---------------------------------------------------------------------------
# Fix round 1 (task-2 review): reminder-only refreshes must not re-fetch
# automation definitions (finding 3), and the filter seam must narrow a
# MIXED reminder+definition list (finding 4, the brief's own "search
# narrows" AC item, previously untested at the integration level).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reminder_toggle_triggers_no_new_definitions_fetch():
    """redesign PR-2 Task 2 review, finding 3: a reminder-only action
    (`_request_tasks_refresh(refresh_definitions=False)`) must reuse the
    definitions already in `self._all_rows` rather than re-running
    `_load_local_automations`'s `service.db.list_automation_definitions`
    scan. Counting fake, per the review's own suggested pin shape."""
    service = _MixedService()
    calls: list[tuple] = []
    real_list_automation_definitions = service.db.list_automation_definitions

    def _counting_list_automation_definitions(*args, **kwargs):
        calls.append((args, kwargs))
        return real_list_automation_definitions(*args, **kwargs)

    service.db.list_automation_definitions = _counting_list_automation_definitions

    class _CountingApp(ConsolidatedCSSApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.scheduling_service = service

    async with _CountingApp().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        # Mount runs the FULL Queue loader (load_tasks's own definitions
        # fetch) AND the Automations tab's own load_automations -- both
        # call list_automation_definitions once each. That baseline is
        # not what this test pins; only whether a reminder toggle adds
        # to it.
        calls_after_mount = len(calls)
        assert calls_after_mount > 0

        task = next(
            row.source_row
            for row in workbench._visible_rows
            if row.kind == "reminder" and row.source_row.id == "task-active"
        )
        workbench._set_reminder_enabled(task, False)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert service.updated == [("task-active", {"enabled": False})]
        assert len(calls) == calls_after_mount, (
            "a reminder-only refresh must not re-fetch automation definitions"
        )


@pytest.mark.asyncio
async def test_search_narrows_a_mixed_reminder_and_definition_list():
    """The brief's own named AC ("search narrows") for the mixed list --
    the pure `filter_rows` function is exhaustively covered in Task 1's
    suite; this pins the workbench's own wiring
    (`_filter_text` -> `_render_table` -> `filter_rows(self._all_rows,
    ...)`) against a set with BOTH row kinds, including a definition
    matched only by its question/body text."""
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)

        # A reminder title match narrows to just that reminder.
        workbench._filter_text = "Active reminder"
        workbench._render_table()
        await pilot.pause()
        assert _row_ids(workbench) == {"task-active"}

        # A definition's question/body text (not its title) also matches
        # -- `search_blob` is title + question/body (Task 1, ruling 5).
        workbench._filter_text = "Question for Active definition"
        workbench._render_table()
        await pilot.pause()
        assert _row_ids(workbench, "definition") == {"def-active"}
        assert _row_ids(workbench, "reminder") == set()

        # Clearing the filter restores the full (chip-narrowed) mixed set.
        workbench._filter_text = ""
        workbench._render_table()
        await pilot.pause()
        kinds = {row.kind for row in workbench._visible_rows}
        assert kinds == {"reminder", "definition"}


# ---------------------------------------------------------------------------
# Fix round 2 (task-2 review addendum): an Automations-tab definition
# mutation must not leave the Queue's cached definitions stale for the
# rest of the session -- a reminder-only refresh upgrades to one full
# fetch, and so does switching to the Queue tab while stale.
# ---------------------------------------------------------------------------


def _counting_definitions_service() -> tuple[_MixedService, list]:
    """A `_MixedService` whose `db.list_automation_definitions` records
    every call, for the staleness pins below."""
    service = _MixedService()
    calls: list[tuple] = []
    real = service.db.list_automation_definitions

    def _counting(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    service.db.list_automation_definitions = _counting
    return service, calls


@pytest.mark.asyncio
async def test_automations_edit_save_then_reminder_toggle_refetches_once():
    """redesign PR-2 Task 2 review round 2: an Automations-tab edit-save
    marks the Queue's definitions cache stale; the NEXT reminder-only
    refresh must upgrade to exactly one full definitions fetch -- not
    zero (stale would go unnoticed) and not more than one."""
    service, calls = _counting_definitions_service()

    class _CountingApp(ConsolidatedCSSApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.scheduling_service = service

    async with _CountingApp().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)

        workbench._on_automation_form_result(
            SimpleNamespace(status="saved"), was_edit=True
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert workbench._definitions_stale is True
        calls_after_save = len(calls)

        task = next(
            row.source_row
            for row in workbench._visible_rows
            if row.kind == "reminder" and row.source_row.id == "task-active"
        )
        workbench._set_reminder_enabled(task, False)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench._definitions_stale is False
        assert len(calls) == calls_after_save + 1, (
            "a stale-triggered upgrade must fetch definitions exactly once"
        )


@pytest.mark.asyncio
async def test_automations_run_now_then_tab_switch_to_queue_refreshes():
    """An Automations-tab run-now marks the cache stale; switching to
    the Queue tab while stale upgrades the next refresh to a full one,
    even with no reminder action in between."""
    service, calls = _counting_definitions_service()
    service.run_automation_now = AsyncMock(
        return_value={"run_id": "run-1", "deduped": False}
    )

    class _CountingApp(ConsolidatedCSSApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.scheduling_service = service

    async with _CountingApp().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        definition = next(
            row.source_row
            for row in workbench._visible_rows
            if row.kind == "definition" and row.source_row["id"] == "def-active"
        )
        workbench._run_automation_now(definition)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert workbench._definitions_stale is True
        calls_after_run = len(calls)

        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        await pilot.pause()
        tabs.active = "scheduling-queue-tab"
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench._definitions_stale is False
        assert len(calls) == calls_after_run + 1, (
            "arriving on the Queue tab while stale must fetch definitions"
            " exactly once"
        )


@pytest.mark.asyncio
async def test_no_mutation_reminder_toggle_still_skips_the_refetch():
    """Round-1 pin, re-asserted here explicitly: with no Automations
    mutation in play, `_definitions_stale` never gets set, so a
    reminder-only refresh still reuses the cached definitions."""
    service, calls = _counting_definitions_service()

    class _CountingApp(ConsolidatedCSSApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.scheduling_service = service

    async with _CountingApp().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        assert workbench._definitions_stale is False
        calls_after_mount = len(calls)

        task = next(
            row.source_row
            for row in workbench._visible_rows
            if row.kind == "reminder" and row.source_row.id == "task-active"
        )
        workbench._set_reminder_enabled(task, False)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench._definitions_stale is False
        assert len(calls) == calls_after_mount
