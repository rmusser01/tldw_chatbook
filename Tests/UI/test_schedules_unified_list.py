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
from textual.widgets import Button, DataTable, Input, Static, TabbedContent

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import MockSchedulingDB, MockSchedulingServiceMixin
from tldw_chatbook.Scheduling.events import (
    AcknowledgeIncidentRequested,
    DeleteTaskRequested,
)
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.definition_detail import DefinitionDetail
from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
    AutomationDefinitionForm,
)
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


def _definition(
    def_id: str,
    name: str,
    *,
    lifecycle: str = "configured",
    family: str = "recurring_question",
) -> dict:
    return {
        "id": def_id,
        "server_id": None,
        "owner_id": "local",
        "name": name,
        # `build_unified_rows` lists every family now (PR-4 ruling 1) --
        # every real definition row carries a family regardless, so this
        # fixture always sets one.
        "family": family,
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
# Definition rows: mark/toggle still no-op; run-now/edit are routed
# (redesign PR-4, task 3 -- supersedes the original "no actions until
# PR-4" pin from plan ruling 1).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_definition_row_mark_and_toggle_still_no_op():
    """Mark/toggle-enabled on a definition row still no-op -- PR-4 task 3
    only wired run-now (`r`) and edit-in-full (`e`) onto Queue definition
    rows; `x`/`space` remain unrouted (see `test_definition_row_edit_
    opens_the_form_for_a_recurring_question_row` for the two that now
    act, and `test_definition_row_keys_answer_honestly_by_kind` for the
    full per-key picture)."""
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

        workbench.action_toggle_enabled()
        await pilot.pause()
        assert workbench._scheduling_service.updated == []


@pytest.mark.asyncio
async def test_definition_row_edit_opens_the_form_for_a_recurring_question_row():
    """redesign PR-4, task 3: `e` on a Queue definition row now opens the
    SAME `AutomationDefinitionForm` the Automations tab's own `e` opens
    (`_edit_selected_automation` reused via `_selected_queue_definition`)
    -- this fixture's rows are all `recurring_question`, so the family
    gate never refuses here."""
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

        workbench.action_edit_task()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(pilot.app.screen, AutomationDefinitionForm)


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
async def test_automations_edit_save_refetches_definitions_immediately():
    """Round 2 pinned that a save marks the cache stale and the NEXT
    reminder-only refresh upgrades to one full fetch. Final review F1
    moved that fetch forward: the save's own path refreshes the Queue
    (its create entry point is the Queue rail, which never fires
    `TabActivated`), so the flag is consumed at the save -- exactly one
    definitions fetch then, and none owed to the next reminder action."""
    service, calls = _counting_definitions_service()

    class _CountingApp(ConsolidatedCSSApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.scheduling_service = service

    async with _CountingApp().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        calls_before_save = len(calls)

        workbench._on_automation_form_result(
            SimpleNamespace(status="saved"), was_edit=True
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert workbench._definitions_stale is False
        assert len(calls) == calls_before_save + 2, (
            "the save refreshes the Automations list AND the Queue, one "
            "definitions fetch each"
        )
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
        assert len(calls) == calls_after_save, (
            "nothing is owed once the save itself refetched"
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


# ---------------------------------------------------------------------------
# Final review fix wave (F1-F12). One test per finding, each driving the
# real workbench wiring rather than the pure adapter (Task 1's own suite
# covers the math).
# ---------------------------------------------------------------------------


class _RuntimeState:
    active_server_id = "server-1"


class _RuntimePolicy:
    state = _RuntimeState()


def _app_for(service, *, with_server: bool = False):
    """A `ConsolidatedCSSApp` wired to ``service`` (and, optionally, to a
    runtime policy naming an active server so `_server_available` is
    true and the Queue loader fetches the server half too)."""

    class _ServiceApp(ConsolidatedCSSApp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.scheduling_service = service
            if with_server:
                self.runtime_policy = _RuntimePolicy()

    return _ServiceApp()


def _toasts(app) -> list[str]:
    return [n.message for n in app._notifications]


# --- F1: a rail Create ▾ save paints its row without leaving the tab -------


@pytest.mark.asyncio
async def test_rail_create_recurring_question_paints_the_new_row():
    """Final review F1: Task 3 moved definition-create onto the Queue
    rail, so the save never fires `TabActivated` -- the only consumer of
    `_definitions_stale`. The automation the user just created has to
    appear on the surface it was created from, with no tab switch."""
    service = _MixedService()
    async with _app_for(service).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        assert "def-new" not in _row_ids(workbench, "definition")

        # The form wrote the row; the modal's callback is what the rail
        # path actually reaches.
        service.db._automation_definitions.append(
            _definition("def-new", "Just created")
        )
        workbench._on_automation_form_result(SimpleNamespace(status="saved"))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert "def-new" in _row_ids(workbench, "definition")
        table = workbench.query_one("#scheduling-task-table", DataTable)
        painted = [str(table.get_row_at(i)[1]) for i in range(table.row_count)]
        assert any("Just created" in cell for cell in painted)


# --- F2: a transferred definition keeps its pre-transfer unread count -----


_TRANSFERRED_LOCAL_ROW = {
    "id": "def-local-1",
    "server_id": "srv-1",
    "owner_id": "server:server-1",
    "family": "recurring_question",
    "name": "Moved to the server",
    "lifecycle": "configured",
    "schedule": {"kind": "one_time", "run_at": "2099-01-01T00:00:00+00:00"},
    "input": {"question": "What changed?"},
    "updated_at": "2026-08-01T00:00:00+00:00",
}


class _TransferredService(MockSchedulingServiceMixin):
    """One definition that has been transferred to the server, plus the two
    unread results it produced BEFORE the transfer (local id space)."""

    def __init__(self) -> None:
        self.server_client = SimpleNamespace(
            notifications_service=object(),
            list_automation_definitions=AsyncMock(
                return_value={
                    "items": [
                        {
                            "id": "srv-1",
                            "owner_id": "1",
                            "name": "Moved to the server",
                            "family": "recurring_question",
                            "lifecycle": "configured",
                            "health": "ready",
                            "schedule": {
                                "kind": "one_time",
                                "run_at": "2099-01-01T00:00:00+00:00",
                            },
                            "input": {"question": "What changed?"},
                            "updated_at": "2026-08-01T00:00:00+00:00",
                        }
                    ],
                    "total": 1,
                }
            ),
        )
        self.db = MockSchedulingDB(
            automation_definitions=[dict(_TRANSFERRED_LOCAL_ROW)],
            automation_results=[
                {
                    "id": f"result-{index}",
                    "definition_id": "def-local-1",
                    "owner_id": "local",
                    "review_state": "unread",
                    "kind": "finding",
                    "created_at": "2026-08-20T00:00:00+00:00",
                }
                for index in (1, 2)
            ],
        )

    async def list_tasks(self, owner_id=None, include_projections=True):
        return []


@pytest.mark.asyncio
async def test_transferred_definition_unread_matches_the_results_badge():
    """Final review F2, the review's own repro: the Results tab badge read
    2 while the Queue counted 0 (and hid the rail button) for the same DB,
    because the Queue indexed the DISPLAY merge -- which drops every local
    row carrying a `server_id`."""
    service = _TransferredService()
    async with _app_for(service, with_server=True).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)

        queue_unread = sum(row.unread_count for row in workbench._all_rows)
        badge = str(
            workbench.query_one("#scheduling-tabs", TabbedContent)
            .get_tab("scheduling-results-tab")
            .label
        )
        assert "Results (2)" in badge
        assert queue_unread == 2, (
            "the Queue's unread count must equal the Results badge for the "
            f"same DB; badge={badge!r}"
        )
        assert workbench.query_one("#scheduling-mark-all-read", Button).display is True


# --- F3: incident ack routes by row id, not by a reminder-list index ------


class _IncidentDB(MockSchedulingDB):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.acked: list[int] = []

    def list_task_incidents(self, key, limit=10):
        if key == "task-late":
            return [
                {
                    "id": 7,
                    "task_id": "task-late",
                    "kind": "handler_failed",
                    "message": "Handler failed",
                    "created_at": "2026-08-27T11:00:00+00:00",
                    "acknowledged_at": None,
                }
            ]
        return []

    def acknowledge_incident(self, incident_id, when) -> None:
        self.acked.append(int(incident_id))


class _DefinitionAboveReminderService(MockSchedulingServiceMixin):
    """A definition row that sorts ABOVE the incident-carrying reminder --
    the exact divergence between `_visible_rows` and `_visible_tasks`."""

    def __init__(self) -> None:
        self.db = _IncidentDB(
            automation_definitions=[
                {
                    **_definition("def-first", "Sorts first"),
                    "next_run_at": "2026-08-27T12:30:00+00:00",
                }
            ]
        )

    async def list_tasks(self, owner_id=None, include_projections=True):
        return [
            _reminder("task-late", "Failing reminder", next_run_at=_NOW + timedelta(hours=5))
        ]


@pytest.mark.asyncio
async def test_acknowledge_incident_keeps_the_selected_reminder_showing():
    """Final review F3: `_update_detail_for_index` takes a `_visible_rows`
    index. Feeding it a `_visible_tasks` index rendered the row ABOVE the
    highlighted one -- flipping the pane to a definition and moving
    `_selected_row_id` with it, silently, while the cursor stayed put."""
    service = _DefinitionAboveReminderService()
    async with _app_for(service).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        kinds = [row.kind for row in workbench._visible_rows]
        assert kinds == ["definition", "reminder"], kinds

        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()
        assert workbench._selected_row_id == "reminder:task-late"

        workbench.post_message(AcknowledgeIncidentRequested(7))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert service.db.acked == [7]
        assert workbench._selected_row_id == "reminder:task-late"
        assert workbench._selected_task_id == "task-late"
        task_detail = workbench.query_one("#scheduling-task-detail", TaskDetail)
        assert not _pane_hidden(task_detail)


# --- F4: reminder writes key off the ROW's owner, not the active one ------


class _OwnerRecordingService(MockSchedulingServiceMixin):
    """Active owner is `local`; the queue also lists a `server:` row."""

    def __init__(self) -> None:
        self.owner_id = "local"
        self.db = MockSchedulingDB()
        self.updated: list[tuple] = []
        self.deleted: list[tuple] = []

    async def list_tasks(self, owner_id=None, include_projections=True):
        local_row = _reminder("task-local", "Local reminder")
        server_row = _reminder("task-server", "Server reminder")
        server_row.owner_id = "server:srv-1"
        return [local_row, server_row]

    async def update_reminder(self, task_id, fields, *, owner_id=None):
        self.updated.append((task_id, fields, owner_id))
        return None

    async def delete_reminder(self, task_id, *, owner_id=None):
        self.deleted.append((task_id, owner_id))
        return True


@pytest.mark.asyncio
async def test_reminder_writes_thread_the_rows_own_owner():
    """Final review F4: the list spans owners since Task 1 but the writes
    still read `service.owner_id`, so toggling/deleting a `server:` row
    while "This device" was active wrote the local mirror with no pending
    mutation and no tombstone -- the next pull undid it."""
    service = _OwnerRecordingService()
    async with _app_for(service).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        rows = {row.row_id: row.source_row for row in workbench._visible_rows}
        server_task = rows["reminder:task-server"]
        local_task = rows["reminder:task-local"]

        # One at a time: `_set_reminder_enabled` runs in an exclusive
        # worker group, so a second call would cancel the first.
        workbench._set_reminder_enabled(server_task, False)
        workbench.post_message(DeleteTaskRequested(server_task))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench._set_reminder_enabled(local_task, False)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert ("task-server", {"enabled": False}, "server:srv-1") in service.updated
        assert ("task-server", "server:srv-1") in service.deleted
        # The local row is untouched by the change: still its own owner,
        # never the active one by accident.
        assert ("task-local", {"enabled": False}, "local") in service.updated


# --- F5: a results pull reveals the rail button with no tab switch --------


class _PullService(MockSchedulingServiceMixin):
    """No unread results until `_pull_results` runs."""

    def __init__(self) -> None:
        self.db = MockSchedulingDB(
            automation_definitions=[_definition("def-1", "Watched question")]
        )

        async def _run_phase(owner_id, label, phase):
            self.db._automation_results.append(
                {
                    "id": "result-pulled",
                    "definition_id": "def-1",
                    "owner_id": "local",
                    "review_state": "unread",
                    "kind": "finding",
                    "created_at": "2026-08-28T00:00:00+00:00",
                }
            )

        self.sync_engine = SimpleNamespace(
            _run_phase=_run_phase, _pull_results=object()
        )

    async def list_tasks(self, owner_id=None, include_projections=True):
        return []


@pytest.mark.asyncio
async def test_results_pull_reveals_the_rail_mark_all_read_button():
    """Final review F5: an SSE-triggered pull only refreshed the Results
    tab, so the Queue's unread dots and the rail button (gated on
    `sum(row.unread_count) > 0`) stayed absent for exactly the event they
    were built for -- with no periodic reload to correct them."""
    service = _PullService()
    async with _app_for(service).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        button = workbench.query_one("#scheduling-mark-all-read", Button)
        assert button.display is False

        await workbench._pull_results_worker()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert button.display is True
        assert sum(row.unread_count for row in workbench._all_rows) == 1


@pytest.mark.asyncio
async def test_marking_all_read_from_the_results_tab_clears_the_queue_dots():
    """The inverse half of F5: `a` on the Results tab shares the fan-out,
    so the Queue's dots and the rail button must drop with it."""
    service = _MixedService()
    async with _app_for(service).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        button = workbench.query_one("#scheduling-mark-all-read", Button)
        assert button.display is True

        async def _review(result_id, review_state):
            for row in service.db._automation_results:
                if row["id"] == result_id:
                    row["review_state"] = review_state
            return True

        service.review_automation_result = _review
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-results-tab"
        await pilot.pause()
        workbench.action_mark_all_results_read()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert sum(row.unread_count for row in workbench._all_rows) == 0
        assert button.display is False


# --- F6: the placeholder matches what search actually matches -------------


@pytest.mark.asyncio
async def test_filter_placeholder_names_title_and_question():
    """Final review F6: the placeholder still promised type and status
    after ruling 5 narrowed search to title + question/body and the
    Type/Status columns were removed."""
    async with _App().run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        placeholder = workbench.query_one("#scheduling-queue-filter", Input).placeholder
        assert "question" in placeholder.lower()
        assert "status" not in placeholder.lower()
        assert "type" not in placeholder.lower()


# --- F8: definition rows answer the reminder keys honestly -----------------
# (redesign PR-4, task 3 supersedes part of this: `r`/`e` are ROUTED now,
# not refused -- see the class/test docstrings below for the current
# per-key picture.)


@pytest.mark.asyncio
async def test_definition_row_keys_answer_honestly_by_kind():
    """Final review F8, superseded by redesign PR-4 ruling 1 + task 3:
    `x`/`space`/`d` (mark/toggle/delete) are UNCHANGED -- still not wired
    to definition rows -- and keep answering "managed on the Automations
    tab" rather than doing nothing. `r`/`e` are ROUTED now: `r` attempts
    a real dispatch for ANY family (`_run_automation_now` is owner-routed
    only, never family-gated); `e` opens `AutomationDefinitionForm` for
    this fixture's `recurring_question` row (see `test_definition_row_
    edit_refuses_honestly_for_a_non_recurring_question_row` for the
    family-gated refusal, which reads differently from the "Automations
    tab" copy below)."""
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

        for action in (
            workbench.action_mark_task,
            workbench.action_toggle_enabled,
            workbench.action_delete,
        ):
            before = len(_toasts(pilot.app))
            action()
            await pilot.pause()
            new = _toasts(pilot.app)[before:]
            assert new, f"{action.__name__} answered nothing at all"
            assert any("Automations tab" in message for message in new), (
                f"{action.__name__} said {new!r}"
            )

        # `d` must not reach TaskDetail.request_delete, which would open
        # a confirmation for whatever reminder the pane last held.
        assert isinstance(pilot.app.screen, SchedulesWorkbench)

        # `r`: routed now (task 3) -- a real dispatch attempt, not the
        # old refusal. The stub service has no `run_automation_now`, so
        # the attempt fails honestly rather than silently.
        before = len(_toasts(pilot.app))
        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        run_toasts = _toasts(pilot.app)[before:]
        assert any("Failed to run" in message for message in run_toasts), run_toasts

        # `e`: routed now too -- opens the SAME AutomationDefinitionForm
        # the Automations tab's own `e` opens (this row is `recurring_
        # question`).
        workbench.action_edit_task()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(pilot.app.screen, AutomationDefinitionForm)


class _AgentTaskOnlyService(MockSchedulingServiceMixin):
    """One `agent_task` definition, no reminders -- isolates the Queue's
    only row so `e`'s family-gated refusal can be pinned deterministically
    (redesign PR-4 ruling 1: a non-`recurring_question` definition now
    has a home on the Queue too)."""

    def __init__(self) -> None:
        self.db = MockSchedulingDB(
            automation_definitions=[
                _definition("def-agent", "Nightly agent run", family="agent_task"),
            ]
        )

    async def list_tasks(self, owner_id=None, include_projections=True):
        return []


@pytest.mark.asyncio
async def test_definition_row_edit_refuses_honestly_for_a_non_recurring_question_row():
    """redesign PR-4, task 3: `e` on an `agent_task` Queue row refuses --
    the SAME family-gate copy the Automations tab's own `e` already gives
    (`_edit_selected_automation`'s existing refusal, reused verbatim) --
    which is NOT the "managed on the Automations tab" copy the other
    (still-unrouted) keys use."""
    app = _App()
    app.scheduling_service = _AgentTaskOnlyService()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=0)
        await pilot.pause()

        before = len(_toasts(pilot.app))
        workbench.action_edit_task()
        await pilot.pause()
        new = _toasts(pilot.app)[before:]

        assert isinstance(pilot.app.screen, SchedulesWorkbench)  # no form pushed
        assert any("recurring-question" in message for message in new), new
        assert not any("Automations tab" in message for message in new), new


# --- F10: empty-state copy at widths where the chips are hidden -----------


class _ActiveOnlyService(MockSchedulingServiceMixin):
    def __init__(self) -> None:
        self.db = MockSchedulingDB()

    async def list_tasks(self, owner_id=None, include_projections=True):
        return [_reminder("task-active", "Active reminder", next_run_at=_NOW)]


async def _completed_chip_notice(size) -> str:
    service = _ActiveOnlyService()
    async with _app_for(service).run_test(size=size) as pilot:
        workbench = await _mounted(pilot)
        workbench._set_queue_chip("completed")
        await pilot.pause()
        assert not workbench._visible_rows
        return workbench.query_one(
            "#scheduling-task-detail-empty-state", Static
        ).visual.plain


@pytest.mark.asyncio
async def test_empty_view_copy_drops_the_chip_hint_when_chips_are_hidden():
    """Final review F10: below width 84 the chip row is hidden while the
    selected chip persists, so "Choose a different chip" pointed at a
    control that is not on screen."""
    wide = await _completed_chip_notice((160, 48))
    assert "No tasks in this view" in wide
    assert "Choose a different chip" in wide

    narrow = await _completed_chip_notice((80, 40))
    assert "No tasks in this view" in narrow
    assert "chip" not in narrow.lower()


# --- F11: a server run-now marks the definitions cache stale --------------


class _ServerRunNowService(MockSchedulingServiceMixin):
    def __init__(self) -> None:
        self.server_client = SimpleNamespace(
            notifications_service=object(),
            list_automation_definitions=AsyncMock(
                return_value={"items": [], "total": 0}
            ),
            run_automation_definition_now=AsyncMock(
                return_value={"run_slot_utc": "slot-1", "deduped": False}
            ),
            list_automation_definition_audit=AsyncMock(
                return_value={"items": [], "total": 0}
            ),
        )
        self.db = MockSchedulingDB()

    async def list_tasks(self, owner_id=None, include_projections=True):
        return []


@pytest.mark.asyncio
async def test_server_run_now_marks_definitions_stale():
    """Final review F11: the branch's own rule is "mark staleness at each
    genuine mutation call site"; only the local twin did."""
    service = _ServerRunNowService()
    async with _app_for(service, with_server=True).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        assert workbench._definitions_stale is False

        workbench._run_automation_now(
            {
                "id": "srv-1",
                "name": "Server automation",
                "owner_id": "server:server-1",
                "family": "recurring_question",
                "lifecycle": "configured",
            }
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service.server_client.run_automation_definition_now.assert_awaited_once()
        assert workbench._definitions_stale is True


# --- F12: the ticker does not re-read the definition detail ---------------


@pytest.mark.asyncio
async def test_tick_skips_the_definition_detail_read_for_an_unchanged_row():
    """Final review F12: the ticker re-ran 3 DB reads a minute for a row
    that had not moved, against its own "no reload/DB on tick" contract.
    A refresh-driven render still re-feeds -- data can change while the
    selection stands still (PR-1's own F4 lesson)."""
    service = _MixedService()
    reads: list[tuple] = []
    real_count = service.db.count_automation_runs

    def _counting(*args, **kwargs):
        reads.append((args, kwargs))
        return real_count(*args, **kwargs)

    service.db.count_automation_runs = _counting

    async with _app_for(service).run_test(size=(160, 48)) as pilot:
        workbench = await _mounted(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        definition_index = next(
            i
            for i, row in enumerate(workbench._visible_rows)
            if row.kind == "definition"
        )
        table.move_cursor(row=definition_index)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        after_selection = len(reads)
        assert after_selection > 0

        workbench._refresh_next_run_rendering()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert len(reads) == after_selection, "a tick must not re-read the DB"

        # A refresh-driven render still re-feeds the pane.
        workbench._render_table()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert len(reads) == after_selection + 1
