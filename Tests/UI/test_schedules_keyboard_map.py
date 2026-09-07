"""Tests for the redesign PR-4, task 4 spec §12 keyboard map on
`SchedulesWorkbench`: `1`-`4`/`f` chips, `/` search, `n` create, `p`
pause/resume, `m` move owner, `r` mark read.

Real `SchedulingService` + a tmp_path `ScheduledTasksDB` throughout (same
rationale `test_schedules_transfer_actions.py`/`test_schedules_workbench.py`
give: routing/refusal correctness is proven against the real facade, not a
hand-rolled stub of it). Each test file in this package duplicates its own
small harness rather than importing another test file's (established
no-cross-file-test-coupling precedent).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, DataTable, Input, Select

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.UI.Screens.scheduling.definition_detail import DefinitionDetail
from tldw_chatbook.UI.Screens.scheduling.forms.new_task_choice_modal import (
    NewTaskChoiceModal,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail


class WorkbenchTestApp(ConsolidatedCSSApp):
    """Minimal test app; `scheduling_service` is set by `_real_service`."""

    scheduling_service = None


def _real_service(tmp_path, app):
    """A real (tmp_path-file) `ScheduledTasksDB` + `SchedulingService`, no
    server -- these tests exercise LOCAL-only routing (reminder enable/
    disable, definition lifecycle, mark-read); the owner-row dropdown's
    OWN server-offered-option behavior is already covered by `test_
    schedules_workbench.py`'s `test_runs_on_dropdown_*` suite."""
    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    service = SchedulingService(db=db, runtime_source="local", app_getter=lambda: app)
    app.scheduling_service = service
    return db, service


async def _select_row(pilot, index: int = 0) -> None:
    table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
    table.cursor_coordinate = (index, 0)
    await pilot.pause()


# ---------------------------------------------------------------------------
# 1-4 / f: chip switching, incl. at a width where the chip row is hidden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digit_keys_switch_the_chip_and_paint_the_selected_button():
    app = WorkbenchTestApp()
    async with app.run_test(size=(220, 60)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        assert workbench._chip == "all"

        await pilot.press("2")
        await pilot.pause()
        assert workbench._chip == "active"
        assert (
            workbench.query_one("#scheduling-chip-active", Button).variant == "primary"
        )
        assert workbench.query_one("#scheduling-chip-all", Button).variant == "default"

        await pilot.press("3")
        await pilot.pause()
        assert workbench._chip == "paused"

        await pilot.press("4")
        await pilot.pause()
        assert workbench._chip == "completed"

        await pilot.press("1")
        await pilot.pause()
        assert workbench._chip == "all"


@pytest.mark.asyncio
async def test_f_key_cycles_through_every_chip_and_wraps():
    app = WorkbenchTestApp()
    async with app.run_test(size=(220, 60)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        seen = [workbench._chip]
        for _ in range(4):
            await pilot.press("f")
            await pilot.pause()
            seen.append(workbench._chip)

        assert seen == ["all", "active", "paused", "completed", "all"], (
            "f must cycle all -> active -> paused -> completed and wrap"
        )


@pytest.mark.asyncio
async def test_chip_keys_work_even_when_the_chip_row_is_hidden():
    """task-4 brief: chip keys must work "at every width (incl.
    collapsed-chip mode later -- don't couple to chip visibility)".

    The chip row hides below 84 cols today via `#scheduling-workbench
    .compact` (`on_resize`'s `hide_detail` class, `_scheduling.tcss`) --
    that CSS rule lives in the app bundle, which this bare harness (no
    `CSS_PATH` override) does not load, so `.display` is forced directly
    here instead of via the class, to isolate what this test actually
    claims: `_set_queue_chip`/`action_cycle_chip` only ever touch
    `self._chip` and the still-MOUNTED buttons' `.variant`, never
    `.display`, so a key press works identically whether or not the row
    is visible.
    """
    app = WorkbenchTestApp()
    async with app.run_test(size=(220, 60)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        chips = workbench.query_one("#scheduling-queue-chips")
        chips.styles.display = "none"
        await pilot.pause()
        assert chips.display is False

        await pilot.press("2")
        await pilot.pause()
        assert workbench._chip == "active"

        await pilot.press("f")
        await pilot.pause()
        assert workbench._chip == "paused"


# ---------------------------------------------------------------------------
# /: focus search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_key_focuses_the_filter_input():
    app = WorkbenchTestApp()
    async with app.run_test(size=(220, 60)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        await pilot.press("/")
        await pilot.pause()
        assert isinstance(pilot.app.focused, Input)
        assert pilot.app.focused.id == "scheduling-queue-filter"


# ---------------------------------------------------------------------------
# n: create chooser
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n_key_opens_the_create_chooser():
    app = WorkbenchTestApp()
    async with app.run_test(size=(220, 60)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        await pilot.press("n")
        await pilot.pause()
        assert isinstance(pilot.app.screen, NewTaskChoiceModal)


# ---------------------------------------------------------------------------
# p: pause/resume, routed by kind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p_key_toggles_a_reminder_enabled_state(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="Nightly check",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
            enabled=True,
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await _select_row(pilot)

            await pilot.press("p")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_reminder_task(task_id)["enabled"] is False
    finally:
        db.close()


@pytest.mark.asyncio
async def test_p_key_on_a_locked_reminder_refuses_honestly(tmp_path):
    """The reminder branch reuses `action_toggle_enabled` verbatim, which
    already refuses via `_refuse_if_transfer_locked` -- pinned here via
    the actual `p` KEY (not a direct method call), end to end."""
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="Moving",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
            enabled=True,
        )
        db.set_transfer_state(
            "reminder_task", task_id, "to_server_pending", expected=(None,)
        )
        notifications: list[str] = []
        async with app.run_test(size=(220, 60)) as pilot:
            pilot.app.notify = lambda message, **kw: notifications.append(message)
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await _select_row(pilot)

            await pilot.press("p")
            await pilot.pause()

            assert notifications
            assert "read-only" in notifications[0]
            assert db.get_reminder_task(task_id)["enabled"] is True
    finally:
        db.close()


@pytest.mark.asyncio
async def test_p_key_toggles_a_definition_lifecycle(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={},
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await _select_row(pilot)

            await pilot.press("p")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert db.get_automation_definition(definition_id)["lifecycle"] == "paused"

            await pilot.press("p")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert (
                db.get_automation_definition(definition_id)["lifecycle"]
                == "configured"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_p_key_on_a_locked_definition_refuses_honestly(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={},
        )
        db.set_transfer_state(
            "automation_definition", definition_id, "to_server_pending",
            expected=(None,),
        )
        notifications: list[str] = []
        async with app.run_test(size=(220, 60)) as pilot:
            pilot.app.notify = lambda message, **kw: notifications.append(message)
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await _select_row(pilot)

            await pilot.press("p")
            await pilot.pause()

            assert notifications, "a locked row must refuse with a reason, not go silent"
            assert (
                db.get_automation_definition(definition_id)["lifecycle"] == "configured"
            ), "the lifecycle must be untouched by a refused toggle"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# m: open the selected row's Runs-on dropdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m_key_opens_the_runs_on_dropdown_for_a_reminder_row(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
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
            await _select_row(pilot)

            detail = pilot.app.screen.query_one("#scheduling-task-detail", TaskDetail)
            assert len(detail._runs_on_row.query(Select)) == 0

            await pilot.press("m")
            await pilot.pause()

            assert len(detail._runs_on_row.query(Select)) > 0, (
                "m must activate the row's own editor, same as Enter/click"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_m_key_opens_the_runs_on_dropdown_for_a_definition_row(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={},
        )
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await _select_row(pilot)

            detail = pilot.app.screen.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            assert len(detail._runs_on_row.query(Select)) == 0

            await pilot.press("m")
            await pilot.pause()

            assert len(detail._runs_on_row.query(Select)) > 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_m_key_with_nothing_selected_refuses_honestly():
    app = WorkbenchTestApp()
    notifications: list[str] = []
    async with app.run_test(size=(220, 60)) as pilot:
        pilot.app.notify = lambda message, **kw: notifications.append(message)
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        await pilot.press("m")
        await pilot.pause()

        assert notifications


# ---------------------------------------------------------------------------
# r: mark a definition row's unread results read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_key_marks_a_definitions_unread_results_read(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        definition_id = db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={},
        )
        result_id = db.create_automation_result(
            owner_id="local",
            definition_id=definition_id,
            run_id="run-1",
            kind="finding",
            title="Daily stand-up summary",
            summary="Two blockers reported.",
            dedupe_key="d1",
            answer="The team is blocked on CI flakiness.",
            source_refs=[{"source_type": "message", "source_id": "msg-1"}],
            review_state="unread",
        )
        assert result_id is not None
        async with app.run_test(size=(220, 60)) as pilot:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await _select_row(pilot)

            await pilot.press("r")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_automation_result(result_id)["review_state"] == "read"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_r_key_on_a_definition_with_nothing_unread_refuses_honestly(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        db.create_automation_definition(
            "local",
            "recurring_question",
            "Nightly digest",
            schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
            input={"question": "What shipped?"},
            config={},
        )
        notifications: list[str] = []
        async with app.run_test(size=(220, 60)) as pilot:
            pilot.app.notify = lambda message, **kw: notifications.append(message)
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await _select_row(pilot)

            await pilot.press("r")
            await pilot.pause()

            assert notifications
            assert "unread" in notifications[0].lower()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_r_key_on_a_reminder_row_is_an_honest_no_op(tmp_path):
    app = WorkbenchTestApp()
    db, service = _real_service(tmp_path, app)
    try:
        task_id = db.create_reminder_task(
            owner_id="local",
            title="Nightly check",
            schedule_kind="one_time",
            run_at="2030-01-01T00:00:00+00:00",
        )
        notifications: list[str] = []
        async with app.run_test(size=(220, 60)) as pilot:
            pilot.app.notify = lambda message, **kw: notifications.append(message)
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await _select_row(pilot)

            await pilot.press("r")
            await pilot.pause()

            assert notifications
            assert "no results" in notifications[0].lower()
            # Nothing about the task itself changes.
            assert db.get_reminder_task(task_id)["enabled"] is True
    finally:
        db.close()
