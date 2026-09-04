"""The 80x24 responsive floor for the single-surface Schedules workbench.

redesign PR-4, task 6 (spec S11, ruling 6). Below 84 columns the docked
detail region no longer blank-hides with a "widen the window" notice --
`Enter` on a queue row PUSHES the same pane class (`TaskDetail` /
`DefinitionDetail`) as a fresh instance inside Task 1's
`WorkbenchHostScreen`, fed by the same service-backed loads the docked
panes use. The four filter chips collapse to a single cycling control,
and the rail header degrades to a one-row button strip instead of
vanishing, so every spec-named operation stays reachable at 80x24.

Real `ScheduledTasksDB` + `SchedulingService` throughout (the same
rationale `test_schedules_keyboard_map.py` gives: routing correctness is
proven against the real facade, not a stub of it), and
`CSS_PATH = BUNDLED_STYLESHEET` so the width-driven `display` rules this
file asserts on actually resolve -- without the app tier every `.compact`
rule is absent and the geometry claims measure nothing.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, DataTable, Input, Select

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import settle_schedules_workbench
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.UI.Screens.scheduling.conflicts_tab import ConflictsTab
from tldw_chatbook.UI.Screens.scheduling.definition_detail import DefinitionDetail
from tldw_chatbook.UI.Screens.scheduling.forms.new_task_choice_modal import (
    NewTaskChoiceModal,
)
from tldw_chatbook.UI.Screens.scheduling.results_tab import ResultsTab
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    ReminderForm,
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail
from tldw_chatbook.UI.Screens.scheduling.workbench_host_screen import (
    WorkbenchHostScreen,
)
from tldw_chatbook.Widgets.detail_value_row import DetailValueRow

#: The floor the spec pins (S14) and the two wider layouts that must keep
#: their docked panes.
FLOOR = (80, 24)
MID = (110, 40)
WIDE = (220, 60)


class BundledCSSWorkbenchApp(ConsolidatedCSSApp):
    """Harness with the app CSS tier, where the `.compact` rules live."""

    CSS_PATH = BUNDLED_STYLESHEET
    scheduling_service = None


def _real_service(tmp_path, app: App) -> tuple[ScheduledTasksDB, SchedulingService]:
    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    service = SchedulingService(db=db, runtime_source="local", app_getter=lambda: app)
    app.scheduling_service = service
    return db, service


def _reminder(db: ScheduledTasksDB, title: str = "Nightly check") -> str:
    return db.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
        enabled=True,
    )


def _definition(db: ScheduledTasksDB, name: str = "Nightly digest") -> str:
    return db.create_automation_definition(
        "local",
        "recurring_question",
        name,
        schedule={"kind": "one_time", "run_at": "2030-01-01T00:00:00+00:00"},
        input={"question": "What shipped?"},
        config={},
    )


def _painted(app: App) -> str:
    """Every glyph the compositor actually paints, newline-joined.

    Textual 8.2.8 has no `App.export_text()`, so `Screen._compositor.
    render_strips()` is the only honest read of a painted frame (post-CSS,
    post-clip) -- the same idiom `test_compact_focus_outline_render.py`
    uses. `Widget.region` alone would happily report a plausible region for
    something the terminal never draws.
    """
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(seg.text for seg in strip) for strip in strips)


async def _open_workbench(pilot, size_note: str = "") -> SchedulesWorkbench:
    await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
    await pilot.pause()
    workbench = pilot.app.screen
    await settle_schedules_workbench(pilot, workbench)
    return workbench


async def _select_row(pilot, index: int = 0) -> None:
    table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
    table.focus()
    table.cursor_coordinate = (index, 0)
    await pilot.pause()


async def _push_detail(pilot) -> WorkbenchHostScreen:
    """Enter on the highlighted row, the way a user reaches the detail at
    the floor width."""
    await pilot.press("enter")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen


# ---------------------------------------------------------------------------
# The floor itself: what 80x24 paints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_at_the_floor_the_queue_is_full_width_and_the_rows_paint(tmp_path):
    """The rail owned 24 of 80 columns while the two hidden panes owned the
    rest, so the queue wrapped into a 20-column gutter and no row was
    painted at all. With the detail region hidden the queue takes the full
    width."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db, "Nightly check")
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)

            list_pane = workbench.query_one("#scheduling-list-pane")
            table = workbench.query_one("#scheduling-task-table", DataTable)
            assert workbench.query_one("#scheduling-detail-pane").display is False
            assert list_pane.region.width >= 70, (
                f"the queue rail is {list_pane.region.width} columns wide at "
                "80x24 while it is the only visible pane"
            )
            assert table.region.height >= 2, (
                f"the queue table gets {table.region.height} row(s) at 80x24"
            )
            assert "Nightly check" in _painted(pilot.app)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_at_the_floor_the_chips_collapse_to_the_cycling_control(tmp_path):
    """Spec S11: the four chips collapse into one cycling control rather
    than disappearing outright -- a mouse user keeps a filter affordance."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)

            cycle = workbench.query_one("#scheduling-chip-cycle", Button)
            assert cycle.display is True
            for chip_id in (
                "scheduling-chip-all",
                "scheduling-chip-active",
                "scheduling-chip-paused",
                "scheduling-chip-completed",
            ):
                assert workbench.query_one(f"#{chip_id}", Button).display is False
            assert "All" in str(cycle.label)

            assert await pilot.click("#scheduling-chip-cycle")
            await pilot.pause()
            await pilot.pause()
            assert workbench._chip == "active"
            assert "Active" in str(cycle.label)
            assert "Filter: Active" in _painted(pilot.app)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_at_the_floor_the_chip_keys_are_independent_of_chip_visibility(tmp_path):
    """Task 4's `1`-`4`/`f` bindings never read chip visibility; this pins
    that at the width where the four chips are actually hidden (the earlier
    pin forced `display` by hand at a wide size)."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            assert workbench.query_one("#scheduling-chip-all", Button).display is False

            await pilot.press("3")
            await pilot.pause()
            assert workbench._chip == "paused"
            assert "Paused" in str(
                workbench.query_one("#scheduling-chip-cycle", Button).label
            )

            await pilot.press("f")
            await pilot.pause()
            assert workbench._chip == "completed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_at_the_floor_the_rail_and_status_strip_stay_readable(tmp_path):
    """The rail header used to hide entirely below 84, taking `Create ▾`,
    `Mark all read` and `Results` with it. It degrades to a one-row button
    strip instead, and the status strip keeps its conflicts badge."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)

            header = workbench.query_one("#scheduling-list-header")
            assert header.display is True
            assert header.region.height == 1, (
                f"the rail header costs {header.region.height} rows at the "
                "floor; it must degrade to a single row"
            )
            painted = _painted(pilot.app)
            assert "Create" in painted
            assert "Results" in painted
            assert "Conflicts" in painted
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Enter pushes the hosted detail (ruling 6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_at_the_floor_enter_pushes_the_hosted_reminder_detail(tmp_path):
    """A FRESH `TaskDetail` instance -- never the docked pane reparented --
    painted with the row's own data."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db, "Nightly check")
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            docked = workbench.query_one("#scheduling-task-detail", TaskDetail)
            await _select_row(pilot)

            host = await _push_detail(pilot)

            assert isinstance(host, WorkbenchHostScreen)
            pushed = host.query_one(TaskDetail)
            assert pushed is not docked, "the docked pane was reparented"
            assert docked.is_mounted, "the docked pane must stay put"
            assert pushed._current_task is not None
            assert pushed._current_task.title == "Nightly check"
            assert "Nightly check" in _painted(pilot.app)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_at_the_floor_enter_pushes_the_hosted_definition_detail(tmp_path):
    """Same push for a definition row, and the pushed pane gets the same
    service-fed counts the docked pane's own loader reads."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        definition_id = _definition(db, "Nightly digest")
        db.create_automation_run(
            "local",
            definition_id,
            1,
            "manual",
            status="succeeded",
        )
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            docked = workbench.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            )
            await _select_row(pilot)

            host = await _push_detail(pilot)

            assert isinstance(host, WorkbenchHostScreen)
            pushed = host.query_one(DefinitionDetail)
            assert pushed is not docked
            assert pushed._definition is not None
            assert pushed._definition.get("name") == "Nightly digest"
            assert "Nightly digest" in _painted(pilot.app)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_escape_pops_the_pushed_detail_back_to_the_queue(tmp_path):
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            await _select_row(pilot)
            await _push_detail(pilot)

            await pilot.press("escape")
            await pilot.pause()

            assert pilot.app.screen is workbench
            assert workbench._pushed_detail is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_escape_with_a_hosted_editor_open_closes_the_editor_not_the_screen(
    tmp_path,
):
    """Task 1's review rider, landed at its first editor-bearing consumer.

    `WorkbenchHostScreen` binds `escape` to dismiss, and the hosted pane's
    rows bind `escape` to close an open in-pane editor. `DetailValueRow.
    _on_key` stops the key while an editor is open, so it never reaches the
    screen binding -- source-traced in task 1, PINNED here.
    """
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)
            await _select_row(pilot)
            host = await _push_detail(pilot)
            pushed = host.query_one(TaskDetail)

            row = pushed.runs_on_row
            row.focus()
            await pilot.pause()
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            assert row.query(Select), "the Runs-on editor did not open"

            await pilot.press("escape")
            await pilot.pause()

            assert pilot.app.screen is host, (
                "escape popped the host screen instead of closing the editor"
            )
            assert not row.query(Select), "the editor stayed open"

            await pilot.press("escape")
            await pilot.pause()
            assert pilot.app.screen is not host, (
                "escape with no editor open must still pop the host"
            )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Every operation, at the floor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_is_reachable_at_the_floor(tmp_path):
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)

            assert await pilot.click("#scheduling-new-task")
            await pilot.pause()
            assert isinstance(pilot.app.screen, NewTaskChoiceModal)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_edit_in_full_is_reachable_from_the_pushed_detail(tmp_path):
    """The pushed pane's `Edit` button posts `EditTaskRequested`, which
    bubbles to the HOST screen -- the workbench underneath never sees a
    pushed screen's messages unless they are routed there."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)
            await _select_row(pilot)
            host = await _push_detail(pilot)

            host.query_one("#scheduling-edit-task", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert isinstance(pilot.app.screen, ReminderForm)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_pause_resume_is_reachable_from_the_pushed_detail(tmp_path):
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        definition_id = _definition(db)
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)
            await _select_row(pilot)
            host = await _push_detail(pilot)

            host.query_one("#scheduling-automation-pause-resume", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert db.get_automation_definition(definition_id)["lifecycle"] == "paused"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_an_in_pane_edit_inside_the_pushed_detail_persists(tmp_path):
    """The in-pane row editors work inside the pushed pane: the commit
    message routes to the workbench's own persistence bridge."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        task_id = _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)
            await _select_row(pilot)
            host = await _push_detail(pilot)
            pushed = host.query_one(TaskDetail)

            row = pushed._at_row
            row.focus()
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()
            editor = row.query_one(Input)
            editor.focus()
            editor.value = "2031-02-03 04:05"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert "2031-02-03" in str(db.get_reminder_task(task_id)["run_at"])
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_transfer_dropdown_opens_inside_the_pushed_detail(tmp_path):
    """Ruling 2's one transfer surface -- the Runs-on row's dropdown --
    reachable at the floor from inside the pushed pane."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)
            await _select_row(pilot)
            host = await _push_detail(pilot)
            pushed = host.query_one(TaskDetail)

            row = pushed.runs_on_row
            row.focus()
            row.post_message(DetailValueRow.Activated(row))
            await pilot.pause()

            assert row.query(Select), "no owner picker opened on the Runs-on row"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_results_and_conflicts_are_reachable_at_the_floor(tmp_path):
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)

            assert await pilot.click("#scheduling-results-badge")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert pilot.app.screen.query(ResultsTab)

            await pilot.press("escape")
            await pilot.pause()

            assert await pilot.click("#scheduling-conflicts-badge")
            await pilot.pause()
            assert pilot.app.screen.query(ConflictsTab)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# The wider layouts stay docked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_mid_width_layout_keeps_the_docked_detail_and_does_not_push(tmp_path):
    """~110 columns: the detail pane is docked (only the inspector yields),
    so Enter must NOT push a second copy of it."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db, "Nightly check")
        async with app.run_test(size=MID) as pilot:
            workbench = await _open_workbench(pilot)

            assert workbench.query_one("#scheduling-detail-pane").display is True
            assert workbench.query_one("#scheduling-inspector-pane").display is False
            assert workbench.query_one("#scheduling-chip-cycle", Button).display is False
            assert workbench.query_one("#scheduling-chip-all", Button).display is True
            table = workbench.query_one("#scheduling-task-table", DataTable)
            assert table.region.height >= 2

            await _select_row(pilot)
            await pilot.press("enter")
            await pilot.pause()
            assert pilot.app.screen is workbench
            assert workbench._pushed_detail is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_full_width_layout_keeps_all_three_panes(tmp_path):
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db, "Nightly check")
        async with app.run_test(size=WIDE) as pilot:
            workbench = await _open_workbench(pilot)

            for pane_id in (
                "#scheduling-list-pane",
                "#scheduling-detail-pane",
                "#scheduling-inspector-pane",
            ):
                assert workbench.query_one(pane_id).display is True
            assert not workbench.query_one("#scheduling-workbench").has_class("compact")
            assert workbench.query_one("#scheduling-list-header").region.height == 3
            assert workbench.query_one("#scheduling-chip-cycle", Button).display is False
            assert "Nightly check" in _painted(pilot.app)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pushed-detail data correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_pushed_pane_repaints_from_the_same_refresh_as_the_docked_one(
    tmp_path,
):
    """A mutation refreshes the queue and re-feeds the detail; the pushed
    instance is fed by that SAME seam, so it never shows a value the
    docked pane has already moved past."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        task_id = _reminder(db, "Nightly check")
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            await _select_row(pilot)
            host = await _push_detail(pilot)
            pushed = host.query_one(TaskDetail)

            db.update_reminder_task(task_id, title="Renamed check")
            workbench._request_tasks_refresh(refresh_definitions=False)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert pushed._current_task is not None
            assert pushed._current_task.title == "Renamed check"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Final-review fix wave: the floor's remaining traps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m_at_the_floor_opens_the_dropdown_in_the_pane_the_user_can_see(
    tmp_path,
):
    """F1: `m` below the threshold must not activate the HIDDEN pane.

    It used to mount a `Select` inside `#scheduling-detail-pane` while that
    pane was `display: none` -- a zero-region editor that painted nothing,
    stole focus off the queue table (so Up/Down stopped moving the list) and
    said nothing. `m` now takes the same push route `Enter` does and
    activates the Runs-on row of the pane that is actually on screen.
    """
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db)
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            await _select_row(pilot)
            assert workbench._detail_hidden(), "the floor must hide the docked pane"

            await pilot.press("m")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            host = pilot.app.screen
            assert isinstance(host, WorkbenchHostScreen), (
                "`m` at the floor must push the detail, not activate a hidden pane"
            )
            row = host.query_one(TaskDetail).runs_on_row
            editors = row.query(Select)
            assert editors, "no owner picker opened in the pushed pane"
            region = editors.first().region
            assert region.width > 0 and region.height > 0, (
                f"the dropdown is unpainted ({region}) -- the zero-region trap"
            )
            # The invisible docked pane was left alone.
            docked = workbench.query_one("#scheduling-task-detail", TaskDetail)
            assert not docked.query(Select)

            # Escape closes the editor, Escape again returns to a WORKING queue
            # (the old trap left the list unresponsive with nothing painted).
            await pilot.press("escape")
            await pilot.pause()
            assert pilot.app.screen is host
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert pilot.app.screen is workbench
            table = workbench.query_one("#scheduling-task-table", DataTable)
            table.focus()
            await pilot.press("down")
            await pilot.pause()
            assert workbench.focused is table, "the queue never got its keys back"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_m_at_the_floor_on_an_empty_queue_refuses_out_loud(tmp_path):
    """F1's other half: no row under the cursor still says why."""
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        async with app.run_test(size=FLOOR) as pilot:
            await _open_workbench(pilot)
            notifications: list[str] = []
            pilot.app.notify = lambda message, **kw: notifications.append(message)

            await pilot.press("m")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, WorkbenchHostScreen)
            assert notifications, "`m` with nothing to move must not go silent"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_pushed_detail_never_repaints_from_another_rows_data(tmp_path):
    """F2: the pushed pane is pinned to the row it was pushed FOR.

    `_render_table`'s restore falls back to `target_index = 0` whenever the
    selected row is gone, and every re-feed used to reach the pushed instance
    too -- so a pane headed "ZZZ Second reminder" could end up painting (and
    wiring its Delete button to) "AAA First reminder". The identity gate in
    `_detail_panes` makes that frame impossible.
    """
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db, "AAA First reminder")
        _reminder(db, "ZZZ Second reminder")
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            rows = workbench._visible_rows
            zzz_index = next(
                index
                for index, row in enumerate(rows)
                if row.title == "ZZZ Second reminder"
            )
            await _select_row(pilot, zzz_index)
            host = await _push_detail(pilot)
            pushed = host.query_one(TaskDetail)
            assert pushed._current_task.title == "ZZZ Second reminder"
            assert host.title == "ZZZ Second reminder"

            # The seam every refresh goes through, aimed at the OTHER row.
            other_index = 1 - zzz_index
            workbench._update_detail_for_index(other_index)
            await pilot.pause()

            docked = workbench.query_one("#scheduling-task-detail", TaskDetail)
            assert docked._current_task.title == "AAA First reminder"
            assert pushed._current_task.title == "ZZZ Second reminder", (
                "the pushed pane retargeted while its Header still named ZZZ"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_pushed_detail_closes_with_a_notice_when_its_row_is_gone(tmp_path):
    """F2's honest gone-state (the documented choice: auto-pop + notice).

    A background refresh that drops the open row leaves the pane nothing true
    to show -- and its Delete / Disable / Run-now controls pointed at a
    reminder that no longer exists. Popping removes both problems.
    """
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        _reminder(db, "AAA First reminder")
        zzz_id = _reminder(db, "ZZZ Second reminder")
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)
            zzz_index = next(
                index
                for index, row in enumerate(workbench._visible_rows)
                if row.title == "ZZZ Second reminder"
            )
            await _select_row(pilot, zzz_index)
            host = await _push_detail(pilot)
            pushed = host.query_one(TaskDetail)

            notifications: list[str] = []
            pilot.app.notify = lambda message, **kw: notifications.append(message)

            db.delete_reminder_task(zzz_id)
            workbench._request_tasks_refresh(refresh_definitions=False)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert pilot.app.screen is workbench, "the stale pane stayed open"
            assert workbench._pushed_detail is None
            assert workbench._pushed_row_id is None
            assert any("ZZZ Second reminder" in text for text in notifications), (
                "the pane vanished without saying why"
            )
            assert pushed._current_task.title == "ZZZ Second reminder", (
                "the pane took another row's data on its way out"
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_resolving_a_conflict_in_the_pushed_view_reloads_the_queue(tmp_path):
    """F3: `ConflictsTab.ConflictResolved` reaches the workbench LIVE.

    Task 5 moved the only `ConflictsTab` instance onto a pushed screen, and
    that push carried no `route_message` -- so the message bubbled tab -> host
    -> App and `@on(ConflictsTab.ConflictResolved)` never ran. The badge count
    still updated on pop, which is why the loss looked like nothing.

    Asserted through the REAL message path (post from the hosted tab), never a
    direct `_on_conflict_resolved(...)` call -- a direct call is exactly what
    could not observe this break.
    """
    app = BundledCSSWorkbenchApp()
    db, _service = _real_service(tmp_path, app)
    try:
        task_id = _reminder(db, "Losing side")
        async with app.run_test(size=FLOOR) as pilot:
            workbench = await _open_workbench(pilot)

            assert await pilot.click("#scheduling-conflicts-badge")
            await pilot.pause()
            host = pilot.app.screen
            tab = host.query_one(ConflictsTab)

            # What "the winning side" looks like to the queue behind.
            db.update_reminder_task(task_id, title="Winning side")
            tab.post_message(ConflictsTab.ConflictResolved("conflict-1", "local"))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert pilot.app.screen is host, "the overlay must still be open"
            assert [task.title for task in workbench._tasks] == ["Winning side"], (
                "resolving a conflict did not reload the queue behind the overlay"
            )
    finally:
        db.close()
