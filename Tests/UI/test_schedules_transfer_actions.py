"""Detail-pane transfer-lock + toast-copy tests (schedules-handoff spec §6,
PR-5 task 7; narrowed by redesign PR-4 task 4).

redesign PR-4 task 4 (ruling 2) retired the legacy button-driven transfer
surface this file used to cover in depth (`TaskDetail`'s Move/Retry/Cancel
buttons, `set_transfer_reasons`, and the Automations-tab-only `m`/`M`/`y`/`k`
keybindings driving `_begin_automation_transfer`/`_cancel_automation_
transfer`) -- the Runs-on row's dropdown + mini-bar (PR-3 task 5,
`test_schedules_workbench.py`'s `test_runs_on_dropdown_*`/`test_definition_
runs_on_*`/`test_runs_on_*_button_*` tests) is now the ONE transfer surface,
and already carries equivalent coverage for the routing/confirm-dialog/
begin/cancel/retry behaviors those deleted tests pinned. What survives here:

- Widget-level (`_DetailHarnessApp`, a bare `TaskDetail`): `set_lifecycle_
  lock`'s Edit/Enable/Disable/Delete read-only-with-reason rendering --
  unaffected by the legacy button deletion (a SEPARATE lock surface from
  the retired `set_transfer_reasons`).
- Workbench-level (`TransferWorkbenchTestApp`, a REAL `SchedulingService`):
  the same lock, sourced from the real facade's `transfer_lock_reason`, and
  the `e`/`space` key-bound refusal it drives.
- The DataTable row's own transfer-badge suffix rendering (never routed
  through the legacy buttons at all).
- `_transfer_confirm_dialog`'s literal-bracket escaping (a shared helper,
  still used by the Runs-on dropdown flow).
- Direct unit tests for `_cancel_toast_text`/`_transfer_pending_toast_text`
  (shared, unchanged pure functions -- the dropdown flow's `_run_owner_
  transfer`/`_run_owner_cancel` already called them before this task) --
  the deleted tests' honest-copy assertions, preserved as a cheap direct-
  call belt rather than re-adding a full integration round-trip that
  `test_runs_on_dropdown_confirm_begins_to_server_transfer`/`test_runs_on_
  cancel_button_cancels_the_dormant_copy_using_its_own_id`
  (`test_schedules_workbench.py`) already exercise for the surrounding
  DB-state behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from textual import on
from textual.widgets import Button, DataTable, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.events import DuplicateTaskRequested
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
    _cancel_toast_text,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail


def _reminder(**kwargs) -> ReminderTask:
    defaults = dict(
        id="task-1",
        title="Backup check",
        schedule_kind=ScheduleKind.RECURRING,
        cron="0 * * * *",
        timezone="UTC",
    )
    defaults.update(kwargs)
    return ReminderTask(**defaults)


# ---------------------------------------------------------------------------
# Widget-level: TaskDetail's lifecycle-lock rendering (the legacy button
# show/hide + `set_transfer_reasons` disabled-with-reason tests that used to
# live here were deleted with the buttons -- redesign PR-4 task 4)
# ---------------------------------------------------------------------------


class _DetailHarnessApp(ConsolidatedCSSApp):
    """Bare app mounting one TaskDetail, matching the workbench's compose."""

    def compose(self):
        yield TaskDetail()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["to_server_pending", "to_server_sent", "from_server_pending"]
)
async def test_in_flight_row_disables_lifecycle_actions_with_a_reason(state):
    """Final review I7 / spec §6.3: an in-flight row is read-only except
    cancel. The reason must be in TEXT, not only a tooltip (UX-073).

    task-31823 AC#2: `Duplicate` joins Edit/Delete in the same gate -- a
    row mid-transfer is not a settled row to fork from."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(owner_id="local", transfer_state=state))
        detail.set_lifecycle_lock("This row is moving -- read-only.")

        for button_id in (
            "scheduling-edit-task",
            "scheduling-delete-task",
            "scheduling-enable-task",
            "scheduling-disable-task",
            "scheduling-duplicate-task",
        ):
            assert detail.query_one(f"#{button_id}", Button).disabled, button_id
        why = detail.query_one("#scheduling-transfer-why", Static)
        assert "read-only" in why.visual.plain


@pytest.mark.asyncio
async def test_clearing_the_lifecycle_lock_restores_the_buttons():
    """An editable row keeps its normal enable/disable logic, and the lock
    line is removed from the shared reason Static."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(owner_id="local", transfer_state="to_server_sent"))
        detail.set_lifecycle_lock("locked for now")
        detail.set_task(_reminder(owner_id="local", transfer_state=None, enabled=True))
        detail.set_lifecycle_lock(None)

        assert not detail.query_one("#scheduling-edit-task", Button).disabled
        assert not detail.query_one("#scheduling-delete-task", Button).disabled
        assert not detail.query_one("#scheduling-duplicate-task", Button).disabled
        # set_task's own UX-059 rule still owns these two.
        assert detail.query_one("#scheduling-enable-task", Button).disabled
        assert not detail.query_one("#scheduling-disable-task", Button).disabled
        why = detail.query_one("#scheduling-transfer-why", Static)
        assert "locked for now" not in why.visual.plain


# ---------------------------------------------------------------------------
# task-31823: the reminder pane's secondary-actions row (Duplicate / View
# runs / View results -- the rest of spec §5's deferred kebab list).
# ---------------------------------------------------------------------------


class _CapturingTaskDetailApp(_DetailHarnessApp):
    """`_DetailHarnessApp`'s twin with its own message capture (mirrors
    `test_schedules_workbench.py`'s `_CapturingDefinitionDetailApp`): a
    plain bare harness cannot observe a posted `Message` bubbling past
    it -- an `@on` handler on the App itself records what `TaskDetail`
    posts, no workbench needed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.duplicate_events: list = []

    @on(DuplicateTaskRequested)
    def _capture_duplicate(self, event: DuplicateTaskRequested) -> None:
        self.duplicate_events.append(event.task)


@pytest.mark.asyncio
async def test_duplicate_button_posts_duplicate_task_requested():
    """AC#1: `Duplicate` is reachable from the reminder pane and posts
    `DuplicateTaskRequested` carrying the painted task -- the pane
    performs no I/O of its own (same "post a message" shape every other
    lifecycle button here already uses)."""
    async with _CapturingTaskDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        task = _reminder(owner_id="local")
        detail.set_task(task)
        await pilot.pause()

        button = detail.query_one("#scheduling-duplicate-task", Button)
        assert button.disabled is False
        detail.on_button_pressed(Button.Pressed(button))
        await pilot.pause()

        assert pilot.app.duplicate_events == [task]


@pytest.mark.asyncio
async def test_view_results_is_permanently_disabled_with_a_reason():
    """AC#2 (UX-073): reminders have no automation-results surface at
    all (results are keyed by `definition_id`, a `recurring_question`
    concept), so `View results` is disabled unconditionally -- unlike
    the mid-transfer gates above, this never re-evaluates per task. The
    reason lives in an always-visible Static, not just the tooltip."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(owner_id="local"))
        await pilot.pause()

        button = detail.query_one("#scheduling-view-results-task", Button)
        assert button.disabled is True
        assert "recurring questions" in str(button.tooltip)
        why = detail.query_one("#scheduling-task-detail-secondary-why", Static)
        assert "recurring questions" in why.visual.plain


@pytest.mark.asyncio
async def test_view_runs_button_scrolls_the_run_history_section_into_view():
    """AC#1: `View runs` is reachable from the reminder pane and reuses
    the SAME scroll-to-"Recent runs:" affordance 31712 AC#5's history
    link row already performs -- reminders have no separate run-history
    VIEW to push, only this inline section."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(owner_id="local"))
        await pilot.pause()

        run_history = detail.query_one("#scheduling-task-detail-run-history", Static)
        calls: list = []
        run_history.scroll_visible = lambda *args, **kwargs: calls.append(True)

        button = detail.query_one("#scheduling-view-runs-task", Button)
        assert button.disabled is False
        detail.on_button_pressed(Button.Pressed(button))
        await pilot.pause()

        assert calls == [True]


# ---------------------------------------------------------------------------
# Workbench-level: routing, confirm dialog, and cancel/toast copy units
# ---------------------------------------------------------------------------


@pytest.fixture
def transfer_db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


class _FakeServerClient:
    """Reads as "a server is connected" (`notifications_service is not
    None`) without making any real network call -- `begin_transfer_to_*`
    only records a local mutation, it never calls the server client
    directly (that is `SyncEngine`'s job, out of this file's scope)."""

    def __init__(self) -> None:
        self.notifications_service = object()


class TransferWorkbenchTestApp(ConsolidatedCSSApp):
    """A real Textual test app wired to a REAL `SchedulingService` over a
    tmp_path DB. `transfer_refusal`/`transfer_warnings`/`begin_transfer_*`/
    `cancel_transfer` all run for real (Task 6's own suite is their
    correctness proof); this app only proves the workbench calls them
    right and renders what they say.
    """

    def __init__(
        self, db, *args, connected: bool = True, server_client=None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        active_server_id = "1" if connected else None
        self.active_server_id = active_server_id
        self.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(active_server_id=active_server_id)
        )
        if server_client is None and connected:
            server_client = _FakeServerClient()
        self.scheduling_service = SchedulingService(
            db=db,
            server_client=server_client,
            runtime_source="local",
            app_getter=lambda: self,
        )


async def _select_row(pilot, index: int = 0) -> None:
    table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
    table.cursor_coordinate = (index, 0)
    await pilot.pause()


@pytest.mark.asyncio
async def test_queue_row_shows_transfer_badge_suffix(transfer_db):
    """A minimal, always-on signal that the state machine is doing
    something -- spec §9's badge language, pulled forward just far enough
    (plan ruling 1 keeps full badge/owner-column polish PR-6 scope).
    Never routed through the (now-retired) legacy transfer buttons."""
    db = transfer_db
    reminder_id = db.create_reminder_task(
        owner_id="local",
        title="Nightly check",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    db.set_transfer_state(
        "reminder_task", reminder_id, "to_server_pending", expected=(None,)
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        row = table.get_row_at(0)
        # redesign PR-2, Task 2: column 0 is now the glyph, column 1 the
        # title (old single-primitive shape was Title/Type/Status/Next Run).
        assert "Moving to server" in str(row[1])


@pytest.mark.asyncio
async def test_workbench_locks_lifecycle_for_a_transferring_row(transfer_db):
    """The workbench sources the lock from the REAL facade's
    `transfer_lock_reason` on every selection."""
    db = transfer_db
    task_id = db.create_reminder_task(
        owner_id="local",
        title="Moving",
        schedule_kind="one_time",
        run_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )
    db.set_transfer_state(
        "reminder_task", task_id, "to_server_pending", expected=(None,)
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_row(pilot)

        edit_button = pilot.app.screen.query_one("#scheduling-edit-task", Button)
        assert edit_button.disabled
        why = pilot.app.screen.query_one("#scheduling-transfer-why", Static)
        assert "read-only" in why.visual.plain


@pytest.mark.asyncio
async def test_key_bound_edit_refuses_on_a_transferring_row(transfer_db):
    """The `e`/`d`/`space` verbs share the same lock -- pressing them says
    why instead of silently no-oping against the facade guard."""
    db = transfer_db
    task_id = db.create_reminder_task(
        owner_id="local",
        title="Moving",
        schedule_kind="one_time",
        run_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )
    db.set_transfer_state(
        "reminder_task", task_id, "from_server_pending", expected=(None,)
    )
    app = TransferWorkbenchTestApp(db)
    notifications: list[str] = []
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_row(pilot)
        pilot.app.notify = lambda message, **kw: notifications.append(message)

        pilot.app.screen.action_edit_task()
        pilot.app.screen.action_toggle_enabled()
        await pilot.pause()

        assert len(notifications) == 2
        assert all("read-only" in message for message in notifications)


@pytest.mark.asyncio
async def test_transfer_confirm_dialog_renders_brackets_literally():
    """Task 6's escaping saga, fourth surface.

    The Move dialog's copy interpolates a user-authored definition/task
    name and the server's own warning strings, and is rendered by a
    Textual `Label` -> `Content.from_markup`, whose tokenizer consumes
    ANY `[...]`. It was escaped with `rich.markup.escape`, which only
    covers `[a-z#/@]...` tags -- so an uppercase token was dropped and a
    lowercase one was safe, the exact split that hid the detail-pane bug
    in round 1.

    Asserted on the RENDERED label: `ConfirmationDialog.message` is the
    pre-render string and would pass either way.
    """
    dialog = SchedulesWorkbench._transfer_confirm_dialog(
        'Nightly [PR-6] digest',
        "to_server",
        ["Field [bold] does not transfer"],
    )
    app = ConsolidatedCSSApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(dialog)
        await pilot.pause()
        rendered = str(pilot.app.screen.query_one(".dialog-message").render())
        assert "Nightly [PR-6] digest" in rendered
        assert "Field [bold] does not transfer" in rendered


def test_cancel_toast_does_not_promise_no_server_effect():
    """Final review L13: `from_server_pending` also covers a release whose
    delete already landed but whose ack was lost, so the toast may not
    claim nothing happened -- only that nothing further will be sent.

    redesign PR-4 task 4: was an integration test driven through the now-
    retired legacy Cancel button; `_cancel_toast_text` is unchanged and
    still the SAME function the Runs-on row's own Cancel now drives
    (`_run_owner_cancel`, also pinned end-to-end in `test_schedules_
    workbench.py::test_runs_on_cancel_button_cancels_the_dormant_copy_
    using_its_own_id`) -- a direct call is the cheapest belt for the copy
    itself.
    """
    message = _cancel_toast_text("Standup")
    assert "nothing further will be sent" in message
    assert "no server-side effect" not in message


def test_transfer_pending_toast_is_honest_about_still_running_locally():
    """spec §6.1.1: a queued-not-sent `to_server` transfer must not claim
    the task stopped running here -- it still executes locally until the
    server accepts it. redesign PR-4 task 4: was an integration test
    driven through the now-retired legacy Move-to-server button; `_
    transfer_pending_toast_text` is unchanged and still the SAME
    `@staticmethod` the Runs-on row's own dropdown now drives
    (`_run_owner_transfer`, also pinned end-to-end in `test_schedules_
    workbench.py::test_runs_on_dropdown_confirm_begins_to_server_
    transfer`) -- a direct call is the cheapest belt for the copy
    itself."""
    to_server = SchedulesWorkbench._transfer_pending_toast_text(
        "Standup", "to_server"
    )
    assert "still runs on this device" in to_server

    to_local = SchedulesWorkbench._transfer_pending_toast_text(
        "Standup", "to_local"
    )
    assert "dormant copy" in to_local
