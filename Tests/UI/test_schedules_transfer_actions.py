"""Detail-pane Move/Cancel transfer UI tests (schedules-handoff spec §6,
PR-5 task 7).

Two layers, matching `test_schedules_missed_notice.py`'s split:

- Widget-level (`_DetailHarnessApp`, a bare `TaskDetail`): pins the
  disabled-with-reason rendering pipeline itself -- button show/hide per
  row structure, and `set_transfer_reasons` quoting whatever reason
  string it is given verbatim (including a health-shaped one), without
  needing a real scheduling service.
- Workbench-level (`TransferWorkbenchTestApp`, a REAL `SchedulingService`
  over a tmp_path `ScheduledTasksDB`): pins the UI's *routing* --
  `transfer_refusal`/`transfer_warnings` run for real (Task 6 owns their
  correctness; this file only proves the UI calls them and acts on what
  they say), confirm-dialog wiring, honest toast copy, and that cancel
  is called with the right row id per state (the dormant-copy case in
  particular -- Task 6's handoff note: the release leg's `cancel_
  transfer` row_id is the COPY's own id, never the mirror's).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from textual.widgets import Button, DataTable, Static, TabbedContent

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.Scheduling.services import (
    scheduling_service as scheduling_service_module,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

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
# Widget-level: TaskDetail's own show/hide + disabled-with-reason rendering
# ---------------------------------------------------------------------------


class _DetailHarnessApp(ConsolidatedCSSApp):
    """Bare app mounting one TaskDetail, matching the workbench's compose."""

    def compose(self):
        yield TaskDetail()


def _buttons(detail: TaskDetail) -> dict[str, Button]:
    return {
        "to_server": detail.query_one("#scheduling-transfer-to-server", Button),
        "to_local": detail.query_one("#scheduling-transfer-to-local", Button),
        "retry": detail.query_one("#scheduling-retry-transfer", Button),
        "cancel": detail.query_one("#scheduling-cancel-transfer", Button),
    }


@pytest.mark.asyncio
async def test_local_row_shows_only_move_to_server():
    """A plain local row (never transferred) offers only Move to server."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(owner_id="local", transfer_state=None))
        buttons = _buttons(detail)
        assert buttons["to_server"].display
        assert not buttons["to_local"].display
        assert not buttons["retry"].display
        assert not buttons["cancel"].display


@pytest.mark.asyncio
async def test_server_mirror_shows_only_move_to_local():
    """A server-owned mirror (never transferring) offers only Move to local."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _reminder(owner_id="server:1", server_id="srv-1", transfer_state=None)
        )
        buttons = _buttons(detail)
        assert not buttons["to_server"].display
        assert buttons["to_local"].display
        assert not buttons["retry"].display
        assert not buttons["cancel"].display


@pytest.mark.asyncio
async def test_to_server_failed_shows_retry_alongside_move_to_server():
    """spec §6.1.5: a definitively-failed transfer offers Retry (plus the
    Cancel escape hatch) alongside the still-shown Move to server."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _reminder(owner_id="local", transfer_state="to_server_failed")
        )
        buttons = _buttons(detail)
        assert buttons["to_server"].display
        assert not buttons["to_local"].display
        assert buttons["retry"].display
        assert buttons["cancel"].display


@pytest.mark.asyncio
async def test_dormant_local_copy_shows_cancel_only():
    """A `from_server_pending` copy (owner_id local) can only be cancelled."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _reminder(owner_id="local", transfer_state="from_server_pending")
        )
        buttons = _buttons(detail)
        assert buttons["to_server"].display
        assert not buttons["retry"].display
        assert buttons["cancel"].display


@pytest.mark.asyncio
async def test_selecting_a_projection_hides_the_whole_transfer_row():
    """A ScheduledTask projection (watchlist/briefing) has no transfer_state
    at all -- the row must not appear (or error querying it)."""
    from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus

    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            ScheduledTask(
                id="watchlist:1",
                title="Watchlist",
                type="watchlist_job",
                status=TaskStatus.WAITING,
            )
        )
        transfer_row = detail.query_one("#scheduling-task-detail-transfer")
        assert not transfer_row.display


@pytest.mark.asyncio
async def test_set_transfer_reasons_renders_disabled_with_reason():
    """A refusal reason (health-quoted, verbatim) disables the button AND
    appears in the always-visible Static (UX-073 -- tooltip alone is not
    enough for keyboard users)."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(owner_id="server:1", server_id="srv-1"))
        health_reason = (
            "Local health is not ready: RAG dependencies are not installed."
        )
        detail.set_transfer_reasons(
            to_server_reason=None,
            to_local_reason=health_reason,
            retry_reason=None,
            cancel_reason=None,
            retry_errors=[],
        )
        buttons = _buttons(detail)
        assert buttons["to_local"].disabled
        assert str(buttons["to_local"].tooltip) == health_reason
        why = detail.query_one("#scheduling-transfer-why", Static)
        assert f"Move to local: {health_reason}" in why.visual.plain


@pytest.mark.asyncio
async def test_set_transfer_reasons_shows_retry_errors():
    """The stored `transfer_errors` render beside Retry (Task 6's ruling 3)."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(transfer_state="to_server_failed"))
        detail.set_transfer_reasons(
            to_server_reason=None,
            to_local_reason=None,
            retry_reason=None,
            cancel_reason=None,
            retry_errors=["schedule_invalid: cron field out of range"],
        )
        why = detail.query_one("#scheduling-transfer-why", Static)
        assert "schedule_invalid: cron field out of range" in why.visual.plain


@pytest.mark.asyncio
async def test_allowed_action_clears_disabled_and_reason():
    """`None` reasons re-enable and clear any stale reason text."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(owner_id="local"))
        detail.set_transfer_reasons(
            to_server_reason="blocked",
            to_local_reason=None,
            retry_reason=None,
            cancel_reason=None,
            retry_errors=[],
        )
        detail.set_transfer_reasons(
            to_server_reason=None,
            to_local_reason=None,
            retry_reason=None,
            cancel_reason=None,
            retry_errors=[],
        )
        buttons = _buttons(detail)
        assert not buttons["to_server"].disabled
        why = detail.query_one("#scheduling-transfer-why", Static)
        assert why.visual.plain == ""


# ---------------------------------------------------------------------------
# Workbench-level: routing, confirm dialog, toasts, cancel row-id correctness
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
async def test_move_to_server_button_opens_confirm_dialog_with_warnings(
    transfer_db,
):
    """An imminent one-time run_at surfaces in the confirm dialog (spec §6.4)."""
    db = transfer_db
    imminent = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    db.create_reminder_task(
        owner_id="local",
        title="Almost due",
        schedule_kind="one_time",
        run_at=imminent,
        timeout_seconds=30,
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_row(pilot)

        button = pilot.app.screen.query_one(
            "#scheduling-transfer-to-server", Button
        )
        assert not button.disabled
        button.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(pilot.app.screen, ConfirmationDialog)
        message = pilot.app.screen.message
        assert "Almost due" in message
        assert "unverified" in message  # imminent run_at warning
        assert "timeout_seconds" in message  # non-transferring field


@pytest.mark.asyncio
async def test_confirming_move_to_server_queues_transfer_with_honest_toast(
    transfer_db, monkeypatch
):
    """Confirming starts the transfer, and the toast says the task still
    runs locally while only queued (spec §6.1.1)."""
    db = transfer_db
    reminder_id = db.create_reminder_task(
        owner_id="local",
        title="Nightly check",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    app = TransferWorkbenchTestApp(db)
    notifications: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, severity="information", **kw: notifications.append(
            (message, severity)
        ),
    )
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_row(pilot)

        pilot.app.screen.query_one(
            "#scheduling-transfer-to-server", Button
        ).press()
        await pilot.pause()
        assert isinstance(pilot.app.screen, ConfirmationDialog)
        pilot.app.screen.dismiss(True)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_reminder_task(reminder_id)
        assert row["transfer_state"] == "to_server_pending"
        assert row["owner_id"] == "local"  # still executes locally while queued

        toast_messages = [message for message, _ in notifications]
        assert any(
            "still runs on this device" in message for message in toast_messages
        )


@pytest.mark.asyncio
async def test_cancelling_declines_when_dialog_dismissed(transfer_db):
    """Dismissing the confirm dialog leaves the row untouched."""
    db = transfer_db
    reminder_id = db.create_reminder_task(
        owner_id="local",
        title="Nightly check",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_row(pilot)

        pilot.app.screen.query_one(
            "#scheduling-transfer-to-server", Button
        ).press()
        await pilot.pause()
        pilot.app.screen.dismiss(False)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_reminder_task(reminder_id)
        assert row["transfer_state"] is None


@pytest.mark.asyncio
async def test_cancel_on_dormant_copy_uses_the_copys_own_id(transfer_db):
    """Task 6 handoff note: the release leg's `cancel_transfer` row_id is
    the DORMANT COPY's own id, never the mirror's. Selecting the copy row
    (which is what the UI actually shows -- the mirror never carries
    `from_server_pending` itself) and pressing Cancel must delete exactly
    that copy and leave the mirror alone.
    """
    db = transfer_db
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-9",
        title="Server task",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        # Start the release for real through the facade so the dormant
        # copy is created exactly the way begin_transfer_to_local does.
        service = app.scheduling_service
        outcome = await service.begin_transfer_to_local("reminder_task", mirror_id)
        assert outcome.status == "pending"
        copy_id = outcome.row_id
        assert copy_id is not None
        assert copy_id != mirror_id

        # The device view (owner_id="local", the service's default) is
        # what the copy lives under -- reload the queue and select it.
        await pilot.app.screen.load_tasks()
        await pilot.pause()
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        assert table.row_count == 1  # only the dormant copy is local-owned
        await _select_row(pilot)

        cancel_button = pilot.app.screen.query_one(
            "#scheduling-cancel-transfer", Button
        )
        assert not cancel_button.disabled
        cancel_button.press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.get_reminder_task(copy_id) is None  # the copy is gone
        mirror_row = db.get_reminder_task(mirror_id)
        assert mirror_row is not None  # the mirror is untouched
        assert mirror_row["transfer_state"] is None


@pytest.mark.asyncio
async def test_retry_button_only_appears_after_definitive_failure(transfer_db):
    """Retry is invisible on a healthy local row and appears once the row
    is `to_server_failed`, carrying the stored transfer_errors."""
    db = transfer_db
    reminder_id = db.create_reminder_task(
        owner_id="local",
        title="Flaky sync",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_row(pilot)

        retry_button = pilot.app.screen.query_one(
            "#scheduling-retry-transfer", Button
        )
        assert not retry_button.display

        db.set_transfer_state(
            "reminder_task", reminder_id, "to_server_pending", expected=(None,)
        )
        db.set_transfer_state(
            "reminder_task",
            reminder_id,
            "to_server_sent",
            expected=("to_server_pending",),
        )
        db.set_transfer_state(
            "reminder_task",
            reminder_id,
            "to_server_failed",
            expected=("to_server_sent",),
        )
        db.record_pending_mutation(
            reminder_id,
            "reminder_task",
            "server:1",
            {
                "action": "transfer_to_server",
                "task_payload": {"title": "Flaky sync"},
                "transfer_errors": ["schedule_invalid: bad cron"],
            },
        )
        await pilot.app.screen.load_tasks()
        await pilot.pause()
        await _select_row(pilot)

        retry_button = pilot.app.screen.query_one(
            "#scheduling-retry-transfer", Button
        )
        assert retry_button.display
        assert not retry_button.disabled

        why = pilot.app.screen.query_one("#scheduling-transfer-why", Static)
        assert "schedule_invalid: bad cron" in why.visual.plain


@pytest.mark.asyncio
async def test_no_server_connection_disables_move_to_server_with_reason(
    transfer_db,
):
    """spec §6.4: no server connection is the first refusal reason."""
    db = transfer_db
    db.create_reminder_task(
        owner_id="local",
        title="Solo task",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    app = TransferWorkbenchTestApp(db, connected=False)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_row(pilot)

        button = pilot.app.screen.query_one(
            "#scheduling-transfer-to-server", Button
        )
        assert button.disabled
        assert "No server connection" in str(button.tooltip)
        why = pilot.app.screen.query_one("#scheduling-transfer-why", Static)
        assert "No server connection" in why.visual.plain


@pytest.mark.asyncio
async def test_queue_row_shows_transfer_badge_suffix(transfer_db):
    """A minimal, always-on signal that the state machine is doing
    something -- spec §9's badge language, pulled forward just far enough
    (plan ruling 1 keeps full badge/owner-column polish PR-6 scope)."""
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
        assert "Moving to server" in str(row[0])


# ---------------------------------------------------------------------------
# Automations-tab transfer actions (Task 7 fix round item 1)
#
# The tab has no per-row detail widget -- Move-to-local/Retry/Cancel are
# keybindings routed by active tab (`m`/`y`/`k`), same idiom as the tab's
# existing Run-now/Edit (`r`/`e`). A REAL `SchedulingService` + tmp_path DB
# throughout, same rationale as the reminder-side tests above.
# ---------------------------------------------------------------------------


def _stub_ready_health(monkeypatch) -> None:
    """A `recurring_question` `to_local` refusal also gates on local
    health (spec §6.4/§7.4) -- irrelevant to what THIS file tests (Task
    6 owns health-quoting correctness), so stubbed ready, mirroring Task
    6's own `_stub_health` test helper exactly."""
    monkeypatch.setattr(
        scheduling_service_module,
        "compute_local_health",
        lambda app, row: ("ready", ""),
    )


def _local_definition(db, **overrides):
    """A local `automation_definitions` row, mirroring Task 6's own
    `_make_definition` test helper exactly (same required fields)."""
    kwargs = dict(
        owner_id="local",
        family="recurring_question",
        name="Daily digest",
        schedule={"kind": "interval", "every_seconds": 3600},
        input={"question": "What happened today?"},
        config={},
    )
    kwargs.update(overrides)
    return db.create_automation_definition(**kwargs)


class _FakeAutomationsServerClient(_FakeServerClient):
    """Adds a minimal `list_automation_definitions` server-fetch stub on
    top of the reminder-transfer fake -- the Automations tab's server
    half is a live fetch (`_load_server_automations`), never read off a
    local mirror row directly, so a server-owned row reaches the UI
    exactly this way in real usage too."""

    def __init__(self, items: list[dict]) -> None:
        super().__init__()
        self._items = items

    async def list_automation_definitions(self, limit: int, offset: int):
        page = self._items[offset : offset + limit]
        return {"items": page, "total": len(self._items), "has_more": False}

    async def list_automation_definition_audit(self, definition_id: str):
        # Selecting a row fires the run-history fetch; empty is a valid,
        # honest response and keeps this fake minimal.
        return {"items": [], "total": 0}


async def _select_automations_tab_row(pilot, index: int = 0) -> None:
    workbench = pilot.app.screen
    tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
    tabs.active = "scheduling-automations-tab"
    table = workbench.query_one("#scheduling-automations-table", DataTable)
    table.cursor_coordinate = (index, 0)
    await pilot.pause()


@pytest.mark.asyncio
async def test_begin_to_local_from_a_server_mirror_row(transfer_db, monkeypatch):
    """Selecting a live server-fetched definition and pressing `m` mirrors
    it locally (same seam `_edit_selected_automation` already uses), then
    starts a real `begin_transfer_to_local`: a dormant local copy is
    created and the mirror itself stays untouched (spec §6.2)."""
    _stub_ready_health(monkeypatch)
    db = transfer_db
    server_client = _FakeAutomationsServerClient(
        [{"id": "srv-9", "name": "Nightly digest", "family": "recurring_question"}]
    )
    app = TransferWorkbenchTestApp(db, server_client=server_client)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_automations_tab_row(pilot)
        assert pilot.app.screen._selected_automation_id == "srv-9"

        pilot.app.screen.action_move_automation_to_local()
        await pilot.pause()

        assert isinstance(pilot.app.screen, ConfirmationDialog)
        assert "Nightly digest" in pilot.app.screen.message
        pilot.app.screen.dismiss(True)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        rows = db.list_automation_definitions(owner_id="server:1")
        assert len(rows) == 1
        mirror = rows[0]
        assert mirror["server_id"] == "srv-9"
        assert mirror["transfer_state"] is None  # untouched by the release

        local_rows = db.list_automation_definitions(owner_id="local")
        assert len(local_rows) == 1
        copy = local_rows[0]
        assert copy["server_id"] is None
        assert copy["transfer_state"] == "from_server_pending"


@pytest.mark.asyncio
async def test_automation_move_to_local_refusal_renders_inline(transfer_db):
    """A row that isn't server-owned refuses `to_local` -- the reason
    renders inline in the tab's notice Static (fix round item 1's
    "refusal -> inline reason" flow; the tab has no per-row Static)."""
    db = transfer_db
    _local_definition(db, name="Local only")
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_automations_tab_row(pilot)

        pilot.app.screen.action_move_automation_to_local()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        notice = pilot.app.screen.query_one("#scheduling-automations-notice", Static)
        assert "not server-owned" in notice.visual.plain
        # No dialog opened -- refused before ever confirming.
        assert not isinstance(pilot.app.screen, ConfirmationDialog)


@pytest.mark.asyncio
async def test_cancel_automation_transfer_routes_to_dormant_copy(
    transfer_db, monkeypatch
):
    """Cancel on the selected (dormant-copy) row deletes exactly that
    copy and leaves the mirror's transfer_state untouched -- same
    critical case as the reminder side's cancel-row-id test, driven
    through the Automations tab's own keybinding this time."""
    _stub_ready_health(monkeypatch)
    db = transfer_db
    mirror_id = db.create_automation_definition(
        owner_id="server:1",
        server_id="srv-7",
        family="recurring_question",
        name="Weekly roundup",
        schedule={"kind": "interval", "every_seconds": 604800},
        input={"question": "What happened this week?"},
        config={},
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        service = app.scheduling_service
        outcome = await service.begin_transfer_to_local(
            "automation_definition", mirror_id
        )
        assert outcome.status == "pending"
        copy_id = outcome.row_id
        assert copy_id is not None and copy_id != mirror_id

        await pilot.app.screen.load_automations()
        await pilot.pause()
        await _select_automations_tab_row(pilot)
        assert pilot.app.screen._selected_automation_id == copy_id

        pilot.app.screen.action_cancel_automation_transfer()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.get_automation_definition(copy_id) is None
        mirror_row = db.get_automation_definition(mirror_id)
        assert mirror_row is not None
        assert mirror_row["transfer_state"] is None


@pytest.mark.asyncio
async def test_retry_automation_transfer_only_on_failed(transfer_db):
    """Retry re-arms a definitively-failed local -> server transfer
    (spec §6.1.5) -- same retry leg Task 6 built for reminders, exercised
    here through the Automations tab's `y` keybinding."""
    db = transfer_db
    definition_id = _local_definition(db, name="Flaky automation")
    db.set_transfer_state(
        "automation_definition", definition_id, "to_server_pending", expected=(None,)
    )
    db.set_transfer_state(
        "automation_definition",
        definition_id,
        "to_server_sent",
        expected=("to_server_pending",),
    )
    db.set_transfer_state(
        "automation_definition",
        definition_id,
        "to_server_failed",
        expected=("to_server_sent",),
    )
    app = TransferWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await _select_automations_tab_row(pilot)
        assert pilot.app.screen._selected_automation_id == definition_id

        pilot.app.screen.action_retry_automation_transfer()
        await pilot.pause()
        assert isinstance(pilot.app.screen, ConfirmationDialog)
        pilot.app.screen.dismiss(True)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_automation_definition(definition_id)
        assert row["transfer_state"] == "to_server_pending"


@pytest.mark.asyncio
async def test_automation_transfer_actions_no_op_outside_automations_tab(
    transfer_db,
):
    """Pressing m/y/k while the Queue tab is active refuses with a
    switch-tabs notice rather than acting on a stale selection."""
    db = transfer_db
    _local_definition(db)
    app = TransferWorkbenchTestApp(db)
    notifications: list[str] = []
    async with app.run_test() as pilot:
        pilot.app.notify = lambda message, **kw: notifications.append(message)
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        # Queue tab is active by default -- never switched.

        pilot.app.screen.action_move_automation_to_local()
        pilot.app.screen.action_retry_automation_transfer()
        pilot.app.screen.action_cancel_automation_transfer()
        await pilot.pause()

        assert len(notifications) == 3
        assert all("Automations tab" in message for message in notifications)
