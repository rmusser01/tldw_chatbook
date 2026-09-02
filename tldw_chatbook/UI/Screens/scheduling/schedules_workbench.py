"""Schedules workbench shell for run timing, triggers, and recovery."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Button, DataTable, Input, Static, TabbedContent, TabPane

from ...Navigation.base_app_screen import BaseAppScreen
from ...Navigation.screen_state_store import RuntimeIdentity
from ...Workbench.workbench_state import (
    RecoveryState,
    WorkbenchHeaderState,
    WorkbenchStatus,
)
from ...Workbench.workbench_widgets import DestinationHeader, RecoveryCallout
from ....runtime_policy.bootstrap import set_authoritative_runtime_source
from ....Scheduling.automation_health import compute_local_health
from ....Scheduling.events import (
    CancelTransferRequested,
    DeleteTaskRequested,
    DisableTaskRequested,
    AcknowledgeIncidentRequested,
    EditTaskRequested,
    EnableTaskRequested,
    RetryTransferRequested,
    RunReminderNowRequested,
    SyncCompleted,
    SyncFailed,
    TransferToLocalRequested,
    TransferToServerRequested,
)
from ....Scheduling.models import ReminderTask, ScheduledTask
from ....Scheduling.services.server_client import (
    ServerClientError,
    ServerClientValidationError,
)
from ....UI.Screens.scheduling.conflicts_tab import ConflictsTab
from ....UI.Screens.scheduling.results_tab import ResultsTab, solved_eligibility
from ....UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
from ....Widgets.confirmation_dialog import ConfirmationDialog
from .forms.automation_definition_form import AutomationDefinitionForm
from .forms.new_task_choice_modal import NewTaskChoiceModal
from .forms.reminder_form import ReminderForm
from .task_detail import (
    SCHEDULES_EMPTY_CONSOLE_RECOVERY,
    TaskDetail,
    TaskInspector,
    _format_next_run,
    _managed_elsewhere_notice,
    _task_status,
    _task_type_label,
    _transfer_row_suffix,
    _underlying_status,
    _was_missed_while_away,
    status_badge_text,
    transfer_row_dict,
)

if TYPE_CHECKING:
    from tldw_chatbook.Scheduling.services.scheduling_service import (
        SchedulingService,
        TransferOutcome,
    )
    from tldw_chatbook.app import TldwCli


logger = logger.bind(module="SchedulesWorkbench")

SCHEDULES_COMPACT_WORKBENCH_MAX_WIDTH = 120

#: Debounce for the queue filter `Input` -- mirrors the console picker
#: family's 0.2 s shape (`console_prompt_picker_modal.py`). A full render
#: pass clears and rebuilds the whole `DataTable` (task-15476).
QUEUE_FILTER_DEBOUNCE_SECONDS = 0.2

#: Cadence for re-rendering the relative next-run column ("in 25m" goes
#: stale otherwise -- task-23111 review F9). Paused while the screen is
#: not current, per the hidden-progress-clock rule (TASK-23022).
NEXT_RUN_REFRESH_SECONDS = 60.0

#: Defensive cap for the Automations tab's follow-the-pages load -- the loop
#: exists so the tail of a large definition list is never silently hidden,
#: not to render unbounded rows.
AUTOMATIONS_LOAD_MAX_ROWS = 500


def automation_execution_target_label(definition: dict[str, Any]) -> str:
    """Render one definition's per-task execution target (ADR-077 AC#7).

    ``input.provider``/``input.model`` ride the definition payload and the
    server executor honors them. The column shows what was PINNED here:
    when neither key is set the label is ``auto`` -- the definition pins
    nothing, and the server resolves the run target from its own
    automation-config executor defaults (``[Scheduled_Tasks_Automation]
    executor_provider``/``executor_model``) falling back to the server
    default. Those layers live in server config, not the payload, so
    ``auto`` is the honest client-side rendering, not a claim about which
    server layer actually won.

    Args:
        definition: One row from the server's definition list, as the raw
            dict the scheduling server client returns.

    Returns:
        A short cell label: ``provider/model``, either part alone, or
        ``auto`` when neither is set.
    """
    source = definition.get("input") if isinstance(definition.get("input"), dict) else {}
    provider = str(source.get("provider") or "").strip()
    model = str(source.get("model") or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    if provider:
        return provider
    if model:
        return model
    return "auto"


def automation_name_cell(definition: dict[str, Any]) -> str:
    """Name cell for the merged local+server Automations list (task-5 fix round).

    The tab now mixes local-owned rows into what used to be a server-only
    list, so every row needs a visible owner distinction or a local save
    reads as indistinguishable from a server one. A prefix on the existing
    Name cell is the smallest honest rendering -- no new column, no CSS
    changes -- since the table's own `key=` already disambiguates rows for
    everything that acts on them; this prefix is purely for the human
    reading the table.

    Args:
        definition: One merged row (local DB dict or server API dict --
            both carry `owner_id` and `name`, confirmed against the real
            server fixture `automation_definition_list.json`).

    Returns:
        `"[This device] <name>"` for a local row, `"[<server id>] <name>"`
        for a server-scoped one, and `"[<server id> · pending sync]
        <name>"` for one authored offline that has not reached the server
        yet (`pending_sync`, stamped by `_load_local_automations`) -- that
        row is only on this device, and saying "server" flat would claim a
        definition the server has never heard of (final review I5).
    """
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    owner_id = str(definition.get("owner_id") or "local")
    name = str(definition.get("name") or definition.get("id") or "")
    if is_server_scoped_owner(owner_id):
        label = owner_id.split(":", 1)[1] if ":" in owner_id else owner_id
        if definition.get("pending_sync"):
            label = f"{label} · pending sync"
    else:
        label = "This device"
    return f"[{label}] {name}"


def _cancel_toast_text(name: str) -> str:
    """Honest cancel confirmation (final review L13).

    The old copy ("Transfer cancelled for 'X'.") and the docs beside it
    promised "no server-side effect", which spec §6.3's own table does
    not support: `from_server_pending` also covers a release whose
    server-side delete already landed but whose ack was lost, and that
    delete cannot be undone from here. What IS true in every cancellable
    state is that the queued mutation is gone, so nothing further goes
    out -- which is what this now says.
    """
    return (
        f"Transfer cancelled for '{name}' — nothing further will be sent "
        "to the server."
    )


#: Delayed second fetch of the run-history pane after a Run-now dispatch:
#: the terminal audit event lands only after the server finishes executing
#: the run, so an immediate fetch alone would usually miss it.
AUTOMATION_HISTORY_FOLLOWUP_SECONDS = 5.0


class SchedulesWorkbench(BaseAppScreen):
    """Main workbench for managing scheduled runs, reminders, and jobs."""

    BINDINGS = [
        Binding("c", "create_reminder", "Create"),
        Binding("e", "edit_task", "Edit"),
        Binding("r", "run_task_now", "Run now"),
        Binding("space", "toggle_enabled", "Enable/Disable"),
        Binding("d", "delete", "Delete"),
        Binding("x", "mark_task", "Mark"),
        Binding("escape", "clear_marks", "Clear marks"),
        Binding("s", "sync_now", "Sync"),
        # Automations-tab-only (schedules-handoff PR-5 task 7 fix round):
        # the tab has no per-row detail widget, so its actions are
        # keybindings routed by active tab -- same idiom as r/e above,
        # not new buttons (mirrors _edit_selected_automation/
        # _run_automation_now, the tab's existing action grammar).
        Binding("m", "move_automation_to_local", "Move to local"),
        Binding("M", "move_automation_to_server", "Move to server"),
        Binding("y", "retry_automation_transfer", "Retry transfer"),
        Binding("k", "cancel_automation_transfer", "Cancel transfer"),
        # Results-tab-only (schedules-handoff PR-6 task 3): read/dismiss
        # reuse r/d via the SAME active-tab routing action_run_task_now/
        # action_delete already do for Automations -- r=Read and d=Dismiss
        # are natural readings of those same keys on this tab. Mark
        # solved/Mark all read have no existing-key mnemonic to reuse, so
        # they get fresh letters (o/a), guarded the same way m/M/y/k are.
        Binding("o", "mark_result_solved", "Mark solved"),
        Binding("a", "mark_all_results_read", "Mark all read"),
    ]

    # Footer hints must stay 1:1 with BINDINGS and only advertise implemented
    # actions (ADR-031). Single letters are safe: focused inputs consume
    # printable keys before screen bindings fire.
    SCHEDULES_SHORTCUTS = (
        ("c", "create"),
        ("e", "edit"),
        ("r", "run now"),
        ("space", "toggle"),
        ("d", "delete"),
        ("x", "mark"),
        ("s", "sync"),
        ("m", "move to local"),
        ("M", "move to server"),
        ("y", "retry transfer"),
        ("k", "cancel transfer"),
        ("o", "mark solved"),
        ("a", "mark all read"),
    )

    def __init__(
        self, app_instance: "TldwCli", screen_name: str = "schedules", **kwargs
    ):
        super().__init__(app_instance, screen_name, **kwargs)
        self._scheduling_service = getattr(app_instance, "scheduling_service", None)
        self._tasks: list[ReminderTask | ScheduledTask] = []
        self._visible_tasks: list[ReminderTask | ScheduledTask] = []
        self._filter_text = ""
        self._filter_debounce_timer: Timer | None = None
        self._next_run_refresh_timer: Timer | None = None
        # task-15476: the task id currently shown in the detail/inspector
        # panes, tracked independently of row index so a filter keystroke
        # can restore the same selection instead of always jumping to row 0.
        self._selected_task_id: str | None = None
        self._marked_ids: set[str] = set()
        #: The current hidden-panes notice from on_resize; combined with
        #: the marks/glyph legend in _update_pane_notice (task-23107).
        self._resize_notice = ""
        self._sync_running = False
        # ADR-077: server-owned automation definitions shown in the
        # Automations tab. Kept as the raw dicts the server client returns
        # (model_dump(mode="json")) -- the server owns the enum vocabularies
        # and the tab must not break when a new lifecycle/health value ships.
        self._automations: list[dict[str, Any]] = []
        self._selected_automation_id: str | None = None
        self._current_console_follow_item = None
        self._latest_console_follow_item_id: str | None = None
        self._latest_console_launch_kwargs: dict[str, Any] | None = None
        self._latest_console_context_loaded = False

    def _active_server_id(self) -> str | None:
        runtime_policy = getattr(self.app_instance, "runtime_policy", None)
        runtime_state = runtime_policy.state if runtime_policy is not None else None
        return getattr(runtime_state, "active_server_id", None)

    @staticmethod
    def _server_available(service: Any, active_server_id: str | None) -> bool:
        """Return whether Schedules can switch ownership to a live server."""
        return (
            service is not None
            and bool(active_server_id)
            and service.server_client.notifications_service is not None
        )

    @staticmethod
    def _update_static_content(target: Static, content: str) -> None:
        """Preserve layout-aware updates while skipping identical timer copy."""
        if target.content != content:
            target.update(content)

    def compose_content(self) -> ComposeResult:
        """Build the three-pane scheduling workbench layout."""
        service = self._service()
        owner_id = service.owner_id if service else "local"
        active_server_id = self._active_server_id()
        server_available = self._server_available(service, active_server_id)
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Schedules",
                subtitle="When jobs, watchlists, and workflows run.",
                status="loading",
                status_label="Checking sync status…",
            ),
            id="schedules-destination-header",
        )
        if service is None:
            # Visible recovery instead of a silently empty workbench (UX-043).
            yield RecoveryCallout(
                RecoveryState(
                    title="Scheduling service unavailable",
                    body=(
                        "The scheduling service did not start, so the queue and "
                        "sync are offline. Check the scheduling configuration, "
                        "then restart the app."
                    ),
                    action=None,
                    visible=True,
                ),
                id="scheduling-recovery",
            )
        with Vertical(id="schedules-shell"):
            yield SyncStatusWidget(
                id="scheduling-sync-status",
                current_owner=owner_id,
                active_server_id=active_server_id,
                server_available=server_available,
            )
            # TASK-26025: scheduler liveness -- a stale heartbeat reads
            # distinctly from an empty queue and a never-started loop.
            yield Static("", id="scheduling-liveness")
            with TabbedContent(id="scheduling-tabs"):
                with TabPane("Queue", id="scheduling-queue-tab"):
                    with Horizontal(id="scheduling-workbench"):
                        with Vertical(id="scheduling-list-pane"):
                            with Horizontal(id="scheduling-list-header"):
                                yield Static(
                                    "Schedule Queue",
                                    id="scheduling-list-title",
                                    classes="scheduling-column-title",
                                )
                                yield Button(
                                    "+ New",
                                    id="scheduling-new-task",
                                    variant="primary",
                                    tooltip="Schedule a new task (c).",
                                )
                            yield Input(
                                placeholder="Filter: title, type, or status…",
                                id="scheduling-queue-filter",
                            )
                            yield DataTable(
                                id="scheduling-task-table", cursor_type="row"
                            )
                            yield Static("", id="scheduling-pane-notice")
                        with Vertical(id="scheduling-detail-pane"):
                            yield TaskDetail(id="scheduling-task-detail")
                        with Vertical(id="scheduling-inspector-pane"):
                            yield TaskInspector(id="scheduling-task-inspector")
                with TabPane("Automations", id="scheduling-automations-tab"):
                    with Horizontal(id="scheduling-automations-split"):
                        with Vertical(id="scheduling-automations-pane"):
                            with Horizontal(id="scheduling-automations-header"):
                                yield Static(
                                    "Server Automations",
                                    id="scheduling-automations-title",
                                    classes="scheduling-column-title",
                                )
                                yield Button(
                                    "+ New",
                                    id="scheduling-new-automation",
                                    variant="primary",
                                    tooltip="Schedule a new recurring question.",
                                )
                            yield DataTable(
                                id="scheduling-automations-table", cursor_type="row"
                            )
                            yield Static("", id="scheduling-automations-notice")
                        with Vertical(id="scheduling-automation-history-pane"):
                            yield Static(
                                "Run history",
                                id="scheduling-automation-history-title",
                                classes="scheduling-column-title",
                            )
                            yield DataTable(
                                id="scheduling-automation-history-table",
                                cursor_type="row",
                            )
                            yield Static(
                                "",
                                id="scheduling-automation-history-notice",
                            )
                with TabPane("Conflicts", id="scheduling-conflicts-tab"):
                    yield ConflictsTab(
                        id="scheduling-conflicts",
                        sync_engine=service.sync_engine if service else None,
                    )
                with TabPane("Results", id="scheduling-results-tab"):
                    yield ResultsTab(id="scheduling-results")

    def _service(self) -> "SchedulingService | None":
        """Return the app's scheduling service, if available."""
        return self._scheduling_service

    def _register_footer_shortcuts(self) -> None:
        """Register Scheduling shortcuts via BaseAppScreen's persisting API."""
        self.register_footer_shortcuts(
            source="schedules", shortcuts=self.SCHEDULES_SHORTCUTS
        )

    def on_mount(self) -> None:
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.
        self._sync_responsive_workbench()
        self._register_footer_shortcuts()
        self._refresh_owner_select()
        self._refresh_conflicts_tab()
        self._refresh_results_tab()
        table = self.query_one("#scheduling-task-table", DataTable)
        table.add_columns("Title", "Type", "Status", "Next Run")
        automations_table = self.query_one("#scheduling-automations-table", DataTable)
        automations_table.add_columns(
            "Name", "Family", "Lifecycle", "Health", "Model"
        )
        history_table = self.query_one(
            "#scheduling-automation-history-table", DataTable
        )
        history_table.add_columns("Time", "Event", "Summary")
        # task-23111 review F9: the relative next-run column ("in 25m")
        # is render-time text; refresh it periodically while visible.
        self._next_run_refresh_timer = self.set_interval(
            NEXT_RUN_REFRESH_SECONDS, self._refresh_next_run_rendering
        )
        self._request_tasks_refresh()
        self._request_automations_refresh()
        self._refresh_scheduler_liveness()

    def _refresh_scheduler_liveness(self) -> None:
        """Update the scheduler-liveness line from the durable heartbeat
        (TASK-26025). Never raises -- a diagnostics read must not break the
        screen."""
        try:
            from datetime import datetime, timezone

            from ....config import get_cli_setting
            from ....Scheduling.constants import SCHEDULER_POLL_INTERVAL_SECONDS
            from ....Scheduling.scheduler_heartbeat import (
                default_heartbeat_path,
                read_heartbeat,
                scheduler_liveness_line,
            )

            poll = get_cli_setting(
                "scheduling",
                "scheduler_poll_interval_seconds",
                SCHEDULER_POLL_INTERVAL_SECONDS,
            )
            line = scheduler_liveness_line(
                read_heartbeat(default_heartbeat_path()),
                now=datetime.now(timezone.utc),
                poll_interval=float(poll or SCHEDULER_POLL_INTERVAL_SECONDS),
            )
            self._update_static_content(
                self.query_one("#scheduling-liveness", Static), line
            )
        except Exception:  # noqa: BLE001 -- liveness display never breaks the screen
            pass

    def _refresh_next_run_rendering(self) -> None:
        """Re-render the queue so relative next-run text stays honest.

        Also refreshes the scheduler-liveness line (TASK-26025) so a loop
        that dies while the screen is open turns visibly stale.

        Skips unless this screen is the top of the stack. (Textual's
        ``is_current`` also counts screens behind the top one --
        ``_background_screens`` always includes the screen directly
        beneath the top regardless of opacity -- so it cannot express
        "covered"; the suspend/resume handlers pause the timer while
        covered and refresh on uncover.) Also skips an empty queue:
        nothing to refresh, and the no-service path must keep its own
        detail-pane copy.
        """
        if self.app.screen is not self:
            return
        # TASK-26025: refresh liveness even on an empty queue -- a stall
        # with nothing queued is exactly the case AC#2 must distinguish.
        self._refresh_scheduler_liveness()
        if not self._visible_tasks:
            return
        self._render_table()

    def on_screen_suspend(self) -> None:
        """Stop the relative-time refresh while another screen covers this.

        Hidden clocks must not tick unseen (TASK-23022); the resume
        handler refreshes immediately so no stale text is ever shown.
        """
        if self._next_run_refresh_timer is not None:
            self._next_run_refresh_timer.pause()

    def on_screen_resume(self) -> None:
        """Refresh relative times and restart the cadence when uncovered.

        No ``super().on_screen_resume()``: Textual's dispatcher invokes
        every handler along the MRO for one event (see BaseAppScreen's
        MRO contract), so the base handler runs regardless.
        """
        if self._next_run_refresh_timer is not None:
            self._next_run_refresh_timer.resume()
        self._refresh_next_run_rendering()

    def _sync_responsive_workbench(self) -> None:
        """Keep the primary queue and detail action visible at narrow widths."""
        self.set_class(
            self.size.width <= SCHEDULES_COMPACT_WORKBENCH_MAX_WIDTH,
            "schedules-workbench-compact",
        )

    def _request_tasks_refresh(self) -> None:
        """Schedule the task loader through its exclusive worker group."""
        self.run_worker(
            self.load_tasks,
            exclusive=True,
            group="schedules-load-tasks",
        )  # type: ignore[arg-type]

    async def load_tasks(self) -> None:
        """Fetch reminders from the scheduling service and populate the table."""
        service = self._scheduling_service
        if service is None:
            logger.debug("No scheduling_service available; cannot load tasks")
            await self._refresh_console_context()
            return

        try:
            tasks = await service.list_tasks()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load tasks")
            self.app_instance.notify(
                "Could not load tasks. Check the scheduling service and retry.",
                severity="error",
            )
            self._tasks = []
            table = self.query_one("#scheduling-task-table", DataTable)
            table.clear()
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=True
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            await self._refresh_console_context()
            return

        self._tasks = list(tasks)
        # Marks must always refer to rows that still exist (task-23107
        # review F1): a task deleted or filtered out of existence must not
        # linger as an invisible mark a bulk verb would act on.
        self._marked_ids.intersection_update({task.id for task in self._tasks})
        self._render_table()
        await self._refresh_console_context()

    def _render_table(self, now: datetime | None = None) -> None:
        """Rebuild the queue rows from the current tasks + filter text.

        Restores the previously selected task's row (by id) when it is
        still visible after the filter narrows, instead of always jumping
        the detail/inspector panes back to row 0 (task-15476): a filter
        keystroke must not discard what the user was looking at.

        ``now`` is one shared reference for every row's relative
        next-run rendering (review F9: per-row ``datetime.now()`` let a
        single frame straddle a bucket boundary); injectable for
        deterministic tests.
        """
        render_now = now if now is not None else datetime.now(timezone.utc)
        previous_selected_id = self._selected_task_id
        text = self._filter_text.strip().lower()
        self._visible_tasks = [
            task
            for task in self._tasks
            if not text
            or text in task.title.lower()
            or text in _task_type_label(task).lower()
            or text in _task_status(task).value.lower().replace("_", " ")
            or text in _task_status(task).value.lower()
            # Underlying status too (review F5): a disabled task whose
            # last dispatch failed must still answer a "missed" filter.
            or text in _underlying_status(task).value.lower().replace("_", " ")
            or text in _underlying_status(task).value.lower()
            # task-18937: filtering for "missed" finds late-dispatch rows too,
            # not just handler-failure ones -- both are honest matches for a
            # user asking "what went wrong while I wasn't looking".
            or (_was_missed_while_away(task) and "missed" in text)
        ]
        rows: list[tuple[str, str, Text, str]] = [
            (
                ("● " if task.id in self._marked_ids else "")
                + ("◇ " if _was_missed_while_away(task) else "")
                + task.title
                + _transfer_row_suffix(task),
                _task_type_label(task),
                status_badge_text(_task_status(task)),
                # Compact: same relative form as the detail pane, without
                # the timezone token (task-23111); one shared `now` for
                # every row (review F9).
                _format_next_run(task, now=render_now, compact=True),
            )
            for task in self._visible_tasks
        ]

        table = self.query_one("#scheduling-task-table", DataTable)
        table.clear()
        for row in rows:
            table.add_row(*row)
        self._update_pane_notice()

        if rows:
            target_index = 0
            if previous_selected_id is not None:
                for index, task in enumerate(self._visible_tasks):
                    if task.id == previous_selected_id:
                        target_index = index
                        break
            if table.row_count:
                table.move_cursor(row=target_index)
            self._update_detail_for_index(target_index)
        else:
            self._selected_task_id = None
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=not self._tasks
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            if self._tasks and self._filter_text.strip():
                # Everything filtered out: say so instead of "select a task".
                self._update_static_content(
                    self.query_one("#scheduling-task-detail-empty-state", Static),
                    f"No tasks match '{self._filter_text.strip()}'. "
                    "Clear the filter to see the queue.",
                )

    @on(Input.Changed, "#scheduling-queue-filter")
    def _on_queue_filter_changed(self, event: Input.Changed) -> None:
        """Filter the queue rows by title substring (debounced).

        A settled render clears and rebuilds the whole `DataTable`, so it
        must not run on every keystroke (task-15476).
        """
        self._filter_text = event.value
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
        self._filter_debounce_timer = self.set_timer(
            QUEUE_FILTER_DEBOUNCE_SECONDS, self._apply_queue_filter_debounced
        )

    def _apply_queue_filter_debounced(self) -> None:
        self._filter_debounce_timer = None
        self._render_table()

    @on(DataTable.RowHighlighted)
    def _on_task_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update the detail pane when the user highlights a task row."""
        self._update_detail_for_index(event.cursor_row)

    def _incidents_for(self, task_id) -> list:
        """TASK-26027: the selected task's failure incidents (fail-safe).

        Reminders key their incidents by the raw task id; briefings (whose
        failures the briefing handler records) key by "briefing:<id>", so
        both spellings are queried.
        """
        db = getattr(self._scheduling_service, "db", None)
        reader = getattr(db, "list_task_incidents", None)
        if not callable(reader):
            return []
        rows: list = []
        for key in (str(task_id), f"briefing:{task_id}"):
            try:
                rows.extend(reader(key, limit=10))
            except Exception:  # noqa: BLE001 -- never breaks the pane
                pass
        return rows

    def _run_history_for(self, task_id) -> list:
        """TASK-26026: the selected task's durable run history, newest first.

        Fail-safe: a missing service/method or a read error yields an empty
        history rather than breaking the detail pane.
        """
        service = self._scheduling_service
        db = getattr(service, "db", None)
        reader = getattr(db, "list_task_runs", None)
        if not callable(reader):
            return []
        try:
            return list(reader(str(task_id), limit=8))
        except Exception:  # noqa: BLE001 -- history read never breaks the pane
            return []

    def _update_detail_for_index(self, index: int) -> None:
        """Render task details in the detail and inspector panes."""
        if not (0 <= index < len(self._visible_tasks)):
            self._selected_task_id = None
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=not self._tasks
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            return

        task = self._visible_tasks[index]
        self._selected_task_id = task.id
        task_detail = self.query_one("#scheduling-task-detail", TaskDetail)
        task_detail.set_task(
            task,
            run_history=self._run_history_for(task.id),
            incidents=self._incidents_for(task.id),
        )
        self._update_transfer_actions(task_detail, task)
        self.query_one("#scheduling-task-inspector", TaskInspector).set_task(task)

    def _update_transfer_actions(
        self, task_detail: TaskDetail, task: ReminderTask | ScheduledTask
    ) -> None:
        """Compute Move/Retry/Cancel disabled-reasons for the detail pane.

        Reasons come straight from `SchedulingService.transfer_refusal`/
        `cancel_refusal` (spec §6.4/§6.3) so the UI never re-derives the
        refusal rules itself (health quoting included, and -- fix round
        finding 1 -- cancel's own state branching too) -- each reason is
        only computed for the button `TaskDetail.set_task` already
        decided is structurally relevant (e.g. `to_local_reason` is never
        computed for a local row, which would always trivially refuse
        with "not server-owned" noise).
        """
        service = self._scheduling_service
        if service is None or not isinstance(task, ReminderTask):
            task_detail.set_transfer_reasons(
                to_server_reason=None,
                to_local_reason=None,
                retry_reason=None,
                cancel_reason=None,
                retry_errors=[],
            )
            task_detail.set_lifecycle_lock(None)
            return

        row = transfer_row_dict(task)
        is_server_owned = str(task.owner_id or "").startswith("server:")
        transfer_state = task.transfer_state

        # `transfer_refusal` already returns None for `to_server_failed`
        # (Task 6's retry leg deliberately excludes it from "already in
        # progress"), so this needs no extra state carve-out -- the same
        # call also backs the Retry button below.
        to_server_reason = (
            service.transfer_refusal(row, "to_server") if not is_server_owned else None
        )
        to_local_reason = (
            service.transfer_refusal(row, "to_local") if is_server_owned else None
        )
        cancel_reason = service.cancel_refusal(row)

        # A `to_server_failed` row is never server-owned, so this is the
        # exact same call `to_server_reason` already made above -- reused
        # rather than repeated.
        retry_reason: str | None = None
        retry_errors: list[str] = []
        if transfer_state == "to_server_failed":
            retry_reason = to_server_reason
            # fix round finding 3: keyed off the mutation's OWN owner_id
            # column, not a guess via "today's active server" -- a
            # `to_server_failed` row's mutation was recorded under
            # whatever server was active at the time of the failed
            # attempt, which silently stops matching after a server
            # switch if guessed instead of read.
            mutation = service.db.get_pending_mutation_for_local_id(
                task.id, "reminder_task"
            )
            if mutation is not None:
                payload = mutation.get("payload") or {}
                errors = payload.get("transfer_errors")
                if errors:
                    retry_errors = list(errors)

        task_detail.set_transfer_reasons(
            to_server_reason=to_server_reason,
            to_local_reason=to_local_reason,
            retry_reason=retry_reason,
            cancel_reason=cancel_reason,
            retry_errors=retry_errors,
        )
        # spec §6.3 read-only-except-cancel (final review I7). Applied
        # AFTER set_transfer_reasons, which owns the same reason Static.
        task_detail.set_lifecycle_lock(service.transfer_lock_reason(row))

    def _refuse_if_transfer_locked(self, task: Any, verb: str) -> bool:
        """Notify and return True when ``task`` is read-only mid-transfer.

        Spec §6.3 (final review I7): the key-bound Edit / Delete /
        Enable-Disable verbs share the detail pane's lock, so pressing
        `e`/`d`/`space` on an in-flight row says why instead of silently
        no-oping against the facade's own guard. The reason string comes
        from `SchedulingService.transfer_lock_reason` -- never re-derived
        here.
        """
        service = self._scheduling_service
        if service is None or not isinstance(task, ReminderTask):
            return False
        reason = service.transfer_lock_reason(transfer_row_dict(task))
        if reason is None:
            return False
        self.app_instance.notify(f"Cannot {verb}: {reason}", severity="warning")
        return True

    async def _refresh_console_context(self) -> None:
        """Load the latest Schedules Console-follow context."""
        latest_console_item = await self._latest_console_follow_item_from_adapter()
        latest_console_launch = None
        if latest_console_item is None:
            latest_console_launch = await self._latest_reading_digest_console_launch()
        self._apply_console_context(latest_console_item, latest_console_launch)

    async def _latest_console_follow_item_from_adapter(self) -> Any | None:
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        build_dashboard_input = getattr(adapter, "build_dashboard_input", None)
        if not callable(build_dashboard_input):
            return None
        try:
            providers = getattr(self.app_instance, "providers_models", {}) or {}
            runtime_identity = RuntimeIdentity.from_state(
                self.app_instance.runtime_policy.state
            )
            has_recent_work = self.app_instance.screen_state_store.has_snapshots(
                runtime_identity
            )
            dashboard_input = build_dashboard_input(
                providers_models=providers,
                has_recent_work=has_recent_work,
            )
            if inspect.isawaitable(dashboard_input):
                dashboard_input = await dashboard_input
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load Schedules Console follow item from Home active-work adapter.",
            )
            return None
        for item in tuple(getattr(dashboard_input, "active_work_items", ()) or ()):
            if (
                getattr(item, "source", None) == "Schedules"
                and bool(getattr(item, "console_available", False))
                and getattr(item, "item_id", None)
            ):
                return item
        return None

    async def _latest_reading_digest_console_launch(self) -> dict[str, Any] | None:
        service = getattr(self.app_instance, "local_media_reading_service", None)
        list_outputs = getattr(service, "list_reading_digest_outputs", None)
        if not callable(list_outputs):
            return None
        try:
            output_listing = list_outputs(schedule_id=None, limit=1, offset=0)
            if inspect.isawaitable(output_listing):
                output_listing = await output_listing
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load Schedules Console launch context from local reading digest outputs.",
            )
            return None
        items = (
            output_listing.get("items") if isinstance(output_listing, Mapping) else None
        )
        latest_output = next(iter(tuple(items or ())), None)
        if not isinstance(latest_output, Mapping):
            return None

        output_id = latest_output.get("output_id") or latest_output.get("id")
        if output_id in (None, ""):
            return None

        metadata = latest_output.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        schedule_name = str(
            metadata.get("schedule_name")
            or latest_output.get("schedule_name")
            or latest_output.get("schedule_id")
            or ""
        ).strip()
        title = str(
            latest_output.get("title") or schedule_name or "Reading digest output"
        ).strip()
        item_count = metadata.get("item_count", latest_output.get("item_count"))
        payload = {
            "target_id": f"local:reading_digest_output:{output_id}",
            "output_id": output_id,
            "schedule_id": latest_output.get("schedule_id"),
            "schedule_name": schedule_name or None,
            "download_url": latest_output.get("download_url")
            or latest_output.get("storage_path"),
            "created_at": latest_output.get("created_at"),
            "item_count": item_count,
        }
        return {
            "source": "schedules",
            "title": title,
            "payload": payload,
            "status": "ready",
            "recovery": "Review this reading digest output from Schedules or return to Library.",
            "action_label": "Open schedule output",
        }

    def _apply_console_context(
        self,
        latest_console_item: Any | None,
        latest_console_launch: dict[str, Any] | None,
    ) -> None:
        self._current_console_follow_item = latest_console_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        self._latest_console_launch_kwargs = latest_console_launch
        self._latest_console_context_loaded = True
        self._update_follow_button_state()

    def _update_follow_button_state(self) -> None:
        task_detail = self.query_one("#scheduling-task-detail", TaskDetail)
        available = (
            self._latest_console_follow_item_id is not None
            or self._latest_console_launch_kwargs is not None
        )
        task_detail.set_follow_available(available)

    @on(DeleteTaskRequested)
    def _on_delete_task_requested(self, event: DeleteTaskRequested) -> None:
        """Delete the requested task and refresh the queue."""
        event.stop()
        self._marked_ids.discard(event.task.id)
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot delete task.",
                severity="warning",
            )
            return

        async def _delete_and_refresh() -> None:
            try:
                await service.delete_reminder(event.task.id)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to delete reminder {}", event.task.id)
                self.app_instance.notify(
                    f"Failed to delete '{event.task.title}'.",
                    severity="error",
                )
            else:
                self.app_instance.notify(
                    f"Deleted '{event.task.title}'.",
                    severity="information",
                )
            self._request_tasks_refresh()

        self.run_worker(
            _delete_and_refresh,
            exclusive=True,
            group="schedules-delete-task",
        )  # type: ignore[arg-type]

    @on(TransferToServerRequested)
    def _on_transfer_to_server_requested(
        self, event: TransferToServerRequested
    ) -> None:
        event.stop()
        self._begin_transfer(event.task, "to_server")

    @on(TransferToLocalRequested)
    def _on_transfer_to_local_requested(
        self, event: TransferToLocalRequested
    ) -> None:
        event.stop()
        self._begin_transfer(event.task, "to_local")

    @on(RetryTransferRequested)
    def _on_retry_transfer_requested(self, event: RetryTransferRequested) -> None:
        # Obligation (f)/spec §6.1.5: retrying a `to_server_failed` row is
        # the SAME facade call as a first-time begin -- `transfer_refusal`
        # deliberately narrows its "already in progress" check to exclude
        # `to_server_failed`, and `begin_transfer_to_server`'s CAS accepts
        # both `None` and `to_server_failed` as its starting state.
        event.stop()
        self._begin_transfer(event.task, "to_server")

    @on(CancelTransferRequested)
    def _on_cancel_transfer_requested(self, event: CancelTransferRequested) -> None:
        event.stop()
        self._cancel_transfer(event.task)

    @staticmethod
    def _transfer_confirm_dialog(
        name: str, direction: str, warnings: list[str]
    ) -> ConfirmationDialog:
        """Build the Move confirm dialog (spec §6.1/§6.2/§6.4) -- shared by
        the reminder (`_begin_transfer`) and Automations-tab
        (`_begin_automation_transfer`) flows (Task 7 fix round item 1: two
        near-identical call sites, one copy of the confirm-dialog copy)."""
        destination_label = "the server" if direction == "to_server" else "this device"
        lines = [f'Move "{escape_markup(name)}" to {destination_label}?']
        if warnings:
            lines.append("")
            lines.extend(f"- {escape_markup(warning)}" for warning in warnings)
        if direction == "to_server":
            lines.append("")
            lines.append(
                "It keeps running on this device until the server accepts "
                "the transfer -- nothing goes dark while this is only "
                "queued."
            )
        return ConfirmationDialog(
            title="Move to server" if direction == "to_server" else "Move to local",
            message="\n".join(lines),
            confirm_label="Move",
            cancel_label="Cancel",
        )

    @staticmethod
    def _transfer_pending_toast_text(name: str, direction: str) -> str:
        """Honest §6.1.1 "still runs here" / dormant-copy copy, shared by
        the reminder and Automations-tab transfer flows."""
        if direction == "to_server":
            return (
                f"'{name}' is queued to move to the server -- it still "
                "runs on this device until the server accepts it."
            )
        return (
            f"'{name}' is queued to move to this device -- a dormant copy "
            "is ready and will arm once the server releases it."
        )

    def _begin_transfer(self, task: ReminderTask, direction: str) -> None:
        """Confirm, then start a transfer for ``task`` (spec §6.1/§6.2).

        Always confirms first -- Move is not a casual action, and the
        dialog is the one place `transfer_warnings` (imminent one-time
        `run_at`, non-transferring `timeout_seconds`) actually reaches the
        user before they commit, not only when something happens to be
        wrong. `transfer_refusal` is checked again defensively (the
        button should already be disabled when refused) before ever
        opening the dialog.
        """
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot start a transfer.",
                severity="warning",
            )
            return

        row = transfer_row_dict(task)
        reason = service.transfer_refusal(row, direction)
        if reason is not None:
            self.app_instance.notify(reason, severity="warning")
            return
        warnings = service.transfer_warnings(row, direction)
        dialog = self._transfer_confirm_dialog(task.title, direction, warnings)

        async def _confirm_and_begin() -> None:
            confirmed = await self.app.push_screen_wait(dialog)
            if not confirmed:
                return
            try:
                if direction == "to_server":
                    outcome = await service.begin_transfer_to_server(
                        "reminder_task", task.id
                    )
                else:
                    outcome = await service.begin_transfer_to_local(
                        "reminder_task", task.id
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to begin transfer for {}", task.id)
                self.app_instance.notify(
                    f"Failed to start the transfer for '{task.title}'.",
                    severity="error",
                )
                self._request_tasks_refresh()
                return
            self._notify_transfer_outcome(task, direction, outcome)
            self._request_tasks_refresh()

        self.run_worker(
            _confirm_and_begin,
            exclusive=True,
            group="schedules-transfer",
        )  # type: ignore[arg-type]

    def _notify_transfer_outcome(
        self, task: ReminderTask, direction: str, outcome: "TransferOutcome"
    ) -> None:
        """Toast a transfer's result -- honest about what actually happened
        (spec §6.1.1: a queued-not-sent transfer never claims the task
        stopped running here)."""
        if outcome.status == "pending":
            self.app_instance.notify(
                self._transfer_pending_toast_text(task.title, direction),
                severity="information",
            )
        elif outcome.status == "refused":
            self.app_instance.notify(
                outcome.reason or f"Could not move '{task.title}'.",
                severity="warning",
            )
        elif outcome.status == "not_found":
            self.app_instance.notify(
                f"'{task.title}' no longer exists.", severity="warning"
            )

    def _cancel_transfer(self, task: ReminderTask) -> None:
        """Cancel ``task``'s in-progress transfer immediately.

        No confirm dialog: cancel is the escape hatch spec §6.3 exists
        for, and gating the escape hatch behind its own confirmation
        fights the point of it. `task.id` is whichever row is currently
        selected -- for a release's dormant local copy that is already
        the copy's OWN id (it is a first-class queue row, not reached
        through the mirror), which is exactly the id `cancel_transfer`
        needs for that leg (Task 6 handoff note).
        """
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot cancel the "
                "transfer.",
                severity="warning",
            )
            return

        async def _cancel_and_refresh() -> None:
            try:
                outcome = await service.cancel_transfer("reminder_task", task.id)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cancel transfer for {}", task.id)
                self.app_instance.notify(
                    f"Failed to cancel the transfer for '{task.title}'.",
                    severity="error",
                )
                self._request_tasks_refresh()
                return
            if outcome.status == "cancelled":
                self.app_instance.notify(
                    _cancel_toast_text(task.title),
                    severity="information",
                )
            else:
                self.app_instance.notify(
                    outcome.reason
                    or f"Could not cancel the transfer for '{task.title}'.",
                    severity="warning",
                )
            self._request_tasks_refresh()

        self.run_worker(
            _cancel_and_refresh,
            exclusive=True,
            group="schedules-transfer",
        )  # type: ignore[arg-type]

    @on(Button.Pressed, "#scheduling-new-task")
    def _on_new_task_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_new_task_chooser()

    @on(Button.Pressed, "#scheduling-new-automation")
    def _on_new_automation_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_create_automation()

    @on(Button.Pressed, "#schedules-follow-in-console")
    def follow_latest_schedule_run_in_console(self, event: Button.Pressed) -> None:
        """Hand off the active schedule run or digest output to the Console."""
        event.stop()
        if event.button.disabled:
            return
        target_id = self._latest_console_follow_item_id
        if target_id:
            open_active_item_in_console = getattr(
                self.app_instance, "open_active_home_item_in_console", None
            )
            if not callable(open_active_item_in_console):
                self.app_instance.notify(
                    "Console follow is unavailable for Schedules in this runtime.",
                    severity="warning",
                )
                return
            open_active_item_in_console(
                target_id=target_id,
                target_route="chat",
            )
            return

        launch_kwargs = self._latest_console_launch_kwargs
        if launch_kwargs is not None:
            open_in_console = getattr(
                self.app_instance, "open_console_for_live_work", None
            )
            if not callable(open_in_console):
                self.app_instance.notify(
                    "Console launch is unavailable for Schedules in this runtime.",
                    severity="warning",
                )
                return
            open_in_console(**launch_kwargs)
            return

        self.app_instance.notify(
            SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
            severity="warning",
        )

    def _task_timezones(self) -> list[str]:
        """Zones already used by tasks, offered in the form's selector."""
        zones: list[str] = []
        for task in self._tasks:
            zone = getattr(task, "timezone", None)
            if zone and zone not in zones:
                zones.append(zone)
        return zones

    def _runs_on_options(self) -> tuple[list[tuple[str, str]], str]:
        """Runs-on choices for the reminder/automation forms.

        The current screen owner leads and is always the default (spec
        sec 8); "Server (<id>)" is offered only when a server owner is
        actually connected (mirrors `_server_available`'s own gate). The
        owner might not literally be "local" or match the offered server
        option (e.g. it drifted to a different server id) -- appended as
        a labeled fallback so the Select never receives a value outside
        its own options (same precedent as the reminder form's timezone
        selector, review F4).
        """
        service = self._service()
        owner_id = service.owner_id if service else "local"
        options: list[tuple[str, str]] = [("This device", "local")]
        active_server_id = self._active_server_id()
        if self._server_available(service, active_server_id):
            options.append((f"Server ({active_server_id})", f"server:{active_server_id}"))
        if owner_id not in {value for _, value in options}:
            options.append((owner_id, owner_id))
        return options, owner_id

    def action_create_reminder(self) -> None:
        """Open the create-reminder form."""
        options, default_owner = self._runs_on_options()
        self.app.push_screen(
            ReminderForm(
                known_timezones=self._task_timezones(),
                available_owners=options,
                default_owner=default_owner,
            ),
            callback=self._on_reminder_form_result,
        )

    def action_create_automation(self) -> None:
        """Open the create-recurring-question-automation form (task-5)."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot create an automation.",
                severity="warning",
            )
            return
        options, default_owner = self._runs_on_options()
        self.app.push_screen(
            AutomationDefinitionForm(
                service, available_owners=options, default_owner=default_owner
            ),
            callback=self._on_automation_form_result,
        )

    def _open_new_task_chooser(self) -> None:
        """Ask which kind of scheduled task to create (task-5)."""
        self.app.push_screen(NewTaskChoiceModal(), callback=self._on_new_task_choice)

    def _on_new_task_choice(self, choice: str | None) -> None:
        if choice == "reminder":
            self.action_create_reminder()
        elif choice == "recurring_question":
            self.action_create_automation()

    def _on_automation_form_result(
        self, outcome: Any | None, *, was_edit: bool = False
    ) -> None:
        """Notify and refresh the Automations list after a definition save.

        `was_edit` only changes the toast wording ("updated" vs
        "created") -- an edit reusing the create-mode "created" copy
        would misreport what actually happened.
        """
        if outcome is None:
            return
        status = getattr(outcome, "status", None)
        verb = "updated" if was_edit else "created"
        if status == "saved":
            self.app_instance.notify(
                f"Automation {verb}.", severity="information"
            )
        elif status == "queued":
            self.app_instance.notify(
                f"Automation {verb} locally — it will sync to the server.",
                severity="information",
            )
        self._request_automations_refresh()

    def _on_reminder_form_result(
        self, form_data: dict[str, Any] | None, task_id: str | None = None
    ) -> None:
        """Create or update a reminder from the form and refresh the queue."""
        if form_data is None:
            return

        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot save the scheduled task.",
                severity="warning",
            )
            return

        # Routing only (task-5): which owner the reminder is written under,
        # not a ReminderTask field. Defaults to the service's current owner
        # so a form built before this selector existed (or a caller that
        # omits it) behaves exactly as before.
        target_owner = form_data.pop("owner_id", None) or service.owner_id
        # This list is owner-scoped, so a save aimed elsewhere cannot appear
        # in it -- say where it went instead of a bare "created" that reads
        # as a lost save. Label, not raw id: same vocabulary as the "Runs on"
        # selector the owner was picked from.
        owner_label = {
            value: label for label, value in self._runs_on_options()[0]
        }.get(target_owner, target_owner)
        created_message = (
            "Scheduled task created."
            if target_owner == service.owner_id
            else f"Scheduled task created for {owner_label} — switch to that "
            "owner to see it."
        )

        async def _save_and_refresh() -> None:
            # The owner is threaded through the call rather than flipped onto
            # the service around it: `service.owner_id` is shared mutable
            # state that the sync/refresh/run-now workers also read, and a
            # flip held across an awaited network round-trip is visible to
            # every one of them.
            try:
                if task_id is None:
                    await service.create_reminder(form_data, owner_id=target_owner)
                    self.app_instance.notify(created_message, severity="information")
                else:
                    await service.update_reminder(
                        task_id, form_data, owner_id=target_owner
                    )
                    self.app_instance.notify(
                        "Scheduled task updated.", severity="information"
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to save reminder")
                self.app_instance.notify(
                    "Failed to save the scheduled task. Check the form values and try again.",
                    severity="error",
                )
            self._request_tasks_refresh()

        self.run_worker(
            _save_and_refresh,
            exclusive=True,
            group="schedules-save-reminder",
        )  # type: ignore[arg-type]

    @on(AcknowledgeIncidentRequested)
    def _on_acknowledge_incident(self, event) -> None:
        """TASK-26027: acknowledge one incident, then refresh the detail."""
        event.stop()
        db = getattr(self._scheduling_service, "db", None)
        ack = getattr(db, "acknowledge_incident", None)
        if not callable(ack):
            return
        try:
            from datetime import datetime, timezone

            ack(int(event.incident_id), datetime.now(timezone.utc))
        except Exception:  # noqa: BLE001 -- ack failure never breaks the screen
            logger.debug("acknowledge_incident failed")
            return
        # re-render the detail so the acked incident drops out of the
        # alerting set and the button hides.
        if self._selected_task_id is not None:
            for index, task in enumerate(self._visible_tasks):
                if task.id == self._selected_task_id:
                    self._update_detail_for_index(index)
                    break

    @on(EditTaskRequested)
    def _on_edit_task_requested(self, event: EditTaskRequested) -> None:
        """Open the reminder form pre-filled for editing."""
        event.stop()
        options, default_owner = self._runs_on_options()
        self.app.push_screen(
            ReminderForm(
                event.task,
                known_timezones=self._task_timezones(),
                available_owners=options,
                default_owner=default_owner,
            ),
            callback=lambda result: self._on_reminder_form_result(
                result, event.task.id
            ),
        )

    @on(EnableTaskRequested)
    def _on_enable_task_requested(self, event: EnableTaskRequested) -> None:
        """Enable the requested reminder and refresh the queue."""
        event.stop()
        self._set_reminder_enabled(event.task, True)

    @on(DisableTaskRequested)
    def _on_disable_task_requested(self, event: DisableTaskRequested) -> None:
        """Disable the requested reminder and refresh the queue."""
        event.stop()
        self._set_reminder_enabled(event.task, False)

    @on(RunReminderNowRequested)
    def _on_run_reminder_now_requested(self, event: RunReminderNowRequested) -> None:
        """Dispatch the requested reminder immediately."""
        event.stop()
        self._run_reminder_now(event.task)

    def action_run_task_now(self) -> None:
        """Run the highlighted task immediately (``r`` key).

        Routes by active tab: the Automations tab's ``r`` dispatches a
        server-side run (ADR-077 -- the server owns execution); the
        Results tab's ``r`` marks the selected result read instead
        (schedules-handoff PR-6 task 3 -- a natural reading of the same
        key); everywhere else it is the local reminder Run-now
        (task-18938).
        """
        try:
            active_pane = self.query_one("#scheduling-tabs", TabbedContent).active
        except Exception:  # noqa: BLE001
            active_pane = None
        if active_pane == "scheduling-automations-tab":
            definition = self._selected_automation()
            if definition is not None:
                self._run_automation_now(definition)
            return
        if active_pane == "scheduling-results-tab":
            self._review_selected_result("read")
            return
        task = self._selected_reminder_task()
        if task is not None:
            self._run_reminder_now(task)

    def _selected_reminder_task(self) -> ReminderTask | None:
        """Return the highlighted task when it is a reminder (not a projection)."""
        for task in self._visible_tasks:
            if task.id == self._selected_task_id and isinstance(task, ReminderTask):
                return task
        return None

    def _run_reminder_now(self, task: ReminderTask) -> None:
        """Dispatch one reminder through the scheduler's own path (task-18938)."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot run the scheduled task.",
                severity="warning",
            )
            return
        loop = getattr(self.app_instance, "scheduler_loop", None)
        if loop is None:
            self.app_instance.notify(
                "The scheduler is not running; cannot run scheduled tasks manually.",
                severity="warning",
            )
            return

        # ADR-077 decision 1: server-scoped reminders are executed by the
        # server; a local Run-now would be a dispatch on the wrong side.
        # The refusal is precise rather than the generic did-not-run copy.
        # The predicate is the shared source of truth (queue.py) so the
        # UI can never drift from the scheduler/service refusal behavior.
        from tldw_chatbook.Scheduling.scheduler.queue import (
            is_server_scoped_owner,
        )

        if is_server_scoped_owner(getattr(task, "owner_id", None)):
            self.app_instance.notify(
                f"'{task.title}' is server-scheduled: the server runs it and "
                "delivers the notification -- it cannot be run from here.",
                severity="warning",
            )
            return

        was_disabled = not bool(getattr(task, "enabled", True))

        async def _run_and_refresh() -> None:
            try:
                result = await service.run_reminder_now(task.id, loop=loop)
                if result is None:
                    self.app_instance.notify(
                        f"'{task.title}' did not run -- it is missing, the "
                        "handler for it is unavailable, or its handler "
                        "failed (the task's status shows which).",
                        severity="warning",
                    )
                else:
                    suffix = " (still disabled)" if was_disabled else ""
                    self.app_instance.notify(
                        f"'{task.title}' ran now{suffix}.",
                        severity="information",
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to run reminder now")
                self.app_instance.notify(
                    f"Failed to run '{task.title}'.",
                    severity="error",
                )
            self._request_tasks_refresh()

        self.run_worker(
            _run_and_refresh,
            exclusive=True,
            group="schedules-run-reminder-now",
        )  # type: ignore[arg-type]

    def _request_automations_refresh(self) -> None:
        """Schedule the automations loader through its exclusive worker group."""
        self.run_worker(
            self.load_automations,
            exclusive=True,
            group="schedules-load-automations",
        )  # type: ignore[arg-type]

    async def load_automations(self) -> None:
        """Fetch and merge local + server automation definitions (task-5 fix round).

        This tab used to be server-only: a locally-saved recurring
        question had no on-screen home. Local rows are now merged in
        (owner distinguished via `automation_name_cell`'s Name-cell
        prefix) so a local save's refresh actually shows the new row.
        """
        notice = self.query_one("#scheduling-automations-notice", Static)
        table = self.query_one("#scheduling-automations-table", DataTable)
        service = self._scheduling_service
        if service is None:
            self._automations = []
            self._selected_automation_id = None
            table.clear()
            self._update_static_content(
                notice, "Server automations need a connected server."
            )
            self._clear_automation_history(
                "Run history needs a connected server."
            )
            return

        # task-15476 discipline: a rebuild must reconcile the selection by
        # id -- keep the cursor on the same definition when it survives the
        # refresh, and clear it when it does not, so `r`/`e` can never act
        # on a row the user is no longer looking at.
        previous_selection = self._selected_automation_id

        local_items = await self._load_local_automations(service)

        server_client = getattr(service, "server_client", None)
        server_available = server_client is not None and self._server_available(
            service, self._active_server_id()
        )
        server_items: list[dict[str, Any]] = []
        total_server = 0
        server_error = False
        if server_available:
            try:
                server_items, total_server = await self._load_server_automations(
                    server_client
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to load server automations (server_id={})",
                    self._active_server_id(),
                )
                server_error = True

        items = local_items + server_items
        self._automations = items
        table.clear()
        for definition in items:
            table.add_row(
                automation_name_cell(definition),
                str(definition.get("family", "?")),
                str(definition.get("lifecycle", "?")),
                str(definition.get("health", "?")),
                automation_execution_target_label(definition),
                key=str(definition.get("id")),
            )
        row_keys = [str(definition.get("id")) for definition in items]
        if previous_selection in row_keys:
            # Restoring the cursor fires RowHighlighted, which re-records
            # the same id -- belt and braces, set both explicitly.
            table.cursor_coordinate = (row_keys.index(previous_selection), 0)
        else:
            self._selected_automation_id = None
            self._clear_automation_history("Select an automation to see its history.")

        notice_text = self._automations_notice_text(
            local_items, server_items, server_available, total_server, server_error
        )
        self._update_static_content(notice, notice_text)

    async def _load_local_automations(
        self, service: "SchedulingService"
    ) -> list[dict[str, Any]]:
        """Every definition that exists ONLY on this device, off the event loop.

        That is not the same as `owner_id="local"`: an automation authored
        offline with "Runs on: Server" is stored under `owner_id=
        "server:<id>"` with `server_id IS NULL` until a sync pushes it, so
        filtering on the local owner hid it from this half while the
        server half could not know about it either -- it appeared in
        NEITHER list (final review I5). Any row with no `server_id`
        belongs here, whoever owns it; rows that HAVE a `server_id` are the
        server half's by construction, so the two halves cannot duplicate.

        Rows in that pending state are stamped `pending_sync` so the
        renderer can say so (`automation_name_cell`) and the actions that
        need a real server id can refuse honestly rather than calling the
        server with a local uuid.

        Health is never persisted (`automation_health.py`'s own docstring)
        -- it is computed fresh here the same way `run_automation_now`
        computes it before dispatching, so the column never shows the
        create-time placeholder (`execution_unavailable`) as if it were
        live.
        """
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        try:
            all_rows = await asyncio.to_thread(service.db.list_automation_definitions)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load local automation definitions")
            return []
        rows = [row for row in all_rows if not row.get("server_id")]
        for row in rows:
            if is_server_scoped_owner(row.get("owner_id")):
                row["pending_sync"] = True
            health, _reason = compute_local_health(self.app_instance, row)
            row["health"] = health
        return rows

    async def _load_server_automations(
        self, server_client: Any
    ) -> tuple[list[dict[str, Any]], int]:
        """Follow `has_more` pages so the tab never silently hides the tail
        of a large definition list; the cap is a defensive bound, not an
        expected cliff.

        Every item is stamped with `owner_id` if the response omitted one
        (some fixtures/older server versions do): these rows are known to
        be server-scoped by construction (this IS the server fetch), so
        the run-now/history/edit routing that reads `owner_id` off the row
        must never depend on the wire response actually including it.
        """
        items: list[dict[str, Any]] = []
        total = 0
        offset = 0
        while True:
            response = await server_client.list_automation_definitions(
                limit=50, offset=offset
            )
            page = list(response.get("items", []))
            items.extend(page)
            total = int(response.get("total", len(items)) or 0)
            offset += len(page)
            if (
                not page
                or not response.get("has_more")
                or len(items) >= AUTOMATIONS_LOAD_MAX_ROWS
            ):
                break
        active_server_id = self._active_server_id()
        for item in items:
            if not item.get("owner_id"):
                item["owner_id"] = f"server:{active_server_id}"
        return items, total

    @staticmethod
    def _automations_notice_text(
        local_items: list[dict[str, Any]],
        server_items: list[dict[str, Any]],
        server_available: bool,
        total_server: int,
        server_error: bool,
    ) -> str:
        """Compose the Automations-pane notice honestly from what actually loaded.

        A server failure never hides local rows that DID load -- the two
        sources degrade independently, so the server segment reports its
        own outcome and a local-count addendum is appended only when there
        is one to report.
        """
        if server_error:
            base = "Could not load server automations — see the log."
        elif server_available:
            shown = len(server_items)
            suffix = f" (showing {shown} of {total_server})" if total_server > shown else ""
            base = (
                f"{shown} automation{'' if shown == 1 else 's'} on the server{suffix}."
                if shown
                else "No automations on the server yet."
            )
        else:
            base = "Server automations need a connected server."
        if local_items:
            base += f" {len(local_items)} on this device."
        return base

    def _clear_automation_history(self, notice_text: str) -> None:
        """Reset the run-history pane to an explanatory notice."""
        table = self.query_one("#scheduling-automation-history-table", DataTable)
        notice = self.query_one("#scheduling-automation-history-notice", Static)
        title = self.query_one("#scheduling-automation-history-title", Static)
        table.clear()
        self._update_static_content(notice, notice_text)
        self._update_static_content(title, "Run history")

    def _request_automation_history(self, definition_id: str) -> None:
        """Schedule the audit-trail loader through its exclusive worker group."""
        # run_worker takes no worker arguments in Textual 8.x -- bind the id
        # in a closure (same shape as _run_automation_now's _run).
        async def _load() -> None:
            await self._load_automation_history(definition_id)

        self.run_worker(
            _load,
            exclusive=True,
            group="schedules-load-automation-history",
        )  # type: ignore[arg-type]

    async def _load_automation_history(self, definition_id: str) -> None:
        """Fetch and render one definition's durable execution-audit trail."""
        table = self.query_one("#scheduling-automation-history-table", DataTable)
        notice = self.query_one("#scheduling-automation-history-notice", Static)
        title = self.query_one("#scheduling-automation-history-title", Static)
        # A newer selection may have won the race with this worker; render
        # nothing for a stale definition id.
        if definition_id != self._selected_automation_id:
            return
        definition = self._selected_automation()
        name = str((definition or {}).get("name") or definition_id)
        self._update_static_content(title, f"Run history — {name}")
        # Drop the previous definition's rows BEFORE awaiting: a slow fetch
        # must never leave another definition's trail under the new title.
        table.clear()
        self._update_static_content(notice, "Loading run history…")

        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        owner_id = (definition or {}).get("owner_id")
        if (definition or {}).get("pending_sync"):
            # Never synced: the audit endpoint has no id to look up, and
            # asking it with a local uuid would render a bare error.
            table.clear()
            self._update_static_content(
                notice,
                "This automation hasn't synced to the server yet, so it has "
                "no run history.",
            )
            return
        if not is_server_scoped_owner(owner_id):
            # Honest gap (task-5 fix round): local automation runs are not
            # tracked in a durable audit trail yet -- only the server side
            # is. Never claim a server-shaped history for a local row.
            table.clear()
            self._update_static_content(
                notice, "Local automation history isn't available yet."
            )
            return

        service = self._scheduling_service
        server_client = getattr(service, "server_client", None) if service else None
        if server_client is None:
            table.clear()
            self._update_static_content(
                notice, "Run history needs a connected server."
            )
            return
        try:
            response = await server_client.list_automation_definition_audit(
                definition_id
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to load automation audit trail (definition_id={})",
                definition_id,
            )
            table.clear()
            self._update_static_content(
                notice, "Could not load the run history — see the log."
            )
            return
        if definition_id != self._selected_automation_id:
            return
        items = list(response.get("items", []))
        total = int(response.get("total", len(items)) or 0)
        table.clear()
        for event in items:
            created = str(event.get("created_at") or "")
            # Keep the timestamp compact: date and minute-level time,
            # no microseconds or timezone noise in a table cell.
            stamp = created[:16].replace("T", " ") if created else "?"
            summary = str(event.get("summary") or "")
            table.add_row(
                stamp,
                str(event.get("event_type") or "?"),
                summary,
            )
        suffix = f" of {total}" if total > len(items) else ""
        self._update_static_content(
            notice,
            f"{len(items)} event{'' if len(items) == 1 else 's'}{suffix}."
            if items
            else "No recorded events for this automation yet.",
        )

    @on(DataTable.RowHighlighted, "#scheduling-automations-table")
    def _on_automations_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        """Track the highlighted definition for Run-now and its history pane."""
        new_id = (
            str(event.row_key.value) if event.row_key and event.row_key.value else None
        )
        if new_id == self._selected_automation_id:
            return
        self._selected_automation_id = new_id
        if new_id is None:
            self._clear_automation_history("Select an automation to see its history.")
        else:
            self._request_automation_history(new_id)

    def _selected_automation(self) -> dict[str, Any] | None:
        """Return the highlighted automation definition, if any."""
        for definition in self._automations:
            if str(definition.get("id")) == self._selected_automation_id:
                return definition
        return None

    def _run_automation_now(self, definition: dict[str, Any]) -> None:
        """Dispatch one automation definition now, routed by its owner.

        Local rows use the existing PR-2 `SchedulingService.run_automation_
        now` seam (the same claim/spawn machinery the scheduled dispatch
        path uses); server rows keep the existing control-plane dispatch.
        Never both -- ADR-077 decision 1, one executor per owner.
        """
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        if definition.get("pending_sync"):
            # Server-owned but never pushed: the server has no id for it,
            # and this side must not run a server-owned definition (§3).
            self.app_instance.notify(
                "This automation hasn't synced to the server yet — it will "
                "run there after the next successful sync.",
                severity="warning",
            )
            return
        if is_server_scoped_owner(definition.get("owner_id")):
            self._run_automation_now_server(definition)
        else:
            self._run_automation_now_local(definition)

    def _run_automation_now_local(self, definition: dict[str, Any]) -> None:
        """Dispatch a local automation through the existing run-now seam."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot run the automation.",
                severity="warning",
            )
            return
        definition_id = str(definition.get("id"))
        name = str(definition.get("name") or definition_id)

        async def _run() -> None:
            try:
                result = await service.run_automation_now(definition_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Local automation run-now failed for definition {}",
                    definition_id,
                )
                self.app_instance.notify(
                    f"Failed to run '{name}'.", severity="error"
                )
                return
            if result is None:
                self.app_instance.notify(
                    f"'{name}' did not run — it is missing, paused/"
                    "archived, mid-transfer, or its health is not ready "
                    "(the definition's own state shows which).",
                    severity="warning",
                )
                return
            deduped = (
                " — deduped, a run was already in flight"
                if result.get("deduped")
                else ""
            )
            self.app_instance.notify(
                f"'{name}' ran now{deduped}.", severity="information"
            )
            self._request_automations_refresh()

        self.run_worker(
            _run,
            exclusive=True,
            group="schedules-run-automation-now",
        )  # type: ignore[arg-type]

    def _run_automation_now_server(self, definition: dict[str, Any]) -> None:
        """Dispatch one server automation through the control-plane run endpoint."""
        service = self._scheduling_service
        server_client = getattr(service, "server_client", None) if service else None
        if server_client is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot run the automation.",
                severity="warning",
            )
            return
        definition_id = str(definition.get("id"))
        name = str(definition.get("name") or definition_id)

        async def _run() -> None:
            try:
                result = await server_client.run_automation_definition_now(
                    definition_id
                )
            except ServerClientValidationError as exc:
                # Lifecycle refusals (paused/archived) and policy denials
                # arrive here with the server's own reason text.
                self.app_instance.notify(
                    f"'{name}' refused: {exc}", severity="warning"
                )
                return
            except ServerClientError as exc:
                logger.opt(exception=True).warning(
                    "Server run-now failed for definition {}",
                    definition_id,
                )
                self.app_instance.notify(
                    f"Failed to run '{name}': {exc}", severity="error"
                )
                return
            deduped = (
                " — deduped, a run for this slot was already queued"
                if result.get("deduped")
                else ""
            )
            run_slot = str(result.get("run_slot_utc") or "unknown slot")
            self.app_instance.notify(
                f"'{name}' dispatched to the server (slot {run_slot}){deduped}. "
                "The result arrives as a notification.",
                severity="information",
            )
            # The dispatch returns when the run is ENQUEUED, not finished:
            # the terminal audit event lands only after the server executes
            # (an LLM call -- seconds). One immediate fetch catches the
            # dispatch-time events; a delayed fetch catches quick terminal
            # outcomes. Long runs stay stale until the next selection or
            # sync -- honest, and the notification still reports the result.
            self._request_automation_history(definition_id)
            self.set_timer(
                AUTOMATION_HISTORY_FOLLOWUP_SECONDS,
                lambda: self._request_automation_history(definition_id)
                if self._selected_automation_id == definition_id
                else None,
            )

        self.run_worker(
            _run,
            exclusive=True,
            group="schedules-run-automation-now",
        )  # type: ignore[arg-type]

    def _edit_selected_automation(self) -> None:
        """Open the selected automation definition for editing (e key).

        `agent_task` rows are excluded -- only `recurring_question`
        authoring exists (the same v1 scope guard `save_definition`
        itself enforces via `_reject_unsupported_family`).
        """
        definition = self._selected_automation()
        if definition is None:
            self.app_instance.notify(
                "Nothing to edit — select an automation first.",
                severity="warning",
            )
            return
        if definition.get("family") != "recurring_question":
            self.app_instance.notify(
                "Only recurring-question automations can be edited here "
                "(agent-task authoring is not yet available).",
                severity="warning",
            )
            return
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot edit the automation.",
                severity="warning",
            )
            return

        async def _open() -> None:
            definition_id = await self._resolve_local_definition_id(
                service, definition
            )
            if definition_id is None:
                self.app_instance.notify(
                    "Could not prepare this automation for editing — see "
                    "the log.",
                    severity="error",
                )
                return
            owner_id = str(definition.get("owner_id") or "local")
            options, _default = self._runs_on_options()
            if owner_id not in {value for _, value in options}:
                # Same defensive fallback as the reminder form's timezone/
                # owner selectors: the row's real owner always round-trips
                # even when it is not among the currently offered choices
                # (e.g. a server owner other than the active one).
                options = [*options, (owner_id, owner_id)]
            self.app.push_screen(
                AutomationDefinitionForm(
                    service,
                    definition_row=definition,
                    definition_id=definition_id,
                    available_owners=options,
                    default_owner=owner_id,
                ),
                callback=lambda outcome: self._on_automation_form_result(
                    outcome, was_edit=True
                ),
            )

        self.run_worker(_open, exclusive=True, group="schedules-edit-automation")

    async def _resolve_local_definition_id(
        self, service: "SchedulingService", definition: dict[str, Any]
    ) -> str | None:
        """Return the LOCAL row id `save_definition`'s `definition_id` needs.

        `save_definition`'s `definition_id` parameter is always a LOCAL
        id (Task 4's own contract). A row shown here from a pure server
        fetch may have no local shadow yet (nothing has synced or saved
        it locally before) -- editing it still needs one, so this mirrors
        it in place via the same `upsert_automation_definitions_from_
        server` the sync pull and `save_definition`'s own online-create
        path already use, rather than inventing a second mirroring path.
        """
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        owner_id = str(definition.get("owner_id") or "local")
        if definition.get("pending_sync") or not is_server_scoped_owner(owner_id):
            # A `pending_sync` row IS a local row (server-owned, never
            # synced): its `id` is already the local one, and treating it
            # as a server id here would mirror it back as a SECOND row
            # keyed by that uuid.
            local_id = definition.get("id")
            return str(local_id) if local_id else None

        server_id = str(definition.get("id"))
        existing = await asyncio.to_thread(
            service.db.get_automation_definition_by_server_id, owner_id, server_id
        )
        if existing is not None:
            return existing.get("id")
        try:
            await asyncio.to_thread(
                service.db.upsert_automation_definitions_from_server,
                owner_id,
                [definition],
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to mirror server automation {} before editing",
                server_id,
            )
            return None
        mirrored = await asyncio.to_thread(
            service.db.get_automation_definition_by_server_id, owner_id, server_id
        )
        return mirrored.get("id") if mirrored else None

    # -- Automations-tab transfer actions (schedules-handoff spec §6,
    # PR-5 task 7 fix round item 1) -----------------------------------
    #
    # With PR-5 as first shipped, `begin_transfer_to_server`/`begin_
    # transfer_to_local`/`cancel_transfer` had no UI call site at all for
    # `table_kind="automation_definition"` -- the facade fully supports
    # it (Task 6), but nothing let a user ever start, cancel, or retry a
    # definition's transfer. The tab has no per-row detail widget (no
    # `TaskDetail` equivalent), so these are keybindings routed by active
    # tab -- the SAME idiom `action_run_task_now`/`action_edit_task`
    # already use for this tab's Run-now/Edit, not new buttons. A
    # refusal renders inline in the tab's own notice Static (its only
    # existing status affordance -- reused rather than adding a second
    # one); an allowed Move reuses the same `ConfirmationDialog` +
    # honest-toast shape the Queue tab's reminder flow already
    # established. Move to server (`M`) and Retry (`y`) share one call:
    # a retry IS a re-begin, so the difference is the label and which
    # state each is offered in, not the code path.

    def action_move_automation_to_local(self) -> None:
        """m key: queue a server-owned automation mirror to move here
        (spec §6.2), Automations-tab only."""
        self._begin_automation_transfer("to_local")

    def action_move_automation_to_server(self) -> None:
        """M key: queue a LOCAL automation definition to move to the
        server (spec §6.1), Automations-tab only.

        The missing half of the definitions transfer UI (final review
        M8): `y`/Retry already reached `begin_transfer_to_server`, but
        only ever as a retry beside a failed transfer, so a plain local
        definition had no way to start one. Same facade call -- a retry
        IS a re-begin (`transfer_refusal` excludes `to_server_failed`
        from "in progress", and the CAS accepts both `None` and
        `to_server_failed`) -- given its own honest label and key, rather
        than teaching users that "Retry" means "Move".
        """
        self._begin_automation_transfer("to_server")

    def action_retry_automation_transfer(self) -> None:
        """y key: retry a definitively-failed local -> server automation
        transfer (spec §6.1.5), Automations-tab only."""
        self._begin_automation_transfer("to_server")

    def action_cancel_automation_transfer(self) -> None:
        """k key: cancel the selected automation's in-progress transfer
        (spec §6.3), Automations-tab only."""
        self._cancel_automation_transfer()

    def _is_automations_tab_active(self) -> bool:
        try:
            return (
                self.query_one("#scheduling-tabs", TabbedContent).active
                == "scheduling-automations-tab"
            )
        except Exception:  # noqa: BLE001
            return False

    def _show_automations_inline_reason(self, reason: str) -> None:
        """Surface a transfer refusal/failure inline (fix round item 1's
        "refusal -> inline reason" flow) -- the Automations pane's notice
        Static is the tab's only existing status affordance
        (`_automations_notice_text`'s own home), reused rather than
        adding a second one. The next `load_automations()` (any other
        action, or a completed transfer's own reload) naturally replaces
        it with the aggregate summary again.
        """
        try:
            notice = self.query_one("#scheduling-automations-notice", Static)
        except Exception:  # noqa: BLE001
            return
        self._update_static_content(notice, reason)

    def _begin_automation_transfer(self, direction: str) -> None:
        """Move-to-local / Retry for the selected automation (spec
        §6.1/§6.2), Automations-tab only. Same flow as the Queue tab's
        `_begin_transfer`: refusal -> inline reason; allowed ->
        `ConfirmationDialog` listing `transfer_warnings`; honest toast on
        completion; reload via the tab's existing `_request_automations_
        refresh` wiring.
        """
        if not self._is_automations_tab_active():
            self.app_instance.notify(
                "Switch to the Automations tab to move or retry an "
                "automation's transfer.",
                severity="warning",
            )
            return
        definition = self._selected_automation()
        if definition is None:
            self.app_instance.notify(
                "Nothing selected — select an automation first.",
                severity="warning",
            )
            return
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot start a transfer.",
                severity="warning",
            )
            return

        async def _resolve_and_begin() -> None:
            local_id = await self._resolve_local_definition_id(service, definition)
            if local_id is None:
                self.app_instance.notify(
                    "Could not prepare this automation for transfer — see "
                    "the log.",
                    severity="error",
                )
                return
            row = await asyncio.to_thread(
                service.db.get_automation_definition, local_id
            )
            if row is None:
                self.app_instance.notify(
                    "This automation no longer exists.", severity="warning"
                )
                self._request_automations_refresh()
                return
            name = str(row.get("name") or definition.get("name") or local_id)
            reason = service.transfer_refusal(row, direction)
            if reason is not None:
                self._show_automations_inline_reason(reason)
                return
            warnings = service.transfer_warnings(row, direction)
            dialog = self._transfer_confirm_dialog(name, direction, warnings)
            confirmed = await self.app.push_screen_wait(dialog)
            if not confirmed:
                return
            try:
                if direction == "to_server":
                    outcome = await service.begin_transfer_to_server(
                        "automation_definition", local_id
                    )
                else:
                    outcome = await service.begin_transfer_to_local(
                        "automation_definition", local_id
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to begin automation transfer for {}", local_id
                )
                self.app_instance.notify(
                    f"Failed to start the transfer for '{name}'.",
                    severity="error",
                )
                self._request_automations_refresh()
                return
            if outcome.status == "pending":
                self.app_instance.notify(
                    self._transfer_pending_toast_text(name, direction),
                    severity="information",
                )
            elif outcome.status == "refused":
                self._show_automations_inline_reason(
                    outcome.reason or f"Could not move '{name}'."
                )
            elif outcome.status == "not_found":
                self.app_instance.notify(
                    f"'{name}' no longer exists.", severity="warning"
                )
            self._request_automations_refresh()

        self.run_worker(
            _resolve_and_begin,
            exclusive=True,
            group="schedules-automation-transfer",
        )  # type: ignore[arg-type]

    def _cancel_automation_transfer(self) -> None:
        """Cancel the selected automation's in-progress transfer (spec
        §6.3), Automations-tab only. No confirm dialog -- same rationale
        as the Queue tab's cancel: it is the escape hatch, gating it
        behind its own confirmation fights the point. The resolved LOCAL
        row id is already the dormant copy's own id for a release in
        progress -- `_load_local_automations` shows that copy directly (a
        `server_id`-less local row), never through the mirror, so
        resolution never routes to the mirror's id for that leg (same
        construction as the reminder side's cancel).
        """
        if not self._is_automations_tab_active():
            self.app_instance.notify(
                "Switch to the Automations tab to cancel an automation's "
                "transfer.",
                severity="warning",
            )
            return
        definition = self._selected_automation()
        if definition is None:
            self.app_instance.notify(
                "Nothing selected — select an automation first.",
                severity="warning",
            )
            return
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot cancel the "
                "transfer.",
                severity="warning",
            )
            return

        async def _resolve_and_cancel() -> None:
            local_id = await self._resolve_local_definition_id(service, definition)
            if local_id is None:
                self.app_instance.notify(
                    "Could not resolve this automation locally — see the "
                    "log.",
                    severity="error",
                )
                return
            name = str(definition.get("name") or local_id)
            try:
                outcome = await service.cancel_transfer(
                    "automation_definition", local_id
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to cancel automation transfer for {}", local_id
                )
                self.app_instance.notify(
                    f"Failed to cancel the transfer for '{name}'.",
                    severity="error",
                )
                self._request_automations_refresh()
                return
            if outcome.status == "cancelled":
                self.app_instance.notify(
                    _cancel_toast_text(name), severity="information"
                )
            else:
                self._show_automations_inline_reason(
                    outcome.reason
                    or f"Could not cancel the transfer for '{name}'."
                )
            self._request_automations_refresh()

        self.run_worker(
            _resolve_and_cancel,
            exclusive=True,
            group="schedules-automation-transfer",
        )  # type: ignore[arg-type]

    # -- Results-tab actions (schedules-handoff PR-6 task 3) ---------------
    #
    # Read/dismiss reuse r/d via action_run_task_now/action_delete's own
    # tab routing above. Mark-solved/Mark-all-read get fresh keys (o/a),
    # guarded the same way m/M/y/k refuse off the Automations tab.

    def _is_results_tab_active(self) -> bool:
        try:
            return (
                self.query_one("#scheduling-tabs", TabbedContent).active
                == "scheduling-results-tab"
            )
        except Exception:  # noqa: BLE001
            return False

    def _review_selected_result(self, review_state: str) -> None:
        """r/d on the Results tab: read/dismiss the selected result.

        `SchedulingService.review_automation_result` writes the local row
        and, for a server mirror, queues the PR-3 pushback mutation in the
        SAME DB transaction -- nothing extra to do here for that half.
        """
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        results_tab = self.query_one("#scheduling-results", ResultsTab)
        result = results_tab.selected_result()
        if result is None:
            self.app_instance.notify("Select a result first.", severity="warning")
            return

        async def _review() -> None:
            updated = await service.review_automation_result(
                result["id"], review_state
            )
            if not updated:
                self.app_instance.notify(
                    "Could not update this result — see the log.",
                    severity="error",
                )
            self._refresh_results_tab()

        self.run_worker(
            _review, exclusive=True, group="schedules-review-result"
        )  # type: ignore[arg-type]

    def action_mark_result_solved(self) -> None:
        """o key: mark the selected finding's definition solved (Task 2's
        facade), Results-tab only. Refused client-side for a row `solved_
        eligibility` already rules out (wrong kind, already solved, or an
        unknown definition); a still-eligible row can still be refused by
        the facade itself (transfer lock, offline+server-owned -- UX-073),
        surfaced from `ResolveOutcome.reason`.
        """
        if not self._is_results_tab_active():
            self.app_instance.notify(
                "Switch to the Results tab to mark a result solved.",
                severity="warning",
            )
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        results_tab = self.query_one("#scheduling-results", ResultsTab)
        result = results_tab.selected_result()
        if result is None:
            self.app_instance.notify("Select a result first.", severity="warning")
            return
        eligible, reason = solved_eligibility(result, results_tab.definitions_by_id)
        if not eligible:
            self.app_instance.notify(
                reason or "This result cannot be marked solved.",
                severity="warning",
            )
            return

        async def _mark_solved() -> None:
            outcome = await service.resolve_definition(
                result["definition_id"], solved=True, result_id=result["id"]
            )
            if outcome.status == "saved":
                self.app_instance.notify("Marked solved.", severity="information")
            else:
                self.app_instance.notify(
                    outcome.reason or "Could not mark this result solved.",
                    severity="warning",
                )
            self._refresh_results_tab()

        self.run_worker(
            _mark_solved, exclusive=True, group="schedules-mark-solved"
        )  # type: ignore[arg-type]

    def action_mark_all_results_read(self) -> None:
        """a key: mark every currently-loaded unread result read,
        Results-tab only. Per-row `review_automation_result` calls -- there
        is no bulk DB primitive for this (spec's documented fan-out),
        mirroring `_on_bulk_delete_confirmed`'s loop-and-count shape.
        """
        if not self._is_results_tab_active():
            self.app_instance.notify(
                "Switch to the Results tab to mark all results read.",
                severity="warning",
            )
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        results_tab = self.query_one("#scheduling-results", ResultsTab)
        unread_ids = [
            result["id"]
            for result in results_tab.results()
            if result.get("review_state") == "unread"
        ]
        if not unread_ids:
            self.app_instance.notify("Nothing unread.", severity="information")
            return

        async def _mark_all() -> None:
            errors = 0
            for result_id in unread_ids:
                if not await service.review_automation_result(result_id, "read"):
                    errors += 1
            count = len(unread_ids) - errors
            self.app_instance.notify(
                f"Marked {count} result{'s' if count != 1 else ''} read"
                + (f" ({errors} failed)" if errors else "")
                + ".",
                severity="information" if not errors else "warning",
            )
            self._refresh_results_tab()

        self.run_worker(
            _mark_all, exclusive=True, group="schedules-mark-all-read"
        )  # type: ignore[arg-type]

    def _set_reminder_enabled(self, task: ReminderTask, enabled: bool) -> None:
        """Update a reminder's enabled state and refresh the queue."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot update the scheduled task.",
                severity="warning",
            )
            return

        async def _update_and_refresh() -> None:
            try:
                await service.update_reminder(task.id, {"enabled": enabled})
                status = "enabled" if enabled else "disabled"
                self.app_instance.notify(
                    f"'{task.title}' {status}.", severity="information"
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to update reminder enabled state")
                self.app_instance.notify(
                    f"Failed to update '{task.title}'.",
                    severity="error",
                )
            self._request_tasks_refresh()

        self.run_worker(
            _update_and_refresh,
            exclusive=True,
            group="schedules-set-reminder-enabled",
        )  # type: ignore[arg-type]

    def _refresh_owner_select(self) -> None:
        status = self.query_one("#scheduling-sync-status", SyncStatusWidget)
        service = self._service()
        if service is None:
            status.set_owner_state("local", None, False)
            status.update_status(None, None, [])
            self._sync_header_status("blocked", "Scheduling unavailable")
            return
        active_server_id = self._active_server_id()
        server_available = self._server_available(service, active_server_id)
        status.set_owner_state(service.owner_id, active_server_id, server_available)
        state = service.db.get_sync_state(service.owner_id) or {}
        sync_errors = state.get("sync_errors") or []
        # A runtime-mode refusal is "sync not applicable", never a failure.
        # New refusals are no longer recorded (task-2722, SyncEngine), but
        # profiles that synced on older builds still carry persisted ones —
        # keep them off the error surface instead of badging local-only
        # profiles with an error the user did nothing to cause.
        sync_errors = [
            entry
            for entry in sync_errors
            if "requires server mode" not in str(entry.get("message", ""))
        ]
        status.update_status(
            last_pull_at=state.get("last_pull_at"),
            last_push_at=state.get("last_push_at"),
            sync_errors=sync_errors,
        )
        if sync_errors:
            count = len(sync_errors)
            self._sync_header_status(
                "error", f"{count} sync error{'s' if count != 1 else ''}"
            )
        elif not server_available:
            self._sync_header_status("empty", "Local only — no server connection")
        elif service.owner_id.startswith("server:"):
            self._sync_header_status("ready", "Synced with server")
        else:
            self._sync_header_status("ready", "Local schedules")

    def _sync_header_status(self, status: WorkbenchStatus, label: str) -> None:
        """Reflect real sync health in the destination header chip."""
        try:
            header = self.query_one("#schedules-destination-header", DestinationHeader)
        except Exception:  # noqa: BLE001 - header not mounted yet
            return
        header.sync_state(
            WorkbenchHeaderState(
                title="Schedules",
                subtitle="When jobs, watchlists, and workflows run.",
                status=status,
                status_label=label,
            )
        )

    def on_resize(self) -> None:
        """Hide side panes (with a notice) instead of clipping them."""
        self._sync_responsive_workbench()
        try:
            width = self.size.width
            inspector = self.query_one("#scheduling-inspector-pane")
            detail = self.query_one("#scheduling-detail-pane")
        except Exception:  # noqa: BLE001 - panes not mounted yet
            return
        hide_inspector = 0 < width < 118
        hide_detail = 0 < width < 84
        inspector.set_class(hide_inspector, "pane-hidden")
        detail.set_class(hide_detail, "pane-hidden")
        # At detail-hiding widths the pane chrome also gets too tall to fit:
        # the Queue tab label already names this pane, so the in-pane title
        # yields its row to the table + notice (see _scheduling.tcss).
        self.query_one("#scheduling-workbench").set_class(hide_detail, "compact")
        if hide_detail:
            # The create CTA normally lives in the (now hidden) detail pane;
            # keep it reachable at compact widths when the queue is empty.
            base = "Detail and inspector hidden — widen the window to see them."
            if not self._tasks:
                base += " Press c to schedule your first task."
            self._resize_notice = base
        elif hide_inspector:
            self._resize_notice = "Inspector hidden — widen the window to see it."
        else:
            self._resize_notice = ""
        self._update_pane_notice()

    def _update_pane_notice(self) -> None:
        """Compose the queue-pane notice: hidden panes, marks, glyph legend.

        task-23107: while rows are marked, visible text states the count,
        the keys that act on all marked rows, and how to clear the marks;
        the ◇ missed-while-away glyph gets an on-screen explanation
        whenever a visible row carries it.
        """
        try:
            notice = self.query_one("#scheduling-pane-notice", Static)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        parts: list[str] = []
        if self._resize_notice:
            parts.append(self._resize_notice)
        # Marking is reminder-only and marks are pruned on load, so the
        # legend count IS the count the bulk verbs act on (review F1).
        marked_count = len(self._marked_reminder_tasks())
        if marked_count:
            visible_ids = {task.id for task in self._visible_tasks}
            hidden = sum(
                1 for task_id in self._marked_ids if task_id not in visible_ids
            )
            hidden_note = f" ({hidden} hidden by the filter)" if hidden else ""
            parts.append(
                f"{marked_count} marked{hidden_note} — space toggles all "
                "· d deletes all · esc clears"
            )
        if any(_was_missed_while_away(task) for task in self._visible_tasks):
            parts.append("◇ = ran late (dispatched after its scheduled time)")
        self._update_static_content(notice, "\n".join(parts))

    @on(Button.Pressed, "#scheduling-owner-local")
    def _on_owner_local(self) -> None:
        self._set_owner("local")

    @on(Button.Pressed, "#scheduling-owner-server")
    def _on_owner_server(self) -> None:
        service = self._service()
        if service is None:
            return
        active_server_id = self._active_server_id()
        if not self._server_available(service, active_server_id):
            self.app_instance.notify("No server connection", severity="warning")
            return
        self._set_owner(f"server:{active_server_id}")

    def _set_owner(self, new_owner: str) -> None:
        service = self._service()
        if service is None:
            return
        service.set_owner(new_owner)
        runtime_source = "server" if new_owner.startswith("server:") else "local"
        set_authoritative_runtime_source(
            self.app_instance.runtime_policy,
            runtime_source,
            app_config=self.app_instance.app_config,
        )
        self._refresh_owner_select()
        self._request_tasks_refresh()
        self._refresh_conflicts_tab()

    @on(Button.Pressed, "#scheduling-clear-error")
    def _on_clear_sync_errors(self) -> None:
        service = self._service()
        if service is None:
            return
        service.db.update_sync_state(service.owner_id, sync_errors=[])
        self._refresh_owner_select()

    @on(SyncCompleted)
    def _on_sync_completed(self, event: SyncCompleted) -> None:
        self._sync_running = False
        outcome = event.outcome
        status = getattr(outcome, "status", None)
        pulled = int(getattr(outcome, "pulled", 0) or 0)
        pushed = int(getattr(outcome, "pushed", 0) or 0)
        if outcome is None:
            # Legacy sender without an outcome.
            message = "Sync completed."
        elif status == "not_applicable":
            message = (
                "Sync skipped — not applicable in this mode; nothing was "
                "pulled or pushed."
            )
        elif pulled or pushed:
            message = f"Sync completed — pulled {pulled}, pushed {pushed}."
        else:
            message = "Sync finished — nothing to pull or push."
        self.app_instance.notify(message, severity="information")
        self._refresh_owner_select()
        self._request_tasks_refresh()
        self._request_automations_refresh()
        self._refresh_conflicts_tab()
        self._refresh_results_tab()

    @on(SyncFailed)
    def _on_sync_failed(self, event: SyncFailed) -> None:
        self._sync_running = False
        self.app_instance.notify(f"Sync failed: {event.error}", severity="error")
        self._refresh_owner_select()
        self._request_tasks_refresh()
        self._request_automations_refresh()
        self._refresh_conflicts_tab()
        self._refresh_results_tab()

    @on(ConflictsTab.ConflictResolved)
    def _on_conflict_resolved(self, event: ConflictsTab.ConflictResolved) -> None:
        self._request_tasks_refresh()
        self._refresh_conflicts_tab()

    def _refresh_conflicts_tab(self) -> None:
        service = self._service()
        if service is None:
            return
        conflicts_tab = self.query_one("#scheduling-conflicts", ConflictsTab)
        conflicts = service.db.get_conflicts(
            service.owner_id, primitive="reminder_task"
        )
        conflicts_tab.populate(conflicts)
        # Surface the conflict count on the tab label itself (UX-063).
        try:
            pane = self.query_one("#scheduling-conflicts-tab", TabPane)
            pane.label = f"Conflicts ({len(conflicts)})" if conflicts else "Conflicts"
        except Exception:  # noqa: BLE001 - pane not mounted
            pass

    def _refresh_results_tab(self) -> None:
        """Reload the Results tab and its unread badge (schedules-handoff
        PR-6 task 3). Mirrors `_refresh_conflicts_tab`'s shape: direct
        `service.db.*` calls (list_automation_results/count_unread_
        results span every owner -- Task 1), no worker -- this is a local
        DB-only read, same cost class as `get_conflicts`. Also called
        after Task 4's notification-triggered pull and after every
        read/dismiss/mark-solved/mark-all-read action below.
        """
        service = self._service()
        if service is None:
            return
        results_tab = self.query_one("#scheduling-results", ResultsTab)
        results = service.db.list_automation_results(owner_id=None)
        unread = service.db.count_unread_results(owner_id=None)
        definitions_by_id = {
            row["id"]: row
            for row in service.db.list_automation_definitions(owner_id=None)
        }
        results_tab.populate(results, definitions_by_id)
        # Surface the unread count on the tab label itself (spec §4's
        # inbox badge, same UX-063 idiom as the Conflicts tab above).
        try:
            pane = self.query_one("#scheduling-results-tab", TabPane)
            pane.label = f"Results ({unread})" if unread else "Results"
        except Exception:  # noqa: BLE001 - pane not mounted
            pass

    def action_delete(self) -> None:
        """Delete marked tasks in bulk, else the selected one (confirmed).

        While ANY mark exists, d never falls through to the highlighted,
        unmarked row (task-23107 review F1): acting on a row the user
        never marked is worse than refusing. On the Results tab, ``d``
        dismisses the selected result instead (schedules-handoff PR-6
        task 3) -- same key, the tab-appropriate "remove from view" verb.
        """
        try:
            active_pane = self.query_one("#scheduling-tabs", TabbedContent).active
        except Exception:  # noqa: BLE001
            active_pane = None
        if active_pane == "scheduling-results-tab":
            self._review_selected_result("dismissed")
            return
        if self._marked_ids:
            marked = self._marked_reminder_tasks()
            if not marked:
                # Defensive: marking is reminder-only and marks are pruned
                # on every load, so this means the marked rows vanished
                # between renders. Refuse instead of falling through.
                self._marked_ids.clear()
                self._render_table()
                self.app_instance.notify(
                    "The marked rows are no longer in the queue — marks "
                    "cleared; nothing was deleted.",
                    severity="warning",
                )
                return
            from ....Widgets.delete_confirmation_dialog import (
                DeleteConfirmationDialog,
            )

            self.app.push_screen(
                DeleteConfirmationDialog(
                    item_type="Scheduled tasks",
                    item_name=f"{len(marked)} marked tasks",
                    permanent=True,
                ),
                callback=lambda confirmed: self._on_bulk_delete_confirmed(
                    confirmed, marked
                ),
            )
            return
        if not self._tasks:
            self.app_instance.notify(
                "Nothing to delete — the queue is empty.",
                severity="warning",
            )
            return
        if self._refuse_if_transfer_locked(self._selected_task(), "delete this task"):
            return
        self.query_one("#scheduling-task-detail", TaskDetail).request_delete()

    def _on_bulk_delete_confirmed(self, confirmed, marked: list[ReminderTask]) -> None:
        """Delete all marked tasks after the confirmation dialog."""
        if not confirmed:
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot delete tasks.",
                severity="warning",
            )
            return

        async def _bulk_delete() -> None:
            errors = 0
            for task in marked:
                try:
                    # A False return is a refusal, not a crash (e.g. the
                    # transfer read-only guard, final review I7) -- count
                    # it, so the toast never claims a delete that the
                    # facade declined.
                    if not await service.delete_reminder(task.id):
                        errors += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to delete reminder {}", task.id)
                    errors += 1
            count = len(marked) - errors
            self.app_instance.notify(
                f"Deleted {count} marked task{'s' if count != 1 else ''}"
                + (f" ({errors} failed)" if errors else "")
                + ".",
                severity="information" if not errors else "warning",
            )
            self._marked_ids.clear()
            self._request_tasks_refresh()

        self.run_worker(
            _bulk_delete,
            exclusive=True,
            group="schedules-bulk-delete",
        )  # type: ignore[arg-type]

    def _selected_task(self) -> ReminderTask | ScheduledTask | None:
        """Return the task under the queue cursor, if any."""
        if not self._visible_tasks:
            return None
        table = self.query_one("#scheduling-task-table", DataTable)
        row = table.cursor_row
        if row is None or not (0 <= row < len(self._visible_tasks)):
            return None
        return self._visible_tasks[row]

    def action_edit_task(self) -> None:
        """Open the highlighted task/definition in its edit form (e key).

        Routes by active tab, same shape as `action_run_task_now`: the
        Automations tab's `e` opens `AutomationDefinitionForm` pre-filled
        for a `recurring_question` row (either owner); everywhere else it
        is the existing reminder edit flow.
        """
        try:
            active_pane = self.query_one("#scheduling-tabs", TabbedContent).active
        except Exception:  # noqa: BLE001
            active_pane = None
        if active_pane == "scheduling-automations-tab":
            self._edit_selected_automation()
            return
        task = self._selected_task()
        if task is None:
            self.app_instance.notify(
                "Nothing to edit — select a task first.",
                severity="warning",
            )
            return
        if not isinstance(task, ReminderTask):
            # task-23106: say who owns the row instead of exposing the
            # internal reminder/projection split.
            self.app_instance.notify(
                _managed_elsewhere_notice(task, verb="edit"),
                severity="warning",
            )
            return
        if self._refuse_if_transfer_locked(task, "edit this task"):
            return
        self.post_message(EditTaskRequested(task))

    def action_mark_task(self) -> None:
        """Mark/unmark the highlighted task for bulk actions (x key).

        Only rows the bulk verbs can act on are markable (task-23107
        review F1): marking a read-only projection row would either be
        silently ignored by the bulk actions or, worse, let them fall
        through to an unmarked row.
        """
        task = self._selected_task()
        if task is None:
            self.app_instance.notify(
                "Nothing to mark — select a task first.",
                severity="warning",
            )
            return
        if not isinstance(task, ReminderTask):
            self.app_instance.notify(
                _managed_elsewhere_notice(task, verb="manage"),
                severity="warning",
            )
            return
        if task.id in self._marked_ids:
            self._marked_ids.discard(task.id)
        else:
            self._marked_ids.add(task.id)
        self._render_table()

    def action_clear_marks(self) -> None:
        """Clear all bulk marks (escape key)."""
        if self._marked_ids:
            self._marked_ids.clear()
            self._render_table()

    def _marked_reminder_tasks(self) -> list[ReminderTask]:
        """Marked tasks that support bulk operations."""
        return [
            task
            for task in self._tasks
            if task.id in self._marked_ids and isinstance(task, ReminderTask)
        ]

    def action_toggle_enabled(self) -> None:
        """Enable/disable marked tasks in bulk, else the highlighted one.

        While ANY mark exists, space never falls through to the
        highlighted, unmarked row (task-23107 review F1).
        """
        if self._marked_ids:
            marked = self._marked_reminder_tasks()
            if not marked:
                self._marked_ids.clear()
                self._render_table()
                self.app_instance.notify(
                    "The marked rows are no longer in the queue — marks "
                    "cleared; nothing was toggled.",
                    severity="warning",
                )
                return
            self._bulk_toggle_marked(marked)
            return

        task = self._selected_task()
        if task is None:
            self.app_instance.notify(
                "Nothing to toggle — select a task first.",
                severity="warning",
            )
            return
        if not isinstance(task, ReminderTask):
            # task-23106: say who owns the row instead of exposing the
            # internal reminder/projection split.
            self.app_instance.notify(
                _managed_elsewhere_notice(task, verb="enable or disable"),
                severity="warning",
            )
            return
        if self._refuse_if_transfer_locked(task, "enable or disable this task"):
            return
        self._set_reminder_enabled(task, not task.enabled)

    def _bulk_toggle_marked(self, marked: list[ReminderTask]) -> None:
        """Toggle every marked task's enabled state (space with marks)."""
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot update the scheduled tasks.",
                severity="warning",
            )
            return

        async def _bulk_toggle() -> None:
            errors = 0
            for task in marked:
                try:
                    # A None return is a refusal, not a crash -- same
                    # reasoning as the bulk delete above.
                    if await service.update_reminder(
                        task.id, {"enabled": not task.enabled}
                    ) is None:
                        errors += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to toggle reminder {}", task.id)
                    errors += 1
            count = len(marked) - errors
            self.app_instance.notify(
                f"Toggled {count} marked task{'s' if count != 1 else ''}"
                + (f" ({errors} failed)" if errors else "")
                + ".",
                severity="information" if not errors else "warning",
            )
            self._marked_ids.clear()
            self._request_tasks_refresh()

        self.run_worker(
            _bulk_toggle,
            exclusive=True,
            group="schedules-bulk-toggle",
        )  # type: ignore[arg-type]

    def action_sync_now(self) -> None:
        """Sync schedule state now."""
        if self._sync_running:
            self.app_instance.notify("Sync already in progress", severity="warning")
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot sync.",
                severity="warning",
            )
            return
        if not self._server_available(service, self._active_server_id()):
            # Honest no-op: never claim "Sync completed" when nothing can
            # sync. Same predicate as the sync bar's collapse (review F10):
            # the bar and the s key must agree on whether sync is possible.
            self.app_instance.notify(
                "Local only — nothing to sync (no server connection).",
                severity="information",
            )
            return
        self._sync_running = True
        self.run_worker(self._run_sync, exclusive=True, group="schedules-sync-now")

    async def _run_sync(self) -> None:
        service = self._service()
        if service is None:
            self._sync_running = False
            return
        for btn_id in ("#scheduling-owner-local", "#scheduling-owner-server"):
            self.query_one(btn_id, Button).disabled = True
        try:
            owner_id = service.owner_id
            # task-23105 review F3: the engine swallows server errors into
            # persisted sync-error state, so its returned SyncOutcome is
            # the only honest report of what the attempt did -- a failed
            # sync must not surface as an info-severity no-op.
            outcome = await service.sync_now(owner_id)
            if outcome is not None and getattr(outcome, "status", None) == "error":
                self.post_message(
                    SyncFailed(
                        owner_id, getattr(outcome, "error", None) or "sync error"
                    )
                )
                return
            conflicts = service.db.get_conflicts(owner_id, primitive="reminder_task")
            self.post_message(
                SyncCompleted(
                    owner_id,
                    conflict_count=len(conflicts),
                    outcome=outcome,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sync failed")
            self.post_message(SyncFailed(service.owner_id, str(exc)))
        finally:
            for btn_id in ("#scheduling-owner-local", "#scheduling-owner-server"):
                self.query_one(btn_id, Button).disabled = False
            self._refresh_owner_select()
            self._sync_running = False
