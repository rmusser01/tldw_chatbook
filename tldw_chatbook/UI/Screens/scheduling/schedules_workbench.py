"""Schedules workbench shell for run timing, triggers, and recovery."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Button, DataTable, Input, Static, TabbedContent, TabPane

from ...Navigation.base_app_screen import BaseAppScreen
from ...Navigation.screen_state_store import RuntimeIdentity
from ...Workbench.workbench_state import RecoveryState, WorkbenchHeaderState, WorkbenchStatus
from ...Workbench.workbench_widgets import DestinationHeader, RecoveryCallout
from ....runtime_policy.bootstrap import set_authoritative_runtime_source
from ....Scheduling.events import (
    DeleteTaskRequested,
    DisableTaskRequested,
    EditTaskRequested,
    EnableTaskRequested,
    RunReminderNowRequested,
    SyncCompleted,
    SyncFailed,
)
from ....Scheduling.models import ReminderTask, ScheduledTask
from ....UI.Screens.scheduling.conflicts_tab import ConflictsTab
from ....UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
from .forms.reminder_form import ReminderForm
from .task_detail import (
    SCHEDULES_EMPTY_CONSOLE_RECOVERY,
    TaskDetail,
    TaskInspector,
    _format_next_run,
    _managed_elsewhere_notice,
    _task_status,
    _task_type_label,
    _underlying_status,
    _was_missed_while_away,
    status_badge_text,
)

if TYPE_CHECKING:
    from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
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
            with TabbedContent(id="scheduling-tabs"):
                with TabPane("Queue", id="scheduling-queue-tab"):
                    with Horizontal(id="scheduling-workbench"):
                        with Vertical(id="scheduling-list-pane"):
                            yield Static(
                                "Schedule Queue",
                                id="scheduling-list-title",
                                classes="scheduling-column-title",
                            )
                            yield Input(
                                placeholder="Filter: title, type, or status…",
                                id="scheduling-queue-filter",
                            )
                            yield DataTable(id="scheduling-task-table", cursor_type="row")
                            yield Static("", id="scheduling-pane-notice")
                        with Vertical(id="scheduling-detail-pane"):
                            yield TaskDetail(id="scheduling-task-detail")
                        with Vertical(id="scheduling-inspector-pane"):
                            yield TaskInspector(id="scheduling-task-inspector")
                with TabPane("Conflicts", id="scheduling-conflicts-tab"):
                    yield ConflictsTab(
                        id="scheduling-conflicts",
                        sync_engine=service.sync_engine if service else None,
                    )

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
        table = self.query_one("#scheduling-task-table", DataTable)
        table.add_columns("Title", "Type", "Status", "Next Run")
        # task-23111 review F9: the relative next-run column ("in 25m")
        # is render-time text; refresh it periodically while visible.
        self._next_run_refresh_timer = self.set_interval(
            NEXT_RUN_REFRESH_SECONDS, self._refresh_next_run_rendering
        )
        self.run_worker(
            self.load_tasks,
            exclusive=True,
            group="schedules-load-tasks",
        )  # type: ignore[arg-type]

    def _refresh_next_run_rendering(self) -> None:
        """Re-render the queue so relative next-run text stays honest.

        Skips unless this screen is the top of the stack. (Textual's
        ``is_current`` also counts screens behind the top one --
        ``_background_screens`` always includes the screen directly
        beneath the top regardless of opacity -- so it cannot express
        "covered"; the suspend/resume handlers pause the timer while
        covered and refresh on uncover.) Also skips an empty queue:
        nothing to refresh, and the no-service path must keep its own
        detail-pane copy.
        """
        if self.app.screen is not self or not self._visible_tasks:
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
        self._marked_ids.intersection_update(
            {task.id for task in self._tasks}
        )
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
            or (
                _was_missed_while_away(task)
                and "missed" in text
            )
        ]
        rows: list[tuple[str, str, Text, str]] = [
            (
                ("● " if task.id in self._marked_ids else "")
                + ("◇ " if _was_missed_while_away(task) else "")
                + task.title,
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
                self.query_one(
                    "#scheduling-task-detail-empty-state", Static
                ).update(
                    f"No tasks match '{self._filter_text.strip()}'. "
                    "Clear the filter to see the queue."
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
        self.query_one("#scheduling-task-detail", TaskDetail).set_task(task)
        self.query_one("#scheduling-task-inspector", TaskInspector).set_task(task)

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
            await self.load_tasks()

        self.run_worker(
            _delete_and_refresh,
            exclusive=True,
            group="schedules-delete-task",
        )  # type: ignore[arg-type]

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

    def action_create_reminder(self) -> None:
        """Open the create-reminder form."""
        self.app.push_screen(
            ReminderForm(known_timezones=self._task_timezones()),
            callback=self._on_reminder_form_result,
        )

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

        async def _save_and_refresh() -> None:
            try:
                if task_id is None:
                    await service.create_reminder(form_data)
                    self.app_instance.notify(
                        "Scheduled task created.", severity="information"
                    )
                else:
                    await service.update_reminder(task_id, form_data)
                    self.app_instance.notify(
                        "Scheduled task updated.", severity="information"
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to save reminder")
                self.app_instance.notify(
                    "Failed to save the scheduled task. Check the form values and try again.",
                    severity="error",
                )
            await self.load_tasks()

        self.run_worker(
            _save_and_refresh,
            exclusive=True,
            group="schedules-save-reminder",
        )  # type: ignore[arg-type]

    @on(EditTaskRequested)
    def _on_edit_task_requested(self, event: EditTaskRequested) -> None:
        """Open the reminder form pre-filled for editing."""
        event.stop()
        self.app.push_screen(
            ReminderForm(event.task, known_timezones=self._task_timezones()),
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
        """Run the highlighted reminder immediately (``r`` key)."""
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
            await self.load_tasks()

        self.run_worker(
            _run_and_refresh,
            exclusive=True,
            group="schedules-run-reminder-now",
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
            await self.load_tasks()

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
            hidden_note = (
                f" ({hidden} hidden by the filter)" if hidden else ""
            )
            parts.append(
                f"{marked_count} marked{hidden_note} — space toggles all "
                "· d deletes all · esc clears"
            )
        if any(_was_missed_while_away(task) for task in self._visible_tasks):
            parts.append("◇ = ran late (dispatched after its scheduled time)")
        notice.update("\n".join(parts))

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
        self.run_worker(self.load_tasks, exclusive=True, group="schedules-load-tasks")
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
        self.run_worker(self.load_tasks, exclusive=True, group="schedules-load-tasks")
        self._refresh_conflicts_tab()

    @on(SyncFailed)
    def _on_sync_failed(self, event: SyncFailed) -> None:
        self._sync_running = False
        self.app_instance.notify(f"Sync failed: {event.error}", severity="error")
        self._refresh_owner_select()
        self.run_worker(self.load_tasks, exclusive=True, group="schedules-load-tasks")
        self._refresh_conflicts_tab()

    @on(ConflictsTab.ConflictResolved)
    def _on_conflict_resolved(self, event: ConflictsTab.ConflictResolved) -> None:
        self.run_worker(self.load_tasks, exclusive=True, group="schedules-load-tasks")
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
            pane.label = (
                f"Conflicts ({len(conflicts)})" if conflicts else "Conflicts"
            )
        except Exception:  # noqa: BLE001 - pane not mounted
            pass

    def action_delete(self) -> None:
        """Delete marked tasks in bulk, else the selected one (confirmed).

        While ANY mark exists, d never falls through to the highlighted,
        unmarked row (task-23107 review F1): acting on a row the user
        never marked is worse than refusing.
        """
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
                    await service.delete_reminder(task.id)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to delete reminder {}", task.id)
                    errors += 1
            count = len(marked) - errors
            self.app_instance.notify(
                f"Deleted {count} marked task{'s' if count != 1 else ''}"
                + (f" ({errors} failed)" if errors else "") + ".",
                severity="information" if not errors else "warning",
            )
            self._marked_ids.clear()
            await self.load_tasks()

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
        """Open the highlighted task in the edit form (e key)."""
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
                    await service.update_reminder(
                        task.id, {"enabled": not task.enabled}
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to toggle reminder {}", task.id)
                    errors += 1
            count = len(marked) - errors
            self.app_instance.notify(
                f"Toggled {count} marked task{'s' if count != 1 else ''}"
                + (f" ({errors} failed)" if errors else "") + ".",
                severity="information" if not errors else "warning",
            )
            self._marked_ids.clear()
            await self.load_tasks()

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
                    SyncFailed(owner_id, getattr(outcome, "error", None) or "sync error")
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
