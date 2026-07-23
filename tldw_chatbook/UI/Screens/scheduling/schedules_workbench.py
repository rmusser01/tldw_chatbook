"""Schedules workbench shell for run timing, triggers, and recovery."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Static, TabbedContent, TabPane

from ...Navigation.base_app_screen import BaseAppScreen
from ....runtime_policy.bootstrap import set_authoritative_runtime_source
from ....Scheduling.events import (
    DeleteTaskRequested,
    DisableTaskRequested,
    EditTaskRequested,
    EnableTaskRequested,
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
    _task_status,
    _task_type_label,
    status_badge_text,
)

if TYPE_CHECKING:
    from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
    from tldw_chatbook.app import TldwCli


logger = logger.bind(module="SchedulesWorkbench")


class SchedulesWorkbench(BaseAppScreen):
    """Main workbench for managing scheduled runs, reminders, and jobs."""

    BINDINGS = [
        Binding("ctrl+c", "create_reminder", "Create"),
        Binding("ctrl+r", "run_now", "Run now"),
        Binding("ctrl+p", "pause_resume", "Pause/Resume"),
        Binding("ctrl+d", "delete", "Delete"),
        Binding("ctrl+s", "sync_now", "Sync"),
    ]

    SCHEDULES_SHORTCUTS = (
        ("ctrl+c", "create reminder"),
        ("ctrl+r", "run now"),
        ("ctrl+p", "pause/resume"),
        ("ctrl+d", "delete"),
        ("ctrl+s", "sync now"),
    )

    def __init__(
        self, app_instance: "TldwCli", screen_name: str = "schedules", **kwargs
    ):
        super().__init__(app_instance, screen_name, **kwargs)
        self._scheduling_service = getattr(app_instance, "scheduling_service", None)
        self._tasks: list[ReminderTask | ScheduledTask] = []
        self._sync_running = False
        self._current_console_follow_item = None
        self._latest_console_follow_item_id: str | None = None
        self._latest_console_launch_kwargs: dict[str, Any] | None = None
        self._latest_console_context_loaded = False

    def _active_server_id(self) -> str | None:
        runtime_state = getattr(
            getattr(self.app_instance, "runtime_policy", None), "state", None
        )
        return getattr(runtime_state, "active_server_id", None)

    def compose_content(self) -> ComposeResult:
        """Build the three-pane scheduling workbench layout."""
        service = self._service()
        owner_id = service.owner_id if service else "local"
        active_server_id = self._active_server_id()
        server_available = (
            service is not None
            and service.server_client.notifications_service is not None
        )
        yield SyncStatusWidget(
            id="scheduling-sync-status",
            current_owner=owner_id,
            active_server_id=active_server_id,
            server_available=server_available,
        )
        with TabbedContent():
            with TabPane("Queue", id="scheduling-queue-tab"):
                with Horizontal(id="scheduling-workbench"):
                    with Vertical(id="scheduling-list-pane"):
                        yield Static("Schedule Queue", id="scheduling-list-title")
                        yield DataTable(id="scheduling-task-table")
                    with Vertical(id="scheduling-detail-pane"):
                        yield TaskDetail(id="scheduling-task-detail")
                    with Vertical(id="scheduling-inspector-pane"):
                        yield TaskInspector(id="scheduling-task-inspector")
            with TabPane("Conflicts", id="scheduling-conflicts-tab"):
                yield ConflictsTab(id="scheduling-conflicts", sync_engine=service.sync_engine if service else None)

    def _service(self) -> "SchedulingService | None":
        """Return the app's scheduling service, if available."""
        return self._scheduling_service

    def _register_footer_shortcuts(self) -> None:
        """Register Scheduling shortcuts via BaseAppScreen's persisting API."""
        self.register_footer_shortcuts(
            source="schedules", shortcuts=self.SCHEDULES_SHORTCUTS
        )

    def on_mount(self) -> None:
        super().on_mount()
        self._register_footer_shortcuts()
        self._refresh_owner_select()
        table = self.query_one("#scheduling-task-table", DataTable)
        table.add_columns("Title", "Type", "Status", "Next Run")
        self.run_worker(self.load_tasks, exclusive=True)  # type: ignore[arg-type]

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

        rows: list[tuple[str, str, Text, str]] = [
            (
                task.title,
                _task_type_label(task),
                status_badge_text(_task_status(task)),
                _format_next_run(task),
            )
            for task in self._tasks
        ]

        table = self.query_one("#scheduling-task-table", DataTable)
        table.clear()
        for row in rows:
            table.add_row(*row)

        if rows:
            self._update_detail_for_index(0)
        else:
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=True
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)

        await self._refresh_console_context()

    @on(DataTable.RowHighlighted)
    def _on_task_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update the detail pane when the user highlights a task row."""
        self._update_detail_for_index(event.cursor_row)

    def _update_detail_for_index(self, index: int) -> None:
        """Render task details in the detail and inspector panes."""
        if not (0 <= index < len(self._tasks)):
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=not self._tasks
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            return

        task = self._tasks[index]
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
            has_recent_work = bool(getattr(self.app_instance, "_screen_states", {}))
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

        self.run_worker(_delete_and_refresh, exclusive=True)  # type: ignore[arg-type]

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

    def action_create_reminder(self) -> None:
        """Open the create-reminder form."""
        self.app.push_screen(ReminderForm(), callback=self._on_reminder_form_result)

    def _on_reminder_form_result(
        self, form_data: dict[str, Any] | None, task_id: str | None = None
    ) -> None:
        """Create or update a reminder from the form and refresh the queue."""
        if form_data is None:
            return

        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot save reminder.",
                severity="warning",
            )
            return

        async def _save_and_refresh() -> None:
            try:
                if task_id is None:
                    await service.create_reminder(form_data)
                    self.app_instance.notify("Reminder created.", severity="information")
                else:
                    await service.update_reminder(task_id, form_data)
                    self.app_instance.notify("Reminder updated.", severity="information")
            except Exception:  # noqa: BLE001
                logger.exception("Failed to save reminder")
                self.app_instance.notify(
                    "Failed to save reminder. Check the form values and try again.",
                    severity="error",
                )
            await self.load_tasks()

        self.run_worker(_save_and_refresh, exclusive=True)  # type: ignore[arg-type]

    @on(EditTaskRequested)
    def _on_edit_task_requested(self, event: EditTaskRequested) -> None:
        """Open the reminder form pre-filled for editing."""
        event.stop()
        self.app.push_screen(
            ReminderForm(event.task),
            callback=lambda result: self._on_reminder_form_result(result, event.task.id),
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

    def _set_reminder_enabled(self, task: ReminderTask, enabled: bool) -> None:
        """Update a reminder's enabled state and refresh the queue."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot update reminder.",
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

        self.run_worker(_update_and_refresh, exclusive=True)  # type: ignore[arg-type]

    def _refresh_owner_select(self) -> None:
        status = self.query_one("#scheduling-sync-status", SyncStatusWidget)
        service = self._service()
        if service is None:
            status.set_owner_state("local", None, False)
            status.update_status(None, None, [])
            return
        active_server_id = self._active_server_id()
        server_available = service.server_client.notifications_service is not None
        status.set_owner_state(
            service.owner_id, active_server_id, server_available
        )
        state = service.db.get_sync_state(service.owner_id) or {}
        status.update_status(
            last_pull_at=state.get("last_pull_at"),
            last_push_at=state.get("last_push_at"),
            sync_errors=state.get("sync_errors") or [],
        )

    @on(Button.Pressed, "#scheduling-owner-local")
    def _on_owner_local(self) -> None:
        self._set_owner("local")

    @on(Button.Pressed, "#scheduling-owner-server")
    def _on_owner_server(self) -> None:
        service = self._service()
        if service is None:
            return
        active_server_id = self._active_server_id()
        if active_server_id is None or service.server_client.notifications_service is None:
            self.app_instance.notify("No server connection", severity="warning")
            return
        self._set_owner(f"server:{active_server_id}")

    def _set_owner(self, new_owner: str) -> None:
        service = self._service()
        if service is None:
            return
        service.set_owner(new_owner)
        runtime_source = "server" if new_owner.startswith("server:") else "local"
        set_authoritative_runtime_source(self.app_instance, runtime_source)
        self._refresh_owner_select()
        self.run_worker(self.load_tasks, exclusive=True)
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
        self.app_instance.notify("Sync completed.", severity="information")
        self._refresh_owner_select()
        self.run_worker(self.load_tasks, exclusive=True)
        self._refresh_conflicts_tab()

    @on(SyncFailed)
    def _on_sync_failed(self, event: SyncFailed) -> None:
        self._sync_running = False
        self.app_instance.notify(f"Sync failed: {event.error}", severity="error")
        self._refresh_owner_select()
        self.run_worker(self.load_tasks, exclusive=True)
        self._refresh_conflicts_tab()

    @on(ConflictsTab.ConflictResolved)
    def _on_conflict_resolved(self, event: ConflictsTab.ConflictResolved) -> None:
        self.run_worker(self.load_tasks, exclusive=True)
        self._refresh_conflicts_tab()

    def _refresh_conflicts_tab(self) -> None:
        service = self._service()
        if service is None:
            return
        conflicts_tab = self.query_one("#scheduling-conflicts", ConflictsTab)
        conflicts = service.db.get_conflicts(service.owner_id, primitive="reminder_task")
        conflicts_tab.populate(conflicts)

    def action_run_now(self) -> None:
        """Run the selected schedule immediately (stub for later tasks)."""
        self.app_instance.notify("Not yet available", severity="warning")

    def action_pause_resume(self) -> None:
        """Pause or resume the selected schedule (stub for later tasks)."""
        self.app_instance.notify("Not yet available", severity="warning")

    def action_delete(self) -> None:
        """Delete the selected schedule after confirmation."""
        self.query_one("#scheduling-task-detail", TaskDetail).request_delete()

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
        self._sync_running = True
        self.run_worker(self._run_sync, exclusive=True)

    async def _run_sync(self) -> None:
        service = self._service()
        if service is None:
            self._sync_running = False
            return
        for btn_id in ("#scheduling-owner-local", "#scheduling-owner-server"):
            self.query_one(btn_id, Button).disabled = True
        try:
            owner_id = service.owner_id
            await service.sync_now(owner_id)
            conflicts = service.db.get_conflicts(owner_id, primitive="reminder_task")
            self.post_message(SyncCompleted(owner_id, conflict_count=len(conflicts)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sync failed")
            self.post_message(SyncFailed(service.owner_id, str(exc)))
        finally:
            for btn_id in ("#scheduling-owner-local", "#scheduling-owner-server"):
                self.query_one(btn_id, Button).disabled = False
            self._refresh_owner_select()
            self._sync_running = False
