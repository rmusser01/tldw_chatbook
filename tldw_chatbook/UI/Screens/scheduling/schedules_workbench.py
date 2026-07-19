"""Schedules workbench shell for run timing, triggers, and recovery."""

from __future__ import annotations

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Static

from ...Navigation.base_app_screen import BaseAppScreen
from ....Scheduling.models import ReminderTask
from .task_detail import TaskDetail, TaskInspector, _format_next_run


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

    def __init__(self, app_instance, screen_name: str = "schedules", **kwargs):
        super().__init__(app_instance, screen_name, **kwargs)
        self._scheduling_service = getattr(app_instance, "scheduling_service", None)
        self._tasks: list[ReminderTask] = []

    def compose_content(self) -> ComposeResult:
        """Build the three-pane scheduling workbench layout."""
        with Horizontal(id="scheduling-workbench"):
            with Vertical(id="scheduling-list-pane"):
                yield Static("Schedule Queue", id="scheduling-list-title")
                yield DataTable(id="scheduling-task-table")
            with Vertical(id="scheduling-detail-pane"):
                yield TaskDetail(id="scheduling-task-detail")
            with Vertical(id="scheduling-inspector-pane"):
                yield TaskInspector(id="scheduling-task-inspector")

    def _service(self):
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
        table = self.query_one("#scheduling-task-table", DataTable)
        table.add_columns("Title", "Kind", "Status", "Next Run")
        self.run_worker(self.load_tasks, exclusive=True)

    async def load_tasks(self) -> None:
        """Fetch reminders from the scheduling service and populate the table."""
        service = self._scheduling_service
        if service is None:
            logger.debug("No scheduling_service available; cannot load tasks")
            return

        try:
            tasks = await service.list_reminders()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load reminders")
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(None)
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            return

        self._tasks = list(tasks)

        rows: list[tuple[str, str, str, str]] = [
            (
                task.title,
                task.schedule_kind.value,
                task.last_status.value,
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
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(None)
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)

    @on(DataTable.RowHighlighted)
    def _on_task_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update the detail pane when the user highlights a task row."""
        self._update_detail_for_index(event.cursor_row)

    def _update_detail_for_index(self, index: int) -> None:
        """Render task details in the detail and inspector panes."""
        if not (0 <= index < len(self._tasks)):
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(None)
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            return

        task = self._tasks[index]
        self.query_one("#scheduling-task-detail", TaskDetail).set_task(task)
        self.query_one("#scheduling-task-inspector", TaskInspector).set_task(task)

    def action_create_reminder(self) -> None:
        """Create a new reminder (stub for Task 4.4+)."""
        logger.debug("action_create_reminder invoked")
        if self._service() is None:
            logger.debug("No scheduling_service available; create_reminder is a no-op")

    def action_run_now(self) -> None:
        """Run the selected schedule immediately (stub for later tasks)."""
        logger.debug("action_run_now invoked")

    def action_pause_resume(self) -> None:
        """Pause or resume the selected schedule (stub for later tasks)."""
        logger.debug("action_pause_resume invoked")

    def action_delete(self) -> None:
        """Delete the selected schedule (stub for later tasks)."""
        logger.debug("action_delete invoked")

    def action_sync_now(self) -> None:
        """Sync schedule state now (stub for later tasks)."""
        logger.debug("action_sync_now invoked")
