"""Schedules workbench shell for run timing, triggers, and recovery."""

from __future__ import annotations

from loguru import logger
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from ...Navigation.base_app_screen import BaseAppScreen


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

    def compose_content(self) -> ComposeResult:
        """Build the three-pane scheduling workbench layout."""
        with Horizontal(id="scheduling-workbench"):
            with Vertical(id="scheduling-list-pane"):
                yield Static("Schedule Queue")
            with Vertical(id="scheduling-detail-pane"):
                yield Static("Run Detail")
            with Vertical(id="scheduling-inspector-pane"):
                yield Static("Status Inspector")

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
