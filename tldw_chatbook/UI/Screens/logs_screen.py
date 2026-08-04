"""Logs screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Logs_Window import LogsWindow
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class LogsScreen(BaseAppScreen):
    """
    Logs screen wrapper.
    """

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "logs", **kwargs)
        self.logs_window = None

    def compose_content(self) -> ComposeResult:
        """Compose the logs window content with its destination header."""
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Logs",
                subtitle="Application logs and diagnostics.",
                status="ready",
                status_label="Listening",
            ),
            id="logs-destination-header",
        )
        self.logs_window = LogsWindow(self.app_instance, classes="window", id="logs-window")
        # Leave room for the destination header above the window.
        self.logs_window.styles.height = "1fr"
        yield self.logs_window

    def on_mount(self) -> None:
        """Route live log records through the rebuilt LogsWindow."""
        super().on_mount()
        self.register_footer_shortcuts(
            source="logs", shortcuts=LogsWindow.LOGS_SHORTCUTS
        )
        self.app_instance._current_logs_window = self.logs_window
        try:
            self.logs_window.load_from_app()
        except Exception as e:
            from loguru import logger

            logger.error(f"Failed to load buffered logs: {e}")

    def on_unmount(self) -> None:
        """When the logs screen is unmounted, clear the widget reference."""
        super().on_unmount()
        if hasattr(self.app_instance, "_current_logs_window"):
            self.app_instance._current_logs_window = None
        # Clear the current log widget reference
        if hasattr(self.app_instance, "_current_log_widget"):
            self.app_instance._current_log_widget = None

    def save_state(self):
        """Save logs window state."""
        state = super().save_state()
        # Add any logs-specific state here
        return state

    def restore_state(self, state):
        """Restore logs window state."""
        super().restore_state(state)
        # Restore any logs-specific state here

    # Copy buttons are handled inside LogsWindow itself (copy-visible from
    # the filtered structured records, copy-all from the unbounded session
    # buffer) — one widget, one clipboard path.
