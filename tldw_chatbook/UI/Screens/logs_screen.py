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

    # Screen-level mirrors of LogsWindow.BINDINGS: widget bindings only fire
    # when focus is inside the window, so the advertised keys were dead from
    # the landed state (nav bar has initial focus). Both layers delegate to
    # the same actions; whichever is nearest the focus wins.
    BINDINGS = LogsWindow.BINDINGS

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "logs", **kwargs)
        self.logs_window = None

    def action_focus_filter(self) -> None:
        self.logs_window.action_focus_filter()

    def action_toggle_pause(self) -> None:
        self.logs_window.action_toggle_pause()

    def action_level(self, chip_id: str) -> None:
        self.logs_window.action_level(chip_id)

    def action_copy_visible(self) -> None:
        self.logs_window.action_copy_visible()

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
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.
        self.register_footer_shortcuts(
            source="logs", shortcuts=LogsWindow.LOGS_SHORTCUTS
        )
        self.app_instance._current_logs_window = self.logs_window
        try:
            self.logs_window.load_from_app()
        except Exception as e:
            from loguru import logger

            buffered = getattr(self.app_instance, "_log_records", None)
            logger.error(
                "Logs screen failed to load buffered log records on mount "
                f"(buffered_records={len(buffered) if buffered is not None else 'n/a'}): {e}"
            )

    def on_unmount(self) -> None:
        """When the logs screen is unmounted, clear the widget reference.

        No super().on_unmount(): the dispatcher already invokes
        BaseAppScreen.on_unmount separately for this Unmount event (TASK-31418).
        """
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
