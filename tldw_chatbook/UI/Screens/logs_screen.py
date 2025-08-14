"""Logs screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Logs_Window import LogsWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class LogsScreen(BaseAppScreen):
    """
    Logs screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "logs", **kwargs)
        self.logs_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the logs window content."""
        self.logs_window = LogsWindow(self.app_instance)
        yield self.logs_window
    
    def save_state(self):
        """Save logs window state."""
        state = super().save_state()
        # Add any logs-specific state here
        return state
    
    def restore_state(self, state):
        """Restore logs window state."""
        super().restore_state(state)
        # Restore any logs-specific state here