"""Tools & Settings screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Tools_Settings_Window import ToolsSettingsWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ToolsSettingsScreen(BaseAppScreen):
    """
    Tools & Settings screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "tools_settings", **kwargs)
        self.tools_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the tools settings window content."""
        self.tools_window = ToolsSettingsWindow(self.app_instance, classes="window")
        # Yield the window widget directly
        yield self.tools_window
    
    def save_state(self):
        """Save tools window state."""
        state = super().save_state()
        # Add any tools-specific state here
        return state
    
    def restore_state(self, state):
        """Restore tools window state."""
        super().restore_state(state)
        # Restore any tools-specific state here