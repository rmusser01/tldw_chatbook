"""Customize screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Customize_Window import CustomizeWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class CustomizeScreen(BaseAppScreen):
    """
    Customize screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "customize", **kwargs)
        self.customize_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the customize window content."""
        self.customize_window = CustomizeWindow(self.app_instance)
        # Yield the window widget directly
        yield self.customize_window
    
    def save_state(self):
        """Save customize window state."""
        state = super().save_state()
        # Add any customize-specific state here
        return state
    
    def restore_state(self, state):
        """Restore customize window state."""
        super().restore_state(state)
        # Restore any customize-specific state here