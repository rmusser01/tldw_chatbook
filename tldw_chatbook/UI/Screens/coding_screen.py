"""Coding screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Coding_Window import CodingWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class CodingScreen(BaseAppScreen):
    """
    Coding screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "coding", **kwargs)
        self.coding_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the coding window content."""
        self.coding_window = CodingWindow(self.app_instance, classes="window")
        # Yield the window widget directly
        yield self.coding_window
    
    def save_state(self):
        """Save coding window state."""
        state = super().save_state()
        # Add any coding-specific state here
        return state
    
    def restore_state(self, state):
        """Restore coding window state."""
        super().restore_state(state)
        # Restore any coding-specific state here