"""Notes screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Notes_Window import NotesWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class NotesScreen(BaseAppScreen):
    """
    Notes management screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "notes", **kwargs)
        self.notes_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the notes window content."""
        self.notes_window = NotesWindow(self.app_instance)
        # Yield from the window's compose method instead of the container itself
        yield from self.notes_window.compose()
    
    def save_state(self):
        """Save notes window state."""
        state = super().save_state()
        # Add any notes-specific state here
        return state
    
    def restore_state(self, state):
        """Restore notes window state."""
        super().restore_state(state)
        # Restore any notes-specific state here