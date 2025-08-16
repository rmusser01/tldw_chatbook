"""Notes screen implementation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.widgets import Button

from ..Navigation.base_app_screen import BaseAppScreen
from ..Notes_Window import NotesWindow
from ...Widgets.emoji_picker import EmojiSelected
from ...Event_Handlers.Audio_Events.dictation_integration_events import InsertDictationTextEvent

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
        self.notes_window = NotesWindow(self.app_instance, classes="window")
        # Yield the window widget directly
        yield self.notes_window
    
    def save_state(self):
        """Save notes window state."""
        state = super().save_state()
        # Add any notes-specific state here
        return state
    
    def restore_state(self, state):
        """Restore notes window state."""
        super().restore_state(state)
        # Restore any notes-specific state here
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Forward button events to the NotesWindow handler."""
        if self.notes_window:
            self.notes_window.on_button_pressed(event)
    
    def on_insert_dictation_text_event(self, event: InsertDictationTextEvent) -> None:
        """Forward dictation events to the NotesWindow."""
        if self.notes_window:
            self.notes_window.on_insert_dictation_text_event(event)
    
    def on_emoji_picker_emoji_selected(self, message: EmojiSelected) -> None:
        """Forward emoji selection events to the NotesWindow."""
        if self.notes_window:
            self.notes_window.on_emoji_picker_emoji_selected(message)
    
    async def on_mount(self) -> None:
        """Called when the screen is mounted."""
        super().on_mount()  # Don't await - parent's on_mount is not async
        # The notes window will be mounted automatically by Textual
        # No need to manually call on_mount