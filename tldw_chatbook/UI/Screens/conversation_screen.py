"""Conversation/Character screen implementation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.widgets import Button, Input, Select, TextArea

from ..Navigation.base_app_screen import BaseAppScreen
from ..Conv_Char_Window import CCPWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ConversationScreen(BaseAppScreen):
    """
    Conversation/Character management screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "conversation", **kwargs)
        self.conv_char_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the conversation/character window content."""
        self.conv_char_window = CCPWindow(self.app_instance, classes="window")
        # Yield the window widget directly
        yield self.conv_char_window
    
    def save_state(self):
        """Save conversation window state."""
        state = super().save_state()
        # Add any conversation-specific state here
        return state
    
    def restore_state(self, state):
        """Restore conversation window state."""
        super().restore_state(state)
        # Restore any conversation-specific state here
    
    # Note: We don't override event handlers here because CCPWindow doesn't have its own handlers.
    # Events will naturally bubble up to the App where they are handled by the button_handler_map
    # and other event handlers. This follows Textual's event bubbling model.