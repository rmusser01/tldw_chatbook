"""Chat screen implementation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container

from ..Navigation.base_app_screen import BaseAppScreen

# Import the existing chat window to reuse its functionality
from ..Chat_Window_Enhanced import ChatWindowEnhanced

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ChatScreen(BaseAppScreen):
    """
    Chat screen that wraps the existing ChatWindowEnhanced functionality.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "chat", **kwargs)
        self.chat_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the chat content."""
        # Create and yield the chat window container
        self.chat_window = ChatWindowEnhanced(self.app_instance, id="chat-window", classes="window")
        yield self.chat_window
    
    def save_state(self):
        """Save chat state."""
        state = super().save_state()
        if self.chat_window:
            # Save relevant chat state
            state['conversation_id'] = getattr(self.chat_window, 'current_conversation_id', None)
            # Add more state as needed
        return state
    
    def restore_state(self, state):
        """Restore chat state."""
        super().restore_state(state)
        if self.chat_window and 'conversation_id' in state:
            # Restore conversation
            # self.chat_window.load_conversation(state['conversation_id'])
            pass