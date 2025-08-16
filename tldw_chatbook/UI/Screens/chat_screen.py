"""Chat screen implementation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button
from textual.events import Key

from ..Navigation.base_app_screen import BaseAppScreen

# Import the existing chat window to reuse its functionality
from ..Chat_Window_Enhanced import ChatWindowEnhanced
from ...Widgets.voice_input_widget import VoiceInputMessage

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
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button events - forward to ChatWindowEnhanced if it needs them,
        otherwise let them bubble to the App.
        """
        # ChatWindowEnhanced handles its own buttons
        if self.chat_window and hasattr(self.chat_window, 'on_button_pressed'):
            # Check if this button belongs to the chat window
            button = event.button
            if button in self.chat_window.walk_children(Button):
                # This button is a child of chat_window, let it handle it
                await self.chat_window.on_button_pressed(event)
                event.stop()  # Stop bubbling since we handled it
                return
        # For all other buttons, let the event bubble up to the App
        # Do NOT stop the event - let it bubble naturally