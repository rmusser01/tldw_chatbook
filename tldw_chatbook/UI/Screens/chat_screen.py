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
        Handle button events at the screen level.
        This ensures buttons work properly with screen-based navigation.
        """
        button_id = event.button.id
        
        # Log for debugging
        logger.debug(f"ChatScreen handling button: {button_id}")
        
        # Handle sidebar toggle buttons directly at screen level
        if button_id in ["toggle-chat-left-sidebar", "toggle-chat-right-sidebar"]:
            # These buttons affect app-level state, so we need to handle them specially
            await self._handle_sidebar_toggle(button_id)
            event.stop()
            return
            
        # For all other buttons in the chat window, delegate to ChatWindowEnhanced
        if self.chat_window:
            # The chat window knows how to handle its own buttons
            await self.chat_window.on_button_pressed(event)
            event.stop()  # Prevent bubbling to app level
    
    async def _handle_sidebar_toggle(self, button_id: str) -> None:
        """Handle sidebar toggle buttons."""
        # Access the app instance to toggle the sidebars
        if button_id == "toggle-chat-left-sidebar":
            self.app_instance.chat_sidebar_collapsed = not self.app_instance.chat_sidebar_collapsed
        elif button_id == "toggle-chat-right-sidebar":
            self.app_instance.chat_right_sidebar_collapsed = not self.app_instance.chat_right_sidebar_collapsed