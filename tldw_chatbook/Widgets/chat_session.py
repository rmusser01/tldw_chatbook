# chat_session.py
# Description: Widget representing a single chat session within a tab
#
# Imports
import time
from typing import TYPE_CHECKING, Optional
#
# 3rd-Party Imports
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea, Static
from textual.reactive import reactive
#
# Local Imports
from ..Chat.chat_models import ChatSessionData
from ..Utils.Emoji_Handling import get_char, EMOJI_SEND, FALLBACK_SEND, EMOJI_STOP, FALLBACK_STOP
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Classes:

class ChatSession(Container):
    """
    A widget representing a single chat session.
    
    This contains the chat log, input area, and controls for a single conversation.
    It's designed to be used within a tabbed interface where multiple sessions
    can exist simultaneously.
    """
    
    # Session data
    session_data: reactive[ChatSessionData] = reactive(ChatSessionData(tab_id="default"))
    
    # UI state
    is_send_button: reactive[bool] = reactive(True)
    
    # Debouncing for button clicks
    _last_send_stop_click = 0
    DEBOUNCE_MS = 300
    
    def __init__(self, app_instance: 'TldwCli', session_data: ChatSessionData, **kwargs):
        """
        Initialize a chat session.
        
        Args:
            app_instance: Reference to the main TldwCli app
            session_data: The session data for this chat
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.session_data = session_data
        logger.debug(f"ChatSession initialized for tab: {session_data.tab_id}")
    
    def compose(self) -> ComposeResult:
        """Compose the chat session UI."""
        logger.debug(f"Composing ChatSession UI for tab: {self.session_data.tab_id}")
        
        # Chat log area
        yield VerticalScroll(id=f"chat-log-{self.session_data.tab_id}", classes="chat-log")
        
        # Image attachment indicator (for enhanced chat)
        if hasattr(self.app_instance, 'chat_enhanced_mode') and self.app_instance.chat_enhanced_mode:
            yield Static(
                "",
                id=f"image-attachment-indicator-{self.session_data.tab_id}",
                classes="image-attachment-indicator hidden"
            )
        
        # Input area
        with Horizontal(id=f"chat-input-area-{self.session_data.tab_id}", classes="chat-input-area"):
            yield TextArea(
                id=f"chat-input-{self.session_data.tab_id}", 
                classes="chat-input"
            )
            
            # Send/Stop button
            yield Button(
                get_char(EMOJI_SEND if self.is_send_button else EMOJI_STOP,
                        FALLBACK_SEND if self.is_send_button else FALLBACK_STOP),
                id=f"send-stop-chat-{self.session_data.tab_id}",
                classes="send-button",
                tooltip="Send message" if self.is_send_button else "Stop generation"
            )
            
            # Attach file button (if enhanced mode)
            if hasattr(self.app_instance, 'chat_enhanced_mode') and self.app_instance.chat_enhanced_mode:
                from ..config import get_cli_setting
                show_attach_button = get_cli_setting("chat.images", "show_attach_button", True)
                if show_attach_button:
                    yield Button(
                        "📎",
                        id=f"attach-image-{self.session_data.tab_id}",
                        classes="action-button attach-button",
                        tooltip="Attach file"
                    )
            
            # Suggest button
            yield Button(
                "💡",
                id=f"respond-for-me-button-{self.session_data.tab_id}",
                classes="action-button suggest-button",
                tooltip="Suggest a response"
            )
    
    async def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Set up periodic state checking for streaming
        self.set_interval(0.5, self._check_streaming_state)
        self._update_button_state()
    
    def _update_button_state(self) -> None:
        """Update the send/stop button based on streaming state."""
        is_streaming = self.session_data.is_streaming
        has_worker = (self.session_data.current_worker and 
                     self.session_data.current_worker.is_running)
        
        # Update button state
        self.is_send_button = not (is_streaming or has_worker)
        
        # Update button appearance
        try:
            button = self.query_one(f"#send-stop-chat-{self.session_data.tab_id}", Button)
            button.label = get_char(EMOJI_SEND if self.is_send_button else EMOJI_STOP,
                                  FALLBACK_SEND if self.is_send_button else FALLBACK_STOP)
            button.tooltip = "Send message" if self.is_send_button else "Stop generation"
            
            # Update button styling
            if self.is_send_button:
                button.remove_class("stop-state")
            else:
                button.add_class("stop-state")
        except Exception as e:
            logger.debug(f"Could not update button: {e}")
    
    def _check_streaming_state(self) -> None:
        """Periodically check streaming state and update button."""
        self._update_button_state()
    
    async def handle_send_stop_button(self, event):
        """Handle send/stop button press with debouncing."""
        from ..Event_Handlers.Chat_Events import chat_events_tabs
        
        current_time = time.time() * 1000
        
        # Debounce rapid clicks
        if current_time - self._last_send_stop_click < self.DEBOUNCE_MS:
            logger.debug("Button click debounced")
            return
        self._last_send_stop_click = current_time
        
        # Disable button during operation
        try:
            button = self.query_one(f"#send-stop-chat-{self.session_data.tab_id}", Button)
            button.disabled = True
        except Exception:
            pass
        
        try:
            # Check current state and route to appropriate handler
            if self.session_data.is_streaming or (
                self.session_data.current_worker and 
                self.session_data.current_worker.is_running
            ):
                # Stop operation
                logger.info(f"Send/Stop button pressed - stopping generation for tab {self.session_data.tab_id}")
                # Use tab-aware handler
                await chat_events_tabs.handle_stop_chat_generation_pressed_with_tabs(
                    self.app_instance, event, self.session_data
                )
            else:
                # Send operation
                logger.info(f"Send/Stop button pressed - sending message for tab {self.session_data.tab_id}")
                # Use tab-aware handler
                await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                    self.app_instance, event, self.session_data
                )
        finally:
            # Re-enable button and update state after operation
            try:
                button = self.query_one(f"#send-stop-chat-{self.session_data.tab_id}", Button)
                button.disabled = False
            except Exception:
                pass
            self._update_button_state()
    
    async def handle_suggest_button(self, event):
        """Handle suggest response button press."""
        from ..Event_Handlers.Chat_Events import chat_events_tabs
        logger.info(f"Suggest button pressed for tab {self.session_data.tab_id}")
        # Use tab-aware handler
        await chat_events_tabs.handle_respond_for_me_button_pressed_with_tabs(
            self.app_instance, event, self.session_data
        )
    
    async def handle_attach_button(self, event):
        """Handle file attachment button press."""
        logger.info(f"Attach button pressed for tab {self.session_data.tab_id}")
        # This will need to be implemented in the enhanced version
        self.app_instance.notify("File attachment coming soon for tabbed chat!", severity="information")
    
    def get_chat_input(self) -> TextArea:
        """Get the chat input TextArea for this session."""
        return self.query_one(f"#chat-input-{self.session_data.tab_id}", TextArea)
    
    def get_chat_log(self) -> VerticalScroll:
        """Get the chat log container for this session."""
        return self.query_one(f"#chat-log-{self.session_data.tab_id}", VerticalScroll)
    
    def clear_chat(self) -> None:
        """Clear the chat log for this session."""
        chat_log = self.get_chat_log()
        chat_log.remove_children()
        logger.info(f"Cleared chat log for tab {self.session_data.tab_id}")
    
    # Button event handlers
    @on(Button.Pressed)
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses within this session."""
        button_id = event.button.id
        
        if button_id == f"send-stop-chat-{self.session_data.tab_id}":
            await self.handle_send_stop_button(event)
        elif button_id == f"respond-for-me-button-{self.session_data.tab_id}":
            await self.handle_suggest_button(event)
        elif button_id == f"attach-image-{self.session_data.tab_id}":
            await self.handle_attach_button(event)

#
# End of chat_session.py
#######################################################################################################################