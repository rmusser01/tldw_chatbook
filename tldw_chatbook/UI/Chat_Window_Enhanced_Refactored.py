# Chat_Window_Enhanced_Refactored.py
# Description: Enhanced Chat Window following Textual best practices
#
# Imports
import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea, Input, Static, Select
from textual.reactive import reactive
from textual.screen import Screen
from textual import work, on
from textual.worker import Worker, get_current_worker, WorkerCancelled
from textual.css.query import NoMatches

# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
from tldw_chatbook.Widgets.Chat_Widgets.chat_right_sidebar import create_chat_right_sidebar
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from ..Widgets.voice_input_widget import VoiceInputWidget, VoiceInputMessage
from ..config import get_cli_setting
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import (
    get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, 
    EMOJI_SEND, FALLBACK_SEND, EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON,
    EMOJI_STOP, FALLBACK_STOP
)

# Import modular handlers and messages
from .Chat_Modules import (
    ChatInputHandler,
    ChatAttachmentHandler,
    ChatVoiceHandler,
    ChatSidebarHandler,
    ChatMessageManager,
    ChatInputMessage,
    ChatAttachmentMessage,
    ChatVoiceMessage,
    ChatSidebarMessage,
    ChatMessageDisplayMessage,
    ChatStreamingMessage
)

# Configure logger with context
logger = logger.bind(module="Chat_Window_Enhanced")

if TYPE_CHECKING:
    from ..app import TldwCli


class ChatWindowEnhanced(Screen):
    """Enhanced Screen for the Chat Tab's UI with image support.
    
    This screen manages the chat interface following Textual best practices:
    - Uses Screen as base class for proper view management
    - Implements reactive properties with proper validators
    - Uses @on decorators for event handling
    - Follows CSS separation of concerns
    """
    
    # CSS Path - Explicitly declare the stylesheet
    CSS_PATH = "css/features/_chat.tcss"
    
    # Key bindings
    BINDINGS = [
        ("ctrl+shift+left", "resize_sidebar_shrink", "Shrink sidebar"),
        ("ctrl+shift+right", "resize_sidebar_expand", "Expand sidebar"),
        ("ctrl+e", "edit_focused_message", "Edit focused message"),
        ("ctrl+m", "toggle_voice_input", "Toggle voice input"),
    ]
    
    # Reactive properties with proper typing
    pending_image: reactive[Optional[dict]] = reactive(None, layout=False)
    is_send_button: reactive[bool] = reactive(True, layout=False)
    
    # Cached widget references to avoid repeated queries
    _chat_input: Optional[TextArea] = None
    _send_button: Optional[Button] = None
    _attachment_indicator: Optional[Static] = None
    _tab_container: Optional[ChatTabContainer] = None
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the chat window with modular handlers.
        
        Args:
            app_instance: Reference to the main application instance
            **kwargs: Additional keyword arguments for Screen
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
        # Initialize modular handlers
        self.input_handler = ChatInputHandler(self)
        self.attachment_handler = ChatAttachmentHandler(self)
        self.voice_handler = ChatVoiceHandler(self)
        self.sidebar_handler = ChatSidebarHandler(self)
        self.message_manager = ChatMessageManager(self)
        
        # Initialize attachment state
        self.pending_attachment = None
        
        # Voice input state
        self.voice_input_widget: Optional[VoiceInputWidget] = None
        self.is_voice_recording = False
        
        logger.debug("ChatWindowEnhanced initialized with modular handlers")
    
    async def on_mount(self) -> None:
        """Handle post-composition setup.
        
        Configures widget visibility and initializes UI state.
        """
        await self._configure_widget_visibility()
        self._cache_widget_references()
        self._update_button_state()
    
    def _cache_widget_references(self) -> None:
        """Cache frequently accessed widget references."""
        self._chat_input = self.query_one_or_none("#chat-input", TextArea)
        self._send_button = self.query_one_or_none("#send-stop-chat", Button)
        self._attachment_indicator = self.query_one_or_none("#image-attachment-indicator", Static)
        
        if get_cli_setting("chat_defaults", "enable_tabs", False):
            self._tab_container = self.query_one_or_none(ChatTabContainer)
    
    async def _configure_widget_visibility(self) -> None:
        """Configure visibility of optional widgets based on settings."""
        with self.app.batch_update():
            # Hide mic button if disabled
            if not get_cli_setting("chat.voice", "show_mic_button", True):
                mic_button = self.query_one_or_none("#mic-button", Button)
                if mic_button:
                    mic_button.display = False
            
            # Hide attach button if disabled
            if not get_cli_setting("chat.images", "show_attach_button", True):
                attach_button = self.query_one_or_none("#attach-image", Button)
                if attach_button:
                    attach_button.display = False
    
    # Event Handlers using @on decorators for cleaner code
    
    @on(Button.Pressed, "#send-stop-chat")
    async def handle_send_stop_button(self, event: Button.Pressed) -> None:
        """Handle send/stop button press with built-in throttling."""
        event.stop()  # Prevent bubbling
        
        if self.is_send_button:
            await self.input_handler.handle_enhanced_send_button(event)
        else:
            from ..Event_Handlers.Chat_Events import chat_events
            await chat_events.handle_stop_chat_generation_pressed(self.app_instance, event)
    
    @on(Button.Pressed, "#attach-image")
    async def handle_attach_image(self, event: Button.Pressed) -> None:
        """Handle image attachment button."""
        event.stop()
        await self.attachment_handler.handle_attach_image_button(event)
    
    @on(Button.Pressed, "#clear-image")
    async def handle_clear_image(self, event: Button.Pressed) -> None:
        """Handle clear image button."""
        event.stop()
        await self.attachment_handler.handle_clear_image_button(event)
    
    @on(Button.Pressed, "#mic-button")
    async def handle_mic_button(self, event: Button.Pressed) -> None:
        """Handle microphone button."""
        event.stop()
        await self.voice_handler.handle_mic_button(event)
    
    @on(Button.Pressed, ".chat-sidebar-toggle-button")
    async def handle_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Handle sidebar toggle buttons."""
        from ..Event_Handlers.Chat_Events import chat_events
        await chat_events.handle_chat_tab_sidebar_toggle(self.app_instance, event)
    
    # Core chat buttons
    @on(Button.Pressed, "#chat-new-conversation-button")
    async def handle_new_conversation(self, event: Button.Pressed) -> None:
        """Handle new conversation button."""
        from ..Event_Handlers.Chat_Events import chat_events
        await chat_events.handle_chat_new_conversation_button_pressed(self.app_instance, event)
    
    @on(Button.Pressed, "#chat-save-current-chat-button")
    async def handle_save_chat(self, event: Button.Pressed) -> None:
        """Handle save chat button."""
        from ..Event_Handlers.Chat_Events import chat_events
        await chat_events.handle_chat_save_current_chat_button_pressed(self.app_instance, event)
    
    # Message handlers for custom events
    
    async def on_chat_input_message_send_requested(self, message: ChatInputMessage.SendRequested) -> None:
        """Handle send request via message system."""
        logger.debug(f"Send requested: {len(message.text)} chars, {len(message.attachments)} attachments")
        await self.input_handler.handle_enhanced_send_button(None)
    
    async def on_chat_streaming_message_stream_started(self, message: ChatStreamingMessage.StreamStarted) -> None:
        """Handle stream start."""
        logger.debug(f"Stream started for message {message.message_id}")
        self.is_send_button = False
    
    async def on_chat_streaming_message_stream_completed(self, message: ChatStreamingMessage.StreamCompleted) -> None:
        """Handle stream completion."""
        logger.debug(f"Stream completed for message {message.message_id}")
        self.is_send_button = True
    
    async def on_voice_input_message(self, event: VoiceInputMessage) -> None:
        """Handle voice input messages."""
        if event.is_final and event.text and self._chat_input:
            with self.app.batch_update():
                current_text = self._chat_input.text
                separator = ' ' if current_text and not current_text.endswith(' ') else ''
                self._chat_input.load_text(current_text + separator + event.text)
                self._chat_input.focus()
    
    # Reactive property validators and watchers
    
    def validate_pending_image(self, image_data: Any) -> Optional[dict]:
        """Validate pending image data.
        
        Args:
            image_data: The image data to validate
            
        Returns:
            Validated image data or None if invalid
        """
        if image_data is not None and not isinstance(image_data, dict):
            logger.warning(f"Invalid pending_image type: {type(image_data)}")
            return None
        return image_data
    
    def watch_is_send_button(self, is_send: bool) -> None:
        """React to button state changes.
        
        Args:
            is_send: True if button should show send, False for stop
        """
        if not self._send_button:
            return
        
        with self.app.batch_update():
            self._send_button.label = get_char(
                EMOJI_SEND if is_send else EMOJI_STOP,
                FALLBACK_SEND if is_send else FALLBACK_STOP
            )
            self._send_button.tooltip = "Send message" if is_send else "Stop generation"
            
            if is_send:
                self._send_button.remove_class("stop-state")
            else:
                self._send_button.add_class("stop-state")
    
    def watch_pending_image(self, image_data: Optional[dict]) -> None:
        """React to pending image changes.
        
        Args:
            image_data: The new pending image data
        """
        self._update_attachment_ui()
    
    # Worker methods with proper thread safety
    
    @work(exclusive=True, thread=True)
    async def process_file_attachment(self, file_path: str) -> None:
        """Process file attachment in background thread.
        
        Args:
            file_path: Path to the file to attach
        """
        worker = get_current_worker()
        
        if worker.is_cancelled:
            return
        
        from ..Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
        from ..Utils.path_validation import is_safe_path
        import os
        
        try:
            # Validate path safety
            if not is_safe_path(file_path, os.path.expanduser("~")):
                self.call_from_thread(
                    self.app_instance.notify,
                    "Error: File path is outside allowed directory",
                    severity="error"
                )
                return
            
            path = Path(file_path)
            
            if not path.exists():
                self.call_from_thread(
                    self.app_instance.notify,
                    f"File not found: {file_path}",
                    severity="error"
                )
                return
            
            # Check for cancellation before processing
            if worker.is_cancelled:
                return
            
            # Process the image
            image_data, mime_type = await ChatImageHandler.process_image_file(str(path))
            
            # Update UI from thread
            self.call_from_thread(self._store_pending_image, {
                'data': image_data,
                'mime_type': mime_type,
                'path': str(path)
            })
            
            self.call_from_thread(
                self.app_instance.notify,
                f"Image attached: {path.name}"
            )
            
        except Exception as e:
            logger.error(f"Error processing attachment: {e}")
            self.call_from_thread(
                self.app_instance.notify,
                f"Error: {str(e)}",
                severity="error"
            )
    
    def _store_pending_image(self, image_data: dict) -> None:
        """Store pending image data (called from thread).
        
        Args:
            image_data: The processed image data
        """
        self.pending_image = image_data
    
    def _update_attachment_ui(self) -> None:
        """Update attachment indicator UI."""
        if self._attachment_indicator:
            if self.pending_image:
                path = Path(self.pending_image.get('path', ''))
                self._attachment_indicator.update(f"📎 {path.name}")
            else:
                self._attachment_indicator.update("")
    
    def _update_button_state(self) -> None:
        """Update send/stop button state."""
        # Trigger reactive watcher
        self.is_send_button = self.is_send_button
    
    # Actions for key bindings
    
    async def action_resize_sidebar_shrink(self) -> None:
        """Shrink sidebar width."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_shrink(self.app_instance, None)
    
    async def action_resize_sidebar_expand(self) -> None:
        """Expand sidebar width."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_expand(self.app_instance, None)
    
    async def action_edit_focused_message(self) -> None:
        """Edit the currently focused message."""
        await self.message_manager.edit_focused_message()
    
    def action_toggle_voice_input(self) -> None:
        """Toggle voice input mode."""
        self.voice_handler.toggle_voice_input()
        self.is_voice_recording = self.voice_handler.is_voice_recording
    
    # Composition
    
    def compose(self) -> ComposeResult:
        """Compose the chat UI structure.
        
        Yields:
            The widgets that make up the chat interface
        """
        logger.debug("Composing ChatWindowEnhanced UI")
        
        # Settings Sidebar (Left)
        yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)
        
        # Left sidebar toggle
        yield Button(
            get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
            id="toggle-chat-left-sidebar",
            classes="chat-sidebar-toggle-button",
            tooltip="Toggle left sidebar (Ctrl+[)"
        )
        
        # Main Chat Content
        with Container(id="chat-main-content"):
            if get_cli_setting("chat_defaults", "enable_tabs", False):
                logger.info("Chat tabs enabled - using ChatTabContainer")
                tab_container = ChatTabContainer(self.app_instance)
                tab_container.enhanced_mode = True
                yield tab_container
            else:
                # Single session mode
                yield VerticalScroll(id="chat-log")
                yield Static("", id="image-attachment-indicator")
                
                with Horizontal(id="chat-input-area"):
                    yield TextArea(id="chat-input", classes="chat-input")
                    yield Button(
                        get_char("🎤", "⚫"),
                        id="mic-button",
                        classes="mic-button",
                        tooltip="Voice input (Ctrl+M)"
                    )
                    yield Button(
                        get_char(EMOJI_SEND, FALLBACK_SEND),
                        id="send-stop-chat",
                        classes="send-button",
                        tooltip="Send message"
                    )
                    yield Button(
                        "📎",
                        id="attach-image",
                        classes="action-button attach-button",
                        tooltip="Attach file"
                    )
        
        # Right sidebar toggle
        yield Button(
            get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON),
            id="toggle-chat-right-sidebar",
            classes="chat-sidebar-toggle-button",
            tooltip="Toggle right sidebar (Ctrl+])"
        )
        
        # Character Details Sidebar (Right)
        yield from create_chat_right_sidebar(
            "chat",
            initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
        )
    
    # Public API methods
    
    def get_pending_image(self) -> Optional[dict]:
        """Get the pending image attachment data.
        
        Returns:
            The pending image data or None
        """
        return self.pending_image
    
    def get_pending_attachment(self) -> Optional[dict]:
        """Get the pending attachment data.
        
        Returns:
            The pending attachment data or None
        """
        return self.pending_attachment
    
    def clear_attachment_state(self) -> None:
        """Clear all attachment state."""
        self.pending_image = None
        self.pending_attachment = None
        self._update_attachment_ui()


# End of Chat_Window_Enhanced_Refactored.py