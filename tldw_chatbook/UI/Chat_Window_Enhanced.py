# Chat_Window_Enhanced.py
# Description: Enhanced Chat Window with image attachment support
#
# Imports
import asyncio
from typing import TYPE_CHECKING, Optional, Any, Dict
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea, Input, Static, Select
from textual.reactive import reactive
from textual import work, on
from textual.worker import Worker, get_current_worker, WorkerCancelled
from textual.css.query import NoMatches
#
# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
# Right sidebar removed - functionality moved to settings_sidebar
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from ..Widgets.voice_input_widget import VoiceInputWidget, VoiceInputMessage
from ..config import get_cli_setting
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP

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

#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################

#
# Functions:

class ChatWindowEnhanced(Container):
    """Enhanced Container for the Chat Tab's UI with image support.
    
    This container manages the chat interface following Textual best practices:
    - Uses Container as base (wrapped by ChatScreen which provides Screen functionality)
    - Implements reactive properties with proper validators
    - Uses @on decorators for clean event handling
    - Follows CSS separation of concerns
    - Implements proper worker thread safety
    """
    
    # Explicit CSS path declaration following best practices
    CSS_PATH = "css/features/_chat.tcss"
    
    BINDINGS = [
        ("ctrl+shift+left", "resize_sidebar_shrink", "Shrink sidebar"),
        ("ctrl+shift+right", "resize_sidebar_expand", "Expand sidebar"),
        ("ctrl+e", "edit_focused_message", "Edit focused message"),
        ("ctrl+m", "toggle_voice_input", "Toggle voice input"),
    ]
    
    # Reactive properties with proper type hints
    pending_image: reactive[Optional[Dict[str, Any]]] = reactive(None, layout=False)
    is_send_button: reactive[bool] = reactive(True, layout=False)
    
    # Cached widget references to avoid repeated queries
    _chat_input: Optional[TextArea] = None
    _send_button: Optional[Button] = None
    _attachment_indicator: Optional[Static] = None
    _tab_container: Optional['ChatTabContainer'] = None
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the chat window with modular handlers.
        
        Args:
            app_instance: Reference to the main application instance
            **kwargs: Additional keyword arguments for Container
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
        self.pending_attachment: Optional[Dict[str, Any]] = None
        
        # Voice input state (for compatibility)
        self.voice_input_widget: Optional[VoiceInputWidget] = None
        self.is_voice_recording = False
        
        logger.debug("ChatWindowEnhanced initialized with modular handlers")
    
    async def on_mount(self) -> None:
        """Handle post-composition setup.
        
        Configures widget visibility, caches widget references, and initializes UI state.
        """
        # Cache frequently accessed widgets to avoid repeated queries
        self._cache_widget_references()
        
        # Configure widget visibility based on settings
        await self._configure_widget_visibility()
        
        # Initialize button state
        self._update_button_state()
    
    # Message Handlers using Textual's Message System
    
    async def on_chat_input_message_send_requested(self, message: ChatInputMessage.SendRequested) -> None:
        """Handle send request via message system."""
        logger.debug(f"Send requested via message: {len(message.text)} chars, {len(message.attachments)} attachments")
        # Forward to input handler
        await self.input_handler.handle_enhanced_send_button(None)
    
    async def on_chat_input_message_stop_requested(self, message: ChatInputMessage.StopRequested) -> None:
        """Handle stop request via message system."""
        logger.debug("Stop requested via message")
        from ..Event_Handlers.Chat_Events import chat_events
        await chat_events.handle_stop_chat_generation_pressed(self.app_instance, None)
    
    async def on_chat_attachment_message_file_selected(self, message: ChatAttachmentMessage.FileSelected) -> None:
        """Handle file selection via message system."""
        logger.debug(f"File selected via message: {message.file_path}")
        await self.attachment_handler.process_file_attachment(str(message.file_path))
    
    async def on_chat_voice_message_transcript_received(self, message: ChatVoiceMessage.TranscriptReceived) -> None:
        """Handle voice transcript via message system."""
        logger.debug(f"Transcript received via message: {message.text} (final: {message.is_final})")
        if message.is_final:
            chat_input = self._get_chat_input()
            if chat_input:
                current = chat_input.value
                chat_input.value = current + (" " if current else "") + message.text
    
    async def on_chat_sidebar_message_sidebar_toggled(self, message: ChatSidebarMessage.SidebarToggled) -> None:
        """Handle sidebar toggle via message system."""
        logger.debug(f"Sidebar {message.sidebar_id} toggled to {message.visible}")
        self.sidebar_handler.toggle_sidebar_visibility(message.sidebar_id)
    
    async def on_chat_message_display_message_edit_requested(self, message: ChatMessageDisplayMessage.EditRequested) -> None:
        """Handle edit request via message system."""
        logger.debug(f"Edit requested for message {message.message_id}")
        await self.message_manager.edit_focused_message()
    
    async def on_chat_streaming_message_stream_started(self, message: ChatStreamingMessage.StreamStarted) -> None:
        """Handle stream start via message system."""
        logger.debug(f"Stream started for message {message.message_id}")
        self.is_send_button = False  # Switch to stop button
    
    async def on_chat_streaming_message_stream_completed(self, message: ChatStreamingMessage.StreamCompleted) -> None:
        """Handle stream completion via message system."""
        logger.debug(f"Stream completed for message {message.message_id}")
        self.is_send_button = True  # Switch back to send button
    
    def _cache_widget_references(self) -> None:
        """Cache frequently accessed widget references to optimize performance."""
        try:
            self._chat_input = self.query_one("#chat-input", TextArea)
        except NoMatches:
            self._chat_input = None
            
        try:
            self._send_button = self.query_one("#send-stop-chat", Button)
        except NoMatches:
            self._send_button = None
            
        try:
            self._attachment_indicator = self.query_one("#image-attachment-indicator", Static)
        except NoMatches:
            self._attachment_indicator = None
        
        if get_cli_setting("chat_defaults", "enable_tabs", False):
            try:
                self._tab_container = self.query_one(ChatTabContainer)
            except NoMatches:
                self._tab_container = None
    
    async def _configure_widget_visibility(self) -> None:
        """Configure visibility of optional widgets based on settings."""
        try:
            app = self.app
        except Exception:
            # App not available yet
            return
        
        with app.batch_update():
            # Hide mic button if disabled in settings
            if not get_cli_setting("chat.voice", "show_mic_button", True):
                try:
                    mic_button = self.query_one("#mic-button", Button)
                    mic_button.display = False
                except NoMatches:
                    pass  # Button doesn't exist, nothing to hide
            
            # Hide attach button if disabled in settings
            if not get_cli_setting("chat.images", "show_attach_button", True):
                try:
                    attach_button = self.query_one("#attach-image", Button)
                    attach_button.display = False
                except NoMatches:
                    pass  # Button doesn't exist, nothing to hide
    
    def _get_send_button(self) -> Optional[Button]:
        """Get the cached send/stop button widget.
        
        Returns:
            The send button widget or None if not found
        """
        return self._send_button
    
    def _get_chat_input(self) -> Optional[TextArea]:
        """Get the cached chat input widget.
        
        Returns:
            The chat input widget or None if not found
        """
        return self._chat_input
    
    def _get_attachment_indicator(self) -> Optional[Static]:
        """Get the cached attachment indicator widget.
        
        Returns:
            The attachment indicator widget or None if not found
        """
        return self._attachment_indicator
    
    def _get_tab_container(self) -> Optional['ChatTabContainer']:
        """Get the cached tab container if tabs are enabled.
        
        Returns:
            The tab container widget or None if not found
        """
        return self._tab_container
    
    def _get_chat_log(self) -> Optional[VerticalScroll]:
        """Get the chat log widget from the app instance.
        
        Returns:
            The chat log widget or None if not found
        """
        try:
            return self.app_instance.query_one("#chat-log", VerticalScroll)
        except NoMatches:
            return None

    # Event Handlers using @on decorators for cleaner code
    
    @on(Button.Pressed, "#send-stop-chat")
    async def handle_send_stop_button_press(self, event: Button.Pressed) -> None:
        """Handle send/stop button press.
        
        Args:
            event: The button press event
        """
        event.stop()  # Prevent bubbling
        await self.handle_send_stop_button(self.app_instance, event)
    
    @on(Button.Pressed, "#attach-image")
    async def handle_attach_image_press(self, event: Button.Pressed) -> None:
        """Handle image attachment button press.
        
        Args:
            event: The button press event
        """
        event.stop()
        await self.attachment_handler.handle_attach_image_button(event)
    
    @on(Button.Pressed, "#clear-image")
    async def handle_clear_image_press(self, event: Button.Pressed) -> None:
        """Handle clear image button press.
        
        Args:
            event: The button press event
        """
        event.stop()
        await self.attachment_handler.handle_clear_image_button(event)
    
    @on(Button.Pressed, "#mic-button")
    async def handle_mic_button_press(self, event: Button.Pressed) -> None:
        """Handle microphone button press.
        
        Args:
            event: The button press event
        """
        event.stop()
        await self.voice_handler.handle_mic_button(event)
    
    @on(Button.Pressed, ".chat-sidebar-toggle-button")
    async def handle_sidebar_toggle_press(self, event: Button.Pressed) -> None:
        """Handle sidebar toggle button press.
        
        Args:
            event: The button press event
        """
        from ..Event_Handlers.Chat_Events import chat_events
        await chat_events.handle_chat_tab_sidebar_toggle(self.app_instance, event)
    
    # Legacy button handler for buttons not yet migrated to @on decorators
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle remaining button press events.
        
        This method handles buttons that haven't been migrated to @on decorators yet.
        It will be removed once all buttons are migrated.
        
        Args:
            event: The button press event
        """
        button_id = event.button.id
        if not button_id:
            return

        logger.debug(f"Button pressed: {button_id}")
        
        # Skip buttons that are handled by @on decorators
        decorator_handled_buttons = {
            "send-stop-chat",
            "attach-image",
            "chat-mic",
            "clear-image"
        }
        if button_id in decorator_handled_buttons:
            # Already handled by @on decorator, skip
            return

        # Check for tab-specific buttons first
        if self._is_tab_specific_button(button_id):
            return  # Let the tab's session handle it
        
        # Route to appropriate handler based on button ID pattern
        if await self._handle_core_buttons(button_id, event):
            event.stop()
            return
            
        if await self._handle_sidebar_buttons(button_id, event):
            event.stop()
            return
            
        # Check if this is an app-level button that should bubble up
        if self._is_app_level_button(button_id):
            return
            
        logger.warning(f"No handler found for button: {button_id}")
    
    def _is_tab_specific_button(self, button_id: str) -> bool:
        """Check if this button belongs to a specific tab session."""
        tab_container = self._get_tab_container()
        if tab_container:
            # Tab-specific buttons have session IDs appended
            for session_id in tab_container.sessions.keys():
                if button_id.endswith(f"-{session_id}"):
                    logger.debug(f"Tab-specific button detected for session {session_id}")
                    return True
        return False
    
    def _is_app_level_button(self, button_id: str) -> bool:
        """Check if this button should be handled at app level."""
        app_level_buttons = {
            "chat-notes-search-button",
            "chat-notes-load-button",
            "chat-notes-create-button",
            "chat-notes-delete-button",
            "chat-notes-save-button"
        }
        
        # Navigation buttons are also handled at app level
        if button_id and button_id.startswith("nav-"):
            return True
            
        return button_id in app_level_buttons
    
    async def _handle_core_buttons(self, button_id: str, event: Button.Pressed) -> bool:
        """Handle core chat functionality buttons."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        # Use the comprehensive CHAT_BUTTON_HANDLERS from chat_events
        # This includes all button handlers for chat functionality
        if hasattr(chat_events, 'CHAT_BUTTON_HANDLERS'):
            if button_id in chat_events.CHAT_BUTTON_HANDLERS:
                logger.debug(f"Handling button via CHAT_BUTTON_HANDLERS: {button_id}")
                await chat_events.CHAT_BUTTON_HANDLERS[button_id](self.app_instance, event)
                return True
        
        # Fallback to individual handlers for backwards compatibility
        core_handlers = {
            # "send-stop-chat" is handled by @on decorator, removed to avoid duplicate handling
            "chat-new-conversation-button": chat_events.handle_chat_new_conversation_button_pressed,
            "chat-new-temp-chat-button": chat_events.handle_chat_new_temp_chat_button_pressed,
            "chat-save-current-chat-button": chat_events.handle_chat_save_current_chat_button_pressed,
            "chat-clone-current-chat-button": chat_events.handle_chat_clone_current_chat_button_pressed,
            "chat-save-conversation-details-button": chat_events.handle_chat_save_details_button_pressed,
            "chat-convert-to-note-button": chat_events.handle_chat_convert_to_note_button_pressed,
            "chat-conversation-load-selected-button": chat_events.handle_chat_load_selected_button_pressed,
            "chat-apply-template-button": chat_events.handle_chat_apply_template_button_pressed,
        }
        
        if button_id in core_handlers:
            logger.debug(f"Handling core button: {button_id}")
            await core_handlers[button_id](self.app_instance, event)
            return True
        return False
    
    async def _handle_sidebar_buttons(self, button_id: str, event: Button.Pressed) -> bool:
        """Handle sidebar-related buttons."""
        from ..Event_Handlers.Chat_Events import chat_events
        from ..Event_Handlers.Chat_Events import chat_events_sidebar
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        
        # Sidebar toggles
        if button_id in ["toggle-chat-left-sidebar", "toggle-chat-right-sidebar"]:
            await chat_events.handle_chat_tab_sidebar_toggle(self.app_instance, event)
            return True
            
        # Additional sidebar-specific handlers
        sidebar_handlers = {
            "chat-notes-expand-button": self.handle_notes_expand_button,
            "chat-notes-search-button": chat_events.handle_chat_notes_search_button_pressed if hasattr(chat_events, 'handle_chat_notes_search_button_pressed') else None,
            "chat-notes-load-button": chat_events.handle_chat_notes_load_button_pressed if hasattr(chat_events, 'handle_chat_notes_load_button_pressed') else None,
            "chat-notes-create-new-button": chat_events.handle_chat_notes_create_new_button_pressed if hasattr(chat_events, 'handle_chat_notes_create_new_button_pressed') else None,
            "chat-notes-save-button": chat_events.handle_chat_notes_save_button_pressed if hasattr(chat_events, 'handle_chat_notes_save_button_pressed') else None,
            "chat-notes-copy-button": chat_events.handle_chat_notes_copy_button_pressed if hasattr(chat_events, 'handle_chat_notes_copy_button_pressed') else None,
        }
        
        if button_id in sidebar_handlers:
            logger.debug(f"Handling sidebar button: {button_id}")
            await sidebar_handlers[button_id](self.app_instance, event)
            return True
            
        # Check sidebar module handlers
        if button_id in chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS:
            await chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS[button_id](self.app_instance, event)
            return True
            
        if button_id in chat_events_sidebar_resize.CHAT_SIDEBAR_RESIZE_HANDLERS:
            await chat_events_sidebar_resize.CHAT_SIDEBAR_RESIZE_HANDLERS[button_id](self.app_instance, event)
            return True
            
        return False
    
    # Note: _handle_attachment_buttons removed as functionality moved to @on decorators

    async def handle_attach_image_button(self, app_instance, event) -> None:
        """Handle attach image button click.
        
        Args:
            app_instance: The app instance
            event: The button press event
        """
        await self.attachment_handler.handle_attach_image_button(event)
    
    async def handle_clear_image_button(self, app_instance, event) -> None:
        """Handle clear image button click.
        
        Args:
            app_instance: The app instance
            event: The button press event
        """
        await self.attachment_handler.handle_clear_image_button(event)

    async def handle_enhanced_send_button(self, app_instance, event) -> None:
        """Handle enhanced send button click.
        
        Args:
            app_instance: The app instance
            event: The button press event
        """
        await self.input_handler.handle_enhanced_send_button(event)

    async def process_file_attachment(self, file_path: str) -> None:
        """Process a file attachment.
        
        Args:
            file_path: Path to the file to attach
        """
        await self.attachment_handler.process_file_attachment(file_path)
    
    @work(exclusive=True, thread=True)
    async def handle_image_path_submitted(self, event):
        """Handle image path submission from file input field.
        
        This method is for backward compatibility with tests that expect
        the old file input field behavior. Uses proper thread safety.
        
        Args:
            event: The event containing the file path
        """
        worker = get_current_worker()
        
        if worker.is_cancelled:
            return
        
        from ..Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
        from ..Utils.path_validation import is_safe_path
        from pathlib import Path
        import os
        
        try:
            file_path = event.value
            if not file_path:
                return
            
            # Check for cancellation before validation
            if worker.is_cancelled:
                return
            
            # Validate the file path is safe
            if not is_safe_path(file_path, os.path.expanduser("~")):
                self.call_from_thread(
                    self.app_instance.notify,
                    "Error: File path is outside allowed directory",
                    severity="error"
                )
                return
            
            path = Path(file_path)
            
            # Validate file exists
            if not path.exists():
                self.call_from_thread(
                    self.app_instance.notify,
                    f"Error attaching image: Image file not found: {file_path}",
                    severity="error"
                )
                return
            
            # Check for cancellation before processing
            if worker.is_cancelled:
                return
            
            # Process the image
            try:
                image_data, mime_type = await ChatImageHandler.process_image_file(str(path))
                
                # Check for cancellation before updating UI
                if worker.is_cancelled:
                    return
                
                # Store the pending image using thread-safe method
                image_dict = {
                    'data': image_data,
                    'mime_type': mime_type,
                    'path': str(path)
                }
                
                self.call_from_thread(self._store_pending_image, image_dict)
                
                # Hide file input if it exists
                if hasattr(event, 'input') and event.input:
                    self.call_from_thread(
                        lambda: setattr(event.input.styles, 'display', 'none')
                    )
                
                # Notify user
                self.call_from_thread(
                    self.app_instance.notify,
                    f"Image attached: {path.name}"
                )
                
            except (IOError, OSError) as e:
                logger.error(f"Error reading image file: {e}")
                self.call_from_thread(
                    self.app_instance.notify,
                    f"Cannot read image: {e}",
                    severity="error"
                )
            except ValueError as e:
                logger.error(f"Invalid image data: {e}")
                self.call_from_thread(
                    self.app_instance.notify,
                    "Invalid image format",
                    severity="error"
                )
                
        except ValueError as e:
            logger.error(f"Invalid image path: {e}")
            self.call_from_thread(
                self.app_instance.notify,
                "Invalid file path",
                severity="error"
            )
    
    def _store_pending_image(self, image_data: Dict[str, Any]) -> None:
        """Store pending image data (thread-safe).
        
        Args:
            image_data: The processed image data dictionary
        """
        self.pending_image = image_data


    def compose(self) -> ComposeResult:
        """Compose the chat UI structure.
        
        Follows Textual best practices:
        - Doesn't read reactive properties during composition
        - Yields all widgets directly
        - Uses consistent structure regardless of config
        
        Yields:
            The widgets that make up the chat interface
        """
        logger.debug("Composing ChatWindowEnhanced UI")
        
        # Settings Sidebar (Left)
        yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)

        # Left sidebar toggle button
        yield Button(
            get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), 
            id="toggle-chat-left-sidebar",
            classes="chat-sidebar-toggle-button",
            tooltip="Toggle left sidebar (Ctrl+[)"
        )

        # Main Chat Content Area
        with Container(id="chat-main-content"):
            # Check if tabs are enabled
            enable_tabs = get_cli_setting("chat_defaults", "enable_tabs", False)
            
            if enable_tabs:
                logger.info("Chat tabs are enabled - using ChatTabContainer in enhanced mode")
                # Use the tab container for multiple sessions
                tab_container = ChatTabContainer(self.app_instance)
                tab_container.enhanced_mode = True  # Flag for enhanced features
                yield tab_container
            else:
                # Legacy single-session mode
                yield VerticalScroll(id="chat-log")
                
                # Image attachment indicator (always present, controlled via CSS)
                yield Static(
                    "",
                    id="image-attachment-indicator"
                )
                
                # Input area with all buttons (visibility controlled in on_mount)
                with Horizontal(id="chat-input-area"):
                    yield TextArea(id="chat-input", classes="chat-input")
                    
                    # Microphone button (visibility controlled via CSS/on_mount)
                    yield Button(
                        get_char("🎤", "⚫"),
                        id="mic-button",
                        classes="mic-button",
                        tooltip="Voice input (Ctrl+M)"
                    )
                    
                    # Send/Stop button (label updated via reactive watcher)
                    yield Button(
                        get_char(EMOJI_SEND, FALLBACK_SEND),  # Default to send
                        id="send-stop-chat",
                        classes="send-button",
                        tooltip="Send message"  # Default tooltip
                    )
                    
                    # Attach button (visibility controlled via CSS/on_mount)
                    yield Button(
                        "📎", 
                        id="attach-image", 
                        classes="action-button attach-button",
                        tooltip="Attach file"
                    )

        # No right sidebar - all functionality moved to left sidebar

    def get_pending_image(self) -> Optional[Dict[str, Any]]:
        """Get the pending image attachment data.
        
        Returns:
            The pending image data dictionary or None
        """
        return self.pending_image
    
    def get_pending_attachment(self) -> Optional[Dict[str, Any]]:
        """Get the pending attachment data.
        
        Returns:
            The pending attachment data dictionary or None
        """
        return self.pending_attachment
    
    def _clear_attachment_state(self) -> None:
        """Clear all attachment state."""
        self.pending_image = None
        self.pending_attachment = None
        self._update_attachment_ui()
    
    def _update_attachment_ui(self) -> None:
        """Update attachment indicator UI based on current state."""
        if self._attachment_indicator:
            if self.pending_image:
                from pathlib import Path
                path = Path(self.pending_image.get('path', ''))
                self._attachment_indicator.update(f"📎 {path.name}")
            else:
                self._attachment_indicator.update("")
    
    async def toggle_attach_button_visibility(self, show: bool) -> None:
        """Toggle the visibility of the attach file button.
        
        Args:
            show: True to show the button, False to hide it
        """
        try:
            attach_button = self.query_one("#attach-image", Button)
            if show:
                # Button already exists
                return
        except NoMatches:
            attach_button = None
        
        if show:
            # Find the input area and send button
            try:
                input_area = self.query_one("#chat-input-area", Horizontal)
                send_button = self.query_one("#send-stop-chat", Button)
            except NoMatches:
                logger.warning("Input area or send button not found")
                return
            
            # Create and mount the button after the send button
            new_button = Button(
                "📎", 
                id="attach-image", 
                classes="action-button attach-button",
                tooltip="Attach file"
            )
            await input_area.mount(new_button, after=send_button)
        else:
            # Remove the button if it exists
            if attach_button:
                await attach_button.remove()
                # Clear attachment state when hiding the button
                self._clear_attachment_state()
    
    
    async def handle_notes_expand_button(self, app, event) -> None:
        """Handle notes expand button click.
        
        Args:
            app: The app instance
            event: The button press event
        """
        await self.sidebar_handler.handle_notes_expand_button(event)
    
    async def action_resize_sidebar_shrink(self) -> None:
        """Shrink sidebar width (keyboard shortcut action)."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_shrink(self.app_instance, None)
    
    async def action_resize_sidebar_expand(self) -> None:
        """Expand sidebar width (keyboard shortcut action)."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_expand(self.app_instance, None)
    
    async def action_edit_focused_message(self) -> None:
        """Edit the currently focused message (keyboard shortcut action)."""
        await self.message_manager.edit_focused_message()
    
    def _update_button_state(self) -> None:
        """Update send/stop button state.
        
        Triggers the reactive watcher to update the button UI.
        """
        # Trigger reactive watcher by reassigning
        self.is_send_button = self.is_send_button
    
    def watch_is_send_button(self, is_send: bool) -> None:
        """React to button state changes.
        
        Args:
            is_send: True if button should show send, False for stop
        """
        if not self._send_button:
            logger.debug("Send button not found in watcher")
            return
        
        # Check if app is available (needed for tests)
        try:
            app = self.app
        except Exception:
            # App not available yet (during initialization or tests)
            return
        
        # Batch multiple button updates for performance
        with app.batch_update():
            self._send_button.label = get_char(
                EMOJI_SEND if is_send else EMOJI_STOP,
                FALLBACK_SEND if is_send else FALLBACK_STOP
            )
            self._send_button.tooltip = "Send message" if is_send else "Stop generation"
            
            # Update button styling
            if is_send:
                self._send_button.remove_class("stop-state")
            else:
                self._send_button.add_class("stop-state")
    
    def watch_pending_image(self, image_data: Optional[Dict[str, Any]]) -> None:
        """React to pending image changes.
        
        Args:
            image_data: The new pending image data
        """
        self._update_attachment_ui()
    
    def validate_pending_image(self, image_data: Any) -> Optional[Dict[str, Any]]:
        """Validate pending image data.
        
        Args:
            image_data: The image data to validate
            
        Returns:
            Validated image data dictionary or None if invalid
        """
        if image_data is not None and not isinstance(image_data, dict):
            logger.warning(f"Invalid pending_image type: {type(image_data)}")
            return None
        return image_data
    
    
    async def handle_send_stop_button(self, app_instance, event) -> None:
        """Handle send/stop button click.
        
        Args:
            app_instance: The app instance
            event: The button press event
        """
        if self.is_send_button:
            await self.input_handler.handle_enhanced_send_button(event)
        else:
            from ..Event_Handlers.Chat_Events import chat_events
            await chat_events.handle_stop_chat_generation_pressed(app_instance, event)
    
    async def handle_mic_button(self, app_instance, event: Button.Pressed) -> None:
        """Handle microphone button click.
        
        Args:
            app_instance: The app instance
            event: The button press event
        """
        await self.voice_handler.handle_mic_button(event)
    
    def action_toggle_voice_input(self) -> None:
        """Toggle voice input mode (keyboard shortcut action)."""
        self.voice_handler.toggle_voice_input()
        # Update local state for compatibility
        self.is_voice_recording = self.voice_handler.is_voice_recording
    
    def _insert_voice_text(self, text: str) -> None:
        """Insert voice text into chat input.
        
        Args:
            text: The text to insert
        """
        if not self._chat_input:
            logger.warning("Chat input widget not found for voice text")
            return
        
        try:
            app = self.app
        except Exception:
            # App not available yet
            return
        
        # Use batch update for multiple operations
        with app.batch_update():
            current_text = self._chat_input.text
            
            # Add space if there's existing text
            if current_text and not current_text.endswith(' '):
                text = ' ' + text
            
            # Append transcribed text
            self._chat_input.load_text(current_text + text)
            
            # Focus the input
            self._chat_input.focus()
    
    def on_voice_input_message(self, event: VoiceInputMessage) -> None:
        """Handle voice input messages.
        
        Args:
            event: The voice input message event
        """
        if event.is_final and event.text and self._chat_input:
            try:
                app = self.app
            except Exception:
                # App not available yet
                return
            
            with app.batch_update():
                current_text = self._chat_input.text
                separator = ' ' if current_text and not current_text.endswith(' ') else ''
                self._chat_input.load_text(current_text + separator + event.text)
                self._chat_input.focus()

#
# End of Chat_Window_Enhanced.py
#######################################################################################################################