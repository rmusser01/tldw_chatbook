# Chat_Window_Enhanced.py
# Description: Enhanced Chat Window with image attachment support
#
# Imports
import asyncio
from typing import TYPE_CHECKING, Optional, Any
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea, Input, Static, Select
from textual.reactive import reactive
from textual import work
from textual.worker import Worker, get_current_worker, WorkerCancelled
from textual.css.query import NoMatches
#
# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
from tldw_chatbook.Widgets.Chat_Widgets.chat_right_sidebar import create_chat_right_sidebar
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
    """
    Enhanced Container for the Chat Tab's UI with image support.
    """
    
    BINDINGS = [
        ("ctrl+shift+left", "resize_sidebar_shrink", "Shrink sidebar"),
        ("ctrl+shift+right", "resize_sidebar_expand", "Expand sidebar"),
        ("ctrl+e", "edit_focused_message", "Edit focused message"),
        ("ctrl+m", "toggle_voice_input", "Toggle voice input"),
    ]
    
    # CSS moved to tldw_chatbook/css/features/_chat.tcss for better maintainability
    # The styles are automatically loaded by Textual from the CSS directory
    
    # Track pending image attachment with proper reactive pattern
    pending_image = reactive(None, layout=False, recompose=False)
    
    # Track button state for Send/Stop functionality with automatic UI updates
    is_send_button = reactive(True, layout=False, recompose=False)
    
    # Debouncing for button clicks
    _last_send_stop_click = 0
    DEBOUNCE_MS = 300
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
        # Initialize modular handlers
        self.input_handler = ChatInputHandler(self)
        self.attachment_handler = ChatAttachmentHandler(self)
        self.voice_handler = ChatVoiceHandler(self)
        self.sidebar_handler = ChatSidebarHandler(self)
        self.message_manager = ChatMessageManager(self)
        
        # Initialize attachment state
        self.pending_attachment = None  # New unified attachment system
        
        # Voice input state (for compatibility)
        self.voice_input_widget: Optional[VoiceInputWidget] = None
        self.is_voice_recording = False
        
        logger.debug("ChatWindowEnhanced initialized with modular handlers.")
    
    async def on_mount(self) -> None:
        """Called when the widget is mounted.
        
        Handles post-composition setup:
        - Configure visibility based on settings
        - Initialize button states
        """
        # Configure widget visibility based on settings
        await self._configure_widget_visibility()
        
        # Token counter will be initialized when tab is switched to chat
        # Watch for streaming state changes
        self._update_button_state()
        # Button state will be updated on-demand when streaming state actually changes
    
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
    
    async def _configure_widget_visibility(self) -> None:
        """Configure visibility of optional widgets based on settings."""
        # Use batch update for multiple DOM operations
        with self.app.batch_update():
            # Hide mic button if disabled in settings
            show_mic_button = get_cli_setting("chat.voice", "show_mic_button", True)
            if not show_mic_button:
                try:
                    mic_button = self.query_one("#mic-button", Button)
                    mic_button.display = False
                except NoMatches:
                    pass  # Button doesn't exist, nothing to hide
            
            # Hide attach button if disabled in settings
            show_attach_button = get_cli_setting("chat.images", "show_attach_button", True)
            if not show_attach_button:
                try:
                    attach_button = self.query_one("#attach-image", Button)
                    attach_button.display = False
                except NoMatches:
                    pass  # Button doesn't exist, nothing to hide
    
    def _get_send_button(self) -> Optional[Button]:
        """Get the send/stop button widget."""
        try:
            return self.query_one("#send-stop-chat", Button)
        except NoMatches:
            return None
    
    def _get_chat_input(self) -> Optional[TextArea]:
        """Get the chat input widget."""
        try:
            return self.query_one("#chat-input", TextArea)
        except NoMatches:
            return None
    
    def _get_attachment_indicator(self) -> Optional[Static]:
        """Get the attachment indicator widget."""
        try:
            return self.query_one("#image-attachment-indicator", Static)
        except NoMatches:
            return None
    
    def _get_tab_container(self) -> Optional['ChatTabContainer']:
        """Get the tab container if tabs are enabled."""
        enable_tabs = get_cli_setting("chat_defaults", "enable_tabs", False)
        if enable_tabs:
            try:
                from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
                return self.query_one(ChatTabContainer)
            except NoMatches:
                return None
        return None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button press events using Textual's event system.
        Delegates to specific handlers based on button ID patterns.
        """
        button_id = event.button.id
        if not button_id:
            logger.warning("Button pressed with no ID")
            return

        logger.debug(f"Button pressed: {button_id}")

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
            
        if await self._handle_attachment_buttons(button_id, event):
            event.stop()
            return
            
        # Check if this is an app-level button that should bubble up
        if self._is_app_level_button(button_id):
            # Let it bubble up to app level
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
        return button_id in app_level_buttons
    
    async def _handle_core_buttons(self, button_id: str, event: Button.Pressed) -> bool:
        """Handle core chat functionality buttons."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        core_handlers = {
            "send-stop-chat": self.handle_send_stop_button,
            "chat-new-conversation-button": chat_events.handle_chat_new_conversation_button_pressed,
            "chat-save-current-chat-button": chat_events.handle_chat_save_current_chat_button_pressed,
            "chat-save-conversation-details-button": chat_events.handle_chat_save_details_button_pressed,
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
            
        # Character and prompt buttons
        sidebar_handlers = {
            "chat-prompt-load-selected-button": chat_events.handle_chat_view_selected_prompt_button_pressed,
            "chat-prompt-copy-system-button": chat_events.handle_chat_copy_system_prompt_button_pressed,
            "chat-prompt-copy-user-button": chat_events.handle_chat_copy_user_prompt_button_pressed,
            "chat-load-character-button": chat_events.handle_chat_load_character_button_pressed,
            "chat-clear-active-character-button": chat_events.handle_chat_clear_active_character_button_pressed,
            "chat-notes-expand-button": self.handle_notes_expand_button,
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
    
    async def _handle_attachment_buttons(self, button_id: str, event: Button.Pressed) -> bool:
        """Handle attachment and voice input buttons."""
        attachment_handlers = {
            "attach-image": self.handle_attach_image_button,
            "clear-image": self.handle_clear_image_button,
            "mic-button": self.handle_mic_button,
        }
        
        if button_id in attachment_handlers:
            logger.debug(f"Handling attachment button: {button_id}")
            await attachment_handlers[button_id](self.app_instance, event)
            return True
        return False

    async def handle_attach_image_button(self, app_instance, event):
        """Delegate to attachment handler."""
        await self.attachment_handler.handle_attach_image_button(event)
    
    async def handle_clear_image_button(self, app_instance, event):
        """Delegate to attachment handler."""
        await self.attachment_handler.handle_clear_image_button(event)

    async def handle_enhanced_send_button(self, app_instance, event):
        """Delegate to input handler."""
        await self.input_handler.handle_enhanced_send_button(event)

    async def process_file_attachment(self, file_path: str) -> None:
        """Delegate to attachment handler."""
        await self.attachment_handler.process_file_attachment(file_path)
    
    @work(exclusive=True)
    async def handle_image_path_submitted(self, event):
        """Handle image path submission from file input field.
        
        This method is for backward compatibility with tests that expect
        the old file input field behavior.
        """
        from ..Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
        from ..Utils.path_validation import is_safe_path
        from pathlib import Path
        import os
        
        try:
            file_path = event.value
            if not file_path:
                return
            
            # Validate the file path is safe
            if not is_safe_path(file_path, os.path.expanduser("~")):
                self.app_instance.notify(
                    "Error: File path is outside allowed directory",
                    severity="error"
                )
                return
            
            path = Path(file_path)
            
            # Validate file exists
            if not path.exists():
                self.app_instance.notify(
                    f"Error attaching image: Image file not found: {file_path}",
                    severity="error"
                )
                return
            
            # Process the image
            try:
                image_data, mime_type = await ChatImageHandler.process_image_file(str(path))
                
                # Store the pending image
                self.pending_image = {
                    'data': image_data,
                    'mime_type': mime_type,
                    'path': str(path)
                }
                
                # Use centralized UI update
                self._update_attachment_ui()
                
                # Hide file input if it exists
                if hasattr(event, 'input') and event.input:
                    event.input.styles.display = "none"
                
                # Notify user
                self.app_instance.notify(f"Image attached: {path.name}")
                
            except (IOError, OSError) as e:
                logger.error(f"Error reading image file: {e}")
                self.app_instance.notify(f"Cannot read image: {e}", severity="error")
            except ValueError as e:
                logger.error(f"Invalid image data: {e}")
                self.app_instance.notify("Invalid image format", severity="error")
                self.app_instance.notify(
                    f"Error attaching image: {str(e)}",
                    severity="error"
                )
                
        except ValueError as e:
            logger.error(f"Invalid image path: {e}")
            self.app_instance.notify("Invalid file path", severity="error")
            self.app_instance.notify(
                f"Error processing image path: {e}",
                severity="error"
            )


    def compose(self) -> ComposeResult:
        """Compose the ChatWindowEnhanced UI structure.
        
        Following Textual best practices:
        - Don't read reactive properties during composition
        - Yield all widgets directly
        - Use consistent structure regardless of config
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

        # Right sidebar toggle button
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

    def get_pending_image(self) -> Optional[dict]:
        """Get the pending image attachment data."""
        return self.pending_image
    
    def get_pending_attachment(self) -> Optional[dict]:
        """Get the pending attachment data (new unified system)."""
        return self.pending_attachment
    
    def _clear_attachment_state(self):
        """Delegate to attachment handler."""
        self.attachment_handler.clear_attachment_state()
    
    def _update_attachment_ui(self):
        """Delegate to attachment handler."""
        self.attachment_handler.update_attachment_ui()
    
    async def toggle_attach_button_visibility(self, show: bool) -> None:
        """Toggle the visibility of the attach file button."""
        try:
            if show:
                # Check if button already exists
                try:
                    attach_button = self.query_one("#attach-image", Button)
                    # Button already exists, no need to add
                    return
                except NoMatches:
                    pass  # Button doesn't exist, need to add it
                
                # Find the input area and send button
                try:
                    input_area = self.query_one("#chat-input-area", Horizontal)
                    send_button = self.query_one("#send-stop-chat", Button)
                except NoMatches:
                    logger.warning("Input area or send button not found")
                    return
                
                # Create and mount the button after the send button
                attach_button = Button(
                    "📎", 
                    id="attach-image", 
                    classes="action-button attach-button",
                    tooltip="Attach file"
                )
                await input_area.mount(attach_button, after=send_button)
                
            else:
                # Remove the button if it exists
                try:
                    attach_button = self.query_one("#attach-image", Button)
                    await attach_button.remove()
                    # Clear attachment state when hiding the button
                    self._clear_attachment_state()
                except NoMatches:
                    # Button doesn't exist, nothing to remove
                    pass
                    
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Error toggling attach button visibility: {e}")
    
    
    async def handle_notes_expand_button(self, app, event) -> None:
        """Delegate to sidebar handler."""
        await self.sidebar_handler.handle_notes_expand_button(event)
    
    async def action_resize_sidebar_shrink(self) -> None:
        """Action for keyboard shortcut to shrink sidebar."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_shrink(self.app_instance, None)
    
    async def action_resize_sidebar_expand(self) -> None:
        """Action for keyboard shortcut to expand sidebar."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_expand(self.app_instance, None)
    
    async def action_edit_focused_message(self) -> None:
        """Delegate to message manager."""
        await self.message_manager.edit_focused_message()
    
    def _update_button_state(self) -> None:
        """Delegate to input handler."""
        self.input_handler.update_button_state()
    
    def watch_is_send_button(self, is_send: bool) -> None:
        """Watch for changes to button state and update UI accordingly."""
        button = self._get_send_button()
        if not button:
            logger.debug("Send button not found in watcher")
            return
        
        # Batch multiple button updates
        with self.app.batch_update():
            button.label = get_char(
                EMOJI_SEND if is_send else EMOJI_STOP,
                FALLBACK_SEND if is_send else FALLBACK_STOP
            )
            button.tooltip = "Send message" if is_send else "Stop generation"
            
            # Update button styling
            if is_send:
                button.remove_class("stop-state")
            else:
                button.add_class("stop-state")
    
    def watch_pending_image(self, image_data) -> None:
        """Watch for changes to pending image and update UI."""
        self._update_attachment_ui()
    
    def validate_pending_image(self, image_data) -> Any:
        """Validate pending image data."""
        if image_data is not None and not isinstance(image_data, dict):
            logger.warning(f"Invalid pending_image type: {type(image_data)}")
            return None
        return image_data
    
    
    async def handle_send_stop_button(self, app_instance, event):
        """Delegate to input handler."""
        await self.input_handler.handle_send_stop_button(event)
    
    async def handle_enhanced_send_button(self, app_instance, event):
        """Delegate to input handler."""
        await self.input_handler.handle_enhanced_send_button(event)
    
    async def handle_mic_button(self, app_instance, event: Button.Pressed) -> None:
        """Delegate to voice handler."""
        await self.voice_handler.handle_mic_button(event)
    
    def action_toggle_voice_input(self) -> None:
        """Delegate to voice handler."""
        self.voice_handler.toggle_voice_input()
        # Update local state for compatibility
        self.is_voice_recording = self.voice_handler.is_voice_recording
    
    def _insert_voice_text(self, text: str):
        """Insert voice text into chat input."""
        chat_input = self._get_chat_input()
        if not chat_input:
            logger.warning("Chat input widget not found for voice text")
            return
        
        # Use batch update for multiple operations
        with self.app.batch_update():
            current_text = chat_input.text
            
            # Add space if there's existing text
            if current_text and not current_text.endswith(' '):
                text = ' ' + text
            
            # Append transcribed text
            chat_input.load_text(current_text + text)
            
            # Focus the input
            chat_input.focus()
    
    def on_voice_input_message(self, event: VoiceInputMessage) -> None:
        """Handle voice input messages."""
        if event.is_final and event.text:
            # Add transcribed text to chat input
            chat_input = self._get_chat_input()
            if not chat_input:
                logger.warning("Chat input widget not found")
                return
            
            # Use batch update for multiple operations
            with self.app.batch_update():
                current_text = chat_input.text
                
                # Add space if there's existing text
                if current_text and not current_text.endswith(' '):
                    event.text = ' ' + event.text
                
                # Append transcribed text
                chat_input.load_text(current_text + event.text)
                
                # Focus the input
                chat_input.focus()

#
# End of Chat_Window_Enhanced.py
#######################################################################################################################