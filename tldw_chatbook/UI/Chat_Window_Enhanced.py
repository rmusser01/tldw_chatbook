# Chat_Window_Enhanced.py
# Description: Enhanced Chat Window with image attachment support
#
# Imports
from typing import TYPE_CHECKING, Optional
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea, Input, Static
from textual.reactive import reactive
#
# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
from ..Widgets.chat_right_sidebar import create_chat_right_sidebar
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP

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
    ]
    
    # CSS for hidden elements
    DEFAULT_CSS = """
    .hidden {
        display: none;
    }
    
    #image-file-path {
        margin: 0 1;
        height: 3;
    }
    
    #image-attachment-indicator {
        margin: 0 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        height: 3;
    }
    """
    
    # Track pending image attachment
    pending_image = reactive(None)
    
    # Track button state for Send/Stop functionality
    is_send_button = reactive(True)
    
    # Debouncing for button clicks
    _last_send_stop_click = 0
    DEBOUNCE_MS = 300
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("ChatWindowEnhanced initialized.")
    
    async def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Token counter will be initialized when tab is switched to chat
        # Watch for streaming state changes
        self._update_button_state()
        # Set up periodic state checking (every 500ms)
        self.set_interval(0.5, self._check_streaming_state)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Central handler for button presses in the ChatWindow.
        Delegates to the appropriate handler in chat_events.py.
        """
        from ..Event_Handlers.Chat_Events import chat_events
        from ..Event_Handlers.Chat_Events import chat_events_sidebar
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize

        button_id = event.button.id
        if not button_id:
            logger.warning("Button pressed with no ID")
            return

        logger.debug(f"Button pressed: {button_id}")

        # Map of button IDs to their handler functions
        button_handlers = {
            "send-stop-chat": self.handle_send_stop_button,  # New unified handler
            "respond-for-me-button": chat_events.handle_respond_for_me_button_pressed,
            "toggle-chat-left-sidebar": chat_events.handle_chat_tab_sidebar_toggle,
            "toggle-chat-right-sidebar": chat_events.handle_chat_tab_sidebar_toggle,
            "chat-new-conversation-button": chat_events.handle_chat_new_conversation_button_pressed,
            "chat-save-current-chat-button": chat_events.handle_chat_save_current_chat_button_pressed,
            "chat-save-conversation-details-button": chat_events.handle_chat_save_details_button_pressed,
            "chat-conversation-load-selected-button": chat_events.handle_chat_load_selected_button_pressed,
            "chat-prompt-load-selected-button": chat_events.handle_chat_view_selected_prompt_button_pressed,
            "chat-prompt-copy-system-button": chat_events.handle_chat_copy_system_prompt_button_pressed,
            "chat-prompt-copy-user-button": chat_events.handle_chat_copy_user_prompt_button_pressed,
            "chat-load-character-button": chat_events.handle_chat_load_character_button_pressed,
            "chat-clear-active-character-button": chat_events.handle_chat_clear_active_character_button_pressed,
            "chat-apply-template-button": chat_events.handle_chat_apply_template_button_pressed,
            # New image-related handlers
            "attach-image": self.handle_attach_image_button,
            "clear-image": self.handle_clear_image_button,
            # Notes expand/collapse handler
            "chat-notes-expand-button": self.handle_notes_expand_button,
        }

        # Add sidebar button handlers
        button_handlers.update(chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS)
        # Add sidebar resize handlers
        button_handlers.update(chat_events_sidebar_resize.CHAT_SIDEBAR_RESIZE_HANDLERS)

        # Check if we have a handler for this button
        handler = button_handlers.get(button_id)
        if handler:
            logger.debug(f"Calling handler for button: {button_id}")
            # Call the handler with the app instance and event
            await handler(self.app_instance, event)
            # Stop the event from propagating
            event.stop()
        else:
            logger.warning(f"No handler found for button: {button_id}")

    async def handle_attach_image_button(self, app_instance, event):
        """Show image file path input."""
        file_input = self.query_one("#image-file-path")
        file_input.remove_class("hidden")
        file_input.focus()

    async def handle_clear_image_button(self, app_instance, event):
        """Clear attached image."""
        self.pending_image = None
        attach_button = self.query_one("#attach-image")
        attach_button.label = "📎"
        
        # Hide indicator
        indicator = self.query_one("#image-attachment-indicator")
        indicator.add_class("hidden")
        
        app_instance.notify("Image attachment cleared")

    async def handle_enhanced_send_button(self, app_instance, event):
        """Enhanced send handler that includes image data."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        # First call the original handler
        await chat_events.handle_chat_send_button_pressed(app_instance, event)
        
        # Clear any pending image after sending
        if self.pending_image:
            self.pending_image = None
            attach_button = self.query_one("#attach-image")
            attach_button.label = "📎"
            
            # Hide indicator
            indicator = self.query_one("#image-attachment-indicator")
            indicator.add_class("hidden")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle image path submission."""
        if event.input.id == "image-file-path":
            await self.handle_image_path_submitted(event)

    async def handle_image_path_submitted(self, event: Input.Submitted) -> None:
        """Process submitted image path."""
        from ..Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
        
        try:
            file_path = event.value.strip()
            if not file_path:
                return
            
            # Process image
            image_data, mime_type = await ChatImageHandler.process_image_file(file_path)
            
            # Store in temporary state
            self.pending_image = {
                'data': image_data,
                'mime_type': mime_type,
                'path': file_path
            }
            
            # Update UI to show image is attached
            attach_button = self.query_one("#attach-image")
            attach_button.label = "📎✓"
            
            # Show indicator with filename
            from pathlib import Path
            filename = Path(file_path).name
            indicator = self.query_one("#image-attachment-indicator")
            indicator.update(f"📷 {filename}")
            indicator.remove_class("hidden")
            
            self.app_instance.notify(f"Image attached: {filename}")
            
            # Hide file input
            event.input.add_class("hidden")
            event.input.value = ""
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {file_path}")
            self.app_instance.notify(f"Image file not found: {file_path}", severity="error")
            event.input.value = ""
            event.input.focus()
        except ValueError as e:
            # This is thrown by ChatImageHandler for format/size issues
            logger.error(f"Invalid image: {e}")
            self.app_instance.notify(str(e), severity="error")
            event.input.value = ""
            event.input.focus()
        except PermissionError:
            logger.error(f"Permission denied accessing image: {file_path}")
            self.app_instance.notify(f"Permission denied: Cannot access {file_path}", severity="error")
            event.input.value = ""
            event.input.focus()
        except MemoryError:
            logger.error(f"Image too large to load into memory: {file_path}")
            self.app_instance.notify("Image is too large to process. Please use a smaller image.", severity="error")
            event.input.value = ""
            event.input.focus()
        except Exception as e:
            logger.error(f"Unexpected error attaching image: {e}", exc_info=True)
            self.app_instance.notify(f"Unexpected error: {e}", severity="error")
            event.input.value = ""
            event.input.focus()

    def compose(self) -> ComposeResult:
        logger.debug("Composing ChatWindowEnhanced UI")
        # Settings Sidebar (Left)
        yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)

        # Main Chat Content Area
        with Container(id="chat-main-content"):
            yield VerticalScroll(id="chat-log")
            
            # Hidden file path input for image selection
            yield Input(
                id="image-file-path",
                placeholder="Enter image file path or drag & drop...",
                classes="hidden"
            )
            
            # Image attachment indicator
            yield Static(
                "",
                id="image-attachment-indicator",
                classes="hidden"
            )
            
            with Horizontal(id="chat-input-area"):
                yield Button(get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), id="toggle-chat-left-sidebar",
                             classes="sidebar-toggle")
                yield TextArea(id="chat-input", classes="chat-input")
                yield Button("📎", id="attach-image", classes="action-button attach-button")
                yield Button(
                    get_char(EMOJI_SEND if self.is_send_button else EMOJI_STOP, 
                            FALLBACK_SEND if self.is_send_button else FALLBACK_STOP),
                    id="send-stop-chat",
                    classes="send-button",
                    tooltip="Send message" if self.is_send_button else "Stop generation"
                )
                yield Button("💡", id="respond-for-me-button", classes="action-button suggest-button")
                logger.debug("'respond-for-me-button' composed.")
                yield Button(get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), id="toggle-chat-right-sidebar",
                             classes="sidebar-toggle")

        # Character Details Sidebar (Right)
        yield from create_chat_right_sidebar(
            "chat",
            initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
        )

    def get_pending_image(self) -> Optional[dict]:
        """Get the pending image attachment data."""
        return self.pending_image
    
    async def handle_notes_expand_button(self, app, event) -> None:
        """Handle the notes expand/collapse button."""
        try:
            button = app.query_one("#chat-notes-expand-button", Button)
            textarea = app.query_one("#chat-notes-content-textarea", TextArea)
            
            # Toggle between expanded and normal states
            if "notes-textarea-expanded" in textarea.classes:
                # Collapse
                textarea.remove_class("notes-textarea-expanded")
                textarea.add_class("notes-textarea-normal")
                textarea.styles.height = 10
                button.label = "⬆ Expand"
            else:
                # Expand
                textarea.remove_class("notes-textarea-normal")
                textarea.add_class("notes-textarea-expanded")
                textarea.styles.height = 25
                button.label = "⬇ Collapse"
                
            # Focus the textarea after expanding
            textarea.focus()
            
        except Exception as e:
            logger.error(f"Error handling notes expand button: {e}")
    
    async def action_resize_sidebar_shrink(self) -> None:
        """Action for keyboard shortcut to shrink sidebar."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_shrink(self.app_instance, None)
    
    async def action_resize_sidebar_expand(self) -> None:
        """Action for keyboard shortcut to expand sidebar."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_expand(self.app_instance, None)
    
    async def action_edit_focused_message(self) -> None:
        """Action for keyboard shortcut to edit the focused message."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        try:
            # Get the chat log container
            chat_log = self.app_instance.query_one("#chat-log", VerticalScroll)
            
            # Find the focused widget
            focused_widget = self.app_instance.focused
            
            # Check if the focused widget is a ChatMessage or if we need to find one
            from ..Widgets.chat_message import ChatMessage
            from ..Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            if isinstance(focused_widget, (ChatMessage, ChatMessageEnhanced)):
                message_widget = focused_widget
            else:
                # Try to find the last message in the chat log as a fallback
                messages = chat_log.query(ChatMessage)
                enhanced_messages = chat_log.query(ChatMessageEnhanced)
                all_messages = list(messages) + list(enhanced_messages)
                if all_messages:
                    message_widget = all_messages[-1]
                    message_widget.focus()
                else:
                    logger.debug("No messages found to edit")
                    return
            
            # Find the edit button in the message widget
            try:
                edit_button = message_widget.query_one(".edit-button", Button)
                # Trigger the edit action by simulating button press
                await chat_events.handle_chat_action_button_pressed(
                    self.app_instance, 
                    edit_button, 
                    message_widget
                )
            except Exception as e:
                logger.debug(f"Could not find or click edit button: {e}")
                
        except Exception as e:
            logger.error(f"Error in edit_focused_message action: {e}")
            self.app_instance.notify("Could not enter edit mode", severity="warning")
    
    def _update_button_state(self) -> None:
        """Update the send/stop button based on streaming state."""
        is_streaming = self.app_instance.get_current_chat_is_streaming()
        has_worker = (hasattr(self.app_instance, 'current_chat_worker') and 
                     self.app_instance.current_chat_worker and 
                     self.app_instance.current_chat_worker.is_running)
        
        # Update button state
        self.is_send_button = not (is_streaming or has_worker)
        
        # Update button appearance
        try:
            button = self.query_one("#send-stop-chat", Button)
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
    
    def watch_is_send_button(self, is_send: bool) -> None:
        """Watch for changes to button state to update appearance."""
        self._update_button_state()
    
    def _check_streaming_state(self) -> None:
        """Periodically check streaming state and update button."""
        self._update_button_state()
    
    async def handle_send_stop_button(self, app_instance, event):
        """Unified handler for Send/Stop button with debouncing."""
        from ..Event_Handlers.Chat_Events import chat_events
        import time
        
        current_time = time.time() * 1000
        
        # Debounce rapid clicks
        if current_time - self._last_send_stop_click < self.DEBOUNCE_MS:
            logger.debug("Button click debounced")
            return
        self._last_send_stop_click = current_time
        
        # Disable button during operation
        try:
            button = self.query_one("#send-stop-chat", Button)
            button.disabled = True
        except Exception:
            pass
        
        try:
            # Check current state and route to appropriate handler
            if self.app_instance.get_current_chat_is_streaming() or (
                hasattr(self.app_instance, 'current_chat_worker') and 
                self.app_instance.current_chat_worker and 
                self.app_instance.current_chat_worker.is_running
            ):
                # Stop operation
                logger.info("Send/Stop button pressed - stopping generation")
                await chat_events.handle_stop_chat_generation_pressed(app_instance, event)
            else:
                # Send operation - use enhanced handler that includes image
                logger.info("Send/Stop button pressed - sending message")
                await self.handle_enhanced_send_button(app_instance, event)
        finally:
            # Re-enable button and update state after operation
            try:
                button = self.query_one("#send-stop-chat", Button)
                button.disabled = False
            except Exception:
                pass
            self._update_button_state()
    
    async def handle_enhanced_send_button(self, app_instance, event):
        """Enhanced send handler that includes image data."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        # First call the original handler
        await chat_events.handle_chat_send_button_pressed(app_instance, event)
        
        # Clear any pending image after sending
        if self.pending_image:
            self.pending_image = None
            attach_button = self.query_one("#attach-image")
            attach_button.label = "📎"
            
            # Hide indicator
            indicator = self.query_one("#image-attachment-indicator")
            indicator.add_class("hidden")

#
# End of Chat_Window_Enhanced.py
#######################################################################################################################