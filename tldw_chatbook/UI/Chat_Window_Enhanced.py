# Chat_Window_Enhanced.py
# Description: Enhanced Chat Window with image attachment support
#
# Imports
from typing import TYPE_CHECKING, Optional
from pathlib import Path
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
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
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
        self.pending_attachment = None  # New unified attachment system
        self.pending_image = None  # Deprecated - kept for backward compatibility
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
        """Show file picker dialog for attachments or legacy file input."""
        # Check if we're in test mode with a mocked file input
        try:
            # Try to find a file input field (legacy mode for tests)
            file_input = self.query_one("#file-path-input", Input)
            # If found, show it and focus
            file_input.remove_class("hidden")
            file_input.focus()
            return
        except Exception:
            # Normal mode - use file picker dialog
            pass
        
        from fnmatch import fnmatch
        from pathlib import Path
        
        def on_file_selected(file_path: Optional[Path]):
            if file_path:
                # Process the selected file
                async def process_async():
                    await self.process_file_attachment(str(file_path))
                self.app_instance.call_later(process_async)
        
        # Create filter functions
        def create_filter(patterns: str):
            """Create a filter function from semicolon-separated patterns."""
            pattern_list = patterns.split(';')
            def filter_func(path: Path) -> bool:
                return any(fnmatch(path.name, pattern) for pattern in pattern_list)
            return filter_func
        
        # Create comprehensive file filters
        file_filters = Filters(
            ("All Supported Files", create_filter("*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg;*.txt;*.md;*.log;*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.json;*.yaml;*.yml;*.csv;*.tsv;*.pdf;*.doc;*.docx;*.rtf;*.odt;*.epub;*.mobi;*.azw;*.azw3;*.fb2")),
            ("Image Files", create_filter("*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg")),
            ("Document Files", create_filter("*.pdf;*.doc;*.docx;*.rtf;*.odt")),
            ("E-book Files", create_filter("*.epub;*.mobi;*.azw;*.azw3;*.fb2")),
            ("Text Files", create_filter("*.txt;*.md;*.log;*.text;*.rst")),
            ("Code Files", create_filter("*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.swift;*.kt;*.php;*.r;*.m;*.lua;*.sh;*.bash;*.ps1;*.sql;*.html;*.css;*.xml")),
            ("Data Files", create_filter("*.json;*.yaml;*.yml;*.csv;*.tsv")),
            ("All Files", lambda path: True)
        )
        
        # Push the FileOpen dialog directly
        self.app_instance.push_screen(
            FileOpen(location=".",
                title="Select File to Attach",
                filters=file_filters,
                context="chat_images"),
            callback=on_file_selected
        )

    async def handle_clear_image_button(self, app_instance, event):
        """Clear attached image."""
        self.pending_image = None
        
        # Update attach button if it exists
        try:
            attach_button = self.query_one("#attach-image")
            attach_button.label = "📎"
        except Exception:
            pass
        
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
            
            # Update attach button if it exists
            try:
                attach_button = self.query_one("#attach-image")
                attach_button.label = "📎"
            except Exception:
                pass
            
            # Hide indicator
            indicator = self.query_one("#image-attachment-indicator")
            indicator.add_class("hidden")

    async def process_file_attachment(self, file_path: str) -> None:
        """Process selected file using appropriate handler."""
        from ..Utils.file_handlers import file_handler_registry
        from pathlib import Path
        
        try:
            logger.info(f"Processing file attachment: {file_path}")
            # Process the file
            processed_file = await file_handler_registry.process_file(file_path)
            logger.info(f"File processed successfully: {processed_file}")
            
            if processed_file.insert_mode == "inline":
                # For text/code/data files, insert content directly into chat input
                try:
                    logger.info("Attempting to insert inline content")
                    chat_input = self.query_one("#chat-input", TextArea)
                    logger.info(f"Found chat input: {chat_input}")
                    
                    # Get current content
                    current_text = chat_input.text
                    logger.info(f"Current text length: {len(current_text)}")
                    
                    # Add file content
                    if current_text:
                        # If there's existing text, add a newline before the file content
                        new_text = current_text + "\n\n" + processed_file.content
                    else:
                        new_text = processed_file.content
                    
                    logger.info(f"New text length: {len(new_text)}")
                    # Update the text area
                    chat_input.text = new_text
                    # Move cursor to end - TextArea cursor_location needs (row, column) tuple
                    try:
                        # Calculate the row and column for the end position
                        lines = new_text.split('\n')
                        last_row = len(lines) - 1
                        last_col = len(lines[-1]) if lines else 0
                        chat_input.cursor_location = (last_row, last_col)
                    except Exception as cursor_error:
                        logger.warning(f"Failed to set cursor location: {cursor_error}")
                    
                    # Show notification
                    emoji_map = {
                        "text": "📄",
                        "code": "💻", 
                        "data": "📊",
                        "pdf": "📕",
                        "ebook": "📚",
                        "document": "📝",
                        "file": "📎"
                    }
                    emoji = emoji_map.get(processed_file.file_type, "📎")
                    self.app_instance.notify(f"{emoji} {processed_file.display_name} content inserted")
                    
                except Exception as e:
                    logger.error(f"Failed to insert file content: {e}", exc_info=True)
                    self.app_instance.notify(f"Failed to insert content: {str(e)}", severity="error")
                    
            elif processed_file.insert_mode == "attachment":
                # For images and other attachments, store as pending
                self.pending_attachment = {
                    'data': processed_file.attachment_data,
                    'mime_type': processed_file.attachment_mime_type,
                    'path': file_path,
                    'display_name': processed_file.display_name,
                    'file_type': processed_file.file_type,
                    'insert_mode': processed_file.insert_mode
                }
                
                # For backward compatibility, also set pending_image if it's an image
                if processed_file.file_type == "image":
                    self.pending_image = {
                        'data': processed_file.attachment_data,
                        'mime_type': processed_file.attachment_mime_type,
                        'path': file_path
                    }
                
                # Update UI to show file is attached
                try:
                    attach_button = self.query_one("#attach-image")
                    attach_button.label = "📎✓"
                except Exception:
                    pass
                
                # Show indicator with filename
                indicator = self.query_one("#image-attachment-indicator")
                emoji_map = {"image": "📷", "file": "📎"}
                emoji = emoji_map.get(processed_file.file_type, "📎")
                indicator.update(f"{emoji} {processed_file.display_name}")
                indicator.remove_class("hidden")
                
                self.app_instance.notify(f"{processed_file.display_name} attached")
                
        except Exception as e:
            logger.error(f"Error processing file attachment: {e}", exc_info=True)
            self.app_instance.notify(f"Error processing file: {str(e)}", severity="error")

    async def handle_image_path_submitted(self, event):
        """Handle image path submission from file input field.
        
        This method is for backward compatibility with tests that expect
        the old file input field behavior.
        """
        from ..Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
        from pathlib import Path
        
        try:
            file_path = event.value
            if not file_path:
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
                
                # Update UI elements
                try:
                    # Update attach button
                    attach_button = self.query_one("#attach-image")
                    attach_button.label = "📎✓"
                except Exception:
                    pass
                
                # Show indicator
                try:
                    indicator = self.query_one("#image-attachment-indicator")
                    indicator.update(f"📷 {path.name}")
                    indicator.remove_class("hidden")
                except Exception:
                    pass
                
                # Hide file input if it exists
                if hasattr(event, 'input') and event.input:
                    event.input.add_class("hidden")
                
                # Notify user
                self.app_instance.notify(f"Image attached: {path.name}")
                
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                self.app_instance.notify(
                    f"Error attaching image: {str(e)}",
                    severity="error"
                )
                
        except Exception as e:
            logger.error(f"Error in handle_image_path_submitted: {e}", exc_info=True)
            self.app_instance.notify(
                f"Error processing image path: {e}",
                severity="error"
            )


    def compose(self) -> ComposeResult:
        logger.debug("Composing ChatWindowEnhanced UI")
        # Settings Sidebar (Left)
        yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)

        # Main Chat Content Area
        with Container(id="chat-main-content"):
            yield VerticalScroll(id="chat-log")
            
            # Image attachment indicator
            yield Static(
                "",
                id="image-attachment-indicator",
                classes="hidden"
            )
            
            with Horizontal(id="chat-input-area"):
                yield Button(
                    get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), 
                    id="toggle-chat-left-sidebar",
                    classes="sidebar-toggle",
                    tooltip="Toggle left sidebar (Ctrl+[)"
                )
                yield TextArea(id="chat-input", classes="chat-input")
                
                yield Button(
                    get_char(EMOJI_SEND if self.is_send_button else EMOJI_STOP, 
                            FALLBACK_SEND if self.is_send_button else FALLBACK_STOP),
                    id="send-stop-chat",
                    classes="send-button",
                    tooltip="Send message" if self.is_send_button else "Stop generation"
                )
                
                # Check config to see if attach button should be shown
                from ..config import get_cli_setting
                show_attach_button = get_cli_setting("chat.images", "show_attach_button", True)
                if show_attach_button:
                    yield Button(
                        "📎", 
                        id="attach-image", 
                        classes="action-button attach-button",
                        tooltip="Attach file"
                    )
                yield Button(
                    "💡", 
                    id="respond-for-me-button", 
                    classes="action-button suggest-button",
                    tooltip="Suggest a response"
                )
                logger.debug("'respond-for-me-button' composed.")
                yield Button(
                    get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), 
                    id="toggle-chat-right-sidebar",
                    classes="sidebar-toggle",
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
    
    async def toggle_attach_button_visibility(self, show: bool) -> None:
        """Toggle the visibility of the attach file button."""
        try:
            if show:
                # Check if button already exists
                try:
                    self.query_one("#attach-image")
                    # Button already exists, no need to add
                    return
                except Exception:
                    # Button doesn't exist, need to add it
                    pass
                
                # Find the input area and add the button
                input_area = self.query_one("#chat-input-area", Horizontal)
                send_button = self.query_one("#send-stop-chat", Button)
                
                # Create and mount the button after the send button
                attach_button = Button(
                    "📎", 
                    id="attach-image", 
                    classes="action-button attach-button",
                    tooltip="Attach file"
                )
                await input_area.mount(attach_button, after=send_button)
                
                # Re-arrange the buttons to maintain order
                await self._rearrange_chat_buttons()
                
            else:
                # Remove the button if it exists
                try:
                    attach_button = self.query_one("#attach-image")
                    await attach_button.remove()
                    # Clear any pending image when hiding the button
                    self.pending_image = None
                    # Hide the indicator as well
                    indicator = self.query_one("#image-attachment-indicator")
                    indicator.add_class("hidden")
                except Exception:
                    # Button doesn't exist, nothing to remove
                    pass
                    
        except Exception as e:
            logger.error(f"Error toggling attach button visibility: {e}")
    
    async def _rearrange_chat_buttons(self) -> None:
        """Rearrange chat buttons to maintain proper order after dynamic addition."""
        try:
            input_area = self.query_one("#chat-input-area", Horizontal)
            
            # Get all the buttons in the desired order
            send_stop_button = self.query_one("#send-stop-chat", Button)
            attach_button = self.query_one("#attach-image", Button)
            respond_button = self.query_one("#respond-for-me-button", Button)
            right_sidebar_button = self.query_one("#toggle-chat-right-sidebar", Button)
            
            # Move them to the end in the correct order
            await send_stop_button.move(parent=input_area, after=-1)
            await attach_button.move(parent=input_area, after=-1)
            await respond_button.move(parent=input_area, after=-1)
            await right_sidebar_button.move(parent=input_area, after=-1)
            
        except Exception as e:
            logger.debug(f"Could not rearrange buttons: {e}")
    
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
                button.label = "Expand Notes"
            else:
                # Expand
                textarea.remove_class("notes-textarea-normal")
                textarea.add_class("notes-textarea-expanded")
                textarea.styles.height = 25
                button.label = "Collapse Notes"
                
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
            
            # Update attach button if it exists
            try:
                attach_button = self.query_one("#attach-image")
                attach_button.label = "📎"
            except Exception:
                pass
            
            # Hide indicator
            indicator = self.query_one("#image-attachment-indicator")
            indicator.add_class("hidden")

#
# End of Chat_Window_Enhanced.py
#######################################################################################################################