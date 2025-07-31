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
from textual.widgets import Button, TextArea, Input, Static, Select
from textual.reactive import reactive
#
# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
from ..Widgets.chat_right_sidebar import create_chat_right_sidebar
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Widgets.chat_tab_container import ChatTabContainer
from ..Widgets.voice_input_widget import VoiceInputWidget, VoiceInputMessage
from ..config import get_cli_setting
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP
from ..Event_Handlers.Audio_Events import DictationStartedEvent, DictationStoppedEvent

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
        
        # Voice input state
        self.voice_input_widget: Optional[VoiceInputWidget] = None
        self.is_voice_recording = False
        
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

        # Check if tabs are enabled and if this is a tab-specific button
        enable_tabs = get_cli_setting("chat_defaults", "enable_tabs", False)
        if enable_tabs and hasattr(self, 'tab_container'):
            # Check if the button is from a chat session
            # Tab-specific buttons will have IDs like "send-stop-chat-abc123"
            for session_id, session in self.tab_container.sessions.items():
                if button_id.endswith(f"-{session_id}"):
                    # This is a tab-specific button, let the session handle it
                    logger.debug(f"Tab-specific button detected, delegating to session {session_id}")
                    return  # The ChatSession will handle this via its own @on decorator

        # Map of button IDs to their handler functions
        button_handlers = {
            "send-stop-chat": self.handle_send_stop_button,  # New unified handler
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
            # Voice input handler
            "mic-button": self.handle_mic_button,
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
            # These buttons are handled at the app level via @on decorators, so don't warn
            app_level_buttons = {
                "chat-notes-search-button",
                "chat-notes-load-button",
                "chat-notes-create-button",
                "chat-notes-delete-button",
                "chat-notes-save-button"
            }
            if button_id not in app_level_buttons:
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
        """Clear attached file."""
        # Clear all attachment data
        self._clear_attachment_state()
        
        app_instance.notify("File attachment cleared")

    async def handle_enhanced_send_button(self, app_instance, event):
        """Enhanced send handler that includes image data."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        # First call the original handler
        await chat_events.handle_chat_send_button_pressed(app_instance, event)
        
        # Clear attachment states after successful send
        self._clear_attachment_state()

    async def process_file_attachment(self, file_path: str) -> None:
        """Process selected file using appropriate handler."""
        from ..Utils.file_handlers import file_handler_registry
        from ..Utils.path_validation import is_safe_path
        from pathlib import Path
        
        try:
            logger.info(f"Processing file attachment: {file_path}")
            
            # Validate the file path is safe (within user's home directory)
            import os
            if not is_safe_path(file_path, os.path.expanduser("~")):
                raise ValueError("File path is outside allowed directory")
            
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
                logger.info(f"DEBUG: Set pending_attachment - file_type: {processed_file.file_type}, mime_type: {processed_file.attachment_mime_type}, data_size: {len(processed_file.attachment_data) if processed_file.attachment_data else 0}")
                
                # For backward compatibility, also set pending_image if it's an image
                if processed_file.file_type == "image":
                    self.pending_image = {
                        'data': processed_file.attachment_data,
                        'mime_type': processed_file.attachment_mime_type,
                        'path': file_path
                    }
                    
                    # Check if current model supports vision
                    try:
                        from ...model_capabilities import is_vision_capable
                        provider_widget = self.app_instance.query_one("#chat-api-provider", Select)
                        model_widget = self.app_instance.query_one("#chat-api-model", Select)
                        
                        selected_provider = str(provider_widget.value) if provider_widget.value != Select.BLANK else None
                        selected_model = str(model_widget.value) if model_widget.value != Select.BLANK else None
                        
                        if selected_provider and selected_model:
                            vision_capable = is_vision_capable(selected_provider, selected_model)
                            if not vision_capable:
                                self.app_instance.notify(
                                    f"⚠️ {selected_model} doesn't support images. Select a vision model to send images.",
                                    severity="warning",
                                    timeout=6
                                )
                                logger.warning(f"User attached image but model {selected_provider}/{selected_model} doesn't support vision")
                    except Exception as e:
                        logger.debug(f"Could not check vision capability: {e}")
                
                # Use centralized UI update
                self._update_attachment_ui()
                
                self.app_instance.notify(f"{processed_file.display_name} attached")
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            self.app_instance.notify(f"File not found: {Path(file_path).name}", severity="error")
            # Clear any partial state
            self._clear_attachment_state()
        except PermissionError as e:
            logger.error(f"Permission denied accessing file: {file_path}")
            self.app_instance.notify(f"Permission denied: {Path(file_path).name}", severity="error")
            self._clear_attachment_state()
        except ValueError as e:
            # File handler validation errors
            logger.error(f"File validation error: {e}")
            self.app_instance.notify(str(e), severity="error")
            self._clear_attachment_state()
        except MemoryError as e:
            logger.error(f"Out of memory processing file: {file_path}")
            self.app_instance.notify("File too large to process", severity="error")
            self._clear_attachment_state()
        except Exception as e:
            logger.error(f"Unexpected error processing file attachment: {e}", exc_info=True)
            self.app_instance.notify(f"Error processing file: {str(e)}", severity="error")
            self._clear_attachment_state()

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

        # Left sidebar toggle button
        yield Button(
            get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), 
            id="toggle-chat-left-sidebar",
            classes="chat-sidebar-toggle-button",
            tooltip="Toggle left sidebar (Ctrl+\[)"
        )

        # Check if tabs are enabled
        enable_tabs = get_cli_setting("chat_defaults", "enable_tabs", False)
        
        # Main Chat Content Area
        with Container(id="chat-main-content"):
            if enable_tabs:
                logger.info("Chat tabs are enabled - using ChatTabContainer in enhanced mode")
                # Use the tab container for multiple sessions
                self.tab_container = ChatTabContainer(self.app_instance)
                self.tab_container.enhanced_mode = True  # Flag for enhanced features
                yield self.tab_container
            else:
                # Legacy single-session mode
                yield VerticalScroll(id="chat-log")
                
                # Image attachment indicator
                yield Static(
                    "",
                    id="image-attachment-indicator",
                    classes="hidden"
                )
                
                with Horizontal(id="chat-input-area"):
                    yield TextArea(id="chat-input", classes="chat-input")
                    
                    # Microphone button for voice input
                    show_mic_button = get_cli_setting("chat.voice", "show_mic_button", True)
                    if show_mic_button:
                        yield Button(
                            get_char("🎤", "⚫"),
                            id="mic-button",
                            classes="mic-button",
                            tooltip="Voice input (Ctrl+M)"
                        )
                    
                    yield Button(
                        get_char(EMOJI_SEND if self.is_send_button else EMOJI_STOP, 
                                FALLBACK_SEND if self.is_send_button else FALLBACK_STOP),
                        id="send-stop-chat",
                        classes="send-button",
                        tooltip="Send message" if self.is_send_button else "Stop generation"
                    )
                    
                    # Check config to see if attach button should be shown
                    show_attach_button = get_cli_setting("chat.images", "show_attach_button", True)
                    if show_attach_button:
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
            tooltip="Toggle right sidebar (Ctrl+\])"
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
        """Clear all attachment state and update UI consistently."""
        # Clear data
        self.pending_image = None
        self.pending_attachment = None
        
        # Update attach button
        try:
            attach_button = self.query_one("#attach-image")
            attach_button.label = "📎"
        except Exception:
            pass
        
        # Hide indicator
        try:
            indicator = self.query_one("#image-attachment-indicator")
            indicator.add_class("hidden")
        except Exception:
            pass
    
    def _update_attachment_ui(self):
        """Update UI elements based on current attachment state."""
        try:
            # Update attach button appearance based on attachment state
            attach_button = self.query_one("#attach-image", Button)
            
            if self.pending_image or self.pending_attachment:
                # Show indicator that file is attached
                attach_button.label = "📎✓"
                
                # Update indicator visibility and text
                try:
                    indicator = self.query_one("#image-attachment-indicator", Static)
                    
                    if self.pending_attachment:
                        # For new unified attachment system
                        display_name = self.pending_attachment.get('display_name', 'File')
                        file_type = self.pending_attachment.get('file_type', 'file')
                        emoji_map = {"image": "📷", "file": "📎", "code": "💻", "text": "📄", "data": "📊"}
                        emoji = emoji_map.get(file_type, "📎")
                        indicator.update(f"{emoji} {display_name}")
                    elif self.pending_image:
                        # For legacy image system
                        if isinstance(self.pending_image, dict):
                            # Extract filename from path if available
                            path = self.pending_image.get('path', '')
                            if path:
                                from pathlib import Path
                                filename = Path(path).name
                                indicator.update(f"📷 {filename}")
                            else:
                                indicator.update("📷 Image attached")
                        else:
                            indicator.update("📷 Image attached")
                    
                    indicator.remove_class("hidden")
                except Exception:
                    # Indicator might not exist yet
                    pass
            else:
                # No attachment - reset to default
                attach_button.label = "📎"
                
                # Hide indicator
                try:
                    indicator = self.query_one("#image-attachment-indicator")
                    indicator.add_class("hidden")
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Error updating attachment UI: {e}")
    
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
                
            else:
                # Remove the button if it exists
                try:
                    attach_button = self.query_one("#attach-image")
                    await attach_button.remove()
                    # Clear attachment state when hiding the button
                    self._clear_attachment_state()
                except Exception:
                    # Button doesn't exist, nothing to remove
                    pass
                    
        except Exception as e:
            logger.error(f"Error toggling attach button visibility: {e}")
    
    
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
        
        # Clear attachment states after successful send
        self._clear_attachment_state()
    
    async def handle_mic_button(self, app_instance, event: Button.Pressed) -> None:
        """Handle microphone button press for voice input."""
        # Call the toggle action
        self.action_toggle_voice_input()
    
    def action_toggle_voice_input(self) -> None:
        """Toggle voice input recording."""
        if not hasattr(self, 'voice_dictation_service'):
            # Create voice dictation service if not exists
            self._create_voice_input_widget()
            
        if not hasattr(self, 'voice_dictation_service') or not self.voice_dictation_service:
            self.app_instance.notify("Voice input not available", severity="error")
            return
        
        if self.is_voice_recording:
            # Stop recording
            self._stop_voice_recording()
        else:
            # Start recording
            self._start_voice_recording()
    
    def _create_voice_input_widget(self):
        """Create the voice input widget."""
        try:
            # Use a simpler approach - just use the dictation service directly
            from ..Audio.dictation_service_lazy import LazyLiveDictationService, AudioInitializationError
            
            self.voice_dictation_service = LazyLiveDictationService(
                transcription_provider=get_cli_setting('transcription', 'default_provider', 'faster-whisper'),
                transcription_model=get_cli_setting('transcription', 'default_model', 'base'),
                language=get_cli_setting('transcription', 'default_language', 'en'),
                enable_punctuation=True,
                enable_commands=False
            )
            logger.info("Voice dictation service created")
        except Exception as e:
            logger.error(f"Failed to create voice dictation service: {e}")
            self.voice_dictation_service = None
            # Don't show error here - will show when user actually tries to use it
    
    def _start_voice_recording(self):
        """Start voice recording."""
        try:
            from ..Audio.dictation_service_lazy import AudioInitializationError
            
            # Update UI
            mic_button = self.query_one("#mic-button", Button)
            mic_button.label = "🛑"  # Stop icon
            mic_button.variant = "error"
            
            # Start dictation
            success = self.voice_dictation_service.start_dictation(
                on_partial_transcript=self._on_voice_partial,
                on_final_transcript=self._on_voice_final,
                on_error=self._on_voice_error
            )
            
            if success:
                self.is_voice_recording = True
                self.app_instance.notify("🎤 Listening...", timeout=2)
            else:
                self.app_instance.notify("Failed to start recording", severity="error")
                # Reset button
                mic_button.label = "🎤"
                mic_button.variant = "default"
                
        except AudioInitializationError as e:
            logger.error(f"Audio initialization error: {e}")
            # Show the specific error message which includes instructions
            self.app_instance.notify(str(e), severity="error", timeout=10)
            # Reset button
            mic_button = self.query_one("#mic-button", Button)
            mic_button.label = "🎤"
            mic_button.variant = "default"
        except Exception as e:
            logger.error(f"Error starting voice recording: {e}")
            if "no default" in str(e).lower() or "invalid input device" in str(e).lower():
                self.app_instance.notify(
                    "No microphone access. Grant permissions in System Settings > Privacy > Microphone",
                    severity="error",
                    timeout=10
                )
            else:
                self.app_instance.notify(f"Voice recording error: {str(e)}", severity="error")
            # Reset button
            mic_button = self.query_one("#mic-button", Button)
            mic_button.label = "🎤"
            mic_button.variant = "default"
    
    def _stop_voice_recording(self):
        """Stop voice recording."""
        try:
            # Stop dictation
            result = self.voice_dictation_service.stop_dictation()
            
            # Update UI
            mic_button = self.query_one("#mic-button", Button)
            mic_button.label = "🎤"
            mic_button.variant = "default"
            
            self.is_voice_recording = False
            
            # Insert final transcript if any
            if result.transcript:
                self._insert_voice_text(result.transcript)
                word_count = len(result.transcript.split())
                self.app_instance.notify(f"✓ Added {word_count} words", timeout=2)
            else:
                self.app_instance.notify("No speech detected", severity="warning")
                
        except Exception as e:
            logger.error(f"Error stopping voice recording: {e}")
            self.app_instance.notify("Error stopping recording", severity="error")
    
    def _on_voice_partial(self, text: str):
        """Handle partial voice transcript."""
        # Could show preview in status bar or tooltip
        pass
    
    def _on_voice_final(self, text: str):
        """Handle final voice transcript segment."""
        # For continuous transcription, could insert segments as they complete
        pass
    
    def _on_voice_error(self, error: Exception):
        """Handle voice recording error."""
        logger.error(f"Voice recording error: {error}")
        self.app_instance.notify(f"Voice error: {str(error)}", severity="error")
        # Reset UI
        try:
            mic_button = self.query_one("#mic-button", Button)
            mic_button.label = "🎤"
            mic_button.variant = "default"
        except:
            pass
        self.is_voice_recording = False
    
    def _insert_voice_text(self, text: str):
        """Insert voice text into chat input."""
        try:
            chat_input = self.query_one("#chat-input", TextArea)
            current_text = chat_input.text
            
            # Add space if there's existing text
            if current_text and not current_text.endswith(' '):
                text = ' ' + text
            
            # Append transcribed text
            chat_input.load_text(current_text + text)
            
            # Focus the input
            chat_input.focus()
        except Exception as e:
            logger.error(f"Failed to insert voice text: {e}")
    
    def on_voice_input_message(self, event: VoiceInputMessage) -> None:
        """Handle voice input messages."""
        if event.is_final and event.text:
            # Add transcribed text to chat input
            try:
                chat_input = self.query_one("#chat-input", TextArea)
                current_text = chat_input.text
                
                # Add space if there's existing text
                if current_text and not current_text.endswith(' '):
                    event.text = ' ' + event.text
                
                # Append transcribed text
                chat_input.load_text(current_text + event.text)
                
                # Focus the input
                chat_input.focus()
            except Exception as e:
                logger.error(f"Failed to add voice input to chat: {e}")

#
# End of Chat_Window_Enhanced.py
#######################################################################################################################