# Complete Chat Window Architecture Analysis

## Part 1: The Old Chat Window (Full Code)

### Main Window: Chat_Window_Enhanced.py

```python
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
from textual.widgets import Button, TextArea, Input, Static, Select
from textual.reactive import reactive
#
# Local Imports
from ..Widgets.Chat_Widgets.unified_chat_sidebar import UnifiedChatSidebar
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from ..Widgets.voice_input_widget import VoiceInputWidget, VoiceInputMessage
from ..config import get_cli_setting
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP

# Configure logger with context
logger = logger.bind(module="Chat_Window_Enhanced")

#
if TYPE_CHECKING:
    from ..app import TldwCli

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
            "respond-for-me-button": chat_events.handle_respond_for_me_button_pressed,  # RESTORED
            "toggle-unified-sidebar": self.handle_toggle_unified_sidebar,  # NEW unified sidebar toggle
            "toggle-chat-left-sidebar": chat_events.handle_chat_tab_sidebar_toggle,  # Legacy compatibility
            "toggle-chat-right-sidebar": chat_events.handle_chat_tab_sidebar_toggle,  # Legacy compatibility
            "chat-new-conversation-button": chat_events.handle_chat_new_conversation_button_pressed,
            "chat-new-temp-chat-button": chat_events.handle_chat_new_conversation_button_pressed,  # New temp chat
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
        logger.debug("Composing ChatWindowEnhanced UI with unified sidebar")
        
        # Unified sidebar (replaces both left and right sidebars)
        yield UnifiedChatSidebar(self.app_instance)

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
                    # Sidebar toggle button (left side)
                    yield Button(
                        get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), 
                        id="toggle-unified-sidebar",
                        classes="sidebar-toggle",
                        tooltip="Toggle sidebar (Ctrl+\\)"
                    )
                    
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
                    
                    # Respond for me button (suggest response)
                    yield Button(
                        "💡", 
                        id="respond-for-me-button", 
                        classes="action-button suggest-button",
                        tooltip="Suggest a response"
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

        # Note: Right sidebar functionality is now integrated into the unified sidebar

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
    
    # ... [remaining 400+ lines of voice input, button handling, etc.]
```

## Part 2: The New Unified Sidebar (Full Code)

### UnifiedChatSidebar.py

```python
# unified_chat_sidebar.py
# Description: Unified single sidebar for chat interface with tabbed organization
#
# Imports
from typing import TYPE_CHECKING, Optional, Dict, Any, Set
import logging
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Static, TabbedContent, TabPane, Button, Input, Label, 
    TextArea, Checkbox, Select, ListView, ListItem
)
from textual.reactive import reactive
from textual.message import Message
from textual import on
#
# Local Imports
from ...config import get_cli_setting, save_setting_to_cli_config, get_cli_providers_and_models
from ...Utils.Emoji_Handling import get_char

if TYPE_CHECKING:
    from ...app import TldwCli

#
#######################################################################################################################
#
# State Management
#

class ChatSidebarState:
    """Centralized state management for the unified sidebar."""
    
    def __init__(self):
        self.active_tab: str = "session"
        self.search_query: str = ""
        self.search_filter: str = "all"
        self.collapsed_sections: Set[str] = set()
        self.sidebar_width: int = 30  # percentage
        self.sidebar_position: str = "left"  # Settings traditionally on left
        self.advanced_mode: bool = False
        self.current_page: Dict[str, int] = {}  # pagination state per content type
        
    def save_preferences(self):
        """Persist user preferences to config."""
        save_setting_to_cli_config("chat_sidebar", "width", self.sidebar_width)
        save_setting_to_cli_config("chat_sidebar", "position", self.sidebar_position)
        save_setting_to_cli_config("chat_sidebar", "active_tab", self.active_tab)
        save_setting_to_cli_config("chat_sidebar", "advanced_mode", self.advanced_mode)
    
    def load_preferences(self):
        """Load user preferences from config."""
        self.sidebar_width = get_cli_setting("chat_sidebar", "width", 30)
        self.sidebar_position = get_cli_setting("chat_sidebar", "position", "right")
        self.active_tab = get_cli_setting("chat_sidebar", "active_tab", "session")
        self.advanced_mode = get_cli_setting("chat_sidebar", "advanced_mode", False)

#
#######################################################################################################################
#
# Compound Widgets
#

class CompactField(Horizontal):
    """Space-efficient form field combining label and input."""
    
    def __init__(self, label: str, field_id: str, value: str = "", 
                 input_type: str = "input", placeholder: str = "", **kwargs):
        super().__init__(**kwargs)
        self.label_text = label
        self.field_id = field_id
        self.value = value
        self.input_type = input_type
        self.placeholder = placeholder
    
    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="compact-label")
        if self.input_type == "input":
            yield Input(id=self.field_id, value=self.value, 
                       placeholder=self.placeholder if self.placeholder else "", 
                       classes="compact-input")
        elif self.input_type == "textarea":
            yield TextArea(self.value, id=self.field_id, classes="compact-textarea")


class SearchableList(Container):
    """Reusable search interface for any content type."""
    
    def __init__(self, content_type: str, placeholder: str = "Search...", **kwargs):
        super().__init__(**kwargs)
        self.content_type = content_type
        self.placeholder = placeholder
        self.current_page = 1
        self.total_pages = 1
        self.results_per_page = 10
    
    def compose(self) -> ComposeResult:
        with Container(classes="searchable-list-container"):
            # Search input with integrated button
            with Horizontal(classes="search-bar"):
                yield Input(
                    placeholder=self.placeholder,
                    id=f"{self.content_type}-search-input",
                    classes="search-input"
                )
                yield Button("🔍", id=f"{self.content_type}-search-btn", classes="search-button")
            
            # Results list
            yield ListView(
                id=f"{self.content_type}-results",
                classes="search-results-list"
            )
            
            # Pagination controls
            with Horizontal(classes="pagination-controls"):
                yield Button("◀", id=f"{self.content_type}-prev", classes="page-btn", disabled=True)
                yield Label(f"Page {self.current_page}/{self.total_pages}", 
                          id=f"{self.content_type}-page-label", 
                          classes="page-label")
                yield Button("▶", id=f"{self.content_type}-next", classes="page-btn", disabled=True)


class SmartCollapsible(Container):
    """Collapsible section with auto-collapse and state memory."""
    
    collapsed = reactive(False)
    
    def __init__(self, title: str, section_id: str, auto_collapse: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.section_id = section_id
        self.auto_collapse = auto_collapse
        self.has_unsaved_changes = False
    
    def compose(self) -> ComposeResult:
        with Container(classes="smart-collapsible"):
            # Header with toggle
            with Horizontal(classes="collapsible-header"):
                yield Button(
                    f"{'▶' if self.collapsed else '▼'} {self.title}",
                    id=f"{self.section_id}-toggle",
                    classes="collapsible-toggle"
                )
            
            # Content container
            if not self.collapsed:
                with Container(id=f"{self.section_id}-content", classes="collapsible-content"):
                    yield from self.compose_content()
    
    def compose_content(self) -> ComposeResult:
        """Override in subclasses to provide content."""
        yield Static("Override compose_content in subclass")
    
    def toggle(self):
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
        self.refresh()

#
#######################################################################################################################
#
# Tab Content Components
#

class SessionTab(Container):
    """Session management tab content."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
    
    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="tab-content session-tab"):
            # Current chat info
            yield Static("Current Chat", classes="section-title")
            
            # Chat ID (read-only) - Using old ID for compatibility
            yield Label("Chat ID:", classes="sidebar-label")
            yield Input(
                id="chat-conversation-uuid-display",
                value="Temp Chat",
                disabled=True,
                classes="sidebar-input"
            )
            
            # Editable fields - Using old IDs for compatibility
            yield Label("Title:", classes="sidebar-label")
            yield Input(
                id="chat-conversation-title-input",
                placeholder="Enter chat title...",
                classes="sidebar-input"
            )
            
            yield Label("Keywords (comma-sep):", classes="sidebar-label")
            yield TextArea(
                "",
                id="chat-conversation-keywords-input",
                classes="sidebar-textarea chat-keywords-textarea"
            )
            
            # Action buttons
            yield Static("Actions", classes="section-title")
            with Horizontal(classes="button-group"):
                yield Button("New Temp Chat", id="chat-new-temp-chat-button", variant="primary")
                yield Button("New Chat", id="chat-new-conversation-button", variant="success")
            
            with Horizontal(classes="button-group"):
                yield Button("Save Details", id="chat-save-conversation-details-button", variant="primary")
                yield Button("Save Temp Chat", id="chat-save-current-chat-button", variant="success")
            
            with Horizontal(classes="button-group"):
                yield Button("🔄 Clone Chat", id="chat-clone-current-chat-button", variant="default")
                yield Button("📋 Convert to Note", id="chat-convert-to-note-button", variant="default")
            
            # Options
            yield Static("Options", classes="section-title")
            yield Checkbox("Strip Thinking Tags", id="chat-strip-thinking-tags-checkbox", value=True)


class SettingsTab(Container):
    """LLM settings tab with progressive disclosure."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.advanced_mode = reactive(False)
    
    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="tab-content settings-tab"):
            # Quick Settings (always visible)
            yield Static("Quick Settings", classes="section-title")
            
            # Provider and model selection
            providers_models = get_cli_providers_and_models()
            providers = list(providers_models.keys())
            
            yield Label("Provider", classes="field-label")
            provider_options = [(p, p) for p in providers]
            yield Select(
                options=provider_options,
                id="settings-provider",
                value=providers[0] if providers else None,
                classes="settings-select"
            )
            
            yield Label("Model", classes="field-label")
            yield Select(
                options=[],
                id="settings-model",
                classes="settings-select"
            )
            
            yield CompactField("Temperature:", "settings-temperature", value="0.7")
            
            # Advanced mode toggle
            yield Checkbox("Show Advanced Settings", id="settings-advanced-toggle", value=False)
            
            # Advanced settings (conditionally visible)
            with Container(id="advanced-settings", classes="hidden"):
                yield Static("Advanced Settings", classes="section-title")
                
                yield Label("System Prompt", classes="field-label")
                yield TextArea(
                    "",
                    id="settings-system-prompt",
                    classes="settings-textarea"
                )
                
                yield CompactField("Top-p:", "settings-top-p", value="0.95")
                yield CompactField("Top-k:", "settings-top-k", value="50")
                yield CompactField("Min-p:", "settings-min-p", value="0.05")
            
            # RAG Settings in collapsible
            yield Static("", classes="section-spacer")
            yield Button("▶ RAG Settings", id="rag-toggle", classes="collapsible-toggle")
            
            with Container(id="rag-settings", classes="hidden"):
                yield Checkbox("Enable RAG", id="settings-rag-enabled", value=False)
                
                yield Label("Pipeline", classes="field-label")
                pipeline_options = [
                    ("Manual Configuration", "none"),
                    ("Speed Optimized", "speed_optimized_v2"),
                    ("High Accuracy", "high_accuracy"),
                    ("Hybrid Search", "hybrid"),
                ]
                yield Select(
                    options=pipeline_options,
                    id="settings-rag-pipeline",
                    value="none",
                    classes="settings-select"
                )
                
                yield Button("Configure RAG", id="settings-rag-configure", variant="default")


class ContentTab(Container):
    """Unified content search and management tab."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_filter = "all"
    
    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="tab-content content-tab"):
            # Media Search Section (preserving old functionality)
            yield Static("Search Media", classes="section-title")
            
            yield Label("Search Term:", classes="sidebar-label")
            yield Input(
                id="chat-media-search-input",
                placeholder="Search title, content...",
                classes="sidebar-input"
            )
            
            yield Label("Filter by Keywords (comma-sep):", classes="sidebar-label")
            yield Input(
                id="chat-media-keyword-filter-input",
                placeholder="e.g., python, tutorial",
                classes="sidebar-input"
            )
            
            yield Button(
                "Search",
                id="chat-media-search-button",
                classes="sidebar-button"
            )
            
            # Results List
            yield ListView(id="chat-media-search-results-listview", classes="sidebar-listview")
            
            # Pagination controls
            with Horizontal(classes="pagination-controls", id="chat-media-pagination-controls"):
                yield Button("Prev", id="chat-media-prev-page-button", disabled=True)
                yield Label("Page 1/1", id="chat-media-page-label")
                yield Button("Next", id="chat-media-next-page-button", disabled=True)
            
            # Selected Media Details
            yield Static("--- Selected Media Details ---", classes="sidebar-label")
            
            # Media detail fields with copy buttons
            with Horizontal(classes="detail-field-container"):
                yield Label("Title:", classes="detail-label")
                yield TextArea(
                    "",
                    id="chat-media-title-display",
                    classes="detail-textarea",
                    disabled=True
                )
                yield Button("Copy", id="chat-media-copy-title-button", disabled=True, classes="copy-button")
            
            # ... [more detail fields]

#
#######################################################################################################################
#
# Main Unified Sidebar Widget
#

class UnifiedChatSidebar(Container):
    """Single unified sidebar for all chat functionality."""
    
    BINDINGS = [
        ("alt+1", "switch_tab('session')", "Session Tab"),
        ("alt+2", "switch_tab('settings')", "Settings Tab"),
        ("alt+3", "switch_tab('content')", "Content Tab"),
        ("ctrl+\\", "toggle_sidebar", "Toggle Sidebar"),
    ]
    
    DEFAULT_CSS = """
    UnifiedChatSidebar {
        dock: left;
        width: 30%;
        min-width: 250;
        max-width: 50%;
        background: $surface;
        border-right: solid $primary-darken-2;
        padding: 1;
    }
    
    UnifiedChatSidebar.collapsed {
        width: 0;
        min-width: 0;
        padding: 0;
        border: none;
        display: none;
    }
    
    .sidebar-header {
        height: 3;
        background: $boost;
        padding: 1;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        margin: 1 0;
        color: $text;
    }
    
    .tab-content {
        height: 100%;
        padding: 1;
    }
    
    .button-group {
        margin: 1 0;
    }
    
    .hidden {
        display: none;
    }
    """
    
    collapsed = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.state = ChatSidebarState()
        # Only load preferences if not in test mode
        import sys
        if 'pytest' not in sys.modules:
            self.state.load_preferences()
        logger.debug("UnifiedChatSidebar initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the unified sidebar with tabbed interface."""
        with Container(classes="sidebar-container"):
            # Header with collapse button
            with Horizontal(classes="sidebar-header"):
                yield Static("Chat Controls", classes="sidebar-title")
                yield Button("◀", id="sidebar-collapse", classes="collapse-btn")
            
            # Main tabbed content
            with TabbedContent(id="sidebar-tabs", initial=self.state.active_tab):
                with TabPane("Session", id="session"):
                    yield SessionTab(self.app_instance)
                
                with TabPane("Settings", id="settings"):
                    yield SettingsTab(self.app_instance)
                
                with TabPane("Content", id="content"):
                    yield ContentTab(self.app_instance)
    
    def on_mount(self):
        """Initialize sidebar on mount."""
        # Apply saved width
        self.styles.width = f"{self.state.sidebar_width}%"
        
        # Apply docking position (default is left, which is already in CSS)
        if self.state.sidebar_position == "right":
            self.styles.dock = "right"
            self.styles.border_right = None
            self.styles.border_left = "solid $primary-darken-2"
    
    def action_toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.add_class("collapsed")
        else:
            self.remove_class("collapsed")
    
    def action_switch_tab(self, tab_id: str):
        """Switch to specified tab."""
        try:
            tabs = self.query_one("#sidebar-tabs", TabbedContent)
            # Only switch if the tab exists
            if tab_id in ["session", "settings", "content"]:
                tabs.active = tab_id
                self.state.active_tab = tab_id
                # Only save preferences if not in test mode
                import sys
                if 'pytest' not in sys.modules:
                    self.state.save_preferences()
            else:
                logger.warning(f"Invalid tab ID: {tab_id}")
        except Exception as e:
            logger.error(f"Error switching tab: {e}")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses within the sidebar."""
        button_id = event.button.id
        
        if button_id == "sidebar-collapse":
            self.action_toggle_sidebar()
        elif button_id == "settings-advanced-toggle":
            await self._toggle_advanced_settings()
        elif button_id == "rag-toggle":
            await self._toggle_rag_settings()
        else:
            # Forward button press to the main app for handling
            # Map new IDs to old IDs for compatibility
            id_mapping = {
                "session-save-chat": "chat-save-current-chat-button",
                "session-new-chat": "chat-new-conversation-button",
                "session-clone-chat": "chat-clone-current-chat-button",
                "session-to-note": "chat-convert-to-note-button",
                "content-search-btn": "chat-media-search-button",
                "content-load": "chat-load-selected-note-button",
                "settings-rag-configure": "chat-rag-configure-button"
            }
            
            # If this is a mapped button, create a fake event with the old ID
            if button_id in id_mapping:
                old_id = id_mapping[button_id]
                # Post a message to the app with the old button ID
                from textual.message import Message
                class MappedButtonPress(Message):
                    def __init__(self, button_id: str):
                        super().__init__()
                        self.button_id = button_id
                
                # Find the chat window and trigger its handler
                try:
                    chat_window = self.app_instance.query_one("ChatWindowEnhanced")
                    # Create a fake button with the old ID for the event handler
                    fake_button = Button("", id=old_id)
                    fake_event = Button.Pressed(fake_button)
                    await chat_window.on_button_pressed(fake_event)
                except Exception as e:
                    logger.error(f"Error forwarding button press {button_id} -> {old_id}: {e}")
            else:
                # For non-mapped buttons, still forward to chat window handler
                # This ensures all sidebar buttons are properly handled
                try:
                    chat_window = self.app_instance.query_one("ChatWindowEnhanced")
                    await chat_window.on_button_pressed(event)
                except Exception as e:
                    logger.error(f"Error forwarding button press {button_id}: {e}")
    
    # ... [remainder of event handlers]
```

## Part 3: Critical Issues Analysis

### Architectural Problems

#### 1. The Button Handler Nightmare

**Problem**: Central dispatcher with 40+ button mappings
```python
button_handlers = {
    "send-stop-chat": self.handle_send_stop_button,
    "respond-for-me-button": chat_events.handle_respond_for_me_button_pressed,
    # ... 40+ more mappings
}
```

**Why it's bad**:
- Single point of failure
- Can't test buttons in isolation
- Adding features requires modifying core file
- No clear ownership of functionality

#### 2. State Management Chaos

**Problem**: State scattered everywhere
```python
self.pending_attachment = None  # Instance variable
pending_image = reactive(None)  # Reactive property
is_send_button = reactive(True)  # Another reactive
self.voice_input_widget = None  # Service state
# Plus UI element states, CSS classes, etc.
```

**Why it's bad**:
- No single source of truth
- State synchronization bugs
- Can't serialize/restore state
- Testing requires complex setup

#### 3. Mixed Responsibilities

**Problem**: ChatWindowEnhanced does everything
- UI rendering
- Business logic
- Event routing
- File I/O
- Service management
- State management

**Why it's bad**:
- 1000+ line files
- Can't reuse components
- Changes ripple everywhere
- Impossible to unit test

#### 4. Backward Compatibility Debt

**Problem**: Multiple systems for same functionality
```python
self.pending_attachment = None  # New system
self.pending_image = None  # Old system kept for compatibility
# Special test mode with hidden inputs
# ID mapping for legacy buttons
```

**Why it's bad**:
- Duplicate code paths
- Confusing for developers
- Bugs fixed in one place, not the other
- Testing complexity doubles

### What Can Be Salvaged

#### Good Patterns to Keep:

1. **Reactive Properties**
   - Work well for UI updates
   - Just need better organization

2. **Tabbed Interface** (from unified sidebar)
   - Good UX pattern
   - Clean separation of concerns

3. **State Persistence** (from ChatSidebarState)
   - Good pattern
   - Just needs to be applied everywhere

4. **Compound Widgets** (CompactField, SearchableList)
   - Reusable components
   - Good abstraction level

5. **File Filters**
   - Comprehensive file type support
   - Good user experience

#### Components Worth Refactoring:

1. **Voice Input**
   - Extract to standalone service
   - Make it a plugin

2. **File Attachment**
   - Separate attachment manager
   - Plugin-based handlers

3. **Settings Management**
   - Extract to settings service
   - Reactive updates to components

## Part 4: Rebuild Strategy

### New Architecture Proposal

```
ChatWindow (Container)
├── ChatStore (State Management)
│   ├── SessionState
│   ├── SettingsState
│   ├── AttachmentState
│   └── UIState
├── ChatEventBus (Event System)
│   ├── Publishers
│   └── Subscribers
├── ChatLayout (UI Structure)
│   ├── Sidebar
│   │   ├── SessionPanel
│   │   ├── SettingsPanel
│   │   └── ContentPanel
│   ├── MessageArea
│   │   ├── MessageList
│   │   └── MessageRenderer
│   └── InputArea
│       ├── TextInput
│       ├── AttachmentManager
│       └── ActionButtons
└── ChatServices (Business Logic)
    ├── MessageService
    ├── AttachmentService
    ├── VoiceService
    └── PersistenceService
```

### Key Principles for Rebuild

1. **Separation of Concerns**
   - UI components only render
   - Services handle business logic
   - Store manages state
   - Event bus handles communication

2. **Plugin Architecture**
   - Core functionality only
   - Everything else is a plugin
   - Clear extension points

3. **Testability First**
   - Every component testable in isolation
   - Mock-friendly interfaces
   - No global state

4. **Progressive Enhancement**
   - Start with basic chat
   - Add features as plugins
   - Graceful degradation

This analysis provides the foundation for a complete rebuild that addresses the architectural issues while preserving the good patterns and user experience elements.