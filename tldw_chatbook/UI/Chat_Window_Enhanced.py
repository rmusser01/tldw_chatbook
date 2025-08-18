# Chat_Window_Enhanced.py
# Description: Enhanced Chat Window with image attachment support
#
# Imports
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
    pending_image = reactive(None, layout=False)
    
    # Track button state for Send/Stop functionality with automatic UI updates
    is_send_button = reactive(True, layout=False)
    
    # Debouncing for button clicks
    _last_send_stop_click = 0
    DEBOUNCE_MS = 300
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
        # Initialize cached widget references first (before reactive properties)
        self._send_button: Optional[Button] = None
        self._chat_input: Optional[TextArea] = None
        self._mic_button: Optional[Button] = None
        self._attach_button: Optional[Button] = None
        self._attachment_indicator: Optional[Static] = None
        self._chat_input_area: Optional[Horizontal] = None
        self._chat_log: Optional[VerticalScroll] = None
        self._provider_select: Optional[Select] = None
        self._model_select: Optional[Select] = None
        self._file_path_input: Optional[Input] = None  # For test mode
        
        # Now set reactive properties (which may trigger watchers)
        self.pending_attachment = None  # New unified attachment system
        self.pending_image = None  # Deprecated - kept for backward compatibility
        
        # Voice input state
        self.voice_input_widget: Optional[VoiceInputWidget] = None
        self.is_voice_recording = False
        
        logger.debug("ChatWindowEnhanced initialized.")
    
    async def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Cache frequently accessed widgets for performance
        self._cache_widgets()
        
        # Token counter will be initialized when tab is switched to chat
        # Watch for streaming state changes
        self._update_button_state()
        # REMOVED: Periodic polling was causing performance issues
        # Button state will be updated on-demand when streaming state actually changes
    
    def _cache_widgets(self) -> None:
        """Cache frequently accessed widgets to avoid repeated DOM queries."""
        from textual.css.query import NoMatches
        
        # Cache core widgets
        try:
            self._send_button = self.query_one("#send-stop-chat", Button)
        except NoMatches:
            logger.debug("Send button not found during caching")
            self._send_button = None
        
        try:
            self._chat_input = self.query_one("#chat-input", TextArea)
        except NoMatches:
            logger.debug("Chat input not found during caching")
            self._chat_input = None
        
        try:
            self._mic_button = self.query_one("#mic-button", Button)
        except NoMatches:
            logger.debug("Mic button not found during caching")
            self._mic_button = None
        
        try:
            self._attach_button = self.query_one("#attach-image", Button)
        except NoMatches:
            logger.debug("Attach button not found during caching")
            self._attach_button = None
        
        try:
            self._attachment_indicator = self.query_one("#image-attachment-indicator", Static)
        except NoMatches:
            logger.debug("Attachment indicator not found during caching")
            self._attachment_indicator = None
        
        try:
            self._chat_input_area = self.query_one("#chat-input-area", Horizontal)
        except NoMatches:
            logger.debug("Chat input area not found during caching")
            self._chat_input_area = None
        
        # Cache app-level widgets
        try:
            self._chat_log = self.app_instance.query_one("#chat-log", VerticalScroll)
        except NoMatches:
            logger.debug("Chat log not found during caching")
            self._chat_log = None
        
        try:
            self._provider_select = self.app_instance.query_one("#chat-api-provider", Select)
        except NoMatches:
            logger.debug("Provider select not found during caching")
            self._provider_select = None
        
        try:
            self._model_select = self.app_instance.query_one("#chat-api-model", Select)
        except NoMatches:
            logger.debug("Model select not found during caching")
            self._model_select = None
        
        # Cache test mode widget
        try:
            self._file_path_input = self.query_one("#file-path-input", Input)
        except NoMatches:
            # This is expected in normal mode
            self._file_path_input = None
        
        logger.debug(f"Widget caching complete. Cached: send={bool(self._send_button)}, "
                    f"input={bool(self._chat_input)}, mic={bool(self._mic_button)}, "
                    f"attach={bool(self._attach_button)}")

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
        enable_tabs = get_cli_setting("chat_defaults", "enable_tabs", False)
        if enable_tabs and hasattr(self, 'tab_container'):
            # Tab-specific buttons have session IDs appended
            for session_id in self.tab_container.sessions.keys():
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
        """Show file picker dialog for attachments or legacy file input."""
        # Check if we're in test mode with a mocked file input
        if self._file_path_input:
            # Legacy mode for tests
            self._file_path_input.styles.display = "block"
            self._file_path_input.focus()
            return
        
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
        """Process selected file using appropriate handler with worker pattern."""
        # Run file processing in a worker to prevent UI blocking
        self.run_worker(
            self._process_file_worker,
            file_path,
            exclusive=True,  # Cancel any previous file processing
            name="file_processor"
        )
    
    @work(thread=True)
    def _process_file_worker(self, file_path: str) -> None:
        """Worker to process file attachment in background thread."""
        from ..Utils.file_handlers import file_handler_registry
        from ..Utils.path_validation import is_safe_path
        from pathlib import Path
        import os
        
        try:
            logger.info(f"Processing file attachment: {file_path}")
            
            # Validate the file path is safe (within user's home directory)
            if not is_safe_path(file_path, os.path.expanduser("~")):
                self.call_from_thread(
                    self.app_instance.notify,
                    "File path is outside allowed directory",
                    severity="error"
                )
                return
            
            # Process the file synchronously (required for thread workers)
            # Note: If file_handler_registry.process_file is async, we need to use the sync version
            # or run it in an event loop
            import asyncio
            import inspect
            
            if inspect.iscoroutinefunction(file_handler_registry.process_file):
                # If it's async, we need to run it in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    processed_file = loop.run_until_complete(file_handler_registry.process_file(file_path))
                finally:
                    loop.close()
            else:
                # If it's already sync, just call it
                processed_file = file_handler_registry.process_file(file_path)
            logger.info(f"File processed successfully: {processed_file}")
            
            # Update UI from worker thread
            self.call_from_thread(self._handle_processed_file, processed_file, file_path)
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            self.call_from_thread(
                self.app_instance.notify,
                f"File not found: {Path(file_path).name}",
                severity="error"
            )
            self.call_from_thread(self._clear_attachment_state)
        except PermissionError as e:
            logger.error(f"Permission denied accessing file: {file_path}")
            self.call_from_thread(
                self.app_instance.notify,
                f"Permission denied: {Path(file_path).name}",
                severity="error"
            )
            self.call_from_thread(self._clear_attachment_state)
        except ValueError as e:
            logger.error(f"File validation error: {e}")
            self.call_from_thread(
                self.app_instance.notify,
                str(e),
                severity="error"
            )
            self.call_from_thread(self._clear_attachment_state)
        except MemoryError as e:
            logger.error(f"Out of memory processing file: {file_path}")
            self.call_from_thread(
                self.app_instance.notify,
                "File too large to process",
                severity="error"
            )
            self.call_from_thread(self._clear_attachment_state)
        except (IOError, OSError) as e:
            logger.error(f"File system error processing attachment: {e}", exc_info=True)
            self.call_from_thread(
                self.app_instance.notify,
                f"File system error: {str(e)}",
                severity="error"
            )
            self.call_from_thread(self._clear_attachment_state)
        except Exception as e:
            # Keep generic catch as last resort for truly unexpected errors
            logger.critical(f"Unexpected error processing file attachment: {e}", exc_info=True)
            self.call_from_thread(
                self.app_instance.notify,
                "An unexpected error occurred",
                severity="error"
            )
            self.call_from_thread(self._clear_attachment_state)
            self.call_from_thread(
                self.app_instance.notify,
                f"Error processing file: {str(e)}",
                severity="error"
            )
            self.call_from_thread(self._clear_attachment_state)
    
    def _handle_processed_file(self, processed_file, file_path: str) -> None:
        """Handle the processed file result on the main thread."""
        try:
            if processed_file.insert_mode == "inline":
                # For text/code/data files, insert content directly into chat input
                try:
                    logger.info("Attempting to insert inline content")
                    chat_input = self._chat_input
                    if not chat_input:
                        logger.warning("Chat input widget not cached")
                        return
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
                    except (IndexError, ValueError) as cursor_error:
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
                    
                except AttributeError as e:
                    logger.error(f"Chat input widget not available: {e}")
                    self.app_instance.notify("Chat input not available", severity="error")
                except Exception as e:
                    logger.error(f"Failed to insert file content: {e}", exc_info=True)
                    self.app_instance.notify("Failed to insert content", severity="error")
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
                        provider_widget = self._provider_select
                        model_widget = self._model_select
                        if not provider_widget or not model_widget:
                            logger.warning("Provider or model widget not cached")
                            # Fall back to query if needed
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
                    except (ImportError, AttributeError) as e:
                        logger.debug(f"Could not check vision capability: {e}")
                
                # Use centralized UI update
                self._update_attachment_ui()
                
                self.app_instance.notify(f"{processed_file.display_name} attached")
        except (AttributeError, KeyError) as e:
            logger.error(f"Invalid processed file structure: {e}")
            self.app_instance.notify("Invalid file data", severity="error")
        except Exception as e:
            logger.error(f"Error handling processed file: {e}", exc_info=True)
            self.app_instance.notify("Failed to process file", severity="error")
            self.app_instance.notify(f"Error: {str(e)}", severity="error")

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
                    id="image-attachment-indicator"
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
        """Clear all attachment state and update UI consistently."""
        # Clear data - reactive watchers will handle UI updates
        self.pending_image = None
        self.pending_attachment = None
        
        # Update attach button
        if self._attach_button:
            try:
                self._attach_button.label = "📎"
            except AttributeError:
                # Widget might not exist yet
                pass
        
        # Hide indicator using CSS class
        if self._attachment_indicator:
            try:
                self._attachment_indicator.remove_class("visible")
            except AttributeError:
                # Widget might not exist yet
                pass
    
    def _update_attachment_ui(self):
        """Update UI elements based on current attachment state."""
        if self._attach_button:
            # Update attach button appearance based on attachment state
            if self.pending_image or self.pending_attachment:
                # Show indicator that file is attached
                self._attach_button.label = "📎✓"
                
                # Update indicator visibility and text
                if self._attachment_indicator:
                    
                    if self.pending_attachment:
                        # For new unified attachment system
                        display_name = self.pending_attachment.get('display_name', 'File')
                        file_type = self.pending_attachment.get('file_type', 'file')
                        emoji_map = {"image": "📷", "file": "📎", "code": "💻", "text": "📄", "data": "📊"}
                        emoji = emoji_map.get(file_type, "📎")
                        self._attachment_indicator.update(f"{emoji} {display_name}")
                    elif self.pending_image:
                        # For legacy image system
                        if isinstance(self.pending_image, dict):
                            # Extract filename from path if available
                            path = self.pending_image.get('path', '')
                            if path:
                                from pathlib import Path
                                filename = Path(path).name
                                self._attachment_indicator.update(f"📷 {filename}")
                            else:
                                self._attachment_indicator.update("📷 Image attached")
                        else:
                            self._attachment_indicator.update("📷 Image attached")
                    
                    # Show indicator using CSS class
                    self._attachment_indicator.add_class("visible")
            else:
                # No attachment - reset to default
                self._attach_button.label = "📎"
                
                # Hide indicator using CSS class
                if self._attachment_indicator:
                    try:
                        self._attachment_indicator.remove_class("visible")
                    except Exception:
                        pass
    
    async def toggle_attach_button_visibility(self, show: bool) -> None:
        """Toggle the visibility of the attach file button."""
        try:
            if show:
                # Check if button already exists
                if self._attach_button:
                    # Button already exists, no need to add
                    return
                
                # Find the input area and add the button
                input_area = self._chat_input_area
                send_button = self._send_button
                if not input_area or not send_button:
                    logger.warning("Input area or send button not cached")
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
                if self._attach_button:
                    try:
                        await self._attach_button.remove()
                        self._attach_button = None  # Clear cache reference
                        # Clear attachment state when hiding the button
                        self._clear_attachment_state()
                    except Exception:
                        # Button doesn't exist, nothing to remove
                        pass
                    
        except (AttributeError, RuntimeError) as e:
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
            
        except (AttributeError, KeyError) as e:
            logger.error(f"Error handling notes expand button - widget not found: {e}")
    
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
            chat_log = self._chat_log
            if not chat_log:
                logger.debug("Chat log not cached")
                return
            
            # Find the focused widget
            focused_widget = self.app_instance.focused
            
            # Check if the focused widget is a ChatMessage or if we need to find one
            from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage
            from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
            
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
            except (AttributeError, NoMatches) as e:
                logger.debug(f"Could not find or click edit button: {e}")
                
        except NoMatches as e:
            logger.debug(f"No message widget found to edit: {e}")
        except AttributeError as e:
            logger.error(f"Error in edit_focused_message action: {e}")
            self.app_instance.notify("Could not enter edit mode", severity="warning")
    
    def _update_button_state(self) -> None:
        """Update the send/stop button based on streaming state."""
        try:
            is_streaming = self.app_instance.get_current_chat_is_streaming()
            has_worker = (hasattr(self.app_instance, 'current_chat_worker') and 
                         self.app_instance.current_chat_worker and 
                         self.app_instance.current_chat_worker.is_running)
            
            # Update reactive property - watcher will handle UI update
            new_state = not (is_streaming or has_worker)
            if self.is_send_button != new_state:
                self.is_send_button = new_state
        except Exception as e:
            logger.debug(f"Error updating button state: {e}")
    
    def watch_is_send_button(self, is_send: bool) -> None:
        """Watch for changes to button state and update UI accordingly."""
        try:
            button = self._send_button
            if not button:
                logger.debug("Send button not cached in watcher")
                return
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
        except Exception as e:
            logger.debug(f"Could not update button in watcher: {e}")
    
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
        """Unified handler for Send/Stop button with debouncing and error recovery."""
        from ..Event_Handlers.Chat_Events import chat_events
        import time
        
        current_time = time.time() * 1000
        
        # Debounce rapid clicks
        if current_time - self._last_send_stop_click < self.DEBOUNCE_MS:
            logger.debug("Button click debounced", extra={"time_diff": current_time - self._last_send_stop_click})
            return
        self._last_send_stop_click = current_time
        
        # Disable button during operation
        button = self._send_button
        if button:
            try:
                button.disabled = True
            except Exception as e:
                logger.warning(f"Could not disable send/stop button: {e}")
        
        try:
            # Check current state and route to appropriate handler
            if self.app_instance.get_current_chat_is_streaming() or (
                hasattr(self.app_instance, 'current_chat_worker') and 
                self.app_instance.current_chat_worker and 
                self.app_instance.current_chat_worker.is_running
            ):
                # Stop operation
                logger.info("Send/Stop button pressed - stopping generation", 
                          extra={"action": "stop", "is_streaming": self.app_instance.get_current_chat_is_streaming()})
                await chat_events.handle_stop_chat_generation_pressed(app_instance, event)
            else:
                # Send operation - use enhanced handler that includes image
                logger.info("Send/Stop button pressed - sending message", 
                          extra={"action": "send", "has_attachment": bool(self.pending_attachment or self.pending_image)})
                await self.handle_enhanced_send_button(app_instance, event)
        except Exception as e:
            logger.error(f"Error handling send/stop button: {e}", exc_info=True, 
                        extra={"button_state": "send" if self.is_send_button else "stop"})
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
        finally:
            # Re-enable button and update state after operation
            if button:
                try:
                    button.disabled = False
                except Exception as e:
                    logger.warning(f"Could not re-enable send/stop button: {e}")
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
    
    @work(thread=True)
    def _start_voice_recording_worker(self):
        """Start voice recording in a worker thread."""
        try:
            from ..Audio.dictation_service_lazy import AudioInitializationError
            
            # Start dictation (should be synchronous for thread workers)
            success = self.voice_dictation_service.start_dictation(
                on_partial_transcript=self._on_voice_partial,
                on_final_transcript=self._on_voice_final,
                on_error=self._on_voice_error
            )
            
            if success:
                self.call_from_thread(self._on_voice_recording_started)
            else:
                self.call_from_thread(
                    self.app_instance.notify,
                    "Failed to start recording",
                    severity="error"
                )
                self.call_from_thread(self._reset_mic_button)
                
        except AudioInitializationError as e:
            logger.error(f"Audio initialization error: {e}", extra={"error_type": "audio_init"})
            self.call_from_thread(
                self.app_instance.notify,
                str(e),
                severity="error",
                timeout=10
            )
            self.call_from_thread(self._reset_mic_button)
        except Exception as e:
            logger.error(f"Error starting voice recording: {e}", extra={"error_type": "voice_recording"})
            error_msg = self._get_voice_error_message(e)
            self.call_from_thread(
                self.app_instance.notify,
                error_msg,
                severity="error",
                timeout=10 if "permission" in error_msg.lower() else 5
            )
            self.call_from_thread(self._reset_mic_button)
    
    def _start_voice_recording(self):
        """Start voice recording with proper worker management."""
        try:
            # Update UI immediately
            if self._mic_button:
                self._mic_button.label = "🛑"  # Stop icon
                self._mic_button.variant = "error"
            
            # Run recording in worker
            self.run_worker(
                self._start_voice_recording_worker,
                exclusive=True,
                name="voice_recorder"
            )
        except Exception as e:
            logger.error(f"Failed to start voice recording worker: {e}")
            self._reset_mic_button()
    
    def _on_voice_recording_started(self):
        """Handle successful voice recording start."""
        self.is_voice_recording = True
        self.app_instance.notify("🎤 Listening...", timeout=2)
    
    def _reset_mic_button(self):
        """Reset microphone button to default state."""
        if self._mic_button:
            try:
                self._mic_button.label = "🎤"
                self._mic_button.variant = "default"
            except AttributeError:
                # Widget might not exist yet
                pass
    
    def _get_voice_error_message(self, error: Exception) -> str:
        """Get user-friendly error message for voice recording errors."""
        error_str = str(error).lower()
        if "no default" in error_str or "invalid input device" in error_str:
            return "No microphone access. Grant permissions in System Settings > Privacy > Microphone"
        elif "permission" in error_str:
            return "Microphone permission denied. Please check your system settings."
        elif "audio" in error_str:
            return "Audio system error. Please check your microphone is connected."
        else:
            return f"Voice recording error: {str(error)}"
    
    def _stop_voice_recording(self):
        """Stop voice recording."""
        try:
            # Stop dictation
            result = self.voice_dictation_service.stop_dictation()
            
            # Update UI
            if self._mic_button:
                self._mic_button.label = "🎤"
                self._mic_button.variant = "default"
            
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
        if self._mic_button:
            try:
                self._mic_button.label = "🎤"
                self._mic_button.variant = "default"
            except:
                pass
        self.is_voice_recording = False
    
    def _insert_voice_text(self, text: str):
        """Insert voice text into chat input."""
        try:
            if not self._chat_input:
                logger.warning("Chat input widget not cached for voice text")
                return
            current_text = self._chat_input.text
            
            # Add space if there's existing text
            if current_text and not current_text.endswith(' '):
                text = ' ' + text
            
            # Append transcribed text
            self._chat_input.load_text(current_text + text)
            
            # Focus the input
            self._chat_input.focus()
        except Exception as e:
            logger.error(f"Failed to insert voice text: {e}")
    
    def on_voice_input_message(self, event: VoiceInputMessage) -> None:
        """Handle voice input messages."""
        if event.is_final and event.text:
            # Add transcribed text to chat input
            try:
                if not self._chat_input:
                    logger.warning("Chat input widget not cached")
                    return
                current_text = self._chat_input.text
                
                # Add space if there's existing text
                if current_text and not current_text.endswith(' '):
                    event.text = ' ' + event.text
                
                # Append transcribed text
                self._chat_input.load_text(current_text + event.text)
                
                # Focus the input
                self._chat_input.focus()
            except Exception as e:
                logger.error(f"Failed to add voice input to chat: {e}")

#
# End of Chat_Window_Enhanced.py
#######################################################################################################################