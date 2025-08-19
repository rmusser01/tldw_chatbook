"""
Chat Attachment Handler Module

Handles all file attachment functionality including:
- File selection and validation
- Image processing and display
- PDF and document handling
- Attachment UI updates
- File processing workers
"""

import os
from typing import TYPE_CHECKING, Optional, Any
from pathlib import Path
from loguru import logger
from textual import work
from textual.widgets import Button, Static
from textual.worker import get_current_worker

if TYPE_CHECKING:
    from ..Chat_Window_Enhanced import ChatWindowEnhanced

logger = logger.bind(module="ChatAttachmentHandler")


class ChatAttachmentHandler:
    """Handles file attachments and image processing."""
    
    def __init__(self, chat_window: 'ChatWindowEnhanced'):
        """Initialize the attachment handler.
        
        Args:
            chat_window: Parent ChatWindowEnhanced instance
        """
        self.chat_window = chat_window
        self.app_instance = chat_window.app_instance
    
    async def handle_attach_image_button(self, event):
        """Show file picker dialog for attachments or legacy file input.
        
        Args:
            event: Button.Pressed event
        """
        # Check if we're in test mode with a mocked file input
        if self.chat_window._file_path_input:
            # Legacy mode for tests
            self.chat_window._file_path_input.styles.display = "block"
            self.chat_window._file_path_input.focus()
            return
        
        from fnmatch import fnmatch
        from ...Widgets.enhanced_file_picker import FileOpen, Filters
        
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
    
    async def handle_clear_image_button(self, event):
        """Clear attached file.
        
        Args:
            event: Button.Pressed event
        """
        # Clear all attachment data
        self.clear_attachment_state()
        self.app_instance.notify("File attachment cleared")
    
    async def process_file_attachment(self, file_path: str) -> None:
        """Process selected file using appropriate handler with worker pattern.
        
        Args:
            file_path: Path to the file to process
        """
        # Run file processing in a worker to prevent UI blocking
        self.chat_window.run_worker(
            self._process_file_worker,
            file_path,
            exclusive=True,  # Cancel any previous file processing
            name="file_processor"
        )
    
    @work(thread=True)
    def _process_file_worker(self, file_path: str) -> None:
        """Worker to process file attachment in background thread.
        
        Args:
            file_path: Path to the file to process
        """
        from ...Utils.file_handlers import file_handler_registry
        from ...Utils.path_validation import is_safe_path
        
        try:
            logger.info(f"Processing file attachment: {file_path}")
            
            # Validate the file path is safe (within user's home directory)
            if not is_safe_path(file_path, os.path.expanduser("~")):
                self.chat_window.call_from_thread(
                    self.app_instance.notify,
                    "File path is outside allowed directories",
                    severity="error"
                )
                self.chat_window.call_from_thread(self.clear_attachment_state)
                return
            
            # Check file exists
            if not os.path.exists(file_path):
                self.chat_window.call_from_thread(
                    self.app_instance.notify,
                    f"File not found: {file_path}",
                    severity="error"
                )
                self.chat_window.call_from_thread(self.clear_attachment_state)
                return
            
            # Get file size for validation
            file_size = os.path.getsize(file_path)
            max_size = 100 * 1024 * 1024  # 100MB limit
            if file_size > max_size:
                self.chat_window.call_from_thread(
                    self.app_instance.notify,
                    f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)",
                    severity="error"
                )
                self.chat_window.call_from_thread(self.clear_attachment_state)
                return
            
            # Process file using appropriate handler
            processed_file = file_handler_registry.process_file(file_path)
            
            # Update UI with processed file data
            self.chat_window.call_from_thread(
                self._handle_processed_file,
                processed_file
            )
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                f"File not found: {file_path}",
                severity="error"
            )
            self.chat_window.call_from_thread(self.clear_attachment_state)
        except PermissionError as e:
            logger.error(f"Permission denied accessing file: {e}")
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                "Permission denied accessing file",
                severity="error"
            )
            self.chat_window.call_from_thread(self.clear_attachment_state)
        except ValueError as e:
            logger.error(f"File validation error: {e}")
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                str(e),
                severity="error"
            )
            self.chat_window.call_from_thread(self.clear_attachment_state)
        except MemoryError as e:
            logger.error(f"Out of memory processing file: {file_path}")
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                "File too large to process",
                severity="error"
            )
            self.chat_window.call_from_thread(self.clear_attachment_state)
        except (IOError, OSError) as e:
            logger.error(f"File system error processing attachment: {e}", exc_info=True)
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                f"File system error: {str(e)}",
                severity="error"
            )
            self.chat_window.call_from_thread(self.clear_attachment_state)
        except Exception as e:
            # Keep generic catch as last resort for truly unexpected errors
            logger.critical(f"Unexpected error processing file attachment: {e}", exc_info=True)
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                "An unexpected error occurred",
                severity="error"
            )
            self.chat_window.call_from_thread(self.clear_attachment_state)
    
    def _handle_processed_file(self, processed_file: Any) -> None:
        """Handle the processed file data and update UI.
        
        Args:
            processed_file: Processed file data from file handler
        """
        try:
            if processed_file.insert_mode == "inline":
                # Insert text content into chat input
                chat_input = self.chat_window._chat_input
                if chat_input:
                    try:
                        # Insert at cursor or append
                        current_text = chat_input.value
                        new_text = current_text + "\n" + processed_file.content if current_text else processed_file.content
                        chat_input.value = new_text
                        
                        # Move cursor to end
                        try:
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
                        
                        # Check if model supports images for image files
                        if processed_file.file_type == "image":
                            try:
                                from ...model_capabilities import is_vision_capable
                                provider_widget = self.chat_window._provider_select
                                model_widget = self.chat_window._model_select
                                if not provider_widget or not model_widget:
                                    logger.warning("Provider or model widget not cached")
                                    # Fall back to query if needed
                                    provider_widget = self.app_instance.query_one("#chat-api-provider")
                                    model_widget = self.app_instance.query_one("#chat-api-model")
                                
                                from textual.widgets import Select
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
                            except ImportError:
                                logger.warning("model_capabilities module not available")
                        
                        self.app_instance.notify(f"{emoji} File content inserted: {Path(processed_file.path).name}")
                        
                    except AttributeError as e:
                        logger.error(f"Chat input widget not available: {e}")
                        self.app_instance.notify("Chat input not available", severity="error")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid file content or cursor position: {e}")
                        self.app_instance.notify(f"Failed to insert content: {str(e)}", severity="error")
                    except RuntimeError as e:
                        logger.error(f"Runtime error inserting content: {e}")
                        self.app_instance.notify("Failed to insert content", severity="error")
                        
            elif processed_file.insert_mode == "attachment":
                # Store as attachment
                session_id = self.app_instance.active_session_id or "default"
                
                # Store different data based on file type
                if processed_file.file_type == "image":
                    # Store image data
                    self.chat_window.pending_image = {
                        "path": processed_file.path,
                        "data": processed_file.content,
                        "mime_type": processed_file.mime_type
                    }
                    self.chat_window.pending_attachment = processed_file.path
                else:
                    # Store non-image attachment
                    self.chat_window.pending_attachment = processed_file.path
                
                # Add to app's attachment list
                if session_id not in self.app_instance.chat_attached_files:
                    self.app_instance.chat_attached_files[session_id] = []
                
                self.app_instance.chat_attached_files[session_id].append({
                    "path": processed_file.path,
                    "type": processed_file.file_type,
                    "content": processed_file.content if processed_file.file_type != "image" else None,
                    "mime_type": processed_file.mime_type
                })
                
                # Update UI
                self.update_attachment_ui()
                
                # Notify user
                file_name = Path(processed_file.path).name
                self.app_instance.notify(f"📎 Attached: {file_name}")
                
        except (AttributeError, KeyError) as e:
            logger.error(f"Invalid processed file structure: {e}")
            self.app_instance.notify("Invalid file data", severity="error")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data type or value in processed file: {e}")
            self.app_instance.notify(f"Failed to process file: {str(e)}", severity="error")
        except RuntimeError as e:
            logger.error(f"Runtime error handling processed file: {e}", exc_info=True)
            self.app_instance.notify("Failed to process file", severity="error")
    
    def clear_attachment_state(self):
        """Clear all attachment state."""
        self.chat_window.pending_image = None
        self.chat_window.pending_attachment = None
        
        # Clear from app's attachment list
        session_id = self.app_instance.active_session_id or "default"
        if session_id in self.app_instance.chat_attached_files:
            self.app_instance.chat_attached_files[session_id] = []
        
        # Update UI
        self.update_attachment_ui()
    
    def update_attachment_ui(self):
        """Update the attachment indicator UI."""
        if not self.chat_window._attachment_indicator:
            return
        
        try:
            indicator = self.chat_window._attachment_indicator
            
            if self.chat_window.pending_image or self.chat_window.pending_attachment:
                # Show attachment indicator
                file_path = self.chat_window.pending_attachment or (
                    self.chat_window.pending_image.get("path") if isinstance(self.chat_window.pending_image, dict) else None
                )
                if file_path:
                    file_name = Path(file_path).name
                    indicator.update(f"📎 {file_name}")
                    indicator.add_class("has-attachment")
                else:
                    indicator.update("")
                    indicator.remove_class("has-attachment")
            else:
                # Hide attachment indicator
                indicator.update("")
                indicator.remove_class("has-attachment")
                
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Could not update attachment indicator: {e}")