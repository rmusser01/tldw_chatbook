"""
Chat Attachment Handler Module

Handles all file attachment functionality including:
- File selection and validation
- Image processing and display
- PDF and document handling
- Attachment UI updates
- File processing workers
"""

import asyncio
from typing import TYPE_CHECKING, Optional, Any
from pathlib import Path
from loguru import logger

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
        # Check if we're in legacy/test mode with a mocked file input.
        file_path_input = getattr(self.chat_window, "_file_path_input", None)
        if file_path_input is None and not getattr(self.chat_window, "is_attached", False):
            try:
                file_path_input = self.chat_window.query_one("#image-file-path-input")
                self.chat_window._file_path_input = file_path_input
            except Exception:
                file_path_input = None

        if file_path_input:
            # Legacy mode for tests
            if hasattr(file_path_input, "remove_class"):
                file_path_input.remove_class("hidden")
            else:
                file_path_input.styles.display = "block"
            file_path_input.focus()
            return
        
        from fnmatch import fnmatch
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters
        
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
        
        from ...Chat.attachment_core import attachment_filter_specs

        file_filters = Filters(
            *[(label, create_filter(patterns)) for label, patterns in attachment_filter_specs()],
            ("All Files", lambda path: True),
        )
        
        # Push the picker directly — EnhancedFileOpen like every other picker
        # surface: the plain FileOpen re-export accepts no `context` kwarg and
        # raised TypeError the moment this branch was exercised (TASK-219).
        self.app_instance.push_screen(
            EnhancedFileOpen(
                location=".",
                title="Select File to Attach",
                filters=file_filters,
                context="chat_images",
            ),
            callback=on_file_selected,
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
        if self._can_use_worker_processing():
            # Run file processing in a worker to prevent UI blocking.
            self.chat_window.run_worker(
                lambda: self._process_file_worker(file_path),
                exclusive=True,
                name="file_processor",
                thread=True,
            )
            return

        try:
            processed_file = await self._load_processed_file(file_path)
            self._handle_processed_file(processed_file, file_path)
        except Exception as e:
            self._handle_processing_error(file_path, e)

    def _can_use_worker_processing(self) -> bool:
        """Return True when the chat window is attached to a live Textual app."""
        return bool(
            getattr(self.chat_window, "is_attached", False)
            and getattr(self.chat_window, "is_mounted", False)
        )

    def _get_session_id(self) -> str:
        """Return the current chat session id with a safe default."""
        session_id = getattr(self.app_instance, "active_session_id", None)
        return session_id if isinstance(session_id, str) and session_id else "default"

    def _get_chat_attached_files(self) -> dict[str, list[dict[str, Any]]]:
        """Return the app attachment map, creating a dict when mocks expose a placeholder."""
        attached_files = getattr(self.app_instance, "chat_attached_files", None)
        if not isinstance(attached_files, dict):
            attached_files = {}
            self.app_instance.chat_attached_files = attached_files
        return attached_files

    async def _load_processed_file(self, file_path: str) -> Any:
        """Validate and process a file attachment via the shared core."""
        from ...Chat.attachment_core import load_processed_file

        return await load_processed_file(file_path)
    
    def _process_file_worker(self, file_path: str) -> None:
        """Worker to process file attachment in background thread.
        
        Args:
            file_path: Path to the file to process
        """
        try:
            processed_file = asyncio.run(self._load_processed_file(file_path))
            self.chat_window.call_from_thread(
                self._handle_processed_file,
                processed_file,
                file_path,
            )
        except Exception as e:
            self.chat_window.call_from_thread(
                self._handle_processing_error,
                file_path,
                e,
            )
    
    def _handle_processing_error(self, file_path: str, error: Exception) -> None:
        """Handle attachment processing errors on the main thread."""
        file_name = Path(file_path).name

        if isinstance(error, FileNotFoundError):
            logger.error(f"File not found: {file_path}")
            self.app_instance.notify(f"File not found: {file_name}", severity="error")
        elif isinstance(error, PermissionError):
            logger.error(f"Permission denied accessing file: {file_path}")
            self.app_instance.notify(
                f"Permission denied: {file_name}",
                severity="error",
            )
        elif isinstance(error, ValueError):
            logger.error(f"File validation error: {error}")
            self.app_instance.notify(str(error), severity="error")
        elif isinstance(error, MemoryError):
            logger.error(f"Out of memory processing file: {file_path}")
            self.app_instance.notify("File too large to process", severity="error")
        elif isinstance(error, (IOError, OSError)):
            logger.opt(exception=True).error(
                f"File system error processing attachment: {error}",
            )
            self.app_instance.notify(
                f"File system error: {str(error)}",
                severity="error",
            )
        else:
            logger.opt(exception=True).critical(
                f"Unexpected error processing file attachment: {error}",
            )
            self.app_instance.notify("An unexpected error occurred", severity="error")

        self.clear_attachment_state()

    def _handle_processed_file(
        self,
        processed_file: Any,
        file_path: Optional[str] = None,
    ) -> None:
        """Handle the processed file data and update UI.
        
        Args:
            processed_file: Processed file data from file handler
            file_path: Original file path for metadata and notifications
        """
        try:
            resolved_path = file_path or getattr(processed_file, "path", "")
            display_name = getattr(
                processed_file,
                "display_name",
                Path(resolved_path).name if resolved_path else "attachment",
            )
            file_type = getattr(processed_file, "file_type", "file")
            attachment_data = getattr(processed_file, "attachment_data", None)
            attachment_mime_type = getattr(
                processed_file,
                "attachment_mime_type",
                getattr(processed_file, "mime_type", None),
            )

            if processed_file.insert_mode == "inline":
                # Insert text content into chat input
                chat_input = self.chat_window._get_chat_input()
                if chat_input:
                    try:
                        current_text = getattr(chat_input, "text", None)
                        if current_text is None:
                            current_text = getattr(chat_input, "value", "")

                        new_text = (
                            current_text + "\n\n" + processed_file.content
                            if current_text
                            else processed_file.content
                        )

                        if hasattr(chat_input, "load_text"):
                            chat_input.load_text(new_text)
                        if hasattr(chat_input, "text"):
                            chat_input.text = new_text
                        if hasattr(chat_input, "value"):
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
                        emoji = emoji_map.get(file_type, "📎")
                        
                        # Check if model supports images for image files
                        if file_type == "image":
                            try:
                                from ...model_capabilities import is_vision_capable
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
                            except Exception as vision_error:
                                logger.debug(f"Could not check vision capability: {vision_error}")
                        
                        self.app_instance.notify(f"{emoji} {display_name} content inserted")
                        
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
                self.chat_window.pending_attachment = {
                    "data": attachment_data,
                    "mime_type": attachment_mime_type,
                    "path": resolved_path,
                    "display_name": display_name,
                    "file_type": file_type,
                    "insert_mode": processed_file.insert_mode,
                }

                if file_type == "image":
                    self.chat_window.pending_image = {
                        "path": resolved_path,
                        "data": attachment_data,
                        "mime_type": attachment_mime_type,
                    }
                else:
                    self.chat_window.pending_image = None

                session_id = self._get_session_id()
                attached_files = self._get_chat_attached_files()
                if session_id not in attached_files:
                    attached_files[session_id] = []

                attached_files[session_id].append(
                    {
                        "path": resolved_path,
                        "type": file_type,
                        "content": processed_file.content if file_type != "image" else None,
                        "mime_type": attachment_mime_type,
                    }
                )
                
                # Update UI
                self.update_attachment_ui()
                
                # Notify user
                self.app_instance.notify(f"{display_name} attached")
                
        except (AttributeError, KeyError) as e:
            logger.error(f"Invalid processed file structure: {e}")
            self.app_instance.notify("Invalid file data", severity="error")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data type or value in processed file: {e}")
            self.app_instance.notify(f"Failed to process file: {str(e)}", severity="error")
        except RuntimeError as e:
            logger.opt(exception=True).error(f"Runtime error handling processed file: {e}")
            self.app_instance.notify("Failed to process file", severity="error")
    
    def clear_attachment_state(self):
        """Clear all attachment state."""
        self.chat_window.pending_image = None
        self.chat_window.pending_attachment = None
        
        # Clear from app's attachment list
        session_id = self._get_session_id()
        attached_files = self._get_chat_attached_files()
        if session_id in attached_files:
            attached_files[session_id] = []
        
        # Update UI
        self.update_attachment_ui()
    
    def update_attachment_ui(self):
        """Update the attachment indicator UI."""
        indicator = self.chat_window._get_attachment_indicator()
        try:
            attach_button = self.chat_window.query_one("#attach-image")
        except Exception:
            attach_button = None
        if not indicator:
            return
        
        try:
            
            if self.chat_window.pending_image or self.chat_window.pending_attachment:
                if attach_button is not None:
                    attach_button.label = "📎✓"
                # Show attachment indicator
                attachment = self.chat_window.pending_attachment
                file_path = (
                    attachment.get("path")
                    if isinstance(attachment, dict)
                    else self.chat_window.pending_image.get("path")
                    if isinstance(self.chat_window.pending_image, dict)
                    else None
                )
                if file_path:
                    display_name = (
                        attachment.get("display_name")
                        if isinstance(attachment, dict)
                        else Path(file_path).name
                    )
                    indicator.update(f"📎 {display_name}")
                    indicator.add_class("has-attachment")
                    indicator.remove_class("hidden")
                else:
                    indicator.update("")
                    indicator.remove_class("has-attachment")
            else:
                if attach_button is not None:
                    attach_button.label = "📎"
                # Hide attachment indicator
                indicator.update("")
                indicator.remove_class("has-attachment")
                indicator.add_class("hidden")
                
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Could not update attachment indicator: {e}")
