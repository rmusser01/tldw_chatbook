# tldw_chatbook/Widgets/chat_message_enhanced.py
# Description: Enhanced ChatMessage widget with image support
#
# Imports
#
# Standard Library
import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

# 3rd-party Libraries
from PIL import Image as PILImage
from rich_pixels import Pixels
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import QueryError
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Static, Markdown

# Optional import for textual-image
try:
    from textual_image.widget import Image as TextualImage
    TEXTUAL_IMAGE_AVAILABLE = True
except ImportError:
    TEXTUAL_IMAGE_AVAILABLE = False
    logging.info("textual-image not available, will use rich-pixels fallback")

#
# Local Imports
from tldw_chatbook.Utils.file_extraction import FileExtractor
#
#######################################################################################################################
#
# Functions:

class ChatMessageEnhanced(Widget):
    """Enhanced chat message widget with image support."""
    
    class Action(Message):
        """Posted when a button on the message is pressed."""
        def __init__(self, message_widget: "ChatMessageEnhanced", button: Button) -> None:
            super().__init__()
            self.message_widget = message_widget
            self.button = button
    
    DEFAULT_CSS = """
    ChatMessageEnhanced {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    ChatMessageEnhanced > Vertical {
        border: round $surface;
        background: $panel;
        padding: 0 1;
        width: 100%;
        height: auto;
    }
    ChatMessageEnhanced.-user > Vertical {
        background: $boost;
        border: round $accent;
    }
    .message-header {
        width: 100%;
        padding: 0 1;
        background: $surface-darken-1;
        text-style: bold;
    }
    .message-text {
        padding: 1;
        width: 100%;
        height: auto;
    }
    .message-text Markdown {
        width: 100%;
        background: transparent;
        margin: 0;
        padding: 0;
    }
    .message-image-container {
        width: 100%;
        padding: 1;
        margin: 1 0;
    }
    .message-image {
        margin: 1 0;
        align: center middle;
        min-height: 5;
    }
    .image-controls {
        layout: horizontal;
        height: 3;
        margin-top: 1;
        align: center middle;
    }
    .image-controls Button {
        min-width: 12;
        height: 3;
        margin: 0 1;
    }
    .message-actions {
        height: auto;
        width: 100%;
        padding: 0 1;
        border-top: solid $surface-lighten-1;
        align: right middle;
        display: block;
    }
    .message-actions Button {
        min-width: 8;
        height: 1;
        margin: 0 0 0 1;
        border: none;
        background: $surface-lighten-2;
        color: $text-muted;
    }
    .message-actions Button:hover {
        background: $surface;
        color: $text;
    }
    .message-actions .delete-button:hover {
        background: $error;
        color: white;
    }
    ChatMessageEnhanced.-ai .message-actions.-generating {
        display: none;
    }
    .message-text.tts-generating {
        border-left: solid $accent;
        padding-left: 1;
    }
    .tts-generating-button {
        opacity: 0.8;
    }
    .tts-play-button, .tts-pause-button {
        background: $success-darken-1;
    }
    .tts-play-button:hover, .tts-pause-button:hover {
        background: $success;
    }
    .tts-save-button {
        background: $primary-darken-1;
    }
    .tts-save-button:hover {
        background: $primary;
    }
    .tts-stop-button {
        background: $error-darken-1;
    }
    .tts-stop-button:hover {
        background: $error;
    }
    .variant-navigation {
        layout: horizontal;
        height: 3;
        width: 100%;
        padding: 0 1;
        align: center middle;
        background: $surface-darken-2;
        border-top: solid $surface-lighten-1;
        border-bottom: solid $surface-lighten-1;
    }
    .variant-nav-button {
        min-width: 3;
        height: 3;
        margin: 0 1;
    }
    .variant-indicator {
        margin: 0 2;
        color: $text-muted;
    }
    .variant-select-button {
        margin: 0 1 0 2;
        background: $success-darken-1;
    }
    .variant-select-button:hover {
        background: $success;
    }
    """
    
    # Reactive properties
    message_text = reactive("")  # Remove repaint=True to prevent double rendering during streaming
    role = reactive("User", repaint=True)
    pixel_mode = reactive(False)
    _generation_complete_internal = reactive(True)
    
    # Internal state
    message_id_internal: reactive[Optional[str]] = reactive(None)
    message_version_internal: reactive[Optional[int]] = reactive(None)
    timestamp: reactive[Optional[str]] = reactive(None)
    image_data: reactive[Optional[bytes]] = reactive(None)
    image_mime_type: reactive[Optional[str]] = reactive(None)
    # Store feedback (thumbs up/down)
    feedback: reactive[Optional[str]] = reactive(None)
    # Store extracted files
    _extracted_files = None
    _file_extractor = None
    # TTS state tracking
    tts_state: reactive[str] = reactive("idle")  # "idle", "generating", "ready", "playing", "paused"
    tts_audio_file: reactive[Optional[Path]] = reactive(None)
    tts_progress: reactive[float] = reactive(0.0)
    
    # Variant tracking
    variant_of: reactive[Optional[str]] = reactive(None)  # Original message ID if this is a variant
    variant_id: reactive[Optional[str]] = reactive(None)  # This variant's ID
    variant_number: reactive[int] = reactive(1)  # Position in variant list
    total_variants: reactive[int] = reactive(1)  # Total number of variants
    is_selected_variant: reactive[bool] = reactive(True)  # Whether this variant is selected
    has_variants: reactive[bool] = reactive(False)  # Whether this message has variants
    
    def __init__(
        self,
        message: str,
        role: str,
        generation_complete: bool = True,
        message_id: Optional[str] = None,
        message_version: Optional[int] = None,
        timestamp: Optional[str] = None,
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
        feedback: Optional[str] = None,
        sender: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.message_text = message
        self.role = role
        self._generation_complete_internal = generation_complete
        self.message_id_internal = message_id
        self.message_version_internal = message_version
        self.timestamp = timestamp
        self.image_data = image_data
        self.image_mime_type = image_mime_type
        self.feedback = feedback
        self._image_widget = None
        
        # Add role-specific class
        # User messages have sender="User" even if role is the username
        if (sender and sender.lower() == "user") or role.lower() == "user":
            self.add_class("-user")
        else:
            self.add_class("-ai")
    
    @property
    def generation_complete(self) -> bool:
        """Public property to access the generation status."""
        return self._generation_complete_internal
    
    def compose(self) -> ComposeResult:
        with Vertical():
            # Message header
            header_text = f"{self.role}"
            if self.timestamp:
                header_text += f" - {self.timestamp}"
            yield Label(header_text, classes="message-header")
            
            # Variant navigation (only for AI messages with variants)
            if self.has_class("-ai") and (self.has_variants or self.total_variants > 1):
                with Horizontal(classes="variant-navigation"):
                    yield Button("◀", id="prev-variant", classes="variant-nav-button", 
                               disabled=(self.variant_number <= 1),
                               tooltip="Previous variant")
                    yield Label(f"Response {self.variant_number} of {self.total_variants}", 
                              classes="variant-indicator")
                    yield Button("▶", id="next-variant", classes="variant-nav-button",
                               disabled=(self.variant_number >= self.total_variants),
                               tooltip="Next variant")
                    if not self.is_selected_variant:
                        yield Button("✓ Use This", id="select-variant", classes="variant-select-button",
                                   tooltip="Use this response to continue the conversation")
            
            # Message content
            yield Markdown(self.message_text, classes="message-text")
            
            # Image display if present
            if self.image_data:
                with Container(classes="message-image-container"):
                    # Create image display widget
                    self._image_widget = Container(classes="message-image")
                    yield self._image_widget
                    
                    # Image controls
                    with Container(classes="image-controls"):
                        yield Button("Toggle View", id="toggle-image-mode", tooltip="Switch between image display modes")
                        yield Button("Save Image", id="save-image", tooltip="Save image to Downloads")
            
            # Action buttons
            actions_class = "message-actions"
            if self.has_class("-ai") and not self.generation_complete:
                actions_class += " -generating"
            
            with Horizontal(classes=actions_class) as actions_bar:
                actions_bar.id = f"actions-bar-{self.id or self.message_id_internal or 'new'}"
                # Common buttons
                yield Button("Edit", classes="action-button edit-button", tooltip="Edit message")
                yield Button("📋", classes="action-button copy-button", id="copy", tooltip="Copy message to clipboard")
                yield Button("📝", classes="action-button note-button", id="create-note", tooltip="Create note from message")
                
                # Add file extraction button if files detected
                if self._extracted_files is None:
                    self._check_for_files()
                if self._extracted_files:
                    file_count = len(self._extracted_files)
                    yield Button(f"📎 {file_count}", classes="action-button file-extract-button", 
                               id="extract-files", 
                               tooltip=f"Extract {file_count} file{'s' if file_count > 1 else ''} from message")
                
                # TTS buttons based on state
                if self.tts_state == "idle":
                    yield Button("🔊", classes="action-button speak-button", id="speak", tooltip="Read message aloud")
                elif self.tts_state == "generating":
                    yield Button("⏳", classes="action-button tts-generating-button", id="tts-generating", 
                               tooltip=f"Generating audio... {self.tts_progress:.0%}", disabled=True)
                elif self.tts_state in ["ready", "paused"]:
                    yield Button("▶️", classes="action-button tts-play-button", id="tts-play", tooltip="Play audio")
                    yield Button("💾", classes="action-button tts-save-button", id="tts-save", tooltip="Save audio")
                    yield Button("⏹️", classes="action-button tts-stop-button", id="tts-stop", tooltip="Stop and clear audio")
                elif self.tts_state == "playing":
                    yield Button("⏸️", classes="action-button tts-pause-button", id="tts-pause", tooltip="Pause audio")
                    yield Button("💾", classes="action-button tts-save-button", id="tts-save", tooltip="Save audio")
                    yield Button("⏹️", classes="action-button tts-stop-button", id="tts-stop", tooltip="Stop and clear audio")
                
                # AI-specific buttons
                if self.has_class("-ai"):
                    # Display feedback state on thumb buttons
                    thumb_up_label = "👍✓" if self.feedback == "1;" else "👍"
                    thumb_down_label = "👎✓" if self.feedback == "2;" else "👎"
                    yield Button(thumb_up_label, classes="action-button thumb-up-button", id="thumb-up", tooltip="Mark as helpful")
                    yield Button(thumb_down_label, classes="action-button thumb-down-button", id="thumb-down", tooltip="Mark as unhelpful")
                    yield Button("🔄", classes="action-button regenerate-button", id="regenerate", tooltip="Regenerate response")
                    yield Button("↪️", id="continue-response-button", classes="action-button continue-button", tooltip="Continue response")
                    yield Button("💡", classes="action-button suggest-response-button", id="suggest-response", tooltip="Suggest a response")
                
                # Delete button
                yield Button("🗑️", classes="action-button delete-button", tooltip="Delete message")
    
    def on_mount(self) -> None:
        """Render image when widget is mounted."""
        if self.image_data and self._image_widget:
            self._render_image()
        # Ensure initial state of buttons
        self.watch__generation_complete_internal(self._generation_complete_internal)
    
    def _render_image(self) -> None:
        """Render image based on current mode."""
        if not self._image_widget or not self.image_data:
            return
        
        # Clear existing content
        self._image_widget.remove_children()
        
        try:
            if self.pixel_mode:
                # Render with rich-pixels
                self._render_pixelated()
            else:
                # Render with textual-image or fallback
                self._render_regular()
        except Exception as e:
            logging.error(f"Error rendering image: {e}")
            # Use call_after_refresh for error mounting too
            self.call_after_refresh(
                lambda: self._image_widget.mount(
                    Static(f"[red]Error rendering image: {e}[/red]")
                ) if self._image_widget else None
            )
    
    def _render_pixelated(self) -> None:
        """Render image using rich-pixels."""
        try:
            # Convert bytes to PIL Image
            image_buffer = BytesIO(self.image_data)
            pil_image = PILImage.open(image_buffer)
            
            # Resize for terminal display if needed
            max_width = 80  # characters
            max_height = 40  # lines
            
            # Calculate scaling
            width_ratio = max_width / pil_image.width
            height_ratio = max_height / pil_image.height
            ratio = min(width_ratio, height_ratio, 1.0)
            
            if ratio < 1.0:
                new_size = (
                    int(pil_image.width * ratio),
                    int(pil_image.height * ratio)
                )
                pil_image = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
            
            # Create Pixels renderable directly from PIL image
            pixels = Pixels.from_image(pil_image)
            self._image_widget.mount(Static(pixels))
        except Exception as e:
            logging.error(f"Error in pixelated rendering: {e}")
            self._render_fallback()
    
    def _render_regular(self) -> None:
        """Render image using textual-image or fallback."""
        if TEXTUAL_IMAGE_AVAILABLE:
            try:
                # TextualImage expects a PIL image, not raw bytes
                pil_image = PILImage.open(BytesIO(self.image_data))
                image = TextualImage(pil_image)
                # Defer the mounting to ensure proper widget tree attachment
                self.call_after_refresh(lambda: self._mount_textual_image(image))
            except Exception as e:
                logging.error(f"Error with textual-image rendering: {e}")
                self._render_fallback()
        else:
            self._render_fallback()
    
    def _mount_textual_image(self, image: TextualImage) -> None:
        """Mount the TextualImage widget after the widget tree is stable."""
        try:
            if self._image_widget and not image.parent:
                self._image_widget.mount(image)
        except Exception as e:
            logging.error(f"Error mounting textual-image: {e}")
            if self._image_widget:
                self._render_fallback()
    
    def _render_fallback(self) -> None:
        """Fallback rendering for unsupported terminals."""
        # Display image info and base64 preview
        image_size = len(self.image_data) / 1024  # KB
        preview = base64.b64encode(self.image_data[:100]).decode()[:50]
        
        fallback_text = f"""[dim]
📷 Image ({self.image_mime_type or 'unknown'})
Size: {image_size:.1f} KB
Preview: {preview}...
\\[Click "Save Image" to view in external viewer\\]
[/dim]"""
        
        self._image_widget.mount(Static(fallback_text))
    
    def watch_pixel_mode(self, pixel_mode: bool) -> None:
        """Handle pixel mode changes."""
        if self.image_data and self._image_widget:
            self._render_image()
    
    def watch_tts_state(self, old_state: str, new_state: str) -> None:
        """Handle TTS state changes."""
        logging.info(f"TTS state changed from {old_state} to {new_state} for message {self.message_id_internal}")
        # Force a recompose of the action buttons
        self.refresh()
    
    def watch_tts_progress(self, progress: float) -> None:
        """Handle TTS progress updates."""
        # Update the tooltip on the generating button if it exists
        try:
            if self.tts_state == "generating":
                generating_btn = self.query_one("#tts-generating", Button)
                generating_btn.tooltip = f"Generating audio... {progress:.0%}"
        except Exception:
            pass
    
    def watch__generation_complete_internal(self, complete: bool) -> None:
        """Watcher for the internal generation status."""
        logging.info(f"watch__generation_complete_internal called with complete={complete}")
        if self.has_class("-ai"):
            try:
                actions_container = self.query_one(".message-actions")
                if complete:
                    actions_container.remove_class("-generating")
                    actions_container.styles.display = "block"
                    self.refresh()
                else:
                    actions_container.add_class("-generating")
                
                # Handle continue button
                try:
                    continue_button = self.query_one("#continue-response-button", Button)
                    continue_button.display = complete
                except QueryError:
                    logging.debug("Continue button not found")
            except QueryError as qe:
                logging.warning(f"QueryError in watch__generation_complete_internal: {qe}")
            except Exception as e:
                logging.error(f"Error in watch__generation_complete_internal: {e}", exc_info=True)
    
    @on(Button.Pressed, "#toggle-image-mode")
    def handle_toggle_mode(self) -> None:
        """Toggle between pixel and regular rendering."""
        self.pixel_mode = not self.pixel_mode
    
    @on(Button.Pressed, "#save-image")
    async def handle_save_image(self) -> None:
        """Save image to file."""
        if not self.image_data:
            return
        
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = self.image_mime_type.split('/')[-1] if self.image_mime_type else 'png'
            filename = f"chat_image_{timestamp}.{extension}"
            
            # Save to user's downloads directory
            downloads_path = Path.home() / "Downloads" / filename
            downloads_path.write_bytes(self.image_data)
            
            self.app.notify(f"Image saved to: {downloads_path}")
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            from rich.markup import escape
            self.app.notify(f"Error saving image: {escape(str(e))}", severity="error")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button inside this message is pressed."""
        # Only handle buttons that don't have specific handlers
        if event.button.id not in ["toggle-image-mode", "save-image"]:
            # Post our custom Action message
            self.post_message(self.Action(self, event.button))
            # Stop the event from bubbling up
            event.stop()
    
    def update_tts_state(self, state: str, audio_file: Optional[Path] = None) -> None:
        """Update TTS state and audio file."""
        self.tts_state = state
        if audio_file:
            self.tts_audio_file = audio_file
        elif state == "idle":
            self.tts_audio_file = None
            self.tts_progress = 0.0
    
    def update_tts_progress(self, progress: float, status: str = "") -> None:
        """Update TTS generation progress."""
        self.tts_progress = progress
        if status:
            logging.debug(f"TTS progress for message {self.message_id_internal}: {progress:.0%} - {status}")
    
    def mark_generation_complete(self):
        """Marks the AI message generation as complete."""
        logging.info(f"mark_generation_complete called")
        if self.has_class("-ai"):
            self._generation_complete_internal = True
            self.refresh()
    
    def update_message_chunk(self, chunk: str):
        """Appends a chunk of text to an AI message during streaming."""
        if self.has_class("-ai") and not self._generation_complete_internal:
            self.message_text += chunk
    
    def _check_for_files(self):
        """Check if the message contains extractable files."""
        if not self._file_extractor:
            self._file_extractor = FileExtractor()
        
        try:
            self._extracted_files = self._file_extractor.extract_files(self.message_text)
        except Exception as e:
            logging.error(f"Error extracting files from message: {e}")
            self._extracted_files = []
    
    def update_variant_info(self, variant_num: int, total: int, is_selected: bool = False):
        """Update variant information for this message."""
        self.variant_number = variant_num
        self.total_variants = total
        self.is_selected_variant = is_selected
        self.has_variants = total > 1
        
        # Update the variant navigation UI if it exists
        try:
            if self.has_variants:
                # Update indicator label
                indicator = self.query_one(".variant-indicator", Label)
                indicator.update(f"Response {variant_num} of {total}")
                
                # Update button states
                prev_btn = self.query_one("#prev-variant", Button)
                next_btn = self.query_one("#next-variant", Button)
                prev_btn.disabled = (variant_num <= 1)
                next_btn.disabled = (variant_num >= total)
                
                # Update select button visibility
                try:
                    select_btn = self.query_one("#select-variant", Button)
                    select_btn.display = not is_selected
                except QueryError:
                    pass  # Select button may not exist if already selected
        except QueryError:
            # Variant navigation not present yet, will be created on refresh
            self.refresh(recompose=True)

#
#
#######################################################################################################################