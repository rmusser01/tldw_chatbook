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
from textual.widgets import Button, Label, Static

# Optional import for textual-image
try:
    from textual_image.widget import Image as TextualImage
    TEXTUAL_IMAGE_AVAILABLE = True
except ImportError:
    TEXTUAL_IMAGE_AVAILABLE = False
    logging.info("textual-image not available, will use rich-pixels fallback")

#
# Local Imports
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
    """
    
    # Reactive properties
    message_text = reactive("", repaint=True)
    role = reactive("User", repaint=True)
    pixel_mode = reactive(False)
    _generation_complete_internal = reactive(True)
    
    # Internal state
    message_id_internal: reactive[Optional[str]] = reactive(None)
    message_version_internal: reactive[Optional[int]] = reactive(None)
    timestamp: reactive[Optional[str]] = reactive(None)
    image_data: reactive[Optional[bytes]] = reactive(None)
    image_mime_type: reactive[Optional[str]] = reactive(None)
    
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
        self._image_widget = None
        
        # Add role-specific class
        if role.lower() == "user":
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
            
            # Message content
            yield Static(self.message_text, classes="message-text")
            
            # Image display if present
            if self.image_data:
                with Container(classes="message-image-container"):
                    # Create image display widget
                    self._image_widget = Container(classes="message-image")
                    yield self._image_widget
                    
                    # Image controls
                    with Container(classes="image-controls"):
                        yield Button("Toggle View", id="toggle-image-mode")
                        yield Button("Save Image", id="save-image")
            
            # Action buttons
            actions_class = "message-actions"
            if self.has_class("-ai") and not self.generation_complete:
                actions_class += " -generating"
            
            with Horizontal(classes=actions_class) as actions_bar:
                actions_bar.id = f"actions-bar-{self.id or self.message_id_internal or 'new'}"
                # Common buttons
                yield Button("Edit", classes="action-button edit-button")
                yield Button("📋", classes="action-button copy-button", id="copy")
                yield Button("🔊", classes="action-button speak-button", id="speak")
                
                # AI-specific buttons
                if self.has_class("-ai"):
                    yield Button("👍", classes="action-button thumb-up-button", id="thumb-up")
                    yield Button("👎", classes="action-button thumb-down-button", id="thumb-down")
                    yield Button("🔄", classes="action-button regenerate-button", id="regenerate")
                    yield Button("↪️", id="continue-response-button", classes="action-button continue-button")
                
                # Delete button
                yield Button("🗑️", classes="action-button delete-button")
    
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
            self._image_widget.mount(
                Static(f"[red]Error rendering image: {e}[/red]")
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
            
            # Save to temporary buffer
            temp_buffer = BytesIO()
            pil_image.save(temp_buffer, format='PNG')
            temp_buffer.seek(0)
            
            # Create Pixels renderable
            pixels = Pixels.from_image(temp_buffer)
            self._image_widget.mount(Static(pixels))
        except Exception as e:
            logging.error(f"Error in pixelated rendering: {e}")
            self._render_fallback()
    
    def _render_regular(self) -> None:
        """Render image using textual-image or fallback."""
        if TEXTUAL_IMAGE_AVAILABLE:
            try:
                image = TextualImage(self.image_data)
                self._image_widget.mount(image)
            except Exception as e:
                logging.error(f"Error with textual-image rendering: {e}")
                self._render_fallback()
        else:
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
[\\[Click "Save Image" to view in external viewer\\]]
[/dim]"""
        
        self._image_widget.mount(Static(fallback_text))
    
    def watch_pixel_mode(self, pixel_mode: bool) -> None:
        """Handle pixel mode changes."""
        if self.image_data and self._image_widget:
            self._render_image()
    
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
            self.app.notify(f"Error saving image: {e}", severity="error")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button inside this message is pressed."""
        # Post our custom Action message
        self.post_message(self.Action(self, event.button))
        # Stop the event from bubbling up
        event.stop()
    
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

#
#
#######################################################################################################################