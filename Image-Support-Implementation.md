# Image Support Implementation for Chat Messages

## Overview
This document outlines how to implement image support in tldw_chatbook's chat messages, allowing users to embed images and toggle between regular and pixelated rendering using both `textual-image` and `rich-pixels`.

## Current State
The database already supports image storage:
- `image_data` (BLOB) - stores binary image data
- `image_mime_type` (TEXT) - stores MIME type
- `add_message()` function accepts image parameters

## Implementation Plan

### 1. Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
images = [
    "textual-image[textual]>=0.6.0",
    "rich-pixels>=3.0.0",
    "pillow>=10.0.0",
]
```

### 2. Enhanced ChatMessage Widget

```python
# tldw_chatbook/Widgets/chat_message.py

from textual import on
from textual.app import ComposeResult
from textual.widgets import Static, Container
from textual.reactive import reactive
from rich_pixels import Pixels
from PIL import Image as PILImage
from io import BytesIO
import base64

try:
    from textual_image.widget import Image as TextualImage
    TEXTUAL_IMAGE_AVAILABLE = True
except ImportError:
    TEXTUAL_IMAGE_AVAILABLE = False

class ChatMessage(Container):
    """Enhanced chat message widget with image support."""
    
    DEFAULT_CSS = """
    ChatMessage {
        layout: vertical;
        margin: 1 0;
        padding: 1;
    }
    
    .message-image {
        margin: 1 0;
        align: center middle;
    }
    
    .image-controls {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }
    """
    
    # Reactive property for render mode
    pixel_mode = reactive(False)
    
    def __init__(
        self,
        user: str,
        content: str,
        timestamp: str,
        image_data: bytes = None,
        image_mime_type: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.user = user
        self.content = content
        self.timestamp = timestamp
        self.image_data = image_data
        self.image_mime_type = image_mime_type
        self._image_widget = None
        
    def compose(self) -> ComposeResult:
        # Message header
        yield Static(f"[bold]{self.user}[/bold] - {self.timestamp}")
        
        # Message content
        yield Static(self.content, classes="message-content")
        
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
                    
    def on_mount(self) -> None:
        """Render image when widget is mounted."""
        if self.image_data and self._image_widget:
            self._render_image()
            
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
            self._image_widget.mount(
                Static(f"[red]Error rendering image: {e}[/red]")
            )
            
    def _render_pixelated(self) -> None:
        """Render image using rich-pixels."""
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
        
    def _render_regular(self) -> None:
        """Render image using textual-image or fallback."""
        if TEXTUAL_IMAGE_AVAILABLE:
            # Use textual-image for high quality
            try:
                image = TextualImage.from_bytes(self.image_data)
                self._image_widget.mount(image)
            except Exception:
                # Fallback to text representation
                self._render_fallback()
        else:
            self._render_fallback()
            
    def _render_fallback(self) -> None:
        """Fallback rendering for unsupported terminals."""
        # Display image info and base64 preview
        image_size = len(self.image_data) / 1024  # KB
        preview = base64.b64encode(self.image_data[:100]).decode()[:50]
        
        fallback_text = f"""[dim]
📷 Image ({self.image_mime_type})
Size: {image_size:.1f} KB
Preview: {preview}...
[Click "Save Image" to view in external viewer]
[/dim]"""
        
        self._image_widget.mount(Static(fallback_text))
        
    @on(Button.Pressed, "#toggle-image-mode")
    def handle_toggle_mode(self) -> None:
        """Toggle between pixel and regular rendering."""
        self.pixel_mode = not self.pixel_mode
        self._render_image()
        
    @on(Button.Pressed, "#save-image")
    async def handle_save_image(self) -> None:
        """Save image to file."""
        if not self.image_data:
            return
            
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = self.image_mime_type.split('/')[-1] if self.image_mime_type else 'png'
        filename = f"chat_image_{timestamp}.{extension}"
        
        # Save to user's downloads directory
        downloads_path = Path.home() / "Downloads" / filename
        downloads_path.write_bytes(self.image_data)
        
        self.app.notify(f"Image saved to: {downloads_path}")
```

### 3. Image Input Handler

```python
# tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py

from pathlib import Path
import mimetypes
from PIL import Image as PILImage

class ChatImageHandler:
    """Handle image operations for chat."""
    
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    
    @staticmethod
    async def process_image_file(file_path: str) -> tuple[bytes, str]:
        """Process an image file for chat attachment."""
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        # Check file extension
        if path.suffix.lower() not in ChatImageHandler.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}")
            
        # Check file size
        if path.stat().st_size > ChatImageHandler.MAX_IMAGE_SIZE:
            raise ValueError(f"Image file too large (max {ChatImageHandler.MAX_IMAGE_SIZE / 1024 / 1024}MB)")
            
        # Read image data
        image_data = path.read_bytes()
        
        # Determine MIME type
        mime_type = mimetypes.guess_type(str(path))[0] or 'image/png'
        
        # Optionally resize if too large
        try:
            pil_image = PILImage.open(path)
            if pil_image.width > 2048 or pil_image.height > 2048:
                # Resize maintaining aspect ratio
                pil_image.thumbnail((2048, 2048), PILImage.Resampling.LANCZOS)
                
                # Save to bytes
                buffer = BytesIO()
                format_name = 'PNG' if path.suffix.lower() == '.png' else 'JPEG'
                pil_image.save(buffer, format=format_name, optimize=True)
                image_data = buffer.getvalue()
        except Exception:
            # If PIL fails, use original data
            pass
            
        return image_data, mime_type
```

### 4. Chat Input Enhancement

```python
# Modifications to Chat_Window.py

class ChatWindow(Container):
    def compose(self) -> ComposeResult:
        # ... existing code ...
        
        # Enhanced input area with image support
        with Container(id="chat-input-container"):
            yield TextArea(id="chat-input", classes="chat-input")
            with Container(classes="chat-input-toolbar"):
                yield Button("Send", id="send-chat")
                yield Button("📎 Attach", id="attach-image")
                yield Button("Stop", id="stop-chat-generation")
                # ... other buttons ...
                
        # Hidden file path input for image selection
        yield Input(
            id="image-file-path",
            placeholder="Enter image file path...",
            classes="hidden"
        )
        
    @on(Button.Pressed, "#attach-image")
    def handle_attach_image(self) -> None:
        """Show image file path input."""
        file_input = self.query_one("#image-file-path")
        file_input.remove_class("hidden")
        file_input.focus()
        
    @on(Input.Submitted, "#image-file-path")
    async def handle_image_path_submitted(self, event: Input.Submitted) -> None:
        """Process submitted image path."""
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
            self.query_one("#attach-image").label = "📎 Image Attached"
            self.app.notify(f"Image attached: {Path(file_path).name}")
            
            # Hide file input
            event.input.add_class("hidden")
            event.input.value = ""
            
        except Exception as e:
            self.app.notify(f"Error attaching image: {e}", severity="error")
```

### 5. Terminal Detection and Fallback

```python
# tldw_chatbook/Utils/terminal_utils.py

import os
import sys

def detect_terminal_capabilities():
    """Detect terminal image support capabilities."""
    term = os.environ.get('TERM', '')
    term_program = os.environ.get('TERM_PROGRAM', '')
    
    capabilities = {
        'sixel': False,
        'tgp': False,
        'unicode': True,  # Assume unicode support
        'recommended_mode': 'pixels'  # Default to rich-pixels
    }
    
    # Check for specific terminals
    if 'kitty' in term or 'kitty' in term_program:
        capabilities['tgp'] = True
        capabilities['recommended_mode'] = 'regular'
    elif 'wezterm' in term.lower():
        capabilities['tgp'] = True
        capabilities['sixel'] = True
        capabilities['recommended_mode'] = 'regular'
    elif 'xterm' in term and '256color' in term:
        capabilities['sixel'] = True
        capabilities['recommended_mode'] = 'regular'
    
    return capabilities
```

### 6. Configuration Options

Add to config:
```toml
[chat.images]
enabled = true
default_render_mode = "auto"  # auto, pixels, regular
max_size_mb = 10
auto_resize = true
resize_max_dimension = 2048
save_location = "~/Downloads"
```

## Usage Flow

1. **Attaching an Image**:
   - User clicks "📎 Attach" button
   - File path input appears
   - User enters image path and presses Enter
   - Image is validated and processed
   - UI shows "Image Attached"

2. **Sending with Image**:
   - When user sends message, image data is included
   - Message is stored with image in database
   - Chat history shows message with image

3. **Viewing Images**:
   - Images render automatically based on terminal capabilities
   - "Toggle View" switches between pixel/regular rendering
   - "Save Image" exports to file system

4. **Fallback Behavior**:
   - Unsupported terminals show image metadata
   - Base64 preview for verification
   - Save option always available

## Future Enhancements

1. **File Browser Widget**: Custom file picker for easier selection
2. **Drag & Drop**: Terminal emulator specific implementations
3. **Multiple Images**: Support for multiple images per message
4. **Image Editing**: Basic crop/resize before sending
5. **Clipboard Support**: Paste images from clipboard
6. **URL Support**: Load images from URLs
7. **Thumbnail Generation**: For performance with many images

## Performance Considerations

1. **Lazy Loading**: Only render visible images
2. **Caching**: Cache rendered representations
3. **Compression**: Automatic image compression
4. **Virtual Scrolling**: For chat history with many images
5. **Background Processing**: Async image processing

## Testing

1. Test with various image formats
2. Test with large images
3. Test terminal compatibility
4. Test performance with many images
5. Test error handling for corrupt files

This implementation provides a robust image support system that works across different terminals while maintaining good UX through progressive enhancement.