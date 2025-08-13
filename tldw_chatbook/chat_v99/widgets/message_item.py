"""Individual message item widget."""

from textual.widgets import Static
from textual.reactive import reactive
from textual.containers import Container

try:
    from ..models import ChatMessage
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import ChatMessage


class MessageItem(Container):
    """Individual message item with reactive updates."""
    
    DEFAULT_CSS = """
    MessageItem {
        margin: 1 0;
        padding: 1;
        background: $panel;
        border: round $primary;
        width: 100%;
    }
    
    MessageItem.user {
        align: right middle;
        background: $primary;
    }
    
    MessageItem.assistant {
        align: left middle;
    }
    
    MessageItem.system {
        align: center middle;
        background: $surface;
        border: dashed $primary;
    }
    
    MessageItem.tool {
        align: left middle;
        background: $accent;
        border: solid $accent;
    }
    
    MessageItem.tool_result {
        align: left middle;
        background: $panel;
        border: dashed $accent;
    }
    
    MessageItem.streaming {
        border: dashed $accent;
        opacity: 0.8;
    }
    
    MessageItem:hover {
        background: $panel-lighten-1;
        border: solid $accent;
    }
    
    .message-role {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .message-content {
        padding: 0 1;
        color: $text;
    }
    
    .message-timestamp {
        text-style: dim;
        color: #888888;
        margin-top: 1;
    }
    
    .message-attachments {
        margin-top: 1;
        padding: 1;
        background: $surface;
        border: round #888888;
    }
    """
    
    # Reactive content for streaming updates
    content: reactive[str] = reactive("")
    
    def __init__(self, message: ChatMessage, is_streaming: bool = False):
        """Initialize message item.
        
        Args:
            message: The chat message to display
            is_streaming: Whether this is a streaming message
        """
        super().__init__()
        self.message = message
        self.is_streaming = is_streaming
        self.content = message.content
        
        # Add CSS classes based on role and state
        self.add_class(message.role)
        if is_streaming:
            self.add_class("streaming")
    
    def compose(self):
        """Compose message display."""
        # Role label
        role_display = self.message.role.replace("_", " ").title()
        yield Static(
            f"{role_display}:",
            classes="message-role"
        )
        
        # Message content (reactive)
        yield Static(
            self.content,
            classes="message-content",
            id="content"
        )
        
        # Attachments if present
        if self.message.attachments:
            attachments_text = f"📎 Attachments: {', '.join(self.message.attachments)}"
            yield Static(
                attachments_text,
                classes="message-attachments"
            )
        
        # Timestamp (not shown for streaming messages)
        if not self.is_streaming:
            yield Static(
                self.message.timestamp.strftime("%H:%M:%S"),
                classes="message-timestamp"
            )
    
    def watch_content(self, old_content: str, new_content: str):
        """React to content changes for streaming updates.
        This follows the reactive pattern instead of direct manipulation."""
        # Only query widgets if we're mounted (compose has been called)
        if self.is_mounted:
            try:
                content_widget = self.query_one("#content", Static)
                content_widget.update(new_content)
            except:
                pass  # Widget not ready yet
    
    def update_streaming_content(self, content: str):
        """Update content for streaming messages using reactive pattern.
        
        Args:
            content: The new content to display
        """
        self.content = content
    
    def finalize_streaming(self):
        """Convert streaming message to regular message."""
        self.remove_class("streaming")
        self.is_streaming = False
        
        # Add timestamp when streaming ends
        self.mount(Static(
            self.message.timestamp.strftime("%H:%M:%S"),
            classes="message-timestamp"
        ))