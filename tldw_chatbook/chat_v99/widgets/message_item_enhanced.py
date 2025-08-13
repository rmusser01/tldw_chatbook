"""Enhanced message item with action buttons."""

from textual.widgets import Static, Button
from textual.reactive import reactive
from textual.containers import Container, Horizontal
from textual.message import Message
from textual import on
from datetime import datetime

try:
    from ..models import ChatMessage
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import ChatMessage


class MessageItemEnhanced(Container):
    """Enhanced message item with action buttons and editing support."""
    
    class MessageAction(Message):
        """Event fired when a message action is triggered."""
        def __init__(self, action: str, message: ChatMessage):
            super().__init__()
            self.action = action
            self.message = message
    
    DEFAULT_CSS = """
    MessageItemEnhanced {
        margin: 1 0;
        padding: 1;
        background: $panel;
        border: round $primary;
        width: 100%;
    }
    
    MessageItemEnhanced.user {
        align: right middle;
        background: $primary 20%;
    }
    
    MessageItemEnhanced.assistant {
        align: left middle;
        background: $panel;
    }
    
    MessageItemEnhanced.system {
        align: center middle;
        background: $surface;
        border: dashed $primary;
    }
    
    MessageItemEnhanced.streaming {
        border: dashed $accent;
        opacity: 0.8;
    }
    
    MessageItemEnhanced:hover {
        background: $panel-lighten-1;
        border: solid $accent;
    }
    
    MessageItemEnhanced:hover .message-actions {
        display: block;
    }
    
    .message-header {
        layout: horizontal;
        width: 100%;
        height: auto;
    }
    
    .message-role {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
        width: auto;
    }
    
    .message-actions {
        layout: horizontal;
        height: 3;
        dock: right;
        display: none;
    }
    
    .action-button {
        width: auto;
        height: 3;
        margin: 0 1;
        padding: 0 1;
        background: $surface;
        border: none;
        color: $text-muted;
    }
    
    .action-button:hover {
        background: $primary;
        color: $text;
    }
    
    .message-content {
        padding: 0 1;
        color: $text;
        width: 100%;
    }
    
    .message-content-editing {
        width: 100%;
        padding: 1;
        background: $surface;
        border: solid $accent;
    }
    
    .message-timestamp {
        text-style: dim;
        color: #888888;
        margin-top: 1;
    }
    
    .message-token-count {
        text-style: dim;
        color: #888888;
        margin-left: 2;
    }
    """
    
    # Reactive content for streaming/editing
    content: reactive[str] = reactive("")
    is_editing: reactive[bool] = reactive(False)
    
    def __init__(self, message: ChatMessage, is_streaming: bool = False):
        """Initialize enhanced message item.
        
        Args:
            message: The chat message to display
            is_streaming: Whether this is a streaming message
        """
        super().__init__()
        self.message = message
        self.is_streaming = is_streaming
        self.content = message.content
        # Calculate initial token count
        self.token_count = len(message.content) // 4 if message.content else 0
        
        # Add CSS classes
        self.add_class(message.role)
        if is_streaming:
            self.add_class("streaming")
    
    def compose(self):
        """Compose enhanced message display."""
        # Header with role and actions
        with Horizontal(classes="message-header"):
            # Role label
            role_display = self.message.role.replace("_", " ").title()
            yield Static(
                f"{role_display}:",
                classes="message-role"
            )
            
            # Action buttons (only for user/assistant messages)
            if self.message.role in ["user", "assistant"] and not self.is_streaming:
                with Container(classes="message-actions"):
                    yield Button("📝", id="edit-btn", classes="action-button")
                    yield Button("📋", id="copy-btn", classes="action-button")
                    yield Button("🔄", id="regenerate-btn", classes="action-button")
                    yield Button("➕", id="continue-btn", classes="action-button")
                    yield Button("📌", id="pin-btn", classes="action-button")
                    yield Button("🗑️", id="delete-btn", classes="action-button")
        
        # Message content
        yield Static(
            self.content,
            classes="message-content",
            id="content"
        )
        
        # Footer with timestamp and token count
        if not self.is_streaming:
            with Horizontal():
                yield Static(
                    self.message.timestamp.strftime("%H:%M:%S"),
                    classes="message-timestamp"
                )
                if self.token_count > 0:
                    yield Static(
                        f"({self.token_count} tokens)",
                        classes="message-token-count"
                    )
    
    def watch_content(self, old_content: str, new_content: str):
        """React to content changes."""
        # Only query widgets if we're mounted (compose has been called)
        if self.is_mounted:
            try:
                content_widget = self.query_one("#content", Static)
                content_widget.update(new_content)
            except:
                pass  # Widget not ready yet
        
        # Update token count
        self.update_token_count()
    
    def update_token_count(self):
        """Update token count estimate."""
        # Simple approximation: ~4 chars per token
        self.token_count = len(self.content) // 4
    
    @on(Button.Pressed, "#edit-btn")
    def handle_edit(self):
        """Handle edit button press."""
        self.post_message(self.MessageAction("edit", self.message))
    
    @on(Button.Pressed, "#copy-btn")
    def handle_copy(self):
        """Handle copy button press."""
        import pyperclip
        try:
            pyperclip.copy(self.content)
            self.app.notify("Copied to clipboard", severity="information")
        except:
            # Fallback if pyperclip not available
            self.post_message(self.MessageAction("copy", self.message))
    
    @on(Button.Pressed, "#regenerate-btn")
    def handle_regenerate(self):
        """Handle regenerate button press."""
        if self.message.role == "assistant":
            self.post_message(self.MessageAction("regenerate", self.message))
    
    @on(Button.Pressed, "#continue-btn")
    def handle_continue(self):
        """Handle continue button press."""
        if self.message.role == "assistant":
            self.post_message(self.MessageAction("continue", self.message))
    
    @on(Button.Pressed, "#pin-btn")
    def handle_pin(self):
        """Handle pin button press."""
        self.post_message(self.MessageAction("pin", self.message))
        # Visual feedback following Textual pattern
        pin_btn = self.query_one("#pin-btn", Button)
        # Toggle between unpinned (📌) and pinned (📍)
        current_label = str(pin_btn.label)
        pin_btn.label = "📍" if current_label == "📌" else "📌"
    
    @on(Button.Pressed, "#delete-btn")
    def handle_delete(self):
        """Handle delete button press."""
        self.post_message(self.MessageAction("delete", self.message))
    
    def update_streaming_content(self, content: str):
        """Update content for streaming messages.
        
        Args:
            content: The new content to display
        """
        self.content = content
    
    def finalize_streaming(self):
        """Convert streaming message to regular message."""
        self.remove_class("streaming")
        self.is_streaming = False
        
        # Following Textual best practice: don't try to modify composed structure
        # Instead, just update the state and let reactive patterns handle it
        # The timestamp and actions should already be in the composed structure
        
        # Update token count if needed
        self.update_token_count()