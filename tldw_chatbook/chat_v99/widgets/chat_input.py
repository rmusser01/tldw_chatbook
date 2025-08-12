"""Chat input widget with validation and event posting."""

from textual.containers import Horizontal
from textual.widgets import TextArea, Button
from textual.reactive import reactive
from textual import on
from typing import List  # Fixed: Added List import as identified in review

try:
    from ..messages import MessageSent
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from messages import MessageSent


class ChatInput(Horizontal):
    """Chat input area with validation and reactive state."""
    
    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        max-height: 12;
        min-height: 5;
        padding: 1;
        dock: bottom;
        background: $panel;
        border-top: solid $primary;
    }
    
    #input-area {
        width: 1fr;
        min-height: 3;
        max-height: 10;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }
    
    #input-area:focus {
        border: solid $accent;
    }
    
    .input-button {
        width: auto;
        margin: 0 1;
        min-width: 8;
    }
    
    #send-button {
        background: $primary;
    }
    
    #send-button:hover {
        background: $accent;
    }
    
    #send-button:disabled {
        opacity: 0.5;
        background: #666666;
    }
    
    #attach-button {
        background: $surface;
        border: solid $primary;
    }
    
    #attach-button:hover {
        background: $panel;
        border: solid $accent;
    }
    
    .attachment-count {
        color: $accent;
        text-style: bold;
    }
    """
    
    # Reactive state with proper typing
    is_valid: reactive[bool] = reactive(False)
    is_sending: reactive[bool] = reactive(False)
    attachments: reactive[List[str]] = reactive(list)
    char_count: reactive[int] = reactive(0)
    
    def compose(self):
        """Compose input widgets."""
        yield TextArea(
            id="input-area",
            tab_behavior="focus"
        )
        
        yield Button(
            "📎",
            id="attach-button",
            classes="input-button",
            tooltip="Attach file (not yet implemented)"
        )
        
        yield Button(
            "Send",
            id="send-button",
            classes="input-button",
            variant="primary",
            disabled=True
        )
    
    def on_mount(self):
        """Focus input on mount."""
        self.query_one("#input-area").focus()
    
    @on(TextArea.Changed, "#input-area")
    def validate_input(self, event: TextArea.Changed):
        """Validate input and update button state."""
        content = event.text_area.text.strip()
        self.char_count = len(content)
        self.is_valid = bool(content) and not self.is_sending
        
        # Update send button state
        send_button = self.query_one("#send-button", Button)
        send_button.disabled = not self.is_valid
        
        # Update button label with character count if content exists
        if self.char_count > 0:
            send_button.label = f"Send ({self.char_count})"
        else:
            send_button.label = "Send"
    
    @on(Button.Pressed, "#send-button")
    async def send_message(self):
        """Send the message using event system."""
        if not self.is_valid or self.is_sending:
            return
        
        input_area = self.query_one("#input-area", TextArea)
        content = input_area.text.strip()
        
        if not content:
            return
        
        # Set sending state
        self.is_sending = True
        send_button = self.query_one("#send-button", Button)
        send_button.label = "Sending..."
        send_button.disabled = True
        
        # Post message event - let screen handle the logic
        self.post_message(MessageSent(content, list(self.attachments)))
        
        # Clear input
        input_area.clear()
        self.attachments = []
        self.char_count = 0
        
        # Update attach button if had attachments
        if self.attachments:
            attach_button = self.query_one("#attach-button", Button)
            attach_button.label = "📎"
        
        # Reset state
        self.is_sending = False
        send_button.label = "Send"
        send_button.disabled = True
        
        # Refocus input
        input_area.focus()
    
    # Note: TextArea doesn't have a Submitted event in current Textual
    # We can handle this through keybindings or in the input validation
    
    @on(Button.Pressed, "#attach-button")
    async def handle_attachment(self):
        """Handle file attachment (placeholder for now)."""
        # This would open a file picker in the real implementation
        # For now, just simulate adding an attachment
        
        # Simulate attachment
        self.attachments = [*self.attachments, f"file_{len(self.attachments) + 1}.txt"]
        
        # Update button label
        attach_button = self.query_one("#attach-button", Button)
        if self.attachments:
            attach_button.label = f"📎 ({len(self.attachments)})"
            attach_button.add_class("attachment-count")
        
        # Show notification
        self.notify(f"Added attachment: file_{len(self.attachments)}.txt (simulated)")
    
    def clear_attachments(self):
        """Clear all attachments."""
        self.attachments = []
        attach_button = self.query_one("#attach-button", Button)
        attach_button.label = "📎"
        attach_button.remove_class("attachment-count")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the input area.
        
        Args:
            enabled: Whether to enable the input
        """
        input_area = self.query_one("#input-area", TextArea)
        send_button = self.query_one("#send-button", Button)
        attach_button = self.query_one("#attach-button", Button)
        
        input_area.disabled = not enabled
        attach_button.disabled = not enabled
        
        # Send button state depends on content too
        if not enabled:
            send_button.disabled = True
        else:
            send_button.disabled = not self.is_valid