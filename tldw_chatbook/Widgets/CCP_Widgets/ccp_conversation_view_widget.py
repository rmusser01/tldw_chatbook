"""Conversation view widget for the CCP screen.

This widget displays conversation messages and handles conversation-related UI.
Following Textual best practices with focused, reusable components.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Label, Button
from textual.reactive import reactive
from textual import on
from textual.message import Message
from textual.widget import Widget

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen, CCPScreenState

logger = logger.bind(module="CCPConversationViewWidget")


# ========== Messages ==========

class ConversationViewMessage(Message):
    """Base message for conversation view events."""
    pass


class MessageSelected(ConversationViewMessage):
    """User selected a message in the conversation."""
    def __init__(self, message_id: int, message_data: Dict[str, Any]) -> None:
        super().__init__()
        self.message_id = message_id
        self.message_data = message_data


class MessageEditRequested(ConversationViewMessage):
    """User requested to edit a message."""
    def __init__(self, message_id: int) -> None:
        super().__init__()
        self.message_id = message_id


class MessageDeleteRequested(ConversationViewMessage):
    """User requested to delete a message."""
    def __init__(self, message_id: int) -> None:
        super().__init__()
        self.message_id = message_id


class RegenerateRequested(ConversationViewMessage):
    """User requested to regenerate a message."""
    def __init__(self, message_id: int) -> None:
        super().__init__()
        self.message_id = message_id


class ContinueConversationRequested(ConversationViewMessage):
    """User requested to continue the conversation."""
    pass


# ========== Message Widget Component ==========

class ConversationMessageWidget(Container):
    """Widget representing a single message in a conversation."""
    
    DEFAULT_CSS = """
    ConversationMessageWidget {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: round $surface;
        background: $surface;
    }
    
    ConversationMessageWidget.user-message {
        background: $primary-background-darken-1;
        border: round $primary-darken-1;
    }
    
    ConversationMessageWidget.assistant-message {
        background: $secondary-background-darken-1;
        border: round $secondary-darken-1;
    }
    
    ConversationMessageWidget.system-message {
        background: $warning-background-darken-1;
        border: round $warning-darken-1;
    }
    
    ConversationMessageWidget:hover {
        background: $surface-lighten-1;
    }
    
    .message-header {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .message-role {
        width: auto;
        text-style: bold;
        color: $primary;
    }
    
    .message-timestamp {
        width: 1fr;
        text-align: right;
        color: $text-muted;
    }
    
    .message-content {
        width: 100%;
        padding: 0 1;
        color: $text;
    }
    
    .message-actions {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-top: 1;
        display: none;
    }
    
    ConversationMessageWidget:hover .message-actions {
        display: block;
    }
    
    .message-action-button {
        width: auto;
        height: 3;
        margin-right: 1;
        padding: 0 1;
    }
    """
    
    def __init__(self, message_data: Dict[str, Any], **kwargs):
        """Initialize a conversation message widget.
        
        Args:
            message_data: Dictionary containing message information
            **kwargs: Additional arguments for Container
        """
        # Determine message type for styling
        role = message_data.get('role', 'user')
        classes = f"{role}-message"
        
        super().__init__(classes=classes, **kwargs)
        
        self.message_data = message_data
        self.message_id = message_data.get('id', 0)
        self.role = role
        self.content = message_data.get('content', '')
        self.timestamp = message_data.get('timestamp', '')
    
    def compose(self) -> ComposeResult:
        """Compose the message UI."""
        # Message header
        with Container(classes="message-header"):
            yield Label(self.role.capitalize(), classes="message-role")
            if self.timestamp:
                yield Label(self.timestamp, classes="message-timestamp")
        
        # Message content
        yield Static(self.content, classes="message-content")
        
        # Message actions (shown on hover)
        with Container(classes="message-actions"):
            if self.role != "system":
                yield Button("Edit", classes="message-action-button edit-msg-btn", id=f"edit-msg-{self.message_id}")
                yield Button("Delete", classes="message-action-button delete-msg-btn", id=f"delete-msg-{self.message_id}")
                if self.role == "assistant":
                    yield Button("Regenerate", classes="message-action-button regen-msg-btn", id=f"regen-msg-{self.message_id}")


# ========== Conversation View Widget ==========

class CCPConversationViewWidget(Container):
    """
    Conversation view widget for the CCP screen.
    
    This widget displays conversation messages and provides interaction controls,
    following Textual best practices for focused components.
    """
    
    DEFAULT_CSS = """
    CCPConversationViewWidget {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    CCPConversationViewWidget.hidden {
        display: none !important;
    }
    
    .conversation-header {
        width: 100%;
        height: 3;
        background: $primary-background-darken-1;
        padding: 0 1;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
    }
    
    .conversation-messages-container {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .no-conversation-message {
        width: 100%;
        height: 100%;
        align: center middle;
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    
    .conversation-controls {
        layout: horizontal;
        height: 3;
        width: 100%;
        padding: 1;
        background: $surface;
        border-top: thick $background-darken-1;
    }
    
    .conversation-control-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .conversation-control-button:last-child {
        margin-right: 0;
    }
    """
    
    # Reactive state reference (will be linked to parent screen's state)
    state: reactive[Optional['CCPScreenState']] = reactive(None)
    
    def __init__(self, parent_screen: Optional['CCPScreen'] = None, **kwargs):
        """Initialize the conversation view widget.
        
        Args:
            parent_screen: Reference to the parent CCP screen
            **kwargs: Additional arguments for Container
        """
        super().__init__(id="ccp-conversation-messages-view", classes="ccp-view-area", **kwargs)
        self.parent_screen = parent_screen
        
        # Cache for message widgets
        self._message_widgets: List[ConversationMessageWidget] = []
        self._messages_container: Optional[VerticalScroll] = None
        
        logger.debug("CCPConversationViewWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the conversation view UI."""
        # Header
        yield Static("Conversation History", classes="conversation-header pane-title")
        
        # Messages container
        with VerticalScroll(classes="conversation-messages-container", id="conversation-messages-scroll"):
            # Default message when no conversation is loaded
            yield Static(
                "No conversation loaded.\nSelect a conversation from the sidebar to view messages.",
                classes="no-conversation-message",
                id="no-conversation-placeholder"
            )
        
        # Conversation controls
        with Container(classes="conversation-controls"):
            yield Button("Continue", classes="conversation-control-button", id="continue-conversation-btn")
            yield Button("Export", classes="conversation-control-button", id="export-conversation-btn")
            yield Button("Clear", classes="conversation-control-button", id="clear-conversation-btn")
    
    async def on_mount(self) -> None:
        """Handle widget mount."""
        # Cache the messages container
        self._messages_container = self.query_one("#conversation-messages-scroll", VerticalScroll)
        
        # Link to parent screen's state if available
        if self.parent_screen and hasattr(self.parent_screen, 'state'):
            self.state = self.parent_screen.state
        
        logger.debug("CCPConversationViewWidget mounted")
    
    # ===== Public Methods =====
    
    def load_conversation_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Load and display conversation messages.
        
        Args:
            messages: List of message dictionaries to display
        """
        if not self._messages_container:
            logger.warning("Messages container not available")
            return
        
        # Clear existing messages
        self.clear_messages()
        
        # Remove the placeholder if it exists
        try:
            placeholder = self._messages_container.query_one("#no-conversation-placeholder")
            placeholder.remove()
        except:
            pass
        
        # Add new message widgets
        for message_data in messages:
            message_widget = ConversationMessageWidget(message_data)
            self._message_widgets.append(message_widget)
            self._messages_container.mount(message_widget)
        
        logger.info(f"Loaded {len(messages)} conversation messages")
    
    def add_message(self, message_data: Dict[str, Any]) -> None:
        """Add a single message to the conversation.
        
        Args:
            message_data: Message dictionary to add
        """
        if not self._messages_container:
            logger.warning("Messages container not available")
            return
        
        # Remove placeholder if this is the first message
        if not self._message_widgets:
            try:
                placeholder = self._messages_container.query_one("#no-conversation-placeholder")
                placeholder.remove()
            except:
                pass
        
        # Add the new message widget
        message_widget = ConversationMessageWidget(message_data)
        self._message_widgets.append(message_widget)
        self._messages_container.mount(message_widget)
        
        # Scroll to bottom to show new message
        self._messages_container.scroll_to_bottom()
    
    def update_message(self, message_id: int, new_content: str) -> None:
        """Update the content of an existing message.
        
        Args:
            message_id: ID of the message to update
            new_content: New content for the message
        """
        for widget in self._message_widgets:
            if widget.message_id == message_id:
                # Update the widget's content
                widget.content = new_content
                content_widget = widget.query_one(".message-content", Static)
                content_widget.update(new_content)
                break
    
    def remove_message(self, message_id: int) -> None:
        """Remove a message from the conversation.
        
        Args:
            message_id: ID of the message to remove
        """
        for i, widget in enumerate(self._message_widgets):
            if widget.message_id == message_id:
                widget.remove()
                del self._message_widgets[i]
                break
        
        # If no messages left, show placeholder
        if not self._message_widgets and self._messages_container:
            placeholder = Static(
                "No conversation loaded.\nSelect a conversation from the sidebar to view messages.",
                classes="no-conversation-message",
                id="no-conversation-placeholder"
            )
            self._messages_container.mount(placeholder)
    
    def clear_messages(self) -> None:
        """Clear all messages from the view."""
        # Remove all message widgets
        for widget in self._message_widgets:
            widget.remove()
        
        self._message_widgets.clear()
        
        # Show placeholder
        if self._messages_container:
            # Check if placeholder already exists
            try:
                self._messages_container.query_one("#no-conversation-placeholder")
            except:
                placeholder = Static(
                    "No conversation loaded.\nSelect a conversation from the sidebar to view messages.",
                    classes="no-conversation-message",
                    id="no-conversation-placeholder"
                )
                self._messages_container.mount(placeholder)
    
    def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the conversation."""
        if self._messages_container:
            self._messages_container.scroll_to_bottom()
    
    def scroll_to_message(self, message_id: int) -> None:
        """Scroll to a specific message.
        
        Args:
            message_id: ID of the message to scroll to
        """
        for widget in self._message_widgets:
            if widget.message_id == message_id:
                widget.scroll_visible()
                break
    
    # ===== Event Handlers =====
    
    @on(Button.Pressed, "#continue-conversation-btn")
    async def handle_continue_conversation(self, event: Button.Pressed) -> None:
        """Handle continue conversation button press."""
        event.stop()
        self.post_message(ContinueConversationRequested())
    
    @on(Button.Pressed, "#export-conversation-btn")
    async def handle_export_conversation(self, event: Button.Pressed) -> None:
        """Handle export conversation button press."""
        event.stop()
        # This would trigger export functionality
        logger.info("Export conversation requested")
    
    @on(Button.Pressed, "#clear-conversation-btn")
    async def handle_clear_conversation(self, event: Button.Pressed) -> None:
        """Handle clear conversation button press."""
        event.stop()
        self.clear_messages()
    
    @on(Button.Pressed, ".edit-msg-btn")
    async def handle_edit_message(self, event: Button.Pressed) -> None:
        """Handle edit message button press."""
        event.stop()
        if event.button.id and event.button.id.startswith("edit-msg-"):
            message_id = int(event.button.id.replace("edit-msg-", ""))
            self.post_message(MessageEditRequested(message_id))
    
    @on(Button.Pressed, ".delete-msg-btn")
    async def handle_delete_message(self, event: Button.Pressed) -> None:
        """Handle delete message button press."""
        event.stop()
        if event.button.id and event.button.id.startswith("delete-msg-"):
            message_id = int(event.button.id.replace("delete-msg-", ""))
            self.post_message(MessageDeleteRequested(message_id))
    
    @on(Button.Pressed, ".regen-msg-btn")
    async def handle_regenerate_message(self, event: Button.Pressed) -> None:
        """Handle regenerate message button press."""
        event.stop()
        if event.button.id and event.button.id.startswith("regen-msg-"):
            message_id = int(event.button.id.replace("regen-msg-", ""))
            self.post_message(RegenerateRequested(message_id))