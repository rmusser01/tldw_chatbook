"""Message list widget with reactive updates."""

from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual import on
from typing import List, Optional

try:
    from .message_item_enhanced import MessageItemEnhanced as MessageItem
    from ..models import ChatSession, ChatMessage
    from ..messages import MessageReceived, StreamingChunk
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import ChatSession, ChatMessage
    from messages import MessageReceived, StreamingChunk


class MessageList(VerticalScroll):
    """Message list with reactive updates and streaming support.
    
    References:
    - Reactive attributes: https://textual.textualize.io/guide/reactivity/#reactive-attributes
    - Containers: https://textual.textualize.io/widgets/verticalscroll/
    """
    
    DEFAULT_CSS = """
    MessageList {
        height: 1fr;
        padding: 1;
        background: $surface;
        scrollbar-gutter: stable;
    }
    
    MessageList:focus {
        border: solid $accent;
    }
    
    MessageList > MessageItem {
        width: 100%;
    }
    
    MessageList > MessageItem.user {
        align: right middle;
        margin-left: 10;
    }
    
    MessageList > MessageItem.assistant {
        align: left middle;
        margin-right: 10;
    }
    
    MessageList > MessageItem.system {
        align: center middle;
        margin: 0 5;
    }
    """
    
    # Reactive state with proper typing - recompose=True for automatic updates
    session: reactive[Optional[ChatSession]] = reactive(None, init=False)
    messages: reactive[List[ChatMessage]] = reactive(list, recompose=True)
    is_streaming: reactive[bool] = reactive(False)
    streaming_content: reactive[str] = reactive("")
    streaming_message: reactive[Optional[MessageItem]] = reactive(None, init=False)
    
    def compose(self):
        """Compose message items.
        Called automatically when messages change due to recompose=True."""
        # Use MessageItemEnhanced (aliased as MessageItem) for all messages
        for message in self.messages:
            yield MessageItem(message, is_streaming=False)
        
        # Add streaming message if active (reactive pattern)
        if self.is_streaming and self.streaming_content:
            streaming_msg = ChatMessage(
                role="assistant",
                content=self.streaming_content
            )
            streaming_item = MessageItem(streaming_msg, is_streaming=True)
            self.streaming_message = streaming_item
            yield streaming_item
    
    def watch_session(self, old_session: Optional[ChatSession], new_session: Optional[ChatSession]):
        """React to session changes.
        Per https://textual.textualize.io/guide/reactivity/#watch-methods"""
        if new_session:
            self.messages = list(new_session.messages)
        else:
            self.messages = []
    
    def watch_messages(self, old_messages: List[ChatMessage], new_messages: List[ChatMessage]):
        """React to message list changes.
        With recompose=True, compose() is called automatically.
        Per https://textual.textualize.io/guide/reactivity/#recompose
        Note: When recompose=True, watchers may not receive old value."""
        # Scroll to bottom after recompose
        self.call_after_refresh(self.scroll_end)
    
    def watch_streaming_content(self, old_content: str, new_content: str):
        """React to streaming content changes using reactive pattern.
        This avoids direct widget manipulation."""
        if self.streaming_message and self.is_streaming:
            # Use reactive update on the streaming message
            self.streaming_message.update_streaming_content(new_content)
    
    def add_user_message(self, content: str, attachments: List[str] = None) -> ChatMessage:
        """Add a user message to the list.
        
        Args:
            content: Message content
            attachments: Optional list of attachment paths
            
        Returns:
            The created ChatMessage
        """
        message = ChatMessage(
            role="user",
            content=content,
            attachments=attachments or []
        )
        # Create new list to trigger reactive update
        self.messages = [*self.messages, message]
        return message
    
    def add_assistant_message(self, content: str) -> ChatMessage:
        """Add an assistant message.
        
        Args:
            content: Message content
            
        Returns:
            The created ChatMessage
        """
        message = ChatMessage(
            role="assistant",
            content=content
        )
        self.messages = [*self.messages, message]
        return message
    
    def add_system_message(self, content: str) -> ChatMessage:
        """Add a system message.
        
        Args:
            content: Message content
            
        Returns:
            The created ChatMessage
        """
        message = ChatMessage(
            role="system",
            content=content
        )
        self.messages = [*self.messages, message]
        return message
    
    def start_streaming(self):
        """Start streaming response using reactive pattern."""
        self.is_streaming = True
        self.streaming_content = ""
        # Trigger recompose to add streaming message
        self.refresh(recompose=True)
    
    def update_streaming(self, content: str, done: bool = False):
        """Update streaming content using reactive patterns only.
        
        Args:
            content: Content chunk to add
            done: Whether streaming is complete
        """
        if self.is_streaming:
            # Update reactive attribute to trigger watcher
            self.streaming_content = self.streaming_content + content
            
            if done:
                # Convert to regular message
                final_message = ChatMessage(
                    role="assistant",
                    content=self.streaming_content
                )
                # Add to messages and stop streaming
                self.messages = [*self.messages, final_message]
                self.is_streaming = False
                self.streaming_content = ""
                self.streaming_message = None
    
    def load_session(self, session: Optional[ChatSession]):
        """Load a chat session.
        
        Args:
            session: The session to load, or None to clear
        """
        self.session = session
    
    @on(MessageReceived)
    def handle_message_received(self, event: MessageReceived):
        """Handle new message event.
        Per https://textual.textualize.io/guide/events/#handler-methods"""
        # Update reactive list - triggers recompose
        self.messages = [*self.messages, event.message]
    
    @on(StreamingChunk)
    def handle_streaming_chunk(self, event: StreamingChunk):
        """Handle streaming updates for progressive display."""
        self.update_streaming(event.content, event.done)
    
    def clear_messages(self):
        """Clear all messages."""
        self.messages = []
        self.is_streaming = False
        self.streaming_content = ""
        self.streaming_message = None