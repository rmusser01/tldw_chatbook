"""Main chat screen following Textual patterns."""

from textual.screen import Screen
from textual.containers import Container, Horizontal
from textual import on, work
from datetime import datetime
from typing import Optional

# Handle both relative and absolute imports
try:
    from ..widgets import MessageList, ChatInput, ChatSidebar
    from ..messages import MessageSent, SessionChanged, SidebarToggled, StreamingChunk, ErrorOccurred
    from ..models import ChatSession, ChatMessage
    from ..workers.llm_worker import LLMWorker
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from widgets import MessageList, ChatInput, ChatSidebar
    from messages import MessageSent, SessionChanged, SidebarToggled, StreamingChunk, ErrorOccurred
    from models import ChatSession, ChatMessage
    from workers.llm_worker import LLMWorker


class ChatScreen(Screen):
    """Main chat screen with proper composition and worker patterns.
    
    References:
    - Screens: https://textual.textualize.io/guide/screens/
    - CSS: https://textual.textualize.io/guide/CSS/#css-files
    - Workers: https://textual.textualize.io/guide/workers/
    """
    
    # Inline CSS following documentation patterns
    CSS = """
    ChatScreen {
        layout: horizontal;
    }
    
    #chat-container {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }
    
    #sidebar {
        dock: left;
        width: 30;
        transition: offset 200ms in_out_cubic;
    }
    
    #sidebar.-hidden {
        offset-x: -100%;
    }
    
    #message-list {
        height: 1fr;
        padding: 1;
        border: round $primary;
    }
    
    #chat-input {
        height: auto;
        min-height: 5;
        max-height: 15;
        dock: bottom;
    }
    
    .loading-indicator {
        dock: top;
        height: 3;
        background: $accent;
        display: none;
    }
    
    .loading-indicator.active {
        display: block;
    }
    """
    
    def compose(self):
        """Compose the screen layout.
        Per https://textual.textualize.io/guide/screens/#composing-screens"""
        # Sidebar (can be hidden via CSS)
        yield ChatSidebar(id="sidebar")
        
        # Main chat area
        with Container(id="chat-container"):
            # Loading indicator
            yield Container(classes="loading-indicator", id="loading")
            
            # Message list
            yield MessageList(id="message-list")
            
            # Input area
            yield ChatInput(id="chat-input")
    
    def on_mount(self):
        """Set up screen after mounting.
        Per https://textual.textualize.io/events/mount/"""
        # Initialize sidebar visibility
        self.update_sidebar_visibility()
        
        # Initialize LLM worker
        self.llm_worker = LLMWorker(self.app.settings)
        
        # Add welcome message
        message_list = self.query_one("#message-list", MessageList)
        message_list.add_system_message(
            "Welcome to Chat v99! This interface follows Textual's reactive patterns."
        )
    
    def update_sidebar_visibility(self):
        """Update sidebar visibility via CSS classes.
        Per https://textual.textualize.io/guide/CSS/#setting-classes"""
        sidebar = self.query_one("#sidebar")
        if self.app.sidebar_visible:
            sidebar.remove_class("-hidden")
        else:
            sidebar.add_class("-hidden")
    
    @on(SessionChanged)
    def handle_session_changed(self, event: SessionChanged):
        """Handle session change from app."""
        message_list = self.query_one("#message-list", MessageList)
        message_list.load_session(event.session)
        
        # Update session title in sidebar
        if event.session:
            session_title = self.query_one("#session-title")
            if session_title:
                session_title.value = event.session.title
    
    @on(SidebarToggled)
    def handle_sidebar_toggled(self, event: SidebarToggled):
        """Handle sidebar toggle."""
        self.update_sidebar_visibility()
    
    @on(MessageSent)
    async def handle_message_sent(self, event: MessageSent):
        """Handle message sent from input widget with proper reactive updates.
        Fixed: Using new session object to trigger reactive updates."""
        # Add user message
        message_list = self.query_one("#message-list", MessageList)
        user_msg = message_list.add_user_message(event.content, event.attachments)
        
        # Update session reactively (create new object to trigger watchers)
        if self.app.current_session:
            # Create new session with updated messages to trigger reactive update
            updated_messages = [*self.app.current_session.messages, user_msg]
            self.app.current_session = ChatSession(
                id=self.app.current_session.id,
                title=self.app.current_session.title,
                messages=updated_messages,
                created_at=self.app.current_session.created_at,
                updated_at=datetime.now(),
                metadata=self.app.current_session.metadata
            )
        
        # Process with LLM using worker
        self.process_message(event.content)
    
    @work(exclusive=True)
    async def process_message(self, content: str):
        """Process message with LLM using worker pattern.
        Per https://textual.textualize.io/guide/workers/#thread-workers
        Workers don't return values - use callbacks."""
        try:
            # Show loading indicator
            self.call_from_thread(self.show_loading, True)
            
            # Get message list for streaming
            message_list = self.query_one("#message-list", MessageList)
            
            # Start streaming
            self.call_from_thread(message_list.start_streaming)
            
            # Stream response
            async for chunk in self.llm_worker.stream_completion(
                content,
                messages=self.app.current_session.messages if self.app.current_session else []
            ):
                self.call_from_thread(
                    self.post_message,
                    StreamingChunk(chunk.content, done=chunk.done)
                )
            
            # Update session with assistant message
            if self.app.current_session:
                # Get the final streaming content from message list
                final_content = message_list.streaming_content
                if final_content:
                    assistant_msg = ChatMessage(
                        role="assistant",
                        content=final_content
                    )
                    
                    # Create new session with assistant message (reactive update)
                    updated_messages = [*self.app.current_session.messages, assistant_msg]
                    self.call_from_thread(
                        self.update_session,
                        updated_messages
                    )
            
        except Exception as e:
            # Post error message
            self.call_from_thread(
                self.post_message,
                ErrorOccurred(str(e))
            )
            
            # Show error notification
            self.call_from_thread(
                self.notify,
                f"Error: {str(e)}",
                severity="error"
            )
        finally:
            # Hide loading indicator
            self.call_from_thread(self.show_loading, False)
    
    def update_session(self, messages: list):
        """Update session with new messages (called from worker thread).
        
        Args:
            messages: Updated message list
        """
        if self.app.current_session:
            self.app.current_session = ChatSession(
                id=self.app.current_session.id,
                title=self.app.current_session.title,
                messages=messages,
                created_at=self.app.current_session.created_at,
                updated_at=datetime.now(),
                metadata=self.app.current_session.metadata
            )
    
    def show_loading(self, show: bool):
        """Show or hide loading indicator.
        
        Args:
            show: Whether to show the loading indicator
        """
        loading = self.query_one("#loading")
        if show:
            loading.add_class("active")
        else:
            loading.remove_class("active")
    
    @on(StreamingChunk)
    def handle_streaming_chunk(self, event: StreamingChunk):
        """Handle streaming chunk from worker."""
        message_list = self.query_one("#message-list", MessageList)
        message_list.update_streaming(event.content, event.done)
    
    @on(ErrorOccurred)
    def handle_error(self, event: ErrorOccurred):
        """Handle error events."""
        # Add error message to chat
        message_list = self.query_one("#message-list", MessageList)
        message_list.add_system_message(f"⚠️ Error: {event.error}")
        
        # Re-enable input
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.set_enabled(True)
    
    def action_focus_input(self):
        """Focus the chat input."""
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.query_one("#input-area").focus()
    
    def action_scroll_up(self):
        """Scroll message list up."""
        message_list = self.query_one("#message-list", MessageList)
        message_list.scroll_up()
    
    def action_scroll_down(self):
        """Scroll message list down."""
        message_list = self.query_one("#message-list", MessageList)
        message_list.scroll_down()