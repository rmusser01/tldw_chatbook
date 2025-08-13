"""Main chat screen following Textual patterns."""

from textual.screen import Screen
from textual.containers import Container, Horizontal
from textual import on, work
from textual.message import Message
from loguru import logger
from datetime import datetime
from typing import Optional

# Handle both relative and absolute imports
try:
    from ..widgets import MessageList, ChatInput, ChatSidebar
    from ..widgets.message_item_enhanced import MessageItemEnhanced
    from ..messages import MessageSent, SessionChanged, SidebarToggled, StreamingChunk, ErrorOccurred
    from ..models import ChatSession, ChatMessage
    from ..workers.llm_worker import LLMWorker
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from widgets import MessageList, ChatInput, ChatSidebar
    from widgets.message_item_enhanced import MessageItemEnhanced
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
        
        # Initialize LLM worker with REAL provider
        self.llm_worker = LLMWorker(self.app.settings)
        self._is_generating = False
        
        # Add welcome message
        message_list = self.query_one("#message-list", MessageList)
        
        # Show real provider status
        if self.llm_worker.validate_settings():
            message_list.add_system_message(
                f"✅ Connected to {self.app.settings.provider} ({self.app.settings.model})"
            )
        else:
            message_list.add_system_message(
                f"⚠️ Please configure your LLM provider in Settings"
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
    
    @work(exclusive=True, thread=True)
    async def process_message(self, content: str):
        """Process message with LLM using worker pattern.
        Per https://textual.textualize.io/guide/workers/#thread-workers
        Workers don't return values - use callbacks."""
        try:
            # Mark generation as active
            self._is_generating = True
            
            # Show loading indicator
            self.call_from_thread(self.show_loading, True)
            
            # Get message list for streaming
            message_list = self.query_one("#message-list", MessageList)
            
            # Start streaming
            self.call_from_thread(message_list.start_streaming)
            
            # Stream response from REAL LLM
            async for chunk in self.llm_worker.stream_completion(
                content,
                messages=self.app.current_session.messages if self.app.current_session else []
            ):
                # Check for errors
                if chunk.error:
                    self.call_from_thread(
                        self.post_message,
                        ErrorOccurred(chunk.error)
                    )
                    break
                
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
            # Mark generation as inactive
            self._is_generating = False
            
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
    
    def stop_generation(self):
        """Stop the current LLM generation."""
        if self._is_generating and self.llm_worker:
            self.llm_worker.stop_generation()
            self._is_generating = False
            self.notify("Generation stopped", severity="warning")
    
    @on(MessageItemEnhanced.MessageAction)
    def on_message_action(self, event: MessageItemEnhanced.MessageAction) -> None:
        """Handle MessageAction events from MessageItemEnhanced widgets.
        
        Using Textual's @on decorator for proper event handling.
        """
        action = event.action
        message = event.message
        
        if action == "edit":
            # Enable editing mode for the message
            self.notify("Edit mode not yet implemented", severity="information")
        
        elif action == "copy":
            # Copy message to clipboard
            try:
                import pyperclip
                pyperclip.copy(message.content)
                self.notify("Copied to clipboard", severity="information")
            except:
                self.notify("Copy to clipboard failed", severity="error")
        
        elif action == "regenerate":
            # Regenerate the assistant's response
            if message.role == "assistant" and self.app.current_session:
                # Find the preceding user message
                messages = self.app.current_session.messages
                try:
                    idx = messages.index(message)
                    if idx > 0 and messages[idx-1].role == "user":
                        user_msg = messages[idx-1]
                        # Remove the assistant message using reactive pattern
                        from ..models import ChatSession
                        self.app.current_session = ChatSession(
                            id=self.app.current_session.id,
                            title=self.app.current_session.title,
                            messages=messages[:idx],
                            created_at=self.app.current_session.created_at,
                            updated_at=datetime.now(),
                            metadata=self.app.current_session.metadata
                        )
                        # Regenerate response using worker
                        self.process_message(user_msg.content)
                        self.notify("Regenerating response...", severity="information")
                except ValueError:
                    pass
        
        elif action == "continue":
            # Continue from where the assistant left off
            if message.role == "assistant":
                continue_prompt = "Continue from where you left off."
                message_list = self.query_one("#message-list", MessageList)
                message_list.add_user_message(continue_prompt)
                self.process_message(continue_prompt)
        
        elif action == "delete":
            # Delete the message from the session following reactive pattern
            if self.app.current_session:
                # Create new message list without the deleted message
                messages = [m for m in self.app.current_session.messages if m != message]
                
                # Create new session object to trigger reactive update
                from ..models import ChatSession
                self.app.current_session = ChatSession(
                    id=self.app.current_session.id,
                    title=self.app.current_session.title,
                    messages=messages,
                    created_at=self.app.current_session.created_at,
                    updated_at=datetime.now(),
                    metadata=self.app.current_session.metadata
                )
                self.notify("Message deleted", severity="information")
        
        elif action == "pin":
            # Pin/unpin the message
            self.notify("Pin feature not yet implemented", severity="information")