"""
Fixed Chat Window Enhanced that uses Textual message system.

This is a partial update showing how to handle the new messages.
The full conversion would need to be done gradually.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.reactive import reactive
from textual.widgets import Button, TextArea, Label, Markdown
from textual.screen import Screen

from loguru import logger

# Import the new message types
from ..Event_Handlers.Chat_Events.chat_messages import (
    UserMessageSent,
    LLMResponseStarted,
    LLMResponseChunk,
    LLMResponseCompleted,
    LLMResponseError,
    ChatError,
    SessionLoaded,
    CharacterLoaded,
    RAGResultsReceived,
    TokenCountUpdated,
    GenerationStopped,
    ClearConversationRequested,
    NewSessionRequested,
    SaveSessionRequested,
    LoadSessionRequested
)

# Import widgets
from ..Widgets.chat_message_enhanced import ChatMessageEnhanced


class ChatWindowEnhanced(Screen):
    """
    Enhanced Chat Window that uses Textual's message system.
    
    This version:
    - Uses reactive attributes for state
    - Handles messages instead of direct manipulation
    - Updates UI through reactive patterns
    """
    
    # ==================== REACTIVE STATE ====================
    
    # Chat state
    messages: reactive[List[Dict[str, Any]]] = reactive([])
    is_streaming: reactive[bool] = reactive(False)
    current_session_id: reactive[Optional[str]] = reactive(None)
    is_ephemeral: reactive[bool] = reactive(False)
    
    # Character state
    active_character: reactive[Optional[Dict[str, Any]]] = reactive(None)
    
    # UI state
    left_sidebar_visible: reactive[bool] = reactive(True)
    right_sidebar_visible: reactive[bool] = reactive(False)
    
    # Token counting
    token_count: reactive[int] = reactive(0)
    max_tokens: reactive[int] = reactive(4096)
    
    # Streaming buffer
    streaming_content: reactive[str] = reactive("")
    streaming_widget: reactive[Optional[ChatMessageEnhanced]] = reactive(None)
    
    # Attachment state
    pending_attachment: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, app_instance=None, **kwargs):
        """Initialize the enhanced chat window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
    def compose(self) -> ComposeResult:
        """Compose the chat window UI."""
        # This would be the full UI composition
        # For now, just a placeholder
        yield VerticalScroll(id="chat-log")
        yield TextArea(id="chat-input", max_height=5)
        yield Horizontal(
            Button("Send", id="send-stop-chat", variant="primary"),
            Button("Clear", id="clear-chat"),
            Button("New", id="new-chat"),
            id="chat-buttons"
        )
    
    # ==================== MESSAGE HANDLERS ====================
    
    @on(UserMessageSent)
    def handle_user_message_sent(self, event: UserMessageSent) -> None:
        """
        Handle user message sent event.
        
        Updates reactive state instead of direct manipulation.
        """
        logger.debug(f"User message sent: {event.content[:50]}...")
        
        # Add to messages (triggers UI update)
        self.messages = [
            *self.messages,
            {
                "role": "user",
                "content": event.content,
                "attachments": event.attachments,
                "timestamp": event.timestamp
            }
        ]
        
        # Clear input (still need one query for input field)
        try:
            input_area = self.query_one("#chat-input", TextArea)
            input_area.clear()
            input_area.focus()
        except Exception as e:
            logger.warning(f"Could not clear input: {e}")
    
    @on(LLMResponseStarted)
    def handle_llm_response_started(self, event: LLMResponseStarted) -> None:
        """
        Handle LLM response started.
        
        Sets streaming state reactively.
        """
        logger.debug("LLM response started")
        
        # Update streaming state
        self.is_streaming = True
        self.streaming_content = ""
        
        # Add placeholder message
        self.messages = [
            *self.messages,
            {
                "role": "assistant",
                "content": "🤔 Thinking...",
                "streaming": True,
                "session_id": event.session_id
            }
        ]
    
    @on(LLMResponseChunk)
    def handle_llm_response_chunk(self, event: LLMResponseChunk) -> None:
        """
        Handle streaming chunk.
        
        Updates reactive streaming buffer.
        """
        # Update streaming content (triggers UI update)
        self.streaming_content = self.streaming_content + event.chunk
        
        # Update last message if it's streaming
        if self.messages and self.messages[-1].get("streaming"):
            # Create new list to trigger reactive update
            updated_messages = self.messages[:-1]
            last_msg = self.messages[-1].copy()
            last_msg["content"] = self.streaming_content
            self.messages = [*updated_messages, last_msg]
    
    @on(LLMResponseCompleted)
    def handle_llm_response_completed(self, event: LLMResponseCompleted) -> None:
        """
        Handle LLM response completion.
        
        Finalizes the streaming message.
        """
        logger.debug(f"LLM response completed: {len(event.full_response)} chars")
        
        # Update streaming state
        self.is_streaming = False
        
        # Update last message with final content
        if self.messages and self.messages[-1].get("streaming"):
            updated_messages = self.messages[:-1]
            last_msg = self.messages[-1].copy()
            last_msg["content"] = event.full_response
            last_msg["streaming"] = False
            last_msg["timestamp"] = event.timestamp
            self.messages = [*updated_messages, last_msg]
        
        # Clear streaming buffer
        self.streaming_content = ""
    
    @on(LLMResponseError)
    def handle_llm_response_error(self, event: LLMResponseError) -> None:
        """
        Handle LLM error.
        
        Shows error in chat.
        """
        logger.error(f"LLM error: {event.error}")
        
        # Update streaming state
        self.is_streaming = False
        
        # Add error message
        self.messages = [
            *self.messages,
            {
                "role": "system",
                "content": f"❌ Error: {event.error}",
                "error": True,
                "session_id": event.session_id
            }
        ]
    
    @on(ChatError)
    def handle_chat_error(self, event: ChatError) -> None:
        """
        Handle general chat errors.
        
        Shows notification based on severity.
        """
        severity_map = {
            "info": "information",
            "warning": "warning",
            "error": "error"
        }
        
        self.app_instance.notify(
            event.error,
            severity=severity_map.get(event.severity, "error"),
            timeout=5
        )
    
    @on(SessionLoaded)
    def handle_session_loaded(self, event: SessionLoaded) -> None:
        """
        Handle session loaded.
        
        Updates messages reactively.
        """
        logger.info(f"Session loaded: {event.session_id}")
        
        # Update session state
        self.current_session_id = event.session_id
        self.is_ephemeral = False
        
        # Load messages (triggers UI rebuild)
        self.messages = event.messages
    
    @on(CharacterLoaded)
    def handle_character_loaded(self, event: CharacterLoaded) -> None:
        """
        Handle character loaded.
        
        Updates character state reactively.
        """
        logger.info(f"Character loaded: {event.character_data.get('name', 'Unknown')}")
        
        # Update character state
        self.active_character = event.character_data
        
        # Could update system prompt here
        if self.app_instance:
            self.app_instance.notify(
                f"Loaded character: {event.character_data.get('name', 'Unknown')}",
                severity="information"
            )
    
    @on(TokenCountUpdated)
    def handle_token_count_updated(self, event: TokenCountUpdated) -> None:
        """
        Handle token count update.
        
        Updates reactive token state.
        """
        self.token_count = event.count
        self.max_tokens = event.max_tokens
        
        # Warning if approaching limit
        if event.count > event.max_tokens * 0.9:
            self.app_instance.notify(
                f"Approaching token limit: {event.count}/{event.max_tokens}",
                severity="warning"
            )
    
    @on(RAGResultsReceived)
    def handle_rag_results(self, event: RAGResultsReceived) -> None:
        """
        Handle RAG results.
        
        Could show in UI or just log.
        """
        logger.info(f"RAG results: {len(event.results)} items")
        
        if event.context:
            # Could add a system message showing RAG was applied
            self.messages = [
                *self.messages,
                {
                    "role": "system",
                    "content": f"📚 RAG Context Applied ({len(event.context)} chars)",
                    "rag": True
                }
            ]
    
    @on(GenerationStopped)
    def handle_generation_stopped(self, event: GenerationStopped) -> None:
        """
        Handle generation stopped.
        
        Updates streaming state.
        """
        logger.info("Generation stopped by user")
        
        # Update state
        self.is_streaming = False
        
        # Mark last message as incomplete
        if self.messages and self.messages[-1].get("streaming"):
            updated_messages = self.messages[:-1]
            last_msg = self.messages[-1].copy()
            last_msg["streaming"] = False
            last_msg["incomplete"] = True
            self.messages = [*updated_messages, last_msg]
    
    @on(ClearConversationRequested)
    def handle_clear_conversation(self, event: ClearConversationRequested) -> None:
        """
        Handle clear conversation request.
        
        Clears messages reactively.
        """
        logger.info("Clearing conversation")
        
        # Clear messages (triggers UI update)
        self.messages = []
        self.current_session_id = None
        self.active_character = None
    
    @on(NewSessionRequested)
    def handle_new_session(self, event: NewSessionRequested) -> None:
        """
        Handle new session request.
        
        Creates new session reactively.
        """
        logger.info(f"New session requested (ephemeral: {event.ephemeral})")
        
        # Clear current state
        self.messages = []
        self.current_session_id = None if event.ephemeral else "new_session"
        self.is_ephemeral = event.ephemeral
    
    # ==================== WATCH METHODS (REACTIVE) ====================
    
    def watch_messages(self, old_messages: List, new_messages: List) -> None:
        """
        React to message changes.
        
        This is called automatically when messages change.
        The UI will update based on this.
        """
        # The compose method would rebuild the message list
        # For now, just log
        logger.debug(f"Messages updated: {len(old_messages)} -> {len(new_messages)}")
    
    def watch_is_streaming(self, old: bool, new: bool) -> None:
        """
        React to streaming state changes.
        
        Updates button states, etc.
        """
        if new:
            # Change send button to stop
            try:
                send_button = self.query_one("#send-stop-chat", Button)
                send_button.label = "Stop"
                send_button.variant = "warning"
            except Exception:
                pass
        else:
            # Change stop button back to send
            try:
                send_button = self.query_one("#send-stop-chat", Button)
                send_button.label = "Send"
                send_button.variant = "primary"
            except Exception:
                pass
    
    def watch_token_count(self, old: int, new: int) -> None:
        """
        React to token count changes.
        
        Could update a token counter display.
        """
        # Update token display if it exists
        try:
            token_label = self.query_one("#token-count", Label)
            percentage = (new / self.max_tokens * 100) if self.max_tokens > 0 else 0
            token_label.update(f"{new}/{self.max_tokens} ({percentage:.0f}%)")
        except Exception:
            pass
    
    # ==================== BUTTON HANDLERS (TRANSITION) ====================
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button presses by posting messages.
        
        This is the transition from direct handlers to messages.
        """
        button_id = event.button.id
        
        if button_id == "send-stop-chat":
            if self.is_streaming:
                # Stop generation
                from ..Event_Handlers.Chat_Events.chat_messages import StopGenerationRequested
                self.post_message(StopGenerationRequested())
            else:
                # Send message
                try:
                    input_area = self.query_one("#chat-input", TextArea)
                    content = input_area.text.strip()
                    if content:
                        self.post_message(UserMessageSent(content, [self.pending_attachment] if self.pending_attachment else None))
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
        
        elif button_id == "clear-chat":
            self.post_message(ClearConversationRequested())
        
        elif button_id == "new-chat":
            self.post_message(NewSessionRequested(ephemeral=False))
        
        # Let other handlers continue for now
        # This allows gradual migration


# ==================== MIGRATION NOTES ====================
"""
To fully migrate Chat_Window_Enhanced.py:

1. Replace all direct widget manipulation with reactive updates
2. Convert button handlers to post messages
3. Add message handlers for all chat events
4. Use reactive attributes for all state
5. Remove all query_one() calls except for getting input values
6. Let Textual's reactive system handle UI updates

The key is to think in terms of state changes, not widget manipulation.
When state changes, the UI updates automatically.
"""