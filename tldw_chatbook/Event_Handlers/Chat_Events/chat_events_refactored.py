"""
Refactored chat event handlers following Textual best practices.

This module replaces direct widget manipulation with proper reactive patterns,
message-based communication, and worker-based async operations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
import asyncio
from datetime import datetime

from textual import on, work
from textual.worker import Worker, get_current_worker
from textual.reactive import reactive
from loguru import logger

# Import our new message types
from .chat_messages import (
    UserMessageSent,
    LLMResponseStarted,
    LLMResponseChunk,
    LLMResponseCompleted,
    LLMResponseError,
    ChatError,
    TokenCountUpdated,
    SessionLoaded,
    CharacterLoaded,
    RAGResultsReceived
)

# Import existing business logic (we'll use it, not duplicate it)
from tldw_chatbook.Chat.Chat_Functions import (
    chat_api_call,
    approximate_token_count,
    save_chat_history_to_db_wrapper,
    update_chat_content
)
from tldw_chatbook.DB.ChaChaNotes_DB import get_chachanotes_db_lazy

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ChatEventHandler:
    """
    Refactored chat event handler that follows Textual best practices.
    
    Key improvements:
    - No direct widget manipulation
    - All updates via messages
    - Proper use of @work decorator
    - Reactive state management
    - No blocking operations
    """
    
    def __init__(self, app: 'TldwCli'):
        """Initialize the handler with app reference."""
        self.app = app
        self.db = get_chachanotes_db_lazy()
        
        # Reactive state (should be on the widget/screen, but for compatibility...)
        self.current_session_id: Optional[str] = None
        self.is_streaming: bool = False
        self.current_worker: Optional[Worker] = None
        
    # ==================== Message Handlers ====================
    
    @on(UserMessageSent)
    async def handle_user_message(self, event: UserMessageSent) -> None:
        """
        Handle user sending a message.
        
        This handler:
        - Validates the message
        - Adds it to the conversation
        - Triggers LLM response
        - All without direct widget manipulation
        """
        logger.debug(f"Handling user message: {event.content[:50]}...")
        
        # Validate message
        if not event.content.strip():
            self.post_message(ChatError("Message cannot be empty"))
            return
        
        # Check if we're already streaming
        if self.is_streaming:
            self.post_message(ChatError("Please wait for the current response to complete"))
            return
        
        # Start processing the message
        self.process_user_message(event.content, event.attachments)
    
    @work(exclusive=True)
    async def process_user_message(self, content: str, attachments: List[str]) -> None:
        """
        Process user message in a worker.
        
        This runs in a worker thread to avoid blocking the UI.
        """
        worker = get_current_worker()
        
        try:
            # Get current configuration
            config = await self.get_chat_configuration()
            
            # Apply RAG if enabled
            rag_context = await self.get_rag_context(content)
            if rag_context:
                content = f"{rag_context}\n\n{content}"
                self.post_message_thread_safe(
                    RAGResultsReceived([], rag_context)
                )
            
            # Build chat history
            history = await self.get_chat_history()
            
            # Check token count
            token_count = approximate_token_count(history)
            max_tokens = config.get('max_tokens', 4096)
            
            if token_count > max_tokens * 0.9:
                self.post_message_thread_safe(
                    ChatError(f"Approaching token limit: {token_count}/{max_tokens}", "warning")
                )
            
            # Update token display
            self.post_message_thread_safe(
                TokenCountUpdated(token_count, max_tokens)
            )
            
            # Save user message to database
            if self.current_session_id:
                await self.save_message_to_db(
                    self.current_session_id,
                    "user",
                    content,
                    attachments
                )
            
            # Check for cancellation
            if worker.is_cancelled:
                return
            
            # Start LLM generation
            self.post_message_thread_safe(LLMResponseStarted(self.current_session_id))
            
            # Make the API call with streaming
            await self.stream_llm_response(
                content,
                history,
                config,
                worker
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            self.post_message_thread_safe(
                LLMResponseError(str(e), self.current_session_id)
            )
        finally:
            self.is_streaming = False
    
    async def stream_llm_response(
        self,
        message: str,
        history: List[Dict],
        config: Dict[str, Any],
        worker: Worker
    ) -> None:
        """
        Stream LLM response with proper cancellation support.
        """
        self.is_streaming = True
        full_response = ""
        
        try:
            # Set up streaming callback
            def stream_callback(chunk: str):
                if worker.is_cancelled:
                    return False  # Stop streaming
                
                full_response += chunk
                self.post_message_thread_safe(
                    LLMResponseChunk(chunk, self.current_session_id)
                )
                return True  # Continue streaming
            
            # Make the API call
            response = await asyncio.to_thread(
                chat_api_call,
                message=message,
                history=history,
                provider=config['provider'],
                model=config['model'],
                system_prompt=config.get('system_prompt', ''),
                temperature=config.get('temperature', 0.7),
                streaming=True,
                stream_callback=stream_callback,
                **config.get('extra_params', {})
            )
            
            # Check if cancelled
            if worker.is_cancelled:
                self.post_message_thread_safe(
                    ChatError("Generation cancelled by user")
                )
                return
            
            # Save assistant response to database
            if self.current_session_id and response:
                await self.save_message_to_db(
                    self.current_session_id,
                    "assistant",
                    response
                )
            
            # Post completion message
            self.post_message_thread_safe(
                LLMResponseCompleted(response or full_response, self.current_session_id)
            )
            
        except Exception as e:
            logger.error(f"Error in LLM streaming: {e}")
            self.post_message_thread_safe(
                LLMResponseError(str(e), self.current_session_id)
            )
    
    # ==================== Database Operations (Async) ====================
    
    async def save_message_to_db(
        self,
        conversation_id: str,
        role: str,
        content: str,
        attachments: Optional[List[str]] = None
    ) -> None:
        """Save message to database asynchronously."""
        try:
            await asyncio.to_thread(
                self.db.add_message,
                conversation_id=conversation_id,
                role=role,
                message=content,
                timestamp=datetime.now().isoformat(),
                attachments=attachments
            )
        except Exception as e:
            logger.error(f"Failed to save message to DB: {e}")
            # Don't fail the whole operation if DB save fails
    
    async def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history from database or current session."""
        if not self.current_session_id:
            return []
        
        try:
            messages = await asyncio.to_thread(
                self.db.get_messages_for_conversation,
                self.current_session_id
            )
            return messages or []
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    # ==================== Configuration ====================
    
    async def get_chat_configuration(self) -> Dict[str, Any]:
        """
        Get current chat configuration from UI.
        
        This should ideally come from reactive state, but for compatibility
        we'll gather it from the current settings.
        """
        from tldw_chatbook.config import get_cli_setting
        
        config = {
            'provider': get_cli_setting('chat_defaults', 'provider', 'openai'),
            'model': get_cli_setting('chat_defaults', 'model', 'gpt-3.5-turbo'),
            'temperature': get_cli_setting('chat_defaults', 'temperature', 0.7),
            'max_tokens': get_cli_setting('chat_defaults', 'max_tokens', 4096),
            'system_prompt': get_cli_setting('chat_defaults', 'system_prompt', ''),
            'streaming': get_cli_setting('chat_defaults', 'enable_streaming', True),
            'extra_params': {}
        }
        
        # Add any additional parameters
        for param in ['top_p', 'top_k', 'min_p', 'presence_penalty', 'frequency_penalty']:
            value = get_cli_setting('chat_defaults', param, None)
            if value is not None:
                config['extra_params'][param] = value
        
        return config
    
    # ==================== RAG Integration ====================
    
    async def get_rag_context(self, query: str) -> Optional[str]:
        """Get RAG context if enabled."""
        from tldw_chatbook.config import get_cli_setting
        
        if not get_cli_setting('rag', 'enabled', False):
            return None
        
        try:
            # This would integrate with the RAG system
            # For now, return None
            return None
        except Exception as e:
            logger.error(f"RAG context failed: {e}")
            return None
    
    # ==================== Utility Methods ====================
    
    def post_message_thread_safe(self, message: Any) -> None:
        """
        Post a message from a worker thread.
        
        This ensures thread-safe message posting.
        """
        if hasattr(self.app, 'call_from_thread'):
            self.app.call_from_thread(self.app.post_message, message)
        else:
            # Fallback for testing
            self.app.post_message(message)
    
    def post_message(self, message: Any) -> None:
        """Post a message from the main thread."""
        self.app.post_message(message)


# ==================== Refactored Handler Functions ====================

async def handle_send_button_pressed(app: 'TldwCli', event: Any) -> None:
    """
    Refactored send button handler.
    
    Instead of direct manipulation, posts a message.
    """
    # Get the input content (this is the only direct query we need)
    try:
        from textual.widgets import TextArea
        text_area = app.query_one("#chat-input", TextArea)
        content = text_area.text.strip()
        
        if content:
            # Post message instead of direct manipulation
            app.post_message(UserMessageSent(content))
            
            # Clear input
            text_area.clear()
    except Exception as e:
        logger.error(f"Error in send button handler: {e}")
        app.post_message(ChatError(str(e)))


async def handle_stop_generation(app: 'TldwCli', event: Any) -> None:
    """
    Refactored stop generation handler.
    
    Cancels the current worker instead of manipulating state.
    """
    # Find and cancel any active chat workers
    for worker in app.workers:
        if worker.name and 'chat' in worker.name.lower():
            worker.cancel()
            logger.info("Cancelled chat generation worker")
            app.post_message(ChatError("Generation stopped", "info"))
            return
    
    app.post_message(ChatError("No active generation to stop", "warning"))


async def handle_new_session(app: 'TldwCli', ephemeral: bool = False) -> None:
    """
    Refactored new session handler.
    
    Creates session through proper channels.
    """
    from .chat_messages import NewSessionRequested
    app.post_message(NewSessionRequested(ephemeral))


async def handle_save_session(app: 'TldwCli', title: str, keywords: List[str]) -> None:
    """
    Refactored save session handler.
    """
    from .chat_messages import SaveSessionRequested
    app.post_message(SaveSessionRequested(title, keywords))


async def handle_load_session(app: 'TldwCli', session_id: str) -> None:
    """
    Refactored load session handler.
    """
    from .chat_messages import LoadSessionRequested
    app.post_message(LoadSessionRequested(session_id))