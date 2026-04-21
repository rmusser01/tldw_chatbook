"""
Refactored streaming event handlers using reactive patterns.

This module handles streaming LLM responses using Textual's reactive
attributes and message system instead of direct widget manipulation.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import asyncio

from textual.reactive import reactive
from textual.message import Message
from textual import on
from loguru import logger

from .chat_messages import (
    LLMResponseChunk,
    LLMResponseCompleted,
    LLMResponseError
)


@dataclass 
class StreamingState:
    """Reactive state for streaming responses."""
    is_streaming: bool = False
    current_content: str = ""
    message_index: Optional[int] = None
    session_id: Optional[str] = None


class StreamingHandler:
    """
    Handles streaming LLM responses with reactive updates.
    
    Key improvements:
    - Uses reactive attributes for state
    - No direct widget manipulation
    - Proper message passing
    - Efficient chunk batching
    """
    
    # Reactive streaming state
    streaming_state: reactive[StreamingState] = reactive(StreamingState())
    streaming_buffer: reactive[str] = reactive("")
    
    def __init__(self):
        """Initialize the streaming handler."""
        self._chunk_buffer = []
        self._buffer_timer = None
        self._batch_size = 5  # Batch chunks for efficiency
        self._batch_delay = 0.05  # 50ms delay for batching
    
    # ==================== Message Handlers ====================
    
    @on(LLMResponseChunk)
    async def handle_streaming_chunk(self, event: LLMResponseChunk) -> None:
        """
        Handle incoming streaming chunk with batching.
        
        Instead of updating the widget directly, we:
        1. Update reactive state
        2. Batch chunks for efficiency
        3. Let reactive system handle UI updates
        """
        if not self.streaming_state.is_streaming:
            # Start streaming
            self.streaming_state = StreamingState(
                is_streaming=True,
                current_content="",
                session_id=event.session_id
            )
        
        # Add chunk to buffer
        self._chunk_buffer.append(event.chunk)
        
        # Batch chunks for efficiency
        if len(self._chunk_buffer) >= self._batch_size:
            await self._flush_buffer()
        else:
            # Set timer to flush buffer after delay
            if self._buffer_timer:
                self._buffer_timer.cancel()
            
            self._buffer_timer = asyncio.create_task(
                self._delayed_flush()
            )
    
    async def _delayed_flush(self) -> None:
        """Flush buffer after a delay."""
        await asyncio.sleep(self._batch_delay)
        await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """
        Flush the chunk buffer to reactive state.
        
        This triggers UI updates through the reactive system.
        """
        if not self._chunk_buffer:
            return
        
        # Combine chunks
        combined_chunk = "".join(self._chunk_buffer)
        self._chunk_buffer.clear()
        
        # Update reactive state (triggers UI update)
        current = self.streaming_state.current_content
        self.streaming_state = StreamingState(
            is_streaming=True,
            current_content=current + combined_chunk,
            session_id=self.streaming_state.session_id
        )
        
        # Also update the buffer for display
        self.streaming_buffer = self.streaming_state.current_content
    
    @on(LLMResponseCompleted)
    async def handle_stream_done(self, event: LLMResponseCompleted) -> None:
        """
        Handle stream completion.
        
        Finalizes the streaming state and triggers final UI update.
        """
        # Flush any remaining chunks
        await self._flush_buffer()
        
        # Update state to completed
        self.streaming_state = StreamingState(
            is_streaming=False,
            current_content=event.full_response,
            session_id=event.session_id
        )
        
        # Clear buffer timer
        if self._buffer_timer:
            self._buffer_timer.cancel()
            self._buffer_timer = None
        
        logger.debug(f"Streaming completed for session {event.session_id}")
    
    @on(LLMResponseError)
    async def handle_stream_error(self, event: LLMResponseError) -> None:
        """Handle streaming error."""
        # Reset streaming state
        self.streaming_state = StreamingState(is_streaming=False)
        
        # Clear buffers
        self._chunk_buffer.clear()
        self.streaming_buffer = ""
        
        if self._buffer_timer:
            self._buffer_timer.cancel()
            self._buffer_timer = None
        
        logger.error(f"Streaming error: {event.error}")
    
    # ==================== Watch Methods (Reactive) ====================
    
    def watch_streaming_state(self, old_state: StreamingState, new_state: StreamingState) -> None:
        """
        React to streaming state changes.
        
        This is called automatically when streaming_state changes.
        The UI will update based on this state.
        """
        if new_state.is_streaming and not old_state.is_streaming:
            logger.debug("Streaming started")
        elif not new_state.is_streaming and old_state.is_streaming:
            logger.debug("Streaming ended")
    
    def watch_streaming_buffer(self, old_buffer: str, new_buffer: str) -> None:
        """
        React to buffer changes.
        
        This triggers UI updates for the streaming content.
        """
        # The UI will automatically update based on this reactive attribute
        pass
    
    # ==================== Public Methods ====================
    
    def start_streaming(self, session_id: Optional[str] = None) -> None:
        """Start a new streaming session."""
        self.streaming_state = StreamingState(
            is_streaming=True,
            current_content="",
            session_id=session_id
        )
        self.streaming_buffer = ""
        self._chunk_buffer.clear()
    
    def stop_streaming(self) -> None:
        """Stop the current streaming session."""
        self.streaming_state = StreamingState(is_streaming=False)
        
        if self._buffer_timer:
            self._buffer_timer.cancel()
            self._buffer_timer = None
        
        self._chunk_buffer.clear()
    
    def get_current_content(self) -> str:
        """Get the current streaming content."""
        return self.streaming_state.current_content
    
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self.streaming_state.is_streaming


class ReactiveStreamingWidget:
    """
    Example widget showing how to use streaming with reactive patterns.
    
    This would be mixed into the actual chat widget.
    """
    
    # Reactive attribute for streaming content
    streaming_content: reactive[str] = reactive("")
    is_streaming: reactive[bool] = reactive(False)
    
    def compose(self):
        """
        Compose method that uses reactive state.
        
        The UI automatically updates when streaming_content changes.
        """
        # This is just an example - the actual implementation would be in the chat widget
        pass
    
    def watch_streaming_content(self, old_content: str, new_content: str) -> None:
        """
        Watch for streaming content changes.
        
        This is called automatically when streaming_content changes.
        No manual widget manipulation needed!
        """
        # The UI updates automatically through the reactive system
        # No need for manual updates or queries
        pass
    
    @on(LLMResponseChunk)
    def on_llm_response_chunk(self, event: LLMResponseChunk) -> None:
        """
        Handle streaming chunk message.
        
        Just update the reactive attribute - the UI updates automatically!
        """
        self.streaming_content = self.streaming_content + event.chunk
        self.is_streaming = True
    
    @on(LLMResponseCompleted)
    def on_llm_response_completed(self, event: LLMResponseCompleted) -> None:
        """Handle stream completion."""
        self.streaming_content = event.full_response
        self.is_streaming = False
    
    @on(LLMResponseError)
    def on_llm_response_error(self, event: LLMResponseError) -> None:
        """Handle streaming error."""
        self.is_streaming = False
        # Could show error in UI through another reactive attribute