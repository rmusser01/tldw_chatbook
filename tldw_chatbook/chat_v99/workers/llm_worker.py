"""Worker for LLM interactions with streaming support."""

from typing import AsyncGenerator, List, Optional
from dataclasses import dataclass
import asyncio
import random

try:
    from ..models import Settings, ChatMessage
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import Settings, ChatMessage


@dataclass
class StreamChunk:
    """Represents a chunk of streaming response."""
    content: str
    done: bool = False
    error: Optional[str] = None


class LLMWorker:
    """Worker for LLM interactions.
    
    This worker handles LLM API calls and streaming responses.
    In production, this would integrate with actual LLM APIs.
    """
    
    def __init__(self, settings: Settings):
        """Initialize LLM worker with settings.
        
        Args:
            settings: Application settings including provider and model
        """
        self.settings = settings
    
    async def stream_completion(
        self,
        prompt: str,
        messages: List[ChatMessage] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream LLM completion response.
        
        Args:
            prompt: The user's prompt
            messages: Previous messages for context
            **kwargs: Additional parameters for the LLM
            
        Yields:
            StreamChunk objects with content and status
        """
        # In production, this would use the actual LLM API
        # For now, simulate streaming with a sample response
        
        if self.settings.streaming:
            # Simulate streaming response
            async for chunk in self._simulate_streaming(prompt, messages):
                yield chunk
        else:
            # Non-streaming response
            response = await self._simulate_response(prompt, messages)
            yield StreamChunk(content=response, done=True)
    
    async def _simulate_streaming(
        self,
        prompt: str,
        messages: List[ChatMessage] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Simulate streaming response for testing.
        
        Args:
            prompt: The user's prompt
            messages: Previous messages for context
            
        Yields:
            StreamChunk objects simulating streaming
        """
        # Generate a contextual response based on the prompt
        response = self._generate_sample_response(prompt, messages)
        
        # Split into words for streaming simulation
        words = response.split()
        
        # Stream word by word with realistic delays
        for i, word in enumerate(words):
            # Add space before word (except first)
            if i > 0:
                yield StreamChunk(content=" ", done=False)
            
            # Stream the word character by character for effect
            for char in word:
                yield StreamChunk(content=char, done=False)
                # Small delay between characters
                await asyncio.sleep(random.uniform(0.01, 0.03))
        
        # Signal completion
        yield StreamChunk(content="", done=True)
    
    async def _simulate_response(
        self,
        prompt: str,
        messages: List[ChatMessage] = None
    ) -> str:
        """Simulate non-streaming response.
        
        Args:
            prompt: The user's prompt
            messages: Previous messages for context
            
        Returns:
            Complete response string
        """
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        return self._generate_sample_response(prompt, messages)
    
    def _generate_sample_response(
        self,
        prompt: str,
        messages: List[ChatMessage] = None
    ) -> str:
        """Generate a sample response based on the prompt.
        
        Args:
            prompt: The user's prompt
            messages: Previous messages for context
            
        Returns:
            A contextual sample response
        """
        # Convert prompt to lowercase for checking
        prompt_lower = prompt.lower()
        
        # Contextual responses based on prompt content
        if "hello" in prompt_lower or "hi" in prompt_lower:
            responses = [
                "Hello! I'm the Chat v99 assistant running on Textual's reactive framework. How can I help you today?",
                "Hi there! Welcome to the new chat interface. What would you like to discuss?",
                "Greetings! This chat interface follows Textual's best practices. How may I assist you?"
            ]
        elif "how are you" in prompt_lower:
            responses = [
                "I'm functioning perfectly with reactive state management! The new architecture is working smoothly.",
                "I'm doing great! All systems are operational and following Textual patterns.",
                "Excellent! The reactive updates and message-based communication are working flawlessly."
            ]
        elif "test" in prompt_lower:
            responses = [
                "This is a test response demonstrating the streaming capability of Chat v99. "
                "The text is being streamed character by character using async generators and reactive updates.",
                "Testing confirmed! The message system is working correctly. "
                "All events are bubbling properly and reactive state is updating as expected.",
                "Test successful! The worker pattern is processing messages without blocking the UI."
            ]
        elif "help" in prompt_lower:
            responses = [
                "I can help you with various tasks! This chat interface supports:\n"
                "• Real-time streaming responses\n"
                "• Session management\n"
                "• Multiple LLM providers\n"
                "• File attachments (simulated)\n"
                "• Reactive state updates\n\n"
                "Try different commands or ask me questions!",
                "Here's what you can do:\n"
                "• Ctrl+N: New session\n"
                "• Ctrl+S: Save session\n"
                "• Ctrl+\\: Toggle sidebar\n"
                "• Ctrl+K: Clear messages\n"
                "• Ctrl+Enter: Send message\n\n"
                "What would you like to know more about?"
            ]
        elif "code" in prompt_lower or "programming" in prompt_lower:
            responses = [
                "This chat interface is built with Textual, a modern TUI framework. "
                "It uses reactive programming patterns, message-based communication, "
                "and CSS-driven layouts. The architecture follows best practices from "
                "the official Textual documentation.",
                "The codebase demonstrates several key patterns:\n"
                "• Reactive attributes with proper typing\n"
                "• Worker threads for async operations\n"
                "• Custom Textual messages for events\n"
                "• Inline CSS for styling\n"
                "• Proper widget composition",
                "Would you like to see some code examples? The implementation includes "
                "reactive state management, streaming support, and proper separation of concerns."
            ]
        elif "textual" in prompt_lower:
            responses = [
                "Textual is an amazing framework for building TUIs! This chat interface "
                "showcases its capabilities including reactive programming, CSS styling, "
                "and event-driven architecture. Every component follows Textual's best practices.",
                "The Textual framework provides powerful features that this chat uses:\n"
                "• Reactive attributes that automatically update the UI\n"
                "• CSS-like styling with transitions and animations\n"
                "• Async worker support for non-blocking operations\n"
                "• Rich widget library with customization options",
                "This implementation strictly follows Textual's documentation patterns. "
                "No direct widget manipulation, proper use of workers, and reactive updates throughout!"
            ]
        else:
            # Default responses for general queries
            responses = [
                f"I understand you're asking about '{prompt}'. "
                f"This is a simulated response from the Chat v99 system. "
                f"In production, this would connect to a real LLM API like {self.settings.provider}'s {self.settings.model}.",
                f"Regarding your message: '{prompt}'\n\n"
                "This demonstrates the streaming capability of the new chat interface. "
                "Each character is being sent as a separate chunk and rendered reactively.",
                f"You said: '{prompt}'\n\n"
                "This response is being streamed to showcase the reactive update system. "
                "The UI remains responsive while the message is being generated."
            ]
        
        # Add context awareness if there are previous messages
        if messages and len(messages) > 2:
            response = random.choice(responses)
            response += f"\n\n(Context: This is message #{len(messages) + 1} in our conversation)"
        else:
            response = random.choice(responses)
        
        return response
    
    async def complete(
        self,
        prompt: str,
        messages: List[ChatMessage] = None,
        **kwargs
    ) -> str:
        """Get a complete (non-streaming) response.
        
        Args:
            prompt: The user's prompt
            messages: Previous messages for context
            **kwargs: Additional parameters
            
        Returns:
            Complete response string
        """
        return await self._simulate_response(prompt, messages)
    
    def validate_settings(self) -> bool:
        """Validate that the worker has necessary settings.
        
        Returns:
            True if settings are valid
        """
        if not self.settings.provider:
            return False
        if not self.settings.model:
            return False
        # In production, would check API keys etc.
        return True