"""Worker for REAL LLM interactions - NO SIMULATION."""

from typing import AsyncGenerator, List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import json
from loguru import logger

try:
    from ..models import Settings, ChatMessage
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import Settings, ChatMessage

# Import REAL LLM API functions
from tldw_chatbook.LLM_Calls.LLM_API_Calls import (
    chat_with_openai, chat_with_anthropic, chat_with_cohere,
    chat_with_groq, chat_with_openrouter, chat_with_deepseek,
    chat_with_mistral, chat_with_huggingface, chat_with_google,
    chat_with_moonshot, chat_with_zai
)
from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
    chat_with_ollama, chat_with_llama, chat_with_kobold,
    chat_with_oobabooga, chat_with_tabbyapi, chat_with_vllm,
    chat_with_local_llm, chat_with_custom_openai
)
from tldw_chatbook.config import get_api_key


@dataclass
class StreamChunk:
    """Represents a chunk of streaming response."""
    content: str
    done: bool = False
    error: Optional[str] = None
    token_count: Optional[int] = None


class LLMWorker:
    """Worker for REAL LLM interactions - connects to actual APIs."""
    
    # Map providers to their handler functions
    PROVIDER_HANDLERS = {
        'openai': chat_with_openai,
        'anthropic': chat_with_anthropic,
        'cohere': chat_with_cohere,
        'groq': chat_with_groq,
        'openrouter': chat_with_openrouter,
        'deepseek': chat_with_deepseek,
        'mistral': chat_with_mistral,
        'mistralai': chat_with_mistral,
        'google': chat_with_google,
        'huggingface': chat_with_huggingface,
        'moonshot': chat_with_moonshot,
        'zai': chat_with_zai,
        'ollama': chat_with_ollama,
        'llama_cpp': chat_with_llama,
        'koboldcpp': chat_with_kobold,
        'oobabooga': chat_with_oobabooga,
        'tabbyapi': chat_with_tabbyapi,
        'vllm': chat_with_vllm,
        'local-llm': chat_with_local_llm,
        'custom_openai': chat_with_custom_openai,
    }
    
    def __init__(self, settings: Settings):
        """Initialize LLM worker with settings.
        
        Args:
            settings: Application settings including provider and model
        """
        self.settings = settings
        self._stop_requested = False
    
    def stop_generation(self):
        """Stop the current generation."""
        self._stop_requested = True
    
    async def stream_completion(
        self,
        prompt: str,
        messages: List[ChatMessage] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream REAL LLM completion response.
        
        Args:
            prompt: The user's prompt
            messages: Previous messages for context
            **kwargs: Additional parameters for the LLM
            
        Yields:
            StreamChunk objects with content and status
        """
        self._stop_requested = False
        
        # Get the handler for this provider
        handler = self.PROVIDER_HANDLERS.get(self.settings.provider.lower())
        if not handler:
            yield StreamChunk(
                content=f"Error: Unsupported provider '{self.settings.provider}'",
                done=True,
                error=f"Provider '{self.settings.provider}' not supported"
            )
            return
        
        # Get API key
        api_key = self.settings.api_key or get_api_key(self.settings.provider)
        if not api_key and self.settings.provider not in ['ollama', 'local-llm', 'llama_cpp']:
            yield StreamChunk(
                content=f"Error: No API key for {self.settings.provider}",
                done=True,
                error="Missing API key"
            )
            return
        
        try:
            # Prepare conversation history for the API
            conversation_history = self._prepare_conversation(messages, prompt)
            
            # Call the REAL API
            response = await asyncio.to_thread(
                handler,
                conversation_history,
                api_key=api_key,
                model=self.settings.model,
                temp=self.settings.temperature,
                system_prompt=self.settings.system_prompt or "",
                streaming=self.settings.streaming,
                max_tokens=self.settings.max_tokens
            )
            
            if self.settings.streaming and hasattr(response, '__iter__'):
                # Handle streaming response
                accumulated = ""
                for chunk in response:
                    if self._stop_requested:
                        yield StreamChunk(content="", done=True)
                        break
                    
                    content = self._extract_content(chunk)
                    if content:
                        accumulated += content
                        yield StreamChunk(content=content, done=False)
                
                yield StreamChunk(content="", done=True)
            else:
                # Non-streaming response
                content = self._extract_content(response)
                yield StreamChunk(content=content, done=True)
                
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            yield StreamChunk(
                content=f"Error calling {self.settings.provider}: {str(e)}",
                done=True,
                error=str(e)
            )
    
    def _prepare_conversation(
        self,
        messages: List[ChatMessage],
        current_prompt: str
    ) -> List[List[str]]:
        """Prepare conversation history for API calls.
        
        Args:
            messages: Previous messages
            current_prompt: Current user prompt
            
        Returns:
            List of [user, assistant] pairs for the API
        """
        history = []
        
        if messages:
            user_msg = None
            for msg in messages:
                if msg.role == "user":
                    user_msg = msg.content
                elif msg.role == "assistant" and user_msg:
                    history.append([user_msg, msg.content])
                    user_msg = None
        
        # Add current prompt with empty response
        history.append([current_prompt, ""])
        
        return history
    
    def _extract_content(self, response: Any) -> str:
        """Extract content from various API response formats.
        
        Args:
            response: Response from the API
            
        Returns:
            Extracted text content
        """
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            # OpenAI format
            if 'choices' in response:
                choices = response.get('choices', [])
                if choices:
                    if 'delta' in choices[0]:
                        return choices[0]['delta'].get('content', '')
                    elif 'message' in choices[0]:
                        return choices[0]['message'].get('content', '')
                    elif 'text' in choices[0]:
                        return choices[0]['text']
            # Direct content
            elif 'content' in response:
                return response['content']
            elif 'text' in response:
                return response['text']
        elif isinstance(response, tuple) and len(response) == 2:
            # Some APIs return (input, output) tuple
            return response[1]
        
        return str(response)
    
    async def complete(
        self,
        prompt: str,
        messages: List[ChatMessage] = None,
        **kwargs
    ) -> str:
        """Get a complete (non-streaming) response from REAL LLM.
        
        Args:
            prompt: The user's prompt
            messages: Previous messages for context
            **kwargs: Additional parameters
            
        Returns:
            Complete response string
        """
        result = ""
        async for chunk in self.stream_completion(prompt, messages, **kwargs):
            if chunk.error:
                return f"Error: {chunk.error}"
            result += chunk.content
        return result
    
    def validate_settings(self) -> bool:
        """Validate that the worker has necessary settings.
        
        Returns:
            True if settings are valid
        """
        if not self.settings.provider:
            logger.error("No provider configured")
            return False
        if not self.settings.model:
            logger.error("No model configured")
            return False
        
        # Check if provider is supported
        if self.settings.provider.lower() not in self.PROVIDER_HANDLERS:
            logger.error(f"Unsupported provider: {self.settings.provider}")
            return False
        
        return True