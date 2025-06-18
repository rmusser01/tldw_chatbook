# llm_interface.py
# Description: LLM provider interface for evaluations
#
"""
LLM Interface for Evaluations
----------------------------

Provides a unified interface for calling different LLM providers during evaluations.
Supports both commercial and local LLM providers with async operations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger

from ...LLM_Calls.LLM_API_Calls import (
    chat_with_openai, chat_with_anthropic, chat_with_cohere, 
    chat_with_groq, chat_with_openrouter, chat_with_huggingface, 
    chat_with_deepseek
)
from ...config import load_settings

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.config = config
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously."""
        pass
    
    @abstractmethod
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities asynchronously."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('openai_api_key')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            # Run synchronous function in thread pool
            loop = asyncio.get_event_loop()
            
            # Prepare arguments for the OpenAI function
            generation_config = {
                'temperature': kwargs.get('temperature', 0.0),
                'max_tokens': kwargs.get('max_tokens', 100)
            }
            
            result = await loop.run_in_executor(
                None,
                self._call_openai_sync,
                prompt,
                generation_config
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def _call_openai_sync(self, prompt: str, config: Dict[str, Any]) -> str:
        """Call OpenAI API synchronously."""
        # Create a temporary file-like object with the prompt
        # This is needed because the existing function expects a file path
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(prompt)
            temp_file_path = temp_file.name
        
        try:
            result = chat_with_openai(
                api_key=self.api_key,
                file_path=temp_file_path,
                custom_prompt_arg="",  # Empty since prompt is in file
                streaming=False
            )
            return result
        finally:
            os.unlink(temp_file_path)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities (not fully implemented)."""
        logger.warning("Log probabilities not yet implemented for OpenAI provider")
        return {'logprobs': [], 'tokens': []}

class AnthropicProvider(LLMProvider):
    """Anthropic provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('anthropic_api_key')
        if not self.api_key:
            raise ValueError("Anthropic API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_anthropic_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def _call_anthropic_sync(self, prompt: str, config: Dict[str, Any]) -> str:
        """Call Anthropic API synchronously."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(prompt)
            temp_file_path = temp_file.name
        
        try:
            result = chat_with_anthropic(
                api_key=self.api_key,
                file_path=temp_file_path,
                model=self.model_id,
                custom_prompt_arg="",
                streaming=False
            )
            return result
        finally:
            os.unlink(temp_file_path)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities (not fully implemented)."""
        logger.warning("Log probabilities not yet implemented for Anthropic provider")
        return {'logprobs': [], 'tokens': []}

class CohereProvider(LLMProvider):
    """Cohere provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('cohere_api_key')
        if not self.api_key:
            raise ValueError("Cohere API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Cohere API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_cohere_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Cohere generation failed: {e}")
            raise
    
    def _call_cohere_sync(self, prompt: str, config: Dict[str, Any]) -> str:
        """Call Cohere API synchronously."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(prompt)
            temp_file_path = temp_file.name
        
        try:
            result = chat_with_cohere(
                api_key=self.api_key,
                file_path=temp_file_path,
                model=self.model_id,
                custom_prompt_arg="",
                streaming=False
            )
            return result
        finally:
            os.unlink(temp_file_path)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities (not fully implemented)."""
        logger.warning("Log probabilities not yet implemented for Cohere provider")
        return {'logprobs': [], 'tokens': []}

class GroqProvider(LLMProvider):
    """Groq provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('groq_api_key')
        if not self.api_key:
            raise ValueError("Groq API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Groq API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_groq_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def _call_groq_sync(self, prompt: str, config: Dict[str, Any]) -> str:
        """Call Groq API synchronously."""
        result = chat_with_groq(
            api_key=self.api_key,
            input_data=prompt,
            custom_prompt_arg="",
            streaming=False
        )
        return result
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities (not fully implemented)."""
        logger.warning("Log probabilities not yet implemented for Groq provider")
        return {'logprobs': [], 'tokens': []}

class OpenRouterProvider(LLMProvider):
    """OpenRouter provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('openrouter_api_key')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenRouter API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_openrouter_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise
    
    def _call_openrouter_sync(self, prompt: str, config: Dict[str, Any]) -> str:
        """Call OpenRouter API synchronously."""
        result = chat_with_openrouter(
            api_key=self.api_key,
            input_data=prompt,
            custom_prompt_arg="",
            streaming=False
        )
        return result
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities (not fully implemented)."""
        logger.warning("Log probabilities not yet implemented for OpenRouter provider")
        return {'logprobs': [], 'tokens': []}

def get_llm_provider(provider_name: str, model_id: str, config: Dict[str, Any]) -> LLMProvider:
    """Get an LLM provider instance based on provider name."""
    provider_classes = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'cohere': CohereProvider,
        'groq': GroqProvider,
        'openrouter': OpenRouterProvider,
    }
    
    if provider_name.lower() not in provider_classes:
        raise ValueError(f"Unsupported provider: {provider_name}")
    
    provider_class = provider_classes[provider_name.lower()]
    return provider_class(model_id, config)

# For backwards compatibility and easier usage
class LLMInterface:
    """Main interface for LLM operations in evaluations."""
    
    def __init__(self, provider_name: str, model_id: str, config: Dict[str, Any]):
        self.provider = get_llm_provider(provider_name, model_id, config)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text."""
        return await self.provider.generate_async(prompt, **kwargs)
    
    async def get_logprobs(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities."""
        return await self.provider.get_logprobs_async(text, **kwargs)