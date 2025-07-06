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

from tldw_chatbook.config import load_settings

# Import functions at runtime to avoid circular imports
def _get_llm_functions():
    """Import LLM functions at runtime to avoid circular imports."""
    from ...LLM_Calls.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere, 
        chat_with_groq, chat_with_openrouter, chat_with_huggingface, 
        chat_with_deepseek
    )
    return {
        'chat_with_openai': chat_with_openai,
        'chat_with_anthropic': chat_with_anthropic,
        'chat_with_cohere': chat_with_cohere,
        'chat_with_groq': chat_with_groq,
        'chat_with_openrouter': chat_with_openrouter,
        'chat_with_huggingface': chat_with_huggingface,
        'chat_with_deepseek': chat_with_deepseek
    }

def _get_local_llm_functions():
    """Import local LLM functions at runtime to avoid circular imports."""
    from ...LLM_Calls.LLM_API_Calls_Local import (
        chat_with_ollama, chat_with_vllm, chat_with_llama
    )
    return {
        'chat_with_ollama': chat_with_ollama,
        'chat_with_vllm': chat_with_vllm,
        'chat_with_llama': chat_with_llama
    }

# Define our own error classes to avoid circular imports
class EvalProviderError(Exception):
    """Base error for evaluation provider issues."""
    def __init__(self, message: str, provider: str = None):
        self.provider = provider
        super().__init__(message)

class EvalAPIError(EvalProviderError):
    """API-related errors during evaluation."""
    pass

class EvalAuthenticationError(EvalProviderError):
    """Authentication errors during evaluation."""
    pass

class EvalRateLimitError(EvalProviderError):
    """Rate limit errors during evaluation."""
    pass

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
    
    async def get_completion_logprobs_async(self, prompt: str, completion: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities for a specific completion (default implementation)."""
        # Default implementation - just use the completion as text for logprobs
        return await self.get_logprobs_async(completion, **kwargs)
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt asynchronously (default implementation)."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)

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
            
            result = await loop.run_in_executor(
                None,
                self._call_openai_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"OpenAI authentication error: {e}")
                raise EvalAuthenticationError(f"OpenAI authentication failed: {e}", provider="openai")
            elif 'rate limit' in str(e).lower():
                logger.error(f"OpenAI rate limit error: {e}")
                raise EvalRateLimitError(f"OpenAI rate limit exceeded: {e}", provider="openai")
            else:
                logger.error(f"OpenAI generation failed: {e}")
                raise EvalProviderError(f"OpenAI generation failed: {e}", provider="openai")
    
    def _call_openai_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call OpenAI API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_openai = llm_functions['chat_with_openai']
        
        # Prepare message format expected by OpenAI API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_openai(
            input_data=messages,
            api_key=self.api_key,
            model=self.model_id,
            system_message=kwargs.get('system_prompt'),
            temp=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 100),
            logprobs=kwargs.get('logprobs', False),
            top_logprobs=kwargs.get('top_logprobs'),
            streaming=False
        )
        return result
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from OpenAI API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._get_logprobs_sync,
                text,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI logprobs failed: {e}")
            return {'logprobs': [], 'tokens': [], 'error': str(e)}
    
    async def get_completion_logprobs_async(self, prompt: str, completion: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities for a specific completion given a prompt."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._get_completion_logprobs_sync,
                prompt,
                completion,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI completion logprobs failed: {e}")
            return {'logprobs': [], 'tokens': [], 'error': str(e)}
    
    def _get_completion_logprobs_sync(self, prompt: str, completion: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get logprobs for a specific completion synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_openai = llm_functions['chat_with_openai']
        
        # For completion logprobs, we need to evaluate the probability of the completion
        # given the prompt. This is typically done by providing the prompt and asking
        # the model to "continue" with high temperature and logprobs enabled
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        
        try:
            # Use the completion as the expected response to get its probability
            response_data = chat_with_openai(
                input_data=messages,
                api_key=self.api_key,
                model=self.model_id,
                temp=0.0,
                max_tokens=len(completion.split()) + 10,  # Allow enough tokens
                logprobs=True,
                top_logprobs=kwargs.get('top_logprobs', 5),
                streaming=False
            )
            
            # Extract logprobs and compare with expected completion
            extracted_logprobs = self._extract_openai_logprobs(response_data)
            extracted_logprobs['expected_completion'] = completion
            extracted_logprobs['prompt'] = prompt
            
            return extracted_logprobs
            
        except Exception as e:
            logger.error(f"Failed to get completion logprobs: {e}")
            return {
                'logprobs': [],
                'tokens': [],
                'error': str(e),
                'prompt': prompt,
                'expected_completion': completion
            }
    
    def _get_logprobs_sync(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get log probabilities synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_openai = llm_functions['chat_with_openai']
        
        # For logprobs, we want to evaluate the probability of the given text
        # We'll use the text as a completion and ask for logprobs
        messages = [{"role": "user", "content": text}]
        
        try:
            response_data = chat_with_openai(
                input_data=messages,
                api_key=self.api_key,
                model=self.model_id,
                temp=0.0,
                max_tokens=1,  # Minimal generation, focus on logprobs
                logprobs=True,
                top_logprobs=kwargs.get('top_logprobs', 5),
                streaming=False
            )
            
            # Extract logprobs from OpenAI response format
            extracted_logprobs = self._extract_openai_logprobs(response_data)
            return extracted_logprobs
            
        except Exception as e:
            logger.error(f"Failed to get logprobs: {e}")
            return {'logprobs': [], 'tokens': [], 'error': str(e)}
    
    def _extract_openai_logprobs(self, response_data: Any) -> Dict[str, Any]:
        """Extract logprobs from OpenAI API response."""
        try:
            # Handle both string responses (text only) and dict responses (full API response)
            if isinstance(response_data, str):
                # If we just get text back, logprobs weren't available
                return {
                    'logprobs': [],
                    'tokens': [],
                    'content': response_data,
                    'note': 'Response was text-only, no logprobs structure available'
                }
            
            if not isinstance(response_data, dict):
                return {
                    'logprobs': [],
                    'tokens': [],
                    'error': f'Unexpected response type: {type(response_data)}'
                }
            
            # Extract from OpenAI response structure
            choices = response_data.get('choices', [])
            if not choices:
                return {
                    'logprobs': [],
                    'tokens': [],
                    'error': 'No choices in response'
                }
            
            choice = choices[0]  # Get first choice
            logprobs_data = choice.get('logprobs')
            
            if not logprobs_data:
                return {
                    'logprobs': [],
                    'tokens': [],
                    'content': choice.get('message', {}).get('content', ''),
                    'note': 'No logprobs data in response'
                }
            
            # Extract tokens and their probabilities
            content_tokens = logprobs_data.get('content', [])
            tokens = []
            logprobs = []
            top_logprobs = []
            
            for token_data in content_tokens:
                if isinstance(token_data, dict):
                    tokens.append(token_data.get('token', ''))
                    logprobs.append(token_data.get('logprob', 0.0))
                    
                    # Extract top alternatives if available
                    top_alternatives = token_data.get('top_logprobs', [])
                    top_logprobs.append([
                        {
                            'token': alt.get('token', ''),
                            'logprob': alt.get('logprob', 0.0)
                        }
                        for alt in top_alternatives if isinstance(alt, dict)
                    ])
            
            return {
                'tokens': tokens,
                'logprobs': logprobs,
                'top_logprobs': top_logprobs,
                'content': choice.get('message', {}).get('content', ''),
                'model': response_data.get('model', ''),
                'usage': response_data.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"Error extracting OpenAI logprobs: {e}")
            return {
                'logprobs': [],
                'tokens': [],
                'error': f'Extraction failed: {str(e)}',
                'raw_response': response_data
            }

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
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"Anthropic authentication error: {e}")
                raise EvalAuthenticationError(f"Anthropic authentication failed: {e}", provider="anthropic")
            elif 'rate limit' in str(e).lower():
                logger.error(f"Anthropic rate limit error: {e}")
                raise EvalRateLimitError(f"Anthropic rate limit exceeded: {e}", provider="anthropic")
            else:
                logger.error(f"Anthropic generation failed: {e}")
                raise EvalProviderError(f"Anthropic generation failed: {e}", provider="anthropic")
    
    def _call_anthropic_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Anthropic API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_anthropic = llm_functions['chat_with_anthropic']
        
        # Prepare message format expected by Anthropic API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_anthropic(
            input_data=messages,
            api_key=self.api_key,
            model=self.model_id,
            system_prompt=kwargs.get('system_prompt'),
            temp=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 100),
            streaming=False
        )
        return result
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from Anthropic API."""
        # Note: Anthropic Claude models don't directly support logprobs like OpenAI
        # This is a placeholder for future implementation if they add this feature
        logger.warning("Log probabilities not directly supported by Anthropic API")
        return {'logprobs': [], 'tokens': [], 'note': 'Anthropic does not support logprobs'}

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
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"Cohere authentication error: {e}")
                raise EvalAuthenticationError(f"Cohere authentication failed: {e}", provider="cohere")
            elif 'rate limit' in str(e).lower():
                logger.error(f"Cohere rate limit error: {e}")
                raise EvalRateLimitError(f"Cohere rate limit exceeded: {e}", provider="cohere")
            else:
                logger.error(f"Cohere generation failed: {e}")
                raise EvalProviderError(f"Cohere generation failed: {e}", provider="cohere")
    
    def _call_cohere_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Cohere API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_cohere = llm_functions['chat_with_cohere']
        
        # Prepare message format expected by Cohere API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_cohere(
            input_data=messages,
            api_key=self.api_key,
            model=self.model_id,
            system_prompt=kwargs.get('system_prompt'),
            temp=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 100),
            streaming=False
        )
        return result
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from Cohere API."""
        # Note: Cohere API may have limited logprobs support compared to OpenAI
        logger.warning("Log probabilities have limited support in Cohere API")
        return {'logprobs': [], 'tokens': [], 'note': 'Cohere has limited logprobs support'}

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
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"Groq authentication error: {e}")
                raise EvalAuthenticationError(f"Groq authentication failed: {e}", provider="groq")
            elif 'rate limit' in str(e).lower():
                logger.error(f"Groq rate limit error: {e}")
                raise EvalRateLimitError(f"Groq rate limit exceeded: {e}", provider="groq")
            else:
                logger.error(f"Groq generation failed: {e}")
                raise EvalProviderError(f"Groq generation failed: {e}", provider="groq")
    
    def _call_groq_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Groq API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_groq = llm_functions['chat_with_groq']
        
        # Prepare message format expected by Groq API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_groq(
            input_data=messages,
            api_key=self.api_key,
            model=self.model_id,
            system_message=kwargs.get('system_prompt'),
            temp=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 100),
            streaming=False
        )
        return result
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from Groq API."""
        # Groq supports logprobs similar to OpenAI since it uses OpenAI-compatible API
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._get_groq_logprobs_sync,
                text,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Groq logprobs failed: {e}")
            return {'logprobs': [], 'tokens': [], 'error': str(e)}
    
    def _get_groq_logprobs_sync(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get log probabilities from Groq synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_groq = llm_functions['chat_with_groq']
        
        messages = [{"role": "user", "content": text}]
        
        try:
            # Note: Need to check if Groq function supports logprobs parameter
            result = chat_with_groq(
                input_data=messages,
                api_key=self.api_key,
                model=self.model_id,
                temp=0.0,
                max_tokens=1,
                streaming=False
            )
            
            return {
                'logprobs': [],
                'tokens': [],
                'raw_response': result,
                'note': 'Groq logprobs support depends on API capabilities'
            }
        except Exception as e:
            logger.error(f"Failed to get Groq logprobs: {e}")
            return {'logprobs': [], 'tokens': [], 'error': str(e)}

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
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"OpenRouter authentication error: {e}")
                raise EvalAuthenticationError(f"OpenRouter authentication failed: {e}", provider="openrouter")
            elif 'rate limit' in str(e).lower():
                logger.error(f"OpenRouter rate limit error: {e}")
                raise EvalRateLimitError(f"OpenRouter rate limit exceeded: {e}", provider="openrouter")
            else:
                logger.error(f"OpenRouter generation failed: {e}")
                raise EvalProviderError(f"OpenRouter generation failed: {e}", provider="openrouter")
    
    def _call_openrouter_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call OpenRouter API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_openrouter = llm_functions['chat_with_openrouter']
        
        # Prepare message format expected by OpenRouter API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_openrouter(
            input_data=messages,
            api_key=self.api_key,
            model=self.model_id,
            system_message=kwargs.get('system_prompt'),
            temp=kwargs.get('temperature', 0.0),
            streaming=False
        )
        return result
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities (not fully implemented)."""
        logger.warning("Log probabilities not yet implemented for OpenRouter provider")
        return {'logprobs': [], 'tokens': []}

class OllamaProvider(LLMProvider):
    """Ollama provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        ollama_config = api_settings.get('ollama', {})
        
        # Get API URL from config
        self.api_url = config.get('api_url') or ollama_config.get('api_url', 'http://localhost:11434/v1')
        self.model = model_id or ollama_config.get('model')
        if not self.model:
            raise ValueError("Ollama model name not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_ollama_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"Ollama connection error: {e}")
                raise EvalProviderError(f"Ollama connection failed - is the server running?: {e}", provider="ollama")
            elif 'model' in str(e).lower() and 'not found' in str(e).lower():
                logger.error(f"Ollama model error: {e}")
                raise EvalProviderError(f"Ollama model '{self.model}' not found: {e}", provider="ollama")
            else:
                logger.error(f"Ollama generation failed: {e}")
                raise EvalProviderError(f"Ollama generation failed: {e}", provider="ollama")
    
    def _call_ollama_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Ollama API synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_ollama = local_llm_functions['chat_with_ollama']
        
        # Prepare message format expected by Ollama API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_ollama(
            input_data=messages,
            api_url=self.api_url,
            model=self.model,
            system_message=kwargs.get('system_prompt'),
            temperature=kwargs.get('temperature', 0.0),
            num_predict=kwargs.get('max_tokens', 100),
            streaming=False
        )
        
        # Extract text from response
        if isinstance(result, dict):
            if 'choices' in result and result['choices']:
                return result['choices'][0].get('message', {}).get('content', '')
            elif 'content' in result:
                return result['content']
            elif 'text' in result:
                return result['text']
        return str(result)
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from Ollama API."""
        # Note: Ollama may not support logprobs in the same way as OpenAI
        logger.warning("Log probabilities not fully supported by Ollama API")
        return {'logprobs': [], 'tokens': [], 'note': 'Ollama does not provide detailed logprobs'}

class LlamaCppProvider(LLMProvider):
    """llama.cpp provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        llama_config = api_settings.get('llama_cpp', {})
        
        # Get API URL from config
        self.api_url = config.get('api_url') or llama_config.get('api_url', 'http://localhost:8080/completion')
        # llama.cpp typically serves one model at a time, so model_id is optional
        self.model = model_id or "loaded_model"
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using llama.cpp API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_llama_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"llama.cpp connection error: {e}")
                raise EvalProviderError(f"llama.cpp connection failed - is the server running?: {e}", provider="llama_cpp")
            else:
                logger.error(f"llama.cpp generation failed: {e}")
                raise EvalProviderError(f"llama.cpp generation failed: {e}", provider="llama_cpp")
    
    def _call_llama_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call llama.cpp API synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_llama = local_llm_functions['chat_with_llama']
        
        # Prepare message format
        messages = [{"role": "user", "content": prompt}]
        
        # Map max_tokens to n_predict for llama.cpp
        if 'max_tokens' in kwargs and 'n_predict' not in kwargs:
            kwargs['n_predict'] = kwargs['max_tokens']
        
        result = chat_with_llama(
            input_data=messages,
            api_url=self.api_url,
            api_key=None,  # llama.cpp typically doesn't use API keys
            model=self.model,
            system_message=kwargs.get('system_prompt'),
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 100),
            streaming=False,
            **kwargs  # Pass through any additional llama.cpp-specific parameters
        )
        
        # Extract text from response
        if isinstance(result, dict):
            # OpenAI-compatible response
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0].get('message', {}).get('content', '')
            # Direct text response
            elif 'content' in result:
                return result['content']
        
        return str(result)
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from llama.cpp API."""
        # llama.cpp's native API doesn't provide detailed logprobs like OpenAI
        logger.warning("Detailed log probabilities not available from llama.cpp native API")
        return {'logprobs': [], 'tokens': [], 'note': 'llama.cpp does not provide detailed logprobs'}

class VllmProvider(LLMProvider):
    """vLLM provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        vllm_config = api_settings.get('vllm_api', {})
        
        # Get API URL from config
        self.api_url = config.get('api_url') or vllm_config.get('api_url')
        if not self.api_url:
            raise ValueError("vLLM API URL not found in config or settings")
        
        self.api_key = config.get('api_key') or vllm_config.get('api_key')
        self.model = model_id or vllm_config.get('model')
        if not self.model:
            raise ValueError("vLLM model name not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using vLLM API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_vllm_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"vLLM connection error: {e}")
                raise EvalProviderError(f"vLLM connection failed - is the server running?: {e}", provider="vllm")
            elif 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"vLLM authentication error: {e}")
                raise EvalAuthenticationError(f"vLLM authentication failed: {e}", provider="vllm")
            else:
                logger.error(f"vLLM generation failed: {e}")
                raise EvalProviderError(f"vLLM generation failed: {e}", provider="vllm")
    
    def _call_vllm_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call vLLM API synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_vllm = local_llm_functions['chat_with_vllm']
        
        # Prepare message format expected by vLLM API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_vllm(
            input_data=messages,
            api_key=self.api_key,
            model=self.model,
            system_prompt=kwargs.get('system_prompt'),
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 100),
            streaming=False
        )
        
        # Extract text from response
        if isinstance(result, dict):
            if 'choices' in result and result['choices']:
                return result['choices'][0].get('message', {}).get('content', '')
            elif 'content' in result:
                return result['content']
            elif 'text' in result:
                return result['text']
        return str(result)
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from vLLM API."""
        # vLLM supports OpenAI-compatible logprobs
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._get_vllm_logprobs_sync,
                text,
                kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"vLLM logprobs failed: {e}")
            return {'logprobs': [], 'tokens': [], 'error': str(e)}
    
    def _get_vllm_logprobs_sync(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get log probabilities from vLLM synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_vllm = local_llm_functions['chat_with_vllm']
        
        messages = [{"role": "user", "content": text}]
        
        try:
            result = chat_with_vllm(
                input_data=messages,
                api_key=self.api_key,
                model=self.model,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                streaming=False
            )
            
            # vLLM should return OpenAI-compatible response with logprobs
            if isinstance(result, dict) and 'choices' in result:
                return self._extract_openai_logprobs(result)
            else:
                return {
                    'logprobs': [],
                    'tokens': [],
                    'raw_response': result,
                    'note': 'vLLM response format not recognized'
                }
        except Exception as e:
            logger.error(f"Failed to get vLLM logprobs: {e}")
            return {'logprobs': [], 'tokens': [], 'error': str(e)}
    
    def _extract_openai_logprobs(self, response_data: Any) -> Dict[str, Any]:
        """Extract logprobs from OpenAI-compatible API response (same as OpenAIProvider)."""
        try:
            # Handle both string responses (text only) and dict responses (full API response)
            if isinstance(response_data, str):
                # If we just get text back, logprobs weren't available
                return {
                    'logprobs': [],
                    'tokens': [],
                    'content': response_data,
                    'note': 'Response was text-only, no logprobs structure available'
                }
            
            if not isinstance(response_data, dict):
                return {
                    'logprobs': [],
                    'tokens': [],
                    'error': f'Unexpected response type: {type(response_data)}'
                }
            
            # Extract from OpenAI response structure
            choices = response_data.get('choices', [])
            if not choices:
                return {
                    'logprobs': [],
                    'tokens': [],
                    'error': 'No choices in response'
                }
            
            choice = choices[0]  # Get first choice
            logprobs_data = choice.get('logprobs')
            
            if not logprobs_data:
                return {
                    'logprobs': [],
                    'tokens': [],
                    'content': choice.get('message', {}).get('content', ''),
                    'note': 'No logprobs data in response'
                }
            
            # Extract tokens and their probabilities
            content_tokens = logprobs_data.get('content', [])
            tokens = []
            logprobs = []
            top_logprobs = []
            
            for token_data in content_tokens:
                if isinstance(token_data, dict):
                    tokens.append(token_data.get('token', ''))
                    logprobs.append(token_data.get('logprob', 0.0))
                    
                    # Extract top alternatives if available
                    top_alternatives = token_data.get('top_logprobs', [])
                    top_logprobs.append([
                        {
                            'token': alt.get('token', ''),
                            'logprob': alt.get('logprob', 0.0)
                        }
                        for alt in top_alternatives if isinstance(alt, dict)
                    ])
            
            return {
                'tokens': tokens,
                'logprobs': logprobs,
                'top_logprobs': top_logprobs,
                'content': choice.get('message', {}).get('content', ''),
                'model': response_data.get('model', ''),
                'usage': response_data.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"Error extracting vLLM logprobs: {e}")
            return {
                'logprobs': [],
                'tokens': [],
                'error': f'Extraction failed: {str(e)}',
                'raw_response': response_data
            }

def get_llm_provider(provider_name: str, model_id: str, config: Dict[str, Any]) -> LLMProvider:
    """Get an LLM provider instance based on provider name."""
    provider_classes = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'cohere': CohereProvider,
        'groq': GroqProvider,
        'openrouter': OpenRouterProvider,
        'ollama': OllamaProvider,
        'llama_cpp': LlamaCppProvider,
        'vllm': VllmProvider,
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
    
    async def generate_with_system(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        return await self.provider.generate_with_system_async(prompt, system_prompt, **kwargs)
    
    async def get_logprobs(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities."""
        return await self.provider.get_logprobs_async(text, **kwargs)
    
    async def get_completion_logprobs(self, prompt: str, completion: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities for a specific completion given a prompt."""
        return await self.provider.get_completion_logprobs_async(prompt, completion, **kwargs)