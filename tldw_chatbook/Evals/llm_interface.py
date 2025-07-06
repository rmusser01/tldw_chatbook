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
        chat_with_deepseek, chat_with_google, chat_with_mistral
    )
    return {
        'chat_with_openai': chat_with_openai,
        'chat_with_anthropic': chat_with_anthropic,
        'chat_with_cohere': chat_with_cohere,
        'chat_with_groq': chat_with_groq,
        'chat_with_openrouter': chat_with_openrouter,
        'chat_with_huggingface': chat_with_huggingface,
        'chat_with_deepseek': chat_with_deepseek,
        'chat_with_google': chat_with_google,
        'chat_with_mistral': chat_with_mistral
    }

def _get_local_llm_functions():
    """Import local LLM functions at runtime to avoid circular imports."""
    from ...LLM_Calls.LLM_API_Calls_Local import (
        chat_with_ollama, chat_with_vllm, chat_with_llama,
        chat_with_kobold, chat_with_tabbyapi, chat_with_aphrodite,
        chat_with_custom_openai, chat_with_mlx_lm
    )
    return {
        'chat_with_ollama': chat_with_ollama,
        'chat_with_vllm': chat_with_vllm,
        'chat_with_llama': chat_with_llama,
        'chat_with_kobold': chat_with_kobold,
        'chat_with_tabbyapi': chat_with_tabbyapi,
        'chat_with_aphrodite': chat_with_aphrodite,
        'chat_with_custom_openai': chat_with_custom_openai,
        'chat_with_mlx_lm': chat_with_mlx_lm
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
        
        # Extract text from response
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Handle OpenAI response format
            if 'choices' in result and result['choices']:
                return result['choices'][0].get('message', {}).get('content', '')
            elif 'content' in result:
                return result['content']
        
        return str(result)
    
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
        
        # Extract text from response
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Handle Anthropic response format
            if 'content' in result:
                if isinstance(result['content'], list) and result['content']:
                    # Anthropic returns content as list of content blocks
                    return result['content'][0].get('text', '')
                else:
                    return str(result['content'])
        
        return str(result)
    
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

class HuggingFaceProvider(LLMProvider):
    """HuggingFace provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('huggingface_api_key')
        if not self.api_key:
            raise ValueError("HuggingFace API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using HuggingFace API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_huggingface_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"HuggingFace authentication error: {e}")
                raise EvalAuthenticationError(f"HuggingFace authentication failed: {e}", provider="huggingface")
            elif 'rate limit' in str(e).lower():
                logger.error(f"HuggingFace rate limit error: {e}")
                raise EvalRateLimitError(f"HuggingFace rate limit exceeded: {e}", provider="huggingface")
            else:
                logger.error(f"HuggingFace generation failed: {e}")
                raise EvalProviderError(f"HuggingFace generation failed: {e}", provider="huggingface")
    
    def _call_huggingface_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call HuggingFace API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_huggingface = llm_functions['chat_with_huggingface']
        
        # Prepare message format expected by HuggingFace API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_huggingface(
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
        """Get log probabilities from HuggingFace API."""
        logger.warning("Log probabilities not directly supported by HuggingFace Inference API")
        return {'logprobs': [], 'tokens': [], 'note': 'HuggingFace Inference API does not support logprobs'}

class DeepSeekProvider(LLMProvider):
    """DeepSeek provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('deepseek_api_key')
        if not self.api_key:
            raise ValueError("DeepSeek API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using DeepSeek API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_deepseek_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"DeepSeek authentication error: {e}")
                raise EvalAuthenticationError(f"DeepSeek authentication failed: {e}", provider="deepseek")
            elif 'rate limit' in str(e).lower():
                logger.error(f"DeepSeek rate limit error: {e}")
                raise EvalRateLimitError(f"DeepSeek rate limit exceeded: {e}", provider="deepseek")
            else:
                logger.error(f"DeepSeek generation failed: {e}")
                raise EvalProviderError(f"DeepSeek generation failed: {e}", provider="deepseek")
    
    def _call_deepseek_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call DeepSeek API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_deepseek = llm_functions['chat_with_deepseek']
        
        # Prepare message format expected by DeepSeek API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_deepseek(
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
        """Get log probabilities from DeepSeek API."""
        # DeepSeek uses OpenAI-compatible API, may support logprobs
        logger.info("DeepSeek may support logprobs through OpenAI-compatible API")
        return {'logprobs': [], 'tokens': [], 'note': 'DeepSeek logprobs support depends on API version'}

class GoogleProvider(LLMProvider):
    """Google (Gemini) provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('google_api_key')
        if not self.api_key:
            raise ValueError("Google API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Google API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_google_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"Google authentication error: {e}")
                raise EvalAuthenticationError(f"Google authentication failed: {e}", provider="google")
            elif 'rate limit' in str(e).lower() or 'quota' in str(e).lower():
                logger.error(f"Google rate limit error: {e}")
                raise EvalRateLimitError(f"Google rate limit exceeded: {e}", provider="google")
            else:
                logger.error(f"Google generation failed: {e}")
                raise EvalProviderError(f"Google generation failed: {e}", provider="google")
    
    def _call_google_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Google API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_google = llm_functions['chat_with_google']
        
        # Prepare message format expected by Google API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_google(
            input_data=messages,
            api_key=self.api_key,
            model=self.model_id,
            system_message=kwargs.get('system_prompt'),
            temperature=kwargs.get('temperature', 0.0),
            streaming=False
        )
        return result
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from Google API."""
        logger.warning("Log probabilities not supported by Google Gemini API")
        return {'logprobs': [], 'tokens': [], 'note': 'Google Gemini does not support logprobs'}

class MistralProvider(LLMProvider):
    """Mistral provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        self.api_key = config.get('api_key') or settings.get('mistral_api_key')
        if not self.api_key:
            raise ValueError("Mistral API key not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Mistral API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_mistral_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'authentication' in str(e).lower() or 'api key' in str(e).lower():
                logger.error(f"Mistral authentication error: {e}")
                raise EvalAuthenticationError(f"Mistral authentication failed: {e}", provider="mistral")
            elif 'rate limit' in str(e).lower():
                logger.error(f"Mistral rate limit error: {e}")
                raise EvalRateLimitError(f"Mistral rate limit exceeded: {e}", provider="mistral")
            else:
                logger.error(f"Mistral generation failed: {e}")
                raise EvalProviderError(f"Mistral generation failed: {e}", provider="mistral")
    
    def _call_mistral_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Mistral API synchronously."""
        # Get functions at runtime to avoid circular imports
        llm_functions = _get_llm_functions()
        chat_with_mistral = llm_functions['chat_with_mistral']
        
        # Prepare message format expected by Mistral API
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_mistral(
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
        """Get log probabilities from Mistral API."""
        logger.warning("Log probabilities support in Mistral API is limited")
        return {'logprobs': [], 'tokens': [], 'note': 'Mistral has limited logprobs support'}

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

class KoboldProvider(LLMProvider):
    """Kobold.cpp provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        kobold_config = api_settings.get('kobold_cpp', {})
        
        # Get API URL from config
        self.api_url = config.get('api_url') or kobold_config.get('api_url', 'http://localhost:5001')
        # Kobold typically serves one model at a time
        self.model = model_id or "loaded_model"
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Kobold API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_kobold_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"Kobold connection error: {e}")
                raise EvalProviderError(f"Kobold connection failed - is the server running?: {e}", provider="kobold")
            else:
                logger.error(f"Kobold generation failed: {e}")
                raise EvalProviderError(f"Kobold generation failed: {e}", provider="kobold")
    
    def _call_kobold_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Kobold API synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_kobold = local_llm_functions['chat_with_kobold']
        
        # Prepare message format
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_kobold(
            input_data=messages,
            api_url=self.api_url,
            model=self.model,
            system_message=kwargs.get('system_prompt'),
            temp=kwargs.get('temperature', 0.0),
            max_length=kwargs.get('max_tokens', 100),
            streaming=False
        )
        
        # Extract text from response
        if isinstance(result, dict):
            if 'choices' in result and result['choices']:
                return result['choices'][0].get('message', {}).get('content', '')
            elif 'results' in result and result['results']:
                return result['results'][0].get('text', '')
            elif 'text' in result:
                return result['text']
        return str(result)
    
    async def generate_with_system_async(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text with optional system prompt."""
        if system_prompt:
            kwargs['system_prompt'] = system_prompt
        return await self.generate_async(prompt, **kwargs)
    
    async def get_logprobs_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get log probabilities from Kobold API."""
        logger.warning("Log probabilities not supported by Kobold API")
        return {'logprobs': [], 'tokens': [], 'note': 'Kobold does not support logprobs'}

class TabbyAPIProvider(LLMProvider):
    """TabbyAPI provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        tabby_config = api_settings.get('tabbyapi', {})
        
        # Get API URL and key from config
        self.api_url = config.get('api_url') or tabby_config.get('api_url')
        if not self.api_url:
            raise ValueError("TabbyAPI URL not found in config or settings")
        
        self.api_key = config.get('api_key') or tabby_config.get('api_key')
        self.model = model_id or tabby_config.get('model', 'default')
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using TabbyAPI."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_tabby_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"TabbyAPI connection error: {e}")
                raise EvalProviderError(f"TabbyAPI connection failed - is the server running?: {e}", provider="tabbyapi")
            elif 'authentication' in str(e).lower():
                logger.error(f"TabbyAPI authentication error: {e}")
                raise EvalAuthenticationError(f"TabbyAPI authentication failed: {e}", provider="tabbyapi")
            else:
                logger.error(f"TabbyAPI generation failed: {e}")
                raise EvalProviderError(f"TabbyAPI generation failed: {e}", provider="tabbyapi")
    
    def _call_tabby_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call TabbyAPI synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_tabbyapi = local_llm_functions['chat_with_tabbyapi']
        
        # Prepare message format
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_tabbyapi(
            input_data=messages,
            api_url=self.api_url,
            api_key=self.api_key,
            model=self.model,
            system_message=kwargs.get('system_prompt'),
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
        """Get log probabilities from TabbyAPI."""
        # TabbyAPI uses OpenAI-compatible format, may support logprobs
        logger.info("TabbyAPI may support logprobs through OpenAI-compatible API")
        return {'logprobs': [], 'tokens': [], 'note': 'TabbyAPI logprobs support depends on configuration'}

class AphroditeProvider(LLMProvider):
    """Aphrodite (vLLM fork) provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        aphrodite_config = api_settings.get('aphrodite', {})
        
        # Get API URL from config
        self.api_url = config.get('api_url') or aphrodite_config.get('api_url')
        if not self.api_url:
            raise ValueError("Aphrodite API URL not found in config or settings")
        
        self.api_key = config.get('api_key') or aphrodite_config.get('api_key')
        self.model = model_id or aphrodite_config.get('model')
        if not self.model:
            raise ValueError("Aphrodite model name not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Aphrodite API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_aphrodite_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"Aphrodite connection error: {e}")
                raise EvalProviderError(f"Aphrodite connection failed - is the server running?: {e}", provider="aphrodite")
            elif 'authentication' in str(e).lower():
                logger.error(f"Aphrodite authentication error: {e}")
                raise EvalAuthenticationError(f"Aphrodite authentication failed: {e}", provider="aphrodite")
            else:
                logger.error(f"Aphrodite generation failed: {e}")
                raise EvalProviderError(f"Aphrodite generation failed: {e}", provider="aphrodite")
    
    def _call_aphrodite_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Aphrodite API synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_aphrodite = local_llm_functions['chat_with_aphrodite']
        
        # Prepare message format
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_aphrodite(
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
        """Get log probabilities from Aphrodite API."""
        # Aphrodite is vLLM-based and should support OpenAI-compatible logprobs
        logger.info("Aphrodite supports logprobs through OpenAI-compatible API")
        return {'logprobs': [], 'tokens': [], 'note': 'Aphrodite logprobs implementation pending'}

class CustomOpenAIProvider(LLMProvider):
    """Custom OpenAI-compatible provider for evaluations."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        custom_config = api_settings.get('custom_openai', {})
        
        # Get API URL and key from config
        self.api_url = config.get('api_url') or custom_config.get('api_url')
        if not self.api_url:
            raise ValueError("Custom OpenAI API URL not found in config or settings")
        
        self.api_key = config.get('api_key') or custom_config.get('api_key')
        self.model = model_id or custom_config.get('model', 'default')
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Custom OpenAI-compatible API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_custom_openai_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"Custom OpenAI connection error: {e}")
                raise EvalProviderError(f"Custom OpenAI connection failed - is the server running?: {e}", provider="custom_openai")
            elif 'authentication' in str(e).lower():
                logger.error(f"Custom OpenAI authentication error: {e}")
                raise EvalAuthenticationError(f"Custom OpenAI authentication failed: {e}", provider="custom_openai")
            else:
                logger.error(f"Custom OpenAI generation failed: {e}")
                raise EvalProviderError(f"Custom OpenAI generation failed: {e}", provider="custom_openai")
    
    def _call_custom_openai_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call Custom OpenAI API synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_custom_openai = local_llm_functions['chat_with_custom_openai']
        
        # Prepare message format
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_custom_openai(
            input_data=messages,
            api_key=self.api_key,
            api_url=self.api_url,
            model=self.model,
            system_message=kwargs.get('system_prompt'),
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
        """Get log probabilities from Custom OpenAI API."""
        # Should support logprobs if truly OpenAI-compatible
        logger.info("Custom OpenAI may support logprobs if API is fully compatible")
        return {'logprobs': [], 'tokens': [], 'note': 'Custom OpenAI logprobs support depends on implementation'}

class MLXProvider(LLMProvider):
    """MLX-LM provider for evaluations (Apple Silicon optimized)."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        settings = load_settings()
        api_settings = settings.get('api_settings', {})
        mlx_config = api_settings.get('mlx', {})
        
        # Get API URL from config
        self.api_url = config.get('api_url') or mlx_config.get('api_url', 'http://localhost:8080')
        self.model = model_id or mlx_config.get('model')
        if not self.model:
            raise ValueError("MLX model name not found in config or settings")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using MLX-LM API."""
        try:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._call_mlx_sync,
                prompt,
                kwargs
            )
            
            return result
            
        except Exception as e:
            # Handle known API errors
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                logger.error(f"MLX connection error: {e}")
                raise EvalProviderError(f"MLX connection failed - is the server running?: {e}", provider="mlx")
            else:
                logger.error(f"MLX generation failed: {e}")
                raise EvalProviderError(f"MLX generation failed: {e}", provider="mlx")
    
    def _call_mlx_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Call MLX-LM API synchronously."""
        # Get functions at runtime to avoid circular imports
        local_llm_functions = _get_local_llm_functions()
        chat_with_mlx_lm = local_llm_functions['chat_with_mlx_lm']
        
        # Prepare message format
        messages = [{"role": "user", "content": prompt}]
        
        result = chat_with_mlx_lm(
            input_data=messages,
            api_url=self.api_url,
            model=self.model,
            system_message=kwargs.get('system_prompt'),
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
        """Get log probabilities from MLX-LM API."""
        logger.warning("Log probabilities not supported by MLX-LM API")
        return {'logprobs': [], 'tokens': [], 'note': 'MLX-LM does not support logprobs'}

def get_llm_provider(provider_name: str, model_id: str, config: Dict[str, Any]) -> LLMProvider:
    """Get an LLM provider instance based on provider name."""
    provider_classes = {
        # Commercial providers
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'cohere': CohereProvider,
        'groq': GroqProvider,
        'openrouter': OpenRouterProvider,
        'huggingface': HuggingFaceProvider,
        'deepseek': DeepSeekProvider,
        'google': GoogleProvider,
        'mistral': MistralProvider,
        # Local providers
        'ollama': OllamaProvider,
        'llama_cpp': LlamaCppProvider,
        'llamacpp': LlamaCppProvider,  # Alias
        'vllm': VllmProvider,
        'kobold': KoboldProvider,
        'kobold_cpp': KoboldProvider,  # Alias
        'tabbyapi': TabbyAPIProvider,
        'aphrodite': AphroditeProvider,
        'custom_openai': CustomOpenAIProvider,
        'mlx': MLXProvider,
        'mlx_lm': MLXProvider,  # Alias
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