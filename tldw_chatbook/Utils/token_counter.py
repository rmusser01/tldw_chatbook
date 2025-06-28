# token_counter.py
# Description: Token counting utilities for various LLM models
#
# Imports
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
#
# 3rd-Party Imports
from loguru import logger
#
# Local Imports - Import with error handling to avoid circular imports
try:
    from .custom_tokenizers import count_tokens_with_custom, count_messages_with_custom
    CUSTOM_TOKENIZERS_AVAILABLE = True
except ImportError:
    CUSTOM_TOKENIZERS_AVAILABLE = False
    count_tokens_with_custom = None
    count_messages_with_custom = None
#
########################################################################################################################
#
# Functions:

# Try to import tiktoken for OpenAI models
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available. Token counting will use character-based estimation.")

# Model to encoding mapping for tiktoken
TIKTOKEN_MODEL_ENCODINGS = {
    # GPT-4 models
    "gpt-4": "cl100k_base",
    "gpt-4-0314": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0314": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    
    # GPT-3.5 models
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    
    # Text models
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    
    # Code models
    "code-davinci-002": "p50k_base",
    "code-cushman-001": "p50k_base",
}

# Approximate tokens per character for different model families
TOKENS_PER_CHAR_ESTIMATES = {
    "openai": 0.25,      # ~4 chars per token
    "anthropic": 0.25,   # Similar to OpenAI
    "google": 0.3,       # Slightly more aggressive tokenization
    "cohere": 0.25,      # Similar to OpenAI
    "deepseek": 0.25,    # Similar to OpenAI
    "mistral": 0.25,     # Similar to OpenAI
    "groq": 0.25,        # Similar to OpenAI
    "huggingface": 0.3,  # Varies by model
    "openrouter": 0.25,  # Depends on underlying model
    "default": 0.25      # Default fallback
}

# Token limits per model (approximate)
MODEL_TOKEN_LIMITS = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    
    # Anthropic
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-instant-1.2": 100000,
    
    # Google
    "gemini-pro": 30720,
    "gemini-pro-vision": 12288,
    
    # Others
    "mistral-large": 32000,
    "mistral-medium": 32000,
    "mistral-small": 32000,
    "mixtral-8x7b": 32000,
    
    # Default for unknown models
    "default": 4096
}


def get_tiktoken_encoding(model: str) -> Optional[Any]:
    """Get the tiktoken encoding for a specific model."""
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        # Try to get specific encoding for model
        if model in TIKTOKEN_MODEL_ENCODINGS:
            encoding_name = TIKTOKEN_MODEL_ENCODINGS[model]
            return tiktoken.get_encoding(encoding_name)
        
        # Try to get encoding by model name
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Default to cl100k_base for unknown models
            return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.error(f"Error getting tiktoken encoding: {e}")
        return None


def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens using tiktoken for OpenAI models."""
    encoding = get_tiktoken_encoding(model)
    if encoding:
        try:
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens with tiktoken: {e}")
    
    # Fallback to character estimation
    return int(len(text) * TOKENS_PER_CHAR_ESTIMATES.get("openai", 0.25))


def count_tokens_messages(messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens for a list of messages in OpenAI format.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: The model name for accurate token counting
        
    Returns:
        Total token count including message formatting overhead
    """
    if not messages:
        return 0
    
    # Different models have different message formatting overhead
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        # Each message has overhead: <|start|>role<|end|>content<|end|>
        tokens_per_message = 3
        tokens_per_name = 1  # If name is present
        base_tokens = 3  # Every reply is primed with <|start|>assistant<|message|>
    else:
        # Conservative estimate for other models
        tokens_per_message = 2
        tokens_per_name = 1
        base_tokens = 2
    
    total_tokens = base_tokens
    
    for message in messages:
        total_tokens += tokens_per_message
        
        # Count tokens in role
        role = message.get("role", "")
        if role:
            total_tokens += count_tokens_tiktoken(role, model) if TIKTOKEN_AVAILABLE else len(role.split())
        
        # Count tokens in content
        content = message.get("content", "")
        if content:
            total_tokens += count_tokens_tiktoken(content, model) if TIKTOKEN_AVAILABLE else len(content.split())
        
        # Count tokens in name if present
        name = message.get("name", "")
        if name:
            total_tokens += tokens_per_name
            total_tokens += count_tokens_tiktoken(name, model) if TIKTOKEN_AVAILABLE else len(name.split())
    
    return total_tokens


def count_tokens_chat_history(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]], 
    model: str = "gpt-3.5-turbo",
    provider: str = "openai"
) -> int:
    """
    Count tokens in chat history format (list of tuples or message dicts).
    
    Args:
        history: Chat history in various formats
        model: The model name for accurate counting
        provider: The LLM provider name
        
    Returns:
        Total estimated token count
    """
    if not history:
        return 0
    
    # Convert history to message format
    messages = []
    
    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            # (user_msg, bot_msg) format
            user_msg, bot_msg = item
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        elif isinstance(item, dict) and "role" in item and "content" in item:
            # Already in message format
            messages.append(item)
        else:
            logger.warning(f"Unknown history format: {type(item)}")
    
    # Try custom tokenizers first
    if CUSTOM_TOKENIZERS_AVAILABLE and count_messages_with_custom:
        custom_count = count_messages_with_custom(messages, model, provider)
        if custom_count is not None:
            logger.debug(f"Using custom tokenizer for {model} ({provider}): {custom_count} tokens")
            return custom_count
    
    # Use provider-specific counting if available
    if provider == "openai" and TIKTOKEN_AVAILABLE:
        return count_tokens_messages(messages, model)
    else:
        # Fallback to character-based estimation
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            total_chars += len(content)
        
        # Add some overhead for message formatting
        total_chars += len(messages) * 10  # Rough estimate for role and formatting
        
        # Use provider-specific ratio
        ratio = TOKENS_PER_CHAR_ESTIMATES.get(provider, TOKENS_PER_CHAR_ESTIMATES["default"])
        return int(total_chars * ratio)


def get_model_token_limit(model: str, provider: str = "openai") -> int:
    """
    Get the token limit for a specific model.
    
    Args:
        model: The model name
        provider: The LLM provider
        
    Returns:
        Maximum token limit for the model
    """
    # Check specific model limits
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]
    
    # Check by model prefix
    for model_prefix, limit in MODEL_TOKEN_LIMITS.items():
        if model.startswith(model_prefix):
            return limit
    
    # Provider-specific defaults
    provider_defaults = {
        "anthropic": 100000,  # Conservative for Claude
        "google": 30720,      # Gemini default
        "openai": 4096,       # GPT-3.5 default
        "mistral": 32000,     # Mistral default
    }
    
    return provider_defaults.get(provider, MODEL_TOKEN_LIMITS["default"])


def estimate_remaining_tokens(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
    max_tokens_response: int = 2048,
    system_prompt: Optional[str] = None
) -> Tuple[int, int, int]:
    """
    Estimate remaining tokens available for response.
    
    Args:
        history: Chat history
        model: The model name
        provider: The LLM provider
        max_tokens_response: Max tokens reserved for response
        system_prompt: Optional system prompt to include in count
        
    Returns:
        Tuple of (used_tokens, total_limit, remaining_for_input)
    """
    # Count current tokens
    current_tokens = count_tokens_chat_history(history, model, provider)
    
    # Add system prompt if present
    if system_prompt:
        if provider == "openai" and TIKTOKEN_AVAILABLE:
            current_tokens += count_tokens_tiktoken(system_prompt, model)
        else:
            current_tokens += int(len(system_prompt) * TOKENS_PER_CHAR_ESTIMATES.get(provider, 0.25))
    
    # Get model limit
    total_limit = get_model_token_limit(model, provider)
    
    # Calculate remaining
    remaining = total_limit - current_tokens - max_tokens_response
    
    return current_tokens, total_limit, max(0, remaining)


def format_token_display(used: int, limit: int) -> str:
    """
    Format token count for display.
    
    Args:
        used: Number of tokens used
        limit: Total token limit
        
    Returns:
        Formatted string for display
    """
    percentage = (used / limit * 100) if limit > 0 else 0
    
    # Add warning indicators
    if percentage >= 95:
        indicator = "🔴"  # Red - very close to limit
    elif percentage >= 80:
        indicator = "🟡"  # Yellow - approaching limit
    else:
        indicator = "🟢"  # Green - plenty of space
    
    return f"{indicator} Tokens: {used:,} / {limit:,} ({percentage:.0f}%)"


#
# End of token_counter.py
########################################################################################################################