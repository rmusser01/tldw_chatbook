# token_counter.py
# Description: Token counting utilities for various LLM models
#
# Imports
from typing import List, Dict, Any, Union, Optional, Tuple

#
# 3rd-Party Imports
from loguru import logger

#
# Local Imports - Import with error handling to avoid circular imports
try:
    from .custom_tokenizers import (
        count_tokens_with_custom,
        count_messages_with_custom,
        custom_tokenizers_available,
    )

    CUSTOM_TOKENIZERS_AVAILABLE = True
except ImportError:
    CUSTOM_TOKENIZERS_AVAILABLE = False
    count_tokens_with_custom = None
    count_messages_with_custom = None
    custom_tokenizers_available = None
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
    logger.warning(
        "tiktoken not available. Token counting will use character-based estimation."
    )

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
    "openai": 0.25,  # ~4 chars per token
    "anthropic": 0.25,  # Similar to OpenAI
    "google": 0.3,  # Slightly more aggressive tokenization
    "cohere": 0.25,  # Similar to OpenAI
    "deepseek": 0.25,  # Similar to OpenAI
    "mistral": 0.25,  # Similar to OpenAI
    "groq": 0.25,  # Similar to OpenAI
    "huggingface": 0.3,  # Varies by model
    "openrouter": 0.25,  # Depends on underlying model
    "default": 0.25,  # Default fallback
}

# Conservative chars-based estimate constants (used when no tokenizer is available).
CJK_TOKENS_PER_CHAR = 1.0   # each CJK code point is >= ~1 token
ESTIMATE_HEADROOM = 1.2     # documented headroom so estimates lean high (safe)

_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ext-A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Fullwidth / halfwidth (CJK punctuation)
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _norm_provider(provider: str) -> str:
    """Normalize a provider name for case-insensitive dict lookups.

    Args:
        provider: Provider name in any casing (e.g. ``"OpenAI"``, ``"google"``).

    Returns:
        The lower-cased, stripped provider name (``""`` for ``None``/blank).
    """
    return str(provider or "").strip().lower()


def _chars_estimate(text: str, provider: str) -> int:
    """Conservative chars-based token floor; weights CJK higher, applies headroom.

    Non-empty text always estimates to at least 1 token — ``int()`` truncation
    would otherwise round very short strings (e.g. "hi") down to 0, which would
    under-count and defeat the conservative-floor guarantee.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    base_ratio = TOKENS_PER_CHAR_ESTIMATES.get(
        _norm_provider(provider) or "default", TOKENS_PER_CHAR_ESTIMATES["default"]
    )
    return max(1, int((other * base_ratio + cjk * CJK_TOKENS_PER_CHAR) * ESTIMATE_HEADROOM))


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo", provider: str = "") -> int:
    """Estimate the token count of a text string with one consistent strategy.

    Tiers: a custom tokenizer (only when one is actually installed), else
    tiktoken (when available), else a conservative chars-based floor. Never uses
    a whitespace word count.

    Args:
        text: The text to estimate.
        model: Model name (selects the tiktoken encoding / custom tokenizer).
        provider: Provider name (case-insensitive); selects the chars-path ratio
            and the custom tokenizer's provider patterns.

    Returns:
        Estimated token count (0 for empty text).
    """
    if not text:
        return 0
    if CUSTOM_TOKENIZERS_AVAILABLE and custom_tokenizers_available():
        custom = count_tokens_with_custom(text, model, _norm_provider(provider))
        if custom is not None:
            return custom
    if TIKTOKEN_AVAILABLE:
        return count_tokens_tiktoken(text, model)
    return _chars_estimate(text, provider)


# Token limits per model (approximate)
MODEL_TOKEN_LIMITS = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 1047576,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16384,
    "o1": 200000,
    "o1-mini": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    "o4-mini": 200000,
    # Anthropic
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-instant-1.2": 100000,
    # Google
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-pro": 30720,
    "gemini-pro-vision": 12288,
    # Others
    "mistral-large": 128000,
    "mistral-medium": 32000,
    "mistral-small": 32000,
    "mixtral-8x7b": 32000,
    # Default for unknown models
    "default": 4096,
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


def count_tokens_messages(
    messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo", provider: str = ""
) -> int:
    """Count tokens for OpenAI-format messages (framing overhead + estimate_tokens)."""
    if not messages:
        return 0

    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
        base_tokens = 3
    else:
        tokens_per_message = 2
        tokens_per_name = 1
        base_tokens = 2

    total_tokens = base_tokens
    for message in messages:
        total_tokens += tokens_per_message
        role = message.get("role", "")
        if role:
            total_tokens += estimate_tokens(role, model, provider)
        content = message.get("content", "")
        if content:
            total_tokens += estimate_tokens(content, model, provider)
        name = message.get("name", "")
        if name:
            total_tokens += tokens_per_name
            total_tokens += estimate_tokens(name, model, provider)
    return total_tokens


def count_tokens_chat_history(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
) -> int:
    """Count tokens in chat-history format (tuples or message dicts) via the one estimator."""
    if not history:
        return 0

    messages: List[Dict[str, Any]] = []
    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        elif isinstance(item, dict) and "role" in item and "content" in item:
            messages.append(item)
        else:
            logger.warning(f"Unknown history format: {type(item)}")

    return count_tokens_messages(messages, model, provider)


def get_model_token_limit(model: str, provider: str = "openai") -> int:
    """
    Get the input context-window token limit for a specific model.

    Resolves in priority order: the per-model capability `context_window`
    (config-overridable), an exact table entry, the longest matching table
    prefix, then a conservative provider default. Fallbacks lean conservative
    on purpose: under-estimating the window degrades gracefully (more trimming),
    while over-estimating is the only way to overflow the model on dispatch.
    """
    provider_key = _norm_provider(provider)

    # OpenRouter model IDs are "upstream_provider/model" (e.g. "openai/gpt-4o-mini");
    # resolve against the upstream provider/model so they don't fall through to the
    # generic default. Split once and re-dispatch -- the re-dispatch provider is the
    # upstream (never "openrouter"), so this cannot recurse indefinitely.
    if provider_key == "openrouter" and "/" in model:
        upstream_provider, upstream_model = model.split("/", 1)
        return get_model_token_limit(upstream_model, upstream_provider)

    # 1. Per-model capability context window (authoritative, config-overridable).
    try:
        from tldw_chatbook.model_capabilities import get_context_window

        window = get_context_window(provider, model)
        if window is not None:
            return window
    except Exception as e:  # never let capability resolution break token limits
        logger.debug(f"context_window lookup failed for {provider}/{model}: {e}")

    # 2. Exact table match.
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]

    # 3. Longest matching table prefix (so "gpt-4" can't shadow "gpt-4-turbo").
    best_limit = None
    best_len = -1
    for model_prefix, limit in MODEL_TOKEN_LIMITS.items():
        if model_prefix == "default":
            continue
        if model.startswith(model_prefix) and len(model_prefix) > best_len:
            best_limit = limit
            best_len = len(model_prefix)
    if best_limit is not None:
        return best_limit

    # 4. Conservative provider default.
    provider_defaults = {
        "anthropic": 200000,  # every modern Claude is >= 200k; safe floor
        "google": 30720,
        "openai": 4096,
        "mistral": 32000,
    }
    return provider_defaults.get(provider_key, MODEL_TOKEN_LIMITS["default"])


def estimate_remaining_tokens(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
    max_tokens_response: int = 2048,
    system_prompt: Optional[str] = None,
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
        current_tokens += estimate_tokens(system_prompt, model, provider)

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
