# token_counter.py
# Description: Token counting utilities for various LLM models
#
# Imports
import re
import threading
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
CJK_TOKENS_PER_CHAR = 1.0  # each CJK code point is >= ~1 token
ESTIMATE_HEADROOM = 1.2  # documented headroom so estimates lean high (safe)

#: Fixed contribution for a non-text part of a multimodal message (an image /
#: attachment block in the OpenAI part-list shape). 1024 matches the repo's
#: existing per-image budget charge (``console_history_budget``'s
#: ``DEFAULT_PER_IMAGE_TOKENS`` aliases THIS constant so the two can never
#: drift) — using a lower figure here would let image-heavy no-usage turns
#: slip past ``max_total_tokens`` enforcement (Qodo, PR #1783). Deliberately
#: conservative, consistent with the estimator's floor-not-exact contract
#: (TASK-17610).
NON_TEXT_PART_TOKEN_ESTIMATE = 1024

_CJK_RANGES = (
    (0x3000, 0x303F),  # CJK Symbols and Punctuation (。、「」etc.)
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


#: Character class matching exactly the code points `_is_cjk` accepts, built
#: from the same `_CJK_RANGES` tuple so the two can never disagree.
#: TASK-18602: the CJK share used to be counted as
#: `sum(1 for ch in text if _is_cjk(ch))` -- one Python call per character,
#: with a 7-range generator inside it. Measured at 158.8 ms for a 640 KB
#: payload, re-run over the whole conversation every turn.
_CJK_RE = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _CJK_RANGES) + "]"
)


def _count_cjk(text: str) -> int:
    """Count CJK-weighted characters in ``text``.

    Two tiers, both avoiding the per-character Python loop this replaced:

    * All-ASCII text -- the overwhelmingly common case for prompts, code,
      and English prose -- cannot contain a CJK code point at all, and
      ``str.isascii()`` settles it in one C-level scan (0.001 ms on 640 KB,
      against 158.8 ms for the old loop).
    * Anything else is counted by ``_CJK_RE.subn``, which does the same
      count in a single C-level pass.

    Args:
        text: The text to scan.

    Returns:
        Number of characters inside `_CJK_RANGES`.
    """
    if text.isascii():
        return 0
    return _CJK_RE.subn("", text)[1]


#: Bounded memo for `estimate_tokens`. TASK-18602: a conversation is
#: append-only, but every caller re-estimates the WHOLE message list each
#: turn, so an N-turn run pays O(N^2) to learn N answers -- 33.1 s of pure
#: CPU across a simulated 400-turn run, against 0.16 s for counting only
#: each turn's new text.
#:
#: Keyed by `(model, provider, len(text), hash(text))` rather than by the
#: text itself, deliberately: the key holds NO strong reference, so memoing
#: a 600 KB message cannot pin it in memory after the conversation moves
#: on. CPython caches a str's hash on the object after first use, so repeat
#: lookups of the same message cost a dict probe, not a rescan.
#:
#: A hash collision would serve one text's estimate for another's. Guarded
#: by including the length and the tokenizer identity in the key, and
#: bounded in consequence by what this function is: a token ESTIMATE whose
#: own chars tier already applies approximation headroom. It is never used
#: where an exact count is required.
_ESTIMATE_CACHE: "dict[tuple[str, str, int, int], int]" = {}
_ESTIMATE_CACHE_LOCK = threading.Lock()
#: Cleared wholesale on overflow rather than evicted LRU-style: a miss costs
#: exactly one recompute, so the simplest correct policy is the right one,
#: and it keeps the locked section O(1).
ESTIMATE_CACHE_MAX_ENTRIES = 4096


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
    cjk = _count_cjk(text)
    other = len(text) - cjk
    base_ratio = TOKENS_PER_CHAR_ESTIMATES.get(
        _norm_provider(provider) or "default", TOKENS_PER_CHAR_ESTIMATES["default"]
    )
    return max(
        1, int((other * base_ratio + cjk * CJK_TOKENS_PER_CHAR) * ESTIMATE_HEADROOM)
    )


def _flatten_message_content(content: Any) -> "tuple[str, int]":
    """Normalize message ``content`` into countable text + non-text part count.

    Message content is usually a plain string, but multimodal/attachment turns
    carry the OpenAI part-list shape (``[{"type": "text", "text": ...},
    {"type": "image_url", ...}]``). Iterating that list as if it were a string
    crashed the char estimator (``ord()`` on dict items — TASK-17610).

    Args:
        content: A message's ``content`` value in any shape.

    Returns:
        ``(text, non_text_parts)`` — the concatenated text of every string
        part (dict parts contribute their ``"text"`` value when it is a
        string), and the count of parts that carry no countable text (each
        later contributes :data:`NON_TEXT_PART_TOKEN_ESTIMATE`).
    """
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        texts: list[str] = []
        non_text = 0
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
            else:
                non_text += 1
        return "\n".join(texts), non_text
    # Any other shape (None, dict, number): count its string form — same
    # conservative-floor posture as the rest of the estimator.
    return ("" if content is None else str(content)), 0


def estimate_tokens(text: Any, model: str = "gpt-3.5-turbo", provider: str = "") -> int:
    """Estimate the token count of message content with one consistent strategy.

    Tiers: a custom tokenizer (only when one is actually installed), else
    tiktoken (when available), else a conservative chars-based floor. Never uses
    a whitespace word count. Non-string content (the multimodal part-list
    shape) is normalized first: text parts are estimated normally and each
    non-text part contributes :data:`NON_TEXT_PART_TOKEN_ESTIMATE`.

    Args:
        text: The text (or part-list content) to estimate.
        model: Model name (selects the tiktoken encoding / custom tokenizer).
        provider: Provider name (case-insensitive); selects the chars-path ratio
            and the custom tokenizer's provider patterns.

    Returns:
        Estimated token count (0 for empty content).
    """
    non_text_parts = 0
    if not isinstance(text, str):
        text, non_text_parts = _flatten_message_content(text)
        if non_text_parts:
            return (
                estimate_tokens(text, model, provider)
                + non_text_parts * NON_TEXT_PART_TOKEN_ESTIMATE
            )
    if not text:
        return 0
    # TASK-18602: memoized because every caller re-estimates the whole
    # append-only conversation each turn. The tiers below are all O(len)
    # or worse -- the tiktoken tier re-encodes, which is the dominant cost
    # on installs that have it -- so recomputing an unchanged message is
    # the single largest avoidable cost on the send and agent-turn paths.
    # See `_ESTIMATE_CACHE` for the key's design and its collision
    # argument.
    cache_key = (model, _norm_provider(provider), len(text), hash(text))
    cached = _ESTIMATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if CUSTOM_TOKENIZERS_AVAILABLE and custom_tokenizers_available():
        custom = count_tokens_with_custom(text, model, _norm_provider(provider))
        if custom is not None:
            _cache_estimate(cache_key, custom)
            return custom
    if TIKTOKEN_AVAILABLE:
        counted = count_tokens_tiktoken(text, model)
    else:
        counted = _chars_estimate(text, provider)
    _cache_estimate(cache_key, counted)
    return counted


def _cache_estimate(key: "tuple[str, str, int, int]", value: int) -> None:
    """Store one memoized estimate, bounding the cache.

    The read in `estimate_tokens` is deliberately unlocked: a dict `get` is
    atomic under the GIL, and a racing writer can only cause a miss (one
    extra recompute), never a torn or wrong value. Only the write takes the
    lock, so concurrent estimation from the agent worker thread and the UI
    thread never corrupts the dict's internal state.
    """
    with _ESTIMATE_CACHE_LOCK:
        if len(_ESTIMATE_CACHE) >= ESTIMATE_CACHE_MAX_ENTRIES:
            _ESTIMATE_CACHE.clear()
        _ESTIMATE_CACHE[key] = value


def clear_estimate_cache() -> None:
    """Drop every memoized estimate.

    Nothing depends on this for correctness -- entries are keyed by the
    text they describe, so a stale entry cannot be served for changed
    text. Exposed for tests that swap the tokenizer tier underneath the
    estimator and need the next call to actually recompute.
    """
    with _ESTIMATE_CACHE_LOCK:
        _ESTIMATE_CACHE.clear()


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
    # Default for unknown models. 16k, not the historical 4k: this value is what
    # every local provider (llama.cpp/ollama/vllm/koboldcpp) resolves to, since
    # their GGUF/model names never match the table above. 4k was far below what
    # modern local models actually serve -- a llama.cpp server advertising
    # n_ctx=64000 was being budgeted at 4096 -- which trimmed conversation
    # history almost immediately and cost characters their memory.
    "default": 16384,
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

    return _chars_estimate(text, "openai")


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


def get_table_model_token_limit(model: str, provider: str = "openai") -> int | None:
    """Return a known exact/prefix table limit without a provider fallback.

    Settings uses this tier to distinguish a detected model window from the
    deliberately conservative unknown-model fallback.
    """

    provider_key = _norm_provider(provider)
    if provider_key == "openrouter" and "/" in model:
        upstream_provider, upstream_model = model.split("/", 1)
        return get_table_model_token_limit(upstream_model, upstream_provider)
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]
    best_limit = None
    best_len = -1
    for model_prefix, limit in MODEL_TOKEN_LIMITS.items():
        if model_prefix == "default":
            continue
        if model.startswith(model_prefix) and len(model_prefix) > best_len:
            best_limit = limit
            best_len = len(model_prefix)
    return best_limit


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

    # OpenRouter IDs carry the upstream provider. Re-dispatch the full
    # resolution chain so an upstream provider fallback remains available.
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

    # 2-3. Exact or longest-prefix table match.
    table_limit = get_table_model_token_limit(model, provider)
    if table_limit is not None:
        return table_limit

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
