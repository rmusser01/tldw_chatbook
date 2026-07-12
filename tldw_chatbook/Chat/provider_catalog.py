"""Shared provider-catalog display data.

Single source of truth for human display names and grouping of provider
config keys (task-180 / task-191). Consumed by the Settings screen and the
Console settings modal so both surfaces render identical labels; the
underlying config/readiness keys are never changed by this module.
"""

from __future__ import annotations

# task-180: single source of human display names for provider keys rendered
# in Settings and Console. Labels are display-only; the underlying
# config/readiness keys are unchanged so existing config files keep working.
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "aphrodite": "Aphrodite Engine",
    "cohere": "Cohere",
    "custom": "Custom OpenAI-compatible",
    "custom_2": "Custom OpenAI-compatible #2",
    "deepseek": "DeepSeek",
    "google": "Google Gemini",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "koboldcpp": "KoboldCpp",
    "llama_cpp": "llama.cpp",
    "local_llamacpp": "llama.cpp (legacy alias)",
    "local_llamafile": "Llamafile",
    "local_llm": "Local LLM (legacy generic)",
    "local_mlx_lm": "MLX-LM (Apple silicon)",
    "local_ollama": "Ollama (legacy alias)",
    "local_vllm": "vLLM (legacy alias)",
    "mistral": "Mistral AI (legacy alias)",
    "mistralai": "Mistral AI",
    "moonshot": "Moonshot AI",
    "ollama": "Ollama",
    "oobabooga": "Text Generation WebUI (Oobabooga)",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "tabbyapi": "TabbyAPI",
    "vllm": "vLLM",
    "zai": "Z.ai",
}

PROVIDER_GROUP_CLOUD = "Cloud"
PROVIDER_GROUP_LOCAL = "Local"
PROVIDER_GROUP_CUSTOM = "Custom & legacy aliases"
PROVIDER_GROUP_ORDER = (
    PROVIDER_GROUP_CLOUD,
    PROVIDER_GROUP_LOCAL,
    PROVIDER_GROUP_CUSTOM,
)

# task-180: legacy/alias and custom keys stay selectable for config
# compatibility but sort last, so new users pick the canonical entry
# (llama_cpp over local_llamacpp, ollama over local_ollama, mistralai over
# mistral, vllm over local_vllm).
PROVIDER_CUSTOM_GROUP_KEYS = frozenset(
    {
        "custom",
        "custom_2",
        "local_llamacpp",
        "local_llm",
        "local_ollama",
        "local_vllm",
        "mistral",
    }
)


def provider_display_name(provider_key: str) -> str:
    """Return the human display name for a provider config key.

    Args:
        provider_key: Provider key as it appears in config/readiness maps.

    Returns:
        The mapped display name, or the key itself when unmapped so unknown
        providers stay identifiable rather than blank.
    """
    key = (provider_key or "").strip()
    return PROVIDER_DISPLAY_NAMES.get(key, key)
