"""Shared provider-catalog display data.

Single source of truth for human display names and grouping of provider
config keys (task-180 / task-191). Consumed by the Settings screen and the
Console settings modal so both surfaces render identical labels; the
underlying config/readiness keys are never changed by this module.
"""

from __future__ import annotations

from collections.abc import Mapping

from tldw_chatbook.config import normalize_provider_config_key

# task-180: single source of human display names for provider keys rendered
# in Settings and Console. Labels are display-only; the underlying
# config/readiness keys are unchanged so existing config files keep working.
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "aphrodite": "Aphrodite Engine",
    "arcee": "Arcee AI",
    "azure": "Azure OpenAI",
    "baseten": "Baseten",
    "byteplus": "ByteDance Seed (BytePlus)",
    "cerebras": "Cerebras",
    "cloudflare": "Cloudflare Workers AI",
    "cohere": "Cohere",
    "commandcode": "Command Code",
    "custom": "Custom OpenAI-compatible",
    "custom_2": "Custom OpenAI-compatible #2",
    # ADR-179's execution-only engine key. Setup cannot own it, so neither the
    # Settings picker nor first-run setup lists it (TASK-33621.14).
    "custom_hosted": "Custom Hosted",
    "databricks": "Databricks",
    "deepinfra": "DeepInfra",
    "deepseek": "DeepSeek",
    "fireworks": "Fireworks",
    "gmi": "GMI Cloud",
    "google": "Google Gemini",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "kilo": "Kilo Gateway",
    "koboldcpp": "KoboldCpp",
    "llama_cpp": "llama.cpp",
    "local_llamacpp": "llama.cpp (legacy alias)",
    "local_llamafile": "Llamafile",
    "local_llm": "Local LLM (legacy generic)",
    "local_mlx_lm": "MLX-LM (Apple silicon)",
    "local_ollama": "Ollama (legacy alias)",
    "local_onnx": "ONNX Runtime (local)",
    "local_transformers": "Transformers (local)",
    "local_vllm": "vLLM (legacy alias)",
    "meta": "Meta (Muse Spark)",
    "mimo": "Xiaomi MiMo",
    "minimax": "MiniMax",
    "mistral": "Mistral AI (legacy alias)",
    "mistralai": "Mistral AI",
    "moonshot": "Moonshot AI",
    "nebius": "Nebius Token Factory",
    "nous": "Nous Research",
    "novita": "Novita AI",
    "nvidia": "NVIDIA NIM",
    "ollama": "Ollama",
    "ollama_cloud": "Ollama Cloud",
    "oobabooga": "Text Generation WebUI (Oobabooga)",
    "openai": "OpenAI",
    "opencode_zen": "OpenCode Zen",
    "openrouter": "OpenRouter",
    "qianfan": "Baidu Qianfan",
    "qwencloud": "QwenCloud",
    "sambanova": "SambaNova",
    "siliconflow": "SiliconFlow",
    "stepfun": "StepFun",
    "tabbyapi": "TabbyAPI",
    "together": "Together",
    "tokenhub": "Tencent TokenHub",
    "upstage": "Upstage",
    "venice": "Venice",
    "vercel": "Vercel AI Gateway",
    "vllm": "vLLM",
    "wandb": "W&B Inference (CoreWeave)",
    "zai": "Z.ai",
    "zenmux": "ZenMux",
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
#: The legacy aliases alone: the custom group minus the built-in custom slots,
#: which stay listable (ADR-146) while Console hides an alias unless it is
#: configured or current (ADR-066).
PROVIDER_LEGACY_ALIAS_KEYS = PROVIDER_CUSTOM_GROUP_KEYS - {"custom", "custom_2"}


def provider_display_name(
    provider_key: str, app_config: Mapping[str, object] | None = None
) -> str:
    """Return the human display name for a provider config key.

    Args:
        provider_key: Provider key in any saved spelling (``Llama_cpp``,
            ``local-llm``), or a ``custom-ep:<slug>`` registry id.
        app_config: Config holding the ADR-146 endpoint registry; when given,
            a registry id renders as its entry's ``display_name``.

    Returns:
        The mapped display name, or the key itself when unmapped so unknown
        providers stay identifiable rather than blank.
    """
    key = (provider_key or "").strip()
    if app_config is not None:
        # Lazy: the registry imports console_provider_support, which imports
        # this module.
        from tldw_chatbook.Chat.custom_endpoint_registry import entry_for

        entry = entry_for(app_config, key)
        if entry is not None:
            return entry.display_name
    return PROVIDER_DISPLAY_NAMES.get(normalize_provider_config_key(key), key)
