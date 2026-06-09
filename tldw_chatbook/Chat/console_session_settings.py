"""Pure Console session settings contracts and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

from tldw_chatbook.Chat.console_provider_support import (
    DIRECT_CONSOLE_PROVIDER_KEYS,
    resolve_console_provider_identity,
    supported_console_provider_catalog,
    supported_console_provider_readiness_keys,
)
from tldw_chatbook.Chat.console_provider_endpoints import (
    URL_BASED_PROVIDER_KEYS,
    first_configured_endpoint,
    generic_endpoint_differs,
    normalize_generic_endpoint_for_compare,
    provider_uses_endpoint,
    safe_endpoint_display,
    unsaved_endpoint_copy,
)
from tldw_chatbook.Chat.provider_readiness import (
    get_provider_readiness,
    provider_config_key,
)
from tldw_chatbook.Utils.input_validation import validate_url


NATIVE_CONSOLE_PROVIDER_KEYS = DIRECT_CONSOLE_PROVIDER_KEYS
CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS = frozenset(
    {
        "anthropic",
        "aphrodite",
        "cohere",
        "custom-openai-api",
        "custom-openai-api-2",
        "deepseek",
        "google",
        "groq",
        "huggingface",
        "koboldcpp",
        "llama_cpp",
        "local-llm",
        "local_llamacpp",
        "local_llamafile",
        "local_mlx_lm",
        "local_ollama",
        "local_vllm",
        "mistral",
        "mistralai",
        "mlx_lm",
        "moonshot",
        "ollama",
        "oobabooga",
        "openai",
        "openrouter",
        "tabbyapi",
        "vllm",
        "zai",
    }
)
DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:9099"
INVALID_LLAMACPP_BASE_URL_COPY = (
    "Provider blocked: invalid llama.cpp base URL. "
    "Use an http(s) URL such as http://127.0.0.1:9099."
)
MODEL_OPTION_PLACEHOLDER_VALUES = frozenset({"none", "null"})
TokenCounter = Callable[[Sequence[Mapping[str, str]], str, str], int]
TokenLimitResolver = Callable[[str, str], int]
CONSOLE_MODEL_TOKEN_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-instant-1.2": 100000,
    "gemini-pro": 30720,
    "gemini-pro-vision": 12288,
    "mistral-large": 32000,
    "mistral-medium": 32000,
    "mistral-small": 32000,
    "mixtral-8x7b": 32000,
    "default": 4096,
}
CONSOLE_PROVIDER_TOKEN_LIMIT_DEFAULTS = {
    "anthropic": 100000,
    "google": 30720,
    "openai": 4096,
    "mistral": 32000,
}
CONSOLE_TOKEN_CHAR_RATIOS = {
    "google": 0.3,
    "huggingface": 0.3,
    "default": 0.25,
}
_REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_REASONING_SUMMARY_VALUES = frozenset({"auto", "concise", "detailed", "none"})
_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
_THINKING_EFFORT_VALUES = frozenset({"off", "low", "medium", "high", "xhigh", "max"})


def normalize_llamacpp_base_url(api_url: str | None) -> str:
    """Return the llama.cpp origin root used before appending OpenAI paths."""
    raw_url = str(api_url or "").strip()
    if not raw_url:
        return DEFAULT_LLAMACPP_BASE_URL

    candidate = raw_url if "://" in raw_url else f"http://{raw_url}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return raw_url.rstrip("/")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return raw_url.rstrip("/")

    path = parsed.path.rstrip("/")
    normalized_endpoint_paths = {
        "/v1",
        "/v1/models",
        "/models",
        "/v1/chat/completions",
        "/chat/completions",
        "/completion",
        "/completions",
    }
    if path.lower() in normalized_endpoint_paths:
        path = ""
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")
    return normalized or DEFAULT_LLAMACPP_BASE_URL


@dataclass(frozen=True)
class ConsoleSessionSettings:
    """User-editable Console chat session settings."""

    provider: str
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    thinking_effort: str | None = None
    thinking_budget_tokens: int | None = None
    streaming: bool = True
    persona_label: str = "General"
    character_label: str = ""


@dataclass(frozen=True)
class ConsoleSettingsOption:
    """Selectable settings option for provider and model controls."""

    label: str
    value: str


@dataclass(frozen=True)
class ConsoleSettingsReadiness:
    """Readiness copy for the currently selected Console settings."""

    label: str
    detail: str
    native_send_supported: bool


@dataclass(frozen=True)
class ConsoleSettingsContextEstimate:
    """Estimated context usage for the current Console session."""

    used_tokens: int | None
    token_limit: int | None
    label: str
    staged_source_count: int = 0
    staged_context_summary: str = ""


@dataclass(frozen=True)
class ConsoleSettingsSummaryState:
    """Compact Console settings summary rows for rail display."""

    model_row: str
    context_row: str
    sampling_row: str
    identity_row: str
    readiness_label: str = ""
    provider_row: str = ""
    endpoint_row: str = ""
    credential_row: str = ""
    transport_row: str = ""
    action_label: str = "Configure"
    action_tooltip: str = "Configure Console settings"


def build_console_provider_options(
    providers_models: Mapping[str, Sequence[str]],
) -> list[ConsoleSettingsOption]:
    """Return sorted Console-sendable provider options plus configured providers."""
    provider_keys = sorted({key for key in (provider_config_key(provider) for provider in providers_models) if key})
    provider_keys = sorted(
        {
            *provider_keys,
            *(
                entry.readiness_key
                for entry in supported_console_provider_catalog(
                    CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
                )
            ),
        }
    )
    return [ConsoleSettingsOption(label=provider_key, value=provider_key) for provider_key in provider_keys]


def build_console_model_options(
    provider: str,
    providers_models: Mapping[str, Sequence[str]],
    current_model: str | None = None,
) -> list[ConsoleSettingsOption]:
    """Return model options for a provider, preserving the current model."""
    provider_key = provider_config_key(provider)
    model_values: list[str] = []

    current_model_value = normalize_console_model_value(current_model)
    if current_model_value and current_model_value not in model_values:
        model_values.append(current_model_value)

    for configured_provider, configured_models in providers_models.items():
        if provider_config_key(configured_provider) != provider_key:
            continue
        for configured_model in configured_models:
            configured_model_value = normalize_console_model_value(configured_model)
            if configured_model_value and configured_model_value not in model_values:
                model_values.append(configured_model_value)

    return [ConsoleSettingsOption(label=model, value=model) for model in model_values]


def build_default_console_session_settings(
    app_config: Mapping[str, object],
    provider: str | None = None,
    model: str | None = None,
) -> ConsoleSessionSettings:
    """Build default Console settings from chat defaults and provider config."""
    chat_defaults = _chat_defaults_with_streaming_compat(
        _mapping_value(app_config, "chat_defaults")
    )
    configured_provider = provider_config_key(_string_value(provider) or _string_setting(chat_defaults, "provider"))
    provider_settings = _provider_settings(app_config, configured_provider)
    configured_model = _first_string(
        model,
        provider_settings.get("model"),
        provider_settings.get("api_model"),
        provider_settings.get("default_model"),
        chat_defaults.get("model"),
    )
    model_profile = _model_default_profile(provider_settings, configured_model)
    default_sources = (model_profile, chat_defaults, provider_settings)

    return ConsoleSessionSettings(
        provider=configured_provider,
        model=configured_model,
        base_url=_default_base_url(configured_provider, provider_settings),
        temperature=_float_setting_from_sources(default_sources, "temperature", 0.7),
        top_p=_float_setting_from_sources(default_sources, "top_p", 0.95),
        min_p=_optional_float_setting_from_sources(default_sources, "min_p"),
        top_k=_optional_int_setting_from_sources(default_sources, "top_k"),
        max_tokens=_optional_int_setting_from_sources(default_sources, "max_tokens"),
        seed=_optional_int_setting_from_sources(default_sources, "seed"),
        presence_penalty=_optional_float_setting_from_sources(default_sources, "presence_penalty"),
        frequency_penalty=_optional_float_setting_from_sources(default_sources, "frequency_penalty"),
        reasoning_effort=_optional_string_setting_from_sources(default_sources, "reasoning_effort"),
        reasoning_summary=_optional_string_setting_from_sources(default_sources, "reasoning_summary"),
        verbosity=_optional_string_setting_from_sources(default_sources, "verbosity"),
        thinking_effort=_optional_string_setting_from_sources(default_sources, "thinking_effort"),
        thinking_budget_tokens=_optional_int_setting_from_sources(default_sources, "thinking_budget_tokens"),
        streaming=_bool_setting_from_sources(default_sources, "streaming", True),
    )


def validate_console_session_settings(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
) -> list[str]:
    """Return user-facing validation errors for Console settings."""
    errors: list[str] = []
    provider_key = provider_config_key(settings.provider)
    provider_settings = _provider_settings(app_config, provider_key)

    if not provider_key:
        errors.append("Provider is required.")
    if provider_key not in NATIVE_CONSOLE_PROVIDER_KEYS and not _string_value(settings.model):
        errors.append("Model is required.")

    base_url = _string_value(settings.base_url)
    if base_url and _is_url_based_provider(provider_key, provider_settings) and not _valid_base_url(provider_key, base_url):
        errors.append("Base URL must be a valid http(s) URL.")

    if not _float_in_range(settings.temperature, 0.0, 2.0):
        errors.append("Temperature must be between 0 and 2.")
    if not _float_in_range(settings.top_p, 0.0, 1.0):
        errors.append("Top P must be between 0 and 1.")
    if not _is_blank_value(settings.min_p) and not _float_in_range(settings.min_p, 0.0, 1.0):
        errors.append("Min P must be between 0 and 1.")
    if not _is_blank_value(settings.top_k) and not _optional_int_at_least(settings.top_k, 0):
        errors.append("Top K must be 0 or greater.")
    if not _is_blank_value(settings.max_tokens) and not _optional_int_at_least(settings.max_tokens, 1):
        errors.append("Max tokens must be 1 or greater.")
    if not _is_blank_value(settings.seed) and not _optional_int_at_least(settings.seed, 0):
        errors.append("Seed must be 0 or greater.")
    if not _is_blank_value(settings.presence_penalty) and not _float_in_range(settings.presence_penalty, -2.0, 2.0):
        errors.append("Presence penalty must be between -2 and 2.")
    if not _is_blank_value(settings.frequency_penalty) and not _float_in_range(settings.frequency_penalty, -2.0, 2.0):
        errors.append("Frequency penalty must be between -2 and 2.")
    if not _is_blank_value(settings.reasoning_effort) and settings.reasoning_effort not in _REASONING_EFFORT_VALUES:
        errors.append("Reasoning effort must be one of none, minimal, low, medium, high, or xhigh.")
    if not _is_blank_value(settings.reasoning_summary) and settings.reasoning_summary not in _REASONING_SUMMARY_VALUES:
        errors.append("Reasoning summary must be one of auto, concise, detailed, or none.")
    if not _is_blank_value(settings.verbosity) and settings.verbosity not in _VERBOSITY_VALUES:
        errors.append("Verbosity must be one of low, medium, or high.")
    if not _is_blank_value(settings.thinking_effort) and settings.thinking_effort not in _THINKING_EFFORT_VALUES:
        errors.append("Thinking effort must be one of off, low, medium, high, xhigh, or max.")
    if not _is_blank_value(settings.thinking_budget_tokens) and not _optional_int_at_least(settings.thinking_budget_tokens, 1024):
        errors.append("Thinking budget tokens must be at least 1024.")

    return errors


def build_console_settings_readiness(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
    native_provider_keys: set[str] | None = None,
) -> ConsoleSettingsReadiness:
    """Build readiness copy without probing networks or mutating state."""
    identity = resolve_console_provider_identity(
        settings.provider,
        handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )
    provider_key = identity.readiness_key
    supported_keys = _supported_readiness_keys(native_provider_keys)
    send_capable_keys = _send_capable_readiness_keys(native_provider_keys)

    base_url = _string_value(settings.base_url)
    provider_settings = _provider_settings(app_config, provider_key)
    if base_url and _is_url_based_provider(provider_key, provider_settings) and not _valid_base_url(provider_key, base_url):
        detail = (
            INVALID_LLAMACPP_BASE_URL_COPY
            if provider_key in NATIVE_CONSOLE_PROVIDER_KEYS
            else "Provider blocked: invalid base URL. Use an http(s) URL."
        )
        return ConsoleSettingsReadiness(
            label="Invalid URL",
            detail=detail,
            native_send_supported=False,
        )
    if (
        base_url
        and _is_url_based_provider(provider_key, provider_settings)
        and _endpoint_differs_for_provider(provider_key, base_url, provider_settings)
    ):
        return ConsoleSettingsReadiness(
            label="Endpoint not saved",
            detail=unsaved_endpoint_copy(base_url, provider_settings),
            native_send_supported=False,
        )

    readiness = get_provider_readiness(provider_key, app_config, environ=environ)
    provider_supported = provider_key in supported_keys
    native_send_supported = provider_key in send_capable_keys and readiness.ready

    if not provider_supported:
        return ConsoleSettingsReadiness(
            label="Unknown",
            detail=(
                f"Provider blocked: '{provider_key}' is not available in Console yet. "
                "Choose a supported provider."
            ),
            native_send_supported=False,
        )

    if readiness.reason == "Unknown provider":
        return ConsoleSettingsReadiness(
            label="Unknown",
            detail=readiness.user_message,
            native_send_supported=False,
        )

    if readiness.ready:
        if not native_send_supported:
            return ConsoleSettingsReadiness(
                label="Pending",
                detail=f"Provider ready; Console send support is pending for '{provider_key}'.",
                native_send_supported=False,
            )
        return ConsoleSettingsReadiness(
            label="Ready",
            detail=readiness.user_message,
            native_send_supported=native_send_supported,
        )

    if readiness.reason == "Missing API key":
        return ConsoleSettingsReadiness(
            label="Missing key",
            detail=readiness.user_message,
            native_send_supported=False,
        )

    return ConsoleSettingsReadiness(
        label="Not ready",
        detail=readiness.user_message,
        native_send_supported=False,
    )


def _supported_readiness_keys(native_provider_keys: set[str] | None = None) -> frozenset[str]:
    """Return readiness keys accepted by Console readiness.

    ``native_provider_keys`` is retained for older tests/callers that injected a
    support set before generic Console provider support existed.
    """
    supported_keys = supported_console_provider_readiness_keys(
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )
    if native_provider_keys is not None:
        injected_keys = frozenset(
            resolve_console_provider_identity(
                provider,
                handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
            ).readiness_key
            for provider in native_provider_keys
        )
        return supported_keys | injected_keys
    return supported_keys


def _send_capable_readiness_keys(native_provider_keys: set[str] | None = None) -> frozenset[str]:
    """Return readiness keys that currently have a wired Console send path."""
    send_capable_keys = supported_console_provider_readiness_keys(
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )
    if native_provider_keys is not None:
        injected_keys = frozenset(
            resolve_console_provider_identity(
                provider,
                handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
            ).readiness_key
            for provider in native_provider_keys
        )
        return send_capable_keys | injected_keys
    return send_capable_keys


def build_console_settings_summary_state(
    settings: ConsoleSessionSettings,
    context_estimate: ConsoleSettingsContextEstimate,
    readiness: ConsoleSettingsReadiness,
) -> ConsoleSettingsSummaryState:
    """Build compact display rows for the Console settings summary widget."""
    provider_label = _string_value(settings.provider) or "Unknown"
    model_value = _string_value(settings.model)
    readiness_label = _string_value(readiness.label) or ""
    model_is_missing = not model_value and readiness_label == "Missing model"
    model_label = model_value or ("Missing" if model_is_missing else "Default")
    readiness_suffix = (
        ""
        if readiness_label in {"", "Ready"} or model_is_missing
        else f" ({readiness_label})"
    )
    action_label = "Configure"
    action_tooltip = "Configure Console settings"
    if model_is_missing:
        action_label = "Choose Model"
        action_tooltip = "Choose a model for this Console session"

    sampling_parts = [
        f"T {_format_summary_float(settings.temperature)}",
        f"P {_format_summary_float(settings.top_p)}",
    ]
    if settings.min_p is not None:
        sampling_parts.append(f"min_p {_format_summary_float(settings.min_p)}")
    if settings.top_k is not None:
        sampling_parts.append(f"top_k {settings.top_k}")
    if settings.max_tokens is not None:
        sampling_parts.append(f"max_tokens {settings.max_tokens}")
    if settings.seed is not None:
        sampling_parts.append(f"seed {settings.seed}")
    if settings.reasoning_effort:
        sampling_parts.append(f"reasoning {settings.reasoning_effort}")
    elif settings.thinking_effort:
        sampling_parts.append(f"thinking {settings.thinking_effort}")

    character_label = _string_value(settings.character_label)
    persona_label = _string_value(settings.persona_label) or "General"
    identity_row = f"Character: {character_label}" if character_label else f"Persona: {persona_label}"

    return ConsoleSettingsSummaryState(
        model_row=f"Model: {model_label}{readiness_suffix}",
        context_row=_format_context_summary_row(context_estimate.label),
        sampling_row=f"Sampling: {', '.join(sampling_parts)}",
        identity_row=identity_row,
        readiness_label=readiness_label,
        provider_row=f"Provider: {provider_label}",
        endpoint_row=_format_endpoint_summary_row(settings),
        credential_row=_format_credential_summary_row(readiness),
        transport_row=f"Streaming: {'on' if settings.streaming else 'off'}",
        action_label=action_label,
        action_tooltip=action_tooltip,
    )


def build_console_context_estimate(
    messages: Sequence[Mapping[str, str]],
    provider: str,
    model: str | None,
    staged_source_count: int = 0,
    staged_context_summary: str = "",
    max_tokens_response: int | None = None,
    system_prompt: str | None = None,
    *,
    token_counter: TokenCounter | None = None,
    token_limit_resolver: TokenLimitResolver | None = None,
) -> ConsoleSettingsContextEstimate:
    """Estimate current context tokens for display in Console settings."""
    model_name = _string_value(model)
    if not model_name:
        return ConsoleSettingsContextEstimate(
            used_tokens=None,
            token_limit=None,
            label="Context: unavailable",
            staged_source_count=staged_source_count,
            staged_context_summary=staged_context_summary,
        )

    provider_key = provider_config_key(provider)
    estimate_messages: list[Mapping[str, str]] = []
    if system_prompt:
        estimate_messages.append({"role": "system", "content": system_prompt})
    estimate_messages.extend(messages)

    try:
        counter = token_counter or _estimate_tokens_locally
        limit_resolver = token_limit_resolver or _resolve_token_limit_locally
        used_tokens = counter(list(estimate_messages), model_name, provider_key)
        token_limit = limit_resolver(model_name, provider_key)
    except Exception:
        return ConsoleSettingsContextEstimate(
            used_tokens=None,
            token_limit=None,
            label="Context: unavailable",
            staged_source_count=staged_source_count,
            staged_context_summary=staged_context_summary,
        )

    label = f"{used_tokens:,} / {token_limit:,} tokens"
    if max_tokens_response is not None:
        label = f"{label}; {max_tokens_response:,} response tokens reserved"
    if staged_source_count:
        source_word = "source" if staged_source_count == 1 else "sources"
        label = f"{label}; {staged_source_count} {source_word} staged"

    return ConsoleSettingsContextEstimate(
        used_tokens=used_tokens,
        token_limit=token_limit,
        label=label,
        staged_source_count=staged_source_count,
        staged_context_summary=staged_context_summary,
    )


def _mapping_value(source: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = source.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _chat_defaults_with_streaming_compat(
    chat_defaults: Mapping[str, object],
) -> Mapping[str, object]:
    """Return chat defaults with the legacy streaming key bridged.

    `chat_defaults.streaming` is the canonical Console default. Older config can
    still provide `chat_defaults.enable_streaming`; it is only read when the
    canonical key is absent.
    """
    if "streaming" in chat_defaults or "enable_streaming" not in chat_defaults:
        return chat_defaults
    compatible_defaults = dict(chat_defaults)
    compatible_defaults["streaming"] = chat_defaults.get("enable_streaming")
    return compatible_defaults


def _provider_settings(app_config: Mapping[str, object], provider_key: str) -> Mapping[str, object]:
    api_settings = _mapping_value(app_config, "api_settings")
    value = {}
    for configured_provider, configured_value in api_settings.items():
        if provider_config_key(configured_provider) == provider_key:
            value = configured_value
            break
    return value if isinstance(value, Mapping) else {}


def _model_default_profile(
    provider_settings: Mapping[str, object],
    model: str | None,
) -> Mapping[str, object]:
    model_name = _string_value(model)
    if not model_name:
        return {}
    model_defaults = provider_settings.get("model_defaults", {})
    if not isinstance(model_defaults, Mapping):
        return {}
    profile = model_defaults.get(model_name, {})
    return profile if isinstance(profile, Mapping) else {}


def _has_provider_settings_key(app_config: Mapping[str, object], provider_key: str) -> bool:
    api_settings = _mapping_value(app_config, "api_settings")
    return any(provider_config_key(configured_provider) == provider_key for configured_provider in api_settings)


def _default_base_url(provider_key: str, provider_settings: Mapping[str, object]) -> str | None:
    base_url = _first_string(
        provider_settings.get("api_url"),
        provider_settings.get("base_url"),
        provider_settings.get("api_base"),
    )
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        return normalize_llamacpp_base_url(base_url or DEFAULT_LLAMACPP_BASE_URL)
    return base_url


def _is_url_based_provider(provider_key: str, provider_settings: Mapping[str, object]) -> bool:
    return provider_uses_endpoint(provider_key, provider_settings)


def _endpoint_differs_for_provider(
    provider_key: str,
    base_url: str | None,
    provider_settings: Mapping[str, object],
) -> bool:
    """Return whether a selected endpoint differs from persisted provider settings."""
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        configured_endpoint = first_configured_endpoint(provider_settings)
        if not configured_endpoint:
            selected = normalize_generic_endpoint_for_compare(
                normalize_llamacpp_base_url(base_url)
            )
            default = normalize_generic_endpoint_for_compare(DEFAULT_LLAMACPP_BASE_URL)
            return bool(selected) and selected != default
        selected = normalize_generic_endpoint_for_compare(normalize_llamacpp_base_url(base_url))
        configured = normalize_generic_endpoint_for_compare(normalize_llamacpp_base_url(configured_endpoint))
        return selected != configured
    return generic_endpoint_differs(base_url, provider_settings)


def _valid_base_url(provider_key: str, base_url: str) -> bool:
    try:
        candidate = (
            normalize_llamacpp_base_url(base_url)
            if provider_key in NATIVE_CONSOLE_PROVIDER_KEYS
            else base_url
        )
    except ValueError:
        return False
    return validate_url(candidate) and _has_valid_url_port(candidate)


def _has_valid_url_port(url: str) -> bool:
    try:
        parsed = urlparse(url)
        parsed.port
    except ValueError:
        return False
    return parsed.port is None or 0 < parsed.port <= 65535


def _float_in_range(value: object, minimum: float, maximum: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return minimum <= number <= maximum


def _optional_int_at_least(value: object, minimum: int) -> bool:
    parsed = _parse_optional_int(value)
    return parsed is not None and parsed >= minimum


def _is_blank_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _float_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    value = primary.get(key) if key in primary else fallback.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _setting_value_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: object = None,
) -> object:
    for source in sources:
        if key in source:
            value = source.get(key)
            if not _is_blank_value(value):
                return value
    return default


def _float_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: float,
) -> float:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if _is_blank_value(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _optional_float_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
) -> float | None:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if _is_blank_value(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _optional_int_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
) -> int | None:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if _is_blank_value(value):
            continue
        parsed = _parse_optional_int(value)
        if parsed is not None:
            return parsed
    return None


def _optional_string_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
) -> str | None:
    for source in sources:
        value = source.get(key)
        text = _string_value(value)
        if text:
            return text
    return None


def _bool_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: bool,
) -> bool:
    for source in sources:
        if key not in source:
            continue
        value = source.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
    return default


def _optional_float_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
) -> float | None:
    if key in primary:
        value = primary.get(key)
    else:
        value = fallback.get(key)
    if _is_blank_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
) -> int | None:
    if key in primary:
        value = primary.get(key)
    else:
        value = fallback.get(key)
    return _parse_optional_int(value)


def _parse_optional_int(value: object) -> int | None:
    if _is_blank_value(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdecimal():
            return int(stripped)
        if stripped.startswith("-") and stripped[1:].isdecimal():
            return int(stripped)
    return None


def _estimate_tokens_locally(
    messages: Sequence[Mapping[str, str]],
    model: str,
    provider: str,
) -> int:
    del model
    ratio = CONSOLE_TOKEN_CHAR_RATIOS.get(provider, CONSOLE_TOKEN_CHAR_RATIOS["default"])
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("role", "")))
        total_chars += len(str(message.get("content", "")))
    message_overhead = len(messages) * 10
    return int((total_chars + message_overhead) * ratio)


def _resolve_token_limit_locally(model: str, provider: str) -> int:
    if model in CONSOLE_MODEL_TOKEN_LIMITS:
        return CONSOLE_MODEL_TOKEN_LIMITS[model]

    model_limits = (
        (prefix, limit)
        for prefix, limit in CONSOLE_MODEL_TOKEN_LIMITS.items()
        if prefix != "default"
    )
    for model_prefix, limit in sorted(model_limits, key=lambda item: len(item[0]), reverse=True):
        if model.startswith(model_prefix):
            return limit

    return CONSOLE_PROVIDER_TOKEN_LIMIT_DEFAULTS.get(provider, CONSOLE_MODEL_TOKEN_LIMITS["default"])


def _bool_setting(
    primary: Mapping[str, object],
    fallback: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = primary.get(key) if key in primary else fallback.get(key, default)
    return value if isinstance(value, bool) else default


def _first_string(*values: object) -> str | None:
    for value in values:
        text = _string_value(value)
        if text:
            return text
    return None


def _string_setting(source: Mapping[str, object], key: str) -> str:
    return _string_value(source.get(key)) or ""


def _string_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def normalize_console_model_value(value: object) -> str | None:
    """Return a model value unless it is blank or a placeholder sentinel."""
    text = _string_value(value)
    if text is None or text.lower() in MODEL_OPTION_PLACEHOLDER_VALUES:
        return None
    return text


def _format_summary_float(value: float) -> str:
    return f"{float(value):.2f}"


def _format_context_summary_row(label: str) -> str:
    label_text = _string_value(label) or "unavailable"
    if label_text.lower() in {"unknown", "context: unknown"}:
        label_text = "Context: unavailable"
    return label_text if label_text.startswith("Context: ") else f"Context: {label_text}"


def _format_endpoint_summary_row(settings: ConsoleSessionSettings) -> str:
    endpoint = safe_endpoint_display(settings.base_url)
    return f"Endpoint: {endpoint or 'provider default'}"


def _format_credential_summary_row(readiness: ConsoleSettingsReadiness) -> str:
    label = (_string_value(readiness.label) or "").lower()
    detail = (_string_value(readiness.detail) or "").lower()
    if label == "missing key" or "missing api key" in detail:
        return "Credential: missing"
    if "no api key is required" in detail:
        return "Credential: not required"
    if "api key found" in detail:
        return "Credential: ready"
    return "Credential: check setup"
