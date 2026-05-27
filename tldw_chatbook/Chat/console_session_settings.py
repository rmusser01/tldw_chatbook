"""Pure Console session settings contracts and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

from tldw_chatbook.Chat.provider_readiness import (
    get_provider_readiness,
    provider_config_key,
)
from tldw_chatbook.Utils.input_validation import validate_url


NATIVE_CONSOLE_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})
DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:9099"
INVALID_LLAMACPP_BASE_URL_COPY = (
    "Provider blocked: invalid llama.cpp base URL. "
    "Use an http(s) URL such as http://127.0.0.1:9099."
)
TokenCounter = Callable[[Sequence[Mapping[str, str]], str, str], int]
TokenLimitResolver = Callable[[str, str], int]
URL_BASED_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "koboldcpp",
        "llama_cpp",
        "local_llamacpp",
        "local_llamafile",
        "local_ollama",
        "local_vllm",
        "ollama",
        "oobabooga",
        "tabbyapi",
        "vllm",
    }
)
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


def build_console_provider_options(
    providers_models: Mapping[str, Sequence[str]],
) -> list[ConsoleSettingsOption]:
    """Return sorted provider options from the configured model registry."""
    provider_keys = sorted({key for key in (provider_config_key(provider) for provider in providers_models) if key})
    return [ConsoleSettingsOption(label=provider_key, value=provider_key) for provider_key in provider_keys]


def build_console_model_options(
    provider: str,
    providers_models: Mapping[str, Sequence[str]],
    current_model: str | None = None,
) -> list[ConsoleSettingsOption]:
    """Return model options for a provider, preserving the current model."""
    provider_key = provider_config_key(provider)
    model_values: list[str] = []

    current_model_value = _string_value(current_model)
    if current_model_value and current_model_value not in model_values:
        model_values.append(current_model_value)

    for configured_provider, configured_models in providers_models.items():
        if provider_config_key(configured_provider) != provider_key:
            continue
        for configured_model in configured_models:
            configured_model_value = _string_value(configured_model)
            if configured_model_value and configured_model_value not in model_values:
                model_values.append(configured_model_value)

    return [ConsoleSettingsOption(label=model, value=model) for model in model_values]


def build_default_console_session_settings(
    app_config: Mapping[str, object],
    provider: str | None = None,
    model: str | None = None,
) -> ConsoleSessionSettings:
    """Build default Console settings from chat defaults and provider config."""
    chat_defaults = _mapping_value(app_config, "chat_defaults")
    configured_provider = provider_config_key(_string_value(provider) or _string_setting(chat_defaults, "provider"))
    provider_settings = _provider_settings(app_config, configured_provider)

    return ConsoleSessionSettings(
        provider=configured_provider,
        model=_first_string(
            model,
            provider_settings.get("model"),
            provider_settings.get("api_model"),
            provider_settings.get("default_model"),
            chat_defaults.get("model"),
        ),
        base_url=_default_base_url(configured_provider, provider_settings),
        temperature=_float_setting(chat_defaults, provider_settings, "temperature", 0.7),
        top_p=_float_setting(chat_defaults, provider_settings, "top_p", 0.95),
        min_p=_optional_float_setting(chat_defaults, provider_settings, "min_p"),
        top_k=_optional_int_setting(chat_defaults, provider_settings, "top_k"),
        max_tokens=_optional_int_setting(chat_defaults, provider_settings, "max_tokens"),
        streaming=_bool_setting(chat_defaults, provider_settings, "streaming", True),
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

    return errors


def build_console_settings_readiness(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
    native_provider_keys: set[str] | None = None,
) -> ConsoleSettingsReadiness:
    """Build readiness copy without probing networks or mutating state."""
    provider_key = provider_config_key(settings.provider)
    native_keys = {
        provider_config_key(provider)
        for provider in (native_provider_keys if native_provider_keys is not None else NATIVE_CONSOLE_PROVIDER_KEYS)
    }

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

    readiness = get_provider_readiness(settings.provider, app_config, environ=environ)
    native_send_supported = provider_key in native_keys and readiness.ready
    provider_is_configured = _has_provider_settings_key(app_config, provider_key)
    if provider_key not in native_keys and (provider_is_configured or readiness.reason != "Unknown provider"):
        detail = f"Console native provider '{provider_key}' is not wired yet."
        if readiness.reason == "Missing API key":
            detail = f"{detail} This provider also has a missing API key."
        elif readiness.reason and readiness.reason != "Ready":
            detail = f"{detail} {readiness.user_message}"
        return ConsoleSettingsReadiness(
            label="WIP",
            detail=detail,
            native_send_supported=False,
        )

    if readiness.reason == "Unknown provider":
        return ConsoleSettingsReadiness(
            label="Unknown",
            detail=readiness.user_message,
            native_send_supported=False,
        )

    if readiness.ready:
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

    character_label = _string_value(settings.character_label)
    persona_label = _string_value(settings.persona_label) or "General"
    identity_row = f"Character: {character_label}" if character_label else f"Persona: {persona_label}"

    return ConsoleSettingsSummaryState(
        model_row=f"Model: {provider_label} / {model_label}{readiness_suffix}",
        context_row=_format_context_summary_row(context_estimate.label),
        sampling_row=f"Sampling: {', '.join(sampling_parts)}",
        identity_row=identity_row,
        readiness_label=readiness_label,
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


def _provider_settings(app_config: Mapping[str, object], provider_key: str) -> Mapping[str, object]:
    api_settings = _mapping_value(app_config, "api_settings")
    value = {}
    for configured_provider, configured_value in api_settings.items():
        if provider_config_key(configured_provider) == provider_key:
            value = configured_value
            break
    return value if isinstance(value, Mapping) else {}


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
    return provider_key in URL_BASED_PROVIDER_KEYS or any(
        key in provider_settings for key in ("api_url", "base_url", "api_base")
    )


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


def _format_summary_float(value: float) -> str:
    return f"{float(value):.2f}"


def _format_context_summary_row(label: str) -> str:
    label_text = _string_value(label) or "unavailable"
    if label_text.lower() in {"unknown", "context: unknown"}:
        label_text = "Context: unavailable"
    return label_text if label_text.startswith("Context: ") else f"Context: {label_text}"
