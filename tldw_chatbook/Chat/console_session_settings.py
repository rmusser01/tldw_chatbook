"""Pure Console session settings contracts and helpers."""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence, overload
from urllib.parse import urlparse, urlunparse

from tldw_chatbook.Chat.console_provider_support import (
    DIRECT_CONSOLE_PROVIDER_KEYS,
    build_local_thinking_payload_fields,
    resolve_console_provider_identity,
    supported_console_provider_catalog,
    supported_console_provider_readiness_keys,
)
from tldw_chatbook.Chat.console_provider_endpoints import (
    URL_BASED_PROVIDER_KEYS,  # noqa: F401  (re-exported; console_settings_modal imports it from here)
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
from tldw_chatbook.config import ProviderSettingsError, provider_settings_for_key
from tldw_chatbook.model_capabilities import anthropic_model_rejects_disabled_thinking
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.token_counter import count_tokens_messages
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label


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
        "qwencloud",
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
    "default": 8001,
}
CONSOLE_PROVIDER_TOKEN_LIMIT_DEFAULTS = {
    "anthropic": 100000,
    "google": 30720,
    "openai": 8001,
    "mistral": 32000,
}
_REASONING_EFFORT_VALUES = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
_REASONING_SUMMARY_VALUES = frozenset({"auto", "concise", "detailed", "none"})
_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
_THINKING_EFFORT_VALUES = frozenset({"off", "low", "medium", "high", "xhigh", "max"})
_LEGACY_CHAT_PROVIDER_ALIASES = {
    "openai_compatible": "openai",
}
# Generation-aware: dotted Qwen3.x generations consume effort levels;
# original Qwen3 is a thinking toggle only. The dotted-Qwen regex in
# reasoning_effort_hint_for_model enforces generation specificity; needle
# order within this table is immaterial (no needle contains another).
_REASONING_EFFORT_MODEL_HINTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("gpt-oss", frozenset({"low", "medium", "high"})),
    ("qwen3", frozenset({"none"})),
)
# "none" and "high" are live-verified on dotted Qwens: "none" is consumed
# via our enable_thinking=false mapping, and the template aliases "high" to
# "xhigh" — neither must warn as unconsumed.
_QWEN_DOTTED_EFFORT_VALUES = frozenset({"low", "medium", "high", "xhigh", "none"})
# local-llm sends compose llama.cpp-family wire fields (chat_template_kwargs
# reasoning/enable_thinking + reasoning_budget_tokens), so its users need the
# --jinja/b9982 requirements note too.
_LLAMA_CPP_FAMILY_PROVIDERS = frozenset(
    {"llama_cpp", "local_llamacpp", "local_llamafile", "local_llm"}
)
_LLAMACPP_THINKING_REQUIREMENTS_NOTE = (
    "Thinking controls on llama.cpp need llama-server started with --jinja; "
    "per-request reasoning_budget_tokens needs llama.cpp b9982 or newer."
)


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
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip(
        "/"
    )
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
    character_label: str = ""
    #: Optional per-session system prompt prepended as the first provider
    #: message on every native Console send (submit/retry/regenerate/continue).
    #: Defaults to ``None`` (native Console sends no system message unless a
    #: user explicitly sets one for this session) -- it is never seeded from
    #: ``[chat_defaults].system_prompt``.
    system_prompt: str | None = None
    #: Provenance of this snapshot: ``"derived"`` for config-derived defaults
    #: (refreshable when config changes while the session is unused) vs
    #: ``"user"`` for explicit user selections (never auto-replaced).
    source: str = "derived"
    #: Pinned response prefill applied to every submit/retry/regenerate;
    #: persisted per-conversation in conversations.metadata (one-shot
    #: prefill is transient store state, not settings).
    pinned_prefill: str | None = None


@dataclass(frozen=True, slots=True)
class EffectiveChatConfiguration:
    """Canonical provider, model, and endpoint selected for a chat session."""

    provider: str
    model: str | None
    base_url: str | None
    model_source: str


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
    token_limit_verified: bool | None = None
    token_limit_source: str = ""


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


def _summary_row_value(row: str) -> str:
    text = str(row or "").strip()
    _label, separator, value = text.partition(":")
    return value.strip() if separator else text


CONSOLE_MODEL_SECTION_MODEL_MAX_CHARS = 24


def _truncate_model_section_value(
    value: str, limit: int = CONSOLE_MODEL_SECTION_MODEL_MAX_CHARS
) -> str:
    """Truncate a rail model label so the provider/model line stays on one row.

    Long local model names (e.g. ``Qwen3.6-27B-UD-Q4_K_XL.gguf``) word-wrap in
    the narrow left rail; with the rail line clipped to one row the whole model
    token silently disappeared, leaving ``"llama_cpp / "`` on screen.
    """
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip() + "…"


def build_console_model_section_lines(
    summary: ConsoleSettingsSummaryState,
) -> tuple[str, str]:
    """Build the two compact Model rail-section lines from summary rows.

    Args:
        summary: Preformatted Console settings summary rows.

    Returns:
        Tuple of ``(provider/model line, sampling·context·streaming line)``.
    """
    provider = _summary_row_value(summary.provider_row) or "not selected"
    model = _summary_row_value(summary.model_row) or "no model"
    model = _truncate_model_section_value(model)
    sampling = _summary_row_value(summary.sampling_row).partition(",")[0].strip()
    context = _summary_row_value(summary.context_row).partition(";")[0].strip()
    transport = str(summary.transport_row or "").strip()
    detail_parts = [part for part in (sampling, context, transport) if part]
    return f"{provider} / {model}", " · ".join(detail_parts)


CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS = 40
CONSOLE_RAIL_SYSTEM_NONE_LINE = "System: none"


def _collapse_system_prompt_preview_whitespace(text: str) -> str:
    """Collapse a (possibly multi-line) system prompt onto a single rail row."""
    return " ".join(text.split())


def build_console_rail_system_line(system_prompt: str | None) -> str:
    """Build the Model rail-section ``System: <preview>`` line.

    Mirrors ``build_console_model_section_lines``'s long-value handling
    (task-186): the rail line is clipped to one row, so a long or
    multi-line system prompt must be collapsed to a single line AND
    truncated in the text itself -- not left to CSS ``text-overflow:
    ellipsis`` alone -- or it silently word-wraps onto the hidden second
    row. An unset (``None``/blank) system prompt renders the dim
    ``"System: none"`` sentinel line instead.

    Args:
        system_prompt: The session's current system prompt text, or
            ``None``/blank when unset. This is a display-only preview --
            the value is collapsed/truncated here but never mutated in
            storage or in the provider payload.

    Returns:
        ``"System: none"`` when ``system_prompt`` is ``None`` or blank;
        otherwise ``"System: <preview>"`` with the prompt collapsed to a
        single line and truncated to
        ``CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS`` characters.
    """
    normalized = str(system_prompt or "").strip()
    if not normalized:
        return CONSOLE_RAIL_SYSTEM_NONE_LINE
    preview = _truncate_model_section_value(
        _collapse_system_prompt_preview_whitespace(normalized),
        CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS,
    )
    return f"System: {preview}"


def build_console_provider_options(
    providers_models: Mapping[str, Sequence[str]],
) -> list[ConsoleSettingsOption]:
    """Return sorted Console-sendable provider options plus configured providers."""
    provider_keys = sorted(
        {
            key
            for key in (provider_config_key(provider) for provider in providers_models)
            if key
        }
    )
    supported_provider_keys = supported_console_provider_readiness_keys(
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
    )
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
    return [
        ConsoleSettingsOption(
            label=provider_key
            if provider_key in supported_provider_keys
            else f"{provider_key} (WIP)",
            value=provider_key,
        )
        for provider_key in provider_keys
    ]


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
    *,
    excluded_model_profile_fields: frozenset[str] = frozenset(),
) -> ConsoleSessionSettings:
    """Build default Console settings from chat defaults and provider config."""
    chat_defaults = _chat_defaults_with_streaming_compat(
        _mapping_value(app_config, "chat_defaults")
    )
    effective = resolve_effective_chat_configuration(
        app_config,
        provider=provider,
        model=model,
    )
    configured_provider = effective.provider
    provider_settings = _provider_settings(app_config, configured_provider)
    configured_model = effective.model
    model_profile = _model_default_profile(provider_settings, configured_model)
    if excluded_model_profile_fields:
        model_profile = {
            name: value
            for name, value in model_profile.items()
            if name not in excluded_model_profile_fields
        }
    # TASK-342: [console.provider_defaults.<provider>] holds ONLY values the
    # Console's Save-as-default wrote, so it outranks everything except a
    # model profile. chat_defaults stays ahead of raw [api_settings.*]
    # scalars (f14d22dc3, review feedback): factory provider templates carry
    # sampling values for every provider and must not shadow user-tuned
    # global defaults — which is precisely why saved defaults need their own
    # section instead of writing into api_settings.
    saved_defaults = _mapping_value(
        _mapping_value(_mapping_value(app_config, "console"), "provider_defaults"),
        configured_provider,
    )
    default_sources = (model_profile, saved_defaults, chat_defaults, provider_settings)

    return ConsoleSessionSettings(
        provider=configured_provider,
        model=configured_model,
        base_url=effective.base_url,
        temperature=_float_setting_from_sources(default_sources, "temperature", 0.7),
        top_p=_float_setting_from_sources(default_sources, "top_p", 0.95),
        min_p=_optional_float_setting_from_sources(default_sources, "min_p"),
        top_k=_optional_int_setting_from_sources(default_sources, "top_k"),
        max_tokens=_optional_int_setting_from_sources(default_sources, "max_tokens"),
        seed=_optional_int_setting_from_sources(default_sources, "seed"),
        presence_penalty=_optional_float_setting_from_sources(
            default_sources, "presence_penalty"
        ),
        frequency_penalty=_optional_float_setting_from_sources(
            default_sources, "frequency_penalty"
        ),
        reasoning_effort=_optional_string_setting_from_sources(
            default_sources, "reasoning_effort"
        ),
        reasoning_summary=_optional_string_setting_from_sources(
            default_sources, "reasoning_summary"
        ),
        verbosity=_optional_string_setting_from_sources(default_sources, "verbosity"),
        thinking_effort=_optional_string_setting_from_sources(
            default_sources, "thinking_effort"
        ),
        thinking_budget_tokens=_optional_int_setting_from_sources(
            default_sources, "thinking_budget_tokens"
        ),
        streaming=_bool_setting_from_sources(default_sources, "streaming", True),
    )


def default_console_session_settings(
    app_config: Mapping[str, object],
    provider: str | None = None,
    model: str | None = None,
    *,
    excluded_model_profile_fields: frozenset[str] = frozenset(),
) -> ConsoleSessionSettings:
    """The default settings snapshot a NEW Console session starts from.

    `build_default_console_session_settings` plus the one rule that always
    accompanied it at the screen's call site: a llama.cpp session takes no
    `base_url` from configuration (the gateway normalizes the origin at
    send time; a stale configured URL here would pin it).

    Named here, and not left inline in
    `ConsoleSessionController._default_console_session_settings`, because
    task-15860's launch wake builds a session with no screen in existence
    and must start from the same defaults rather than a second spelling of
    them.

    Args:
        app_config: The live app configuration snapshot.
        provider: An explicit provider override (the Console control bar's
            selection when there is a view), or ``None``.
        model: An explicit model override, or ``None``.
        excluded_model_profile_fields: Exact model-profile fields to skip so
            inherited controls can re-resolve lower-precedence defaults.

    Returns:
        The default settings for a new session.
    """
    settings = build_default_console_session_settings(
        app_config,
        provider,
        model,
        excluded_model_profile_fields=excluded_model_profile_fields,
    )
    provider_key = provider_config_key(settings.provider)
    return replace(
        settings,
        base_url=(
            None
            if provider_key in {"llama_cpp", "local_llamacpp"}
            else settings.base_url
        ),
    )


def blank_console_session_settings(
    app_config: Mapping[str, object],
) -> ConsoleSessionSettings:
    """Return config-owned defaults for one eligible blank Console chat."""
    return default_console_session_settings(app_config)


def build_target_default_console_session_settings(
    app_config: Mapping[str, object],
    provider: str,
    model: str | None,
    *,
    excluded_model_profile_fields: frozenset[str] = frozenset(),
) -> ConsoleSessionSettings:
    """Return fresh effective defaults for one provider and literal model ID.

    This is the provider/model rebase entry point. It deliberately delegates to
    :func:`default_console_session_settings` so exact-model profiles keep the
    established precedence and provider endpoint normalization.

    Args:
        app_config: The live application configuration snapshot.
        provider: Provider selected by the settings transaction.
        model: Literal model ID selected by the settings transaction.
        excluded_model_profile_fields: Exact model-profile fields to skip so
            inherited controls can re-resolve lower-precedence defaults.

    Returns:
        A fresh settings value resolved for the exact target.
    """

    return replace(
        default_console_session_settings(
            app_config,
            provider,
            model,
            excluded_model_profile_fields=excluded_model_profile_fields,
        )
    )


def normalized_console_model_profile_overrides(
    app_config: Mapping[str, object],
    provider: str,
    model: str | None,
) -> dict[str, object]:
    """Return valid normalized overrides from one exact model profile.

    Blank and invalid entries are omitted so callers can distinguish inheritance
    from an explicit override. The conversions delegate to the same per-field
    helpers used by :func:`build_default_console_session_settings`; this function
    does not resolve any fallback precedence.
    """

    provider_settings = _provider_settings(
        app_config,
        _canonical_chat_provider_id(provider),
    )
    profile = _model_default_profile(provider_settings, model)
    sources = (profile,)
    overrides: dict[str, object] = {}

    for name in (
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = _optional_float_setting_from_sources(sources, name)
        if value is not None:
            overrides[name] = value
    for name in (
        "top_k",
        "max_tokens",
        "seed",
        "thinking_budget_tokens",
    ):
        value = _optional_int_setting_from_sources(sources, name)
        if value is not None:
            overrides[name] = value
    for name in (
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
    ):
        value = _optional_string_setting_from_sources(sources, name)
        if value is not None:
            overrides[name] = value
    streaming = _bool_setting_from_sources(sources, "streaming", None)
    if streaming is not None:
        overrides["streaming"] = streaming
    return overrides


def resolve_effective_chat_configuration(
    app_config: Mapping[str, object],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> EffectiveChatConfiguration:
    """Resolve canonical chat defaults without mutating loaded configuration."""
    chat_defaults = _chat_defaults_with_streaming_compat(
        _mapping_value(app_config, "chat_defaults")
    )
    provider_id = _canonical_chat_provider_id(
        _string_value(provider) or _string_setting(chat_defaults, "provider")
    )
    provider_settings = _provider_settings(app_config, provider_id)
    candidates = (
        ("session", model),
        ("chat_defaults", chat_defaults.get("model")),
        ("provider_fallback", provider_settings.get("model")),
        ("provider_fallback", provider_settings.get("api_model")),
        ("provider_fallback", provider_settings.get("default_model")),
    )
    model_source = "none"
    resolved_model = None
    for candidate_source, candidate_model in candidates:
        resolved_model = _string_value(candidate_model)
        if resolved_model is not None:
            model_source = candidate_source
            break

    return EffectiveChatConfiguration(
        provider=provider_id,
        model=resolved_model,
        base_url=_default_base_url(provider_id, provider_settings),
        model_source=model_source,
    )


def build_canonical_chat_defaults_mutation(
    effective: EffectiveChatConfiguration,
) -> dict[str, dict[str, str]]:
    """Build the canonical provider/model fragment for an explicit save."""
    chat_defaults: dict[str, str] = {}
    provider_id = _canonical_chat_provider_id(effective.provider)
    model = _string_value(effective.model)
    if provider_id:
        chat_defaults["provider"] = provider_id
    if model:
        chat_defaults["model"] = model
    return {"chat_defaults": chat_defaults}


def reasoning_effort_hint_for_model(model: str | None) -> frozenset[str] | None:
    """Return the effort values this model family's template consumes.

    Args:
        model: Model identifier as selected in the Console (e.g.
            ``"Qwen3.8-27B"``). Case-insensitive; ``None``/blank allowed.

    Returns:
        The set of ``reasoning_effort`` values the model family's chat
        template consumes, or ``None`` when the family is unknown and no
        hint should be shown.
    """
    lowered = str(model or "").strip().lower()
    if not lowered:
        return None
    if re.search(r"qwen3\.\d", lowered):
        return _QWEN_DOTTED_EFFORT_VALUES
    for needle, values in _REASONING_EFFORT_MODEL_HINTS:
        if needle in lowered:
            return values
    return None


def console_settings_warnings(settings: ConsoleSessionSettings) -> list[str]:
    """Return non-blocking warnings for the Console settings modal.

    Warnings never block a save or send (ADR-066); blocking validation
    lives in :func:`validate_console_session_settings`.

    Args:
        settings: The freshly parsed Console session settings.

    Returns:
        Zero or more user-facing warning strings: an effort value the
        selected model family does not consume; for llama.cpp-family
        providers with a thinking value set — the server requirements note
        (``--jinja``; per-request budget needs llama.cpp b9982+); and for
        thinking effort ``off`` on an always-on-thinking Anthropic model
        (Fable 5 / Mythos 5) — that thinking cannot actually be turned off
        there (TASK-18800: the API rejects the explicit disabled config, so
        the request omits the parameter and adaptive thinking still runs).
    """
    warnings: list[str] = []
    effort = str(settings.reasoning_effort or "").strip().lower()
    has_thinking_value = bool(effort) or settings.thinking_budget_tokens is not None
    if effort:
        hint = reasoning_effort_hint_for_model(settings.model)
        if hint is not None and effort not in hint:
            warnings.append(
                f"Reasoning effort '{effort}' is not consumed by this model "
                f"family; expected one of: {', '.join(sorted(hint))}."
            )
    if has_thinking_value and settings.provider in _LLAMA_CPP_FAMILY_PROVIDERS:
        warnings.append(_LLAMACPP_THINKING_REQUIREMENTS_NOTE)
    thinking_effort = str(settings.thinking_effort or "").strip().lower()
    if thinking_effort == "off" and anthropic_model_rejects_disabled_thinking(
        settings.model
    ):
        warnings.append(
            f"{settings.model} always thinks: the API rejects an explicit "
            "thinking-off setting, so 'off' sends no thinking parameter and "
            "adaptive thinking still runs (and is billed)."
        )
    return warnings


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
    if provider_key not in NATIVE_CONSOLE_PROVIDER_KEYS and not _string_value(
        settings.model
    ):
        errors.append("Model is required.")

    base_url = _string_value(settings.base_url)
    if (
        base_url
        and _is_url_based_provider(provider_key, provider_settings)
        and not _valid_base_url(provider_key, base_url)
    ):
        errors.append("Base URL must be a valid http(s) URL.")

    if not _float_in_range(settings.temperature, 0.0, 2.0):
        errors.append("Temperature must be between 0 and 2.")
    if not _float_in_range(settings.top_p, 0.0, 1.0):
        errors.append("Top P must be between 0 and 1.")
    if not _is_blank_value(settings.min_p) and not _float_in_range(
        settings.min_p, 0.0, 1.0
    ):
        errors.append("Min P must be between 0 and 1.")
    if not _is_blank_value(settings.top_k) and not _optional_int_at_least(
        settings.top_k, 0
    ):
        errors.append("Top K must be 0 or greater.")
    if not _is_blank_value(settings.max_tokens) and not _optional_int_at_least(
        settings.max_tokens, 1
    ):
        errors.append("Response max tokens must be 1 or greater.")
    if not _is_blank_value(settings.seed) and not _optional_int_at_least(
        settings.seed, 0
    ):
        errors.append("Seed must be 0 or greater.")
    if not _is_blank_value(settings.presence_penalty) and not _float_in_range(
        settings.presence_penalty, -2.0, 2.0
    ):
        errors.append("Presence penalty must be between -2 and 2.")
    if not _is_blank_value(settings.frequency_penalty) and not _float_in_range(
        settings.frequency_penalty, -2.0, 2.0
    ):
        errors.append("Frequency penalty must be between -2 and 2.")
    if (
        not _is_blank_value(settings.reasoning_effort)
        and settings.reasoning_effort not in _REASONING_EFFORT_VALUES
    ):
        errors.append(
            "Reasoning effort must be one of none, minimal, low, medium, high, or xhigh."
        )
    if (
        not _is_blank_value(settings.reasoning_summary)
        and settings.reasoning_summary not in _REASONING_SUMMARY_VALUES
    ):
        errors.append(
            "Reasoning summary must be one of auto, concise, detailed, or none."
        )
    if (
        not _is_blank_value(settings.verbosity)
        and settings.verbosity not in _VERBOSITY_VALUES
    ):
        errors.append("Verbosity must be one of low, medium, or high.")
    if (
        not _is_blank_value(settings.thinking_effort)
        and settings.thinking_effort not in _THINKING_EFFORT_VALUES
    ):
        errors.append(
            "Thinking effort must be one of off, low, medium, high, xhigh, or max."
        )
    if not _is_blank_value(
        settings.thinking_budget_tokens
    ) and not _optional_int_at_least(settings.thinking_budget_tokens, 1024):
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
    if (
        base_url
        and _is_url_based_provider(provider_key, provider_settings)
        and not _valid_base_url(provider_key, base_url)
    ):
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
        and not identity.uses_direct_llama_path
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

    if not provider_key:
        # An unset provider is not an unsupported one: the template below
        # would interpolate the empty key as "'' is not available in Console
        # yet" (FR-07). Use the canonical no-provider readiness copy instead.
        return ConsoleSettingsReadiness(
            label="Unknown",
            detail=readiness.user_message,
            native_send_supported=False,
        )

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


def _default_supported_readiness_keys() -> frozenset[str]:
    """Return the no-injection supported set, computed once.

    TASK-18909: ``build_console_settings_readiness`` runs ~400 times during
    one warm Console screen switch, and this set is a pure function of a
    module constant -- recomputing it resolved provider identity for all
    29 handler keys on every call (24k resolutions, the largest app-side
    cost of the switch). Cached at first use; ``_supported_readiness_keys``
    retains the test-injection seam uncached.
    """
    return supported_console_provider_readiness_keys(
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )


_default_supported_readiness_keys = functools.cache(_default_supported_readiness_keys)


def _supported_readiness_keys(
    native_provider_keys: set[str] | None = None,
) -> frozenset[str]:
    """Return readiness keys accepted by Console readiness.

    ``native_provider_keys`` is retained for older tests/callers that injected a
    support set before generic Console provider support existed.
    """
    if native_provider_keys is None:
        return _default_supported_readiness_keys()
    supported_keys = _default_supported_readiness_keys()
    injected_keys = frozenset(
        resolve_console_provider_identity(
            provider,
            handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
        ).readiness_key
        for provider in native_provider_keys
    )
    return supported_keys | injected_keys


def _send_capable_readiness_keys(
    native_provider_keys: set[str] | None = None,
) -> frozenset[str]:
    """Return readiness keys that currently have a wired Console send path."""
    if native_provider_keys is None:
        # Same constant inputs as the supported set (see
        # `_default_supported_readiness_keys`); one shared cache serves both.
        return _default_supported_readiness_keys()
    send_capable_keys = _default_supported_readiness_keys()
    injected_keys = frozenset(
        resolve_console_provider_identity(
            provider,
            handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
        ).readiness_key
        for provider in native_provider_keys
    )
    return send_capable_keys | injected_keys


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
    if settings.thinking_budget_tokens is not None:
        sampling_parts.append(f"think budget {settings.thinking_budget_tokens}")
    if settings.reasoning_effort or settings.thinking_budget_tokens is not None:
        identity = resolve_console_provider_identity(settings.provider)
        wire_fields = build_local_thinking_payload_fields(
            identity.execution_key,
            settings.reasoning_effort,
            settings.thinking_budget_tokens,
        )
        if wire_fields:
            wire_parts = []
            template_kwargs = wire_fields.get("chat_template_kwargs")
            if template_kwargs:
                rendered = ", ".join(
                    f"{k}={v}" for k, v in sorted(template_kwargs.items())
                )
                wire_parts.append(f"chat_template_kwargs[{rendered}]")
            if "reasoning_budget_tokens" in wire_fields:
                wire_parts.append(
                    f"reasoning_budget_tokens={wire_fields['reasoning_budget_tokens']}"
                )
            if "reasoning_effort" in wire_fields:
                wire_parts.append(f"reasoning_effort={wire_fields['reasoning_effort']}")
            sampling_parts.append("wire: " + "; ".join(wire_parts))

    character_label = sanitize_character_display_label(
        settings.character_label,
        max_characters=180,
    )
    identity_row = (
        f"Character: {character_label}" if character_label else "Assistant: General"
    )

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
    staged_text: str = "",
    token_counter: TokenCounter | None = None,
    token_limit_resolver: TokenLimitResolver | None = None,
) -> ConsoleSettingsContextEstimate:
    """Estimate current context tokens for display in Console settings.

    Args:
        staged_text: The staged evidence text a send would actually carry
            (task-6) -- e.g. the joined snippets of staged, prompt-eligible
            evidence references. Passed in by the caller so this builder
            stays pure (no I/O, no bundle parsing here); blank/whitespace
            text contributes nothing, matching how an unset staged context
            already behaves. Folded into `used_tokens` as one additional
            message so the context estimate stops silently reporting zero
            for content it is about to send -- `staged_source_count` still
            drives only the label's "; N sources staged" suffix, unchanged.
    """
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
    if staged_text and staged_text.strip():
        estimate_messages.append({"role": "user", "content": staged_text})

    try:
        counter = token_counter or _estimate_tokens_locally
        limit_resolver = token_limit_resolver or _resolve_token_limit_locally
        used_tokens = counter(list(estimate_messages), model_name, provider_key)
        if token_limit_resolver is None:
            token_limit, token_limit_verified, token_limit_source = (
                _resolve_token_limit_locally_with_provenance(
                    model_name,
                    provider_key,
                )
            )
        else:
            token_limit = limit_resolver(model_name, provider_key)
            token_limit_verified = True
            token_limit_source = "provided resolver"
    except Exception:
        return ConsoleSettingsContextEstimate(
            used_tokens=None,
            token_limit=None,
            label="Context: unavailable",
            staged_source_count=staged_source_count,
            staged_context_summary=staged_context_summary,
        )

    label = f"{used_tokens:,} / {token_limit:,} tokens"
    if not token_limit_verified:
        label = f"{label} (estimated; model unverified)"
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
        token_limit_verified=token_limit_verified,
        token_limit_source=token_limit_source,
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


def _canonical_chat_provider_id(provider: str | None) -> str:
    normalized = provider_config_key(provider)
    normalized = _LEGACY_CHAT_PROVIDER_ALIASES.get(normalized, normalized)
    return resolve_console_provider_identity(
        normalized,
        handler_keys=CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    ).readiness_key


def _provider_settings(
    app_config: Mapping[str, object], provider_key: str
) -> Mapping[str, object]:
    api_settings = _mapping_value(app_config, "api_settings")
    try:
        return provider_settings_for_key(api_settings, provider_key)
    except ProviderSettingsError:
        return {}


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


def _has_provider_settings_key(
    app_config: Mapping[str, object], provider_key: str
) -> bool:
    api_settings = _mapping_value(app_config, "api_settings")
    return any(
        provider_config_key(configured_provider) == provider_key
        for configured_provider in api_settings
    )


def _default_base_url(
    provider_key: str, provider_settings: Mapping[str, object]
) -> str | None:
    base_url = _first_string(
        provider_settings.get("api_base_url"),
        provider_settings.get("api_base"),
        provider_settings.get("base_url"),
        provider_settings.get("api_url"),
    )
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        return normalize_llamacpp_base_url(base_url or DEFAULT_LLAMACPP_BASE_URL)
    return base_url


def _is_url_based_provider(
    provider_key: str, provider_settings: Mapping[str, object]
) -> bool:
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
        selected = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(base_url)
        )
        configured = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(configured_endpoint)
        )
        return selected != configured
    return generic_endpoint_differs(base_url, provider_settings)


def _console_endpoint_restart_fallback(
    provider_key: str,
    provider_settings: Mapping[str, object],
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None,
) -> str | None:
    """Return the endpoint the next boot would derive for this provider.

    Mirrors the selection fallback chain in
    ``ChatScreen._build_console_provider_selection_uncached``: llama.cpp
    resolves env override -> ``[console] llama_cpp_base_url_override`` -> the
    provider's configured endpoint -> the built-in default; other URL-based
    providers resolve only their configured endpoint.

    Args:
        provider_key: Normalized provider readiness key.
        provider_settings: The provider's persisted ``api_settings`` section.
        app_config: Application configuration mapping.
        environ: Environment override source; ``None`` reads ``os.environ``.

    Returns:
        The restart fallback endpoint, or ``None`` for URL-based providers
        with nothing configured.
    """
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        env = environ if environ is not None else os.environ
        console_config = _mapping_value(app_config, "console")
        fallback = (
            env.get("TLDW_CONSOLE_LLAMA_CPP_BASE_URL")
            or _string_value(console_config.get("llama_cpp_base_url_override"))
            or first_configured_endpoint(provider_settings)
            or DEFAULT_LLAMACPP_BASE_URL
        )
        return normalize_llamacpp_base_url(fallback)
    return first_configured_endpoint(provider_settings)


def console_session_endpoint_survives_restart(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the session endpoint is backed for the next boot.

    ``True`` when the provider uses no endpoint, the session carries no
    endpoint, or the session endpoint equals the restart fallback chain's
    value (so re-deriving defaults next boot reproduces it). ``False`` means
    the endpoint lives only in this session and is silently lost on restart
    -- the task-16473 persistence trap.

    Args:
        settings: Console session settings carrying the endpoint to check.
        app_config: Application configuration mapping.
        environ: Environment override source; ``None`` reads ``os.environ``.

    Returns:
        Whether re-deriving defaults on the next boot would reproduce the
        session's endpoint.
    """
    provider_key = provider_config_key(settings.provider)
    provider_settings = _provider_settings(app_config, provider_key)
    base_url = _string_value(settings.base_url)
    if not base_url or not _is_url_based_provider(provider_key, provider_settings):
        return True
    fallback = _console_endpoint_restart_fallback(
        provider_key, provider_settings, app_config, environ
    )
    if provider_key in {"llama_cpp", "local_llamacpp"}:
        selected = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(base_url)
        )
        resolved = normalize_generic_endpoint_for_compare(
            normalize_llamacpp_base_url(fallback or DEFAULT_LLAMACPP_BASE_URL)
        )
        return selected == resolved
    if not fallback:
        return False
    return normalize_generic_endpoint_for_compare(
        base_url
    ) == normalize_generic_endpoint_for_compare(fallback)


def unsaved_console_endpoint_warning(
    settings: ConsoleSessionSettings,
    *,
    app_config: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Build the session-only endpoint warning copy, or ``None`` when backed.

    task-16473: llama.cpp readiness reports "Ready" for session-scoped
    endpoints (the direct llama path skips the endpoint-saved check), so
    nothing else tells the user their endpoint evaporates on restart. This is
    the warning that apply surfaces.

    Args:
        settings: Console session settings carrying the endpoint to describe.
        app_config: Application configuration mapping.
        environ: Environment override source; ``None`` reads ``os.environ``.

    Returns:
        User-facing warning copy, or ``None`` when the endpoint is backed for
        the next boot.
    """
    if console_session_endpoint_survives_restart(
        settings, app_config=app_config, environ=environ
    ):
        return None
    display = safe_endpoint_display(settings.base_url) or "the current endpoint"
    return (
        f"Endpoint {display} is saved for this session only and will not "
        "survive a restart. Use Save as default (or Settings) to keep it."
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


@overload
def _bool_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: bool,
) -> bool: ...


@overload
def _bool_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: None,
) -> bool | None: ...


def _bool_setting_from_sources(
    sources: Sequence[Mapping[str, object]],
    key: str,
    default: bool | None,
) -> bool | None:
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
    """Estimate a message list's token count with the real counter.

    Delegates to :func:`count_tokens_messages` (custom tokenizer -> tiktoken
    -> conservative chars floor, never a whitespace word count -- see
    ``Utils/token_counter.py``), which both `model` and `provider` actually
    drive: `model` selects the tokenizer/encoding and the chat-format
    per-message framing convention below; `provider` selects the chars-floor
    ratio table when no tokenizer is installed.

    The previous placeholder's fake `len(messages) * 10` *chars* overhead is
    retired outright, not replaced: `count_tokens_messages` already supplies
    a real per-message allowance (OpenAI's documented chat-format framing --
    3 tokens/message + 3 base for `gpt-3.5`/`gpt-4` models, 2/2 for others),
    the same convention this codebase already uses for dispatch-time history
    budgeting (`console_history_budget.py`) and agent runs
    (`Agents/agent_service.py`) -- so no bespoke overhead is invented here.
    """
    return count_tokens_messages(list(messages), model, provider)


def _resolve_token_limit_locally(model: str, provider: str) -> int:
    """Resolve a local model-window estimate without exposing provenance."""
    limit, _verified, _source = _resolve_token_limit_locally_with_provenance(
        model,
        provider,
    )
    return limit


def _resolve_token_limit_locally_with_provenance(
    model: str,
    provider: str,
) -> tuple[int, bool, str]:
    """Resolve the model window and report whether it is model-specific."""
    if model in CONSOLE_MODEL_TOKEN_LIMITS:
        return CONSOLE_MODEL_TOKEN_LIMITS[model], True, "model catalog"

    model_limits = (
        (prefix, limit)
        for prefix, limit in CONSOLE_MODEL_TOKEN_LIMITS.items()
        if prefix != "default"
    )
    for model_prefix, limit in sorted(
        model_limits, key=lambda item: len(item[0]), reverse=True
    ):
        if model.startswith(model_prefix):
            return limit, True, "model family"

    if provider in CONSOLE_PROVIDER_TOKEN_LIMIT_DEFAULTS:
        return (
            CONSOLE_PROVIDER_TOKEN_LIMIT_DEFAULTS[provider],
            False,
            "provider fallback",
        )
    return CONSOLE_MODEL_TOKEN_LIMITS["default"], False, "application fallback"


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
    return (
        label_text if label_text.startswith("Context: ") else f"Context: {label_text}"
    )


def _format_endpoint_summary_row(settings: ConsoleSessionSettings) -> str:
    endpoint = safe_endpoint_display(settings.base_url)
    return f"Endpoint: {endpoint or 'provider default'}"


def _format_credential_summary_row(readiness: ConsoleSettingsReadiness) -> str:
    label = (_string_value(readiness.label) or "").lower()
    detail = _string_value(readiness.detail) or ""
    detail_lower = detail.lower()
    if label == "missing key" or "missing api key" in detail_lower:
        return "Credential: missing"
    if "no api key is required" in detail_lower:
        return "Credential: not required"
    source_marker = "api key found via "
    source_index = detail_lower.find(source_marker)
    if source_index >= 0:
        source_tail = detail[source_index + len(source_marker) :]
        source_line = source_tail.splitlines()[0] if source_tail else ""
        source = source_line.strip().rstrip(".").strip()
        source_lower = source.lower()
        if source_lower.startswith("env:"):
            env_name = source[len("env:") :].strip()
            return f"Credential: env {env_name}" if env_name else "Credential: env"
        if source_lower.startswith("config:"):
            config_name = source[len("config:") :].strip()
            return (
                f"Credential: config {config_name}"
                if config_name
                else "Credential: config"
            )
        return f"Credential: {source or 'ready'}"
    if "api key found" in detail_lower:
        return "Credential: ready"
    return "Credential: check setup"
