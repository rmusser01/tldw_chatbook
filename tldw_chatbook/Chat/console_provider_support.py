"""Console provider identity helpers."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from tldw_chatbook.Chat.provider_readiness import (
    PROVIDERS_REQUIRING_API_KEY_KEYS,
    provider_config_key,
)
from tldw_chatbook.model_capabilities import (
    anthropic_model_rejects_fixed_thinking_budget,
    moonshot_model_supports_reasoning_effort,
    zai_model_supports_reasoning_effort,
)


DIRECT_CONSOLE_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})

ConsoleGenerationControl = Literal[
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
    "thinking_effort",
    "thinking_budget_tokens",
]
ConsoleControlSupport = Literal["supported", "unsupported", "unknown"]

_READINESS_TO_EXECUTION_ALIASES = {
    "custom": "custom-openai-api",
    "custom_2": "custom-openai-api-2",
    "local_llm": "local-llm",
    "local_mlx_lm": "local_mlx_lm",
    "mistralai": "mistralai",
}

_EXECUTION_TO_READINESS_ALIASES = {
    "custom-openai-api": "custom",
    "custom-openai-api-2": "custom_2",
    "local-llm": "local_llm",
    "mlx_lm": "local_mlx_lm",
}


@dataclass(frozen=True)
class ConsoleProviderIdentity:
    """Resolved Console provider identities for config, readiness, and send.

    Attributes:
        display_key: Normalized provider key used by Console controls.
        readiness_key: Provider key used for configuration/readiness lookup.
        execution_key: Provider key passed to ``chat_api_call``.
        is_supported: Whether Console can send through this provider.
        uses_direct_llama_path: Whether the provider bypasses the generic
            adapter and uses the direct llama.cpp path.
    """

    display_key: str
    readiness_key: str
    execution_key: str
    is_supported: bool
    uses_direct_llama_path: bool = False


@dataclass(frozen=True)
class ConsoleProviderCatalogEntry:
    """Provider option Settings can display for Console-compatible sends."""

    readiness_key: str
    execution_key: str
    display_name: str
    requires_api_key: bool
    uses_direct_llama_path: bool = False


_PROVIDER_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "cohere": "Cohere",
    "custom": "Custom OpenAI",
    "custom_2": "Custom OpenAI 2",
    "deepseek": "DeepSeek",
    "google": "Google",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "llama_cpp": "llama.cpp",
    "local_llamacpp": "local llama.cpp",
    "local_mlx_lm": "MLX LM",
    "local_vllm": "local vLLM",
    "mistral": "Mistral",
    "mistralai": "MistralAI",
    "moonshot": "Moonshot",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "qwencloud": "QwenCloud",
    "vllm": "vLLM",
    "zai": "Z.ai",
}


def _provider_display_name(provider_key: str) -> str:
    """Return a compact human-readable label for a provider key."""
    return _PROVIDER_DISPLAY_NAMES.get(
        provider_key,
        provider_key.replace("_", " ").replace("-", " ").title(),
    )


# ADR-066: per-execution-key wire formats for Console thinking controls.
# Level = reasoning_effort; budget = thinking_budget_tokens.
_LLAMA_CPP_THINKING_KEYS = frozenset(
    {"llama_cpp", "local_llamacpp", "local_llamafile", "local-llm"}
)
_VLLM_THINKING_KEYS = frozenset({"vllm", "local_vllm"})
_CUSTOM_OPENAI_THINKING_KEYS = frozenset({"custom-openai-api", "custom-openai-api-2"})
# MLX-LM: template-kwargs shape pending live verification of mlx_lm.server
# support; if unsupported this row degrades to drop-and-log.
_TEMPLATE_KWARGS_THINKING_KEYS = frozenset({"local_mlx_lm"})
# Live-verified (llama.cpp b10430 + Qwen3.8): strict chat templates such as
# Qwen3.8's validate reasoning_effort and raise on unknown values ("minimal"
# -> HTTP 500). "high" is aliased to "xhigh" by the template and is safe;
# "none" is safe because we pair it with enable_thinking=false which
# short-circuits the template's validation block.
_TEMPLATE_SAFE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "none"})

_LOCAL_REASONING_EXECUTION_KEYS = (
    _LLAMA_CPP_THINKING_KEYS
    | _VLLM_THINKING_KEYS
    | _CUSTOM_OPENAI_THINKING_KEYS
    | _TEMPLATE_KWARGS_THINKING_KEYS
)
_LOCAL_BUDGET_EXECUTION_KEYS = _LLAMA_CPP_THINKING_KEYS
_LOCAL_DROPPED_CONTROLS = frozenset(
    {"reasoning_summary", "verbosity", "thinking_effort"}
)


def console_generation_control_support(
    provider: str,
    model: str | None,
    control: ConsoleGenerationControl,
) -> ConsoleControlSupport:
    """Return existing authoritative support for one generation control.

    The answer describes whether the current Console send path and known model
    family consume the control. An unrecognised model stays ``unknown`` rather
    than inheriting a negative result from a capability predicate whose false
    value also covers names outside that predicate's domain.
    """
    identity = resolve_console_provider_identity(provider)
    execution_key = identity.execution_key
    readiness_key = identity.readiness_key

    if execution_key in _LOCAL_REASONING_EXECUTION_KEYS:
        if control in _LOCAL_DROPPED_CONTROLS:
            return "unsupported"
        if control == "thinking_budget_tokens":
            return (
                "supported"
                if execution_key in _LOCAL_BUDGET_EXECUTION_KEYS
                else "unsupported"
            )
        if control == "reasoning_effort":
            # A custom OpenAI-compatible server accepts an arbitrary model and
            # does not provide authoritative model capability metadata.
            if execution_key in (
                _CUSTOM_OPENAI_THINKING_KEYS | _TEMPLATE_KWARGS_THINKING_KEYS
            ):
                return "unknown"
            return "supported"

    if identity.is_supported:
        from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP

        provider_params = PROVIDER_PARAM_MAP.get(execution_key)
        if provider_params is not None and control not in provider_params:
            return "unsupported"

    if readiness_key == "anthropic":
        if anthropic_model_rejects_fixed_thinking_budget(model):
            return "unsupported" if control == "thinking_budget_tokens" else "supported"
        return "unknown"

    if readiness_key == "moonshot":
        return (
            "supported"
            if moonshot_model_supports_reasoning_effort(model)
            else "unknown"
        )

    if readiness_key == "zai":
        return "supported" if zai_model_supports_reasoning_effort(model) else "unknown"

    return "unknown"


def build_local_thinking_payload_fields(
    execution_key: str | None,
    reasoning_effort: str | None,
    thinking_budget_tokens: int | None,
) -> dict[str, Any]:
    """Compose thinking-control payload fragments for a local provider.

    Args:
        execution_key: ``chat_api_call`` provider key (e.g. ``llama_cpp``).
        reasoning_effort: Verbatim user-selected effort level, if any.
        thinking_budget_tokens: Max thinking tokens, if any.

    Returns:
        Fragments to merge into an OpenAI-compatible chat payload. Empty
        dict when the key has no thinking support or no values are set.
    """
    key = str(execution_key or "").strip().lower()
    effort = str(reasoning_effort or "").strip().lower() or None
    budget: int | None = (
        thinking_budget_tokens
        if isinstance(thinking_budget_tokens, int)
        and not isinstance(thinking_budget_tokens, bool)
        else None
    )
    fields: dict[str, Any] = {}
    if key in _LLAMA_CPP_THINKING_KEYS or key in _TEMPLATE_KWARGS_THINKING_KEYS:
        if effort is not None:
            if effort in _TEMPLATE_SAFE_EFFORTS:
                template_kwargs: dict[str, Any] = {"reasoning_effort": effort}
                if effort == "none":
                    template_kwargs["enable_thinking"] = False
                fields["chat_template_kwargs"] = template_kwargs
            else:
                logger.debug(
                    "reasoning effort '{}' is not consumable by strict chat "
                    "templates; dropped from chat_template_kwargs",
                    effort,
                )
        if budget is not None and key in _LLAMA_CPP_THINKING_KEYS:
            fields["reasoning_budget_tokens"] = budget
        if budget is not None and key in _TEMPLATE_KWARGS_THINKING_KEYS:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    elif key in _VLLM_THINKING_KEYS:
        if effort is not None:
            fields["reasoning_effort"] = effort
            if effort in _TEMPLATE_SAFE_EFFORTS:
                fields["chat_template_kwargs"] = {"reasoning_effort": effort}
            else:
                logger.debug(
                    "reasoning effort '{}' is not consumable by strict chat "
                    "templates; dropped from chat_template_kwargs",
                    effort,
                )
        if budget is not None:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    elif key in _CUSTOM_OPENAI_THINKING_KEYS:
        if effort is not None:
            fields["reasoning_effort"] = effort
        if budget is not None:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    return fields


def _handler_keys(handler_keys: Collection[str] | None = None) -> frozenset[str]:
    """Return supported ``chat_api_call`` execution keys."""
    if handler_keys is not None:
        return frozenset(handler_keys)

    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    return frozenset(API_CALL_HANDLERS)


def resolve_console_provider_identity(
    provider: str | None,
    *,
    handler_keys: Collection[str] | None = None,
) -> ConsoleProviderIdentity:
    """Resolve Console provider display, readiness, and execution keys.

    Args:
        provider: Raw provider name from config or Console controls.
        handler_keys: Optional ``chat_api_call`` handler keys for deterministic
            tests or side-effect-free callers.

    Returns:
        Resolved provider identity describing display, readiness, and execution
        keys plus whether the provider is supported.
    """
    raw_provider = (provider or "").strip()
    display_key = provider_config_key(raw_provider)
    exact_key = raw_provider.lower()

    if (
        exact_key in DIRECT_CONSOLE_PROVIDER_KEYS
        or display_key in DIRECT_CONSOLE_PROVIDER_KEYS
    ):
        direct_key = (
            exact_key if exact_key in DIRECT_CONSOLE_PROVIDER_KEYS else display_key
        )
        return ConsoleProviderIdentity(
            display_key=direct_key,
            readiness_key=direct_key,
            execution_key=direct_key,
            is_supported=True,
            uses_direct_llama_path=True,
        )

    handlers = _handler_keys(handler_keys)
    normalized_handler_keys = {
        provider_config_key(handler_key): handler_key for handler_key in handlers
    }
    handler_exact_key = (
        exact_key
        if exact_key in handlers
        else normalized_handler_keys.get(display_key, exact_key)
    )
    readiness_key = _EXECUTION_TO_READINESS_ALIASES.get(handler_exact_key, display_key)
    execution_key = _READINESS_TO_EXECUTION_ALIASES.get(readiness_key)
    if execution_key is None:
        execution_key = (
            handler_exact_key if handler_exact_key in handlers else readiness_key
        )

    return ConsoleProviderIdentity(
        display_key=display_key,
        readiness_key=readiness_key,
        execution_key=execution_key,
        is_supported=execution_key in handlers,
        uses_direct_llama_path=False,
    )


def supported_console_provider_catalog(
    handler_keys: Collection[str] | None = None,
) -> tuple[ConsoleProviderCatalogEntry, ...]:
    """Return Console-sendable provider catalog entries for Settings.

    Args:
        handler_keys: Optional ``chat_api_call`` handler keys for deterministic
            tests or side-effect-free callers.

    Returns:
        Stable, de-duplicated provider entries keyed by readiness/config key.
    """
    handlers = _handler_keys(handler_keys)
    entries: dict[str, ConsoleProviderCatalogEntry] = {}
    for handler_key in sorted(handlers):
        identity = resolve_console_provider_identity(
            handler_key,
            handler_keys=handlers,
        )
        if not identity.is_supported:
            continue
        entries.setdefault(
            identity.readiness_key,
            ConsoleProviderCatalogEntry(
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
                display_name=_provider_display_name(identity.readiness_key),
                requires_api_key=identity.readiness_key
                in PROVIDERS_REQUIRING_API_KEY_KEYS,
                uses_direct_llama_path=identity.uses_direct_llama_path,
            ),
        )
    return tuple(sorted(entries.values(), key=lambda entry: entry.readiness_key))


def supported_console_provider_readiness_keys(
    handler_keys: Collection[str] | None = None,
) -> frozenset[str]:
    """Return readiness keys supported by Console provider execution.

    Args:
        handler_keys: Optional ``chat_api_call`` handler keys for deterministic
            tests or side-effect-free callers.

    Returns:
        Set of normalized readiness keys whose providers can be sent from
        Console.
    """
    handlers = _handler_keys(handler_keys)
    return frozenset(
        resolve_console_provider_identity(
            handler_key,
            handler_keys=handlers,
        ).readiness_key
        for handler_key in handlers
    )
