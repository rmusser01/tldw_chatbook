"""Console provider identity helpers."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from tldw_chatbook.Chat.provider_readiness import provider_config_key


DIRECT_CONSOLE_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})

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
    """Resolved Console provider identities for config, readiness, and send."""

    display_key: str
    readiness_key: str
    execution_key: str
    is_supported: bool
    uses_direct_llama_path: bool = False


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
    """Resolve Console provider display, readiness, and execution keys."""
    raw_provider = (provider or "").strip()
    display_key = provider_config_key(raw_provider)
    exact_key = raw_provider.lower()

    if exact_key in DIRECT_CONSOLE_PROVIDER_KEYS or display_key in DIRECT_CONSOLE_PROVIDER_KEYS:
        direct_key = exact_key if exact_key in DIRECT_CONSOLE_PROVIDER_KEYS else display_key
        return ConsoleProviderIdentity(
            display_key=direct_key,
            readiness_key=direct_key,
            execution_key=direct_key,
            is_supported=True,
            uses_direct_llama_path=True,
        )

    handlers = _handler_keys(handler_keys)
    readiness_key = _EXECUTION_TO_READINESS_ALIASES.get(exact_key, display_key)
    execution_key = _READINESS_TO_EXECUTION_ALIASES.get(readiness_key)
    if execution_key is None:
        execution_key = exact_key if exact_key in handlers else readiness_key

    return ConsoleProviderIdentity(
        display_key=display_key,
        readiness_key=readiness_key,
        execution_key=execution_key,
        is_supported=execution_key in handlers,
        uses_direct_llama_path=False,
    )


def supported_console_provider_readiness_keys(
    handler_keys: Collection[str] | None = None,
) -> frozenset[str]:
    """Return readiness keys supported by Console provider execution."""
    handlers = _handler_keys(handler_keys)
    return frozenset(
        resolve_console_provider_identity(
            handler_key,
            handler_keys=handlers,
        ).readiness_key
        for handler_key in handlers
    )
