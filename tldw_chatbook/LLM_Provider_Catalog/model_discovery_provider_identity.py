"""Provider identity helpers for model discovery persistence."""

from __future__ import annotations

from collections.abc import Mapping

from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    ProviderModelListKeyResolution,
)


def resolve_provider_list_key(
    provider: str | None,
    providers: Mapping[str, object],
) -> ProviderModelListKeyResolution:
    """Resolve provider input to an exact existing top-level provider list key.

    Args:
        provider: User-selected or execution provider name.
        providers: Existing top-level ``[providers]`` mapping.

    Returns:
        Resolution result. A key is resolved only when exactly one existing
        top-level provider key normalizes to the requested Console readiness
        identity; missing and ambiguous states never synthesize a new key.
    """
    requested_provider = (provider or "").strip()
    normalized_provider = resolve_console_provider_identity(provider).readiness_key
    matches = tuple(
        existing_key
        for existing_key in providers
        if provider_config_key(existing_key) == normalized_provider
    )

    if len(matches) == 1:
        return ProviderModelListKeyResolution(
            requested_provider=requested_provider,
            normalized_provider=normalized_provider,
            provider_list_key=matches[0],
            status="resolved",
            matches=matches,
        )

    if not matches:
        return ProviderModelListKeyResolution(
            requested_provider=requested_provider,
            normalized_provider=normalized_provider,
            provider_list_key=None,
            status="missing",
        )

    return ProviderModelListKeyResolution(
        requested_provider=requested_provider,
        normalized_provider=normalized_provider,
        provider_list_key=None,
        status="ambiguous",
        matches=matches,
    )
