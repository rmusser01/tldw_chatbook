"""Shared provider/model resolution for Console and Settings surfaces."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from loguru import logger

from ...Chat.provider_readiness import provider_config_key
from ...LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
    SELECTOR_MERGE_CAP,
)
from ...LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry


MODEL_CAPABILITY_UNKNOWN_WARNING = (
    "Capabilities unknown until saved or verified; text chat is assumed."
)
_ENDPOINT_CATALOG_SOURCES = {"persisted_discovered", "runtime_discovered"}
_ENDPOINT_AUTHORITATIVE_PROVIDER_KEYS = {
    provider_config_key(provider) for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
}


@dataclass(frozen=True)
class EffectiveProviderModel:
    """Resolved provider/model values and the source each value came from."""

    provider: Any
    model: Any
    provider_source: str
    model_source: str


@dataclass(frozen=True)
class ResolvedProviderModelOption:
    """Console model selector option with runtime discovery metadata."""

    label: str
    model_id: str
    source: str
    capability_status: str
    persisted: bool
    warning: str = ""


def _selected_text(value: Any) -> bool:
    """Return whether a provider/model-like value is meaningfully selected."""
    if value is None or value is False:
        return False
    text = str(value).strip()
    return bool(text) and text != "None" and not text.startswith("Select.")


def _saved_models_for_provider(
    providers_models: Mapping[str, Sequence[str]],
    provider: str,
) -> list[str]:
    provider_key = provider_config_key(provider)
    model_ids: list[str] = []
    for configured_provider, configured_models in providers_models.items():
        if provider_config_key(str(configured_provider)) != provider_key:
            continue
        if not isinstance(configured_models, Sequence) or isinstance(
            configured_models, (str, bytes)
        ):
            continue
        for configured_model in configured_models:
            model_id = str(configured_model or "").strip()
            if model_id and model_id not in model_ids:
                model_ids.append(model_id)
    return model_ids


def _warning_for_model(source: str, capability_status: str) -> str:
    if (
        source in {"runtime_discovered", "persisted_discovered"}
        and capability_status == "unknown"
    ):
        return MODEL_CAPABILITY_UNKNOWN_WARNING
    return ""


def _option_from_entry(entry: MergedModelEntry) -> ResolvedProviderModelOption:
    model_id = str(entry.model_id).strip()
    source = str(entry.source)
    capability_status = str(entry.capability_status)
    label = model_id
    if source == "runtime_discovered":
        label = f"{model_id} | runtime discovered"
    if capability_status == "unknown" and source != "saved":
        label = f"{label} | capability unknown"
    return ResolvedProviderModelOption(
        label=label,
        model_id=model_id,
        source=source,
        capability_status=capability_status,
        persisted=bool(entry.persisted),
        warning=_warning_for_model(source, capability_status),
    )


def _option_from_saved_model(model_id: str) -> ResolvedProviderModelOption:
    return ResolvedProviderModelOption(
        label=model_id,
        model_id=model_id,
        source="saved",
        capability_status="known",
        persisted=True,
    )


def _option_from_current_unlisted(model_id: str) -> ResolvedProviderModelOption:
    """Preserve an active model without advertising it as a new choice."""
    return ResolvedProviderModelOption(
        label=f"{model_id} | current, not in latest catalog",
        model_id=model_id,
        source="current_unlisted",
        capability_status="unknown",
        persisted=False,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _merged_model_entries_from_scope(
    catalog_scope_service: Any | None,
    *,
    provider: str,
) -> tuple[MergedModelEntry, ...]:
    merge_models = getattr(
        catalog_scope_service,
        "merge_saved_and_discovered_models",
        None,
    )
    if not callable(merge_models):
        return ()
    try:
        result = await _maybe_await(
            merge_models(
                mode="local",
                provider=provider,
            )
        )
    except ValueError:
        # FB-09 (TASK-2154.18): the local catalog legitimately does not
        # cover every provider -- a cloud-only provider or an empty local
        # catalog makes the local service raise "Unknown or ambiguous
        # local LLM provider". The merge is best-effort enrichment over
        # the saved list, so degrade to saved-only quietly instead of
        # tracebacking through the Alt+M popover path (where the caller
        # logs logger.exception) or the model search picker (which has no
        # exception handler at all). Other exception types still
        # propagate to those callers.
        logger.debug(
            "Local model catalog has no entry for provider={}; saved models only",
            provider,
        )
        return ()
    return tuple(entry for entry in result if isinstance(entry, MergedModelEntry))


async def _has_endpoint_snapshot_from_scope(
    catalog_scope_service: Any | None,
    *,
    provider: str,
) -> bool | None:
    """Return snapshot presence, or None when the scope lacks that contract."""
    has_snapshot = getattr(
        catalog_scope_service,
        "has_discovered_model_snapshot",
        None,
    )
    if not callable(has_snapshot):
        return None
    try:
        return bool(
            await _maybe_await(
                has_snapshot(
                    mode="local",
                    provider=provider,
                )
            )
        )
    except ValueError:
        return False


async def resolve_provider_model_options(
    providers_models: Mapping[str, Sequence[str]],
    catalog_scope_service: Any | None,
    *,
    provider: str,
    current_model: str | None = None,
    merge_cap: int | None = SELECTOR_MERGE_CAP,
) -> list[ResolvedProviderModelOption]:
    """Return endpoint-authoritative or saved fallback selector options.

    Auto-refreshed cloud providers use their current endpoint snapshot as the
    choice authority (ADR-020). Other providers remain additive. Pass
    ``merge_cap=None`` for the uncapped search picker.
    """
    if not isinstance(providers_models, Mapping):
        raise TypeError("providers_models must be a mapping")
    provider_key = provider_config_key(provider)
    merged_entries = await _merged_model_entries_from_scope(
        catalog_scope_service,
        provider=provider_key,
    )
    has_endpoint_entries = any(
        str(entry.source) in _ENDPOINT_CATALOG_SOURCES for entry in merged_entries
    )
    snapshot_present = None
    if (
        provider_key in _ENDPOINT_AUTHORITATIVE_PROVIDER_KEYS
        and not has_endpoint_entries
    ):
        snapshot_present = await _has_endpoint_snapshot_from_scope(
            catalog_scope_service,
            provider=provider_key,
        )
    saved_models = _saved_models_for_provider(providers_models, provider_key)
    merged_by_model_id: dict[str, MergedModelEntry] = {}
    for entry in merged_entries:
        model_id = str(entry.model_id).strip()
        if model_id and model_id not in merged_by_model_id:
            merged_by_model_id[model_id] = entry
    ordered_entries: list[MergedModelEntry] = []
    for model_id in saved_models:
        entry = merged_by_model_id.pop(model_id, None)
        if entry is None:
            entry = MergedModelEntry(
                provider=provider_key,
                provider_list_key=provider_key,
                model_id=model_id,
                display_name=model_id,
                source="saved",
                capability_status="known",
                persisted=True,
            )
        ordered_entries.append(entry)
    for entry in merged_entries:
        model_id = str(entry.model_id).strip()
        if model_id not in merged_by_model_id:
            continue
        ordered_entries.append(entry)
        merged_by_model_id.pop(model_id, None)
    merged_entries = tuple(ordered_entries)
    has_endpoint_catalog = (
        provider_key in _ENDPOINT_AUTHORITATIVE_PROVIDER_KEYS
        and (snapshot_present is True or has_endpoint_entries)
    )
    endpoint_entries = tuple(
        entry
        for entry in merged_entries
        if str(entry.source) in _ENDPOINT_CATALOG_SOURCES
    )
    endpoint_catalog_over_cap = (
        merge_cap is not None and len(endpoint_entries) > merge_cap
    )
    options: list[ResolvedProviderModelOption] = []
    seen_model_ids: set[str] = set()
    for entry in merged_entries:
        source = str(entry.source)
        if has_endpoint_catalog and source == "saved":
            continue
        if endpoint_catalog_over_cap and source == "runtime_discovered":
            continue
        option = _option_from_entry(entry)
        if option.model_id and option.model_id not in seen_model_ids:
            options.append(option)
            seen_model_ids.add(option.model_id)

    current_model_id = str(current_model or "").strip()
    if current_model_id and current_model_id not in seen_model_ids:
        current_option = (
            _option_from_current_unlisted(current_model_id)
            if has_endpoint_catalog
            else _option_from_saved_model(current_model_id)
        )
        options.insert(0, current_option)
    return options


def resolve_effective_provider_model(
    persisted_defaults: Mapping[str, Any],
    *,
    console_provider: Any = None,
    console_model: Any = None,
    settings_provider: Any = None,
    settings_model: Any = None,
) -> EffectiveProviderModel:
    """Resolve the canonical provider/model pair for Console-adjacent UI.

    Args:
        persisted_defaults: Persisted ``chat_defaults`` configuration mapping.
        console_provider: Provider selected by the Console control surface.
        console_model: Model selected by the Console control surface.
        settings_provider: Provider staged in Settings before save.
        settings_model: Model staged in Settings before save.

    Returns:
        Resolved provider/model values plus labels naming each selected source.

    Settings drafts win because they are what the user is evaluating before save.
    Console controls win next because they are the active run surface.
    """
    if not isinstance(persisted_defaults, Mapping):
        raise TypeError("persisted_defaults must be a mapping")
    configured_provider = persisted_defaults.get("provider")

    if _selected_text(settings_provider):
        provider = settings_provider
        provider_source = "settings_draft"
    elif _selected_text(console_provider):
        provider = console_provider
        provider_source = "console_session"
    else:
        provider = configured_provider
        provider_source = "chat_defaults"

    configured_model = persisted_defaults.get("model")

    if _selected_text(settings_model):
        model = settings_model
        model_source = "settings_draft"
    elif _selected_text(console_model):
        model = console_model
        model_source = "console_session"
    else:
        model = configured_model
        model_source = "chat_defaults"

    return EffectiveProviderModel(
        provider=provider,
        model=model,
        provider_source=provider_source,
        model_source=model_source,
    )
