"""Explicit persistence helpers for runtime-discovered model IDs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import PersistenceResult
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_provider_identity import (
    resolve_provider_list_key,
)

SaveCallback = Callable[[Mapping[str, Mapping[str, list[str]]]], bool]


def _valid_model_id(model_id: object) -> str | None:
    """Return a stripped raw model ID when persistence can accept it."""
    if not isinstance(model_id, str):
        return None
    stripped = model_id.strip()
    return stripped or None


def _valid_model_list(models: object) -> list[str]:
    """Return valid string model IDs from a config model list."""
    if isinstance(models, str) or not isinstance(models, Sequence):
        return []
    return [model for model in models if isinstance(model, str)]


def append_models_to_provider_list(
    providers: Mapping[str, Sequence[object]],
    provider_list_key: str,
    model_ids: Sequence[object],
) -> dict[str, list[str]]:
    """Return a providers mapping with new model IDs appended to one exact key."""
    updated = {
        str(provider): _valid_model_list(models)
        for provider, models in providers.items()
    }
    existing_models = list(updated.get(provider_list_key, []))
    seen_model_ids = set(existing_models)
    for raw_model_id in model_ids:
        model_id = _valid_model_id(raw_model_id)
        if model_id is None or model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        existing_models.append(model_id)
    updated[provider_list_key] = existing_models
    return updated


def _newly_saved_model_ids(
    existing_models: Sequence[object],
    selected_model_ids: Sequence[object],
) -> tuple[str, ...]:
    """Return selected IDs that were not already saved, preserving selected order."""
    existing = {model_id for model_id in existing_models if isinstance(model_id, str)}
    saved: list[str] = []
    seen_selected: set[str] = set()
    for raw_model_id in selected_model_ids:
        model_id = _valid_model_id(raw_model_id)
        if model_id is None or model_id in existing or model_id in seen_selected:
            continue
        seen_selected.add(model_id)
        saved.append(model_id)
    return tuple(saved)


def _default_save_callback(section_values: Mapping[str, Mapping[str, list[str]]]) -> bool:
    """Persist via the shared CLI config writer."""
    from tldw_chatbook.config import save_settings_to_cli_config

    return save_settings_to_cli_config(section_values)


def persist_discovered_models_to_settings(
    *,
    providers_config: Mapping[str, Sequence[object]],
    requested_provider: str,
    model_ids: Sequence[object],
    save_callback: SaveCallback | None = None,
) -> PersistenceResult:
    """Explicitly append discovered raw model IDs to the existing provider list."""
    resolution = resolve_provider_list_key(requested_provider, providers_config)
    if resolution.status == "missing":
        return PersistenceResult(
            provider=requested_provider,
            provider_list_key=None,
            status="missing_provider_key",
            message="No matching provider model list exists in [providers].",
        )
    if resolution.status == "ambiguous":
        return PersistenceResult(
            provider=requested_provider,
            provider_list_key=None,
            status="ambiguous_provider_key",
            message="Multiple provider model lists match this provider. Rename or remove duplicates first.",
        )

    provider_list_key = resolution.provider_list_key or ""
    existing_models = providers_config.get(provider_list_key, [])
    saved_model_ids = _newly_saved_model_ids(existing_models, model_ids)
    updated_providers = append_models_to_provider_list(
        providers_config,
        provider_list_key,
        model_ids,
    )
    save = save_callback or _default_save_callback
    try:
        saved = bool(save({"providers": updated_providers}))
    except Exception:
        saved = False
    if not saved:
        return PersistenceResult(
            provider=requested_provider,
            provider_list_key=provider_list_key,
            status="error",
            saved_model_ids=(),
            message="Could not save discovered models to the providers list.",
        )

    return PersistenceResult(
        provider=requested_provider,
        provider_list_key=provider_list_key,
        status="saved",
        saved_model_ids=saved_model_ids,
        message=f"Saved {len(saved_model_ids)} discovered model(s) to {provider_list_key}.",
    )
