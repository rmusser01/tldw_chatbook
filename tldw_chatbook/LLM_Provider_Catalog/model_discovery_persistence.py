"""Explicit persistence helpers for runtime-discovered model IDs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    PersistenceResult,
)
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
    """Return a providers mapping with new model IDs appended to one exact key.

    The list it rewrites also loses the shipped sample ids (TASK-35010): a
    Settings write is when an older file's samples leave it.
    """
    from tldw_chatbook.config import SHIPPED_SAMPLE_MODEL_IDS

    updated = {
        str(provider): _valid_model_list(models)
        for provider, models in providers.items()
    }
    seen_model_ids = set(SHIPPED_SAMPLE_MODEL_IDS)
    existing_models = [
        model
        for model in updated.get(provider_list_key, [])
        if model not in seen_model_ids
    ]
    seen_model_ids.update(existing_models)
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


def refresh_providers_models(app: object, *, force: bool = False) -> None:
    """Make ``app.providers_models`` the config file's saved lists, in place.

    The Default model picker keeps the dict it was composed with, so the
    dict is updated, never rebound (TASK-35010). Call it after a config
    write or reload, never inside a config transaction.

    Args:
        app: The running app.
        force: Republish the config caches from the file first. Config
            reload needs it: an in-place edit keeps the file's identity, so
            the settings cache still matches it.
    """
    from tldw_chatbook import config

    if force:
        config.refresh_runtime_config_from_cli_config()
    fresh = config.get_cli_providers_and_models()
    current = getattr(app, "providers_models", None)
    if isinstance(current, dict):
        current.clear()
        current.update(fresh)
    else:
        app.providers_models = fresh


def _persist_under_config_lock(
    requested_provider: str,
    provider_list_key: str,
    model_ids: Sequence[object],
) -> PersistenceResult:
    """Append to the list the file holds, read under the config write lock.

    One read-modify-write: the list, the ids added and "already saved" all
    come from the locked snapshot, never from the app's startup copy, so a
    model added or removed elsewhere is respected (TASK-35010, C3).
    """
    from tldw_chatbook.config import (
        LiteralSettingsMutation,
        apply_literal_settings_transaction_to_cli_config,
    )

    added: list[str] | None = None

    def build(snapshot) -> LiteralSettingsMutation:
        # TASK-33641.6: pure. Only the snapshot it is handed; no config read
        # or write and no task, which the write lock would refuse.
        nonlocal added
        providers = snapshot.effective_values.get("providers")
        if not isinstance(providers, Mapping) or provider_list_key not in providers:
            return LiteralSettingsMutation(section_values={}, delete_keys={})
        existing = _valid_model_list(providers[provider_list_key])
        updated = append_models_to_provider_list(
            {provider_list_key: existing}, provider_list_key, model_ids
        )[provider_list_key]
        added = [model for model in updated if model not in existing]
        if updated == existing:
            return LiteralSettingsMutation(section_values={}, delete_keys={})
        # D-a: a list that only lost its sample ids is written too.
        return LiteralSettingsMutation(
            section_values={("providers",): {provider_list_key: updated}},
            delete_keys={},
        )

    result = apply_literal_settings_transaction_to_cli_config(build)
    if result.failure_phase is not None:
        return PersistenceResult(
            provider=requested_provider,
            provider_list_key=provider_list_key,
            status="error",
            saved_model_ids=(),
            message="Could not save discovered models to the providers list.",
        )
    if added is None:
        return PersistenceResult(
            provider=requested_provider,
            provider_list_key=None,
            status="missing_provider_key",
            message="No matching provider model list exists in [providers].",
        )
    return PersistenceResult(
        provider=requested_provider,
        provider_list_key=provider_list_key,
        status="saved",
        saved_model_ids=tuple(added),
        message=(
            f"Saved {len(added)} discovered model(s) to {provider_list_key}."
            if added
            else f"No new discovered models to save for {provider_list_key}."
        ),
    )


def persist_discovered_models_to_settings(
    *,
    providers_config: Mapping[str, Sequence[object]],
    requested_provider: str,
    model_ids: Sequence[object],
    save_callback: SaveCallback | None = None,
) -> PersistenceResult:
    """Explicitly append discovered raw model IDs to the existing provider list.

    ``providers_config`` resolves the list key. With no ``save_callback``
    (production), the list itself is read from the file under the config
    write lock; an injected callback keeps the in-memory contract.
    """
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
    if save_callback is None:
        return _persist_under_config_lock(
            requested_provider, provider_list_key, model_ids
        )
    existing_models = providers_config.get(provider_list_key, [])
    saved_model_ids = _newly_saved_model_ids(existing_models, model_ids)
    if not saved_model_ids:
        return PersistenceResult(
            provider=requested_provider,
            provider_list_key=provider_list_key,
            status="saved",
            saved_model_ids=(),
            message=f"No new discovered models to save for {provider_list_key}.",
        )

    updated_providers = append_models_to_provider_list(
        providers_config,
        provider_list_key,
        model_ids,
    )
    try:
        saved = bool(
            save_callback(
                {"providers": {provider_list_key: updated_providers[provider_list_key]}}
            )
        )
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
