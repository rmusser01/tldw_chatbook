"""Merge saved and runtime-discovered model selector entries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    CapabilityStatus,
    DiscoveredModel,
    MergedModelEntry,
)

CapabilityResolver = Callable[[str, str], Mapping[str, Any] | None]


def _non_empty_model_id(model_id: object) -> str | None:
    """Return a stripped raw model ID if it is a non-empty string."""
    if not isinstance(model_id, str):
        return None
    stripped = model_id.strip()
    return stripped or None


def _capabilities_from_resolver(
    provider: str,
    model_id: str,
    capability_resolver: CapabilityResolver | None,
) -> Mapping[str, Any] | None:
    """Return resolver capabilities while treating resolver errors as unknown."""
    if capability_resolver is None:
        return None
    try:
        capabilities = capability_resolver(provider, model_id)
    except Exception:
        return None
    return capabilities if isinstance(capabilities, Mapping) and capabilities else None


def _metadata_has_positive_capability(metadata: Mapping[str, Any]) -> bool:
    """Return whether safe discovery metadata suggests positive capabilities."""
    if metadata.get("vision") is True:
        return True
    modalities = metadata.get("modalities")
    if isinstance(modalities, Sequence) and not isinstance(modalities, str):
        normalized = {str(modality).strip().lower() for modality in modalities}
        return bool(normalized & {"image", "images", "vision", "visual"})
    input_modalities = metadata.get("input_modalities")
    if isinstance(input_modalities, Sequence) and not isinstance(input_modalities, str):
        normalized = {str(modality).strip().lower() for modality in input_modalities}
        return bool(normalized & {"image", "images", "vision", "visual"})
    return False


def resolve_discovered_model_capability_status(
    provider: str,
    model_id: str,
    metadata_raw_safe: Mapping[str, Any] | None,
    capability_resolver: CapabilityResolver | None = None,
) -> CapabilityStatus:
    """Resolve known, inferred, or unknown capability status for a model."""
    if _capabilities_from_resolver(provider, model_id, capability_resolver) is not None:
        return "known"
    if isinstance(metadata_raw_safe, Mapping) and _metadata_has_positive_capability(
        metadata_raw_safe
    ):
        return "inferred"
    return "unknown"


def merge_saved_and_discovered_models(
    *,
    saved_model_ids: Sequence[object],
    discovered_models: Sequence[DiscoveredModel],
    provider: str,
    provider_list_key: str,
    capability_resolver: CapabilityResolver | None = None,
) -> tuple[MergedModelEntry, ...]:
    """Merge saved model IDs first, followed by new runtime-discovered models."""
    merged: list[MergedModelEntry] = []
    seen_model_ids: set[str] = set()
    for raw_model_id in saved_model_ids:
        model_id = _non_empty_model_id(raw_model_id)
        if model_id is None or model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        merged.append(
            MergedModelEntry(
                provider=provider,
                provider_list_key=provider_list_key,
                model_id=model_id,
                display_name=model_id,
                source="saved",
                capability_status=resolve_discovered_model_capability_status(
                    provider,
                    model_id,
                    {},
                    capability_resolver=capability_resolver,
                ),
                persisted=True,
            )
        )

    for discovered_model in discovered_models:
        model_id = _non_empty_model_id(discovered_model.model_id)
        if model_id is None or model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        capability_status = discovered_model.capability_status
        if capability_status == "unknown":
            capability_status = resolve_discovered_model_capability_status(
                discovered_model.provider,
                model_id,
                discovered_model.metadata_raw_safe,
                capability_resolver=capability_resolver,
            )
        merged.append(
            MergedModelEntry(
                provider=discovered_model.provider,
                provider_list_key=discovered_model.provider_list_key,
                model_id=model_id,
                display_name=discovered_model.display_name,
                source=discovered_model.source,
                capability_status=capability_status,
                persisted=discovered_model.persisted,
            )
        )
    return tuple(merged)
