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
    """Return resolver capabilities that prove a model is explicitly known."""
    if capability_resolver is None:
        return None
    try:
        capabilities = capability_resolver(provider, model_id)
    except Exception:
        return None
    if not isinstance(capabilities, Mapping) or not capabilities:
        return None
    if capabilities.get("vision") is True:
        return capabilities
    if capabilities.get("known") is True or capabilities.get("configured") is True:
        return capabilities
    return None


_IMAGE_MODALITIES = frozenset({"image", "images", "vision", "visual"})


def _lists_image(modalities: object) -> bool:
    """Return whether a listing's modality list names images."""
    if isinstance(modalities, str) or not isinstance(modalities, Sequence):
        return False
    return bool(
        {str(modality).strip().lower() for modality in modalities} & _IMAGE_MODALITIES
    )


def _metadata_has_positive_capability(metadata: Mapping[str, Any]) -> bool:
    """Return whether safe discovery metadata suggests positive capabilities.

    Listings say "takes images" in several spellings, and any one of them is
    enough (TASK-34402):

    - a ``vision`` or ``supports_image_input`` flag (the latter is
      Fireworks'), or a ``vision`` flag nested under ``capabilities`` (the
      shape Mistral documents; no capture of it is in this repo);
    - an ``input_modalities`` list, top-level or nested under
      ``architecture`` (OpenRouter);
    - ``modalities``, as Vercel's ``{"input": [...], "output": [...]}``
      mapping, of which only the input side counts, or as a flat list.

    A flat ``modalities`` list does not say input or output, so an entry in
    it is read as input, as it always was. Wherever a listing does tell the
    two apart, a model that outputs images but takes text is not
    vision-capable.
    """
    if metadata.get("vision") is True or metadata.get("supports_image_input") is True:
        return True
    capabilities = metadata.get("capabilities")
    if isinstance(capabilities, Mapping) and capabilities.get("vision") is True:
        return True
    modalities = metadata.get("modalities")
    if isinstance(modalities, Mapping):
        modalities = modalities.get("input")
    architecture = metadata.get("architecture")
    nested = (
        architecture.get("input_modalities")
        if isinstance(architecture, Mapping)
        else None
    )
    return any(
        _lists_image(hint)
        for hint in (modalities, metadata.get("input_modalities"), nested)
    )


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
    """Merge saved IDs first while preserving endpoint-confirmed provenance."""
    merged: list[MergedModelEntry] = []
    seen_model_ids: set[str] = set()
    discovered_by_id = {
        model_id: discovered_model
        for discovered_model in discovered_models
        if (model_id := _non_empty_model_id(discovered_model.model_id)) is not None
    }
    for raw_model_id in saved_model_ids:
        model_id = _non_empty_model_id(raw_model_id)
        if model_id is None or model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        discovered_model = discovered_by_id.get(model_id)
        if discovered_model is not None:
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
                    source="persisted_discovered",
                    capability_status=capability_status,
                    persisted=True,
                )
            )
            continue
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
