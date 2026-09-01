# console_auxiliary_routing.py
"""TASK-26024: route Console side tasks to a cheaper auxiliary model.

Pure selection helpers only -- the I/O (resolving a provider) stays in the
controller. The Console's one auxiliary LLM call is compaction (titling is
deterministic string truncation), so this routes that call and nothing on
the user-visible send path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any


def auxiliary_selection_from_config(
    main_selection: Any,
    *,
    provider: str | None,
    model: str | None,
) -> Any | None:
    """Build an override selection for a side task, or ``None`` if unconfigured.

    ``None`` (AC#2: no auxiliary configured) tells the caller to keep the
    main selection unchanged. When only ``model`` is set the main provider
    is kept; a cross-provider auxiliary drops the main ``base_url`` so the
    new provider resolves against its own endpoint.
    """
    aux_provider = (provider or "").strip() or None
    aux_model = (model or "").strip() or None
    if aux_provider is None and aux_model is None:
        return None
    effective_provider = aux_provider or main_selection.provider
    changes: dict[str, Any] = {"provider": effective_provider}
    if aux_model is not None:
        changes["explicit_model"] = aux_model
        changes["configured_model"] = aux_model
    if aux_provider is not None and aux_provider != main_selection.provider:
        changes["base_url"] = None
    return replace(main_selection, **changes)


def select_auxiliary_or_main(auxiliary_resolution: Any, main_resolution: Any) -> Any:
    """The auxiliary resolution when ready, else the main (AC#3 fallback)."""
    if auxiliary_resolution is not None and getattr(
        auxiliary_resolution, "ready", False
    ):
        return auxiliary_resolution
    return main_resolution
