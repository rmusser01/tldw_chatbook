"""Effective workspace assistant default resolution (server reason-code contract)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .models import WorkspaceAssistantDefaults

DEGRADED_REASONS = (
    "persona_deleted",
    "persona_unavailable",
    "persona_feature_disabled",
    "permission_denied",
    "invalid_default",
    "unsupported_assistant_kind",
)


@dataclass(frozen=True)
class WorkspaceEffectiveAssistantDefault:
    status: str  # "available" | "unavailable" | "none"
    source: str  # "workspace" | "none"
    assistant_kind: str | None = None
    assistant_id: str | None = None
    label: str | None = None
    persona_memory_mode: str | None = None
    degraded_reason: str | None = None


_NONE = WorkspaceEffectiveAssistantDefault(status="none", source="none")


def resolve_effective_assistant_default(
    defaults: WorkspaceAssistantDefaults | None,
    persona_lookup: Callable[[str], Mapping | None],
) -> WorkspaceEffectiveAssistantDefault:
    if defaults is None:
        return _NONE
    if defaults.assistant_kind != "persona":
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="unsupported_assistant_kind"
        )
    record = persona_lookup(defaults.assistant_id)
    if record is None or record.get("deleted"):
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="persona_deleted"
        )
    if not isinstance(record, Mapping) or not str(record.get("id") or ""):
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="persona_unavailable"
        )
    return WorkspaceEffectiveAssistantDefault(
        status="available",
        source="workspace",
        assistant_kind="persona",
        assistant_id=defaults.assistant_id,
        label=str(record.get("name") or "") or None,
        persona_memory_mode=defaults.persona_memory_mode,
    )
