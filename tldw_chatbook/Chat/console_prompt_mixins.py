"""Mix-in request composition with stable public value-model exports."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .console_prompt_mixin_models import (
    LOCAL_COMPILED_LANE_LIMIT,
    LOCAL_DEFINITION_LIMIT,
    LOCAL_REQUEST_LIMIT,
    LiteralMixinDefinition,
    MixinAttemptSnapshot,
    MixinAttemptWrite,
    MixinForkAttempt,
    MixinForkState,
    MixinPlacement,
    MixinPreferenceConflictError,
    MixinPreferenceRead,
    MixinPreferences,
    MixinReplayRead,
    MixinSelection,
    MixinSnapshot,
    MixinSnapshotUnavailableError,
    MixinSourceChangedError,
    MixinSourceRef,
    decode_mixin_preferences,
    encode_mixin_preferences,
)

__all__ = [
    "LOCAL_COMPILED_LANE_LIMIT",
    "LOCAL_DEFINITION_LIMIT",
    "LOCAL_REQUEST_LIMIT",
    "LiteralMixinDefinition",
    "MixinAttemptSnapshot",
    "MixinAttemptWrite",
    "MixinForkAttempt",
    "MixinForkState",
    "MixinPlacement",
    "MixinPreferenceConflictError",
    "MixinPreferenceRead",
    "MixinPreferences",
    "MixinReplayRead",
    "MixinSelection",
    "MixinSnapshot",
    "MixinSnapshotUnavailableError",
    "MixinSourceChangedError",
    "MixinSourceRef",
    "compose_mixin_content",
    "decode_mixin_preferences",
    "encode_mixin_preferences",
    "reconcile_mixin_placements",
]


def reconcile_mixin_placements(
    preferences: MixinPreferences, snapshots: tuple[MixinSnapshot, ...]
) -> MixinPreferences:
    """Refresh observed facts; changed placements move last in the new group."""
    entries = list(preferences.entries)
    for snapshot in snapshots:
        index = next(
            (i for i, row in enumerate(entries) if row.source == snapshot.source), None
        )
        if index is None:
            continue
        prior = entries[index]
        updated = replace(
            prior,
            name=snapshot.name,
            placement=snapshot.definition.placement,
            last_used_version=snapshot.record_version,
        )
        if prior.placement == updated.placement:
            entries[index] = updated
        else:
            entries.pop(index)
            entries.append(updated)
    return replace(
        preferences,
        entries=tuple(
            row
            for placement in ("before", "after")
            for row in entries
            if row.placement == placement
        ),
    )


def compose_mixin_content(
    content: str | list[dict[str, Any]], snapshot: MixinAttemptSnapshot
) -> str | list[dict[str, Any]]:
    """Wrap only supplied content with literal bodies, without interpreting it."""
    if not snapshot.items:
        return content
    before = [
        item.definition.body
        for item in snapshot.items
        if item.definition.placement == "before"
    ]
    after = [
        item.definition.body
        for item in snapshot.items
        if item.definition.placement == "after"
    ]
    if isinstance(content, str):
        return "\n\n".join([*before, content, *after])
    return [
        *({"type": "text", "text": body} for body in before),
        *content,
        *({"type": "text", "text": body} for body in after),
    ]
