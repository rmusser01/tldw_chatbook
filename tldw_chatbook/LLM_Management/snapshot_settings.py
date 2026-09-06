"""Validated configuration boundary for llama.cpp prompt-cache snapshots."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from tldw_chatbook.config import (
    apply_settings_mutation_to_cli_config,
    get_cli_setting,
    save_settings_to_cli_config,
)


class SnapshotPreferences(BaseModel):
    """Persisted user preferences for manual prompt-cache snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: StrictBool = False
    keep_count: Annotated[int, Field(strict=True, ge=1, le=1000)] = 10


def load_snapshot_preferences() -> SnapshotPreferences:
    """Load and strictly validate the effective snapshot preferences."""

    return SnapshotPreferences(
        enabled=get_cli_setting("llamacpp_snapshots", "enabled", False),
        keep_count=get_cli_setting("llamacpp_snapshots", "keep_count", 10),
    )


class SnapshotPreferencesConflict(ValueError):
    """The saved preference pair changed after this draft was loaded."""


def save_snapshot_preferences(
    value: SnapshotPreferences, *, expected: SnapshotPreferences | None = None
) -> bool:
    """Persist both preferences through one atomic config-owner mutation."""

    sections = {"llamacpp_snapshots": value.model_dump()}
    if expected is None:
        return save_settings_to_cli_config(sections)

    def unchanged(snapshot):
        values = snapshot.values.get("llamacpp_snapshots", {})
        return (
            SnapshotPreferences(
                enabled=values.get("enabled", False),
                keep_count=values.get("keep_count", 10),
            )
            == expected
        )

    result = apply_settings_mutation_to_cli_config(
        sections, locked_snapshot_precondition=unchanged
    )
    if result.conflict:
        raise SnapshotPreferencesConflict(
            "Snapshot preferences changed; reload the draft."
        )
    return result.fully_applied or (
        result.failure_phase is None and not result.file_replaced
    )
