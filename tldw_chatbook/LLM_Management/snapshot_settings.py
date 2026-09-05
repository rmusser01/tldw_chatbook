"""Validated configuration boundary for llama.cpp prompt-cache snapshots."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from tldw_chatbook.config import get_cli_setting, save_settings_to_cli_config


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


def save_snapshot_preferences(value: SnapshotPreferences) -> bool:
    """Persist both preferences through one atomic config-owner mutation."""

    return save_settings_to_cli_config({"llamacpp_snapshots": value.model_dump()})
