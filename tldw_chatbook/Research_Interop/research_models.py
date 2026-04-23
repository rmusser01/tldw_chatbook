"""Normalized Research Sessions models shared by local and server adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import sys
from typing import Any, Literal, Mapping


ResearchSource = Literal["local", "server"]

_DATACLASS_KWARGS: dict[str, Any] = {"frozen": True}
if sys.version_info >= (3, 10):
    _DATACLASS_KWARGS["slots"] = True


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not str(value or "").strip():
        raise ValueError(f"{field_name} is required")


@dataclass(**_DATACLASS_KWARGS)
class ResearchRun:
    source: ResearchSource
    id: str
    query: str | None = None
    status: str = "draft"
    phase: str = "planning"
    control_state: str = "paused"
    source_policy: str = "balanced"
    autonomy_mode: str = "checkpointed"
    progress_percent: float | None = None
    progress_message: str | None = None
    active_job_id: str | None = None
    latest_checkpoint_id: str | None = None
    completed_at: str | datetime | None = None
    chat_id: str | None = None
    created_at: str | datetime | None = None
    updated_at: str | datetime | None = None
    limits_json: Mapping[str, Any] = field(default_factory=dict)
    provider_overrides: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")


@dataclass(**_DATACLASS_KWARGS)
class ResearchArtifact:
    source: ResearchSource
    run_id: str
    artifact_name: str
    content_type: str
    content: Any
    artifact_version: int = 1
    phase: str | None = None
    job_id: str | None = None
    created_at: str | datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, field_name="run_id")
        _require_non_empty(self.artifact_name, field_name="artifact_name")
        if self.artifact_version < 1:
            raise ValueError("artifact_version must be >= 1")
