"""Normalization helpers for local and server Research Sessions records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .research_models import ResearchArtifact, ResearchRun


def _as_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_research_run(source: str, record: Any) -> ResearchRun:
    data = _as_mapping(record)
    metadata = dict(data.get("metadata") or {})
    return ResearchRun(
        source="server" if source == "server" else "local",
        id=str(data["id"]),
        query=data.get("query"),
        status=str(data.get("status") or "draft"),
        phase=str(data.get("phase") or "planning"),
        control_state=str(data.get("control_state") or "paused"),
        source_policy=str(data.get("source_policy") or "balanced"),
        autonomy_mode=str(data.get("autonomy_mode") or "checkpointed"),
        progress_percent=data.get("progress_percent"),
        progress_message=data.get("progress_message"),
        active_job_id=data.get("active_job_id"),
        latest_checkpoint_id=data.get("latest_checkpoint_id"),
        completed_at=data.get("completed_at"),
        chat_id=data.get("chat_id"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at") or data.get("last_modified"),
        limits_json=dict(data.get("limits_json") or {}),
        provider_overrides=dict(data.get("provider_overrides") or {}),
        metadata=metadata,
    )


def normalize_research_artifact(source: str, record: Any) -> ResearchArtifact:
    data = _as_mapping(record)
    return ResearchArtifact(
        source="server" if source == "server" else "local",
        run_id=str(data.get("run_id") or data.get("session_id") or ""),
        artifact_name=str(data["artifact_name"]),
        content_type=str(data.get("content_type") or "application/json"),
        content=data.get("content"),
        artifact_version=int(data.get("artifact_version") or 1),
        phase=data.get("phase"),
        job_id=data.get("job_id"),
        created_at=data.get("created_at"),
    )
