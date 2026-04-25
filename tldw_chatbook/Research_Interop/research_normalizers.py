"""Normalization helpers for source-aware research records."""

from __future__ import annotations

from typing import Any, Mapping


_KIND_TO_RECORD_TYPE = {
    "session": "research_session",
    "run": "research_run",
    "artifact": "research_artifact",
    "event": "research_run_event",
}


def normalize_research_record(source: str, kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a source-labeled research record without hiding backend-native IDs."""

    payload = dict(record)
    record_type = _KIND_TO_RECORD_TYPE[kind]
    if kind == "artifact":
        record_id = f"{payload.get('run_id')}:{payload.get('artifact_name') or payload.get('name') or payload.get('id')}"
    elif kind == "event":
        record_id = f"{payload.get('run_id')}:{payload.get('id')}"
    else:
        record_id = str(payload.get("id"))
    payload["source"] = source
    payload["record_type"] = record_type
    payload["record_id"] = f"{source}:{record_type}:{record_id}"
    return payload
