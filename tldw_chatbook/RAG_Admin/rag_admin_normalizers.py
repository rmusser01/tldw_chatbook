"""Normalization helpers for local/server retrieval-admin records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    return value


def _to_mapping(value: Any) -> dict[str, Any]:
    value = _model_dump(value)
    if isinstance(value, Mapping):
        return dict(value)

    payload: dict[str, Any] = {}
    for attribute in ("id", "name", "description", "template_json", "metadata", "count", "embedding_dimension"):
        if hasattr(value, attribute):
            payload[attribute] = getattr(value, attribute)
    return payload


def _parse_template_payload(value: Any) -> dict[str, Any]:
    value = _model_dump(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_template_record(backend: str, payload: Any) -> dict[str, Any]:
    data = _to_mapping(payload)
    name = str(data.get("name") or data.get("template_name") or "").strip()
    template = _parse_template_payload(data.get("template") or data.get("template_json"))
    template_json = data.get("template_json")
    if not isinstance(template_json, str):
        template_json = json.dumps(template or {})

    return {
        "record_id": f"{backend}:chunking_template:{name}",
        "record_type": "chunking_template",
        "backend": backend,
        "backing_id": data.get("id") or name,
        "backing_template_name": name,
        "name": name,
        "description": data.get("description") or "",
        "template_json": template_json,
        "template": template,
        "is_builtin": bool(data.get("is_builtin", data.get("is_system", False))),
        "tags": list(data.get("tags") or []),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "version": _safe_int(data.get("version")) or 1,
        "user_id": data.get("user_id"),
    }


def normalize_collection_record(backend: str, payload: Any) -> dict[str, Any]:
    data = _to_mapping(payload)
    name = str(data.get("name") or "").strip()
    metadata_value = _model_dump(data.get("metadata") or {})
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    embedding_dimension = _safe_int(data.get("embedding_dimension"))
    if embedding_dimension is None:
        embedding_dimension = _safe_int(metadata.get("embedding_dimension"))

    return {
        "record_id": f"{backend}:embedding_collection:{name}",
        "record_type": "embedding_collection",
        "backend": backend,
        "backing_collection_name": name,
        "name": name,
        "count": _safe_int(data.get("count")) or 0,
        "embedding_dimension": embedding_dimension,
        "metadata": metadata,
        "provider": metadata.get("provider"),
        "status": data.get("status") or "ready",
    }
