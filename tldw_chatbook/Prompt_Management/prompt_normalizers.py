"""Normalization helpers for local/server prompt parity surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected prompt mapping-like value, got {type(value).__name__}")


def _normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = value

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def normalize_prompt_record(record: Any, *, backend: str) -> dict[str, Any]:
    """Return a source-stable prompt record for UI and sync-facing callers."""
    data = _to_plain_dict(record)
    source_id = data.get("uuid") or data.get("id") or data.get("name")
    if source_id in (None, ""):
        raise ValueError("Prompt record must include uuid, id, or name.")

    backend_value = str(backend)
    normalized: dict[str, Any] = {
        "id": f"{backend_value}:prompt:{source_id}",
        "backend": backend_value,
        "source_id": str(source_id),
        "local_id": data.get("id") if backend_value == "local" else None,
        "server_id": data.get("id") if backend_value == "server" else None,
        "uuid": data.get("uuid"),
        "name": data.get("name"),
        "author": data.get("author"),
        "details": data.get("details"),
        "system_prompt": data.get("system_prompt"),
        "user_prompt": data.get("user_prompt"),
        "prompt_format": data.get("prompt_format") or "legacy",
        "prompt_schema_version": data.get("prompt_schema_version"),
        "prompt_definition": data.get("prompt_definition"),
        "keywords": _normalize_keywords(data.get("keywords")),
        "deleted": bool(data.get("deleted", False)),
        "version": data.get("version"),
        "last_modified": data.get("last_modified"),
        "usage_count": int(data.get("usage_count", 0) or 0),
        "last_used_at": data.get("last_used_at"),
    }
    return normalized


def normalize_prompt_list(payload: Any, *, backend: str, page: int = 1, per_page: int = 10) -> dict[str, Any]:
    """Normalize paginated prompt list responses from local DBs or the server API."""
    if isinstance(payload, tuple) and len(payload) == 4:
        items, total_pages, current_page, total_items = payload
        return {
            "items": [normalize_prompt_record(item, backend=backend) for item in items],
            "total_pages": total_pages,
            "current_page": current_page,
            "total_items": total_items,
            "page": current_page,
            "per_page": per_page,
        }

    data = _to_plain_dict(payload)
    raw_items = data.get("items", [])
    return {
        "items": [normalize_prompt_record(item, backend=backend) for item in raw_items],
        "total_pages": int(data.get("total_pages", 0) or 0),
        "current_page": int(data.get("current_page", data.get("page", page)) or page),
        "total_items": int(data.get("total_items", len(raw_items)) or 0),
        "page": int(data.get("current_page", data.get("page", page)) or page),
        "per_page": int(data.get("per_page", per_page) or per_page),
    }


def normalize_prompt_version_record(record: Any, *, backend: str) -> dict[str, Any]:
    data = _to_plain_dict(record)
    return {
        "backend": str(backend),
        "version": data.get("version"),
        "created_at": data.get("created_at"),
        "comment": data.get("comment"),
        "name": data.get("name"),
        "author": data.get("author"),
        "details": data.get("details"),
        "system_prompt": data.get("system_prompt"),
        "user_prompt": data.get("user_prompt"),
        "prompt_uuid": data.get("prompt_uuid"),
        "prompt_format": data.get("prompt_format") or "legacy",
        "prompt_schema_version": data.get("prompt_schema_version"),
        "prompt_definition": data.get("prompt_definition"),
    }


def normalize_prompt_version_list(payload: Any, *, backend: str) -> list[dict[str, Any]]:
    return [normalize_prompt_version_record(item, backend=backend) for item in list(payload or [])]


def normalize_prompt_collection_record(record: Any, *, backend: str) -> dict[str, Any]:
    data = _to_plain_dict(record)
    collection_id = data.get("collection_id")
    if collection_id in (None, ""):
        raise ValueError("Prompt collection record must include collection_id.")
    backend_value = str(backend)
    return {
        "id": f"{backend_value}:prompt_collection:{collection_id}",
        "backend": backend_value,
        "collection_id": int(collection_id),
        "name": data.get("name"),
        "description": data.get("description"),
        "prompt_ids": list(data.get("prompt_ids") or []),
    }


def normalize_prompt_collection_list(payload: Any, *, backend: str, limit: int = 200, offset: int = 0) -> dict[str, Any]:
    data = _to_plain_dict(payload)
    raw_items = data.get("collections", [])
    return {
        "collections": [normalize_prompt_collection_record(item, backend=backend) for item in raw_items],
        "limit": int(data.get("limit", limit) or limit),
        "offset": int(data.get("offset", offset) or offset),
        "total": int(data.get("total", len(raw_items)) or 0),
    }
