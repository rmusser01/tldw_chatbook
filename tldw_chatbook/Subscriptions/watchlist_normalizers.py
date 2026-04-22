"""Normalization helpers for local/server watchlist services."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

_ALLOWED_WATCHLIST_SOURCE_TYPES = {"rss", "site"}
_LEGACY_LOCAL_SOURCE_TYPE_MAP = {"url": "site"}


def build_watchlist_item_id(backend: str, entity_kind: str, source_id: Any) -> str:
    return f"{str(backend)}:{str(entity_kind)}:{str(source_id)}"


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return None


def _coerce_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)):
                value = decoded
            else:
                value = [part.strip() for part in stripped.split(",")]
        else:
            value = [part.strip() for part in stripped.split(",")]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = str(item).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(cleaned)
        return normalized
    return []


def _coerce_group_ids(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result
    return []


def _clean_timestamp(*values: Any) -> str | None:
    for value in values:
        if value in (None, ""):
            continue
        return str(value)
    return None


def _coerce_settings(value: Any) -> dict[str, Any] | None:
    mapping = _coerce_mapping(value)
    if mapping is not None:
        return mapping
    return None


def _local_status_summary(row: Mapping[str, Any]) -> str:
    if bool(row.get("is_paused")):
        return "paused"
    if not bool(row.get("is_active", True)):
        return "inactive"
    if row.get("last_error") not in (None, ""):
        return "error"
    return "active"


def normalize_watchlist_source_type(raw_source_type: Any, *, backend: str) -> str:
    if raw_source_type in (None, ""):
        raise ValueError(f"Unsupported {backend} watchlist source type: {raw_source_type}")

    normalized = str(raw_source_type).strip().lower()
    if backend == "local":
        normalized = _LEGACY_LOCAL_SOURCE_TYPE_MAP.get(normalized, normalized)

    if normalized not in _ALLOWED_WATCHLIST_SOURCE_TYPES:
        raise ValueError(f"Unsupported {backend} watchlist source type: {raw_source_type}")
    return normalized


def normalize_local_subscription_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source_id = row["id"]
    return {
        "id": build_watchlist_item_id("local", "subscription", source_id),
        "backend": "local",
        "entity_kind": "subscription",
        "source_id": source_id,
        "title": row["name"],
        "source_type": normalize_watchlist_source_type(row.get("type"), backend="local"),
        "url": row["source"],
        "active": bool(row["is_active"]) and not bool(row["is_paused"]),
        "tags": _coerce_tags(row.get("tags")),
        "status_summary": _local_status_summary(row),
        "last_checked_or_scraped_at": row.get("last_checked"),
        "created_at": _clean_timestamp(row.get("created_at")),
        "updated_at": _clean_timestamp(row.get("updated_at")),
    }


def normalize_server_watchlist_source(source: Mapping[str, Any] | Any) -> dict[str, Any]:
    source_mapping = _coerce_mapping(source) or {}
    source_id = source_mapping.get("id")
    return {
        "id": build_watchlist_item_id("server", "watchlist_source", source_id),
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": source_id,
        "title": source_mapping.get("name") or "",
        "source_type": normalize_watchlist_source_type(source_mapping.get("source_type"), backend="server"),
        "url": str(source_mapping.get("url")) if source_mapping.get("url") is not None else None,
        "active": bool(source_mapping.get("active", True)),
        "tags": _coerce_tags(source_mapping.get("tags")),
        "group_ids": _coerce_group_ids(source_mapping.get("group_ids")),
        "settings": _coerce_settings(source_mapping.get("settings")),
        "status_summary": str(source_mapping.get("status") or ("active" if source_mapping.get("active", True) else "inactive")),
        "last_checked_or_scraped_at": _clean_timestamp(
            source_mapping.get("last_scraped_at"),
            source_mapping.get("last_checked"),
        ),
        "created_at": _clean_timestamp(source_mapping.get("created_at")),
        "updated_at": _clean_timestamp(source_mapping.get("updated_at")),
    }


__all__ = [
    "build_watchlist_item_id",
    "normalize_local_subscription_row",
    "normalize_server_watchlist_source",
    "normalize_watchlist_source_type",
]
