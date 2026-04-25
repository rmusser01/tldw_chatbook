"""Normalize local subscriptions and server watchlist sources for one shell."""

from __future__ import annotations

import json
from typing import Any, Mapping


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value or {})


def _coerce_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return _coerce_tags(parsed)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def normalize_local_subscription_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a local subscriptions DB row as a watch item."""
    source_id = row["id"]
    active = bool(row.get("is_active", True)) and not bool(row.get("is_paused", False))
    error_count = int(row.get("error_count") or 0)
    last_error = row.get("last_error")
    status_summary = "active" if active else "inactive"
    if last_error:
        status_summary = f"error ({error_count})" if error_count else "error"

    return {
        "id": f"local:subscription:{source_id}",
        "backend": "local",
        "entity_kind": "subscription",
        "source_id": source_id,
        "title": row.get("name") or "Untitled subscription",
        "description": row.get("description"),
        "source_type": row.get("type"),
        "url": row.get("source"),
        "active": active,
        "tags": _coerce_tags(row.get("tags")),
        "group_ids": [],
        "settings": {},
        "status_summary": status_summary,
        "last_checked_or_scraped_at": row.get("last_checked") or row.get("last_successful_check"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def normalize_server_watchlist_source(source: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize a server watchlist source response as a watch item."""
    payload = _model_to_dict(source)
    source_id = payload["id"]
    return {
        "id": f"server:watchlist_source:{source_id}",
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": source_id,
        "title": payload.get("name") or "Untitled source",
        "description": payload.get("description"),
        "source_type": payload.get("source_type"),
        "url": payload.get("url"),
        "active": bool(payload.get("active", True)),
        "tags": _coerce_tags(payload.get("tags")),
        "group_ids": list(payload.get("group_ids") or []),
        "settings": dict(payload.get("settings") or {}),
        "status_summary": "active" if payload.get("active", True) else "inactive",
        "last_checked_or_scraped_at": payload.get("last_checked_at") or payload.get("last_scraped_at"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def normalize_server_delete_response(response: Mapping[str, Any] | Any, *, source_id: Any) -> dict[str, Any]:
    """Normalize server reversible-delete metadata."""
    payload = _model_to_dict(response)
    return {
        "success": bool(payload.get("success", True)),
        "id": f"server:watchlist_source:{payload.get('source_id', source_id)}",
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": payload.get("source_id", source_id),
        "restore_window_seconds": payload.get("restore_window_seconds"),
        "restore_expires_at": payload.get("restore_expires_at"),
    }
