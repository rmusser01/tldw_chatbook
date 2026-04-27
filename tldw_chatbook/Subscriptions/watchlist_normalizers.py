"""Normalize local subscriptions and server watchlist sources for one shell."""

from __future__ import annotations

import json
from typing import Any, Mapping


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value or {})


def build_watchlist_item_id(backend: str, entity_kind: str, source_id: Any) -> str:
    """Build the canonical local/server watchlist item id."""
    return f"{backend}:{entity_kind}:{source_id}"


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


def _json_mapping(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _local_source_settings(row: Mapping[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    scalar_fields = (
        "check_frequency",
        "extraction_method",
        "change_threshold",
        "auto_ingest",
    )
    for field in scalar_fields:
        value = row.get(field)
        if value is not None:
            settings[field] = value

    for field in (
        "extraction_rules",
        "processing_options",
        "notification_config",
        "rate_limit_config",
    ):
        parsed = _json_mapping(row.get(field))
        if parsed:
            settings[field] = parsed

    ignore_selectors = row.get("ignore_selectors")
    if ignore_selectors:
        settings["ignore_selectors"] = [
            selector.strip()
            for selector in str(ignore_selectors).split("\n")
            if selector.strip()
        ]
    return settings


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
        "id": build_watchlist_item_id("local", "subscription", source_id),
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
        "settings": _local_source_settings(row),
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
        "id": build_watchlist_item_id("server", "watchlist_source", source_id),
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
        "id": build_watchlist_item_id("server", "watchlist_source", payload.get("source_id", source_id)),
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": payload.get("source_id", source_id),
        "restore_window_seconds": payload.get("restore_window_seconds"),
        "restore_expires_at": payload.get("restore_expires_at"),
    }


def normalize_watchlist_run(source: str, run: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize local or server watchlist run metadata."""
    payload = _model_to_dict(run)
    run_id = payload["id"]
    source_id = payload.get("source_id")
    job_id = payload.get("job_id")
    if source_id is None and source == "local":
        source_id = job_id
    return {
        "id": build_watchlist_item_id(source, "watchlist_run", run_id),
        "backend": source,
        "entity_kind": "watchlist_run",
        "run_id": run_id,
        "job_id": job_id,
        "source_id": source_id,
        "status": payload.get("status") or "unknown",
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "stats": dict(payload.get("stats") or {}),
        "error_msg": payload.get("error_msg"),
        "filter_tallies": payload.get("filter_tallies"),
        "log_text": payload.get("log_text"),
        "log_path": payload.get("log_path"),
        "truncated": bool(payload.get("truncated", False)),
        "filtered_sample": payload.get("filtered_sample"),
    }


def _coerce_condition_value(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def normalize_watchlist_alert_rule(source: str, rule: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize local or server watchlist alert-rule metadata."""
    payload = _model_to_dict(rule)
    rule_id = payload["id"]
    job_id = payload.get("job_id")
    return {
        "id": build_watchlist_item_id(source, "watchlist_alert_rule", rule_id),
        "backend": source,
        "entity_kind": "watchlist_alert_rule",
        "rule_id": rule_id,
        "user_id": payload.get("user_id") or ("local" if source == "local" else None),
        "job_id": job_id,
        "source_id": payload.get("source_id") or job_id,
        "name": payload.get("name") or "Untitled alert rule",
        "enabled": bool(payload.get("enabled", True)),
        "condition_type": payload.get("condition_type"),
        "condition_value": _coerce_condition_value(payload.get("condition_value")),
        "severity": payload.get("severity") or "warning",
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }
