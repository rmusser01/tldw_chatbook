"""Normalization helpers for the study flashcards compat seam."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump(mode="json"))
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"Expected mapping-like study record, got {type(record)!r}")


def _parse_local_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str) and metadata.strip():
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _parse_local_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    seen: set[str] = set()
    tags: list[str] = []
    for item in text.split():
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            tags.append(cleaned)
    return tags


def _normalize_backend(backend: str) -> str:
    normalized = str(backend or "").strip().lower()
    if normalized not in {"local", "server"}:
        raise ValueError(f"Invalid study backend: {backend!r}")
    return normalized


def _normalize_selection_reason(reason: Any) -> str:
    normalized = str(reason or "").strip().lower()
    mapping = {
        "learning_due": "learning",
        "review_due": "review",
        "new": "new",
        "due": "due",
        "review": "review",
        "learning": "learning",
        "none": "none",
    }
    return mapping.get(normalized, "unknown")


def normalize_study_deck_record(backend: str, record: Any) -> dict[str, Any]:
    normalized_backend = _normalize_backend(backend)
    raw = _as_dict(record)
    metadata = _parse_local_metadata(raw) if normalized_backend == "local" else {}

    backing_id = raw.get("id")
    updated_at = raw.get("updated_at") or raw.get("last_modified")
    deleted = bool(raw.get("is_deleted")) if normalized_backend == "local" else bool(raw.get("deleted"))
    client_id = raw.get("client_id") or raw.get("created_by") or raw.get("last_modified_by")

    if normalized_backend == "server":
        scheduler_type = raw.get("scheduler_type")
        if raw.get("scheduler_settings") is not None:
            metadata["scheduler_settings"] = raw.get("scheduler_settings")
        if raw.get("scheduler_settings_json") is not None:
            metadata["scheduler_settings_json"] = raw.get("scheduler_settings_json")
        workspace_id = raw.get("workspace_id")
    else:
        scheduler_type = metadata.get("scheduler_type")
        workspace_id = None

    return {
        "record_id": f"{normalized_backend}:study_deck:{backing_id}",
        "record_type": "study_deck",
        "backend": normalized_backend,
        "backing_id": backing_id,
        "name": str(raw.get("name") or ""),
        "description": raw.get("description"),
        "workspace_id": workspace_id,
        "created_at": raw.get("created_at"),
        "updated_at": updated_at,
        "deleted": deleted,
        "version": raw.get("version"),
        "client_id": client_id,
        "scheduler_type": scheduler_type,
        "metadata": metadata,
    }


def normalize_study_flashcard_record(backend: str, record: Any) -> dict[str, Any]:
    normalized_backend = _normalize_backend(backend)
    raw = _as_dict(record)
    metadata = _parse_local_metadata(raw) if normalized_backend == "local" else {}

    if normalized_backend == "local":
        backing_id = raw.get("id")
        deck_id = raw.get("deck_id")
        interval_days = raw.get("interval")
        repetitions = raw.get("repetitions")
        due_at = raw.get("next_review")
        last_reviewed_at = raw.get("last_review")
        queue_state = "suspended" if bool(raw.get("is_suspended")) else ("new" if not last_reviewed_at and not due_at and int(interval_days or 0) == 0 and int(repetitions or 0) == 0 else "review")
        card_kind = str(raw.get("type") or metadata.get("card_kind") or "unknown")
        tags = _parse_local_tags(raw.get("tags"))
        notes = metadata.get("notes")
        extra = metadata.get("extra")
        created_at = raw.get("created_at")
        updated_at = raw.get("updated_at")
        deleted = bool(raw.get("is_deleted"))
        client_id = raw.get("client_id") or raw.get("created_by") or raw.get("last_modified_by")
        review_detail_available = False
    else:
        backing_id = raw.get("uuid") or raw.get("id")
        deck_id = raw.get("deck_id")
        interval_days = raw.get("interval_days")
        repetitions = raw.get("repetitions")
        due_at = raw.get("due_at")
        last_reviewed_at = raw.get("last_reviewed_at")
        queue_state = raw.get("queue_state")
        card_kind = raw.get("model_type") or ("cloze" if raw.get("is_cloze") else "unknown")
        tags = [str(item) for item in list(raw.get("tags") or []) if str(item).strip()]
        notes = raw.get("notes")
        extra = raw.get("extra")
        created_at = raw.get("created_at")
        updated_at = raw.get("last_modified")
        deleted = bool(raw.get("deleted"))
        client_id = raw.get("client_id")
        review_detail_available = True
        for key in ("next_intervals", "scheduler_type", "lapses", "source_ref_type", "source_ref_id", "conversation_id", "message_id", "step_index", "suspended_reason"):
            if raw.get(key) is not None:
                metadata[key] = raw.get(key)

    return {
        "record_id": f"{normalized_backend}:study_flashcard:{backing_id}",
        "record_type": "study_flashcard",
        "backend": normalized_backend,
        "backing_id": backing_id,
        "deck_id": deck_id,
        "deck_record_id": f"{normalized_backend}:study_deck:{deck_id}" if deck_id is not None else None,
        "front": str(raw.get("front") or ""),
        "back": str(raw.get("back") or ""),
        "notes": notes,
        "extra": extra,
        "tags": tags,
        "card_kind": str(card_kind or "unknown"),
        "due_at": due_at,
        "last_reviewed_at": last_reviewed_at,
        "interval_days": interval_days,
        "repetitions": repetitions,
        "ease_factor": raw.get("ease_factor") if normalized_backend == "local" else raw.get("ef"),
        "queue_state": queue_state,
        "suspended": bool(raw.get("is_suspended")) if normalized_backend == "local" else str(queue_state or "") == "suspended",
        "created_at": created_at,
        "updated_at": updated_at,
        "deleted": deleted,
        "version": raw.get("version"),
        "client_id": client_id,
        "review_detail_available": review_detail_available,
        "metadata": metadata,
    }


def normalize_study_review_candidate(
    backend: str,
    *,
    card: Any,
    selection_reason: Any,
    review_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_card = None if card is None else (
        card if isinstance(card, Mapping) and "record_id" in card else normalize_study_flashcard_record(backend, card)
    )
    card_metadata = normalized_card.get("metadata", {}) if normalized_card else {}
    next_intervals = card_metadata.get("next_intervals")
    return {
        "card": normalized_card,
        "selection_reason": _normalize_selection_reason(selection_reason),
        "next_intervals": next_intervals,
        "review_session": review_session,
        "detail_available": bool(normalized_card and normalized_card.get("review_detail_available")),
    }


def merge_review_outcome_record(
    backend: str,
    *,
    current_card: Mapping[str, Any] | None,
    review_response: Any,
    rating: int,
) -> dict[str, Any]:
    normalized_backend = _normalize_backend(backend)
    raw = _as_dict(review_response)

    if normalized_backend == "server" and current_card is not None and "front" not in raw:
        merged_card = dict(current_card)
        merged_card.update(
            {
                "due_at": raw.get("due_at"),
                "last_reviewed_at": raw.get("last_reviewed_at"),
                "interval_days": raw.get("interval_days"),
                "repetitions": raw.get("repetitions"),
                "ease_factor": raw.get("ef"),
                "queue_state": raw.get("queue_state"),
                "updated_at": raw.get("last_modified"),
                "version": raw.get("version"),
                "review_detail_available": True,
            }
        )
        merged_metadata = dict(merged_card.get("metadata", {}))
        if raw.get("next_intervals") is not None:
            merged_metadata["next_intervals"] = raw.get("next_intervals")
        if raw.get("scheduler_type") is not None:
            merged_metadata["scheduler_type"] = raw.get("scheduler_type")
        merged_card["metadata"] = merged_metadata
        normalized_card = merged_card
    else:
        payload = raw.get("card", raw)
        normalized_card = (
            payload
            if isinstance(payload, Mapping) and "record_id" in payload
            else normalize_study_flashcard_record(normalized_backend, payload)
        )

    review_session = raw.get("review_session")
    if review_session is None and raw.get("review_session_id") is not None:
        review_session = {"review_session_id": raw.get("review_session_id")}
    next_intervals = raw.get("next_intervals") or normalized_card.get("metadata", {}).get("next_intervals")
    return {
        "card": normalized_card,
        "rating": rating,
        "next_intervals": next_intervals,
        "review_session": review_session,
        "detail_available": bool(normalized_card.get("review_detail_available")) or bool(review_session),
    }
