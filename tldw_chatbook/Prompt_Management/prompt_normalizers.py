"""Normalization helpers for local/server prompt parity surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .prompt_artifact_codec import decode_prompt_artifact, deserialize_definition
from .prompt_restore_errors import PromptRestoreError, PromptRestoreErrorCode
from .prompt_source_capabilities import (
    PromptCapabilityError,
    PromptSourceCapabilities,
    validate_console_artifact_payload,
    validate_prompt_request_size,
)


_HISTORY_COMPATIBILITY_REASONS = {
    "malformed_payload": "Retained snapshot JSON is malformed.",
    "non_object_payload": "Retained snapshot payload must be a JSON object.",
    "malformed_fields": "Retained Prompt fields are malformed.",
    "malformed_keywords": "Captured keywords are not a canonical keyword list.",
    "unknown_format": "Prompt format is unsupported.",
    "malformed_definition": "Structured definition is malformed.",
    "schema_mismatch": (
        "Prompt schema version does not match the definition schema version."
    ),
    "compiled_text_mismatch": (
        "Stored System/User text does not match the structured definition."
    ),
    "artifact_kind_mismatch": (
        "Artifact type does not match the structured definition kind."
    ),
    "unsupported_schema": "Prompt schema version is unsupported.",
    "unsupported_artifact_type": "Artifact type is unsupported.",
    "foreign_v1": "Structured-v1 artifacts are preview-only.",
    "unsupported_definition_kind": "Structured definition kind is unsupported.",
    "legacy_recipe": "Legacy Recipe snapshots are preview-only.",
    "current_capability_unsupported": (
        "This retained version is not supported by current local Prompt capabilities."
    ),
}

_HISTORY_CHANGED_FIELDS = (
    ("name", "Name"),
    ("author", "Author"),
    ("details", "Description"),
    ("system_prompt", "System"),
    ("user_prompt", "User"),
    ("prompt_format", "Format"),
    ("prompt_schema_version", "Schema version"),
    ("prompt_definition", "Definition"),
    ("artifact_type", "Artifact type"),
)


def _normalize_read_artifact_type(value: Any) -> tuple[str, str | None]:
    """Classify untrusted record types without weakening strict write validation."""
    if value is None:
        return "prompt", None
    if isinstance(value, str) and value in {"prompt", "recipe"}:
        return value, None
    raw_value = value if isinstance(value, str) else type(value).__name__
    return "unsupported", raw_value


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


def _lane_flag(data: Mapping[str, Any], flag: str, text_field: str) -> bool:
    advertised = data.get(flag)
    if isinstance(advertised, bool):
        return advertised
    return bool(str(data.get(text_field) or "").strip())


def _history_positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"Invalid retained history row: {field} must be a positive integer."
        )
    return value


def _history_canonical_keywords(value: Any) -> tuple[list[str], bool]:
    if type(value) is not list or not all(type(item) is str for item in value):
        return [], False
    canonical = sorted(
        {normalized for item in value if (normalized := " ".join(item.split()).lower())}
    )
    if len(canonical) != len(value) or list(value) != canonical:
        return [], False
    return list(value), True


def _set_history_compatibility(
    normalized: dict[str, Any],
    *,
    state: str,
    definition_state: str,
) -> None:
    normalized["compatibility_state"] = state
    normalized["definition_state"] = definition_state
    normalized["compatibility_reason"] = _HISTORY_COMPATIBILITY_REASONS.get(state)
    normalized["restore_eligible"] = state == "compatible"


def _history_restore_fields(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: snapshot[field]
        for field in (
            "name",
            "author",
            "details",
            "system_prompt",
            "user_prompt",
            "prompt_format",
            "prompt_schema_version",
            "prompt_definition",
            "artifact_type",
        )
    }


def _apply_history_capabilities(
    normalized: dict[str, Any], capabilities: PromptSourceCapabilities | None
) -> None:
    """Fail closed with bounded copy when a valid v2 snapshot exceeds its source."""
    if (
        capabilities is None
        or not normalized["restore_eligible"]
        or normalized["prompt_format"] != "structured"
    ):
        return
    try:
        update_data = validate_console_artifact_payload(
            _history_restore_fields(normalized), capabilities
        )
        validate_prompt_request_size(update_data, capabilities)
    except (PromptCapabilityError, ValueError):
        _set_history_compatibility(
            normalized,
            state="current_capability_unsupported",
            definition_state=normalized["definition_state"],
        )


def _classify_history_artifact(
    payload: Mapping[str, Any], normalized: dict[str, Any]
) -> None:
    raw_artifact_type = payload.get("artifact_type")
    artifact_type, artifact_type_raw = _normalize_read_artifact_type(raw_artifact_type)
    normalized["artifact_type"] = artifact_type
    if artifact_type_raw is not None:
        normalized["artifact_type_raw"] = artifact_type_raw
        _set_history_compatibility(
            normalized,
            state="unsupported_artifact_type",
            definition_state="unsupported",
        )
        return

    prompt_format = payload.get("prompt_format")
    if prompt_format is None:
        prompt_format = "legacy"
    normalized["prompt_format"] = prompt_format
    if prompt_format == "legacy":
        if artifact_type == "recipe":
            _set_history_compatibility(
                normalized, state="legacy_recipe", definition_state="legacy"
            )
            return
        _set_history_compatibility(
            normalized, state="compatible", definition_state="legacy"
        )
        return
    if prompt_format != "structured":
        _set_history_compatibility(
            normalized, state="unknown_format", definition_state="unsupported"
        )
        return

    schema_version = payload.get("prompt_schema_version")
    if type(schema_version) is int and schema_version == 1:
        # Foreign v1 is intentionally classified from its outer discriminator only.
        # Its definition remains opaque and never enters the Console-v2 decoder.
        _set_history_compatibility(
            normalized, state="foreign_v1", definition_state="foreign_v1"
        )
        return
    if type(schema_version) is not int or schema_version != 2:
        _set_history_compatibility(
            normalized, state="unsupported_schema", definition_state="unsupported"
        )
        return

    raw_definition = deserialize_definition(payload.get("prompt_definition"))
    if raw_definition is None:
        _set_history_compatibility(
            normalized, state="malformed_definition", definition_state="malformed"
        )
        return
    if (
        type(raw_definition.get("schema_version")) is not int
        or raw_definition.get("schema_version") != schema_version
    ):
        _set_history_compatibility(
            normalized, state="schema_mismatch", definition_state="mismatched"
        )
        return

    kind = raw_definition.get("kind")
    expected_kind = "block_prompt" if artifact_type == "prompt" else "block_recipe"
    if kind in {"block_prompt", "block_recipe"} and kind != expected_kind:
        _set_history_compatibility(
            normalized, state="artifact_kind_mismatch", definition_state="mismatched"
        )
        return
    if raw_definition.get("definition_kind") == "single_text_recipe":
        _set_history_compatibility(
            normalized,
            state="unsupported_definition_kind",
            definition_state="unsupported",
        )
        return

    decoded = decode_prompt_artifact(payload)
    normalized["compiled_system_prompt"] = decoded.compiled_system
    normalized["compiled_user_prompt"] = decoded.compiled_user
    normalized["compatibility_stale"] = decoded.compatibility_stale
    if decoded.state != "supported_v2":
        _set_history_compatibility(
            normalized, state="malformed_definition", definition_state=decoded.state
        )
    elif decoded.compatibility_stale:
        _set_history_compatibility(
            normalized,
            state="compiled_text_mismatch",
            definition_state="supported_v2",
        )
    else:
        _set_history_compatibility(
            normalized, state="compatible", definition_state="supported_v2"
        )


def _normalize_prompt_history_row(
    record: Any,
    *,
    backend: str,
    capabilities: PromptSourceCapabilities | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("Invalid retained history row: expected an object.")
    data = dict(record)
    change_id = _history_positive_int(data.get("change_id"), field="change_id")
    version = _history_positive_int(data.get("version"), field="version")
    operation = data.get("operation")
    if operation not in {"create", "update"}:
        raise ValueError(
            "Invalid retained history row: operation must be create or update."
        )
    timestamp = data.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(
            "Invalid retained history row: timestamp must be a non-empty string."
        )
    prompt_uuid = data.get("entity_uuid")
    if not isinstance(prompt_uuid, str) or not prompt_uuid.strip():
        raise ValueError(
            "Invalid retained history row: entity_uuid must be a non-empty string."
        )
    if data.get("entity") not in {None, "Prompts"}:
        raise ValueError("Invalid retained history row: entity must be Prompts.")

    normalized: dict[str, Any] = {
        "backend": str(backend),
        "change_id": change_id,
        "version": version,
        "operation": operation,
        "timestamp": timestamp,
        "prompt_uuid": prompt_uuid,
        "name": None,
        "author": None,
        "details": None,
        "system_prompt": "",
        "user_prompt": "",
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
        "artifact_type": "prompt",
        "keywords": [],
        "keywords_captured": False,
        "keywords_preview": None,
        "payload_preview": None,
        "definition_state": "malformed",
        "compatibility_state": "malformed_payload",
        "compatibility_reason": _HISTORY_COMPATIBILITY_REASONS["malformed_payload"],
        "compatibility_stale": False,
        "compiled_system_prompt": "",
        "compiled_user_prompt": "",
        "restore_eligible": False,
        "changed_fields": [],
        "change_summary": "Earlier baseline unavailable",
    }

    payload_error = data.get("payload_error")
    payload = data.get("payload")
    if payload_error == "malformed_json":
        normalized["payload_preview"] = data.get("raw_payload")
        return normalized
    if payload_error is not None:
        raise ValueError(
            "Invalid retained history row: payload_error is not recognized."
        )
    if not isinstance(payload, Mapping):
        normalized["payload_preview"] = payload
        _set_history_compatibility(
            normalized, state="non_object_payload", definition_state="malformed"
        )
        return normalized

    snapshot = dict(payload)
    raw_system = snapshot.get("system_prompt")
    raw_user = snapshot.get("user_prompt")
    normalized.update(
        {
            "name": snapshot.get("name"),
            "author": snapshot.get("author"),
            "details": snapshot.get("details"),
            "system_prompt": "" if raw_system is None else raw_system,
            "user_prompt": "" if raw_user is None else raw_user,
            "prompt_format": snapshot.get("prompt_format") or "legacy",
            "prompt_schema_version": snapshot.get("prompt_schema_version"),
            "prompt_definition": snapshot.get("prompt_definition"),
            "compiled_system_prompt": "" if raw_system is None else raw_system,
            "compiled_user_prompt": "" if raw_user is None else raw_user,
        }
    )

    valid_fields = (
        isinstance(normalized["name"], str)
        and bool(normalized["name"].strip())
        and all(
            value is None or isinstance(value, str)
            for value in (normalized["author"], normalized["details"])
        )
        and isinstance(normalized["system_prompt"], str)
        and isinstance(normalized["user_prompt"], str)
    )
    if not valid_fields:
        normalized["payload_preview"] = snapshot
        _set_history_compatibility(
            normalized, state="malformed_fields", definition_state="malformed"
        )
        return normalized

    keywords_valid = True
    if "keywords" in snapshot:
        keywords, keywords_valid = _history_canonical_keywords(snapshot["keywords"])
        if keywords_valid:
            normalized["keywords"] = keywords
            normalized["keywords_captured"] = True
        else:
            normalized["keywords_preview"] = snapshot["keywords"]

    _classify_history_artifact(snapshot, normalized)
    if not keywords_valid and normalized["restore_eligible"]:
        _set_history_compatibility(
            normalized, state="malformed_keywords", definition_state="malformed"
        )
    _apply_history_capabilities(normalized, capabilities)
    return normalized


def _history_row_can_supply_baseline(row: Mapping[str, Any]) -> bool:
    return row.get("compatibility_state") not in {
        "malformed_payload",
        "non_object_payload",
        "malformed_fields",
        "malformed_keywords",
        "malformed_definition",
    }


def _add_history_change_summary(
    current: dict[str, Any], older: Mapping[str, Any] | None
) -> None:
    if current["version"] == 1:
        current["change_summary"] = "Created"
        return
    if (
        older is None
        or older["version"] != current["version"] - 1
        or not _history_row_can_supply_baseline(current)
        or not _history_row_can_supply_baseline(older)
    ):
        current["change_summary"] = "Earlier baseline unavailable"
        return

    changed_fields = [
        field
        for field, _label in _HISTORY_CHANGED_FIELDS
        if current.get(field) != older.get(field)
    ]
    if current["keywords_captured"] and older["keywords_captured"]:
        if current["keywords"] != older["keywords"]:
            changed_fields.append("keywords")
    labels = dict(_HISTORY_CHANGED_FIELDS)
    labels["keywords"] = "Keywords"
    current["changed_fields"] = changed_fields
    current["change_summary"] = (
        ", ".join(labels[field] for field in changed_fields)
        if changed_fields
        else "No restorable fields changed"
    )


def normalize_prompt_record(record: Any, *, backend: str) -> dict[str, Any]:
    """Return a source-stable prompt record for UI and sync-facing callers."""
    data = _to_plain_dict(record)
    artifact_type, artifact_type_raw = _normalize_read_artifact_type(
        data.get("artifact_type")
    )
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
        "artifact_type": artifact_type,
        "has_system_prompt": _lane_flag(data, "has_system_prompt", "system_prompt"),
        "has_user_prompt": _lane_flag(data, "has_user_prompt", "user_prompt"),
    }
    if artifact_type_raw is not None:
        normalized["artifact_type_raw"] = artifact_type_raw
        normalized["definition_state"] = "unsupported"
        normalized["compiled_system_prompt"] = str(data.get("system_prompt") or "")
        normalized["compiled_user_prompt"] = str(data.get("user_prompt") or "")
        normalized["compatibility_stale"] = False
    elif data.get("prompt_definition") is not None:
        decoded = decode_prompt_artifact(data)
        normalized["artifact_type"] = decoded.artifact_type
        normalized["definition_state"] = decoded.state
        normalized["prompt_definition"] = (
            decoded.raw_definition
            if decoded.raw_definition is not None
            else data.get("prompt_definition")
        )
        normalized["compiled_system_prompt"] = decoded.compiled_system
        normalized["compiled_user_prompt"] = decoded.compiled_user
        normalized["compatibility_stale"] = decoded.compatibility_stale
    return normalized


def normalize_prompt_list(
    payload: Any, *, backend: str, page: int = 1, per_page: int = 10
) -> dict[str, Any]:
    """Normalize paginated prompt list responses from local DBs or the server API."""
    def page_int(value: Any, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise TypeError(f"{field} must be an integer.")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer.") from exc

    if isinstance(payload, tuple) and len(payload) == 4:
        items, total_pages, current_page, total_items = payload
        current_page = page_int(current_page, field="current_page")
        return {
            "items": [normalize_prompt_record(item, backend=backend) for item in items],
            "total_pages": page_int(total_pages, field="total_pages"),
            "current_page": current_page,
            "total_items": page_int(total_items, field="total_items"),
            "page": current_page,
            "per_page": page_int(per_page, field="per_page"),
        }

    data = _to_plain_dict(payload)
    raw_items = data.get("items", [])
    page_alias = page_int(data["page"] if "page" in data else page, field="page")
    current_page = page_int(
        data["current_page"] if "current_page" in data else page_alias,
        field="current_page",
    )
    return {
        "items": [normalize_prompt_record(item, backend=backend) for item in raw_items],
        "total_pages": page_int(
            data["total_pages"] if "total_pages" in data else 0,
            field="total_pages",
        ),
        "current_page": current_page,
        "total_items": page_int(
            data["total_items"] if "total_items" in data else len(raw_items),
            field="total_items",
        ),
        "page": page_alias if "page" in data else current_page,
        "per_page": page_int(
            data["per_page"] if "per_page" in data else per_page,
            field="per_page",
        ),
    }


def normalize_prompt_history_page(
    payload: Any,
    *,
    backend: str,
    capabilities: PromptSourceCapabilities | None = None,
) -> dict[str, Any]:
    """Normalize one bounded retained-history page without exposing its predecessor."""
    if not isinstance(payload, Mapping):
        raise TypeError("Invalid retained history page: expected an object.")
    data = dict(payload)
    raw_items = data.get("items")
    if type(raw_items) is not list:
        raise TypeError("Invalid retained history page: items must be a list.")
    raw_predecessor = data.get("predecessor")
    if raw_predecessor is not None and not isinstance(raw_predecessor, Mapping):
        raise TypeError(
            "Invalid retained history page: predecessor must be an object or null."
        )
    total_count = data.get("total_count")
    if type(total_count) is not int or total_count < 0 or total_count < len(raw_items):
        raise ValueError(
            "Invalid retained history page: total_count is inconsistent with items."
        )
    has_more = data.get("has_more")
    if type(has_more) is not bool:
        raise TypeError("Invalid retained history page: has_more must be a bool.")
    cursor = data.get("next_before_change_id")
    if has_more:
        if not raw_items or raw_predecessor is None or total_count <= len(raw_items):
            raise ValueError(
                "Invalid retained history page: has_more requires items and a predecessor."
            )
        if type(cursor) is not int or cursor <= 0:
            raise ValueError(
                "Invalid retained history page: has_more requires a positive cursor."
            )
    elif raw_predecessor is not None or cursor is not None:
        raise ValueError(
            "Invalid retained history page: final pages cannot carry a predecessor or cursor."
        )

    items = [
        _normalize_prompt_history_row(item, backend=backend, capabilities=capabilities)
        for item in raw_items
    ]
    change_ids = [item["change_id"] for item in items]
    if any(newer <= older for newer, older in zip(change_ids, change_ids[1:])):
        raise ValueError(
            "Invalid retained history page: items must use descending change IDs."
        )
    if has_more and cursor != change_ids[-1]:
        raise ValueError(
            "Invalid retained history page: cursor must identify the last visible item."
        )

    predecessor = (
        _normalize_prompt_history_row(
            raw_predecessor, backend=backend, capabilities=capabilities
        )
        if raw_predecessor is not None
        else None
    )
    if predecessor is not None and predecessor["change_id"] >= change_ids[-1]:
        raise ValueError(
            "Invalid retained history page: predecessor must be older than visible items."
        )
    prompt_uuids = {item["prompt_uuid"] for item in items}
    if predecessor is not None:
        prompt_uuids.add(predecessor["prompt_uuid"])
    if len(prompt_uuids) > 1:
        raise ValueError(
            "Invalid retained history page: all rows must share one Prompt UUID."
        )

    for index, item in enumerate(items):
        older = items[index + 1] if index + 1 < len(items) else predecessor
        _add_history_change_summary(item, older)

    return {
        "items": items,
        "total_count": total_count,
        "has_more": has_more,
        "next_before_change_id": cursor,
    }


def _prepare_retained_snapshot_for_restore(
    record: Any, *, capabilities: PromptSourceCapabilities
) -> dict[str, Any]:
    """Validate one re-resolved retained row and produce ordinary update fields.

    This deliberately starts from the fail-closed retained-history normalizer,
    so preview-only rows cannot reach a local write path.  The caller invokes it
    only after acquiring the database restore transaction's immediate lock.
    """
    page = normalize_prompt_history_page(
        {
            "items": [record],
            "predecessor": None,
            "total_count": 1,
            "has_more": False,
            "next_before_change_id": None,
        },
        backend=capabilities.backend,
        capabilities=capabilities,
    )
    snapshot = page["items"][0]
    if not snapshot["restore_eligible"]:
        raise ValueError(
            snapshot["compatibility_reason"]
            or "Retained snapshot is preview-only and cannot be restored."
        )

    update_data = _history_restore_fields(snapshot)
    raw_payload = record.get("payload") if isinstance(record, Mapping) else None
    durable_prompt_definition = update_data["prompt_definition"]
    if isinstance(raw_payload, Mapping):
        # ``None`` is a distinct durable value from an empty compatibility lane.
        # The history normalizer renders both safely as empty preview text, but
        # restore must preserve the exact stored lane values for no-change.
        for field in ("system_prompt", "user_prompt"):
            if field in raw_payload:
                update_data[field] = raw_payload[field]
        if isinstance(raw_payload.get("prompt_definition"), str):
            durable_prompt_definition = raw_payload["prompt_definition"]
    if snapshot["prompt_format"] == "structured":
        update_data = validate_console_artifact_payload(update_data, capabilities)
        validate_prompt_request_size(update_data, capabilities)
    return {
        "update_data": update_data,
        "keywords": snapshot["keywords"],
        "keywords_captured": snapshot["keywords_captured"],
        "durable_prompt_definition": durable_prompt_definition,
    }


def prepare_retained_snapshot_for_restore(
    record: Any, *, capabilities: PromptSourceCapabilities
) -> dict[str, Any]:
    """Return a restorable snapshot or one bounded validation category."""
    try:
        return _prepare_retained_snapshot_for_restore(
            record,
            capabilities=capabilities,
        )
    except PromptRestoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PromptRestoreError(PromptRestoreErrorCode.VALIDATION) from exc


def normalize_prompt_version_record(record: Any, *, backend: str) -> dict[str, Any]:
    data = _to_plain_dict(record)
    artifact_type, artifact_type_raw = _normalize_read_artifact_type(
        data.get("artifact_type")
    )
    normalized = {
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
        "artifact_type": artifact_type,
        "has_system_prompt": _lane_flag(data, "has_system_prompt", "system_prompt"),
        "has_user_prompt": _lane_flag(data, "has_user_prompt", "user_prompt"),
    }
    if artifact_type_raw is not None:
        normalized["artifact_type_raw"] = artifact_type_raw
        normalized["definition_state"] = "unsupported"
    elif data.get("prompt_definition") is not None:
        decoded = decode_prompt_artifact(data)
        normalized["definition_state"] = decoded.state
        normalized["prompt_definition"] = (
            decoded.raw_definition
            if decoded.raw_definition is not None
            else data.get("prompt_definition")
        )
    return normalized


def normalize_prompt_version_list(
    payload: Any, *, backend: str
) -> list[dict[str, Any]]:
    return [
        normalize_prompt_version_record(item, backend=backend)
        for item in list(payload or [])
    ]


def normalize_prompt_search(payload: Any, *, backend: str) -> list[dict[str, Any]]:
    """Normalize search results without fetching per-row detail records."""
    if isinstance(payload, (list, tuple)):
        items = payload
    else:
        data = _to_plain_dict(payload)
        items = data.get("items", [])
    return [normalize_prompt_record(item, backend=backend) for item in items]


def normalize_prompt_collection_record(record: Any, *, backend: str) -> dict[str, Any]:
    data = _to_plain_dict(record)
    collection_id = data.get("collection_id")
    if collection_id in (None, ""):
        raise ValueError("Prompt collection record must include collection_id.")
    backend_value = str(backend)
    name = data.get("name")
    return {
        "id": f"{backend_value}:prompt_collection:{collection_id}",
        "backend": backend_value,
        "collection_id": int(collection_id),
        "name": name,
        "display_name": data.get("display_name") or name,
        "description": data.get("description"),
        "prompt_ids": list(data.get("prompt_ids") or []),
    }


def normalize_prompt_collection_list(
    payload: Any, *, backend: str, limit: int = 200, offset: int = 0
) -> dict[str, Any]:
    data = _to_plain_dict(payload)
    raw_items = data.get("collections", [])
    return {
        "collections": [
            normalize_prompt_collection_record(item, backend=backend)
            for item in raw_items
        ],
        "limit": int(data.get("limit", limit) or limit),
        "offset": int(data.get("offset", offset) or offset),
        "total": int(data.get("total", len(raw_items)) or 0),
    }
