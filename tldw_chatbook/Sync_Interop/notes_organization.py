"""Public Sync-v2 contracts and identities for Notes organization domains."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Never, cast


NOTES_ORGANIZATION_DOMAINS = (
    "notes.keyword",
    "notes.keyword_link",
    "notes.keyword_collection",
    "notes.keyword_collection_link",
    "notes.folder",
    "notes.folder_link",
)

_RESOURCE_DOMAINS = frozenset(
    {"notes.keyword", "notes.keyword_collection", "notes.folder"}
)
_LINK_MEMBERS: dict[str, tuple[str, ...]] = {
    "notes.keyword_link": ("subject_type", "subject_id", "keyword_sync_id"),
    "notes.keyword_collection_link": ("collection_sync_id", "keyword_sync_id"),
    "notes.folder_link": ("note_id", "folder_sync_id"),
}


class NotesOrganizationValidationError(ValueError):
    """Validation failure with a stable Notes organization Sync error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def parse_notes_organization_payload(
    domain: str,
    operation: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalize one Notes organization wire payload."""

    if domain not in NOTES_ORGANIZATION_DOMAINS:
        _fail(
            "notes_organization_domain_invalid",
            f"unsupported Notes organization domain: {domain}",
        )
    if operation not in {"upsert", "tombstone"}:
        _fail(
            "notes_organization_operation_invalid",
            f"unsupported Notes organization operation: {operation}",
        )
    if not isinstance(payload, Mapping):
        _fail("notes_organization_payload_invalid", "payload must be an object")

    if operation == "tombstone" and domain in _RESOURCE_DOMAINS:
        if payload:
            _fail(
                "notes_organization_payload_invalid",
                f"{domain} tombstone payload must be empty",
            )
        return {}

    normalized = _parse_upsert_payload(domain, payload)
    _validate_resource_references(domain, normalized)
    return normalized


def new_organization_sync_id() -> str:
    """Return a canonical lowercase UUIDv4 string."""

    return str(uuid.uuid4())


def validate_resource_sync_id(value: str) -> str:
    """Return a canonical UUIDv4 string or raise a contract error."""

    if not isinstance(value, str):
        _invalid_resource_sync_id()
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        _invalid_resource_sync_id()
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or str(parsed) != value:
        _invalid_resource_sync_id()
    return value


def organization_link_id(domain: str, members: Sequence[str]) -> str:
    """Return the domain-tagged hash of canonical relationship identity JSON."""

    expected_members = _LINK_MEMBERS.get(domain)
    if (
        isinstance(members, (str, bytes))
        or expected_members is None
        or len(members) != len(expected_members)
        or any(not isinstance(member, str) for member in members)
    ):
        _fail(
            "notes_organization_link_identity_invalid",
            f"invalid relationship identity for {domain}",
        )
    canonical = json.dumps(
        {"domain": domain, "members": list(members), "schema_version": 1},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"{domain}:sha256:{hashlib.sha256(canonical).hexdigest()}"


def validate_organization_object_id(
    domain: str,
    object_id: str,
    payload: Mapping[str, object],
) -> None:
    """Validate a resource UUID or recompute a relationship object ID."""

    if domain in _RESOURCE_DOMAINS:
        validate_resource_sync_id(object_id)
        return

    member_fields = _LINK_MEMBERS.get(domain)
    if member_fields is None:
        _fail(
            "notes_organization_domain_invalid",
            f"unsupported Notes organization domain: {domain}",
        )
    members: list[str] = []
    for field_name in member_fields:
        value = payload.get(field_name)
        if not isinstance(value, str):
            _fail(
                "notes_organization_payload_invalid",
                f"{domain} payload requires string identity field {field_name}",
            )
        members.append(value)
    if object_id != organization_link_id(domain, members):
        _fail(
            "notes_organization_link_identity_invalid",
            f"{domain} object_id does not match its identity payload",
        )


def _parse_upsert_payload(
    domain: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if domain == "notes.keyword":
        _require_fields(payload, {"keyword"}, {"keyword"})
        return {"keyword": _bounded_name(payload["keyword"], 100)}
    if domain == "notes.keyword_link":
        fields = {"subject_type", "subject_id", "keyword_sync_id"}
        _require_fields(payload, fields, fields)
        subject_type = payload["subject_type"]
        if subject_type not in {"note", "conversation"}:
            _invalid_payload("subject_type must be note or conversation")
        return {
            "subject_type": subject_type,
            "subject_id": _nonempty_string(payload["subject_id"]),
            "keyword_sync_id": _string(payload["keyword_sync_id"]),
        }
    if domain == "notes.keyword_collection":
        _require_fields(payload, {"name", "parent_sync_id"}, {"name"})
        return {
            "name": _bounded_name(payload["name"], 255),
            "parent_sync_id": _optional_string(payload.get("parent_sync_id")),
        }
    if domain == "notes.keyword_collection_link":
        fields = {"collection_sync_id", "keyword_sync_id"}
        _require_fields(payload, fields, fields)
        return {
            "collection_sync_id": _string(payload["collection_sync_id"]),
            "keyword_sync_id": _string(payload["keyword_sync_id"]),
        }
    if domain == "notes.folder":
        _require_fields(payload, {"name", "parent_sync_id"}, {"name"})
        return {
            "name": _bounded_name(payload["name"], 500),
            "parent_sync_id": _optional_string(payload.get("parent_sync_id")),
        }
    fields = {"note_id", "folder_sync_id"}
    _require_fields(payload, fields, fields)
    return {
        "note_id": _string(payload["note_id"]),
        "folder_sync_id": _string(payload["folder_sync_id"]),
    }


def _require_fields(
    payload: Mapping[str, object],
    allowed: set[str],
    required: set[str],
) -> None:
    if set(payload) - allowed or required - set(payload):
        _invalid_payload("payload fields do not match the domain contract")


def _bounded_name(value: object, maximum: int) -> str:
    normalized = _nonempty_string(value)
    if len(normalized) > maximum:
        _invalid_payload(f"name must contain at most {maximum} characters")
    return normalized


def _nonempty_string(value: object) -> str:
    normalized = _string(value).strip()
    if not normalized:
        _invalid_payload("value must not be empty")
    return normalized


def _string(value: object) -> str:
    if not isinstance(value, str):
        _invalid_payload("value must be a string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _validate_resource_references(domain: str, payload: Mapping[str, object]) -> None:
    if domain == "notes.keyword_link":
        validate_resource_sync_id(cast(str, payload["keyword_sync_id"]))
        if payload["subject_type"] == "note":
            validate_resource_sync_id(cast(str, payload["subject_id"]))
    elif domain in {"notes.keyword_collection", "notes.folder"}:
        if payload["parent_sync_id"] is not None:
            validate_resource_sync_id(cast(str, payload["parent_sync_id"]))
    elif domain == "notes.keyword_collection_link":
        validate_resource_sync_id(cast(str, payload["collection_sync_id"]))
        validate_resource_sync_id(cast(str, payload["keyword_sync_id"]))
    elif domain == "notes.folder_link":
        validate_resource_sync_id(cast(str, payload["note_id"]))
        validate_resource_sync_id(cast(str, payload["folder_sync_id"]))


def _invalid_resource_sync_id() -> Never:
    _fail(
        "notes_organization_resource_sync_id_invalid",
        "resource sync_id must be a canonical UUIDv4 string",
    )


def _invalid_payload(message: str) -> Never:
    _fail("notes_organization_payload_invalid", message)


def _fail(error_code: str, message: str) -> Never:
    raise NotesOrganizationValidationError(error_code, message)


__all__ = [
    "NOTES_ORGANIZATION_DOMAINS",
    "NotesOrganizationValidationError",
    "new_organization_sync_id",
    "organization_link_id",
    "parse_notes_organization_payload",
    "validate_organization_object_id",
    "validate_resource_sync_id",
]
