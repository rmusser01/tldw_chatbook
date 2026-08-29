import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

PROFILE_DIALECT_ID = "urn:tldw:profile-core:json-schema:dialect:1"
PROFILE_SCHEMA_ID = "urn:tldw:profile-core:schema:personal-context:1"
PROFILE_SEMANTIC_VOCABULARY_ID = (
    "urn:tldw:profile-core:json-schema:vocabulary:semantic:1"
)
PROFILE_SEMANTIC_KEYWORD = "x-tldw-profile-semantics"
PROFILE_SEMANTIC_RULES = {
    "canonicalPayloadMaxUtf8Bytes": 16 * 1024,
    "pendingProposalExpiryDays": 90,
    "proposalIdentityAndVersionLinks": "exact-v1",
}


class ProfileSemanticError(ValueError):
    """Raised when structurally valid profile data violates semantic rules."""


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ProfileSemanticError(
            "semantic timestamps must be RFC 3339 values"
        ) from error
    if parsed.tzinfo is None:
        raise ProfileSemanticError("semantic timestamps must be timezone-aware")
    return parsed


def _canonical_payload_size(record: Mapping[str, Any]) -> int:
    payload = dict(record["payload"])
    payload.setdefault("schema_version", 1)
    payload.setdefault("kind", record["kind"])
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _validate_record(record: Mapping[str, Any]) -> None:
    if record.get("payload") is None:
        return
    if (
        _canonical_payload_size(record)
        > PROFILE_SEMANTIC_RULES["canonicalPayloadMaxUtf8Bytes"]
    ):
        raise ProfileSemanticError("payload exceeds 16 KiB canonical UTF-8 limit")


def _validate_proposal(proposal: Mapping[str, Any]) -> None:
    pending = proposal["state"] == "pending"
    operation = proposal["operation"]
    proposed_record = proposal.get("proposed_record")
    if pending:
        created_at = _timestamp(proposal["created_at"])
        expires_at = _timestamp(proposal["expires_at"])
        if expires_at != created_at + timedelta(
            days=PROFILE_SEMANTIC_RULES["pendingProposalExpiryDays"]
        ):
            raise ProfileSemanticError(
                "pending proposal expiry must be exactly 90 days"
            )
    if proposed_record is None:
        return
    _validate_record(proposed_record)
    if proposed_record["profile_id"] != proposal["profile_id"]:
        raise ProfileSemanticError("proposal and proposed record profile IDs differ")
    if proposed_record["scope_id"] != proposal["scope_id"]:
        raise ProfileSemanticError("proposal and proposed record scope IDs differ")
    if operation == "create":
        if proposed_record.get("parent_version_id") is not None:
            raise ProfileSemanticError("create proposal has a parent version")
    elif operation == "update":
        if proposed_record["record_id"] != proposal["target_record_id"]:
            raise ProfileSemanticError("proposal and proposed record IDs differ")
        if proposed_record.get("parent_version_id") != proposal["base_version_id"]:
            raise ProfileSemanticError("proposal base and parent versions differ")


def validate_profile_semantics(value: Mapping[str, Any]) -> None:
    """Validate semantic vocabulary rules after Draft 2020-12 validation.

    This dependency-free reference validator intentionally does not perform
    structural JSON Schema validation. Call a Draft 2020-12 structural
    validator first, then pass the same decoded object here.
    """

    if "proposal_id" in value:
        _validate_proposal(value)
    elif "record_id" in value:
        _validate_record(value)
