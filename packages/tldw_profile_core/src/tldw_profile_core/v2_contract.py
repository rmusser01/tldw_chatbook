"""Explicit inactive V2 validation and validated serialization; no native use."""

import hmac
import json
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from pydantic import TypeAdapter

from .canonical import PORTABLE_DATETIME_PATTERN, canonical_bytes
from .schema_export import _record_conditionals, _when
from .v2_models import (
    ProfileManifestV2,
    ProfileProposalV2,
    ProfileRecordV2,
    _decode_json,
    _snapshot,
)

V2Object = ProfileManifestV2 | ProfileRecordV2 | ProfileProposalV2
_MODELS = (ProfileManifestV2, ProfileRecordV2, ProfileProposalV2)


def validate_v2_object(value: object) -> V2Object:
    """Validate complete structure/semantics, without attesting user/source authority.

    Args:
        value: Exact aggregate or supported built-in decoded object.

    Returns:
        A freshly validated explicit V2 aggregate.

    Raises:
        ValueError: Any malformed/unsafe/unsupported aggregate, without input content.
    """
    try:
        snapshot = _snapshot(value)
        if type(snapshot) is not dict:
            raise ValueError
        selectors = [
            k for k in ("proposal_id", "record_id", "revision") if k in snapshot
        ]
        if len(selectors) != 1:
            raise ValueError
        model = {
            "proposal_id": ProfileProposalV2,
            "record_id": ProfileRecordV2,
            "revision": ProfileManifestV2,
        }[selectors[0]]
        return model.model_validate(snapshot)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ValueError("invalid V2 profile object") from None


def validate_v2_json(json_data: str | bytes | bytearray) -> V2Object:
    """Decode duplicate-aware JSON then validate V2 structure and semantics.

    Args:
        json_data: Built-in JSON text or UTF-8 bytes for one complete aggregate.

    Returns:
        A freshly validated explicit V2 aggregate, without source authority.

    Raises:
        ValueError: The JSON or aggregate is malformed or unsupported.
    """
    try:
        return validate_v2_object(_decode_json(json_data))
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ValueError("invalid V2 profile JSON") from None


def canonical_v2_bytes(value: V2Object) -> bytes:
    """Return JCS bytes only after fresh exact complete aggregate validation.

    Args:
        value: An exact V2 manifest, record, or proposal instance.

    Returns:
        RFC 8785 canonical UTF-8 bytes for the complete validated aggregate.

    Raises:
        TypeError: The input is not an exact supported aggregate class.
        ValueError: Stored aggregate fields fail fresh validation.
    """
    if not any(type(value) is model for model in _MODELS):
        raise TypeError("V2 serialization requires an exact aggregate class")
    return canonical_bytes(validate_v2_object(value))


def v2_object_digest(value: V2Object) -> str:
    """Return SHA-256 of the freshly validated complete canonical aggregate.

    Args:
        value: An exact V2 manifest, record, or proposal instance.

    Returns:
        A lowercase hexadecimal SHA-256 digest of the canonical bytes.

    Raises:
        TypeError: The input is not an exact supported aggregate class.
        ValueError: Stored aggregate fields fail fresh validation.
    """
    return sha256(canonical_v2_bytes(value)).hexdigest()


def v2_integrity_tag(value: V2Object, key: bytes) -> str:
    """Return keyed byte integrity; this never proves authorship or permission.

    Args:
        value: An exact V2 manifest, record, or proposal instance.
        key: Exactly 32 built-in bytes of integrity key material.

    Returns:
        The versioned HMAC-SHA-256 tag for the validated canonical bytes.

    Raises:
        TypeError: The input is not an exact supported aggregate class.
        ValueError: The key or stored aggregate fields are invalid.
    """
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("integrity key must be exactly 32 built-in bytes")
    return (
        "hmac-sha256-v1:" + hmac.new(key, canonical_v2_bytes(value), sha256).hexdigest()
    )


# Separate required dialect: importing this module never changes V1 dispatch.
PROFILE_V2_SCHEMA_ID = "urn:tldw:profile-core:schema:personal-context:2"
PROFILE_V2_DIALECT_ID = "urn:tldw:profile-core:json-schema:dialect:2"
PROFILE_V2_VOCABULARY_ID = "urn:tldw:profile-core:json-schema:vocabulary:semantic:2"
PROFILE_V2_SEMANTIC_KEYWORD = "x-tldw-profile-semantics"
PROFILE_V2_RULES = MappingProxyType(
    {
        "canonicalization": "rfc8785-v1",
        "canonicalDateTime": "utc-milliseconds-v1",
        "iJsonMaxSafeInteger": 9007199254740991,
        "canonicalPayloadMaxUtf8Bytes": 16384,
        "canonicalClaimMaxUtf8Bytes": 16384,
        "canonicalRecordMaxUtf8Bytes": 65536,
        "canonicalProposalMaxUtf8Bytes": 98304,
        "canonicalManifestMaxUtf8Bytes": 16384,
        "pendingProposalExpiryDays": 90,
        "aggregateRules": "profile-aggregates-v2",
        "claimProjection": "profile-claim-v2",
        "bindingProjection": "owner-version-evidence-binding-v1",
        "attributionRules": "claim-attribution-v2",
        "temporalRelations": "profile-relations-v2",
        "disclosureRules": "profile-disclosure-ceiling-v2",
        "manifestRequirements": "profile-context-requirements-v2",
    }
)


def _write_schema(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def export_v2_json_schema(path: Path) -> None:
    """Write structural V2 schema; processors also require the V2 semantics."""
    schema = TypeAdapter(V2Object).json_schema(ref_template="#/$defs/{model}")
    schema.update(
        {
            "$id": PROFILE_V2_SCHEMA_ID,
            "$schema": PROFILE_V2_DIALECT_ID,
            "title": "tldw Personal Context Profile v2 (inactive data contract)",
            "version": 2,
            "$comment": "Structural Draft 2020-12 validation alone is insufficient; the required V2 semantic vocabulary and explicit reference validator enforce canonical byte, digest, attribution, ordering and scalar rules. Data validation never qualifies native admission or authority.",
            PROFILE_V2_SEMANTIC_KEYWORD: dict(PROFILE_V2_RULES),
        }
    )

    # CaptureTime describes decoded Python input too; the wire is portable text only.
    def wire(node: object) -> None:
        if type(node) is dict:
            if node.get("type") == "string" and node.get("format") == "date-time":
                node["pattern"] = PORTABLE_DATETIME_PATTERN.pattern
            for child in node.values():
                wire(child)
        elif type(node) is list:
            for child in node:
                wire(child)

    wire(schema)
    null = {"type": "null"}
    non_null = {"not": null}
    record_rules = _record_conditionals()
    record_rules.extend(
        [
            _when(
                {"state": {"const": "deleted"}},
                {
                    "properties": {
                        "provenance": null,
                        "claim": null,
                        "controls": {
                            "properties": {
                                "model_disclosure": {
                                    "properties": {"kind": {"const": "deny"}}
                                }
                            }
                        },
                    }
                },
            ),
            _when(
                {"state": {"not": {"const": "deleted"}}},
                {"properties": {"provenance": non_null, "claim": non_null}},
            ),
        ]
    )
    # V1's payload refs are preserved, only the containing record is V2.
    schema["$defs"]["ProfileRecordV2"]["allOf"] = record_rules
    target = {"target_record_id": non_null, "base_version_id": non_null}
    proposal_rules = [
        _when(
            {"state": {"const": "pending"}}, {"properties": {"provenance": non_null}}
        ),
        _when(
            {"state": {"not": {"const": "pending"}}},
            {"properties": {"proposed_record": null, "provenance": null}},
        ),
        _when(
            {"operation": {"const": "create"}},
            {"properties": {"target_record_id": null, "base_version_id": null}},
        ),
        _when(
            {"operation": {"enum": ["update", "archive", "promote"]}},
            {"properties": target},
        ),
        _when(
            {"operation": {"enum": ["archive", "promote"]}},
            {"properties": {"proposed_record": null}},
        ),
        _when(
            {
                "state": {"const": "pending"},
                "operation": {"enum": ["create", "update"]},
            },
            {
                "properties": {
                    "proposed_record": {
                        "allOf": [
                            non_null,
                            {
                                "properties": {
                                    "state": {"const": "active"},
                                    "claim": {"properties": {"approval_receipt": null}},
                                }
                            },
                        ]
                    },
                }
            },
        ),
    ]
    schema["$defs"]["ProfileProposalV2"]["allOf"] = proposal_rules
    _write_schema(path, schema)


def export_v2_meta_schema(path: Path) -> None:
    """Write a dialect requiring every frozen V2 semantic rule explicitly."""
    draft = "https://json-schema.org/draft/2020-12"
    rules = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            key: {"const": value} for key, value in PROFILE_V2_RULES.items()
        },
        "required": list(PROFILE_V2_RULES),
    }
    _write_schema(
        path,
        {
            "$schema": draft + "/schema",
            "$id": PROFILE_V2_DIALECT_ID,
            "$dynamicAnchor": "meta",
            "$vocabulary": {
                **{
                    draft + "/vocab/" + name: True
                    for name in (
                        "core",
                        "applicator",
                        "unevaluated",
                        "validation",
                        "meta-data",
                        "format-annotation",
                        "content",
                    )
                },
                PROFILE_V2_VOCABULARY_ID: True,
            },
            "title": "tldw Personal Context Profile v2 required semantic dialect",
            "description": "Structural Draft 2020-12 plus mandatory complete V2 semantic processing; no native capability attestation.",
            "allOf": [
                {"$ref": draft + "/schema"},
                _when(
                    {"$schema": {"const": PROFILE_V2_DIALECT_ID}},
                    {"required": [PROFILE_V2_SEMANTIC_KEYWORD]},
                ),
            ],
            "properties": {PROFILE_V2_SEMANTIC_KEYWORD: rules},
        },
    )
