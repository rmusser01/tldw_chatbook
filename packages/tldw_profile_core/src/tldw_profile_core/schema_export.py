import json
from pathlib import Path
from typing import Any, Union

from pydantic import TypeAdapter

from .models import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope

CanonicalObject = Union[ProfileManifest, ProfileScope, ProfileRecord, ProfileProposal]
_PAYLOAD_DEFS = {
    "identity": "IdentityPayload",
    "preference": "PreferencePayload",
    "relationship": "RelationshipPayload",
    "correction": "CorrectionPayload",
    "constraint": "ConstraintPayload",
    "goal": "GoalPayload",
    "convention": "ConventionPayload",
    "working_context": "WorkingContextPayload",
    "legacy_unclassified": "LegacyUnclassifiedPayload",
}


def _when(properties: dict[str, Any], then: dict[str, Any]) -> dict[str, Any]:
    return {
        "if": {"properties": properties, "required": list(properties)},
        "then": then,
    }


def _non_null() -> dict[str, Any]:
    return {"not": {"type": "null"}}


def _record_conditionals() -> list[dict[str, Any]]:
    null = {"type": "null"}
    conditionals = [
        _when(
            {"state": {"const": "deleted"}},
            {
                "properties": {
                    "payload": null,
                    "semantic_key": null,
                    "expires_at": null,
                    "no_expiry": {"const": False},
                }
            },
        ),
        _when(
            {"state": {"not": {"const": "deleted"}}},
            {
                "properties": {"payload": _non_null()},
                "required": ["payload"],
            },
        ),
    ]
    conditionals.extend(
        _when(
            {"kind": {"const": kind}},
            {
                "properties": {
                    "payload": {
                        "anyOf": [
                            {"$ref": f"#/$defs/{payload_def}"},
                            null,
                        ]
                    }
                }
            },
        )
        for kind, payload_def in _PAYLOAD_DEFS.items()
    )
    conditionals.extend(
        [
            _when(
                {
                    "state": {"not": {"const": "deleted"}},
                    "kind": {"const": "working_context"},
                },
                {
                    "oneOf": [
                        {
                            "properties": {
                                "expires_at": _non_null(),
                                "no_expiry": {"const": False},
                            },
                            "required": ["expires_at"],
                        },
                        {
                            "properties": {
                                "expires_at": null,
                                "no_expiry": {"const": True},
                            },
                            "required": ["no_expiry"],
                        },
                    ]
                },
            ),
            _when(
                {"kind": {"not": {"const": "working_context"}}},
                {
                    "properties": {
                        "expires_at": null,
                        "no_expiry": {"const": False},
                    }
                },
            ),
        ]
    )
    return conditionals


def _proposal_conditionals() -> list[dict[str, Any]]:
    null = {"type": "null"}
    non_null = _non_null()
    content = {"proposed_record": non_null}
    target = {"target_record_id": non_null, "base_version_id": non_null}
    conditionals = [
        _when(
            {"state": {"const": "pending"}, "operation": {"const": "create"}},
            {
                "properties": {
                    "target_record_id": null,
                    "base_version_id": null,
                    **content,
                }
            },
        ),
        _when(
            {"state": {"const": "pending"}, "operation": {"const": "update"}},
            {"properties": {**target, **content}},
        ),
    ]
    conditionals.extend(
        _when(
            {"state": {"const": "pending"}, "operation": {"const": operation}},
            {"properties": {**target, "proposed_record": null}},
        )
        for operation in ("archive", "promote")
    )
    conditionals.extend(
        [
            _when(
                {"state": {"not": {"const": "pending"}}},
                {
                    "properties": {
                        "proposed_record": null,
                        "confidence": null,
                    }
                },
            ),
            _when(
                {
                    "state": {"not": {"const": "pending"}},
                    "operation": {"const": "create"},
                },
                {
                    "properties": {
                        "target_record_id": null,
                        "base_version_id": null,
                    }
                },
            ),
            _when(
                {
                    "state": {"not": {"const": "pending"}},
                    "operation": {"enum": ["update", "archive", "promote"]},
                },
                {"properties": target},
            ),
        ]
    )
    return conditionals


def export_json_schema(path: Path) -> None:
    schema = TypeAdapter(CanonicalObject).json_schema(ref_template="#/$defs/{model}")
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "tldw Personal Context Profile v1",
            "version": 1,
        }
    )
    schema["$defs"]["ProfileRecord"]["allOf"] = _record_conditionals()
    schema["$defs"]["ProfileProposal"]["allOf"] = _proposal_conditionals()
    path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
