"""Trajectory import: open a shared trace file as a read-only snapshot.

The consumer side of the ADR-067 export format (task-16320): read a
``tldw-trajectory`` JSON document, validate it through the EXPORT
validator (``Chat/trajectory_export.validate_trajectory_export`` -- the
shared seam, so import and export can never drift apart), and map the
file's sections onto ``Chat/trajectory.derive_trajectory`` inputs to
produce the same ``TrajectorySnapshot`` the live view renders.

Read-only contract (ADR-067 §5)
    Imported traces are ephemeral view data. This module NEVER opens,
    writes, or references the application database -- it holds no DB
    imports at all -- and the snapshot it returns is consumed purely for
    rendering. Nothing here persists imported data back into local
    conversations/messages/sidecar tables.

Purity contract
    No Textual, no widget imports, no DB layer. Stdlib plus the
    projection's own (equally pure) dependencies.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import compare_digest
from typing import Any

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
    derive_trajectory,
)
from tldw_chatbook.Chat.trajectory_export import (
    TRACE_EXPORT_FORMAT,
    TRACE_EXPORT_VERSION,
    TraceExportProfile,
    TrajectoryExportError,
    _has_credential_material,
    _trace_digest,
    validate_trajectory_export,
)

__all__ = [
    "TrajectoryImportError",
    "ImportedTrace",
    "load_imported_trace",
    "load_trajectory_snapshot",
]


class TrajectoryImportError(TrajectoryExportError):
    """A trace file that could not be read, parsed, validated, or mapped.

    Subclasses :class:`TrajectoryExportError` so the shared-validator
    rejections (format marker, version, missing sections) surface as
    import errors too; the message always names the problem file and the
    offending field/section so the user can act on it.
    """


@dataclass(frozen=True, slots=True)
class ImportedTrace:
    """Read-only collaboration state carried beside an imported snapshot."""

    snapshot: TrajectorySnapshot
    manifest: dict[str, Any]
    integrity: dict[str, Any]
    privacy_inventory: dict[str, int]
    operation_event: TrajectoryRecord


def _read_document(source: Path | str | Mapping) -> dict:
    """Read ``source`` into a parsed JSON document.

    ``str`` is treated as a file path (never inline JSON text) so error
    messages can name the file. JSON decode failures surface as
    ``TrajectoryImportError`` with the parser's line/column detail.
    """
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrajectoryImportError(
            f"Cannot read trajectory trace file '{path}': {exc.strerror or exc}"
        ) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrajectoryImportError(
            f"'{path}' is not a valid trajectory trace: not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})"
        ) from exc
    if not isinstance(document, dict):
        raise TrajectoryImportError(
            f"'{path}' is not a valid trajectory trace: top-level document "
            f"must be a JSON object, got {type(document).__name__}"
        )
    return document


def _v1_snapshot(document: Mapping, path: Path | None) -> TrajectorySnapshot:
    """Validate and map the retained ADR-067 version-1 document."""
    try:
        payload = validate_trajectory_export(document)
    except TrajectoryExportError as exc:
        if path is not None:
            raise TrajectoryImportError(f"'{path}': {exc}") from exc
        raise TrajectoryImportError(str(exc)) from exc

    messages = payload["messages"]
    usage_by_id = {
        str(message["id"]): ProviderUsage.from_json(message.get("usage_json"))
        for message in messages
    }
    try:
        return derive_trajectory(
            messages,
            usage_by_id,
            payload["trajectory_rows"],
            payload.get("variants") or (),
            payload["compaction_records"],
            payload.get("active_leaf_message_id"),
        )
    except Exception as exc:  # noqa: BLE001 - mapping boundary; name the file
        where = f"'{path}'" if path is not None else "trace document"
        raise TrajectoryImportError(
            f"{where} passed validation but could not be mapped to a "
            f"trajectory snapshot ({type(exc).__name__}: {exc})"
        ) from exc


_EVENT_REQUIRED = (
    "event_id",
    "seq",
    "source_seq",
    "kind",
    "label",
    "status",
    "conversation_id",
    "turn_id",
    "message_id",
    "actor_kind",
    "actor_id",
    "run_id",
    "parent_event_id",
    "source_event_id",
    "replacement_event_id",
    "observed_at",
    "step_started_at",
    "first_token_at",
    "completed_at",
    "usage",
    "model",
    "provider",
    "content_preview",
    "payload",
    "variants",
    "depth",
    "field_states",
    "sensitivity",
    "field_provenance",
    "redaction_provenance",
)
_REFERENCE_FIELDS = (
    "parent_event_id",
    "source_event_id",
    "replacement_event_id",
)
_IDENTITY_FIELDS = frozenset(
    {
        "event_id",
        "conversation_id",
        "turn_id",
        "message_id",
        "actor_id",
        "run_id",
        *_REFERENCE_FIELDS,
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FIELD_STATES = frozenset(
    {"observed", "redacted", "truncated", "omitted", "not_available", "capture_failed"}
)
_TIMING_FIELDS = (
    "observed_at",
    "step_started_at",
    "first_token_at",
    "completed_at",
)
_MATERIAL_FIELDS = frozenset(
    {"content_preview", "payload", "variants", "model", "provider"}
)
_PRIVACY_KEYS = frozenset(
    {
        "sensitive",
        "redacted",
        "omitted",
        "truncated",
        "included",
        "observed",
        "missing",
        "not_available",
        "capture_failed",
    }
)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _parse_manifest_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrajectoryImportError(
            f"Invalid Trace v2 manifest: '{field}' must be an ISO 8601 datetime"
        ) from exc
    if parsed.tzinfo is None:
        raise TrajectoryImportError(
            f"Invalid Trace v2 manifest: '{field}' must include a timezone"
        )
    return parsed


def _require_v2(
    payload: Mapping,
    key: str,
    expected: type | tuple[type, ...],
) -> Any:
    if key not in payload:
        raise TrajectoryImportError(
            f"Invalid Trace v2 bundle: missing required '{key}' section"
        )
    value = payload[key]
    if not isinstance(value, expected):
        types = expected if isinstance(expected, tuple) else (expected,)
        names = " or ".join(item.__name__ for item in types)
        raise TrajectoryImportError(
            f"Invalid Trace v2 bundle: '{key}' must be {names}, "
            f"got {type(value).__name__}"
        )
    return value


def _validate_v2(document: Mapping) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if document.get("format") != TRACE_EXPORT_FORMAT:
        raise TrajectoryImportError(
            f"Invalid Trace bundle: 'format' must be {TRACE_EXPORT_FORMAT!r}"
        )
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise TrajectoryImportError(
            f"Invalid Trace bundle: 'version' must be an integer, got {version!r}"
        )
    if version > TRACE_EXPORT_VERSION:
        raise TrajectoryImportError(
            f"Unsupported Trace export version {version}: this build reads version "
            f"{TRACE_EXPORT_VERSION}; upgrade the app or request an older export"
        )
    if version != TRACE_EXPORT_VERSION:
        raise TrajectoryImportError(
            f"Invalid Trace bundle: 'version' must be {TRACE_EXPORT_VERSION}, got {version!r}"
        )

    manifest = _require_v2(document, "manifest", dict)
    try:
        TraceExportProfile(manifest.get("profile"))
    except (TypeError, ValueError) as exc:
        raise TrajectoryImportError(
            f"Invalid Trace v2 manifest profile {manifest.get('profile')!r}"
        ) from exc
    if manifest.get("schema_version") != TRACE_EXPORT_VERSION:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: 'schema_version' must be 2"
        )
    if manifest.get("format_version") != TRACE_EXPORT_VERSION:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: 'format_version' must be 2"
        )
    for field, expected in (
        ("exported_at", str),
        ("exported_timestamp", str),
        ("source", dict),
        ("missing_metadata", list),
        ("redaction_provenance", list),
        ("privacy_decisions", list),
        ("export_operation_event_id", str),
        ("integrity_notice", str),
    ):
        if field not in manifest:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest: '{field}' is required"
            )
        if not isinstance(manifest[field], expected):
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest: '{field}' must be {expected.__name__}"
            )
    exported_at = _parse_manifest_timestamp(manifest["exported_at"], "exported_at")
    exported_timestamp = _parse_manifest_timestamp(
        manifest["exported_timestamp"], "exported_timestamp"
    )
    if (
        exported_at != exported_timestamp
        or manifest["exported_at"] != manifest["exported_timestamp"]
    ):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: exported timestamps must match"
        )
    source = manifest["source"]
    if set(source) != {"type", "conversation_ids"} or source.get("type") != (
        "trajectory_snapshot"
    ):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: source must name a trajectory_snapshot "
            "and conversation_ids"
        )
    source_conversation_ids = source.get("conversation_ids")
    if not isinstance(source_conversation_ids, list) or not all(
        isinstance(value, str) and value for value in source_conversation_ids
    ):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: source.conversation_ids must be a list "
            "of non-empty strings"
        )
    if source_conversation_ids != sorted(set(source_conversation_ids)):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: source.conversation_ids must be unique and sorted"
        )
    privacy = manifest.get("privacy_inventory")
    if not isinstance(privacy, dict) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in privacy.values()
    ):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: 'privacy_inventory' must contain "
            "non-negative integer counts"
        )
    missing_privacy_keys = sorted(_PRIVACY_KEYS - privacy.keys())
    if missing_privacy_keys:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: 'privacy_inventory' is missing "
            + ", ".join(missing_privacy_keys)
        )
    unexpected_privacy_keys = sorted(privacy.keys() - _PRIVACY_KEYS)
    if unexpected_privacy_keys:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: 'privacy_inventory' has unsupported fields: "
            + ", ".join(unexpected_privacy_keys)
        )

    raw_events = _require_v2(document, "events", list)
    events: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}]' must be an object"
            )
        for field in _EVENT_REQUIRED:
            if field not in event:
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].{field}' is missing"
                )
        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].event_id' "
                "must be a non-empty string"
            )
        if event_id in ids:
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: duplicate event_id {event_id!r}"
            )
        ids.add(event_id)
        if (
            not isinstance(event["seq"], int)
            or isinstance(event["seq"], bool)
            or event["seq"] < 0
        ):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].seq' must be a "
                "non-negative integer"
            )
        if (
            not isinstance(event["depth"], int)
            or isinstance(event["depth"], bool)
            or event["depth"] < 0
        ):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].depth' must be a "
                "non-negative integer"
            )
        source_seq = event.get("source_seq")
        if source_seq is not None and (
            not isinstance(source_seq, int)
            or isinstance(source_seq, bool)
            or source_seq < 0
        ):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].source_seq' "
                "must be a non-negative integer or null"
            )
        for field in ("kind", "turn_id", "content_preview"):
            if not isinstance(event[field], str):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].{field}' must be a string"
                )
        for field in (
            "conversation_id",
            "message_id",
            "label",
            "status",
            "actor_kind",
            "actor_id",
            "run_id",
            "model",
            "provider",
            "sensitivity",
        ):
            value = event.get(field)
            if value is not None and not isinstance(value, str):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].{field}' "
                    "must be a string or null"
                )
        for field in _TIMING_FIELDS:
            value = event.get(field)
            if value is not None and not _is_number(value):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].{field}' "
                    "must be a finite number or null"
                )
        payload = event["payload"]
        if payload is not None and not isinstance(payload, Mapping):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].payload' "
                "must be an object or null"
            )
        usage = event.get("usage")
        if usage is not None and not isinstance(usage, Mapping):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].usage' "
                "must be an object or null"
            )
        if isinstance(usage, Mapping):
            unknown_usage = sorted(set(usage) - set(_USAGE_FIELDS))
            if unknown_usage:
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].usage' contains "
                    f"unsupported fields: {', '.join(map(str, unknown_usage))}"
                )
            for field in _USAGE_COUNT_FIELDS:
                if field not in usage:
                    continue
                value = usage[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise TrajectoryImportError(
                        f"Invalid Trace v2 bundle: 'events[{index}].usage.{field}' "
                        "must be a non-negative integer"
                    )
            if "transcription_seconds" in usage and (
                not _is_number(usage["transcription_seconds"])
                or usage["transcription_seconds"] < 0
            ):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].usage."
                    "transcription_seconds' must be a finite non-negative number"
                )
            for field in ("provider", "model"):
                if field in usage and not isinstance(usage[field], str):
                    raise TrajectoryImportError(
                        f"Invalid Trace v2 bundle: 'events[{index}].usage.{field}' "
                        "must be a string"
                    )
            if "partial" in usage and not isinstance(usage["partial"], bool):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].usage.partial' "
                    "must be a boolean"
                )
        field_states = event["field_states"]
        if not isinstance(field_states, Mapping):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].field_states' must be an object"
            )
        if not all(
            isinstance(key, str) and isinstance(value, str) and value in _FIELD_STATES
            for key, value in field_states.items()
        ):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].field_states' "
                "contains an unsupported field state"
            )
        field_provenance = event["field_provenance"]
        if not isinstance(field_provenance, Mapping):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].field_provenance' "
                "must be an object"
            )
        for field, detail in field_provenance.items():
            if not isinstance(field, str) or not isinstance(detail, Mapping):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].field_provenance' "
                    "must map field names to objects"
                )
            if (
                detail.get("state") not in _FIELD_STATES
                or not isinstance(detail.get("reason"), str)
                or not isinstance(detail.get("sensitivity"), str)
            ):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].field_provenance.{field}' "
                    "must include valid state, reason, and sensitivity strings"
                )
            if field in field_states and detail.get("state") != field_states[field]:
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].field_provenance."
                    f"{field}.state' does not match field_states.{field}"
                )
        if not isinstance(event["redaction_provenance"], list):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].redaction_provenance' "
                "must be a list"
            )
        if not isinstance(event["variants"], list):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].variants' must be a list"
            )
        if not all(isinstance(value, str) for value in event["variants"]):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}].variants' "
                "must contain only strings"
            )
        for field in _REFERENCE_FIELDS:
            target = event.get(field)
            if target is not None and not isinstance(target, str):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: 'events[{index}].{field}' "
                    "must be a string or null"
                )
        events.append(event)

    if manifest.get("event_count") != len(events):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: 'event_count' does not match events"
        )
    if not events:
        raise TrajectoryImportError(
            "Invalid Trace v2 bundle: events must include a trace_export operation"
        )
    export_operation_event_id = manifest["export_operation_event_id"]
    operation_matches = [
        event for event in events if event["event_id"] == export_operation_event_id
    ]
    if (
        len(operation_matches) != 1
        or events[-1] is not operation_matches[0]
        or operation_matches[0]["kind"] != "trace_export"
        or operation_matches[0]["actor_kind"] != "system"
        or operation_matches[0]["field_states"]
        or operation_matches[0]["field_provenance"]
        or operation_matches[0]["redaction_provenance"]
    ):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: export_operation_event_id must identify "
            "the final ungoverned synthetic trace_export event"
        )
    operation_payload = operation_matches[0]["payload"]
    if not isinstance(operation_payload, Mapping) or operation_payload != {
        "profile": manifest["profile"],
        "privacy_inventory": privacy,
    }:
        raise TrajectoryImportError(
            "Invalid Trace v2 bundle: export operation payload does not match manifest"
        )
    source_events = events[:-1]
    for index, event in enumerate(events):
        for field in _REFERENCE_FIELDS:
            target = event.get(field)
            if target and target not in ids:
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: dangling {field} {target!r} "
                    f"at events[{index}]"
                )
    for index, event in enumerate(source_events):
        field_states = event["field_states"]
        missing_material = sorted(_MATERIAL_FIELDS - set(field_states))
        if missing_material:
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}]' is missing material "
                f"field_states coverage for {', '.join(missing_material)}"
            )
        required_identities = {
            field for field in _IDENTITY_FIELDS if event.get(field) not in {None, ""}
        }
        missing_identities = sorted(required_identities - set(field_states))
        if missing_identities:
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}]' is missing populated "
                f"identity field_states coverage for {', '.join(missing_identities)}"
            )
        if set(event["field_provenance"]) != set(field_states):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'events[{index}]' field_provenance "
                "must exactly match field_states keys"
            )
    expected_conversation_ids = sorted(
        {
            str(event["conversation_id"])
            for event in events
            if event.get("conversation_id")
        }
    )
    if source_conversation_ids != expected_conversation_ids:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: source.conversation_ids does not match events"
        )

    def validate_provenance(entries: list[Any], where: str) -> list[dict[str, str]]:
        validated: list[dict[str, str]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 {where}[{index}]: expected an object"
                )
            event_id = entry.get("event_id")
            field = entry.get("field")
            state = entry.get("state")
            reason = entry.get("reason")
            if (
                event_id not in ids
                or not isinstance(field, str)
                or not field
                or state not in _FIELD_STATES - {"observed"}
                or not isinstance(reason, str)
                or not reason
            ):
                raise TrajectoryImportError(
                    f"Invalid Trace v2 {where}[{index}]: expected existing event_id "
                    "and non-observed field/state/reason strings"
                )
            validated.append(
                {
                    "event_id": str(event_id),
                    "field": field,
                    "state": str(state),
                    "reason": reason,
                }
            )
        return validated

    event_provenance = [
        entry for event in events for entry in event["redaction_provenance"]
    ]
    validated_event_provenance = validate_provenance(
        event_provenance, "event redaction_provenance"
    )
    validated_manifest_provenance = validate_provenance(
        manifest["redaction_provenance"], "manifest redaction_provenance"
    )
    if validated_manifest_provenance != validated_event_provenance:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: redaction_provenance does not match events"
        )

    event_by_id = {str(event["event_id"]): event for event in events}
    privacy_decisions = manifest["privacy_decisions"]
    decision_keys: set[tuple[str, str]] = set()
    validated_decisions: list[dict[str, Any]] = []
    required_decision_fields = {
        "event_id",
        "field",
        "state",
        "reason",
        "sensitive",
        "source_state",
    }
    for index, decision in enumerate(privacy_decisions):
        if not isinstance(decision, Mapping) or set(decision) != (
            required_decision_fields
        ):
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest privacy_decisions[{index}]: expected "
                "the complete decision schema"
            )
        event_id = decision["event_id"]
        field = decision["field"]
        state = decision["state"]
        source_state = decision["source_state"]
        if (
            event_id not in event_by_id
            or not isinstance(field, str)
            or not field
            or state not in _FIELD_STATES
            or source_state not in _FIELD_STATES
            or not isinstance(decision["reason"], str)
            or not decision["reason"]
            or not isinstance(decision["sensitive"], bool)
        ):
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest privacy_decisions[{index}] values"
            )
        key = (str(event_id), field)
        if key in decision_keys:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest privacy_decisions: duplicate {key!r}"
            )
        decision_keys.add(key)
        event = event_by_id[str(event_id)]
        field_states = event["field_states"]
        if field in field_states and field_states[field] != state:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest privacy_decisions[{index}]: state "
                f"does not match event field_states.{field}"
            )
        detail = event["field_provenance"].get(field)
        if not isinstance(detail, Mapping) or detail.get("state") != state:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest privacy_decisions[{index}]: state "
                f"does not match event field_provenance.{field}"
            )
        expected_classification = event.get("sensitivity") or "unspecified"
        if detail.get("sensitivity") != expected_classification:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest privacy_decisions[{index}]: sensitivity "
                f"classification does not match event field_provenance.{field}"
            )
        expected_sensitive = expected_classification != "unspecified" or state in {
            "redacted",
            "truncated",
            "omitted",
        }
        if decision["sensitive"] is not expected_sensitive:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest privacy_decisions[{index}]: sensitive "
                "must match field classification and privacy state"
            )
        validated_decisions.append(dict(decision))

    expected_decision_keys = {
        (str(event["event_id"]), str(field))
        for event in source_events
        for field in event["field_states"]
    }
    if decision_keys != expected_decision_keys:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: privacy_decisions keys must exactly match "
            "source event field_states"
        )

    expected_decision_counts = {
        "sensitive": sum(
            bool(decision["sensitive"]) for decision in validated_decisions
        ),
        "included": sum(
            decision["state"] not in {"not_available", "capture_failed", "omitted"}
            for decision in validated_decisions
        ),
        "observed": sum(
            decision["source_state"] == "observed" for decision in validated_decisions
        ),
    }
    for field, expected_count in expected_decision_counts.items():
        if privacy[field] != expected_count:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest: privacy_inventory.{field} "
                "does not match privacy_decisions"
            )

    provenance_roots = {
        (
            entry["event_id"],
            re.split(r"[.[]", entry["field"], maxsplit=1)[0],
            entry["state"],
        )
        for entry in validated_event_provenance
    }
    for event in events:
        for field, state in event["field_states"].items():
            if state == "observed":
                continue
            detail = event["field_provenance"].get(field)
            if not isinstance(detail, Mapping) or detail.get("state") != state:
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: event {event['event_id']!r} "
                    f"field_provenance.{field} does not match field_states"
                )
            if (str(event["event_id"]), str(field), str(state)) not in provenance_roots:
                raise TrajectoryImportError(
                    f"Invalid Trace v2 bundle: event {event['event_id']!r} "
                    f"field {field!r} lacks matching redaction provenance"
                )

    expected_missing = [
        {"event_id": str(event["event_id"]), "field": str(field), "state": str(state)}
        for event in events
        for field, state in event["field_states"].items()
        if state in {"not_available", "capture_failed"}
    ]
    missing_metadata = manifest["missing_metadata"]
    if not all(isinstance(entry, Mapping) for entry in missing_metadata):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: 'missing_metadata' entries must be objects"
        )
    if [dict(entry) for entry in missing_metadata] != expected_missing:
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: missing_metadata does not match event field states"
        )

    privacy_actions = {
        (
            entry["event_id"],
            re.split(r"[.[]", entry["field"], maxsplit=1)[0],
            entry["state"],
        )
        for entry in validated_manifest_provenance
    }
    for state in ("redacted", "omitted", "truncated"):
        expected_count = sum(action[2] == state for action in privacy_actions)
        if privacy[state] != expected_count:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest: privacy_inventory.{state} "
                "does not match redaction provenance"
            )
    for state in ("not_available", "capture_failed"):
        expected_count = sum(entry["state"] == state for entry in expected_missing)
        if privacy[state] != expected_count:
            raise TrajectoryImportError(
                f"Invalid Trace v2 manifest: privacy_inventory.{state} "
                "does not match missing_metadata"
            )
    if privacy["missing"] != len(expected_missing):
        raise TrajectoryImportError(
            "Invalid Trace v2 manifest: privacy_inventory.missing "
            "does not match missing_metadata"
        )

    lineage = _require_v2(document, "lineage", list)
    allowed_relationships = {
        field.removesuffix("_event_id") for field in _REFERENCE_FIELDS
    }
    for index, edge in enumerate(lineage):
        if not isinstance(edge, Mapping):
            raise TrajectoryImportError(
                f"Invalid Trace v2 bundle: 'lineage[{index}]' must be an object"
            )
        source = edge.get("source")
        target = edge.get("target")
        relationship = edge.get("relationship")
        if source not in ids or target not in ids:
            raise TrajectoryImportError(
                f"Invalid Trace v2 lineage[{index}]: source and target must name "
                "existing event IDs"
            )
        if relationship not in allowed_relationships:
            raise TrajectoryImportError(
                f"Invalid Trace v2 lineage[{index}]: unsupported relationship "
                f"{relationship!r}"
            )
    expected_lineage = [
        {
            "source": str(event["event_id"]),
            "target": str(event[field]),
            "relationship": field.removesuffix("_event_id"),
        }
        for event in events
        for field in _REFERENCE_FIELDS
        if event.get(field)
    ]
    if [dict(edge) for edge in lineage] != expected_lineage:
        raise TrajectoryImportError(
            "Invalid Trace v2 bundle: lineage does not match events reference fields"
        )
    integrity = _require_v2(document, "integrity", dict)
    if integrity.get("algorithm") != "sha256":
        raise TrajectoryImportError(
            "Invalid Trace v2 integrity: 'algorithm' must be 'sha256'"
        )
    if integrity.get("authenticity") is not False:
        raise TrajectoryImportError(
            "Invalid Trace v2 integrity: 'authenticity' must be false because "
            "SHA-256 verifies integrity, not authorship"
        )
    digest = integrity.get("digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise TrajectoryImportError(
            "Invalid Trace v2 integrity digest: expected 64 lowercase hex characters"
        )
    expected = _trace_digest(document)
    if not compare_digest(digest, expected):
        raise TrajectoryImportError(
            "Trace v2 integrity digest mismatch: the file is corrupted or was tampered with; "
            "SHA-256 does not establish authenticity"
        )
    if _has_credential_material(document):
        raise TrajectoryImportError(
            "Trace v2 import blocked: credentials are forbidden in collaboration "
            "bundles; request a privacy-safe export"
        )
    return dict(manifest), events


_USAGE_FIELDS = (
    "uncached_input",
    "cache_read",
    "cache_write",
    "output",
    "audio_input",
    "audio_output",
    "transcription_seconds",
    "provider",
    "model",
    "partial",
)
_USAGE_COUNT_FIELDS = frozenset(
    {
        "uncached_input",
        "cache_read",
        "cache_write",
        "output",
        "audio_input",
        "audio_output",
    }
)


def _event_record(event: Mapping[str, Any]) -> TrajectoryRecord:
    usage_data = event.get("usage")
    usage = (
        ProviderUsage(
            **{
                field: usage_data[field]
                for field in _USAGE_FIELDS
                if field in usage_data
            }
        )
        if isinstance(usage_data, Mapping)
        else None
    )
    payload = event.get("payload")
    if payload is not None and not isinstance(payload, Mapping):
        raise TrajectoryImportError(
            f"Invalid Trace v2 event {event['event_id']!r}: 'payload' must be an object or null"
        )
    return TrajectoryRecord(
        seq=int(event["seq"]),
        kind=str(event["kind"]),
        turn_id=str(event["turn_id"]),
        message_id=(
            str(event["message_id"]) if event.get("message_id") is not None else None
        ),
        content_preview=str(event["content_preview"] or ""),
        usage=usage,
        step_started_at=event.get("step_started_at"),
        first_token_at=event.get("first_token_at"),
        completed_at=event.get("completed_at"),
        model=str(event["model"]) if event.get("model") is not None else None,
        provider=str(event["provider"]) if event.get("provider") is not None else None,
        payload=dict(payload) if payload is not None else None,
        variants=tuple(str(value) for value in event["variants"]),
        depth=int(event["depth"]),
        event_id=str(event["event_id"]),
        conversation_id=(
            str(event["conversation_id"])
            if event.get("conversation_id") is not None
            else None
        ),
        source_seq=(
            int(event["source_seq"]) if event.get("source_seq") is not None else None
        ),
        label=str(event.get("label") or ""),
        status=str(event["status"]) if event.get("status") is not None else None,
        actor_kind=(
            str(event["actor_kind"]) if event.get("actor_kind") is not None else None
        ),
        actor_id=str(event["actor_id"]) if event.get("actor_id") is not None else None,
        run_id=str(event["run_id"]) if event.get("run_id") is not None else None,
        parent_event_id=(
            str(event["parent_event_id"])
            if event.get("parent_event_id") is not None
            else None
        ),
        source_event_id=(
            str(event["source_event_id"])
            if event.get("source_event_id") is not None
            else None
        ),
        replacement_event_id=(
            str(event["replacement_event_id"])
            if event.get("replacement_event_id") is not None
            else None
        ),
        observed_at=event.get("observed_at"),
        field_states={
            str(key): str(value) for key, value in event["field_states"].items()
        },
        sensitivity=(
            str(event["sensitivity"]) if event.get("sensitivity") is not None else None
        ),
    )


def _snapshot_from_events(events: Sequence[Mapping[str, Any]]) -> TrajectorySnapshot:
    turns: list[TrajectoryTurn] = []
    turn_id: str | None = None
    records: list[TrajectoryRecord] = []
    for event in events:
        record = _event_record(event)
        if turn_id is not None and record.turn_id != turn_id:
            turns.append(TrajectoryTurn(turn_id, tuple(records)))
            records = []
        turn_id = record.turn_id
        records.append(record)
    if turn_id is not None:
        turns.append(TrajectoryTurn(turn_id, tuple(records)))
    return TrajectorySnapshot(tuple(turns))


def _import_operation(
    manifest: Mapping[str, Any],
    integrity: Mapping[str, Any],
    privacy_inventory: Mapping[str, int],
    event_count: int,
) -> TrajectoryRecord:
    digest = str(integrity.get("digest") or "legacy")
    return TrajectoryRecord(
        seq=event_count + 1,
        kind="trace_import",
        turn_id="trace",
        message_id=None,
        content_preview="Read-only shared Trace imported",
        usage=None,
        step_started_at=None,
        first_token_at=None,
        completed_at=None,
        model=None,
        provider=None,
        payload={
            "manifest": dict(manifest),
            "integrity": dict(integrity),
            "privacy_inventory": dict(privacy_inventory),
        },
        variants=(),
        depth=0,
        event_id=f"trace_import:{digest[:16]}",
        label="Trace import",
        status="complete",
        actor_kind="system",
        actor_id="trace",
        field_states={"payload": "observed"},
        sensitivity="diagnostic",
    )


def load_imported_trace(source: Path | str | Mapping) -> ImportedTrace:
    """Load v1 or v2 collaboration data without persistence side effects."""
    path = None if isinstance(source, Mapping) else Path(str(source))
    document = _read_document(source)
    if document.get("format") == TRACE_EXPORT_FORMAT:
        try:
            manifest, events = _validate_v2(document)
            snapshot = _snapshot_from_events(events)
        except TrajectoryImportError as exc:
            if path is not None:
                raise TrajectoryImportError(f"'{path}': {exc}") from exc
            raise
        integrity = {
            "algorithm": "sha256",
            "digest": document["integrity"]["digest"],
            "authenticity": False,
            "verified": True,
            "verdict": "valid",
            "notice": (
                "SHA-256 digest valid; source authenticity not established."
            ),
        }
        privacy = dict(manifest["privacy_inventory"])
    else:
        snapshot = _v1_snapshot(document, path)
        event_count = sum(len(turn.records) for turn in snapshot.turns)
        manifest = {
            "schema_version": 1,
            "format_version": 1,
            "profile": "legacy_v1",
            "event_count": event_count,
            "privacy_inventory": {},
        }
        integrity = {
            "algorithm": None,
            "digest": None,
            "authenticity": False,
            "verified": False,
            "verdict": "not_provided_v1",
            "notice": (
                "No integrity digest provided; source authenticity not established."
            ),
        }
        privacy = {}
    operation = _import_operation(
        manifest,
        integrity,
        privacy,
        sum(len(turn.records) for turn in snapshot.turns),
    )
    return ImportedTrace(snapshot, manifest, integrity, privacy, operation)


def load_trajectory_snapshot(source: Path | str | Mapping) -> TrajectorySnapshot:
    """Load a v1 or v2 shared Trace into the compatible snapshot result."""
    return load_imported_trace(source).snapshot
