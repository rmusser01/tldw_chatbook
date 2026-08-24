"""Trajectory export: versioned JSON trace format, writer, and validator.

Implements ADR-067 (``backlog/decisions/067-trajectory-export-format.md``):
one self-contained JSON document per conversation carrying everything the
trajectory projection (``Chat/trajectory.derive_trajectory``) needs to
render the exact TrajectoryScreen view -- messages with usage, the
schema-v38 sidecar rows, compaction attempts, and (when exported from a
live session) variant sets.

Purity contract
    No Textual, no widget imports. The only DB touch is ``build_...``
    reading through the ``CharactersRAGDB`` accessors named below; the
    writer and validator are pure stdlib. The validator is the import
    seam (task-16320), so export and import can never drift apart.

Privacy contract (ADR-067 §3/§4)
    Tool ``payload_json`` may contain file contents, so redaction is the
    DEFAULT: tool rows get preview-only payload stubs unless the caller
    passes ``include_payloads=True``. The document-level ``redacted``
    flag records the mode. The file never carries API keys, config, or
    provider credentials -- only the conversation fields listed in
    ``_MESSAGE_KEYS`` / ``_TRAJECTORY_ROW_KEYS``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .assistant_generation_state import normalize_assistant_generation_state
from .library_preparation import (
    LIBRARY_PREPARATION_EVENT_KIND,
    LibraryPreparationValidationError,
    decode_library_preparation_event,
    encode_library_preparation_event,
)
from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    redact_local_paths,
)

__all__ = [
    "TRAJECTORY_EXPORT_FORMAT",
    "TRAJECTORY_EXPORT_VERSION",
    "PREVIEW_MAX_CHARS",
    "TRACE_EXPORT_FORMAT",
    "TRACE_EXPORT_VERSION",
    "TraceExportProfile",
    "TraceFieldDecision",
    "TraceExportPreflight",
    "TrajectoryExportError",
    "preflight_trace_export",
    "build_trace_export",
    "build_trajectory_export",
    "validate_trajectory_export",
    "write_trajectory_export",
]

#: Document format marker (ADR-067 §1).
TRAJECTORY_EXPORT_FORMAT = "tldw-trajectory"

#: Current format version; a public contract (ADR-067 §2).
TRAJECTORY_EXPORT_VERSION = 1

#: Cap for redacted payload previews, matching the projection's cap.
PREVIEW_MAX_CHARS = 120

TRACE_EXPORT_FORMAT = "tldw-trace"
TRACE_EXPORT_VERSION = 2

_TOOL_KINDS = frozenset({"tool_call", "tool_result"})
_COMPACTION_PURPOSE = "conversation_compaction"

#: Exported per-message fields; image blobs are omitted entirely (ADR-067 §1).
_MESSAGE_KEYS = (
    "id",
    "sender",
    "content",
    "timestamp",
    "parent_message_id",
    "usage_json",
    "assistant_generation_state",
)

#: Exported sidecar fields, mirroring ``TrajectoryRowRead``.
_TRAJECTORY_ROW_KEYS = (
    "message_id",
    "conversation_id",
    "turn_id",
    "seq",
    "event_kind",
    "step_started_at",
    "first_token_at",
    "completed_at",
    "model",
    "provider",
    "payload_json",
)

#: Required keys on each exported sidecar row (others may be ``None``).
_REQUIRED_ROW_KEYS = ("message_id", "conversation_id", "turn_id", "seq", "event_kind")

#: Upper bound for the one-shot message read (single-file export).
_MESSAGE_READ_LIMIT = 1_000_000

#: Matches the repository's maximum ``list_auxiliary_attempts`` page.
_AUX_ATTEMPT_LIMIT = 500


class TrajectoryExportError(Exception):
    """Unknown conversation, or an export payload that fails validation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object (rows and models alike)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(obj: Any) -> dict:
    """Convert a row/model to a plain dict (``TrajectoryRowRead``-shaped)."""
    if isinstance(obj, Mapping):
        return dict(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return dict(obj)


def _single_line(value: Any, limit: int = PREVIEW_MAX_CHARS) -> str:
    """Collapse to one line and cap at ``limit`` chars."""
    return " ".join(str(value or "").split())[:limit]


def _redacted_payload_json(payload_json: str | None) -> str | None:
    """Build the preview-only replacement for a tool ``payload_json``.

    Keeps the ``name`` plus first-120-char single-line previews of the
    result and args; anything unparseable degrades to empty previews
    rather than leaking raw text. Returns JSON text matching the stub
    shape from ADR-067 §3.
    """
    data: dict = {}
    if payload_json:
        try:
            parsed = json.loads(payload_json)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            data = parsed
    args = data.get("args")
    if args is None:
        args_preview = None
    else:
        args_text = (
            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        )
        args_preview = _single_line(args_text)
    stub = {
        "name": str(data.get("name") or ""),
        "result_preview": _single_line(data.get("result")),
        "args_preview": args_preview,
        "redacted": True,
    }
    return json.dumps(stub, ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    """Coerce DB driver values to JSON-native ones (datetime -> ISO string)."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _serialize_variant_set(variant_set: Any) -> dict:
    """Serialize one variant set (``ConsoleVariantSet``-shaped) to JSON data."""
    variants = []
    for item in _field(variant_set, "variants") or ():
        if isinstance(item, str):
            variants.append(item)
        else:
            variants.append(str(_field(item, "content") or ""))
    selected = _field(variant_set, "selected_index", 0)
    return {
        "turn_id": str(_field(variant_set, "turn_id") or ""),
        "variants": variants,
        "selected_index": int(selected) if selected is not None else 0,
    }


# ---------------------------------------------------------------------------
# Trace v2 (pure snapshot -> privacy-governed collaboration bundle)
# ---------------------------------------------------------------------------


class TraceExportProfile(str, Enum):
    """Privacy policy applied to a Trace v2 collaboration bundle."""

    SAFE_SUMMARY = "safe_summary"
    REDACTED_DIAGNOSTIC = "redacted_diagnostic"
    FULL_TRACE = "full_trace"


@dataclasses.dataclass(frozen=True, slots=True)
class TraceFieldDecision:
    """One preflight decision for one export-governed event field."""

    event_id: str
    field: str
    state: str
    reason: str
    sensitive: bool
    source_state: str = "observed"


@dataclasses.dataclass(frozen=True, slots=True)
class TraceExportPreflight:
    """Prepared v2 events and their single-pass privacy inventory."""

    profile: TraceExportProfile
    event_count: int
    privacy_inventory: dict[str, int]
    field_decisions: tuple[TraceFieldDecision, ...]
    prepared_events: tuple[dict[str, Any], ...]
    redaction_provenance: tuple[dict[str, str], ...]
    source_event_ids: tuple[str, ...]


_MATERIAL_FIELDS = ("content_preview", "payload", "variants", "model", "provider")
_TIMING_FIELDS = (
    "observed_at",
    "step_started_at",
    "first_token_at",
    "completed_at",
)
_IDENTITY_DOMAINS = {
    "event_id": "event",
    "conversation_id": "conversation",
    "turn_id": "turn",
    "message_id": "message",
    "actor_id": "actor",
    "run_id": "run",
    "parent_event_id": "event",
    "source_event_id": "event",
    "replacement_event_id": "event",
}
_MISSING_STATES = frozenset({"not_available", "capture_failed"})
_NON_OBSERVED_STATES = frozenset(
    {"redacted", "truncated", "omitted", "not_available", "capture_failed"}
)
_CREDENTIAL_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|private[_-]?key|authorization|auth|access[_-]?token|"
    r"refresh[_-]?token|token|password|passwd|secret|credential)(?:$|[_-])",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE_RES = (
    re.compile(
        r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?"
        r"-----END [^-\n]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:api[_ -]?key|token|password|passwd|secret)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)
_CONTENT_KEYS = frozenset(
    {"args", "arguments", "body", "content", "input", "output", "prompt", "result"}
)
_CREDENTIAL_SAFE_KEYS = frozenset({"first_token_at"})
_CREDENTIAL_REDACTION_VALUES = frozenset({"[credential redacted]"})
_STRUCTURAL_METADATA_KEYS = frozenset(
    {"field_states", "field_provenance", "redaction_provenance"}
)
_IDENTIFIER_KEY_RE = re.compile(r"(?:^|[_-])(?:id|identifier|uuid)s?$", re.IGNORECASE)


def _profile(value: TraceExportProfile | str) -> TraceExportProfile:
    try:
        return TraceExportProfile(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(profile.value for profile in TraceExportProfile)
        raise TrajectoryExportError(
            f"Invalid Trace export profile {value!r}; choose one of: {allowed}"
        ) from exc


def _credential_text(value: str) -> tuple[str, bool]:
    redacted = value
    for pattern in _CREDENTIAL_VALUE_RES:
        redacted = pattern.sub("[credential redacted]", redacted)
    return redacted, redacted != value


def _has_credential(value: Any, key: str = "") -> bool:
    if (
        key
        and key.lower() not in _CREDENTIAL_SAFE_KEYS
        and (_CREDENTIAL_KEY_RE.search(key) or _credential_text(key)[1])
    ):
        return True
    if isinstance(value, Mapping):
        return any(_has_credential(item, str(name)) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_credential(item) for item in value)
    return isinstance(value, str) and _credential_text(value)[1]


def _has_credential_material(value: Any, *, structural_keys: bool = False) -> bool:
    """Detect unsanitized credential material in an already-governed document."""
    if isinstance(value, Mapping):
        for name, item in value.items():
            key = str(name)
            if _credential_text(key)[1]:
                return True
            if (
                not structural_keys
                and key.lower() not in _CREDENTIAL_SAFE_KEYS
                and _CREDENTIAL_KEY_RE.search(key)
                and (
                    not isinstance(item, str)
                    or item not in _CREDENTIAL_REDACTION_VALUES
                )
            ):
                return True
            if _has_credential_material(
                item,
                structural_keys=structural_keys or key in _STRUCTURAL_METADATA_KEYS,
            ):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _has_credential_material(item, structural_keys=structural_keys)
            for item in value
        )
    return isinstance(value, str) and _credential_text(value)[1]


def _preview(value: str) -> tuple[str, bool]:
    single_line = " ".join(value.split())
    if len(single_line) <= PREVIEW_MAX_CHARS:
        return single_line, single_line != value
    return f"{single_line[: PREVIEW_MAX_CHARS - 1]}…", True


def _govern_value(
    value: Any,
    *,
    profile: TraceExportProfile,
    path: str,
    key: str = "",
) -> tuple[Any, list[dict[str, str]], bool, bool]:
    """Return governed value, provenance, truncated flag, sensitive flag."""
    if (
        key
        and key.lower() not in _CREDENTIAL_SAFE_KEYS
        and _CREDENTIAL_KEY_RE.search(key)
    ):
        return (
            "[credential redacted]",
            [{"field": path, "state": "redacted", "reason": "credential"}],
            False,
            True,
        )
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        provenance: list[dict[str, str]] = []
        truncated = sensitive = False
        for name, item in value.items():
            raw_name = str(name)
            governed_name, credential_key = _credential_text(raw_name)
            if credential_key:
                governed_name = "[credential key redacted]"
                while governed_name in output:
                    governed_name += "_"
                provenance.append(
                    {
                        "field": f"{path}.{governed_name}",
                        "state": "redacted",
                        "reason": "credential_key",
                    }
                )
                sensitive = True
            item_path = f"{path}.{governed_name}"
            governed, item_provenance, item_truncated, item_sensitive = _govern_value(
                item, profile=profile, path=item_path, key=raw_name
            )
            output[governed_name] = governed
            provenance.extend(item_provenance)
            truncated |= item_truncated
            sensitive |= item_sensitive
        return output, provenance, truncated, sensitive
    if isinstance(value, (list, tuple)):
        output = []
        provenance = []
        truncated = sensitive = False
        for index, item in enumerate(value):
            governed, item_provenance, item_truncated, item_sensitive = _govern_value(
                item, profile=profile, path=f"{path}[{index}]"
            )
            output.append(governed)
            provenance.extend(item_provenance)
            truncated |= item_truncated
            sensitive |= item_sensitive
        return output, provenance, truncated, sensitive
    if not isinstance(value, str):
        return value, [], False, False

    governed, credential = _credential_text(value)
    provenance = (
        [{"field": path, "state": "redacted", "reason": "credential"}]
        if credential
        else []
    )
    sensitive = credential
    if profile is TraceExportProfile.FULL_TRACE:
        return governed, provenance, False, sensitive

    path_redacted = redact_local_paths(governed)
    if path_redacted != governed:
        governed = path_redacted
        sensitive = True
        provenance.append({"field": path, "state": "redacted", "reason": "local_path"})
    if key and _IDENTIFIER_KEY_RE.search(key):
        governed = "[identifier redacted]"
        sensitive = True
        provenance.append({"field": path, "state": "redacted", "reason": "identifier"})
        return governed, provenance, False, sensitive
    if key.lower() in _CONTENT_KEYS or path == "content_preview":
        governed, truncated = _preview(governed)
        if truncated:
            provenance.append(
                {"field": path, "state": "truncated", "reason": "preview_cap"}
            )
        return governed, provenance, truncated, sensitive
    return governed, provenance, False, sensitive


def _record_dict(record: TrajectoryRecord) -> dict[str, Any]:
    usage = dataclasses.asdict(record.usage) if record.usage is not None else None
    state_aliases = {
        "legacy_missing": "not_available",
        "missing": "not_available",
        "source_unavailable": "not_available",
    }
    return {
        "event_id": record.event_id,
        "seq": record.seq,
        "source_seq": record.source_seq,
        "kind": record.kind,
        "label": record.label,
        "status": record.status,
        "conversation_id": record.conversation_id,
        "turn_id": record.turn_id,
        "message_id": record.message_id,
        "actor_kind": record.actor_kind,
        "actor_id": record.actor_id,
        "run_id": record.run_id,
        "parent_event_id": record.parent_event_id,
        "source_event_id": record.source_event_id,
        "replacement_event_id": record.replacement_event_id,
        "observed_at": record.observed_at,
        "step_started_at": record.step_started_at,
        "first_token_at": record.first_token_at,
        "completed_at": record.completed_at,
        "usage": usage,
        "model": record.model,
        "provider": record.provider,
        "content_preview": record.content_preview,
        "payload": record.payload,
        "variants": list(record.variants),
        "depth": record.depth,
        "field_states": {
            str(field): state_aliases.get(str(state), str(state))
            for field, state in record.field_states.items()
        },
        "sensitivity": record.sensitivity,
    }


def _identity_aliases(
    records: Sequence[TrajectoryRecord],
) -> dict[str, dict[str, str]]:
    """Assign deterministic, bundle-local aliases to every identity domain."""
    aliases: dict[str, dict[str, str]] = {
        domain: {} for domain in set(_IDENTITY_DOMAINS.values())
    }
    for record in records:
        for field, domain in _IDENTITY_DOMAINS.items():
            value = getattr(record, field)
            if value is None or value == "":
                continue
            raw = str(value)
            domain_aliases = aliases[domain]
            if raw not in domain_aliases:
                domain_aliases[raw] = f"{domain}-{len(domain_aliases) + 1:06d}"
    return aliases


def _prepare_identities(
    event: dict[str, Any],
    *,
    profile: TraceExportProfile,
    aliases: Mapping[str, Mapping[str, str]],
) -> tuple[list[TraceFieldDecision], list[dict[str, str]]]:
    """Apply identity policy while keeping all references internally coherent."""
    decisions: list[TraceFieldDecision] = []
    provenance: list[dict[str, str]] = []
    for field, domain in _IDENTITY_DOMAINS.items():
        value = event[field]
        if value is None or value == "":
            continue
        raw = str(value)
        credential = _credential_text(raw)[1]
        source_state = str(event["field_states"].get(field) or "observed")
        source_non_observed = source_state in _NON_OBSERVED_STATES
        if (
            profile is TraceExportProfile.FULL_TRACE
            and not credential
            and not source_non_observed
        ):
            decisions.append(
                TraceFieldDecision(
                    str(event["event_id"]), field, "observed", "included", False
                )
            )
            continue
        event[field] = aliases[domain][raw]
        state = source_state if source_non_observed else "redacted"
        reason = (
            f"source_{source_state}"
            if source_non_observed
            else "credential"
            if credential
            else "identifier_alias"
        )
        decisions.append(
            TraceFieldDecision(
                str(event["event_id"]),
                field,
                state,
                reason,
                True,
                source_state,
            )
        )
        provenance.append(
            {
                "event_id": str(event["event_id"]),
                "field": field,
                "state": state,
                "reason": reason,
            }
        )
    return decisions, provenance


def _prepare_field(
    event: dict[str, Any],
    field: str,
    profile: TraceExportProfile,
) -> tuple[Any, TraceFieldDecision, list[dict[str, str]]]:
    event_id = event["event_id"]
    source_state = str(event["field_states"].get(field) or "observed")
    value = event[field]
    sensitive = bool(event.get("sensitivity")) or _has_credential(value)
    if source_state in _NON_OBSERVED_STATES:
        if source_state in {"omitted", "not_available", "capture_failed"}:
            value = [] if field == "variants" else None
        elif source_state == "redacted":
            value = (
                []
                if field == "variants"
                else None
                if field == "payload"
                else "[redacted]"
            )
        elif source_state == "truncated":
            value, _, _, nested_sensitive = _govern_value(
                value, profile=profile, path=field
            )
            sensitive |= nested_sensitive
        decision = TraceFieldDecision(
            event_id,
            field,
            source_state,
            f"source_{source_state}",
            sensitive,
            source_state,
        )
        return (
            value,
            decision,
            [{"field": field, "state": source_state, "reason": decision.reason}],
        )

    if field in {"payload", "variants"} and profile is TraceExportProfile.SAFE_SUMMARY:
        decision = TraceFieldDecision(
            event_id, field, "omitted", "safe_summary", sensitive
        )
        return (
            ([] if field == "variants" else None),
            decision,
            [{"field": field, "state": "omitted", "reason": "safe_summary"}],
        )

    governed, provenance, truncated, nested_sensitive = _govern_value(
        value, profile=profile, path=field
    )
    sensitive |= nested_sensitive
    states = {item["state"] for item in provenance}
    state = (
        "redacted" if "redacted" in states else "truncated" if truncated else "observed"
    )
    reason = "+".join(sorted({item["reason"] for item in provenance})) or "included"
    return (
        governed,
        TraceFieldDecision(event_id, field, state, reason, sensitive),
        provenance,
    )


def _prepare_safe_timing(
    event: dict[str, Any],
) -> tuple[list[TraceFieldDecision], list[dict[str, str]]]:
    """Bucket safe-summary timing to whole seconds with explicit provenance."""
    decisions: list[TraceFieldDecision] = []
    provenance: list[dict[str, str]] = []
    for field in _TIMING_FIELDS:
        value = event[field]
        source_state = str(event["field_states"].get(field) or "observed")
        if source_state in _NON_OBSERVED_STATES:
            if source_state in _MISSING_STATES | {"omitted"}:
                event[field] = None
            decisions.append(
                TraceFieldDecision(
                    str(event["event_id"]),
                    field,
                    source_state,
                    f"source_{source_state}",
                    False,
                    source_state,
                )
            )
            provenance.append(
                {
                    "event_id": str(event["event_id"]),
                    "field": field,
                    "state": source_state,
                    "reason": f"source_{source_state}",
                }
            )
            continue
        if value is None:
            continue
        event[field] = float(int(float(value)))
        event["field_states"][field] = "truncated"
        decisions.append(
            TraceFieldDecision(
                str(event["event_id"]),
                field,
                "truncated",
                "coarse_timing_1s",
                False,
            )
        )
        provenance.append(
            {
                "event_id": str(event["event_id"]),
                "field": field,
                "state": "truncated",
                "reason": "coarse_timing_1s",
            }
        )
    return decisions, provenance


def _scrub_event_credentials(
    event: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Credential-scrub the complete serialized event as the final privacy gate."""
    structural = {
        key: value for key, value in event.items() if key in _STRUCTURAL_METADATA_KEYS
    }
    event_data = {
        key: value
        for key, value in event.items()
        if key not in _STRUCTURAL_METADATA_KEYS
    }
    scrubbed, nested, _, _ = _govern_value(
        event_data,
        profile=TraceExportProfile.FULL_TRACE,
        path="event",
    )
    for key, value in structural.items():
        scrubbed_value, metadata_provenance = _scrub_structural_metadata(
            value, path=f"event.{key}"
        )
        scrubbed[key] = scrubbed_value
        nested.extend(metadata_provenance)
    normalized = []
    for item in nested:
        field = item["field"]
        if field.startswith("event."):
            field = field[len("event.") :]
        normalized.append({**item, "field": field})
    return dict(scrubbed), normalized


def _scrub_structural_metadata(
    value: Any, *, path: str
) -> tuple[Any, list[dict[str, str]]]:
    """Scrub metadata values while preserving field-name keys such as token_count."""
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        provenance: list[dict[str, str]] = []
        for name, item in value.items():
            item_path = f"{path}.{name}"
            output[str(name)], item_provenance = _scrub_structural_metadata(
                item, path=item_path
            )
            provenance.extend(item_provenance)
        return output, provenance
    if isinstance(value, (list, tuple)):
        output = []
        provenance = []
        for index, item in enumerate(value):
            scrubbed, item_provenance = _scrub_structural_metadata(
                item, path=f"{path}[{index}]"
            )
            output.append(scrubbed)
            provenance.extend(item_provenance)
        return output, provenance
    if not isinstance(value, str):
        return value, []
    scrubbed, credential = _credential_text(value)
    if not credential:
        return value, []
    return scrubbed, [{"field": path, "state": "redacted", "reason": "credential"}]


def _upsert_field_decision(
    decisions: list[TraceFieldDecision], decision: TraceFieldDecision
) -> None:
    """Keep the preflight basis unique by event and root field."""
    existing_index = next(
        (
            index
            for index in range(len(decisions) - 1, -1, -1)
            if decisions[index].event_id == decision.event_id
            and decisions[index].field == decision.field
        ),
        None,
    )
    if existing_index is None:
        decisions.append(decision)
    else:
        decisions[existing_index] = decision


def preflight_trace_export(
    snapshot: TrajectorySnapshot,
    *,
    profile: TraceExportProfile | str = TraceExportProfile.REDACTED_DIAGNOSTIC,
) -> TraceExportPreflight:
    """Classify and prepare every material snapshot field in one traversal.

    Args:
        snapshot: Trace snapshot whose event fields will be inventoried and
            prepared for export.
        profile: Privacy profile governing each field decision.

    Returns:
        The immutable preflight inventory, prepared events, and redaction
        provenance used to build the collaboration bundle.

    Raises:
        TrajectoryExportError: If any snapshot event lacks a stable event ID.
        ValueError: If ``profile`` is not a supported privacy profile.
    """
    selected = _profile(profile)
    records = tuple(record for turn in snapshot.turns for record in turn.records)
    aliases = _identity_aliases(records)
    prepared_events: list[dict[str, Any]] = []
    decisions: list[TraceFieldDecision] = []
    provenance: list[dict[str, str]] = []
    for record in records:
        if not record.event_id:
            raise TrajectoryExportError(
                "Invalid Trace snapshot: every event requires a non-empty event_id"
            )
        event = _record_dict(record)
        field_provenance: dict[str, dict[str, str]] = {}
        identity_decisions, identity_provenance = _prepare_identities(
            event, profile=selected, aliases=aliases
        )
        decisions.extend(identity_decisions)
        for decision in identity_decisions:
            event["field_states"][decision.field] = decision.state
            field_provenance[decision.field] = {
                "state": decision.state,
                "reason": decision.reason,
                "sensitivity": record.sensitivity or "unspecified",
            }
        provenance.extend(identity_provenance)
        event_provenance = list(identity_provenance)
        for field in _MATERIAL_FIELDS:
            governed, decision, nested = _prepare_field(event, field, selected)
            event[field] = governed
            event["field_states"][field] = decision.state
            field_provenance[field] = {
                "state": decision.state,
                "reason": decision.reason,
                "sensitivity": record.sensitivity or "unspecified",
            }
            decisions.append(decision)
            for item in nested:
                entry = {"event_id": event["event_id"], **item}
                event_provenance.append(entry)
                provenance.append(entry)
        if selected is TraceExportProfile.SAFE_SUMMARY:
            timing_decisions, timing_provenance = _prepare_safe_timing(event)
            decisions.extend(timing_decisions)
            for entry in timing_provenance:
                root_field = re.split(r"[.[]", entry["field"], maxsplit=1)[0]
                field_provenance[root_field] = {
                    "state": entry["state"],
                    "reason": entry["reason"],
                    "sensitivity": record.sensitivity or "unspecified",
                }
                event_provenance.append(entry)
                provenance.append(entry)
        for field, state in event["field_states"].items():
            if field in field_provenance:
                continue
            if state not in _NON_OBSERVED_STATES | {"observed"}:
                raise TrajectoryExportError(
                    f"Invalid Trace field state {state!r} for {field!r}"
                )
            reason = f"source_{state}"
            field_provenance[field] = {
                "state": state,
                "reason": reason,
                "sensitivity": record.sensitivity or "unspecified",
            }
            if state != "observed":
                entry = {
                    "event_id": str(event["event_id"]),
                    "field": str(field),
                    "state": str(state),
                    "reason": reason,
                }
                event_provenance.append(entry)
                provenance.append(entry)
            _upsert_field_decision(
                decisions,
                TraceFieldDecision(
                    str(event["event_id"]),
                    str(field),
                    str(state),
                    reason,
                    bool(record.sensitivity)
                    or state in {"redacted", "truncated", "omitted"},
                    str(state),
                ),
            )
        event["field_provenance"] = field_provenance
        event["redaction_provenance"] = event_provenance
        event, credential_provenance = _scrub_event_credentials(event)
        credential_roots: set[str] = set()
        for item in credential_provenance:
            entry = {"event_id": str(event["event_id"]), **item}
            event["redaction_provenance"].append(entry)
            provenance.append(entry)
            credential_roots.add(re.split(r"[.[]", item["field"], maxsplit=1)[0])
        for field in sorted(credential_roots):
            event["field_states"][field] = "redacted"
            event["field_provenance"][field] = {
                "state": "redacted",
                "reason": "credential",
                "sensitivity": record.sensitivity or "unspecified",
            }
            event_id = str(event["event_id"])
            existing = next(
                (
                    decision
                    for decision in reversed(decisions)
                    if decision.event_id == event_id and decision.field == field
                ),
                None,
            )
            _upsert_field_decision(
                decisions,
                TraceFieldDecision(
                    event_id,
                    field,
                    state="redacted",
                    reason="credential",
                    sensitive=True,
                    source_state=(
                        existing.source_state if existing is not None else "observed"
                    ),
                ),
            )
        event_id = str(event["event_id"])
        for index, decision in enumerate(decisions):
            if decision.event_id != event_id:
                continue
            detail = event["field_provenance"][decision.field]
            expected_sensitive = detail[
                "sensitivity"
            ] != "unspecified" or decision.state in {"redacted", "truncated", "omitted"}
            if decision.sensitive != expected_sensitive:
                decisions[index] = dataclasses.replace(
                    decision, sensitive=expected_sensitive
                )
        prepared_events.append(event)

    states = [decision.state for decision in decisions]
    actions = {
        (
            entry["event_id"],
            re.split(r"[.[]", entry["field"], maxsplit=1)[0],
            entry["state"],
        )
        for entry in provenance
    }
    inventory = {
        "sensitive": sum(decision.sensitive for decision in decisions),
        "redacted": sum(state == "redacted" for _, _, state in actions),
        "omitted": sum(state == "omitted" for _, _, state in actions),
        "truncated": sum(state == "truncated" for _, _, state in actions),
        "included": sum(state not in _MISSING_STATES | {"omitted"} for state in states),
        "observed": sum(decision.source_state == "observed" for decision in decisions),
        "missing": sum(state in _MISSING_STATES for state in states),
        "not_available": states.count("not_available"),
        "capture_failed": states.count("capture_failed"),
    }
    return TraceExportPreflight(
        profile=selected,
        event_count=len(prepared_events),
        privacy_inventory=inventory,
        field_decisions=tuple(decisions),
        prepared_events=tuple(prepared_events),
        redaction_provenance=tuple(provenance),
        source_event_ids=tuple(str(record.event_id) for record in records),
    )


def _canonical_trace_bytes(payload: Mapping[str, Any]) -> bytes:
    unsigned = json.loads(json.dumps(payload, ensure_ascii=False))
    integrity = unsigned.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("digest", None)
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _trace_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_trace_bytes(payload)).hexdigest()


def _lineage(events: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for event in events:
        for field in ("parent_event_id", "source_event_id", "replacement_event_id"):
            target = event.get(field)
            if target:
                edges.append(
                    {
                        "source": str(event["event_id"]),
                        "target": str(target),
                        "relationship": field.removesuffix("_event_id"),
                    }
                )
    return edges


def _validate_event_references(events: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(event.get("event_id") or "") for event in events]
    duplicate = next((event_id for event_id in ids if ids.count(event_id) > 1), None)
    if duplicate:
        raise TrajectoryExportError(
            f"Invalid Trace events: duplicate event_id {duplicate!r}"
        )
    known = set(ids)
    for index, event in enumerate(events):
        for field in ("parent_event_id", "source_event_id", "replacement_event_id"):
            target = event.get(field)
            if target and target not in known:
                raise TrajectoryExportError(
                    f"Invalid Trace events[{index}]: dangling {field} {target!r}"
                )


def build_trace_export(
    snapshot: TrajectorySnapshot,
    *,
    preflight: TraceExportPreflight | None = None,
    profile: TraceExportProfile | str | None = None,
    confirm_full: bool = False,
    exported_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a canonical, integrity-protected Trace v2 collaboration bundle."""
    if preflight is None:
        preflight = preflight_trace_export(
            snapshot,
            profile=profile or TraceExportProfile.REDACTED_DIAGNOSTIC,
        )
    elif profile is not None and preflight.profile is not _profile(profile):
        raise TrajectoryExportError("Trace export profile does not match its preflight")
    snapshot_event_ids = tuple(
        str(record.event_id) for turn in snapshot.turns for record in turn.records
    )
    if snapshot_event_ids != preflight.source_event_ids:
        raise TrajectoryExportError(
            "Trace export preflight belongs to a different snapshot; run preflight again"
        )
    if preflight.profile is TraceExportProfile.FULL_TRACE and not confirm_full:
        raise TrajectoryExportError(
            "Full Trace export requires explicit confirm_full=True confirmation"
        )

    if exported_at is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    elif isinstance(exported_at, datetime):
        timestamp = (
            exported_at.replace(tzinfo=timezone.utc)
            if exported_at.tzinfo is None
            else exported_at
        ).isoformat()
    else:
        raw_timestamp = str(exported_at)
        try:
            parsed_timestamp = datetime.fromisoformat(
                raw_timestamp.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise TrajectoryExportError(
                "Invalid Trace exported_at: expected an ISO 8601 datetime"
            ) from exc
        if parsed_timestamp.tzinfo is None:
            parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
        timestamp = parsed_timestamp.isoformat()

    events = [
        json.loads(json.dumps(event, ensure_ascii=False))
        for event in preflight.prepared_events
    ]
    _validate_event_references(events)
    export_event_id = f"trace_export:{timestamp}"
    suffix = 2
    known_ids = {event["event_id"] for event in events}
    while export_event_id in known_ids:
        export_event_id = f"trace_export:{timestamp}:{suffix}"
        suffix += 1
    export_event = {
        "event_id": export_event_id,
        "seq": max((int(event.get("seq") or 0) for event in events), default=0) + 1,
        "source_seq": None,
        "kind": "trace_export",
        "label": "Trace export",
        "status": "complete",
        "conversation_id": events[0].get("conversation_id") if events else None,
        "turn_id": "trace",
        "message_id": None,
        "actor_kind": "system",
        "actor_id": "trace",
        "run_id": None,
        "parent_event_id": None,
        "source_event_id": None,
        "replacement_event_id": None,
        "observed_at": None,
        "step_started_at": None,
        "first_token_at": None,
        "completed_at": None,
        "usage": None,
        "model": None,
        "provider": None,
        "content_preview": f"Trace exported as {preflight.profile.value}",
        "payload": {
            "profile": preflight.profile.value,
            "privacy_inventory": dict(preflight.privacy_inventory),
        },
        "variants": [],
        "depth": 0,
        "field_states": {},
        "sensitivity": "diagnostic",
        "field_provenance": {},
        "redaction_provenance": [],
    }
    events.append(export_event)
    missing_metadata = [
        {
            "event_id": str(event["event_id"]),
            "field": str(field),
            "state": str(state),
        }
        for event in preflight.prepared_events
        for field, state in event["field_states"].items()
        if state in _MISSING_STATES
    ]
    manifest = {
        "schema_version": TRACE_EXPORT_VERSION,
        "format_version": TRACE_EXPORT_VERSION,
        "profile": preflight.profile.value,
        "event_count": len(events),
        "exported_at": timestamp,
        "exported_timestamp": timestamp,
        "source": {
            "type": "trajectory_snapshot",
            "conversation_ids": sorted(
                {
                    str(event["conversation_id"])
                    for event in events
                    if event.get("conversation_id")
                }
            ),
        },
        "missing_metadata": missing_metadata,
        "privacy_inventory": dict(preflight.privacy_inventory),
        "redaction_provenance": list(preflight.redaction_provenance),
        "privacy_decisions": [
            dataclasses.asdict(decision) for decision in preflight.field_decisions
        ],
        "export_operation_event_id": export_event_id,
        "integrity_notice": "SHA-256 detects corruption; it does not prove authenticity",
    }
    payload: dict[str, Any] = {
        "format": TRACE_EXPORT_FORMAT,
        "version": TRACE_EXPORT_VERSION,
        "manifest": manifest,
        "events": events,
        "lineage": _lineage(events),
        "integrity": {"algorithm": "sha256", "authenticity": False},
    }
    if _has_credential_material(payload):
        raise TrajectoryExportError(
            "Trace export blocked: the final bundle still contains credential material"
        )
    payload["integrity"]["digest"] = _trace_digest(payload)
    return payload


# ---------------------------------------------------------------------------
# Build (DB read -> export payload)
# ---------------------------------------------------------------------------


def build_trajectory_export(
    db: Any,
    conversation_id: str,
    *,
    include_payloads: bool = False,
    variant_sets: Sequence[Any] = (),
) -> dict:
    """Build the export payload for one conversation (ADR-067 §1).

    Reads the same seams the live projection uses:
    ``get_conversation_by_id`` / ``get_messages_for_conversation(...,
    include_image_data=False)`` / ``get_trajectory_rows`` /
    ``get_conversation_active_leaf``, plus compaction attempts via
    ``ConsoleContextRepository.list_auxiliary_attempts`` filtered to
    ``purpose == "conversation_compaction"`` (as the projection does).

    Args:
        db: The ``CharactersRAGDB`` instance.
        conversation_id: The conversation to export.
        include_payloads: Explicit opt-in to keep tool ``payload_json``
            verbatim. Default redacts tool payloads to previews.
        variant_sets: Live-session variant sets; serialized under
            ``variants`` only when provided.

    Returns:
        The export payload dict (not yet written to disk).

    Raises:
        TrajectoryExportError: If the conversation does not exist.
    """
    from tldw_chatbook.Chat.console_context_repository import ConsoleContextRepository

    conversation = db.get_conversation_by_id(conversation_id)
    if conversation is None:
        raise TrajectoryExportError(
            f"Conversation '{conversation_id}' not found (deleted or unknown id)"
        )

    messages = db.get_messages_for_conversation(
        conversation_id, limit=_MESSAGE_READ_LIMIT, include_image_data=False
    )
    traj_rows = db.get_trajectory_rows(conversation_id)
    aux_attempts = ConsoleContextRepository(db).list_auxiliary_attempts(
        conversation_id, limit=_AUX_ATTEMPT_LIMIT
    )
    compaction_records = [
        {key: _jsonable(value) for key, value in record.items()}
        for record in aux_attempts
        if str(record.get("purpose") or "") == _COMPACTION_PURPOSE
    ]

    rows_out: list[dict] = []
    for row in traj_rows:
        data = _as_dict(row)
        kind = str(data.get("event_kind") or "")
        payload_json = data.get("payload_json")
        if kind == LIBRARY_PREPARATION_EVENT_KIND:
            try:
                payload_json = encode_library_preparation_event(
                    decode_library_preparation_event(payload_json)
                )
            except LibraryPreparationValidationError:
                payload_json = None
        elif not include_payloads and kind in _TOOL_KINDS and payload_json:
            payload_json = _redacted_payload_json(payload_json)
        exported_row = {key: _jsonable(data.get(key)) for key in _TRAJECTORY_ROW_KEYS}
        exported_row["payload_json"] = payload_json
        rows_out.append(exported_row)

    payload: dict = {
        "format": TRAJECTORY_EXPORT_FORMAT,
        "version": TRAJECTORY_EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "redacted": not include_payloads,
        "conversation": {
            "id": conversation.get("id"),
            "title": _jsonable(conversation.get("title")),
            "created_at": _jsonable(conversation.get("created_at")),
        },
        "active_leaf_message_id": db.get_conversation_active_leaf(conversation_id),
        "messages": [
            {key: _jsonable(message.get(key)) for key in _MESSAGE_KEYS}
            for message in messages
        ],
        "trajectory_rows": rows_out,
        "compaction_records": compaction_records,
    }
    if variant_sets:
        payload["variants"] = [_serialize_variant_set(vs) for vs in variant_sets]
    return payload


# ---------------------------------------------------------------------------
# Write (atomic)
# ---------------------------------------------------------------------------


def write_trajectory_export(path: Path | str, payload: dict) -> Path:
    """Write ``payload`` as pretty JSON to ``path`` atomically.

    Serializes with ``indent=2`` / ``ensure_ascii=False``, writes to a
    sibling temp file, then ``os.replace``s it into place, so readers
    never observe a partial file and an existing file is overwritten
    atomically. Temp files are removed on failure.

    Args:
        path: Destination file path.
        payload: The export payload (as built by ``build_trajectory_export``).

    Returns:
        The resolved destination path.

    Raises:
        OSError: If writing or renaming fails.
    """
    destination = Path(path)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return destination


# ---------------------------------------------------------------------------
# Validate (the import seam)
# ---------------------------------------------------------------------------


def _require(
    payload: Mapping,
    key: str,
    types: tuple[type, ...] | type,
    what: str,
) -> Any:
    """Fetch a required section with a type check; name the field on error."""
    if key not in payload:
        raise TrajectoryExportError(
            f"Invalid trajectory export: missing required '{key}' section"
        )
    value = payload[key]
    if not isinstance(value, types):
        expected = " or ".join(
            t.__name__ for t in (types if isinstance(types, tuple) else (types,))
        )
        raise TrajectoryExportError(
            f"Invalid trajectory export: '{key}' must be {expected}, got {type(value).__name__}"
        )
    return value


def validate_trajectory_export(payload: Any) -> dict:
    """Validate an export payload and return it normalized (import seam).

    Checks the format marker and version (rejecting higher versions with
    an actionable error per ADR-067 §2), required sections and types, and
    required keys on each message / trajectory row entry. Optional
    sections (``compaction_records``, ``variants``,
    ``active_leaf_message_id``) are filled to ``[]`` / ``None`` when
    absent; unknown additive fields are ignored (ADR-067 §2).

    Args:
        payload: The parsed JSON document.

    Returns:
        The normalized payload dict.

    Raises:
        TrajectoryExportError: Naming the offending field for any
            contract violation.
    """
    if not isinstance(payload, Mapping):
        raise TrajectoryExportError(
            "Invalid trajectory export: top-level document must be a JSON object, "
            f"got {type(payload).__name__}"
        )

    fmt = payload.get("format")
    if fmt != TRAJECTORY_EXPORT_FORMAT:
        raise TrajectoryExportError(
            f"Invalid trajectory export: 'format' must be "
            f"'{TRAJECTORY_EXPORT_FORMAT}', got {fmt!r}"
        )

    version = payload.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise TrajectoryExportError(
            f"Invalid trajectory export: 'version' must be an integer, got {version!r}"
        )
    if version > TRAJECTORY_EXPORT_VERSION:
        raise TrajectoryExportError(
            f"Unsupported trajectory export version {version}: this build reads "
            f"version {TRAJECTORY_EXPORT_VERSION}; export with an older version "
            f"or upgrade the app"
        )
    if version < TRAJECTORY_EXPORT_VERSION:
        raise TrajectoryExportError(
            f"Invalid trajectory export: 'version' must be "
            f"{TRAJECTORY_EXPORT_VERSION}, got {version!r}"
        )

    _require(payload, "exported_at", str, "exported_at")
    _require(payload, "redacted", bool, "redacted")
    conversation = _require(payload, "conversation", dict, "conversation")
    if "id" not in conversation:
        raise TrajectoryExportError(
            "Invalid trajectory export: 'conversation.id' is missing"
        )

    active_leaf = payload.get("active_leaf_message_id")
    if active_leaf is not None and not isinstance(active_leaf, str):
        raise TrajectoryExportError(
            "Invalid trajectory export: 'active_leaf_message_id' must be a string or null"
        )

    messages = _require(payload, "messages", list, "messages")
    normalized_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TrajectoryExportError(
                f"Invalid trajectory export: 'messages[{index}]' must be an object"
            )
        normalized_message = dict(message)
        normalized_message.setdefault("assistant_generation_state", None)
        for key in _MESSAGE_KEYS:
            if key not in normalized_message:
                raise TrajectoryExportError(
                    f"Invalid trajectory export: 'messages[{index}].{key}' is missing"
                )
        sender = normalized_message["sender"]
        raw_state = normalized_message["assistant_generation_state"]
        if raw_state is not None and str(sender or "").lower() != "assistant":
            raise TrajectoryExportError(
                f"Invalid trajectory export: 'messages[{index}].assistant_generation_state' "
                "is invalid for a non-assistant message"
            )
        try:
            state = normalize_assistant_generation_state(
                role=sender,
                raw_state=raw_state,
                has_valid_active_continuation=False,
            )
        except ValueError:
            raise TrajectoryExportError(
                f"Invalid trajectory export: 'messages[{index}].assistant_generation_state' "
                "is invalid"
            ) from None
        normalized_message["assistant_generation_state"] = (
            state.value if state is not None else None
        )
        normalized_messages.append(normalized_message)

    rows = _require(payload, "trajectory_rows", list, "trajectory_rows")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TrajectoryExportError(
                f"Invalid trajectory export: 'trajectory_rows[{index}]' must be an object"
            )
        for key in _REQUIRED_ROW_KEYS:
            if key not in row:
                raise TrajectoryExportError(
                    f"Invalid trajectory export: 'trajectory_rows[{index}].{key}' is missing"
                )

    compaction = payload.get("compaction_records")
    if compaction is not None and not isinstance(compaction, list):
        raise TrajectoryExportError(
            "Invalid trajectory export: 'compaction_records' must be a list"
        )
    variants = payload.get("variants")
    if variants is not None and not isinstance(variants, list):
        raise TrajectoryExportError(
            "Invalid trajectory export: 'variants' must be a list"
        )

    normalized = dict(payload)
    normalized["messages"] = normalized_messages
    normalized.setdefault("compaction_records", [])
    normalized.setdefault("variants", [])
    normalized.setdefault("active_leaf_message_id", None)
    return normalized
