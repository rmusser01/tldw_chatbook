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
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "TRAJECTORY_EXPORT_FORMAT",
    "TRAJECTORY_EXPORT_VERSION",
    "PREVIEW_MAX_CHARS",
    "TrajectoryExportError",
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
        if not include_payloads and kind in _TOOL_KINDS and payload_json:
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
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TrajectoryExportError(
                f"Invalid trajectory export: 'messages[{index}]' must be an object"
            )
        for key in _MESSAGE_KEYS:
            if key not in message:
                raise TrajectoryExportError(
                    f"Invalid trajectory export: 'messages[{index}].{key}' is missing"
                )

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
    normalized.setdefault("compaction_records", [])
    normalized.setdefault("variants", [])
    normalized.setdefault("active_leaf_message_id", None)
    return normalized
