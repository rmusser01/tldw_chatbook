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
from collections.abc import Mapping
from pathlib import Path

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import TrajectorySnapshot, derive_trajectory
from tldw_chatbook.Chat.trajectory_export import (
    TrajectoryExportError,
    validate_trajectory_export,
)

__all__ = [
    "TrajectoryImportError",
    "load_trajectory_snapshot",
]


class TrajectoryImportError(TrajectoryExportError):
    """A trace file that could not be read, parsed, validated, or mapped.

    Subclasses :class:`TrajectoryExportError` so the shared-validator
    rejections (format marker, version, missing sections) surface as
    import errors too; the message always names the problem file and the
    offending field/section so the user can act on it.
    """


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


def load_trajectory_snapshot(source: Path | str | Mapping) -> TrajectorySnapshot:
    """Load a shared trajectory trace into a renderable snapshot.

    Read (or accept) the export document, validate it through the shared
    ADR-067 seam (``validate_trajectory_export``), and map the sections
    onto ``derive_trajectory`` inputs: ``messages`` pass through with
    per-message ``ProviderUsage`` parsed from ``usage_json``,
    ``trajectory_rows`` / ``compaction_records`` / ``variants`` feed the
    projection as-is (it accepts mappings), and ``active_leaf_message_id``
    selects the active path. Timestamps are ISO strings in the file; the
    projection's parser handles them.

    Args:
        source: A file path (``Path`` or ``str``) to a trace file, or an
            already-parsed export document (mapping).

    Returns:
        The snapshot; render it with ``TrajectoryScreen`` exactly like a
        live projection result.

    Raises:
        TrajectoryImportError: Unreadable file, invalid JSON, or any
            contract violation named by the shared validator (wrong
            format marker, unsupported version, missing sections, ...).
    """
    path = None if isinstance(source, Mapping) else Path(str(source))
    document = _read_document(source)
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
