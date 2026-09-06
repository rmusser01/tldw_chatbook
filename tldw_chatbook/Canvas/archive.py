"""Validation and bounded inert export helpers for Chatbook Canvas records.

Source is streamed only as UTF-8 bytes for identity verification and archive
copying. This module never parses, compiles, renders, or executes Canvas HTML.
"""

from __future__ import annotations

import codecs
import hashlib
import re
import sqlite3
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from uuid import UUID

from .limits import (
    MAX_CANVAS_ORIGIN_TURN_ID_BYTES,
    MAX_CANVAS_TITLE_BYTES,
    MAX_CANVASES_PER_CONVERSATION,
    MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION,
    MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
    MAX_REVISIONS_PER_CANVAS,
    SUPPORTED_CANVAS_RUNTIME_PROFILE,
    validate_utf8_text,
)

CANVAS_ARCHIVE_EXTENSION_VERSION = "1.0"
CANVAS_ARCHIVE_SUPPORTED_RUNTIME_PROFILE = SUPPORTED_CANVAS_RUNTIME_PROFILE
MAX_CANVAS_ARCHIVE_DOCUMENTS = 1_000
MAX_CANVAS_ARCHIVE_REVISIONS = MAX_CANVAS_ARCHIVE_DOCUMENTS * MAX_REVISIONS_PER_CANVAS
MAX_CANVAS_ARCHIVE_REOPEN_HINTS = 10_000
MAX_CANVAS_ARCHIVE_SOURCE_BYTES = 512 * 1024 * 1024
MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES = 256
MAX_CANVAS_ARCHIVE_MESSAGE_ID_BYTES = 256
MAX_CANVAS_ARCHIVE_RUNTIME_PROFILE_BYTES = 64

_RUNTIME_PROFILE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ACTOR_KINDS = frozenset({"assistant", "user_rename", "user_import"})
CANVAS_ARCHIVE_IO_CHUNK_BYTES = 64 * 1024

if TYPE_CHECKING:
    from tldw_chatbook.Chatbooks.chatbook_models import CanvasArchiveManifest
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class CanvasArchiveValidationError(ValueError):
    """A bounded Canvas archive-metadata validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"invalid Canvas archive: {code}")


def export_canvas_archive(
    db: CharactersRAGDB,
    conversation_ids: tuple[str, ...],
    work_dir: Path,
) -> CanvasArchiveManifest | None:
    """Stream selected durable Canvas histories into inert archive entries."""

    from tldw_chatbook.Chatbooks.chatbook_models import (
        CanvasArchiveDocument,
        CanvasArchiveManifest,
        CanvasArchiveReopenHint,
        CanvasArchiveRevision,
    )

    if not conversation_ids:
        return None
    if len(set(conversation_ids)) != len(conversation_ids):
        raise CanvasArchiveValidationError("duplicate_conversation_id")
    for conversation_id in conversation_ids:
        validate_bounded_identifier(
            conversation_id,
            field_name="conversation_id",
            byte_limit=MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES,
        )

    placeholders = ", ".join("?" for _ in conversation_ids)
    connection = db.get_connection()
    rows = connection.execute(
        "SELECT document.id, document.conversation_id, document.created_at, "
        "document.deleted_at, revision.id, revision.parent_revision_id, "
        "revision.sequence, revision.title, revision.runtime_profile, "
        "revision.content_sha256, revision.html_bytes, revision.actor_kind, "
        "revision.origin_message_id, revision.origin_turn_id, "
        "revision.created_at, revision.deleted_at, message.conversation_id "
        "FROM canvas_documents AS document "
        "JOIN canvas_revisions AS revision ON revision.canvas_id = document.id "
        "JOIN messages AS message ON message.id = revision.origin_message_id "
        f"WHERE document.conversation_id IN ({placeholders}) "
        "ORDER BY document.conversation_id, document.id, revision.sequence",
        conversation_ids,
    ).fetchall()
    if not isinstance(rows, list) or not rows:
        return None

    grouped: dict[str, list[sqlite3.Row]] = {}
    document_order: list[str] = []
    total_bytes = 0
    for row in rows:
        canvas_id = str(row[0])
        conversation_id = str(row[1])
        if str(row[16]) != conversation_id:
            raise CanvasArchiveValidationError("origin_owner_mismatch")
        if canvas_id not in grouped:
            grouped[canvas_id] = []
            document_order.append(canvas_id)
        grouped[canvas_id].append(row)

    documents = []
    for canvas_id in document_order:
        revisions = []
        document_rows = grouped[canvas_id]
        for row in document_rows:
            revision_id = str(row[4])
            source_path = canvas_revision_source_path(canvas_id, revision_id)
            declared_bytes = int(row[10])
            digest, actual_bytes = _stream_repository_source(
                connection,
                revision_id=revision_id,
                destination=work_dir / source_path,
            )
            if actual_bytes != declared_bytes or digest != str(row[9]):
                raise CanvasArchiveValidationError("stored_source_identity_mismatch")
            total_bytes += actual_bytes
            if total_bytes > MAX_CANVAS_ARCHIVE_SOURCE_BYTES:
                raise CanvasArchiveValidationError("archive_source_byte_limit")
            revisions.append(
                CanvasArchiveRevision(
                    revision_id=revision_id,
                    parent_revision_id=str(row[5]) if row[5] is not None else None,
                    sequence=int(row[6]),
                    title=str(row[7]),
                    runtime_profile=str(row[8]),
                    source_path=source_path,
                    content_sha256=digest,
                    source_bytes=actual_bytes,
                    actor_kind=str(row[11]),
                    origin_message_id=str(row[12]),
                    origin_turn_id=str(row[13]),
                    created_at=str(row[14]),
                    deleted_at=str(row[15]) if row[15] is not None else None,
                )
            )
        first = document_rows[0]
        documents.append(
            CanvasArchiveDocument(
                canvas_id=canvas_id,
                conversation_id=str(first[1]),
                created_at=str(first[2]),
                deleted_at=str(first[3]) if first[3] is not None else None,
                revisions=tuple(revisions),
            )
        )

    hint_rows = connection.execute(
        "SELECT hint.conversation_id, hint.last_canvas_id "
        "FROM canvas_conversation_hints AS hint "
        "JOIN canvas_documents AS document ON document.id = hint.last_canvas_id "
        f"WHERE hint.conversation_id IN ({placeholders}) "
        "AND document.deleted = 0 ORDER BY hint.conversation_id, hint.last_canvas_id",
        conversation_ids,
    ).fetchall()
    hints = tuple(
        CanvasArchiveReopenHint(
            conversation_id=str(row[0]),
            canvas_id=str(row[1]),
        )
        for row in hint_rows
    )
    return CanvasArchiveManifest(
        extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
        total_source_bytes=total_bytes,
        documents=tuple(documents),
        reopen_hints=hints,
    )


def validate_exported_canvas_origins(
    canvas_archive: CanvasArchiveManifest | None,
    conversations: tuple[Mapping[str, object], ...],
) -> None:
    """Require Canvas origins to exist in the exact staged conversation graph."""

    if canvas_archive is None:
        return
    staged_messages: dict[str, frozenset[str]] = {}
    for conversation in conversations:
        conversation_id = conversation.get("id")
        messages = conversation.get("messages")
        if not isinstance(conversation_id, str) or not isinstance(messages, list):
            raise CanvasArchiveValidationError("invalid_staged_conversation")
        if conversation_id in staged_messages:
            raise CanvasArchiveValidationError("duplicate_conversation_id")
        message_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, Mapping) or not isinstance(
                message.get("id"), str
            ):
                raise CanvasArchiveValidationError("invalid_staged_message")
            message_id = str(message["id"])
            if message_id in message_ids:
                raise CanvasArchiveValidationError("duplicate_staged_message_id")
            message_ids.add(message_id)
        staged_messages[conversation_id] = frozenset(message_ids)

    for document in canvas_archive.documents:
        message_ids = staged_messages.get(document.conversation_id)
        if message_ids is None:
            raise CanvasArchiveValidationError("conversation_not_staged")
        for revision in document.revisions:
            if revision.origin_message_id not in message_ids:
                raise CanvasArchiveValidationError("origin_message_not_staged")


def _stream_repository_source(
    connection: sqlite3.Connection,
    *,
    revision_id: str,
    destination: Path,
) -> tuple[str, int]:
    """Read one source as bounded BLOB slices and verify strict UTF-8."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    offset = 1
    total = 0
    try:
        with destination.open("xb") as handle:
            while True:
                row = connection.execute(
                    "SELECT substr(CAST(html AS BLOB), ?, ?) "
                    "FROM canvas_revisions WHERE id = ?",
                    (offset, CANVAS_ARCHIVE_IO_CHUNK_BYTES, revision_id),
                ).fetchone()
                if row is None:
                    raise CanvasArchiveValidationError("revision_not_found")
                chunk = bytes(row[0] or b"")
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DURABLE_SOURCE_BYTES_PER_REVISION:
                    raise CanvasArchiveValidationError("revision_source_byte_limit")
                decoder.decode(chunk, final=False)
                digest.update(chunk)
                handle.write(chunk)
                offset += len(chunk)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        raise CanvasArchiveValidationError("invalid_source_utf8") from None
    return digest.hexdigest(), total


def canvas_revision_source_path(canvas_id: str, revision_id: str) -> str:
    """Return the sole canonical, inert source entry path for a revision."""

    safe_canvas_id = validate_archive_uuid(canvas_id, field_name="canvas_id")
    safe_revision_id = validate_archive_uuid(revision_id, field_name="revision_id")
    return f"canvas/{safe_canvas_id}/{safe_revision_id}.html.txt"


def validate_revision_source_path(
    value: str, *, canvas_id: str, revision_id: str
) -> str:
    """Reject every path except the exact canonical inert revision path."""

    validate_inert_source_path_shape(value, revision_id=revision_id)
    expected = canvas_revision_source_path(canvas_id, revision_id)
    if value != expected or PurePosixPath(value).as_posix() != expected:
        raise CanvasArchiveValidationError("invalid_source_path")
    return value


def validate_inert_source_path_shape(value: object, *, revision_id: str) -> str:
    """Validate an inert canonical path before its containing owner is known."""

    if not isinstance(value, str) or len(value) != 89:
        raise CanvasArchiveValidationError("invalid_source_path")
    parts = value.split("/")
    suffix = ".html.txt"
    if len(parts) != 3 or parts[0] != "canvas" or not parts[2].endswith(suffix):
        raise CanvasArchiveValidationError("invalid_source_path")
    validate_archive_uuid(parts[1], field_name="source_canvas_id")
    path_revision_id = parts[2][: -len(suffix)]
    validate_archive_uuid(path_revision_id, field_name="source_revision_id")
    if path_revision_id != revision_id:
        raise CanvasArchiveValidationError("invalid_source_path")
    return value


def validate_archive_uuid(value: object, *, field_name: str) -> str:
    """Require a canonical lowercase UUID suitable for an archive path."""

    if not isinstance(value, str) or len(value) != 36:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise CanvasArchiveValidationError(f"invalid_{field_name}") from None
    if str(parsed) != value:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return value


def validate_bounded_identifier(
    value: object, *, field_name: str, byte_limit: int = 256
) -> str:
    """Validate an opaque non-empty identifier that never becomes a path."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    try:
        validate_utf8_text(value, limit=byte_limit, field_name=field_name)
    except ValueError:
        raise CanvasArchiveValidationError(f"invalid_{field_name}") from None
    return value


def validate_runtime_profile(value: object) -> str:
    """Validate profile syntax while retaining well-formed unknown profiles."""

    if not isinstance(value, str):
        raise CanvasArchiveValidationError("invalid_runtime_profile")
    try:
        validate_utf8_text(
            value,
            limit=MAX_CANVAS_ARCHIVE_RUNTIME_PROFILE_BYTES,
            field_name="runtime_profile",
        )
    except ValueError:
        raise CanvasArchiveValidationError("invalid_runtime_profile") from None
    if _RUNTIME_PROFILE.fullmatch(value) is None:
        raise CanvasArchiveValidationError("invalid_runtime_profile")
    return value


def validate_digest(value: object) -> str:
    """Require the canonical lowercase SHA-256 spelling."""

    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CanvasArchiveValidationError("invalid_content_sha256")
    return value


def validate_non_negative_int(value: object, *, field_name: str, maximum: int) -> int:
    """Validate an archive integer without accepting booleans or coercion."""

    if type(value) is not int or value < 0 or value > maximum:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return value


def validate_positive_int(value: object, *, field_name: str, maximum: int) -> int:
    """Validate a positive bounded archive integer."""

    result = validate_non_negative_int(value, field_name=field_name, maximum=maximum)
    if result == 0:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return result


def validate_timestamp(
    value: object, *, field_name: str, optional: bool = False
) -> str | None:
    """Require a bounded timezone-aware ISO-8601 timestamp."""

    if value is None and optional:
        return None
    if not isinstance(value, str) or not value or len(value) > 64:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise CanvasArchiveValidationError(f"invalid_{field_name}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CanvasArchiveValidationError(f"invalid_{field_name}")
    return value


def validate_title(value: object) -> str:
    """Validate a revisioned Canvas title."""

    if not isinstance(value, str) or not value:
        raise CanvasArchiveValidationError("invalid_title")
    try:
        validate_utf8_text(value, limit=MAX_CANVAS_TITLE_BYTES, field_name="title")
    except ValueError:
        raise CanvasArchiveValidationError("invalid_title") from None
    return value


def validate_actor_kind(value: object) -> str:
    """Require one actor kind supported by the persisted Canvas schema."""

    if not isinstance(value, str) or value not in _ACTOR_KINDS:
        raise CanvasArchiveValidationError("invalid_actor_kind")
    return value


def require_exact_fields(
    value: object, *, required: frozenset[str], optional: frozenset[str] = frozenset()
) -> Mapping[str, object]:
    """Require a mapping with exactly the known fields for extension 1.0."""

    if not isinstance(value, Mapping):
        raise CanvasArchiveValidationError("invalid_record")
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise CanvasArchiveValidationError("invalid_record_fields")
    if not all(isinstance(key, str) for key in keys):
        raise CanvasArchiveValidationError("invalid_record_fields")
    return value


__all__ = [
    "CANVAS_ARCHIVE_EXTENSION_VERSION",
    "CANVAS_ARCHIVE_IO_CHUNK_BYTES",
    "CANVAS_ARCHIVE_SUPPORTED_RUNTIME_PROFILE",
    "MAX_CANVASES_PER_CONVERSATION",
    "MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES",
    "MAX_CANVAS_ARCHIVE_DOCUMENTS",
    "MAX_CANVAS_ARCHIVE_MESSAGE_ID_BYTES",
    "MAX_CANVAS_ARCHIVE_REOPEN_HINTS",
    "MAX_CANVAS_ARCHIVE_REVISIONS",
    "MAX_CANVAS_ARCHIVE_SOURCE_BYTES",
    "MAX_CANVAS_ORIGIN_TURN_ID_BYTES",
    "MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION",
    "MAX_DURABLE_SOURCE_BYTES_PER_REVISION",
    "MAX_REVISIONS_PER_CANVAS",
    "CanvasArchiveValidationError",
    "canvas_revision_source_path",
    "export_canvas_archive",
    "require_exact_fields",
    "validate_actor_kind",
    "validate_archive_uuid",
    "validate_bounded_identifier",
    "validate_digest",
    "validate_exported_canvas_origins",
    "validate_inert_source_path_shape",
    "validate_non_negative_int",
    "validate_positive_int",
    "validate_revision_source_path",
    "validate_runtime_profile",
    "validate_timestamp",
    "validate_title",
]
