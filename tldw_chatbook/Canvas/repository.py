"""Typed persistence for conversation-owned immutable Canvas revision graphs."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypeAlias
from uuid import UUID, uuid4

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

from .limits import (
    MAX_CANVAS_DURABLE_ACTIVE_PATH_MESSAGES,
    CanvasLimitError,
    CanvasRepositoryLimits,
    sha256_utf8,
    utf8_byte_length,
    validate_utf8_text,
)

CanvasActorKind: TypeAlias = Literal["assistant", "user_rename", "user_import"]
CanvasRuntimeProfile: TypeAlias = Literal["canvas-v1"]
_ACTOR_KINDS = frozenset({"assistant", "user_rename", "user_import"})


class CanvasRepositoryError(Exception):
    """Base for bounded repository failures that never echo Canvas source."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CanvasValidationError(CanvasRepositoryError):
    """Raised when an input graph or ownership claim is invalid."""


class CanvasQuotaError(CanvasRepositoryError):
    """Raised when a durable Canvas ceiling would be exceeded."""


class CanvasNotFoundError(CanvasRepositoryError):
    """Raised when an owner-scoped Canvas value is unavailable."""


class CanvasConflictError(CanvasRepositoryError):
    """Raised when a caller-supplied stable identity already exists."""


@dataclass(frozen=True, slots=True)
class CanvasIdentity:
    """One durable Canvas identity without mutable title/source heads."""

    canvas_id: str
    conversation_id: str
    created_at: str
    deleted_at: str | None


@dataclass(frozen=True, slots=True)
class CanvasRevision:
    """One exact immutable revision, including its complete source snapshot."""

    revision_id: str
    canvas_id: str
    parent_revision_id: str | None
    sequence: int
    title: str
    runtime_profile: CanvasRuntimeProfile
    source: str
    content_sha256: str
    source_bytes: int
    actor_kind: CanvasActorKind
    origin_message_id: str
    origin_turn_id: str
    created_at: str
    deleted_at: str | None

    @property
    def html(self) -> str:
        """Compatibility spelling used by the approved conceptual schema."""

        return self.source

    @property
    def html_bytes(self) -> int:
        """Compatibility spelling used by the approved conceptual schema."""

        return self.source_bytes


@dataclass(frozen=True, slots=True)
class CanvasRevisionMetadata:
    """Source-free metadata used by later branch resolution."""

    revision_id: str
    canvas_id: str
    conversation_id: str
    parent_revision_id: str | None
    sequence: int
    title: str
    runtime_profile: CanvasRuntimeProfile
    content_sha256: str
    source_bytes: int
    actor_kind: CanvasActorKind
    origin_message_id: str
    origin_turn_id: str
    canvas_created_at: str
    canvas_deleted_at: str | None
    revision_created_at: str
    revision_deleted_at: str | None


@dataclass(frozen=True, slots=True)
class CanvasCreateResult:
    """The identity and first revision committed by one atomic create."""

    identity: CanvasIdentity
    revision: CanvasRevision


@dataclass(frozen=True, slots=True)
class CanvasPurgeResult:
    """Content-free counts from an owner-scoped hard purge."""

    conversation_id: str
    canvases_deleted: int
    revisions_deleted: int


@dataclass(frozen=True, slots=True)
class CanvasImportDocument:
    """One stable Canvas identity supplied by a validated import graph."""

    canvas_id: str
    conversation_id: str
    created_at: str
    deleted_at: str | None = None


@dataclass(frozen=True, slots=True)
class CanvasImportRevision:
    """One complete stable revision supplied by a validated import graph."""

    revision_id: str
    canvas_id: str
    parent_revision_id: str | None
    sequence: int
    title: str
    runtime_profile: CanvasRuntimeProfile
    source: str
    content_sha256: str
    source_bytes: int
    actor_kind: CanvasActorKind
    origin_message_id: str
    origin_turn_id: str
    created_at: str
    deleted_at: str | None = None


@dataclass(frozen=True, slots=True)
class CanvasImportBatch:
    """A complete owner-scoped import graph validated before its first write."""

    conversation_id: str
    documents: tuple[CanvasImportDocument, ...]
    revisions: tuple[CanvasImportRevision, ...]
    reopen_canvas_id: str | None = None


@dataclass(frozen=True, slots=True)
class CanvasImportResult:
    """Content-free counts from an atomic import."""

    conversation_id: str
    canvases_imported: int
    revisions_imported: int


class CanvasRepository:
    """Read and append durable Canvas graphs through one database owner."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        limits: CanvasRepositoryLimits | None = None,
    ) -> None:
        if not isinstance(db, CharactersRAGDB):
            raise TypeError("db must be a CharactersRAGDB")
        self._db = db
        self._limits = limits or CanvasRepositoryLimits()
        if not isinstance(self._limits, CanvasRepositoryLimits):
            raise TypeError("limits must be CanvasRepositoryLimits")

    @contextmanager
    def _owned_immediate_transaction(self) -> Iterator[sqlite3.Cursor]:
        """Own an immediate transaction or reject a pre-existing transaction."""

        connection = self._db.get_connection()
        if connection.in_transaction:
            raise CanvasRepositoryError("transaction_ownership_required")
        with self._db.transaction(immediate=True) as cursor:
            yield cursor

    def create_canvas(
        self,
        conversation_id: str,
        *,
        title: str,
        source: str,
        runtime_profile: CanvasRuntimeProfile,
        actor_kind: CanvasActorKind,
        origin_message_id: str,
        origin_turn_id: str,
        canvas_id: str | None = None,
        revision_id: str | None = None,
        created_at: str | None = None,
        active_message_ids: tuple[str, ...] | None = None,
    ) -> CanvasCreateResult:
        """Append one identity and its root revision in one immediate transaction."""

        return self.append_first_revision(
            conversation_id,
            title=title,
            source=source,
            runtime_profile=runtime_profile,
            actor_kind=actor_kind,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
            canvas_id=canvas_id,
            revision_id=revision_id,
            created_at=created_at,
            active_message_ids=active_message_ids,
        )

    def append_first_revision(
        self,
        conversation_id: str,
        *,
        title: str,
        source: str,
        runtime_profile: CanvasRuntimeProfile,
        actor_kind: CanvasActorKind,
        origin_message_id: str,
        origin_turn_id: str,
        canvas_id: str | None = None,
        revision_id: str | None = None,
        created_at: str | None = None,
        active_message_ids: tuple[str, ...] | None = None,
    ) -> CanvasCreateResult:
        """Append one identity and root revision, recomputing source identity."""

        conversation_id = _validated_owner_id(conversation_id)
        canvas_id = _validated_uuid(canvas_id or str(uuid4()), "canvas_id")
        revision_id = _validated_uuid(revision_id or str(uuid4()), "revision_id")
        timestamp = _validated_timestamp(created_at or _utc_now(), "created_at")
        values = self._validated_revision_values(
            title=title,
            source=source,
            runtime_profile=runtime_profile,
            actor_kind=actor_kind,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
        )

        try:
            with self._owned_immediate_transaction() as cursor:
                self._require_active_owner(cursor, conversation_id)
                canvas_count = int(
                    cursor.execute(
                        "SELECT COUNT(*) FROM canvas_documents WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()[0]
                )
                if canvas_count >= self._limits.max_canvases_per_conversation:
                    raise CanvasQuotaError("canvas_count")
                self._require_conversation_source_capacity(
                    cursor, conversation_id, values.source_bytes
                )
                self._require_ids_available(cursor, (canvas_id,), (revision_id,))
                if active_message_ids is None:
                    self._require_origin_owner(
                        cursor, conversation_id, values.origin_message_id
                    )
                else:
                    self._require_durable_active_path(
                        cursor,
                        conversation_id,
                        active_message_ids,
                        origin_message_id=values.origin_message_id,
                    )
                cursor.execute(
                    "INSERT INTO canvas_documents "
                    "(id, conversation_id, created_at, deleted, deleted_at) "
                    "VALUES (?, ?, ?, 0, NULL)",
                    (canvas_id, conversation_id, timestamp),
                )
                cursor.execute(
                    _INSERT_REVISION_SQL,
                    (
                        revision_id,
                        canvas_id,
                        None,
                        1,
                        values.title,
                        values.runtime_profile,
                        values.source,
                        values.content_sha256,
                        values.source_bytes,
                        values.actor_kind,
                        values.origin_message_id,
                        values.origin_turn_id,
                        timestamp,
                        None,
                    ),
                )
        except CanvasRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc

        identity = CanvasIdentity(canvas_id, conversation_id, timestamp, None)
        revision = CanvasRevision(
            revision_id=revision_id,
            canvas_id=canvas_id,
            parent_revision_id=None,
            sequence=1,
            title=values.title,
            runtime_profile=values.runtime_profile,
            source=values.source,
            content_sha256=values.content_sha256,
            source_bytes=values.source_bytes,
            actor_kind=values.actor_kind,
            origin_message_id=values.origin_message_id,
            origin_turn_id=values.origin_turn_id,
            created_at=timestamp,
            deleted_at=None,
        )
        return CanvasCreateResult(identity, revision)

    def append_revision(
        self,
        conversation_id: str,
        canvas_id: str,
        *,
        parent_revision_id: str,
        title: str,
        source: str,
        runtime_profile: CanvasRuntimeProfile,
        actor_kind: CanvasActorKind,
        origin_message_id: str,
        origin_turn_id: str,
        revision_id: str | None = None,
        created_at: str | None = None,
        active_message_ids: tuple[str, ...] | None = None,
    ) -> CanvasRevision:
        """Append an immutable child and allocate its sequence under a write lock."""

        conversation_id = _validated_owner_id(conversation_id)
        canvas_id = _validated_uuid(canvas_id, "canvas_id")
        parent_revision_id = _validated_uuid(parent_revision_id, "parent_revision_id")
        revision_id = _validated_uuid(revision_id or str(uuid4()), "revision_id")
        timestamp = _validated_timestamp(created_at or _utc_now(), "created_at")
        values = self._validated_revision_values(
            title=title,
            source=source,
            runtime_profile=runtime_profile,
            actor_kind=actor_kind,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
        )

        try:
            with self._owned_immediate_transaction() as cursor:
                self._require_active_document(cursor, conversation_id, canvas_id)
                parent = cursor.execute(
                    "SELECT canvas_id FROM canvas_revisions WHERE id = ?",
                    (parent_revision_id,),
                ).fetchone()
                if parent is None or str(parent[0]) != canvas_id:
                    raise CanvasValidationError("parent_owner_mismatch")
                revision_count, max_sequence = cursor.execute(
                    "SELECT COUNT(*), COALESCE(MAX(sequence), 0) "
                    "FROM canvas_revisions WHERE canvas_id = ?",
                    (canvas_id,),
                ).fetchone()
                if int(revision_count) >= self._limits.max_revisions_per_canvas:
                    raise CanvasQuotaError("revision_count")
                self._require_conversation_source_capacity(
                    cursor, conversation_id, values.source_bytes
                )
                self._require_ids_available(cursor, (), (revision_id,))
                if active_message_ids is None:
                    self._require_origin_owner(
                        cursor, conversation_id, values.origin_message_id
                    )
                else:
                    self._require_durable_active_path(
                        cursor,
                        conversation_id,
                        active_message_ids,
                        origin_message_id=values.origin_message_id,
                    )
                sequence = int(max_sequence) + 1
                cursor.execute(
                    _INSERT_REVISION_SQL,
                    (
                        revision_id,
                        canvas_id,
                        parent_revision_id,
                        sequence,
                        values.title,
                        values.runtime_profile,
                        values.source,
                        values.content_sha256,
                        values.source_bytes,
                        values.actor_kind,
                        values.origin_message_id,
                        values.origin_turn_id,
                        timestamp,
                        None,
                    ),
                )
        except CanvasRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc

        return CanvasRevision(
            revision_id=revision_id,
            canvas_id=canvas_id,
            parent_revision_id=parent_revision_id,
            sequence=sequence,
            title=values.title,
            runtime_profile=values.runtime_profile,
            source=values.source,
            content_sha256=values.content_sha256,
            source_bytes=values.source_bytes,
            actor_kind=values.actor_kind,
            origin_message_id=values.origin_message_id,
            origin_turn_id=values.origin_turn_id,
            created_at=timestamp,
            deleted_at=None,
        )

    def list_identities(
        self,
        conversation_id: str,
        *,
        include_deleted: bool = False,
    ) -> tuple[CanvasIdentity, ...]:
        """List identities for one active conversation without selecting a head."""

        conversation_id = _validated_owner_id(conversation_id)
        deleted_filter = "" if include_deleted else "AND document.deleted = 0"
        try:
            rows = (
                self._db.get_connection()
                .execute(
                    "SELECT document.id, document.conversation_id, document.created_at, "
                    "document.deleted_at FROM canvas_documents AS document "
                    "JOIN conversations AS conversation "
                    "ON conversation.id = document.conversation_id "
                    "WHERE document.conversation_id = ? AND conversation.deleted = 0 "
                    f"{deleted_filter} ORDER BY document.created_at, document.id",
                    (conversation_id,),
                )
                .fetchall()
            )
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc
        return tuple(_identity_from_row(row) for row in rows)

    def list_revision_metadata(
        self,
        conversation_id: str,
        *,
        include_deleted: bool = False,
    ) -> tuple[CanvasRevisionMetadata, ...]:
        """List source-free revision metadata for one active conversation."""

        conversation_id = _validated_owner_id(conversation_id)
        deleted_filter = (
            ""
            if include_deleted
            else "AND document.deleted = 0 AND revision.deleted_at IS NULL"
        )
        try:
            rows = (
                self._db.get_connection()
                .execute(
                    "SELECT revision.id AS revision_id, revision.canvas_id, "
                    "document.conversation_id, revision.parent_revision_id, "
                    "revision.sequence, revision.title, revision.runtime_profile, "
                    "revision.content_sha256, revision.html_bytes, revision.actor_kind, "
                    "revision.origin_message_id, revision.origin_turn_id, "
                    "document.created_at AS canvas_created_at, "
                    "document.deleted_at AS canvas_deleted_at, "
                    "revision.created_at AS revision_created_at, "
                    "revision.deleted_at AS revision_deleted_at "
                    "FROM canvas_documents AS document "
                    "JOIN conversations AS conversation "
                    "ON conversation.id = document.conversation_id "
                    "JOIN canvas_revisions AS revision ON revision.canvas_id = document.id "
                    "WHERE document.conversation_id = ? AND conversation.deleted = 0 "
                    f"{deleted_filter} "
                    "ORDER BY document.created_at, document.id, revision.sequence",
                    (conversation_id,),
                )
                .fetchall()
            )
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc
        return tuple(_metadata_from_row(row) for row in rows)

    def read_revision(
        self,
        conversation_id: str,
        revision_id: str,
        *,
        include_deleted: bool = False,
    ) -> CanvasRevision:
        """Read one exact owner-scoped revision without resolving an active path."""

        conversation_id = _validated_owner_id(conversation_id)
        revision_id = _validated_uuid(revision_id, "revision_id")
        deleted_filter = (
            ""
            if include_deleted
            else "AND document.deleted = 0 AND revision.deleted_at IS NULL"
        )
        try:
            row = (
                self._db.get_connection()
                .execute(
                    "SELECT revision.id, revision.canvas_id, revision.parent_revision_id, "
                    "revision.sequence, revision.title, revision.runtime_profile, "
                    "revision.html, revision.content_sha256, revision.html_bytes, "
                    "revision.actor_kind, revision.origin_message_id, "
                    "revision.origin_turn_id, revision.created_at, revision.deleted_at "
                    "FROM canvas_revisions AS revision "
                    "JOIN canvas_documents AS document ON document.id = revision.canvas_id "
                    "JOIN conversations AS conversation "
                    "ON conversation.id = document.conversation_id "
                    "WHERE document.conversation_id = ? AND revision.id = ? "
                    "AND conversation.deleted = 0 "
                    f"{deleted_filter} LIMIT 1",
                    (conversation_id, revision_id),
                )
                .fetchone()
            )
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc
        if row is None:
            raise CanvasNotFoundError("revision_not_found")
        return _revision_from_row(row)

    def soft_delete_canvas(
        self, conversation_id: str, canvas_id: str
    ) -> CanvasIdentity:
        """Soft-delete one identity while retaining every immutable revision."""

        return self._set_canvas_deleted(conversation_id, canvas_id, deleted=True)

    def restore_canvas(self, conversation_id: str, canvas_id: str) -> CanvasIdentity:
        """Restore one soft-deleted identity under its active owner."""

        return self._set_canvas_deleted(conversation_id, canvas_id, deleted=False)

    def set_reopen_hint(self, conversation_id: str, canvas_id: str | None) -> None:
        """Set or clear the local last-used Canvas hint for one conversation."""

        conversation_id = _validated_owner_id(conversation_id)
        if canvas_id is not None:
            canvas_id = _validated_uuid(canvas_id, "canvas_id")
        try:
            with self._owned_immediate_transaction() as cursor:
                self._require_active_owner(cursor, conversation_id)
                if canvas_id is None:
                    cursor.execute(
                        "DELETE FROM canvas_conversation_hints WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                    return
                self._require_active_document(cursor, conversation_id, canvas_id)
                cursor.execute(
                    "INSERT INTO canvas_conversation_hints "
                    "(conversation_id, last_canvas_id, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(conversation_id) DO UPDATE SET "
                    "last_canvas_id = excluded.last_canvas_id, "
                    "updated_at = excluded.updated_at",
                    (conversation_id, canvas_id, _utc_now()),
                )
        except CanvasRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc

    def get_reopen_hint(self, conversation_id: str) -> str | None:
        """Read the local last-used hint only while owner and Canvas are active."""

        conversation_id = _validated_owner_id(conversation_id)
        try:
            row = (
                self._db.get_connection()
                .execute(
                    "SELECT hint.last_canvas_id FROM canvas_conversation_hints AS hint "
                    "JOIN conversations AS conversation "
                    "ON conversation.id = hint.conversation_id "
                    "JOIN canvas_documents AS document "
                    "ON document.id = hint.last_canvas_id "
                    "AND document.conversation_id = hint.conversation_id "
                    "WHERE hint.conversation_id = ? AND conversation.deleted = 0 "
                    "AND document.deleted = 0",
                    (conversation_id,),
                )
                .fetchone()
            )
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc
        return str(row[0]) if row is not None else None

    def hard_purge_conversation(self, conversation_id: str) -> CanvasPurgeResult:
        """Delete one owner and its exact Canvas graph through a narrow capability."""

        conversation_id = _validated_owner_id(conversation_id)
        try:
            with self._owned_immediate_transaction() as cursor:
                owner = cursor.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if owner is None:
                    raise CanvasNotFoundError("conversation_not_found")
                canvas_ids = tuple(
                    str(row[0])
                    for row in cursor.execute(
                        "SELECT id FROM canvas_documents WHERE conversation_id = ? "
                        "ORDER BY id",
                        (conversation_id,),
                    ).fetchall()
                )
                revision_count = int(
                    cursor.execute(
                        "SELECT COUNT(*) FROM canvas_revisions AS revision "
                        "JOIN canvas_documents AS document "
                        "ON document.id = revision.canvas_id "
                        "WHERE document.conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()[0]
                )
                if canvas_ids:
                    authorization = (
                        self._db._canvas_revision_deletion_authorization_for_repository(
                            cursor.connection
                        )
                    )
                    with authorization.authorize(cursor, canvas_ids):
                        cursor.execute(
                            "DELETE FROM canvas_documents WHERE conversation_id = ?",
                            (conversation_id,),
                        )
                message_rows = cursor.execute(
                    "SELECT id, parent_message_id FROM messages "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchall()
                if message_rows:
                    from tldw_chatbook.Chat.console_semantic_revision import (
                        SemanticRevisionCoordinator,
                    )

                    coordinator = SemanticRevisionCoordinator(self._db)
                    for message_id in _children_before_parents(message_rows):
                        coordinator.mutate_message(
                            cursor,
                            message_id=message_id,
                            creation_reason="canvas_owner_purge",
                            hard_delete=True,
                        )
                deleted = cursor.execute(
                    "DELETE FROM conversations WHERE id = ?", (conversation_id,)
                )
                if deleted.rowcount != 1:
                    raise CanvasConflictError("owner_purge_conflict")
        except CanvasRepositoryError:
            raise
        except (sqlite3.Error, RuntimeError) as exc:
            raise CanvasRepositoryError("storage_failure") from exc
        return CanvasPurgeResult(
            conversation_id=conversation_id,
            canvases_deleted=len(canvas_ids),
            revisions_deleted=revision_count,
        )

    def import_batch(self, batch: CanvasImportBatch) -> CanvasImportResult:
        """Validate a complete stable graph before mutating, then insert atomically."""

        if not isinstance(batch, CanvasImportBatch):
            raise CanvasValidationError("invalid_import_batch")
        conversation_id = _validated_owner_id(batch.conversation_id)
        try:
            with self._owned_immediate_transaction() as cursor:
                self._require_active_owner(cursor, conversation_id)
                self._validate_import_batch(cursor, batch)
                for document in batch.documents:
                    cursor.execute(
                        "INSERT INTO canvas_documents "
                        "(id, conversation_id, created_at, deleted, deleted_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            document.canvas_id,
                            document.conversation_id,
                            document.created_at,
                            int(document.deleted_at is not None),
                            document.deleted_at,
                        ),
                    )
                for revision in sorted(
                    batch.revisions, key=lambda item: (item.canvas_id, item.sequence)
                ):
                    cursor.execute(
                        _INSERT_REVISION_SQL,
                        (
                            revision.revision_id,
                            revision.canvas_id,
                            revision.parent_revision_id,
                            revision.sequence,
                            revision.title,
                            revision.runtime_profile,
                            revision.source,
                            revision.content_sha256,
                            revision.source_bytes,
                            revision.actor_kind,
                            revision.origin_message_id,
                            revision.origin_turn_id,
                            revision.created_at,
                            revision.deleted_at,
                        ),
                    )
                if batch.reopen_canvas_id is not None:
                    cursor.execute(
                        "INSERT INTO canvas_conversation_hints "
                        "(conversation_id, last_canvas_id, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(conversation_id) DO UPDATE SET "
                        "last_canvas_id = excluded.last_canvas_id, "
                        "updated_at = excluded.updated_at",
                        (conversation_id, batch.reopen_canvas_id, _utc_now()),
                    )
        except CanvasRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc
        return CanvasImportResult(
            conversation_id=conversation_id,
            canvases_imported=len(batch.documents),
            revisions_imported=len(batch.revisions),
        )

    @classmethod
    def append_batch_in_transaction(
        cls,
        cursor: sqlite3.Cursor,
        batch: CanvasImportBatch,
        *,
        anchor_message_id: str,
        require_active_path: bool = True,
        limits: CanvasRepositoryLimits | None = None,
    ) -> CanvasImportResult:
        """Validate and append a partial graph inside a caller-owned transaction.

        Unlike :meth:`import_batch`, documents may be omitted for updates to an
        existing Canvas.  The assistant anchor is already written by the caller;
        its durable active path is re-derived under the same SQLite write lock.
        """

        if not cursor.connection.in_transaction:
            raise CanvasRepositoryError("transaction_ownership_required")
        if not isinstance(batch, CanvasImportBatch):
            raise CanvasValidationError("invalid_import_batch")
        repository = object.__new__(cls)
        repository._limits = limits or CanvasRepositoryLimits()
        conversation_id = _validated_owner_id(batch.conversation_id)
        repository._require_active_owner(cursor, conversation_id)
        repository._validate_append_batch(
            cursor,
            batch,
            anchor_message_id=anchor_message_id,
            require_active_path=require_active_path,
        )
        for document in batch.documents:
            cursor.execute(
                "INSERT INTO canvas_documents "
                "(id, conversation_id, created_at, deleted, deleted_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    document.canvas_id,
                    document.conversation_id,
                    document.created_at,
                    int(document.deleted_at is not None),
                    document.deleted_at,
                ),
            )
        for revision in sorted(
            batch.revisions, key=lambda item: (item.canvas_id, item.sequence)
        ):
            cursor.execute(
                _INSERT_REVISION_SQL,
                (
                    revision.revision_id,
                    revision.canvas_id,
                    revision.parent_revision_id,
                    revision.sequence,
                    revision.title,
                    revision.runtime_profile,
                    revision.source,
                    revision.content_sha256,
                    revision.source_bytes,
                    revision.actor_kind,
                    revision.origin_message_id,
                    revision.origin_turn_id,
                    revision.created_at,
                    revision.deleted_at,
                ),
            )
        return CanvasImportResult(
            conversation_id=conversation_id,
            canvases_imported=len(batch.documents),
            revisions_imported=len(batch.revisions),
        )

    def _validate_append_batch(
        self,
        cursor: sqlite3.Cursor,
        batch: CanvasImportBatch,
        *,
        anchor_message_id: str,
        require_active_path: bool,
    ) -> None:
        if type(batch.documents) is not tuple or type(batch.revisions) is not tuple:
            raise CanvasValidationError("invalid_import_batch")
        if not batch.revisions:
            raise CanvasValidationError("invalid_import_revisions")
        anchor_message_id = _validated_opaque_id(
            anchor_message_id, "origin_message_id", 256
        )
        path: list[str] = []
        seen: set[str] = set()
        current: str | None = anchor_message_id
        while current is not None:
            if current in seen:
                raise CanvasValidationError("invalid_active_path")
            seen.add(current)
            row = cursor.execute(
                "SELECT parent_message_id, conversation_id, deleted FROM messages "
                "WHERE id = ?",
                (current,),
            ).fetchone()
            if row is None or str(row[1]) != batch.conversation_id or int(row[2]):
                raise CanvasValidationError("invalid_active_path")
            path.append(current)
            if len(path) > MAX_CANVAS_DURABLE_ACTIVE_PATH_MESSAGES:
                raise CanvasValidationError("invalid_active_path")
            current = str(row[0]) if row[0] is not None else None
        reachable = set(path)

        new_documents: set[str] = set()
        for document in batch.documents:
            if not isinstance(document, CanvasImportDocument):
                raise CanvasValidationError("invalid_import_document")
            canvas_id = _validated_uuid(document.canvas_id, "canvas_id")
            if canvas_id in new_documents:
                raise CanvasValidationError("duplicate_import_identity")
            if document.conversation_id != batch.conversation_id:
                raise CanvasValidationError("document_owner_mismatch")
            _validated_timestamp(document.created_at, "created_at")
            if document.deleted_at is not None:
                raise CanvasValidationError("invalid_import_document")
            new_documents.add(canvas_id)
        count = int(
            cursor.execute(
                "SELECT COUNT(*) FROM canvas_documents WHERE conversation_id = ?",
                (batch.conversation_id,),
            ).fetchone()[0]
        )
        if count + len(new_documents) > self._limits.max_canvases_per_conversation:
            raise CanvasQuotaError("canvas_count")

        revision_ids: set[str] = set()
        batch_rows: dict[str, CanvasImportRevision] = {}
        per_canvas: dict[str, list[CanvasImportRevision]] = {}
        added_bytes = 0
        for revision in batch.revisions:
            if not isinstance(revision, CanvasImportRevision):
                raise CanvasValidationError("invalid_import_revision")
            revision_id = _validated_uuid(revision.revision_id, "revision_id")
            if revision_id in revision_ids or revision_id in new_documents:
                raise CanvasValidationError("duplicate_import_identity")
            if type(revision.sequence) is not int or revision.sequence <= 0:
                raise CanvasValidationError("invalid_revision_sequence")
            if revision.parent_revision_id is not None:
                _validated_uuid(revision.parent_revision_id, "parent_revision_id")
            if revision.deleted_at is not None:
                raise CanvasValidationError("invalid_import_revision")
            if require_active_path and revision.origin_message_id not in reachable:
                raise CanvasValidationError("origin_owner_mismatch")
            self._require_origin_owner(
                cursor, batch.conversation_id, revision.origin_message_id
            )
            if revision.canvas_id not in new_documents:
                self._require_active_document(
                    cursor, batch.conversation_id, revision.canvas_id
                )
            values = self._validated_revision_values(
                title=revision.title,
                source=revision.source,
                runtime_profile=revision.runtime_profile,
                actor_kind=revision.actor_kind,
                origin_message_id=revision.origin_message_id,
                origin_turn_id=revision.origin_turn_id,
            )
            if (
                revision.content_sha256 != values.content_sha256
                or revision.source_bytes != values.source_bytes
            ):
                raise CanvasValidationError("digest_mismatch")
            _validated_timestamp(revision.created_at, "created_at")
            revision_ids.add(revision_id)
            batch_rows[revision_id] = revision
            per_canvas.setdefault(revision.canvas_id, []).append(revision)
            added_bytes += revision.source_bytes

        if new_documents - per_canvas.keys():
            raise CanvasValidationError("canvas_without_revision")

        self._require_ids_available(cursor, tuple(new_documents), tuple(revision_ids))
        self._require_conversation_source_capacity(
            cursor, batch.conversation_id, added_bytes
        )
        for canvas_id, revisions in per_canvas.items():
            existing_count = int(
                cursor.execute(
                    "SELECT COUNT(*) FROM canvas_revisions WHERE canvas_id = ?",
                    (canvas_id,),
                ).fetchone()[0]
            )
            if existing_count + len(revisions) > self._limits.max_revisions_per_canvas:
                raise CanvasQuotaError("revision_count")
            expected_sequence = existing_count + 1
            for revision in sorted(revisions, key=lambda item: item.sequence):
                if revision.sequence != expected_sequence:
                    raise CanvasValidationError("invalid_revision_sequence")
                if revision.parent_revision_id is None:
                    if canvas_id not in new_documents or revision.sequence != 1:
                        raise CanvasValidationError("invalid_root_parent")
                    expected_sequence += 1
                    continue
                parent = batch_rows.get(revision.parent_revision_id)
                if parent is None:
                    row = cursor.execute(
                        "SELECT sequence, canvas_id, origin_message_id FROM canvas_revisions "
                        "WHERE id = ? AND deleted_at IS NULL",
                        (revision.parent_revision_id,),
                    ).fetchone()
                    if (
                        row is None
                        or str(row[1]) != canvas_id
                        or (require_active_path and str(row[2]) not in reachable)
                    ):
                        raise CanvasValidationError("parent_owner_mismatch")
                    parent_sequence = int(row[0])
                else:
                    if parent.canvas_id != canvas_id:
                        raise CanvasValidationError("parent_owner_mismatch")
                    parent_sequence = parent.sequence
                if parent_sequence >= revision.sequence:
                    raise CanvasValidationError("parent_owner_mismatch")
                expected_sequence += 1

    def _validated_revision_values(
        self,
        *,
        title: str,
        source: str,
        runtime_profile: object,
        actor_kind: object,
        origin_message_id: str,
        origin_turn_id: str,
    ) -> _RevisionValues:
        title = _validated_text(title, "title", self._limits.max_title_bytes)
        try:
            source_bytes = utf8_byte_length(source)
        except CanvasLimitError as exc:
            raise CanvasValidationError("invalid_source") from exc
        if source_bytes > self._limits.max_source_bytes_per_revision:
            raise CanvasQuotaError("revision_source_bytes")
        if runtime_profile != "canvas-v1":
            raise CanvasValidationError("unsupported_runtime_profile")
        if type(actor_kind) is not str or actor_kind not in _ACTOR_KINDS:
            raise CanvasValidationError("invalid_actor_kind")
        origin_message_id = _validated_opaque_id(
            origin_message_id, "origin_message_id", 256
        )
        origin_turn_id = _validated_opaque_id(
            origin_turn_id,
            "origin_turn_id",
            self._limits.max_origin_turn_id_bytes,
        )
        return _RevisionValues(
            title=title,
            source=source,
            runtime_profile="canvas-v1",
            actor_kind=actor_kind,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
            content_sha256=sha256_utf8(source),
            source_bytes=source_bytes,
        )

    def _set_canvas_deleted(
        self,
        conversation_id: str,
        canvas_id: str,
        *,
        deleted: bool,
    ) -> CanvasIdentity:
        conversation_id = _validated_owner_id(conversation_id)
        canvas_id = _validated_uuid(canvas_id, "canvas_id")
        try:
            with self._owned_immediate_transaction() as cursor:
                self._require_active_owner(cursor, conversation_id)
                row = cursor.execute(
                    "SELECT id, conversation_id, created_at, deleted, deleted_at "
                    "FROM canvas_documents WHERE id = ? AND conversation_id = ?",
                    (canvas_id, conversation_id),
                ).fetchone()
                if row is None:
                    raise CanvasNotFoundError("canvas_not_found")
                requested = 1 if deleted else 0
                if int(row[3]) != requested:
                    deleted_at = _utc_now() if deleted else None
                    cursor.execute(
                        "UPDATE canvas_documents SET deleted = ?, deleted_at = ? "
                        "WHERE id = ? AND conversation_id = ?",
                        (requested, deleted_at, canvas_id, conversation_id),
                    )
                    if deleted:
                        cursor.execute(
                            "DELETE FROM canvas_conversation_hints "
                            "WHERE conversation_id = ? AND last_canvas_id = ?",
                            (conversation_id, canvas_id),
                        )
                    row = cursor.execute(
                        "SELECT id, conversation_id, created_at, deleted, deleted_at "
                        "FROM canvas_documents WHERE id = ? AND conversation_id = ?",
                        (canvas_id, conversation_id),
                    ).fetchone()
        except CanvasRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CanvasRepositoryError("storage_failure") from exc
        assert row is not None
        return _identity_from_row(row)

    def _validate_import_batch(
        self, cursor: sqlite3.Cursor, batch: CanvasImportBatch
    ) -> None:
        if type(batch.documents) is not tuple or not batch.documents:
            raise CanvasValidationError("invalid_import_documents")
        if type(batch.revisions) is not tuple or not batch.revisions:
            raise CanvasValidationError("invalid_import_revisions")

        document_ids: set[str] = set()
        document_deleted: dict[str, bool] = {}
        for document in batch.documents:
            if not isinstance(document, CanvasImportDocument):
                raise CanvasValidationError("invalid_import_document")
            canvas_id = _validated_uuid(document.canvas_id, "canvas_id")
            if canvas_id in document_ids:
                raise CanvasValidationError("duplicate_import_identity")
            if document.conversation_id != batch.conversation_id:
                raise CanvasValidationError("document_owner_mismatch")
            _validated_timestamp(document.created_at, "created_at")
            _validated_optional_timestamp(document.deleted_at, "deleted_at")
            document_ids.add(canvas_id)
            document_deleted[canvas_id] = document.deleted_at is not None

        existing_canvas_count = int(
            cursor.execute(
                "SELECT COUNT(*) FROM canvas_documents WHERE conversation_id = ?",
                (batch.conversation_id,),
            ).fetchone()[0]
        )
        if (
            existing_canvas_count + len(document_ids)
            > self._limits.max_canvases_per_conversation
        ):
            raise CanvasQuotaError("canvas_count")

        revision_ids: set[str] = set()
        revisions_by_id: dict[str, CanvasImportRevision] = {}
        revisions_by_canvas: dict[str, list[CanvasImportRevision]] = {
            canvas_id: [] for canvas_id in document_ids
        }
        total_source_bytes = 0
        for revision in batch.revisions:
            if not isinstance(revision, CanvasImportRevision):
                raise CanvasValidationError("invalid_import_revision")
            revision_id = _validated_uuid(revision.revision_id, "revision_id")
            if revision_id in revision_ids or revision_id in document_ids:
                raise CanvasValidationError("duplicate_import_identity")
            if revision.canvas_id not in document_ids:
                raise CanvasValidationError("revision_owner_mismatch")
            if type(revision.sequence) is not int or revision.sequence <= 0:
                raise CanvasValidationError("invalid_revision_sequence")
            values = self._validated_revision_values(
                title=revision.title,
                source=revision.source,
                runtime_profile=revision.runtime_profile,
                actor_kind=revision.actor_kind,
                origin_message_id=revision.origin_message_id,
                origin_turn_id=revision.origin_turn_id,
            )
            if (
                revision.content_sha256 != values.content_sha256
                or type(revision.source_bytes) is not int
                or revision.source_bytes != values.source_bytes
            ):
                raise CanvasValidationError("digest_mismatch")
            _validated_timestamp(revision.created_at, "created_at")
            _validated_optional_timestamp(revision.deleted_at, "deleted_at")
            self._require_origin_owner(
                cursor, batch.conversation_id, revision.origin_message_id
            )
            revision_ids.add(revision_id)
            revisions_by_id[revision_id] = revision
            revisions_by_canvas[revision.canvas_id].append(revision)
            total_source_bytes += revision.source_bytes

        for canvas_id, revisions in revisions_by_canvas.items():
            if not revisions:
                raise CanvasValidationError("canvas_without_revision")
            if len(revisions) > self._limits.max_revisions_per_canvas:
                raise CanvasQuotaError("revision_count")
            ordered = sorted(revisions, key=lambda item: item.sequence)
            if [item.sequence for item in ordered] != list(range(1, len(ordered) + 1)):
                raise CanvasValidationError("invalid_revision_sequence")
            for revision in ordered:
                if revision.sequence == 1:
                    if revision.parent_revision_id is not None:
                        raise CanvasValidationError("invalid_root_parent")
                    continue
                if revision.parent_revision_id is None:
                    raise CanvasValidationError("missing_parent")
                parent = revisions_by_id.get(revision.parent_revision_id)
                if (
                    parent is None
                    or parent.canvas_id != canvas_id
                    or parent.sequence >= revision.sequence
                ):
                    raise CanvasValidationError("parent_owner_mismatch")

        if batch.reopen_canvas_id is not None:
            _validated_uuid(batch.reopen_canvas_id, "reopen_canvas_id")
            if (
                batch.reopen_canvas_id not in document_ids
                or document_deleted[batch.reopen_canvas_id]
            ):
                raise CanvasValidationError("invalid_reopen_hint")

        self._require_ids_available(cursor, tuple(document_ids), tuple(revision_ids))
        self._require_conversation_source_capacity(
            cursor, batch.conversation_id, total_source_bytes
        )

    @staticmethod
    def _require_active_owner(cursor: sqlite3.Cursor, conversation_id: str) -> None:
        row = cursor.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND deleted = 0",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise CanvasNotFoundError("conversation_not_found")

    @staticmethod
    def _require_active_document(
        cursor: sqlite3.Cursor, conversation_id: str, canvas_id: str
    ) -> None:
        row = cursor.execute(
            "SELECT 1 FROM canvas_documents AS document "
            "JOIN conversations AS conversation "
            "ON conversation.id = document.conversation_id "
            "WHERE document.id = ? AND document.conversation_id = ? "
            "AND document.deleted = 0 AND conversation.deleted = 0",
            (canvas_id, conversation_id),
        ).fetchone()
        if row is None:
            raise CanvasNotFoundError("canvas_not_found")

    @staticmethod
    def _require_origin_owner(
        cursor: sqlite3.Cursor, conversation_id: str, origin_message_id: str
    ) -> None:
        row = cursor.execute(
            "SELECT 1 FROM messages WHERE id = ? AND conversation_id = ? "
            "AND deleted = 0",
            (origin_message_id, conversation_id),
        ).fetchone()
        if row is None:
            raise CanvasValidationError("origin_owner_mismatch")

    @staticmethod
    def _require_durable_active_path(
        cursor: sqlite3.Cursor,
        conversation_id: str,
        active_message_ids: tuple[str, ...],
        *,
        origin_message_id: str,
    ) -> None:
        """Revalidate one complete persisted path under the owned write lock."""

        if (
            type(active_message_ids) is not tuple
            or not active_message_ids
            or len(active_message_ids) > MAX_CANVAS_DURABLE_ACTIVE_PATH_MESSAGES
        ):
            raise CanvasValidationError("invalid_active_path")
        if len(set(active_message_ids)) != len(active_message_ids):
            raise CanvasValidationError("invalid_active_path")
        try:
            for message_id in active_message_ids:
                _validated_opaque_id(message_id, "active_message_id", 256)
        except CanvasValidationError:
            raise CanvasValidationError("invalid_active_path") from None
        if active_message_ids[-1] != origin_message_id:
            raise CanvasValidationError("invalid_active_path")

        placeholders = ", ".join("?" for _ in active_message_ids)
        rows = cursor.execute(
            "SELECT id, parent_message_id, conversation_id, deleted FROM messages "
            f"WHERE id IN ({placeholders})",
            active_message_ids,
        ).fetchall()
        by_id = {str(row[0]): row for row in rows}
        if len(by_id) != len(active_message_ids):
            raise CanvasValidationError("invalid_active_path")

        previous_id: str | None = None
        for message_id in active_message_ids:
            row = by_id[message_id]
            parent_id = str(row[1]) if row[1] is not None else None
            if (
                str(row[2]) != conversation_id
                or int(row[3]) != 0
                or parent_id != previous_id
            ):
                raise CanvasValidationError("invalid_active_path")
            previous_id = message_id

    def _require_conversation_source_capacity(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        added_bytes: int,
    ) -> None:
        current_bytes = int(
            cursor.execute(
                "SELECT COALESCE(SUM(revision.html_bytes), 0) "
                "FROM canvas_revisions AS revision "
                "JOIN canvas_documents AS document "
                "ON document.id = revision.canvas_id "
                "WHERE document.conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
        )
        if current_bytes + added_bytes > self._limits.max_source_bytes_per_conversation:
            raise CanvasQuotaError("conversation_source_bytes")

    @staticmethod
    def _require_ids_available(
        cursor: sqlite3.Cursor,
        canvas_ids: tuple[str, ...],
        revision_ids: tuple[str, ...],
    ) -> None:
        all_ids = (*canvas_ids, *revision_ids)
        if len(set(all_ids)) != len(all_ids):
            raise CanvasValidationError("duplicate_import_identity")
        if not all_ids:
            return
        placeholders = ", ".join("?" for _ in all_ids)
        document_collision = cursor.execute(
            f"SELECT 1 FROM canvas_documents WHERE id IN ({placeholders}) LIMIT 1",
            all_ids,
        ).fetchone()
        revision_collision = cursor.execute(
            f"SELECT 1 FROM canvas_revisions WHERE id IN ({placeholders}) LIMIT 1",
            all_ids,
        ).fetchone()
        if document_collision is not None or revision_collision is not None:
            raise CanvasConflictError("identity_collision")


@dataclass(frozen=True, slots=True)
class _RevisionValues:
    title: str
    source: str
    runtime_profile: CanvasRuntimeProfile
    actor_kind: CanvasActorKind
    origin_message_id: str
    origin_turn_id: str
    content_sha256: str
    source_bytes: int


_INSERT_REVISION_SQL = (
    "INSERT INTO canvas_revisions "
    "(id, canvas_id, parent_revision_id, sequence, title, runtime_profile, "
    "html, content_sha256, html_bytes, actor_kind, origin_message_id, "
    "origin_turn_id, created_at, deleted_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _identity_from_row(row: sqlite3.Row) -> CanvasIdentity:
    deleted_at = row[4] if len(row) > 4 else row[3]
    return CanvasIdentity(
        canvas_id=str(row[0]),
        conversation_id=str(row[1]),
        created_at=str(row[2]),
        deleted_at=str(deleted_at) if deleted_at is not None else None,
    )


def _revision_from_row(row: sqlite3.Row) -> CanvasRevision:
    return CanvasRevision(
        revision_id=str(row[0]),
        canvas_id=str(row[1]),
        parent_revision_id=str(row[2]) if row[2] is not None else None,
        sequence=int(row[3]),
        title=str(row[4]),
        runtime_profile=str(row[5]),  # type: ignore[arg-type]
        source=str(row[6]),
        content_sha256=str(row[7]),
        source_bytes=int(row[8]),
        actor_kind=str(row[9]),  # type: ignore[arg-type]
        origin_message_id=str(row[10]),
        origin_turn_id=str(row[11]),
        created_at=str(row[12]),
        deleted_at=str(row[13]) if row[13] is not None else None,
    )


def _metadata_from_row(row: sqlite3.Row) -> CanvasRevisionMetadata:
    return CanvasRevisionMetadata(
        revision_id=str(row[0]),
        canvas_id=str(row[1]),
        conversation_id=str(row[2]),
        parent_revision_id=str(row[3]) if row[3] is not None else None,
        sequence=int(row[4]),
        title=str(row[5]),
        runtime_profile=str(row[6]),  # type: ignore[arg-type]
        content_sha256=str(row[7]),
        source_bytes=int(row[8]),
        actor_kind=str(row[9]),  # type: ignore[arg-type]
        origin_message_id=str(row[10]),
        origin_turn_id=str(row[11]),
        canvas_created_at=str(row[12]),
        canvas_deleted_at=str(row[13]) if row[13] is not None else None,
        revision_created_at=str(row[14]),
        revision_deleted_at=str(row[15]) if row[15] is not None else None,
    )


def _validated_owner_id(value: object) -> str:
    return _validated_opaque_id(value, "conversation_id", 256)


def _children_before_parents(rows: list[sqlite3.Row]) -> tuple[str, ...]:
    """Order one conversation's message DAG for guarded hard deletion."""

    parents = {str(row[0]): str(row[1]) if row[1] is not None else None for row in rows}
    children: dict[str, list[str]] = {message_id: [] for message_id in parents}
    for message_id, parent_id in parents.items():
        if parent_id in children:
            children[parent_id].append(message_id)

    state: dict[str, int] = {}
    ordered: list[str] = []

    for start_id in sorted(parents):
        if state.get(start_id) == 2:
            continue
        stack = [(start_id, False)]
        while stack:
            message_id, expanded = stack.pop()
            marker = state.get(message_id, 0)
            if expanded:
                state[message_id] = 2
                ordered.append(message_id)
                continue
            if marker == 1:
                raise CanvasConflictError("message_graph_cycle")
            if marker == 2:
                continue
            state[message_id] = 1
            stack.append((message_id, True))
            for child_id in sorted(children[message_id], reverse=True):
                child_marker = state.get(child_id, 0)
                if child_marker == 1:
                    raise CanvasConflictError("message_graph_cycle")
                if child_marker == 0:
                    stack.append((child_id, False))
    return tuple(ordered)


def _validated_opaque_id(value: object, field_name: str, limit: int) -> str:
    if type(value) is not str or not value:
        raise CanvasValidationError(f"invalid_{field_name}")
    try:
        validate_utf8_text(value, limit=limit, field_name=field_name)
    except CanvasLimitError as exc:
        raise CanvasValidationError(f"invalid_{field_name}") from exc
    return value


def _validated_uuid(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CanvasValidationError(f"invalid_{field_name}")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise CanvasValidationError(f"invalid_{field_name}") from None
    if str(parsed) != value:
        raise CanvasValidationError(f"invalid_{field_name}")
    return value


def _validated_text(value: object, field_name: str, limit: int) -> str:
    if type(value) is not str or not value.strip():
        raise CanvasValidationError(f"invalid_{field_name}")
    try:
        validate_utf8_text(value, limit=limit, field_name=field_name)
    except CanvasLimitError as exc:
        raise CanvasQuotaError(f"{field_name}_bytes") from exc
    return value


def _validated_timestamp(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise CanvasValidationError(f"invalid_{field_name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise CanvasValidationError(f"invalid_{field_name}") from None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise CanvasValidationError(f"invalid_{field_name}")
    return value


def _validated_optional_timestamp(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _validated_timestamp(value, field_name)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
