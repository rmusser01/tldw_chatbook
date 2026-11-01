"""SQLite replica, search index, and recovery snapshots for File Notes."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import NamedTuple

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Backup_Recovery.participants import (
    _core_access,
    _core_cached_connection,
    _core_closing,
    _core_getter,
    _core_operation,
    _register_core_connection,
)
from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path

from tldw_chatbook.Utils.fts5_match_forms import quote_fts5_phrase

#: ADR-218: most-recent ``pre_edit`` revisions kept per note (protected
#: paths included -- protected saves are the only checkpoint writer, so
#: exempting them would leave the growth ADR-218 exists to bound unbounded;
#: PR #3016 review finding 14).
MAX_REVISIONS_PER_NOTE = 50
#: ADR-218: tombstones and revisions past this age are expired.
RECOVERY_EXPIRY_DAYS = 30
#: Default bounded page size for revision-history listings; one definition
#: shared by the replica and the service so the two layers cannot drift
#: (PR #3016 review finding 5).
REVISION_HISTORY_DEFAULT_LIMIT = 10
#: Hard ceiling for any caller-supplied history page size, clamped before
#: the query runs (PR #3016 review finding 3).
REVISION_HISTORY_MAX_LIMIT = 200


class ReplicaFileInfo(NamedTuple):
    """Metadata needed to reconcile one active disk file with its replica."""

    relative_path: str
    content_hash: str
    size: int
    mtime_ns: int


class ReplicaRevisionInfo(NamedTuple):
    """One bounded history-listing entry for a replicated file."""

    kind: str
    session_key: str | None
    created_at: str
    content_hash: str
    size: int
    #: Stable revision-row identity (``rowid``): multiple ``delete``
    #: revisions share a NULL session key, so this is how a History entry
    #: selects exactly the row it listed (PR #3016 review finding 2).
    revision_id: int


class ReplicaRevisionBytes(NamedTuple):
    """Exact stored bytes of one revision and their recorded digest."""

    raw_bytes: bytes
    content_hash: str


class FileNotesReplica:
    """Store current File Notes bytes without becoming their editor authority."""

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        """Open the replica and initialize its fixed schema.

        Args:
            db_path: SQLite database path, or ``":memory:"`` for a transient
                replica.
        """
        self.is_memory_db = os.fspath(db_path) == ":memory:"
        self.db_path = ":memory:" if self.is_memory_db else lexical_path(Path(db_path).expanduser())
        self._lock = RLock()
        self._connection = None
        _core_access(self)  # Bind the file selector before either constructor stage.
        if not self.is_memory_db:
            from tldw_chatbook.Backup_Recovery.raw_participants import _scope, _mkdirs

            # Directory effects retire before independent SQLite admission.
            with _scope(self, "file_notes_directory", writing=True) as operation:
                _mkdirs(operation)
        try:
            self._initialize_schema()
        except BaseException:
            self.close()
            raise

    @_core_getter
    def _get_connection(self) -> sqlite3.Connection:
        _core_access(self)
        conn = _core_cached_connection(self, self._connection)
        if conn is None:
            conn = connect_private_sqlite(
                "notes.file_notes_replica", self.db_path,
                isolation_level=None, check_same_thread=False,
            )
            try:
                _register_core_connection(self, conn)
                _core_access(self)
                conn.row_factory = sqlite3.Row
                if not self.is_memory_db:
                    conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                _core_access(self)
            except BaseException:
                conn.close()
                raise
            self._connection = conn
        return conn

    @contextmanager
    def _locked_connection(self) -> Iterator[None]:
        with _core_operation(self), self._lock:
            self._get_connection()
            yield

    def close(self) -> None:
        """Retire only on its creating thread after managed borrowers finish."""
        conn = self._connection
        if conn is not None:
            with _core_closing(self, conn) as allowed:
                if allowed:
                    with self._lock:
                        conn.close()
                        if not self.is_memory_db:
                            self._connection = None

    def upsert_file(
        self,
        root: str,
        relative_path: str,
        raw_bytes: bytes,
        *,
        content_hash: str,
        decoded_text: str | None,
        size: int,
        mtime_ns: int,
    ) -> None:
        """Replace one root-namespaced current-byte replica and its FTS row.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.
            raw_bytes: Exact bytes read from disk.
            content_hash: Digest of ``raw_bytes``.
            decoded_text: Searchable text, or ``None`` for non-text content.
            size: File size in bytes.
            mtime_ns: File modification time in nanoseconds.
        """
        with self._transaction() as cursor:
            self._upsert_file(
                cursor,
                root,
                relative_path,
                raw_bytes,
                content_hash=content_hash,
                decoded_text=decoded_text,
                size=size,
                mtime_ns=mtime_ns,
            )

    def get_bytes(self, root: str, relative_path: str) -> bytes | None:
        """Return exact current or tombstoned bytes for a path.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.

        Returns:
            Stored bytes, or ``None`` when the path is not replicated.
        """
        with self._locked_connection():
            row = self._connection.execute(
                """
                SELECT raw_bytes
                FROM files
                WHERE root = ? AND relative_path = ?
                """,
                (root, relative_path),
            ).fetchone()
        return None if row is None else bytes(row["raw_bytes"])

    def list_active_files(self, root: str) -> list[ReplicaFileInfo]:
        """Return active replica metadata for one canonical root.

        Args:
            root: Canonical notes-root identifier.

        Returns:
            Active files ordered by relative path.
        """
        with self._locked_connection():
            rows = self._connection.execute(
                """
                SELECT relative_path, content_hash, size, mtime_ns
                FROM files
                WHERE root = ? AND deleted_at IS NULL
                ORDER BY relative_path
                """,
                (root,),
            ).fetchall()
        return [
            ReplicaFileInfo(
                relative_path=str(row["relative_path"]),
                content_hash=str(row["content_hash"]),
                size=int(row["size"]),
                mtime_ns=int(row["mtime_ns"]),
            )
            for row in rows
        ]

    def search(self, root: str, query: str, *, limit: int = 50) -> list[str]:
        """Return active paths whose decoded current content matches user text.

        Args:
            root: Canonical notes-root identifier.
            query: User text, matched as ONE quoted literal FTS5 PHRASE
                (``quote_fts5_phrase``) -- the words must be adjacent and in
                order, and FTS5 operators in it are inert.
            limit: Maximum number of paths to return.

        Returns:
            Matching relative paths ordered by relevance.
        """
        query = query.strip()
        if not query or limit <= 0 or "\x00" in query:
            return []
        literal_query = quote_fts5_phrase(query)
        try:
            with self._locked_connection():
                rows = self._connection.execute(
                    """
                    SELECT files.relative_path
                    FROM files_fts
                    JOIN files
                      ON files.root = files_fts.root
                     AND files.relative_path = files_fts.relative_path
                    WHERE files_fts MATCH ?
                      AND files.root = ?
                      AND files.deleted_at IS NULL
                    ORDER BY bm25(files_fts), files.relative_path
                    LIMIT ?
                    """,
                    (literal_query, root, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row["relative_path"]) for row in rows]

    def mark_deleted(
        self,
        root: str,
        relative_path: str,
        *,
        deleted_at: str | None = None,
    ) -> bool:
        """Tombstone a missing file while retaining its last observed bytes.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.
            deleted_at: Optional UTC deletion timestamp.

        Returns:
            ``True`` when an existing replica row was tombstoned.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE files
                SET deleted_at = ?
                WHERE root = ? AND relative_path = ?
                """,
                (deleted_at or _utc_now(), root, relative_path),
            )
            if cursor.rowcount == 0:
                return False
            self._delete_fts(cursor, root, relative_path)
        return True

    def move_file(
        self,
        root: str,
        source_path: str,
        destination_path: str,
        raw_bytes: bytes,
        *,
        content_hash: str,
        decoded_text: str | None,
        size: int,
        mtime_ns: int,
    ) -> bool:
        """Publish a moved file and discard its active source atomically.

        Args:
            root: Canonical notes-root identifier.
            source_path: Moved-from path relative to ``root``.
            destination_path: Moved-to path relative to ``root``.
            raw_bytes: Exact destination bytes read from disk.
            content_hash: Digest of ``raw_bytes``.
            decoded_text: Searchable text, or ``None`` for non-text content.
            size: Destination file size in bytes.
            mtime_ns: Destination modification time in nanoseconds.

        Returns:
            ``True`` when an active source row was removed. A genuine source
            tombstone is retained.
        """
        with self._transaction() as cursor:
            self._upsert_file(
                cursor,
                root,
                destination_path,
                raw_bytes,
                content_hash=content_hash,
                decoded_text=decoded_text,
                size=size,
                mtime_ns=mtime_ns,
            )
            self._delete_fts(cursor, root, source_path)
            cursor.execute(
                """
                DELETE FROM files
                WHERE root = ?
                  AND relative_path = ?
                  AND deleted_at IS NULL
                """,
                (root, source_path),
            )
            removed = cursor.rowcount > 0
        return removed

    def clear_tombstone(self, root: str, relative_path: str) -> bool:
        """Clear a deletion marker and restore searchable current content.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.

        Returns:
            ``True`` when a tombstone was cleared.
        """
        with self._transaction() as cursor:
            row = cursor.execute(
                """
                SELECT decoded_text
                FROM files
                WHERE root = ?
                  AND relative_path = ?
                  AND deleted_at IS NOT NULL
                """,
                (root, relative_path),
            ).fetchone()
            if row is None:
                return False
            cursor.execute(
                """
                UPDATE files
                SET deleted_at = NULL
                WHERE root = ? AND relative_path = ?
                """,
                (root, relative_path),
            )
            self._replace_fts(
                cursor,
                root,
                relative_path,
                row["decoded_text"],
            )
        return True

    def forget_file(self, root: str, relative_path: str) -> bool:
        """Drop one replica row and its FTS row without leaving a tombstone.

        task-32552: a file the walk no longer visits (it sits under a
        dot-directory) has not been deleted; tombstoning it would list it
        under "Recently deleted" and keep it searchable.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.

        Returns:
            ``True`` when a row was removed.
        """
        with self._transaction() as cursor:
            self._delete_fts(cursor, root, relative_path)
            cursor.execute(
                """
                DELETE FROM files
                WHERE root = ? AND relative_path = ?
                """,
                (root, relative_path),
            )
            removed = cursor.rowcount > 0
        return removed

    def list_deleted(self, root: str) -> list[str]:
        """List tombstoned paths for one canonical root.

        Args:
            root: Canonical notes-root identifier.

        Returns:
            Tombstoned relative paths, newest deletion first.
        """
        with self._locked_connection():
            rows = self._connection.execute(
                """
                SELECT relative_path
                FROM files
                WHERE root = ? AND deleted_at IS NOT NULL
                ORDER BY deleted_at DESC, relative_path
                """,
                (root,),
            ).fetchall()
        return [str(row["relative_path"]) for row in rows]

    def get_restore_bytes(self, root: str, relative_path: str) -> bytes | None:
        """Return exact bytes only when a path has a deletion tombstone.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.

        Returns:
            Restorable bytes, or ``None`` when no tombstone exists.
        """
        with self._locked_connection():
            row = self._connection.execute(
                """
                SELECT raw_bytes
                FROM files
                WHERE root = ?
                  AND relative_path = ?
                  AND deleted_at IS NOT NULL
                """,
                (root, relative_path),
            ).fetchone()
        return None if row is None else bytes(row["raw_bytes"])

    def protect(
        self,
        root: str,
        relative_path: str,
        *,
        is_prefix: bool = False,
    ) -> None:
        """Protect one exact path or a path-component-bounded folder prefix.

        Args:
            root: Canonical notes-root identifier.
            relative_path: Exact file path or folder prefix.
            is_prefix: Whether ``relative_path`` identifies a folder prefix.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR IGNORE INTO protected_paths (
                    root,
                    relative_path,
                    is_prefix
                )
                VALUES (?, ?, ?)
                """,
                (root, relative_path, int(is_prefix)),
            )

    def unprotect(
        self,
        root: str,
        relative_path: str,
        *,
        is_prefix: bool = False,
    ) -> bool:
        """Remove one exact protection entry.

        Args:
            root: Canonical notes-root identifier.
            relative_path: Exact file path or folder prefix.
            is_prefix: Whether ``relative_path`` identifies a folder prefix.

        Returns:
            ``True`` when a protection entry was removed.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                DELETE FROM protected_paths
                WHERE root = ?
                  AND relative_path = ?
                  AND is_prefix = ?
                """,
                (root, relative_path, int(is_prefix)),
            )
            removed = cursor.rowcount > 0
        return removed

    def is_protected(self, root: str, relative_path: str) -> bool:
        """Return whether an exact or component-bounded prefix protects a path.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.

        Returns:
            ``True`` when an exact entry or folder prefix protects the path.
        """
        with self._locked_connection():
            return _protected_row_exists(
                self._connection,
                root,
                relative_path,
            )

    def checkpoint(
        self,
        root: str,
        relative_path: str,
        raw_bytes: bytes,
        *,
        content_hash: str,
        session_key: str,
        created_at: str | None = None,
    ) -> bool:
        """Record exact pre-edit bytes once for a supplied editing session.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.
            raw_bytes: Exact bytes captured before editing.
            content_hash: Digest of ``raw_bytes``.
            session_key: Identifier used to coalesce session checkpoints.
            created_at: Optional UTC checkpoint timestamp.

        Returns:
            ``True`` when a new checkpoint was inserted.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR IGNORE INTO revisions (
                    root,
                    relative_path,
                    raw_bytes,
                    content_hash,
                    kind,
                    session_key,
                    created_at
                )
                VALUES (?, ?, ?, ?, 'pre_edit', ?, ?)
                """,
                (
                    root,
                    relative_path,
                    raw_bytes,
                    content_hash,
                    session_key,
                    created_at or _utc_now(),
                ),
            )
            inserted = cursor.rowcount > 0
        return inserted

    def list_revisions(
        self,
        root: str,
        relative_path: str,
        *,
        limit: int = REVISION_HISTORY_DEFAULT_LIMIT,
    ) -> list[ReplicaRevisionInfo]:
        """List one file's revisions, most recent first, bounded to ``limit``.

        The caller-supplied ``limit`` is clamped to at most
        :data:`REVISION_HISTORY_MAX_LIMIT` before anything is read, rows are
        ranked by parsed UTC instant -- stored spellings mix ``Z``,
        ``+00:00`` and offsets, so SQL text order is not chronological order
        (PR #3016 review findings 3, 5 and 12) -- and revision bytes stay in
        the database: the listing reads only ``length(raw_bytes)``.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.
            limit: Maximum number of entries returned.

        Returns:
            Revision entries (row id, time, kind, session, hash, size),
            newest first.
        """
        if limit <= 0:
            return []
        bounded_limit = min(limit, REVISION_HISTORY_MAX_LIMIT)
        with self._locked_connection():
            rows = self._connection.execute(
                """
                SELECT rowid, kind, session_key, created_at, content_hash,
                       length(raw_bytes) AS byte_size
                FROM revisions
                WHERE root = ? AND relative_path = ?
                """,
                (root, relative_path),
            ).fetchall()
        ranked = sorted(rows, key=_revision_row_rank, reverse=True)
        return [
            ReplicaRevisionInfo(
                kind=str(row["kind"]),
                session_key=(
                    None if row["session_key"] is None else str(row["session_key"])
                ),
                created_at=str(row["created_at"]),
                content_hash=str(row["content_hash"]),
                size=int(row["byte_size"]),
                revision_id=int(row["rowid"]),
            )
            for row in ranked[:bounded_limit]
        ]

    def get_revision(
        self,
        root: str,
        relative_path: str,
        *,
        kind: str,
        session_key: str | None,
        revision_id: int | None = None,
    ) -> ReplicaRevisionBytes | None:
        """Return one revision's exact bytes and recorded digest.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.
            kind: Revision kind (``pre_edit`` or ``delete``).
            session_key: Coalescing session identifier; ``None`` selects the
                deletion revision, whose session key is NULL.
            revision_id: Exact revision row (from :meth:`list_revisions`).
                Without it, a ``delete`` lookup with a NULL session key
                serves the NEWEST deletion -- the row the current
                tombstone's delete wrote. Every delete cycle inserts its
                own NULL-session row (NULLs do not collide under the UNIQUE
                index), so an unordered lookup could otherwise serve an
                older cycle's bytes (PR #3016 review finding 2).

        Returns:
            Stored bytes with their digest, or ``None`` when absent.
        """
        with self._locked_connection():
            if revision_id is not None:
                row = self._connection.execute(
                    """
                    SELECT raw_bytes, content_hash
                    FROM revisions
                    WHERE root = ?
                      AND relative_path = ?
                      AND kind = ?
                      AND session_key IS ?
                      AND rowid = ?
                    """,
                    (root, relative_path, kind, session_key, revision_id),
                ).fetchone()
            elif kind == "delete" and session_key is None:
                delete_rows = self._connection.execute(
                    """
                    SELECT rowid, created_at, raw_bytes, content_hash
                    FROM revisions
                    WHERE root = ?
                      AND relative_path = ?
                      AND kind = 'delete'
                    """,
                    (root, relative_path),
                ).fetchall()
                row = (
                    max(delete_rows, key=_revision_row_rank)
                    if delete_rows
                    else None
                )
            else:
                row = self._connection.execute(
                    """
                    SELECT raw_bytes, content_hash
                    FROM revisions
                    WHERE root = ?
                      AND relative_path = ?
                      AND kind = ?
                      AND session_key IS ?
                    """,
                    (root, relative_path, kind, session_key),
                ).fetchone()
        if row is None:
            return None
        return ReplicaRevisionBytes(
            raw_bytes=bytes(row["raw_bytes"]),
            content_hash=str(row["content_hash"]),
        )

    def verify_revision(
        self,
        root: str,
        relative_path: str,
        *,
        kind: str,
        session_key: str | None,
        revision_id: int | None = None,
    ) -> bool | None:
        """Check one revision's stored bytes against its recorded digest.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.
            kind: Revision kind (``pre_edit`` or ``delete``).
            session_key: Coalescing session identifier; ``None`` selects the
                deletion revision.
            revision_id: Exact revision row (from :meth:`list_revisions`),
                selecting one specific deletion among several.

        Returns:
            ``True`` when the digest of the stored bytes equals the recorded
            digest, ``False`` on mismatch, ``None`` when the revision is
            absent.
        """
        revision = self.get_revision(
            root,
            relative_path,
            kind=kind,
            session_key=session_key,
            revision_id=revision_id,
        )
        if revision is None:
            return None
        return (
            hashlib.sha256(revision.raw_bytes).hexdigest() == revision.content_hash
        )

    def enforce_retention(
        self,
        root: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Apply the ADR-218 retention bounds for one root atomically.

        Three rules, one transaction: each path keeps at most
        :data:`MAX_REVISIONS_PER_NOTE` most-recent ``pre_edit`` revisions;
        tombstones and revisions older than :data:`RECOVERY_EXPIRY_DAYS` are
        expired; protected paths and the most-recent tombstone (with its own
        ``delete`` revision, matched by row identity) are never expired.
        Timestamps are compared as parsed UTC values -- the stored spellings
        mix ``Z``, ``+00:00`` and offsets, so ordering NEVER happens in SQL
        text -- and a value that cannot be parsed is kept, never evicted.

        Args:
            root: Canonical notes-root identifier.
            now: Retention clock; defaults to the current UTC time.
        """
        observed_now = now or datetime.now(timezone.utc)
        if observed_now.tzinfo is None:
            observed_now = observed_now.replace(tzinfo=timezone.utc)
        cutoff = observed_now - timedelta(days=RECOVERY_EXPIRY_DAYS)
        with self._transaction() as cursor:
            # Rule 1: the per-note checkpoint cap. Every path is capped,
            # protected included (PR #3016 review finding 14): protected
            # saves are the only checkpoint writer, so exempting them would
            # leave exactly the growth ADR-218 exists to bound unbounded.
            # Survivors are chosen by parsed UTC instant (finding 1), with
            # insertion order as the deterministic tie-break; unparseable
            # stamps rank newest, so they are always kept.
            cap_paths = cursor.execute(
                """
                SELECT DISTINCT relative_path
                FROM revisions
                WHERE root = ? AND kind = 'pre_edit'
                """,
                (root,),
            ).fetchall()
            for row in cap_paths:
                relative_path = str(row["relative_path"])
                checkpoint_rows = cursor.execute(
                    """
                    SELECT rowid, created_at
                    FROM revisions
                    WHERE root = ? AND relative_path = ? AND kind = 'pre_edit'
                    """,
                    (root, relative_path),
                ).fetchall()
                ranked = sorted(
                    checkpoint_rows,
                    key=_revision_row_rank,
                    reverse=True,
                )
                expired_ids = [
                    int(expired["rowid"])
                    for expired in ranked[MAX_REVISIONS_PER_NOTE:]
                ]
                if expired_ids:
                    cursor.executemany(
                        "DELETE FROM revisions WHERE rowid = ?",
                        [(row_id,) for row_id in expired_ids],
                    )

            # Rule 2: tombstone expiry. Exactly ONE most-recent tombstone is
            # preserved (PR #3016 review finding 6: a tied greatest
            # ``deleted_at`` no longer exempts every peer), chosen by parsed
            # instant with the files rowid as the deterministic tie-break.
            tombstone_rows = cursor.execute(
                """
                SELECT rowid, relative_path, deleted_at
                FROM files
                WHERE root = ? AND deleted_at IS NOT NULL
                """,
                (root,),
            ).fetchall()
            preserved_tombstone_key: tuple[datetime, int] | None = None
            for row in tombstone_rows:
                deleted_at = _parse_utc_timestamp(str(row["deleted_at"]))
                if deleted_at is None:
                    continue
                key = (deleted_at, int(row["rowid"]))
                if preserved_tombstone_key is None or key > preserved_tombstone_key:
                    preserved_tombstone_key = key
            preserved_tombstone_path: str | None = None
            if preserved_tombstone_key is not None:
                for row in tombstone_rows:
                    if int(row["rowid"]) == preserved_tombstone_key[1]:
                        preserved_tombstone_path = str(row["relative_path"])
                        break
            for row in tombstone_rows:
                relative_path = str(row["relative_path"])
                if _protected_row_exists(cursor, root, relative_path):
                    continue
                deleted_at = _parse_utc_timestamp(str(row["deleted_at"]))
                if deleted_at is None or deleted_at >= cutoff:
                    continue
                if (deleted_at, int(row["rowid"])) == preserved_tombstone_key:
                    continue
                self._delete_fts(cursor, root, relative_path)
                cursor.execute(
                    "DELETE FROM files WHERE root = ? AND relative_path = ?",
                    (root, relative_path),
                )

            # Rule 3: revision expiry. Protected paths are exempt, and the
            # preserved tombstone's own delete revision is exempt by ROW
            # IDENTITY -- not timestamp equality -- because the deletion
            # writer accepts independent ``created_at``/``deleted_at``
            # values (PR #3016 review finding 7). The identity is the
            # newest ``delete`` row of the preserved tombstone's path, the
            # same row a NULL-session lookup serves.
            preserved_delete_row_id: int | None = None
            if preserved_tombstone_path is not None:
                delete_rows = cursor.execute(
                    """
                    SELECT rowid, created_at
                    FROM revisions
                    WHERE root = ? AND relative_path = ? AND kind = 'delete'
                    """,
                    (root, preserved_tombstone_path),
                ).fetchall()
                if delete_rows:
                    newest_delete = max(delete_rows, key=_revision_row_rank)
                    preserved_delete_row_id = int(newest_delete["rowid"])
            revision_rows = cursor.execute(
                """
                SELECT rowid, relative_path, kind, created_at
                FROM revisions
                WHERE root = ?
                """,
                (root,),
            ).fetchall()
            expired_row_ids: list[int] = []
            protected_cache: dict[str, bool] = {}
            for row in revision_rows:
                relative_path = str(row["relative_path"])
                is_protected = protected_cache.get(relative_path)
                if is_protected is None:
                    is_protected = _protected_row_exists(
                        cursor, root, relative_path
                    )
                    protected_cache[relative_path] = is_protected
                if is_protected:
                    continue
                if int(row["rowid"]) == preserved_delete_row_id:
                    # The preserved tombstone's own delete revision is the
                    # same recovery fact as the tombstone kept above.
                    continue
                created_at = _parse_utc_timestamp(str(row["created_at"]))
                if created_at is None or created_at >= cutoff:
                    continue
                expired_row_ids.append(int(row["rowid"]))
            for row_id in expired_row_ids:
                cursor.execute(
                    "DELETE FROM revisions WHERE rowid = ?", (row_id,)
                )

    def prepare_deletion(
        self,
        root: str,
        relative_path: str,
        raw_bytes: bytes,
        *,
        content_hash: str,
        decoded_text: str | None,
        deleted_at: str | None = None,
        created_at: str | None = None,
    ) -> None:
        """Atomically store a deletion snapshot and tombstone its current row.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.
            raw_bytes: Exact bytes captured before deletion.
            content_hash: Digest of ``raw_bytes``.
            decoded_text: Searchable text, or ``None`` for non-text content.
            deleted_at: Optional UTC deletion timestamp.
            created_at: Optional UTC revision timestamp.

        Raises:
            KeyError: If the path has no current replica row to tombstone.
        """
        deletion_time = deleted_at or _utc_now()
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO revisions (
                    root,
                    relative_path,
                    raw_bytes,
                    content_hash,
                    kind,
                    session_key,
                    created_at
                )
                VALUES (?, ?, ?, ?, 'delete', NULL, ?)
                """,
                (
                    root,
                    relative_path,
                    raw_bytes,
                    content_hash,
                    created_at or deletion_time,
                ),
            )
            cursor.execute(
                """
                UPDATE files
                SET raw_bytes = ?,
                    content_hash = ?,
                    decoded_text = ?,
                    size = ?,
                    deleted_at = ?
                WHERE root = ? AND relative_path = ?
                """,
                (
                    raw_bytes,
                    content_hash,
                    decoded_text,
                    len(raw_bytes),
                    deletion_time,
                    root,
                    relative_path,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError((root, relative_path))
            self._delete_fts(cursor, root, relative_path)

    def _initialize_schema(self) -> None:
        with self._locked_connection():
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    root TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    raw_bytes BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    decoded_text TEXT,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    deleted_at TEXT,
                    UNIQUE(root, relative_path)
                );

                CREATE TABLE IF NOT EXISTS revisions (
                    root TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    raw_bytes BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    session_key TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(root, relative_path, kind, session_key)
                );

                CREATE TABLE IF NOT EXISTS protected_paths (
                    root TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    is_prefix INTEGER NOT NULL CHECK(is_prefix IN (0, 1)),
                    UNIQUE(root, relative_path, is_prefix)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                    root UNINDEXED,
                    relative_path UNINDEXED,
                    decoded_text,
                    tokenize = 'unicode61'
                );
                """
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._locked_connection():
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    @classmethod
    def _upsert_file(
        cls,
        cursor: sqlite3.Cursor,
        root: str,
        relative_path: str,
        raw_bytes: bytes,
        *,
        content_hash: str,
        decoded_text: str | None,
        size: int,
        mtime_ns: int,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO files (
                root,
                relative_path,
                raw_bytes,
                content_hash,
                decoded_text,
                size,
                mtime_ns,
                deleted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(root, relative_path) DO UPDATE SET
                raw_bytes = excluded.raw_bytes,
                content_hash = excluded.content_hash,
                decoded_text = excluded.decoded_text,
                size = excluded.size,
                mtime_ns = excluded.mtime_ns,
                deleted_at = NULL
            """,
            (
                root,
                relative_path,
                raw_bytes,
                content_hash,
                decoded_text,
                size,
                mtime_ns,
            ),
        )
        cls._replace_fts(cursor, root, relative_path, decoded_text)

    @staticmethod
    def _delete_fts(
        cursor: sqlite3.Cursor,
        root: str,
        relative_path: str,
    ) -> None:
        cursor.execute(
            """
            DELETE FROM files_fts
            WHERE root = ? AND relative_path = ?
            """,
            (root, relative_path),
        )

    @classmethod
    def _replace_fts(
        cls,
        cursor: sqlite3.Cursor,
        root: str,
        relative_path: str,
        decoded_text: str | None,
    ) -> None:
        cls._delete_fts(cursor, root, relative_path)
        if decoded_text is not None:
            cursor.execute(
                """
                INSERT INTO files_fts (root, relative_path, decoded_text)
                VALUES (?, ?, ?)
                """,
                (root, relative_path, decoded_text),
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _protected_row_exists(
    connection: sqlite3.Connection,
    root: str,
    relative_path: str,
) -> bool:
    """Return whether an exact or component-bounded prefix protects a path."""
    row = connection.execute(
        """
        SELECT 1
        FROM protected_paths
        WHERE root = ?
          AND (
                (is_prefix = 0 AND relative_path = ?)
             OR (
                    is_prefix = 1
                AND (
                       relative_path = ''
                    OR relative_path = ?
                    OR substr(?, 1, length(relative_path) + 1)
                       = relative_path || '/'
                )
             )
          )
        LIMIT 1
        """,
        (root, relative_path, relative_path, relative_path),
    ).fetchone()
    return row is not None


def _parse_utc_timestamp(value: str) -> datetime | None:
    """Parse one stored UTC timestamp, or ``None`` when unparseable.

    Stored spellings mix trailing ``Z``, ``+00:00`` and non-UTC offsets
    across the codebase's history; SQL string ordering cannot compare them,
    so retention and history ranking parse in Python. ``None`` fails safe:
    the caller keeps, never evicts.
    """
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


#: Ranks above every parseable instant, so a revision whose stored stamp
#: cannot be parsed is listed and kept rather than silently dropped or
#: evicted (the ADR-218 fail-safe applied to ordering).
_MAX_SORT_INSTANT = datetime.max.replace(tzinfo=timezone.utc)


def _revision_row_rank(row: sqlite3.Row) -> tuple[datetime, int]:
    """Rank one revision row by parsed UTC instant, then insertion order.

    PR #3016 review findings 1 and 12: ``created_at`` text order is not time
    order for the spellings this replica stores, so every survivor/newest
    choice (the checkpoint cap, history listing, newest-deletion lookup)
    ranks rows with this key instead of ``ORDER BY created_at``.
    """
    parsed = _parse_utc_timestamp(str(row["created_at"]))
    return (
        parsed if parsed is not None else _MAX_SORT_INSTANT,
        int(row["rowid"]),
    )
