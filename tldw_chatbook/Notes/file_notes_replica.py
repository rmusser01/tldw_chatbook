"""SQLite replica, search index, and recovery snapshots for File Notes."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import NamedTuple

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


class ReplicaFileInfo(NamedTuple):
    """Metadata needed to reconcile one active disk file with its replica."""

    relative_path: str
    content_hash: str
    size: int
    mtime_ns: int


class FileNotesReplica:
    """Store current File Notes bytes without becoming their editor authority."""

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        """Open the replica and initialize its fixed schema.

        Args:
            db_path: SQLite database path, or ``":memory:"`` for a transient
                replica.
        """
        path = os.fspath(db_path)
        if path != ":memory:":
            path = os.fspath(Path(path).expanduser())
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self._lock:
            self._connection = connect_private_sqlite(
                "notes.file_notes_replica",
                path,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def close(self) -> None:
        """Close the persistent SQLite connection."""
        with self._lock:
            self._connection.close()

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
            self._replace_fts(cursor, root, relative_path, decoded_text)

    def get_bytes(self, root: str, relative_path: str) -> bytes | None:
        """Return exact current or tombstoned bytes for a path.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.

        Returns:
            Stored bytes, or ``None`` when the path is not replicated.
        """
        with self._lock:
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
        with self._lock:
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
            query: Literal text to find in replicated file content.
            limit: Maximum number of paths to return.

        Returns:
            Matching relative paths ordered by relevance.
        """
        query = query.strip()
        if not query or limit <= 0 or "\x00" in query:
            return []
        escaped_query = query.replace('"', '""')
        literal_query = f'"{escaped_query}"'
        try:
            with self._lock:
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

    def discard_file(self, root: str, relative_path: str) -> bool:
        """Remove one active replica projection without creating a tombstone.

        Args:
            root: Canonical notes-root identifier.
            relative_path: File path relative to ``root``.

        Returns:
            ``True`` when an active replica row was removed.
        """
        with self._transaction() as cursor:
            self._delete_fts(cursor, root, relative_path)
            cursor.execute(
                """
                DELETE FROM files
                WHERE root = ?
                  AND relative_path = ?
                  AND deleted_at IS NULL
                """,
                (root, relative_path),
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

    def list_deleted(self, root: str) -> list[str]:
        """List tombstoned paths for one canonical root.

        Args:
            root: Canonical notes-root identifier.

        Returns:
            Tombstoned relative paths, newest deletion first.
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            row = self._connection.execute(
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
        with self._lock:
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
        with self._lock:
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
