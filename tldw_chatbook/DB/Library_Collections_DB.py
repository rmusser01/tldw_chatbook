"""SQLite persistence for local Library Collections."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

from .base_db import BaseDB


class LibraryCollectionsSchemaError(RuntimeError):
    """Typed failure for an unavailable or unsupported Collections schema."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class LibraryCollectionsDB(BaseDB):
    """Database wrapper for Library-owned local Collections.

    task-15466: connections are held per thread (the ``Workspace_DB``
    idiom, itself the ChaChaNotes ``_get_thread_connection`` shape). The
    old ``closing()``-per-use form paid a full private-SQLite open --
    which re-verifies the database file and its three ``-wal``/``-shm``/
    ``-journal`` sidecars every time -- for every single query on a screen
    that reads Collections repeatedly; task-3011 measured the same
    anti-pattern at 1,352 connections during one Console screen push.

    Thread safety: this DB is reached both from the UI thread and from
    ``asyncio.to_thread`` worker pools, and Python's sqlite3 refuses a
    connection used from a thread other than its creator
    (``check_same_thread`` defaults to True). Thread-local storage is what
    makes the held connection safe here -- each thread owns exactly one.
    """

    _CURRENT_SCHEMA_VERSION = 3
    _WAL_SETUP_TIMEOUT_SECONDS = 5.0

    _CAPTURE_TABLE_NAMES = frozenset(
        {
            "collection_capture_highlights",
            "collection_capture_item_tags",
            "collection_capture_items",
            "collection_capture_note_links",
            "collection_capture_offline_files",
            "collection_capture_saved_searches",
            "collection_capture_scavenge_state",
            "collection_capture_search",
            "collection_capture_tags",
        }
    )
    _CAPTURE_TRIGGER_NAMES = frozenset(
        {
            "collection_capture_item_tags_search_ad",
            "collection_capture_item_tags_search_ai",
            "collection_capture_items_search_ad",
            "collection_capture_items_search_ai",
            "collection_capture_items_search_au",
            "collection_capture_tags_search_au",
        }
    )
    _CAPTURE_V3_COLUMNS = (
        (
            "extraction_owner_token",
            "ALTER TABLE collection_capture_items "
            "ADD COLUMN extraction_owner_token TEXT",
        ),
        (
            "extraction_lease_expires_at",
            "ALTER TABLE collection_capture_items "
            "ADD COLUMN extraction_lease_expires_at TEXT",
        ),
    )
    _LEGACY_REQUIRED_COLUMNS = {
        "library_collections": frozenset(
            {
                "collection_id",
                "name",
                "description",
                "created_at",
                "updated_at",
                "deleted_at",
            }
        ),
        "library_collection_items": frozenset(
            {
                "membership_id",
                "collection_id",
                "source_type",
                "source_id",
                "title",
                "created_at",
            }
        ),
    }

    _LEGACY_SCHEMA_DDL = (
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS library_collections (
            collection_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS library_collection_items (
            membership_id TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(collection_id)
                REFERENCES library_collections(collection_id)
                ON DELETE CASCADE,
            UNIQUE(collection_id, source_type, source_id)
        )
        """,
    )

    _CAPTURE_SCHEMA_DDL = (
        """
        CREATE TABLE IF NOT EXISTS collection_capture_items (
            authority_key TEXT NOT NULL,
            capture_id TEXT NOT NULL,
            submitted_url TEXT NOT NULL,
            canonical_url TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT '',
            title TEXT,
            summary TEXT,
            freeform_note TEXT,
            text_content TEXT,
            clean_html TEXT,
            byline TEXT,
            published_at TEXT,
            read_at TEXT,
            content_hash TEXT,
            word_count INTEGER,
            status TEXT NOT NULL CHECK(status IN ('saved', 'reading', 'read', 'archived')),
            favorite INTEGER NOT NULL CHECK(favorite IN (0, 1)),
            processing_state TEXT NOT NULL
                CHECK(processing_state IN ('queued', 'processing', 'ready', 'failed', 'interrupted')),
            last_fetch_error TEXT,
            extraction_owner_token TEXT,
            extraction_lease_expires_at TEXT,
            media_authority_key TEXT,
            media_item_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
            purge_state TEXT,
            PRIMARY KEY(authority_key, capture_id),
            UNIQUE(authority_key, canonical_url)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS collection_capture_tags (
            authority_key TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            normalized_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            PRIMARY KEY(authority_key, tag_id),
            UNIQUE(authority_key, normalized_name)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS collection_capture_item_tags (
            authority_key TEXT NOT NULL,
            capture_id TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY(authority_key, capture_id, tag_id),
            FOREIGN KEY(authority_key, capture_id)
                REFERENCES collection_capture_items(authority_key, capture_id)
                ON DELETE CASCADE,
            FOREIGN KEY(authority_key, tag_id)
                REFERENCES collection_capture_tags(authority_key, tag_id)
                ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS collection_capture_highlights (
            authority_key TEXT NOT NULL,
            highlight_id TEXT NOT NULL,
            capture_id TEXT NOT NULL,
            quote TEXT NOT NULL,
            note TEXT,
            anchor_json TEXT,
            detached INTEGER NOT NULL DEFAULT 0 CHECK(detached IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
            PRIMARY KEY(authority_key, highlight_id),
            FOREIGN KEY(authority_key, capture_id)
                REFERENCES collection_capture_items(authority_key, capture_id)
                ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS collection_capture_saved_searches (
            authority_key TEXT NOT NULL,
            search_id TEXT NOT NULL,
            name TEXT NOT NULL,
            query_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
            PRIMARY KEY(authority_key, search_id),
            UNIQUE(authority_key, name)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS collection_capture_note_links (
            authority_key TEXT NOT NULL,
            link_id TEXT NOT NULL,
            capture_id TEXT NOT NULL,
            note_authority_key TEXT NOT NULL,
            note_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(authority_key, link_id),
            UNIQUE(authority_key, capture_id, note_authority_key, note_id),
            FOREIGN KEY(authority_key, capture_id)
                REFERENCES collection_capture_items(authority_key, capture_id)
                ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS collection_capture_offline_files (
            authority_key TEXT NOT NULL,
            file_id TEXT NOT NULL,
            capture_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            content_hash TEXT,
            reserved_size INTEGER NOT NULL CHECK(reserved_size >= 0),
            actual_size INTEGER CHECK(actual_size IS NULL OR actual_size >= 0),
            media_type TEXT,
            state TEXT NOT NULL CHECK(state IN ('staging', 'ready', 'failed', 'purging')),
            failure_reason TEXT,
            temporary_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision > 0),
            PRIMARY KEY(authority_key, file_id),
            FOREIGN KEY(authority_key, capture_id)
                REFERENCES collection_capture_items(authority_key, capture_id)
                ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS collection_capture_scavenge_state (
            authority_key TEXT PRIMARY KEY,
            authority_fingerprint TEXT,
            cursor_kind TEXT,
            cursor_value TEXT,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS collection_capture_search USING fts5(
            authority_key UNINDEXED,
            capture_id UNINDEXED,
            title,
            summary,
            freeform_note,
            text_content,
            tag_text
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_items_updated_page
        ON collection_capture_items(authority_key, updated_at DESC, capture_id DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_items_created_page
        ON collection_capture_items(authority_key, created_at DESC, capture_id DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_items_published_page
        ON collection_capture_items(authority_key, published_at DESC, capture_id DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_items_title_page
        ON collection_capture_items(authority_key, title COLLATE NOCASE, capture_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_items_status_page
        ON collection_capture_items(authority_key, status, updated_at DESC, capture_id DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_items_favorite_page
        ON collection_capture_items(authority_key, favorite, updated_at DESC, capture_id DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_items_domain_page
        ON collection_capture_items(authority_key, domain, updated_at DESC, capture_id DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_item_tags_by_tag
        ON collection_capture_item_tags(authority_key, tag_id, capture_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_highlights_by_item
        ON collection_capture_highlights(authority_key, capture_id, created_at, highlight_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_note_links_by_item
        ON collection_capture_note_links(authority_key, capture_id, created_at, link_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_offline_by_item
        ON collection_capture_offline_files(authority_key, capture_id, updated_at, file_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_collection_capture_offline_by_state
        ON collection_capture_offline_files(authority_key, state, updated_at, file_id)
        """,
        """
        CREATE TRIGGER IF NOT EXISTS collection_capture_items_search_ai
        AFTER INSERT ON collection_capture_items
        BEGIN
            INSERT INTO collection_capture_search(
                rowid, authority_key, capture_id, title, summary,
                freeform_note, text_content, tag_text
            ) VALUES (
                new.rowid, new.authority_key, new.capture_id,
                COALESCE(new.title, ''), COALESCE(new.summary, ''),
                COALESCE(new.freeform_note, ''), COALESCE(new.text_content, ''), ''
            );
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS collection_capture_items_search_au
        AFTER UPDATE OF authority_key, capture_id, title, summary, freeform_note, text_content
        ON collection_capture_items
        BEGIN
            DELETE FROM collection_capture_search WHERE rowid = old.rowid;
            INSERT INTO collection_capture_search(
                rowid, authority_key, capture_id, title, summary,
                freeform_note, text_content, tag_text
            ) VALUES (
                new.rowid, new.authority_key, new.capture_id,
                COALESCE(new.title, ''), COALESCE(new.summary, ''),
                COALESCE(new.freeform_note, ''), COALESCE(new.text_content, ''),
                COALESCE((
                    SELECT group_concat(tag.display_name, ' ')
                    FROM collection_capture_item_tags AS item_tag
                    JOIN collection_capture_tags AS tag
                      ON tag.authority_key = item_tag.authority_key
                     AND tag.tag_id = item_tag.tag_id
                    WHERE item_tag.authority_key = new.authority_key
                      AND item_tag.capture_id = new.capture_id
                ), '')
            );
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS collection_capture_items_search_ad
        AFTER DELETE ON collection_capture_items
        BEGIN
            DELETE FROM collection_capture_search WHERE rowid = old.rowid;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS collection_capture_item_tags_search_ai
        AFTER INSERT ON collection_capture_item_tags
        BEGIN
            UPDATE collection_capture_search
            SET tag_text = COALESCE((
                SELECT group_concat(tag.display_name, ' ')
                FROM collection_capture_item_tags AS item_tag
                JOIN collection_capture_tags AS tag
                  ON tag.authority_key = item_tag.authority_key
                 AND tag.tag_id = item_tag.tag_id
                WHERE item_tag.authority_key = new.authority_key
                  AND item_tag.capture_id = new.capture_id
            ), '')
            WHERE authority_key = new.authority_key
              AND capture_id = new.capture_id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS collection_capture_item_tags_search_ad
        AFTER DELETE ON collection_capture_item_tags
        BEGIN
            UPDATE collection_capture_search
            SET tag_text = COALESCE((
                SELECT group_concat(tag.display_name, ' ')
                FROM collection_capture_item_tags AS item_tag
                JOIN collection_capture_tags AS tag
                  ON tag.authority_key = item_tag.authority_key
                 AND tag.tag_id = item_tag.tag_id
                WHERE item_tag.authority_key = old.authority_key
                  AND item_tag.capture_id = old.capture_id
            ), '')
            WHERE authority_key = old.authority_key
              AND capture_id = old.capture_id;
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS collection_capture_tags_search_au
        AFTER UPDATE OF display_name ON collection_capture_tags
        BEGIN
            UPDATE collection_capture_search
            SET tag_text = COALESCE((
                SELECT group_concat(tag.display_name, ' ')
                FROM collection_capture_item_tags AS item_tag
                JOIN collection_capture_tags AS tag
                  ON tag.authority_key = item_tag.authority_key
                 AND tag.tag_id = item_tag.tag_id
                WHERE item_tag.authority_key = collection_capture_search.authority_key
                  AND item_tag.capture_id = collection_capture_search.capture_id
            ), '')
            WHERE authority_key = new.authority_key
              AND capture_id IN (
                  SELECT capture_id
                  FROM collection_capture_item_tags
                  WHERE authority_key = new.authority_key
                    AND tag_id = new.tag_id
              );
        END
        """,
    )

    #: Liveness-ping gate (mirrors `Workspace_DB`/`ChaChaNotes_DB`,
    #: task-261/3011): pinging on every call roughly doubles the statement
    #: count on query-heavy paths. A recently-used held connection is
    #: known-good without a ping.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        # Must precede super().__init__: BaseDB.__init__ calls
        # _initialize_schema(), which already needs the held connection.
        self._thread_local = threading.local()
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory_db:
            self._enable_wal(conn)
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local collections cache)
        # and avoids an fsync per commit. Unlike journal_mode, which is
        # persisted in the file, synchronous is per-connection -- so it must
        # be re-applied here on every NEW connection, which is exactly why
        # this pairing lives in the one place connections are created
        # (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
        # task-3012: a held (long-lived) connection needs true autocommit.
        # Python's default isolation mode auto-BEGINs on any DML, and an
        # implicit transaction accumulated outside `transaction()` makes the
        # explicit BEGIN there fail with "cannot start a transaction within a
        # transaction" -- and silently ROLLS BACK bare DML on close.
        # Audited (task-15466): every write in this file's own module and in
        # `Library/library_collections_service.py` goes through
        # `transaction()`; schema initialization owns its explicit migration
        # transaction.
        conn.isolation_level = None
        return conn

    def _enable_wal(self, conn: sqlite3.Connection) -> None:
        """Enable WAL despite a concurrent opener briefly holding the file."""
        deadline = time.monotonic() + self._WAL_SETUP_TIMEOUT_SECONDS
        while True:
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held connection, opening or reviving it.

        The liveness probe is a plain no-op statement; a connection another
        component closed (or that SQLite invalidated) is transparently
        replaced, mirroring `Workspace_DB._held_connection`.
        """
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            last_used = getattr(self._thread_local, "conn_last_used", None)
            if (
                last_used is None
                or (time.monotonic() - last_used)
                >= self._LIVENESS_PING_IDLE_SECONDS
            ):
                try:
                    conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001 - already unusable
                        pass
                    conn = None
        if conn is None:
            conn = self._get_connection()
            self._thread_local.conn = conn
        self._thread_local.conn_last_used = time.monotonic()
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the thread's held connection (row factory, foreign keys on).

        No transaction is opened: in autocommit mode each statement is its
        own implicit transaction, so single-statement reads cost nothing
        extra and never pin a WAL read snapshot between calls.
        """
        yield self._held_connection()

    @contextmanager
    def read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside a read-only snapshot.

        ``BEGIN DEFERRED`` takes no lock until a statement needs one, so a
        block that only reads never acquires the write lock -- while still
        seeing one consistent snapshot across several SELECTs (a caller
        pairing a COUNT with its page needs exactly that). The block always
        ends with ROLLBACK, so a write placed in here would be DISCARDED;
        rather than lose it silently, the block raises when it sees one
        (see Raises). Rolling back an unwritten transaction is free.

        Nesting: this and ``transaction`` both issue an explicit BEGIN on
        the ONE connection this thread holds, so nesting either inside the
        other raises ``sqlite3.OperationalError: cannot start a transaction
        within a transaction``. Pre-port each block had its own connection
        and nesting silently "worked"; the outer block still rolls back
        cleanly, because the failure propagates through its ``except``.

        Yields:
            The thread's held ``sqlite3.Connection`` in a read snapshot.

        Raises:
            RuntimeError: If the block modified any row. The transaction is
                rolled back first, so the write is undone either way -- the
                error exists so the misuse cannot pass unnoticed.
        """
        conn = self._held_connection()
        # total_changes counts rows inserted/updated/deleted on this
        # connection since it was opened, so a difference across the block
        # means a write happened inside a snapshot that is about to be
        # thrown away. It cannot see a DML statement that matched zero rows
        # (nothing to lose in that case) or bare DDL, so this is a guard
        # against silent DATA LOSS, not a general read-only enforcement.
        changes_before = conn.total_changes
        conn.execute("BEGIN DEFERRED")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        wrote = conn.total_changes != changes_before
        conn.rollback()
        if wrote:
            raise RuntimeError(
                "read_transaction() is read-only: a write inside it is "
                "always rolled back. Use transaction() for writes."
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside a write transaction.

        ``BEGIN IMMEDIATE`` takes the write lock up front. The MCP server
        opens this same database file in its own process, so a deferred
        transaction that read before writing could fail to upgrade; taking
        the lock immediately removes that class of failure. Pure readers
        must use ``read_transaction`` (or ``connection``) instead.

        Nesting: this and ``read_transaction`` both issue an explicit BEGIN
        on the ONE connection this thread holds, so nesting either inside
        the other raises ``sqlite3.OperationalError: cannot start a
        transaction within a transaction``. Pre-port each block had its own
        connection and nesting silently "worked"; the outer block still
        rolls back cleanly, because the failure propagates through its
        ``except``.

        Raises:
            Exception: Re-raised after rolling back, on any error inside
                the ``with`` block. On clean exit the transaction commits.
        """
        conn = self._held_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        """Close the current thread's held connection, if any."""
        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _initialize_schema(self) -> None:
        """Atomically initialize or migrate the local Collections schema."""
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                has_version_table = (
                    conn.execute(
                        "SELECT 1 FROM sqlite_schema "
                        "WHERE type = 'table' AND name = 'schema_version'"
                    ).fetchone()
                    is not None
                )
                current_version = 0
                if has_version_table:
                    row = conn.execute(
                        "SELECT MAX(version) FROM schema_version"
                    ).fetchone()
                    current_version = int(row[0] or 0) if row is not None else 0
                if current_version > self._CURRENT_SCHEMA_VERSION:
                    raise LibraryCollectionsSchemaError("schema_too_new")

                if current_version == 0:
                    for statement in self._LEGACY_SCHEMA_DDL:
                        conn.execute(statement)
                    conn.execute(
                        "INSERT OR IGNORE INTO schema_version (version) VALUES (1)"
                    )

                for statement in self._CAPTURE_SCHEMA_DDL:
                    conn.execute(statement)
                if current_version == 2:
                    columns = {
                        str(row[1])
                        for row in conn.execute(
                            "PRAGMA table_info(collection_capture_items)"
                        )
                    }
                    for column_name, statement in self._CAPTURE_V3_COLUMNS:
                        if column_name not in columns:
                            conn.execute(statement)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                    (self._CURRENT_SCHEMA_VERSION,),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def get_schema_version(self) -> int:
        """Return the initialized schema version."""
        with self.connection() as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0] or 0) if row is not None else 0

    def require_capture_schema(self) -> None:
        """Raise when the authority-scoped capture schema is unavailable."""
        version = self.get_schema_version()
        if version > self._CURRENT_SCHEMA_VERSION:
            raise LibraryCollectionsSchemaError("schema_too_new")
        with self.connection() as conn:
            objects = {
                (str(row[0]), str(row[1]))
                for row in conn.execute(
                    "SELECT type, name FROM sqlite_schema "
                    "WHERE type IN ('table', 'view', 'trigger')"
                )
            }
            capture_item_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(collection_capture_items)"
                )
            }
        tables = {name for kind, name in objects if kind in {"table", "view"}}
        triggers = {name for kind, name in objects if kind == "trigger"}
        if (
            version != self._CURRENT_SCHEMA_VERSION
            or not self._CAPTURE_TABLE_NAMES <= tables
            or not self._CAPTURE_TRIGGER_NAMES <= triggers
            or not {
                "extraction_owner_token",
                "extraction_lease_expires_at",
            }
            <= capture_item_columns
        ):
            raise LibraryCollectionsSchemaError("capture_schema_unavailable")

    def has_compatible_legacy_schema(self) -> bool:
        """Return whether bounded v1 inspection can read the legacy tables."""
        with self.connection() as conn:
            for table, required_columns in self._LEGACY_REQUIRED_COLUMNS.items():
                columns = {
                    str(row[1])
                    for row in conn.execute(f"PRAGMA table_info({table})")
                }
                if not required_columns <= columns:
                    return False
        return True
