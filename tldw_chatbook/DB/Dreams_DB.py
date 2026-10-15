"""SQLite persistence for the Dreams daily discovery subsystem."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Iterator, Union

from tldw_chatbook.Backup_Recovery.participants import (
    _core_access,
    _core_cached_connection,
    _core_closing,
    _core_getter,
    _core_transaction,
    _register_core_connection,
)

from ..Utils.timestamps import utc_now_iso
from .base_db import BaseDB

#: Upper bound for any list read the UI pages through.
MAX_LIST_LIMIT = 200

#: Upper bound for a ``list_tracked_items`` status read (Qodo #9, PR #2890).
#: The active set is already capped by ``tracked_item_cap`` (default 20),
#: but ``retired`` grows forever -- every quiet sweep and passed event
#: lands there -- so an unbounded read would grow with the user's history.
#: 500 retired watches is years of use; callers needing more get paging.
#: A question about ONE row set (does any wrapper own this subscription?)
#: must not be answered by scanning these capped lists -- see
#: ``subscription_dream_owned``.
TRACKED_ITEMS_LIST_LIMIT = 500


def clamp_limit(limit: int) -> int:
    """Bound a caller-supplied page size to ``1..MAX_LIST_LIMIT``.

    SQLite reads a negative ``LIMIT`` as "no limit", so an unchecked value
    would turn a page read into a whole-table read.
    """
    return max(1, min(int(limit), MAX_LIST_LIMIT))


def _utc_now_iso() -> str:
    """Return the current UTC time in the canonical stored shape (ADR-173)."""
    return utc_now_iso()


class DreamsSchemaError(RuntimeError):
    """Typed failure for an unavailable or unsupported Dreams schema."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class DreamsDB(BaseDB):
    """Database wrapper for the Dreams subsystem (discovery + track loops).

    Schema v1 covered the Phase 1 discovery tables (spec §Data model); v2
    adds the Phase 2 track tables (``dream_tracked_items`` /
    ``dream_track_runs``) additively, so v1 files upgrade in place on open;
    v3 widens the interest profile's ``source`` CHECK with ``'guardian'``
    (ADR-204 contract 3: the Guardian aggregate feed writes profile rows
    under that origin). CHECK constraints are baked into a table's DDL, so
    the v2 -> v3 step REBUILDS ``dream_interest_profile`` (create-new,
    copy, drop, rename) inside the same opening transaction -- additive in
    data terms, never destructive. v4 adds one nullable column,
    ``dream_stories.goal``: the goal whose own search query surfaced the
    story (its provenance), kept apart from the topics the story merely
    matched; a v3 file gains it through ``ALTER TABLE ... ADD COLUMN`` on
    open. v5 adds, the same way, ``dream_tracked_items.suspension``: the
    record of what the master switch stopped for a wrapper while Dreams
    is off (NULL when it is not suspended), so switching back on restores
    exactly that and nothing the user stopped themselves. v6 adds three
    more columns there, the same way (``_migrate_tracked_provenance_v6``):
    ``egress_marked``, ``alert_rule_id`` and ``reminder_event_date``.
    Connections are held per thread
    via the ``Library_Collections_DB`` idiom: Python's sqlite3 refuses a
    connection used from a thread other than its creator, and Dreams is
    reached both from the UI thread and from ``asyncio.to_thread`` cycle
    workers, so each thread owns exactly one long-lived connection. Writes
    go through ``transaction()`` (``BEGIN IMMEDIATE``); single-statement
    reads use ``connection()``.

    Date-bucket idempotency: one ``dreams_collections`` row per local date
    (UNIQUE index) is the race guard between scheduler fire, boot catch-up,
    and manual trigger. A row wedged in ``generating`` by a crashed cycle is
    reclaimed by :meth:`fail_stale_generating` before the next date claim.
    """

    _CURRENT_SCHEMA_VERSION = 6
    _WAL_SETUP_TIMEOUT_SECONDS = 5.0
    #: Pinging a recently-used held connection on every call roughly doubles
    #: the statement count on query-heavy paths (task-261/3011); idle ones
    #: are probed so another component's close is transparently healed.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    _SCHEMA_DDL = (
        """
        CREATE TABLE IF NOT EXISTS dream_interest_profile (
            id INTEGER PRIMARY KEY,
            facet TEXT NOT NULL CHECK(facet IN ('topic', 'goal')),
            text TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            searchable INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL
                CHECK(source IN ('user', 'seed', 'personal_context', 'notes',
                                 'media', 'guardian')),
            query_angle TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_boosted_at TEXT,
            UNIQUE(facet, text)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dreams_collections (
            id INTEGER PRIMARY KEY,
            local_date TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ('generating', 'complete', 'partial', 'failed')),
            profile_digest TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            story_count INTEGER NOT NULL DEFAULT 0,
            trigger TEXT NOT NULL
                CHECK(trigger IN ('scheduled', 'catchup', 'manual', 'refresh')),
            degradation_notes TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE(local_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dream_stories (
            id INTEGER PRIMARY KEY,
            collection_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            snippet TEXT NOT NULL,
            body TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('complete', 'empty', 'failed')),
            source TEXT NOT NULL CHECK(source IN ('web', 'watchlist', 'llm')),
            kind TEXT NOT NULL
                CHECK(kind IN ('content', 'event', 'deal', 'social_opportunity',
                               'unknown')),
            event_date TEXT,
            location TEXT,
            matched_topics TEXT NOT NULL,
            query TEXT NOT NULL,
            kept INTEGER NOT NULL DEFAULT 0,
            kept_at TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            goal TEXT,
            FOREIGN KEY(collection_id) REFERENCES dreams_collections(id)
                ON DELETE CASCADE,
            UNIQUE(collection_id, url)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dream_feedback (
            id INTEGER PRIMARY KEY,
            story_id INTEGER NOT NULL,
            kind TEXT NOT NULL
                CHECK(kind IN ('more', 'less', 'kept', 'dived', 'exported',
                               'ingested', 'tracked')),
            created_at TEXT NOT NULL,
            FOREIGN KEY(story_id) REFERENCES dream_stories(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dream_seen_items (
            url TEXT PRIMARY KEY,
            title_digest TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dream_daily_usage (
            local_date TEXT PRIMARY KEY,
            searches INTEGER NOT NULL DEFAULT 0,
            llm_calls INTEGER NOT NULL DEFAULT 0
        )
        """,
        # ----- Schema v2 (Phase 2 track loop), additive on top of v1 -----
        """
        CREATE TABLE IF NOT EXISTS dream_tracked_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_story_id INTEGER,
            mechanism TEXT NOT NULL CHECK(mechanism IN ('page','question')),
            intent TEXT NOT NULL CHECK(intent IN ('event','deal','topic')),
            subscription_id INTEGER,
            query_template TEXT,
            event_date TEXT,
            cadence_seconds INTEGER NOT NULL,
            quiet_retire_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','paused','retired')),
            retired_reason TEXT,
            created_by_dreams INTEGER NOT NULL DEFAULT 0,
            last_checked TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            suspension TEXT,
            egress_marked INTEGER NOT NULL DEFAULT 0,
            alert_rule_id INTEGER,
            reminder_event_date TEXT
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_dream_tracked_status
            ON dream_tracked_items(status)
        """,
        # Task 6 ledger item: Task 5 made per-story reads land
        # (``find_tracked_by_story`` and the story rows' tracked badge),
        # so the origin-story lookup gets its pinned index -- additive
        # DDL, no version bump and no data migration.
        """
        CREATE INDEX IF NOT EXISTS idx_dream_tracked_origin_story
            ON dream_tracked_items(origin_story_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS dream_track_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracked_item_id INTEGER NOT NULL REFERENCES dream_tracked_items(id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK(status IN ('changed','unchanged','baseline','rebaselined','withheld','error','skipped')),
            digest_hash TEXT,
            verdict_note TEXT NOT NULL DEFAULT '',
            notified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_dream_track_runs_item
            ON dream_track_runs(tracked_item_id, created_at DESC)
        """,
        # No index on dream_feedback(story_id): no Phase 1 query looks
        # feedback up by story (the feedback loop scans by created_at, and
        # nothing deletes stories, so the FK cascade never runs). Add one
        # with a plan pin when a per-story read or story deletion lands.
    )

    def __init__(
        self, database_path: Union[str, Path], client_id: str | None = None
    ) -> None:
        # Connections are per THREAD, and every ":memory:" connection is its
        # own empty database: cycle stages running under asyncio.to_thread
        # would see no schema. Refuse it rather than fail mid-cycle.
        if str(database_path) == ":memory:":
            raise ValueError("DreamsDB needs a file path; ':memory:' is not "
                             "shared across its per-thread connections")
        # Must precede super().__init__: BaseDB.__init__ calls
        # _initialize_schema(), which already needs the held connection.
        self._thread_local = threading.local()
        # Every held connection this object opened and has not closed,
        # whatever thread holds it -- what ``close_all`` closes. Strong
        # references on purpose: a connection has to be closed explicitly
        # to give back all of its descriptors (letting one be collected
        # with its thread leaves the private-path layer's descriptor
        # open), and ``sqlite3.Connection`` cannot be weakly referenced.
        self._opened: list[sqlite3.Connection] = []
        self._opened_lock = threading.Lock()
        try:
            super().__init__(
                database_path, client_id if client_id is not None else "default"
            )
        except BaseException:
            self.close_all()
            raise

    # ------------------------------------------------------------------
    # Connection handling (Library_Collections_DB idiom)
    # ------------------------------------------------------------------

    @_core_getter
    def _get_connection(self) -> sqlite3.Connection:
        _core_access(self)
        # ``check_same_thread=False`` (the other thread-local stores here --
        # ChaChaNotes, Prompts, Media -- open theirs the same way): each
        # thread still USES only the connection it holds; the one call
        # that crosses threads is ``close_all`` closing them.
        conn = super()._get_connection(check_same_thread=False)
        _register_core_connection(self, conn)
        try:
            _core_access(self)
            # Refuse unknown stores before WAL changes their header/journal state.
            self._validate_schema_lineage(conn)
            conn.execute("PRAGMA foreign_keys = ON")
            if not self.is_memory_db:
                self._enable_wal(conn)
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.isolation_level = None
        except BaseException:
            conn.close()
            raise
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

    @_core_getter
    def _held_connection(self) -> sqlite3.Connection:
        _core_access(self)
        """Return this thread's held connection, opening or reviving it."""
        conn = _core_cached_connection(self, getattr(self._thread_local, "conn", None))
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
                        sqlite3.Connection.in_transaction.__get__(conn)
                    except sqlite3.ProgrammingError:
                        self._close_connection(conn)
                    else:
                        raise
                    conn = None
        if conn is None:
            conn = self._get_connection()
            self._thread_local.conn = conn
            with self._opened_lock:
                self._opened.append(conn)
        self._thread_local.conn_last_used = time.monotonic()
        return conn

    @_core_transaction
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the thread's held connection (row factory, foreign keys on).

        No transaction is opened: in autocommit mode each statement is its
        own implicit transaction, so single-statement reads cost nothing
        extra and never pin a WAL read snapshot between calls.
        """
        yield self._held_connection()

    @_core_transaction
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside a write transaction.

        ``BEGIN IMMEDIATE`` takes the write lock up front so a read-then-
        write block cannot fail to upgrade under a concurrent writer. On any
        error or interruption inside the ``with`` block the transaction is
        rolled back and the exception re-raised; on clean exit it commits.

        The yielded object is the connection: call ``conn.execute(...)`` and
        use the returned cursor's ``lastrowid``/``rowcount``.
        """
        conn = self._held_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        """Retire only the calling thread's handle when its owner permits."""
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            with _core_closing(self, conn) as allowed:
                if allowed and self._close_connection(conn):
                    self._thread_local.conn = None

    def _close_connection(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - retain uncertain native retirement
            return False
        with self._opened_lock:
            if conn in self._opened:
                self._opened.remove(conn)
        return True

    def close_all(self) -> None:
        """Close EVERY connection this object holds, on whatever thread.

        ``close`` reaches only the calling thread's connection, and each
        ``asyncio.to_thread`` hop may have opened another on a long-lived
        executor thread. A caller that owns the handle outright and is
        done with it -- the startup reconcile's temporary handle for a
        disabled Dreams (review of #2921, round 3, T7), a test fixture --
        calls this so nothing stays open on the file. It must not be
        called while another thread is still working on the handle. The
        object stays usable: a thread that touches it again opens a fresh
        connection.
        """
        with self._opened_lock:
            opened = tuple(self._opened)
        for conn in opened:
            self._close_connection(conn)
        # Existing explicit all-thread owner boundary; public close remains
        # caller-only. Failed native closes stay strongly owned for retry.
        self._thread_local = threading.local()

    @staticmethod
    def _validate_schema_lineage(conn: sqlite3.Connection) -> int:
        """Recognize complete historical catalogs before any schema write."""
        from .dreams_schema import DREAMS_SCHEMAS

        catalog = tuple(" ".join(row[0].split()) for row in conn.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"))
        if not catalog:
            return 0
        try:
            versions = [row[0] for row in conn.execute("SELECT version FROM schema_version")]
        except sqlite3.Error as exc:
            raise DreamsSchemaError("schema_version_missing") from exc
        if not versions or any(type(version) is not int or version < 1 for version in versions):
            raise DreamsSchemaError("schema_version_invalid")
        version = max(versions)
        if version > DreamsDB._CURRENT_SCHEMA_VERSION:
            raise DreamsSchemaError("schema_too_new")
        if not any(version == known and catalog == tuple(" ".join(sql.split()) for sql in schema)
                   for known, schema in DREAMS_SCHEMAS):
            raise DreamsSchemaError("schema_lineage_mismatch")
        return version

    def _initialize_schema(self) -> None:
        """Validate, then atomically initialize or upgrade a known lineage."""
        with self.transaction() as conn:
            current_version = self._validate_schema_lineage(conn)
            if current_version == self._CURRENT_SCHEMA_VERSION:
                return
            if not current_version:
                conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL)")
            for statement in self._SCHEMA_DDL:
                conn.execute(statement)
            self._migrate_profile_source_check_v3(conn)
            self._migrate_story_goal_v4(conn)
            self._migrate_tracked_suspension_v5(conn)
            self._migrate_tracked_provenance_v6(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self._CURRENT_SCHEMA_VERSION,),
            )
            self._validate_schema_lineage(conn)

    @staticmethod
    def _migrate_profile_source_check_v3(conn: sqlite3.Connection) -> None:
        """v2 -> v3: widen the profile ``source`` CHECK with ``'guardian'``.

        A CHECK constraint cannot be altered, so the migration rebuilds the
        table (create with the current DDL shape under a temp name, copy
        every row, drop, rename). It runs ONLY when the on-disk table's DDL
        still carries the old CHECK -- the detection reads
        ``sqlite_master.sql`` for the ``'guardian'`` token, so a fresh v3
        build (whose DDL already has the widened CHECK) and a re-open of an
        already-migrated file both skip the rebuild entirely. Old-source
        rows are a subset of the new CHECK, so the copy never loses data.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master"
            " WHERE type = 'table' AND name = 'dream_interest_profile'"
        ).fetchone()
        if row is None or "'guardian'" in str(row[0] or ""):
            return
        conn.execute("DROP TABLE IF EXISTS dream_interest_profile_v3")
        conn.execute(
            """
            CREATE TABLE dream_interest_profile_v3 (
                id INTEGER PRIMARY KEY,
                facet TEXT NOT NULL CHECK(facet IN ('topic', 'goal')),
                text TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                searchable INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL
                    CHECK(source IN ('user', 'seed', 'personal_context', 'notes',
                                     'media', 'guardian')),
                query_angle TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_boosted_at TEXT,
                UNIQUE(facet, text)
            )
            """
        )
        conn.execute(
            "INSERT INTO dream_interest_profile_v3"
            " SELECT * FROM dream_interest_profile"
        )
        conn.execute("DROP TABLE dream_interest_profile")
        conn.execute(
            "ALTER TABLE dream_interest_profile_v3"
            " RENAME TO dream_interest_profile"
        )

    @staticmethod
    def _migrate_story_goal_v4(conn: sqlite3.Connection) -> None:
        """v3 -> v4: add the nullable ``dream_stories.goal`` column.

        Additive and idempotent: it runs only when the on-disk table lacks
        the column, so a fresh v4 build (whose DDL already declares it, as
        the LAST column -- where ``ADD COLUMN`` also puts it, so both
        shapes order their columns alike) and a re-open of a migrated file
        both skip it. Existing stories keep ``goal`` NULL: no goal's query
        is known to have surfaced them.
        """
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(dream_stories)")
        }
        if "goal" not in columns:
            conn.execute("ALTER TABLE dream_stories ADD COLUMN goal TEXT")

    @staticmethod
    def _migrate_tracked_suspension_v5(conn: sqlite3.Connection) -> None:
        """v4 -> v5: add the nullable ``dream_tracked_items.suspension``.

        Same additive, idempotent shape as the v4 step: it runs only when
        the on-disk table lacks the column (a fresh v5 build declares it
        last, where ``ADD COLUMN`` also puts it). Existing wrappers keep
        NULL -- not suspended.
        """
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(dream_tracked_items)")
        }
        if "suspension" not in columns:
            conn.execute(
                "ALTER TABLE dream_tracked_items ADD COLUMN suspension TEXT")

    @staticmethod
    def _migrate_tracked_provenance_v6(conn: sqlite3.Connection) -> None:
        """v5 -> v6: three more facts a wrapper records about itself.

        Review of #2921, round 3. Same additive, idempotent shape as the
        v4 and v5 steps -- each column is added only when the on-disk
        table lacks it, in the order a fresh v6 build declares them (so
        both shapes order their columns alike):

        * ``egress_marked`` (T1): 1 once the wrapper's subscription has
          been stamped as story-derived (no trusted egress origin).
          Existing wrappers get 0 -- that IS the pending marker the
          one-time backfill works from, so no data step is needed here;
        * ``alert_rule_id`` (T11): the alert rule Dreams created for this
          watch, so it is never again recognised by its name. NULL for
          existing wrappers (adopted lazily, by exact identity);
        * ``reminder_event_date`` (T5): the event date this wrapper's
          reminder was created for. NULL for existing wrappers.
        """
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(dream_tracked_items)")
        }
        for column, declaration in (
            ("egress_marked", "INTEGER NOT NULL DEFAULT 0"),
            ("alert_rule_id", "INTEGER"),
            ("reminder_event_date", "TEXT"),
        ):
            if column not in columns:
                conn.execute(
                    "ALTER TABLE dream_tracked_items"
                    f" ADD COLUMN {column} {declaration}")

    # ------------------------------------------------------------------
    # Collections (one row per local date)
    # ------------------------------------------------------------------

    def create_collection(
        self, local_date: str, trigger: str, profile_digest: str
    ) -> int | None:
        """Insert one collection for a local date; None when the date is taken.

        The UNIQUE(local_date) index is the race guard: scheduler fire, boot
        catch-up, and manual trigger all funnel through this INSERT, and
        SQLite makes exactly one of them win.

        Args:
            local_date: Local calendar date (``YYYY-MM-DD``) bucketing the
                cycle.
            trigger: Which path fired the cycle
                (``scheduled``/``catchup``/``manual``/``refresh``).
            profile_digest: Snapshot identity of the interest profile the
                cycle ran against.

        Returns:
            The new collection id, or ``None`` when the date already has a
            row.
        """
        now = _utc_now_iso()
        with self.transaction() as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO dreams_collections"
                    " (local_date, status, profile_digest, trigger, created_at)"
                    " VALUES (?, 'generating', ?, ?, ?)",
                    (local_date, profile_digest, trigger, now),
                )
            except sqlite3.IntegrityError:
                return None
            return int(cursor.lastrowid)

    def get_collection_by_date(self, local_date: str) -> dict | None:
        """Return the collection row for a local date.

        Args:
            local_date: Local calendar date (``YYYY-MM-DD``).

        Returns:
            The ``dreams_collections`` row as a dict, or None.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM dreams_collections WHERE local_date = ?",
                (local_date,),
            ).fetchone()
        return dict(row) if row is not None else None

    def set_collection_status(
        self,
        collection_id: int,
        status: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        degradation_notes: str | None = None,
        completed_at: str | None = None,
        story_count: int | None = None,
        claim: tuple[str, str] | None = None,
    ) -> bool:
        """Update a cycle's outcome fields; unpassed fields keep their value.

        Args:
            collection_id: Collection to update.
            status: Terminal or in-flight status
                (``generating``/``complete``/``partial``/``failed``).
            provider: Provider that generated the stories, when known.
            model: Model that generated the stories, when known.
            degradation_notes: Degradations observed during the cycle.
            completed_at: Completion timestamp, when the cycle finished.
            story_count: The collection's total story rows, when known.
            claim: ``(local_date, created_at)`` of the row the caller
                claimed (see :meth:`insert_story`); the update then only
                touches that very row.

        Returns:
            True when a row was updated; False when ``collection_id`` no
            longer names the collection the caller claimed (or any).
        """
        with self.transaction() as conn:
            return conn.execute(
                "UPDATE dreams_collections SET"
                " status = ?,"
                " provider = COALESCE(?, provider),"
                " model = COALESCE(?, model),"
                " degradation_notes = COALESCE(?, degradation_notes),"
                " completed_at = COALESCE(?, completed_at),"
                " story_count = COALESCE(?, story_count)"
                " WHERE id = ?" + self._CLAIM_SQL[claim is not None],
                (status, provider, model, degradation_notes, completed_at,
                 story_count, collection_id, *(claim or ())),
            ).rowcount == 1

    #: Appended to a write that must land in the collection its caller
    #: CLAIMED, not merely in whatever row carries that id now.
    _CLAIM_SQL = ("", " AND local_date = ? AND created_at = ?")

    def delete_empty_collection(self, collection_id: int) -> bool:
        """Remove a cycle's own collection row that never received a story.

        For a cycle the master switch stopped before its first story
        (review of #2921, item 7): nothing was produced, so the date is
        handed back -- a ``failed`` row would be read as an attempt, which
        the projection retries only a cadence later and the boot catch-up
        treats as "today already ran". One conditional DELETE: it can only
        remove a row that holds no story and is still ``generating`` -- or
        ``failed``, which is what another cycle's stale sweep
        (``fail_stale_generating``) turns a long-running cycle's row into
        while it is still running (round 3, N5); empty, it is the same
        nothing either way. A ``complete``/``partial`` row is never
        touched.

        The caller must pass the id of the row ITS OWN run created (never
        an append target). Two app processes on one profile are allowed --
        the profile lock is advisory (``TldwCli._maybe_warn_second_instance``
        in ``app.py``) -- so in a rare corner a second process may already
        be appending to this still-empty row when it is removed. Its story
        insert is then refused -- by the foreign key while the id names no
        row, and by the CLAIM check once the id has been handed out again:
        ``id`` is a plain ``INTEGER PRIMARY KEY``, so the next collection
        may well get this very id, and without the check the late story
        landed in it (round 4, I5; see :meth:`insert_story`). That
        process's cycle ends with the error and the date is open for the
        next cycle. A row that already holds a story is never removed.

        Args:
            collection_id: The collection the halted cycle had claimed.

        Returns:
            True when the row was removed.
        """
        with self.transaction() as conn:
            return conn.execute(
                "DELETE FROM dreams_collections"
                " WHERE id = ? AND status IN ('generating', 'failed')"
                " AND NOT EXISTS ("
                "SELECT 1 FROM dream_stories WHERE collection_id = ?)",
                (collection_id, collection_id),
            ).rowcount == 1

    def fail_stale_generating(self, cutoff_iso: str) -> int:
        """Reclaim dates wedged by a crashed cycle (spec §idempotency).

        Args:
            cutoff_iso: ISO timestamp; ``generating`` rows whose
                ``created_at`` predates it are marked ``failed``.

        Returns:
            The number of reclaimed rows.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE dreams_collections SET status = 'failed',"
                " degradation_notes = COALESCE(degradation_notes || '; ', '')"
                " || 'reclaimed: stale generating row'"
                " WHERE status = 'generating' AND created_at < ?",
                (cutoff_iso,),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Stories
    # ------------------------------------------------------------------

    def insert_story(
        self,
        collection_id: int,
        *,
        title: str,
        url: str,
        snippet: str,
        body: str,
        status: str,
        source: str,
        kind: str,
        event_date: str | None,
        location: str | None,
        matched_topics: list[str],
        query: str,
        error: str | None = None,
        goal: str | None = None,
        claim: tuple[str, str] | None = None,
    ) -> int:
        """Append one story outcome row; UNIQUE(collection_id, url) dedupes.

        Every candidate becomes a row — ``complete``, ``empty``, or
        ``failed`` — mirroring ``briefing_service`` discipline.

        Args:
            collection_id: Collection the story belongs to.
            title: Story headline.
            url: Source URL (unique within the collection).
            snippet: Search-result snippet the story was built from.
            body: Generated second-person story body.
            status: Outcome (``complete``/``empty``/``failed``).
            source: Where the candidate came from
                (``web``/``watchlist``/``llm``).
            kind: Event metadata kind
                (``content``/``event``/``deal``/``social_opportunity``/
                ``unknown``).
            event_date: Date extracted from explicit source text, or None.
            location: Location extracted from explicit source text, or None.
            matched_topics: Profile topics the story matched (stored
                JSON); the feedback loop joins these texts back to the
                profile's topic rows.
            query: The search query that surfaced the candidate.
            error: Failure reason for ``failed`` rows, or None.
            goal: Stored text of the goal that ``query`` was written
                for -- the story's goal provenance -- or None when no
                goal's query surfaced it. Never inferred from the
                story's text.
            claim: ``(local_date, created_at)`` of the collection row the
                caller claimed when its cycle began. The story is then
                written only while ``collection_id`` still names THAT row
                (review of #2921, round 4, I5): ids are reused after
                ``delete_empty_collection``, and a late insert carrying a
                stale id otherwise lands in a collection that is not the
                caller's. ``created_at`` is the token -- the date alone
                does not tell a halted cycle's row from the next one made
                for the same day. (Millisecond resolution: a reuse inside
                the same millisecond is not told apart.)

        Returns:
            The new story id.

        Raises:
            sqlite3.IntegrityError: If the URL already exists in the
                collection, or ``collection_id`` no longer names the
                collection that was claimed.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO dream_stories"
                " (collection_id, title, url, snippet, body, status, source,"
                " kind, event_date, location, matched_topics, query, kept,"
                " error, created_at, goal)"
                " SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?"
                + (" WHERE EXISTS (SELECT 1 FROM dreams_collections"
                   " WHERE id = ?" + self._CLAIM_SQL[1] + ")"
                   if claim is not None else ""),
                (
                    collection_id, title, url, snippet, body, status, source,
                    kind, event_date, location, json.dumps(matched_topics),
                    query, error, _utc_now_iso(), goal,
                    *((collection_id, *claim) if claim is not None else ()),
                ),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "dream collection is no longer the one this cycle claimed")
            return int(cursor.lastrowid)

    @staticmethod
    def _story_dict(row: sqlite3.Row) -> dict:
        """Decode one story row, parsing the matched-topics JSON column."""
        story = dict(row)
        story["matched_topics"] = json.loads(story["matched_topics"])
        return story

    def list_stories(self, collection_id: int) -> list[dict]:
        """Return a collection's stories in insert order."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_stories WHERE collection_id = ?"
                " ORDER BY id",
                (collection_id,),
            ).fetchall()
        return [self._story_dict(row) for row in rows]

    def list_recent_stories(self, limit: int = 20) -> list[dict]:
        """Return recent stories, newest collection first.

        Ordering is by collection ``local_date`` descending, kept stories
        first within a collection, then insert order. Each row carries the
        owning collection's ``local_date`` alongside the story columns.

        Args:
            limit: Maximum number of stories to return.

        Returns:
            Story dicts (see :meth:`list_stories`) with ``local_date``
            added.
        """
        limit = clamp_limit(limit)
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT s.*, c.local_date AS local_date"
                " FROM dream_stories AS s"
                " JOIN dreams_collections AS c ON c.id = s.collection_id"
                " ORDER BY c.local_date DESC, s.kept DESC, s.id ASC"
                " LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._story_dict(row) for row in rows]

    def get_story(self, story_id: int) -> dict | None:
        """Return one story row by id, or ``None``.

        task-33164 (Phase 2 Track story): the Artifacts screen's Tracked
        rows resolve their ``origin_story_id`` through this targeted read
        because the ``list_recent_stories`` window may no longer contain
        the origin -- a long-lived watch outlives the bounded recent-stories
        slice. Same row shape as :meth:`list_recent_stories` (all story
        columns, ``matched_topics`` parsed, the owning collection's
        ``local_date`` joined in) so the story modal can open it unchanged.

        Args:
            story_id: The ``dream_stories`` row to read.

        Returns:
            The story dict with ``local_date`` added, or ``None`` when no
            row carries that id.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT s.*, c.local_date AS local_date"
                " FROM dream_stories AS s"
                " JOIN dreams_collections AS c ON c.id = s.collection_id"
                " WHERE s.id = ?",
                (story_id,),
            ).fetchone()
        return self._story_dict(row) if row is not None else None

    def set_story_kept(self, story_id: int, kept: bool) -> None:
        """Mark a story kept (or clear the mark), stamping kept_at."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_stories SET kept = ?, kept_at = ? WHERE id = ?",
                (int(bool(kept)), _utc_now_iso() if kept else None, story_id),
            )

    def record_feedback(self, story_id: int, kind: str) -> None:
        """Append one feedback event for a story (spec §feedback loop).

        Args:
            story_id: Story the feedback applies to.
            kind: Feedback kind (``more``/``less``/``kept``/``dived``/
                ``exported``/``ingested``/``tracked``).
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO dream_feedback (story_id, kind, created_at)"
                " VALUES (?, ?, ?)",
                (story_id, kind, _utc_now_iso()),
            )

    # ------------------------------------------------------------------
    # Seen ledger (cross-cycle URL dedupe)
    # ------------------------------------------------------------------

    def seen_upsert(self, urls: list[tuple[str, str]]) -> None:
        """Record (url, title_digest) pairs as seen, refreshing last_seen.

        Args:
            urls: (url, title_digest) pairs to record.
        """
        if not urls:
            return
        now = _utc_now_iso()
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO dream_seen_items"
                " (url, title_digest, first_seen, last_seen)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(url) DO UPDATE SET"
                " title_digest = excluded.title_digest,"
                " last_seen = excluded.last_seen",
                [(url, digest, now, now) for url, digest in urls],
            )

    def seen_filter_unseen(self, urls: list[str]) -> set[str]:
        """Return the subset of urls not yet in the seen ledger."""
        if not urls:
            return set()
        unique_urls = list(dict.fromkeys(urls))
        placeholders = ", ".join("?" for _ in unique_urls)
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT url FROM dream_seen_items WHERE url IN ({placeholders})",
                unique_urls,
            ).fetchall()
        seen = {str(row[0]) for row in rows}
        return set(unique_urls) - seen

    def prune_seen(self, cutoff_iso: str) -> int:
        """Delete seen items whose last_seen predates the cutoff.

        Args:
            cutoff_iso: ISO timestamp; items last seen before it are purged.

        Returns:
            The number of pruned rows.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM dream_seen_items WHERE last_seen < ?",
                (cutoff_iso,),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Interest profile
    # ------------------------------------------------------------------

    def list_profile(self) -> list[dict]:
        """Return profile entries, heaviest weight first."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_interest_profile"
                " ORDER BY weight DESC, facet, text"
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_profile_entry(
        self,
        facet: str,
        text: str,
        *,
        weight: float,
        searchable: int,
        source: str,
    ) -> None:
        """Insert or refresh one profile entry, keyed on (facet, text).

        ``query_angle`` and ``last_boosted_at`` are owned by other flows
        and are never touched here.

        Args:
            facet: Entry kind (``topic``/``goal``).
            text: The distilled topic or goal text.
            weight: Feedback-adjusted weight.
            searchable: 1 when the entry may appear in search queries.
            source: Where the entry came from (``user``/``seed``/
                ``personal_context``/``notes``/``media``/``guardian``).
        """
        now = _utc_now_iso()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO dream_interest_profile"
                " (facet, text, weight, searchable, source, created_at,"
                " updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(facet, text) DO UPDATE SET"
                " weight = excluded.weight,"
                " searchable = excluded.searchable,"
                " source = excluded.source,"
                " updated_at = excluded.updated_at",
                (facet, text, weight, searchable, source, now, now),
            )

    def delete_profile_entry(self, entry_id: int) -> None:
        """Remove one profile entry by id."""
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM dream_interest_profile WHERE id = ?", (entry_id,)
            )

    def set_goal_query_angle(
        self, facet: str, text: str, *, angle: str | None
    ) -> None:
        """Write (or clear) a GOAL row's query angle (task-33165).

        The other feedback half: feedback on a goal-derived story steers that
        goal's ``query_angle`` — a short note like ``"prefer: deal"`` /
        ``"avoid: event"`` consulted by query synthesis — INSTEAD of any
        weight math, so the UPDATE never touches ``weight``. The write is a
        direct one (no COALESCE): ``angle=None`` is a real CLEAR, not a
        skipped update. Reuses the profile-row ``(facet, text)`` key shape
        but is pinned to goal rows — only goals own a query angle.

        Args:
            facet: Must be ``"goal"``; anything else is a caller bug.
            text: The goal's stored text (exact, parameterized match).
            angle: The steering note to record, or ``None`` to clear it.

        Returns:
            None; a missing row matches zero rows and is a benign no-op.

        Raises:
            ValueError: If ``facet`` is not ``"goal"``.
        """
        if facet != "goal":
            raise ValueError(
                f"query angles are goal-only; got facet {facet!r}")
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_interest_profile"
                " SET query_angle = ?, updated_at = ?"
                " WHERE facet = 'goal' AND text = ?",
                (angle, _utc_now_iso(), text),
            )

    # ------------------------------------------------------------------
    # Daily usage budgets
    # ------------------------------------------------------------------

    def usage_bump(
        self, local_date: str, *, searches: int = 0, llm_calls: int = 0
    ) -> None:
        """Accumulate usage counters for a local date.

        Args:
            local_date: Local calendar date the usage belongs to.
            searches: Search calls to add.
            llm_calls: LLM calls to add.
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO dream_daily_usage (local_date, searches, llm_calls)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(local_date) DO UPDATE SET"
                " searches = searches + excluded.searches,"
                " llm_calls = llm_calls + excluded.llm_calls",
                (local_date, searches, llm_calls),
            )

    #: The read and the write of each reservable counter. The column is
    #: chosen by key from this table, never interpolated from a caller's
    #: string.
    _USAGE_RESERVE_SQL: ClassVar[dict[str, tuple[str, str]]] = {
        "searches": (
            "SELECT searches FROM dream_daily_usage WHERE local_date = ?",
            ("UPDATE dream_daily_usage SET searches = searches + ?"
             " WHERE local_date = ?"),
        ),
        "llm_calls": (
            "SELECT llm_calls FROM dream_daily_usage WHERE local_date = ?",
            ("UPDATE dream_daily_usage SET llm_calls = llm_calls + ?"
             " WHERE local_date = ?"),
        ),
    }

    def usage_reserve(
        self, local_date: str, *, counter: str, limit: int, count: int = 1
    ) -> int:
        """Spend up to ``count`` units of a daily counter, never past its limit.

        The read and the increment happen inside one ``BEGIN IMMEDIATE``
        transaction, so concurrent spenders -- track checks of different
        items and the daily cycle, each on its own worker-thread
        connection -- can never both take the last slot (Qodo #11 and
        review item 6, PR #2921). ``usage_get`` followed by a later
        ``usage_bump`` cannot give that guarantee: both readers see the
        slot free. Every Dreams spend goes through here BEFORE the work
        it pays for.

        Args:
            local_date: Local calendar date the usage belongs to.
            counter: Which counter to spend (``searches``/``llm_calls``).
            limit: The day's budget for that counter.
            count: How many units are wanted. Every production caller
                asks for one, immediately before the search or model call
                it pays for.

        Returns:
            How many units were granted and are now counted: ``count``
            when the day has room, fewer when only part of it is left, 0
            when the counter already meets ``limit`` (nothing changed).

        Raises:
            ValueError: If ``counter`` is not a reservable counter.
        """
        statements = self._USAGE_RESERVE_SQL.get(counter)
        if statements is None:
            raise ValueError(f"not a reservable usage counter: {counter!r}")
        read, write = statements
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO dream_daily_usage (local_date)"
                " VALUES (?)",
                (local_date,),
            )
            spent = int(conn.execute(read, (local_date,)).fetchone()[0])
            granted = max(0, min(int(count), int(limit) - spent))
            if granted:
                conn.execute(write, (granted, local_date))
            return granted

    def usage_get(self, local_date: str) -> dict:
        """Return the day's counters, zeroed when nothing was recorded."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT searches, llm_calls FROM dream_daily_usage"
                " WHERE local_date = ?",
                (local_date,),
            ).fetchone()
        if row is None:
            return {"searches": 0, "llm_calls": 0}
        return {"searches": int(row["searches"]), "llm_calls": int(row["llm_calls"])}

    # ------------------------------------------------------------------
    # Tracked items (Phase 2 track loop)
    # ------------------------------------------------------------------

    def create_tracked_item(
        self,
        *,
        origin_story_id: int | None = None,
        mechanism: str,
        intent: str,
        subscription_id: int | None = None,
        query_template: str | None = None,
        event_date: str | None = None,
        cadence_seconds: int,
        created_by_dreams: int = 0,
        egress_marked: int = 0,
        alert_rule_id: int | None = None,
    ) -> int:
        """Insert one tracked item; the CHECK constraints are the vocabulary.

        Args:
            origin_story_id: Dream story that prompted tracking, if any.
            mechanism: How the item is watched (``page``/``question``).
            intent: What the user wants out of it
                (``event``/``deal``/``topic``).
            subscription_id: Media subscription usable for the checks, if any.
            query_template: Search query the check re-runs, if any.
            event_date: Date the target event is pinned to, if any.
            cadence_seconds: Minimum seconds between checks.
            created_by_dreams: 1 when the Dreams loop (not the user) added it.
            egress_marked: 1 when the caller has stamped the subscription
                as story-derived (see ``subscriptions_awaiting_egress_mark``).
            alert_rule_id: The alert rule Dreams created for the watch's
                page, if any (see ``set_tracked_alert_rule``).

        Returns:
            The new tracked item id.

        Raises:
            sqlite3.IntegrityError: If ``mechanism`` or ``intent`` is
                outside its CHECK vocabulary.
        """
        now = _utc_now_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO dream_tracked_items"
                " (origin_story_id, mechanism, intent, subscription_id,"
                " query_template, event_date, cadence_seconds, status,"
                " created_by_dreams, egress_marked, alert_rule_id,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)",
                (
                    origin_story_id, mechanism, intent, subscription_id,
                    query_template, event_date, cadence_seconds,
                    created_by_dreams, int(bool(egress_marked)),
                    alert_rule_id, now, now,
                ),
            )
            return int(cursor.lastrowid)

    def get_tracked_item(self, tracked_item_id: int) -> dict | None:
        """Return one tracked item row by id, or None.

        Args:
            tracked_item_id: The ``dream_tracked_items`` row to read.

        Returns:
            The full row as a dict (all ``dream_tracked_items``
            columns), or ``None`` when no row carries that id.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE id = ?",
                (tracked_item_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_tracked_items(
        self, status: str = "active", *, limit: int = TRACKED_ITEMS_LIST_LIMIT
    ) -> list[dict]:
        """Return tracked items in one status, newest first, bounded.

        Args:
            status: Lifecycle filter (``active``/``paused``/``retired``).
            limit: Maximum rows returned, bounded to
                ``1..TRACKED_ITEMS_LIST_LIMIT`` (Qodo #9: retired rows
                accumulate forever, so the read is capped the same way
                ``clamp_limit`` caps the run-list reads).

        Returns:
            Up to ``limit`` matching rows ordered ``created_at DESC``
            (``id DESC`` breaks ties so same-millisecond inserts keep
            newest-first order).
        """
        limit = max(1, min(int(limit), TRACKED_ITEMS_LIST_LIMIT))
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE status = ?"
                " ORDER BY created_at DESC, id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def subscription_dream_owned(self, subscription_id: int) -> bool:
        """Whether any tracked item pins dream ownership of a subscription.

        Provenance is a fact about the SUBSCRIPTION, not about one
        wrapper's lifecycle state (ruling P10), so every status counts: a
        retired or paused dream-created row proves ownership exactly as
        well as an active one. A targeted read rather than a scan of
        ``list_tracked_items`` (Qodo #6, PR #2921): those lists are capped
        at ``TRACKED_ITEMS_LIST_LIMIT`` rows per status, so the owning row
        of a long-lived subscription eventually fell outside them and the
        re-track refused (or un-owned) a subscription Dreams created.

        Args:
            subscription_id: The watchlists subscription to check.

        Returns:
            True when at least one ``dream_tracked_items`` row for that
            subscription carries ``created_by_dreams = 1``.
        """
        # ponytail: unindexed -- a scan of one row per watch the user ever
        # tracked, stopped at the first hit. Add an EXPLAIN-pinned
        # (subscription_id) index if that table ever grows past thousands.
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM dream_tracked_items"
                " WHERE subscription_id = ? AND created_by_dreams = 1"
                " LIMIT 1",
                (subscription_id,),
            ).fetchone()
        return row is not None

    def list_tracked_for_story(self, origin_story_id: int) -> list[dict]:
        """Every tracked item a story ever had, any status, oldest first.

        Round 3, T5: whether an event reminder was already made -- and
        delivered -- is a fact about the STORY and its event date, not
        about one wrapper, so it is read across all of the story's
        wrappers (an untracked one, a paused one, the page watch and the
        question watch alike).

        Args:
            origin_story_id: The dream story.

        Returns:
            The full rows naming that story as their origin, ``id``
            ascending (``idx_dream_tracked_origin_story`` seeks them).
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE origin_story_id = ?"
                " ORDER BY id",
                (origin_story_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_tracked_reminder_date(
        self, tracked_item_id: int, event_date: str | None
    ) -> None:
        """Record the event date a wrapper's reminder was made for.

        Args:
            tracked_item_id: The wrapper.
            event_date: The ISO event date, or None.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET reminder_event_date = ?"
                " WHERE id = ?",
                (event_date, tracked_item_id),
            )

    def find_tracked_by_story(
        self, origin_story_id: int, *, mechanism: str | None = None,
        include_paused: bool = False,
    ) -> dict | None:
        """Return the newest ACTIVE tracked item from a story, or None.

        Final review (ruling P10): the story modal's untrack resolves
        through this lookup, so a retired/paused row must never surface --
        re-retiring a sweep-retired row would rewrite its
        ``retired_reason`` audit data (COALESCE) and disable a
        subscription the sweep deliberately left live. With the filter the
        story rows' active-only tracked badge and the ``u`` action agree
        on what "tracked" means.

        One story may own a page watch AND a question watch (Qodo #10/#3,
        PR #2921), and "the newest" then names whichever was added last.
        A caller that means one of them -- the page untrack, or either
        track path looking for the wrapper to reuse -- passes
        ``mechanism``; without it the newest active wrapper of any
        mechanism is returned (the "is this story tracked at all"
        question).

        Args:
            origin_story_id: The dream story whose tracked wrapper to find.
            mechanism: Restrict the lookup to ``"page"`` or ``"question"``
                wrappers; ``None`` (the default) accepts either.
            include_paused: Also accept a PAUSED wrapper (an active one is
                still preferred). Only the two track paths ask for this,
                looking for the wrapper to reuse (round 3, T5): a watch
                the sweep paused is still the story's watch, and tracking
                the story again must not stack a second wrapper -- and a
                second reminder -- beside it. Untrack and the tracked
                badge keep the active-only default.

        Returns:
            The newest matching row naming that story as its origin (dict
            of all columns), or ``None`` when the story has no such
            wrapper.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE origin_story_id = ?"
                " AND (status = 'active' OR (? AND status = 'paused'))"
                " AND (? IS NULL OR mechanism = ?)"
                " ORDER BY (status = 'active') DESC, created_at DESC,"
                " id DESC LIMIT 1",
                (origin_story_id, 1 if include_paused else 0,
                 mechanism, mechanism),
            ).fetchone()
        return dict(row) if row is not None else None

    def set_tracked_plan(
        self, tracked_item_id: int, *, cadence_seconds: int, event_date: str | None
    ) -> None:
        """Rewrite one tracked item's check plan (cadence + event date).

        The re-track reuse path (Qodo #2, PR #2890): when a story already
        has an ACTIVE same-mechanism wrapper, ``track_page`` /
        ``track_question`` refresh the caller's cadence and event date on
        the existing row instead of inserting a second wrapper. Both
        fields are always written -- the caller passes the full new plan,
        so clearing an event date (``event_date=None``) works.

        Args:
            tracked_item_id: The wrapper row whose plan to rewrite.
            cadence_seconds: New minimum seconds between checks.
            event_date: New target-event date (ISO), or ``None`` to clear.

        A missing row simply matches zero rows (the caller just read the
        row, and a concurrent retire is a benign no-op); the UPDATE is
        additionally pinned to ``status='active'`` so a retire landing
        between the read and this write wins.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET"
                " cadence_seconds = ?, event_date = ?, updated_at = ?"
                " WHERE id = ? AND status = 'active'",
                (
                    cadence_seconds, event_date, _utc_now_iso(), tracked_item_id,
                ),
            )

    def set_tracked_suspension(
        self, tracked_item_id: int, record: str | None, *,
        only_live: bool = False,
    ) -> None:
        """Write (or clear) what the master switch stopped for one wrapper.

        While ``[dreams] enabled`` is off, the switch stops the work a
        watch scheduled in other queues -- its subscription, its reminder,
        its alert rule -- and records here exactly what IT stopped
        (review of #2921, item 4). Switching back on restores that record
        and nothing else, so something the user stopped themselves stays
        stopped. The record is opaque text to this layer (the track
        service owns its JSON shape, ``track_service``'s "The master
        switch" section documents it); it is not a lifecycle state, and
        ``status`` is untouched.

        Args:
            tracked_item_id: The wrapper the record belongs to.
            record: The serialized record, or ``None`` when the wrapper is
                no longer suspended.
            only_live: Write only while the wrapper is still active or
                paused. The switch passes work from a list read at their
                start; a wrapper retired since then carries the record of
                its retirement, which is not theirs to rewrite (review of
                #2921, round 4). Decided in the UPDATE itself, so there is
                no read-then-write window.

        Returns:
            None; a missing row -- or, with ``only_live``, a retired one
            -- matches zero rows and is a benign no-op.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET suspension = ?, updated_at = ?"
                " WHERE id = ?"
                + (" AND status IN ('active', 'paused')" if only_live else ""),
                (record, _utc_now_iso(), tracked_item_id),
            )

    #: The startup reconcile's one read: every live (active or paused)
    #: wrapper's suspension record. Bounded by the tracked-item cap plus
    #: the paused watches, and served by the status index.
    _LIVE_SUSPENSIONS_SQL = (
        "SELECT suspension FROM dream_tracked_items"
        " WHERE status IN ('active', 'paused')"
    )

    def live_tracked_suspensions(self) -> list[str | None]:
        """The suspension record of every live wrapper, as stored.

        The startup reconcile's one cheap read (review of #2921, item 3c):
        it runs on every boot, before the Dreams service chain is imported
        at all, and ``track_reconcile.needs_switch`` answers from it
        whether any live watch is not yet settled on the side of the
        master switch the config says. The records stay opaque text to
        this layer.

        Returns:
            One entry per active or paused wrapper: its record, or
            ``None`` for a wrapper that carries none. Retired wrappers
            never appear.
        """
        with self.connection() as conn:
            return [
                row[0] for row in conn.execute(self._LIVE_SUSPENSIONS_SQL)
            ]

    def list_tracked_for_subscription(self, subscription_id: int) -> list[dict]:
        """Every tracked item that rides one subscription, any status.

        Facts about a SUBSCRIPTION -- which alert rule Dreams created on
        it (round 3, T11), who can account for its being disabled (T10) --
        are read from all the wrappers that ever rode it, oldest first.

        Args:
            subscription_id: The watchlists subscription.

        Returns:
            The full rows, ``id`` ascending.
        """
        # ponytail: unindexed, like ``subscription_dream_owned`` -- one row
        # per watch the user ever tracked. Index (subscription_id) with a
        # plan pin if that table ever grows past thousands.
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE subscription_id = ?"
                " ORDER BY id",
                (subscription_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_tracked_alert_rule(
        self, subscription_id: int, rule_id: int, *, only_unrecorded: bool
    ) -> None:
        """Record the alert rule Dreams created on a subscription.

        Dreams identifies its rule by this id and nothing else (round 3,
        T11). ``0`` means "looked, there is none": a wrapper from before
        ids were recorded whose rule could not be identified exactly is
        not looked at again.

        Args:
            subscription_id: The subscription the rule is pinned on.
            rule_id: The rule's id, or 0 for "none".
            only_unrecorded: Write only wrappers that record nothing yet
                (``alert_rule_id IS NULL``), leaving recorded ids alone.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET alert_rule_id = ?"
                " WHERE subscription_id = ?"
                + (" AND alert_rule_id IS NULL" if only_unrecorded else ""),
                (int(rule_id), subscription_id),
            )

    #: Subscriptions Dreams created whose wrappers are not marked yet.
    # ponytail: unindexed -- a scan of one row per watch the user ever
    # tracked. Add an EXPLAIN-pinned partial index if that table ever
    # grows past thousands.
    _EGRESS_BACKLOG_SQL = (
        "SELECT DISTINCT subscription_id FROM dream_tracked_items"
        " WHERE created_by_dreams = 1 AND subscription_id IS NOT NULL"
        " AND egress_marked = 0 ORDER BY subscription_id"
    )

    def subscriptions_awaiting_egress_mark(self) -> list[int]:
        """Subscriptions Dreams created that are not stamped story-derived.

        Review of #2921, round 3, T1. A subscription Dreams creates is
        marked so the watchlist fetch gives its URL no trusted egress
        origin. Ones created before that mark existed carry none, and
        were never even checked for a public destination, so they stayed
        trusted origins indefinitely. Ownership -- any wrapper with
        ``created_by_dreams = 1``, whatever its status: a retired watch's
        source still exists and can be run from Watchlists -- is what
        entitles Dreams to stamp them, and ``egress_marked = 0`` is the
        persisted "still to do" marker (the v6 migration leaves it on
        every existing wrapper).

        Returns:
            The distinct subscription ids still to stamp, ascending.
        """
        with self.connection() as conn:
            return [int(row[0]) for row in conn.execute(self._EGRESS_BACKLOG_SQL)]

    def dream_owned_subscription_ids(self) -> list[int]:
        """Every subscription Dreams created, whatever the marker says.

        What the verifying egress backfill walks (review of #2921, round
        4, B4): ``egress_marked`` lives here and the mark itself in the
        subscriptions store, which can be restored from a backup taken
        before the mark while this store is not.

        Returns:
            The distinct subscription ids of ``created_by_dreams``
            wrappers of any status, ascending.
        """
        # ponytail: unindexed, the same scan as ``_EGRESS_BACKLOG_SQL``.
        with self.connection() as conn:
            return [int(row[0]) for row in conn.execute(
                "SELECT DISTINCT subscription_id FROM dream_tracked_items"
                " WHERE created_by_dreams = 1 AND subscription_id IS NOT NULL"
                " ORDER BY subscription_id")]

    def set_egress_marked(self, subscription_id: int) -> None:
        """Record that a Dreams-created subscription has been stamped.

        Args:
            subscription_id: The subscription just stamped (or found
                deleted -- nothing is left to stamp either way).
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET egress_marked = 1"
                " WHERE subscription_id = ? AND created_by_dreams = 1",
                (subscription_id,),
            )

    _BOOT_BACKLOG_SQL = (
        "SELECT 1 FROM dream_tracked_items"
        " WHERE created_by_dreams = 1 AND subscription_id IS NOT NULL"
        " AND egress_marked = 0 LIMIT 1"
    )

    _LIVE_BOOT_ROWS_SQL = (
        "SELECT id, subscription_id, created_by_dreams, alert_rule_id,"
        " event_date, suspension FROM dream_tracked_items"
        " WHERE status IN ('active', 'paused')"
    )

    _RETIRED_BOOT_ROWS_SQL = (
        "SELECT id, subscription_id, created_by_dreams, alert_rule_id,"
        " event_date, suspension FROM dream_tracked_items"
        " WHERE status = 'retired' AND suspension IS NOT NULL"
    )

    def tracking_boot_rows(
        self, *, retired: bool = False,
    ) -> tuple[list[dict], list[dict], bool]:
        """:meth:`tracking_boot_state`, with each wrapper's row.

        The start check of TASK-34924 needs, beside each live wrapper's
        record, the source and rule it rides and whether Dreams created
        that source -- and, with Dreams off, what a retired wrapper's
        record says Dreams stopped when it retired.

        Args:
            retired: Also read the retired wrappers that carry a record.

        Returns:
            ``(live, retired_rows, egress_backlog)``: one dict per active or
            paused wrapper (``id``, ``subscription_id``,
            ``created_by_dreams``, ``alert_rule_id``, ``event_date``,
            ``suspension``), the same for each retired wrapper with a
            record (empty unless asked for), and the backlog flag of
            :meth:`tracking_boot_state`.
        """
        with self.connection() as conn:
            live = [dict(row) for row in conn.execute(self._LIVE_BOOT_ROWS_SQL)]
            gone = ([dict(row) for row in conn.execute(
                self._RETIRED_BOOT_ROWS_SQL)] if retired else [])
            backlog = conn.execute(
                self._BOOT_BACKLOG_SQL).fetchone() is not None
        return live, gone, backlog

    def tracking_boot_state(self) -> tuple[list[str | None], bool]:
        """Everything the startup reconcile asks, in one hop.

        Returns:
            ``(records, egress_backlog)``: the suspension record of every
            live wrapper (see :meth:`live_tracked_suspensions`), and
            whether any Dreams-created subscription is still awaiting its
            egress mark (see :meth:`subscriptions_awaiting_egress_mark`).
        """
        with self.connection() as conn:
            records = [
                row[0] for row in conn.execute(self._LIVE_SUSPENSIONS_SQL)]
            # ponytail: unindexed -- on EVERY start, one scan of
            # ``dream_tracked_items`` (the status index above does not
            # help: ownership spans every status). The table holds one
            # small row per watch the user ever tracked; nothing prunes
            # retired ones, so it grows with use, by tens to hundreds of
            # rows over years at the 20-watch cap, and the scan stops at
            # the first hit. Add an EXPLAIN-pinned partial index
            # (created_by_dreams = 1 AND egress_marked = 0) if it ever
            # reaches thousands -- the same ceiling as
            # ``_EGRESS_BACKLOG_SQL``.
            backlog = conn.execute(
                self._BOOT_BACKLOG_SQL).fetchone() is not None
        return records, backlog

    def set_tracked_status(
        self,
        tracked_item_id: int,
        status: str,
        *,
        retired_reason: str | None = None,
    ) -> None:
        """Move a tracked item between lifecycle states, stamping updated_at.

        ``retired_reason`` keeps the stored value when not passed (COALESCE),
        so pausing or reactivating never erases why an item was retired.

        Args:
            tracked_item_id: Item to update.
            status: New lifecycle state
                (``active``/``paused``/``retired``).
            retired_reason: Why the item was retired, when retiring.

        Returns:
            None; a missing row matches zero rows and is a benign no-op.

        Raises:
            sqlite3.IntegrityError: If ``status`` is outside its CHECK
                vocabulary.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET"
                " status = ?,"
                " retired_reason = COALESCE(?, retired_reason),"
                " updated_at = ?"
                " WHERE id = ?",
                (status, retired_reason, _utc_now_iso(), tracked_item_id),
            )

    def touch_tracked_checked(self, tracked_item_id: int, now_iso: str) -> None:
        """Stamp a tracked item's last_checked (the scheduler's due clock).

        Args:
            tracked_item_id: Item that was just checked.
            now_iso: Check-completion timestamp (canonical UTC shape).

        Returns:
            None; a missing row matches zero rows and is a benign no-op.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET"
                " last_checked = ?, updated_at = ?"
                " WHERE id = ?",
                (now_iso, _utc_now_iso(), tracked_item_id),
            )

    def count_active_tracked(self) -> int:
        """Return how many tracked items are currently active.

        Returns:
            The COUNT of ``dream_tracked_items`` rows with
            ``status='active'`` -- the figure the ``tracked_item_cap``
            guard compares against.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM dream_tracked_items WHERE status = 'active'"
            ).fetchone()
        return int(row[0])

    def count_active_tracked_for_subscription(self, subscription_id: int) -> int:
        """Return how many ACTIVE tracked items ride one subscription.

        Two stories with the same URL resolve to one watchlists
        subscription and each owns a wrapper. Retiring one of them may
        disable the shared, dream-created source only when this is zero
        (Qodo #4, PR #2921) -- otherwise the remaining wrappers would
        stay active on a source nobody checks.

        Args:
            subscription_id: The watchlists subscription to count for.

        Returns:
            The COUNT of ``status='active'`` rows naming that
            subscription (the active set is bounded by
            ``tracked_item_cap``, and the status index seeks it).
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM dream_tracked_items"
                " WHERE status = 'active' AND subscription_id = ?",
                (subscription_id,),
            ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # Track runs (check outcomes per tracked item)
    # ------------------------------------------------------------------

    def insert_track_run(
        self,
        tracked_item_id: int,
        *,
        status: str,
        digest_hash: str | None,
        verdict_note: str = "",
        notified: int = 0,
    ) -> int:
        """Append one check-outcome row for a tracked item.

        Args:
            tracked_item_id: Item the run checked.
            status: Disposition (``changed``/``unchanged``/``baseline``/
                ``rebaselined``/``withheld``/``error``/``skipped``).
            digest_hash: Content digest the verdict was computed from.
            verdict_note: Short human-facing explanation, '' when none.
            notified: 1 when the run produced a user notification.

        Returns:
            The new run id.

        Raises:
            sqlite3.IntegrityError: If ``status`` is outside its CHECK
                vocabulary.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO dream_track_runs"
                " (tracked_item_id, status, digest_hash, verdict_note,"
                " notified, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tracked_item_id, status, digest_hash, verdict_note,
                    notified, _utc_now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def list_recent_track_runs(
        self, tracked_item_id: int, limit: int = 5
    ) -> list[dict]:
        """Return a tracked item's newest runs first.

        Args:
            tracked_item_id: Item whose runs to read.
            limit: Maximum number of runs (bounded to 1..MAX_LIST_LIMIT).

        Returns:
            Run rows ordered ``created_at DESC`` (``id DESC`` breaks ties).
        """
        limit = clamp_limit(limit)
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_track_runs WHERE tracked_item_id = ?"
                " ORDER BY created_at DESC, id DESC LIMIT ?",
                (tracked_item_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_anchor_digest(self, tracked_item_id: int) -> str | None:
        """Digest of the newest run that may anchor a comparison, or None.

        ``error``/``skipped``/``withheld`` runs never anchor (they carry
        no trustworthy digest, or a deliberately suppressed one); the
        newest run of any other status -- ``baseline``, ``unchanged``,
        ``changed``, ``rebaselined`` -- carries the digest the next check
        compares against. Read directly (Qodo #12, PR #2921): looking for
        it in a bounded slice of recent runs lost the anchor of a watch
        that had logged more non-anchoring runs than the slice held, and
        its next real change was then filed as a first baseline.

        Args:
            tracked_item_id: Item whose anchor to read.

        Returns:
            That run's ``digest_hash``, or ``None`` when the item has no
            anchoring run yet (or the anchoring run stored no digest).
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT digest_hash FROM dream_track_runs"
                " WHERE tracked_item_id = ?"
                " AND status NOT IN ('error', 'skipped', 'withheld')"
                " ORDER BY created_at DESC, id DESC LIMIT 1",
                (tracked_item_id,),
            ).fetchone()
        if row is None or not row["digest_hash"]:
            return None
        return str(row["digest_hash"])

    def consecutive_track_dispositions(
        self, tracked_item_id: int, status: str
    ) -> int:
        """Count the trailing run of one disposition, newest run backwards.

        The quiet-retire gate: N consecutive ``unchanged`` (or ``error``)
        runs is what retires an item without notifying the user, so the
        count must stop at the newest run whose status differs. With runs
        (oldest first) ``baseline, unchanged, unchanged, changed`` the
        trailing run of ``unchanged`` is 0 and of ``changed`` is 1.

        Args:
            tracked_item_id: Item whose run history to scan.
            status: Disposition to count the trailing run of.

        Returns:
            How many of the item's newest runs, walking backwards, hold
            ``status`` before any run with a different status appears.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM dream_track_runs AS r"
                " WHERE r.tracked_item_id = ? AND r.status = ?"
                " AND r.id > COALESCE(("
                "     SELECT b.id FROM dream_track_runs AS b"
                "     WHERE b.tracked_item_id = ? AND b.status <> ?"
                "     ORDER BY b.created_at DESC, b.id DESC LIMIT 1"
                " ), 0)",
                (tracked_item_id, status, tracked_item_id, status),
            ).fetchone()
        return int(row[0])
