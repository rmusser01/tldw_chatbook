"""SQLite persistence for the Dreams daily discovery subsystem."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

from ..Utils.timestamps import utc_now_iso
from .base_db import BaseDB


#: Upper bound for any list read the UI pages through.
MAX_LIST_LIMIT = 200


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
    ``dream_track_runs``) additively, so v1 files upgrade in place on open.
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

    _CURRENT_SCHEMA_VERSION = 2
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
                CHECK(source IN ('user', 'seed', 'personal_context', 'notes', 'media')),
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
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_dream_tracked_status
            ON dream_tracked_items(status)
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
        super().__init__(
            database_path, client_id if client_id is not None else "default"
        )

    # ------------------------------------------------------------------
    # Connection handling (Library_Collections_DB idiom)
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory_db:
            self._enable_wal(conn)
        # NORMAL is safe under WAL (app-crash-safe) and avoids an fsync per
        # commit; synchronous is per-connection, so it is re-applied on every
        # NEW connection here, the one place connections are created.
        conn.execute("PRAGMA synchronous = NORMAL")
        # A held (long-lived) connection needs true autocommit: Python's
        # default isolation mode auto-BEGINs on DML and would make the
        # explicit BEGIN in transaction() fail (task-3012).
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
        """Return this thread's held connection, opening or reviving it."""
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
        """Close the current thread's held connection, if any."""
        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _initialize_schema(self) -> None:
        """Atomically initialize the Dreams schema (additive, idempotent)."""
        with self.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version"
                " (version INTEGER PRIMARY KEY NOT NULL)"
            )
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current_version = int(row[0] or 0) if row is not None else 0
            if current_version > self._CURRENT_SCHEMA_VERSION:
                raise DreamsSchemaError("schema_too_new")
            for statement in self._SCHEMA_DDL:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self._CURRENT_SCHEMA_VERSION,),
            )

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
    ) -> None:
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
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dreams_collections SET"
                " status = ?,"
                " provider = COALESCE(?, provider),"
                " model = COALESCE(?, model),"
                " degradation_notes = COALESCE(?, degradation_notes),"
                " completed_at = COALESCE(?, completed_at),"
                " story_count = COALESCE(?, story_count)"
                " WHERE id = ?",
                (status, provider, model, degradation_notes, completed_at,
                 story_count, collection_id),
            )

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
            matched_topics: Profile topics the story matched (stored JSON).
            query: The search query that surfaced the candidate.
            error: Failure reason for ``failed`` rows, or None.

        Returns:
            The new story id.

        Raises:
            sqlite3.IntegrityError: If the URL already exists in the
                collection.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO dream_stories"
                " (collection_id, title, url, snippet, body, status, source,"
                " kind, event_date, location, matched_topics, query, kept,"
                " error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    collection_id, title, url, snippet, body, status, source,
                    kind, event_date, location, json.dumps(matched_topics),
                    query, error, _utc_now_iso(),
                ),
            )
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
                ``personal_context``/``notes``/``media``).
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
                " created_by_dreams, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                (
                    origin_story_id, mechanism, intent, subscription_id,
                    query_template, event_date, cadence_seconds,
                    created_by_dreams, now, now,
                ),
            )
            return int(cursor.lastrowid)

    def get_tracked_item(self, tracked_item_id: int) -> dict | None:
        """Return one tracked item row by id, or None."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE id = ?",
                (tracked_item_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_tracked_items(self, status: str = "active") -> list[dict]:
        """Return tracked items in one status, newest first.

        Args:
            status: Lifecycle filter (``active``/``paused``/``retired``).

        Returns:
            Matching rows ordered ``created_at DESC`` (``id DESC`` breaks
            ties so same-millisecond inserts keep newest-first order).
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE status = ?"
                " ORDER BY created_at DESC, id DESC",
                (status,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_tracked_by_story(self, origin_story_id: int) -> dict | None:
        """Return the newest tracked item originating from a story, or None."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM dream_tracked_items WHERE origin_story_id = ?"
                " ORDER BY created_at DESC, id DESC LIMIT 1",
                (origin_story_id,),
            ).fetchone()
        return dict(row) if row is not None else None

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
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE dream_tracked_items SET"
                " last_checked = ?, updated_at = ?"
                " WHERE id = ?",
                (now_iso, _utc_now_iso(), tracked_item_id),
            )

    def count_active_tracked(self) -> int:
        """Return how many tracked items are currently active."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM dream_tracked_items WHERE status = 'active'"
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
