"""SQLite persistence for agent run records (primary + sub-agent).

Follows the Workspace_DB pattern (task-3011 form): BaseDB with a
thread-local held connection — the earlier per-call shape paid full
private-SQLite connection setup on every read/write, per agent step —
and ``transaction()`` (BEGIN IMMEDIATE) for writes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Iterator, Union

from loguru import logger

from tldw_chatbook.Agents.agent_models import (
    AgentDefinition,
    TERMINAL_RUN_STATUSES,
    validate_agent_definition,
)
from .base_db import BaseDB


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class AgentRunsDB(BaseDB):
    """Run records for the agent runtime (vertical-slice spec data model).

    SCHEMA VERSIONING (task-15669 resolution, folded into the v11
    migration per the fleet PR3b coordinator ruling #3): this DB has no
    migration framework -- ``_initialize_schema`` runs CREATE TABLE IF NOT
    EXISTS plus guarded idempotent ALTERs, and appends one
    ``INSERT OR IGNORE INTO schema_version`` row per version. For years
    ``_CURRENT_SCHEMA_VERSION`` sat at 3 while the version table grew
    (4..10), because each migration followed the append pattern without
    touching the constant. From v11 on the CONTRACT is: the constant
    equals the HIGHEST version row a freshly created database records,
    and every new migration bumps BOTH (the constant here, and a new
    ``INSERT OR IGNORE`` row at the end of ``_initialize_schema``) --
    ``Tests/DB/test_agent_runs_db.py::test_schema_version_constant_
    agrees_with_the_version_table`` fails if they ever diverge again.
    An existing older file still opens unchanged: the guarded ALTERs are
    the effective migration, and the version table is a write-only audit
    trail (nothing branches on it at runtime).
    """

    _CURRENT_SCHEMA_VERSION = 11
    _swept_paths: set[str] = set()  # DB files already reconciled this process

    #: Liveness-ping gate (mirrors ChaChaNotes/WorkspaceDB, task-261/3011):
    #: a per-call ``SELECT 1`` would double statement count on the per-step
    #: persistence path; a recently-used held connection is known-good.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        self._thread_local = threading.local()
        super().__init__(db_path, client_id)
        # After super().__init__: the agent_runs table exists (base_db ran
        # _initialize_schema) and self.is_memory_db is set. Reconcile once per
        # file per process so a crash mid-run doesn't leave a 'running' row
        # orphaned forever. reconcile_orphaned_runs() itself guards against
        # memory DBs and against re-sweeping a path already swept this
        # process, so a later explicit call is also a no-op (see its
        # docstring).
        try:
            self.reconcile_orphaned_runs()
        except Exception as exc:  # noqa: BLE001 — reconcile is best-effort
            logger.warning(f"AgentRunsDB reconcile skipped: {exc}")

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        # busy_timeout FIRST: the journal_mode=WAL conversion below is the
        # one PRAGMA here that can itself contend (switching a rollback-
        # journal file to WAL briefly needs an exclusive lock), so it must
        # not run while busy_timeout is still 0 -- a contended cross-process
        # first conversion would otherwise raise 'database is locked'
        # immediately instead of waiting. busy_timeout is harmless to set
        # for in-memory DBs too, so it's unconditional (kept for
        # uniformity); WAL itself is unavailable for in-memory DBs, so that
        # one stays guarded on is_memory_db.
        conn.execute("PRAGMA busy_timeout = 5000")
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit or two, acceptable for this local agent-run
        # ledger) and avoids an fsync on every commit -- the default FULL was
        # fsyncing the WAL on every commit despite WAL already being enabled,
        # on a per-agent-step persistence path. See Library_Ingest_Jobs_DB.py:
        # 57-61 for the original template (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        # task-3012: the held (long-lived) connection needs true autocommit.
        # Python's default isolation mode auto-BEGINs on any DML, and an
        # implicit transaction accumulated outside `transaction()` makes the
        # explicit `BEGIN IMMEDIATE` there fail with "cannot start a
        # transaction within a transaction" (per-call connections masked
        # this — and silently ROLLED BACK any bare DML on close). Audited:
        # every `connection()` site is read-only except `_initialize_schema`,
        # whose `executescript` self-commits under either mode.
        conn.isolation_level = None
        return conn

    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held connection, opening or reviving it.

        task-3012: mirrors ``WorkspaceDB._held_connection`` (itself the
        ChaChaNotes idiom). Every per-connection property this DB relies on
        — WAL, busy_timeout, foreign keys, row factory — is applied by
        ``_get_connection`` when the held connection is (re)opened.
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
        """Yield the calling thread's held read connection.

        Yields:
            The thread's held ``sqlite3.Connection`` (per-thread isolation
            replaces the old per-call isolation; WAL keeps concurrent
            reader/writer threads and processes non-blocking).
        """
        yield self._held_connection()

    def close(self) -> None:
        """Close the current thread's held connection, if any."""

        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a write connection inside an immediate transaction.

        Uses ``BEGIN IMMEDIATE`` (not the deferred default ``BEGIN``) so
        the write lock is acquired up front: with multiple workers writing
        concurrently (e.g. a primary run and its sub-agent runs), a
        deferred transaction that reads then writes can hit a lock-upgrade
        conflict between two readers; acquiring the write lock immediately
        avoids that class of failure.

        Yields:
            A ``sqlite3.Connection`` with a write transaction already
            started.

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

    def _initialize_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (4);

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    parent_run_id TEXT,
                    agent_kind TEXT NOT NULL,
                    task TEXT,
                    status TEXT NOT NULL,
                    steps TEXT NOT NULL DEFAULT '[]',
                    result TEXT,
                    budget TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    assistant_message_id TEXT,
                    agent_definition TEXT,
                    definition_fingerprint TEXT,
                    wake_delivered_at TEXT,
                    -- v11 (fleet PR3b Task 4, spec SS6): when this run is
                    -- a CONTINUATION of a finished sub-agent -- a NEW run
                    -- seeded from the old one's retained in-memory
                    -- transcript via send_to_agent -- this records the
                    -- run it resumed from. NULL for every ordinary run.
                    -- Lineage only: parent_run_id still points at the
                    -- RESUMING turn's primary, never at the old run.
                    resumed_from_run_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
                    ON agent_runs(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_parent
                    ON agent_runs(parent_run_id);

                -- v3 (TASK-1971, Agent Change Review): one row per
                -- (run, root) pair recording that turn's shadow-repo
                -- baseline/end snapshots. CREATE IF NOT EXISTS on every
                -- open IS this DB's migration mechanism (see the v1->v2
                -- note below).
                CREATE TABLE IF NOT EXISTS change_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    root TEXT NOT NULL,
                    baseline_sha TEXT NOT NULL,
                    end_sha TEXT NOT NULL,
                    files_changed INTEGER NOT NULL DEFAULT 0,
                    adds INTEGER NOT NULL DEFAULT 0,
                    dels INTEGER NOT NULL DEFAULT 0,
                    reverted TEXT NOT NULL DEFAULT '',
                    tracking_error TEXT NOT NULL DEFAULT '',
                    untracked_oversize INTEGER NOT NULL DEFAULT 0,
                    nested_repos TEXT NOT NULL DEFAULT '[]',
                    -- v5->v6 (PR3a-1 Task 6c): which WINDOW this row
                    -- covers. 'turn' is the turn's own B/E window;
                    -- 'turn_concurrent_subagent' is the same window taken
                    -- while a sub-agent from an EARLIER turn was writing
                    -- (so the diff may include changes this turn's agent
                    -- did not make -- disclosed, never implied);
                    -- 'subagent_post_turn' is the window AFTER a turn's E
                    -- during which its surviving sub-agents were still
                    -- working. The default keeps every pre-existing row
                    -- reading as what it was.
                    kind TEXT NOT NULL DEFAULT 'turn',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_change_snapshots_run
                    ON change_snapshots(run_id);

                -- v5 (fleet spec §4, PR 1): user-authored agent
                -- definitions. DURABILITY NOTE: from v5 on this DB holds
                -- durable USER-AUTHORED CONTENT, not just run telemetry --
                -- any future "clear run history" feature must NOT treat
                -- the file as disposable.
                CREATE TABLE IF NOT EXISTS agent_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    instructions TEXT NOT NULL DEFAULT '',
                    tool_allowlist TEXT NOT NULL DEFAULT '[]',
                    model TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                -- Partial unique index: a live name is unique, but a
                -- soft-deleted row releases its name for re-creation.
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_definitions_name
                    ON agent_definitions(name) WHERE deleted = 0;

                -- v8 (TASK-16800 annotate loop, spec §1): user-authored
                -- feedback notes anchored to a specific hunk of a turn's
                -- diff. DURABILITY NOTE (carries the v5 note forward):
                -- this DB holds durable USER-AUTHORED CONTENT -- notes
                -- extend that -- any "clear run history" tooling must not
                -- treat this table as disposable. No denormalized
                -- conversation_id: agent_runs already carries
                -- conversation_id NOT NULL, so pending_notes_for_conversation
                -- joins through change_notes.run_id = agent_runs.id --
                -- one source of truth, and the card never needs to learn
                -- the conversation id at insert time.
                CREATE TABLE IF NOT EXISTS change_notes (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    root TEXT NOT NULL,
                    path TEXT NOT NULL,
                    hunk_index INTEGER NOT NULL,
                    hunk_header TEXT NOT NULL,
                    hunk_excerpt TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    -- v9 (TASK-16800 Task 6 fix round): which run's
                    -- COMPLETION actually stamped this note delivered --
                    -- distinct from `run_id` above, which anchors the note
                    -- to the run whose DIFF it critiques. A note written
                    -- against an earlier run's diff is commonly delivered
                    -- on a LATER run's completion, so resume re-derivation
                    -- needs this to place the disclosure row at the
                    -- delivering run's position (matching live emission)
                    -- instead of fragmenting/mis-anchoring at the
                    -- annotated run. NULL until stamped, and NULL forever
                    -- on rows stamped by code that predates this column.
                    delivered_by_run_id TEXT,
                    -- v10 (Qodo #6, PR #1779 fix round): the exact
                    -- change_snapshots row (its own DB `id`) this note's
                    -- hunk was read from. Two windows on the SAME run+
                    -- root+path (a turn's own window and its surviving
                    -- sub-agents' post-turn window, PR3a-1 Task 6c) can
                    -- theoretically carry the same hunk_header at the
                    -- same hunk_index (identical position/line-count
                    -- edits against different baselines produce
                    -- byte-identical "@@ ... @@" headers, since a header
                    -- encodes only positions/counts, never content) --
                    -- (run_id, hunk_index, hunk_header) alone can't then
                    -- say which window's diff was actually annotated.
                    -- NULL for legacy rows saved before this column
                    -- existed; the card falls back to hunk_index+
                    -- hunk_header matching for those.
                    snapshot_id INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_change_notes_pending
                    ON change_notes(run_id) WHERE delivered_at IS NULL;
                """
            )
            # v1->v2: this DB has no migration framework -- _initialize_schema
            # only runs CREATE TABLE IF NOT EXISTS, so a file created before
            # assistant_message_id existed keeps its old 11-column table and
            # never picks up the new one from the DDL above. Guard against
            # that with an idempotent ALTER TABLE, run on every open.
            existing_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(agent_runs)").fetchall()
            }
            if "assistant_message_id" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN assistant_message_id TEXT"
                )
            # v4->v5 (fleet spec §4): definition audit identity on runs --
            # same idempotent-ALTER mechanism as above.
            if "agent_definition" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN agent_definition TEXT"
                )
            if "definition_fingerprint" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN definition_fingerprint TEXT"
                )
            # v6->v7 (PR3a-2 Task 5, auto-wake): the per-run delivered
            # ledger -- the UTC instant a wake turn carried this run's
            # result to its supervisor, NULL while undelivered. Same
            # idempotent-ALTER mechanism as above. Durable ON the run row
            # (not in any in-memory registry) because exactly-once delivery
            # must survive both screen teardown and an app restart; the
            # conversation-level FLEET_UNSEEN mark cannot carry per-run
            # identity, and a drain can mix children settled minutes apart
            # (so no timestamp rule against the mark can recover which runs
            # a wake already delivered).
            if "wake_delivered_at" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN wake_delivered_at TEXT"
                )
            # v10->v11 (fleet PR3b Task 4): continuation lineage -- the
            # run a resumed sub-agent was seeded from. Same idempotent-
            # ALTER mechanism as every column above; NULL (no DEFAULT) is
            # exactly right for every pre-existing row (an ordinary,
            # non-resumed run).
            if "resumed_from_run_id" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN resumed_from_run_id TEXT"
                )
            # v3->v4 (TASK-1975): oversize disclosure count on snapshot
            # rows -- same idempotent-ALTER migration mechanism as above.
            snapshot_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(change_snapshots)"
                ).fetchall()
            }
            if "untracked_oversize" not in snapshot_columns:
                conn.execute(
                    "ALTER TABLE change_snapshots ADD COLUMN "
                    "untracked_oversize INTEGER NOT NULL DEFAULT 0"
                )
            if "nested_repos" not in snapshot_columns:
                conn.execute(
                    "ALTER TABLE change_snapshots ADD COLUMN "
                    "nested_repos TEXT NOT NULL DEFAULT '[]'"
                )
            # v5->v6 (PR3a-1 Task 6c): the window a row covers -- same
            # idempotent-ALTER mechanism. The DEFAULT is what makes every
            # row written before this column existed read as 'turn'.
            if "kind" not in snapshot_columns:
                conn.execute(
                    "ALTER TABLE change_snapshots ADD COLUMN "
                    "kind TEXT NOT NULL DEFAULT 'turn'"
                )
            # v8->v9 (TASK-16800 Task 6 fix round): a file created while
            # change_notes existed but before delivered_by_run_id did
            # keeps its old 10-column table -- same idempotent-ALTER
            # mechanism as every column above. No DEFAULT: every
            # pre-existing row correctly reads as NULL (unknown delivering
            # run), which is exactly the legacy fallback resume
            # re-derivation is built to handle.
            note_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(change_notes)"
                ).fetchall()
            }
            if "delivered_by_run_id" not in note_columns:
                conn.execute(
                    "ALTER TABLE change_notes ADD COLUMN delivered_by_run_id TEXT"
                )
            # v9->v10 (Qodo #6, PR #1779 fix round): a file created while
            # change_notes existed but before snapshot_id did keeps its
            # old 11-column table -- same idempotent-ALTER mechanism as
            # every column above. No DEFAULT: every pre-existing row
            # correctly reads as NULL (unknown snapshot), which is exactly
            # the legacy hunk_index+hunk_header fallback the card's
            # matching is built to handle.
            if "snapshot_id" not in note_columns:
                conn.execute(
                    "ALTER TABLE change_notes ADD COLUMN snapshot_id INTEGER"
                )
            # Keep the (write-only, audit) version table in step with the
            # DDL -- append-per-version, matching the INSERT OR IGNORE
            # convention above (UPDATE would collide on the UNIQUE column
            # when older version rows exist).
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (4)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (5)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (6)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (7)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (8)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (9)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (10)"
            )
            # v11 is ALSO where _CURRENT_SCHEMA_VERSION was re-synced to
            # the version table (task-15669; see the class docstring for
            # the from-now-on contract).
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (11)"
            )

    def record_change_snapshot(
        self,
        *,
        run_id: str,
        root: str,
        baseline_sha: str,
        end_sha: str,
        files_changed: int = 0,
        adds: int = 0,
        dels: int = 0,
        tracking_error: str = "",
        untracked_oversize: int = 0,
        nested_repos: "Sequence[str]" = (),
        kind: str = "turn",
    ) -> None:
        """Record one root's turn snapshot pair (TASK-1971).

        Args:
            run_id: The owning agent run.
            root: Canonical root path.
            baseline_sha: The B snapshot tip ("" when tracking failed).
            end_sha: The E snapshot tip ("" when tracking failed).
            files_changed: Changed-file count between B and E.
            adds: Total added lines.
            dels: Total deleted lines.
            tracking_error: Non-empty when tracking failed for this root.
            untracked_oversize: Files over the size cap left untracked at
                the turn's end (TASK-1975 disclosure).
            nested_repos: Root-relative nested repos excluded from tracking
                (TASK-1976 disclosure).
            kind: Which window this row covers — ``"turn"``,
                ``"turn_concurrent_subagent"`` (a turn whose window
                overlapped an earlier turn's still-running sub-agent) or
                ``"subagent_post_turn"`` (the window after a turn's E,
                while its survivors kept working). PR3a-1 Task 6c.
        """
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO change_snapshots
                    (run_id, root, baseline_sha, end_sha, files_changed,
                     adds, dels, tracking_error, untracked_oversize,
                     nested_repos, kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    root,
                    baseline_sha,
                    end_sha,
                    files_changed,
                    adds,
                    dels,
                    tracking_error,
                    untracked_oversize,
                    json.dumps(list(nested_repos)),
                    kind,
                    _now_iso(),
                ),
            )

    def delete_change_snapshots_older_than(self, cutoff_iso: str) -> int:
        """Delete snapshot rows created before ``cutoff_iso`` (TASK-1975).

        Args:
            cutoff_iso: ISO-8601 UTC timestamp in this DB's own format
                (lexicographic compare is valid for it).

        Returns:
            Number of rows deleted.
        """
        with self.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM change_snapshots WHERE created_at < ?",
                (cutoff_iso,),
            )
            return int(cur.rowcount or 0)

    def roots_with_change_snapshots(self) -> set[str]:
        """Roots still referenced by at least one snapshot row (TASK-1975).

        Returns:
            The distinct ``root`` values across all remaining rows.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT root FROM change_snapshots"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def update_change_snapshot_reverted(
        self, row_id: int, reverted_paths: list[str]
    ) -> None:
        """Record which of a snapshot row's paths were reverted (TASK-1974).

        Args:
            row_id: The ``change_snapshots`` row id.
            reverted_paths: Paths restored to baseline; appended to any
                previously recorded set (a second partial revert must not
                erase the first's record).
        """
        with self.transaction() as conn:
            current = conn.execute(
                "SELECT reverted FROM change_snapshots WHERE id = ?",
                (row_id,),
            ).fetchone()
            existing = json.loads(current["reverted"]) if current and current["reverted"] else []
            merged = list(dict.fromkeys([*existing, *reverted_paths]))
            conn.execute(
                "UPDATE change_snapshots SET reverted = ? WHERE id = ?",
                (json.dumps(merged), row_id),
            )

    def change_snapshots_for_run(self, run_id: str) -> list[dict]:
        """Return a run's change-snapshot rows, oldest first.

        Args:
            run_id: The agent run id.

        Returns:
            One dict per (run, root) row.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM change_snapshots WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def change_snapshots_for_run_review(self, run_id: str) -> list[dict]:
        """Return one run's change-snapshot rows in the review row shape.

        Identical joined shape to :meth:`change_snapshots_for_conversation`
        (``run_created_at``/``run_status`` included) so the change-review
        provider can build a single run's ``ReviewTurn`` without scanning
        the whole conversation's history (Qodo, PR #1728) — and without
        the two paths ever diverging on row shape. Distinct from
        :meth:`change_snapshots_for_run`, whose bare-row shape existing
        revert/tracking callers depend on.

        Args:
            run_id: The agent run id.

        Returns:
            The run's rows joined with their run, oldest first.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cs.*, ar.created_at AS run_created_at, ar.status AS run_status
                FROM change_snapshots cs
                JOIN agent_runs ar ON ar.id = cs.run_id
                WHERE cs.run_id = ?
                ORDER BY cs.id
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def change_snapshots_for_conversation(self, conversation_id: str) -> list[dict]:
        """Return a conversation's change-snapshot rows for turn history.

        Args:
            conversation_id: The Console conversation id.

        Returns:
            Rows joined with their runs, oldest first — the Review screen's
            "Last turn" selector data.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cs.*, ar.created_at AS run_created_at, ar.status AS run_status
                FROM change_snapshots cs
                JOIN agent_runs ar ON ar.id = cs.run_id
                WHERE ar.conversation_id = ?
                ORDER BY cs.id
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_change_note(
        self,
        *,
        run_id: str,
        root: str,
        path: str,
        hunk_index: int,
        hunk_header: str,
        hunk_excerpt: str,
        note: str,
        snapshot_id: int | None = None,
    ) -> int:
        """Record a user-authored note anchored to one hunk of a turn's diff.

        TASK-16800 (spec §1). The anchor is ``(run_id, root, path,
        hunk_index, hunk_header)``; ``hunk_excerpt`` is captured once, at
        note-creation time, from the full diff text the card already has
        -- it is the retention safety net that keeps display and delivery
        self-contained even after shadow-repo snapshot pruning.

        Args:
            run_id: The agent run whose diff this note is anchored to.
            root: Canonical root path of the changed file.
            path: The changed file's path (root-relative).
            hunk_index: 0-based index of the hunk over the FULL diff.
            hunk_header: The hunk's ``"@@ -a,b +c,d @@ ..."`` line, verbatim.
            hunk_excerpt: The hunk body captured at note time (already
                capped/elided by the caller).
            note: The user's note text.
            snapshot_id: The owning ``change_snapshots`` row's own DB
                ``id`` (Qodo #6, PR #1779 fix round) -- disambiguates
                which of TWO same-``run_id``/root/path windows (a turn's
                own window and its surviving sub-agents' post-turn
                window, PR3a-1 Task 6c) this note's hunk actually came
                from, for the rare case where both windows happen to
                produce the exact same ``hunk_index``/``hunk_header``
                (identical position/line-count edits against different
                baselines yield byte-identical headers, since a header
                encodes only positions/counts, never content). ``None``
                (the default) only by callers that predate this column or
                have no snapshot row to anchor to.

        Returns:
            The newly created note's row id.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO change_notes
                    (run_id, root, path, hunk_index, hunk_header,
                     hunk_excerpt, note, created_at, delivered_at,
                     snapshot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    run_id,
                    root,
                    path,
                    hunk_index,
                    hunk_header,
                    hunk_excerpt,
                    note,
                    _now_iso(),
                    snapshot_id,
                ),
            )
            return int(cursor.lastrowid)

    def delete_change_note(self, note_id: int) -> bool:
        """Delete a pending (undelivered) note.

        Delivered notes are record -- once ``delivered_at`` is set a note
        can no longer be deleted, matching the card's "delivered notes
        lose the delete affordance" rule.

        Args:
            note_id: The note's row id.

        Returns:
            True if a pending note was deleted; False if the note does
            not exist or has already been delivered.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM change_notes WHERE id = ? AND delivered_at IS NULL",
                (note_id,),
            )
            return cursor.rowcount > 0

    def notes_for_run(self, run_id: str) -> list[dict]:
        """Return a run's change notes, oldest first.

        Args:
            run_id: The agent run id.

        Returns:
            One dict per note row (all columns), oldest first.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM change_notes WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_notes_for_conversation(self, conversation_id: str) -> list[dict]:
        """Return a conversation's undelivered notes, oldest first.

        Joins through ``change_notes.run_id = agent_runs.id`` rather than
        a denormalized conversation id column (spec §1) -- ``agent_runs``
        is the one source of truth for which conversation a run belongs
        to, and notes span however many runs a conversation has had.

        Args:
            conversation_id: The Console conversation id.

        Returns:
            Pending (``delivered_at IS NULL``) note rows across every run
            of the conversation, oldest first.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cn.* FROM change_notes cn
                JOIN agent_runs ar ON ar.id = cn.run_id
                WHERE ar.conversation_id = ? AND cn.delivered_at IS NULL
                ORDER BY cn.id
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delivered_notes_for_conversation(self, conversation_id: str) -> list[dict]:
        """Return a conversation's delivered notes, oldest first.

        TASK-16800 Task 6 fix round: the batched counterpart to
        ``pending_notes_for_conversation`` (same join shape, same
        precedent as the TASK-1972 no-N+1 fix for ``change_snapshots``)
        -- resume re-derivation needs every delivered note across every
        run of the conversation in ONE query, not one ``notes_for_run``
        call per re-derived run.

        Args:
            conversation_id: The Console conversation id.

        Returns:
            Delivered (``delivered_at IS NOT NULL``) note rows across
            every run of the conversation, oldest first. Each row carries
            ``delivered_by_run_id`` -- NULL for notes stamped before that
            column existed.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cn.* FROM change_notes cn
                JOIN agent_runs ar ON ar.id = cn.run_id
                WHERE ar.conversation_id = ? AND cn.delivered_at IS NOT NULL
                ORDER BY cn.id
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_notes_delivered(
        self, note_ids: Sequence[int], delivered_by_run_id: str | None = None
    ) -> list[int]:
        """Stamp exactly the given notes as delivered.

        Spec §4: the delivery seam captures the precise id list it
        attached to the outbound message and stamps only that list at run
        completion -- never "all pending for the conversation". A note
        created after the list was captured (the mid-run race) is not in
        ``note_ids`` and so stays pending, riding the next send.

        Qodo #4 (PR #1779 fix round): the UPDATE's own
        ``AND delivered_at IS NULL`` guard means a note already stamped by
        a concurrent delivery (elsewhere) is silently skipped rather than
        re-stamped -- correct for the DB, but the caller previously had no
        way to know that skip happened, so the bridge's completion seam
        disclosed every id it captured regardless of what actually got
        stamped. Concurrent replies on one conversation are architecturally
        serialized today, so that race is not currently reachable in
        practice, but this makes the seam self-defending rather than
        relying on that invariant holding forever.

        Args:
            note_ids: The note ids to stamp delivered.
            delivered_by_run_id: The id of the run whose completion is
                doing this delivery (TASK-16800 Task 6 fix round) --
                distinct from each note's own ``run_id`` (the run whose
                diff it critiques). Stamped verbatim onto every row in
                ``note_ids``, alongside the same ``delivered_at``
                timestamp, so resume re-derivation can anchor the
                disclosure at the run that actually delivered it. Left
                ``None`` (the default) only by callers that predate this
                column or do not know/care which run is delivering.

        Returns:
            The subset of ``note_ids`` that this call ACTUALLY stamped
            (i.e. were still pending at the moment of the UPDATE) -- never
            more than ``note_ids``, and possibly fewer when a concurrent
            caller already delivered one of them first. Order is not
            significant to callers, all of which only ever test set
            membership against it.
        """
        ids = [int(note_id) for note_id in note_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        stamp = _now_iso()
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE change_notes SET delivered_at = ?, "
                f"delivered_by_run_id = ? "
                f"WHERE id IN ({placeholders}) AND delivered_at IS NULL",
                (stamp, delivered_by_run_id, *ids),
            )
            # Portable (works against any bundled SQLite -- no RETURNING
            # dependency): re-select, in the SAME transaction/connection,
            # exactly the rows this UPDATE just wrote. `delivered_at =
            # stamp` alone (a value only this call could have produced)
            # already uniquely identifies them; `delivered_by_run_id IS ?`
            # is belt-and-suspenders (SQLite's `IS` compares correctly
            # against a NULL-bound parameter too).
            rows = conn.execute(
                f"SELECT id FROM change_notes WHERE id IN ({placeholders}) "
                f"AND delivered_at = ? AND delivered_by_run_id IS ?",
                (*ids, stamp, delivered_by_run_id),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["steps"] = json.loads(record["steps"] or "[]")
        record["budget"] = json.loads(record["budget"]) if record["budget"] else None
        return record

    def create_run(
        self,
        *,
        conversation_id: str,
        agent_kind: str,
        task: str | None = None,
        parent_run_id: str | None = None,
        budget: dict | None = None,
        assistant_message_id: str | None = None,
        agent_definition: str | None = None,
        definition_fingerprint: str | None = None,
        resumed_from_run_id: str | None = None,
    ) -> str:
        """Create a new run record in ``running`` status.

        Args:
            conversation_id: The owning Console conversation's id.
            agent_kind: ``"primary"`` or ``"subagent"``.
            task: The sub-agent's task text; ``None`` for a primary run.
            parent_run_id: The parent run's id for a sub-agent; ``None``
                for a primary run.
            budget: The run's ``RunBudget`` serialized to a dict, stored
                as JSON for later inspection; ``None`` if not recorded.
            assistant_message_id: The persisted id of the assistant reply
                this run produced, if already known at creation time;
                ``None`` (the common case) when it will be recorded later
                via ``set_run_assistant_message_id`` once the reply is
                persisted.
            agent_definition: The name of the agent definition used for
                this run, if spawned from a definition; ``None`` otherwise.
            definition_fingerprint: The fingerprint hash of the agent
                definition used for this run, for audit trail purposes;
                ``None`` if not applicable.
            resumed_from_run_id: For a CONTINUATION of a finished
                sub-agent (fleet PR3b Task 4): the run id this run was
                seeded from. ``None`` for every ordinary run.

        Returns:
            The newly created run's id (a hex UUID4).
        """
        run_id = uuid.uuid4().hex
        now = _now_iso()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (id, conversation_id, parent_run_id, agent_kind, task,
                    status, steps, result, budget, created_at, updated_at,
                    assistant_message_id, agent_definition, definition_fingerprint,
                    resumed_from_run_id)
                   VALUES (?, ?, ?, ?, ?, 'running', '[]', NULL, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    conversation_id,
                    parent_run_id,
                    agent_kind,
                    task,
                    json.dumps(budget) if budget is not None else None,
                    now,
                    now,
                    assistant_message_id,
                    agent_definition,
                    definition_fingerprint,
                    resumed_from_run_id,
                ),
            )
        return run_id

    def create_agent_definition(self, defn: AgentDefinition) -> str:
        """Insert a definition; returns its id.

        Raises:
            ValueError: On validation failure, or a duplicate live name.
        """
        errors = validate_agent_definition(defn)
        if errors:
            raise ValueError("; ".join(errors))
        definition_id = uuid.uuid4().hex
        now = _now_iso()
        try:
            with self.transaction() as conn:
                conn.execute(
                    """INSERT INTO agent_definitions
                       (id, name, description, instructions, tool_allowlist,
                        model, enabled, deleted, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                    (
                        definition_id,
                        defn.name,
                        defn.description,
                        defn.instructions,
                        json.dumps(list(defn.tool_allowlist)),
                        defn.model,
                        1 if defn.enabled else 0,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"an agent named '{defn.name}' already exists"
            ) from exc
        return definition_id

    def update_agent_definition(
        self, definition_id: str, defn: AgentDefinition
    ) -> None:
        """Replace a definition's fields (same raises as create).

        Raises:
            ValueError: On validation failure, a duplicate live name, or
                when ``definition_id`` doesn't match any live (non-deleted)
                row -- without this check a missing/soft-deleted id was a
                silent no-op and the caller (Settings ▸ Agents) would still
                report "Saved".
        """
        errors = validate_agent_definition(defn)
        if errors:
            raise ValueError("; ".join(errors))
        try:
            with self.transaction() as conn:
                cursor = conn.execute(
                    """UPDATE agent_definitions
                       SET name = ?, description = ?, instructions = ?,
                           tool_allowlist = ?, model = ?, enabled = ?,
                           updated_at = ?
                       WHERE id = ? AND deleted = 0""",
                    (
                        defn.name,
                        defn.description,
                        defn.instructions,
                        json.dumps(list(defn.tool_allowlist)),
                        defn.model,
                        1 if defn.enabled else 0,
                        _now_iso(),
                        definition_id,
                    ),
                )
                if cursor.rowcount == 0:
                    raise ValueError(
                        f"agent definition not found: {definition_id}"
                    )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"an agent named '{defn.name}' already exists"
            ) from exc

    def soft_delete_agent_definition(self, definition_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE agent_definitions SET deleted = 1, updated_at = ? "
                "WHERE id = ?",
                (_now_iso(), definition_id),
            )

    def _definition_row_to_dict(self, row: sqlite3.Row) -> dict:
        data = {key: row[key] for key in row.keys()}
        data["tool_allowlist"] = json.loads(data["tool_allowlist"] or "[]")
        data.pop("deleted", None)
        return data

    def list_agent_definitions(self, enabled_only: bool = False) -> list[dict]:
        """Live (non-deleted) definitions ordered by name."""
        query = "SELECT * FROM agent_definitions WHERE deleted = 0"
        if enabled_only:
            query += " AND enabled = 1"
        query += " ORDER BY name"
        with self.connection() as conn:
            rows = conn.execute(query).fetchall()
        return [self._definition_row_to_dict(row) for row in rows]

    def get_agent_definition(self, definition_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_definitions WHERE id = ? AND deleted = 0",
                (definition_id,),
            ).fetchone()
        return self._definition_row_to_dict(row) if row else None

    def append_steps(self, run_id: str, steps: list[dict]) -> None:
        """Append step records to a run's step log.

        Args:
            run_id: The run to append to.
            steps: Serialized ``AgentStep`` dicts, appended in order after
                any steps already recorded.

        Raises:
            KeyError: If ``run_id`` does not exist.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT steps FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run id: {run_id}")
            existing = json.loads(row["steps"] or "[]")
            existing.extend(steps)
            conn.execute(
                "UPDATE agent_runs SET steps = ?, updated_at = ? WHERE id = ?",
                (json.dumps(existing), _now_iso(), run_id),
            )

    def set_status(self, run_id: str, status: str, result: str | None = None) -> bool:
        """Update a run's terminal (or in-progress) status.

        A run already in a terminal status is never rewritten (first-writer-wins),
        because an abandoned child thread can persist after the coordinator recorded
        a terminal status, and we must not overwrite that with a late update.

        Args:
            run_id: The run to update.
            status: The new status (e.g. ``"done"``, ``"stuck"``,
                ``"error"``, ``"cancelled"``, ``"superseded"``).
            result: The final answer text (primary) or sub-agent result
                text; when ``None`` the existing ``result`` column is left
                unchanged (``COALESCE``), so a status-only update never
                clobbers a previously recorded result.

        Returns:
            True if the run was updated (a row changed), False if the run is
            already terminal or does not exist.
        """
        placeholders = ",".join("?" for _ in TERMINAL_RUN_STATUSES)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE agent_runs SET status = ?, "
                "result = COALESCE(?, result), updated_at = ? "
                f"WHERE id = ? AND status NOT IN ({placeholders})",
                (status, result, _now_iso(), run_id, *sorted(TERMINAL_RUN_STATUSES)),
            )
        return cursor.rowcount > 0

    def reconcile_orphaned_runs(self) -> int:
        """Mark runs left ``running`` by a crashed process as ``error``.

        A hard crash between run start (``create_run`` -> ``running``) and run
        end (``set_status`` at finalize) leaves a row stuck ``running``
        forever. On open, flip all such rows to ``error`` with a default
        ``result`` (preserving any partial result via COALESCE). Assumes a
        single app instance per data dir: a second instance sharing the file
        would flip the first's actively-running run — an accepted edge case,
        matching Library_Ingest_Jobs' "Interrupted by app restart" behavior.

        No-ops (returns ``0`` without touching the database) for in-memory
        databases and for any file path already reconciled once in this
        process (tracked via ``_swept_paths``). The guard lives here, not
        just in ``__init__``'s auto-call, so a later *explicit* call to this
        method within the same process is also a no-op -- it must not sweep
        up a run that legitimately started running after the first sweep
        (e.g. one created by this same still-live process).

        The path is registered in ``_swept_paths`` only *after* the sweep's
        transaction has committed successfully (i.e. after the ``with
        self.transaction()`` block below exits normally). ``transaction()``
        rolls back and re-raises on any error -- e.g. a transient
        ``sqlite3.OperationalError: database is locked`` -- so registering
        beforehand would leave the path permanently marked "swept" even
        though nothing was actually reconciled, silently defeating AC#2's
        crash-recovery guarantee for the rest of the process. A clean sweep
        that finds zero orphaned rows still registers the path (it commits
        successfully; it just has nothing to update).

        Timing note: despite this being framed as an "on app start" sweep
        (see the backlog AC), it actually fires lazily -- the first time
        something constructs an ``AgentRunsDB`` on this path, which today is
        ``ChatScreen._ensure_console_agent_bridge()``'s first call, not app
        boot. No surface can currently observe a stale ``running`` row
        before that: the only readers of ``agent_runs.db`` are that bridge
        itself and the chat-screen rail's sub-agent-count summary, which
        also routes through ``_ensure_console_agent_bridge()`` first. A
        future entry point that opens this DB file WITHOUT going through
        that bridge construction path would not inherit this guarantee and
        could observe a not-yet-reconciled orphaned row.

        Returns:
            The number of rows reconciled (``0`` if skipped by a guard).

        Raises:
            Exception: Re-raised (from ``transaction()``) on any error
                while sweeping; the path is left unregistered so a later
                call in this process retries the sweep.
        """
        if self.is_memory_db or self.db_path_str in self._swept_paths:
            return 0
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE agent_runs "
                "SET status = 'error', "
                "    result = COALESCE(result, 'Interrupted by app restart'), "
                "    updated_at = ? "
                "WHERE status = 'running'",
                (_now_iso(),),
            )
            rowcount = cur.rowcount
        self._swept_paths.add(self.db_path_str)
        return rowcount

    def set_run_assistant_message_id(
        self, run_id: str, assistant_message_id: str | None
    ) -> None:
        """Record the persisted id of the assistant reply a run produced.

        Args:
            run_id: The run to update.
            assistant_message_id: The persisted message id of the
                assistant reply this run produced; ``None`` clears a
                previously recorded id.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE agent_runs SET assistant_message_id = ?, "
                "updated_at = ? WHERE id = ?",
                (assistant_message_id, _now_iso(), run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        """Fetch one run record.

        Args:
            run_id: The run to fetch.

        Returns:
            The run as a dict (``steps``/``budget`` JSON-decoded), or
            ``None`` if ``run_id`` does not exist.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_run_fresh(self, run_id: str) -> dict | None:
        """Fetch one run through a dedicated, immediately-closed connection.

        task-15863: the per-thread held connection (``_held_connection``)
        is a WAL reader, and in Python's ``sqlite3`` ANY unfinalized
        statement on a connection holds its implicit read transaction
        open -- pinning that connection's snapshot. Every later read on
        the same thread then reports the world as of the pin: live
        verification caught the auto-wake notice labelling a child
        ``running`` a full minute after its terminal ``done`` committed.
        This escape hatch reads through a brand-new connection that
        cannot inherit any pinned snapshot; callers use it when a held
        read returns a state the caller can PROVE stale (the wake path's
        rule: a settled child's row can never legitimately read
        non-terminal).

        In-memory databases fall back to the held read: a second
        connection to ``:memory:`` opens a different, empty database.

        Args:
            run_id: The run to fetch.

        Returns:
            The run as a dict (``steps``/``budget`` JSON-decoded), or
            ``None`` if ``run_id`` does not exist.
        """
        if self.is_memory_db:
            return self.get_run(run_id)
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def latest_primary_run(self, conversation_id: str) -> dict | None:
        """Fetch the newest non-superseded PRIMARY run for a conversation.

        A single bounded query (``LIMIT 1``) so hot callers (the user-Stop
        path's run-anchor lookup) never materialize the whole run history;
        interleaved newer sub-agent runs cannot hide the newest primary
        because the kind filter is in the SQL.

        Args:
            conversation_id: The conversation whose runs to inspect.

        Returns:
            The newest matching run as a dict (``steps``/``budget``
            JSON-decoded), or ``None`` when the conversation has no
            non-superseded primary run.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE conversation_id = ? "
                "AND agent_kind = 'primary' AND status != 'superseded' "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(
        self,
        conversation_id: str,
        include_superseded: bool = True,
        limit: int | None = None,
        agent_kind: str | None = None,
    ) -> list[dict]:
        """List a conversation's run records, newest first.

        Args:
            conversation_id: The conversation to list runs for.
            include_superseded: When ``False``, excludes runs whose
                status is ``"superseded"``.
            limit: When set, caps the result to the ``limit`` most recent
                runs (``ORDER BY created_at DESC, id DESC``). ``None``
                (the default) returns every matching run, preserving prior
                behavior.
            agent_kind: When set (``"primary"`` or ``"subagent"``),
                restricts to that kind IN THE QUERY -- e.g.
                ``search_run_log``'s ``scope="conversation"`` (task-1273
                review finding A) wants only the conversation's PRIMARY
                runs, and filtering here (rather than fetching everything
                and discarding client-side) is what lets ``limit`` bound
                the actual number of ROWS RETURNED to what the caller can
                use, instead of a limit over an unfiltered set that a
                subagent-heavy conversation could still starve. ``None``
                (the default) applies no kind filter, preserving prior
                behavior.

        Returns:
            The matching runs as dicts, newest first.
        """
        query = "SELECT * FROM agent_runs WHERE conversation_id = ?"
        params: list = [conversation_id]
        if not include_superseded:
            query += " AND status != 'superseded'"
        if agent_kind is not None:
            query += " AND agent_kind = ?"
            params.append(agent_kind)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_runs(
        self,
        conversation_id: str,
        include_superseded: bool = True,
        agent_kind: str | None = None,
    ) -> int:
        """Count a conversation's run records, without materializing rows.

        task-1273 review finding A: a caller that only needs to know HOW
        MANY runs exist beyond a windowed ``list_runs(..., limit=N)`` call
        (to report an exact "N more exist" count) must not fetch every row
        just to take its length -- that is exactly the unbounded query the
        finding flagged. ``COUNT(*)`` returns a single row regardless of
        how many runs match, so pairing this with a ``limit``-bounded
        ``list_runs`` call keeps BOTH queries' returned size independent of
        the conversation's total run count.

        Args:
            conversation_id: The conversation to count runs for.
            include_superseded: When ``False``, excludes runs whose status
                is ``"superseded"`` -- mirrors ``list_runs``' own filter.
            agent_kind: When set (``"primary"`` or ``"subagent"``),
                restricts the count to that kind -- mirrors ``list_runs``'
                own filter. ``None`` (the default) counts every kind.

        Returns:
            The number of matching runs.
        """
        query = "SELECT COUNT(*) AS n FROM agent_runs WHERE conversation_id = ?"
        params: list = [conversation_id]
        if not include_superseded:
            query += " AND status != 'superseded'"
        if agent_kind is not None:
            query += " AND agent_kind = ?"
            params.append(agent_kind)
        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["n"])

    def undelivered_wake_runs(self, conversation_id: str) -> list[dict]:
        """Sub-agent SURVIVOR runs whose result no wake has delivered yet.

        PR3a-2 Task 5 (auto-wake). The durable definition of "owed to the
        supervisor", composed entirely from run rows so it survives screen
        teardown and app restart:

        - a sub-agent run of ``conversation_id`` in a real terminal state
          (``done``/``error``/``cancelled`` -- never ``superseded``: a
          superseded row is retracted work, delivering it would announce a
          result a retry already replaced);
        - whose ``wake_delivered_at`` ledger column is still NULL;
        - whose parent run is itself terminal AND terminal-stamped no later
          than the child (``child.updated_at >= parent.updated_at``): a
          child that settled BEFORE its parent's turn ended was collected
          in-turn by ``wait_agents``/the end-of-turn net and is the turn's
          own news, never a wake's. A survivor by construction settles
          after its parent's terminal write. The comparison is on ISO-8601
          UTC strings of one fixed format (``_now_iso``), where
          lexicographic order IS chronological order. ``>=`` (not ``>``)
          so a restart-reconcile sweep that stamps an orphaned child and
          its parent in the same pass still reports the child.

        Rows come back oldest-settled first, so a coalesced wake notice
        reads in the order the children actually finished.

        Args:
            conversation_id: The bridge's durable conversation id.

        Returns:
            Matching run records (``steps``/``budget`` JSON-decoded),
            oldest ``updated_at`` first.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT child.* FROM agent_runs AS child "
                "JOIN agent_runs AS parent ON parent.id = child.parent_run_id "
                "WHERE child.conversation_id = ? "
                "AND child.agent_kind != 'primary' "
                "AND child.wake_delivered_at IS NULL "
                "AND child.status IN ('done', 'error', 'cancelled') "
                f"AND parent.status IN ({', '.join('?' for _ in TERMINAL_RUN_STATUSES)}) "
                "AND child.updated_at >= parent.updated_at "
                "ORDER BY child.updated_at ASC, child.id ASC",
                (conversation_id, *sorted(TERMINAL_RUN_STATUSES)),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def mark_wake_delivered(self, run_ids: Sequence[str]) -> int:
        """Stamp runs as wake-delivered; already-stamped rows are left alone.

        PR3a-2 Task 5. Called by the wake coordinator ONLY after the wake
        turn was actually accepted (never at compose/schedule time -- a
        refused wake must leave every run undelivered so the retry still
        finds it). First-writer-wins per row (``wake_delivered_at IS
        NULL`` in the WHERE), so a duplicate stamp -- e.g. the immediate
        path and a mount claim racing -- can never move an existing
        delivery timestamp. ``updated_at`` is deliberately NOT bumped:
        that column records the run's own lifecycle (terminal time), which
        :meth:`undelivered_wake_runs`' survivor comparison depends on.

        Args:
            run_ids: The run ids the accepted wake actually carried.

        Returns:
            How many rows were newly stamped.
        """
        ids = [str(run_id) for run_id in run_ids if run_id]
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE agent_runs SET wake_delivered_at = ? "
                f"WHERE id IN ({placeholders}) AND wake_delivered_at IS NULL",
                (_now_iso(), *ids),
            )
        return int(cursor.rowcount or 0)

    def count_subagent_runs(self, conversation_id: str) -> int:
        """Count a conversation's sub-agent runs (all statuses, historical).

        Args:
            conversation_id: The conversation to count sub-agent runs for.

        Returns:
            The number of runs with ``agent_kind == "subagent"`` for that
            conversation, regardless of status.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM agent_runs "
                "WHERE conversation_id = ? AND agent_kind = 'subagent'",
                (conversation_id,),
            ).fetchone()
        return int(row["n"])

    def count_subagents_by_conversation(
        self,
        conversation_ids: list[str],
    ) -> dict[str, int]:
        """Count sub-agent runs for many conversations in a single query.

        Plan-B Task 7 Finding A: the Console conversation browser's
        ``[N Sub-Agents]`` badge previously called ``count_subagent_runs``
        once per visible conversation row (opening a fresh sqlite
        connection each time) on every 0.2s rail poll tick -- up to ~75
        connections/queries per tick. This batches all of them into one
        parameterized ``GROUP BY`` query.

        Args:
            conversation_ids: The conversations to count sub-agent runs
                for. Duplicates and blank entries are ignored.

        Returns:
            Mapping of ``conversation_id -> count`` for conversations that
            have at least one sub-agent run. A conversation with zero
            sub-agent runs is simply absent from the mapping (not present
            with a ``0`` value) -- callers should treat a missing key as
            zero, matching plain ``GROUP BY`` semantics.
        """
        ids = [cid for cid in dict.fromkeys(conversation_ids) if cid]
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT conversation_id, COUNT(*) AS n FROM agent_runs "
                f"WHERE agent_kind = 'subagent' AND conversation_id IN ({placeholders}) "
                "GROUP BY conversation_id",
                ids,
            ).fetchall()
        return {row["conversation_id"]: int(row["n"]) for row in rows}

    def supersede_run_tree(self, run_id: str) -> int:
        """Mark a run, and its already-terminal direct children, ``superseded``.

        PR3a-1 Task 2 lets a sub-agent outlive its turn, so a still-
        ``running`` child is not a dead attempt -- it is a live cross-turn
        survivor a retry/regenerate/variant call must not disturb. Flipping
        a row straight to ``superseded`` (itself a member of
        ``TERMINAL_RUN_STATUSES``) would not stop its live worker thread;
        it would only make ``set_status``'s first-writer-wins guard
        silently drop that run's real terminal write when it finishes for
        real, losing its result. So this only ever touches rows already in
        a terminal status -- a live row is skipped entirely and settles
        normally through its own later ``set_status`` call.

        This guard applies to the run identified by ``run_id`` itself, not
        just its children: a first draft assumed ``run_turn``'s own
        "the primary's record is always persisted before this returns"
        guarantee made the *target* primary always terminal by the time
        this runs. That guarantee is real but narrower than it looks -- it
        only covers ``run_turn``'s OWN coroutine returning, not a
        *different, earlier* run whose coroutine already returned to the
        UI (e.g. via Stop) while its ``asyncio.to_thread`` worker survives
        Task cancellation and keeps running. ``supersede_run_id`` is
        resolved as the newest non-superseded primary for the whole
        conversation, not necessarily the run tied to the message being
        retried, so retrying an older failed message can reach a
        different, still-live, stopped-but-not-dead primary. Guarding the
        primary's own row the same way the child guard does closes that
        hole without adding a second code path.

        Args:
            run_id: The run whose tree (itself + rows with
                ``parent_run_id == run_id``) should be marked superseded.
                Used by retry/regenerate to retire a prior attempt while
                keeping it for drill-in history.

        Returns:
            The number of rows updated -- the run itself plus any direct
            sub-agent children, restricted to whichever of those were
            already terminal. A live row (primary or child) is not
            counted -- it stays parented and running, untouched.
        """
        placeholders = ",".join("?" for _ in TERMINAL_RUN_STATUSES)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE agent_runs SET status = 'superseded', "
                "updated_at = ? WHERE (id = ? OR parent_run_id = ?) "
                f"AND status IN ({placeholders})",
                (_now_iso(), run_id, run_id, *sorted(TERMINAL_RUN_STATUSES)),
            )
            return cursor.rowcount
