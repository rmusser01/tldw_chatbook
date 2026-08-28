"""SQLite persistence for run records keyed by an unconstrained kind string.

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
from typing import Any, Iterator, Mapping, Sequence, Union

from loguru import logger

from tldw_chatbook.Agents.agent_models import (
    AGENT_LIFECYCLE_INDEX_BASE,
    AgentDefinition,
    TERMINAL_RUN_STATUSES,
    validate_agent_definition,
)
from tldw_chatbook.Agents.run_log import DEFAULT_MAX_RECORD_BYTES
from .base_db import BaseDB


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# SQLite caps host parameters per statement. The ceiling is build-dependent:
# 32766 on SQLite >= 3.32, but 999 on older builds, and this project's floor
# is Python 3.11, which can ship either. 900 is under the OLD ceiling, so the
# chunked read is correct on every build rather than on the newest one.
_IN_CLAUSE_CHUNK = 900

# JSON text can expand each live UTF-8 byte to a six-byte escape. These caps
# bound Python allocation while accommodating every valid raw-CLI field.
_LOCAL_COMMAND_CALL_ARGS_JSON_BYTES = 160 * 1024
_LOCAL_COMMAND_RESULT_ARGS_JSON_BYTES = 256 * 1024
_LOCAL_COMMAND_STEPS_JSON_BYTES = 1024
_LOCAL_COMMAND_CALL_PAYLOAD_BYTES = 512 * 1024
# JSON can expand one durable output byte to a six-byte ``\u00xx`` escape.
# The remaining allowance covers the independently bounded result args and
# fixed step envelope before SQLite parses the payload.
_LOCAL_COMMAND_RESULT_PAYLOAD_BYTES = (
    DEFAULT_MAX_RECORD_BYTES * 6
    + _LOCAL_COMMAND_RESULT_ARGS_JSON_BYTES
    + _LOCAL_COMMAND_STEPS_JSON_BYTES
)
_LOCAL_COMMAND_STATUS_BYTES = 16
_LOCAL_COMMAND_CREATED_AT_BYTES = 64


class AgentStepConflictError(ValueError):
    """A durable step index already owns a different canonical payload."""


def _canonical_step_payload(index: int, payload: dict) -> str:
    """Validate and serialize one explicit-index step before locking SQLite."""
    if type(index) is not int:
        raise TypeError("step index must be an int")
    if index < 0:
        raise ValueError("step index must be non-negative")
    if not isinstance(payload, dict):
        raise TypeError("step payload must be a dict")
    if "index" not in payload:
        raise ValueError("step payload must include index")
    payload_index = payload["index"]
    if type(payload_index) is not int:
        raise TypeError("step payload index must be an int")
    if payload_index != index:
        raise ValueError("step payload index must match sequence index")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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

    _CURRENT_SCHEMA_VERSION = 14
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
                    resumed_from_run_id TEXT,
                    -- v14 (ADR-080): the stable parent Trace event that
                    -- caused this run to exist. NULL for primary and
                    -- legacy runs whose precise cause was not captured.
                    spawn_event_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
                    ON agent_runs(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_parent
                    ON agent_runs(parent_run_id);

                -- v13 (task-18601 part A): agent_runs.steps was a single
                -- JSON blob column that append_steps rewrote WHOLE on
                -- every appended step -- read the entire blob, json.loads
                -- it, extend, json.dumps, rewrite the whole column. O(n)
                -- per append, O(n^2) per run; measured 44x slower by the
                -- 2000th append on a real DB (~5.4 minutes of write churn
                -- extrapolated to a 25k-step run). Steps now live here
                -- instead, one row per step, keyed (run_id, seq) so
                -- append_steps becomes a pure INSERT with no read of the
                -- existing log. `agent_runs.steps` is left exactly as it
                -- was (still the legacy blob column, still defaulting to
                -- '[]' for every run created from now on) -- an existing
                -- run's history stays in the blob; only NEW appends land
                -- here. See `_rows_to_dicts`'s dual-read for how a run's
                -- full step list is reassembled at read time (blob steps
                -- first, then these rows, in order), and `append_steps`'s
                -- own docstring for why concurrent callers on different
                -- threads can't race on `seq`. ON DELETE CASCADE needs
                -- `PRAGMA foreign_keys = ON`, already set unconditionally
                -- by `_get_connection` for every connection this DB opens.
                CREATE TABLE IF NOT EXISTS agent_run_steps (
                    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                );

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
                    snapshot_id INTEGER,
                    -- v12 (TASK-18060 Task 1, review-rail spec §4): anchor
                    -- kind + diff-line anchoring, for the Review screen's
                    -- plannotator-style comments alongside the card's
                    -- existing hunk notes. 'hunk' is every pre-v11 row's
                    -- true, honest kind (nothing else existed before this
                    -- column), so the DEFAULT both back-declares existing
                    -- rows correctly and keeps every V1.5 caller (which
                    -- never passes anchor_kind) byte-compatible.
                    -- diff_line_index is 0-based over the file's FULL diff
                    -- text (same semantics as hunk_index, a distinct
                    -- axis), NULL except for 'diff_line' rows.
                    -- diff_line_text is the anchored line, captured
                    -- verbatim at note-creation time -- same retention
                    -- posture as hunk_excerpt (self-contained even after
                    -- shadow-repo snapshot pruning), NULL except for
                    -- 'diff_line' rows. For 'diff_line' rows the hunk
                    -- fields ABOVE are ALSO populated (the hunk the line
                    -- falls in) -- a deliberate convergence so the card's
                    -- existing hunk-note filter renders line comments
                    -- under their hunk with no card changes. For 'file'
                    -- rows (whole-file comments) hunk_index=-1,
                    -- hunk_header='', hunk_excerpt='' -- sentinels that
                    -- can never match a real hunk, keeping them out of
                    -- the card's hunk-note filter.
                    anchor_kind TEXT NOT NULL DEFAULT 'hunk',
                    diff_line_index INTEGER,
                    diff_line_text TEXT
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
            # v13->v14 (ADR-080): precise spawn causality. NULL is the
            # honest migration value for every historical run.
            if "spawn_event_id" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN spawn_event_id TEXT"
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
            # v10->v11 (TASK-18060 Task 1, review-rail spec §4): a file
            # created while change_notes existed but before the anchor-kind
            # extension did keeps its old 12-column table -- same
            # idempotent-ALTER mechanism as every column above. The
            # DEFAULT 'hunk' on anchor_kind is what makes every row written
            # before this column existed read as 'hunk' -- truthfully, since
            # 'hunk' was the only kind that could ever have been written.
            # No DEFAULT on the two diff_line_* columns: every pre-existing
            # row correctly reads as NULL (not a diff_line note).
            if "anchor_kind" not in note_columns:
                conn.execute(
                    "ALTER TABLE change_notes ADD COLUMN "
                    "anchor_kind TEXT NOT NULL DEFAULT 'hunk'"
                )
            if "diff_line_index" not in note_columns:
                conn.execute(
                    "ALTER TABLE change_notes ADD COLUMN diff_line_index INTEGER"
                )
            if "diff_line_text" not in note_columns:
                conn.execute(
                    "ALTER TABLE change_notes ADD COLUMN diff_line_text TEXT"
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
            # v12: change_notes anchor kinds (TASK-18060 Task 1) --
            # renumbered from 11 at rebase time: task-15669 minted v11 on
            # dev concurrently, and the from-now-on contract requires each
            # migration to own a fresh number AND bump the constant.
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (12)"
            )
            # v13 (task-18601 part A): agent_run_steps child table -- see
            # the CREATE TABLE comment above for the full rationale.
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (13)"
            )
            # v14 (ADR-080): agent_runs.spawn_event_id.
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (14)"
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
        self.record_change_snapshots_batch(
            run_id=run_id,
            records=(
                {
                    "root": root,
                    "baseline_sha": baseline_sha,
                    "end_sha": end_sha,
                    "files_changed": files_changed,
                    "adds": adds,
                    "dels": dels,
                    "tracking_error": tracking_error,
                    "untracked_oversize": untracked_oversize,
                    "nested_repos": nested_repos,
                },
            ),
            kind=kind,
        )

    def record_change_snapshots_batch(
        self,
        *,
        run_id: str,
        records: Sequence[Mapping[str, Any]],
        kind: str = "turn",
    ) -> None:
        """Atomically record every root row for one completed review window."""
        if not records:
            return
        created_at = _now_iso()
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO change_snapshots
                    (run_id, root, baseline_sha, end_sha, files_changed,
                     adds, dels, tracking_error, untracked_oversize,
                     nested_repos, kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(
                    (
                        run_id,
                        str(record.get("root") or ""),
                        str(record.get("baseline_sha") or ""),
                        str(record.get("end_sha") or ""),
                        int(record.get("files_changed") or 0),
                        int(record.get("adds") or 0),
                        int(record.get("dels") or 0),
                        str(record.get("tracking_error") or ""),
                        int(record.get("untracked_oversize") or 0),
                        json.dumps(list(record.get("nested_repos") or ())),
                        kind,
                        created_at,
                    )
                    for record in records
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
        anchor_kind: str = "hunk",
        diff_line_index: int | None = None,
        diff_line_text: str | None = None,
    ) -> int:
        """Record a user-authored note anchored to a turn's diff.

        TASK-16800 (spec §1), anchor kinds extended by TASK-18060 Task 1
        (review-rail spec §4). The base anchor is ``(run_id, root, path,
        hunk_index, hunk_header)``; ``hunk_excerpt`` is captured once, at
        note-creation time, from the full diff text the caller already has
        -- it is the retention safety net that keeps display and delivery
        self-contained even after shadow-repo snapshot pruning.

        Three anchor kinds share this table (spec §4):

        - ``"hunk"`` (default, V1.5): anchored to a whole hunk, exactly as
          before this method gained the new keywords.
        - ``"diff_line"``: anchored to one line of the file's full diff
          text via ``diff_line_index``/``diff_line_text``, with the hunk
          fields ALSO populated (the hunk the line falls in) -- a
          deliberate convergence so the card's existing hunk-note filter
          renders line comments under their hunk with no card changes.
        - ``"file"``: a whole-file comment; callers pass the
          ``hunk_index=-1, hunk_header=''`` sentinels (and typically
          ``hunk_excerpt=''``) so the row can never match a real hunk.

        Args:
            run_id: The agent run whose diff this note is anchored to.
            root: Canonical root path of the changed file.
            path: The changed file's path (root-relative).
            hunk_index: 0-based index of the hunk over the FULL diff, or
                ``-1`` for a ``"file"`` note's sentinel.
            hunk_header: The hunk's ``"@@ -a,b +c,d @@ ..."`` line, verbatim,
                or ``""`` for a ``"file"`` note's sentinel.
            hunk_excerpt: The hunk body captured at note time (already
                capped/elided by the caller), or ``""`` for a ``"file"``
                note.
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
            anchor_kind: ``"hunk"`` (default), ``"file"``, or
                ``"diff_line"`` (TASK-18060 Task 1). The default keeps
                every V1.5 caller of this method byte-compatible.
            diff_line_index: 0-based index over the file's FULL diff text
                (a distinct axis from ``hunk_index``'s hunk-count
                semantics), required for ``"diff_line"`` notes and
                ``None`` otherwise.
            diff_line_text: The anchored line, captured verbatim at
                note-creation time -- same retention posture as
                ``hunk_excerpt`` -- required for ``"diff_line"`` notes and
                ``None`` otherwise.

        Returns:
            The newly created note's row id.
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO change_notes
                    (run_id, root, path, hunk_index, hunk_header,
                     hunk_excerpt, note, created_at, delivered_at,
                     snapshot_id, anchor_kind, diff_line_index,
                     diff_line_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
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
                    anchor_kind,
                    diff_line_index,
                    diff_line_text,
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

    def change_note_counts_for_conversation(
        self, conversation_id: str
    ) -> dict[tuple[str, str], int]:
        """Per-file note counts across a conversation's whole history.

        TASK-18060 Task 1 (review-rail spec §1/§4): the cross-turn "Changed
        files" rail section badges each file with a ``✎ N`` note count.
        Mirrors :meth:`pending_notes_for_conversation`'s JOIN-through-
        ``agent_runs`` shape (``agent_runs`` is the one source of truth for
        which conversation a run belongs to), but counts ALL of the
        conversation's notes -- pending and delivered alike, every anchor
        kind -- grouped by ``(root, path)`` in one parameterized query
        (no N+1 per file).

        Args:
            conversation_id: The Console conversation id.

        Returns:
            ``{(root, path): count}`` over every note across every run of
            the conversation. A ``(root, path)`` with zero notes is simply
            absent from the dict.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cn.root, cn.path, COUNT(*) AS note_count
                FROM change_notes cn
                JOIN agent_runs ar ON ar.id = cn.run_id
                WHERE ar.conversation_id = ?
                GROUP BY cn.root, cn.path
                """,
                (conversation_id,),
            ).fetchall()
        return {
            (str(row["root"]), str(row["path"])): int(row["note_count"])
            for row in rows
        }

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

    #: Every ``agent_runs`` column EXCEPT ``steps`` -- the explicit list
    #: the metadata-only read path (AC#2) selects, so a caller that only
    #: wants status/budget/result/etc never even pulls the (potentially
    #: large, legacy-blob) ``steps`` TEXT value off the page, let alone
    #: parses it. Kept as one constant so the two metadata SELECTs below
    #: can't drift apart from each other.
    _METADATA_COLUMNS = (
        "id, conversation_id, parent_run_id, agent_kind, task, status, "
        "result, budget, created_at, updated_at, assistant_message_id, "
        "agent_definition, definition_fingerprint, wake_delivered_at, "
        "resumed_from_run_id, spawn_event_id"
    )

    def _batch_hydrate_steps(
        self, conn: sqlite3.Connection, run_ids: Sequence[str]
    ) -> dict[str, list[dict]]:
        """Fetch every ``agent_run_steps`` row for many runs in ONE query.

        Mirrors this file's existing no-N+1 precedent (e.g. TASK-1972's
        conversation-level ``change_snapshots`` fetch): a multi-row read
        (``list_runs``, ``undelivered_wake_runs``) must not issue one
        child-table query per returned run.

        Args:
            conn: An open connection (read or write).
            run_ids: The run ids to fetch step rows for. Duplicates are
                harmless; an empty sequence short-circuits without a query.

        Returns:
            ``{run_id: [step_dict, ...]}`` in ``seq`` order, for every
            run_id that has at least one row. A run_id with zero rows is
            simply absent (not present with an empty list).

        Note:
            Issued in chunks of ``_IN_CLAUSE_CHUNK`` ids, because a bound
            parameter per id runs into SQLite's host-parameter ceiling and
            no caller bounds the list (``ConsoleAgentController.
            subagent_runs`` asks for every run in a conversation). Chunking
            keeps the no-N+1 property -- one query per 900 runs, not per run.
        """
        ids = list(dict.fromkeys(run_ids))
        if not ids:
            return {}
        grouped: dict[str, list[dict]] = {}
        for start in range(0, len(ids), _IN_CLAUSE_CHUNK):
            chunk = ids[start : start + _IN_CLAUSE_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT run_id, payload FROM agent_run_steps "
                f"WHERE run_id IN ({placeholders}) ORDER BY run_id, seq",
                chunk,
            ).fetchall()
            for row in rows:
                grouped.setdefault(row["run_id"], []).append(
                    json.loads(row["payload"])
                )
        return grouped

    def _rows_to_dicts(
        self, conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]
    ) -> list[dict]:
        """Turn ``agent_runs`` rows into API dicts, with full step hydration.

        DUAL READ (task-18601 part A, AC#4): a run's steps can live in
        TWO places -- the legacy ``agent_runs.steps`` JSON blob (every run
        written before this change, and never touched again by
        ``append_steps`` from now on) and the ``agent_run_steps`` child
        table (every append from now on, for both brand-new runs and a
        legacy run that gets appended to again after this change landed).

        Chosen strategy: READ AND CONCATENATE both sources (blob steps
        first, then child rows in ``seq`` order) rather than migrating a
        legacy blob into rows on first append. Trade-off, deliberately
        accepted: a run that mixes legacy blob-steps with new child rows
        pays one extra query per hydrating read (parse the blob + select
        the child rows) -- negligible next to the O(n^2) writer cost this
        task fixes, and a steps-hydrating read of n steps is at best O(n)
        regardless (it returns n items). The alternative (migrate-on-
        first-append) would need a write on a READ-triggered path or an
        extra step inside ``append_steps`` proper, and would still need a
        migration guard forever (a DB can be opened by an older binary
        between two appends). Concatenate-at-read is simpler and never
        mutates data implicitly. A run created after this change has an
        empty blob ('[]'), so its hydration is one indexed child-table
        SELECT with no JSON blob to parse at all.

        Args:
            conn: An open connection (read or write) -- used for the
                child-table query; a bare ``sqlite3.Row`` has no DB access
                of its own, so this can no longer be a ``@staticmethod``.
            rows: The ``agent_runs`` rows to convert.

        Returns:
            One dict per input row, in the same order, with ``steps`` a
            list of dicts (blob steps then child-row steps, in order) and
            ``budget`` JSON-decoded.
        """
        rows = list(rows)
        child_by_run = self._batch_hydrate_steps(conn, [r["id"] for r in rows])
        records: list[dict] = []
        for row in rows:
            record = dict(row)
            blob_steps = json.loads(record["steps"] or "[]")
            record["steps"] = blob_steps + child_by_run.get(record["id"], [])
            record["budget"] = (
                json.loads(record["budget"]) if record["budget"] else None
            )
            records.append(record)
        return records

    def _row_to_dict(self, conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
        """Single-row convenience wrapper around :meth:`_rows_to_dicts`."""
        return self._rows_to_dicts(conn, [row])[0]

    @staticmethod
    def _metadata_row_to_dict(row: sqlite3.Row) -> dict:
        """Row -> dict for the metadata-only read path (AC#2).

        No ``steps`` key at all -- deliberately, not ``[]`` and not
        ``None``: every caller of the metadata-only methods below is
        audited to never touch ``record["steps"]`` (see their docstrings
        for exactly which real call site each replaced and why it is
        provably steps-free), so a future caller that reaches for it by
        mistake gets a loud ``KeyError`` instead of silently reading "no
        steps recorded" off a query that never asked the DB about steps
        at all. This never touches ``agent_run_steps`` and never
        ``json.loads``es the legacy blob -- the caller's SELECT (see
        ``_METADATA_COLUMNS``) doesn't even fetch the ``steps`` column.
        """
        record = dict(row)
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
        spawn_event_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Create a new run record in ``running`` status.

        Args:
            conversation_id: The owning Console conversation's id.
            agent_kind: Caller-owned kind, such as ``"primary"``,
                ``"subagent"``, or ``"local_command"``.
            task: A generic run label or sub-agent task; ``None`` when the
                caller does not record one.
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
            spawn_event_id: Stable parent Trace event that caused this run.
                ``None`` for primary and legacy runs.
            run_id: Preallocated stable identity; generated when omitted.

        Returns:
            The newly created run's id (a hex UUID4).
        """
        run_id = run_id or uuid.uuid4().hex
        now = _now_iso()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (id, conversation_id, parent_run_id, agent_kind, task,
                    status, steps, result, budget, created_at, updated_at,
                    assistant_message_id, agent_definition, definition_fingerprint,
                    resumed_from_run_id, spawn_event_id)
                   VALUES (?, ?, ?, ?, ?, 'running', '[]', NULL, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    spawn_event_id,
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

        task-18601 part A: this used to be read-modify-write on the whole
        ``agent_runs.steps`` JSON blob (read it, ``json.loads``, extend,
        ``json.dumps``, rewrite the whole column) -- O(n) work per call
        where n is every step ever recorded for the run, so O(n^2) over a
        run's lifetime. Measured on a real DB: the 2000th append cost 44x
        the 1st, ~5.4 minutes of write churn extrapolated to a 25k-step
        run. Now a pure ``INSERT`` into ``agent_run_steps`` -- no read of
        the existing log at all, just an existence check on ``run_id``
        (an indexed point lookup) and a ``SELECT MAX(seq)`` (also indexed,
        via the ``(run_id, seq)`` primary key -- SQLite answers a
        ``MAX(seq) WHERE run_id = ?`` by walking straight to the last
        matching index entry, not by scanning every row). ``agent_runs.
        steps`` itself is left untouched -- see ``_rows_to_dicts``'s
        dual-read docstring for how a run's full step list is reassembled
        at read time.

        Concurrency: computing ``next_seq`` and inserting the new rows
        both happen inside the SAME ``self.transaction()`` (``BEGIN
        IMMEDIATE``), which is this file's existing multi-writer-thread
        discipline (see ``transaction()``'s own docstring: "a primary run
        and its sub-agent runs" write concurrently today, each from its
        own thread's held connection). ``BEGIN IMMEDIATE`` acquires
        SQLite's single write lock up front, so a second thread's
        ``append_steps`` call on ANY run blocks (up to ``busy_timeout``)
        until this transaction commits or rolls back -- there is no
        window between the ``MAX(seq)`` read and the ``INSERT`` for a
        second writer to compute the same ``next_seq`` and collide on the
        ``(run_id, seq)`` primary key.

        Args:
            run_id: The run to append to.
            steps: Serialized ``AgentStep`` dicts, appended in order after
                any steps already recorded (both the legacy blob's steps
                and any already-inserted rows).

        Raises:
            KeyError: If ``run_id`` does not exist.
        """
        stamp = _now_iso()
        with self.transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown run id: {run_id}")
            if steps:
                max_row = conn.execute(
                    "SELECT MAX(seq) AS max_seq FROM agent_run_steps "
                    "WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                next_seq = (
                    int(max_row["max_seq"]) + 1
                    if max_row and max_row["max_seq"] is not None
                    else 0
                )
                conn.executemany(
                    "INSERT INTO agent_run_steps (run_id, seq, payload, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (run_id, next_seq + offset, json.dumps(step), stamp)
                        for offset, step in enumerate(steps)
                    ],
                )
            conn.execute(
                "UPDATE agent_runs SET updated_at = ? WHERE id = ?",
                (stamp, run_id),
            )

    def insert_steps_at_indices(
        self, run_id: str, steps: Sequence[tuple[int, dict]]
    ) -> None:
        """Insert caller-indexed steps without rewriting existing rows.

        Live capture calls this with one step; terminal recovery calls it
        with the complete outcome. Validation and canonical JSON encoding
        finish before the write lock. Under the lock, an identical retry is
        a no-op, missing rows are inserted, and divergent durable indices are
        collected. The transaction commits before ``AgentStepConflictError``
        reports those conflicts, so recovery never loses unrelated rows.
        Step inserts do not change ``agent_runs.updated_at`` because that
        timestamp records lifecycle transitions used by wake classification.

        Raises:
            KeyError: If ``run_id`` does not exist.
            TypeError: If an index or payload has the wrong type, or JSON
                serialization fails.
            ValueError: If an index is negative or disagrees with its payload.
            AgentStepConflictError: If one index has divergent payloads.
        """
        prepared: dict[int, str] = {}
        for index, payload in steps:
            canonical = _canonical_step_payload(index, payload)
            if index in prepared and prepared[index] != canonical:
                raise AgentStepConflictError(
                    f"conflicting step payloads for run index {index}"
                )
            prepared[index] = canonical

        stamp = _now_iso()
        conflicts: list[int] = []
        with self.transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown run id: {run_id}")
            for index, canonical in prepared.items():
                existing = conn.execute(
                    "SELECT payload FROM agent_run_steps "
                    "WHERE run_id = ? AND seq = ?",
                    (run_id, index),
                ).fetchone()
                if existing is not None:
                    try:
                        stored = json.dumps(
                            json.loads(existing["payload"]),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    except (TypeError, ValueError):
                        conflicts.append(index)
                        continue
                    if stored != canonical:
                        conflicts.append(index)
                    continue
                conn.execute(
                    "INSERT INTO agent_run_steps (run_id, seq, payload, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, index, canonical, stamp),
                )
        if conflicts:
            indices = ", ".join(str(index) for index in conflicts)
            raise AgentStepConflictError(
                f"step payload conflicts with durable indices: {indices}"
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

    def set_terminal_with_step(
        self,
        run_id: str,
        status: str,
        result: str | None,
        terminal_step: dict,
    ) -> bool:
        """Atomically persist a first-writer terminal state and observation."""
        if status not in TERMINAL_RUN_STATUSES:
            raise ValueError("status must be terminal")
        index = terminal_step.get("index")
        canonical = _canonical_step_payload(index, terminal_step)
        placeholders = ",".join("?" for _ in TERMINAL_RUN_STATUSES)
        stamp = _now_iso()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run id: {run_id}")
            existing = conn.execute(
                "SELECT payload FROM agent_run_steps WHERE run_id = ? AND seq = ?",
                (run_id, index),
            ).fetchone()
            if existing is not None:
                stored = json.dumps(
                    json.loads(existing["payload"]),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if stored != canonical:
                    raise AgentStepConflictError(
                        f"step payload conflicts with durable index: {index}"
                    )
            if row["status"] in TERMINAL_RUN_STATUSES:
                return False
            if existing is None:
                conn.execute(
                    "INSERT INTO agent_run_steps (run_id, seq, payload, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, index, canonical, stamp),
                )
            cursor = conn.execute(
                "UPDATE agent_runs SET status = ?, "
                "result = COALESCE(?, result), updated_at = ? "
                f"WHERE id = ? AND status NOT IN ({placeholders})",
                (status, result, stamp, run_id, *sorted(TERMINAL_RUN_STATUSES)),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("terminal status changed during transaction")
        return True

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
            def run_observations(run_id: str) -> tuple[list[dict], int, str]:
                parsed: list[dict] = []
                for step_row in conn.execute(
                    "SELECT payload FROM agent_run_steps WHERE run_id = ?",
                    (run_id,),
                ).fetchall():
                    try:
                        parsed.append(json.loads(step_row["payload"]))
                    except (TypeError, json.JSONDecodeError):
                        continue
                ordered = [
                    step
                    for step in parsed
                    if isinstance(step.get("owner_seq"), int)
                    and isinstance(step.get("index"), int)
                ]
                latest = max(
                    ordered,
                    key=lambda step: (step["owner_seq"], step["index"]),
                    default=None,
                )
                owner_seq = latest["owner_seq"] if latest is not None else -1
                parent = (
                    f"agent-step:{run_id}:{latest['index']}"
                    if latest is not None
                    else f"agent-run:{run_id}"
                )
                return parsed, owner_seq, parent

            def insert_recovery_diagnostic(
                run_id: str,
                index: int,
                summary: str,
                field_states: dict[str, str],
            ) -> None:
                _steps, owner_seq, parent = run_observations(run_id)
                diagnostic = {
                    "index": index,
                    "kind": "capture_failed",
                    "summary": summary,
                    "created_at": observed_at,
                    "status": "incomplete",
                    "owner_seq": owner_seq + 1,
                    "parent_event_id": parent,
                    "source_event_id": None,
                    "field_states": {
                        "payload": "capture_failed",
                        **field_states,
                    },
                    "sensitivity": "diagnostic",
                }
                canonical = _canonical_step_payload(index, diagnostic)
                conn.execute(
                    "INSERT OR IGNORE INTO agent_run_steps "
                    "(run_id, seq, payload, created_at) VALUES (?, ?, ?, ?)",
                    (run_id, index, canonical, observed_at),
                )

            orphan_ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM agent_runs WHERE status = 'running' "
                    "AND agent_kind IN ('primary', 'subagent')"
                ).fetchall()
            ]
            local_orphan_ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM agent_runs WHERE status = 'running' "
                    "AND agent_kind = 'local_command'"
                ).fetchall()
            ]
            observed_at = _now_iso()
            diagnostic_index = AGENT_LIFECYCLE_INDEX_BASE + 500
            for run_id in orphan_ids:
                insert_recovery_diagnostic(
                    run_id,
                    diagnostic_index,
                    "Terminal state repaired after app restart",
                    {"reconciliation": "observed"},
                )
                conn.execute(
                    "UPDATE agent_runs SET status = 'error', "
                    "result = COALESCE(result, 'Interrupted by app restart'), "
                    "updated_at = ? WHERE id = ? AND status = 'running'",
                    (observed_at, run_id),
                )
            for run_id in local_orphan_ids:
                conn.execute(
                    "UPDATE agent_runs SET status = 'error', updated_at = ? "
                    "WHERE id = ? AND status = 'running' "
                    "AND agent_kind = 'local_command'",
                    (observed_at, run_id),
                )
            terminal_kind = {
                "done": "agent_run_completed",
                "cancelled": "agent_run_cancelled",
                "superseded": "agent_run_superseded",
                "error": "agent_run_failed",
                "stuck": "agent_run_failed",
            }
            terminal_rows = conn.execute(
                "SELECT id, status FROM agent_runs WHERE status != 'running' "
                "AND agent_kind IN ('primary', 'subagent')"
            ).fetchall()
            split_rows = 0
            repaired_orphans = set(orphan_ids)
            for row in terminal_rows:
                if row["id"] in repaired_orphans:
                    continue
                expected_kind = terminal_kind.get(row["status"])
                if expected_kind is None:
                    continue
                steps, _owner_seq, _parent = run_observations(row["id"])
                if any(step.get("kind") == expected_kind for step in steps):
                    continue
                insert_recovery_diagnostic(
                    row["id"],
                    AGENT_LIFECYCLE_INDEX_BASE + 501,
                    "Preexisting terminal state lacked lifecycle capture",
                    {
                        "reconciliation": "observed",
                        expected_kind: "not_observed",
                    },
                )
                split_rows += 1
            rowcount = len(orphan_ids) + len(local_orphan_ids) + split_rows
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
            return self._row_to_dict(conn, row) if row else None

    def get_run_metadata(self, run_id: str) -> dict | None:
        """Fetch one run's METADATA ONLY -- never touches the step log.

        task-18601 part A, AC#2: a plain single-row SELECT over
        ``_METADATA_COLUMNS`` (everything except ``steps``), so this
        never fetches the ``steps`` blob off the page, never queries
        ``agent_run_steps``, and never ``json.loads``es anything but
        ``budget``. The returned dict has NO ``steps`` key at all -- see
        ``_metadata_row_to_dict`` for why that is deliberate.

        Use this instead of :meth:`get_run` wherever the caller only
        inspects status/budget/result/task/etc, never
        ``record["steps"]``. Two real call sites were switched to this
        when it was added: ``ConsoleFleetWakeCoordinator._rows_for``
        (the pending-wake poll -- ``compose_wake_notice`` only reads
        id/agent_definition/status/task/result) and
        ``AgentService.send_to_agent``'s "run finished in an earlier
        session" check (only reads agent_kind/conversation_id/status).
        :meth:`get_run` keeps its full (steps-hydrating) contract
        unchanged for every other caller -- e.g.
        ``change_review_screen.tool_touched_relpaths``, which reads
        ``record["steps"]`` directly.

        Args:
            run_id: The run to fetch.

        Returns:
            The run as a dict with ``budget`` JSON-decoded and NO
            ``steps`` key, or ``None`` if ``run_id`` does not exist.
        """
        with self.connection() as conn:
            row = conn.execute(
                f"SELECT {self._METADATA_COLUMNS} FROM agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._metadata_row_to_dict(row) if row else None

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
            return self._row_to_dict(conn, row) if row else None
        finally:
            conn.close()

    def get_run_metadata_fresh(self, run_id: str) -> dict | None:
        """``get_run_metadata`` through the same dedicated-connection
        escape hatch ``get_run_fresh`` uses -- see that method's
        docstring for the pinned-snapshot rationale (task-15863). Used by
        ``ConsoleFleetWakeCoordinator._rows_for``'s re-read-on-stale-
        non-terminal path, which previously called ``get_run_fresh`` for
        a result it only ever inspects ``status``/``wake_delivered_at``
        on.

        Args:
            run_id: The run to read.

        Returns:
            The same metadata-only dict ``get_run_metadata`` returns (no
            ``steps`` key), or ``None`` if no run has that id.

        Raises:
            sqlite3.Error: Propagated unchanged from the read. The private
                connection is closed either way.
        """
        if self.is_memory_db:
            return self.get_run_metadata(run_id)
        conn = self._get_connection()
        try:
            row = conn.execute(
                f"SELECT {self._METADATA_COLUMNS} FROM agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            return self._metadata_row_to_dict(row) if row else None
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
            return self._row_to_dict(conn, row) if row else None

    def latest_primary_run_metadata(self, conversation_id: str) -> dict | None:
        """``latest_primary_run`` METADATA ONLY -- never touches the step log.

        task-18601 part A, AC#2: same query/ordering as
        :meth:`latest_primary_run`, but over ``_METADATA_COLUMNS`` (no
        ``steps``, no ``agent_run_steps`` query). Both of that method's
        real callers only ever read ``id``/``assistant_message_id``
        (``ConsoleAgentController.latest_primary_run_id`` and its
        assistant-message-anchor sibling in
        ``Chat/console_agent_bridge.py``) -- neither touches
        ``record["steps"]`` -- so they were switched to this.
        :meth:`latest_primary_run` keeps its full contract for any future
        caller that does need steps.

        Args:
            conversation_id: The conversation whose runs to inspect.

        Returns:
            The newest matching run as a dict with ``budget`` JSON-
            decoded and NO ``steps`` key, or ``None`` when the
            conversation has no non-superseded primary run.
        """
        with self.connection() as conn:
            row = conn.execute(
                f"SELECT {self._METADATA_COLUMNS} FROM agent_runs "
                "WHERE conversation_id = ? "
                "AND agent_kind = 'primary' AND status != 'superseded' "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return self._metadata_row_to_dict(row) if row else None

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
            agent_kind: When set, restricts to that exact caller-owned kind
                IN THE QUERY -- e.g.
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
            return self._rows_to_dicts(conn, rows)

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
            agent_kind: When set, restricts the count to that exact
                caller-owned kind -- mirrors ``list_runs``'
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

    def local_command_resume_records(self, conversation_id: str) -> list[dict]:
        """Return bounded structural projections for local-command markers.

        SQL admits only the exact kind, exact two-step card shape, bounded
        metadata, and bounded JSON ``args`` objects. It never selects the
        full step payload or tool-result ``result``. The strict display
        parser remains the canonical validator for every field inside those
        bounded objects.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                WITH eligible_local_commands AS MATERIALIZED (
                    SELECT
                        ar.rowid AS run_rowid,
                        ar.id,
                        ar.status,
                        ar.assistant_message_id,
                        ar.created_at,
                        call_step.rowid AS call_step_rowid,
                        result_step.rowid AS result_step_rowid
                    FROM agent_runs AS ar
                    JOIN agent_run_steps AS call_step
                      ON call_step.run_id = ar.id AND call_step.seq = 0
                    JOIN agent_run_steps AS result_step
                      ON result_step.run_id = ar.id AND result_step.seq = 1
                    WHERE ar.conversation_id = ?
                      AND ar.agent_kind = 'local_command'
                      AND ar.status != 'superseded'
                      AND typeof(ar.id) = 'text'
                      AND length(CAST(ar.id AS BLOB)) BETWEEN 1 AND 128
                      AND (
                          ar.assistant_message_id IS NULL
                          OR (
                              typeof(ar.assistant_message_id) = 'text'
                              AND length(CAST(
                                  ar.assistant_message_id AS BLOB
                              )) BETWEEN 1 AND 128
                          )
                      )
                      AND typeof(ar.status) = 'text'
                      AND length(CAST(ar.status AS BLOB))
                          BETWEEN 1 AND ?
                      AND typeof(ar.created_at) = 'text'
                      AND length(CAST(ar.created_at AS BLOB))
                          BETWEEN 1 AND ?
                      AND typeof(ar.steps) = 'text'
                      AND length(CAST(ar.steps AS BLOB))
                          BETWEEN 2 AND ?
                      AND typeof(call_step.payload) = 'text'
                      AND length(CAST(call_step.payload AS BLOB))
                          BETWEEN 2 AND ?
                      AND typeof(result_step.payload) = 'text'
                      AND length(CAST(result_step.payload AS BLOB))
                          BETWEEN 2 AND ?
                      AND (
                          SELECT COUNT(*) FROM agent_run_steps AS all_steps
                          WHERE all_steps.run_id = ar.id
                      ) = 2
                ), projected AS (
                    SELECT
                        CAST(eligible.id AS BLOB) AS id,
                        CAST(eligible.status AS BLOB) AS status,
                        CAST(eligible.assistant_message_id AS BLOB)
                            AS assistant_message_id,
                        eligible.created_at,
                        CAST(json_extract(
                            call_step.payload, '$.args'
                        ) AS BLOB) AS call_args_json,
                        CAST(json_extract(
                            result_step.payload, '$.args'
                        ) AS BLOB) AS result_args_json,
                        CAST(json_extract(
                            result_step.payload, '$.status'
                        ) AS BLOB) AS result_status
                    FROM eligible_local_commands AS eligible
                    JOIN agent_runs AS ar ON ar.rowid = eligible.run_rowid
                    JOIN agent_run_steps AS call_step
                      ON call_step.rowid = eligible.call_step_rowid
                    JOIN agent_run_steps AS result_step
                      ON result_step.rowid = eligible.result_step_rowid
                    WHERE json_valid(call_step.payload) = 1
                      AND json_valid(result_step.payload) = 1
                      AND CASE WHEN json_valid(ar.steps) = 1
                               THEN json_type(ar.steps) END = 'array'
                      AND CASE WHEN json_valid(ar.steps) = 1
                               THEN json_array_length(ar.steps) END = 0
                      AND json_type(call_step.payload, '$') = 'object'
                      AND json_type(call_step.payload, '$.index') = 'integer'
                      AND json_extract(call_step.payload, '$.index') = 0
                      AND json_type(call_step.payload, '$.kind') = 'text'
                      AND json_extract(call_step.payload, '$.kind') = 'tool_call'
                      AND json_type(call_step.payload, '$.tool_name') = 'text'
                      AND json_extract(call_step.payload, '$.tool_name') = 'raw_cli'
                      AND json_type(call_step.payload, '$.args') = 'object'
                      AND length(CAST(json_extract(
                          call_step.payload, '$.args'
                      ) AS BLOB)) BETWEEN 2 AND ?
                      AND json_type(result_step.payload, '$') = 'object'
                      AND json_type(result_step.payload, '$.index') = 'integer'
                      AND json_extract(result_step.payload, '$.index') = 1
                      AND json_type(result_step.payload, '$.kind') = 'text'
                      AND json_extract(result_step.payload, '$.kind') = 'tool_result'
                      AND json_type(result_step.payload, '$.tool_name') = 'text'
                      AND json_extract(result_step.payload, '$.tool_name') = 'raw_cli'
                      AND json_type(result_step.payload, '$.args') = 'object'
                      AND length(CAST(json_extract(
                          result_step.payload, '$.args'
                      ) AS BLOB)) BETWEEN 2 AND ?
                      AND json_type(result_step.payload, '$.status') = 'text'
                      AND length(CAST(json_extract(
                          result_step.payload, '$.status'
                      ) AS BLOB)) BETWEEN 1 AND 16
                )
                SELECT
                    id,
                    status,
                    assistant_message_id,
                    call_args_json,
                    result_args_json,
                    result_status
                FROM projected
                ORDER BY created_at ASC, id ASC
                """,
                (
                    conversation_id,
                    _LOCAL_COMMAND_STATUS_BYTES,
                    _LOCAL_COMMAND_CREATED_AT_BYTES,
                    _LOCAL_COMMAND_STEPS_JSON_BYTES,
                    _LOCAL_COMMAND_CALL_PAYLOAD_BYTES,
                    _LOCAL_COMMAND_RESULT_PAYLOAD_BYTES,
                    _LOCAL_COMMAND_CALL_ARGS_JSON_BYTES,
                    _LOCAL_COMMAND_RESULT_ARGS_JSON_BYTES,
                ),
            ).fetchall()

        records: list[dict] = []
        for row in rows:
            try:
                encoded_fields = (
                    row["id"],
                    row["status"],
                    row["call_args_json"],
                    row["result_args_json"],
                    row["result_status"],
                )
                if any(type(value) is not bytes for value in encoded_fields):
                    continue
                (
                    run_id,
                    status,
                    call_args_json,
                    result_args_json,
                    result_status,
                ) = (value.decode("utf-8") for value in encoded_fields)
                encoded_anchor = row["assistant_message_id"]
                if encoded_anchor is None:
                    anchor = None
                elif type(encoded_anchor) is bytes:
                    anchor = encoded_anchor.decode("utf-8")
                else:
                    continue
                if (
                    not run_id.strip()
                    or len(run_id.encode("utf-8")) > 128
                    or (
                        anchor is not None
                        and (not anchor.strip() or len(anchor.encode("utf-8")) > 128)
                    )
                ):
                    continue
                call_args = json.loads(call_args_json)
                result_args = json.loads(result_args_json)
                if type(call_args) is not dict or type(result_args) is not dict:
                    continue
            except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
                continue
            records.append(
                {
                    "id": run_id,
                    "agent_kind": "local_command",
                    "status": status,
                    "assistant_message_id": anchor,
                    "steps": [
                        {
                            "index": 0,
                            "kind": "tool_call",
                            "tool_name": "raw_cli",
                            "args": call_args,
                        },
                        {
                            "index": 1,
                            "kind": "tool_result",
                            "tool_name": "raw_cli",
                            "status": result_status,
                            "args": result_args,
                        },
                    ],
                }
            )
        return records

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
                "AND child.agent_kind = 'subagent' "
                "AND child.wake_delivered_at IS NULL "
                "AND child.status IN ('done', 'error', 'cancelled') "
                f"AND parent.status IN ({', '.join('?' for _ in TERMINAL_RUN_STATUSES)}) "
                "AND child.updated_at >= parent.updated_at "
                "ORDER BY child.updated_at ASC, child.id ASC",
                (conversation_id, *sorted(TERMINAL_RUN_STATUSES)),
            ).fetchall()
            return self._rows_to_dicts(conn, rows)

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
        """Mark an exact primary and terminal direct sub-agents superseded.

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
            run_id: The exact primary whose tree (itself + exact sub-agent
                rows with ``parent_run_id == run_id``) should be marked superseded.
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
                "updated_at = ? WHERE ("
                "(id = ? AND agent_kind = 'primary') OR "
                "(parent_run_id = ? AND agent_kind = 'subagent' AND EXISTS ("
                "SELECT 1 FROM agent_runs AS parent "
                "WHERE parent.id = ? AND parent.agent_kind = 'primary'"
                "))"
                ") "
                f"AND status IN ({placeholders})",
                (
                    _now_iso(),
                    run_id,
                    run_id,
                    run_id,
                    *sorted(TERMINAL_RUN_STATUSES),
                ),
            )
            return cursor.rowcount
