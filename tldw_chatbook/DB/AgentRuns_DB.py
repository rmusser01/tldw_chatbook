"""SQLite persistence for agent run records (primary + sub-agent).

Follows the Workspace_DB pattern: BaseDB, per-call connections (reads get
their own connection automatically), transaction() for writes.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Union

from .base_db import BaseDB


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class AgentRunsDB(BaseDB):
    """Run records for the agent runtime (vertical-slice spec data model)."""

    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: Union[str, Path],
                 client_id: str = "default") -> None:
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a fresh read connection, closed on exit.

        Yields:
            A ``sqlite3.Connection`` scoped to this ``with`` block; every
            caller gets its own connection (no shared/cached state).
        """
        with closing(self._get_connection()) as conn:
            yield conn

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
        with closing(self._get_connection()) as conn:
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
                INSERT OR IGNORE INTO schema_version (version) VALUES (1);

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
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
                    ON agent_runs(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_parent
                    ON agent_runs(parent_run_id);
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["steps"] = json.loads(record["steps"] or "[]")
        record["budget"] = (json.loads(record["budget"])
                            if record["budget"] else None)
        return record

    def create_run(self, *, conversation_id: str, agent_kind: str,
                   task: str | None = None, parent_run_id: str | None = None,
                   budget: dict | None = None) -> str:
        """Create a new run record in ``running`` status.

        Args:
            conversation_id: The owning Console conversation's id.
            agent_kind: ``"primary"`` or ``"subagent"``.
            task: The sub-agent's task text; ``None`` for a primary run.
            parent_run_id: The parent run's id for a sub-agent; ``None``
                for a primary run.
            budget: The run's ``RunBudget`` serialized to a dict, stored
                as JSON for later inspection; ``None`` if not recorded.

        Returns:
            The newly created run's id (a hex UUID4).
        """
        run_id = uuid.uuid4().hex
        now = _now_iso()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (id, conversation_id, parent_run_id, agent_kind, task,
                    status, steps, result, budget, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'running', '[]', NULL, ?, ?, ?)""",
                (run_id, conversation_id, parent_run_id, agent_kind, task,
                 json.dumps(budget) if budget is not None else None,
                 now, now),
            )
        return run_id

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
                "SELECT steps FROM agent_runs WHERE id = ?",
                (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run id: {run_id}")
            existing = json.loads(row["steps"] or "[]")
            existing.extend(steps)
            conn.execute(
                "UPDATE agent_runs SET steps = ?, updated_at = ? "
                "WHERE id = ?",
                (json.dumps(existing), _now_iso(), run_id))

    def set_status(self, run_id: str, status: str,
                   result: str | None = None) -> None:
        """Update a run's terminal (or in-progress) status.

        Args:
            run_id: The run to update.
            status: The new status (e.g. ``"done"``, ``"stuck"``,
                ``"error"``, ``"cancelled"``, ``"superseded"``).
            result: The final answer text (primary) or sub-agent result
                text; when ``None`` the existing ``result`` column is left
                unchanged (``COALESCE``), so a status-only update never
                clobbers a previously recorded result.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE agent_runs SET status = ?, "
                "result = COALESCE(?, result), updated_at = ? WHERE id = ?",
                (status, result, _now_iso(), run_id))

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
                "SELECT * FROM agent_runs WHERE id = ?",
                (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(self, conversation_id: str,
                  include_superseded: bool = True,
                  limit: int | None = None) -> list[dict]:
        """List a conversation's run records, newest first.

        Args:
            conversation_id: The conversation to list runs for.
            include_superseded: When ``False``, excludes runs whose
                status is ``"superseded"``.
            limit: When set, caps the result to the ``limit`` most recent
                runs (``ORDER BY created_at DESC, id DESC``). ``None``
                (the default) returns every matching run, preserving prior
                behavior.

        Returns:
            The matching runs as dicts, newest first.
        """
        query = "SELECT * FROM agent_runs WHERE conversation_id = ?"
        params: list = [conversation_id]
        if not include_superseded:
            query += " AND status != 'superseded'"
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

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
                (conversation_id,)).fetchone()
        return int(row["n"])

    def supersede_run_tree(self, run_id: str) -> int:
        """Mark a run and its direct children ``superseded``.

        Args:
            run_id: The run whose tree (itself + rows with
                ``parent_run_id == run_id``) should be marked superseded.
                Used by retry/regenerate to retire a prior attempt while
                keeping it for drill-in history.

        Returns:
            The number of rows updated (the run itself plus its direct
            sub-agent children).
        """
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE agent_runs SET status = 'superseded', "
                "updated_at = ? WHERE id = ? OR parent_run_id = ?",
                (_now_iso(), run_id, run_id))
            return cursor.rowcount
