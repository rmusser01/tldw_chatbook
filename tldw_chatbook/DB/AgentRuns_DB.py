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
        with closing(self._get_connection()) as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._get_connection()) as conn:
            conn.execute("BEGIN")
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
        with self.transaction() as conn:
            conn.execute(
                "UPDATE agent_runs SET status = ?, "
                "result = COALESCE(?, result), updated_at = ? WHERE id = ?",
                (status, result, _now_iso(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?",
                (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(self, conversation_id: str,
                  include_superseded: bool = True) -> list[dict]:
        query = "SELECT * FROM agent_runs WHERE conversation_id = ?"
        params: list = [conversation_id]
        if not include_superseded:
            query += " AND status != 'superseded'"
        query += " ORDER BY created_at DESC, id DESC"
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_subagent_runs(self, conversation_id: str) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM agent_runs "
                "WHERE conversation_id = ? AND agent_kind = 'subagent'",
                (conversation_id,)).fetchone()
        return int(row["n"])

    def supersede_run_tree(self, run_id: str) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE agent_runs SET status = 'superseded', "
                "updated_at = ? WHERE id = ? OR parent_run_id = ?",
                (_now_iso(), run_id, run_id))
            return cursor.rowcount
