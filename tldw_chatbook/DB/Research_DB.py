"""Local SQLite store for standalone Research Sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union
from uuid import uuid4

from .base_db import BaseDB


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def _json_loads(value: str | None, fallback: Any = None) -> Any:
    if value in {None, ""}:
        return {} if fallback is None else fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {} if fallback is None else fallback


class ResearchDatabase(BaseDB):
    """SQLite persistence for local research runs and run artifacts."""

    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: Union[str, Path], client_id: str = "default"):
        self._local = threading.local()
        super().__init__(db_path, client_id)

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._get_connection()
        return self._local.conn

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    @contextmanager
    def transaction(self):
        conn = self.conn
        nested = conn.in_transaction
        if not nested:
            conn.execute("BEGIN")
        try:
            yield conn
            if not nested:
                conn.commit()
        except Exception:
            if not nested:
                conn.rollback()
            raise

    def _initialize_schema(self):
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    source_policy TEXT NOT NULL DEFAULT 'balanced',
                    autonomy_mode TEXT NOT NULL DEFAULT 'checkpointed',
                    status TEXT NOT NULL DEFAULT 'draft',
                    phase TEXT NOT NULL DEFAULT 'planning',
                    control_state TEXT NOT NULL DEFAULT 'paused',
                    progress_percent REAL,
                    progress_message TEXT,
                    active_job_id TEXT,
                    latest_checkpoint_id TEXT,
                    completed_at TEXT,
                    chat_id TEXT,
                    limits_json TEXT NOT NULL DEFAULT '{}',
                    provider_overrides_json TEXT NOT NULL DEFAULT '{}',
                    follow_up_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    client_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS research_artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
                    artifact_name TEXT NOT NULL,
                    artifact_version INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    phase TEXT,
                    job_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, artifact_name, artifact_version)
                );

                CREATE INDEX IF NOT EXISTS idx_research_runs_updated
                    ON research_runs(deleted, updated_at);
                CREATE INDEX IF NOT EXISTS idx_research_artifacts_latest
                    ON research_artifacts(run_id, artifact_name, artifact_version);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO research_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(self._CURRENT_SCHEMA_VERSION)),
            )

    def _row_to_run(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "query": row["query"],
            "source_policy": row["source_policy"],
            "autonomy_mode": row["autonomy_mode"],
            "status": row["status"],
            "phase": row["phase"],
            "control_state": row["control_state"],
            "progress_percent": row["progress_percent"],
            "progress_message": row["progress_message"],
            "active_job_id": row["active_job_id"],
            "latest_checkpoint_id": row["latest_checkpoint_id"],
            "completed_at": row["completed_at"],
            "chat_id": row["chat_id"],
            "limits_json": _json_loads(row["limits_json"]),
            "provider_overrides": _json_loads(row["provider_overrides_json"]),
            "follow_up": _json_loads(row["follow_up_json"]),
            "metadata": _json_loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_artifact(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "artifact_name": row["artifact_name"],
            "artifact_version": row["artifact_version"],
            "content_type": row["content_type"],
            "content": _json_loads(row["content_json"], fallback=None),
            "phase": row["phase"],
            "job_id": row["job_id"],
            "created_at": row["created_at"],
        }

    def create_run(
        self,
        *,
        query: str,
        source_policy: str = "balanced",
        autonomy_mode: str = "checkpointed",
        limits_json: dict[str, Any] | None = None,
        provider_overrides: dict[str, Any] | None = None,
        chat_handoff: dict[str, Any] | None = None,
        follow_up: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> dict[str, Any]:
        if not str(query or "").strip():
            raise ValueError("query is required")
        run_id = id or f"rs_local_{uuid4().hex}"
        now = _utc_now()
        metadata = {"chat_handoff": chat_handoff} if chat_handoff else {}
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO research_runs (
                    id, query, source_policy, autonomy_mode, status, phase, control_state,
                    limits_json, provider_overrides_json, follow_up_json, metadata_json,
                    client_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(query).strip(),
                    source_policy,
                    autonomy_mode,
                    "draft",
                    "planning",
                    "paused",
                    _json_dumps(limits_json),
                    _json_dumps(provider_overrides),
                    _json_dumps(follow_up),
                    _json_dumps(metadata),
                    self.client_id,
                    now,
                    now,
                ),
            )
        return self.get_run(run_id)

    def list_runs(self, *, limit: int = 25) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM research_runs
            WHERE deleted = 0
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [self._row_to_run(row) for row in rows if row is not None]

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM research_runs WHERE id = ? AND deleted = 0",
            (run_id,),
        ).fetchone()
        record = self._row_to_run(row)
        if record is None:
            raise KeyError(run_id)
        return record

    def update_run_state(
        self,
        run_id: str,
        *,
        status: str | None = None,
        phase: str | None = None,
        control_state: str | None = None,
        progress_percent: float | None = None,
        progress_message: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_run(run_id)
        updates = {
            "status": status if status is not None else current["status"],
            "phase": phase if phase is not None else current["phase"],
            "control_state": control_state if control_state is not None else current["control_state"],
            "progress_percent": progress_percent,
            "progress_message": progress_message,
            "completed_at": _utc_now() if status in {"completed", "failed", "cancelled"} else current["completed_at"],
            "updated_at": _utc_now(),
        }
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE research_runs
                SET status = ?, phase = ?, control_state = ?, progress_percent = ?,
                    progress_message = ?, completed_at = ?, updated_at = ?
                WHERE id = ? AND deleted = 0
                """,
                (
                    updates["status"],
                    updates["phase"],
                    updates["control_state"],
                    updates["progress_percent"],
                    updates["progress_message"],
                    updates["completed_at"],
                    updates["updated_at"],
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def save_artifact(
        self,
        run_id: str,
        *,
        artifact_name: str,
        content_type: str,
        content: Any,
        phase: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        self.get_run(run_id)
        row = self.conn.execute(
            """
            SELECT MAX(artifact_version) AS latest
            FROM research_artifacts
            WHERE run_id = ? AND artifact_name = ?
            """,
            (run_id, artifact_name),
        ).fetchone()
        version = int((row["latest"] if row is not None else 0) or 0) + 1
        artifact_id = f"ra_{uuid4().hex}"
        now = _utc_now()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO research_artifacts (
                    id, run_id, artifact_name, artifact_version, content_type,
                    content_json, phase, job_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    run_id,
                    artifact_name,
                    version,
                    content_type,
                    json.dumps(content, sort_keys=True),
                    phase,
                    job_id,
                    now,
                ),
            )
            conn.execute(
                "UPDATE research_runs SET updated_at = ? WHERE id = ?",
                (now, run_id),
            )
        return self.get_artifact(run_id, artifact_name)

    def get_artifact(self, run_id: str, artifact_name: str) -> dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT * FROM research_artifacts
            WHERE run_id = ? AND artifact_name = ?
            ORDER BY artifact_version DESC
            LIMIT 1
            """,
            (run_id, artifact_name),
        ).fetchone()
        record = self._row_to_artifact(row)
        if record is None:
            raise KeyError(artifact_name)
        return record

    def get_bundle(self, run_id: str) -> dict[str, Any]:
        self.get_run(run_id)
        rows = self.conn.execute(
            """
            SELECT a.*
            FROM research_artifacts a
            JOIN (
                SELECT artifact_name, MAX(artifact_version) AS latest_version
                FROM research_artifacts
                WHERE run_id = ?
                GROUP BY artifact_name
            ) latest
                ON latest.artifact_name = a.artifact_name
               AND latest.latest_version = a.artifact_version
            WHERE a.run_id = ?
            ORDER BY a.artifact_name
            """,
            (run_id, run_id),
        ).fetchall()
        bundle: dict[str, Any] = {}
        for row in rows:
            artifact = self._row_to_artifact(row)
            if artifact is not None:
                bundle[str(artifact["artifact_name"])] = artifact["content"]
        return bundle
