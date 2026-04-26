"""SQLite-backed local research session/run service."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .research_normalizers import normalize_research_record


class LocalResearchService:
    """Local-first persistence for research sessions, runs, events, and artifacts."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        notification_dispatcher: Any | None = None,
        notification_app: Any | None = None,
    ):
        self.db_path = Path(db_path)
        self.notification_dispatcher = notification_dispatcher
        self.notification_app = notification_app
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _check_version(row: dict[str, Any], expected_version: int | None) -> None:
        if expected_version is not None and int(row["version"]) != int(expected_version):
            raise ValueError("version conflict")

    @staticmethod
    def _dump_json(value: Any) -> str:
        return json.dumps(value or {}, sort_keys=True)

    @staticmethod
    def _load_json(value: str | None) -> Any:
        if not value:
            return {}
        return json.loads(value)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    phase TEXT NOT NULL DEFAULT 'local_planning',
                    control_state TEXT NOT NULL DEFAULT 'running',
                    progress_percent REAL,
                    progress_message TEXT,
                    source_policy TEXT NOT NULL DEFAULT 'balanced',
                    autonomy_mode TEXT NOT NULL DEFAULT 'checkpointed',
                    limits_json TEXT NOT NULL DEFAULT '{}',
                    provider_overrides_json TEXT NOT NULL DEFAULT '{}',
                    chat_handoff_json TEXT NOT NULL DEFAULT '{}',
                    follow_up_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)
                );
                CREATE TABLE IF NOT EXISTS research_run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    data_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES research_runs(id)
                );
                CREATE TABLE IF NOT EXISTS research_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    content_json TEXT,
                    content_text TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, artifact_name),
                    FOREIGN KEY(run_id) REFERENCES research_runs(id)
                );
                """
            )

    def _fetch_one(self, table: str, item_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND deleted = 0",
                (item_id,),
            ).fetchone()
        return dict(row) if row else None

    def _require_one(self, table: str, item_id: str, label: str) -> dict[str, Any]:
        row = self._fetch_one(table, item_id)
        if not row:
            raise ValueError(f"{label} not found")
        return row

    def _record_event(self, conn: sqlite3.Connection, run_id: str, event: str, data: dict[str, Any] | None = None) -> None:
        conn.execute(
            """
            INSERT INTO research_run_events (run_id, event, data_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, event, self._dump_json(data), self._now()),
        )

    def _update_row(
        self,
        *,
        table: str,
        item_id: str,
        label: str,
        expected_version: int | None,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        row = self._require_one(table, item_id, label)
        self._check_version(row, expected_version)
        updates = dict(fields)
        if not updates:
            return row
        updates["updated_at"] = self._now()
        updates["version"] = int(row["version"]) + 1
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {table} SET {assignments} WHERE id = ?",
                (*updates.values(), item_id),
            )
        return self._require_one(table, item_id, label)

    def _soft_delete(self, table: str, item_id: str, label: str, expected_version: int | None) -> bool:
        row = self._require_one(table, item_id, label)
        self._check_version(row, expected_version)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {table} SET deleted = 1, updated_at = ?, version = ? WHERE id = ?",
                (self._now(), int(row["version"]) + 1, item_id),
            )
        return True

    @staticmethod
    def _normalize_session(row: dict[str, Any]) -> dict[str, Any]:
        return normalize_research_record("local", "session", row)

    @staticmethod
    def _normalize_run(row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        for source_key, target_key in (
            ("limits_json", "limits"),
            ("provider_overrides_json", "provider_overrides"),
            ("chat_handoff_json", "chat_handoff"),
            ("follow_up_json", "follow_up"),
        ):
            payload[target_key] = LocalResearchService._load_json(payload.pop(source_key, None))
        return normalize_research_record("local", "run", payload)

    @staticmethod
    def _normalize_artifact(row: dict[str, Any]) -> dict[str, Any]:
        content = row.get("content_text")
        if content is None:
            content = LocalResearchService._load_json(row.get("content_json"))
        return {
            "artifact_name": row["artifact_name"],
            "content_type": row["content_type"],
            "content": content,
        }

    @staticmethod
    def _normalize_event(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "event": row["event"],
            "data": LocalResearchService._load_json(row["data_json"]),
            "created_at": row["created_at"],
        }

    def create_session(self, *, title: str, query: str, notes: str | None = None, **kwargs: Any) -> dict[str, Any]:
        session_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_sessions (
                    id, title, query, status, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    title,
                    query,
                    kwargs.get("status") or "active",
                    notes,
                    now,
                    now,
                ),
            )
        return self._normalize_session(self._require_one("research_sessions", session_id, "research session"))

    def list_sessions(self, *, limit: int = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM research_sessions WHERE deleted = 0"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._normalize_session(dict(row)) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("research_sessions", session_id)
        return self._normalize_session(row) if row else None

    def update_session(
        self,
        session_id: str,
        *,
        expected_version: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        allowed = {key: value for key, value in fields.items() if key in {"title", "query", "status", "notes"}}
        row = self._update_row(
            table="research_sessions",
            item_id=session_id,
            label="research session",
            expected_version=expected_version,
            fields=allowed,
        )
        return self._normalize_session(row)

    def delete_session(self, session_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("research_sessions", session_id, "research session", expected_version)

    def launch_run(
        self,
        *,
        session_id: str | None = None,
        query: str | None = None,
        source_policy: str = "balanced",
        autonomy_mode: str = "checkpointed",
        limits_json: dict[str, Any] | None = None,
        provider_overrides: dict[str, Any] | None = None,
        chat_handoff: dict[str, Any] | None = None,
        follow_up: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        session = self._require_one("research_sessions", session_id, "research session") if session_id else None
        run_query = query or (session["query"] if session else None)
        if not run_query:
            raise ValueError("query is required")
        run_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_runs (
                    id, session_id, query, status, phase, control_state,
                    progress_percent, progress_message, source_policy, autonomy_mode,
                    limits_json, provider_overrides_json, chat_handoff_json, follow_up_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    run_query,
                    kwargs.get("status") or "running",
                    kwargs.get("phase") or "local_planning",
                    kwargs.get("control_state") or "running",
                    kwargs.get("progress_percent"),
                    kwargs.get("progress_message"),
                    source_policy,
                    autonomy_mode,
                    self._dump_json(limits_json),
                    self._dump_json(provider_overrides),
                    self._dump_json(chat_handoff),
                    self._dump_json(follow_up),
                    now,
                    now,
                ),
            )
            self._record_event(conn, run_id, "created")
        return self._normalize_run(self._require_one("research_runs", run_id, "research run"))

    def list_runs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        session_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM research_runs WHERE deleted = 0"
        params: list[Any] = []
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._normalize_run(dict(row)) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("research_runs", run_id)
        return self._normalize_run(row) if row else None

    def delete_run(self, run_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("research_runs", run_id, "research run", expected_version)

    def _update_run_state(self, run_id: str, event: str, **fields: Any) -> dict[str, Any]:
        row = self._require_one("research_runs", run_id, "research run")
        updates = dict(fields)
        updates["updated_at"] = self._now()
        updates["version"] = int(row["version"]) + 1
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE research_runs SET {assignments} WHERE id = ?",
                (*updates.values(), run_id),
            )
            self._record_event(conn, run_id, event)
        updated = self._normalize_run(self._require_one("research_runs", run_id, "research run"))
        self._dispatch_terminal_run_notification(updated)
        return updated

    def pause_run(self, run_id: str) -> dict[str, Any]:
        return self._update_run_state(run_id, "paused", control_state="paused")

    def resume_run(self, run_id: str) -> dict[str, Any]:
        return self._update_run_state(run_id, "resumed", control_state="running")

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._update_run_state(run_id, "cancelled", status="cancelled", control_state="cancelled")

    def complete_run(self, run_id: str, *, progress_message: str | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "status": "completed",
            "control_state": "completed",
            "phase": "completed",
            "progress_percent": 100.0,
        }
        if progress_message is not None:
            fields["progress_message"] = progress_message
        return self._update_run_state(run_id, "completed", **fields)

    def fail_run(self, run_id: str, *, error_msg: str | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "status": "failed",
            "control_state": "failed",
            "phase": "failed",
        }
        if error_msg is not None:
            fields["progress_message"] = error_msg
        return self._update_run_state(run_id, "failed", **fields)

    def _dispatch_terminal_run_notification(self, run: dict[str, Any]) -> None:
        status = str(run.get("status") or "").strip()
        if status not in {"completed", "failed", "cancelled"}:
            return
        dispatcher = self.notification_dispatcher
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return
        severity = "information"
        if status == "failed":
            severity = "error"
        elif status == "cancelled":
            severity = "warning"
        dispatch(
            app=self.notification_app,
            category="research",
            title=f"Research run {status}",
            message=str(run.get("query") or run.get("progress_message") or run.get("id") or "Research run updated"),
            severity=severity,
            source_backend="local",
            source_entity_kind="research_run",
            source_entity_id=str(run.get("id")),
            payload={
                "run_id": run.get("id"),
                "session_id": run.get("session_id"),
                "status": run.get("status"),
                "control_state": run.get("control_state"),
                "query": run.get("query"),
            },
        )

    def save_artifact(
        self,
        run_id: str,
        *,
        artifact_name: str,
        content_type: str,
        content: Any,
    ) -> dict[str, Any]:
        self._require_one("research_runs", run_id, "research run")
        content_text = content if isinstance(content, str) else None
        content_json = None if isinstance(content, str) else json.dumps(content, sort_keys=True)
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_artifacts (
                    run_id, artifact_name, content_type, content_json, content_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, artifact_name) DO UPDATE SET
                    content_type = excluded.content_type,
                    content_json = excluded.content_json,
                    content_text = excluded.content_text,
                    created_at = excluded.created_at
                """,
                (run_id, artifact_name, content_type, content_json, content_text, now),
            )
            self._record_event(conn, run_id, "artifact_saved", {"artifact_name": artifact_name})
        return self.get_artifact(run_id, artifact_name)

    def get_artifact(self, run_id: str, artifact_name: str) -> dict[str, Any] | None:
        self._require_one("research_runs", run_id, "research run")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM research_artifacts
                WHERE run_id = ? AND artifact_name = ?
                """,
                (run_id, artifact_name),
            ).fetchone()
        return self._normalize_artifact(dict(row)) if row else None

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        self._require_one("research_runs", run_id, "research run")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM research_artifacts
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._normalize_artifact(dict(row)) for row in rows]

    def get_bundle(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("research run not found")
        return {"run": run, "artifacts": self.list_artifacts(run_id)}

    def list_run_events(self, run_id: str, *, after_id: int = 0) -> Iterable[dict[str, Any]]:
        self._require_one("research_runs", run_id, "research run")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM research_run_events
                WHERE run_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (run_id, after_id),
            ).fetchall()
        return [self._normalize_event(dict(row)) for row in rows]
