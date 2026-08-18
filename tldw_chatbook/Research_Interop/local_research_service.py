"""SQLite-backed local research session/run service."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

from .research_normalizers import (
    ResearchRecord,
    ResearchRecordList,
    normalize_research_record,
)


class LocalResearchService:
    """Local-first persistence for research sessions, runs, events, and artifacts."""

    def __init__(
        self,
        db_path: str | Path | Any,
        *,
        notification_dispatcher: Any | None = None,
        notification_dispatch_service: Any | None = None,
        notification_app: Any | None = None,
    ):
        self.db = None
        self._memory_conn: sqlite3.Connection | None = None
        try:
            self.db_path = Path(db_path)
        except TypeError:
            self.db = db_path
            self.db_path = None
        self.notification_dispatcher = (
            notification_dispatcher or notification_dispatch_service
        )
        self.notification_app = notification_app
        if self.db_path is not None:
            self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self.db_path is None:
            raise RuntimeError("Path-backed research database is not configured.")
        if str(self.db_path) == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = connect_private_sqlite(
                    "research.local",
                    self.db_path,
                )
                self._memory_conn.row_factory = sqlite3.Row
                # synchronous is harmless (and a no-op performance-wise) on an
                # in-memory database; set for uniformity with the file-backed
                # branch below (task-15465).
                self._memory_conn.execute("PRAGMA synchronous = NORMAL")
            return self._memory_conn
        conn = connect_private_sqlite("research.local", self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local research
        # session/run store) and avoids an fsync per commit. This DB opens a
        # fresh connection per operation, so synchronous must be re-applied
        # on every open, not just the first (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def close(self) -> None:
        """Close the persistent in-memory connection, when present."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    @staticmethod
    def _format_timestamp(moment: datetime) -> str:
        """Render a UTC ``datetime`` in the one format lease timestamps use.

        ``claim_run``'s atomicity depends on a plain string comparison
        between the persisted ``leased_until`` and "now" (task-18060), so
        every timestamp that can appear on either side of that comparison
        MUST be produced by this method. ``timespec="microseconds"`` pins
        the fractional-seconds field so it is never dropped when the
        microsecond value happens to be zero -- plain ``isoformat()``
        omits it in that case, which otherwise makes a whole-second
        timestamp sort *below* one with a non-zero fraction (``'.'`` sorts
        below alphanumerics) and lets a live lease be claimed twice.

        Args:
            moment: A timezone-aware datetime. Converted to UTC before
                formatting.

        Returns:
            An ISO-8601 string with microsecond precision and a trailing
            ``Z``, e.g. ``"2026-08-18T08:31:01.000000Z"``.
        """
        return (
            moment.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _now() -> str:
        """Current UTC time in the shared lease-comparable timestamp format.

        Returns:
            An ISO-8601 UTC timestamp string produced by
            ``_format_timestamp``.
        """
        return LocalResearchService._format_timestamp(datetime.now(timezone.utc))

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    @property
    def _uses_external_db(self) -> bool:
        return self.db is not None

    @staticmethod
    def _awaitable_list(items: Iterable[Any]) -> ResearchRecordList:
        return ResearchRecordList(items)

    @staticmethod
    def _as_local_run(record: dict[str, Any]) -> ResearchRecord:
        payload = dict(record)
        payload.setdefault("source", "local")
        payload.setdefault("record_type", "research_run")
        payload.setdefault("record_id", f"local:research_run:{payload.get('id')}")
        return ResearchRecord(payload)

    @staticmethod
    def _as_local_artifact(
        record: dict[str, Any], *, run_id: str | None = None
    ) -> ResearchRecord:
        payload = dict(record)
        if run_id is not None:
            payload.setdefault("run_id", run_id)
        payload.setdefault("source", "local")
        payload.setdefault("record_type", "research_artifact")
        payload.setdefault(
            "record_id",
            f"local:research_artifact:{payload.get('run_id')}:{payload.get('artifact_name') or payload.get('id')}",
        )
        return ResearchRecord(payload)

    def _dispatch_external_run_notification(
        self, run: dict[str, Any], *, event: str
    ) -> None:
        dispatcher = self.notification_dispatcher
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return
        status = str(run.get("status") or event)
        severity = "information"
        if status == "failed":
            severity = "error"
        elif status == "cancelled":
            severity = "warning"
        dispatch(
            app=self.notification_app,
            category="research",
            title=f"Local research session {event}",
            message=str(
                run.get("query")
                or run.get("progress_message")
                or run.get("id")
                or "Research session updated"
            ),
            severity=severity,
            source_backend="local",
            source_entity_kind="research_run",
            source_entity_id=str(run.get("id")),
            payload={
                "run_id": run.get("id"),
                "status": run.get("status"),
                "control_state": run.get("control_state"),
                "query": run.get("query"),
            },
        )

    @staticmethod
    def _check_version(row: dict[str, Any], expected_version: int | None) -> None:
        if expected_version is not None and int(row["version"]) != int(
            expected_version
        ):
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
                CREATE TABLE IF NOT EXISTS research_checkpoints (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    resolution TEXT,
                    proposed_payload_json TEXT NOT NULL DEFAULT '{}',
                    user_patch_payload_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
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
            self._ensure_run_lease_columns(conn)

    #: Columns added after the original schema shipped. CREATE TABLE IF NOT
    #: EXISTS never revisits an existing database, so each one is applied by
    #: an idempotent ALTER guarded on PRAGMA table_info (task-18060).
    _RUN_COLUMN_ADDITIONS = (
        ("lease_owner", "TEXT"),
        ("lease_id", "TEXT"),
        ("leased_until", "TEXT"),
        ("lease_attempts", "INTEGER NOT NULL DEFAULT 0"),
    )

    def _ensure_run_lease_columns(self, conn: sqlite3.Connection) -> None:
        """Add lease columns to research_runs when they are absent.

        Args:
            conn: An open connection inside the caller's transaction.
        """
        existing = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(research_runs)").fetchall()
        }
        for column, declaration in self._RUN_COLUMN_ADDITIONS:
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE research_runs ADD COLUMN {column} {declaration}"
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

    def _record_event(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
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

    def _soft_delete(
        self, table: str, item_id: str, label: str, expected_version: int | None
    ) -> bool:
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
            payload[target_key] = LocalResearchService._load_json(
                payload.pop(source_key, None)
            )
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
    def _normalize_checkpoint(row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload["proposed_payload"] = LocalResearchService._load_json(
            payload.pop("proposed_payload_json", None)
        )
        payload["user_patch"] = LocalResearchService._load_json(
            payload.pop("user_patch_payload_json", None)
        )
        return payload

    @staticmethod
    def _normalize_event(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "event": row["event"],
            "data": LocalResearchService._load_json(row["data_json"]),
            "created_at": row["created_at"],
        }

    def create_session(
        self, *, title: str, query: str, notes: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
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
        return self._normalize_session(
            self._require_one("research_sessions", session_id, "research session")
        )

    def list_sessions(
        self, *, limit: int = 100, offset: int = 0, status: str | None = None
    ) -> list[dict[str, Any]]:
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
        allowed = {
            key: value
            for key, value in fields.items()
            if key in {"title", "query", "status", "notes"}
        }
        row = self._update_row(
            table="research_sessions",
            item_id=session_id,
            label="research session",
            expected_version=expected_version,
            fields=allowed,
        )
        return self._normalize_session(row)

    def delete_session(
        self, session_id: str, *, expected_version: int | None = None
    ) -> bool:
        return self._soft_delete(
            "research_sessions", session_id, "research session", expected_version
        )

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
        **kwargs: Any,
    ) -> ResearchRecord:
        """Create a draft local research run for the run-centric interop API."""
        if self._uses_external_db:
            run = self.db.create_run(
                query=query,
                source_policy=source_policy,
                autonomy_mode=autonomy_mode,
                limits_json=limits_json,
                provider_overrides=provider_overrides,
                chat_handoff=chat_handoff,
                follow_up=follow_up,
                id=kwargs.get("id"),
            )
            record = self._as_local_run(run)
            self._dispatch_external_run_notification(record, event="created")
            return record
        return ResearchRecord(
            self.launch_run(
                query=query,
                source_policy=source_policy,
                autonomy_mode=autonomy_mode,
                limits_json=limits_json,
                provider_overrides=provider_overrides,
                chat_handoff=chat_handoff,
                follow_up=follow_up,
                status=kwargs.get("status") or "draft",
                phase=kwargs.get("phase") or "planning",
                control_state=kwargs.get("control_state") or "paused",
                id=kwargs.get("id"),
            )
        )

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
        session = (
            self._require_one("research_sessions", session_id, "research session")
            if session_id
            else None
        )
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
        return self._normalize_run(
            self._require_one("research_runs", run_id, "research run")
        )

    def list_runs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        session_id: str | None = None,
        status: str | None = None,
    ) -> ResearchRecordList:
        if self._uses_external_db:
            runs = self.db.list_runs(limit=limit)
            if status:
                runs = [run for run in runs if run.get("status") == status]
            if offset:
                runs = runs[offset:]
            return self._awaitable_list(self._as_local_run(run) for run in runs)
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
        return self._awaitable_list(self._normalize_run(dict(row)) for row in rows)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if self._uses_external_db:
            try:
                return self._as_local_run(self.db.get_run(run_id))
            except KeyError:
                return None
        row = self._fetch_one("research_runs", run_id)
        return self._normalize_run(row) if row else None

    def delete_run(self, run_id: str, *, expected_version: int | None = None) -> bool:
        if self._uses_external_db:
            current = self.db.get_run(run_id)
            if expected_version is not None and int(current.get("version", 1)) != int(
                expected_version
            ):
                raise ValueError("version conflict")
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE research_runs SET deleted = 1, updated_at = ? WHERE id = ?",
                    (self._now(), run_id),
                )
            return True
        return self._soft_delete(
            "research_runs", run_id, "research run", expected_version
        )

    def _update_run_state(
        self, run_id: str, event: str, **fields: Any
    ) -> dict[str, Any]:
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
        updated = self._normalize_run(
            self._require_one("research_runs", run_id, "research run")
        )
        self._dispatch_terminal_run_notification(updated)
        return updated

    def _timestamp_after(self, seconds: float) -> str:
        """An ISO timestamp ``seconds`` in the future, in the same format as
        ``_now`` so string comparison orders correctly.

        Args:
            seconds: Offset from the current time, in seconds. Negative
                values clamp to 0 so a non-positive lease duration yields
                an already-expired timestamp rather than one in the past
                relative to itself.

        Returns:
            An ISO-8601 UTC timestamp string produced by
            ``_format_timestamp``, guaranteed to compare correctly against
            ``_now()``'s output.
        """
        moment = datetime.now(timezone.utc) + timedelta(
            seconds=max(0.0, float(seconds))
        )
        return self._format_timestamp(moment)

    def claim_run(
        self, run_id: str, *, worker_id: str, lease_seconds: float
    ) -> str | None:
        """Take the execution lease on a run, or decline it.

        The claim is atomic: the UPDATE only matches when no live lease
        exists, so two racing executors cannot both succeed (task-18060).

        Args:
            run_id: The run to claim.
            worker_id: Identifies the claiming executor.
            lease_seconds: How long the lease is valid without renewal.

        Returns:
            A new lease id when the claim succeeded, otherwise None.
        """
        self._require_one("research_runs", run_id, "research run")
        lease_id = uuid.uuid4().hex
        now = self._now()
        expires = self._timestamp_after(lease_seconds)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE research_runs
                   SET lease_owner = ?, lease_id = ?, leased_until = ?,
                       updated_at = ?
                 WHERE id = ?
                   AND (leased_until IS NULL OR leased_until <= ?)
                """,
                (worker_id, lease_id, expires, now, run_id, now),
            )
            if cursor.rowcount != 1:
                return None
        return lease_id

    def pause_run(self, run_id: str) -> dict[str, Any]:
        if self._uses_external_db:
            return self._as_local_run(
                self.db.update_run_state(run_id, control_state="paused")
            )
        return self._update_run_state(run_id, "paused", control_state="paused")

    def resume_run(self, run_id: str) -> dict[str, Any]:
        if self._uses_external_db:
            return self._as_local_run(
                self.db.update_run_state(run_id, control_state="running")
            )
        return self._update_run_state(run_id, "resumed", control_state="running")

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        if self._uses_external_db:
            record = self._as_local_run(
                self.db.update_run_state(
                    run_id, status="cancelled", control_state="cancelled"
                )
            )
            self._dispatch_external_run_notification(record, event="cancelled")
            return record
        return self._update_run_state(
            run_id, "cancelled", status="cancelled", control_state="cancelled"
        )

    def complete_run(
        self, run_id: str, *, progress_message: str | None = None
    ) -> dict[str, Any]:
        if self._uses_external_db:
            record = self._as_local_run(
                self.db.update_run_state(
                    run_id,
                    status="completed",
                    phase="completed",
                    control_state="completed",
                    progress_percent=100.0,
                    progress_message=progress_message,
                )
            )
            self._dispatch_external_run_notification(record, event="completed")
            return record
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
        if self._uses_external_db:
            record = self._as_local_run(
                self.db.update_run_state(
                    run_id,
                    status="failed",
                    phase="failed",
                    control_state="failed",
                    progress_message=error_msg,
                )
            )
            self._dispatch_external_run_notification(record, event="failed")
            return record
        fields: dict[str, Any] = {
            "status": "failed",
            "control_state": "failed",
            "phase": "failed",
        }
        if error_msg is not None:
            fields["progress_message"] = error_msg
        return self._update_run_state(run_id, "failed", **fields)

    # Hardcoded assignment fragments for update_run_progress's external-DB
    # branch (task-16814): the ONLY source of SQL text for those columns.
    _RUN_PROGRESS_FIELD_SQL = {
        "phase": "phase = ?",
        "progress_percent": "progress_percent = ?",
        "progress_message": "progress_message = ?",
        "status": "status = ?",
        "control_state": "control_state = ?",
    }

    # Patch keys allowed per checkpoint type (task-16482; server
    # checkpoint_service parity, scoped to the local engine's phases).
    _CHECKPOINT_PATCH_KEYS = {
        "plan_review": {"limits"},
        "sources_review": {"pinned_source_ids", "dropped_source_ids", "recollect"},
    }

    def create_checkpoint(
        self,
        run_id: str,
        *,
        checkpoint_type: str,
        proposed_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a pending review checkpoint for a run (task-16482)."""
        self._require_one("research_runs", run_id, "research run")
        checkpoint_id = f"chk-{self._new_id()}"
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_checkpoints (
                    id, run_id, checkpoint_type, status,
                    proposed_payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    run_id,
                    checkpoint_type,
                    self._dump_json(proposed_payload),
                    now,
                    now,
                ),
            )
            self._record_event(
                conn,
                run_id,
                "checkpoint_created",
                {"checkpoint_id": checkpoint_id, "checkpoint_type": checkpoint_type},
            )
        return self._normalize_checkpoint(
            self._require_one("research_checkpoints", checkpoint_id, "research checkpoint")
        )

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        self._require_one("research_runs", run_id, "research run")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_checkpoints WHERE run_id = ? ORDER BY rowid ASC",
                (run_id,),
            ).fetchall()
        return [self._normalize_checkpoint(dict(row)) for row in rows]

    def latest_pending_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        for checkpoint in reversed(self.list_checkpoints(run_id)):
            if checkpoint.get("status") == "pending":
                return checkpoint
        return None

    def approved_checkpoint(
        self, run_id: str, checkpoint_type: str
    ) -> dict[str, Any] | None:
        """The most recent APPROVED checkpoint of one type (the engine's
        pass-through signal on re-execution)."""
        for checkpoint in reversed(self.list_checkpoints(run_id)):
            if (
                checkpoint.get("checkpoint_type") == checkpoint_type
                and checkpoint.get("status") == "approved"
            ):
                return checkpoint
        return None

    def patch_and_approve_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        patch_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Approve a pending checkpoint with a type-validated patch
        (task-16482). Raises ValueError on non-pending state, unexpected
        patch keys, or sources patches referencing unknown/overlapping ids.
        """
        self._require_one("research_runs", run_id, "research run")
        row = self._require_one(
            "research_checkpoints", checkpoint_id, "research checkpoint"
        )
        if row.get("run_id") != run_id:
            raise ValueError("research checkpoint belongs to a different run")
        if row.get("status") != "pending":
            raise ValueError(f"research checkpoint {checkpoint_id} is not pending")
        patch = dict(patch_payload or {})
        checkpoint_type = row.get("checkpoint_type")
        allowed = self._CHECKPOINT_PATCH_KEYS.get(checkpoint_type)
        if allowed is None:
            raise ValueError(f"unsupported checkpoint type: {checkpoint_type!r}")
        unexpected = set(patch) - allowed
        if unexpected:
            raise ValueError(
                f"unexpected patch keys for {checkpoint_type}: {sorted(unexpected)}"
            )
        if checkpoint_type == "sources_review":
            proposed_ids = set(
                (self._load_json(row.get("proposed_payload_json")) or {}).get(
                    "source_ids"
                )
                or []
            )
            pinned = set(patch.get("pinned_source_ids") or [])
            dropped = set(patch.get("dropped_source_ids") or [])
            unknown = (pinned | dropped) - proposed_ids
            if unknown:
                raise ValueError(
                    f"patch references ids not in the proposed inventory: {sorted(unknown)}"
                )
            if pinned & dropped:
                raise ValueError(
                    "pinned and dropped source ids must be disjoint"
                )
            recollect = patch.get("recollect")
            if recollect is not None and not isinstance(recollect, dict):
                raise ValueError("recollect patch must be an object")
        updates = {
            "status": "approved",
            "resolution": "approved",
            "user_patch_payload_json": self._dump_json(patch),
            "updated_at": self._now(),
            "version": int(row["version"]) + 1,
        }
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE research_checkpoints SET {assignments} WHERE id = ?",
                (*updates.values(), checkpoint_id),
            )
            self._record_event(
                conn, run_id, "checkpoint_approved", {"checkpoint_id": checkpoint_id}
            )
        return self._normalize_checkpoint(
            self._require_one(
                "research_checkpoints", checkpoint_id, "research checkpoint"
            )
        )

    def update_run_progress(
        self,
        run_id: str,
        *,
        phase: str | None = None,
        progress_percent: float | None = None,
        progress_message: str | None = None,
        status: str | None = None,
        control_state: str | None = None,
        event: str = "progress",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Engine-facing non-terminal transition (task-16322): update phase/
       progress (and optionally status/control when a draft is started)
        while recording an event with the same data the UI will stream.

        Terminal and control transitions (pause/resume/cancel/complete/fail)
        stay on their dedicated methods; this one never dispatches terminal
        notifications because it never sets a terminal status.
        """
        if self._uses_external_db:
            # delete_run's precedent (task-16814): raw statement inside the
            # db's own transaction() context, per the standardized
            # transaction-handling rule.
            fields: dict[str, Any] = {}
            if phase is not None:
                fields["phase"] = phase
            if progress_percent is not None:
                fields["progress_percent"] = progress_percent
            if progress_message is not None:
                fields["progress_message"] = progress_message
            if status is not None:
                fields["status"] = status
            if control_state is not None:
                fields["control_state"] = control_state
            if not fields:
                return self._as_local_run(self.db.get_run(run_id))
            # SQL is never built from variable text: each field maps to a
            # HARDCODED literal assignment fragment, field names only select
            # among the literals, and every value stays parameterized
            # (house rule: no SQL via string interpolation of identifiers).
            try:
                assignments = ", ".join(
                    self._RUN_PROGRESS_FIELD_SQL[key] for key in fields
                )
            except KeyError as exc:
                raise ValueError(
                    f"unsupported run-progress column: {exc.args[0]!r}"
                ) from exc
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE research_runs SET "
                    + assignments
                    + ", updated_at = ?, version = version + 1 WHERE id = ?",
                    (*fields.values(), self._now(), run_id),
                )
            return self._as_local_run(self.db.get_run(run_id))
        fields = {
            key: value
            for key, value in (
                ("phase", phase),
                ("progress_percent", progress_percent),
                ("progress_message", progress_message),
                ("status", status),
                ("control_state", control_state),
            )
            if value is not None
        }
        row = self._require_one("research_runs", run_id, "research run")
        updates = dict(fields)
        updates["updated_at"] = self._now()
        updates["version"] = int(row["version"]) + 1
        assignments = ", ".join(f"{key} = ?" for key in updates)
        event_data = dict(data or {})
        for key in ("phase", "progress_percent"):
            if fields.get(key) is not None:
                event_data.setdefault(key, fields[key])
        with self._connect() as conn:
            conn.execute(
                f"UPDATE research_runs SET {assignments} WHERE id = ?",
                (*updates.values(), run_id),
            )
            self._record_event(conn, run_id, event, event_data or None)
        return self._normalize_run(
            self._require_one("research_runs", run_id, "research run")
        )

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
            message=str(
                run.get("query")
                or run.get("progress_message")
                or run.get("id")
                or "Research run updated"
            ),
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
        if self._uses_external_db:
            return self._as_local_artifact(
                self.db.save_artifact(
                    run_id,
                    artifact_name=artifact_name,
                    content_type=content_type,
                    content=content,
                ),
                run_id=run_id,
            )
        self._require_one("research_runs", run_id, "research run")
        content_text = content if isinstance(content, str) else None
        content_json = (
            None if isinstance(content, str) else json.dumps(content, sort_keys=True)
        )
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
            self._record_event(
                conn, run_id, "artifact_saved", {"artifact_name": artifact_name}
            )
        return self.get_artifact(run_id, artifact_name)

    def get_artifact(self, run_id: str, artifact_name: str) -> dict[str, Any] | None:
        if self._uses_external_db:
            try:
                return self._as_local_artifact(
                    self.db.get_artifact(run_id, artifact_name), run_id=run_id
                )
            except KeyError:
                return None
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
        if self._uses_external_db:
            bundle = self.db.get_bundle(run_id)
            return self._awaitable_list(
                self._as_local_artifact(
                    {
                        "run_id": run_id,
                        "artifact_name": name,
                        "content_type": "application/json"
                        if not isinstance(content, str)
                        else "text/plain",
                        "content": content,
                    },
                    run_id=run_id,
                )
                for name, content in bundle.items()
            )
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
        if self._uses_external_db:
            return ResearchRecord(self.db.get_bundle(run_id))
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("research run not found")
        return {"run": run, "artifacts": self.list_artifacts(run_id)}

    def list_run_events(
        self, run_id: str, *, after_id: int = 0
    ) -> Iterable[dict[str, Any]]:
        if self._uses_external_db:
            return self._awaitable_list(
                self._external_run_events(run_id, after_id=after_id)
            )
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

    def _external_run_events(
        self, run_id: str, *, after_id: int = 0
    ) -> list[ResearchRecord]:
        run = self._as_local_run(self.db.get_run(run_id))
        events: list[ResearchRecord] = [
            ResearchRecord(
                {
                    "event": "snapshot",
                    "id": "1",
                    "data": {"run": run},
                }
            )
        ]
        bundle = self.db.get_bundle(run_id)
        if bundle:
            events.append(
                ResearchRecord(
                    {
                        "event": "bundle",
                        "id": "2",
                        "data": {
                            "artifact_names": sorted(bundle),
                            "bundle": bundle,
                        },
                    }
                )
            )
        return [event for event in events if int(event["id"]) > int(after_id or 0)]

    async def stream_run_events(self, run_id: str, *, after_id: int = 0):
        for event in self.list_run_events(run_id, after_id=after_id):
            yield event
