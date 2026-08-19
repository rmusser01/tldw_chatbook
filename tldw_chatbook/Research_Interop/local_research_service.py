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

__all__ = ["LocalResearchService", "LeaseBudgetExhausted", "TERMINAL_RUN_STATUSES"]

#: Statuses from which a run cannot be claimed or further executed. The
#: single source of truth: ``local_research_engine.py`` imports this rather
#: than keeping its own copy, so the two modules cannot drift apart (task-3
#: review finding 1). Defined here, not in the engine, so this service has
#: no dependency on the engine module -- ``claim_run`` needs the set and the
#: service must not import the engine to get it.
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


class LeaseBudgetExhausted(Exception):
    """Raised by ``claim_run`` when a run's crash-retry budget is spent.

    Reclaiming an EXPIRED lease counts against ``max_attempts``. This is
    distinct from a normal claim refusal (another executor holds a LIVE
    lease, signalled by ``claim_run`` returning ``None``): once the retry
    budget itself is spent, the run's executor keeps dying rather than
    merely losing a race, and the caller must fail the run instead of
    leaving it ``status=running`` forever and unclaimable by anyone
    (task-18060 review finding 1).

    Attributes:
        run_id: The run whose retry budget is exhausted.
        attempts: How many times the run's lease was claimed and then
            abandoned (left to expire unrenewed and unreleased) before this
            attempt.
        max_attempts: The configured budget that was reached.
    """

    def __init__(self, run_id: str, *, attempts: int, max_attempts: int) -> None:
        super().__init__(
            f"research run {run_id} exhausted its lease retry budget "
            f"({attempts} of {max_attempts} allowed attempts)"
        )
        self.run_id = run_id
        self.attempts = attempts
        self.max_attempts = max_attempts


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
        #: Degraded, per-instance lease bookkeeping used only in external-db
        #: mode (task-3 review finding 6): ``self.db`` is an arbitrary
        #: external object with no lease columns and no lease API of its
        #: own (nothing in production constructs the service this way
        #: today), so a real, persisted, cross-process lease cannot be
        #: implemented against it. Rather than raise (breaking
        #: ``execute_run``'s now-unconditional claim outright) this map
        #: gives external-db mode a real, in-memory single-executor
        #: exclusion for the lifetime of THIS service instance -- it stops
        #: two engines sharing one instance from double-executing a run,
        #: but confers no protection across process or instance boundaries.
        #: ``run_id -> {"lease_id", "worker_id", "leased_until", "attempts"}``.
        self._external_leases: dict[str, dict[str, Any]] = {}
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
        self,
        run_id: str,
        event: str,
        *,
        lease_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Apply a run-state transition, optionally lease-conditional.

        When ``lease_id`` is given, the UPDATE's WHERE clause matches it
        directly (single atomic statement -- not a separate check followed
        by an unconditional write), so a displaced executor whose lease no
        longer matches gets a no-op instead of a race with whoever holds
        the run now (task-3 review finding 4: this closes the
        check-then-act gap a standalone ``holds_lease()`` pre-check leaves
        between the check and the write).

        Args:
            run_id: The run to transition.
            event: Event name recorded when the write lands.
            lease_id: When provided, the write only lands if this still
                matches the run's current ``lease_id``. Omit (the default)
                for callers that transition run state independent of lease
                ownership (``pause_run``/``resume_run``/``cancel_run``, and
                any ``complete_run``/``fail_run`` call made on a path that
                never held a lease).
            **fields: Columns to set.

        Returns:
            The run's record: updated when the write landed, or the
            CURRENT unmodified record when a ``lease_id`` was given and did
            not match (a losing write) -- mirroring
            ``_quiet_lease_lost_return``'s "return the truth, not a lie"
            contract elsewhere in the lease design.
        """
        row = self._require_one("research_runs", run_id, "research run")
        updates = dict(fields)
        updates["updated_at"] = self._now()
        updates["version"] = int(row["version"]) + 1
        assignments = ", ".join(f"{key} = ?" for key in updates)
        sql = f"UPDATE research_runs SET {assignments} WHERE id = ?"
        params: list[Any] = [*updates.values(), run_id]
        if lease_id is not None:
            sql += " AND lease_id = ?"
            params.append(lease_id)
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            landed = cursor.rowcount == 1
            if landed:
                self._record_event(conn, run_id, event)
        updated = self._normalize_run(
            self._require_one("research_runs", run_id, "research run")
        )
        if landed:
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
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_seconds: float,
        max_attempts: int = 3,
    ) -> str | None:
        """Take the execution lease on a run, or decline it.

        The claim is atomic: the UPDATE matches only when no live lease
        exists, so two racing executors cannot both succeed -- that WIN/LOSE
        decision between racing claimants is exactly what the single UPDATE
        below decides, and a losing race still returns None. Every
        successful claim counts against ``max_attempts`` (the run's total
        attempt budget), but the budget is only ENFORCED when reclaiming an
        EXPIRED lease -- a run's very first claim always succeeds. A run
        whose executor keeps dying is broken rather than slow, and must
        stop being retried once the budget is spent (task-18060, following
        the server job manager's retry budget) -- signalled by raising
        ``LeaseBudgetExhausted`` rather than returning None, so a caller
        cannot mistake "the budget is spent" for "someone else holds it
        live" (review finding 1: those two cases need different responses --
        the first must fail the run, the second must leave it alone). A
        clean ``release_lease`` resets the counter to 0, since a run that
        was voluntarily released was not abandoned -- so the budget tracks
        CONSECUTIVE abandonments (crashes that leave the lease to expire),
        not the run's lifetime claim count.

        Whether a claim is a "reclaim" for budget purposes depends on the
        previous lease being EXPIRED, not merely present: a healthy
        executor's still-live lease must never trip the budget check for a
        second, merely-racing claimant -- that claimant is simply declined
        below by the atomic UPDATE's WHERE clause. Reusing the same ``now``
        snapshot for both the reclaim check and the UPDATE's own
        ``leased_until <= ?`` comparison guarantees the two judgments of
        "is this lease live" cannot disagree with each other.

        A run in a ``TERMINAL_RUN_STATUSES`` status can never be claimed
        (task-3 review finding 1): the caller's own terminal check (e.g.
        ``execute_run``'s pre-flight ``ValueError``) runs BEFORE this call,
        so a cancellation or completion landing in the gap between that
        check and this one would otherwise let a finished run be claimed
        and re-executed. The status condition lives in the SAME atomic
        UPDATE as the lease-expiry check, so the win/lose decision and the
        terminal check cannot themselves race apart.

        Args:
            run_id: The run to claim.
            worker_id: Identifies the claiming executor.
            lease_seconds: How long the lease is valid without renewal.
            max_attempts: How many times, in total, a run may be claimed
                (its first claim plus subsequent reclaims of an expired
                lease) before further reclaims are refused.

        Returns:
            A new lease id when the claim succeeded, otherwise None (either
            another executor currently holds a live lease, or the run is
            already terminal).

        Raises:
            LeaseBudgetExhausted: When reclaiming an EXPIRED lease would
                exceed ``max_attempts`` -- the caller must fail the run
                rather than treat this like a routine claim refusal.
        """
        if self._uses_external_db:
            return self._claim_run_external(
                run_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
            )
        row = self._require_one("research_runs", run_id, "research run")
        if str(row["status"] or "") in TERMINAL_RUN_STATUSES:
            # Nothing to reclaim and nothing to budget-check -- a terminal
            # run is simply unclaimable, full stop. The atomic UPDATE below
            # enforces this for real; this is only the cheap early-out so a
            # terminal run never even reaches the retry-budget check (which
            # would otherwise misfire "exhausted" on a run that was never
            # abandoned, just finished).
            return None
        previous = row["leased_until"] if "leased_until" in row.keys() else None
        attempts = int(row["lease_attempts"] or 0) if "lease_attempts" in row.keys() else 0
        now = self._now()
        reclaiming = previous is not None and str(previous) <= now
        if reclaiming and attempts >= int(max_attempts):
            raise LeaseBudgetExhausted(
                run_id, attempts=attempts, max_attempts=int(max_attempts)
            )
        lease_id = uuid.uuid4().hex
        expires = self._timestamp_after(lease_seconds)
        next_attempts = attempts + 1
        status_placeholders = ", ".join("?" for _ in TERMINAL_RUN_STATUSES)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE research_runs
                   SET lease_owner = ?, lease_id = ?, leased_until = ?,
                       lease_attempts = ?, updated_at = ?
                 WHERE id = ?
                   AND (leased_until IS NULL OR leased_until <= ?)
                   AND status NOT IN ({status_placeholders})
                """,
                (
                    worker_id,
                    lease_id,
                    expires,
                    next_attempts,
                    now,
                    run_id,
                    now,
                    *sorted(TERMINAL_RUN_STATUSES),
                ),
            )
            if cursor.rowcount != 1:
                return None
        return lease_id

    def _claim_run_external(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_seconds: float,
        max_attempts: int,
    ) -> str | None:
        """``claim_run``'s degraded, in-memory counterpart for external-db
        mode (task-3 review finding 6). See ``self._external_leases``'s
        docstring in ``__init__`` for why this cannot be a real, persisted
        lease. Mirrors ``claim_run``'s semantics (terminal refusal, live-
        lease decline, expired-lease reclaim budget) against the in-memory
        map instead of a SQL UPDATE.
        """
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("research run not found")
        if str(run.get("status") or "") in TERMINAL_RUN_STATUSES:
            return None
        now = self._now()
        state = self._external_leases.get(run_id)
        attempts = int(state["attempts"]) if state is not None else 0
        live = state is not None and str(state["leased_until"]) > now
        if live:
            return None
        reclaiming = state is not None
        if reclaiming and attempts >= int(max_attempts):
            raise LeaseBudgetExhausted(
                run_id, attempts=attempts, max_attempts=int(max_attempts)
            )
        lease_id = uuid.uuid4().hex
        self._external_leases[run_id] = {
            "lease_id": lease_id,
            "worker_id": worker_id,
            "leased_until": self._timestamp_after(lease_seconds),
            "attempts": attempts + 1,
        }
        return lease_id

    def renew_lease(
        self, run_id: str, *, lease_id: str, lease_seconds: float
    ) -> bool:
        """Extend a lease the caller still holds.

        The lease id is a fencing token: a worker that stalled past its lease
        and was taken over still matches on worker id, so matching on the id
        alone would let it act on a run it no longer owns (task-18060).

        The lease must also still be LIVE (task-3 review finding 3): matching
        on ``lease_id`` alone lets a worker whose lease already expired
        extend it again as long as nobody has taken over YET, contradicting
        takeover -- a stalled worker could resurrect a claim it had already
        lost. ``now`` is computed once and reused for both the UPDATE's
        expiry comparison and ``updated_at``, the same pattern ``claim_run``
        uses for its own comparisons.

        Args:
            run_id: The leased run.
            lease_id: The token returned by ``claim_run``.
            lease_seconds: How much longer the lease should be valid.

        Returns:
            True when the lease was extended, False when it was lost (wrong
            id, or the lease had already expired).
        """
        if self._uses_external_db:
            return self._renew_lease_external(
                run_id, lease_id=lease_id, lease_seconds=lease_seconds
            )
        now = self._now()
        expires = self._timestamp_after(lease_seconds)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE research_runs
                   SET leased_until = ?, updated_at = ?
                 WHERE id = ? AND lease_id = ? AND leased_until > ?
                """,
                (expires, now, run_id, lease_id, now),
            )
            return cursor.rowcount == 1

    def _renew_lease_external(
        self, run_id: str, *, lease_id: str, lease_seconds: float
    ) -> bool:
        """``renew_lease``'s degraded, in-memory counterpart for
        external-db mode (task-3 review finding 6)."""
        state = self._external_leases.get(run_id)
        if state is None or state["lease_id"] != lease_id:
            return False
        if str(state["leased_until"]) <= self._now():
            return False
        state["leased_until"] = self._timestamp_after(lease_seconds)
        return True

    def release_lease(self, run_id: str, *, lease_id: str) -> bool:
        """Drop a lease the caller holds so another executor may claim it.

        Only a release while the lease is still LIVE is a clean hand-off:
        it clears the lease columns and resets the crash-retry budget
        (PR-1822 review follow-up). Releasing a lease that had already
        EXPIRED is an abandonment being acknowledged after the fact, not a
        clean hand-off -- matching ``lease_id`` alone let a systematically
        stalling-but-alive executor loop claim -> expire -> release forever
        without ever spending the budget, defeating the "fail a run whose
        executor keeps dying" contract (AC #1b). An expired release
        therefore leaves the lease record exactly as a crashed executor
        would (expired, still on the books): the next claim is a RECLAIM,
        the budget check applies, and the run stays free for the next
        claimant either way -- an expired lease is already claimable.

        Args:
            run_id: The leased run.
            lease_id: The token returned by ``claim_run``.

        Returns:
            True when the lease was the caller's to release (live: cleared;
            expired: left in place as an abandonment), False when it was
            already lost to a takeover.
        """
        if self._uses_external_db:
            return self._release_lease_external(run_id, lease_id=lease_id)
        now = self._now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT leased_until FROM research_runs WHERE id = ? AND lease_id = ?",
                (run_id, lease_id),
            ).fetchone()
            if row is None:
                return False
            if str(row["leased_until"] or "") <= now:
                # Already expired: leave the record so the next claim counts
                # this as the abandonment it was.
                return True
            cursor = conn.execute(
                """
                UPDATE research_runs
                   SET lease_owner = NULL, lease_id = NULL, leased_until = NULL,
                       lease_attempts = 0, updated_at = ?
                 WHERE id = ? AND lease_id = ?
                """,
                (now, run_id, lease_id),
            )
            return cursor.rowcount == 1

    def _release_lease_external(self, run_id: str, *, lease_id: str) -> bool:
        """``release_lease``'s degraded, in-memory counterpart for
        external-db mode (task-3 review finding 6). Mirrors the SQLite
        path's live/expired split: a live release deletes the entry (the
        next claim starts fresh); an expired release leaves it so the next
        claim counts the abandonment against the retry budget."""
        state = self._external_leases.get(run_id)
        if state is None or state["lease_id"] != lease_id:
            return False
        if str(state["leased_until"]) <= self._now():
            return True
        del self._external_leases[run_id]
        return True

    def holds_lease(self, run_id: str, *, lease_id: str) -> bool:
        """Whether ``lease_id`` is still the live lease on the run.

        Args:
            run_id: The run to check.
            lease_id: The token returned by ``claim_run``.

        Returns:
            True when the token matches an unexpired lease.
        """
        if self._uses_external_db:
            return self._holds_lease_external(run_id, lease_id=lease_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT lease_id, leased_until FROM research_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None or row["lease_id"] != lease_id:
            return False
        return str(row["leased_until"] or "") > self._now()

    def _holds_lease_external(self, run_id: str, *, lease_id: str) -> bool:
        """``holds_lease``'s degraded, in-memory counterpart for
        external-db mode (task-3 review finding 6)."""
        state = self._external_leases.get(run_id)
        if state is None or state["lease_id"] != lease_id:
            return False
        return str(state["leased_until"]) > self._now()

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
        self,
        run_id: str,
        *,
        progress_message: str | None = None,
        lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a run to its terminal "completed" state.

        Args:
            run_id: The run to complete.
            progress_message: Optional final status message.
            lease_id: When provided (SQLite-backed mode only -- external-db
                mode has no lease concept and ignores this), the write only
                lands if this still matches the run's current lease
                (task-3 review finding 4). The engine passes its own lease
                id on every call it makes while holding one; a caller on a
                path that never held a lease omits it, so its write is
                unconditional.

        Returns:
            The run's record: "completed" when the write landed, or the
            CURRENT unmodified record when a ``lease_id`` was given and no
            longer matched (a displaced executor's write is a no-op, never
            an overwrite of whoever holds the run now).
        """
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
        return self._update_run_state(
            run_id, "completed", lease_id=lease_id, **fields
        )

    def fail_run(
        self,
        run_id: str,
        *,
        error_msg: str | None = None,
        lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a run to its terminal "failed" state.

        Args:
            run_id: The run to fail.
            error_msg: Optional failure message.
            lease_id: When provided (SQLite-backed mode only -- external-db
                mode has no lease concept and ignores this), the write only
                lands if this still matches the run's current lease
                (task-3 review finding 4). The engine passes its own lease
                id on every call it makes while holding one; a caller on a
                path that never held a lease (e.g. a spent retry budget,
                where ``claim_run`` raised before any lease was granted)
                omits it, so its write is unconditional.

        Returns:
            The run's record: "failed" when the write landed, or the
            CURRENT unmodified record when a ``lease_id`` was given and no
            longer matched (a displaced executor's write is a no-op, never
            an overwrite of whoever holds the run now).
        """
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
        return self._update_run_state(run_id, "failed", lease_id=lease_id, **fields)

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

    def record_run_event(
        self, run_id: str, event: str, data: dict[str, Any] | None = None
    ) -> None:
        """Append an event to the run's stream WITHOUT touching run state.

        PR-1822 review follow-up: ``update_run_progress`` writes both an
        event and the run row, so it is reserved for the lease holder (a
        non-holder's call stomps the live executor's progress message and
        bumps the version mid-flight). The event log is append-only and
        overwrites nothing, so an observer -- e.g. an executor that was
        DECLINED a lease -- may record what it observed without violating
        the single-writer principle.

        Args:
            run_id: The run the event concerns.
            event: Event name.
            data: Optional event payload.
        """
        if self._uses_external_db:
            return
        self._require_one("research_runs", run_id, "research run")
        with self._connect() as conn:
            self._record_event(conn, run_id, event, data)

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
