"""Private goal records sharing AgentRunsDB's durable automatic-work transaction."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from tldw_chatbook.Agents.goal_models import GoalReport, GoalRequest, GoalSnapshot
from tldw_chatbook.DB.automatic_work import _identity

if TYPE_CHECKING:
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

SCHEMA = """
CREATE TABLE IF NOT EXISTS goal_runs (
    id TEXT PRIMARY KEY,
    launch_id TEXT NOT NULL UNIQUE,
    payload_hash TEXT NOT NULL,
    request_json TEXT NOT NULL CHECK (length(CAST(request_json AS BLOB)) <= 131072),
    conversation_id TEXT NOT NULL UNIQUE,
    chain_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_chains(id),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    status TEXT NOT NULL DEFAULT 'starting',
    pause_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS goal_iterations (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    launch_id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL UNIQUE REFERENCES automatic_wake_attempts(id),
    run_id TEXT REFERENCES agent_runs(id),
    status TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    check_results_json TEXT NOT NULL DEFAULT '[]' CHECK (length(CAST(check_results_json AS BLOB)) <= 131072),
    evidence_refs_json TEXT NOT NULL DEFAULT '[]' CHECK (length(CAST(evidence_refs_json AS BLOB)) <= 8192),
    UNIQUE(goal_id, ordinal)
);
CREATE TABLE IF NOT EXISTS goal_reports (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    iteration_id TEXT UNIQUE REFERENCES goal_iterations(id),
    payload_json TEXT NOT NULL CHECK (length(CAST(payload_json AS BLOB)) <= 65536)
);
CREATE TRIGGER IF NOT EXISTS goal_launch_immutable
BEFORE UPDATE OF launch_id, payload_hash, request_json, conversation_id, chain_id ON goal_runs
WHEN OLD.launch_id IS NOT NEW.launch_id OR OLD.payload_hash IS NOT NEW.payload_hash
 OR OLD.request_json IS NOT NEW.request_json OR OLD.conversation_id IS NOT NEW.conversation_id
 OR OLD.chain_id IS NOT NEW.chain_id
BEGIN SELECT RAISE(ABORT, 'goal launch is immutable'); END;
"""


class GoalRunsStore:
    """Typed launch owner; no dispatch or cross-store transactions."""

    def __init__(self, db: AgentRunsDB) -> None:
        self.db = db

    def create(self, request: GoalRequest, *, launch_id: str) -> GoalSnapshot:
        """Allocate launch, conversation UUID and chain together or reuse identity."""
        if type(request) is not GoalRequest:
            raise TypeError("request must be GoalRequest")
        _identity(launch_id)
        payload = request.canonical_json()
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self.db.automatic_work.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM goal_runs WHERE launch_id=?", (launch_id,)
            ).fetchone()
            if existing:
                if (
                    existing["payload_hash"] != digest
                    or existing["request_json"] != payload
                ):
                    raise ValueError("launch_payload_conflict")
                return self._snapshot(conn, existing)
            goal_id, conversation_id = str(uuid4()), str(uuid4())
            chain_id = self.db.automatic_work._create_chain(
                conn,
                conversation_id,
                root_submission_id="goal:" + goal_id,
                limits=request.policy.chain_limits(),
            )
            now = time.time()
            conn.execute(
                "INSERT INTO goal_runs (id, launch_id, payload_hash, request_json, conversation_id, chain_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    goal_id,
                    launch_id,
                    digest,
                    payload,
                    conversation_id,
                    chain_id,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM goal_runs WHERE id=?", (goal_id,)
            ).fetchone()
            return self._snapshot(conn, row)

    def get(self, goal_id: str) -> GoalSnapshot:
        """Read immutable launch data and bounded private results."""
        _identity(goal_id)
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM goal_runs WHERE id=?", (goal_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown_goal")
            return self._snapshot(conn, row)

    def set_provisioning(
        self,
        snapshot: GoalSnapshot,
        *,
        status: Literal["starting", "ready", "paused"],
        pause_reason: str | None = None,
    ) -> GoalSnapshot:
        """Compare-and-swap setup state; this never authorizes an iteration."""
        if type(snapshot) is not GoalSnapshot:
            raise TypeError("snapshot must be GoalSnapshot")
        if status not in ("starting", "ready", "paused"):
            raise ValueError("invalid_provisioning_status")
        if pause_reason is not None:
            _identity(pause_reason)
        with self.db.automatic_work.transaction() as conn:
            changed = conn.execute(
                "UPDATE goal_runs SET status=?, pause_reason=?, revision=revision+1, updated_at=? WHERE id=? AND revision=?",
                (status, pause_reason, time.time(), snapshot.id, snapshot.revision),
            ).rowcount
            if changed != 1:
                raise ValueError("revision_conflict")
            return self._snapshot(
                conn,
                conn.execute(
                    "SELECT * FROM goal_runs WHERE id=?", (snapshot.id,)
                ).fetchone(),
            )

    def _snapshot(self, conn: sqlite3.Connection, row: sqlite3.Row) -> GoalSnapshot:
        reports = conn.execute(
            "SELECT payload_json FROM goal_reports WHERE goal_id=? ORDER BY rowid",
            (row["id"],),
        ).fetchall()
        return GoalSnapshot(
            id=row["id"],
            launch_id=row["launch_id"],
            payload_hash=row["payload_hash"],
            conversation_id=row["conversation_id"],
            chain_id=row["chain_id"],
            revision=row["revision"],
            status=row["status"],
            pause_reason=row["pause_reason"],
            iteration_count=conn.execute(
                "SELECT count(*) FROM goal_iterations WHERE goal_id=?", (row["id"],)
            ).fetchone()[0],
            request=GoalRequest.model_validate_json(row["request_json"]),
            reports=tuple(
                GoalReport.model_validate_json(r["payload_json"]) for r in reports
            ),
            accounting=self.db.automatic_work._snapshot(conn, row["chain_id"]),
        )
