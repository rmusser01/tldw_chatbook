"""Private goal records sharing AgentRunsDB's durable automatic-work transaction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from tldw_chatbook.Agents.goal_models import (
    GoalCheckpoint,
    GoalEvidence,
    GoalReport,
    GoalRequest,
    GoalSnapshot,
    IterationReport,
)
from tldw_chatbook.DB.automatic_work import _identity

if TYPE_CHECKING:
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

SCHEMA = """
BEGIN IMMEDIATE;
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
CREATE TABLE IF NOT EXISTS goal_waits (
    goal_id TEXT PRIMARY KEY REFERENCES goal_runs(id),
    retry_at REAL NOT NULL,
    reason TEXT NOT NULL
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
CREATE TABLE IF NOT EXISTS goal_checkpoints (
    attempt_id TEXT PRIMARY KEY REFERENCES automatic_wake_attempts(id),
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072)
);
CREATE TABLE IF NOT EXISTS goal_evidence (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    attempt_id TEXT NOT NULL REFERENCES automatic_wake_attempts(id),
    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072)
);
CREATE TABLE IF NOT EXISTS goal_payload_reservations (
    attempt_id TEXT PRIMARY KEY REFERENCES automatic_wake_attempts(id),
    goal_id TEXT NOT NULL REFERENCES goal_runs(id),
    bytes INTEGER NOT NULL CHECK(bytes >= 0)
);
DROP TRIGGER IF EXISTS goal_launch_immutable;
CREATE TRIGGER goal_launch_immutable
BEFORE UPDATE OF launch_id, payload_hash, request_json, conversation_id, chain_id ON goal_runs
WHEN OLD.launch_id IS NOT NEW.launch_id OR OLD.payload_hash IS NOT NEW.payload_hash
 OR (OLD.request_json IS NOT NEW.request_json AND NOT (NEW.request_json='' AND NEW.status='removed'
     AND OLD.status IN ('completed','paused','awaiting_result_review','stopped')
     AND NOT EXISTS (SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id
         WHERE i.goal_id=OLD.id AND a.state IN ('prepared','accepted','review_required'))))
 OR OLD.conversation_id IS NOT NEW.conversation_id OR OLD.chain_id IS NOT NEW.chain_id
BEGIN SELECT RAISE(ABORT, 'goal launch is immutable'); END;

UPDATE goal_reports SET payload_json = json_remove(json_set(payload_json,
 '$.candidate_draft', COALESCE(json_extract(payload_json,'$.draft'),''),
 '$.evidence_ids', json(COALESCE(json_extract(payload_json,'$.evidence_refs'),'[]')),
 '$.completion_recommended', json('false')), '$.draft', '$.evidence_refs')
 WHERE json_valid(payload_json) AND (json_type(payload_json,'$.draft') IS NOT NULL OR json_type(payload_json,'$.evidence_refs') IS NOT NULL);
COMMIT;
"""


class GoalRunsStore:
    """Typed launch owner; no dispatch or cross-store transactions."""

    def __init__(self, db: AgentRunsDB) -> None:
        self.db = db
        self.payload_limit = 128 * 1024 * 1024

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
                if existing["status"] == "removed":
                    raise ValueError("goal_payload_removed")
                if (
                    existing["payload_hash"] != digest
                    or existing["request_json"] != payload
                ):
                    stored = existing["request_json"]
                    if (
                        hashlib.sha256(stored.encode("utf-8")).hexdigest()
                        != existing["payload_hash"]
                        or GoalRequest.model_validate_json(stored) != request
                    ):
                        raise ValueError("launch_payload_conflict")
                return self._snapshot(conn, existing)
            request.validate_verifier_invocations()
            self._require_capacity(
                conn, len(payload.encode()), request.policy.payload_bytes
            )
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

    def control(self, goal_id: str, *, stop: bool = False) -> GoalSnapshot:
        """Close successor admission without releasing the execution owner."""
        with self.db.automatic_work.transaction() as conn:
            goal = self.get(goal_id)
            if goal.status in {
                "removed",
                "completed",
                "closed",
                "stopped",
                "recovery_required",
            }:
                return goal
            active = conn.execute(
                "SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id "
                "WHERE i.goal_id=? AND a.state IN ('prepared','accepted')",
                (goal_id,),
            ).fetchone()
            state = (
                ("stopping" if active else "stopped")
                if stop
                else ("pause_requested" if active else "paused")
            )
            if goal.status == "stopping":
                state = "stopping"
            if goal.status == state:
                return goal
            conn.execute(
                "UPDATE goal_runs SET status=?,pause_reason=?,revision=revision+1,updated_at=? WHERE id=?",
                (state, "user_stop" if stop else "user_pause", time.time(), goal_id),
            )
            return self.get(goal_id)

    def settle_control(self, goal_id: str) -> GoalSnapshot:
        """Finalize controls only after the coordinator has drained physical work."""
        with self.db.automatic_work.transaction() as conn:
            goal = self.get(goal_id)
            if goal.status not in {"pause_requested", "stopping"}:
                return goal
            active = conn.execute(
                "SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id WHERE i.goal_id=? AND a.state IN ('prepared','accepted','review_required')",
                (goal_id,),
            ).fetchone()
            if active:
                return goal
            conn.execute(
                "UPDATE goal_runs SET status=?,revision=revision+1 WHERE id=?",
                ("stopped" if goal.status == "stopping" else "paused", goal_id),
            )
            return self.get(goal_id)

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

    def _payload_bytes(self, conn) -> int:
        total = 0
        for table, column in (
            ("goal_runs", "request_json"),
            ("goal_reports", "payload_json"),
            ("goal_checkpoints", "payload_json"),
            ("goal_evidence", "payload_json"),
            ("goal_iterations", "check_results_json"),
            ("goal_iterations", "evidence_refs_json"),
        ):
            total += conn.execute(
                f"SELECT COALESCE(SUM(length(CAST({column} AS BLOB))),0) FROM {table}"
            ).fetchone()[0]
        total += conn.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM goal_payload_reservations"
        ).fetchone()[0]
        return total

    def _require_capacity(self, conn, amount: int, policy_limit: int) -> None:
        if self._payload_bytes(conn) + amount > min(self.payload_limit, policy_limit):
            raise ValueError("goal_payload_capacity")

    def reserve_result(self, conn, goal, attempt_id: str) -> None:
        from tldw_chatbook.Agents.goal_iteration import GOAL_BYTES, RECORD_BYTES

        retained = conn.execute(
            "SELECT COALESCE(SUM(length(CAST(payload_json AS BLOB))),0) FROM goal_evidence WHERE goal_id=?",
            (goal.id,),
        ).fetchone()[0]
        slots = min(32, (GOAL_BYTES - retained) // RECORD_BYTES)
        if slots <= 0:
            raise ValueError("goal_evidence_capacity")
        amount = (slots + 2) * RECORD_BYTES
        self._require_capacity(conn, amount, goal.policy.payload_bytes)
        conn.execute(
            "INSERT INTO goal_payload_reservations VALUES (?,?,?)",
            (attempt_id, goal.id, amount),
        )

    def result_record_capacity(self, attempt_id: str) -> int:
        """Return the pre-admitted number of full private observation slots."""
        from tldw_chatbook.Agents.goal_iteration import RECORD_BYTES

        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT bytes FROM goal_payload_reservations WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            return max(0, row[0] // RECORD_BYTES - 2) if row else 0

    def _checkpoints(self, conn, goal_id: str) -> tuple[GoalCheckpoint, ...]:
        return tuple(
            GoalCheckpoint.model_validate_json(r[0])
            for r in conn.execute(
                "SELECT c.payload_json FROM goal_checkpoints c JOIN goal_iterations i ON i.attempt_id=c.attempt_id WHERE c.goal_id=? ORDER BY i.ordinal",
                (goal_id,),
            )
        )

    def evidence(self, goal_id: str) -> tuple[GoalEvidence, ...]:
        """Read only private copies; source logs and scratch artifacts may be pruned."""
        with self.db.connection() as conn:
            return self._evidence(conn, goal_id)

    def _evidence(self, conn, goal_id):
        # Only observations belonging to accepted stored checkpoints are reusable.
        return tuple(
            GoalEvidence.model_validate_json(r[0])
            for r in conn.execute(
                "SELECT e.payload_json FROM goal_evidence e JOIN goal_checkpoints c ON c.attempt_id=e.attempt_id "
                "JOIN goal_iterations i ON i.attempt_id=e.attempt_id JOIN agent_runs r ON r.id=i.run_id "
                "JOIN goal_runs g ON g.id=e.goal_id WHERE e.goal_id=? AND c.goal_id=g.id AND i.goal_id=g.id "
                "AND r.conversation_id=g.conversation_id AND r.work_chain_id=g.chain_id ORDER BY e.rowid",
                (goal_id,),
            )
        )

    def checkpoint(self, result, *, binding_available: bool = True) -> GoalSnapshot:
        """Persist private evidence and exact matching attempt transition in FULL commit."""
        from tldw_chatbook.Agents.agent_models import RunTerminationReason
        from tldw_chatbook.Agents.goal_iteration import (
            GOAL_BYTES,
            evaluate_iteration,
            goal_criteria,
            parse_iteration_report,
            refresh_evidence,
            resolve_runtime_evidence,
        )
        from tldw_chatbook.Agents.goal_models import GoalIterationResult

        if type(result) is not GoalIterationResult or not result.attempt_id:
            raise ValueError("checkpoint_requires_accepted_attempt")
        result_payload = asdict(result)
        # Optional selection identity must not change pre-selection checkpoint hashes.
        for record in result_payload["tool_records"]:
            if record["invocation"].get("verifier_id") is None:
                record["invocation"].pop("verifier_id", None)
        fingerprint = hashlib.sha256(
            json.dumps(result_payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        with self.db.automatic_work.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM goal_runs WHERE id=?", (result.goal_id,)
            ).fetchone()
            if row is None or row["status"] == "removed":
                raise ValueError("goal_payload_removed")
            existing = conn.execute(
                "SELECT * FROM goal_checkpoints WHERE attempt_id=?",
                (result.attempt_id,),
            ).fetchone()
            if existing:
                if (
                    existing["goal_id"] != result.goal_id
                    or existing["payload_hash"] != fingerprint
                ):
                    raise ValueError("checkpoint_conflict")
                return self._snapshot(conn, row)
            goal = self._snapshot(conn, row)
            attempt = conn.execute(
                "SELECT * FROM automatic_wake_attempts WHERE id=?", (result.attempt_id,)
            ).fetchone()
            iteration = conn.execute(
                "SELECT * FROM goal_iterations WHERE attempt_id=?", (result.attempt_id,)
            ).fetchone()
            if (
                not attempt
                or not iteration
                or attempt["state"] not in ("accepted", "review_required")
                or attempt["attempt_kind"] != "goal_iteration"
                or attempt["chain_id"] != goal.chain_id
                or attempt["conversation_id"] != goal.conversation_id
                or iteration["goal_id"] != goal.id
                or iteration["ordinal"] != result.ordinal
                or iteration["run_id"] != result.native_run_id
            ):
                raise ValueError("checkpoint_ownership")
            if result.native_run_id:
                run = conn.execute(
                    "SELECT * FROM agent_runs WHERE id=?", (result.native_run_id,)
                ).fetchone()
                if (
                    not run
                    or run["conversation_id"] != goal.conversation_id
                    or run["work_chain_id"] != goal.chain_id
                ):
                    raise ValueError("checkpoint_run_ownership")
            if not conn.execute(
                "SELECT 1 FROM goal_payload_reservations WHERE attempt_id=?",
                (result.attempt_id,),
            ).fetchone():
                raise ValueError("checkpoint_capacity_unreserved")
            prior = goal.checkpoints[-1] if goal.checkpoints else None
            error = None
            try:
                report = parse_iteration_report(
                    result.outcome.final_text if result.outcome else ""
                )
            except ValueError:
                report = IterationReport()
                error = "malformed_report"
            if result.termination_reason in (
                RunTerminationReason.PRE_EFFECT_RATE_LIMIT,
                RunTerminationReason.PRE_EFFECT_PERMANENT,
            ):
                error = None  # a local adapter rejection produced no model report
            current = resolve_runtime_evidence(goal, result)
            retained = self._evidence(conn, goal.id)
            # Capacity was reserved before effects; keep only bounded private copies.
            size = sum(len(e.model_dump_json().encode()) for e in retained)
            accepted = []
            for e in current:
                payload = e.model_dump_json()
                if size + len(payload.encode()) > GOAL_BYTES:
                    error = error or "evidence_capacity"
                    continue
                size += len(payload.encode())
                accepted.append(e)
                conn.execute(
                    "INSERT INTO goal_evidence VALUES (?,?,?,?)",
                    (e.id, goal.id, result.attempt_id, payload),
                )
            evidence = tuple(refresh_evidence(goal, e) for e in retained) + tuple(
                accepted
            )
            if not binding_available:
                evidence = tuple(
                    e.model_copy(update={"fresh": False}) for e in evidence
                )
            decision = evaluate_iteration(report, evidence, prior, goal_criteria(goal))
            failed = (
                error is not None
                or bool(decision.evidence_errors)
                or result.termination_reason != RunTerminationReason.DONE
                or any(e.verifier_id and not e.passed for e in current)
            )
            failed_count = (
                (prior.decision.failed_count if prior else 0) + 1 if failed else 0
            )
            decision = decision.model_copy(update={"failed_count": failed_count})
            unresolved = conn.execute(
                "SELECT 1 FROM automatic_work_reservations WHERE chain_id=? AND "
                "(state IN ('reserved','uncertain') OR (kind='tokens' AND state='committed')) LIMIT 1",
                (goal.chain_id,),
            ).fetchone()
            uncertain = bool(
                unresolved
                or attempt["state"] == "review_required"
                or result.outcome is None
                or result.termination_reason == RunTerminationReason.UNKNOWN_EFFECT
                or any(e.result.exit_code is None for e in result.tool_records)
            )
            if uncertain:
                decision = decision.model_copy(
                    update={"action": "recovery_required", "reason": "uncertain_effect"}
                )
            elif failed:
                decision = decision.model_copy(
                    update={
                        "action": "pause" if failed_count >= 3 else "continue",
                        "reason": error or "iteration_failed",
                    }
                )
            if not uncertain and result.termination_reason in (
                RunTerminationReason.TOKEN_LIMIT,
                RunTerminationReason.STEP_LIMIT,
                RunTerminationReason.MODEL_TURN_LIMIT,
                RunTerminationReason.WALL_LIMIT,
                RunTerminationReason.AUTOMATIC_LIMIT,
                RunTerminationReason.PERMISSION_REFUSED,
                RunTerminationReason.AUTHORITY_CHANGED,
                RunTerminationReason.PRE_EFFECT_PERMANENT,
                RunTerminationReason.CANCELLED,
            ):
                decision = decision.model_copy(
                    update={
                        "action": "pause",
                        "reason": "iteration_" + result.termination_reason.value,
                    }
                )
            if (
                not uncertain
                and result.termination_reason
                == RunTerminationReason.PRE_EFFECT_RATE_LIMIT
            ):
                decision = decision.model_copy(
                    update={
                        "no_progress_count": prior.decision.no_progress_count
                        if prior
                        else 0,
                        "reason": "pre_effect_retry_exhausted"
                        if failed_count >= 3
                        else "pre_effect_rate_limit",
                    }
                )
            if not uncertain and decision.action == "continue":
                accounting = goal.accounting
                if decision.no_progress_count >= 2:
                    decision = decision.model_copy(
                        update={"action": "pause", "reason": "no_progress"}
                    )
                elif any(
                    accounting.available.get(k, 1) <= 0
                    for k in ("generation", "model_call", "tokens")
                ) or (accounting.deadline_at and accounting.deadline_at <= time.time()):
                    decision = decision.model_copy(
                        update={"action": "pause", "reason": "goal_budget_exhausted"}
                    )
            if (
                not uncertain
                and decision.action == "continue"
                and result.termination_reason
                == RunTerminationReason.PRE_EFFECT_RATE_LIMIT
            ):
                decision = decision.model_copy(
                    update={"reason": "pre_effect_rate_limit"}
                )
            if not uncertain and goal.status in {
                "pause_requested",
                "stopping",
                "stopped",
                "paused",
            }:
                decision = decision.model_copy(
                    update={
                        "action": "pause",
                        "reason": goal.pause_reason or "user_pause",
                    }
                )
            artifact_digest = hashlib.sha256(
                json.dumps(
                    [
                        goal.payload_hash,
                        decision.draft_digest,
                        sorted(
                            (
                                e.id,
                                e.source_digest,
                                e.checked_manifest,
                                e.verifier_sha256,
                            )
                            for e in evidence
                            if e.id in report.evidence_ids
                        ),
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            decision = decision.model_copy(
                update={
                    "checkpoint_id": result.attempt_id,
                    "artifact_digest": artifact_digest,
                }
            )
            checkpoint = GoalCheckpoint(
                id=result.attempt_id,
                artifact_digest=artifact_digest,
                ordinal=result.ordinal,
                report=report,
                decision=decision,
                report_error=error,
            )
            conn.execute(
                "INSERT INTO goal_reports (id,goal_id,iteration_id,payload_json) VALUES (?,?,?,?)",
                (uuid4().hex, goal.id, iteration["id"], report.model_dump_json()),
            )
            conn.execute(
                "INSERT INTO goal_checkpoints VALUES (?,?,?,?)",
                (result.attempt_id, goal.id, fingerprint, checkpoint.model_dump_json()),
            )
            state = (
                "ready"
                if decision.action == "continue"
                else "paused"
                if decision.action == "pause"
                else decision.action
            )
            if not uncertain and goal.status in {"stopping", "stopped"}:
                state = "stopped"
            conn.execute(
                "UPDATE goal_runs SET status=?,pause_reason=?,revision=revision+1,updated_at=? WHERE id=?",
                (state, decision.reason, time.time(), goal.id),
            )
            conn.execute(
                "UPDATE goal_iterations SET status=?,check_results_json=?,evidence_refs_json=?,revision=revision+1 WHERE attempt_id=?",
                (
                    state,
                    json.dumps([c.model_dump() for c in decision.checks]),
                    json.dumps([e.id for e in accepted]),
                    result.attempt_id,
                ),
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state=?,completed_at=? WHERE id=?",
                (
                    "review_required" if uncertain else "completed",
                    None if uncertain else time.time(),
                    result.attempt_id,
                ),
            )
            if uncertain:
                conn.execute(
                    "UPDATE automatic_work_chains SET status='review_required',pause_reason='uncertain_effect' WHERE id=?",
                    (goal.chain_id,),
                )
            conn.execute(
                "DELETE FROM goal_payload_reservations WHERE attempt_id=?",
                (result.attempt_id,),
            )
            self._require_capacity(conn, 0, goal.policy.payload_bytes)
            return self._snapshot(
                conn,
                conn.execute(
                    "SELECT * FROM goal_runs WHERE id=?", (goal.id,)
                ).fetchone(),
            )

    def remove_payloads(self, goal_id: str) -> GoalSnapshot:
        """Remove settled private bodies only; retain non-resumable accounting identity."""
        with self.db.automatic_work.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM goal_runs WHERE id=?", (goal_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown_goal")
            if row["status"] == "removed":
                return self._snapshot(conn, row)
            active = conn.execute(
                "SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id "
                "WHERE i.goal_id=? AND a.state IN ('prepared','accepted','review_required')",
                (goal_id,),
            ).fetchone()
            uncertain = conn.execute(
                "SELECT 1 FROM automatic_work_reservations WHERE chain_id=? AND (state IN ('reserved','uncertain') OR (kind='tokens' AND state='committed'))",
                (row["chain_id"],),
            ).fetchone()
            if (
                row["status"]
                not in ("paused", "completed", "awaiting_result_review", "stopped")
                or active
                or uncertain
            ):
                raise ValueError("goal_not_settled")
            for table in (
                "goal_evidence",
                "goal_checkpoints",
                "goal_reports",
                "goal_payload_reservations",
                "goal_waits",
            ):
                conn.execute(f"DELETE FROM {table} WHERE goal_id=?", (goal_id,))
            conn.execute(
                "UPDATE goal_iterations SET check_results_json='[]', evidence_refs_json='[]' WHERE goal_id=?",
                (goal_id,),
            )
            conn.execute(
                "UPDATE goal_runs SET request_json='',status='removed',pause_reason='payload_removed',revision=revision+1 WHERE id=?",
                (goal_id,),
            )
            return self._snapshot(
                conn,
                conn.execute(
                    "SELECT * FROM goal_runs WHERE id=?", (goal_id,)
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
                "SELECT count(*) FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id WHERE i.goal_id=? AND a.accepted_at IS NOT NULL",
                (row["id"],),
            ).fetchone()[0],
            retry_at=(
                wait[0]
                if (
                    wait := conn.execute(
                        "SELECT retry_at,reason FROM goal_waits WHERE goal_id=?",
                        (row["id"],),
                    ).fetchone()
                )
                else None
            ),
            retry_reason=wait[1] if wait else None,
            request=GoalRequest.model_validate_json(row["request_json"])
            if row["status"] != "removed"
            else None,
            checkpoints=self._checkpoints(conn, row["id"]),
            reports=tuple(
                GoalReport.model_validate_json(r["payload_json"]) for r in reports
            ),
            accounting=self.db.automatic_work._snapshot(conn, row["chain_id"]),
        )
