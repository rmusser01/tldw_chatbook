"""Passive private workflow history in the existing AgentRunsDB transaction owner."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING, get_args
from uuid import uuid4

from pydantic import TypeAdapter

from tldw_chatbook.Agents.agent_models import RUN_DONE, TERMINAL_RUN_STATUSES
from tldw_chatbook.Agents.workflow_models import (
    RECORD_BYTES,
    TOTAL_BYTES,
    WORKFLOW_BYTES,
    Identity,
    WorkflowAlreadyExists,
    WorkflowDecision,
    WorkflowPayloadReservation,
    WorkflowPlan,
    WorkflowRequest,
    WorkflowSnapshot,
    WorkflowStatus,
    WorkflowTaskAttempt,
    WorkflowTaskResult,
)

_identity = TypeAdapter(Identity).validate_python
# AgentRunsDB opens fully migrated files. Archive validation supplies its authenticated older stage.
_CURRENT_WORKFLOW_STAGE = 29
# Native preparation creates these related rows atomically. Validate that original
# custody before using either the result slot or a never-launched tombstone.
_CLAIM_JOINS = """
    LEFT JOIN workflow_task_claims c ON c.task_attempt_id=t.id
    LEFT JOIN workflow_phase_attempts ph ON ph.id=c.phase_attempt_id
    LEFT JOIN automatic_wake_attempts wake ON wake.id=ph.native_attempt_id
    LEFT JOIN automatic_work_reservations a ON a.id=c.child_reservation_id
    LEFT JOIN agent_runs launched ON launched.id=c.launch_run_id
"""
_CLAIM_OWNED = f"""
    ph.workflow_id=t.workflow_id AND ph.plan_revision=t.plan_revision AND ph.phase_id=t.phase_id
    AND ph.purpose='execution' AND ph.command_id IS NOT NULL AND ph.control_revision IS NOT NULL
    AND wake.attempt_kind='workflow_phase' AND wake.chain_id=w.chain_id AND wake.conversation_id=w.conversation_id
    AND (t.run_id IS NULL OR t.run_id=c.launch_run_id)
    AND p.workflow_id=t.workflow_id AND p.bytes={RECORD_BYTES}
    AND (c.child_reservation_id IS NULL OR
         (a.chain_id=w.chain_id AND a.owner_id=wake.owner_id AND a.kind='child_launch' AND a.amount=1))
"""
_UNUSED_TASK_PROOF = """
    c.task_attempt_id IS NOT NULL AND t.status='closed' AND t.retirement='drained'
    AND t.run_id IS NULL AND (c.child_reservation_id IS NULL OR a.state IN ('reserved','released'))
    AND launched.id IS NULL
"""
_FAILED_NATIVE_STATES = TERMINAL_RUN_STATUSES - {RUN_DONE}

if TYPE_CHECKING:
    from .AgentRuns_DB import AgentRunsDB


def _hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class _WindowPublication:
    store: object
    reservation: object


@dataclass(frozen=True)
class _WindowUnpublished:
    """Original body refusal observed after FULL rollback and scope return."""

    store: object
    reservation: object
    phase: object
    error: BaseException


class WorkflowRunsStore:
    """Storage only: construction and reads never recover, admit or dispatch work."""

    def __init__(self, db: AgentRunsDB) -> None:
        self.db = db

    @staticmethod
    def _row(conn, workflow_id):
        _identity(workflow_id)
        row = conn.execute(
            "SELECT * FROM workflow_runs WHERE id=?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown_workflow")
        return row

    @staticmethod
    def _capacity(conn, workflow_id, amount, *, stage=_CURRENT_WORKFLOW_STAGE):
        """Check all owners and the portfolio in the caller's single transaction."""
        sources = [
            "SELECT id, length(CAST(request_json AS BLOB)) FROM workflow_runs",
            "SELECT workflow_id, length(CAST(payload_json AS BLOB)) FROM workflow_plan_revisions",
            "SELECT workflow_id, length(CAST(payload_json AS BLOB)) FROM workflow_decisions",
            "SELECT workflow_id, bytes FROM workflow_payload_reservations WHERE state!='removed'",
        ]
        if stage >= 26:
            sources.append(
                "SELECT workflow_id,result_bytes FROM workflow_phase_attempts"
            )
        if stage >= 28:
            sources.append(
                "SELECT workflow_id,reserved_bytes FROM workflow_change_windows"
            )
        if stage >= 29:
            sources.append(
                "SELECT workflow_id,length(CAST(repair_instruction_sha256 AS BLOB)) FROM workflow_phase_attempts"
            )
        totals = Counter(
            dict(
                conn.execute(
                    "WITH charges(owner,bytes) AS ("
                    + " UNION ALL ".join(sources)
                    + ") SELECT owner,SUM(bytes) FROM charges GROUP BY owner"
                )
            )
        )
        held_roots = WorkflowRunsStore._held_workflow_roots(conn, stage)
        WorkflowRunsStore._required_results(conn, None, stage=stage)
        if stage >= 27:
            totals.subtract(
                WorkflowRunsStore._published_slack(conn, held_roots=held_roots)
            )
        if workflow_id is not None:
            totals[workflow_id] += amount
        if sum(totals.values()) > TOTAL_BYTES or any(
            n > WORKFLOW_BYTES for n in totals.values()
        ):
            raise ValueError("workflow_payload_capacity")

    @staticmethod
    def _held_workflow_roots(conn, stage):
        """Check all workflow allowance families once, including removed histories."""
        physical_holds = """
            SELECT workflow_id FROM workflow_task_attempts WHERE retirement!='drained' OR status IN ('accepted','prepared','uncertain')
            UNION SELECT workflow_id FROM workflow_phase_attempts WHERE retirement!='drained' OR status IN ('accepted','prepared','uncertain')
            UNION SELECT workflow_id FROM workflow_active_intervals WHERE ended_at IS NULL
        """
        if stage >= 28:
            physical_holds += " UNION SELECT workflow_id FROM workflow_change_windows WHERE state IN ('active','unresolved')"
        held = set()
        for root, status in conn.execute(
            """WITH held_members(id) AS (
                SELECT id FROM automatic_work_chains WHERE status='review_required'
                UNION SELECT chain_id FROM automatic_wake_attempts WHERE state IN ('prepared','accepted','review_required')
                UNION SELECT chain_id FROM automatic_chat_start_attempts WHERE state IN ('prepared','accepted','review_required')
                UNION SELECT chain_id FROM automatic_work_reservations WHERE state IN ('reserved','uncertain') OR (kind='tokens' AND state='committed')
            ), held_roots(id) AS (
                SELECT DISTINCT COALESCE(c.allowance_root_chain_id,c.id) FROM automatic_work_chains c JOIN held_members m ON m.id=c.id
            ), held_workflows(id) AS ("""
            + physical_holds
            + """)
            SELECT w.chain_id,w.status FROM workflow_runs w
            WHERE w.chain_id IN (SELECT id FROM held_roots) OR w.id IN (SELECT id FROM held_workflows)
        """
        ):
            if status == "removed":
                raise ValueError("workflow_required_result_missing")
            held.add(root)
        return held

    @staticmethod
    def _published_slack(conn, workflow_id=None, *, held_roots=None):
        """Return unused reservation bytes per workflow after authenticating stored pairs."""
        from tldw_chatbook.Agents.workflow_models import WorkflowCheckpoint

        source = "workflow_payload_reservations p"
        attempt_key = "p.attempt_id"
        if workflow_id is not None:
            # Delete uses the workflow on each table. Start from their union, including
            # a checkpoint whose independent workflow FK differs from its result's.
            source = """(
                SELECT id AS attempt_id FROM workflow_task_attempts WHERE workflow_id=:owner
                UNION SELECT attempt_id FROM workflow_payload_reservations WHERE workflow_id=:owner
                UNION SELECT attempt_id FROM workflow_results WHERE workflow_id=:owner
                UNION SELECT attempt_id FROM workflow_checkpoints WHERE workflow_id=:owner
            ) selected LEFT JOIN workflow_payload_reservations p ON p.attempt_id=selected.attempt_id"""
            attempt_key = "selected.attempt_id"
        rows = conn.execute(
            f"""
            SELECT p.workflow_id,p.attempt_id,p.bytes,p.state,t.run_id,t.status,t.retirement,
                   r.payload_hash AS result_hash,r.payload_json AS result_json,
                   k.payload_hash AS checkpoint_hash,k.payload_json AS checkpoint_json,
                   t.workflow_id IS p.workflow_id AND r.workflow_id IS p.workflow_id
                     AND g.conversation_id IS w.conversation_id AND g.work_chain_id IS w.chain_id
                     AND root.conversation_id IS w.conversation_id
                     AND (t.native_attempt_id IS NULL OR (native.chain_id=w.chain_id AND native.conversation_id=w.conversation_id))
                     AND root.root_submission_id IS ('workflow:' || w.id) AND root.allowance_root_chain_id IS NULL AS result_owner,
                   k.workflow_id IS p.workflow_id AS checkpoint_owner,
                   w.status IN ('completed','closed','stopped') AS settled, w.chain_id,
                   g.status AS run_status, c.task_attempt_id AS claim,
                   (c.launch_run_id=t.run_id AND ph.workflow_id=t.workflow_id AND ph.plan_revision=t.plan_revision AND ph.phase_id=t.phase_id
                    AND ph.purpose='execution' AND wake.attempt_kind='workflow_phase' AND wake.conversation_id=w.conversation_id
                    AND wake.chain_id=w.chain_id AND child.owner_id=wake.owner_id AND child.chain_id=w.chain_id
                    AND child.kind='child_launch' AND child.amount=1) AS claim_owner,
                   ph.status AS phase_status,ph.retirement AS phase_retirement,wake.state AS wake_state,child.state AS child_state
            FROM {source}
            LEFT JOIN workflow_results r ON r.attempt_id={attempt_key}
            LEFT JOIN workflow_checkpoints k ON k.attempt_id={attempt_key}
            LEFT JOIN workflow_task_attempts t ON t.id={attempt_key}
            LEFT JOIN workflow_task_claims c ON c.task_attempt_id=t.id
            LEFT JOIN workflow_phase_attempts ph ON ph.id=c.phase_attempt_id
            LEFT JOIN automatic_wake_attempts wake ON wake.id=ph.native_attempt_id
            LEFT JOIN automatic_work_reservations child ON child.id=c.child_reservation_id
            LEFT JOIN workflow_runs w ON w.id=p.workflow_id
            LEFT JOIN automatic_work_chains root ON root.id=w.chain_id
            LEFT JOIN automatic_wake_attempts native ON native.id=t.native_attempt_id
            LEFT JOIN agent_runs g ON g.id=t.run_id
            WHERE (p.state='published' OR r.attempt_id IS NOT NULL OR k.attempt_id IS NOT NULL
                   OR (p.attempt_id IS NOT NULL AND p.workflow_id IS NOT t.workflow_id))
        """,
            {"owner": workflow_id} if workflow_id is not None else (),
        )
        rows.row_factory = sqlite3.Row
        slack = Counter()
        for row in rows:
            owner, attempt, run = row["workflow_id"], row["attempt_id"], row["run_id"]
            bound, status = row["bytes"], row["status"]
            payload, saved = row["result_json"], row["checkpoint_json"]
            result_hash = row["result_hash"]
            if (
                not row["result_owner"]
                or row["state"] != "published"
                or not isinstance(payload, str)
            ):
                raise ValueError("invalid_workflow_result_ownership")
            result = WorkflowTaskResult.model_validate_json(payload)
            result_bytes = payload.encode("utf-8")
            size = len(result_bytes)
            if (
                (result.workflow_id, result.attempt_id, result.run_id)
                != (owner, attempt, run)
                or hashlib.sha256(result_bytes).hexdigest() != result_hash
                or size > bound
                or status not in (result.status, "closed")
            ):
                raise ValueError("invalid_workflow_result_ownership")
            if row["claim"] and not row["claim_owner"]:
                raise ValueError("invalid_workflow_task_claim")
            if (
                result.status == "completed"
                and row["run_status"] in _FAILED_NATIVE_STATES
            ):
                raise ValueError("invalid_workflow_result_ownership")
            if saved is None:
                continue
            checkpoint = WorkflowCheckpoint.model_validate_json(saved)
            checkpoint_bytes = saved.encode("utf-8")
            size += len(checkpoint_bytes)
            if (
                not row["checkpoint_owner"]
                or hashlib.sha256(checkpoint_bytes).hexdigest()
                != row["checkpoint_hash"]
                or size > bound
                or (
                    checkpoint.workflow_id,
                    checkpoint.attempt_id,
                    checkpoint.run_id,
                    checkpoint.result_digest,
                )
                != (owner, attempt, run, _hash(result.canonical_json()))
            ):
                raise ValueError("invalid_workflow_checkpoint")
            refs = checkpoint.change_window_refs
            if refs:
                placeholders = ",".join("?" for _ in refs)
                found = conn.execute(
                    f"SELECT COUNT(*) FROM change_snapshots WHERE run_id=? AND id IN ({placeholders})",
                    (run, *refs),
                ).fetchone()[0]
                if found != len(set(refs)):
                    raise ValueError("invalid_workflow_checkpoint")
            if (
                not row["settled"]
                or held_roots is None
                or row["chain_id"] in held_roots
                or status not in ("completed", "failed")
                or row["retirement"] != "drained"
            ):
                continue
            if row["claim"] and (
                row["phase_status"] not in ("completed", "closed", "failed")
                or row["phase_retirement"] != "drained"
                or row["wake_state"] != "completed"
                or row["child_state"] != "committed"
            ):
                continue
            if row["run_status"] in TERMINAL_RUN_STATUSES:
                slack[owner] += bound - size
        return slack

    def create(self, request: WorkflowRequest, *, launch_id: str) -> WorkflowSnapshot:
        """Create the structural launch and chain in one FULL commit, without admission."""
        if type(request) is not WorkflowRequest:
            raise TypeError("request must be WorkflowRequest")
        request = WorkflowRequest.model_validate_json(request.canonical_json())
        _identity(launch_id)
        payload = request.canonical_json()
        with self.db.automatic_work.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_runs WHERE launch_id=?", (launch_id,)
            ).fetchone()
            if row:
                if row["request_json"] != payload or row["payload_hash"] != _hash(
                    payload
                ):
                    raise ValueError("launch_payload_conflict")
                return self._snapshot(conn, row)
            existing = conn.execute(
                "SELECT id FROM workflow_runs WHERE conversation_id=? "
                "AND status NOT IN ('stopped','completed','closed','removed') "
                "ORDER BY created_at,id LIMIT 1",
                (request.conversation_id,),
            ).fetchone()
            if existing:
                raise WorkflowAlreadyExists(existing["id"])
            workflow_id = str(uuid4())
            self._capacity(conn, workflow_id, len(payload.encode()))
            chain_id = self.db.automatic_work._create_chain(
                conn,
                request.conversation_id,
                root_submission_id="workflow:" + workflow_id,
                limits=request.policy,
            )
            now = time.time()
            conn.execute(
                "INSERT INTO workflow_runs (id,launch_id,payload_hash,request_json,conversation_id,chain_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    workflow_id,
                    launch_id,
                    _hash(payload),
                    payload,
                    request.conversation_id,
                    chain_id,
                    now,
                    now,
                ),
            )
            return self._snapshot(conn, self._row(conn, workflow_id))

    def _snapshot(self, conn, row, *, _accounting=None):
        request = WorkflowRequest.model_validate_json(row["request_json"])
        chain = conn.execute(
            "SELECT conversation_id,allowance_root_chain_id,root_submission_id,limits_json FROM automatic_work_chains WHERE id=?",
            (row["chain_id"],),
        ).fetchone()
        if (
            _hash(row["request_json"]) != row["payload_hash"]
            or request.conversation_id != row["conversation_id"]
            or not chain
            or chain["conversation_id"] != request.conversation_id
            or chain["allowance_root_chain_id"] is not None
            or chain["root_submission_id"] != "workflow:" + row["id"]
            or json.loads(chain["limits_json"]) != asdict(request.policy)
        ):
            raise ValueError("invalid_workflow_ownership")
        plans = []
        for revision, saved in enumerate(
            conn.execute(
                "SELECT * FROM workflow_plan_revisions WHERE workflow_id=? ORDER BY revision",
                (row["id"],),
            ),
            1,
        ):
            plan = WorkflowPlan.model_validate_json(saved["payload_json"])
            if (
                saved["revision"] != revision
                or plan.request != request
                or _hash(saved["payload_json"]) != saved["payload_hash"]
            ):
                raise ValueError("invalid_workflow_plan")
            self._plan_rows(conn, row["id"], revision, plan)
            plans.append(plan)
        if row["status"] == "removed" and self.db.goal_runs.unsettled_root(
            conn, row["chain_id"]
        ):
            raise ValueError("workflow_required_result_missing")
        self._required_results(conn, row["id"], status=row["status"])
        results = tuple(
            self._read_result(conn, row["id"], r[0])
            for r in conn.execute(
                "SELECT attempt_id FROM workflow_results WHERE workflow_id=? ORDER BY rowid",
                (row["id"],),
            )
        )
        return WorkflowSnapshot(
            id=row["id"],
            launch_id=row["launch_id"],
            chain_id=row["chain_id"],
            request=request,
            control_revision=row["control_revision"],
            progress_cursor=row["progress_cursor"],
            status=row["status"],
            plans=tuple(plans),
            results=results,
            accounting=self.db.automatic_work._snapshot(conn, row["chain_id"], _accounting=_accounting),
        )

    @staticmethod
    def _plan_rows(conn, workflow_id, revision, plan):
        phases = {(phase.id, ordinal) for ordinal, phase in enumerate(plan.phases, 1)}
        actual_phases = {
            tuple(row)
            for row in conn.execute(
                "SELECT phase_id,ordinal FROM workflow_phases WHERE workflow_id=? AND plan_revision=?",
                (workflow_id, revision),
            )
        }
        tasks = {
            (phase.id, task.id, ordinal)
            for phase in plan.phases
            for ordinal, task in enumerate(phase.tasks, 1)
        }
        actual_tasks = {
            tuple(row)
            for row in conn.execute(
                "SELECT phase_id,task_id,ordinal FROM workflow_tasks WHERE workflow_id=? AND plan_revision=?",
                (workflow_id, revision),
            )
        }
        if actual_phases != phases or actual_tasks != tasks:
            raise ValueError("invalid_workflow_plan_ownership")

    def get(self, workflow_id: str, *, _accounting=None) -> WorkflowSnapshot:
        """Read a deferred multi-query snapshot without acquiring a writer lock.

        Join an active caller transaction without finishing it; otherwise roll
        back only this method's read transaction. No recovery sweep runs here.
        """
        with self.db.connection() as conn:
            if conn.in_transaction:
                return self._snapshot(conn, self._row(conn, workflow_id), _accounting=_accounting)
            conn.execute("BEGIN")
            try:
                return self._snapshot(conn, self._row(conn, workflow_id), _accounting=_accounting)
            finally:
                conn.rollback()

    def list_for_conversation(
        self, conversation_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[dict, ...]:
        """Return a body-free bounded page in one autocommit-atomic SELECT.

        An active caller transaction is joined without committing or rolling it
        back; passive history does not need the execution fence's writer lock.
        """
        _identity(conversation_id)
        if (
            type(limit) is not int
            or not 1 <= limit <= 50
            or type(offset) is not int
            or not 0 <= offset < 2**63
        ):
            raise ValueError("invalid_workflow_page")
        with self.db.connection() as conn:
            return tuple(
                dict(row)
                for row in conn.execute(
                    "SELECT id,conversation_id,control_revision,progress_cursor,status,updated_at FROM workflow_runs WHERE conversation_id=? ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
                    (conversation_id, limit, offset),
                )
            )

    @staticmethod
    def _competing(conn, workflow_id, conversation_id):
        return (
            conn.execute(
                "SELECT 1 FROM workflow_runs WHERE conversation_id=? AND id!=? "
                "AND status NOT IN ('completed','stopped','closed','removed') LIMIT 1",
                (conversation_id, workflow_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _open_control(conn, row):
        if row["status"] in {"completed", "stopped", "closed", "removed"}:
            raise ValueError("workflow_terminal")
        if WorkflowRunsStore._competing(conn, row["id"], row["conversation_id"]):
            raise ValueError("workflow_chat_conflict")

    @staticmethod
    def _cas(conn, workflow_id, expected_control_revision, status):
        if type(expected_control_revision) is not int or expected_control_revision < 1:
            raise ValueError("invalid_revision")
        row = WorkflowRunsStore._row(conn, workflow_id)
        if row["control_revision"] != expected_control_revision:
            raise ValueError("revision_conflict")
        WorkflowRunsStore._open_control(conn, row)
        if status == "removed":
            raise ValueError("invalid_workflow_status")
        conn.execute(
            "UPDATE workflow_runs SET control_revision=control_revision+1,status=?,updated_at=? WHERE id=?",
            (status, time.time(), workflow_id),
        )

    def save_plan(
        self, workflow_id: str, plan: WorkflowPlan, *, expected_control_revision: int
    ) -> WorkflowSnapshot:
        """Append immutable structural rows; a saved plan grants no phase approval."""
        with self.db.automatic_work.transaction() as conn:
            return self._save_plan(
                conn,
                workflow_id,
                plan,
                expected_control_revision=expected_control_revision,
            )

    def _save_plan(self, conn, workflow_id, plan, *, expected_control_revision):
        """Share publication with atomic planning settlement; callers own FULL commit."""
        if type(plan) is not WorkflowPlan:
            raise TypeError("plan must be WorkflowPlan")
        plan = WorkflowPlan.model_validate_json(plan.canonical_json())
        payload = plan.canonical_json()
        row = self._row(conn, workflow_id)
        if plan.request != WorkflowRequest.model_validate_json(row["request_json"]):
            raise ValueError("plan_request_conflict")
        if self._active(conn, workflow_id):
            raise ValueError("workflow_has_active_evidence")
        self._cas(conn, workflow_id, expected_control_revision, "awaiting_approval")
        revision = conn.execute(
            "SELECT COALESCE(MAX(revision),0)+1 FROM workflow_plan_revisions WHERE workflow_id=?",
            (workflow_id,),
        ).fetchone()[0]
        self._capacity(conn, workflow_id, len(payload.encode()))
        conn.execute(
            "INSERT INTO workflow_plan_revisions VALUES (?,?,?,?)",
            (workflow_id, revision, _hash(payload), payload),
        )
        for ordinal, phase in enumerate(plan.phases, 1):
            conn.execute(
                "INSERT INTO workflow_phases VALUES (?,?,?,?)",
                (workflow_id, revision, phase.id, ordinal),
            )
            for task_ordinal, task in enumerate(phase.tasks, 1):
                conn.execute(
                    "INSERT INTO workflow_tasks VALUES (?,?,?,?,?)",
                    (workflow_id, revision, phase.id, task.id, task_ordinal),
                )
        conn.execute(
            "UPDATE workflow_runs SET progress_cursor=progress_cursor+1 WHERE id=?",
            (workflow_id,),
        )
        return self._snapshot(conn, self._row(conn, workflow_id))

    def set_status(
        self,
        workflow_id: str,
        *,
        status: WorkflowStatus,
        expected_control_revision: int,
    ) -> WorkflowSnapshot:
        """CAS passive control state; this allocates no runtime capability."""
        if status not in get_args(WorkflowStatus) or status == "removed":
            raise ValueError("invalid_workflow_status")
        with self.db.automatic_work.transaction() as conn:
            self._row(conn, workflow_id)
            if status in {"completed", "stopped", "closed"} and self._active(
                conn, workflow_id
            ):
                raise ValueError("workflow_has_active_evidence")
            self._cas(conn, workflow_id, expected_control_revision, status)
            return self._snapshot(conn, self._row(conn, workflow_id))

    @staticmethod
    def _ownership(conn, attempt, *, allow_removed=False):
        row = conn.execute(
            "SELECT conversation_id,chain_id,status FROM workflow_runs WHERE id=?",
            (attempt.workflow_id,),
        ).fetchone()
        if not row or (row["status"] == "removed" and not allow_removed):
            raise ValueError("invalid_workflow_ownership")
        if attempt.run_id:
            run = conn.execute(
                "SELECT conversation_id,work_chain_id FROM agent_runs WHERE id=?",
                (attempt.run_id,),
            ).fetchone()
            if not run or tuple(run) != (row["conversation_id"], row["chain_id"]):
                raise ValueError("invalid_workflow_run_ownership")
        if attempt.native_attempt_id:
            native = conn.execute(
                "SELECT conversation_id,chain_id FROM automatic_wake_attempts WHERE id=?",
                (attempt.native_attempt_id,),
            ).fetchone()
            if not native or tuple(native) != (row["conversation_id"], row["chain_id"]):
                raise ValueError("invalid_workflow_attempt_ownership")

    def record_attempt(self, attempt: WorkflowTaskAttempt) -> WorkflowTaskAttempt:
        """Persist a structural link to existing identities; never launch or charge it."""
        attempt = WorkflowTaskAttempt.model_validate_json(attempt.canonical_json())
        with self.db.automatic_work.transaction() as conn:
            self._ownership(conn, attempt)
            prior = conn.execute(
                "SELECT * FROM workflow_task_attempts WHERE id=?", (attempt.id,)
            ).fetchone()
            if prior:
                if WorkflowTaskAttempt(**dict(prior)) != attempt:
                    raise ValueError("attempt_identity_conflict")
                return attempt
            self._open_control(conn, self._row(conn, attempt.workflow_id))
            conn.execute(
                "INSERT INTO workflow_task_attempts VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt.id,
                    attempt.workflow_id,
                    attempt.plan_revision,
                    attempt.phase_id,
                    attempt.task_id,
                    attempt.ordinal,
                    attempt.run_id,
                    attempt.native_attempt_id,
                    attempt.status,
                    attempt.retirement,
                ),
            )
            conn.execute(
                "UPDATE workflow_runs SET progress_cursor=progress_cursor+1 WHERE id=?",
                (attempt.workflow_id,),
            )
        return attempt

    def record_retirement(
        self, workflow_id: str, attempt_id: str, *, run_id: str, drained: bool
    ) -> None:
        """Record the native owner's explicit drain observation, never infer it from result/status."""
        if type(drained) is not bool:
            raise ValueError("invalid_retirement_observation")
        with self.db.automatic_work.transaction() as conn:
            self._record_retirement(
                conn, workflow_id, attempt_id, run_id=run_id, drained=drained
            )

    def _record_retirement(self, conn, workflow_id, attempt_id, *, run_id, drained):
        row = conn.execute(
            "SELECT * FROM workflow_task_attempts WHERE id=? AND workflow_id=? AND run_id=?",
            (attempt_id, workflow_id, run_id),
        ).fetchone()
        if not row:
            raise ValueError("invalid_workflow_retirement_ownership")
        self._ownership(conn, WorkflowTaskAttempt(**dict(row)))
        if row["retirement"] == "drained" and not drained:
            raise ValueError("retirement_conflict")
        conn.execute(
            "UPDATE workflow_task_attempts SET retirement=? WHERE id=?",
            ("drained" if drained else "unknown", attempt_id),
        )

    @staticmethod
    def _never_run_task(conn, workflow_id, attempt_id):
        """No requested native row, committed allowance, result or checkpoint exists."""
        return (
            conn.execute(
                "SELECT 1 FROM workflow_task_attempts t "
                "JOIN workflow_task_claims c ON c.task_attempt_id=t.id "
                "LEFT JOIN automatic_work_reservations a ON a.id=c.child_reservation_id "
                "WHERE t.id=? AND t.workflow_id=? AND t.run_id IS NULL "
                "AND (c.child_reservation_id IS NULL OR a.state IN ('reserved','released')) "
                "AND NOT EXISTS (SELECT 1 FROM agent_runs WHERE id=c.launch_run_id) "
                "AND NOT EXISTS (SELECT 1 FROM workflow_results WHERE attempt_id=t.id) "
                "AND NOT EXISTS (SELECT 1 FROM workflow_checkpoints WHERE attempt_id=t.id)",
                (attempt_id, workflow_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _closed_without_native_run(
        conn: sqlite3.Connection, workflow_id: str, attempt_id: str
    ) -> bool:
        """Recognize only positively retired, unused original task custody."""
        return (
            conn.execute(
                "SELECT 1 FROM workflow_task_attempts t "
                "JOIN workflow_runs w ON w.id=t.workflow_id "
                "JOIN workflow_payload_reservations p ON p.attempt_id=t.id "
                + _CLAIM_JOINS
                + " WHERE t.id=? AND t.workflow_id=? AND "
                + _CLAIM_OWNED
                + " AND "
                + _UNUSED_TASK_PROOF,
                (attempt_id, workflow_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _release_unused_result(
        conn: sqlite3.Connection, workflow_id: str, attempt_id: str
    ) -> bool:
        """Tombstone an unused slot inside its caller's conclusive FULL closure."""
        if not conn.in_transaction:
            raise ValueError("workflow_closure_transaction_required")
        if not WorkflowRunsStore._closed_without_native_run(
            conn, workflow_id, attempt_id
        ):
            return False
        return (
            conn.execute(
                "UPDATE workflow_payload_reservations SET state='removed' "
                "WHERE attempt_id=? AND workflow_id=? AND state='reserved' "
                "AND NOT EXISTS (SELECT 1 FROM workflow_results WHERE attempt_id=?) "
                "AND NOT EXISTS (SELECT 1 FROM workflow_checkpoints WHERE attempt_id=?)",
                (attempt_id, workflow_id, attempt_id, attempt_id),
            ).rowcount
            == 1
        )

    def reserve_result(
        self, workflow_id: str, attempt_id: str, *, bytes: int = RECORD_BYTES
    ) -> WorkflowPayloadReservation:
        """Reserve the full mandatory record capacity before dispatch is possible."""
        if type(bytes) is not int or not 0 < bytes <= RECORD_BYTES:
            raise ValueError("workflow_result_record_capacity")
        with self.db.automatic_work.transaction() as conn:
            attempt = conn.execute(
                "SELECT * FROM workflow_task_attempts WHERE id=? AND workflow_id=?",
                (attempt_id, workflow_id),
            ).fetchone()
            if not attempt:
                raise ValueError("invalid_workflow_attempt_ownership")
            self._ownership(conn, WorkflowTaskAttempt(**dict(attempt)))
            prior = conn.execute(
                "SELECT * FROM workflow_payload_reservations WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if prior:
                if (
                    prior["workflow_id"] != workflow_id
                    or prior["bytes"] != bytes
                    or prior["state"] == "removed"
                ):
                    raise ValueError("reservation_conflict")
                return WorkflowPayloadReservation(**dict(prior))
            self._open_control(conn, self._row(conn, workflow_id))
            self._capacity(conn, workflow_id, bytes)
            conn.execute(
                "INSERT INTO workflow_payload_reservations VALUES (?,?,?,'reserved')",
                (attempt_id, workflow_id, bytes),
            )
            return WorkflowPayloadReservation(
                attempt_id=attempt_id,
                workflow_id=workflow_id,
                bytes=bytes,
                state="reserved",
            )

    def publish_result(self, result: WorkflowTaskResult) -> WorkflowSnapshot:
        """Publish an exact-owned mandatory result without truncation or control approval."""
        with self.db.automatic_work.transaction() as conn:
            return self._publish_result(conn, result)

    def _publish_result(self, conn, result):
        result = WorkflowTaskResult.model_validate_json(result.canonical_json())
        payload = result.canonical_json()
        row = conn.execute(
            "SELECT * FROM workflow_task_attempts WHERE id=? AND workflow_id=?",
            (result.attempt_id, result.workflow_id),
        ).fetchone()
        if not row or row["run_id"] != result.run_id:
            raise ValueError("invalid_workflow_result_ownership")
        self._ownership(conn, WorkflowTaskAttempt(**dict(row)))
        res = conn.execute(
            "SELECT * FROM workflow_payload_reservations WHERE attempt_id=? AND workflow_id=?",
            (result.attempt_id, result.workflow_id),
        ).fetchone()
        if not res or res["state"] == "removed":
            raise ValueError("workflow_result_unreserved")
        if len(payload.encode()) > res["bytes"]:
            raise ValueError("workflow_result_record_capacity")
        prior = conn.execute(
            "SELECT payload_json FROM workflow_results WHERE attempt_id=?",
            (result.attempt_id,),
        ).fetchone()
        if prior:
            if prior[0] != payload:
                raise ValueError("result_identity_conflict")
            return self._snapshot(conn, self._row(conn, result.workflow_id))
        parent = self._row(conn, result.workflow_id)
        if parent["status"] in {"completed", "closed", "stopped"} and not self._active(
            conn, result.workflow_id
        ):
            raise ValueError("workflow_terminal")
        conn.execute(
            "INSERT INTO workflow_results VALUES (?,?,?,?)",
            (result.attempt_id, result.workflow_id, _hash(payload), payload),
        )
        conn.execute(
            "UPDATE workflow_payload_reservations SET state='published' WHERE attempt_id=?",
            (result.attempt_id,),
        )
        conn.execute(
            "UPDATE workflow_task_attempts SET status=? WHERE id=?",
            (result.status, result.attempt_id),
        )
        conn.execute(
            "UPDATE workflow_runs SET progress_cursor=progress_cursor+1,updated_at=? WHERE id=?",
            (time.time(), result.workflow_id),
        )
        return self._snapshot(conn, self._row(conn, result.workflow_id))

    def _read_result(self, conn, workflow_id, attempt_id):
        row = conn.execute(
            "SELECT r.*,a.run_id,a.status AS attempt_status,p.bytes,p.state FROM workflow_results r JOIN workflow_task_attempts a ON a.id=r.attempt_id AND a.workflow_id=r.workflow_id JOIN workflow_payload_reservations p ON p.attempt_id=r.attempt_id AND p.workflow_id=r.workflow_id WHERE r.workflow_id=? AND r.attempt_id=?",
            (workflow_id, attempt_id),
        ).fetchone()
        if not row:
            raise ValueError("workflow_required_result_missing")
        result = WorkflowTaskResult.model_validate_json(row["payload_json"])
        if (
            (result.workflow_id, result.attempt_id, result.run_id)
            != (workflow_id, attempt_id, row["run_id"])
            or _hash(row["payload_json"]) != row["payload_hash"]
            or row["state"] != "published"
            or row["attempt_status"] not in {result.status, "closed"}
            or len(row["payload_json"].encode()) > row["bytes"]
        ):
            raise ValueError("invalid_workflow_result_ownership")
        attempt = conn.execute(
            "SELECT * FROM workflow_task_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        self._ownership(conn, WorkflowTaskAttempt(**dict(attempt)))
        return result

    def read_result(self, workflow_id: str, attempt_id: str) -> WorkflowTaskResult:
        """Missing or foreign mandatory evidence is an explicit failure."""
        with self.db.connection() as conn:
            return self._read_result(conn, workflow_id, attempt_id)

    @staticmethod
    def _window_roots(roots):
        roots = tuple(str(root) for root in roots)
        payload = json.dumps(roots, ensure_ascii=False, separators=(",", ":"))
        if (
            not 1 <= len(roots) <= 64
            or len(set(roots)) != len(roots)
            or any(not root or len(root.encode()) > 4096 for root in roots)
            or len(payload.encode()) > 32768
        ):
            raise ValueError("workflow_window_roots_bound")
        return payload

    def open_change_window(self, phase, reservation, roots):
        """Commit the original inert reservation before its first native job."""
        from tldw_chatbook.Workspaces.change_review_finalization import (
            ChangeReviewReservation,
        )

        if type(reservation) is not ChangeReviewReservation:
            raise TypeError("original change-review reservation required")
        body_error = None
        try:
            with self.db.automatic_work.transaction() as conn:
                try:
                    payload = self._window_roots(roots)
                    phase.check()
                    attempt = conn.execute(
                        "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=? AND workflow_id=?",
                        (phase.attempt.id, phase.attempt.workflow_id),
                    ).fetchone()
                    if attempt is None or attempt["status"] != "accepted":
                        raise ValueError("workflow_window_phase_required")
                    self._open_control(conn, self._row(conn, attempt["workflow_id"]))
                    ordinal = conn.execute(
                        "SELECT COALESCE(MAX(ordinal),0)+1 FROM workflow_change_windows WHERE workflow_id=?",
                        (attempt["workflow_id"],),
                    ).fetchone()[0]
                    if ordinal > 64:
                        raise ValueError("workflow_window_count_bound")
                    self._capacity(conn, attempt["workflow_id"], 65536)
                    conn.execute(
                        "INSERT INTO workflow_change_windows(id,workflow_id,phase_attempt_id,ordinal,roots_json,state,reserved_bytes) VALUES (?,?,?,?,?,'active',65536)",
                        (
                            reservation.id,
                            attempt["workflow_id"],
                            attempt["id"],
                            ordinal,
                            payload,
                        ),
                    )
                except BaseException as error:
                    body_error = error
                    raise
        except BaseException as error:
            # The original FULL context reraises the body only after rollback,
            # policy restoration and its connection/storage scope have returned.
            # Commit and replacement unwind errors carry no no-commit proof.
            if error is not body_error:
                raise
            return _WindowUnpublished(self, reservation, phase, error)
        return _WindowPublication(self, reservation)

    def anchor_change_window(self, window_id, authorization, owner):
        """Associate only an already committed original native member."""
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner

        # bind_native runs before the model loop installs its run ContextVar.
        # Its committed original owner and actual joined rows are the anchor.
        authorization._check_continuation()
        if (
            not authorization._binding_state
            or authorization._binding_state[0][0] is not owner
            or current_execution_owner() is not owner
            or owner.run_id != authorization.claim.launch_run_id
        ):
            raise ValueError("workflow_window_original_owner_required")
        authorization._runtime.worktree_activity.validate_claim(
            authorization._binding_state[0][1],
            owner=owner,
            footprint=authorization.physical_footprint,
        )
        with self.db.automatic_work.transaction() as conn:
            window = conn.execute(
                "SELECT w.*,p.native_attempt_id FROM workflow_change_windows w "
                "JOIN workflow_phase_attempts p ON p.id=w.phase_attempt_id WHERE w.id=?",
                (window_id,),
            ).fetchone()
            if (
                window is None
                or window["state"] != "active"
                or window["workflow_id"] != authorization.claim.workflow_id
                or window["native_attempt_id"] != authorization.claim.phase_attempt_id
            ):
                raise ValueError("workflow_window_member_required")
            committed = conn.execute(
                "SELECT t.run_id,t.status,r.status,a.state FROM workflow_task_attempts t "
                "JOIN agent_runs r ON r.id=t.run_id "
                "JOIN workflow_task_claims c ON c.task_attempt_id=t.id "
                "JOIN automatic_work_reservations a ON a.id=c.child_reservation_id "
                "WHERE t.id=? AND c.phase_attempt_id=?",
                (authorization.claim.id, window["phase_attempt_id"]),
            ).fetchone()
            if committed is None or tuple(committed) != (
                owner.run_id,
                "accepted",
                "running",
                "committed",
            ):
                raise ValueError("workflow_window_committed_member_required")
            conn.execute(
                "UPDATE workflow_change_windows SET anchor_attempt_id=? WHERE id=? AND anchor_attempt_id IS NULL",
                (authorization.claim.id, window_id),
            )

    def baseline_change_window(self, window_id, roots, disposition):
        """Publish actual B disposition; B alone never settles an effects window."""
        payload = self._window_roots(roots)
        if disposition not in {"completed", "disclosed_failure", "unresolved"}:
            raise ValueError("workflow_window_baseline_required")
        with self.db.automatic_work.transaction() as conn:
            if (
                conn.execute(
                    "UPDATE workflow_change_windows SET roots_json=?,baseline=? WHERE id=? AND state='active' AND baseline='pending'",
                    (payload, disposition, window_id),
                ).rowcount
                != 1
            ):
                raise ValueError("workflow_window_baseline_conflict")

    def unused_change_window(self, window_id):
        """Close a conclusively returned B with no actually committed member."""
        with self.db.automatic_work.transaction() as conn:
            if (
                conn.execute(
                    "UPDATE workflow_change_windows SET state=CASE baseline WHEN 'completed' THEN 'unused' ELSE 'disclosed_failure' END,ending='unused',snapshot_count=0 "
                    "WHERE id=? AND state='active' AND anchor_attempt_id IS NULL AND baseline IN ('completed','disclosed_failure')",
                    (window_id,),
                ).rowcount
                != 1
            ):
                raise ValueError("workflow_window_not_unused")

    def publish_change_window(self, window_id, *, run_id, records, kind):
        """Atomically publish native E rows, one ref representation and disposition."""
        records = tuple(records)
        if len(records) > 64:
            raise ValueError("workflow_window_refs_bound")
        with self.db.automatic_work.transaction() as conn:
            window = conn.execute(
                "SELECT w.*,t.run_id FROM workflow_change_windows w LEFT JOIN workflow_task_attempts t ON t.id=w.anchor_attempt_id WHERE w.id=?",
                (window_id,),
            ).fetchone()
            if (
                window is None
                or window["state"] != "active"
                or window["run_id"] != run_id
            ):
                raise ValueError("workflow_window_anchor_required")
            roots = set(json.loads(window["roots_json"]))
            if any(str(record.get("root") or "") not in roots for record in records):
                raise ValueError("workflow_window_foreign_root")
            ids = self.db._insert_change_snapshots(
                conn, run_id=run_id, records=records, kind=kind
            )
            conn.executemany(
                "INSERT INTO workflow_change_window_refs VALUES (?,?)",
                ((window_id, snapshot_id) for snapshot_id in ids),
            )
            failed = window["baseline"] != "completed" or any(
                record.get("tracking_error") for record in records
            )
            disposition = "disclosed_failure" if failed else "completed"
            conn.execute(
                "UPDATE workflow_change_windows SET state=?,ending=?,snapshot_count=? WHERE id=?",
                (disposition, disposition, len(ids), window_id),
            )

    @staticmethod
    def _active(conn, workflow_id, *, stage=_CURRENT_WORKFLOW_STAGE):
        if (
            stage >= 28
            and conn.execute(
                "SELECT 1 FROM workflow_change_windows WHERE workflow_id=? AND state IN ('active','unresolved') LIMIT 1",
                (workflow_id,),
            ).fetchone()
        ):
            return True
        return (
            conn.execute(
                "SELECT 1 FROM workflow_task_attempts WHERE workflow_id=? AND (status IN ('prepared','accepted','uncertain') OR retirement!='drained') UNION ALL SELECT 1 FROM workflow_phase_attempts WHERE workflow_id=? AND (status IN ('prepared','accepted','uncertain') OR retirement!='drained') UNION ALL SELECT 1 FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL LIMIT 1",
                (workflow_id, workflow_id, workflow_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _required_phase_claims(conn, workflow_id, claimed, completed):
        """Check the original task set for each native execution, including remaining-only retries."""
        scope = "AND ph.workflow_id=:owner" if workflow_id is not None else ""
        phases = conn.execute(
            f"""
            SELECT ph.id AS phase_attempt,ph.workflow_id,ph.plan_revision,ph.phase_id,ph.control_revision,
                   t.task_id,d.id AS decision_id,d.payload_json FROM workflow_phase_attempts ph
            LEFT JOIN workflow_tasks t ON (t.workflow_id,t.plan_revision,t.phase_id)=(ph.workflow_id,ph.plan_revision,ph.phase_id)
            LEFT JOIN workflow_decisions d ON t.ordinal=1 AND (d.workflow_id,d.plan_revision,d.phase_id,d.control_revision)=
                (ph.workflow_id,ph.plan_revision,ph.phase_id,ph.control_revision-1)
            WHERE ph.purpose='execution' AND ph.command_id IS NOT NULL {scope}
            ORDER BY ph.workflow_id,ph.plan_revision,ph.phase_id,ph.ordinal,ph.id,t.ordinal
        """,
            {"owner": workflow_id} if workflow_id is not None else (),
        )
        phases.row_factory = sqlite3.Row
        previous = None
        for phase_attempt_id, rows in groupby(phases, itemgetter("phase_attempt")):
            # One phase has at most 12 planned tasks. Only task ordinal 1 carries the decision body.
            planned = tuple(rows)
            first = planned[0]
            logical = (first["workflow_id"], first["plan_revision"], first["phase_id"])
            tasks = {row["task_id"] for row in planned}
            if previous == logical:
                # Only a remaining-work approval can prepare the same phase again.
                # Its retained result references specify which original tasks to omit.
                if first["payload_json"] is None:
                    raise ValueError("invalid_workflow_task_claim")
                try:
                    approval = WorkflowDecision.model_validate_json(
                        first["payload_json"]
                    )
                except ValueError:
                    raise ValueError("invalid_workflow_task_claim") from None
                if (
                    approval.id,
                    approval.workflow_id,
                    approval.plan_revision,
                    approval.phase_id,
                    approval.control_revision,
                    approval.action,
                    approval.reason,
                ) != (
                    first["decision_id"],
                    *logical,
                    first["control_revision"] - 1,
                    "approve",
                    "remaining_ready",
                ):
                    raise ValueError("invalid_workflow_task_claim")
                for ref in approval.result_refs:
                    retained = completed.get(ref)
                    if retained is not None and retained[:3] == logical:
                        tasks.discard(retained[3])
            if (
                not tasks
                or None in tasks
                or tasks != claimed.get(phase_attempt_id, set())
            ):
                raise ValueError("invalid_workflow_task_claim")
            previous = logical

    @staticmethod
    def _required_results(
        conn,
        workflow_id: str | None,
        *,
        status: str | None = None,
        stage=_CURRENT_WORKFLOW_STAGE,
    ) -> None:
        """Validate mandatory slot states from attempts, once per selected history or portfolio."""
        if status == "removed" and WorkflowRunsStore._active(
            conn, workflow_id, stage=stage
        ):
            raise ValueError("workflow_required_result_missing")
        joins = ""
        proof = "0"
        claim_owner = "NULL"
        claim_phase = "NULL"
        checkpoint_owner = "NULL"
        claimed = {}
        completed = {}
        if stage >= 26:
            joins = _CLAIM_JOINS
            claim_phase = "c.phase_attempt_id"
            claim_owner = _CLAIM_OWNED
        if stage >= 27:
            proof = _UNUSED_TASK_PROOF
            joins += " LEFT JOIN workflow_checkpoints k ON k.attempt_id=t.id"
            checkpoint_owner = "k.workflow_id"
        owner_scope = "WHERE t.workflow_id=:owner" if workflow_id is not None else ""
        rows = conn.execute(
            f"""SELECT t.id,t.workflow_id,t.plan_revision,t.phase_id,t.task_id,t.status,t.retirement,t.run_id,w.status AS parent,p.workflow_id AS reservation_owner,
                p.state AS reservation_state,r.workflow_id AS result_owner,g.status AS run_status,
                CASE WHEN t.status='closed' AND g.status!='done' THEN r.payload_json END AS closed_result,
                {checkpoint_owner} AS checkpoint_owner, {proof} AS unused, {claim_phase} AS claim_phase, {claim_owner} AS claim_owned
            FROM workflow_task_attempts t JOIN workflow_runs w ON w.id=t.workflow_id
            LEFT JOIN agent_runs g ON g.id=t.run_id
            LEFT JOIN workflow_payload_reservations p ON p.attempt_id=t.id
            LEFT JOIN workflow_results r ON r.attempt_id=t.id
            {joins} {owner_scope}""",
            {"owner": workflow_id} if workflow_id is not None else (),
        )
        rows.row_factory = sqlite3.Row
        for row in rows:
            owner = row["workflow_id"]
            state = row["reservation_state"]
            result = row["result_owner"] is not None
            removed = row["parent"] == "removed"
            required = row["status"] in {"completed", "failed", "uncertain"} or (
                row["status"] == "closed" and row["run_id"] is not None
            )
            if row["checkpoint_owner"] is not None and (
                row["checkpoint_owner"] != owner or not result
            ):
                raise ValueError("invalid_workflow_checkpoint")
            if any(
                value not in (None, owner)
                for value in (row["reservation_owner"], row["result_owner"])
            ):
                raise ValueError("workflow_required_result_missing")
            if row["claim_phase"] is not None:
                tasks = claimed.setdefault(row["claim_phase"], set())
                if not row["claim_owned"] or row["task_id"] in tasks:
                    raise ValueError("invalid_workflow_task_claim")
                tasks.add(row["task_id"])
            if (
                stage >= 26
                and row["status"] == "completed"
                and row["retirement"] == "drained"
                and row["run_id"] is not None
            ):
                completed[row["id"]] = (
                    owner,
                    row["plan_revision"],
                    row["phase_id"],
                    row["task_id"],
                )
            if result and row["run_status"] in _FAILED_NATIVE_STATES:
                outcome = row["status"]
                if outcome == "closed":
                    outcome = WorkflowTaskResult.model_validate_json(
                        row["closed_result"]
                    ).status
                if outcome == "completed":
                    raise ValueError("invalid_workflow_result_ownership")
            if removed:
                invalid = (
                    result
                    or state not in (None, "removed")
                    or (required and state is None)
                )
            elif state == "published":
                invalid = not result or not required
            elif state == "removed":
                invalid = result or required or not row["unused"]
            else:
                invalid = (
                    result
                    or required
                    or (row["status"] == "accepted" and state is None)
                )
            if invalid:
                raise ValueError("workflow_required_result_missing")

        if stage >= 26:
            WorkflowRunsStore._required_phase_claims(
                conn, workflow_id, claimed, completed
            )

    def remove_payloads(
        self, workflow_id: str, *, expected_control_revision: int
    ) -> WorkflowSnapshot:
        """Remove settled required results, retaining immutable plans and accounting tombstones."""
        with self.db.automatic_work.transaction() as conn:
            row = self._row(conn, workflow_id)
            if row["status"] not in {"completed", "stopped", "closed"} or self._active(
                conn, workflow_id
            ):
                raise ValueError("workflow_has_active_evidence")
            if self.db.goal_runs.unsettled_root(conn, row["chain_id"]):
                raise ValueError("workflow_has_active_evidence")
            if (
                type(expected_control_revision) is not int
                or expected_control_revision != row["control_revision"]
            ):
                raise ValueError("revision_conflict")
            self._required_results(conn, workflow_id, status=row["status"])
            # Authenticate only this history's private bytes before deleting them.
            self._published_slack(conn, workflow_id)
            from .recovery_workflow_schema import _validate_workflow_windows

            issues = _validate_workflow_windows(conn, workflow_id)
            if issues:
                raise ValueError(issues[0])
            conn.execute(
                "UPDATE workflow_runs SET status='removed',control_revision=control_revision+1,updated_at=? "
                "WHERE id=? AND control_revision=?",
                (time.time(), workflow_id, expected_control_revision),
            )
            conn.execute(
                "DELETE FROM workflow_checkpoints WHERE workflow_id=?", (workflow_id,)
            )
            conn.execute(
                "DELETE FROM workflow_results WHERE workflow_id=?", (workflow_id,)
            )
            conn.execute(
                "UPDATE workflow_payload_reservations SET state='removed' WHERE workflow_id=?",
                (workflow_id,),
            )
            conn.execute(
                "DELETE FROM workflow_change_window_refs WHERE window_id IN (SELECT id FROM workflow_change_windows WHERE workflow_id=?)",
                (workflow_id,),
            )
            conn.execute(
                "UPDATE workflow_change_windows SET state='removed',roots_json='[]',reserved_bytes=0,snapshot_count=0 WHERE workflow_id=?",
                (workflow_id,),
            )
            return self._snapshot(conn, self._row(conn, workflow_id))

    def _remaining_partition(self, conn, workflow_id, revision, phase_id):
        """Return exact retained result refs and conclusively unused task IDs."""
        if (
            self._active(conn, workflow_id)
            or not conn.execute(
                "SELECT 1 FROM workflow_phase_attempts WHERE workflow_id=? AND plan_revision=? AND phase_id=? AND native_attempt_id IS NOT NULL AND retirement='drained'",
                (workflow_id, revision, phase_id),
            ).fetchone()
        ):
            raise ValueError("remaining_requires_clean_checkpoint")
        retained, remaining = {}, []
        for task in conn.execute(
            "SELECT task_id FROM workflow_tasks WHERE workflow_id=? AND plan_revision=? AND phase_id=? ORDER BY ordinal",
            (workflow_id, revision, phase_id),
        ):
            attempts = conn.execute(
                "SELECT * FROM workflow_task_attempts WHERE workflow_id=? AND plan_revision=? AND task_id=? ORDER BY ordinal",
                (workflow_id, revision, task["task_id"]),
            ).fetchall()
            if not attempts:
                raise ValueError("remaining_requires_repair")
            completed = []
            for attempt in attempts:
                if attempt["run_id"] is None:
                    if not self._closed_without_native_run(
                        conn, workflow_id, attempt["id"]
                    ) or not self._never_run_task(conn, workflow_id, attempt["id"]):
                        raise ValueError("remaining_requires_repair")
                elif (
                    attempt["status"] == "completed"
                    and attempt["retirement"] == "drained"
                ):
                    result = self._read_result(conn, workflow_id, attempt["id"])
                    if result.status != "completed":
                        raise ValueError("remaining_requires_repair")
                    completed.append(attempt["id"])
                else:
                    raise ValueError("remaining_requires_repair")
            if completed:
                if len(completed) != 1 or attempts[-1]["id"] != completed[0]:
                    raise ValueError("remaining_requires_repair")
                retained[task["task_id"]] = completed[0]
            else:
                remaining.append(task["task_id"])
        if not remaining:
            raise ValueError("remaining_partition_required")
        return retained, tuple(remaining)

    @staticmethod
    def _attempt_decision(conn, attempt):
        rows = conn.execute(
            "SELECT payload_json FROM workflow_decisions WHERE workflow_id=? AND plan_revision=? AND phase_id=? AND control_revision=?",
            (
                attempt["workflow_id"],
                attempt["plan_revision"],
                attempt["phase_id"],
                attempt["control_revision"] - 1,
            ),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("workflow_decision_link_unproven")
        decision = WorkflowDecision.model_validate_json(rows[0][0])
        if decision.action != "approve" or decision.reason not in {
            "ready",
            "remaining_ready",
        }:
            raise ValueError("workflow_decision_link_unproven")
        return decision

    def save_decision(self, decision: WorkflowDecision) -> WorkflowSnapshot:
        """Append an inert decision record with CAS; it is not phase authorization."""
        with self.db.automatic_work.transaction() as conn:
            self._save_decision(conn, decision)
            return self._snapshot(conn, self._row(conn, decision.workflow_id))

    def _save_decision(self, conn, decision, *, status=None):
        decision = WorkflowDecision.model_validate_json(decision.canonical_json())
        payload = decision.canonical_json()
        row = self._row(conn, decision.workflow_id)
        prior = conn.execute(
            "SELECT payload_json FROM workflow_decisions WHERE id=?", (decision.id,)
        ).fetchone()
        if prior:
            if prior[0] != payload:
                raise ValueError("decision_identity_conflict")
            return
        for ref in decision.result_refs:
            self._read_result(conn, decision.workflow_id, ref)
        self._capacity(conn, decision.workflow_id, len(payload.encode()))
        self._cas(
            conn,
            decision.workflow_id,
            decision.control_revision,
            status or row["status"],
        )
        conn.execute(
            "INSERT INTO workflow_decisions VALUES (?,?,?,?,?,?)",
            (
                decision.id,
                decision.workflow_id,
                decision.plan_revision,
                decision.phase_id,
                decision.control_revision,
                payload,
            ),
        )
