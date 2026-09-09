"""Durable goal lifecycle, exact result review and conservative recovery."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tldw_chatbook.Agents.automatic_work_budget import RuntimeRecoveryResult
from tldw_chatbook.Agents.goal_models import (
    GoalDecision,
    GoalHistoryEntry,
    GoalIterationResult,
    GoalRequest,
    GoalSnapshot,
    RecoveryResolution,
)
from tldw_chatbook.Chat.chat_persistence_service import (
    ChatPersistenceService,
    GoalConversationConflict,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class GoalRunService:
    """Own saved lifecycle; execution admission remains the shared ledger’s fence."""

    def __init__(self, db: AgentRunsDB, persistence: ChatPersistenceService) -> None:
        self.db = db
        self.persistence = persistence
        self._control_listener: Callable[[GoalSnapshot], None] | None = None

    def list_goals(self, *, limit: int = 50) -> tuple[GoalHistoryEntry, ...]:
        """List at most 100 body-free recent history entries without recovery."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid_history_limit")
        with self.db.connection() as conn:
            return tuple(
                GoalHistoryEntry(**dict(row))
                for row in conn.execute(
                    "SELECT id,conversation_id,revision,status,pause_reason,updated_at FROM goal_runs ORDER BY updated_at DESC,id LIMIT ?",
                    (limit,),
                )
            )

    def get(self, goal_id: str) -> GoalSnapshot:
        """Inspect saved state without dispatch, recovery audit or provisioning."""
        return self.db.goal_runs.get(goal_id)

    def pause(self, goal_id: str) -> GoalSnapshot:
        snapshot = self.db.goal_runs.control(goal_id)
        if self._control_listener:
            self._control_listener(snapshot)
        return snapshot

    def stop(self, goal_id: str) -> GoalSnapshot:
        snapshot = self.db.goal_runs.control(goal_id, stop=True)
        if self._control_listener:
            self._control_listener(snapshot)
        return snapshot

    def checkpoint(self, result: GoalIterationResult) -> GoalSnapshot:
        """Atomically retain a native iteration; never dispatch its successor."""
        goal = self.get(result.goal_id)
        return self.db.goal_runs.checkpoint(
            result,
            binding_available=goal.request is not None
            and not self._binding_reason(goal.request),
        )

    def checkpoint_failed(self, goal_id: str) -> GoalSnapshot:
        """Fence an already drained increment when checkpoint durability is uncertain."""
        with self.db.automatic_work.transaction() as conn:
            goal = self.get(goal_id)
            if goal.status in {"completed", "removed", "closed"}:
                return goal
            conn.execute(
                "UPDATE automatic_work_chains SET status='review_required',pause_reason='checkpoint_unavailable' WHERE id=?",
                (goal.chain_id,),
            )
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='review_required' WHERE id IN (SELECT attempt_id FROM goal_iterations WHERE goal_id=?) AND state IN ('prepared','accepted')",
                (goal_id,),
            )
            conn.execute(
                "UPDATE goal_runs SET status='recovery_required',pause_reason='checkpoint_unavailable',revision=revision+1 WHERE id=?",
                (goal_id,),
            )
            return self.get(goal_id)

    def completion_check(
        self,
        goal_id: str,
        *,
        checkpoint_id: str | None = None,
        artifact_digest: str | None = None,
    ) -> GoalDecision:
        """Recheck current versions for review; never approve or release an attempt."""
        from tldw_chatbook.Agents.goal_iteration import (
            evaluate_iteration,
            goal_criteria,
            refresh_evidence,
        )
        from tldw_chatbook.Agents.goal_models import GoalDecision

        goal = self.get(goal_id)
        if goal.request is None:
            raise ValueError("goal_payload_removed")
        with self.db.connection() as conn:
            unsettled = conn.execute(
                "SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id WHERE i.goal_id=? AND a.state IN ('prepared','accepted','review_required')",
                (goal_id,),
            ).fetchone()
        if goal.status == "recovery_required" or unsettled:
            return GoalDecision(action="recovery_required", reason="uncertain_effect")
        if not goal.checkpoints:
            return GoalDecision(action="continue", reason="no_checkpoint")
        checkpoint = goal.checkpoints[-1]
        if (checkpoint_id is not None and checkpoint_id != checkpoint.id) or (
            artifact_digest is not None
            and artifact_digest != checkpoint.artifact_digest
        ):
            raise ValueError("stale_result_review")
        evidence = (
            tuple(
                refresh_evidence(goal, item)
                for item in self.db.goal_runs.evidence(goal_id)
            )
            if not self._binding_reason(goal.request)
            else ()
        )
        decision = evaluate_iteration(
            checkpoint.report, evidence, checkpoint, goal_criteria(goal)
        )
        return decision.model_copy(
            update={
                "checkpoint_id": checkpoint.id,
                "artifact_digest": checkpoint.artifact_digest,
            }
        )

    def defer(self, goal_id: str, *, reason: str, delay: float) -> GoalSnapshot:
        """Persist one bounded retry time inside the original chain deadline."""
        import time

        from tldw_chatbook.Agents.automatic_work_budget import (
            AutomaticWorkLimits,
            AutomaticWorkRefused,
        )

        goal = self.get(goal_id)
        if goal.status != "ready":
            return goal
        try:
            self.db.automatic_work.check_active(
                goal.chain_id,
                limits=AutomaticWorkLimits.from_settings("goal_iteration"),
            )
        except AutomaticWorkRefused as exc:
            return self._state(goal, "paused", exc.reason)
        if (
            reason not in {"primary_capacity", "pre_effect_rate_limit"}
            or not 0 < delay <= 30
        ):
            raise ValueError("invalid_goal_retry")
        with self.db.automatic_work.transaction() as conn:
            goal = self.get(goal_id)
            if goal.status != "ready":
                return goal
            when = time.time() + delay
            if goal.accounting.deadline_at is not None:
                when = min(when, goal.accounting.deadline_at)
            conn.execute(
                "INSERT INTO goal_waits VALUES (?,?,?) ON CONFLICT(goal_id) DO UPDATE SET retry_at=excluded.retry_at,reason=excluded.reason",
                (goal_id, when, reason),
            )
            conn.execute(
                "UPDATE goal_runs SET pause_reason=?,revision=revision+1 WHERE id=?",
                (reason, goal_id),
            )
            return self.get(goal_id)

    def clear_wait(self, goal_id: str) -> None:
        with self.db.automatic_work.transaction() as conn:
            conn.execute("DELETE FROM goal_waits WHERE goal_id=?", (goal_id,))

    def project_recovery(
        self, audit: RuntimeRecoveryResult
    ) -> tuple[GoalSnapshot, ...]:
        """Project the shared owner's exact audit, never run another takeover."""
        if (
            audit is not getattr(self.db.automatic_work, "recovery_result", None)
            or audit is None
        ):
            raise ValueError("untrusted_recovery_audit")
        with self.db.automatic_work.transaction() as conn:
            self.db.automatic_work._check_runtime_owner(conn, audit.owner_id)
            for goal_id, revision, state in audit.goals:
                conn.execute(
                    "UPDATE goal_runs SET status=?,pause_reason=?,revision=revision+1 WHERE id=? AND revision=? "
                    "AND status NOT IN ('completed','removed','closed','stopped')",
                    (
                        state,
                        "interrupted_work"
                        if state == "recovery_required"
                        else "restart_checkpoint",
                        goal_id,
                        revision,
                    ),
                )
            return tuple(self.get(goal_id) for goal_id, _, _ in audit.goals)

    def resume(self, goal_id: str, *, expected_revision: int) -> GoalSnapshot:
        """Permit future admission only from a clean saved checkpoint, without refill."""
        from tldw_chatbook.Agents.automatic_work_budget import (
            AutomaticWorkLimits,
            AutomaticWorkRefused,
        )

        goal = self.get(goal_id)
        if goal.revision != expected_revision:
            raise ValueError("revision_conflict")
        if goal.status != "paused" or not goal.checkpoints or goal.accounting.uncertain:
            raise ValueError("resume_requires_clean_checkpoint")
        reason = self._binding_reason(goal.request)
        if reason:
            return self._state(goal, "paused", reason)
        try:
            with self.db.automatic_work._admission_transaction(goal.chain_id) as conn:
                current = self.get(goal_id)
                if current.revision != expected_revision:
                    raise ValueError("revision_conflict")
                if conn.execute(
                    "SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id WHERE i.goal_id=? AND a.state IN ('prepared','accepted','review_required')",
                    (goal_id,),
                ).fetchone():
                    raise ValueError("resume_requires_clean_checkpoint")
                limits = AutomaticWorkLimits.from_settings("goal_iteration")
                for kind in ("generation", "model_call", "tokens"):
                    self.db.automatic_work._check_admission(
                        conn, goal.chain_id, kind=kind, amount=1, limits=limits
                    )
                conn.execute(
                    "UPDATE goal_runs SET status='ready',pause_reason=NULL,revision=revision+1 WHERE id=?",
                    (goal_id,),
                )
        except AutomaticWorkRefused as exc:
            return self._state(self.get(goal_id), "paused", exc.reason)
        return self.get(goal_id)

    def review_result(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        checkpoint_id: str,
        artifact_digest: str,
        accepted: bool,
    ) -> GoalSnapshot:
        """Record quality only for exact fresh evidence; review spends no allowance."""
        if type(accepted) is not bool:
            raise TypeError("accepted must be bool")
        with self.db.automatic_work.transaction() as conn:
            goal = self.get(goal_id)
            if goal.revision != expected_revision:
                raise ValueError("revision_conflict")
            if goal.status != "awaiting_result_review":
                raise ValueError("result_review_unavailable")
            decision = self.completion_check(
                goal_id, checkpoint_id=checkpoint_id, artifact_digest=artifact_digest
            )
            if decision.action not in {"awaiting_result_review", "completed"}:
                raise ValueError("objective_proof_unavailable")
            conn.execute(
                "UPDATE goal_runs SET status=?,pause_reason=?,revision=revision+1 WHERE id=?",
                (
                    "completed" if accepted else "paused",
                    "quality_accepted" if accepted else "quality_rejected",
                    goal_id,
                ),
            )
            return self.get(goal_id)

    def resolve_recovery(
        self, goal_id: str, *, expected_revision: int, resolution: RecoveryResolution
    ) -> GoalSnapshot:
        """Close an interruption conservatively; no human assertion certifies effects."""
        from tldw_chatbook.Agents.goal_models import RecoveryResolution

        if type(resolution) is not RecoveryResolution:
            raise TypeError("typed recovery resolution required")
        with self.db.automatic_work.transaction() as conn:
            goal = self.get(goal_id)
            if goal.revision != expected_revision:
                raise ValueError("revision_conflict")
            if goal.status != "recovery_required":
                raise ValueError("recovery_unavailable")
            conn.execute(
                "UPDATE goal_runs SET status='closed',pause_reason='closed_uncertain',revision=revision+1 WHERE id=?",
                (goal_id,),
            )
            return self.get(goal_id)

    def remove_payloads(self, goal_id: str) -> GoalSnapshot:
        return self.db.goal_runs.remove_payloads(goal_id)

    def create(self, request: GoalRequest, *, launch_id: str) -> GoalSnapshot:
        """Persist intent first and reconcile exact chat/workspace identity.

        Store failures retain Starting and are retryable through the same launch.
        Binding and identity conflicts pause setup. Ready is not execution authority.
        """
        snapshot = self.db.goal_runs.create(request, launch_id=launch_id)
        # Completed provisioning is not a request to reset runtime lifecycle.
        if snapshot.status not in {"starting", "paused", "ready"}:
            return snapshot
        with self.db.connection() as conn:
            accepted = conn.execute(
                "SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id WHERE i.goal_id=? AND a.accepted_at IS NOT NULL LIMIT 1",
                (snapshot.id,),
            ).fetchone()
        if accepted:
            return snapshot
        reason = self._binding_reason(snapshot.request)
        if not snapshot.policy.admission_enabled:
            reason = "goal_policy_disabled"
        if reason:
            return self._state(snapshot, "paused", reason)
        try:
            self.persistence.provision_goal_conversation(snapshot.provisioning)
        except GoalConversationConflict:
            return self._state(snapshot, "paused", "conversation_identity_conflict")
        except Exception:  # noqa: BLE001 - reconcile an uncertain cross-store commit
            # The cross-store call may already have committed. Do not delete or
            # allocate another conversation, expose error bodies, or run a model.
            return self._state(snapshot, "starting", "provisioning_pending")
        # Binding authority can change while either external store is writing.
        reason = self._binding_reason(snapshot.request)
        return self._state(snapshot, "paused" if reason else "ready", reason)

    def _state(
        self, snapshot: GoalSnapshot, status: str, reason: str | None
    ) -> GoalSnapshot:
        if (snapshot.status, snapshot.pause_reason) == (status, reason):
            return snapshot
        try:
            return self.db.goal_runs.set_provisioning(
                snapshot, status=status, pause_reason=reason
            )
        except ValueError as exc:
            if str(exc) != "revision_conflict":
                raise
            # A concurrent delivery won the CAS; never overwrite its projection.
            return self.get(snapshot.id)

    def _binding_reason(self, request: GoalRequest) -> str | None:
        registry = self.persistence.workspace_registry
        if registry is None:
            return "binding_missing"
        for reference in (request.binding, *request.source_bindings):
            workspace = registry.get_workspace(reference.workspace_id)
            binding = registry.get_runtime_binding(reference.binding_id)
            if workspace is None or binding is None:
                return "binding_missing"
            if workspace.archived or binding.status.value != "ready":
                return "binding_unavailable"
            if (
                binding.workspace_id != reference.workspace_id
                or binding.binding_kind.value != "local-filesystem"
                or binding.locator != reference.locator
                or binding.metadata.get("access", "ro") != reference.access
            ):
                return "binding_changed"
            path = Path(reference.locator)
            if not path.is_dir():
                return "binding_missing"
            if str(path.resolve()) != reference.locator:
                return "binding_changed"
        return None
