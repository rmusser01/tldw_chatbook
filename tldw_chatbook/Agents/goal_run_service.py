"""Durable goal launch/provisioning, without any model or tool dispatch authority."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Agents.goal_models import GoalRequest, GoalSnapshot
from tldw_chatbook.Chat.chat_persistence_service import (
    ChatPersistenceService,
    GoalConversationConflict,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class GoalRunService:
    """Own recoverable setup. Runtime iteration admission is a separate fence."""

    def __init__(self, db: AgentRunsDB, persistence: ChatPersistenceService) -> None:
        self.db = db
        self.persistence = persistence

    def get(self, goal_id: str) -> GoalSnapshot:
        """Inspect saved state without dispatch, recovery audit or provisioning."""
        return self.db.goal_runs.get(goal_id)

    def create(self, request: GoalRequest, *, launch_id: str) -> GoalSnapshot:
        """Persist intent first and reconcile exact chat/workspace identity.

        Store failures retain Starting and are retryable through the same launch.
        Binding and identity conflicts pause setup. Ready is not execution authority.
        """
        snapshot = self.db.goal_runs.create(request, launch_id=launch_id)
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
