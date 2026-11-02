"""Private accepted workflow custody; execution coordination belongs to ConsoleRuntime."""

from __future__ import annotations

import os
from threading import RLock

from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

_KEY = object()


def native_workflow_qualified() -> bool:
    """Return the metadata-only platform qualification for native group custody."""
    return os.name == "posix" and hasattr(os, "killpg")


def _live_owner(controller, session_id, attempt):
    """Validate native live owner, selected session and exact durable database."""
    bridge = controller._agent_bridge
    owner = controller._fleet_wake
    if (
        controller._disposed
        or controller._shutdown_requested.is_set()
        or owner.runtime_owner_id != attempt.owner_id
    ):
        raise AutomaticWorkRefused("workflow_authority_changed")
    session = next((s for s in controller.store.sessions() if s.id == session_id), None)
    if (
        session is None
        or attempt.session_id != session_id
        or (session.persisted_conversation_id or session.id) != attempt.conversation_id
    ):
        raise AutomaticWorkRefused("workflow_session_changed")
    return bridge.runs_db.automatic_work, session


class WorkflowPhaseAuthorization:
    """Process-local accepted custody, never constructible from a saved ID alone."""

    def __init__(self, controller, attempt, *, _key, _session=None):
        if _key is not _KEY:
            raise PermissionError("workflow authority is coordinator-internal")
        self._controller = controller
        self._task_scope_lock = RLock()
        self._task_scopes = {}
        self.attempt = attempt
        self._ledger, self._session = _live_owner(
            controller, attempt.session_id, attempt
        )
        if self._session is not _session:
            raise AutomaticWorkRefused("workflow_session_changed")
        self.context = AutomaticWorkContext(
            self._ledger,
            attempt.chain_id,
            attempt.owner_id,
            attempt.id,
            attempt_kind="workflow_phase",
            workflow=self,
        )
        self.context.mark_accepted()

    def check(self) -> None:
        """Recheck revocation before every attributable operation."""
        ledger, session = _live_owner(
            self._controller, self.attempt.session_id, self.attempt
        )
        if ledger is not self._ledger:
            raise AutomaticWorkRefused("workflow_database_changed")
        if session is not self._session:
            raise AutomaticWorkRefused("workflow_session_changed")
        from tldw_chatbook.Agents.activation import execution

        with execution(self._ledger), self._ledger._db.connection() as conn:
            self._ledger._workflow_current(
                conn, self.attempt.id, self.attempt.owner_id, accepted=True
            )

    def __repr__(self):
        return "WorkflowPhaseAuthorization(authority=<redacted>)"


def _accept_workflow_phase(controller, attempt_id: str, *, session_id: str):
    """Issue custody only to the live owner that won this FULL acceptance.

    Duplicate accepted IDs return no capability. The later phase coordinator keeps
    the original process-local object; reload/import cannot recreate it.
    """
    if not native_workflow_qualified():
        raise AutomaticWorkRefused("workflow_platform_unqualified")
    owner_id = controller._fleet_wake.runtime_owner_id
    ledger = controller._agent_bridge.runs_db.automatic_work
    attempt = ledger.read_workflow_attempt(attempt_id, owner_id=owner_id)
    current_ledger, session = _live_owner(controller, session_id, attempt)
    if current_ledger is not ledger:
        raise AutomaticWorkRefused("workflow_database_changed")
    from tldw_chatbook.Agents.activation import execution

    with execution(ledger):
        if not ledger.accept_workflow_phase(attempt_id, owner_id=owner_id):
            return None
    accepted = ledger.read_workflow_attempt(attempt_id, owner_id=owner_id)
    return WorkflowPhaseAuthorization(controller, accepted, _key=_KEY, _session=session)
