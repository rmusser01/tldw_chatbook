"""Native accepted workflow custody and its durable SQL fences (ADR-356)."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields
from inspect import getattr_static
from threading import RLock
from weakref import ReferenceType, WeakSet, ref

from tldw_chatbook import config

from .automatic_work_budget import AutomaticWorkLimits, AutomaticWorkRefused


_PHASE_KEY = object()
_KEY = object()
_POLICY_ENV = ("TLDW_AGENTS_WORKFLOW_RUNS_ENABLED",) + tuple(
    "TLDW_AGENTS_MAX_WORKFLOW_" + item.name.upper()
    for item in fields(AutomaticWorkLimits)
)


def workflow_policy_identity():
    return config.current_config_identity(), tuple(
        os.environ.get(name) for name in _POLICY_ENV
    )


def require_workflow_fence_sources(phase, ledger):
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

    if not isinstance(phase, WorkflowPhaseAuthorization) or not isinstance(
        ledger, AutomaticWorkLedger
    ):
        raise AutomaticWorkRefused("workflow_native_fence_required")
    if phase._ledger is not ledger:
        raise AutomaticWorkRefused("workflow_database_changed")


@dataclass(frozen=True)
class _WorkflowFence:
    phase: object
    connection: object
    limits: AutomaticWorkLimits
    policy: tuple
    thread: object
    operation: str
    key: object


def workflow_fence_connection(fence, ledger):
    """Use only the phase's active, thread-bound connection and current policy."""
    if not isinstance(fence, _WorkflowFence):
        raise AutomaticWorkRefused("workflow_native_fence_changed")
    require_workflow_fence_sources(fence.phase, ledger)
    if (
        fence.key is not _KEY
        or fence.phase._active_fence is not fence
        or fence.thread is not threading.current_thread()
        or getattr(ledger._db._thread_local, "conn", None) is not fence.connection
        or fence.operation not in {"child_commit", "control"}
        or fence.policy != workflow_policy_identity()
    ):
        raise AutomaticWorkRefused("workflow_native_fence_changed")
    return fence.connection


@dataclass(frozen=True)
class _WorkflowTaskDispatch:
    phase: object
    task: object
    owner: object
    activity: object
    service: object
    decision_id: str
    revision: int
    payload: str
    key: object


def workflow_dispatch_identity(proof, ledger):
    """SQL consumers inspect sealed originals; never acquire a control lock."""
    if (
        type(proof) is not _WorkflowTaskDispatch
        or proof.key is not _KEY
        or proof.phase._ledger is not ledger
        or proof.phase._retired
        or proof.phase._paused_tasks.get(proof.task.claim.id) is not proof
        or len(proof.phase._committed_tasks.get(proof.task.claim.id, ())) != 4
        or not all(
            retained is original
            for retained, original in zip(
                proof.phase._committed_tasks.get(proof.task.claim.id, ()),
                (proof.task, proof.owner, proof.activity, proof.service),
                strict=True,
            )
        )
    ):
        raise AutomaticWorkRefused("workflow_authority_changed")
    return proof


def native_workflow_qualified() -> bool:
    """Return the platform qualification used by native Console workflows."""
    from tldw_chatbook.Utils.platform_files import supports_native_process_groups

    return supports_native_process_groups()


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

    def __init__(self, controller, attempt, *, _key, _session=None, _planning=None):
        if _key is not _PHASE_KEY:
            raise PermissionError("workflow authority is coordinator-internal")
        self._controller = controller
        self._task_scope_lock = RLock()
        self._task_scopes = {}
        self._control_lock = RLock()
        self._active_fence = None
        self._committed_tasks = {}
        self._paused_tasks = {}
        self._retired = False
        self._quiescence = None
        self.attempt = attempt
        self._planning_preparation = ref(_planning) if _planning is not None else None
        self._ledger, self._session = _live_owner(
            controller, attempt.session_id, attempt
        )
        if self._session is not _session:
            raise AutomaticWorkRefused("workflow_session_changed")
        from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

        self.context = AutomaticWorkContext(
            self._ledger,
            attempt.chain_id,
            attempt.owner_id,
            attempt.id,
            attempt_kind="workflow_phase",
            workflow=self,
        )
        self.context.mark_accepted()

    def _check_owner(self):
        """Check original live identities without consulting phase control."""
        ledger, session = _live_owner(
            self._controller, self.attempt.session_id, self.attempt
        )
        if ledger is not self._ledger:
            raise AutomaticWorkRefused("workflow_database_changed")
        if session is not self._session:
            raise AutomaticWorkRefused("workflow_session_changed")

    def check(self) -> None:
        """Strict new-work check; a paused phase never authorizes preparation."""
        self._check_owner()
        if self._retired:
            raise AutomaticWorkRefused("workflow_authority_changed")
        from tldw_chatbook.Agents.activation import execution

        with execution(self._ledger), self._ledger._db.connection() as conn:
            self._ledger._workflow_current(
                conn, self.attempt.id, self.attempt.owner_id, accepted=True
            )

    @contextmanager
    def _fence(self, operation, *, _launch=None, _control=None):
        from tldw_chatbook.Agents.activation import execution

        if operation not in {"child_commit", "control"}:
            raise AutomaticWorkRefused("workflow_native_fence_operation_changed")
        self._check_owner()
        if operation == "control":
            from tldw_chatbook.Agents.workflow_models import WorkflowDecision

            if not (
                self.attempt.purpose == "planning"
                and type(_control) is int
                and _control > 0
                or self.attempt.purpose == "execution"
                and type(_control) is WorkflowDecision
                and _control.workflow_id == self.attempt.workflow_id
                and _control.plan_revision == self.attempt.revision_id
                and _control.phase_id == self.attempt.phase_id
                and _control.action in {"pause", "stop"}
            ):
                raise AutomaticWorkRefused("workflow_native_control_required")
        if operation == "child_commit":
            from tldw_chatbook.Agents.agent_service import AgentService
            from tldw_chatbook.Agents.execution_capacity import current_execution_owner

            if _launch is None:
                raise AutomaticWorkRefused("workflow_native_commit_required")
            task, owner, activity, service = _launch
            if (
                not isinstance(service, AgentService)
                or current_execution_owner() is not owner
                or self._task_scopes.get(task.task.id) is not task
                or task.phase is not self
                or task._native_dispatch != [service]
                or service._workflow_authorization is not task
                or service._automatic_work is not self.context
                or service.db is not self._ledger._db
                or service.runtime_capacity is not owner.capacity
            ):
                raise AutomaticWorkRefused("workflow_native_commit_required")
            task._check_sources()
            task._runtime.worktree_activity.validate_claim(
                activity, owner=owner, footprint=task.physical_footprint
            )
        before = workflow_policy_identity()
        if operation == "child_commit":
            self._ledger._workflow_enabled()
        limits = AutomaticWorkLimits.from_settings("workflow_phase")
        if before != workflow_policy_identity():
            raise AutomaticWorkRefused("workflow_policy_source_changed")
        with execution(self._ledger), self._ledger._db.connection() as conn:
            fence = _WorkflowFence(
                self, conn, limits, before, threading.current_thread(), operation, _KEY
            )
            with self._control_lock:
                if self._active_fence is not None:
                    raise AutomaticWorkRefused("workflow_control_reentry")
                self._active_fence = fence
                try:
                    workflow_fence_connection(fence, self._ledger)
                    yield fence
                finally:
                    self._active_fence = None

    def _commit_task_launch(self, task, owner, activity, service):
        with self._fence(
            "child_commit", _launch=(task, owner, activity, service)
        ) as fence:
            if self._retired or task.claim.id in self._committed_tasks:
                raise AutomaticWorkRefused("child_already_launched")
            committed = self._ledger.commit(
                task.claim.reservation_id,
                owner_id=self.attempt.owner_id,
                limits=fence.limits,
                _native=fence,
            )
            if committed:
                self._committed_tasks[task.claim.id] = (task, owner, activity, service)
            return committed

    def _task_dispatch(self, task):
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner
        from tldw_chatbook.Agents.workflow_task_scope import current_task_authorization

        self._check_owner()
        from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

        service = (
            task._required_evidence[0].service if task._required_evidence else None
        )
        with owned_goal_storage(service) if service is not None else nullcontext():
            task._check_sources()
        with self._control_lock:
            proof = self._paused_tasks.get(task.claim.id)
            if proof is None:
                return None
            workflow_dispatch_identity(proof, self._ledger)
        if (
            proof.task is not task
            or current_task_authorization() is not task
            or current_execution_owner() is not proof.owner
            or not proof.owner.is_admitted
            or proof.owner.run_id not in (None, task.claim.launch_run_id)
            or task._native_dispatch != [proof.service]
            or proof.service._workflow_authorization is not task
            or self._task_scopes.get(task.task.id) is not task
        ):
            raise AutomaticWorkRefused("workflow_task_owner_changed")
        task._runtime.worktree_activity.validate_claim(
            proof.activity, owner=proof.owner, footprint=task.physical_footprint
        )
        return proof

    def _control(self, decision=None, *, expected_control_revision=None):
        """Atomically revoke new work; only original committed tasks may finish Pause."""
        from tldw_chatbook.Agents.workflow_models import WorkflowDecision
        from tldw_chatbook.Agents.workflow_task_scope import require_workflow_runtime

        if decision is None:
            if self.attempt.purpose != "planning" or expected_control_revision is None:
                raise ValueError("workflow_control_identity_conflict")
            store = self._ledger._db.workflow_runs
            with self._fence("control", _control=expected_control_revision) as fence:
                if self._retired:
                    raise AutomaticWorkRefused("workflow_authority_changed")
                with self._ledger.transaction(_native=fence) as conn:
                    self._ledger._check_runtime_owner(conn, self.attempt.owner_id)
                    native = self._ledger._attempt(
                        conn, self.attempt.id, self.attempt.owner_id, "workflow_phase"
                    )
                    current = store._row(conn, self.attempt.workflow_id)
                    if (
                        native["state"] != "accepted"
                        or current["control_revision"] != self.attempt.control_revision
                    ):
                        raise AutomaticWorkRefused("workflow_authority_changed")
                    store._cas(
                        conn,
                        self.attempt.workflow_id,
                        expected_control_revision,
                        "stopped",
                    )
            runtime = require_workflow_runtime(self)
            for prepared in tuple(runtime._workflow_preparations):
                if prepared.phase is self:
                    prepared.cancel_event.set()
            return store.get(self.attempt.workflow_id)
        if (
            type(decision) is not WorkflowDecision
            or decision.action not in {"pause", "stop"}
            or decision.workflow_id != self.attempt.workflow_id
            or decision.plan_revision != self.attempt.revision_id
            or decision.phase_id != self.attempt.phase_id
            or self.attempt.purpose != "execution"
        ):
            raise ValueError("workflow_control_identity_conflict")
        payload = decision.canonical_json()
        store = self._ledger._db.workflow_runs
        with self._fence("control", _control=decision) as fence:
            if self._retired:
                raise AutomaticWorkRefused("workflow_authority_changed")
            with self._ledger.transaction(_native=fence) as conn:
                self._ledger._check_runtime_owner(conn, self.attempt.owner_id)
                native = self._ledger._attempt(
                    conn, self.attempt.id, self.attempt.owner_id, "workflow_phase"
                )
                if native["state"] != "accepted":
                    raise AutomaticWorkRefused("acceptance_required")
                prior = conn.execute(
                    "SELECT payload_json FROM workflow_decisions WHERE id=?",
                    (decision.id,),
                ).fetchone()
                if prior is not None:
                    if prior[0] != payload:
                        raise ValueError("decision_identity_conflict")
                    fresh = False
                else:
                    current = store._row(conn, decision.workflow_id)
                    if (
                        current["status"]
                        not in {"running", "awaiting_approval", "paused"}
                        or decision.action == "pause"
                        and current["control_revision"] != self.attempt.control_revision
                    ):
                        raise AutomaticWorkRefused("workflow_authority_changed")
                    store._save_decision(
                        conn,
                        decision,
                        status="paused" if decision.action == "pause" else "stopped",
                    )
                    fresh = True
                revision = store._row(conn, decision.workflow_id)["control_revision"]
            if fresh:
                self._paused_tasks = (
                    {
                        task.claim.id: _WorkflowTaskDispatch(
                            self,
                            task,
                            owner,
                            activity,
                            service,
                            decision.id,
                            revision,
                            payload,
                            _KEY,
                        )
                        for task, owner, activity, service in self._committed_tasks.values()
                    }
                    if decision.action == "pause"
                    else {}
                )
        if decision.action == "stop":
            runtime = require_workflow_runtime(self)
            for wave in tuple(runtime._workflow_waves):
                if wave.phase is self:
                    wave.stopping.set()
            for task, _owner, _claim, _service in tuple(self._committed_tasks.values()):
                launcher = task._native_preparation[0].service
                for handle in launcher.live_subagent_handles():
                    if handle.run_id == task.claim.launch_run_id:
                        launcher.cancel_subagent(handle.handle_id)
        return store.get(decision.workflow_id)

    def __repr__(self):
        return "WorkflowPhaseAuthorization(authority=<redacted>)"


@dataclass(eq=False)
class WorkflowRemainingAcceptance:
    """One original review invocation's result and physical custody."""

    controller: object
    decision: object
    attempt: object = None
    owner: object = None
    claim: object = None
    phase: object = None
    host: bool = False
    returned: bool = False
    scopes_returned: bool = False
    host_returned: bool = False
    worker_returned: bool = False
    check_source: object = None
    terminal_callback: object = None
    terminal_scopes_returned: bool = False
    issuing: object = None
    service: object = None
    body_error: object = None
    read_return: object = None
    terminal_read_return: object = None
    terminal_result: object = None
    accepted_attempt: object = None
    preparation_entered: bool = False
    owner_return: list = field(default_factory=list)


_issued_remaining_acceptances: WeakSet[WorkflowRemainingAcceptance] = WeakSet()


def require_remaining_unused(read, service, phase):
    """Admit only the original returned read's required unused terminal write."""
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner

    from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import preparation_source_for

    if (
        type(read) is not WorkflowRemainingAcceptance
        or read not in _issued_remaining_acceptances
        or read.service is not service
        or read.phase is not phase
        or read.controller is not phase._controller
        or phase._ledger._db is not service.db
        or not read.scopes_returned
        or not read.host_returned
        or not read.worker_returned
        or read.returned
        or not read.owner.is_admitted
        or current_execution_owner() is not read.owner
        or getattr(read.owner, "_workflow_remaining_acceptance", None) is not read
        or preparation_source_for(read.controller) is not read
        or phase._task_scopes
        or phase._committed_tasks
        or phase._paused_tasks
    ):
        raise AutomaticWorkRefused("workflow_original_unused_acceptance_required")
    read.check_source()
    runtime = read.controller.app.console_runtime
    runtime.worktree_activity.validate_opaque_claim(read.claim, owner=read.owner)
    if not any(
        record.source is read
        and read.terminal_read_return is not None
        and record is read.terminal_read_return[0]
        and record.task is read.issuing
        for record in runtime._preparation_reads
    ):
        raise AutomaticWorkRefused("workflow_original_unused_acceptance_required")


def accept_workflow_phase(
    controller, attempt_id: str, *, session_id: str, _remaining_read=None, _planning=None
):
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

    with execution(ledger), ledger._db.connection() as conn:
        repeated = conn.execute(
            "SELECT 1 FROM workflow_phase_attempts p JOIN workflow_phase_attempts prior "
            "ON prior.workflow_id=p.workflow_id AND prior.plan_revision=p.plan_revision "
            "AND prior.phase_id=p.phase_id AND prior.ordinal<p.ordinal "
            "WHERE p.native_attempt_id=? AND prior.native_attempt_id IS NOT NULL",
            (attempt_id,),
        ).fetchone()
        if repeated is not None and attempt.state == "prepared":
            from tldw_chatbook.Agents.execution_capacity import current_execution_owner

            from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import preparation_source_for

            read = _remaining_read
            if (
                type(read) is not WorkflowRemainingAcceptance
                or read not in _issued_remaining_acceptances
                or read.controller is not controller
                or read.attempt != attempt
                or preparation_source_for(controller) is not read
                or current_execution_owner() is not read.owner
                or getattr(read.owner, "_workflow_remaining_acceptance", None)
                is not read
                or read.returned
                or read.phase is not None
            ):
                raise AutomaticWorkRefused("workflow_owned_remaining_review_required")
            phase_row = conn.execute(
                "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if (
                ledger._db.workflow_runs._attempt_decision(conn, phase_row)
                != read.decision
            ):
                raise AutomaticWorkRefused("workflow_owned_remaining_review_required")
            controller.app.console_runtime.worktree_activity.validate_opaque_claim(
                read.claim, owner=read.owner
            )
    if _planning is not None:
        from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import preparation_source_for

        from .execution_capacity import current_execution_owner
        from .workflow_task_scope import require_workflow_runtime_for_controller

        if (
            attempt.purpose != "planning"
            or preparation_source_for(controller) is not _planning
        ):
            raise AutomaticWorkRefused("workflow_original_planning_preparation_required")
        runtime = require_workflow_runtime_for_controller(controller)
        finite = _planning._finite
        if (
            _planning.phase is not None
            or _planning.controller is not controller
            or _planning.session is not session
            or _planning.runtime is not runtime
            or _planning.service is not runtime._workflow_service
            or _planning.task_id is not None
            or _planning not in runtime._workflow_preparations
            or finite is None
            or current_execution_owner() is not finite[0]
        ):
            raise AutomaticWorkRefused("workflow_original_planning_preparation_required")
        runtime.worktree_activity.validate_opaque_claim(finite[1], owner=finite[0])
    ledger._install_workflow_phase_validators(
        workflow_fence_connection, workflow_dispatch_identity
    )
    with execution(ledger):
        if not ledger.accept_workflow_phase(attempt_id, owner_id=owner_id):
            return None
    accepted = ledger.read_workflow_attempt(attempt_id, owner_id=owner_id)
    return WorkflowPhaseAuthorization(
        controller, accepted, _key=_PHASE_KEY, _session=session, _planning=_planning
    )


def begin_remaining_acceptance(controller, decision, *, service):
    """Create the result cell for the original live coordinator invocation."""
    import asyncio

    from .workflow_models import WorkflowDecision
    from .workflow_task_scope import require_workflow_runtime_for_controller

    issuing = asyncio.current_task()
    runtime = require_workflow_runtime_for_controller(controller)
    if (
        issuing is None
        or runtime._workflow_service is not service
        or type(decision) is not WorkflowDecision
        or decision.reason not in {"ready", "remaining_ready"}
    ):
        raise AutomaticWorkRefused("workflow_native_source_changed")
    cell = WorkflowRemainingAcceptance(controller, decision, issuing=issuing, service=service)
    _issued_remaining_acceptances.add(cell)
    return cell


def _original_planning_phase(prepared):
    # A captured phase never needs to keep a released Console preparation alive.
    phase = getattr_static(prepared, "phase", None)
    if type(phase) is not WorkflowPhaseAuthorization:
        return None
    producer = phase._planning_preparation
    if (
        type(producer) is ReferenceType
        and producer() is prepared
        and phase.attempt.purpose == "planning"
    ):
        return phase
    return None


def registered_planning_preparation(prepared):
    """Return the original planning producer while its native registration is live."""
    if (
        _original_planning_phase(prepared) is not None
        and prepared.task_id is None
        and prepared in prepared.runtime._workflow_preparations
    ):
        return prepared
    return None


def require_returned_planning(prepared, service, error: BaseException) -> None:
    """Require the accepted phase's original planning return for this service."""
    if (
        _original_planning_phase(prepared) is None
        or prepared.service is not service
        or prepared.runtime is not service._runtime
        or not isinstance(error, BaseException)
    ):
        raise ValueError("workflow_planning_return_required")
    returned = prepared.planning_recovery_return()
    if returned.error is not error:
        raise ValueError("workflow_planning_return_required")
