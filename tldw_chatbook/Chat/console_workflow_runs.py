"""Private accepted workflow custody; execution coordination belongs to ConsoleRuntime."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field, fields
from threading import RLock
from typing import TYPE_CHECKING

from tldw_chatbook import config
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkRefused,
)
from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

if TYPE_CHECKING:
    import asyncio

    from tldw_chatbook.Agents.workflow_models import (
        WorkflowDecision,
        WorkflowExecutionRecovery,
        WorkflowRequest,
        WorkflowRuntimeSnapshot,
        WorkflowSetup,
        WorkflowSnapshot,
    )

_KEY = object()
_CONFIG_IDENTITY = config.current_config_identity
_POLICY_ENV = ("TLDW_AGENTS_WORKFLOW_RUNS_ENABLED",) + tuple(
    "TLDW_AGENTS_MAX_WORKFLOW_" + item.name.upper()
    for item in fields(AutomaticWorkLimits)
)


def _policy_identity():
    return config.current_config_identity(), tuple(
        os.environ.get(name) for name in _POLICY_ENV
    )


def _native_fence_sources(phase, ledger):
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


def _fence_connection(fence, ledger):
    """Use only the phase's active, thread-bound connection and current policy."""
    if not isinstance(fence, _WorkflowFence):
        raise AutomaticWorkRefused("workflow_native_fence_changed")
    _native_fence_sources(fence.phase, ledger)
    if (
        fence.key is not _KEY
        or fence.phase._active_fence is not fence
        or fence.thread is not threading.current_thread()
        or getattr(ledger._db._thread_local, "conn", None) is not fence.connection
        or fence.operation not in {"child_commit", "control"}
        or fence.policy != _policy_identity()
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


def _dispatch_identity(proof, ledger):
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
        self._control_lock = RLock()
        self._active_fence = None
        self._committed_tasks = {}
        self._paused_tasks = {}
        self._retired = False
        self._quiescence = None
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
        before = _policy_identity()
        if operation == "child_commit":
            self._ledger._workflow_enabled()
        limits = AutomaticWorkLimits.from_settings("workflow_phase")
        if before != _policy_identity():
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
                    _fence_connection(fence, self._ledger)
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
            _dispatch_identity(proof, self._ledger)
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
        from tldw_chatbook.Agents.workflow_task_scope import _runtime

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
            runtime = _runtime(self)
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
            runtime = _runtime(self)
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


@dataclass
class _RemainingAcceptance:
    """One original review invocation's result and physical custody."""

    controller: object
    decision: object
    key: object
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


@contextmanager
def _remaining_storage_return(databases):
    """Observe these original worker handles before and after every native scope."""
    originals = []
    for database in dict.fromkeys(db for db in databases if db is not None):
        namespace = vars(database)
        if namespace.get("is_memory_db") is True:
            continue
        slot = "_local" if namespace.get("_local") is not None else "_thread_local"
        local = namespace.get(slot)
        if local is None:
            raise RuntimeError("original source database local unavailable")
        originals.append((database, slot, local, getattr(local, "conn", None)))
    try:
        yield
    finally:
        if not all(
            vars(database).get(slot) is local
            and getattr(local, "conn", None) is original
            for database, slot, local, original in originals
        ):
            raise RuntimeError("original source database handle did not return")


def _check_remaining_unused(read, service, phase):
    """Admit only the original returned read's required unused terminal write."""
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner

    from .console_preparation_reads import preparation_source_for

    if (
        type(read) is not _RemainingAcceptance
        or read.key is not _KEY
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


def _accept_workflow_phase(
    controller, attempt_id: str, *, session_id: str, _remaining_read=None
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

            from .console_preparation_reads import preparation_source_for

            read = _remaining_read
            if (
                type(read) is not _RemainingAcceptance
                or read.key is not _KEY
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
    with execution(ledger):
        if not ledger.accept_workflow_phase(attempt_id, owner_id=owner_id):
            return None
    accepted = ledger.read_workflow_attempt(attempt_id, owner_id=owner_id)
    return WorkflowPhaseAuthorization(controller, accepted, _key=_KEY, _session=session)


def _capture_workflow_read(controller, source, issuing):
    import asyncio

    from .console_preparation_reads import preparation_read_for

    read = preparation_read_for(controller)
    if (
        read is None
        or read.source is not source
        or read.task is not issuing
        or not isinstance(read._producer, asyncio.Future)
        or not isinstance(read.retired, asyncio.Future)
        or read.retired.done()
    ):
        raise PermissionError("active owned workflow read required")
    return read, read._producer, read.retired


def _prove_workflow_read(proof, controller, source, issuing):
    import asyncio

    if proof is None:
        raise RuntimeError("original workflow read return missing")
    read, producer, retired = proof
    if (
        read.creator is not controller
        or read.source is not source
        or read.task is not issuing
        or read._producer is not producer
        or read.retired is not retired
        or not asyncio.Future.done(producer)
        or asyncio.Future.cancelled(producer)
        or not read.callback_returned
        or not asyncio.Future.done(retired)
        or asyncio.Future.cancelled(retired)
        or asyncio.Future.result(retired) is not None
    ):
        raise RuntimeError("original workflow reader return unproven")


async def _accept_remaining_workflow_phase(
    controller,
    decision,
    *,
    command_id: str,
    session_id: str,
    binding,
    _initial=False,
    _pending_run=None,
):
    """Accept one fresh remaining decision under an original finite physical read.

    The worker retains its read claim and source/storage scopes through the host's
    acceptance of the original result cell. No returned ID can recreate custody.
    """
    import asyncio

    from tldw_chatbook.Agents.activation import _guard, execution
    from tldw_chatbook.Agents.execution_capacity import CapacityRefused, WorkOrigin
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowDecision
    from tldw_chatbook.Agents.workflow_task_scope import _check_hooks
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    from .console_preparation_reads import run_preparation_read

    if type(decision) is not WorkflowDecision or decision.reason != (
        "ready" if _initial else "remaining_ready"
    ):
        raise ValueError("remaining_decision_required")
    if not native_workflow_qualified():
        raise AutomaticWorkRefused("workflow_platform_unqualified")
    runtime = getattr(controller.app, "console_runtime", None)
    bridge, store = controller._agent_bridge, controller.store
    service = runtime.ensure_workflow_service()
    ledger, database = bridge.runs_db.automatic_work, bridge.runs_db
    capacity, activity = runtime.execution_capacity, runtime.worktree_activity
    session = next((s for s in store.sessions() if s.id == session_id), None)
    if session is None:
        raise AutomaticWorkRefused("workflow_session_changed")
    owner_id = controller._fleet_wake.runtime_owner_id
    persistence = service.persistence
    registry = getattr(persistence, "workspace_registry", None)
    selection = (
        session.workspace_id,
        session.persisted_conversation_id,
        session.project_instruction_state,
    )
    policy, sources = _policy_identity(), _guard.captured_sources(bridge)
    loop, issuing = asyncio.get_running_loop(), asyncio.current_task()
    # This one invocation's original cap remains here on cancellation as well.
    cell = _RemainingAcceptance(
        controller, decision, _KEY, issuing=issuing, service=service
    )
    if _pending_run is not None:
        coordinator = getattr(runtime, "_workflow_coordinator", None)
        if (
            type(_pending_run) is not _WorkflowRun
            or type(coordinator) is not ConsoleWorkflowCoordinator
            or coordinator.controller is not controller
            or coordinator.service is not service
            or coordinator._runs.get(decision.workflow_id) is not _pending_run
            or _pending_run.workflow_id != decision.workflow_id
            or _pending_run.session_id != session_id
            or _pending_run.binding is not binding
            or _pending_run.issuing is not issuing
            or _pending_run.command_id != command_id
            or _pending_run.decision != decision
            or _pending_run.phase is not None
            or _pending_run.acceptance is not None
        ):
            raise PermissionError("original workflow pending command required")
        _pending_run.acceptance = cell
    databases = (
        database,
        getattr(persistence, "db", None),
        getattr(registry, "db", None),
    )

    def originals():
        if (
            controller.app.console_runtime is not runtime
            or runtime.chat_controller is not controller
            or runtime.agent_bridge is not bridge
            or controller._agent_bridge is not bridge
            or controller.store is not store
            or bridge._store is not store
            or bridge.runs_db is not database
            or runtime._workflow_service is not service
            or service.db is not database
            or service.persistence is not persistence
            or getattr(persistence, "workspace_registry", None) is not registry
            or runtime.execution_capacity is not capacity
            or runtime.worktree_activity is not activity
            or controller._fleet_wake.runtime_owner_id != owner_id
            or not any(s is session for s in store.sessions())
            or selection
            != (
                session.workspace_id,
                session.persisted_conversation_id,
                session.project_instruction_state,
            )
            or session.workspace_id != binding.workspace_id
            or policy != _policy_identity()
        ):
            raise AutomaticWorkRefused("workflow_native_source_changed")

    cell.check_source = originals

    def current():
        originals()
        if (
            controller._disposed
            or controller._shutdown_requested.is_set()
            or runtime._disposed
            or runtime._workflow_closing
            or session_id in runtime._admission_fenced_sessions
        ):
            raise AutomaticWorkRefused("workflow_native_source_changed")

    async def handoff():
        # The worker waits with its original scopes through actual host discard.
        try:
            current()
            if issuing.cancelling():
                return
            cell.host = True
        finally:
            cell.host_returned = True

    def read():
        cell.read_return = _capture_workflow_read(controller, cell, issuing)
        owner = operation = claim = None
        try:
            with (
                _remaining_storage_return(databases),
                execution(bridge, sources=sources),
                owned_goal_storage(service),
            ):
                current()
                if _guard.captured_sources(bridge) != sources:
                    raise AutomaticWorkRefused("workflow_native_source_changed")
                with activity.prelaunch(
                    footprint=WorktreeFootprint(opaque=True)
                ) as claim:
                    if claim is None:
                        raise CapacityRefused("workflow_review_activity_busy")
                    owner = capacity.begin_execution(
                        origin=WorkOrigin.AUTOMATIC,
                        conversation_id=session.persisted_conversation_id or session.id,
                    )
                    # Retain the exact result cell on its existing physical owner
                    # if any later source/finish return becomes uncertain.
                    owner._workflow_remaining_acceptance = cell
                    cell.owner, cell.claim = owner, claim
                    try:
                        operation = owner.reserve_tool()
                        try:
                            claim.bind_owner(owner)
                        except BaseException:
                            operation.finish()
                            raise
                    except BaseException:
                        owner.finish_root()
                        raise
                error = None
                with owner.activate():
                    try:
                        current()
                        activity.validate_opaque_claim(claim, owner=owner)
                        _check_hooks(runtime, session_id)
                        if not binding.guard(False):
                            raise AutomaticWorkRefused("workflow_task_binding_changed")
                        snapshot = service.get(decision.workflow_id)
                        if (
                            snapshot.request.binding.workspace_id
                            != binding.workspace_id
                            or snapshot.request.binding.binding_id != binding.binding_id
                            or snapshot.request.binding.locator != str(binding.root)
                            or snapshot.request.conversation_id
                            != (session.persisted_conversation_id or session.id)
                        ):
                            raise ValueError("workflow_remaining_selection_changed")
                        with database.connection() as conn:
                            prior = conn.execute(
                                "SELECT * FROM workflow_phase_attempts WHERE command_id=?",
                                (command_id,),
                            ).fetchone()
                        if prior is None:
                            if _initial:
                                service.validate_decision(decision)
                            else:
                                service.validate_remaining_decision(decision)
                        from .console_workflow_preparation import (
                            _resolve_workflow_targets,
                        )

                        resolutions = asyncio.run_coroutine_threadsafe(
                            _resolve_workflow_targets(
                                controller, session, snapshot.request.targets
                            ),
                            loop,
                        ).result()
                        current()
                        cell.preparation_entered = True
                        attempt = ledger._prepare_workflow(
                            decision.workflow_id,
                            revision_id=decision.plan_revision,
                            phase_id=decision.phase_id,
                            purpose="execution",
                            owner_id=owner_id,
                            session_id=session_id,
                            command_id=command_id,
                            _remaining=None if _initial else decision,
                            _initial=decision if _initial else None,
                        )
                        if prior is not None and attempt.state == "prepared":
                            raise AutomaticWorkRefused(
                                "workflow_remaining_prepared_retry_requires_review"
                            )
                        current()
                        cell.attempt = attempt
                        cell.phase = _accept_workflow_phase(
                            controller,
                            attempt.id,
                            session_id=session_id,
                            _remaining_read=cell,
                        )
                        if cell.phase is not None:
                            cell.accepted_attempt = cell.phase.attempt
                            cell.phase._execution_resolutions = resolutions
                        asyncio.run_coroutine_threadsafe(handoff(), loop).result()
                        current()
                        if not binding.guard(False):
                            raise AutomaticWorkRefused("workflow_task_binding_changed")
                    except BaseException as caught:  # noqa: BLE001 - finish the original finite read before surfacing its outcome.
                        error = caught
                    operation.finish()
            cell.scopes_returned = True
            cell.body_error = error
        except BaseException:
            if owner is not None and not cell.scopes_returned:
                owner.mark_cleanup_unproven()
            raise

    with runtime._first_send_source_scope():
        error = None
        try:
            await run_preparation_read(
                read,
                creator=controller,
                session_id=session_id,
                reads=controller._preparation_reads,
                observers=(runtime._preparation_reads,),
                require_current=originals,
                source=cell,
            )
            current()
        except BaseException as caught:  # noqa: BLE001 - preserve cancellation through original physical retirement.
            error = caught
        # Cancellation is not the worker's result: inspect the exact original
        # producer and retirement futures before releasing any physical custody.
        try:
            _prove_workflow_read(cell.read_return, controller, cell, issuing)
            originals()
        except BaseException:
            if cell.owner is not None:
                cell.owner.mark_cleanup_unproven()
            raise
        cell.worker_returned = True
        if error is None:
            error = cell.body_error
        if not cell.scopes_returned:
            raise AutomaticWorkRefused("workflow_original_storage_return_required")
        if (error is not None or not cell.host) and cell.phase is not None:
            from tldw_chatbook.Backup_Recovery.participants import _core_operation

            def settle_unused():
                cell.terminal_read_return = _capture_workflow_read(
                    controller, cell, issuing
                )
                with (
                    _remaining_storage_return(databases),
                    execution(bridge, sources=sources),
                    cell.owner.activate(),
                    owned_goal_storage(service),
                    _core_operation(database),
                ):
                    originals()
                    if _guard.captured_sources(bridge) != sources:
                        raise AutomaticWorkRefused("workflow_native_source_changed")
                    if not binding.guard(False):
                        raise AutomaticWorkRefused("workflow_task_binding_changed")
                    cell.terminal_result = service.settle_execution_phase(
                        cell.phase, outcome="cancelled", _remaining_read=cell
                    )
                # Produced only by this original callback, after ALL native
                # source/storage/handle scopes have actually returned.
                cell.terminal_scopes_returned = True

            cell.terminal_callback = settle_unused
            try:
                # SAME owner/cell and existing physical read observations; no
                # cancellation can release the original source ahead of SQL.
                await run_preparation_read(
                    settle_unused,
                    creator=controller,
                    session_id=session_id,
                    reads=controller._preparation_reads,
                    observers=(runtime._preparation_reads,),
                    require_current=originals,
                    source=cell,
                )
            except asyncio.CancelledError as cancelled:
                # The tracked reader consumes the physical worker result before
                # re-delivery. Only its original positive scope return settles.
                if not cell.terminal_scopes_returned:
                    cell.owner.mark_cleanup_unproven()
                    raise
                error = cancelled
            except BaseException:
                cell.owner.mark_cleanup_unproven()
                raise
            try:
                _prove_workflow_read(
                    cell.terminal_read_return, controller, cell, issuing
                )
                originals()
            except BaseException:
                cell.owner.mark_cleanup_unproven()
                raise
            if not cell.terminal_scopes_returned:
                cell.owner.mark_cleanup_unproven()
                raise AutomaticWorkRefused("workflow_original_storage_return_required")
        try:
            originals()
            activity.validate_opaque_claim(cell.claim, owner=cell.owner)
            cell.owner.on_drained(cell.owner_return.append)
            cell.owner.finish_root()
            if cell.owner_return != [True]:
                raise RuntimeError("original workflow acceptance owner did not drain")
            cell.claim.settle(durable=True)
            if cell.claim.id in activity._claims:
                raise RuntimeError("original workflow acceptance claim did not return")
            cell.returned = True
            if cell.terminal_scopes_returned:
                cell.phase = None
            del cell.owner._workflow_remaining_acceptance
        except BaseException:
            cell.owner.mark_cleanup_unproven()
            raise
        if error is not None:
            raise error
        return cell.phase


def _planning_driver_source(coordinator, run, *, entered=False):
    import asyncio

    _metadata_source(coordinator)
    if (
        coordinator._runs.get(run.workflow_id) is not run
        or run.prepared is None
        or run.prepared.phase is not run.phase
        or run.prepared.controller is not coordinator.controller
        or run.prepared.service is not coordinator.service
        or entered
        and asyncio.current_task() is not run.driver
    ):
        raise PermissionError("workflow planning owner changed")
    return coordinator._plan


def _metadata_source(coordinator):
    """Validate the runtime resources behind passive coordinator reads."""
    if (
        not isinstance(coordinator, ConsoleWorkflowCoordinator)
        or coordinator.runtime._workflow_coordinator is not coordinator
        or coordinator.runtime._workflow_service is not coordinator.service
        or coordinator.runtime.chat_controller is not coordinator.controller
        or coordinator.controller._agent_bridge is not coordinator.runtime.agent_bridge
    ):
        raise PermissionError("workflow coordinator resources changed")


def _metadata_publication_return(coordinator, run, cell):
    """Consume only the same original final read's positive host disposition."""
    import asyncio

    _metadata_source(coordinator)
    if (
        type(cell) is not _METADATA_CELL_CLASS
        or cell is not run.final_metadata
        or cell.issuing is not run.driver
        or cell.issuing is not asyncio.current_task()
        or cell.run is not run
        or cell.phase is not run.phase
        or type(cell.disposed) is not asyncio.Future
        or not asyncio.Future.done(cell.disposed)
        or asyncio.Future.cancelled(cell.disposed)
        or asyncio.Future.result(cell.disposed) is not True
        or cell.body_error is not None
        or cell.source_error is not None
    ):
        raise RuntimeError("original workflow publication return unproven")
    if cell.entered:
        _prove_workflow_read(
            cell.proof, coordinator.controller, coordinator, run.driver
        )
    elif cell.proof is not None or not isinstance(
        cell.caller_error, asyncio.CancelledError
    ):
        raise RuntimeError("original workflow publication discard unproven")


def _execution_release_source(coordinator, binding, *, released=False):
    """Validate the actual execution driver and its retained admission slots."""
    import asyncio

    _metadata_source(coordinator)
    (
        original,
        run,
        phase,
        issuing,
        loop,
        runtime,
        controller,
        store,
        wake,
        intake,
        token,
        registry_lock,
        wake_loop,
    ) = binding
    if (
        original is not coordinator
        or coordinator.runtime is not runtime
        or coordinator.controller is not controller
        or coordinator.loop is not loop
        or asyncio.get_running_loop() is not loop
        or asyncio.current_task() is not issuing
        or run.driver is not issuing
        or run.phase is not phase
        or phase.attempt.purpose != "execution"
        or controller.store is not store
        or controller._fleet_wake is not wake
        or wake._controller is not controller
        or wake._registry_lock is not registry_lock
        or wake._loop is not wake_loop
        or run.intake is not intake
        or run.token is not token
        or (
            run.workflow_id in coordinator._runs
            if released
            else coordinator._runs.get(run.workflow_id) is not run
        )
    ):
        raise PermissionError("workflow execution release owner changed")


@dataclass(eq=False)
class _WorkflowRun:
    """Original native producers retained until their actual physical returns."""

    workflow_id: str
    session_id: str
    binding: object
    phase: object
    prepared: object = None
    driver: object = None
    token: object = None
    intake: object = None
    control: str | None = None
    status: str | None = None
    control_return: object = None
    control_read: object = None
    control_unknown: bool = False
    wake: object = None
    preparations: dict = field(default_factory=dict)
    wait_footprints: dict = field(default_factory=dict)
    acceptance: object = None
    issuing: object = None
    command_id: str | None = None
    decision: object = None
    final_metadata: object = None
    planning_input: object = None
    planning_snapshot: object = None
    planning_reads: object = None
    planning_entered: bool = False
    planning_constructing: bool = False
    planning_returned: bool = False
    planning_error: object = None
    preparation_returns: list = field(default_factory=list)
    waves: list = field(default_factory=list)
    accounting_error: object = None


@dataclass(eq=False)
class _WorkflowMetadataRead:
    """One passive read's original physical return and finite host disposition."""

    workflow_id: str
    issuing: object
    run: object
    phase: object
    disposed: object
    proof: object = None
    entered: bool = False
    body_error: object = None
    caller_error: object = None
    producer_error: object = None
    source_error: object = None


@dataclass(eq=False)
class _WorkflowQuiescence:
    """The same accepted phase's current accounting return, never authority."""

    coordinator: object
    run: object
    phase: object
    issuing: object
    preparations: tuple
    waves: tuple
    mode: str = "quiesce"
    proof: object = None
    callback: object = None
    interval: object = None
    idle_proof: object = None
    idle_callback: object = None
    error: object = None
    producer_error: object = None


def _quiescence_source(cell):
    """Check only the original retained receiving objects and definitions."""
    from . import console_workflow_preparation as preparation

    if type(cell) is not _QUIESCENCE_CLASS:
        raise PermissionError("original workflow accounting record required")
    coordinator, run, phase = cell.coordinator, cell.run, cell.phase
    if (
        type(cell) is not _QUIESCENCE_CLASS
        or type(coordinator) is not _COORDINATOR_CLASS
        or coordinator._runs.get(run.workflow_id) is not run
        or run.phase is not phase
        or phase._quiescence is not cell
        or coordinator.controller.app.console_runtime is not coordinator.runtime
        or coordinator.runtime._workflow_coordinator is not coordinator
        or coordinator.runtime._workflow_service is not coordinator.service
        or coordinator.runtime.chat_controller is not coordinator.controller
        or phase._controller is not coordinator.controller
        or phase._ledger is not coordinator.service.db.automatic_work
        or run.driver is not cell.issuing
        or run.control_unknown
        or run.status == "recovery_required"
        or len(run.preparation_returns) != len(cell.preparations)
        or any(a is not b for a, b in zip(run.preparation_returns, cell.preparations))
        or len(run.waves) != len(cell.waves)
        or any(a is not b for a, b in zip(run.waves, cell.waves))
    ):
        raise AutomaticWorkRefused("workflow_quiescence_source_changed")
    _native_fence_sources(phase, phase._ledger)
    _quiescence_wave._workflow_wave_sources()
    for saved in cell.preparations:
        item, auth, native, frame, configuration, source, lifetime, scope, reads = saved
        if (
            type(item) is not preparation._PreparedWorkflowReply
            or item.phase is not phase
            or item.authorization is not auth
            or phase._task_scopes.get(item.task_id) is not auth
            or auth._bridge_preparation[0] is not item
            or auth._native_preparation[0] is not native
            or item.native is not frame
            or item.configuration is not configuration
            or item.source is not source
            or item._lifetime is not lifetime
            or item._finite_reads is not reads
            or not item._closed
            and item._source_scope is not scope
        ):
            raise AutomaticWorkRefused("workflow_quiescence_preparation_changed")
        item._execution_return_source()


def _quiescence_physical(cell):
    """Consume original acceptance, finite reads, native receipts and B/E returns."""
    _quiescence_source(cell)
    run, coordinator = cell.run, cell.coordinator
    acceptance = run.acceptance
    if (
        type(acceptance) is not _RemainingAcceptance
        or acceptance.phase is not cell.phase
        or acceptance.issuing is not run.issuing
        or not acceptance.returned
        or not acceptance.scopes_returned
        or not acceptance.worker_returned
        or acceptance.owner_return != [True]
        or acceptance.owner.is_admitted
        or acceptance.claim.id in coordinator.runtime.worktree_activity._claims
    ):
        raise RuntimeError("workflow acceptance return unproven")
    _prove_workflow_read(
        acceptance.read_return, coordinator.controller, acceptance, run.issuing
    )
    acceptance.check_source()
    for saved in cell.preparations:
        item, *_rest, records = saved
        if item._finite is not None or not records:
            raise RuntimeError("workflow preparation still active")
        for record in records:
            item._finite_read_return(record)
            if (
                record.prepared is not item
                or not record.positive
                or not record.scopes_returned
                or record.owner is not None
                and (record.drains != [True] or record.owner.is_admitted)
                or record.claim is not None
                and record.claim.id in coordinator.runtime.worktree_activity._claims
            ):
                raise RuntimeError("workflow finite return unproven")
    for saved in cell.waves:
        wave = saved[0]
        if wave.phase is not cell.phase:
            raise RuntimeError("workflow wave phase changed")
        _quiescence_wave._workflow_wave_return(wave, saved)


def _quiescence_receiving(cell, ledger, mode):
    """Accept the exact original accounting callback in its actual native read."""
    _quiescence_source(cell)
    if (
        cell.phase._ledger is not ledger
        or cell.mode != mode
        or _capture_workflow_read(cell.coordinator.controller, cell, cell.issuing)
        != cell.proof
    ):
        raise PermissionError("original workflow accounting callback required")
    _quiescence_physical(cell)


def _quiescence_evidence(
    ledger, conn, row, *, context=None, dispatch=None, native=None
):
    """Match one unknown contribution using only already-held native context."""
    from tldw_chatbook.Agents import automatic_work_runtime as automatic
    from tldw_chatbook.Agents import workflow_task_scope as tasks

    phase = None
    if context is not None:
        if type(context) is not _WorkflowMetadataRead or context.run is None:
            return False
        phase = context.phase
        cell = getattr(phase, "_quiescence", None)
        if cell is None:
            return False
        coordinator = cell.coordinator
        _metadata_source(coordinator)
        if (
            context.run is not cell.run
            or coordinator._metadata_cells.get(context) is not context.disposed
            or context.disposed.done()
            or context.proof
            != _capture_workflow_read(
                coordinator.controller, coordinator, context.issuing
            )
        ):
            return False
    elif dispatch is not None:
        phase = _dispatch_identity(dispatch, ledger).phase
    elif native is not None:
        _fence_connection(native, ledger)
        phase = native.phase
    else:
        current = automatic.current_automatic_work()
        if type(current) is not AutomaticWorkContext or current.ledger is not ledger:
            return False
        phase = current.workflow
        if (
            phase is None
            or phase.context is not current
            or current.attempt_id != phase.attempt.id
            or current.owner_id != phase.attempt.owner_id
            or current.chain_id != phase.attempt.chain_id
            or not current._accepted.is_set()
        ):
            return False
        task = tasks.current_task_authorization()
        if task is not None and (
            type(task) is not tasks.WorkflowTaskAuthorization
            or task.phase is not phase
            or phase._task_scopes.get(task.task.id) is not task
            or task.claim.owner_id != current.owner_id
            or task.claim.chain_id != current.chain_id
        ):
            return False
    cell = getattr(phase, "_quiescence", None)
    if (
        cell is None
        or cell.phase is not phase
        or phase._ledger is not ledger
        or row["native_attempt_id"] != phase.attempt.id
        or row["workflow_id"] != phase.attempt.workflow_id
        or row["native_state"] != "accepted"
        or row["native_owner_id"] != phase.attempt.owner_id
        or cell.error is not None
        or cell.interval is None
        or cell.mode not in {"idle", "resume", "cleanup"}
    ):
        return False
    try:
        ledger._check_runtime_owner(conn, phase.attempt.owner_id)
        _quiescence_physical(cell)
        _prove_workflow_read(
            cell.idle_proof, cell.coordinator.controller, cell, cell.issuing
        )
        if cell.idle_proof[1].result() != cell.interval:
            return False
        if (
            cell.mode != "idle"
            and _capture_workflow_read(cell.coordinator.controller, cell, cell.issuing)
            != cell.proof
        ):
            return False
        actual = conn.execute(
            "SELECT * FROM workflow_active_intervals WHERE id=? AND workflow_id=? AND ended_at IS NOT NULL",
            (cell.interval[0], row["workflow_id"]),
        ).fetchone()
        return actual is not None and tuple(actual) == cell.interval
    except (PermissionError, RuntimeError, AutomaticWorkRefused):
        return False


class ConsoleWorkflowCoordinator:
    """Public commands composed from the existing native workflow producers."""

    def __init__(self, controller, service):
        import asyncio

        self.controller, self.service = controller, service
        self.runtime = controller.app.console_runtime
        self.loop = asyncio.get_running_loop()
        self._runs = {}
        self._commands = {}
        self._subscribers = set()
        self._closing = False
        self._metadata_cells = {}
        self._metadata_tail = None
        self._metadata_floor = {}
        self._pending_planning = set()

    def subscribe(
        self, callback: Callable[[WorkflowRuntimeSnapshot], None]
    ) -> Callable[[], None]:
        """Observe bounded runtime metadata until this exact subscription retires."""
        if self._closing:
            raise RuntimeError("workflow coordinator closed")
        token = (object(), callback)
        self._subscribers.add(token)

        def unsubscribe():
            self._subscribers.discard(token)

        return unsubscribe

    def _metadata_lower_bound(self, snapshot):
        current = self._metadata_floor.get(snapshot.id, (0, 0))
        self._metadata_floor[snapshot.id] = (
            max(current[0], snapshot.control_revision),
            max(current[1], snapshot.progress_cursor),
        )

    @asynccontextmanager
    async def _metadata(self, workflow_id, *, history, capture, terminal_run=None):
        """Sequence the same native read and its host return without another task."""
        import asyncio

        from tldw_chatbook.Agents.activation import _guard, worker_guard
        from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
        from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

        from . import console_preparation_reads as native

        _metadata_source(self)
        issuing = asyncio.current_task()
        run = self._runs.get(workflow_id)
        owned = (run is not None and issuing is run.driver) or any(
            task is issuing for _identity, task in self._commands.values()
        )
        if self._closing and not owned:
            raise RuntimeError("workflow coordinator closed")
        if terminal_run is not None and (
            run is not terminal_run or issuing is not run.driver
        ):
            raise PermissionError("original workflow publication driver required")
        if any(
            future.done() and future.result() is False
            for future in self._metadata_cells.values()
        ):
            raise RuntimeError("workflow original metadata return unproven")
        controller, runtime, service = self.controller, self.runtime, self.service
        bridge, store = controller._agent_bridge, controller.store
        database, persistence = service.db, service.persistence
        registry = persistence.workspace_registry
        sources = _guard.captured_sources(bridge)
        future = self.loop.create_future()
        cell = _WorkflowMetadataRead(
            workflow_id, issuing, run, run.phase if run is not None else None, future
        )
        capture.append(cell)
        if terminal_run is not None:
            terminal_run.final_metadata = cell
        prior = self._metadata_tail
        self._metadata_cells[cell] = future
        self._metadata_tail = future
        proven = False
        outcome = None
        result = None

        def original_sources():
            _metadata_source(self)
            if (
                type(self) is not _COORDINATOR_CLASS
                or controller.app.console_runtime is not runtime
                or runtime._workflow_coordinator is not self
                or runtime.chat_controller is not controller
                or runtime.agent_bridge is not bridge
                or controller._agent_bridge is not bridge
                or controller.store is not store
                or bridge._store is not store
                or runtime._workflow_service is not service
                or type(service) is not WorkflowRunService
                or service.db is not database
                or database is not bridge.runs_db
                or service.persistence is not persistence
                or persistence.workspace_registry is not registry
                or _guard.captured_sources(bridge) != sources
                or self._metadata_cells.get(cell) is not future
                or cell.disposed is not future
                or cell.issuing is not issuing
                or future.done()
            ):
                raise PermissionError("workflow original metadata source changed")

        def associated():
            return self._runs.get(workflow_id) is run and (
                run is None or run.phase is cell.phase
            )

        def read_metadata():
            nonlocal result
            with owned_goal_storage(service):
                original_sources()
                with database.connection() as conn:
                    # This async producer owns a fresh worker read transaction.
                    # Never borrow an unexplained pre-existing worker transaction.
                    if conn.in_transaction:
                        raise RuntimeError(
                            "workflow metadata transaction already active"
                        )
                    conn.execute("BEGIN")
                    try:
                        try:
                            full = (
                                service.get(workflow_id, _accounting=cell)
                                if history
                                else None
                            )
                            view = service.runtime_snapshot(
                                workflow_id, _accounting=cell
                            )
                            result = full, view
                        except Exception as error:  # noqa: BLE001 - carry an ordinary body refusal through positive native scopes.
                            cell.body_error = error
                    finally:
                        conn.rollback()
                original_sources()

        def invoke():
            cell.proof = _capture_workflow_read(controller, self, issuing)
            with _remaining_storage_return((database, persistence.db, registry.db)):
                guarded_read()

        guarded_read = None
        try:
            if prior is not None:
                while not prior.done():
                    try:
                        await asyncio.shield(prior)
                    except asyncio.CancelledError as error:
                        cell.caller_error = error
                if prior.cancelled() or prior.result() is not True:
                    proven = True  # This invocation never entered a read.
                    raise RuntimeError("workflow preceding metadata return unproven")
            original_sources()
            if cell.caller_error is not None:
                proven = True
                raise cell.caller_error
            if not associated():
                proven = True
                raise ValueError("workflow_snapshot_superseded")
            guarded_read = worker_guard(bridge)(read_metadata)
            cell.entered = True
            try:
                await native.run_preparation_read(
                    invoke,
                    creator=controller,
                    session_id=run.session_id if run is not None else None,
                    reads=controller._preparation_reads,
                    observers=(runtime._preparation_reads,),
                    require_current=original_sources,
                    source=self,
                )
            except BaseException as error:  # noqa: BLE001 - inspect the same producer independently of caller cancellation.
                outcome = error
                if isinstance(error, asyncio.CancelledError):
                    cell.caller_error = error
            if cell.proof is not None:
                producer = cell.proof[1]
                if asyncio.Future.done(producer) and not asyncio.Future.cancelled(
                    producer
                ):
                    cell.producer_error = asyncio.Future.exception(producer)
            try:
                if cell.proof is None:
                    raise RuntimeError("original metadata read missing")
                _prove_workflow_read(cell.proof, controller, self, issuing)
                original_sources()
                if outcome is not None and not isinstance(
                    outcome, asyncio.CancelledError
                ):
                    raise outcome
            except BaseException as error:
                cell.source_error = error
                if cell.caller_error is not None:
                    raise cell.caller_error
                raise
            proven = True
            if cell.caller_error is not None:
                raise cell.caller_error
            if cell.body_error is not None:
                raise cell.body_error
            if not associated():
                raise ValueError("workflow_snapshot_superseded")
            full, view = result
            floor = self._metadata_floor.get(workflow_id, (0, 0))
            if view.control_revision < floor[0] or view.progress_cursor < floor[1]:
                raise ValueError("workflow_snapshot_superseded")
            self._metadata_lower_bound(view)
            if run is not None and run.status is not None:
                view = view.model_copy(update={"status": run.status})
            yield full, view
        finally:
            if proven:
                try:
                    original_sources()
                except BaseException as error:  # noqa: BLE001 - keep the original failed source and caller outcome.
                    proven = False
                    cell.source_error = error
            if not proven and run is not None:
                run.status = "recovery_required"
            if not future.done():
                future.set_result(proven)
            if proven:
                self._metadata_cells.pop(cell)
                if self._metadata_tail is future:
                    self._metadata_tail = None
            if (
                cell.source_error is not None
                and outcome is None
                and cell.caller_error is None
            ):
                raise cell.source_error

    async def _publish(self, workflow_id, *, _terminal_run=None):
        import asyncio

        from loguru import logger

        _metadata_source(self)
        captured = []
        try:
            async with self._metadata(
                workflow_id, history=False, capture=captured, terminal_run=_terminal_run
            ) as (_full, snapshot):
                for token in tuple(self._subscribers):
                    if self._closing or token not in self._subscribers:
                        continue
                    try:
                        token[1](snapshot)
                    except Exception:  # noqa: BLE001 - observers cannot change native execution.
                        logger.exception("Workflow snapshot observer failed")
        except ValueError as error:
            if str(error) != "workflow_snapshot_superseded":
                raise
        except asyncio.CancelledError:
            cell = captured[0] if captured else None
            if (
                _terminal_run is None
                or cell is None
                or not cell.disposed.done()
                or cell.disposed.result() is not True
                or cell.body_error is not None
            ):
                raise
            # The final driver consumes this exact positive discard, releases
            # synchronously, then redelivers this original caller cancellation.
        return captured[0] if captured else None

    async def get(self, workflow_id: str) -> WorkflowSnapshot:
        """Return one coherent private history/view or refuse a superseded read."""
        _metadata_source(self)
        async with self._metadata(workflow_id, history=True, capture=[]) as (
            full,
            view,
        ):
            return full.model_copy(update={"status": view.status})

    async def _command(self, key, identity, operation):
        import asyncio

        if self._closing:
            raise RuntimeError("workflow coordinator closed")
        prior = self._commands.get(key)
        if prior is None:
            task = asyncio.create_task(operation(), name="native-workflow-command")
            self._commands[key] = (identity, task)
        else:
            if prior[0] != identity:
                raise ValueError("workflow_command_conflict")
            task = prior[1]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
                except BaseException:  # noqa: BLE001 - consume exact task failure before caller cancellation.
                    # Preserve caller cancellation after consuming this exact
                    # command's outcome; its retained cell records uncertainty.
                    break
            if not task.cancelled():
                task.exception()
            raise

    def _selection(self, request):
        from .console_chat_controller import capture_run_admitted_workspace_roots

        session = next(
            (
                s
                for s in self.controller.store.sessions()
                if (s.persisted_conversation_id or s.id) == request.conversation_id
            ),
            None,
        )
        if session is None:
            raise PermissionError("workflow owning Chat unavailable")
        roots = capture_run_admitted_workspace_roots(
            session=session, registry=self.service.persistence.workspace_registry
        )
        binding = next(
            (
                b
                for b in roots
                if b.workspace_id == request.binding.workspace_id
                and b.binding_id == request.binding.binding_id
                and str(b.root) == request.binding.locator
                and ("rw" if b.allow_write else "ro") == request.binding.access
            ),
            None,
        )
        if binding is None:
            raise PermissionError("workflow binding unavailable")
        return session, binding

    def _pending_planning_source(self, run, *, issuing=False):
        """Check the original pending command without granting it admission."""
        import asyncio

        if (
            type(self) is not _COORDINATOR_CLASS
            or type(run) is not _WorkflowRun
            or run not in self._pending_planning
            or self.runtime._workflow_coordinator is not self
            or self.controller.app.console_runtime is not self.runtime
            or self.runtime.chat_controller is not self.controller
            or self.runtime._workflow_service is not self.service
            or run.issuing is None
            or not any(
                task is run.issuing for _identity, task in self._commands.values()
            )
            or issuing
            and asyncio.current_task() is not run.issuing
            or run.workflow_id is not None
            and self._runs.get(run.workflow_id) is not run
        ):
            raise PermissionError("original pending planning command changed")
        if run.prepared is not None and (
            run.prepared._planning_pending_run is not run
            or run.prepared._planning_reads is not run.planning_reads
            or run.prepared.controller is not self.controller
            or run.prepared.session.id != run.session_id
            or run.prepared.binding is not run.binding
        ):
            raise PermissionError("original pending planning preparation changed")

    async def _planning_identity(self, run, prepared, snapshot):
        """Receive only the original persistence callback's passive identity."""
        self._pending_planning_source(run)
        if (
            run.prepared is not prepared
            or run.planning_snapshot is not None
            or run.workflow_id is not None
            and run.workflow_id != snapshot.id
            or snapshot.id in self._runs
            and self._runs[snapshot.id] is not run
        ):
            raise PermissionError("original planning identity handoff changed")
        run.workflow_id = snapshot.id
        run.planning_snapshot = snapshot
        self._runs[snapshot.id] = run

    def _planning_unused_return(self, run):
        self._pending_planning_source(run)
        prepared = run.prepared
        if prepared is None:
            return run.planning_entered and not run.planning_constructing
        prepared._planning_reads_returned()
        return (
            run.planning_returned
            and prepared._planning_preparation_returned.is_set()
            and prepared._closed
            and prepared._finite is None
            and prepared.primary_owner is None
            and prepared._planning_intake is None
            and prepared._planning_token is None
            and prepared._source_scope is None
            and not run.control_unknown
            and run.status != "recovery_required"
        )

    def _release_pending_planning(self, run):
        if not self._planning_unused_return(run):
            raise RuntimeError("original pending planning return unproven")
        if run.workflow_id is not None:
            self._runs.pop(run.workflow_id)
        self._pending_planning.remove(run)

    async def _pending_planning_control(self, run, revision, action):
        import asyncio

        self._pending_planning_source(run)
        snapshot = run.planning_snapshot
        if (
            snapshot is None
            or snapshot.control_revision != revision
            or run.issuing.done()
            or run.prepared is None
        ):
            raise PermissionError("original pending planning control unavailable")
        run.control = action
        if run.status != "recovery_required":
            run.status = "pausing" if action == "pause" else "stopping"
        if action == "pause":
            return
        returned = run.control_return = asyncio.Event()
        cancellation = None
        run.prepared.cancel_event.set()
        run.issuing.cancel()
        try:
            while not run.issuing.done():
                try:
                    await asyncio.shield(run.issuing)
                except asyncio.CancelledError as error:
                    if asyncio.current_task().cancelling():
                        cancellation = error
                except BaseException:  # noqa: BLE001 - retain exact original command failure.
                    break
            if not run.issuing.cancelled():
                run.issuing.exception()
            if not self._planning_unused_return(run):
                raise RuntimeError("original pending planning settlement unproven")
            current = await self.service.call(self.service.get, run.workflow_id)
            if (
                current.control_revision != revision
                or current.request != snapshot.request
                or current.plans != snapshot.plans
            ):
                raise ValueError("revision_conflict")
            await self.service.call(
                self.service.set_status,
                run.workflow_id,
                expected_control_revision=revision,
                status="stopped",
            )
            self._release_pending_planning(run)
        except BaseException:
            run.control_unknown = True
            run.status = "recovery_required"
            raise
        finally:
            returned.set()
        if cancellation is not None:
            raise cancellation

    async def start(
        self, request: WorkflowRequest | WorkflowSetup, *, launch_id: str
    ) -> WorkflowSnapshot:
        """Accept the exact immutable request through original native planning."""
        from tldw_chatbook.Agents.workflow_models import (
            WorkflowPhaseAttempt,
            WorkflowRequest,
            WorkflowSetup,
        )

        from .console_workflow_preparation import prepare_workflow_planning

        if type(request) not in (WorkflowRequest, WorkflowSetup):
            raise TypeError("WorkflowRequest or WorkflowSetup required")

        if (
            type(request) is WorkflowSetup
            and WorkflowSetup.model_validate_json(request.canonical_json()) != request
        ):
            raise ValueError("invalid_workflow_setup")

        async def accept():
            import asyncio

            session, binding = self._selection(request)
            run = _WorkflowRun(
                None,
                session.id,
                binding,
                None,
                issuing=asyncio.current_task(),
                command_id=launch_id,
                planning_input=request,
                status="planning",
            )
            self._pending_planning.add(run)
            try:
                self._pending_planning_source(run, issuing=True)
                prepared = await prepare_workflow_planning(
                    self.controller,
                    session_id=session.id,
                    binding=binding,
                    command_id=launch_id,
                    objective=request.objective,
                    policy=request.policy,
                    _request=request,
                    _pending_run=run,
                )
                run.planning_returned = True
                self._pending_planning_source(run, issuing=True)
            except BaseException as error:
                run.planning_error = error
                run.planning_returned = True
                try:
                    if run.control != "stop":
                        self._release_pending_planning(run)
                except BaseException:  # noqa: BLE001 - retain original pending custody on any missing proof.
                    run.status = "recovery_required"
                raise
            if type(prepared) is WorkflowPhaseAttempt:
                self._release_pending_planning(run)
                return await self.get(prepared.workflow_id)
            prepared._planning_reads_returned()
            if run.prepared is not prepared:
                raise PermissionError("original planning preparation changed")
            run.phase = prepared.phase
            self._pending_planning.remove(run)
            try:
                driver = _planning_driver_source(self, run)
                run.driver = asyncio.create_task(
                    driver(run), name="native-workflow-planning"
                )
            except BaseException:
                run.status = "recovery_required"
                raise
            _metadata_source(self)
            await self._publish(run.workflow_id)
            return await self.get(run.workflow_id)

        return await self._command(
            ("start", launch_id), request.canonical_json(), accept
        )

    async def _plan(self, run):
        import asyncio

        from loguru import logger

        from tldw_chatbook.Agents.workflow_models import WorkflowSnapshot

        from . import console_workflow_preparation as planning_source

        planning_returned = False
        ready_to_release = False
        cancellation = None
        prepared = None
        release_return = None
        released_notification = None

        def release_source():
            planning_source._planning_release_source(
                prepared, prepared._planning_release_binding
            )

        try:
            _planning_driver_source(self, run, entered=True)
            prepared = run.prepared
            result = await prepared.run_planning(_coordinator_run=run)
            planning_returned = True
            release_source()
            release_return = prepared._planning_release_received()
            if run.control_return is not None:
                await run.control_return.wait()
            if run.control_unknown or run.status == "recovery_required":
                raise RuntimeError("workflow control return unproven")
            if self._runs.get(run.workflow_id) is not run:
                return
            if run.control == "pause":
                await self.service.call(
                    self.service.set_status,
                    run.workflow_id,
                    status="paused",
                    expected_control_revision=result.control_revision,
                )
            if run.control_unknown or run.status == "recovery_required":
                raise RuntimeError("workflow original return unproven")
            run.status = None
            ready_to_release = True
        except asyncio.CancelledError as error:
            cancellation = error
            try:
                if (
                    planning_returned
                    or run.control_unknown
                    or run.status == "recovery_required"
                ):
                    raise RuntimeError("workflow cancellation return unproven")
                _planning_driver_source(self, run, entered=True)
                prepared = run.prepared
                prepared._planning_return_source()
                _prove_workflow_read(
                    prepared._planning_read_return,
                    self.controller,
                    prepared,
                    run.driver,
                )
                terminal = prepared._planning_read_return[1].result()
                owner, claim = prepared.primary_owner, prepared.primary_claim
                drained = []
                if owner is not None:
                    owner.on_drained(drained.append)
                if (
                    type(terminal) is not WorkflowSnapshot
                    or terminal.id != run.workflow_id
                    or not prepared._planning_settled
                    or not prepared._closed
                    or owner is None
                    or owner.is_admitted
                    or drained != [True]
                    or claim is None
                    or claim.id in self.runtime.worktree_activity._claims
                    or prepared._planning_intake is not None
                    or prepared._planning_token is not None
                ):
                    raise RuntimeError("workflow cancellation retirement unproven")
                release_source()
                release_return = prepared._planning_release_received(
                    _cancellation=error
                )
                run.status = None
                ready_to_release = True
            except BaseException:  # noqa: BLE001 - uncertain original return retains custody.
                run.status = "recovery_required"
                logger.exception("Native workflow cancellation remains unresolved")
        except BaseException:  # noqa: BLE001 - original producer retains uncertain custody.
            run.status = "recovery_required"
            logger.exception("Native workflow planning did not retire normally")
        try:
            _metadata_source(self)
            publication = await self._publish(
                run.workflow_id, _terminal_run=run if ready_to_release else None
            )
            if ready_to_release:
                cancellation = publication.caller_error or cancellation
                _metadata_publication_return(self, run, publication)
                if (
                    self._runs.get(run.workflow_id) is not run
                    or run.control_unknown
                    or run.status == "recovery_required"
                ):
                    raise RuntimeError("workflow original publication return unproven")
                popped_run = self._runs.pop(run.workflow_id)
                if popped_run is not run:
                    raise RuntimeError("original planning run did not return")
                released_notification = (
                    prepared,
                    release_return,
                    publication,
                    popped_run,
                )
        except BaseException as error:  # noqa: BLE001 - retain original metadata failure separately from cancellation.
            self._retain_final_metadata(run, error)
            if isinstance(error, asyncio.CancelledError):
                cancellation = error
            logger.exception("Native workflow final publication did not return")
        if released_notification is not None:
            try:
                release_source()
                prepared._notify_planning_release(
                    release_return, _publication=publication, _popped_run=popped_run
                )
            except BaseException as error:
                logger.exception("Native planning returned; wake notification failed")
                if cancellation is not None:
                    raise cancellation from error
                raise
        if cancellation is not None:
            if (
                run.final_metadata is not None
                and run.final_metadata.producer_error is not None
            ):
                raise cancellation from run.final_metadata.producer_error
            raise cancellation

    def _retain_final_metadata(self, run, error):
        run.status = "recovery_required"
        cell = run.final_metadata
        if cell is not None:
            cell.source_error = cell.source_error or error
            self._metadata_cells[cell] = cell.disposed

    async def approve(
        self, decision: WorkflowDecision, *, command_id: str
    ) -> WorkflowSnapshot:
        """Accept initial or remaining work, or record explicit quality acceptance."""
        from tldw_chatbook.Agents.workflow_models import WorkflowDecision

        from .console_chat_models import ConsoleSubmissionOrigin

        if type(decision) is not WorkflowDecision:
            raise TypeError("WorkflowDecision required")

        async def accept():
            import asyncio

            if decision.action == "complete":
                await self.service.call(
                    self.service.review_quality, decision, accepted=True
                )
                _metadata_source(self)
                await self._publish(decision.workflow_id)
                return await self.get(decision.workflow_id)
            if decision.action != "approve":
                raise ValueError("workflow_approval_required")
            snapshot = await self.service.call(self.service.get, decision.workflow_id)
            session, binding = self._selection(snapshot.request)
            if decision.workflow_id in self._runs:
                raise PermissionError("workflow original producer still active")
            token = object()
            if not self.controller._fleet_wake.try_claim_automatic_primary(
                session.id, token
            ):
                raise PermissionError("workflow primary unavailable")
            intake = self.controller.store.claim_received_turn(
                session.id, command_id, origin=ConsoleSubmissionOrigin.WORKFLOW_PHASE
            )
            if intake is None:
                self.controller._fleet_wake.release_automatic_primary(session.id, token)
                raise PermissionError("workflow intake unavailable")
            run = _WorkflowRun(
                decision.workflow_id,
                session.id,
                binding,
                None,
                token=token,
                intake=intake,
                status="running",
                issuing=asyncio.current_task(),
                command_id=command_id,
                decision=decision,
            )
            self._runs[run.workflow_id] = run
            try:
                phase = await _accept_remaining_workflow_phase(
                    self.controller,
                    decision,
                    command_id=command_id,
                    session_id=session.id,
                    binding=binding,
                    _initial=decision.reason == "ready",
                    _pending_run=run,
                )
            except BaseException as error:
                # No effect precedes cell association in this exact producer.
                # Missing handles AFTER association never count as retirement.
                early = run.acceptance is None
                if early or self._acceptance_returned(run, refusal=error):
                    if run.control is None:
                        self._release_admission(run)
                else:
                    run.status = "recovery_required"
                raise
            if phase is None:
                if not self._acceptance_returned(run, duplicate=True):
                    run.status = "recovery_required"
                    raise RuntimeError("original workflow acceptance return unproven")
                self._release_admission(run)
                return await self.get(decision.workflow_id)
            run.phase = phase
            run.driver = asyncio.create_task(
                self._execute(run), name="native-workflow-phase"
            )
            _metadata_source(self)
            await self._publish(run.workflow_id)
            return await self.get(run.workflow_id)

        return await self._command(
            ("approve", command_id), decision.canonical_json(), accept
        )

    def _acceptance_returned(self, run, *, duplicate=False, refusal=None):
        cell = run.acceptance
        if (
            type(cell) is not _RemainingAcceptance
            or cell.controller is not self.controller
            or cell.service is not self.service
            or cell.issuing is not run.issuing
            or cell.decision != run.decision
            or not cell.returned
            or not cell.scopes_returned
            or not cell.worker_returned
            or cell.owner_return != [True]
            or cell.owner is None
            or cell.owner.is_admitted
            or cell.claim is None
            or cell.claim.id in self.runtime.worktree_activity._claims
        ):
            return False
        if (
            refusal is not None
            and cell.body_error is refusal
            and not cell.preparation_entered
        ):
            try:
                _prove_workflow_read(
                    cell.read_return, self.controller, cell, run.issuing
                )
                cell.check_source()
            except BaseException:  # noqa: BLE001 - any missing original proof retains custody.
                return False
            return (
                cell.attempt is None
                and cell.phase is None
                and cell.accepted_attempt is None
            )
        if cell.accepted_attempt is not None:
            return (
                cell.terminal_scopes_returned
                and cell.terminal_result is not None
                and cell.phase is None
            )
        return (
            duplicate and cell.attempt is not None and cell.attempt.state != "prepared"
        )

    def _release_admission(self, run):
        if self._runs.get(run.workflow_id) is not run:
            raise PermissionError("original workflow run changed")
        if run.control_unknown or run.status == "recovery_required":
            raise RuntimeError("workflow original return unproven")
        self.controller.store.release_received_turn(run.intake)
        self.controller._fleet_wake.release_automatic_primary(run.session_id, run.token)
        self._runs.pop(run.workflow_id)

    async def _pending_control(self, run, expected_control_revision, action):
        import asyncio

        if (
            run.issuing is None
            or run.issuing.done()
            or run.decision is None
            or expected_control_revision
            not in {run.decision.control_revision, run.decision.control_revision + 1}
        ):
            raise PermissionError("original pending workflow command required")
        run.control = action
        if run.status != "recovery_required":
            run.status = "pausing" if action == "pause" else "stopping"
        run.issuing.cancel()
        while not run.issuing.done():
            try:
                await asyncio.shield(run.issuing)
            except asyncio.CancelledError:
                continue
            except BaseException:  # noqa: BLE001 - inspect original issuing task before any release.
                break
        if not run.issuing.cancelled():
            run.issuing.exception()
        try:
            cell = run.acceptance
            if not self._acceptance_returned(run):
                raise RuntimeError("original pending workflow settlement unproven")
            attempt, terminal = cell.accepted_attempt, cell.terminal_result
            if (
                attempt.workflow_id != run.workflow_id
                or attempt.command_id != run.command_id
                or attempt.phase_id != run.decision.phase_id
                or attempt.revision_id != run.decision.plan_revision
                or attempt.control_revision != run.decision.control_revision + 1
                or terminal.id != run.workflow_id
                or terminal.status != "awaiting_resume"
                or terminal.control_revision != run.decision.control_revision + 2
            ):
                raise ValueError("workflow_pending_transition_changed")
            current = await self.service.call(self.service.get, run.workflow_id)
            if (
                current.control_revision != terminal.control_revision
                or current.status != terminal.status
            ):
                raise ValueError("revision_conflict")
            if action == "stop":
                await self.service.call(
                    self.service.set_status,
                    run.workflow_id,
                    expected_control_revision=terminal.control_revision,
                    status="stopped",
                )
            self._release_admission(run)
        except BaseException:
            run.status = "recovery_required"
            raise

    def _subscribe_availability(self, run):
        """Use native release notifications; callbacks only wake this owning loop."""
        import asyncio
        from uuid import uuid4

        run.wake = asyncio.Event()
        bridge = self.controller._agent_bridge
        name = "workflow-" + uuid4().hex

        def released(*_args):
            self.loop.call_soon_threadsafe(run.wake.set)

        unsubscribe_capacity = (
            self.runtime.execution_capacity.subscribe_tool_availability(released)
        )
        unsubscribe_activity = self.runtime.worktree_activity.subscribe(released)
        bridge.on_fleet_activity(name, released)

        def unsubscribe():
            unsubscribe_capacity()
            unsubscribe_activity()
            with bridge._fleet_activity_lock:
                if bridge._fleet_activity_consumers.get(name) is released:
                    bridge._fleet_activity_consumers.pop(name)

        return unsubscribe

    def _wave_ready(self, run, candidates):
        from tldw_chatbook.Agents.execution_capacity import WorkOrigin

        capacity, activity = (
            self.runtime.execution_capacity,
            self.runtime.worktree_activity,
        )
        activity._prune_recovery()
        observed = capacity.snapshot()
        if observed.closed or self.runtime._workflow_closing:
            raise PermissionError("workflow runtime closed")
        selected, fleets = [], {}
        for item in candidates:
            fleet = item.authorization._native_preparation[0].fleet
            with fleet._lock:
                if fleet._fenced:
                    run.control = "stop"
                    return []  # No launch occurred; return the unused native preparations.
                fleet_ready = (
                    len(fleet._live_ids) + fleets.get(fleet, 0) < fleet.max_live
                )
            count = len(selected) + 1
            if not fleet_ready or not capacity._workflow_children_ready(count):
                continue
            automatic_tools = sum(
                owner.tool_workers
                for owner in observed.executions
                if owner.origin is WorkOrigin.AUTOMATIC
            )
            if (
                observed.tool_workers + 2 * count > capacity.max_tool_workers
                or automatic_tools + 2 * count
                > capacity.max_tool_workers - capacity.reserved_manual_tool_workers
            ):
                continue
            footprint = item.authorization.physical_footprint
            if any(
                footprint._conflicts(other.authorization.physical_footprint)
                for other in selected
            ):
                continue
            waiting_footprint = run.wait_footprints.get(item, footprint)
            with activity._lock:
                blocked = activity._ready_manual or any(
                    waiting_footprint._conflicts(other)
                    for other in (
                        *(state.footprint for state in activity._claims.values()),
                        *activity._recovery.values(),
                        *(fence.footprint for fence in activity._waves.values()),
                    )
                )
            if blocked:
                continue
            selected.append(item)
            fleets[fleet] = fleets.get(fleet, 0) + 1
        return selected

    async def _accounting(self, run, mode):
        """Track the original native clock transaction through actual return."""
        import asyncio

        from tldw_chatbook.Agents.activation import worker_guard
        from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

        from . import console_preparation_reads as native

        if (
            mode not in {"quiesce", "resume", "cleanup"}
            or asyncio.current_task() is not run.driver
        ):
            raise PermissionError("original accounting driver required")
        phase = run.phase
        cell = phase._quiescence
        if mode == "quiesce":
            if cell is not None:
                if cell.mode != "idle" or cell.error is not None:
                    raise RuntimeError("original workflow accounting unresolved")
                _quiescence_source(cell)
                return
            cell = _WorkflowQuiescence(
                self,
                run,
                phase,
                asyncio.current_task(),
                tuple(run.preparation_returns),
                tuple(run.waves),
            )
            phase._quiescence = cell
        elif cell is None:
            return
        elif cell.mode != "idle" or cell.error is not None:
            raise RuntimeError("original workflow accounting unresolved")
        # Consume idle BEFORE admitting the resume/cleanup read. A failed write
        # can never restore it or make a later passive observation exact.
        cell.mode = mode
        cell.proof = None
        try:
            _quiescence_physical(cell)
        except BaseException as error:
            cell.error = run.accounting_error = error
            raise
        ledger = phase._ledger

        @worker_guard(self.controller._agent_bridge)
        def operation():
            with owned_goal_storage(self.service), phase.context.scope():
                _quiescence_source(cell)
                if mode == "quiesce":
                    phase.check()
                    return ledger.quiesce_workflow_phase(
                        phase.attempt.id,
                        owner_id=phase.attempt.owner_id,
                        drained=True,
                        _quiescence=cell,
                    )
                return ledger.resume_workflow_phase(
                    phase.attempt.id,
                    owner_id=phase.attempt.owner_id,
                    _quiescence=cell,
                    _cleanup=mode == "cleanup",
                )

        def invoke():
            cell.proof = _capture_workflow_read(self.controller, cell, cell.issuing)
            with _remaining_storage_return(
                (
                    self.service.db,
                    self.service.persistence.db,
                    self.service.persistence.workspace_registry.db,
                )
            ):
                return operation()

        cell.callback = invoke
        try:
            await native.run_preparation_read(
                invoke,
                creator=self.controller,
                session_id=run.session_id,
                reads=self.controller._preparation_reads,
                observers=(self.runtime._preparation_reads,),
                require_current=lambda: _quiescence_source(cell),
                source=cell,
            )
            _quiescence_source(cell)
            _prove_workflow_read(cell.proof, self.controller, cell, cell.issuing)
            if mode == "quiesce":
                cell.interval = cell.proof[1].result()
                if not isinstance(cell.interval, tuple):
                    raise RuntimeError("original workflow interval return missing")
                cell.idle_proof, cell.idle_callback = cell.proof, invoke
                cell.mode = "idle"
            else:
                cell.mode = "active"
                phase._quiescence = None
        except BaseException as error:
            cell.error = error
            if cell.proof is not None:
                producer = cell.proof[1]
                if producer.done() and not producer.cancelled():
                    cell.producer_error = producer.exception()
            run.accounting_error = error
            raise

    async def _execute(self, run):
        import asyncio

        from loguru import logger

        from .console_goal_runs import goal_provider_ref
        from .console_workflow_preparation import prepare_workflow_reply
        from .console_workflow_wave import run_workflow_wave

        unsubscribe = self._subscribe_availability(
            run
        )  # Subscribe BEFORE every initial recheck.
        pending = []
        failed = False
        cancellation = None
        released_notification = None
        try:
            wake = self.controller._fleet_wake
            release_binding = (
                self,
                run,
                run.phase,
                asyncio.current_task(),
                self.loop,
                self.runtime,
                self.controller,
                self.controller.store,
                wake,
                run.intake,
                run.token,
                wake._registry_lock,
                wake._loop,
            )
            _execution_release_source(self, release_binding)
            snapshot = await self.service.call(self.service.get, run.workflow_id)
            definition = next(
                p
                for p in snapshot.plans[run.phase.attempt.revision_id - 1].phases
                if p.id == run.phase.attempt.phase_id
            )
            ids = set(await self.service.call(self.service.phase_task_ids, run.phase))
            pending = [task for task in definition.tasks if task.id in ids]
            done = set(
                await self.service.call(
                    self.service.reviewed_task_ids, run.phase, run.decision
                )
            )
            while pending and run.control is None and not failed:
                run.wake.clear()
                ready = [task for task in pending if set(task.dependencies) <= done]
                if not ready:
                    raise ValueError("workflow dependencies blocked")
                # Native finite preparation uses OPAQUE custody. Observe its
                # existing admission gate before entering, without claiming it.
                activity = self.runtime.worktree_activity
                activity._prune_recovery()
                with activity._lock:
                    busy = bool(
                        activity._claims
                        or activity._waves
                        or activity._recovery
                        or activity._ready_manual
                    )
                if not busy:
                    for task in ready:
                        if run.control is not None:
                            break
                        if task.id in run.preparations:
                            continue
                        await self._accounting(run, "resume")
                        resolution = next(
                            r
                            for r in run.phase._execution_resolutions
                            if goal_provider_ref(r) == task.target
                        )
                        run.preparations[task.id] = await prepare_workflow_reply(
                            self.controller._agent_bridge,
                            phase=run.phase,
                            task_id=task.id,
                            binding=run.binding,
                            resolution=resolution,
                            confirm_project_instruction_dispatch=self.controller._confirm_project_instruction_dispatch,
                        )
                        item = run.preparations[task.id]
                        run.preparation_returns.append(
                            (
                                item,
                                item.authorization,
                                item.authorization._native_preparation[0],
                                item.native,
                                item.configuration,
                                item.source,
                                item._lifetime,
                                item._source_scope,
                                item._finite_reads,
                            )
                        )
                if run.control is not None:
                    break
                selected = self._wave_ready(
                    run,
                    [run.preparations[t.id] for t in ready if t.id in run.preparations],
                )
                if run.control is not None:
                    break
                if not selected:
                    await self._accounting(run, "quiesce")
                    if run.control is not None or run.wake.is_set():
                        continue
                    if self._wave_ready(
                        run,
                        [
                            run.preparations[t.id]
                            for t in ready
                            if t.id in run.preparations
                        ],
                    ):
                        continue
                    await run.wake.wait()
                    continue
                await self._accounting(run, "resume")
                wave = await run_workflow_wave(run.phase, selected, _retain_waits=True)
                run.waves.append(_quiescence_wave._workflow_wave_return(wave))
                for item in wave.unlaunched:
                    if wave.wait_footprint is not None:
                        run.wait_footprints[item] = wave.wait_footprint
                snapshot = await self.service.call(self.service.get, run.workflow_id)
                results = {result.attempt_id: result for result in snapshot.results}
                for item, _receipt in wave.receipts:
                    result = results.get(item.authorization.claim.id)
                    failed |= result is None or result.status != "completed"
                    done.add(item.task_id)
                    pending = [task for task in pending if task.id != item.task_id]
                    run.preparations.pop(item.task_id)
                    run.wait_footprints.pop(item, None)
                _metadata_source(self)
                await self._publish(run.workflow_id)
            cleanup_error = None
            try:
                await self._accounting(run, "cleanup")
            except BaseException as error:  # noqa: BLE001 - accounting refusal cannot prevent already-owned physical cleanup.
                cleanup_error = error
            for item in run.preparations.values():
                await item.close()
            if cleanup_error is not None:
                raise cleanup_error
            if run.control_return is not None:
                await run.control_return.wait()
            if run.control_unknown:
                raise RuntimeError("workflow control return unproven")
            await self.service.call(
                self.service.settle_execution_phase,
                run.phase,
                outcome="failed"
                if failed
                else "cancelled"
                if pending or run.control
                else "completed",
            )
            if run.status == "recovery_required" or run.control_unknown:
                raise RuntimeError("workflow original return unproven")
            run.status = None
            _metadata_source(self)
            publication = await self._publish(run.workflow_id, _terminal_run=run)
            cancellation = publication.caller_error or cancellation
            _execution_release_source(self, release_binding)
            _metadata_publication_return(self, run, publication)
            released_notification = self._release(run, release_binding, publication)
        except BaseException as error:  # noqa: BLE001 - physical failure cannot retry or shed original custody.
            self._retain_final_metadata(run, error)
            if isinstance(error, asyncio.CancelledError):
                cancellation = error
            logger.exception("Native workflow phase did not retire normally")
        finally:
            unsubscribe()
            if run.accounting_error is not None:
                for item in run.preparations.values():
                    if not item._closed:
                        try:
                            await item.close()
                        except BaseException:  # noqa: BLE001 - original cleanup refusal remains retained with its accounting error.
                            logger.exception(
                                "Original workflow cleanup remains unresolved"
                            )
        if released_notification is not None:
            try:
                _execution_release_source(self, released_notification, released=True)
                self._notify_execution_release(released_notification)
            except BaseException as error:
                logger.exception("Native workflow returned; wake notification failed")
                if cancellation is not None:
                    raise cancellation from error
                raise
        if run.final_metadata is None:
            _metadata_source(self)
            await self._publish(run.workflow_id)
        if cancellation is not None:
            raise cancellation

    def _release(self, run, binding, publication):
        _execution_release_source(self, binding)
        if binding[1] is not run or not run.phase._retired:
            raise PermissionError("original workflow release invocation required")
        _metadata_publication_return(self, run, publication)
        if run.control_unknown or run.status == "recovery_required":
            raise RuntimeError("workflow original return unproven")
        store, wake, intake, token, registry_lock = binding[7:12]
        with store._preparation_lock:
            if (
                intake._store_ref() is not store
                or store._preparations_by_session.get(run.session_id) is not intake
            ):
                raise PermissionError("original workflow received claim changed")
        with registry_lock:
            if wake._automatic_primary_claims.get(run.session_id) is not token:
                raise PermissionError("original workflow primary token changed")
        if store.release_received_turn(intake) is not True:
            raise PermissionError("original workflow received claim did not return")
        if wake.release_automatic_primary(run.session_id, token) is not True:
            raise PermissionError("original workflow primary token did not return")
        if self._runs.pop(run.workflow_id) is not run:
            raise PermissionError("original workflow run did not return")
        return binding

    def _notify_execution_release(self, binding):
        _execution_release_source(self, binding, released=True)
        runtime, wake, registry_lock = binding[5], binding[8], binding[11]
        with registry_lock:
            if wake._registry_lock is not registry_lock:
                raise PermissionError("original workflow wake lock changed")
            if runtime._workflow_closing and wake._disposed is not True:
                raise PermissionError("workflow wake shutdown fence unproven")
        wake.retry_soon()

    async def _native_control(self, run, action, decision, expected_control_revision):
        """Track the original control's complete worker and reader return."""
        import asyncio

        from tldw_chatbook.Agents.activation import _guard, worker_guard
        from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

        from . import console_preparation_reads as native

        phase, controller, bridge = (
            run.phase,
            self.controller,
            self.controller._agent_bridge,
        )
        service, runtime, store = self.service, self.runtime, self.controller.store
        issuing = asyncio.current_task()
        sources = _guard.captured_sources(bridge)
        databases = (
            service.db,
            service.persistence.db,
            service.persistence.workspace_registry.db,
        )
        returned = []
        if (
            run.control_unknown
            or run.control_return is not None
            and not run.control_return.is_set()
        ):
            raise PermissionError("workflow original control pending")
        control_return = run.control_return = asyncio.Event()

        def current():
            if (
                controller.app.console_runtime is not runtime
                or runtime._workflow_coordinator is not self
                or runtime.chat_controller is not controller
                or controller._agent_bridge is not bridge
                or controller.store is not store
                or bridge._store is not store
                or runtime._workflow_service is not service
                or service.db is not bridge.runs_db
                or self._runs.get(run.workflow_id) is not run
                or run.phase is not phase
                or run.control_return is not control_return
                or _guard.captured_sources(bridge) != sources
            ):
                raise PermissionError("workflow original control source changed")
            phase._check_owner()

        @worker_guard(bridge)
        def controlled():
            with owned_goal_storage(service):
                current()
                return phase._control(
                    decision, expected_control_revision=expected_control_revision
                )

        def invoke():
            run.control_read = _capture_workflow_read(controller, phase, issuing)
            with _remaining_storage_return(databases):
                result = controlled()
            returned.append(result)
            return result

        error = None
        try:
            await native.run_preparation_read(
                invoke,
                creator=controller,
                session_id=run.session_id,
                reads=controller._preparation_reads,
                observers=(runtime._preparation_reads,),
                require_current=current,
                source=phase,
            )
        except BaseException as caught:  # noqa: BLE001 - preserve cancellation separately from original return proof.
            error = caught
        try:
            current()
            if run.control_read is None or not returned:
                raise RuntimeError("workflow control worker return unproven")
            read, producer, retired = run.control_read
            if (
                read.creator is not controller
                or read.source is not phase
                or read.task is not issuing
                or read._producer is not producer
                or read.retired is not retired
                or not asyncio.Future.done(producer)
                or asyncio.Future.cancelled(producer)
                or asyncio.Future.exception(producer) is not None
                or not asyncio.Future.done(retired)
                or asyncio.Future.cancelled(retired)
                or asyncio.Future.result(retired) is not None
                or error is not None
                and not isinstance(error, asyncio.CancelledError)
            ):
                raise RuntimeError("workflow control reader return unproven")
            self._metadata_lower_bound(returned[0])
            run.control = action
            if run.status != "recovery_required":
                run.status = "pausing" if action == "pause" else "stopping"
            if run.wake is not None:
                run.wake.set()
        except BaseException:
            run.control_unknown = True
            run.status = "recovery_required"
            raise
        finally:
            control_return.set()
        if error is not None:
            raise error
        return returned[0]

    async def _control(self, workflow_id, expected_control_revision, action):
        from uuid import uuid4

        from tldw_chatbook.Agents.workflow_models import WorkflowDecision

        async def control():
            snapshot = await self.service.call(self.service.get, workflow_id)
            if snapshot.control_revision != expected_control_revision:
                raise ValueError("revision_conflict")
            run = self._runs.get(workflow_id)
            if run is None:
                await self.service.call(
                    self.service.set_status,
                    workflow_id,
                    status=action + "d" if action == "pause" else "stopped",
                    expected_control_revision=expected_control_revision,
                )
            else:
                if (
                    run.control_unknown
                    or run.control_return is not None
                    and not run.control_return.is_set()
                ):
                    raise PermissionError("workflow original control pending")
                if run in self._pending_planning:
                    await self._pending_planning_control(
                        run, expected_control_revision, action
                    )
                elif run.phase is None:
                    await self._pending_control(run, expected_control_revision, action)
                elif run.phase.attempt.purpose == "planning":
                    if action == "stop":
                        await self._native_control(
                            run, action, None, expected_control_revision
                        )
                else:
                    decision = WorkflowDecision(
                        id=str(uuid4()),
                        workflow_id=workflow_id,
                        plan_revision=run.phase.attempt.revision_id,
                        phase_id=run.phase.attempt.phase_id,
                        control_revision=expected_control_revision,
                        action=action,
                        reason="user_request",
                    )
                    await self._native_control(
                        run, action, decision, expected_control_revision
                    )
                if self._runs.get(workflow_id) is run:
                    run.control = action
                    if run.status != "recovery_required":
                        run.status = "pausing" if action == "pause" else "stopping"
            _metadata_source(self)
            await self._publish(workflow_id)
            return await self.get(workflow_id)

        return await self._command(
            (action, workflow_id, expected_control_revision), action, control
        )

    async def pause(
        self, workflow_id: str, *, expected_control_revision: int
    ) -> WorkflowSnapshot:
        """Stop new launches while the original committed work returns."""
        return await self._control(workflow_id, expected_control_revision, "pause")

    async def stop(
        self, workflow_id: str, *, expected_control_revision: int
    ) -> WorkflowSnapshot:
        """Revoke the original phase before requesting native cancellation."""
        return await self._control(workflow_id, expected_control_revision, "stop")

    async def resume(
        self, workflow_id: str, *, expected_control_revision: int, command_id: str
    ) -> WorkflowSnapshot:
        """Request fresh explicit review without reconstructing an execution cap."""

        async def resume():
            if workflow_id in self._runs:
                raise PermissionError("workflow original producer still active")
            await self.service.call(
                self.service.resume,
                workflow_id,
                expected_control_revision=expected_control_revision,
                command_id=command_id,
            )
            _metadata_source(self)
            await self._publish(workflow_id)
            return await self.get(workflow_id)

        return await self._command(
            ("resume", command_id), (workflow_id, expected_control_revision), resume
        )

    async def reconcile(
        self,
        workflow_id: str,
        *,
        expected_control_revision: int,
        decision: WorkflowExecutionRecovery,
        command_id: str,
    ) -> WorkflowSnapshot:
        """Record explicit review of interrupted history after native work retires."""

        async def reconcile():
            revision = expected_control_revision
            run = self._runs.get(workflow_id)
            if run is not None:
                if run.prepared is None or run.driver is None or not run.driver.done():
                    raise ValueError("workflow_resources_still_live")
                run.prepared.planning_recovery_return()
                if run.prepared._planning_recovery_snapshot is None:
                    recorded = await run.prepared.record_planning_recovery(
                        expected_control_revision=revision
                    )
                    revision = recorded.control_revision
            await self.service.call(
                self.service.resolve_recovery,
                workflow_id,
                expected_control_revision=revision,
                decision=decision,
            )
            if run is not None:
                run.prepared.planning_recovery_return()
                if self._runs.get(workflow_id) is not run:
                    raise ValueError("workflow_runtime_changed")
                self._runs.pop(workflow_id)
            await self._publish(workflow_id)
            return await self.get(workflow_id)

        return await self._command(
            ("reconcile", command_id),
            (workflow_id, expected_control_revision, decision.canonical_json()),
            reconcile,
        )

    async def remove_results(
        self, workflow_id: str, *, expected_control_revision: int, command_id: str
    ) -> WorkflowSnapshot:
        """Remove terminal payloads through the existing owned history service."""

        async def remove():
            await self.service.call(
                self.service.remove_payloads,
                workflow_id,
                expected_control_revision=expected_control_revision,
            )
            await self._publish(workflow_id)
            return await self.get(workflow_id)

        return await self._command(
            ("remove", command_id), (workflow_id, expected_control_revision), remove
        )

    async def revise(
        self,
        workflow_id: str,
        *,
        expected_control_revision: int,
        instruction: str,
        command_id: str,
    ) -> WorkflowSnapshot:
        """Prepare a native repair against the unchanged saved request and history."""
        from tldw_chatbook.Agents.workflow_models import WorkflowPhaseAttempt

        from .console_workflow_preparation import prepare_workflow_planning

        async def revise():
            import asyncio

            if workflow_id in self._runs:
                raise PermissionError("workflow original producer still active")
            snapshot = await self.service.call(self.service.get, workflow_id)
            if snapshot.control_revision != expected_control_revision:
                raise ValueError("revision_conflict")
            session, binding = self._selection(snapshot.request)
            run = _WorkflowRun(
                workflow_id,
                session.id,
                binding,
                None,
                issuing=asyncio.current_task(),
                command_id=command_id,
                planning_input=(len(snapshot.plans), instruction),
                status="planning",
            )
            self._runs[workflow_id] = run
            self._pending_planning.add(run)
            try:
                self._pending_planning_source(run, issuing=True)
                prepared = await prepare_workflow_planning(
                    self.controller,
                    session_id=session.id,
                    binding=binding,
                    command_id=command_id,
                    workflow_id=workflow_id,
                    revision_id=len(snapshot.plans),
                    repair_instruction=instruction,
                    _pending_run=run,
                )
                run.planning_returned = True
                self._pending_planning_source(run, issuing=True)
            except BaseException as error:
                run.planning_error = error
                run.planning_returned = True
                try:
                    if run.control != "stop":
                        self._release_pending_planning(run)
                except BaseException:  # noqa: BLE001 - retain original pending custody on any missing proof.
                    run.status = "recovery_required"
                raise
            if type(prepared) is WorkflowPhaseAttempt:
                self._release_pending_planning(run)
                return await self.get(workflow_id)
            prepared._planning_reads_returned()
            if run.prepared is not prepared:
                raise PermissionError("original planning preparation changed")
            run.phase = prepared.phase
            self._pending_planning.remove(run)
            try:
                driver = _planning_driver_source(self, run)
                run.driver = asyncio.create_task(
                    driver(run), name="native-workflow-repair"
                )
            except BaseException:
                run.status = "recovery_required"
                raise
            _metadata_source(self)
            await self._publish(workflow_id)
            return await self.get(workflow_id)

        return await self._command(
            ("revise", command_id),
            (workflow_id, expected_control_revision, instruction),
            revise,
        )

    async def close(self, *, _runtime_disposal=None) -> asyncio.CancelledError | None:
        """Seal subscriptions and await the same command and driver invocations."""
        import asyncio

        if _runtime_disposal is not None and (
            _runtime_disposal is not self.runtime
            or self.controller.app.console_runtime is not _runtime_disposal
            or _runtime_disposal._workflow_coordinator is not self
        ):
            raise PermissionError("original workflow disposal runtime required")
        self._closing = True
        cancellation = None
        self._subscribers.clear()
        for _identity, command in tuple(self._commands.values()):
            if not command.done():
                command.cancel()
        for run in set(self._runs.values()) | self._pending_planning:
            run.control = "stop"
            if run.wake is not None:
                run.wake.set()
            if run.prepared is not None:
                run.prepared.cancel_event.set()
            for wave in tuple(self.runtime._workflow_waves):
                if wave.phase is run.phase:
                    wave.stopping.set()
        tasks = [task for _identity, task in self._commands.values()]
        tasks.extend(
            run.driver for run in self._runs.values() if run.driver is not None
        )
        for task in tasks:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    if asyncio.current_task().cancelling():
                        cancellation = error
                    continue
                except BaseException:  # noqa: BLE001 - retain original failure, never fabricate retirement.
                    break
            if not task.cancelled():
                task.exception()
        for run in tuple(self._runs.values()):
            if (
                run.prepared is not None
                and run.driver is not None
                and run.driver.done()
                and run.prepared._planning_recovery_return is not None
            ):
                run.prepared.planning_recovery_return()
                self._runs.pop(run.workflow_id)
                continue  # Durable unreviewed history survives runtime disposal.
            if (
                run not in self._pending_planning
                and run.phase is None
                and self._acceptance_returned(run)
            ):
                self._release_admission(run)
        for run in tuple(self._pending_planning):
            if self._planning_unused_return(run):
                self._release_pending_planning(run)
        for cell, disposition in tuple(self._metadata_cells.items()):
            if cell.issuing is asyncio.current_task():
                raise RuntimeError("workflow metadata cannot drain its own host")
            while not disposition.done():
                try:
                    await asyncio.shield(disposition)
                except asyncio.CancelledError as error:
                    cancellation = error
            if disposition.cancelled() or disposition.result() is not True:
                raise RuntimeError("workflow_original_metadata_shutdown_unresolved")
        if self._metadata_cells:
            raise RuntimeError("workflow_original_metadata_shutdown_unresolved")
        if self._runs or self._pending_planning:
            raise RuntimeError("workflow_original_shutdown_unresolved")
        if cancellation is not None:
            if _runtime_disposal is self.runtime:
                return cancellation
            raise cancellation


from . import console_workflow_wave as _quiescence_wave

_COORDINATOR_CLASS = ConsoleWorkflowCoordinator
_QUIESCENCE_CLASS = _WorkflowQuiescence
_METADATA_CELL_CLASS = _WorkflowMetadataRead
