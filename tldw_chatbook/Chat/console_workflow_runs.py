"""Private accepted workflow custody; execution coordination belongs to ConsoleRuntime."""

from __future__ import annotations

import inspect
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from threading import RLock

from tldw_chatbook import config
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkRefused,
)
from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

_KEY = object()
_CONFIG_IDENTITY = config.current_config_identity
_CONFIG_IDENTITY_CODE = _CONFIG_IDENTITY.__code__
_CLOCKS = (time.time, time.monotonic)
_POLICY_ENV = ("TLDW_AGENTS_WORKFLOW_RUNS_ENABLED",) + tuple(
    "TLDW_AGENTS_MAX_WORKFLOW_" + item.name.upper()
    for item in fields(AutomaticWorkLimits)
)


def _policy_identity():
    if (
        config.current_config_identity is not _CONFIG_IDENTITY
        or _CONFIG_IDENTITY.__code__ is not _CONFIG_IDENTITY_CODE
    ):
        raise AutomaticWorkRefused("workflow_policy_source_changed")
    return _CONFIG_IDENTITY(), tuple(os.environ.get(name) for name in _POLICY_ENV)


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
    """Consume only the exact active original invocation, without source I/O."""
    if (
        type(fence) is not _WorkflowFence
        or fence.key is not _KEY
        or fence.phase._active_fence is not fence
        or fence.phase._ledger is not ledger
        or fence.thread is not threading.current_thread()
        or getattr(ledger._db._thread_local, "conn", None) is not fence.connection
        or (ledger._wall_clock, ledger._monotonic_clock) != _CLOCKS
        or (time.time, time.monotonic) != _CLOCKS
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
    def _fence(self, operation):
        from tldw_chatbook.Agents.activation import execution

        caller = inspect.currentframe().f_back.f_back
        producer = (
            WorkflowPhaseAuthorization._commit_task_launch
            if operation == "child_commit"
            else WorkflowPhaseAuthorization._control
            if operation == "control"
            else None
        )
        if (
            producer is None
            or caller is None
            or caller.f_code is not producer.__code__
            or caller.f_locals.get("self") is not self
        ):
            raise AutomaticWorkRefused("workflow_original_fence_producer_required")
        self._check_owner()
        before = _policy_identity()
        if operation == "child_commit":
            self._ledger._workflow_enabled()
        limits = AutomaticWorkLimits.from_settings("workflow_phase")
        if before != _policy_identity():
            raise AutomaticWorkRefused("workflow_policy_source_changed")
        if (self._ledger._wall_clock, self._ledger._monotonic_clock) != _CLOCKS:
            raise AutomaticWorkRefused("workflow_native_clock_required")
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
        from tldw_chatbook.Agents.agent_service import AgentService
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner

        caller = inspect.currentframe().f_back
        if (
            caller.f_code is not AgentService._run_owned.__code__
            or caller.f_locals.get("self") is not service
            or caller.f_locals.get("workflow_task_authorization") is not task
            or current_execution_owner() is not owner
            or self._task_scopes.get(task.task.id) is not task
            or task._native_dispatch != [service]
        ):
            raise AutomaticWorkRefused("workflow_native_commit_required")
        task._check_sources()
        task._runtime.worktree_activity.validate_claim(
            activity, owner=owner, footprint=task.physical_footprint
        )
        with self._fence("child_commit") as fence:
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

    def _control(self, decision):
        """Atomically revoke new work; only original committed tasks may finish Pause."""
        from tldw_chatbook.Agents.workflow_models import WorkflowDecision
        from tldw_chatbook.Agents.workflow_task_scope import _runtime

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
        with self._fence("control") as fence:
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


async def _accept_remaining_workflow_phase(
    controller, decision, *, command_id: str, session_id: str, binding
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

    if type(decision) is not WorkflowDecision or decision.reason != "remaining_ready":
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
    cell = _RemainingAcceptance(controller, decision, _KEY)

    def current():
        if (
            controller._disposed
            or controller._shutdown_requested.is_set()
            or runtime._disposed
            or session_id in runtime._admission_fenced_sessions
            or controller.app.console_runtime is not runtime
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

    async def handoff():
        # This coroutine runs on the original host loop while the worker waits
        # with its original source, storage, operation and OPAQUE claim retained.
        current()
        if issuing.cancelling():
            return
        cell.host = True

    def read():
        owner = operation = claim = None
        try:
            with (
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
                            service.validate_remaining_decision(decision)
                        attempt = ledger._prepare_workflow(
                            decision.workflow_id,
                            revision_id=decision.plan_revision,
                            phase_id=decision.phase_id,
                            purpose="execution",
                            owner_id=owner_id,
                            session_id=session_id,
                            command_id=command_id,
                            _remaining=decision,
                        )
                        current()
                        cell.attempt = attempt
                        cell.phase = _accept_workflow_phase(
                            controller,
                            attempt.id,
                            session_id=session_id,
                            _remaining_read=cell,
                        )
                        asyncio.run_coroutine_threadsafe(handoff(), loop).result()
                        current()
                        if not binding.guard(False):
                            raise AutomaticWorkRefused("workflow_task_binding_changed")
                    except BaseException as caught:  # noqa: BLE001 - finish the original finite read before surfacing its outcome.
                        error = caught
                    operation.finish()
            cell.scopes_returned = True
            if error is not None:
                raise error
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
                require_current=current,
                source=cell,
            )
            current()
        except BaseException as caught:  # noqa: BLE001 - preserve cancellation through original physical retirement.
            error = caught
        # run_preparation_read has now observed actual native worker RETURN.
        # A failed source/finish unwind has no release proof, even on cancellation.
        if cell.scopes_returned:
            try:
                cell.owner.finish_root()
                cell.claim.settle(durable=True)
                cell.returned = True
                del cell.owner._workflow_remaining_acceptance
            except BaseException:
                cell.owner.mark_cleanup_unproven()
                raise
        if error is not None or not cell.host:
            if cell.returned and cell.phase is not None:
                await service.call(
                    service.settle_execution_phase, cell.phase, outcome="cancelled"
                )
                cell.phase = None
            if error is not None:
                raise error
        return cell.phase
