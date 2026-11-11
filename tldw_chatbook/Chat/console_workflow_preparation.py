"""Finite host custody for the Console's shared native reply producer (ADR-285)."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

from tldw_chatbook.Agents.activation import execution, owned_goal_storage
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Agents.run_context import bind_workflow_preparation
from tldw_chatbook.Agents.workflow_task_scope import (
    prepare_workflow_task,
    require_workflow_hooks_idle,
    require_workflow_runtime,
)

from .console_provider_endpoints import goal_provider_ref
from .console_preparation_reads import (
    current_planning_preparation as current_planning_preparation,
    current_workflow_preparation,
    run_preparation_read,
)

_PLANNING_INPUT = object()


@dataclass(frozen=True, eq=False)
class _PlanningRecoveryReturn:
    prepared: object = field(repr=False)
    phase: object = field(repr=False)
    read: object = field(repr=False)
    owner: object = field(repr=False)
    claim: object = field(repr=False)
    error: BaseException = field(repr=False)


@dataclass(eq=False)
class _PlanningPreparationRead:
    """Return evidence for one original finite callback, never admission."""

    prepared: object
    issuing: object
    callback: object = None
    proof: object = None
    owner: object = None
    claim: object = None
    operation: object = None
    drains: list = field(default_factory=list)
    scopes_returned: bool = False
    positive: bool = False
    body_error: object = None
    wrapper_error: object = None
    result: object = None
    pending: object = field(default=None, repr=False)
    completion: object = field(default=None, repr=False)
    completion_future: object = field(default=None, repr=False)
    native_read: object = field(default=None, repr=False)
    closing: bool = False
    before_issue: object = field(default=None, repr=False)
    configuration_return: object = field(default=None, repr=False)


def _planning_release_source(prepared, binding, *, scope_live=False, slots_live=False):
    """Validate one registered planning driver and its retained native resources."""
    if binding is not prepared._planning_release_binding or len(binding) != 22:
        raise PermissionError("planning release owner required")
    (
        original,
        phase,
        source,
        issuing,
        loop,
        _unused,
        coordinator,
        run,
        runtime,
        controller,
        store,
        session,
        wake,
        intake,
        token,
        wake_lock,
        wake_loop,
        scope,
        registry,
        received_lock,
        activity,
        acquisition,
    ) = binding
    if (
        original is not prepared
        or prepared.phase is not phase
        or phase.attempt.purpose != "planning"
        or prepared.source is not source
        or prepared._planning_driver is not issuing
        or asyncio.current_task() is not issuing
        or asyncio.get_running_loop() is not loop
        or prepared._loop is not loop
        or prepared.runtime is not runtime
        or prepared.controller is not controller
        or prepared.store is not store
        or prepared.session is not session
        or runtime.chat_controller is not controller
        or runtime._chat_store is not store
        or controller._hooks_v2_runtime is not runtime
        or controller.store is not store
        or controller._fleet_wake is not wake
        or wake._controller is not controller
        or wake._loop is not wake_loop
        or wake._registry_lock is not wake_lock
        or store._preparation_lock is not received_lock
        or runtime._workflow_preparations is not registry
        or runtime.worktree_activity is not activity
        or intake._store_ref() is not store
        or intake.session_id != session.id
        or acquisition is not prepared._planning_configuration_acquisition
        or any(
            a is not b
            for a, b in zip(acquisition[:3], (runtime, registry, scope), strict=True)
        )
    ):
        raise PermissionError("planning release resources changed")
    if run is not None and (
        runtime._workflow_coordinator is not coordinator
        or coordinator.runtime is not runtime
        or coordinator.controller is not controller
        or coordinator.service is not prepared.service
        or coordinator.loop is not loop
        or run.driver is not issuing
        or run.prepared is not prepared
        or run.phase is not phase
    ):
        raise PermissionError("planning coordinator owner changed")
    prepared._planning_configuration_source(
        acquisition, inspect.GEN_SUSPENDED if scope_live else inspect.GEN_CLOSED
    )
    if scope_live and (
        prepared._source_scope is not scope
        or prepared not in registry
        or store._sessions.get(session.id) is not session
    ):
        raise PermissionError("planning configuration changed")
    if slots_live:
        if (
            prepared._planning_intake is not intake
            or prepared._planning_token is not token
        ):
            raise PermissionError("planning admission slots changed")
        with received_lock:
            if store._preparations_by_session.get(session.id) is not intake:
                raise PermissionError("planning intake changed")
        with wake_lock:
            if (
                token is None
                or wake._automatic_primary_claims.get(session.id) is not token
            ):
                raise PermissionError("planning primary token changed")


def current_workflow_execution_local_tools(registry, allowed, local_provider):
    """Keep the original accepted task's captured local catalog in its request."""
    from tldw_chatbook.Agents.workflow_task_scope import WorkflowTaskAuthorization

    prepared = current_workflow_preparation()
    if (
        type(prepared) is not _PreparedWorkflowReply
        or prepared not in prepared.runtime._workflow_preparations
    ):
        return ()
    authorization = prepared.authorization
    if (
        type(authorization) is not WorkflowTaskAuthorization
        or authorization.registry is not registry
        or prepared.phase is not authorization.phase
        or prepared.phase._task_scopes.get(prepared.task_id) is not authorization
        or frozenset(allowed) != frozenset(authorization.task.allowed_tools)
    ):
        return ()
    for provider, (specs, _roots, _aliases) in prepared.local_providers.items():
        if provider is local_provider:
            prepared.borrow_preparation(prepared.runtime)
            prepared.check()
            return tuple(
                name for name, _spec in specs if authorization.permits_name(name)
            )
    return ()


class _PreparedWorkflowReply:
    """Original source and retained native frame, never restored from saved IDs."""

    def __init__(
        self,
        bridge,
        phase,
        task_id,
        binding,
        resolution,
        kwargs,
        *,
        controller=None,
        session=None,
        _key=None,
    ):
        if phase is not None and phase.attempt.purpose == "planning":
            raise PermissionError("planning requires the original pre-request producer")
        if phase is None and (
            _key is not _PLANNING_INPUT or controller is None or session is None
        ):
            raise PermissionError("original planning input producer required")
        self.bridge = bridge
        self.phase = phase
        self.task_id = task_id
        self.binding = binding
        self.resolution = resolution
        self._planning_resolution = None
        self._planning_system_prompt = None
        self._planning_composed_system = None
        self._planning_repair_source = None
        self.controller = phase._controller if phase is not None else controller
        self.runtime = (
            require_workflow_runtime(phase) if phase is not None else controller._hooks_v2_runtime
        )
        self.store = self.controller.store
        self.app = self.controller.app
        self._planning_gateway = self.bridge._gateway
        self.session = phase._session if phase is not None else session
        self.project_state = self.session.project_instruction_state
        self._planning_consent_callback = (
            self.controller._confirm_project_instruction_dispatch
        )
        self.service = self.runtime.ensure_workflow_service()
        self.capacity = self.runtime.execution_capacity
        self.kwargs = kwargs
        self.cancel_event = threading.Event()
        self.primary_owner = self.primary_claim = self.planning_service = None
        self._planning_dispatch_closed = False
        self._planning_physical_returned = False
        self._planning_settled = False
        self._planning_driver = None
        self._planning_terminal_returned = False
        self._planning_provider_failure = False
        self._planning_recovery_return = None
        self._planning_recovery_snapshot = None
        self._planning_read_return = None
        self._planning_configuration_acquisition = None
        self._planning_release_binding = None
        self._planning_release_return = None
        self._planning_reads = []
        self._finite_reads = []
        self._planning_attempt = None
        self._planning_prepare_entered = False
        self._planning_pending_run = None
        self._planning_preparation_returned = asyncio.Event()
        self._planning_token = self._planning_intake = None
        self.assistant_message_id = None
        self.generation_token = None
        self.should_cancel = (
            self._planning_cancelled
            if phase is None or phase.attempt.purpose == "planning"
            else kwargs.pop("should_cancel", self.controller._shutdown_requested.is_set)
        )
        if not callable(self.should_cancel):
            raise TypeError("native cancellation callback required")
        self.authorization = None
        self.native = None
        self._planning_native_frame = None
        self._planning_configuration_closed = False
        self.turn_lifeline = None
        self._lifetime = None
        self._closed = False
        self._closing = False
        self._snapshot = None
        self._loop = asyncio.get_running_loop()
        self._finite = None
        self._wave = None
        self.source = None
        self._configured_source_observation = None
        self.configuration = None
        self._source_scope = None
        self._terminal_configuration_return = None
        self._configuration_transition = None
        self._configuration_transition_error = None
        self._consent_configuration_return = None
        self._native_closed = False
        self._close_operation = None
        self._launch_operation = None
        self._tool_workers = []
        self._no_launch_close_guard = None
        self._configuration_selection = self._skill_preparation = (
            self._skill_catalog
        ) = None
        from tldw_chatbook import config

        self._config_identity = config.current_config_identity()
        self.local_providers = {}
        self._planning_builtins = ()
        self._planning_executors = {}
        self._planning_shell_source = self._planning_shell_schema = None
        self.require_current()

    def _planning_cancelled(self):
        return (
            self.cancel_event.is_set()
            or self.controller._shutdown_requested.is_set()
            or self._planning_driver is not None
            and self._planning_driver.cancelling() > 0
        )

    def _planning_return_source(self):
        """Validate retained originals for return, without reopening admission."""
        if (
            self.controller.store is not self.store
            or self.bridge._store is not self.store
            or self.runtime.agent_bridge is not self.bridge
            or self.runtime.chat_controller is not self.controller
            or self.controller._agent_bridge is not self.bridge
            or self.service is not self.runtime._workflow_service
            or self.store._sessions.get(self.session.id) is not self.session
            or self._planning_resolution is not None
            and self.resolution is not self._planning_resolution
            or self._planning_token is not None
            and self.controller._fleet_wake._automatic_primary_claims.get(
                self.session.id
            )
            is not self._planning_token
            or self._planning_intake is not None
            and self.store._preparations_by_session.get(self.session.id)
            is not self._planning_intake
        ):
            raise PermissionError("original planning custody changed")
        if self.phase is not None and self._planning_repair_source is not None:
            from hashlib import sha256

            command, workflow, revision, instruction = self._planning_repair_source
            attempt = self.phase.attempt
            if (
                attempt.command_id != command
                or workflow is not None
                and attempt.workflow_id != workflow
                or attempt.revision_id != revision
                or attempt.repair_instruction_sha256
                != (
                    None
                    if instruction is None
                    else sha256(instruction.encode("utf-8")).hexdigest()
                )
            ):
                raise PermissionError("original planning repair input changed")
        self._planning_native_sources()
        if self.source is not None:
            self.source.check_memory()

    def _planning_native_sources(self):
        """Validate the native gateway, prepared frame and service resources."""
        if (
            self.controller.provider_gateway is not self._planning_gateway
            or self.bridge._gateway is not self._planning_gateway
        ):
            raise PermissionError("planning provider changed")
        if self._planning_native_frame is not None:
            original, lifetime = self._planning_native_frame
            if (
                self.native is not original
                or self._lifetime is not lifetime
                or original.service.db is not self.service.db
                or original.service.registry is not original.first_request_plan.registry
                or original.turn_lifeline is not self.turn_lifeline
                or self.planning_service is not None
                and self.planning_service is not original.service
            ):
                raise PermissionError("planning native resources changed")

    def require_current(self):
        """Pure object/selection checks; native reads stay in the finite callback."""
        from tldw_chatbook import config

        if self.phase is None or self.phase.attempt.purpose == "planning":
            self._planning_return_source()
            if self._planning_provider_failure:
                raise PermissionError("planning provider return remains uncertain")
        if (
            config.current_config_identity() != self._config_identity
            or self.runtime.agent_bridge is not self.bridge
            or self.runtime.chat_controller is not self.controller
            or self.controller._agent_bridge is not self.bridge
            or self.controller.store is not self.store
            or self.bridge._store is not self.store
            or self.controller._disposed
            or self.controller._shutdown_requested.is_set()
            or self.should_cancel()
            or self.runtime._disposed
            or (self.phase is None or self.phase.attempt.purpose == "planning")
            and self.runtime._workflow_closing
            or not any(s is self.session for s in self.store.sessions())
            or self.session.project_instruction_state != self.project_state
            or self.session.workspace_id != self.binding.workspace_id
            or self.service is not getattr(self.runtime, "_workflow_service", None)
        ):
            raise PermissionError("workflow native source changed")

    def check_planning_executor(self, executor=None):
        """Keep planning readers on the selected local binding and exclusions."""
        from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor
        from tldw_chatbook.Utils.filesystem_identity import DirectoryChain

        if executor is None:
            return
        if not isinstance(executor, WorkspaceToolExecutor):
            raise PermissionError("planning workspace executor required")
        values = (
            executor._remote_root,
            executor._workspace_root,
            executor._authority_chain,
            executor._user_exclusion_paths,
        )
        if (
            values[0] is not None
            or values[1] != self.binding.root
            or not isinstance(values[2], DirectoryChain)
            or values[2].canonical_root != self.binding.root
            or values[3] is not self.binding.exclusions_provider
        ):
            raise PermissionError("planning executor binding changed")
        retained = self._planning_executors.setdefault(executor, values)
        if any(a is not b for a, b in zip(retained, values, strict=True)):
            raise PermissionError("planning executor fields changed")

    def check_local_sources(self):
        from tldw_chatbook.Agents.ordinary_tree import check_ordinary_local_provider_sources

        planning = self.phase is None or self.phase.attempt.purpose == "planning"
        if planning:
            for executor in self._planning_executors:
                self.check_planning_executor(executor)
        for provider, retained in self.local_providers.items():
            check_ordinary_local_provider_sources(provider, retained)
        if self._planning_builtins:
            from tldw_chatbook.Agents.tool_catalog import (
                planning_builtin_selection_changed,
            )

            for provider, tools in self._planning_builtins:
                if planning_builtin_selection_changed(provider, tools):
                    raise PermissionError("planning builtin selection changed")

    @contextmanager
    def activate_source(self):
        with (
            bind_workflow_preparation(self),
            self.source.activate() if self.source is not None else nullcontext(),
        ):
            yield

    def borrow_preparation(self, runtime):
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner

        if (
            current_workflow_preparation() is not self
            or runtime is not self.runtime
            or self._finite is None
            and self.primary_owner is None
        ):
            raise PermissionError("original finite workflow preparation required")
        owner, claim = self._finite or (self.primary_owner, self.primary_claim)
        if current_execution_owner() is not owner:
            raise PermissionError("original finite workflow owner required")
        self.runtime.worktree_activity.validate_opaque_claim(claim, owner=owner)
        return owner

    def accepts_skill_context(self, prepared, selection):
        self.borrow_preparation(self.runtime)
        return (
            selection is self._configuration_selection
            and prepared.catalog is self._skill_catalog
            and prepared.catalog.preparation is self._skill_preparation
            and prepared.catalog.turn_id == self.phase.attempt.id
        )

    def _planning_configuration_source(self, acquisition, state):
        runtime, registry, scope, _unused = acquisition
        if (
            self.runtime is not runtime
            or self.controller._hooks_v2_runtime is not runtime
            or runtime._workflow_preparations is not registry
            or not isinstance(registry, set)
            or scope is not None
            and acquisition is not self._planning_configuration_acquisition
        ):
            raise PermissionError("planning configuration owner changed")
        if state == inspect.GEN_CLOSED:
            if (
                not self._planning_configuration_closed
                or self._source_scope is not None
            ):
                raise RuntimeError("planning configuration return unproven")
        elif scope is not None and (
            self._source_scope is not scope or self not in registry
        ):
            raise PermissionError("planning configuration scope changed")

    def _retain_configuration(self):
        if self._configuration_transition is not None:
            raise RuntimeError("configuration entry unproven")
        if self._source_scope is not None:
            return
        planning = self.phase is None or self.phase.attempt.purpose == "planning"
        if planning and self._planning_configuration_acquisition is not None:
            raise PermissionError("planning configuration already acquired")
        registry = self.runtime._workflow_preparations
        scope = self.runtime._first_send_source_scope()
        self._consent_configuration_return = None
        self._configuration_transition = self, scope, registry
        try:
            scope.__enter__()
            self._source_scope = scope
            registry.add(self)
            if planning:
                self._planning_configuration_acquisition = (
                    self.runtime,
                    registry,
                    scope,
                    None,
                )
                self._planning_configuration_closed = False
        except BaseException as error:
            self._configuration_transition_error = error
            registry.add(self)
            raise
        self._configuration_transition = None

    def _release_configuration(self):
        if self._configuration_transition is not None:
            raise RuntimeError("configuration release unproven")
        scope = self._source_scope
        if scope is None:
            return None
        registry = self.runtime._workflow_preparations
        planning = self.phase is None or self.phase.attempt.purpose == "planning"
        acquisition = self._planning_configuration_acquisition
        if (
            self not in registry
            or planning
            and (acquisition is None or acquisition[2] is not scope)
        ):
            raise PermissionError("configuration scope changed")
        returned = self, scope, registry
        self._configuration_transition = returned
        try:
            if scope.__exit__(None, None, None):
                raise RuntimeError("configuration exit unproven")
            registry.remove(self)
            self._source_scope = None
        except BaseException as error:
            self._configuration_transition_error = error
            raise
        if planning:
            self._planning_configuration_closed = True
            self._configuration_transition = None
            return *returned, acquisition
        return returned

    def _configuration_sources(self, scope=None):
        """Check the retained runtime and configuration registry."""
        self._execution_reads_source()
        if (
            self.controller._agent_bridge is not self.bridge
            or self.runtime.chat_controller is not self.controller
            or not isinstance(self.runtime._workflow_preparations, set)
        ):
            raise PermissionError("workflow configuration resources changed")

    def _configuration_received(self, returned):
        """Consume this preparation's completed configuration release once."""
        self._configuration_sources()
        if (
            returned is not self._configuration_transition
            or not isinstance(returned, tuple)
            or len(returned) != 3
            or returned[0] is not self
            or returned[1] is None
            or returned[2] is not self.runtime._workflow_preparations
            or self._configuration_transition_error is not None
            or self._source_scope is not None
            or self in returned[2]
        ):
            raise RuntimeError("workflow configuration return unproven")
        self._configuration_transition = None
        return returned

    def _consent_configuration_source(self, record):
        """Bind a None-scope read to its original, positively received release."""
        self._configuration_sources()
        saved = self._consent_configuration_return
        if type(saved) is not tuple or len(saved) != 11:
            raise RuntimeError("original consent configuration return unproven")
        (
            returned,
            phase,
            auth,
            source,
            native,
            lifetime,
            snapshot,
            prefix,
            issuer,
            _callback,
            observation,
        ) = saved
        if (
            record.configuration_return is not saved
            or type(returned) is not tuple
            or len(returned) != 3
            or returned[0] is not self
            or returned[1] is None
            or returned[2] is not self.runtime._workflow_preparations
            or self._source_scope is not None
            or self in returned[2]
            or self._configuration_transition is not None
            or self._configuration_transition_error is not None
            or phase is not self.phase
            or auth is not self.authorization
            or auth is None
            or source is not self.source
            or native is not self.native
            or lifetime is not self._lifetime
            or snapshot is not self._snapshot
            or observation is not self._configured_source_observation
            or type(prefix) is not tuple
            or len(self._finite_reads) <= len(prefix)
            or any(a is not b for a, b in zip(prefix, self._finite_reads, strict=False))
            or self._finite_reads[len(prefix)] is not record
            or record.prepared is not self
            or record.issuing is not issuer
        ):
            raise RuntimeError("original consent configuration return unproven")
        with auth._lock:
            if (
                phase._task_scopes.get(self.task_id) is not auth
                or not auth._bridge_preparation
                or auth._bridge_preparation[0] is not self
                or auth.phase is not phase
            ):
                raise RuntimeError("original consent preparation changed")
        self._configuration_sources(returned[1])
        return returned

    def _execution_configuration_return(self):
        """Prove this preparation's own terminal read and configuration return."""
        self._configuration_sources()
        terminal = self._terminal_configuration_return
        if type(terminal) is not tuple or len(terminal) != 3:
            raise RuntimeError("original configuration return unproven")
        returned, closing_read, records = terminal
        if (
            type(returned) is not tuple
            or len(returned) != 3
            or returned[0] is not self
            or returned[1] is None
            or returned[2] is not self.runtime._workflow_preparations
            or self in returned[2]
            or self._source_scope is not None
            or self._configuration_transition is not None
            or self._configuration_transition_error is not None
            or not self._closed
            or self._closing
            or self._finite is not None
            or self._launch_operation is not None
            or self._close_operation is not None
            or type(records) is not tuple
            or not records
            or len(records) != len(self._finite_reads)
            or any(a is not b for a, b in zip(records, self._finite_reads, strict=True))
            or records[-1] is not closing_read
            or not closing_read.positive
            or closing_read.wrapper_error is not None
            or closing_read.body_error is not None
            or not closing_read.closing
        ):
            raise RuntimeError("original configuration return unproven")
        self._configuration_sources(returned[1])
        self._finite_read_return(closing_read)
        return terminal

    def _execution_terminal_return(self):
        self._execution_configuration_return()
        if self._wave is not None:
            from .console_workflow_wave import _workflow_wave_release_return

            _workflow_wave_release_return(self._wave, self)

    def check(self):
        self.require_current()
        self.phase.check()
        require_workflow_hooks_idle(self.runtime, self.session.id)
        if not self.binding.guard(False):
            raise PermissionError("workflow native binding changed")
        if self.authorization is not None:
            self.authorization.check_preparation()
            if self.phase.attempt.purpose == "planning":
                if self.resolution is not self._planning_resolution:
                    raise PermissionError("workflow native target changed")
            elif goal_provider_ref(self.resolution) != self.authorization.task.target:
                raise PermissionError("workflow native target changed")

    def _check_task_source(self, authorization):
        if self._closed or self._closing or authorization is not self.authorization:
            raise PermissionError("workflow native preparation retired")
        self.require_current()
        self.check_local_sources()
        if goal_provider_ref(self.resolution) != authorization.task.target:
            raise PermissionError("workflow native target changed")

    @contextmanager
    def _unpublished_storage_return(self, publication):
        """Prove this original unused frame returned its worker-owned handles."""
        from tldw_chatbook.DB.workflow_runs import _WindowUnpublished

        wave, source = self._wave, self.source
        if (
            type(publication) is not _WindowUnpublished
            or wave is None
            or source is None
            or publication.store is not self.service.db.workflow_runs
            or publication.phase is not self.phase
            or publication.reservation is not wave.reservation
            or wave.phase is not self.phase
            or wave.runtime is not self.runtime
            or self.runtime._workflow_waves.get(wave) is not wave
            or self not in self.runtime._workflow_preparations
            or not any(item is self for item in wave.prepared)
            or source.workflow_preparation is not self
            or self.service.db is not source.databases[0]
            or self.service.persistence is not source.persistence
            or getattr(self.service.persistence, "workspace_registry", None)
            is not source.registry
        ):
            raise PermissionError("original unpublished frame storage required")
        originals = []
        for database in dict.fromkeys(db for db in source.databases if db is not None):
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
            # Check after BOTH the native frame/source and the outer storage
            # scopes. Native close may suppress an error while retaining a live
            # cache; frame flags or a later close cannot certify this return.
            if not all(
                vars(database).get(slot) is local
                and getattr(local, "conn", None) is original
                for database, slot, local, original in originals
            ):
                raise RuntimeError("original source database handle did not return")

    async def _read(self, callback, *, roots=True, closing=False, _unpublished=None):
        """Own a complete finite callback; no native scope survives its return."""
        if self.phase is None or self.phase.attempt.purpose == "planning":
            if _unpublished is not None:
                raise PermissionError("execution-only unpublished proof")
            return await self._read_planning(callback, roots=roots, closing=closing)
        if _unpublished is not None and not closing:
            raise PermissionError("unpublished storage proof is close-only")
        self._execution_reads_returned()
        from .console_workflow_runs import _capture_workflow_read

        record = _PlanningPreparationRead(self, asyncio.current_task(), closing=closing)
        if self._source_scope is None:
            record.configuration_return = self._consent_configuration_return
        self._finite_reads.append(record)
        issuing = record.issuing

        def invoke_owned():
            result = failure = owner = claim = operation = None
            with self.phase.context.scope():
                try:
                    self.check()
                except BaseException as error:  # noqa: BLE001 - validation refusal still returns native resources.
                    record.body_error = error
                    return None, error, None, None, None
                # shortcut: finite preflight serializes against all root writes;
                # batch eligible preparations before launching disjoint tasks.
                from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

                footprint = WorktreeFootprint(opaque=True) if roots else None
                activity = self.runtime.worktree_activity
                with (
                    activity.prelaunch(footprint=footprint)
                    if roots
                    else nullcontext() as claim
                ):
                    if roots and claim is None:
                        from tldw_chatbook.Agents.agent_service import (
                            WorkflowChildWait,
                        )

                        failure = record.body_error = WorkflowChildWait(
                            "worktree_activity", "workflow preparation busy"
                        )
                    else:
                        owner = self.capacity.begin_execution(
                            origin=WorkOrigin.AUTOMATIC,
                            conversation_id=self.phase.attempt.conversation_id,
                        )
                        record.owner, record.claim = owner, claim
                        self._finite = (owner, claim)
                        if claim is not None:
                            claim.bind_owner(owner)
                        operation = owner.reserve_tool()
                        record.operation = operation
                        with owner.activate(), self.activate_source():
                            try:
                                self.check()
                                self.check_local_sources()
                                result = callback()
                            except BaseException as error:  # noqa: BLE001 - preserve the original callback outcome; scope exits are outside.
                                failure = record.body_error = error
                            if failure is None and inspect.isawaitable(result):
                                from tldw_chatbook.Agents.activation import (
                                    async_worker_guard,
                                )

                                pending = record.pending = result

                                @async_worker_guard(self.bridge)
                                async def complete():
                                    with (
                                        owned_goal_storage(self.service),
                                        self.activate_source(),
                                    ):
                                        value = body_error = None
                                        try:
                                            value = await pending
                                        except BaseException as error:  # noqa: BLE001 - preserve the original callback outcome; scope exits are outside.
                                            body_error = record.body_error = error
                                        return pending, value, body_error

                                record.completion = complete()
                                record.completion_future = (
                                    asyncio.run_coroutine_threadsafe(
                                        record.completion, self._loop
                                    )
                                )
                                outcome = record.completion_future.result()
                                if (
                                    type(outcome) is not tuple
                                    or len(outcome) != 3
                                    or outcome[0] is not pending
                                    or outcome[2] is not record.body_error
                                ):
                                    raise RuntimeError(
                                        "original finite async result changed"
                                    )
                                _, result, failure = outcome
                            if failure is None:
                                try:
                                    self.check()
                                except BaseException as error:  # noqa: BLE001 - preserve refusal after native callback return.
                                    failure = record.body_error = error
            return result, failure, owner, claim, operation

        def invoke():
            proof = record.proof = _capture_workflow_read(
                self.controller, self, issuing
            )
            result = failure = owner = claim = operation = None
            with (
                self._unpublished_storage_return(_unpublished)
                if _unpublished is not None
                else nullcontext(),
                owned_goal_storage(self.service),
                execution(self.bridge),
            ):
                if closing:
                    try:
                        result = callback()
                    except BaseException as error:  # noqa: BLE001 - preserve the original callback outcome; scope exits are outside.
                        failure = record.body_error = error
                else:
                    result, failure, owner, claim, operation = invoke_owned()
            # No owning scope is crossed by a callback exception. An exit or
            # finish failure retains this read and is never retried as cleanup.
            if (
                record.owner is not owner
                or record.claim is not claim
                or record.operation is not operation
                or record.body_error is not failure
            ):
                raise RuntimeError("original workflow finite outcome changed")
            if operation is not None:
                operation.finish()
            if owner is not None:
                owner.on_drained(record.drains.append)
                owner.finish_root()
                if record.drains != [True]:
                    raise RuntimeError("workflow preparation retirement unproven")
            if claim is not None:
                claim.settle(durable=True)
            self._finite = None
            record.scopes_returned = True
            record.result = result
            return (
                record,
                self,
                issuing,
                proof[0],
                proof,
                owner,
                claim,
                operation,
                result,
                failure,
            )

        record.callback = invoke
        try:
            # Validate the original receiver before selecting its bound validator.
            self._execution_reads_source()
            issuing_thread = threading.current_thread()

            def read_current():
                from .console_preparation_reads import preparation_read_for

                native = preparation_read_for(self.controller)
                if native is not None:
                    record.native_read = native
                try:
                    self._execution_reads_source()
                    if not closing:
                        self.require_current()
                except BaseException as error:
                    if (
                        native is None
                        and threading.current_thread() is issuing_thread
                        and asyncio.current_task() is issuing
                    ):
                        record.before_issue = error
                    raise

            await run_preparation_read(
                invoke,
                creator=self.controller,
                session_id=self.session.id,
                reads=self.controller._preparation_reads,
                observers=(self.runtime._preparation_reads,),
                require_current=read_current,
                source=self,
            )
        except BaseException as error:
            record.wrapper_error = error
            raise
        self._execution_return_source()
        self._finite_read_return(record)
        record.positive = True
        if record.body_error is not None:
            if isinstance(record.body_error, StopIteration):
                raise RuntimeError(
                    "coroutine raised StopIteration"
                ) from record.body_error
            raise record.body_error
        return record.result

    def _execution_reads_source(self):
        """Pure original source/reader checks, without renewing admission."""
        _storage_source_current(self.source, preparation=self)

    def _execution_no_issue_return(self, record):
        """Recognize a refused reader before any callback resources were acquired."""
        self._execution_reads_source()
        if (
            record.prepared is not self
            or not any(read is record for read in self._finite_reads)
            or record.wrapper_error is None
            or record.proof is not None
            or record.owner is not None
            or record.claim is not None
            or record.operation is not None
            or record.scopes_returned
            or record.positive
            or record.drains
            or record.body_error is not None
            or record.result is not None
            or self._finite is not None
        ):
            raise RuntimeError("workflow read refusal unproven")
        read = record.native_read
        if read is None:
            if record.before_issue is not record.wrapper_error:
                raise RuntimeError("workflow reader admission unproven")
            return
        if (
            read.creator is not self.controller
            or read.source is not self
            or read.task is not record.issuing
            or read.callback_returned
            or not read.retired.done()
            or read.retired.cancelled()
            or read._producer is None
            or not read._producer.done()
            or read._producer.cancelled()
            or read._producer.exception() is None
        ):
            raise RuntimeError("workflow reader return unproven")

    def _execution_reads_returned(self):
        """Keep unknown original finite reads from shedding their source custody."""
        self._execution_reads_source()
        if (
            self._configuration_transition is not None
            or self._configuration_transition_error is not None
        ):
            raise RuntimeError(
                "original workflow read issue unproven"
            ) from self._configuration_transition_error
        for record in self._finite_reads:
            if record.proof is None:
                self._execution_no_issue_return(record)
            else:
                self._finite_read_return(record)
        if self._finite is not None:
            raise RuntimeError("original workflow finite return unproven")

    def _finite_read_return(self, record):
        """Consume identities carried by the same native producer's result."""
        from .console_workflow_runs import _prove_workflow_read

        _prove_workflow_read(record.proof, self.controller, self, record.issuing)
        actual = record.proof[0].result
        expected = (
            record,
            self,
            record.issuing,
            record.proof[0],
            record.proof,
            record.owner,
            record.claim,
            record.operation,
            record.result,
            record.body_error,
        )
        if (
            type(actual) is not tuple
            or len(actual) != len(expected)
            or any(
                first is not second
                for first, second in zip(actual, expected, strict=True)
            )
            or not record.scopes_returned
        ):
            raise RuntimeError("original workflow finite result changed")
        owner, claim, operation = actual[5:8]
        if owner is not None:
            _finite_owner_return(owner, self.capacity, operation)
            if record.drains != [True]:
                raise RuntimeError("original workflow finite drain unproven")
        if claim is not None:
            activity = self.runtime.worktree_activity
            with activity._lock:
                if claim.id in activity._claims or claim.id in activity._members:
                    raise RuntimeError("original workflow finite claim unreturned")
        return actual

    def _execution_return_source(self):
        """Inspect retained execution sources without renewing a revoked phase."""

        if (
            self.controller.store is not self.store
            or self.runtime.agent_bridge is not self.bridge
            or self.runtime.chat_controller is not self.controller
            or self.controller._agent_bridge is not self.bridge
            or self.bridge._store is not self.store
            or self.service is not self.runtime._workflow_service
            or self.store._sessions.get(self.session.id) is not self.session
        ):
            raise PermissionError("original workflow finite source changed")
        self.check_local_sources()
        if self.source is not None:
            self._configured_storage_sources()

    def _configured_storage_sources(self):
        """Return only the observation issued by this original configuration."""
        _storage_source_current(self.source)
        observation = self._configured_source_observation
        if (
            type(observation) is not tuple
            or len(observation) != 3
            or observation[0] is not self
            or observation[1] is not self.source
            or observation[2] is not self.source.sources
            or self.source.workflow_preparation is not self
        ):
            raise PermissionError("original workflow storage observation changed")
        self.source.check_memory()
        return observation

    async def _read_planning(self, callback, *, roots=True, closing=False):
        """The existing tracked reader owns planning's finite native return."""
        from tldw_chatbook.Agents.activation import async_worker_guard
        from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

        from .console_workflow_runs import _capture_workflow_read

        record = _PlanningPreparationRead(self, asyncio.current_task(), closing=closing)
        self._planning_reads.append(record)
        issuing = record.issuing

        def invoke():
            from tldw_chatbook.Agents.execution_capacity import CapacityRefused

            proof = record.proof = _capture_workflow_read(
                self.controller, self, issuing
            )
            failure = result = None
            owner = operation = None
            with (
                self._planning_storage_return(),
                owned_goal_storage(self.service),
                execution(self.bridge),
                self.runtime.worktree_activity.prelaunch(
                    footprint=WorktreeFootprint(opaque=True)
                )
                if roots
                else nullcontext() as claim,
            ):
                if roots and claim is None:
                    raise PermissionError("planning preparation busy")
                owner = self.capacity.begin_execution(
                    origin=WorkOrigin.AUTOMATIC,
                    conversation_id=self.session.persisted_conversation_id
                    or self.session.id,
                )
                record.owner, record.claim = owner, claim
                if claim is not None:
                    claim.bind_owner(owner)
                self._finite = (owner, claim)
                try:
                    operation = owner.reserve_tool()
                    record.operation = operation
                except CapacityRefused as error:
                    failure = error
                if operation is not None:
                    with (
                        owner.activate(),
                        self.phase.context.scope()
                        if self.phase is not None
                        else nullcontext(),
                        self.activate_source(),
                    ):
                        try:
                            if not closing:
                                self.check() if self.phase is not None else self.require_current()
                            result = callback()
                        except BaseException as error:  # noqa: BLE001 - preserve the original callback outcome; scope exits are outside.
                            failure = record.body_error = error
                        if failure is None and inspect.isawaitable(result):
                            pending = record.pending = result

                            @async_worker_guard(self.bridge)
                            async def complete():
                                with (
                                    self._planning_storage_return(),
                                    owned_goal_storage(self.service),
                                    self.activate_source(),
                                ):
                                    value = body_error = None
                                    try:
                                        value = await pending
                                    except BaseException as error:  # noqa: BLE001 - preserve the original callback outcome; scope exits are outside.
                                        body_error = record.body_error = error
                                    return pending, value, body_error

                            record.completion = complete()
                            record.completion_future = asyncio.run_coroutine_threadsafe(
                                record.completion, self._loop
                            )
                            outcome = record.completion_future.result()
                            if (
                                type(outcome) is not tuple
                                or len(outcome) != 3
                                or outcome[0] is not pending
                                or outcome[2] is not record.body_error
                            ):
                                raise RuntimeError(
                                    "original planning async result changed"
                                )
                            _, result, failure = outcome
            # Only positively returned original scopes reach their held finish.
            # An unknown entry/unwind keeps _finite, its source, claim and slot.
            if operation is not None:
                operation.finish()
            owner.on_drained(record.drains.append)
            owner.finish_root()
            if record.drains != [True]:
                raise RuntimeError("planning preparation retirement unproven")
            if claim is not None:
                claim.settle(durable=True)
            self._finite = None
            record.scopes_returned = True
            record.body_error, record.result = failure, result
            return (
                record,
                self,
                issuing,
                proof[0],
                proof,
                owner,
                claim,
                operation,
                result,
                failure,
            )

        record.callback = invoke
        try:
            await run_preparation_read(
                invoke,
                creator=self.controller,
                session_id=self.session.id,
                reads=self.controller._preparation_reads,
                observers=(self.runtime._preparation_reads,),
                require_current=self._planning_return_source,
                source=self,
            )
        except BaseException as error:  # noqa: BLE001 - preserve caller and original producer outcomes separately.
            record.wrapper_error = error
        self._planning_return_source()
        self._finite_read_return(record)
        if (
            record.prepared is not self
            or not record.scopes_returned
            or record.drains != [True]
            or record.owner is None
            or record.owner.is_admitted
            or record.wrapper_error is not None
            and not isinstance(record.wrapper_error, asyncio.CancelledError)
        ):
            raise RuntimeError("original planning preparation return unproven")
        record.positive = True
        if record.wrapper_error is not None:
            raise record.wrapper_error
        if record.body_error is not None:
            raise record.body_error
        if not closing:
            self.require_current()
        return record.result

    def _planning_abort_source(self, ledger, attempt):
        """Validate the original unused-close callback, not a supplied receipt."""
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner

        from .console_workflow_runs import _capture_workflow_read

        self._planning_return_source()
        record = self._planning_reads[-1] if self._planning_reads else None
        if (
            ledger is not self.service.db.automatic_work
            or attempt is not self._planning_attempt
            or not self._planning_prepare_entered
            or self.phase is not None
            or current_workflow_preparation() is not self
            or record is None
            or record.prepared is not self
            or record.owner is not current_execution_owner()
            or record.positive
            or self._finite != (record.owner, record.claim)
        ):
            raise PermissionError("original prepared planning abort required")
        actual = _capture_workflow_read(self.controller, self, record.issuing)
        if any(a is not b for a, b in zip(record.proof, actual, strict=True)):
            raise PermissionError("original planning abort reader changed")
        for previous in self._planning_reads[:-1]:
            self._finite_read_return(previous)
            if not previous.positive or not previous.scopes_returned:
                raise PermissionError("original planning preparation return required")
        self.runtime.worktree_activity.validate_opaque_claim(
            record.claim, owner=record.owner
        )

    def _planning_reads_returned(self):
        """Re-prove each exact issued finite reader before unused release."""

        self._planning_return_source()
        run = self._planning_pending_run
        if run is not None and (
            run.prepared is not self or run.planning_reads is not self._planning_reads
        ):
            raise PermissionError("original pending planning readers changed")
        for record in self._planning_reads:
            self._finite_read_return(record)
            if (
                type(record) is not _PlanningPreparationRead
                or record.prepared is not self
                or not record.positive
                or not record.scopes_returned
                or record.drains != [True]
                or record.owner is None
                or record.owner.is_admitted
                or record.claim is not None
                and record.claim.id in self.runtime.worktree_activity._claims
                or run is not None
                and record.issuing is not run.issuing
            ):
                raise RuntimeError("original planning reader return unproven")
        if self._planning_prepare_entered and self._planning_attempt is None:
            raise RuntimeError("original planning preparation receipt missing")

    def bind_catalog(self, catalog):
        """Consume the one real composition before building the task-only plan."""
        registry, allowed, builtin, local, skills = catalog
        if self.phase.attempt.purpose == "planning":
            from tldw_chatbook.Agents import tool_catalog
            from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
            from tldw_chatbook.Agents.workflow_task_scope import (
                prepare_planning_authorization,
            )

            from tldw_chatbook.Agents.ordinary_tree import capture_ordinary_local_provider_sources

            if skills:
                raise PermissionError("original bounded planning catalog required")
            names = tuple(
                name
                for name in allowed
                if name in {"fs_list", "fs_read", "calculator", "get_current_datetime"}
            )
            providers = {}
            builtins = set()
            for name in names:
                _tool_id, provider = registry.resolve_owner_for_name(name)
                if name in {"fs_list", "fs_read"}:
                    if type(
                        provider
                    ) is not LocalToolProvider or provider._native_ordinary_specs.get(
                        name
                    ) is not provider._specs.get(name):
                        raise PermissionError("original bounded planning tool required")
                    providers.setdefault(provider, []).append(name)
                elif not tool_catalog.is_native_builtin_provider(provider):
                    raise PermissionError("original pure planning tool required")
                else:
                    builtins.add(provider)
            self.local_providers = {
                p: capture_ordinary_local_provider_sources(p, n) for p, n in providers.items()
            }
            self._planning_builtins = tuple(
                (provider, tuple(provider._tools.items())) for provider in builtins
            )
            self.check_local_sources()
            self.authorization = prepare_planning_authorization(self, registry, names)
            registry._planning_future_shell = self._capture_planning_shell_metadata()
            messages = self.service.prepare_planning_inputs(self, catalog)
            self.check()
            return list(messages), (
                registry,
                names,
                tuple(n for n in builtin if n in names),
                tuple(n for n in local if n in names),
                frozenset(),
            )
        self.authorization = prepare_workflow_task(
            self.phase,
            task_id=self.task_id,
            registry=registry,
            binding=self.binding,
        )
        self.authorization._bridge_preparation.append(self)
        if not self.authorization.physical_footprint.opaque:
            from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

            from tldw_chatbook.Agents.ordinary_tree import capture_ordinary_local_provider_sources

            providers = {}

            self.check_local_sources()
            for name in self.authorization.task.allowed_tools:
                resolved = registry.resolve_owner_for_name(name)
                if resolved is None:
                    raise PermissionError("workflow native tool unavailable")
                provider = resolved[1]
                if type(provider) is LocalToolProvider:
                    spec = provider._specs.get(name)
                    if provider._native_ordinary_specs.get(name) is not spec:
                        raise PermissionError(
                            "workflow bounded native tool source required"
                        )
                    providers.setdefault(provider, []).append(name)
            self.local_providers = {
                provider: capture_ordinary_local_provider_sources(provider, names)
                for provider, names in providers.items()
            }
        self.check()
        if not set(self.authorization.task.allowed_tools).issubset(allowed):
            raise PermissionError("workflow native tools changed")
        messages = self.service.prepare_task_inputs(self.authorization)
        return list(messages), (
            registry,
            tuple(
                name
                for name in allowed
                if name in self.authorization.task.allowed_tools
            ),
            builtin,
            local,
            skills,
        )

    async def _configure(self):
        from dataclasses import replace

        from tldw_chatbook.Agents.builtin_tool_gate import build_builtin_gate
        from tldw_chatbook.Agents.persona_policy import parse_persona_policy_from_rules
        from tldw_chatbook.MCP.console_snapshot import (
            _capture_console_definition_selection,
        )

        from tldw_chatbook.Agents.native_source_storage import (
            NativeAgentSources as _AgentSources,
        )
        from .console_configuration_capture import (
            CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        )
        from .console_configuration_preparation import (
            ConsoleTurnCaptureSelection,
            capture_console_received_configuration_preparation,
            capture_console_skill_catalog_owned,
            capture_runtime_configuration_owned,
            consume_native_mcp_permission_publication,
            finish_console_skill_catalog_owned,
            original_mcp_capture_matches,
        )
        from .console_turn_context import capture_change_review_admission

        selection = ConsoleTurnCaptureSelection(
            provider_selection=self.controller._provider_selection_for_session(
                self.session.id
            ),
            presentation_context=self.controller._presentation_context_for(
                self.session.id
            ),
            rag_defaults=None,
            tool_configuration=None,
            skill_workspace_id=self.store.session_workspace_id(self.session.id),
            project_bindings_eligible=True,
            agent_runtime_enabled=self.controller._agent_runtime_enabled,
        )
        metadata = capture_console_received_configuration_preparation(
            self.controller.app, self.store, self.controller, session_id=self.session.id
        )
        if metadata is None:
            raise PermissionError("original workflow configuration source unavailable")
        self._configuration_selection = selection
        self._skill_preparation = metadata
        skills = None
        if (
            self.phase.attempt.purpose != "planning"
            and metadata.trust_source is not None
        ):
            catalog = await capture_console_skill_catalog_owned(
                self.controller.app,
                self.store,
                self.controller,
                session_id=self.session.id,
                turn_id=self.phase.attempt.id,
                skill_workspace_id=selection.skill_workspace_id,
                preparation=metadata,
                reads=self.controller._preparation_reads,
                observers=(self.runtime._preparation_reads,),
                require_current=self.require_current,
            )
            self._skill_catalog = catalog
            skills = await finish_console_skill_catalog_owned(
                catalog,
                reads=self.controller._preparation_reads,
                observers=(self.runtime._preparation_reads,),
                require_current=self.require_current,
            )
        configuration = await capture_runtime_configuration_owned(
            self.controller,
            self.session.id,
            require_current=self.require_current,
            selection=selection,
            prepared_skills=skills,
        )
        self.configuration = configuration
        _storage_source_current()
        self.source = _AgentSources(
            self.controller, self.runtime, self.bridge, self.session, configuration
        )
        self.source.workflow_preparation = self
        _storage_source_current(self.source)
        self.source.prepare()
        _storage_source_current(self.source)
        if self._configured_source_observation is not None:
            raise PermissionError(
                "original workflow storage observation already issued"
            )
        self._configured_source_observation = (self, self.source, self.source.sources)
        with self.activate_source():
            if self.phase.attempt.purpose == "planning" and self.source.mcp is not None:
                from tldw_chatbook.MCP.console_snapshot import _CapturedSources

                permission_source = _CapturedSources(self.source.mcp)
                self._planning_permission_payload = await run_preparation_read(
                    permission_source.read_permission_payload,
                    creator=self.controller,
                    session_id=self.session.id,
                    reads=self.controller._preparation_reads,
                    observers=(self.runtime._preparation_reads,),
                    require_current=self.require_current,
                    source=self,
                )
                consume_native_mcp_permission_publication(self.source)
            if (
                self.phase.attempt.purpose != "planning"
                and configuration.mcp_capture_pending
            ):
                if not original_mcp_capture_matches(
                    self.controller, self.session, configuration, pending=True
                ):
                    raise PermissionError("original workflow MCP metadata required")
                selected = await _capture_console_definition_selection(
                    self.source.mcp,
                    CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
                    profile_id=configuration.tool_policy_profile_id,
                    persona_policy_rules=configuration.persona_policy_rules,
                    _admitted=True,
                )
                consume_native_mcp_permission_publication(self.source)
                if not original_mcp_capture_matches(
                    self.controller, self.session, configuration, pending=True
                ):
                    raise PermissionError("original workflow MCP metadata changed")
                configuration = replace(
                    configuration,
                    mcp_tool_maximum=frozenset(selected.definitions),
                    mcp_definition_maximum=selected.definitions,
                    mcp_definition_sources=selected.sources,
                    mcp_definition_complete=selected.complete,
                    mcp_capture_pending=False,
                    mcp_definition_capture="captured",
                )
            roots, aliases, skipped = capture_change_review_admission(
                self.controller.app, self.session.workspace_id
            )
            configuration = replace(
                configuration,
                workspace_roots=roots,
                change_review_root_aliases=aliases,
                change_review_skipped_roots=skipped,
                change_review_capture_pending=False,
            )
            self.configuration = self.source.configuration = configuration
            policy = parse_persona_policy_from_rules(configuration.persona_policy_rules)
            mcp = (
                None
                if self.phase.attempt.purpose == "planning"
                else await self.controller._compose_mcp_provider(
                    self.session.id,
                    publish_counts=False,
                    profile_id_provider=lambda: configuration.tool_policy_profile_id,
                    persona_policy_provider=lambda: policy,
                    maximum_tool_ids=configuration.mcp_tool_maximum,
                    maximum_definition_hashes=configuration.mcp_definition_maximum,
                )
            )
            if self.phase.attempt.purpose == "planning":
                self.check_local_sources()
            local, review = self.controller._compose_local_provider(
                self.session.id,
                turn_context=configuration,
                project_root=self.binding.root,
                allow_write=self.binding.allow_write,
                admitted_roots=(self.binding,),
            )
            self.kwargs.setdefault("mcp_provider", mcp)
            self.kwargs.setdefault("local_provider", local)
            self.kwargs.setdefault(
                "builtin_gate",
                build_builtin_gate(
                    self.source.mcp, profile_id=configuration.tool_policy_profile_id
                ),
            )
            self.kwargs.setdefault("review_tool_calls", review)
            self.kwargs.setdefault(
                "persona_policy_rules", configuration.persona_policy_rules
            )

    def _enter(self):
        with self.phase._task_scope_lock:
            if self.task_id in self.phase._task_scopes:
                raise PermissionError("workflow task already prepared")
        saved = self.service.get(self.phase.attempt.workflow_id)
        task = (
            None
            if self.phase.attempt.purpose == "planning"
            else next(
                task
                for phase in saved.plans[self.phase.attempt.revision_id - 1].phases
                if phase.id == self.phase.attempt.phase_id
                for task in phase.tasks
                if task.id == self.task_id
            )
        )
        if (
            self.resolution is not self._planning_resolution
            if task is None
            else goal_provider_ref(self.resolution) != task.target
        ) or not self.resolution.ready:
            raise PermissionError("workflow native target changed")
        if task is None:
            from .console_chat_store import ConsoleMessageRole

            self.planning_request = saved.request
            assistant = self.store.append_message(
                self.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
            )
            self.assistant_message_id = assistant.id
            self.generation_token = self.store.begin_generation_attempt(assistant.id)
        kwargs = dict(self.kwargs)
        if (
            task is not None
            and task.execution_mode == "exclusive"
            and "shell_exec" in task.allowed_tools
            and "raw_shell_provider" not in kwargs
        ):
            from tldw_chatbook.Agents.raw_shell_tool_provider import (
                RawShellToolProvider,
            )

            from tldw_chatbook.Tools.raw_cli_executor import ordinary_raw_shell_metadata

            from .console_chat_controller import build_combined_review_hook
            from .console_raw_cli_custody import NativeRawCliSource

            source = self.runtime._raw_cli_source

            def shell_source():
                if (
                    type(source) is not NativeRawCliSource
                    or self.runtime._raw_cli_source is not source
                    or source.runtime is not self.runtime
                    or source.app is not self.app
                    or source.raw is not self.app.raw_cli_runtime
                ):
                    raise PermissionError("original workflow shell source required")
                raw, reachable, values = ordinary_raw_shell_metadata(self)
                if raw is not source.raw or not reachable or values[:1] == ("unknown",):
                    raise PermissionError("workflow shell unavailable")
                return values

            shell_inputs = shell_source()
            shell, shell_review = self.controller._compose_raw_shell_provider(
                session_id=self.session.id,
                turn_context=self.configuration,
                project_root=self.binding.root,
            )
            self.check_local_sources()
            if (
                shell_source() != shell_inputs
                or type(shell) is not RawShellToolProvider
                or shell._runtime is not source.raw
                or shell.console_session_id != self.session.id
                or shell_review is None
            ):
                raise PermissionError("workflow shell unavailable")
            review = kwargs.get("review_tool_calls")
            kwargs["raw_shell_provider"] = shell
            kwargs["review_tool_calls"] = (
                build_combined_review_hook([review, shell_review])
                if review
                else shell_review
            )
        if task is None:
            from functools import partial

            kwargs["scratch_root"] = self.source.scratch.root
            kwargs["scratch_lease"] = partial(
                self.source.scratch_manager.lease, self.source.scratch
            )
        if self.project_state.project_instructions_enabled:
            from tldw_chatbook.Agents.project_instruction_resolver import (
                ProjectInstructionResolver,
            )
            from tldw_chatbook.config import (
                DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                coerce_int_setting,
                get_cli_setting,
            )

            if (
                self.project_state.working_folder_binding_id != self.binding.binding_id
                or self.project_state.working_folder_locator_fingerprint
                != self.binding.locator_fingerprint
            ):
                raise PermissionError("workflow instruction selection changed")
            if task is None:
                from .console_chat_controller import resolve_project_instruction_binding

                self._planning_instruction_selection = (
                    resolve_project_instruction_binding(
                        self.session, self.source.registry
                    )
                )
                kwargs["confirm_project_instruction_dispatch"] = (
                    self._planning_dispatch_consent
                )
            kwargs["startup_instruction_candidate"] = (
                ProjectInstructionResolver().resolve_startup(
                    binding_id=self.binding.binding_id,
                    binding_root=self.binding.root,
                    locator_fingerprint=self.binding.locator_fingerprint,
                    max_bytes=coerce_int_setting(
                        get_cli_setting(
                            "console",
                            "project_instructions_startup_max_bytes",
                            DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        ),
                        DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        minimum=MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        maximum=MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                    ),
                    dispatch_started_wall_ns=time.time_ns(),
                    excluded_dirs=frozenset(
                        self.binding.exclusions_provider()
                        if self.binding.exclusions_provider
                        else ()
                    ),
                )
            )
        self._lifetime = self.bridge._prepare_native_reply(
            conversation_id=self.phase.attempt.conversation_id,
            session_id=self.session.id,
            resolution=self.resolution,
            assistant_message_id=self.assistant_message_id or "",
            generation_token=self.generation_token,
            model=self.resolution.model,
            session_system_prompt=(
                self._planning_system_prompt or "" if task is None else ""
            ),
            agent_messages=[],
            should_cancel=self.should_cancel,
            work_origin=WorkOrigin.AUTOMATIC,
            work_chain_id=self.phase.attempt.chain_id,
            workspace_id=self.binding.workspace_id,
            workspace_ephemeral=False,
            workspace_binding_authority=(self.binding,),
            skills_context={},
            _workflow_preparation=self,
            **kwargs,
        )
        self.native = self._lifetime.__enter__()
        if task is None:
            self._planning_native_frame = self.native, self._lifetime
            self._planning_native_sources()
        candidate = self.native.service.startup_instruction_candidate
        if candidate is not None:
            plan = self.native.first_request_plan
            _request, self._snapshot = (
                self.native.service.build_project_instruction_request(
                    candidate=candidate,
                    config=plan.config,
                    api_endpoint=plan.api_endpoint,
                    runtime_schemas=list(plan.schemas.runtime_schemas),
                    messages=plan.messages,
                    active_schemas=plan.schemas.active_schemas,
                )
            )

    def _capture(self):
        plan = self.native.first_request_plan
        if not plan.schemas.request_fits:
            raise PermissionError("workflow required task seed does not fit")
        if self.phase.attempt.purpose == "planning":
            selected = self._planning_system_prompt
            if selected and selected not in plan.config.system_prompt:
                raise PermissionError("required planning system context was removed")
            self._planning_composed_system = plan.config.system_prompt
            self.service.validate_planning_inputs(self, plan.messages)
        else:
            self.service.validate_task_inputs(self.authorization, plan.messages)
        if self._snapshot is not None:
            _request, current = self.native.service.build_project_instruction_request(
                candidate=self.native.service.startup_instruction_candidate,
                config=plan.config,
                api_endpoint=plan.api_endpoint,
                runtime_schemas=list(plan.schemas.runtime_schemas),
                messages=plan.messages,
                active_schemas=plan.schemas.active_schemas,
            )
            if current != self._snapshot:
                raise PermissionError("workflow instruction source changed")
        if self.phase.attempt.purpose == "planning":
            self.planning_service = self.native.service
            self.native.service._workflow_planning_preparation = self
            return
        self.native.service._prepare_workflow_child(
            authorization=self.authorization,
            first_request_plan=plan,
            target_ref=goal_provider_ref(self.resolution),
            startup_instruction_snapshot=self._snapshot,
        )

    async def close(self, *, _unpublished=None, _wave=None):
        """Retire once after actual child drain and required durable publication."""
        if self.phase is None or self.phase.attempt.purpose == "planning":
            try:
                self._planning_reads_returned()
            except BaseException:  # noqa: BLE001 - retain all ambiguous original readers.
                self.cancel_event.set()
                return  # Original ambiguous read/prepare retains intake and source.
            if self._finite is not None:
                self.cancel_event.set()
                return  # Unknown finite return retains its original source/claim.
            if self.primary_owner is not None and not self._planning_settled:
                self.cancel_event.set()
                return  # Original tracked runner alone can prove terminal return.
            if not self._closed:

                def close_unused():
                    if self._lifetime is not None and not self._native_closed:
                        self._lifetime.__exit__(None, None, None)
                        self._native_closed = True
                    if self.assistant_message_id is not None:
                        self.store.retire_generation_attempt(
                            self.assistant_message_id, self.generation_token
                        )
                    if self.phase is not None:
                        self.phase._ledger.settle_workflow_phase(
                            self.phase.attempt.id,
                            owner_id=self.phase.attempt.owner_id,
                            drained=True,
                            plan=None,
                            outcome="cancelled",
                        )
                    elif (
                        self._planning_attempt is not None
                        and self._planning_attempt.state == "prepared"
                    ):
                        _abort_workflow_planning(
                            self.service.db.automatic_work, self._planning_attempt, _prepared=self
                        )

                cancellation = None
                try:
                    await self._read(close_unused, closing=True)
                except asyncio.CancelledError as error:
                    self._planning_reads_returned()
                    cancellation = error
                self._closed = True
                self._release_configuration()
                self._release_planning_slots()
                if cancellation is not None:
                    raise cancellation
            return
        self._execution_reads_returned()
        _storage_source_current(self.source, preparation=self)
        if self._closed:
            if _unpublished is not None:
                raise PermissionError("unpublished frame close cannot be replayed")
            self._execution_terminal_return()
            return

        from .console_workflow_wave import (
            _workflow_close_association,
            _workflow_close_current,
        )

        closing_wave = self._wave
        association = _workflow_close_association(self, _wave)

        def finish():
            _storage_source_current(self.source, preparation=self)
            _workflow_close_current(association)
            auth = self.authorization
            with auth._lock if auth is not None else nullcontext():
                if self._closed:
                    return
                if self._closing:
                    raise PermissionError("workflow native retirement already selected")
                native = (
                    auth._native_preparation[0]
                    if auth is not None and auth._native_preparation
                    else None
                )
                if (
                    self._wave is not closing_wave
                    or self._launch_operation is not None
                    or auth is not None
                    and auth._native_launch == [None]
                    or native is not None
                    and (
                        native.uncertain_start
                        or native.wait_return
                        and native.wait_return[0] is None
                    )
                ):
                    raise PermissionError("workflow launch return has not completed")
                if auth is not None and auth._native_launch:
                    tracker = self.service._tasks.get(auth.claim.id)
                    if (
                        tracker is None
                        or tracker.receipt is None
                        or tracker.drain is not True
                    ):
                        raise PermissionError("workflow native owner has not drained")
                    if tracker.receipt.failure != "no_run":
                        self.service._checkpoint(auth.claim.workflow_id, auth.claim.id)
                # A refused close leaves admitted work authorized. The launch
                # claim rechecks this transition under the same lock.
                self._closing = True
            # Failed retirement must not readmit a partly closed native frame.
            if self.native is not None and not self._native_closed:
                self._no_launch_close_guard()
                with self.activate_source():
                    self._lifetime.__exit__(None, None, None)
                self._native_closed = True
            elif self.turn_lifeline is not None:
                self.turn_lifeline.shutdown()
            _workflow_close_current(association)
            with auth._lock if auth is not None else nullcontext():
                if self._wave is not closing_wave or self._launch_operation is not None:
                    raise PermissionError("original workflow close association changed")
                self._closed = True
                self._closing = False

        await self._read(finish, roots=False, closing=True, _unpublished=_unpublished)
        _workflow_close_current(association)
        if self._wave is None:
            records = tuple(self._finite_reads)
            closing_read = records[-1]
            self._finite_read_return(closing_read)
            if self._source_scope is not None:
                try:
                    returned = self._release_configuration()
                    self._configuration_received(returned)
                except BaseException as error:
                    self._configuration_transition_error = error
                    raise
            else:
                saved = self._consent_configuration_return
                if type(saved) is not tuple or len(saved) != 11:
                    raise RuntimeError("original configuration return unproven")
                prefix = saved[7]
                if type(prefix) is not tuple or len(self._finite_reads) <= len(prefix):
                    raise RuntimeError("original consent read unproven")
                returned = self._consent_configuration_source(
                    self._finite_reads[len(prefix)]
                )
            self._terminal_configuration_return = returned, closing_read, records
            self._execution_configuration_return()

    def _close_on_native_owner(self, owner):
        """Return the original frame inside its pre-held native child operation."""
        from tldw_chatbook.Agents.execution_capacity import require_workflow_operation

        _storage_source_current(self.source, preparation=self)
        with self.authorization._lock:
            operation = self._close_operation
        if operation is None or operation.owner is not owner or not owner.is_admitted:
            raise PermissionError("original workflow close obligation required")
        # A tool may outlive its logical timeout. Its actual native thread
        # returns after source/storage and finish callbacks; do not unbind a
        # provider while that original effectful frame can still resume.
        for worker in self._tool_workers:
            worker.join()
        _storage_source_current(self.source, preparation=self)
        with (
            owner.activate(),
            self.activate_source(),
            owned_goal_storage(self.service),
            execution(self.bridge),
        ):
            if self.native is not None and not self._native_closed:
                self.check_local_sources()
                self._lifetime.__exit__(None, None, None)
                self._native_closed = True
            require_workflow_operation(operation)
            operation.finish()
        require_workflow_operation(operation)
        with self.authorization._lock:
            if self._close_operation is not operation:
                raise PermissionError("original workflow close operation changed")
            self._close_operation = None

    def _planning_release_proof(self, returned):
        """Read native retirement and the same planning result after release."""
        from .console_workflow_runs import _prove_workflow_read

        if returned is not self._planning_release_return or len(returned) != 6:
            raise RuntimeError("planning release result required")
        binding, proof, owner, claim, configuration, _cancellation = returned
        _prove_workflow_read(proof, self.controller, self, binding[3])
        drained = []
        owner.on_drained(drained.append)
        if (
            binding is not self._planning_release_binding
            or proof is not self._planning_read_return
            or owner is not self.primary_owner
            or owner.is_admitted
            or drained != [True]
            or claim is not self.primary_claim
            or claim._activity is not binding[20]
            or claim.id in binding[20]._claims
            or not self._planning_settled
            or not self._closed
            or self._planning_intake is not None
            or self._planning_token is not None
            or any(
                a is not b
                for a, b in zip(
                    configuration,
                    (self, binding[17], binding[18], binding[21]),
                    strict=True,
                )
            )
            or not self._planning_configuration_closed
            or self._source_scope is not None
            or self in binding[18]
        ):
            raise RuntimeError("planning native return unproven")
        return proof[0].result

    def _planning_release_received(self, *, _cancellation=None):
        from tldw_chatbook.Agents.workflow_models import WorkflowSnapshot

        returned = self._planning_release_return
        terminal = self._planning_release_proof(returned)
        binding = returned[0]
        coordinator, run = binding[6:8]
        if (
            run is None
            or asyncio.current_task() is not run.driver
            or run.prepared is not self
            or coordinator._runs.get(run.workflow_id) is not run
            or not isinstance(terminal, WorkflowSnapshot)
            or terminal.id != run.workflow_id
            or returned[5] is not _cancellation
        ):
            raise PermissionError("planning result owner required")
        return returned

    def _notify_planning_release(
        self, returned, *, _publication=None, _popped_run=None
    ):
        self._planning_release_proof(returned)
        binding = returned[0]
        coordinator, run, runtime = binding[6:9]
        if asyncio.current_task() is not binding[3]:
            raise PermissionError("planning notification driver required")
        if run is not None:
            if _popped_run is not run or run.workflow_id in coordinator._runs:
                raise PermissionError("planning run release unproven")
            from .console_workflow_runs import _metadata_publication_return

            _metadata_publication_return(coordinator, run, _publication)
        elif _publication is not None or _popped_run is not None:
            raise PermissionError("unexpected planning publication")
        wake, lock = binding[12], binding[15]
        with lock:
            if wake._registry_lock is not lock:
                raise PermissionError("planning wake owner changed")
            if runtime._workflow_closing and not wake._disposed:
                raise PermissionError("planning wake shutdown pending")
        wake.retry_soon()

    def _release_planning_slots(self, *, _completion=None, _configuration_return=None):
        if _completion is not None:
            binding = _completion
            _planning_release_source(self, binding, slots_live=True)
            drained = []
            self.primary_owner.on_drained(drained.append)
            if (
                self.primary_owner.is_admitted
                or drained != [True]
                or self.primary_claim.id in binding[20]._claims
                or any(
                    a is not b
                    for a, b in zip(
                        _configuration_return,
                        (self, binding[17], binding[18], binding[21]),
                        strict=True,
                    )
                )
                or self._source_scope is not None
                or self in binding[18]
            ):
                raise PermissionError("planning retirement unproven")
        if (
            self.controller._active_cancel_events.get(self.session.id)
            is self.cancel_event
        ):
            self.controller._active_cancel_events.pop(self.session.id)
        if self._planning_intake is not None:
            if self.store.release_received_turn(self._planning_intake) is not True:
                raise PermissionError("planning intake release unproven")
            self._planning_intake = None
        if self._planning_token is not None:
            if (
                self.controller._fleet_wake.release_automatic_primary(
                    self.session.id, self._planning_token
                )
                is not True
            ):
                raise PermissionError("planning primary release unproven")
            self._planning_token = None
        return _completion

    async def _planning_instruction_decision(self, decision):
        from .console_chat_controller import (
            commit_project_instruction_dispatch_decision,
        )
        from .console_project_instructions import project_instruction_notice_key

        self.require_current()
        result = commit_project_instruction_dispatch_decision(
            store=self.store,
            session_id=self.session.id,
            registry=self.source.registry,
            expected_state=self.project_state,
            expected_selection=self._planning_instruction_selection,
            notice_key=project_instruction_notice_key(
                self.binding.locator_fingerprint,
                self.resolution.execution_key or self.resolution.provider,
                self.resolution.base_url,
            ),
            decision=decision,
        )
        if result == "proceed":
            # The original commit returned this exact same session publication.
            self.project_state = self.session.project_instruction_state
            self.controller._remember_project_instruction_delivery(
                self.session.id, self._snapshot
            )
        elif result != "prompt":
            self.controller._clear_project_instruction_delivery(self.session.id)
        return result

    def _planning_dispatch_consent(self, snapshot):
        from .console_chat_controller import build_project_instruction_dispatch_notice

        if (
            snapshot != self._snapshot
            or self.controller._confirm_project_instruction_dispatch
            is not self._planning_consent_callback
        ):
            raise PermissionError("original planning consent source changed")
        if self.primary_owner is not None:
            from tldw_chatbook.Agents.execution_capacity import current_execution_owner

            self.check()
            if current_execution_owner() is not self.primary_owner:
                raise PermissionError("original planning instruction receiver required")
            return "proceed"
        callback = self._planning_consent_callback
        if callback is None:
            return "cancel"
        return callback(
            build_project_instruction_dispatch_notice(
                snapshot,
                session_id=self.session.id,
                resolution=self.resolution,
            )
        )

    def _capture_planning_shell_metadata(self):
        """Consume original already-published metadata without a shell provider."""
        from tldw_chatbook.Agents import raw_shell_tool_provider as native
        from tldw_chatbook.Agents.persona_policy import (
            evaluate_tool_policy,
            parse_persona_policy_from_rules,
        )
        from tldw_chatbook.MCP.permission_store import resolve_effective_state_by_key

        from tldw_chatbook.Tools.raw_cli_executor import ordinary_raw_shell_metadata

        from .console_raw_cli_custody import NativeRawCliSource

        source = self.runtime._raw_cli_source
        raw, reachable, values = ordinary_raw_shell_metadata(self)
        from tldw_chatbook.MCP.console_snapshot import _CapturedSources

        payload = (
            _CapturedSources(self.source.mcp).read_permission_payload()
            if self.source.mcp is not None
            else None
        )
        if (
            type(source) is not NativeRawCliSource
            or source.runtime is not self.runtime
            or source.app is not self.app
            or source.raw is not raw
            or not reachable
            or values[:1] == ("unknown",)
            or not self.configuration.tool_configuration.get("local_tools_enabled")
            or payload is None
            or payload.get("kill_switch", False)
            or not evaluate_tool_policy(
                parse_persona_policy_from_rules(
                    self.configuration.persona_policy_rules
                ),
                rule_kind="mcp_tool",
                tool_name=native.RAW_SHELL_TOOL_NAME,
            ).advertised
            or native.resolve_raw_shell_state(
                resolve_effective_state_by_key(
                    payload,
                    native.RAW_SHELL_SERVER_KEY,
                    native.RAW_SHELL_TOOL_NAME,
                    profile_id=self.configuration.tool_policy_profile_id,
                )
            )
            == "deny"
            or native.planning_shell_schema_changed()
        ):
            return None
        schema = native.raw_shell_schema(
            f"{native.SOURCE}:{native.RAW_SHELL_TOOL_NAME}"
        )
        original = (source, raw, values, self.source.permission)
        if self._planning_shell_source is None:
            self._planning_shell_source, self._planning_shell_schema = original, schema
        elif (
            original != self._planning_shell_source
            or schema != self._planning_shell_schema
        ):
            raise PermissionError("original prospective shell metadata changed")
        return schema

    def _admit_planning_primary(self, service):
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner

        self.check()
        if (
            service is not self.planning_service
            or self.native.service is not service
            or current_execution_owner() is not self.primary_owner
            or self._planning_dispatch_closed
        ):
            raise PermissionError("original planning primary required")
        self.runtime.worktree_activity.validate_opaque_claim(
            self.primary_claim, owner=self.primary_owner
        )
        return self.primary_owner

    def require_seed_in_payload(self, messages):
        import json

        self.check()
        original = json.loads(self.planning_seed)
        if not all(any(row == required for row in messages) for required in original):
            raise PermissionError("required planning input was changed or removed")
        expected = self._planning_composed_system
        if expected is None or not any(
            row.get("role") == "system"
            and type(row.get("content")) is str
            and row["content"].startswith(expected)
            for row in messages
        ):
            raise PermissionError(
                "required planning system context was changed or removed"
            )

    @contextmanager
    def _planning_storage_return(self):
        originals = []
        databases = (
            self.source.databases
            if self.source is not None
            else (
                self.service.db,
                self.service.persistence.db,
                self.service.persistence.workspace_registry.db,
            )
        )
        for database in dict.fromkeys(db for db in databases if db is not None):
            namespace = vars(database)
            if namespace.get("is_memory_db") is True:
                continue
            slot = "_local" if namespace.get("_local") is not None else "_thread_local"
            local = namespace.get(slot)
            if local is None:
                raise RuntimeError("original planning database local unavailable")
            originals.append((database, slot, local, getattr(local, "conn", None)))
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        current = threading.current_thread()
        participants = tuple(
            vars(db).get("_maintenance_participant") for db, *_ in originals
        )
        if any(participant is None for participant in participants):
            raise RuntimeError("original planning storage participant unavailable")
        paths = tuple(participant.path for participant in participants)

        def native_state():
            # Exact native thread/source participants only; no IO/callback under lock.
            with storage._lock:
                return (
                    tuple(
                        (
                            participant,
                            frozenset(
                                (conn, lease)
                                for conn, lease in participant.connections.items()
                                if lease.resource_thread is current
                            ),
                            current in participant.retiring_threads,
                        )
                        for participant in participants
                    ),
                    frozenset(
                        operation
                        for operation in storage._operations
                        if operation.thread is current
                        and operation.participant in participants
                    ),
                    frozenset(
                        acquisition
                        for acquisition in storage._pending_acquisitions
                        if acquisition.thread is current
                        and getattr(acquisition.operation, "participant", None)
                        in participants
                    ),
                    frozenset(
                        lease
                        for lease in storage._live_leases
                        if lease.resource_thread is current
                        and lease.resource_path in paths
                    ),
                )

        before = native_state()
        try:
            yield
        finally:
            if (
                not all(
                    vars(db).get(slot) is local and getattr(local, "conn", None) is conn
                    for db, slot, local, conn in originals
                )
                or native_state() != before
            ):
                self._planning_provider_failure = True
                raise RuntimeError("original planning database scope did not return")

    def _planning_native_returned(self):
        lifeline = self.turn_lifeline
        if (
            not self._planning_dispatch_closed
            or self._planning_provider_failure
            or not self._native_closed
            or not self._planning_physical_returned
            or self.native.service is not self.planning_service
            or self.source.workflow_preparation is not self
            or lifeline is not self.native.turn_lifeline
            or any(worker.is_alive() for worker in self._tool_workers)
            or not lifeline._cleanup_returned
            or not lifeline._driver_returned
            or lifeline._thread.is_alive()
            or not lifeline.loop.is_closed()
        ):
            raise RuntimeError("planning physical return unproven")

    def _planning_physical_proof(self):
        self._planning_return_source()
        self._planning_native_returned()
        owner = self.primary_owner
        self.source.check_memory()
        with self.capacity._lock:
            if (
                self.capacity._owners.get(owner.execution_id) is not owner
                or owner._root_finished
                or owner._operations
                or owner._cleanup_unproven
            ):
                raise RuntimeError("planning root return unproven")

    def planning_recovery_return(self):
        """Validate the actual returned owner and released slots, independently of usage."""
        returned = self._planning_recovery_return
        if (
            not isinstance(returned, _PlanningRecoveryReturn)
            or returned.prepared is not self
            or returned.phase is not self.phase
            or returned.read is not self._planning_read_return
            or returned.owner is not self.primary_owner
            or returned.claim is not self.primary_claim
            or not self.phase._retired
            or not self._closed
            or self._source_scope is not None
            or not self._planning_configuration_closed
            or self in self.runtime._workflow_preparations
            or self._planning_intake is not None
            or self._planning_token is not None
        ):
            raise RuntimeError("workflow planning recovery return unproven")
        read, producer, retired = returned.read
        if (
            read.creator is not self.controller
            or read.source is not self
            or read.task is not self._planning_driver
            or producer is not read._producer
            or retired is not read.retired
            or not producer.done()
            or producer.cancelled()
            or producer.exception() is not returned.error
            or not retired.done()
            or retired.cancelled()
            or retired.result() is not None
        ):
            raise RuntimeError("workflow planning recovery return unproven")
        self._planning_native_returned()
        drained = []
        returned.owner.on_drained(drained.append)
        if returned.owner.is_admitted or drained != [True]:
            raise RuntimeError("workflow planning recovery return unproven")
        activity = self.runtime.worktree_activity
        with activity._lock:
            if (
                returned.claim.id in activity._claims
                or returned.claim.id in activity._members
            ):
                raise RuntimeError("workflow planning recovery return unproven")
        return returned

    async def record_planning_recovery(self, *, expected_control_revision=None):
        """Join one retryable accounting recording after actual resource return."""
        returned = self.planning_recovery_return()
        recording = asyncio.create_task(
            self.service.call(
                self.service.record_returned_planning,
                self,
                error=returned.error,
                expected_control_revision=expected_control_revision,
            )
        )
        cancellation = None
        while not recording.done():
            try:
                await asyncio.shield(recording)
            except asyncio.CancelledError as cancelled:
                cancellation = cancellation or cancelled
            except BaseException:  # noqa: BLE001 - inspect and rethrow the actual recording future's outcome.
                break
        try:
            snapshot = recording.result()
        except BaseException as error:
            if cancellation is not None:
                raise cancellation from error
            raise
        self.planning_recovery_return()
        self._planning_recovery_snapshot = snapshot
        if cancellation is not None:
            raise cancellation
        return snapshot

    async def _retire_returned_planning(self, error):
        """Hand durable admission/charges to recovery before retryable recording."""
        self._planning_physical_proof()
        from tldw_chatbook.Agents.workflow_models import workflow_root_identity
        from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
        from tldw_chatbook.Workspaces.worktree_activity import (
            WorktreeFootprint,
            WorktreeScope,
        )

        try:
            chain = capture_directory_chain(self.binding.root)
            if workflow_root_identity(chain) == self.planning_root:
                self.runtime.worktree_activity.protect_recovery(
                    self.phase.attempt.workflow_id,
                    WorktreeFootprint(
                        writes=(WorktreeScope(chain.canonical_root, True, chain),)
                    ),
                )
        except (OSError, ValueError):
            pass  # A replacement pathname grants no historical ownership.
        owner, claim = self.primary_owner, self.primary_claim
        drained = []
        owner.on_drained(drained.append)
        owner.finish_root()
        if drained != [True]:
            raise RuntimeError("planning root retirement unproven")
        # The admission and charges are durable; this hands their unresolved tail to recovery.
        claim.settle(durable=True)
        self.phase._retired = True
        self._closed = True
        self._release_configuration()
        self._release_planning_slots()
        self._planning_recovery_return = _PlanningRecoveryReturn(
            self, self.phase, self._planning_read_return, owner, claim, error
        )
        self.planning_recovery_return()
        await self.record_planning_recovery()

    async def run_planning(self, *, _coordinator_run=None):
        """Run the same native primary and settle only after its actual returns."""
        from tldw_chatbook.Agents.activation import async_worker_guard
        from tldw_chatbook.Agents.agent_models import RUN_DONE
        from tldw_chatbook.Agents.workflow_models import WorkflowPlan
        from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

        from .console_workflow_runs import (
            _planning_driver_source,
            _WorkflowRun,
        )

        driver = asyncio.current_task()
        coordinator = getattr(self.runtime, "_workflow_coordinator", None)
        phase, source = self.phase, self.source
        if _coordinator_run is None:
            if coordinator is not None and any(
                run.driver is driver or run.prepared is self
                for run in coordinator._runs.values()
            ):
                raise PermissionError("original G planning association required")
        elif type(_coordinator_run) is not _WorkflowRun:
            raise PermissionError("original G planning run required")

        def coordinator_source():
            if _coordinator_run is None:
                return
            _planning_driver_source(coordinator, _coordinator_run, entered=True)
            if (
                self.runtime._workflow_coordinator is not coordinator
                or _coordinator_run.prepared is not self
                or _coordinator_run.phase is not phase
                or self.phase is not phase
                or self.source is not source
                or asyncio.current_task() is not driver
            ):
                raise PermissionError("original G planning association changed")

        coordinator_source()

        if self.phase.attempt.purpose != "planning" or self.primary_owner is not None:
            raise PermissionError("unused original planning preparation required")
        self.require_current()
        self._planning_driver = asyncio.current_task()
        if (
            self._planning_release_binding is not None
            or self._planning_release_return is not None
        ):
            raise PermissionError("unused original planning release required")
        activity = self.runtime.worktree_activity
        wake = self.controller._fleet_wake
        release_binding = (
            self,
            phase,
            source,
            driver,
            self._loop,
            None,
            coordinator,
            _coordinator_run,
            self.runtime,
            self.controller,
            self.store,
            self.session,
            wake,
            self._planning_intake,
            self._planning_token,
            wake._registry_lock,
            wake._loop,
            self._source_scope,
            self.runtime._workflow_preparations,
            self.store._preparation_lock,
            activity,
            self._planning_configuration_acquisition,
        )
        self._planning_release_binding = release_binding

        def release_source(*, scope_live=False, slots_live=False):
            _planning_release_source(
                self, release_binding, scope_live=scope_live, slots_live=slots_live
            )

        release_source(scope_live=True, slots_live=True)
        with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
            if claim is None:
                raise PermissionError("planning worktree busy")
            owner = self.capacity.begin_execution(
                origin=WorkOrigin.AUTOMATIC,
                conversation_id=self.phase.attempt.conversation_id,
            )
            claim.bind_owner(owner)
            self.primary_owner, self.primary_claim = owner, claim

        def invoke():

            from .console_workflow_runs import _capture_workflow_read

            self._planning_read_return = _capture_workflow_read(
                self.controller, self, self._planning_driver
            )
            plan, outcome = None, None
            failure = None
            with (
                self._planning_storage_return(),
                owned_goal_storage(self.service),
                execution(self.bridge),
                owner.activate(),
                self.phase.context.scope(),
                self.activate_source(),
            ):
                try:
                    run_id, outcome = self.native.run_primary()
                    self._planning_dispatch_closed = True
                    if outcome.status == RUN_DONE:
                        plan = WorkflowPlan.model_validate_json(outcome.final_text)
                        self.service.validate_planning_plan(self, plan)
                except Exception as error:  # noqa: BLE001 - retain original uncertainty or inspect exact native failure.
                    failure = error
                finally:
                    self._planning_dispatch_closed = True
                    for worker in self._tool_workers:
                        worker.join()
                    self._lifetime.__exit__(None, None, None)
                    self._native_closed = True
                if outcome is not None:

                    @async_worker_guard(self.bridge)
                    async def finalize():
                        with (
                            owned_goal_storage(self.service),
                            self.activate_source(),
                        ):
                            return await self.controller._finalize_agent_reply(
                                self.assistant_message_id,
                                self.session.id,
                                outcome,
                                variant_mode=False,
                                cancel_event=self.cancel_event,
                                run_id=run_id,
                                resolution=self.resolution,
                            )

                    asyncio.run_coroutine_threadsafe(finalize(), self._loop).result()
                self.store.retire_generation_attempt(
                    self.assistant_message_id, self.generation_token
                )
            self._planning_physical_returned = True
            self._planning_physical_proof()
            with (
                self._planning_storage_return(),
                owned_goal_storage(self.service),
                execution(self.bridge),
                owner.activate(),
                self.activate_source(),
            ):
                self._planning_physical_proof()
                cancelled = self._planning_cancelled()
                try:
                    self.phase.check()
                except Exception:  # noqa: BLE001 - retain original uncertainty or inspect exact native failure.
                    cancelled = True
                self.phase._ledger.settle_workflow_phase(
                    self.phase.attempt.id,
                    owner_id=self.phase.attempt.owner_id,
                    drained=True,
                    plan=plan if failure is None and not cancelled else None,
                    outcome="cancelled"
                    if cancelled
                    else "failed"
                    if failure is not None or plan is None
                    else "completed",
                )
                result = self.service.get(self.phase.attempt.workflow_id)
            self._planning_physical_proof()
            self._planning_terminal_returned = True
            return result

        cancellation = None
        try:
            result = await run_preparation_read(
                invoke,
                creator=self.controller,
                session_id=self.session.id,
                reads=self.controller._preparation_reads,
                observers=(self.runtime._preparation_reads,),
                require_current=self._planning_return_source,
                source=self,
            )
        except BaseException as error:
            proof = self._planning_read_return
            producer_error = (
                proof[1].exception()
                if proof is not None and proof[1].done() and not proof[1].cancelled()
                else None
            )
            if producer_error is not None and self._planning_physical_returned:
                await self._retire_returned_planning(producer_error)
                raise
            if (
                not isinstance(error, asyncio.CancelledError)
                or not self._planning_terminal_returned
            ):
                raise
            cancellation = error
            result = None
        read, producer, retired = self._planning_read_return
        if (
            read.creator is not self.controller
            or read.source is not self
            or read.task is not self._planning_driver
            or read._producer is not producer
            or read.retired is not retired
            or not asyncio.Future.done(producer)
            or asyncio.Future.cancelled(producer)
            or asyncio.Future.exception(producer) is not None
            or not asyncio.Future.done(retired)
            or asyncio.Future.cancelled(retired)
            or asyncio.Future.result(retired) is not None
        ):
            raise RuntimeError("original planning tracked worker return unproven")
        self._planning_physical_proof()
        if _coordinator_run is not None:
            coordinator_source()
            control_return = _coordinator_run.control_return
            while control_return is not None and not control_return.is_set():
                try:
                    await control_return.wait()
                except asyncio.CancelledError as error:
                    cancellation = error
            coordinator_source()
            if (
                _coordinator_run.control_return is not control_return
                or _coordinator_run.control_unknown
                or _coordinator_run.status == "recovery_required"
            ):
                raise RuntimeError("original workflow control return unproven")
            self._planning_return_source()
            self._planning_physical_proof()
        release_source(scope_live=True, slots_live=True)
        drain = []
        owner.on_drained(drain.append)
        owner.finish_root()
        if drain != [True]:
            raise RuntimeError("planning root retirement unproven")
        release_source(scope_live=True, slots_live=True)
        self.source.check_memory()
        claim.settle(durable=True)
        self._planning_settled = self._closed = True
        release_source(scope_live=True, slots_live=True)
        configuration_return = self._release_configuration()
        release_source(slots_live=True)
        slots_return = self._release_planning_slots(
            _completion=release_binding, _configuration_return=configuration_return
        )
        if slots_return is not release_binding:
            raise RuntimeError("original planning slots return unproven")
        release_return = (
            release_binding,
            self._planning_read_return,
            owner,
            self.primary_claim,
            configuration_return,
            cancellation,
        )
        self._planning_release_return = release_return
        if _coordinator_run is None:
            from loguru import logger

            try:
                release_source()
                self._notify_planning_release(release_return)
            except BaseException as error:
                logger.exception("Native planning returned; wake notification failed")
                if cancellation is not None:
                    raise cancellation from error
                raise
        if cancellation is not None:
            raise cancellation
        return result


async def prepare_workflow_reply(
    bridge, *, phase, task_id, binding, resolution, **kwargs
):
    """Private producer; the public coordinator still requires phase snapshot custody."""
    from .console_workflow_runs import WorkflowPhaseAuthorization

    if (
        type(phase) is not WorkflowPhaseAuthorization
        or phase.context.workflow is not phase
    ):
        raise PermissionError("original workflow phase required")
    if require_workflow_runtime(phase)._workflow_closing:
        raise PermissionError("workflow admission is closing")
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
    from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

    from .console_provider_gateway import ConsoleProviderResolution

    if (
        type(binding) is not RunAdmittedWorkspaceRoot
        or type(resolution) is not ConsoleProviderResolution
    ):
        raise PermissionError(
            "original workflow binding and native resolution required"
        )
    tracker = bridge._change_tracker
    coordinator = bridge._change_finalization_coordinator
    if tracker is not None and (
        not isinstance(tracker, ChangeTurnTracker)
        or coordinator is not None
        and coordinator._tracker is not tracker
    ):
        raise PermissionError("native workflow change tracker required")
    # These are native composition inputs only. A caller cannot donate chat history,
    # a draft, a continuation, or its own task instruction/primary identity.
    allowed = {
        "mcp_provider",
        "builtin_gate",
        "local_provider",
        "virtual_cli_provider",
        "raw_shell_provider",
        "review_tool_calls",
        "revoke_approvals",
        "on_tool_terminal",
        "on_tool_result_terminal",
        "on_run_terminal",
        "native_tools_enabled",
        "run_budget",
        "confirm_project_instruction_dispatch",
        "on_project_instruction_activation",
        "project_instruction_nested_max_bytes",
        "persona_policy_rules",
        "should_cancel",
    }
    if set(kwargs) - allowed:
        raise PermissionError("unsupported workflow preparation input")
    if phase.attempt.purpose == "planning":
        raise PermissionError("planning requires the original pre-request producer")
    prepared = _PreparedWorkflowReply(
        bridge, phase, task_id, binding, resolution, kwargs
    )
    try:
        return await _finish_native_preparation(prepared)
    except BaseException:
        await prepared.close()
        raise


async def _finish_native_preparation(prepared):
    prepared._retain_configuration()
    await prepared._read(prepared._configure)
    await prepared._read(prepared._enter)
    if prepared._snapshot is not None and prepared.phase.attempt.purpose == "planning":
        decision = await prepared._read(
            lambda: prepared._planning_instruction_decision(None)
        )
        if decision == "prompt":
            answer = await prepared._read(
                lambda: prepared._planning_dispatch_consent(prepared._snapshot),
                roots=False,
            )
            if answer not in {"proceed", "cancel", "disable"}:
                answer = "cancel"
            decision = await prepared._read(
                lambda: prepared._planning_instruction_decision(answer)
            )
        if decision != "proceed":
            raise PermissionError("workflow project instructions declined")
    elif prepared._snapshot is not None:
        if prepared.native.service.confirm_project_instruction_dispatch is None:
            raise PermissionError("workflow project instruction consent unavailable")

        def consent():
            return prepared.native.service.confirm_project_instruction_dispatch(
                prepared._snapshot
            )

        if prepared.phase.attempt.purpose != "planning":
            prefix = tuple(prepared._finite_reads)
            issuing = asyncio.current_task()
            try:
                returned = prepared._release_configuration()
                prepared._configuration_received(returned)
            except BaseException as error:
                prepared._configuration_transition_error = error
                raise
            prepared._consent_configuration_return = (
                returned,
                prepared.phase,
                prepared.authorization,
                prepared.source,
                prepared.native,
                prepared._lifetime,
                prepared._snapshot,
                prefix,
                issuing,
                consent,
                prepared._configured_source_observation,
            )
        decision = await prepared._read(consent, roots=False)
        if decision != "proceed":
            raise PermissionError("workflow project instructions declined")
        prepared._retain_configuration()
        prepared.require_current()
    await prepared._read(prepared._capture)
    return prepared


async def _resolve_workflow_targets(
    controller, session, targets, *, selected=None, setup=None
):
    """Resolve passive references through native settings, without donating routing."""
    from dataclasses import replace

    from tldw_chatbook.Agents.goal_models import GoalProviderRef
    from tldw_chatbook.Agents.workflow_models import WorkflowSetup

    from .console_chat_controller import build_console_provider_selection_from_settings
    from .console_received_dispatch import resolve_received_provider

    if setup is None:
        if any(type(ref) is not GoalProviderRef for ref in targets):
            raise PermissionError("original frozen workflow target required")
    elif type(setup) is not WorkflowSetup or targets is not setup.targets:
        raise PermissionError("original workflow setup choices required")
    original = controller._provider_selection_for_session(session.id)
    settings = controller.store.effective_session_settings(session.id)

    def matches(resolution, target):
        if setup is not None:
            return (resolution.provider, resolution.model) == target
        return goal_provider_ref(resolution) == target

    results = []
    for target in targets:
        provider, model = (
            target if setup is not None else (target.provider, target.model)
        )
        if selected is not None and matches(selected, target):
            resolution = selected
        else:
            resolution = None
            if original.provider == provider:
                candidate = selected or await resolve_received_provider(
                    controller, original
                )
                if candidate.ready and matches(candidate, target):
                    resolution = candidate
            if resolution is not None:
                results.append(resolution)
                continue
            selection = build_console_provider_selection_from_settings(
                replace(
                    settings,
                    provider=provider,
                    model=model,
                    base_url=None,
                ),
                app_config=controller._provider_config()
                if controller._provider_config
                else {},
                workspace_context=original.workspace_context,
                identity_session=session,
                global_user_name=controller._global_user_display_name,
            )
            resolution = await resolve_received_provider(controller, selection)
        if not resolution.ready or not matches(resolution, target):
            raise PermissionError("workflow requested target unavailable")
        results.append(resolution)
    if original != controller._provider_selection_for_session(session.id):
        raise PermissionError("workflow provider selection changed")
    return tuple(results)


async def prepare_workflow_planning(
    controller,
    *,
    session_id,
    binding,
    command_id,
    objective=None,
    policy=None,
    workflow_id=None,
    revision_id=None,
    repair_instruction=None,
    _request=None,
    _pending_run=None,
):
    """Prepare an explicit native Chat, persisting initial identity before request.

    Initial inputs are objective and policy, not an already-stored request whose
    identity could be rewritten. Repair consumes the unchanged saved request.
    The same preparation, intake and primary token survive native first save,
    actual provider resolution, acceptance, catalog and instruction consent.
    """
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
    from tldw_chatbook.Agents.goal_models import GoalBindingRef, GoalProviderRef
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
    from tldw_chatbook.Agents.workflow_models import WorkflowRequest, WorkflowSetup

    from .console_chat_models import ConsoleSubmissionOrigin
    from .console_received_dispatch import (
        resolve_received_provider,
    )
    from .console_workflow_runs import _accept_workflow_phase, native_workflow_qualified

    runtime = controller._hooks_v2_runtime
    coordinator = getattr(runtime, "_workflow_coordinator", None)
    issuing = asyncio.current_task()
    if _pending_run is None:
        if coordinator is not None and any(
            run.issuing is issuing for run in coordinator._pending_planning
        ):
            raise PermissionError("original pending G planning association required")
    else:
        coordinator._pending_planning_source(_pending_run, issuing=True)
        if (
            _pending_run.session_id != session_id
            or _pending_run.binding is not binding
            or _pending_run.command_id != command_id
            or _pending_run.workflow_id != workflow_id
            or _pending_run.planning_input
            != (_request if workflow_id is None else (revision_id, repair_instruction))
        ):
            raise PermissionError("original pending planning inputs changed")
        _pending_run.planning_entered = True
    session = next((s for s in controller.store.sessions() if s.id == session_id), None)
    if (
        not native_workflow_qualified()
        or runtime._workflow_closing
        or session is None
        or session.ephemeral
        or session.runtime_backend != "local"
        or type(binding) is not RunAdmittedWorkspaceRoot
        or session.workspace_id != binding.workspace_id
        or not controller._agent_runtime_enabled
        or session.assistant_kind == "character"
    ):
        raise PermissionError("native workflow planning unavailable")
    if workflow_id is None and (
        not isinstance(objective, str) or type(policy) is not AutomaticWorkLimits
    ):
        raise ValueError("initial planning requires an objective and policy")
    if workflow_id is not None and (objective is not None or policy is not None):
        raise ValueError("repair cannot replace the immutable request")
    setup = type(_request) is WorkflowSetup
    if setup and (_pending_run is None or _pending_run.planning_input is not _request):
        raise PermissionError("original pending workflow setup required")
    if type(_request) is WorkflowRequest and any(
        type(ref) is not GoalProviderRef for ref in _request.targets
    ):
        raise PermissionError("original frozen workflow target required")
    if _request is not None and (
        type(_request) not in (WorkflowRequest, WorkflowSetup)
        or workflow_id is not None
        or _request.objective != objective
        or _request.policy != policy
        or _request.conversation_id != (session.persisted_conversation_id or session.id)
        or _request.binding
        != GoalBindingRef(
            workspace_id=binding.workspace_id,
            binding_id=binding.binding_id,
            locator=str(binding.root),
            access="rw" if binding.allow_write else "ro",
        )
    ):
        raise ValueError("workflow request selection conflict")
    from tldw_chatbook.Agents.workflow_models import RECORD_BYTES

    if workflow_id is None:
        if repair_instruction is not None:
            raise ValueError("initial planning cannot include repair instructions")
    elif (
        type(repair_instruction) is not str
        or not repair_instruction
        or len(repair_instruction.encode("utf-8")) > RECORD_BYTES
        or revision_id is None
    ):
        raise ValueError("repair requires an explicit bounded instruction and revision")
    if _pending_run is not None:
        _pending_run.planning_constructing = True
    prepared = _PreparedWorkflowReply(
        runtime.agent_bridge,
        None,
        None,
        binding,
        None,
        {},
        controller=controller,
        session=session,
        _key=_PLANNING_INPUT,
    )
    if _pending_run is not None:
        prepared._planning_pending_run = _pending_run
        _pending_run.prepared = prepared
        _pending_run.planning_reads = prepared._planning_reads
    prepared._planning_repair_source = (
        command_id,
        workflow_id,
        revision_id,
        repair_instruction,
    )
    prepared._planning_token = object()
    try:
        if not controller._fleet_wake.try_claim_automatic_primary(
            session_id, prepared._planning_token
        ):
            prepared._planning_token = None
            raise PermissionError("planning primary unavailable")
        prepared._planning_intake = prepared.store.claim_received_turn(
            session_id, command_id, origin=ConsoleSubmissionOrigin.WORKFLOW_PHASE
        )
        if prepared._planning_intake is None:
            raise PermissionError("planning intake unavailable")
        controller._active_cancel_events[session_id] = prepared.cancel_event
        prepared._retain_configuration()

        async def select_native():
            prepared.service.db.automatic_work._workflow_enabled()
            require_workflow_hooks_idle(runtime, session_id)
            if not binding.guard(False):
                raise PermissionError("workflow native binding changed")
            selection = controller._provider_selection_for_session(session_id)
            if setup and _request.planner != (
                selection.provider,
                selection.explicit_model or selection.configured_model or "",
            ):
                raise PermissionError(
                    "Displayed planning provider or model changed. Reopen setup."
                )
            resolved = await resolve_received_provider(controller, selection)
            if not resolved.ready or not resolved.model:
                raise PermissionError("workflow provider unavailable")
            if controller._provider_selection_for_session(session_id) != selection:
                raise PermissionError("workflow provider selection changed")
            prepared.resolution = prepared._planning_resolution = resolved
            prepared.planning_selection = selection
            if setup and not _request.targets:
                prepared.execution_resolutions = (resolved,)
            elif _request is not None:
                prepared.execution_resolutions = await _resolve_workflow_targets(
                    controller,
                    session,
                    _request.targets,
                    selected=resolved,
                    setup=_request if setup else None,
                )
            from .console_session_settings import ConsoleSessionSettings

            settings = session.settings
            if type(settings) is not ConsoleSessionSettings:
                raise PermissionError("original selected planning settings required")
            prompt = vars(settings).get("system_prompt")
            if prompt is not None and type(prompt) is not str:
                raise PermissionError("invalid selected planning system context")
            # Later session edits are next-turn input, independent of native unwind.
            prepared._planning_system_prompt = prompt

        await prepared._read(select_native)

        def commit_request():
            # Consume the original native return. Never adopt a later current ID
            # as proof that this invocation persisted/published the chosen Chat.
            from .chat_persistence_service import (
                ChatPersistenceService,
            )
            from .console_chat_store import (
                ConsoleChatStore,
            )
            from .console_library_policy_repository import (
                _NATIVE_FIRST_LIBRARY_REPOSITORY,
            )

            persistence = prepared.store.persistence
            repository = persistence.console_library_policy_repository
            if (
                type(prepared.store) is not ConsoleChatStore
                or type(persistence) is not ChatPersistenceService
                or prepared.service.persistence is not persistence
                or type(repository) is not _NATIVE_FIRST_LIBRARY_REPOSITORY[0]
                or repository.db is not persistence.db
            ):
                raise PermissionError("original planning persistence producer changed")
            conversation = prepared.store.persist_session_if_needed(session_id)
            if not prepared.store.received_turn_is_current(prepared._planning_intake):
                raise PermissionError("original planning intake changed")
            if (
                conversation is None
                or session.persisted_conversation_id != conversation
            ):
                raise PermissionError("original conversation publication unavailable")
            reference = GoalBindingRef(
                workspace_id=binding.workspace_id,
                binding_id=binding.binding_id,
                locator=str(binding.root),
                access="rw" if binding.allow_write else "ro",
            )
            if workflow_id is None:
                request = WorkflowRequest(
                    conversation_id=conversation,
                    objective=objective,
                    binding=reference,
                    targets=tuple(
                        goal_provider_ref(r) for r in prepared.execution_resolutions
                    )
                    if setup
                    else _request.targets
                    if _request is not None
                    else (goal_provider_ref(prepared.resolution),),
                    policy=policy,
                )
                snapshot = prepared.service.db.workflow_runs.create(
                    request, launch_id=command_id
                )
            else:
                snapshot = prepared.service.get(workflow_id)
                if (
                    snapshot.request.conversation_id != conversation
                    or snapshot.request.binding != reference
                ):
                    raise PermissionError("saved workflow selection changed")
            if _pending_run is not None:
                coordinator._pending_planning_source(_pending_run)
                asyncio.run_coroutine_threadsafe(
                    coordinator._planning_identity(_pending_run, prepared, snapshot),
                    prepared._loop,
                ).result()
            prepared.require_current()

            def admission_state():
                with prepared.service.db.connection() as conn:
                    return tuple(
                        tuple(row)
                        for row in conn.execute(
                            "SELECT p.*,a.* FROM workflow_phase_attempts p "
                            "JOIN automatic_wake_attempts a ON a.id=p.native_attempt_id WHERE p.command_id=?",
                            (command_id,),
                        )
                    )

            before_admission = admission_state()
            prepared._planning_prepare_entered = True
            try:
                attempt = prepared.service.db.automatic_work.prepare_workflow_planning(
                    snapshot.id,
                    owner_id=controller._fleet_wake.runtime_owner_id,
                    session_id=session_id,
                    command_id=command_id,
                    revision_id=revision_id,
                    repair_instruction=repair_instruction,
                )
            except BaseException:
                # A native transactional refusal owns no new admission; a changed
                # or unreadable durable return keeps the preparation protected.
                try:
                    unchanged = admission_state() == before_admission
                except Exception:  # noqa: BLE001 - preserve the original admission failure.
                    unchanged = False
                if unchanged:
                    prepared._planning_prepare_entered = False
                raise
            prepared._planning_attempt = attempt
            prepared.require_current()
            if type(prepared) is not _PreparedWorkflowReply:
                raise PermissionError("original planning preparation required")
            prepared.phase = _accept_workflow_phase(
                controller, attempt.id, session_id=session_id, _planning=prepared
            )
            return prepared.phase, attempt

        phase, receipt = await prepared._read(commit_request)
        if phase is None:
            await prepared.close()
            return receipt
        return await _finish_native_preparation(prepared)
    except BaseException:
        await prepared.close()
        raise
    finally:
        prepared._planning_preparation_returned.set()


def _finite_owner_return(owner, capacity, operation=None):
    """Require positive native retirement of the actual owner and operation."""
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner, OwnedOperation

    if not isinstance(owner, ExecutionOwner) or owner.capacity is not capacity:
        raise PermissionError("workflow native execution owner required")
    drained = []
    owner.on_drained(drained.append)
    if drained != [True] or owner.is_admitted:
        raise RuntimeError("workflow finite owner unreturned")
    if operation is not None:
        if not isinstance(operation, OwnedOperation) or operation.owner is not owner:
            raise RuntimeError("workflow finite operation changed")
        with capacity._lock:
            if operation in owner._operations:
                raise RuntimeError("workflow finite operation unreturned")


def _storage_source_current(source=None, *, preparation=None):
    from tldw_chatbook.Agents.native_source_storage import (
        STORAGE_SOURCE_CLASS as _STORAGE_SOURCE_CLASS,
    )

    if preparation is not None and (
        not isinstance(preparation, _PreparedWorkflowReply)
        or source is not preparation.source
    ):
        raise PermissionError("workflow storage receiver changed")
    if source is not None and (
        type(source) is not _STORAGE_SOURCE_CLASS
        or type(source.sources) is not tuple
        or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not _STORAGE_PATH_CLASS
            for row in source.sources
        )
    ):
        raise PermissionError("original workflow storage observations required")
    if preparation is None:
        return
    observation = preparation._configured_source_observation
    if source is None:
        if (
            observation is not None
            or preparation.native is not None
            or preparation._lifetime is not None
            or preparation.authorization is not None
        ):
            raise PermissionError("original workflow initial storage source changed")
        return
    if (
        type(observation) is not tuple
        or len(observation) != 3
        or observation[0] is not preparation
        or observation[1] is not source
        or observation[2] is not source.sources
        or source.workflow_preparation is not preparation
    ):
        raise PermissionError("original workflow storage observation changed")
    auth = preparation.authorization
    if auth is not None:
        if type(auth) is not _STORAGE_AUTH_CLASS:
            raise PermissionError("original workflow storage authorization changed")
        with auth._lock:
            if (
                auth.phase is not preparation.phase
                or not auth._bridge_preparation
                or auth._bridge_preparation[0] is not preparation
            ):
                raise PermissionError("original workflow storage authorization changed")
            if auth._native_preparation:
                native = auth._native_preparation[0]
                if (
                    len(auth._native_preparation) != 1
                    or type(native) is not _STORAGE_NATIVE_CLASS
                    or preparation.native is None
                    or native.service is not preparation.native.service
                    or native.phase is not preparation.phase
                    or native.source_observation is not observation
                    or native.service._workflow_authorization is not auth
                ):
                    raise PermissionError(
                        "original workflow storage observation changed"
                    )


from pathlib import Path as _StoragePath

from tldw_chatbook.Agents.agent_service import (
    WorkflowChildPreparation as _STORAGE_NATIVE_CLASS,
)
from tldw_chatbook.Agents.workflow_task_scope import (
    WorkflowTaskAuthorization as _STORAGE_AUTH_CLASS,
)

_STORAGE_PATH_CLASS = type(_StoragePath())


def _abort_workflow_planning(ledger, original, *, _prepared) -> bool:
    """Abort only the original returned, unaccepted planning invocation."""
    from tldw_chatbook.Agents.workflow_models import WorkflowPhaseAttempt
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    if (
        type(original) is not WorkflowPhaseAttempt
        or type(_prepared) is not _PreparedWorkflowReply
    ):
        raise PermissionError("original prepared planning abort required")
    _prepared._planning_abort_source(ledger, original)
    with ledger.transaction() as conn:
        _prepared._planning_abort_source(ledger, original)
        ledger._check_runtime_owner(conn, original.owner_id)
        native = ledger._attempt(
            conn, original.id, original.owner_id, "workflow_phase"
        )
        phase = conn.execute(
            "SELECT * FROM workflow_phase_attempts WHERE native_attempt_id=?",
            (original.id,),
        ).fetchone()
        workflow = ledger._db.workflow_runs._row(conn, original.workflow_id)
        current = ledger.read_workflow_attempt(
            original.id, owner_id=original.owner_id
        )
        if (
            current != original
            or original.state != "prepared"
            or original.purpose != "planning"
            or original.phase_id is not None
            or phase["status"] != "prepared"
            or phase["retirement"] != "pending"
            or workflow["control_revision"] != original.control_revision
            or conn.execute(
                "SELECT MAX(revision) FROM workflow_plan_revisions WHERE workflow_id=?",
                (original.workflow_id,),
            ).fetchone()[0]
            != original.revision_id
            or conn.execute(
                "SELECT 1 FROM workflow_task_claims WHERE phase_attempt_id=? "
                "UNION ALL SELECT 1 FROM workflow_change_windows WHERE phase_attempt_id=? "
                "UNION ALL SELECT 1 FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL "
                "UNION ALL SELECT 1 FROM workflow_task_attempts WHERE native_attempt_id=? LIMIT 1",
                (phase["id"], phase["id"], original.workflow_id, original.id),
            ).fetchone()
            or conn.execute(
                "SELECT 1 FROM automatic_work_reservations WHERE chain_id=? AND created_at>=? "
                "AND kind IN ('model_call','tokens') AND state NOT IN ('released','reserved') LIMIT 1",
                (original.chain_id, native["created_at"]),
            ).fetchone()
        ):
            raise AutomaticWorkRefused("workflow_planning_abort_unproven")
        now = ledger._wall_clock()
        if (
            conn.execute(
                "UPDATE automatic_work_reservations SET state='released',updated_at=? "
                "WHERE id=? AND chain_id=? AND owner_id=? AND kind='generation' AND amount=1 AND state='reserved'",
                (
                    now,
                    native["generation_reservation_id"],
                    original.chain_id,
                    original.owner_id,
                ),
            ).rowcount
            != 1
        ):
            raise AutomaticWorkRefused("workflow_planning_abort_unproven")
        if (
            conn.execute(
                "UPDATE automatic_wake_attempts SET state='aborted',completed_at=? "
                "WHERE id=? AND owner_id=? AND attempt_kind='workflow_phase' AND state='prepared'",
                (now, original.id, original.owner_id),
            ).rowcount
            != 1
        ):
            raise AutomaticWorkRefused("workflow_planning_abort_unproven")
        if (
            conn.execute(
                "UPDATE workflow_phase_attempts SET status='closed',retirement='drained',result_bytes=0 "
                "WHERE id=? AND native_attempt_id=? AND purpose='planning' AND phase_id IS NULL "
                "AND status='prepared' AND retirement='pending'",
                (phase["id"], original.id),
            ).rowcount
            != 1
        ):
            raise AutomaticWorkRefused("workflow_planning_abort_unproven")
    return True
