"""Finite host custody for the Console's shared native reply producer (ADR-285)."""

from __future__ import annotations

import asyncio
import inspect
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

from tldw_chatbook.Agents.activation import execution
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
from tldw_chatbook.Agents.workflow_task_scope import (
    _check_hooks,
    _prepare_workflow_task,
    _runtime,
)

from .console_goal_runs import goal_provider_ref
from .console_preparation_reads import run_preparation_read
from .console_workflow_runs import WorkflowPhaseAuthorization

_CURRENT = ContextVar("native_workflow_preparation", default=None)
_PLANNING_INPUT = object()


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


def current_workflow_preparation(controller=None):
    prepared = _CURRENT.get()
    if (
        prepared is None
        or controller is not None
        and prepared.controller is not controller
    ):
        return None
    return prepared


def current_planning_preparation(controller=None):
    """Return only the original current native planning producer, not a flag."""
    prepared = current_workflow_preparation(controller)
    if (
        type(prepared) is _PreparedWorkflowReply
        and type(prepared.phase) is WorkflowPhaseAuthorization
        and prepared.phase.attempt.purpose == "planning"
        and prepared.task_id is None
        and prepared in prepared.runtime._workflow_preparations
    ):
        return prepared
    return None


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
        self._planning_system_source = None
        self._planning_composed_system = None
        self._planning_repair_source = None
        self.controller = phase._controller if phase is not None else controller
        self.runtime = (
            _runtime(phase) if phase is not None else controller._hooks_v2_runtime
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
        self._planning_read_return = None
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
        self._native_closed = False
        self._close_operation = None
        self._launch_operation = None
        self._tool_workers = []
        self._no_launch_close_guard = None
        self._configuration_selection = self._skill_preparation = (
            self._skill_catalog
        ) = None
        from tldw_chatbook import config
        from tldw_chatbook.Agents import local_tool_provider

        from .console_ordinary_tree import _capture_native_local_sources

        self._config_identity = config.current_config_identity()
        self.local_providers = {}
        self._planning_builtins = ()
        self._planning_executors = {}
        self._planning_shell_source = self._planning_shell_schema = None
        self.local_native, self.local_source_values, self.local_function_values = (
            _capture_native_local_sources(self.controller, local_tool_provider)
        )
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
        from .console_ordinary_tree import _function_unchanged

        if any(
            inspect.getattr_static(self, name, None) is not descriptor
            or any(not _function_unchanged(record) for record in records)
            for name, descriptor, records in _PLANNING_PREPARATION_METHODS
        ):
            raise PermissionError("original planning receiver changed")
        if any(
            inspect.getattr_static(self.service.db.automatic_work, name) is not method
            or not _function_unchanged(record)
            for name, method, record in _PLANNING_LEDGER_METHODS
        ):
            raise PermissionError("original planning ledger receiver changed")
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
        if self._planning_system_source is not None:
            settings, prompt = self._planning_system_source
            if (
                self.session.settings is not settings
                or vars(settings).get("system_prompt") != prompt
                or inspect.getattr_static(settings, "system_prompt") != prompt
            ):
                raise PermissionError("selected planning system context changed")
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
        self._planning_read_sources()
        if self.source is not None:
            self.source.check_memory()

    def _planning_read_sources(self):
        from . import console_preparation_reads as native
        from .console_ordinary_tree import _function_unchanged

        read_class, binding_class, binding_slot, functions, classes = (
            native._NATIVE_PLANNING_READ_RETURN
        )
        if (
            native.ConsolePreparationRead is not read_class
            or native._HookSourceBinding is not binding_class
            or native._source_binding is not binding_slot
            or native.run_preparation_read is not functions[0][0]
            or run_preparation_read is not functions[0][0]
            or native._retire is not functions[1][0]
            or any(not _function_unchanged(record) for record in functions)
            or any(
                inspect.getattr_static(owner, name) is not descriptor
                for owner, descriptors in classes
                for name, descriptor in descriptors
            )
        ):
            raise PermissionError("original planning tracked-read producer changed")
        return read_class, binding_class, binding_slot

    def _planning_native_sources(self):
        from tldw_chatbook.Agents import agent_service

        from . import console_agent_bridge as native
        from . import console_chat_controller as host
        from .console_ordinary_tree import _function_unchanged
        from .console_provider_gateway import (
            _CONSOLE_PLANNING_LOOP_CLOSE_NATIVE,
            _CONSOLE_PLANNING_STREAM_NATIVE,
        )

        stream = _CONSOLE_PLANNING_STREAM_NATIVE
        if inspect.getattr_static(
            self._planning_gateway, "_stream_generic_chat"
        ) is not stream[0] or not _function_unchanged(stream):
            raise PermissionError("original planning provider worker changed")
        close = _CONSOLE_PLANNING_LOOP_CLOSE_NATIVE
        if (
            self.controller.provider_gateway is not self._planning_gateway
            or self.bridge._gateway is not self._planning_gateway
            or inspect.getattr_static(self._planning_gateway, "aclose_current_loop")
            is not close[0]
            or not _function_unchanged(close)
            or self.turn_lifeline is not None
            and (
                getattr(self.turn_lifeline._close_current_loop, "__self__", None)
                is not self._planning_gateway
                or getattr(self.turn_lifeline._close_current_loop, "__func__", None)
                is not close[0]
            )
        ):
            raise PermissionError("original planning client cleanup changed")
        for owner, name, method, records in native._NATIVE_PLANNING_TRANSPORT:
            receiver = (
                self.bridge
                if owner is native.ConsoleAgentBridge
                else (
                    self.turn_lifeline
                    if owner is native._ModelCallLifeline
                    else self.native.adapter
                    if self.native is not None
                    else None
                )
            )
            if (
                (
                    receiver is not None
                    and (
                        type(receiver) is not owner
                        or inspect.getattr_static(receiver, name) is not method
                    )
                )
                or inspect.getattr_static(owner, name) is not method
                or any(not _function_unchanged(record) for record in records)
            ):
                raise PermissionError("original planning native transport changed")
        for name, method, record in host._NATIVE_PLANNING_FINALIZERS:
            if inspect.getattr_static(
                self.controller, name
            ) is not method or not _function_unchanged(record):
                raise PermissionError("original planning finalizer changed")
        for name, record in host._NATIVE_PLANNING_CONSENT_HELPERS:
            if vars(host).get(name) is not record[0] or not _function_unchanged(record):
                raise PermissionError("original planning consent helper changed")
        for name, method, records in agent_service._NATIVE_PLANNING_SERVICE:
            receiver = (
                self.native.service
                if self.native is not None
                else agent_service.AgentService
            )
            if inspect.getattr_static(receiver, name) is not method or any(
                not _function_unchanged(record) for record in records
            ):
                raise PermissionError("original planning service changed")
        if self._planning_native_frame is not None:
            original, lifetime, generator, primary, closure = (
                self._planning_native_frame
            )
            if (
                self.native is not original
                or self._lifetime is not lifetime
                or lifetime.gen is not generator
                or generator.gi_code
                is not native.ConsoleAgentBridge._prepare_native_reply.__wrapped__.__code__
                or original.run_primary is not primary
                or primary.__code__ not in generator.gi_code.co_consts
                or primary.__closure__ is not closure
            ):
                raise PermissionError("original planning primary frame changed")
            bindings = dict(
                zip(primary.__code__.co_freevars, primary.__closure__, strict=True)
            )
            for name, expected in (
                ("self", self.bridge),
                ("planning", self),
                ("service", original.service),
                ("first_request_plan", original.first_request_plan),
                ("turn_lifeline", self.turn_lifeline),
            ):
                if bindings[name].cell_contents is not expected:
                    raise PermissionError(
                        "original planning primary association changed"
                    )

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
        """Prove the original factory, concrete receiver and bounded readers."""
        from tldw_chatbook.Agents import local_tool_provider as local
        from tldw_chatbook.Tools import local_tool_impls as readers
        from tldw_chatbook.Tools import workspace_tool_executor as native

        from .console_ordinary_tree import _function_unchanged

        cls = native._NATIVE_ORDINARY_EXECUTOR_CLASS
        if (
            local._filesystem_specs is not local._ORDINARY_NATIVE_FUNCTIONS[1][0]
            or not _function_unchanged(local._ORDINARY_NATIVE_FUNCTIONS[1])
            or local.WorkspaceToolExecutor is not cls
            or native.WorkspaceToolExecutor is not cls
            or inspect.getattr_static(cls, "__getattribute__")
            is not object.__getattribute__
            or inspect.getattr_static(cls, "__new__") is not object.__new__
            or inspect.getattr_static(cls, "__hash__") is not object.__hash__
            or inspect.getattr_static(cls, "__eq__") is not object.__eq__
            or inspect.getattr_static(cls, "__getattr__", None) is not None
            or inspect.getattr_static(cls, "__bool__", None) is not None
            or inspect.getattr_static(cls, "__len__", None) is not None
            or any(
                inspect.getattr_static(cls, name, None) is not value
                for name, value in native._NATIVE_ORDINARY_EXECUTOR_METHODS
            )
            or any(
                vars(native).get(name) is not value
                for name, value in native._NATIVE_ORDINARY_EXECUTOR_HELPERS
            )
            or any(
                not _function_unchanged(record)
                for record in native._NATIVE_ORDINARY_EXECUTOR_FUNCTIONS
            )
            or any(
                vars(readers).get(name) is not record[0]
                or name == "resolve_workspace_path"
                and vars(native).get(name) is not record[0]
                or not _function_unchanged(record)
                for name, record in readers._NATIVE_PLANNING_READERS
            )
        ):
            raise PermissionError("original planning filesystem source changed")
        if executor is None:
            return
        if type(executor) is not cls or any(
            inspect.getattr_static(executor, name, None) is not value
            for name, value in native._NATIVE_ORDINARY_EXECUTOR_METHODS
        ):
            raise PermissionError("original planning executor receiver changed")
        namespace = object.__getattribute__(executor, "__dict__")
        names = (
            "_remote_root",
            "_workspace_root",
            "_authority_chain",
            "_user_exclusion_paths",
        )
        values = tuple(namespace.get(name) for name in names)
        if (
            any(
                inspect.getattr_static(executor, name, None) is not value
                for name, value in zip(names, values, strict=True)
            )
            or values[0] is not None
            or values[1] != self.binding.root
            or type(values[2]) is not native.DirectoryChain
            or values[2].canonical_root != self.binding.root
            or values[3] is not self.binding.exclusions_provider
        ):
            raise PermissionError("original planning executor binding changed")
        retained = self._planning_executors.setdefault(executor, values)
        if any(a is not b for a, b in zip(retained, values, strict=True)):
            raise PermissionError("original planning executor fields changed")

    def check_local_sources(self):
        from .console_ordinary_tree import (
            _check_local_provider_sources,
            _function_unchanged,
        )

        if any(
            not _function_unchanged(record) for record in self.local_function_values
        ) or any(
            inspect.getattr_static(obj, name, None) is not value
            or code is not None
            and value.__code__ is not code
            for obj, name, value, code in self.local_source_values
        ):
            raise PermissionError("workflow native local source changed")
        planning = self.phase is None or self.phase.attempt.purpose == "planning"
        if planning:
            self.check_planning_executor()
            for executor in tuple(self._planning_executors):
                self.check_planning_executor(executor)
        for provider, retained in self.local_providers.items():
            _check_local_provider_sources(provider, retained)
            if planning:
                for _alias, entries in retained[2]:
                    for name, spec in entries:
                        if name not in {"fs_list", "fs_read"}:
                            continue
                        cells = dict(
                            zip(
                                spec.handler.__code__.co_freevars,
                                spec.handler.__closure__ or (),
                                strict=True,
                            )
                        )
                        cell = cells.get("workspace_executor")
                        if (
                            cell is None
                            or cell.cell_contents not in self._planning_executors
                        ):
                            raise PermissionError(
                                "original planning handler receiver changed"
                            )
                        self.check_planning_executor(cell.cell_contents)
        if self._planning_builtins:
            from tldw_chatbook.Agents import tool_catalog

            if any(
                not _function_unchanged(record)
                for record in (
                    *tool_catalog._NATIVE_ORDINARY_BUILTIN_IMPLEMENTATIONS,
                    *tool_catalog._PLANNING_PURE_TOOL_FUNCTIONS,
                )
            ):
                raise PermissionError("original pure planning source changed")
            for provider, tools in self._planning_builtins:
                if tuple(provider._tools.items()) != tools:
                    raise PermissionError("original pure planning tools changed")
                for name, descriptor in tool_catalog._NATIVE_ORDINARY_BUILTIN_METHODS:
                    if inspect.getattr_static(provider, name, None) is not descriptor:
                        raise PermissionError("original pure planning provider changed")
                for tool in provider._tools.values():
                    if (
                        type(tool)
                        not in tool_catalog._NATIVE_ORDINARY_ALWAYS_ON_CLASSES
                    ):
                        raise PermissionError("unknown planning builtin")
                    for (
                        cls,
                        name,
                        descriptor,
                    ) in tool_catalog._PLANNING_PURE_TOOL_METHODS:
                        if (
                            type(tool) is cls
                            and inspect.getattr_static(tool, name, None)
                            is not descriptor
                        ):
                            raise PermissionError(
                                "original pure planning implementation changed"
                            )

    @contextmanager
    def activate_source(self):
        token = _CURRENT.set(self)
        try:
            with self.source.activate() if self.source is not None else nullcontext():
                yield
        finally:
            _CURRENT.reset(token)

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

    def _retain_configuration(self):
        if self._source_scope is None:
            scope = self.runtime._first_send_source_scope()
            scope.__enter__()
            self._source_scope = scope
            self.runtime._workflow_preparations.add(self)

    def _release_configuration(self):
        if self._source_scope is not None:
            scope, self._source_scope = self._source_scope, None
            scope.__exit__(None, None, None)
            self.runtime._workflow_preparations.discard(self)

    def check(self):
        self.require_current()
        self.phase.check()
        _check_hooks(self.runtime, self.session.id)
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
        retirement = []
        from .console_workflow_runs import _capture_workflow_read

        record = _PlanningPreparationRead(self, asyncio.current_task())
        self._finite_reads.append(record)
        issuing = record.issuing

        def invoke_owned():
            with self.phase.context.scope():
                self.check()
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
                            _WorkflowChildWait,
                        )

                        raise _WorkflowChildWait(
                            "worktree_activity", "workflow preparation busy"
                        )
                    owner = self.capacity.begin_execution(
                        origin=WorkOrigin.AUTOMATIC,
                        conversation_id=self.phase.attempt.conversation_id,
                    )
                    record.owner, record.claim = owner, claim
                    operation = None
                    proven = False
                    try:
                        if claim is not None:
                            claim.bind_owner(owner)
                        operation = owner.reserve_tool()
                        record.operation = operation
                        self._finite = (owner, claim)
                        with owner.activate(), self.activate_source():
                            self.check()
                            self.check_local_sources()
                            result = callback()
                            if inspect.isawaitable(result):
                                from tldw_chatbook.Agents.activation import (
                                    async_worker_guard,
                                )

                                pending = result

                                @async_worker_guard(self.bridge)
                                async def complete():
                                    with (
                                        owned_goal_storage(self.service),
                                        self.activate_source(),
                                    ):
                                        return await pending

                                result = asyncio.run_coroutine_threadsafe(
                                    complete(), self._loop
                                ).result()
                            self.check()
                            with self.activate_source():
                                operation.finish()
                            proven = True
                            return result
                    finally:
                        if operation is not None and not proven:
                            with owner.activate(), self.activate_source():
                                operation.finish()
                                proven = True
                        self._finite = None
                        if proven:
                            retirement.append((owner, claim, operation))

        def retire():
            for owner, claim, _operation in retirement:
                owner.on_drained(record.drains.append)
                owner.finish_root()
                if claim is not None:
                    claim.settle(durable=True)

        def invoke():
            proof = record.proof = _capture_workflow_read(
                self.controller, self, invoke, issuing
            )
            body_error = None
            try:
                with (
                    self._unpublished_storage_return(_unpublished)
                    if _unpublished is not None
                    else nullcontext(),
                    owned_goal_storage(self.service),
                    execution(self.bridge),
                ):
                    try:
                        result = callback() if closing else invoke_owned()
                    except BaseException as error:
                        body_error = error
                        raise
            except BaseException as error:
                # An original callback exception is conclusive only if its
                # storage/execution exits returned without replacing it.
                if error is body_error:
                    retire()
                raise
            else:
                retire()
                record.scopes_returned = True
                record.result = result
                owner, claim, operation = (
                    retirement[0] if retirement else (None, None, None)
                )
                return (
                    record,
                    self,
                    issuing,
                    invoke,
                    proof,
                    owner,
                    claim,
                    operation,
                    result,
                    None,
                )

        record.callback = invoke
        try:
            await run_preparation_read(
                invoke,
                creator=self.controller,
                session_id=self.session.id,
                reads=self.controller._preparation_reads,
                observers=(self.runtime._preparation_reads,),
                require_current=(lambda: None) if closing else self.require_current,
                source=self,
            )
        except BaseException as error:
            record.wrapper_error = error
            raise
        self._execution_return_source()
        self._finite_read_return(record)
        record.positive = True
        return record.result

    def _finite_read_return(self, record):
        """Consume identities carried by the same native producer's result."""
        from .console_ordinary_tree import _function_unchanged
        from .console_workflow_runs import _prove_workflow_read

        if _finite_owner_return is not _FINITE_OWNER_CHECK[
            0
        ] or not _function_unchanged(_FINITE_OWNER_CHECK):
            raise PermissionError("original workflow finite owner source changed")
        _prove_workflow_read(record.proof, self.controller, self, record.issuing)
        actual = asyncio.Future.result(record.proof[1])
        expected = (
            record,
            self,
            record.issuing,
            record.callback,
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
            or any(first is not second for first, second in zip(actual, expected))
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
        from .console_ordinary_tree import _function_unchanged

        if any(
            inspect.getattr_static(self, name, None) is not descriptor
            or any(not _function_unchanged(record) for record in records)
            for name, descriptor, records in _PLANNING_PREPARATION_METHODS
        ) or (
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
        if (
            _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
            or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
            or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
            or _storage_source_current.__kwdefaults__ is not _STORAGE_SOURCE_CHECK[3]
            or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
        ):
            raise PermissionError("original workflow storage source changed")
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

        record = _PlanningPreparationRead(self, asyncio.current_task())
        self._planning_reads.append(record)
        issuing = record.issuing

        def invoke():
            from tldw_chatbook.Agents.execution_capacity import CapacityRefused

            self._planning_read_sources()
            proof = record.proof = _capture_workflow_read(
                self.controller, self, invoke, issuing
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
                            if inspect.isawaitable(result):
                                pending = result

                                @async_worker_guard(self.bridge)
                                async def complete():
                                    with (
                                        self._planning_storage_return(),
                                        owned_goal_storage(self.service),
                                        self.activate_source(),
                                    ):
                                        return await pending

                                result = asyncio.run_coroutine_threadsafe(
                                    complete(), self._loop
                                ).result()
                        except BaseException as error:  # noqa: BLE001 - retain original uncertainty or inspect exact native failure.
                            failure = error
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
                invoke,
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
            or record.callback is not invoke
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
        caller = inspect.currentframe().f_back.f_back
        record = self._planning_reads[-1] if self._planning_reads else None
        if (
            caller is None
            or caller.f_code is not _PLANNING_ABORT_CALLER
            or caller.f_locals.get("self") is not self
            or ledger is not self.service.db.automatic_work
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
        actual = _capture_workflow_read(
            self.controller, self, record.callback, record.issuing
        )
        if record.proof != actual:
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
                _prepare_planning_authorization,
            )

            from .console_ordinary_tree import (
                _capture_local_provider_sources,
                _function_unchanged,
            )

            if (
                not self.local_native
                or any(
                    not _function_unchanged(r)
                    for r in tool_catalog._NATIVE_ORDINARY_BUILTIN_IMPLEMENTATIONS
                )
                or skills
            ):
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
                elif (
                    type(provider) is not tool_catalog._NATIVE_ORDINARY_BUILTIN_PROVIDER
                ):
                    raise PermissionError("original pure planning tool required")
                else:
                    builtins.add(provider)
            self.local_providers = {
                p: _capture_local_provider_sources(p, n) for p, n in providers.items()
            }
            self._planning_builtins = tuple(
                (provider, tuple(provider._tools.items())) for provider in builtins
            )
            self.check_local_sources()
            self.authorization = _prepare_planning_authorization(self, registry, names)
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
        self.authorization = _prepare_workflow_task(
            self.phase,
            task_id=self.task_id,
            registry=registry,
            binding=self.binding,
        )
        self.authorization._bridge_preparation.append(self)
        if not self.authorization.physical_footprint.opaque:
            from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

            from .console_ordinary_tree import _capture_local_provider_sources

            providers = {}

            self.check_local_sources()
            for name in self.authorization.task.allowed_tools:
                resolved = registry.resolve_owner_for_name(name)
                if resolved is None:
                    raise PermissionError("workflow native tool unavailable")
                provider = resolved[1]
                if type(provider) is LocalToolProvider:
                    spec = provider._specs.get(name)
                    if (
                        not self.local_native
                        or provider._native_ordinary_specs.get(name) is not spec
                    ):
                        raise PermissionError(
                            "workflow bounded native tool source required"
                        )
                    providers.setdefault(provider, []).append(name)
            self.local_providers = {
                provider: _capture_local_provider_sources(provider, names)
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

        from .console_agent_sources import _AgentSources
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
        if (
            _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
            or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
            or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
            or _storage_source_current.__kwdefaults__ is not _STORAGE_SOURCE_CHECK[3]
            or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
        ):
            raise PermissionError("original workflow storage source changed")
        _storage_source_current()
        self.source = _AgentSources(
            self.controller, self.runtime, self.bridge, self.session, configuration
        )
        self.source.workflow_preparation = self
        if (
            _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
            or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
            or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
            or _storage_source_current.__kwdefaults__ is not _STORAGE_SOURCE_CHECK[3]
            or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
        ):
            raise PermissionError("original workflow storage source changed")
        _storage_source_current(self.source)
        self.source.prepare()
        if (
            _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
            or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
            or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
            or _storage_source_current.__kwdefaults__ is not _STORAGE_SOURCE_CHECK[3]
            or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
        ):
            raise PermissionError("original workflow storage source changed")
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
                self._planning_system_source[1] or ""
                if task is None and self._planning_system_source is not None
                else ""
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
            self._planning_native_frame = (
                self.native,
                self._lifetime,
                self._lifetime.gen,
                self.native.run_primary,
                self.native.run_primary.__closure__,
            )
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
            selected = self._planning_system_source[1]
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

    async def close(self, *, _unpublished=None):
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
                        self.service.db.automatic_work.abort_workflow_planning(
                            self._planning_attempt, _prepared=self
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
        if (
            _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
            or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
            or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
            or _storage_source_current.__kwdefaults__ is not _STORAGE_SOURCE_CHECK[3]
            or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
        ):
            raise PermissionError("original workflow storage source changed")
        _storage_source_current(self.source, preparation=self)
        if self._closed:
            if _unpublished is not None:
                raise PermissionError("unpublished frame close cannot be replayed")
            return

        from .console_workflow_wave import (
            _WAVE_CLOSE_CURRENT_SOURCE,
            _WAVE_CLOSE_SOURCE,
            _workflow_close_association,
            _workflow_close_current,
        )

        close_frame = sys._getframe()
        closing_wave = self._wave
        if (
            _workflow_close_association is not _WAVE_CLOSE_SOURCE[0]
            or _workflow_close_association.__code__ is not _WAVE_CLOSE_SOURCE[1]
            or _workflow_close_association.__defaults__ is not _WAVE_CLOSE_SOURCE[2]
            or _workflow_close_association.__kwdefaults__ is not _WAVE_CLOSE_SOURCE[3]
            or _workflow_close_association.__closure__ is not _WAVE_CLOSE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        association = _workflow_close_association(self, close_frame, close_frame.f_back)

        def finish():
            if (
                _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
                or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
                or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
                or _storage_source_current.__kwdefaults__
                is not _STORAGE_SOURCE_CHECK[3]
                or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
            ):
                raise PermissionError("original workflow storage source changed")
            _storage_source_current(self.source, preparation=self)
            if (
                _workflow_close_current is not _WAVE_CLOSE_CURRENT_SOURCE[0]
                or _workflow_close_current.__code__ is not _WAVE_CLOSE_CURRENT_SOURCE[1]
                or _workflow_close_current.__defaults__
                is not _WAVE_CLOSE_CURRENT_SOURCE[2]
                or _workflow_close_current.__kwdefaults__
                is not _WAVE_CLOSE_CURRENT_SOURCE[3]
                or _workflow_close_current.__closure__
                is not _WAVE_CLOSE_CURRENT_SOURCE[5]
            ):
                raise PermissionError("original workflow helper source changed")
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
            if (
                _workflow_close_current is not _WAVE_CLOSE_CURRENT_SOURCE[0]
                or _workflow_close_current.__code__ is not _WAVE_CLOSE_CURRENT_SOURCE[1]
                or _workflow_close_current.__defaults__
                is not _WAVE_CLOSE_CURRENT_SOURCE[2]
                or _workflow_close_current.__kwdefaults__
                is not _WAVE_CLOSE_CURRENT_SOURCE[3]
                or _workflow_close_current.__closure__
                is not _WAVE_CLOSE_CURRENT_SOURCE[5]
            ):
                raise PermissionError("original workflow helper source changed")
            _workflow_close_current(association)
            with auth._lock if auth is not None else nullcontext():
                if self._wave is not closing_wave or self._launch_operation is not None:
                    raise PermissionError("original workflow close association changed")
                self._closed = True
                self._closing = False

        await self._read(finish, roots=False, closing=True, _unpublished=_unpublished)
        if (
            _workflow_close_current is not _WAVE_CLOSE_CURRENT_SOURCE[0]
            or _workflow_close_current.__code__ is not _WAVE_CLOSE_CURRENT_SOURCE[1]
            or _workflow_close_current.__defaults__ is not _WAVE_CLOSE_CURRENT_SOURCE[2]
            or _workflow_close_current.__kwdefaults__
            is not _WAVE_CLOSE_CURRENT_SOURCE[3]
            or _workflow_close_current.__closure__ is not _WAVE_CLOSE_CURRENT_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_close_current(association)
        if self._wave is None:
            self._release_configuration()

    def _close_on_native_owner(self, owner):
        """Return the original frame inside its pre-held native child operation."""
        from tldw_chatbook.Agents.agent_service import (
            _WF_OPERATION_SOURCE,
            _workflow_operation_sources,
        )

        if (
            _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
            or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
            or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
            or _storage_source_current.__kwdefaults__ is not _STORAGE_SOURCE_CHECK[3]
            or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
        ):
            raise PermissionError("original workflow storage source changed")
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
        if (
            _storage_source_current is not _STORAGE_SOURCE_CHECK[0]
            or _storage_source_current.__code__ is not _STORAGE_SOURCE_CHECK[1]
            or _storage_source_current.__defaults__ is not _STORAGE_SOURCE_CHECK[2]
            or _storage_source_current.__kwdefaults__ is not _STORAGE_SOURCE_CHECK[3]
            or _storage_source_current.__closure__ is not _STORAGE_SOURCE_CHECK[5]
        ):
            raise PermissionError("original workflow storage source changed")
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
            if (
                _workflow_operation_sources is not _WF_OPERATION_SOURCE[0]
                or _workflow_operation_sources.__code__ is not _WF_OPERATION_SOURCE[1]
                or _workflow_operation_sources.__defaults__
                is not _WF_OPERATION_SOURCE[2]
                or _workflow_operation_sources.__kwdefaults__
                is not _WF_OPERATION_SOURCE[3]
                or _workflow_operation_sources.__closure__
                is not _WF_OPERATION_SOURCE[5]
            ):
                raise PermissionError("original workflow helper source changed")
            _workflow_operation_sources(operation)
            operation.finish()
        if (
            _workflow_operation_sources is not _WF_OPERATION_SOURCE[0]
            or _workflow_operation_sources.__code__ is not _WF_OPERATION_SOURCE[1]
            or _workflow_operation_sources.__defaults__ is not _WF_OPERATION_SOURCE[2]
            or _workflow_operation_sources.__kwdefaults__ is not _WF_OPERATION_SOURCE[3]
            or _workflow_operation_sources.__closure__ is not _WF_OPERATION_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_operation_sources(operation)
        with self.authorization._lock:
            if self._close_operation is not operation:
                raise PermissionError("original workflow close operation changed")
            self._close_operation = None

    def _release_planning_slots(self):
        if (
            self.controller._active_cancel_events.get(self.session.id)
            is self.cancel_event
        ):
            self.controller._active_cancel_events.pop(self.session.id)
        if self._planning_intake is not None:
            self.store.release_received_turn(self._planning_intake)
            self._planning_intake = None
        if self._planning_token is not None:
            self.controller._fleet_wake.release_automatic_primary(
                self.session.id, self._planning_token
            )
            self._planning_token = None

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

        from . import console_ordinary_tree
        from .console_ordinary_tree import _function_unchanged, _raw_shell_metadata
        from .console_raw_cli_custody import NativeRawCliSource

        if _raw_shell_metadata is not console_ordinary_tree._NATIVE_RAW_SHELL_METADATA[
            0
        ] or not _function_unchanged(console_ordinary_tree._NATIVE_RAW_SHELL_METADATA):
            raise PermissionError("original prospective shell source changed")
        source = self.runtime._raw_cli_source
        raw, reachable, values = _raw_shell_metadata(self)
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
            or native.raw_shell_schema is not native._PLANNING_SHELL_DESCRIPTOR[0]
            or not _function_unchanged(native._PLANNING_SHELL_DESCRIPTOR)
            or native._MODEL_SCHEMA != native._PLANNING_SHELL_SCHEMA
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

    def _planning_physical_proof(self):
        caller = inspect.currentframe().f_back
        if (
            caller.f_code not in _PLANNING_TERMINAL_CALLERS
            or caller.f_locals.get("self") is not self
        ):
            raise PermissionError("original planning terminal receiver required")
        self._planning_return_source()
        owner = self.primary_owner
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
        self.source.check_memory()
        with self.capacity._lock:
            if (
                self.capacity._owners.get(owner.execution_id) is not owner
                or owner._root_finished
                or owner._operations
                or owner._cleanup_unproven
            ):
                raise RuntimeError("planning root return unproven")

    async def run_planning(self, *, _coordinator_run=None):
        """Run the same native primary and settle only after its actual returns."""
        from tldw_chatbook.Agents.activation import async_worker_guard
        from tldw_chatbook.Agents.agent_models import RUN_DONE
        from tldw_chatbook.Agents.workflow_models import WorkflowPlan
        from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

        from .console_workflow_runs import (
            _COORDINATOR_PLAN_RECORD,
            _planning_driver_source,
            _WorkflowRun,
        )

        driver = asyncio.current_task()
        coordinator = getattr(self.runtime, "_workflow_coordinator", None)
        phase, source = self.phase, self.source
        if _coordinator_run is None:
            if (
                getattr(driver.get_coro(), "cr_code", None)
                is _COORDINATOR_PLAN_RECORD[1]
                or coordinator is not None
                and any(
                    run.driver is driver or run.prepared is self
                    for run in coordinator._runs.values()
                )
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
        with self.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            if claim is None:
                raise PermissionError("planning worktree busy")
            owner = self.capacity.begin_execution(
                origin=WorkOrigin.AUTOMATIC,
                conversation_id=self.phase.attempt.conversation_id,
            )
            claim.bind_owner(owner)
            self.primary_owner, self.primary_claim = owner, claim

        def invoke():
            import os

            read_class, binding_class, binding_slot = self._planning_read_sources()
            binding = binding_slot.get()
            if (
                type(binding) is not binding_class
                or not binding.active
                or binding.process != os.getpid()
                or binding.thread is not threading.current_thread()
                or binding.task is not None
                or type(binding.read) is not read_class
                or binding.read.creator is not self.controller
                or binding.read.source is not self
                or binding.read.callback is not invoke
                or binding.read.task is not self._planning_driver
                or type(binding.read._producer) is not asyncio.Future
                or type(binding.read.retired) is not asyncio.Future
            ):
                raise PermissionError("original planning tracked-read binding required")
            self._planning_read_return = (
                binding.read,
                binding.read._producer,
                binding.read.retired,
                invoke,
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
        except asyncio.CancelledError as error:
            if not self._planning_terminal_returned:
                raise
            cancellation = error
            result = None
        self._planning_read_sources()
        read, producer, retired, callback = self._planning_read_return
        if (
            read.creator is not self.controller
            or read.source is not self
            or read.callback is not callback
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
        drain = []
        owner.on_drained(drain.append)
        owner.finish_root()
        if drain != [True]:
            raise RuntimeError("planning root retirement unproven")
        self.source.check_memory()
        self.primary_claim.settle(durable=True)
        self._planning_settled = self._closed = True
        self._release_configuration()
        self._release_planning_slots()
        if cancellation is not None:
            raise cancellation
        return result


async def prepare_workflow_reply(
    bridge, *, phase, task_id, binding, resolution, **kwargs
):
    """Private producer; the public coordinator still requires phase snapshot custody."""
    if (
        type(phase) is not WorkflowPhaseAuthorization
        or phase.context.workflow is not phase
    ):
        raise PermissionError("original workflow phase required")
    if _runtime(phase)._workflow_closing:
        raise PermissionError("workflow admission is closing")
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot

    from .console_provider_gateway import ConsoleProviderResolution

    if (
        type(binding) is not RunAdmittedWorkspaceRoot
        or type(resolution) is not ConsoleProviderResolution
    ):
        raise PermissionError(
            "original workflow binding and native resolution required"
        )
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
        if prepared.phase.attempt.purpose != "planning":
            prepared._release_configuration()
        decision = await prepared._read(
            lambda: prepared.native.service.confirm_project_instruction_dispatch(
                prepared._snapshot
            ),
            roots=False,
        )
        if decision != "proceed":
            raise PermissionError("workflow project instructions declined")
        prepared._retain_configuration()
        prepared.require_current()
    await prepared._read(prepared._capture)
    return prepared


async def _resolve_workflow_targets(controller, session, targets, *, selected=None):
    """Resolve passive references through native settings, without donating routing."""
    from dataclasses import replace

    from .console_chat_controller import build_console_provider_selection_from_settings
    from .console_received_dispatch import resolve_received_provider

    original = controller._provider_selection_for_session(session.id)
    settings = controller.store.effective_session_settings(session.id)
    results = []
    for target in targets:
        if selected is not None and goal_provider_ref(selected) == target:
            resolution = selected
        else:
            resolution = None
            if original.provider == target.provider:
                candidate = selected or await resolve_received_provider(
                    controller, original
                )
                if candidate.ready and goal_provider_ref(candidate) == target:
                    resolution = candidate
            if resolution is not None:
                results.append(resolution)
                continue
            selection = build_console_provider_selection_from_settings(
                replace(
                    settings,
                    provider=target.provider,
                    model=target.model,
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
        if not resolution.ready or goal_provider_ref(resolution) != target:
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
    from tldw_chatbook.Agents.goal_models import GoalBindingRef
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
    from tldw_chatbook.Agents.workflow_models import WorkflowRequest

    from .console_chat_models import ConsoleSubmissionOrigin
    from .console_provider_gateway import _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS
    from .console_received_dispatch import (
        resolve_received_provider,
        stock_native_methods,
    )
    from .console_workflow_runs import _accept_workflow_phase, native_workflow_qualified

    runtime = controller._hooks_v2_runtime
    from .console_workflow_runs import _PLANNING_COMMAND_CODES

    coordinator = getattr(runtime, "_workflow_coordinator", None)
    issuing = asyncio.current_task()
    if _pending_run is None:
        if issuing.get_coro().cr_code in _PLANNING_COMMAND_CODES or (
            coordinator is not None
            and any(run.issuing is issuing for run in coordinator._pending_planning)
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
        or not stock_native_methods(
            controller.provider_gateway, _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS
        )
    ):
        raise PermissionError("native workflow planning unavailable")
    if workflow_id is None and (
        not isinstance(objective, str) or type(policy) is not AutomaticWorkLimits
    ):
        raise ValueError("initial planning requires an objective and policy")
    if workflow_id is not None and (objective is not None or policy is not None):
        raise ValueError("repair cannot replace the immutable request")
    if _request is not None and (
        type(_request) is not WorkflowRequest
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
            _check_hooks(runtime, session_id)
            if not binding.guard(False):
                raise PermissionError("workflow native binding changed")
            selection = controller._provider_selection_for_session(session_id)
            resolved = await resolve_received_provider(controller, selection)
            if not resolved.ready or not resolved.model:
                raise PermissionError("workflow provider unavailable")
            if controller._provider_selection_for_session(session_id) != selection:
                raise PermissionError("workflow provider selection changed")
            prepared.resolution = prepared._planning_resolution = resolved
            prepared.planning_selection = selection
            if _request is not None:
                prepared.execution_resolutions = await _resolve_workflow_targets(
                    controller, session, _request.targets, selected=resolved
                )
            from .console_session_settings import ConsoleSessionSettings

            settings = session.settings
            if type(settings) is not ConsoleSessionSettings:
                raise PermissionError("original selected planning settings required")
            prompt = vars(settings).get("system_prompt")
            if prompt is not None and type(prompt) is not str:
                raise PermissionError("invalid selected planning system context")
            if inspect.getattr_static(settings, "system_prompt") != prompt:
                raise PermissionError(
                    "selected planning system context receiver changed"
                )
            prepared._planning_system_source = (settings, prompt)

        await prepared._read(select_native)

        def commit_request():
            # Consume the original native return. Never adopt a later current ID
            # as proof that this invocation persisted/published the chosen Chat.
            from .chat_persistence_service import (
                _NATIVE_PLANNING_PERSISTENCE_METHODS,
                ChatPersistenceService,
            )
            from .console_chat_store import (
                _NATIVE_PLANNING_PERSISTENCE_STORE,
                ConsoleChatStore,
            )
            from .console_library_policy_repository import (
                _NATIVE_FIRST_LIBRARY_REPOSITORY,
            )
            from .console_ordinary_tree import _function_unchanged

            persistence = prepared.store.persistence
            repository = persistence.console_library_policy_repository
            if (
                type(prepared.store) is not ConsoleChatStore
                or type(persistence) is not ChatPersistenceService
                or prepared.service.persistence is not persistence
                or type(repository) is not _NATIVE_FIRST_LIBRARY_REPOSITORY[0]
                or repository.db is not persistence.db
                or any(
                    inspect.getattr_static(prepared.store, name) is not method
                    or any(not _function_unchanged(record) for record in records)
                    for name, method, records in _NATIVE_PLANNING_PERSISTENCE_STORE
                )
                or any(
                    inspect.getattr_static(persistence, name) is not method
                    or not _function_unchanged(record)
                    for name, method, record in _NATIVE_PLANNING_PERSISTENCE_METHODS
                )
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
                    targets=_request.targets
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
            prepared._planning_prepare_entered = True
            attempt = prepared.service.db.automatic_work.prepare_workflow_planning(
                snapshot.id,
                owner_id=controller._fleet_wake.runtime_owner_id,
                session_id=session_id,
                command_id=command_id,
                revision_id=revision_id,
                repair_instruction=repair_instruction,
            )
            prepared._planning_attempt = attempt
            prepared.require_current()
            prepared.phase = _accept_workflow_phase(
                controller, attempt.id, session_id=session_id
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


# Definition-time originals for planning's exact receiver, not a durable capability.
from tldw_chatbook.Agents import execution_capacity as _finite_capacity

from .console_ordinary_tree import _function_record as _planning_function_record


def _finite_owner_return(owner, capacity, operation=None):
    """Observe native positive retirement without accepting replacement methods."""
    from .console_ordinary_tree import _function_unchanged

    if (
        type(owner) is not _FINITE_OWNER_CLASS
        or _finite_capacity.ExecutionOwner is not _FINITE_OWNER_CLASS
        or _finite_capacity.OwnedOperation is not _FINITE_OPERATION_CLASS
        or owner.capacity is not capacity
        or any(
            inspect.getattr_static(target, name, None) is not descriptor
            or any(not _function_unchanged(record) for record in records)
            for target, name, descriptor, records in _FINITE_OWNER_METHODS
        )
        or any(
            inspect.getattr_static(owner, name, None) is not descriptor
            for target, name, descriptor, _records in _FINITE_OWNER_METHODS
            if target is _FINITE_OWNER_CLASS
        )
    ):
        raise PermissionError("original workflow finite owner source changed")
    drained = []
    owner.on_drained(drained.append)
    if drained != [True] or owner.is_admitted:
        raise RuntimeError("original workflow finite owner unreturned")
    if operation is not None:
        if (
            type(operation) is not _FINITE_OPERATION_CLASS
            or operation.owner is not owner
            or any(
                inspect.getattr_static(operation, name, None) is not descriptor
                for target, name, descriptor, _records in _FINITE_OWNER_METHODS
                if target is _FINITE_OPERATION_CLASS
            )
        ):
            raise RuntimeError("original workflow finite operation changed")
        with capacity._lock:
            if operation in owner._operations:
                raise RuntimeError("original workflow finite operation unreturned")


_FINITE_OWNER_CLASS = _finite_capacity.ExecutionOwner
_FINITE_OPERATION_CLASS = _finite_capacity.OwnedOperation
_FINITE_OWNER_METHODS = tuple(
    (
        target,
        name,
        descriptor,
        tuple(
            _planning_function_record(function)
            for function in (
                (descriptor.fget,)
                if isinstance(descriptor, property)
                else (descriptor.__func__,)
                if isinstance(descriptor, (staticmethod, classmethod))
                else (
                    descriptor,
                    *(
                        [descriptor.__wrapped__]
                        if hasattr(descriptor, "__wrapped__")
                        else []
                    ),
                )
            )
        ),
    )
    for target in (_FINITE_OWNER_CLASS, _FINITE_OPERATION_CLASS)
    for name, descriptor in vars(target).items()
    if inspect.isfunction(descriptor)
    or isinstance(descriptor, (property, staticmethod, classmethod))
)
_FINITE_OWNER_CHECK = _planning_function_record(_finite_owner_return)

_PLANNING_ENTRY_RECORD = _planning_function_record(prepare_workflow_planning)

_PLANNING_PREPARATION_METHODS = tuple(
    (
        name,
        descriptor,
        tuple(
            _planning_function_record(function)
            for function in (
                descriptor,
                *(
                    [descriptor.__wrapped__]
                    if hasattr(descriptor, "__wrapped__")
                    else []
                ),
            )
        ),
    )
    for name, descriptor in vars(_PreparedWorkflowReply).items()
    if callable(descriptor)
)
_PLANNING_TERMINAL_CALLERS = (
    _PreparedWorkflowReply.run_planning.__code__,
    next(
        code
        for code in _PreparedWorkflowReply.run_planning.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "invoke"
    ),
)
_PLANNING_ABORT_CALLER = next(
    code
    for code in _PreparedWorkflowReply.close.__code__.co_consts
    if inspect.iscode(code) and code.co_name == "close_unused"
)
from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger as _PlanningLedger

_PLANNING_LEDGER_METHODS = tuple(
    (name, method, _planning_function_record(method))
    for name in (
        "prepare_workflow_planning",
        "abort_workflow_planning",
        "read_workflow_attempt",
        "accept_workflow_phase",
    )
    for method in (getattr(_PlanningLedger, name),)
)


from pathlib import Path as _StoragePath

from tldw_chatbook.Agents import activation as _storage_activation
from tldw_chatbook.Agents import agent_service as _storage_agent_service
from tldw_chatbook.Agents import workflow_task_scope as _storage_task_scope
from tldw_chatbook.Backup_Recovery.admission_runtime import (
    RecoveryAdmissionGuard as _StorageGuard,
)

from . import console_agent_sources as _storage_agent_sources
from .console_ordinary_tree import _function_unchanged as _storage_unchanged


def _storage_source_current(source=None, *, preparation=None):
    checker = _storage_unchanged
    if (
        checker is not _STORAGE_CHECKER[0]
        or checker.__code__ is not _STORAGE_CHECKER[1]
        or checker.__defaults__ is not _STORAGE_CHECKER[2]
        or checker.__kwdefaults__ is not _STORAGE_CHECKER[3]
        or checker.__closure__ is not _STORAGE_CHECKER[5]
        or _storage_agent_sources._AgentSources is not _STORAGE_SOURCE_CLASS
        or _storage_agent_sources._guard is not _STORAGE_GUARD
        or _storage_activation._guard is not _STORAGE_GUARD
        or type(_STORAGE_GUARD) is not _StorageGuard
        or inspect.getattr_static(_STORAGE_SOURCE_CLASS, "__getattribute__")
        is not object.__getattribute__
        or inspect.getattr_static(_STORAGE_SOURCE_CLASS, "__getattr__", None)
        is not None
        or any(
            vars(module).get(name) is not record[0] or not checker(record)
            for module, name, record in _STORAGE_SOURCE_FUNCTIONS
        )
        or any(
            inspect.getattr_static(target, name, None) is not descriptor
            or any(not checker(record) for record in records)
            or source is not None
            and target is _STORAGE_SOURCE_CLASS
            and inspect.getattr_static(source, name, None) is not descriptor
            or target is _StorageGuard
            and inspect.getattr_static(_STORAGE_GUARD, name, None) is not descriptor
            for target, name, descriptor, records in _STORAGE_SOURCE_METHODS
        )
    ):
        raise PermissionError("original workflow storage source changed")
    if preparation is not None and (
        _PreparedWorkflowReply is not _STORAGE_PREPARATION_CLASS
        or _storage_agent_service._PreparedWorkflowChild is not _STORAGE_NATIVE_CLASS
        or _storage_task_scope.WorkflowTaskAuthorization is not _STORAGE_AUTH_CLASS
        or type(preparation) is not _STORAGE_PREPARATION_CLASS
        or source is not preparation.source
        or any(
            inspect.getattr_static(preparation, name, None) is not descriptor
            or any(not checker(record) for record in records)
            for name, descriptor, records in _PLANNING_PREPARATION_METHODS
        )
    ):
        raise PermissionError("original workflow storage receiver changed")
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


_STORAGE_PREPARATION_CLASS = _PreparedWorkflowReply
_STORAGE_NATIVE_CLASS = _storage_agent_service._PreparedWorkflowChild
_STORAGE_AUTH_CLASS = _storage_task_scope.WorkflowTaskAuthorization
_STORAGE_SOURCE_CLASS = _storage_agent_sources._AgentSources
_STORAGE_GUARD = _storage_activation._guard
_STORAGE_PATH_CLASS = type(_StoragePath())
_STORAGE_CHECKER = _planning_function_record(_storage_unchanged)
_STORAGE_SOURCE_METHODS = tuple(
    (
        target,
        name,
        descriptor,
        tuple(
            _planning_function_record(function)
            for function in (
                descriptor,
                *(
                    [descriptor.__wrapped__]
                    if hasattr(descriptor, "__wrapped__")
                    else []
                ),
            )
        ),
    )
    for target, names in (
        (
            _STORAGE_SOURCE_CLASS,
            (
                "__init__",
                "prepare",
                "check_memory",
                "_check_original_sources",
                "_check_original_bindings",
                "_allowed_roots",
                "_check_writer",
                "activate",
                "check_storage",
            ),
        ),
        (_StorageGuard, ("captured_sources", "execution")),
    )
    for name in names
    for descriptor in (vars(target)[name],)
)
_STORAGE_SOURCE_FUNCTIONS = tuple(
    (module, name, _planning_function_record(vars(module)[name]))
    for module, names in (
        (_storage_activation, ("execution", "_sources")),
        (_storage_agent_sources, ("execution", "_sources")),
    )
    for name in names
)
_STORAGE_SOURCE_CHECK = _planning_function_record(_storage_source_current)
