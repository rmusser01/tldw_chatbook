"""Finite host custody for the Console's shared native reply producer (ADR-285)."""

from __future__ import annotations

import time
from contextlib import nullcontext

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


class _PreparedWorkflowReply:
    """Original source and retained native frame, never restored from saved IDs."""

    def __init__(self, bridge, phase, task_id, binding, resolution, kwargs):
        self.bridge = bridge
        self.phase = phase
        self.task_id = task_id
        self.binding = binding
        self.resolution = resolution
        self.controller = phase._controller
        self.runtime = _runtime(phase)
        self.store = self.controller.store
        self.session = phase._session
        self.project_state = self.session.project_instruction_state
        self.service = self.runtime.ensure_workflow_service()
        self.capacity = self.runtime.execution_capacity
        self.kwargs = kwargs
        self.should_cancel = kwargs.pop(
            "should_cancel", self.controller._shutdown_requested.is_set
        )
        if not callable(self.should_cancel):
            raise TypeError("native cancellation callback required")
        self.authorization = None
        self.native = None
        self.turn_lifeline = None
        self._lifetime = None
        self._closed = False
        self._closing = False
        self._snapshot = None
        self.require_current()

    def require_current(self):
        """Pure object/selection checks; native reads stay in the finite callback."""
        if (
            self.runtime.agent_bridge is not self.bridge
            or self.runtime.chat_controller is not self.controller
            or self.controller._agent_bridge is not self.bridge
            or self.controller.store is not self.store
            or self.bridge._store is not self.store
            or self.controller._disposed
            or self.controller._shutdown_requested.is_set()
            or self.should_cancel()
            or self.runtime._disposed
            or not any(s is self.session for s in self.store.sessions())
            or self.session.project_instruction_state != self.project_state
            or self.session.workspace_id != self.binding.workspace_id
            or self.service is not getattr(self.runtime, "_workflow_service", None)
        ):
            raise PermissionError("workflow native source changed")

    def check(self):
        self.require_current()
        self.phase.check()
        _check_hooks(self.runtime, self.session.id)
        if not self.binding.guard(False):
            raise PermissionError("workflow native binding changed")
        if self.authorization is not None:
            self.authorization.check_preparation()
            if goal_provider_ref(self.resolution) != self.authorization.task.target:
                raise PermissionError("workflow native target changed")

    def _check_task_source(self, authorization):
        if self._closed or self._closing or authorization is not self.authorization:
            raise PermissionError("workflow native preparation retired")
        self.require_current()
        if goal_provider_ref(self.resolution) != authorization.task.target:
            raise PermissionError("workflow native target changed")

    async def _read(self, callback, *, roots=True, closing=False):
        """Own a complete finite callback; no native scope survives its return."""

        def invoke():
            with owned_goal_storage(self.service), execution(self.bridge):
                if closing:
                    return callback()
                with self.phase.context.scope():
                    self.check()
                    # shortcut: finite preflight serializes against all root writes;
                    # batch eligible preparations before launching disjoint tasks.
                    footprint = (
                        self.service._preparation_read_footprint(
                            self.phase, self.task_id, self.binding
                        )
                        if roots
                        else None
                    )
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
                        try:
                            if claim is not None:
                                claim.bind_owner(owner)
                            with owner.activate():
                                if (
                                    roots
                                    and footprint
                                    != self.service._preparation_read_footprint(
                                        self.phase, self.task_id, self.binding
                                    )
                                ):
                                    raise ValueError(
                                        "freshness_source_identity_changed"
                                    )
                                result = callback()
                                self.check()
                                return result
                        finally:
                            owner.finish_root()
                            if claim is not None:
                                claim.settle(durable=True)

        return await run_preparation_read(
            invoke,
            creator=self.controller,
            session_id=self.session.id,
            reads=self.controller._preparation_reads,
            observers=(self.runtime._preparation_reads,),
            require_current=(lambda: None) if closing else self.require_current,
            source=self,
        )

    def bind_catalog(self, catalog):
        """Consume the one real composition before building the task-only plan."""
        registry, allowed, builtin, local, skills = catalog
        self.authorization = _prepare_workflow_task(
            self.phase,
            task_id=self.task_id,
            registry=registry,
            binding=self.binding,
        )
        self.authorization._bridge_preparation.append(self)
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

    def _enter(self):
        with self.phase._task_scope_lock:
            if self.task_id in self.phase._task_scopes:
                raise PermissionError("workflow task already prepared")
        saved = self.service.get(self.phase.attempt.workflow_id)
        task = next(
            task
            for phase in saved.plans[self.phase.attempt.revision_id - 1].phases
            if phase.id == self.phase.attempt.phase_id
            for task in phase.tasks
            if task.id == self.task_id
        )
        if (
            goal_provider_ref(self.resolution) != task.target
            or not self.resolution.ready
        ):
            raise PermissionError("workflow native target changed")
        kwargs = dict(self.kwargs)
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
            assistant_message_id="",
            model=self.resolution.model,
            session_system_prompt="",
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
        self.native.service._prepare_workflow_child(
            authorization=self.authorization,
            first_request_plan=plan,
            target_ref=goal_provider_ref(self.resolution),
            startup_instruction_snapshot=self._snapshot,
        )

    async def close(self):
        """Retire once after actual child drain and required durable publication."""
        if self._closed:
            return

        def finish():
            auth = self.authorization
            with auth._lock if auth is not None else nullcontext():
                if self._closed:
                    return
                if self._closing:
                    raise PermissionError("workflow native retirement already selected")
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
            if self.native is not None:
                self._lifetime.__exit__(None, None, None)
            elif self.turn_lifeline is not None:
                self.turn_lifeline.shutdown()
            with auth._lock if auth is not None else nullcontext():
                self._closed = True
                self._closing = False

        await self._read(finish, roots=False, closing=True)


async def prepare_workflow_reply(
    bridge, *, phase, task_id, binding, resolution, **kwargs
):
    """Private producer; the public coordinator still requires phase snapshot custody."""
    if (
        type(phase) is not WorkflowPhaseAuthorization
        or phase.context.workflow is not phase
    ):
        raise PermissionError("original workflow phase required")
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
    if (
        bridge.change_tracking_enabled
        or bridge._change_finalization_coordinator is not None
    ):
        raise PermissionError("workflow phase snapshot ownership required")
    prepared = _PreparedWorkflowReply(
        bridge, phase, task_id, binding, resolution, kwargs
    )
    try:
        await prepared._read(prepared._enter)
        if prepared._snapshot is not None:
            if prepared.native.service.confirm_project_instruction_dispatch is None:
                raise PermissionError(
                    "workflow project instruction consent unavailable"
                )
            decision = await prepared._read(
                lambda: prepared.native.service.confirm_project_instruction_dispatch(
                    prepared._snapshot
                ),
                roots=False,
            )
            if decision != "proceed":
                raise PermissionError("workflow project instructions declined")
        await prepared._read(prepared._capture)
        return prepared
    except BaseException:
        await prepared.close()
        raise
