"""Finite host custody for the Console's shared native reply producer (ADR-285)."""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

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


def current_workflow_preparation(controller=None):
    prepared = _CURRENT.get()
    if (
        prepared is None
        or controller is not None
        and prepared.controller is not controller
    ):
        return None
    return prepared


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
        self._loop = asyncio.get_running_loop()
        self._finite = None
        self._wave = None
        self.source = None
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
        self.local_native, self.local_source_values, self.local_function_values = (
            _capture_native_local_sources(self.controller, local_tool_provider)
        )
        self.require_current()

    def require_current(self):
        """Pure object/selection checks; native reads stay in the finite callback."""
        from tldw_chatbook import config

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
            or not any(s is self.session for s in self.store.sessions())
            or self.session.project_instruction_state != self.project_state
            or self.session.workspace_id != self.binding.workspace_id
            or self.service is not getattr(self.runtime, "_workflow_service", None)
        ):
            raise PermissionError("workflow native source changed")

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
        for provider, retained in self.local_providers.items():
            _check_local_provider_sources(provider, retained)

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
        ):
            raise PermissionError("original finite workflow preparation required")
        owner, claim = self._finite
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
            if goal_provider_ref(self.resolution) != self.authorization.task.target:
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

        if _unpublished is not None and not closing:
            raise PermissionError("unpublished storage proof is close-only")
        retirement = []

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
                    operation = None
                    proven = False
                    try:
                        if claim is not None:
                            claim.bind_owner(owner)
                        operation = owner.reserve_tool()
                        self._finite = (owner, claim)
                        with owner.activate(), self.activate_source():
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
                            retirement.append((owner, claim))

        def retire():
            for owner, claim in retirement:
                owner.finish_root()
                if claim is not None:
                    claim.settle(durable=True)

        def invoke():
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
                return result

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
        if metadata.trust_source is not None:
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
        self.source = _AgentSources(
            self.controller, self.runtime, self.bridge, self.session, configuration
        )
        self.source.workflow_preparation = self
        self.source.prepare()
        with self.activate_source():
            if configuration.mcp_capture_pending:
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
            mcp = await self.controller._compose_mcp_provider(
                self.session.id,
                publish_counts=False,
                profile_id_provider=lambda: configuration.tool_policy_profile_id,
                persona_policy_provider=lambda: policy,
                maximum_tool_ids=configuration.mcp_tool_maximum,
                maximum_definition_hashes=configuration.mcp_definition_maximum,
            )
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

    async def close(self, *, _unpublished=None):
        """Retire once after actual child drain and required durable publication."""
        if self._closed:
            if _unpublished is not None:
                raise PermissionError("unpublished frame close cannot be replayed")
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
            if self.native is not None and not self._native_closed:
                self._no_launch_close_guard()
                with self.activate_source():
                    self._lifetime.__exit__(None, None, None)
                self._native_closed = True
            elif self.turn_lifeline is not None:
                self.turn_lifeline.shutdown()
            with auth._lock if auth is not None else nullcontext():
                self._closed = True
                self._closing = False

        await self._read(finish, roots=False, closing=True, _unpublished=_unpublished)
        if self._wave is None:
            self._release_configuration()

    def _close_on_native_owner(self, owner):
        """Return the original frame inside its pre-held native child operation."""
        operation = self._close_operation
        if operation is None or operation.owner is not owner or not owner.is_admitted:
            raise PermissionError("original workflow close obligation required")
        # A tool may outlive its logical timeout. Its actual native thread
        # returns after source/storage and finish callbacks; do not unbind a
        # provider while that original effectful frame can still resume.
        for worker in self._tool_workers:
            worker.join()
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
            operation.finish()
        self._close_operation = None


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
    prepared = _PreparedWorkflowReply(
        bridge, phase, task_id, binding, resolution, kwargs
    )
    try:
        prepared._retain_configuration()
        await prepared._read(prepared._configure)
        await prepared._read(prepared._enter)
        if prepared._snapshot is not None:
            if prepared.native.service.confirm_project_instruction_dispatch is None:
                raise PermissionError(
                    "workflow project instruction consent unavailable"
                )
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
    except BaseException:
        await prepared.close()
        raise
