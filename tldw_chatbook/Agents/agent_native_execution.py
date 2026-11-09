"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from typing import Literal

from . import agent_service as native_host


def _settle_review_capture(
    self,
    service,
    run_id: str,
    receipt_id: str | None = None,
    *,
    close_capture: bool = True,
) -> None:
    """Close only an actually associated attempt; never raises (see helper)."""
    from .agent_request_execution import settle_review_capture

    settle_review_capture(
        self, service, run_id, receipt_id, close_capture=close_capture
    )


def _fork_review_child(
    self, service, *, actor, max_tool_result_chars: int, grant=None
) -> native_host.AgentService:
    """Retain execution owners, with private actor/catalog state for this run."""
    from copy import copy

    fork = copy(self)
    # Request construction writes this cache; keep child schemas private and
    # independent of a later primary turn clearing its own protocol cache.
    fork._tool_protocol_cache = {}
    fork.registry = self.registry.for_review_child(
        service.for_run(actor, grant=grant, max_tool_result_chars=max_tool_result_chars)
    )
    return fork


def _check_workflow_launch_authorization(self, authorization):
    from .workflow_task_scope import WorkflowTaskAuthorization

    if type(authorization) is not WorkflowTaskAuthorization:
        raise PermissionError("original workflow task custody required")
    authorization.check_preparation()
    if (
        authorization.phase._task_scopes.get(authorization.task.id) is not authorization
        or authorization.phase.context is not self._automatic_work
        or authorization.phase._ledger._db is not self.db
        or authorization.registry is not self.registry
        or authorization._runtime.execution_capacity is not self.runtime_capacity
        or self.guard_tool_calls is not None
        or self._hooks_v2_engine is not None
        or self._post_tool_call is not None
    ):
        raise PermissionError("native workflow preparation owner mismatch")


def _prepare_workflow_child(
    self,
    *,
    authorization,
    first_request_plan,
    target_ref,
    parent_owner: native_host.ExecutionOwner | None = None,
    startup_instruction_snapshot: native_host.InstructionSnapshot | None = None,
    continuation_durability: Literal["persistent", "ephemeral"] = "persistent",
) -> None:
    """Capture the bridge's actual native preparation on original task custody.

    Task8 supplies the genuine first-request plan and fully wired service.
    This prepares no provider call, capacity, run row or execution authority.
    """
    from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
        normalize_generic_endpoint_for_compare,
    )

    from .agent_routing import SpawnTarget
    from .agent_service import ConsoleFirstRequestPlan
    from .execution_capacity import ExecutionOwner
    from .goal_models import GoalProviderRef
    from .run_log import RunLogWriter

    self._check_workflow_launch_authorization(authorization)
    if (
        type(first_request_plan) is not ConsoleFirstRequestPlan
        or type(target_ref) is not GoalProviderRef
    ):
        raise PermissionError("native first-request preparation required")
    config = first_request_plan.config
    if (
        first_request_plan.registry is not self.registry
        or target_ref != authorization.task.target
        or (config.provider, config.model) != (target_ref.provider, target_ref.model)
        or (
            normalize_generic_endpoint_for_compare(config.base_url)
            or "provider-default"
        )
        != target_ref.endpoint_ref
        or self._injected_fleet_coordinator is None
    ):
        raise PermissionError("native workflow target or fleet mismatch")
    # The real coordinator supplies task-only, bounded handoffs, never the
    # parent conversation. JSON detaches nested rows without duplicating prompts.
    if (
        sum(
            row.get("role") == "user"
            and row.get("content") == authorization.task.instructions
            for row in first_request_plan.messages
        )
        != 1
    ):
        raise PermissionError("native task seed must retain approved instructions once")
    messages_json = native_host.json.dumps(
        first_request_plan.messages, ensure_ascii=False, allow_nan=False
    )
    for tracker in authorization._required_evidence:
        tracker.service.validate_task_inputs(authorization, first_request_plan.messages)
    parent_run_id = None
    if parent_owner is not None:
        if (
            type(parent_owner) is not ExecutionOwner
            or parent_owner.capacity is not self.runtime_capacity
            or parent_owner.conversation_id != authorization.claim.conversation_id
            or parent_owner.run_id is None
        ):
            raise PermissionError("genuine native parent owner required")
        parent_run_id = parent_owner.run_id
        parent = self.db.get_run(parent_run_id)
        if (
            parent is None
            or parent["conversation_id"] != authorization.claim.conversation_id
            or parent["work_chain_id"] != authorization.claim.chain_id
        ):
            raise PermissionError("native workflow parent lineage mismatch")
    bridge_preparations = authorization._bridge_preparation
    preparation = bridge_preparations[0] if bridge_preparations else None
    source_observation = None
    if preparation is not None:
        native_host._workflow_launch_sources(self)
        if (
            len(bridge_preparations) != 1
            or preparation.authorization is not authorization
            or preparation.phase is not authorization.phase
        ):
            raise PermissionError("original workflow preparation source changed")
        preparation._execution_return_source()
        source_observation = preparation._configured_storage_sources()
    with self._workflow_preparation_lock, authorization._lock:
        if (
            authorization._native_preparation
            or authorization._binding_state
            or self._workflow_authorization is not None
        ):
            raise PermissionError("workflow child already prepared")
        writer = (
            self._injected_run_log_writer
            if self._injected_run_log_writer is not None
            else RunLogWriter()
        )
        # The bridge owns this service per native run tree, as in run_turn.
        self.run_log_writer = writer
        self._fleet = self._injected_fleet_coordinator
        self._turn_definitions = list(
            first_request_plan.schemas.agent_definitions or ()
        )
        target = SpawnTarget(
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            params=config.sampling_params,
            source="inherit",
            execution_provider=config.execution_provider
            or first_request_plan.api_endpoint,
        )
        kwargs, _budget = self._compose_native_child(
            authorization.task.instructions,
            config=config,
            api_endpoint=first_request_plan.api_endpoint,
            parent_raw_provider=first_request_plan.selected_provider,
            fleet=self._fleet,
            inline=False,
            started=self.clock(),
            resolved=None,
            target=target,
            fallback_targets=config.preset_fallback_targets,
            child_execution_provider=target.execution_provider,
            allowed_tools=tuple(
                n for n in config.allowed_tools if n in authorization.task.allowed_tools
            ),
            isolation=None,
            run_id=parent_run_id,
            conversation_id=authorization.claim.conversation_id,
            continuation_durability=continuation_durability,
            chain_id=authorization.claim.chain_id,
            writer=writer,
            spawn_step_index=None,
            child_sequence=authorization.task.id,
            startup_instruction_snapshot=startup_instruction_snapshot,
        )
        kwargs["requested_run_id"] = authorization.claim.launch_run_id
        kwargs["spawn_parent_event_id"] = (
            f"agent-run:{parent_run_id}" if parent_run_id is not None else None
        )
        kwargs.pop("messages")
        if (
            authorization._bridge_preparation is not bridge_preparations
            or (preparation is None and bridge_preparations)
            or preparation is not None
            and (
                len(bridge_preparations) != 1
                or bridge_preparations[0] is not preparation
                or preparation.authorization is not authorization
                or preparation.phase is not authorization.phase
                or preparation._configured_source_observation is not source_observation
                or preparation.source is not source_observation[1]
                or preparation.source.sources is not source_observation[2]
            )
        ):
            raise PermissionError("original workflow storage observation changed")
        authorization._native_preparation.append(
            native_host._PreparedWorkflowChild(
                service=self,
                phase=authorization.phase,
                kwargs=native_host.MappingProxyType(kwargs),
                messages_json=messages_json,
                fleet=self._fleet,
                parent_run_id=parent_run_id,
                roster=tuple(first_request_plan.schemas.agent_definitions or ()),
                target_ref=target_ref,
                cancels=self._fleet_cancels,
                threads=self._fleet_threads,
                source_observation=source_observation,
            )
        )
        self._workflow_authorization = authorization


def launch_workflow_task(
    self,
    task: native_host.WorkflowTaskSpec,
    *,
    authorization: native_host.WorkflowTaskAuthorization,
    should_cancel: native_host.Callable[[], bool],
) -> native_host.WorkflowChildReceipt:
    """Return native custody after this call's own core and guard scopes.

    Caller-owned outer storage, phase and worker scopes remain independent.
    """
    native_host._workflow_launch_sources(self)
    result = self._workflow_launch_result(
        task, authorization=authorization, should_cancel=should_cancel
    )
    self._finish_workflow_launch(result)
    if result.error is not None:
        raise result.error
    return result.receipt


def _workflow_launch_result(self, task, *, authorization, should_cancel, _wave=None):
    """Produce only this invocation's local value after normal own return."""
    from tldw_chatbook.Backup_Recovery.participants import core_operation

    from .activation import execution
    from .execution_capacity import (
        CapacityRefused,
        require_workflow_capacity,
        require_workflow_operation,
    )

    native_host._workflow_launch_sources(self)
    self._check_workflow_launch_authorization(authorization)
    if task is not authorization.task:
        raise PermissionError("original prepared workflow task required")
    with authorization._lock:
        native_list = authorization._native_preparation
        bridge_list = authorization._bridge_preparation
        if (
            not native_list
            or native_list[0].service is not self
            or native_list[0].phase is not authorization.phase
        ):
            raise PermissionError("native workflow child preparation required")
        native = native_list[0]
        preparation = bridge_list[0] if bridge_list else None
        wave = preparation._wave if preparation is not None else None
    if wave is not _wave:
        raise PermissionError("registered workflow wave required")
    mode = "N" if preparation is None else "W" if wave is not None else "D"
    entry = native_host._WorkflowLaunchEntry(
        self,
        authorization,
        native,
        preparation,
        wave,
        mode,
        authorization._lock,
        authorization._native_launch,
        native_list,
        bridge_list,
        native.wait_return,
        self.runtime_capacity._lock,
    )
    native_host._workflow_launch_current(entry)
    with authorization._lock:
        native_host._workflow_launch_locked(entry)
        with authorization.phase._control_lock:
            if authorization.phase._retired:
                raise PermissionError("workflow phase retired")
            if entry.marker:
                receipt = entry.marker[0]
                if not isinstance(receipt, native_host.WorkflowChildReceipt):
                    raise PermissionError("workflow child launch already claimed")
                error = (
                    native_host._WorkflowChildLaunchFailed(receipt)
                    if receipt.failure is not None
                    else None
                )
                return native_host.WorkflowLaunchReturn(
                    entry, receipt=receipt, error=error, duplicate=True
                )
            if preparation is not None and (
                preparation._launch_operation is not None
                or preparation._close_operation is not None
                or preparation._closed
                or preparation._closing
                or mode == "D"
                and preparation.bridge.change_tracking_enabled
            ):
                raise PermissionError("original capture wave required")
            if native.uncertain_start or entry.waits and entry.waits[0] is None:
                raise PermissionError("original workflow launch return unproven")
            entry.waits.clear()
            entry.marker.append(None)
            native.launch_entry[:] = [entry]

    @core_operation(self.db)
    def launch():
        owner = close_operation = operation = operations = None
        error = receipt = None
        wait = False
        known = True
        bound = False
        with authorization._runtime.worktree_activity.prelaunch(
            footprint=authorization.physical_footprint,
            _wave_member=authorization,
        ) as claim:
            if claim is None:
                error = native_host.WorkflowChildWait(
                    "worktree_activity", "workflow worktree activity busy"
                )
                wait = True
            else:
                try:
                    owner = self.runtime_capacity.begin_execution(
                        origin=self.work_origin,
                        conversation_id=authorization.claim.conversation_id,
                        child=True,
                    )
                except CapacityRefused as refused:
                    if not native_host._workflow_capacity_wait(
                        entry, refused, None, ()
                    ):
                        raise
                    error, wait = refused, True
                if owner is not None:
                    claim.bind_owner(owner)
                    native.activity_claims.append(claim)
                    bound = True
                    require_workflow_capacity(self.runtime_capacity, owner)
                    with self.runtime_capacity._lock:
                        operations = owner._operations
                    with owner.activate():
                        for tracker in authorization._required_evidence:
                            validation = tracker.service._task_input_result(
                                authorization,
                                native_host.json.loads(native.messages_json),
                            )
                            if validation.error is not None:
                                if not validation.conclusive:
                                    raise validation.error
                                error = validation.error
                                break
                        if error is None and preparation is not None:
                            try:
                                require_workflow_capacity(
                                    self.runtime_capacity, owner
                                )
                                close_operation = owner.reserve_tool()
                                with authorization._lock:
                                    native_host._workflow_launch_locked(entry)
                                    preparation._close_operation = close_operation
                                require_workflow_capacity(
                                    self.runtime_capacity, owner
                                )
                                operation = owner.reserve_tool()
                                with authorization._lock:
                                    native_host._workflow_launch_locked(entry)
                                    preparation._launch_operation = operation
                            except CapacityRefused as refused:
                                if not native_host._workflow_capacity_wait(
                                    entry,
                                    refused,
                                    owner,
                                    (close_operation, operation),
                                    operations,
                                ):
                                    raise
                                error, wait = refused, True
                    if error is None:
                        kwargs = dict(native.kwargs)
                        kwargs["messages"] = native_host.json.loads(
                            native.messages_json
                        )
                        kwargs["workflow_task_authorization"] = authorization
                        kwargs["workflow_activity_claim"] = claim
                        try:
                            result = (
                                self._launch_prepared_child_result
                                if preparation is not None
                                else self._launch_prepared_child
                            )(
                                task.instructions,
                                None,
                                kwargs,
                                fleet=native.fleet,
                                run_id=native.parent_run_id,
                                message_chain_id=authorization.claim.chain_id,
                                hook_run=None,
                                should_cancel=should_cancel,
                                my_handle_ids=native.handle_ids,
                                durable_handle_ids=native.durable_ids,
                                fleet_cancels=native.cancels,
                                fleet_threads=native.threads,
                                child_owner=owner,
                                child_reservation=authorization.claim.reservation_id,
                                child_outlives_turn=False,
                                _workflow_entry=entry,
                            )
                        except native_host._WorkflowChildLaunchFailed as failed:
                            if preparation is not None:
                                raise
                            receipt, error = failed.receipt, failed
                            known = receipt.failure != "unknown"
                        else:
                            if type(result) is native_host._NativeWorkflowReturn:
                                receipt, error, known = (
                                    result.receipt,
                                    result.error,
                                    result.known,
                                )
                            else:
                                handle, failure = result
                                if isinstance(failure, native_host._FleetAdmissionWait):
                                    error = native_host.WorkflowChildWait(
                                        "fleet_capacity", failure.error
                                    )
                                    wait = True
                                elif failure is not None:
                                    error = PermissionError(failure.error)
                                else:
                                    receipt = native_host.WorkflowChildReceipt(
                                        authorization.claim.id,
                                        authorization.claim.launch_run_id,
                                        handle.handle_id,
                                        owner,
                                        claim,
                                    )
                                    with authorization._lock:
                                        native_host._workflow_launch_locked(entry)
                                        entry.marker[0] = receipt
        # Original owner/activity scopes return before undispatched cleanup.
        if receipt is None:
            if close_operation is not None:
                native_host._workflow_launch_current(entry)
                require_workflow_operation(close_operation)
                close_operation.finish()
                require_workflow_operation(close_operation)
                with authorization._lock:
                    native_host._workflow_launch_locked(entry)
                    if preparation._close_operation is not close_operation:
                        raise PermissionError(
                            "original workflow close operation changed"
                        )
                    preparation._close_operation = None
            if owner is not None:
                require_workflow_capacity(self.runtime_capacity, owner)
                owner.finish_root()
            if bound:
                claim.settle(durable=True)
                native.activity_claims.clear()
            if wait:
                with self.runtime_capacity._lock:
                    live = not self.runtime_capacity._closed
                with native.fleet._lock:
                    live = live and not native.fleet._fenced
                if not live:
                    wait = False
                else:
                    with authorization._lock:
                        native_host._workflow_launch_locked(entry)
                        entry.waits[:] = [None]
        return native_host.WorkflowLaunchReturn(
            entry, owner, claim, operation, receipt, error, wait, known
        )

    native_host._workflow_launch_current(entry)
    observations = entry.native.source_observation[2] if entry.mode == "D" else ()
    with execution(self, sources=observations):
        result = launch()
        if result.error is not None and result.receipt is not None:
            self._finalize_workflow_child_log(
                authorization, status=native_host.RUN_ERROR
            )
    # Register the result only after this launch's own scopes returned.
    native_host._workflow_launch_current(entry)
    with entry.lock:
        entry.pending.append(result)
    return result


def _finish_workflow_launch(self, result):
    """Consume only the original invocation's actual own/wider return."""
    from .execution_capacity import require_workflow_capacity, require_workflow_operation

    if (
        type(result) is not native_host.WorkflowLaunchReturn
        or result.entry.service is not self
    ):
        raise PermissionError("original workflow launch result required")
    entry = result.entry
    native_host._workflow_launch_current(entry)
    if result.duplicate:
        return
    with entry.lock:
        if (
            len(entry.native.launch_entry) != 1
            or entry.native.launch_entry[0] is not entry
            or len(entry.pending) != 1
            or entry.pending[0] is not result
        ):
            raise PermissionError("registered native launch result required")
    if not result.known or entry.native.uncertain_start:
        if result.error is not None:
            raise result.error
        raise PermissionError("original workflow launch return unproven")
    operation = result.operation
    if operation is not None:
        require_workflow_operation(operation)
        with entry.lock:
            native_host._workflow_launch_locked(entry)
            if entry.preparation._launch_operation is not operation:
                raise PermissionError("original workflow launch operation changed")
        owner = result.owner
        with owner.capacity._lock:
            if (
                operation.owner is not owner
                or owner.capacity is not self.runtime_capacity
                or owner.capacity._owners.get(owner.execution_id) is not owner
                or operation not in owner._operations
            ):
                raise PermissionError("original workflow launch membership changed")
        operation.finish()
        require_workflow_operation(operation)
        native_host._workflow_launch_current(entry)
        with entry.lock:
            native_host._workflow_launch_locked(entry)
            if entry.preparation._launch_operation is not operation:
                raise PermissionError("original workflow launch operation changed")
            entry.preparation._launch_operation = None
    if result.receipt is None:
        if result.owner is not None:
            require_workflow_capacity(self.runtime_capacity, result.owner)
            drains = []
            result.owner.on_drained(drains.append)
            if drains != [True]:
                raise PermissionError("workflow launch cleanup return unproven")
        activity = entry.authorization._runtime.worktree_activity
        with activity._lock:
            claim = result.claim
            if claim is not None and (
                claim.id in activity._claims or claim.id in activity._members
            ):
                raise PermissionError("workflow launch activity return unproven")
        native_host._workflow_launch_current(entry)
        with entry.lock:
            native_host._workflow_launch_locked(entry)
            if entry.marker != [None] or entry.authorization._native_dispatch:
                raise PermissionError("workflow undispatched launch changed")
            if result.wait:
                entry.waits[:] = [(result.error, result.owner, result.claim)]
            entry.marker.clear()
    with entry.lock:
        entry.pending.clear()
        entry.native.launch_entry.clear()


def _finalize_workflow_child_log(
    self, authorization, *, status: str, total_tokens: int = 0
) -> None:
    with authorization._lock:
        if not authorization._native_preparation:
            return
        prepared = authorization._native_preparation[0]
        if (
            prepared.service is not self
            or prepared.parent_run_id is not None
            or prepared.log_finalized
        ):
            return
        prepared.log_finalized.append(True)
    kwargs = prepared.kwargs
    try:
        self._finalize_run_log(
            kwargs.get("run_log_writer"),
            run_id=authorization.claim.launch_run_id,
            config=kwargs["config"],
            api_endpoint=kwargs["api_endpoint"],
            status=status,
            total_tokens=total_tokens,
        )
    except Exception:  # noqa: BLE001 - optional logs cannot break settlement
        native_host.logger.warning("could not finalize workflow child run log")


def _finalize_run_log(
    writer,
    *,
    run_id,
    config,
    api_endpoint,
    status,
    total_tokens,
    superseded_run_id="",
) -> None:
    if writer is None:
        return
    from .native_source_storage import console_writer_source

    # Optional logging cannot invoke a replacement callback on the original
    # Console writer, including when finalizing a refused child.
    try:
        console_writer_source(writer)
    except RuntimeError:
        return
    writer.write_manifest(
        {
            "run_id": run_id,
            "model": config.model,
            "api_endpoint": api_endpoint,
            "allowed_tools": list(config.allowed_tools),
            "budget": native_host.dataclasses.asdict(config.budget),
            "status": status,
            "superseded_run_id": superseded_run_id,
            "total_tokens": total_tokens,
        }
    )
    console_writer_source(writer)
    writer.close()


def _compose_native_child(
    self,
    spawn_task,
    *,
    config,
    api_endpoint,
    parent_raw_provider,
    fleet,
    inline,
    started,
    resolved,
    target,
    fallback_targets,
    child_execution_provider,
    allowed_tools,
    isolation,
    run_id,
    conversation_id,
    continuation_durability,
    chain_id,
    writer,
    spawn_step_index,
    child_sequence,
    startup_instruction_snapshot,
    review_claim=None,
):
    """Compose the existing child budget, instructions and frozen route once."""
    turn_scoped = fleet is None or inline
    if turn_scoped:
        child_budget = native_host.clamp_child_budget(
            config.budget,
            parent_remaining_seconds=(
                config.budget.max_wall_seconds - (self.clock() - started)
            ),
        )
    else:
        child_max_wall_seconds = native_host._coerce_child_max_wall_seconds(
            native_host._setting(
                native_host.CHILD_MAX_WALL_SECONDS_KEY,
                native_host.DEFAULT_CHILD_MAX_WALL_SECONDS,
            )
        )
        child_budget = native_host.contain_child_budget(
            config.budget, child_max_wall_seconds
        )
    if resolved is not None and resolved.max_wall_seconds is not None:
        child_budget = native_host.dataclasses.replace(
            child_budget,
            max_wall_seconds=min(
                child_budget.max_wall_seconds, resolved.max_wall_seconds
            ),
        )
    # Q6/Task-12: an explicit override (a skill's own narrowed
    # allow-list -- builtins + local tool names, intersect-only so
    # a skill narrows but never grants; see SkillRunner.run)
    # replaces the default entirely; the default itself preserves
    # the shipped
    # behavior (spawn_subagent's child inherits the parent's
    # allow-list minus the spawn tool itself, so a depth-1 child
    # never re-offers spawn_subagent) -- MINUS any skill-tool names
    # too (pre-merge review MINOR 3). An ordinary native-spawn
    # child can never actually run a skill (max_subagents is
    # always clamped to 0 for every child, one-deep-only by
    # construction), so a skill name surviving into its allow-list
    # only meant a call to it fell through to that numeric
    # budget-exhausted refusal below instead of the permission
    # gate every other disallowed tool hits -- fragile (an
    # incidental side effect of the budget clamp, not a modeled
    # boundary) and inconsistent with the skill-driven child's own
    # explicit narrowed allow-list. Excluding skill names
    # here too means a child can neither discover (find_tools/
    # disclosure) nor invoke one; a stray direct call still gets a
    # graceful "Tool not permitted" ToolResult from invoke_tool's
    # skill branch, never reaching skill_runner.run.
    child_allowed_tools = (
        allowed_tools
        if allowed_tools is not None
        else tuple(
            n
            for n in config.allowed_tools
            if n != native_host.SPAWN_TOOL_NAME
            and not (
                self.skill_runner is not None and self.skill_runner.is_skill_tool(n)
            )
        )
    )
    if isolation == "worktree":
        # Finding 6 (Qodo round, TASK-28238 P2 T7): worktree
        # routing only ever covers `_PATH_AUTHORITY_LOCAL_NAMES`
        # -- `shell_exec` and `virtual_cli` execute against the
        # ORIGINAL workspace root regardless (virtual CLI binds
        # its `WorkspaceToolExecutor` at compose time; raw shell
        # runs real commands there too). Applied AFTER the
        # override branch above, not only inside the default
        # composition, so an explicit `allowed_tools=` override
        # can never reintroduce either name for an isolated
        # child -- the exclusion is unconditional whenever
        # isolation is requested.
        from .raw_shell_tool_provider import RAW_SHELL_TOOL_NAME
        from .virtual_cli_provider import VIRTUAL_CLI_TOOL_NAME

        child_allowed_tools = tuple(
            n
            for n in child_allowed_tools
            if n not in (RAW_SHELL_TOOL_NAME, VIRTUAL_CLI_TOOL_NAME)
        )
    child_system_prompt = native_host.get_internal_prompt("agents.subagent_system")
    # The resolver owns model/provider selection now (ADR-344): the
    # legacy inline `if resolved.model: child_model = ...` preset
    # override lives INSIDE resolve_spawn_target, so a preset model
    # with no provider keeps its same-endpoint behavior there.
    child_model = target.model or config.model
    plugin_instructions = ()
    if resolved is not None:
        from .agent_models import compose_agent_instructions

        child_system_prompt, child_allowed_tools, plugin_instructions = (
            compose_agent_instructions(
                resolved, child_system_prompt, child_allowed_tools
            )
        )
    if review_claim is not None:
        child_allowed_tools = tuple(
            name
            for name in child_allowed_tools
            if name in {"code_review_read", "code_review_submit"}
        )
        plugin_instructions = ()
    child_config = native_host.AgentConfig(
        review_binding=review_claim.binding if review_claim else None,
        model=child_model,
        # Children report provider-level faults against the same
        # provider the parent is using (review minor, 2026-08-31 --
        # previously they reported "unknown provider").
        provider=target.provider,
        execution_provider=child_execution_provider,
        preset_fallback_targets=fallback_targets,
        system_prompt=child_system_prompt,
        allowed_tools=child_allowed_tools,
        budget=child_budget,
        native_tools=config.native_tools,
        reasoning_replay=(
            config.reasoning_replay
            if child_model == config.model
            and target.provider == (parent_raw_provider or api_endpoint)
            else None
        ),
        # A sub-agent operates on the same workspace roots as its
        # parent, so it inherits the same environment note verbatim
        # (appended to its own prompt in call_model, after its identity
        # prefix). Empty for the default workspace, so no change there.
        workspace_context_note="" if review_claim else config.workspace_context_note,
        personal_context_block="" if review_claim else config.personal_context_block,
        response_reserve_tokens=config.response_reserve_tokens,
        # The resolved target's own endpoint override and sampling
        # params (ADR-344) -- empty/None on the plain inherit path's
        # built-in providers, populated for routed children.
        base_url=target.base_url,
        sampling_params=target.params,
    )
    spawn_event_id = (
        self._resolved_control_event_id(
            run_id, spawn_step_index, native_host.STEP_SPAWN
        )
        or self._resolved_control_event_id(
            run_id, spawn_step_index, native_host.STEP_TOOL_CALL
        )
        if spawn_step_index is not None
        else None
    )
    # C1: snapshot/restore whatever review_state_scope owns (see
    # __init__'s own comment) around the ENTIRE nested run -- the
    # child's own turns must never be able to leave the parent's
    # per-turn review state (e.g. MCPToolProvider._stamped_
    # decisions) mutated once control returns here. A no-op
    # contextlib.nullcontext() when no scope was wired (every
    # non-MCP run, and every caller before this task).
    #
    # PR2a Task 5: scoped to THIS (the PARENT's) run id -- it is
    # the parent's own already-decided verdicts a nested run must
    # not disturb, and the child's id does not exist yet here
    # anyway (`_run_one` mints it). No longer the load-bearing
    # protection: both gates key verdicts by run, so the child
    # writes to its own slice and cannot reach this one. Retained
    # as belt-and-braces for the inline path.
    child_kwargs = dict(
        conversation_id=conversation_id,
        messages=[
            *plugin_instructions,
            {"role": "user", "content": spawn_task},
        ],
        config=child_config,
        api_endpoint=target.provider,
        # ADR-344 snapshot (schema v21): the resolved target frozen
        # onto the child run row so resume/continuation (Task 8)
        # reuses IT rather than live re-resolution. Params persist
        # as JSON only when the resolver produced any.
        fallback_targets_json=(
            native_host.json.dumps(
                [
                    {
                        "provider": target.provider,
                        "model": child_model,
                        "base_url": target.base_url,
                        "params_json": native_host.json.dumps(
                            native_host.params_to_dict(target.params)
                        ),
                        "execution_provider": child_execution_provider,
                    },
                    *[
                        {
                            "provider": candidate.provider,
                            "model": candidate.model,
                            "base_url": candidate.target.base_url
                            if candidate.target
                            else None,
                            "params_json": native_host.json.dumps(
                                native_host.params_to_dict(candidate.target.params)
                            )
                            if candidate.target
                            else "{}",
                            "execution_provider": candidate.target.execution_provider
                            if candidate.target
                            else "",
                            "ready": candidate.ready,
                            "skip_reason": candidate.skip_reason,
                        }
                        for candidate in fallback_targets
                    ],
                ]
            )
            if fallback_targets
            else None
        ),
        resolved_provider=target.provider,
        resolved_model=child_model,
        resolved_base_url=target.base_url,
        resolved_params_json=(
            native_host.json.dumps(native_host.params_to_dict(target.params))
            if target.params
            else None
        ),
        agent_kind=native_host.AGENT_KIND_SUBAGENT,
        task=f"Review {review_claim.review_id} slot {review_claim.slot.slot_id}"
        if review_claim
        else spawn_task,
        parent_run_id=run_id,
        spawn_event_id=spawn_event_id,
        spawn_parent_event_id=spawn_event_id or f"agent-run:{run_id}",
        agent_definition=(resolved.name if resolved else None),
        definition_fingerprint=(
            native_host.compute_definition_fingerprint(resolved) if resolved else None
        ),
        continuation_durability=continuation_durability,
        chain_id=f"{chain_id}:child-{child_sequence}",
        # PR3a-1 Task 3: THIS run tree's writer, captured here on
        # the PARENT's thread rather than looked up later from the
        # child's. A child that outlives the turn (Task 2) may not
        # even reach `_run_one` until `run_turn` has replaced
        # `self.run_log_writer` for the next turn; resolving it at
        # spawn removes that race entirely, and the child then
        # records into its own tree for its whole life. Sharing the
        # parent's writer is not "reusing a writer across trees"
        # (which `bind()`'s permanent latch forbids) -- parent and
        # child ARE one tree, and one writer per tree is what keeps
        # record numbers unique across it.
        run_log_writer=writer,
        startup_instruction_snapshot=startup_instruction_snapshot,
    )
    return child_kwargs, child_budget


def _launch_prepared_child(
    self,
    spawn_task,
    agent_name,
    child_kwargs,
    isolation=None,
    *,
    fleet,
    run_id,
    message_chain_id,
    hook_run,
    should_cancel,
    my_handle_ids,
    durable_handle_ids,
    fleet_cancels,
    fleet_threads,
    child_owner,
    child_reservation,
    definition_wall_seconds=None,
    resumed_managed=None,
    managed_resume_ceiling=None,
    child_outlives_turn=None,
    _workflow_entry=None,
):
    """Preserve the ordinary/no-bridge native tuple and raising contract."""
    result = self._launch_prepared_child_result(
        spawn_task,
        agent_name,
        child_kwargs,
        isolation,
        fleet=fleet,
        run_id=run_id,
        message_chain_id=message_chain_id,
        hook_run=hook_run,
        should_cancel=should_cancel,
        my_handle_ids=my_handle_ids,
        durable_handle_ids=durable_handle_ids,
        fleet_cancels=fleet_cancels,
        fleet_threads=fleet_threads,
        child_owner=child_owner,
        child_reservation=child_reservation,
        definition_wall_seconds=definition_wall_seconds,
        resumed_managed=resumed_managed,
        managed_resume_ceiling=managed_resume_ceiling,
        child_outlives_turn=child_outlives_turn,
        _workflow_entry=_workflow_entry,
    )
    if type(result) is native_host._NativeWorkflowReturn:
        raise result.error
    return result


def _launch_prepared_child_result(
    self,
    spawn_task,
    agent_name,
    child_kwargs,
    isolation=None,
    *,
    fleet,
    run_id,
    message_chain_id,
    hook_run,
    should_cancel,
    my_handle_ids,
    durable_handle_ids,
    fleet_cancels,
    fleet_threads,
    child_owner,
    child_reservation,
    definition_wall_seconds=None,
    resumed_managed=None,
    managed_resume_ceiling=None,
    child_outlives_turn=None,
    _workflow_entry=None,
):
    """The native fleet launch/settlement tail, with captured parent inputs."""
    authorization = child_kwargs.get("workflow_task_authorization")
    entry = _workflow_entry
    if authorization is not None:
        if (
            type(entry) is not native_host._WorkflowLaunchEntry
            or entry.service is not self
            or entry.authorization is not authorization
            or len(entry.native.launch_entry) != 1
            or entry.native.launch_entry[0] is not entry
        ):
            raise PermissionError("registered native launch required")
        native_host._workflow_launch_current(entry)
    launched = False
    native_thread_started = False
    thread = None
    failure_started = False
    handle = None
    child_run_id = None
    preserve_workflow_reservation = False

    def failed_workflow_launch(handle, child_run_id, reason, message):
        nonlocal preserve_workflow_reservation, failure_started
        failure_started = True
        authorization = child_kwargs.get("workflow_task_authorization")
        if authorization is None:
            return
        claim = child_kwargs["workflow_activity_claim"]
        if reason == "unknown":
            child_owner.mark_cleanup_unproven()
            preserve_workflow_reservation = True
        try:
            row = self.db.get_run(child_run_id)
            if row is None:
                if reason == "unknown":
                    raise RuntimeError("native_creation_unproven")
                reason = "no_run"
                with self.db.automatic_work.transaction() as conn:
                    conn.execute(
                        "UPDATE workflow_task_attempts SET status='closed',retirement='drained' WHERE id=? AND run_id IS NULL",
                        (authorization.claim.id,),
                    )
                    self.db.workflow_runs._release_unused_result(
                        conn, authorization.claim.workflow_id, authorization.claim.id
                    )
            else:
                self.db.automatic_work.bind_workflow_task_run(
                    authorization.claim.phase_attempt_id,
                    task_id=authorization.claim.task_id,
                    run_id=child_run_id,
                    owner_id=authorization.claim.owner_id,
                )
                child_owner.bind_run(child_run_id)
                if reason != "unknown":
                    self._set_terminal_status(child_run_id, native_host.RUN_ERROR)
        except Exception:  # noqa: BLE001 - unknown creation cannot release custody
            reason = "unknown"
            child_owner.mark_cleanup_unproven()
            preserve_workflow_reservation = True
        receipt = native_host.WorkflowChildReceipt(
            authorization.claim.id,
            child_run_id,
            handle.handle_id,
            child_owner,
            claim,
            reason,
        )
        with authorization._lock:
            authorization._native_launch[0] = receipt
        if reason == "no_run":
            claim.settle(durable=True)
        error = native_host._WorkflowChildLaunchFailed(receipt, message)
        if entry is not None:
            return native_host._NativeWorkflowReturn(
                receipt, error, reason != "unknown"
            )
        raise error

    try:
        # -- FLEET path: register, launch, return a handle.
        handle = fleet.reserve(
            task=spawn_task,
            agent=agent_name,
            isolation=isolation,
            definition_wall_seconds=definition_wall_seconds,
            # Live metadata follows the active frozen config. The
            # resolved_* kwargs remain the original persisted audit.
            resolved_provider=child_kwargs["config"].provider,
            resolved_model=child_kwargs["config"].model,
        )
        if handle is None:
            # At the live cap. Unlike a budget refusal this is
            # RETRYABLE -- collecting a finished child frees a slot --
            # so it must not consume a spawn from the per-turn
            # ceiling, exactly like the unknown-agent refusal above
            # ("a typo costs no sub-agent slot"). The check/increment
            # itself stays where it has always been; only this one
            # no-child-was-created path unwinds it.
            preserve_workflow_reservation = (
                child_kwargs.get("workflow_task_authorization") is not None
            )
            return None, native_host._FleetAdmissionWait(
                ok=False,
                error=(
                    f"live sub-agent limit reached ({fleet.live_count()} "
                    "already running); call wait_agents to collect a "
                    "finished sub-agent before starting another"
                ),
            )
        fleet.set_managed_custody(
            handle.handle_id,
            required=self.managed_child_required
            or bool(resumed_managed and resumed_managed.managed_required),
            resume_ceiling=managed_resume_ceiling,
            resume_pin=resumed_managed.managed_pin if resumed_managed else None,
            resume_run_id=resumed_managed.run_id if resumed_managed else None,
            resume_handle_id=(resumed_managed.handle_id if resumed_managed else None),
        )
        child_run_id = (
            child_kwargs.get("requested_run_id") or native_host.uuid.uuid4().hex
        )
        tree = getattr(self, "_ordinary_tree", None)
        if tree is not None:
            tree.row_started(child_owner, child_run_id)
        try:
            self.db.create_run(
                run_id=child_run_id,
                conversation_id=child_kwargs["conversation_id"],
                agent_kind=child_kwargs["agent_kind"],
                task=native_host._safe_agent_task_summary(child_kwargs.get("task")),
                parent_run_id=child_kwargs.get("parent_run_id"),
                budget=native_host.dataclasses.asdict(child_kwargs["config"].budget),
                agent_definition=child_kwargs.get("agent_definition"),
                definition_fingerprint=child_kwargs.get("definition_fingerprint"),
                resumed_from_run_id=child_kwargs.get("resumed_from_run_id"),
                spawn_event_id=child_kwargs.get("spawn_event_id"),
                work_chain_id=self._work_chain_id,
                resolved_provider=child_kwargs.get("resolved_provider"),
                resolved_model=child_kwargs.get("resolved_model"),
                resolved_base_url=child_kwargs.get("resolved_base_url"),
                resolved_params_json=child_kwargs.get("resolved_params_json"),
                fallback_targets_json=child_kwargs.get("fallback_targets_json"),
                active_fallback_index=child_kwargs.get("active_fallback_index", 0),
            )
        except Exception as exc:  # noqa: BLE001 — spawn refusal, never parent abort
            fleet.finish(
                handle.handle_id,
                native_host.RUN_ERROR,
                error="could not create durable sub-agent run",
            )
            native_host.logger.warning(
                "could not precreate sub-agent run error_type={}",
                native_host._safe_exception_type(exc),
            )
            terminal = failed_workflow_launch(
                handle,
                child_run_id,
                "setup",
                "could not create durable sub-agent run",
            )
            if terminal is not None:
                return terminal
            return None, native_host.SpawnAdmissionRefusal(
                ok=False, error="could not create durable sub-agent run"
            )
        if child_kwargs.get("review_launch") is not None:
            from .agent_request_execution import attach_review_launch
            from .code_review_models import ReviewRefused

            try:
                attach_review_launch(
                    child_kwargs["review_service"],
                    child_kwargs["review_launch"],
                    child_run_id,
                )
            except ReviewRefused as exc:
                # No worker was started. Settle the precreated native
                # state before the existing admission-refusal unwind.
                fleet.finish(handle.handle_id, native_host.RUN_ERROR, error=exc.reason)
                self._set_terminal_status(
                    child_run_id, native_host.RUN_ERROR, result=exc.reason
                )
                return None, native_host.SpawnAdmissionRefusal(
                    ok=False, error=exc.reason
                )
        workflow = child_kwargs.get("workflow_task_authorization")
        if workflow is not None:
            self.db.automatic_work.bind_workflow_task_run(
                workflow.claim.phase_attempt_id,
                task_id=workflow.claim.task_id,
                run_id=child_run_id,
                owner_id=workflow.claim.owner_id,
            )
            child_owner.bind_run(child_run_id)
        owner_seq, lifecycle_event_id = self._append_run_lifecycle(
            child_run_id,
            native_host.STEP_AGENT_RUN_RESERVED,
            "reserved",
            owner_seq=0,
            parent_event_id=child_kwargs.get("spawn_parent_event_id"),
        )
        owner_seq, lifecycle_event_id = self._append_run_lifecycle(
            child_run_id,
            native_host.STEP_AGENT_RUN_CREATED,
            "created",
            owner_seq=owner_seq,
            parent_event_id=lifecycle_event_id,
        )
        resumed_from = child_kwargs.get("resumed_from_run_id")
        if resumed_from:
            owner_seq, lifecycle_event_id = self._append_run_lifecycle(
                child_run_id,
                native_host.STEP_AGENT_RUN_RESUMED,
                "resumed",
                owner_seq=owner_seq,
                parent_event_id=lifecycle_event_id,
                source_event_id=f"agent-run:{resumed_from}",
            )
        child_kwargs["precreated_run_id"] = child_run_id
        if isolation == "worktree":
            refusal = self._admit_agent_worktree(handle, child_run_id, child_owner)
            if refusal is not None:
                # I1 (TASK-28238 P2 T7 final fix wave): `db.create_run`
                # above already wrote this row as "running" --
                # attach_run (below) never happens on this path, so
                # nothing else ever marks it terminal. Matches the
                # sibling thread-start-failure path's try/except shape
                # exactly (round 2, item 4).
                try:
                    fleet.finish(handle.handle_id, native_host.RUN_ERROR, error=refusal)
                    self._set_terminal_status(
                        child_run_id, native_host.RUN_ERROR, result=refusal
                    )
                except Exception:  # noqa: BLE001 — refusal must reach parent
                    native_host.logger.warning(
                        "could not persist failed sub-agent launch"
                    )
                return None, native_host.SpawnAdmissionRefusal(ok=False, error=refusal)
        child_kwargs["lifecycle_owner_seq_start"] = owner_seq
        fleet.attach_run(handle.handle_id, child_run_id)
        durable_handle_ids[handle.handle_id] = child_run_id
        # Cooperative cancellation for THIS child specifically, on top
        # of the run-wide `should_cancel` it also honours. wait_agents
        # and the end-of-turn settle both set it to unwind stragglers
        # without cancelling the parent.
        child_cancel = native_host.threading.Event()
        child_kwargs["continuation_agent_kind"] = "fleet"
        child_kwargs["progress_sender_factory"] = lambda: (
            fleet.bind_progress_sender(
                handle.handle_id,
                parent_run_id=run_id,
                chain_id=message_chain_id,
            )
            if run_id is not None
            else None
        )
        child_kwargs["peer_messenger_factory"] = lambda: (
            fleet.bind_peer_messenger(
                handle.handle_id,
                parent_run_id=run_id,
                chain_id=message_chain_id,
            )
            if run_id is not None
            else None
        )
        # PR3b Task 1: THIS child's steering drain -- a closure over
        # its own mailbox on the conversation-lifetime coordinator,
        # reachable from the UI thread and any turn's supervisor while
        # the child runs on its own thread. handle_id is default-bound
        # (the run_child style) so the closure can never pick up a
        # later spawn's handle.
        child_kwargs["drain_mailbox_with_causes"] = lambda handle_id=handle.handle_id: (
            fleet.drain_steering_with_causes(handle_id)
        )
        child_kwargs["drain_mailbox"] = lambda handle_id=handle.handle_id: (
            fleet.drain_steering(handle_id)
        )
        fleet_cancels[handle.handle_id] = child_cancel
        my_handle_ids.append(handle.handle_id)

        # PR3b Task 5 (spec Sec 8, Stop-semantics decoupling): with
        # `subagents_outlive_turn` ON, a child's cancellation is ITS
        # OWN Event only -- the parent's run-wide probe stays out of
        # its poll, so a user Stop (which flips that probe and leaves
        # it flipped forever) no longer kills background work at the
        # child's next loop boundary. Every path that must still stop
        # a child sets the Event (`_cancel_fleet_handles`: the
        # end-of-turn settle for non-survivors, `wait_agents`' budget
        # branch, per-row Cancel, and "Cancel all agents"). Read ONCE,
        # at spawn: a child's Stop-coupling contract is fixed when it
        # launches, not flappable mid-run by a config write --
        # `_surviving_handles` still reads the key at settle time, so
        # flipping the switch OFF mid-run still lets the very next
        # settle cancel the child through its Event. With the key OFF
        # the closure is the pre-Task-5 line, byte-identical.
        if child_outlives_turn is None:
            child_outlives_turn = native_host._coerce_subagents_outlive_turn(
                native_host._setting(
                    native_host.SUBAGENTS_OUTLIVE_TURN_KEY,
                    native_host.DEFAULT_SUBAGENTS_OUTLIVE_TURN,
                )
            )
        if child_outlives_turn:

            def child_should_cancel() -> bool:
                return child_cancel.is_set()

        else:

            def child_should_cancel() -> bool:
                return should_cancel() or child_cancel.is_set()

        admission: native_host.Future[None] = native_host.Future()
        child_body_entered = False

        def run_child(
            handle: native_host.FleetHandle = handle,
            child_kwargs: dict = child_kwargs,
            child_should_cancel=child_should_cancel,
            *,
            admission_error: BaseException | None = None,
        ) -> None:
            """Run one child to completion, then release its handle.

            Deliberately NOT wrapped in the parent's
            `review_state_scope`, which the inline path above still
            takes. That scope snapshots and RESTORES the parent's
            verdict slice, which is sound only for a strictly nested
            (LIFO) inline child: with siblings running concurrently,
            one child's exit would roll the parent's slice back to a
            snapshot taken before another child -- or before the
            parent's own later turn -- had stamped anything, wiping
            live verdicts. PR2a Task 5 made per-run keying the
            load-bearing protection precisely so this scope is not
            needed here: the child stamps `(child_run_id, tool)` and
            cannot reach the parent's keys at all.
            """
            nonlocal child_body_entered
            status = native_host.RUN_ERROR
            result_text = ""
            error_text = ""
            # PR2b Task 5 (cost rollup): only ever set from a
            # SUCCESSFULLY-returned `child_outcome` below -- a child
            # that raised before `_run_one` returned has no measured
            # spend to report, so this stays 0 (never a fabricated or
            # partial figure) exactly like `result_text`/`error_text`
            # staying at their own "nothing to report" defaults on that
            # path.
            total_tokens_spent = 0
            # PR3b Task 4: the coherent transcript, retained by
            # `fleet.finish` below ATOMICALLY with the terminal
            # transition. Stays None on the raise path -- a child that
            # died before the loop returned has no coherent transcript
            # to retain, and retention honestly refuses None.
            final_messages = None
            try:
                if admission_error is not None:
                    # The worker guard refused before _run_owned could
                    # own/refund this reservation. Run only the shared
                    # error teardown below; no child execution starts.
                    if child_reservation is not None:
                        self._automatic_work.ledger.release(
                            child_reservation,
                            owner_id=self._automatic_work.owner_id,
                        )
                    raise admission_error
                child_body_entered = True
                if not admission.set_running_or_notify_cancel():
                    if child_reservation is not None:
                        self._automatic_work.ledger.release(
                            child_reservation,
                            owner_id=self._automatic_work.owner_id,
                        )
                    raise native_host.AgentActivationRequired()
                admission.set_result(None)
                # PR3a-1 Task 1: this child's own model-call lifeline,
                # entered HERE -- on the child's thread, before its run
                # starts -- and exited when the run ends, so it lives
                # exactly as long as the child does rather than as long
                # as the turn that spawned it. See
                # `self._child_model_scope`.
                _child_id, child_outcome = self._run_owned(
                    fleet_child=True,
                    execution_owner=child_owner,
                    automatic_child_reservation=child_reservation,
                    finish_root=False,
                    should_cancel=child_should_cancel,
                    on_run_id=(lambda rid: fleet.attach_run(handle.handle_id, rid)),
                    **child_kwargs,
                )
                status = child_outcome.status
                result_text = child_outcome.final_text
                if child_kwargs.get("review_launch") is not None:
                    claim = child_kwargs["review_launch"]
                    result_text = f"review {claim.review_id}; slot {claim.slot.slot_id}; read originals with code_review_read"
                total_tokens_spent = child_outcome.total_tokens
                final_messages = child_outcome.final_messages
                if status != native_host.RUN_DONE:
                    error_text = f"sub-agent {status}"
            except BaseException as exc:  # noqa: BLE001 — see below
                # EVERY exception, including BaseException: this runs
                # on a daemon thread whose exception would otherwise
                # go to the default excepthook and leave the handle
                # live forever -- stranding the parent's end-of-turn
                # join until its own timeout, every time, for what may
                # be a trivial bug. Same containment rule as
                # `_call_with_timeout._runner`.
                error_text = f"sub-agent failed: {exc}"
                native_host.logger.warning("sub-agent thread raised")
            finally:
                # PR3b Task 4: `transcript=` makes retention atomic
                # with the terminal transition -- ONE coordinator
                # critical section, so a send_to_agent continuation
                # racing this teardown can never observe a retainable
                # child terminal-but-unretained (the Qodo race finding
                # on plan PR #1773; the plan's original
                # retain-after-finish two-step had that window).
                # First-writer-wins covers retention too: if a
                # settle-cancel already finished this handle, this
                # whole call -- transcript included -- is ignored, so
                # a user-cancelled child is never retained.
                settled_id = child_owner.run_id
                if hook_run is not None and settled_id:
                    self._set_terminal_status(settled_id, status)
                    self._install_child_stop(settled_id, status, run_id, should_cancel)
                fleet.finish(
                    handle.handle_id,
                    status,
                    result=result_text,
                    error=error_text,
                    total_tokens=total_tokens_spent,
                    transcript=final_messages,
                )
                # Review fix (PR2a final review): `_persist` -- called
                # from INSIDE `_run_one`'s own try/except -- is
                # normally the only thing that writes this child's
                # terminal DB status. But `_run_one`'s try/except
                # wraps ONLY the `run_agent_loop(...)` call; an
                # exception raised between `create_run()` and that
                # try block (e.g. `initial_disclosure` recursing into
                # the tool catalog's RLock and raising RecursionError)
                # unwinds `_run_one` entirely, past `_persist`, and
                # lands in the `except BaseException` above instead --
                # leaving the DB row `running` for the life of the
                # process, violating "DB is truth" (spec Sec 3
                # invariant 3). `attach_run` has already fired by the
                # time any post-`create_run` exception can, so
                # `fleet.get()` (re-fetched, not the possibly-stale
                # `handle` closed over above) reliably has the run id.
                # `set_status` is first-writer-wins (AgentRunsDB), so
                # this call is a safe no-op on the normal path where
                # `_persist` already wrote a terminal status -- it
                # only matters on the setup-phase-exception path,
                # where it is the only writer. Same defensive shape as
                # `_settle_fleet`'s abandonment path: a DB failure
                # here must not take down a turn that has already
                # produced its answer.
                try:
                    self._retire_agent_worktree(
                        child_kwargs.get("precreated_run_id"),
                        handle.handle_id,
                        discard=admission_error is not None,
                    )
                except Exception:
                    native_host.logger.opt(exception=True).warning(
                        "could not retire agent worktree routing for handle {}",
                        handle.handle_id,
                    )
                current = fleet.get(handle.handle_id)
                child_run_id = current.run_id if current is not None else None
                if child_run_id:
                    try:
                        self._set_terminal_status(child_run_id, status)
                    except Exception:  # noqa: BLE001
                        native_host.logger.warning(
                            "could not persist terminal status for sub-agent run"
                        )
                authorization = child_kwargs.get("workflow_task_authorization")
                if authorization is not None:
                    self._finalize_workflow_child_log(
                        authorization,
                        status=status,
                        total_tokens=total_tokens_spent,
                    )
                if child_run_id and child_kwargs.get("review_service") is not None:
                    self._settle_review_capture(
                        child_kwargs["review_service"], child_run_id
                    )
                # PR3a-2 Task 2: the settle signal, LAST -- after
                # `fleet.finish` and after the terminal-status
                # fallback, so at fire time the row is terminal on
                # the happy path (`_persist` wrote it) AND on the
                # setup-exception path (`set_status` just did). See
                # `on_child_settled`'s __init__ comment for why no
                # earlier point can offer that. Wrapped never-raise:
                # this is a daemon thread's teardown, and a notifier
                # bug must not kill it (same containment rule as the
                # `except BaseException` above).
                if self._on_child_settled is not None:
                    try:
                        self._on_child_settled(child_run_id, status)
                    except Exception as exc:  # noqa: BLE001
                        native_host.logger.warning(
                            "on_child_settled consumer raised (exception_type={})",
                            type(exc).__name__,
                        )

        def run_child_scoped() -> None:
            def run_original_child():
                try:
                    admitted_child()
                finally:
                    if (
                        workflow_task_authorization := child_kwargs.get(
                            "workflow_task_authorization"
                        )
                    ) and workflow_task_authorization._bridge_preparation:
                        bridge_preparation = (
                            workflow_task_authorization._bridge_preparation[0]
                        )
                        bridge_preparation._close_on_native_owner(child_owner)

            tree = getattr(self, "_ordinary_tree", None)
            if tree is None:
                authorization = child_kwargs.get("workflow_task_authorization")
                if authorization is None or not authorization._bridge_preparation:
                    run_original_child()
                else:
                    bridge_preparation = authorization._bridge_preparation[0]
                    with bridge_preparation.activate_source():
                        run_original_child()
                return
            from .native_source_storage import (
                original_retained_source_stack,
            )
            from tldw_chatbook.DB.base_db import operation_owned_connection

            with original_retained_source_stack(tree, tree.source.databases) as stack:
                for db in dict.fromkeys(
                    db for db in tree.source.databases if db is not None
                ):
                    stack.enter_context(operation_owned_connection(db))
                run_original_child()

        def run_child_owned() -> None:
            error = None
            try:
                run_child_scoped()
            except BaseException as exc:  # noqa: BLE001 - native worker must publish admission
                error = exc
                if child_body_entered:
                    child_owner.mark_cleanup_unproven()
                    fleet.finish(
                        handle.handle_id,
                        native_host.RUN_ERROR,
                        error="native child cleanup failed",
                    )
                    try:
                        self._set_terminal_status(child_run_id, native_host.RUN_ERROR)
                    except Exception:  # noqa: BLE001 - admission still must return
                        native_host.logger.warning(
                            "could not persist native child failure"
                        )
            finally:
                try:
                    # Release the root only after all native source scopes exit.
                    if child_body_entered or admission.cancelled():
                        self._finish_execution(child_owner)
                except BaseException as exc:  # noqa: BLE001 - cleanup uncertainty retains protection
                    child_owner.mark_cleanup_unproven()
                    error = error or exc
                finally:
                    if not admission.done():
                        admission.set_exception(
                            error
                            or RuntimeError("native child returned before admission")
                        )

        admission_stage = "setup"
        try:
            admitted_child = native_host.worker_guard(self)(run_child)
            thread = native_host.threading.Thread(
                target=run_child_owned,
                name=f"fleet-{handle.handle_id[:8]}",
                daemon=True,
            )
            admission_stage = "thread_start"
            try:
                thread.start()
            except BaseException as start_error:
                if thread.ident is not None:
                    native_thread_started = launched = True
                    fleet_threads[handle.handle_id] = thread
                    if child_kwargs.get("review_launch") is not None and entry is None:
                        # The worker exists; its admission and return remain authoritative.
                        native_host.logger.warning(
                            "review child start raised after worker admission"
                        )
                    else:
                        child_cancel.set()
                        admission.cancel()
                        preserve_workflow_reservation = True
                        child_owner.mark_cleanup_unproven()
                        if entry is not None:
                            with authorization._lock:
                                native_host._workflow_launch_locked(entry)
                                entry.native.uncertain_start[:] = [
                                    (thread, run_child_owned, child_kwargs, start_error)
                                ]
                        raise
                else:
                    raise
            native_thread_started = True
        except Exception as exc:  # noqa: BLE001 — setup or thread exhaustion
            if native_thread_started:
                terminal = failed_workflow_launch(
                    handle,
                    child_run_id,
                    "unknown",
                    "native child start return unproven",
                )
                if terminal is not None:
                    return terminal
                return handle, native_host.SpawnAdmissionRefusal(
                    ok=False, error="native child start cleanup pending"
                )
            # `Thread.start()` raises RuntimeError ("can't start new
            # thread") when the process is out of thread slots. Every
            # piece of state this spawn reserved has to be unwound
            # here, because NOTHING else will: `run_child` never runs,
            # so the handle would stay live forever -- making the
            # end-of-turn settle burn the ENTIRE remaining wall-clock
            # waiting for a child that does not exist. Registering the
            # thread only AFTER a successful start is the other half:
            # `_settle_fleet` joins every registered thread, and
            # joining an unstarted one raises RuntimeError out of
            # `run_turn`, skipping `write_manifest()` and
            # `run_log_writer.close()` (leaking a file descriptor).
            fleet.finish(
                handle.handle_id,
                native_host.RUN_ERROR,
                error=f"could not start sub-agent thread: {exc}",
            )
            try:
                self._set_terminal_status(child_run_id, native_host.RUN_ERROR)
            finally:
                try:
                    self._retire_agent_worktree(
                        child_run_id, handle.handle_id, discard=True
                    )
                except Exception:
                    native_host.logger.opt(exception=True).warning(
                        "could not retire agent worktree routing for handle {}",
                        handle.handle_id,
                    )
            fleet_cancels.pop(handle.handle_id, None)
            if my_handle_ids and my_handle_ids[-1] == handle.handle_id:
                my_handle_ids.pop()
            # No child was created, so this spawn costs no slot --
            # same rule as the cap refusal above.
            if child_kwargs.get("review_launch") is not None:
                with native_host.contextlib.suppress(ReviewRefused):
                    child_kwargs["review_service"].retire_unstarted_launch(
                        child_kwargs["review_launch"], child_run_id
                    )
            native_host.logger.warning("could not start sub-agent thread")
            terminal = failed_workflow_launch(
                handle,
                child_run_id,
                admission_stage,
                f"could not start sub-agent: {exc}",
            )
            if terminal is not None:
                return terminal
            return None, native_host.SpawnAdmissionRefusal(
                ok=False,
                error=f"could not start sub-agent: {exc}",
            )
        fleet_threads[handle.handle_id] = thread
        try:
            admission.result(timeout=native_host.FLEET_JOIN_TIMEOUT_SECONDS)
        except BaseException as admission_error:
            caller_error = not isinstance(
                admission_error, native_host.FutureTimeoutError
            ) and (
                not admission.done()
                or admission.cancelled()
                or admission.exception() is not admission_error
            )
            if (
                isinstance(admission_error, native_host.FutureTimeoutError)
                and not admission.cancel()
                and (
                    admission.running()
                    or (not admission.cancelled() and admission.exception() is None)
                )
            ):
                launched = True
                return handle, None
            child_cancel.set()
            admission.cancel()
            thread.join(native_host.FLEET_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                launched = True
                preserve_workflow_reservation = True
                child_owner.mark_cleanup_unproven()
                if caller_error:
                    raise
                terminal = failed_workflow_launch(
                    handle,
                    child_run_id,
                    "unknown",
                    "native child admission cleanup pending",
                )
                if terminal is not None:
                    return terminal
                return None, native_host.SpawnAdmissionRefusal(
                    ok=False, error="native child admission cleanup pending"
                )
            try:
                if (
                    fleet.get(handle.handle_id).status
                    not in native_host.TERMINAL_RUN_STATUSES
                ):
                    run_child(admission_error=admission_error)
            finally:
                self._finish_execution(child_owner)
            launched = True
            if caller_error:
                raise
            terminal = failed_workflow_launch(
                handle, child_run_id, "setup", "native child admission refused"
            )
            if terminal is not None:
                return terminal
            return None, native_host.SpawnAdmissionRefusal(
                ok=False, error="native child admission refused"
            )
        launched = True
        return handle, None

    except Exception as exc:
        authorization = child_kwargs.get("workflow_task_authorization")
        if (
            authorization is not None
            and not isinstance(exc, native_host._WorkflowChildLaunchFailed)
            and (entry is None or not failure_started)
            and handle is not None
            and child_run_id is not None
        ):
            if not native_thread_started:
                fleet.finish(
                    handle.handle_id,
                    native_host.RUN_ERROR,
                    error="workflow native setup failed",
                )
            terminal = failed_workflow_launch(
                handle,
                child_run_id,
                "unknown" if native_thread_started else "setup",
                "workflow native setup failed",
            )
            if terminal is not None:
                return terminal
        raise
    finally:
        if not launched:
            try:
                authorization = child_kwargs.get("workflow_task_authorization")
                if (
                    authorization is not None
                    and handle is not None
                    and not native_thread_started
                    and not preserve_workflow_reservation
                ):
                    with child_owner.activate():
                        self._finalize_workflow_child_log(
                            authorization, status=native_host.RUN_ERROR
                        )
                    if authorization._bridge_preparation:
                        preparation = authorization._bridge_preparation[0]
                        preparation._close_on_native_owner(child_owner)
                if child_reservation is not None and not preserve_workflow_reservation:
                    self._automatic_work.ledger.release(
                        child_reservation,
                        owner_id=self._automatic_work.owner_id,
                    )
            finally:
                self._finish_execution(child_owner)


def _install_child_stop(self, child_id, status, parent_id, should_cancel):
    held = self.__dict__.get("_ordinary_child_stops", {}).pop(child_id, None)
    if held is None:
        if getattr(self, "_ordinary_tree", None) is not None:
            raise RuntimeError("original child completion event unavailable")
        event = self._hooks_v2_lifecycle.event(
            "SubagentStop",
            run_id=parent_id,
            data={"child_run_id": child_id, "status": status},
            initiator="child",
        )
        scope = None
    else:
        original, scope = held
        event = original.model_copy(
            update={"data": {"child_run_id": child_id, "status": status}}
        )
    self._hooks_v2_lifecycle.install(
        event,
        parent_id,
        current=lambda: not should_cancel(),
        _execution=scope,
    )


def _finish_execution(self, owner):
    prepared = getattr(self, "_workflow_planning_preparation", None)
    if prepared is not None:
        if (
            prepared.planning_service is not self
            or prepared.native.service is not self
            or prepared.primary_owner is not owner
        ):
            raise PermissionError("original planning primary required")
        return  # The original root owns native finalization and required storage.
    tree = getattr(self, "_ordinary_tree", None)
    if tree is not None:
        if owner is tree.owner:
            return  # Original submit owns primary cleanup and late producers.
        with tree.source.activate():
            child_id = self.__dict__.get("_ordinary_child_keys", {}).pop(owner, None)
            legacy = self.__dict__.get("_ordinary_child_notifications", {}).pop(
                child_id, None
            )
            if (
                legacy is not None
                and legacy[1] is not None
                and not legacy[1].notification_claimed
            ):
                legacy[1].abandon_unstarted()
            tree.close_unused_events(owner)
            tree.host_returned(owner)
            owner.finish_root()
        return
    owner.finish_root()


def _run_owned(
    self,
    *,
    execution_owner: native_host.ExecutionOwner | None = None,
    fleet_child: bool = False,
    finish_root: bool = True,
    automatic_child_reservation: str | None = None,
    workflow_task_authorization: native_host.Any = None,
    workflow_activity_claim: native_host.Any = None,
    **kwargs: native_host.Any,
) -> tuple[str, native_host.RunOutcome]:
    """Keep the execution owned through its run and child-driver cleanup."""
    from .automatic_work_budget import AutomaticWorkLimits, AutomaticWorkRefused
    from .workflow_task_scope import (
        WorkflowTaskAuthorization,
        require_workflow_hooks_idle,
        require_workflow_runtime,
    )

    if (
        workflow_task_authorization is not None
        and type(workflow_task_authorization) is not WorkflowTaskAuthorization
    ):
        raise AutomaticWorkRefused("workflow_task_private_custody_required")

    if workflow_task_authorization is not None:
        self._check_workflow_launch_authorization(workflow_task_authorization)
        authorization = workflow_task_authorization
        with self._workflow_preparation_lock, authorization._lock:
            if (
                self._workflow_authorization is not None
                and self._workflow_authorization is not authorization
            ) or authorization._native_dispatch:
                raise PermissionError("workflow native dispatch already claimed")
            self._workflow_authorization = authorization
            authorization._native_dispatch.append(self)
            kwargs.setdefault(
                "startup_instruction_snapshot", self._startup_instruction_snapshot
            )

    owner = execution_owner or self.runtime_capacity.begin_execution(
        origin=self.work_origin,
        conversation_id=kwargs["conversation_id"],
        child=kwargs["agent_kind"] != native_host.AGENT_KIND_PRIMARY,
    )
    automatic_work = self._automatic_work
    try:
        if (
            self.work_origin is native_host.WorkOrigin.AUTOMATIC
            and automatic_work is None
        ):
            raise AutomaticWorkRefused("acceptance_required")
        with (
            owner.activate(),
            automatic_work.scope()
            if automatic_work
            else native_host.contextlib.nullcontext(),
            workflow_task_authorization.scope()
            if workflow_task_authorization
            else native_host.contextlib.nullcontext(),
        ):
            if automatic_work is not None and automatic_work.workflow is not None:
                phase = automatic_work.workflow
                require_workflow_hooks_idle(
                    require_workflow_runtime(phase), phase.attempt.session_id
                )
                if (
                    phase.attempt.purpose == "execution"
                    and workflow_task_authorization is None
                ):
                    raise AutomaticWorkRefused("workflow_task_private_custody_required")
            if (
                automatic_work is not None
                and automatic_work.workflow is not None
                and automatic_work.workflow.attempt.purpose == "planning"
            ):
                from .run_context import (
                    current_planning_preparation,
                )

                prepared = current_planning_preparation()
                if (
                    prepared is None
                    or prepared.phase.context is not automatic_work
                    or prepared.native.service is not self
                    or prepared.primary_owner is not owner
                ):
                    raise AutomaticWorkRefused(
                        "workflow_planning_private_custody_required"
                    )
                prepared._admit_planning_primary(self)
            if workflow_task_authorization is not None:
                authorization = workflow_task_authorization
                authorization.check_preparation()
                if (
                    authorization.phase.context is not automatic_work
                    or authorization.phase._ledger._db is not self.db
                    or authorization.registry is not self.registry
                    or automatic_child_reservation != authorization.claim.reservation_id
                    or (
                        kwargs.get("precreated_run_id")
                        or kwargs.get("requested_run_id")
                    )
                    != authorization.claim.launch_run_id
                    or self.guard_tool_calls is not None
                    or self._hooks_v2_engine is not None
                    or self._post_tool_call is not None
                ):
                    raise AutomaticWorkRefused(
                        "workflow_task_native_preparation_refused"
                    )
                kwargs["workflow_activity_claim"] = workflow_activity_claim
            if automatic_work is not None:
                automatic_work.check()
                if automatic_child_reservation is not None and not (
                    workflow_task_authorization.phase._commit_task_launch(
                        workflow_task_authorization,
                        owner,
                        workflow_activity_claim,
                        self,
                    )
                    if workflow_task_authorization is not None
                    else automatic_work.ledger.commit(
                        automatic_child_reservation,
                        owner_id=automatic_work.owner_id,
                        limits=AutomaticWorkLimits.from_settings(
                            automatic_work.policy_kind()
                        ),
                    )
                ):
                    raise AutomaticWorkRefused("child_already_launched")
                # A survivor drops its parent's Stop probe, but keeps this
                # chain deadline and kill switch for its entire lifetime.
                run_cancel = kwargs["should_cancel"]
                kwargs["should_cancel"] = lambda: (
                    run_cancel() or automatic_work.should_cancel()
                )
            if fleet_child:
                scope = self._child_model_scope()
            elif kwargs["agent_kind"] != native_host.AGENT_KIND_PRIMARY:
                scope = self._inline_child_model_scope()
            else:
                scope = native_host.contextlib.nullcontext()
            with scope:
                return self._run_one(**kwargs)
    finally:
        try:
            if (
                kwargs["agent_kind"] == native_host.AGENT_KIND_PRIMARY
                and self._archive_recovery_provider is not None
            ):
                self._archive_recovery_provider.finish()
        finally:
            try:
                if (
                    automatic_work is not None
                    and automatic_child_reservation is not None
                ):
                    # Only uncommitted work can refund; preserve owned driver cleanup.
                    automatic_work.ledger.release(
                        automatic_child_reservation,
                        owner_id=automatic_work.owner_id,
                    )
            finally:
                if finish_root:
                    self._finish_execution(owner)
