"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from . import console_agent_bridge as native_host


@native_host.contextlib.contextmanager
def _prepare_native_reply(
    self,
    *,
    conversation_id: str,
    session_id: str,
    resolution: native_host.Any,
    assistant_message_id: str,
    model: str,
    session_system_prompt: str,
    agent_messages: list[dict],
    should_cancel: native_host.Callable[[], bool],
    expected_progress_owner_id: str | None = None,
    work_origin: native_host.WorkOrigin = native_host.WorkOrigin.MANUAL,
    work_chain_id: str | None = None,
    provider_stream_signals: native_host.ConsoleProviderStreamSignals | None = None,
    supersede_previous: bool = False,
    mcp_provider: native_host.Any | None = None,
    builtin_gate: native_host.Any | None = None,
    scratch_root: native_host.Path | None = None,
    scratch_lease: native_host.Callable[
        [], native_host.ContextManager[native_host.Path]
    ]
    | None = None,
    # PR2a Task 5: `(calls, run_id)` -- forwarded straight to
    # `AgentService(review_tool_calls=...)`, which binds each run's own
    # id in before handing it to `LoopDeps`.
    review_tool_calls: (
        native_host.Callable[
            [list[native_host.ToolCall], str], dict[str, native_host.ToolReviewValue]
        ]
        | None
    ) = None,
    on_steer_ready: native_host.Callable[
        [native_host.Callable[[str], str | None]], None
    ]
    | None = None,
    # TASK-28227: fired once the run's mailbox registers, with a bound
    # `redirect(text) -> refusal | None` -- the Redirect button's and
    # /redirect's hook, exactly like on_steer_ready is /steer's.
    on_redirect_ready: native_host.Callable[
        [native_host.Callable[[str], str | None]], None
    ]
    | None = None,
    change_roots: native_host.Sequence[native_host.Path] | None = None,
    worktree_repo_authority: native_host.RunAdmittedWorkspaceRoot | None = None,
    change_root_aliases: native_host.Sequence[str] = (),
    change_review_skipped_roots: native_host.Sequence[
        native_host.SkippedReviewRoot
    ] = (),
    turn_skill_bindings: tuple[str, ...] = (),
    turn_bundle_block: str = "",
    request_skill_install_confirm: native_host.Callable[[str], bool] | None = None,
    request_skill_script_confirm: native_host.Callable[[dict], dict] | None = None,
    request_chat_create_confirm: native_host.Callable[[dict], dict] | None = None,
    execute_agent_chat_create: native_host.Callable[[dict], dict] | None = None,
    prepare_agent_chat_create: native_host.Callable[[dict], dict] | None = None,
    # TASK-28238 phase 2 Task 6: forwarded straight to
    # `AgentService.run_turn(request_worktree_merge_confirm=...)` --
    # unlike `request_skill_script_confirm` above, this is never
    # consumed inside a bridge-built closure; `merge_agent_worktree`/
    # `discard_agent_worktree` are the service's OWN closures (Task 5),
    # so there is nothing for `run_reply` to do with it but pass it on.
    request_worktree_merge_confirm: native_host.Callable[[dict], dict] | None = None,
    local_provider: native_host.Any | None = None,
    virtual_cli_provider: native_host.Any | None = None,
    raw_shell_provider: native_host.Any | None = None,
    library_provider: native_host.Any | None = None,
    library_authority: native_host.Any | None = None,
    managed_skill_promotion_gate: native_host.Any | None = None,
    profile_provider: native_host.Any | None = None,
    canvas_provider: native_host.Any | None = None,
    canvas_authority: native_host.Any | None = None,
    skills_context: native_host.Mapping[str, native_host.Any] | None = None,
    workspace_id: str | None = None,
    workspace_ephemeral: bool | None = None,
    workspace_read_binding_ids: tuple[str, ...] | None = None,
    workspace_write_binding_ids: tuple[str, ...] | None = None,
    workspace_binding_authority: tuple[native_host.Any, ...] | None = None,
    # PR2a Task 7: called with the run id of every sub-agent this turn
    # cancels or abandons, so its still-armed approval cards are failed
    # closed and taken off screen instead of staying pressable for a
    # run that is already over. Forwarded straight to
    # `AgentService(revoke_approvals=...)`; `None` (a caller with no UI)
    # leaves cancellation exactly as it was.
    revoke_approvals: native_host.Callable[[str], object] | None = None,
    plugin_cancel_root: native_host.Callable[[], object] | None = None,
    on_tool_terminal: native_host.Callable[[str, str, str], object] | None = None,
    on_tool_result_terminal: (
        native_host.Callable[[str, str, str, native_host.ToolResult], object] | None
    ) = None,
    on_run_terminal: native_host.Callable[[str], object] | None = None,
    native_tools_enabled: bool | None = None,
    restore_provider_continuation: native_host.ProviderContinuationCheckpoint
    | None = None,
    restore_provider_target: native_host.ContinuationRestoreTarget | None = None,
    expand_provider_continuation: (
        native_host.Callable[[native_host.ProviderContinuationCheckpoint], list[dict]]
        | None
    ) = None,
    resume_provider_continuation: bool = False,
    continuation_sidecar: tuple[native_host.ProviderContinuationSidecar, ...] = (),
    continuation_target: native_host.ContinuationRestoreTarget | None = None,
    continuation_owner_key: str | None = None,
    thinking_sidecar: tuple[native_host.ProviderThinkingSidecar, ...] = (),
    thinking_policy: native_host.ThinkingHistoryPolicy = "auto",
    thinking_owner_key: str | None = None,
    generation_token: int | None = None,
    startup_instruction_candidate: native_host.StartupInstructionCandidate
    | None = None,
    project_instruction_nested_max_bytes: int | None = None,
    run_budget: native_host.RunBudget | None = None,
    confirm_project_instruction_dispatch: (
        native_host.Callable[[native_host.InstructionSnapshot], str] | None
    ) = None,
    on_project_instruction_activation: (
        native_host.Callable[[native_host.ProjectInstructionActivationEvent], None]
        | None
    ) = None,
    propagate_trace_call_persistence_errors: bool = False,
    capture_mode: native_host.ConsoleTraceCaptureMode = native_host.ConsoleTraceCaptureMode.CAPTURE_OFF,
    trace_request: native_host.PreparedConsoleRequest | None = None,
    persona_policy_rules: tuple[native_host.Mapping[str, native_host.Any], ...]
    | None = None,
    profile_context_service: native_host.Any | None = None,
    personal_context_snapshot: native_host.ProfileContextSnapshot | None = None,
    accepted_turn_id: str | None = None,
    parent_budget_deadline: float | None = None,
    on_primary_budget_settled: native_host.Callable[
        [native_host.RunBudget | bool], None
    ]
    | None = None,
    recovery_provider: native_host.Any | None = None,
    code_review_provider: native_host.Any | None = None,
    on_tool_surface_selected: native_host.Callable[[int], None] | None = None,
    _workflow_preparation: native_host.Any | None = None,
):
    from .console_workflow_preparation import current_planning_preparation

    planning = current_planning_preparation()
    if planning is not None and planning is not _workflow_preparation:
        raise PermissionError("original planning producer required")
    replay = getattr(resolution, "reasoning_replay", None)
    if replay is not None and thinking_policy in {"include", "exclude"}:
        resolution = native_host.dataclass_replace(
            resolution,
            reasoning_replay=native_host.dataclass_replace(
                replay,
                mode="all" if thinking_policy == "include" else "off",
                source="conversation",
            ),
        )
    canvas_turn_controller = None
    canvas_run_id = None
    lifecycle_reader = getattr(canvas_provider, "lifecycle_binding", None)
    if callable(lifecycle_reader):
        try:
            lifecycle_binding = lifecycle_reader(canvas_authority)
            if (
                isinstance(lifecycle_binding, tuple)
                and len(lifecycle_binding) == 2
                and isinstance(lifecycle_binding[1], str)
                and lifecycle_binding[1]
            ):
                canvas_turn_controller, canvas_run_id = lifecycle_binding
        except Exception:
            canvas_turn_controller = None
            canvas_run_id = None
    if generation_token is None and _workflow_preparation is None:
        generation_token = self._store.begin_generation_attempt(assistant_message_id)
    # SQLite preparation must finish outside the native identity scope.
    self._store.prepare_progress_inbox(session_id, message_store=self.message_store)
    # Capture before setup can yield to native close/state replacement.
    # Reusing a native ID must not let this delayed run bind its successor.
    with self._store.progress_owner_scope(
        session_id, message_store=self.message_store
    ) as initial_progress_owner_id:
        if initial_progress_owner_id is None or (
            expected_progress_owner_id is not None
            and initial_progress_owner_id != expected_progress_owner_id
        ):
            from tldw_chatbook.Agents.fleet_messages import MessageError

            raise MessageError("unavailable")
    protocol = getattr(resolution, "continuation_protocol", None)
    if continuation_target is None and isinstance(protocol, str) and protocol:
        provider = getattr(resolution, "execution_key", None)
        model_name = getattr(resolution, "model", None)
        base_url = getattr(resolution, "base_url", None)
        if not all(
            isinstance(value, str) and value
            for value in (provider, model_name, base_url)
        ):
            raise ValueError("Provider continuation request is not pinned.")
        continuation_target = native_host.ContinuationRestoreTarget(
            provider=provider,
            protocol=protocol,
            model=model_name,
            api_base_url=base_url,
            selected_provider=getattr(resolution, "selected_provider", None)
            or getattr(resolution, "provider", None),
        )
    if continuation_target is not None and continuation_owner_key is None:
        continuation_owner_key = native_host.CONTINUATION_OWNER_KEY
    # Per-run tool registry + allow-list (Task 12, extended by P5-T6 for
    # MCP, by task-545/T6 for a per-run builtin_gate, and extended again
    # for local tools): rebuilt FRESH for this run whenever there is a
    # skills service, an already-composed MCP or local provider, OR a
    # builtin_gate for this run (never
    # cached across runs, and never the shared self._registry/
    # self._allowed_tools built at construction) -- so a skill or MCP
    # tool approved/edited/revoked since the last run always takes
    # effect on the very next one. `mcp_provider` is built and
    # composed by the CALLER (ConsoleChatController._compose_mcp_
    # provider, on the running Textual main loop, BEFORE this method
    # is dispatched onto asyncio.to_thread) -- see MCPToolProvider's
    # own module docstring for why `compose_catalog()`'s async I/O can
    # never run from inside this worker-thread method. `builtin_gate`
    # MUST route through this fresh-build branch rather than the
    # shared fast path below: the shared path's own `BuiltinToolProvider`
    # is built once at bridge-construction time with `gate=None` (its
    # own lazy default), which would be a SECOND, independently-built
    # gate the run's review hook never stamps -- see
    # `_compose_run_registry_and_allowed`'s own docstring for the
    # desync this would cause. None of skills service, MCP provider,
    # local provider, or builtin_gate: the shipped shared
    # registry/allow-list is used unchanged -- the no-skills, no-MCP,
    # no-local-tools, no-gate path stays
    # byte-identical to before this task (existing callers that never
    # pass `builtin_gate` see no behavior change at all).
    skill_runner = None
    # TASK-1366: this run's UI-side diff channel. When this run takes
    # the fresh-build branch below, the provider's strip seam
    # (BuiltinToolProvider.invoke) appends
    # ``(tool_name, file_path, old, new)`` here BEFORE the raw contents
    # are removed from the LLM/run-log-bound result; the on_step
    # handler below pairs each capture with its STEP_TOOL_RESULT (via
    # ``_pair_step_diff``) and hangs it on the TOOL marker message
    # (session-only ``tool_diff``). Threading: invoke() runs on the
    # tool call's PER-CALL DAEMON THREAD (AgentService.
    # _call_with_timeout) while on_step runs on this run's worker
    # thread. In the normal case the daemon thread is joined before
    # the result step is emitted, so a capture always precedes its
    # step; on timeout/cancel the thread is abandoned unjoined and a
    # late capture can land cross-thread AFTER its step -- the pairing
    # rule (most-recent match, everything older is stale) tolerates
    # both orderings, and deque append/scan/del degrade to a cosmetic
    # missed pairing at worst. The shared fast path's construction-
    # time provider has no sink (a cross-run provider must not capture
    # into one run's queue), so gate-less callers simply get no diff
    # capture -- their rows render exactly as before. Production always
    # passes builtin_gate (console_chat_controller), i.e. the
    # fresh-build branch.
    pending_diffs: native_host.deque[tuple[str, str, str, str]] = native_host.deque()
    # task-4 (skills-fork-reachability): one SkillFileBindings per run,
    # handed to BOTH AgentService (the loop's authorization + reader
    # closure -- Task 3) and this run's _BridgeSkillRunner (which grants
    # a spawned skill's own name pre-spawn) -- never two independently-
    # seeded copies, or the runner's grant would never reach the loop's
    # check. `authorized` starts empty here (Task 5 seeds the turn's
    # $skill names); the reader is a SYNC adapter over the async scope-
    # service read, matching _BridgeSkillRunner.run's own
    # asyncio.run-in-worker-thread pattern just below.
    skill_file_bindings = None
    context: native_host.Mapping[str, native_host.Any] = skills_context or {}
    if self._skills_service is not None and skills_context is None:
        context = native_host.asyncio.run(
            self._skills_service.get_context(mode="local")
        )
    plugin_service = (
        None
        if planning is not None
        else getattr(
            getattr(self._skills_service, "local_service", None),
            "plugin_service",
            None,
        )
    )
    if restore_provider_continuation is not None:
        if plugin_service is not None:
            context = native_host.asyncio.run(
                plugin_service.resume_maximum(
                    context,
                    restore_provider_continuation,
                    self._store._session_or_raise(session_id).persisted_conversation_id
                    or "",
                    assistant_message_id,
                )
            )
        elif restore_provider_continuation.managed_resume is not None:
            raise PermissionError("Managed continuation service unavailable.")
        else:
            context = dict(context, plugin_resume_constraint="zero")
            context["available_skills"] = [
                row
                for row in context.get("available_skills", ())
                if not row.get("plugin_owned")
            ]
            context["context_text"] = "\n".join(
                str(row.get("name", "")) for row in context["available_skills"]
            )
    if plugin_service is not None and (
        context.get("plugin_resume_constraint") is not None
        or context.get("_plugin_requirement_ceiling")
        or any(item.get("plugin_owned") for item in context.get("available_skills", ()))
    ):
        context = native_host.asyncio.run(
            plugin_service.admit(
                context,
                context.get("plugin_run_id") or "pending:" + native_host.uuid4().hex,
            )
        )
    plugin_entries = tuple(
        row for row in context.get("available_skills", ()) if row.get("plugin_owned")
    )
    run_workspace_id = workspace_id
    run_is_ephemeral = bool(workspace_ephemeral)
    if self._store is not None and workspace_id is None:
        try:
            run_workspace_id = self._store.session_workspace_id(session_id)
        except KeyError:
            pass
    if self._store is not None and workspace_ephemeral is None:
        try:
            run_is_ephemeral = self._store.session_is_ephemeral(session_id)
        except KeyError:
            pass
    # TASK-34100.5.3: plain chat's light surface. Beyond what the plan
    # discloses, the runtime tools' closures are not even built, so a
    # model naming one anyway has nothing to dispatch.
    light_surface = self.light_surface_for(
        turn_skill_bindings, session_id=session_id, conversation_id=conversation_id
    )
    if light_surface:
        request_skill_install_confirm = request_skill_script_confirm = None
        request_chat_create_confirm = request_worktree_merge_confirm = None
        managed_skill_promotion_gate = None
    from tldw_chatbook.Skills_Interop.skill_script_runner import (
        sandbox_supported,
    )

    script_tool_enabled = bool(
        self._skills_service is not None
        and request_skill_script_confirm is not None
        and sandbox_supported()
    )
    native_tools = (
        bool(native_tools_enabled)
        if native_tools_enabled is not None
        else (
            True
            if self._native_tools_enabled is None
            else bool(self._native_tools_enabled())
        )
    )
    run_budget = (
        native_host.intersect_console_run_budget(
            run_budget, native_host.console_run_budget()
        )
        if run_budget is not None
        else native_host.console_run_budget()
    )
    if planning is not None:
        run_budget = native_host.dataclass_replace(run_budget, max_subagents=0)
        runtime_definitions, fleet_max_live = (), 0
    else:
        runtime_definitions, fleet_max_live = (
            native_host._console_first_request_runtime_context(self._db, run_budget)
        )
    if plugin_entries and not run_is_ephemeral:
        plugin_catalog = tuple(
            row
            for provider in (
                self._registry,
                mcp_provider,
                local_provider,
                virtual_cli_provider,
                raw_shell_provider,
                library_provider,
                profile_provider,
                canvas_provider,
            )
            if provider is not None
            for row in provider.list_catalog()
        )
        eligible = frozenset(row.name for row in plugin_catalog)
        managed = native_host.asyncio.run(
            plugin_service.agent_presets(
                plugin_entries,
                eligible,
                registry={row.id: row.name for row in plugin_catalog},
                parent_provider=str(
                    getattr(resolution, "selected_provider", "")
                    or getattr(resolution, "execution_key", "")
                    or getattr(resolution, "provider", "")
                ),
            )
        )
        if {item.name for item in managed} & {
            item.name for item in runtime_definitions
        }:
            raise PermissionError("plugin_agent_namespace_collision")
        runtime_definitions = (*runtime_definitions, *managed)
    # Build the exact outbound user payload before schema planning. Both
    # automatic riders can change whether direct catalog disclosure still
    # leaves the configured response reserve, so the planner and live run
    # must receive the same immutable message snapshot.
    planning_messages = agent_messages
    if turn_bundle_block:
        planning_messages, _ = native_host._append_to_last_user_message(
            planning_messages, turn_bundle_block
        )
    diff_feedback_included_ids: list[int] = []
    diff_feedback_included_notes: list[dict] = []
    if _workflow_preparation is None:
        try:
            pending_notes = self._db.pending_notes_for_conversation(conversation_id)
            diff_feedback_block, diff_feedback_included_ids = (
                native_host.render_diff_feedback_block(pending_notes)
            )
            diff_feedback_included_notes = pending_notes[
                : len(diff_feedback_included_ids)
            ]
            if diff_feedback_block:
                planning_messages, attached = native_host._append_to_last_user_message(
                    planning_messages, diff_feedback_block
                )
                if not attached:
                    native_host.logger.warning(
                        "change_review: "
                        f"{len(diff_feedback_included_ids)} pending diff-"
                        "feedback note(s) held back -- no user message "
                        "could carry the block this turn"
                    )
                    diff_feedback_included_ids = []
                    diff_feedback_included_notes = []
        except Exception:  # noqa: BLE001 -- notes must never break the reply
            native_host.logger.opt(exception=True).warning(
                "change_review: could not attach pending diff-feedback notes"
            )
            diff_feedback_included_ids = []
            diff_feedback_included_notes = []
    # advertised) rather than auto-denying every call, the same
    # advertised-equals-usable rule as `install_skill_tool`/
    # `run_skill_script_tool` below. Built up here (before the first
    # request plan) because the plan's schema flags must reflect
    # whether the tools exist.
    # `run_id`: the run's own id only exists once `run_turn` mints it
    # (far below), so the originating assistant message id -- this
    # turn's stable identity, already in scope -- rides in the payload
    # under the `run_id` key instead.
    fork_chat_tool = new_chat_tool = None
    if (
        request_chat_create_confirm is not None
        and execute_agent_chat_create is not None
    ):
        fork_chat_tool, new_chat_tool = native_host.build_chat_create_tool_closures(
            confirm=request_chat_create_confirm,
            execute=execute_agent_chat_create,
            prepare=prepare_agent_chat_create,
            session_id=session_id,
            run_id=assistant_message_id,
        )

    # TASK-32874 (Qodo round, finding 1): the prebuilt plan is the one
    # the Console actually runs with, so the routing flags MUST be
    # computed here or the gated provider/model args are never
    # advertised -- this also un-blinds spawn_subagent's own override
    # args on this path (same flags, same plan builder).
    from tldw_chatbook.Agents.agent_routing import (
        load_agents_routing_config as _load_spawn_routing,
    )
    from tldw_chatbook.Agents.agent_service import spawn_override_targets

    try:
        _spawn_routing_cfg = None if planning is not None else _load_spawn_routing()
    except ValueError:
        _spawn_routing_cfg = None
    chat_create_spawn_override_enabled = bool(
        _spawn_routing_cfg is not None and _spawn_routing_cfg.spawn_override_enabled
    )
    chat_create_spawn_override_targets = (
        spawn_override_targets(native_host.load_settings(), _spawn_routing_cfg)
        if chat_create_spawn_override_enabled
        else ()
    )

    catalog = None
    if _workflow_preparation is not None:
        catalog = native_host._compose_first_request_catalog(
            storage_provider=self._storage_provider,
            shared_registry=self._registry,
            shared_allowed_tools=self._allowed_tools,
            context=context,
            skills_present=planning is None and self._skills_service is not None,
            mcp_provider=mcp_provider,
            builtin_gate=builtin_gate,
            local_provider=local_provider,
            virtual_cli_provider=virtual_cli_provider,
            raw_shell_provider=raw_shell_provider,
            code_review_provider=code_review_provider,
            recovery_provider=recovery_provider,
            library_provider=library_provider,
            library_authority=library_authority,
            profile_provider=profile_provider,
            workspace_id=run_workspace_id,
            workspace_read_binding_ids=workspace_read_binding_ids,
            workspace_write_binding_ids=workspace_write_binding_ids,
            workspace_binding_authority=workspace_binding_authority,
            ephemeral=run_is_ephemeral,
            diff_sink=pending_diffs.append,
            scratch_root=scratch_root,
            scratch_lease=scratch_lease,
            persona_policy_rules=persona_policy_rules,
            canvas_provider=canvas_provider,
            canvas_authority=canvas_authority,
            preview_canvas_profile_snapshot=None,
        )
        planning_messages, catalog = _workflow_preparation.bind_catalog(catalog)
        if planning is None:
            assistant_message_id = (
                _workflow_preparation.authorization.claim.launch_run_id
            )

    first_request_plan = native_host.build_console_first_request_plan(
        storage_provider=self._storage_provider,
        _catalog=catalog,
        shared_registry=self._registry,
        shared_allowed_tools=self._allowed_tools,
        context=context,
        skills_present=planning is None and self._skills_service is not None,
        mcp_provider=mcp_provider,
        builtin_gate=builtin_gate,
        local_provider=local_provider,
        virtual_cli_provider=virtual_cli_provider,
        raw_shell_provider=raw_shell_provider,
        recovery_provider=recovery_provider,
        library_provider=library_provider,
        library_authority=library_authority,
        profile_provider=profile_provider,
        code_review_provider=code_review_provider,
        canvas_provider=canvas_provider,
        canvas_authority=canvas_authority,
        workspace_id=run_workspace_id,
        workspace_read_binding_ids=workspace_read_binding_ids,
        workspace_write_binding_ids=workspace_write_binding_ids,
        workspace_binding_authority=workspace_binding_authority,
        ephemeral=run_is_ephemeral,
        diff_sink=pending_diffs.append,
        scratch_root=scratch_root,
        scratch_lease=scratch_lease,
        resolution=resolution,
        fallback_model=model,
        session_system_prompt=session_system_prompt,
        native_tools=native_tools,
        turn_skill_bindings=turn_skill_bindings,
        turn_bundle_block="",
        install_skill_enabled=bool(
            self._skills_service is not None
            and request_skill_install_confirm is not None
        ),
        run_skill_script_enabled=script_tool_enabled,
        fork_chat_enabled=bool(fork_chat_tool is not None),
        new_chat_enabled=bool(new_chat_tool is not None),
        spawn_override_enabled=chat_create_spawn_override_enabled,
        spawn_override_targets=chat_create_spawn_override_targets,
        worktree_merge_enabled=request_worktree_merge_confirm is not None,
        agent_messages=planning_messages,
        agent_definitions=runtime_definitions,
        fleet_max_live=fleet_max_live,
        run_budget=run_budget,
        persona_policy_rules=persona_policy_rules,
        profile_context_service=profile_context_service,
        personal_context_snapshot=personal_context_snapshot,
        progress_inbox_exists=planning is None
        and self._session_progress_inbox(session_id) is not None,
        light_surface=light_surface,
    )
    self._remember_tool_surface(
        session_id, first_request_plan, light_surface=light_surface
    )
    if on_tool_surface_selected is not None:
        on_tool_surface_selected(
            len(first_request_plan.schemas.runtime_schemas)
            + len(first_request_plan.schemas.active_schemas)
        )
    registry = first_request_plan.registry
    config = first_request_plan.config
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

    automatic = current_automatic_work()
    if automatic is not None and automatic.goal is not None:
        config = automatic.goal.narrow_config(config, registry)
    project_instruction_context = None
    service_confirm_project_instruction_dispatch = confirm_project_instruction_dispatch
    if startup_instruction_candidate is not None:
        project_instruction_context = native_host._ProjectInstructionDispatchContext(
            nested_max_bytes=native_host.coerce_int_setting(
                project_instruction_nested_max_bytes
                if project_instruction_nested_max_bytes is not None
                else native_host.get_cli_setting(
                    "console",
                    "project_instructions_nested_max_bytes",
                    native_host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                ),
                native_host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                minimum=native_host.MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                maximum=native_host.MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            ),
            on_activation=on_project_instruction_activation,
            final_payload_fits=(
                None
                if config.native_tools
                and native_host.provider_supports_native_tools(
                    first_request_plan.api_endpoint,
                    reasoning_replay=config.reasoning_replay,
                )
                else lambda rows: native_host._fenced_project_instruction_payload_fits(
                    rows,
                    model=config.model,
                    provider=first_request_plan.api_endpoint,
                    response_reserve_tokens=config.response_reserve_tokens,
                    reasoning_replay=config.reasoning_replay,
                )
            ),
        )
        original_confirm = confirm_project_instruction_dispatch

        def service_confirm_project_instruction_dispatch(snapshot):
            project_instruction_context.accept_primary_snapshot(snapshot)
            bind_promotion_context = getattr(
                local_provider, "bind_instruction_promotion_context", None
            )
            if callable(bind_promotion_context):
                bind_promotion_context(
                    snapshotter=project_instruction_context.snapshot_promotion_target,
                    revalidator=project_instruction_context.revalidate_promotion_target,
                )
            decision = "cancel"
            try:
                decision = (
                    original_confirm(snapshot)
                    if original_confirm is not None
                    else "cancel"
                )
                return decision
            finally:
                if decision != "proceed":
                    project_instruction_context.discard_primary_snapshot()
                    unbind_promotion_context = getattr(
                        local_provider, "unbind_instruction_promotion_context", None
                    )
                    if callable(unbind_promotion_context):
                        unbind_promotion_context()

    admitted_skill_definition_digests = {
        str(item["name"]): str(item["definition_digest"])
        for item in native_host._eligible_skill_entries(context)
        if item.get("name") and item.get("definition_digest")
    }
    admitted_skill_definition_digests.update(
        (entry["name"], entry["definition_digest"]) for entry in plugin_entries
    )
    if self._skills_service is not None:
        local_skills = getattr(self._skills_service, "local_service", None)
        if local_skills is None and hasattr(self._skills_service, "trust_service"):
            local_skills = self._skills_service
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner

        from .console_ordinary_tree import current_ordinary_tree

        ordinary = current_ordinary_tree()
        preparation = getattr(ordinary, "skill_preparation", None)
        owner = current_execution_owner()
        if (
            preparation is not None
            and ordinary is not None
            and preparation.trust_source is not None
            and preparation.trust_source.local is local_skills
            and owner in ordinary.members
            and owner.is_admitted
        ):
            from .console_configuration_preparation import (
                finish_native_skill_trust_publication,
                native_skill_trust_publication,
            )

            with native_skill_trust_publication(preparation) as publication:
                trust_service = getattr(local_skills, "trust_service", None)
                finish_native_skill_trust_publication(publication, trust_service)
        else:
            trust_service = getattr(local_skills, "trust_service", None)

        def current_skill_definition_digest(skill_name: str) -> str | None:
            for entry in plugin_entries:
                if skill_name in {entry["name"], entry["tool_name"]}:
                    return plugin_service.current_definition(
                        skill_name, entry["plugin_admission"]
                    )
            if trust_service is None:
                return None
            return trust_service.current_fingerprint_digest(skill_name)

        skill_file_bindings = native_host.SkillFileBindings(
            authorized=set(),
            reader=lambda skill_name, path: native_host.asyncio.run(
                self._skills_service.read_skill_file(
                    skill_name,
                    path,
                    mode="local",
                    **next(
                        (
                            {"plugin_admission": entry["plugin_admission"]}
                            for entry in plugin_entries
                            if skill_name in {entry["name"], entry["tool_name"]}
                        ),
                        {},
                    ),
                )
            ),
            current_definition_digest=current_skill_definition_digest,
        )
        skill_runner = native_host._BridgeSkillRunner(
            skills_service=self._skills_service,
            plugin_entries=plugin_entries,
            plugin_registry=registry,
            plugin_eligible=frozenset(first_request_plan.config.allowed_tools),
            plugin_provider=str(
                getattr(resolution, "selected_provider", "")
                or getattr(resolution, "execution_key", "")
                or getattr(resolution, "provider", "")
            ),
            skill_names=first_request_plan.skill_names,
            builtin_names=first_request_plan.builtin_names,
            local_names=first_request_plan.local_names,
            owned_mcp_names=tuple(
                entry.name
                for entry in registry.list_catalog()
                if entry.source == "plugin_mcp"
                and entry.name in first_request_plan.config.allowed_tools
            ),
            skill_file_bindings=skill_file_bindings,
            definition_digests={
                name: digest
                for name, digest in admitted_skill_definition_digests.items()
                if name in first_request_plan.skill_names
            },
        )
    prepare_managed_skill_promotion_tool = None
    if self._skills_service is not None and managed_skill_promotion_gate is not None:
        managed_skill_promotion_gate.bind_reader(
            lambda skill_name: native_host.asyncio.run(
                self._skills_service.get_skill(skill_name, mode="local")
            )
        )
        prepare_managed_skill_promotion_tool = managed_skill_promotion_gate.invoke
    # task-5 (skills-fork-reachability): seed this run's own bindings
    # with the names the CONTROLLER already resolved/spliced for the
    # triggering turn (a leading `$skill` mention, or embedded mentions
    # that actually spliced) -- so the primary agent's very first turn
    # can already read that skill's bundle via skill_file, matching
    # what a spawned skill child gets for its OWN bundle (Task 4).
    # `skill_file_bindings` is None whenever there is no skills service
    # for this run, in which case a non-empty `turn_skill_bindings`
    # (which can only happen when the controller's own skills-service-
    # gated substitution ran) has nothing to seed.
    if skill_file_bindings is not None:
        for skill_name in turn_skill_bindings:
            skill_file_bindings.authorize(
                skill_name,
                admitted_skill_definition_digests.get(skill_name),
            )

    # Agent-callable skill install (5th runtime tool). Built only when
    # BOTH a skills service AND a confirm callback exist -- without a
    # callback the tool is simply absent (never advertised) rather than
    # auto-denying every call; wired to AgentService, which pins/
    # dispatches it for the top-level agent only. Order (load-bearing):
    # enforce policy (no prompt on denial) -> classify URL (no prompt on
    # a bad URL) -> in-chat confirm (plain blocking call, OUTSIDE asyncio.run)
    # -> asyncio.run(install) -> broad-catch wrap. import_skill_file
    # raises a bare ValueError("local_skill_exists:...") on collision, so
    # the install catch is broad.
    def goal_tool_refusal(name: str) -> native_host.ToolResult | None:
        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
        from tldw_chatbook.Agents.automatic_work_runtime import (
            current_automatic_work,
        )

        context = current_automatic_work()
        if context is None or context.goal is None:
            return None
        try:
            context.check()
            if not context.goal.permits_runtime(name):
                raise AutomaticWorkRefused("goal_tool_scope")
        except AutomaticWorkRefused as exc:
            return native_host.ToolResult(
                ok=False,
                error=exc.reason,
                termination_reason=exc.termination_reason,
            )
        return None

    from tldw_chatbook.Agents.agent_models import RunTerminationReason

    def goal_tool_failure(
        message: str, reason: RunTerminationReason
    ) -> native_host.ToolResult:
        return native_host.ToolResult(
            ok=False,
            error=message,
            termination_reason=(
                reason if automatic is not None and automatic.goal is not None else None
            ),
        )

    install_skill_tool = None
    if self._skills_service is not None and request_skill_install_confirm is not None:
        scope = self._skills_service

        def install_skill_tool(url: str) -> native_host.ToolResult:
            from tldw_chatbook.Agents.automatic_work_budget import (
                AutomaticWorkRefused,
            )

            refusal = goal_tool_refusal("install_skill")
            if refusal is not None:
                return refusal
            from tldw_chatbook.Skills_Interop.skill_remote_fetch import (
                classify_skill_source_url,
                install_skill_from_url,
            )
            from tldw_chatbook.runtime_policy.types import PolicyDeniedError

            try:
                scope.enforce_install_remote()
            except PolicyDeniedError as exc:
                return goal_tool_failure(
                    exc.user_message, RunTerminationReason.PERMISSION_REFUSED
                )
            except Exception as exc:  # noqa: BLE001
                return goal_tool_failure(
                    str(exc), RunTerminationReason.AUTHORITY_CHANGED
                )
            try:
                classify_skill_source_url(url)
            except Exception as exc:  # noqa: BLE001 (RemoteSkillError etc.)
                return goal_tool_failure(
                    str(exc), RunTerminationReason.AUTHORITY_CHANGED
                )
            try:
                allowed = bool(
                    request_skill_install_confirm(url)
                    if request_skill_install_confirm is not None
                    else False
                )
            except Exception:  # noqa: BLE001 — a UI error fails closed
                allowed = False
            if not allowed:
                return goal_tool_failure(
                    "The user declined to install this skill.",
                    RunTerminationReason.PERMISSION_REFUSED,
                )
            try:
                from tldw_chatbook.Agents.automatic_work_runtime import (
                    current_automatic_work,
                )

                automatic_work = current_automatic_work()
                if automatic_work is not None:
                    automatic_work.check()
                refusal = goal_tool_refusal("install_skill")
                if refusal is not None:
                    return refusal
                result = native_host.asyncio.run(
                    install_skill_from_url(url, scope_service=scope)
                )
            except AutomaticWorkRefused as exc:
                return goal_tool_failure(exc.reason, exc.termination_reason)
            except PolicyDeniedError as exc:
                return goal_tool_failure(
                    exc.user_message, RunTerminationReason.PERMISSION_REFUSED
                )
            except Exception as exc:  # noqa: BLE001
                return goal_tool_failure(str(exc), RunTerminationReason.UNKNOWN_EFFECT)
            name = result.get("name", "") if isinstance(result, dict) else ""
            return native_host.ToolResult(
                ok=True,
                content=(
                    f'Installed "{name}" — it is pending your review and '
                    "cannot run until you approve it in Library > Skills."
                ),
            )

    # Trust-gated skill script execution (6th runtime tool). Built only
    # when a skills service AND a confirm callback exist AND this
    # platform's sandbox is actually usable -- without any one of the
    # three the tool is absent (never advertised) rather than
    # auto-denying every call. The platform check matters because the
    # sandbox (skill_script_runner.run_script_subprocess) depends on
    # POSIX-only primitives (process-group teardown, RLIMIT_* via the
    # `resource` module) that do not exist on Windows: "advertised must
    # equal usable" -- a tool the model can call but that always raises
    # is a defect, not a graceful degradation, so it must simply not be
    # wired on an unsupported platform. Order (load-bearing): enforce
    # policy (no prompt on denial) -> describe/resolve (no prompt on a
    # bad path or an unrunnable type) -> grant check (no prompt when the
    # user already granted this skill) -> confirm (plain blocking call,
    # OUTSIDE any asyncio.run) -> run -> broad-catch wrap.
    # run_skill_script re-verifies policy/trust/path authoritatively, so
    # a stale plan can never widen what actually executes.
    run_skill_script_tool = None
    if script_tool_enabled:
        scope = self._skills_service
        trust_service = getattr(
            getattr(scope, "local_service", None), "trust_service", None
        )

        def run_skill_script_tool(
            skill_name: str, script_path: str, args: list[str]
        ) -> native_host.ToolResult:
            from tldw_chatbook.Agents.automatic_work_budget import (
                AutomaticWorkRefused,
            )
            from tldw_chatbook.Agents.workflow_task_scope import (
                check_current_task_call,
            )

            task_call = native_host.ToolCall(
                "run_skill_script",
                {"skill": skill_name, "script": script_path, "args": args},
            )
            try:
                check_current_task_call(task_call)
            except Exception:  # noqa: BLE001 - authority failures must refuse
                return native_host.ToolResult.blocked("workflow task scope refused")

            refusal = goal_tool_refusal("run_skill_script")
            if refusal is not None:
                return refusal
            from tldw_chatbook.runtime_policy.types import PolicyDeniedError
            from tldw_chatbook.Skills_Interop.skill_trust_models import (
                SkillTrustBlockedError,
            )

            try:
                scope.enforce_run_script()
            except PolicyDeniedError as exc:
                return goal_tool_failure(
                    exc.user_message, RunTerminationReason.PERMISSION_REFUSED
                )
            except Exception as exc:  # noqa: BLE001
                return goal_tool_failure(
                    str(exc), RunTerminationReason.AUTHORITY_CHANGED
                )
            try:
                plan = native_host.asyncio.run(
                    scope.describe_skill_script(skill_name, script_path)
                )
            except Exception as exc:  # noqa: BLE001 (trust/path/type)
                return goal_tool_failure(
                    f"run_skill_script: {exc}",
                    RunTerminationReason.AUTHORITY_CHANGED,
                )

            granted = False
            if trust_service is not None:
                try:
                    granted = bool(trust_service.script_execution_granted(skill_name))
                except Exception:  # noqa: BLE001 — doubt ⇒ prompt
                    granted = False
            if not granted:
                try:
                    decision = request_skill_script_confirm(
                        {
                            # plan.skill_name, not the agent's raw
                            # spelling: the service normalizes the name
                            # it acts on, and a consent card must show
                            # the value that will actually be used.
                            "skill_name": str(
                                getattr(plan, "skill_name", None) or skill_name
                            ),
                            "script_path": script_path,
                            "mechanism": plan.mechanism,
                            "interpreter": plan.interpreter_display,
                            "is_binary": plan.is_binary,
                            "args": [str(a) for a in args],
                        }
                    )
                except Exception:  # noqa: BLE001 — a UI error fails closed
                    decision = {"allow": False, "remember": False}
                if not isinstance(decision, native_host.Mapping):
                    decision = {"allow": False, "remember": False}
                if not decision.get("allow", False):
                    return goal_tool_failure(
                        "The user declined to run this script.",
                        RunTerminationReason.PERMISSION_REFUSED,
                    )
                if decision.get("remember", False) and trust_service is not None:
                    # Deliberate ordering: this persists the standing
                    # grant BEFORE run_skill_script below actually runs,
                    # so "remember my choice" sticks even if this
                    # particular run then fails (e.g. trust revoked
                    # mid-flight). That is fine -- run_skill_script
                    # re-verifies policy/trust/path authoritatively on
                    # every call regardless of this grant, so recording
                    # it early never widens what is allowed to execute.
                    try:
                        trust_service.grant_script_execution(skill_name)
                    except Exception:  # noqa: BLE001 — grant is best-effort
                        native_host.logger.opt(exception=True).debug(
                            "Failed to persist skill script grant"
                        )
            try:
                from tldw_chatbook.Agents.automatic_work_runtime import (
                    current_automatic_work,
                )

                automatic_work = current_automatic_work()
                if automatic_work is not None:
                    automatic_work.check()
                check_current_task_call(task_call)
                refusal = goal_tool_refusal("run_skill_script")
                if refusal is not None:
                    return refusal
                if scratch_root is not None and scratch_lease is not None:
                    with scratch_lease():
                        outcome = native_host.asyncio.run(
                            scope.run_skill_script(
                                skill_name,
                                script_path,
                                list(args),
                                output_root=(scratch_root / "skill_script_output"),
                            )
                        )
                else:
                    outcome = native_host.asyncio.run(
                        scope.run_skill_script(
                            skill_name,
                            script_path,
                            list(args),
                        )
                    )
            except AutomaticWorkRefused as exc:
                return goal_tool_failure(exc.reason, exc.termination_reason)
            except PolicyDeniedError as exc:
                return goal_tool_failure(
                    exc.user_message, RunTerminationReason.PERMISSION_REFUSED
                )
            except SkillTrustBlockedError:
                return goal_tool_failure(
                    "Skill trust changed.", RunTerminationReason.AUTHORITY_CHANGED
                )
            except Exception as exc:  # noqa: BLE001
                return goal_tool_failure(
                    f"run_skill_script: {exc}", RunTerminationReason.UNKNOWN_EFFECT
                )

            lines = [f"exit_code: {outcome.exit_code}"]
            if outcome.goal_evidence_id:
                lines.append(f"goal_evidence_id: {outcome.goal_evidence_id}")
            if outcome.timed_out:
                lines.append("timed out — the script was killed")
            if outcome.output_capped:
                lines.append("output was truncated at the size cap")
            for warning in outcome.sandbox_warnings:
                lines.append(f"note: {warning}")
            if outcome.stdout:
                lines.append(f"stdout:\n{outcome.stdout}")
            if outcome.stderr:
                lines.append(f"stderr:\n{outcome.stderr}")
            # task-584: report WHAT was produced, never its contents. This
            # string enters the model's context, and a script's artifact is
            # not trust-reviewed material -- the listing is name + size, and
            # the path lets the user (or a file tool, where enabled) open it.
            if outcome.output_files:
                listed = ", ".join(
                    f"{item['name']} ({item['size']} bytes)"
                    for item in outcome.output_files
                )
                lines.append(f"produced {len(outcome.output_files)} file(s): {listed}")
                display_output_dir = str(outcome.output_dir)
                if scratch_root is not None and outcome.output_dir is not None:
                    try:
                        display_output_dir = str(
                            native_host.Path(outcome.output_dir)
                            .resolve()
                            .relative_to(scratch_root.resolve())
                        )
                    except ValueError:
                        display_output_dir = "private scratch output"
                lines.append(f"output directory: {display_output_dir}")
            return native_host.ToolResult(
                ok=True,
                content="\n".join(lines),
                termination_reason=RunTerminationReason.UNKNOWN_EFFECT
                if automatic_work is not None
                and automatic_work.goal is not None
                and outcome.exit_code is None
                else None,
            )

    # One event loop for the whole run (PR #629 Fix 1(c)): every turn
    # this run makes -- primary tool-call turns, any sub-agent turns,
    # and the final-answer turn -- bridges through this same loop via
    # `_StreamingModelAdapter.chat_call`, instead of each turn spinning
    # up (and tearing down) its own loop via `asyncio.run()`. That
    # per-turn churn forced a client swap on the gateway's owned httpx
    # client every single turn (see
    # `ConsoleProviderGateway._active_http_client`); reusing one loop
    # for the whole run means at most one swap per run.
    #
    # PR2a Task 6.5: the loop gets its OWN thread. It is no longer
    # driven by `run_until_complete` from whichever agent's thread is
    # calling, because with the fleet ON by default several of them
    # call at once and a loop may only ever be driven by one thread --
    # the second concurrent driver raises "This event loop is already
    # running" and kills that child's run. `chat_call` now submits
    # with `run_coroutine_threadsafe`, so this thread is the single
    # driver and the callers merely wait on their own futures.
    # Constructed here (with the loop it drives) but STARTED only just
    # before the try/finally that owns its shutdown, a few hundred
    # lines below: a raise in the composition work between the two
    # would otherwise leave a daemon thread spinning `run_forever`
    # forever with nothing to stop it. Same ordering rule -- and the
    # same failure -- as `AgentService`'s own `thread.start()` guard.
    #
    # PR3a-1 Task 1: the loop+thread pair is now a `_ModelCallLifeline`
    # (identical construction, start ordering and teardown -- see its
    # docstring), because a fleet CHILD now owns one of its own from
    # birth via `adapter.child_lifeline`. This one stays exactly what
    # it always was: the PRIMARY agent's, turn-scoped.
    turn_lifeline = (
        native_host._ModelCallLifeline(
            "console-agent-loop",
            close_current_loop=getattr(self._gateway, "aclose_current_loop", None),
        )
        if _workflow_preparation is None or planning is not None
        else None
    )
    if _workflow_preparation is not None:
        _workflow_preparation.turn_lifeline = turn_lifeline
    thinking_capture = native_host.ThinkingCapture(
        assistant_owner_id=assistant_message_id
    )
    adapter = native_host._StreamingModelAdapter(
        store=self._store,
        provider_gateway=self._gateway,
        resolution=resolution,
        assistant_message_id=assistant_message_id,
        session_id=session_id if _workflow_preparation is not None else None,
        should_cancel=should_cancel,
        loop=turn_lifeline.loop if turn_lifeline is not None else None,
        native_tools=(
            config.native_tools
            and native_host.provider_supports_native_tools(
                first_request_plan.api_endpoint,
                reasoning_replay=config.reasoning_replay,
            )
        ),
        provider_stream_signals=provider_stream_signals,
        continuation_sidecar=continuation_sidecar,
        continuation_target=continuation_target,
        continuation_owner_key=continuation_owner_key,
        thinking_sidecar=thinking_sidecar,
        thinking_policy=thinking_policy,
        thinking_owner_key=thinking_owner_key,
        thinking_capture=thinking_capture,
        generation_token=generation_token,
        capture_mode=capture_mode,
        trace_request=trace_request,
    )

    # PR3a-1 Task 6b (audit F1): this turn's own key into
    # `self._live[conversation_id]`. Minted here because `run_turn`
    # only hands back the primary run id when the turn ENDS, and the
    # rail must show "running" from the moment it starts; every write
    # below (start, each primary step, the terminal publish) uses this
    # one key, so an earlier turn's surviving child -- which writes
    # under its OWN run id -- can never land in it.
    primary_live_key = native_host.uuid4().hex
    adapter._live_usage_sink = native_host.functools.partial(
        self._observe_live_usage,
        conversation_id=conversation_id,
    )
    child_change_state = native_host._ChildChangeState(
        owner_key=primary_live_key,
        survivor_key=assistant_message_id,
    )
    child_path_root = None
    if scratch_root is not None:
        try:
            child_path_root = scratch_root.expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            native_host.logger.warning(
                "change_review: could not normalize the attributed child WRITE root"
            )
    live_steps = native_host._LiveStepFeed()
    subagents: list[native_host.SubAgentSummary] = []
    #: run key -> that run's own live step feed. The primary's entry is
    #: `live_steps` itself, so every existing reader of that local is
    #: unchanged.
    run_live_steps: dict[str, native_host._LiveStepFeed] = {
        primary_live_key: live_steps
    }
    buddy_tool_sequences: dict[str, native_host.deque[int]] = {}
    primary_buddy_run_ids: set[str] = set()
    raw_shell_progress_run_ids: set[str] = set()
    if _workflow_preparation is None:
        self._publish_live(
            conversation_id,
            primary_live_key,
            native_host.AgentLiveSnapshot(status="running"),
            primary=True,
        )
        # PR3a-1 Task 6b (audit F1): bound the per-run slots -- one turn's
        # worth plus whatever is still live. Must run AFTER the publish
        # above, which is what tells it which key is this turn's.
        self._prune_live_run_slots(conversation_id)
        # A live run is starting -- live_snapshot takes over as the rail's
        # source of truth for this conversation from here on, so any
        # previously cached historical (DB-derived) summary is stale.
        self._historical_cache.pop(conversation_id, None)
    planning_deriver = native_host._PendingPrimaryPlanningDeriver()
    from .console_tool_activity import ConsoleToolActivity

    tool_activity = ConsoleToolActivity(self._store, session_id)
    tool_activity_run_ids: set[str] = set()

    def on_tool_activity(
        step: native_host.AgentStep, agent_kind: str, run_id: str
    ) -> None:
        if agent_kind != native_host.AGENT_KIND_PRIMARY or not run_id:
            return
        self._tool_activity_runs[run_id] = tool_activity
        tool_activity_run_ids.add(run_id)
        if step.kind == "tool_proposed":
            planning = planning_deriver.observe(
                native_host.dataclass_replace(step, kind=native_host.STEP_TOOL_CALL),
                agent_kind,
                actual_thinking_round_ordinals=native_host._thinking_round_ordinals(
                    thinking_capture.snapshot().envelope
                ),
            )
            if planning is not None:
                self._append_marker(
                    session_id,
                    planning.content,
                    activity_presentation=planning.activity_presentation,
                    activity_round_ordinal=planning.activity_round_ordinal,
                )
        tool_activity.observe(step, planning_deriver.active_round_ordinal)

    def on_resolved_target(run_id: str, provider: str, model: str) -> None:
        for index, summary in enumerate(subagents):
            if summary.run_id != run_id:
                continue
            subagents[index] = native_host.dataclass_replace(
                summary, resolved_provider=provider, resolved_model=model
            )
            snapshot = self._live.get(conversation_id, {}).get(primary_live_key)
            if snapshot is not None:
                self._publish_live(
                    conversation_id,
                    primary_live_key,
                    native_host.dataclass_replace(snapshot, subagents=tuple(subagents)),
                    primary=False,
                )
            break

    def on_step(step: native_host.AgentStep, agent_kind: str, run_id: str) -> None:
        if agent_kind == native_host.AGENT_KIND_PRIMARY and run_id:
            self._live_primary_runs[conversation_id] = run_id
        # PR 2a (task-3): AgentService now attributes every step to its
        # run id so a fleet of concurrent children can be told apart.
        # PR3a-1 Task 6b (audit F1): that attribution is finally USED.
        # PR 2b Task 2 argued it was redundant because the bridge runs
        # one primary at a time -- true only while a run cannot outlive
        # its turn. It can now, so a child's step arriving after this
        # turn returned would otherwise be published into the slot the
        # NEXT turn owns: wrong step list, wrong count, and a
        # `status="running"` that never clears. Each run's feed and
        # each run's published snapshot are keyed by that run instead.
        # (`service.fleet_snapshot()` below still supplies the
        # `subagents` rows, and is still the authority for a child's
        # STATUS -- this splits the STEP feed, not the row list.)
        live_key = (
            primary_live_key
            if agent_kind == native_host.AGENT_KIND_PRIMARY
            # An empty run id (no run attributed) falls back to this
            # turn's own key -- the pre-fix destination, so nothing
            # regresses for a step that cannot be attributed.
            else (run_id or primary_live_key)
        )
        if agent_kind != native_host.AGENT_KIND_PRIMARY and run_id:
            touched_paths = native_host.ChangeTurnTracker.tool_touched_paths((step,))
            normalized_paths: list[str] = []
            inherited_claim = None
            for raw_path in touched_paths:
                try:
                    path = native_host.Path(raw_path)
                    if child_path_root is not None:
                        normalized_paths.append(
                            str(
                                native_host.validate_path(
                                    path,
                                    child_path_root,
                                    redact_paths=True,
                                    allow_hidden=True,
                                )
                            )
                        )
                    elif path.is_absolute():
                        normalized_paths.append(str(path))
                    else:
                        normalized_paths.append(raw_path)
                except ValueError:
                    native_host.logger.warning(
                        "change_review: could not normalize one attributed "
                        "child WRITE path"
                    )
            with self._change_window_lock:
                child_change_state.touched_paths.update(normalized_paths)
                window = self._post_turn_change_windows.get(conversation_id)
                if (
                    touched_paths
                    and window is not None
                    and window.successor_claim is not None
                    and any(
                        state is child_change_state for state in window.child_states
                    )
                ):
                    inherited_claim = window.successor_claim
            if normalized_paths and self._change_finalization_coordinator is not None:
                self._change_finalization_coordinator.record_survivor_paths(
                    assistant_message_id,
                    normalized_paths,
                )
            if inherited_claim is not None:
                try:
                    claim_ready = inherited_claim.ready.wait(
                        native_host._CHANGE_BOUNDARY_WAIT_SECONDS
                    )
                except Exception:  # noqa: BLE001 -- tracking is best effort
                    claim_ready = False
                claim_failed = True
                claim_handle = None
                if claim_ready:
                    with self._change_window_lock:
                        claim_failed = inherited_claim.failed
                        claim_handle = inherited_claim.handle
                baseline_trusted = False
                if not claim_failed and claim_handle is not None:
                    try:
                        claim_handle.await_baseline()
                        baselines = dict(claim_handle.baselines)
                        roots = tuple(claim_handle.roots)
                        baseline_trusted = not claim_handle.errors and all(
                            baselines.get(str(root)) for root in roots
                        )
                    except Exception:  # noqa: BLE001 -- tracking is best effort
                        baseline_trusted = False
                if not baseline_trusted:
                    with self._change_window_lock:
                        inherited_claim.failed = True
        buddy_run_id = run_id or live_key
        buddy_sink = self._buddy_sink
        if buddy_sink is not None:
            if agent_kind == native_host.AGENT_KIND_PRIMARY:
                primary_buddy_run_ids.add(buddy_run_id)
            if step.kind == native_host.STEP_TOOL_CALL:
                buddy_tool_sequences.setdefault(
                    buddy_run_id, native_host.deque()
                ).append(step.index)
                buddy_sink.tool_step(
                    buddy_run_id, step.index, step.kind, session_id=session_id
                )
            elif step.kind == native_host.STEP_TOOL_RESULT:
                sequences = buddy_tool_sequences.get(buddy_run_id)
                sequence = sequences.popleft() if sequences else step.index
                buddy_sink.tool_step(
                    buddy_run_id, sequence, step.kind, session_id=session_id
                )
                if sequences is not None and not sequences:
                    buddy_tool_sequences.pop(buddy_run_id, None)
            elif step.kind == native_host.STEP_ERROR:
                buddy_sink.release_run(buddy_run_id)
        key_steps = run_live_steps.setdefault(live_key, native_host._LiveStepFeed())
        key_steps.append(
            # `time.monotonic()` HERE is the step's real start: this hook
            # runs synchronously inside the runtime loop, before the work
            # the step announces (a STEP_TOOL_CALL lands one statement
            # ahead of `deps.invoke_tool`). See `AgentLiveStep.started_at`.
            native_host.AgentLiveStep(
                step.kind,
                self._summarize(step),
                agent_kind,
                native_host.time.monotonic(),
            )
        )
        # TASK-1366: pair this result step with the raw before/after
        # contents the provider's diff_sink captured at the strip seam,
        # when this call was a diff-carrying file write. Sub-agent
        # result steps pair-and-discard too (their writes can ride this
        # run's provider) even though only primary steps drop markers.
        # See _pair_step_diff for the threading model and the staleness
        # rule that keeps an abandoned call's late capture from pairing
        # with a later write.
        tool_diff: tuple[str, str, str] | None = None
        if step.kind == native_host.STEP_TOOL_RESULT and pending_diffs:
            tool_diff = native_host._pair_step_diff(pending_diffs, step.tool_name)
        planning_marker = planning_deriver.observe(
            step,
            agent_kind,
            actual_thinking_round_ordinals=native_host._thinking_round_ordinals(
                thinking_capture.snapshot().envelope
            ),
        )
        if agent_kind == native_host.AGENT_KIND_PRIMARY:
            # Adoption may succeed before its display update raises. Register
            # cleanup first so an interrupted run still detaches that marker.
            if step.tool_name == "shell_exec" and run_id:
                raw_shell_progress_run_ids.add(run_id)
            raw_shell_projected = self._project_raw_shell_step(
                session_id,
                run_id,
                step,
                agent_kind,
            )
            if planning_marker is not None:
                self._append_marker(
                    session_id,
                    planning_marker.content,
                    full_output=planning_marker.tool_output_full,
                    activity_presentation=(planning_marker.activity_presentation),
                    activity_round_ordinal=(planning_marker.activity_round_ordinal),
                )
            if step.kind == native_host.STEP_SPAWN:
                # PR2b Task 2: this is this run's ONLY source of rows
                # on the inline path (fleet off, `[agents]
                # max_live_subagents <= 1`) -- there is no coordinator
                # there, ever, so every entry appended here stays live
                # for the run's whole duration. On a fleet-ON run it
                # is superseded, publish by publish, as soon as ANY
                # handle has been reserved (`_subagent_summaries_
                # from_fleet` then uses `service.fleet_snapshot()`
                # exclusively, not a merge -- see that function's own
                # docstring for why a positional merge was tried and
                # reverted: it silently duplicated a child's row
                # forever whenever an EARLIER spawn in the same run
                # had been refused, e.g. an unknown named agent).
                # Entries appended here past that point are still
                # correct, just unused by that publish.
                subagents.append(native_host.SubAgentSummary(step.summary or ""))
            # format_agent_step_marker is the single source of truth for
            # marker text -- shared with resume_marker_messages below --
            # so live and resume-rebuilt transcripts render identically
            # (Plan-B final-review Medium-1). See its docstring for why
            # the text must stay raw/unescaped.
            marker_text = (
                None
                if raw_shell_projected
                else native_host.format_agent_step_marker(
                    step.kind,
                    tool_name=step.tool_name,
                    result=step.result,
                    summary=step.summary,
                )
            )
            if step.kind == native_host.STEP_TOOL_RESULT and not raw_shell_projected:
                lifecycle_content = marker_text or f"⚙ {step.tool_name} → {step.result}"
                if tool_activity.complete(
                    step,
                    native_host.build_step_activity_presentation(
                        step.kind,
                        tool_name=step.tool_name,
                        result=step.result,
                        tool_outcome=step.tool_outcome,
                    ),
                    lifecycle_content,
                    tool_diff,
                    record_trajectory=marker_text is not None,
                ):
                    marker_text = None
            if marker_text is not None:
                self._append_marker(
                    session_id,
                    marker_text,
                    full_output=native_host.full_step_output(
                        step.kind,
                        result=step.result,
                        summary=step.summary,
                        marker_text=marker_text,
                    ),
                    tool_diff=tool_diff,
                    activity_presentation=native_host.build_step_activity_presentation(
                        step.kind,
                        tool_name=step.tool_name,
                        result=step.result,
                        tool_outcome=step.tool_outcome,
                    ),
                    activity_round_ordinal=(planning_deriver.active_round_ordinal),
                )
        # Content-free operational logging for tool outcomes. The actual
        # invocation lives inside AgentService, so this intentionally
        # records no arguments, results, summaries, or provider ids.
        if step.kind == native_host.STEP_TOOL_RESULT:
            native_host.logger.debug(
                "agent tool call: agent_kind={agent_kind} tool={tool_name} "
                "outcome=completed step={step_index}",
                agent_kind=agent_kind,
                tool_name=step.tool_name,
                step_index=step.index,
            )
        elif step.kind == native_host.STEP_ERROR:
            native_host.logger.warning(
                "agent step error: agent_kind={agent_kind} tool={tool_name} "
                "outcome=error step={step_index}",
                agent_kind=agent_kind,
                tool_name=step.tool_name,
                step_index=step.index,
            )
        self._publish_live(
            conversation_id,
            live_key,
            native_host.AgentLiveSnapshot(
                status="running",
                step=key_steps.count,
                steps=key_steps.tail(5),
                # PR2b Task 2: rebuilt from the fleet's REAL live state
                # on every publish, not appended once and left stuck at
                # the "running" default -- see
                # `_subagent_summaries_from_fleet`'s docstring.
                subagents=native_host._subagent_summaries_from_fleet(
                    service.fleet_snapshot(), subagents
                ),
            ),
            # PR3a-1 Task 6b: a step from a child NEVER moves the rail's
            # summary pointer. Only `run_reply` (this turn) may.
            primary=False,
        )

    # C1 (probe-verified security regression): thread the composed MCP
    # provider's stamp_scope() through as AgentService's generic
    # review_state_scope seam whenever one was composed for this run --
    # see AgentService.__init__'s own comment and
    # MCPToolProvider.stamp_scope's docstring for the exact adversarial
    # interleave this protects against (a spawned child's own turn(s)
    # clobbering the parent turn's already-decided MCP approval stamps).
    # getattr(..., None) rather than a bare attribute access: mcp_provider
    # is typed Any (a ToolProvider-shaped double in tests may not define
    # stamp_scope at all, and MUST NOT be forced to); production always
    # hands in a real, fully-composed MCPToolProvider here, which always
    # has it.
    # task-628: the seam holds ONE context manager, but two components
    # now own per-turn stamp state a nested sub-agent run would clobber
    # -- the MCP provider's `_stamped_decisions` and the built-in gate's
    # `_stamps`. Compose whichever exist rather than leaving the gate's
    # state unguarded (it was, before this task, and unlike MCP it has
    # no per-call approval fallback to degrade to: a lost stamp fails
    # closed outright). ADR-300 adds the local provider's stamp state
    # to the same composition.
    _scopes = [
        scope
        for scope in (
            getattr(mcp_provider, "stamp_scope", None)
            if mcp_provider is not None
            else None,
            getattr(builtin_gate, "stamp_scope", None)
            if builtin_gate is not None
            else None,
            getattr(local_provider, "stamp_scope", None)
            if local_provider is not None
            else None,
            getattr(virtual_cli_provider, "stamp_scope", None)
            if virtual_cli_provider is not None
            else None,
            getattr(raw_shell_provider, "stamp_scope", None)
            if raw_shell_provider is not None
            else None,
        )
        if scope is not None
    ]
    review_state_scope = native_host._combine_state_scopes(_scopes)

    # TASK-1971 (Agent Change Review): kick the baseline snapshot in the
    # background NOW -- it rides the model's first-token latency -- and
    # gate tool dispatch on its completion by wrapping the review hook,
    # which the runtime invokes before every tool batch executes. A tool
    # writing before B settles would race its own change into the
    # baseline and vanish from the diff. Tracking failures never block
    # the run (spec failure posture): begin_turn cannot raise, and the
    # wrapper's await records timeouts as per-root disclosures.
    change_handle = None
    change_reservation = None
    successor_claim: native_host._SuccessorBoundaryClaim | None = None
    inherited_child_states_at_b: tuple[native_host._ChildChangeState, ...] = ()
    # PR3a-1 Task 6c: measured BEFORE this turn's B. Any sub-agent
    # running now belongs to an EARLIER turn (this one has not spawned
    # anything yet), and a working-tree differ cannot tell its writes
    # from this turn's agent's -- so this turn's record is stamped
    # `turn_concurrent_subagent` and discloses the overlap rather than
    # implying sole authorship.
    concurrent_subagent = self._live_child_count(conversation_id) > 0
    if _workflow_preparation is not None:
        change_roots = ()
    if self._native_change_review_expected and change_roots:
        try:
            native_source = self._change_review_native.capture(
                self,
                session_id=session_id,
                conversation_id=conversation_id,
                origin=work_origin,
                roots=change_roots,
            )
            change_reservation = self._change_finalization_coordinator.register(
                change_roots,
                survivor_key=assistant_message_id,
                native_source=native_source,
                on_unadmitted=self._store.capture_change_review_failure_notice(
                    session_id,
                    assistant_message_id,
                    native_host.format_change_review_skipped_marker(
                        "workspace",
                        "native snapshot unavailable; this turn is not tracked",
                    ),
                ),
            )
            if change_reservation.admission_error:
                native_source.close_unused()
        except Exception:  # noqa: BLE001 - native mismatch cannot enter legacy
            self._append_marker(
                session_id,
                native_host.format_change_review_skipped_marker(
                    "workspace",
                    "native snapshot source unavailable; this turn is not tracked",
                ),
            )
    elif self._change_finalization_coordinator is not None and change_roots:
        try:
            change_reservation = self._change_finalization_coordinator.register(
                change_roots,
                survivor_key=assistant_message_id,
            )
        except Exception:  # noqa: BLE001 -- tracking must never block a run
            native_host.logger.opt(exception=True).warning(
                "change_review: coordinator admission failed; turn untracked"
            )
    elif self._change_tracker is not None and change_roots:
        boundary_failed = False
        while True:
            close_window = None
            close_done = None
            claim_to_release = None
            begin_failed = False
            with self._change_window_lock:
                prior_window = self._post_turn_change_windows.get(conversation_id)
                if prior_window is not None and prior_window.closing:
                    close_window = prior_window
                    close_done = prior_window.close_done
                else:
                    inherited_states = {
                        state.owner_key: state
                        for state in self._child_change_states.get(
                            conversation_id, {}
                        ).values()
                    }
                    successor_claim = None
                    if prior_window is not None:
                        successor_claim = native_host._SuccessorBoundaryClaim()
                        prior_window.successor_claim = successor_claim
                        inherited_states.update(
                            {
                                state.owner_key: state
                                for state in prior_window.child_states
                            }
                        )
                    concurrent_subagent = concurrent_subagent or any(
                        state.live_scopes > 0 or state.pending_scopes > 0
                        for state in inherited_states.values()
                    )
                    inherited_child_states_at_b = tuple(inherited_states.values())
                    begin_paths = sorted(
                        {
                            path
                            for state in inherited_child_states_at_b
                            for path in state.touched_paths
                        }
                    )
                    try:
                        change_handle = self._change_tracker.begin_turn(
                            change_roots,
                            touched_paths=begin_paths,
                        )
                    except Exception:  # noqa: BLE001 -- tracking is best effort
                        change_handle = None
                        begin_failed = True
                    if successor_claim is not None:
                        if successor_claim.failed:
                            change_handle = None
                            successor_claim.handle = None
                        else:
                            successor_claim.handle = change_handle
                            successor_claim.failed = change_handle is None
                        claim_to_release = successor_claim

            if close_done is None:
                if claim_to_release is not None:
                    claim_to_release.ready.set()
                if begin_failed:
                    native_host.logger.warning(
                        "change_review: begin_turn failed; turn untracked"
                    )
                break
            try:
                close_completed = close_done.wait(
                    native_host._CHANGE_BOUNDARY_WAIT_SECONDS
                )
            except Exception:  # noqa: BLE001 -- tracking is best effort
                close_completed = False
            if close_completed:
                with self._change_window_lock:
                    close_completed = close_window.close_succeeded is True
            if not close_completed:
                boundary_failed = True
                break

        if boundary_failed:
            change_handle = None
            successor_claim = None
            native_host.logger.warning(
                "change_review: prior survivor close did not finish "
                "successfully; successor turn untracked"
            )
    # run-hooks (Task 5): resolve the app-owned engine ONCE per turn (its
    # presence answer re-runs while unconfigured -- the runtime's R17
    # rule -- so a first-ever [hooks] entry takes effect on the next
    # turn). `None` (unconfigured, or a bridge built without the runtime
    # accessor -- every test harness) builds the service exactly as
    # before: no PostToolUse dep, byte-identical run behavior.
    run_hooks_engine = (
        self._ensure_run_hooks()
        if _workflow_preparation is None and self._ensure_run_hooks is not None
        else None
    )
    # A restriction-only guard sees every call, including Canvas tools
    # that intentionally skip interactive approval. The permission chain
    # remains independent and cannot override a hook refusal.
    guard_tool_calls = (
        run_hooks_engine.wrap_review(lambda calls, run_id: {}, session_id=session_id)
        if run_hooks_engine is not None
        else None
    )
    baseline_gate = change_reservation or change_handle
    before_tool_dispatch = None
    alias_by_root: dict[str, str] = {}
    if baseline_gate is not None:
        timeout_warned = False
        alias_by_root = {
            str(root): str(alias)
            for root, alias in zip(
                baseline_gate.roots,
                change_root_aliases,
                strict=False,
            )
        }

        def on_baseline_timeout() -> None:
            nonlocal timeout_warned
            if timeout_warned:
                return
            timeout_warned = True
            handle = getattr(baseline_gate, "_handle", baseline_gate)
            timed_out_roots = [
                root
                for root, error in handle.errors.items()
                if "baseline snapshot still running" in error
            ]
            for index, root in enumerate(timed_out_roots, start=1):
                self._append_marker(
                    session_id,
                    native_host.format_change_review_skipped_marker(
                        alias_by_root.get(root, f"workspace {index}"),
                        "baseline timed out; this turn's changes are not tracked",
                    ),
                )

        if (
            change_reservation is not None
            and self._change_finalization_coordinator is not None
        ):
            coordinator_wait = getattr(
                self._change_finalization_coordinator,
                "await_baseline",
                None,
            )
            await_baseline = (
                native_host.functools.partial(coordinator_wait, change_reservation)
                if callable(coordinator_wait)
                else change_reservation.await_baseline
            )
        else:
            await_baseline = baseline_gate.await_baseline
        before_tool_dispatch = native_host.build_change_review_dispatch_gate(
            await_baseline,
            on_timeout=on_baseline_timeout,
        )

    if plugin_entries:
        original_review = review_tool_calls
        original_dispatch = before_tool_dispatch

        def review_tool_calls(calls, kind):
            active_plugin_entries = bind_plugin_actor()
            plugin_service.check_entries_from_agent(active_plugin_entries)
            decisions = original_review(calls, kind) if original_review else {}
            plugin_service.check_entries_from_agent(active_plugin_entries)
            return decisions

        def before_tool_dispatch(calls, pure_runtime_tools):
            active_plugin_entries = bind_plugin_actor()
            plugin_service.check_entries_from_agent(active_plugin_entries)
            if original_dispatch is not None:
                original_dispatch(calls, pure_runtime_tools)
            plugin_service.check_entries_from_agent(active_plugin_entries)

    run_log_writer = None
    if scratch_root is not None and scratch_lease is not None:
        from tldw_chatbook.Agents.run_log import RunLogWriter, resolve_log_root

        from .console_agent_sources import current_agent_sources

        run_log_root: native_host.Path | None = None
        original_sources = current_agent_sources()
        if original_sources is not None:
            original_sources.check_memory()
            if original_sources.bridge is not self:
                raise RuntimeError("original agent bridge changed")
            run_log_root = original_sources.root
        else:
            try:
                with scratch_lease():
                    run_log_root = resolve_log_root(
                        sandbox_root=scratch_root,
                        workspace_id=run_workspace_id,
                    )
            except Exception:  # noqa: BLE001 -- writer will fail closed on lease
                run_log_root = None
        run_log_writer = RunLogWriter(
            root=run_log_root or scratch_root,
            access_scope=scratch_lease,
            on_bound=native_host.functools.partial(
                self._remember_run_log_authority,
                session_id=session_id,
                conversation_id=conversation_id,
                access_scope=scratch_lease,
            ),
        )
        if original_sources is not None:
            original_sources.bind_writer(run_log_writer)

    def on_child_settled(run_id: str | None, status: str) -> None:
        try:
            original = getattr(service, "_ordinary_child_notifications", {}).pop(
                run_id, None
            )
            engine = (
                original[0]
                if original is not None
                else (
                    run_hooks_engine
                    if _workflow_preparation is not None
                    else self._ensure_run_hooks()
                    if self._ensure_run_hooks
                    else None
                )
            )
            if engine is not None and (
                ordinary_tree is None
                or original is not None
                and original[1] is not None
            ):
                engine.notify(
                    "SubagentStop",
                    _prepared=original[1] if original is not None else None,
                    session_id=session_id,
                    run_id=run_id,
                    data={"child_run_id": run_id, "status": status},
                )
        except Exception:  # noqa: BLE001 -- observation cannot prevent settlement
            native_host.logger.warning(
                "SubagentStop observer failed; continuing settlement"
            )
        try:
            if not service.live_subagent_handles():
                with self._change_window_lock:
                    child_change_state.pending_scopes = 0
                    states = self._child_change_states.get(conversation_id)
                    if (
                        states is not None
                        and states.get(child_change_state.owner_key)
                        is child_change_state
                    ):
                        states.pop(child_change_state.owner_key, None)
                        if not states:
                            self._child_change_states.pop(conversation_id, None)
                    has_window = conversation_id in self._post_turn_change_windows
                if has_window:
                    self._close_post_turn_change_window_if_idle(conversation_id)
        finally:
            self._on_fleet_child_settled(
                conversation_id,
                session_id,
                assistant_message_id,
                run_id,
                status,
                workflow_task=(
                    _workflow_preparation.authorization.claim
                    if _workflow_preparation is not None
                    and _workflow_preparation.phase.attempt.purpose == "execution"
                    and _workflow_preparation.authorization is not None
                    and _workflow_preparation.authorization.claim.launch_run_id
                    == run_id
                    else None
                ),
            )

    def _redirect_ready(redirect_fn, abort_probe):
        # TASK-28227: arm the primary stream's abort probe,
        # then hand the Console its redirect hook. The probe
        # reaches ONLY the adapter's stream_cut predicate --
        # LoopDeps.should_cancel never sees it, so a
        # redirect can never kill the run.
        adapter._primary_stream_abort = abort_probe
        if on_redirect_ready is not None:
            on_redirect_ready(redirect_fn)

    # TASK-34100.5.3: a light-surface run creates no fleet (it offers no
    # sub-agent tools), so it never asks for a temporary progress inbox.
    create_fleet = _workflow_preparation is not None or bool(
        not light_surface
        and (
            first_request_plan.schemas.runtime_schemas
            or first_request_plan.schemas.active_schemas
        )
    )
    progress_store = self.message_store
    self._store.prepare_progress_inbox(
        session_id, message_store=progress_store, create_temporary=create_fleet
    )
    with self._store.progress_owner_scope(
        session_id, message_store=progress_store
    ) as progress_owner_id:
        if progress_owner_id != initial_progress_owner_id:
            from tldw_chatbook.Agents.fleet_messages import MessageError

            raise MessageError("unavailable")
        message_inbox = progress_store.get_inbox(progress_owner_id)
        if create_fleet and message_inbox is None:
            from tldw_chatbook.Agents.fleet_messages import MessageError

            raise MessageError("unavailable")
        fleet_coordinator = self._conversation_fleet_coordinator(
            conversation_id,
            create=create_fleet,
            progress_owner_id=progress_owner_id,
            message_inbox=message_inbox,
        )
    if light_surface:
        # No fleet and no progress inbox: either would add fleet or
        # read_agent_messages tools the light surface does not offer.
        fleet_coordinator = message_inbox = None
    plugin_root_id = ""
    plugin_producer_pin = None

    def bind_plugin_root(actual_run_id: str) -> None:
        nonlocal plugin_root_id, plugin_producer_pin
        plugin_root_id = actual_run_id
        if plugin_entries:
            if plugin_cancel_root is None:
                raise PermissionError("Plugin run cancellation owner is unavailable.")

            def cancel_root():
                plugin_cancel_root()
                if revoke_approvals is not None:
                    revoke_approvals(actual_run_id)

            plugin_service.bind_run(plugin_entries, actual_run_id, cancel_root)
            if plugin_producer_pin is None:
                # Capture capabilities at real binding. The store finalizer
                # binds its durable conversation ID only after persisting it.
                conversation_id = (
                    self._store._session_or_raise(session_id).persisted_conversation_id
                    or session_id
                )
                plugin_producer_pin = plugin_service.capture_resume_pin(
                    plugin_entries,
                    actual_run_id,
                    conversation_id,
                    assistant_message_id,
                )

    def bind_plugin_actor():
        from tldw_chatbook.Agents.run_context import current_run_actor

        actor = current_run_actor()
        if actor is None:
            raise PermissionError("Plugin run identity is unavailable.")
        actual_run_id = actor.run_id
        if actor.kind == "primary":
            bind_plugin_root(actual_run_id)
            return plugin_entries
        handle = next(
            (item for item in service.fleet_snapshot() if item.run_id == actual_run_id),
            None,
        )
        if handle is None:
            if actor.parent_run_id != plugin_root_id or plugin_cancel_root is None:
                raise PermissionError("Plugin child ownership is unavailable.")

            def cancel_inline_child():
                plugin_cancel_root()
                if revoke_approvals is not None:
                    revoke_approvals(actual_run_id)

            plugin_service.bind_run(
                plugin_entries,
                actual_run_id,
                cancel_inline_child,
                parent_run_id=actor.parent_run_id,
            )
            return plugin_entries
        ceiling = handle.managed_resume_ceiling
        if handle.managed_resume_handle_id is not None:
            ceiling = plugin_service.fleet_resume_ceiling(
                handle.managed_resume_pin,
                handle.managed_resume_run_id,
                handle.managed_resume_handle_id,
                plugin_entries,
            )
        child_entries = plugin_service.constrain_entries(plugin_entries, ceiling)
        plugin_service.bind_run(
            child_entries,
            actual_run_id,
            lambda: service.cancel_subagent(handle.handle_id),
            handle.handle_id,
            parent_run_id=actor.parent_run_id,
            component_ceiling=ceiling,
        )
        if handle.managed_pin is None:
            pin = plugin_service.capture_resume_pin(
                child_entries,
                actual_run_id,
                session_id,
                handle.handle_id,
            )
            service._fleet.set_managed_custody(
                handle.handle_id,
                required=bool(plugin_entries),
                pin=pin,
                resume_ceiling=ceiling,
                resume_pin=handle.managed_resume_pin,
                resume_run_id=handle.managed_resume_run_id,
                resume_handle_id=handle.managed_resume_handle_id,
            )
        return child_entries

    def plugin_run_terminal(actual_run_id: str) -> None:
        if plugin_entries:
            plugin_service.complete_run(actual_run_id)
        if on_run_terminal is not None:
            on_run_terminal(actual_run_id)

    def plugin_checked_chat_call(**kwargs):
        active_plugin_entries = ()
        if plugin_entries:
            active_plugin_entries = bind_plugin_actor()
            plugin_service.check_entries_from_agent(active_plugin_entries)
            from tldw_chatbook.Plugins.context import check_send_context

            kwargs["messages_payload"] = check_send_context(kwargs["messages_payload"])
        result = adapter.chat_call(**kwargs)
        if plugin_entries:
            plugin_service.check_entries_from_agent(active_plugin_entries)
        return result

    def finalize_managed_checkpoint(checkpoint, conversation_id, message_id, run_id):
        if plugin_producer_pin is None or run_id != plugin_root_id:
            raise PermissionError("Managed continuation producer is unavailable.")
        pin = native_host.json.loads(plugin_producer_pin)
        session = self._store._session_or_raise(session_id)
        if (
            session.persisted_conversation_id != conversation_id
            or message_id != assistant_message_id
        ):
            raise PermissionError("Managed continuation durable owner changed.")
        if pin["conversation_id"] not in {session_id, conversation_id}:
            raise PermissionError("Managed continuation original owner changed.")
        pin["conversation_id"] = conversation_id
        return plugin_service.seal_resume_checkpoint(
            native_host.json.dumps(pin),
            checkpoint,
            conversation_id,
            message_id,
        )

    def persist_continuation(event):
        if (
            plugin_entries
            and event.context.durability == "persistent"
            and event.context.agent_kind == "primary"
        ):
            bind_plugin_root(event.context.run_id)
        self._store.persist_provider_continuation_event(
            event,
            continuation_target=continuation_target,
            checkpoint_finalizer=(
                finalize_managed_checkpoint if plugin_entries else None
            ),
        )
        if capture_mode is native_host.ConsoleTraceCaptureMode.CAPTURE_ON:
            adapter._live_native_persisted_event = event

    hooks_v2_engine = self._get_hooks_v2(session_id) if self._get_hooks_v2 else None
    service = native_host.AgentService(
        self._db,
        registry,
        chat_call=plugin_checked_chat_call if plugin_entries else adapter.chat_call,
        runtime_capacity=self.runtime_capacity,
        work_origin=work_origin,
        work_chain_id=work_chain_id,
        worktree_repo_authority=worktree_repo_authority,
        clock=self._clock,
        app_config=self._app_config,
        on_step=on_step,
        on_resolved_target=on_resolved_target,
        on_tool_activity=on_tool_activity,
        # TASK-25903: hands the controller a steer(text) bound to THIS
        # run once its mailbox registers -- run ids are minted inside
        # run_turn, so the caller cannot key by one.
        on_primary_steer_ready=on_steer_ready,
        on_primary_redirect_ready=_redirect_ready,
        skill_runner=skill_runner,
        skill_file_bindings=skill_file_bindings,
        review_tool_calls=review_tool_calls,
        guard_tool_calls=guard_tool_calls,
        hooks_v2_engine=hooks_v2_engine,
        hooks_v2_session_id=session_id,
        hooks_v2_turn_id=accepted_turn_id or assistant_message_id,
        on_primary_budget_settled=on_primary_budget_settled,
        # Standalone work has no plugin dependency graph.
        hooks_v2_required_handler_ids=lambda _run_id: (),
        hooks_v2_render_context=(
            hooks_v2_engine.native_plugins.render_context
            if hooks_v2_engine is not None
            and getattr(hooks_v2_engine, "native_plugins", None) is not None
            else None
        ),
        hooks_v2_definition_requirements=(
            (
                lambda definition, messages: (
                    hooks_v2_engine.native_plugins.requirements_for(
                        definition, registry, messages
                    )
                )
            )
            if hooks_v2_engine is not None
            and getattr(hooks_v2_engine, "native_plugins", None) is not None
            else None
        ),
        before_tool_dispatch=before_tool_dispatch,
        review_state_scope=review_state_scope,
        install_skill_tool=install_skill_tool,
        prepare_managed_skill_promotion_tool=(prepare_managed_skill_promotion_tool),
        run_skill_script_tool=run_skill_script_tool,
        archive_recovery_provider=recovery_provider,
        fork_chat_tool=fork_chat_tool,
        new_chat_tool=new_chat_tool,
        run_log_writer=run_log_writer,
        post_tool_call=(
            run_hooks_engine.post_tool_dep(session_id=session_id)
            if run_hooks_engine is not None
            else None
        ),
        run_log_request_plan=first_request_plan.run_log,
        revoke_approvals=revoke_approvals,
        on_tool_terminal=on_tool_terminal,
        on_tool_result_terminal=on_tool_result_terminal,
        on_run_terminal=lambda run_id: self._on_live_run_terminal(
            run_id, plugin_run_terminal if plugin_entries else on_run_terminal
        ),
        run_model_scope=native_host.functools.partial(
            self._live_usage_run_scope,
            adapter,
            conversation_id,
            primary_live_key,
            inline_subagents=subagents if fleet_coordinator is None else None,
        ),
        managed_child_required=bool(plugin_entries),
        managed_child_resume=(
            (
                lambda retained: plugin_service.fleet_resume_ceiling(
                    retained.managed_pin,
                    retained.run_id,
                    retained.handle_id,
                    plugin_entries,
                )
            )
            if plugin_service is not None
            else None
        ),
        persist_provider_continuation=persist_continuation,
        acknowledge_provider_continuation=(
            adapter._acknowledge_live_native_continuation
            if capture_mode is native_host.ConsoleTraceCaptureMode.CAPTURE_ON
            else None
        ),
        expand_provider_continuation=expand_provider_continuation,
        prepare_provider_continuation_request=bool(
            continuation_target is not None
            and callable(getattr(self._gateway, "prepare_chat_request", None))
        ),
        # PR3a-1 Task 1: every fleet child gets its own model-call
        # lifeline, entered on the child's own thread and torn down
        # when the CHILD finishes -- never when this turn does.
        # PR3a-1 Task 6c wraps that lifeline (it does not replace it)
        # so the same enter/exit also counts live children: "the last
        # child of this conversation just finished" is the signal that
        # closes a survivor's change-review window, and nothing else
        # in the bridge knows it (the coordinator marks a handle
        # terminal only AFTER this scope exits).
        inline_child_model_scope=adapter.child_lifeline,
        child_model_scope=native_host.functools.partial(
            self._child_run_scope,
            conversation_id,
            adapter,
            child_change_state,
        ),
        # PR3a-2 Task 2: the settle half of the same seam. The wrapper
        # above binds this turn's IDENTITY -- session + originating
        # assistant message -- and removes Task 3's live WRITE state
        # only after this service's final handle settles, then forwards
        # to the existing fan-out path. The run id arrives per call from
        # `run_child`'s finally, where it is first knowable.
        on_child_settled=on_child_settled,
        # PR3a-1 Task 6a: this CONVERSATION's coordinator, not this
        # turn's -- the only thing that makes `[agents]
        # max_live_subagents` a bound on the fleet rather than on one
        # message, and the only thing an earlier turn's survivor is
        # still visible and stoppable through. `None` when the fleet
        # kill switch is on, which leaves `AgentService` to take its
        # own inline path exactly as before.
        fleet_coordinator=fleet_coordinator,
        message_inbox=message_inbox,
        startup_instruction_candidate=startup_instruction_candidate,
        confirm_project_instruction_dispatch=(
            service_confirm_project_instruction_dispatch
        ),
        project_instruction_context=project_instruction_context,
        on_ephemeral_runtime_warning=(
            (
                lambda code, _names, _count: on_project_instruction_activation(
                    native_host.ProjectInstructionActivationEvent(
                        outcome_codes=(code,),
                    )
                )
            )
            if on_project_instruction_activation is not None
            else None
        ),
        propagate_trace_call_persistence_errors=(
            propagate_trace_call_persistence_errors
        ),
    )
    # PR2b Task 1: publish BEFORE `run_turn` is called (below) -- see
    # `self._fleet_services`'s own docstring in `__init__` for the
    # lifetime/thread-safety contract this relies on.
    if _workflow_preparation is None:
        self._fleet_services[conversation_id] = service
        # PR3a-2 Task 4: open the survivor-discriminator window for THIS
        # turn -- a child settling while this id is present finished
        # within its turn (see `_inflight_turn_message_ids`); the matching
        # discard is the FIRST statement of the `finally` below.
        with self._change_window_lock:
            self._inflight_turn_message_ids.add(assistant_message_id)

    from .console_ordinary_tree import current_ordinary_tree

    ordinary_tree = current_ordinary_tree()
    service._ordinary_tree = ordinary_tree
    if ordinary_tree is not None:
        service._ordinary_child_notifications = {}

        def prepare_child_notification(child_id):
            service._ordinary_child_notifications[child_id] = (
                run_hooks_engine,
                run_hooks_engine.prepare_notification("SubagentStop", session_id)
                if run_hooks_engine is not None
                else None,
            )

        service._prepare_child_notification = prepare_child_notification
    supersede_run_id = (
        self._previous_primary_run_id(conversation_id) if supersede_previous else None
    )
    run_messages = list(first_request_plan.messages)
    execution_owner = None
    run_id = outcome = None
    primary_started = False

    def run_primary():
        nonlocal execution_owner, run_id, outcome, config, primary_started
        if (_workflow_preparation is not None and planning is None) or primary_started:
            raise PermissionError("primary dispatch unavailable")
        primary_started = True
        execution_owner = (
            planning._admit_planning_primary(service)
            if planning is not None
            else ordinary_tree.borrow(ordinary_tree.runtime, opaque=False)
            if ordinary_tree is not None
            else self.runtime_capacity.begin_execution(
                origin=work_origin, conversation_id=conversation_id
            )
        )
        # FIRST statement in the block that owns this thread's
        # shutdown -- see its construction above. Not merely *before*
        # the try: one inserted line there would silently re-open the
        # leak. If start() itself raises (thread exhaustion), the
        # finally still runs and still closes the loop; `is_alive()`
        # is False for a never-started thread, so the close branch is
        # the one taken and no fd leaks.
        turn_lifeline.start(owner=execution_owner)
        if parent_budget_deadline is not None:
            wall = parent_budget_deadline - native_host.time.monotonic()
            if wall <= 0:
                raise RuntimeError("native_parent_budget_exhausted")
            config = native_host.dataclass_replace(
                config,
                budget=native_host.dataclass_replace(
                    config.budget,
                    max_wall_seconds=min(config.budget.max_wall_seconds, wall),
                ),
            )
        from .console_continuation_binding import continuation_can_be_durable

        run_id, outcome = service.run_turn(
            execution_owner=execution_owner,
            conversation_id=conversation_id,
            messages=run_messages,
            config=config,
            # Intentionally NOT forwarding the native in-memory id here.
            # create_run would store it, but the native id can never match
            # any persisted_message_id -- so a run left unfinished (stopped
            # mid-run, cancelled, failed, or crashed) would hold a stale,
            # non-null id that resume anchoring can never match (and Task 3
            # would drop as "off-path", silently hiding its markers). By
            # omitting it the run row starts NULL; the controller writes the
            # durable PERSISTED id onto the run on EVERY terminal path once
            # the reply is persisted (see record_run_assistant_message), and
            # any still-unfinished run stays NULL for resume's null->ordinal
            # fallback. run_turn's assistant_message_id threading stays a
            # generic service capability (its own tests call it directly).
            # execution_key-first (Task 5): the service's capability
            # check keys off api_endpoint, and execution_key is by
            # definition "Provider key passed to chat_api_call" — the
            # PROVIDER_PARAM_MAP key space provider_supports_native_tools
            # matches against. `provider` (the display key) and then
            # "agent" remain fallbacks for fakes lacking either
            # attribute (e.g. resolution=object() in existing tests),
            # which keeps them on the fence path unchanged.
            api_endpoint=first_request_plan.api_endpoint,
            # Raw selection identity for spawn resolution: an inheriting
            # child of custom-ep:<slug> must snapshot the endpoint (and
            # its registry base URL), not the flattened execution family.
            parent_raw_provider=first_request_plan.selected_provider,
            should_cancel=should_cancel,
            supersede_run_id=supersede_run_id,
            continuation_owner_message_id=assistant_message_id,
            continuation_durability=(
                "persistent"
                if continuation_can_be_durable(
                    self._store, session_id, assistant_message_id, resolution
                )
                else "ephemeral"
            ),
            restore_provider_continuation=restore_provider_continuation,
            restore_provider_target=restore_provider_target,
            resume_provider_continuation=resume_provider_continuation,
            continuation_sidecar=continuation_sidecar,
            continuation_target=continuation_target,
            continuation_owner_key=continuation_owner_key,
            first_request_schema_plan=first_request_plan.schemas,
            request_worktree_merge_confirm=request_worktree_merge_confirm,
            requested_run_id=canvas_run_id,
        )
        return run_id, outcome

    prepared = native_host._PreparedNativeReply(
        service=service,
        first_request_plan=first_request_plan,
        session_id=session_id,
        progress_owner_id=initial_progress_owner_id,
        resolution=resolution,
        adapter=adapter,
        turn_lifeline=turn_lifeline,
        project_instruction_context=project_instruction_context,
        review_state_scope=review_state_scope,
        run_log_writer=run_log_writer,
        child_change_state=child_change_state,
        run_primary=run_primary,
    )
    if _workflow_preparation is not None:

        def no_launch_metadata_only():
            from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

            if (
                canvas_turn_controller is not None
                or managed_skill_promotion_gate is not None
                or raw_shell_progress_run_ids
                or primary_buddy_run_ids
                or tool_activity._rows
                or service.live_subagent_handles()
                or local_provider is not None
                and type(local_provider) is not LocalToolProvider
            ):
                raise PermissionError("workflow no-launch cleanup is not metadata only")
            _workflow_preparation.check_local_sources()

        _workflow_preparation._no_launch_close_guard = no_launch_metadata_only
    try:
        yield prepared
    finally:
        # PR3a-2 Task 4: this turn is over -- from here on a settling
        # child of it is a background (after-turn) completion. First
        # statement of the finally so the window closes even when
        # `run_turn` raised, before any teardown below can block.
        with self._change_window_lock:
            self._inflight_turn_message_ids.discard(assistant_message_id)
        if canvas_turn_controller is not None:
            try:
                canvas_turn_controller.finish_assistant_run(
                    assistant_message_id,
                    actual_run_id=(run_id if run_id is not None else ""),
                    terminal_status=(
                        outcome.status if outcome is not None else "error"
                    ),
                )
            except Exception:
                native_host.logger.warning("canvas run settlement failed closed")
        if managed_skill_promotion_gate is not None:
            managed_skill_promotion_gate.unbind_reader()
        unbind_promotion_context = getattr(
            local_provider, "unbind_instruction_promotion_context", None
        )
        if callable(unbind_promotion_context):
            unbind_promotion_context()
        run_cancelled = (
            outcome is not None and outcome.status == native_host.RUN_CANCELLED
        )
        self._clear_raw_shell_progress(
            raw_shell_progress_run_ids, cancelled=run_cancelled
        )
        try:
            tool_activity.finish(run_cancelled)
        except Exception as exc:  # noqa: BLE001 — display cannot prevent run teardown
            native_host.logger.warning(
                "Console tool activity display could not be settled ({})",
                type(exc).__name__,
            )
        for activity_run_id in tool_activity_run_ids:
            self._tool_activity_runs.pop(activity_run_id, None)
        if self._buddy_sink is not None:
            for buddy_run_id in primary_buddy_run_ids:
                self._buddy_sink.release_run(buddy_run_id)
        # PR2b Task 1: clear the published service in the SAME
        # teardown path that already tears this run down -- not a
        # second one. From this point `fleet_snapshot` reverts to `[]`
        # for this conversation (unless a newer run has already
        # published over this one -- see `_teardown_fleet_service`),
        # even on a raised/cancelled run (this `finally` always runs).
        # Pinned choice: a completed run's snapshot goes back to `[]`,
        # not the run's terminal handles (see
        # `test_fleet_snapshot_reflects_two_live_handles_in_flight_
        # then_empty_after_run_completes`).
        self._teardown_fleet_service(conversation_id, service)
        # PR2a Task 6.5: stop the driver thread before closing, and
        # join it (the stop/join/close ordering, and why a wedged
        # thread keeps its loop open, now live in
        # `_ModelCallLifeline.shutdown`).
        #
        # PR3a-1 Task 1: this tears down the PRIMARY agent's loop and
        # nothing else. It used to be the whole run tree's -- and the
        # comment here used to argue that was safe because "run_turn
        # has already settled every fleet child ... so nothing should
        # still be submitting", an invariant PR 3a deliberately
        # breaks. A fleet child submits to a lifeline it owns itself
        # (see `_StreamingModelAdapter.child_lifeline`), so a child
        # still running when this line executes keeps a live transport
        # to the model rather than losing one out from under it.
        try:
            if turn_lifeline is not None:
                turn_lifeline.shutdown()
        finally:
            if (
                execution_owner is not None
                and ordinary_tree is None
                and planning is None
            ):
                execution_owner.finish_root()
        # TASK-1971: E snapshot on EVERY terminal path -- completed,
        # failed, cancelled, or crashed. A run that died halfway through
        # editing is when review matters most. `run_id` is unbound when
        # run_turn itself raised before creating the run row; the
        # records are then logged instead of stored (nothing to attach
        # them to), and the exception still propagates unchanged.
        if change_reservation is not None:
            try:
                if run_id is not None:
                    _steps = outcome.steps if outcome is not None else []
                    _primary_paths = native_host.ChangeTurnTracker.tool_touched_paths(
                        _steps
                    )
                    _touched_paths = list(
                        dict.fromkeys(
                            (
                                *_primary_paths,
                                *sorted(child_change_state.touched_paths),
                            )
                        )
                    )
                    finalization = self._change_finalization_coordinator.finalize(
                        change_reservation,
                        run_id=run_id,
                        touched_paths=_touched_paths,
                        kind=(
                            native_host.CHANGE_KIND_TURN_CONCURRENT_SUBAGENT
                            if concurrent_subagent
                            else native_host.CHANGE_KIND_TURN
                        ),
                        has_live_survivors=(
                            self._live_child_count_for_turn(assistant_message_id) > 0
                        ),
                    )
                    if (
                        finalization
                        is native_host.ChangeReviewFinalizeResult.OVERLOAD_VISIBLE
                    ):
                        if self._native_change_review_expected:
                            change_reservation.notify_unadmitted()
                        else:
                            error = (
                                change_reservation.admission_error
                                or "change-review finalization queue unavailable"
                            )
                            for index, root in enumerate(
                                change_reservation.roots,
                                start=1,
                            ):
                                self._append_marker(
                                    session_id,
                                    native_host.format_change_tracking_failure_marker(
                                        alias_by_root.get(
                                            str(root), f"workspace {index}"
                                        ),
                                        error,
                                    ),
                                )
                else:
                    self._change_finalization_coordinator.cancel(change_reservation)
            except Exception:  # noqa: BLE001 -- never mask the run outcome
                native_host.logger.opt(exception=True).warning(
                    "change_review: finalization scheduling failed; "
                    "turn changes untracked"
                )
        elif change_handle is not None:
            # PR3a-1 Task 6c: close the EARLIER turn's survivor
            # window before this turn's E. A timed-out close waiter
            # cannot prove that close-time handoff finished, so this
            # turn must fail closed instead of overtaking it.
            boundary_safe = self._close_post_turn_change_window(conversation_id)
            with self._change_window_lock:
                claim_failed = successor_claim is not None and successor_claim.failed
            if not boundary_safe or claim_failed:
                change_handle = None
        if change_handle is not None:
            try:
                _steps = outcome.steps if outcome is not None else []
                _current_live_handles = service.live_subagent_handles()
                _current_state_pending = bool(_current_live_handles)
                with self._change_window_lock:
                    if _current_state_pending:
                        _live_states = self._child_change_states.setdefault(
                            conversation_id, {}
                        )
                        _live_states[child_change_state.owner_key] = child_change_state
                        child_change_state.pending_scopes = max(
                            0,
                            len(_current_live_handles) - child_change_state.live_scopes,
                        )
                    _pending_child_states_before_e = tuple(
                        self._child_change_states.get(conversation_id, {}).values()
                    )
                    _e_child_states = {
                        state.owner_key: state for state in inherited_child_states_at_b
                    }
                    _e_child_states.update(
                        {
                            state.owner_key: state
                            for state in _pending_child_states_before_e
                        }
                    )
                    if child_change_state.touched_paths:
                        _e_child_states[child_change_state.owner_key] = (
                            child_change_state
                        )
                    _child_paths_before_e = sorted(
                        {
                            path
                            for state in _e_child_states.values()
                            for path in state.touched_paths
                        }
                    )
                # A handle can settle between the parent-visible query
                # and registration. Keep the captured reference for E,
                # but do not strand a dead state in the live map after
                # its one settle callback already ran.
                if _current_state_pending and not service.live_subagent_handles():
                    with self._change_window_lock:
                        states = self._child_change_states.get(conversation_id)
                        if (
                            states is not None
                            and states.get(child_change_state.owner_key)
                            is child_change_state
                            and child_change_state.live_scopes == 0
                        ):
                            child_change_state.pending_scopes = 0
                            states.pop(child_change_state.owner_key, None)
                            if not states:
                                self._child_change_states.pop(conversation_id, None)
                _primary_paths = native_host.ChangeTurnTracker.tool_touched_paths(
                    _steps
                )
                _touched_paths = list(
                    dict.fromkeys((*_primary_paths, *_child_paths_before_e))
                )
                _records = self._change_tracker.end_turn(
                    change_handle,
                    touched_paths=_touched_paths,
                )
                if run_id is not None:
                    self._record_change_snapshots(
                        run_id=run_id,
                        records=_records,
                        kind=(
                            native_host.CHANGE_KIND_TURN_CONCURRENT_SUBAGENT
                            if concurrent_subagent
                            else native_host.CHANGE_KIND_TURN
                        ),
                    )
                    self._append_change_markers(
                        session_id,
                        run_id,
                        _records,
                        kind=(
                            native_host.CHANGE_KIND_TURN_CONCURRENT_SUBAGENT
                            if concurrent_subagent
                            else native_host.CHANGE_KIND_TURN
                        ),
                    )
                    if _pending_child_states_before_e:
                        self._open_post_turn_change_window(
                            conversation_id,
                            run_id=run_id,
                            session_id=session_id,
                            handle=change_handle,
                            child_states=_pending_child_states_before_e,
                        )
                elif _records:
                    native_host.logger.warning(
                        "change_review: run crashed before a run row "
                        f"existed; {len(_records)} change record(s) "
                        "not stored"
                    )
            except Exception:  # noqa: BLE001 -- never mask the run's outcome
                native_host.logger.opt(exception=True).warning(
                    "change_review: end_turn failed; turn changes untracked"
                )
        if run_id is not None and change_review_skipped_roots:
            self._append_skipped_change_review_markers(
                session_id,
                change_review_skipped_roots,
            )
        # task-5 (turn-file-annotate, spec §4): stamp exactly the
        # notes this run's attach seam included (captured above,
        # before `run_turn` was even called) and disclose what was
        # sent. Placed BESIDE the marker seam above -- not inside
        # `_append_change_markers`, and deliberately NOT nested under
        # `if change_handle is not None:`: diff-feedback notes are
        # about what the user told the agent, unrelated to whether
        # THIS turn's own tracked roots changed, or whether change
        # tracking is even configured for this run at all -- the
        # marker seam's internal `if files:` gate ("nothing to
        # report") must not become a reason to strand notes that were
        # genuinely delivered in the outbound payload. Gated only on
        # the run having actually produced assistant output: a run
        # that errors, gets cancelled empty-handed, or crashes before
        # a run row exists leaves every attached note pending for the
        # retry -- the block only ever lived in the outbound COPY (see
        # the attach seam above run_turn), so nothing is lost by not
        # stamping. `run_id`/`outcome` are bound together by the same
        # assignment, so the one `locals()` check covers both. Double
        # delivery is impossible by construction (the pending query
        # excludes already-stamped rows). Never breaks the reply.
        if run_id is not None and outcome.final_text and diff_feedback_included_ids:
            try:
                # task-6 fix round: stamp with the id of THIS
                # (completing) run -- the run whose completion is
                # actually doing the delivering, which resume
                # re-derivation anchors the disclosure row to. `run_id`
                # is already known-bound by the `locals()` check above.
                #
                # Qodo #4 (PR #1779 fix round): disclose only the ids
                # `mark_notes_delivered` reports ACTUALLY stamped, not
                # every id this seam captured -- a concurrent delivery
                # elsewhere could have already stamped one of them
                # first (the UPDATE's own `delivered_at IS NULL` guard
                # silently skips it), and disclosing a note nobody
                # verifiably delivered on THIS run's completion would
                # be dishonest. Concurrent replies on one conversation
                # are architecturally serialized today, so this is
                # defense in depth, not a reachable-today bug.
                stamped_ids = set(
                    self._db.mark_notes_delivered(
                        diff_feedback_included_ids, delivered_by_run_id=run_id
                    )
                )
                disclosed_notes = [
                    note
                    for note in diff_feedback_included_notes
                    if int(note["id"]) in stamped_ids
                ]
                if disclosed_notes:
                    self._store.append_message(
                        session_id,
                        role=native_host.ConsoleMessageRole.TOOL,
                        content=native_host.format_diff_feedback_disclosure(
                            disclosed_notes
                        ),
                        activity_presentation=native_host.ConsoleActivityPresentation(
                            "feedback", "Feedback delivered", "done"
                        ),
                    )
            except Exception:  # noqa: BLE001 -- notes must never break the reply
                native_host.logger.opt(exception=True).warning(
                    "change_review: could not stamp/disclose diff feedback"
                )
    if _workflow_preparation is not None and (planning is None or outcome is None):
        return
    capture_outcome = (
        "complete"
        if outcome.status == native_host.RUN_DONE
        else "stopped"
        if outcome.status == native_host.RUN_CANCELLED
        else "failed"
    )
    capture_update = thinking_capture.settle(capture_outcome)
    if capture_update.envelope is not None:
        self._store.settle_message_thinking(
            assistant_message_id,
            capture_update.envelope,
            generation_token=generation_token,
        )
    from tldw_chatbook.Agents.tool_disclosure_diagnostics import log_agent_steps

    log_agent_steps(outcome.steps, run_id)
    native_host.logger.info(
        "console agent bridge run_reply end",
        conversation_id=conversation_id,
        session_id=session_id,
        outcome_status=outcome.status,
        final_text_len=len(outcome.final_text),
        step_count=len(outcome.steps),
    )
    # PR3a-1 Task 6b (audit F1): under THIS turn's own key, so the
    # terminal status it writes is final -- a surviving child's later
    # steps land in the child's own slot and can no longer reset this
    # one to "running" for the rest of the process's life.
    self._publish_live(
        conversation_id,
        primary_live_key,
        native_host.AgentLiveSnapshot(
            status=outcome.status,
            step=live_steps.count,
            steps=live_steps.tail(5),
            # PR2b Task 2: `service` here -- NOT `self.fleet_snapshot(
            # conversation_id)` -- deliberately: the `finally` block
            # just above already ran `self._teardown_fleet_service`,
            # which pops THIS run's own `self._fleet_services` entry
            # (assuming no overlapping resend already overwrote it --
            # see that method's docstring), so a lookup by
            # conversation_id here would see `[]` and wipe out every
            # child's final status right when the run ends. The
            # `service` local variable is unaffected by that pop -- by
            # this point `_settle_fleet` (called from inside
            # `run_turn`, before it returned above) has already
            # joined/abandoned every fleet child, so every handle
            # `service.fleet_snapshot()` returns here is already
            # terminal.
            subagents=native_host._subagent_summaries_from_fleet(
                service.fleet_snapshot(), subagents
            ),
        ),
        primary=True,
    )
    # The run just finished -- drop any stale historical cache entry so
    # a *later* resume (in a future process) always re-derives fresh
    # rather than reading this run's now-superseded snapshot (belt and
    # braces on top of the pop at run start above).
    self._historical_cache.pop(conversation_id, None)
    return run_id, outcome
