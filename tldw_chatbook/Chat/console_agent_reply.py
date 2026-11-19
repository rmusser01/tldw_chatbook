"""Agent reply orchestration for the existing Console controller.

The caller retains identity, state, and public entry points. Original module
bindings are read live through host, including after awaits and in callbacks.
"""

from __future__ import annotations

from . import console_chat_controller as host


async def _run_agent_reply(
    self,
    *,
    request_project_instruction_binding: host.Callable[
        [str, tuple[host.Any, ...], host.ProjectInstructionBindingRecovery],
        host.Awaitable[tuple[bool, host.Any]],
    ],
    resolution: host.Any,
    provider_messages: list[dict[str, host.Any]],
    assistant_message_id: str,
    prepare_retry: bool,
    variant_mode: bool,
    skill_bindings: tuple[str, ...] = (),
    skill_bundle_block: str = "",
    citation_repair_session: host.ConsoleCitationRepairSession | None = None,
    stream_signals: host.ConsoleProviderStreamSignals | None = None,
    turn_context: host.ConsoleTurnExecutionContext | None = None,
    restore_provider_continuation: host.ProviderContinuationCheckpoint | None = None,
    restore_provider_target: host.ContinuationRestoreTarget | None = None,
    expand_provider_continuation: (
        host.Callable[[host.ProviderContinuationCheckpoint], list[dict]] | None
    ) = None,
    resume_provider_continuation: bool = False,
    restore_provider_message_version: int | None = None,
    continuation_sidecar: tuple[host.ProviderContinuationSidecar, ...] = (),
    continuation_history_target: host.ContinuationRestoreTarget | None = None,
    thinking_sidecar: tuple[host.ProviderThinkingSidecar, ...] = (),
    thinking_policy: host.ThinkingHistoryPolicy = "auto",
    preparation_id: str | None = None,
    generation_token: int | None = None,
    before_provider_dispatch: host.Callable[[], host.Awaitable[None]] | None = None,
    capture_mode_override: host.ConsoleTraceCaptureMode | None = None,
    trace_request: host.PreparedConsoleRequest | None = None,
    propagate_trace_call_persistence_errors: bool = False,
    _generation_handoff: host._GenerationTokenHandoff | None = None,
    _remote_session_key: str | None = None,
    trusted_profile_user_message_id: str | None = None,
    work_origin: host.WorkOrigin,
    work_chain_id: str | None = None,
    wake_authorization: host.AgentWakeAuthorization | None = None,
    chat_start_authorization: host.AgentChatStartAuthorization | None = None,
) -> host.ConsoleSubmitResult:
    """Run the agent loop as the reply engine, streaming into the target row."""
    host.logger.info(
        "console agent reply start",
        assistant_message_id=assistant_message_id,
        variant_mode=variant_mode,
        prepare_retry=prepare_retry,
    )
    if _generation_handoff is None:
        raise RuntimeError("Agent generation handoff is required.")
    # Resolve the run's OWNING session FIRST (Task 3b): every write
    # below -- the per-session stream/cancel maps AND run state -- must
    # target it explicitly rather than whatever the user currently has
    # open (parallel-agents spec §2). Moved ahead of those writes
    # (previously ran after them, back when they were single shared
    # slots with no session to key by).
    try:
        session_id = self.store.session_id_for_message(assistant_message_id)
    except KeyError:
        return self._session_closed_result()
    # Capture before preparatory awaits or executor scheduling. A restored
    # native session can reuse its ID without setting this run's cancel event.
    progress_owner_id = self.store.progress_owner_id(session_id)
    if progress_owner_id is None:
        return self._session_closed_result(session_id=session_id)
    turn_context = self._require_complete_turn_execution_context(turn_context)
    if turn_context.session_id != session_id:
        raise ValueError("Console turn context does not own the assistant row.")
    frozen_workspace_id = (
        turn_context.project_authority.workspace_id
        if turn_context.project_authority is not None
        else turn_context.provider_selection.workspace_context.active_workspace_id
    )
    frozen_workspace_ephemeral = bool(
        turn_context.tool_configuration.get("session_ephemeral", False)
    )
    frozen_workspace_bindings = (
        turn_context.project_authority.options
        if turn_context.project_authority is not None
        else ()
    )
    frozen_workspace_read_binding_ids = tuple(
        binding.binding_id for binding in frozen_workspace_bindings
    )
    frozen_workspace_write_binding_ids = tuple(
        binding.binding_id
        for binding in frozen_workspace_bindings
        if binding.allow_write
    )
    frozen_workspace_binding_authority = tuple(frozen_workspace_bindings)
    scratch_snapshot = turn_context.scratch_space
    if scratch_snapshot is None:
        return self._block(session_id, "Private scratch space is unavailable.")
    thinking_block = self._thinking_persistence_preflight(
        session_id=session_id,
        resolution=resolution,
    )
    if thinking_block is not None:
        return thinking_block
    scratch_lease = host.functools.partial(
        self._scratch_spaces.lease,
        scratch_snapshot,
    )
    session = next((s for s in self.store.sessions() if s.id == session_id), None)
    project_snapshot = turn_context.project_authority
    project_state = (
        host.ProjectInstructionControlState(
            project_instructions_enabled=project_snapshot.enabled,
            working_folder_binding_id=(project_snapshot.working_folder_binding_id),
            working_folder_locator_fingerprint=(
                project_snapshot.working_folder_locator_fingerprint
            ),
            project_instruction_notice_key=(
                project_snapshot.project_instruction_notice_key
            ),
        )
        if project_snapshot is not None
        else session.project_instruction_state
        if session is not None
        else host.ProjectInstructionControlState.legacy_disabled()
    )
    startup_candidate: host.StartupInstructionCandidate | None = None
    project_selection: host.ProjectInstructionBindingSelection | None = None
    confirm_project_dispatch = None
    project_authority_guard = None
    project_activation_callback = None
    if session is not None and project_state.project_instructions_enabled:
        try:
            registry = getattr(self.app, "workspace_registry_service", None)
        except Exception:
            registry = None
        try:
            if project_snapshot is not None:
                if project_snapshot.selected is None:
                    raise host.ProjectInstructionBindingRecovery("choose_binding")
                project_selection = host._project_selection_from_snapshot(
                    project_snapshot.selected
                )
            else:
                project_selection = host.resolve_project_instruction_binding(
                    session,
                    registry,
                    status_cache=host._app_remote_binding_status_cache(),
                )
        except host.ProjectInstructionBindingRecovery as exc:
            expected_setup_state = project_state
            if project_snapshot is not None:
                options = tuple(
                    host._project_selection_from_snapshot(item)
                    for item in project_snapshot.options
                )
            else:
                try:
                    options = host.list_project_instruction_bindings(
                        session,
                        registry,
                        status_cache=host._app_remote_binding_status_cache(),
                    )
                except host.ProjectInstructionBindingRecovery:
                    options = ()
            # An unselected session with no usable folder is a valid
            # scratch-only Chat/Workspace, not a project-instruction
            # setup failure. Keep the optional feature armed so a folder
            # added later can be selected, but do not block this send or
            # ask for one now. A previously selected folder still fails
            # closed and reaches recovery when it disappears or changes.
            if expected_setup_state.working_folder_binding_id is None and not options:
                project_selection = None
            else:
                if self._chat_start.is_accepted(session_id):
                    return self._refuse_turn_for_project_setup(
                        assistant_message_id, session_id, "project_binding_required"
                    )
                available, decision = await request_project_instruction_binding(
                    session_id, options, exc
                )
                if not available:
                    return self._refuse_turn_for_project_setup(
                        assistant_message_id, session_id, str(exc)
                    )
                action, binding_id = decision
                action, project_selection = (
                    host.commit_project_instruction_setup_decision(
                        store=self.store,
                        session_id=session_id,
                        registry=registry,
                        expected_state=expected_setup_state,
                        expected_options=options,
                        action=action,
                        binding_id=binding_id,
                    )
                )
                if action == "disable":
                    self._clear_project_instruction_delivery(session_id)
                    return self._refuse_turn_for_project_setup(
                        assistant_message_id,
                        session_id,
                        "project_instructions_disabled",
                    )
                if action != "select" or project_selection is None:
                    self._clear_project_instruction_delivery(session_id)
                    return self._refuse_turn_for_project_setup(
                        assistant_message_id, session_id, str(exc)
                    )
        if project_selection is not None:
            state = project_state
            if state.working_folder_binding_id is None:
                state = host.ProjectInstructionControlState(
                    project_instructions_enabled=True,
                    working_folder_binding_id=(project_selection.binding.binding_id),
                    working_folder_locator_fingerprint=(
                        project_selection.locator_fingerprint
                    ),
                    project_instruction_notice_key=None,
                )
                # task-33081: both steps write SQLite / scan the binding
                # root -- they must not run on the UI event loop that
                # owns this coroutine (the resolve-time preview twin
                # already runs them via asyncio.to_thread).
                await host.asyncio.to_thread(
                    self.store.set_session_project_instruction_state,
                    session_id,
                    state,
                )
            if host.is_remote(project_selection.root):
                # Task 19: AGENTS.md reads route through the remote
                # executor (reader strategy). ADR-317 prep-failure
                # posture: any reader failure warns content-free and
                # the dispatch proceeds without startup instructions.
                startup_candidate = (
                    await self._resolve_remote_project_instruction_startup(
                        project_selection,
                        registry=registry,
                        session_key=_remote_session_key,
                        startup_max_bytes=host.coerce_int_setting(
                            turn_context.tool_configuration.get(
                                "project_instructions_startup_max_bytes",
                                host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                            ),
                            host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                            minimum=host.MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                            maximum=host.MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        ),
                    )
                )
            else:
                startup_candidate = await host.asyncio.to_thread(
                    host.ProjectInstructionResolver().resolve_startup,
                    binding_id=project_selection.binding.binding_id,
                    binding_root=project_selection.root,
                    locator_fingerprint=project_selection.locator_fingerprint,
                    max_bytes=host.coerce_int_setting(
                        turn_context.tool_configuration.get(
                            "project_instructions_startup_max_bytes",
                            host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        ),
                        host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        minimum=host.MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                        maximum=host.MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                    ),
                    dispatch_started_wall_ns=host.time.time_ns(),
                    excluded_dirs=host._project_instruction_excluded_dirs(
                        project_selection
                    ),
                )
            from .console_project_dispatch import build_project_dispatch_callbacks

            (
                project_activation_callback,
                project_authority_guard,
                confirm_project_dispatch,
            ) = build_project_dispatch_callbacks(
                self,
                session_id=session_id,
                registry=registry,
                project_selection=project_selection,
                state=state,
                resolution=resolution,
            )

    self._active_assistant_message_ids[session_id] = assistant_message_id
    self._active_stream_tasks[session_id] = host.asyncio.current_task()
    self._stop_requested = False
    self._mcp_provider = None
    # A fresh per-run Event, captured by `should_cancel` below by
    # closure (not read off `self` each time) -- see
    # `_active_cancel_events`'s docstring for why this, rather than
    # `_stop_requested` alone, is what makes a still-running bridge
    # thread observe a Stop correctly (task-227).
    cancel_event = host.threading.Event()
    self._active_cancel_events[session_id] = cancel_event
    self.prompt_queue_coordinator.bind_hook_parent_cancellation(
        session_id, assistant_message_id, cancel_event
    )
    self._set_run_state(
        host.ConsoleRunState(host.ConsoleRunStatus.STREAMING, "Agent running."),
        session_id=session_id,
    )

    # Split the leading session system message off the payload; the
    # agent config carries it (composed with the operating prompt).
    session_system_prompt = ""
    agent_messages = list(provider_messages)
    if (
        agent_messages
        and agent_messages[0].get("role") == host.ConsoleMessageRole.SYSTEM.value
    ):
        session_system_prompt = str(agent_messages[0].get("content", ""))
        agent_messages = agent_messages[1:]

    conversation_id = self._agent_conversation_id(session_id)
    work_conversation_id = conversation_id
    if work_origin is host.WorkOrigin.AUTOMATIC:
        work_conversation_id = self._automatic_source_conversation(
            session_id,
            work_chain_id,
            wake_authorization=wake_authorization,
            chat_start_authorization=chat_start_authorization,
        )
    # noqa: E731 — tiny closure. Fix round 1 (Critical 1): reads ONLY
    # `cancel_event` -- captured by value, not via `self.
    # _active_cancel_events[session_id]` -- never the shared
    # `_stop_requested` flag. `_stop_requested` is GLOBAL (set by ANY
    # session's Stop/Close via `_signal_stop`), so OR'ing it in here
    # let stopping an unrelated session silently cancel THIS run too.
    # `cancel_event` alone is still correct once this run's own
    # `finally` below has already reset `_stop_requested` while the
    # bridge's background thread is still running (task-227: an
    # `asyncio.to_thread` call survives Task cancellation, so the
    # coroutine can finish handling a Stop and reset its own shared
    # bookkeeping well before the OS thread it detached from actually
    # returns) -- `stop_active_run`/`close_session`/`shutdown` all set
    # THIS session's `cancel_event` via `_signal_stop(session_id=...)`
    # the moment Stop is requested, and nothing ever clears it again
    # for this run, so a late poll from the surviving thread still
    # sees the cancellation regardless of `_stop_requested`'s state.
    should_cancel = lambda: cancel_event.is_set()  # noqa: E731
    capture_mode = capture_mode_override or host.ConsoleTraceCaptureMode.CAPTURE_OFF
    if capture_mode_override is None and preparation_id is not None:
        preparation = self._preparation_by_id(preparation_id)
        if preparation is not None:
            capture_mode = preparation.capture_mode

    # task-32344: from here to the provider dispatch is the window the
    # first send of a process pays in full -- the MCP catalog composition
    # below (async discovery I/O) and the lazy Personal Context bootstrap
    # above all, plus the profile and canvas providers built from it. The
    # row says so for the WHOLE window: entering only after composition
    # left a slow discovery rendering a blank row (Qodo #6 on PR #2586).
    from .console_pre_provider_stop import PreProviderSetupStop

    setup_stop = PreProviderSetupStop(
        self,
        assistant_message_id,
        cancel_event,
        prepares_row=prepare_retry or variant_mode,
    )
    # Outermost, so a raising phase exit cannot strand its registration.
    async with setup_stop, self._pre_provider_setup_phase(work_conversation_id):
        # P5-T6: compose this run's MCP tool provider (if eligible) HERE,
        # on the running main loop, BEFORE the bridge is dispatched onto
        # asyncio.to_thread below -- see `_compose_mcp_provider`'s own
        # docstring for why `compose_catalog()`'s async I/O can never run
        # from the worker thread. `(None, None)` (no service, kill switch
        # on, or nothing composed) leaves the bridge's MCP-free path
        # byte-identical to before this task.
        #
        # task-545/T6: `_compose_mcp_provider`'s own `mcp_review_hook`
        # (built from `build_mcp_review_hook`) is deliberately discarded
        # here rather than wired -- it is `None` whenever MCP is not
        # eligible for this run, and built-in tools (calculator/datetime
        # today) must be gated regardless of MCP eligibility. Changing
        # `_compose_mcp_provider`'s own return contract to drop that
        # second element was considered and rejected: several existing
        # test suites (`Tests/Chat/test_console_agent_swap.py`,
        # `Tests/UI/test_console_internals_decomposition.py`) assert its
        # exact `(provider, hook)` / `(None, None)` shape directly and sit
        # outside this task's file scope, so keeping that function
        # byte-identical and building the run-level hook separately here
        # is the lower-blast-radius choice.
        from .console_automatic_submission import capture_execution_roots

        run_admitted_roots, project_selection, project_authority_guard = (
            capture_execution_roots(
                self,
                session_id,
                project_selection,
                project_authority_guard,
                _remote_session_key,
            )
        )
        worktree_repo_authority = None
        if (
            project_selection is not None
            and project_selection.allow_write
            and len(run_admitted_roots) == 1
        ):
            selected_authority = run_admitted_roots[0]
            kill_switch_reader = self._console_tool_kill_switch_reader()

            def worktree_authority_guard(
                write: bool,
                authority=selected_authority,
                kill_switch=kill_switch_reader,
            ) -> bool:
                if kill_switch is None:
                    return False
                try:
                    return not kill_switch() and bool(authority.guard(write))
                except Exception:  # noqa: BLE001 - mutation gate fails closed
                    return False

            worktree_repo_authority = host.replace(
                selected_authority,
                guard=worktree_authority_guard,
            )
        (
            mcp_provider,
            builtin_gate,
            local_provider,
            local_review_hook,
        ) = await self._compose_agent_request_providers(
            session_id=session_id,
            project_selection=project_selection,
            project_authority_guard=project_authority_guard,
            turn_context=turn_context,
            admitted_roots=run_admitted_roots,
        )
        virtual_cli_provider, virtual_cli_review_hook = (
            self._compose_virtual_cli_provider(
                session_id=session_id,
                turn_context=turn_context,
                project_root=(
                    project_selection.root if project_selection is not None else None
                ),
                project_root_identity=(
                    project_selection.root_identity
                    if project_selection is not None
                    else None
                ),
                project_root_guard=project_authority_guard,
                admitted_roots=run_admitted_roots,
            )
        )
        raw_shell_provider, raw_shell_review_hook = self._compose_raw_shell_provider(
            session_id=session_id,
            turn_context=turn_context,
            project_root=(
                project_selection.root if project_selection is not None else None
            ),
        )
        self._mcp_provider = mcp_provider

        # Resolve the direct Library provider before building the shared
        # review hook: Agent Lesson saves are classified by that exact
        # provider instance and its approve-once authority must be visible
        # to the same instance later registered for dispatch.
        library_provider: host.Any | None = None
        library_provider_authority: host.Any | None = None
        if self._library_provider_factory is not None:
            try:
                library_selection = self._library_provider_for_context(turn_context)
                if library_selection is not None:
                    library_provider, library_provider_authority = library_selection
            except Exception:  # noqa: BLE001 -- never block a send
                host.logger.opt(exception=True).warning(
                    "library_provider_factory failed; running without Library tools"
                )

        # task-545/T6: build THIS run's built-in permission gate and hand
        # the SAME instance to both the review hook (below) and
        # `ConsoleAgentBridge.run_reply` (which threads it into the
        # `BuiltinToolProvider` that actually invokes tools) -- a second,
        # independently-built gate would silently desynchronize stamps:
        # a decision made here would never be visible to `invoke()`'s own
        # gate, and vice versa. `build_builtin_gate(None)` (no
        # `unified_mcp_service` on the app) is fail-closed-correct, not
        # "ungated" -- see that function's own docstring.
        # Only `.tool_for(name)` is used by the review hook below, to
        # resolve a `ToolCall.name` to the `Tool` object `builtin_gate.
        # resolve` needs -- this instance is never used to invoke a tool,
        # so it does not need to be the SAME `BuiltinToolProvider` object
        # the bridge's registry actually dispatches through (its `_tools`
        # dict is stateless data rebuilt identically by any instance).
        # Round 1 review CRITICAL 1: resolve THIS run's OWN workspace id --
        # the SAME lookup `ConsoleAgentBridge.run_reply` makes
        # (`self._store.session_workspace_id(session_id)`) for the real
        # `BuiltinToolProvider(workspace_id=...)` dispatch below -- and
        # thread it into the review hook so its `path_precheck_failed`
        # pre-flight resolves the IDENTICAL workspace dispatch will, never
        # whatever happens to be active in the UI for a parked/background
        # session. `KeyError` (an already-closed session) degrades to
        # `None`, matching `allowed_file_roots`'s own fail-safe posture.
        review_workspace_id = frozen_workspace_id
        builtin_review_provider = host.BuiltinToolProvider(
            gate=builtin_gate,
            workspace_id=review_workspace_id,
            workspace_read_binding_ids=frozen_workspace_read_binding_ids,
            workspace_write_binding_ids=frozen_workspace_write_binding_ids,
            workspace_binding_authority=frozen_workspace_binding_authority,
            sandbox_root=scratch_snapshot.root,
            sandbox_lease=scratch_lease,
        )
        # Task 9: bind THIS run's owning session id into the approval
        # bridge so `request_mcp_approvals` can (a) scope its cancellation
        # check to this run's own cancel event rather than falling back to
        # whichever session is currently VIEWED (finding #1), and (b) park
        # rather than mount when `session_id` is not the active session.
        review_hook = host.build_tool_review_hook(
            builtin_gate,
            builtin_review_provider,
            mcp_provider,
            host.functools.partial(self.request_mcp_approvals, session_id=session_id),
            workspace_id=review_workspace_id,
            # TASK-631: the switch must cover the tool families NEITHER
            # provider claims (skills/spawn/find/load), and this hook is the
            # only choke point they all pass. Read fresh per turn so a
            # mid-run flip takes effect on the next batch. Absent service ->
            # no switch to honor (None), matching `_compose_mcp_provider`.
            kill_switch=self._console_tool_kill_switch_reader(),
            library_provider=library_provider,
            recovery_provider_reader=lambda: recovery_provider,
            recovery_persona_rules=turn_context.persona_policy_rules,
        )

        # Local tools (ADR-300): same per-run composition point. Both
        # hooks see every batch; each gates only what its provider owns,
        # so the combined hook is a collision-free merge.
        if local_review_hook is not None:
            review_hook = host.build_combined_review_hook(
                [review_hook, local_review_hook]
            )
        if virtual_cli_review_hook is not None:
            review_hook = host.build_combined_review_hook(
                [review_hook, virtual_cli_review_hook]
            )
        if raw_shell_review_hook is not None:
            review_hook = host.build_combined_review_hook(
                [review_hook, raw_shell_review_hook]
            )
        managed_skill_promotion_gate = None
        if library_provider is not None:
            from tldw_chatbook.Agents.agent_lesson_promotion import (
                ManagedSkillProposalGate,
            )

            managed_skill_promotion_gate = ManagedSkillProposalGate()
            managed_skill_review_hook = host.build_managed_skill_promotion_review_hook(
                managed_skill_promotion_gate,
                host.functools.partial(
                    self.request_mcp_approvals, session_id=session_id
                ),
            )
            review_hook = host.build_combined_review_hook(
                [review_hook, managed_skill_review_hook]
            )

        # TASK-1971 (Agent Change Review): THIS run's tracked roots -- the
        # same workspace folder bindings the file tools resolve against.
        # Best-effort: an unavailable registry yields no roots and an
        # untracked (but otherwise normal) run.
        from tldw_chatbook.Tools.workspace_file_roots import frozen_workspace_roots

        live_frozen_roots = set(
            frozen_workspace_roots(
                frozen_workspace_id,
                frozen_workspace_binding_authority,
                registry=getattr(self.app, "workspace_registry_service", None),
            )
        )
        change_roots: list = [
            host.Path(root)
            for root in turn_context.workspace_roots
            if host.Path(root) in live_frozen_roots
        ]

        # Swap site: the agent loop runs synchronously on a worker thread via
        # asyncio.to_thread, so Stop is cooperative-only -- `should_cancel` is
        # polled between chunks/steps inside the bridge, never preempts the
        # thread itself. A provider that hangs mid-request without emitting a
        # single chunk cannot be interrupted here; RunBudget.max_wall_seconds
        # (agent_models.py) is what bounds a run overall, but only once
        # control returns to a checkpoint the loop actually polls -- it is
        # not a hard timeout on an in-flight, zero-chunk provider call.
        if self._teardown_refuses_turn(session_id):
            return self._accepted_shutdown_before_dispatch(
                assistant_message_id, session_id
            )
        await self._wait_for_trace_maintenance_dispatch()
        self._trace_last_provider_activity = host.time.monotonic()
        if self.prompt_queue_coordinator.continuation_cancelled(session_id):
            self._signal_stop(session_id=session_id)
            raise host.asyncio.CancelledError
        if before_provider_dispatch is not None:
            try:
                await before_provider_dispatch()
            except host.asyncio.CancelledError:
                if not cancel_event.is_set():
                    raise
                # No agent worker exists yet; the durable handoff has drained
                # and its published checkpoint can now settle normally.
                stopped = self._mark_stream_stopped(
                    assistant_message_id, visible_copy="Response stopped."
                )
                return host.ConsoleSubmitResult(True, True, stopped.content)
        elif preparation_id is not None and not self._transition_preparation(
            preparation_id,
            host.ConsoleTurnPreparationState.ACCEPTED,
            host.ConsoleTurnPreparationState.DISPATCH_STARTED,
        ):
            raise RuntimeError("Prepared turn changed before provider dispatch.")
        if generation_token is None:
            generation_token = self.store.begin_generation_attempt(
                assistant_message_id,
                expected_continuation=(
                    restore_provider_continuation
                    if resume_provider_continuation
                    else None
                ),
                expected_continuation_message_version=(
                    restore_provider_message_version
                    if resume_provider_continuation
                    else None
                ),
            )
        _generation_handoff.issue(generation_token)
        if variant_mode:
            self.store.begin_variant_stream(
                assistant_message_id,
                generation_token=generation_token,
            )
        elif prepare_retry:
            self.store.prepare_message_retry(
                assistant_message_id,
                generation_token=generation_token,
            )
        setup_stop.row_prepared = True  # a Stop now settles the new attempt
        personal_context_service = await self._personal_context_service()
        profile_context_service = await self._personal_context_builder(
            personal_context_service
        )
        trusted_profile_user_message = None
        if trusted_profile_user_message_id is not None:
            try:
                candidate = self.store.get_message(trusted_profile_user_message_id)
                if (
                    candidate.role is host.ConsoleMessageRole.USER
                    and not getattr(candidate.metadata, "origin", None)
                    and self.store.session_id_for_message(candidate.id) == session_id
                ):
                    trusted_profile_user_message = candidate
            except KeyError:
                pass
        from .console_review_delivery import compose_turn_review_provider

        code_review_provider = await compose_turn_review_provider(
            self,
            session_id=session_id,
            message=trusted_profile_user_message,
            roots=run_admitted_roots,
            resolution=resolution,
            conversation_id=conversation_id,
            wake_authorization=wake_authorization,
            tool_configuration=turn_context.tool_configuration,
            skill_bindings=skill_bindings,
            project_selection=project_selection,
            project_authority_guard=project_authority_guard,
            confirm_project_dispatch=confirm_project_dispatch,
            startup_candidate=startup_candidate,
        )
        profile_provider = None
        if personal_context_service is not None and session is not None:
            profile_provider = await host.asyncio.to_thread(
                host._compose_profile_tool_provider,
                personal_context_service,
                workspace_id=session.workspace_id,
                ephemeral=session.ephemeral,
                run_id=assistant_message_id,
                session_id=session_id,
                current_user_message=trusted_profile_user_message,
                kill_switch=(
                    self._console_tool_kill_switch_reader() or (lambda: False)
                ),
            )
        canvas_provider = None
        canvas_authority = None
        canvas_controller = getattr(self.store, "canvas_turn_controller", None)
        try:
            canvas_enabled = self._canvas_enabled_reader() is True
        except Exception:  # noqa: BLE001 - Canvas tool advertising fails closed
            canvas_enabled = False
        if canvas_enabled and canvas_controller is not None and session is not None:
            from tldw_chatbook.Agents.canvas_tool_provider import CanvasToolProvider
            from tldw_chatbook.Canvas.models import CanvasScope

            canvas_run_id = str(host.uuid4())
            canvas_scope = CanvasScope(
                session_id=session_id,
                conversation_id=conversation_id,
                active_message_ids=self.store.canvas_active_path_message_ids(
                    session_id
                ),
                selected_canvas_id=None,
                selected_revision_id=None,
                run_id=canvas_run_id,
            )
            canvas_scope = canvas_controller.capture_selected_scope(canvas_scope)
            canvas_run = canvas_controller.register_run(
                canvas_scope,
                assistant_message_id=assistant_message_id,
                temporary=session.ephemeral,
            )
            canvas_provider = CanvasToolProvider(
                canvas_run,
                scope=canvas_scope,
                enabled_reader=self._canvas_enabled_reader,
                disabled_reader=self._canvas_disabled_reader,
            )
            canvas_authority = canvas_provider.issue_registration_authority()
    if setup_stop.stopped:  # TASK-33621.22: Stop while "Connecting tools…"
        return await setup_stop.settle(session_id)
    recovery_runtime = getattr(self, "_archive_recovery_runtime", None)
    recovery_provider = (
        recovery_runtime.admit(turn_context, session)
        if recovery_runtime is not None
        and session is not None
        and work_origin is host.WorkOrigin.MANUAL
        and not resume_provider_continuation
        else None
    )
    # TASK-35311: bound by the provider-worker await inside the try; a
    # cancellation landing before that await returns leaves them None,
    # and the CancelledError handler recovers a finished outcome from
    # the shielded worker instead.
    run_id: str | None = None
    outcome = None
    try:
        if recovery_provider is not None:
            from .archive_recovery.runtime import prepare_archive_followup

            agent_messages, trace_request = await prepare_archive_followup(
                self,
                recovery_runtime,
                recovery_provider,
                turn_context,
                resolution,
                agent_messages,
                trace_request,
                trusted_profile_user_message_id,
            )
        # run_reply returns (run_id, outcome): run_id lets us write the
        # produced reply's PERSISTED id back onto the run after
        # completion (the load-bearing write for resume marker anchoring).
        reply_loop = host.asyncio.get_running_loop()
        settled_budgets = []
        accepted_chain = self.prompt_queue_coordinator._chains.get(session_id)
        accepted_turn_id = getattr(
            getattr(accepted_chain, "request", None), "turn_id", None
        )
        provider_worker = self._run_maintenance_agent_call(
            self._agent_bridge.run_reply,
            work_origin=work_origin,
            work_chain_id=work_chain_id,
            conversation_id=work_conversation_id,
            session_id=session_id,
            expected_progress_owner_id=progress_owner_id,
            resolution=resolution,
            assistant_message_id=assistant_message_id,
            accepted_turn_id=accepted_turn_id,
            parent_budget_deadline=turn_context.tool_configuration.get(
                "native_parent_budget_deadline"
            ),
            on_primary_budget_settled=lambda budget: settled_budgets.append(
                (budget, host.time.monotonic())
            ),
            model=(
                getattr(resolution, "model", None) or turn_context.effective_model or ""
            ),
            session_system_prompt=session_system_prompt,
            agent_messages=agent_messages,
            should_cancel=should_cancel,
            provider_stream_signals=stream_signals,
            supersede_previous=bool(prepare_retry or variant_mode),
            mcp_provider=mcp_provider,
            builtin_gate=builtin_gate,
            scratch_root=scratch_snapshot.root,
            scratch_lease=scratch_lease,
            review_tool_calls=review_hook,
            on_steer_ready=(
                lambda steer, _sid=session_id: self._active_steer_hooks.__setitem__(
                    _sid, steer
                )
            ),
            on_redirect_ready=(
                lambda redirect, _sid=session_id: (
                    self._active_redirect_hooks.__setitem__(_sid, redirect)
                )
            ),
            local_provider=local_provider,
            virtual_cli_provider=virtual_cli_provider,
            raw_shell_provider=raw_shell_provider,
            recovery_provider=recovery_provider,
            library_provider=library_provider,
            library_authority=library_provider_authority,
            managed_skill_promotion_gate=managed_skill_promotion_gate,
            # Workspace assistant defaults (Task 7): this turn's persona
            # policy rules, from the same captured turn context the
            # providers above were composed under -- the bridge applies
            # them as the run's narrowing-only advertising filter and
            # per-run call caps.
            persona_policy_rules=turn_context.persona_policy_rules,
            code_review_provider=code_review_provider,
            profile_provider=profile_provider,
            canvas_provider=canvas_provider,
            canvas_authority=canvas_authority,
            native_tools_enabled=bool(
                turn_context.tool_configuration.get("native_tool_calls_enabled", True)
            ),
            change_roots=change_roots,
            worktree_repo_authority=worktree_repo_authority,
            change_root_aliases=turn_context.change_review_root_aliases,
            change_review_skipped_roots=(turn_context.change_review_skipped_roots),
            on_tool_surface_selected=(
                lambda count: reply_loop.call_soon_threadsafe(
                    self._publish_selected_tool_count, session_id, count
                )
            ),
            turn_skill_bindings=skill_bindings,
            turn_bundle_block=skill_bundle_block,
            skills_context=(
                turn_context.skill_context_maximum
                if turn_context.skill_context_maximum.get("backend") == "local"
                else None
            ),
            workspace_id=frozen_workspace_id,
            workspace_ephemeral=frozen_workspace_ephemeral,
            workspace_read_binding_ids=frozen_workspace_read_binding_ids,
            workspace_write_binding_ids=frozen_workspace_write_binding_ids,
            workspace_binding_authority=frozen_workspace_binding_authority,
            request_skill_install_confirm=host.functools.partial(
                self.request_skill_install_confirm, session_id=session_id
            ),
            startup_instruction_candidate=startup_candidate,
            project_instruction_nested_max_bytes=host.coerce_int_setting(
                turn_context.tool_configuration.get(
                    "project_instructions_nested_max_bytes",
                    host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                ),
                host.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                minimum=host.MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
                maximum=host.MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            ),
            run_budget=turn_context.tool_configuration.get("agent_run_budget_maximum"),
            confirm_project_instruction_dispatch=confirm_project_dispatch,
            on_project_instruction_activation=project_activation_callback,
            # Advertised must equal usable (the #847 lesson, restated in
            # the run_skill_script docstring below): only pass the
            # confirm callback -- and therefore only let the bridge
            # build/advertise the run_skill_script tool at all -- once a
            # UI sink is actually wired. Until then
            # `request_skill_script_confirm`'s own no-UI guard would
            # auto-deny every call, offering the model a tool it can
            # never successfully use.
            request_skill_script_confirm=(
                host.functools.partial(
                    self.request_skill_script_confirm, session_id=session_id
                )
                if self.set_pending_skill_script is not None
                or self._interrupt_host.has_retained_decision_target(session_id)
                else None
            ),
            request_chat_create_confirm=(
                host.functools.partial(
                    self.request_chat_create_confirm, session_id=session_id
                )
            ),
            execute_agent_chat_create=self.execute_agent_chat_create,
            prepare_agent_chat_create=self.prepare_agent_chat_create,
            # TASK-28238 phase 2 Task 6/Task 7: same "advertised must
            # equal usable" gate as `request_skill_script_confirm`
            # above. Task 5 originally advertised
            # `merge_agent_worktree`/`discard_agent_worktree` whenever
            # the fleet was active regardless, refusing per-call
            # instead of disappearing -- the Task 7 ruling replaced
            # that: `build_console_first_request_plan` now discloses
            # the pair only when a confirm surface is wired
            # (`worktree_merge_enabled=request_worktree_merge_confirm
            # is not None`), so passing `None` here when no UI sink is
            # wired both hides the tools from disclosure and (belt-
            # and-braces) lets `AgentService`'s own "no approval
            # surface is available in this session" refusal fire
            # directly rather than routing through
            # `request_worktree_merge_confirm`'s no-UI guard to
            # produce a less accurate "declined" message.
            request_worktree_merge_confirm=(
                host.functools.partial(
                    self.request_worktree_merge_confirm, session_id=session_id
                )
                if self.worktree_confirmation_enabled
                else None
            ),
            # PR2a Task 7: the fleet cancels/abandons children on the
            # bridge's worker thread and hands each stopped run's id
            # here, so a card still on screen for a child that is
            # already `cancelled` is denied and cleared rather than
            # left pressable (an approval that would still EXECUTE the
            # tool for real). Run-keyed, so a live sibling child --
            # which shares this same session -- keeps its own card.
            revoke_approvals=self.revoke_approval_rounds_for_run,
            plugin_cancel_root=cancel_event.set,
            on_tool_terminal=self.complete_definitive_tool,
            on_tool_result_terminal=host.functools.partial(
                self.observe_response_rule_tool_result,
                session_id,
                assistant_message_id,
            ),
            on_run_terminal=self.complete_definitive_run,
            restore_provider_continuation=restore_provider_continuation,
            restore_provider_target=restore_provider_target,
            expand_provider_continuation=expand_provider_continuation,
            resume_provider_continuation=resume_provider_continuation,
            continuation_sidecar=continuation_sidecar,
            continuation_target=continuation_history_target,
            continuation_owner_key=(
                host.NATIVE_MESSAGE_ID_KEY if continuation_sidecar else None
            ),
            thinking_sidecar=thinking_sidecar,
            thinking_policy=thinking_policy,
            thinking_owner_key=(
                host.NATIVE_MESSAGE_ID_KEY if thinking_sidecar else None
            ),
            generation_token=generation_token,
            propagate_trace_call_persistence_errors=(
                propagate_trace_call_persistence_errors
            ),
            capture_mode=capture_mode,
            trace_request=trace_request,
            _generation_handoff=_generation_handoff,
            profile_context_service=profile_context_service,
        )
        # The submit may stop before this wrapper's native call returns.
        # Its existing maintenance owner observes late wrapper failures too.
        provider_task = self._retain_maintenance_task(provider_worker)
        self._active_provider_workers[session_id] = provider_task

        async def _shielded_provider_operation() -> host.Any:
            return await host.asyncio.shield(provider_task)

        try:
            run_id, outcome = await self._chat_start.run_provider_worker(
                session_id, _shielded_provider_operation()
            )
        except BaseException:
            # The registry entry deliberately survives a cancellation:
            # the CancelledError handler below recovers this task's
            # finished result from it before clearing.
            raise
        else:
            if self._active_provider_workers.get(session_id) is provider_task:
                self._active_provider_workers.pop(session_id, None)
        if self.response_rules is not None and len(settled_budgets) == 1:
            self.response_rules.record_generation_budget(
                assistant_message_id, accepted_turn_id, *settled_budgets[0]
            )
        from tldw_chatbook.Agents.automatic_work_runtime import (
            current_automatic_work,
        )

        automatic = current_automatic_work()
        if automatic is not None and automatic.goal is not None:
            automatic.goal.record_outcome(run_id, outcome)
    except host.asyncio.CancelledError:
        if work_origin is host.WorkOrigin.AUTOMATIC:
            cancel_event.set()
        if cancel_event.is_set():
            # TASK-35311: a shutdown or session-close cancel landing
            # AFTER the provider worker already produced RUN_DONE must
            # not stamp ``stopped`` on a reply that provably finished.
            # Recover the outcome -- bound above when the await
            # returned before the cancellation, or retained by the
            # shielded provider worker when it did not -- and run the
            # success finalization, which the stop-reason-aware
            # ``stopped_now`` guard lets through. A user Stop keeps
            # the stopped contract below, as does any stream whose
            # outcome is not DONE.
            from tldw_chatbook.Agents.agent_models import RUN_DONE

            done_outcome = None
            worker = self._active_provider_workers.get(session_id)
            if getattr(outcome, "status", None) == RUN_DONE:
                done_outcome = outcome
            else:
                if (
                    worker is not None
                    and worker.done()
                    and not worker.cancelled()
                    and worker.exception() is None
                ):
                    run_id, maybe_done = worker.result()
                    if getattr(maybe_done, "status", None) == RUN_DONE:
                        done_outcome = maybe_done
            if worker is not None:
                if self._active_provider_workers.get(session_id) is worker:
                    self._active_provider_workers.pop(session_id, None)
            if (
                done_outcome is not None
                and self._active_stop_reason(session_id) != "user"
            ):
                # The cancellation has been consumed. Citation selection
                # must stay on this active stream task; its existing cancel
                # path retains the original body without dispatching repair.
                if citation_repair_session is not None:
                    citation_repair_session.cancel_reason = self._active_stop_reason(
                        session_id
                    )
                return await self._finalize_agent_reply(
                    assistant_message_id,
                    session_id,
                    done_outcome,
                    variant_mode=variant_mode,
                    cancel_event=cancel_event,
                    run_id=run_id,
                    citation_repair_session=citation_repair_session,
                    stream_signals=stream_signals,
                    resolution=resolution,
                )
            # Whatever the provider already billed for this turn's
            # completed steps is real money -- record it (partial)
            # before the terminal mark, exactly as the direct path's
            # own CancelledError branch does.
            self._attach_stream_usage(
                assistant_message_id, stream_signals, resolution, partial=True
            )
            # PR3a-1 Task 6b (audit F3): a Stop ends the TURN, not its
            # surviving children -- so this branch needs the same
            # post-turn watch the normal finalizer sets. PR3a-2 Task 3:
            # message + partial flag ride along so the drain fold can
            # re-attach to this row with the same (partial) semantics.
            self._watch_post_turn_usage(
                session_id,
                stream_signals,
                resolution,
                assistant_message_id=assistant_message_id,
                partial=True,
            )
            try:
                stopped = self._mark_stream_stopped(
                    assistant_message_id, visible_copy="Response stopped."
                )
            except KeyError:
                return self._session_closed_result(session_id=session_id)
            # task-543: this is the dominant user-Stop path --
            # ``task.cancel()`` raised before ``(run_id, outcome)`` ever
            # bound, so recover the active run's id via the bridge's
            # latest-unanchored-primary lookup and record the stopped
            # reply's persisted id, same as every finalizer terminal
            # path. A never-persisted stop (or an anchored/missing row)
            # no-ops and leaves the row NULL -> ordinal fallback.
            self._record_run_assistant_message(
                self._latest_unanchored_primary_run_id(work_conversation_id),
                stopped,
            )
            return host.ConsoleSubmitResult(True, True, stopped.content)
        raise
    except host.ConsoleDispatchSettlementError:
        raise
    except Exception as exc:
        if propagate_trace_call_persistence_errors and isinstance(
            exc, host.TraceCallPersistenceError
        ):
            raise
        # Bridge failures can originate OUTSIDE AgentService's own
        # narrow loop guard (agent_service.py wraps only
        # `run_agent_loop`; `db.create_run`, `_persist`
        # (append_steps/set_status), and `supersede_run_tree` are not
        # covered). Left uncaught here, run_state would stay STREAMING
        # forever and every future send on every session would be
        # rejected ("A run is already running in this tab.") until app
        # restart (Plan-B Task 6 Critical 1). Mirror the legacy stream
        # path's catch-all above, including the Task-1 variant-restore
        # semantics: `begin_variant_stream`/`prepare_message_retry`
        # already ran before the bridge call, so `mark_message_failed`
        # resolves the correct terminal content on its own (restores
        # the pre-regenerate base + status for a failed regenerate;
        # preserves whatever partial content already streamed
        # otherwise).
        visible_copy = host.console_failure_copy(exc, "Agent run failed")
        visible_copy += host.image_refusal_hint(  # TASK-33621.11 AC#3: typed
            exc, lambda: self._session_history_carries_images(session_id)
        )
        try:
            self.store.mark_message_failed(assistant_message_id)
        except KeyError:
            return self._session_closed_result(session_id=session_id)
        self._append_failure_system_row(session_id, visible_copy)
        self._set_run_state(
            host.ConsoleRunState(host.ConsoleRunStatus.FAILED, visible_copy),
            session_id=session_id,
        )
        return host.ConsoleSubmitResult(True, True, visible_copy)
    finally:
        self._active_provider_workers.pop(session_id, None)
        if recovery_provider is not None:
            await recovery_runtime.release(recovery_provider.authority)
        # Task 3b: this finally intentionally pops NONE of the three
        # per-session entries (stream task, assistant message id,
        # cancel event) -- `_finalize_agent_reply` below (and, through
        # it, `_finalize_agent_success`'s citation-repair post-
        # generation check) still runs AFTER this try/finally, on this
        # SAME task, and both `owns_request()` (stream task/assistant
        # message id) and `cancellation_requested()` (cancel_event,
        # NOT clear()'d for the same reason noted where it was created
        # above -- task-227) need to see this run as still live and
        # still cancellable. The wrapper (`_stream_assistant_response_
        # inner`), which awaits this entire call including
        # `_finalize_agent_reply`, is what clears every per-session
        # entry once everything has actually finished.
        run_state = self.run_state_for(session_id)
        host.logger.info(
            "console agent reply end",
            assistant_message_id=assistant_message_id,
            run_status=run_state.status.value,
            run_copy=run_state.visible_copy,
        )

    # Captured here, before `_finalize_agent_reply` runs: this run's own
    # cancel_event is the authority on whether IT was stopped,
    # independent of what status `mark_message_stopped` may have left
    # the message at (task-227 AC3 follow-up -- see the guard below).
    # Kept as a direct await on purpose (TASK-35311): finalize must run
    # in THIS task -- the citation-repair post-generation step keys on
    # the running task's context -- and the shutdown-window recovery is
    # handled on this same task in the CancelledError handler above.
    return await self._finalize_agent_reply(
        assistant_message_id,
        session_id,
        outcome,
        variant_mode=variant_mode,
        cancel_event=cancel_event,
        run_id=run_id,
        citation_repair_session=citation_repair_session,
        stream_signals=stream_signals,
        resolution=resolution,
    )
