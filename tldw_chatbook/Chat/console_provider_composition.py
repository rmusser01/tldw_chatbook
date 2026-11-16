"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from . import console_chat_controller as native_host


async def _compose_mcp_provider(
    self,
    session_id: str | None = None,
    *,
    publish_counts: bool = True,
    profile_id_provider: native_host.Callable[[], str] | None = None,
    persona_policy_provider: native_host.Callable[
        [], native_host.PersonaToolPolicy | None
    ]
    | None = None,
    maximum_tool_ids: frozenset[str] | None = None,
    maximum_definition_hashes: native_host.Mapping[str, str] | None = None,
    plugin_maximum: native_host.Mapping | None = None,
    local_kill_switch: "native_host.LocalKillSwitchRead | None" = None,
    _freeze_definition_maximum_on_compose: bool = False,
    _retain_empty_plugin_providers: bool = False,
) -> native_host.MCPToolProvider | None:
    """Build + compose THIS run's MCPToolProvider for the running main loop.

    MUST be awaited on the real Textual main loop (``_run_agent_reply``,
    BEFORE the bridge's own ``asyncio.to_thread``): the provider is bound
    to that loop for tool calls. TASK-33620.15.1: one kill-switch read
    decides MCP and local tools (``local_kill_switch``), and a catalog is
    reused while nothing it was composed from has changed
    (``console_run_start_tools``).

    TASK-632: returns the provider ALONE. It used to also build and
    return a per-run `build_mcp_review_hook` closure, but TASK-545's
    run-level `build_tool_review_hook` made that second element dead --
    the production call site unpacked it into `_unused_...` and threw it
    away, while the closure was still constructed on every run.

    Returns ``None`` whenever MCP tools should not be offered
    this run: no ``unified_mcp_service`` on the app, the kill switch
    is on, ``get_kill_switch``/``compose_catalog`` raised, or the
    composed catalog is empty (nothing to register, and -- since
    ``not_connected_count`` is only ever non-zero for servers that
    already contributed at least one eligible tool -- nothing an
    empty catalog could usefully report either). Live composition
    publishes this run's inspector counts on every return path via
    ``_publish_mcp_inspector_counts``. Disposable Context composition
    passes ``publish_counts=False`` and never mutates those app-level
    inspector values.

    Args:
        session_id: The run's OWNING session (Task 3/9) -- threaded
            into the composed provider's ``approval_callback`` (via a
            ``functools.partial`` binding, since ``MCPToolProvider``
            calls it with a fixed ``[pending]`` single-list arg) so a
            single-call fallback approval raised through
            ``invoke()``'s own gate parks/mounts and scopes its cancel
            check exactly like the batch review-hook path does.
            ``None`` (the default -- every pre-Task-9 call site) keeps
            ``request_mcp_approvals``' legacy no-session behavior.
        publish_counts: Whether to publish app-level MCP inspector counts.
            Live dispatch keeps the default; disposable preview disables it.
        profile_id_provider: Workspace assistant defaults (Task 7) --
            callable returning the workspace's named permission profile
            this run's MCP gates resolve under (catalog composition,
            batch review, and `invoke`'s own fresh gate). ``None`` keeps
            the provider's own ``"default"`` fallback -- byte-identical
            to the pre-profiles behavior.
        persona_policy_provider: Persona require_confirmation floor
            (final review) -- callable returning this run's parsed
            persona tool policy; the provider floors matching tools'
            invoke-time gates to "ask" under it. ``None`` keeps the
            pre-feature behavior.

    Returns:
        A composed ``MCPToolProvider`` ready to hand to
        ``ConsoleAgentBridge.run_reply`` when eligible; ``None``
        otherwise.
    """

    def publish(tool_count: int | None, not_connected: int | None) -> None:
        if local_kill_switch is not None and local_kill_switch._current is not None:
            local_kill_switch._current()
        if publish_counts:
            composed_switch = (
                (service, local_kill_switch.value)
                if (
                    local_kill_switch is not None
                    and local_kill_switch.read
                    and local_kill_switch.error is None
                    and (local_kill_switch._include_mcp or local_kill_switch._need_local)
                )
                else None
            )
            self._publish_mcp_inspector_counts(
                tool_count, not_connected, session_id, composed_switch=composed_switch
            )

    service = getattr(self.app, "unified_mcp_service", None)
    if service is None:
        if _freeze_definition_maximum_on_compose:
            raise PermissionError("mcp_catalog_source_changed")
        publish(None, None)
        return None
    bound_request_approvals = native_host.functools.partial(
        self.request_mcp_approvals, session_id=session_id
    )
    uncomposed = native_host.MCPToolProvider(
        service=service,
        main_loop=native_host.asyncio.get_running_loop(),
        approval_callback=bound_request_approvals,
        # task-1337 (plan Task 8): the Console always excludes the
        # shadowed built-in raw names, in BOTH retrieval modes -- the
        # run's own Library provider (direct or bounded-RAG) is the only
        # Library retrieval path Console agents get.
        builtin_raw_name_exclusions=native_host.CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        profile_id_provider=profile_id_provider,
        persona_policy_provider=persona_policy_provider,
        # TASK-33106: ADR-183's built-in character writes follow THIS
        # session's runtime source, like the Console's character_save.
        runtime_source_provider=(
            None
            if session_id is None
            else native_host.functools.partial(self._session_runtime_source, session_id)
        ),
        maximum_tool_ids=maximum_tool_ids,
        maximum_definition_hashes=maximum_definition_hashes,
        **(
            {"_freeze_definition_maximum_on_compose": True}
            if _freeze_definition_maximum_on_compose
            else {}
        ),
    )
    # TASK-33620.15.1: composed here, on the loop, or reused while nothing
    # it read has changed; ``None`` = switched off or a read failed.
    from .console_ordinary_tree import bind_ordinary_tree, current_ordinary_tree
    from .console_run_start_tools import LocalKillSwitchRead, compose_run_mcp_provider

    if publish_counts and local_kill_switch is None:
        local_kill_switch = LocalKillSwitchRead()
    ordinary = current_ordinary_tree(self)
    if ordinary is None:
        from .console_preparation_reads import current_workflow_preparation

        preparation = current_workflow_preparation(self)
        if preparation is None:
            provider = await compose_run_mcp_provider(
                service, uncomposed, local_kill_switch
            )
        else:
            operation = preparation.borrow_preparation(
                self._hooks_v2_runtime
            ).reserve_tool()
            with preparation.activate_source():
                try:
                    provider = await compose_run_mcp_provider(
                        service, uncomposed, local_kill_switch
                    )
                finally:
                    operation.finish()
    else:
        operation = ordinary.borrow(self._hooks_v2_runtime, opaque=False).reserve_tool()
        with (
            ordinary.source.activate()
            if ordinary.source is not None
            else bind_ordinary_tree(ordinary)
        ):
            try:
                provider = await compose_run_mcp_provider(
                    service, uncomposed, local_kill_switch
                )
            finally:
                operation.finish()
    if provider is None:
        publish(None, None)
        return None
    providers = [provider]
    if plugin_maximum is not None:
        from tldw_chatbook.MCP.connection_ownership import ConnectionOwnership
        from tldw_chatbook.Plugins.mcp_provider import (
            MCPProviderGroup,
            PluginMCPProvider,
        )

        local = getattr(self._skills_service, "local_service", None)
        plugins = getattr(local, "plugin_service", None)
        if plugins is None:
            raise PermissionError("plugin_mcp_authority_unavailable")
        snapshots = await plugins.component_snapshots(plugin_maximum)
        local_mcp = service.local_service
        ownership = local_mcp.connection_ownership
        if ownership is None:
            ownership = ConnectionOwnership(
                plugin_service=plugins, local_service=local_mcp
            )
        if ownership.plugins is not plugins:
            raise PermissionError("plugin_mcp_owner_changed")
        for snapshot in snapshots:
            if not any(
                snapshot.inspection.inventory[key].kind == "mcp"
                for key in snapshot.selection
            ):
                continue
            owned = PluginMCPProvider(
                plugin_service=plugins,
                ownership=ownership,
                snapshot=snapshot,
                service=service,
                main_loop=native_host.asyncio.get_running_loop(),
                approval_callback=bound_request_approvals,
                builtin_raw_name_exclusions=native_host.CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
                profile_id_provider=profile_id_provider,
                persona_policy_provider=persona_policy_provider,
                runtime_source_provider=(
                    None
                    if session_id is None
                    else native_host.functools.partial(
                        self._session_runtime_source, session_id
                    )
                ),
                maximum_tool_ids=maximum_tool_ids,
                maximum_definition_hashes=maximum_definition_hashes,
            )
            await owned.compose_catalog()
            providers.append(owned)
        provider = MCPProviderGroup(providers)
    catalog = provider.list_catalog()
    if not catalog:
        publish(None, None)
        # Explicit discovery needs the actual owned connection before any
        # tools are mapped. All ordinary callers keep the empty result.
        if _retain_empty_plugin_providers and plugin_maximum is not None:
            return provider
        return None
    from tldw_chatbook.Agents.plain_chat_tool_surface import offered_mcp_tool_count

    publish(
        offered_mcp_tool_count(self._agent_bridge, provider, session_id),
        provider.not_connected_count,
    )
    return provider


async def _compose_agent_request_providers(
    self,
    *,
    session_id: str,
    project_selection: native_host.ProjectInstructionBindingSelection | None,
    project_authority_guard: native_host.Callable[[], bool] | None,
    turn_context: native_host.ConsoleTurnExecutionContext | None = None,
    publish_mcp_counts: bool = True,
    admitted_roots: native_host.Sequence[native_host.Any] | None = None,
) -> tuple[
    native_host.MCPToolProvider | None,
    "native_host.BuiltinToolGate",
    native_host.LocalToolProvider | None,
    native_host.Callable[[list["native_host.ToolCall"]], dict[str, str]] | None,
]:
    """Compose providers shared by live and preview.

    ``publish_mcp_counts`` is true for live dispatch and false for the
    disposable Context preflight, whose composition must not mutate the
    app-level inspector counters.
    """
    # Workspace assistant defaults (Task 7): the workspace's named
    # permission profile rides the turn context into the MCP provider's
    # profile seam. A None/absent context, or the "default" profile,
    # omits the keyword entirely -- the call shape (and behavior) stays
    # byte-identical for every legacy caller and signature-exact double.
    profile_id = (
        turn_context.tool_policy_profile_id if turn_context is not None else "default"
    )
    mcp_profile_kwargs: dict[str, native_host.Any] = {}
    if profile_id != "default":
        mcp_profile_kwargs["profile_id_provider"] = lambda: profile_id
    # Persona require_confirmation floor (final review): thread THIS
    # run's parsed persona policy into the MCP provider's invoke-time
    # gates, mirroring the local provider's `_resolve_state` layering.
    # Parsed once here; the provider reads it per gate. The kwarg is
    # only added when rules exist, so every no-persona run keeps the
    # exact pre-feature call shape.
    if turn_context is not None and turn_context.persona_policy_rules:
        # Lazy import (boot budget, ADR-097): per-run path only.
        from tldw_chatbook.Agents.persona_policy import (
            parse_persona_policy_from_rules,
        )

        parsed_persona_policy = parse_persona_policy_from_rules(
            turn_context.persona_policy_rules
        )
        if parsed_persona_policy.kinds:
            mcp_profile_kwargs["persona_policy_provider"] = lambda: (
                parsed_persona_policy
            )
    if (
        turn_context is not None
        and turn_context.skill_context_maximum.get("plugin_run_id")
        and (
            turn_context.skill_context_maximum.get("_plugin_requirement_ceiling")
            or any(
                row.get("plugin_owned")
                for row in turn_context.skill_context_maximum.get(
                    "available_skills", ()
                )
            )
        )
    ):
        mcp_profile_kwargs["plugin_maximum"] = turn_context.skill_context_maximum
    deferred = (
        turn_context is not None
        and turn_context.mcp_definition_capture == "composition"
    )
    if deferred and not native_host._stock_console_tool_composition_current(self):
        raise PermissionError("mcp_catalog_source_changed")
    from .console_run_start_tools import capture_run_tool_read

    local_kill_switch = capture_run_tool_read(
        self,
        session_id,
        turn_context,
        project_authority_guard,
        plugins="plugin_maximum" in mcp_profile_kwargs,
    )
    mcp_provider = await self._compose_mcp_provider(
        session_id,
        publish_counts=publish_mcp_counts,
        local_kill_switch=local_kill_switch,
        **mcp_profile_kwargs,
        maximum_tool_ids=(
            turn_context.mcp_tool_maximum if turn_context is not None else None
        ),
        maximum_definition_hashes=(
            turn_context.mcp_definition_maximum
            if turn_context is not None and not deferred
            else None
        ),
        **({"_freeze_definition_maximum_on_compose": True} if deferred else {}),
    )
    builtin_gate = native_host.build_builtin_gate(
        getattr(self.app, "unified_mcp_service", None), profile_id=profile_id
    )
    if admitted_roots is None:
        session = next(
            (item for item in self.store.sessions() if item.id == session_id), None
        )
        admitted_roots = native_host.capture_run_admitted_workspace_roots(
            session=session,
            registry=getattr(self.app, "workspace_registry_service", None),
            project_selection=project_selection,
            project_authority_guard=project_authority_guard,
        )
    index_service, index_scopes = await native_host.asyncio.to_thread(
        self._prepare_project_index_scopes, tuple(admitted_roots)
    )
    index_kwargs = (
        {
            "project_index_service": index_service,
            "project_index_scopes": index_scopes,
        }
        if index_scopes
        else {}
    )
    local_provider, local_review_hook = self._compose_local_provider(
        session_id=session_id,
        turn_context=turn_context,
        project_root=(project_selection.root if project_selection else None),
        allow_write=(project_selection.allow_write if project_selection else True),
        project_root_identity=(
            project_selection.root_identity if project_selection else None
        ),
        project_root_guard=project_authority_guard,
        admitted_roots=admitted_roots,
        local_kill_switch=local_kill_switch,
        **index_kwargs,
    )
    return mcp_provider, builtin_gate, local_provider, local_review_hook


def _compose_local_provider(
    self,
    session_id: str | None = None,
    turn_context: native_host.ConsoleTurnExecutionContext | None = None,
    *,
    project_root: native_host.Path | None = None,
    allow_write: bool = True,
    project_root_identity: tuple[tuple[str, int, int, int], ...] | None = None,
    project_root_guard: native_host.Callable[[], bool] | None = None,
    admitted_roots: native_host.Sequence[native_host.Any] | None = None,
    local_kill_switch: "native_host.LocalKillSwitchRead | None" = None,
    project_index_service: native_host.Any = None,
    project_index_scopes: native_host.Mapping[str, native_host.Any] | None = None,
) -> tuple[
    native_host.LocalToolProvider | None,
    native_host.Callable[[list["native_host.ToolCall"]], dict[str, str]] | None,
]:
    """Build THIS run's LocalToolProvider + review hook, or ``(None, None)``.

    Sync, unlike ``_compose_mcp_provider``: ``LocalToolProvider``'s
    specs are static (no async catalog composition), so there is no
    main-loop I/O constraint -- but it is still called from
    ``_run_agent_reply`` alongside the MCP composition, before the
    bridge is dispatched onto ``asyncio.to_thread``.

    Returns ``(None, None)`` whenever local tools should not be
    offered this run:

    - ``[console] local_tools_enabled`` is false (the master flag);
    - no ``unified_mcp_service`` on the app -- ADR-300 reuses the MCP
      permission store under the synthetic ``local:__local__`` server
      key, so without the service there is no state source, no
      session-approval cache, and no persistence path;
    - the kill switch is on, or reading it raised (fail closed --
      mirrors ``_compose_mcp_provider``).

    Wiring (all straight MCP parity): ``resolve_state`` is the
    service's own ``gate_tool_test`` -- the exact payload source the
    MCP gate uses (fresh ``store.load()`` + ``resolve_effective_
    state`` per call, fail-closed to "ask" with no store);
    ``persist_approval`` routes ``approve_session`` to the in-memory
    session cache and ``always_allow`` to ``set_tool_state``, which
    fingerprints ``definition_hash(description, input_schema)``
    itself (the rug-pull guard, spec §3.2); ``record_decision`` is
    the same ``record_tool_decision`` audit path the MCP provider
    uses (``initiator="agent"``), recording local refusals as
    "denied"/"denied-timeout" under the ``local:__local__`` server
    key.

    Todo wiring (TASK-13216): when ``session_id`` resolves to a live
    session, the provider is handed that session's stable task store
    plus an ``on_todo_change`` callback that renders defensive task
    snapshots via ``ConsoleAgentBridge.append_todo_marker``. Without a
    session (or without a bridge), the provider stays context-free and
    no task specs are registered.

    Args:
        session_id: THIS run's owning session id, bound into the
            approval bridge exactly as ``_compose_mcp_provider`` and
            the built-in review hook bind it, so cancellation checks
            and card parking scope to this run's own session.

    Returns:
        ``(provider, review_tool_calls)`` when eligible -- a
        ``LocalToolProvider`` confined to the resolved workspace root
        and this run's ``build_local_review_hook``-built batch-review
        closure; ``(None, None)`` otherwise.
    """
    from .console_ordinary_tree import current_ordinary_tree

    ordinary = current_ordinary_tree(self)
    if ordinary is not None:
        ordinary.check_local_sources()

    # task-3240 fix round 1 (Critical 2): get_cli_setting returns the RAW
    # TOML value -- a hand-typed quoted "false" is a non-empty string
    # and therefore truthy under bare `not` truthiness, which would
    # COMPOSE the entire local tool group while the MCP-hub gate
    # checkbox (Agents/builtin_tool_gate.py's all_tool_gates()) and
    # mcp_workbench.py's own [console] read both show it OFF. Note
    # this is NOT normalized by load_settings() the way some [console]
    # keys are -- that normalization lives in load_settings(), never
    # in get_cli_setting() itself, which reads the raw bootstrap
    # config. coerce_bool_setting is the arc's sixth such site.
    local_tools_enabled = (
        turn_context.tool_configuration.get(
            "local_tools_enabled", native_host.LOCAL_TOOLS_DEFAULT_ENABLED
        )
        if turn_context is not None
        else native_host.get_cli_setting(
            "console", "local_tools_enabled", native_host.LOCAL_TOOLS_DEFAULT_ENABLED
        )
    )
    if not native_host.coerce_bool_setting(
        local_tools_enabled, native_host.LOCAL_TOOLS_DEFAULT_ENABLED
    ):
        return None, None
    service = getattr(self.app, "unified_mcp_service", None)
    if service is None:
        return None, None
    from .console_run_start_tools import local_tools_kill_switch

    # Fail closed: the run-start hop's read, else a read made here.
    if local_tools_kill_switch(service, local_kill_switch):
        return None, None

    profile_id = (
        turn_context.tool_policy_profile_id if turn_context is not None else "default"
    )
    profile_kwargs = {} if profile_id == "default" else {"profile_id": profile_id}

    bound_request_approvals = native_host.functools.partial(
        self.request_mcp_approvals, session_id=session_id
    )

    def _kill_switch() -> bool:
        # invoke()-time read, mirroring MCPToolProvider._kill_switch_
        # engaged: never raises, and a read failure must not block
        # execution (compose-time read above already failed closed).
        try:
            return bool(service.get_kill_switch())
        except Exception as exc:  # noqa: BLE001 -- invoke must never raise
            native_host.logger.warning(
                f"ConsoleChatController: get_kill_switch failed during local invoke: {exc}"
            )
            return False

    def _persist_approval(hub: "native_host.HubTool", decision: str) -> None:
        if decision == "approve_session":
            service.approve_for_session(hub.server_key, hub.name, **profile_kwargs)
        elif decision == "always_allow":
            # set_tool_state computes definition_hash(hub.description,
            # hub.input_schema) itself -- required for the rug-pull guard.
            service.set_tool_state(
                hub.server_key,
                hub.name,
                "allow",
                tool=hub,
                **profile_kwargs,
            )

    def _record_decision(hub: "native_host.HubTool", decision: str) -> None:
        # Same audit path MCPToolProvider._record_decision_safe uses;
        # the provider guards the call (never-raise seam).
        service.record_tool_decision(
            hub.server_key,
            hub.name,
            decision=decision,
            initiator="agent",
        )

    if project_root is None or native_host.is_remote(project_root):
        # Phase 4a: a REMOTE working-folder selection contributes its
        # authority through ``admitted_roots`` (the per-alias remote
        # executor), never as the provider's base root — the base
        # root is a laptop-path construction input (standalone
        # executor, spill home), and a RemoteRoot there is the
        # Phase 3a laptop-disk boundary. The run's private scratch
        # stands in, so a BLOCKED remote selection proceeds on
        # scratch/local (the degraded posture) instead of crashing.
        snapshot = turn_context.scratch_space if turn_context is not None else None
        if snapshot is None:
            return None, None
        root = snapshot.root
        authority_scope = native_host.functools.partial(
            self._scratch_spaces.lease, snapshot
        )
    else:
        root = project_root
        authority_scope = None
    from .console_preparation_reads import current_planning_preparation

    planning = current_planning_preparation(self)
    watchlists_service = watchlists_command_service = None
    if planning is None:
        subscriptions_db = getattr(self.app, "subscriptions_db", None)
        watchlists_service = native_host.WatchlistsToolService(
            db_resolver=lambda: subscriptions_db,
            # Owner-module loader (TASK-18609): constructing the store here
            # violated the runtime-policy ownership boundary.
            runtime_source_loader=native_host.load_default_runtime_source_state,
        )
        local_watchlists_service = getattr(self.app, "local_watchlists_service", None)
        watchlist_bundle_service = getattr(self.app, "watchlist_bundle_service", None)
        watchlists_coordinator = getattr(
            self.app,
            "watchlists_operation_coordinator",
            None,
        )

        def _create_collection(**kwargs: native_host.Any) -> native_host.Any:
            if watchlist_bundle_service is None:
                raise RuntimeError("Watchlists collection service unavailable")
            return watchlist_bundle_service.create_with_sources(**kwargs)

        def _update_collection_sources(**kwargs: native_host.Any) -> native_host.Any:
            if watchlist_bundle_service is None:
                raise RuntimeError("Watchlists collection service unavailable")
            return watchlist_bundle_service.update_sources(**kwargs)

        def _create_sources_batch(
            rows: list[native_host.Mapping[str, native_host.Any]],
        ) -> native_host.Any:
            if local_watchlists_service is None:
                raise RuntimeError("Watchlists source service unavailable")
            return local_watchlists_service.create_sources_exact_batch_sync(rows)

        watchlists_command_service = getattr(
            self.app, "watchlists_command_service", None
        )
        if watchlists_command_service is None:
            watchlists_command_service = native_host.WatchlistsCommandService(
                runtime_source_loader=native_host.load_default_runtime_source_state,
                create_sources_batch=_create_sources_batch,
                create_collection=_create_collection,
                update_collection_sources=_update_collection_sources,
                accept_source_checks=(
                    watchlists_coordinator.submit_checks
                    if watchlists_coordinator is not None
                    else None
                ),
                accept_briefing=(
                    watchlists_coordinator.submit_briefing
                    if watchlists_coordinator is not None
                    else None
                ),
                resolve_collection_sources=(
                    watchlist_bundle_service.list_sources
                    if watchlist_bundle_service is not None
                    else None
                ),
            )

    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

    # Workspace assistant defaults (Task 7): THIS run's posture, from the
    # turn context. The resolve-state closure layers, narrowing-only:
    # (1) the workspace's named permission profile selects WHICH profile
    # the fresh per-call gate resolves under (the Task 6 `_profile_kwargs`
    # convention -- the default profile keeps calling `gate_tool_test`
    # exactly as before, so signature-exact service doubles stay valid);
    # (2) the persona floor then lowers allow->ask for tools whose rules
    # demand confirmation. No path here can raise a state or add a tool.
    # Lazy import (boot budget, ADR-097): per-run provider gate only.
    from tldw_chatbook.Agents.persona_policy import (
        parse_persona_policy_from_rules,
        persona_floor_state,
    )

    persona_policy = parse_persona_policy_from_rules(
        turn_context.persona_policy_rules if turn_context is not None else None
    )

    def _resolve_state(hub: "native_host.HubTool") -> native_host.Any:
        if profile_id == "default":
            state = service.gate_tool_test(hub)
        else:
            state = service.gate_tool_test_for_profile(hub, profile_id)
        return persona_floor_state(state, persona_policy, hub.name)

    from tldw_chatbook.Chat.approval_presentation import profile_authority

    provider = LocalToolProvider(
        presentation_authority=profile_authority("local", profile_id, str(root), "call"),
        workspace_root=root,
        allow_write=allow_write,
        authority_scope=authority_scope,
        result_redaction_root=(root if project_root is None else None),
        root_guard=(
            project_root_guard
            if project_root_guard is not None
            else (
                lambda: (
                    native_host._project_root_identity_matches(
                        root, project_root_identity
                    )
                    if project_root_identity is not None
                    else True
                )
            )
        ),
        resolve_state=_resolve_state,
        kill_switch=_kill_switch,
        approval_callback=bound_request_approvals,
        is_session_approved=lambda hub: service.is_session_approved(
            hub.server_key, hub.name, **profile_kwargs
        ),
        persist_approval=_persist_approval,
        record_decision=_record_decision,
        watchlists_service=watchlists_service,
        watchlists_command_service=watchlists_command_service,
        admitted_roots=admitted_roots,
        project_index_service=project_index_service,
        project_index_scopes=project_index_scopes,
        **({} if planning is not None else self._todo_wiring(session_id)),
        **({} if planning is not None else self._ask_user_wiring(session_id)),
        **({} if planning is not None else self._character_wiring(session_id)),
    )
    return provider, native_host.build_local_review_hook(
        provider, bound_request_approvals
    )


def _compose_virtual_cli_provider(
    self,
    session_id: str | None = None,
    turn_context: native_host.ConsoleTurnExecutionContext | None = None,
    *,
    project_root: native_host.Path | None = None,
    project_root_identity: tuple[tuple[str, int, int, int], ...] | None = None,
    project_root_guard: native_host.Callable[[], bool] | None = None,
    admitted_roots: native_host.Sequence[native_host.Any] | None = None,
) -> tuple[
    native_host.VirtualCliProvider | None,
    native_host.Callable[[list["native_host.ToolCall"], str], dict[str, str]] | None,
]:
    """Compose the read-only virtual CLI over this run's local root."""
    configured = (
        turn_context.tool_configuration.get(
            "local_tools_enabled", native_host.LOCAL_TOOLS_DEFAULT_ENABLED
        )
        if turn_context is not None
        else native_host.get_cli_setting(
            "console", "local_tools_enabled", native_host.LOCAL_TOOLS_DEFAULT_ENABLED
        )
    )
    if not native_host.coerce_bool_setting(
        configured, native_host.LOCAL_TOOLS_DEFAULT_ENABLED
    ):
        return None, None
    if admitted_roots is not None and not admitted_roots:
        return None, None
    service = getattr(self.app, "unified_mcp_service", None)
    if service is None:
        return None, None
    try:
        if service.get_kill_switch():
            return None, None
    except Exception:  # noqa: BLE001 -- unreadable security state fails closed
        return None, None

    profile_id = (
        turn_context.tool_policy_profile_id if turn_context is not None else "default"
    )
    profile_kwargs = {} if profile_id == "default" else {"profile_id": profile_id}

    def resolve_state(hub: "native_host.HubTool") -> native_host.Any:
        if profile_id == "default":
            return service.gate_tool_test(hub)
        return service.gate_tool_test_for_profile(hub, profile_id)

    if project_root is None or native_host.is_remote(project_root):
        # Phase 4a: a REMOTE working-folder selection contributes its
        # authority through ``admitted_roots`` (the per-alias remote
        # executor), never as the provider's base root — the base
        # root is a laptop-path construction input (standalone
        # executor, spill home), and a RemoteRoot there is the
        # Phase 3a laptop-disk boundary. The run's private scratch
        # stands in, so a BLOCKED remote selection proceeds on
        # scratch/local (the degraded posture) instead of crashing.
        snapshot = turn_context.scratch_space if turn_context is not None else None
        if snapshot is None:
            return None, None
        root = snapshot.root
        authority_scope = native_host.functools.partial(
            self._scratch_spaces.lease, snapshot
        )
    else:
        root = project_root
        authority_scope = None

    root_guard = (
        project_root_guard
        if project_root_guard is not None
        else lambda: (
            native_host._project_root_identity_matches(root, project_root_identity)
            if project_root_identity is not None
            else True
        )
    )
    bound_request = native_host.functools.partial(
        self.request_mcp_approvals, session_id=session_id
    )

    def persist(hub: "native_host.HubTool", decision: str) -> None:
        if decision == "approve_session":
            service.approve_for_session(hub.server_key, hub.name, **profile_kwargs)
        elif decision == "always_allow":
            service.set_tool_state(
                hub.server_key,
                hub.name,
                "allow",
                tool=hub,
                **profile_kwargs,
            )

    def record(hub: "native_host.HubTool", decision: str) -> None:
        service.record_tool_decision(
            hub.server_key,
            hub.name,
            decision=decision,
            initiator="agent",
        )

    def persist_arg_rule(
        hub: "native_host.HubTool", args: native_host.Mapping[str, native_host.Any]
    ) -> None:
        # task-32281: same shape as `MCPToolProvider._apply_verdict()`'s
        # own "allow_matching" handling -- persist scoped to EXACTLY
        # the displayed arguments, never a whole-tool allow.
        service.add_tool_arg_rule(
            hub.server_key,
            hub.name,
            args=dict(args),
            tool=hub,
            **profile_kwargs,
        )

    def arg_rule_allows(
        hub: "native_host.HubTool", args: native_host.Mapping[str, native_host.Any]
    ) -> bool:
        return service.arg_rule_allows_call(hub, dict(args), **profile_kwargs)

    from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider

    from tldw_chatbook.Chat.approval_presentation import profile_authority

    provider = VirtualCliProvider(
        presentation_authority=profile_authority("virtual_cli", profile_id, str(root), "call"),
        workspace_root=root,
        resolve_state=resolve_state,
        local_tools_enabled=lambda: native_host.coerce_bool_setting(
            configured, native_host.LOCAL_TOOLS_DEFAULT_ENABLED
        ),
        kill_switch=lambda: bool(service.get_kill_switch()),
        approval_callback=bound_request,
        is_session_approved=lambda hub: service.is_session_approved(
            hub.server_key, hub.name, **profile_kwargs
        ),
        persist_approval=persist,
        persist_arg_rule=persist_arg_rule,
        arg_rule_allows=arg_rule_allows,
        record_decision=record,
        root_guard=root_guard,
        authority_scope=authority_scope,
        result_redaction_root=(root if project_root is None else None),
        admitted_roots=admitted_roots,
    )
    return provider, native_host.build_virtual_cli_review_hook(provider, bound_request)


def _compose_raw_shell_provider(
    self,
    session_id: str | None = None,
    turn_context: native_host.ConsoleTurnExecutionContext | None = None,
    *,
    project_root: native_host.Path | None = None,
) -> tuple[
    native_host.RawShellToolProvider | None,
    native_host.Callable[[list["native_host.ToolCall"], str], dict[str, str]] | None,
]:
    """Compose model raw shell only while every live safety gate is open."""
    configured = (
        turn_context.tool_configuration.get(
            "local_tools_enabled", native_host.LOCAL_TOOLS_DEFAULT_ENABLED
        )
        if turn_context is not None
        else native_host.get_cli_setting(
            "console", "local_tools_enabled", native_host.LOCAL_TOOLS_DEFAULT_ENABLED
        )
    )
    local_enabled = native_host.coerce_bool_setting(
        configured, native_host.LOCAL_TOOLS_DEFAULT_ENABLED
    )
    if not local_enabled or not session_id:
        return None, None
    service = getattr(self.app, "unified_mcp_service", None)
    runtime = getattr(self.app, "raw_cli_runtime", None)
    if service is None or runtime is None:
        return None, None
    try:
        if (
            runtime.permitted is not True
            or runtime.armed is not True
            or service.get_kill_switch()
        ):
            return None, None
    except Exception:
        return None, None

    profile_id = (
        turn_context.tool_policy_profile_id if turn_context is not None else "default"
    )

    def resolve_state(hub: "native_host.HubTool") -> native_host.Any:
        if profile_id == "default":
            return service.gate_tool_test(hub)
        return service.gate_tool_test_for_profile(hub, profile_id)

    if project_root is not None and not native_host.is_remote(project_root):
        initial_directory = project_root
    else:
        # Phase 4a: raw shell is local-only (ADR-094); a remote
        # working-folder selection never becomes its cwd — the run's
        # private scratch stands in (the remote authority rides the
        # fs_* admitted roots, not the shell).
        snapshot = turn_context.scratch_space if turn_context is not None else None
        if snapshot is None:
            return None, None
        initial_directory = snapshot.root

    def kill_switch() -> bool:
        try:
            return bool(service.get_kill_switch())
        except Exception:
            return True

    bound_request = native_host.functools.partial(
        self.request_mcp_approvals, session_id=session_id
    )
    agent_bridge = getattr(self, "_agent_bridge", None)
    from tldw_chatbook.Agents.raw_shell_tool_provider import (
        RawShellToolProvider,
    )

    try:
        project_authority = getattr(turn_context, "project_authority", None)
        workspace_id = (
            project_authority.workspace_id
            if project_authority is not None
            else self.store.session_workspace_id(session_id)
        )
    except (AttributeError, KeyError, ValueError):
        workspace_id = None

    def protected_roots() -> tuple[native_host.Path, ...]:
        from tldw_chatbook.DB.base_db import operation_owned_connection
        from tldw_chatbook.Workspaces.models import (
            DEFAULT_WORKSPACE_ID,
            RuntimeBindingKind,
        )

        if workspace_id in (
            native_host.CONSOLE_GLOBAL_WORKSPACE_ID,
            DEFAULT_WORKSPACE_ID,
        ):
            return ()
        if not workspace_id:
            raise ValueError("raw shell workspace unavailable")
        registry = self.app.workspace_registry_service
        if registry is None:
            raise ValueError("workspace registry unavailable")
        with operation_owned_connection(registry.db):
            bindings = registry.list_runtime_bindings(workspace_id)
        return tuple(
            native_host.Path(binding.locator)
            for binding in bindings
            if binding.binding_kind == RuntimeBindingKind.LOCAL_FILESYSTEM
        )

    provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id=session_id,
        initial_directory=lambda: initial_directory,
        protected_roots=protected_roots,
        resolve_state=resolve_state,
        local_tools_enabled=lambda: local_enabled,
        kill_switch=kill_switch,
        progress_sink=(
            agent_bridge.raw_shell_progress_sink if agent_bridge is not None else None
        ),
    )
    providers = getattr(self, "_raw_shell_providers", None)
    if providers is None:
        providers = native_host.weakref.WeakSet()
        self._raw_shell_providers = providers
    providers.add(provider)
    runtime.set_model_authority_revoker(self.revoke_raw_shell_authority)
    if not provider.catalog_enabled():
        return None, None
    return provider, native_host.build_raw_shell_review_hook(provider, bound_request)
