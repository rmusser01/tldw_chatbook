class PlainControlGateway(MountedRecoveryGateway):
    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.stream_calls += 1
        self.tools_seen.append(tools)
        self.messages_seen.append(messages)
        yield "Plain native control reply."

@asynccontextmanager
async def mounted_plain_control(
    recovery_stores,
    tmp_path,
    monkeypatch,
    *,
    size=(160, 44),
    entry="button",
    prepare=None,
    search_arguments=None,
    direct=True,
    gateway_override=None,
):
    services, target_id, db = recovery_stores
    message_id = db.add_message(
        {
            "conversation_id": target_id,
            "sender": "user",
            "content": "Original retained history",
        }
    )
    db.set_conversation_active_leaf(target_id, message_id)
    if prepare is not None:
        prepare(services, target_id, db)
    app = _build_test_app()
    _configure_native_ready_console(app)

    async def empty_skill_context(**_kwargs):
        return {"available_skills": []}

    # Match the empty skill inventory in the captured execution configuration.
    # Archive tests must not start unrelated profile-backed skill discovery.
    app.local_skills_service = app.skills_scope_service = SimpleNamespace(
        get_context=empty_skill_context
    )
    app.chachanotes_db = db
    app.workspace_registry_service = services.workspace_registry_service
    app.local_chat_conversation_service = services.local_chat_conversation_service
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=app.local_chat_conversation_service, server_service=None
    )
    persistence = ChatPersistenceService(
        db, workspace_registry=app.workspace_registry_service
    )
    source_id = db.add_conversation({"title": "Source"})
    persistence.console_library_policy_repository.insert(
        source_id,
        replace(
            _allowed_candidate(allowed=True), auto_retrieve=ConsoleAutoRetrieve.NEVER
        ),
    )
    store = ConsoleChatStore(persistence=persistence)
    source = store.restore_persisted_session(
        title="Source",
        workspace_id=None,
        persisted_conversation_id=source_id,
        all_nodes=(),
    )
    await store.hydrate_session_library_policy(source.id)
    gateway = PlainControlGateway()
    gateway.search_arguments = search_arguments
    gateway.destination_snapshots = []
    gateway.source = source
    runs_db = AgentRunsDB(tmp_path / "runs.sqlite", "recovery-ui")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs_db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: True,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        agent_bridge=bridge,
        library_provider_factory=lambda context: (
            LibraryToolProvider(
                LocalLibraryToolService(
                    conversation_service=app.local_chat_conversation_service
                )
            )
            if direct
            else LibraryRagToolProvider(None)
        ),
    )
    owner = ensure_console_runtime(app)
    owner.set_chat_store(store)
    owner.set_provider_gateway(gateway)
    owner.set_agent_bridge(bridge)
    owner.set_chat_controller(controller)
    recovery = owner.ensure_archive_recovery()
    controller._archive_recovery_runtime = recovery
    host = ConsoleHarness(app)
    # The unmounted runtime owner schedules UI callbacks on the mounted host.
    monkeypatch.setattr(app, "call_later", host.call_later)
    turn = None
    try:
        async with host.run_test(size=size) as pilot:
            await wait_until(
                pilot, lambda: bool(host.screen.query("#console-native-transcript"))
            )
            console = host.screen
            console._console_agent_bridge = bridge

            def configuration(sid):
                captured = _execution_configuration(
                    sid,
                    direct=direct,
                    native_tools=True,
                    agent_runtime=True,
                    scratch_space=controller._scratch_spaces.snapshot(sid),
                )
                return replace(
                    captured,
                    provider_selection=replace(
                        captured.provider_selection, explicit_model="gpt-4o"
                    ),
                )

            monkeypatch.setattr(
                console._session, "_build_console_turn_execution_context", configuration
            )
            controller.app = host
            await console._sync_native_console_chat_ui()
            await wait_until(
                pilot,
                lambda: console._archive_recovery_view().is_answerable(source.id),
                lambda: (
                    console.is_mounted,
                    console.is_current,
                    owner.view is console,
                    store.active_session_id,
                    bool(console.query("#recovery-confirm")),
                ),
            )
            from tldw_chatbook.UI.Console_Modules.wiring import (
                _admit_console_turn_to_runtime,
            )

            turn_id = _admit_console_turn_to_runtime(
                console, "Reply with a short plain response", source.id
            )
            turn = asyncio.create_task(owner.wait_for_turn(turn_id))
            await wait_until(
                pilot,
                turn.done,
                lambda: (
                    controller.run_state_for(source.id),
                    gateway.tools_seen,
                    gateway.messages_seen,
                ),
            )
            await turn
            assert gateway.stream_calls == 1
            assert controller._agent_bridge is bridge and owner.agent_bridge is bridge
            yield SimpleNamespace(
                app=app,
                db=db,
                target_id=target_id,
                message_id=message_id,
                original_leaf_id=message_id,
                controller=controller,
                source=source,
                store=store,
                gateway=gateway,
                owner=owner,
                recovery=recovery,
                host=host,
                pilot=pilot,
                console=console,
                turn=turn,
            )
    finally:
        try:
            await dispose_console_runtime(app)
        finally:
            if turn is not None:
                if not turn.done():
                    turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
            runs_db.close()

async def nonarchive_native_control(recovery_stores, tmp_path, monkeypatch, entry, size):
    async with mounted_plain_control(
        recovery_stores, tmp_path, monkeypatch, entry=entry, size=size
    ) as rig:
        assert any(
            "Plain native control reply." in message.content
            for message in rig.store.messages_for_session(rig.source.id)
        )
