"""Application-loop recovery adapters use real archive storage."""

import asyncio
import json
import threading

import pytest

from Tests.Chat.test_archive_recovery_operations import (
    recovery_stores as recovery_stores,  # noqa: PLC0414 - pytest fixture export
)
from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveRecoveryError
from tldw_chatbook.Chat.archive_recovery.runtime import ArchiveRecoveryRuntime


@pytest.mark.asyncio
async def test_search_publishes_typed_app_record_and_manual_refs_stay_manual(
    recovery_stores,
):
    app, cid, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    presentations = []
    runtime.present_results = presentations.append
    result = await runtime.search("source", {}, model_visible=False)
    assert result["items"][0]["conversation_id"] == cid
    assert (
        len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
        <= 32768
    )
    record = presentations[0]
    assert record.session_id == "source"
    assert record.request.query == ""
    assert record.projection == result
    with pytest.raises(ArchiveRecoveryError):
        runtime.results.resolve(
            "source", result["items"][0]["result_ref"], for_agent=True
        )
    await runtime.close_session("source")
    with pytest.raises(ArchiveRecoveryError):
        await runtime.search("source", {}, model_visible=False)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_unknown_fields_and_foreign_cursor_refused(recovery_stores):
    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    for args in ({"conversation_id": "raw"}, {"cursor": "invented"}):
        with pytest.raises(ArchiveRecoveryError):
            await runtime.search("source", args, model_visible=True)
    await runtime.shutdown()


from dataclasses import replace
from types import SimpleNamespace

from Tests.Chat.test_console_turn_library_authority import (
    _allowed_candidate,
    _execution_configuration,
    _RecordingGateway,
)
from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.UI.Console_Modules.wiring import _admit_console_turn_to_runtime


class RecoveryGateway(_RecordingGateway):
    def __init__(self):
        super().__init__([])
        self.tools_seen = []
        self.messages_seen = []
        self.result_ref = None
        self.before_search = lambda: None

    async def stream_chat(self, _resolution, messages, tools=None, **kwargs):
        self.tools_seen.append(tools)
        self.messages_seen.append(messages)
        index = self.stream_calls % 4
        self.stream_calls += 1
        if index == 0:
            name, arguments = (
                "load_tools",
                {
                    "ids": [
                        "archive_recovery:search_archived_conversations",
                        "archive_recovery:request_conversation_restore",
                    ]
                },
            )
        elif index == 1:
            self.before_search()
            name, arguments = "search_archived_conversations", {}
        elif index == 2:
            for message in messages:
                if message.get("role") == "tool":
                    try:
                        payload = json.loads(message["content"])
                        self.result_ref = payload["items"][0]["result_ref"]
                    except (KeyError, ValueError, TypeError):
                        pass
            if self.result_ref is None:
                yield "No recovery available."
                return
            name, arguments = (
                "request_conversation_restore",
                {"result_ref": self.result_ref, "scope": "chat_and_workspace"},
            )
        else:
            yield "Recovery finished."
            return
        yield ProviderToolCalls(
            [
                {
                    "id": f"native-recovery-{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "direct,allowed,restriction",
    [
        (True, True, None),
        (False, True, None),
        (True, True, "received"),
        (False, True, "received"),
        (True, True, "received_unproven"),
        (True, True, "cold_view"),
        (False, True, "cold_view"),
        (True, True, "received_cold_view"),
        (True, True, "prepare_error"),
        (True, False, None),
        (True, True, "kill"),
        (True, True, "persona"),
        (True, True, "legacy"),
        (True, True, "ask_allow"),
        (True, True, "ask_deny"),
        (True, True, "ask_session"),
        (True, True, "review_error"),
        (True, True, "queued_block"),
        (True, True, "capacity"),
        (True, True, "ask_missing"),
        (True, True, "ask_timeout"),
        (True, True, "ask_no_view"),
        (True, True, "agent_wake"),
    ],
)
async def test_real_console_native_dispatch_waits_for_exact_confirmation(
    recovery_stores, tmp_path, direct, allowed, restriction
):
    app, target_id, db = recovery_stores
    source_id = db.add_conversation({"title": "Source"})
    persistence = ChatPersistenceService(db)
    persistence.console_library_policy_repository.insert(
        source_id,
        replace(
            _allowed_candidate(allowed=allowed), auto_retrieve=ConsoleAutoRetrieve.NEVER
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
    gateway = RecoveryGateway()
    runs_db = AgentRunsDB(tmp_path / "runs.sqlite", "test")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs_db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: True,
    )
    capacity_hold = None
    if restriction == "capacity":
        from tldw_chatbook.Agents.agent_models import WorkOrigin
        from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

        capacity = RuntimeCapacity(max_tool_workers=1)
        bridge.runtime_capacity = capacity
        capacity_owner = capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="occupied"
        )
        capacity_hold = capacity_owner.reserve_tool()
    owner = ConsoleRuntime(app)
    owner.set_chat_store(store)
    recovery = owner.ensure_archive_recovery()
    presented = []
    confirmation_ready = asyncio.Event()
    cards = []
    recovery.is_answerable = lambda sid: sid == source.id
    recovery.present_results = presented.append

    def present(card):
        cards.append(card)
        confirmation_ready.set()
        return True

    recovery.present_confirmation = present
    prepared = []
    if restriction in {"cold_view", "received_cold_view", "prepare_error"}:
        view_ready = False
        recovery.is_answerable = lambda sid: view_ready and sid == source.id

        async def prepare_view():
            nonlocal view_ready
            assert owner.has_custodied_turns(source.id)
            prepared.append(source.id)
            await asyncio.sleep(0)
            if restriction == "prepare_error":
                raise RuntimeError("View unavailable")
            view_ready = True

        recovery.prepare_view = prepare_view
    native_keys = []
    original_request = recovery.coordinator.request

    async def record_request(*args, **kwargs):
        native_keys.append(kwargs.get("call_key"))
        return await original_request(*args, **kwargs)

    recovery.coordinator.request = record_request
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
            if context.library_authority.direct_library_tools
            else LibraryRagToolProvider(None)
        ),
    )
    owner.set_chat_controller(controller)
    controller._archive_recovery_runtime = recovery
    approval_ready = asyncio.Event()
    approval_payloads = []
    loop = asyncio.get_running_loop()
    main_thread = threading.get_ident()

    def marshal(callback, *args):
        if threading.get_ident() == main_thread:
            return callback(*args)

        async def project():
            return callback(*args)

        return asyncio.run_coroutine_threadsafe(project(), loop).result()

    app.call_from_thread = marshal

    def pending(payload):
        if payload:
            approval_payloads.append(payload)
            approval_ready.set()

    from Tests.Chat.test_console_runtime_lifetime import _View

    view = _View(
        {
            "set_pending_approval": pending,
            "present_confirmation": present,
            "is_answerable": recovery.is_answerable,
            "prepare_view": recovery.prepare_view,
            "present_results": presented.append,
        }
    )
    attached = owner.attach_view(view)
    owner.finish_view_reconciliation(view, attached)
    if restriction == "ask_no_view":
        gateway.before_search = lambda: owner.detach_view(view, attached)
    if restriction == "review_error":

        def fail_review(*args, **kwargs):
            raise RuntimeError("injected review fault")

        controller.request_mcp_approvals = fail_review
    if restriction == "ask_missing":
        controller.request_mcp_approvals = lambda *_a, **_kw: {}
    if restriction == "ask_timeout":

        def timeout_review(*args, **kwargs):
            raise TimeoutError("injected review timeout")

        controller.request_mcp_approvals = timeout_review
    if restriction == "kill":
        controller._console_tool_kill_switch_reader = lambda: lambda: True
    ask = restriction in {
        "ask_allow",
        "ask_deny",
        "ask_session",
        "review_error",
        "ask_missing",
        "ask_timeout",
        "ask_no_view",
    }

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
            persona_policy_rules=(
                {"rule_kind": "mcp_tool", "rule_name": "*", "allowed": False},
            )
            if restriction == "persona"
            else (
                {"rule_kind": "mcp_tool", "rule_name": "*", "allowed": True},
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "search_archived_conversations",
                    "allowed": True,
                    "require_confirmation": True,
                },
            )
            if ask
            else (),
        )

    screen = SimpleNamespace(
        _ensure_console_chat_store=lambda: store,
        _session=SimpleNamespace(_build_console_turn_execution_context=configuration),
        _console_runtime=lambda: owner,
    )
    if restriction == "agent_wake":
        from Tests.Chat.test_console_fleet_wake import (
            _drain,
            _survivor,
            _terminal_subagent_run,
        )
        from tldw_chatbook.Agents.agent_models import WorkOrigin
        from tldw_chatbook.Chat.conversation_local_marks_service import (
            ConversationLocalMarksService,
        )

        app.conversation_local_marks_service = ConversationLocalMarksService(db)
        controller.fleet_wake.wire(app=app)
        proof = recovery.issue_user_proof(source.id)
        # Try carrying a genuine UI-issued proof into the real unattended
        # producer; admission must still honor the executed wake origin.
        controller.resolve_runtime_turn_configuration_snapshot = lambda sid: replace(
            configuration(sid),
            recovery_user_origin=proof,
        )
        wake_dispatch = []
        run_reply = bridge.run_reply

        def observe_run_reply(**kwargs):
            wake_dispatch.append(kwargs)
            return run_reply(**kwargs)

        bridge.run_reply = observe_run_reply
        wake_turn_id = asyncio.get_running_loop().create_future()

        def submit_wake(notice, **kwargs):
            turn_id = owner._submit_fleet_wake(notice, **kwargs)
            assert (
                owner._turn_custody[turn_id].request.configuration.recovery_user_origin
                is proof
            )
            assert recovery._accepted_proofs[id(proof)][1] == turn_id
            wake_turn_id.set_result(turn_id)
            return turn_id

        controller.fleet_wake.bind_runtime_submitter(submit_wake)
        _parent, child = _terminal_subagent_run(runs_db, source_id)
        controller.fleet_wake.on_fleet_drained(
            _drain(source_id, _survivor(child, session_id=source.id)),
        )

        async def await_wake():
            return await owner.wait_for_turn(await wake_turn_id)

        turn = asyncio.create_task(await_wake())
    elif restriction in {"received", "received_unproven", "received_cold_view"}:
        from tldw_chatbook.Chat.console_configuration_preparation import (
            ConsoleTurnCaptureSelection,
        )
        from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent

        async def capture(sid, **_kwargs):
            return configuration(sid)

        controller.capture_turn_configuration_snapshot = capture
        store.set_session_draft(source.id, "Find the archived chat")
        captured = configuration(source.id)
        launch, revision, _notice = owner.snapshot_console_staged_evidence()
        intent = ConsoleReceivedTurnIntent(
            turn_id="received-recovery",
            session_id=source.id,
            inputs=store.session_input_snapshot(source.id),
            selection=ConsoleTurnCaptureSelection(
                provider_selection=captured.provider_selection,
                presentation_context=captured.presentation_context,
                rag_defaults={},
                tool_configuration={"agent_runtime_enabled": True},
                skill_workspace_id=None,
                project_bindings_eligible=False,
                agent_runtime_enabled=True,
            ),
            staged_evidence_launch=launch,
            staged_evidence_revision=revision,
            view_attachment_generation=attached,
            recovery_user_origin=(
                recovery.issue_user_proof(source.id)
                if restriction != "received_unproven"
                else None
            ),
        )
        turn_id = owner.accept_received_intent(intent)
        turn = asyncio.create_task(owner.wait_for_turn(turn_id))
    elif restriction == "legacy":
        turn = asyncio.create_task(
            controller.submit_draft(
                "Find the archived chat",
                session_id=source.id,
                configuration=configuration(source.id),
            )
        )
    else:
        turn_id = _admit_console_turn_to_runtime(
            screen, "Find the archived chat", source.id
        )
        turn = asyncio.create_task(owner.wait_for_turn(turn_id))
    try:
        if ask and restriction not in {
            "review_error",
            "ask_missing",
            "ask_timeout",
            "ask_no_view",
        }:
            ready = asyncio.create_task(approval_ready.wait())
            await asyncio.wait_for(
                asyncio.wait({turn, ready}, return_when=asyncio.FIRST_COMPLETED), 15
            )
            assert approval_payloads, str(gateway.messages_seen)
            payload = approval_payloads[0]
            assert payload["calls"][0]["options"] == ["approve_once", "deny"]
            assert not presented
            controller.resolve_pending_approval(
                {
                    payload["calls"][0]["call_id"]: {
                        "ask_allow": "approve_once",
                        "ask_deny": "deny",
                        "ask_session": "approve_session",
                    }[restriction]
                },
                round_id=payload["round_id"],
            )
        if allowed and restriction in {
            None,
            "ask_allow",
            "queued_block",
            "received",
            "cold_view",
            "received_cold_view",
        }:
            ready = asyncio.create_task(confirmation_ready.wait())
            await asyncio.wait_for(
                asyncio.wait({turn, ready}, return_when=asyncio.FIRST_COMPLETED), 15
            )
            assert cards, (
                f"No confirmation: tool rows={[m for messages in gateway.messages_seen for m in messages if m.get('role') == 'tool']}; turn={turn.result() if turn.done() else 'running'}"
            )
            assert not turn.done()
            assert presented[0].session_id == source.id
            assert presented[0].projection["items"][0]["conversation_id"] == target_id
            authority = next(iter(recovery._authorities.values()))
            assert runs_db.get_run(authority.run_id)["agent_kind"] == "primary"
            assert native_keys[0].run_id == authority.run_id
            assert native_keys[0].call_id == "native-recovery-2"
            assert native_keys[0].session_id == source.id
            if restriction == "queued_block":
                from tldw_chatbook.UI.Console_Modules.prompt_queue import (
                    ConsolePromptQueueUIController,
                )

                snapshot = controller.prompt_queue_registry.snapshot(source.id)
                queued = await controller.queue_prompt(
                    source.id,
                    text="queued",
                    expected_revision=snapshot.revision,
                    configuration=configuration(source.id),
                )
                assert queued.applied
                editor = ConsolePromptQueueUIController.__new__(
                    ConsolePromptQueueUIController
                )
                editor._chat_controller_accessor = lambda: controller
                editor._capture_configuration = configuration
                editor._edit_refusal = lambda _: ""
                snapshot = controller.prompt_queue_registry.snapshot(source.id)
                entry_id = snapshot.entries[0].entry_id
                edited = editor.edit_waiting(
                    source.id,
                    entry_id,
                    text="Queued recovery",
                    expected_revision=snapshot.revision,
                )
                assert edited.applied
                store.stage_session_library_policy(
                    source.id,
                    replace(
                        _allowed_candidate(allowed=False),
                        auto_retrieve=ConsoleAutoRetrieve.NEVER,
                    ),
                )
                await store.save_session_library_policy(source.id)
            outcome = await recovery.coordinator.decide(
                source.id, cards[0].request_id, allow=True, scope="chat_and_workspace"
            )
            assert outcome.chat_changed and outcome.workspace_changed
            await asyncio.wait_for(turn, 15)
            assert not recovery._authorities
            if restriction == "queued_block":
                assert len(presented) == 1
                assert "not permitted" in str(gateway.messages_seen[-1]).lower()
            else:
                assert "restored" in str(gateway.messages_seen[-1])
        else:
            await asyncio.wait_for(turn, 15)
            assert not cards and not presented
            if ask:
                assert (
                    "not approved" in str(gateway.messages_seen[-1]).lower()
                    or "review failed" in str(gateway.messages_seen[-1]).lower()
                    or "approval unavailable" in str(gateway.messages_seen[-1]).lower()
                )
            elif restriction == "capacity":
                assert "capacity" in str(gateway.messages_seen[-1]).lower()
            elif restriction != "kill":
                assert all(
                    "search_archived_conversations"
                    not in {s["function"]["name"] for s in (tools or [])}
                    for tools in gateway.tools_seen
                )
                assert "not permitted" in str(gateway.messages_seen[-1]).lower()
            else:
                assert "kill switch" in str(gateway.messages_seen[-1]).lower()
        if restriction in {"cold_view", "received_cold_view", "prepare_error"}:
            assert prepared == [source.id]
        if restriction == "agent_wake":
            assert len(wake_dispatch) == 1
            assert wake_dispatch[0]["work_origin"] is WorkOrigin.AUTOMATIC
            assert wake_dispatch[0]["recovery_provider"] is None
            assert not recovery._authorities
            assert any(
                "not user input" in str(messages).lower()
                for messages in gateway.messages_seen
            )
        assert store.active_session_id == source.id
    finally:
        await recovery.shutdown()
        await controller.shutdown()
        runs_db.close()
        if capacity_hold is not None:
            capacity_hold.finish()
            capacity_owner.finish_root()


from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_memory,workspace_memory",
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_search_respects_both_real_database_thread_owners(
    tmp_path, monkeypatch, chat_memory, workspace_memory
):
    db = CharactersRAGDB(
        ":memory:" if chat_memory else tmp_path / "chat.db", "thread-test"
    )
    workspace = WorkspaceDB(
        ":memory:" if workspace_memory else tmp_path / "workspace.db"
    )
    app = SimpleNamespace(
        local_chat_conversation_service=ChatConversationService(db),
        workspace_registry_service=LocalWorkspaceRegistryService(workspace),
    )
    cid = db.add_conversation({"title": "Found"})
    db.set_conversations_archived([cid], archived=True, expected_versions={cid: 1})
    owner = threading.get_ident()
    threads = []
    search = ArchiveSearchService.search

    def observed(self, *args, **kwargs):
        threads.append(threading.get_ident())
        return search(self, *args, **kwargs)

    monkeypatch.setattr(ArchiveSearchService, "search", observed)
    runtime = ArchiveRecoveryRuntime(app)
    try:
        result = await runtime.search("source", {}, model_visible=True)
        assert result["items"][0]["conversation_id"] == cid
        assert (threads == [owner]) is (chat_memory or workspace_memory)
    finally:
        await runtime.shutdown()
        try:
            with db.quiesce_connections(timeout_seconds=2.0):
                pass
        finally:
            workspace.close()


@pytest.mark.asyncio
async def test_cancelled_shutdown_joins_started_search_and_blocks_late_publication(
    recovery_stores, monkeypatch
):
    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    started = threading.Event()
    release = threading.Event()
    search = ArchiveSearchService.search

    def held(self, *args, **kwargs):
        started.set()
        assert release.wait(10)
        return search(self, *args, **kwargs)

    monkeypatch.setattr(ArchiveSearchService, "search", held)
    searching = asyncio.create_task(runtime.search("source", {}, model_visible=True))
    await asyncio.to_thread(started.wait)
    shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    shutdown.cancel()
    await asyncio.sleep(0)
    assert not shutdown.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    with pytest.raises(ArchiveRecoveryError):
        await searching
    assert runtime.presentations("source") == ()
    assert not runtime._searches


@pytest.mark.asyncio
async def test_detach_preserves_typed_results_and_pending_coordinator_state(
    recovery_stores,
):
    from Tests.Chat.test_console_runtime_lifetime import _View

    app, _, _ = recovery_stores
    owner = ConsoleRuntime(app)
    runtime = owner.ensure_archive_recovery()
    view = _View(
        {"present_confirmation": lambda _: True, "is_answerable": lambda _: True}
    )
    generation = owner.attach_view(view)
    projection = await runtime.search("source", {}, model_visible=False)
    card = await runtime.coordinator.request(
        "source", projection["items"][0]["result_ref"], call_key=None, scope="chat_only"
    )
    assert owner.detach_view(view, generation)
    assert not runtime.answerable("source")
    assert (
        runtime.coordinator.snapshot("source").confirmation.request_id
        == card.request_id
    )
    assert runtime.presentations("source")[0].projection == projection
    remounts = []
    next_view = _View(
        {
            "present_confirmation": lambda value: remounts.append(value) or True,
            "is_answerable": lambda _: True,
        }
    )
    owner.attach_view(next_view)
    assert runtime.coordinator.present_pending("source")
    assert remounts[0] == card
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_native_unicode_projection_fits_entire_envelope_without_skipping_rows(
    recovery_stores,
):
    from Tests.Agents.test_archive_recovery_tools import admitted
    from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor

    app, original, db = recovery_stores
    with app.workspace_registry_service.db.transaction() as conn:
        conn.execute(
            "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
            ("界" * 480, "project"),
        )
    expected = {original}
    for index in range(20):
        cid = db.add_conversation(
            {
                "title": f"{index}:" + "界" * 1020,
                "scope_type": "workspace",
                "workspace_id": "project",
            }
        )
        db.set_conversations_archived([cid], archived=True, expected_versions={cid: 1})
        expected.add(cid)
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    provider = admitted(runtime)
    await asyncio.to_thread(provider.activate, "native-bounds")
    search, _ = provider.build_closures()

    def invoke(arguments):
        with use_run_actor(CurrentRunActor("primary", "native-bounds", None)):
            return search(arguments)

    seen = []
    arguments = {"limit": 20}
    while True:
        result = await asyncio.to_thread(invoke, arguments)
        assert result.ok
        assert len(result.content.encode("utf-8")) <= 32768
        projection = json.loads(result.content)
        seen.extend(row["conversation_id"] for row in projection["items"])
        if projection["cursor"] is None:
            break
        changed = await asyncio.to_thread(
            invoke, {"cursor": projection["cursor"], "query": "changed"}
        )
        assert not changed.ok
        arguments = {"cursor": projection["cursor"]}
    assert len(seen) == len(expected)
    assert set(seen) == expected
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_console_terminal_session_close_revokes_recovery_state(recovery_stores):
    app, _, _ = recovery_stores
    owner = ConsoleRuntime(app)
    store = ConsoleChatStore()
    source = store.create_session(title="Source")
    controller = ConsoleChatController(store=store, provider_gateway=None)
    owner.set_chat_store(store)
    owner.set_chat_controller(controller)
    runtime = owner.ensure_archive_recovery()
    await runtime.search(source.id, {}, model_visible=False)
    await owner.close_session(
        source.id, expected_revision=controller.lifecycle_impact().revision
    )
    assert source.id in runtime._closed_sessions
    assert not runtime.presentations(source.id)
    with pytest.raises(ArchiveRecoveryError):
        await runtime.search(source.id, {}, model_visible=False)
    await owner.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_console_dispose_joins_recovery_before_closing_native_database(
    recovery_stores, tmp_path, monkeypatch, cancel_waiter
):
    app, _, _ = recovery_stores
    owner = ConsoleRuntime(app)
    runtime = owner.ensure_archive_recovery()
    runs = AgentRunsDB(tmp_path / "dispose-runs.sqlite", "test")
    owner._agent_runs_db = runs
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    search = ArchiveSearchService.search
    close = runs.close

    def held(self, *args, **kwargs):
        started.set()
        assert release.wait(10)
        return search(self, *args, **kwargs)

    def checked_close():
        assert release.is_set() and not runtime._searches
        close()
        closed.set()

    monkeypatch.setattr(ArchiveSearchService, "search", held)
    monkeypatch.setattr(runs, "close", checked_close)
    searching = asyncio.create_task(runtime.search("source", {}, model_visible=True))
    await asyncio.to_thread(started.wait)
    disposing = asyncio.create_task(owner.dispose())
    await asyncio.sleep(0)
    assert not closed.is_set() and not disposing.done()
    if cancel_waiter:
        disposing.cancel()
        await asyncio.sleep(0)
        assert not closed.is_set() and not disposing.done()
    release.set()
    if cancel_waiter:
        with pytest.raises(asyncio.CancelledError):
            await disposing
    else:
        await disposing
    with pytest.raises(ArchiveRecoveryError):
        await searching
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiters", [False, True])
async def test_shutdown_joins_first_search_already_owned_by_session_close(
    recovery_stores, monkeypatch, cancel_waiters
):
    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    started, release = threading.Event(), threading.Event()
    original = ArchiveSearchService.search

    def held(self, *args, **kwargs):
        started.set()
        assert release.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ArchiveSearchService, "search", held)
    search = asyncio.create_task(runtime.search("source", {}, model_visible=True))
    await asyncio.to_thread(started.wait)
    close = asyncio.create_task(runtime.close_session("source"))
    shutdown = None
    try:

        async def close_has_custody():
            while "source" in runtime._searches:
                await asyncio.sleep(0)

        await asyncio.wait_for(close_has_custody(), 1)
        assert runtime.presentations("source") == ()
        assert not runtime._authorities
        assert not runtime._close_tasks["source"].done()
        shutdown = asyncio.create_task(runtime.shutdown())
        # Drain ready callbacks so an incorrectly empty shutdown can finish.
        for _ in range(20):
            await asyncio.sleep(0)
        if cancel_waiters:
            close.cancel()
            shutdown.cancel()
            for _ in range(20):
                await asyncio.sleep(0)
        returned_before_storage_finished = shutdown.done()
        assert not close.done()
    finally:
        release.set()
        await asyncio.gather(
            close, *([shutdown] if shutdown else []), return_exceptions=True
        )
        with pytest.raises(ArchiveRecoveryError):
            await search
    assert not returned_before_storage_finished, (
        "Shutdown abandoned a retained session-close storage search"
    )
    assert runtime._close_tasks["source"].done()
    assert not runtime._searches
    if cancel_waiters:
        assert close.cancelled() and shutdown.cancelled()


@pytest.mark.asyncio
async def test_first_queue_edit_initializes_recovery_for_the_existing_controller(
    recovery_stores,
):
    from dataclasses import dataclass

    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptQueueUIController,
    )

    @dataclass(frozen=True)
    class Configuration:
        recovery_user_origin: object = None

    app, _, _ = recovery_stores
    owner = app.console_runtime = ConsoleRuntime(app)
    controller = SimpleNamespace()
    owner.set_chat_controller(controller)
    assert owner._archive_recovery is None
    editor = ConsolePromptQueueUIController.__new__(ConsolePromptQueueUIController)
    editor._chat_controller_accessor = lambda: controller
    editor._capture_configuration = lambda _: Configuration()
    captured = editor._capture_user_configuration("source")
    assert owner._archive_recovery is not None
    assert controller._archive_recovery_runtime is owner._archive_recovery
    proof = captured.recovery_user_origin
    assert owner._archive_recovery._proofs[id(proof)] is proof
    await owner.dispose()
