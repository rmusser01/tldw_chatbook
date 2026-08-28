"""Production-composition qualification for Console Library authority."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import tldw_chatbook.Chat.console_chat_controller as controller_module
from Tests.fixtures.console_library_recording_provider import (
    RecordingConsoleProvider,
    RetrievalScript,
    StreamScript,
    ToolBatchScript,
)
from Tests.Chat.test_console_dispatch_continuation_handoff import (
    _continuation,
    _deepseek_acceptance,
)
from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _reconcile,
    _restored_store,
    _start,
)
from tldw_chatbook.Agents.library_rag_tool_provider import (
    LibraryRagToolProvider,
    RAG_TOOL_NAME,
)
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Chat.console_agent_bridge import (
    _compose_run_registry_and_allowed,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryKind
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleContinuationHandoff,
    ConsoleDispatchResultStatus,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_prompt_queue import PromptQueueMode
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
    initial_preparation_state,
)
from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _context(
    auto: ConsoleAutoRetrieve,
    assistant: ConsoleAssistantLibraryAccess,
    *,
    direct: bool,
    attempt_id: str = "attempt-1",
) -> ConsoleTurnExecutionContext:
    authority = ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(auto, assistant, 1, "durable"),
        direct_library_tools=direct,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent(
            "llama_cpp", "qualification-model", "http://127.0.0.1:9099"
        ),
        attempt_id=attempt_id,
    )
    return ConsoleTurnExecutionContext(
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id="session-1",
            provider_selection=ConsoleProviderSelection(
                provider="llama_cpp", explicit_model="qualification-model"
            ),
        ),
        library_authority=authority,
        resolved_destination=ConsoleResolvedDestination(
            provider="llama_cpp",
            model="qualification-model",
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
    )


def _preparation(context: ConsoleTurnExecutionContext) -> ConsoleTurnPreparation:
    return ConsoleTurnPreparation(
        preparation_id="preparation-1",
        attempt_id=context.library_authority.attempt_id,
        session_id="session-1",
        origin="manual",
        queue_entry_id=None,
        executed_draft="exact qualification draft",
        execution_context=context,
        transient_user_message_id=None,
        attachment_ids=(),
        evidence_ids=(),
        prefill_id=None,
        queue_generation=None,
        pre_send_title="Qualification",
        pre_send_conversation_id=None,
        state=initial_preparation_state(
            context.library_authority.policy.auto_retrieve
        ),
        pause_kind=None,
        one_shot_bypass=False,
        ephemeral=False,
    )


def _provider_factory(recorder: RecordingConsoleProvider):
    def build(context: ConsoleTurnExecutionContext):
        if context.library_authority.direct_library_tools:
            return LibraryToolProvider(
                recorder,
                activity_attempt_id=context.library_authority.attempt_id,
                activity_sink=recorder.activity_events.append,
            )
        return LibraryRagToolProvider(
            recorder,
            activity_attempt_id=context.library_authority.attempt_id,
            activity_sink=recorder.activity_events.append,
        )

    return build


@pytest.mark.parametrize(
    ("auto", "assistant"),
    (
        (ConsoleAutoRetrieve.NEVER, ConsoleAssistantLibraryAccess.BLOCKED),
        (ConsoleAutoRetrieve.NEVER, ConsoleAssistantLibraryAccess.ALLOWED),
        (ConsoleAutoRetrieve.AUTOMATIC, ConsoleAssistantLibraryAccess.BLOCKED),
        (ConsoleAutoRetrieve.AUTOMATIC, ConsoleAssistantLibraryAccess.ALLOWED),
    ),
)
@pytest.mark.parametrize("direct", (True, False), ids=("direct", "rag"))
def test_four_policy_combinations_compose_only_the_authorized_provider(
    auto: ConsoleAutoRetrieve,
    assistant: ConsoleAssistantLibraryAccess,
    direct: bool,
) -> None:
    recorder = RecordingConsoleProvider()
    context = _context(auto, assistant, direct=direct)
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=recorder,
        library_provider_factory=_provider_factory(recorder),
    )

    selected = controller._library_provider_for_context(context)
    provider, authority = selected if selected is not None else (None, None)
    registry, allowed, _builtins, _locals = _compose_run_registry_and_allowed(
        {}, library_provider=provider, library_authority=authority, ephemeral=True
    )
    library_names = {
        entry.name for entry in registry.list_catalog() if entry.source == "library"
    }

    if assistant is ConsoleAssistantLibraryAccess.BLOCKED:
        assert selected is None
        assert library_names == set()
    elif direct:
        assert library_names == set(LIBRARY_TOOL_DESCRIPTORS)
        assert set(LIBRARY_TOOL_DESCRIPTORS).issubset(allowed)
    else:
        assert library_names == {RAG_TOOL_NAME}
        assert RAG_TOOL_NAME in allowed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script", "expected_state", "expected_pause", "expected_evidence"),
    (
        (RetrievalScript.success(), ConsoleTurnPreparationState.READY, None, True),
        (RetrievalScript.zero(), ConsoleTurnPreparationState.READY, None, False),
        (RetrievalScript.failure(), ConsoleTurnPreparationState.PAUSED,
         ConsolePreparationPauseKind.RETRIEVAL, False),
        (RetrievalScript.timeout(), ConsoleTurnPreparationState.PAUSED,
         ConsolePreparationPauseKind.RETRIEVAL, False),
    ),
)
async def test_automatic_preparation_uses_fixed_categories_and_scripted_outcomes(
    script: RetrievalScript,
    expected_state: ConsoleTurnPreparationState,
    expected_pause: ConsolePreparationPauseKind | None,
    expected_evidence: bool,
) -> None:
    recorder = RecordingConsoleProvider(retrieval_scripts=[script])
    context = _context(
        ConsoleAutoRetrieve.AUTOMATIC,
        ConsoleAssistantLibraryAccess.BLOCKED,
        direct=True,
    )
    preparation = _preparation(context)
    store = ConsoleChatStore()
    store.create_session(session_id="session-1", title="Qualification")
    assert store.begin_preparation(preparation) is preparation
    controller = ConsoleChatController(
        store=store,
        provider_gateway=recorder,
        library_preparation_timeout=0.01,
    )
    controller.app = SimpleNamespace(library_rag_search_service=recorder)

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.state is expected_state
    assert (outcome.evidence_bundle is not None) is expected_evidence
    assert store.preparation_for_session("session-1").pause_kind is expected_pause
    retrieval = recorder.calls_of("retrieval")
    assert len(retrieval) == 1
    assert retrieval[0].metadata["source_types"] == AUTOMATIC_LIBRARY_SOURCE_TYPES
    assert "PRIVATE USER BODY" not in repr(recorder.calls)
    assert "PRIVATE LIBRARY BODY" not in repr(recorder.calls)
    assert "PRIVATE LIBRARY BODY" not in repr(recorder.activity_events)


@pytest.mark.asyncio
async def test_recording_provider_covers_stream_tool_continuation_and_redacts_bodies() -> None:
    recorder = RecordingConsoleProvider(
        stream_scripts=[StreamScript.tokens("hel", "lo")],
        model_scripts=[ToolBatchScript.library_search_then_continue()],
    )
    resolution = await recorder.resolve_for_send(
        ConsoleProviderSelection("llama_cpp", "qualification-model")
    )

    chunks = [
        chunk
        async for chunk in recorder.stream_chat(
            resolution,
            [{"role": "user", "content": "PRIVATE USER BODY"}],
        )
    ]
    turn = recorder.next_model_turn([], ())

    assert chunks == ["hel", "lo"]
    assert len(turn.tool_calls) == 1
    assert turn.provider_continuation is not None
    assert "PRIVATE USER BODY" not in repr(recorder.calls)
    assert recorder.calls_of("readiness")[0].destination == "on_device"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("started", "action"),
    ((False, "retry"), (True, "discard")),
    ids=("accepted-retry", "dispatch-started-discard"),
)
async def test_restart_recovery_requires_explicit_production_action(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    started: bool,
    action: str,
) -> None:
    real_get_cli_setting = controller_module.get_cli_setting
    monkeypatch.setattr(
        controller_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            False
            if (section, key) == ("console", "direct_library_tools")
            else real_get_cli_setting(section, key, default)
        ),
    )
    db, conversation_id, repository = _database(
        tmp_path / f"recovery-{action}.sqlite"
    )
    checkpoint = _insert(db, repository, _acceptance(conversation_id))
    if started:
        _start(repository, checkpoint)
    store, session_id = _restored_store(db, conversation_id)
    recorder = RecordingConsoleProvider(
        stream_scripts=[StreamScript.tokens("recovered")],
        fixed_provider="llama_cpp",
        fixed_model="test-model",
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=recorder,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )

    await asyncio.sleep(0)

    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.kind is (
        ConsoleDispatchRecoveryKind.DISPATCH_STARTED
        if started
        else ConsoleDispatchRecoveryKind.ACCEPTED
    )
    assert recorder.calls == []

    if action == "retry":
        result = await controller.retry_dispatch_recovery(session_id)
        assert result.accepted is True
        assert [call.kind for call in recorder.calls] == ["readiness", "stream"]
    else:
        result = await controller.discard_dispatch_recovery(session_id)
        assert result.accepted is True
        assert recorder.calls == []

    assert store.dispatch_recovery_for_session(session_id) is None


@pytest.mark.asyncio
async def test_prompt_queue_drains_through_controller_and_recording_gateway() -> None:
    release = [asyncio.Event(), asyncio.Event()]
    recorder = RecordingConsoleProvider(
        stream_scripts=[
            StreamScript.tokens("first reply"),
            StreamScript.tokens("second reply"),
        ],
        stream_gates=release,
    )
    store = ConsoleChatStore()
    session = store.create_session(title="Queue qualification", ephemeral=True)
    controller = ConsoleChatController(store=store, provider_gateway=recorder)

    chain = asyncio.create_task(
        controller.run_prompt_chain("first prompt", session_id=session.id)
    )
    await asyncio.wait_for(recorder.stream_started[0].wait(), timeout=1)
    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    queued = controller.queue_prompt(
        session.id,
        text="second prompt",
        expected_revision=snapshot.revision,
    )
    assert queued.applied is True

    release[0].set()
    await asyncio.wait_for(recorder.stream_started[1].wait(), timeout=1)
    assert controller.prompt_queue_registry.snapshot(session.id).total_count == 0
    release[1].set()
    result = await asyncio.wait_for(chain, timeout=1)

    assert result.accepted is True
    assert controller.prompt_queue_registry.snapshot(session.id).mode is (
        PromptQueueMode.DRAINING
    )
    assert [call.kind for call in recorder.calls] == [
        "readiness",
        "stream",
        "readiness",
        "stream",
    ]
    assert "first prompt" not in repr(recorder.calls)
    assert "second prompt" not in repr(recorder.calls)


@pytest.mark.asyncio
async def test_on_device_to_public_change_discloses_during_production_send(
    tmp_path,
) -> None:
    release = [asyncio.Event(), asyncio.Event()]
    recorder = RecordingConsoleProvider(
        stream_scripts=[
            StreamScript.tokens("local reply"),
            StreamScript.tokens("public reply"),
        ],
        stream_gates=release,
    )
    db = CharactersRAGDB(tmp_path / "destination.sqlite", "qualification")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(
        title="Destination qualification",
    )
    store.stage_session_library_policy(
        session.id,
        ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    controller = ConsoleChatController(store=store, provider_gateway=recorder)

    local_send = asyncio.create_task(
        controller.submit_draft("local turn", session_id=session.id)
    )
    await asyncio.wait_for(recorder.stream_started[0].wait(), timeout=1)
    release[0].set()
    assert (await asyncio.wait_for(local_send, timeout=1)).accepted is True

    recorder.egress = ConsoleEgressClass.PUBLIC_NETWORK
    public_send = asyncio.create_task(
        controller.submit_draft("public turn", session_id=session.id)
    )
    await asyncio.wait_for(recorder.stream_started[1].wait(), timeout=1)
    disclosure = session.library_destination_runtime.disclosure

    assert disclosure is not None
    assert disclosure.previous_resolved_identity[3] is ConsoleEgressClass.ON_DEVICE
    assert disclosure.resolved_destination.egress_class is (
        ConsoleEgressClass.PUBLIC_NETWORK
    )
    assert [call.destination for call in recorder.calls_of("readiness")] == [
        "on_device",
        "public_network",
    ]
    release[1].set()
    assert (await asyncio.wait_for(public_send, timeout=1)).accepted is True
    assert session.library_destination_runtime.disclosure is None


def test_permanent_conversation_purge_cascades_every_library_sidecar(tmp_path) -> None:
    db, conversation_id, repository = _database(tmp_path / "purge.sqlite")
    checkpoint = _insert(db, repository, _acceptance(conversation_id))
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "INSERT INTO message_trajectory_metadata "
            "(message_id, conversation_id, turn_id, seq, event_kind, payload_json) "
            "VALUES (?, ?, ?, 1, 'library_preparation', ?)",
            (
                checkpoint.user_message_id,
                conversation_id,
                checkpoint.user_message_id,
                '{"version":1,"outcome":"zero_results"}',
            ),
        )

    connection = db.get_connection()
    sidecars = (
        "console_conversation_library_policy",
        "console_dispatch_checkpoints",
        "message_trajectory_metadata",
    )
    assert {
        table: connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        for table in sidecars
    } == {table: 1 for table in sidecars}

    with db.transaction(immediate=True) as cursor:
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    assert {
        table: connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        for table in sidecars
    } == {table: 0 for table in sidecars}


def test_continuation_handoff_deletes_dispatch_owner_before_any_provider_call(
    tmp_path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "handoff.sqlite")
    started = _start(
        repository,
        _insert(db, repository, _deepseek_acceptance(conversation_id)),
    )
    continuation_json = dump_provider_continuation_json(_continuation())
    assert continuation_json is not None

    result = repository.handoff_to_provider_continuation(
        ConsoleContinuationHandoff(
            assistant_message_id=started.assistant_message_id,
            expected_checkpoint_revision=started.checkpoint_revision,
            expected_user_message_version=started.user_message_version,
            expected_assistant_message_version=started.assistant_message_version,
            provider_continuation_json=continuation_json,
        )
    )

    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    assert _reconcile(repository, conversation_id).kind is (
        ConsoleDispatchRecoveryKind.CONTINUATION
    )
    with db.transaction() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints "
            "WHERE conversation_id = ?",
            (conversation_id,),
        )
        assert cursor.fetchone()[0] == 0
