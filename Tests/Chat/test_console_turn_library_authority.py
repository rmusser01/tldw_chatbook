"""Execution-time ordering and immutability for Console Library authority."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import ContinuationEventContext, ToolBatchReady
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleSubmitResult,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.Chat.console_library_policy_repository import (
    ConsoleLibraryPolicyRepository,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Chat.console_prompt_queue import PromptQueueReservation
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _policy(
    *,
    auto_retrieve: ConsoleAutoRetrieve = ConsoleAutoRetrieve.AUTOMATIC,
    assistant_access: ConsoleAssistantLibraryAccess = (
        ConsoleAssistantLibraryAccess.ALLOWED
    ),
    source: str = "durable",
    error_code: str | None = None,
) -> ConsoleLibraryPolicySnapshot:
    return ConsoleLibraryPolicySnapshot(
        auto_retrieve=auto_retrieve,
        assistant_access=assistant_access,
        policy_revision=7 if source == "durable" else None,
        source=source,
        error_code=error_code,
    )


def _destination(provider: str = "openai") -> ConsoleResolvedDestination:
    return ConsoleResolvedDestination(
        provider=provider,
        model="model-a",
        endpoint_identity="https://gateway.example.invalid/v1",
        egress_class=ConsoleEgressClass.UNKNOWN,
    )


class _RecordingCoordinator:
    def __init__(
        self,
        events: list[str],
        snapshot: ConsoleLibraryPolicySnapshot,
    ) -> None:
        self.events = events
        self.snapshot = snapshot
        self.calls: list[str] = []

    async def capture_for_execution(self, session_id: str):
        self.events.append("policy")
        self.calls.append(session_id)
        return self.snapshot


class _RecordingGateway:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.release_resolution = asyncio.Event()
        self.release_resolution.set()
        self.resolved = asyncio.Event()
        self.stream_calls = 0

    async def resolve_for_send(self, selection):
        self.events.append("gateway")
        self.resolved.set()
        await self.release_resolution.wait()
        return SimpleNamespace(
            ready=True,
            provider=selection.provider,
            model=selection.explicit_model or selection.configured_model,
            base_url=selection.base_url or "https://gateway.example.invalid/v1",
            visible_copy="",
            resolved_destination=_destination(selection.provider),
        )

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        self.stream_calls += 1
        yield "reply"


class _DestinationSequenceGateway:
    def __init__(
        self,
        store: ConsoleChatStore,
        session_id: str,
        destinations: list[ConsoleResolvedDestination],
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.destinations = list(destinations)
        self.snapshots = []
        self.starts = [asyncio.Event() for _ in destinations]
        self.releases = [asyncio.Event() for _ in destinations]
        self.block_streams = False
        self._stream_index = 0

    async def resolve_for_send(self, _selection):
        destination = self.destinations.pop(0)
        return SimpleNamespace(
            ready=True,
            provider=destination.provider,
            model=destination.model,
            base_url=destination.endpoint_identity,
            visible_copy="",
            resolved_destination=destination,
        )

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        index = self._stream_index
        self._stream_index += 1
        session = next(
            item for item in self.store.sessions() if item.id == self.session_id
        )
        self.snapshots.append(session.library_destination_runtime)
        self.starts[index].set()
        if self.block_streams:
            await self.releases[index].wait()
        yield f"reply-{index}"


def _local_then_destination(
    egress_class: ConsoleEgressClass,
) -> list[ConsoleResolvedDestination]:
    external_identity = {
        ConsoleEgressClass.PRIVATE_NETWORK: "http://10.20.30.40:8000",
        ConsoleEgressClass.PUBLIC_NETWORK: "https://api.openai.com",
        ConsoleEgressClass.UNKNOWN: "external/unknown",
    }[egress_class]
    return [
        ConsoleResolvedDestination(
            provider="llama_cpp",
            model="model-a",
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
        ConsoleResolvedDestination(
            provider="openai",
            model="model-a",
            endpoint_identity=external_identity,
            egress_class=egress_class,
        ),
    ]


def _allowed_candidate(*, allowed: bool) -> ConsoleLibraryPolicyCandidate:
    return ConsoleLibraryPolicyCandidate(
        auto_retrieve=(
            ConsoleAutoRetrieve.AUTOMATIC if allowed else ConsoleAutoRetrieve.NEVER
        ),
        assistant_access=(
            ConsoleAssistantLibraryAccess.ALLOWED
            if allowed
            else ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )


def _execution_configuration(
    session_id: str,
    *,
    direct: bool = True,
) -> ConsoleTurnConfigurationSnapshot:
    return ConsoleTurnConfigurationSnapshot.capture(
        session_id=session_id,
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="model-a",
            base_url="https://api.openai.com/v1",
        ),
        tool_configuration={
            "agent_runtime_enabled": False,
            "direct_library_tools": direct,
        },
    )


@pytest.mark.asyncio
async def test_immediate_capture_follows_admission_and_precedes_gateway(monkeypatch):
    events: list[str] = []
    store = ConsoleChatStore()
    session = store.create_session()
    coordinator = _RecordingCoordinator(events, _policy())
    store.library_policy_coordinator = coordinator
    gateway = _RecordingGateway(events)

    def capture_configuration(session_id: str):
        events.append("configuration")
        return ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider="openai", explicit_model="model-a"
            ),
            tool_configuration={"agent_runtime_enabled": False},
        )

    real_authority = ConsoleTurnLibraryAuthority

    def record_authority(**kwargs):
        events.append("authority")
        return real_authority(**kwargs)

    monkeypatch.setattr(
        controller_module,
        "ConsoleTurnLibraryAuthority",
        record_authority,
    )

    async def capture_rag(_draft, turn_context=None, **_kwargs):
        assert isinstance(turn_context, ConsoleTurnExecutionContext)
        events.append("rag")
        return SimpleNamespace(context=None)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        turn_context_provider=capture_configuration,
        rag_capture_provider=capture_rag,
        agent_runtime_enabled=False,
    )

    rejected = await controller.submit_draft("   ", session_id=session.id)
    assert rejected.accepted is False
    assert events == []

    accepted = await controller.submit_draft("question", session_id=session.id)

    assert accepted.accepted is True
    assert events[:5] == [
        "configuration",
        "policy",
        "authority",
        "gateway",
        "rag",
    ]
    assert coordinator.calls == [session.id]


@pytest.mark.asyncio
async def test_unavailable_fresh_read_defeats_cached_allowed_holder():
    events: list[str] = []
    store = ConsoleChatStore()
    session = store.create_session()
    store.stage_session_library_policy(
        session.id,
        ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    unavailable = _policy(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        source="unavailable",
        error_code="policy_read_error",
    )
    store.library_policy_coordinator = _RecordingCoordinator(events, unavailable)
    observed: list[ConsoleTurnExecutionContext] = []

    async def capture_rag(_draft, turn_context=None, **_kwargs):
        observed.append(turn_context)
        return SimpleNamespace(context=None)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(events),
        rag_capture_provider=capture_rag,
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft("question", session_id=session.id)

    assert result.accepted is True
    authority = observed[0].library_authority
    assert authority.policy == unavailable
    assert authority.policy.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert authority.policy.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
    assert authority.policy.source == "unavailable"
    assert authority.policy.error_code == "policy_read_error"


@pytest.mark.asyncio
async def test_real_execution_capture_defeats_stale_allowed_and_freezes_current_turn(
    tmp_path,
) -> None:
    """Two live holders cannot bypass a second-process durable policy commit."""
    path = tmp_path / "integrated-authority.sqlite"
    first_db = CharactersRAGDB(path, "runtime-coordinator")
    second_db = None
    try:
        conversation_id = first_db.add_conversation({"title": "authority"})
        assert conversation_id is not None
        first_repository = ConsoleLibraryPolicyRepository(first_db)
        assert first_repository.insert(
            conversation_id,
            _allowed_candidate(allowed=True),
        ).status is ConsoleLibraryPolicyWriteStatus.COMMITTED

        store = ConsoleChatStore(persistence=ChatPersistenceService(first_db))
        first_session = store.restore_persisted_session(
            title="first",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=(),
        )
        second_session = store.restore_persisted_session(
            title="second",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=(),
        )
        await store.hydrate_session_library_policy(first_session.id)
        await store.hydrate_session_library_policy(second_session.id)
        assert first_session.library_policy_holder.snapshot.policy_revision == 1
        assert second_session.library_policy_holder.snapshot.policy_revision == 1

        second_db = CharactersRAGDB(path, "second-process")
        second_repository = ConsoleLibraryPolicyRepository(second_db)
        assert second_repository.compare_and_swap(
            conversation_id,
            1,
            _allowed_candidate(allowed=False),
        ).status is ConsoleLibraryPolicyWriteStatus.COMMITTED

        factory_calls: list[ConsoleTurnExecutionContext] = []

        def factory(context: ConsoleTurnExecutionContext):
            from tldw_chatbook.Agents.library_rag_tool_provider import (
                LibraryRagToolProvider,
            )
            from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

            factory_calls.append(context)
            return (
                LibraryToolProvider(SimpleNamespace(invoke=lambda *_args: {}))
                if context.library_authority.direct_library_tools
                else LibraryRagToolProvider(None)
            )

        controller = ConsoleChatController(
            store=store,
            provider_gateway=_RecordingGateway([]),
            library_provider_factory=factory,
        )
        _resolution, blocked = (
            await controller._capture_and_resolve_turn_execution_context(
                first_session.id,
                _execution_configuration(first_session.id),
            )
        )
        assert blocked is not None
        assert blocked.library_authority.policy.policy_revision == 2
        assert blocked.library_authority.policy.assistant_access is (
            ConsoleAssistantLibraryAccess.BLOCKED
        )
        assert controller._library_provider_for_context(blocked) is None
        assert factory_calls == []
        assert first_session.library_policy_holder.snapshot == (
            second_session.library_policy_holder.snapshot
        )

        assert second_repository.compare_and_swap(
            conversation_id,
            2,
            _allowed_candidate(allowed=True),
        ).status is ConsoleLibraryPolicyWriteStatus.COMMITTED
        _resolution, captured = (
            await controller._capture_and_resolve_turn_execution_context(
                first_session.id,
                _execution_configuration(first_session.id, direct=True),
            )
        )
        assert captured is not None
        selected = controller._library_provider_for_context(captured)
        assert selected is not None
        assert captured.library_authority.policy.policy_revision == 3

        assert second_repository.compare_and_swap(
            conversation_id,
            3,
            _allowed_candidate(allowed=False),
        ).status is ConsoleLibraryPolicyWriteStatus.COMMITTED
        # A commit after capture cannot mutate the already-running context.
        assert captured.library_authority.policy.assistant_access is (
            ConsoleAssistantLibraryAccess.ALLOWED
        )
        assert controller._library_provider_for_context(captured) is not None

        _resolution, next_turn = (
            await controller._capture_and_resolve_turn_execution_context(
                second_session.id,
                _execution_configuration(second_session.id),
            )
        )
        assert next_turn is not None
        assert next_turn.library_authority.policy.policy_revision == 4
        assert next_turn.library_authority.policy.assistant_access is (
            ConsoleAssistantLibraryAccess.BLOCKED
        )
        assert controller._library_provider_for_context(next_turn) is None
    finally:
        if second_db is not None:
            second_db.close_connection()
        first_db.close_connection()


@pytest.mark.asyncio
async def test_missing_policy_coordinator_fails_closed_instead_of_using_holder():
    store = ConsoleChatStore()
    session = store.create_session()
    store.stage_session_library_policy(
        session.id,
        ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway([]),
    )
    configuration = controller.resolve_turn_configuration_snapshot(session.id)

    authority = await controller._capture_turn_library_authority(
        session.id,
        configuration,
    )

    assert authority.policy == ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=None,
        source="unavailable",
        error_code="policy_read_error",
    )


@pytest.mark.asyncio
async def test_running_turn_freezes_selector_scope_provider_and_destination():
    events: list[str] = []
    store = ConsoleChatStore()
    session = store.create_session()
    session.rag_scope_holder.set(
        RagScope(
            items=(
                ScopeItem("note", "note-1"),
                ScopeItem("media", "media-1"),
            ),
            updated_at="2026-08-22T00:00:00+00:00",
        )
    )
    coordinator = _RecordingCoordinator(events, _policy())
    store.library_policy_coordinator = coordinator
    live = {
        "provider": "openai",
        "model": "model-a",
        "endpoint": "https://intent-a.example.invalid/v1",
        "direct": True,
    }

    def capture_configuration(session_id: str):
        return ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider=live["provider"],
                explicit_model=live["model"],
                base_url=live["endpoint"],
            ),
            tool_configuration={
                "agent_runtime_enabled": False,
                "direct_library_tools": live["direct"],
            },
        )

    gateway = _RecordingGateway(events)
    gateway.release_resolution.clear()
    observed: list[ConsoleTurnExecutionContext] = []

    async def capture_rag(_draft, turn_context=None, **_kwargs):
        observed.append(turn_context)
        return SimpleNamespace(context=None)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        turn_context_provider=capture_configuration,
        rag_capture_provider=capture_rag,
        agent_runtime_enabled=False,
    )
    task = asyncio.create_task(controller.submit_draft("question", session_id=session.id))
    await gateway.resolved.wait()

    live.update(
        provider="anthropic",
        model="model-b",
        endpoint="https://intent-b.example.invalid/v1",
        direct=False,
    )
    session.rag_scope_holder.set(
        RagScope(
            items=(ScopeItem("note", "note-2"),),
            updated_at="2026-08-22T00:01:00+00:00",
        )
    )
    coordinator.snapshot = replace(
        coordinator.snapshot,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=8,
    )
    gateway.release_resolution.set()
    result = await task

    assert result.accepted is True
    context = observed[0]
    assert context.provider_selection.provider == "openai"
    assert context.effective_model == "model-a"
    assert context.library_authority.direct_library_tools is True
    assert context.library_authority.source_types == AUTOMATIC_LIBRARY_SOURCE_TYPES
    assert context.library_authority.scope_snapshot.note_ids == ("note-1",)
    assert context.library_authority.scope_snapshot.media_ids == ("media-1",)
    assert context.library_authority.scope_snapshot.conversations_allowed is False
    assert context.library_authority.provider_intent.provider == "openai"
    assert context.library_authority.provider_intent.model == "model-a"
    assert context.library_authority.provider_intent.endpoint == (
        "https://intent-a.example.invalid/v1"
    )
    assert context.library_authority.policy.policy_revision == 7
    assert context.resolved_destination == _destination("openai")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auto_retrieve", "assistant_access"),
    [
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.BLOCKED,
        ),
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.BLOCKED,
        ),
    ],
)
async def test_context_resolution_does_not_observe_destination_before_dispatch(
    auto_retrieve: ConsoleAutoRetrieve,
    assistant_access: ConsoleAssistantLibraryAccess,
) -> None:
    events: list[str] = []
    policy = _policy(
        auto_retrieve=auto_retrieve,
        assistant_access=assistant_access,
    )
    store = ConsoleChatStore()
    session = store.create_session()
    holder_before = session.library_policy_holder.snapshot
    store.library_policy_coordinator = _RecordingCoordinator(events, policy)
    destinations = [
        ConsoleResolvedDestination(
            provider="llama_cpp",
            model="model-a",
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
        ConsoleResolvedDestination(
            provider="openai",
            model="model-a",
            endpoint_identity="https://api.openai.com",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    ]

    class SequenceGateway:
        async def resolve_for_send(self, _selection):
            destination = destinations.pop(0)
            return SimpleNamespace(
                ready=True,
                provider=destination.provider,
                model=destination.model,
                base_url=destination.endpoint_identity,
                visible_copy="",
                resolved_destination=destination,
            )

    controller = ConsoleChatController(
        store=store,
        provider_gateway=SequenceGateway(),
        agent_runtime_enabled=False,
    )
    configuration = controller.resolve_turn_configuration_snapshot(session.id)

    await controller._capture_and_resolve_turn_execution_context(
        session.id,
        configuration,
    )
    await controller._capture_and_resolve_turn_execution_context(
        session.id,
        configuration,
    )

    runtime = session.library_destination_runtime
    assert runtime.resolved_destination is None
    assert runtime.last_resolved_identity is None
    assert runtime.disclosure is None
    assert runtime.owner_attempt_id is None
    assert runtime.owner_message_id is None
    assert session.library_policy_holder.snapshot == holder_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auto_retrieve", "assistant_access", "expects_disclosure"),
    [
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.BLOCKED,
            True,
        ),
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.ALLOWED,
            True,
        ),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
            True,
        ),
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.BLOCKED,
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "external_class",
    [
        ConsoleEgressClass.PRIVATE_NETWORK,
        ConsoleEgressClass.PUBLIC_NETWORK,
        ConsoleEgressClass.UNKNOWN,
    ],
)
async def test_submit_draft_observes_resolved_destination_at_real_dispatch_boundary(
    auto_retrieve: ConsoleAutoRetrieve,
    assistant_access: ConsoleAssistantLibraryAccess,
    expects_disclosure: bool,
    external_class: ConsoleEgressClass,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    store.library_policy_coordinator = _RecordingCoordinator(
        [],
        _policy(
            auto_retrieve=auto_retrieve,
            assistant_access=assistant_access,
        ),
    )
    holder_before = session.library_policy_holder.snapshot
    gateway = _DestinationSequenceGateway(
        store,
        session.id,
        _local_then_destination(external_class),
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    first = await controller.submit_draft("first", session_id=session.id)
    second = await controller.submit_draft("second", session_id=session.id)

    assert first.accepted is True
    assert second.accepted is True
    assert len(gateway.snapshots) == 2
    assert gateway.snapshots[0].resolved_destination is not None
    assert (
        gateway.snapshots[0].resolved_destination.egress_class
        is ConsoleEgressClass.ON_DEVICE
    )
    assert gateway.snapshots[0].disclosure is None
    assert gateway.snapshots[0].owner_attempt_id is not None
    assert gateway.snapshots[0].owner_message_id == first.assistant_message_id
    assert gateway.snapshots[1].resolved_destination.egress_class is external_class
    if external_class is ConsoleEgressClass.UNKNOWN:
        assert gateway.snapshots[1].resolved_destination.endpoint_identity == (
            "external/unknown"
        )
        assert "on-device" not in repr(gateway.snapshots[1]).lower()
    assert (gateway.snapshots[1].disclosure is not None) is expects_disclosure
    assert gateway.snapshots[1].owner_attempt_id is not None
    assert gateway.snapshots[1].owner_message_id == second.assistant_message_id
    assert session.library_destination_runtime.owner_attempt_id is None
    assert session.library_destination_runtime.owner_message_id is None
    assert session.library_policy_holder.snapshot == holder_before


@pytest.mark.asyncio
async def test_queued_submit_observes_destination_only_after_dequeue_dispatch() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    store.library_policy_coordinator = _RecordingCoordinator(
        [],
        _policy(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        ),
    )
    gateway = _DestinationSequenceGateway(
        store,
        session.id,
        _local_then_destination(ConsoleEgressClass.PUBLIC_NETWORK),
    )
    gateway.block_streams = True
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    chain = asyncio.create_task(
        controller.run_prompt_chain("manual", session_id=session.id)
    )
    await gateway.starts[0].wait()
    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    queued = controller.queue_prompt(
        session.id,
        text="queued",
        expected_revision=snapshot.revision,
    )
    assert queued.applied is True
    assert len(gateway.snapshots) == 1
    gateway.releases[0].set()
    await gateway.starts[1].wait()

    assert len(gateway.snapshots) == 2
    assert gateway.snapshots[1].disclosure is not None
    assert (
        gateway.snapshots[1].resolved_destination.egress_class
        is ConsoleEgressClass.PUBLIC_NETWORK
    )
    assert gateway.snapshots[1].owner_attempt_id is not None
    assert gateway.snapshots[1].owner_message_id is not None
    gateway.releases[1].set()
    await chain


@pytest.mark.asyncio
async def test_queued_configuration_and_policy_capture_only_after_dequeue():
    events: list[str] = []
    store = ConsoleChatStore()
    session = store.create_session()
    coordinator = _RecordingCoordinator(events, _policy())
    store.library_policy_coordinator = coordinator

    class _QueueGateway(_RecordingGateway):
        def __init__(self):
            super().__init__(events)
            self.starts = [asyncio.Event(), asyncio.Event()]
            self.releases = [asyncio.Event(), asyncio.Event()]
            self.calls = 0

        async def stream_chat(self, _resolution, _messages, **_kwargs):
            index = self.calls
            self.calls += 1
            self.starts[index].set()
            await self.releases[index].wait()
            yield "reply"

    gateway = _QueueGateway()
    captures: list[str] = []

    def capture_configuration(session_id: str):
        captures.append(session_id)
        return ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider="openai", explicit_model="model-a"
            ),
            tool_configuration={"agent_runtime_enabled": False},
        )

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        turn_context_provider=capture_configuration,
        agent_runtime_enabled=False,
    )
    chain = asyncio.create_task(
        controller.run_prompt_chain("manual", session_id=session.id)
    )
    await gateway.starts[0].wait()
    before_queue = len(captures)
    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    queued = controller.queue_prompt(
        session.id,
        text="queued",
        expected_revision=snapshot.revision,
    )
    assert queued.applied is True
    assert len(captures) == before_queue
    assert len(coordinator.calls) == before_queue

    coordinator.snapshot = replace(coordinator.snapshot, policy_revision=9)
    gateway.releases[0].set()
    await gateway.starts[1].wait()

    assert len(captures) == before_queue + 1
    assert len(coordinator.calls) == before_queue + 1
    gateway.releases[1].set()
    await chain


@pytest.mark.asyncio
async def test_queued_turn_reads_second_process_policy_only_after_claim(tmp_path) -> None:
    path = tmp_path / "queued-authority.sqlite"
    first_db = CharactersRAGDB(path, "queue-runtime")
    second_db = None
    try:
        conversation_id = first_db.add_conversation({"title": "queued"})
        assert conversation_id is not None
        first_repository = ConsoleLibraryPolicyRepository(first_db)
        assert first_repository.insert(
            conversation_id,
            _allowed_candidate(allowed=True),
        ).status is ConsoleLibraryPolicyWriteStatus.COMMITTED
        store = ConsoleChatStore(persistence=ChatPersistenceService(first_db))
        session = store.restore_persisted_session(
            title="queued",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=(),
        )
        await store.hydrate_session_library_policy(session.id)

        class QueueGateway(_RecordingGateway):
            def __init__(self):
                super().__init__([])
                self.starts = [asyncio.Event(), asyncio.Event()]
                self.releases = [asyncio.Event(), asyncio.Event()]

            async def stream_chat(self, _resolution, _messages, **_kwargs):
                index = self.stream_calls
                self.stream_calls += 1
                self.starts[index].set()
                await self.releases[index].wait()
                yield f"reply-{index}"

        observed: list[ConsoleTurnExecutionContext] = []

        async def capture_rag(_draft, turn_context=None, **_kwargs):
            assert isinstance(turn_context, ConsoleTurnExecutionContext)
            observed.append(turn_context)
            return SimpleNamespace(context=None)

        gateway = QueueGateway()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            turn_context_provider=lambda sid: _execution_configuration(sid),
            rag_capture_provider=capture_rag,
            agent_runtime_enabled=False,
        )
        chain = asyncio.create_task(
            controller.run_prompt_chain("manual", session_id=session.id)
        )
        await gateway.starts[0].wait()
        assert len(observed) == 1
        assert observed[0].library_authority.policy.assistant_access is (
            ConsoleAssistantLibraryAccess.ALLOWED
        )
        queue_snapshot = controller.prompt_queue_registry.snapshot(session.id)
        queued = controller.queue_prompt(
            session.id,
            text="queued",
            expected_revision=queue_snapshot.revision,
        )
        assert queued.applied is True

        second_db = CharactersRAGDB(path, "queue-second-process")
        second_repository = ConsoleLibraryPolicyRepository(second_db)
        assert second_repository.compare_and_swap(
            conversation_id,
            1,
            _allowed_candidate(allowed=False),
        ).status is ConsoleLibraryPolicyWriteStatus.COMMITTED
        # Enqueue did not capture policy; only the running first turn exists.
        assert len(observed) == 1
        gateway.releases[0].set()
        await gateway.starts[1].wait()

        assert len(observed) == 2
        assert observed[1].library_authority.policy.policy_revision == 2
        assert observed[1].library_authority.policy.assistant_access is (
            ConsoleAssistantLibraryAccess.BLOCKED
        )
        gateway.releases[1].set()
        await chain
    finally:
        if second_db is not None:
            second_db.close_connection()
        first_db.close_connection()


@pytest.mark.asyncio
async def test_direct_provider_boundary_rejects_configuration_only_snapshot():
    events: list[str] = []
    store = ConsoleChatStore()
    session = store.create_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    gateway = _RecordingGateway(events)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="model-a",
        ),
        tool_configuration={"agent_runtime_enabled": False},
    )
    resolution = await gateway.resolve_for_send(configuration.provider_selection)

    with pytest.raises(TypeError, match="complete ConsoleTurnExecutionContext"):
        await controller._stream_assistant_response_inner(
            resolution=resolution,
            provider_messages=[{"role": "user", "content": "question"}],
            assistant_message_id=assistant.id,
            turn_context=configuration,
        )

    assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_agent_boundary_rejects_configuration_before_library_composition():
    store = ConsoleChatStore()
    session = store.create_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    factory_calls: list[object] = []

    def library_factory(turn_context):
        factory_calls.append(turn_context)
        raise AssertionError("incomplete context reached Library composition")

    gateway = _RecordingGateway([])
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        agent_bridge=object(),
        library_provider_factory=library_factory,
    )
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="model-a",
        ),
        tool_configuration={"agent_runtime_enabled": True},
    )
    resolution = await gateway.resolve_for_send(configuration.provider_selection)

    with pytest.raises(TypeError, match="complete ConsoleTurnExecutionContext"):
        await controller._run_agent_reply(
            resolution=resolution,
            provider_messages=[{"role": "user", "content": "question"}],
            assistant_message_id=assistant.id,
            prepare_retry=False,
            variant_mode=False,
            turn_context=configuration,
        )

    assert factory_calls == []


def test_library_composition_rejects_configuration_only_snapshot():
    store = ConsoleChatStore()
    session = store.create_session()
    factory_calls: list[object] = []

    def library_factory(turn_context):
        factory_calls.append(turn_context)
        return object()

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway([]),
        library_provider_factory=library_factory,
    )
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=ConsoleProviderSelection(provider="openai"),
    )

    with pytest.raises(TypeError, match="complete ConsoleTurnExecutionContext"):
        controller._library_provider_for_context(configuration)

    assert factory_calls == []


@pytest.mark.asyncio
async def test_continuation_recovery_freshly_finalizes_before_agent_boundary(
    monkeypatch,
):
    database = CharactersRAGDB(":memory:", "task-19900-2-continuation")
    try:
        events: list[str] = []
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Use the calculator",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        monkeypatch.setattr(
            store,
            "ensure_provider_continuation_durable",
            lambda **_kwargs: SimpleNamespace(ready=True, reason="local_durable"),
        )
        checkpoint = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="moonshot",
            protocol="chat_completions",
            model="kimi-k2",
            api_base_url="https://api.moonshot.ai/v1",
            state="active",
            rounds=(
                ContinuationRound(
                    assistant_content="",
                    reasoning_blocks=(),
                    calls=(
                        ContinuationCall(
                            call_id="call-1",
                            name="calculator",
                            arguments='{"expression":"2+2"}',
                            state="pending",
                        ),
                    ),
                ),
            ),
        )
        store.persist_provider_continuation_event(
            ToolBatchReady(
                ContinuationEventContext(
                    assistant.id,
                    "run-1",
                    "primary",
                    "persistent",
                ),
                checkpoint,
                None,
            )
        )

        class UnavailableCoordinator:
            async def capture_for_execution(self, captured_session_id: str):
                assert captured_session_id == session.id
                events.append("policy")
                raise RuntimeError("durable policy unavailable")

        class ContinuationGateway:
            def expand_provider_continuation(self, _checkpoint):
                return []

            async def resolve_for_send(self, selection):
                events.append("gateway")
                return SimpleNamespace(
                    ready=True,
                    provider="Moonshot",
                    model="kimi-k2",
                    base_url="https://api.moonshot.ai/v1",
                    api_mode="chat_completions",
                    visible_copy="",
                    resolved_destination=ConsoleResolvedDestination(
                        provider="Moonshot",
                        model="kimi-k2",
                        endpoint_identity="https://api.moonshot.ai/v1",
                        egress_class=ConsoleEgressClass.UNKNOWN,
                    ),
                )

        def capture_configuration(session_id: str):
            events.append("configuration")
            return ConsoleTurnConfigurationSnapshot.capture(
                session_id=session_id,
                provider_selection=ConsoleProviderSelection(
                    provider="Moonshot",
                    explicit_model="kimi-k2",
                    base_url="https://api.moonshot.ai/v1",
                ),
                tool_configuration={"agent_runtime_enabled": True},
            )

        store.library_policy_coordinator = UnavailableCoordinator()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=ContinuationGateway(),
            agent_runtime_enabled=True,
            agent_bridge=object(),
            turn_context_provider=capture_configuration,
        )
        captured: dict[str, object] = {}

        async def assert_complete_agent_boundary(**kwargs):
            events.append("agent-boundary")
            turn_context = kwargs["turn_context"]
            assert isinstance(turn_context, ConsoleTurnExecutionContext)
            captured.update(kwargs)
            store._message_or_raise(assistant.id).provider_continuation = None
            return ConsoleSubmitResult(True, True, "done")

        monkeypatch.setattr(
            controller,
            "_run_agent_reply",
            assert_complete_agent_boundary,
        )
        version = store.get_message(
            assistant.id
        ).provider_continuation_message_version
        assert version is not None

        assert await controller.recover_provider_continuation(
            "resume",
            assistant.id,
            version,
        )

        assert events == [
            "configuration",
            "policy",
            "gateway",
            "agent-boundary",
        ]
        turn_context = captured["turn_context"]
        assert isinstance(turn_context, ConsoleTurnExecutionContext)
        assert turn_context.library_authority.policy.source == "unavailable"
        assert turn_context.library_authority.policy.auto_retrieve is (
            ConsoleAutoRetrieve.NEVER
        )
        assert turn_context.library_authority.policy.assistant_access is (
            ConsoleAssistantLibraryAccess.BLOCKED
        )
        assert turn_context.resolved_destination.provider == "Moonshot"
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_queued_retry_captures_complete_context_only_after_recovery_claim(
    monkeypatch,
):
    events: list[str] = []
    store = ConsoleChatStore()
    session = store.create_session()
    coordinator = _RecordingCoordinator(events, _policy())
    store.library_policy_coordinator = coordinator

    class QueueRetryGateway:
        def __init__(self):
            self.starts = [asyncio.Event() for _ in range(3)]
            self.releases = [asyncio.Event() for _ in range(3)]
            self.stream_calls = 0

        async def resolve_for_send(self, selection):
            events.append("gateway")
            return SimpleNamespace(
                ready=True,
                provider=selection.provider,
                model=selection.explicit_model,
                base_url=selection.base_url,
                visible_copy="",
                resolved_destination=_destination(selection.provider),
            )

        async def stream_chat(self, _resolution, _messages, **_kwargs):
            index = self.stream_calls
            self.stream_calls += 1
            self.starts[index].set()
            await self.releases[index].wait()
            yield f"reply-{index}"
            if index == 0:
                raise RuntimeError("planned first-turn failure")

    capture_reservations: list[PromptQueueReservation] = []

    def capture_configuration(session_id: str):
        events.append("configuration")
        capture_reservations.append(
            controller.prompt_queue_registry.snapshot(session_id).reservation
        )
        return ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider="openai",
                explicit_model="model-a",
            ),
            tool_configuration={"agent_runtime_enabled": False},
        )

    gateway = QueueRetryGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        turn_context_provider=capture_configuration,
        agent_runtime_enabled=False,
    )
    observed_contexts: list[ConsoleTurnExecutionContext] = []
    boundary_starts = [asyncio.Event() for _ in range(3)]
    real_inner = controller._stream_assistant_response_inner

    async def assert_complete_provider_boundary(**kwargs):
        turn_context = kwargs["turn_context"]
        observed_contexts.append(turn_context)
        boundary_starts[len(observed_contexts) - 1].set()
        assert isinstance(turn_context, ConsoleTurnExecutionContext)
        return await real_inner(**kwargs)

    monkeypatch.setattr(
        controller,
        "_stream_assistant_response_inner",
        assert_complete_provider_boundary,
    )

    first = asyncio.create_task(
        controller.run_prompt_chain("first", session_id=session.id)
    )
    await gateway.starts[0].wait()
    before_enqueue_config = len(capture_reservations)
    before_enqueue_policy = len(coordinator.calls)
    queue_snapshot = controller.prompt_queue_registry.snapshot(session.id)
    queued = controller.queue_prompt(
        session.id,
        text="after retry",
        expected_revision=queue_snapshot.revision,
    )
    assert queued.applied
    assert len(capture_reservations) == before_enqueue_config
    assert len(coordinator.calls) == before_enqueue_policy
    gateway.releases[0].set()
    await first
    failed = next(
        item
        for item in store.messages_for_session(session.id)
        if item.role is ConsoleMessageRole.ASSISTANT and item.status == "failed"
    )

    recovery = asyncio.create_task(controller.retry_failed_queue_turn(failed.id))
    await boundary_starts[1].wait()

    assert capture_reservations[-1] is PromptQueueReservation.HELD
    assert len(capture_reservations) == before_enqueue_config + 1
    assert len(coordinator.calls) == before_enqueue_policy + 1
    assert isinstance(observed_contexts[-1], ConsoleTurnExecutionContext)
    assert observed_contexts[-1].resolved_destination == _destination()

    await gateway.starts[1].wait()
    gateway.releases[1].set()
    await gateway.starts[2].wait()
    gateway.releases[2].set()
    await recovery

    assert len(coordinator.calls) == 3
    assert len({item.library_authority.attempt_id for item in observed_contexts}) == 3
