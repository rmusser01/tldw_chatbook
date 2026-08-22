"""Execution-time ordering and immutability for Console Library authority."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
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
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem


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
        yield "reply"


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
