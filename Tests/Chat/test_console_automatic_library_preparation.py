"""Task 13: automatic Library retrieval is a fail-closed send gate."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsolePreparationOutcome,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.library_preparation import LibraryPreparationContribution
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow


class _StreamingFence:
    def __init__(self) -> None:
        self.resolve_calls = 0
        self.provider_calls = 0
        self.messages = None

    async def resolve_for_send(self, _selection):
        self.resolve_calls += 1
        return ConsoleProviderResolution(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
            resolved_destination=ConsoleResolvedDestination(
                provider="llama_cpp",
                model="test-model",
                endpoint_identity="http://127.0.0.1:9099",
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
        )

    async def stream_chat(self, _resolution, messages, **_kwargs):
        self.provider_calls += 1
        self.messages = messages
        yield "ok"


class _BlockingFirstFence(_StreamingFence):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(self, _resolution, messages, **_kwargs):
        self.provider_calls += 1
        self.messages = messages
        self.started.set()
        await self.release.wait()
        yield "ok"


class _PolicyCoordinator:
    def __init__(self, auto_retrieve: ConsoleAutoRetrieve) -> None:
        self.auto_retrieve = auto_retrieve

    def register_holder(self, *_args, **_kwargs) -> None:
        return None

    def unregister_holder(self, *_args, **_kwargs) -> None:
        return None

    async def capture_for_execution(self, _session_id: str):
        return ConsoleLibraryPolicySnapshot(
            auto_retrieve=self.auto_retrieve,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=7,
            source="durable",
        )


class _RagService:
    def __init__(self, result=None, *, delay: float = 0.0, error=None) -> None:
        self.result = result if result is not None else {"results": []}
        self.delay = delay
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def search(self, query, source_types, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "source_types": tuple(source_types),
                "mode": mode,
                **kwargs,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result


class _HeldRagService(_RagService):
    def __init__(self) -> None:
        super().__init__({"results": []})
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def search(self, query, source_types, mode, **kwargs):
        self.calls.append({"query": query, "source_types": tuple(source_types), "mode": mode, **kwargs})
        self.entered.set()
        await self.release.wait()
        return self.result


def _row(content: str = "sealed evidence") -> LibraryRagResultRow:
    return LibraryRagResultRow.from_result(
        {
            "source_id": "note-1",
            "chunk_id": "chunk-1",
            "title": "Exact note",
            "content": content,
            "score": 0.9,
            "runtime_backend": "local",
            "source_type": "notes",
        }
    )


def _context(
    *,
    session_id: str = "session-1",
    attempt_id: str = "attempt-1",
    auto_retrieve: ConsoleAutoRetrieve = ConsoleAutoRetrieve.AUTOMATIC,
    scope: ConsoleLibraryItemScopeSnapshot | None = None,
) -> ConsoleTurnExecutionContext:
    return ConsoleTurnExecutionContext(
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider="llama_cpp", explicit_model="test-model"
            ),
        ),
        library_authority=ConsoleTurnLibraryAuthority(
            policy=ConsoleLibraryPolicySnapshot(
                auto_retrieve=auto_retrieve,
                assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                policy_revision=7,
                source="durable",
            ),
            direct_library_tools=True,
            source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
            scope_snapshot=scope
            or ConsoleLibraryItemScopeSnapshot((), (), True),
            provider_intent=ConsoleProviderIntent(
                "llama_cpp", "test-model", "http://127.0.0.1:9099"
            ),
            attempt_id=attempt_id,
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="llama_cpp",
            model="test-model",
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
    )


def _preparation(
    *,
    session_id: str = "session-1",
    preparation_id: str = "preparation-1",
    state: ConsoleTurnPreparationState = ConsoleTurnPreparationState.PREPARING,
    pause_kind: ConsolePreparationPauseKind | None = None,
    origin: str = "manual",
    queue_entry_id: str | None = None,
    queue_generation: int | None = None,
    draft: str = "exact executed draft",
    context: ConsoleTurnExecutionContext | None = None,
) -> ConsoleTurnPreparation:
    turn_context = context or _context(session_id=session_id)
    return ConsoleTurnPreparation(
        preparation_id=preparation_id,
        attempt_id=turn_context.library_authority.attempt_id,
        session_id=session_id,
        origin=origin,  # type: ignore[arg-type]
        queue_entry_id=queue_entry_id,
        executed_draft=draft,
        execution_context=turn_context,
        transient_user_message_id=None,
        attachment_ids=("attachment-1",),
        evidence_ids=("evidence-1",),
        prefill_id="prefill-1",
        queue_generation=queue_generation,
        pre_send_title="Chat 1",
        pre_send_conversation_id=None,
        state=state,
        pause_kind=pause_kind,
        one_shot_bypass=False,
        ephemeral=False,
    )


def _controller_for_preparation(
    preparation: ConsoleTurnPreparation,
    service: _RagService,
    *,
    timeout: float = 5.0,
) -> tuple[ConsoleChatController, ConsoleChatStore]:
    store = ConsoleChatStore()
    store.create_session(session_id=preparation.session_id, title="Chat 1")
    assert store.begin_preparation(preparation) is preparation
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        library_preparation_timeout=timeout,
    )
    controller.app = SimpleNamespace(library_rag_search_service=service)
    return controller, store


def test_store_preparation_cas_is_exact_and_survives_controller_replacement():
    store = ConsoleChatStore()
    store.create_session(session_id="session-1")
    preparation = _preparation()

    assert store.begin_preparation(preparation) is preparation
    assert store.begin_preparation(_preparation(preparation_id="other")) is None
    replacement = ConsoleChatController(store=store, provider_gateway=_StreamingFence())

    assert replacement.store.preparation_for_session("session-1") is preparation
    stale = ConsolePreparationTransition(
        preparation_id="other",
        expected_state=ConsoleTurnPreparationState.PREPARING,
        new_state=ConsoleTurnPreparationState.READY,
        pause_kind=None,
        new_attempt_id=None,
    )
    assert store.compare_and_set_preparation("session-1", stale) is None
    assert store.preparation_for_session("session-1") is preparation


def test_store_racing_actions_have_one_winner():
    store = ConsoleChatStore()
    store.create_session(session_id="session-1")
    paused = _preparation(
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )
    assert store.begin_preparation(paused) is paused
    barrier = threading.Barrier(4)
    wins: list[ConsoleTurnPreparation] = []
    lock = threading.Lock()

    transitions = (
        ConsolePreparationTransition(
            paused.preparation_id,
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.PREPARING,
            None,
            "attempt-retry",
        ),
        ConsolePreparationTransition(
            paused.preparation_id,
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.READY,
            None,
            None,
        ),
        ConsolePreparationTransition(
            paused.preparation_id,
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.CANCELLED,
            None,
            None,
        ),
    )

    def act(transition: ConsolePreparationTransition) -> None:
        barrier.wait()
        result = store.compare_and_set_preparation("session-1", transition)
        if result is not None:
            with lock:
                wins.append(result)

    threads = [threading.Thread(target=act, args=(transition,)) for transition in transitions]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(wins) == 1
    assert store.preparation_for_session("session-1") is wins[0]


@pytest.mark.asyncio
async def test_evidence_success_uses_exact_draft_fixed_categories_and_exact_bundle():
    preparation = _preparation(draft="immutable executed draft")
    service = _RagService({"runtime_backend": "local", "results": [_row()]})
    controller, store = _controller_for_preparation(preparation, service)

    outcome = await controller.prepare_library_for_turn(preparation.preparation_id)

    assert isinstance(outcome, ConsolePreparationOutcome)
    assert outcome.state is ConsoleTurnPreparationState.READY
    assert outcome.evidence_bundle is not None
    assert outcome.evidence_bundle is controller.preparation_outcome(
        preparation.preparation_id
    ).evidence_bundle
    assert outcome.contribution is None
    assert service.calls == [
        {
            "query": "immutable executed draft",
            "source_types": AUTOMATIC_LIBRARY_SOURCE_TYPES,
            "mode": "rag",
            "top_k": 5,
            "include_citations": True,
        }
    ]
    assert store.preparation_for_session("session-1").state is ConsoleTurnPreparationState.READY


@pytest.mark.asyncio
async def test_active_scope_uses_exact_note_media_allowlist_and_excludes_conversations():
    context = _context(
        scope=ConsoleLibraryItemScopeSnapshot(
            note_ids=("note-2", "note-1"),
            media_ids=("media-1",),
            conversations_allowed=False,
        )
    )
    service = _RagService({"results": []})
    controller, _store = _controller_for_preparation(
        _preparation(context=context), service
    )

    await controller.prepare_library_for_turn("preparation-1")

    scope = service.calls[0]["scope"]
    assert scope.state == "scoped"
    assert scope.allowlist == {
        "notes": frozenset({"note-1", "note-2"}),
        "media": frozenset({"media-1"}),
    }
    assert "conversations" not in scope.allowlist


@pytest.mark.asyncio
async def test_zero_matches_readies_with_one_bounded_contribution():
    controller, store = _controller_for_preparation(
        _preparation(), _RagService({"runtime_backend": "local", "results": []})
    )

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.state is ConsoleTurnPreparationState.READY
    assert outcome.evidence_bundle is None
    assert isinstance(outcome.contribution, LibraryPreparationContribution)
    assert outcome.contribution.event.outcome == "zero_matches"
    assert outcome.contribution.event.result_count == 0
    assert outcome.contribution.event.source_types == AUTOMATIC_LIBRARY_SOURCE_TYPES
    assert store.preparation_for_session("session-1").state is ConsoleTurnPreparationState.READY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "timeout", "error_code"),
    [
        (_RagService(error=RuntimeError("secret exception body")), 5.0, "library_retrieval_failed"),
        (_RagService(delay=0.05), 0.001, "library_retrieval_timeout"),
    ],
)
async def test_failure_and_timeout_pause_with_bounded_error(service, timeout, error_code):
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=timeout
    )

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.state is ConsoleTurnPreparationState.PAUSED
    assert outcome.error_code == error_code
    assert "secret" not in repr(outcome)
    paused = store.preparation_for_session("session-1")
    assert paused.pause_kind is ConsolePreparationPauseKind.RETRIEVAL


@pytest.mark.asyncio
async def test_retry_reuses_frozen_request_and_destination_with_new_attempt():
    original_context = _context(attempt_id="attempt-original")
    paused = _preparation(
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
        context=original_context,
        draft="frozen query",
    )
    service = _RagService({"results": []})
    controller, store = _controller_for_preparation(paused, service)
    store.set_session_draft(paused.session_id, "refreshed composer")

    outcome = await controller.retry_library_preparation(paused.preparation_id)

    retried = store.preparation_for_session("session-1")
    assert outcome.state is ConsoleTurnPreparationState.READY
    assert outcome.attempt_id != "attempt-original"
    assert retried.preparation_id == paused.preparation_id
    assert retried.executed_draft == paused.executed_draft
    assert retried.execution_context.configuration == paused.execution_context.configuration
    assert retried.execution_context.resolved_destination == paused.execution_context.resolved_destination
    assert retried.execution_context.library_authority.policy == paused.execution_context.library_authority.policy
    assert service.calls[0]["query"] == "frozen query"


def test_bypass_readies_with_contribution_without_mutating_policy():
    paused = _preparation(
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )
    controller, store = _controller_for_preparation(paused, _RagService())
    original_policy = paused.execution_context.library_authority.policy

    outcome = controller.bypass_library_preparation(paused.preparation_id)

    ready = store.preparation_for_session("session-1")
    assert outcome.state is ConsoleTurnPreparationState.READY
    assert outcome.contribution.event.outcome == "bypassed"
    assert ready.one_shot_bypass is True
    assert ready.execution_context.library_authority.policy is original_policy


def test_manual_cancel_preserves_staged_state_and_removes_only_transient_echo():
    store = ConsoleChatStore()
    session = store.create_session(session_id="session-1", title="Original title")
    store.set_session_draft(session.id, "exact draft")
    store.set_session_one_shot_prefill(session.id, "exact prefill")
    survivor = store.append_message(
        session.id, role=ConsoleMessageRole.SYSTEM, content="keep me"
    )
    transient = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="exact draft",
        persist=False,
    )
    preparation = _preparation(draft="exact draft")
    preparation = ConsoleTurnPreparation(
        **{
            name: getattr(preparation, name)
            for name in preparation.__dataclass_fields__
            if name != "transient_user_message_id"
        },
        transient_user_message_id=transient.id,
    )
    assert store.begin_preparation(preparation) is preparation

    cancelled = store.cancel_preparation(
        session.id,
        preparation.preparation_id,
        expected_state=ConsoleTurnPreparationState.PREPARING,
    )

    assert cancelled.state is ConsoleTurnPreparationState.CANCELLED
    assert store.session_draft(session.id) == "exact draft"
    assert store.session_one_shot_prefill(session.id) == "exact prefill"
    assert store.messages_for_session(session.id) == [survivor]


@pytest.mark.asyncio
async def test_retrieval_failure_never_dispatches_provider():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _StreamingFence()
    service = _RagService(error=RuntimeError("backend unavailable"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    result = await controller.submit_draft("exact user text")

    assert not result.accepted
    assert gateway.resolve_calls == 1
    assert gateway.provider_calls == 0
    preparation = store.preparation_for_session(store.active_session_id)
    assert preparation.state is ConsoleTurnPreparationState.PAUSED
    assert preparation.executed_draft == "exact user text"


@pytest.mark.asyncio
async def test_success_injects_the_sealed_bundle_into_the_same_dispatched_request():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _StreamingFence()
    service = _RagService({"runtime_backend": "local", "results": [_row("needle body")]})
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    result = await controller.submit_draft("exact user text")

    assert result.accepted
    assert gateway.provider_calls == 1
    final_user = next(row for row in reversed(gateway.messages) if row["role"] == "user")
    assert "needle body" in final_user["content"]
    preparation = store.preparation_for_session(result.session_id)
    outcome = controller.preparation_outcome(preparation.preparation_id)
    assert outcome.evidence_bundle.query == "exact user text"
    assert preparation.state is ConsoleTurnPreparationState.SETTLED


@pytest.mark.asyncio
async def test_never_and_nonordinary_text_skip_automatic_retrieval():
    for policy, draft in (
        (ConsoleAutoRetrieve.NEVER, "ordinary text"),
        (ConsoleAutoRetrieve.AUTOMATIC, "/command"),
        (ConsoleAutoRetrieve.AUTOMATIC, "$mention"),
    ):
        store = ConsoleChatStore()
        store.library_policy_coordinator = _PolicyCoordinator(policy)
        service = _RagService(error=AssertionError("automatic retrieval spent"))
        gateway = _StreamingFence()
        controller = ConsoleChatController(store=store, provider_gateway=gateway)
        controller.app = SimpleNamespace(library_rag_search_service=service)

        result = await controller.submit_draft(draft)

        assert result.accepted
        assert service.calls == []
        assert gateway.provider_calls == 1


@pytest.mark.asyncio
async def test_explicit_staged_evidence_skips_duplicate_automatic_retrieval():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    service = _RagService(error=AssertionError("duplicate automatic retrieval"))
    gateway = _StreamingFence()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        staged_evidence_provider=lambda _session_id: True,
    )
    controller.app = SimpleNamespace(library_rag_search_service=service)

    result = await controller.submit_draft("ordinary text")

    assert result.accepted
    assert service.calls == []
    assert gateway.provider_calls == 1


def test_close_cancels_preparation_through_same_store_path():
    store = ConsoleChatStore()
    store.create_session(session_id="session-1")
    preparation = _preparation()
    assert store.begin_preparation(preparation) is preparation

    store.close_session("session-1")

    assert store.preparation_for_session("session-1") is None


@pytest.mark.asyncio
async def test_close_wins_race_with_inflight_retrieval_without_dispatch():
    service = _HeldRagService()
    preparation = _preparation()
    controller, store = _controller_for_preparation(preparation, service)

    running = asyncio.create_task(
        controller.prepare_library_for_turn(preparation.preparation_id)
    )
    await service.entered.wait()
    store.close_session(preparation.session_id)
    service.release.set()
    outcome = await running

    assert outcome.state is ConsoleTurnPreparationState.CANCELLED
    assert store.preparation_for_session(preparation.session_id) is None


@pytest.mark.asyncio
async def test_shutdown_uses_exact_store_cancel_without_provider_dispatch():
    paused = _preparation(
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )
    controller, store = _controller_for_preparation(paused, _RagService())

    await controller.shutdown()

    cancelled = store.preparation_for_session(paused.session_id)
    assert cancelled is not None
    assert cancelled.preparation_id == paused.preparation_id
    assert cancelled.state is ConsoleTurnPreparationState.CANCELLED
    assert controller.provider_gateway.provider_calls == 0


@pytest.mark.asyncio
async def test_queued_failure_returns_exact_claim_without_foreground_copy():
    store = ConsoleChatStore()
    policy = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    store.library_policy_coordinator = policy
    session = store.create_session(session_id="session-1")
    store.set_session_draft(session.id, "foreground stays here")
    gateway = _BlockingFirstFence()
    service = _RagService(error=RuntimeError("queued retrieval failed"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    chain = asyncio.create_task(
        controller.run_prompt_chain("accepted owner", session_id=session.id)
    )
    await gateway.started.wait()
    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    admitted = controller.queue_prompt(
        session.id,
        text="exact queued body",
        expected_revision=snapshot.revision,
    )
    assert admitted.applied
    entry_id = admitted.entry_id
    policy.auto_retrieve = ConsoleAutoRetrieve.AUTOMATIC
    gateway.release.set()

    result = await chain

    paused = store.preparation_for_session(session.id)
    assert paused is not None
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert controller.cancel_library_preparation(paused.preparation_id)
    after_cancel = controller.prompt_queue_registry.snapshot(session.id)
    assert result.accepted
    assert after_cancel.claimed_count == 0
    assert after_cancel.waiting_count == 1
    assert after_cancel.entries[0].entry_id == entry_id
    assert store.session_draft(session.id) == "foreground stays here"
    assert gateway.provider_calls == 1
