"""Task 13: automatic Library retrieval is a fail-closed send gate."""

from __future__ import annotations

import asyncio
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest

from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsolePreparationOutcome,
    ConsoleSubmitResult,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
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
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_prompt_queue import (
    PromptQueueMode,
    PromptQueueReservation,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
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
from tldw_chatbook.UI.Console_Modules import retrieval as retrieval_module
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController
from tldw_chatbook.UI.Views.RAGSearch.search_handoff import (
    build_library_rag_evidence_bundle,
)


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


class _DestinationSequenceFence(_StreamingFence):
    def __init__(self, endpoints: tuple[str, ...]) -> None:
        super().__init__()
        self.endpoints = list(endpoints)

    async def resolve_for_send(self, _selection):
        self.resolve_calls += 1
        endpoint = self.endpoints.pop(0)
        return ConsoleProviderResolution(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url=endpoint,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
            resolved_destination=ConsoleResolvedDestination(
                provider="llama_cpp",
                model="test-model",
                endpoint_identity=endpoint,
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
        )


class _ResolverFailureFence(_StreamingFence):
    async def resolve_for_send(self, selection):
        if self.resolve_calls:
            self.resolve_calls += 1
            raise RuntimeError("destination unreadable")
        return await super().resolve_for_send(selection)


class _MissingDestinationFence(_StreamingFence):
    async def resolve_for_send(self, _selection):
        self.resolve_calls += 1
        return SimpleNamespace(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            visible_copy="",
        )


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


class _PrepareFailureFence(_StreamingFence):
    def __init__(self, store: ConsoleChatStore) -> None:
        super().__init__()
        self.store = store
        self.observed_state = None

    def prepare_chat_request(self, *_args, **_kwargs):
        preparation = self.store.preparation_for_session(self.store.active_session_id)
        self.observed_state = preparation.state if preparation is not None else None
        raise RuntimeError("direct preparation failed")


class _StateCapturingAgentBridge:
    def __init__(self, store: ConsoleChatStore) -> None:
        self.store = store
        self.observed_state = None
        self.calls = 0

    def run_reply(self, **_kwargs):
        self.calls += 1
        preparation = self.store.preparation_for_session(self.store.active_session_id)
        self.observed_state = preparation.state if preparation is not None else None
        return "run-1", RunOutcome(status=RUN_DONE, steps=[], final_text="agent ok")


class _CancellationResistantBoundary:
    """Hold one awaited production seam even after its task is cancelled."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()


class _CancellationBoundary:
    """Hold an awaited production seam until ordinary task cancellation."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.entered.set()
        await self.release.wait()


class _OwnerThreadTask(asyncio.Task):
    """Test task that rejects cancellation from outside its owner thread."""

    def __init__(self, coroutine, *, loop: asyncio.AbstractEventLoop) -> None:
        self._owner_thread_id = threading.get_ident()
        super().__init__(coroutine, loop=loop)

    def cancel(self, msg=None) -> bool:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("Task.cancel called outside the owner thread")
        return super().cancel(msg)


class _HeldResolveFence(_StreamingFence):
    def __init__(self, boundary) -> None:
        super().__init__()
        self.boundary = boundary

    async def resolve_for_send(self, selection):
        await self.boundary.wait()
        return await super().resolve_for_send(selection)


class _TwoResolveFence(_StreamingFence):
    def __init__(self) -> None:
        super().__init__()
        self.entered_count = 0
        self.both_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def resolve_for_send(self, selection):
        self.entered_count += 1
        if self.entered_count == 2:
            self.both_entered.set()
        await self.release.wait()
        return await super().resolve_for_send(selection)


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
        self.calls.append(
            {
                "query": query,
                "source_types": tuple(source_types),
                "mode": mode,
                **kwargs,
            }
        )
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
            scope_snapshot=scope or ConsoleLibraryItemScopeSnapshot((), (), True),
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


def _pending_image(name: str, data: bytes) -> PendingAttachment:
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=data,
        mime_type="image/png",
        original_size=len(data),
        processed_size=len(data),
    )


def _real_retrieval_controller_for_launch(state: dict[str, object]):
    def consume_launch():
        return state.get("launch")

    def release_launch(launch, result):
        state.setdefault("released", []).append((launch, result))
        if state.get("launch") is launch:
            state["launch"] = None

    controller = ConsoleRetrievalController(
        app_instance=SimpleNamespace(),
        active_native_session=lambda: None,
        current_conversation_id=lambda: None,
        clear_evidence_sent_notice=lambda: None,
        consume_pending_launch=consume_launch,
        release_consumed_launch=release_launch,
        is_mounted=lambda: False,
        sync_retrieval_scope_row=lambda: None,
        sync_control_bar=lambda: None,
        request_control_bar_sync=lambda: None,
        dictionary_scope_service=lambda: None,
        set_library_rag_source_scope=lambda _value: None,
        set_library_rag_query=lambda _value: None,
        run_library_rag_action=lambda: None,
        library_rag_source_scope=lambda: (),
        library_rag_top_k=lambda: 10,
        pending_launch=lambda: state.get("launch"),
        set_pending_launch=lambda launch: state.__setitem__("launch", launch),
        set_pending_auto_open=lambda _value: None,
        set_evidence_sent_notice=lambda _value: None,
        sync_pending_launch_surfaces=lambda: True,
        refresh_screen=lambda: None,
        has_staged_evidence=lambda: state.get("launch") is not None,
    )
    return controller


async def _paused_queued_send(*, persistence=None):
    store = ConsoleChatStore(persistence=persistence)
    policy = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    store.library_policy_coordinator = policy
    session = store.create_session(session_id="session-1")
    gateway = _BlockingFirstFence()
    service = _RagService(error=RuntimeError("queued retrieval failed"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    chain = asyncio.create_task(
        controller.run_prompt_chain("owner", session_id=session.id)
    )
    await gateway.started.wait()
    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    first = controller.queue_prompt(
        session.id, text="frozen queued", expected_revision=snapshot.revision
    )
    second = controller.queue_prompt(
        session.id,
        text="later queued",
        expected_revision=first.snapshot.revision,
    )
    policy.auto_retrieve = ConsoleAutoRetrieve.AUTOMATIC
    gateway.release.set()
    await chain
    preparation = store.preparation_for_session(session.id)
    assert preparation is not None
    assert preparation.queue_entry_id == first.entry_id
    return controller, store, gateway, service, preparation, first, second


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

    threads = [
        threading.Thread(target=act, args=(transition,)) for transition in transitions
    ]
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
    assert (
        outcome.evidence_bundle
        is controller.preparation_outcome(preparation.preparation_id).evidence_bundle
    )
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
    assert (
        store.preparation_for_session("session-1").state
        is ConsoleTurnPreparationState.READY
    )


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
    assert (
        store.preparation_for_session("session-1").state
        is ConsoleTurnPreparationState.READY
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "timeout", "error_code"),
    [
        (
            _RagService(error=RuntimeError("secret exception body")),
            5.0,
            "library_retrieval_failed",
        ),
        (_RagService(delay=0.05), 0.001, "library_retrieval_timeout"),
    ],
)
async def test_failure_and_timeout_pause_with_bounded_error(
    service, timeout, error_code
):
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=timeout
    )

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.state is ConsoleTurnPreparationState.PAUSED
    assert outcome.error_code == error_code
    assert "secret" not in repr(outcome)
    paused = store.preparation_for_session("session-1")
    assert paused.pause_kind is ConsolePreparationPauseKind.RETRIEVAL


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


def test_controller_cancel_removes_exact_owner_and_sidecars_without_touching_staged_inputs():
    store = ConsoleChatStore()
    session = store.create_session(session_id="session-1", title="Original title")
    store.set_session_draft(session.id, "exact draft")
    store.set_session_one_shot_prefill(session.id, "exact prefill")
    attachment = PendingAttachment(
        file_path="/tmp/exact.txt",
        display_name="exact.txt",
        file_type="document",
        insert_mode="attachment",
        data=b"exact attachment",
        mime_type="text/plain",
        original_size=16,
        processed_size=16,
    )
    assert store.add_pending_attachment(session.id, attachment)
    evidence_bundle = build_library_rag_evidence_bundle(
        [_row("exact evidence")], query="exact query"
    )
    explicit_evidence = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Exact staged evidence",
        payload={"evidence_bundle": evidence_bundle.to_payload()},
        status="ready",
    )
    evidence_state: dict[str, object] = {
        "launch": explicit_evidence,
        "released": [],
    }
    retrieval = _real_retrieval_controller_for_launch(evidence_state)
    survivor = store.append_message(
        session.id, role=ConsoleMessageRole.SYSTEM, content="keep me"
    )
    transient = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="exact draft", persist=False
    )
    base = _preparation(
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
        draft="exact draft",
    )
    preparation = ConsoleTurnPreparation(
        **{
            name: getattr(base, name)
            for name in base.__dataclass_fields__
            if name
            not in {"transient_user_message_id", "attachment_ids", "evidence_ids"}
        },
        transient_user_message_id=transient.id,
        attachment_ids=(attachment.attachment_id,),
        evidence_ids=("exact-evidence",),
    )
    assert store.begin_preparation(preparation) is preparation
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        rag_capture_provider=retrieval._capture_console_staged_rag,
    )
    controller._preparation_outcomes[preparation.preparation_id] = (
        ConsolePreparationOutcome(
            preparation_id=preparation.preparation_id,
            attempt_id=preparation.attempt_id,
            state=preparation.state,
            evidence_bundle=None,
            contribution=None,
            error_code="library_retrieval_failed",
        )
    )
    controller._prepared_send_continuations[preparation.preparation_id] = object()

    cancelled = controller.cancel_library_preparation(preparation.preparation_id)

    assert isinstance(cancelled, ConsoleSubmitResult)
    assert not cancelled.accepted
    assert store.preparation_for_session(session.id) is None
    assert controller.preparation_outcome(preparation.preparation_id) is None
    assert controller._prepared_send_continuations == {}
    assert store.session_draft(session.id) == "exact draft"
    assert store.session_one_shot_prefill(session.id) == "exact prefill"
    assert store.pending_attachments(session.id) == [attachment]
    assert store.pending_attachments(session.id)[0] is attachment
    assert evidence_state["launch"] is explicit_evidence
    assert evidence_state["released"] == []
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
    service = _RagService(
        {"runtime_backend": "local", "results": [_row("needle body")]}
    )
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    result = await controller.submit_draft("exact user text")

    assert result.accepted
    assert gateway.provider_calls == 1
    final_user = next(
        row for row in reversed(gateway.messages) if row["role"] == "user"
    )
    assert "needle body" in final_user["content"]
    assert store.preparation_for_session(result.session_id) is None
    assert controller._preparation_outcomes == {}


@pytest.mark.asyncio
async def test_live_state_reaches_dispatch_started_only_at_provider_attempt():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _BlockingFirstFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("exact user text"))
    await gateway.started.wait()

    preparation = store.preparation_for_session(store.active_session_id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.DISPATCH_STARTED
    assert gateway.provider_calls == 1

    gateway.release.set()
    result = await task
    assert result.accepted
    assert store.preparation_for_session(result.session_id) is None


@pytest.mark.asyncio
async def test_close_during_provider_attempt_settles_live_preparation_without_leak():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _BlockingFirstFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("exact user text"))
    await gateway.started.wait()
    session_id = store.active_session_id
    preparation = store.preparation_for_session(session_id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.DISPATCH_STARTED
    assert session_id in controller._active_stream_tasks
    assert session_id in controller._active_assistant_message_ids

    controller.close_session(session_id)
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert store.preparation_for_session(session_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}


@pytest.mark.asyncio
async def test_repeated_successes_leave_no_preparation_or_outcome_accumulation():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _StreamingFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=_RagService())

    for index in range(25):
        result = await controller.submit_draft(f"turn {index}")
        assert result.accepted

    assert gateway.provider_calls == 25
    assert store.preparation_for_session(store.active_session_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}


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


def test_store_close_keeps_accepted_preparation_until_live_turn_settles():
    store = ConsoleChatStore()
    store.create_session(session_id="session-1")
    accepted = _preparation(state=ConsoleTurnPreparationState.ACCEPTED)
    assert store.begin_preparation(accepted) is accepted

    store.close_session("session-1")

    assert store.preparation_for_session("session-1") is accepted
    assert (
        store.remove_preparation(
            "session-1",
            accepted.preparation_id,
            expected_states=frozenset({ConsoleTurnPreparationState.ACCEPTED}),
        )
        is accepted
    )


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
    assert controller._preparation_outcomes == {}


@pytest.mark.asyncio
async def test_shutdown_uses_exact_store_cancel_without_provider_dispatch():
    paused = _preparation(
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )
    controller, store = _controller_for_preparation(paused, _RagService())

    await controller.shutdown()

    assert store.preparation_for_session(paused.session_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}
    assert controller.provider_gateway.provider_calls == 0


def test_controller_close_removes_cancellable_preparation_sidecars():
    paused = _preparation(
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )
    controller, store = _controller_for_preparation(paused, _RagService())
    controller._preparation_outcomes[paused.preparation_id] = ConsolePreparationOutcome(
        preparation_id=paused.preparation_id,
        attempt_id=paused.attempt_id,
        state=paused.state,
        evidence_bundle=None,
        contribution=None,
        error_code="library_retrieval_failed",
    )
    controller._prepared_send_continuations[paused.preparation_id] = object()

    controller.close_session(paused.session_id)

    assert store.preparation_for_session(paused.session_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}


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


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
async def test_manual_recovery_continues_same_frozen_send_without_second_submit(action):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    session = store.create_session(session_id="session-1")
    store.set_session_draft(session.id, "composer changed after admission")
    store.set_session_one_shot_prefill(session.id, "exact prefill")
    gateway = _DestinationSequenceFence(
        ("http://127.0.0.1:9099", "http://127.0.0.1:9099")
    )
    service = _RagService(error=RuntimeError("first search failed"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    first = await controller.submit_draft(
        "frozen admitted draft", session_id=session.id
    )
    paused = store.preparation_for_session(session.id)
    assert not first.accepted
    assert paused is not None
    original = paused
    service.error = None
    service.result = {"runtime_backend": "local", "results": [_row("retry evidence")]}

    if action == "retry":
        recovered = await controller.retry_library_preparation(paused.preparation_id)
    else:
        recovered = await controller.bypass_library_preparation(paused.preparation_id)

    assert isinstance(recovered, ConsoleSubmitResult)
    assert recovered.accepted
    assert gateway.provider_calls == 1
    assert gateway.resolve_calls == 2
    assert service.calls[-1]["query"] == original.executed_draft
    assert (
        original.execution_context.configuration.provider_selection.provider
        == "llama_cpp"
    )
    assert (
        original.execution_context.resolved_destination.endpoint_identity
        == "http://127.0.0.1:9099"
    )
    assert "frozen admitted draft" in repr(gateway.messages)
    if action == "retry":
        assert "retry evidence" in repr(gateway.messages)
    else:
        assert len(service.calls) == 1
        assert (
            original.execution_context.library_authority.policy.auto_retrieve
            is ConsoleAutoRetrieve.AUTOMATIC
        )
    assert store.preparation_for_session(session.id) is None
    assert controller.preparation_outcome(paused.preparation_id) is None


@pytest.mark.asyncio
async def test_text_with_attachment_is_automatic_eligible_and_preserved_until_ready(
    monkeypatch,
):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda *_args: True)
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    session = store.create_session(session_id="session-1")
    attachment = PendingAttachment(
        file_path="/tmp/exact.png",
        display_name="exact.png",
        file_type="image",
        insert_mode="attachment",
        data=b"exact-image",
        mime_type="image/png",
        original_size=11,
        processed_size=11,
    )
    assert store.add_pending_attachment(session.id, attachment)
    service = _RagService(error=RuntimeError("pause with attachment"))
    gateway = _StreamingFence()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="vision-model"
    )
    controller.app = SimpleNamespace(library_rag_search_service=service)

    first = await controller.submit_draft("describe attachment", session_id=session.id)

    assert not first.accepted
    assert service.calls[0]["query"] == "describe attachment"
    assert store.pending_attachments(session.id)[0] is attachment
    assert gateway.provider_calls == 0


@pytest.mark.asyncio
async def test_evidence_probe_error_skips_duplicate_automatic_spend():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _StreamingFence()
    service = _RagService(error=AssertionError("must not duplicate retrieve"))
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        staged_evidence_provider=lambda _session_id: (_ for _ in ()).throw(
            RuntimeError()
        ),
    )
    controller.app = SimpleNamespace(library_rag_search_service=service)

    result = await controller.submit_draft("ordinary text")

    assert result.accepted
    assert service.calls == []
    assert gateway.provider_calls == 1


@pytest.mark.asyncio
async def test_missing_typed_destination_fails_closed_without_dispatch():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _MissingDestinationFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    result = await controller.submit_draft("ordinary text")

    assert not result.accepted
    assert gateway.provider_calls == 0
    assert store.preparation_for_session(store.active_session_id) is None


@pytest.mark.asyncio
async def test_recovery_destination_change_pauses_without_dispatch():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _DestinationSequenceFence(
        ("http://127.0.0.1:9099", "http://127.0.0.1:9199")
    )
    service = _RagService(error=RuntimeError("first search failed"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)
    first = await controller.submit_draft("frozen draft")
    paused = store.preparation_for_session(first.session_id)
    service.error = None

    recovered = await controller.retry_library_preparation(paused.preparation_id)

    assert not recovered.accepted
    current = store.preparation_for_session(first.session_id)
    assert current.state is ConsoleTurnPreparationState.PAUSED
    assert current.pause_kind is ConsolePreparationPauseKind.DESTINATION_CHANGED
    assert gateway.provider_calls == 0


@pytest.mark.asyncio
async def test_skill_refusal_cleans_preparation_and_allows_next_send(monkeypatch):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _StreamingFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    async def refuse(_messages):
        return _messages, "skill refused", (), (), None

    monkeypatch.setattr(controller, "_apply_skill_substitution", refuse)
    first = await controller.submit_draft("first")

    assert not first.accepted
    assert store.preparation_for_session(first.session_id) is None
    monkeypatch.undo()
    second = await controller.submit_draft("second", session_id=first.session_id)
    assert second.accepted
    assert gateway.provider_calls == 1


@pytest.mark.asyncio
async def test_user_persistence_failure_rolls_back_preaccept_preparation(monkeypatch):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    session = store.create_session(session_id="session-1")
    store.set_session_draft(session.id, "exact draft")
    gateway = _StreamingFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    def fail_user_persistence(_message_id):
        raise RuntimeError("user persistence failed")

    monkeypatch.setattr(store, "persist_message_if_needed", fail_user_persistence)

    with pytest.raises(RuntimeError, match="user persistence failed"):
        await controller.submit_draft("exact draft", session_id=session.id)

    assert store.preparation_for_session(session.id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}
    assert store.session_draft(session.id) == "exact draft"
    assert not any(
        row.role is ConsoleMessageRole.USER and row.content == "exact draft"
        for row in store.messages_for_session(session.id)
    )
    assert gateway.provider_calls == 0


@pytest.mark.asyncio
async def test_assistant_acceptance_failure_rolls_back_preparation_and_transient_echo(
    monkeypatch,
):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    session = store.create_session(session_id="session-1")
    store.set_session_draft(session.id, "exact draft")
    gateway = _StreamingFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    original_append = store.append_message

    def fail_assistant(*args, **kwargs):
        if kwargs.get("role") is ConsoleMessageRole.ASSISTANT:
            raise RuntimeError("assistant acceptance failed")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(store, "append_message", fail_assistant)

    with pytest.raises(RuntimeError, match="assistant acceptance failed"):
        await controller.submit_draft("exact draft", session_id=session.id)

    assert store.preparation_for_session(session.id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}
    assert store.session_draft(session.id) == "exact draft"
    assert not any(
        row.role is ConsoleMessageRole.USER and row.content == "exact draft"
        for row in store.messages_for_session(session.id)
    )
    assert gateway.provider_calls == 0


@pytest.mark.asyncio
async def test_provider_preflight_refusal_never_claims_dispatch_started_or_wedges_next_send(
    monkeypatch,
):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _StreamingFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    observed_states: list[ConsoleTurnPreparationState] = []

    async def refuse_preflight(
        *, session_id, provider_messages, assistant_message_id, **_kwargs
    ):
        preparation = store.preparation_for_session(session_id)
        assert preparation is not None
        observed_states.append(preparation.state)
        return provider_messages, controller._block_context_preflight(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            visible_copy="preflight refused",
        )

    monkeypatch.setattr(
        controller, "_apply_conversation_memory_preflight", refuse_preflight
    )

    refused = await controller.submit_draft("first")

    assert refused.accepted
    assert observed_states == [ConsoleTurnPreparationState.ACCEPTED]
    assert gateway.provider_calls == 0
    assert store.preparation_for_session(refused.session_id) is None
    assert controller._preparation_outcomes == {}
    monkeypatch.undo()
    second = await controller.submit_draft("second", session_id=refused.session_id)
    assert second.accepted
    assert gateway.provider_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
async def test_queued_recovery_reclaims_same_entry_then_advances_without_spin(action):
    store = ConsoleChatStore()
    policy = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    store.library_policy_coordinator = policy
    session = store.create_session(session_id="session-1")
    gateway = _BlockingFirstFence()
    service = _RagService(error=RuntimeError("queued retrieval failed"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    chain = asyncio.create_task(
        controller.run_prompt_chain("owner", session_id=session.id)
    )
    await gateway.started.wait()
    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    first = controller.queue_prompt(
        session.id, text="frozen queued", expected_revision=snapshot.revision
    )
    snapshot = first.snapshot
    second = controller.queue_prompt(
        session.id, text="later queued", expected_revision=snapshot.revision
    )
    policy.auto_retrieve = ConsoleAutoRetrieve.AUTOMATIC
    gateway.release.set()
    await chain

    paused = store.preparation_for_session(session.id)
    before = controller.prompt_queue_registry.snapshot(session.id)
    assert paused.queue_entry_id == first.entry_id
    assert before.waiting_count == 2
    assert before.entries[0].entry_id == first.entry_id
    assert before.entries[1].entry_id == second.entry_id
    assert gateway.provider_calls == 1
    service.error = None
    service.result = {"results": []}

    if action == "retry":
        recovered = await controller.retry_library_preparation(paused.preparation_id)
    else:
        recovered = await controller.bypass_library_preparation(paused.preparation_id)

    assert recovered.accepted
    final = controller.prompt_queue_registry.snapshot(session.id)
    assert final.total_count == 0
    assert gateway.provider_calls == 3
    assert store.preparation_for_session(session.id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
@pytest.mark.parametrize("refusal", ["skill", "authorization"])
async def test_queued_recovery_refusal_returns_exact_claim_to_head_without_spin(
    monkeypatch, action, refusal
):
    (
        controller,
        store,
        gateway,
        service,
        paused,
        first,
        second,
    ) = await _paused_queued_send()
    service.error = None

    if refusal == "skill":

        async def refuse(messages):
            return messages, "skill refused", (), (), None

        monkeypatch.setattr(controller, "_apply_skill_substitution", refuse)
    else:
        monkeypatch.setattr(
            controller.prompt_queue_coordinator, "authorizes", lambda *_args: False
        )

    if action == "retry":
        result = await controller.retry_library_preparation(paused.preparation_id)
    else:
        result = await controller.bypass_library_preparation(paused.preparation_id)

    assert isinstance(result, ConsoleSubmitResult)
    assert not result.accepted
    snapshot = controller.prompt_queue_registry.snapshot(paused.session_id)
    assert snapshot.claimed_count == 0
    assert snapshot.waiting_count == 2
    assert [entry.entry_id for entry in snapshot.entries] == [
        first.entry_id,
        second.entry_id,
    ]
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.reservation is PromptQueueReservation.RELEASED
    assert gateway.provider_calls == 1
    assert store.preparation_for_session(paused.session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
async def test_queued_recovery_preaccept_exception_returns_exact_claim_to_head(
    monkeypatch, action
):
    (
        controller,
        store,
        gateway,
        service,
        paused,
        first,
        second,
    ) = await _paused_queued_send()
    service.error = None

    async def fail_composition(_messages, _session_id):
        raise RuntimeError("composition failed")

    monkeypatch.setattr(controller, "_apply_chat_dictionaries", fail_composition)

    if action == "retry":
        result = await controller.retry_library_preparation(paused.preparation_id)
    else:
        result = await controller.bypass_library_preparation(paused.preparation_id)

    assert isinstance(result, ConsoleSubmitResult)
    assert not result.accepted
    snapshot = controller.prompt_queue_registry.snapshot(paused.session_id)
    assert snapshot.claimed_count == 0
    assert snapshot.waiting_count == 2
    assert [entry.entry_id for entry in snapshot.entries] == [
        first.entry_id,
        second.entry_id,
    ]
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.reservation is PromptQueueReservation.RELEASED
    assert gateway.provider_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
@pytest.mark.parametrize("path", ["direct", "agent"])
async def test_queued_recovery_postaccept_exception_settles_exact_claim_once(
    monkeypatch, action, path
):
    from Tests.Chat.test_console_chat_store import FakePersistence

    persistence = FakePersistence()
    (
        controller,
        store,
        gateway,
        service,
        paused,
        first,
        second,
    ) = await _paused_queued_send(persistence=persistence)
    service.error = None
    failure_gateway = _PrepareFailureFence(store)
    controller.provider_gateway = failure_gateway
    if path == "agent":
        bridge = _StateCapturingAgentBridge(store)
        controller._agent_bridge = bridge
        controller._agent_runtime_enabled = True
        store.set_session_project_instruction_state(
            paused.session_id, ProjectInstructionControlState.legacy_disabled()
        )

        async def fail_agent_setup(**_kwargs):
            raise RuntimeError("agent setup failed")

        monkeypatch.setattr(
            controller, "_compose_agent_request_providers", fail_agent_setup
        )

    if action == "retry":
        result = await controller.retry_library_preparation(paused.preparation_id)
    else:
        result = await controller.bypass_library_preparation(paused.preparation_id)

    assert isinstance(result, ConsoleSubmitResult)
    assert result.accepted
    assert result.queue_entry_id == first.entry_id
    assert result.user_message_id == paused.transient_user_message_id
    assert result.assistant_message_id is not None
    snapshot = controller.prompt_queue_registry.snapshot(paused.session_id)
    assert snapshot.claimed_count == 0
    assert snapshot.waiting_count == 1
    assert snapshot.entries[0].entry_id == second.entry_id
    assert snapshot.mode is PromptQueueMode.PAUSED
    queued_users = [
        row
        for row in persistence.created_messages
        if row["sender"] == "user" and row["content"] == "frozen queued"
    ]
    assert len(queued_users) == 1
    live_rows = store.messages_for_session(paused.session_id)
    assert (
        len(
            [
                row
                for row in live_rows
                if row.role is ConsoleMessageRole.USER
                and row.content == "frozen queued"
            ]
        )
        == 1
    )
    assert sum(row.role is ConsoleMessageRole.ASSISTANT for row in live_rows) == 2
    assert live_rows[-1].role is ConsoleMessageRole.ASSISTANT
    assert live_rows[-1].id == result.assistant_message_id
    assert live_rows[-1].status == "failed"
    assert store.preparation_for_session(paused.session_id) is None
    assert failure_gateway.provider_calls == 0
    assert store.preparation_for_session(paused.session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
async def test_recovery_uses_frozen_staged_inputs_and_leaves_new_state_staged(
    monkeypatch, action
):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda *_args: True)
    captured_contexts = []

    async def capture_evidence(_app, launch, *, user_message):
        captured_contexts.append((launch, user_message))
        return SimpleNamespace(
            context="[S1] newly staged evidence",
            citation_builder=None,
            prompt_evidence_set_id=None,
            citation_repair_contract=None,
        )

    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture_evidence
    )
    evidence_state: dict[str, object] = {"launch": None, "released": []}
    retrieval = _real_retrieval_controller_for_launch(evidence_state)
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    session = store.create_session(session_id="session-1")
    original_attachment = _pending_image("original.png", b"original-image")
    new_attachment = _pending_image("new.png", b"new-image")
    assert store.add_pending_attachment(session.id, original_attachment)
    store.set_session_one_shot_prefill(session.id, "original prefill")
    gateway = _StreamingFence()
    service = _RagService(error=RuntimeError("pause first"))
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=retrieval._capture_console_staged_rag,
        staged_evidence_provider=lambda _session_id: (
            evidence_state["launch"] is not None
        ),
        model="vision-model",
    )
    controller.app = SimpleNamespace(library_rag_search_service=service)

    first = await controller.submit_draft("frozen draft", session_id=session.id)
    paused = store.preparation_for_session(session.id)
    assert not first.accepted
    assert paused is not None

    assert store.add_pending_attachment(session.id, new_attachment)
    store.set_session_one_shot_prefill(session.id, "new prefill")
    store.set_session_draft(session.id, "new composer draft")
    new_bundle = build_library_rag_evidence_bundle(
        [_row("new staged evidence")], query="new evidence query"
    )
    new_launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="New staged evidence",
        payload={"evidence_bundle": new_bundle.to_payload()},
        status="ready",
    )
    evidence_state["launch"] = new_launch
    service.error = None
    service.result = {"results": []}

    if action == "retry":
        result = await controller.retry_library_preparation(paused.preparation_id)
    else:
        result = await controller.bypass_library_preparation(paused.preparation_id)

    assert result.accepted
    user_payload = next(
        message for message in reversed(gateway.messages) if message["role"] == "user"
    )
    assert "b3JpZ2luYWwtaW1hZ2U=" in str(user_payload)
    assert "bmV3LWltYWdl" not in str(user_payload)
    assert gateway.messages[-1] == {
        "role": "assistant",
        "content": "original prefill",
    }
    assert store.pending_attachments(session.id) == [new_attachment]
    assert store.session_one_shot_prefill(session.id) == "new prefill"
    assert store.session_draft(session.id) == "new composer draft"
    assert evidence_state["launch"] is new_launch
    assert evidence_state["released"] == []
    assert captured_contexts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
async def test_recovery_resolution_exception_pauses_without_ready_wedge(action):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _ResolverFailureFence()
    service = _RagService(error=RuntimeError("pause first"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)
    first = await controller.submit_draft("frozen draft")
    paused = store.preparation_for_session(first.session_id)
    service.error = None

    if action == "retry":
        result = await controller.retry_library_preparation(paused.preparation_id)
    else:
        result = await controller.bypass_library_preparation(paused.preparation_id)

    assert isinstance(result, ConsoleSubmitResult)
    assert not result.accepted
    current = store.preparation_for_session(first.session_id)
    assert current is not None
    assert current.state is ConsoleTurnPreparationState.PAUSED
    assert current.pause_kind is ConsolePreparationPauseKind.DESTINATION_CHANGED
    assert gateway.provider_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
@pytest.mark.parametrize("reclaim_failure", ["refusal", "exception"])
async def test_queue_reclaim_failure_pauses_without_claim_or_ready_wedge(
    monkeypatch, action, reclaim_failure
):
    (
        controller,
        store,
        gateway,
        service,
        paused,
        first,
        second,
    ) = await _paused_queued_send()
    service.error = None

    if reclaim_failure == "refusal":

        def replacement(*_args):
            return None
    else:

        def replacement(*_args):
            raise RuntimeError("reclaim failed")

    monkeypatch.setattr(
        controller.prompt_queue_coordinator, "reclaim_prepared_entry", replacement
    )
    if action == "retry":
        result = await controller.retry_library_preparation(paused.preparation_id)
    else:
        result = await controller.bypass_library_preparation(paused.preparation_id)

    assert isinstance(result, ConsoleSubmitResult)
    assert not result.accepted
    current = store.preparation_for_session(paused.session_id)
    assert current is not None
    assert current.state is ConsoleTurnPreparationState.PAUSED
    assert current.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    snapshot = controller.prompt_queue_registry.snapshot(paused.session_id)
    assert snapshot.claimed_count == 0
    assert [entry.entry_id for entry in snapshot.entries] == [
        first.entry_id,
        second.entry_id,
    ]
    assert gateway.provider_calls == 1


@pytest.mark.asyncio
async def test_repeated_recovery_actions_return_stable_submit_results_without_keyerror():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    service = _RagService(error=RuntimeError("pause first"))
    controller = ConsoleChatController(store=store, provider_gateway=_StreamingFence())
    controller.app = SimpleNamespace(library_rag_search_service=service)
    first = await controller.submit_draft("frozen draft")
    paused = store.preparation_for_session(first.session_id)

    cancelled = controller.cancel_library_preparation(paused.preparation_id)
    retry_loser = await controller.retry_library_preparation(paused.preparation_id)
    bypass_loser = await controller.bypass_library_preparation(paused.preparation_id)
    cancel_loser = controller.cancel_library_preparation(paused.preparation_id)

    for result in (cancelled, retry_loser, bypass_loser, cancel_loser):
        assert isinstance(result, ConsoleSubmitResult)
        assert not result.accepted


@pytest.mark.asyncio
async def test_losing_ready_commit_cas_returns_paused_submit_result(monkeypatch):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    service = _RagService(error=RuntimeError("pause first"))
    controller = ConsoleChatController(store=store, provider_gateway=_StreamingFence())
    controller.app = SimpleNamespace(library_rag_search_service=service)
    first = await controller.submit_draft("frozen draft")
    paused = store.preparation_for_session(first.session_id)
    service.error = None
    original_cas = store.compare_and_set_preparation

    def lose_ready_commit(session_id, transition):
        if (
            transition.expected_state is ConsoleTurnPreparationState.READY
            and transition.new_state is ConsoleTurnPreparationState.COMMITTING
        ):
            committed = original_cas(session_id, transition)
            assert committed is not None
            original_cas(
                session_id,
                ConsolePreparationTransition(
                    preparation_id=transition.preparation_id,
                    expected_state=ConsoleTurnPreparationState.COMMITTING,
                    new_state=ConsoleTurnPreparationState.PAUSED,
                    pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
                    new_attempt_id=None,
                ),
            )
            return None
        return original_cas(session_id, transition)

    monkeypatch.setattr(store, "compare_and_set_preparation", lose_ready_commit)
    result = await controller.retry_library_preparation(paused.preparation_id)

    assert isinstance(result, ConsoleSubmitResult)
    assert not result.accepted
    current = store.preparation_for_session(first.session_id)
    assert current is not None
    assert current.state is ConsoleTurnPreparationState.PAUSED
    assert current.pause_kind is ConsolePreparationPauseKind.PERSISTENCE


@pytest.mark.asyncio
async def test_direct_request_preparation_failure_never_claims_dispatch_started():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _PrepareFailureFence(store)
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    with pytest.raises(RuntimeError, match="direct preparation failed"):
        await controller.submit_draft("frozen draft")

    assert gateway.observed_state is ConsoleTurnPreparationState.ACCEPTED
    assert gateway.provider_calls == 0
    assert store.preparation_for_session(store.active_session_id) is None


@pytest.mark.asyncio
async def test_agent_setup_failure_never_claims_dispatch_started(monkeypatch):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    bridge = _StateCapturingAgentBridge(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    session = store.create_session(session_id="session-1")
    store.set_session_project_instruction_state(
        session.id, ProjectInstructionControlState.legacy_disabled()
    )
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    observed = []

    async def fail_setup(**_kwargs):
        preparation = store.preparation_for_session(store.active_session_id)
        observed.append(preparation.state)
        raise RuntimeError("agent setup failed")

    monkeypatch.setattr(controller, "_compose_agent_request_providers", fail_setup)

    with pytest.raises(RuntimeError, match="agent setup failed"):
        await controller.submit_draft("frozen draft", session_id=session.id)

    assert observed == [ConsoleTurnPreparationState.ACCEPTED]
    assert bridge.calls == 0
    assert store.preparation_for_session(session.id) is None


@pytest.mark.asyncio
async def test_actual_agent_call_observes_dispatch_started():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    bridge = _StateCapturingAgentBridge(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    session = store.create_session(session_id="session-1")
    store.set_session_project_instruction_state(
        session.id, ProjectInstructionControlState.legacy_disabled()
    )
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))

    result = await controller.submit_draft("frozen draft", session_id=session.id)

    assert result.accepted
    assert bridge.calls == 1
    assert bridge.observed_state is ConsoleTurnPreparationState.DISPATCH_STARTED
    assert store.preparation_for_session(result.session_id) is None


@pytest.mark.asyncio
async def test_close_preserves_accepted_owner_until_cancelled_task_finally_settles():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _BlockingFirstFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    task = asyncio.create_task(controller.submit_draft("frozen draft"))
    await gateway.started.wait()
    session_id = store.active_session_id
    preparation = store.preparation_for_session(session_id)
    assert preparation is not None

    controller.close_session(session_id)

    assert not task.done()
    live = store.preparation_by_id(preparation.preparation_id)
    assert live is not None
    assert live.state is ConsoleTurnPreparationState.DISPATCH_STARTED
    assert preparation.preparation_id in controller._prepared_send_continuations

    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert store.preparation_by_id(preparation.preparation_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}


@pytest.mark.asyncio
async def test_zero_match_contribution_survives_until_provider_attempt_then_cleans():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    controller = None

    class ContributionFence(_StreamingFence):
        async def stream_chat(self, resolution, messages, **kwargs):
            preparation = store.preparation_for_session(store.active_session_id)
            outcome = controller.preparation_outcome(preparation.preparation_id)
            assert outcome is not None
            assert outcome.contribution is not None
            assert outcome.contribution.outcome == "zero_matches"
            async for chunk in super().stream_chat(resolution, messages, **kwargs):
                yield chunk

    gateway = ContributionFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=_RagService())

    result = await controller.submit_draft("frozen draft")

    assert result.accepted
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}


@pytest.mark.asyncio
async def test_same_text_one_shot_rearm_survives_inflight_send_completion():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _BlockingFirstFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.create_session(session_id="session-1")
    store.set_session_one_shot_prefill(session.id, "same prefill")

    task = asyncio.create_task(controller.submit_draft("frozen draft"))
    await gateway.started.wait()
    store.set_session_one_shot_prefill(session.id, "same prefill")
    gateway.release.set()
    result = await task

    assert result.accepted
    assert store.session_one_shot_prefill(session.id) == "same prefill"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["direct", "agent"])
async def test_shutdown_tracks_accepted_submit_and_rechecks_before_external_call(
    monkeypatch, path
):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _StreamingFence()
    bridge = _StateCapturingAgentBridge(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge if path == "agent" else None,
        agent_runtime_enabled=path == "agent",
    )
    session = store.create_session(session_id="session-1")
    store.set_session_project_instruction_state(
        session.id, ProjectInstructionControlState.legacy_disabled()
    )
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    held = _CancellationResistantBoundary()
    if path == "direct":

        async def hold_preflight(*, provider_messages, **_kwargs):
            await held.wait()
            return provider_messages, None

        monkeypatch.setattr(
            controller, "_apply_conversation_memory_preflight", hold_preflight
        )
    else:

        async def hold_agent_setup(**_kwargs):
            await held.wait()
            return None, None, None, None

        monkeypatch.setattr(
            controller, "_compose_agent_request_providers", hold_agent_setup
        )

    submit = asyncio.create_task(controller.submit_draft("frozen draft"))
    await held.entered.wait()
    shutdown = asyncio.create_task(controller.shutdown())
    for _ in range(10):
        await asyncio.sleep(0)
        if held.cancelled.is_set() or shutdown.done():
            break
    shutdown_waited = not shutdown.done()
    cancellation_reached_boundary = held.cancelled.is_set()
    held.release.set()
    if not cancellation_reached_boundary:
        submit.cancel()
    await asyncio.gather(shutdown, return_exceptions=True)
    (result,) = await asyncio.gather(submit, return_exceptions=True)

    assert shutdown_waited
    assert cancellation_reached_boundary
    assert submit.done()
    assert isinstance(result, ConsoleSubmitResult)
    assert gateway.provider_calls == 0
    assert bridge.calls == 0
    assert store.preparation_for_session(session.id) is None
    assert controller._active_submit_tasks == {}


@pytest.mark.asyncio
async def test_shutdown_tracks_committing_submit_before_stream_registration(
    monkeypatch,
):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    controller = ConsoleChatController(store=store, provider_gateway=_StreamingFence())
    controller.app = SimpleNamespace(library_rag_search_service=_RagService())
    held = _CancellationResistantBoundary()

    async def hold_history(_draft):
        preparation = store.preparation_for_session(store.active_session_id)
        assert preparation is not None
        assert preparation.state is ConsoleTurnPreparationState.COMMITTING
        await held.wait()

    monkeypatch.setattr(controller, "_record_prompt_history", hold_history)
    submit = asyncio.create_task(controller.submit_draft("frozen draft"))
    await held.entered.wait()
    preparation = store.preparation_for_session(store.active_session_id)
    outcome = controller.preparation_outcome(preparation.preparation_id)
    assert outcome is not None and outcome.contribution is not None

    shutdown = asyncio.create_task(controller.shutdown())
    for _ in range(10):
        await asyncio.sleep(0)
        if held.cancelled.is_set() or shutdown.done():
            break
    shutdown_waited = not shutdown.done()
    cancellation_reached_boundary = held.cancelled.is_set()
    held.release.set()
    if not cancellation_reached_boundary:
        submit.cancel()
    await asyncio.gather(shutdown, return_exceptions=True)
    await asyncio.gather(submit, return_exceptions=True)

    assert shutdown_waited
    assert cancellation_reached_boundary
    assert submit.done()
    assert controller.provider_gateway.provider_calls == 0
    assert store.preparation_by_id(preparation.preparation_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}
    assert controller._active_submit_tasks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "expected_state"),
    [
        ("history", ConsoleTurnPreparationState.COMMITTING),
        ("preflight", ConsoleTurnPreparationState.ACCEPTED),
    ],
)
async def test_close_preserves_submit_owner_until_cancelled_task_finalizer(
    monkeypatch, boundary, expected_state
):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    controller = ConsoleChatController(store=store, provider_gateway=_StreamingFence())
    controller.app = SimpleNamespace(library_rag_search_service=_RagService())
    held = _CancellationResistantBoundary()
    if boundary == "history":

        async def hold_history(_draft):
            await held.wait()

        monkeypatch.setattr(controller, "_record_prompt_history", hold_history)
    else:

        async def hold_preflight(*, provider_messages, **_kwargs):
            await held.wait()
            return provider_messages, None

        monkeypatch.setattr(
            controller, "_apply_conversation_memory_preflight", hold_preflight
        )

    submit = asyncio.create_task(controller.submit_draft("frozen draft"))
    await held.entered.wait()
    session_id = store.active_session_id
    preparation = store.preparation_for_session(session_id)
    assert preparation is not None and preparation.state is expected_state

    controller.close_session(session_id)
    for _ in range(10):
        await asyncio.sleep(0)
        if held.cancelled.is_set():
            break
    cancellation_reached_boundary = held.cancelled.is_set()
    live = store.preparation_by_id(preparation.preparation_id)
    owner_preserved = live is not None and live.state is expected_state
    continuation_preserved = (
        preparation.preparation_id in controller._prepared_send_continuations
    )
    outcome = controller.preparation_outcome(preparation.preparation_id)
    contribution_preserved = outcome is not None and outcome.contribution is not None

    held.release.set()
    if not cancellation_reached_boundary:
        submit.cancel()
    await asyncio.gather(submit, return_exceptions=True)
    assert cancellation_reached_boundary
    assert owner_preserved
    assert continuation_preserved
    assert contribution_preserved
    assert store.preparation_by_id(preparation.preparation_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}
    assert controller._active_submit_tasks == {}


def _staged_evidence_launch(title: str) -> ConsoleLiveWorkLaunch:
    bundle = build_library_rag_evidence_bundle(
        [_row(f"{title} evidence")], query=f"{title} query"
    )
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title=title,
        payload={"evidence_bundle": bundle.to_payload()},
        status="ready",
    )


async def _capture_staged_evidence(_app, launch, *, user_message):
    return SimpleNamespace(
        context=f"[S1] {launch.title}: {user_message}",
        citation_builder=None,
        prompt_evidence_set_id=None,
        citation_repair_contract=None,
    )


@pytest.mark.asyncio
async def test_explicit_evidence_lease_survives_preaccept_failure(monkeypatch):
    state: dict[str, object] = {
        "launch": _staged_evidence_launch("original"),
        "released": [],
    }
    original = state["launch"]
    retrieval = _real_retrieval_controller_for_launch(state)
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        rag_capture_provider=retrieval._capture_console_staged_rag,
    )
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        _capture_staged_evidence,
    )

    async def fail_dictionary(_messages, _session_id):
        raise RuntimeError("dictionary failed")

    monkeypatch.setattr(controller, "_apply_chat_dictionaries", fail_dictionary)
    with pytest.raises(RuntimeError, match="dictionary failed"):
        await controller.submit_draft("frozen draft")

    assert state["launch"] is original
    assert state["released"] == []


@pytest.mark.asyncio
async def test_explicit_evidence_lease_releases_exact_launch_only_after_acceptance(
    monkeypatch,
):
    state: dict[str, object] = {
        "launch": _staged_evidence_launch("original"),
        "released": [],
    }
    original = state["launch"]
    retrieval = _real_retrieval_controller_for_launch(state)
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _BlockingFirstFence()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=retrieval._capture_console_staged_rag,
    )
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        _capture_staged_evidence,
    )

    submit = asyncio.create_task(controller.submit_draft("frozen draft"))
    await gateway.started.wait()
    launch_released = state["launch"] is None
    released = list(state["released"])
    gateway.release.set()
    assert (await submit).accepted
    assert launch_released
    assert len(released) == 1
    assert released[0][0] is original


@pytest.mark.asyncio
async def test_explicit_evidence_lease_never_releases_newer_launch(monkeypatch):
    state: dict[str, object] = {
        "launch": _staged_evidence_launch("original"),
        "released": [],
    }
    original = state["launch"]
    newer = _staged_evidence_launch("newer")
    retrieval = _real_retrieval_controller_for_launch(state)
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        rag_capture_provider=retrieval._capture_console_staged_rag,
    )
    held = _CancellationResistantBoundary()

    async def capture_evidence(_app, launch, *, user_message):
        assert launch is original
        await held.wait()
        return SimpleNamespace(
            context="[S1] original evidence",
            citation_builder=None,
            prompt_evidence_set_id=None,
            citation_repair_contract=None,
        )

    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture_evidence
    )
    submit = asyncio.create_task(controller.submit_draft("frozen draft"))
    await held.entered.wait()
    state["launch"] = newer
    held.release.set()
    assert (await submit).accepted

    assert state["launch"] is newer
    assert state["released"][0][0] is original


@pytest.mark.asyncio
async def test_explicit_evidence_lease_cancel_keeps_original_staged(monkeypatch):
    state: dict[str, object] = {
        "launch": _staged_evidence_launch("original"),
        "released": [],
    }
    original = state["launch"]
    retrieval = _real_retrieval_controller_for_launch(state)
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        rag_capture_provider=retrieval._capture_console_staged_rag,
    )
    held = _CancellationResistantBoundary()

    async def capture_evidence(_app, launch, *, user_message):
        await held.wait()
        return SimpleNamespace(
            context="[S1] original evidence",
            citation_builder=None,
            prompt_evidence_set_id=None,
            citation_repair_contract=None,
        )

    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture_evidence
    )
    submit = asyncio.create_task(controller.submit_draft("frozen draft"))
    await held.entered.wait()
    preparation = store.preparation_for_session(store.active_session_id)
    cancelled = controller.cancel_library_preparation(preparation.preparation_id)
    assert not cancelled.accepted
    held.release.set()
    await submit

    assert state["launch"] is original
    assert state["released"] == []


@pytest.mark.asyncio
async def test_same_session_refusal_cannot_overwrite_first_submit_owner():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    session = store.create_session(session_id="session-1")
    boundary = _CancellationResistantBoundary()
    gateway = _HeldResolveFence(boundary)
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    first = asyncio.create_task(
        controller.submit_draft("first draft", session_id=session.id)
    )
    await boundary.entered.wait()
    refused = await controller.submit_draft("second draft", session_id=session.id)

    registry_preserved_first = controller._active_submit_tasks == {first: session.id}

    shutdown = asyncio.create_task(controller.shutdown())
    for _ in range(10):
        await asyncio.sleep(0)
        if boundary.cancelled.is_set() or shutdown.done():
            break
    shutdown_waited = not shutdown.done()
    cancellation_reached_first = boundary.cancelled.is_set()
    if not cancellation_reached_first:
        first.cancel()
    boundary.release.set()
    await asyncio.gather(shutdown, return_exceptions=True)
    (result,) = await asyncio.gather(first, return_exceptions=True)

    assert not refused.accepted
    assert registry_preserved_first
    assert shutdown_waited
    assert cancellation_reached_first
    assert isinstance(result, ConsoleSubmitResult)
    assert not result.accepted
    assert gateway.provider_calls == 0
    assert controller._active_submit_tasks == {}


@pytest.mark.asyncio
async def test_submit_registry_tracks_multiple_sessions_and_cleans_exact_done_tasks():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    first_session = store.create_session(session_id="session-1")
    second_session = store.create_session(session_id="session-2")
    gateway = _TwoResolveFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    first = asyncio.create_task(
        controller.submit_draft("first draft", session_id=first_session.id)
    )
    second = asyncio.create_task(
        controller.submit_draft("second draft", session_id=second_session.id)
    )
    await gateway.both_entered.wait()

    registry_owned_both = controller._active_submit_tasks == {
        first: first_session.id,
        second: second_session.id,
    }

    gateway.release.set()
    results = await asyncio.gather(first, second)

    assert registry_owned_both
    assert all(result.accepted for result in results)
    assert gateway.provider_calls == 2
    assert controller._active_submit_tasks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["direct", "agent"])
async def test_postaccept_cancellation_returns_exact_accepted_result(monkeypatch, path):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    gateway = _StreamingFence()
    bridge = _StateCapturingAgentBridge(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge if path == "agent" else None,
        agent_runtime_enabled=path == "agent",
    )
    session = store.create_session(session_id="session-1")
    store.set_session_project_instruction_state(
        session.id, ProjectInstructionControlState.legacy_disabled()
    )
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    held = _CancellationBoundary()
    if path == "direct":

        async def hold_preflight(*, provider_messages, **_kwargs):
            await held.wait()
            return provider_messages, None

        monkeypatch.setattr(
            controller, "_apply_conversation_memory_preflight", hold_preflight
        )
    else:

        async def hold_agent_setup(**_kwargs):
            await held.wait()
            return None, None, None, None

        monkeypatch.setattr(
            controller, "_compose_agent_request_providers", hold_agent_setup
        )

    submit = asyncio.create_task(
        controller.submit_draft("frozen draft", session_id=session.id)
    )
    await held.entered.wait()
    preparation = store.preparation_for_session(session.id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.ACCEPTED
    accepted_rows = store.messages_for_session(session.id)
    assert [row.role for row in accepted_rows] == [
        ConsoleMessageRole.USER,
        ConsoleMessageRole.ASSISTANT,
    ]

    submit.cancel()
    (result,) = await asyncio.gather(submit, return_exceptions=True)

    final_rows = store.messages_for_session(session.id)
    assert isinstance(result, ConsoleSubmitResult)
    assert result == ConsoleSubmitResult(
        accepted=True,
        should_clear_draft=True,
        visible_copy="Accepted turn failed before provider dispatch.",
        session_id=session.id,
        user_message_id=accepted_rows[0].id,
        assistant_message_id=accepted_rows[1].id,
        terminal_status=ConsoleRunStatus.FAILED,
        origin=ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id=None,
        committed_context_epoch=result.committed_context_epoch,
    )
    assert result.committed_context_epoch is not None
    assert [row.id for row in final_rows] == [row.id for row in accepted_rows]
    assert final_rows[1].status == "failed"
    assert gateway.provider_calls == 0
    assert bridge.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["direct", "agent"])
async def test_off_thread_begin_shutdown_schedules_owner_loop_cancellation(
    monkeypatch, path
):
    loop = asyncio.get_running_loop()
    old_debug = loop.get_debug()
    loop.set_debug(True)
    try:
        store = ConsoleChatStore()
        store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
        gateway = _StreamingFence()
        bridge = _StateCapturingAgentBridge(store)
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            agent_bridge=bridge if path == "agent" else None,
            agent_runtime_enabled=path == "agent",
        )
        session = store.create_session(session_id="session-1")
        store.set_session_project_instruction_state(
            session.id, ProjectInstructionControlState.legacy_disabled()
        )
        controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
        held = _CancellationBoundary()
        if path == "direct":

            async def hold_preflight(*, provider_messages, **_kwargs):
                await held.wait()
                return provider_messages, None

            monkeypatch.setattr(
                controller, "_apply_conversation_memory_preflight", hold_preflight
            )
        else:

            async def hold_agent_setup(**_kwargs):
                await held.wait()
                return None, None, None, None

            monkeypatch.setattr(
                controller, "_compose_agent_request_providers", hold_agent_setup
            )

        submit = _OwnerThreadTask(
            controller.submit_draft("frozen draft", session_id=session.id), loop=loop
        )
        await held.entered.wait()
        shutdown_error = None
        try:
            await asyncio.to_thread(controller.begin_shutdown)
        except BaseException as exc:  # recorded for the RED assertion below
            shutdown_error = exc
        held.release.set()
        (result,) = await asyncio.gather(submit, return_exceptions=True)
        await controller.shutdown()

        assert shutdown_error is None
        assert isinstance(result, ConsoleSubmitResult)
        assert result.accepted
        assert result.user_message_id is not None
        assert result.assistant_message_id is not None
        assert gateway.provider_calls == 0
        assert bridge.calls == 0
        assert controller._active_submit_tasks == {}
    finally:
        loop.set_debug(old_debug)


def test_closed_loop_pending_submit_drops_exact_volatile_ownership(monkeypatch):
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    session = store.create_session(session_id="closed-session")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
    )
    controller.app = SimpleNamespace(library_rag_search_service=_RagService())
    held = asyncio.Event()

    async def hold_history(_draft):
        held.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(controller, "_record_prompt_history", hold_history)
    closed_loop = asyncio.new_event_loop()
    loop_errors: list[dict[str, object]] = []
    closed_loop.set_debug(True)
    closed_loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(dict(context))
    )
    submit = closed_loop.create_task(
        controller.submit_draft("unreachable draft", session_id=session.id)
    )
    submit_ref = weakref.ref(submit)
    try:
        for _ in range(20):
            closed_loop.run_until_complete(asyncio.sleep(0))
            if held.is_set():
                break
        preparation = store.preparation_for_session(session.id)
        assert held.is_set()
        assert preparation is not None
        assert preparation.state is ConsoleTurnPreparationState.COMMITTING
        assert controller.preparation_outcome(preparation.preparation_id) is not None
        assert preparation.preparation_id in controller._prepared_send_continuations

        closed_loop.close()
        controller.begin_shutdown()

        assert controller._active_submit_tasks == {}
        assert store.preparation_for_session(session.id) is None
        assert controller._preparation_outcomes == {}
        assert controller._prepared_send_continuations == {}
        assert controller._shutdown_requested.is_set()
        assert loop_errors == []
    finally:
        if not closed_loop.is_closed():
            closed_loop.close()
        submit._log_destroy_pending = False
        submit.get_coro().close()
        del submit
        gc.collect()

    assert submit_ref() is None
    assert loop_errors == []


@pytest.mark.asyncio
async def test_closed_loop_peer_never_blocks_same_session_live_submit_shutdown(
    monkeypatch,
):
    running_loop = asyncio.get_running_loop()
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    session = store.create_session(session_id="shared-session")
    controller = ConsoleChatController(store=store, provider_gateway=_StreamingFence())
    held = _CancellationBoundary()

    async def hold_preflight(*, provider_messages, **_kwargs):
        await held.wait()
        return provider_messages, None

    monkeypatch.setattr(
        controller, "_apply_conversation_memory_preflight", hold_preflight
    )
    live_submit = asyncio.create_task(
        controller.submit_draft("live draft", session_id=session.id)
    )
    await held.entered.wait()
    preparation = store.preparation_for_session(session.id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.ACCEPTED

    closed_loop = asyncio.new_event_loop()
    closed_submit = closed_loop.create_task(
        controller.submit_draft("closed peer", session_id=session.id)
    )
    closed_submit_ref = weakref.ref(closed_submit)
    controller._register_submit_task(closed_submit, session.id)
    controller._bind_submit_preparation(closed_submit, preparation.preparation_id)
    closed_loop.close()
    try:
        controller.begin_shutdown()

        assert closed_submit not in controller._active_submit_tasks
        assert live_submit in controller._active_submit_tasks
        live_preparation = store.preparation_for_session(session.id)
        assert live_preparation is not None
        assert live_preparation.preparation_id == preparation.preparation_id
        assert live_preparation.state is ConsoleTurnPreparationState.ACCEPTED
        assert controller._owner_loop is running_loop
        (result,) = await asyncio.gather(live_submit, return_exceptions=True)

        assert isinstance(result, ConsoleSubmitResult)
        assert result.accepted
        assert controller.provider_gateway.provider_calls == 0
        assert store.preparation_for_session(session.id) is None
        assert controller._active_submit_tasks == {}
    finally:
        closed_submit._log_destroy_pending = False
        closed_submit.get_coro().close()
        del closed_submit
        gc.collect()

    assert closed_submit_ref() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation", ["same_thread", "off_thread"])
async def test_shutdown_callback_failure_rethrows_after_all_task_cleanup(
    monkeypatch, invocation
):
    running_loop = asyncio.get_running_loop()
    prior_exception_handler = running_loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    running_loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(dict(context))
    )
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    session = store.create_session(session_id="live-session")
    closed_session = store.create_session(session_id="closed-session")
    controller = ConsoleChatController(store=store, provider_gateway=_StreamingFence())
    held = _CancellationBoundary()

    async def hold_preflight(*, provider_messages, **_kwargs):
        await held.wait()
        return provider_messages, None

    monkeypatch.setattr(
        controller, "_apply_conversation_memory_preflight", hold_preflight
    )
    live_submit = asyncio.create_task(
        controller.run_prompt_chain("live draft", session_id=session.id)
    )
    await held.entered.wait()
    preparation = store.preparation_for_session(session.id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.ACCEPTED

    stream_started = asyncio.Event()

    async def hold_stream():
        stream_started.set()
        await asyncio.Event().wait()

    stream_task = asyncio.create_task(hold_stream())
    await stream_started.wait()
    controller._active_stream_tasks[closed_session.id] = stream_task

    closed_loop = asyncio.new_event_loop()
    closed_submit = closed_loop.create_task(
        controller.submit_draft("closed draft", session_id=closed_session.id)
    )
    closed_submit_ref = weakref.ref(closed_submit)
    controller._register_submit_task(closed_submit, closed_session.id)
    closed_preparation = _preparation(
        session_id=closed_session.id,
        preparation_id="closed-preparation",
        state=ConsoleTurnPreparationState.READY,
    )
    assert store.begin_preparation(closed_preparation) is closed_preparation
    controller._bind_submit_preparation(
        closed_submit, closed_preparation.preparation_id
    )
    controller._preparation_outcomes[closed_preparation.preparation_id] = (
        ConsolePreparationOutcome(
            preparation_id=closed_preparation.preparation_id,
            attempt_id=closed_preparation.attempt_id,
            state=closed_preparation.state,
            evidence_bundle=None,
            contribution=None,
            error_code=None,
        )
    )
    controller._prepared_send_continuations[closed_preparation.preparation_id] = (
        object()
    )  # type: ignore[assignment]
    closed_loop.close()
    headless_cancel = threading.Event()
    controller._headless_visit_cancel = headless_cancel
    callback_failure = RuntimeError("private presentation failure")

    def fail_activity_callback(_session_id):
        raise callback_failure

    controller.prompt_queue_coordinator.on_activity_changed = fail_activity_callback
    try:
        if invocation == "same_thread":
            with pytest.raises(RuntimeError) as caught:
                controller.begin_shutdown()
            assert caught.value is callback_failure
        else:
            await asyncio.to_thread(controller.begin_shutdown)
            await asyncio.sleep(0)
        controller.prompt_queue_coordinator.on_activity_changed = None
        assert controller._shutdown_requested.is_set()
        assert headless_cancel.is_set()
        assert closed_submit not in controller._active_submit_tasks
        submit_result, stream_result = await asyncio.gather(
            live_submit, stream_task, return_exceptions=True
        )

        assert isinstance(submit_result, ConsoleSubmitResult)
        assert submit_result.accepted
        assert isinstance(stream_result, asyncio.CancelledError)
        assert controller.provider_gateway.provider_calls == 0
        assert store.preparation_for_session(session.id) is None
        assert controller._preparation_outcomes == {}
        assert controller._prepared_send_continuations == {}
        assert controller._active_submit_tasks == {}
        assert loop_errors == []
    finally:
        running_loop.set_exception_handler(prior_exception_handler)
        controller.prompt_queue_coordinator.on_activity_changed = None
        if not live_submit.done():
            live_submit.cancel()
        if not stream_task.done():
            stream_task.cancel()
        await asyncio.gather(live_submit, stream_task, return_exceptions=True)
        closed_submit._log_destroy_pending = False
        closed_submit.get_coro().close()
        del closed_submit
        gc.collect()

    assert closed_submit_ref() is None


@pytest.mark.asyncio
async def test_ready_close_removes_echo_idempotently_and_preserves_evidence_launch(
    monkeypatch,
):
    state: dict[str, object] = {
        "launch": _staged_evidence_launch("original"),
        "released": [],
    }
    original = state["launch"]
    retrieval = _real_retrieval_controller_for_launch(state)
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StreamingFence(),
        rag_capture_provider=retrieval._capture_console_staged_rag,
    )
    held = _CancellationBoundary()

    async def hold_capture(_app, launch, *, user_message):
        assert launch is original
        await held.wait()
        return SimpleNamespace(
            context=f"[S1] held evidence: {user_message}",
            citation_builder=None,
            prompt_evidence_set_id=None,
            citation_repair_contract=None,
        )

    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", hold_capture
    )
    session = store.create_session(session_id="session-1")
    submit = asyncio.create_task(
        controller.submit_draft("frozen draft", session_id=session.id)
    )
    await held.entered.wait()
    preparation = store.preparation_for_session(session.id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.READY

    controller.close_session(session.id)
    (result,) = await asyncio.gather(submit, return_exceptions=True)

    assert isinstance(result, ConsoleSubmitResult)
    assert not result.accepted
    assert state["launch"] is original
    assert state["released"] == []
    assert store.preparation_by_id(preparation.preparation_id) is None
    assert controller._preparation_outcomes == {}
    assert controller._prepared_send_continuations == {}
    assert controller._active_submit_tasks == {}


def test_one_shot_prefill_revision_rejects_bool_non_int_and_negative():
    store = ConsoleChatStore()
    session = store.create_session(session_id="session-1")
    store.set_session_one_shot_prefill(session.id, "exact prefill")
    value, revision = store.session_one_shot_prefill_snapshot(session.id)
    assert value == "exact prefill"
    assert revision == 1

    for malformed in (True, 1.0, "1", -1):
        with pytest.raises(ValueError, match="non-negative integer"):
            store.consume_session_one_shot_prefill(session.id, malformed)  # type: ignore[arg-type]
        assert store.session_one_shot_prefill_snapshot(session.id) == (
            "exact prefill",
            revision,
        )

    assert store.consume_session_one_shot_prefill(session.id, revision)
    assert store.session_one_shot_prefill_snapshot(session.id) == (None, revision + 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["direct", "agent"])
async def test_recovered_queue_acknowledges_postaccept_cancellation_once(
    monkeypatch, path
):
    from Tests.Chat.test_console_chat_store import FakePersistence

    persistence = FakePersistence()
    (
        controller,
        store,
        gateway,
        service,
        paused,
        first,
        second,
    ) = await _paused_queued_send(persistence=persistence)
    service.error = None
    held = _CancellationBoundary()
    if path == "direct":

        async def hold_preflight(*, provider_messages, **_kwargs):
            await held.wait()
            return provider_messages, None

        monkeypatch.setattr(
            controller, "_apply_conversation_memory_preflight", hold_preflight
        )
    else:
        bridge = _StateCapturingAgentBridge(store)
        controller._agent_bridge = bridge
        controller._agent_runtime_enabled = True
        store.set_session_project_instruction_state(
            paused.session_id, ProjectInstructionControlState.legacy_disabled()
        )

        async def hold_agent_setup(**_kwargs):
            await held.wait()
            return None, None, None, None

        monkeypatch.setattr(
            controller, "_compose_agent_request_providers", hold_agent_setup
        )

    recovery = asyncio.create_task(
        controller.retry_library_preparation(paused.preparation_id)
    )
    await held.entered.wait()
    accepted_rows = store.messages_for_session(paused.session_id)
    recovery.cancel()
    result = await recovery

    snapshot = controller.prompt_queue_registry.snapshot(paused.session_id)
    assert result.accepted
    assert result.queue_entry_id == first.entry_id
    assert result.user_message_id == paused.transient_user_message_id
    assert result.assistant_message_id == accepted_rows[-1].id
    assert result.terminal_status is ConsoleRunStatus.FAILED
    assert snapshot.claimed_count == 0
    assert snapshot.waiting_count == 1
    assert snapshot.entries[0].entry_id == second.entry_id
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert gateway.provider_calls == 1
    assert (
        len(
            [
                row
                for row in persistence.created_messages
                if row["sender"] == "user" and row["content"] == "frozen queued"
            ]
        )
        == 1
    )
