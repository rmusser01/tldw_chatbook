"""Joined Console model-thinking lifecycle evidence.

These tests keep the provider at the adapter boundary while driving the real
controller, capture, persistence, history preparation, exchange, sync, and UI
projection seams.  They intentionally decode each durable owner instead of
searching one aggregate representation for a canary.
"""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from textual.app import App, ComposeResult

from Tests.Chat.test_console_automatic_library_preparation import (
    _PolicyCoordinator,
    _RagService,
)
from Tests.console_provider_doubles import with_destination
from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
)
from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    load_chat_history_from_file_and_save_to_db,
)
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.Chat_Functions import generate_chat_history_content
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    PROPRIETARY_THINKING_NOTICE,
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_prepared_request import (
    THINKING_OWNER_KEY,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ProviderProprietaryThinkingEvidence,
    ProviderThinkingDelta,
)
from tldw_chatbook.Chat.console_thinking_history import (
    ProviderThinkingSidecar,
    ThinkingReplayTarget,
    resolve_thinking_history,
)
from tldw_chatbook.Chat.console_turn_grouping import project_thinking_activities
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
    parse_thinking_blocks_json,
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.tldw_api import SyncV2Envelope
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleActivityDisclosure,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

VISIBLE_ANSWER = "VISIBLE-ANSWER-CANARY"
DISPLAYABLE_THINKING = "DISPLAYABLE-THINKING-CANARY"
RAW_CONTINUATION = "RAW-CONTINUATION-CANARY"


class _TranscriptHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


class _PausedThinkingGateway(ConsoleProviderGateway):
    """Real request preparation with a controllable adapter-edge stream."""

    # This double replaces the real transport path that defers the callback,
    # so the controller owns the provider-dispatch boundary.
    deferred_dispatch_boundary = False

    def __init__(self) -> None:
        super().__init__(environ={})
        self.evidence_seen = asyncio.Event()
        self.release_answer = asyncio.Event()
        self.prepared = None
        self.provider_contacts = 0

    async def resolve_for_send(self, _selection):
        return with_destination(
            ConsoleProviderResolution(
                provider="llama_cpp",
                model="joined-reasoner",
                base_url="http://127.0.0.1:9099",
                ready=True,
                execution_key="llama_cpp",
                continuation_protocol="chat_completions",
                thinking_stream_disposition="displayable",
                thinking_round_trip_version=1,
            )
        )

    def prepare_chat_request(self, resolution, messages, **kwargs):
        self.prepared = super().prepare_chat_request(
            resolution,
            messages,
            context_window_override_tokens=10_000,
            **kwargs,
        )
        return self.prepared

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        self.provider_contacts += 1
        yield ProviderThinkingDelta(
            text=DISPLAYABLE_THINKING,
            provider="llama_cpp",
            model="joined-reasoner",
            protocol="chat_completions",
            source_format="start_anchored_think",
        )
        self.evidence_seen.set()
        await self.release_answer.wait()
        yield VISIBLE_ANSWER


class _DispositionGateway(ConsoleProviderGateway):
    """Typed capability resolution plus an adapter-contact spy."""

    # This double replaces the real transport path that defers the callback,
    # so the controller owns the provider-dispatch boundary.
    deferred_dispatch_boundary = False

    def __init__(self, disposition: str) -> None:
        super().__init__(environ={})
        self.disposition = disposition
        self.provider_contacts = 0

    async def resolve_for_send(self, _selection):
        return with_destination(
            ConsoleProviderResolution(
                provider="vllm",
                model="joined-reasoner",
                base_url="http://127.0.0.1:9099",
                ready=True,
                execution_key="vllm",
                continuation_protocol="chat_completions",
                thinking_stream_disposition=self.disposition,
                thinking_round_trip_version=(
                    None if self.disposition == "ignored" else 1
                ),
            )
        )

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        self.provider_contacts += 1
        yield "CONTROL-ANSWER"


class _ScriptedEvidenceGateway(_DispositionGateway):
    def __init__(self, disposition: str, *events: object) -> None:
        super().__init__(disposition)
        self.events = events

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        self.provider_contacts += 1
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event


class _VersionedPersistence:
    """Established persistent adapter shape with selectable thinking support."""

    def __init__(self, delegate: ChatPersistenceService, version: int) -> None:
        self._delegate = delegate
        self.db = delegate.db
        self.version = version

    def thinking_round_trip_version(self) -> int:
        return self.version

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def _inline_attachment() -> PendingAttachment:
    return PendingAttachment(
        file_path="/tmp/frozen-context.txt",
        display_name="frozen-context.txt",
        file_type="text",
        insert_mode="inline",
        text_content="FROZEN-ATTACHMENT-CANARY",
        original_size=26,
        processed_size=26,
    )


def _complete_displayable_envelope() -> ThinkingEnvelope:
    return ThinkingEnvelope(
        (
            DisplayableThinkingBlock(
                block_id="joined-thinking-0",
                round_ordinal=0,
                provider="llama_cpp",
                model="Qwen3.8-27B",
                protocol="chat_completions",
                source_format="start_anchored_think",
                status="complete",
                text=DISPLAYABLE_THINKING,
            ),
        )
    )


def _complete_continuation_json() -> str:
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content=VISIBLE_ANSWER,
                reasoning_blocks=(RAW_CONTINUATION,),
                calls=(),
            ),
        ),
    )
    raw = dump_provider_continuation_json(checkpoint)
    assert raw is not None
    return raw


def _seed_exchange_conversation(db: CharactersRAGDB, *, title: str) -> tuple[str, str]:
    conversation_id = db.add_conversation(
        {"title": title, "thinking_history_policy": "include"}
    )
    user_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "Question",
        }
    )
    assistant_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": user_id,
            "sender": "assistant",
            "role": "assistant",
            "content": VISIBLE_ANSWER,
            "thinking_blocks_json": dump_thinking_blocks_json(
                _complete_displayable_envelope()
            ),
            "provider_continuation_json": _complete_continuation_json(),
            "assistant_generation_state": "complete",
        }
    )
    db.set_conversation_active_leaf(conversation_id, assistant_id)
    return str(conversation_id), str(assistant_id)


def _resume_store(
    db: CharactersRAGDB, conversation_id: str
) -> tuple[ConsoleChatStore, object]:
    tree = ChatConversationService(db).get_conversation_tree(
        conversation_id,
        depth_cap=10_000,
        root_limit=10_000,
    )
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = db
    nodes = screen._message._console_messages_from_conversation_tree(tree)
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Thinking integration",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=nodes,
        active_leaf_persisted_id=db.get_conversation_active_leaf(conversation_id),
    )
    return store, session


def _disclosure(transcript: ConsoleTranscript, assistant) -> ConsoleActivityDisclosure:
    activity = project_thinking_activities(assistant=assistant)[0]
    return transcript.query_one(
        f"#console-activity-disclosure-{activity.activity_id}",
        ConsoleActivityDisclosure,
    )


@pytest.mark.asyncio
async def test_actual_turn_expands_collapses_once_and_restarts_collapsed(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "thinking-lifecycle.sqlite", "joined-source")
    gateway = _PausedThinkingGateway()
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.create_session(title="Thinking integration")
    store.active_session_id = session.id

    try:
        send = asyncio.create_task(controller.submit_draft("Question"))
        evidence = asyncio.create_task(gateway.evidence_seen.wait())
        done, _pending = await asyncio.wait(
            {send, evidence}, timeout=2, return_when=asyncio.FIRST_COMPLETED
        )
        assert evidence in done, (
            f"send completed before adapter evidence: {send.result()!r}"
            if send in done
            else "adapter evidence timed out"
        )
        live_messages = store.messages_for_session(session.id)
        live = live_messages[-1]
        assert live.role is ConsoleMessageRole.ASSISTANT
        assert live.content == ""
        assert live.thinking is not None
        assert [block.text for block in live.thinking.blocks] == [DISPLAYABLE_THINKING]

        app = _TranscriptHarness()
        async with app.run_test(size=(100, 28)):
            transcript = app.query_one(ConsoleTranscript)
            transcript.set_messages(live_messages, session_id=session.id)
            await transcript.refresh_messages()
            disclosure = _disclosure(transcript, live)
            activity_id = disclosure.activity_message_id
            assert disclosure.expanded
            assert activity_id in transcript._pending_thinking_auto_collapse

            gateway.release_answer.set()
            result = await asyncio.wait_for(send, timeout=2)
            assert result.accepted is True
            completed_messages = store.messages_for_session(session.id)
            completed = completed_messages[-1]
            transcript.set_messages(completed_messages, session_id=session.id)
            await transcript.refresh_messages()

            assert _disclosure(transcript, completed) is disclosure
            assert disclosure.expanded is False
            assert activity_id not in transcript._pending_thinking_auto_collapse

            # Re-projecting the same terminal state cannot re-arm auto-collapse.
            transcript.set_messages(completed_messages, session_id=session.id)
            await transcript.refresh_messages()
            assert disclosure.expanded is False
            assert activity_id not in transcript._pending_thinking_auto_collapse

        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None
        durable = db.get_message_by_id(completed.persisted_message_id)
        assert durable["content"] == VISIBLE_ANSWER
        assert durable["thinking_blocks_json"] is not None

        resumed_store, resumed_session = _resume_store(db, conversation_id)
        restored = resumed_store.messages_for_session(resumed_session.id)[-1]
        assert restored.thinking == completed.thinking
        assert restored.status == "complete"

        restarted_app = _TranscriptHarness()
        async with restarted_app.run_test(size=(100, 28)):
            transcript = restarted_app.query_one(ConsoleTranscript)
            transcript.set_messages(
                resumed_store.messages_for_session(resumed_session.id),
                session_id=resumed_session.id,
            )
            await transcript.refresh_messages()
            historical = _disclosure(transcript, restored)
            assert historical.expanded is False
            assert historical.detail_available is True
            assert not historical.detail_stack.children
    finally:
        if not gateway.release_answer.is_set():
            gateway.release_answer.set()
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved_policy", "continuation_required", "expected_policy", "expected_count"),
    [
        ("auto", False, "auto", 1),
        ("include", False, "include", 1),
        ("exclude", False, "exclude", 0),
        ("include", True, "required", 1),
    ],
)
async def test_durable_owner_replay_is_counted_and_dispatched_exactly_once(
    saved_policy: str,
    continuation_required: bool,
    expected_policy: str,
    expected_count: int,
) -> None:
    owner_id = "assistant-durable-owner"
    endpoint = "http://127.0.0.1:9099"
    dispatched: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/v1/chat/completions"
        dispatched.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "next answer"}}]},
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=endpoint,
        ),
        environ={},
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp",
            base_url=endpoint,
            explicit_model="Qwen3.8-27B",
            reasoning_effort="low",
            streaming=False,
        )
    )
    target = ThinkingReplayTarget(
        provider=resolution.execution_key,
        model=resolution.model or "",
        protocol=resolution.continuation_protocol or "chat_completions",
        disposition=resolution.thinking_stream_disposition,
        round_trip_version=resolution.thinking_round_trip_version,
    )
    resolved = resolve_thinking_history(
        target=target,
        policy=saved_policy,
        sidecars=(ProviderThinkingSidecar(owner_id, _complete_displayable_envelope()),),
        continuation_required=continuation_required,
    )
    assistant_row = {"role": "assistant", "content": VISIBLE_ANSWER}
    if resolved.groups:
        assistant_row[THINKING_OWNER_KEY] = owner_id
    semantic = build_console_request(
        [
            {"role": "user", "content": "Prior question"},
            assistant_row,
            {"role": "user", "content": "Next question"},
        ],
        thinking_groups=resolved.groups,
        thinking_policy=resolved.saved_policy,
        effective_thinking_policy=resolved.effective_policy,
    )
    counted_payloads: list[list[dict]] = []

    def count_spy(messages: list[dict], _model: str) -> int:
        counted_payloads.append(messages)
        return len(str(messages))

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider=resolution.execution_key,
        model=resolution.model or "",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=count_spy,
    )
    wire = [thaw_json(row) for row in prepared.messages]

    try:
        assert [item async for item in gateway.stream_chat(resolution, prepared)] == [
            "next answer"
        ]
        assert resolution.thinking_stream_disposition == "displayable"
        assert resolved.effective_policy == expected_policy
        assert str(wire).count(DISPLAYABLE_THINKING) == expected_count
        assert counted_payloads[-1] == wire
        assert (
            str(dispatched[0]["messages"]).count(DISPLAYABLE_THINKING) == expected_count
        )
        assert all(THINKING_OWNER_KEY not in row for row in wire)
    finally:
        await gateway.aclose()


def test_required_overlay_cannot_be_saved_or_downgraded() -> None:
    resolved = resolve_thinking_history(
        target=ThinkingReplayTarget(
            provider="llama_cpp",
            model="Qwen3.8-27B",
            protocol="chat_completions",
            disposition="displayable",
            round_trip_version=1,
        ),
        policy="exclude",
        sidecars=(
            ProviderThinkingSidecar(
                "assistant-durable-owner", _complete_displayable_envelope()
            ),
        ),
        continuation_required=True,
    )

    assert resolved.saved_policy == "exclude"
    assert resolved.effective_policy == "required"
    assert resolved.groups == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["displayable", "proprietary"])
async def test_unsupported_persistent_backend_refuses_before_provider_and_recovers(
    tmp_path,
    disposition: str,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"unsupported-{disposition}.sqlite", "unsupported-backend"
    )
    supported = ChatPersistenceService(db)
    gateway = _DispositionGateway(disposition)
    store = ConsoleChatStore(
        persistence=_VersionedPersistence(supported, 0)  # type: ignore[arg-type]
    )
    session = store.create_session(title="Unsupported backend")
    store.active_session_id = session.id
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    try:
        refused = await controller.submit_draft("RECOVERABLE-DRAFT-CANARY")

        assert refused.accepted is False
        assert refused.should_clear_draft is False
        assert refused.visible_copy == (
            "This persistent backend cannot preserve model thinking version 1. "
            "Upgrade it before sending."
        )
        assert "RECOVERABLE-DRAFT-CANARY" not in refused.visible_copy
        assert gateway.provider_contacts == 0
        assert store.messages_for_session(session.id) == []
        assert db.execute_query("SELECT COUNT(*) FROM messages").fetchone()[0] == 0

        # The same draft remains usable after the backend is upgraded.
        store.persistence = supported
        accepted = await controller.submit_draft("RECOVERABLE-DRAFT-CANARY")
        assert accepted.accepted is True
        assert gateway.provider_contacts == 1
        messages = store.messages_for_session(session.id)
        assert [message.content for message in messages] == [
            "RECOVERABLE-DRAFT-CANARY",
            "CONTROL-ANSWER",
        ]
        assert messages[-1].thinking is None
        durable = db.get_message_by_id(messages[-1].persisted_message_id)
        assert durable["thinking_blocks_json"] is None
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "bypass"])
async def test_resumed_preparation_thinking_refusal_preserves_exact_owner(
    tmp_path,
    action: str,
) -> None:
    db = CharactersRAGDB(tmp_path / f"resumed-{action}.sqlite", f"resumed-{action}")
    supported = ChatPersistenceService(db)
    persistence = _VersionedPersistence(supported, 1)
    store = ConsoleChatStore(persistence=persistence)  # type: ignore[arg-type]
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    session = store.create_session(title="Pre-send title")
    store.active_session_id = session.id
    store.set_session_draft(session.id, "INTERVENING-DRAFT-CANARY")
    attachment = _inline_attachment()
    assert store.add_pending_attachment(session.id, attachment)
    gateway = _DispositionGateway("displayable")
    service = _RagService(error=RuntimeError("pause before commit"))
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    try:
        first = await controller.submit_draft(
            "RESUMED-OWNER-CANARY", session_id=session.id
        )
        paused = store.preparation_for_session(session.id)
        assert first.accepted is False
        assert paused is not None
        assert paused.state is ConsoleTurnPreparationState.PAUSED
        assert paused.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
        assert paused.transient_user_message_id is not None
        owner_before = store.get_message(paused.transient_user_message_id)
        continuation_before = controller._prepared_send_continuations[
            paused.preparation_id
        ]
        assert owner_before is not None
        assert continuation_before.attachments == (attachment,)

        # This state changed after admission and is not owned by the preparation.
        intervening_conversation_id = str(
            db.add_conversation({"title": "Intervening conversation"})
        )
        session.title = "Intervening session title"
        session.persisted_conversation_id = intervening_conversation_id
        persistence.version = 0
        service.error = None

        refused = (
            await controller.retry_library_preparation(paused.preparation_id)
            if action == "retry"
            else await controller.bypass_library_preparation(paused.preparation_id)
        )

        assert refused.accepted is False
        assert refused.should_clear_draft is False
        assert refused.visible_copy == (
            "This persistent backend cannot preserve model thinking version 1. "
            "Upgrade it before sending."
        )
        assert "RESUMED-OWNER-CANARY" not in refused.visible_copy
        current = store.preparation_for_session(session.id)
        assert current is not None
        assert current.state is ConsoleTurnPreparationState.PAUSED
        assert current.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
        assert current.transient_user_message_id == paused.transient_user_message_id
        assert store.get_message(paused.transient_user_message_id) == owner_before
        assert store.session_draft(session.id) == "INTERVENING-DRAFT-CANARY"
        assert store.pending_attachments(session.id) == [attachment]
        assert (
            controller._prepared_send_continuations[paused.preparation_id]
            is continuation_before
        )
        assert session.title == "Intervening session title"
        assert session.persisted_conversation_id == intervening_conversation_id
        assert gateway.provider_contacts == 0

        # Move back to a valid first-persistence identity before proving that
        # the same prepared owner can proceed after the backend upgrade.
        session.persisted_conversation_id = None
        persistence.version = 1
        recovered = await controller.retry_library_preparation(paused.preparation_id)

        assert recovered.accepted is True
        assert gateway.provider_contacts == 1
        assert store.preparation_for_session(session.id) is None
        assert store.get_message(paused.transient_user_message_id) is not None
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("persistent", "disposition", "round_trip_version"),
    [
        (True, "displayable", 1),
        (True, "ignored", 0),
        (False, "proprietary", 0),
    ],
    ids=["v1-compatible", "ignored-disposition", "ephemeral"],
)
async def test_persistence_preflight_controls_dispatch(
    tmp_path,
    persistent: bool,
    disposition: str,
    round_trip_version: int,
) -> None:
    db = CharactersRAGDB(
        tmp_path / f"control-{disposition}-{persistent}.sqlite", "control-backend"
    )
    supported = ChatPersistenceService(db)
    persistence = (
        _VersionedPersistence(supported, round_trip_version) if persistent else None
    )
    gateway = _DispositionGateway(disposition)
    store = ConsoleChatStore(persistence=persistence)  # type: ignore[arg-type]
    session = store.create_session(
        title="Preflight control",
        ephemeral=not persistent,
    )
    store.active_session_id = session.id
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    try:
        result = await controller.submit_draft("Control question")

        assert result.accepted is True
        assert gateway.provider_contacts == 1
        assistant = store.messages_for_session(session.id)[-1]
        assert assistant.content == "CONTROL-ANSWER"
        assert assistant.thinking is None
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_plain_local_model_uses_real_resolver_and_dispatches_on_v0_backend(
    tmp_path,
) -> None:
    endpoint = "http://127.0.0.1:9099"
    provider_contacts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal provider_contacts
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/v1/chat/completions"
        provider_contacts += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "PLAIN-MODEL-ANSWER"}}]},
        )

    db = CharactersRAGDB(tmp_path / "plain-v0.sqlite", "plain-v0")
    supported = ChatPersistenceService(db)
    store = ConsoleChatStore(
        persistence=_VersionedPersistence(supported, 0)  # type: ignore[arg-type]
    )
    session = store.create_session(title="Plain local model")
    store.active_session_id = session.id
    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=endpoint,
        ),
        environ={},
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="Llama-3.3-8B-Instruct",
        base_url=endpoint,
        streaming=False,
    )

    try:
        result = await controller.submit_draft("Plain model question")

        assert result.accepted is True
        assert provider_contacts == 1
        assert [
            message.content for message in store.messages_for_session(session.id)
        ] == [
            "Plain model question",
            "PLAIN-MODEL-ANSWER",
        ]
    finally:
        await gateway.aclose()
        db.close_connection()


@pytest.mark.asyncio
async def test_unsupported_preflight_preserves_existing_conversation_exactly(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "existing-refusal.sqlite", "existing-refusal")
    supported = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=supported)
    session = store.create_session(title="Existing conversation")
    store.active_session_id = session.id
    existing = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Existing content",
        persist=True,
    )
    conversation_id = session.persisted_conversation_id
    before_row = db.get_message_by_id(existing.persisted_message_id)
    gateway = _DispositionGateway("displayable")
    store.persistence = _VersionedPersistence(supported, 0)  # type: ignore[assignment]
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    try:
        refused = await controller.submit_draft("Unsaved retry draft")

        assert refused.accepted is False
        assert gateway.provider_contacts == 0
        assert session.persisted_conversation_id == conversation_id
        assert session.title == "Existing conversation"
        assert store.messages_for_session(session.id) == [existing]
        assert db.get_message_by_id(existing.persisted_message_id) == before_row
        assert db.execute_query("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_stop_and_failure_preserve_only_received_displayable_evidence(
    tmp_path,
) -> None:
    stopped_db = CharactersRAGDB(tmp_path / "stopped.sqlite", "stopped-source")
    stopped_gateway = _PausedThinkingGateway()
    stopped_store = ConsoleChatStore(persistence=ChatPersistenceService(stopped_db))
    stopped_session = stopped_store.create_session(title="Stopped evidence")
    stopped_store.active_session_id = stopped_session.id
    stopped_controller = ConsoleChatController(
        store=stopped_store, provider_gateway=stopped_gateway
    )

    failed_db = CharactersRAGDB(tmp_path / "failed.sqlite", "failed-source")
    failed_gateway = _ScriptedEvidenceGateway(
        "displayable",
        ProviderThinkingDelta(
            text=DISPLAYABLE_THINKING,
            provider="vllm",
            model="joined-reasoner",
            protocol="chat_completions",
            source_format="start_anchored_think",
        ),
        RuntimeError("transport failed"),
    )
    failed_store = ConsoleChatStore(persistence=ChatPersistenceService(failed_db))
    failed_session = failed_store.create_session(title="Failed evidence")
    failed_store.active_session_id = failed_session.id
    failed_controller = ConsoleChatController(
        store=failed_store, provider_gateway=failed_gateway
    )

    try:
        stopped_send = asyncio.create_task(stopped_controller.submit_draft("Stop"))
        await asyncio.wait_for(stopped_gateway.evidence_seen.wait(), timeout=2)
        assert stopped_controller.stop_active_run() is True
        stopped_gateway.release_answer.set()
        await asyncio.wait_for(stopped_send, timeout=2)
        stopped = next(
            message
            for message in stopped_store.messages_for_session(stopped_session.id)
            if message.role is ConsoleMessageRole.ASSISTANT
        )
        assert stopped.status == "stopped"
        assert stopped.content == ""
        assert stopped.thinking is not None
        assert {block.status for block in stopped.thinking.blocks} == {"stopped"}
        stopped_row = stopped_db.get_message_by_id(stopped.persisted_message_id)
        assert {
            block.status
            for block in parse_thinking_blocks_json(
                stopped_row["thinking_blocks_json"]
            ).blocks
        } == {"stopped"}

        failed_result = await failed_controller.submit_draft("Fail")
        assert failed_result.accepted is True
        failed = next(
            message
            for message in failed_store.messages_for_session(failed_session.id)
            if message.role is ConsoleMessageRole.ASSISTANT
        )
        assert failed.status == "failed"
        assert failed.content == ""
        assert failed.thinking is not None
        assert {block.status for block in failed.thinking.blocks} == {"failed"}
        failed_row = failed_db.get_message_by_id(failed.persisted_message_id)
        assert {
            block.status
            for block in parse_thinking_blocks_json(
                failed_row["thinking_blocks_json"]
            ).blocks
        } == {"failed"}
    finally:
        stopped_gateway.release_answer.set()
        stopped_db.close_connection()
        failed_db.close_connection()


@pytest.mark.asyncio
async def test_proprietary_and_capable_no_evidence_turns_are_honest(tmp_path) -> None:
    proprietary_db = CharactersRAGDB(
        tmp_path / "proprietary.sqlite", "proprietary-source"
    )
    proprietary_gateway = _ScriptedEvidenceGateway(
        "proprietary",
        ProviderProprietaryThinkingEvidence(
            provider="vllm",
            model="joined-reasoner",
            protocol="chat_completions",
            source_format="reasoning_content",
        ),
        VISIBLE_ANSWER,
    )
    proprietary_store = ConsoleChatStore(
        persistence=ChatPersistenceService(proprietary_db)
    )
    proprietary_session = proprietary_store.create_session(title="Proprietary evidence")
    proprietary_store.active_session_id = proprietary_session.id
    proprietary_controller = ConsoleChatController(
        store=proprietary_store, provider_gateway=proprietary_gateway
    )

    no_evidence_db = CharactersRAGDB(
        tmp_path / "no-evidence.sqlite", "no-evidence-source"
    )
    no_evidence_gateway = _ScriptedEvidenceGateway("displayable", VISIBLE_ANSWER)
    no_evidence_store = ConsoleChatStore(
        persistence=ChatPersistenceService(no_evidence_db)
    )
    no_evidence_session = no_evidence_store.create_session(title="No evidence")
    no_evidence_store.active_session_id = no_evidence_session.id
    no_evidence_controller = ConsoleChatController(
        store=no_evidence_store, provider_gateway=no_evidence_gateway
    )

    try:
        proprietary_result = await proprietary_controller.submit_draft("Question")
        assert proprietary_result.accepted is True
        proprietary = proprietary_store.messages_for_session(proprietary_session.id)[-1]
        assert proprietary.thinking is not None
        assert len(proprietary.thinking.blocks) == 1
        assert isinstance(proprietary.thinking.blocks[0], ProprietaryThinkingBlock)
        raw = proprietary_db.get_message_by_id(proprietary.persisted_message_id)[
            "thinking_blocks_json"
        ]
        assert PROPRIETARY_THINKING_NOTICE not in raw
        assert "text" not in raw

        app = _TranscriptHarness()
        async with app.run_test(size=(100, 28)):
            transcript = app.query_one(ConsoleTranscript)
            transcript.set_messages(
                proprietary_store.messages_for_session(proprietary_session.id),
                session_id=proprietary_session.id,
            )
            await transcript.refresh_messages()
            disclosure = _disclosure(transcript, proprietary)
            assert disclosure.status == "unavailable"
            assert (
                transcript.thinking_detail_text(disclosure.activity_message_id)
                == PROPRIETARY_THINKING_NOTICE
            )

        no_evidence_result = await no_evidence_controller.submit_draft("Question")
        assert no_evidence_result.accepted is True
        no_evidence = no_evidence_store.messages_for_session(no_evidence_session.id)[-1]
        assert no_evidence.content == VISIBLE_ANSWER
        assert no_evidence.thinking is None
        assert project_thinking_activities(assistant=no_evidence) == ()
        no_evidence_row = no_evidence_db.get_message_by_id(
            no_evidence.persisted_message_id
        )
        assert no_evidence_row["thinking_blocks_json"] is None
    finally:
        proprietary_db.close_connection()
        no_evidence_db.close_connection()


def test_selected_json_import_hydrates_thinking_and_policy_in_second_db(
    tmp_path,
) -> None:
    source = CharactersRAGDB(tmp_path / "json-source.sqlite", "json-source")
    target = CharactersRAGDB(tmp_path / "json-target.sqlite", "json-target")
    try:
        conversation_id, _assistant_id = _seed_exchange_conversation(
            source, title="Selected JSON thinking"
        )
        rows = source.get_messages_for_conversation(conversation_id, limit=100)
        content, _filename = generate_chat_history_content(
            rows,
            conversation_id,
            None,
            db_instance=source,
        )
        payload = json.loads(content)
        assert payload["thinking_history_policy"] == "include"
        assert payload["sensitive_data_warning"].startswith(
            "This conversation export contains model thinking"
        )

        imported_id, _character_id = load_chat_history_from_file_and_save_to_db(
            target,
            io.BytesIO(content.encode("utf-8")),
        )
        assert imported_id is not None
        restored_store, restored_session = _resume_store(target, str(imported_id))
        restored = restored_store.messages_for_session(restored_session.id)[-1]
        assert restored.content == VISIBLE_ANSWER
        assert restored.thinking == _complete_displayable_envelope()
        assert (
            restored_store.session_thinking_history_policy(restored_session.id)
            == "include"
        )
    finally:
        source.close_connection()
        target.close_connection()


def test_chatbook_v2_import_hydrates_thinking_and_policy_in_second_db(
    tmp_path,
) -> None:
    source_path = tmp_path / "chatbook-source.sqlite"
    source = CharactersRAGDB(source_path, "chatbook-source")
    conversation_id, _assistant_id = _seed_exchange_conversation(
        source, title="Chatbook thinking"
    )
    source.close_connection()

    archive_path = tmp_path / "thinking.chatbook.zip"
    creator = ChatbookCreator({"ChaChaNotes": str(source_path)})
    creator.temp_dir = tmp_path
    success, message, _dependencies = creator.create_chatbook(
        name="Thinking",
        description="Thinking integration",
        content_selections={ContentType.CONVERSATION: [conversation_id]},
        output_path=archive_path,
    )
    assert success, message

    target_path = tmp_path / "chatbook-target.sqlite"
    target = CharactersRAGDB(target_path, "chatbook-target-bootstrap")
    target.close_connection()
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(target_path)})
    importer.temp_dir = tmp_path / "chatbook-import"
    importer.temp_dir.mkdir()
    imported, import_message = importer.import_chatbook(
        archive_path,
        import_status=status,
    )
    assert imported, import_message
    assert status.failed_items == 0

    target = CharactersRAGDB(target_path, "chatbook-target-assert")
    try:
        imported_conversation = target.get_conversation_by_name("Chatbook thinking")[0]
        restored_store, restored_session = _resume_store(
            target, str(imported_conversation["id"])
        )
        restored = restored_store.messages_for_session(restored_session.id)[-1]
        assert restored.content == VISIBLE_ANSWER
        assert restored.thinking == _complete_displayable_envelope()
        assert (
            restored_store.session_thinking_history_policy(restored_session.id)
            == "include"
        )
    finally:
        target.close_connection()


def test_sync_v2_applies_one_complete_generation_then_hydrates(tmp_path) -> None:
    source = CharactersRAGDB(tmp_path / "sync-source.sqlite", "sync-source")
    target = CharactersRAGDB(tmp_path / "sync-target.sqlite", "sync-target")
    state = SyncStateRepository(tmp_path / "sync-state.sqlite")
    dataset_key = generate_dataset_key()
    try:
        conversation_id, assistant_id = _seed_exchange_conversation(
            source, title="Sync thinking"
        )
        assert (
            target.add_conversation(
                {
                    "id": conversation_id,
                    "title": "Sync thinking",
                    "thinking_history_policy": "include",
                }
            )
            == conversation_id
        )
        state.set_sync_v2_profile_state(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            profile_mode="local_first",
            device_id="sync-source",
            dataset_id="dataset-1",
        )
        row = source.get_message_by_id(assistant_id)
        payload_hash = canonical_payload_hash(
            {
                "assistant_generation_state": "complete",
                "content": VISIBLE_ANSWER,
                "provider_continuation_json": _complete_continuation_json(),
                "role": "assistant",
                "thinking_blocks_json": dump_thinking_blocks_json(
                    _complete_displayable_envelope()
                ),
            }
        )
        result = ChatSyncV2OutboxProducer(
            state_repository=state,
            dataset_keys={"dataset-1": dataset_key},
            source=source,
        ).reconcile_chat_message_intent(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope=None,
            message_id=assistant_id,
            message_version=row["version"],
            payload_hash=payload_hash,
        )
        assert result["status"] == "enqueued"
        envelope = SyncV2Envelope.model_validate(result["outbox_entry"]["envelope"])
        assert SyncEnvelopeApplier(dataset_key=dataset_key, local_store=target).apply(
            envelope
        ) == {"status": "applied"}

        restored_store, restored_session = _resume_store(target, conversation_id)
        restored = restored_store.messages_for_session(restored_session.id)[-1]
        assert restored.content == VISIBLE_ANSWER
        assert restored.thinking == _complete_displayable_envelope()
        target_row = target.get_message_by_id(assistant_id)
        assert target_row["provider_continuation_json"] == (
            _complete_continuation_json()
        )
    finally:
        state.close()
        source.close_connection()
        target.close_connection()


def test_edit_and_replacement_clear_while_soft_delete_retains_private_state(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "ownership.sqlite", "ownership-source")
    try:
        conversation_id, assistant_id = _seed_exchange_conversation(
            db, title="Ownership clearing"
        )
        store, session = _resume_store(db, conversation_id)
        assistant = store.messages_for_session(session.id)[-1]

        store.set_message_feedback(assistant.id, "up")
        preserved = db.get_message_by_id(assistant_id)
        assert preserved["thinking_blocks_json"] == dump_thinking_blocks_json(
            _complete_displayable_envelope()
        )
        assert preserved["provider_continuation_json"] == _complete_continuation_json()

        store.update_message_content(assistant.id, "User-edited answer")
        edited = db.get_message_by_id(assistant_id)
        assert edited["content"] == "User-edited answer"
        assert edited["thinking_blocks_json"] is None
        assert edited["provider_continuation_json"] is None

        deleted_thinking = dump_thinking_blocks_json(_complete_displayable_envelope())
        deleted_continuation = _complete_continuation_json()
        delete_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "parent_message_id": assistant_id,
                "sender": "assistant",
                "role": "assistant",
                "content": VISIBLE_ANSWER,
                "thinking_blocks_json": deleted_thinking,
                "provider_continuation_json": deleted_continuation,
                "assistant_generation_state": "complete",
            }
        )
        delete_row = db.get_message_by_id(delete_id)
        assert db.soft_delete_message(delete_id, expected_version=delete_row["version"])
        deleted = db.execute_query(
            "SELECT deleted, thinking_blocks_json, provider_continuation_json "
            "FROM messages WHERE id = ?",
            (delete_id,),
        ).fetchone()
        assert dict(deleted) == {
            "deleted": 1,
            "thinking_blocks_json": deleted_thinking,
            "provider_continuation_json": deleted_continuation,
        }
        assert db.get_message_by_id(delete_id) is None
        assert all(
            row["id"] != delete_id
            for row in db.get_messages_for_conversation(conversation_id, limit=100)
        )
        deleted_store, deleted_session = _resume_store(db, conversation_id)
        provider_messages = ConsoleChatController(
            store=deleted_store,
            provider_gateway=_DispositionGateway("displayable"),
        )._provider_messages_for_session(deleted_session.id)
        assert {"role": "assistant", "content": "User-edited answer"} in (
            provider_messages
        )
        assert VISIBLE_ANSWER not in json.dumps(provider_messages)

        delete_payload_row = db.execute_query(
            "SELECT payload FROM sync_log WHERE entity = 'messages' "
            "AND entity_id = ? AND operation = 'delete' "
            "ORDER BY change_id DESC LIMIT 1",
            (delete_id,),
        ).fetchone()
        delete_payload = json.loads(delete_payload_row["payload"])
        assert set(delete_payload) == {
            "id",
            "deleted",
            "last_modified",
            "assistant_generation_state",
            "base_payload_hash",
            "version",
            "client_id",
        }
        assert delete_payload["id"] == delete_id
        assert delete_payload["deleted"] == 1
        assert VISIBLE_ANSWER not in json.dumps(delete_payload)
        assert DISPLAYABLE_THINKING not in json.dumps(delete_payload)
        assert RAW_CONTINUATION not in json.dumps(delete_payload)

        replacement_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "parent_message_id": assistant_id,
                "sender": "assistant",
                "role": "assistant",
                "content": VISIBLE_ANSWER,
                "thinking_blocks_json": dump_thinking_blocks_json(
                    _complete_displayable_envelope()
                ),
                "provider_continuation_json": _complete_continuation_json(),
                "assistant_generation_state": "complete",
            }
        )
        replacement = db.get_message_by_id(replacement_id)
        ChatPersistenceService(db).replace_assistant_generation_projection(
            message_id=replacement_id,
            content="Replacement generation",
            thinking_blocks_json=None,
            provider_continuation_json=None,
            assistant_generation_state="complete",
            usage_json=None,
            expected_version=replacement["version"],
        )
        replaced = db.get_message_by_id(replacement_id)
        assert replaced["content"] == "Replacement generation"
        assert replaced["thinking_blocks_json"] is None
        assert replaced["provider_continuation_json"] is None
    finally:
        db.close_connection()
