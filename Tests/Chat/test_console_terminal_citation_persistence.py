from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

import tldw_chatbook.Chat.console_chat_store as console_chat_store_module
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_source_locators import CanonicalSourceKind
from tldw_chatbook.Chat.citation_trace_builder import (
    CitationTraceBuilder,
    LocalPromptEvidenceCapture,
    LocalRetrievalCandidateCapture,
    LocalRetrievalRunMetadata,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    LocalCitationIdentityContext,
)
from tldw_chatbook.Chat.citation_trace_models import (
    PolicyCapability,
    RetrievalScoreKind,
    RetrievalScoreScale,
    SealedCitationWrite,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    ActiveCitationTraceState,
    CitationPersistenceUnavailable,
    CitationTraceRepository,
    load_local_citation_identity_context,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_MISSING = object()
_NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
_BODY_SENTINEL = "Exact terminal body 🧪\nwith arbitrary boundaries."
_QUERY_SENTINEL = "PRIVATE_QUERY_SENTINEL"
_TITLE_SENTINEL = "PRIVATE_TITLE_SENTINEL"
_SNAPSHOT_SENTINEL = "PRIVATE_SNAPSHOT_SENTINEL"
_SOURCE_SENTINEL = "PRIVATE_SOURCE_SENTINEL"
_LOCATOR_SENTINEL = "PRIVATE_LOCATOR_SENTINEL"
_EXCEPTION_SENTINEL = "PRIVATE_EXCEPTION_SENTINEL"


class _PersistenceBase:
    db = None

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.outcomes = list(outcomes or [])

    def create_conversation(self, **kwargs: Any) -> str:
        return "conv-1"

    def update_message_content(self, **kwargs: Any) -> bool:
        self.update_calls.append(kwargs)
        return True


class _ReadyCitationPersistence(_PersistenceBase):
    canonical_citation_writes_ready = True

    def create_message(
        self,
        *,
        conversation_id: str,
        sender: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        message_id: str | None = None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
        citation_write: SealedCitationWrite | None | object = _MISSING,
    ) -> str:
        call = {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
            "image_data": image_data,
            "image_mime_type": image_mime_type,
            "message_id": message_id,
            "parent_message_id": parent_message_id,
            "feedback": feedback,
        }
        if citation_write is not _MISSING:
            call["citation_write"] = citation_write
        self.create_calls.append(call)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is not None:
                return str(outcome)
        return message_id or f"msg-{len(self.create_calls)}"


class _NoCitationKwargPersistence(_PersistenceBase):
    canonical_citation_writes_ready = True

    def create_message(
        self,
        *,
        conversation_id: str,
        sender: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        message_id: str | None = None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
    ) -> str:
        call = {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
            "image_data": image_data,
            "image_mime_type": image_mime_type,
            "message_id": message_id,
            "parent_message_id": parent_message_id,
            "feedback": feedback,
        }
        self.create_calls.append(call)
        return message_id or f"msg-{len(self.create_calls)}"


class _MissingReadinessPersistence(_PersistenceBase):
    create_message = _ReadyCitationPersistence.create_message


class _FalseReadinessPersistence(_ReadyCitationPersistence):
    canonical_citation_writes_ready = False


class _RaisingReadinessPersistence(_ReadyCitationPersistence):
    @property
    def canonical_citation_writes_ready(self) -> bool:
        raise RuntimeError("readiness-sentinel")


class _FailingConversationPersistence(_ReadyCitationPersistence):
    def create_conversation(self, **kwargs: Any) -> str:
        raise RuntimeError("append-setup-sentinel")


class _RaisingLeafDb:
    def set_conversation_active_leaf(
        self, conversation_id: str, message_id: str | None
    ) -> None:
        raise RuntimeError(_EXCEPTION_SENTINEL)


class _BookkeepingFailurePersistence(_ReadyCitationPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.db = _RaisingLeafDb()


class _RaisingSyncProducer:
    def enqueue_chat_message(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(_EXCEPTION_SENTINEL)


class _DeterministicallyUnavailableRepository(CitationTraceRepository):
    def _fail_after(self, row_family: str) -> None:
        if row_family == "owner":
            raise CitationPersistenceUnavailable("forced_deterministic_unavailable")


class _RealDirectGateway:
    def __init__(self, chunks: tuple[str, ...]) -> None:
        self.chunks = chunks

    async def resolve_for_send(self, _selection: object) -> SimpleNamespace:
        return SimpleNamespace(
            ready=True,
            visible_copy="",
            provider="llama_cpp",
            model="test-model",
            max_tokens=128,
        )

    async def stream_chat(
        self,
        _resolution: object,
        _messages: object,
    ):
        for chunk in self.chunks:
            yield chunk


@dataclass(slots=True)
class _RealCitationStack:
    db_path: Path
    client_id: str
    db: CharactersRAGDB
    codec: CitationFingerprintCodec
    repository: CitationTraceRepository
    store: ConsoleChatStore
    closed: bool = False

    def close(self) -> None:
        if not self.closed:
            self.db.close_connection()
            self.closed = True


@pytest.fixture
def real_citation_stack_factory(tmp_path: Path):
    stacks: list[_RealCitationStack] = []

    def create(
        name: str,
        repository_type: type[CitationTraceRepository] = CitationTraceRepository,
    ) -> _RealCitationStack:
        client_id = f"{name}-client"
        db_path = tmp_path / f"{name}.sqlite"
        db = CharactersRAGDB(db_path, client_id=client_id)
        identity = load_local_citation_identity_context(db)
        assert identity is not None
        codec = CitationFingerprintCodec(b"task-553.14-real-integration-key")
        repository = repository_type(
            db,
            policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
            identity_context=identity,
            fingerprint_codec=codec,
        )
        persistence = ChatPersistenceService(
            db,
            citation_repository=repository,
        )
        store = ConsoleChatStore(persistence=persistence)
        session = store.ensure_session(
            settings=ConsoleSessionSettings(provider="llama_cpp")
        )
        session.persisted_conversation_id = persistence.create_conversation(
            runtime_backend="local"
        )
        stack = _RealCitationStack(
            db_path=db_path,
            client_id=client_id,
            db=db,
            codec=codec,
            repository=repository,
            store=store,
        )
        stacks.append(stack)
        return stack

    yield create

    for stack in stacks:
        stack.close()


def _assert_terminal_state_paired(store: ConsoleChatStore) -> None:
    assert set(store._terminal_citation_finalizers) == set(
        store._terminal_persistence_deferred_ids
    )


def _append_eligible(
    store: ConsoleChatStore,
    finalizer: Callable[[str], object | None] = lambda body: None,
    *,
    persist: bool = True,
) -> tuple[str, str]:
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=persist,
        terminal_citation_finalizer=finalizer,
    )
    return session.id, message.id


def _sealed_write_for_body(body: str) -> SealedCitationWrite:
    builder = CitationTraceBuilder.local(
        request_id="request-terminal-store",
        generation_id="generation-terminal-store",
        identity_context=LocalCitationIdentityContext(
            profile_id="profile-terminal-store",
            local_authority_id="authority-terminal-store",
            fingerprint_key_id="fingerprint-terminal-store",
        ),
        fingerprint_codec=CitationFingerprintCodec(b"0123456789abcdef0123456789abcdef"),
        policy_version="terminal-store-policy-v1",
        policy_capabilities=(
            PolicyCapability.VIEW_SNAPSHOT,
            PolicyCapability.VIEW_SOURCE_IDENTITY,
        ),
        created_at=_NOW,
    )
    run_id = builder.record_retrieval_run(
        stage="hybrid",
        raw_query=_QUERY_SENTINEL,
        candidates=(
            LocalRetrievalCandidateCapture(
                candidate_rank=1,
                source_kind=CanonicalSourceKind.MEDIA_DB,
                source_id=_SOURCE_SENTINEL,
                title=_TITLE_SENTINEL,
                score_kind=RetrievalScoreKind.VECTOR_SIMILARITY,
                score_scale=RetrievalScoreScale.ZERO_TO_ONE,
                score=0.9,
                chunk_id=_LOCATOR_SENTINEL,
            ),
        ),
        retrieval_metadata=LocalRetrievalRunMetadata(
            search_mode="hybrid",
            requested_top_k=1,
            max_context_characters=1_000,
            rerank_enabled=False,
            source_kinds=(CanonicalSourceKind.MEDIA_DB,),
            scope_state="unscoped",
        ),
        started_at=_NOW,
        ended_at=_NOW,
    )
    prompt_set_id = builder.record_prompt_evidence_set(
        run_id=run_id,
        evidence=(
            LocalPromptEvidenceCapture(
                candidate_rank=1,
                snapshot_text=f"[S1] {_SNAPSHOT_SENTINEL}",
            ),
        ),
        created_at=_NOW,
    )
    attempt_id = builder.record_initial_answer_attempt(
        prompt_evidence_set_id=prompt_set_id,
        answer_body=body,
        completed_at=_NOW,
    )
    return builder.seal(
        selected_attempt_id=attempt_id,
        sealed_at=_NOW + timedelta(seconds=1),
    )


def _real_captured_builder(
    repository: CitationTraceRepository,
) -> tuple[CitationTraceBuilder, str]:
    builder = repository.create_local_trace_builder(
        request_id="request-real-terminal",
        generation_id="generation-real-terminal",
    )
    assert builder is not None
    captured_at = datetime.now(UTC)
    run_id = builder.record_retrieval_run(
        stage="hybrid",
        raw_query=_QUERY_SENTINEL,
        candidates=(
            LocalRetrievalCandidateCapture(
                candidate_rank=1,
                source_kind=CanonicalSourceKind.MEDIA_DB,
                source_id=_SOURCE_SENTINEL,
                title=_TITLE_SENTINEL,
                score_kind=RetrievalScoreKind.VECTOR_SIMILARITY,
                score_scale=RetrievalScoreScale.ZERO_TO_ONE,
                score=0.9,
                chunk_id=_LOCATOR_SENTINEL,
            ),
        ),
        retrieval_metadata=LocalRetrievalRunMetadata(
            search_mode="hybrid",
            requested_top_k=1,
            max_context_characters=1_000,
            rerank_enabled=False,
            source_kinds=(CanonicalSourceKind.MEDIA_DB,),
            scope_state="unscoped",
        ),
        started_at=captured_at,
        ended_at=captured_at,
    )
    prompt_id = builder.record_prompt_evidence_set(
        run_id=run_id,
        evidence=(
            LocalPromptEvidenceCapture(
                candidate_rank=1,
                snapshot_text=f"[S1] {_SNAPSHOT_SENTINEL}",
            ),
        ),
        created_at=captured_at,
    )
    return builder, prompt_id


def _real_controller(
    stack: _RealCitationStack,
    builder: CitationTraceBuilder,
    prompt_id: str,
) -> ConsoleChatController:
    async def capture(_draft: str) -> SimpleNamespace:
        return SimpleNamespace(
            context=f"[S1] MEDIA — {_TITLE_SENTINEL}\n{_SNAPSHOT_SENTINEL}",
            citation_builder=builder,
            prompt_evidence_set_id=prompt_id,
        )

    return ConsoleChatController(
        store=stack.store,
        provider_gateway=_RealDirectGateway(
            ("Exact terminal body 🧪\n", "with arbitrary boundaries.")
        ),
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )


def _real_assistant(store: ConsoleChatStore):
    return next(
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    )


def _citation_row_counts(db: CharactersRAGDB) -> dict[str, int]:
    connection = db.get_connection()
    return {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in (
            "rag_citation_traces",
            "rag_evidence_runs",
            "rag_evidence_snapshots",
            "rag_answer_attempt_payloads",
            "rag_trace_evidence_refs",
            "rag_message_trace_owners",
        )
    }


def _capture_logs() -> tuple[StringIO, int]:
    stream = StringIO()
    handler_id = logger.add(stream, format="{message}", level="DEBUG")
    return stream, handler_id


def _assert_content_free_diagnostics(
    log_output: str,
    *,
    sealed_write: SealedCitationWrite | None = None,
) -> None:
    for sentinel in (
        _BODY_SENTINEL,
        _QUERY_SENTINEL,
        _TITLE_SENTINEL,
        _SNAPSHOT_SENTINEL,
        _SOURCE_SENTINEL,
        _LOCATOR_SENTINEL,
        _EXCEPTION_SENTINEL,
    ):
        assert sentinel not in log_output
    if sealed_write is not None:
        assert (
            sealed_write.answer_attempt_payloads[0].body_integrity_hmac
            not in log_output
        )


def test_capability_ready_adapter_arms_callback_and_deferral_in_append() -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)

    def finalizer(body: str) -> None:
        return None

    _, message_id = _append_eligible(store, finalizer)

    assert store._terminal_citation_finalizers[message_id] is finalizer
    assert message_id in store._terminal_persistence_deferred_ids
    assert message_id in store._pending_persistence_message_ids
    assert persistence.create_calls == []
    _assert_terminal_state_paired(store)


@pytest.mark.parametrize(
    ("persistence", "persist"),
    [
        (None, True),
        (_ReadyCitationPersistence(), False),
        (_NoCitationKwargPersistence(), True),
        (_MissingReadinessPersistence(), True),
        (_FalseReadinessPersistence(), True),
        (_RaisingReadinessPersistence(), True),
    ],
    ids=[
        "no-persistence",
        "persist-false",
        "citation-kwarg-missing",
        "readiness-missing",
        "readiness-false",
        "readiness-raises",
    ],
)
def test_capability_ineligible_adapters_do_not_arm_and_keep_ordinary_behavior(
    persistence: _PersistenceBase | None,
    persist: bool,
) -> None:
    store = ConsoleChatStore(persistence=persistence)
    session_id, message_id = _append_eligible(store, persist=persist)

    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)

    store.append_stream_chunk(message_id, "first")
    assert store.get_message(message_id).content == "first"
    if persistence is not None and persist:
        assert len(persistence.create_calls) == 1
        assert persistence.create_calls[0]["content"] == "first"
    else:
        assert persistence is None or persistence.create_calls == []
    assert store.messages_for_session(session_id)[-1].content == "first"


def test_capability_signature_inspection_exception_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)

    def raise_signature_error(value: object) -> object:
        raise RuntimeError("signature-sentinel")

    monkeypatch.setattr(
        console_chat_store_module.inspect, "signature", raise_signature_error
    )

    _, message_id = _append_eligible(store)

    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)


@pytest.mark.parametrize(
    "append_kwargs",
    [
        {"terminal_citation_finalizer": object()},
        {
            "role": ConsoleMessageRole.USER,
            "terminal_citation_finalizer": lambda body: None,
        },
        {
            "content": "already present",
            "terminal_citation_finalizer": lambda body: None,
        },
        {
            "attachments": (
                MessageAttachment(
                    data=b"image",
                    mime_type="image/png",
                    display_name="image.png",
                    position=0,
                ),
            ),
            "terminal_citation_finalizer": lambda body: None,
        },
        {
            "image_data": b"scalar-image",
            "image_mime_type": "image/png",
            "terminal_citation_finalizer": lambda body: None,
        },
    ],
    ids=[
        "non-callable",
        "non-assistant",
        "non-empty",
        "attachments",
        "scalar-image",
    ],
)
def test_placement_invalid_callback_rejected_before_tree_or_session_mutation(
    append_kwargs: dict[str, Any],
) -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    original_updated_at = session.updated_at
    kwargs: dict[str, Any] = {
        "role": ConsoleMessageRole.ASSISTANT,
        "content": "",
        "persist": True,
    }
    kwargs.update(append_kwargs)

    with pytest.raises(ValueError):
        store.append_message(session.id, **kwargs)

    assert store.messages_for_session(session.id) == []
    assert store.active_leaf(session.id) is None
    assert session.updated_at == original_updated_at
    assert persistence.create_calls == []
    _assert_terminal_state_paired(store)


def test_deferred_reads_materialize_chunks_without_early_create() -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session_id, message_id = _append_eligible(store)

    store.append_stream_chunk(message_id, "α")
    assert store.get_message(message_id).content == "α"
    store.append_stream_chunk(message_id, "β")
    assert store.messages_for_session(session_id)[-1].content == "αβ"

    assert persistence.create_calls == []
    assert message_id in store._pending_persistence_message_ids
    _assert_terminal_state_paired(store)


def test_explicit_persist_keeps_terminal_message_deferred_until_completion() -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    sealed_write = _sealed_write_for_body(_BODY_SENTINEL)
    session_id, message_id = _append_eligible(store, lambda body: sealed_write)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)

    assert store.get_message(message_id).content == _BODY_SENTINEL
    assert store.messages_for_session(session_id)[-1].content == _BODY_SENTINEL
    explicitly_persisted = store.persist_message_if_needed(message_id)

    assert explicitly_persisted.persisted_message_id is None
    assert persistence.create_calls == []
    assert message_id in store._pending_persistence_message_ids
    assert message_id in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)

    completed = store.mark_message_complete(message_id)

    assert completed.persisted_message_id == message_id
    assert len(persistence.create_calls) == 1
    assert persistence.create_calls[0]["message_id"] == message_id
    assert persistence.create_calls[0]["citation_write"] is sealed_write


def test_ordinary_empty_assistant_keeps_first_content_persistence_timing() -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.append_stream_chunk(message.id, "first")
    assert persistence.create_calls == []
    assert store.get_message(message.id).content == "first"

    assert len(persistence.create_calls) == 1
    assert persistence.create_calls[0]["content"] == "first"
    assert message.id not in store._pending_persistence_message_ids


def test_append_failure_clears_callback_and_deferral_before_reraising() -> None:
    store = ConsoleChatStore(persistence=_FailingConversationPersistence())
    session = store.ensure_session()

    with pytest.raises(RuntimeError, match="append-setup-sentinel"):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
            terminal_citation_finalizer=lambda body: None,
        )

    registered_ids = set(store._nodes_by_session[session.id])
    assert len(registered_ids) == 1
    message_id = registered_ids.pop()
    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)


def test_cleanup_clear_is_idempotent_and_preserves_ordinary_pending() -> None:
    store = ConsoleChatStore(persistence=_ReadyCitationPersistence())
    _, message_id = _append_eligible(store)

    store.clear_terminal_citation_state(message_id)
    store.clear_terminal_citation_state(message_id)

    assert message_id in store._pending_persistence_message_ids
    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)


def test_cleanup_close_session_sweeps_terminal_state() -> None:
    store = ConsoleChatStore(persistence=_ReadyCitationPersistence())
    session_id, message_id = _append_eligible(store)

    store.close_session(session_id)

    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)


def test_cleanup_subtree_deletion_sweeps_terminal_state() -> None:
    store = ConsoleChatStore(persistence=_ReadyCitationPersistence())
    session = store.ensure_session()
    parent = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="complete parent",
    )
    _, child_id = _append_eligible(store)

    store.delete_message(parent.id)

    with pytest.raises(KeyError):
        store.get_message(child_id)
    assert child_id not in store._terminal_citation_finalizers
    assert child_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)


def test_cleanup_restore_state_sweeps_terminal_state() -> None:
    store = ConsoleChatStore(persistence=_ReadyCitationPersistence())
    _append_eligible(store)

    store.restore_state(sessions=[])

    assert store._terminal_citation_finalizers == {}
    assert store._terminal_persistence_deferred_ids == set()
    _assert_terminal_state_paired(store)


@pytest.mark.parametrize(
    ("terminal_method_name", "expected_status"),
    [("mark_message_stopped", "stopped"), ("mark_message_failed", "failed")],
)
def test_cleanup_non_success_terminal_clears_deferral_before_ordinary_persistence(
    terminal_method_name: str,
    expected_status: str,
) -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    finalizer_calls: list[str] = []
    _, message_id = _append_eligible(
        store, lambda body: finalizer_calls.append(body) or None
    )
    store.append_stream_chunk(message_id, "partial")

    terminal_method = getattr(store, terminal_method_name)
    result = terminal_method(message_id)

    assert result.status == expected_status
    assert result.content == "partial"
    assert finalizer_calls == []
    assert len(persistence.create_calls) == 1
    assert persistence.create_calls[0]["content"] == "partial"
    assert persistence.create_calls[0].get("citation_write") is None
    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)


def test_exact_body_completion_uses_stable_id_and_same_sealed_write_identity() -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    sealed_write = _sealed_write_for_body(_BODY_SENTINEL)
    finalized_bodies: list[str] = []

    def finalize(body: str) -> SealedCitationWrite:
        finalized_bodies.append(body)
        return sealed_write

    session_id, message_id = _append_eligible(store, finalize)
    store.append_stream_chunk(message_id, "Exact terminal ")
    assert store.get_message(message_id).content == "Exact terminal "
    store.append_stream_chunk(message_id, "body 🧪")
    store.append_stream_chunk(message_id, "\nwith arbitrary")
    store.append_stream_chunk(message_id, " boundaries.")

    completed = store.mark_message_complete(message_id)

    assert completed.content == _BODY_SENTINEL
    assert completed.status == "complete"
    assert completed.persisted_message_id == message_id
    assert finalized_bodies == [_BODY_SENTINEL]
    assert len(persistence.create_calls) == 1
    assert persistence.create_calls[0]["content"] == _BODY_SENTINEL
    assert persistence.create_calls[0]["message_id"] == message_id
    assert persistence.create_calls[0]["citation_write"] is sealed_write
    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids
    _assert_terminal_state_paired(store)

    assert store.get_message(message_id).content == _BODY_SENTINEL
    assert store.messages_for_session(session_id)[-1].content == _BODY_SENTINEL
    assert finalized_bodies == [_BODY_SENTINEL]
    assert len(persistence.create_calls) == 1


def test_finalizer_none_uses_one_ordinary_stable_id_create() -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    _, message_id = _append_eligible(store, lambda body: None)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)

    completed = store.mark_message_complete(message_id)

    assert completed.status == "complete"
    assert completed.persisted_message_id == message_id
    assert len(persistence.create_calls) == 1
    assert persistence.create_calls[0]["message_id"] == message_id
    assert "citation_write" not in persistence.create_calls[0]


def test_finalizer_exception_logs_fixed_diagnostic_and_uses_ordinary_stable_id() -> (
    None
):
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)

    def unavailable_finalizer(body: str) -> SealedCitationWrite:
        raise RuntimeError(f"{_EXCEPTION_SENTINEL}:{body}")

    _, message_id = _append_eligible(store, unavailable_finalizer)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)
    log_stream, handler_id = _capture_logs()
    try:
        completed = store.mark_message_complete(message_id)
    finally:
        logger.remove(handler_id)

    output = log_stream.getvalue()
    assert completed.status == "complete"
    assert completed.persisted_message_id == message_id
    assert len(persistence.create_calls) == 1
    assert persistence.create_calls[0]["message_id"] == message_id
    assert "citation_write" not in persistence.create_calls[0]
    assert "terminal_finalizer_unavailable" in output
    _assert_content_free_diagnostics(output)


def test_fallback_first_unavailable_uses_one_ordinary_same_id_create() -> None:
    persistence = _ReadyCitationPersistence(
        outcomes=[CitationPersistenceUnavailable("deterministic"), None]
    )
    store = ConsoleChatStore(persistence=persistence)
    sealed_write = _sealed_write_for_body(_BODY_SENTINEL)
    _, message_id = _append_eligible(store, lambda body: sealed_write)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)

    completed = store.mark_message_complete(message_id)

    assert completed.status == "complete"
    assert completed.persisted_message_id == message_id
    assert len(persistence.create_calls) == 2
    assert [call["message_id"] for call in persistence.create_calls] == [
        message_id,
        message_id,
    ]
    assert persistence.create_calls[0]["citation_write"] is sealed_write
    assert "citation_write" not in persistence.create_calls[1]


def test_ambiguous_first_failure_retries_same_id_and_same_write_once() -> None:
    persistence = _ReadyCitationPersistence(
        outcomes=[RuntimeError(_EXCEPTION_SENTINEL), None]
    )
    store = ConsoleChatStore(persistence=persistence)
    sealed_write = _sealed_write_for_body(_BODY_SENTINEL)
    _, message_id = _append_eligible(store, lambda body: sealed_write)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)

    completed = store.mark_message_complete(message_id)

    assert completed.status == "complete"
    assert completed.persisted_message_id == message_id
    assert len(persistence.create_calls) == 2
    assert [call["message_id"] for call in persistence.create_calls] == [
        message_id,
        message_id,
    ]
    assert all(
        call["citation_write"] is sealed_write for call in persistence.create_calls
    )


@pytest.mark.parametrize(
    "second_failure",
    [
        RuntimeError(f"second:{_EXCEPTION_SENTINEL}"),
        CitationPersistenceUnavailable("second-deterministic"),
    ],
    ids=["ambiguous", "citation-unavailable"],
)
def test_ambiguous_retry_failure_is_abandoned_without_ordinary_insert(
    second_failure: BaseException,
) -> None:
    persistence = _ReadyCitationPersistence(
        outcomes=[RuntimeError(f"first:{_EXCEPTION_SENTINEL}"), second_failure]
    )
    store = ConsoleChatStore(persistence=persistence)
    sealed_write = _sealed_write_for_body(_BODY_SENTINEL)
    session_id, message_id = _append_eligible(store, lambda body: sealed_write)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)
    log_stream, handler_id = _capture_logs()
    try:
        completed = store.mark_message_complete(message_id)
    finally:
        logger.remove(handler_id)

    output = log_stream.getvalue()
    assert completed.status == "complete"
    assert completed.content == _BODY_SENTINEL
    assert completed.persisted_message_id is None
    assert len(persistence.create_calls) == 2
    assert all(
        call["message_id"] == message_id and call["citation_write"] is sealed_write
        for call in persistence.create_calls
    )
    assert message_id not in store._pending_persistence_message_ids
    assert "terminal_citation_persistence_abandoned" in output
    _assert_content_free_diagnostics(output, sealed_write=sealed_write)

    store.get_message(message_id)
    store.messages_for_session(session_id)
    assert len(persistence.create_calls) == 2


def test_fallback_failure_is_abandoned_without_later_polling_create() -> None:
    persistence = _ReadyCitationPersistence(
        outcomes=[
            CitationPersistenceUnavailable("deterministic"),
            RuntimeError(_EXCEPTION_SENTINEL),
        ]
    )
    store = ConsoleChatStore(persistence=persistence)
    sealed_write = _sealed_write_for_body(_BODY_SENTINEL)
    session_id, message_id = _append_eligible(store, lambda body: sealed_write)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)

    completed = store.mark_message_complete(message_id)

    assert completed.status == "complete"
    assert completed.persisted_message_id is None
    assert len(persistence.create_calls) == 2
    assert persistence.create_calls[0]["citation_write"] is sealed_write
    assert "citation_write" not in persistence.create_calls[1]
    assert {call["message_id"] for call in persistence.create_calls} == {message_id}
    assert message_id not in store._pending_persistence_message_ids
    store.get_message(message_id)
    store.messages_for_session(session_id)
    assert len(persistence.create_calls) == 2


def test_finalizer_none_ordinary_failure_is_abandoned_without_later_create() -> None:
    persistence = _ReadyCitationPersistence(
        outcomes=[RuntimeError(_EXCEPTION_SENTINEL)]
    )
    store = ConsoleChatStore(persistence=persistence)
    session_id, message_id = _append_eligible(store, lambda body: None)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)

    completed = store.mark_message_complete(message_id)

    assert completed.status == "complete"
    assert completed.persisted_message_id is None
    assert len(persistence.create_calls) == 1
    assert persistence.create_calls[0]["message_id"] == message_id
    assert "citation_write" not in persistence.create_calls[0]
    assert message_id not in store._pending_persistence_message_ids
    store.get_message(message_id)
    store.messages_for_session(session_id)
    assert len(persistence.create_calls) == 1


def test_empty_terminal_completion_skips_finalizer_and_keeps_ordinary_pending() -> None:
    persistence = _ReadyCitationPersistence()
    store = ConsoleChatStore(persistence=persistence)
    finalizer_calls: list[str] = []
    _, message_id = _append_eligible(
        store, lambda body: finalizer_calls.append(body) or None
    )

    completed = store.mark_message_complete(message_id)

    assert completed.status == "complete"
    assert completed.content == ""
    assert completed.persisted_message_id is None
    assert finalizer_calls == []
    assert persistence.create_calls == []
    assert message_id in store._pending_persistence_message_ids
    assert message_id not in store._terminal_citation_finalizers
    assert message_id not in store._terminal_persistence_deferred_ids


@pytest.mark.parametrize("failure_kind", ["active-leaf", "sync"])
def test_bookkeeping_failure_after_create_never_replays_terminal_create(
    failure_kind: str,
) -> None:
    persistence: _ReadyCitationPersistence
    store_kwargs: dict[str, Any] = {}
    if failure_kind == "active-leaf":
        persistence = _BookkeepingFailurePersistence()
    else:
        persistence = _ReadyCitationPersistence()
        store_kwargs = {
            "sync_v2_chat_producer": _RaisingSyncProducer(),
            "sync_v2_server_profile_id": "server-terminal-store",
        }
    store = ConsoleChatStore(persistence=persistence, **store_kwargs)
    sealed_write = _sealed_write_for_body(_BODY_SENTINEL)
    _, message_id = _append_eligible(store, lambda body: sealed_write)
    store.append_stream_chunk(message_id, _BODY_SENTINEL)
    log_stream, handler_id = _capture_logs()
    try:
        completed = store.mark_message_complete(message_id)
    finally:
        logger.remove(handler_id)

    output = log_stream.getvalue()
    assert completed.status == "complete"
    assert completed.persisted_message_id == message_id
    assert message_id not in store._pending_persistence_message_ids
    assert len(persistence.create_calls) == 1
    assert "terminal_persistence_bookkeeping_unavailable" in output
    _assert_content_free_diagnostics(output, sealed_write=sealed_write)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_atomic_direct_controller_persists_exact_body_and_trace_on_restart(
    real_citation_stack_factory,
) -> None:
    stack = real_citation_stack_factory("real-atomic")
    builder, prompt_id = _real_captured_builder(stack.repository)
    controller = _real_controller(stack, builder, prompt_id)
    log_stream, handler_id = _capture_logs()
    try:
        result = await controller.submit_draft("question")
    finally:
        logger.remove(handler_id)

    assistant = _real_assistant(stack.store)
    persisted = stack.db.get_message_by_id(assistant.id)
    assert result.accepted is True
    assert assistant.status == "complete"
    assert assistant.content == _BODY_SENTINEL
    assert assistant.persisted_message_id == assistant.id
    assert persisted is not None
    assert persisted["id"] == assistant.id
    assert persisted["content"] == assistant.content
    assert _citation_row_counts(stack.db) == {
        "rag_citation_traces": 1,
        "rag_evidence_runs": 1,
        "rag_evidence_snapshots": 1,
        "rag_answer_attempt_payloads": 1,
        "rag_trace_evidence_refs": 1,
        "rag_message_trace_owners": 1,
    }

    connection = stack.db.get_connection()
    trace_row = connection.execute(
        "SELECT trace_id, aggregate_json FROM rag_citation_traces"
    ).fetchone()
    attempt_row = connection.execute(
        """
        SELECT attempt_id, answer_body, body_integrity_hmac
        FROM rag_answer_attempt_payloads
        """
    ).fetchone()
    owner_row = connection.execute(
        """
        SELECT message_id, message_revision, trace_id, state
        FROM rag_message_trace_owners
        """
    ).fetchone()
    aggregate = json.loads(trace_row["aggregate_json"])

    assert attempt_row["attempt_id"] == aggregate["selected_attempt_id"]
    assert attempt_row["answer_body"] == persisted["content"]
    assert aggregate["answer_attempts"][0]["occurrences"] == []
    assert len(aggregate["evidence_runs"]) == 1
    assert len(aggregate["prompt_evidence_sets"]) == 1
    assert _BODY_SENTINEL not in trace_row["aggregate_json"]
    assert attempt_row["body_integrity_hmac"] not in trace_row["aggregate_json"]
    assert "body_integrity_hmac" not in trace_row["aggregate_json"]
    assert owner_row["message_id"] == assistant.id
    assert owner_row["message_revision"] == persisted["version"]
    assert owner_row["trace_id"] == trace_row["trace_id"]
    assert owner_row["state"] == "active"

    log_output = log_stream.getvalue()
    _assert_content_free_diagnostics(log_output)
    assert attempt_row["body_integrity_hmac"] not in log_output

    persisted_version = persisted["version"]
    trace_id = trace_row["trace_id"]
    stack.close()
    reopened = CharactersRAGDB(stack.db_path, client_id=stack.client_id)
    try:
        reopened_identity = load_local_citation_identity_context(reopened)
        assert reopened_identity is not None
        restarted_repository = CitationTraceRepository(
            reopened,
            policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
            identity_context=reopened_identity,
            fingerprint_codec=stack.codec,
        )

        active = restarted_repository.get_active_trace_for_message(
            assistant.id,
            persisted_version,
            assistant.content,
            stack.codec,
        )

        assert active.state is ActiveCitationTraceState.ACTIVE
        assert active.summary is not None
        assert active.summary.trace.trace_id == trace_id
        assert restarted_repository.verify_active_trace_result(active) is True
    finally:
        reopened.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_rollback_deterministic_unavailable_falls_back_without_trace_rows(
    real_citation_stack_factory,
) -> None:
    stack = real_citation_stack_factory(
        "real-rollback",
        _DeterministicallyUnavailableRepository,
    )
    builder, prompt_id = _real_captured_builder(stack.repository)
    controller = _real_controller(stack, builder, prompt_id)
    log_stream, handler_id = _capture_logs()
    try:
        result = await controller.submit_draft("question")
    finally:
        logger.remove(handler_id)

    assistant = _real_assistant(stack.store)
    persisted = stack.db.get_message_by_id(assistant.id)
    assert result.accepted is True
    assert assistant.status == "complete"
    assert assistant.content == _BODY_SENTINEL
    assert assistant.persisted_message_id == assistant.id
    assert persisted is not None
    assert persisted["id"] == assistant.id
    assert persisted["content"] == assistant.content
    assert _citation_row_counts(stack.db) == {
        "rag_citation_traces": 0,
        "rag_evidence_runs": 0,
        "rag_evidence_snapshots": 0,
        "rag_answer_attempt_payloads": 0,
        "rag_trace_evidence_refs": 0,
        "rag_message_trace_owners": 0,
    }

    active = stack.repository.get_active_trace_for_message(
        assistant.id,
        persisted["version"],
        persisted["content"],
        stack.codec,
    )
    assert active.state is ActiveCitationTraceState.NOT_FOUND

    log_output = log_stream.getvalue()
    _assert_content_free_diagnostics(log_output)
    assert builder.answer_attempt_payloads[0].body_integrity_hmac not in log_output
