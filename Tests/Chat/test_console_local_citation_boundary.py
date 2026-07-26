from __future__ import annotations

import gc
import asyncio
import threading
import time
import weakref
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Agents.agent_models import (
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    RunOutcome,
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
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class _RequestBuilder:
    pass


class _WeakCitationBuilder(CitationTraceBuilder):
    """Real citation builder that remains weak-referenceable for lifecycle tests."""

    __slots__ = ("__weakref__",)


_PRIVATE_QUERY = "PRIVATE_QUERY_SENTINEL_TASK_553_14"
_PRIVATE_TITLE = "PRIVATE_TITLE_SENTINEL_TASK_553_14"
_PRIVATE_SNAPSHOT = "PRIVATE_SNAPSHOT_SENTINEL_TASK_553_14"
_PRIVATE_EXCEPTION = "PRIVATE_EXCEPTION_SENTINEL_TASK_553_14"
_MISSING = object()


def _citation_builder(
    *,
    weak_referenceable: bool = False,
) -> tuple[CitationTraceBuilder, str]:
    now = datetime.now(UTC) - timedelta(seconds=2)
    builder_type = _WeakCitationBuilder if weak_referenceable else CitationTraceBuilder
    builder = builder_type.local(
        request_id="request-console-terminal",
        generation_id="generation-console-terminal",
        identity_context=LocalCitationIdentityContext(
            profile_id="profile-console-terminal",
            local_authority_id="authority-console-terminal",
            fingerprint_key_id="fingerprint-console-terminal",
        ),
        fingerprint_codec=CitationFingerprintCodec(b"0123456789abcdef0123456789abcdef"),
        policy_version="console-terminal-policy-v1",
        policy_capabilities=(
            PolicyCapability.VIEW_SNAPSHOT,
            PolicyCapability.VIEW_SOURCE_IDENTITY,
        ),
        created_at=now,
    )
    run_id = builder.record_retrieval_run(
        stage="hybrid",
        raw_query=_PRIVATE_QUERY,
        candidates=(
            LocalRetrievalCandidateCapture(
                candidate_rank=1,
                source_kind=CanonicalSourceKind.MEDIA_DB,
                source_id="media-console-terminal",
                title=_PRIVATE_TITLE,
                score_kind=RetrievalScoreKind.VECTOR_SIMILARITY,
                score_scale=RetrievalScoreScale.ZERO_TO_ONE,
                score=0.9,
                chunk_id="chunk-console-terminal",
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
        started_at=now,
        ended_at=now,
    )
    prompt_id = builder.record_prompt_evidence_set(
        run_id=run_id,
        evidence=(
            LocalPromptEvidenceCapture(
                candidate_rank=1,
                snapshot_text=f"[S1] {_PRIVATE_SNAPSHOT}",
            ),
        ),
        created_at=now,
    )
    return builder, prompt_id


class _ReadyCitationPersistence:
    canonical_citation_writes_ready = True
    db = None

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []

    def create_conversation(self, **_kwargs: Any) -> str:
        return "conversation-1"

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
        return message_id or f"persisted-{len(self.create_calls)}"

    def update_message_content(self, **_kwargs: Any) -> bool:
        return True


class _RecordingGateway:
    def __init__(
        self,
        builder_ref=None,
        *,
        chunks: tuple[str, ...] = ("answer",),
        error: BaseException | None = None,
    ):
        self.builder_ref = builder_ref
        self.chunks = chunks
        self.error = error
        self.messages_seen = None

    async def resolve_for_send(self, _selection):
        return SimpleNamespace(
            ready=True,
            visible_copy="",
            provider="llama_cpp",
            model="test-model",
            max_tokens=128,
        )

    async def stream_chat(self, _resolution, messages):
        if self.builder_ref is not None:
            assert self.builder_ref() is not None
        self.messages_seen = messages
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


def _persisted_store(
    persistence: _ReadyCitationPersistence | None = None,
) -> ConsoleChatStore:
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        settings=ConsoleSessionSettings(provider="llama_cpp")
    )
    session.persisted_conversation_id = "conversation-1"
    return store


def _capture_result(
    builder: CitationTraceBuilder,
    prompt_id: str,
    *,
    context: Any = "[S1] MEDIA — Source\nExact evidence",
) -> SimpleNamespace:
    return SimpleNamespace(
        context=context,
        citation_builder=builder,
        prompt_evidence_set_id=prompt_id,
    )


def _assistant(store: ConsoleChatStore):
    return next(
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    )


def _citation_calls(
    persistence: _ReadyCitationPersistence,
) -> list[dict[str, Any]]:
    return [
        call
        for call in persistence.create_calls
        if call.get("citation_write") is not None
    ]


def _assert_no_terminal_state(store: ConsoleChatStore) -> None:
    assert store._terminal_citation_finalizers == {}
    assert store._terminal_persistence_deferred_ids == set()


def _final_user_content(messages) -> str:
    return next(
        message["content"]
        for message in reversed(messages)
        if message["role"] == ConsoleMessageRole.USER.value
    )


@pytest.mark.asyncio
async def test_console_canonical_evidence_is_added_after_prompt_transforms_and_builder_is_local():
    ordinary_prompt = "ORDINARY_PROMPT_SENTINEL_TASK_553_13"
    transformed_prompt = "TRANSFORMED_PROMPT_SENTINEL_TASK_553_13"
    evidence_title = "EVIDENCE_TITLE_SENTINEL_TASK_553_13"
    evidence_body = "  EVIDENCE_BODY_SENTINEL_TASK_553_13  \n\t"
    canonical_context = f"[S1] MEDIA — {evidence_title}\n{evidence_body}"
    builder, prompt_id = _citation_builder(weak_referenceable=True)
    builder_holder = [builder]
    builder_ref = weakref.ref(builder_holder[0])
    del builder

    async def capture(_draft):
        return SimpleNamespace(
            context=canonical_context,
            citation_builder=builder_holder.pop(),
            prompt_evidence_set_id=prompt_id,
        )

    def apply_dictionary(_conversation_id, text):
        return (
            text.replace(ordinary_prompt, transformed_prompt)
            .replace(evidence_title, "MUTATED_EVIDENCE_TITLE")
            .replace(evidence_body, "MUTATED_EVIDENCE_BODY")
        )

    store = _persisted_store()
    gateway = _RecordingGateway(builder_ref)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        chat_dictionary_applier=apply_dictionary,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft(ordinary_prompt)

    assert result.accepted is True
    stored_user = next(
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.USER
    )
    assert stored_user.content == ordinary_prompt
    provider_user = _final_user_content(gateway.messages_seen)
    assert provider_user == (
        f"Evidence: {canonical_context}\n\n---\n\n{transformed_prompt}"
    )
    assert "MUTATED_EVIDENCE_TITLE" not in provider_user
    assert "MUTATED_EVIDENCE_BODY" not in provider_user
    assert evidence_body.encode("utf-8") in provider_user.encode("utf-8")

    gc.collect()
    assert builder_ref() is None


@pytest.mark.asyncio
async def test_console_without_builder_keeps_compatibility_transform_order():
    context = "[S1] MEDIA — LEGACY_EVIDENCE_TITLE\nlegacy evidence body"

    async def capture(_draft):
        return SimpleNamespace(context=context, citation_builder=None)

    def apply_dictionary(_conversation_id, text):
        return text.replace("legacy evidence body", "transformed evidence body")

    store = _persisted_store()
    gateway = _RecordingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        chat_dictionary_applier=apply_dictionary,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft("question")

    assert result.accepted is True
    assert _final_user_content(gateway.messages_seen) == (
        "Evidence: [S1] MEDIA — LEGACY_EVIDENCE_TITLE\n"
        "transformed evidence body\n\n---\n\nquestion"
    )


@pytest.mark.asyncio
async def test_console_canonical_evidence_reaches_agent_and_keeps_builder_alive():
    context = "[S1] MEDIA — Agent source\nexact agent evidence"
    builder, prompt_id = _citation_builder(weak_referenceable=True)
    builder_holder = [builder]
    builder_ref = weakref.ref(builder_holder[0])
    del builder

    async def capture(_draft):
        return SimpleNamespace(
            context=context,
            citation_builder=builder_holder.pop(),
            prompt_evidence_set_id=prompt_id,
        )

    bridge_calls = []

    def run_reply(**kwargs):
        assert builder_ref() is not None
        bridge_calls.append(kwargs)
        store.append_stream_chunk(kwargs["assistant_message_id"], "agent answer")
        return "run-test", RunOutcome(
            status=RUN_DONE,
            steps=[],
            final_text="agent answer",
        )

    store = _persisted_store()
    gateway = _RecordingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=True,
    )
    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)

    result = await controller.submit_draft("question")

    assert result.accepted is True
    assert len(bridge_calls) == 1
    assert _final_user_content(bridge_calls[0]["agent_messages"]) == (
        f"Evidence: {context}\n\n---\n\nquestion"
    )
    assert gateway.messages_seen is None
    gc.collect()
    assert builder_ref() is None


def test_prepend_evidence_context_preserves_multimodal_parts_and_input():
    original = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        }
    ]

    updated = ConsoleChatController._prepend_evidence_context(
        original,
        "[S1] MEDIA — Image source\nimage evidence",
    )

    assert updated[0]["content"][0]["text"] == (
        "Evidence: [S1] MEDIA — Image source\nimage evidence\n\n---\n\nquestion"
    )
    assert updated[0]["content"][1] == original[0]["content"][1]
    assert original[0]["content"][0]["text"] == "question"


@pytest.mark.asyncio
async def test_console_capture_failure_logs_no_sensitive_text_and_sends_without_evidence():
    failure_sentinel = "CAPTURE_FAILURE_SENTINEL_TASK_553_13"

    async def capture(_draft):
        raise ValueError(failure_sentinel)

    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="DEBUG",
        format="{message}",
    )
    try:
        store = _persisted_store()
        gateway = _RecordingGateway()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            rag_capture_provider=capture,
            agent_runtime_enabled=False,
        )

        result = await controller.submit_draft("question")
    finally:
        loguru_logger.remove(sink_id)

    assert result.accepted is True
    assert _final_user_content(gateway.messages_seen) == "question"
    assert failure_sentinel not in "".join(str(message) for message in captured_logs)


@pytest.mark.asyncio
async def test_direct_initial_rag_success_seals_exact_materialized_body_once():
    body = "Exact direct answer 🧪\nacross chunk boundaries."
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    gateway = _RecordingGateway(
        chunks=("Exact direct ", "answer 🧪\n", "across chunk boundaries.")
    )

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft("question")

    assistant = _assistant(store)
    citation_calls = _citation_calls(persistence)
    assert result.accepted is True
    assert assistant.content == body
    assert assistant.persisted_message_id == assistant.id
    assert len(citation_calls) == 1
    assert citation_calls[0]["content"] == assistant.content
    assert citation_calls[0]["message_id"] == assistant.id
    write = citation_calls[0]["citation_write"]
    assert isinstance(write, SealedCitationWrite)
    assert write.answer_attempt_payloads[0].answer_body == assistant.content
    assert write.trace.answer_attempts[0].prompt_evidence_set_id == prompt_id
    assert builder.is_sealed is True
    assert len(builder.answer_attempts) == 1
    _assert_no_terminal_state(store)


@pytest.mark.asyncio
async def test_direct_initial_rag_prefill_seals_exact_prefill_plus_stream_body():
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    store.set_session_pinned_prefill(store.active_session_id, "Prefill: ")
    gateway = _RecordingGateway(chunks=("streamed ", "answer"))

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )

    await controller.submit_draft("question")

    assistant = _assistant(store)
    citation_calls = _citation_calls(persistence)
    assert assistant.content == "Prefill: streamed answer"
    assert len(citation_calls) == 1
    assert citation_calls[0]["content"] == assistant.content
    assert (
        citation_calls[0]["citation_write"].answer_attempt_payloads[0].answer_body
        == assistant.content
    )
    _assert_no_terminal_state(store)


@pytest.mark.parametrize(
    "body",
    (
        "Native marker [S1] remains ordinary.",
        "Native markers [S1] and [S2] remain ordinary.",
    ),
)
@pytest.mark.asyncio
async def test_direct_marker_answer_uses_ordinary_stable_persistence_and_fixed_diagnostic(
    body: str,
):
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    captured_logs: list[Any] = []
    sink_id = loguru_logger.add(captured_logs.append, format="{message}", level="DEBUG")
    try:
        controller = ConsoleChatController(
            store=store,
            provider_gateway=_RecordingGateway(chunks=(body,)),
            rag_capture_provider=capture,
            agent_runtime_enabled=False,
        )
        await controller.submit_draft("question")
    finally:
        loguru_logger.remove(sink_id)

    assistant = _assistant(store)
    assistant_calls = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    output = "".join(str(message) for message in captured_logs)
    assert assistant.content == body
    assert assistant.persisted_message_id == assistant.id
    assert len(assistant_calls) == 1
    assert assistant_calls[0]["message_id"] == assistant.id
    assert "citation_write" not in assistant_calls[0]
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()
    assert "occurrence_mapping_unavailable" in output
    for sentinel in (
        body,
        _PRIVATE_QUERY,
        _PRIVATE_TITLE,
        _PRIVATE_SNAPSHOT,
    ):
        assert sentinel not in output
    _assert_no_terminal_state(store)


@pytest.mark.parametrize(
    "body",
    (
        r"Escaped literal \[S1] remains eligible.",
        "Inline code `[S1]` remains eligible.",
        "```text\n[S1]\n```\nCode fence remains eligible.",
    ),
)
@pytest.mark.asyncio
async def test_direct_markers_in_markdown_literals_still_seal(body: str):
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(chunks=(body,)),
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    await controller.submit_draft("question")

    citation_calls = _citation_calls(persistence)
    assert len(citation_calls) == 1
    assert (
        citation_calls[0]["citation_write"].answer_attempt_payloads[0].answer_body
        == _assistant(store).content
    )
    assert builder.is_sealed is True
    _assert_no_terminal_state(store)


@pytest.mark.parametrize(
    ("chunks", "error", "raises"),
    (
        ((), None, False),
        (("partial",), RuntimeError("provider failed"), False),
        (("partial",), asyncio.CancelledError(), True),
    ),
    ids=("empty", "provider-error", "cancelled-error"),
)
@pytest.mark.asyncio
async def test_direct_non_success_does_not_seal_and_clears_terminal_state(
    chunks: tuple[str, ...],
    error: BaseException | None,
    raises: bool,
):
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(chunks=chunks, error=error),
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )

    if raises:
        with pytest.raises(asyncio.CancelledError):
            await controller.submit_draft("question")
    else:
        await controller.submit_draft("question")

    assert _citation_calls(persistence) == []
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


@pytest.mark.asyncio
async def test_direct_user_stop_does_not_seal_and_clears_terminal_state():
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    stream_parked = asyncio.Event()

    class _ParkedGateway(_RecordingGateway):
        async def stream_chat(self, _resolution, messages):
            self.messages_seen = messages
            yield "partial"
            stream_parked.set()
            await asyncio.Event().wait()

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_ParkedGateway(),
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await stream_parked.wait()

    assert controller.stop_active_run() is True
    await task

    assert _assistant(store).status == "stopped"
    assert _citation_calls(persistence) == []
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


@pytest.mark.parametrize(
    ("context", "builder_kind", "prompt_id_kind", "expects_context"),
    (
        (None, "valid", "valid", False),
        (123, "valid", "valid", False),
        ("legacy context", "invalid", "valid", True),
        ("canonical context", "valid", "missing", True),
        ("canonical context", "valid", "blank", True),
        ("canonical context", "valid", "invalid", True),
    ),
    ids=(
        "missing-context",
        "invalid-context",
        "invalid-builder",
        "missing-prompt-id",
        "blank-prompt-id",
        "invalid-prompt-id",
    ),
)
@pytest.mark.asyncio
async def test_invalid_capture_parts_preserve_context_compatibility_without_finalizer(
    context: Any,
    builder_kind: str,
    prompt_id_kind: str,
    expects_context: bool,
):
    builder, prompt_id = _citation_builder()
    captured_builder: Any = builder if builder_kind == "valid" else _RequestBuilder()
    captured_prompt_id: Any = {
        "valid": prompt_id,
        "missing": None,
        "blank": "  ",
        "invalid": 17,
    }[prompt_id_kind]
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    gateway = _RecordingGateway()

    async def capture(_draft):
        return SimpleNamespace(
            context=context,
            citation_builder=captured_builder,
            prompt_evidence_set_id=captured_prompt_id,
        )

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    await controller.submit_draft("question")

    final_user = _final_user_content(gateway.messages_seen)
    assert ("Evidence:" in final_user) is expects_context
    assert _citation_calls(persistence) == []
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


@pytest.mark.asyncio
async def test_finalizer_exception_logs_only_fixed_content_safe_diagnostic(monkeypatch):
    body = "PRIVATE_ANSWER_SENTINEL_TASK_553_14"
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)

    def fail_attempt(_self, **_kwargs):
        raise RuntimeError(_PRIVATE_EXCEPTION)

    monkeypatch.setattr(
        CitationTraceBuilder,
        "record_initial_answer_attempt",
        fail_attempt,
    )

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    captured_logs: list[Any] = []
    sink_id = loguru_logger.add(captured_logs.append, format="{message}", level="DEBUG")
    try:
        controller = ConsoleChatController(
            store=store,
            provider_gateway=_RecordingGateway(chunks=(body,)),
            rag_capture_provider=capture,
            agent_runtime_enabled=False,
        )
        await controller.submit_draft("question")
    finally:
        loguru_logger.remove(sink_id)

    output = "".join(str(message) for message in captured_logs)
    assistant_calls = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert len(assistant_calls) == 1
    assert "citation_write" not in assistant_calls[0]
    assert "attempt_or_seal_failure" in output
    for sentinel in (
        body,
        _PRIVATE_QUERY,
        _PRIVATE_TITLE,
        _PRIVATE_SNAPSHOT,
        _PRIVATE_EXCEPTION,
    ):
        assert sentinel not in output
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


class _AgentBridge:
    def __init__(
        self,
        store: ConsoleChatStore,
        *,
        status: str = RUN_DONE,
        text: str = "agent answer",
        append_text: bool = True,
    ) -> None:
        self.store = store
        self.status = status
        self.text = text
        self.append_text = append_text
        self.calls: list[dict[str, Any]] = []

    def run_reply(self, **kwargs):
        self.calls.append(kwargs)
        if self.append_text and self.text:
            self.store.append_stream_chunk(
                kwargs["assistant_message_id"],
                self.text,
            )
        return "run-console-terminal", RunOutcome(
            status=self.status,
            steps=[],
            final_text=self.text,
        )

    def record_run_assistant_message(self, _run_id, _message_id):
        return None


@pytest.mark.asyncio
async def test_agent_initial_rag_success_seals_exact_runtime_materialized_body_once():
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    bridge = _AgentBridge(store, text="agent answer")

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        rag_capture_provider=capture,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )

    await controller.submit_draft("question")

    assistant = _assistant(store)
    citation_calls = _citation_calls(persistence)
    assert assistant.content == "agent answer"
    assert len(bridge.calls) == 1
    assert len(citation_calls) == 1
    assert citation_calls[0]["content"] == assistant.content
    assert citation_calls[0]["message_id"] == assistant.id
    assert (
        citation_calls[0]["citation_write"].answer_attempt_payloads[0].answer_body
        == assistant.content
    )
    assert builder.is_sealed is True
    assert len(builder.answer_attempts) == 1
    _assert_no_terminal_state(store)


@pytest.mark.asyncio
async def test_agent_empty_done_disarms_before_ordinary_fallback():
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    bridge = _AgentBridge(store, text="", append_text=False)

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        rag_capture_provider=capture,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    await controller.submit_draft("question")

    assistant = _assistant(store)
    assistant_calls = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert assistant.content == "No response was generated."
    assert len(assistant_calls) == 1
    assert "citation_write" not in assistant_calls[0]
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


@pytest.mark.parametrize("status", (RUN_CANCELLED, RUN_ERROR))
@pytest.mark.asyncio
async def test_agent_non_success_does_not_seal(status: str):
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    bridge = _AgentBridge(store, status=status, text="agent partial")

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        rag_capture_provider=capture,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    await controller.submit_draft("question")

    assert _citation_calls(persistence) == []
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


@pytest.mark.asyncio
async def test_agent_user_stop_does_not_seal():
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    started = threading.Event()

    class _StoppedAgentBridge(_AgentBridge):
        def run_reply(self, **kwargs):
            started.set()
            while not kwargs["should_cancel"]():
                time.sleep(0.001)
            return "run-console-terminal", RunOutcome(
                status=RUN_CANCELLED,
                steps=[],
                final_text="",
            )

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        rag_capture_provider=capture,
        agent_bridge=_StoppedAgentBridge(store),
        agent_runtime_enabled=True,
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await asyncio.to_thread(started.wait)

    assert controller.stop_active_run() is True
    await task

    assert _citation_calls(persistence) == []
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


@pytest.mark.asyncio
async def test_agent_replaced_placeholder_does_not_transfer_finalizer():
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)

    class _ReplacingAgentBridge(_AgentBridge):
        def run_reply(self, **kwargs):
            original_id = kwargs["assistant_message_id"]
            session_id = self.store.session_id_for_message(original_id)
            session = next(
                session for session in self.store.sessions() if session.id == session_id
            )
            retained = [
                message
                for message in self.store.messages_for_session(session_id)
                if message.id != original_id
            ]
            self.store.restore_state(
                sessions=[session],
                messages_by_session={session_id: retained},
                active_session_id=session_id,
            )
            replacement = self.store.append_message(
                session_id,
                role=ConsoleMessageRole.ASSISTANT,
                content="",
                persist=True,
            )
            self.store.append_stream_chunk(replacement.id, "replacement answer")
            return "run-console-terminal", RunOutcome(
                status=RUN_DONE,
                steps=[],
                final_text="replacement answer",
            )

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        rag_capture_provider=capture,
        agent_bridge=_ReplacingAgentBridge(store),
        agent_runtime_enabled=True,
    )
    await controller.submit_draft("question")

    assistant = _assistant(store)
    assistant_calls = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert assistant.content == "replacement answer"
    assert len(assistant_calls) == 1
    assert "citation_write" not in assistant_calls[0]
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    _assert_no_terminal_state(store)


@pytest.mark.asyncio
async def test_retry_and_regenerate_never_inherit_initial_rag_finalizer():
    builder, prompt_id = _citation_builder()
    persistence = _ReadyCitationPersistence()
    store = _persisted_store(persistence)
    gateway = _RecordingGateway(
        chunks=("initial partial",),
        error=RuntimeError("initial failure"),
    )

    async def capture(_draft):
        return _capture_result(builder, prompt_id)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    await controller.submit_draft("question")
    failed = _assistant(store)
    assert failed.status == "failed"
    assert builder.answer_attempts == ()

    gateway.chunks = ("retry answer",)
    gateway.error = None
    await controller.retry_message(failed.id)
    retried = store.get_message(failed.id)
    assert retried.status == "complete"
    assert retried.content == "retry answer"

    gateway.chunks = ("regenerated answer",)
    await controller.regenerate_message(failed.id)
    regenerated = store.get_message(store.active_leaf(store.active_session_id))
    assert regenerated.id != failed.id
    assert regenerated.content == "regenerated answer"

    assert _citation_calls(persistence) == []
    assert builder.is_sealed is False
    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()
    _assert_no_terminal_state(store)
