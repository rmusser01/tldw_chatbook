from __future__ import annotations

import gc
import asyncio
import logging
import threading
import time
import weakref
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Agents.agent_models import (
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    RunOutcome,
)
from tldw_chatbook.Chat.citation_source_locators import CanonicalSourceKind
from tldw_chatbook.Chat.citation_repair import (
    CitationRepairContract,
    build_citation_repair_messages,
)
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
    MarkerNamespace,
    PolicyCapability,
    RetrievalScoreKind,
    RetrievalScoreScale,
    SealedCitationWrite,
    StructuralValidationState,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    NO_PROVIDER_CONTENT_COPY,
    ConsoleProviderResolution,
    ProviderToolCalls,
)
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
_CANCEL_ROW_PERSISTENCE_EXCEPTION = (
    "CANCEL_ROW_PERSISTENCE_EXCEPTION_SENTINEL_TASK_553_15"
)
_CANCEL_ROW_PERSISTENCE_FAILURE_CODE = (
    "citation_repair_cancel_record_persistence_failed"
)
_REPAIR_INITIAL_BODY_SENTINEL = "REPAIR_INITIAL_BODY_SENTINEL_TASK_553_15"
_REPAIR_REPAIRED_BODY_SENTINEL = "REPAIR_REPAIRED_BODY_SENTINEL_TASK_553_15"
_REPAIR_EVIDENCE_SENTINEL = "REPAIR_EVIDENCE_SENTINEL_TASK_553_15"
_REPAIR_SOURCE_IDENTITY_SENTINEL = "REPAIR_SOURCE_IDENTITY_SENTINEL_TASK_553_15"
_REPAIR_LOCATOR_SENTINEL = "REPAIR_LOCATOR_SENTINEL_TASK_553_15"
_REPAIR_FULL_PROMPT_SENTINEL = "REPAIR_FULL_PROMPT_SENTINEL_TASK_553_15"
_REPAIR_PROVIDER_EXCEPTION_SENTINEL = "REPAIR_PROVIDER_EXCEPTION_SENTINEL_TASK_553_15"
_MISSING = object()
_OMITTED = object()


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


class _CancelRowFailingPersistence(_ReadyCitationPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.create_attempts: list[dict[str, Any]] = []
        self.cancel_row_attempts = 0

    def create_message(self, **kwargs: Any) -> str:
        self.create_attempts.append(dict(kwargs))
        if (
            kwargs["sender"] == ConsoleMessageRole.SYSTEM.value
            and kwargs["content"] == "Citation repair canceled by user."
        ):
            self.cancel_row_attempts += 1
            raise RuntimeError(_CANCEL_ROW_PERSISTENCE_EXCEPTION)
        return super().create_message(**kwargs)


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

    async def stream_chat(self, _resolution, messages, signals=None):
        if self.builder_ref is not None:
            assert self.builder_ref() is not None
        self.messages_seen = messages
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class _RecordingCitationStore(ConsoleChatStore):
    def __init__(self, *, persistence=None):
        super().__init__(persistence=persistence)
        self.assistant_append_kwargs: list[dict[str, Any]] = []
        self.message_append_calls: list[dict[str, Any]] = []
        self.completion_calls: list[str] = []
        self.stopped_calls: list[str] = []
        self.failed_calls: list[str] = []
        self.events: list[str] | None = None

    def append_message(self, session_id, *, role, content, **kwargs):
        self.message_append_calls.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "kwargs": dict(kwargs),
            }
        )
        if role is ConsoleMessageRole.ASSISTANT:
            self.assistant_append_kwargs.append(dict(kwargs))
        return super().append_message(
            session_id,
            role=role,
            content=content,
            **kwargs,
        )

    def mark_message_complete(self, message_id):
        if self.events is not None:
            self.events.append("complete")
        self.completion_calls.append(message_id)
        return super().mark_message_complete(message_id)

    def mark_message_stopped(self, message_id):
        self.stopped_calls.append(message_id)
        return super().mark_message_stopped(message_id)

    def mark_message_failed(self, message_id):
        self.failed_calls.append(message_id)
        return super().mark_message_failed(message_id)


class _ScriptedCitationGateway:
    def __init__(
        self,
        scripts: tuple[tuple[object, ...], ...],
        *,
        mark_fallback_calls: frozenset[int] = frozenset(),
    ) -> None:
        self.resolution = ConsoleProviderResolution(
            provider="openai",
            base_url="https://provider.invalid/v1",
            model="repair-model",
            ready=True,
            readiness_key="openai",
            execution_key="openai",
            api_key="secret",
            temperature=0.23,
            top_p=0.87,
            min_p=0.04,
            top_k=17,
            max_tokens=321,
            seed=42,
            presence_penalty=0.15,
            frequency_penalty=0.25,
            reasoning_effort="medium",
            reasoning_summary="auto",
            verbosity="low",
            thinking_effort="high",
            thinking_budget_tokens=777,
            streaming=True,
        )
        self.scripts = scripts
        self.mark_fallback_calls = mark_fallback_calls
        self.calls: list[dict[str, Any]] = []
        self.on_call = None

    async def resolve_for_send(self, _selection):
        return self.resolution

    async def stream_chat(
        self,
        resolution,
        messages,
        tools=_OMITTED,
        signals=_OMITTED,
    ):
        call_index = len(self.calls)
        self.calls.append(
            {
                "resolution": resolution,
                "messages": messages,
                "tools": tools,
                "signals": signals,
            }
        )
        if self.on_call is not None:
            self.on_call(call_index)
        if call_index in self.mark_fallback_calls:
            assert signals is not _OMITTED
            signals.mark_synthetic_fallback()
        for item in self.scripts[call_index]:
            if isinstance(item, BaseException):
                raise item
            yield item


class _ControlledCitationGateway(_ScriptedCitationGateway):
    def __init__(
        self,
        scripts: tuple[tuple[object, ...], ...],
        *,
        repair_call_index: int,
        pause_before_first_chunk: bool = False,
        pause_after_first_chunk: bool = False,
        yield_late_chunk_on_cancel: bool = False,
        pause_initial_after_first_chunk: bool = False,
    ) -> None:
        super().__init__(scripts)
        self.repair_call_index = repair_call_index
        self.pause_before_first_chunk = pause_before_first_chunk
        self.pause_after_first_chunk = pause_after_first_chunk
        self.yield_late_chunk_on_cancel = yield_late_chunk_on_cancel
        self.pause_initial_after_first_chunk = pause_initial_after_first_chunk
        self.repair_started = asyncio.Event()
        self.first_repair_chunk_collected = asyncio.Event()
        self.first_initial_chunk_collected = asyncio.Event()
        self.release_repair = asyncio.Event()

    async def stream_chat(
        self,
        resolution,
        messages,
        tools=_OMITTED,
        signals=_OMITTED,
    ):
        call_index = len(self.calls)
        self.calls.append(
            {
                "resolution": resolution,
                "messages": messages,
                "tools": tools,
                "signals": signals,
            }
        )
        script = self.scripts[call_index]
        if call_index != self.repair_call_index:
            for index, item in enumerate(script):
                if isinstance(item, BaseException):
                    raise item
                yield item
                if index == 0 and self.pause_initial_after_first_chunk:
                    self.first_initial_chunk_collected.set()
                    await self.release_repair.wait()
            return

        self.repair_started.set()
        if self.pause_before_first_chunk:
            try:
                await self.release_repair.wait()
            except asyncio.CancelledError:
                if self.yield_late_chunk_on_cancel:
                    yield "LATE REPAIR MUST NOT WIN [S1]"
                    return
                raise
        for index, item in enumerate(script):
            if isinstance(item, BaseException):
                raise item
            yield item
            if index == 0:
                self.first_repair_chunk_collected.set()
                if self.pause_after_first_chunk:
                    try:
                        await self.release_repair.wait()
                    except asyncio.CancelledError:
                        if self.yield_late_chunk_on_cancel:
                            yield "LATE REPAIR MUST NOT WIN [S1]"
                            return
                        raise


def _persisted_store(
    persistence: _ReadyCitationPersistence | None = None,
) -> ConsoleChatStore:
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        settings=ConsoleSessionSettings(provider="llama_cpp")
    )
    session.persisted_conversation_id = "conversation-1"
    return store


def _repair_contract() -> CitationRepairContract:
    return CitationRepairContract(
        schema_version=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        allowed_ordinals=(1,),
        evidence_context="[S1] MEDIA — Repair source\nExact repair evidence.",
    )


def _privacy_repair_contract() -> CitationRepairContract:
    return CitationRepairContract(
        schema_version=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        allowed_ordinals=(1,),
        evidence_context=(
            f"[S1] {_REPAIR_SOURCE_IDENTITY_SENTINEL} "
            f"{_REPAIR_LOCATOR_SENTINEL}\n"
            f"{_REPAIR_EVIDENCE_SENTINEL}"
        ),
    )


def _repair_governed_sentinels() -> tuple[str, ...]:
    return (
        _REPAIR_INITIAL_BODY_SENTINEL,
        _REPAIR_REPAIRED_BODY_SENTINEL,
        _REPAIR_EVIDENCE_SENTINEL,
        _REPAIR_SOURCE_IDENTITY_SENTINEL,
        _REPAIR_LOCATOR_SENTINEL,
        _REPAIR_FULL_PROMPT_SENTINEL,
        _REPAIR_PROVIDER_EXCEPTION_SENTINEL,
    )


def _assert_content_free_repair_state(*values: object) -> None:
    rendered = "\n".join(repr(value) for value in values)
    for governed_text in _repair_governed_sentinels():
        assert governed_text not in rendered


def _repair_capture(contract: CitationRepairContract) -> SimpleNamespace:
    return SimpleNamespace(
        context=contract.evidence_context,
        citation_builder=None,
        prompt_evidence_set_id=None,
        citation_repair_contract=contract,
    )


def _recording_citation_store(
    persistence: _ReadyCitationPersistence | None = None,
) -> _RecordingCitationStore:
    store = _RecordingCitationStore(persistence=persistence)
    session = store.ensure_session(
        settings=ConsoleSessionSettings(provider="openai", model="repair-model")
    )
    if persistence is not None:
        session.persisted_conversation_id = "conversation-1"
    return store


async def _run_direct_citation_repair(
    scripts: tuple[tuple[object, ...], ...],
    *,
    persistence: _ReadyCitationPersistence | None = None,
    mark_fallback_calls: frozenset[int] = frozenset(),
):
    contract = _repair_contract()
    store = _recording_citation_store(persistence)
    gateway = _ScriptedCitationGateway(
        scripts,
        mark_fallback_calls=mark_fallback_calls,
    )

    async def capture(_draft):
        return _repair_capture(contract)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    result = await controller.submit_draft("question with history")
    return result, controller, store, gateway, contract


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


# Task 3b rebase note: the controller's in-flight bookkeeping is now a
# PER-SESSION map (keyed by owning session id), not a single shared slot --
# these tests predate that and read the ACTIVE session's own entry, which is
# equivalent for every single-session scenario in this file (including
# post-close/post-second-submit re-checks, where `active_session_id` still
# resolves to whichever session -- or `""` -- the assertion cares about).
def _active_citation_repair_session(controller: ConsoleChatController):
    return controller._active_citation_repair_sessions.get(
        controller.store.active_session_id or ""
    )


def _active_assistant_message_id(controller: ConsoleChatController):
    return controller._active_assistant_message_ids.get(
        controller.store.active_session_id or ""
    )


def _active_stream_task(controller: ConsoleChatController):
    return controller._active_stream_tasks.get(
        controller.store.active_session_id or ""
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
async def test_capture_provider_failure_logs_only_structural_context():
    async def capture(_draft):
        raise RuntimeError(_PRIVATE_EXCEPTION)

    controller = ConsoleChatController(
        store=_persisted_store(),
        provider_gateway=_RecordingGateway(),
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="DEBUG",
        format="{message}",
    )
    try:
        captured = await controller._capture_rag_context(_PRIVATE_QUERY)
    finally:
        loguru_logger.remove(sink_id)

    assert captured == (None, None, None, None)
    rendered_logs = "".join(str(message) for message in captured_logs)
    assert "reason=capture_provider_failure" in rendered_logs
    assert f"draft_length={len(_PRIVATE_QUERY)}" in rendered_logs
    assert _PRIVATE_QUERY not in rendered_logs
    assert _PRIVATE_EXCEPTION not in rendered_logs


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
async def test_repair_contract_exact_context_order_follows_all_prompt_transforms():
    ordinary_prompt = "ORDINARY_PROMPT_REPAIR_CONTRACT_ORDER"
    dictionary_prompt = "DICTIONARY_PROMPT_REPAIR_CONTRACT_ORDER"
    world_prompt = "WORLD_PROMPT_REPAIR_CONTRACT_ORDER"
    canonical_context = (
        "[S1] MEDIA — EVIDENCE_TITLE_REPAIR_CONTRACT_ORDER\n"
        "EVIDENCE_BODY_REPAIR_CONTRACT_ORDER"
    )
    contract = CitationRepairContract(
        schema_version=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        allowed_ordinals=(1,),
        evidence_context=canonical_context,
    )

    async def capture(_draft):
        return SimpleNamespace(
            context=canonical_context,
            citation_builder=None,
            prompt_evidence_set_id=None,
            citation_repair_contract=contract,
        )

    def apply_dictionary(_conversation_id, text):
        return (
            text.replace(ordinary_prompt, dictionary_prompt)
            .replace("EVIDENCE_TITLE_REPAIR_CONTRACT_ORDER", "MUTATED_DICT_TITLE")
            .replace("EVIDENCE_BODY_REPAIR_CONTRACT_ORDER", "MUTATED_DICT_BODY")
        )

    def apply_world_info(_conversation_id, text, _history):
        return (
            text.replace(dictionary_prompt, world_prompt)
            .replace("EVIDENCE_TITLE_REPAIR_CONTRACT_ORDER", "MUTATED_WORLD_TITLE")
            .replace("EVIDENCE_BODY_REPAIR_CONTRACT_ORDER", "MUTATED_WORLD_BODY")
        )

    store = _persisted_store()
    gateway = _RecordingGateway(chunks=("answer [S1]",))
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        chat_dictionary_applier=apply_dictionary,
        world_info_applier=apply_world_info,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft(ordinary_prompt)

    assert result.accepted is True
    provider_user = _final_user_content(gateway.messages_seen)
    assert provider_user == f"Evidence: {canonical_context}\n\n---\n\n{world_prompt}"
    assert contract.evidence_context.encode("utf-8") in provider_user.encode("utf-8")
    assert "MUTATED_DICT" not in provider_user
    assert "MUTATED_WORLD" not in provider_user


@pytest.mark.asyncio
async def test_legacy_raw_without_repair_contract_keeps_early_transform_order():
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
    ("body", "expected_markers", "expected_evidence", "expected_states"),
    (
        (
            "Native marker [S1] persists.",
            ("[S1]",),
            (1,),
            (StructuralValidationState.VALID,),
        ),
        (
            "Native markers [S1] and [S2] persist.",
            ("[S1]", "[S2]"),
            (1, None),
            (
                StructuralValidationState.VALID,
                StructuralValidationState.UNKNOWN_MARKER,
            ),
        ),
    ),
)
@pytest.mark.asyncio
async def test_direct_marker_answer_persists_sealed_mapped_occurrences(
    body: str,
    expected_markers: tuple[str, ...],
    expected_evidence: tuple[int | None, ...],
    expected_states: tuple[StructuralValidationState, ...],
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
    write = assistant_calls[0]["citation_write"]
    assert isinstance(write, SealedCitationWrite)
    assert builder.is_sealed is True
    assert write.answer_attempt_payloads[0].answer_body == body
    attempt = write.trace.answer_attempts[0]
    assert attempt.attempt_id == write.trace.selected_attempt_id
    assert tuple(item.raw_marker for item in attempt.occurrences) == expected_markers
    assert (
        tuple(item.evidence_ordinal for item in attempt.occurrences)
        == expected_evidence
    )
    assert tuple(item.structural_state for item in attempt.occurrences) == (
        expected_states
    )
    expected_offsets: list[tuple[int, int]] = []
    search_start = 0
    for marker in expected_markers:
        marker_start = body.index(marker, search_start)
        marker_end = marker_start + len(marker)
        expected_offsets.append((marker_start, marker_end))
        search_start = marker_end
    assert tuple(
        (item.marker_start, item.marker_end) for item in attempt.occurrences
    ) == tuple(expected_offsets)
    assert "occurrence_mapping_unavailable" not in output
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
    write = citation_calls[0]["citation_write"]
    assert write.answer_attempt_payloads[0].answer_body == _assistant(store).content
    assert write.trace.answer_attempts[0].occurrences == ()
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


@pytest.mark.asyncio
async def test_citation_repair_direct_initial_session_defers_without_persistence_or_finalizer():
    observed: list[tuple[object, object, object]] = []
    result, controller, store, gateway, contract = await _run_direct_citation_repair(
        (("Claim [S1]",),)
    )

    assert result.accepted is True
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["signals"] is not _OMITTED
    append_kwargs = store.assistant_append_kwargs[0]
    assert append_kwargs["persist"] is False
    assert append_kwargs["terminal_citation_finalizer"] is None
    assert append_kwargs["defer_terminal_persistence"] is True
    assert store.completion_calls == [_assistant(store).id]
    assert _active_citation_repair_session(controller) is None

    def observe_session(call_index):
        if call_index == 0:
            session = _active_citation_repair_session(controller)
            observed.append((session.contract, session.resolution, session.phase))

    second_store = _recording_citation_store()
    second_gateway = _ScriptedCitationGateway((("Claim [S1]",),))
    second_gateway.on_call = observe_session
    controller.store = second_store
    controller.provider_gateway = second_gateway
    await controller.submit_draft("second question")

    assert len(observed) == 1
    assert observed[0][0] is contract
    assert observed[0][1] is second_gateway.resolution
    assert observed[0][2] == "initial_streaming"


@pytest.mark.asyncio
async def test_citation_repair_cleaned_session_contains_no_governed_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    contract = _privacy_repair_contract()
    real_build_messages = controller_module.build_citation_repair_messages
    exact_repair_messages = real_build_messages(
        contract,
        _REPAIR_INITIAL_BODY_SENTINEL,
    )
    assert exact_repair_messages is not None
    exact_repair_messages = [dict(message) for message in exact_repair_messages]
    exact_repair_messages[0]["content"] += f"\n{_REPAIR_FULL_PROMPT_SENTINEL}"
    assert _REPAIR_FULL_PROMPT_SENTINEL not in contract.evidence_context
    assert _REPAIR_FULL_PROMPT_SENTINEL not in _REPAIR_INITIAL_BODY_SENTINEL
    assert _REPAIR_FULL_PROMPT_SENTINEL not in _REPAIR_REPAIRED_BODY_SENTINEL
    assert _REPAIR_FULL_PROMPT_SENTINEL in repr(exact_repair_messages)

    def build_exact_messages(
        received_contract: CitationRepairContract,
        initial_answer: str,
    ) -> list[dict[str, str]]:
        assert received_contract is contract
        assert initial_answer == _REPAIR_INITIAL_BODY_SENTINEL
        return [dict(message) for message in exact_repair_messages]

    monkeypatch.setattr(
        controller_module,
        "build_citation_repair_messages",
        build_exact_messages,
    )
    persistence = _ReadyCitationPersistence()
    store = _recording_citation_store(persistence)
    gateway = _ScriptedCitationGateway(
        (
            (_REPAIR_INITIAL_BODY_SENTINEL,),
            (RuntimeError(_REPAIR_PROVIDER_EXCEPTION_SENTINEL),),
        )
    )
    retained_sessions: list[object] = []

    def retain_session(call_index: int) -> None:
        if call_index == 0:
            retained_sessions.append(_active_citation_repair_session(controller))

    gateway.on_call = retain_session

    async def capture(_draft: str) -> SimpleNamespace:
        return _repair_capture(contract)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    loguru_records: list[object] = []
    sink_id = loguru_logger.add(
        loguru_records.append,
        level="DEBUG",
        format="{message}",
    )
    caplog.set_level(logging.DEBUG)
    try:
        result = await controller.submit_draft("question")
    finally:
        loguru_logger.remove(sink_id)

    assert len(retained_sessions) == 1
    assert gateway.calls[1]["messages"] == exact_repair_messages
    cleaned_session = retained_sessions[0]
    assert cleaned_session is not None
    assert cleaned_session.contract is None
    assert cleaned_session.resolution is None
    assistant = _assistant(store)
    assert result.visible_copy == assistant.content == _REPAIR_INITIAL_BODY_SENTINEL
    _assert_content_free_repair_state(
        caplog.text,
        loguru_records,
        *(call["signals"] for call in gateway.calls),
        assistant.citation_presentation,
        cleaned_session,
        _sanitize_selected_persistence(
            persistence,
            _REPAIR_INITIAL_BODY_SENTINEL,
        ),
        controller.run_state,
        controller.run_state_history,
        store._terminal_citation_finalizers,
        store._provisional_terminal_selection_ids,
        store._terminal_persistence_deferred_ids,
    )
    assert _active_citation_repair_session(controller) is None


@pytest.mark.asyncio
async def test_citation_repair_missing_owner_privacy_scrubs_session() -> None:
    store = _recording_citation_store()
    gateway = _ScriptedCitationGateway(())
    repair_session = controller_module.ConsoleCitationRepairSession(
        contract=_privacy_repair_contract(),
        resolution=gateway.resolution,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    result = await controller._stream_assistant_response(
        resolution=gateway.resolution,
        provider_messages=[],
        assistant_message_id="missing-assistant",
        citation_repair_session=repair_session,
    )

    assert result.visible_copy == "Session closed."
    assert repair_session.contract is None
    assert repair_session.resolution is None


@pytest.mark.parametrize("failure_seam", ("compaction", "window-bound"))
@pytest.mark.asyncio
async def test_citation_repair_predispatch_exception_privacy_scrubs_session(
    failure_seam: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _recording_citation_store()
    session_id = store.active_session_id
    assert session_id is not None
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        defer_terminal_persistence=True,
    )
    gateway = _ScriptedCitationGateway(())
    repair_session = controller_module.ConsoleCitationRepairSession(
        contract=_privacy_repair_contract(),
        resolution=gateway.resolution,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(_REPAIR_PROVIDER_EXCEPTION_SENTINEL)

    if failure_seam == "compaction":
        monkeypatch.setattr(
            controller,
            "_apply_context_summary_compaction",
            fail,
        )
    else:
        monkeypatch.setattr(controller_module, "bound_messages_to_window", fail)

    with pytest.raises(RuntimeError, match=_REPAIR_PROVIDER_EXCEPTION_SENTINEL):
        await controller._stream_assistant_response(
            resolution=gateway.resolution,
            provider_messages=[{"role": "user", "content": "question"}],
            assistant_message_id=assistant.id,
            citation_repair_session=repair_session,
        )

    assert repair_session.contract is None
    assert repair_session.resolution is None


@pytest.mark.parametrize(
    "failure_mode",
    (
        "request-fit",
        "provider-raise",
        "empty-output",
        "oversized-output",
        "invalid-markers",
        "changed-claims",
        "fallback-bypass",
    ),
)
@pytest.mark.asyncio
async def test_citation_repair_failure_privacy_sentinels_are_confined_to_selected_body(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    initial_body = _REPAIR_INITIAL_BODY_SENTINEL
    repair_scripts: dict[str, tuple[object, ...]] = {
        "provider-raise": (RuntimeError(_REPAIR_PROVIDER_EXCEPTION_SENTINEL),),
        "empty-output": (),
        "oversized-output": (
            _REPAIR_REPAIRED_BODY_SENTINEL + ("x" * (1024 * 1024 + 1)),
        ),
        "invalid-markers": (f"{initial_body} [S2] {_REPAIR_REPAIRED_BODY_SENTINEL}",),
        "changed-claims": (f"{_REPAIR_REPAIRED_BODY_SENTINEL} [S1]",),
    }
    scripts = (
        (initial_body,),
        *(
            ()
            if failure_mode in {"request-fit", "fallback-bypass"}
            else (repair_scripts[failure_mode],)
        ),
    )
    if failure_mode == "request-fit":
        monkeypatch.setattr(
            controller_module,
            "repair_request_fits_model_window",
            lambda *_args, **_kwargs: False,
        )

    contract = _privacy_repair_contract()
    persistence = _ReadyCitationPersistence()
    store = _recording_citation_store(persistence)
    gateway = _ScriptedCitationGateway(
        scripts,
        mark_fallback_calls=(
            frozenset({0}) if failure_mode == "fallback-bypass" else frozenset()
        ),
    )
    retained_sessions: list[object] = []

    def retain_session(call_index: int) -> None:
        if call_index == 0:
            retained_sessions.append(_active_citation_repair_session(controller))

    gateway.on_call = retain_session

    async def capture(_draft: str) -> SimpleNamespace:
        return _repair_capture(contract)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_runtime_enabled=False,
    )
    expected_repair_messages = build_citation_repair_messages(contract, initial_body)
    assert expected_repair_messages is not None
    assert _REPAIR_FULL_PROMPT_SENTINEL not in repr(expected_repair_messages)
    loguru_records: list[object] = []
    sink_id = loguru_logger.add(
        loguru_records.append,
        level="DEBUG",
        format="{message}",
    )
    caplog.set_level(logging.DEBUG)
    try:
        result = await controller.submit_draft("question")
    finally:
        loguru_logger.remove(sink_id)

    assistant = _assistant(store)
    assistant_writes = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert result.visible_copy == assistant.content == initial_body
    if len(gateway.calls) == 2:
        assert gateway.calls[1]["messages"] == expected_repair_messages
    assert len(assistant_writes) == 1
    assert assistant_writes[0]["content"] == initial_body
    sanitized_writes = [
        {
            **call,
            "content": (
                "<selected-body>" if call is assistant_writes[0] else call["content"]
            ),
        }
        for call in persistence.create_calls
    ]
    cleaned_session = retained_sessions[0]
    assert cleaned_session.contract is None
    assert cleaned_session.resolution is None
    assert _active_citation_repair_session(controller) is None
    assert _active_assistant_message_id(controller) is None
    assert _active_stream_task(controller) is None
    assert controller.original_attempt_for_message(assistant.id) is None
    _assert_content_free_repair_state(
        caplog.text,
        loguru_records,
        *(call["signals"] for call in gateway.calls),
        assistant.citation_presentation,
        cleaned_session,
        sanitized_writes,
        controller.run_state,
        controller.run_state_history,
        store._terminal_citation_finalizers,
        store._provisional_terminal_selection_ids,
        store._terminal_persistence_deferred_ids,
    )


@pytest.mark.asyncio
async def test_citation_repair_direct_valid_initial_completes_once_without_repair():
    persistence = _ReadyCitationPersistence()
    result, controller, store, gateway, _contract = await _run_direct_citation_repair(
        (("Supported claim [S1]",),),
        persistence=persistence,
    )

    assistant = _assistant(store)
    assistant_writes = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert result.visible_copy == assistant.content == "Supported claim [S1]"
    assert len(gateway.calls) == 1
    assert store.completion_calls == [assistant.id]
    assert len(assistant_writes) == 1
    assert assistant_writes[0]["message_id"] == assistant.id
    assert assistant_writes[0]["content"] == assistant.content
    assert assistant.citation_presentation.phase is ConsoleCitationPhase.SELECTED
    assert assistant.citation_presentation.notice_code is None
    assert controller.original_attempt_for_message(assistant.id) is None
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED
    assert ConsoleRunStatus.CHECKING_CITATIONS not in controller.run_state_history


@pytest.mark.parametrize(
    "initial_body",
    (
        "Supported claim",
        "Supported claim [S9]",
    ),
    ids=("missing", "invalid"),
)
@pytest.mark.asyncio
async def test_citation_repair_direct_repairs_once_with_exact_request_and_resolution(
    initial_body: str,
):
    persistence = _ReadyCitationPersistence()
    result, controller, store, gateway, contract = await _run_direct_citation_repair(
        ((initial_body,), ("Supported claim [S1]",)),
        persistence=persistence,
    )

    assistant = _assistant(store)
    assistant_writes = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert len(gateway.calls) == 2
    assert gateway.calls[0]["resolution"] is gateway.resolution
    assert gateway.calls[1]["resolution"] is gateway.resolution
    assert gateway.calls[1]["messages"] == build_citation_repair_messages(
        contract,
        initial_body,
    )
    assert gateway.calls[1]["tools"] is _OMITTED
    assert gateway.calls[0]["signals"] is gateway.calls[1]["signals"]
    assert gateway.calls[0]["signals"] is not _OMITTED
    assert result.visible_copy == assistant.content == "Supported claim [S1]"
    assert store.completion_calls == [assistant.id]
    assert len(assistant_writes) == 1
    assert assistant_writes[0]["message_id"] == assistant.id
    assert assistant_writes[0]["content"] == "Supported claim [S1]"
    assert assistant.citation_presentation.phase is ConsoleCitationPhase.SELECTED
    assert (
        assistant.citation_presentation.notice_code
        is ConsoleCitationNoticeCode.REPAIRED
    )
    assert assistant.citation_presentation.original_attempt_available is True
    assert controller.original_attempt_for_message(assistant.id) == initial_body
    assert ConsoleRunStatus.CHECKING_CITATIONS in controller.run_state_history
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


def test_original_attempt_cache_is_eight_entry_access_ordered_lru():
    store = _recording_citation_store()
    session_id = store.active_session_id
    assert session_id is not None
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        agent_runtime_enabled=False,
    )
    messages = [
        store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=f"Repaired answer {index} [S1]",
        )
        for index in range(9)
    ]
    for index, message in enumerate(messages[:8]):
        store.set_citation_presentation(
            message.id,
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
            ),
        )
        controller._remember_original_attempt(message.id, f"original {index}")

    assert controller.original_attempt_for_message(messages[0].id) == "original 0"
    store.set_citation_presentation(
        messages[8].id,
        ConsoleCitationPresentation(
            phase=ConsoleCitationPhase.SELECTED,
            notice_code=ConsoleCitationNoticeCode.REPAIRED,
        ),
    )
    controller._remember_original_attempt(messages[8].id, "original 8")

    assert controller.original_attempt_for_message(messages[1].id) is None
    assert controller.original_attempt_for_message(messages[0].id) == "original 0"
    assert controller.original_attempt_for_message(messages[8].id) == "original 8"
    assert (
        store.get_message(
            messages[1].id
        ).citation_presentation.original_attempt_available
        is False
    )
    assert len(controller._original_attempts) == 8

    controller.clear_original_attempt(messages[8].id)
    assert controller.original_attempt_for_message(messages[8].id) is None
    assert (
        store.get_message(
            messages[8].id
        ).citation_presentation.original_attempt_available
        is False
    )


@pytest.mark.asyncio
async def test_original_attempt_cache_cleans_up_and_is_never_reconstructed():
    store = _recording_citation_store()
    first_session_id = store.active_session_id
    assert first_session_id is not None
    first = store.append_message(
        first_session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="First repaired [S1]",
    )
    second_session = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="repair-model")
    )
    second = store.append_message(
        second_session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Second repaired [S1]",
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        agent_runtime_enabled=False,
    )
    for message in (first, second):
        store.set_citation_presentation(
            message.id,
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
            ),
        )
        controller._remember_original_attempt(message.id, f"original {message.id}")

    controller.close_session(first_session_id)

    assert controller.original_attempt_for_message(first.id) is None
    assert controller.original_attempt_for_message(second.id) == f"original {second.id}"

    restarted = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        agent_runtime_enabled=False,
    )
    assert restarted.original_attempt_for_message(second.id) is None

    await controller.shutdown()
    assert controller.original_attempt_for_message(second.id) is None


@pytest.mark.parametrize(
    "initial_body",
    (
        "x" * (1024 * 1024 + 1),
        "[S1]" * 1_001,
    ),
    ids=("oversized", "marker-flood"),
)
@pytest.mark.asyncio
async def test_citation_repair_direct_unavailable_initial_keeps_original_without_call(
    initial_body: str,
):
    persistence = _ReadyCitationPersistence()
    result, controller, store, gateway, _contract = await _run_direct_citation_repair(
        ((initial_body,),),
        persistence=persistence,
    )

    assistant = _assistant(store)
    assistant_writes = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert len(gateway.calls) == 1
    assert result.visible_copy == assistant.content == initial_body
    assert store.completion_calls == [assistant.id]
    assert len(assistant_writes) == 1
    assert assistant.citation_presentation.phase is ConsoleCitationPhase.SELECTED
    assert (
        assistant.citation_presentation.notice_code
        is ConsoleCitationNoticeCode.UNAVAILABLE
    )
    assert assistant.citation_presentation.original_attempt_available is False
    assert controller.original_attempt_for_message(assistant.id) is None
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.parametrize(
    "repair_script",
    (
        (RuntimeError("private provider failure"),),
        (),
        (ProviderToolCalls(tool_calls=()),),
        ("Supported claim",),
        ("Supported claim [S2]",),
        ("Changed claim [S1]",),
        ("[S1]" * 1_001,),
        ("x" * (1024 * 1024 + 1),),
    ),
    ids=(
        "provider-raise",
        "empty",
        "tool-call",
        "second-missing",
        "unknown-marker",
        "changed-claim",
        "marker-flood",
        "oversized",
    ),
)
@pytest.mark.asyncio
async def test_citation_repair_direct_failure_keeps_original_and_completes_once(
    repair_script: tuple[object, ...],
):
    persistence = _ReadyCitationPersistence()
    initial_body = "Supported claim"
    result, controller, store, gateway, _contract = await _run_direct_citation_repair(
        ((initial_body,), repair_script),
        persistence=persistence,
    )

    assistant = _assistant(store)
    assistant_writes = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert len(gateway.calls) == 2
    assert result.visible_copy == assistant.content == initial_body
    assert store.completion_calls == [assistant.id]
    assert len(assistant_writes) == 1
    assert assistant_writes[0]["content"] == initial_body
    assert assistant.citation_presentation.phase is ConsoleCitationPhase.SELECTED
    assert (
        assistant.citation_presentation.notice_code
        is ConsoleCitationNoticeCode.UNAVAILABLE
    )
    assert assistant.citation_presentation.original_attempt_available is False
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_citation_repair_direct_failed_window_fit_keeps_original(monkeypatch):
    from tldw_chatbook.Chat import console_chat_controller

    monkeypatch.setattr(
        console_chat_controller,
        "repair_request_fits_model_window",
        lambda *_args, **_kwargs: False,
    )
    persistence = _ReadyCitationPersistence()
    result, _controller, store, gateway, _contract = await _run_direct_citation_repair(
        (("Supported claim",),),
        persistence=persistence,
    )

    assistant = _assistant(store)
    assert len(gateway.calls) == 1
    assert result.visible_copy == assistant.content == "Supported claim"
    assert (
        assistant.citation_presentation.notice_code
        is ConsoleCitationNoticeCode.UNAVAILABLE
    )


@pytest.mark.parametrize(
    ("mark_initial", "expected_calls", "expected_body"),
    (
        (True, 1, NO_PROVIDER_CONTENT_COPY),
        (False, 2, f"{NO_PROVIDER_CONTENT_COPY} [S1]"),
    ),
    ids=("marked-bypass", "genuine-equal-text-repairs"),
)
@pytest.mark.asyncio
async def test_citation_repair_direct_initial_signal_is_source_provenance(
    mark_initial: bool,
    expected_calls: int,
    expected_body: str,
):
    result, _controller, store, gateway, _contract = await _run_direct_citation_repair(
        (
            (NO_PROVIDER_CONTENT_COPY,),
            (f"{NO_PROVIDER_CONTENT_COPY} [S1]",),
        ),
        mark_fallback_calls=frozenset({0}) if mark_initial else frozenset(),
    )

    assert len(gateway.calls) == expected_calls
    assert result.visible_copy == _assistant(store).content == expected_body


@pytest.mark.asyncio
async def test_citation_repair_direct_marked_repair_output_cannot_be_selected():
    result, _controller, store, gateway, _contract = await _run_direct_citation_repair(
        (("Supported claim",), ("Supported claim [S1]",)),
        mark_fallback_calls=frozenset({1}),
    )

    assistant = _assistant(store)
    assert len(gateway.calls) == 2
    assert gateway.calls[0]["signals"] is gateway.calls[1]["signals"]
    assert result.visible_copy == assistant.content == "Supported claim"
    assert (
        assistant.citation_presentation.notice_code
        is ConsoleCitationNoticeCode.UNAVAILABLE
    )


@pytest.mark.parametrize(
    "initial_script",
    (
        (),
        (RuntimeError("private initial failure"),),
    ),
    ids=("empty", "provider-failure"),
)
@pytest.mark.asyncio
async def test_citation_repair_direct_initial_non_success_never_repairs(
    initial_script: tuple[object, ...],
):
    _result, controller, store, gateway, _contract = await _run_direct_citation_repair(
        (initial_script,)
    )

    assert len(gateway.calls) == 1
    assert store.completion_calls == []
    assert _assistant(store).status == "failed"
    assert controller.run_state.status is ConsoleRunStatus.FAILED


def _observe_clean_lifecycle_request(controller, store, observed):
    def observe(call_index):
        if call_index != 1:
            return
        observed.append(
            {
                "repair_session": _active_citation_repair_session(controller),
                "finalizers": dict(store._terminal_citation_finalizers),
                "provisional_ids": set(store._provisional_terminal_selection_ids),
                "deferred_ids": set(store._terminal_persistence_deferred_ids),
            }
        )

    return observe


def _assert_clean_lifecycle_call(controller, store, gateway, observed):
    assert observed == [
        {
            "repair_session": None,
            "finalizers": {},
            "provisional_ids": set(),
            "deferred_ids": set(),
        }
    ]
    assert len(gateway.calls) == 2
    assert gateway.calls[0]["signals"] is not _OMITTED
    assert gateway.calls[1]["signals"] is _OMITTED
    assert _active_citation_repair_session(controller) is None
    assert store._terminal_citation_finalizers == {}
    assert store._provisional_terminal_selection_ids == set()
    assert store._terminal_persistence_deferred_ids == set()


@pytest.mark.parametrize(
    ("lifecycle", "replacement_body"),
    (
        ("regenerate", "regenerated reply without markers"),
        ("edit_resend", "edited reply without markers"),
        ("continue", "continued reply without markers"),
    ),
)
@pytest.mark.asyncio
async def test_citation_repair_direct_completed_initial_does_not_leak_into_lifecycle(
    lifecycle: str,
    replacement_body: str,
):
    initial_body = "Original supported claim [S1]"
    persistence = _ReadyCitationPersistence()
    _result, controller, store, gateway, _contract = await _run_direct_citation_repair(
        ((initial_body,), (replacement_body,)),
        persistence=persistence,
    )
    initial_user = next(
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.USER
    )
    initial_assistant = _assistant(store)
    assert initial_assistant.content == initial_body
    _assert_no_terminal_state(store)
    assert store._provisional_terminal_selection_ids == set()

    observed = []
    gateway.on_call = _observe_clean_lifecycle_request(controller, store, observed)
    if lifecycle == "regenerate":
        lifecycle_result = await controller.regenerate_message(initial_assistant.id)
    elif lifecycle == "edit_resend":
        lifecycle_result = await controller.edit_and_resend_message(
            initial_user.id,
            "edited question",
        )
    else:
        lifecycle_result = await controller.continue_from_message(initial_assistant.id)

    assert lifecycle_result.accepted is True
    replacement = store.get_message(store.active_leaf(store.active_session_id))
    assert replacement.id != initial_assistant.id
    assert replacement.content == replacement_body
    assert replacement.status == "complete"
    assert replacement.variants is None
    unchanged_initial = store.get_message(initial_assistant.id)
    assert unchanged_initial.content == initial_body
    assert unchanged_initial.variants is None
    assert store.get_message(initial_user.id).content == "question with history"
    if lifecycle == "edit_resend":
        active_users = [
            message.content
            for message in store.messages_for_session(store.active_session_id)
            if message.role is ConsoleMessageRole.USER
        ]
        assert active_users == ["edited question"]
    _assert_clean_lifecycle_call(controller, store, gateway, observed)


@pytest.mark.asyncio
async def test_citation_repair_direct_failed_initial_does_not_leak_into_retry():
    replacement_body = "retried reply without markers"
    persistence = _ReadyCitationPersistence()
    _result, controller, store, gateway, _contract = await _run_direct_citation_repair(
        (
            (RuntimeError("private initial failure"),),
            (replacement_body,),
        ),
        persistence=persistence,
    )
    failed = _assistant(store)
    assert failed.status == "failed"
    _assert_no_terminal_state(store)
    assert store._provisional_terminal_selection_ids == set()

    observed = []
    gateway.on_call = _observe_clean_lifecycle_request(controller, store, observed)
    retry_result = await controller.retry_message(failed.id)

    retried = store.get_message(failed.id)
    assert retry_result.accepted is True
    assert retried.id == failed.id
    assert retried.content == replacement_body
    assert retried.status == "complete"
    assert retried.variants is None
    assert store.completion_calls == [failed.id]
    _assert_clean_lifecycle_call(controller, store, gateway, observed)


class _AgentBridge:
    def __init__(
        self,
        store: ConsoleChatStore,
        *,
        status: str = RUN_DONE,
        text: str = "agent answer",
        append_text: bool = True,
        outcome_text: str | None = None,
        mark_synthetic_fallback: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.store = store
        self.status = status
        self.text = text
        self.append_text = append_text
        self.outcome_text = text if outcome_text is None else outcome_text
        self.mark_synthetic_fallback = mark_synthetic_fallback
        self.events = events
        self.calls: list[dict[str, Any]] = []
        self.anchors: list[tuple[str, str]] = []

    def run_reply(self, **kwargs):
        self.calls.append(kwargs)
        if self.mark_synthetic_fallback:
            kwargs["provider_stream_signals"].mark_synthetic_fallback()
        if self.append_text and self.text:
            self.store.append_stream_chunk(
                kwargs["assistant_message_id"],
                self.text,
            )
        return "run-console-terminal", RunOutcome(
            status=self.status,
            steps=[],
            final_text=self.outcome_text,
        )

    def record_run_assistant_message(self, run_id, message_id):
        if self.events is not None:
            self.events.append("anchor")
        self.anchors.append((run_id, message_id))


async def _run_agent_citation_repair(
    *,
    initial_body: str,
    repair_scripts: tuple[tuple[object, ...], ...] = (),
    persistence: _ReadyCitationPersistence | None = None,
    status: str = RUN_DONE,
    append_text: bool = True,
    outcome_text: str | None = None,
    mark_synthetic_fallback: bool = False,
    events: list[str] | None = None,
):
    contract = _repair_contract()
    store = _recording_citation_store(persistence)
    store.events = events
    gateway = _ScriptedCitationGateway(repair_scripts)
    if events is not None:
        gateway.on_call = lambda _call_index: events.append("repair")
    bridge = _AgentBridge(
        store,
        status=status,
        text=initial_body,
        append_text=append_text,
        outcome_text=outcome_text,
        mark_synthetic_fallback=mark_synthetic_fallback,
        events=events,
    )

    async def capture(_draft):
        return _repair_capture(contract)

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    result = await controller.submit_draft("question")
    return result, controller, store, gateway, bridge, contract


@pytest.mark.asyncio
async def test_citation_repair_agent_run_done_selects_exact_store_body_before_completion_and_anchor():
    initial_body = "Store-owned claim without a marker"
    repaired_body = f"{initial_body} [S1]"
    outcome_body = "Divergent outcome body [S1]"
    events: list[str] = []
    persistence = _ReadyCitationPersistence()

    (
        result,
        _controller,
        store,
        gateway,
        bridge,
        contract,
    ) = await _run_agent_citation_repair(
        initial_body=initial_body,
        repair_scripts=((repaired_body,),),
        persistence=persistence,
        outcome_text=outcome_body,
        events=events,
    )
    assert store.completion_calls == [_assistant(store).id]
    assert bridge.anchors

    assistant = _assistant(store)
    assert result.visible_copy == assistant.content == repaired_body
    assert len(bridge.calls) == 1
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["resolution"] is gateway.resolution
    assert gateway.calls[0]["messages"] == build_citation_repair_messages(
        contract,
        initial_body,
    )
    assert gateway.calls[0]["tools"] is _OMITTED
    assert bridge.calls[0]["provider_stream_signals"] is gateway.calls[0]["signals"]
    assert events == ["repair", "complete", "anchor"]
    assert bridge.anchors == [("run-console-terminal", assistant.persisted_message_id)]
    assert assistant.persisted_message_id == assistant.id
    assert outcome_body not in assistant.content


@pytest.mark.parametrize(
    (
        "status",
        "initial_body",
        "append_text",
        "mark_fallback",
        "expected_status",
        "expected_body",
    ),
    (
        (RUN_ERROR, "partial failure", True, False, "failed", "partial failure"),
        (
            RUN_CANCELLED,
            "partial cancellation",
            True,
            False,
            "failed",
            "partial cancellation",
        ),
        (RUN_DONE, "", False, False, "complete", "No response was generated."),
        (
            RUN_DONE,
            NO_PROVIDER_CONTENT_COPY,
            True,
            True,
            "complete",
            NO_PROVIDER_CONTENT_COPY,
        ),
    ),
    ids=("failure", "runtime-cancel", "empty", "synthesized-fallback"),
)
@pytest.mark.asyncio
async def test_citation_repair_agent_ineligible_outcomes_never_dispatch(
    status: str,
    initial_body: str,
    append_text: bool,
    mark_fallback: bool,
    expected_status: str,
    expected_body: str,
):
    (
        result,
        _controller,
        store,
        gateway,
        _bridge,
        _contract,
    ) = await _run_agent_citation_repair(
        initial_body=initial_body,
        status=status,
        append_text=append_text,
        mark_synthetic_fallback=mark_fallback,
    )

    assistant = _assistant(store)
    assert gateway.calls == []
    assert assistant.status == expected_status
    assert assistant.content == expected_body
    assert result.accepted is True


@pytest.mark.asyncio
async def test_citation_repair_agent_missing_placeholder_keeps_runtime_row_without_repair():
    store = _recording_citation_store()
    gateway = _ScriptedCitationGateway(())
    contract = _repair_contract()

    class _ReplacingBridge(_AgentBridge):
        def run_reply(self, **kwargs):
            self.calls.append(kwargs)
            original_id = kwargs["assistant_message_id"]
            session_id = self.store.session_id_for_message(original_id)
            session = next(
                item for item in self.store.sessions() if item.id == session_id
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
                persist=False,
            )
            self.store.append_stream_chunk(replacement.id, "runtime replacement")
            return "run-console-terminal", RunOutcome(
                status=RUN_DONE,
                steps=[],
                final_text="runtime replacement",
            )

    async def capture(_draft):
        return _repair_capture(contract)

    bridge = _ReplacingBridge(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )

    result = await controller.submit_draft("question")

    assistant = _assistant(store)
    assert gateway.calls == []
    assert assistant.content == "runtime replacement"
    assert assistant.status == "complete"
    assert result.visible_copy == assistant.content


@pytest.mark.asyncio
async def test_citation_repair_agent_genuine_fallback_copy_still_repairs():
    repaired = f"{NO_PROVIDER_CONTENT_COPY} [S1]"
    (
        result,
        _controller,
        store,
        gateway,
        bridge,
        _contract,
    ) = await _run_agent_citation_repair(
        initial_body=NO_PROVIDER_CONTENT_COPY,
        repair_scripts=((repaired,),),
    )

    assert len(gateway.calls) == 1
    assert (
        bridge.calls[0]["provider_stream_signals"].synthetic_fallback_emitted is False
    )
    assert result.visible_copy == _assistant(store).content == repaired


def _controlled_citation_repair(
    *,
    agent: bool,
    persistence: _ReadyCitationPersistence | None = None,
    contract: CitationRepairContract | None = None,
    initial_body: str = "Original claim without marker",
    repaired_body: str | None = None,
    pause_before_first_chunk: bool = False,
    pause_after_first_chunk: bool = False,
    yield_late_chunk_on_cancel: bool = False,
    pause_initial_after_first_chunk: bool = False,
):
    repaired_body = repaired_body or f"{initial_body} [S1]"
    repair_call_index = 0 if agent else 1
    scripts = ((repaired_body,),) if agent else ((initial_body,), (repaired_body,))
    gateway = _ControlledCitationGateway(
        scripts,
        repair_call_index=repair_call_index,
        pause_before_first_chunk=pause_before_first_chunk,
        pause_after_first_chunk=pause_after_first_chunk,
        yield_late_chunk_on_cancel=yield_late_chunk_on_cancel,
        pause_initial_after_first_chunk=pause_initial_after_first_chunk,
    )
    store = _recording_citation_store(persistence)
    contract = contract or _repair_contract()

    async def capture(_draft):
        return _repair_capture(contract)

    bridge = _AgentBridge(store, text=initial_body) if agent else None
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
        agent_bridge=bridge,
        agent_runtime_enabled=agent,
    )
    return controller, store, gateway, bridge, initial_body, repaired_body


async def _wait_for_citation_checking(controller: ConsoleChatController) -> None:
    for _ in range(1_000):
        if controller.run_state.status is ConsoleRunStatus.CHECKING_CITATIONS:
            return
        await asyncio.sleep(0)
    raise AssertionError("citation checking was not observable")


def _sanitize_selected_persistence(
    persistence: _ReadyCitationPersistence,
    selected_body: str,
) -> list[dict[str, Any]]:
    return [
        {
            **call,
            "content": (
                "<selected-body>"
                if call["sender"] == ConsoleMessageRole.ASSISTANT.value
                and call["content"] == selected_body
                else call["content"]
            ),
        }
        for call in persistence.create_calls
    ]


def _assert_user_citation_repair_cancel(
    *,
    controller: ConsoleChatController,
    store: _RecordingCitationStore,
    persistence: _ReadyCitationPersistence | None,
    initial_body: str,
) -> None:
    assistant = _assistant(store)
    assert assistant.content == initial_body
    assert assistant.status == "complete"
    assert assistant.citation_presentation == ConsoleCitationPresentation(
        phase=ConsoleCitationPhase.SELECTED,
        notice_code=ConsoleCitationNoticeCode.CANCELED,
        original_attempt_available=False,
    )
    assert controller.original_attempt_for_message(assistant.id) is None
    assert store.completion_calls == [assistant.id]
    assert store.stopped_calls == []
    assert controller.run_state.status is ConsoleRunStatus.STOPPED
    system_messages = [
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]
    assert [message.content for message in system_messages] == [
        "Citation repair canceled by user."
    ]
    append_call = next(
        call
        for call in store.message_append_calls
        if call["role"] is ConsoleMessageRole.SYSTEM
        and call["content"] == "Citation repair canceled by user."
    )
    assert append_call["kwargs"]["persist"] is (persistence is not None)

    if persistence is not None:
        assistant_write = next(
            call
            for call in persistence.create_calls
            if call["sender"] == ConsoleMessageRole.ASSISTANT.value
        )
        system_write = next(
            call
            for call in persistence.create_calls
            if call["sender"] == ConsoleMessageRole.SYSTEM.value
            and call["content"] == "Citation repair canceled by user."
        )
        assert persistence.create_calls.index(
            assistant_write
        ) < persistence.create_calls.index(system_write)
        assert system_write["parent_message_id"] == assistant.persisted_message_id


def test_citation_repair_checking_run_state_is_stoppable_but_send_blocked():
    checking = ConsoleRunState(
        ConsoleRunStatus.CHECKING_CITATIONS,
        "Checking citations…",
    )
    streaming = ConsoleRunState(
        ConsoleRunStatus.STREAMING,
        "Streaming response.",
    )

    assert checking.is_send_allowed is False
    assert checking.is_stop_allowed is True
    assert streaming.is_stop_allowed is True


@pytest.mark.parametrize("agent", (False, True), ids=("direct", "agent"))
@pytest.mark.asyncio
async def test_citation_repair_user_cancellation_privacy_sentinels(
    agent: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    persistence = _ReadyCitationPersistence()
    initial_body = _REPAIR_INITIAL_BODY_SENTINEL
    repaired_body = f"{initial_body} [S1] {_REPAIR_REPAIRED_BODY_SENTINEL}"
    controller, store, gateway, _bridge, _initial, _repaired = (
        _controlled_citation_repair(
            agent=agent,
            persistence=persistence,
            contract=_privacy_repair_contract(),
            initial_body=initial_body,
            repaired_body=repaired_body,
            pause_before_first_chunk=True,
        )
    )
    loguru_records: list[object] = []
    sink_id = loguru_logger.add(
        loguru_records.append,
        level="DEBUG",
        format="{message}",
    )
    caplog.set_level(logging.DEBUG)
    try:
        task = asyncio.create_task(controller.submit_draft("question"))
        await gateway.repair_started.wait()
        retained_session = _active_citation_repair_session(controller)
        assert retained_session is not None
        assert controller.stop_active_run() is True
        await task
    finally:
        loguru_logger.remove(sink_id)

    assistant = _assistant(store)
    assistant_writes = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert len(assistant_writes) == 1
    assert assistant_writes[0]["content"] == initial_body
    assert retained_session.contract is None
    assert retained_session.resolution is None
    _assert_content_free_repair_state(
        caplog.text,
        loguru_records,
        *(call["signals"] for call in gateway.calls),
        assistant.citation_presentation,
        retained_session,
        _sanitize_selected_persistence(persistence, initial_body),
        controller.run_state,
        controller.run_state_history,
        _active_citation_repair_session(controller),
        _active_assistant_message_id(controller),
        _active_stream_task(controller),
    )


@pytest.mark.asyncio
async def test_citation_repair_late_chunk_privacy_sentinels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    persistence = _ReadyCitationPersistence()
    initial_body = _REPAIR_INITIAL_BODY_SENTINEL
    controller, store, gateway, _bridge, _initial, _repaired = (
        _controlled_citation_repair(
            agent=False,
            persistence=persistence,
            contract=_privacy_repair_contract(),
            initial_body=initial_body,
            repaired_body=f"{initial_body} [S1] {_REPAIR_REPAIRED_BODY_SENTINEL}",
            pause_after_first_chunk=True,
            yield_late_chunk_on_cancel=True,
        )
    )
    loguru_records: list[object] = []
    sink_id = loguru_logger.add(
        loguru_records.append,
        level="DEBUG",
        format="{message}",
    )
    caplog.set_level(logging.DEBUG)
    try:
        task = asyncio.create_task(controller.submit_draft("question"))
        await gateway.first_repair_chunk_collected.wait()
        retained_session = _active_citation_repair_session(controller)
        assert retained_session is not None
        assert controller.stop_active_run() is True
        await task
    finally:
        loguru_logger.remove(sink_id)

    assistant = _assistant(store)
    assert assistant.content == initial_body
    assert retained_session.contract is None
    assert retained_session.resolution is None
    _assert_content_free_repair_state(
        caplog.text,
        loguru_records,
        *(call["signals"] for call in gateway.calls),
        assistant.citation_presentation,
        retained_session,
        _sanitize_selected_persistence(persistence, initial_body),
        controller.run_state,
        controller.run_state_history,
        _active_citation_repair_session(controller),
        _active_assistant_message_id(controller),
        _active_stream_task(controller),
    )


@pytest.mark.asyncio
async def test_citation_repair_session_close_privacy_sentinels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    persistence = _ReadyCitationPersistence()
    initial_body = _REPAIR_INITIAL_BODY_SENTINEL
    controller, store, gateway, _bridge, _initial, _repaired = (
        _controlled_citation_repair(
            agent=False,
            persistence=persistence,
            contract=_privacy_repair_contract(),
            initial_body=initial_body,
            repaired_body=f"{initial_body} [S1] {_REPAIR_REPAIRED_BODY_SENTINEL}",
            pause_before_first_chunk=True,
            yield_late_chunk_on_cancel=True,
        )
    )
    loguru_records: list[object] = []
    sink_id = loguru_logger.add(
        loguru_records.append,
        level="DEBUG",
        format="{message}",
    )
    caplog.set_level(logging.DEBUG)
    try:
        task = asyncio.create_task(controller.submit_draft("question"))
        await gateway.repair_started.wait()
        retained_session = _active_citation_repair_session(controller)
        assert retained_session is not None
        session_id = store.active_session_id
        assert session_id is not None
        controller.close_session(session_id)
        result = await task
    finally:
        loguru_logger.remove(sink_id)

    assert result.visible_copy == "Session closed."
    assert retained_session.contract is None
    assert retained_session.resolution is None
    _assert_content_free_repair_state(
        caplog.text,
        loguru_records,
        *(call["signals"] for call in gateway.calls),
        retained_session,
        persistence.create_calls,
        controller.run_state,
        controller.run_state_history,
        _active_citation_repair_session(controller),
        _active_assistant_message_id(controller),
        _active_stream_task(controller),
        store.sessions(),
    )


@pytest.mark.asyncio
async def test_citation_repair_stop_during_initial_generation_keeps_ordinary_stop_behavior():
    controller, store, gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=False,
            pause_initial_after_first_chunk=True,
        )
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await gateway.first_initial_chunk_collected.wait()

    assert controller.stop_active_run() is True
    await task

    assistant = _assistant(store)
    assert assistant.content == initial_body
    assert assistant.status == "stopped"
    assert store.stopped_calls
    assert store.completion_calls == []
    system_messages = [
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]
    assert [message.content for message in system_messages] == [
        "Response stopped by user."
    ]


@pytest.mark.parametrize("agent", (False, True), ids=("direct", "agent"))
@pytest.mark.asyncio
async def test_citation_repair_stop_while_checking_cancels_before_dispatch(agent: bool):
    persistence = _ReadyCitationPersistence() if not agent else None
    controller, store, gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=agent,
            persistence=persistence,
        )
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await _wait_for_citation_checking(controller)

    blocked = await controller.submit_draft("must stay blocked")
    assert blocked.accepted is False
    assert controller.run_state.is_stop_allowed is True
    assert controller.stop_active_run() is True
    result = await task

    assert result.accepted is True
    assert len(gateway.calls) == (0 if agent else 1)
    _assert_user_citation_repair_cancel(
        controller=controller,
        store=store,
        persistence=persistence,
        initial_body=initial_body,
    )


@pytest.mark.parametrize("agent", (False, True), ids=("direct", "agent"))
@pytest.mark.asyncio
async def test_citation_repair_stop_during_collection_cancels_without_stopping_message(
    agent: bool,
):
    controller, store, gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=agent,
            pause_before_first_chunk=True,
        )
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await gateway.repair_started.wait()

    assert controller.stop_active_run() is True
    result = await task

    assert result.accepted is True
    assert len(gateway.calls) == (1 if agent else 2)
    _assert_user_citation_repair_cancel(
        controller=controller,
        store=store,
        persistence=None,
        initial_body=initial_body,
    )


@pytest.mark.parametrize("agent", (False, True), ids=("direct", "agent"))
@pytest.mark.asyncio
async def test_citation_repair_cancel_row_persistence_failure_is_fail_soft(
    agent: bool,
):
    persistence = _CancelRowFailingPersistence()
    controller, store, gateway, bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=agent,
            persistence=persistence,
            pause_before_first_chunk=True,
        )
    )
    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="WARNING",
        format="{message}",
    )
    try:
        task = asyncio.create_task(controller.submit_draft("question"))
        await gateway.repair_started.wait()

        assert controller.stop_active_run() is True
        result = await task
    finally:
        loguru_logger.remove(sink_id)

    assistant = _assistant(store)
    assert result.accepted is True
    assert result.visible_copy == assistant.content == initial_body
    assert assistant.status == "complete"
    assert assistant.persisted_message_id is not None
    assert assistant.citation_presentation == ConsoleCitationPresentation(
        phase=ConsoleCitationPhase.SELECTED,
        notice_code=ConsoleCitationNoticeCode.CANCELED,
        original_attempt_available=False,
    )
    assert store.completion_calls == [assistant.id]
    assert store.stopped_calls == []
    assert store.failed_calls == []
    assert controller.run_state.status is ConsoleRunStatus.STOPPED

    system_contents = [
        message.content
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]
    assert "Response stopped by user." not in system_contents
    assert all(
        not content.startswith("Provider stream failed:") for content in system_contents
    )
    assert system_contents.count("Citation repair canceled by user.") <= 1
    cancel_appends = [
        call
        for call in store.message_append_calls
        if call["role"] is ConsoleMessageRole.SYSTEM
        and call["content"] == "Citation repair canceled by user."
    ]
    assert len(cancel_appends) == 1
    assert cancel_appends[0]["kwargs"]["persist"] is True
    assert persistence.cancel_row_attempts == 1
    assistant_writes = [
        call
        for call in persistence.create_calls
        if call["sender"] == ConsoleMessageRole.ASSISTANT.value
    ]
    assert len(assistant_writes) == 1
    assert assistant_writes[0]["content"] == initial_body
    cancel_attempt = next(
        call
        for call in persistence.create_attempts
        if call["sender"] == ConsoleMessageRole.SYSTEM.value
    )
    assert persistence.create_attempts.index(
        next(
            call
            for call in persistence.create_attempts
            if call["sender"] == ConsoleMessageRole.ASSISTANT.value
        )
    ) < persistence.create_attempts.index(cancel_attempt)
    assert cancel_attempt["parent_message_id"] == assistant.persisted_message_id

    if agent:
        assert bridge is not None
        assert bridge.anchors == [
            ("run-console-terminal", assistant.persisted_message_id)
        ]
    else:
        assert bridge is None

    logs = "\n".join(str(record) for record in captured_logs)
    assert _CANCEL_ROW_PERSISTENCE_FAILURE_CODE in logs
    assert _CANCEL_ROW_PERSISTENCE_EXCEPTION not in logs


@pytest.mark.asyncio
async def test_citation_repair_close_unrelated_session_preserves_cancel_ownership():
    controller, store, gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=False,
            pause_before_first_chunk=True,
        )
    )
    owner_session_id = store.active_session_id
    unrelated = controller.new_session(title="Unrelated")
    controller.switch_session(owner_session_id)
    task = asyncio.create_task(controller.submit_draft("question"))
    await gateway.repair_started.wait()
    repair_session = _active_citation_repair_session(controller)

    controller.close_session(unrelated.id)

    assert _active_citation_repair_session(controller) is repair_session
    assert controller.stop_active_run() is True
    await task
    _assert_user_citation_repair_cancel(
        controller=controller,
        store=store,
        persistence=None,
        initial_body=initial_body,
    )


@pytest.mark.asyncio
async def test_citation_repair_cancel_consumes_used_one_shot_prefill():
    controller, store, gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=False,
            pause_before_first_chunk=True,
        )
    )
    session_id = store.active_session_id
    prefill = "ONE-SHOT PREFIX: "
    store.set_session_one_shot_prefill(session_id, prefill)
    task = asyncio.create_task(controller.submit_draft("question"))
    await gateway.repair_started.wait()

    assert controller.stop_active_run() is True
    await task

    _assert_user_citation_repair_cancel(
        controller=controller,
        store=store,
        persistence=None,
        initial_body=f"{prefill}{initial_body}",
    )
    assert store.session_one_shot_prefill(session_id) is None


@pytest.mark.asyncio
async def test_citation_repair_cancel_after_chunk_discards_late_output():
    persistence = _ReadyCitationPersistence()
    controller, store, gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=False,
            persistence=persistence,
            pause_after_first_chunk=True,
            yield_late_chunk_on_cancel=True,
        )
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await gateway.first_repair_chunk_collected.wait()

    assert controller.stop_active_run() is True
    await task

    _assert_user_citation_repair_cancel(
        controller=controller,
        store=store,
        persistence=persistence,
        initial_body=initial_body,
    )
    assert "LATE REPAIR MUST NOT WIN" not in _assistant(store).content


@pytest.mark.asyncio
async def test_citation_repair_stop_immediately_before_selection_commit_wins(
    monkeypatch,
):
    controller, store, _gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(agent=False)
    )
    stop_results: list[bool] = []
    real_select = controller_module.select_repaired_body

    def stop_before_commit(*args, **kwargs):
        selected = real_select(*args, **kwargs)
        stop_results.append(controller.stop_active_run())
        return selected

    monkeypatch.setattr(
        controller_module,
        "select_repaired_body",
        stop_before_commit,
    )

    await controller.submit_draft("question")

    assert stop_results == [True]
    _assert_user_citation_repair_cancel(
        controller=controller,
        store=store,
        persistence=None,
        initial_body=initial_body,
    )


@pytest.mark.asyncio
async def test_citation_repair_stop_after_selection_commit_is_noop():
    controller, store, _gateway, _bridge, _initial_body, repaired_body = (
        _controlled_citation_repair(agent=False)
    )
    stop_results: list[bool] = []
    real_set_presentation = store.set_citation_presentation

    def stop_after_commit(message_id, presentation):
        updated = real_set_presentation(message_id, presentation)
        if (
            presentation is not None
            and presentation.phase is ConsoleCitationPhase.SELECTED
            and presentation.notice_code is ConsoleCitationNoticeCode.REPAIRED
        ):
            stop_results.append(controller.stop_active_run())
        return updated

    store.set_citation_presentation = stop_after_commit
    result = await controller.submit_draft("question")

    assistant = _assistant(store)
    assert stop_results == [False]
    assert result.visible_copy == assistant.content == repaired_body
    assert assistant.status == "complete"
    assert store.stopped_calls == []
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_citation_repair_shutdown_during_collection_privacy_has_no_user_stop_row():
    controller, store, gateway, _bridge, initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=False,
            contract=_privacy_repair_contract(),
            pause_before_first_chunk=True,
        )
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await gateway.repair_started.wait()
    retained_session = _active_citation_repair_session(controller)
    assert retained_session is not None

    await controller.shutdown()
    await task

    assistant = _assistant(store)
    assert assistant.content == initial_body
    assert assistant.status == "complete"
    assert store.stopped_calls == []
    assert all(
        message.role is not ConsoleMessageRole.SYSTEM
        for message in store.messages_for_session(store.active_session_id)
    )
    assert retained_session.contract is None
    assert retained_session.resolution is None


@pytest.mark.asyncio
async def test_citation_repair_close_during_collection_never_resurrects_session_or_message():
    persistence = _ReadyCitationPersistence()
    controller, store, gateway, _bridge, _initial_body, _repaired_body = (
        _controlled_citation_repair(
            agent=False,
            persistence=persistence,
            pause_before_first_chunk=True,
            yield_late_chunk_on_cancel=True,
        )
    )
    task = asyncio.create_task(controller.submit_draft("question"))
    await gateway.repair_started.wait()
    session_id = store.active_session_id

    controller.close_session(session_id)
    result = await task

    assert result.visible_copy == "Session closed."
    assert store.sessions() == []
    assert store.active_session_id is None
    assert _active_citation_repair_session(controller) is None
    assert _active_assistant_message_id(controller) is None
    assert _active_stream_task(controller) is None
    assert not any(
        call["sender"] == ConsoleMessageRole.SYSTEM.value
        for call in persistence.create_calls
    )


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
