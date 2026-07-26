from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest
from loguru import logger as loguru_logger
from pydantic import ValidationError

import tldw_chatbook.Chat.Chat_Functions as chat_functions
from tldw_chatbook.Chat.Chat_Functions import ChatDictionary
from tldw_chatbook.Chat.citation_source_locators import CanonicalSourceKind
from tldw_chatbook.Chat.citation_trace_builder import (
    CitationTraceBuilder,
    LocalRetrievalCandidateCapture,
    LocalRetrievalRunMetadata,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    LocalCitationIdentityContext,
)
from tldw_chatbook.Chat.citation_trace_models import (
    RetrievalScoreKind,
    RetrievalScoreScale,
)
from tldw_chatbook.RAG_Search.local_citation_capture import (
    format_local_evidence_context,
    normalize_local_result,
)


NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
SECRET = b"0123456789abcdef0123456789abcdef"


def _canonical_capture():
    title_one = "EVIDENCE_TITLE_ONE_SENTINEL_TASK_553_13"
    title_two = "EVIDENCE_TITLE_TWO_SENTINEL_TASK_553_13"
    body_one = "EVIDENCE_BODY_ONE_SENTINEL_TASK_553_13"
    body_two = "  EVIDENCE_BODY_TWO_SENTINEL_TASK_553_13  \n\t"
    raw_results = (
        {
            "id": "media-1",
            "source": "media",
            "title": title_one,
            "content": body_one,
            "score": 0.9,
            "metadata": {"source_type": "media", "source_id": "media-1"},
        },
        {
            "id": "media-2",
            "source": "media",
            "title": title_two,
            "content": body_two,
            "score": 0.8,
            "metadata": {"source_type": "media", "source_id": "media-2"},
        },
    )
    normalized = tuple(
        normalize_local_result(result, candidate_rank=rank)
        for rank, result in enumerate(raw_results, start=1)
    )
    formatted = format_local_evidence_context(normalized, max_length=10_000)
    builder = CitationTraceBuilder.local(
        request_id="request-boundary",
        generation_id="generation-boundary",
        identity_context=LocalCitationIdentityContext(
            profile_id="profile-1",
            local_authority_id="authority-1",
            fingerprint_key_id="fingerprint-key-1",
        ),
        fingerprint_codec=CitationFingerprintCodec(SECRET),
        created_at=NOW,
    )
    candidates = tuple(
        LocalRetrievalCandidateCapture(
            candidate_rank=result.candidate_rank,
            source_kind=CanonicalSourceKind.MEDIA_DB,
            source_id=result.source_id,
            title=result.title,
            score_kind=RetrievalScoreKind.VECTOR_SIMILARITY,
            score_scale=RetrievalScoreScale.ZERO_TO_ONE,
            score=result.score,
        )
        for result in normalized
    )
    run_id = builder.record_retrieval_run(
        stage="plain",
        raw_query="BOUNDARY_QUERY_SENTINEL_TASK_553_13",
        candidates=candidates,
        retrieval_metadata=LocalRetrievalRunMetadata(
            search_mode="plain",
            requested_top_k=2,
            max_context_characters=10_000,
            rerank_enabled=False,
            source_kinds=(CanonicalSourceKind.MEDIA_DB,),
            scope_state="unscoped",
        ),
        started_at=NOW,
        ended_at=NOW,
    )
    builder.record_prompt_evidence_set(
        run_id=run_id,
        evidence=formatted.entries,
        created_at=NOW,
    )
    return formatted.context, builder


def _provider_user_text(messages_payload):
    current_user = messages_payload[-1]
    content = current_user["content"]
    if isinstance(content, str):
        return content
    return next(part["text"] for part in content if part.get("type") == "text")


@pytest.mark.parametrize("api_endpoint", ["openai", "deepseek"])
def test_chat_dictionary_never_mutates_canonical_evidence_at_provider_boundary(
    api_endpoint,
    monkeypatch,
    caplog,
):
    ordinary_prompt = "ORDINARY_PROMPT_SENTINEL_TASK_553_13"
    transformed_prompt = "ORDINARY_PROMPT_TRANSFORMED_TASK_553_13"
    custom_prompt = "CUSTOM_PROMPT_SENTINEL_TASK_553_13"
    answer = "NONSTREAM_ANSWER_SENTINEL_TASK_553_13"
    canonical_context, builder = _canonical_capture()
    captured_call = {}

    def fake_chat_api_call(**kwargs):
        captured_call.update(kwargs)
        return answer

    monkeypatch.setattr(chat_functions, "chat_api_call", fake_chat_api_call)
    monkeypatch.setattr(chat_functions, "load_settings", lambda: {})
    chatdict_entries = [
        ChatDictionary(key=ordinary_prompt, content=transformed_prompt),
        ChatDictionary(
            key="EVIDENCE_TITLE_ONE_SENTINEL_TASK_553_13",
            content="MUTATED_EVIDENCE_TITLE_TASK_553_13",
        ),
        ChatDictionary(
            key="EVIDENCE_BODY_TWO_SENTINEL_TASK_553_13",
            content="MUTATED_EVIDENCE_BODY_TASK_553_13",
        ),
    ]
    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="DEBUG",
        format="{message}",
    )

    try:
        with caplog.at_level(logging.DEBUG):
            response = chat_functions.chat(
                message=ordinary_prompt,
                history=[
                    {
                        "role": "user",
                        "content": "HISTORY_TEXT_SENTINEL_TASK_553_13",
                    }
                ],
                media_content={"evidence": canonical_context},
                selected_parts=["evidence"],
                api_endpoint=api_endpoint,
                api_key="test-key",
                custom_prompt=custom_prompt,
                temperature=0.7,
                streaming=False,
                model="test-model",
                chatdict_entries=chatdict_entries,
            )
    finally:
        loguru_logger.remove(sink_id)

    assert response == answer
    provider_user_text = _provider_user_text(captured_call["messages_payload"])
    assert transformed_prompt in provider_user_text
    assert ordinary_prompt not in provider_user_text
    assert custom_prompt in provider_user_text
    assert "MUTATED_EVIDENCE_TITLE_TASK_553_13" not in provider_user_text
    assert "MUTATED_EVIDENCE_BODY_TASK_553_13" not in provider_user_text
    for payload in builder.evidence_snapshot_payloads:
        assert payload.snapshot_text is not None
        assert payload.snapshot_text in provider_user_text
    whitespace_snapshot = builder.evidence_snapshot_payloads[-1].snapshot_text
    assert whitespace_snapshot is not None
    assert whitespace_snapshot.endswith("  \n\t")
    assert whitespace_snapshot.encode("utf-8") in provider_user_text.encode("utf-8")

    rendered_logs = "".join(
        record.getMessage()
        for record in caplog.records
        if record.pathname.endswith("Chat_Functions.py")
    ) + "".join(str(message) for message in captured_logs)
    for sentinel in (
        ordinary_prompt,
        "HISTORY_TEXT_SENTINEL_TASK_553_13",
        custom_prompt,
        "EVIDENCE_TITLE_ONE_SENTINEL_TASK_553_13",
        "EVIDENCE_TITLE_TWO_SENTINEL_TASK_553_13",
        "EVIDENCE_BODY_ONE_SENTINEL_TASK_553_13",
        "EVIDENCE_BODY_TWO_SENTINEL_TASK_553_13",
        answer,
    ):
        assert sentinel not in rendered_logs


def test_chat_validation_failure_log_does_not_render_sensitive_input(
    monkeypatch,
    caplog,
):
    validation_sentinel = "CHAT_VALIDATION_FAILURE_SENTINEL_TASK_553_13"
    validation_error = ValidationError.from_exception_data(
        "SensitiveInput",
        [
            {
                "type": "value_error",
                "loc": ("message",),
                "input": validation_sentinel,
                "ctx": {"error": ValueError("invalid")},
            }
        ],
    )
    monkeypatch.setattr(
        chat_functions,
        "process_user_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(validation_error),
    )
    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="DEBUG",
        format="{message}",
    )

    try:
        with caplog.at_level(logging.DEBUG):
            chat_functions.chat(
                message="trigger",
                history=[],
                media_content={},
                selected_parts=[],
                api_endpoint="openai",
                api_key="test-key",
                custom_prompt=None,
                temperature=0.7,
                chatdict_entries=[ChatDictionary(key="trigger", content="replacement")],
            )
    finally:
        loguru_logger.remove(sink_id)

    rendered_logs = "".join(
        record.getMessage()
        for record in caplog.records
        if record.pathname.endswith("Chat_Functions.py")
    ) + "".join(str(message) for message in captured_logs)
    assert validation_sentinel not in rendered_logs
