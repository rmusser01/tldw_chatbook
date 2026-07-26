from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class _RequestBuilder:
    pass


class _RecordingGateway:
    def __init__(self, builder_ref=None):
        self.builder_ref = builder_ref
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
        yield "answer"


def _persisted_store() -> ConsoleChatStore:
    store = ConsoleChatStore()
    session = store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    return store


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
    builder_holder = [_RequestBuilder()]
    builder_ref = weakref.ref(builder_holder[0])

    async def capture(_draft):
        return SimpleNamespace(
            context=canonical_context,
            citation_builder=builder_holder.pop(),
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
    builder_holder = [_RequestBuilder()]
    builder_ref = weakref.ref(builder_holder[0])

    async def capture(_draft):
        return SimpleNamespace(
            context=context,
            citation_builder=builder_holder.pop(),
        )

    bridge_calls = []

    def run_reply(**kwargs):
        assert builder_ref() is not None
        bridge_calls.append(kwargs)
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
