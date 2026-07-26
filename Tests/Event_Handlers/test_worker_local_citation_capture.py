from __future__ import annotations

import gc
import json
import weakref
from unittest.mock import MagicMock

from loguru import logger as loguru_logger

import tldw_chatbook.Event_Handlers.worker_events as worker_events


class _RequestBuilder:
    pass


def _app() -> MagicMock:
    app = MagicMock()
    app.loguru_logger = loguru_logger
    app.current_chat_worker = None
    app.app_config = {"chat_defaults": {"strip_thinking_tags": True}}
    app.post_message = MagicMock()
    return app


def _request_kwargs(builder: object, *, streaming: bool) -> dict:
    return {
        "message": "WORKER_QUERY_SENTINEL_TASK_553_13",
        "history": [],
        "media_content": {"evidence": "WORKER_EVIDENCE_SNAPSHOT_SENTINEL_TASK_553_13"},
        "selected_parts": ["evidence"],
        "citation_trace_builder": builder,
        "api_endpoint": "openai",
        "api_key": "test-key",
        "custom_prompt": "WORKER_CUSTOM_PROMPT_SENTINEL_TASK_553_13",
        "temperature": 0.7,
        "streaming": streaming,
        "model": "test-model",
        "llm_logprobs": True,
        "llm_top_logprobs": 2,
    }


def test_streaming_worker_keeps_builder_local_and_logs_no_request_or_answer_content(
    monkeypatch,
):
    app = _app()
    builder = _RequestBuilder()
    builder_ref = weakref.ref(builder)
    media_content = {"evidence": "WORKER_EVIDENCE_SNAPSHOT_SENTINEL_TASK_553_13"}
    selected_parts = ["evidence"]
    answer = "STREAMING_ANSWER_SENTINEL_TASK_553_13"
    reasoning = "STREAMING_REASONING_SENTINEL_TASK_553_13"
    observed = {}

    def fake_core_chat_function(**kwargs):
        observed.update(kwargs)
        assert "citation_trace_builder" not in kwargs
        assert kwargs["media_content"] is media_content
        assert kwargs["selected_parts"] is selected_parts

        def stream():
            assert builder_ref() is not None
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": answer,
                                "reasoning_content": reasoning,
                            },
                            "logprobs": {
                                "content": [{"token": answer, "logprob": -0.1}]
                            },
                        }
                    ]
                }
            )
            assert builder_ref() is not None
            yield "data: [DONE]"

        return stream()

    monkeypatch.setattr(worker_events, "core_chat_function", fake_core_chat_function)
    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="DEBUG",
        format="{message}",
    )
    kwargs = _request_kwargs(builder, streaming=True)
    kwargs["media_content"] = media_content
    kwargs["selected_parts"] = selected_parts
    del builder

    try:
        result = worker_events.chat_wrapper_function(app, **kwargs)
    finally:
        loguru_logger.remove(sink_id)

    assert result == "STREAMING_HANDLED_BY_EVENTS"
    assert "citation_trace_builder" not in observed
    del kwargs
    gc.collect()
    assert builder_ref() is None
    rendered_logs = "".join(str(message) for message in captured_logs)
    for sentinel in (
        "WORKER_QUERY_SENTINEL_TASK_553_13",
        "WORKER_EVIDENCE_SNAPSHOT_SENTINEL_TASK_553_13",
        "WORKER_CUSTOM_PROMPT_SENTINEL_TASK_553_13",
        answer,
        reasoning,
    ):
        assert sentinel not in rendered_logs


def test_non_streaming_worker_removes_builder_without_changing_rag_seam(
    monkeypatch,
):
    app = _app()
    builder = _RequestBuilder()
    media_content = {"evidence": "NONSTREAM_EVIDENCE_SENTINEL_TASK_553_13"}
    selected_parts = ["evidence"]
    answer = "NONSTREAM_ANSWER_SENTINEL_TASK_553_13"
    observed = {}

    def fake_core_chat_function(**kwargs):
        observed.update(kwargs)
        return answer

    monkeypatch.setattr(worker_events, "core_chat_function", fake_core_chat_function)
    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="DEBUG",
        format="{message}",
    )
    kwargs = _request_kwargs(builder, streaming=False)
    kwargs["media_content"] = media_content
    kwargs["selected_parts"] = selected_parts

    try:
        result = worker_events.chat_wrapper_function(app, **kwargs)
    finally:
        loguru_logger.remove(sink_id)

    assert result == answer
    assert "citation_trace_builder" not in observed
    assert observed["media_content"] is media_content
    assert observed["selected_parts"] is selected_parts
    rendered_logs = "".join(str(message) for message in captured_logs)
    for sentinel in (
        "WORKER_QUERY_SENTINEL_TASK_553_13",
        "NONSTREAM_EVIDENCE_SENTINEL_TASK_553_13",
        "WORKER_CUSTOM_PROMPT_SENTINEL_TASK_553_13",
        answer,
    ):
        assert sentinel not in rendered_logs


def test_worker_exception_log_uses_structural_reason_only(monkeypatch):
    app = _app()
    failure_sentinel = "WORKER_VALIDATION_FAILURE_SENTINEL_TASK_553_13"

    def fake_core_chat_function(**_kwargs):
        raise ValueError(failure_sentinel)

    monkeypatch.setattr(worker_events, "core_chat_function", fake_core_chat_function)
    captured_logs = []
    sink_id = loguru_logger.add(
        captured_logs.append,
        level="DEBUG",
        format="{message}",
    )

    try:
        result = worker_events.chat_wrapper_function(
            app,
            **_request_kwargs(_RequestBuilder(), streaming=False),
        )
    finally:
        loguru_logger.remove(sink_id)

    assert result == "STREAMING_HANDLED_BY_EVENTS"
    assert failure_sentinel not in "".join(str(message) for message in captured_logs)
