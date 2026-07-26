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
    media_content = {"evidence": "WORKER_EVIDENCE_SNAPSHOT_SENTINEL_TASK_553_13"}
    selected_parts = ["evidence"]
    answer = "STREAMING_ANSWER_SENTINEL_TASK_553_13"
    reasoning = "STREAMING_REASONING_SENTINEL_TASK_553_13"
    observed = {}
    builder_alive_during_stream = []

    def fake_core_chat_function(**kwargs):
        observed.update(kwargs)
        assert "citation_trace_builder" not in kwargs
        assert kwargs["media_content"] is media_content
        assert kwargs["selected_parts"] is selected_parts

        def stream():
            builder_alive_during_stream.append(builder_ref() is not None)
            assert builder_alive_during_stream[-1]
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
            builder_alive_during_stream.append(builder_ref() is not None)
            assert builder_alive_during_stream[-1]
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
    builder_box = [kwargs.pop("citation_trace_builder")]
    assert "citation_trace_builder" not in kwargs
    builder_ref = weakref.ref(builder_box[0])
    del builder

    try:
        result = worker_events.chat_wrapper_function(
            app,
            citation_trace_builder=builder_box.pop(),
            **kwargs,
        )
    finally:
        loguru_logger.remove(sink_id)

    assert builder_alive_during_stream == [True, True]
    assert result == "STREAMING_HANDLED_BY_EVENTS"
    assert builder_box == []
    assert "citation_trace_builder" not in observed
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


def test_streaming_first_chunk_structure_is_logged_once_per_request(monkeypatch):
    answers = [
        "FIRST_REQUEST_ANSWER_SENTINEL_TASK_553_13",
        "SECOND_REQUEST_ANSWER_SENTINEL_TASK_553_13",
    ]

    def fake_core_chat_function(**_kwargs):
        answer = answers.pop(0)

        def stream():
            for suffix in ("one", "two"):
                yield "data: " + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": f"{answer}-{suffix}"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": f"{answer}-{suffix}",
                                            "logprob": -0.1,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                )
            yield "data: [DONE]"

        return stream()

    monkeypatch.setattr(worker_events, "core_chat_function", fake_core_chat_function)
    diagnostic = (
        "First streaming chunk received with logprobs enabled; "
        "provider=openai; choice_count=1"
    )
    per_request_logs = []

    for answer in tuple(answers):
        captured_logs = []
        sink_id = loguru_logger.add(
            captured_logs.append,
            level="DEBUG",
            format="{message}",
        )
        try:
            result = worker_events.chat_wrapper_function(
                _app(),
                **_request_kwargs(_RequestBuilder(), streaming=True),
            )
        finally:
            loguru_logger.remove(sink_id)

        assert result == "STREAMING_HANDLED_BY_EVENTS"
        rendered_logs = "".join(str(message) for message in captured_logs)
        assert answer not in rendered_logs
        per_request_logs.append(rendered_logs)

    assert not hasattr(worker_events.chat_wrapper_function, "_logged_structure")
    assert [logs.count(diagnostic) for logs in per_request_logs] == [1, 1]


def test_non_streaming_worker_keeps_builder_local_without_changing_rag_seam(
    monkeypatch,
):
    app = _app()
    builder = _RequestBuilder()
    builder_ref = weakref.ref(builder)
    media_content = {"evidence": "NONSTREAM_EVIDENCE_SENTINEL_TASK_553_13"}
    selected_parts = ["evidence"]
    answer = "NONSTREAM_ANSWER_SENTINEL_TASK_553_13"
    observed = {}
    builder_alive_in_core = []

    def fake_core_chat_function(**kwargs):
        builder_alive_in_core.append(builder_ref() is not None)
        assert builder_alive_in_core[-1]
        observed.update(kwargs)
        assert "citation_trace_builder" not in kwargs
        assert kwargs["media_content"] is media_content
        assert kwargs["selected_parts"] is selected_parts
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
    builder_box = [kwargs.pop("citation_trace_builder")]
    assert "citation_trace_builder" not in kwargs
    del builder

    try:
        result = worker_events.chat_wrapper_function(
            app,
            citation_trace_builder=builder_box.pop(),
            **kwargs,
        )
    finally:
        loguru_logger.remove(sink_id)

    assert builder_alive_in_core == [True]
    assert result == answer
    assert builder_box == []
    assert "citation_trace_builder" not in observed
    assert observed["media_content"] is media_content
    assert observed["selected_parts"] is selected_parts
    gc.collect()
    assert builder_ref() is None
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
