"""Streaming contracts for the QwenCloud dual-API adapter."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
import tldw_chatbook.LLM_Calls.qwencloud as qwencloud
from tldw_chatbook.LLM_Calls.qwencloud_streaming import (
    QwenCloudStream,
    QwenResponsesStreamTranslator,
    iter_sse_data_records,
)


class _ByteStreamResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}
        self.close_calls = 0
        self.iter_content_calls = 0

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)  # type: ignore[arg-type]

    def close(self) -> None:
        self.close_calls += 1

    def iter_content(self, chunk_size: int) -> Any:
        assert chunk_size > 0
        self.iter_content_calls += 1
        yield from self.chunks


class _ByteStreamSession:
    def __init__(self, responses: list[_ByteStreamResponse]) -> None:
        self.responses = responses
        self.posts: list[dict[str, Any]] = []
        self.mounts: list[tuple[str, object]] = []
        self.close_calls = 0

    def mount(self, prefix: str, adapter: object) -> None:
        self.mounts.append((prefix, adapter))

    def post(self, url: str, **kwargs: Any) -> _ByteStreamResponse:
        self.posts.append({"url": url, **deepcopy(kwargs)})
        return self.responses[len(self.posts) - 1]

    def close(self) -> None:
        self.close_calls += 1


def _message_item(
    item_id: str, text: str, *, status: str = "completed"
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_item(
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    *,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _function_added(
    sequence: int,
    output_index: int,
    item_id: str,
    call_id: str,
    name: str,
) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "sequence_number": sequence,
        "output_index": output_index,
        "item": _function_item(item_id, call_id, name, "", status="in_progress"),
    }


def _arguments_delta(
    sequence: int,
    output_index: int,
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.delta",
        "sequence_number": sequence,
        "output_index": output_index,
        "item_id": item_id,
        "delta": delta,
    }


def _arguments_done(
    sequence: int,
    output_index: int,
    item_id: str,
    arguments: str,
) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.done",
        "sequence_number": sequence,
        "output_index": output_index,
        "item_id": item_id,
        "arguments": arguments,
    }


def _message_added(sequence: int, output_index: int, item_id: str) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "sequence_number": sequence,
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
    }


def _content_added(
    sequence: int,
    output_index: int,
    item_id: str,
    *,
    content_index: int = 0,
) -> dict[str, Any]:
    return {
        "type": "response.content_part.added",
        "sequence_number": sequence,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "part": {"type": "output_text", "text": "", "annotations": []},
    }


def _text_delta(
    sequence: int,
    output_index: int,
    item_id: str,
    delta: str,
    *,
    content_index: int = 0,
) -> dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "delta": delta,
        "logprobs": [],
    }


def _text_done(
    sequence: int,
    output_index: int,
    item_id: str,
    text: str,
    *,
    content_index: int = 0,
) -> dict[str, Any]:
    return {
        "type": "response.output_text.done",
        "sequence_number": sequence,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "text": text,
        "logprobs": [],
    }


def _content_done(
    sequence: int,
    output_index: int,
    item_id: str,
    text: str,
    *,
    content_index: int = 0,
) -> dict[str, Any]:
    return {
        "type": "response.content_part.done",
        "sequence_number": sequence,
        "output_index": output_index,
        "item_id": item_id,
        "content_index": content_index,
        "part": {"type": "output_text", "text": text, "annotations": []},
    }


def _output_done(
    sequence: int, output_index: int, item: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "sequence_number": sequence,
        "output_index": output_index,
        "item": deepcopy(item),
    }


def _terminal(
    sequence: int,
    output: list[dict[str, Any]],
    *,
    event_type: str = "response.completed",
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": "resp_verbatim",
        "object": "response",
        "status": status,
        "output": deepcopy(output),
    }
    if usage is not None:
        response["usage"] = deepcopy(usage)
    if incomplete_reason is not None:
        response["incomplete_details"] = {"reason": incomplete_reason}
    return {
        "type": event_type,
        "sequence_number": sequence,
        "response": response,
    }


def test_sse_records_survive_adversarial_byte_boundaries() -> None:
    wire = (
        b'data: {"type":"response.output_text.delta","delta":"caf\xc3\xa9"}\r\n'
        b"\r\n"
        b'data: {"type":"response.completed"}\n\n'
    )
    split_points = sorted(
        {
            1,
            wire.index(b"data:") + 3,
            wire.index(b"\xc3") + 1,
            wire.index(b"\r\n") + 1,
            wire.index(b"\r\n\r\n") + 3,
            len(wire) - 1,
        }
    )
    chunks: list[bytes] = []
    start = 0
    for stop in split_points:
        chunks.append(wire[start:stop])
        start = stop
    chunks.append(wire[start:])

    assert list(iter_sse_data_records(chunks)) == [
        '{"type":"response.output_text.delta","delta":"café"}',
        '{"type":"response.completed"}',
    ]


def test_sse_comments_and_multiline_data_frame_without_decoding() -> None:
    chunks = [
        b": heartbeat\r",
        b"\nevent: response.output_text.delta\r\nid: 17\r\n",
        b'data: {"type":"response.output_text.delta",\r\n',
        b'data: "delta":"safe"}\r\nretry: 1000\r\n\r',
        b"\n: keepalive\ndata: [DONE]\n\n",
    ]

    assert list(iter_sse_data_records(chunks)) == [
        '{"type":"response.output_text.delta",\n"delta":"safe"}',
        "[DONE]",
    ]


def test_responses_text_delta_done_recovery_is_exactly_once() -> None:
    translator = QwenResponsesStreamTranslator()
    chunks: list[dict[str, Any]] = []

    chunks.extend(translator.feed(_message_added(0, 0, "msg_primary")))
    chunks.extend(translator.feed(_content_added(1, 0, "msg_primary")))
    chunks.extend(translator.feed(_text_delta(2, 0, "msg_primary", "Hel")))
    chunks.extend(translator.feed(_text_delta(3, 0, "msg_primary", "lo")))
    chunks.extend(translator.feed(_text_done(4, 0, "msg_primary", "Hello")))
    chunks.extend(translator.feed(_content_done(5, 0, "msg_primary", "Hello")))
    chunks.extend(
        translator.feed(_output_done(6, 0, _message_item("msg_primary", "Hello")))
    )

    chunks.extend(translator.feed(_message_added(7, 1, "msg_recovered")))
    chunks.extend(translator.feed(_content_added(8, 1, "msg_recovered")))
    chunks.extend(translator.feed(_text_done(9, 1, "msg_recovered", " recovered")))
    chunks.extend(translator.feed(_content_done(10, 1, "msg_recovered", " recovered")))
    chunks.extend(
        translator.feed(
            _output_done(11, 1, _message_item("msg_recovered", " recovered"))
        )
    )
    chunks.extend(
        translator.feed(
            _terminal(
                12,
                [
                    _message_item("msg_primary", "Hello"),
                    _message_item("msg_recovered", " recovered"),
                ],
                usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            )
        )
    )

    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == [
        "Hel",
        "lo",
        " recovered",
        "",
    ]
    assert translator.finish() == ()


def test_responses_sequence_duplicate_conflict_and_decrease() -> None:
    translator = QwenResponsesStreamTranslator()
    created = {
        "type": "response.created",
        "sequence_number": 10,
        "response": {"id": "resp_safe", "status": "in_progress", "output": []},
    }
    in_progress = {
        "type": "response.in_progress",
        "sequence_number": 11,
        "response": {"id": "resp_safe", "status": "in_progress", "output": []},
    }
    assert translator.feed(created) == ()
    assert translator.feed(in_progress) == ()
    assert translator.feed(deepcopy(created)) == ()

    conflicting_duplicate = deepcopy(created)
    conflicting_duplicate["type"] = "response.in_progress"
    with pytest.raises(ChatProviderError, match="sequence") as conflict_info:
        translator.feed(conflicting_duplicate)
    assert conflict_info.value.provider == "qwencloud"

    decreasing = QwenResponsesStreamTranslator()
    assert decreasing.feed(created) == ()
    with pytest.raises(ChatProviderError, match="sequence") as decrease_info:
        decreasing.feed(
            {
                "type": "response.in_progress",
                "sequence_number": 9,
                "response": {
                    "id": "resp_safe",
                    "status": "in_progress",
                    "output": [],
                },
            }
        )
    assert decrease_info.value.provider == "qwencloud"

    with pytest.raises(ChatProviderError, match="sequence"):
        QwenResponsesStreamTranslator().feed({"type": "response.created"})


def test_responses_terminal_usage_finish_and_empty_delta() -> None:
    usage = {
        "input_tokens": 9,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 12,
    }
    translator = QwenResponsesStreamTranslator()
    assert translator.feed(_message_added(0, 0, "msg_terminal")) == ()
    assert translator.feed(_content_added(1, 0, "msg_terminal")) == ()
    empty_delta = translator.feed(_text_delta(2, 0, "msg_terminal", ""))
    text_delta = translator.feed(_text_delta(3, 0, "msg_terminal", "safe"))
    terminal = translator.feed(
        _terminal(4, [_message_item("msg_terminal", "safe")], usage=usage)
    )

    assert empty_delta == ({"choices": [{"delta": {"content": ""}}]},)
    assert text_delta == ({"choices": [{"delta": {"content": "safe"}}]},)
    assert terminal == (
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
            "usage": usage,
        },
    )
    with pytest.raises(ChatProviderError, match="terminal"):
        translator.feed(
            {
                "type": "response.in_progress",
                "sequence_number": 5,
                "response": {"status": "in_progress", "output": []},
            }
        )

    incomplete = QwenResponsesStreamTranslator()
    incomplete.feed(_message_added(0, 0, "msg_partial"))
    incomplete.feed(_content_added(1, 0, "msg_partial"))
    incomplete.feed(_text_delta(2, 0, "msg_partial", "partial"))
    assert incomplete.feed(
        _terminal(
            3,
            [_message_item("msg_partial", "partial", status="incomplete")],
            event_type="response.incomplete",
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )
    ) == (
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": "length"}],
            "usage": {},
        },
    )

    for event_type, status in (
        ("response.failed", "failed"),
        ("response.cancelled", "cancelled"),
    ):
        with pytest.raises(ChatProviderError) as failure_info:
            QwenResponsesStreamTranslator().feed(
                _terminal(0, [], event_type=event_type, status=status)
            )
        assert failure_info.value.provider == "qwencloud"

    with pytest.raises(ChatProviderError, match="terminal"):
        QwenResponsesStreamTranslator().finish()
    malformed_terminal = _terminal(0, [])
    del malformed_terminal["response"]["status"]
    with pytest.raises(ChatProviderError) as malformed_info:
        QwenResponsesStreamTranslator().feed(malformed_terminal)
    assert malformed_info.value.provider == "qwencloud"


def test_responses_function_call_fragments_recover_without_duplication() -> None:
    translator = QwenResponsesStreamTranslator()
    chunks: list[dict[str, Any]] = []
    chunks.extend(
        translator.feed(_function_added(0, 0, "fc_transport_a", "call_a", "alpha"))
    )
    chunks.extend(
        translator.feed(_function_added(1, 1, "fc_transport_b", "call_b", "beta"))
    )
    chunks.extend(translator.feed(_arguments_delta(2, 1, "fc_transport_b", '{"b":')))
    chunks.extend(translator.feed(_arguments_delta(3, 0, "fc_transport_a", '{"a":1')))
    chunks.extend(translator.feed(_arguments_done(4, 0, "fc_transport_a", '{"a":1}')))
    chunks.extend(
        translator.feed(
            _output_done(
                5,
                1,
                _function_item("fc_transport_b", "call_b", "beta", '{"b":2}'),
            )
        )
    )
    chunks.extend(
        translator.feed(_function_added(6, 2, "fc_transport_c", "call_c", "gamma"))
    )
    terminal_output = [
        _function_item("fc_transport_a", "call_a", "alpha", '{"a":1}'),
        _function_item("fc_transport_b", "call_b", "beta", '{"b":2}'),
        _function_item("fc_transport_c", "call_c", "gamma", '{"c":3}'),
    ]
    chunks.extend(
        translator.feed(
            _terminal(
                7,
                terminal_output,
                usage={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            )
        )
    )

    fragments: dict[int, list[str]] = {0: [], 1: [], 2: []}
    identities: dict[int, tuple[str, str]] = {}
    terminal_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        choice = chunk["choices"][0]
        if choice.get("finish_reason") is not None:
            terminal_chunks.append(chunk)
            continue
        for tool_delta in choice["delta"].get("tool_calls", []):
            index = tool_delta["index"]
            function = tool_delta.get("function", {})
            fragments[index].append(function.get("arguments", ""))
            if "id" in tool_delta:
                identities[index] = (tool_delta["id"], function["name"])

    assert identities == {
        0: ("call_a", "alpha"),
        1: ("call_b", "beta"),
        2: ("call_c", "gamma"),
    }
    assert "fc_transport_a" not in repr(chunks)
    assert "fc_transport_b" not in repr(chunks)
    assert "fc_transport_c" not in repr(chunks)
    assert {index: "".join(parts) for index, parts in fragments.items()} == {
        0: '{"a":1}',
        1: '{"b":2}',
        2: '{"c":3}',
    }
    assert terminal_chunks == [
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": "tool_calls"}],
            "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        }
    ]
    assert translator.finish() == ()


def test_responses_partial_or_mismatched_call_never_surfaces() -> None:
    invalid_json = QwenResponsesStreamTranslator()
    invalid_json.feed(
        _function_added(0, 0, "fc_private", "call_private", "private_tool")
    )
    invalid_json.feed(
        _arguments_delta(1, 0, "fc_private", '{"RAW-CALL-PRIVATE-CANARY":')
    )
    with pytest.raises(ChatProviderError) as invalid_info:
        invalid_json.feed(
            _arguments_done(2, 0, "fc_private", '{"RAW-CALL-PRIVATE-CANARY":')
        )
    assert invalid_info.value.provider == "qwencloud"
    assert "RAW-CALL-PRIVATE-CANARY" not in str(invalid_info.value)
    assert invalid_info.value.__cause__ is None
    assert invalid_info.value.__context__ is None

    non_object = QwenResponsesStreamTranslator()
    non_object.feed(_function_added(0, 0, "fc_array", "call_array", "lookup"))
    with pytest.raises(ChatProviderError):
        non_object.feed(_arguments_done(1, 0, "fc_array", "[]"))

    for malformed_item in (
        _function_item("fc_missing_id", "", "lookup", ""),
        _function_item("fc_missing_name", "call_name", "", ""),
    ):
        with pytest.raises(ChatProviderError):
            QwenResponsesStreamTranslator().feed(
                {
                    "type": "response.output_item.added",
                    "sequence_number": 0,
                    "output_index": 0,
                    "item": malformed_item,
                }
            )

    duplicate = QwenResponsesStreamTranslator()
    duplicate.feed(_function_added(0, 0, "fc_first", "call_dup", "lookup"))
    with pytest.raises(ChatProviderError):
        duplicate.feed(_function_added(1, 1, "fc_second", "call_dup", "lookup"))

    mismatches = (
        _arguments_delta(1, 1, "fc_expected", "{}"),
        _arguments_delta(1, 0, "fc_wrong_transport", "{}"),
        _output_done(
            1,
            0,
            _function_item("fc_expected", "call_wrong", "lookup", "{}"),
        ),
        _output_done(
            1,
            0,
            _function_item("fc_expected", "call_expected", "wrong_name", "{}"),
        ),
    )
    for mismatched_event in mismatches:
        translator = QwenResponsesStreamTranslator()
        translator.feed(_function_added(0, 0, "fc_expected", "call_expected", "lookup"))
        with pytest.raises(ChatProviderError) as mismatch_info:
            translator.feed(mismatched_event)
        assert mismatch_info.value.provider == "qwencloud"

    conflicting_arguments = QwenResponsesStreamTranslator()
    conflicting_arguments.feed(
        _function_added(0, 0, "fc_conflict", "call_conflict", "lookup")
    )
    conflicting_arguments.feed(_arguments_delta(1, 0, "fc_conflict", '{"value":1}'))
    with pytest.raises(ChatProviderError):
        conflicting_arguments.feed(
            _output_done(
                2,
                0,
                _function_item("fc_conflict", "call_conflict", "lookup", '{"value":2}'),
            )
        )

    missing_terminal_call = QwenResponsesStreamTranslator()
    missing_terminal_call.feed(
        _function_added(0, 0, "fc_missing", "call_missing", "lookup")
    )
    with pytest.raises(ChatProviderError) as terminal_info:
        missing_terminal_call.feed(_terminal(1, []))
    assert terminal_info.value.provider == "qwencloud"


def test_chat_stream_preserves_openai_deltas_and_usage() -> None:
    events = [
        {
            "id": "chatcmpl_safe",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_safe",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_safe",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_safe",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"safe"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": "chatcmpl_safe",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        },
    ]
    wire = (
        b"".join(
            b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\r\n\r\n"
            for event in events
        )
        + b"data: [DONE]\r\n\r\n"
    )
    response = _ByteStreamResponse([wire[:17], wire[17:89], wire[89:]])
    session = _ByteStreamSession([response])

    assert (
        list(
            QwenCloudStream(
                response=response,  # type: ignore[arg-type]
                session=session,  # type: ignore[arg-type]
                api_mode="chat_completions",
            )
        )
        == events
    )
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_stream_retries_only_before_first_consumed_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retryable = _ByteStreamResponse([], status_code=503)
    malformed_after_body = _ByteStreamResponse(
        [b'data: {"choices":[{"delta":{"content":"private"}}]}\n\n', b"data: {"]
    )
    replay_canary = _ByteStreamResponse(
        [b'data: {"choices":[{"delta":{"content":"replayed"}}]}\n\n']
    )
    session = _ByteStreamSession([retryable, malformed_after_body, replay_canary])
    monkeypatch.setattr(
        qwencloud,
        "requests",
        SimpleNamespace(Session=lambda: session),
    )
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {"timeout": 3, "retries": 2, "retry_delay": 0}
                }
            }
        ),
    )

    stream = qwencloud.chat_with_qwencloud(
        input_data=[{"role": "user", "content": "hello"}],
        model="qwen3.8-max",
        api_key="key",
        streaming=True,
        api_base_url="https://qwen.example/v1",
        api_mode="chat_completions",
    )

    assert not isinstance(stream, dict)
    assert len(session.posts) == 2
    assert retryable.close_calls == 1
    assert malformed_after_body.close_calls == 0
    assert next(stream)["choices"][0]["delta"]["content"] == "private"
    with pytest.raises(ChatProviderError) as exc_info:
        next(stream)
    assert exc_info.value.provider == "qwencloud"
    assert len(session.posts) == 2
    assert replay_canary.iter_content_calls == 0
    assert malformed_after_body.close_calls == 1
    assert session.close_calls == 1


def test_stream_close_is_idempotent_and_closes_response_and_session() -> None:
    response = _ByteStreamResponse(
        [b'data: {"choices":[{"delta":{"content":"unused"}}]}\n\n']
    )
    session = _ByteStreamSession([response])
    stream = QwenCloudStream(
        response=response,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        api_mode="chat_completions",
    )

    stream.close()
    stream.close()
    assert response.close_calls == 1
    assert session.close_calls == 1
    with pytest.raises(StopIteration):
        next(stream)


@pytest.mark.parametrize(
    "chunks",
    (
        [b"data: {not-json}\n\n"],
        [b"data: [1,2,3]\n\n"],
        [b'data: {"type":"error","error":{"message":"RAW-ERROR-CANARY"}}\n\n'],
        [b"data: \xff\n\n"],
        [b'data: {"choices":[]}'],
    ),
    ids=(
        "malformed-json",
        "non-object-json",
        "provider-error-event",
        "invalid-utf8",
        "incomplete-record",
    ),
)
def test_stream_malformed_json_and_error_event_are_typed_closed_and_not_retried(
    chunks: list[bytes],
) -> None:
    response = _ByteStreamResponse(chunks)
    session = _ByteStreamSession([response])
    stream = QwenCloudStream(
        response=response,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        api_mode="chat_completions",
    )

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert "RAW-ERROR-CANARY" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert response.iter_content_calls == 1
    assert response.close_calls == 1
    assert session.close_calls == 1
