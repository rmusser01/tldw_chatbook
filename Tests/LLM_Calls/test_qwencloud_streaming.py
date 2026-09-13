"""Streaming contracts for the QwenCloud dual-API adapter."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import requests
from loguru import logger

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
import tldw_chatbook.LLM_Calls.qwencloud as qwencloud
import tldw_chatbook.LLM_Calls.qwencloud_streaming as qwencloud_streaming
import tldw_chatbook.LLM_Calls.hosted_chat_streaming as hosted_streaming
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


class _FailingByteStreamResponse(_ByteStreamResponse):
    def __init__(
        self,
        chunks: list[bytes],
        *,
        error: requests.exceptions.RequestException,
    ) -> None:
        super().__init__(chunks)
        self.error = error

    def iter_content(self, chunk_size: int) -> Any:
        assert chunk_size > 0
        self.iter_content_calls += 1
        yield from self.chunks
        raise self.error


class _CloseFailingByteStreamResponse(_ByteStreamResponse):
    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("RAW-RESPONSE-CLOSE-CANARY")


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


class _CloseFailingByteStreamSession(_ByteStreamSession):
    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("RAW-SESSION-CLOSE-CANARY")


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
    *,
    status: str | None = "in_progress",
) -> dict[str, Any]:
    item = _function_item(item_id, call_id, name, "", status="in_progress")
    if status is None:
        del item["status"]
    else:
        item["status"] = status
    return {
        "type": "response.output_item.added",
        "sequence_number": sequence,
        "output_index": output_index,
        "item": item,
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


def _message_added(
    sequence: int,
    output_index: int,
    item_id: str,
    *,
    status: str | None = "in_progress",
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [],
    }
    if status is not None:
        item["status"] = status
    return {
        "type": "response.output_item.added",
        "sequence_number": sequence,
        "output_index": output_index,
        "item": item,
    }


def _reasoning_item(item_id: str, *, status: str | None) -> dict[str, Any]:
    item: dict[str, Any] = {"id": item_id, "type": "reasoning", "summary": []}
    if status is not None:
        item["status"] = status
    return item


def _reasoning_added(
    sequence: int,
    output_index: int,
    item_id: str,
    *,
    status: str | None = "in_progress",
) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "sequence_number": sequence,
        "output_index": output_index,
        "item": _reasoning_item(item_id, status=status),
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


def _chat_wire(events: list[dict[str, Any]], *, done: bool = True) -> list[bytes]:
    wire = b"".join(
        b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
        for event in events
    )
    if done:
        wire += b"data: [DONE]\n\n"
    return [wire]


def _chat_stream(
    events: list[dict[str, Any]], *, done: bool = True
) -> tuple[QwenCloudStream, _ByteStreamResponse, _ByteStreamSession]:
    response = _ByteStreamResponse(_chat_wire(events, done=done))
    session = _ByteStreamSession([response])
    stream = QwenCloudStream(
        response=response,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        api_mode="chat_completions",
    )
    return stream, response, session


def _chat_tool_start() -> dict[str, Any]:
    return {
        "id": "chatcmpl_metadata",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_metadata",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": ""},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    }


def _chat_tool_terminal() -> dict[str, Any]:
    return {
        "id": "chatcmpl_metadata",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"safe":true}'},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _chat_text_terminal(*, reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl_metadata",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "safe"},
                "finish_reason": reason,
            }
        ],
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


@pytest.mark.parametrize("cap", ("events", "line", "record"))
def test_stream_caps_provider_controlled_event_and_record_sizes(
    monkeypatch: pytest.MonkeyPatch,
    cap: str,
) -> None:
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ]
        },
        _chat_text_terminal(),
    ]
    if cap == "events":
        monkeypatch.setattr(qwencloud_streaming, "_MAX_STREAM_EVENTS", 1, raising=False)
    elif cap == "line":
        monkeypatch.setattr(
            qwencloud_streaming, "_MAX_SSE_LINE_CHARS", 32, raising=False
        )
    else:
        monkeypatch.setattr(
            qwencloud_streaming, "_MAX_SSE_RECORD_CHARS", 32, raising=False
        )
    stream, response, session = _chat_stream(events)

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert response.close_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize("cap", ("segments", "data-lines"))
def test_stream_caps_sse_reference_counts_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    cap: str,
) -> None:
    if cap == "segments":
        monkeypatch.setattr(
            qwencloud_streaming, "_MAX_SSE_LINE_SEGMENTS", 2, raising=False
        )
        event = json.dumps(_chat_text_terminal(), separators=(",", ":")).encode()
        chunks = [b"data: ", event[:1], event[1:], b"\n\ndata: [DONE]\n\n"]
    else:
        monkeypatch.setattr(
            qwencloud_streaming, "_MAX_SSE_DATA_LINES", 2, raising=False
        )
        chunks = [
            b'data: {"id":"RAW-SSE-REFERENCE-CANARY",\n',
            b'data: "choices":\n',
            b'data: [{"index":0,"delta":{"content":"safe"},',
            b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        ]
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
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "RAW-SSE-REFERENCE-CANARY" not in str(exc_info.value)
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_sse_accepts_long_valid_record_below_private_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwencloud_streaming, "_MAX_SSE_LINE_CHARS", 2048, raising=False)
    monkeypatch.setattr(
        qwencloud_streaming, "_MAX_SSE_RECORD_CHARS", 2048, raising=False
    )
    terminal = _chat_text_terminal()
    terminal["choices"][0]["delta"]["content"] = "x" * 512
    stream, response, session = _chat_stream([terminal])

    assert list(stream) == [terminal]
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_sse_line_accumulation_is_structurally_linear() -> None:
    source = inspect.getsource(hosted_streaming.SSERecordDecoder._consume_text)

    assert "buffered +=" not in source
    assert "segments" in source


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


def test_responses_sequence_state_is_compact_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwencloud_streaming, "_MAX_TRACKED_SEQUENCES", 2, raising=False)
    translator = QwenResponsesStreamTranslator()
    created = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_compact_RAW-SEQUENCE-CANARY",
            "status": "in_progress",
            "output": [],
        },
    }
    translator.feed(created)
    stored = translator._seen_sequences[0]  # noqa: SLF001
    assert isinstance(stored, bytes)
    assert len(stored) <= 64
    assert "RAW-SEQUENCE-CANARY" not in repr(stored)
    translator.feed(
        {
            "type": "response.in_progress",
            "sequence_number": 1,
            "response": {"id": "resp_compact", "status": "in_progress"},
        }
    )

    with pytest.raises(ChatProviderError) as exc_info:
        translator.feed(
            {
                "type": "response.in_progress",
                "sequence_number": 2,
                "response": {"id": "resp_compact", "status": "in_progress"},
            }
        )

    assert exc_info.value.provider == "qwencloud"
    assert "RAW-SEQUENCE-CANARY" not in str(exc_info.value)


@pytest.mark.parametrize("kind", ("text", "arguments"))
def test_responses_accumulated_output_cap_is_typed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    metadata_chars = len("msg_cap") if kind == "text" else len("fc_capcall_caplookup")
    monkeypatch.setattr(
        qwencloud_streaming,
        "_MAX_OUTPUT_CHARS",
        metadata_chars + 5,
        raising=False,
    )
    translator = QwenResponsesStreamTranslator()
    if kind == "text":
        translator.feed(_message_added(0, 0, "msg_cap"))
        translator.feed(_content_added(1, 0, "msg_cap"))
        translator.feed(_text_delta(2, 0, "msg_cap", "abc"))
        crossing = _text_delta(3, 0, "msg_cap", "RAW")
    else:
        translator.feed(_function_added(0, 0, "fc_cap", "call_cap", "lookup"))
        translator.feed(_arguments_delta(1, 0, "fc_cap", "abc"))
        crossing = _arguments_delta(2, 0, "fc_cap", "RAW")

    with pytest.raises(ChatProviderError) as exc_info:
        translator.feed(crossing)

    assert exc_info.value.provider == "qwencloud"
    assert "RAW" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_responses_text_and_argument_fragments_use_linear_lists_then_release() -> None:
    translator = QwenResponsesStreamTranslator()
    translator.feed(_message_added(0, 0, "msg_fragments"))
    translator.feed(_content_added(1, 0, "msg_fragments"))
    translator.feed(_text_delta(2, 0, "msg_fragments", "ab"))
    translator.feed(_text_delta(3, 0, "msg_fragments", "cd"))
    text_state = translator._text_parts[(0, 0)]  # noqa: SLF001
    assert text_state.fragments == ["ab", "cd"]
    assert not hasattr(text_state, "emitted_text")
    translator.feed(_text_done(4, 0, "msg_fragments", "abcd"))
    assert text_state.fragments == []
    assert text_state.final_text == "abcd"

    translator.feed(_function_added(5, 1, "fc_fragments", "call_fragments", "lookup"))
    translator.feed(_arguments_delta(6, 1, "fc_fragments", '{"a":'))
    translator.feed(_arguments_delta(7, 1, "fc_fragments", "1}"))
    call_state = translator._function_calls[1]  # noqa: SLF001
    assert call_state.argument_fragments == ['{"a":', "1}"]
    assert not hasattr(call_state, "emitted_arguments")
    translator.feed(_arguments_done(8, 1, "fc_fragments", '{"a":1}'))
    assert call_state.argument_fragments == []
    assert call_state.final_arguments == '{"a":1}'


def test_responses_retained_metadata_is_charged_exactly_once() -> None:
    translator = QwenResponsesStreamTranslator()
    expected_characters = 0
    for index in range(500):
        item_id = f"fc_{index}_" + "i" * 64
        call_id = f"call_{index}_" + "c" * 64
        name = f"tool_{index}_" + "n" * 64
        translator.feed(_function_added(index, index, item_id, call_id, name))
        expected_characters += len(item_id) + len(call_id) + len(name)

    assert translator._output_chars == expected_characters  # noqa: SLF001
    last_item_id = "fc_499_" + "i" * 64
    translator.feed(_arguments_delta(500, 499, last_item_id, "{}"))
    assert translator._output_chars == expected_characters + 2  # noqa: SLF001


@pytest.mark.parametrize(
    "field",
    ("message-id", "reasoning-id", "function-id", "call-id", "name", "cumulative"),
)
def test_responses_retained_metadata_fields_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    monkeypatch.setattr(qwencloud_streaming, "_MAX_METADATA_CHARS", 8, raising=False)
    monkeypatch.setattr(qwencloud_streaming, "_MAX_OUTPUT_CHARS", 16, raising=False)
    too_long = "RAW-METADATA-CANARY"
    if field == "message-id":
        event = _message_added(0, 0, too_long)
    elif field == "reasoning-id":
        event = _reasoning_added(0, 0, too_long)
    else:
        event = _function_added(
            0,
            0,
            too_long
            if field == "function-id"
            else "item12"
            if field == "cumulative"
            else "item",
            too_long
            if field == "call-id"
            else "call12"
            if field == "cumulative"
            else "call",
            too_long
            if field == "name"
            else "tool12"
            if field == "cumulative"
            else "tool",
        )

    with pytest.raises(ChatProviderError) as exc_info:
        QwenResponsesStreamTranslator().feed(event)

    assert exc_info.value.provider == "qwencloud"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert too_long not in str(exc_info.value)


@pytest.mark.parametrize("field", ("id", "name", "cumulative"))
def test_chat_retained_tool_metadata_is_bounded_and_charged_once(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    monkeypatch.setattr(qwencloud_streaming, "_MAX_METADATA_CHARS", 8, raising=False)
    monkeypatch.setattr(qwencloud_streaming, "_MAX_OUTPUT_CHARS", 16, raising=False)
    start = _chat_tool_start()
    tool = start["choices"][0]["delta"]["tool_calls"][0]
    too_long = "RAW-METADATA-CANARY"
    if field == "id":
        tool["id"] = too_long
    elif field == "name":
        tool["function"]["name"] = too_long
    else:
        tool["id"] = "12345678"
        tool["function"]["name"] = "abcdefgh"
        tool["function"]["arguments"] = "x"
    stream, response, session = _chat_stream([start, _chat_tool_terminal()])

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert too_long not in str(exc_info.value)
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_chat_repeated_tool_metadata_is_not_double_charged() -> None:
    start = _chat_tool_start()
    terminal = _chat_tool_terminal()
    terminal_tool = terminal["choices"][0]["delta"]["tool_calls"][0]
    terminal_tool["id"] = "call_metadata"
    terminal_tool["type"] = "function"
    terminal_tool["function"]["name"] = "lookup"
    stream, _, _ = _chat_stream([start, terminal])

    list(stream)

    expected = len("call_metadata") + len("lookup") + len('{"safe":true}')
    assert stream._translator._output_chars == expected  # noqa: SLF001


def test_responses_terminal_replay_is_strict_and_finish_safe() -> None:
    translator = QwenResponsesStreamTranslator()
    translator.feed(_message_added(0, 0, "msg_replay"))
    translator.feed(_content_added(1, 0, "msg_replay"))
    translator.feed(_text_delta(2, 0, "msg_replay", "safe"))
    completed = _terminal(
        3,
        [_message_item("msg_replay", "safe")],
        usage={"input_tokens": 0, "output_tokens": 1, "total_tokens": 1},
    )

    assert translator.feed(completed)[-1]["choices"][0]["finish_reason"] == "stop"
    assert translator.feed(deepcopy(completed)) == ()
    reordered = {
        "response": deepcopy(completed["response"]),
        "sequence_number": completed["sequence_number"],
        "type": completed["type"],
    }
    assert translator.feed(reordered) == ()
    assert translator.finish() == ()

    with pytest.raises(ChatProviderError, match="object"):
        translator.feed([])  # type: ignore[arg-type]
    with pytest.raises(ChatProviderError, match="sequence"):
        translator.feed({"type": "response.in_progress"})
    with pytest.raises(ChatProviderError, match="terminal"):
        translator.feed(
            {
                "type": "response.in_progress",
                "sequence_number": 4,
                "response": {"status": "in_progress", "output": []},
            }
        )


@pytest.mark.parametrize("replacement", (False, 0.0), ids=("bool", "float"))
def test_responses_sequence_replay_uses_type_sensitive_json_equality(
    replacement: bool | float,
) -> None:
    translator = QwenResponsesStreamTranslator()
    event = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_strict",
            "status": "in_progress",
            "output": [{"index": 0}],
        },
    }
    assert translator.feed(event) == ()
    conflict = deepcopy(event)
    conflict["response"]["output"][0]["index"] = replacement

    with pytest.raises(ChatProviderError, match="sequence") as exc_info:
        translator.feed(conflict)

    assert exc_info.value.provider == "qwencloud"


def test_responses_direct_feed_rejects_non_json_nested_containers() -> None:
    event = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_direct",
            "status": "in_progress",
            "output": ({"index": 0},),
        },
    }

    with pytest.raises(ChatProviderError) as exc_info:
        QwenResponsesStreamTranslator().feed(event)

    assert exc_info.value.provider == "qwencloud"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_responses_direct_feed_accepts_top_level_mapping_copy() -> None:
    event = MappingProxyType(
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_direct",
                "status": "in_progress",
                "output": [],
            },
        }
    )

    assert QwenResponsesStreamTranslator().feed(event) == ()


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


def test_responses_function_indexes_preserve_terminal_output_order() -> None:
    translator = QwenResponsesStreamTranslator()
    chunks: list[dict[str, Any]] = []
    chunks.extend(
        translator.feed(_function_added(0, 1, "fc_one", "call_one", "second"))
    )
    chunks.extend(
        translator.feed(_function_added(1, 0, "fc_zero", "call_zero", "first"))
    )
    chunks.extend(
        translator.feed(
            _terminal(
                2,
                [
                    _function_item("fc_zero", "call_zero", "first", '{"n":0}'),
                    _function_item("fc_one", "call_one", "second", '{"n":1}'),
                ],
            )
        )
    )

    identities = {
        tool_call["index"]: tool_call["id"]
        for chunk in chunks
        for tool_call in chunk["choices"][0]["delta"].get("tool_calls", [])
        if "id" in tool_call
    }
    assert identities == {1: "call_one", 0: "call_zero"}
    assert [identities[index] for index in sorted(identities)] == [
        "call_zero",
        "call_one",
    ]


def test_responses_stable_function_indexes_survive_interleaved_output_types() -> None:
    translator = QwenResponsesStreamTranslator()
    chunks: list[dict[str, Any]] = []
    chunks.extend(
        translator.feed(_function_added(0, 3, "fc_three", "call_three", "third"))
    )
    chunks.extend(translator.feed(_message_added(1, 0, "msg_zero")))
    chunks.extend(translator.feed(_content_added(2, 0, "msg_zero")))
    chunks.extend(translator.feed(_text_delta(3, 0, "msg_zero", "safe")))
    chunks.extend(translator.feed(_reasoning_added(4, 1, "reasoning_one")))
    chunks.extend(
        translator.feed(_function_added(5, 2, "fc_two", "call_two", "second"))
    )
    chunks.extend(
        translator.feed(
            _terminal(
                6,
                [
                    _message_item("msg_zero", "safe"),
                    _reasoning_item("reasoning_one", status="completed"),
                    _function_item("fc_two", "call_two", "second", '{"n":2}'),
                    _function_item("fc_three", "call_three", "third", '{"n":3}'),
                ],
            )
        )
    )

    identities = {
        tool_call["index"]: tool_call["id"]
        for chunk in chunks
        for tool_call in chunk["choices"][0]["delta"].get("tool_calls", [])
        if "id" in tool_call
    }
    assert identities == {3: "call_three", 2: "call_two"}
    assert [identities[index] for index in sorted(identities)] == [
        "call_two",
        "call_three",
    ]


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


@pytest.mark.parametrize("source", ("output-item-done", "terminal-only"))
@pytest.mark.parametrize(
    "status", ("incomplete", "failed", "cancelled", "unknown", None)
)
def test_responses_function_status_cannot_become_successful_execution(
    source: str,
    status: str | None,
) -> None:
    translator = QwenResponsesStreamTranslator()
    chunks = list(
        translator.feed(_function_added(0, 0, "fc_status", "call_status", "lookup"))
    )
    item = _function_item("fc_status", "call_status", "lookup", '{"safe":true}')
    if status is None:
        del item["status"]
    else:
        item["status"] = status
    event = (
        _output_done(1, 0, item)
        if source == "output-item-done"
        else _terminal(1, [item])
    )

    with pytest.raises(ChatProviderError) as exc_info:
        chunks.extend(translator.feed(event))

    assert exc_info.value.provider == "qwencloud"
    assert all(
        chunk["choices"][0].get("finish_reason") != "tool_calls" for chunk in chunks
    )


@pytest.mark.parametrize("status", ("failed", "cancelled", "unknown", None))
def test_responses_completed_message_status_must_be_completed(
    status: str | None,
) -> None:
    translator = QwenResponsesStreamTranslator()
    translator.feed(_message_added(0, 0, "msg_status"))
    translator.feed(_content_added(1, 0, "msg_status"))
    translator.feed(_text_delta(2, 0, "msg_status", "safe"))
    item = _message_item("msg_status", "safe")
    if status is None:
        del item["status"]
    else:
        item["status"] = status

    with pytest.raises(ChatProviderError) as exc_info:
        translator.feed(_terminal(3, [item]))

    assert exc_info.value.provider == "qwencloud"


@pytest.mark.parametrize("item_type", ("message", "function_call", "reasoning"))
@pytest.mark.parametrize("status", ("completed", "incomplete", "unknown", None))
def test_responses_output_item_added_requires_in_progress_status(
    item_type: str,
    status: str | None,
) -> None:
    if item_type == "message":
        event = _message_added(0, 0, "item_status", status=status)
    elif item_type == "function_call":
        event = _function_added(
            0,
            0,
            "item_status",
            "call_status",
            "lookup",
            status=status,
        )
    else:
        event = _reasoning_added(0, 0, "item_status", status=status)

    with pytest.raises(ChatProviderError) as exc_info:
        QwenResponsesStreamTranslator().feed(event)

    assert exc_info.value.provider == "qwencloud"


def test_responses_done_terminal_and_incomplete_statuses_are_consistent() -> None:
    done_mismatch = QwenResponsesStreamTranslator()
    done_mismatch.feed(_message_added(0, 0, "msg_done"))
    done_mismatch.feed(_content_added(1, 0, "msg_done"))
    done_mismatch.feed(_text_delta(2, 0, "msg_done", "safe"))
    done_mismatch.feed(
        _output_done(3, 0, _message_item("msg_done", "safe", status="completed"))
    )
    with pytest.raises(ChatProviderError):
        done_mismatch.feed(
            _terminal(
                4,
                [_message_item("msg_done", "safe", status="incomplete")],
                event_type="response.incomplete",
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        )

    terminal_mismatch = QwenResponsesStreamTranslator()
    terminal_mismatch.feed(_message_added(0, 0, "msg_partial"))
    terminal_mismatch.feed(_content_added(1, 0, "msg_partial"))
    terminal_mismatch.feed(_text_delta(2, 0, "msg_partial", "partial"))
    with pytest.raises(ChatProviderError):
        terminal_mismatch.feed(
            _terminal(
                3,
                [_message_item("msg_partial", "partial", status="completed")],
                event_type="response.incomplete",
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        )

    valid_partial = QwenResponsesStreamTranslator()
    valid_partial.feed(_message_added(0, 0, "msg_partial"))
    valid_partial.feed(_content_added(1, 0, "msg_partial"))
    valid_partial.feed(_text_delta(2, 0, "msg_partial", "partial"))
    assert (
        valid_partial.feed(
            _terminal(
                3,
                [_message_item("msg_partial", "partial", status="incomplete")],
                event_type="response.incomplete",
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        )[-1]["choices"][0]["finish_reason"]
        == "length"
    )


def test_responses_reasoning_status_is_validated_without_surface() -> None:
    translator = QwenResponsesStreamTranslator()
    assert translator.feed(_reasoning_added(0, 0, "reasoning_safe")) == ()
    assert translator.feed(_message_added(1, 1, "msg_safe")) == ()
    assert translator.feed(_content_added(2, 1, "msg_safe")) == ()
    assert translator.feed(_text_delta(3, 1, "msg_safe", "safe"))
    with pytest.raises(ChatProviderError):
        translator.feed(
            _output_done(
                4,
                0,
                _reasoning_item("reasoning_safe", status="failed"),
            )
        )

    terminal = QwenResponsesStreamTranslator()
    terminal.feed(_reasoning_added(0, 0, "reasoning_safe"))
    terminal.feed(_message_added(1, 1, "msg_safe"))
    terminal.feed(_content_added(2, 1, "msg_safe"))
    terminal.feed(_text_delta(3, 1, "msg_safe", "safe"))
    with pytest.raises(ChatProviderError):
        terminal.feed(
            _terminal(
                4,
                [
                    _reasoning_item("reasoning_safe", status="unknown"),
                    _message_item("msg_safe", "safe"),
                ],
            )
        )

    valid = QwenResponsesStreamTranslator()
    assert valid.feed(_reasoning_added(0, 0, "reasoning_safe")) == ()
    assert valid.feed(_message_added(1, 1, "msg_safe")) == ()
    assert valid.feed(_content_added(2, 1, "msg_safe")) == ()
    text = valid.feed(_text_delta(3, 1, "msg_safe", "safe"))
    assert (
        valid.feed(
            _output_done(
                4,
                0,
                _reasoning_item("reasoning_safe", status="completed"),
            )
        )
        == ()
    )
    terminal_chunks = valid.feed(
        _terminal(
            5,
            [
                _reasoning_item("reasoning_safe", status="completed"),
                _message_item("msg_safe", "safe"),
            ],
        )
    )
    assert text == ({"choices": [{"delta": {"content": "safe"}}]},)
    assert terminal_chunks == (
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
            "usage": {},
        },
    )


@pytest.mark.parametrize(
    "case",
    (
        "choice-index-bool",
        "choice-index-float",
        "choice-index-negative",
        "tool-index-bool",
        "tool-index-float",
        "tool-index-negative",
        "id-number",
        "id-blank",
        "type-other",
        "type-mapping",
        "function-nonmapping",
        "name-number",
        "name-blank",
        "arguments-number",
        "content-number",
        "role-number",
    ),
)
def test_chat_stream_rejects_malformed_choice_and_tool_metadata(case: str) -> None:
    start = _chat_tool_start()
    choice = start["choices"][0]
    tool = choice["delta"]["tool_calls"][0]
    if case == "choice-index-bool":
        choice["index"] = False
    elif case == "choice-index-float":
        choice["index"] = 0.0
    elif case == "choice-index-negative":
        choice["index"] = -1
    elif case == "tool-index-bool":
        tool["index"] = False
    elif case == "tool-index-float":
        tool["index"] = 0.0
    elif case == "tool-index-negative":
        tool["index"] = -1
    elif case == "id-number":
        tool["id"] = 7
    elif case == "id-blank":
        tool["id"] = "   "
    elif case == "type-other":
        tool["type"] = "builtin"
    elif case == "type-mapping":
        tool["type"] = {"RAW-METADATA-CANARY": True}
    elif case == "function-nonmapping":
        tool["function"] = "RAW-METADATA-CANARY"
    elif case == "name-number":
        tool["function"]["name"] = 7
    elif case == "name-blank":
        tool["function"]["name"] = "   "
    elif case == "arguments-number":
        tool["function"]["arguments"] = 7
    elif case == "content-number":
        choice["delta"]["content"] = 7
    else:
        choice["delta"]["role"] = 7
    stream, response, session = _chat_stream([start, _chat_tool_terminal()])

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert "RAW-METADATA-CANARY" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert response.close_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize("case", ("conflicting-id", "conflicting-name", "duplicate-id"))
def test_chat_stream_rejects_conflicting_or_duplicate_tool_identity(case: str) -> None:
    start = _chat_tool_start()
    later = _chat_tool_terminal()
    later_tool = later["choices"][0]["delta"]["tool_calls"][0]
    if case == "conflicting-id":
        later_tool["id"] = "call_conflict"
    elif case == "conflicting-name":
        later_tool["function"]["name"] = "other_tool"
    else:
        duplicate = deepcopy(start["choices"][0]["delta"]["tool_calls"][0])
        duplicate["index"] = 1
        start["choices"][0]["delta"]["tool_calls"].append(duplicate)
        later["choices"][0]["delta"]["tool_calls"].append(
            {"index": 1, "function": {"arguments": "{}"}}
        )
    stream, response, session = _chat_stream([start, later])

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert response.close_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize(
    "case",
    (
        "arbitrary-reason",
        "blank-reason",
        "missing-reason",
        "stop-with-tools",
        "length-with-tools",
        "tool-calls-without-tools",
    ),
)
def test_chat_stream_terminal_reason_matches_tool_fragment_state(case: str) -> None:
    if case in {"stop-with-tools", "length-with-tools"}:
        terminal = _chat_tool_terminal()
        terminal["choices"][0]["finish_reason"] = case.removesuffix("-with-tools")
        events = [_chat_tool_start(), terminal]
    else:
        terminal = _chat_text_terminal(
            reason={
                "arbitrary-reason": "content_filter",
                "blank-reason": "   ",
                "missing-reason": "stop",
                "tool-calls-without-tools": "tool_calls",
            }[case]
        )
        if case == "missing-reason":
            del terminal["choices"][0]["finish_reason"]
        events = [terminal]
    stream, response, session = _chat_stream(events)

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert response.close_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize(
    "case",
    (
        "done-before-terminal",
        "eof-before-terminal",
        "usage-before-terminal",
        "chunk-after-terminal",
        "one-of-two-choices-unterminated",
    ),
)
def test_chat_stream_requires_ordered_complete_terminal_lifecycle(case: str) -> None:
    done = case != "eof-before-terminal"
    if case in {"done-before-terminal", "eof-before-terminal"}:
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "partial"},
                        "finish_reason": None,
                    }
                ]
            }
        ]
    elif case == "usage-before-terminal":
        events = [
            {"choices": [], "usage": {"total_tokens": 1}},
            _chat_text_terminal(),
        ]
    elif case == "chunk-after-terminal":
        events = [
            _chat_text_terminal(),
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "late"},
                        "finish_reason": None,
                    }
                ]
            },
        ]
    else:
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "done"},
                        "finish_reason": "stop",
                    },
                    {
                        "index": 1,
                        "delta": {"content": "partial"},
                        "finish_reason": None,
                    },
                ]
            }
        ]
    stream, response, session = _chat_stream(events, done=done)

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_chat_stream_preserves_multiple_valid_choices_role_tools_and_usage() -> None:
    events = [
        {
            "id": "chatcmpl_multi",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                },
                {
                    "index": 1,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 2,
                                "id": "call_multi",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{"},
                            }
                        ],
                    },
                    "finish_reason": None,
                },
            ],
        },
        {
            "id": "chatcmpl_multi",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "safe"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 2,
                                "function": {"arguments": "}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
            ],
        },
        {
            "id": "chatcmpl_multi",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    ]
    stream, response, session = _chat_stream(events)

    assert list(stream) == events
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_chat_stream_drops_unvalidated_private_extras_without_coercion() -> None:
    event = _chat_tool_start()
    event["id"] = "chatcmpl_safe"
    event["private"] = "RAW-PRIVATE-CANARY"
    choice = event["choices"][0]
    choice["private"] = "RAW-PRIVATE-CANARY"
    choice["delta"]["private"] = "RAW-PRIVATE-CANARY"
    tool = choice["delta"]["tool_calls"][0]
    tool["private"] = "RAW-PRIVATE-CANARY"
    tool["function"]["private"] = "RAW-PRIVATE-CANARY"
    terminal = _chat_tool_terminal()
    stream, response, session = _chat_stream([event, terminal])

    emitted = list(stream)

    assert emitted[0] == {
        "id": "chatcmpl_safe",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_metadata",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": ""},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    }
    assert "RAW-PRIVATE-CANARY" not in repr(emitted)
    assert response.close_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize("nullable_field", ("tool_calls", "usage"))
def test_chat_stream_nullable_projection_paths_close_normally(
    nullable_field: str,
) -> None:
    terminal = _chat_text_terminal()
    if nullable_field == "tool_calls":
        terminal["choices"][0]["delta"]["tool_calls"] = None
    else:
        terminal["usage"] = None
    stream, response, session = _chat_stream([terminal])

    assert list(stream) == [_chat_text_terminal()]
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_chat_stream_nullable_fields_preserve_valid_accumulator_flow() -> None:
    events = [
        {
            "id": "chatcmpl_nullable",
            "usage": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_nullable",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_nullable",
            "usage": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "safe", "tool_calls": None},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_nullable",
            "usage": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"safe":true}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": "chatcmpl_nullable",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    ]
    expected = deepcopy(events)
    for event in expected[:3]:
        event.pop("usage")
    expected[0]["choices"][0]["delta"].pop("content")
    expected[1]["choices"][0]["delta"].pop("tool_calls")
    stream, response, session = _chat_stream(events)

    chunks = list(stream)

    from tldw_chatbook.Chat.console_provider_gateway import _ToolCallAccumulator

    accumulator = _ToolCallAccumulator()
    for chunk in chunks:
        accumulator.feed_payload(chunk)
    assert chunks == expected
    assert accumulator.calls() == (
        {
            "id": "call_nullable",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"safe":true}'},
        },
    )
    assert response.close_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize("stage", ("decode", "feed", "finish"))
def test_stream_unexpected_exceptions_are_typed_redacted_and_closed(
    stage: str,
) -> None:
    canary = "RAW-UNEXPECTED-STREAM-CANARY"
    stream, response, session = _chat_stream([_chat_text_terminal()])

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(canary)

    if stage == "decode":
        stream._decode_event = explode  # type: ignore[method-assign]  # noqa: SLF001
    elif stage == "feed":
        stream._translator.feed = explode  # type: ignore[method-assign]  # noqa: SLF001
    else:
        stream._translator.finish = explode  # type: ignore[method-assign]  # noqa: SLF001

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    try:
        if stage == "finish":
            assert next(stream)["choices"][0]["finish_reason"] == "stop"
        with pytest.raises(ChatProviderError) as exc_info:
            next(stream)
    finally:
        logger.remove(sink_id)

    assert exc_info.value.provider == "qwencloud"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert canary not in str(exc_info.value) + "".join(records)
    assert response.close_calls == 1
    assert session.close_calls == 1


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
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"private"},'
            b'"finish_reason":null}]}\n\n',
            b"data: {",
        ]
    )
    replay_canary = _ByteStreamResponse(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"replayed"},'
            b'"finish_reason":"stop"}]}\n\n'
        ]
    )
    session = _ByteStreamSession([retryable, malformed_after_body, replay_canary])
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
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


@pytest.mark.parametrize(
    "error_type",
    (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.Timeout,
    ),
)
@pytest.mark.parametrize(
    "after_event", (False, True), ids=("before-bytes", "after-event")
)
def test_stream_body_read_failures_are_typed_closed_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[requests.exceptions.RequestException],
    after_event: bool,
) -> None:
    canary = "RAW-STREAM-READ-CANARY"
    chunks = (
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"safe"},'
            b'"finish_reason":null}]}\n\n'
        ]
        if after_event
        else []
    )
    response = _FailingByteStreamResponse(chunks, error=error_type(canary))
    session = _ByteStreamSession([response])
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
    )
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {"timeout": 3, "retries": 3, "retry_delay": 0}
                }
            }
        ),
    )
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    try:
        stream = qwencloud.chat_with_qwencloud(
            input_data=[{"role": "user", "content": "hello"}],
            model="qwen3.8-max",
            api_key="key",
            streaming=True,
            api_base_url="https://qwen.example/v1",
            api_mode="chat_completions",
        )
        if after_event:
            assert next(stream)["choices"][0]["delta"]["content"] == "safe"
        with pytest.raises(ChatProviderError) as exc_info:
            next(stream)
    finally:
        logger.remove(sink_id)

    assert exc_info.value.provider == "qwencloud"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    disclosure = str(exc_info.value) + "".join(records)
    assert canary not in disclosure
    assert "data:" not in disclosure
    assert len(session.posts) == 1
    assert response.iter_content_calls == 1
    assert response.close_calls == 1
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


@pytest.mark.parametrize("path", ("primary-error", "normal", "explicit"))
def test_stream_cleanup_failures_never_mask_primary_or_normal_result(path: str) -> None:
    terminal = _chat_text_terminal()
    chunks = (
        [b"data: {not-json}\n\n"] if path == "primary-error" else _chat_wire([terminal])
    )
    response = _CloseFailingByteStreamResponse(chunks)
    session = _CloseFailingByteStreamSession([response])
    stream = QwenCloudStream(
        response=response,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        api_mode="chat_completions",
    )
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    try:
        if path == "primary-error":
            with pytest.raises(ChatProviderError) as exc_info:
                list(stream)
            assert exc_info.value.provider == "qwencloud"
            assert exc_info.value.__cause__ is None
            assert exc_info.value.__context__ is None
            disclosure = str(exc_info.value) + "".join(records)
            assert "RAW-RESPONSE-CLOSE-CANARY" not in disclosure
            assert "RAW-SESSION-CLOSE-CANARY" not in disclosure
        elif path == "normal":
            assert list(stream) == [terminal]
        else:
            assert stream.close() is None
            assert stream.close() is None
    finally:
        logger.remove(sink_id)

    assert response.close_calls == 1
    assert session.close_calls == 1


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


@pytest.mark.parametrize("case", ("raw-recursion", "depth-cap", "node-cap"))
def test_stream_deep_json_is_typed_redacted_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    canary = "RAW-DEEP-JSON-CANARY"
    if case == "raw-recursion":
        nested = "[" * 1200 + json.dumps(canary) + "]" * 1200
        raw = (
            '{"choices":[{"index":0,"delta":{"content":"safe"},'
            '"finish_reason":"stop"}],"private":' + nested + "}"
        )
        response = _ByteStreamResponse(
            [b"data: " + raw.encode("utf-8") + b"\n\ndata: [DONE]\n\n"]
        )
        session = _ByteStreamSession([response])
        stream = QwenCloudStream(
            response=response,  # type: ignore[arg-type]
            session=session,  # type: ignore[arg-type]
            api_mode="chat_completions",
        )
    else:
        terminal = _chat_text_terminal()
        if case == "depth-cap":
            monkeypatch.setattr(
                qwencloud_streaming, "_MAX_JSON_DEPTH", 4, raising=False
            )
            terminal["private"] = {"a": {"b": {"c": canary}}}
        else:
            monkeypatch.setattr(
                qwencloud_streaming, "_MAX_JSON_NODES", 8, raising=False
            )
            terminal["private"] = [canary] * 20
        stream, response, session = _chat_stream([terminal])

    with pytest.raises(ChatProviderError) as exc_info:
        list(stream)

    assert exc_info.value.provider == "qwencloud"
    assert canary not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_responses_deep_direct_event_is_typed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwencloud_streaming, "_MAX_JSON_DEPTH", 4, raising=False)
    event = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_deep",
            "status": "in_progress",
            "private": {"a": {"b": {"RAW-DEEP-EVENT-CANARY": True}}},
        },
    }

    with pytest.raises(ChatProviderError) as exc_info:
        QwenResponsesStreamTranslator().feed(event)

    assert exc_info.value.provider == "qwencloud"
    assert "RAW-DEEP-EVENT-CANARY" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("case", ("depth-cap", "raw-recursion"))
def test_responses_deep_function_arguments_are_typed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    if case == "depth-cap":
        monkeypatch.setattr(qwencloud_streaming, "_MAX_JSON_DEPTH", 4, raising=False)
        arguments = '{"a":{"b":{"c":{"RAW-DEEP-ARGS-CANARY":true}}}}'
    else:
        arguments = (
            '{"value":' + "[" * 1200 + '"RAW-DEEP-ARGS-CANARY"' + "]" * 1200 + "}"
        )
    translator = QwenResponsesStreamTranslator()
    translator.feed(_function_added(0, 0, "fc_deep", "call_deep", "lookup"))

    with pytest.raises(ChatProviderError) as exc_info:
        translator.feed(_arguments_done(1, 0, "fc_deep", arguments))

    assert exc_info.value.provider == "qwencloud"
    assert "RAW-DEEP-ARGS-CANARY" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
