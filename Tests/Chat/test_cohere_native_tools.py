"""
Tests for Cohere v2 /chat migration and native tool-calls (task-267).

Task 1: `chat_with_cohere` migrates from Cohere v1 `/chat` to v2 `/chat`,
text paths only -- byte-compat pinned. v1's flat `parameter_definitions`
shape cannot express nested JSON Schema (MCP tools inexpressible) and
`tool_results` lives outside the history model with no call ids; v2 is
OpenAI-shaped end-to-end (`messages` array incl. `role:"tool"`, JSON-Schema
`tools`, `tool_calls` with ids, incremental streaming deltas). This file
pins the TEXT-ONLY request/response/streaming behavior before any tool
code lands (Tasks 2-4 extend this file).

Mirrors the mocking pattern from `Tests/Chat/test_google_native_tools.py` /
`test_anthropic_native_tools.py`: patch `requests.Session.post`, drive the
real dispatcher via `chat_api_call`, and inspect the JSON payload actually
sent (or the normalized response/stream returned).
"""

import json
import logging
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call

COHERE_V2_URL = "https://api.cohere.com/v2/chat"


def _cohere_text_response(text="ok", finish_reason="COMPLETE"):
    return {
        "id": "resp_1",
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": text}]},
        "finish_reason": finish_reason,
        "usage": {"billed_units": {"input_tokens": 1, "output_tokens": 1}},
    }


def _call_cohere(mock_post, messages, **extra):
    """Shared mock plumbing: stub a non-streaming Cohere v2 response and
    drive the real dispatch path, returning the JSON body actually posted."""
    mock_response = Mock()
    mock_response.json.return_value = _cohere_text_response()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    chat_api_call(
        "cohere",
        messages_payload=messages,
        api_key="test-key",
        model="command-a-03-2025",
        streaming=False,
        **extra,
    )
    return mock_post.call_args[1]["json"]


def _call_cohere_get_result(mock_post, response_json, messages, **extra):
    """Like `_call_cohere` but returns the normalized `chat_api_call` result
    instead of the request JSON that was sent."""
    mock_response = Mock()
    mock_response.json.return_value = response_json
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    return chat_api_call(
        "cohere",
        messages_payload=messages,
        api_key="test-key",
        model="command-a-03-2025",
        streaming=False,
        **extra,
    )


def _cohere_stream_lines(events):
    return [f"data: {json.dumps(e)}".encode() for e in events]


def _call_cohere_stream(mock_post, sse_lines, messages, **extra):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = sse_lines
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    result = chat_api_call(
        "cohere",
        messages_payload=messages,
        api_key="test-key",
        model="command-a-03-2025",
        streaming=True,
        **extra,
    )
    return list(result)


def _decode_sse_chunks(sse_lines):
    chunks = []
    for line in sse_lines:
        assert line.startswith("data: ")
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


# ---------------------------------------------------------------------------
# (a) non-streaming text-only request: v2 endpoint, messages array, no
# v1-only keys, response normalization + finish_reason mapping.
# ---------------------------------------------------------------------------

@patch("requests.Session.post")
def test_request_hits_v2_chat_endpoint(mock_post):
    _call_cohere(mock_post, [{"role": "user", "content": "hi"}])
    assert mock_post.call_args[0][0] == COHERE_V2_URL


@patch("requests.Session.post")
def test_leading_system_message_becomes_system_role_entry(mock_post):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3?"},
    ]
    sent = _call_cohere(mock_post, messages)
    assert sent["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3?"},
    ]


@patch("requests.Session.post")
def test_system_prompt_param_becomes_system_role_entry(mock_post):
    sent = _call_cohere(
        mock_post, [{"role": "user", "content": "hi"}],
        system_message="Be terse.",
    )
    assert sent["messages"][0] == {"role": "system", "content": "Be terse."}


@patch("requests.Session.post")
def test_no_v1_only_keys_in_payload(mock_post):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ]
    sent = _call_cohere(mock_post, messages)
    assert "preamble" not in sent
    assert "chat_history" not in sent
    assert "message" not in sent


@patch("requests.Session.post")
def test_non_streaming_text_response_normalizes_to_openai_shape(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_text_response("Hi there", "COMPLETE"),
        [{"role": "user", "content": "hi"}],
    )
    choice = result["choices"][0]
    assert choice["message"]["content"] == "Hi there"
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


@patch("requests.Session.post")
def test_max_tokens_finish_reason_maps_to_length(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_text_response("cut off", "MAX_TOKENS"),
        [{"role": "user", "content": "hi"}],
    )
    assert result["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# (b) streaming: content-delta -> delta.content chunks, message-end -> finish
# ---------------------------------------------------------------------------

@patch("requests.Session.post")
def test_streaming_content_delta_emits_openai_delta_content(mock_post):
    events = [
        {"type": "message-start", "delta": {"message": {"role": "assistant"}}},
        {"type": "content-start", "index": 0},
        {"type": "content-delta", "index": 0,
         "delta": {"message": {"content": {"text": "Hel"}}}},
        {"type": "content-delta", "index": 0,
         "delta": {"message": {"content": {"text": "lo"}}}},
        {"type": "content-end", "index": 0},
        {"type": "message-end", "delta": {"finish_reason": "COMPLETE"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "hi"}])
    chunks = _decode_sse_chunks(raw)
    texts = [c["choices"][0].get("delta", {}).get("content") for c in chunks
             if c["choices"][0].get("delta", {}).get("content")]
    assert texts == ["Hel", "lo"]
    finishes = [c["choices"][0].get("finish_reason") for c in chunks
                if c["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]
    assert raw[-1] == "data: [DONE]\n\n"


@patch("requests.Session.post")
def test_streaming_max_tokens_finish_reason_maps_to_length(mock_post):
    events = [
        {"type": "content-delta", "index": 0,
         "delta": {"message": {"content": {"text": "partial"}}}},
        {"type": "message-end", "delta": {"finish_reason": "MAX_TOKENS"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "hi"}])
    chunks = _decode_sse_chunks(raw)
    finishes = [c["choices"][0].get("finish_reason") for c in chunks
                if c["choices"][0].get("finish_reason")]
    assert finishes == ["length"]


# ---------------------------------------------------------------------------
# (c) params map to v2 names; num_generations (v1-only) dropped w/ debug log
# ---------------------------------------------------------------------------

@patch("requests.Session.post")
def test_params_map_to_v2_names(mock_post):
    sent = _call_cohere(
        mock_post, [{"role": "user", "content": "hi"}],
        temp=0.5, topp=0.8, topk=10, max_tokens=100,
        stop=["END"], seed=42,
        frequency_penalty=0.1, presence_penalty=0.2,
    )
    assert sent["temperature"] == 0.5
    assert sent["p"] == 0.8
    assert sent["k"] == 10
    assert sent["max_tokens"] == 100
    assert sent["stop_sequences"] == ["END"]
    assert sent["seed"] == 42
    assert sent["frequency_penalty"] == 0.1
    assert sent["presence_penalty"] == 0.2


@patch("requests.Session.post")
def test_num_generations_is_dropped_with_debug_log(mock_post):
    from loguru import logger as loguru_logger

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        sent = _call_cohere(
            mock_post, [{"role": "user", "content": "hi"}],
            n=3,
        )
    finally:
        loguru_logger.remove(sink_id)

    assert "num_generations" not in sent
    assert any("num_generations" in m for m in messages)


@patch("requests.Session.post")
def test_plain_chat_payload_unchanged(mock_post):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]
    sent = _call_cohere(mock_post, messages)

    assert "tools" not in sent
    assert sent["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]
