# Tests/Chat/test_anthropic_streaming_usage.py
"""chat_with_anthropic streaming must surface usage as empty-choices SSE
chunks (message_start -> input/cache buckets; end of stream -> output)."""

import json
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}".encode("utf-8")


ANTHROPIC_STREAM_LINES = [
    _sse(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "usage": {
                    "input_tokens": 3571,
                    "cache_read_input_tokens": 6656,
                    "cache_creation_input_tokens": 1024,
                },
            },
        }
    ),
    _sse(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
    ),
    _sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 727},
        }
    ),
    _sse({"type": "message_stop"}),
]


def _usage_chunks(raw_chunks):
    found = []
    for raw in raw_chunks:
        body = raw.removeprefix("data:").strip()
        if not body or body == "[DONE]":
            continue
        payload = json.loads(body)
        if payload.get("usage") is not None:
            assert payload.get("choices") == [], "usage chunks carry no choices"
            found.append(payload["usage"])
    return found


@patch("requests.Session.post")
def test_streaming_emits_input_then_output_usage_chunks(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = iter(ANTHROPIC_STREAM_LINES)
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    generator = chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="claude-sonnet-4-6",
        streaming=True,
    )
    chunks = list(generator)

    usages = _usage_chunks(chunks)
    assert len(usages) == 2
    assert usages[0]["input_tokens"] == 3571
    assert usages[0]["cache_read_input_tokens"] == 6656
    assert usages[0]["cache_creation_input_tokens"] == 1024
    assert usages[1]["output_tokens"] == 727
    # Text chunks still flow, and [DONE] still terminates.
    assert any('"content": "Hello"' in c for c in chunks)
    assert chunks[-1].strip() == "data: [DONE]"


@patch("requests.Session.post")
def test_streaming_without_usage_events_emits_no_usage_chunk(mock_post):
    lines = [
        _sse(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            }
        ),
        _sse({"type": "message_stop"}),
    ]
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = iter(lines)
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    generator = chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="claude-sonnet-4-6",
        streaming=True,
    )
    assert _usage_chunks(list(generator)) == []
