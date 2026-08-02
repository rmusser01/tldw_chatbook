"""chat_with_openai: stream_options opt-in + graceful 400 fallback +
Responses-API usage passthrough."""

import json
from unittest.mock import Mock, patch

import requests

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import _responses_stream_to_chat_sse


def _streaming_ok_response(lines):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(lines)
    response.close = Mock()
    return response


@patch("requests.Session.post")
def test_streaming_payload_includes_stream_options(mock_post):
    mock_post.return_value = _streaming_ok_response(
        ['data: {"choices": [{"delta": {"content": "hi"}}]}', "data: [DONE]"]
    )
    generator = chat_api_call(
        "openai",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    list(generator)
    sent_payload = mock_post.call_args[1]["json"]
    assert sent_payload["stream_options"] == {"include_usage": True}


@patch("requests.Session.post")
def test_non_streaming_payload_omits_stream_options(mock_post):
    ok = Mock()
    ok.status_code = 200
    ok.raise_for_status = Mock()
    ok.json.return_value = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    mock_post.return_value = ok
    chat_api_call(
        "openai",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=False,
    )
    assert "stream_options" not in mock_post.call_args[1]["json"]


@patch("requests.Session.post")
def test_400_naming_stream_options_retries_without_it(mock_post):
    bad = Mock()
    bad.status_code = 400
    bad.text = '{"error": {"message": "Unknown parameter: stream_options"}}'
    bad.raise_for_status.side_effect = requests.exceptions.HTTPError(response=bad)
    ok = _streaming_ok_response(
        ['data: {"choices": [{"delta": {"content": "hi"}}]}', "data: [DONE]"]
    )
    mock_post.side_effect = [bad, ok]

    generator = chat_api_call(
        "openai",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    chunks = list(generator)

    assert mock_post.call_count == 2
    retry_payload = mock_post.call_args_list[1][1]["json"]
    assert "stream_options" not in retry_payload
    assert any("hi" in c for c in chunks)


def test_responses_completed_event_carries_usage_through():
    lines = [
        'data: {"type": "response.output_text.delta", "delta": "hi"}',
        (
            'data: {"type": "response.completed", "response": {"usage": '
            '{"input_tokens": 1200, "output_tokens": 90, '
            '"input_tokens_details": {"cached_tokens": 1024}}}}'
        ),
    ]
    response = _streaming_ok_response(lines)
    chunks = list(_responses_stream_to_chat_sse(response, model="gpt-5-mini"))

    completed = [
        json.loads(c.removeprefix("data:").strip())
        for c in chunks
        if c.strip() not in ("data: [DONE]",) and '"usage"' in c
    ]
    assert len(completed) == 1
    assert completed[0]["usage"]["input_tokens"] == 1200
    assert completed[0]["usage"]["input_tokens_details"]["cached_tokens"] == 1024
