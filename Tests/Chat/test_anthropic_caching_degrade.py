"""A 400 naming cache_control retries ONCE without breakpoints; any other
error behaves exactly as before (caching must never break sends)."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import (
    _contains_cache_control,
    _without_cache_control,
)


def _ok_response(text="ok"):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    return response


def _bad_response(body):
    response = Mock()
    response.status_code = 400
    response.text = body
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=response
    )
    return response


@patch("requests.Session.post")
def test_400_naming_cache_control_retries_stripped(mock_post):
    bad = _bad_response('{"error": {"message": "cache_control is not supported"}}')
    mock_post.side_effect = [bad, _ok_response()]

    chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="claude-sonnet-4-6",
        system_message="be terse",
        streaming=False,
    )

    assert mock_post.call_count == 2
    # The retry is the SAME payload with every cache_control key stripped --
    # the system block array survives as a block array, it just loses its
    # cache key (`_without_cache_control` removes only that one key).
    retry_body = mock_post.call_args_list[1][1]["json"]
    assert "cache_control" not in json.dumps(retry_body)
    assert retry_body["system"] == [{"type": "text", "text": "be terse"}]
    # ...and the FIRST attempt did carry a breakpoint, so the retry is a real
    # degrade rather than a payload that never had one.
    first_body = mock_post.call_args_list[0][1]["json"]
    assert "cache_control" in json.dumps(first_body)
    assert retry_body["messages"] == _without_cache_control(first_body["messages"])


@patch("requests.Session.post")
def test_degrade_is_visible_to_metrics(mock_post):
    """A silent degrade is a silent cost regression: every send would pay the
    cache-write premium and never read. Emit a counter so it is observable."""
    bad = _bad_response('{"error": {"message": "cache_control is not supported"}}')
    mock_post.side_effect = [bad, _ok_response()]

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.log_counter"
    ) as mock_log_counter:
        chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            model="claude-sonnet-4-6",
            system_message="be terse",
            streaming=False,
        )

    counters = [call.args[0] for call in mock_log_counter.call_args_list if call.args]
    assert "anthropic_cache_control_degrade" in counters


@patch("requests.Session.post")
def test_400_not_naming_cache_control_raises_unretried(mock_post):
    bad = _bad_response('{"error": {"message": "max_tokens too large"}}')
    mock_post.return_value = bad

    with pytest.raises(Exception):
        chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            model="claude-sonnet-4-6",
            streaming=False,
        )
    assert mock_post.call_count == 1


@patch("requests.Session.post")
def test_no_retry_when_payload_has_no_cache_control(mock_post):
    """Non-caching model: even a cache_control-naming 400 must not retry
    (nothing to strip -- the guard requires the param in OUR payload)."""
    bad = _bad_response('{"error": {"message": "cache_control invalid"}}')
    mock_post.return_value = bad

    with pytest.raises(Exception):
        chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            model="claude-2.1",
            streaming=False,
        )
    assert mock_post.call_count == 1


def test_without_cache_control_strips_recursively():
    data = {
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
        "tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}],
        "max_tokens": 5,
    }
    stripped = _without_cache_control(data)
    assert "cache_control" not in json.dumps(stripped)
    assert stripped["messages"][0]["content"][0]["text"] == "hi"
    assert stripped["max_tokens"] == 5
    assert _contains_cache_control(data) is True
    assert _contains_cache_control(stripped) is False
