"""A 400 naming cache_control retries ONCE without breakpoints; any other
error behaves exactly as before (caching must never break sends)."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import (
    _anthropic_caching_enabled,
    chat_with_anthropic,
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


def test_caching_enabled_defaults_true_and_warns_on_config_read_failure():
    """A broken config read must fail OPEN (never silently change request
    shapes) but must not fail SILENT -- the operator needs a signal that the
    kill-switch read is broken, not just that caching happened to stay on."""
    from loguru import logger as loguru_logger

    messages = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        with patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting",
            side_effect=RuntimeError("config store unavailable"),
        ):
            result = _anthropic_caching_enabled()
    finally:
        loguru_logger.remove(sink_id)

    assert result is True
    assert len(messages) == 1
    assert "caching config read failed" in messages[0]
    assert "defaulting anthropic prompt caching ON" in messages[0]
    assert "config store unavailable" in messages[0]


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


# --- TASK-26014: 1-hour cache TTL tier --------------------------------------


def test_cache_marker_defaults_to_5m_bare_ephemeral():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import _cache_control_marker

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting", return_value="5m"
    ):
        assert _cache_control_marker("claude-opus-5") == {"type": "ephemeral"}


def test_cache_marker_emits_1h_when_configured_and_supported():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import _cache_control_marker

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting", return_value="1h"
    ):
        assert _cache_control_marker("claude-opus-5") == {
            "type": "ephemeral",
            "ttl": "1h",
        }


def test_cache_marker_falls_back_to_5m_on_unsupported_or_junk():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import _cache_control_marker

    for value in ("banana", "", None, "2h"):
        with patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting",
            return_value=value,
        ):
            assert _cache_control_marker("claude-opus-5") == {"type": "ephemeral"}


@patch("tldw_chatbook.LLM_Calls.LLM_API_Calls.create_default_session")
def test_1h_ttl_flows_into_the_payload_and_adds_the_beta_header(mock_session):
    posted = {}

    def _post(url, headers=None, json=None, **kwargs):
        posted["headers"] = headers
        posted["json"] = json
        return _ok_response()

    session = Mock()
    session.post.side_effect = _post
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    mock_session.return_value = session

    def _setting(section, key, default=None):
        if key == "cache_ttl":
            return "1h"
        return default

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting", side_effect=_setting
    ):
        chat_with_anthropic(
            input_data=[{"role": "user", "content": "hi"}],
            api_key="k",
            model="claude-opus-5",
            system_prompt="stable prefix",
            prompt_caching=True,
            streaming=False,
        )

    system_block = posted["json"]["system"][0]
    assert system_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    beta = posted["headers"].get("anthropic-beta", "")
    assert "extended-cache-ttl" in beta, "1h markers require the beta opt-in header"


@patch("tldw_chatbook.LLM_Calls.LLM_API_Calls.create_default_session")
def test_5m_default_adds_no_beta_header(mock_session):
    posted = {}

    def _post(url, headers=None, json=None, **kwargs):
        posted["headers"] = headers
        return _ok_response()

    session = Mock()
    session.post.side_effect = _post
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    mock_session.return_value = session

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting",
        side_effect=lambda s, k, d=None: d,
    ):
        chat_with_anthropic(
            input_data=[{"role": "user", "content": "hi"}],
            api_key="k",
            model="claude-opus-5",
            system_prompt="stable prefix",
            prompt_caching=True,
            streaming=False,
        )

    assert "anthropic-beta" not in posted["headers"]
