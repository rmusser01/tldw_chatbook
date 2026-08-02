"""Cache-prefix stability pins (cost-ticker PR2).

Anthropic caching is a byte-exact prefix match over tools -> system ->
messages. These tests pin that consecutive turn builds keep the shared
prefix identical: same system bytes, same tool bytes, and message history
content-identical except (a) the appended turn and (b) the per-turn
cache_control marker, which MOVES to the newest message each build
(metadata designating the cache boundary -- earlier content bytes stay
identical, which is what the server matches on).
"""

import json
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import _without_cache_control


def _sent_body(mock_post, messages):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    mock_post.return_value = mock_response
    chat_api_call(
        "anthropic",
        messages_payload=messages,
        api_key="test-key",
        model="claude-sonnet-4-6",
        system_message="You are terse.\n\nAlways answer in one line.",
        streaming=False,
    )
    return mock_post.call_args[1]["json"]


TURN_1 = [{"role": "user", "content": "first question"}]
TURN_2 = TURN_1 + [
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
]


@patch("requests.Session.post")
def test_system_bytes_identical_across_consecutive_builds(mock_post):
    body_1 = _sent_body(mock_post, TURN_1)
    body_2 = _sent_body(mock_post, TURN_2)
    assert json.dumps(body_1["system"], sort_keys=True) == json.dumps(
        body_2["system"], sort_keys=True
    )


@patch("requests.Session.post")
def test_history_prefix_content_identical_across_builds(mock_post):
    """Build 2's earlier messages == build 1's messages, modulo the moved
    per-turn marker (strip cache_control from both sides before comparing)."""
    body_1 = _sent_body(mock_post, TURN_1)
    body_2 = _sent_body(mock_post, TURN_2)
    prefix_2 = _without_cache_control(body_2["messages"][: len(body_1["messages"])])
    stripped_1 = _without_cache_control(body_1["messages"])
    assert prefix_2 == stripped_1


@patch("requests.Session.post")
def test_marker_sits_only_on_newest_message_each_build(mock_post):
    body_2 = _sent_body(mock_post, TURN_2)
    dumped_history = json.dumps(body_2["messages"][:-1])
    assert "cache_control" not in dumped_history
    assert body_2["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }


@patch("requests.Session.post")
def test_no_volatile_keys_reach_the_wire(mock_post):
    """No timestamps/uuids/internal annotations in the request body."""
    body = _sent_body(mock_post, TURN_2)
    dumped = json.dumps(body)
    for forbidden in ("_native_message_id", "timestamp", "uuid"):
        assert forbidden not in dumped
