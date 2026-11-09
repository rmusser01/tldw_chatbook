"""Replay the indexed non-streaming tool call captured by TASK-39147."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from Tests.LLM_Calls.test_groq_openrouter_migration_characterization import (
    _nonstreaming_ok_response,
)
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.deepseek import DeepSeekFinishPolicy
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    normalize_hosted_chat_response,
)
from tldw_chatbook.Provider_Contract import ChatProviderError

pytestmark = pytest.mark.bootstrap_profile
FIXTURE = Path(__file__).with_name("fixtures") / "deepseek_indexed_tool_call.json"


def call_deepseek(body):
    response = _nonstreaming_ok_response(body)
    with patch("requests.Session.post", return_value=response) as post:
        result = chat_api_call(
            "deepseek",
            messages_payload=[{"role": "user", "content": "Review"}],
            api_key="fixture-key",
            model="deepseek-flash",
            streaming=False,
        )
    post.assert_called_once()
    response.close.assert_called_once()
    return result


@pytest.mark.parametrize("count", [1, 2])
def test_captured_indexed_calls_preserve_arguments_identity_order_and_usage(count):
    body = json.loads(FIXTURE.read_text())
    calls = body["choices"][0]["message"]["tool_calls"]
    if count == 2:
        calls.append({**deepcopy(calls[0]), "id": "call-second", "index": 1})
    original = deepcopy(body)
    result = call_deepseek(body)
    expected = [
        {key: value for key, value in call.items() if key != "index"} for call in calls
    ]
    assert result["choices"][0]["message"]["tool_calls"] == expected
    assert result.terminal_turn.tool_calls == tuple(expected)
    assert result["usage"] == original["usage"]
    assert body == original


@pytest.mark.parametrize(
    "change", ["extra", "arguments", "duplicate_id", "missing_id", "index_value"]
)
def test_index_allowance_keeps_tool_validation(change):
    body = json.loads(FIXTURE.read_text())
    calls = body["choices"][0]["message"]["tool_calls"]
    if change == "extra":
        calls[0]["undocumented"] = "refused"
    elif change == "arguments":
        calls[0]["function"]["arguments"] = '{"review_id":'
    elif change == "duplicate_id":
        calls.append({**deepcopy(calls[0]), "index": 1})
    elif change == "missing_id":
        del calls[0]["id"]
    else:
        calls[0]["index"] = ["not-an-allowed-metadata-value"]
    with pytest.raises(ChatProviderError):
        call_deepseek(body)


def test_tool_index_allowance_does_not_change_shared_default():
    with pytest.raises(HostedChatProtocolError, match="tool call is malformed"):
        normalize_hosted_chat_response(
            json.loads(FIXTURE.read_text()),
            finish_policy=DeepSeekFinishPolicy(),
            allowed_choice_keys=frozenset({"logprobs"}),
        )
