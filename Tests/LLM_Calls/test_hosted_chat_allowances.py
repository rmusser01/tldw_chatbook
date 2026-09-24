# Tests/LLM_Calls/test_hosted_chat_allowances.py
import pytest

from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    normalize_hosted_chat_response,
)


class _Policy:
    reasoning_disposition = "ignored"

    def validate_finish(self, *, finish_reason, has_text, has_calls):
        assert finish_reason == "stop"
        return finish_reason

    def validate_reasoning_content(self, value):
        return None


def _ok_response(**extra):
    body = {
        "id": "r1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    body.update(extra)
    return body


def test_unknown_top_level_key_fails_closed_without_allowance():
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(_ok_response(service_tier="default"), finish_policy=_Policy())


def test_allowlisted_extra_key_is_ignored():
    turn = normalize_hosted_chat_response(
        _ok_response(service_tier="default"),
        finish_policy=_Policy(),
        allowed_extra_keys=frozenset({"service_tier"}),
    )
    assert turn.text == "hi"


def test_allowance_never_relaxes_required_shapes():
    bad = _ok_response()
    bad["choices"] = []  # empty choices is a required-shape violation
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            bad, finish_policy=_Policy(), allowed_extra_keys=frozenset({"service_tier"})
        )
