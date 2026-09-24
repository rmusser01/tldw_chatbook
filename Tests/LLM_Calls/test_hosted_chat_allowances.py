# Tests/LLM_Calls/test_hosted_chat_allowances.py
import json

import pytest

from tldw_chatbook.LLM_Calls import hosted_chat
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
    HostedHTTPTransportConfig,
    hosted_chat_request,
    normalize_hosted_chat_response,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord


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


def _stream_chunk_event(**extra):
    event = {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hi"},
                "finish_reason": None,
            }
        ],
    }
    event.update(extra)
    return event


_STREAM_TERMINAL_EVENT = {
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _stream_records(*events):
    return iter(
        [
            *(SSERecord(event=None, data=json.dumps(event)) for event in events),
            SSERecord(event=None, data="[DONE]"),
        ]
    )


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


def test_stream_unknown_event_key_fails_closed_without_allowance():
    stream = HostedChatStream(
        _stream_records(_stream_chunk_event(service_tier="default")),
        finish_policy=_Policy(),
    )
    with pytest.raises(HostedChatProtocolError):
        list(stream)


def test_stream_allowlisted_extra_key_is_ignored():
    events = [_stream_chunk_event(service_tier="default"), _STREAM_TERMINAL_EVENT]
    stream = HostedChatStream(
        _stream_records(*events),
        finish_policy=_Policy(),
        allowed_extra_keys=frozenset({"service_tier"}),
    )
    assert list(stream) == events
    assert stream.terminal_turn.text == "hi"
    assert stream.terminal_turn.usage == {
        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
    }


def test_stream_allowance_does_not_extend_to_choice_or_delta_keys():
    allowance = frozenset({"service_tier"})
    choice_extra = _stream_chunk_event()
    choice_extra["choices"][0]["logprobs"] = None  # unknown key inside the choice
    with pytest.raises(HostedChatProtocolError):
        list(
            HostedChatStream(
                _stream_records(choice_extra),
                finish_policy=_Policy(),
                allowed_extra_keys=allowance,
            )
        )
    delta_extra = _stream_chunk_event()
    delta_extra["choices"][0]["delta"]["refusal"] = None  # unknown key inside the delta
    with pytest.raises(HostedChatProtocolError):
        list(
            HostedChatStream(
                _stream_records(delta_extra),
                finish_policy=_Policy(),
                allowed_extra_keys=allowance,
            )
        )


def test_hosted_chat_request_forwards_stream_allowance(monkeypatch: pytest.MonkeyPatch):
    records = _stream_records(
        _stream_chunk_event(service_tier="default"), _STREAM_TERMINAL_EVENT
    )
    monkeypatch.setattr(hosted_chat, "owned_json_post", lambda **_kwargs: records)

    result = hosted_chat_request(
        config=HostedHTTPTransportConfig(
            provider="zai",
            base_url="https://example.test/v1",
            api_key="secret",
            timeout=10,
            retries=0,
            retry_delay=0,
        ),
        payload={"model": "m", "messages": [], "stream": True},
        streaming=True,
        finish_policy=_Policy(),
        allowed_extra_keys=frozenset({"service_tier"}),
    )

    assert isinstance(result, HostedChatStream)
    assert len(list(result)) == 2
    assert result.terminal_turn.text == "hi"
    assert result.terminal_turn.usage == {
        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
    }
