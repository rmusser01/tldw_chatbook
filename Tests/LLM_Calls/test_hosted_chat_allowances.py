# Tests/LLM_Calls/test_hosted_chat_allowances.py
import json
from copy import deepcopy

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
    # Qodo finding 5: tolerated extras are validated then DROPPED from the
    # visible frames (spec: drop, not passthrough). Terminal accounting is
    # unchanged.
    frames = list(stream)
    assert len(frames) == 2
    assert all("service_tier" not in frame for frame in frames)
    assert frames[0]["choices"][0]["delta"]["content"] == "hi"
    assert stream.terminal_turn.text == "hi"
    assert stream.terminal_turn.usage == {
        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
    }


def test_stream_allowance_does_not_extend_to_choice_or_delta_keys():
    # Phase 1 pin, re-scoped by Phase 2 Task 4: this is the STRICT default
    # (no level allowances, no tolerant profile). A top/event allowance can
    # never widen the choice/delta levels -- only allowed_choice_keys /
    # allowed_message_keys or tolerant_top_level_extras do, and moonshot/zai
    # (byte-identity pinned) pass neither.
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


# --- ADR-179 Phase 2 Task 4: level-keyed allowances + tolerant profile ---
#
# Level split: response_allowances stays top/event-scoped (Phase 1);
# allowed_choice_keys / allowed_message_keys subtract at the choice and
# message/delta levels; tolerant_top_level_extras is the fixture-gated
# long-tail profile (custom family only).

_CHOICE_ALLOWANCES = frozenset({"logprobs", "stop_reason"})


class _ToolCallsPolicy:
    """Strict finish policy that also accepts tool-call finishes."""

    reasoning_disposition = "ignored"

    def validate_finish(self, *, finish_reason, has_text, has_calls):
        if finish_reason == "tool_calls" and has_calls:
            return finish_reason
        if finish_reason == "stop" and has_text and not has_calls:
            return finish_reason
        raise HostedChatProtocolError("finish state is malformed")

    def validate_reasoning_content(self, value):
        return None


def _tool_response() -> dict:
    """An ollama-shaped tool body: the call object carries an extra ``index``."""
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "index": 0,  # extra key (controller ruling a)
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Tokyo"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    }


# Matrix 1: moonshot/zai byte-identity -- a top-level allowance can never
# widen the choice/message levels (their strict parse stays byte-identical
# even if a synthetic top allowance exists).
def test_top_level_allowance_does_not_widen_choice_or_message_levels():
    choice_extra = _ok_response()
    choice_extra["choices"][0]["logprobs"] = None  # unknown CHOICE key
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            choice_extra,
            finish_policy=_Policy(),
            allowed_extra_keys=frozenset({"service_tier"}),  # synthetic top allowance
        )
    message_extra = _ok_response()
    message_extra["choices"][0]["message"]["annotations"] = None  # unknown MESSAGE key
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            message_extra,
            finish_policy=_Policy(),
            allowed_extra_keys=frozenset({"service_tier"}),
        )


# Matrix 2: curated choice allowances accept the two fixture-known non-null
# shapes (logprobs object-or-null, stop_reason scalar) and drop them; without
# the allowance both fail closed (body and stream).
@pytest.mark.parametrize(
    "logprobs_value",
    [None, {"content": [{"token": "hi", "logprob": -0.25, "bytes": [104, 105]}]}],
    ids=["null", "object"],
)
def test_choice_allowances_accept_logprobs_and_stop_reason(logprobs_value):
    body = _ok_response()
    body["choices"][0]["logprobs"] = logprobs_value
    body["choices"][0]["stop_reason"] = "stop"  # scalar: the matched stop string
    turn = normalize_hosted_chat_response(
        body, finish_policy=_Policy(), allowed_choice_keys=_CHOICE_ALLOWANCES
    )
    assert turn.text == "hi"
    # Allowlisted extras are validated then dropped, never passed through.
    assert set(turn.assistant_message) == {"role", "content"}


def test_message_allowance_accepts_null_unknown_message_key():
    body = _ok_response()
    body["choices"][0]["message"]["annotations"] = None
    turn = normalize_hosted_chat_response(
        body,
        finish_policy=_Policy(),
        allowed_message_keys=frozenset({"annotations"}),
    )
    assert turn.text == "hi"
    assert set(turn.assistant_message) == {"role", "content"}


def test_choice_allowance_value_rule_rejects_list_values():
    body = _ok_response()
    body["choices"][0]["logprobs"] = [{"token": "hi"}]  # neither scalar nor mapping
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            body, finish_policy=_Policy(), allowed_choice_keys=_CHOICE_ALLOWANCES
        )


def test_unknown_choice_key_fails_closed_without_allowance_body():
    body = _ok_response()
    body["choices"][0]["logprobs"] = {"content": []}
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(body, finish_policy=_Policy())


def test_unknown_choice_key_fails_closed_without_allowance_stream():
    events = [_stream_chunk_event(), deepcopy(_STREAM_TERMINAL_EVENT)]
    events[1]["choices"][0]["stop_reason"] = "stop"
    with pytest.raises(HostedChatProtocolError):
        list(HostedChatStream(_stream_records(*events), finish_policy=_Policy()))


def test_stream_choice_allowance_accepts_null_choice_extra():
    events = [_stream_chunk_event(), deepcopy(_STREAM_TERMINAL_EVENT)]
    for event in events:
        event["choices"][0]["logprobs"] = None
    stream = HostedChatStream(
        _stream_records(*events),
        finish_policy=_Policy(),
        allowed_choice_keys=_CHOICE_ALLOWANCES,
    )
    assert len(list(stream)) == 2
    assert stream.terminal_turn.text == "hi"
    assert stream.terminal_turn.usage == {
        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
    }


def test_stream_message_allowance_accepts_null_delta_extra():
    events = [_stream_chunk_event(), deepcopy(_STREAM_TERMINAL_EVENT)]
    events[0]["choices"][0]["delta"]["refusal"] = None
    stream = HostedChatStream(
        _stream_records(*events),
        finish_policy=_Policy(),
        allowed_message_keys=frozenset({"refusal"}),
    )
    assert len(list(stream)) == 2
    assert stream.terminal_turn.text == "hi"


# Matrix 3: the tolerant profile (custom family only).
def test_tolerant_drops_shape_safe_unknown_top_level_key():
    turn = normalize_hosted_chat_response(
        _ok_response(timings={"prompt_ms": 20.5, "predicted_n": 10}),  # llama-server
        finish_policy=_Policy(),
        tolerant_top_level_extras=True,
    )
    assert turn.text == "hi"


def test_tolerant_drops_null_unknown_choice_and_message_keys():
    body = _ok_response()
    body["choices"][0]["logprobs"] = None
    body["choices"][0]["message"]["annotations"] = None
    turn = normalize_hosted_chat_response(
        body, finish_policy=_Policy(), tolerant_top_level_extras=True
    )
    assert turn.text == "hi"
    assert set(turn.assistant_message) == {"role", "content"}


def test_tolerant_fails_closed_on_non_null_unknown_choice_key():
    body = _ok_response()
    body["choices"][0]["logprobs"] = {"content": []}  # non-null, not allowlisted
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            body, finish_policy=_Policy(), tolerant_top_level_extras=True
        )


def test_tolerant_accepts_non_null_allowlisted_choice_key():
    body = _ok_response()
    body["choices"][0]["stop_reason"] = "stop"  # non-null but allowlisted
    turn = normalize_hosted_chat_response(
        body,
        finish_policy=_Policy(),
        allowed_choice_keys=_CHOICE_ALLOWANCES,
        tolerant_top_level_extras=True,
    )
    assert turn.text == "hi"


def test_strict_tool_call_extra_key_fails_closed():
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(_tool_response(), finish_policy=_ToolCallsPolicy())


def test_tolerant_tool_call_extra_keys_are_ignored():
    turn = normalize_hosted_chat_response(
        _tool_response(),
        finish_policy=_ToolCallsPolicy(),
        tolerant_top_level_extras=True,
    )
    # Controller ruling (a): id/type/function stay mandatory; extras are
    # dropped from the normalized call, never passed through.
    assert turn.tool_calls == (
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
        },
    )


def test_tolerant_stream_terminal_without_usage_is_a_usage_none_turn():
    terminal = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    stream = HostedChatStream(
        _stream_records(_stream_chunk_event(), terminal),
        finish_policy=_Policy(),
        tolerant_top_level_extras=True,
    )
    assert len(list(stream)) == 2
    assert stream.terminal_turn.text == "hi"
    assert stream.terminal_turn.usage is None


def test_strict_stream_terminal_without_usage_still_fails():
    terminal = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    stream = HostedChatStream(
        _stream_records(_stream_chunk_event(), terminal),
        finish_policy=_Policy(),
    )
    with pytest.raises(HostedChatProtocolError):
        list(stream)


def test_tolerant_usageless_terminal_still_shape_checks_present_usage():
    terminal = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": [],  # present but malformed: still fails closed
    }
    stream = HostedChatStream(
        _stream_records(_stream_chunk_event(), terminal),
        finish_policy=_Policy(),
        tolerant_top_level_extras=True,
    )
    with pytest.raises(HostedChatProtocolError):
        list(stream)


def test_hosted_chat_request_forwards_level_allowances_and_tolerance(
    monkeypatch: pytest.MonkeyPatch,
):
    events = [_stream_chunk_event(), deepcopy(_STREAM_TERMINAL_EVENT)]
    del events[1]["usage"]  # usage-less terminal: needs the tolerant profile
    for event in events:
        event["choices"][0]["logprobs"] = None
        event["choices"][0]["stop_reason"] = "stop"
    records = _stream_records(*events)
    monkeypatch.setattr(hosted_chat, "owned_json_post", lambda **_kwargs: records)

    result = hosted_chat_request(
        config=HostedHTTPTransportConfig(
            provider="custom-hosted",
            base_url="https://example.test/v1",
            api_key="",
            timeout=10,
            retries=0,
            retry_delay=0,
        ),
        payload={"model": "m", "messages": [], "stream": True},
        streaming=True,
        finish_policy=_Policy(),
        allowed_choice_keys=_CHOICE_ALLOWANCES,
        allowed_message_keys=frozenset({"refusal"}),
        tolerant_top_level_extras=True,
    )

    assert isinstance(result, HostedChatStream)
    assert len(list(result)) == 2
    assert result.terminal_turn.text == "hi"
    assert result.terminal_turn.usage is None


# Matrix 4 (Qodo finding 5): visible stream frames keep known protocol
# keys only -- tolerated/allowanced extras never reach the caller.
def test_stream_tolerant_unknown_event_key_dropped_from_visible_frame():
    chunk = _stream_chunk_event(prompt="You said hi")  # llama-server extra
    frames = list(
        HostedChatStream(
            _stream_records(chunk, _STREAM_TERMINAL_EVENT),
            finish_policy=_Policy(),
            tolerant_top_level_extras=True,
        )
    )
    assert "prompt" not in frames[0]
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant", "content": "hi"}


def test_stream_allowanced_null_choice_and_delta_extras_dropped():
    events = [_stream_chunk_event(), deepcopy(_STREAM_TERMINAL_EVENT)]
    for event in events:
        event["choices"][0]["logprobs"] = None
    events[0]["choices"][0]["delta"]["refusal"] = None
    frames = list(
        HostedChatStream(
            _stream_records(*events),
            finish_policy=_Policy(),
            allowed_choice_keys=_CHOICE_ALLOWANCES,
            allowed_message_keys=frozenset({"refusal"}),
        )
    )
    for frame in frames:
        assert set(frame["choices"][0]) <= {"index", "delta", "finish_reason"}
        assert set(frame["choices"][0]["delta"]) <= {
            "role", "content", "reasoning_content", "tool_calls"
        }


def test_stream_tolerant_tool_call_extras_normalized_in_visible_frame():
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            # ollama-style extra on the call object:
                            "extra": "dropped",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Tokyo"}',
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    terminal = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    frames = list(
        HostedChatStream(
            _stream_records(chunk, terminal),
            finish_policy=_ToolCallsPolicy(),
            tolerant_top_level_extras=True,
        )
    )
    assert frames[0]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
        }
    ]


def test_stream_known_keys_only_frames_pass_through_unchanged():
    # zai/moonshot byte-identity: they carry no allowances and no tolerance,
    # so validation admits known keys only and the filtered frame is the
    # original event, value for value.
    events = [_stream_chunk_event(), deepcopy(_STREAM_TERMINAL_EVENT)]
    frames = list(
        HostedChatStream(_stream_records(*events), finish_policy=_Policy())
    )
    assert frames == events
