"""TASK-34100.2 AC#16: a real OpenRouter reply is accepted, not 'malformed'.

Every OpenRouter send failed with ChatProviderError 502 'OpenRouter returned a
malformed successful response.' The shared hosted_chat engine fails closed on
any response key it does not know, and openrouter.py passed it no allowances,
while every real OpenRouter reply carries extra fields. The bodies and stream
events below are the key shapes of real replies recorded on 2026-10-04
(content replaced; no ids or keys): meta-llama/llama-3.1-8b-instruct (the
shipped default) and openai/gpt-4o-mini answer with top-level ``provider`` and
``service_tier``, choice-level ``native_finish_reason`` and ``logprobs``, and
message-level ``refusal`` and ``reasoning``; a reasoning model
(deepseek/deepseek-r1-0528) adds ``reasoning_details``, a list of mappings.

The allowances are OpenRouter's own, not a blanket tolerance: a key outside
them still fails closed.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.Chat_Functions import chat_api_call

# These fake the transport, not the config getters, so they keep the bootstrap
# profile exactly as test_groq_openrouter_migration_characterization.py does
# (Tests/conftest.py: the hosted transport's session construction reads config
# through the participant admission, which the per-test redirect refuses).
pytestmark = pytest.mark.bootstrap_profile

_MESSAGES = [{"role": "user", "content": "Say OK"}]


def _response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = payload
    response.close = Mock()
    return response


def _stream(*events: dict) -> Mock:
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_content = Mock(
        return_value=iter([b": OPENROUTER PROCESSING\n\n", body, b"data: [DONE]\n\n"])
    )
    response.close = Mock()
    return response


def _body(message: dict, **top: object) -> dict:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": "meta-llama/llama-3.1-8b-instruct",
        "provider": "Novita",
        "system_fingerprint": None,
        "service_tier": None,
        "choices": [
            {
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 2,
            "total_tokens": 15,
            "cost": 3.564e-07,
            "is_byok": False,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "cost_details": {"upstream_inference_cost": 3.6e-07},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        **top,
    }


def _send(response: Mock, *, streaming: bool):
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = response
        result = chat_api_call(
            "openrouter",
            messages_payload=_MESSAGES,
            api_key="sk-or-test",
            model="meta-llama/llama-3.1-8b-instruct",
            streaming=streaming,
        )
        if streaming:
            return "".join(result)
        return result


def test_a_real_non_streaming_reply_is_accepted() -> None:
    message = {"role": "assistant", "content": "OK", "refusal": None, "reasoning": None}

    result = _send(_response(_body(message)), streaming=False)

    assert result["choices"][0]["message"]["content"] == "OK"


def test_a_reasoning_models_reply_with_reasoning_details_is_accepted() -> None:
    message = {
        "role": "assistant",
        "content": "OK",
        "refusal": None,
        "reasoning": "The user wants OK.",
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "The user wants OK.",
                "format": "unknown",
                "index": 0,
            }
        ],
    }

    result = _send(
        _response(_body(message, service_tier="default")), streaming=False
    )

    message = result["choices"][0]["message"]
    assert message["content"] == "OK"
    # Allowances are validated, then dropped -- never passed through.
    assert "reasoning_details" not in message


def test_a_real_streamed_reply_is_accepted() -> None:
    def event(delta: dict, finish: str | None, **top: object) -> dict:
        return {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek/deepseek-r1-0528",
            "provider": "Novita",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                    "native_finish_reason": finish,
                }
            ],
            **top,
        }

    text = _send(
        _stream(
            event(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "Thinking",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "Thinking", "index": 0}
                    ],
                },
                None,
            ),
            event({"role": "assistant", "content": "OK", "reasoning": None}, None),
            event(
                {"role": "assistant", "content": ""},
                "stop",
                service_tier=None,
                usage={"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            ),
        ),
        streaming=True,
    )

    assert "OK" in text


def test_a_key_outside_openrouters_allowances_still_fails_closed() -> None:
    message = {"role": "assistant", "content": "OK"}

    with pytest.raises(ChatProviderError, match="malformed"):
        _send(_response(_body(message, unexpected_top_level={"x": 1})), streaming=False)


def test_reasoning_details_must_be_a_list_of_mappings() -> None:
    message = {"role": "assistant", "content": "OK", "reasoning_details": ["text"]}

    with pytest.raises(ChatProviderError, match="malformed"):
        _send(_response(_body(message)), streaming=False)


# --- Recorded real replies (review round 1, F3 / F4) ------------------------
#
# The fixture above was assembled by hand and had ONE terminal chunk. Every
# real OpenRouter stream (gpt-4o-mini, llama-3.1-8b via Novita, deepseek-r1
# via DeepInfra, recorded 2026-10-05) sends the terminal choice twice: once
# plain, then again with ``usage`` (and ``service_tier``) before [DONE]. The
# engine refused the second as 'stream data followed terminal state', so
# every streamed OpenRouter reply failed with a 502. A non-streaming tool
# call carries ``index`` on each ``message.tool_calls`` item, which the
# engine refused as 'tool call is malformed'.

from copy import deepcopy
from pathlib import Path

_RECORDED = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures" / "openrouter_live_replies.json"
    ).read_text()
)


def _recorded(name: str):
    return deepcopy(_RECORDED[name])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("stream_plain_gpt_4o_mini", "OK!"),
        ("stream_plain_llama_3_1_8b", "OK"),
        ("stream_reasoning_deepseek_r1", "OK."),
    ],
)
def test_a_recorded_real_stream_is_accepted(name: str, expected: str) -> None:
    text = _send(_stream(*_recorded(name)), streaming=True)

    frames = [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(
        choice["delta"].get("content") or ""
        for frame in frames
        for choice in frame["choices"]
    )
    assert content == expected
    # The repeated terminal choice is passed on as the usage-only frame an
    # OpenAI stream ends with: one finish, then the usage.
    finishes = [
        choice["finish_reason"]
        for frame in frames
        for choice in frame["choices"]
        if choice.get("finish_reason")
    ]
    assert finishes == ["stop"]
    assert frames[-1]["choices"] == []
    assert frames[-1]["usage"]["completion_tokens"] >= 1


def test_a_recorded_real_streamed_tool_call_is_accepted() -> None:
    text = _send(_stream(*_recorded("stream_tool_gpt_4o_mini")), streaming=True)

    frames = [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    arguments = "".join(
        call["function"].get("arguments", "")
        for frame in frames
        for choice in frame["choices"]
        for call in choice["delta"].get("tool_calls") or ()
    )
    assert json.loads(arguments) == {"timezone": "UTC"}
    assert frames[-1]["choices"] == [] and frames[-1]["usage"]["total_tokens"] == 70


def test_a_recorded_real_non_streaming_tool_call_is_accepted() -> None:
    result = _send(_response(_recorded("body_tool_gpt_4o_mini")), streaming=False)

    (call,) = result["choices"][0]["message"]["tool_calls"]
    assert call["function"] == {
        "name": "get_current_time",
        "arguments": '{"timezone":"UTC"}',
    }
    # The allowance is validated, then dropped -- never passed through.
    assert "index" not in call


@pytest.mark.parametrize("index", ["0", -1, True, 1.0, None])
def test_a_tool_call_index_must_be_a_small_integer(index: object) -> None:
    body = _recorded("body_tool_gpt_4o_mini")
    body["choices"][0]["message"]["tool_calls"][0]["index"] = index

    with pytest.raises(ChatProviderError, match="malformed"):
        _send(_response(body), streaming=False)


def test_a_tool_call_key_outside_openrouters_allowances_still_fails_closed() -> None:
    body = _recorded("body_tool_gpt_4o_mini")
    body["choices"][0]["message"]["tool_calls"][0]["unexpected"] = 1

    with pytest.raises(ChatProviderError, match="malformed"):
        _send(_response(body), streaming=False)


def _trailing_variants():
    plain = _recorded("stream_plain_gpt_4o_mini")
    # The repeated terminal choice may not carry new output ...
    with_text = deepcopy(plain)
    with_text[-1]["choices"][0]["delta"]["content"] = "more"
    # ... nor change the finish reason ...
    other_finish = deepcopy(plain)
    other_finish[-1]["choices"][0]["finish_reason"] = "length"
    # ... nor come without usage ...
    no_usage = deepcopy(plain)
    del no_usage[-1]["usage"]
    # ... nor come twice.
    twice = deepcopy(plain) + [deepcopy(plain[-1])]
    # ... nor carry a tool call.
    with_call = deepcopy(plain)
    with_call[-1]["choices"][0]["delta"]["tool_calls"] = [
        {
            "index": 0,
            "id": "call_x",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }
    ]
    return {
        "new text": with_text,
        "changed finish": other_finish,
        "no usage": no_usage,
        "repeated twice": twice,
        "tool call": with_call,
    }


@pytest.mark.parametrize("variant", sorted(_trailing_variants()))
def test_only_a_bare_usage_repeat_of_the_terminal_choice_is_accepted(
    variant: str,
) -> None:
    from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError

    events = _trailing_variants()[variant]
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _send(_stream(*events), streaming=True)


def test_the_terminal_repeat_is_openrouters_allowance_not_the_engines_default() -> None:
    """Every other hosted provider still fails closed on data after the
    terminal choice: the repeat is accepted only through OpenRouter's own
    call-owned wire normalizer (dev's ADR-179 seam, which replaced this
    branch's engine-level opt-in when the branch was merged onto dev)."""
    from tldw_chatbook.LLM_Calls import openrouter
    from tldw_chatbook.LLM_Calls.hosted_chat import (
        HostedChatProtocolError,
        HostedChatStream,
    )
    from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord

    def run(*, normalized: bool) -> None:
        records = [
            SSERecord(event=None, data=json.dumps(event))
            for event in _recorded("stream_plain_llama_3_1_8b")
        ] + [SSERecord(event=None, data="[DONE]")]
        stream = HostedChatStream(
            iter(records),
            finish_policy=openrouter._FINISH_POLICY,
            allowed_extra_keys=openrouter._OPENROUTER_RESPONSE_KEYS,
            allowed_choice_keys=openrouter._OPENROUTER_CHOICE_KEYS,
            allowed_message_keys=openrouter._OPENROUTER_MESSAGE_KEYS,
            wire_normalizer=(
                openrouter._OpenRouterWireNormalizer(streaming=True)
                if normalized
                else None
            ),
        )
        list(stream)
        assert stream.terminal_turn.usage["total_tokens"] == 15

    run(normalized=True)
    with pytest.raises(HostedChatProtocolError, match="followed terminal state"):
        run(normalized=False)
