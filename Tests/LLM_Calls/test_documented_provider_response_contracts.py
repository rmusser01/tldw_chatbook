"""Primary-schema actual-adapter regressions (TASK-34367.1-.5).

Only HTTP is replaced. These are offline contract fixtures, not paid captures.
The primary references and exact reconciliation scope are archived in
Docs/superpowers/qa/2026-10-04-provider-response-audit/audit.md.
"""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from Tests.LLM_Calls.test_groq_openrouter_migration_characterization import (
    _nonstreaming_ok_response,
    _sse_stream_response,
)
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError

pytestmark = pytest.mark.bootstrap_profile

MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "openai/gpt-4o",
    "together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "fireworks": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "cerebras": "llama3.1-8b",
    "moonshot": "moonshot-v1-8k",
    "mistral": "mistral-small-latest",
    "zai": "glm-4.5",
}
PROVIDERS = tuple(MODELS)[:5]
USAGE = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def _body(provider: str) -> dict:
    return {
        "id": "fixture",
        "created": 1,
        "model": MODELS[provider],
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
            }
        ],
        "usage": deepcopy(USAGE),
    }


def _stream_events(body: dict) -> list[dict]:
    common = {k: deepcopy(v) for k, v in body.items() if k not in {"choices", "usage"}}
    common["object"] = "chat.completion.chunk"
    choice = deepcopy(body["choices"][0])
    message = choice.pop("message")
    return [
        {**common, "choices": [{**choice, "finish_reason": None, "delta": message}]},
        {
            **common,
            "choices": [{**choice, "delta": {}}],
            "usage": deepcopy(body["usage"]),
        },
    ]


def _call(provider: str, body: dict | list[dict], streaming: bool):
    if streaming:
        wire = "".join("data: " + json.dumps(event) + "\n\n" for event in body)
        response = _sse_stream_response((wire + "data: [DONE]\n\n").encode())
    else:
        response = _nonstreaming_ok_response(body)
    with patch("requests.Session.post", return_value=response):
        result = chat_api_call(
            provider,
            messages_payload=[{"role": "user", "content": "hello"}],
            api_key="fixture-key",
            model=MODELS[provider],
            streaming=streaming,
        )
        if streaming:
            events = []
            for item in result:
                if isinstance(item, dict):
                    events.append(item)
                elif "[DONE]" not in item:
                    events.append(json.loads(item[6:]))
            text = "".join(
                choice.get("delta", {}).get("content") or ""
                for event in events
                for choice in event.get("choices", [])
            )
            return result, text, events
        return result, result["choices"][0]["message"]["content"], []


def _assert_answer_usage(provider: str, body: dict | list[dict], streaming: bool):
    result, text, events = _call(provider, body, streaming)
    assert text == "hi"
    if streaming:
        assert any(event.get("usage") == USAGE for event in events)
        assert result.terminal_turn.usage == USAGE
    else:
        assert result["usage"] == USAGE
    return result, events


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
def test_documented_provider_response_reaches_real_adapter(provider, streaming):
    """Missing scoped allowances previously rejected all ten replies."""
    body = _body(provider)
    body["choices"][0]["logprobs"] = None
    if provider == "groq":
        body["x_groq"] = {"id": "fixture"}
    elif provider == "openrouter":
        body["choices"][0]["native_finish_reason"] = "stop"
    elif provider == "cerebras":
        body["time_info"] = {"total_time": 0.1}
    _assert_answer_usage(
        provider, _stream_events(body) if streaming else body, streaming
    )


@pytest.mark.parametrize("provider", ["moonshot", "mistral", "zai"])
@pytest.mark.parametrize("streaming", [False, True])
def test_existing_provider_baseline_contract_is_preserved(provider, streaming):
    body = _body(provider)
    _assert_answer_usage(
        provider, _stream_events(body) if streaming else body, streaming
    )


@pytest.mark.parametrize("placement", ["finish", "trailing"])
@pytest.mark.parametrize("duplicate_top", [False, True])
def test_groq_nested_stream_usage_reaches_terminal_accounting(placement, duplicate_top):
    events = _stream_events(_body("groq"))
    terminal = events[-1]
    usage = terminal.pop("usage")
    if placement == "trailing":
        terminal = {"choices": []}
        events.append(terminal)
    terminal["x_groq"] = {"id": "fixture", "usage": usage}
    if duplicate_top:
        terminal["usage"] = deepcopy(USAGE)
    _assert_answer_usage("groq", events, True)


def test_groq_complete_hardware_cache_usage_does_not_replace_accounting():
    body = _body("groq")
    body["x_groq"] = {
        "id": "fixture",
        "usage": {"dram_cached_tokens": 2, "sram_cached_tokens": 1},
    }
    _assert_answer_usage("groq", body, False)


@pytest.mark.parametrize("streaming", [False, True])
def test_groq_logprobs_objects_are_scoped_annotations(streaming):
    body = _body("groq")
    body["choices"][0]["logprobs"] = {
        "content": [
            {"token": "hi", "logprob": -0.1, "bytes": [104, 105], "top_logprobs": []}
        ]
    }
    _assert_answer_usage("groq", _stream_events(body) if streaming else body, streaming)


@pytest.mark.parametrize(
    "error", ["private error body", "", {"message": "private error body"}]
)
def test_groq_nested_stream_error_never_yields_a_success_sentinel(error):
    events = _stream_events(_body("groq"))
    events[-1]["x_groq"] = {"error": error}
    with pytest.raises((ChatProviderError, HostedChatProtocolError)) as raised:
        _call("groq", events, True)
    assert "private error body" not in str(raised.value)


@pytest.mark.parametrize(
    "bad_usage",
    [{**USAGE, "total_tokens": 99}, {**USAGE, "prompt_tokens": True}, "bad"],
)
def test_groq_conflicting_or_malformed_stream_usage_fails_closed(bad_usage):
    events = _stream_events(_body("groq"))
    events[-1]["x_groq"] = {"usage": bad_usage}
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call("groq", events, True)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "fault",
    ["envelope", "choice", "message", "index", "role", "content", "usage", "finish"],
)
def test_documented_providers_reject_unknown_and_malformed_required_data(
    provider, streaming, fault
):
    body = _body(provider)
    choice = body["choices"][0]
    message = choice["message"]
    if fault == "envelope":
        body["unexpected_field"] = None
    elif fault == "choice":
        choice["unexpected_field"] = None
    elif fault == "message":
        message["unexpected_field"] = None
    elif fault == "index":
        choice["index"] = True
    elif fault == "role":
        message["role"] = "user"
    elif fault == "content":
        message["content"] = {"private": "wrong"}
    elif fault == "usage":
        body["usage"] = "wrong"
    elif fault == "finish":
        choice["finish_reason"] = "made_up"
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call(provider, _stream_events(body) if streaming else body, streaming)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "metadata", ["bad", {"id": 1}, {"seed": True}, {"unknown": None}]
)
def test_groq_metadata_consumed_by_normalizer_stays_closed(streaming, metadata):
    body = _body("groq")
    body["x_groq"] = metadata
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call("groq", _stream_events(body) if streaming else body, streaming)


@pytest.mark.parametrize("streaming", [False, True])
def test_openrouter_documented_response_annotations_preserve_reply_and_usage(streaming):
    body = _body("openrouter")
    body.update(
        {
            "provider": "OpenAI",
            "service_tier": "default",
            "openrouter_metadata": {
                "attempt": 1,
                "strategy": "direct",
                "endpoints": {
                    "total": 1,
                    "available": [{"provider": "OpenAI", "selected": True}],
                },
            },
        }
    )
    body["choices"][0].update(
        {
            "native_finish_reason": "stop",
            "logprobs": {
                "content": [{"token": "hi", "logprob": -0.1, "top_logprobs": []}]
            },
        }
    )
    body["choices"][0]["message"]["reasoning"] = "private reasoning"
    result, events = _assert_answer_usage(
        "openrouter", _stream_events(body) if streaming else body, streaming
    )
    assert "private reasoning" not in json.dumps(events if streaming else result)


@pytest.mark.parametrize(
    "delta", [{}, {"content": None}, {"content": "", "role": "assistant"}]
)
def test_openrouter_repeated_terminal_usage_choice_is_normalized_once(delta):
    events = _stream_events(_body("openrouter"))
    events[-1].pop("usage")
    events.append(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "stop",
                    "native_finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": deepcopy(USAGE),
        }
    )
    _assert_answer_usage("openrouter", events, True)


@pytest.mark.parametrize(
    "fault",
    [
        "finish",
        "index",
        "native",
        "role",
        "content",
        "tool_calls",
        "reasoning",
        "unknown_choice",
        "unknown_delta",
        "duplicate_usage",
    ],
)
def test_openrouter_repeated_terminal_usage_remains_strict(fault):
    body = _body("openrouter")
    body["choices"][0]["native_finish_reason"] = "stop"
    events = _stream_events(body)
    events[-1].pop("usage")
    final = {
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
                "native_finish_reason": "stop",
            }
        ],
        "usage": deepcopy(USAGE),
    }
    choice = final["choices"][0]
    if fault == "finish":
        choice["finish_reason"] = "length"
    elif fault == "index":
        choice["index"] = True
    elif fault == "native":
        choice["native_finish_reason"] = "length"
    elif fault == "role":
        choice["delta"]["role"] = "user"
    elif fault in {"content", "tool_calls", "reasoning"}:
        choice["delta"][fault] = "extra output"
    elif fault == "unknown_choice":
        choice["unknown"] = None
    elif fault == "unknown_delta":
        choice["delta"]["unknown"] = None
    events.append(final)
    if fault == "duplicate_usage":
        events.append(deepcopy(final))
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call("openrouter", events, True)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("location", ["envelope", "choice"])
def test_provider_error_frames_never_look_successful(provider, location):
    events = _stream_events(_body(provider))
    if location == "envelope":
        events[-1]["error"] = {"message": "private upstream failure", "code": 500}
    else:
        events[-1]["choices"][0]["error"] = {
            "message": "private upstream failure",
            "code": 500,
        }
    with pytest.raises((ChatProviderError, HostedChatProtocolError)) as raised:
        _call(provider, events, True)
    assert "private upstream failure" not in str(raised.value)


@pytest.mark.parametrize("provider", ["together", "fireworks", "cerebras"])
@pytest.mark.parametrize("streaming", [False, True])
def test_generic_inference_documented_annotations_survive_actual_handler(
    provider, streaming
):
    body = _body(provider)
    choice = body["choices"][0]
    choice["logprobs"] = {
        "tokens": ["hi"],
        "token_logprobs": [-0.1],
        "top_logprobs": {},
    }
    if provider == "together":
        choice.update({"seed": 42, "top_logprobs": {}, "text": "annotation"})
        choice["message"]["reasoning"] = "private annotation"
        body.update(
            {
                "prompt": [{"text": "hello", "logprobs": None}],
                "warnings": [{"message": "fixture warning"}],
            }
        )
    elif provider == "fireworks":
        choice["raw_output"] = {"completion": "hi", "completion_token_ids": [1]}
        body["perf_metrics"] = {"prompt-tokens": 3, "server-processing-time": 0.1}
        body["prompt_token_ids"] = [1, 2, 3]
    elif provider == "cerebras":
        choice["reasoning_logprobs"] = {"content": []}
        choice["message"]["reasoning"] = "private annotation"
        body.update(
            {
                "time_info": {
                    "queue_time": 0,
                    "prompt_time": 0.01,
                    "completion_time": 0.01,
                    "total_time": 0.02,
                },
                "service_tier": "auto",
                "service_tier_used": "default",
            }
        )
    events = _stream_events(body) if streaming else None
    if events and provider == "fireworks":
        events[0].pop("perf_metrics")  # documented streaming metrics are terminal-only
    result, normalized = _assert_answer_usage(
        provider, events if streaming else body, streaming
    )
    assert "private annotation" not in json.dumps(normalized if streaming else result)


@pytest.mark.parametrize(
    "provider,field",
    [
        ("openrouter", "native_finish_reason"),
        ("together", "seed"),
        ("fireworks", "raw_output"),
        ("cerebras", "reasoning_logprobs"),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_provider_allowance_is_not_global(provider, field, streaming):
    body = _body("moonshot")
    body["choices"][0][field] = None
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call("moonshot", _stream_events(body) if streaming else body, streaming)


@pytest.mark.parametrize("placement", ["finish", "trailing"])
def test_groq_nested_usage_details_and_timings_are_retained(placement):
    events = _stream_events(_body("groq"))
    terminal = events[-1]
    terminal.pop("usage")
    usage = {
        **USAGE,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
        "queue_time": 0.001,
        "prompt_time": 0.002,
        "completion_time": 0.003,
        "total_time": 0.005,
    }
    if placement == "trailing":
        terminal = {"choices": []}
        events.append(terminal)
    terminal["x_groq"] = {"id": "fixture", "seed": 42, "usage": usage}
    result, text, normalized = _call("groq", events, True)
    assert text == "hi"
    assert result.terminal_turn.usage == usage
    assert any(event.get("usage") == usage for event in normalized)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "field,bad",
    [
        ("native_finish_reason", 42),
        ("native_finish_reason", []),
        ("logprobs", 42),
        ("logprobs", []),
        ("logprobs", "bad"),
    ],
)
def test_openrouter_native_finish_and_logprobs_shapes_are_validated(
    streaming, field, bad
):
    # Native finish is an upstream string, so arbitrary strings remain valid.
    body = _body("openrouter")
    body["choices"][0][field] = bad
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call("openrouter", _stream_events(body) if streaming else body, streaming)


@pytest.mark.parametrize(
    "details",
    [
        "bad",
        {"reasoning_tokens": True},
        {"reasoning_tokens": -1},
        {"reasoning_tokens": 1, "unknown": 1},
    ],
)
def test_groq_nested_usage_details_fail_closed(details):
    events = _stream_events(_body("groq"))
    events[-1].pop("usage")
    events[-1]["x_groq"] = {"usage": {**USAGE, "completion_tokens_details": details}}
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call("groq", events, True)


def test_groq_documented_stream_obfuscation_is_a_scoped_annotation():
    events = _stream_events(_body("groq"))
    for event in events:
        event["obfuscation"] = "random-padding"
    _assert_answer_usage("groq", events, True)


@pytest.mark.parametrize("metadata", [{}, {"id": None}])
def test_groq_complete_metadata_requires_its_documented_request_id(metadata):
    body = _body("groq")
    body["x_groq"] = metadata
    with pytest.raises(ChatProviderError):
        _call("groq", body, False)


@pytest.mark.parametrize("bad", [42, [], {}])
def test_groq_stream_obfuscation_rejects_malformed_values(bad):
    events = _stream_events(_body("groq"))
    events[0]["obfuscation"] = bad
    with pytest.raises((ChatProviderError, HostedChatProtocolError)):
        _call("groq", events, True)
