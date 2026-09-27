"""Engine finish policy and response/stream wrappers (ADR-179).

Ports the ``ZAIFinishPolicy``/``normalize_zai_response``/``ZAIStream``/
``ZAIResponse`` contracts from ``Tests/LLM_Calls/test_zai.py`` onto the
generic engine parameterized by a ``provider_registry.ProviderRecord``
preset. Engine divergences from the zai template pinned here:

- Provider-terminal finish reasons are record data (``finish_provider_errors``),
  not a hardcoded Z.ai trio; they raise ``ChatProviderError`` with the
  record's key/display-name identity.
- Reasoning disposition is record data: ``"ignored"`` drops reasoning
  everywhere (policy returns None, terminal turn carries none), while
  ``"displayable"``/``"proprietary"`` validate and retain it. Visible stream
  deltas strip ``reasoning_content`` only for non-displayable dispositions
  (zai strips unconditionally: its disposition is proprietary).
- ``record.response_allowances`` feed the shared hosted boundary's tolerated
  extra top-level response/stream keys.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedPresetFinishPolicy,
    HostedProviderResponse,
    HostedProviderStream,
    normalize_hosted_provider_response,
)
from tldw_chatbook.provider_registry import DATABRICKS

# Synthetic presets for record-gated behavior Databricks Phase 1 does not
# exercise (same technique as the Task 5 payload suite).
_DISPLAYABLE = replace(DATABRICKS, reasoning_disposition="displayable")
_PROPRIETARY = replace(DATABRICKS, reasoning_disposition="proprietary")
_PROVIDER_ERRORS = replace(
    DATABRICKS,
    finish_provider_errors=frozenset(
        {"sensitive", "model_context_window_exceeded", "network_error"}
    ),
)
# Phase 2 Task 4 synthetic custom profile: the exact shape Task 6's real
# CUSTOM_HOSTED record will ship (tolerant + the two fixture-known non-null
# choice allowances). Pins the future record via the synthetic only.
_TOLERANT = replace(
    DATABRICKS,
    tolerant_response_extras=True,
    choice_allowances=frozenset({"logprobs", "stop_reason"}),
)


# --- finish policy (brief's tests + ported zai policy tests) ---


def test_allowed_terminals_pass_and_inconsistent_state_fails():
    policy = HostedPresetFinishPolicy(DATABRICKS)
    assert policy.validate_finish(finish_reason="stop", has_text=True, has_calls=False) == "stop"
    assert policy.validate_finish(finish_reason="tool_calls", has_text=False, has_calls=True) == "tool_calls"
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(finish_reason="tool_calls", has_text=True, has_calls=False)
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(finish_reason="content_filter", has_text=True, has_calls=False)


def test_reasoning_disposition_ignored_drops_reasoning():
    policy = HostedPresetFinishPolicy(DATABRICKS)
    assert DATABRICKS.reasoning_disposition == "ignored"
    assert policy.validate_reasoning_content("thinking...") is None


def test_policy_disposition_comes_from_the_record():
    assert HostedPresetFinishPolicy(DATABRICKS).reasoning_disposition == "ignored"
    assert HostedPresetFinishPolicy(_DISPLAYABLE).reasoning_disposition == "displayable"
    assert HostedPresetFinishPolicy(_PROPRIETARY).reasoning_disposition == "proprietary"


@pytest.mark.parametrize(
    "finish_reason", ["sensitive", "model_context_window_exceeded", "network_error"]
)
def test_provider_error_finishes_are_safe_provider_errors(finish_reason: str):
    with pytest.raises(ChatProviderError) as exc_info:
        HostedPresetFinishPolicy(_PROVIDER_ERRORS).validate_finish(
            finish_reason=finish_reason,
            has_text=False,
            has_calls=False,
        )
    assert exc_info.value.provider == "databricks"
    assert finish_reason not in str(exc_info.value)


@pytest.mark.parametrize("disposition", ["displayable", "proprietary"])
def test_kept_dispositions_validate_string_or_none(disposition: str):
    record = replace(DATABRICKS, reasoning_disposition=disposition)
    policy = HostedPresetFinishPolicy(record)
    assert policy.validate_reasoning_content(None) is None
    assert policy.validate_reasoning_content("PRIVATE-REASONING") == "PRIVATE-REASONING"
    with pytest.raises(HostedChatProtocolError):
        policy.validate_reasoning_content(7)


# --- response normalization (ported normalize_zai_response tests) ---


def _tool_call_response(arguments: object) -> dict[str, object]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Working.",
                    "reasoning_content": "PRIVATE",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


def test_response_normalizes_object_arguments_deterministically():
    response = _tool_call_response({"z": 1, "a": "2+2"})

    turn = normalize_hosted_provider_response(_DISPLAYABLE, response)

    assert turn.tool_calls[0]["function"]["arguments"] == '{"a":"2+2","z":1}'
    assert turn.reasoning_content == "PRIVATE"
    assert response["choices"][0]["message"]["tool_calls"][0]["function"][  # type: ignore[index]
        "arguments"
    ] == {"z": 1, "a": "2+2"}


def test_response_drops_reasoning_when_disposition_is_ignored():
    turn = normalize_hosted_provider_response(DATABRICKS, _tool_call_response('{"a":"2+2"}'))

    assert turn.text == "Working."
    assert turn.reasoning_content is None


@pytest.mark.parametrize("arguments", [7, True, [], None])
def test_response_rejects_non_object_non_string_arguments(
    arguments: object,
):
    with pytest.raises(ChatProviderError):
        normalize_hosted_provider_response(DATABRICKS, _tool_call_response(arguments))


def test_response_protocol_errors_map_to_display_name_provider_error():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Answer"},
                "finish_reason": "content_filter",
            }
        ]
    }
    with pytest.raises(ChatProviderError) as exc_info:
        normalize_hosted_provider_response(DATABRICKS, response)

    assert exc_info.value.provider == "databricks"
    assert exc_info.value.status_code == 502
    assert "Databricks" in str(exc_info.value)
    assert "content_filter" not in str(exc_info.value)


def test_response_allowances_flow_through_the_shared_boundary():
    response = {
        "id": "resp_1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "databricks_field": {"quota": "ok"},
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    with pytest.raises(ChatProviderError):
        normalize_hosted_provider_response(DATABRICKS, response)

    allowed = replace(DATABRICKS, response_allowances=frozenset({"databricks_field"}))
    turn = normalize_hosted_provider_response(allowed, response)
    assert turn.text == "Answer"
    assert turn.finish_reason == "stop"


# --- Phase 2 Task 4: tolerant finish branch + level-keyed plumb-through ---


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_tolerant_finish_policy_accepts_empty_text_without_calls(finish_reason):
    """Legacy empty-reply behavior for the custom family: stop/length with
    no text and no calls is an empty-text turn, not a protocol failure."""
    policy = HostedPresetFinishPolicy(_TOLERANT)
    assert (
        policy.validate_finish(
            finish_reason=finish_reason, has_text=False, has_calls=False
        )
        == finish_reason
    )


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_tolerant_finish_policy_still_fails_calls_conflicts(finish_reason):
    policy = HostedPresetFinishPolicy(_TOLERANT)
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(
            finish_reason=finish_reason, has_text=False, has_calls=True
        )


def test_tolerant_finish_policy_still_fails_tool_calls_without_calls():
    policy = HostedPresetFinishPolicy(_TOLERANT)
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(
            finish_reason="tool_calls", has_text=True, has_calls=False
        )


def test_tolerant_finish_policy_still_fails_non_terminal_reasons():
    policy = HostedPresetFinishPolicy(_TOLERANT)
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(
            finish_reason="content_filter", has_text=False, has_calls=False
        )


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_strict_finish_policy_still_rejects_empty_text(finish_reason):
    """Default (non-tolerant) records keep the Phase 1 byte-identical
    behavior: an empty-text stop/length finish is a protocol failure."""
    with pytest.raises(HostedChatProtocolError):
        HostedPresetFinishPolicy(DATABRICKS).validate_finish(
            finish_reason=finish_reason, has_text=False, has_calls=False
        )


def test_engine_response_plumbs_tolerant_profile_and_level_allowances():
    response = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Ok."},
                "finish_reason": "stop",
                "logprobs": None,
                "stop_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        "timings": {"prompt_ms": 20.5},  # llama-server top-level extra
    }

    with pytest.raises(ChatProviderError):
        normalize_hosted_provider_response(DATABRICKS, response)

    turn = normalize_hosted_provider_response(_TOLERANT, response)
    assert turn.text == "Ok."
    assert turn.finish_reason == "stop"
    assert set(turn.assistant_message) == {"role", "content"}


def test_engine_tolerant_response_accepts_legacy_empty_text_stop():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ]
    }
    turn = normalize_hosted_provider_response(_TOLERANT, response)
    assert turn.text == ""
    assert turn.finish_reason == "stop"


# --- stream wrapper (ported ZAIStream visible-chunk tests) ---


def _engine_stream(record):
    payloads = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "PRIVATE"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Evidence."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4}},
    ]
    hosted = HostedChatStream(
        iter(
            [SSERecord(event=None, data=json.dumps(payload)) for payload in payloads]
            + [SSERecord(event=None, data="[DONE]")]
        ),
        finish_policy=HostedPresetFinishPolicy(record),
        allowed_extra_keys=record.response_allowances,
    )
    return HostedProviderStream(hosted, record=record)


def test_stream_reasoning_and_control_frames_do_not_become_console_fallback_copy():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderStreamSignals,
    )

    response = _engine_stream(DATABRICKS)
    signals = ConsoleProviderStreamSignals()

    chunks = list(
        ConsoleProviderGateway.normalize_provider_response(response, signals=signals)
    )

    assert chunks == ["Evidence."]
    assert signals.synthetic_fallback_emitted is False
    assert signals.usage_payloads() == [{"prompt_tokens": 8, "completion_tokens": 4}]
    # Ignored disposition: reasoning never survives anywhere, terminal or visible.
    assert response.terminal_turn.reasoning_content is None
    assert response.terminal_turn.finish_reason == "stop"


@pytest.mark.parametrize(
    "record, terminal_reasoning",
    [(DATABRICKS, None), (_PROPRIETARY, "PRIVATE")],
    ids=["ignored", "proprietary"],
)
def test_stream_strips_reasoning_from_visible_deltas_for_non_displayable_records(
    record, terminal_reasoning
):
    # Both non-displayable dispositions share one strip flag in the
    # implementation; pin the delta-level pop for each so a refactor that
    # breaks either case fails here (the gateway tests cannot see a leaked
    # reasoning_content: they render chunks from delta content only).
    frames = list(_engine_stream(record))
    reasoning_delta = frames[1]["choices"][0]["delta"]
    assert "reasoning_content" not in reasoning_delta
    assert reasoning_delta["content"] == ""
    # Empty control frames stay explicit so generic consumers do not render
    # fallback diagnostics for them.
    assert frames[0]["choices"][0]["delta"]["content"] == ""
    assert frames[3]["choices"][0]["delta"]["content"] == ""
    assert frames[2]["choices"][0]["delta"]["content"] == "Evidence."
    terminal = _engine_stream(record)
    list(terminal)  # terminal metadata exists only after clean exhaustion
    # Ignored drops reasoning everywhere; proprietary keeps it terminal-only.
    assert terminal.terminal_turn.reasoning_content == terminal_reasoning


def test_stream_displayable_disposition_keeps_reasoning_visible():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderStreamSignals,
    )

    frames = list(_engine_stream(_DISPLAYABLE))
    reasoning_delta = frames[1]["choices"][0]["delta"]
    assert reasoning_delta["reasoning_content"] == "PRIVATE"
    assert reasoning_delta["content"] == ""

    response = _engine_stream(_DISPLAYABLE)
    signals = ConsoleProviderStreamSignals()
    chunks = list(
        ConsoleProviderGateway.normalize_provider_response(response, signals=signals)
    )
    assert chunks == ["Evidence."]
    assert signals.synthetic_fallback_emitted is False
    assert response.terminal_turn.reasoning_content == "PRIVATE"
    assert response.terminal_turn.finish_reason == "stop"


def test_stream_preserves_safe_terminal_provider_error_type():
    stream = HostedChatStream(
        iter(
            [
                SSERecord(
                    event=None,
                    data=(
                        '{"choices":[{"index":0,"delta":{},'
                        '"finish_reason":"sensitive"}]}'
                    ),
                )
            ]
        ),
        finish_policy=HostedPresetFinishPolicy(_PROVIDER_ERRORS),
    )

    with pytest.raises(ChatProviderError) as exc_info:
        next(HostedProviderStream(stream, record=_PROVIDER_ERRORS))

    assert exc_info.value.provider == "databricks"
    assert "sensitive" not in str(exc_info.value)


def test_stream_close_closes_the_underlying_hosted_stream():
    stream = _engine_stream(DATABRICKS)
    stream.close()
    with pytest.raises(HostedChatProtocolError):
        stream.terminal_turn


def test_stream_continuation_requires_resolution_metadata():
    stream = _engine_stream(DATABRICKS)
    list(stream)  # exhaust cleanly so terminal metadata exists
    with pytest.raises(HostedChatProtocolError, match="Databricks"):
        stream.provider_continuation


# --- response wrapper (ported ZAIResponse terminal metadata) ---


def test_response_wrapper_holds_terminal_metadata_out_of_mapping():
    turn = normalize_hosted_provider_response(
        _DISPLAYABLE, _tool_call_response('{"a":"2+2"}')
    )
    response = HostedProviderResponse(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": turn.text},
                    "finish_reason": turn.finish_reason,
                }
            ]
        },
        terminal_turn=turn,
        provider_continuation=None,
    )

    assert response["choices"][0]["message"]["content"] == "Working."
    assert response.terminal_turn is turn
    assert response.terminal_turn.reasoning_content == "PRIVATE"
    assert response.provider_continuation is None
    assert "PRIVATE" not in repr(response)
