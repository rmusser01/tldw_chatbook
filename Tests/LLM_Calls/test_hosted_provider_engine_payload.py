"""Engine payload builder for engine-driven hosted providers (ADR-179).

Ports the ``build_zai_chat_payload`` contracts from
``Tests/LLM_Calls/test_zai.py`` onto the generic engine parameterized by the
Databricks preset record. Engine divergences from the zai template pinned
here:

- ``resolution.model`` may resolve to ``""`` (Databricks ships no default
  model); the payload layer fails closed on it with display-name copy
  (Task 4 cross-task contract).
- ``reasoning_effort`` is record-off for Databricks Phase 1: supplying a
  value is a bad request, never a silent drop.
- No provider-invented fields: the payload is exactly the OpenAI shape the
  record's ``payload_flags`` describe plus bounded ``extra_body_fields``
  merged last (no zai ``thinking``/``request_id``/``user_id`` quirks).
- Continuation checkpoints are applied to messages (Task 7 dependency).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
    parse_provider_continuation_json,
)
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    build_hosted_chat_payload,
)
from tldw_chatbook.provider_registry import DATABRICKS


def _resolution(streaming=True, model="gpt-4o"):
    return HostedProviderResolution(
        provider="databricks", model=model, api_key="k",
        base_url="https://dbc-1.cloud.databricks.com/openai/v1",
        timeout=90.0, retries=3, retry_delay=5.0, streaming=streaming,
    )


def test_databricks_payload_snapshot():
    payload = build_hosted_chat_payload(
        DATABRICKS, resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hi"}],
        system_message="be brief", temperature=0.2, max_tokens=512,
    )
    assert payload == {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 512,
    }


def test_tools_and_choice_pass_through_openai_shape():
    tools = [{"type": "function", "function": {
        "name": "get_time", "description": "time", "parameters": {"type": "object"}}}]
    payload = build_hosted_chat_payload(
        DATABRICKS, resolution=_resolution(streaming=False),
        messages_payload=[{"role": "user", "content": "hi"}],
        tools=tools, tool_choice="auto",
    )
    assert payload["tools"] == tools and payload["tool_choice"] == "auto"
    assert payload["stream"] is False


def test_message_and_tool_validation_is_strict():
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            DATABRICKS, resolution=_resolution(),
            messages_payload=[{"role": "wizard", "content": "hi"}],
        )
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            DATABRICKS, resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "1bad", "description": "d",
                                                      "parameters": {"type": "object"}}}],
        )


def test_flag_off_fields_are_dropped():
    # reasoning_effort is flag-off for Databricks Phase 1
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            DATABRICKS, resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )


def test_empty_model_fails_closed_with_display_name_copy():
    # Task 4 contract: a blank-default record resolves model to "" and the
    # payload layer gates it (readiness requires key and URL, not a model).
    with pytest.raises(ChatBadRequestError) as exc_info:
        build_hosted_chat_payload(
            DATABRICKS, resolution=_resolution(model=""),
            messages_payload=[{"role": "user", "content": "hi"}],
        )
    assert "Databricks" in str(exc_info.value)
    assert exc_info.value.provider == "databricks"


def test_resolution_provider_mismatch_fails_closed():
    mismatched = HostedProviderResolution(
        provider="zai", model="glm-5.2", api_key="k",
        base_url="https://api.z.ai/api/paas/v4",
        timeout=90.0, retries=3, retry_delay=5.0, streaming=True,
    )
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            DATABRICKS, resolution=mismatched,
            messages_payload=[{"role": "user", "content": "hi"}],
        )


def test_flag_on_optional_fields_pass_through_validated():
    payload = build_hosted_chat_payload(
        DATABRICKS, resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hi"}],
        top_p=0.9, stop=["END"], response_format={"type": "json_object"},
        seed=7, n=2, user="user-1",
    )
    assert payload["top_p"] == 0.9
    assert payload["stop"] == ["END"]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["seed"] == 7
    assert payload["n"] == 2
    assert payload["user"] == "user-1"


def test_payload_flag_off_supplied_value_fails_closed():
    # A preset without a payload flag must not silently drop a caller
    # supplied value: that hides caller intent (same fail-closed rule the
    # reasoning-effort flag uses). Flag-on fields on the same record pass.
    record = replace(DATABRICKS, payload_flags=frozenset({"temperature", "max_tokens"}))
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            record, resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hi"}],
            top_p=0.5,
        )
    payload = build_hosted_chat_payload(
        record, resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hi"}],
        temperature=0.2,
    )
    assert payload["temperature"] == 0.2
    assert "top_p" not in payload


def test_extra_body_fields_merge_last_bounded():
    record = replace(
        DATABRICKS,
        extra_body_fields={"databricks_options": {"max_concurrent_requests": 1}},
    )
    payload = build_hosted_chat_payload(
        record, resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hi"}],
    )
    assert payload["databricks_options"] == {"max_concurrent_requests": 1}
    assert record.extra_body_fields == {"databricks_options": {"max_concurrent_requests": 1}}


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }


def test_continuation_restores_reasoning_onto_owner():
    # The canonical checkpoint format parses provider keys the continuation
    # module's _PAIRINGS admit (moonshot/zai/deepseek, plus databricks from
    # Task 7's registration), so the happy path runs on the registry preset
    # itself -- the exact flow any engine preset with continuation support
    # takes.
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "databricks",
            "protocol": "chat_completions",
            "model": "gpt-4o",
            "api_base_url": "https://dbc-1.cloud.databricks.com/openai/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "Working.",
                    "reasoning_blocks": ["PRIVATE-REASONING"],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                            "state": "completed",
                            "result": "4",
                        }
                    ],
                }
            ],
        }
    )
    resolution = HostedProviderResolution(
        provider="databricks", model="gpt-4o", api_key="k",
        base_url="https://dbc-1.cloud.databricks.com/openai/v1",
        timeout=90.0, retries=3, retry_delay=5.0, streaming=True,
    )
    history = [
        {"role": "user", "content": "Calculate."},
        {
            "role": "assistant",
            "content": "Working.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "4"},
    ]
    payload = build_hosted_chat_payload(
        DATABRICKS, resolution=resolution,
        messages_payload=history,
        tools=[_tool()], tool_choice="auto",
        provider_continuations=[checkpoint],
    )
    assert payload["messages"][1]["reasoning_content"] == "PRIVATE-REASONING"
    assert payload["messages"][1]["tool_calls"] == history[1]["tool_calls"]
    assert payload["tools"] == [_tool()]


def test_continuation_for_unpaired_provider_fails_closed():
    # Providers outside the continuation format's provider pairings
    # (moonshot/zai/deepseek/databricks) must fail closed with display-name
    # copy rather than restore silently. The pairing set is still literal
    # data in ``Chat/provider_continuation.py`` (registry-derived pairings
    # are a later phase), so a groq-keyed synthetic preset stands in for any
    # not-yet-admitted engine provider.
    record = replace(DATABRICKS, key="groq", display_name="Groq")
    resolution = HostedProviderResolution(
        provider="groq", model="gpt-4o", api_key="k",
        base_url="https://dbc-1.cloud.databricks.com/openai/v1",
        timeout=90.0, retries=3, retry_delay=5.0, streaming=True,
    )
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="groq",
        protocol="chat_completions",
        model="gpt-4o",
        api_base_url="https://dbc-1.cloud.databricks.com/openai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="Working.",
                reasoning_blocks=(),
                calls=(
                    ContinuationCall(
                        call_id="call_1",
                        name="calculator",
                        arguments="{}",
                        state="completed",
                        result=ContinuationResult("4"),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ChatBadRequestError) as exc_info:
        build_hosted_chat_payload(
            record,
            resolution=resolution,
            messages_payload=[{"role": "user", "content": "hi"}],
            provider_continuations=[checkpoint],
        )
    assert "Groq" in str(exc_info.value)
