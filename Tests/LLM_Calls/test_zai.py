"""Z.ai/GLM provider-local resolution, request, and response contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

import tldw_chatbook.LLM_Calls.LLM_API_Calls as legacy_adapters
import tldw_chatbook.LLM_Calls.zai as zai
from tldw_chatbook.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.zai import (
    ZAIFinishPolicy,
    ZAIResolution,
    build_zai_chat_payload,
    chat_with_zai,
    normalize_zai_response,
    resolve_zai_request,
)


def _resolution(**overrides: object) -> ZAIResolution:
    values: dict[str, object] = {
        "provider": "zai",
        "model": "glm-5.2",
        "api_key": "secret-key",
        "base_url": "https://api.z.ai/api/paas/v4",
        "timeout": 90.0,
        "retries": 3,
        "retry_delay": 1.0,
        "streaming": True,
    }
    values.update(overrides)
    return ZAIResolution(**values)  # type: ignore[arg-type]


def test_zai_declares_reasoning_as_proprietary() -> None:
    assert ZAIFinishPolicy().reasoning_disposition == "proprietary"


def _tool(name: str = "calculator") -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Evaluate arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }


def _history() -> list[dict[str, object]]:
    return [
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


def test_resolve_zai_request_uses_canonical_precedence_and_current_defaults() -> None:
    config = {
        "api_settings": {
            "zai": {
                "api_key": " config-key ",
                "api_key_env_var": "TEAM_ZAI_KEY",
                "api_base_url": "https://config.example/api/paas/v4/chat/completions",
                "model": "glm-4.6",
                "timeout": 12,
                "retries": 4,
                "retry_delay": 0.5,
                "streaming": False,
            }
        }
    }
    original = deepcopy(config)

    configured = resolve_zai_request(
        app_config=config,
        environ={"TEAM_ZAI_KEY": "env-key", "ZAI_API_KEY": "fallback"},
    )
    explicit = resolve_zai_request(
        explicit_api_key=" explicit-key ",
        explicit_base_url="https://explicit.example/v4",
        explicit_model="glm-5.2",
        app_config=config,
        environ={},
    )

    assert configured == ZAIResolution(
        provider="zai",
        model="glm-4.6",
        api_key="config-key",
        base_url="https://config.example/api/paas/v4",
        timeout=12.0,
        retries=4,
        retry_delay=0.5,
        streaming=False,
    )
    assert explicit.model == "glm-5.2"
    assert explicit.api_key == "explicit-key"
    assert config == original
    assert "config-key" not in repr(configured)

    defaults = resolve_zai_request(
        app_config={"api_settings": {"zai": {}}},
        environ={"ZAI_API_KEY": "env-key"},
    )
    assert defaults.model == "glm-5.2"
    assert defaults.base_url == "https://api.z.ai/api/paas/v4"
    assert zai._DEFAULT_RETRY_DELAY == 5.0
    assert defaults.retry_delay == zai._DEFAULT_RETRY_DELAY


def test_resolve_zai_request_rejects_normalized_table_alias_conflict() -> None:
    with pytest.raises(ChatConfigurationError, match="unambiguous"):
        resolve_zai_request(
            app_config={
                "api_settings": {
                    "ZAI": {"api_key": "alias-secret-canary"},
                    "zai": {"api_key": "canonical-secret-canary"},
                }
            },
            environ={},
        )


def test_resolve_zai_request_rejects_null_canonical_table() -> None:
    with pytest.raises(ChatConfigurationError, match="configuration table"):
        resolve_zai_request(
            app_config={"api_settings": {"zai": None}},
            environ={"ZAI_API_KEY": "must-not-enable-null-settings"},
        )


@pytest.mark.parametrize(
    "app_config",
    [
        {"api_settings": []},
        {"api_settings": {"zai": []}},
        {"api_settings": {"zai": {"api_key": "YOUR_KEY"}}},
        {"api_settings": {"zai": {"model": " "}}},
        {"api_settings": {"zai": {"api_base_url": "https://bad/v4/responses"}}},
        {"api_settings": {"zai": {"timeout": True}}},
        {"api_settings": {"zai": {"retries": 1.5}}},
        {"api_settings": {"zai": {"retry_delay": -1}}},
        {"api_settings": {"zai": {"streaming": "yes"}}},
    ],
)
def test_resolve_zai_request_malformed_exact_table_fails_closed(
    app_config: object,
) -> None:
    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_zai_request(
            app_config=app_config,  # type: ignore[arg-type]
            environ={"ZAI_API_KEY": "must-not-rescue"},
        )

    assert exc_info.value.provider == "zai"
    assert "must-not-rescue" not in str(exc_info.value)


def test_zai_ordinary_chat_uses_clear_thinking_and_exact_allowlist() -> None:
    payload = build_zai_chat_payload(
        resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hello"}],
        streaming=False,
        do_sample=True,
        temperature=0.4,
        top_p=0.8,
        reasoning_effort="high",
        max_tokens=128,
        request_id="request-1",
        user="user-1",
        unknown_private="discarded",
    )

    assert payload == {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
        "do_sample": True,
        "stream": False,
        "thinking": {"type": "enabled", "clear_thinking": True},
        "temperature": 0.4,
        "top_p": 0.8,
        "reasoning_effort": "high",
        "max_tokens": 128,
        "request_id": "request-1",
        "user_id": "user-1",
    }


def test_zai_function_run_preserves_reasoning_and_complete_tool_history() -> None:
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "zai",
            "protocol": "chat_completions",
            "model": "glm-5.2",
            "api_base_url": "https://api.z.ai/api/paas/v4",
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
    history = _history()
    payload = build_zai_chat_payload(
        resolution=_resolution(),
        messages_payload=history,
        tools=[_tool()],
        tool_choice="auto",
        provider_continuations=[checkpoint],
    )

    assert payload["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert payload["messages"][1]["reasoning_content"] == "PRIVATE-REASONING"
    assert payload["messages"][1]["tool_calls"] == history[1]["tool_calls"]
    assert payload["tools"] == [_tool()]
    assert "tool_stream" not in payload


@pytest.mark.parametrize("tool_choice", ["none", "required", {"type": "function"}, 7])
def test_zai_rejects_unsupported_tool_choice(tool_choice: object) -> None:
    with pytest.raises(ChatBadRequestError):
        build_zai_chat_payload(
            resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hello"}],
            tools=[_tool()],
            tool_choice=tool_choice,
        )


@pytest.mark.parametrize(
    "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
)
def test_glm_5_2_accepts_exact_reasoning_efforts(effort: str) -> None:
    payload = build_zai_chat_payload(
        resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hello"}],
        reasoning_effort=effort,
    )
    assert payload["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["", "auto", "ultra", {}, True])
def test_zai_rejects_invalid_or_unsupported_reasoning_effort(effort: object) -> None:
    with pytest.raises(ChatBadRequestError):
        build_zai_chat_payload(
            resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hello"}],
            reasoning_effort=effort,
        )


@pytest.mark.parametrize("model", ["glm-5.3", "glm-6", "glm-5.2-air"])
def test_glm_family_at_or_above_floor_accepts_reasoning_effort(model: str) -> None:
    """TASK-18803: the old exact-id ``glm-5.2`` pin client-side-rejected
    every newer GLM release before a request was ever made. The family
    predicate (version floor 5.2) must let them through."""
    payload = build_zai_chat_payload(
        resolution=_resolution(model=model),
        messages_payload=[{"role": "user", "content": "hello"}],
        reasoning_effort="medium",
    )
    assert payload["reasoning_effort"] == "medium"


@pytest.mark.parametrize("model", ["glm-4.6", "glm-5.1", "glm-5"])
def test_glm_below_floor_still_rejects_reasoning_effort(model: str) -> None:
    """Control: releases below the known-supported floor keep the historical
    client-side rejection (no wire evidence exists to liberalise them)."""
    with pytest.raises(ChatBadRequestError):
        build_zai_chat_payload(
            resolution=_resolution(model=model),
            messages_payload=[{"role": "user", "content": "hello"}],
            reasoning_effort="medium",
        )


def test_zai_response_normalizes_object_arguments_deterministically() -> None:
    response = {
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
                                "arguments": {"z": 1, "a": "2+2"},
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }

    turn = normalize_zai_response(response)

    assert turn.tool_calls[0]["function"]["arguments"] == '{"a":"2+2","z":1}'
    assert turn.reasoning_content == "PRIVATE"
    assert response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] == {"z": 1, "a": "2+2"}


@pytest.mark.parametrize("arguments", [7, True, [], None])
def test_zai_response_rejects_non_object_non_string_arguments(
    arguments: object,
) -> None:
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    with pytest.raises(ChatProviderError):
        normalize_zai_response(response)


@pytest.mark.parametrize(
    "finish_reason", ["sensitive", "model_context_window_exceeded", "network_error"]
)
def test_zai_terminal_error_finishes_are_safe_provider_errors(
    finish_reason: str,
) -> None:
    with pytest.raises(ChatProviderError) as exc_info:
        ZAIFinishPolicy().validate_finish(
            finish_reason=finish_reason,
            has_text=False,
            has_calls=False,
        )
    assert exc_info.value.provider == "zai"
    assert finish_reason not in str(exc_info.value)


def test_zai_stream_preserves_safe_terminal_provider_error_type() -> None:
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
        finish_policy=ZAIFinishPolicy(),
    )

    with pytest.raises(ChatProviderError) as exc_info:
        next(stream)

    assert exc_info.value.provider == "zai"
    assert "sensitive" not in str(exc_info.value)


def test_chat_with_zai_joins_payload_transport_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(zai, "resolve_zai_request", lambda **_kwargs: _resolution())

    def fake_request(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    monkeypatch.setattr(zai, "owned_json_post", fake_request)
    result = chat_with_zai(
        input_data=[{"role": "user", "content": "hello"}],
        model="glm-5.2",
        api_key="secret",
        streaming=False,
        reasoning_effort="max",
    )

    assert captured["streaming"] is False
    assert captured["payload"]["reasoning_effort"] == "max"  # type: ignore[index]
    assert result["choices"][0]["message"]["content"] == "Answer"


def test_zai_malformed_success_is_redacted_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(zai, "resolve_zai_request", lambda **_kwargs: _resolution())
    monkeypatch.setattr(
        zai,
        "owned_json_post",
        lambda **_kwargs: (_ for _ in ()).throw(
            HostedChatProtocolError("PRIVATE-PROVIDER-PAYLOAD")
        ),
    )
    with pytest.raises(ChatProviderError) as exc_info:
        chat_with_zai(
            input_data=[{"role": "user", "content": "hello"}],
            api_key="secret",
            streaming=False,
        )
    assert "PRIVATE-PROVIDER-PAYLOAD" not in str(exc_info.value)


def test_legacy_zai_handler_delegates_without_rewriting_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_strict(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(legacy_adapters, "_strict_chat_with_zai", fake_strict)
    result = legacy_adapters.chat_with_zai(
        input_data=[{"role": "user", "content": "hello"}],
        model="glm-5.2",
        api_key="secret",
        temp=0.4,
        maxp=0.8,
        streaming=False,
        tool_choice="auto",
        reasoning_effort="max",
    )

    assert result == {"ok": True}
    assert captured["temp"] == 0.4
    assert captured["maxp"] == 0.8
    assert captured["tool_choice"] == "auto"
    assert captured["reasoning_effort"] == "max"
