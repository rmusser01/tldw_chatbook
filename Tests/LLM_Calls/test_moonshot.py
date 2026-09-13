"""Moonshot/Kimi provider-local resolution and request contracts."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterator

import pytest

from tldw_chatbook.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json
from tldw_chatbook.LLM_Calls.moonshot import (
    MoonshotFinishPolicy,
    MoonshotResolution,
    MoonshotStream,
    build_moonshot_chat_payload,
    chat_with_moonshot,
    resolve_moonshot_request,
)
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
    HostedChatTurn,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
import tldw_chatbook.LLM_Calls.moonshot as moonshot
import tldw_chatbook.LLM_Calls.LLM_API_Calls as legacy_adapters


def _resolution(**overrides: object) -> MoonshotResolution:
    values: dict[str, object] = {
        "provider": "moonshot",
        "model": "kimi-k3",
        "api_key": "secret-key",
        "base_url": "https://api.moonshot.ai/v1",
        "timeout": 90.0,
        "retries": 3,
        "retry_delay": 1.0,
        "streaming": True,
    }
    values.update(overrides)
    return MoonshotResolution(**values)  # type: ignore[arg-type]


def test_moonshot_declares_reasoning_as_proprietary() -> None:
    assert MoonshotFinishPolicy().reasoning_disposition == "proprietary"


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


def _tool_history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Calculate."},
        {
            "role": "assistant",
            "content": "",
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
        {"role": "assistant", "content": "The answer is 4."},
    ]


def test_resolve_moonshot_request_uses_exact_precedence_and_defaults() -> None:
    config = {
        "api_settings": {
            "moonshot": {
                "api_key": " config-key ",
                "api_key_env_var": "CUSTOM_MOONSHOT_KEY",
                "api_base_url": "https://config.example/v1/chat/completions",
                "model": "moonshot-v1-32k",
                "timeout": 12,
                "retries": 4,
                "retry_delay": 0.5,
                "streaming": False,
            }
        }
    }
    original = deepcopy(config)
    environ = {
        "CUSTOM_MOONSHOT_KEY": "custom-env-key",
        "MOONSHOT_API_KEY": "default-env-key",
    }

    configured = resolve_moonshot_request(app_config=config, environ=environ)
    explicit = resolve_moonshot_request(
        explicit_api_key=" explicit-key ",
        explicit_base_url="https://explicit.example/v1",
        explicit_model="kimi-k3",
        app_config=config,
        environ=environ,
    )

    assert configured == MoonshotResolution(
        provider="moonshot",
        model="moonshot-v1-32k",
        api_key="config-key",
        base_url="https://config.example/v1",
        timeout=12.0,
        retries=4,
        retry_delay=0.5,
        streaming=False,
    )
    assert explicit.model == "kimi-k3"
    assert explicit.api_key == "explicit-key"
    assert explicit.base_url == "https://explicit.example/v1"
    assert config == original
    assert "config-key" not in repr(configured)

    defaults = resolve_moonshot_request(
        app_config={"api_settings": {"moonshot": {}}},
        environ={"MOONSHOT_API_KEY": "env-key"},
    )
    assert defaults.model == "kimi-k3"
    assert defaults.base_url == "https://api.moonshot.ai/v1"
    assert defaults.streaming is True


def test_resolve_moonshot_request_rejects_normalized_table_alias_conflict() -> None:
    with pytest.raises(ChatConfigurationError, match="unambiguous"):
        resolve_moonshot_request(
            app_config={
                "api_settings": {
                    " MOONSHOT ": {"api_key": "alias-secret-canary"},
                    "moonshot": {"api_key": "canonical-secret-canary"},
                }
            },
            environ={},
        )


def test_resolve_moonshot_request_rejects_null_canonical_table() -> None:
    with pytest.raises(ChatConfigurationError, match="configuration table"):
        resolve_moonshot_request(
            app_config={"api_settings": {"moonshot": None}},
            environ={"MOONSHOT_API_KEY": "must-not-enable-null-settings"},
        )


def test_resolve_moonshot_request_configured_env_and_region_fallback() -> None:
    resolution = resolve_moonshot_request(
        app_config={
            "api_settings": {
                "moonshot": {
                    "api_key_env_var": "TEAM_KIMI_KEY",
                    "api_region": "china",
                }
            }
        },
        environ={"TEAM_KIMI_KEY": "team-key", "MOONSHOT_API_KEY": "fallback"},
    )

    assert resolution.api_key == "team-key"
    assert resolution.base_url == "https://api.moonshot.cn/v1"


@pytest.mark.parametrize(
    "app_config",
    [
        {"api_settings": []},
        {"api_settings": {"moonshot": []}},
        {"api_settings": {"moonshot": {"api_key": "YOUR_KEY"}}},
        {"api_settings": {"moonshot": {"model": " "}}},
        {"api_settings": {"moonshot": {"api_base_url": "https://bad/v1/responses"}}},
        {"api_settings": {"moonshot": {"timeout": True}}},
        {"api_settings": {"moonshot": {"timeout": 0}}},
        {"api_settings": {"moonshot": {"retries": 1.5}}},
        {"api_settings": {"moonshot": {"retry_delay": -1}}},
        {"api_settings": {"moonshot": {"streaming": "yes"}}},
        {"api_settings": {"moonshot": {"api_region": "unknown"}}},
        {"api_settings": {"moonshot": {"api_region": []}}},
    ],
)
def test_resolve_moonshot_request_malformed_canonical_table_fails_closed(
    app_config: object,
) -> None:
    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_moonshot_request(
            app_config=app_config,  # type: ignore[arg-type]
            environ={"MOONSHOT_API_KEY": "must-not-rescue"},
        )

    assert exc_info.value.provider == "moonshot"
    assert "must-not-rescue" not in str(exc_info.value)


def test_build_moonshot_payload_copies_complete_tool_history_and_system() -> None:
    messages = _tool_history()
    tools = [_tool()]
    original_messages = deepcopy(messages)
    original_tools = deepcopy(tools)

    payload = build_moonshot_chat_payload(
        resolution=_resolution(),
        messages_payload=messages,
        system_message="Follow instructions.",
        streaming=True,
        tools=tools,
        tool_choice="required",
        reasoning_effort="high",
        max_tokens=128,
    )

    assert payload == {
        "model": "kimi-k3",
        "messages": [{"role": "system", "content": "Follow instructions."}, *messages],
        "stream": True,
        "max_completion_tokens": 128,
        "tools": tools,
        "tool_choice": "required",
        "reasoning_effort": "high",
        "stream_options": {"include_usage": True},
    }
    assert messages == original_messages
    assert tools == original_tools


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "unknown", "content": "x"}],
        [{"role": "user", "content": 3}],
        [{"role": "tool", "tool_call_id": "missing", "content": "x"}],
        _tool_history()[:-2],
        [
            *_tool_history(),
            {"role": "tool", "tool_call_id": "call_1", "content": "again"},
        ],
        [
            {"role": "user", "content": "Calculate."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "not-json",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
        ],
    ],
)
def test_build_moonshot_payload_rejects_malformed_or_unpaired_history(
    messages: list[dict[str, object]],
) -> None:
    with pytest.raises(ChatBadRequestError):
        build_moonshot_chat_payload(
            resolution=_resolution(),
            messages_payload=messages,
        )


@pytest.mark.parametrize("name", ["a", "ab", "1abc", "-bad", "has space"])
def test_build_moonshot_payload_rejects_non_common_function_names(name: str) -> None:
    with pytest.raises(ChatBadRequestError):
        build_moonshot_chat_payload(
            resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "use tool"}],
            tools=[_tool(name)],
        )


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        "none",
        "required",
        {"type": "function", "function": {"name": "calculator"}},
    ],
)
def test_build_moonshot_payload_accepts_documented_tool_choices(
    tool_choice: object,
) -> None:
    payload = build_moonshot_chat_payload(
        resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "use tool"}],
        tools=[_tool()],
        tool_choice=tool_choice,
    )

    assert payload["tool_choice"] == tool_choice


@pytest.mark.parametrize("tool_choice", ["forced", 7, {"type": "web_search"}])
def test_build_moonshot_payload_rejects_unsupported_tool_choices(
    tool_choice: object,
) -> None:
    with pytest.raises(ChatBadRequestError):
        build_moonshot_chat_payload(
            resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "use tool"}],
            tools=[_tool()],
            tool_choice=tool_choice,
        )


def test_kimi_k3_payload_omits_legacy_sampler_fields() -> None:
    payload = build_moonshot_chat_payload(
        resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hello"}],
        streaming=False,
        temperature=0.7,
        top_p=0.9,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
        seed=7,
        user="ignored",
    )

    assert set(payload) == {"model", "messages", "stream"}


@pytest.mark.parametrize(
    "model",
    ["kimi-k3-turbo", "kimi-k4", "kimi-k2.5", "kimi-k2.6", "kimi-latest"],
)
def test_kimi_family_accepts_reasoning_effort(model: str) -> None:
    """TASK-18803: the wire accepts ``reasoning_effort`` across the kimi
    series (probe-verified 200s on kimi-k2.6, kimi-k2.7-code and
    kimi-latest), so the builder must not client-side-reject a release-day
    kimi id the way the old literal ``kimi-k3`` pin did."""
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model=model),
        messages_payload=[{"role": "user", "content": "hello"}],
        reasoning_effort="high",
    )
    assert payload["reasoning_effort"] == "high"


def test_kimi_k3_accepts_medium_reasoning_effort() -> None:
    """``reasoning_effort: "medium"`` answers 200 on kimi-k3
    (chatcmpl-6a872b62bea2d202c1d3f6fa); the old value set client-side
    rejected it."""
    payload = build_moonshot_chat_payload(
        resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hello"}],
        reasoning_effort="medium",
    )
    assert payload["reasoning_effort"] == "medium"


def test_moonshot_v1_still_rejects_reasoning_effort() -> None:
    """Control: the legacy family keeps its historical client-side check."""
    with pytest.raises(ChatBadRequestError):
        build_moonshot_chat_payload(
            resolution=_resolution(model="moonshot-v1-32k"),
            messages_payload=[{"role": "user", "content": "hello"}],
            reasoning_effort="high",
        )


def test_versioned_kimi_family_drops_sampling() -> None:
    """The value-level sampling rejection covers the whole versioned kimi
    family (probe-verified 400s on kimi-k2.5/k2.6/k3), not just kimi-k3."""
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model="kimi-k2.6"),
        messages_payload=[{"role": "user", "content": "hello"}],
        streaming=False,
        temperature=0.4,
        top_p=0.9,
        n=1,
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )
    assert set(payload) == {"model", "messages", "stream"}


def test_kimi_latest_passes_sampling_through() -> None:
    """kimi-latest accepts the full sampling set on the wire
    (chatcmpl-6a872b9816ceb0c0ae780b1e); the old ``moonshot-v1-`` allowlist
    silently discarded the user's settings for it."""
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model="kimi-latest"),
        messages_payload=[{"role": "user", "content": "hello"}],
        streaming=False,
        temperature=0.4,
        top_p=0.9,
        n=1,
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )
    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.9
    assert payload["n"] == 1
    assert payload["presence_penalty"] == 0.1
    assert payload["frequency_penalty"] == 0.1


def test_unrecognized_model_passes_sampling_through() -> None:
    """A future non-kimi family must not have its settings silently dropped
    by a frozen allowlist -- the server adjudicates them."""
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model="moonshot-v2-8k"),
        messages_payload=[{"role": "user", "content": "hello"}],
        streaming=False,
        temperature=0.4,
        top_p=0.9,
    )
    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.9


def test_min_temperature_interplay_scoped_to_v1_family() -> None:
    """The documented n>1/temperature>=0.3 interplay belongs to the
    moonshot-v1 family; kimi-latest must not inherit it."""
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model="kimi-latest"),
        messages_payload=[{"role": "user", "content": "hello"}],
        streaming=False,
        temperature=0.2,
        n=2,
    )
    assert payload["temperature"] == 0.2
    assert payload["n"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning_effort", "extreme"),
        ("reasoning_effort", {}),
        ("max_tokens", True),
        ("temperature", 3),
        ("top_p", -1),
        ("n", 0),
        ("presence_penalty", 9),
        ("frequency_penalty", -9),
        ("seed", True),
    ],
)
def test_kimi_k3_rejects_invalid_supplied_generic_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ChatBadRequestError):
        build_moonshot_chat_payload(
            resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hello"}],
            **{field: value},
        )


def test_legacy_moonshot_family_retains_documented_sampling_surface() -> None:
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model="moonshot-v1-32k"),
        messages_payload=[{"role": "user", "content": "hello"}],
        temperature=0.4,
        top_p=0.8,
        n=2,
        presence_penalty=0.2,
        frequency_penalty=-0.2,
    )

    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.8
    assert payload["n"] == 2
    assert payload["presence_penalty"] == 0.2
    assert payload["frequency_penalty"] == -0.2


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_object", "private": "metadata"},
        {"type": "json_schema"},
        {
            "type": "json_schema",
            "json_schema": {"schema": {"nested": {"too": {"deep": {}}}}},
        },
    ],
)
def test_moonshot_response_format_is_exact_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    response_format: dict[str, object],
) -> None:
    monkeypatch.setattr(moonshot, "_MAX_JSON_DEPTH", 3)

    with pytest.raises(ChatBadRequestError):
        build_moonshot_chat_payload(
            resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hello"}],
            response_format=response_format,
        )


def test_legacy_moonshot_multiple_choices_require_documented_temperature() -> None:
    with pytest.raises(ChatBadRequestError):
        build_moonshot_chat_payload(
            resolution=_resolution(model="moonshot-v1-32k"),
            messages_payload=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            n=2,
        )


def test_kimi_k3_replays_validated_reasoning_from_complete_checkpoint() -> None:
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k3",
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "Visible answer",
                    "reasoning_blocks": ["PRIVATE-REASONING"],
                    "calls": [],
                }
            ],
        }
    )
    payload = build_moonshot_chat_payload(
        resolution=_resolution(),
        messages_payload=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Visible answer"},
            {"role": "user", "content": "next"},
        ],
        provider_continuations=[checkpoint],
    )

    assert payload["messages"][1] == {
        "role": "assistant",
        "content": "Visible answer",
        "reasoning_content": "PRIVATE-REASONING",
    }


@pytest.mark.parametrize("model", ["kimi-k2.6", "kimi-k3-turbo"])
def test_kimi_family_replays_validated_reasoning_from_complete_checkpoint(
    model: str,
) -> None:
    """TASK-19170: preserved-thinking replay is a versioned-kimi family shape.

    Wire probes: kimi-k2.6 returns reasoning_content on every turn
    (chatcmpl-6a8768a3b5c429b466fbc42d) and accepts the replayed
    reasoning_content on a follow-up turn (chatcmpl-6a8768cbb5c429b466fbc43a),
    so the k3-only replay gate was a real gap, not a wire constraint."""
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": model,
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "Visible answer",
                    "reasoning_blocks": ["PRIVATE-REASONING"],
                    "calls": [],
                }
            ],
        }
    )
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model=model),
        messages_payload=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Visible answer"},
            {"role": "user", "content": "next"},
        ],
        provider_continuations=[checkpoint],
    )

    assert payload["messages"][1] == {
        "role": "assistant",
        "content": "Visible answer",
        "reasoning_content": "PRIVATE-REASONING",
    }


def test_kimi_family_complete_tool_checkpoint_expands_private_rounds_off_k3() -> None:
    """A k3-style complete checkpoint (tool rounds + final reasoning round)
    must expand its private rounds for every family member that produces the
    shape, not only the exact kimi-k3 id."""
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k2.6",
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": ["tool reasoning"],
                    "calls": [
                        _call_payload(
                            "call_1", state="completed", result="4"
                        )
                    ],
                },
                {
                    "assistant_content": "Visible answer",
                    "reasoning_blocks": ["final reasoning"],
                    "calls": [],
                },
            ],
        }
    )
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model="kimi-k2.6"),
        messages_payload=[
            {"role": "user", "content": "Calculate."},
            {"role": "assistant", "content": "Visible answer"},
            {"role": "user", "content": "next"},
        ],
        provider_continuations=[checkpoint],
    )

    messages = payload["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert messages[1]["reasoning_content"] == "tool reasoning"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "4",
    }
    assert messages[3] == {
        "role": "assistant",
        "content": "Visible answer",
        "reasoning_content": "final reasoning",
    }


def test_kimi_family_old_style_complete_checkpoint_annotates_tool_rounds() -> None:
    """Durable-data control: pre-19170 non-k3 complete checkpoints (no final
    reasoning round) keep the annotate-in-place replay they always had."""
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k2.6",
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": ["tool reasoning"],
                    "calls": [
                        _call_payload(
                            "call_1", state="completed", result="4"
                        )
                    ],
                }
            ],
        }
    )
    history = _tool_history()
    payload = build_moonshot_chat_payload(
        resolution=_resolution(model="kimi-k2.6"),
        messages_payload=history,
        provider_continuations=[checkpoint],
    )

    assert payload["messages"][1]["reasoning_content"] == "tool reasoning"
    assert payload["messages"][3] == {
        "role": "assistant",
        "content": "The answer is 4.",
    }


def _call_payload(
    call_id: str,
    *,
    state: str = "pending",
    result: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "call_id": call_id,
        "name": "calculator",
        "arguments": '{"expression":"2+2"}',
        "state": state,
    }
    if result is not None:
        value["result"] = result
    return value


def _turn(
    *,
    text: str = "Visible answer",
    reasoning_content: str | None = "PRIVATE-REASONING",
    tool_calls: tuple = (),
) -> HostedChatTurn:
    return HostedChatTurn(
        text=text,
        tool_calls=tool_calls,
        assistant_message={"role": "assistant", "content": text},
        finish_reason="tool_calls" if tool_calls else "stop",
        reasoning_content=reasoning_content,
        usage=None,
    )


def test_kimi_family_plain_turn_preserves_reasoning_checkpoint() -> None:
    """kimi-k2.6 returns reasoning_content on plain turns (TASK-19170 probe
    B, chatcmpl-6a8768a9b5c429b466fbc42f): the preserved-thinking checkpoint
    must be created exactly as it is for kimi-k3."""
    candidate = moonshot._moonshot_continuation_candidate(
        _turn(),
        resolution=_resolution(model="kimi-k2.6"),
        provider_continuations=[],
    )

    assert candidate is not None
    assert candidate.state == "complete"
    assert candidate.model == "kimi-k2.6"
    assert candidate.rounds[-1].calls == ()
    assert candidate.rounds[-1].reasoning_blocks == ("PRIVATE-REASONING",)


def test_kimi_latest_plain_turn_produces_no_checkpoint() -> None:
    """kimi-latest returned NO reasoning_content on the wire (TASK-19170
    probe A, chatcmpl-6a8768a616ceb0c0ae780f2c): even a crafted reasoning
    payload must not mint a preserved-thinking checkpoint for it."""
    candidate = moonshot._moonshot_continuation_candidate(
        _turn(),
        resolution=_resolution(model="kimi-latest"),
        provider_continuations=[],
    )

    assert candidate is None


def _active_family_checkpoint(model: str):
    return parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 2,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": model,
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": ["tool reasoning"],
                    "calls": [
                        _call_payload(
                            "call_1", state="completed", result="4"
                        )
                    ],
                }
            ],
        }
    )


def test_kimi_family_tool_loop_completion_appends_final_reasoning_round() -> None:
    """The loop-completing kimi-k2.6 turn returns reasoning_content on the
    wire (TASK-19170 probe D1/D2, chatcmpl-6a876916/6a876918...): completion
    must preserve it as the k3-style final reasoning round."""
    candidate = moonshot._moonshot_continuation_candidate(
        _turn(reasoning_content="FINAL-REASONING"),
        resolution=_resolution(model="kimi-k2.6"),
        provider_continuations=[_active_family_checkpoint("kimi-k2.6")],
    )

    assert candidate is not None
    assert candidate.state == "complete"
    assert candidate.rounds[-1].calls == ()
    assert candidate.rounds[-1].reasoning_blocks == ("FINAL-REASONING",)
    assert candidate.rounds[0].calls[0].state == "completed"


def test_kimi_family_tool_loop_completion_without_reasoning_keeps_stored_shape() -> (
    None
):
    """Durable-shape control: a family completion turn with no reasoning
    keeps the pre-19170 complete shape instead of failing canonical parse."""
    candidate = moonshot._moonshot_continuation_candidate(
        _turn(reasoning_content=None),
        resolution=_resolution(model="kimi-k2.6"),
        provider_continuations=[_active_family_checkpoint("kimi-k2.6")],
    )

    assert candidate is not None
    assert candidate.state == "complete"
    assert len(candidate.rounds) == 1
    assert candidate.rounds[-1].calls[0].state == "completed"


def test_moonshot_finish_policy_accepts_mixed_tools_and_rejects_contradictions() -> (
    None
):
    policy = MoonshotFinishPolicy()

    assert (
        policy.validate_finish(
            finish_reason="tool_calls", has_text=True, has_calls=True
        )
        == "tool_calls"
    )
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(finish_reason="stop", has_text=True, has_calls=True)
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(finish_reason="stop", has_text=False, has_calls=False)


def test_chat_with_moonshot_joins_resolution_payload_transport_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        moonshot, "resolve_moonshot_request", lambda **_kwargs: _resolution()
    )

    def fake_request(**kwargs: object) -> HostedChatTurn:
        captured.update(kwargs)
        return HostedChatTurn(
            text="Answer",
            tool_calls=(),
            assistant_message={
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "PRIVATE-REASONING",
            },
            finish_reason="stop",
            reasoning_content="PRIVATE-REASONING",
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    monkeypatch.setattr(moonshot, "hosted_chat_request", fake_request)

    result = chat_with_moonshot(
        input_data=[{"role": "user", "content": "hello"}],
        model="kimi-k3",
        api_key="secret",
        streaming=False,
        reasoning_effort="high",
    )

    assert captured["streaming"] is False
    assert captured["payload"] == {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "reasoning_effort": "high",
    }
    assert result == {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    assert result.provider_continuation is not None
    assert result.provider_continuation.rounds[-1].reasoning_blocks == (
        "PRIVATE-REASONING",
    )


def test_chat_with_moonshot_maps_malformed_success_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        moonshot, "resolve_moonshot_request", lambda **_kwargs: _resolution()
    )
    monkeypatch.setattr(
        moonshot,
        "hosted_chat_request",
        lambda **_kwargs: (_ for _ in ()).throw(
            HostedChatProtocolError("PRIVATE-PROVIDER-PAYLOAD")
        ),
    )

    with pytest.raises(ChatProviderError) as exc_info:
        chat_with_moonshot(
            input_data=[{"role": "user", "content": "hello"}],
            api_key="secret",
            streaming=False,
        )

    assert exc_info.value.provider == "moonshot"
    assert "PRIVATE-PROVIDER-PAYLOAD" not in str(exc_info.value)


def test_legacy_moonshot_handler_delegates_without_rewriting_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_strict(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(legacy_adapters, "_strict_chat_with_moonshot", fake_strict)

    result = legacy_adapters.chat_with_moonshot(
        input_data=[{"role": "user", "content": "hello"}],
        model="kimi-k3",
        api_key="secret",
        temp=0.4,
        maxp=0.8,
        streaming=False,
        tool_choice="none",
        reasoning_effort="max",
    )

    assert result == {"ok": True}
    assert captured["input_data"] == [{"role": "user", "content": "hello"}]
    assert captured["temp"] == 0.4
    assert captured["maxp"] == 0.8
    assert captured["tool_choice"] == "none"
    assert captured["reasoning_effort"] == "max"


def test_moonshot_stream_hides_reasoning_chunks_but_retains_terminal_turn() -> None:
    records: Iterator[SSERecord] = iter(
        [
            SSERecord(
                event=None,
                data=(
                    '{"choices":[{"index":0,"delta":{"role":"assistant",'
                    '"content":"Answer","reasoning_content":"PRIVATE"},'
                    '"finish_reason":"stop"}]}'
                ),
            ),
            SSERecord(
                event=None,
                data=(
                    '{"choices":[],"usage":{"prompt_tokens":2,'
                    '"completion_tokens":1,"total_tokens":3}}'
                ),
            ),
            SSERecord(event=None, data="[DONE]"),
        ]
    )
    stream = MoonshotStream(
        HostedChatStream(records, finish_policy=MoonshotFinishPolicy())
    )

    events = list(stream)

    assert "reasoning_content" not in events[0]["choices"][0]["delta"]
    assert stream.terminal_turn.reasoning_content == "PRIVATE"
    assert stream.terminal_turn.usage == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
