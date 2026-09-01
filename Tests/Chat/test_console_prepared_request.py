from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.Chat import console_prepared_request as prepared_request
from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    MEMORY_CLOSE_TAG,
    MEMORY_OPEN_TAG,
    PreparedConsoleRequest,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    tagged_memory_message,
    thaw_json,
)
from tldw_chatbook.Chat.provider_continuation import (
    continuation_owner_group,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)


def _word_count(messages: list[dict], _model: str) -> int:
    count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            count += len(content.split()) + 1
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    count += len(str(part.get("text", "")).split())
                else:
                    count += 10
        if message.get("tool_calls"):
            count += 10
    return count


def _capacity(ceiling: int | None):
    if ceiling is None:
        return resolve_request_capacity(context_window_tokens=None)
    # A 512-token minimum margin and a 10-token reply reserve leave `ceiling`.
    return resolve_request_capacity(
        context_window_tokens=ceiling + 522,
        requested_response_tokens=10,
    )


def test_idle_request_sentinel_is_fixed_immutable_and_app_owned() -> None:
    sentinel = getattr(prepared_request, "IDLE_REQUEST_SENTINEL", None)
    assert sentinel is not None, "canonical idle sentinel must be defined"

    assert sentinel["role"] == "user"
    assert sentinel["content"] == prepared_request.IDLE_REQUEST_SENTINEL_TEXT
    assert (
        sentinel[prepared_request.IDLE_REQUEST_OWNER_KEY]
        == prepared_request.IDLE_REQUEST_OWNER_VALUE
    )
    with pytest.raises(TypeError):
        sentinel["content"] = "mutated"

    projected = prepare_provider_request(
        PreparedConsoleRequest(active_request=(sentinel,)),
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(None),
        count_fn=_word_count,
        apply_safety_window=False,
    )
    assert prepared_request.IDLE_REQUEST_OWNER_KEY not in projected.messages[0]


def test_semantic_request_is_immutable_and_preserves_complete_units() -> None:
    source = [
        {"role": "system", "content": "  original system bytes  "},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]

    request = build_console_request(source)
    source[0]["content"] = "mutated"

    assert request.system[0]["content"] == "  original system bytes  "
    assert [row["role"] for row in request.compactable[0].messages] == [
        "user",
        "assistant",
    ]
    assert [row["role"] for row in request.active_request] == [
        "user",
        "assistant",
        "tool",
    ]
    assert [row["role"] for row in request.active_tool_loop] == ["assistant", "tool"]
    with pytest.raises(TypeError):
        request.system[0]["content"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.active_request = ()  # type: ignore[misc]


def test_fenced_tool_loop_rows_have_a_distinct_provider_neutral_category() -> None:
    request = build_console_request(
        [
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": '```tool_call\n{"name":"lookup","arguments":{}}\n```',
            },
            {"role": "user", "content": "Tool result for lookup: found"},
        ]
    )

    assert [row["role"] for row in request.active_request] == [
        "user",
        "assistant",
        "user",
    ]
    assert [row["role"] for row in request.active_tool_loop] == [
        "assistant",
        "user",
    ]
    assert [row["role"] for row in request.flattened_messages()] == [
        "user",
        "assistant",
        "user",
    ]


@pytest.mark.parametrize("tag", ("tool", "tool_calls", "tool_call_schema"))
def test_fenced_tool_lookalikes_remain_ordinary_assistant_text(tag: str) -> None:
    request = build_console_request(
        [
            {"role": "user", "content": "Explain this"},
            {
                "role": "assistant",
                "content": f'```{tag}\n{{"name":"lookup","arguments":{{}}}}\n```',
            },
        ]
    )

    assert request.active_tool_loop == ()


def test_completed_tool_loop_keeps_following_assistant_in_message_order() -> None:
    request = build_console_request(
        [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "tool_call_id": "c", "content": "found"},
            {"role": "assistant", "content": "The answer is found."},
            {"role": "user", "content": "Next"},
        ]
    )

    assert [row["role"] for row in request.compactable[0].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [row["role"] for row in request.compactable[0].tool_loop] == [
        "assistant",
        "tool",
    ]
    assert [row["role"] for row in request.flattened_messages()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]


def test_single_preamble_serialization_keeps_original_and_tagged_memory_owned() -> None:
    original = "  keep this stored system unchanged  "
    memory = tagged_memory_message("Earlier facts")
    semantic = build_console_request(
        [
            {"role": "system", "content": original},
            {"role": "user", "content": "now"},
        ],
        memory=[memory],
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        model="m",
        capacity=_capacity(None),
        count_fn=_word_count,
    )

    assert semantic.system[0]["content"] == original
    assert prepared.semantic.memory[0]["content"].startswith(MEMORY_OPEN_TAG)
    assert prepared.semantic.memory[0]["content"].endswith(MEMORY_CLOSE_TAG)
    assert prepared.system_message == (
        f"{original.strip()}\n\n{prepared.semantic.memory[0]['content']}"
    )
    assert [row["role"] for row in prepared.messages_payload] == ["user"]


def test_response_format_is_frozen_and_dispatched_only_when_present() -> None:
    semantic = build_console_request([{"role": "user", "content": "now"}])
    source = {
        "type": "json_schema",
        "json_schema": {"required": ["answer"]},
    }
    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        model="gpt-5.6-terra",
        provider="openai",
        capacity=_capacity(None),
        count_fn=_word_count,
        response_format=source,
    )
    source["json_schema"]["required"].append("mutated")
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.6-terra",
        ready=True,
        execution_key="openai",
        api_key="k",
        streaming=False,
    )

    assert prepared.response_format is not None
    assert prepared.response_format["json_schema"]["required"] == ("answer",)
    kwargs = ConsoleProviderGateway._chat_api_kwargs_from_prepared(resolution, prepared)
    assert kwargs["response_format"]["json_schema"]["required"] == ["answer"]
    assert kwargs["api_base_url"] == "https://api.openai.com/v1"

    plain = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        model="gpt-5.6-terra",
        provider="openai",
        capacity=_capacity(None),
        count_fn=_word_count,
    )
    plain_kwargs = ConsoleProviderGateway._chat_api_kwargs_from_prepared(
        resolution, plain
    )
    assert "response_format" not in plain_kwargs


def test_distinct_role_serialization_keeps_system_and_memory_rows_separate() -> None:
    original = "original"
    memory = tagged_memory_message("Earlier facts")
    semantic = build_console_request(
        [
            {"role": "system", "content": original},
            {"role": "user", "content": "now"},
        ],
        memory=[memory],
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(None),
        count_fn=_word_count,
    )

    assert prepared.system_message is None
    assert [row["role"] for row in prepared.messages] == [
        "system",
        "system",
        "user",
    ]
    assert prepared.messages[0]["content"] == original
    assert prepared.messages[1]["content"] == memory["content"]
    assert "_tldw_context_owner" not in prepared.messages[1]


@pytest.mark.parametrize("wire_style", ["distinct_roles", "single_preamble"])
def test_memory_wire_projection_is_unique_owned_and_private_anchor_free(
    wire_style: str,
) -> None:
    memory = tagged_memory_message("PROJECTED-MEMORY-CANARY")
    semantic = build_console_request(
        [
            {
                "role": "system",
                "content": "ORIGINAL-SYSTEM-CANARY",
                prepared_request.PERSISTED_MESSAGE_ID_KEY: "system-private",
                prepared_request.PERSISTED_CONVERSATION_ID_KEY: "conversation-1",
            },
            {
                "role": "user",
                "content": "active",
                prepared_request.PERSISTED_MESSAGE_ID_KEY: "u1",
                prepared_request.PERSISTED_CONVERSATION_ID_KEY: "conversation-1",
            },
        ],
        memory=(memory,),
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style=wire_style,
        model="m",
        capacity=_capacity(None),
        count_fn=_word_count,
        apply_safety_window=False,
    )

    wire = "\n".join(
        str(row.get("content", "")) for row in prepared.messages_payload
    )
    if prepared.system_message:
        wire = prepared.system_message + "\n" + wire
    assert wire.count("PROJECTED-MEMORY-CANARY") == 1
    assert wire.index("ORIGINAL-SYSTEM-CANARY") < wire.index(
        "PROJECTED-MEMORY-CANARY"
    )
    assert all(
        prepared_request.PERSISTED_MESSAGE_ID_KEY not in row
        and prepared_request.PERSISTED_CONVERSATION_ID_KEY not in row
        for row in (*prepared.messages, *prepared.messages_payload)
    )
    assert not any(
        row.get("role") == "user"
        and "PROJECTED-MEMORY-CANARY" in str(row.get("content", ""))
        for row in prepared.messages_payload
    )
    assert prepared.accounting.memory_tokens > 0


def test_tagged_memory_survives_raw_agent_handoff_without_becoming_user_system() -> (
    None
):
    memory = tagged_memory_message("Earlier facts")
    semantic = build_console_request(
        [
            {"role": "system", "content": "agent operating prompt"},
            dict(memory),
            {"role": "user", "content": "continue"},
        ]
    )

    assert [row["content"] for row in semantic.system] == ["agent operating prompt"]
    assert semantic.memory == (memory,)


def test_untrimmed_projection_reports_overflow_without_dropping_units() -> None:
    semantic = build_console_request(
        [
            {"role": "user", "content": "old one two three"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active request"},
        ]
    )
    projected = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(3),
        count_fn=_word_count,
        apply_safety_window=False,
    )

    assert projected.known_overflow is True
    assert projected.dropped_units == 0
    assert len(projected.semantic.compactable) == 1


def test_capacity_uses_output_cap_without_hidden_half_window_clamp() -> None:
    capacity = resolve_request_capacity(
        context_window_tokens=10_000,
        provider_input_cap_tokens=9_000,
        provider_output_cap_tokens=2_000,
        requested_response_tokens=8_000,
    )

    assert capacity.requested_response_tokens == 8_000
    assert capacity.effective_response_tokens == 2_000
    assert capacity.safety_margin_tokens == 512
    assert capacity.effective_input_ceiling_tokens == 7_488


def test_windowing_drops_oldest_whole_units_deterministically() -> None:
    semantic = build_console_request(
        [
            {"role": "system", "content": "contract"},
            {"role": "user", "content": "old one two three"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new one two"},
            {"role": "assistant", "content": "new answer"},
            {"role": "user", "content": "active request"},
        ]
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(12),
        count_fn=_word_count,
    )

    assert prepared.dropped_units == 1
    assert prepared.dropped_messages == 2
    assert [row["content"] for row in prepared.semantic.compactable[0].messages] == [
        "new one two",
        "new answer",
    ]
    assert prepared.known_overflow is False


def test_known_mandatory_overflow_is_explicit_and_compaction_cannot_remove_it() -> None:
    semantic = build_console_request(
        [
            {"role": "system", "content": "large mandatory system contract"},
            {"role": "user", "content": "active request is also mandatory"},
        ]
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(3),
        count_fn=_word_count,
    )

    assert prepared.known_overflow is True
    assert prepared.dropped_units == 0
    assert prepared.accounting.non_compactable_tokens == (
        prepared.accounting.total_input_tokens
    )


def test_unknown_and_user_overridden_limits_are_honestly_labeled() -> None:
    semantic = build_console_request([{"role": "user", "content": "hello"}])
    unknown = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="unknown",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )
    overridden = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="unknown",
        capacity=resolve_request_capacity(
            context_window_tokens=None,
            context_window_override_tokens=4_000,
        ),
        count_fn=_word_count,
    )
    input_bounded = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="unknown",
        capacity=resolve_request_capacity(
            context_window_tokens=None,
            provider_input_cap_tokens=2_000,
        ),
        count_fn=_word_count,
    )

    assert unknown.capacity.effective_input_ceiling_tokens is None
    assert unknown.safety_label == "limit unknown; provider safety unverified"
    assert overridden.capacity.effective_input_ceiling_tokens == 2_464
    assert overridden.safety_label == "user-bounded; provider safety unverified"
    assert input_bounded.capacity.effective_input_ceiling_tokens == 2_000
    assert (
        input_bounded.safety_label == "provider input-bounded; total context unverified"
    )


def test_unknown_capacity_does_not_invent_a_provider_safe_fallback() -> None:
    semantic = build_console_request(
        [
            {"role": "user", "content": "older question " * 100},
            {"role": "assistant", "content": "older answer " * 100},
            {"role": "user", "content": "current request"},
        ]
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="unknown",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    # ADR-052 deliberately keeps unknown capacity unverified. Guessing a
    # fallback here could still overflow a smaller provider window while
    # falsely presenting the request as bounded; provider error handling is
    # the final boundary until capability data or a user bound is supplied.
    assert prepared.capacity.effective_input_ceiling_tokens is None
    assert prepared.dropped_units == 0
    assert prepared.safety_label == "limit unknown; provider safety unverified"


def test_multimodal_and_tool_schema_material_is_in_exact_accounting() -> None:
    plain = build_console_request([{"role": "user", "content": "describe"}])
    rich = build_console_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png"}},
                ],
            }
        ],
        tools=[{"type": "function", "function": {"name": "inspect"}}],
    )
    kwargs = {
        "wire_style": "distinct_roles",
        "model": "gpt-4o",
        "capacity": _capacity(None),
    }

    plain_prepared = prepare_provider_request(plain, **kwargs)
    rich_prepared = prepare_provider_request(rich, **kwargs)

    assert rich_prepared.accounting.total_input_tokens > (
        plain_prepared.accounting.total_input_tokens + 900
    )
    # TASK-26019: tool schemas are now their own bucket instead of riding
    # inside mandatory -- the pin's intent (tools ARE counted) is unchanged.
    assert rich_prepared.accounting.tool_schema_tokens > 0
    assert rich_prepared.tools == rich.tools


def test_prepared_console_request_requires_an_active_request() -> None:
    with pytest.raises(ValueError, match="active request"):
        PreparedConsoleRequest()


def _resolution(*, max_tokens: int | None = None) -> ConsoleProviderResolution:
    return ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-4o",
        ready=True,
        execution_key="openai",
        max_tokens=max_tokens,
    )


@pytest.mark.asyncio
async def test_gateway_prepares_once_and_dispatches_that_exact_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched: dict = {}
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kwargs: (
            dispatched.update(kwargs) or {"choices": [{"message": {"content": "ok"}}]}
        )
    )
    prepared_calls = []
    original_prepare = gateway.prepare_chat_request

    def prepare_spy(*args, **kwargs):
        artifact = original_prepare(*args, **kwargs)
        prepared_calls.append(artifact)
        return artifact

    monkeypatch.setattr(gateway, "prepare_chat_request", prepare_spy)
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "hello"},
    ]

    chunks = [chunk async for chunk in gateway.stream_chat(_resolution(), messages)]

    assert chunks == ["ok"]
    assert len(prepared_calls) == 1
    artifact = prepared_calls[0]
    assert dispatched["system_message"] == artifact.system_message
    assert dispatched["messages_payload"] == [
        thaw_json(item) for item in artifact.messages_payload
    ]
    assert artifact.accounting.total_input_tokens > 0
    await gateway.aclose()


@pytest.mark.asyncio
async def test_gateway_does_not_rebuild_a_supplied_prepared_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched: dict = {}
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kwargs: (
            dispatched.update(kwargs) or {"choices": [{"message": {"content": "ok"}}]}
        )
    )
    semantic = build_console_request([{"role": "user", "content": "hello"}])
    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        provider="openai",
        model="gpt-4o",
        capacity=resolve_request_capacity(context_window_tokens=None),
    )

    monkeypatch.setattr(
        gateway,
        "prepare_chat_request",
        lambda *_args, **_kwargs: pytest.fail("prepared request was rebuilt"),
    )
    chunks = [chunk async for chunk in gateway.stream_chat(_resolution(), prepared)]

    assert chunks == ["ok"]
    assert dispatched["messages_payload"] == [
        thaw_json(item) for item in prepared.messages_payload
    ]
    await gateway.aclose()


@pytest.mark.asyncio
async def test_gateway_never_dispatches_a_known_overflow() -> None:
    called = False

    def chat_call(**_kwargs):
        nonlocal called
        called = True
        return {"choices": [{"message": {"content": "not reached"}}]}

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_call)
    semantic = build_console_request(
        [
            {"role": "system", "content": "mandatory contract words"},
            {"role": "user", "content": "mandatory active words"},
        ]
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        provider="openai",
        model="gpt-4o",
        capacity=_capacity(1),
        count_fn=_word_count,
    )

    with pytest.raises(ChatBadRequestError, match="Compaction cannot remove"):
        _ = [chunk async for chunk in gateway.stream_chat(_resolution(), prepared)]

    assert called is False
    await gateway.aclose()


@pytest.mark.asyncio
async def test_gateway_dispatches_effective_not_requested_response_limit() -> None:
    dispatched: dict = {}
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kwargs: (
            dispatched.update(kwargs) or {"choices": [{"message": {"content": "ok"}}]}
        )
    )
    semantic = build_console_request([{"role": "user", "content": "hello"}])
    capacity = resolve_request_capacity(
        context_window_tokens=10_000,
        provider_output_cap_tokens=128,
        requested_response_tokens=512,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        provider="openai",
        model="gpt-4o",
        capacity=capacity,
    )

    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            _resolution(max_tokens=512),
            prepared,
        )
    ]

    assert dispatched["max_tokens"] == 128
    await gateway.aclose()


@pytest.mark.asyncio
async def test_gateway_agent_tool_payload_is_prepared_once_as_complete_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched: dict = {}
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kwargs: (
            dispatched.update(kwargs) or {"choices": [{"message": {"content": "done"}}]}
        )
    )
    artifacts = []
    original_prepare = gateway.prepare_chat_request

    def prepare_spy(*args, **kwargs):
        artifact = original_prepare(*args, **kwargs)
        artifacts.append(artifact)
        return artifact

    monkeypatch.setattr(gateway, "prepare_chat_request", prepare_spy)
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    messages = [
        {"role": "user", "content": "Find it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            _resolution(),
            messages,
            tools=tools,
        )
    ]

    assert chunks == ["done"]
    assert len(artifacts) == 1
    assert [row["role"] for row in artifacts[0].semantic.active_request] == [
        "user",
        "assistant",
        "tool",
    ]
    assert [row["role"] for row in artifacts[0].semantic.active_tool_loop] == [
        "assistant",
        "tool",
    ]
    assert dispatched["tools"] == tools
    await gateway.aclose()


@pytest.mark.asyncio
async def test_gateway_resolves_separate_context_input_and_output_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.model_capabilities as capability_module

    class Registry:
        @staticmethod
        def get_model_capabilities(_provider: str, _model: str) -> dict[str, int]:
            return {
                "context_window": 20_000,
                "max_input_tokens": 15_000,
                "max_output_tokens": 1_000,
            }

    monkeypatch.setattr(
        capability_module,
        "get_model_capabilities",
        lambda: Registry(),
    )
    gateway = ConsoleProviderGateway()

    prepared = gateway.prepare_chat_request(
        _resolution(max_tokens=4_000),
        [{"role": "user", "content": "hello"}],
    )

    assert prepared.capacity.context_window_tokens == 20_000
    assert prepared.capacity.provider_input_cap_tokens == 15_000
    assert prepared.capacity.provider_output_cap_tokens == 1_000
    assert prepared.capacity.requested_response_tokens == 4_000
    assert prepared.capacity.effective_response_tokens == 1_000
    assert prepared.capacity.effective_input_ceiling_tokens == 15_000
    await gateway.aclose()


def _private_group(owner_id: str, *, call_id: str, active: bool = False):
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active" if active else "complete",
            "rounds": [
                {
                    "assistant_content": "" if active else "answer",
                    "reasoning_blocks": ["PRIVATE-PREPARED-CANARY " * 10],
                    "calls": [
                        {
                            "call_id": call_id,
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "pending" if active else "completed",
                            **({} if active else {"result": "done"}),
                        }
                    ],
                }
            ],
        }
    )
    return continuation_owner_group(
        {"id": owner_id, "role": "assistant", "content": ""}, checkpoint
    )


def test_prepared_request_counts_and_evicts_private_groups_with_owner_units() -> None:
    groups = (
        _private_group("a1", call_id="call_1"),
        _private_group("a2", call_id="call_2"),
    )
    semantic = build_console_request(
        [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": "answer",
                CONTINUATION_OWNER_KEY: "a1",
            },
            {"role": "user", "content": "new"},
            {
                "role": "assistant",
                "content": "answer",
                CONTINUATION_OWNER_KEY: "a2",
            },
            {"role": "user", "content": "current"},
        ],
        continuation_groups=groups,
    )
    selected = prepare_provider_request(
        semantic.without_oldest_units(1),
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(None),
        count_fn=_word_count,
    )
    exact = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(selected.accounting.total_input_tokens),
        count_fn=_word_count,
    )
    one_under = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(selected.accounting.total_input_tokens - 1),
        count_fn=_word_count,
    )

    assert exact.dropped_units == 1
    assert [group.owner_message_id for group in exact.continuation_groups] == ["a2"]
    assert one_under.dropped_units == 2
    assert one_under.continuation_groups == ()
    assert all(CONTINUATION_OWNER_KEY not in row for row in exact.messages_payload)
    assert "PRIVATE-PREPARED-CANARY" not in repr(exact)


def test_active_private_group_is_mandatory_and_fails_closed_when_over_budget() -> None:
    group = _private_group("a1", call_id="call_1", active=True)
    semantic = build_console_request(
        [
            {"role": "user", "content": "current"},
            {
                "role": "assistant",
                "content": "",
                CONTINUATION_OWNER_KEY: "a1",
            },
        ],
        continuation_groups=(group,),
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=_capacity(1),
        count_fn=_word_count,
    )

    assert prepared.known_overflow is True
    assert prepared.dropped_units == 0
    assert prepared.continuation_groups == (group,)


# --- TASK-26019: category breakdown fields on the accounting ----------------


def _breakdown_policy():
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        new_opaque_id,
    )

    return FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )


def test_tool_schemas_split_out_of_mandatory_by_construction() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "words " * 50,
                "parameters": {"type": "object"},
            },
        }
    ]
    with_tools = prepare_provider_request(
        build_console_request(
            [{"role": "user", "content": "describe"}], tools=tools
        ),
        wire_style="distinct_roles",
        model="gpt-4o",
        capacity=_capacity(None),
    )
    without_tools = prepare_provider_request(
        build_console_request([{"role": "user", "content": "describe"}]),
        wire_style="distinct_roles",
        model="gpt-4o",
        capacity=_capacity(None),
    )

    accounting = with_tools.accounting
    assert accounting.tool_schema_tokens > 0
    assert accounting.mandatory_tokens == 0, (
        "tool schemas must no longer masquerade as mandatory context"
    )
    assert without_tools.accounting.tool_schema_tokens == 0
    # the split is a partition of the same total, not a new estimate
    assert accounting.total_input_tokens == (
        accounting.system_tokens
        + accounting.memory_tokens
        + accounting.tool_schema_tokens
        + accounting.mandatory_tokens
        + accounting.compactable_tokens
        + accounting.active_request_tokens
    )
    # the one existing consumer reads non_compactable -- unchanged meaning
    assert accounting.non_compactable_tokens == (
        accounting.total_input_tokens - accounting.compactable_tokens
    )


def test_attachment_tokens_cover_conversation_image_parts() -> None:
    prepared = prepare_provider_request(
        build_console_request(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "a1"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA"},
                        },
                    ],
                },
            ]
        ),
        wire_style="distinct_roles",
        model="gpt-4o",
        capacity=_capacity(None),
    )

    accounting = prepared.accounting
    assert accounting.attachment_tokens > 0
    assert accounting.attachment_tokens <= (
        accounting.compactable_tokens + accounting.active_request_tokens
    )


def test_rag_context_attributes_only_with_provenance() -> None:
    from tldw_chatbook.Chat.console_trace_provenance import (
        ProviderArtifactTraceProvenance,
        TraceProvenanceSource,
    )

    policy = _breakdown_policy()
    rag_row = {"role": "system", "content": "retrieved snippet " * 30}
    plain_row = {"role": "system", "content": "instructions"}

    without_provenance = prepare_provider_request(
        build_console_request(
            [{"role": "user", "content": "q"}],
            mandatory=[rag_row, plain_row],
        ),
        wire_style="distinct_roles",
        model="gpt-4o",
        capacity=_capacity(None),
    )
    assert without_provenance.accounting.rag_attributed is False
    assert without_provenance.accounting.rag_context_tokens == 0

    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleTraceCaptureMode,
    )

    with_provenance = prepare_provider_request(
        build_console_request(
            [{"role": "user", "content": "q"}],
            mandatory=[rag_row, plain_row],
            message_provenance=(
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.ACTIVE_REQUEST, policy
                ),
            ),
            memory_provenance=(),
            mandatory_provenance=[
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.RAG_CONTEXT, policy
                ),
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.MANDATORY_CONTEXT, policy
                ),
            ],
            tool_provenance=(),
            capture_policy=policy,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        ),
        wire_style="distinct_roles",
        model="gpt-4o",
        capacity=_capacity(None),
    )
    accounting = with_provenance.accounting
    assert accounting.rag_attributed is True
    assert 0 < accounting.rag_context_tokens < accounting.mandatory_tokens
