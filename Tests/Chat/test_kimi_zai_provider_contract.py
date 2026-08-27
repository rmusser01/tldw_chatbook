from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from tldw_chatbook.Chat import console_provider_gateway as gateway_module
from tldw_chatbook.Chat import Chat_Functions
from tldw_chatbook.Chat.console_history_budget import ProviderContinuationSidecar
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ContinuationRestoreTarget,
    ContinuationResult,
    ProviderContinuationCheckpoint,
    parse_provider_continuation_json,
)
from tldw_chatbook.LLM_Calls import moonshot, zai
import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ContinuationEventContext,
    ModelTurn,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
    HostedChatTurn,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.Chat.console_agent_bridge import (
    _ModelCallLifeline,
    _StreamingModelAdapter,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleMessageRole,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _checkpoint(*, provider: str = "moonshot") -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider=provider,  # type: ignore[arg-type]
        protocol="chat_completions",
        model="kimi-k3" if provider == "moonshot" else "glm-5.2",
        api_base_url=(
            "https://api.moonshot.ai/v1"
            if provider == "moonshot"
            else "https://api.z.ai/api/paas/v4"
        ),
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="answer",
                reasoning_blocks=("PRIVATE-REASONING-CANARY",),
                calls=(),
            ),
        ),
    )


def _active_checkpoint(provider: str) -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider=provider,  # type: ignore[arg-type]
        protocol="chat_completions",
        model="kimi-k3" if provider == "moonshot" else "glm-5.2",
        api_base_url=(
            "https://api.moonshot.ai/v1"
            if provider == "moonshot"
            else "https://api.z.ai/api/paas/v4"
        ),
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("PRIVATE-REASONING-CANARY",),
                calls=(
                    ContinuationCall(
                        call_id="call_1",
                        name="calculator",
                        arguments='{"expression":"2+2"}',
                        state="pending",
                    ),
                ),
            ),
        ),
    )


def test_terminal_metadata_is_typed_and_repr_safe() -> None:
    metadata_type = getattr(gateway_module, "ProviderTurnMetadata", None)

    assert metadata_type is not None
    metadata = metadata_type(
        finish_reason="stop",
        provider_continuation=_checkpoint(),
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    )
    sentinel = gateway_module.ProviderToolCalls((), metadata=metadata)
    hosted_turn = HostedChatTurn(
        text="answer",
        tool_calls=(),
        assistant_message={
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "PRIVATE-REASONING-CANARY",
        },
        finish_reason="stop",
        reasoning_content="PRIVATE-REASONING-CANARY",
        usage={"private_usage": "PRIVATE-REASONING-CANARY"},
    )

    assert sentinel.metadata is metadata
    assert "PRIVATE-REASONING-CANARY" not in repr(metadata)
    assert "PRIVATE-REASONING-CANARY" not in repr(sentinel)
    assert "PRIVATE-REASONING-CANARY" not in repr(hosted_turn)


def test_moonshot_nonstream_tool_turn_exposes_private_typed_candidate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        moonshot,
        "resolve_moonshot_request",
        lambda **_kwargs: moonshot.MoonshotResolution(
            provider="moonshot",
            model="kimi-k3",
            api_key="secret",
            base_url="https://api.moonshot.ai/v1",
            timeout=90.0,
            retries=3,
            retry_delay=1.0,
            streaming=False,
        ),
    )
    monkeypatch.setattr(
        moonshot,
        "hosted_chat_request",
        lambda **_kwargs: HostedChatTurn(
            text="checking",
            tool_calls=(
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                },
            ),
            assistant_message={
                "role": "assistant",
                "content": "checking",
                "reasoning_content": "PRIVATE-REASONING-CANARY",
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
            finish_reason="tool_calls",
            reasoning_content="PRIVATE-REASONING-CANARY",
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        ),
    )

    response = moonshot.chat_with_moonshot(
        input_data=[{"role": "user", "content": "2+2?"}],
        model="kimi-k3",
        api_key="secret",
        streaming=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert isinstance(response, dict)
    assert "reasoning_content" not in response["choices"][0]["message"]
    candidate = response.provider_continuation  # type: ignore[attr-defined]
    assert candidate.provider == "moonshot"
    assert candidate.state == "active"
    assert candidate.checkpoint_revision == 1
    assert candidate.rounds[0].assistant_content == "checking"
    assert candidate.rounds[0].reasoning_blocks == ("PRIVATE-REASONING-CANARY",)
    assert candidate.rounds[0].calls[0].call_id == "call_1"
    assert candidate.rounds[0].calls[0].state == "pending"
    assert "PRIVATE-REASONING-CANARY" not in repr(response)


def test_zai_nonstream_tool_turn_exposes_private_typed_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        zai,
        "resolve_zai_request",
        lambda **_kwargs: zai.ZAIResolution(
            provider="zai",
            model="glm-5.2",
            api_key="secret",
            base_url="https://api.z.ai/api/paas/v4",
            timeout=90.0,
            retries=3,
            retry_delay=1.0,
            streaming=False,
        ),
    )
    monkeypatch.setattr(
        zai,
        "owned_json_post",
        lambda **_kwargs: {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "checking",
                        "reasoning_content": "PRIVATE-REASONING-CANARY",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": {"expression": "2+2"},
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )

    response = zai.chat_with_zai(
        input_data=[{"role": "user", "content": "2+2?"}],
        model="glm-5.2",
        api_key="secret",
        streaming=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert isinstance(response, dict)
    assert "reasoning_content" not in response["choices"][0]["message"]
    candidate = response.provider_continuation  # type: ignore[attr-defined]
    assert candidate.provider == "zai"
    assert candidate.state == "active"
    assert candidate.checkpoint_revision == 1
    assert candidate.rounds[0].reasoning_blocks == ("PRIVATE-REASONING-CANARY",)
    assert candidate.rounds[0].calls[0].arguments == '{"expression":"2+2"}'
    assert "PRIVATE-REASONING-CANARY" not in repr(response)


def test_moonshot_stream_candidate_requires_clean_terminal_exhaustion(
    monkeypatch,
) -> None:
    resolution = moonshot.MoonshotResolution(
        provider="moonshot",
        model="kimi-k3",
        api_key="secret",
        base_url="https://api.moonshot.ai/v1",
        timeout=90.0,
        retries=3,
        retry_delay=1.0,
        streaming=True,
    )
    records = iter(
        [
            SSERecord(
                event=None,
                data=(
                    '{"choices":[{"index":0,"delta":{"role":"assistant",'
                    '"reasoning_content":"PRIVATE-REASONING-CANARY",'
                    '"tool_calls":[{"index":0,"id":"call_1","type":"function",'
                    '"function":{"name":"calculator",'
                    '"arguments":"{\\"expression\\":\\"2+2\\"}"}}]},'
                    '"finish_reason":"tool_calls"}]}'
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
    monkeypatch.setattr(
        moonshot, "resolve_moonshot_request", lambda **_kwargs: resolution
    )
    monkeypatch.setattr(
        moonshot,
        "hosted_chat_request",
        lambda **_kwargs: HostedChatStream(
            records, finish_policy=moonshot.MoonshotFinishPolicy()
        ),
    )

    response = moonshot.chat_with_moonshot(
        input_data=[{"role": "user", "content": "2+2?"}],
        api_key="secret",
        streaming=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    with pytest.raises(HostedChatProtocolError):
        _ = response.provider_continuation  # type: ignore[union-attr]
    events = list(response)
    candidate = response.provider_continuation  # type: ignore[union-attr]

    assert "reasoning_content" not in events[0]["choices"][0]["delta"]
    assert candidate is not None
    assert candidate.rounds[-1].calls[0].call_id == "call_1"
    assert candidate.rounds[-1].reasoning_blocks == ("PRIVATE-REASONING-CANARY",)


def test_zai_stream_candidate_requires_clean_terminal_exhaustion(monkeypatch) -> None:
    resolution = zai.ZAIResolution(
        provider="zai",
        model="glm-5.2",
        api_key="secret",
        base_url="https://api.z.ai/api/paas/v4",
        timeout=90.0,
        retries=3,
        retry_delay=1.0,
        streaming=True,
    )
    records = iter(
        [
            SSERecord(
                event=None,
                data=(
                    '{"choices":[{"index":0,"delta":{"role":"assistant",'
                    '"reasoning_content":"PRIVATE-REASONING-CANARY",'
                    '"tool_calls":[{"index":0,"id":"call_1","type":"function",'
                    '"function":{"name":"calculator",'
                    '"arguments":"{\\"expression\\":\\"2+2\\"}"}}]},'
                    '"finish_reason":"tool_calls"}]}'
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
    monkeypatch.setattr(zai, "resolve_zai_request", lambda **_kwargs: resolution)
    monkeypatch.setattr(zai, "owned_json_post", lambda **_kwargs: records)

    response = zai.chat_with_zai(
        input_data=[{"role": "user", "content": "2+2?"}],
        api_key="secret",
        streaming=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    with pytest.raises(HostedChatProtocolError):
        _ = response.provider_continuation  # type: ignore[union-attr]
    list(response)
    candidate = response.provider_continuation  # type: ignore[union-attr]

    assert candidate is not None
    assert candidate.provider == "zai"
    assert candidate.rounds[-1].calls[0].call_id == "call_1"


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("moonshot", "kimi-k3", "https://api.moonshot.ai/v1"),
        ("zai", "glm-5.2", "https://api.z.ai/api/paas/v4"),
    ],
)
def test_prepared_kwargs_pin_hosted_base_and_private_checkpoints(
    provider: str,
    model: str,
    base_url: str,
) -> None:
    rounds = (
        [
            {
                "assistant_content": "answer",
                "reasoning_blocks": ["PRIVATE-REASONING-CANARY"],
                "calls": [],
            }
        ]
        if provider == "moonshot"
        else [
            {
                "assistant_content": "",
                "reasoning_blocks": ["PRIVATE-REASONING-CANARY"],
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
        ]
    )
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": provider,
            "protocol": "chat_completions",
            "model": model,
            "api_base_url": base_url,
            "state": "complete",
            "rounds": rounds,
        }
    )
    resolution = gateway_module.ConsoleProviderResolution(
        provider=provider,
        base_url=base_url,
        model=model,
        ready=True,
        readiness_key=provider,
        execution_key=provider,
        api_key="PRIVATE-API-KEY-CANARY",
        streaming=True,
        continuation_protocol="chat_completions",
    )
    gateway = gateway_module.ConsoleProviderGateway(environ={})
    prepared = gateway.prepare_chat_request(
        resolution,
        [
            {"_owner": "a1", "role": "assistant", "content": "answer"},
            {"_owner": "u2", "role": "user", "content": "next"},
        ],
        continuation_target=ContinuationRestoreTarget(
            provider=provider,
            model=model,
            protocol="chat_completions",
            api_base_url=base_url,
        ),
        continuation_sidecar=(ProviderContinuationSidecar("a1", checkpoint),),
        continuation_owner_key="_owner",
    )

    kwargs = gateway._chat_api_kwargs_from_prepared(resolution, prepared)

    assert kwargs["api_base_url"] == base_url
    assert kwargs["provider_continuations"] == [checkpoint]
    assert "api_mode" not in kwargs
    assert "PRIVATE-API-KEY-CANARY" not in repr(prepared)


@pytest.mark.parametrize("provider", ["moonshot", "zai"])
def test_dispatcher_forwards_only_hosted_provider_continuations(
    monkeypatch,
    provider: str,
) -> None:
    captured: dict[str, object] = {}

    def handler(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, provider, handler)
    checkpoint = _active_checkpoint(provider)

    result = Chat_Functions.chat_api_call(
        api_endpoint=provider,
        messages_payload=[{"role": "user", "content": "next"}],
        provider_continuations=(checkpoint,),
    )

    assert result == {"ok": True}
    assert captured["provider_continuations"] == (checkpoint,)


@pytest.mark.asyncio
async def test_gateway_emits_one_final_typed_metadata_sentinel() -> None:
    checkpoint = _active_checkpoint("moonshot")
    turn = HostedChatTurn(
        text="checking",
        tool_calls=(
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"2+2"}',
                },
            },
        ),
        assistant_message={
            "role": "assistant",
            "content": "checking",
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
        finish_reason="tool_calls",
        reasoning_content="PRIVATE-REASONING-CANARY",
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    )
    response = moonshot.MoonshotResponse(
        {
            "choices": [
                {
                    "index": 0,
                    "message": turn.assistant_message,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": turn.usage,
        },
        terminal_turn=turn,
        provider_continuation=checkpoint,
    )
    gateway = gateway_module.ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: response,
        environ={},
    )
    resolution = gateway_module.ConsoleProviderResolution(
        provider="moonshot",
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k3",
        ready=True,
        execution_key="moonshot",
        api_key="secret",
        streaming=False,
        continuation_protocol="chat_completions",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate.",
                "parameters": {"type": "object"},
            },
        }
    ]

    items = [
        item
        async for item in gateway.stream_chat(
            resolution,
            [{"role": "user", "content": "2+2?"}],
            tools=tools,
        )
    ]

    assert items[0] == "checking"
    assert len(items) == 2
    sentinel = items[-1]
    assert isinstance(sentinel, gateway_module.ProviderToolCalls)
    assert sentinel.metadata is not None
    assert sentinel.metadata.finish_reason == "tool_calls"
    assert sentinel.metadata.provider_continuation == checkpoint
    assert sentinel.metadata.usage == turn.usage


@pytest.mark.asyncio
async def test_gateway_cancellation_never_reads_terminal_provider_metadata() -> None:
    class CancelledProviderStream:
        def __init__(self) -> None:
            self._first = True
            self.closed = threading.Event()
            self.exhausted = threading.Event()
            self.terminal_accessed = threading.Event()
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._first:
                self._first = False
                return {"choices": [{"delta": {"content": "partial"}}]}
            self.closed.wait(timeout=5)
            self.exhausted.set()
            raise StopIteration

        @property
        def terminal_turn(self):
            self.terminal_accessed.set()
            raise AssertionError("cancelled streams have no terminal metadata")

        def close(self) -> None:
            self.close_calls += 1
            self.closed.set()

    response = CancelledProviderStream()
    gateway = gateway_module.ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: response,
        environ={},
    )
    resolution = gateway_module.ConsoleProviderResolution(
        provider="moonshot",
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k3",
        ready=True,
        execution_key="moonshot",
        api_key="secret",
        streaming=True,
        continuation_protocol="chat_completions",
    )
    stream = gateway.stream_chat(
        resolution,
        [{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert await anext(stream) == "partial"
    await stream.aclose()
    assert await asyncio.to_thread(response.exhausted.wait, 1)
    await asyncio.sleep(0.05)

    assert response.close_calls == 1
    assert response.terminal_accessed.is_set() is False


def test_streaming_adapter_carries_terminal_candidate_into_model_turn(tmp_path) -> None:
    checkpoint = _active_checkpoint("moonshot")
    metadata = gateway_module.ProviderTurnMetadata(
        finish_reason="tool_calls",
        provider_continuation=checkpoint,
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    )
    calls = (
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"2+2"}',
            },
        },
    )

    class Gateway:
        async def stream_chat(self, _resolution, _messages, **_kwargs):
            yield "checking"
            yield gateway_module.ProviderToolCalls(calls, metadata=metadata)

    class Resolution:
        provider = "moonshot"
        execution_key = "moonshot"
        model = "kimi-k3"

    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="2+2?")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    lifeline = _ModelCallLifeline("kimi-zai-contract-loop")
    lifeline.start()
    try:
        adapter = _StreamingModelAdapter(
            store=store,
            provider_gateway=Gateway(),
            resolution=Resolution(),
            assistant_message_id=assistant.id,
            should_cancel=lambda: False,
            loop=lifeline.loop,
            native_tools=True,
        )
        registry = ToolCatalogRegistry()
        registry.register_provider(BuiltinToolProvider())
        service = AgentService(
            db=AgentRunsDB(tmp_path / "runs.db", client_id="contract"),
            registry=registry,
            chat_call=adapter.chat_call,
        )
        turn = service._make_call_model(
            AgentConfig(model="kimi-k3", system_prompt="system", native_tools=True),
            "moonshot",
            [],
        )([{"role": "user", "content": "2+2?"}], ())
    finally:
        lifeline.shutdown()

    assert turn.provider_continuation == checkpoint
    assert turn.tokens == 3
    assert turn.assistant_message is not None
    assert "PRIVATE-REASONING-CANARY" not in repr(turn.assistant_message)


def test_runtime_passes_exact_transitioned_checkpoint_to_next_model_call() -> None:
    first = _active_checkpoint("moonshot")
    seen: list[ProviderContinuationCheckpoint | None] = []
    call = ToolCall(
        name="calculator",
        args={"expression": "2+2"},
        call_id="call_1",
        raw_arguments='{"expression":"2+2"}',
    )

    def call_model_with_continuation(_messages, _active, current):
        seen.append(current)
        if current is None:
            return ModelTurn(
                text="",
                tool_calls=(call,),
                assistant_message={
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
                provider_continuation=first,
            )
        assert current.checkpoint_revision == 3
        assert current.rounds[-1].calls[0].state == "completed"
        return ModelTurn(
            text="4",
            provider_continuation=ProviderContinuationCheckpoint(
                schema_version=1,
                checkpoint_revision=4,
                provider="moonshot",
                protocol="chat_completions",
                model="kimi-k3",
                api_base_url="https://api.moonshot.ai/v1",
                state="complete",
                rounds=(
                    *current.rounds,
                    ContinuationRound(
                        assistant_content="4",
                        reasoning_blocks=("PRIVATE-FINAL-REASONING",),
                        calls=(),
                    ),
                ),
            ),
        )

    events = []
    outcome = run_agent_loop(
        AgentConfig(
            model="kimi-k3",
            system_prompt="system",
            allowed_tools=("calculator",),
        ),
        [{"role": "user", "content": "2+2?"}],
        [
            ToolSchema(
                id="builtin:calculator",
                name="calculator",
                description="Calculate.",
                parameters={"type": "object"},
            )
        ],
        LoopDeps(
            call_model=lambda _messages, _active: pytest.fail(
                "continuation-aware model seam was bypassed"
            ),
            call_model_with_continuation=call_model_with_continuation,
            invoke_tool=lambda _call: ToolResult(ok=True, content="4"),
            spawn=lambda _task: ToolResult(ok=False, error="unused"),
            find_tools=lambda _query: [],
            load_schemas=lambda _ids: [],
            should_cancel=lambda: False,
            clock=lambda: 0.0,
            continuation_context=ContinuationEventContext(
                "owner", "run", "primary", "persistent"
            ),
            persist_provider_continuation=events.append,
        ),
    )

    assert outcome.status == "done"
    assert outcome.final_text == "4"
    assert seen[0] is None
    assert seen[1] is not None


def test_agent_service_forwards_current_checkpoint_as_one_owner_group(tmp_path) -> None:
    captured: dict[str, object] = {}
    current = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=3,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("PRIVATE-REASONING-CANARY",),
                calls=(
                    ContinuationCall(
                        call_id="call_1",
                        name="calculator",
                        arguments='{"expression":"2+2"}',
                        state="completed",
                        result=ContinuationResult("4"),
                    ),
                ),
            ),
        ),
    )

    def chat_call(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "4"}}],
            "usage": {"total_tokens": 3},
        }

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=AgentRunsDB(tmp_path / "service.db", client_id="contract"),
        registry=registry,
        chat_call=chat_call,
        prepare_provider_continuation_request=True,
    )
    call_model = service._make_call_model(
        AgentConfig(model="kimi-k3", system_prompt="system", native_tools=True),
        "moonshot",
        [],
        continuation_owner_key="_owner",
        continuation_owner_message_id="owner-1",
    )

    turn = call_model(
        [
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
                "_owner": "owner-1",
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
        ],
        (),
        current,
    )

    groups = captured["continuation_groups"]
    assert isinstance(groups, tuple)
    assert len(groups) == 1
    assert groups[0].owner_message_id == "owner-1"
    assert groups[0].checkpoint == current
    assert captured["messages_payload"][1]["_owner"] == "owner-1"
    assert turn.tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("moonshot", "kimi-k3", "https://api.moonshot.ai/v1"),
        ("zai", "glm-5.2", "https://api.z.ai/api/paas/v4"),
    ],
)
async def test_console_freezes_hosted_transport_policy_before_config_mutation(
    provider: str,
    model: str,
    base_url: str,
) -> None:
    settings: dict[str, object] = {
        "api_key": "PRIVATE-API-KEY-CANARY",
        "model": model,
        "api_base_url": base_url,
        "timeout": 41.5,
        "retries": 2,
        "retry_delay": 0.25,
    }
    gateway = gateway_module.ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {provider: settings}},
        environ={},
    )
    resolution = await gateway.resolve_for_send(
        gateway_module.ConsoleProviderSelection(provider=provider)
    )
    settings.update(timeout=999, retries=99, retry_delay=99)
    prepared = gateway.prepare_chat_request(
        resolution,
        [{"role": "user", "content": "hello"}],
    )

    kwargs = gateway._chat_api_kwargs_from_prepared(resolution, prepared)

    assert resolution.ready is True
    assert kwargs["request_timeout"] == 41.5
    assert kwargs["request_retries"] == 2
    assert kwargs["request_retry_delay"] == 0.25
    assert "api_mode" not in kwargs


@pytest.mark.parametrize("provider", ["moonshot", "zai"])
def test_strict_hosted_adapter_uses_frozen_transport_policy(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    module = moonshot if provider == "moonshot" else zai
    chat = module.chat_with_moonshot if provider == "moonshot" else module.chat_with_zai
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    provider: {
                        "timeout": 999,
                        "retries": 99,
                        "retry_delay": 99,
                    }
                }
            }
        ),
    )
    if provider == "moonshot":
        monkeypatch.setattr(
            module,
            "hosted_chat_request",
            lambda **kwargs: (
                captured.update(kwargs)
                or HostedChatTurn(
                    text="answer",
                    tool_calls=(),
                    assistant_message={"role": "assistant", "content": "answer"},
                    finish_reason="stop",
                )
            ),
        )
    else:
        monkeypatch.setattr(
            module,
            "owned_json_post",
            lambda **kwargs: (
                captured.update(kwargs)
                or {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
        )

    chat(
        input_data=[{"role": "user", "content": "hello"}],
        api_key="secret",
        streaming=False,
        request_timeout=41.5,
        request_retries=2,
        request_retry_delay=0.25,
    )

    config = captured["config"]
    assert config.timeout == 41.5  # type: ignore[union-attr]
    assert config.retries == 2  # type: ignore[union-attr]
    assert config.retry_delay == 0.25  # type: ignore[union-attr]
