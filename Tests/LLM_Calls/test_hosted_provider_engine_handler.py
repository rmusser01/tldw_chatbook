"""Engine handler factory: chat_with_* closure, continuations, metrics (ADR-179).

Pins the Task 7 contracts of ``build_hosted_chat_handler``:

- The returned callable carries the ``chat_with_zai`` signature minus the
  provider-invented ``do_sample``/``request_id`` parameters, so the shared
  per-provider param map can drive every engine preset.
- resolve -> payload -> transport -> response wiring mirrors
  ``chat_with_zai`` (monkeypatched ``owned_json_post``, the zai-suite style):
  the non-streaming path returns an OpenAI-shaped response with terminal
  metadata, and the streaming path returns visible chunks plus terminal
  turn usage.
- The continuation candidate round-trips through the canonical checkpoint
  format with the record's key/protocol (Databricks admitted by the Task 7
  ``_PAIRINGS`` registration in ``Chat.provider_continuation``), and is
  skipped entirely when the record ships no continuation protocol.
- The response message keeps ``reasoning_content`` only for
  ``"displayable"`` records (zai pops unconditionally; the engine decision
  pops when the disposition is not displayable).
- The metric counters replicate the ``chat_with_zai``/``chat_with_moonshot``
  compatibility-wrapper shapes (request/success/error counters, response
  time histograms) with ``provider=record.key`` identity.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.LLM_Calls import hosted_provider_engine
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    SSERecord,
)
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    HostedProviderResponse,
    HostedProviderStream,
    build_hosted_chat_handler,
)
from tldw_chatbook.provider_registry import DATABRICKS


def _resolution(streaming: bool = False, model: str = "databricks-gpt-4o"):
    return HostedProviderResolution(
        provider="databricks",
        model=model,
        api_key="secret",
        base_url="https://dbc-1.cloud.databricks.com/openai/v1",
        timeout=90.0,
        retries=3,
        retry_delay=5.0,
        streaming=streaming,
    )


def _tool_call_response() -> dict[str, object]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                # Some gateways return dict arguments; the
                                # response layer coerces them to
                                # deterministic JSON strings.
                                "arguments": {"tz": "utc"},
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch, response: object
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_request(**kwargs: object) -> object:
        captured.update(kwargs)
        return response

    monkeypatch.setattr(hosted_provider_engine, "owned_json_post", fake_request)
    return captured


def test_factory_signature_accepts_chat_api_call_kwargs() -> None:
    handler = build_hosted_chat_handler(DATABRICKS)
    params = set(inspect.signature(handler).parameters)
    expected = {
        "input_data", "model", "api_key", "system_message", "temp", "maxp",
        "streaming", "max_tokens", "tools", "custom_prompt_arg",
        "api_base_url", "tool_choice", "stop", "response_format", "user",
        "reasoning_effort", "provider_continuations", "request_timeout",
        "request_retries", "request_retry_delay",
    }
    assert expected <= params
    assert "do_sample" not in params
    assert "request_id" not in params


def test_continuation_roundtrip_builds_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = build_hosted_chat_handler(DATABRICKS)
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda record, **_kwargs: _resolution(),
    )
    _patch_transport(monkeypatch, _tool_call_response())

    result = handler(
        input_data=[{"role": "user", "content": "what time is it?"}],
        api_key="secret",
        streaming=False,
    )

    assert isinstance(result, HostedProviderResponse)
    checkpoint = result.provider_continuation
    assert checkpoint is not None
    assert checkpoint.provider == "databricks"
    assert checkpoint.protocol == "chat_completions"
    assert checkpoint.state == "active"
    assert checkpoint.model == "databricks-gpt-4o"
    assert checkpoint.api_base_url == "https://dbc-1.cloud.databricks.com/openai/v1"
    assert len(checkpoint.rounds) == 1
    call = checkpoint.rounds[0].calls[0]
    assert call.name == "get_time"
    assert call.call_id == "call_1"
    assert call.arguments == '{"tz":"utc"}'
    assert call.state == "pending"


def test_non_streaming_tool_call_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = build_hosted_chat_handler(DATABRICKS)
    resolve_kwargs: dict[str, object] = {}

    def fake_resolve(record, **kwargs: object):
        resolve_kwargs.update(kwargs)
        return _resolution()

    monkeypatch.setattr(
        hosted_provider_engine, "resolve_hosted_request", fake_resolve
    )
    captured = _patch_transport(monkeypatch, _tool_call_response())

    result = handler(
        input_data=[{"role": "user", "content": "what time is it?"}],
        model="databricks-gpt-4o",
        api_key="secret",
        api_base_url="https://dbc-1.cloud.databricks.com",
        streaming=False,
        request_timeout=30.0,
    )

    # Resolution wiring: caller kwargs map onto the resolver's explicit slot.
    assert resolve_kwargs["explicit_api_key"] == "secret"
    assert resolve_kwargs["explicit_base_url"] == "https://dbc-1.cloud.databricks.com"
    assert resolve_kwargs["explicit_model"] == "databricks-gpt-4o"
    assert resolve_kwargs["explicit_timeout"] == 30.0
    # Transport wiring: record identity and resolved endpoint, chat route.
    config = captured["config"]
    assert config.provider == "databricks"
    assert config.base_url == "https://dbc-1.cloud.databricks.com/openai/v1"
    assert captured["route"] == "chat/completions"
    assert captured["streaming"] is False
    payload = captured["payload"]
    assert payload["model"] == "databricks-gpt-4o"
    assert payload["stream"] is False
    # Response shape: the OpenAI dict plus terminal metadata kept off it.
    assert isinstance(result, HostedProviderResponse)
    message = result["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_time", "arguments": '{"tz":"utc"}'},
        }
    ]
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result["usage"] == {
        "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7
    }
    assert result.terminal_turn.finish_reason == "tool_calls"
    assert result.provider_continuation is not None


def test_streaming_visible_chunks_and_terminal_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderStreamSignals,
    )

    handler = build_hosted_chat_handler(DATABRICKS)
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda record, **_kwargs: _resolution(streaming=True),
    )
    payloads = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "PRIVATE"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Hi"}}]},
        {"choices": [{"index": 0, "delta": {"content": " there"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4}},
    ]
    records = iter(
        [SSERecord(event=None, data=json.dumps(item)) for item in payloads]
        + [SSERecord(event=None, data="[DONE]")]
    )
    _patch_transport(monkeypatch, records)

    stream = handler(
        input_data=[{"role": "user", "content": "hi"}],
        api_key="secret",
        streaming=True,
    )

    assert isinstance(stream, HostedProviderStream)
    events = list(stream)
    signals = ConsoleProviderStreamSignals()
    chunks = list(
        ConsoleProviderGateway.normalize_provider_response(
            iter(events), signals=signals
        )
    )
    assert chunks == ["Hi", " there"]
    assert signals.synthetic_fallback_emitted is False
    assert signals.usage_payloads() == [{"prompt_tokens": 8, "completion_tokens": 4}]
    # Databricks' ignored disposition: private reasoning never leaks into a
    # visible delta (empty content placeholder instead).
    assert all(
        "reasoning_content" not in event["choices"][0]["delta"]
        for event in events
        if event.get("choices")
    )
    assert stream.terminal_turn.finish_reason == "stop"
    assert stream.terminal_turn.usage == {"prompt_tokens": 8, "completion_tokens": 4}
    assert stream.provider_continuation is None


def test_response_message_reasoning_disposition_gates_public_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canned = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "reasoning_content": "PRIVATE-TRACE",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    def run(record):
        monkeypatch.setattr(
            hosted_provider_engine,
            "resolve_hosted_request",
            lambda _record, **_kwargs: _resolution(),
        )
        _patch_transport(monkeypatch, canned)
        handler = build_hosted_chat_handler(record)
        return handler(
            input_data=[{"role": "user", "content": "hi"}],
            api_key="secret",
            streaming=False,
        )

    proprietary = run(replace(DATABRICKS, reasoning_disposition="proprietary"))
    assert "reasoning_content" not in proprietary["choices"][0]["message"]
    assert proprietary.terminal_turn.reasoning_content == "PRIVATE-TRACE"

    displayable = run(replace(DATABRICKS, reasoning_disposition="displayable"))
    assert displayable["choices"][0]["message"]["reasoning_content"] == "PRIVATE-TRACE"


def test_continuation_protocol_none_skips_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = replace(DATABRICKS, continuation_protocol=None)
    handler = build_hosted_chat_handler(record)
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda _record, **_kwargs: _resolution(),
    )
    _patch_transport(monkeypatch, _tool_call_response())

    result = handler(
        input_data=[{"role": "user", "content": "what time is it?"}],
        api_key="secret",
        streaming=False,
    )

    # Tool-call turn, but the preset ships no continuation protocol: no
    # checkpoint is built for it.
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result.provider_continuation is None


def test_handler_metrics_match_wrapper_counter_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters: list[tuple[object, ...]] = []
    histograms: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        hosted_provider_engine,
        "log_counter",
        lambda name, labels=None: counters.append((name, labels)),
    )
    monkeypatch.setattr(
        hosted_provider_engine,
        "log_histogram",
        lambda name, value, labels=None: histograms.append((name, value, labels)),
    )
    handler = build_hosted_chat_handler(DATABRICKS)
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda record, **_kwargs: _resolution(),
    )
    _patch_transport(monkeypatch, _tool_call_response())

    handler(
        input_data=[{"role": "user", "content": "hi"}],
        model="databricks-gpt-4o",
        api_key="secret",
        streaming=False,
    )

    labels = {"model": "databricks-gpt-4o", "streaming": "False"}
    assert counters == [
        ("databricks_api_request", labels),
        ("databricks_api_success", labels),
    ]
    assert len(histograms) == 1
    assert histograms[0][0] == "databricks_api_response_time"
    assert histograms[0][2] == labels

    # Error path: the error counter carries the exception type label and a
    # response-time histogram, then the failure re-raises.
    counters.clear()
    histograms.clear()
    monkeypatch.setattr(
        hosted_provider_engine,
        "owned_json_post",
        lambda **_kwargs: (_ for _ in ()).throw(
            HostedChatProtocolError("PRIVATE-PROVIDER-PAYLOAD")
        ),
    )
    with pytest.raises(ChatProviderError) as exc_info:
        handler(
            input_data=[{"role": "user", "content": "hi"}],
            model="databricks-gpt-4o",
            api_key="secret",
            streaming=False,
        )
    assert "PRIVATE-PROVIDER-PAYLOAD" not in str(exc_info.value)
    assert counters[0] == ("databricks_api_request", labels)
    assert counters[1][0] == "databricks_api_error"
    assert counters[1][1] == {**labels, "error_type": "ChatProviderError"}
    assert histograms[0][0] == "databricks_api_error_response_time"
    assert histograms[0][2] == labels


def test_resolve_hosted_engine_request_alias_forwards_to_resolver() -> None:
    resolution = hosted_provider_engine.resolve_hosted_engine_request(
        DATABRICKS,
        explicit_api_key="secret",
        explicit_base_url="https://dbc-1.cloud.databricks.com",
        explicit_model="databricks-gpt-4o",
        app_config={"api_settings": {}},
        environ={},
    )
    assert resolution.provider == "databricks"
    assert resolution.model == "databricks-gpt-4o"
    assert resolution.base_url == "https://dbc-1.cloud.databricks.com/openai/v1"
    assert resolution.api_key == "secret"


def test_resolve_hosted_engine_request_alias_matches_private_signature():
    """PIN (green on arrival): public alias forwards the full private surface."""
    import inspect
    from tldw_chatbook.LLM_Calls import hosted_provider_engine as engine

    assert list(inspect.signature(engine.resolve_hosted_engine_request).parameters) == (
        list(inspect.signature(engine.resolve_hosted_request).parameters)
    )
