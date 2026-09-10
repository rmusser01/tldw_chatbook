"""Local structured reasoning reaches canonical events and native tool transport."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ProviderThinkingDelta,
    ProviderToolCalls,
)
from tldw_chatbook.Chat.local_reasoning import resolve_reasoning_policy


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_llama_separate_reasoning_is_typed_and_not_answer(streaming):
    def respond(request):
        message = {"content": "221", "reasoning_content": "Compute the product"}
        body = {"choices": [{"message": message}]}
        if streaming:
            return httpx.Response(
                200,
                text="data: "
                + json.dumps({"choices": [{"delta": message}]})
                + "\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        result = [
            item
            async for item in gateway.stream_chat(
                ConsoleProviderResolution(
                    provider="llama_cpp",
                    execution_key="llama_cpp",
                    model="alias",
                    base_url="http://localhost:9099",
                    ready=True,
                    streaming=streaming,
                    thinking_stream_disposition="displayable",
                    thinking_round_trip_version=1,
                ),
                [{"role": "user", "content": "13*17"}],
            )
        ]
    assert "".join(x for x in result if isinstance(x, str)) == "221"
    events = [x for x in result if isinstance(x, ProviderThinkingDelta)]
    assert [(x.text, x.source_format) for x in events] == [
        ("Compute the product", "reasoning_content")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,key", [("local_vllm", "reasoning_content"), ("local_ollama", "reasoning")]
)
@pytest.mark.parametrize("streaming", [True, False])
async def test_generic_local_reasoning_is_a_typed_event(provider, key, streaming):
    message = {"content": "done", key: "Retained thought"}

    def chat_call(**kwargs):
        if not streaming:
            return {"choices": [{"message": message}]}
        return iter(
            [
                "data: "
                + json.dumps({"choices": [{"delta": {key: "Retained thought"}}]}),
                "data: " + json.dumps({"choices": [{"delta": {"content": "done"}}]}),
                "data: [DONE]",
            ]
        )

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_call)
    result = [
        item
        async for item in gateway.stream_chat(
            ConsoleProviderResolution(
                provider=provider,
                execution_key=provider,
                model="alias",
                base_url="http://localhost:9099",
                ready=True,
                streaming=streaming,
                thinking_stream_disposition="displayable",
                thinking_round_trip_version=1,
            ),
            [{"role": "user", "content": "test"}],
        )
    ]
    assert "".join(x for x in result if isinstance(x, str)) == "done"
    assert [
        (x.text, x.source_format)
        for x in result
        if isinstance(x, ProviderThinkingDelta)
    ] == [("Retained thought", key)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,key",
    [
        ("llama_cpp", "reasoning_content"),
        ("local_vllm", "reasoning_content"),
        ("local_ollama", "reasoning"),
    ],
)
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("capture_on", [True, False])
async def test_real_local_dispatcher_preserves_native_schema_and_reasoning(
    monkeypatch, provider, key, streaming, capture_on
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    name: {"api_key": "STALE-KEY-CANARY", "credential_source": "none"}
                    for name in (
                        "llama_cpp",
                        "local_llamacpp",
                        "vllm",
                        "vllm_api",
                        "local_vllm",
                        "ollama",
                        "local_ollama",
                    )
                }
            }
        ),
    )
    sent = []
    call = {
        "id": "calc-1",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression":"13*17"}'},
    }
    message = {"content": "", key: "Use calculator", "tool_calls": [call]}

    def respond(session, url, **kwargs):
        assert "Authorization" not in kwargs.get("headers", {})
        sent.append((url, kwargs["json"]))
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": message, "finish_reason": "tool_calls"}]
        }
        delta = {**message, "tool_calls": [{"index": 0, **call}]}
        response.iter_lines.return_value = iter(
            [
                "data: "
                + json.dumps(
                    {"choices": [{"delta": delta, "finish_reason": "tool_calls"}]}
                ),
                "data: [DONE]",
            ]
        )
        return response

    monkeypatch.setattr("requests.Session.post", respond)
    template = Path("Tests/fixtures/reasoning_templates/gemma4.jinja").read_text()
    resolution = ConsoleProviderResolution(
        provider=provider,
        execution_key=provider,
        model="alias",
        base_url="http://localhost:12345",
        ready=True,
        streaming=streaming,
        thinking_stream_disposition="displayable",
        thinking_round_trip_version=1,
        reasoning_replay=resolve_reasoning_policy(
            "auto", template=template, native_tools=True
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }
    ]
    gateway = ConsoleProviderGateway()
    if capture_on:
        from Tests.Chat.test_console_provider_gateway import (
            _capture_on_prepared_request,
        )
        from tldw_chatbook.Chat.console_trace_provenance import (
            ConsoleRequestRoute,
            ConsoleTraceCaptureMode,
        )

        prepared = _capture_on_prepared_request(gateway, resolution, tools=tools)
        result = [
            item
            async for item in gateway.stream_chat(
                resolution,
                prepared,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
    else:
        result = [
            item
            async for item in gateway.stream_chat(
                resolution, [{"role": "user", "content": "13*17"}], tools=tools
            )
        ]
    assert sent[0][0] == "http://localhost:12345/v1/chat/completions"
    assert sent[0][1]["tools"] == tools
    assert [x.text for x in result if isinstance(x, ProviderThinkingDelta)] == [
        "Use calculator"
    ]
    assert [
        dict(x.tool_calls[0]) for x in result if isinstance(x, ProviderToolCalls)
    ] == [call]


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/chat/completions"])
async def test_optional_template_probe_uses_server_root_and_exact_fingerprint(suffix):
    requests = []
    template = Path("Tests/fixtures/reasoning_templates/qwen38.jinja").read_text()

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"chat_template": template})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        resolution = await gateway._resolve_reasoning_history(
            ConsoleProviderResolution(
                provider="local_vllm",
                model="unrelated-alias",
                base_url="http://localhost:8000" + suffix,
                ready=True,
            ),
            {},
        )
    assert requests == ["http://localhost:8000/tokenizer_info"]
    assert resolution.reasoning_replay.mode == "all"
    assert resolution.reasoning_replay.template_family == "Qwen3.8"
    assert resolution.local_structured_thinking


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404),
        httpx.Response(200, text="bad json"),
        httpx.Response(200, text="x" * 262145),
    ],
)
async def test_optional_template_metadata_failure_preserves_server_default(response):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response)
    ) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        resolved = await gateway._resolve_reasoning_history(
            ConsoleProviderResolution(
                provider="llama_cpp",
                model="alias",
                base_url="http://localhost:9099",
                ready=True,
            ),
            {},
        )
    assert resolved.ready
    assert resolved.reasoning_replay.mode == "server_default"
    assert not resolved.reasoning_replay.native_tools


def test_final_trace_verifier_uses_paired_template_projection():
    from Tests.Chat.test_reasoning_history_policy import prepare
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode

    prepared, _, _ = prepare(family="Gemma 4")
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        execution_key="llama_cpp",
        model="alias",
        base_url="http://localhost:9099",
        ready=True,
        reasoning_replay=prepared.semantic.reasoning_replay,
    )
    gateway = ConsoleProviderGateway()
    kwargs = gateway._chat_api_kwargs_from_prepared(resolution, prepared)
    shadow = gateway._verify_trace_shadow(
        resolution, prepared, kwargs, capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON
    )
    assert shadow.available
    assert kwargs["messages_payload"][-1]["role"] == "tool"
    assert "keep going" in kwargs["messages_payload"][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["local_vllm", "local_ollama"])
@pytest.mark.parametrize("streaming", [True, False])
async def test_generic_reasoning_only_does_not_invent_answer_without_signals(
    provider, streaming
):
    key = "reasoning" if provider == "local_ollama" else "reasoning_content"
    thought = {key: "only thought", "content": ""}
    response = (
        {"choices": [{"message": thought}]}
        if not streaming
        else iter(
            ["data: " + json.dumps({"choices": [{"delta": thought}]}), "data: [DONE]"]
        )
    )
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **_: response)
    resolution = ConsoleProviderResolution(
        provider=provider,
        execution_key=provider,
        model="alias",
        base_url="http://localhost:9099",
        ready=True,
        streaming=streaming,
        local_structured_thinking=True,
    )
    result = [
        x
        async for x in gateway.stream_chat(
            resolution, [{"role": "user", "content": "test"}]
        )
    ]
    assert [x.text for x in result if isinstance(x, ProviderThinkingDelta)] == [
        "only thought"
    ]
    assert not [x for x in result if isinstance(x, str)]


@pytest.mark.parametrize(
    "provider,key",
    [
        ("llama_cpp", "reasoning"),
        ("local_vllm", "reasoning"),
        ("local_ollama", "reasoning_content"),
    ],
)
def test_structured_capture_accepts_only_declared_transport_encoding(provider, key):
    from tldw_chatbook.Chat.console_provider_gateway import _structured_local_thinking

    assert (
        _structured_local_thinking(
            {"choices": [{"message": {key: "undeclared"}}]},
            provider=provider,
            model="alias",
            protocol="chat_completions",
        )
        is None
    )
