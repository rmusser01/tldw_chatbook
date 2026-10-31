"""Local SSE control frames must never become visible assistant deltas."""

import asyncio
from io import BytesIO
import json
import threading
from types import SimpleNamespace

import pytest
import requests

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
    ProviderToolCalls,
)
from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local


pytestmark = pytest.mark.bootstrap_profile


def _wire_response(monkeypatch, body: str):
    response = requests.Response()
    response.status_code = 200
    response.encoding = "utf-8"
    response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
    response.raw = BytesIO(body.encode("utf-8"))
    posted = []

    def post(url, **kwargs):
        posted.append((url, kwargs["json"]))
        return response

    monkeypatch.setattr(
        local,
        "create_pooled_default_session",
        lambda **kwargs: SimpleNamespace(post=post),
    )
    return response, posted


def _local_stream():
    return local._chat_with_openai_compatible_local_server(
        api_base_url="http://127.0.0.1:8080/v1",
        model_name="qwen-test",
        input_data=[{"role": "user", "content": "hi"}],
        streaming=True,
    )


async def test_llamacpp_native_stream_excludes_sse_control_lines(monkeypatch):
    tool = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
    }
    tool_delta = {"choices": [{"delta": {"tool_calls": [{"index": 0, **tool}]}}]}
    _, posted = _wire_response(
        monkeypatch,
        ": heartbeat\r\n\r\nevent: heartbeat\r\ndata: {}\r\n\r\n"
        "id: private-id\nretry: 5000\nunknown: ignored\n\n"
        'data: {"choices":[{"delta":{"content":"A heartbeat"}}]}\n\n'
        ": another heartbeat\n\n"
        'data: {"choices":[{"delta":{"content":" is normal."}}]}\n\n'
        f"data: {json.dumps(tool_delta)}\n\n"
        "data: [DONE]\n\n",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        execution_key="llama_cpp",
        readiness_key="llama_cpp",
        base_url="http://127.0.0.1:8080",
        model="qwen-test",
        ready=True,
    )
    gateway = ConsoleProviderGateway()
    try:
        chunks = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}], tools=tools
            )
        ]
    finally:
        await gateway.aclose()

    assert [chunk for chunk in chunks if isinstance(chunk, str)] == [
        "A heartbeat",
        " is normal.",
    ]
    assert isinstance(chunks[-1], ProviderToolCalls)
    assert chunks[-1].tool_calls == (tool,)
    assert len(posted) == 1
    assert posted[0][1]["tools"] == tools
    assert posted[0][1]["stream"] is True


def test_local_stream_keeps_error_data_after_control_frames(monkeypatch):
    response, _ = _wire_response(
        monkeypatch,
        ": heartbeat\n\nevent: error\n"
        'data: {"error":{"message":"provider failed","type":"stream_error"}}\n\n',
    )
    stream = _local_stream()
    try:
        normalized = ConsoleProviderGateway.normalize_provider_response(stream)
        with pytest.raises(RuntimeError, match="Provider stream error"):
            next(normalized)
    finally:
        stream.close()
    assert response.raw.closed


def test_local_stream_stop_after_heartbeat_closes_response(monkeypatch):
    response, _ = _wire_response(
        monkeypatch,
        ": heartbeat\n\n"
        'data: {"choices":[{"delta":{"content":"heartbeat..."}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"later"}}]}\n\n',
    )
    stream = _local_stream()
    try:
        normalized = ConsoleProviderGateway.normalize_provider_response(stream)
        assert next(normalized) == "heartbeat..."
    finally:
        stream.close()
    assert response.raw.closed


@pytest.mark.parametrize(
    ("provider", "with_tools"),
    [("llama_cpp", True), ("strata", False), ("polystrata", False)],
)
async def test_local_stop_during_control_only_stream_closes_active_transport(
    monkeypatch, provider, with_tools
):
    reading = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    read_exited = threading.Event()
    close_calls = []

    class HeartbeatBody:
        def stream(self, _size, decode_content=True):
            try:
                yield b": keep-alive\n\n"
                reading.set()
                while not release.wait(0.005):
                    yield b": keep-alive\n\n"
            finally:
                read_exited.set()

        def close(self):
            close_calls.append(True)
            closed.set()
            release.set()

    response, posted = _wire_response(monkeypatch, "")
    response.raw = HeartbeatBody()
    workers = []

    def retain_worker(task, _close):
        workers.append(task)
        return True

    signals = ConsoleProviderStreamSignals(provider_work_callback=retain_worker)
    gateway = ConsoleProviderGateway()
    resolution = ConsoleProviderResolution(
        provider=provider,
        execution_key=provider,
        readiness_key=provider,
        base_url="http://127.0.0.1:8080",
        model="qwen-test",
        ready=True,
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "calculator", "parameters": {"type": "object"}},
        }
    ]
    stream = gateway.stream_chat(
        resolution,
        [{"role": "user", "content": "hi"}],
        tools=tools if with_tools else None,
        signals=signals,
    )
    pending = asyncio.create_task(anext(stream))
    try:
        assert await asyncio.to_thread(reading.wait, 3)
        assert not pending.done(), "control frames must not produce an assistant chunk"
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert await asyncio.to_thread(
            closed.wait, 1
        ), "Stop must close active HTTP reads"
        assert await asyncio.to_thread(read_exited.wait, 1)
        await asyncio.wait_for(asyncio.gather(*workers), 2)
        assert close_calls == [True]
        assert len(posted) == 1
        assert posted[0][1]["stream"] is True
        if with_tools:
            assert posted[0][1]["tools"] == tools
        else:
            assert "tools" not in posted[0][1]
    finally:
        release.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
        await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), 3)
        await gateway.aclose()
