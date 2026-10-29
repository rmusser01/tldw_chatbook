"""A self-hosted server that refuses streaming still answers (TASK-37421).

Since TASK-33620.10 every Console provider streams by default. A self-hosted
OpenAI-compatible server that cannot stream (a Custom endpoint, vLLM, ...)
answered the first streamed request of every chat with HTTP 400, and the
reply failed with advice about the model. llama.cpp already retried such a
request once without streaming; owner decision 2026-10-29: every self-hosted
and Custom OpenAI-compatible server gets the same one-time retry. Cloud
providers do not -- a silent second paid call is not wanted there -- and
their refused stream names the Streaming setting instead (TASK-37180).

These tests are joined end to end: the real ``ConsoleProviderGateway``
resolves each provider from config and dispatches through the real
``chat_api_call`` and its real adapter (``chat_with_custom_openai``, the
strict hosted engine for a custom endpoint, ``chat_with_vllm``,
``chat_with_anthropic``, a hosted engine preset), driven by the real
``ConsoleAgentBridge`` a Console send uses. The server is a loopback OpenAI-compatible endpoint that refuses
``"stream": true`` the way such servers do. Only the socket is local.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

#: The adapters read provider settings through the guarded config loader;
#: under the per-test redirect that admission fails closed with
#: RecoveryRequired("raw_source_selection_changed"). bootstrap_profile keeps
#: the collection-time sandbox profile (still isolated from the user's).
pytestmark = pytest.mark.bootstrap_profile

_REPLY = "REFUSED-STREAM-RETRY-REPLY-SENTINEL"
_MODEL = "local-model"
_USAGE = {"prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26}
#: What a server that cannot stream says (llama.cpp-style error envelope).
_REFUSAL = "Streaming is not supported by this server."
_STREAMING_ADVICE = "turn Streaming off"


class _OpenAICompatibleServer(ThreadingHTTPServer):
    """Loopback chat-completions endpoint that cannot stream."""

    daemon_threads = True

    def __init__(
        self,
        *,
        stream_status: int = 400,
        empty_stream: bool = False,
        refuse_all: bool = False,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.stream_status = stream_status
        self.empty_stream = empty_stream
        self.refuse_all = refuse_all
        self.stream_flags: list[bool] = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - http.server contract
        body = json.dumps(
            {"object": "list", "data": [{"id": _MODEL, "object": "model"}]}
        ).encode()
        self._send(200, "application/json", body)

    def do_POST(self) -> None:  # noqa: N802 - http.server contract
        server = self.server
        assert isinstance(server, _OpenAICompatibleServer)
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        streamed = bool(payload.get("stream"))
        server.stream_flags.append(streamed)
        if streamed and server.empty_stream:
            self._send(200, "text/event-stream", b"data: [DONE]\n\n")
            return
        if streamed or server.refuse_all:
            body = json.dumps(
                {
                    "error": {
                        "code": server.stream_status,
                        "message": _REFUSAL,
                        "type": "invalid_request_error",
                    }
                }
            ).encode()
            self._send(server.stream_status, "application/json", body)
            return
        body = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": _MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _REPLY},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            }
        ).encode()
        self._send(200, "application/json", body)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _server(**kwargs: Any) -> Iterator[_OpenAICompatibleServer]:
    server = _OpenAICompatibleServer(**kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _gateway_config(provider: str, base_url: str) -> dict[str, Any]:
    """The config a user writes for ``provider`` at ``base_url``."""
    if provider == "custom-ep:box":
        return {
            "custom_endpoints": {
                "box": {
                    "display_name": "Box",
                    "family": "openai_compatible",
                    "base_url": base_url,
                }
            }
        }
    if provider in {"custom", "vllm"}:
        return {"api_settings": {provider: {"api_url": base_url, "model": _MODEL}}}
    return {
        "api_settings": {
            provider: {"api_base_url": base_url, "api_key": "test-key-not-real-0000"}
        }
    }


def _resolve(
    provider: str, base_url: str, *, streaming: bool = True
) -> tuple[ConsoleProviderGateway, ConsoleProviderResolution]:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: _gateway_config(provider, base_url),
        environ={},
    )
    resolution = asyncio.run(
        gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=provider,
                base_url=base_url,
                explicit_model=_MODEL,
                streaming=streaming,
            )
        )
    )
    assert resolution.ready, resolution.visible_copy
    assert resolution.streaming is streaming
    return gateway, resolution


def _run_console_reply(
    tmp_path: Path,
    gateway: ConsoleProviderGateway,
    resolution: ConsoleProviderResolution,
) -> tuple[Any, ConsoleChatStore, ConsoleProviderStreamSignals]:
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    signals = ConsoleProviderStreamSignals()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="refused-stream"),
        store=store,
        provider_gateway=gateway,
    )
    _run_id, outcome = bridge.run_reply(
        conversation_id="refused-stream",
        session_id=session.id,
        resolution=resolution,
        assistant_message_id=assistant.id,
        model=resolution.model,
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hello"}],
        should_cancel=lambda: False,
        provider_stream_signals=signals,
    )
    return outcome, store, signals


def _copy(outcome: Any) -> str:
    return ConsoleChatController._agent_failure_visible_copy(outcome) or ""


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "provider", ["custom", "custom-ep:box", "vllm"], ids=lambda p: p
)
def test_self_hosted_refused_stream_is_retried_once_without_streaming(
    tmp_path, provider
):
    with _server() as server:
        gateway, resolution = _resolve(provider, server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status == "done", _copy(outcome)
    assert outcome.final_text == _REPLY
    # One refused streamed request, then exactly one retry without streaming.
    assert server.stream_flags == [True, False]
    # The retry's usage is recorded once: the refused request reported none.
    assert outcome.total_tokens == _USAGE["total_tokens"]
    payloads = signals.usage_payloads()
    assert len(payloads) == 1, payloads
    assert payloads[0]["prompt_tokens"] == _USAGE["prompt_tokens"]
    assert payloads[0]["completion_tokens"] == _USAGE["completion_tokens"]


@pytest.mark.loopback_network
def test_self_hosted_empty_stream_is_retried_once_without_streaming(tmp_path):
    """A server that accepts ``stream`` but sends no text cannot stream."""
    with _server(empty_stream=True) as server:
        gateway, resolution = _resolve("custom", server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status == "done", _copy(outcome)
    assert outcome.final_text == _REPLY
    assert server.stream_flags == [True, False]
    assert outcome.total_tokens == _USAGE["total_tokens"]
    assert len(signals.usage_payloads()) == 1


@pytest.mark.loopback_network
@pytest.mark.parametrize("status", [401, 429], ids=["auth", "rate-limit"])
def test_self_hosted_auth_or_rate_limit_refusal_is_not_retried(tmp_path, status):
    """A retry cannot fix a refused key or a rate limit: one request only."""
    with _server(stream_status=status) as server:
        gateway, resolution = _resolve("custom", server.base_url)
        outcome, _store, _signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status != "done"
    assert server.stream_flags[0] is True
    assert False not in server.stream_flags
    assert _STREAMING_ADVICE not in _copy(outcome)


@pytest.mark.loopback_network
def test_self_hosted_retry_that_also_fails_shows_the_retry_failure(tmp_path):
    """Refused with streaming and without: the non-streamed failure is shown.

    Streaming was not the problem, so the copy does not name it.
    """
    with _server(refuse_all=True) as server:
        gateway, resolution = _resolve("custom", server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status != "done"
    assert server.stream_flags == [True, False]
    assert signals.usage_payloads() == []
    copy = _copy(outcome)
    assert "400" in copy and "model picker" in copy, copy
    assert _STREAMING_ADVICE not in copy, copy


@pytest.mark.loopback_network
@pytest.mark.parametrize("provider", ["anthropic", "zenmux"], ids=lambda p: p)
def test_cloud_refused_stream_is_not_retried_and_names_streaming(tmp_path, provider):
    """A cloud 400 is usually a real request error: no silent second call.

    The failure names the Streaming setting beside the model advice
    (TASK-37180). ``zenmux`` stands for the strict hosted-engine presets.
    """
    with _server() as server:
        gateway, resolution = _resolve(provider, server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status != "done"
    assert server.stream_flags == [True]
    assert signals.usage_payloads() == []
    copy = _copy(outcome)
    assert "400" in copy, copy
    assert _REFUSAL in copy, copy
    assert "model picker" in copy, copy
    assert _STREAMING_ADVICE in copy, copy


@pytest.mark.loopback_network
def test_cloud_refusal_of_a_request_that_was_not_streamed_does_not_name_streaming(
    tmp_path,
):
    """TASK-37180 AC#2: the advice belongs to a streamed request only."""
    with _server(refuse_all=True) as server:
        gateway, resolution = _resolve("anthropic", server.base_url, streaming=False)
        outcome, _store, _signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status != "done"
    assert server.stream_flags == [False]
    copy = _copy(outcome)
    assert "400" in copy and "model picker" in copy, copy
    assert _STREAMING_ADVICE not in copy, copy
