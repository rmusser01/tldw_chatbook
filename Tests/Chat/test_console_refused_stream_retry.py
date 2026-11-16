"""A self-hosted server that refuses streaming still answers (TASK-37421).

Since TASK-33620.10 every Console provider streams by default. A self-hosted
OpenAI-compatible server that cannot stream (a Custom endpoint, vLLM, ...)
answered the first streamed request of every chat with HTTP 400, and the
reply failed with advice about the model. llama.cpp already retried such a
request once without streaming; owner decision 2026-10-29: every self-hosted
and Custom OpenAI-compatible server gets the same one-time retry, a named
custom endpoint run by the hosted engine included (TASK-37762, once its
traced request aligned). Cloud providers do not -- a silent second paid call
is not wanted there -- and their refused stream names the Streaming setting
instead (TASK-37180).

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
from tldw_chatbook.Chat.provider_usage import ProviderUsage
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
        empty_stream_usage: dict[str, Any] | None = None,
        refuse_all: bool = False,
        ignore_stream: bool = False,
        json_tools: bool = False,
        tool_first: bool = False,
        tool_on_retry: bool = False,
        finish_records: bool = False,
        broken_stream: str | None = None,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.stream_status = stream_status
        self.empty_stream = empty_stream
        self.empty_stream_usage = empty_stream_usage
        self.refuse_all = refuse_all
        self.ignore_stream = ignore_stream
        self.json_tools = json_tools
        self.tool_first = tool_first
        self.tool_on_retry = tool_on_retry
        self.finish_records = finish_records
        self.broken_stream = broken_stream
        self.payloads: list[dict[str, Any]] = []
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
        server.payloads.append(payload)
        streamed = bool(payload.get("stream"))
        server.stream_flags.append(streamed)
        if streamed and server.tool_on_retry and len(server.stream_flags) == 3:
            self._send_stream(_REPLY, _USAGE)
            return
        if streamed and server.tool_first and len(server.stream_flags) == 1:
            tool_text = (
                "```tool_call\n"
                + json.dumps({"name": "calculator", "arguments": {"expression": "6*7"}})
                + "\n```"
            )
            self._send_stream(tool_text, _USAGE)
            return
        if streamed and server.broken_stream == "malformed_json":
            self._send(
                200, "text/event-stream", b'data: {"choices": [\n\ndata: [DONE]\n\n'
            )
            return
        if streamed and server.broken_stream == "read_failure":
            # The body stops short of its declared length and the connection
            # closes: the client's read of the stream fails mid-record.
            partial = b'data: {"choi'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(partial) + 512))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(partial)
            self.wfile.flush()
            self.close_connection = True
            return
        if streamed and server.empty_stream:
            # No text; with ``empty_stream_usage``, the usage of the prompt it
            # read (llama.cpp and vLLM report it when asked for
            # ``include_usage``).
            self._send_stream(None, server.empty_stream_usage)
            return
        if (streamed and not server.ignore_stream) or server.refuse_all:
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
                        "message": (
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "local-call",
                                        "type": "function",
                                        "function": {
                                            "name": "calculator",
                                            "arguments": '{"expression":"1+1"}',
                                        },
                                    }
                                ],
                            }
                            if server.json_tools
                            else {
                                "role": "assistant",
                                "content": (
                                    '```tool_call\n{"name":"calculator","arguments":{"expression":"6*7"}}\n```'
                                    if server.tool_on_retry
                                    else _REPLY
                                ),
                            }
                        ),
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
            indent=2,
        ).encode()
        self._send(200, "application/json; charset=utf-8", body)

    def _send_stream(self, content: str | None, usage: dict[str, Any] | None) -> None:
        """Stream ``content`` and ``usage`` (either may be absent), then [DONE].

        ``finish_records`` sends what an OpenAI-compatible server streams: a
        finish record, then the usage record. The strict hosted engine a
        named custom endpoint runs through reads a stream without a finish
        record as unreadable (TASK-37762); the short form the legacy handlers
        accept is kept byte for byte.
        """
        server = self.server
        assert isinstance(server, _OpenAICompatibleServer)
        meta = {"id": "chatcmpl-empty", "object": "chat.completion.chunk", "model": _MODEL}
        events: list[dict[str, Any]] = []
        if server.finish_records:
            if content is not None:
                events.append(
                    {**meta, "choices": [{"index": 0, "delta": {"content": content}}]}
                )
            events.append(
                {**meta, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            )
            if usage is not None:
                events.append({**meta, "choices": [], "usage": usage})
        elif content is not None:
            events.append({"choices": [{"delta": {"content": content}}], "usage": usage})
        elif usage is not None:
            events.append({**meta, "choices": [], "usage": usage})
        body = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
        self._send(200, "text/event-stream", body + b"data: [DONE]\n\n")

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
    "provider", ["custom", "vllm", "custom-ep:box"], ids=lambda p: p
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
@pytest.mark.parametrize(
    "provider", ["custom", "vllm", "custom-ep:box"], ids=lambda p: p
)
def test_self_hosted_empty_stream_is_retried_once_without_streaming(tmp_path, provider):
    """A server that accepts ``stream`` but sends no text cannot stream.

    ``custom-ep:box`` runs through the hosted engine, which reads a stream
    that ends with no record as unreadable rather than as an empty reply.
    """
    with _server(empty_stream=True) as server:
        gateway, resolution = _resolve(provider, server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status == "done", _copy(outcome)
    assert outcome.final_text == _REPLY
    assert server.stream_flags == [True, False]
    assert outcome.total_tokens == _USAGE["total_tokens"]
    assert len(signals.usage_payloads()) == 1


@pytest.mark.loopback_network
@pytest.mark.parametrize("shape", ["finish_record", "whole_json"])
def test_named_custom_endpoint_stream_with_nothing_readable_is_retried_once(
    tmp_path, shape
):
    """TASK-37762: the hosted engine's other two ways to stream no text.

    ``finish_record``: the server finishes cleanly with no text, which the
    engine reports as an empty terminal turn (finish reason, no content and
    no tool call). ``whole_json``: the server ignores ``stream`` and answers
    whole, which the engine cannot read as a stream. Either is asked once
    more without streaming, and the reply's usage is recorded once.
    """
    server_kwargs = (
        {"empty_stream": True, "finish_records": True}
        if shape == "finish_record"
        else {"ignore_stream": True}
    )
    with _server(**server_kwargs) as server:
        gateway, resolution = _resolve("custom-ep:box", server.base_url)
        assert resolution.execution_key == "custom-hosted"
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status == "done", _copy(outcome)
    assert outcome.final_text == _REPLY
    assert server.stream_flags == [True, False]
    assert outcome.total_tokens == _USAGE["total_tokens"]
    assert signals.usage_payloads() == [_USAGE]


@pytest.mark.loopback_network
@pytest.mark.parametrize("shape", ["malformed_json", "read_failure"])
def test_named_custom_endpoint_stream_failing_for_another_reason_is_not_asked_again(
    tmp_path, shape
):
    """TASK-37762 review: only an empty or unreadable stream is asked again.

    The hosted engine raises the same protocol error type for every stream it
    cannot use. A stream that ENDS with nothing in it (no terminal record, or
    a ``[DONE]`` with no finish reason) is what a server that cannot stream
    sends; a malformed record or a failed read is a broken reply, which a
    Custom slot is not asked again for either. Asking again without
    streaming would be a silent second generation for a failure streaming did
    not cause.
    """
    with _server(broken_stream=shape) as server:
        gateway, resolution = _resolve("custom-ep:box", server.base_url)
        assert resolution.execution_key == "custom-hosted"
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status != "done"
    assert server.stream_flags, "the endpoint was never asked"
    assert False not in server.stream_flags, server.stream_flags
    assert signals.usage_payloads() == []


@pytest.mark.loopback_network
@pytest.mark.parametrize("shape", ["empty", "whole_json"])
def test_cloud_hosted_preset_stream_with_nothing_readable_is_not_asked_again(
    tmp_path, shape
):
    """TASK-37762 review: the empty-stream retry stays with retried servers.

    ``zenmux`` streams through the same hosted engine as a named custom
    endpoint and raises the same error for a stream with nothing readable,
    but it is a cloud preset: a second request there is a silent second paid
    call, so it is never asked again without streaming.
    """
    server_kwargs = (
        {"empty_stream": True} if shape == "empty" else {"ignore_stream": True}
    )
    with _server(**server_kwargs) as server:
        gateway, resolution = _resolve("zenmux", server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status != "done"
    assert server.stream_flags, "the provider was never asked"
    assert False not in server.stream_flags, server.stream_flags
    assert signals.usage_payloads() == []


@pytest.mark.loopback_network
def test_empty_stream_usage_is_not_merged_into_the_retry_usage(tmp_path):
    """The retry's usage is the reply's usage, never key-merged with the
    empty stream's.

    Review of TASK-37421: the empty stream reported its prompt with a cache
    read, the retry reported its own prompt and output. Merging the two
    payloads key by key billed the retry's prompt beside the empty call's
    ``cached_tokens`` -- a cache read the retry never made -- and the bill
    read as one call that was mostly cached.
    """
    empty_usage = {
        "prompt_tokens": 21,
        "completion_tokens": 0,
        "total_tokens": 21,
        "prompt_tokens_details": {"cached_tokens": 16},
    }
    with _server(empty_stream=True, empty_stream_usage=empty_usage) as server:
        gateway, resolution = _resolve("custom", server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)

    assert outcome.status == "done", _copy(outcome)
    assert server.stream_flags == [True, False]
    payloads = signals.usage_payloads()
    assert payloads == [empty_usage, _USAGE]
    usage = ProviderUsage.from_provider_payload(
        payloads[0], provider="custom", model=_MODEL
    ).plus(
        ProviderUsage.from_provider_payload(
            payloads[1], provider="custom", model=_MODEL
        )
    )
    assert usage.cache_read == 16
    assert usage.uncached_input == 26
    assert usage.output == 5
    assert usage.total_tokens == 47
    # The agent's weighted budget is not raw token usage. Verify through its
    # existing strict normalizer and pricing, never by dropping the first leg.
    from tldw_chatbook.Agents.agent_service import budget_weighted_tokens

    assert outcome.total_tokens == budget_weighted_tokens(
        {
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 5,
                "total_tokens": 47,
                "prompt_tokens_details": {"cached_tokens": 16},
            }
        },
        provider="custom",
        model=_MODEL,
    )


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
    assert "400" in copy and _REFUSAL in copy, copy
    assert _STREAMING_ADVICE not in copy, copy


@pytest.mark.loopback_network
@pytest.mark.parametrize("provider", ["anthropic", "zenmux"], ids=lambda p: p)
def test_cloud_refused_stream_is_not_retried_and_names_streaming(tmp_path, provider):
    """A cloud 400 is usually a real request error: no silent second call.

    The failure names the Streaming setting and its actual settings locations
    (TASK-37180). ``zenmux`` stands for the strict hosted-engine presets: the
    hosted engine a named custom endpoint shares does not make a cloud
    preset retried (TASK-37762).
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
    assert "Chat settings" in copy and "config.toml" in copy, copy
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
    assert "400" in copy and _REFUSAL in copy, copy
    assert _STREAMING_ADVICE not in copy, copy


@pytest.mark.loopback_network
@pytest.mark.parametrize("provider", ["custom", "vllm"])
def test_whole_json_on_stream_preserves_answer_usage_without_regeneration(
    tmp_path, provider
):
    with _server(ignore_stream=True) as server:
        gateway, resolution = _resolve(provider, server.base_url)
        outcome, _store, signals = _run_console_reply(tmp_path, gateway, resolution)
    assert outcome.status == "done", _copy(outcome)
    assert outcome.final_text == _REPLY
    assert server.stream_flags == [True]
    assert signals.usage_payloads() == [_USAGE]
    assert outcome.total_tokens == 26
