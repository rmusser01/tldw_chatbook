"""Joined Kimi/GLM native-tool tests through the real Console HTTP path."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any

import pytest
import requests

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_history_budget import ProviderContinuationSidecar
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.provider_continuation import ContinuationRestoreTarget
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.LLM_Calls.moonshot import chat_with_moonshot
from tldw_chatbook.LLM_Calls.zai import chat_with_zai


_FINAL = "HOSTED-NATIVE-FINAL"
_RESULT_A = '{"expression": "6*7", "result": 42, "result_type": "int"}'
_RESULT_B = '{"expression": "8*8", "result": 64, "result_type": "int"}'
_CANCEL_MARKER = "hosted-partial-call-observed"
# StreamGate retains a fence-sized suffix, so keep visible padding after the
# checkpoint that drives the cancellation hook.
_CANCEL_STREAM_TEXT = f"{_CANCEL_MARKER}-visible-padding"
_STREAM_HOLD_PADDING = b": hold-open\n\n" * 1024
_Validator = Callable[[str, dict[str, str], dict[str, Any]], None]


@dataclass(frozen=True)
class _StalledSSE:
    prefix: bytes
    client_close_started: threading.Event


_ScriptedBody = bytes | _StalledSSE


def _sse(events: Sequence[dict[str, Any]], *, done: bool = True) -> bytes:
    wire = b"".join(
        b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
        for event in events
    )
    return wire + (b"data: [DONE]\n\n" if done else b"")


def _stream_usage(provider: str, usage: dict[str, int]) -> tuple[dict, list[dict]]:
    if provider == "moonshot":
        return {"usage": usage}, [{"choices": [], "usage": usage}]
    return {}, [{"choices": [], "usage": usage}]


def _tool_turn(provider: str) -> bytes:
    choice_usage, trailing_usage = _stream_usage(
        provider,
        {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    )
    return _sse(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "Checking both.",
                            "reasoning_content": f"PRIVATE-{provider}-TOOL-REASONING",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_A",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":"6*7"}',
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "call_B",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":"8*8"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                **(
                    {"system_fingerprint": "fp_kimi_live"}
                    if provider == "moonshot"
                    else {}
                ),
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                        **choice_usage,
                    }
                ]
            },
            *trailing_usage,
        ]
    )


def _error_tool_turn(provider: str) -> bytes:
    return _sse(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": f"PRIVATE-{provider}-ERROR-REASONING",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_error",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":"1/0"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            },
        ]
    )


def _partial_call_then_text(provider: str) -> bytes:
    return _sse(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": _CANCEL_STREAM_TEXT,
                            "reasoning_content": f"PRIVATE-{provider}-PARTIAL",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_partial",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ],
        done=False,
    )


def _assert_partial_call_precedes_marker(body: bytes) -> None:
    events = [
        json.loads(record.removeprefix(b"data: "))
        for record in body.split(b"\n\n")
        if record.startswith(b"data: {")
    ]
    matching = []
    for event in events:
        choices = event.get("choices")
        if (
            isinstance(choices, list)
            and choices
            and choices[0].get("delta", {}).get("content") == _CANCEL_STREAM_TEXT
        ):
            matching.append(event)
    assert matching
    tool_calls = matching[0]["choices"][0]["delta"].get("tool_calls")
    assert isinstance(tool_calls, list) and len(tool_calls) == 1
    assert tool_calls[0]["function"] == {
        "name": "calculator",
        "arguments": '{"expression":',
    }


def _final_turn(
    provider: str,
    *,
    text: str = _FINAL,
    reasoning: str | None = None,
) -> bytes:
    reasoning = reasoning or f"PRIVATE-{provider}-FINAL-REASONING"
    choice_usage, trailing_usage = _stream_usage(
        provider,
        {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
    )
    return _sse(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": text,
                            "reasoning_content": reasoning,
                        },
                        "finish_reason": "stop",
                        **choice_usage,
                    }
                ]
            },
            *trailing_usage,
        ]
    )


class _ScriptedServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        bodies: Sequence[_ScriptedBody],
        validators: Sequence[_Validator],
    ) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedHandler)
        self.bodies = list(bodies)
        self.validators = list(validators)
        self.requests: list[dict[str, Any]] = []
        self.validation_errors: list[str] = []
        self.stall_started = threading.Event()
        self.stall_timed_out = threading.Event()
        self._lock = threading.Lock()

    def next_response(
        self,
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[int, _ScriptedBody]:
        with self._lock:
            self.requests.append({"path": path, "headers": headers, "payload": payload})
            if not self.bodies or not self.validators:
                self.validation_errors.append("provider script exhausted")
                return 422, b'{"error":{"message":"provider script exhausted"}}'
            try:
                self.validators.pop(0)(path, headers, payload)
            except AssertionError as exc:
                message = str(exc) or "scripted request validation failed"
                self.validation_errors.append(message)
                return 422, json.dumps({"error": {"message": message}}).encode()
            return 200, self.bodies.pop(0)


class _ScriptedHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        server = self.server
        assert isinstance(server, _ScriptedServer)
        status, body = server.next_response(
            self.path,
            {key.lower(): value for key, value in self.headers.items()},
            payload,
        )
        if isinstance(body, _StalledSSE):
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(f"{len(body.prefix):X}\r\n".encode())
            self.wfile.write(body.prefix)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            server.stall_started.set()
            if not body.client_close_started.wait(timeout=2):
                server.stall_timed_out.set()
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "text/event-stream" if status == 200 else "application/json",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _scripted_server(
    bodies: Sequence[_ScriptedBody],
    validators: Sequence[_Validator],
) -> Iterator[_ScriptedServer]:
    server = _ScriptedServer(bodies, validators)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _resolution(server: _ScriptedServer, provider: str) -> ConsoleProviderResolution:
    host, port = server.server_address[:2]
    path = "/v1" if provider == "moonshot" else "/api/paas/v4"
    return ConsoleProviderResolution(
        provider=provider,
        base_url=f"http://{host}:{port}{path}",
        model="kimi-k3" if provider == "moonshot" else "glm-5.2",
        ready=True,
        readiness_key=provider,
        execution_key=provider,
        api_key="test-hosted-key",
        streaming=True,
        continuation_protocol="chat_completions",
        request_timeout=5.0,
        request_retries=0,
        request_retry_delay=0.0,
    )


def _calls() -> list[dict[str, Any]]:
    return [
        {
            "id": "call_A",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"6*7"}',
            },
        },
        {
            "id": "call_B",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"8*8"}',
            },
        },
    ]


def _validate_common(
    provider: str,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> None:
    expected_prefix = "/v1" if provider == "moonshot" else "/api/paas/v4"
    assert path == f"{expected_prefix}/chat/completions"
    assert headers["authorization"] == "Bearer test-hosted-key"
    assert headers["content-type"] == "application/json"
    assert payload["model"] == ("kimi-k3" if provider == "moonshot" else "glm-5.2")
    assert payload["stream"] is True
    assert any(
        tool.get("function", {}).get("name") == "calculator"
        for tool in payload["tools"]
    )
    if provider == "moonshot":
        assert payload["stream_options"] == {"include_usage": True}
        assert "thinking" not in payload
    else:
        assert payload["thinking"] == {"type": "enabled", "clear_thinking": False}
        assert "stream_options" not in payload


def _initial_validator(provider: str) -> _Validator:
    def validate(
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> None:
        _validate_common(provider, path, headers, payload)
        assert payload["messages"][-1] == {
            "role": "user",
            "content": "Calculate two expressions.",
        }
        assert [row for row in payload["messages"] if row["role"] == "user"] == [
            {"role": "user", "content": "Calculate two expressions."}
        ]

    return validate


def _continuation_validator(provider: str) -> _Validator:
    def validate(
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> None:
        _validate_common(provider, path, headers, payload)
        assert payload["messages"][-3:] == [
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": _calls(),
                "reasoning_content": f"PRIVATE-{provider}-TOOL-REASONING",
            },
            {"role": "tool", "tool_call_id": "call_A", "content": _RESULT_A},
            {"role": "tool", "tool_call_id": "call_B", "content": _RESULT_B},
        ]
        assert [row for row in payload["messages"] if row["role"] == "user"] == [
            {"role": "user", "content": "Calculate two expressions."}
        ]

    return validate


def _later_k3_validator() -> _Validator:
    def validate(
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> None:
        _validate_common("moonshot", path, headers, payload)
        assert payload["messages"][1:] == [
            {"role": "user", "content": "Calculate two expressions."},
            {
                "role": "assistant",
                "content": "Checking both.",
                "reasoning_content": "PRIVATE-moonshot-TOOL-REASONING",
                "tool_calls": _calls(),
            },
            {"role": "tool", "tool_call_id": "call_A", "content": _RESULT_A},
            {"role": "tool", "tool_call_id": "call_B", "content": _RESULT_B},
            {
                "role": "assistant",
                "content": _FINAL,
                "reasoning_content": "PRIVATE-moonshot-FINAL-REASONING",
            },
            {"role": "user", "content": "What did you calculate?"},
        ]

    return validate


def _zai_ordinary_validator() -> _Validator:
    def validate(
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> None:
        expected_path = "/api/paas/v4/chat/completions"
        assert path == expected_path
        assert headers["authorization"] == "Bearer test-hosted-key"
        assert payload["model"] == "glm-5.2"
        assert payload["stream"] is True
        assert payload["thinking"] == {
            "type": "enabled",
            "clear_thinking": True,
        }
        assert "tools" not in payload
        assert "stream_options" not in payload
        assert payload["messages"] == [
            {"role": "user", "content": "Ordinary follow up."}
        ]

    return validate


def _error_continuation_validator(provider: str) -> _Validator:
    def validate(
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> None:
        _validate_common(provider, path, headers, payload)
        assistant, result = payload["messages"][-2:]
        assert assistant == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_error",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"1/0"}',
                    },
                }
            ],
            "reasoning_content": f"PRIVATE-{provider}-ERROR-REASONING",
        }
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_error"
        assert result["content"].startswith("ERROR: ")
        assert "zero" in result["content"].lower()

    return validate


async def _collect_stream(stream: Any) -> list[Any]:
    return [item async for item in stream]


class _CaptureGateway(ConsoleProviderGateway):
    def __init__(self) -> None:
        super().__init__()
        self.dispatches: list[dict[str, Any]] = []

    def _chat_api_kwargs_from_prepared(self, resolution, request):
        kwargs = super()._chat_api_kwargs_from_prepared(resolution, request)
        self.dispatches.append(kwargs)
        return kwargs


class _CaptureStore(ConsoleChatStore):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.continuation_errors: list[str] = []

    def persist_provider_continuation_event(self, event: Any) -> None:
        try:
            super().persist_provider_continuation_event(event)
        except Exception as exc:
            self.continuation_errors.append(f"{type(exc).__name__}: {exc}")
            raise


@pytest.mark.allow_network
@pytest.mark.parametrize("provider", ["moonshot", "zai"])
def test_console_runs_two_native_calls_with_private_continuation(
    tmp_path: Any,
    provider: str,
) -> None:
    bodies = [_tool_turn(provider), _final_turn(provider)]
    validators = [_initial_validator(provider), _continuation_validator(provider)]
    if provider == "moonshot":
        bodies.append(
            _final_turn(
                provider,
                text="K3-LATER-ANSWER",
                reasoning="PRIVATE-moonshot-LATER-REASONING",
            )
        )
        validators.append(_later_k3_validator())
    else:
        bodies.append(
            _final_turn(
                provider,
                text="GLM-ORDINARY-ANSWER",
                reasoning="PRIVATE-zai-ORDINARY-REASONING",
            )
        )
        validators.append(_zai_ordinary_validator())
    with _scripted_server(bodies, validators) as server:
        db = AgentRunsDB(tmp_path / f"{provider}.db", client_id="hosted-native")
        chat_db = CharactersRAGDB(
            tmp_path / f"{provider}-chat.db", f"hosted-native-{provider}"
        )
        store = _CaptureStore(persistence=ChatPersistenceService(chat_db))
        session = store.create_session(title="Hosted native continuation")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Calculate two expressions.",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        gateway = _CaptureGateway()
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=gateway,
        )

        _run_id, outcome = bridge.run_reply(
            conversation_id=f"{provider}-conversation",
            session_id=session.id,
            resolution=_resolution(server, provider),
            assistant_message_id=assistant.id,
            model="kimi-k3" if provider == "moonshot" else "glm-5.2",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "Calculate two expressions."}],
            should_cancel=lambda: False,
            native_tools_enabled=True,
        )
        later_outcome = None
        ordinary_items = None
        if provider == "moonshot":
            checkpoint = store.get_message(assistant.id).provider_continuation
            assert checkpoint is not None
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="What did you calculate?",
                persist=True,
            )
            later_assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="",
                persist=True,
            )
            _later_run_id, later_outcome = bridge.run_reply(
                conversation_id=f"{provider}-conversation",
                session_id=session.id,
                resolution=_resolution(server, provider),
                assistant_message_id=later_assistant.id,
                model="kimi-k3",
                session_system_prompt="",
                agent_messages=[
                    {"role": "user", "content": "Calculate two expressions."},
                    {
                        "role": "assistant",
                        "content": _FINAL,
                        "_owner": assistant.id,
                    },
                    {"role": "user", "content": "What did you calculate?"},
                ],
                should_cancel=lambda: False,
                native_tools_enabled=True,
                continuation_sidecar=(
                    ProviderContinuationSidecar(assistant.id, checkpoint),
                ),
                continuation_target=ContinuationRestoreTarget(
                    provider="moonshot",
                    protocol="chat_completions",
                    model="kimi-k3",
                    api_base_url=_resolution(server, provider).base_url,
                ),
                continuation_owner_key="_owner",
            )
        else:
            ordinary_items = asyncio.run(
                _collect_stream(
                    gateway.stream_chat(
                        _resolution(server, provider),
                        [{"role": "user", "content": "Ordinary follow up."}],
                    )
                )
            )
        chat_db.close_connection()

    owner = store.get_message(assistant.id)
    assert outcome.status == "done", {
        "steps": [step.summary for step in outcome.steps],
        "validation_errors": server.validation_errors,
        "requests": server.requests,
        "dispatch_messages": [
            dispatch.get("messages_payload") for dispatch in gateway.dispatches
        ],
        "continuation_counts": [
            len(dispatch.get("provider_continuations", ()))
            for dispatch in gateway.dispatches
        ],
    }
    assert outcome.final_text == _FINAL
    assert outcome.total_tokens == 65
    assert owner.content == _FINAL
    assert owner.provider_continuation is not None
    assert owner.provider_continuation.state == "complete"
    assert owner.provider_continuation.rounds[0].reasoning_blocks == (
        f"PRIVATE-{provider}-TOOL-REASONING",
    )
    if provider == "moonshot":
        assert owner.provider_continuation.rounds[-1].reasoning_blocks == (
            "PRIVATE-moonshot-FINAL-REASONING",
        )
    if provider == "moonshot":
        assert later_outcome is not None
        assert later_outcome.status == "done", {
            "steps": [step.summary for step in later_outcome.steps],
            "validation_errors": server.validation_errors,
            "request_count": len(server.requests),
            "continuation_errors": store.continuation_errors,
        }
        assert later_outcome.final_text == "K3-LATER-ANSWER"
    else:
        assert ordinary_items == ["GLM-ORDINARY-ANSWER"]
    assert len(server.requests) == 3
    assert server.validation_errors == []
    assert server.validators == []
    assert server.bodies == []
    assert "PRIVATE-" not in owner.content


@pytest.mark.allow_network
@pytest.mark.parametrize("provider", ["moonshot", "zai"])
def test_hosted_tool_error_continues_structurally(
    tmp_path: Any,
    provider: str,
) -> None:
    with _scripted_server(
        [_error_tool_turn(provider), _final_turn(provider, text="ERROR-RECOVERED")],
        [_initial_validator(provider), _error_continuation_validator(provider)],
    ) as server:
        db = AgentRunsDB(tmp_path / f"{provider}-error.db", client_id="hosted-error")
        chat_db = CharactersRAGDB(
            tmp_path / f"{provider}-error-chat.db", f"hosted-error-{provider}"
        )
        store = _CaptureStore(persistence=ChatPersistenceService(chat_db))
        session = store.create_session(title="Hosted native error")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Calculate two expressions.",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=_CaptureGateway(),
        )

        _run_id, outcome = bridge.run_reply(
            conversation_id=f"{provider}-error-conversation",
            session_id=session.id,
            resolution=_resolution(server, provider),
            assistant_message_id=assistant.id,
            model="kimi-k3" if provider == "moonshot" else "glm-5.2",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "Calculate two expressions."}],
            should_cancel=lambda: False,
            native_tools_enabled=True,
        )
        owner = store.get_message(assistant.id)
        chat_db.close_connection()

    assert outcome.status == "done"
    assert outcome.final_text == "ERROR-RECOVERED"
    assert len(server.requests) == 2
    assert server.validation_errors == []
    assert store.continuation_errors == []
    assert owner.provider_continuation is not None
    call = owner.provider_continuation.rounds[0].calls[0]
    assert call.state == "failed"
    assert call.result is not None
    assert call.result.value.startswith("ERROR: ")


@pytest.mark.allow_network
@pytest.mark.parametrize("provider", ["moonshot", "zai"])
@pytest.mark.parametrize("malformed", ["duplicate_ids", "out_of_order"])
def test_hosted_invalid_tool_history_fails_before_server_advancement(
    provider: str,
    malformed: str,
) -> None:
    calls = _calls()
    if malformed == "duplicate_ids":
        calls[1]["id"] = "call_A"
    results = [
        {"role": "tool", "tool_call_id": "call_A", "content": _RESULT_A},
        {"role": "tool", "tool_call_id": "call_B", "content": _RESULT_B},
    ]
    if malformed == "out_of_order":
        results.reverse()
    messages = [
        {"role": "user", "content": "Calculate two expressions."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": calls,
        },
        *results,
    ]

    with _scripted_server([], []) as server:
        resolution = _resolution(server, provider)
        handler = chat_with_moonshot if provider == "moonshot" else chat_with_zai
        with pytest.raises(ChatBadRequestError):
            handler(
                input_data=messages,
                model=resolution.model,
                api_key="test-hosted-key",
                streaming=True,
                api_base_url=resolution.base_url,
                request_retries=0,
            )

    assert server.requests == []


def test_partial_call_fixture_rejects_text_only_mutation() -> None:
    with pytest.raises(AssertionError):
        _assert_partial_call_precedes_marker(
            _final_turn("moonshot", text=_CANCEL_STREAM_TEXT)
        )


@pytest.mark.allow_network
@pytest.mark.parametrize("provider", ["moonshot", "zai"])
def test_hosted_partial_call_cancellation_never_executes(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    executions: list[str] = []
    real_execute = CalculatorTool.execute

    async def recording_execute(self: CalculatorTool, expression: str) -> dict:
        executions.append(expression)
        return await real_execute(self, expression)

    monkeypatch.setattr(CalculatorTool, "execute", recording_execute)
    cancelled = threading.Event()
    marker_observed = threading.Event()
    client_close_started = threading.Event()
    close_calls: list[int] = []
    body = _partial_call_then_text(provider)
    _assert_partial_call_precedes_marker(body)
    real_append_stream_chunk = ConsoleChatStore.append_stream_chunk
    real_response_close = requests.Response.close

    def recording_stream_chunk(
        store: ConsoleChatStore, message_id: str, chunk: str
    ) -> Any:
        message = real_append_stream_chunk(store, message_id, chunk)
        if _CANCEL_MARKER in chunk:
            marker_observed.set()
            cancelled.set()
        return message

    monkeypatch.setattr(ConsoleChatStore, "append_stream_chunk", recording_stream_chunk)

    with _scripted_server(
        [_StalledSSE(body + _STREAM_HOLD_PADDING, client_close_started)],
        [_initial_validator(provider)],
    ) as server:
        resolution = _resolution(server, provider)

        def recording_response_close(response: requests.Response) -> None:
            if str(response.url).startswith(resolution.base_url):
                close_calls.append(id(response))
                client_close_started.set()
            real_response_close(response)

        monkeypatch.setattr(requests.Response, "close", recording_response_close)
        db = AgentRunsDB(tmp_path / f"{provider}-cancel.db", client_id="hosted-cancel")
        chat_db = CharactersRAGDB(
            tmp_path / f"{provider}-cancel-chat.db", f"hosted-cancel-{provider}"
        )
        store = _CaptureStore(persistence=ChatPersistenceService(chat_db))
        session = store.create_session(title="Hosted native cancellation")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Calculate two expressions.",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=_CaptureGateway(),
        )
        _run_id, outcome = bridge.run_reply(
            conversation_id=f"{provider}-cancel-conversation",
            session_id=session.id,
            resolution=resolution,
            assistant_message_id=assistant.id,
            model=resolution.model or "",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "Calculate two expressions."}],
            should_cancel=cancelled.is_set,
            native_tools_enabled=True,
        )
        owner = store.get_message(assistant.id)
        chat_db.close_connection()

    assert outcome.status == "cancelled"
    assert marker_observed.is_set()
    assert server.stall_started.is_set()
    assert not server.stall_timed_out.is_set()
    assert len(server.requests) == 1
    assert len(close_calls) == 1
    assert executions == []
    assert owner.provider_continuation is None
    assert store.continuation_errors == []
    assert not any(step.kind in {"tool_call", "tool_result"} for step in outcome.steps)
