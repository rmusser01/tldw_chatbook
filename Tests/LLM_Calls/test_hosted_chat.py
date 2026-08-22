"""Contracts for the provider-neutral hosted Chat-Completions boundary."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import traceback
from typing import Any

import pytest
import requests

import tldw_chatbook.LLM_Calls.hosted_chat as hosted_chat
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedHTTPTransportConfig,
    HostedChatProtocolError,
    HostedChatStream,
    HostedChatTurn,
    hosted_chat_request,
    normalize_hosted_chat_base_url,
    normalize_hosted_chat_response,
    owned_json_post,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import (
    HostedSSEReadError,
    OwnedSSEStream,
    SSERecord,
)
from tldw_chatbook.Utils.sensitive_llm_logging import sensitive_llm_request


class _FinishPolicy:
    def validate_finish(
        self,
        *,
        finish_reason: object,
        has_text: bool,
        has_calls: bool,
    ) -> str:
        if finish_reason not in {"stop", "tool_calls", "length"}:
            raise HostedChatProtocolError("finish state is malformed")
        if (finish_reason == "tool_calls") != has_calls:
            raise HostedChatProtocolError("finish state conflicts with calls")
        if finish_reason == "stop" and not has_text:
            raise HostedChatProtocolError("finish state has no text")
        return finish_reason

    def validate_reasoning_content(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HostedChatProtocolError("reasoning is malformed")
        return value


_POLICY = _FinishPolicy()


def test_hosted_chat_request_composes_nonstream_transport_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    monkeypatch.setattr(hosted_chat, "owned_json_post", lambda **_kwargs: response)

    result = hosted_chat_request(
        config=HostedHTTPTransportConfig(
            provider="moonshot",
            base_url="https://example.test/v1",
            api_key="secret",
            timeout=10,
            retries=0,
            retry_delay=0,
        ),
        payload={"model": "kimi-k3", "messages": [], "stream": False},
        streaming=False,
        finish_policy=_POLICY,
    )

    assert isinstance(result, HostedChatTurn)
    assert result.text == "hello"


def test_hosted_chat_request_composes_stream_transport_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = iter([SSERecord(event=None, data="[DONE]")])
    monkeypatch.setattr(hosted_chat, "owned_json_post", lambda **_kwargs: records)

    result = hosted_chat_request(
        config=HostedHTTPTransportConfig(
            provider="moonshot",
            base_url="https://example.test/v1",
            api_key="secret",
            timeout=10,
            retries=0,
            retry_delay=0,
        ),
        payload={"model": "kimi-k3", "messages": [], "stream": True},
        streaming=True,
        finish_policy=_POLICY,
    )

    assert isinstance(result, HostedChatStream)


class _ScriptedHostedServer(ThreadingHTTPServer):
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedHostedHandler)
        self.actions = actions
        self.requests: list[dict[str, Any]] = []


class _ScriptedHostedHandler(BaseHTTPRequestHandler):
    server: _ScriptedHostedServer

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.requests.append(
            {"path": self.path, "headers": dict(self.headers), "body": body}
        )
        action = self.server.actions.pop(0)
        delay = action.get("delay", 0)
        if delay:
            time.sleep(delay)
        status = action.get("status", 200)
        response_body = action.get("body", b"{}")
        headers = action.get("headers", {})
        self.send_response(status)
        self.send_header("Content-Type", action.get("content_type", "application/json"))
        self.send_header(
            "Content-Length",
            str(len(response_body) + action.get("extra_content_length", 0)),
        )
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(response_body)
        self.wfile.flush()
        if action.get("extra_content_length"):
            self.close_connection = True

    def log_message(self, _format: str, *args: object) -> None:
        del args


@contextmanager
def _scripted_hosted_server(
    actions: list[dict[str, Any]],
) -> Any:
    server = _ScriptedHostedServer(actions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _TrackingSession(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0
        self.response_close_calls: list[int] = []

    def post(self, *args: Any, **kwargs: Any) -> requests.Response:
        response = super().post(*args, **kwargs)
        real_close = response.close
        close_index = len(self.response_close_calls)
        self.response_close_calls.append(0)

        def tracked_close() -> None:
            self.response_close_calls[close_index] += 1
            real_close()

        response.close = tracked_close  # type: ignore[method-assign]
        return response

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _track_transport_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_TrackingSession]:
    sessions: list[_TrackingSession] = []

    def create_session() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(hosted_chat, "create_default_session", create_session)
    return sessions


def _transport_config(base_url: str, **overrides: object) -> HostedHTTPTransportConfig:
    values: dict[str, object] = {
        "provider": "moonshot",
        "base_url": base_url,
        "api_key": "SECRET-TRANSPORT-CANARY",
        "timeout": 0.2,
        "retries": 0,
        "retry_delay": 0.0,
    }
    values.update(overrides)
    return HostedHTTPTransportConfig(**values)  # type: ignore[arg-type]


def _tool_delta(
    *,
    index: int,
    arguments: str,
    call_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    tool: dict[str, Any] = {"index": index, "function": {"arguments": arguments}}
    if call_id is not None:
        tool.update({"id": call_id, "type": "function"})
    if name is not None:
        tool["function"]["name"] = name
    return tool


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "https://api.example/v1"),
        ("https://api.example/v1", "https://api.example/v1"),
        ("https://api.example/v1/", "https://api.example/v1"),
        (
            "https://api.example/tenant/v1/chat/completions",
            "https://api.example/tenant/v1",
        ),
        (
            "http://[2001:db8::1]:8080/tenant/v1",
            "http://[2001:db8::1]:8080/tenant/v1",
        ),
        (
            "https://api.example/tenant%20alpha/v1",
            "https://api.example/tenant%20alpha/v1",
        ),
        (
            "https://api.example/tenant%2520alpha/v1",
            "https://api.example/tenant%2520alpha/v1",
        ),
        (
            "https://api.example/responses-extra/v1",
            "https://api.example/responses-extra/v1",
        ),
    ],
)
def test_normalize_hosted_chat_base_url_accepts_structural_bases(
    value: object,
    expected: str,
) -> None:
    original = value

    assert (
        normalize_hosted_chat_base_url(value, default="https://api.example/v1")
        == expected
    )
    assert value == original


@pytest.mark.parametrize(
    "value",
    [
        7,
        "",
        "   ",
        " https://api.example/v1",
        "https://api.example/v1 ",
        "https://api.example/v1\n",
        "api.example/v1",
        "ftp://api.example/v1",
        "https:///v1",
        "https://user:secret@api.example/v1",
        "https://api.example/v1?tenant=a",
        "https://api.example/v1#fragment",
        "https://api.example\\evil/v1",
        "https://api.example//v1",
        "https://api.example/v1/./chat/completions",
        "https://api.example/v1/../chat/completions",
        "https://api.example/v1/%zz",
        "https://api.example/v1/responses",
        "https://api.example/v1/RESPONSES",
        "https://api.example/v1/models",
        "https://api.example/v1/chat/COMPLETIONS",
        "https://api.example/v1/CHAT/completions",
        "https://api.example/v1/chat/completions/extra",
        "https://api.example/v1/chat/completions/chat/completions",
        "https://api.example/v1/responses/chat/completions",
        "https://api.example/v1/chat/completions/responses",
        "https://api.example/v1/api%2Fv2",
        "https://api.example/v1/api%252fv2",
        "https://api.example/v1/api%5Cv2",
        "https://api.example/v1/%2e/chat/completions",
        "https://api.example/v1/%252e%252e/chat/completions",
        "https://api.example/v1/res%70onses",
        "https://api.example/v1/chat/%63ompletions",
    ],
)
def test_normalize_hosted_chat_base_url_rejects_ambiguous_or_unsafe_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="Hosted Chat API base URL") as exc_info:
        normalize_hosted_chat_base_url(value, default="https://api.example/v1")

    assert "secret" not in str(exc_info.value)


def test_normalize_hosted_chat_base_url_enforces_preparse_length_cap() -> None:
    value = "https://api.example/" + ("a" * 2_000)

    with pytest.raises(ValueError, match="Hosted Chat API base URL"):
        normalize_hosted_chat_base_url(value, default="https://api.example/v1")


@pytest.mark.parametrize("default", [None, "", 3, "https://api.example/responses"])
def test_normalize_hosted_chat_base_url_validates_default(default: object) -> None:
    with pytest.raises(ValueError, match="Hosted Chat API base URL"):
        normalize_hosted_chat_base_url(None, default=default)


def test_normalize_hosted_chat_response_preserves_text_tools_reasoning_and_usage() -> (
    None
):
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The result is 4.",
                    "reasoning_content": "Use the calculator.",
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
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    original = deepcopy(response)

    turn = normalize_hosted_chat_response(response, finish_policy=_POLICY)

    assert turn == HostedChatTurn(
        text="The result is 4.",
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
            "content": "The result is 4.",
            "reasoning_content": "Use the calculator.",
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
        reasoning_content="Use the calculator.",
        usage={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    )
    assert response == original


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": "wrong"},
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "user", "content": "wrong"},
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": 7},
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": "not-json",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                        "reasoning_content": 9,
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "unknown",
                }
            ]
        },
    ],
)
def test_normalize_hosted_chat_response_rejects_malformed_results(
    response: object,
) -> None:
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(response, finish_policy=_POLICY)


def test_normalize_hosted_chat_response_enforces_json_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hosted_chat, "_MAX_JSON_DEPTH", 3)
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"nested": {"too": {"deep": 1}}},
    }

    with pytest.raises(HostedChatProtocolError, match="JSON"):
        normalize_hosted_chat_response(response, finish_policy=_POLICY)


def test_hosted_chat_stream_accumulates_interleaved_calls_and_terminal_usage() -> None:
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "hel",
                        "reasoning_content": "thin",
                        "tool_calls": [
                            _tool_delta(
                                index=0,
                                call_id="call_1",
                                name="calculator",
                                arguments='{"expression":"2+',
                            )
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "lo",
                        "reasoning_content": "k",
                        "tool_calls": [
                            _tool_delta(index=0, arguments='2"}'),
                            _tool_delta(
                                index=1,
                                call_id="call_2",
                                name="calculator",
                                arguments='{"expression":"3+3"}',
                            ),
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    records = [
        *(SSERecord(event="message", data=json.dumps(event)) for event in events),
        SSERecord(event=None, data="[DONE]"),
    ]
    stream = HostedChatStream(iter(records), finish_policy=_POLICY)

    with pytest.raises(HostedChatProtocolError, match="incomplete"):
        _ = stream.terminal_turn
    assert list(stream) == events
    assert stream.terminal_turn == HostedChatTurn(
        text="hello",
        tool_calls=(
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"2+2"}',
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"3+3"}',
                },
            },
        ),
        assistant_message={
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "think",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"3+3"}',
                    },
                },
            ],
        },
        finish_reason="tool_calls",
        reasoning_content="think",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def test_hosted_chat_stream_accepts_usage_only_after_terminal_choice() -> None:
    records = iter(
        [
            SSERecord(
                event=None,
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "done"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ),
            ),
            SSERecord(
                event=None,
                data=json.dumps({"choices": [], "usage": {"total_tokens": 3}}),
            ),
            SSERecord(event=None, data="[DONE]"),
        ]
    )
    stream = HostedChatStream(records, finish_policy=_POLICY)

    assert len(list(stream)) == 2
    assert stream.terminal_turn.usage == {"total_tokens": 3}


def test_hosted_chat_stream_accepts_terminal_choice_usage() -> None:
    event = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "done"},
                "finish_reason": "stop",
                "usage": {"total_tokens": 3},
            }
        ]
    }
    stream = HostedChatStream(
        iter(
            [
                SSERecord(event=None, data=json.dumps(event)),
                SSERecord(event=None, data="[DONE]"),
            ]
        ),
        finish_policy=_POLICY,
    )

    assert list(stream) == [event]
    assert stream.terminal_turn.usage == {"total_tokens": 3}


def test_hosted_chat_stream_accepts_identical_trailing_usage_duplicate() -> None:
    usage = {"total_tokens": 3}
    records = iter(
        [
            SSERecord(
                event=None,
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "done"},
                                "finish_reason": "stop",
                                "usage": usage,
                            }
                        ]
                    }
                ),
            ),
            SSERecord(
                event=None,
                data=json.dumps({"choices": [], "usage": usage}),
            ),
            SSERecord(event=None, data="[DONE]"),
        ]
    )
    stream = HostedChatStream(records, finish_policy=_POLICY)

    assert len(list(stream)) == 2
    assert stream.terminal_turn.usage == usage


def test_hosted_chat_stream_rejects_differing_trailing_usage_duplicate() -> None:
    records = iter(
        [
            SSERecord(
                event=None,
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "done"},
                                "finish_reason": "stop",
                                "usage": {"total_tokens": 3},
                            }
                        ]
                    }
                ),
            ),
            SSERecord(
                event=None,
                data=json.dumps({"choices": [], "usage": {"total_tokens": 4}}),
            ),
            SSERecord(event=None, data="[DONE]"),
        ]
    )
    stream = HostedChatStream(records, finish_policy=_POLICY)

    with pytest.raises(
        HostedChatProtocolError,
        match=r"^Hosted Chat stream usage is malformed\.$",
    ):
        list(stream)


@pytest.mark.parametrize(
    "choice_usage,trailing_usages",
    [
        ({"total_tokens": 3}, [{"total_tokens": 3}, {"total_tokens": 3}]),
        (None, [{"total_tokens": 3}, {"total_tokens": 3}]),
        ({"total_tokens": 1}, [{"total_tokens": True}]),
    ],
)
def test_hosted_chat_stream_rejects_unobserved_usage_duplicates(
    choice_usage: dict[str, object] | None,
    trailing_usages: list[dict[str, object]],
) -> None:
    choice: dict[str, object] = {
        "index": 0,
        "delta": {"content": "done"},
        "finish_reason": "stop",
    }
    if choice_usage is not None:
        choice["usage"] = choice_usage
    records = iter(
        [
            SSERecord(
                event=None,
                data=json.dumps({"choices": [choice]}),
            ),
            *(
                SSERecord(
                    event=None,
                    data=json.dumps({"choices": [], "usage": usage}),
                )
                for usage in trailing_usages
            ),
            SSERecord(event=None, data="[DONE]"),
        ]
    )
    stream = HostedChatStream(records, finish_policy=_POLICY)

    with pytest.raises(
        HostedChatProtocolError,
        match=r"^Hosted Chat stream usage is malformed\.$",
    ):
        list(stream)


@pytest.mark.parametrize(
    "choice_usage,top_level_usage,finish_reason,error_match",
    [
        (True, None, "stop", r"^Hosted Chat stream usage is malformed\.$"),
        (
            {"total_tokens": 3},
            {"total_tokens": 3},
            "stop",
            r"^Hosted Chat stream usage is malformed\.$",
        ),
        (
            {"total_tokens": 3},
            None,
            None,
            r"^Hosted Chat stream usage preceded terminal state\.$",
        ),
    ],
)
def test_hosted_chat_stream_rejects_malformed_or_misplaced_choice_usage(
    choice_usage: object,
    top_level_usage: object,
    finish_reason: str | None,
    error_match: str,
) -> None:
    event = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "done"},
                "finish_reason": finish_reason,
                "usage": choice_usage,
            }
        ]
    }
    if top_level_usage is not None:
        event["usage"] = top_level_usage
    stream = HostedChatStream(
        iter(
            [
                SSERecord(event=None, data=json.dumps(event)),
                SSERecord(event=None, data="[DONE]"),
            ]
        ),
        finish_policy=_POLICY,
    )

    with pytest.raises(HostedChatProtocolError, match=error_match):
        list(stream)


@pytest.mark.parametrize("fingerprint", ["fp_kimi_live", None])
def test_hosted_chat_stream_accepts_system_fingerprint(
    fingerprint: str | None,
) -> None:
    event = {
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 3},
    }
    stream = HostedChatStream(
        iter(
            [
                SSERecord(event=None, data=json.dumps(event)),
                SSERecord(event=None, data="[DONE]"),
            ]
        ),
        finish_policy=_POLICY,
    )

    assert list(stream) == [event]
    assert stream.terminal_turn.text == "done"


@pytest.mark.parametrize(
    "fingerprint",
    [True, 1, "", "x" * (hosted_chat._MAX_METADATA_CHARS + 1)],
)
def test_hosted_chat_stream_rejects_malformed_system_fingerprint(
    fingerprint: object,
) -> None:
    event = {
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 3},
    }
    stream = HostedChatStream(
        iter(
            [
                SSERecord(event=None, data=json.dumps(event)),
                SSERecord(event=None, data="[DONE]"),
            ]
        ),
        finish_policy=_POLICY,
    )

    with pytest.raises(
        HostedChatProtocolError,
        match=r"^Hosted Chat system fingerprint is malformed\.$",
    ):
        list(stream)


def test_hosted_chat_stream_rejects_unknown_top_level_metadata() -> None:
    event = {
        "unexpected_live_metadata": "value",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 3},
    }
    stream = HostedChatStream(
        iter(
            [
                SSERecord(event=None, data=json.dumps(event)),
                SSERecord(event=None, data="[DONE]"),
            ]
        ),
        finish_policy=_POLICY,
    )

    with pytest.raises(
        HostedChatProtocolError,
        match=r"^Hosted Chat stream event is malformed\.$",
    ):
        list(stream)


@pytest.mark.parametrize(
    "records",
    [
        [SSERecord(event=None, data="[DONE]")],
        [
            SSERecord(
                event=None,
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "done"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"total_tokens": 3},
                    }
                ),
            )
        ],
        [
            SSERecord(
                event=None,
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "done"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"total_tokens": 3},
                    }
                ),
            ),
            SSERecord(
                event=None,
                data=json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "late"},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
        ],
    ],
)
def test_hosted_chat_stream_rejects_missing_or_post_terminal_data(
    records: list[SSERecord],
) -> None:
    stream = HostedChatStream(iter(records), finish_policy=_POLICY)

    with pytest.raises(HostedChatProtocolError):
        list(stream)
    with pytest.raises(HostedChatProtocolError, match="incomplete"):
        _ = stream.terminal_turn


def test_hosted_chat_stream_close_keeps_terminal_metadata_unavailable() -> None:
    stream = HostedChatStream(iter(()), finish_policy=_POLICY)

    stream.close()
    stream.close()
    assert list(stream) == []
    with pytest.raises(HostedChatProtocolError, match="incomplete"):
        _ = stream.terminal_turn


def test_transport_config_repr_hides_api_key() -> None:
    config = _transport_config("https://api.example/v1")

    assert "SECRET-TRANSPORT-CANARY" not in repr(config)
    assert config == _transport_config(
        "https://api.example/v1", api_key="DIFFERENT-KEY"
    )


def test_owned_json_post_maps_invalid_base_url_to_redacted_transport_error() -> None:
    """Map malformed bases to a redacted public transport error."""
    base_url = "https://user:RAW-URL-CANARY@example.com/v1"

    with pytest.raises(ChatProviderError) as exc_info:
        owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload={},
            streaming=False,
        )

    assert type(exc_info.value) is ChatProviderError
    assert exc_info.value.__suppress_context__ is True
    rendered = "".join(
        traceback.format_exception(
            exc_info.type,
            exc_info.value,
            exc_info.tb,
        )
    )
    assert base_url not in rendered
    assert "RAW-URL-CANARY" not in rendered


@pytest.mark.allow_network
@pytest.mark.parametrize("route", ["chat/completions", "responses"])
def test_owned_json_post_sends_exact_route_headers_payload_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    sessions = _track_transport_sessions(monkeypatch)
    payload = {"model": "test-model", "stream": False}
    with _scripted_hosted_server([{"body": json.dumps({"ok": True}).encode()}]) as (
        server,
        base_url,
    ):
        result = owned_json_post(
            config=_transport_config(base_url, timeout=1.25),
            route=route,  # type: ignore[arg-type]
            payload=payload,
            streaming=False,
        )

    assert result == {"ok": True}
    assert server.requests == [
        {
            "path": f"/v1/{route}",
            "headers": server.requests[0]["headers"],
            "body": json.dumps(payload).encode(),
        }
    ]
    assert server.requests[0]["headers"]["Authorization"] == (
        "Bearer SECRET-TRANSPORT-CANARY"
    )
    assert server.requests[0]["headers"]["Content-Type"] == "application/json"
    assert sessions[0].close_calls == 1
    assert sessions[0].response_close_calls == [1]


@pytest.mark.allow_network
def test_owned_json_post_retries_statuses_with_one_global_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _track_transport_sessions(monkeypatch)
    with _scripted_hosted_server(
        [
            {"status": 503, "body": b"service unavailable"},
            {"status": 429, "headers": {"Retry-After": "0"}, "body": b"rate"},
            {"body": b'{"ok":true}'},
        ]
    ) as (server, base_url):
        result = owned_json_post(
            config=_transport_config(base_url, retries=2),
            route="chat/completions",
            payload={"stream": False},
            streaming=False,
        )

    assert result == {"ok": True}
    assert len(server.requests) == 3
    assert sessions[0].response_close_calls == [1, 1, 1]
    assert sessions[0].close_calls == 1


@pytest.mark.allow_network
def test_owned_json_post_honors_http_date_and_malformed_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(hosted_chat.time, "sleep", sleeps.append)
    future = formatdate(time.time() + 2, usegmt=True)
    with _scripted_hosted_server(
        [
            {"status": 503, "headers": {"Retry-After": future}},
            {"status": 503, "headers": {"Retry-After": "RAW-RETRY-CANARY"}},
            {"body": b'{"ok":true}'},
        ]
    ) as (_server, base_url):
        result = owned_json_post(
            config=_transport_config(base_url, retries=2, retry_delay=0.25),
            route="chat/completions",
            payload={},
            streaming=False,
        )

    assert result == {"ok": True}
    assert 0 < sleeps[0] <= 2
    assert sleeps[1] == pytest.approx(0.5)


@pytest.mark.allow_network
def test_sensitive_request_forces_transport_retries_to_zero() -> None:
    with _scripted_hosted_server([{"status": 503}, {"body": b'{"ok":true}'}]) as (
        server,
        base_url,
    ):
        with sensitive_llm_request(), pytest.raises(ChatProviderError):
            owned_json_post(
                config=_transport_config(base_url, retries=5),
                route="chat/completions",
                payload={},
                streaming=False,
            )

    assert len(server.requests) == 1


@pytest.mark.allow_network
@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, ChatAuthenticationError),
        (403, ChatAuthenticationError),
        (429, ChatRateLimitError),
        (400, ChatBadRequestError),
        (500, ChatProviderError),
    ],
)
def test_owned_json_post_maps_http_failures_without_body_disclosure(
    status: int,
    error_type: type[Exception],
) -> None:
    canary = b"RAW-RESPONSE-BODY-CANARY"
    with _scripted_hosted_server([{"status": status, "body": canary}]) as (
        _server,
        base_url,
    ):
        with pytest.raises(error_type) as exc_info:
            owned_json_post(
                config=_transport_config(base_url),
                route="chat/completions",
                payload={},
                streaming=False,
            )

    assert "RAW-RESPONSE-BODY-CANARY" not in str(exc_info.value)
    assert "SECRET-TRANSPORT-CANARY" not in str(exc_info.value)
    assert base_url not in str(exc_info.value)


@pytest.mark.allow_network
@pytest.mark.parametrize(
    "action",
    [
        {"body": b"not-json"},
        {"body": b"[]"},
        {"body": b'{"ok":true}', "extra_content_length": 7},
    ],
)
def test_nonstreaming_2xx_malformed_body_is_not_retried(
    action: dict[str, Any],
) -> None:
    with _scripted_hosted_server([action, {"body": b'{"ok":true}'}]) as (
        server,
        base_url,
    ):
        with pytest.raises(ChatProviderError):
            owned_json_post(
                config=_transport_config(base_url, retries=2),
                route="chat/completions",
                payload={},
                streaming=False,
            )

    assert len(server.requests) == 1


@pytest.mark.allow_network
def test_owned_sse_stream_transfers_ownership_and_closes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _track_transport_sessions(monkeypatch)
    body = b'data: {"ok":true}\n\ndata: [DONE]\n\n'
    with _scripted_hosted_server(
        [{"body": body, "content_type": "text/event-stream"}]
    ) as (_server, base_url):
        stream = owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload={"stream": True},
            streaming=True,
        )
        assert isinstance(stream, OwnedSSEStream)
        assert sessions[0].close_calls == 0
        assert list(stream) == [
            SSERecord(event=None, data='{"ok":true}'),
            SSERecord(event=None, data="[DONE]"),
        ]
        stream.close()
        stream.close()

    assert sessions[0].response_close_calls == [1]
    assert sessions[0].close_calls == 1


@pytest.mark.allow_network
def test_owned_sse_stream_does_not_retry_after_any_body_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _track_transport_sessions(monkeypatch)
    action = {
        "body": b"data: partial",
        "content_type": "text/event-stream",
        "extra_content_length": 20,
    }
    with _scripted_hosted_server(
        [action, {"body": b"data: [DONE]\n\n", "content_type": "text/event-stream"}]
    ) as (server, base_url):
        stream = owned_json_post(
            config=_transport_config(base_url, retries=2),
            route="chat/completions",
            payload={"stream": True},
            streaming=True,
        )
        assert isinstance(stream, OwnedSSEStream)
        with pytest.raises(HostedSSEReadError):
            list(stream)

    assert len(server.requests) == 1
    assert sessions[0].response_close_calls == [1]
    assert sessions[0].close_calls == 1
