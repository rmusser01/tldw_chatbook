"""ADR-222: per-thread provider HTTP session reuse; payloads by reference.

Covers the two contracts of TASK-34418:

- ``provider_sessions`` registry: same-key reuse within a thread, never
  across threads, mounted adapters retained, close helpers scoped to the
  calling thread.
- ``hosted_chat.owned_json_post``: the payload mapping is passed to
  ``requests`` by reference (no per-attempt ``deepcopy``) and the wire
  body is byte-identical to what the deepcopy path produced; the session
  is registry-owned, so calls stop closing it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tldw_chatbook.LLM_Calls import hosted_chat
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedHTTPTransportConfig,
    owned_json_post,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.provider_sessions import (
    close_all_for_current_thread,
    close_session,
    get_session,
)

# Registry hermeticity between tests is provided by the root conftest's
# autouse ``reset_provider_session_registry`` fixture (ADR-222 §3).


class _CloseTrackingSession(requests.Session):
    """Session whose ``close()`` calls are counted for lifecycle pins."""

    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


# ---------------------------------------------------------------------------
# Registry contracts (ADR-222 §1)
# ---------------------------------------------------------------------------


def test_same_key_returns_same_session_within_one_thread() -> None:
    builds: list[requests.Session] = []

    def factory() -> requests.Session:
        session = requests.Session()
        builds.append(session)
        return session

    first = get_session("provider:https://api.example.test", factory)
    second = get_session("provider:https://api.example.test", factory)
    assert first is second
    assert len(builds) == 1


def test_different_keys_get_different_sessions() -> None:
    def factory() -> requests.Session:
        return requests.Session()

    assert get_session("a:https://one.test", factory) is not get_session(
        "b:https://two.test", factory
    )


def test_different_threads_get_different_sessions() -> None:
    """requests.Session is not thread-safe; the registry must never share one."""

    builds: list[tuple[int, requests.Session]] = []
    lock = threading.Lock()
    held: dict[int, requests.Session] = {}

    def factory() -> requests.Session:
        session = requests.Session()
        with lock:
            builds.append((threading.get_ident(), session))
        return session

    main_session = get_session("shared:https://api.example.test", factory)
    held[threading.get_ident()] = main_session

    thread = threading.Thread(
        target=lambda: held.__setitem__(
            threading.get_ident(),
            get_session("shared:https://api.example.test", factory),
        )
    )
    thread.start()
    thread.join()

    thread_ids = {ident for ident, _session in builds}
    assert len(builds) == 2, "each thread must build its own session"
    assert len(thread_ids) == 2, "the two builds must come from two threads"
    other_session = held[thread_ids.difference({threading.get_ident()}).pop()]
    assert main_session is not other_session


def test_session_retains_mounted_retry_adapter() -> None:
    """The factory's adapter mount survives for the session's lifetime."""

    def factory() -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=7, backoff_factor=0.5))
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    session = get_session("provider:https://api.example.test", factory)
    again = get_session("provider:https://api.example.test", factory)
    assert session is again
    adapter = session.get_adapter("https://api.example.test/v1/chat/completions")
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 7
    assert again.get_adapter("http://api.example.test") is adapter


def test_close_all_for_current_thread_closes_only_the_calling_thread() -> None:
    other_sessions: list[_CloseTrackingSession] = []

    def factory() -> _CloseTrackingSession:
        session = _CloseTrackingSession()
        other_sessions.append(session)
        return session

    thread = threading.Thread(
        target=lambda: get_session("shared:https://api.example.test", factory)
    )
    thread.start()
    thread.join()
    other_thread_session = other_sessions[0]

    mine = get_session("shared:https://api.example.test", factory)
    close_all_for_current_thread()

    assert mine.close_calls == 1
    assert other_thread_session.close_calls == 0
    # A fresh build replaces the closed entry for this thread only.
    rebuilt = get_session("shared:https://api.example.test", factory)
    assert rebuilt is not mine
    assert other_sessions == [other_thread_session, mine, rebuilt]


def test_close_session_is_surgical_per_key() -> None:
    def factory() -> _CloseTrackingSession:
        return _CloseTrackingSession()

    one = get_session("a:https://one.test", factory)
    two = get_session("b:https://two.test", factory)
    close_session("a:https://one.test")

    assert one.close_calls == 1
    assert two.close_calls == 0
    assert get_session("a:https://one.test", factory) is not one
    assert get_session("b:https://two.test", factory) is two


def test_get_session_clears_cookie_jar_on_every_hit() -> None:
    """Parity with the fresh-session-per-call semantics ADR-222 replaces."""

    def factory() -> requests.Session:
        return requests.Session()

    session = get_session("provider:https://api.example.test", factory)
    session.cookies.set("session", "stale-value")
    again = get_session("provider:https://api.example.test", factory)

    assert again is session
    assert dict(again.cookies) == {}


# ---------------------------------------------------------------------------
# hosted_chat integration: registry ownership + payload by reference
# ---------------------------------------------------------------------------


class _ScriptedHostedServer(ThreadingHTTPServer):
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedHostedHandler)
        self.actions = actions
        self.requests: list[dict[str, Any]] = []


class _ScriptedHostedHandler(BaseHTTPRequestHandler):
    server: _ScriptedHostedServer

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.requests.append(
            {"path": self.path, "headers": dict(self.headers), "body": body}
        )
        action = self.server.actions.pop(0)
        status = action.get("status", 200)
        response_body = action.get("body", b"{}")
        self.send_response(status)
        self.send_header("Content-Type", action.get("content_type", "application/json"))
        self.send_header("Content-Length", str(len(response_body)))
        for name, value in action.get("headers", {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(response_body)
        self.wfile.flush()

    def log_message(self, _format: str, *args: object) -> None:
        del args


@contextmanager
def _scripted_server(actions: list[dict[str, Any]]) -> Iterator[Any]:
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


def _transport_config(base_url: str, **overrides: object) -> HostedHTTPTransportConfig:
    values: dict[str, object] = {
        "provider": "moonshot",
        "base_url": base_url,
        "api_key": "SECRET-TRANSPORT-CANARY",
        "timeout": 2,
        "retries": 0,
        "retry_delay": 0.0,
    }
    values.update(overrides)
    return HostedHTTPTransportConfig(**values)  # type: ignore[arg-type]


def _sample_payload() -> dict[str, Any]:
    return {
        "model": "test-model",
        "stream": False,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ],
        "tools": [{"type": "function", "function": {"name": "ping", "arguments": {}}}],
    }


def _track_hosted_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_CloseTrackingSession]:
    sessions: list[_CloseTrackingSession] = []

    def create_session() -> _CloseTrackingSession:
        session = _CloseTrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(hosted_chat, "create_default_session", create_session)
    return sessions


@pytest.mark.allow_network
def test_owned_json_post_builds_one_session_per_key_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _track_hosted_sessions(monkeypatch)
    payload = {"model": "test-model", "stream": False}
    with _scripted_server([{"body": b'{"ok":1}'}, {"body": b'{"ok":2}'}]) as (
        server,
        base_url,
    ):
        first = owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload=payload,
            streaming=False,
        )
        second = owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload=payload,
            streaming=False,
        )

    assert first == {"ok": 1}
    assert second == {"ok": 2}
    assert len(server.requests) == 2
    assert len(sessions) == 1, "same (provider, base_url) must reuse one session"
    assert sessions[0].close_calls == 0, "registry owns the session, not the call"


@pytest.mark.allow_network
def test_owned_json_post_distinct_base_urls_build_distinct_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _track_hosted_sessions(monkeypatch)
    payload = {"model": "test-model", "stream": False}
    with (
        _scripted_server([{"body": b'{"ok":1}'}]) as (_one, base_one),
        _scripted_server([{"body": b'{"ok":2}'}]) as (_two, base_two),
    ):
        owned_json_post(
            config=_transport_config(base_one),
            route="chat/completions",
            payload=payload,
            streaming=False,
        )
        owned_json_post(
            config=_transport_config(base_two),
            route="chat/completions",
            payload=payload,
            streaming=False,
        )

    assert len(sessions) == 2


@pytest.mark.allow_network
def test_owned_json_post_payload_unchanged_after_successful_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _track_hosted_sessions(monkeypatch)
    payload = _sample_payload()
    snapshot = deepcopy(payload)
    with _scripted_server([{"body": b'{"ok":true}'}]) as (_server, base_url):
        result = owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload=payload,
            streaming=False,
        )

    assert result == {"ok": True}
    assert payload == snapshot, "requests must not mutate the json= argument"


@pytest.mark.allow_network
def test_owned_json_post_payload_unchanged_after_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Chat.Chat_Deps import ChatProviderError

    _track_hosted_sessions(monkeypatch)
    payload = _sample_payload()
    snapshot = deepcopy(payload)
    with (
        _scripted_server([{"status": 503, "body": b"down"}] * 3) as (_server, base_url),
        pytest.raises(ChatProviderError),
    ):
        owned_json_post(
            config=_transport_config(base_url, retries=2),
            route="chat/completions",
            payload=payload,
            streaming=False,
        )

    assert len(_server.requests) == 3
    assert payload == snapshot, "retry attempts must not mutate the payload either"


@pytest.mark.allow_network
def test_owned_json_post_wire_body_is_byte_identical_to_deepcopy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Golden capture: the by-reference body equals the old deepcopy body."""
    _track_hosted_sessions(monkeypatch)
    payload = _sample_payload()
    golden = json.dumps(deepcopy(dict(payload))).encode()
    with _scripted_server([{"body": b'{"ok":true}'}]) as (server, base_url):
        owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload=payload,
            streaming=False,
        )

    assert server.requests[0]["body"] == golden


@pytest.mark.allow_network
def test_owned_json_post_streaming_keeps_session_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _track_hosted_sessions(monkeypatch)
    body = b'data: {"ok":true}\n\ndata: [DONE]\n\n'
    with _scripted_server(
        [
            {"body": body, "content_type": "text/event-stream"},
            {"body": b'{"ok":2}'},
        ]
    ) as (_server, base_url):
        stream = owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload={"stream": True},
            streaming=True,
        )
        records = list(stream)
        # The next call on the same key reuses the warmed session.
        owned_json_post(
            config=_transport_config(base_url),
            route="chat/completions",
            payload={"stream": False},
            streaming=False,
        )

    assert records == [
        SSERecord(event=None, data='{"ok":true}'),
        SSERecord(event=None, data="[DONE]"),
    ]
    assert len(sessions) == 1
    assert sessions[0].close_calls == 0, (
        "the stream closes its response, not the session"
    )


# ---------------------------------------------------------------------------
# TLS/TCP handshake evidence (ADR-222 Consequences)
# ---------------------------------------------------------------------------


class _ConnectionCountingServer(ThreadingHTTPServer):
    """HTTP/1.1 server that counts accepted TCP connections."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ConnectionCountingHandler)
        self.connections = 0
        self.lock = threading.Lock()

    def get_request(self) -> tuple[Any, Any]:
        with self.lock:
            self.connections += 1
        return super().get_request()


class _ConnectionCountingHandler(BaseHTTPRequestHandler):
    server: _ConnectionCountingServer
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        response_body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)
        self.wfile.flush()

    def log_message(self, _format: str, *args: object) -> None:
        del args


@contextmanager
def _counting_server() -> Iterator[Any]:
    server = _ConnectionCountingServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.allow_network
def test_three_back_to_back_calls_share_one_tcp_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence for ADR-222: 3 calls on one key open 1 connection, not 3.

    The pre-change number (3 calls -> 3 connections, one fresh session per
    call) is recorded from the same probe run against pristine HEAD in the
    task report; it cannot be expressed as a test on this branch because
    the registry path no longer builds per-call sessions.
    """
    sessions = _track_hosted_sessions(monkeypatch)
    payload = {"model": "test-model", "stream": False}
    with _counting_server() as (server, base_url):
        for _ in range(3):
            owned_json_post(
                config=_transport_config(base_url),
                route="chat/completions",
                payload=payload,
                streaming=False,
            )

    assert server.connections == 1, "keep-alive: one TCP connection for three calls"
    assert len(sessions) == 1
