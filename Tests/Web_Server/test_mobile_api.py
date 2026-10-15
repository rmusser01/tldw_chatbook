"""Mobile REST API under /api/v1/mobile — tasks 4-5, ADR-234 Phase 1b.

HTTP-contract tests over the authed route table: auth, CSRF, response
shapes/statuses, the unknown-session 404, the text bounds, and the
``mobile_surface`` registration gate — plus the task-5 event stream
WebSocket (``GET /api/v1/mobile/stream``): subprotocol upgrade auth, the
server->client event frames, the JSON ping/pong contract, and the 4401 /
1011 close codes. The engine is NOT under test here — the service is
injected as a fake (Task 3's ``test_mobile_service.py`` owns the engine
behavior).
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import WSMsgType, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

from tldw_chatbook.Canvas.web_auth import CSRF_HEADER_NAME, build_web_auth_policy
from tldw_chatbook.config import build_canvas_config_policy
from tldw_chatbook.Web_Server import mobile_api, serve
from tldw_chatbook.Web_Server.session_store import WebSessionStore

pytestmark = [
    pytest.mark.skipif(
        not serve.check_web_server_available(),
        reason="web server optional dependencies are unavailable",
    ),
    pytest.mark.loopback_network,
]


# The server/client scaffolding below (_FakeBaseServer, _mobile_server,
# _app_without_control_broker, _logged_in_client, _csrf) is duplicated from
# Tests/Web_Server/test_web_auth_integration.py: no shared
# Tests/Web_Server/conftest.py exists (task-4 pre-flight ruling), so each
# file carries its own copy. Keep the two in sync.


class _FakeBaseServer:
    def __init__(
        self,
        command: str,
        host: str,
        port: int,
        title: str,
        *,
        public_url: str | None = None,
        statics_path: str,
        templates_path: str,
    ) -> None:
        self.command = command
        self.host = host
        self.port = port
        self.title = title
        self.public_url = public_url or f"http://{host}:{port}"
        self.statics_path = statics_path
        self.templates_path = templates_path
        self.download_manager = object()
        self.debug = False

    async def on_startup(self, _app) -> None:
        return None

    async def on_shutdown(self, _app) -> None:
        return None

    async def handle_download(self, _request):
        from aiohttp import web

        return web.Response(text="download")


class FakeMobileService:
    """Stub MobileChatService: records calls, returns canned dicts.

    Models the Task-3 non-blocking send contract (live-check round 1):
    ``send`` pre-validates synchronously (unknown session ``KeyError``,
    configured ``send_error``, one turn in flight → ``MobileSendRefused``),
    appends the user row, then schedules the scripted event sequence on a
    background task — so the REST 202 returns while the turn is pending,
    exactly like the real service scheduling ``_run_turn``. Park the
    scripted turn by setting ``turn_gate`` (an ``asyncio.Event`` the task
    waits on) to observe the pending window deterministically.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.shutdown_count = 0
        self.cancel_result = True
        self.send_error: Exception | None = None
        self.turn_gate: asyncio.Event | None = None
        self.turn_running = False
        self._turn_task: asyncio.Task | None = None
        self._next_message_number = 3
        self._event_queues: set[asyncio.Queue] = set()
        self._event_seq = 0
        self._sessions: dict[str, dict[str, Any]] = {
            "s-known": {
                "id": "s-known",
                "title": "phone chat",
                "workspace_id": "global",
                "has_folder_tools": False,
            }
        }
        self._messages: dict[str, list[dict[str, Any]]] = {
            "s-known": [
                {
                    "id": "m1",
                    "role": "user",
                    "content": "hi",
                    "status": "complete",
                    "created_at": None,
                },
                {
                    "id": "m2",
                    "role": "assistant",
                    "content": "hello",
                    "status": "complete",
                    "created_at": None,
                },
            ]
        }

    @property
    def sessions(self) -> list[dict[str, Any]]:
        self.calls.append(("sessions", {}))
        # Task-3 contract: the sessions list is newest-first.
        return [dict(session) for session in reversed(list(self._sessions.values()))]

    async def create_session(
        self, *, title: str, workspace_id: str | None = None
    ) -> dict[str, Any]:
        self.calls.append(
            ("create_session", {"title": title, "workspace_id": workspace_id})
        )
        session_id = f"s-new-{len(self._sessions)}"
        session = {
            "id": session_id,
            "title": title,
            "workspace_id": workspace_id or "global",
            "has_folder_tools": False,
        }
        self._sessions[session_id] = session
        self._messages[session_id] = []
        return dict(session)

    def session_messages(
        self, session_id: str, *, after_message_id: str | None = None
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "session_messages",
                {"session_id": session_id, "after_message_id": after_message_id},
            )
        )
        messages = self._messages.get(session_id)
        if messages is None:
            raise KeyError(session_id)
        if after_message_id is not None:
            ids = [message["id"] for message in messages]
            if after_message_id not in ids:
                raise ValueError("cursor message not found")
            messages = messages[ids.index(after_message_id) + 1 :]
        wire = [dict(message) for message in messages]
        cursor = wire[-1]["id"] if wire else (after_message_id or None)
        return {"messages": wire, "cursor": cursor}

    async def send(self, session_id: str, draft: str) -> None:
        from tldw_chatbook.Web_Server.mobile_service import MobileSendRefused

        self.calls.append(("send", {"session_id": session_id, "draft": draft}))
        messages = self._messages.get(session_id)
        if messages is None:
            raise KeyError(session_id)
        if self.send_error is not None:
            raise self.send_error
        if self.turn_running:
            raise MobileSendRefused("a turn is already in flight")
        messages.append(
            {
                "id": f"m{self._next_message_number}",
                "role": "user",
                "content": draft,
                "status": "complete",
                "created_at": None,
            }
        )
        self._next_message_number += 1
        self.turn_running = True
        self._turn_task = asyncio.create_task(self._scripted_turn(session_id))

    async def _scripted_turn(self, session_id: str) -> None:
        """Background scripted run (Task 3's wire contract): the same event
        sequence the real service fans out for one turn — run_started,
        deltas, message_done, run_completed, and the trailing
        session_updated — emitted from a task so ``send`` can return 202
        while the turn is pending."""
        try:
            if self.turn_gate is not None:
                await self.turn_gate.wait()
            self.emit("run_started", session_id, {"status": "generating"})
            self.emit(
                "delta", session_id, {"message_id": "m9", "offset": 0, "text": "Hel"}
            )
            self.emit(
                "delta", session_id, {"message_id": "m9", "offset": 3, "text": "lo"}
            )
            self.emit(
                "message_done", session_id, {"message_id": "m9", "status": "complete"}
            )
            self.emit("run_completed", session_id, {"status": "completed"})
            self.emit(
                "session_updated",
                session_id,
                {"session_id": session_id, "reason": "run_completed"},
            )
        finally:
            self.turn_running = False

    def emit(self, type_: str, session_id: str = "", payload: dict | None = None):
        """Fan one event out to every current subscriber (Task 3 shape)."""
        from tldw_chatbook.Web_Server.mobile_service import MobileEvent

        self._event_seq += 1
        event = MobileEvent(
            type=type_,
            session_id=session_id,
            payload=dict(payload or {}),
            seq=self._event_seq,
        )
        for queue in list(self._event_queues):
            queue.put_nowait(event)
        return event

    async def events(self):
        """One queue per subscriber; discarded on close (Task 3 shape)."""
        queue: asyncio.Queue = asyncio.Queue()
        self._event_queues.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._event_queues.discard(queue)

    def cancel(self, session_id: str) -> bool:
        self.calls.append(("cancel", {"session_id": session_id}))
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self.cancel_result

    def list_approvals(self) -> list[dict[str, Any]]:
        self.calls.append(("list_approvals", {}))
        return [
            {
                "round_id": "r1",
                "session_id": "s-known",
                "calls": [
                    {
                        "call_id": "c1",
                        "llm_name": "fs_list",
                        "options": ["approve_once", "deny"],
                        "arguments": {"path": "."},
                    }
                ],
                "timeout_seconds": 300,
            }
        ]

    def decide_approval(self, round_id: str, decisions: dict[str, str]) -> bool:
        self.calls.append(
            ("decide_approval", {"round_id": round_id, "decisions": dict(decisions)})
        )
        return round_id == "r1"

    async def shutdown(self) -> None:
        self.shutdown_count += 1


_MOBILE_ACCESS_TOKEN = "mobile-api-secret"
_MOBILE_PUBLIC_ORIGIN = "http://chatbook.example"
_MOBILE_REQUEST_HEADERS = {
    "Origin": _MOBILE_PUBLIC_ORIGIN,
    "X-Forwarded-For": "203.0.113.9",
    "X-Forwarded-Proto": "http",
    "X-Forwarded-Host": "chatbook.example",
}


def _mobile_server(tmp_path, policy, *, service: FakeMobileService):
    """Build a config-free served server with the fake service injected."""

    templates = tmp_path / "templates"
    statics = tmp_path / "static"
    (statics / "js").mkdir(parents=True, exist_ok=True)
    templates.mkdir(exist_ok=True)
    (templates / "app_index.html").write_text(
        (
            '<html><head><script src="{{ config.static.url }}js/textual.js">'
            "</script></head><body>{{ application.name }}</body></html>"
        ),
        encoding="utf-8",
    )
    (statics / "js" / "textual.js").write_text("fixed-runtime", encoding="utf-8")
    cls = serve.build_chatbook_web_server_class(_FakeBaseServer)
    return cls(
        command="python -m tldw_chatbook.app",
        host=policy.bind_host,
        port=policy.port,
        title="Test",
        public_url=f"{policy.external_scheme}://{next(iter(policy.allowed_hosts))}",
        statics_path=str(statics),
        templates_path=str(templates),
        web_auth_policy=policy,
        canvas_policy=build_canvas_config_policy(
            {}, environ={"TLDW_CANVAS_ENABLED": "false"}
        ),
        session_store=WebSessionStore(tmp_path / "web_sessions.json"),
        mobile_service=service,
    )


async def _app_without_control_broker(server):
    app = await server._make_app()
    app.on_startup.clear()
    app.on_shutdown.clear()
    return app


@asynccontextmanager
async def _logged_in_client(tmp_path, *, service: FakeMobileService):
    """Yield one token-authenticated TestClient against a fresh server."""

    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=_MOBILE_ACCESS_TOKEN,
        public_url=_MOBILE_PUBLIC_ORIGIN,
        allow_insecure_remote_http=True,
        trusted_proxy_addresses=["127.0.0.1"],
    )
    server = _mobile_server(tmp_path, policy, service=service)
    client = TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_MOBILE_REQUEST_HEADERS),
    )
    client.chatbook_test_server = server
    async with client:
        login = await client.post(
            "/auth/login",
            data={"access_token": _MOBILE_ACCESS_TOKEN, "remember": "1"},
            allow_redirects=False,
        )
        assert login.status == 303
        assert login.headers["Location"] == "/"
        yield client


async def _csrf(client) -> str:
    """Read the per-session CSRF token from the devices page's meta tag."""

    page = await client.get("/auth/devices")
    assert page.status == 200
    match = re.search(
        r'<meta name="chatbook-csrf" content="([^"]*)">', await page.text()
    )
    assert match is not None
    return match.group(1)


async def _current_session_id(client) -> str:
    """Read the requesting device's session id from the devices page."""

    page = await client.get("/auth/devices")
    assert page.status == 200
    match = re.search(
        r'data-session-id="([^"]*)"\s+data-current-device="true"', await page.text()
    )
    assert match is not None
    return match.group(1)


async def _assert_subscribers_drained(service: FakeMobileService) -> None:
    """Wait briefly for the server to drain the WS subscriber queue set."""

    for _ in range(100):
        if not service._event_queues:
            return
        await asyncio.sleep(0.02)
    assert service._event_queues == set()


@pytest.fixture(autouse=True)
def mobile_surface_gate(monkeypatch):
    """Pin the ``[web_server] mobile_surface`` gate without loading config.

    These tests must not touch the guarded config loader (the per-test
    sandbox fails those reads closed), so the gate's ``get_cli_setting``
    reference is patched; the returned installer flips the gate per test
    BEFORE the server app (and its route table) is built.
    """

    def install(enabled: bool) -> None:
        monkeypatch.setattr(
            mobile_api,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                enabled
                if (section, key) == ("web_server", "mobile_surface")
                else default
            ),
        )

    install(True)
    return install


@pytest.mark.asyncio
async def test_mobile_routes_require_an_authenticated_session(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=_MOBILE_ACCESS_TOKEN,
        public_url=_MOBILE_PUBLIC_ORIGIN,
        allow_insecure_remote_http=True,
        trusted_proxy_addresses=["127.0.0.1"],
    )
    server = _mobile_server(tmp_path, policy, service=service)
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_MOBILE_REQUEST_HEADERS),
    ) as client:
        response = await client.get("/api/v1/mobile/conversations")

    assert response.status == 401
    assert service.calls == []


@pytest.mark.asyncio
async def test_send_requires_the_csrf_header(tmp_path, mobile_surface_gate) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        denied = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "phone question"},
        )
        assert denied.status == 401
        assert service.calls == []

        accepted = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "phone question"},
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )
        assert accepted.status == 202


@pytest.mark.asyncio
async def test_conversations_list_shape(tmp_path, mobile_surface_gate) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        response = await client.get("/api/v1/mobile/conversations")

        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = await response.json()
        assert body == {
            "sessions": [
                {
                    "id": "s-known",
                    "title": "phone chat",
                    "workspace_id": "global",
                    "has_folder_tools": False,
                }
            ]
        }
        assert ("sessions", {}) in service.calls


@pytest.mark.asyncio
async def test_new_conversation_creates_201_and_lists(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)

        explicit = await client.post(
            "/api/v1/mobile/conversations",
            json={"title": "New chat", "workspace_id": "ws-1"},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert explicit.status == 201
        assert explicit.headers["Cache-Control"] == "no-store"
        assert (await explicit.json())["session"] == {
            "id": "s-new-1",
            "title": "New chat",
            "workspace_id": "ws-1",
            "has_folder_tools": False,
        }

        # An empty body still creates a session (default title/workspace).
        default = await client.post(
            "/api/v1/mobile/conversations",
            json={},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert default.status == 201
        default_session = (await default.json())["session"]
        assert default_session["id"] == "s-new-2"
        assert default_session["title"]
        assert default_session["workspace_id"] == "global"

        listing = await (await client.get("/api/v1/mobile/conversations")).json()
        assert [session["id"] for session in listing["sessions"]] == [
            "s-new-2",
            "s-new-1",
            "s-known",
        ]
        assert ("create_session", {"title": "New chat", "workspace_id": "ws-1"}) in (
            service.calls
        )


@pytest.mark.asyncio
async def test_send_202_then_polled_history_shows_the_new_message(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)

        before = await (
            await client.get("/api/v1/mobile/conversations/s-known/messages")
        ).json()
        assert [message["id"] for message in before["messages"]] == ["m1", "m2"]
        assert before["cursor"] == "m2"

        sent = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "phone question"},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert sent.status == 202
        assert (await sent.json()) == {"ok": True}
        assert service.calls[-1] == (
            "send",
            {"session_id": "s-known", "draft": "phone question"},
        )

        after = await (
            await client.get(
                "/api/v1/mobile/conversations/s-known/messages?after=m2"
            )
        ).json()
        assert [message["id"] for message in after["messages"]] == ["m3"]
        assert after["messages"][0]["role"] == "user"
        assert after["messages"][0]["content"] == "phone question"
        assert after["cursor"] == "m3"


@pytest.mark.asyncio
async def test_unknown_session_is_404_on_every_route(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)

        history = await client.get("/api/v1/mobile/conversations/nope/messages")
        assert history.status == 404
        assert (await history.json()) == {"error": "unknown_session"}

        sent = await client.post(
            "/api/v1/mobile/conversations/nope/messages",
            json={"text": "anyone?"},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert sent.status == 404
        assert (await sent.json()) == {"error": "unknown_session"}

        cancelled = await client.post(
            "/api/v1/mobile/conversations/nope/cancel",
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert cancelled.status == 404
        assert (await cancelled.json()) == {"error": "unknown_session"}


@pytest.mark.asyncio
async def test_send_text_bounds(tmp_path, mobile_surface_gate) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)

        empty = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": ""},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert empty.status == 400
        assert (await empty.json()) == {"error": "invalid_text"}

        wrong_type = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": 42},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert wrong_type.status == 400
        assert (await wrong_type.json()) == {"error": "invalid_text"}

        oversized = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "x" * 32769},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert oversized.status == 413
        assert (await oversized.json()) == {"error": "text_too_large"}

        boundary = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "x" * 32768},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert boundary.status == 202


@pytest.mark.asyncio
async def test_unknown_cursor_message_is_400(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        response = await client.get(
            "/api/v1/mobile/conversations/s-known/messages?after=missing-id"
        )

        assert response.status == 400
        assert (await response.json()) == {"error": "invalid_cursor"}


@pytest.mark.asyncio
async def test_cancel_maps_the_service_bool(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)

        stopped = await client.post(
            "/api/v1/mobile/conversations/s-known/cancel",
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert stopped.status == 200
        assert (await stopped.json()) == {"ok": True}

        service.cancel_result = False
        idle = await client.post(
            "/api/v1/mobile/conversations/s-known/cancel",
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert idle.status == 200
        assert (await idle.json()) == {"ok": False}
        assert service.calls.count(("cancel", {"session_id": "s-known"})) == 2


@pytest.mark.asyncio
async def test_approvals_list_and_decide(tmp_path, mobile_surface_gate) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)

        listing = await client.get("/api/v1/mobile/approvals")
        assert listing.status == 200
        assert listing.headers["Cache-Control"] == "no-store"
        body = await listing.json()
        assert body["approvals"][0]["round_id"] == "r1"
        assert body["approvals"][0]["calls"][0]["llm_name"] == "fs_list"

        decided = await client.post(
            "/api/v1/mobile/approvals/r1",
            json={"decisions": {"c1": "approve_once"}},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert decided.status == 200
        assert (await decided.json()) == {"ok": True}
        assert (
            "decide_approval",
            {"round_id": "r1", "decisions": {"c1": "approve_once"}},
        ) in service.calls

        unknown = await client.post(
            "/api/v1/mobile/approvals/no-such-round",
            json={"decisions": {"c1": "deny"}},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert unknown.status == 200
        assert (await unknown.json()) == {"ok": False}

        invalid = await client.post(
            "/api/v1/mobile/approvals/r1",
            json={"decisions": {"c1": "not-a-verdict"}},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert invalid.status == 400
        assert (await invalid.json()) == {"error": "invalid_decisions"}

        missing = await client.post(
            "/api/v1/mobile/approvals/r1",
            json={},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert missing.status == 400
        assert (await missing.json()) == {"error": "invalid_decisions"}


@pytest.mark.asyncio
async def test_send_refused_maps_to_409(tmp_path, mobile_surface_gate) -> None:
    from tldw_chatbook.Web_Server.mobile_service import MobileSendRefused

    service = FakeMobileService()
    service.send_error = MobileSendRefused("hooks not reviewed")
    async with _logged_in_client(tmp_path, service=service) as client:
        response = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "hello?"},
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )

        assert response.status == 409
        assert (await response.json()) == {"error": "send_refused"}


@pytest.mark.asyncio
async def test_send_returns_202_while_the_turn_is_still_pending(
    tmp_path, mobile_surface_gate
) -> None:
    """Non-blocking send (spec 4.2 / live-check round 1, finding 1).

    The scripted turn parks on ``turn_gate``; the POST must answer 202 in
    bounded time with the turn observably still pending, a concurrent send
    must be refused 409, and releasing the gate must let the scripted
    events flow (the full frame sequence is proven by the stream tests,
    which now run against the same async fake).
    """
    service = FakeMobileService()
    service.turn_gate = asyncio.Event()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)
        loop = asyncio.get_running_loop()
        started = loop.time()

        response = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "slow turn"},
            headers={CSRF_HEADER_NAME: csrf},
        )

        assert response.status == 202
        assert (await response.json()) == {"ok": True}
        assert loop.time() - started < 1.0, "send blocked on the turn"
        assert service.turn_running is True

        second = await client.post(
            "/api/v1/mobile/conversations/s-known/messages",
            json={"text": "too soon"},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert second.status == 409
        assert (await second.json()) == {"error": "send_refused"}

        service.turn_gate.set()
        for _ in range(100):
            if not service.turn_running:
                break
            await asyncio.sleep(0.02)
        assert service.turn_running is False

    await _assert_subscribers_drained(service)


@pytest.mark.asyncio
async def test_mobile_surface_false_leaves_the_routes_unregistered(
    tmp_path, mobile_surface_gate
) -> None:
    mobile_surface_gate(False)
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)

        for method, path, kwargs in (
            ("GET", "/api/v1/mobile/conversations", {}),
            ("POST", "/api/v1/mobile/conversations", {"json": {}}),
            (
                "GET",
                "/api/v1/mobile/conversations/s-known/messages",
                {},
            ),
            (
                "POST",
                "/api/v1/mobile/conversations/s-known/messages",
                {"json": {"text": "hi"}},
            ),
            (
                "POST",
                "/api/v1/mobile/conversations/s-known/cancel",
                {},
            ),
            ("GET", "/api/v1/mobile/approvals", {}),
            (
                "POST",
                "/api/v1/mobile/approvals/r1",
                {"json": {"decisions": {}}},
            ),
        ):
            headers = dict(_MOBILE_REQUEST_HEADERS)
            if method == "POST":
                headers[CSRF_HEADER_NAME] = csrf
            response = await client.request(
                method, path, headers=headers, **kwargs
            )
            assert response.status == 404, (method, path)

        # The stream WebSocket 404s too — with the gate off the middleware
        # must not even run the subprotocol challenge against a dead route.
        with pytest.raises(WSServerHandshakeError) as info:
            await client.ws_connect(
                "/api/v1/mobile/stream",
                protocols=["chatbook-v1", f"csrf.{csrf}"],
            )
        assert info.value.status == 404

    assert service.calls == []


@pytest.mark.asyncio
async def test_shutdown_stops_the_mobile_service(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        assert (await client.get("/api/v1/mobile/conversations")).status == 200
        await client.chatbook_test_server.on_shutdown(None)

    assert service.shutdown_count == 1


# --------------------------------------------------------------------- #
# Task 5: the mobile stream WebSocket (GET /api/v1/mobile/stream)       #
# --------------------------------------------------------------------- #


class _EngineErrorService(FakeMobileService):
    """Service whose event subscription fails before the first event."""

    async def events(self):
        raise RuntimeError("engine unavailable")
        yield  # pragma: no cover - makes this an async generator


@pytest.mark.asyncio
async def test_stream_requires_an_authenticated_session(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=_MOBILE_ACCESS_TOKEN,
        public_url=_MOBILE_PUBLIC_ORIGIN,
        allow_insecure_remote_http=True,
        trusted_proxy_addresses=["127.0.0.1"],
    )
    server = _mobile_server(tmp_path, policy, service=service)
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_MOBILE_REQUEST_HEADERS),
    ) as client:
        with pytest.raises(WSServerHandshakeError) as info:
            await client.ws_connect(
                "/api/v1/mobile/stream",
                protocols=["chatbook-v1", "csrf.anything"],
            )

    assert info.value.status == 401
    await _assert_subscribers_drained(service)


@pytest.mark.asyncio
async def test_stream_upgrade_requires_both_subprotocols(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)
        for protocols in (
            None,
            ["chatbook-v1"],
            [f"csrf.{csrf}"],
            ["chatbook-v1", "csrf.not-the-token"],
        ):
            with pytest.raises(WSServerHandshakeError) as info:
                await client.ws_connect("/api/v1/mobile/stream", protocols=protocols)
            assert info.value.status == 401, protocols

    await _assert_subscribers_drained(service)


@pytest.mark.asyncio
async def test_stream_delivers_run_frames_in_order(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)
        async with client.ws_connect(
            "/api/v1/mobile/stream",
            protocols=["chatbook-v1", f"csrf.{csrf}"],
        ) as ws:
            assert ws.protocol == "chatbook-v1"
            # The subscription lands only once the server handler starts
            # draining the generator; wait for it so the scripted emit
            # below cannot race ahead of the subscribe.
            for _ in range(100):
                if service._event_queues:
                    break
                await asyncio.sleep(0.02)
            assert service._event_queues
            sent = await client.post(
                "/api/v1/mobile/conversations/s-known/messages",
                json={"text": "stream me"},
                headers={CSRF_HEADER_NAME: csrf},
            )
            assert sent.status == 202

            frames = []
            while len(frames) < 6:
                message = await asyncio.wait_for(ws.receive(), timeout=5)
                assert message.type == WSMsgType.TEXT
                frames.append(message.json())

    assert [frame["type"] for frame in frames] == [
        "run_started",
        "delta",
        "delta",
        "message_done",
        "run_completed",
        "session_updated",
    ]
    assert all(frame["session_id"] == "s-known" for frame in frames)
    seqs = [frame["seq"] for frame in frames]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert frames[1]["payload"] == {"message_id": "m9", "offset": 0, "text": "Hel"}
    # The client-side close drains the server's subscriber queue.
    await _assert_subscribers_drained(service)


@pytest.mark.asyncio
async def test_stream_answers_client_pings_and_pings_when_idle(
    tmp_path, mobile_surface_gate, monkeypatch
) -> None:
    monkeypatch.setattr(mobile_api, "IDLE_PING_SECONDS", 0.05)
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)
        async with client.ws_connect(
            "/api/v1/mobile/stream",
            protocols=["chatbook-v1", f"csrf.{csrf}"],
        ) as ws:
            await ws.send_json({"type": "ping"})
            seen: set[str] = set()
            for _ in range(8):
                message = await asyncio.wait_for(ws.receive(), timeout=5)
                assert message.type == WSMsgType.TEXT
                frame = message.json()
                assert frame["type"] in {"ping", "pong"}
                seen.add(frame["type"])
                if seen == {"ping", "pong"}:
                    break
            assert seen == {"ping", "pong"}

    await _assert_subscribers_drained(service)


@pytest.mark.asyncio
async def test_stream_closes_4401_when_the_session_is_revoked(
    tmp_path, mobile_surface_gate
) -> None:
    service = FakeMobileService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)
        session_id = await _current_session_id(client)
        async with client.ws_connect(
            "/api/v1/mobile/stream",
            protocols=["chatbook-v1", f"csrf.{csrf}"],
        ) as ws:
            # The revocation channel registers before the subscription,
            # so a live subscriber implies the channel is armed too.
            for _ in range(100):
                if service._event_queues:
                    break
                await asyncio.sleep(0.02)
            assert service._event_queues
            revoked = await client.post(
                "/auth/devices/revoke",
                json={"session_id": session_id},
                headers={CSRF_HEADER_NAME: csrf},
            )
            assert revoked.status == 200
            assert (await revoked.json()) == {"ok": True}

            message = await asyncio.wait_for(ws.receive(), timeout=5)
            assert message.type == WSMsgType.CLOSE
            assert message.data == 4401

    await _assert_subscribers_drained(service)


@pytest.mark.asyncio
async def test_stream_closes_1011_when_the_engine_stream_fails(
    tmp_path, mobile_surface_gate
) -> None:
    service = _EngineErrorService()
    async with _logged_in_client(tmp_path, service=service) as client:
        csrf = await _csrf(client)
        async with client.ws_connect(
            "/api/v1/mobile/stream",
            protocols=["chatbook-v1", f"csrf.{csrf}"],
        ) as ws:
            message = await asyncio.wait_for(ws.receive(), timeout=5)
            assert message.type == WSMsgType.CLOSE
            assert message.data == 1011

    await _assert_subscribers_drained(service)
