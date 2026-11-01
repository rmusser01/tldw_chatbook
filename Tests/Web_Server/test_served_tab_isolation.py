"""Real authentication, WebSockets and Canvas transport with an echo child seam."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from aiohttp import WSMsgType, WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer

from tldw_chatbook.Canvas.control_protocol import CanvasControlClient
from tldw_chatbook.Canvas.gateway import (
    CanvasGateway,
    CanvasGatewayProjection,
    CanvasGatewayScope,
    ServedCanvasControlHandler,
)
from tldw_chatbook.Canvas.profiles import load_profile_snapshot, runtime_snapshot_id
from tldw_chatbook.Canvas.web_auth import SESSION_COOKIE_NAME, WEBSOCKET_PROTOCOL
from tldw_chatbook.config import build_canvas_config_policy
from tldw_chatbook.Web_Server import serve

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.loopback_network,
    pytest.mark.bootstrap_profile,
]


class _BaseServer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.download_manager = object()
        self.debug = False

    async def on_startup(self, app):
        pass

    async def on_shutdown(self, app):
        pass

    async def handle_download(self, request):
        raise web.HTTPNotFound()


class _ChildCanvasAuthority:
    async def describe_selection(self, scope):
        return CanvasGatewayProjection(
            scope=scope,
            options=(),
            title="Echo child Canvas",
            sequence=1,
            parent_revision_id=None,
            source_bytes=0,
            content_sha256="0" * 64,
            origin_message_id="message-1",
            origin_turn_id="turn-1",
            temporary=False,
            following=False,
        )


class _EchoChild:
    """Replace the child process only; the broker/control client stay real."""

    instances = []

    def __init__(self, command, *, write_str, **kwargs):
        self.app_service_id = uuid4().hex
        self._write_str = write_str
        self.control_client = None
        self.stop_entered = asyncio.Event()
        self.allow_stop = asyncio.Event()
        self.allow_stop.set()
        self.instances.append(self)

    def _build_environment(self, width=80, height=24):
        return {}

    async def start(self, width, height):
        environment = self._build_environment(width, height)
        if "CHATBOOK_CANVAS_CONTROL_HOST" in environment:
            handler = ServedCanvasControlHandler()
            handler.bind(
                _ChildCanvasAuthority(),
                CanvasGatewayScope(
                    browser_session_id=self.app_service_id,
                    conversation_session_id="conversation-" + self.app_service_id,
                    canvas_id="canvas-" + self.app_service_id,
                    revision_id="revision-" + self.app_service_id,
                ),
            )
            self.control_client = CanvasControlClient(
                environment,
                runtime_snapshot_id=runtime_snapshot_id(load_profile_snapshot()),
                handler=handler.handle,
            )
            await self.control_client.start()
        await self._write_str("started:" + self.app_service_id)

    async def send_bytes(self, value):
        await self._write_str(value.decode())

    async def stop(self):
        self.stop_entered.set()
        await self.allow_stop.wait()
        if self.control_client is not None:
            await self.control_client.aclose()
            self.control_client = None


class _HTML(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.elements = []
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def terminal_id(self):
        return next(
            attrs for _, attrs in self.elements if attrs.get("id") == "terminal"
        )["data-terminal-id"]


@asynccontextmanager
async def _stack(tmp_path, *, canvas=True, canvas_capacity=None):
    statics = tmp_path / "static"
    (statics / "js").mkdir(parents=True)
    (statics / "js/textual.js").write_text("unused", encoding="utf-8")
    templates = tmp_path / "templates"
    templates.mkdir()
    cls = serve.build_chatbook_web_server_class(_BaseServer, _EchoChild)
    server = cls(
        command="echo-child",
        host="127.0.0.1",
        port=8000,
        title="Test",
        public_url="http://127.0.0.1:8000",
        statics_path=str(statics),
        templates_path=str(templates),
        canvas_policy=build_canvas_config_policy({"canvas": {"enabled": canvas}}),
    )
    if canvas_capacity is not None:
        await server._served_canvas_gateway.aclose()
        server._served_canvas_gateway = CanvasGateway(
            authority=serve._ServedCanvasAuthorityProxy(server),
            profile_snapshot=server._canvas_profile_snapshot,
            max_browser_sessions=canvas_capacity,
            max_shell_bindings=canvas_capacity,
        )
    _EchoChild.instances = []
    try:
        async with TestClient(TestServer(await server._make_app())) as client:
            yield server, client
    finally:
        for child in _EchoChild.instances:
            child.allow_stop.set()


def _login(server):
    grant = server._web_auth.authenticate_local(client_ip="127.0.0.1")
    headers = {
        "Host": "127.0.0.1:8000",
        "Origin": "http://127.0.0.1:8000",
        "Cookie": f"{SESSION_COOKIE_NAME}={grant.cookie_value}",
    }
    return headers, (WEBSOCKET_PROTOCOL, f"csrf.{grant.csrf_token}"), grant.csrf_token


async def _connect(client, headers, protocols, page_id):
    websocket = await client.ws_connect(
        f"/ws?terminal_id={page_id}&width=100&height=40",
        headers=headers,
        protocols=protocols,
    )
    message = await websocket.receive(timeout=3)
    assert message.type == WSMsgType.TEXT, message
    assert message.data.startswith("started:"), message
    return websocket, message.data.removeprefix("started:")


async def _eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("canvas", [True, False])
async def test_two_tabs_with_one_login_keep_independent_terminals_and_logout_together(
    tmp_path, canvas
):
    async with _stack(tmp_path, canvas=canvas) as (server, client):
        headers, protocols, csrf = _login(server)
        first, child_a = await _connect(client, headers, protocols, "a" * 32)
        second, child_b = await _connect(client, headers, protocols, "b" * 32)
        assert child_a != child_b
        for websocket, text in ((first, "first-tab"), (second, "second-tab")):
            await websocket.send_json(["stdin", text])
            assert (await websocket.receive(timeout=3)).data == text
        logged_out = await client.post(
            "/auth/logout",
            headers={**headers, "X-Chatbook-CSRF": csrf},
            allow_redirects=False,
        )
        assert logged_out.status == 303
        for websocket in (first, second):
            assert (await websocket.receive(timeout=3)).type == WSMsgType.CLOSE
        await _eventually(lambda: not server._served_browser_children)


async def test_each_render_issues_a_fresh_page_id_and_owned_script_cache_keys(tmp_path):
    async with _stack(tmp_path) as (server, client):
        headers, _, _ = _login(server)
        pages = [await client.get("/", headers=headers) for _ in range(2)]
        assert all(response.status == 200 for response in pages)
        parsed = [_HTML(await response.text()) for response in pages]
        ids = [page.terminal_id() for page in parsed]
        assert ids[0] != ids[1]
        assert all(re.fullmatch(r"[a-f0-9]{32}", page_id) for page_id in ids)
        assert all(
            response.headers["Cache-Control"] == "no-store" for response in pages
        )
        assert not server._served_browser_children
        for _, attrs in parsed[0].elements:
            path = attrs.get("src", "")
            if "chatbook-auth.js" in path or "served-shell.js" in path:
                assert re.fullmatch(
                    r"[a-f0-9]{16}", parse_qs(urlsplit(path).query)["v"][0]
                )
                response = await client.get(path, headers=headers)
                assert response.status == 200
                assert (
                    parse_qs(urlsplit(path).query)["v"][0]
                    == hashlib.sha256(await response.read()).hexdigest()[:16]
                )


async def test_refresh_starts_while_old_child_cleanup_is_pending(tmp_path):
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        old, child_a = await _connect(client, headers, protocols, "a" * 32)
        child = _EchoChild.instances[-1]
        child.allow_stop.clear()
        try:
            await old.close()
            await asyncio.wait_for(child.stop_entered.wait(), 3)
            fresh, child_b = await _connect(client, headers, protocols, "b" * 32)
            assert child_a != child_b
        finally:
            child.allow_stop.set()
        await _eventually(
            lambda: child_a not in server._served_browser_children.values()
        )
        await fresh.send_json(["stdin", "fresh-tab-survives"])
        assert (await fresh.receive(timeout=3)).data == "fresh-tab-survives"
        await fresh.close()
        await _eventually(lambda: not server._served_browser_children)


async def test_canvas_matches_each_terminal_and_remains_fenced_to_its_login(tmp_path):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        other_headers, other_protocols, _ = _login(server)
        first, child_a = await _connect(client, headers, protocols, "a" * 32)
        second, child_b = await _connect(client, headers, protocols, "b" * 32)
        other, child_c = await _connect(
            client, other_headers, other_protocols, "a" * 32
        )
        states = []
        for request_headers, page_id, child_id in (
            (headers, "a" * 32, child_a),
            (headers, "b" * 32, child_b),
            (other_headers, "a" * 32, child_c),
        ):
            await server._canvas_control_broker.wait_connected(child_id, timeout=3)
            response = await client.get(
                f"/canvas/api/session?terminal_id={page_id}", headers=request_headers
            )
            state = await response.json()
            assert state["status"] == "ready", state
            assert state["canvas_id"] == "canvas-" + child_id
            states.append(state)
        assert len({state["url"] for state in states}) == 3
        for state in states[:2]:
            path = urlsplit(state["url"]).path
            assert (await client.get(path, headers=headers)).status == 200
            assert (await client.get(path, headers=other_headers)).status == 404
        first_url = urlsplit(states[0]["url"])
        first_base = first_url.path.rstrip("/")
        boot = await client.post(
            first_base + "/api/boot",
            headers=headers,
            json={"bootstrap": parse_qs(first_url.fragment)["boot"][0]},
        )
        assert boot.status == 200, await boot.text()
        canvas_cookie = SimpleCookie(boot.headers["Set-Cookie"])["canvas_session"].value
        cookie = headers["Cookie"] + "; canvas_session=" + canvas_cookie
        canvas_csrf = (await boot.json())["csrf"]
        frame_headers = {**headers, "Cookie": cookie, "X-Canvas-CSRF": canvas_csrf}
        owning = await client.post(
            first_base + "/api/frame", headers=frame_headers, json={}
        )
        assert owning.status == 200
        other_base = urlsplit(states[1]["url"]).path.rstrip("/")
        wrong_terminal = await client.post(
            other_base + "/api/frame", headers=frame_headers, json={}
        )
        assert wrong_terminal.status == 403
        copied_cookie = other_headers["Cookie"] + "; canvas_session=" + canvas_cookie
        copied = await client.post(
            first_base + "/api/frame",
            headers={
                **other_headers,
                "Cookie": copied_cookie,
                "X-Canvas-CSRF": canvas_csrf,
            },
            json={},
        )
        assert copied.status == 404
        alien = await client.get(
            "/canvas/api/session?terminal_id=" + "b" * 32, headers=other_headers
        )
        assert await alien.json() == {"status": "terminal_only"}
        await first.close()
        await _eventually(
            lambda: child_a not in server._served_browser_children.values()
        )
        assert (
            await client.get(urlsplit(states[0]["url"]).path, headers=headers)
        ).status == 404
        assert (
            await client.get(urlsplit(states[1]["url"]).path, headers=headers)
        ).status == 200
        await second.close()
        await other.close()


@pytest.mark.parametrize(
    "query",
    [
        "terminal_id=",
        "terminal_id=../bad",
        "terminal_id=" + "a" * 33,
        "terminal_id=" + "a" * 32 + "&terminal_id=" + "b" * 32,
    ],
)
async def test_invalid_page_id_rejected_before_websocket_spawn_and_canvas_lookup(
    tmp_path, query
):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        with pytest.raises(WSServerHandshakeError) as error:
            await client.ws_connect(
                "/ws?" + query, headers=headers, protocols=protocols
            )
        assert error.value.status == 400
        assert not _EchoChild.instances
        assert (
            await client.get("/canvas/api/session?" + query, headers=headers)
        ).status == 400


async def test_duplicate_page_does_not_steal_binding_and_capacity_slot_is_reusable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(serve, "_MAX_SERVED_TERMINALS", 2, raising=False)
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        first, child_a = await _connect(client, headers, protocols, "a" * 32)
        duplicate = await client.ws_connect(
            "/ws?terminal_id=" + "a" * 32, headers=headers, protocols=protocols
        )
        assert (await duplicate.receive(timeout=3)).type == WSMsgType.CLOSE
        second, _ = await _connect(client, headers, protocols, "b" * 32)
        excess = await client.ws_connect(
            "/ws?terminal_id=" + "c" * 32, headers=headers, protocols=protocols
        )
        assert (await excess.receive(timeout=3)).type == WSMsgType.CLOSE
        assert len(server._served_browser_children) == 2
        assert (
            await client.get(
                "/canvas/api/session?terminal_id=" + "a" * 32, headers=headers
            )
        ).status == 404
        assert len(server._served_browser_children) == 2
        await first.send_json(["stdin", "still-first"])
        assert (await first.receive(timeout=3)).data == "still-first"
        await first.close()
        await _eventually(
            lambda: child_a not in server._served_browser_children.values()
        )
        third, _ = await _connect(client, headers, protocols, "c" * 32)
        await second.close()
        await third.close()


async def test_owned_scripts_send_page_id_without_corrupting_upstream_size_query(
    tmp_path,
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the owned browser scripts")
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, _, csrf = _login(server)
        scripts = [
            await (await client.get(path, headers=headers)).text()
            for path in (
                "/static/js/chatbook-auth.js",
                "/static/chatbook/served-shell.js",
            )
        ]
        harness = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const calls = {fetches: []};
const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {
    dataset: {}, classList: {add() {}, remove() {}}, textContent: '',
    addEventListener() {}, setAttribute() {}, getAttribute() {},
    removeAttribute() {}, focus() {},
  });
  return nodes.get(id);
}
node('terminal').dataset.terminalId = 'a'.repeat(32);
const location = {protocol: 'http:', host: '127.0.0.1:8000', origin: 'http://127.0.0.1:8000'};
const window = {
  location,
  WebSocket: function(url, protocols) {calls.socket = {url: String(url), protocols};},
  addEventListener() {}, setInterval() {return 1;}, clearInterval() {},
};
const context = vm.createContext({
  window, location, URL, Event, document: {
    getElementById: node,
    querySelector() {return {content: input.csrf};},
  },
  fetch(url) {calls.fetches.push(url); return new Promise(() => {});},
});
for (const script of input.scripts) vm.runInContext(script, context);
new window.WebSocket(node('terminal').dataset.sessionWebsocketUrl + '?width=100&height=40');
process.stdout.write(JSON.stringify(calls));
"""
        result = subprocess.run(
            [node, "-e", harness],
            input=json.dumps({"scripts": scripts, "csrf": csrf}),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        calls = json.loads(result.stdout)
        socket = urlsplit(calls["socket"]["url"])
        assert socket.netloc == "127.0.0.1:8000"
        assert parse_qs(socket.query) == {
            "width": ["100"],
            "height": ["40"],
            "terminal_id": ["a" * 32],
        }
        assert calls["socket"]["protocols"] == [WEBSOCKET_PROTOCOL, f"csrf.{csrf}"]
        assert calls["fetches"] == ["/canvas/api/session?terminal_id=" + "a" * 32]


async def test_login_expiry_closes_every_terminal_tab(tmp_path):
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        session = server._web_auth.sessions_snapshot()[0]
        session.absolute_expires_at = server._web_auth._clock() + 0.5
        first, _ = await _connect(client, headers, protocols, "a" * 32)
        second, _ = await _connect(client, headers, protocols, "b" * 32)
        for websocket in (first, second):
            message = await websocket.receive(timeout=3)
            assert message.type == WSMsgType.CLOSE
            assert message.data == 4001
        await _eventually(lambda: not server._served_browser_children)


async def test_live_canvas_disable_keeps_terminal_ownership_and_input(
    tmp_path, monkeypatch
):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        first, _ = await _connect(client, headers, protocols, "a" * 32)
        second, _ = await _connect(client, headers, protocols, "b" * 32)
        monkeypatch.setattr(serve, "get_canvas_execution_enabled", lambda: False)
        await _eventually(lambda: server._canvas_control_broker is None)
        disabled = await client.get(
            "/canvas/api/session?terminal_id=" + "a" * 32, headers=headers
        )
        assert disabled.status == 404
        assert len(server._served_browser_children) == 2
        for websocket in (first, second):
            await websocket.send_json(["stdin", "terminal-survives"])
            assert (await websocket.receive(timeout=3)).data == "terminal-survives"
            await websocket.close()
        await _eventually(lambda: not server._served_browser_children)


async def test_legacy_client_retains_its_single_child_scope(tmp_path):
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        legacy = await client.ws_connect("/ws", headers=headers, protocols=protocols)
        assert (await legacy.receive(timeout=3)).data.startswith("started:")
        sibling, _ = await _connect(client, headers, protocols, "a" * 32)
        await legacy.send_json(["stdin", "legacy-still-works"])
        assert (await legacy.receive(timeout=3)).data == "legacy-still-works"
        await legacy.close()
        await sibling.close()


async def test_closed_terminal_releases_booted_canvas_capacity_for_a_new_page(tmp_path):
    async with _stack(tmp_path, canvas_capacity=2) as (server, client):
        headers, protocols, _ = _login(server)
        sockets = []
        for page_id in ("a" * 32, "b" * 32):
            websocket, child_id = await _connect(client, headers, protocols, page_id)
            sockets.append(websocket)
            await server._canvas_control_broker.wait_connected(child_id, timeout=3)
            response = await client.get(
                "/canvas/api/session?terminal_id=" + page_id, headers=headers
            )
            state = await response.json()
            launch = urlsplit(state["url"])
            boot = await client.post(
                launch.path.rstrip("/") + "/api/boot",
                headers=headers,
                json={"bootstrap": parse_qs(launch.fragment)["boot"][0]},
            )
            assert boot.status == 200
        await sockets[0].close()
        await _eventually(lambda: len(server._served_browser_children) == 1)
        fresh, child_id = await _connect(client, headers, protocols, "c" * 32)
        await server._canvas_control_broker.wait_connected(child_id, timeout=3)
        response = await client.get(
            "/canvas/api/session?terminal_id=" + "c" * 32, headers=headers
        )
        assert response.status == 200, await response.text()
        state = await response.json()
        assert state["status"] == "ready"
        assert state["canvas_id"] == "canvas-" + child_id
        await sockets[1].send_json(["stdin", "sibling-keeps-its-canvas"])
        assert (await sockets[1].receive(timeout=3)).data == "sibling-keeps-its-canvas"
        await sockets[1].close()
        await fresh.close()


@pytest.mark.parametrize("transition", ["close", "disable"])
async def test_pending_canvas_poll_cannot_recreate_a_retired_page(
    tmp_path, monkeypatch, transition
):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        owner, child_id = await _connect(client, headers, protocols, "a" * 32)
        sibling, _ = await _connect(client, headers, protocols, "b" * 32)
        broker = server._canvas_control_broker
        await broker.wait_connected(child_id, timeout=3)
        retired_scope = next(
            scope
            for scope, child in server._served_browser_children.items()
            if child == child_id
        )
        snapshot_ready = asyncio.Event()
        release_snapshot = asyncio.Event()
        original_request = broker.request

        async def delayed_snapshot(request_child, message_type, payload, **kwargs):
            response = await original_request(
                request_child, message_type, payload, **kwargs
            )
            if request_child == child_id and message_type == "scope.snapshot.request":
                snapshot_ready.set()
                await release_snapshot.wait()
            return response

        monkeypatch.setattr(broker, "request", delayed_snapshot)
        pending = asyncio.create_task(
            client.get("/canvas/api/session?terminal_id=" + "a" * 32, headers=headers)
        )
        try:
            await asyncio.wait_for(snapshot_ready.wait(), timeout=3)
            if transition == "close":
                await owner.close()
                await _eventually(
                    lambda: retired_scope not in server._served_browser_children
                )
            else:
                await server._disable_canvas_runtime()
        finally:
            release_snapshot.set()
        response = await asyncio.wait_for(pending, timeout=3)
        state = await response.json()
        assert state["status"] != "ready"
        assert "url" not in state
        assert retired_scope not in server._served_canvas_launches
        assert not server._served_canvas_gateway.has_shell_binding(retired_scope)
        await sibling.send_json(["stdin", "sibling-survives-late-poll"])
        assert (await sibling.receive(timeout=3)).data == "sibling-survives-late-poll"
        await owner.close()
        await sibling.close()
