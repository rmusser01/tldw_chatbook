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
from tldw_chatbook.Canvas.config_policy import build_canvas_config_policy
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

    async def send_meta(self, value):
        if value["type"] == "console_reattach":
            for kind in ("console_attachment_begin", "console_attachment_ready"):
                await self._write_str(
                    json.dumps([kind, {"generation": value["generation"]}])
                )
            await self._write_str(json.dumps(["started", self.app_service_id]))

    async def send_bytes(self, value):
        await self._write_str(json.dumps(["echo", value.decode()]))

    async def set_terminal_size(self, width, height):
        pass

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
    query = "" if page_id is None else f"terminal_id={page_id}&"
    websocket = await client.ws_connect(
        f"/ws?{query}width=100&height=40",
        headers=headers,
        protocols=protocols,
    )
    begin = await websocket.receive_json(timeout=3)
    ready = await websocket.receive_json(timeout=3)
    started = await websocket.receive_json(timeout=3)
    assert begin[0] == "console_attachment_begin"
    assert ready == ["console_attachment_ready", begin[1]]
    assert started[0] == "started"
    return websocket, started[1]


async def _echo(websocket, text):
    await websocket.send_json(["stdin", text])
    assert await websocket.receive_json(timeout=3) == ["echo", text]


async def _closed(websocket, *, code=None):
    message = await websocket.receive(timeout=3)
    assert message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED), message
    if code is not None:
        assert message.data == code


async def _eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("canvas", [True, False])
async def test_two_tabs_replace_one_terminal_and_logout_keeps_child(tmp_path, canvas):
    async with _stack(tmp_path, canvas=canvas) as (server, client):
        headers, protocols, csrf = _login(server)
        first, child_a = await _connect(client, headers, protocols, "a" * 32)
        second, child_b = await _connect(client, headers, protocols, "b" * 32)
        assert child_a == child_b and len(_EchoChild.instances) == 1
        await _closed(first)
        await _echo(second, "current-tab")
        logged_out = await client.post(
            "/auth/logout",
            headers={**headers, "X-Chatbook-CSRF": csrf},
            allow_redirects=False,
        )
        assert logged_out.status == 303
        await _closed(second, code=4001)
        await _eventually(lambda: not server._served_browser_children)
        assert not _EchoChild.instances[0].stop_entered.is_set()


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


async def test_refresh_survives_actual_old_attachment_detach_cleanup(
    tmp_path, monkeypatch
):
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        old, child_a = await _connect(client, headers, protocols, "a" * 32)
        attachment = next(iter(server._served_browser_attachments.values()))
        original = attachment.detach
        entered, release, retired = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = 0

        async def delayed_detach():
            nonlocal calls
            calls += 1
            delayed = calls == 1  # actual old WebSocket finally, before replacement
            if delayed:
                entered.set()
                await release.wait()
            await original()
            if delayed:
                retired.set()

        monkeypatch.setattr(attachment, "detach", delayed_detach)
        fresh = None
        try:
            await old.close()
            await asyncio.wait_for(entered.wait(), 3)
            fresh, child_b = await _connect(client, headers, protocols, "a" * 32)
            assert child_a == child_b
            successor = next(iter(server._served_browser_attachments.values()))
            assert successor is not attachment
            await _echo(fresh, "fresh-during-old-cleanup")
            release.set()
            await asyncio.wait_for(retired.wait(), 3)
            await _echo(fresh, "fresh-after-old-cleanup")
            assert next(iter(server._served_browser_attachments.values())) is successor
            assert len(_EchoChild.instances) == 1
            assert not _EchoChild.instances[0].stop_entered.is_set()
        finally:
            release.set()
            await old.close()
            if fresh is not None:
                await fresh.close()
        await _eventually(lambda: not server._served_browser_children)


async def test_canvas_replacement_remains_fenced_to_its_page_and_login(tmp_path):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        other_headers, other_protocols, _ = _login(server)
        first, child = await _connect(client, headers, protocols, "a" * 32)
        await server._canvas_control_broker.wait_connected(child, timeout=3)

        async def state(request_headers, page):
            response = await client.get(
                "/canvas/api/session?terminal_id=" + page, headers=request_headers
            )
            value = await response.json()
            assert value["status"] == "ready", value
            assert value["canvas_id"] == "canvas-" + child
            return value

        original = await state(headers, "a" * 32)
        launch = urlsplit(original["url"])
        base = launch.path.rstrip("/")
        assert (await client.get(base, headers=other_headers)).status == 404
        boot = await client.post(
            base + "/api/boot",
            headers=headers,
            json={"bootstrap": parse_qs(launch.fragment)["boot"][0]},
        )
        assert boot.status == 200
        cookie = SimpleCookie(boot.headers["Set-Cookie"])["canvas_session"].value
        frame_headers = {
            **headers,
            "Cookie": headers["Cookie"] + "; canvas_session=" + cookie,
            "X-Canvas-CSRF": (await boot.json())["csrf"],
        }
        assert (
            await client.post(base + "/api/frame", headers=frame_headers, json={})
        ).status == 200
        second, same_child = await _connect(client, headers, protocols, "b" * 32)
        assert same_child == child
        await _closed(first)
        successor = await state(headers, "b" * 32)
        successor_base = urlsplit(successor["url"]).path.rstrip("/")
        assert successor["url"] != original["url"]
        assert (await client.get(base, headers=headers)).status == 404
        assert (
            await client.post(
                successor_base + "/api/frame", headers=frame_headers, json={}
            )
        ).status == 403
        copied = {
            **other_headers,
            "Cookie": other_headers["Cookie"] + "; canvas_session=" + cookie,
            "X-Canvas-CSRF": frame_headers["X-Canvas-CSRF"],
        }
        assert (
            await client.post(successor_base + "/api/frame", headers=copied, json={})
        ).status == 404
        other, same_child = await _connect(
            client, other_headers, other_protocols, "b" * 32
        )
        assert same_child == child
        await _closed(second)
        foreign = await state(other_headers, "b" * 32)
        assert (
            await client.get(urlsplit(foreign["url"]).path, headers=headers)
        ).status == 404
        assert (await client.get(successor_base, headers=headers)).status == 404
        assert await (
            await client.get(
                "/canvas/api/session?terminal_id=" + "b" * 32, headers=headers
            )
        ).json() == {"status": "terminal_only"}
        await _echo(other, "current-login")
        await other.close()
        assert len(_EchoChild.instances) == 1


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


async def test_same_page_replacement_and_capacity_slot_are_reusable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(serve, "_MAX_SERVED_TERMINALS", 1)
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        previous, child = await _connect(client, headers, protocols, "a" * 32)
        for page in ("a" * 32, "b" * 32, "c" * 32):
            current, same_child = await _connect(client, headers, protocols, page)
            assert same_child == child
            await _closed(previous)
            assert len(server._served_browser_children) == 1
            with pytest.raises(
                serve.ServedCanvasUnavailable, match="terminal_session_capacity"
            ):
                server.bind_served_browser(
                    "extra-capacity-scope", "another-child", attachment=object()
                )
            await _echo(current, "current-page")
            previous = current
        await previous.close()
        await _eventually(lambda: not server._served_browser_children)
        fresh, same_child = await _connect(client, headers, protocols, "a" * 32)
        assert same_child == child and len(_EchoChild.instances) == 1
        await _echo(fresh, "reused-slot")
        await fresh.close()


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
  WebSocket: function(url, protocols) {calls.socket = {url: String(url), protocols}; this.send = function() {}; this.addEventListener = function() {};},
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


async def test_login_expiry_closes_current_replacement_and_retains_child(tmp_path):
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        session = server._web_auth.sessions_snapshot()[0]
        session.absolute_expires_at = server._web_auth._clock() + 0.5
        first, child = await _connect(client, headers, protocols, "a" * 32)
        second, same_child = await _connect(client, headers, protocols, "b" * 32)
        assert same_child == child
        await _closed(first)
        await _closed(second, code=4001)
        await _eventually(lambda: not server._served_browser_children)
        assert not _EchoChild.instances[0].stop_entered.is_set()


async def test_live_canvas_disable_keeps_broker_console_and_current_terminal(
    tmp_path, monkeypatch
):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        first, child = await _connect(client, headers, protocols, "a" * 32)
        second, same_child = await _connect(client, headers, protocols, "b" * 32)
        assert child == same_child
        await _closed(first)
        broker = server._canvas_control_broker
        monkeypatch.setattr(serve, "get_canvas_execution_enabled", lambda: False)
        await _eventually(lambda: server._canvas_disabled_latched)
        assert server._canvas_control_broker is broker
        disabled = await client.get(
            "/canvas/api/session?terminal_id=" + "b" * 32, headers=headers
        )
        assert disabled.status == 404
        assert len(server._served_browser_children) == 1
        await _echo(second, "terminal-survives")
        await broker.wait_connected(child, timeout=3)
        assert (
            await broker.request(child, "scope.snapshot.request", {}, timeout=1)
        ).payload["selected_canvas_id"] == "canvas-" + child
        await second.close()
        await _eventually(lambda: not server._served_browser_children)
        assert not _EchoChild.instances[0].stop_entered.is_set()


async def test_legacy_client_and_page_replace_one_retained_child_scope(tmp_path):
    async with _stack(tmp_path, canvas=False) as (server, client):
        headers, protocols, _ = _login(server)
        legacy, child = await _connect(client, headers, protocols, None)
        sibling, same_child = await _connect(client, headers, protocols, "a" * 32)
        assert child == same_child
        await _closed(legacy)
        await _echo(sibling, "page-replaces-legacy")
        successor, same_child = await _connect(client, headers, protocols, None)
        assert same_child == child
        await _closed(sibling)
        await _echo(successor, "legacy-replaces-page")
        await successor.close()
        assert len(_EchoChild.instances) == 1


async def test_replaced_terminal_releases_booted_canvas_capacity_for_new_page(tmp_path):
    async with _stack(tmp_path, canvas_capacity=1) as (server, client):
        headers, protocols, _ = _login(server)
        previous, previous_url = None, None
        for page in ("a" * 32, "a" * 32, "b" * 32):
            current, child = await _connect(client, headers, protocols, page)
            if previous is not None:
                await _closed(previous)
                assert (await client.get(previous_url, headers=headers)).status == 404
            await server._canvas_control_broker.wait_connected(child, timeout=3)
            response = await client.get(
                "/canvas/api/session?terminal_id=" + page, headers=headers
            )
            state = await response.json()
            assert state["status"] == "ready", state
            launch = urlsplit(state["url"])
            boot = await client.post(
                launch.path.rstrip("/") + "/api/boot",
                headers=headers,
                json={"bootstrap": parse_qs(launch.fragment)["boot"][0]},
            )
            assert boot.status == 200, await boot.text()
            await _echo(current, "current-canvas-owner")
            previous, previous_url = current, launch.path
        await previous.close()
        await _eventually(lambda: not server._served_browser_children)
        assert len(_EchoChild.instances) == 1


@pytest.mark.parametrize("transition", ["close", "disable", "broker"])
async def test_pending_canvas_poll_cannot_recreate_a_retired_page(
    tmp_path, monkeypatch, transition
):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        owner, child_id = await _connect(client, headers, protocols, "a" * 32)
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
            elif transition == "disable":
                await server._disable_canvas_runtime()
            else:
                await broker.aclose()
                server._canvas_control_broker = None
        finally:
            release_snapshot.set()
        response = await asyncio.wait_for(pending, timeout=3)
        state = await response.json()
        assert state["status"] != "ready"
        assert "url" not in state
        assert retired_scope not in server._served_canvas_launches
        assert not server._served_canvas_gateway.has_shell_binding(retired_scope)
        successor, same_child = await _connect(client, headers, protocols, "b" * 32)
        assert same_child == child_id
        await _echo(successor, "successor-survives-late-poll")
        await owner.close()
        await successor.close()


@pytest.mark.parametrize("boundary", ("snapshot", "launch"))
@pytest.mark.parametrize("transition", ("cancel", "disable", "gateway"))
async def test_canvas_await_retirement_releases_launch_lock_and_keeps_console(
    tmp_path, monkeypatch, boundary, transition
):
    async with _stack(tmp_path) as (server, client):
        headers, protocols, _ = _login(server)
        terminal, child = await _connect(client, headers, protocols, "a" * 32)
        scope = next(iter(server._served_browser_children))
        attachment = server._served_browser_attachments[scope]
        broker, gateway = server._canvas_control_broker, server._served_canvas_gateway
        await broker.wait_connected(child, timeout=3)
        entered, release = asyncio.Event(), asyncio.Event()
        target, method = (
            (broker, "request") if boundary == "snapshot" else (gateway, "start")
        )
        original = getattr(target, method)
        first = True

        async def delayed(*args, **kwargs):
            nonlocal first
            if boundary == "snapshot":
                result = await original(*args, **kwargs)
            if first:
                first = False
                entered.set()
                await release.wait()
            return result if boundary == "snapshot" else await original(*args, **kwargs)

        monkeypatch.setattr(target, method, delayed)
        pending = asyncio.create_task(server.served_canvas_state(scope))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if transition == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                if transition == "disable":
                    await server._disable_canvas_runtime()
                    assert server._canvas_control_broker is broker
                else:
                    await gateway.aclose()
                    server._served_canvas_gateway = CanvasGateway(
                        authority=serve._ServedCanvasAuthorityProxy(server),
                        profile_snapshot=server._canvas_profile_snapshot,
                    )
                release.set()
                with pytest.raises(
                    (serve.ServedCanvasUnavailable, RuntimeError),
                    match="canvas_session_unavailable|closed",
                ):
                    await pending
            assert not server._served_canvas_launch_lock.locked()
            assert scope not in server._served_canvas_launches
            assert not gateway.has_shell_binding(scope)
            assert server._served_browser_attachments[scope] is attachment
            await _echo(terminal, "console-survives-" + transition)
            if transition == "cancel":
                release.set()
                state = await server.served_canvas_state(scope)
                launch = urlsplit(state["url"])
                boot = await client.post(
                    launch.path.rstrip("/") + "/api/boot",
                    headers=headers,
                    json={"bootstrap": parse_qs(launch.fragment)["boot"][0]},
                )
                assert boot.status == 200, await boot.text()
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await terminal.close()
            await gateway.aclose()
