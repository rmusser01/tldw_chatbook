"""Real-browser coverage for Chatbook's owned served terminal/Canvas shell."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tomllib
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from playwright.async_api import async_playwright, expect

from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.control_protocol import (
    CONTROL_PROTOCOL_VERSION,
    CanvasControlBroker,
    CanvasControlClient,
    ControlMessage,
    ControlProtocolError,
    decode_control_frame,
    encode_control_frame,
)
from tldw_chatbook.Canvas.gateway import (
    BridgeConfirmationRequest,
    BridgeConfirmationResponse,
    BridgePreparationResponse,
    CanvasGateway,
    CanvasGatewayEvent,
    CanvasGatewayOption,
    CanvasGatewayProjection,
    CanvasGatewayScope,
    CanvasSourceResponse,
    ServedCanvasControlHandler,
)
from tldw_chatbook.Canvas.limits import CanvasLimits, sha256_utf8
from tldw_chatbook.Canvas.models import CanvasBridgeRequest, CanvasScope
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Canvas.service import CanvasService
from tldw_chatbook.Canvas.web_auth import (
    SESSION_COOKIE_NAME,
    RequestFacts,
    build_web_auth_policy,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Web_Server import serve

pytestmark = [pytest.mark.loopback_network, pytest.mark.asyncio]

_SERVED_STATIC = (
    Path(__file__).resolve().parents[3] / "tldw_chatbook" / "Web_Server" / "static"
)


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
        self.statics_path = Path(statics_path)
        self.templates_path = Path(templates_path)
        self.download_manager = object()
        self.debug = False

    async def on_startup(self, _app) -> None:
        return None

    async def on_shutdown(self, _app) -> None:
        return None

    async def handle_download(self, _request):
        return web.Response(text="download")


class _FlowBroker:
    """Content-free browser fixture for the parent/child session seam."""

    def __init__(self) -> None:
        self.states: dict[str, dict[str, object]] = {}

    async def browser_state(self, child_id: str) -> dict[str, object]:
        state = self.states.get(child_id)
        if state is None:
            raise RuntimeError("child_not_connected")
        return dict(state)


class _MountedAuthority:
    def __init__(self) -> None:
        self.effects: list[str] = []
        self.events: list[CanvasGatewayEvent] = []

    async def resolve_render_plan(self, scope):
        return compile_canvas_document(f"<!doctype html><h1>{scope.revision_id}</h1>")

    async def read_source(self, scope):
        source = f"<!doctype html><h1>{scope.revision_id}</h1>"
        return CanvasSourceResponse(source, sha256_utf8(source))

    async def describe_selection(self, scope):
        source = f"<!doctype html><h1>{scope.revision_id}</h1>"
        return CanvasGatewayProjection(
            scope=scope,
            options=(CanvasGatewayOption(scope.canvas_id, scope.revision_id, "Plan"),),
            title="Plan",
            sequence=1,
            parent_revision_id=None,
            source_bytes=len(source.encode("utf-8")),
            content_sha256=sha256_utf8(source),
            origin_message_id="message-1",
            origin_turn_id="turn-1",
            temporary=False,
            following=False,
        )

    async def read_events(self, scope, *, after_event_id):
        events = tuple(
            event for event in self.events if event.canvas_id == scope.canvas_id
        )
        if after_event_id is None:
            return events[-1:]
        for index, event in enumerate(events):
            if event.event_id == after_event_id:
                return events[index + 1 :]
        return events[-1:]

    async def prepare_bridge(self, scope, request):
        complete_text = request.submit_text() if request.kind == "submit" else None
        return (
            BridgePreparationResponse(
                request_id=request.request_id,
                kind=request.kind,
                conversation_id=scope.conversation_session_id,
                canvas_id=scope.canvas_id,
                revision_id=scope.revision_id,
                canvas_title="Plan",
                revision_number=1,
                complete_text=complete_text,
                byte_size=len((complete_text or "").encode("utf-8")),
            ),
            object(),
        )

    async def confirm_bridge(
        self, scope, request, *, settlement=None, preparation=None
    ):
        del scope, preparation
        settled = (
            request.approved
            and settlement is not None
            and settlement.try_settle(
                lambda: self.effects.append(request.request.request_id)
            )
        )
        status = "confirmed" if settled else "cancelled"
        return BridgeConfirmationResponse(request.request.request_id, status)


def _server(tmp_path: Path, *, port: int = 8000):
    statics = tmp_path / "static"
    templates = tmp_path / "templates"
    (statics / "js").mkdir(parents=True)
    templates.mkdir()
    (statics / "js" / "textual.js").write_text(
        "document.body.dataset.terminalRuntime = 'ready';",
        encoding="utf-8",
    )
    policy = build_web_auth_policy(host="127.0.0.1", port=port, access_token=None)
    cls = serve.build_chatbook_web_server_class(_FakeBaseServer)
    server = cls(
        command="python -m tldw_chatbook.app",
        host="127.0.0.1",
        port=port,
        title="Chatbook test",
        public_url=f"http://127.0.0.1:{port}",
        statics_path=str(statics),
        templates_path=str(templates),
        web_auth_policy=policy,
    )
    server._canvas_control_broker = _FlowBroker()
    return server


def _chromium_executable(browser_type: object) -> str:
    configured = os.environ.get("TLDW_CANVAS_CHROMIUM_EXECUTABLE")
    if configured and Path(configured).is_file():
        return configured
    declared = Path(browser_type.executable_path)
    if declared.is_file():
        return str(declared)
    root = Path(__file__).resolve().parents[3]
    caches = [Path.home() / "Library" / "Caches" / "ms-playwright"]
    caches.extend(
        ancestor / "Library" / "Caches" / "ms-playwright" for ancestor in root.parents
    )
    candidates = sorted(
        (
            executable
            for cache in caches
            for pattern in (
                "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
                "chromium_headless_shell-*/chrome-mac/headless_shell",
                "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
                "chromium-*/chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium",
            )
            for executable in cache.glob(pattern)
        ),
        reverse=True,
    )
    executable = str(candidates[0]) if candidates else shutil.which("chromium")
    if not executable:
        pytest.fail("real Playwright Chromium is required for the served Canvas flow")
    return executable


async def _browser_app(server) -> web.Application:
    app = await server._make_app()
    app.on_startup.clear()
    app.on_shutdown.clear()
    return app


def _response_cookie(response, name: str) -> str:
    cookies = SimpleCookie()
    for value in response.headers.getall("Set-Cookie", []):
        cookies.load(value)
    return cookies[name].value


async def _load_owned_shell(page) -> None:
    source = (_SERVED_STATIC / "served_shell.html").read_text(encoding="utf-8")
    source = (
        source.replace("__CHATBOOK_CSRF__", "test-csrf")
        .replace("__CHATBOOK_TITLE__", "Chatbook test")
        .replace("__APP_WEBSOCKET_URL__", "ws://127.0.0.1:8000/ws")
        .replace("__FONT_SIZE__", "12")
    )
    # The outer server owns the first two scripts. Browser-flow tests install
    # only Chatbook's served-shell asset so no test double replaces its logic.
    source = source.replace(
        '<script src="/static/js/chatbook-auth.js"></script>', ""
    ).replace('<script src="/static/js/textual.js"></script>', "")
    source = source.replace(
        '<script src="/static/chatbook/served-shell.js" defer></script>', ""
    )
    source = source.replace(
        '<link rel="stylesheet" href="/static/css/xterm.css">', ""
    ).replace('<link rel="stylesheet" href="/static/chatbook/served-shell.css">', "")

    async def fulfill(route) -> None:
        if "/canvas/fixture/" in route.request.url:
            await route.fulfill(
                status=200,
                content_type="text/html",
                body=(
                    "<!doctype html><style>body{font:16px system-ui;margin:0;padding:40px;"
                    "color:#172026;background:#f7f8f6}h1{font-size:36px;margin:0 0 12px}"
                    "p{max-width:48ch;line-height:1.6}</style>"
                    "<h1>Release planner</h1><p>Interactive Canvas preview is isolated "
                    "from the terminal while remaining on the authenticated Chatbook origin.</p>"
                ),
            )
            return
        await route.fulfill(
            status=200,
            content_type="text/html",
            body="<!doctype html><title>fixture</title>",
        )

    await page.route("http://chatbook.test/**", fulfill)
    await page.goto("http://chatbook.test/")
    await page.set_content(source)
    await page.add_style_tag(path=_SERVED_STATIC / "served_shell.css")
    await page.add_script_tag(path=_SERVED_STATIC / "served_shell.js")


async def test_owned_shell_mounts_from_chatbook_origin_before_canvas(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    server = _server(tmp_path, port=unused_tcp_port)
    app = await _browser_app(server)
    test_server = TestServer(app, host="127.0.0.1", port=unused_tcp_port)
    broker = CanvasControlBroker()
    child: CanvasControlClient | None = None
    await test_server.start_server()
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=_chromium_executable(playwright.chromium),
            )
            page = await browser.new_page(viewport={"width": 1024, "height": 720})
            await page.goto(str(test_server.make_url("/")))
            await expect(page.locator("#terminal-region")).to_be_visible()
            await expect(page.locator("#served-canvas-region")).to_be_hidden()
            await expect(page.get_by_text("Terminal only", exact=True)).to_be_visible()
            assert (
                await page.locator("#terminal").get_attribute(
                    "data-session-websocket-url"
                )
                == f"ws://127.0.0.1:{unused_tcp_port}/ws"
            )

            await broker.start()
            server._canvas_control_broker = broker
            handler = ServedCanvasControlHandler()
            handler.bind(
                _MountedAuthority(),
                CanvasGatewayScope(
                    browser_session_id="child-browser-mount",
                    conversation_session_id="conversation-browser-mount",
                    canvas_id="canvas-browser-mount",
                    revision_id="revision-browser-mount",
                ),
            )
            child_launch = broker.issue_child("child-browser-mount")
            child = CanvasControlClient(
                child_launch.environment, handler=handler.handle
            )
            await child.start()
            cookies = await page.context.cookies()
            outer_cookie = next(
                cookie["value"]
                for cookie in cookies
                if cookie["name"] == SESSION_COOKIE_NAME
            )
            browser_session = server._web_auth.authenticate_request(
                RequestFacts(
                    method="GET",
                    path="/",
                    peer_ip="127.0.0.1",
                    scheme="http",
                    host=f"127.0.0.1:{unused_tcp_port}",
                    cookie_value=outer_cookie,
                )
            )
            server.bind_served_browser(
                browser_session.session_id, "child-browser-mount"
            )
            await expect(page.locator("#served-canvas-region")).to_be_visible(
                timeout=5000
            )
            canvas_shell = page.frame_locator("#served-canvas-frame")
            await expect(canvas_shell.locator("#canvas-heading")).to_be_attached(
                timeout=5000
            )
            await expect(canvas_shell.locator("#canvas-title")).to_have_value("Plan")
            await expect(canvas_shell.locator("#connection-state")).to_have_text(
                "Connected"
            )
            await canvas_shell.locator("#close-button").click()
            await expect(page.locator("#served-canvas-region")).to_be_hidden()
            await expect(page.locator("#served-open-canvas")).to_be_visible()
            await page.wait_for_timeout(1500)
            await expect(page.locator("#served-canvas-region")).to_be_hidden()
            await page.locator("#served-open-canvas").click()
            await expect(page.locator("#served-canvas-region")).to_be_visible()
            await browser.close()
    finally:
        if child is not None:
            await child.aclose()
        await broker.aclose()
        await test_server.close()
        await server._served_canvas_gateway.aclose()


async def test_mounted_production_authority_renders_and_settles_submit(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    """Catch opaque-renderer auth and durable-conversation bridge mismatches."""

    session_id = "runtime-session"
    conversation_id = "durable-conversation"
    child_id = "child-production-authority"
    drafts: list[str] = []
    db = CharactersRAGDB(tmp_path / "served-production.sqlite", "served-production")
    conversations = ChatConversationService(db)
    conversation_id = conversations.create_conversation(
        id=conversation_id,
        title="Served production",
        scope_type="global",
        state="in-progress",
    )
    user_id = db.add_message(
        {
            "id": "served-production-user",
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "Build the profile.",
        }
    )
    assistant_id = db.add_message(
        {
            "id": "served-production-assistant",
            "conversation_id": conversation_id,
            "parent_message_id": user_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "Profile Canvas",
        }
    )
    db.set_conversation_active_cursor(
        conversation_id,
        active_leaf_message_id=assistant_id,
        before_message_id=None,
    )
    controller = ConsoleCanvasController(durable_service=CanvasService(db))

    def scope_resolver(requested: str) -> CanvasScope:
        assert requested == session_id
        return CanvasScope(
            session_id=session_id,
            conversation_id=conversation_id,
            active_message_ids=(user_id, assistant_id),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="served-production-run",
        )

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=scope_resolver,
        canvas_controller=controller,
        bridge_prepare=lambda _target: drafts.append,
    )
    created = authority.import_html(
        session_id=session_id,
        source=(
            "<!doctype html><h1 id='profile-identity'>Profile Alpha</h1>"
            "<button id='send-result'>Send result</button>"
            "<script>document.getElementById('send-result').addEventListener("
            "'click', () => canvas.submit({profile: 'alpha'}));</script>"
        ),
        create_new=True,
    )
    child_scope = authority.gateway_scope(
        session_id=session_id,
        browser_session_id=child_id,
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    handler = ServedCanvasControlHandler()
    handler.bind(authority, child_scope)

    server = _server(tmp_path, port=unused_tcp_port)
    app = await _browser_app(server)
    test_server = TestServer(app, host="127.0.0.1", port=unused_tcp_port)
    broker = CanvasControlBroker()
    child: CanvasControlClient | None = None
    await test_server.start_server()
    await broker.start()
    server._canvas_control_broker = broker
    try:
        child_launch = broker.issue_child(child_id)
        child = CanvasControlClient(child_launch.environment, handler=handler.handle)
        await child.start()

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=_chromium_executable(playwright.chromium),
            )
            page = await browser.new_page(viewport={"width": 1100, "height": 760})
            page.set_default_timeout(7_000)
            await page.goto(str(test_server.make_url("/")))
            outer_cookie = next(
                cookie["value"]
                for cookie in await page.context.cookies()
                if cookie["name"] == SESSION_COOKIE_NAME
            )
            browser_session = server._web_auth.authenticate_request(
                RequestFacts(
                    method="GET",
                    path="/",
                    peer_ip="127.0.0.1",
                    scheme="http",
                    host=f"127.0.0.1:{unused_tcp_port}",
                    cookie_value=outer_cookie,
                )
            )
            server.bind_served_browser(browser_session.session_id, child_id)

            canvas_shell = page.frame_locator("#served-canvas-frame")
            preview = canvas_shell.frame_locator("#canvas-preview")
            await expect(preview.locator("#profile-identity")).to_have_text(
                "Profile Alpha"
            )
            await expect(canvas_shell.locator("#loading-state")).to_be_hidden()

            await preview.get_by_role("button", name="Send result").click()
            await expect(canvas_shell.locator("#bridge-dialog")).to_be_visible()
            await canvas_shell.get_by_role("button", name="Send to composer").click()
            await expect(canvas_shell.locator("#bridge-dialog")).to_be_hidden()
            assert drafts == ['{"profile":"alpha"}']
            await browser.close()
    finally:
        if child is not None:
            await child.aclose()
        await broker.aclose()
        await test_server.close()
        await server._served_canvas_gateway.aclose()
        db.close_connection()


async def test_two_real_browser_profiles_stay_isolated_when_one_child_disconnects(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    server = _server(tmp_path, port=unused_tcp_port)
    app = await _browser_app(server)
    test_server = TestServer(app, host="127.0.0.1", port=unused_tcp_port)
    broker = CanvasControlBroker()
    children: list[CanvasControlClient] = []
    await test_server.start_server()
    await broker.start()
    server._canvas_control_broker = broker
    try:
        for child_id, canvas_id, revision_id in (
            ("child-profile-a", "canvas-profile-a", "revision-profile-a"),
            ("child-profile-b", "canvas-profile-b", "revision-profile-b"),
        ):
            handler = ServedCanvasControlHandler()
            handler.bind(
                _MountedAuthority(),
                CanvasGatewayScope(
                    browser_session_id=child_id,
                    conversation_session_id=f"conversation-{child_id}",
                    canvas_id=canvas_id,
                    revision_id=revision_id,
                ),
            )
            launch = broker.issue_child(child_id)
            child = CanvasControlClient(launch.environment, handler=handler.handle)
            await child.start()
            children.append(child)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=_chromium_executable(playwright.chromium),
            )
            contexts = [await browser.new_context() for _ in range(2)]
            pages = [await context.new_page() for context in contexts]
            for page in pages:
                await page.goto(str(test_server.make_url("/")))

            for page, child_id in zip(
                pages, ("child-profile-a", "child-profile-b"), strict=True
            ):
                cookie = next(
                    item["value"]
                    for item in await page.context.cookies()
                    if item["name"] == SESSION_COOKIE_NAME
                )
                session = server._web_auth.authenticate_request(
                    RequestFacts(
                        method="GET",
                        path="/",
                        peer_ip="127.0.0.1",
                        scheme="http",
                        host=f"127.0.0.1:{unused_tcp_port}",
                        cookie_value=cookie,
                    )
                )
                server.bind_served_browser(session.session_id, child_id)

            for page in pages:
                await expect(page.locator("#served-canvas-region")).to_be_visible(
                    timeout=5000
                )
                await expect(
                    page.frame_locator("#served-canvas-frame").locator(
                        "#connection-state"
                    )
                ).to_have_text("Connected")

            first_url = (
                await pages[0].locator("#served-canvas-frame").get_attribute("src")
            )
            second_url = (
                await pages[1].locator("#served-canvas-frame").get_attribute("src")
            )
            assert first_url and second_url and first_url != second_url

            await children[0].aclose()
            await expect(
                pages[0].get_by_text("Canvas reconnecting", exact=True)
            ).to_be_visible(timeout=5000)
            await expect(pages[0].locator("#terminal-region")).to_be_visible()
            await expect(pages[1].locator("#served-canvas-region")).to_be_visible()
            await expect(
                pages[1]
                .frame_locator("#served-canvas-frame")
                .locator("#connection-state")
            ).to_have_text("Connected")

            assert (
                await pages[0]
                .locator("#terminal")
                .get_attribute("data-session-websocket-url")
            )
            await browser.close()
    finally:
        for child in children:
            await child.aclose()
        await broker.aclose()
        await test_server.close()
        await server._served_canvas_gateway.aclose()


async def test_terminal_websocket_echo_survives_private_control_channel_loss(
    tmp_path: Path,
) -> None:
    class EchoAppService:
        instances: ClassVar[list[EchoAppService]] = []

        def __init__(
            self,
            _command,
            *,
            write_bytes,
            write_str,
            close,
            download_manager,
            debug,
        ) -> None:
            del write_str, close, download_manager, debug
            self.app_service_id = "app-service-terminal-survival"
            self._write_bytes = write_bytes
            self.control_client: CanvasControlClient | None = None
            self.handler = ServedCanvasControlHandler()
            self.handler.bind(
                _MountedAuthority(),
                CanvasGatewayScope(
                    browser_session_id=self.app_service_id,
                    conversation_session_id="conversation-terminal-survival",
                    canvas_id="canvas-terminal-survival",
                    revision_id="revision-terminal-survival",
                ),
            )
            self.instances.append(self)

        def _build_environment(self, width: int = 80, height: int = 24):
            del width, height
            return {}

        async def start(self, width: int, height: int) -> None:
            del width, height
            self.control_client = CanvasControlClient(
                self._build_environment(), handler=self.handler.handle
            )
            await self.control_client.start()

        async def stop(self) -> None:
            if self.control_client is not None:
                await self.control_client.aclose()
                self.control_client = None

        async def send_bytes(self, payload: bytes) -> None:
            await self._write_bytes(b"echo:" + payload)

        async def set_terminal_size(self, _width: int, _height: int) -> None:
            return None

        async def blur(self) -> None:
            return None

        async def focus(self) -> None:
            return None

    server = _server(tmp_path)
    broker = CanvasControlBroker()
    await broker.start()
    server._canvas_control_broker = broker
    server._chatbook_app_service_class = serve.build_chatbook_app_service_class(
        EchoAppService
    )
    app = await _browser_app(server)
    grant = server._web_auth.authenticate_local(client_ip="127.0.0.1")
    headers = {
        "Host": "127.0.0.1:8000",
        "Origin": "http://127.0.0.1:8000",
        "Cookie": f"{SESSION_COOKIE_NAME}={grant.cookie_value}",
    }

    async with TestClient(TestServer(app)) as client:
        websocket = await client.ws_connect(
            "/ws?width=80&height=24",
            headers=headers,
            protocols=("chatbook-v1", f"csrf.{grant.csrf_token}"),
        )
        for _ in range(100):
            if EchoAppService.instances:
                break
            await asyncio.sleep(0.01)
        assert EchoAppService.instances
        service = EchoAppService.instances[-1]
        await broker.wait_connected(service.app_service_id, timeout=1)

        await broker.revoke_child(service.app_service_id)
        assert service.control_client is not None
        await service.control_client.wait_disconnected(timeout=1)
        await websocket.send_json(["stdin", "terminal survives"])
        message = await websocket.receive(timeout=1)

        assert message.type == WSMsgType.BINARY
        assert message.data == b"echo:terminal survives"
        assert not websocket.closed
        await websocket.close()

    await broker.aclose()
    await server._served_canvas_gateway.aclose()


async def test_owned_shell_starts_terminal_only_then_opens_and_reopens_canvas(
    tmp_path: Path,
) -> None:
    del tmp_path
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await _load_owned_shell(page)

        terminal = page.locator("#terminal-region")
        canvas = page.locator("#served-canvas-region")
        await expect(terminal).to_be_visible()
        await expect(canvas).to_be_hidden()
        await expect(page.get_by_text("Terminal only", exact=True)).to_be_visible()
        await page.evaluate(
            """() => {
              window.__canvasStatusMutations = 0;
              new MutationObserver(() => { window.__canvasStatusMutations += 1; })
                .observe(document.querySelector('#served-canvas-state'), {
                  childList: true, characterData: true, subtree: true
                });
            }"""
        )

        await page.evaluate(
            """() => window.dispatchEvent(new CustomEvent('chatbook:canvas-state', {
              detail: {status: 'ready', url: '/canvas/fixture/canvas-a/revision-1', revision_id: 'revision-1'}
            }))"""
        )
        await expect(canvas).to_be_visible()
        await expect(page.get_by_text("Canvas connected", exact=True)).to_be_visible()
        first_mutation_count = await page.evaluate("window.__canvasStatusMutations")
        await page.evaluate(
            """() => window.dispatchEvent(new CustomEvent('chatbook:canvas-state', {
              detail: {status: 'ready', url: '/canvas/fixture/canvas-a/revision-1', revision_id: 'revision-1'}
            }))"""
        )
        await page.wait_for_timeout(10)
        assert (
            await page.evaluate("window.__canvasStatusMutations")
            == first_mutation_count
        )

        capture_dir = os.environ.get("TLDW_CANVAS_CAPTURE_DIR")
        if capture_dir:
            output = Path(capture_dir)
            output.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=output / "served-canvas-wide.png")

        await page.get_by_role("button", name="Close Canvas").click()
        await expect(canvas).to_be_hidden()
        assert await page.evaluate("document.activeElement.id") == "served-open-canvas"
        await page.get_by_role("button", name="Open Canvas").click()
        await expect(canvas).to_be_visible()
        assert await page.evaluate("document.activeElement.id") == "served-close-canvas"

        await page.set_viewport_size({"width": 390, "height": 844})
        terminal_box = await terminal.bounding_box()
        canvas_box = await canvas.bounding_box()
        assert terminal_box is not None and canvas_box is not None
        assert canvas_box["y"] >= terminal_box["y"]
        if capture_dir:
            await page.screenshot(path=output / "served-canvas-narrow.png")
        await browser.close()


async def test_hot_reload_branch_switch_exact_reopen_and_control_loss_keep_terminal(
    tmp_path: Path,
) -> None:
    del tmp_path
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=_chromium_executable(playwright.chromium),
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await _load_owned_shell(page)
        for revision, mode in (
            ("revision-2", "updated"),
            ("revision-branch", "branch"),
            ("revision-1", "historical"),
        ):
            await page.evaluate(
                """([revision, mode]) => window.dispatchEvent(new CustomEvent(
                  'chatbook:canvas-state', {detail: {
                    status: 'ready', url: `/canvas/fixture/canvas-a/${revision}`,
                    revision_id: revision, mode
                  }}))""",
                [revision, mode],
            )
            await expect(page.locator("#served-canvas-frame")).to_have_attribute(
                "src", f"/canvas/fixture/canvas-a/{revision}"
            )

        await page.evaluate(
            """() => window.dispatchEvent(new CustomEvent('chatbook:canvas-state', {
              detail: {status: 'disconnected'}
            }))"""
        )
        await expect(
            page.get_by_text("Canvas reconnecting", exact=True)
        ).to_be_visible()
        await expect(page.locator("#terminal-region")).to_be_visible()
        assert await page.locator("#terminal").get_attribute(
            "data-session-websocket-url"
        )
        await browser.close()


async def test_two_browser_sessions_cannot_reuse_child_canvas_scope(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    await _browser_app(server)
    server.bind_served_browser("browser-a", "child-a")
    server.bind_served_browser("browser-b", "child-b")
    with pytest.raises(serve.ServedCanvasUnavailable):
        server.bind_served_browser("browser-c", "child-a")
    server._canvas_control_broker.states.update(
        {
            "child-a": {
                "status": "ready",
                "canvas_id": "canvas-a",
                "revision_id": "revision-a",
                "url": "/canvas/a/revision-a",
            },
            "child-b": {
                "status": "ready",
                "canvas_id": "canvas-b",
                "revision_id": "revision-b",
                "url": "/canvas/b/revision-b",
            },
        }
    )

    assert (await server.served_canvas_state("browser-a"))["canvas_id"] == "canvas-a"
    assert (await server.served_canvas_state("browser-b"))["canvas_id"] == "canvas-b"
    with pytest.raises(serve.ServedCanvasUnavailable):
        await server.served_canvas_state("browser-unknown")

    server.unbind_served_browser("browser-a", "child-a")
    with pytest.raises(serve.ServedCanvasUnavailable):
        await server.served_canvas_state("browser-a")


async def test_connected_child_without_canvas_is_terminal_only_then_reconnecting(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    await _browser_app(server)
    broker = CanvasControlBroker()
    await broker.start()
    server._canvas_control_broker = broker
    handler = ServedCanvasControlHandler()
    child_launch = broker.issue_child("child-empty")
    child = CanvasControlClient(child_launch.environment, handler=handler.handle)
    await child.start()
    server.bind_served_browser("browser-empty", "child-empty")

    assert await server.served_canvas_state("browser-empty") == {
        "status": "terminal_only"
    }

    await child.aclose()
    with pytest.raises(serve.ServedCanvasUnavailable):
        await server.served_canvas_state("browser-empty")
    await broker.aclose()


async def test_gateway_mounts_trusted_handlers_on_existing_origin_without_listener() -> (
    None
):
    gateway = CanvasGateway(authority=_MountedAuthority())
    app = web.Application()
    gateway.mount_on_app(app, origin="http://127.0.0.1:8000")
    scope = CanvasGatewayScope(
        browser_session_id="browser-served",
        conversation_session_id="conversation-served",
        canvas_id="canvas-served",
        revision_id="revision-served",
    )
    launch = await gateway.open_shell(scope)

    assert launch.clean_url.startswith("http://127.0.0.1:8000/canvas/")
    assert any("/api/plan" in str(route.resource) for route in app.router.routes())
    assert gateway.start_count == 0
    await gateway.aclose()


async def test_served_shell_assets_are_declared_for_wheel_and_source_archive() -> None:
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]
    assert set(package_data["tldw_chatbook.Web_Server"]) == {
        "static/served_shell.html",
        "static/served_shell.css",
        "static/served_shell.js",
    }
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include tldw_chatbook/Web_Server/static *" in manifest


async def test_child_control_handler_exposes_only_its_bound_canvas_scope() -> None:
    authority = _MountedAuthority()
    scope = CanvasGatewayScope(
        browser_session_id="child-a",
        conversation_session_id="conversation-a",
        canvas_id="canvas-a",
        revision_id="revision-a",
    )
    handler = ServedCanvasControlHandler()
    handler.bind(authority, scope)

    snapshot = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "scope.snapshot.request",
            "request-snapshot",
            9999999999999,
            {},
        )
    )
    assert snapshot.payload["selected_canvas_id"] == "canvas-a"

    read = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "canvas.read.request",
            "request-read",
            9999999999999,
            {"canvas_id": "canvas-a", "revision_id": "revision-a"},
        )
    )
    assert read.payload["canvas_id"] == "canvas-a"
    assert read.payload["render_metadata"]["source"].endswith("<h1>revision-a</h1>")

    with pytest.raises(ControlProtocolError, match="canvas_unavailable"):
        await handler.handle(
            ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "canvas.read.request",
                "request-foreign",
                9999999999999,
                {"canvas_id": "canvas-b", "revision_id": "revision-b"},
            )
        )


async def test_child_snapshot_reconciles_authority_branch_before_stale_read() -> None:
    class BranchAuthority(_MountedAuthority):
        def control_scope_snapshot(self, _scope):
            return CanvasScope(
                session_id="conversation-a",
                conversation_id="conversation-a",
                active_message_ids=("message-branch",),
                selected_canvas_id="canvas-a",
                selected_revision_id="revision-branch",
                run_id="turn-branch",
            )

        async def describe_selection(self, scope):
            assert scope.revision_id == "revision-branch"
            return await super().describe_selection(scope)

    handler = ServedCanvasControlHandler()
    handler.bind(
        BranchAuthority(),
        CanvasGatewayScope(
            browser_session_id="child-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-old",
        ),
    )

    snapshot = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "scope.snapshot.request",
            "request-branch-snapshot",
            9999999999999,
            {},
        )
    )

    assert snapshot.payload["selected_revision_id"] == "revision-branch"
    assert snapshot.payload["active_message_ids"] == ["message-branch"]


async def test_child_forwards_authoritative_events_with_cursor_metadata() -> None:
    authority = _MountedAuthority()
    authority.events.extend(
        (
            CanvasGatewayEvent(
                "event-one",
                "updated",
                "canvas-a",
                "revision-one",
                {"notice": "Updated", "sequence": 1},
            ),
            CanvasGatewayEvent(
                "event-two",
                "disconnected",
                "canvas-a",
                "revision-two",
                {"notice": "unavailable_on_branch"},
            ),
        )
    )
    handler = ServedCanvasControlHandler()
    handler.bind(
        authority,
        CanvasGatewayScope(
            browser_session_id="child-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-two",
        ),
    )

    response = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "canvas.events.request",
            "request-events",
            9999999999999,
            {"after_event_id": "event-one"},
        )
    )

    assert response.payload["events"] == [
        {
            "event_id": "event-two",
            "kind": "disconnected",
            "canvas_id": "canvas-a",
            "revision_id": "revision-two",
            "metadata": {"notice": "unavailable_on_branch"},
        }
    ]


async def test_child_bridge_preparation_is_single_pending_and_self_expiring() -> None:
    authority = _MountedAuthority()
    handler = ServedCanvasControlHandler(preparation_ttl_seconds=0.01)
    handler.bind(
        authority,
        CanvasGatewayScope(
            browser_session_id="child-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-a",
        ),
    )

    def bridge_message(request_id: str) -> ControlMessage:
        return ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "bridge.request",
            f"prepare-{request_id}",
            9999999999999,
            {
                "request": {
                    "version": "canvas-v1",
                    "request_id": request_id,
                    "kind": "submit",
                    "value": request_id,
                }
            },
        )

    first = await handler.handle(bridge_message("request-one"))
    second = await handler.handle(bridge_message("request-two"))
    with pytest.raises(ControlProtocolError, match="bridge_refused"):
        await handler.handle(
            ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "bridge.decision.request",
                "decision-one",
                9999999999999,
                {
                    "request_id": "request-one",
                    "preparation_nonce": first.payload["preparation_nonce"],
                    "approved": True,
                },
            )
        )

    await asyncio.sleep(0.02)
    with pytest.raises(ControlProtocolError, match="bridge_refused"):
        await handler.handle(
            ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "bridge.decision.request",
                "decision-two",
                9999999999999,
                {
                    "request_id": "request-two",
                    "preparation_nonce": second.payload["preparation_nonce"],
                    "approved": True,
                },
            )
        )


async def test_parent_refusal_never_authorizes_child_bridge_effect() -> None:
    class Broker:
        def __init__(self) -> None:
            self.approvals: list[bool] = []

        async def request(self, _child_id, message_type, payload, *, timeout):
            del timeout
            assert message_type == "bridge.decision.request"
            approved = bool(payload["approved"])
            self.approvals.append(approved)
            return ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "bridge.decision.response",
                "reply-decision",
                None,
                {
                    "request_id": payload["request_id"],
                    "status": "confirmed" if approved else "cancelled",
                },
            )

    class RefusingSettlement:
        def reserve_external(self):
            return False

        def commit_external(self):
            raise AssertionError("a refused reservation cannot commit")

    broker = Broker()
    owner = SimpleNamespace(
        _served_browser_children={"browser-a": "child-a"},
        _canvas_control_broker=broker,
    )
    proxy = serve._ServedCanvasAuthorityProxy(owner)
    request = CanvasBridgeRequest("canvas-v1", "request-submit", "submit", "ok")
    result = await proxy.confirm_bridge(
        CanvasGatewayScope(
            browser_session_id="browser-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-a",
        ),
        BridgeConfirmationRequest(True, request),
        settlement=RefusingSettlement(),
        preparation="prepare-parent-refusal",
    )

    assert result.status == "refused"
    assert broker.approvals == [False]


async def test_parent_commits_only_after_child_receipt_and_replays_lost_reply() -> None:
    class Broker:
        def __init__(self) -> None:
            self.calls = 0
            self.effects: list[str] = []

        async def request(self, _child_id, message_type, payload, *, timeout):
            del timeout
            assert message_type == "bridge.decision.request"
            self.calls += 1
            if self.calls == 1:
                self.effects.append(str(payload["request_id"]))
                raise ControlProtocolError("deadline_exceeded")
            return ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "bridge.decision.response",
                "reply-decision",
                None,
                {"request_id": payload["request_id"], "status": "confirmed"},
            )

    class Settlement:
        def __init__(self) -> None:
            self.reserved = False
            self.committed = False

        def reserve_external(self):
            self.reserved = True
            return True

        def commit_external(self):
            assert self.reserved
            self.committed = True
            return True

    broker = Broker()
    owner = SimpleNamespace(
        _served_browser_children={"browser-a": "child-a"},
        _canvas_control_broker=broker,
    )
    settlement = Settlement()
    request = CanvasBridgeRequest("canvas-v1", "request-submit", "submit", "ok")
    result = await serve._ServedCanvasAuthorityProxy(owner).confirm_bridge(
        CanvasGatewayScope(
            browser_session_id="browser-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-a",
        ),
        BridgeConfirmationRequest(True, request),
        settlement=settlement,
        preparation="prepare-retry",
    )

    assert result.status == "confirmed"
    assert settlement.committed
    assert broker.calls == 2
    assert broker.effects == ["request-submit"]


async def test_child_refusal_never_publishes_parent_confirmation() -> None:
    class Broker:
        async def request(self, _child_id, _message_type, payload, *, timeout):
            del timeout
            return ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "bridge.decision.response",
                "reply-decision",
                None,
                {"request_id": payload["request_id"], "status": "refused"},
            )

    class Settlement:
        def __init__(self) -> None:
            self.commit_calls = 0

        def reserve_external(self):
            return True

        def commit_external(self):
            self.commit_calls += 1
            return True

    owner = SimpleNamespace(
        _served_browser_children={"browser-a": "child-a"},
        _canvas_control_broker=Broker(),
    )
    settlement = Settlement()
    request = CanvasBridgeRequest("canvas-v1", "request-submit", "submit", "ok")
    result = await serve._ServedCanvasAuthorityProxy(owner).confirm_bridge(
        CanvasGatewayScope(
            browser_session_id="browser-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-a",
        ),
        BridgeConfirmationRequest(True, request),
        settlement=settlement,
        preparation="prepare-child-refusal",
    )

    assert result.status == "refused"
    assert settlement.commit_calls == 0


async def test_child_bridge_decision_receipt_is_idempotent() -> None:
    authority = _MountedAuthority()
    handler = ServedCanvasControlHandler()
    handler.bind(
        authority,
        CanvasGatewayScope(
            browser_session_id="child-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-a",
        ),
    )
    request = {
        "version": "canvas-v1",
        "request_id": "request-submit",
        "kind": "submit",
        "value": "ok",
    }
    preparation = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "bridge.request",
            "prepare-submit",
            9999999999999,
            {"request": request},
        )
    )
    nonce = preparation.payload["preparation_nonce"]

    responses = []
    for request_id in ("decision-first", "decision-retry"):
        responses.append(
            await handler.handle(
                ControlMessage(
                    CONTROL_PROTOCOL_VERSION,
                    "bridge.decision.request",
                    request_id,
                    9999999999999,
                    {
                        "request_id": "request-submit",
                        "preparation_nonce": nonce,
                        "approved": True,
                    },
                )
            )
        )

    assert [response.payload["status"] for response in responses] == [
        "confirmed",
        "confirmed",
    ]
    assert authority.effects == ["request-submit"]

    next_preparation = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "bridge.request",
            "prepare-next-load",
            9999999999999,
            {"request": request},
        )
    )
    next_nonce = next_preparation.payload["preparation_nonce"]
    assert next_nonce != nonce
    next_response = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "bridge.decision.request",
            "decision-next-load",
            9999999999999,
            {
                "request_id": "request-submit",
                "preparation_nonce": next_nonce,
                "approved": True,
            },
        )
    )
    assert next_response.payload["status"] == "confirmed"
    assert authority.effects == ["request-submit", "request-submit"]
    handler.clear()


async def test_control_frame_accepts_documented_multimegabyte_download_payload() -> (
    None
):
    payload = "x" * CanvasLimits().download_payload_bytes
    message = ControlMessage(
        CONTROL_PROTOCOL_VERSION,
        "bridge.request",
        "request-large-download",
        9999999999999,
        {
            "request": {
                "version": "canvas-v1",
                "request_id": "download-large",
                "kind": "download",
                "value": {
                    "filename": "result.txt",
                    "mime_type": "text/plain",
                    "data": payload,
                },
            }
        },
    )

    frame = encode_control_frame(message)
    decoded = decode_control_frame(frame[4:])

    assert decoded.payload["request"]["value"]["data"] == payload


async def test_mounted_canvas_url_is_not_transferable_between_authenticated_browsers(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    broker = CanvasControlBroker()
    await broker.start()
    server._canvas_control_broker = broker
    app = await _browser_app(server)

    authority_a = _MountedAuthority()
    authority_b = _MountedAuthority()
    handler_a = ServedCanvasControlHandler()
    handler_a.bind(
        authority_a,
        CanvasGatewayScope(
            browser_session_id="child-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id="revision-a",
        ),
    )
    handler_b = ServedCanvasControlHandler()
    handler_b.bind(
        authority_b,
        CanvasGatewayScope(
            browser_session_id="child-b",
            conversation_session_id="conversation-b",
            canvas_id="canvas-b",
            revision_id="revision-b",
        ),
    )
    launch_a = broker.issue_child("child-a")
    launch_b = broker.issue_child("child-b")
    child_a = CanvasControlClient(launch_a.environment, handler=handler_a.handle)
    child_b = CanvasControlClient(launch_b.environment, handler=handler_b.handle)
    await child_a.start()
    await child_b.start()

    grant_a = server._web_auth.authenticate_local(client_ip="127.0.0.1")
    grant_b = server._web_auth.authenticate_local(client_ip="127.0.0.1")

    def browser_session(cookie: str):
        return server._web_auth.authenticate_request(
            RequestFacts(
                method="GET",
                path="/",
                peer_ip="127.0.0.1",
                scheme="http",
                host="127.0.0.1:8000",
                cookie_value=cookie,
            )
        )

    browser_a = browser_session(grant_a.cookie_value)
    browser_b = browser_session(grant_b.cookie_value)
    server.bind_served_browser(browser_a.session_id, "child-a")
    server.bind_served_browser(browser_b.session_id, "child-b")
    state_a = await server.served_canvas_state(browser_a.session_id)
    state_b = await server.served_canvas_state(browser_b.session_id)
    url_a = urlsplit(str(state_a["url"]))
    path_a = url_a.path
    path_b = urlsplit(str(state_b["url"])).path
    bootstrap_a = parse_qs(url_a.fragment)["boot"][0]

    async with TestClient(TestServer(app)) as client:
        headers_a = {
            "Host": "127.0.0.1:8000",
            "Cookie": f"{SESSION_COOKIE_NAME}={grant_a.cookie_value}",
        }
        headers_b = {
            "Host": "127.0.0.1:8000",
            "Cookie": f"{SESSION_COOKIE_NAME}={grant_b.cookie_value}",
        }
        owner = await client.get(path_a, headers=headers_a)
        query_credential = await client.get(
            f"{path_a}?bootstrap=secret", headers=headers_a
        )
        missing_origin = await client.post(
            f"{path_a.rstrip('/')}/api/boot",
            headers={**headers_a, "Content-Type": "application/json"},
            json={"bootstrap": "invalid"},
        )
        copied = await client.get(path_a, headers=headers_b)
        reverse_copied = await client.get(path_b, headers=headers_a)
        shell_id = path_a.rstrip("/").rsplit("/", 1)[-1]
        guessed = await client.get(
            path_a.replace(shell_id, f"shell-{'0' * 32}"), headers=headers_b
        )
        copied_body = await copied.read()
        guessed_body = await guessed.read()

        base_a = path_a.rstrip("/")
        mutation_headers_a = {
            **headers_a,
            "Origin": "http://127.0.0.1:8000",
            "Content-Type": "application/json",
        }
        boot = await client.post(
            f"{base_a}/api/boot",
            headers=mutation_headers_a,
            data=json.dumps({"bootstrap": bootstrap_a}),
        )
        assert boot.status == 200
        boot_body = await boot.json()
        canvas_cookie = _response_cookie(boot, "canvas_session")
        inner_headers_a = {
            **headers_a,
            "Cookie": (
                f"{SESSION_COOKIE_NAME}={grant_a.cookie_value}; "
                f"canvas_session={canvas_cookie}"
            ),
        }
        inner_headers_b = {
            **headers_b,
            "Cookie": (
                f"{SESSION_COOKIE_NAME}={grant_b.cookie_value}; "
                f"canvas_session={canvas_cookie}"
            ),
        }
        authority_a.events.append(
            CanvasGatewayEvent(
                "event-served-updated",
                "updated",
                "canvas-a",
                "revision-a",
                {"notice": "Updated", "sequence": 2},
            )
        )
        owner_events = await client.get(f"{base_a}/api/events", headers=inner_headers_a)
        owner_events_body = await owner_events.json()
        copied_events = await client.get(
            f"{base_a}/api/events", headers=inner_headers_b
        )

        frame = await client.post(
            f"{base_a}/api/frame",
            headers={
                **inner_headers_a,
                "Origin": "http://127.0.0.1:8000",
                "Content-Type": "application/json",
                "X-Canvas-CSRF": boot_body["csrf"],
            },
            data="{}",
        )
        assert frame.status == 200

        async def action_capability(action: str) -> str:
            response = await client.post(
                f"{base_a}/api/actions",
                headers={
                    **inner_headers_a,
                    "Origin": "http://127.0.0.1:8000",
                    "Content-Type": "application/json",
                    "X-Canvas-CSRF": boot_body["csrf"],
                },
                data=json.dumps({"action": action}),
            )
            assert response.status == 200
            return (await response.json())["capability"]

        download_capability = await action_capability("source_download")
        source_download = await client.get(
            f"{base_a}/api/source-download",
            headers={
                **inner_headers_a,
                "Authorization": f"CanvasCapability {download_capability}",
            },
        )
        copied_download = await client.get(
            f"{base_a}/api/source-download",
            headers={
                **inner_headers_b,
                "Authorization": f"CanvasCapability {download_capability}",
            },
        )

        submit_request = {
            "version": "canvas-v1",
            "request_id": "submit-a",
            "kind": "submit",
            "value": {"answer": 42},
        }
        prepare_capability = await action_capability("bridge_prepare")
        prepare = await client.post(
            f"{base_a}/api/bridge/prepare",
            headers={
                **inner_headers_a,
                "Origin": "http://127.0.0.1:8000",
                "Content-Type": "application/json",
                "X-Canvas-CSRF": boot_body["csrf"],
                "Authorization": f"CanvasCapability {prepare_capability}",
            },
            data=json.dumps({"request": submit_request}),
        )
        assert prepare.status == 200, await prepare.text()
        confirm_capability = await action_capability("bridge_confirm")
        submit = await client.post(
            f"{base_a}/api/bridge",
            headers={
                **inner_headers_a,
                "Origin": "http://127.0.0.1:8000",
                "Content-Type": "application/json",
                "X-Canvas-CSRF": boot_body["csrf"],
                "Authorization": f"CanvasCapability {confirm_capability}",
            },
            data=json.dumps({"approved": True, "request": submit_request}),
        )
        submit_body = await submit.json()
        copied_submit = await client.post(
            f"{base_a}/api/bridge",
            headers={
                **inner_headers_b,
                "Origin": "http://127.0.0.1:8000",
                "Content-Type": "application/json",
                "X-Canvas-CSRF": boot_body["csrf"],
                "Authorization": f"CanvasCapability {confirm_capability}",
            },
            data=json.dumps({"approved": True, "request": submit_request}),
        )

        generated_download_request = {
            "version": "canvas-v1",
            "request_id": "download-a",
            "kind": "download",
            "value": {
                "filename": "result.txt",
                "mime_type": "text/plain",
                "data": "result",
            },
        }
        download_prepare_capability = await action_capability("bridge_prepare")
        generated_download_prepare = await client.post(
            f"{base_a}/api/bridge/prepare",
            headers={
                **inner_headers_a,
                "Origin": "http://127.0.0.1:8000",
                "Content-Type": "application/json",
                "X-Canvas-CSRF": boot_body["csrf"],
                "Authorization": (f"CanvasCapability {download_prepare_capability}"),
            },
            data=json.dumps({"request": generated_download_request}),
        )
        generated_download_confirm_capability = await action_capability(
            "bridge_confirm"
        )
        generated_download = await client.post(
            f"{base_a}/api/bridge",
            headers={
                **inner_headers_a,
                "Origin": "http://127.0.0.1:8000",
                "Content-Type": "application/json",
                "X-Canvas-CSRF": boot_body["csrf"],
                "Authorization": (
                    f"CanvasCapability {generated_download_confirm_capability}"
                ),
            },
            data=json.dumps({"approved": True, "request": generated_download_request}),
        )
        generated_download_body = await generated_download.json()

    assert owner.status == 200
    assert owner.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in owner.headers["Content-Security-Policy"]
    assert query_credential.status == 400
    assert missing_origin.status == 403
    assert copied.status == reverse_copied.status == guessed.status == 404
    assert copied_body == guessed_body
    assert copied.headers["Content-Type"] == guessed.headers["Content-Type"]
    assert copied.headers["Cache-Control"] == guessed.headers["Cache-Control"]
    assert owner_events.status == 200
    assert owner_events_body["events"] == [
        {
            "event_id": "event-served-updated",
            "kind": "updated",
            "canvas_id": "canvas-a",
            "revision_id": "revision-a",
            "metadata": {"notice": "Updated", "sequence": 2},
        }
    ]
    assert copied_events.status == 404
    assert source_download.status == 200
    assert source_download.headers["Content-Disposition"].startswith("attachment;")
    assert copied_download.status == 404
    assert submit.status == 200
    assert submit_body["status"] == "confirmed"
    assert copied_submit.status == 404
    assert generated_download_prepare.status == 200
    assert generated_download.status == 200
    assert generated_download_body["status"] == "confirmed"
    assert authority_a.effects == ["submit-a", "download-a"]
    assert authority_b.effects == []
    assert browser_a.session_id != browser_b.session_id
    await child_a.aclose()
    await child_b.aclose()
    await broker.aclose()
