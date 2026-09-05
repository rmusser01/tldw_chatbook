"""Real-browser coverage for Chatbook's owned served terminal/Canvas shell."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import sqlite3
import tomllib
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright, expect

from Tests.Canvas.browser.canvas_live_chatbook_child import _ScriptedCanvasGateway
from Tests.Canvas.browser.canvas_live_harness import (
    LiveServedStack,
    ProductRouteRecorder,
    adversarial_cases,
    assert_only_owned_browser_traffic,
    egress_probe,
    exercise_adversarial_preview,
    start_live_served_stack,
)
from Tests.Canvas.browser.canvas_live_harness import (
    chromium_executable as live_chromium_executable,
)
from tldw_chatbook.Agents.agent_models import FENCE_TOOL_RESULT_PREFIX
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
        return {"selection_generation": "fixture-generation", **state}


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


async def _login_live_page(page, *, origin: str, access_token: str) -> None:
    """Complete the real remote-login path without exposing the credential."""

    await page.goto(origin)
    await page.get_by_label("Access token").fill(access_token)
    await page.get_by_role("button", name="Sign in").click()
    await expect(page.locator("#terminal-region")).to_be_visible()


async def _live_browser_session_id(server, page, *, port: int) -> str:
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
            scheme="https",
            host=f"127.0.0.1:{port}",
            cookie_value=cookie,
            forwarded_for="127.0.0.1"
            if server._web_auth.policy.trusted_proxy_addresses
            else None,
            forwarded_proto="https"
            if server._web_auth.policy.trusted_proxy_addresses
            else None,
            forwarded_host=f"127.0.0.1:{port}"
            if server._web_auth.policy.trusted_proxy_addresses
            else None,
        )
    )
    return session.session_id


async def _live_child_for_page(server, page, *, port: int) -> str:
    browser_session_id = await _live_browser_session_id(server, page, port=port)
    return server._served_browser_children[browser_session_id]


async def _send_terminal_command(page, command: str) -> None:
    await page.locator("#terminal .xterm-helper-textarea").press(
        {
            "update": "u",
            "branch": "b",
            "reopen-root": "r",
            "ping": "p",
            "next-adversarial": "n",
        }[command]
    )


async def _served_shell_projection(page) -> dict[str, object]:
    """Read the trusted shell's exact source-free state over its product route."""

    iframe = await page.locator("#served-canvas-frame").element_handle()
    assert iframe is not None
    shell_frame = await iframe.content_frame()
    assert shell_frame is not None
    result = await shell_frame.evaluate(
        """async () => {
            const response = await fetch(new URL('api/state', location.href));
            return {status: response.status, body: await response.json()};
        }"""
    )
    assert isinstance(result, dict)
    assert result.get("status") == 200
    body = result.get("body")
    assert isinstance(body, dict)
    return body


async def _send_console_prompt(page, prompt: str, focus_ack: Path) -> None:
    keyboard_target = page.locator("#terminal .xterm-helper-textarea")
    await keyboard_target.focus()
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        focus_ack.unlink(missing_ok=True)
        await keyboard_target.press("Escape")
        await keyboard_target.press("F11")
        while not focus_ack.exists() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        if focus_ack.exists() and focus_ack.read_text(encoding="ascii") == "focused":
            break
    else:
        raise AssertionError("actual Console composer focus was not acknowledged")
    await keyboard_target.press_sequentially(prompt, delay=5)
    await expect(page.locator("#terminal")).to_contain_text(prompt, timeout=5_000)
    await keyboard_target.press("Enter")


async def _wait_for_gateway_calls(path: Path, expected: int) -> None:
    for _ in range(600):
        if path.exists():
            observed = int(path.read_text(encoding="ascii"))
            if observed >= expected:
                return
        await asyncio.sleep(0.05)
    observed = path.read_text(encoding="ascii") if path.exists() else "missing"
    pytest.fail(
        "actual Chatbook provider call count did not reach "
        f"{expected}; observed={observed}"
    )


async def _scripted_gateway_reply(gateway, messages: list[dict[str, str]]) -> str:
    chunks = [chunk async for chunk in gateway.stream_chat(None, messages)]
    assert len(chunks) == 1
    return chunks[0]


@pytest.mark.parametrize("discovery", [False, True], ids=["direct", "progressive"])
async def test_actual_chatbook_scripted_gateway_emits_create_then_stable_update(
    tmp_path: Path, monkeypatch, discovery: bool
) -> None:
    """Keep the deterministic live provider faithful to real tool disclosure."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    gateway = _ScriptedCanvasGateway()
    system = (
        "use find_tools, then load_tools"
        if discovery
        else "available tools: canvas_create canvas_update"
    )
    messages = [{"role": "system", "content": system}]

    if discovery:
        assert '"name":"find_tools"' in await _scripted_gateway_reply(gateway, messages)
        assert '"name":"load_tools"' in await _scripted_gateway_reply(gateway, messages)
    create = await _scripted_gateway_reply(gateway, messages)
    assert '"name": "canvas_create"' in create
    canvas_id = "00000000-0000-4000-8000-000000000001"
    revision_id = "00000000-0000-4000-8000-000000000002"
    create_result = json.dumps(
        {
            "status": "staged",
            "canvas": {"canvas_id": canvas_id, "revision_id": revision_id},
        }
    )
    messages.append(
        {
            "role": "user",
            "content": f"{FENCE_TOOL_RESULT_PREFIX}canvas_create: {create_result}",
        }
    )
    assert await _scripted_gateway_reply(gateway, messages) == (
        "CHATBOOK_CANVAS_CREATED"
    )

    messages = [{"role": "system", "content": system}]
    if discovery:
        assert '"name":"find_tools"' in await _scripted_gateway_reply(gateway, messages)
        assert '"name":"load_tools"' in await _scripted_gateway_reply(gateway, messages)
    update = await _scripted_gateway_reply(gateway, messages)
    fenced = update.removeprefix("```tool_call\n").removesuffix("\n```")
    call = json.loads(fenced)
    assert call["name"] == "canvas_update"
    assert call["arguments"]["canvas_id"] == canvas_id
    assert call["arguments"]["expected_parent_revision_id"] == revision_id


@pytest.mark.loopback_network
async def test_actual_chatbook_console_finalizes_canvas_create_and_update(
    tmp_path: Path, monkeypatch
) -> None:
    """Run deterministic provider tool cycles through an actual TldwCli child."""

    access_token = secrets.token_urlsafe(24)
    stack = await start_live_served_stack(
        tmp_path,
        monkeypatch,
        access_token=access_token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )

    async def assert_persisted_complete(marker):
        database_path = tmp_path / "test_data" / "canvas-live-chatbook.sqlite"
        for _ in range(300):
            with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
                rows = database.execute(
                    "SELECT assistant_generation_state FROM messages "
                    "WHERE role = 'assistant' AND content = ? AND deleted = 0",
                    (marker,),
                ).fetchall()
            if rows == [("complete",)]:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("exact persisted assistant row is not complete")

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=live_chromium_executable(playwright.chromium),
            )
            page = await browser.new_page(ignore_https_errors=True)
            page.set_default_timeout(45_000)
            await _login_live_page(page, origin=stack.origin, access_token=access_token)
            await expect(page.locator("body")).to_have_class(re.compile("first-byte"))
            await expect(page.locator("#terminal")).to_contain_text(
                "Composer", timeout=45_000
            )
            await expect(page.locator("#terminal")).not_to_contain_text("Check online")
            child_id = await _live_child_for_page(stack.server, page, port=stack.port)
            await stack.server._canvas_control_broker.wait_connected(
                child_id, timeout=10.0
            )

            focus_ack = tmp_path / "test_data" / "canvas-live-composer-focused"
            await _send_console_prompt(page, "Create the requested Canvas", focus_ack)
            await _wait_for_gateway_calls(
                tmp_path / "test_data" / "canvas-live-gateway-calls", 2
            )
            disclosure = (
                tmp_path / "test_data" / "canvas-live-tool-disclosure"
            ).read_text(encoding="ascii")
            assert disclosure == (
                "mode=direct;find_tools=False;load_tools=False;canvas_create=True"
            )
            assert (tmp_path / "test_data" / "canvas-live-tool-status").read_text(
                encoding="utf-8"
            ) == "canvas"
            await expect(page.locator("#terminal")).to_contain_text(
                "CHATBOOK_CANVAS_CREATED", timeout=45_000
            )
            await assert_persisted_complete("CHATBOOK_CANVAS_CREATED")
            shell = page.frame_locator("#served-canvas-frame")
            await expect(shell.locator("#connection-state")).to_have_text(
                "Connected", timeout=45_000
            )
            preview = shell.frame_locator("#canvas-preview")
            await expect(preview.locator("#chatbook-app-canvas")).to_have_text(
                "CHATBOOK_APP_CANVAS", timeout=60_000
            )
            await expect(preview.locator("#chatbook-app-revision")).to_have_text("v1")
            await expect(shell.get_by_text("Revision 1", exact=True)).to_be_visible()

            await page.wait_for_timeout(1_000)
            await _send_console_prompt(page, "Revise the active Canvas", focus_ack)
            await _wait_for_gateway_calls(
                tmp_path / "test_data" / "canvas-live-gateway-calls", 3
            )
            await _wait_for_gateway_calls(
                tmp_path / "test_data" / "canvas-live-gateway-calls", 4
            )
            assert (tmp_path / "test_data" / "canvas-live-tool-status").read_text(
                encoding="ascii"
            ) == "canvas_create,staged"
            await expect(page.locator("#terminal")).to_contain_text(
                "CHATBOOK_CANVAS_UPDATED", timeout=45_000
            )
            await assert_persisted_complete("CHATBOOK_CANVAS_UPDATED")
            await expect(preview.locator("#chatbook-app-revision")).to_have_text(
                "v2", timeout=15_000
            )
            await expect(shell.get_by_text("Revision 2", exact=True)).to_be_visible()
            await page.locator("#terminal .xterm-helper-textarea").focus()
            await page.locator("#terminal .xterm-helper-textarea").press("F12")
            for _ in range(100):
                if (tmp_path / "test_data" / "canvas-live-card-pressed").exists():
                    break
                await asyncio.sleep(0.02)
            assert (tmp_path / "test_data" / "canvas-live-card-pressed").exists(), (
                "exact card handler completion was not acknowledged"
            )
            assert (tmp_path / "test_data" / "canvas-live-card-pressed").read_text(
                encoding="ascii"
            ) == "selected-pinned"
            await expect(preview.locator("#chatbook-app-revision")).to_have_text(
                "v1", timeout=15_000
            )
            await expect(shell.get_by_text("Revision 1", exact=True)).to_be_visible()
            await expect(shell.get_by_text("Pinned", exact=True)).to_be_visible()
            await expect(page.locator("#terminal.-connected")).to_be_visible()

            database_path = tmp_path / "test_data" / "canvas-live-chatbook.sqlite"

            def persisted_canvas_rows():
                with sqlite3.connect(
                    f"file:{database_path}?mode=ro", uri=True
                ) as database:
                    return database.execute(
                        "SELECT d.conversation_id, r.canvas_id, r.id, r.content_sha256, r.sequence "
                        "FROM canvas_documents d JOIN canvas_revisions r ON r.canvas_id=d.id "
                        "WHERE d.deleted_at IS NULL AND r.deleted_at IS NULL ORDER BY r.sequence"
                    ).fetchall()

            original_rows = persisted_canvas_rows()
            assert len(original_rows) == 2
            original_root = original_rows[0]
            original_control_generation = stack.server._canvas_control_broker._children[
                child_id
            ].generation
            browser_id = await _live_browser_session_id(
                stack.server, page, port=stack.port
            )
            old_url = await page.locator("#served-canvas-frame").get_attribute("src")
            await stack.server._canvas_control_broker.revoke_child(child_id)
            await page.goto("about:blank")
            for _ in range(300):
                if browser_id not in stack.server._served_browser_children:
                    break
                await asyncio.sleep(0.05)
            assert browser_id not in stack.server._served_browser_children
            await page.goto(stack.origin)
            await expect(page.locator("#terminal.-connected")).to_be_visible(
                timeout=15_000
            )
            replacement = await _live_child_for_page(
                stack.server, page, port=stack.port
            )
            assert replacement != child_id
            await stack.server._canvas_control_broker.wait_connected(
                replacement, timeout=15
            )
            assert (
                stack.server._canvas_control_broker._children[replacement].generation
                != original_control_generation
            )
            await expect(page.locator("#terminal")).to_contain_text(
                "Composer", timeout=45_000
            )
            await page.locator("#terminal .xterm-helper-textarea").focus()
            await page.locator("#terminal .xterm-helper-textarea").press("F10")
            loaded_ack = tmp_path / "test_data" / "canvas-live-saved-loaded"
            for _ in range(300):
                if loaded_ack.exists():
                    break
                await asyncio.sleep(0.05)
            assert loaded_ack.exists(), (
                "normal saved-conversation load was not acknowledged"
            )
            assert loaded_ack.read_text(encoding="ascii") == "loaded-without-provider"
            (tmp_path / "test_data" / "canvas-live-card-pressed").unlink()
            await page.locator("#terminal .xterm-helper-textarea").press("F12")
            for _ in range(300):
                if (tmp_path / "test_data" / "canvas-live-card-pressed").exists():
                    break
                await asyncio.sleep(0.05)
            assert (tmp_path / "test_data" / "canvas-live-card-pressed").read_text(
                encoding="ascii"
            ) == "selected-pinned"
            await expect(shell.locator("#connection-state")).to_have_text(
                "Connected", timeout=15_000
            )
            await expect(preview.locator("#chatbook-app-revision")).to_have_text(
                "v1", timeout=15_000
            )
            await expect(shell.get_by_text("Pinned", exact=True)).to_be_visible()
            restored = await _served_shell_projection(page)
            assert restored["selection"] == {
                "canvas_id": original_root[1],
                "revision_id": original_root[2],
            }
            assert restored["metadata"]["content_sha256"] == original_root[3]
            saved_scope = await stack.server._canvas_control_broker.request(
                replacement, "scope.snapshot.request", {}, timeout=2
            )
            assert saved_scope.payload["conversation_id"] == original_root[0]
            assert saved_scope.payload["selected_canvas_id"] == original_root[1]
            assert saved_scope.payload["selected_revision_id"] == original_root[2]
            assert (
                tmp_path / "test_data" / "canvas-live-restored-provider-calls"
            ).read_text(encoding="ascii") == "0"
            assert persisted_canvas_rows() == original_rows
            assert (await page.request.get(f"{stack.origin}{old_url}")).status == 404
            await browser.close()
    finally:
        await stack.aclose()
    assert all(not path.exists() for path in stack.owned_paths)
    assert not stack.runner.sites
    assert all(
        service._process is None or service._process.returncode is not None
        for service in stack.services
    )


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "trusted_proxy", [False, True], ids=["direct-tls", "trusted-proxy"]
)
async def test_production_tls_two_child_tool_and_reconnect_flow(
    tmp_path: Path, monkeypatch, trusted_proxy: bool
) -> None:
    """Exercise authenticated TLS, real AppServices, tools, isolation, and recovery."""

    access_token = secrets.token_urlsafe(24)
    stack = await start_live_served_stack(
        tmp_path, monkeypatch, access_token=access_token, trusted_proxy=trusted_proxy
    )
    attempted_urls: list[str] = []
    sent_websocket_frames: list[str | bytes] = []
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=live_chromium_executable(playwright.chromium),
            )
            contexts = [
                await browser.new_context(ignore_https_errors=True) for _ in range(2)
            ]
            for context in contexts:
                context.on(
                    "request", lambda request: attempted_urls.append(request.url)
                )
            pages = [await context.new_page() for context in contexts]
            startup_http = [[] for _ in pages]
            for profile_index, page in enumerate(pages):

                def record_startup_response(response, rows=startup_http[profile_index]):
                    category = urlsplit(response.url).path.rsplit("/", 1)[-1]
                    if category not in {
                        "session",
                        "boot",
                        "state",
                        "events",
                        "frame",
                        "plan",
                    }:
                        category = "other"
                    rows.append((category, response.request.method, response.status))
                    del rows[:-16]

                page.on("response", record_startup_response)
                page.on(
                    "websocket",
                    lambda websocket: websocket.on(
                        "framesent", lambda frame: sent_websocket_frames.append(frame)
                    ),
                )
                page.set_default_timeout(15_000)
                await _login_live_page(
                    page, origin=stack.origin, access_token=access_token
                )

            shells = [page.frame_locator("#served-canvas-frame") for page in pages]
            previews = [shell.frame_locator("#canvas-preview") for shell in shells]
            for profile_index, shell in enumerate(shells):
                try:
                    await expect(shell.locator("#connection-state")).to_have_text(
                        "Connected", timeout=15_000
                    )
                except AssertionError:
                    page = pages[profile_index]
                    browser_id = await _live_browser_session_id(
                        stack.server, page, port=stack.port
                    )
                    child_id = stack.server._served_browser_children.get(browser_id)
                    child_state = stack.server._canvas_control_broker._children.get(
                        child_id
                    )
                    service = next(
                        (
                            item
                            for item in stack.services
                            if item.app_service_id == child_id
                        ),
                        None,
                    )
                    process = service._process if service is not None else None
                    facts = {
                        "profile_index": profile_index,
                        "outer": await page.locator(
                            "#served-canvas-state"
                        ).text_content(),
                        "terminal_first_byte": "first-byte"
                        in (await page.locator("body").get_attribute("class") or ""),
                        "frame_kind": await page.locator(
                            "#served-canvas-frame"
                        ).evaluate(
                            "node => {const src=node.getAttribute('src');return !src?'absent':src==='about:blank'?'blank':new URL(src,location.href).pathname.startsWith('/canvas/')?'canvas':'other'}"
                        ),
                        "child_mapped": child_id is not None,
                        "service_matches_child": service is not None,
                        "child_alive": process is not None
                        and process.returncode is None,
                        "child_exit": process.returncode if process else None,
                        "registered": child_state is not None,
                        "control_connected": child_state is not None
                        and child_state.connected.is_set(),
                        "http": startup_http[profile_index],
                    }
                    # Deliberate bounded diagnostic request, not a passive poll.
                    if facts["control_connected"]:
                        try:
                            snapshot = (
                                await stack.server._canvas_control_broker.request(
                                    child_id, "scope.snapshot.request", {}, timeout=2
                                )
                            )
                            facts["diagnostic_child_selected"] = bool(
                                snapshot.payload.get("selected_revision_id")
                            )
                        except (ControlProtocolError, TimeoutError):
                            facts["diagnostic_snapshot_failed"] = True
                    pytest.fail(
                        f"initial served readiness failed: {facts}", pytrace=False
                    )
                await expect(shell.locator("#loading-state")).to_be_hidden()

            port = int(urlsplit(stack.origin).port or 443)
            child_ids = [
                await _live_child_for_page(stack.server, page, port=port)
                for page in pages
            ]
            assert child_ids[0] != child_ids[1]
            markers = [f"child-{child_id[-8:]}" for child_id in child_ids]
            for preview, marker in zip(previews, markers, strict=True):
                await expect(preview.locator("#profile-identity")).to_have_text(marker)
                await expect(preview.locator("#revision-marker")).to_have_text("v1")

            first_src = (
                await pages[0].locator("#served-canvas-frame").get_attribute("src")
            )
            assert first_src is not None
            copied_status = await pages[1].evaluate(
                "async url => (await fetch(url)).status", first_src
            )
            guessed_status = await pages[1].evaluate(
                "async () => (await fetch('/canvas/guessed')).status"
            )
            assert copied_status == guessed_status == 404

            async with pages[0].expect_response(
                lambda response: response.url.endswith("/api/bridge/prepare")
            ) as prepare_info:
                await previews[0].get_by_role("button", name="Send result").click()
            assert (await prepare_info.value).status == 200
            await expect(shells[0].locator("#bridge-dialog")).to_be_visible()
            assert (
                markers[0]
                in await shells[0].locator("#bridge-complete-text").input_value()
            )
            await shells[0].get_by_role("button", name="Send to composer").click()
            await expect(shells[0].locator("#bridge-dialog")).to_be_hidden()

            await previews[1].get_by_role("button", name="Download result").click()
            await expect(shells[1].locator("#bridge-dialog")).to_be_visible()
            async with pages[1].expect_download() as download_info:
                await shells[1].get_by_role("button", name="Download file").click()
            download = await download_info.value
            assert download.suggested_filename == f"{markers[1]}.txt"
            download_path = tmp_path / "downloaded-profile.txt"
            await download.save_as(download_path)
            assert download_path.read_text(encoding="utf-8") == f"{markers[1]}:v1"

            root_selection = (await _served_shell_projection(pages[0]))["selection"]
            await _send_terminal_command(pages[0], "update")
            await pages[0].wait_for_timeout(100)
            assert any("stdin" in str(frame) for frame in sent_websocket_frames)
            await expect(previews[0].locator("#revision-marker")).to_have_text(
                "v2", timeout=15_000
            )
            await expect(
                shells[0].get_by_text("Revision 2", exact=True)
            ).to_be_visible()

            await _send_terminal_command(pages[0], "branch")
            await expect(previews[0].locator("#revision-marker")).to_have_text(
                "branch", timeout=15_000
            )
            await _send_terminal_command(pages[0], "reopen-root")
            await expect(pages[0].locator("#terminal")).to_contain_text(
                "CANVAS_LIVE_REOPENED"
            )
            snapshot = await stack.server._canvas_control_broker.request(
                child_ids[0], "scope.snapshot.request", {}, timeout=1.0
            )
            assert (
                snapshot.payload["selected_revision_id"]
                == root_selection["revision_id"]
            ), "child did not select exact root"
            await expect(previews[0].locator("#revision-marker")).to_have_text(
                "v1", timeout=15_000
            )
            await expect(shells[0].get_by_text("Pinned", exact=True)).to_be_visible()
            reopened = (await _served_shell_projection(pages[0]))["selection"]
            assert reopened["revision_id"] == root_selection["revision_id"]

            await stack.server._canvas_control_broker.revoke_child(child_ids[0])
            await expect(
                pages[0].get_by_text("Canvas reconnecting", exact=True)
            ).to_be_visible(timeout=15_000)
            await expect(pages[0].locator("#terminal-region")).to_be_visible()
            await expect(shells[1].locator("#connection-state")).to_have_text(
                "Connected"
            )

            await _send_terminal_command(pages[0], "ping")
            await pages[0].wait_for_timeout(250)
            await expect(pages[0].locator("#terminal.-connected")).to_be_visible()
            assert any(
                '"stdin","p"' in str(frame).replace(" ", "")
                for frame in sent_websocket_frames
            )

            # Close the old terminal transport before reopening the authenticated
            # page. Temporary history is destroyed, never revived with old auth.
            browser_session_id = await _live_browser_session_id(
                stack.server, pages[0], port=stack.port
            )
            await pages[0].goto("about:blank")
            for _ in range(300):
                if browser_session_id not in stack.server._served_browser_children:
                    break
                await asyncio.sleep(0.05)
            assert browser_session_id not in stack.server._served_browser_children
            await pages[0].goto(stack.origin)
            await expect(pages[0].locator("#terminal.-connected")).to_be_visible(
                timeout=15_000
            )
            for _ in range(300):
                replacement_id = stack.server._served_browser_children.get(
                    browser_session_id
                )
                if replacement_id is not None and replacement_id != child_ids[0]:
                    break
                await asyncio.sleep(0.05)
            assert replacement_id != child_ids[0]
            await stack.server._canvas_control_broker.wait_connected(
                replacement_id, timeout=15.0
            )
            await expect(shells[0].locator("#connection-state")).to_have_text(
                "Connected", timeout=15_000
            )
            await expect(previews[0].locator("#revision-marker")).to_have_text(
                "v1", timeout=15_000
            )
            replacement = (await _served_shell_projection(pages[0]))["selection"]
            assert replacement["revision_id"] != root_selection["revision_id"]
            assert (
                await pages[0].evaluate(
                    "async url => (await fetch(url)).status", first_src
                )
                == 404
            )
            await expect(shells[1].locator("#connection-state")).to_have_text(
                "Connected"
            )
            await expect(previews[1].locator("#profile-identity")).to_have_text(
                markers[1]
            )

            assert_only_owned_browser_traffic(
                attempted_urls, owned_origins=(stack.origin,)
            )
            if trusted_proxy:
                assert stack.server._web_ssl_context is None
                assert stack.proxy_counts["http"] > 0
                assert stack.proxy_counts["websocket"] == 3
            await browser.close()
    finally:
        await stack.aclose()


async def test_product_runtime_failure_must_match_canonical_code():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=live_chromium_executable(playwright.chromium),
        )
        context = await browser.new_context()
        await ProductRouteRecorder().install_execution_boundary(context)
        page = await context.new_page()
        await page.goto("about:blank")
        await page.set_content(
            '<div id="loading-state">loading</div><div id="compatibility">failed</div>'
            '<h2 id="compatibility-title">Preview issue</h2>'
        )
        await page.evaluate("""() => {
            const channel = new MessageChannel();
            channel.port1.postMessage({type:'canvas:status', state:'failed', code:'runtime-error'});
            channel.port1.close(); channel.port2.close();
        }""")
        with pytest.raises(AssertionError, match="runtime failure code"):
            await exercise_adversarial_preview(
                page,
                page,
                {"expected": "failed", "expected_code": "runtime-timeout"},
            )
        await browser.close()


async def test_live_stack_cleanup_failure_still_reaps_child_and_owned_files(tmp_path):
    class Process:
        returncode = None

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def fail_cleanup():
        raise RuntimeError("owned cleanup failure")

    process = Process()
    owned = tmp_path / "owned-child-data"
    owned.mkdir()
    stack = LiveServedStack(
        server=None,
        runner=SimpleNamespace(cleanup=fail_cleanup),
        origin="https://127.0.0.1",
        port=1,
        owned_paths=(owned,),
        services=[SimpleNamespace(_process=process)],
    )
    with pytest.raises(RuntimeError, match="owned cleanup failure"):
        await stack.aclose()
    assert process.returncode == -9
    assert not owned.exists()


@pytest.mark.parametrize("fail_site", [1, 2], ids=["backend", "proxy"])
async def test_live_stack_setup_failure_rolls_back_owned_resources(
    tmp_path, monkeypatch, fail_site
):
    import Tests.Canvas.browser.canvas_live_harness as harness

    sites = []
    sessions = []
    original_start = web.TCPSite.start
    original_session = harness.ClientSession

    async def failing_start(site):
        await original_start(site)
        sites.append(site)
        if len(sites) == fail_site:
            raise RuntimeError("owned setup failure")

    def record_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(web.TCPSite, "start", failing_start)
    monkeypatch.setattr(harness, "ClientSession", record_session)
    try:
        with pytest.raises(RuntimeError, match="owned setup failure"):
            await start_live_served_stack(
                tmp_path,
                monkeypatch,
                access_token="owned-test-token",
                trusted_proxy=True,
            )
        assert all(not site._server.is_serving() for site in sites)
        assert all(session.closed for session in sessions)
        assert not (tmp_path / "test_data").exists()
        assert not list(tmp_path.glob("*.pem"))
    finally:
        # Keep the deliberate RED probe's already-started resources owned too.
        for site in sites:
            await site._runner.cleanup()
        for session in sessions:
            await session.close()


@pytest.mark.loopback_network
async def test_actual_child_abnormal_exit_cleans_owned_state(
    tmp_path, monkeypatch
) -> None:
    stack = await start_live_served_stack(
        tmp_path,
        monkeypatch,
        access_token="owned-test-token",
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=live_chromium_executable(playwright.chromium),
            )
            try:
                page = await browser.new_page(ignore_https_errors=True)
                await _login_live_page(
                    page, origin=stack.origin, access_token="owned-test-token"
                )
                await expect(page.locator("#terminal")).to_contain_text(
                    "Composer", timeout=45_000
                )
                child_temp = tmp_path / "test_data" / "child-tmp"
                assert list(child_temp.glob("tldw-chatbook-test-*"))
                assert len(stack.services) == 1
                process = stack.services[0]._process
                assert process is not None and process.returncode is None
                process.kill()
                await asyncio.wait_for(process.wait(), 5)
            finally:
                await browser.close()
    finally:
        await stack.aclose()
    assert all(not path.exists() for path in stack.owned_paths)
    assert all(service._process.returncode is not None for service in stack.services)


@pytest.mark.parametrize(
    ("mode", "start_index", "update_count"),
    (("fresh", 7, 1), ("rapid", 0, 8)),
)
async def test_served_renderer_case_seven_keeps_exact_scope_and_execution_ack(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    start_index: int,
    update_count: int,
) -> None:
    """Correlate the formerly flaky renderer load on fresh and rapid paths."""

    access_token = secrets.token_urlsafe(24)
    with egress_probe() as probe:
        monkeypatch.setenv("TLDW_CANVAS_EGRESS_PROBE_ORIGIN", probe.origin)
        monkeypatch.setenv("TLDW_CANVAS_ADVERSARIAL_START_INDEX", str(start_index))
        stack = await start_live_served_stack(
            tmp_path, monkeypatch, access_token=access_token
        )
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True,
                    executable_path=live_chromium_executable(playwright.chromium),
                )
                context = await browser.new_context(ignore_https_errors=True)
                recorder = ProductRouteRecorder(served=True)
                await recorder.install_execution_boundary(context)
                page = await context.new_page()
                recorder.attach(context, page)
                page.set_default_timeout(15_000)
                await _login_live_page(
                    page, origin=stack.origin, access_token=access_token
                )
                await expect(page.locator("#terminal.-connected")).to_be_visible()
                browser_session_id = await _live_browser_session_id(
                    stack.server, page, port=stack.port
                )
                child_id = stack.server._served_browser_children[browser_session_id]
                await stack.server._canvas_control_broker.wait_connected(
                    child_id, timeout=15.0
                )
                shell = page.frame_locator("#served-canvas-frame")
                preview = shell.frame_locator("#canvas-preview")
                await expect(shell.locator("#connection-state")).to_have_text(
                    "Connected"
                )
                await expect(shell.locator("#loading-state")).to_be_hidden()
                initial_expected = await stack.server.served_canvas_state(
                    browser_session_id
                )
                initial_actual = await _served_shell_projection(page)
                if (
                    initial_expected["revision_id"]
                    != initial_actual["selection"]["revision_id"]
                ):
                    raise AssertionError(
                        "initial shell revision differs from child scope"
                    )

                ack_count = recorder.execution_ack_count
                cases = adversarial_cases(probe.origin)
                for offset in range(update_count):
                    case_index = start_index + offset
                    recorder.begin_load()
                    await _send_terminal_command(page, "next-adversarial")
                    await expect(page.locator("#terminal")).to_contain_text(
                        f"CANVAS_LIVE_ADVERSARIAL_{case_index}"
                    )
                    expected = await stack.server.served_canvas_state(
                        browser_session_id
                    )
                    try:
                        await expect(
                            shell.get_by_text(f"Revision {offset + 2}", exact=True)
                        ).to_be_visible(timeout=5_000)
                        await expect(
                            preview.locator("#adversarial-marker")
                        ).to_have_text(str(case_index))
                        await exercise_adversarial_preview(
                            shell, preview, cases[case_index], expect_bridge=False
                        )
                        for _ in range(1_000):
                            if recorder.execution_ack_count > ack_count:
                                break
                            await asyncio.sleep(0.01)
                    except AssertionError:
                        actual = await _served_shell_projection(page)
                        state = {
                            "case_index": case_index,
                            "startup_error": recorder.startup_error,
                            "revision_matches_child": expected["revision_id"]
                            == actual["selection"]["revision_id"],
                            "connection": await shell.locator(
                                "#connection-state"
                            ).text_content(),
                            "loading_hidden": await shell.locator(
                                "#loading-state"
                            ).is_hidden(),
                            "compatibility_hidden": await shell.locator(
                                "#compatibility"
                            ).is_hidden(),
                            "marker_visible": await preview.locator(
                                "#adversarial-marker"
                            ).is_visible(),
                            "execution_ack_advanced": (
                                recorder.execution_ack_count > ack_count
                            ),
                        }
                        pytest.fail(
                            f"case seven {mode} renderer boundary failed: {state}",
                            pytrace=False,
                        )
                    if recorder.execution_ack_count <= ack_count:
                        raise AssertionError("renderer execution ack did not advance")
                    ack_count = recorder.execution_ack_count
                    actual = await _served_shell_projection(page)
                    if expected["revision_id"] != actual["selection"]["revision_id"]:
                        raise AssertionError(
                            "trusted shell revision differs from exact child scope"
                        )
                    assert probe.requests == []
                await browser.close()
        finally:
            await stack.aclose()


@pytest.mark.loopback_network
async def test_canonical_adversarial_corpus_stays_in_served_product_route(
    tmp_path: Path, monkeypatch
) -> None:
    """Replay the canonical attacks through authenticated mounted Canvas routes."""

    access_token = secrets.token_urlsafe(24)
    canary_path = "/canvas-generated-same-origin-canary"
    with egress_probe() as probe:
        monkeypatch.setenv("TLDW_CANVAS_EGRESS_PROBE_ORIGIN", probe.origin)
        monkeypatch.setenv("TLDW_CANVAS_SAME_ORIGIN_CANARY_PATH", canary_path)
        stack = await start_live_served_stack(
            tmp_path, monkeypatch, access_token=access_token
        )
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True,
                    executable_path=live_chromium_executable(playwright.chromium),
                )
                context = await browser.new_context(ignore_https_errors=True)
                recorder = ProductRouteRecorder(served=True)
                await recorder.install_execution_boundary(context)
                await context.add_init_script(
                    "Object.defineProperty(window, '__canvasNativeWindowSentinel', {"
                    "value:'served-route-clean',writable:true,configurable:true});"
                    "Object.defineProperty(Object.prototype, "
                    "'__canvasNativePrototypeSentinel', {value:'served-route-clean',"
                    "writable:true,configurable:true});"
                )
                page = await context.new_page()
                recorder.attach(context, page)
                bridge_prepare_responses: list[object] = []
                page.on(
                    "response",
                    lambda response: (
                        bridge_prepare_responses.append(response)
                        if response.url.endswith("/api/bridge/prepare")
                        else None
                    ),
                )
                page.set_default_timeout(15_000)
                await _login_live_page(
                    page, origin=stack.origin, access_token=access_token
                )
                await expect(page.locator("#terminal.-connected")).to_be_visible(
                    timeout=15_000
                )
                child_id = await _live_child_for_page(
                    stack.server, page, port=stack.port
                )
                await stack.server._canvas_control_broker.wait_connected(
                    child_id, timeout=15.0
                )
                shell = page.frame_locator("#served-canvas-frame")
                preview = shell.frame_locator("#canvas-preview")
                await expect(shell.locator("#connection-state")).to_have_text(
                    "Connected", timeout=15_000
                )
                await expect(shell.locator("#loading-state")).to_be_hidden(
                    timeout=15_000
                )

                cases = adversarial_cases(probe.origin)
                cases.append(
                    {
                        "name": "same_origin_relative_request",
                        "script": (
                            f"try {{ fetch('{canary_path}'); }} catch (_error) {{}}"
                        ),
                        "expected": "ready",
                    }
                )

                async def clear_prepared_dialog() -> None:
                    dialog = shell.locator("#bridge-dialog")
                    if not await dialog.is_visible():
                        return
                    async with page.expect_response(
                        lambda response: response.url.endswith("/api/bridge")
                    ) as decision_info:
                        await shell.get_by_role("button", name="Cancel").click()
                    assert (await decision_info.value).status in {200, 409}
                    await dialog.wait_for(state="hidden")

                accepted_sequence = 1
                for case_index, case in enumerate(cases):
                    prior_prepares = len(bridge_prepare_responses)
                    recorder.begin_load()
                    await _send_terminal_command(page, "next-adversarial")
                    await expect(page.locator("#terminal")).to_contain_text(
                        re.compile(
                            rf"CANVAS_LIVE_(?:ADVERSARIAL|REJECTED)_{case_index}"
                        ),
                        timeout=15_000,
                    )
                    if f"CANVAS_LIVE_REJECTED_{case_index}" in (
                        await page.locator("#terminal").inner_text()
                    ):
                        pytest.fail(
                            f"unexpected admission refusal: {case['name']}",
                            pytrace=False,
                        )
                    accepted_sequence += 1
                    try:
                        await expect(
                            shell.get_by_text(
                                f"Revision {accepted_sequence}", exact=True
                            )
                        ).to_be_visible(timeout=15_000)
                        await expect(
                            preview.locator("#adversarial-marker")
                        ).to_have_text(str(case_index), timeout=15_000)
                    except AssertionError:
                        renderer_state = {
                            "startup_error": recorder.startup_error,
                            "http_failures": [
                                (
                                    urlsplit(row.target).path.rsplit("/", 1)[-1],
                                    json.loads(row.detail).get("status"),
                                )
                                for row in recorder.observations
                                if row.kind == "response"
                                and json.loads(row.detail).get("status", 0) >= 400
                            ][-8:],
                            "connection": await shell.locator(
                                "#connection-state"
                            ).text_content(),
                            "loading_hidden": await shell.locator(
                                "#loading-state"
                            ).is_hidden(),
                            "compatibility_hidden": await shell.locator(
                                "#compatibility"
                            ).is_hidden(),
                            "preview_src": bool(
                                await shell.locator("#canvas-preview").get_attribute(
                                    "src"
                                )
                            ),
                        }
                        pytest.fail(
                            f"{case['name']} renderer did not mount; "
                            f"state={renderer_state}",
                            pytrace=False,
                        )
                    emits_expected_bridge = isinstance(
                        case.get("bridge_kind"), str
                    ) or case["name"] in {"bridge_count_limit", "bridge_rate_limit"}
                    if not emits_expected_bridge:
                        await clear_prepared_dialog()
                    try:
                        await exercise_adversarial_preview(
                            shell,
                            preview,
                            case,
                            expect_bridge=False,
                            programmatic_clicks=True,
                        )
                    except PlaywrightTimeoutError:
                        shell_state = await shell.locator(".canvas-workbench").evaluate(
                            "node => ({children:[...node.children].map(child => ({"
                            "id:child.id,hidden:child.hidden,inert:child.inert}))})"
                        )
                        pytest.fail(
                            f"{case['name']} timed out; bridge prepare statuses="
                            f"{[response.status for response in bridge_prepare_responses[-2:]]}; "
                            f"shell={shell_state}",
                            pytrace=False,
                        )
                    dialog = shell.locator("#bridge-dialog")
                    if isinstance(case.get("bridge_kind"), str):
                        for _ in range(1_000):
                            if len(bridge_prepare_responses) > prior_prepares:
                                break
                            await asyncio.sleep(0.01)
                        response = bridge_prepare_responses[-1]
                        if response.status != 200:
                            payload = await response.json()
                            pytest.fail(
                                f"{case['name']} prepare failed with "
                                f"{response.status}:{payload.get('error')}",
                                pytrace=False,
                            )
                        await dialog.wait_for(state="visible", timeout=10_000)
                    elif case["name"] in {"bridge_count_limit", "bridge_rate_limit"}:
                        for _ in range(1_000):
                            if len(bridge_prepare_responses) > prior_prepares:
                                break
                            await asyncio.sleep(0.01)
                        response = bridge_prepare_responses[-1]
                        assert response.status in {200, 409}
                        if response.status == 200:
                            await asyncio.sleep(0.1)
                    await clear_prepared_dialog()
                    assert probe.requests == [], case["name"]
                    assert (
                        await page.evaluate("window.__canvasNativeWindowSentinel")
                        == "served-route-clean"
                    )
                    assert (
                        await page.evaluate(
                            "Object.prototype.__canvasNativePrototypeSentinel"
                        )
                        == "served-route-clean"
                    )
                    assert (
                        await preview.locator("#attack").evaluate(
                            "() => window.__canvasNativeWindowSentinel"
                        )
                        == "served-route-clean"
                    )

                bridge_prepare_statuses = [
                    response.status for response in bridge_prepare_responses
                ]
                assert bridge_prepare_statuses.count(200) >= 3
                assert set(bridge_prepare_statuses) <= {200, 409}

                shell_src = await page.locator("#served-canvas-frame").get_attribute(
                    "src"
                )
                assert shell_src is not None
                route_path = urlsplit(shell_src).path
                recorder.assert_generated_confined(
                    trusted_origin=stack.origin,
                    trusted_route_root=route_path,
                    forbidden_canary_path=canary_path,
                    trusted_shell_paths=(("/canvas/api/session", "GET", "fetch"),),
                    trusted_static_paths=(
                        "/static/chatbook-canvas/canvas_renderer.js",
                        "/static/chatbook-canvas/canvas_runtime_worker.js",
                        "/static/chatbook-canvas/quickjs-runtime.js",
                    ),
                )
                await browser.close()
        finally:
            await stack.aclose()


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


async def test_parent_scope_poll_marks_delayed_selection_as_passive_sync(
    tmp_path: Path,
    monkeypatch,
) -> None:
    server = _server(tmp_path)
    await _browser_app(server)
    server.bind_served_browser("browser-a", "child-a")
    state = {"status": "ready", "canvas_id": "canvas-a", "revision_id": "revision-a"}
    server._canvas_control_broker.states["child-a"] = state
    try:
        initial = await server.served_canvas_state("browser-a")
        calls = []
        monkeypatch.setattr(
            server._served_canvas_gateway,
            "change_selection",
            lambda **kwargs: calls.append(kwargs),
        )
        state["revision_id"] = "revision-b"
        updated = await server.served_canvas_state("browser-a")
        assert updated["url"] == initial["url"]
        assert len(calls) == 1
        assert calls[0]["synchronize_only"] is True
        assert calls[0]["scope"].revision_id == "revision-b"
        await server.served_canvas_state("browser-a")
        assert len(calls) == 1
    finally:
        await server._served_canvas_gateway.aclose()


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


@pytest.mark.parametrize("pinned_revision", ["revision-v1", "revision-v2"])
async def test_queued_follow_cannot_overwrite_later_exact_pin(pinned_revision):
    class SelectionAuthority(_MountedAuthority):
        following = True
        mutations = 0

        def navigate(self, scope, **_kwargs):
            self.mutations += 1
            self.following = True
            return SimpleNamespace(
                scope=scope, projection=SimpleNamespace(following=True)
            )

    authority = SelectionAuthority()
    handler = ServedCanvasControlHandler()
    original = CanvasGatewayScope(
        browser_session_id="child-a",
        conversation_session_id="conversation-a",
        canvas_id="canvas-a",
        revision_id="revision-v2",
    )
    handler.bind(authority, original)
    issued = handler.scope
    queued = ControlMessage(
        CONTROL_PROTOCOL_VERSION,
        "selection.request",
        "queued-follow",
        9999999999999,
        {
            "action": "follow",
            "expected_session_id": issued.conversation_session_id,
            "expected_canvas_id": issued.canvas_id,
            "expected_revision_id": issued.revision_id,
            "expected_selection_generation": issued.selection_generation,
        },
    )
    # The command is already issued. A real exact-open binding is applied before
    # its delivery, including the same-revision change from following to pinned.
    authority.following = False
    handler.bind(
        authority,
        CanvasGatewayScope(
            browser_session_id="child-a",
            conversation_session_id="conversation-a",
            canvas_id="canvas-a",
            revision_id=pinned_revision,
        ),
    )
    with pytest.raises(ControlProtocolError, match="selection_refused"):
        await handler.handle(queued)
    assert handler.scope.revision_id == pinned_revision
    assert authority.following is False
    assert authority.mutations == 0
    pinned_generation = handler.scope.selection_generation
    for index in range(2):
        snapshot = await handler.handle(
            ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "scope.snapshot.request",
                f"stable-snapshot-{index}",
                None,
                {},
            )
        )
        assert snapshot.payload["selection_generation"] == pinned_generation
    current = handler.scope
    response = await handler.handle(
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "selection.request",
            "current-follow",
            None,
            {
                "action": "follow",
                "expected_session_id": current.conversation_session_id,
                "expected_canvas_id": current.canvas_id,
                "expected_revision_id": current.revision_id,
                "expected_selection_generation": current.selection_generation,
            },
        )
    )
    assert response.payload["selection_generation"] != pinned_generation
    assert authority.mutations == 1


@pytest.mark.parametrize(
    "code,freshness",
    [
        ("selection_refused", True),
        ("navigation_refused", False),
        ("scope_unavailable", False),
        ("channel_closed", False),
    ],
)
async def test_proxy_preserves_only_exact_navigation_freshness_refusal(code, freshness):
    from tldw_chatbook.Canvas.gateway import CanvasSelectionChanged

    async def refuse(*_args, **_kwargs):
        raise ControlProtocolError(code)

    owner = SimpleNamespace(
        _served_browser_children={"browser-a": "child-a"},
        _canvas_control_broker=SimpleNamespace(request=refuse),
    )
    proxy = serve._ServedCanvasAuthorityProxy(owner)
    scope = CanvasGatewayScope(
        "browser-a", "session-a", "canvas-a", "revision-a", "intent-a"
    )
    with pytest.raises(
        CanvasSelectionChanged if freshness else serve.ServedCanvasUnavailable
    ):
        await proxy.navigate(scope, action="follow")


async def test_authority_navigation_failure_is_not_selection_freshness():
    class FailingAuthority(_MountedAuthority):
        def navigate(self, *_args, **_kwargs):
            raise ValueError("unavailable")

    handler = ServedCanvasControlHandler()
    handler.bind(
        FailingAuthority(),
        CanvasGatewayScope("child-a", "session-a", "canvas-a", "revision-a"),
    )
    current = handler.scope
    request = ControlMessage(
        CONTROL_PROTOCOL_VERSION,
        "selection.request",
        "failed-navigation",
        None,
        {
            "action": "follow",
            "expected_session_id": current.conversation_session_id,
            "expected_canvas_id": current.canvas_id,
            "expected_revision_id": current.revision_id,
            "expected_selection_generation": current.selection_generation,
        },
    )
    with pytest.raises(ControlProtocolError, match="^navigation_refused$"):
        await handler.handle(request)
    assert handler.scope == current


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
