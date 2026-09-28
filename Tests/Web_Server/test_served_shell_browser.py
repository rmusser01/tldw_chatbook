"""Browser-level integration tests for the served shell and patched bundle.

task-33130 (Qodo review): the viewport tests in
``test_textual_web_viewport.py`` assert on the *text* of the patched
textual-serve bundle. These tests instead execute the real served page in
headless Chromium against a scripted websocket, so the patched resize
behavior and the served-shell fixes are exercised end to end:

- the terminal websocket connects to the page's own origin (same-origin
  URL derivation), not to a server-baked absolute host;
- an idle session sends no resize messages (the pre-fix resize feedback
  loop measured ~54/second at idle);
- a real viewport change produces a bounded resize burst that stops;
- a 404 from ``/canvas/api/session`` stops the 1 Hz session poll.

Requires the ``web`` extra plus playwright with a chromium build; skips
otherwise (CI installs neither today).
"""

from __future__ import annotations

import asyncio
import os
import pwd
from pathlib import Path

import pytest

from tldw_chatbook.Web_Server import serve


def _real_playwright_browsers_path() -> str | None:
    """Locate the real user's playwright browser cache.

    The pytest sandbox redirects HOME per test, which hides the browsers
    installed for the actual account; playwright resolves its cache from
    HOME, so point it at the real one explicitly when it exists.
    """
    try:
        user_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):  # pragma: no cover - unusual environments
        return None
    for candidate in (
        user_home / "Library" / "Caches" / "ms-playwright",
        user_home / ".cache" / "ms-playwright",
    ):
        if candidate.is_dir():
            return str(candidate)
    return None


pytestmark = pytest.mark.skipif(
    not serve.check_web_server_available(),
    reason="web server optional dependencies are unavailable",
)


class _FakeTerminalServeBase:
    """Minimal textual-serve Server stand-in for the mixin's needs."""

    def __init__(self, command, host, port, title, **kwargs):
        self.command = command
        self.host = host
        self.port = port
        self.title = title
        self.public_url = kwargs.get("public_url") or f"http://{host}:{port}"
        self.statics_path = kwargs.get("statics_path", "/tmp/static")
        self.templates_path = kwargs.get("templates_path", "/tmp/templates")

    async def on_startup(self, app):
        return None

    async def on_shutdown(self, app):
        return None

    async def handle_download(self, request):
        from aiohttp import web

        raise web.HTTPNotFound(text="no downloads")


class _ScriptedShellServer(
    serve.build_chatbook_web_server_class(_FakeTerminalServeBase)
):
    """Served shell with a scripted terminal websocket and Canvas latched off.

    Instances are created with ``__new__`` (attributes assigned directly) so
    construction stays independent of the config-bootstrap admission fence
    that gates full server construction in some environments.
    """

    async def on_startup(self, app) -> None:
        # No control broker, no policy watcher: Canvas is latched off.
        return None

    async def handle_websocket(self, request):
        import json as _json

        from aiohttp import WSMsgType, web

        websocket = web.WebSocketResponse(
            heartbeat=15, protocols=(serve.WEBSOCKET_PROTOCOL,)
        )
        await websocket.prepare(request)
        self.ws_opens += 1
        # >10 bytes so the patched first-byte/after-write hooks fire.
        await websocket.send_bytes(b"\x1b[2J\x1b[H" + b"#" * 40)
        async for message in websocket:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                envelope = _json.loads(message.data)
            except ValueError:
                continue
            kind = envelope[0] if isinstance(envelope, list) else None
            if kind == "resize":
                self.resizes.append(envelope[1])
            elif kind == "ping":
                await websocket.send_json(["pong", envelope[1]])
        return websocket

    async def handle_served_canvas_session(self, request):
        from aiohttp import web

        self.session_requests.append(request)
        raise web.HTTPNotFound(text="Canvas unavailable")


def _make_scripted_server(statics_path: str, port: int) -> _ScriptedShellServer:
    from tldw_chatbook.Canvas.web_auth import WebAuthManager, build_web_auth_policy

    server = _ScriptedShellServer.__new__(_ScriptedShellServer)
    server.command = "python -m tldw_chatbook.app"
    server.host = "127.0.0.1"
    server.port = port
    server.title = "scripted"
    server.public_url = "http://127.0.0.1"
    server.statics_path = statics_path
    server.templates_path = "/tmp/templates"
    server.debug = False
    # The policy validates the Host authority (name, port) strictly, so it
    # must be built with the port the test server will actually bind.
    server._web_auth = WebAuthManager(
        build_web_auth_policy(host="127.0.0.1", port=port, access_token=None)
    )
    server._web_ssl_context = None
    server._canvas_disabled_latched = True
    server._canvas_policy_watch_task = None
    server._served_browser_children = {}
    server._served_canvas_launches = {}

    class _GatewayStub:
        async def aclose(self) -> None:
            return None

        def mark_browser_session_unavailable(self, browser_session_id) -> None:
            return None

    server._served_canvas_gateway = _GatewayStub()
    server.ws_opens = 0
    server.resizes = []
    server.session_requests = []
    return server


class _ScriptedContext:
    """Owns the scripted server and a headless-Chromium page for one test."""

    async def start(self):
        playwright_api = pytest.importorskip("playwright.async_api")
        import textual_serve

        # handle_index reads the font-size setting from the shared config
        # cache; under pytest the ADR-126 config admission fence makes that
        # read raise, so serve the default exactly like the font-size tests
        # in test_textual_web_viewport.py do.
        self._original_get_cli_setting = serve.get_cli_setting
        serve.get_cli_setting = lambda *_, default=None: default

        statics_path = str(Path(textual_serve.__file__).parent / "static")
        # Reserve the port before building the app so the auth policy's
        # allowed authorities include the port the browser will actually use.
        import socket

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        self.server = _make_scripted_server(statics_path, port)
        app = await self.server._make_app()

        from aiohttp.test_utils import TestServer

        self.http = TestServer(app, port=port)
        await self.http.start_server()
        self.origin = f"http://127.0.0.1:{self.http.port}"
        browsers_path = _real_playwright_browsers_path()
        if browsers_path is not None:
            os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", browsers_path)
        self.playwright = await playwright_api.async_playwright().start()
        try:
            self.browser = await self.playwright.chromium.launch(headless=True)
        except Exception as error:  # noqa: BLE001 - environment-dependent
            await self.playwright.stop()
            await self.http.close()
            pytest.skip(f"playwright chromium unavailable: {type(error).__name__}: {error}")
        self.page = await self.browser.new_page(
            viewport={"width": 1000, "height": 700}
        )
        await self.page.goto(self.origin + "/", wait_until="domcontentloaded")
        await self.page.wait_for_function(
            "document.body.classList.contains('-first-byte')", timeout=20000
        )

    async def stop(self):
        serve.get_cli_setting = self._original_get_cli_setting
        await self.browser.close()
        await self.playwright.stop()
        await self.http.close()


@pytest.mark.timeout(120)
async def test_served_shell_executes_bounded_resize_behavior_in_browser():
    context = _ScriptedContext()
    await context.start()
    try:
        server = context.server
        page = context.page

        # Same-origin websocket: the attribute derived from the page's own
        # location, and the scripted endpoint actually accepted a client.
        ws_url = await page.evaluate(
            "document.getElementById('terminal').dataset.sessionWebsocketUrl"
        )
        assert ws_url == f"ws://127.0.0.1:{context.http.port}/ws"
        assert server.ws_opens == 1

        # Idle: the pre-fix feedback loop sent ~54 resizes/second; a healthy
        # session sends none. Allow the initial upstream size sync to settle.
        await asyncio.sleep(1.0)
        settled = len(server.resizes)
        await asyncio.sleep(3.0)
        assert len(server.resizes) == settled, (
            f"resize feedback at idle: {server.resizes[settled:]}"
        )

        # A real viewport change produces a small, terminating burst.
        await page.set_viewport_size({"width": 760, "height": 520})
        await asyncio.sleep(2.0)
        burst = len(server.resizes) - settled
        assert 1 <= burst <= 8, f"unexpected resize burst size {burst}"
        after_burst = len(server.resizes)
        await asyncio.sleep(2.5)
        assert len(server.resizes) == after_burst, (
            f"resize stream did not stop: {server.resizes[after_burst:]}"
        )

        # Canvas session 404 stops the 1 Hz poll.
        await asyncio.sleep(1.5)  # first poll + disable to land
        requests_after_disable = len(server.session_requests)
        await asyncio.sleep(3.0)
        assert len(server.session_requests) == requests_after_disable, (
            "canvas session poll kept firing after a 404 disable"
        )
        assert requests_after_disable >= 1
    finally:
        await context.stop()
