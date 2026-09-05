"""Real-Chromium proof for the Canvas V1 worker/renderer security boundary."""

from __future__ import annotations

import base64
import builtins
import hashlib
import html
import importlib
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from urllib.parse import urlsplit

import pytest
from playwright.async_api import async_playwright

from Tests.Canvas.browser.canvas_live_harness import (
    ProductRouteRecorder,
    exercise_adversarial_preview,
)
from Tests.Canvas.browser.canvas_live_harness import (
    attack_source as live_attack_source,
)
from Tests.Canvas.browser.canvas_live_harness import (
    chromium_executable as live_chromium_executable,
)
from tldw_chatbook.Canvas.compiler import CanvasCompileError, compile_canvas_document
from tldw_chatbook.Canvas.gateway import CanvasGateway
from tldw_chatbook.Canvas.models import CanvasScope, RenderNode
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController

ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "tldw_chatbook" / "Canvas" / "static"
FIXTURES = Path(__file__).with_name("fixtures")
RUNTIME_ASSETS = {
    "/static/canvas_renderer.js": "canvas_renderer.js",
    "/static/canvas_runtime_worker.js": "canvas_runtime_worker.js",
    "/static/quickjs-runtime.js": "quickjs-runtime.js",
}
RENDERER_CSP = (
    "default-src 'none'; "
    "script-src 'self' 'wasm-unsafe-eval'; worker-src data:; style-src 'unsafe-inline'; "
    "img-src blob:; connect-src 'none'; font-src 'none'; media-src 'none'; "
    "object-src 'none'; frame-src 'none'; child-src 'none'; form-action 'none'; "
    "base-uri 'none'; manifest-src 'none'; frame-ancestors 'self'; sandbox allow-scripts"
)


def _shell_html(startup_probe_url: str | None = None) -> bytes:
    document = """<!doctype html>
<meta charset="utf-8">
<title>Owned Canvas security harness</title>
<script>
(() => {
  'use strict';
  const state = {
    rendererReady: false,
    messages: [],
    status: null,
    startupApproved: null,
    nonce: null,
    port: null,
  };
  window.__canvasNativeWindowSentinel = 'native-shell-clean';
  Object.prototype.__canvasNativePrototypeSentinel = 'native-shell-clean';
  window.__canvasHarness = state;
  window.addEventListener('message', (event) => {
    const frame = document.getElementById('renderer');
    if (event.source !== frame.contentWindow) return;
    if (!event.data || event.data.type !== 'canvas:renderer-ready') return;
    state.rendererReady = true;
  });
  window.loadCanvas = async (plan) => {
    while (!state.rendererReady) await new Promise((resolve) => setTimeout(resolve, 5));
    const frame = document.getElementById('renderer');
    const channel = new MessageChannel();
    state.nonce = crypto.randomUUID();
    state.port = channel.port1;
    channel.port1.onmessage = async (event) => {
      const message = event.data;
      state.messages.push(message);
      if (message && message.type === 'canvas:execution-started') {
        state.startupApproved = await window.__canvasApproveExecution();
        if (state.startupApproved === true) {
          channel.port1.postMessage({type: 'canvas:execution-ack', nonce: state.nonce});
        }
      }
      if (message && message.type === 'canvas:status') state.status = message;
    };
    channel.port1.start();
    frame.contentWindow.postMessage(
      {type: 'canvas:init', nonce: state.nonce, plan},
      '*',
      [channel.port2],
    );
  };
  window.spoofRenderer = (message) => {
    document.getElementById('renderer').contentWindow.postMessage(message, '*');
  };
})();
</script>
<iframe id="renderer" name="canvas-renderer" sandbox="allow-scripts" src="/renderer.html"></iframe>
"""
    if startup_probe_url is not None:
        document += (
            '<img id="foreign-startup-probe" alt="" src="'
            + html.escape(startup_probe_url, quote=True)
            + '">\n'
        )
    return document.encode("utf-8")


def _renderer_html() -> bytes:
    renderer = STATIC / "canvas_renderer.js"
    digest = base64.b64encode(
        hashlib.sha384(renderer.read_bytes() if renderer.exists() else b"").digest()
    ).decode("ascii")
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="referrer" content="no-referrer">'
        f'<script type="module" src="/static/canvas_renderer.js" integrity="sha384-{digest}" crossorigin="anonymous"></script>'
        '</head><body><div id="canvas-root"></div></body></html>'
    ).encode()


def _expected_worker_bootstrap_url(origin: str) -> str:
    worker_url = f"{origin}/static/canvas_runtime_worker.js"
    bootstrap = (
        f"import({json.dumps(worker_url)}).then((module) => {{ "
        "module.startCanvasRuntimeWorker(globalThis); "
        'postMessage({type: "bootstrap-ready"}); '
        "}).catch((error) => "
        'postMessage({type: "bootstrap-failure", '
        'name: String(error && error.name || "Error")}));'
    )
    return "data:text/javascript;base64," + base64.b64encode(
        bootstrap.encode("utf-8")
    ).decode("ascii")


@dataclass(frozen=True)
class RequestRecord:
    method: str
    path: str
    headers: dict[str, str]


class _OwnedServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, assets: bool) -> None:
        self.assets = assets
        self.requests: list[RequestRecord] = []
        self.runtime_overrides: dict[str, bytes] = {}
        self.startup_probe_url: str | None = None
        self.lock = Lock()
        super().__init__(("127.0.0.1", 0), _OwnedHandler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def record(self, handler: BaseHTTPRequestHandler) -> None:
        with self.lock:
            self.requests.append(
                RequestRecord(
                    method=handler.command,
                    path=handler.path,
                    headers={
                        key.lower(): value for key, value in handler.headers.items()
                    },
                )
            )


class _OwnedHandler(BaseHTTPRequestHandler):
    server: _OwnedServer

    def do_GET(self) -> None:
        self.server.record(self)
        if not self.server.assets:
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"{self.server.origin}/redirect-target")
                self.end_headers()
                return
            self.send_response(204)
            self.end_headers()
            return

        if self.path == "/shell.html":
            self._send(
                200,
                _shell_html(self.server.startup_probe_url),
                "text/html; charset=utf-8",
            )
            return
        if self.path == "/renderer.html":
            self._send(
                200,
                _renderer_html(),
                "text/html; charset=utf-8",
                {
                    "Content-Security-Policy": RENDERER_CSP,
                    "Referrer-Policy": "no-referrer",
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-store",
                },
            )
            return
        if self.path in RUNTIME_ASSETS:
            path = STATIC / RUNTIME_ASSETS[self.path]
            body = self.server.runtime_overrides.get(self.path)
            if body is None and not path.is_file():
                self._send(404, b"missing trusted runtime asset", "text/plain")
                return
            self._send(
                200,
                body if body is not None else path.read_bytes(),
                "text/javascript; charset=utf-8",
                {
                    "Access-Control-Allow-Origin": "*",
                    "Cross-Origin-Resource-Policy": "cross-origin",
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        self.server.record(self)
        if self.server.assets:
            self._send(405, b"method not allowed", "text/plain")
            return
        self.send_response(204)
        self.end_headers()

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        pass


@dataclass(frozen=True)
class BrowserObservation:
    phase: str
    kind: str
    target: str
    detail: str = ""


class BrowserRecorder:
    def __init__(self, asset_server: _OwnedServer, egress_server: _OwnedServer) -> None:
        self._phase = "startup"
        self._lock = Lock()
        self.observations: list[BrowserObservation] = []
        self.asset_server = asset_server
        self.egress_server = egress_server
        self.asset_request_start = len(asset_server.requests)
        self.egress_request_start = len(egress_server.requests)
        self.startup_error: str | None = None
        self.expected_asset_count: int | None = None

    def arm_plan(self, plan: dict[str, Any]) -> None:
        assets = plan.get("assets")
        self.expected_asset_count = len(assets) if isinstance(assets, list) else 0

    def mark_execution(self) -> None:
        with self._lock:
            self._phase = "generated"

    def approve_execution(self) -> bool:
        try:
            self._assert_startup_allowlist()
        except AssertionError as exc:
            self.startup_error = f"startup allowlist rejected execution: {exc}"
            return False
        self.mark_execution()
        return True

    def _assert_startup_allowlist(self) -> None:
        origin = self.asset_server.origin
        expected_urls = {
            f"{origin}/shell.html": "document",
            f"{origin}/renderer.html": "document",
            f"{origin}/static/canvas_renderer.js": "script",
            f"{origin}/static/canvas_runtime_worker.js": "script",
            f"{origin}/static/quickjs-runtime.js": "script",
        }
        with self._lock:
            startup = [item for item in self.observations if item.phase == "startup"]
        requests = [item for item in startup if item.kind == "request"]
        actual_requests = sorted(
            (
                item.target,
                json.loads(item.detail)["method"],
                json.loads(item.detail)["resource_type"],
            )
            for item in requests
        )
        expected_http_requests = sorted(
            (url, "GET", resource_type) for url, resource_type in expected_urls.items()
        )
        blob_requests = [
            item for item in actual_requests if item[0].startswith("blob:null/")
        ]
        assert len(blob_requests) == self.expected_asset_count, blob_requests
        assert len({item[0] for item in blob_requests}) == len(blob_requests)
        assert all(
            method == "GET" and resource_type == "image"
            for _, method, resource_type in blob_requests
        ), blob_requests
        assert [
            item for item in actual_requests if not item[0].startswith("blob:null/")
        ] == expected_http_requests, actual_requests

        responses = [item for item in startup if item.kind == "response"]
        actual_responses = sorted(
            (
                item.target,
                json.loads(item.detail)["status"],
                json.loads(item.detail)["ok"],
            )
            for item in responses
        )
        expected_response_urls = [*expected_urls, *(item[0] for item in blob_requests)]
        assert actual_responses == sorted(
            (url, 200, True) for url in expected_response_urls
        ), actual_responses

        finished = [item for item in startup if item.kind == "request-finished"]
        actual_finished = sorted(
            (
                item.target,
                json.loads(item.detail)["method"],
                json.loads(item.detail)["resource_type"],
            )
            for item in finished
        )
        assert actual_finished == sorted([*expected_http_requests, *blob_requests]), (
            actual_finished
        )

        navigations = sorted(
            (item.target, item.detail) for item in startup if item.kind == "navigation"
        )
        assert navigations == sorted(
            [
                (f"{origin}/shell.html", "top"),
                (f"{origin}/renderer.html", "frame"),
            ]
        ), navigations

        workers = [item.target for item in startup if item.kind == "worker"]
        assert workers == [_expected_worker_bootstrap_url(origin)], workers
        assert all(
            item.kind
            in {"request", "response", "request-finished", "navigation", "worker"}
            for item in startup
        ), startup

        with self.asset_server.lock:
            server_requests = list(
                self.asset_server.requests[self.asset_request_start :]
            )
        actual_http = sorted(
            (
                item.method,
                item.path,
                item.headers.get("origin"),
                item.headers.get("sec-fetch-dest"),
                item.headers.get("sec-fetch-mode"),
                item.headers.get("sec-fetch-site"),
            )
            for item in server_requests
        )
        expected_http = sorted(
            [
                ("GET", "/shell.html", None, "document", "navigate", "none"),
                (
                    "GET",
                    "/renderer.html",
                    None,
                    "iframe",
                    "navigate",
                    "same-origin",
                ),
                *[
                    ("GET", path, "null", "script", "cors", "cross-site")
                    for path in (
                        "/static/canvas_renderer.js",
                        "/static/canvas_runtime_worker.js",
                        "/static/quickjs-runtime.js",
                    )
                ],
            ]
        )
        assert actual_http == expected_http, actual_http
        with self.egress_server.lock:
            egress_requests = list(
                self.egress_server.requests[self.egress_request_start :]
            )
        assert egress_requests == [], egress_requests

    def mark_csp_probe(self) -> None:
        with self._lock:
            self._phase = "native-csp-probe"

    def add(self, kind: str, target: str, detail: str = "") -> None:
        with self._lock:
            self.observations.append(
                BrowserObservation(self._phase, kind, target, detail)
            )

    def generated(self) -> list[BrowserObservation]:
        with self._lock:
            return [item for item in self.observations if item.phase == "generated"]


def _serve(assets: bool) -> Iterator[_OwnedServer]:
    server = _OwnedServer(assets=assets)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def asset_server() -> Iterator[_OwnedServer]:
    yield from _serve(True)


@pytest.fixture
def egress_server() -> Iterator[_OwnedServer]:
    yield from _serve(False)


@pytest.mark.loopback_network
def test_owned_egress_listener_records_get_post_and_redirect_receipt(
    egress_server: _OwnedServer,
) -> None:
    def request(method: str, path: str) -> tuple[int, str | None]:
        connection = HTTPConnection("127.0.0.1", egress_server.server_port, timeout=2)
        try:
            connection.request(
                method, path, body=b"probe" if method == "POST" else None
            )
            response = connection.getresponse()
            response.read()
            return response.status, response.getheader("Location")
        finally:
            connection.close()

    assert request("GET", "/plain") == (204, None)
    assert request("POST", "/form-or-beacon") == (204, None)
    status, location = request("GET", "/redirect")
    assert status == 302
    assert location == f"{egress_server.origin}/redirect-target"
    assert request("GET", "/redirect-target") == (204, None)
    assert [(item.method, item.path) for item in egress_server.requests] == [
        ("GET", "/plain"),
        ("POST", "/form-or-beacon"),
        ("GET", "/redirect"),
        ("GET", "/redirect-target"),
    ]


def _chromium_executable(browser_type: Any) -> str | None:
    configured = os.environ.get("TLDW_CANVAS_CHROMIUM_EXECUTABLE")
    if configured and Path(configured).is_file():
        return configured
    declared = Path(browser_type.executable_path)
    if declared.is_file():
        return str(declared)
    caches = [Path.home() / "Library" / "Caches" / "ms-playwright"]
    caches.extend(
        ancestor / "Library" / "Caches" / "ms-playwright" for ancestor in ROOT.parents
    )
    candidates = sorted(
        (
            executable
            for cache in caches
            for executable in cache.glob(
                "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"
            )
        ),
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return shutil.which("chromium") or shutil.which("chromium-browser")


@pytest.fixture
def playwright_runtime() -> Iterator[Any]:
    try:
        playwright_module = importlib.import_module("playwright.sync_api")
    except ImportError:
        pytest.fail(
            "Python Playwright is required for the Canvas security gate",
            pytrace=False,
        )
    with playwright_module.sync_playwright() as playwright:
        yield playwright


def test_python_playwright_is_a_mandatory_security_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builtin_import = builtins.__import__
    original_module_import = importlib.import_module

    def import_without_playwright(name: str, *args: object, **kwargs: object) -> Any:
        if name == "playwright.sync_api":
            raise ModuleNotFoundError("simulated missing mandatory Playwright")
        return original_builtin_import(name, *args, **kwargs)

    def module_import_without_playwright(
        name: str, *args: object, **kwargs: object
    ) -> Any:
        if name == "playwright.sync_api":
            raise ModuleNotFoundError("simulated missing mandatory Playwright")
        return original_module_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_playwright)
    monkeypatch.setattr(importlib, "import_module", module_import_without_playwright)
    monkeypatch.delitem(sys.modules, "playwright.sync_api", raising=False)
    fixture = playwright_runtime.__wrapped__()
    caught: BaseException | None = None
    try:
        next(fixture)
    except (pytest.fail.Exception, pytest.skip.Exception) as exc:
        caught = exc
    assert isinstance(caught, pytest.fail.Exception)
    assert "Python Playwright is required" in str(caught)


@pytest.fixture
def chromium_browser(playwright_runtime: Any) -> Iterator[Any]:
    executable = _chromium_executable(playwright_runtime.chromium)
    if executable is None:
        pytest.fail("real Playwright Chromium is required for the Canvas security gate")
    browser = playwright_runtime.chromium.launch(
        headless=True, executable_path=executable
    )
    try:
        yield browser
    finally:
        browser.close()


@pytest.fixture(params=["firefox", "webkit"])
def optional_browser(
    request: pytest.FixtureRequest, playwright_runtime: Any
) -> Iterator[tuple[str, Any]]:
    """Launch an optional installed engine, otherwise record an evidence-bearing skip."""
    name = str(request.param)
    browser_type = getattr(playwright_runtime, name)
    executable = Path(browser_type.executable_path)
    if not executable.is_file():
        pytest.skip(
            f"{name} is not installed in this CI/worktree; Chromium is the mandatory gate"
        )
    browser = browser_type.launch(headless=True, executable_path=str(executable))
    try:
        yield name, browser
    finally:
        browser.close()


def _render_node(node: RenderNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "tag": node.tag,
        "attributes": [list(attribute) for attribute in node.attributes],
        "text": node.text,
        "children": [_render_node(child) for child in node.children],
    }


def _wire_plan(source: str) -> dict[str, Any]:
    plan = compile_canvas_document(source)
    return {
        "runtime_profile": plan.runtime_profile,
        "source_identity": asdict(plan.source_identity),
        "root": _render_node(plan.root),
        "assets": [
            {
                "asset_id": asset.asset_id,
                "mime_type": asset.mime_type,
                "data_base64": base64.b64encode(asset.data).decode("ascii"),
            }
            for asset in plan.assets
        ],
        "css_rules": list(plan.css_rules),
        "scripts": list(plan.scripts),
    }


def _node_id_for_html_id(node: dict[str, Any], html_id: str) -> str:
    stack = [node]
    while stack:
        current = stack.pop()
        if ["id", html_id] in current["attributes"]:
            return str(current["node_id"])
        stack.extend(reversed(current["children"]))
    raise AssertionError(f"compiled plan has no node with id={html_id!r}")


def _transaction_worker(transaction: dict[str, Any]) -> bytes:
    encoded = json.dumps(transaction, separators=(",", ":"))
    return f"""import "./quickjs-runtime.js";
export function startCanvasRuntimeWorker(self) {{
  self.__canvasNativeWorkerSentinel = "native-worker-clean";
  self.onmessage = (event) => {{
    if (event.data.type === "prepare") {{
      postMessage({{type: "prepared", native_worker_sentinel: "native-worker-clean"}});
      return;
    }}
    if (event.data.type === "execute") {{
      const transaction = {encoded};
      postMessage({{
        type: "transaction",
        operation_id: event.data.operation_id,
        operation_kind: "startup",
        patches: transaction.patches,
        bridges: transaction.bridges,
        native_worker_sentinel: "native-worker-clean",
      }});
    }}
  }};
}}
""".encode()


def _attack_source(script: str) -> str:
    clobbering_ids = "".join(
        f'<span id="{name}">{name}</span>'
        for name in ("fetch", "location", "parent", "postMessage", "Worker")
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>attack</title></head>'
        f'<body>{clobbering_ids}<button id="attack" type="button">attack</button>'
        f'<output id="status">static</output><script>{script}</script></body></html>'
    )


def _new_page(
    browser: Any, asset_server: _OwnedServer, egress_server: _OwnedServer
) -> tuple[Any, Any, BrowserRecorder]:
    context = browser.new_context(accept_downloads=True)
    context.add_init_script(
        "window.__canvasNativeWindowSentinel = 'native-frame-clean';"
        "Object.prototype.__canvasNativePrototypeSentinel = 'native-frame-clean';"
    )
    recorder = BrowserRecorder(asset_server, egress_server)
    page = context.new_page()
    page.expose_function("__canvasApproveExecution", recorder.approve_execution)
    context.on(
        "request",
        lambda request: recorder.add(
            "request",
            request.url,
            json.dumps(
                {"method": request.method, "resource_type": request.resource_type},
                sort_keys=True,
            ),
        ),
    )
    context.on(
        "response",
        lambda response: recorder.add(
            "response",
            response.url,
            json.dumps(
                {
                    "headers": response.headers,
                    "ok": response.ok,
                    "status": response.status,
                },
                sort_keys=True,
            ),
        ),
    )
    context.on(
        "requestfinished",
        lambda request: recorder.add(
            "request-finished",
            request.url,
            json.dumps(
                {"method": request.method, "resource_type": request.resource_type},
                sort_keys=True,
            ),
        ),
    )
    page.on("websocket", lambda socket: recorder.add("websocket", socket.url))
    page.on(
        "console", lambda message: recorder.add("console", message.text, message.type)
    )
    page.on("pageerror", lambda error: recorder.add("pageerror", str(error)))
    page.on("popup", lambda popup: recorder.add("popup", popup.url))
    page.on(
        "download",
        lambda download: recorder.add("download", download.suggested_filename),
    )
    page.on("worker", lambda worker: recorder.add("worker", worker.url))
    page.on(
        "framenavigated",
        lambda frame: recorder.add(
            "navigation", frame.url, "top" if frame == page.main_frame else "frame"
        ),
    )
    page.goto(f"{asset_server.origin}/shell.html", wait_until="load")
    page.wait_for_function(
        "window.__canvasHarness.rendererReady === true", timeout=5_000
    )
    return context, page, recorder


def _load(
    page: Any, plan: dict[str, Any], recorder: BrowserRecorder | None = None
) -> dict[str, Any]:
    if recorder is not None:
        recorder.arm_plan(plan)
    page.evaluate("plan => window.loadCanvas(plan)", plan)
    try:
        page.wait_for_function(
            "window.__canvasHarness.startupApproved === false || "
            "(window.__canvasHarness.status && "
            "['ready', 'failed'].includes(window.__canvasHarness.status.state))",
            timeout=10_000,
        )
    except Exception as exc:
        messages = page.evaluate("window.__canvasHarness.messages")
        observations = recorder.observations if recorder is not None else []
        raise AssertionError(
            "Canvas runtime produced no terminal status; "
            f"messages={messages!r}; observations={observations!r}"
        ) from exc
    if recorder is not None and recorder.startup_error is not None:
        raise AssertionError(recorder.startup_error)
    return page.evaluate("window.__canvasHarness.status")


def _assert_zero_generated_egress(
    recorder: BrowserRecorder,
    egress_server: _OwnedServer,
) -> None:
    forbidden = [
        item
        for item in recorder.generated()
        if item.kind
        in {"request", "websocket", "navigation", "popup", "download", "worker"}
    ]
    assert forbidden == []
    assert egress_server.requests == []


@pytest.mark.loopback_network
def test_benign_counter_form_timer_svg_and_typed_submit_work_with_zero_egress(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        source = (FIXTURES / "benign_canvas.html").read_text(encoding="utf-8")
        status = _load(page, _wire_plan(source), recorder)
        if status["state"] != "ready":
            print("CANVAS_STARTUP_FAILURE=" + repr((status, recorder.observations)))
        assert status["state"] == "ready", (status, recorder.observations)
        assert status["engine"] == "quickjs-wasm"
        assert status["native_worker_sentinel"] == "native-worker-clean"

        frame = page.frame(name="canvas-renderer")
        assert frame is not None
        frame.locator("#counter").click()
        frame.locator("#counter").click()
        frame.locator("#answer").fill("forty two")
        frame.locator("#submit").click()
        frame.locator("#timer").wait_for(state="visible")
        page.wait_for_function(
            "window.__canvasHarness.messages.some((item) => "
            "item.type === 'canvas:bridge-request' && item.kind === 'submit')",
        )

        assert frame.locator("#count").text_content() == "2"
        assert frame.locator("#submitted").text_content() == "forty two"
        frame.locator("#timer").wait_for()
        assert frame.locator("#timer").text_content() == "timer fired"
        for image_id in ("pixel", "jpeg-pixel", "gif-pixel", "webp-pixel"):
            frame.locator(f"#{image_id}").evaluate("image => image.decode()")
            assert frame.locator(f"#{image_id}").evaluate(
                "image => [image.naturalWidth, image.naturalHeight]"
            ) == [1, 1]
        assert frame.locator("svg circle").count() == 1
        bridge = page.evaluate(
            "window.__canvasHarness.messages.find((item) => "
            "item.type === 'canvas:bridge-request' && item.kind === 'submit')"
        )
        assert bridge["value"] == {"answer": "forty two"}
        assert page.url == f"{asset_server.origin}/shell.html"
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.loopback_network
def test_dom_move_detach_and_reinsert_preserve_virtual_identity_and_bounds(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        expect = importlib.import_module("playwright.sync_api").expect
        source = """<!doctype html><html><body>
<div id="left"><section id="moving"><button id="inside">inside</button><span id="descendant">descendant</span></section></div>
<div id="right"><span id="marker">marker</span></div>
<button id="move">move</button><button id="detach">detach</button>
<button id="reinsert">reinsert</button><button id="remount">remount</button><button id="same">same</button>
<button id="cycle">cycle</button><output id="count">0</output>
<output id="result">ready</output>
<script>
const left = document.getElementById("left");
const right = document.getElementById("right");
const moving = document.getElementById("moving");
const marker = document.getElementById("marker");
const count = document.getElementById("count");
const result = document.getElementById("result");
let clicks = 0;
document.getElementById("inside").addEventListener("click", () => { count.textContent = String(++clicks); });
document.getElementById("move").addEventListener("click", () => { right.insertBefore(moving, marker); });
document.getElementById("detach").addEventListener("click", () => { moving.parentNode.removeChild(moving); });
document.getElementById("reinsert").addEventListener("click", () => { left.appendChild(moving); });
document.getElementById("remount").addEventListener("click", () => {
  right.appendChild(moving); left.appendChild(moving); right.insertBefore(moving, marker);
});
document.getElementById("same").addEventListener("click", () => { moving.parentNode.insertBefore(moving, moving); result.textContent = "same-ok"; });
document.getElementById("cycle").addEventListener("click", () => {
  try { moving.appendChild(left); result.textContent = "cycle-mutated"; }
  catch (_) { result.textContent = "cycle-refused"; }
});
</script></body></html>"""
        status = _load(page, _wire_plan(source), recorder)
        assert status["state"] == "ready"
        frame = page.frame(name="canvas-renderer")
        assert frame is not None
        frame.evaluate("window.__ownedMoving = document.querySelector('#moving')")

        frame.locator("#move").click()
        frame.locator("#right").wait_for()
        assert frame.locator("#right > *").evaluate_all(
            "nodes => nodes.map(node => node.id)"
        ) == ["moving", "marker"]
        assert (
            frame.evaluate("window.__ownedMoving === document.querySelector('#moving')")
            is True
        )
        assert frame.locator("#descendant").text_content() == "descendant"
        frame.locator("#inside").click()
        expect(frame.locator("#count")).to_have_text("1")
        assert frame.locator("#count").text_content() == "1"

        frame.locator("#detach").click()
        frame.locator("#moving").wait_for(state="detached")
        frame.locator("#remount").click()
        expect(frame.locator("#right > #moving")).to_have_count(1)
        assert frame.locator("#right > *").evaluate_all(
            "nodes => nodes.map(node => node.id)"
        ) == ["moving", "marker"]
        assert (
            frame.evaluate("window.__ownedMoving === document.querySelector('#moving')")
            is False
        )

        for _ in range(20):
            frame.locator("#detach").click()
            frame.locator("#moving").wait_for(state="detached")
            frame.locator("#reinsert").click()
            frame.locator("#moving").wait_for(state="attached")
        assert frame.locator("#descendant").text_content() == "descendant"
        frame.locator("#inside").click()
        expect(frame.locator("#count")).to_have_text("2")
        assert frame.locator("#count").text_content() == "2"

        frame.locator("#same").click()
        expect(frame.locator("#result")).to_have_text("same-ok")
        assert frame.locator("#result").text_content() == "same-ok"
        frame.locator("#cycle").click()
        expect(frame.locator("#result")).to_have_text("cycle-refused")
        assert frame.locator("#result").text_content() == "cycle-refused"
        assert frame.locator("#left > #moving").count() == 1
        assert page.evaluate("window.__canvasHarness.status.state") == "ready"
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.loopback_network
def test_adversarial_corpus_has_zero_egress_and_never_mutates_native_realms(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    cases = json.loads(
        (FIXTURES / "adversarial_scripts.json").read_text(encoding="utf-8")
    )
    evidence: list[dict[str, Any]] = []
    for case in cases:
        context, page, recorder = _new_page(
            chromium_browser, asset_server, egress_server
        )
        try:
            script = (
                case["script"]
                .replace("__EGRESS__", egress_server.origin)
                .replace(
                    "__WEBSOCKET__",
                    egress_server.origin.replace("http://", "ws://", 1),
                )
            )
            status = _load(page, _wire_plan(_attack_source(script)), recorder)
            frame = page.frame(name="canvas-renderer")
            assert frame is not None
            if case["expected"] == "failed-after-click":
                assert status["state"] == "ready", case["name"]
                frame.locator("#attack").click(no_wait_after=True)
                page.wait_for_function(
                    "window.__canvasHarness.status.state === 'failed'", timeout=10_000
                )
                status = page.evaluate("window.__canvasHarness.status")
            elif case["expected"] == "failed-after-clicks":
                assert status["state"] == "ready", case["name"]
                for _ in range(case["click_count"]):
                    frame.locator("#attack").click(no_wait_after=True)
                page.wait_for_function(
                    "window.__canvasHarness.status.state === 'failed'", timeout=10_000
                )
                status = page.evaluate("window.__canvasHarness.status")
            elif case["expected"] == "failed-after-ready":
                assert status["state"] == "ready", case["name"]
                page.wait_for_function(
                    "window.__canvasHarness.status.state === 'failed'", timeout=10_000
                )
                status = page.evaluate("window.__canvasHarness.status")
            elif case["expected"] == "failed-after-event-storm":
                assert status["state"] == "ready", case["name"]
                frame.locator("#attack").evaluate(
                    "node => { for (let index = 0; index < 102; index += 1) node.click(); }"
                )
                page.wait_for_function(
                    "window.__canvasHarness.status.state === 'failed'", timeout=10_000
                )
                status = page.evaluate("window.__canvasHarness.status")
            else:
                assert status["state"] == case["expected"], case["name"]
            if case.get("settle_milliseconds"):
                page.wait_for_timeout(case["settle_milliseconds"])
            if case.get("expected_code"):
                assert status["code"] == case["expected_code"], case["name"]
            if case.get("expected_text"):
                assert frame.locator("#status").text_content() == case["expected_text"]
            if case.get("bridge_kind"):
                bridge_messages = page.evaluate(
                    "window.__canvasHarness.messages.filter((item) => "
                    "item.type === 'canvas:bridge-request')"
                )
                assert [item["kind"] for item in bridge_messages] == [
                    case["bridge_kind"]
                ]
            if "expected_bridge_kinds" in case:
                bridge_messages = page.evaluate(
                    "window.__canvasHarness.messages.filter((item) => "
                    "item.type === 'canvas:bridge-request')"
                )
                assert [item["kind"] for item in bridge_messages] == case[
                    "expected_bridge_kinds"
                ], case["name"]

            assert page.evaluate("window.__canvasNativeWindowSentinel") in {
                "native-shell-clean",
                "native-frame-clean",
            }
            assert page.evaluate(
                "Object.prototype.__canvasNativePrototypeSentinel"
            ) in {"native-shell-clean", "native-frame-clean"}
            assert (
                frame.evaluate("window.__canvasNativeWindowSentinel")
                == "native-frame-clean"
            )
            assert (
                frame.evaluate("Object.prototype.__canvasNativePrototypeSentinel")
                == "native-frame-clean"
            )
            assert (
                page.evaluate("typeof window.__canvasGeneratedFunction") == "undefined"
            )
            assert (
                frame.evaluate("typeof window.__canvasGeneratedFunction") == "undefined"
            )
            assert status["native_worker_sentinel"] == "native-worker-clean"
            assert page.url == f"{asset_server.origin}/shell.html"
            _assert_zero_generated_egress(recorder, egress_server)
            evidence.append(
                {
                    "case": case["name"],
                    "state": status["state"],
                    "code": status.get("code"),
                    "post_start_observations": [
                        asdict(item) for item in recorder.generated()
                    ],
                }
            )
        finally:
            context.close()

    print("CANVAS_ZERO_EGRESS_CORPUS=" + json.dumps(evidence, sort_keys=True))


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_canonical_adversarial_corpus_stays_in_native_product_route(
    egress_server: _OwnedServer,
) -> None:
    """Replay every canonical attack through the shipped native gateway."""

    cases = json.loads(
        (FIXTURES / "adversarial_scripts.json").read_text(encoding="utf-8")
    )
    canary_path = "/canvas-generated-same-origin-canary"
    cases.append(
        {
            "name": "same_origin_relative_request",
            "script": f"try {{ fetch('{canary_path}'); }} catch (_error) {{}}",
            "expected": "ready",
        }
    )
    session_id = "native-adversarial"
    scope = CanvasScope(
        session_id=session_id,
        conversation_id=session_id,
        active_message_ids=("assistant-adversarial",),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="native-adversarial-bootstrap",
    )
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: scope,
        canvas_controller=controller,
    )
    gateway = CanvasGateway(authority=authority)
    recorder = ProductRouteRecorder()

    try:
        initial = authority.import_html(
            session_id=session_id,
            source="<!doctype html><title>Adversarial gate</title><p>ready</p>",
            create_new=True,
        )
        browser_scope = authority.gateway_scope(
            session_id=session_id,
            browser_session_id="native-adversarial-browser",
            canvas_id=initial.canvas_id,
            revision_id=initial.revision_id,
        )
        launch = await gateway.open_shell(browser_scope)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                executable_path=live_chromium_executable(playwright.chromium),
            )
            context = await browser.new_context()
            await recorder.install_execution_boundary(context)
            await context.add_init_script(
                "window.__canvasNativeWindowSentinel='native-route-clean';"
                "Object.prototype.__canvasNativePrototypeSentinel='native-route-clean';"
            )
            page = await context.new_page()
            recorder.attach(context, page)
            page.set_default_timeout(12_000)
            await page.goto(launch.browser_url)
            preview = page.frame_locator("#canvas-preview")
            await page.locator("#loading-state").wait_for(state="hidden")
            shell = page
            accepted_sequence = 1
            for index, case in enumerate(cases):
                script = (
                    str(case["script"])
                    .replace("__EGRESS__", egress_server.origin)
                    .replace(
                        "__WEBSOCKET__",
                        egress_server.origin.replace("http://", "ws://", 1),
                    )
                )
                scope = CanvasScope(
                    session_id=session_id,
                    conversation_id=session_id,
                    active_message_ids=("assistant-adversarial",),
                    selected_canvas_id=None,
                    selected_revision_id=None,
                    run_id=f"native-adversarial-{index}",
                )
                recorder.begin_load()
                try:
                    authority.import_html(
                        session_id=session_id,
                        source=live_attack_source(script, marker=str(case["name"])),
                        create_new=False,
                    )
                except CanvasCompileError as error:
                    pytest.fail(
                        f"unexpected admission refusal: {case['name']}:"
                        f"{sorted({issue.code for issue in error.issues})}",
                        pytrace=False,
                    )
                accepted_sequence += 1
                await page.get_by_text(
                    f"Revision {accepted_sequence}", exact=True
                ).wait_for()
                await exercise_adversarial_preview(
                    shell, preview, case, expect_bridge=False
                )
                assert egress_server.requests == [], case["name"]
                assert await page.evaluate("window.__canvasNativeWindowSentinel") == (
                    "native-route-clean"
                )
                assert (
                    await page.evaluate(
                        "Object.prototype.__canvasNativePrototypeSentinel"
                    )
                    == "native-route-clean"
                )
                assert (
                    await preview.locator("#attack").evaluate(
                        "() => window.__canvasNativeWindowSentinel"
                    )
                    == "native-route-clean"
                )
            launch_url = urlsplit(launch.browser_url)
            recorder.assert_generated_confined(
                trusted_origin=f"{launch_url.scheme}://{launch_url.netloc}",
                trusted_route_root=launch_url.path,
                forbidden_canary_path=canary_path,
            )
            await browser.close()
    finally:
        await gateway.aclose()
        controller.close_runtime()


@pytest.mark.parametrize(
    ("kind", "target", "detail"),
    [
        (
            "request",
            "https://127.0.0.1:9443/canvas/cap/api/plan?escape=1",
            json.dumps(
                {
                    "method": "GET",
                    "owner": "https://127.0.0.1:9443/canvas/cap/render",
                    "resource_type": "fetch",
                },
                sort_keys=True,
            ),
        ),
        ("navigation", "data:text/html,escape", "frame"),
        (
            "request",
            "https://127.0.0.1:9443/canvas/cap/api/plan",
            json.dumps(
                {
                    "method": "GET",
                    "owner": "https://127.0.0.1:9443/canvas/cap/render",
                    "resource_type": "fetch",
                },
                sort_keys=True,
            ),
        ),
    ],
    ids=["same-origin-query", "data-navigation", "late-renderer-request"],
)
def test_product_route_recorder_rejects_generated_escape_surfaces(
    kind: str, target: str, detail: str
) -> None:
    recorder = ProductRouteRecorder()
    recorder.mark_generated()
    recorder.add(kind, target, detail)

    with pytest.raises(AssertionError):
        recorder.assert_generated_confined(
            trusted_origin="https://127.0.0.1:9443",
            trusted_route_root="/canvas/cap/",
            forbidden_canary_path="/canvas-generated-same-origin-canary",
        )


def test_product_route_recorder_rejects_foreign_startup_worker() -> None:
    recorder = ProductRouteRecorder()
    recorder.begin_load()
    recorder.add("worker", "data:text/javascript,fetch('/leak')")
    with pytest.raises(AssertionError):
        recorder.assert_generated_confined(
            trusted_origin="https://127.0.0.1:9443",
            trusted_route_root="/canvas/cap/",
            forbidden_canary_path="/canvas-generated-same-origin-canary",
        )


@pytest.mark.parametrize("missing_kind", ["request", "response", "request-finished"])
def test_product_execution_requires_completed_startup_census(missing_kind) -> None:
    recorder = ProductRouteRecorder()
    root = "https://127.0.0.1:9443/canvas/cap"
    expected = [(root + "/render", "document")]
    expected += [
        (root + "/static/" + name, "script")
        for name in (
            "canvas_renderer.js",
            "canvas_runtime_worker.js",
            "quickjs-runtime.js",
        )
    ]
    for url, resource_type in expected:
        for kind in ("request", "response", "request-finished"):
            if kind != missing_kind:
                recorder.add(
                    kind,
                    url,
                    json.dumps(
                        {
                            "method": "GET",
                            "resource_type": resource_type,
                            "owner": root + "/render",
                            "status": 200,
                        }
                    ),
                )
    with pytest.raises(AssertionError, match="startup .*census"):
        recorder.assert_startup_complete(root + "/")
    assert recorder.execution_ack_count == 0


@pytest.mark.loopback_network
def test_opaque_iframe_and_csp_block_native_parent_storage_requests_forms_and_images(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        status = _load(
            page,
            _wire_plan(
                _attack_source(
                    "document.getElementById('status').textContent = 'ready';"
                )
            ),
            recorder,
        )
        assert status["state"] == "ready"
        frame = page.frame(name="canvas-renderer")
        assert frame is not None

        isolation = frame.evaluate(
            """() => {
              const result = {};
              try { result.parent = parent.document.title; } catch (error) { result.parent = error.name; }
              try { localStorage.setItem('escape', '1'); result.storage = 'allowed'; }
              catch (error) { result.storage = error.name; }
              return result;
            }"""
        )
        assert isolation == {"parent": "SecurityError", "storage": "SecurityError"}

        recorder.mark_csp_probe()
        csp_results = frame.evaluate(
            """async (target) => {
              const result = {};
              window.__canvasInlineScriptExecuted = false;
              const script = document.createElement('script');
              script.textContent = "window.__canvasInlineScriptExecuted = true";
              document.body.appendChild(script);
              result.inlineScript = window.__canvasInlineScriptExecuted ? 'allowed' : 'blocked';
              try { await fetch(target + '/csp-fetch'); result.fetch = 'allowed'; }
              catch (_) { result.fetch = 'blocked'; }
              result.image = await new Promise((resolve) => {
                const image = document.createElement('img');
                image.onload = () => resolve('allowed');
                image.onerror = () => resolve('blocked');
                image.src = target + '/csp-image';
                document.body.appendChild(image);
              });
              const form = document.createElement('form');
              form.action = target + '/csp-form';
              form.method = 'POST';
              document.body.appendChild(form);
              try { form.submit(); result.form = 'attempted'; }
              catch (_) { result.form = 'blocked'; }
              await new Promise((resolve) => setTimeout(resolve, 100));
              return result;
            }""",
            egress_server.origin,
        )
        assert csp_results["fetch"] == "blocked"
        assert csp_results["image"] == "blocked"
        assert csp_results["inlineScript"] == "blocked"
        assert egress_server.requests == []
        assert page.url == f"{asset_server.origin}/shell.html"

        iframe = page.locator("#renderer")
        assert iframe.get_attribute("sandbox") == "allow-scripts"
        renderer_requests = [
            request
            for request in asset_server.requests
            if request.path == "/renderer.html"
        ]
        assert len(renderer_requests) == 1
        renderer_response = next(
            item
            for item in recorder.observations
            if item.kind == "response" and item.target.endswith("/renderer.html")
        )
        renderer_headers = json.loads(renderer_response.detail)["headers"]
        assert renderer_headers["content-security-policy"] == RENDERER_CSP
        assert renderer_headers["referrer-policy"] == "no-referrer"
        assert renderer_headers["x-content-type-options"] == "nosniff"
        assert renderer_headers["cache-control"] == "no-store"
        expected_paths = {
            "/shell.html",
            "/renderer.html",
            "/static/canvas_renderer.js",
            "/static/canvas_runtime_worker.js",
            "/static/quickjs-runtime.js",
        }
        assert {request.path for request in asset_server.requests} == expected_paths
        startup_workers = [
            item
            for item in recorder.observations
            if item.kind == "worker" and item.phase == "startup"
        ]
        assert len(startup_workers) == 1
        generated_requests = [
            item
            for item in recorder.observations
            if item.phase == "generated" and item.kind == "request"
        ]
        assert generated_requests == []
        probe_requests = [
            item.target
            for item in recorder.observations
            if item.phase == "native-csp-probe" and item.kind == "request"
        ]
        assert probe_requests == [f"{egress_server.origin}/csp-image"]
        startup_http_requests = [
            {
                "method": request.method,
                "path": request.path,
                "origin": request.headers.get("origin"),
                "sec_fetch_dest": request.headers.get("sec-fetch-dest"),
                "sec_fetch_mode": request.headers.get("sec-fetch-mode"),
                "sec_fetch_site": request.headers.get("sec-fetch-site"),
            }
            for request in asset_server.requests
        ]
        assert all(item["method"] == "GET" for item in startup_http_requests)
        startup_observation_counts: dict[str, int] = {}
        for item in recorder.observations:
            if item.phase == "startup":
                startup_observation_counts[item.kind] = (
                    startup_observation_counts.get(item.kind, 0) + 1
                )
        assert startup_observation_counts == {
            "navigation": 2,
            "request": 5,
            "request-finished": 5,
            "response": 5,
            "worker": 1,
        }
        print(
            "CANVAS_CSP_EVIDENCE="
            + json.dumps(
                {
                    "csp": RENDERER_CSP,
                    "startup_requests": sorted(expected_paths),
                    "startup_http_request_records": startup_http_requests,
                    "startup_observation_counts": startup_observation_counts,
                    "generated_requests": [],
                    "native_csp_probe_request_observations": probe_requests,
                    "server_received_probe_requests": [],
                    "inline_native_script": csp_results["inlineScript"],
                    "opaque_parent_access": isolation["parent"],
                    "opaque_storage_access": isolation["storage"],
                },
                sort_keys=True,
            )
        )
    finally:
        context.close()


@pytest.mark.loopback_network
def test_startup_allowlist_withholds_execution_ack_for_foreign_observation(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    asset_server.startup_probe_url = f"{egress_server.origin}/foreign-startup"
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        with pytest.raises(AssertionError, match="startup allowlist"):
            _load(
                page,
                _wire_plan(
                    _attack_source(
                        "document.getElementById('status').textContent = 'must-not-run';"
                    )
                ),
                recorder,
            )
        assert page.evaluate(
            "window.__canvasHarness.messages.some((item) => "
            "item.type === 'canvas:execution-started')"
        )
        assert not page.evaluate(
            "window.__canvasHarness.messages.some((item) => "
            "item.type === 'canvas:status' && item.state === 'ready')"
        )
        assert [(item.method, item.path) for item in egress_server.requests] == [
            ("GET", "/foreign-startup")
        ]
    finally:
        context.close()


@pytest.mark.loopback_network
def test_window_message_spoof_cannot_forge_worker_status_or_bridge(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        status = _load(
            page,
            _wire_plan(
                _attack_source(
                    "document.getElementById('status').textContent = 'genuine';"
                )
            ),
            recorder,
        )
        assert status["state"] == "ready"
        count_before = page.evaluate("window.__canvasHarness.messages.length")
        page.evaluate(
            """() => window.spoofRenderer({
              type: 'canvas:worker-message',
              nonce: window.__canvasHarness.nonce,
              message: {type: 'bridge', kind: 'submit', value: 'spoofed'}
            })"""
        )
        time.sleep(0.05)
        assert page.evaluate("window.__canvasHarness.messages.length") == count_before
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.loopback_network
def test_renderer_revalidates_tampered_plans_before_worker_start(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    source = _attack_source(
        "document.getElementById('status').textContent = 'must not execute';"
    )
    plans: list[tuple[str, dict[str, Any]]] = []

    def png_header(width: int, height: int) -> bytes:
        return (
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\rIHDR"
            + width.to_bytes(4, "big")
            + height.to_bytes(4, "big")
            + b"\x08\x06\x00\x00\x00"
            + b"\x00\x00\x00\x00"
            + b"\x00\x00\x00\x00IEND\x00\x00\x00\x00"
        )

    tag_plan = _wire_plan(source)
    tag_plan["root"]["tag"] = "script"
    plans.append(("active-tag", tag_plan))

    css_plan = _wire_plan(source)
    css_plan["css_rules"] = [f"@import url('{egress_server.origin}/css-import')"]
    plans.append(("css-import", css_plan))

    asset_plan = _wire_plan(source)
    asset_plan["assets"] = [
        {
            "asset_id": "forged-image",
            "mime_type": "image/png",
            "data_base64": base64.b64encode(b"<svg onload=alert(1)>").decode("ascii"),
        }
    ]
    plans.append(("asset-signature", asset_plan))

    dimension_plan = _wire_plan(source)
    dimension_plan["assets"] = [
        {
            "asset_id": "oversized-pixels",
            "mime_type": "image/png",
            "data_base64": base64.b64encode(png_header(5_000, 5_000)).decode("ascii"),
        }
    ]
    plans.append(("image-dimensions", dimension_plan))

    pixel_plan = _wire_plan(source)
    pixel_plan["assets"] = [
        {
            "asset_id": "oversized-pixel-count",
            "mime_type": "image/png",
            "data_base64": base64.b64encode(png_header(4_096, 2_048)).decode("ascii"),
        }
    ]
    plans.append(("image-pixels", pixel_plan))

    malformed_plan = _wire_plan(source)
    malformed_plan["assets"] = [
        {
            "asset_id": "malformed-image",
            "mime_type": "image/png",
            "data_base64": base64.b64encode(png_header(1, 1)).decode("ascii"),
        }
    ]
    plans.append(("image-native-decode", malformed_plan))

    animation_plan = _wire_plan(source)
    gif_frame = b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x4c\x01\x00"
    animated_gif = (
        b"GIF89a\x01\x00\x01\x00\x00\x00\x00" + gif_frame + gif_frame + b"\x3b"
    )
    animation_plan["assets"] = [
        {
            "asset_id": "animated-image",
            "mime_type": "image/gif",
            "data_base64": base64.b64encode(animated_gif).decode("ascii"),
        }
    ]
    plans.append(("image-animation", animation_plan))

    asset_count_plan = _wire_plan(source)
    asset_count_plan["assets"] = [
        {
            "asset_id": f"asset-{index}",
            "mime_type": "image/png",
            "data_base64": "",
        }
        for index in range(65)
    ]
    plans.append(("asset-count", asset_count_plan))

    node_plan = _wire_plan(source)
    node_plan["root"]["children"].extend(
        {
            "node_id": f"forged-node-{index}",
            "tag": "span",
            "attributes": [],
            "text": None,
            "children": [],
        }
        for index in range(5_001)
    )
    plans.append(("node-count", node_plan))

    nested_css_plan = _wire_plan(source)
    nested_css_plan["css_rules"] = [
        f"#status {{ color: red; & {{ background-image: url('{egress_server.origin}/nested-css') }} }}"
    ]
    plans.append(("nested-style-rule", nested_css_plan))

    late_plan = _wire_plan(
        (FIXTURES / "benign_canvas.html").read_text(encoding="utf-8")
    )
    late_plan["root"]["children"].append(
        {
            "node_id": "invalid-last-node",
            "tag": "script",
            "attributes": [],
            "text": None,
            "children": [],
        }
    )
    plans.append(("invalid-last-after-assets-css-and-dom", late_plan))

    for name, plan in plans:
        context, page, recorder = _new_page(
            chromium_browser, asset_server, egress_server
        )
        try:
            status = _load(page, plan, recorder)
            assert status["state"] == "failed", name
            assert status["code"] == "invalid-plan", name
            assert status["scripts_disabled"] is True, name
            assert not any(item.kind == "worker" for item in recorder.observations), (
                name
            )
            assert not any(
                item.kind == "request" and item.target.startswith(egress_server.origin)
                for item in recorder.observations
            ), name
            frame = page.frame(name="canvas-renderer")
            assert frame is not None
            assert (
                frame.locator("#canvas-root").evaluate("node => node.childNodes.length")
                == 0
            ), name
            assert frame.evaluate("document.adoptedStyleSheets.length") == 0, name
            assert frame.locator("#canvas-root img").count() == 0, name
            assert (
                page.evaluate(
                    "window.__canvasHarness.messages.filter((item) => "
                    "item.type === 'canvas:bridge-request').length"
                )
                == 0
            ), name
            assert egress_server.requests == [], name
        finally:
            context.close()


@pytest.mark.loopback_network
def test_renderer_rejects_whole_invalid_transaction_without_partial_effects(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    plan = _wire_plan((FIXTURES / "benign_canvas.html").read_text(encoding="utf-8"))
    status_id = _node_id_for_html_id(plan["root"], "count")
    valid_patches = [
        {"op": "set-text", "node_id": status_id, "value": "mutated"},
        {
            "op": "set-style",
            "node_id": status_id,
            "name": "color",
            "value": "red",
        },
    ]
    transactions = [
        (
            "invalid-last-patch",
            {
                "patches": [
                    *valid_patches,
                    {
                        "op": "set-attribute",
                        "node_id": status_id,
                        "name": "src",
                        "value": f"{egress_server.origin}/transaction-asset",
                    },
                ],
                "bridges": [],
            },
        ),
        (
            "invalid-last-bridge",
            {
                "patches": valid_patches,
                "bridges": [
                    {
                        "request_id": "valid-first",
                        "kind": "submit",
                        "value": {"before": "invalid-last"},
                    },
                    {
                        "request_id": "invalid-last",
                        "kind": "native-host-effect",
                        "value": "must-not-escape",
                    },
                ],
            },
        ),
        (
            "forged-raster-bridge",
            {
                "patches": valid_patches,
                "bridges": [
                    {
                        "request_id": "forged-raster",
                        "kind": "download",
                        "value": {
                            "filename": "pixel.png",
                            "mime_type": "image/png",
                            "data": "data:image/png;base64,PGh0bWw+",
                        },
                    }
                ],
            },
        ),
        (
            "raw-control-filename-bridge",
            {
                "patches": valid_patches,
                "bridges": [
                    {
                        "request_id": "raw-control-filename",
                        "kind": "download",
                        "value": {
                            "filename": "\nreport.txt",
                            "mime_type": "text/plain",
                            "data": "safe",
                        },
                    }
                ],
            },
        ),
    ]

    for name, transaction in transactions:
        asset_server.runtime_overrides["/static/canvas_runtime_worker.js"] = (
            _transaction_worker(transaction)
        )
        context, page, recorder = _new_page(
            chromium_browser, asset_server, egress_server
        )
        try:
            status = _load(page, plan, recorder)
            assert status["state"] == "failed", name
            assert status["code"] == "invalid-patch", name
            frame = page.frame(name="canvas-renderer")
            assert frame is not None
            assert frame.locator("#count").text_content() == "0", name
            assert frame.locator("#count").get_attribute("style") in {None, ""}, name
            assert (
                frame.locator("#count").evaluate("node => getComputedStyle(node).color")
                == "rgb(12, 34, 56)"
            ), name
            assert frame.evaluate("document.adoptedStyleSheets.length") == 1, name
            image_url = frame.locator("#pixel").get_attribute("src")
            assert isinstance(image_url, str) and image_url.startswith("blob:null/"), (
                name
            )
            assert frame.locator("#pixel").evaluate(
                "image => [image.naturalWidth, image.naturalHeight]"
            ) == [1, 1], name
            assert [
                item for item in recorder.generated() if item.kind == "request"
            ] == [], name
            assert frame.evaluate(
                "async (url) => { const image = new Image(); image.src = url; "
                "try { await image.decode(); return [image.naturalWidth, image.naturalHeight]; } "
                "catch (_) { return null; } }",
                image_url,
            ) == [1, 1], name
            assert (
                page.evaluate(
                    "window.__canvasHarness.messages.filter((item) => "
                    "item.type === 'canvas:bridge-request').length"
                )
                == 0
            ), name
            assert not any(
                item.kind == "request" and item.target.startswith(egress_server.origin)
                for item in recorder.observations
            ), name
            assert egress_server.requests == [], name
        finally:
            context.close()


@pytest.mark.loopback_network
def test_rapid_form_events_preserve_live_control_state_identity_and_fifo_values(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    source = """<!doctype html>
<html><head><meta charset="utf-8"><title>rapid form events</title></head>
<body>
  <form id="rapid-form">
    <label for="answer">Answer</label>
    <input id="answer" name="answer" value="seed">
    <button type="submit">Submit</button>
  </form>
  <output id="status">idle</output>
  <script>
    const answer = document.getElementById('answer');
    const form = document.getElementById('rapid-form');
    const status = document.getElementById('status');
    const trace = [];
    answer.addEventListener('input', () => trace.push('input:' + answer.value));
    answer.addEventListener('change', () => trace.push('change:' + answer.value));
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      canvas.submit({answer: answer.value, trace: trace.slice(-2)});
      status.textContent = answer.value;
    });
  </script>
</body></html>"""
    values = ["alpha-one", "beta-two", "gamma-three"]
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        assert _load(page, _wire_plan(source), recorder)["state"] == "ready"
        frame = page.frame(name="canvas-renderer")
        assert frame is not None
        answer = frame.locator("#answer").element_handle()
        assert answer is not None

        for index, value in enumerate(values, start=1):
            answer.evaluate(
                """(node, nextValue) => {
                  node.focus();
                  node.value = nextValue;
                  node.setSelectionRange(nextValue.length, nextValue.length);
                  node.dispatchEvent(new Event('input', {bubbles: true}));
                  node.dispatchEvent(new Event('change', {bubbles: true}));
                  node.form.querySelector('button').click();
                }""",
                value,
            )
            page.wait_for_function(
                "expected => window.__canvasHarness.messages.filter((item) => "
                "item.type === 'canvas:bridge-request' && item.kind === 'submit').length "
                "=== expected",
                arg=index,
                timeout=2_000,
            )
            assert answer.evaluate(
                """(node) => ({
                  connected: node.isConnected,
                  current: document.getElementById('answer') === node,
                  focused: document.activeElement === node,
                  value: node.value,
                  selectionStart: node.selectionStart,
                  selectionEnd: node.selectionEnd,
                })"""
            ) == {
                "connected": True,
                "current": True,
                "focused": True,
                "value": value,
                "selectionStart": len(value),
                "selectionEnd": len(value),
            }

        submitted = page.evaluate(
            "window.__canvasHarness.messages.filter((item) => "
            "item.type === 'canvas:bridge-request' && item.kind === 'submit')"
            ".map((item) => item.value)"
        )
        assert submitted == [
            {
                "answer": value,
                "trace": [f"input:{value}", f"change:{value}"],
            }
            for value in values
        ]
        assert frame.locator("#status").text_content() == values[-1]
        assert page.evaluate("window.__canvasHarness.status.state") == "ready"
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    ("event_kind", "expected_length"),
    [("input", "8192"), ("keydown", "32")],
)
def test_renderer_caps_event_fields_by_utf8_bytes(
    event_kind: str,
    expected_length: str,
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    source = (
        '<!doctype html><html><head><meta charset="utf-8"></head><body>'
        '<input id="payload"><output id="status">static</output><script>'
        "const payload = document.getElementById('payload');"
        "const status = document.getElementById('status');"
        f"payload.addEventListener('{event_kind}', (event) => {{"
        "status.textContent = String("
        + ("payload.value.length" if event_kind == "input" else "event.key.length")
        + "); });"
        "</script></body></html>"
    )
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        assert _load(page, _wire_plan(source), recorder)["state"] == "ready"
        frame = page.frame(name="canvas-renderer")
        assert frame is not None
        if event_kind == "input":
            frame.locator("#payload").evaluate(
                "(node, value) => { node.value = value; "
                "node.dispatchEvent(new Event('input', {bubbles: true})); }",
                chr(0x1F600) * 5_000,
            )
        else:
            frame.locator("#payload").evaluate(
                "node => node.dispatchEvent(new KeyboardEvent('keydown', "
                "{key: String.fromCodePoint(0x1F600).repeat(20), bubbles: true}))"
            )
        try:
            frame.wait_for_function(
                "expected => document.getElementById('status').textContent === expected",
                arg=expected_length,
                timeout=2_000,
            )
        except Exception as exc:
            actual = frame.locator("#status").text_content()
            status = page.evaluate("window.__canvasHarness.status")
            messages = page.evaluate("window.__canvasHarness.messages")
            raise AssertionError(
                f"event={event_kind!r}, text={actual!r}, "
                f"status={status!r}, messages={messages!r}"
            ) from exc
        assert page.evaluate("window.__canvasHarness.status.state") == "ready"
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.loopback_network
def test_optional_browser_engine_runs_the_worker_boundary_when_installed(
    optional_browser: tuple[str, Any],
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    name, browser = optional_browser
    context, page, recorder = _new_page(browser, asset_server, egress_server)
    try:
        status = _load(
            page,
            _wire_plan(
                _attack_source(
                    f"try {{ fetch('{egress_server.origin}/optional'); }} catch (_) {{}} "
                    "document.getElementById('status').textContent = 'isolated';"
                )
            ),
            recorder,
        )
        assert status["state"] == "ready", name
        assert status["native_worker_sentinel"] == "native-worker-clean"
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()
