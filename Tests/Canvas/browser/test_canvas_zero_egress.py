"""Real-Chromium proof for the Canvas V1 worker/renderer security boundary."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from dataclasses import asdict, dataclass
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
from threading import Lock, Thread
import time
from typing import Any

import pytest

from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.models import RenderNode


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


def _shell_html() -> bytes:
    return b"""<!doctype html>
<meta charset="utf-8">
<title>Owned Canvas security harness</title>
<script>
(() => {
  'use strict';
  const state = {
    rendererReady: false,
    messages: [],
    status: null,
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
        await window.__canvasMarkExecution();
        channel.port1.postMessage({type: 'canvas:execution-ack', nonce: state.nonce});
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
    ).encode("utf-8")


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

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
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
            self._send(200, _shell_html(), "text/html; charset=utf-8")
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
            if not path.is_file():
                self._send(404, b"missing trusted runtime asset", "text/plain")
                return
            self._send(
                200,
                path.read_bytes(),
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

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
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
    def __init__(self) -> None:
        self._phase = "startup"
        self._lock = Lock()
        self.observations: list[BrowserObservation] = []

    def mark_execution(self) -> None:
        with self._lock:
            self._phase = "generated"

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


@pytest.fixture(scope="module")
def playwright_runtime() -> Iterator[Any]:
    playwright_module = pytest.importorskip("playwright.sync_api")
    with playwright_module.sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="module")
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
    browser: Any, asset_server: _OwnedServer
) -> tuple[Any, Any, BrowserRecorder]:
    context = browser.new_context(accept_downloads=True)
    context.add_init_script(
        "window.__canvasNativeWindowSentinel = 'native-frame-clean';"
        "Object.prototype.__canvasNativePrototypeSentinel = 'native-frame-clean';"
    )
    recorder = BrowserRecorder()
    page = context.new_page()
    page.expose_function("__canvasMarkExecution", recorder.mark_execution)
    context.on(
        "request",
        lambda request: recorder.add("request", request.url, request.resource_type),
    )
    context.on(
        "response",
        lambda response: recorder.add(
            "response", response.url, json.dumps(response.headers, sort_keys=True)
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
    page.evaluate("plan => window.loadCanvas(plan)", plan)
    try:
        page.wait_for_function(
            "window.__canvasHarness.status && "
            "['ready', 'failed'].includes(window.__canvasHarness.status.state)",
            timeout=10_000,
        )
    except Exception as exc:
        messages = page.evaluate("window.__canvasHarness.messages")
        observations = recorder.observations if recorder is not None else []
        raise AssertionError(
            "Canvas runtime produced no terminal status; "
            f"messages={messages!r}; observations={observations!r}"
        ) from exc
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
    context, page, recorder = _new_page(chromium_browser, asset_server)
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
        context, page, recorder = _new_page(chromium_browser, asset_server)
        try:
            script = case["script"].replace("__EGRESS__", egress_server.origin)
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
            elif case["expected"] == "failed-after-three-clicks":
                assert status["state"] == "ready", case["name"]
                for _ in range(3):
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
def test_opaque_iframe_and_csp_block_native_parent_storage_requests_forms_and_images(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    context, page, recorder = _new_page(chromium_browser, asset_server)
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
        renderer_headers = json.loads(renderer_response.detail)
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
def test_window_message_spoof_cannot_forge_worker_status_or_bridge(
    chromium_browser: Any,
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    context, page, recorder = _new_page(chromium_browser, asset_server)
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

    for name, plan in plans:
        context, page, recorder = _new_page(chromium_browser, asset_server)
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
            assert egress_server.requests == [], name
        finally:
            context.close()


@pytest.mark.loopback_network
def test_optional_browser_engine_runs_the_worker_boundary_when_installed(
    optional_browser: tuple[str, Any],
    asset_server: _OwnedServer,
    egress_server: _OwnedServer,
) -> None:
    name, browser = optional_browser
    context, page, recorder = _new_page(browser, asset_server)
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
