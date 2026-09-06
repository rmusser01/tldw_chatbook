"""Reusable outer-path helpers for Canvas browser verification."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit

import pytest
from aiohttp import ClientSession, WSMsgType, web

from tldw_chatbook import config as config_module
from tldw_chatbook.Web_Server import serve

FIXTURES = Path(__file__).with_name("fixtures")

_PROVIDER_SECRET_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "APHRODITE_API_KEY",
    "COHERE_API_KEY",
    "CUSTOM_API_KEY",
    "CUSTOM_2_API_KEY",
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "ELEVENLABS_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "HUGGINGFACE_API_KEY",
    "LLAMA_CPP_API_KEY",
    "MISTRAL_API_KEY",
    "MOONSHOT_API_KEY",
    "OOBABOOGA_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "REMOTE_WHISPER_API_KEY",
    "TABBYAPI_API_KEY",
    "VLLM_API_KEY",
    "ZAI_API_KEY",
)


def chromium_executable(browser_type: object) -> str:
    """Resolve the mandatory Chromium binary without downloading anything."""

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
        pytest.fail("real Playwright Chromium is required for Canvas verification")
    return executable


def reserve_loopback_port() -> int:
    """Return one currently free numeric-loopback TCP port."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass(frozen=True)
class ProbeRequest:
    method: str
    path: str


@dataclass(frozen=True)
class RouteObservation:
    """One bounded browser-side observation labeled by execution phase."""

    phase: str
    kind: str
    target: str
    detail: str = ""


class ProductRouteRecorder:
    """Record all browser escape surfaces around a trusted Canvas route."""

    def __init__(self, *, limit: int = 4_096, served: bool = False) -> None:
        self.phase = "bootstrap"
        self.execution_ack_count = 0
        self.limit = limit
        self.overflow = False
        self.observations: list[RouteObservation] = []
        self._request_context: dict[int, tuple[str, str]] = {}
        self.served = served
        self._load_start = 0
        self.startup_error: str | None = None

    @staticmethod
    def _worker_bootstrap(module_url: str) -> str:
        # Independently fixed transport program; never derived from an observed
        # worker URL or source supplied by a generated document.
        program = (
            f"import({json.dumps(module_url)}).then((module) => {{ "
            'module.startCanvasRuntimeWorker(globalThis); postMessage({type: "bootstrap-ready"}); '
            '}).catch((error) => postMessage({type: "bootstrap-failure", name: '
            'String(error && error.name || "Error")}));'
        )
        return (
            "data:text/javascript;base64," + base64.b64encode(program.encode()).decode()
        )

    def assert_startup_complete(self, shell_url: str) -> None:
        """Require the fixed, completed zero-asset corpus load before guest entry."""
        assert not self.overflow, "recorder overflow"
        parsed = urlsplit(shell_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        root = parsed.path.rstrip("/")
        static = "/static/chatbook-canvas" if self.served else root + "/static"
        expected = {
            origin + root + "/render": ("GET", "document"),
            **{
                origin + static + "/" + name: ("GET", "script")
                for name in (
                    "canvas_renderer.js",
                    "canvas_runtime_worker.js",
                    "quickjs-runtime.js",
                )
            },
        }
        rows = self.observations[self._load_start :]
        # These corpus documents contain no passive assets. A future fixture
        # with images must supply an independently plan-derived image census.
        requests = [
            row
            for row in rows
            if row.kind == "request"
            and (
                row.target in expected
                or json.loads(row.detail)["owner"] == "<worker>"
                or urlsplit(json.loads(row.detail)["owner"]).path == root + "/render"
            )
        ]
        assert sorted(
            (
                row.target,
                json.loads(row.detail)["method"],
                json.loads(row.detail)["resource_type"],
            )
            for row in requests
        ) == sorted((url, *rule) for url, rule in expected.items()), (
            "renderer startup request census"
        )
        for kind in ("response", "request-finished"):
            actual = [
                row for row in rows if row.kind == kind and row.target in expected
            ]
            assert sorted(row.target for row in actual) == sorted(expected), (
                "renderer startup completion census"
            )
            if kind == "response":
                assert all(json.loads(row.detail)["status"] == 200 for row in actual), (
                    "renderer startup HTTP failure"
                )
        workers = [row.target for row in rows if row.kind == "worker"]
        assert workers == [
            self._worker_bootstrap(origin + static + "/canvas_runtime_worker.js")
        ], "renderer startup worker census"
        navigations = [row for row in rows if row.kind == "navigation"]
        assert [(row.target, row.detail) for row in navigations] == [
            (origin + root + "/render", "frame")
        ], "renderer startup navigation census"
        assert not any(
            row.kind in {"popup", "download", "websocket", "request-failed"}
            for row in rows
        ), "renderer startup escape surface"

    def add(
        self,
        kind: str,
        target: str,
        detail: str = "",
        *,
        phase: str | None = None,
    ) -> None:
        if len(self.observations) >= self.limit:
            self.overflow = True
            return
        self.observations.append(
            RouteObservation(phase or self.phase, kind, target, detail)
        )

    @staticmethod
    def _owner_url(request) -> str:
        try:
            return request.frame.url
        except Exception:  # noqa: BLE001 - worker requests have no Frame
            return "<worker>"

    def _request(self, request) -> None:
        if request.resource_type == "document" and urlsplit(request.url).path.endswith(
            "/render"
        ):
            self.phase = "trusted-load"
            self._load_start = len(self.observations)
        owner = self._owner_url(request)
        phase = self.phase
        if len(self._request_context) >= self.limit:
            self.overflow = True
            return
        self._request_context[id(request)] = (phase, owner)
        self.add(
            "request",
            request.url,
            json.dumps(
                {
                    "method": request.method,
                    "owner": owner,
                    "resource_type": request.resource_type,
                },
                sort_keys=True,
            ),
            phase=phase,
        )

    def _request_completion(
        self, kind: str, request, status: int | None = None
    ) -> None:
        phase, owner = self._request_context.get(
            id(request), (self.phase, self._owner_url(request))
        )
        self.add(
            kind,
            request.url,
            json.dumps({"owner": owner, "status": status}),
            phase=phase,
        )
        if kind in {"request-finished", "request-failed"}:
            self._request_context.pop(id(request), None)

    def attach(self, context, page) -> None:
        context.on("request", self._request)
        context.on(
            "response",
            lambda response: self._request_completion(
                "response", response.request, response.status
            ),
        )
        context.on(
            "requestfinished",
            lambda request: self._request_completion("request-finished", request),
        )
        context.on(
            "requestfailed",
            lambda request: self._request_completion("request-failed", request),
        )
        page.on("websocket", lambda socket: self.add("websocket", socket.url))
        page.on("worker", lambda worker: self.add("worker", worker.url))
        page.on("popup", lambda popup: self.add("popup", popup.url))
        page.on(
            "download",
            lambda download: self.add("download", download.suggested_filename),
        )
        page.on(
            "framenavigated",
            lambda frame: self.add(
                "navigation", frame.url, "top" if frame == page.main_frame else "frame"
            ),
        )

    async def install_execution_boundary(self, context) -> None:
        """Hold each trusted execution ack until its load phase is recorded."""

        async def approve(source):
            # Request-finished notifications can trail the worker's startup
            # message; hold the real ack until the independent census closes.
            for _ in range(200):
                try:
                    self.assert_startup_complete(source["frame"].url)
                except AssertionError as error:
                    failure = str(error)
                    await asyncio.sleep(0.01)
                else:
                    self.mark_generated()
                    return True
            self.startup_error = failure
            return False

        await context.expose_binding("__canvasRouteExecutionAck", approve)
        await context.add_init_script(
            """
            (() => {
              const original = MessagePort.prototype.postMessage;
              MessagePort.prototype.postMessage = function(message, ...rest) {
                if (message && message.type === 'canvas:status') {
                  globalThis.__canvasRouteRuntimeStatus = {
                    state: message.state, code: message.code ?? null,
                  };
                }
                if (message && message.type === 'canvas:execution-ack') {
                  Promise.resolve(globalThis.__canvasRouteExecutionAck()).then((approved) => {
                    if (approved) Reflect.apply(original, this, [message, ...rest]);
                  });
                  return;
                }
                return Reflect.apply(original, this, [message, ...rest]);
              };
            })();
            """
        )

    def begin_load(self) -> None:
        # Keep the previous renderer in generated phase until navigation
        # actually starts; a test action is not permission for guest traffic.
        if self.phase == "bootstrap":
            self.phase = "trusted-load"

    def mark_generated(self) -> None:
        self.execution_ack_count += 1
        self.phase = "generated"

    def assert_generated_confined(
        self,
        *,
        trusted_origin: str,
        trusted_route_root: str,
        forbidden_canary_path: str,
        trusted_shell_paths: tuple[tuple[str, str, str], ...] = (),
        trusted_static_paths: tuple[str, ...] = (),
    ) -> None:
        assert self.overflow is False
        assert self.startup_error is None, self.startup_error
        root = trusted_route_root.rstrip("/")
        route_rules = {
            root + suffix: rule
            for suffix, rule in {
                "/": ("GET", "document"),
                "/render": ("GET", "document"),
                "/api/boot": ("POST", "fetch"),
                "/api/frame": ("POST", "fetch"),
                "/api/state": ("GET", "fetch"),
                "/api/navigate": ("POST", "fetch"),
                "/api/plan": ("GET", "fetch"),
                "/api/events": ("GET", "fetch"),
                "/api/actions": ("POST", "fetch"),
                "/api/source": ("GET", "fetch"),
                "/api/source-download": ("GET", "fetch"),
                "/api/bridge/prepare": ("POST", "fetch"),
                "/api/bridge": ("POST", "fetch"),
                "/api/close": ("POST", "fetch"),
            }.items()
        }
        route_rules.update(
            {
                path: (method, resource_type)
                for path, method, resource_type in trusted_shell_paths
            }
        )
        static_paths = {
            root + "/static/canvas_renderer.js",
            root + "/static/canvas_runtime_worker.js",
            root + "/static/quickjs-runtime.js",
            *trusted_static_paths,
        }
        for item in self.observations:
            if forbidden_canary_path in item.target:
                raise AssertionError(
                    f"generated same-origin canary escaped: {item.kind}"
                )
            if item.phase == "bootstrap":
                continue
            if item.kind in {"popup", "download", "websocket"}:
                raise AssertionError(f"generated escape surface observed: {item.kind}")
            parsed = urlsplit(item.target)
            if item.kind == "navigation":
                if item.phase != "trusted-load":
                    raise AssertionError("late renderer navigation after execution ack")
                if item.target in {"about:blank", "about:srcdoc"} and item.detail == (
                    "frame"
                ):
                    continue
            if item.kind == "worker":
                module_paths = [
                    path
                    for path in static_paths
                    if path.endswith("/canvas_runtime_worker.js")
                ]
                if item.phase in {"trusted-load", "bootstrap"} and item.target in {
                    self._worker_bootstrap(trusted_origin + path)
                    for path in module_paths
                }:
                    continue
                raise AssertionError("late or foreign worker target")
            if parsed.scheme in {"about", "blob", "data", ""}:
                raise AssertionError(
                    f"unexpected {parsed.scheme or 'relative'} browser event"
                )
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if origin != trusted_origin or parsed.query or parsed.fragment:
                raise AssertionError(
                    f"browser traffic escaped trusted Canvas route: {item.kind}"
                )
            if item.kind == "navigation":
                if parsed.path == root + "/render" and item.detail == "frame":
                    continue
                raise AssertionError("unexpected trusted-route navigation")
            if item.kind in {"response", "request-finished"}:
                if parsed.path in route_rules or parsed.path in static_paths:
                    continue
                raise AssertionError("completion for an untrusted route")
            if item.kind != "request":
                continue
            detail = json.loads(item.detail)
            rule = route_rules.get(parsed.path)
            if parsed.path in static_paths:
                rule = ("GET", "script")
            if rule != (detail["method"], detail["resource_type"]):
                raise AssertionError("request violated exact trusted route contract")
            owner = urlsplit(detail["owner"])
            owner_is_shell = owner.path in {root + "/", "/"}
            owner_is_renderer = owner.path == root + "/render" or detail["owner"] in {
                "about:srcdoc",
                "about:blank",
                "<worker>",
            }
            if owner_is_renderer and item.phase != "trusted-load":
                raise AssertionError("renderer requested after execution ack")
            if not owner_is_shell and not owner_is_renderer:
                raise AssertionError("request owner is not a trusted Canvas realm")


class EgressProbe(ThreadingHTTPServer):
    """Owned listener that records every attempted generated request."""

    daemon_threads = True

    def __init__(self) -> None:
        self.requests: list[ProbeRequest] = []
        self.lock = Lock()
        super().__init__(("127.0.0.1", 0), _EgressHandler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _EgressHandler(BaseHTTPRequestHandler):
    server: EgressProbe

    def _record(self) -> None:
        with self.server.lock:
            self.server.requests.append(ProbeRequest(self.command, self.path))
        self.send_response(204)
        self.end_headers()

    do_GET = _record
    do_POST = _record

    def log_message(self, _format: str, *args: object) -> None:
        return None


@contextmanager
def egress_probe() -> Iterator[EgressProbe]:
    """Run and reliably close one owned zero-egress boundary listener."""

    server = EgressProbe()
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def adversarial_cases(probe_origin: str) -> list[dict[str, object]]:
    """Load the canonical corpus with only the owned probe destinations filled."""

    cases = json.loads(
        (FIXTURES / "adversarial_scripts.json").read_text(encoding="utf-8")
    )
    websocket = probe_origin.replace("http://", "ws://", 1)
    return [
        {
            **case,
            "script": case["script"]
            .replace("__EGRESS__", probe_origin)
            .replace("__WEBSOCKET__", websocket),
        }
        for case in cases
    ]


def attack_source(script: str, *, marker: str) -> str:
    """Wrap one canonical attack in an otherwise inert self-contained document."""

    clobbering_ids = "".join(
        f'<span id="{name}">{name}</span>'
        for name in ("fetch", "location", "parent", "postMessage", "Worker")
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>attack</title></head>"
        f"<body><h1>{marker}</h1>{clobbering_ids}"
        '<button id="attack" type="button">attack</button>'
        f'<output id="status">static</output><script>{script}</script></body></html>'
    )


async def exercise_adversarial_preview(
    shell,
    preview,
    case: dict[str, object],
    *,
    expect_bridge: bool = True,
    programmatic_clicks: bool = False,
) -> None:
    """Drive one canonical case through a trusted product shell."""

    expected = str(case["expected"])
    loading = shell.locator("#loading-state")
    compatibility = shell.locator("#compatibility")
    attack = preview.locator("#attack")
    if expected in {
        "ready",
        "failed-after-click",
        "failed-after-clicks",
        "failed-after-event-storm",
    }:
        await loading.wait_for(state="hidden", timeout=10_000)
    if expected == "failed-after-click":
        if programmatic_clicks:
            await attack.evaluate("node => node.click()")
        else:
            await attack.click(no_wait_after=True)
    elif expected == "failed-after-clicks":
        if programmatic_clicks:
            await attack.evaluate(
                "(node, count) => { for (let index = 0; index < count; "
                "index += 1) node.click(); }",
                int(case["click_count"]),
            )
        else:
            for _ in range(int(case["click_count"])):
                await attack.click(no_wait_after=True)
    elif expected == "failed-after-event-storm":
        await attack.evaluate(
            "node => { for (let index = 0; index < 102; index += 1) node.click(); }"
        )
    if expected.startswith("failed"):
        await compatibility.wait_for(state="visible", timeout=10_000)
        assert await shell.locator("#compatibility-title").text_content() == (
            "Preview issue"
        )
        status = await preview.locator(":root").evaluate(
            "() => globalThis.__canvasRouteRuntimeStatus"
        )
        assert status is not None and status.get("state") == "failed"
        assert status.get("code") == case["expected_code"], "runtime failure code"
    settle = case.get("settle_milliseconds")
    if type(settle) is int:
        await asyncio.sleep(settle / 1000)
    expected_text = case.get("expected_text")
    if isinstance(expected_text, str):
        assert await preview.locator("#status").text_content() == expected_text
    bridge_kind = case.get("bridge_kind")
    dialog = shell.locator("#bridge-dialog")
    if isinstance(bridge_kind, str) and expect_bridge:
        await dialog.wait_for(state="visible", timeout=10_000)
        await shell.get_by_role("button", name="Cancel").click()
    elif case.get("expected_bridge_kinds") == []:
        assert await dialog.is_hidden()


def assert_only_owned_browser_traffic(
    urls: list[str], *, owned_origins: tuple[str, ...]
) -> None:
    """Fail if Chromium attempted traffic outside explicit harness origins."""

    unexpected = []
    for value in urls:
        parsed = urlsplit(value)
        if parsed.scheme in {"about", "blob", "data"}:
            continue
        scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
        origin = f"{scheme}://{parsed.netloc}"
        if origin not in owned_origins:
            unexpected.append(value)
    assert unexpected == []


def generate_tls_material(root: Path) -> tuple[Path, Path]:
    """Generate a disposable localhost certificate without retaining its key."""

    executable = shutil.which("openssl")
    if executable is None:
        pytest.fail("openssl is required for the served TLS verification")
    certificate = root / "served-cert.pem"
    private_key = root / "served-key.pem"
    subprocess.run(
        [
            executable,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return certificate, private_key


@dataclass
class LiveServedStack:
    """Production server routes plus real textual-serve AppService children."""

    server: object
    runner: web.AppRunner
    origin: str
    port: int
    owned_paths: tuple[Path, ...]
    services: list[object]
    proxy_runner: web.AppRunner | None = None
    proxy_session: ClientSession | None = None
    proxy_counts: dict[str, int] | None = None

    async def aclose(self) -> None:
        async def reap(service) -> None:
            process = service._process
            if process is not None:
                if process.returncode is None:
                    process.kill()
                await asyncio.wait_for(process.wait(), 5)

        async with AsyncExitStack() as fallback:
            # Every owned resource gets an independent fallback, even when an
            # earlier cleanup raises. ExitStack propagates failures, not success.
            for path in self.owned_paths:
                if path.is_dir():
                    fallback.callback(shutil.rmtree, path)
                else:
                    fallback.callback(path.unlink, missing_ok=True)
            if self.proxy_session is not None:
                fallback.push_async_callback(self.proxy_session.close)
            if self.proxy_runner is not None:
                fallback.push_async_callback(
                    lambda: asyncio.wait_for(self.proxy_runner.cleanup(), 5)
                )
            for service in self.services:
                fallback.push_async_callback(reap, service)
            cleanup = asyncio.create_task(self.runner.cleanup())
            done, _ = await asyncio.wait({cleanup}, timeout=5)
            if not done:
                # The test command uses exec: these are the exact owned child
                # PIDs, not a shell whose descendants could outlive cleanup.
                for service in self.services:
                    process = service._process
                    if process is not None and process.returncode is None:
                        process.kill()
                await asyncio.wait_for(cleanup, 5)
            else:
                await cleanup


async def start_live_served_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    access_token: str,
    child_module: str = "Tests.Canvas.browser.canvas_live_child",
    trusted_proxy: bool = False,
) -> LiveServedStack:
    """Start production create_server over direct TLS with a deterministic child."""

    async with AsyncExitStack() as rollback:
        stack = await _construct_live_served_stack(
            tmp_path,
            monkeypatch,
            access_token=access_token,
            child_module=child_module,
            trusted_proxy=trusted_proxy,
            rollback=rollback,
        )
        rollback.pop_all()
        return stack


async def _construct_live_served_stack(
    tmp_path, monkeypatch, *, access_token, child_module, trusted_proxy, rollback
) -> LiveServedStack:

    if (
        re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", child_module
        )
        is None
    ):
        raise ValueError("child_module must be an importable dotted module name")
    for name in _PROVIDER_SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    child_temp = tmp_path / "test_data" / "child-tmp"
    child_temp.mkdir(parents=True, exist_ok=True)
    rollback.callback(shutil.rmtree, tmp_path / "test_data")
    monkeypatch.setenv("TMPDIR", str(child_temp))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    port = reserve_loopback_port()
    rollback.callback((tmp_path / "served-cert.pem").unlink, missing_ok=True)
    rollback.callback((tmp_path / "served-key.pem").unlink, missing_ok=True)
    certificate, private_key = generate_tls_material(tmp_path)
    public_url = f"https://127.0.0.1:{port}"
    monkeypatch.setattr(
        serve,
        "get_canvas_config_policy",
        lambda **kwargs: config_module.build_canvas_config_policy(
            {}, web_auth_policy=kwargs.get("web_auth_policy")
        ),
    )
    server = serve.create_server(
        host="0.0.0.0",
        port=port,
        title="Chatbook live verification",
        public_url=public_url,
        access_token=access_token,
        tls_certificate=None if trusted_proxy else str(certificate),
        tls_private_key=None if trusted_proxy else str(private_key),
        trusted_proxy_addresses=["127.0.0.1"] if trusted_proxy else [],
    )
    server.command = f"exec {sys.executable} -u -m {child_module}"
    app = await server._make_app()
    services: list[object] = []
    service_class = server._chatbook_app_service_class

    def record_service(*args, **kwargs):
        service = service_class(*args, **kwargs)
        services.append(service)
        return service

    server._chatbook_app_service_class = record_service
    runner = web.AppRunner(app)
    rollback.push_async_callback(runner.cleanup)
    await runner.setup()
    backend_port = reserve_loopback_port() if trusted_proxy else port
    site = web.TCPSite(
        runner,
        "127.0.0.1",
        backend_port,
        ssl_context=server._web_ssl_context,
    )
    await site.start()
    proxy_runner = None
    proxy_session = None
    proxy_counts = {"http": 0, "websocket": 0}
    if trusted_proxy:
        proxy_session = ClientSession(auto_decompress=False)
        rollback.push_async_callback(proxy_session.close)
        upstream = f"http://127.0.0.1:{backend_port}"

        async def forward(request):
            headers = {
                key: value
                for key, value in request.headers.items()
                if key.lower()
                not in {
                    "host",
                    "connection",
                    "upgrade",
                    "transfer-encoding",
                    "content-length",
                    "x-forwarded-for",
                    "x-forwarded-host",
                    "x-forwarded-proto",
                    "forwarded",
                    "sec-websocket-key",
                    "sec-websocket-version",
                    "sec-websocket-extensions",
                    "sec-websocket-protocol",
                }
            }
            headers.update(
                {
                    "Host": f"127.0.0.1:{port}",
                    "X-Forwarded-Host": f"127.0.0.1:{port}",
                    "X-Forwarded-Proto": "https",
                    "X-Forwarded-For": "127.0.0.1",
                }
            )
            url = upstream + request.raw_path
            if request.headers.get("Upgrade", "").lower() == "websocket":
                proxy_counts["websocket"] += 1
                async with proxy_session.ws_connect(
                    url,
                    headers=headers,
                    protocols=tuple(
                        item.strip()
                        for item in request.headers.get(
                            "Sec-WebSocket-Protocol", ""
                        ).split(",")
                        if item.strip()
                    ),
                ) as backend:
                    browser = web.WebSocketResponse(
                        protocols=(serve.WEBSOCKET_PROTOCOL,)
                    )
                    await browser.prepare(request)

                    async def relay(source, target):
                        async for message in source:
                            if message.type == WSMsgType.TEXT:
                                await target.send_str(message.data)
                            elif message.type == WSMsgType.BINARY:
                                await target.send_bytes(message.data)
                        await target.close()

                    tasks = {
                        asyncio.create_task(relay(browser, backend)),
                        asyncio.create_task(relay(backend, browser)),
                    }
                    try:
                        done, pending = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*done)
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    return browser
            proxy_counts["http"] += 1
            async with proxy_session.request(
                request.method,
                url,
                headers=headers,
                data=await request.read(),
                allow_redirects=False,
            ) as response:
                outgoing = response.headers.copy()
                for header in ("Transfer-Encoding", "Connection", "Content-Length"):
                    outgoing.popall(header, None)
                return web.Response(
                    status=response.status, headers=outgoing, body=await response.read()
                )

        proxy_app = web.Application(client_max_size=16 * 1024 * 1024)
        proxy_app.router.add_route("*", "/{path:.*}", forward)
        proxy_runner = web.AppRunner(proxy_app, shutdown_timeout=3)
        rollback.push_async_callback(proxy_runner.cleanup)
        await proxy_runner.setup()
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificate, private_key)
        await web.TCPSite(proxy_runner, "127.0.0.1", port, ssl_context=tls).start()
    return LiveServedStack(
        server,
        runner,
        public_url,
        port,
        (certificate, private_key, tmp_path / "test_data"),
        services,
        proxy_runner,
        proxy_session,
        proxy_counts,
    )
