#!/usr/bin/env python3
"""Measure Canvas V1 quotas with content-free synthetic fixtures.

This qualification tool never calls an LLM provider and never records source,
runtime messages, generated values, credentials, user data, or tokens. Its
fixtures are deterministic and agent-authored in this file; output contains
only fixture identifiers, counts, timings, statuses, and process-tree RSS.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from tldw_chatbook.Canvas.compiler import CanvasCompileError, compile_canvas_document
from tldw_chatbook.Canvas.limits import CanvasLimits
from tldw_chatbook.Canvas.models import CanvasRenderPlan, RenderNode
from tldw_chatbook.Canvas.runtime_assets import load_canvas_runtime_assets

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "tldw_chatbook" / "Canvas" / "static"
FIXTURE_PROVENANCE = "synthetic-agent-authored"
SUMMARY_SCHEMA_VERSION = 1


class ProbeError(RuntimeError):
    """Raised when qualification cannot produce trustworthy bounded evidence."""


@dataclass(frozen=True, slots=True)
class ProbeFixture:
    """One deterministic synthetic source retained only in process memory."""

    identifier: str
    category: str
    source: str
    expected: str


@dataclass(frozen=True, slots=True)
class CompilerResult:
    """Content-free compiler measurements for one accepted fixture."""

    fixture_id: str
    category: str
    source_bytes: int
    plan_bytes: int
    plan_nodes: int
    css_rules: int
    script_bytes: int
    median_milliseconds: float
    p95_milliseconds: float
    maximum_milliseconds: float


def percentile(values: Sequence[float], percentage: int) -> float:
    """Return a nearest-rank percentile without inventing measured values."""

    if not values or not 1 <= percentage <= 100:
        raise ProbeError("percentile requires samples and a percentage from 1 to 100")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((percentage / 100) * len(ordered)))
    return ordered[rank - 1]


def count_plan_nodes(root: RenderNode) -> int:
    """Count a render plan iteratively so adversarial depth cannot use Python stack."""

    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(reversed(node.children))
    return count


def _dom_source(node_count: int) -> str:
    if node_count < 4:
        raise ProbeError("DOM fixture requires at least four structural nodes")
    children = "".join(
        f'<span id="n-{index}"></span>' for index in range(node_count - 4)
    )
    return (
        f"<!doctype html><html><head></head><body><main>{children}</main></body></html>"
    )


def _css_source(rule_count: int) -> str:
    rules = "".join(
        f".c-{index}{{color:rgb({index % 251},0,0)}}" for index in range(rule_count)
    )
    return (
        f"<!doctype html><html><head><style>{rules}</style></head><body></body></html>"
    )


def _script_source(byte_count: int) -> str:
    if byte_count < 4:
        raise ProbeError("script fixture requires room for a block comment")
    script = "/*" + ("x" * (byte_count - 4)) + "*/"
    return f"<!doctype html><html><head></head><body><script>{script}</script></body></html>"


def _combined_source(limits: CanvasLimits) -> str:
    rules = "".join(
        f".c-{index}{{color:rgb({index % 251},0,0)}}"
        for index in range(limits.css_rules)
    )
    nodes = "".join(
        f'<span id="n-{index}"></span>' for index in range(limits.dom_nodes - 4)
    )
    script = "/*" + ("x" * (limits.script_bytes - 4)) + "*/"
    return (
        "<!doctype html><html><head><style>"
        + rules
        + "</style></head><body><main>"
        + nodes
        + "</main><script>"
        + script
        + "</script></body></html>"
    )


def _representative_source(card_count: int) -> str:
    cards = "".join(
        f'<section class="card"><h2>Item {index}</h2><button type="button">Open</button></section>'
        for index in range(card_count)
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        "#status{color:rgb(12,34,56)}"
        ".card{padding:8px}"
        "</style></head><body><main>"
        + cards
        + '</main><output id="status">idle</output><script>'
        "const status=document.getElementById('status');"
        "document.querySelectorAll('button').forEach((button)=>button.addEventListener('click',()=>{status.textContent='selected'}));"
        "</script></body></html>"
    )


def build_synthetic_fixtures(limits: CanvasLimits) -> tuple[ProbeFixture, ...]:
    """Build representative and adversarial sources without external content."""

    return (
        ProbeFixture(
            "representative-cards-small",
            "representative",
            _representative_source(24),
            "accepted",
        ),
        ProbeFixture(
            "representative-cards-large",
            "representative",
            _representative_source(120),
            "accepted",
        ),
        ProbeFixture(
            "adversarial-combined-at-limit",
            "adversarial",
            _combined_source(limits),
            "accepted",
        ),
        ProbeFixture(
            "adversarial-dom-at-limit",
            "boundary",
            _dom_source(limits.dom_nodes),
            "accepted",
        ),
        ProbeFixture(
            "adversarial-dom-over-limit",
            "boundary",
            _dom_source(limits.dom_nodes + 1),
            "dom-limit",
        ),
        ProbeFixture(
            "adversarial-css-at-limit",
            "boundary",
            _css_source(limits.css_rules),
            "accepted",
        ),
        ProbeFixture(
            "adversarial-css-over-limit",
            "boundary",
            _css_source(limits.css_rules + 1),
            "css-rule-limit",
        ),
        ProbeFixture(
            "adversarial-script-at-limit",
            "boundary",
            _script_source(limits.script_bytes),
            "accepted",
        ),
        ProbeFixture(
            "adversarial-script-over-limit",
            "boundary",
            _script_source(limits.script_bytes + 1),
            "script-limit",
        ),
    )


def _render_node_wire(node: RenderNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "tag": node.tag,
        "attributes": [list(attribute) for attribute in node.attributes],
        "text": node.text,
        "children": [_render_node_wire(child) for child in node.children],
    }


def _plan_wire(plan: CanvasRenderPlan) -> dict[str, Any]:
    return {
        "runtime_profile": plan.runtime_profile,
        "source_identity": asdict(plan.source_identity),
        "root": _render_node_wire(plan.root),
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


def measure_compiler(fixture: ProbeFixture, samples: int) -> CompilerResult:
    """Measure one accepted fixture after one warm-up compilation."""

    if samples < 1:
        raise ProbeError("compiler sample count must be positive")
    plan = compile_canvas_document(fixture.source)
    timings: list[float] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        plan = compile_canvas_document(fixture.source)
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    plan_wire = _plan_wire(plan)
    plan_bytes = len(
        json.dumps(plan_wire, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return CompilerResult(
        fixture_id=fixture.identifier,
        category=fixture.category,
        source_bytes=len(fixture.source.encode("utf-8")),
        plan_bytes=plan_bytes,
        plan_nodes=count_plan_nodes(plan.root),
        css_rules=len(plan.css_rules),
        script_bytes=sum(len(script.encode("utf-8")) for script in plan.scripts),
        median_milliseconds=round(statistics.median(timings), 3),
        p95_milliseconds=round(percentile(timings, 95), 3),
        maximum_milliseconds=round(max(timings), 3),
    )


def measure_compiler_boundaries(fixtures: Sequence[ProbeFixture]) -> dict[str, str]:
    """Execute rejected fixtures and retain only their bounded failure category."""

    observed: dict[str, str] = {}
    for fixture in fixtures:
        if fixture.expected == "accepted":
            continue
        try:
            compile_canvas_document(fixture.source)
        except CanvasCompileError as exc:
            code = exc.issues[0].code if exc.issues else "compile-error"
        else:
            raise ProbeError(f"{fixture.identifier} unexpectedly compiled")
        if code != fixture.expected:
            raise ProbeError(
                f"{fixture.identifier} returned {code}, expected {fixture.expected}"
            )
        observed[fixture.identifier] = code
    return observed


def _object_block(source: str, marker: str, *, after: int = 0) -> str:
    start = source.find(marker, after)
    if start < 0:
        raise ProbeError(f"runtime mirror marker is missing: {marker}")
    start += len(marker)
    end = source.find("});", start)
    if end < 0:
        raise ProbeError(f"runtime mirror object is unterminated: {marker}")
    return source[start:end]


def _safe_integer_expression(expression: str) -> int:
    parsed = ast.parse(expression.strip(), mode="eval")

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left, right = evaluate(node.left), evaluate(node.right)
            return left + right if isinstance(node.op, ast.Add) else left * right
        raise ProbeError("runtime mirror contains a non-integer expression")

    return evaluate(parsed)


def _field(block: str, name: str) -> int:
    match = re.search(rf"(?:^|\n)\s*{re.escape(name)}:\s*([^,\n]+)", block)
    if match is None:
        raise ProbeError(f"runtime mirror field is missing: {name}")
    return _safe_integer_expression(match.group(1))


def validate_runtime_limit_mirrors(
    *, limits: CanvasLimits, worker_source: str, renderer_source: str
) -> dict[str, int]:
    """Fail if Python, worker, virtual facade, and renderer ceilings drift."""

    worker = _object_block(worker_source, "const LIMITS = Object.freeze({")
    facade_start = worker_source.find("const VIRTUAL_RUNTIME_SOURCE")
    facade = _object_block(
        worker_source, "const MAX = Object.freeze({", after=facade_start
    )
    renderer = _object_block(renderer_source, "const MAX = Object.freeze({")
    observed = {
        "python.html_bytes": limits.html_bytes,
        "python.dom_nodes": limits.dom_nodes,
        "python.css_rules": limits.css_rules,
        "python.script_bytes": limits.script_bytes,
        "python.runtime_memory_bytes": limits.runtime_memory_bytes,
        "python.stack_bytes": limits.stack_bytes,
        "python.startup_milliseconds": limits.startup_milliseconds,
        "python.event_milliseconds": limits.event_milliseconds,
        "python.patches_per_event": limits.patches_per_event,
        "worker.dom_nodes": _field(worker, "domNodes"),
        "worker.script_bytes": _field(worker, "scriptBytes"),
        "worker.runtime_memory_bytes": _field(worker, "runtimeMemoryBytes"),
        "worker.stack_bytes": _field(worker, "stackBytes"),
        "worker.startup_milliseconds": _field(worker, "startupMilliseconds"),
        "worker.event_milliseconds": _field(worker, "eventMilliseconds"),
        "worker.patches_per_event": _field(worker, "patchesPerEvent"),
        "virtual_facade.dom_nodes": _field(facade, "nodes"),
        "virtual_facade.patches_per_event": _field(facade, "patches"),
        "renderer.html_bytes": _field(renderer, "htmlBytes"),
        "renderer.dom_nodes": _field(renderer, "domNodes"),
        "renderer.css_rules": _field(renderer, "cssRules"),
        "renderer.script_bytes": _field(renderer, "scriptBytes"),
        "renderer.patches_per_event": _field(renderer, "patchesPerEvent"),
    }
    expected = {
        "worker.dom_nodes": limits.dom_nodes,
        "worker.script_bytes": limits.script_bytes,
        "worker.runtime_memory_bytes": limits.runtime_memory_bytes,
        "worker.stack_bytes": limits.stack_bytes,
        "worker.startup_milliseconds": limits.startup_milliseconds,
        "worker.event_milliseconds": limits.event_milliseconds,
        "worker.patches_per_event": limits.patches_per_event,
        "virtual_facade.dom_nodes": limits.dom_nodes,
        "virtual_facade.patches_per_event": limits.patches_per_event,
        "renderer.html_bytes": limits.html_bytes,
        "renderer.dom_nodes": limits.dom_nodes,
        "renderer.css_rules": limits.css_rules,
        "renderer.script_bytes": limits.script_bytes,
        "renderer.patches_per_event": limits.patches_per_event,
    }
    for key, wanted in expected.items():
        if observed[key] != wanted:
            raise ProbeError(
                f"{key}={observed[key]} does not match Python limit {wanted}"
            )
    return observed


def build_summary(
    *,
    compiler_results: Sequence[CompilerResult],
    compiler_boundaries: Mapping[str, str],
    browser_results: Mapping[str, Any],
    environment: Mapping[str, Any],
    mirrors: Mapping[str, int],
) -> dict[str, Any]:
    """Build the allowlisted, content-free persisted qualification summary."""

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "fixture_provenance": FIXTURE_PROVENANCE,
        "provider_sampling": "none",
        "environment": dict(environment),
        "limits": dict(mirrors),
        "compiler": {
            "fixtures": [asdict(result) for result in compiler_results],
            "boundaries": dict(compiler_boundaries),
        },
        "browser": dict(browser_results),
    }


_RENDERER_CSP = (
    "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; worker-src data:; "
    "style-src 'unsafe-inline'; img-src blob:; connect-src 'none'; font-src 'none'; "
    "media-src 'none'; object-src 'none'; frame-src 'none'; child-src 'none'; "
    "form-action 'none'; base-uri 'none'; manifest-src 'none'; "
    "frame-ancestors 'self'; sandbox allow-scripts"
)


def _shell_html() -> bytes:
    return b"""<!doctype html><meta charset="utf-8"><title>Canvas quota probe</title>
<script>
(() => {
  "use strict";
  const state = {
    rendererReady: false, messages: [], status: null, nonce: null, port: null,
    loadAt: null, preparedAt: null, statusAt: null,
  };
  window.__canvasQuotaProbe = state;
  window.addEventListener("message", (event) => {
    const frame = document.getElementById("renderer");
    if (event.source !== frame.contentWindow || event.data?.type !== "canvas:renderer-ready") return;
    state.rendererReady = true;
  });
  window.loadCanvasForQuotaProbe = async (plan) => {
    while (!state.rendererReady) await new Promise((resolve) => setTimeout(resolve, 5));
    const frame = document.getElementById("renderer");
    const channel = new MessageChannel();
    state.nonce = crypto.randomUUID();
    state.port = channel.port1;
    state.loadAt = performance.now();
    channel.port1.onmessage = async (event) => {
      const message = event.data;
      state.messages.push(message?.type || "invalid");
      if (message?.type === "canvas:execution-started") {
        state.preparedAt = performance.now();
        channel.port1.postMessage({type: "canvas:execution-ack", nonce: state.nonce});
      }
      if (message?.type === "canvas:status") {
        state.statusAt = performance.now();
        state.status = {
          state: message.state,
          code: message.code,
          scripts_disabled: message.scripts_disabled,
          engine: message.engine,
        };
      }
    };
    channel.port1.start();
    frame.contentWindow.postMessage(
      {type: "canvas:init", nonce: state.nonce, plan}, "*", [channel.port2]
    );
  };
})();
</script>
<iframe id="renderer" name="canvas-renderer" sandbox="allow-scripts" src="/renderer.html"></iframe>
"""


def _renderer_html(renderer_javascript: bytes) -> bytes:
    integrity = base64.b64encode(hashlib.sha384(renderer_javascript).digest()).decode(
        "ascii"
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="referrer" content="no-referrer">'
        f'<script type="module" src="/static/canvas_renderer.js" integrity="sha384-{integrity}" crossorigin="anonymous"></script>'
        '</head><body><div id="canvas-root"></div></body></html>'
    ).encode()


def _engine_probe_html() -> bytes:
    return b"""<!doctype html><meta charset="utf-8"><title>Trusted QuickJS quota probe</title>
<script type="module">
import {newQuickJSWASMModule} from "/static/quickjs-runtime.js";

function disposeResult(result) {
  if (result.error) {
    result.error.dispose();
    return "rejected";
  }
  result.value.dispose();
  return "accepted";
}

function evaluateOutcome(context, source, filename) {
  try {
    return disposeResult(context.evalCode(source, filename));
  } catch (_) {
    return "rejected";
  }
}

function memoryUsed(runtime) {
  const handle = runtime.computeMemoryUsage();
  try {
    return runtime.getSystemContext().dump(handle).memory_used_size;
  } finally {
    handle.dispose();
  }
}

window.runTrustedQuickJSQuotaProbe = async (limits) => {
  const quickJS = await newQuickJSWASMModule();
  const runtime = quickJS.newRuntime();
  runtime.setMemoryLimit(limits.heapBytes);
  runtime.setMaxStackSize(limits.stackBytes);
  let deadline = performance.now() + 1000;
  runtime.setInterruptHandler(() => performance.now() > deadline);
  const context = runtime.newContext();
  const baseline = memoryUsed(runtime);
  const acceptedOutcome = evaluateOutcome(
    context,
    `globalThis.__probeRetained = new Uint8Array(${limits.acceptedBytes});`,
    "trusted-heap-accepted.js",
  );
  const acceptedMemory = memoryUsed(runtime);
  deadline = performance.now() + 1000;
  const oversizedOutcome = evaluateOutcome(
    context,
    `globalThis.__probeOversized = new Uint8Array(${limits.oversizedBytes});`,
    "trusted-heap-oversized.js",
  );
  context.dispose();
  runtime.dispose();

  function acceptsDepth(depth) {
    const candidateRuntime = quickJS.newRuntime();
    candidateRuntime.setMemoryLimit(limits.heapBytes);
    candidateRuntime.setMaxStackSize(limits.stackBytes);
    const candidateContext = candidateRuntime.newContext();
    const outcome = evaluateOutcome(
      candidateContext,
      `function descend(n){if(n>0)descend(n-1)}descend(${depth});`,
      "trusted-stack-depth.js",
    );
    try { candidateContext.dispose(); } catch (_) {}
    try { candidateRuntime.dispose(); } catch (_) {}
    return outcome === "accepted";
  }
  const recursionStarted = performance.now();
  let acceptedDepth = 0;
  let rejectedDepth = 16384;
  if (acceptsDepth(rejectedDepth)) throw new Error("trusted stack probe ceiling was too low");
  while (acceptedDepth + 1 < rejectedDepth) {
    const candidate = Math.floor((acceptedDepth + rejectedDepth) / 2);
    if (acceptsDepth(candidate)) acceptedDepth = candidate;
    else rejectedDepth = candidate;
  }

  return {
    baselineMemoryUsedBytes: baseline,
    acceptedAllocationOutcome: acceptedOutcome,
    acceptedMemoryUsedBytes: acceptedMemory,
    oversizedAllocationOutcome: oversizedOutcome,
    recursionOutcome: "rejected",
    recursionDepth: rejectedDepth,
    maximumAcceptedRecursionDepth: acceptedDepth,
    recursionProbeMilliseconds: performance.now() - recursionStarted,
  };
};
window.__trustedQuickJSQuotaProbeReady = true;
</script>
"""


class _ProbeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, runtime_assets: Mapping[str, bytes]) -> None:
        self.runtime_assets = runtime_assets
        super().__init__(("127.0.0.1", 0), _ProbeHandler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _ProbeHandler(BaseHTTPRequestHandler):
    server: _ProbeServer

    def do_GET(self) -> None:
        if self.path == "/shell.html":
            self._send(200, _shell_html(), "text/html; charset=utf-8")
            return
        if self.path == "/renderer.html":
            self._send(
                200,
                _renderer_html(self.server.runtime_assets["canvas_renderer.js"]),
                "text/html; charset=utf-8",
                {"Content-Security-Policy": _RENDERER_CSP},
            )
            return
        if self.path == "/engine-probe.html":
            self._send(200, _engine_probe_html(), "text/html; charset=utf-8")
            return
        if self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        prefix = "/static/"
        if self.path.startswith(prefix):
            body = self.server.runtime_assets.get(self.path.removeprefix(prefix))
            if body is not None:
                self._send(
                    200,
                    body,
                    "text/javascript; charset=utf-8",
                    {
                        "Access-Control-Allow-Origin": "*",
                        "Cross-Origin-Resource-Policy": "cross-origin",
                    },
                )
                return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


@contextmanager
def _serve_runtime_assets() -> Iterator[_ProbeServer]:
    verified = load_canvas_runtime_assets()
    if not verified.enabled:
        raise ProbeError("packaged Canvas runtime verification failed")
    assert verified.javascript is not None
    assert verified.worker_javascript is not None
    assert verified.renderer_javascript is not None
    server = _ProbeServer(
        {
            "quickjs-runtime.js": verified.javascript,
            "canvas_runtime_worker.js": verified.worker_javascript,
            "canvas_renderer.js": verified.renderer_javascript,
        }
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _chromium_executable(browser_type: Any) -> str:
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
    executable = str(candidates[0]) if candidates else shutil.which("chromium")
    if not executable:
        raise ProbeError("real Playwright Chromium is required for quota qualification")
    return executable


def _process_tree_rss_mib(browser: Any) -> float:
    """Return summed Chromium process RSS; shared macOS pages may be double-counted."""

    session = browser.new_browser_cdp_session()
    try:
        process_info = session.send("SystemInfo.getProcessInfo")["processInfo"]
    finally:
        session.detach()
    pids = sorted({int(item["id"]) for item in process_info})
    if not pids:
        raise ProbeError("Chromium exposed no owned process identifiers")
    completed = subprocess.run(
        ["ps", "-o", "pid=,rss=", "-p", ",".join(str(pid) for pid in pids)],
        check=True,
        capture_output=True,
        text=True,
    )
    rss_kib = sum(
        int(line.split()[1]) for line in completed.stdout.splitlines() if line.split()
    )
    if rss_kib <= 0:
        raise ProbeError("Chromium process-tree RSS was unavailable")
    return round(rss_kib / 1024, 3)


def _new_page(browser: Any, origin: str) -> tuple[Any, Any]:
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{origin}/shell.html", wait_until="load")
    page.wait_for_function(
        "window.__canvasQuotaProbe.rendererReady === true", timeout=5_000
    )
    return context, page


def _load_plan(
    page: Any, plan: CanvasRenderPlan
) -> tuple[dict[str, Any], dict[str, float]]:
    page.evaluate("plan => window.loadCanvasForQuotaProbe(plan)", _plan_wire(plan))
    try:
        page.wait_for_function(
            "window.__canvasQuotaProbe.status && "
            "['ready', 'failed'].includes(window.__canvasQuotaProbe.status.state) && "
            "window.__canvasQuotaProbe.preparedAt !== null && "
            "window.__canvasQuotaProbe.statusAt !== null",
            timeout=10_000,
        )
    except Exception as exc:
        diagnostic = page.evaluate(
            "() => ({status: window.__canvasQuotaProbe.status, "
            "message_types: window.__canvasQuotaProbe.messages, "
            "has_load_clock: window.__canvasQuotaProbe.loadAt !== null, "
            "has_prepare_clock: window.__canvasQuotaProbe.preparedAt !== null, "
            "has_status_clock: window.__canvasQuotaProbe.statusAt !== null})"
        )
        raise ProbeError(
            f"browser runtime clocks did not settle: {diagnostic}"
        ) from exc
    state = page.evaluate(
        "() => ({status: window.__canvasQuotaProbe.status, "
        "loadAt: window.__canvasQuotaProbe.loadAt, "
        "preparedAt: window.__canvasQuotaProbe.preparedAt, "
        "statusAt: window.__canvasQuotaProbe.statusAt})"
    )
    if not all(
        isinstance(state[key], (int, float))
        for key in ("loadAt", "preparedAt", "statusAt")
    ):
        raise ProbeError("browser runtime omitted qualification clocks")
    return state["status"], {
        "trusted_prepare": float(state["preparedAt"] - state["loadAt"]),
        "generated_startup": float(state["statusAt"] - state["preparedAt"]),
    }


def _wait_for_locator(
    locator: Any, *, text: str | None = None, attribute: tuple[str, str] | None = None
) -> float:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if text is not None and locator.text_content() == text:
            return float(locator.evaluate("() => performance.now()"))
        if (
            attribute is not None
            and locator.get_attribute(attribute[0]) == attribute[1]
        ):
            return float(locator.evaluate("() => performance.now()"))
        time.sleep(0.005)
    raise ProbeError("browser event result did not settle")


def _wait_for_failed_status(page: Any, frame: Any) -> float:
    page.wait_for_function(
        "window.__canvasQuotaProbe.status?.state === 'failed'", timeout=5_000
    )
    return float(frame.evaluate("performance.now()"))


def _interactive_source(script: str) -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8"></head><body>'
        '<button id="go" type="button">Run</button><output id="out">idle</output>'
        f"<script>{script}</script></body></html>"
    )


def _runtime_case_sources(limits: CanvasLimits) -> tuple[dict[str, Any], ...]:
    patches_at = "".join(
        f"out.setAttribute('data-p-{index}','x');"
        for index in range(limits.patches_per_event)
    )
    patches_over = patches_at + "out.setAttribute('data-p-over','x');"
    prefix = (
        "const go=document.getElementById('go');"
        "const out=document.getElementById('out');"
    )
    listener = "go.addEventListener('click',()=>{"
    return (
        {
            "fixture_id": "representative-interactive",
            "source": _interactive_source(
                prefix + listener + "out.textContent='selected';});"
            ),
            "event": "text",
            "expected": "selected",
            "patches": 1,
        },
        {
            "fixture_id": "adversarial-startup-timeout",
            "source": _interactive_source("while(true){}"),
        },
        {
            "fixture_id": "adversarial-heap-pressure",
            "source": _interactive_source(
                "const retained=[];while(true){retained.push(new Uint8Array(1024*1024));}"
            ),
        },
        {
            "fixture_id": "adversarial-stack-pressure",
            "source": _interactive_source("function descend(){descend()}descend();"),
        },
        {
            "fixture_id": "adversarial-event-timeout",
            "source": _interactive_source(prefix + listener + "while(true){} });"),
            "event": "failure",
        },
        {
            "fixture_id": "adversarial-patches-at-limit",
            "source": _interactive_source(prefix + listener + patches_at + "});"),
            "event": "attribute",
            "expected": f"data-p-{limits.patches_per_event - 1}",
            "patches": limits.patches_per_event,
        },
        {
            "fixture_id": "adversarial-patches-over-limit",
            "source": _interactive_source(prefix + listener + patches_over + "});"),
            "event": "failure",
            "patches": limits.patches_per_event + 1,
        },
    )


def _run_runtime_case(
    browser: Any, origin: str, case: Mapping[str, Any]
) -> dict[str, Any]:
    context, page = _new_page(browser, origin)
    try:
        plan = compile_canvas_document(str(case["source"]))
        status, timings = _load_plan(page, plan)
        event_milliseconds: float | None = None
        if case.get("event") and status["state"] == "ready":
            frame = page.frame(name="canvas-renderer")
            if frame is None:
                raise ProbeError("renderer frame disappeared during qualification")
            locator = frame.locator("#go")
            event_started = float(
                locator.evaluate(
                    "node => { const now=performance.now(); node.click(); return now; }"
                )
            )
            event_kind = case["event"]
            if event_kind == "text":
                event_finished = _wait_for_locator(
                    frame.locator("#out"), text=str(case["expected"])
                )
            elif event_kind == "attribute":
                event_finished = _wait_for_locator(
                    frame.locator("#out"), attribute=(str(case["expected"]), "x")
                )
            else:
                event_finished = _wait_for_failed_status(page, frame)
            event_milliseconds = event_finished - event_started
            status = page.evaluate("window.__canvasQuotaProbe.status")
        result: dict[str, Any] = {
            "fixture_id": case["fixture_id"],
            "state": status["state"],
            "code": status["code"],
            "scripts_disabled": status["scripts_disabled"],
            "trusted_prepare_milliseconds": round(timings["trusted_prepare"], 3),
            "generated_startup_milliseconds": round(timings["generated_startup"], 3),
        }
        if event_milliseconds is not None:
            result["event_round_trip_milliseconds"] = round(event_milliseconds, 3)
        if "patches" in case:
            result["patches"] = int(case["patches"])
            if event_milliseconds and status["state"] == "ready":
                result["patches_per_second"] = round(
                    int(case["patches"]) / (event_milliseconds / 1000), 3
                )
        return result
    finally:
        context.close()


def _aggregate_runtime_cases(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["fixture_id"]), []).append(run)
    results: list[dict[str, Any]] = []
    for fixture_id, samples in grouped.items():
        states = {str(sample["state"]) for sample in samples}
        codes = {sample["code"] for sample in samples}
        if len(states) != 1 or len(codes) != 1:
            raise ProbeError(f"browser result was not deterministic: {fixture_id}")
        result: dict[str, Any] = {
            "fixture_id": fixture_id,
            "state": states.pop(),
            "code": codes.pop(),
            "scripts_disabled": all(
                bool(sample["scripts_disabled"]) for sample in samples
            ),
        }
        if "patches" in samples[0]:
            result["patches"] = int(samples[0]["patches"])
        for field in (
            "trusted_prepare_milliseconds",
            "generated_startup_milliseconds",
            "event_round_trip_milliseconds",
            "patches_per_second",
        ):
            values = [float(sample[field]) for sample in samples if field in sample]
            if values:
                result[field] = {
                    "median": round(statistics.median(values), 3),
                    "p95": round(percentile(values, 95), 3),
                    "maximum": round(max(values), 3),
                }
        results.append(result)
    return results


def _rss_for_source(browser: Any, origin: str, source: str) -> float:
    context, page = _new_page(browser, origin)
    try:
        status, _timings = _load_plan(page, compile_canvas_document(source))
        if status["state"] != "ready":
            raise ProbeError("memory fixture did not become ready")
        return _process_tree_rss_mib(browser)
    finally:
        context.close()


def _run_trusted_quickjs_probe(
    browser: Any, origin: str, limits: CanvasLimits
) -> dict[str, Any]:
    """Measure QuickJS resources from trusted host code, outside the generated facade."""

    accepted_bytes = 16 * 1024 * 1024
    oversized_bytes = limits.runtime_memory_bytes
    context = browser.new_context()
    page = context.new_page()
    try:
        page.goto(f"{origin}/engine-probe.html", wait_until="load")
        page.wait_for_function(
            "window.__trustedQuickJSQuotaProbeReady === true", timeout=10_000
        )
        observed = page.evaluate(
            "limits => window.runTrustedQuickJSQuotaProbe(limits)",
            {
                "heapBytes": limits.runtime_memory_bytes,
                "stackBytes": limits.stack_bytes,
                "acceptedBytes": accepted_bytes,
                "oversizedBytes": oversized_bytes,
            },
        )
    finally:
        context.close()
    if observed["acceptedAllocationOutcome"] != "accepted":
        raise ProbeError("trusted QuickJS accepted allocation was refused")
    if observed["oversizedAllocationOutcome"] != "rejected":
        raise ProbeError("trusted QuickJS oversized allocation was not refused")
    if observed["recursionOutcome"] != "rejected":
        raise ProbeError("trusted QuickJS stack pressure was not refused")
    return {
        "probe_scope": "trusted-direct-engine; not exposed to generated code",
        "heap_limit_bytes": limits.runtime_memory_bytes,
        "baseline_memory_used_bytes": int(observed["baselineMemoryUsedBytes"]),
        "accepted_allocation_bytes": accepted_bytes,
        "accepted_allocation_outcome": observed["acceptedAllocationOutcome"],
        "accepted_memory_used_bytes": int(observed["acceptedMemoryUsedBytes"]),
        "oversized_allocation_bytes": oversized_bytes,
        "oversized_allocation_outcome": observed["oversizedAllocationOutcome"],
        "stack_limit_bytes": limits.stack_bytes,
        "recursion_outcome": observed["recursionOutcome"],
        "recursion_depth": int(observed["recursionDepth"]),
        "maximum_accepted_recursion_depth": int(
            observed["maximumAcceptedRecursionDepth"]
        ),
        "recursion_probe_milliseconds": round(
            float(observed["recursionProbeMilliseconds"]), 3
        ),
    }


def run_browser_probe(limits: CanvasLimits, *, samples: int) -> dict[str, Any]:
    """Run real Chromium runtime, boundary, and process-tree memory probes."""

    if samples < 1:
        raise ProbeError("browser sample count must be positive")
    try:
        playwright_module = importlib.import_module("playwright.sync_api")
    except ImportError as exc:
        raise ProbeError(
            "Python Playwright is required for quota qualification"
        ) from exc
    fixtures = {item.identifier: item for item in build_synthetic_fixtures(limits)}
    with _serve_runtime_assets() as server:  # noqa: SIM117 - scopes are distinct resources
        with playwright_module.sync_playwright() as playwright:
            executable = _chromium_executable(playwright.chromium)
            browser = playwright.chromium.launch(
                headless=True, executable_path=executable
            )
            try:
                warm_context = browser.new_context()
                warm_page = warm_context.new_page()
                warm_page.goto("about:blank")
                warmed_blank = _process_tree_rss_mib(browser)
                warm_context.close()
                trusted_source = "<!doctype html><html><head></head><body><main></main></body></html>"
                trusted_runtime = _rss_for_source(
                    browser, server.origin, trusted_source
                )
                representative = _rss_for_source(
                    browser,
                    server.origin,
                    fixtures["representative-cards-large"].source,
                )
                near_limit = _rss_for_source(
                    browser,
                    server.origin,
                    fixtures["adversarial-combined-at-limit"].source,
                )
                runs = [
                    _run_runtime_case(browser, server.origin, case)
                    for _ in range(samples)
                    for case in _runtime_case_sources(limits)
                ]
                combined_runs = []
                for _ in range(samples):
                    context, page = _new_page(browser, server.origin)
                    try:
                        status, timings = _load_plan(
                            page,
                            compile_canvas_document(
                                fixtures["adversarial-combined-at-limit"].source
                            ),
                        )
                        combined_runs.append(
                            {
                                "fixture_id": "adversarial-combined-at-limit",
                                "state": status["state"],
                                "code": status["code"],
                                "scripts_disabled": status["scripts_disabled"],
                                "trusted_prepare_milliseconds": timings[
                                    "trusted_prepare"
                                ],
                                "generated_startup_milliseconds": timings[
                                    "generated_startup"
                                ],
                            }
                        )
                    finally:
                        context.close()
                runtime_cases = _aggregate_runtime_cases([*runs, *combined_runs])
                quickjs_resources = _run_trusted_quickjs_probe(
                    browser, server.origin, limits
                )
                return {
                    "browser_engine": "chromium",
                    "browser_version": browser.version,
                    "playwright_version": importlib.metadata.version("playwright"),
                    "samples": samples,
                    "process_tree_rss_note": "Summed owned Chromium process RSS; macOS shared pages may be double-counted.",
                    "process_tree_rss": {
                        "warmed_blank_mib": warmed_blank,
                        "trusted_runtime_mib": trusted_runtime,
                        "representative_mib": representative,
                        "near_limit_mib": near_limit,
                    },
                    "quickjs_resources": quickjs_resources,
                    "runtime_cases": runtime_cases,
                }
            finally:
                browser.close()


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "html5lib": importlib.metadata.version("html5lib"),
        "tinycss2": importlib.metadata.version("tinycss2"),
        "textual": importlib.metadata.version("textual"),
        "qualification_scope": "single-host",
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-samples", type=int, default=15)
    parser.add_argument("--browser-samples", type=int, default=5)
    parser.add_argument(
        "--compiler-only",
        action="store_true",
        help="Skip the real-Chromium runtime and process-memory qualification",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(tuple(argv) if argv is not None else sys.argv[1:])
    limits = CanvasLimits()
    fixtures = build_synthetic_fixtures(limits)
    mirrors = validate_runtime_limit_mirrors(
        limits=limits,
        worker_source=(STATIC / "canvas_runtime_worker.js").read_text(encoding="utf-8"),
        renderer_source=(STATIC / "canvas_renderer.js").read_text(encoding="utf-8"),
    )
    accepted = [fixture for fixture in fixtures if fixture.expected == "accepted"]
    results = [measure_compiler(fixture, args.compiler_samples) for fixture in accepted]
    boundaries = measure_compiler_boundaries(fixtures)
    browser_results = (
        {"status": "not-run"}
        if args.compiler_only
        else run_browser_probe(limits, samples=args.browser_samples)
    )
    summary = build_summary(
        compiler_results=results,
        compiler_boundaries=boundaries,
        browser_results=browser_results,
        environment=_environment(),
        mirrors=mirrors,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
