"""Real V2 QuickJS startup through the strict recorded zero-egress boundary."""

import html
import json
import platform
import re
import time
from pathlib import Path

import pytest

from Tests.Canvas.browser.test_canvas_zero_egress import (
    STATIC,
    _assert_zero_generated_egress,
    _load,
    _new_page,
    _wire_plan,
)
from Tests.Canvas.browser.test_canvas_zero_egress import (
    asset_server as asset_server,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Canvas.browser.test_canvas_zero_egress import (
    chromium_browser as chromium_browser,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Canvas.browser.test_canvas_zero_egress import (
    egress_server as egress_server,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Canvas.browser.test_canvas_zero_egress import (
    playwright_runtime as playwright_runtime,  # noqa: PLC0414 - pytest fixture re-export
)

pytestmark = pytest.mark.loopback_network


@pytest.mark.parametrize(
    "diagram,code",
    [
        (
            '%%{init: {"securityLevel":"loose"}}%%\nflowchart TD\nA',
            "unsupported-syntax",
        ),
        ('flowchart TD\nA["<img src=__EGRESS__>"]', "unsupported-label"),
        ('flowchart TD\nA["' + "é" * 171 + '"]', "label-limit"),
        (
            "flowchart TD\n"
            + "\n".join(f"N{i}-->N{j}" for i in range(8) for j in range(i + 1, 8)),
            "edges-limit",
        ),
        ("flowchart TD\n" + "[" * 4000, "unsupported-syntax"),
    ],
    ids=["directive", "markup", "large-graphemes", "dense-dag", "malicious-lexeme"],
)
def test_hostile_diagrams_refuse_with_typed_diagnostics_and_zero_egress(
    candidate_snapshot, chromium_browser, asset_server, egress_server, diagram, code
):
    asset_server.v2 = True
    diagram = diagram.replace("__EGRESS__", egress_server.origin)
    source = (
        '<pre data-canvas-diagram="mermaid">'
        + html.escape(diagram)
        + '</pre><script>document.body.textContent="must not run";</script>'
    )
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        status = _load(
            page,
            _wire_plan(
                source,
                runtime_profile="canvas-v2-mermaid-1",
                snapshot=candidate_snapshot,
            ),
            recorder,
            source=source,
        )
        assert status["state"] == "failed", status
        assert status["diagram"]["code"] == code, status
        assert status["diagram"]["ordinal"] == 1
        frame = page.frame(name="canvas-renderer")
        assert frame.locator("svg").count() == 0
        assert "must not run" not in frame.locator("#canvas-root").inner_text()
        page.wait_for_timeout(100)
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


def test_v2_worker_backstop_terminates_stalled_library_startup(
    candidate_snapshot, chromium_browser, asset_server, egress_server
):
    """A fault-injected uninterruptible worker must be terminated by the renderer."""
    asset_server.v2 = True
    worker = (STATIC / "canvas_runtime_worker_v2.js").read_text()
    token = "function runStartup(operationId) {"
    assert worker.count(token) == 1
    asset_server.runtime_overrides["/static/canvas_runtime_worker_v2.js"] = (
        worker.replace(token, token + "while(true){}").encode()
    )
    renderer = (STATIC / "canvas_renderer_v2.js").read_text()
    asset_server.runtime_overrides["/static/canvas_renderer_v2.js"] = (
        "const actualTerminate=Worker.prototype.terminate; window.__releaseTerminations=0; Worker.prototype.terminate=function(){window.__releaseTerminations++;return actualTerminate.call(this)};\n"
        + renderer
    ).encode()
    source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea]</pre>'
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        status = _load(
            page,
            _wire_plan(
                source,
                runtime_profile="canvas-v2-mermaid-1",
                snapshot=candidate_snapshot,
            ),
            recorder,
            source=source,
        )
        assert (
            status["state"] == "failed" and status["code"] == "worker-unresponsive"
        ), status
        frame = page.frame(name="canvas-renderer")
        assert frame.evaluate("window.__releaseTerminations") == 1
        assert frame.locator("svg").count() == 0
        page.wait_for_timeout(100)
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.parametrize(
    "fixture",
    [
        "branch-rejoin",
        "notes",
        "guide-flow",
        "guide-sequence",
        "unicode",
        "four",
        "mixed",
        "near-limit",
    ],
)
def test_release_useful_examples_execute_complete_v2_startup(
    candidate_snapshot, chromium_browser, asset_server, egress_server, fixture
):
    """Catch diagrams that parse successfully but exceed the real shared startup budget."""
    asset_server.v2 = True
    if fixture.startswith("guide-"):
        examples = re.findall(
            r"```html\n(.*?)\n```",
            (STATIC / "mermaid-authoring.txt").read_text(),
            re.DOTALL,
        )
        assert len(examples) == 2
        source = examples[0 if fixture == "guide-flow" else 1]
    elif fixture in {"four", "mixed"}:
        diagrams = [
            "flowchart TD\nA[Tea] --> B[Cake]",
            "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: Hello",
        ]
        if fixture == "four":
            diagrams *= 2
        source = "<!doctype html><html><body><h1>Offline diagrams</h1>" + "".join(
            '<pre data-canvas-diagram="mermaid">' + html.escape(item) + "</pre>"
            for item in diagrams
        )
        source += '<button id="increment">Count</button><p id="count">0</p><script>let n=0;document.getElementById("increment").addEventListener("click",()=>{document.getElementById("count").textContent=String(++n)});</script></body></html>'
    else:
        diagram = {
            "branch-rejoin": "flowchart TD\nA[Start] --> B{Ready?}\nB -->|Yes| C(Continue)\nB -->|No| D[Revise]\nC --> E[Join]\nD --> E\nE --> F[Finish]",
            "notes": "sequenceDiagram\nparticipant A as Alice\nparticipant B as Bob\nparticipant C as Carol\nA->>B: Hello\nNote left of A: Left\nNote right of B: Right\nNote over A: Alone\nNote over A,C: Shared\nB-->>C: Reply",
            "unicode": 'flowchart LR\nA["中文 é 👨‍👩‍👧‍👦 שלום"] --> B["'
            + "unbreakable" * 10
            + '"]',
            "near-limit": "flowchart TD\n"
            + "\n".join(f"N{i}-->N{i + 1}" for i in range(15)),
        }[fixture]
        source = (
            '<!doctype html><html><body><pre data-canvas-diagram="mermaid">'
            + html.escape(diagram)
            + "</pre></body></html>"
        )
    context, page, recorder = _new_page(
        chromium_browser, asset_server, egress_server, observe_patches=True
    )
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        start = time.perf_counter()
        status = _load(page, plan, recorder, source=source)
        elapsed = (time.perf_counter() - start) * 1000
        assert status["state"] == "ready", status
        frame = page.frame(name="canvas-renderer")
        assert frame.locator("svg").count() == {"four": 4, "mixed": 2}.get(fixture, 1)
        if fixture in {"four", "mixed"}:
            frame.locator("#increment").click()
            from playwright.sync_api import expect

            expect(frame.locator("#count")).to_have_text("1")
        output = Path("output/playwright/mermaid-release")
        output.mkdir(parents=True, exist_ok=True)
        for width, label in ((1600, "wide"), (390, "narrow")):
            page.set_viewport_size({"width": width, "height": 1200})
            page.locator("iframe").evaluate(
                "(node,width)=>{node.style.width=width+'px';node.style.height='1150px'}",
                width - 40,
            )
            page.screenshot(path=str(output / f"{fixture}-{label}.png"), full_page=True)
        metrics = frame.locator("svg").evaluate_all(
            "nodes=>nodes.map(svg=>({width:svg.viewBox.baseVal.width,height:svg.viewBox.baseVal.height,elements:svg.querySelectorAll('*').length+1,scrollWidth:svg.parentElement.scrollWidth,clientWidth:svg.parentElement.clientWidth}))"
        )
        assert all(row["width"] <= 2048 and row["height"] <= 4096 for row in metrics)
        assert all(
            row["scrollWidth"] > row["clientWidth"]
            for row in metrics
            if row["width"] > row["clientWidth"]
        )
        if fixture == "unicode":
            assert metrics[0]["scrollWidth"] > metrics[0]["clientWidth"]
        scroll_metrics = []
        for svg in frame.locator("svg").all():
            scrolling = svg.evaluate(
                "node=>{const parent=node.parentElement;parent.scrollLeft=parent.scrollWidth;return {position:parent.scrollLeft,end:parent.scrollLeft+parent.clientWidth,width:parent.scrollWidth,right:node.getBoundingClientRect().right,viewportRight:parent.getBoundingClientRect().right}}"
            )
            assert scrolling["end"] >= scrolling["width"] - 1
            assert scrolling["right"] <= scrolling["viewportRight"] + 1
            scroll_metrics.append(scrolling)
        frame.locator("svg").last.scroll_into_view_if_needed()
        assert frame.locator("svg").last.is_visible()
        frame.locator("svg").first.scroll_into_view_if_needed()
        assert frame.locator("svg").first.is_visible()
        # Ordinary inherited styles do not override the renderer's explicit defaults.
        frame.locator("#canvas-root body").evaluate(
            "node=>{node.style.fontFamily='serif';node.style.fontSize='40px'}"
        )
        assert (
            frame.locator("svg text").first.evaluate(
                "node=>getComputedStyle(node).fontFamily"
            )
            == "monospace"
        )
        assert (
            frame.locator("svg text").first.evaluate(
                "node=>getComputedStyle(node).fontSize"
            )
            == "16px"
        )
        frame.locator("svg text").first.evaluate("node=>node.style.fontSize='22px'")
        assert (
            frame.locator("svg text").first.evaluate(
                "node=>getComputedStyle(node).fontSize"
            )
            == "22px"
        )
        assert (
            frame.locator("svg").first.evaluate("node=>node.viewBox.baseVal.width")
            == metrics[0]["width"]
        )
        patches = frame.evaluate("window.__releasePatchCounts")
        assert patches and all(0 <= count <= 500 for count in patches)
        (output / f"{fixture}-metrics.json").write_text(
            json.dumps(
                {
                    "browser": chromium_browser.version,
                    "platform": platform.platform(),
                    "source_bytes": len(source.encode()),
                    "load_to_ready_ms": elapsed,
                    "geometry": metrics,
                    "far_right_scroll": scroll_metrics,
                    "transaction_patch_counts": patches,
                },
                indent=2,
            )
        )
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "badPlan.css_rules = null;",
        "badPlan.css_rules = [17];",
        "badPlan.css_rules = Array(901).fill('p {color:red;}');",
        "badPlan.css_rules = ['@media screen {' + 'p {color:red;}'.repeat(900) + '}'];",
        "badPlan.assets = null;",
        "badPlan.assets = [{asset_id:'a', mime_type:'image/png', data_base64:'!'}];",
        "badPlan.assets = [{asset_id:'a', mime_type:'text/html', data_base64:'YWJj'}];",
        "badPlan.assets = [{asset_id:'a', mime_type:'image/png', data_base64:'YWJj', extra:true}];",
        "badPlan.root.tag = 'script';",
        "badPlan.root.text = 17;",
        "badPlan.root.text = '';",
        "badPlan.root.attributes.push(['onclick', 'alert(1)']);",
        "badPlan.root.attributes.push(['title', 17]);",
        "badPlan.root.children[0].tag = '#text';",
        "badPlan.root.children[0].children.push({node_id:'bad-svg',tag:'svg',text:null,attributes:[],children:[{node_id:'bad-html',tag:'div',text:null,attributes:[],children:[]}]});",
        "badPlan.root.children.push({node_id:'bad-img',tag:'img',text:null,attributes:[['data-canvas-asset','missing']],children:[]});",
    ],
)
def test_worker_rejects_generic_record_corruption_before_prepared(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
    mutation,
):
    asset_server.v2 = True
    original = (STATIC / "canvas_renderer_v2.js").read_text()
    before = 'worker.postMessage({type: "prepare", plan: pendingPlan, runtime_data: pendingRuntimeData});'
    assert before in original
    after = (
        "const badPlan = JSON.parse(JSON.stringify(pendingPlan));"
        + mutation
        + 'worker.postMessage({type: "prepare", plan: badPlan, runtime_data: pendingRuntimeData});'
    )
    asset_server.runtime_overrides["/static/canvas_renderer_v2.js"] = original.replace(
        before, after
    ).encode()
    source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea]</pre>'
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "failed", status
        assert status["code"] == "runtime-error", status
        assert page.evaluate("window.__canvasHarness.startupApproved") is None
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


def test_authored_diagram_text_matches_native_text(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
):
    asset_server.v2 = True
    source = """<pre id="diagram" data-canvas-diagram="mermaid">flowchart TD
A[Tea]</pre><script>
const target = document.getElementById("diagram");
canvas.submit({
  target: target.textContent,
  wrapper: target.querySelector("div").textContent,
  svg: target.querySelector("svg").textContent,
  label: target.querySelector("text").textContent,
  source: target.querySelector("pre").textContent
});
</script>"""
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "ready", status
        virtual = page.evaluate("""window.__canvasHarness.messages.find(
            item => item.type === 'canvas:bridge-request' && item.kind === 'submit').value""")
        target = page.frame(name="canvas-renderer").locator("#diagram")
        native = {
            name: locator.text_content()
            for name, locator in {
                "target": target,
                "wrapper": target.locator("div"),
                "svg": target.locator("svg"),
                "label": target.locator("text"),
                "source": target.locator("pre"),
            }.items()
        }
        assert (
            native["target"]
            == "Flowchart. Diagram source follows.Teaflowchart TD\nA[Tea]"
        )
        assert virtual == native
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


def test_generic_worker_admission_preserves_real_css_and_raster_assets(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
):
    asset_server.v2 = True
    from Tests.Canvas.browser.test_canvas_zero_egress import FIXTURES

    source = (FIXTURES / "benign_canvas.html").read_text()
    source = source.replace(
        "</body>",
        '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea]</pre></body>',
    )
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "ready", status
        frame = page.frame(name="canvas-renderer")
        assert frame.locator("img").evaluate_all(
            "images => images.every(image => image.complete && image.naturalWidth === 1)"
        )
        assert (
            frame.locator("pre[data-canvas-diagram] svg text").text_content() == "Tea"
        )
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


def test_unexpected_engine_failure_is_not_reported_as_quota_success(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
):
    asset_server.v2 = True
    original = (STATIC / "canvas_runtime_worker_v2.js").read_text()
    before = 'const library = handleEval(librarySource, "canvas-private-diagrams.js");'
    assert before in original
    asset_server.runtime_overrides["/static/canvas_runtime_worker_v2.js"] = (
        original.replace(
            before, 'deadline = 0; throw new Error("private-engine-detail");'
        ).encode()
    )
    source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA</pre>'
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "failed"
        assert status["code"] == "runtime-error"
        assert "private-engine-detail" not in str(status)
        assert page.frame(name="canvas-renderer").locator("svg").count() == 0
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


def test_diagrams_render_before_authored_code_once_and_keep_handles_private(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
):
    asset_server.v2 = True
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    source = """<pre id="diagram" data-canvas-diagram="mermaid">flowchart TD
A[Tea] --> B[Cake]</pre><p id="observed"></p><script>
const target = document.getElementById("diagram");
document.getElementById("observed").textContent =
  target.querySelectorAll("svg").length + ":" +
  typeof renderDiagrams + ":" + typeof librarySource + ":" + typeof virtualControls;
target.setAttribute("data-canvas-diagram", "changed");
</script>"""
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        assert set(plan) == {
            "runtime_profile",
            "source_identity",
            "root",
            "assets",
            "css_rules",
            "scripts",
            "diagrams",
            "profile_manifest_sha256",
        }
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "ready", (status, recorder.observations)
        frame = page.frame(name="canvas-renderer")
        assert (
            frame.locator("#observed").inner_text() == "1:undefined:undefined:undefined"
        )
        assert frame.locator("#diagram svg").count() == 1
        assert (
            frame.locator("#diagram").get_attribute("data-canvas-diagram") == "changed"
        )
        assert frame.locator("#diagram svg text").all_text_contents() == ["Tea", "Cake"]
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.parametrize("boundary", ["renderer", "worker"])
@pytest.mark.parametrize(
    "damage", ["extra", "target", "text", "manifest", "source", "library"]
)
def test_each_consumer_rejects_malformed_plan_or_runtime_data(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
    boundary,
    damage,
):
    asset_server.v2 = True
    source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea]</pre>'
    plan = _wire_plan(
        source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
    )
    # Corrupt the real prepare message after renderer validation to independently
    # reach worker admission. Native parsing of diagram source is never used.
    mutations = {
        "extra": "badPlan.compatibility_issues = [];",
        "target": "badPlan.diagrams[0].target_node_id = badPlan.root.node_id;",
        "text": "badPlan.diagrams[0].source = 'flowchart TD\\nB';",
        "manifest": "badPlan.profile_manifest_sha256 = '0'.repeat(64);",
        "source": "badData.source += ' ';",
        "library": "badData.library += ' ';",
    }
    if boundary == "worker":
        original = (STATIC / "canvas_renderer_v2.js").read_text()
        before = 'worker.postMessage({type: "prepare", plan: pendingPlan, runtime_data: pendingRuntimeData});'
        assert before in original
        after = (
            "const badPlan = JSON.parse(JSON.stringify(pendingPlan));"
            "const badData = JSON.parse(JSON.stringify(pendingRuntimeData));"
            + mutations[damage]
            + 'worker.postMessage({type: "prepare", plan: badPlan, runtime_data: badData});'
        )
        asset_server.runtime_overrides["/static/canvas_renderer_v2.js"] = (
            original.replace(before, after).encode()
        )
    elif damage == "extra":
        plan["compatibility_issues"] = []
    elif damage == "target":
        plan["diagrams"][0]["target_node_id"] = plan["root"]["node_id"]
    elif damage == "text":
        plan["diagrams"][0]["source"] = "flowchart TD\nB"
    elif damage == "manifest":
        plan["profile_manifest_sha256"] = "0" * 64
    elif damage == "source":
        source += " "
    elif damage == "library":
        asset_server.runtime_overrides["/static/mermaid-subset.json"] = (
            STATIC / "mermaid-subset.json"
        ).read_bytes() + b" "
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "failed", status
        assert status["code"] == (
            "invalid-plan" if boundary == "renderer" else "runtime-error"
        )
        assert page.frame(name="canvas-renderer").locator("svg").count() == 0
        assert page.evaluate("window.__canvasHarness.startupApproved") is None
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


def test_four_mixed_diagrams_share_real_startup_transaction(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
):
    asset_server.v2 = True
    flow = "flowchart TD\nA[Tea] --> B[Cake]"
    sequence = "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: Hello"
    source = "".join(
        '<pre data-canvas-diagram="mermaid">' + item + "</pre>"
        for item in [flow, sequence, flow, sequence]
    )
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "ready", status
        frame = page.frame(name="canvas-renderer")
        assert frame.locator("svg").count() == 4
        for index, expected in enumerate(
            [
                ["Cake", "Tea"],
                ["A", "B", "Hello"],
                ["Cake", "Tea"],
                ["A", "B", "Hello"],
            ]
        ):
            assert (
                sorted(
                    frame.locator("svg").nth(index).locator("text").all_text_contents()
                )
                == expected
            )
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()


@pytest.mark.parametrize(
    "tail",
    [
        '<pre data-canvas-diagram="mermaid">flowchart TD\nA --> A</pre>',
        '<script>throw new Error("source-secret");</script>',
    ],
)
def test_failed_startup_publishes_no_diagram_mutations(
    candidate_snapshot,
    chromium_browser,
    asset_server,
    egress_server,
    tail,
):
    asset_server.v2 = True
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    source = (
        '<pre id="first" data-canvas-diagram="mermaid">flowchart TD\nA[Tea]</pre>'
        + tail
    )
    try:
        plan = _wire_plan(
            source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        status = _load(page, plan, recorder, source=source)
        assert status["state"] == "failed"
        frame = page.frame(name="canvas-renderer")
        assert frame.locator("svg").count() == 0
        assert frame.locator("#first").inner_text() == "flowchart TD\nA[Tea]"
        assert "source-secret" not in str(status)
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()
