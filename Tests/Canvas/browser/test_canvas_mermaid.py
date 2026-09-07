"""Real V2 QuickJS startup through the strict recorded zero-egress boundary."""

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
