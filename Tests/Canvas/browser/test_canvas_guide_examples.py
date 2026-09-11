"""Execute exact packaged examples in Chromium through owned loopback fixtures."""

import pytest
from markdown_it import MarkdownIt
from playwright.sync_api import expect

from Tests.Canvas.browser.test_canvas_zero_egress import (
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
from tldw_chatbook.Canvas.guide import read_canvas_guide
from tldw_chatbook.Canvas.profiles import load_profile_snapshot

pytestmark = pytest.mark.loopback_network


@pytest.mark.parametrize(
    "topic,ordinal,labels",
    [
        ("basics", 0, ("Manual: 8 hours", "Assisted: 5 hours")),
        ("controls", 0, ("Quantity (whole units)", "Unit price")),
        ("mermaid", 0, ("Start", "Ready?", "Continue", "Revise")),
        ("mermaid", 1, ("Reader", "Library", "Request", "Response", "Complete")),
    ],
    ids=["passive-svg", "calculator", "flowchart", "sequence"],
)
def test_packaged_guide_example_executes(
    chromium_browser, asset_server, egress_server, tmp_path, topic, ordinal, labels
):
    examples = [
        token.content
        for token in MarkdownIt().parse(read_canvas_guide(topic))
        if token.type == "fence" and token.info == "html"
    ]
    source = examples[ordinal]
    profile = "canvas-v2-mermaid-1" if topic == "mermaid" else "canvas-v1"
    snapshot = load_profile_snapshot()
    assert any(
        record.profile_id == profile and record.executable
        for record in snapshot.profiles
    )
    asset_server.v2 = topic == "mermaid"
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        status = _load(
            page,
            _wire_plan(source, runtime_profile=profile, snapshot=snapshot),
            recorder,
            source=source,
        )
        assert status["state"] == "ready", status
        frame = page.frame(name="canvas-renderer")
        assert frame is not None
        root = frame.locator("#canvas-root")
        for label in labels:
            expect(root).to_contain_text(label)
        if topic == "controls":
            quantity = frame.get_by_label("Quantity (whole units)")
            price = frame.get_by_label("Unit price", exact=True)
            total = frame.locator("#total")
            expect(total).to_have_text("25.00")
            quantity.focus()
            quantity.press("Tab")
            expect(price).to_be_focused()
            quantity.fill("3")
            price.fill("7")
            expect(total).to_have_text("21.00")
            quantity.fill("")
            expect(total).to_have_text("Enter values within the stated limits.")
            quantity.fill("3")
            expect(total).to_have_text("21.00")
        else:
            expect(frame.locator("svg")).to_have_count(1)
        for width, height in ((390, 844), (1280, 800)):
            page.set_viewport_size({"width": width, "height": height})
            # Size the test harness viewport, leaving the shipped renderer intact.
            page.locator("iframe").evaluate(
                "(node, size) => { node.style.width = size[0] + 'px'; "
                "node.style.height = size[1] + 'px'; }",
                [width - 32, height - 32],
            )
            page.screenshot(
                path=str(tmp_path / f"{topic}-{ordinal}-{width}.png"),
                full_page=True,
            )
        assert not errors
        assert page.evaluate("window.__canvasHarness.status.state") == "ready"
        _assert_zero_generated_egress(recorder, egress_server)
        (tmp_path / "browser-version.txt").write_text(
            chromium_browser.version, encoding="utf-8"
        )
    finally:
        context.close()
