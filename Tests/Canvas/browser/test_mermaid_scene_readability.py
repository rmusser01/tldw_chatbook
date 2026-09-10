"""Scene-only QA in the existing V1 renderer, not V2 startup qualification."""
# ruff: noqa: F401, F811 -- imported pytest fixtures are registered by name.

import html
import json
from pathlib import Path

import pytest

from Tests.Canvas.browser.test_canvas_zero_egress import (
    _assert_zero_generated_egress,
    _load,
    _new_page,
    _wire_plan,
    asset_server,
    chromium_browser,
    egress_server,
    playwright_runtime,
)
from Tests.Canvas.mermaid_probe import run_mermaid_case


def scene_html(node):
    attributes = " ".join(
        f'{key}="{html.escape(value, quote=True)}"' for key, value in node["attributes"]
    )
    return (
        f"<{node['tag']} {attributes}>"
        + html.escape(node["text"])
        + "".join(scene_html(child) for child in node["children"])
        + f"</{node['tag']}>"
    )


@pytest.mark.loopback_network
def test_scene_default_typography_and_intrinsic_scrolling(
    chromium_browser, asset_server, egress_server
):
    sources = [
        "flowchart LR\nA -->|First| B{Long label}\nA -->|Second| C",
        "flowchart TD\nA -->|First| B[Long label]\nA -->|Second!!| C",
        "flowchart TD\nA[Start] --> B{Ready?}\nB -->|Yes| C(Continue)\nB -->|No| D[Revise]\nC --> E[Join]\nD --> E",
        "sequenceDiagram\nparticipant A as Alice\nparticipant B as Bob\nparticipant C as Carol\nA->>B: Hello\nNote left of A: 左 é\nNote right of B: 😀 שלום\nNote over A,C: Shared note\nB-->>C: Reply",
        'flowchart LR\nA["中文 é 👨‍👩‍👧‍👦 שלום"] --> B["' + "unbreakable" * 10 + '"]',
    ]
    evidence = []
    for index, source in enumerate(sources):
        context, page, recorder = _new_page(
            chromium_browser, asset_server, egress_server
        )
        try:
            page.set_viewport_size({"width": 480, "height": 900})
            page.locator("iframe").evaluate(
                "node => {node.style.width='280px';node.style.height='840px'}"
            )
            result = run_mermaid_case({"operation": "layout", "source": source})
            assert result["ok"], result
            document = (
                "<!doctype html><html><head><style>body{font-family:serif;font-size:40px}</style></head><body>"
                + scene_html(result["scene"]["root"])
                + "</body></html>"
            )
            status = _load(page, _wire_plan(document), recorder)
            assert status["state"] == "ready", status
            frame = page.frame(name="canvas-renderer")
            assert frame.locator("pre").text_content() == source
            observed = frame.locator("svg").evaluate("""svg => {
              const box = svg.viewBox.baseVal;
              return {width:box.width,height:box.height,scrollWidth:svg.parentElement.scrollWidth,clientWidth:svg.parentElement.clientWidth,
                texts:Array.from(svg.querySelectorAll('text'), node => {
                  const b=node.getBBox(), s=getComputedStyle(node);
                  return {text:node.textContent,x:b.x,y:b.y,width:b.width,height:b.height,font:s.fontFamily,size:s.fontSize,weight:s.fontWeight,style:s.fontStyle};
                })};
            }""")
            assert observed["scrollWidth"] > observed["clientWidth"]
            for text in observed["texts"]:
                assert text["font"] == "monospace"
                assert (text["size"], text["weight"], text["style"]) == (
                    "16px",
                    "400",
                    "normal",
                )
                assert text["x"] >= 0 and text["y"] >= 0
                assert text["x"] + text["width"] <= observed["width"]
                assert text["y"] + text["height"] <= observed["height"]
                assert text["height"] <= 24
            evidence.append(observed)
            for position, first in enumerate(observed["texts"]):
                for second in observed["texts"][position + 1 :]:
                    assert (
                        first["x"] + first["width"] <= second["x"]
                        or second["x"] + second["width"] <= first["x"]
                        or first["y"] + first["height"] <= second["y"]
                        or second["y"] + second["height"] <= first["y"]
                    )
            output = Path("output/playwright/mermaid-scene")
            output.mkdir(parents=True, exist_ok=True)
            frame.locator("svg").scroll_into_view_if_needed()
            page.screenshot(path=str(output / f"scene-{index}.png"), full_page=True)
            page.set_viewport_size({"width": 1600, "height": 1200})
            page.locator("iframe").evaluate(
                "node => {node.style.width='1500px';node.style.height='1150px'}"
            )
            frame.locator("svg").screenshot(
                path=str(output / f"scene-{index}-wide.png")
            )
            # Explicit authored-style overrides remain effective; they do not rerun layout.
            frame.locator("svg text").first.evaluate(
                "node => node.style.setProperty('font-size', '22px')"
            )
            assert (
                frame.locator("svg text").first.evaluate(
                    "node => getComputedStyle(node).fontSize"
                )
                == "22px"
            )
            assert (
                frame.locator("svg").evaluate("svg => svg.viewBox.baseVal.width")
                == observed["width"]
            )
            _assert_zero_generated_egress(recorder, egress_server)
        finally:
            context.close()
    (output / "geometry.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2)
    )


@pytest.mark.loopback_network
def test_four_mixed_scenes_fit_real_v1_worker_patches(
    chromium_browser, asset_server, egress_server
):
    sources = [
        "flowchart TD\nA[Start] --> B[End]",
        "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: Hello",
    ] * 2
    result = run_mermaid_case({"operation": "render", "sources": sources})
    assert result["ok"]
    # This test-side allocator is authored V1 guest JS. Task4 owns production integration.
    script = (
        "const scenes = "
        + json.dumps(result["scene"], ensure_ascii=True)
        + ";"
        + """
      function build(record, svg) {
        svg = svg || record.tag === 'svg';
        const node = svg ? document.createElementNS('http://www.w3.org/2000/svg', record.tag) : document.createElement(record.tag);
        for (const pair of record.attributes) {
          if (pair[0] === 'style') {
            for (const declaration of pair[1].split(';')) {
              const parts = declaration.split(':'); node.style.setProperty(parts[0], parts[1]);
            }
          } else node.setAttribute(pair[0], pair[1]);
        }
        if (record.text) node.textContent = record.text;
        for (const child of record.children) node.appendChild(build(child, svg));
        return node;
      }
      for (const scene of scenes) document.body.appendChild(build(scene.root, false));
    """
    )
    context, page, recorder = _new_page(chromium_browser, asset_server, egress_server)
    try:
        status = _load(
            page,
            _wire_plan(
                "<!doctype html><html><body><script>"
                + script
                + "</script></body></html>"
            ),
            recorder,
        )
        assert status["state"] == "ready", status
        frame = page.frame(name="canvas-renderer")
        assert frame.locator("svg").count() == 4
        assert frame.locator("pre").all_text_contents() == sources
        _assert_zero_generated_egress(recorder, egress_server)
    finally:
        context.close()
