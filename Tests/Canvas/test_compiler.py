"""Behavioral tests for the closed Canvas V1 document compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from tldw_chatbook.Canvas.compiler import CanvasCompileError, compile_canvas_document
from tldw_chatbook.Canvas.limits import CanvasLimits
from tldw_chatbook.Canvas.models import RenderNode


FIXTURES = Path(__file__).parent / "fixtures" / "compiler"
PNG_BYTES = b"\x89PNG\r\n\x1a\nrest"
PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgpyZXN0"


def _nodes(root: RenderNode) -> tuple[RenderNode, ...]:
    found: list[RenderNode] = []
    pending = [root]
    while pending:
        node = pending.pop()
        found.append(node)
        pending.extend(reversed(node.children))
    return tuple(found)


def _elements(root: RenderNode) -> tuple[RenderNode, ...]:
    return tuple(node for node in _nodes(root) if node.tag != "#text")


def _text(root: RenderNode) -> str:
    return "".join(node.text or "" for node in _nodes(root) if node.tag == "#text")


def _issue_codes(exc: CanvasCompileError) -> set[str]:
    return {issue.code for issue in exc.issues}


def test_complete_document_compiles_to_normalized_immutable_tree_without_markup_sink() -> (
    None
):
    source = (
        '<!doctype html><html lang="en"><head><title>Tea &amp; cake</title>'
        "<style>main { color: red; margin: 0 }</style></head>"
        '<body><main id="app"><p>Hello&nbsp;world</p></main></body></html>'
    )

    plan = compile_canvas_document(source)

    assert plan.runtime_profile == "canvas-v1"
    assert plan.root.tag == "html"
    assert [node.tag for node in plan.root.children] == ["head", "body"]
    assert [node.node_id for node in _nodes(plan.root)] == [
        "node-1",
        "node-2",
        "node-3",
        "node-4",
        "node-5",
        "node-6",
        "node-7",
        "node-8",
    ]
    assert _text(plan.root) == "Tea & cakeHello\N{NO-BREAK SPACE}world"
    assert plan.css_rules == ("main{color:red;margin:0}",)
    assert plan.scripts == ()
    assert plan.compatibility_issues == ()
    assert "<" not in {node.tag for node in _nodes(plan.root)}


def test_fragment_is_wrapped_deterministically_and_discloses_compatibility_change() -> (
    None
):
    plan = compile_canvas_document("<p>Hello</p>")

    assert plan.root.tag == "html"
    assert [child.tag for child in plan.root.children] == ["head", "body"]
    assert [child.tag for child in plan.root.children[1].children] == ["p"]
    assert [issue.code for issue in plan.compatibility_issues] == ["fragment-wrapped"]
    assert plan.compatibility_issues[0].location == "line 1, column 1"


def test_html_parser_recovery_does_not_silently_change_malformed_complete_document() -> (
    None
):
    source = "<!doctype html><html><head></head><body><div></span></body></html>"

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert "html-parse-error" in _issue_codes(caught.value)


def test_classic_inline_scripts_are_extracted_in_document_order_and_omitted_from_tree() -> (
    None
):
    source = (
        "<!doctype html><html><head><script>const first = 1;</script></head>"
        '<body><script type="text/javascript">const second = 2;</script><p>safe</p>'
        "</body></html>"
    )

    plan = compile_canvas_document(source)

    assert plan.scripts == ("const first = 1;", "const second = 2;")
    assert "script" not in {node.tag for node in _elements(plan.root)}
    assert _text(plan.root) == "safe"


@pytest.mark.parametrize(
    ("attributes", "expected_code"),
    [
        ('type="module"', "script-module"),
        ('type="text/vbscript"', "script-type"),
        ('src="data:text/javascript;base64,YWxlcnQoMSk="', "script-source"),
        ('integrity="sha256-abc"', "script-attribute"),
    ],
)
def test_script_fetch_module_and_non_javascript_semantics_fail_closed(
    attributes: str, expected_code: str
) -> None:
    source = (
        f"<!doctype html><html><head></head><body><script {attributes}>"
        "const safe = true;</script></body></html>"
    )

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert expected_code in _issue_codes(caught.value)


def test_oversized_script_is_rejected_before_a_plan_can_hide_the_limit() -> None:
    limits = CanvasLimits(script_bytes=8)
    source = "<!doctype html><html><head></head><body><script>123456789</script></body></html>"

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source, limits=limits)

    assert _issue_codes(caught.value) == {"script-limit"}


def test_local_forms_and_same_document_fragment_links_remain_inert_plan_data() -> None:
    source = (
        '<!doctype html><html><head></head><body><form id="survey">'
        '<label for="name">Name</label><input id="name" name="name" type="text" '
        'value="Ada" required><select name="tea"><option selected>Earl Grey</option>'
        '</select><button type="submit">Save</button></form>'
        '<a href="#survey">Back</a></body></html>'
    )

    plan = compile_canvas_document(source)

    tags = [node.tag for node in _elements(plan.root)]
    assert tags == [
        "html",
        "head",
        "body",
        "form",
        "label",
        "input",
        "select",
        "option",
        "button",
        "a",
    ]
    anchor = next(node for node in _elements(plan.root) if node.tag == "a")
    assert anchor.attributes == (("href", "#survey"),)


def test_basic_svg_shapes_are_namespaced_then_emitted_as_closed_shape_vocabulary() -> (
    None
):
    source = (
        '<!doctype html><html><head></head><body><svg viewBox="0 0 100 100" '
        'aria-label="chart"><g><rect x="1" y="2" width="20" height="30" '
        'fill="red"></rect><circle cx="50" cy="50" r="10"></circle>'
        '<text x="4" y="90">Hi</text></g></svg></body></html>'
    )

    plan = compile_canvas_document(source)

    assert [node.tag for node in _elements(plan.root)] == [
        "html",
        "head",
        "body",
        "svg",
        "g",
        "rect",
        "circle",
        "text",
    ]
    svg = next(node for node in _elements(plan.root) if node.tag == "svg")
    assert ("viewBox", "0 0 100 100") in svg.attributes
    assert _text(plan.root) == "Hi"


def test_standard_document_and_svg_namespace_declarations_do_not_break_basic_shapes() -> (
    None
):
    source = (
        '<!doctype html><html xmlns="http://www.w3.org/1999/xhtml"><head>'
        '<style type="text/css">rect{fill:red}</style></head><body>'
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="5"></rect></svg>'
        "</body></html>"
    )

    plan = compile_canvas_document(source)

    assert [node.tag for node in _elements(plan.root)] == [
        "html",
        "head",
        "body",
        "svg",
        "rect",
    ]
    assert plan.css_rules == ("rect{fill:red}",)
    assert not any(
        name == "xmlns" for node in _elements(plan.root) for name, _ in node.attributes
    )


def test_nonstandard_and_xlink_namespace_declarations_fail_closed() -> None:
    cases = [
        '<!doctype html><html xmlns="https://example.test/ns"><head></head><body></body></html>',
        '<!doctype html><html><head></head><body><svg xmlns:xlink="http://www.w3.org/1999/xlink"><rect></rect></svg></body></html>',
    ]
    for source in cases:
        with pytest.raises(CanvasCompileError) as caught:
            compile_canvas_document(source)
        assert "unsupported-namespace" in _issue_codes(caught.value)


def test_passive_raster_data_image_becomes_opaque_asset_not_a_src_value() -> None:
    source = (
        "<!doctype html><html><head></head><body>"
        f'<img src="{PNG_DATA_URL}" alt="pixel"></body></html>'
    )

    plan = compile_canvas_document(source)

    assert [(asset.asset_id, asset.mime_type, asset.data) for asset in plan.assets] == [
        ("asset-1", "image/png", PNG_BYTES)
    ]
    image = next(node for node in _elements(plan.root) if node.tag == "img")
    assert image.attributes == (("alt", "pixel"), ("data-canvas-asset", "asset-1"))
    assert not any(name == "src" for name, _ in image.attributes)
    assert not any(
        value.startswith(("http:", "https:", "data:")) for _, value in image.attributes
    )


def test_data_image_limits_use_decoded_bytes_and_reject_one_over_custom_ceiling() -> (
    None
):
    source = (
        "<!doctype html><html><head></head><body>"
        f'<img src="{PNG_DATA_URL}"></body></html>'
    )

    assert (
        len(
            compile_canvas_document(source, limits=CanvasLimits(asset_bytes=12))
            .assets[0]
            .data
        )
        == 12
    )
    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source, limits=CanvasLimits(asset_bytes=11))
    assert _issue_codes(caught.value) == {"asset-limit"}


@pytest.mark.parametrize(
    "data_url",
    [
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "data:text/html;base64,PHNjcmlwdD48L3NjcmlwdD4=",
        "data:image/png;charset=utf-8;base64,iVBORw0KGgpyZXN0",
        "data:image/avif;base64,AAAAIGZ0eXBhdmlm",
        "data:image/png;base64,PGh0bWw+",
    ],
)
def test_executable_unknown_parameterized_or_type_confused_image_mime_is_rejected(
    data_url: str,
) -> None:
    source = (
        f'<!doctype html><html><head></head><body><img src="{data_url}"></body></html>'
    )

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert _issue_codes(caught.value) <= {"asset-data", "asset-mime", "asset-signature"}
    assert _issue_codes(caught.value)


def test_stylesheet_and_inline_style_share_the_same_declaration_allowlist() -> None:
    source = (
        "<!doctype html><html><head><style>"
        ".card { display: flex; gap: 8px; background: linear-gradient(red, blue); }"
        "@media (max-width: 600px) { .card { display: block; } }"
        '</style></head><body><div class="card" style="padding: 4px; color: #123">x</div>'
        "</body></html>"
    )

    plan = compile_canvas_document(source)

    assert plan.css_rules == (
        ".card{display:flex;gap:8px;background:linear-gradient(red, blue)}",
        "@media (max-width: 600px){.card{display:block}}",
    )
    card = next(node for node in _elements(plan.root) if node.tag == "div")
    assert ("style", "padding:4px;color:#123") in card.attributes


@pytest.mark.parametrize(
    "script",
    [
        'import("https://example.test/module.js")',
        'import value from "https://example.test/module.js"',
        "export default 1",
        'const value = `${import("https://example.test/module.js")}`',
    ],
)
def test_classic_script_import_export_semantics_cannot_hide_in_dynamic_or_template_code(
    script: str,
) -> None:
    source = f"<!doctype html><html><head></head><body><script>{script}</script></body></html>"

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert "script-module" in _issue_codes(caught.value)


@pytest.mark.parametrize("script", ['const value = "unterminated', "/* unterminated"])
def test_lexically_malformed_script_is_rejected_before_worker_execution(
    script: str,
) -> None:
    source = f"<!doctype html><html><head></head><body><script>{script}</script></body></html>"

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert "script-syntax" in _issue_codes(caught.value)


@pytest.mark.parametrize(
    ("css", "expected_code"),
    [
        ("p { background: url(https://example.test/x.png) }", "css-resource"),
        (r"p { background: u\72l(https://example.test/x.png) }", "css-resource"),
        (
            "p { background-image: image-set('https://example.test/x.png' 1x) }",
            "css-resource",
        ),
        (
            "p { --remote: url(https://example.test/x); background: var(--remote) }",
            "css-custom-property",
        ),
        ('@import "https://example.test/x.css";', "css-at-rule"),
        (r'@\69mport "https://example.test/x.css";', "css-at-rule"),
        (
            '@font-face { font-family: x; src: url("https://example.test/x.woff2") }',
            "css-at-rule",
        ),
        ('@namespace svg url("https://www.w3.org/2000/svg");', "css-at-rule"),
        ("svg|rect { fill: red }", "css-namespace"),
        ("a:visited { color: purple }", "css-visited"),
        ("p { behavior: url(x.htc) }", "css-property"),
        ("p { color: red; broken }", "css-parse-error"),
    ],
)
def test_stylesheet_resource_namespace_visited_and_unknown_constructs_fail_closed(
    css: str, expected_code: str
) -> None:
    source = f"<!doctype html><html><head><style>{css}</style></head><body><p>x</p></body></html>"

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert expected_code in _issue_codes(caught.value)


@pytest.mark.parametrize(
    "style",
    [
        "background: url(https://example.test/x.png)",
        r"background: u\72l(https://example.test/x.png)",
        "--remote: red; color: var(--remote)",
        "unknown-canvas-property: 1",
    ],
)
def test_inline_style_cannot_bypass_stylesheet_token_validation(style: str) -> None:
    source = (
        "<!doctype html><html><head></head><body>"
        f'<p style="{style}">x</p></body></html>'
    )

    with pytest.raises(CanvasCompileError):
        compile_canvas_document(source)


def test_duplicate_source_ids_across_html_and_svg_fail_closed() -> None:
    source = (
        '<!doctype html><html><head></head><body><p id="same">x</p>'
        '<svg><rect id="same"></rect></svg></body></html>'
    )

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert _issue_codes(caught.value) == {"duplicate-source-id"}


def test_native_event_attributes_fail_the_whole_document_in_any_namespace() -> None:
    cases = [
        '<button OnClick="alert(1)">x</button>',
        '<svg><rect onload="alert(1)"></rect></svg>',
    ]
    for body in cases:
        source = f"<!doctype html><html><head></head><body>{body}</body></html>"
        with pytest.raises(CanvasCompileError) as caught:
            compile_canvas_document(source)
        assert "event-handler" in _issue_codes(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        "<widget-panel>custom</widget-panel>",
        "<math><mi>x</mi></math>",
        "<svg><foreignObject><p>x</p></foreignObject></svg>",
        "<audio controls></audio>",
        "<embed>",
        "<template><p>hidden document</p></template>",
    ],
)
def test_custom_foreign_embedded_and_inert_subdocuments_are_not_silently_dropped(
    body: str,
) -> None:
    source = f"<!doctype html><html><head></head><body>{body}</body></html>"

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert _issue_codes(caught.value) & {"unsupported-element", "unsupported-namespace"}


def test_every_known_url_navigation_or_embedding_surface_fixture_is_rejected() -> None:
    cases = json.loads((FIXTURES / "url_surfaces.json").read_text(encoding="utf-8"))

    for case in cases:
        with pytest.raises(CanvasCompileError, match="Canvas document is incompatible"):
            compile_canvas_document(case["html"])


def test_security_self_review_regressions_remain_closed_or_explicitly_supported() -> (
    None
):
    cases = json.loads(
        (FIXTURES / "security_review_regressions.json").read_text(encoding="utf-8")
    )

    for case in cases:
        if case["outcome"] == "accept":
            compile_canvas_document(case["html"])
            continue
        with pytest.raises(CanvasCompileError) as caught:
            compile_canvas_document(case["html"])
        assert case["code"] in _issue_codes(caught.value), case["name"]


def test_diagnostics_are_bounded_position_aware_and_never_echo_source_excerpts() -> (
    None
):
    secret = "SECRET-DO-NOT-ECHO"
    source = f"<!doctype html>\n<html><head></head>\n<body><marquee>{secret}</marquee></body></html>"

    with pytest.raises(CanvasCompileError) as caught:
        compile_canvas_document(source)

    assert caught.value.issues
    for issue in caught.value.issues:
        assert len(issue.code.encode("utf-8")) <= 256
        assert len(issue.message.encode("utf-8")) <= 4 * 1024
        assert issue.location is not None
        assert len(issue.location.encode("utf-8")) <= 512
        assert secret not in issue.message
        assert secret not in issue.location
    assert caught.value.issues[0].location.startswith("line 3, column ")


def test_exact_source_identity_and_compiler_ids_are_stable_across_recompiles() -> None:
    source = "<!doctype html><html><head></head><body><p>é &amp; tea</p></body></html>"

    first = compile_canvas_document(source)
    second = compile_canvas_document(source)

    assert first == second
    assert first.source_identity.source_bytes == 73
    assert (
        first.source_identity.sha256
        == "3cf00f68289a16c422f1a425ec8cf6ff1d1bf9f60941af4b3c5faa10770c9966"
    )


@given(st.text(alphabet="abc XYZ09é茶", min_size=0, max_size=40))
def test_inert_unicode_text_survives_parser_normalization_without_source_or_entity_loss(
    value: str,
) -> None:
    source = f"<!doctype html><html><head></head><body><p>{value}</p></body></html>"

    plan = compile_canvas_document(source)

    assert _text(plan.root) == value
    plan.source_identity.verify_source(source)


@given(
    st.lists(
        st.from_regex(r"[a-z][a-z0-9_-]{0,10}", fullmatch=True),
        min_size=1,
        max_size=12,
        unique=True,
    )
)
def test_distinct_source_ids_remain_distinct_in_the_closed_plan(
    source_ids: list[str],
) -> None:
    children = "".join(f'<span id="{source_id}"></span>' for source_id in source_ids)
    source = f"<!doctype html><html><head></head><body>{children}</body></html>"

    plan = compile_canvas_document(source)

    actual_ids = [
        value
        for node in _elements(plan.root)
        for name, value in node.attributes
        if name == "id"
    ]
    assert actual_ids == source_ids


def test_custom_node_and_css_rule_limits_reject_before_model_default_limits_apply() -> (
    None
):
    node_source = "<!doctype html><html><head></head><body><p>x</p></body></html>"
    with pytest.raises(CanvasCompileError) as node_error:
        compile_canvas_document(node_source, limits=CanvasLimits(dom_nodes=4))
    assert _issue_codes(node_error.value) == {"dom-limit"}

    css_source = (
        "<!doctype html><html><head><style>p{color:red}div{color:blue}</style></head>"
        "<body></body></html>"
    )
    with pytest.raises(CanvasCompileError) as css_error:
        compile_canvas_document(css_source, limits=CanvasLimits(css_rules=1))
    assert _issue_codes(css_error.value) == {"css-rule-limit"}
