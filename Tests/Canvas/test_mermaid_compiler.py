"""Profile-specific declaration admission and exact-source preservation."""

from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.compilation import prepare_canvas_document
from tldw_chatbook.Canvas.compiler import CanvasCompileError, compile_canvas_document
from tldw_chatbook.Canvas.limits import CanvasLimitError
from tldw_chatbook.Canvas.models import CanvasRenderPlan, CanvasRenderPlanV2


def test_v2_declaration_is_data_not_script(candidate_snapshot):
    source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea &amp; cake]</pre>'
    plan = compile_canvas_document(
        source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
    )
    assert plan.diagrams[0].source == "flowchart TD\nA[Tea & cake]"
    assert plan.scripts == ()
    assert plan.source_identity.source_bytes == len(source.encode("utf-8"))


@pytest.mark.parametrize(
    "declaration",
    [
        '<pre data-canvas-diagram="unknown">flowchart TD</pre>',
        '<div data-canvas-diagram="mermaid">flowchart TD</div>',
        '<pre data-canvas-diagram="mermaid"> </pre>',
        '<pre data-canvas-diagram="mermaid">flowchart TD<b>A</b></pre>',
        '<pre data-canvas-diagram="mermaid">flowchart TD<!-- A --></pre>',
        '<pre data-canvas-diagram="mermaid">' + "A" * 8193 + "</pre>",
        '<pre data-canvas-diagram="mermaid">flowchart TD</pre>' * 5,
        ('<pre data-canvas-diagram="mermaid">' + "A" * 6000 + "</pre>") * 3,
    ],
)
def test_v2_refuses_invalid_declarations(candidate_snapshot, declaration):
    with pytest.raises(CanvasCompileError):
        compile_canvas_document(
            declaration,
            runtime_profile="canvas-v2-mermaid-1",
            snapshot=candidate_snapshot,
        )


def test_v2_requires_real_retained_verified_snapshot(profile_snapshot):
    with pytest.raises(CanvasCompileError):
        compile_canvas_document(
            "<p>data</p>",
            runtime_profile="canvas-v2-mermaid-1",
            snapshot=profile_snapshot,
        )


def test_v1_leaves_diagram_attribute_inert():
    plan = compile_canvas_document('<pre data-canvas-diagram="other"><b>data</b></pre>')
    assert type(plan) is CanvasRenderPlan
    assert plan.runtime_profile == "canvas-v1"
    assert plan.scripts == ()


@pytest.mark.parametrize("profile", ["../runtime", "unknown-runtime"])
def test_unknown_profile_fails_with_typed_compiler_error(candidate_snapshot, profile):
    with pytest.raises(CanvasCompileError):
        compile_canvas_document(
            "<p>data</p>", runtime_profile=profile, snapshot=candidate_snapshot
        )


def test_prepare_uses_one_real_html_parse(candidate_snapshot, monkeypatch):
    import html5lib

    parse = html5lib.HTMLParser.parse
    calls = []

    def counting_parse(parser, source, *args, **kwargs):
        calls.append(source)
        return parse(parser, source, *args, **kwargs)

    monkeypatch.setattr(html5lib.HTMLParser, "parse", counting_parse)
    source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea]</pre>'
    plan = prepare_canvas_document(
        source, operation="create", parent_profile=None, snapshot=candidate_snapshot
    )
    assert calls == [source]
    assert plan.diagrams[0].source == "flowchart TD\nA[Tea]"


def test_prepare_selects_profile_from_structure_and_preserves_v2(candidate_snapshot):
    literal = '<script>const text = "data-canvas-diagram";</script>'
    assert (
        type(
            prepare_canvas_document(
                literal,
                operation="create",
                parent_profile=None,
                snapshot=candidate_snapshot,
            )
        )
        is CanvasRenderPlan
    )
    source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA</pre>'
    plan = prepare_canvas_document(
        source, operation="create", parent_profile=None, snapshot=candidate_snapshot
    )
    assert type(plan) is CanvasRenderPlanV2
    updated = prepare_canvas_document(
        "<p>removed</p>",
        operation="update",
        parent_profile=plan.runtime_profile,
        snapshot=candidate_snapshot,
    )
    assert updated.runtime_profile == "canvas-v2-mermaid-1"
    assert updated.diagrams == ()


def test_v2_model_rejects_rebound_or_missing_records(candidate_snapshot):
    plan = compile_canvas_document(
        '<pre data-canvas-diagram="mermaid">flowchart TD\nA</pre>',
        runtime_profile="canvas-v2-mermaid-1",
        snapshot=candidate_snapshot,
    )
    for records in (
        (),
        (replace(plan.diagrams[0], source="flowchart TD\nB"),),
        (replace(plan.diagrams[0], target_node_id=plan.root.node_id),),
    ):
        with pytest.raises(CanvasLimitError):
            replace(plan, diagrams=records)


def test_library_and_authored_source_share_evaluated_byte_limit(candidate_snapshot):
    # Exact evaluated library source is 165599 bytes for this real candidate;
    # the inventory also contains JSON escaping and license bytes (not evaluated).
    from tldw_chatbook.Canvas.profiles import runtime_assets_for

    assets = runtime_assets_for(candidate_snapshot, "canvas-v2-mermaid-1")
    remaining = 262144 - assets.manifest["mermaid_candidate"]["source_bytes"]
    source = "<script>" + " " * remaining + "</script>"
    compile_canvas_document(
        source, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
    )
    with pytest.raises(CanvasCompileError):
        compile_canvas_document(
            source.replace("</script>", " </script>"),
            runtime_profile="canvas-v2-mermaid-1",
            snapshot=candidate_snapshot,
        )
