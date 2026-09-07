"""Selected-profile authoring and executable guide examples."""

from dataclasses import replace

from markdown_it import MarkdownIt

from Tests.Canvas.mermaid_probe import run_mermaid_case
from tldw_chatbook.Canvas.authoring import canvas_authoring_guide
from tldw_chatbook.Canvas.compiler import compile_canvas_document


def test_guide_uses_exact_profile_and_refuses_revoked(candidate_snapshot):
    guide = canvas_authoring_guide(candidate_snapshot, "canvas-v2-mermaid-1")
    assert len(guide.encode("utf-8")) <= 8192
    assert "canvas-v2-mermaid-1" in guide
    v1 = canvas_authoring_guide(candidate_snapshot, "canvas-v1")
    assert "sequenceDiagram" not in v1
    revoked = replace(
        candidate_snapshot,
        profiles=tuple(
            replace(row, executable=False, reason="revoked")
            if row.profile_id == "canvas-v2-mermaid-1"
            else row
            for row in candidate_snapshot.profiles
        ),
    )
    unavailable = canvas_authoring_guide(revoked, "canvas-v2-mermaid-1")
    assert "source-only" in unavailable
    assert "sequenceDiagram" not in unavailable
    assert "source-only" in canvas_authoring_guide(candidate_snapshot, "future-profile")


def test_guide_complete_examples_compile_and_execute(candidate_snapshot):
    guide = canvas_authoring_guide(candidate_snapshot, "canvas-v2-mermaid-1")
    examples = [
        t.content
        for t in MarkdownIt().parse(guide)
        if t.type == "fence" and t.info == "html"
    ]
    assert len(examples) == 2
    for example in examples:
        plan = compile_canvas_document(
            example, runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot
        )
        for diagram in plan.diagrams:
            result = run_mermaid_case({"operation": "parse", "source": diagram.source})
            assert result["ok"] is True, result


def test_provider_context_deduplicates_exact_historical_guides(candidate_snapshot):
    import json

    from Tests.Agents.test_canvas_tool_provider import _provider
    from tldw_chatbook.Agents.canvas_tool_provider import build_canvas_runtime_guidance

    provider, coordinator, _ = _provider()
    coordinator.profile_snapshot = candidate_snapshot
    schema = provider.load_schema("canvas:canvas_create")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "read-1",
                    "type": "function",
                    "function": {"name": "canvas_read", "arguments": "{}"},
                },
                {
                    "id": "read-2",
                    "type": "function",
                    "function": {"name": "canvas_read", "arguments": "{}"},
                },
            ],
        }
    ]
    messages += [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {"status": "ok", "canvas": {"runtime_profile": "future-profile"}}
            ),
        }
        for call_id in ("read-1", "read-2")
    ]
    guide = build_canvas_runtime_guidance([schema, schema], messages=messages)
    assert guide.count("Canvas profile canvas-v2-mermaid-1:") == 1
    assert guide.count("Canvas profile future-profile:") == 1
    assert "source-only" in guide
    assert "sequenceDiagram" not in repr(schema)
    forged = [{"role": "user", "content": messages[1]["content"]}]
    assert "future-profile" not in build_canvas_runtime_guidance(
        [schema], messages=forged
    )
    from Tests.Agents.test_history_projection import _fence_call, _fence_result

    fence = [
        _fence_call("canvas_read", {"canvas_id": "selected"}),
        _fence_result("canvas_read", messages[1]["content"]),
    ]
    assert (
        "Canvas profile future-profile: source-only"
        in build_canvas_runtime_guidance([schema], messages=fence)
    )
    assert "future-profile" not in build_canvas_runtime_guidance(
        [schema], messages=fence[1:]
    )
    from tldw_chatbook.Agents.native_tools import schemas_to_openai_tools

    wire = schemas_to_openai_tools([schema])[0]
    assert set(wire["function"]) == {"name", "description", "parameters"}
    assert "_guides" not in json.dumps(wire)
    assert wire["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 4096},
            "html": {"type": "string", "maxLength": 524288},
        },
        "required": ["title", "html"],
        "additionalProperties": False,
    }
