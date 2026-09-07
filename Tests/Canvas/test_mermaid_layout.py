"""Geometry contracts exercised in the packaged QuickJS runtime."""

import json
from pathlib import Path

import pytest

from Tests.Canvas.mermaid_probe import run_mermaid_case

CASES = json.loads((Path(__file__).parent / "fixtures/mermaid/layout.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_reviewed_exact_scene_and_model_corpus(case):
    assert (
        run_mermaid_case({"operation": "layout", "source": case["source"]})
        == case["expected"]
    )


def test_private_render_entry_shares_budget_and_returns_no_partial_scenes():
    sources = [
        "flowchart TD\nA --> B",
        "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: Hi",
    ]
    result = run_mermaid_case({"operation": "render", "sources": sources})
    assert result["ok"]
    assert (
        result["scene"]
        == run_mermaid_case({"operation": "layout", "sources": sources})["scene"]
    )
    result = run_mermaid_case({"operation": "render", "sources": [sources[0]] * 5})
    assert result["error"]["code"] == "declaration-limit"
    assert result["error"]["ordinal"] == 5
    assert result["scene"] == []


def walk(node):
    yield node
    for child in node["children"]:
        yield from walk(child)


@pytest.mark.parametrize("side", ["left", "right", "over"])
def test_sequence_notes_reserve_the_requested_side(side):
    result = run_mermaid_case(
        {
            "operation": "layout",
            "source": "sequenceDiagram\nparticipant A\nparticipant B\n"
            f"Note {side + ' of' if side != 'over' else side} A: note\nA->>B: later",
        }
    )
    assert result["ok"], result
    texts = {
        n["text"]: dict(n["attributes"])
        for n in walk(result["scene"]["root"])
        if n["tag"] == "text"
    }
    note_x, actor_x = float(texts["note"]["x"]), float(texts["A"]["x"])
    assert {
        "left": note_x < actor_x,
        "right": note_x > actor_x,
        "over": abs(note_x - actor_x) < 64,
    }[side]
    assert float(texts["later"]["y"]) > float(texts["note"]["y"]) + 24


def test_scene_preserves_inspectable_source_and_exact_serialized_bytes():
    import json

    source = "flowchart TD\nA[Hello] --> B[World]"
    result = run_mermaid_case({"operation": "layout", "source": source})
    scene = result["scene"]
    assert any(n["tag"] == "pre" and n["text"] == source for n in walk(scene["root"]))
    assert scene["metrics"]["output"] == len(
        json.dumps(scene, ensure_ascii=False, separators=(",", ":")).encode()
    )


def test_four_small_and_mixed_diagrams_fit_worker_patch_budget():
    flow = "flowchart TD\nA[Start] --> B[End]"
    sequence = "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: Hello"
    for sources in ([flow] * 4, [flow, sequence, flow, sequence]):
        result = run_mermaid_case({"operation": "layout", "sources": sources})
        assert result["ok"], result
        patches = 0
        for scene in result["scene"]:
            for node in walk(scene["root"]):
                patches += 2 + bool(node["text"])
                for key, value in node["attributes"]:
                    patches += len(value.split(";")) if key == "style" else 1
        assert patches <= 500


@pytest.mark.parametrize("direction", ["TD", "TB", "LR"])
def test_long_edge_labels_have_reserved_lanes(direction):
    source = f"flowchart {direction}\nA -->|{'x' * 128}| B"
    result = run_mermaid_case({"operation": "layout", "source": source})
    assert result["ok"], result
    lines = [
        n["text"]
        for n in walk(result["scene"]["root"])
        if n["tag"] == "text" and set(n["text"]) == {"x"}
    ]
    assert "".join(lines) == "x" * 128


@pytest.mark.parametrize(
    "scope,kind,limit",
    [
        ("diagram", "work", 10000),
        ("document", "work", 20000),
        ("diagram", "elements", 250),
        ("document", "elements", 400),
        ("diagram", "output", 49152),
        ("document", "output", 65536),
        ("diagram", "area", 4194304),
        ("document", "area", 8388608),
    ],
)
def test_layout_refuses_each_exhausted_budget_scope(scope, kind, limit):
    result = run_mermaid_case(
        {
            "operation": "layout",
            "source": "flowchart TD\nA",
            "seed": {scope: {kind: limit}},
        }
    )
    assert result["ok"] is False
    assert result["error"]["code"] == kind + "-limit"
    assert result["scene"] is None


@pytest.mark.parametrize("direction,label", [("LR", "a" * 120), ("TD", "a" * 512)])
def test_geometry_extent_refuses_without_truncating(direction, label):
    source = (
        f"flowchart {direction}\n"
        + "\n".join(f'N{i}["{label}"]' for i in range(8))
        + "\n"
        + "\n".join(f"N{i}-->N{i + 1}" for i in range(7))
    )
    result = run_mermaid_case({"operation": "layout", "source": source})
    assert not result["ok"]
    assert result["error"]["code"] in {
        "geometry-limit",
        "elements-limit",
        "output-limit",
    }
    assert result["scene"] is None


@pytest.mark.parametrize(
    "label", ["中" * 12, "é" * 24, "👨‍👩‍👧‍👦" * 4, "שלום" * 8, "a" * 512]
)
def test_unicode_wrap_preserves_clusters_and_source(label):
    source = f'flowchart TD\nA["{label}"]'
    result = run_mermaid_case({"operation": "layout", "source": source})
    assert result["ok"], result
    lines = [n["text"] for n in walk(result["scene"]["root"]) if n["tag"] == "text"]
    assert "".join(lines) == label
    assert all(
        not line.startswith(("\u0301", "\u200d")) and not line.endswith("\u200d")
        for line in lines
    )


def test_over_two_participants_note_spans_both_columns():
    result = run_mermaid_case(
        {
            "operation": "layout",
            "source": "sequenceDiagram\nparticipant A\nparticipant B\nparticipant C\nNote over A,C: Shared",
        }
    )
    svg = result["scene"]["root"]["children"][1]
    note = next(
        n
        for n in svg["children"]
        if n["tag"] == "path" and dict(n["attributes"]).get("fill") == "white"
    )
    assert dict(note["attributes"])["d"] == "M264 104H872V152H264Z"


def test_branch_rejoin_geometry_is_deterministic():
    case = {
        "operation": "layout",
        "source": "flowchart TD\nA[Start] --> B{Ready?}\n"
        "B -->|Yes| C(Continue)\nB -->|No| D[Revise]\n"
        "C --> E[Join]\nD --> E\nE --> F[End]",
    }
    first = run_mermaid_case(case)
    assert first["ok"] is True
    assert "scene" in first
    assert first["scene"] == run_mermaid_case(case)["scene"]
    assert 0 < first["scene"]["width"] <= 2048
    assert 0 < first["scene"]["height"] <= 4096
