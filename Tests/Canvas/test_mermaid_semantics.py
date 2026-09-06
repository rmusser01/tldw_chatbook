"""Exact semantic behavior of the packaged, QuickJS-only Mermaid subset."""

import hashlib
import json
from pathlib import Path

import pytest

from Tests.Canvas.mermaid_probe import run_mermaid_case


@pytest.mark.parametrize(
    "source",
    [
        "flowchart TD\nA[Start] --> B[Finish]",
        "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: Hello",
    ],
)
def test_admitted_models_use_quickjs(source):
    result = run_mermaid_case({"operation": "parse", "source": source})
    assert result["ok"] is True, result
    assert result["model"]["kind"] in {"flow", "sequence"}


CASES = json.loads(
    (Path(__file__).parent / "fixtures/mermaid/semantics.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_exact_semantic_models_and_explicit_refusals(case):
    result = run_mermaid_case({"operation": "parse", "source": case["source"]})
    if "model" in case:
        assert result == {"ok": True, "model": case["model"], "error": None}
    else:
        assert result["ok"] is False, result
        assert result["error"]["code"] == case["error"]
        assert set(result["error"]) == {"code", "ordinal", "line", "column"}
        assert result["error"]["ordinal"] == 1


def test_pinned_graphemes_and_width_preserve_codepoints():
    result = run_mermaid_case(
        {
            "operation": "text",
            "sources": ["e\u0301", "👨‍👩‍👧‍👦", "🇦🇧🇨", "क्ष", "中", "\r\n", "א"],
        }
    )
    assert result["model"] == [
        {"clusters": ["e\u0301"], "widths": [8]},
        {"clusters": ["👨‍👩‍👧‍👦"], "widths": [16]},
        {"clusters": ["🇦🇧", "🇨"], "widths": [16, 16]},
        {"clusters": ["क्ष"], "widths": [8]},
        {"clusters": ["中"], "widths": [16]},
        {"clusters": ["\r\n"], "widths": [0]},
        {"clusters": ["א"], "widths": [8]},
    ]


@pytest.mark.parametrize(
    "source,code",
    [
        ("flowchart TD\n" + "\n".join(f"N{i}" for i in range(17)), "nodes-limit"),
        ("flowchart TD\n" + "\n".join("A-->B" for _ in range(25)), "edges-limit"),
        (
            "sequenceDiagram\n" + "\n".join(f"participant N{i}" for i in range(7)),
            "participants-limit",
        ),
        (
            "sequenceDiagram\nparticipant A\nparticipant B\n" + "A->>B: Hi\n" * 17,
            "messages-limit",
        ),
        ("sequenceDiagram\nparticipant A\n" + "Note over A: Hi\n" * 9, "notes-limit"),
        ('flowchart TD\nA["' + "é" * 257 + '"]', "label-limit"),
        ("flowchart TD\nA\n%%" + "x" * 8192, "input-limit"),
    ],
)
def test_individual_parse_limits(source, code):
    result = run_mermaid_case({"operation": "parse", "source": source})
    assert result["ok"] is False, result
    assert result["error"]["code"] == code


def test_document_budget_is_shared_across_declarations():
    source = "flowchart TD\n" + "\n".join(f"N{i}" for i in range(13))
    result = run_mermaid_case({"operation": "parse", "sources": [source, source]})
    assert result["error"] == {
        "code": "nodes-limit",
        "ordinal": 2,
        "line": None,
        "column": None,
    }
    result = run_mermaid_case(
        {"operation": "parse", "sources": ["flowchart TD\nA"] * 5}
    )
    assert result["error"]["code"] == "declaration-limit"
    assert result["error"]["ordinal"] == 5


def test_unicode_16_official_extended_grapheme_conformance():
    payload = (
        Path(__file__).parent / "fixtures/mermaid/GraphemeBreakTest-16.0.0.txt"
    ).read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "ee2b9354d270ac061b29f09662cafea06341d77e704b8cc6bd72aaeeda363cb5"
    )
    cases = []
    for line in payload.decode().splitlines():
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        clusters, current = [], ""
        for token in tokens:
            if token == "÷":
                if current:
                    clusters.append(current)
                    current = ""
            elif token != "×":
                current += chr(int(token, 16))
        assert not current
        cases.append(clusters)
    assert len(cases) == 1093
    for start in range(0, len(cases), 24):
        expected = cases[start : start + 24]
        result = run_mermaid_case(
            {"operation": "text", "sources": ["".join(c) for c in expected]}
        )
        assert [row["clusters"] for row in result["model"]] == expected


@pytest.mark.parametrize(
    "source",
    [
        "sequenceDiagram\nparticipant A\n# discarded upstream",
        "flowchart TD\nA\nflowchart LR\nB",
        "flowchart TD\nA\ngraph LR\nB",
    ],
)
def test_non_admitted_operations_cannot_be_silently_discarded(source):
    result = run_mermaid_case({"operation": "parse", "source": source})
    assert result["ok"] is False
    assert result["error"]["code"] == "unsupported-syntax"


@pytest.mark.parametrize("count,code", [(16, None), (17, "nodes-limit")])
def test_flow_node_boundary_is_inclusive(count, code):
    source = "flowchart TD\n" + "\n".join(f"N{i}" for i in range(count))
    result = run_mermaid_case({"operation": "parse", "source": source})
    if code:
        assert result["error"]["code"] == code
    else:
        assert len(result["model"]["nodes"]) == 16
        assert result["model"]["nodes"][-1] == {
            "id": "N15",
            "label": "N15",
            "shape": "rect",
        }


@pytest.mark.parametrize(
    "source,repeats,code",
    [
        ("flowchart TD\n" + "A-->B\n" * 17, 2, "edges-limit"),
        (
            "sequenceDiagram\n" + "\n".join(f"participant N{i}" for i in range(5)),
            2,
            "participants-limit",
        ),
        (
            "sequenceDiagram\nparticipant A\nparticipant B\n" + "A->>B: Hi\n" * 13,
            2,
            "messages-limit",
        ),
        (
            "sequenceDiagram\nparticipant A\n" + "Note over A: Hi\n" * 7,
            2,
            "notes-limit",
        ),
        ("flowchart TD\nA\n%%" + "x" * 6000, 3, "input-limit"),
        (
            "flowchart TD\n"
            + "\n".join(f'N{i}["' + "x" * 400 + '"]' for i in range(7)),
            3,
            "labels-limit",
        ),
    ],
    ids=["edges", "participants", "messages", "notes", "input", "labels"],
)
def test_each_aggregate_parse_budget_is_charged(source, repeats, code):
    result = run_mermaid_case({"operation": "parse", "sources": [source] * repeats})
    assert result["ok"] is False
    assert result["error"]["code"] == code
    assert result["error"]["ordinal"] == repeats


@pytest.mark.parametrize("count,ok", [(8, True), (9, False)])
def test_combined_labels_per_diagram_have_an_inclusive_byte_ceiling(count, ok):
    source = "flowchart TD\n" + "\n".join(
        f'N{i}["' + "x" * 512 + '"]' for i in range(count)
    )
    result = run_mermaid_case({"operation": "parse", "source": source})
    assert result["ok"] is ok
    if not ok:
        assert result["error"]["code"] == "labels-limit"


def test_four_small_diagrams_share_one_startup_vm():
    result = run_mermaid_case(
        {"operation": "parse", "sources": ["flowchart TD\nA-->B"] * 4}
    )
    assert result["ok"] is True
    assert len(result["model"]) == 4


def test_keycap_emoji_uses_the_pinned_vs16_width_rule():
    result = run_mermaid_case({"operation": "text", "sources": ["1️⃣"]})
    assert result["model"] == [{"clusters": ["1️⃣"], "widths": [16]}]


@pytest.mark.parametrize(
    "label", ["*italic*", "_italic_", "wrap: Text", "nowrap: Text"]
)
def test_sequence_formatting_and_parser_wrap_directives_are_not_plain_labels(label):
    result = run_mermaid_case(
        {
            "operation": "parse",
            "source": "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: " + label,
        }
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "unsupported-label"


def test_prototype_like_participant_ids_are_safe_default_labels():
    result = run_mermaid_case(
        {
            "operation": "parse",
            "source": "sequenceDiagram\nparticipant __proto__\nparticipant constructor\n__proto__->>constructor: Hi",
        }
    )
    assert result["ok"] is True
    assert result["model"]["participants"] == [
        {"id": "__proto__", "label": "__proto__"},
        {"id": "constructor", "label": "constructor"},
    ]


@pytest.mark.parametrize("direction", ["TD", "LR"])
def test_six_node_branch_rejoin_preserves_source_order(direction):
    result = run_mermaid_case(
        {
            "operation": "parse",
            "source": f"flowchart {direction}\nA-->B\nB-->|Yes|C\nB-->|No|D\nC-->E\nD-->E\nE-->F",
        }
    )
    assert result["ok"] is True, result
    assert result["model"] == {
        "kind": "flow",
        "direction": direction,
        "nodes": [
            {"id": "A", "label": "A", "shape": "rect"},
            {"id": "B", "label": "B", "shape": "rect"},
            {"id": "C", "label": "C", "shape": "rect"},
            {"id": "D", "label": "D", "shape": "rect"},
            {"id": "E", "label": "E", "shape": "rect"},
            {"id": "F", "label": "F", "shape": "rect"},
        ],
        "edges": [
            {"from": "A", "to": "B", "label": ""},
            {"from": "B", "to": "C", "label": "Yes"},
            {"from": "B", "to": "D", "label": "No"},
            {"from": "C", "to": "E", "label": ""},
            {"from": "D", "to": "E", "label": ""},
            {"from": "E", "to": "F", "label": ""},
        ],
    }
