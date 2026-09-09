"""Model JSON is data, never execution or verification authority."""

import importlib
import json

import pytest


def iteration():
    spec = importlib.util.find_spec("tldw_chatbook.Agents.goal_iteration")
    assert spec is not None, "goal report parser missing"
    return importlib.import_module(spec.name)


def report(**changes):
    body = {
        "summary": "observed",
        "learnings": [],
        "next_action": "",
        "candidate_draft": "",
        "evidence_ids": [],
        "completion_recommended": False,
    }
    body.update(changes)
    return json.dumps(body, ensure_ascii=False)


@pytest.mark.parametrize(
    "changes",
    [
        {"completion_recommended": 1},
        {"completion_recommended": "true"},
        {"summary": []},
        {"learnings": "x"},
        {"evidence_ids": ["../secret"]},
        {"evidence_ids": ["https://foreign"]},
        {"verified": True},
        {"budget": 1},
        {"provider": "x"},
        {"workspace_root": "/"},
        {"permission": True},
        {"learnings": ["x"] * 9},
        {"evidence_ids": ["a"] * 33},
        {"candidate_draft": "é" * 16385},
    ],
)
def test_report_rejects_coercion_authority_and_unbounded_payload(changes):
    mod = iteration()
    with pytest.raises(ValueError):
        mod.parse_iteration_report(report(**changes))


@pytest.mark.parametrize(
    "text",
    ["{}", "```json\n{}\n```", "{", " " * 65537, '{"summary":"x","summary":"y"}'],
)
def test_report_requires_exact_complete_json(text):
    mod = iteration()
    with pytest.raises(ValueError):
        mod.parse_iteration_report(text)


def test_report_accepts_utf8_bounded_draft_and_strict_bool():
    parsed = iteration().parse_iteration_report(
        report(candidate_draft="é" * 16384, completion_recommended=True)
    )
    assert parsed.completion_recommended is True
    assert len(parsed.candidate_draft.encode()) == 32768
