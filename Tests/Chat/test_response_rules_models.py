"""Boundary tests catch executable/private payloads and oversized definitions."""

import dataclasses

import pytest
from pydantic import ValidationError

from Tests.Chat.response_rules_fixtures import candidate, inputs
from tldw_chatbook.Chat.response_rules.models import RuleLearningResult


@pytest.mark.parametrize(
    "extra",
    [
        {"rule_id": "model-chosen"},
        {"source": {"session_id": "other"}},
        {"command": ["python", "bad.py"]},
        {"origin": "forged"},
    ],
)
def test_private_fields_and_nested_predicates_are_rejected(extra):
    with pytest.raises(ValidationError):
        candidate(**extra)


@pytest.mark.parametrize(
    "detector",
    [
        {"kind": "include", "literals": ["a"], "case_sensitive": False, "children": []},
        {"kind": "regex", "criteria": ".*", "case_sensitive": False},
        {
            "kind": "include",
            "literals": ["a"],
            "criteria": "irrelevant",
            "case_sensitive": False,
        },
        {
            "kind": "semantic",
            "criteria": "Be clear",
            "literals": ["a"],
            "case_sensitive": False,
        },
        {"kind": "include", "literals": ["a"], "case_sensitive": "false"},
        {"kind": "include", "literals": [], "case_sensitive": False},
        {"kind": "include", "literals": [" "], "case_sensitive": False},
        {"kind": "include", "literals": ["a"]},
    ],
)
def test_closed_predicate_shape(detector):
    with pytest.raises(ValidationError):
        candidate(detector=detector)


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", "a" * 121),
        ("feedback", "é" * 1025),
        ("applicability", {"kind": "semantic", "criteria": "é" * 513}),
        ("applicability", {"kind": "always", "criteria": "hidden predicate"}),
        (
            "detector",
            {"kind": "include", "literals": ["é" * 129], "case_sensitive": False},
        ),
        (
            "detector",
            {
                "kind": "include",
                "literals": [str(n) for n in range(17)],
                "case_sensitive": False,
            },
        ),
        (
            "detector",
            {
                "kind": "headings",
                "literals": [str(n) for n in range(9)],
                "case_sensitive": False,
            },
        ),
        (
            "detector",
            {"kind": "semantic", "criteria": "é" * 1025, "case_sensitive": False},
        ),
    ],
)
def test_exact_utf8_and_item_caps(field, value):
    with pytest.raises(ValidationError):
        candidate(**{field: value})


def test_caps_allow_exact_boundary_and_freeze_nested_collections():
    literals = ["é" * 128]
    rule = candidate(
        title="a" * 120,
        feedback="é" * 1024,
        detector={"kind": "include", "literals": literals, "case_sensitive": False},
    )
    literals.clear()
    assert rule.detector.literals == ("é" * 128,)
    with pytest.raises(ValidationError):
        rule.detector.case_sensitive = True


def test_definition_cap_counts_serialized_payload():
    with pytest.raises(ValidationError):
        candidate(
            feedback="f" * 2048,
            applicability={"kind": "semantic", "criteria": "a" * 1024},
            detector={
                "kind": "include",
                "literals": ["\\" * 256] * 16,
                "case_sensitive": False,
            },
        )


def test_host_inputs_are_frozen_and_private_bodies_do_not_leak_in_repr():
    original = inputs("PRIVATE RESPONSE", request_text="PRIVATE PROMPT")
    fixtures = {"case": original}
    result = RuleLearningResult(
        state="inactive",
        rule=None,
        validation=None,
        fixtures=fixtures,
        reason="invalid",
    )
    fixtures.clear()
    assert result.fixtures["case"] is original
    with pytest.raises(TypeError):
        result.fixtures["other"] = original
    with pytest.raises(dataclasses.FrozenInstanceError):
        original.response_text = "replaced"
    assert "PRIVATE" not in repr(original)
    assert "PRIVATE" not in repr(result)
    assert "Include the evidence" not in repr(candidate())
