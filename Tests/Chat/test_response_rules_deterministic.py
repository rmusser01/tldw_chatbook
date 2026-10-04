"""Detectors classify real response content without semantic calls."""

import pytest

from Tests.Chat.response_rules_fixtures import inputs, revision, source
from tldw_chatbook.Chat.response_rules.evaluator import (
    aggregate_checks,
    check_deterministic,
)
from tldw_chatbook.Chat.response_rules.models import RuleCheck


@pytest.mark.parametrize(
    "kind,literals,text,sensitive,want",
    [
        ("include", ["one", "two"], "one only", False, "violation"),
        ("include", ["Straße"], "STRASSE", False, "pass"),
        ("include", ["Straße"], "STRASSE", True, "violation"),
        ("exclude", ["one", "two"], "two", False, "violation"),
        ("exclude", ["one", "two"], "three", False, "pass"),
        (
            "headings",
            ["Evidence", "Results"],
            "# Evidence\r\nResults\r\n-------\r\n",
            False,
            "pass",
        ),
        ("headings", ["Evidence"], "```md\n# Evidence\n```", False, "violation"),
        ("headings", ["Evidence"], "    # Evidence\n", False, "violation"),
        ("headings", ["Evidence"], "Evidence in plain prose", False, "violation"),
    ],
)
def test_literal_case_and_real_markdown_headings(kind, literals, text, sensitive, want):
    rule = revision(
        detector={"kind": kind, "literals": literals, "case_sensitive": sensitive}
    )
    check = check_deterministic(rule, inputs(text))
    assert check.applicability == "applicable"
    assert check.verdict == want


def test_semantic_applicability_cannot_establish_a_literal_violation():
    rule = revision(applicability={"kind": "semantic", "criteria": "Only test reports"})
    assert check_deterministic(rule, inputs("no Evidence")) is None


@pytest.mark.parametrize(
    "checks,want",
    [
        (
            (
                RuleCheck("a", 1, "applicable", "pass", "", ()),
                RuleCheck("b", 1, "unknown", "couldnt_verify", "unavailable", ()),
            ),
            "couldnt_verify",
        ),
        (
            (
                RuleCheck("a", 1, "applicable", "violation", "", ()),
                RuleCheck("b", 1, "unknown", "couldnt_verify", "unavailable", ()),
            ),
            "violation",
        ),
        ((RuleCheck("a", 1, "inapplicable", None, "", ()),), "no_applicable_rules"),
        ((), "no_applicable_rules"),
        ((RuleCheck("a", 1, "applicable", "pass", "", ()),), "pass"),
        ((RuleCheck("a", 1, "unknown", None, "", ()),), "couldnt_verify"),
    ],
)
def test_mixed_results_do_not_turn_unknown_or_skipped_into_pass(checks, want):
    frozen_input = inputs(evidence_complete=False)
    assessment = aggregate_checks(source(), checks, inputs=frozen_input)
    assert assessment.outcome == want
    assert assessment.evidence_complete is False
    assert assessment.evidence_digest
    assert assessment.state == "completed"


def test_evidence_digest_changes_when_evidence_completeness_changes():
    complete = aggregate_checks(source(), (), inputs=inputs())
    incomplete = aggregate_checks(source(), (), inputs=inputs(evidence_complete=False))
    assert complete.evidence_digest != incomplete.evidence_digest
