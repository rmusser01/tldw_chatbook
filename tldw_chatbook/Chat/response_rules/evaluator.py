"""Native checks and honest combined assessment outcomes."""

from typing import Literal
from uuid import uuid4

from markdown_it import MarkdownIt

from .models import RuleAssessment, RuleCheck, RuleInput, RuleRevision, RuleSource


def _headings(text: str) -> tuple[str, ...]:
    tokens = MarkdownIt("commonmark").parse(text)
    return tuple(
        tokens[i + 1].content
        for i, token in enumerate(tokens[:-1])
        if token.type == "heading_open" and tokens[i + 1].type == "inline"
    )


def check_deterministic(rule: RuleRevision, inputs: RuleInput) -> RuleCheck | None:
    """Evaluate one native predicate, or return None if semantic work is needed."""
    candidate = rule.candidate
    detector = candidate.detector
    if candidate.applicability.kind == "semantic" or detector.kind == "semantic":
        return None
    text = inputs.response_text
    literals = detector.literals
    if not detector.case_sensitive:
        text = text.casefold()
        literals = tuple(literal.casefold() for literal in literals)
    if detector.kind == "include":
        passed = all(literal in text for literal in literals)
    elif detector.kind == "exclude":
        passed = not any(literal in text for literal in literals)
    else:
        headings = _headings(inputs.response_text)
        if not detector.case_sensitive:
            headings = tuple(heading.casefold() for heading in headings)
        passed = all(literal in headings for literal in literals)
    return RuleCheck(
        rule.rule_id,
        rule.revision,
        "applicable",
        "pass" if passed else "violation",
        "native_text_match" if passed else "native_text_mismatch",
        (),
    )


def aggregate_checks(
    source: RuleSource,
    checks: tuple[RuleCheck, ...],
    *,
    inputs: RuleInput,
) -> RuleAssessment:
    """Retain confirmed violations without converting unknown/skipped to pass."""
    outcome: Literal["pass", "violation", "couldnt_verify", "no_applicable_rules"]
    if any(
        c.applicability == "applicable" and c.verdict == "violation" for c in checks
    ):
        outcome = "violation"
    elif any(
        c.applicability == "unknown"
        or (c.applicability == "applicable" and c.verdict != "pass")
        for c in checks
    ):
        outcome = "couldnt_verify"
    elif not any(c.applicability == "applicable" for c in checks):
        outcome = "no_applicable_rules"
    else:
        outcome = "pass"
    return RuleAssessment(
        str(uuid4()),
        source,
        tuple(checks),
        inputs.evidence_complete,
        inputs.evidence_digest(),
        outcome,
        "completed",
    )
