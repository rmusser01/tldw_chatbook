"""Hand-authored inputs for native response-rule tests."""

from tldw_chatbook.Chat.response_rules.models import (
    RuleCandidate,
    RuleInput,
    RuleRevision,
    RuleSource,
)


def source(**changes):
    values = {
        "profile_id": "profile",
        "session_id": "chat",
        "conversation_id": "conversation",
        "branch_id": "branch",
        "message_id": "answer",
        "message_version": 1,
        "parent_turn_id": "turn",
        "operation_id": "operation",
        "settlement_id": "settlement",
        "effective_digest": "rules",
        "authority_epoch": 1,
    }
    return RuleSource(**(values | changes))


def candidate(**changes):
    values = {
        "title": "Show evidence",
        "applicability": {"kind": "always"},
        "detector": {
            "kind": "include",
            "literals": ["Evidence"],
            "case_sensitive": False,
        },
        "feedback": "Include the evidence.",
    }
    return RuleCandidate.model_validate(values | changes)


def revision(rule_id="rule", number=1, **changes):
    return RuleRevision(
        rule_id=rule_id,
        revision=number,
        candidate=candidate(**changes),
        schema_version=1,
        origin=source(),
        created_at="2026-10-04T00:00:00Z",
    )


def inputs(text="Evidence", **changes):
    values = {
        "request_text": "Explain the result",
        "response_text": text,
        "evidence": (),
        "evidence_complete": True,
        "work_revision": None,
    }
    return RuleInput(**(values | changes))
