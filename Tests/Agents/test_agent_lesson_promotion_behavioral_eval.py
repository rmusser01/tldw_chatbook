"""Scripted prompt evidence for useful, restrained lesson promotions.

These fixtures show that the shipped guidance supports the intended choices in
the repository's deterministic fake-model harness.  They are prompt evidence,
not authorization or a model-general guarantee; boundary tests enforce safety.
"""

from __future__ import annotations

import json

from tldw_chatbook.Agents.agent_lesson_promotion import (
    build_agent_lesson_promotion_guidance,
)
from tldw_chatbook.Agents.agent_models import ToolSchema

from Tests.Agents.test_agent_service import ScriptedChat


def _schema(name: str) -> ToolSchema:
    return ToolSchema(
        id=f"eval:{name}",
        name=name,
        description=name,
        parameters={"type": "object"},
    )


SCHEMAS = tuple(
    _schema(name)
    for name in (
        "library_search_notes",
        "library_get_note",
        "fs_write",
        "prepare_managed_skill_promotion",
    )
)


def _scripted_decision(payload: dict, *, role: str = "primary") -> tuple[dict, str]:
    guidance = build_agent_lesson_promotion_guidance(
        SCHEMAS,
        trusted_role=role,  # type: ignore[arg-type]
        repository_target_enabled=True,
    )
    chat = ScriptedChat([json.dumps(payload)])
    response = chat(
        api_endpoint="openai",
        messages_payload=[{"role": "system", "content": guidance}],
        streaming=False,
        model="test-model",
    )
    return json.loads(response["choices"][0]["message"]["content"]), guidance


def test_one_strong_verified_signal_can_nominate_one_small_principled_edit() -> None:
    decision, guidance = _scripted_decision(
        {
            "decision": "nominate",
            "incident_count_required": False,
            "target": "AGENTS.md",
            "candidate": "Re-check the target digest immediately before replacement.",
            "principle": "Validate mutable preconditions at the mutation boundary.",
            "rationale": "An earlier read cannot protect a later write from races.",
            "unknowns": ["Whether external writers use the same lock"],
        }
    )

    assert "One strong signal may qualify" in guidance
    assert decision["decision"] == "nominate"
    assert decision["incident_count_required"] is False
    assert decision["candidate"].count("\n") == 0
    assert decision["principle"] and decision["rationale"]
    assert decision["unknowns"]


def test_weak_or_contradictory_evidence_does_not_nominate() -> None:
    decision, guidance = _scripted_decision(
        {
            "decision": "do_not_nominate",
            "reason": "evidence is contradictory and not independently verified",
            "tool_calls": [],
        }
    )

    assert "contradictory reports do not" in guidance
    assert decision == {
        "decision": "do_not_nominate",
        "reason": "evidence is contradictory and not independently verified",
        "tool_calls": [],
    }


def test_managed_skill_candidate_stays_manual_and_does_not_accumulate_rules() -> None:
    decision, guidance = _scripted_decision(
        {
            "decision": "propose_manual_skill_edit",
            "candidate_sections": ["one focused principle", "why it works"],
            "application": "Library > Skills",
            "automatic_mutation": False,
            "requires_retrust": True,
        }
    )

    assert "smallest focused edit" in guidance
    assert "Library > Skills" in guidance
    assert decision["candidate_sections"] == [
        "one focused principle",
        "why it works",
    ]
    assert decision["automatic_mutation"] is False
    assert decision["requires_retrust"] is True


def test_child_hands_evidence_to_primary_without_review_or_write() -> None:
    decision, guidance = _scripted_decision(
        {
            "action": "return_candidate",
            "evidence_ids": ["note-public-1"],
            "candidate": "Check the current version before saving.",
            "tool_calls": [],
        },
        role="subagent",
    )

    assert "Do not present a promotion approval card" in guidance
    assert decision["action"] == "return_candidate"
    assert decision["tool_calls"] == []
