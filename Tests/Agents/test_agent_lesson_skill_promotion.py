"""Managed local-skill Agent Lesson proposals remain exact and read-only."""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_lesson_promotion import (
    MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED,
    MANAGED_SKILL_PROMOTION_FOREGROUND_REQUIRED,
    MANAGED_SKILL_PROMOTION_STALE,
    ManagedSkillProposalGate,
    sha256_text,
)
from tldw_chatbook.Agents.agent_models import (
    PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
    ToolCall,
)
from tldw_chatbook.Agents.run_context import (
    CurrentRunActor,
    use_run_actor,
    use_tool_call_id,
)


RUN_ID = "managed-promotion-run"
CALL_ID = "managed-promotion-call"
CURRENT = "# Existing skill\n"
REPLACEMENT = "# Existing skill\n\nUse compare-and-swap before replacement.\n"


def _evidence() -> dict:
    return {
        "lesson_note_ids": ["note-public-1"],
        "summary": "Stale replacement can erase concurrent edits.",
        "provenance": "Observed while editing repository instructions.",
        "verification": "A deterministic race test passed.",
        "principle": "Preserve unrelated user changes.",
        "rationale": "The same guard is reusable in this procedural skill.",
        "procedural": True,
        "reusable": True,
        "independently_verified": True,
    }


def _args() -> dict:
    return {
        "skill_name": "safe-editor",
        "skill_public_id": "local:skill:safe-editor",
        "expected_version": 3,
        "expected_trust_state": "trusted",
        "current_sha256": sha256_text(CURRENT),
        "replacement_content": REPLACEMENT,
        "evidence": _evidence(),
    }


def _skill_state() -> dict:
    return {
        "name": "safe-editor",
        "record_id": "local:skill:safe-editor",
        "version": 3,
        "trust_status": "trusted",
        "content": CURRENT,
    }


def _approve(gate: ManagedSkillProposalGate, args: dict) -> ToolCall:
    call = ToolCall(
        PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
        args,
        CALL_ID,
    )
    gate.apply_decisions(RUN_ID, [call], {CALL_ID: "approve_once"})
    return call


def test_exact_approved_request_returns_read_only_single_use_proposal() -> None:
    state = _skill_state()
    gate = ManagedSkillProposalGate(lambda _name: dict(state))
    args = _args()
    actor = CurrentRunActor("primary", RUN_ID, None)

    with use_run_actor(actor):
        card = gate.pending_gate_for(
            PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
            args,
            run_id=RUN_ID,
            call_id=CALL_ID,
        )
        _approve(gate, args)
        with use_tool_call_id(CALL_ID):
            result = gate.invoke(args)
            reused = gate.invoke(args)

    assert card is not None
    assert card.options == ("approve_once", "deny")
    assert card.arguments["replacement_content"] == REPLACEMENT
    assert result.ok
    proposal = json.loads(result.content)
    assert proposal["mode"] == "proposal_only"
    assert proposal["skill_public_id"] == "local:skill:safe-editor"
    assert proposal["expected_version"] == 3
    assert proposal["expected_trust_state"] == "trusted"
    assert proposal["current_sha256"] == sha256_text(CURRENT)
    assert proposal["replacement_content"] == REPLACEMENT
    assert state == _skill_state()
    assert reused.error == MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("record_id", "local:skill:replacement"),
        ("version", 4),
        ("trust_status", "needs_review"),
        ("content", "# Concurrent edit\n"),
    ],
)
def test_state_change_after_approval_returns_content_free_stale_refusal(
    changed_field: str, changed_value: object
) -> None:
    state = _skill_state()
    gate = ManagedSkillProposalGate(lambda _name: dict(state))
    args = _args()
    actor = CurrentRunActor("primary", RUN_ID, None)
    _approve(gate, args)
    state[changed_field] = changed_value

    with use_run_actor(actor), use_tool_call_id(CALL_ID):
        result = gate.invoke(args)

    assert not result.ok
    assert result.error == MANAGED_SKILL_PROMOTION_STALE
    assert "Concurrent edit" not in result.error


def test_denial_wrong_call_and_lifecycle_clear_leave_no_usable_stamp() -> None:
    gate = ManagedSkillProposalGate(lambda _name: _skill_state())
    args = _args()
    call = ToolCall(
        PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
        args,
        CALL_ID,
    )
    actor = CurrentRunActor("primary", RUN_ID, None)

    gate.apply_decisions(RUN_ID, [call], {CALL_ID: "deny"})
    with use_run_actor(actor), use_tool_call_id(CALL_ID):
        assert gate.invoke(args).error == MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED

    _approve(gate, args)
    with use_run_actor(actor), use_tool_call_id("different-call"):
        assert gate.invoke(args).error == MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED

    gate.clear(RUN_ID)
    with use_run_actor(actor), use_tool_call_id(CALL_ID):
        assert gate.invoke(args).error == MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED

    gate.unbind_reader()
    assert not gate.available


def test_subagent_cannot_present_or_invoke_managed_skill_proposal() -> None:
    gate = ManagedSkillProposalGate(lambda _name: _skill_state())
    args = _args()
    actor = CurrentRunActor("subagent", RUN_ID, "parent-run")

    with use_run_actor(actor):
        assert (
            gate.pending_gate_for(
                PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
                args,
                run_id=RUN_ID,
                call_id=CALL_ID,
            )
            is None
        )
        with use_tool_call_id(CALL_ID):
            result = gate.invoke(args)

    assert result.error == MANAGED_SKILL_PROMOTION_FOREGROUND_REQUIRED


def test_direct_unbound_invocation_requires_exact_approval() -> None:
    gate = ManagedSkillProposalGate(lambda _name: _skill_state())

    with use_tool_call_id(CALL_ID):
        result = gate.invoke(_args())

    assert result.error == MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED


@pytest.mark.parametrize(
    "mutation",
    [
        lambda args: args["evidence"].update(lesson_note_ids="note-public-1"),
        lambda args: args["evidence"].update(procedural=1),
        lambda args: args.update(expected_version=True),
        lambda args: args.update(current_sha256="A" * 64),
        lambda args: args.update(extra="unexpected"),
    ],
)
def test_malformed_request_never_produces_an_approval_card(mutation) -> None:
    gate = ManagedSkillProposalGate(lambda _name: _skill_state())
    args = _args()
    mutation(args)

    with use_run_actor(CurrentRunActor("primary", RUN_ID, None)):
        card = gate.pending_gate_for(
            PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
            args,
            run_id=RUN_ID,
            call_id=CALL_ID,
        )

    assert card is None
