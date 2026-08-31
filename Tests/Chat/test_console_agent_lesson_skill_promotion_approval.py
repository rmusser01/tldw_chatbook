"""Console approval wiring for managed-skill lesson proposals."""

from __future__ import annotations

from tldw_chatbook.Agents.agent_lesson_promotion import (
    MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED,
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
from tldw_chatbook.Chat.console_chat_controller import (
    build_managed_skill_promotion_review_hook,
)


def _args() -> dict:
    current = "# Current\n"
    return {
        "skill_name": "safe-editor",
        "skill_public_id": "local:skill:safe-editor",
        "expected_version": 2,
        "expected_trust_state": "trusted",
        "current_sha256": sha256_text(current),
        "replacement_content": "# Current\n\nCheck current state before writing.\n",
        "evidence": {
            "lesson_note_ids": ["note-public-1"],
            "summary": "A stale write erased a concurrent edit.",
            "provenance": "Observed in a deterministic repository test.",
            "verification": "The compare-and-swap regression test passed.",
            "principle": "Do not overwrite state you did not inspect.",
            "rationale": "This is reusable procedural guidance.",
            "procedural": True,
            "reusable": True,
            "independently_verified": True,
        },
    }


def _reader(_name: str) -> dict:
    return {
        "name": "safe-editor",
        "record_id": "local:skill:safe-editor",
        "version": 2,
        "trust_status": "trusted",
        "content": "# Current\n",
    }


def test_hook_uses_exact_call_id_and_approve_once_card() -> None:
    gate = ManagedSkillProposalGate(_reader)
    cards = []
    hook = build_managed_skill_promotion_review_hook(
        gate,
        lambda rows: cards.extend(rows) or {rows[0].call_id: "approve_once"},
    )
    args = _args()
    call = ToolCall(
        PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
        args,
        "skill-proposal-call",
    )
    actor = CurrentRunActor("primary", "run-1", None)

    with use_run_actor(actor):
        verdicts = hook([call], "run-1")
        with use_tool_call_id(call.call_id):
            result = gate.invoke(args)

    assert verdicts == {"skill-proposal-call": "proceed"}
    assert len(cards) == 1
    assert cards[0].call_id == "skill-proposal-call"
    assert cards[0].options == ("approve_once", "deny")
    assert cards[0].arguments["replacement_content"] == args["replacement_content"]
    assert result.ok


def test_hook_denial_cannot_be_reused_by_same_named_call() -> None:
    gate = ManagedSkillProposalGate(_reader)
    hook = build_managed_skill_promotion_review_hook(
        gate, lambda rows: {rows[0].call_id: "deny"}
    )
    args = _args()
    call = ToolCall(
        PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
        args,
        "denied-call",
    )

    with use_run_actor(CurrentRunActor("primary", "run-2", None)):
        verdicts = hook([call], "run-2")
        with use_tool_call_id(call.call_id):
            result = gate.invoke(args)

    assert "denied" in verdicts["denied-call"]
    assert result.error == MANAGED_SKILL_PROMOTION_APPROVAL_REQUIRED
