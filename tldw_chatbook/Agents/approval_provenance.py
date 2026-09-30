"""Ephemeral answer provenance stored with existing approval stamps."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from tldw_chatbook.Agents.agent_models import ApprovalDecision


class ApprovalDecisions(dict[str, str]):
    """One approval round's verdict map, plus the keys nobody actually answered.

    task-32280 fix round (R23). A round the user never answered -- Stop
    mid-card, or a revoked round -- fails CLOSED: every undecided key
    defaults to ``"deny"`` and the runtime must keep seeing exactly that,
    so the tool does not run. But an unanswered card is not a refusal, and
    ``request_mcp_approvals`` already writes the honest
    ``denied-unresolved`` audit row for it. The review hooks, seeing only
    ``"deny"``, then recorded a SECOND row that Audit renders as "Denied by
    you" -- a decision nobody made.

    This is a plain ``dict`` (every consumer keeps treating it as the
    verdict map it always was) carrying one extra attribute so the two
    hooks can tell the two cases apart at the one place it matters:
    ``record_user_denial``. Deliberately NOT a distinct verdict string --
    that would have to be taught to `_apply_verdict`, `apply_batch_
    decisions`, `apply_promotion_decisions`, `builtin_gate.stamp` and the
    refusal loop, and any one of them missing it would let a denied tool
    run.

    Attributes:
        unresolved_keys: The verdict keys (``call_id`` where the runtime
            can address the call, else ``llm_name`` -- the same keying
            ``request_mcp_approvals`` uses) that were defaulted to deny by
            cancellation or revocation rather than chosen by the user.
    """

    unresolved_keys: frozenset[str] = frozenset()

    def __init__(
        self,
        decisions: Mapping[str, str] | None = None,
        *,
        denial_reasons: Mapping[str, object] | None = None,
    ) -> None:
        """Keep optional denial text outside the unchanged decision strings.

        Args:
            decisions: Existing per-call or name-keyed answer map.
            denial_reasons: Transient user text for explicitly denied rows only.
        """
        from tldw_chatbook.Utils.input_validation import (
            normalize_approval_denial_reason,
        )

        super().__init__(decisions or {})
        self.denial_reasons: dict[str, str] = {}
        if isinstance(denial_reasons, Mapping):
            for key, value in denial_reasons.items():
                if self.get(key) == "deny":
                    reason = normalize_approval_denial_reason(value)
                    if reason:
                        self.denial_reasons[key] = reason


@dataclass(frozen=True, slots=True)
class ApprovalStamp:
    """A raw owner decision and whether somebody actually answered it."""

    decision: object
    approval_decision: ApprovalDecision | None = None
    denial_refusal: str = ""


def selected_approval_key(decisions: Mapping, key: str, fallback: str) -> str:
    """Select an existing exact key before the owner's name fallback."""
    return key if key and key in decisions else fallback


def approval_key_unanswered(decisions: Mapping, key: str) -> bool:
    """Read unresolved provenance for the key actually selected by an owner."""
    return key in getattr(decisions, "unresolved_keys", ())


def approval_stamp(
    decision: object,
    *,
    unanswered: bool = False,
    allowing: tuple[str, ...] = ("approve_once", "approve_session", "always_allow"),
) -> ApprovalStamp:
    """Capture a fact without interpreting refusal copy or granting permission."""
    fact = None
    if not unanswered and isinstance(decision, str):
        if decision == "deny":
            fact = "denied"
        elif decision in allowing:
            fact = "approved"
    return ApprovalStamp(decision, fact)


def append_denial_reason(refusal: str, decisions: Mapping[str, str], key: str) -> str:
    """Append quoted user text only to the selected, explicitly denied row.

    Args:
        refusal: The owner's existing refusal, preserved when no reason applies.
        decisions: This round's answers with optional transient denial reasons.
        key: The exact answer key selected by the consumer.

    Returns:
        The refusal with bounded, labeled text, or the original refusal unchanged.
    """
    if (
        refusal == "proceed"
        or decisions.get(key) != "deny"
        or approval_key_unanswered(decisions, key)
    ):
        return refusal
    from tldw_chatbook.Utils.input_validation import normalize_approval_denial_reason

    reasons = getattr(decisions, "denial_reasons", {})
    reason = (
        normalize_approval_denial_reason(reasons.get(key))
        if isinstance(reasons, Mapping)
        else ""
    )
    if not reason:
        return refusal
    return f"{refusal}\nDenial reason (from user, untrusted text): {json.dumps(reason, ensure_ascii=False)}"
