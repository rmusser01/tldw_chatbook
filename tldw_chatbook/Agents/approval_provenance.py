"""Ephemeral answer provenance stored with existing approval stamps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tldw_chatbook.Agents.agent_models import ApprovalDecision


@dataclass(frozen=True, slots=True)
class ApprovalStamp:
    """A raw owner decision and whether somebody actually answered it."""

    decision: object
    approval_decision: ApprovalDecision | None = None


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
