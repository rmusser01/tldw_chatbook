"""One detached read of the original project and mixed decision heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .console_execution_decision_values import decode_pending_decision, encode_pending_decision
from .console_execution_project_values import decode_project_head, encode_project_head
from .console_execution_projection_values import _count, _nonempty_text, _record


@dataclass(frozen=True, slots=True)
class ConsoleDecisionHead:
    """Presentation values for one authenticated visit; never a decision grant."""

    source_revision: int
    session_id: str
    incarnation_id: str
    attachment_id: str
    visit: int
    decision: Any
    project: Any


def _mixed(value, decode):
    return decode_pending_decision(value) if decode else encode_pending_decision(value)


def _project(value, decode):
    return decode_project_head(value) if decode else encode_project_head(value)


def _head(value, decode):
    result = _record(value, ConsoleDecisionHead, {
        "source_revision": _count, "session_id": _nonempty_text,
        "incarnation_id": _nonempty_text, "attachment_id": _nonempty_text,
        "visit": _count, "decision": _mixed, "project": _project,
    }, decode)
    head = result if decode else value
    if (
        head.visit < 1
        or head.decision is not None and head.decision.session_id != head.session_id
        or head.project is not None and (
            head.project.session_id != head.session_id
            or head.project.incarnation_id != head.incarnation_id
        )
    ):
        raise ValueError("Console decision head scope changed")
    return result


def encode_decision_head(value: ConsoleDecisionHead) -> dict:
    """Use the existing closed decision codecs without exporting their owners."""
    return _head(value, False)


def decode_decision_head(value: dict) -> ConsoleDecisionHead:
    """Restore the original widget inputs for this exact read scope."""
    return _head(value, True)
