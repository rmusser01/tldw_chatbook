"""Content-free original project-decision presentation, never folder authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .console_display_state import ConsoleProjectInstructionSourceRow, ConsoleProjectInstructionState
from .console_execution_projection_values import (
    _boolean, _count, _enum, _optional, _record, _text, _tuple,
)

if TYPE_CHECKING:
    from .console_chat_controller import ProjectInstructionDispatchNotice
    from .console_runtime import ConsoleProjectBindingChoice


@dataclass(frozen=True, slots=True)
class ConsoleProjectDecisionHead:
    """One original ordered head for the published session incarnation."""

    session_id: str
    incarnation_id: str
    decision_id: str
    order: int
    kind: str
    notice: ProjectInstructionDispatchNotice | None
    choices: tuple[ConsoleProjectBindingChoice, ...]


_NOTICE = {
    "session_id": _text, "destination_label": _text,
    "relative_source": _optional(_text), "scope": _text, "byte_count": _count,
    "outcomes": _tuple(_text), "warning_codes": _tuple(_text),
}
_CHOICE = {
    "binding_id": _text, "label": _text, "eligible": _boolean, "recovery": _text,
}


def _notice(value, decode):
    from .console_chat_controller import ProjectInstructionDispatchNotice

    return _record(value, ProjectInstructionDispatchNotice, _NOTICE, decode)


def _choice(value, decode):
    from .console_runtime import ConsoleProjectBindingChoice

    return _record(value, ConsoleProjectBindingChoice, _CHOICE, decode)


def _source_row(value, decode):
    return _record(value, ConsoleProjectInstructionSourceRow, {
        "relative_source": _text, "scope": _text, "byte_count": _optional(_count),
        "outcome": _text, "warning_code": _text,
    }, decode)


def project_instruction_status(value, decode):
    """Transfer only the content-free fields shown by the Inspector."""
    return _record(value, ConsoleProjectInstructionState, {
        "status": _text, "enabled": _boolean, "binding_label": _text,
        "locator_match": _enum("match", "mismatch", "not checked"),
        "sources": _tuple(_source_row), "warning_codes": _tuple(_text),
        "recovery_actions": _tuple(_enum("enable", "choose", "disable")),
    }, decode)


_HEAD = {
    "session_id": _text, "incarnation_id": _text, "decision_id": _text,
    "order": _count, "kind": _enum("binding", "dispatch"),
    "notice": _optional(_notice), "choices": _tuple(_choice),
}


def _head(value, decode):
    if value is None:
        return None
    result = _record(value, ConsoleProjectDecisionHead, _HEAD, decode)
    head = result if decode else value
    if (
        not head.session_id or not head.incarnation_id or not head.decision_id
        or head.order < 1
        or (head.kind == "dispatch" and (
            head.notice is None or head.notice.session_id != head.session_id or head.choices
        ))
        or (head.kind == "binding" and (head.notice is not None or not head.choices))
    ):
        raise ValueError("Console project decision identity changed")
    return result


def encode_project_head(value: ConsoleProjectDecisionHead | None) -> dict | None:
    """Detach the original notice or choices without repository bodies."""
    return _head(value, False)


def decode_project_head(value: dict | None) -> ConsoleProjectDecisionHead | None:
    """Restore only the closed original modal input types."""
    return _head(value, True)
