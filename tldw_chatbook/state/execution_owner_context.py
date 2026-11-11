"""Execution owner bound to the current runtime scope."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner


_current_owner: ContextVar[ExecutionOwner | None] = ContextVar(
    "agent_execution_owner", default=None
)


def current_execution_owner() -> ExecutionOwner | None:
    """Return this run thread's owner (explicitly bound again in children)."""
    return _current_owner.get()
