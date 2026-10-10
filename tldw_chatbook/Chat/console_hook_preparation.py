"""Hook preparation compatibility names and exact-task session attribution."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .console_preparation_reads import (
    ConsolePreparationRead as ConsoleHookPreparationRead,
    _current_task,
    drain_preparation_reads as drain_hook_preparation_reads,
    observe_preparation_reads as observe_hook_preparation_reads,
    preparation_reads_for as hook_preparation_reads_for,
    preparation_source_for as hook_preparation_source_for,
    run_preparation_read as run_hook_preparation_read,
)

__all__ = (
    "ConsoleHookAttemptRead",
    "ConsoleHookPreparationRead",
    "drain_hook_preparation_reads",
    "observe_hook_preparation_reads",
    "hook_preparation_reads_for",
    "hook_preparation_source_for",
    "run_hook_preparation_read",
    "bind_hook_preparation_session",
    "hook_preparation_session_for",
)


@dataclass(frozen=True, slots=True, eq=False)
class ConsoleHookAttemptRead:
    """One Send attempt's single pre-commit hook authority read (ADR-225 #3).

    The attempt's first full consent read -- received-intent preparation, or
    the controller's first submission admission on entry points without one
    -- produces this. The same attempt then passes it explicitly, as an
    argument, to its later pre-commit preparation consumers (submission
    admission, v2 hook preparation, legacy UserPromptSubmit target selection)
    so they do not repeat the full read. It is never held by an owner, keyed
    by time or handed to another attempt or session.

    It grants nothing: each consumer accepts it only for its own session and
    owner and only while ``HookPermissions.attempt_read_current`` holds,
    otherwise it performs its existing fresh read. Even then it answers only
    "nothing to do" -- admission that it does not refuse, no UserPromptSubmit
    hook to select, no v2 hook to prepare; a consumer with a hook to select or
    prepare reads fresh. The final pre-dispatch admission is always a fresh
    read, and every hook launch keeps its own fresh ``launch_guard``.

    Attributes:
        session_id: The Console session the attempt belongs to.
        authority: The owner's ``HookAuthorityRead`` for that attempt.
    """

    session_id: str
    authority: Any = field(repr=False)

    def authority_for(self, owner: object, session_id: str | None) -> Any:
        """Return the read for this exact session and owner, else ``None``.

        Args:
            owner: The consent owner the consumer would otherwise read.
            session_id: The consumer's session.

        Returns:
            The ``HookAuthorityRead`` when both match; the owner still decides
            whether it stands (``attempt_read_current``).
        """
        if (
            session_id is None
            or session_id != self.session_id
            or getattr(self.authority, "owner", None) is not owner
        ):
            return None
        return self.authority


@dataclass(frozen=True, slots=True)
class _HookSessionBinding:
    creator: object
    session_id: str | None
    task: asyncio.Task[Any]


_session_binding: ContextVar[_HookSessionBinding | None] = ContextVar(
    "console_hook_preparation_session", default=None
)


@contextmanager
def bind_hook_preparation_session(
    creator: object, session_id: str | None
) -> Iterator[None]:
    """Attribute a known zero-argument hook call without changing its contract."""
    task = _current_task()
    if task is None:
        raise RuntimeError("Hook preparation attribution requires its owning task.")
    token = _session_binding.set(_HookSessionBinding(creator, session_id, task))
    try:
        yield
    finally:
        _session_binding.reset(token)


def hook_preparation_session_for(creator: object) -> str | None:
    """Read exact Task attribution; copied child/thread contexts cannot borrow it."""
    binding = _session_binding.get()
    if binding is None or binding.task is not _current_task():
        return None
    if binding.creator is not creator:
        raise RuntimeError("Hook preparation owner changed.")
    return binding.session_id
