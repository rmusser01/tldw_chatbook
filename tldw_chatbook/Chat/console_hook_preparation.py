"""Hook preparation compatibility names and exact-task session attribution."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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
    "ConsoleHookPreparationRead",
    "drain_hook_preparation_reads",
    "observe_hook_preparation_reads",
    "hook_preparation_reads_for",
    "hook_preparation_source_for",
    "run_hook_preparation_read",
    "bind_hook_preparation_session",
    "hook_preparation_session_for",
)


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
