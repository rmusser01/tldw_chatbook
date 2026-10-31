"""A Console send's start: from its acceptance to its own provider request.

TASK-33620.15.4 (owner-approved exception, 2026-10-24) keeps three Console
display reads on the UI loop during a send's start, from its acceptance to
its provider request and 3 s at most: the Library activity rows and the
Inspector's registry and appearance reads. Off the loop each opened its own
SQLite connection, whose CPU delayed first sends.

TASK-36610: the start belongs to its send. The first version kept one
acceptance time per screen and ended it at the process's latest provider
request, so another session's request ended this send's start early, and
sends the composer did not make (a queued prompt's dispatch, an agent wake,
a goal iteration, an agent's chat start) never opened one. Now each accepted
send's work carries its own ``SendStart`` in a context variable: tasks and
``asyncio.to_thread`` workers created inside copy it, so the provider request
that work makes (``console_send_diagnostics.record_send_stage``) ends that
send's start and no other. The Console asks whether any start on its store
is still open (``send_start_open``); the 3 s limit stays the caller's.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import Coroutine, Iterator
from contextvars import ContextVar
from typing import Any


class SendStart:
    """One accepted send's start; it ends once, at its provider request or end."""

    __slots__ = ("accepted_at", "ended")

    def __init__(self) -> None:
        self.accepted_at = time.monotonic()
        self.ended = False

    def end(self, *_args: object) -> None:
        """End this start (idempotent; also a task done callback)."""
        self.ended = True


_BOUND: ContextVar[SendStart | None] = ContextVar("console_send_start", default=None)

#: The starts opened on each store (the Console's whole session set), weakly
#: keyed so a disposed store takes its starts with it. Touched on the UI loop
#: only; a provider request on a worker thread only flips ``ended``.
_OPEN: weakref.WeakKeyDictionary[Any, list[SendStart]] = weakref.WeakKeyDictionary()


@contextlib.contextmanager
def _bound_start(owner: Any) -> Iterator[SendStart]:
    """Open a start on ``owner``; work created inside carries it.

    A failure inside ends the start. An owner that cannot be weakly keyed (a
    test double) still gets a start, which no Console can see.
    """
    start = SendStart()
    with contextlib.suppress(TypeError):
        starts = _OPEN.setdefault(owner, [])
        starts[:] = [item for item in starts if not item.ended]
        starts.append(start)
    token = _BOUND.set(start)
    try:
        yield start
    except BaseException:
        start.end()
        raise
    finally:
        _BOUND.reset(token)


def start_custody_task(owner: Any, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Create an accepted turn's custody task, carrying a start opened now.

    The start ends at the provider request the task's work makes, or when
    the task ends without one.

    Args:
        owner: The Console store the turn was accepted on.
        coroutine: The turn's custody coroutine.

    Returns:
        The lazy task (``asyncio.Task`` directly, before any configurable
        task factory could step it).
    """
    with _bound_start(owner) as start:
        task = asyncio.Task(coroutine, loop=asyncio.get_running_loop())
    task.add_done_callback(start.end)
    return task


@contextlib.contextmanager
def submission_send_start(owner: Any) -> Iterator[SendStart]:
    """A start for a submission made right here, which ends with it.

    For sends that reach the controller without runtime custody (an agent's
    chat start).

    Args:
        owner: The Console store the submission is made on.

    Yields:
        The start, open until the submission's provider request or its end.
    """
    with _bound_start(owner) as start:
        try:
            yield start
        finally:
            start.end()


def note_provider_request() -> None:
    """End the start of the send whose work makes this provider request."""
    start = _BOUND.get()
    if start is not None:
        start.end()


def send_start_open(owner: Any, limit: float) -> bool:
    """Whether a send on ``owner`` is accepted and has not reached its provider.

    Args:
        owner: The Console store.
        limit: Seconds after acceptance at which a start counts as ended.

    Returns:
        True while any start on ``owner`` is open and younger than ``limit``.
    """
    try:
        starts = _OPEN.get(owner)
    except TypeError:
        return False
    if not starts:
        return False
    now = time.monotonic()
    starts[:] = [
        item for item in starts if not item.ended and now - item.accepted_at < limit
    ]
    return bool(starts)
