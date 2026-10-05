"""Refuse an awaited surface swap that would wait on its own caller forever.

TASK-34000.4 (2026-10-02 Library review, finding L-01). Pressing "Export…" in
the Library's Media list froze the whole app: the press was dispatched on the
Media canvas's message pump, and the handler awaited a screen recompose that
removes that canvas. Nothing repainted and no key was read, Ctrl+Q included,
and the process sat at 0% CPU with no traceback anywhere.

Why it never ends (Textual 8.2.8). ``Widget.remove_children`` /
``Screen.recompose`` return an ``AwaitRemove`` over the pump tasks of the
removed ROOTS. Each removed widget, as its own loop exits, posts ``Prune`` to
its children and then gathers THEIR pump tasks (``Widget._message_loop_exit``)
-- with no timeout. So a coroutine running on the pump task of a widget
*below* a removed root waits for the root, the root waits for its children,
and that chain ends at the pump that is doing the waiting. ``Screen.recompose``
also holds ``App.batch_update`` across the removal, so the screen never paints
again.

The predicate is exact, and deliberately narrow:

* It compares TASKS, not ``textual._context.active_message_pump``. The
  context variable names the pump that *scheduled* the code, not the one
  running it: a ``call_after_refresh`` callback runs on the screen's task with
  the scheduling widget still in the variable (measured: that shape completes,
  and a context-variable check would have refused it).
* Only a STRICT descendant of a removed root counts. ``AwaitRemove`` leaves
  the current task out of what it waits on, so a widget awaiting its own
  removal as a root completes (measured), and refusing it would break code
  that works.

So this refuses only an await that could never have returned. A widget that
needs the surface swapped hands the swap to the screen -- ``screen.call_next``
/ ``screen.call_after_refresh``, or a screen-owned worker -- and returns.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from textual.widget import Widget


class SurfaceSwapSelfAwaitError(RuntimeError):
    """An awaited DOM removal was started on the pump of a widget it removes."""


def pump_awaiting_its_own_removal(removed_roots: Iterable[Widget]) -> Widget | None:
    """Find the widget whose pump is running this code and would be removed.

    Args:
        removed_roots: The widgets an awaited removal is about to remove.

    Returns:
        The strict descendant of a removed root whose message-pump task is the
        current asyncio task, or ``None`` when the removal can be awaited here.
    """
    try:
        current = asyncio.current_task()
    except RuntimeError:
        return None
    if current is None:
        return None
    for root in removed_roots:
        for node in root.walk_children(with_self=False):
            if node._task is current:
                return node
    return None


def ensure_surface_swap_not_self_awaited(
    removed_roots: Iterable[Widget], *, seam: str
) -> None:
    """Raise instead of starting a removal that would never be awaited out of.

    Called before any teardown, so a refusal leaves the DOM exactly as it was.

    Args:
        removed_roots: The widgets the awaited removal is about to remove.
        seam: The swap seam being guarded, for the error message.

    Raises:
        SurfaceSwapSelfAwaitError: If this code is running on the message pump
            of a widget below one of ``removed_roots``.
    """
    parked = pump_awaiting_its_own_removal(removed_roots)
    if parked is None:
        return
    raise SurfaceSwapSelfAwaitError(
        f"{seam}: refusing an awaited surface swap that could never finish. "
        f"It is running on the message pump of {parked!r}, which the swap "
        "removes -- Textual's teardown waits for that pump, and that pump is "
        "waiting for the teardown. Hand the swap to the screen "
        "(screen.call_next / screen.call_after_refresh, or a screen-owned "
        "worker) instead of awaiting it from a handler or a call_later "
        "callback on a widget inside the surface."
    )


def log_refused_surface_swap(
    error: SurfaceSwapSelfAwaitError, *, fallback: str
) -> None:
    """Record, at ERROR, a refusal that a seam turned into its own fallback.

    A seam that already has a safe fallback for a failed projection (the
    Library's open-surface seam schedules a whole-screen recompose on the
    screen's pump) may take it for a refusal too, so the user still gets the
    surface. The refusal is still a bug in the caller -- the await would have
    frozen the app -- so it is logged loudly, by name, never at debug.

    Args:
        error: The refusal.
        fallback: What the seam did instead, for the log line.
    """
    logger.error(
        "Surface swap refused (a handler awaited its own removal); {fallback} "
        "instead. {error}",
        fallback=fallback,
        error=error,
    )
