"""TASK-33620.1: the Console's one cached view of the global emergency stop.

``tldw_chatbook/emergency_stop.py`` owns the durable switch. Before this
module the Console read it only inside a send's refusal, so once
``/emergency-stop``'s toast faded the header said "Ready", the status strip
was unchanged and Send looked live. Worse, the received-intent fast path
(PR #3023) skipped even that read: live on origin/dev 38c9d2edcf a send typed
during an active stop reached the provider.

The screen keeps one copy, ``ChatScreen._console_emergency_stop_copy``: the
stop's visible copy while it holds new work, ``""`` otherwise. It is written
here, and only here:

* on Console arrival, from a read made off the UI loop after first paint
  (``refresh``). The scheduler's per-tick read moved off the loop in
  TASK-33620.15.24; the Console adds no per-tick read back;
* by ``/emergency-stop`` (``run_command``);
* by a send about to take the received-intent fast path
  (``received_send_held``), which reads the file itself, so a stop that
  another writer set still holds it.

The header, the status strip, Send's reason and the empty transcript read
that copy; none of them touches the file. The copy is fixed text (the
reason the user typed is left out): it lands in a markup-parsed Static and
toasts, and a free-text reason must not be parsed as markup there.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.emergency_stop import (
    EmergencyStopState,
    clear_emergency_stop,
    default_emergency_stop_path,
    read_emergency_stop,
    set_emergency_stop,
)

#: ``/emergency-stop <word>`` arguments that lift the stop (TASK-26004).
CLEAR_ARGUMENTS = frozenset({"clear", "off", "resume"})

#: What every Console surface says while the stop holds new work.
HELD_COPY = EmergencyStopState(active=True).visible_copy()

ACTIVATED_NOTICE = (
    "Emergency stop ACTIVE — new runs and scheduled dispatches are held. "
    "Run /emergency-stop clear to resume."
)
CLEARED_NOTICE = "Emergency stop cleared — new work may start."


def held(screen: Any) -> bool:
    """Whether the screen's cached view says the stop holds new work."""
    return bool(screen._console_emergency_stop_copy)


def note_state(
    screen: Any, path: Path, state: EmergencyStopState, *, announce: bool = False
) -> None:
    """Record ``state`` as the screen's view of the stop; repaint what moved.

    Args:
        screen: The Console screen.
        path: The stop file the state was read from or written to.
        state: The stop's state.
        announce: Console arrival: when this is the screen's first view of
            the stop and it is active, say so once (AC#5, a stop persisted
            from an earlier session holds scheduled dispatches too).
    """
    first_view = getattr(screen, "_console_emergency_stop_path", None) is None
    screen._console_emergency_stop_path = Path(path)
    copy = HELD_COPY if state.active else ""
    changed = copy != screen._console_emergency_stop_copy
    screen._console_emergency_stop_copy = copy
    if announce and first_view and copy:
        screen.app_instance.notify(copy, severity="warning", markup=False)
    if changed and screen.is_mounted:
        # Header, strip, Send and the empty transcript all re-read the copy.
        screen._sync_console_control_bar()


def _read_default_stop() -> tuple[Path, EmergencyStopState]:
    path = default_emergency_stop_path()
    return path, read_emergency_stop(path)


async def refresh(screen: Any) -> None:
    """Read the stop off the UI loop and record it (Console arrival).

    Resolving the default path is a guarded config read and the read is file
    I/O, so both run on a worker thread (the TASK-33620.15.24 rule).
    """
    try:
        path, state = await asyncio.to_thread(_read_default_stop)
    except Exception:  # noqa: BLE001 -- an unknown state leaves the view unknown
        logger.opt(exception=True).debug("emergency-stop arrival read failed")
        return
    if screen.is_mounted:
        note_state(screen, path, state, announce=True)


def received_send_held(screen: Any) -> bool:
    """Whether a send must leave the received-intent fast path for the stop.

    AC#7: the fast path skips ``send_refusal_copy``, so it asks here. It
    reads the file (one small read per send, as every send did before
    #3023) rather than trusting the cached view, so a stop set by another
    writer still holds the send. ``True`` sends the draft down the dispatch
    path, whose own gates refuse it with the copy.
    """
    if held(screen):
        return True
    path = getattr(screen, "_console_emergency_stop_path", None)
    try:
        if path is None:  # The arrival read has not landed yet.
            path = default_emergency_stop_path()
        state = read_emergency_stop(path)
    except Exception:  # noqa: BLE001 -- doubt holds: the dispatch path decides
        return True
    if state.active:
        note_state(screen, path, state)
    return state.active


async def run_command(screen: Any, parse: Any) -> bool:
    """``/emergency-stop [clear|<reason>]``: the global stop (TASK-26004).

    Holds all NEW agent runs and scheduled dispatches; in-flight work is
    untouched, and the state survives a restart. ``clear``/``off``/``resume``
    lifts it. The screen's view changes with the file, so the header, strip
    and Send show the hold until it is cleared (TASK-33620.1). The path
    lookup and the fsynced write run off the UI loop.
    """
    arg = (parse.args or "").strip()
    if arg.lower() in CLEAR_ARGUMENTS:
        await _write_off_loop(screen, EmergencyStopState(active=False))
        screen._clear_console_composer_draft()
        screen.app_instance.notify(CLEARED_NOTICE)
        return True
    await _write_off_loop(screen, EmergencyStopState(active=True, reason=arg))
    screen._clear_console_composer_draft()
    screen.app_instance.notify(ACTIVATED_NOTICE, severity="warning")
    return True


async def _write_off_loop(screen: Any, state: EmergencyStopState) -> None:
    """Write ``state`` to the stop file on a worker thread, then record it."""
    known = getattr(screen, "_console_emergency_stop_path", None)

    def write() -> Path:
        path = known if known is not None else default_emergency_stop_path()
        if state.active:
            set_emergency_stop(path, reason=state.reason)
        else:
            clear_emergency_stop(path)
        return path

    note_state(screen, await asyncio.to_thread(write), state)
