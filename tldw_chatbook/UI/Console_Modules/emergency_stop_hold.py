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
* by ``/emergency-stop`` and by the status strip's Clear (``run_command``,
  ``clear_from_strip``);
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
from contextvars import copy_context
from pathlib import Path
from threading import Event
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
    _note_newer_view(screen)
    screen._console_emergency_stop_path = Path(path)
    copy = HELD_COPY if state.active else ""
    changed = copy != screen._console_emergency_stop_copy
    screen._console_emergency_stop_copy = copy
    if announce and first_view and copy:
        screen.app_instance.notify(copy, severity="warning", markup=False)
    if changed and screen.is_mounted:
        # Header, strip, Send and the empty transcript all re-read the copy.
        screen._sync_console_control_bar()


def _note_newer_view(screen: Any) -> None:
    """Count a change to the view, so an arrival read older than it is dropped."""
    screen._console_emergency_stop_serial = _view_serial(screen) + 1


def _view_serial(screen: Any) -> int:
    return getattr(screen, "_console_emergency_stop_serial", 0)


def _read_default_stop() -> tuple[Path, EmergencyStopState]:
    path = default_emergency_stop_path()
    return path, read_emergency_stop(path)


async def refresh(screen: Any) -> None:
    """Read the stop off the UI loop and record it (Console arrival).

    Resolving the default path is a guarded config read and the read is file
    I/O, so both run on a worker thread (the TASK-33620.15.24 rule). A write
    or a send's own read recorded while this read was in flight is newer than
    it, so this result is then dropped rather than restoring a stale view.
    """
    serial = _view_serial(screen)
    try:
        path, state = await asyncio.to_thread(_read_default_stop)
    except Exception as exc:  # noqa: BLE001 -- an unknown state leaves the view unknown
        # The type only: a path-resolution error's message names the data dir.
        logger.debug(
            "emergency-stop arrival read failed (error_type={})", type(exc).__name__
        )
        return
    if screen.is_mounted and _view_serial(screen) == serial:
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
    if path is None:
        # The arrival read has not landed (or failed). Resolving the path here
        # would be a guarded config read on the UI loop; the dispatch path
        # reads the stop itself.
        return True
    try:
        state = read_emergency_stop(path)
    except Exception:  # noqa: BLE001 -- doubt holds: the dispatch path decides
        return True
    if state.active:
        note_state(screen, path, state)
    return state.active


def still_held(screen: Any) -> bool:
    """Whether a refusal from the cached hold is still true.

    The view is refreshed on arrival and by this Console's own writes, so a
    stop that another app instance cleared on the same profile would keep
    Send refused until the next arrival. A send the cached view would refuse
    reads the file first (one small read, only while the view says held); a
    cleared file lifts the hold everywhere and lets the send through.
    """
    if not held(screen):
        return False
    path = getattr(screen, "_console_emergency_stop_path", None)
    if path is None:
        return True
    state = read_emergency_stop(path)  # Never raises: an unreadable file holds.
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
    from .command_draft import CAPTURED_COMMAND_DRAFT, take_command_draft

    composer = screen._console_composer_or_none()
    captured = CAPTURED_COMMAND_DRAFT.get()
    if captured is None and composer is not None:
        captured = composer.capture_draft_for_send()
    arg = (parse.args or "").strip()
    active = arg.lower() not in CLEAR_ARGUMENTS
    state = EmergencyStopState(active=active, reason=arg if active else "")
    if await _write_off_loop(screen, state):
        if captured is not None and composer is screen._console_composer_or_none():
            token = CAPTURED_COMMAND_DRAFT.set(captured)
            try:
                take_command_draft(composer, screen._clear_console_composer_draft)
            finally:
                CAPTURED_COMMAND_DRAFT.reset(token)
        if active:
            screen.app_instance.notify(ACTIVATED_NOTICE, severity="warning")
        else:
            screen.app_instance.notify(CLEARED_NOTICE)
    return True


async def clear_from_strip(screen: Any) -> None:
    """The status strip's Clear: lift the stop (AC#2)."""
    if await _write_off_loop(screen, EmergencyStopState(active=False)):
        screen.app_instance.notify(CLEARED_NOTICE)


async def _write_off_loop(screen: Any, state: EmergencyStopState) -> bool:
    """Write ``state`` to the stop file on a worker thread, then record it.

    Returns whether the write landed. A refused write (a read-only data
    directory, a full disk) is reported, never raised: raised, it escaped the
    command's worker or the strip's message handler and closed the app. The
    view then follows what the file actually says. One app admits one write
    at a time, across disposable screens and command/chip entry points. Its
    native Future remains owned until physical retirement, even if the caller
    is cancelled; a later instruction must retry after that write finishes.
    """
    app = screen.app_instance
    if app.console_emergency_stop_write is not None:
        app.notify(
            "Emergency stop is already being changed; wait for it to finish, then retry.",
            severity="warning",
        )
        return False
    known = getattr(screen, "_console_emergency_stop_path", None)
    _note_newer_view(screen)  # An arrival read already in flight is older.

    def write() -> tuple[Path, EmergencyStopState, Exception | None]:
        path = known if known is not None else default_emergency_stop_path()
        try:
            if state.active:
                set_emergency_stop(path, reason=state.reason)
            else:
                clear_emergency_stop(path)
        except Exception as exc:  # noqa: BLE001 -- reported below, never raised
            return path, read_emergency_stop(path), exc
        return path, state, None

    verb = "set" if state.active else "clear"
    context = copy_context()
    submission = Event()
    future = None

    def invoke():
        # Executor submission may enqueue before raising while starting a
        # thread. Only a successfully returned Future permits the file effect.
        submission.wait()
        if future is not None:
            return context.run(write)
        return None

    try:
        future = asyncio.get_running_loop().run_in_executor(None, invoke)
    except Exception as exc:  # noqa: BLE001 -- no write was admitted
        _report_write_failure(screen, verb, exc, None)
        return False
    finally:
        submission.set()
    app.console_emergency_stop_write = future
    cancelled = False
    success = False
    try:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:  # noqa: BLE001 -- consume the exact outcome below
                break
        try:
            path, recorded, error = future.result()
        except Exception as exc:  # noqa: BLE001 -- the path itself did not resolve
            _report_write_failure(screen, verb, exc, None)
        else:
            note_state(screen, path, recorded)
            success = error is None
            if error is not None:
                _report_write_failure(screen, verb, error, recorded)
    finally:
        if app.console_emergency_stop_write is future:
            app.console_emergency_stop_write = None
    if cancelled:
        raise asyncio.CancelledError
    return success


def _report_write_failure(
    screen: Any, verb: str, exc: Exception, state: EmergencyStopState | None
) -> None:
    """Say the stop was not changed, and what it is now (when that is known)."""
    # The type only: an OSError's message names the data directory.
    error_type = type(exc).__name__
    logger.warning("emergency-stop {} failed (error_type={})", verb, error_type)
    now = "" if state is None else f" It is {'on' if state.active else 'off'}."
    if state is not None and state.active and verb == "clear":
        now = " It is still on."
    screen.app_instance.notify(
        f"Couldn't {verb} the emergency stop ({error_type}).{now}",
        severity="error",
        markup=False,
    )
