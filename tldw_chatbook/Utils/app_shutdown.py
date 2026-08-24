"""Bounded, graceful process termination for the TUI (task-19561).

Why this module exists
----------------------
Before task-19561 the console-script entry point answered ``SIGTERM`` with
``os._exit(0)`` straight from the signal handler. ``os._exit`` is not an
exit, it is an amputation: no ``finally`` blocks, no ``atexit`` callbacks, no
Textual ``on_unmount``, no database connections closed. A worker sitting
inside ``with db.transaction() as conn:`` never reaches its rollback, and the
row it had already flipped to ``running`` stays ``running`` for the life of
the install. That mattered rather more than a corner case would: Textual
takes the terminal into raw mode, which swallows ``SIGINT``, so ``SIGTERM``
is how this application is actually terminated.

Meanwhile three separate blocks tried to stop stray threads from delaying
process exit by assigning ``thread.daemon = True`` to threads that were
already running. CPython forbids that outright -- ``RuntimeError: cannot set
daemon status of active thread`` -- so all three were inert, and one of them
logged an ERROR for every thread it failed on, every single exit.

The shape that replaces both
----------------------------
1. **Ask, don't amputate.** The signal handler hands the running Textual app
   an ordinary ``App.exit()`` through ``loop.call_soon_threadsafe``. Every
   existing cleanup path then runs exactly as it does for a keyboard quit.
2. **Bound the ask.** Arming the exit watchdog starts a *daemon* thread --
   daemon-at-construction, the only moment CPython allows it -- holding a
   deadline. If the process has not finished by then, the watchdog reports
   which threads are still alive and *then* hard-exits. The hard exit is the
   escape hatch after the graceful path has had its bounded chance, never
   the first action. The deadline does not distinguish a wedged process from
   a healthy one that is simply slow, and it cannot: an uninterruptible
   thread worker looks identical either way. See
   ``DEFAULT_SHUTDOWN_GRACE_SECONDS`` for what that costs and why the
   default is sized the way it is.
3. **Escalate on repeat.** A second ``SIGTERM``/``SIGINT`` is an explicit
   operator "I meant it" and exits immediately.

Why ``signal.signal`` rather than ``loop.add_signal_handler``
------------------------------------------------------------
``loop.add_signal_handler`` is the tidier asyncio primitive, but it is owned
by the loop: ``remove_signal_handler`` restores ``SIG_DFL``, and the loop
does not exist during the several seconds of import-and-config work before
``App.run()`` nor during interpreter teardown afterwards. Those windows are
exactly when an unhandled ``SIGTERM`` would kill a half-written config or a
half-closed database. One process-level handler that is installed once and
never removed covers the whole process lifetime; it hands work to the loop
when there is a loop, and unwinds the main thread with ``SystemExit`` when
there is not.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time
from typing import Any

from loguru import logger

__all__ = [
    "DEFAULT_SHUTDOWN_GRACE_SECONDS",
    "arm_exit_watchdog",
    "claim_process_exit",
    "disarm_exit_watchdog",
    "graceful_shutdown_grace_seconds",
    "install_termination_handlers",
    "owns_process_exit",
    "register_running_app",
    "termination_requested",
    "unregister_running_app",
]

#: How long the whole teardown -- Textual unmount, ``asyncio.run`` cleanup and
#: interpreter finalization -- may take before the watchdog stops waiting.
#:
#: Be clear-eyed about what this is NOT: it is not only a stuck-process bound.
#: ``run_worker(..., thread=True)`` runs on the loop's default executor and
#: cannot be interrupted, so teardown genuinely waits for it -- and a
#: *healthy* worker that outlives this deadline is hard-exited exactly like a
#: wedged one, with whatever it was writing abandoned. There are ~180
#: ``thread=True`` sites in this codebase, including media ingest, notes and
#: character export, library export and RAG indexing, so "quit while a big
#: ingest is running" is the live case, not a corner.
#:
#: **Why 120 s and not something tighter.** Measured, clean quit (no signal),
#: one ordinary thread worker holding an open ``BEGIN IMMEDIATE``: at 20 s the
#: process died at 20.1 s with rc 1 and the transaction abandoned, where the
#: merge base had waited 28.8 s and committed. The cliff was exactly the grace
#: period. Nothing is bought by tightening it: the "interpreter exit is not
#: blocked for seconds" requirement is satisfied by the *quiet*-exit
#: measurement (0.60-0.70 s), which this number does not touch at all, because
#: a healthy exit never reaches the deadline. So a smaller value costs a
#: 30-second ingest its write and buys nothing. 120 s still bounds a wedged
#: process to two minutes, which is the thing the watchdog exists for.
#:
#: The asymmetry is what decides it: a slow quit is an annoyance, an abandoned
#: transaction is data loss, and the owner's standing ruling is durability over
#: quick. Deliberately NOT done: extending the deadline when a straggler is
#: reported -- that turns a bound into a suggestion, and it was declined under
#: the same ruling.
DEFAULT_SHUTDOWN_GRACE_SECONDS = 120.0

#: Clamp for the configured value. Below a second the watchdog would start
#: killing healthy shutdowns; above five minutes it stops being a bound.
_MIN_GRACE_SECONDS = 1.0
_MAX_GRACE_SECONDS = 300.0

#: 128 + signal number, the shell convention for "died on a signal".
_EXIT_CODE_SIGTERM = 143
_EXIT_CODE_SIGINT = 130
#: Used when the watchdog, not a signal, is what ends the process.
_EXIT_CODE_WATCHDOG = 1


def _hard_exit(code: int) -> None:  # pragma: no cover - replaced in tests
    """The one place this package is allowed to skip cleanup.

    Isolated behind a function so tests can observe the decision without the
    test process being the thing that dies.
    """
    os._exit(code)


class _ShutdownState:
    """Process-global termination bookkeeping, all mutations under one lock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._signal_count = 0
        self._handlers_installed = False
        self._owns_process_exit = False
        self._app: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watchdog: threading.Thread | None = None
        self._watchdog_deadline: float | None = None
        self._stand_down: threading.Event | None = None

    # -- signals ---------------------------------------------------------
    def record_signal(self) -> int:
        with self._lock:
            self._signal_count += 1
            return self._signal_count

    @property
    def signalled(self) -> bool:
        with self._lock:
            return self._signal_count > 0

    @property
    def handlers_installed(self) -> bool:
        """Whether a *successful* installation has already latched."""
        with self._lock:
            return self._handlers_installed

    def mark_handlers_installed(self) -> bool:
        """Latch installation as done; True the first time only.

        Called only after ``signal.signal`` has actually succeeded for every
        signal -- see ``install_termination_handlers``. Latching it earlier
        made a failed installation permanent.
        """
        with self._lock:
            if self._handlers_installed:
                return False
            self._handlers_installed = True
            return True

    # -- process ownership ------------------------------------------------
    def claim_process_exit(self) -> None:
        with self._lock:
            self._owns_process_exit = True

    @property
    def owns_process_exit(self) -> bool:
        with self._lock:
            return self._owns_process_exit

    # -- the running app -------------------------------------------------
    def set_app(self, app: Any, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._app = app
            self._loop = loop

    def clear_app(self, app: Any) -> None:
        with self._lock:
            if self._app is app:
                self._app = None
                self._loop = None

    def app_and_loop(self) -> tuple[Any, asyncio.AbstractEventLoop | None]:
        with self._lock:
            return self._app, self._loop

    # -- watchdog --------------------------------------------------------
    def arm(self, seconds: float, reason: str, exit_code: int) -> bool:
        """Start (or tighten) the watchdog. True iff a thread was started.

        Monotonic by design: a later, laxer deadline never relaxes a bound
        that is already running, so the tightest arm during one teardown is
        the one that holds.
        """
        now = time.monotonic()
        deadline = now + seconds
        with self._lock:
            # "A bound is already running" is the *deadline*, not the thread's
            # `is_alive()`. Independent review (task-19561) found the
            # `is_alive()` form breakable: it is False for a thread that has
            # been constructed and published but not yet started, and a second
            # arm can land in that window -- from another thread, or on this
            # one, because a signal handler re-enters at a bytecode boundary
            # and the lock is an RLock that re-entry sails straight through.
            # A laxer arm slipping through there silently relaxed a bound that
            # was already running (proven by widening the window). An expired
            # deadline is correctly *not* a live bound, so re-arming after one
            # elapsed still works.
            existing = self._watchdog_deadline
            if existing is not None and existing > now and deadline >= existing:
                return False
            # A watchdog being replaced by a tighter one must be told to stop.
            # Leaving it asleep on its old, longer deadline leaves a live
            # timer nothing can cancel any more -- which is how the test that
            # first exercised this path ended up with a thread scheduled to
            # `os._exit` the pytest process thirty seconds later.
            superseded = self._stand_down
            stand_down = threading.Event()
            self._stand_down = stand_down
            self._watchdog_deadline = deadline
            # `daemon=True` at CONSTRUCTION -- the only point CPython permits
            # it, and the whole reason the three loops this replaces could
            # never have worked. A daemon thread does not itself delay exit,
            # and keeps running through `threading._shutdown()`, so it can
            # still fire while a non-daemon join is the thing hanging.
            thread = threading.Thread(
                target=self._watch,
                args=(deadline, reason, exit_code, stand_down),
                name="tldw-exit-watchdog",
                daemon=True,
            )
            self._watchdog = thread
            # Both of these stay INSIDE the lock, so one arm is atomic from
            # any other thread's point of view: publishing the new watchdog,
            # standing the old one down and starting the new one cannot be
            # observed half-done. `thread.start()` used to sit outside the
            # lock, which is what made the old `is_alive()` guard above
            # breakable. Neither call can deadlock under the lock --
            # `Event.set` takes only its own lock, and `_watch` does not touch
            # `self._lock` until after its own deadline has elapsed.
            if superseded is not None:
                superseded.set()
            thread.start()
        return True

    def stand_down(self) -> None:
        """Cancel any armed watchdog (used by tests and by re-entrant arms)."""
        with self._lock:
            event = self._stand_down
            self._watchdog = None
            self._watchdog_deadline = None
        if event is not None:
            event.set()

    def _watch(
        self,
        deadline: float,
        reason: str,
        exit_code: int,
        stand_down: threading.Event,
    ) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0 and stand_down.wait(remaining):
            return
        if stand_down.is_set():
            return
        with self._lock:
            current = self._watchdog_deadline
            still_current = self._watchdog is threading.current_thread()
        if current is None or not still_current:
            # Stood down (`stand_down()` nulls the deadline) or replaced by a
            # later arm, in either case between our timeout and this check.
            # The deadline comparison alone missed both: it only caught a
            # replacement whose deadline was LATER than ours, so a disarm
            # landing in that window still ended in a hard exit.
            return
        if current > deadline:
            # A tighter arm() replaced us while we slept; it owns the decision.
            return
        _report_hard_exit(reason)
        _hard_exit(exit_code)


_STATE = _ShutdownState()


def _live_thread_names() -> list[str]:
    """Names of every non-main thread still alive, daemon flag included."""
    main = threading.main_thread()
    out: list[str] = []
    for thread in threading.enumerate():
        if thread is main or not thread.is_alive():
            continue
        out.append(f"{thread.name}{'(daemon)' if thread.daemon else ''}")
    return sorted(out)


def _report_hard_exit(reason: str) -> None:
    """Say what was still running before the process is torn down.

    Deliberately paranoid: this runs from a daemon thread that may be racing
    interpreter finalization, where loguru's sinks -- and even ``sys.stderr``
    -- can already be gone. Every channel is attempted and every failure is
    swallowed, because the hard exit must happen either way. This is also the
    diagnostic the removed "Active non-daemon threads remaining" warning was
    reaching for, moved to the moment it is actually actionable.
    """
    names: list[str] = []
    try:
        names = _live_thread_names()
    except Exception:  # noqa: BLE001 - finalization can break anything
        pass
    message = (
        f"Shutdown did not finish within the grace period ({reason}); "
        f"exiting hard. Threads still alive: {names or 'none'}"
    )
    try:
        logger.error(message)
    except Exception:  # noqa: BLE001
        pass
    try:
        print(f"tldw_chatbook: {message}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


def graceful_shutdown_grace_seconds() -> float:
    """How long teardown may take before the watchdog stops waiting.

    Reads ``[general] shutdown_grace_seconds``. Config is consulted lazily,
    inside a ``try``, because this is reachable from a signal handler: an
    import or a config-parse failure at that moment must degrade to the
    default, not take the process down a second, worse way.
    """
    try:
        from tldw_chatbook.config import get_cli_setting

        raw = get_cli_setting(
            "general", "shutdown_grace_seconds", DEFAULT_SHUTDOWN_GRACE_SECONDS
        )
        value = float(raw)
    except Exception:  # noqa: BLE001 - never fail a shutdown over config
        return DEFAULT_SHUTDOWN_GRACE_SECONDS
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return DEFAULT_SHUTDOWN_GRACE_SECONDS
    return max(_MIN_GRACE_SECONDS, min(_MAX_GRACE_SECONDS, value))


def claim_process_exit() -> None:
    """Declare that this app's exit *is* the process's exit.

    Only the two ``app.py`` entry points may say this, and the watchdog
    refuses to arm until one of them has. The distinction is not pedantry:
    Textual's ``run_test()`` mounts and unmounts a real ``TldwCli`` inside a
    pytest process that then goes on to run thousands more tests. Without
    this gate, every such test would arm a timer that hard-exits the *test
    runner* a few seconds later.
    """
    _STATE.claim_process_exit()


def owns_process_exit() -> bool:
    """Whether an entry point has claimed this process's exit."""
    return _STATE.owns_process_exit


def arm_exit_watchdog(
    seconds: float | None = None,
    *,
    reason: str,
    exit_code: int = _EXIT_CODE_WATCHDOG,
) -> bool:
    """Bound everything that happens from here to process death.

    A no-op unless `claim_process_exit` has been called: an embedded or
    under-test app must never be able to end the process hosting it.

    Args:
        seconds: Grace period. ``None`` reads the configured value.
        reason: Free text for the log line if the watchdog does fire.
        exit_code: Status to exit with on a hard exit.

    Returns:
        True iff this call started the watchdog thread.
    """
    if not _STATE.owns_process_exit:
        return False
    grace = graceful_shutdown_grace_seconds() if seconds is None else float(seconds)
    return _STATE.arm(grace, reason, exit_code)


def disarm_exit_watchdog() -> None:
    """Stand the watchdog down (tests; and any caller that changed its mind)."""
    _STATE.stand_down()


def termination_requested() -> bool:
    """Whether a termination signal has been received this process."""
    return _STATE.signalled


def register_running_app(app: Any) -> None:
    """Tell the signal handler which app (and loop) to ask to exit.

    Call from ``App.on_mount``, i.e. on the event loop -- the running loop is
    captured here so the handler never has to guess at one from a signal
    context.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning(
            "register_running_app called with no running loop; a termination "
            "signal will unwind the main thread instead of asking the app to "
            "exit."
        )
        return
    _STATE.set_app(app, loop)


def unregister_running_app(app: Any) -> None:
    """Drop the app registration (no-op if a different app is registered)."""
    _STATE.clear_app(app)


def _request_app_exit(app: Any) -> None:
    """Run on the event loop: ask Textual to quit through its own front door."""
    try:
        app.exit()
    except Exception:  # noqa: BLE001 - fall through to the watchdog
        logger.opt(exception=True).error(
            "Requesting app exit after a termination signal failed; the exit "
            "watchdog remains armed."
        )


def _handle_termination_signal(signum: int, _frame: Any) -> None:
    """Graceful on the first signal, immediate on the second."""
    exit_code = _EXIT_CODE_SIGINT if signum == signal.SIGINT else _EXIT_CODE_SIGTERM
    count = _STATE.record_signal()
    if count > 1:
        # Explicit operator escalation. No logging: a repeat signal usually
        # means the first one already looked stuck, and a log call here can
        # itself block on a sink lock held by a thread we are about to kill.
        _hard_exit(exit_code)
        return

    # Arm a bound BEFORE reading configuration, in two steps. Reading
    # `[general] shutdown_grace_seconds` means importing and locking the
    # config module from inside a signal handler; that cannot self-deadlock
    # here (the lock is re-entrant and this runs on the main thread), but it
    # can do file I/O, and "the escape hatch does not exist yet" is a bad
    # state to be in while finding out. So: an unconditional backstop first,
    # then the configured value.
    #
    # The backstop is the clamp maximum precisely because no configured value
    # can exceed it, which makes the refinement below always tighter (or
    # within a few microseconds of equal) and therefore always accepted by the
    # monotonic arming rule. Arming the *default* first would have been wrong:
    # a user who configured 300 s would then be refused their own value and
    # silently bounded at 120 s -- exactly the abandoned-write direction this
    # grace period exists to avoid.
    arm_exit_watchdog(
        _MAX_GRACE_SECONDS, reason=f"signal {signum} (backstop)", exit_code=exit_code
    )
    grace = graceful_shutdown_grace_seconds()
    arm_exit_watchdog(grace, reason=f"signal {signum}", exit_code=exit_code)

    app, loop = _STATE.app_and_loop()
    if app is None or loop is None or loop.is_closed():
        # No app yet (startup) or none any more (teardown). SystemExit unwinds
        # the main thread through every `finally`, which is still enormously
        # better than `os._exit`, and the watchdog above bounds it.
        logger.info(f"Received signal {signum} outside the app's lifetime; unwinding.")
        raise SystemExit(exit_code)

    logger.info(
        f"Received signal {signum}; running the ordinary shutdown path "
        f"(hard exit in at most {grace:.0f}s)."
    )
    try:
        loop.call_soon_threadsafe(_request_app_exit, app)
    except RuntimeError:
        # Loop closed between the check above and the call.
        raise SystemExit(exit_code) from None


def install_termination_handlers() -> None:
    """Install the process-level SIGTERM/SIGINT handlers exactly once.

    Safe to call from either entry point (and from both). Signal handlers can
    only be installed from the main thread; anywhere else this logs and
    returns rather than raising, because failing to install a *nicety* must
    never be what stops the application from starting.

    **Nothing is latched until installation actually succeeds** (Qodo review
    of PR #1972). The first version claimed process exit and marked the
    handlers installed *before* calling ``signal.signal``, which produced the
    worst possible pair of outcomes from a failure: every later call became a
    no-op, so the handlers were never installed at all -- and the exit
    watchdog was armed and able to hard-exit anyway, because the claim had
    already gone through. So:

    * a failed attempt claims nothing, arms nothing and latches nothing, and
      is therefore retryable -- the second entry point, or a later call from
      the main thread, can still succeed;
    * a *partial* success (one signal installed, one refused) claims the
      process, because a live handler now exists whose graceful exit needs
      the watchdog's bound -- but still does not latch, so a retry can pick
      up the signal that was refused;
    * only a full success latches, which is what keeps the documented
      idempotence: the second entry point calling this must not reinstall
      over handlers that are already in place.

    An app legitimately embedded in someone else's process -- imported, or
    driven from a non-main thread, where ``signal.signal`` can never work --
    therefore ends up exactly where it should: no handlers, no claim, and a
    watchdog that stays inert because ``arm_exit_watchdog`` refuses to arm
    without a claim. That is the same end state ``get_app()`` already relies
    on.
    """
    if _STATE.handlers_installed:
        return
    wanted = (signal.SIGTERM, signal.SIGINT)
    installed: list[Any] = []
    for sig in wanted:
        try:
            signal.signal(sig, _handle_termination_signal)
        except (ValueError, OSError, RuntimeError) as exc:
            logger.warning(f"Could not install a handler for signal {sig}: {exc}")
        else:
            installed.append(sig)
    if not installed:
        logger.warning(
            "No termination handlers could be installed; this process does not "
            "own its own exit, so the exit watchdog stays inert. A later call "
            "from the main thread will retry."
        )
        return
    # At least one handler is live, so its graceful exit needs to be bounded.
    claim_process_exit()
    if len(installed) == len(wanted):
        _STATE.mark_handlers_installed()
