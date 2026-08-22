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
   the first action.
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
#: Generous on purpose: a measured normal exit is well under a second, so this
#: is a stuck-process bound, not a deadline anything healthy races.
DEFAULT_SHUTDOWN_GRACE_SECONDS = 20.0

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

    def mark_handlers_installed(self) -> bool:
        """Return True the first time only, so installation is idempotent."""
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
        deadline = time.monotonic() + seconds
        with self._lock:
            if (
                self._watchdog is not None
                and self._watchdog.is_alive()
                and self._watchdog_deadline is not None
                and deadline >= self._watchdog_deadline
            ):
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
        if current is not None and current > deadline:
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

    arm_exit_watchdog(reason=f"signal {signum}", exit_code=exit_code)

    app, loop = _STATE.app_and_loop()
    if app is None or loop is None or loop.is_closed():
        # No app yet (startup) or none any more (teardown). SystemExit unwinds
        # the main thread through every `finally`, which is still enormously
        # better than `os._exit`, and the watchdog above bounds it.
        logger.info(f"Received signal {signum} outside the app's lifetime; unwinding.")
        raise SystemExit(exit_code)

    logger.info(
        f"Received signal {signum}; running the ordinary shutdown path "
        f"(hard exit in at most {graceful_shutdown_grace_seconds():.0f}s)."
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
    """
    claim_process_exit()
    if not _STATE.mark_handlers_installed():
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_termination_signal)
        except (ValueError, OSError, RuntimeError) as exc:
            logger.warning(f"Could not install a handler for signal {sig}: {exc}")
