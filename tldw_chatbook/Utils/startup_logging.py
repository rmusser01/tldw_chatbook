"""Quiet the pre-TUI terminal (TASK-21147, UAT G-7).

Every cold start used to print a wall of import-time log lines — DEBUG
records from config.py (including the literal string "CRITICAL DEBUG:"),
dependency INFO/WARNINGs — to stderr before Textual took over the screen.
A first-time user's literal first contact with the app was internal debug
spew with alarming words in it.

This module must import nothing from tldw_chatbook: it runs BEFORE the
heavy import chain whose noise it silences. The app's own logging setup
(Logging_Config) later removes all pre-existing sinks and installs the
real ones, so the temporary WARNING-level stderr sink added here never
double-logs.
"""

from __future__ import annotations

import logging
import os
import sys

#: Set (to any non-empty value) to keep the historical verbose startup —
#: the live-verification workflow reads import-time logs when diagnosing
#: boot problems.
VERBOSE_STARTUP_ENV_VAR = "TLDW_VERBOSE_STARTUP"


def startup_stderr_is_quiet() -> bool:
    """Whether pre-TUI terminal logging is capped at WARNING (the default).

    Returns:
        True unless TLDW_VERBOSE_STARTUP is set to a non-empty value.
    """
    return not os.environ.get(VERBOSE_STARTUP_ENV_VAR, "").strip()


#: TASK-34100.13 (entry-exit-handoff-17): warnings raised before the TUI
#: opens -- ``RequestsDependencyWarning``, "python-frontmatter not installed"
#: and other optional-dependency notices -- are held here instead of printed,
#: then written to the log by ``replay_startup_notices`` once the app's log
#: file exists. Bounded: a runaway import cannot grow it without limit.
_HELD_NOTICES: list[tuple[str, str]] = []
_MAX_HELD_NOTICES = 200
_held_sink_ids: list[int] = []
_held_stdlib_handler: logging.Handler | None = None
_exit_flush_registered = False


def _hold(name: str, message: str) -> None:
    if len(_HELD_NOTICES) < _MAX_HELD_NOTICES:
        _HELD_NOTICES.append((name or "startup", message))


class _HoldingHandler(logging.Handler):
    """Holds stdlib WARNING records so ``logging.lastResort`` never prints them."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            return  # the stderr sink below reports errors
        try:
            _hold(record.name, record.getMessage())
        except Exception:
            pass


def _show_warning_held(message, category, filename, lineno, file=None, line=None):
    """``warnings.showwarning`` replacement: hold the warning for the log."""
    try:
        _hold(
            "py.warnings",
            f"{getattr(category, '__name__', 'Warning')}: {message} "
            f"({os.path.basename(str(filename))}:{lineno})",
        )
    except Exception:
        pass


def _flush_unreplayed_notices() -> None:
    """At exit, print held notices that never reached the log to stderr.

    Review F10 (TASK-34100.13): only the app's logging setup replays the held
    notices. ``tldw-cli recovery``, a refused startup unlock or a crash before
    the TUI never configure it, and their warnings -- which reached stderr
    before notices were held -- would otherwise be lost.
    """
    held = list(_HELD_NOTICES)
    _HELD_NOTICES.clear()
    for name, message in held:
        try:
            print(f"{name}: {message}", file=sys.stderr)
        except Exception:
            return


def quiet_startup_stderr() -> None:
    """Keep the pre-TUI terminal clear unless verbosity is opted into.

    Errors still reach stderr. Warnings -- loguru's, the standard library's
    and Python's ``warnings`` -- are held for the log file instead
    (``replay_startup_notices``); if the process exits before that replay
    (a recovery subcommand, a refused unlock, an early crash), they are
    printed to stderr at exit instead of dropped.

    Returns:
        None. Idempotent; safe to call before any tldw_chatbook import.
    """
    global _held_stdlib_handler, _exit_flush_registered
    if os.environ.get(VERBOSE_STARTUP_ENV_VAR, "").strip():
        return
    if not _exit_flush_registered:
        import atexit

        atexit.register(_flush_unreplayed_notices)
        _exit_flush_registered = True
    try:
        from loguru import logger

        logger.remove()
        # diagnose=False explicitly: the LOGURU_DIAGNOSE env default binds
        # only if set before loguru's FIRST import, which this module cannot
        # control (task-2119's credential-leak incident, documented in the
        # package __init__). Never rely on the default here.
        logger.add(sys.stderr, level="ERROR", diagnose=False, backtrace=True)
        _held_sink_ids.append(
            logger.add(
                lambda message: _hold(
                    message.record["name"] or "", message.record["message"]
                ),
                level="WARNING",
                filter=lambda record: record["level"].no < 40,
                format="{message}",
                diagnose=False,
            )
        )
    except Exception:
        # loguru missing or already torn down: stdlib capping still applies.
        pass
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    if _held_stdlib_handler is None:
        _held_stdlib_handler = _HoldingHandler(level=logging.WARNING)
        root.addHandler(_held_stdlib_handler)
    import warnings

    warnings.showwarning = _show_warning_held


def replay_startup_notices() -> None:
    """Write the held pre-TUI notices to the log, then stop holding.

    Called once the app's logging is configured (``Logging_Config``). The
    notices are logged at INFO: they reach the log file, never the terminal.
    """
    global _held_stdlib_handler
    held = list(_HELD_NOTICES)
    _HELD_NOTICES.clear()
    if _held_stdlib_handler is not None:
        logging.getLogger().removeHandler(_held_stdlib_handler)
        _held_stdlib_handler = None
    try:
        from loguru import logger

        for sink_id in _held_sink_ids:
            try:
                logger.remove(sink_id)
            except ValueError:
                pass
    except Exception:
        pass
    _held_sink_ids.clear()
    # From here on Python warnings go through logging (the log file and the
    # app's own console), never raw stderr under the TUI.
    logging.captureWarnings(True)
    notices = logging.getLogger("tldw_chatbook.startup")
    for name, message in held:
        notices.info("Before the app opened (%s): %s", name, message)


# ---------------------------------------------------------------------------
# TASK-34100.13 (E7): reduced motion from the environment, before any config.
# It lives here, beside the other pre-TUI environment switches, because this
# module is dependency-free and already resident at boot: config.load_settings
# folds it into [appearance] without adding a module to the boot census
# (ADR-097's ui-ready ratchet).
# ---------------------------------------------------------------------------

#: Environment variables that ask for reduced motion before any config is read.
REDUCE_MOTION_ENV_VARS = ("TLDW_REDUCE_MOTION", "NO_MOTION")

_FALSE_WORDS = frozenset({"", "0", "false", "no", "off"})


def motion_reduced_by_environment(environ=None) -> bool:
    """True when ``TLDW_REDUCE_MOTION`` or ``NO_MOTION`` is set (not to a false word).

    Args:
        environ: The environment mapping; ``os.environ`` when None.

    Returns:
        Whether the environment asks for reduced motion.
    """
    env = os.environ if environ is None else environ
    return any(
        str(env.get(name, "")).strip().lower() not in _FALSE_WORDS
        for name in REDUCE_MOTION_ENV_VARS
    )


def with_motion_environment(appearance, environ=None) -> dict:
    """An ``[appearance]`` table with ``reduce_motion`` forced on by the environment.

    The override lives in the loaded settings only; nothing is written.

    Args:
        appearance: The ``[appearance]`` table (any mapping).
        environ: The environment mapping; ``os.environ`` when None.

    Returns:
        A new dict.
    """
    merged = dict(appearance) if hasattr(appearance, "items") else {}
    if motion_reduced_by_environment(environ):
        merged["reduce_motion"] = True
    return merged
