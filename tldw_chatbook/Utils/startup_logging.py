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
    """Whether pre-TUI terminal logging is capped at WARNING (the default)."""
    return not os.environ.get(VERBOSE_STARTUP_ENV_VAR, "").strip()


def quiet_startup_stderr() -> None:
    """Cap pre-TUI terminal logging at WARNING unless verbosity is opted into.

    Returns:
        None. Idempotent; safe to call before any tldw_chatbook import.
    """
    if os.environ.get(VERBOSE_STARTUP_ENV_VAR, "").strip():
        return
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        # loguru missing or already torn down: stdlib capping still applies.
        pass
    logging.getLogger().setLevel(logging.WARNING)
