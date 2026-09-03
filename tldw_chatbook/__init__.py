"""
tldw_chatbook - A Textual TUI for chatting with LLMs

A sophisticated Terminal User Interface (TUI) application built with the Textual
framework for interacting with various Large Language Model APIs. Provides a
complete ecosystem for AI-powered interactions including conversation management,
character/persona chat, notes with bidirectional file sync, media ingestion,
and advanced RAG (Retrieval-Augmented Generation) capabilities.
"""

# Disable progress bars early to prevent interference with TUI
# This must be done before any libraries that use progress bars are imported
import os
import sys

from .Utils.tiktoken_runtime import install_tiktoken_runtime as _install_tiktoken_runtime

_install_tiktoken_runtime()

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# task-2119 (security): Loguru's `diagnose` option defaults to True and, on
# any record logged via `logger.opt(exception=True)`, dumps every stack
# frame's LOCAL VARIABLES alongside the traceback -- not just the source
# line. Provider request handlers in LLM_Calls/LLM_API_Calls.py hold the raw
# `Authorization`/`x-api-key` header dict and the resolved API key in scope
# across ~30 exception handlers; with diagnose left at its default, an
# ordinary transient error (timeout, 429, connection reset) prints the live
# credential in cleartext to whatever sink is active. Live-confirmed via a
# genuine Moonshot 429 on 2026-08-03: a verification script that imported
# this package's adapters directly (never running the real app's own
# `Logging_Config.configure_application_logging`) hit loguru's auto-init
# default sink, which was still diagnose=True.
#
# The real app's own sink, and `Metrics/logger_config.py`'s legacy sink,
# both now also pass `diagnose=False` explicitly where they call
# `logger.add(...)` -- that per-sink kwarg is the actual, order-independent
# fix. This block additionally neutralizes loguru's own auto-init default
# sink (created the instant anything, anywhere, does
# `from loguru import logger`, before any application code has run) by
# replacing it outright, and sets the library-wide default for any *future*
# `logger.add(...)` call that forgets to pass `diagnose=` explicitly. Doing
# both (not just the env var) matters: `LOGURU_DIAGNOSE` only takes effect
# for a given `add()` call if it was set before loguru's `_logger.py` module
# was first imported anywhere in the process -- it becomes a bound default
# on `Logger.add`'s signature at that point, not a value read fresh on every
# call -- and `Tests/conftest.py` imports loguru ahead of this package for
# fixture-ordering reasons, which would otherwise defeat the env var alone
# for the whole test session. The explicit `remove(0)` + `add(diagnose=False)`
# below is not subject to that ordering: it runs the moment this package is
# imported, which is unconditionally required before any of its own
# exception handlers can execute.
#
# Only loguru's auto-init sink (always handler id 0) is replaced. A host
# application that configured loguru before importing this package keeps its
# own sinks untouched -- if it removed the default sink, `remove(0)` raises
# ValueError and this block installs nothing, because that host owns the
# logging configuration (and any diagnose=True sink it installed on purpose).
os.environ.setdefault("LOGURU_DIAGNOSE", "0")
try:
    from loguru import logger as _pkg_init_loguru_logger

    try:
        _pkg_init_loguru_logger.remove(0)
    except ValueError:
        pass
    else:
        _pkg_init_loguru_logger.add(sys.stderr, diagnose=False, backtrace=True)
except Exception:
    # Logging safety must never be the reason package import fails.
    pass


def _install_textual_compatibility_shims() -> None:
    """Keep older Textual widget access patterns working across dependency bumps."""
    try:
        from textual.widgets import Static
    except Exception:
        return

    if not hasattr(Static, "renderable"):
        Static.renderable = property(  # type: ignore[attr-defined]
            lambda self: self.content,
            lambda self, value: self.update(value),
        )


_install_textual_compatibility_shims()

__version__ = "0.1.8.1"
__author__ = "Robert Musser"
__email__ = "contact@rmusser.net"
__license__ = "AGPLv3+"

# Version tuple for programmatic comparison
VERSION_TUPLE = (0, 1, 8, 1)

# Export key components when package is imported
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "VERSION_TUPLE",
]
